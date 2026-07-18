"""Relabel apply: rewrite a matched corpus's 20 Hz move labels (fb/lr) from
the fitted labeler — the step that turns forced-MVD features into an
MVD-synth move corpus, and the end-to-end parity measurement for it.

Pipeline (per split):
  1. slim episodes -> features (lag config from the model's meta.json)
     -> GBT probs -> transition-penalized Viterbi at the fitted per-axis
     switch penalty (decode_fit in meta.json) -> native-rate fb/lr classes,
  2. exact ``native_index`` lookup resamples each 20 Hz qobs frame to its
     native prediction (the matched collect records the sampling frame —
     no windowing approximation),
  3. rewrite the fb/lr press-byte bits of a COPY of the qobs ``act_move``;
     ud / jump / attack bits pass through untouched,
  4. score rewritten-vs-truth in the training representation: per-axis
     20 Hz agreement + the segment-parity gate (onset ratio, duration-TV)
     directly on the 20 Hz streams (stride 1 — qobs frames ARE model frames),
  5. ``--write`` emits per-shard ``*_act_move_mvdsynth.npy`` sidecars next
     to the originals plus ``relabel_meta.json`` with provenance.  Without
     it the run is measurement-only.

Usage:
    PYTHONPATH=src python -m qnn.labeler.apply \\
        --matched-dir artifacts/collect/qwd_matched \\
        --model artifacts/labeler/move_gbt_mvd_matched320 \\
        --split precomputed_val [--write]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qnn.actions import decode_move_pressbyte
from .data import FeatureSpec, _load_split, build_features
from .decode import viterbi_smooth
from .gbt import add_velocity_lags
from .seg_stats import segment_parity

# Press-byte bit layout (see qnn.actions.decode_move_pressbyte / the collect
# skill): bit1/2 = fb neg/pos, bit3/4 = lr neg/pos.
_AXIS_BITS = {"fb": (1, 2), "lr": (3, 4)}


def _load_native_index(split_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    """Per-episode native_index arrays + demo_idxs, in manifest order."""
    man = json.loads((split_dir / "manifest.json").read_text())
    out, demos = [], []
    for shard in man["shards"]:
        ni = np.load(split_dir / shard["obs"]["native_index"])
        start = 0
        for length, demo_idx in zip(shard["episode_lengths"], shard["demo_idxs"]):
            out.append(ni[start:start + int(length)].astype(np.int64))
            demos.append(int(demo_idx))
            start += int(length)
    return out, demos


# Episode-chunk size for feature/predict/decode passes.  The devcontainer
# is cgroup-capped at 16 GB; a full-split feature matrix (28M x 63 on the
# matched train split) plus the Viterbi padding busts it.  Lags and the
# Viterbi are both episode-local, so chunking whole episodes is exact.
_CHUNK_FRAMES = 2_000_000


def _episode_chunks(episode_starts: np.ndarray, max_frames: int
                    ) -> list[tuple[int, int]]:
    """Split episodes into contiguous [e_lo, e_hi) groups of whole episodes
    totaling <= max_frames each (a single longer episode gets its own group)."""
    lengths = np.diff(episode_starts)
    chunks, lo, acc = [], 0, 0
    for e, L in enumerate(lengths):
        if acc and acc + int(L) > max_frames:
            chunks.append((lo, e))
            lo, acc = e, 0
        acc += int(L)
    if lo < lengths.shape[0]:
        chunks.append((lo, lengths.shape[0]))
    return chunks


def _predict_log_probs(
    model_dir: Path, slim_split: Path
) -> tuple[dict[str, np.ndarray], np.ndarray, list[np.ndarray], list[int], dict]:
    """Run the GBT over a slim split ONCE (the expensive part), chunked by
    whole episodes so the feature matrix never materializes in full.

    Returns ``(log_probs {axis: (N,3) f32}, episode_starts, ni_eps,
    demo_idxs, meta)`` — decode (cheap) happens per λ downstream.
    """
    import lightgbm as lgb

    meta = json.loads((model_dir / "meta.json").read_text())
    spec = FeatureSpec()          # core 9-dim, matching the GBT trainer
    episodes = _load_split(slim_split)
    ni_eps, demo_idxs = _load_native_index(slim_split)
    assert len(episodes) == len(ni_eps)

    episode_starts = np.zeros(len(episodes) + 1, dtype=np.int64)
    episode_starts[1:] = np.cumsum([ep.n_frames for ep in episodes])
    n = int(episode_starts[-1])
    lag_cols = (0, 1, 2, 6, 7, 8) if meta.get("lag_look") else (0, 1, 2)
    use_lags = meta.get("temporal_lags", True)

    boosters = {axis: lgb.Booster(model_file=str(model_dir / f"booster_{axis}.txt"))
                for axis in ("fb", "lr")}
    log_probs = {axis: np.empty((n, 3), dtype=np.float32) for axis in boosters}
    for e_lo, e_hi in _episode_chunks(episode_starts, _CHUNK_FRAMES):
        X = np.concatenate([
            build_features(ep.self_velocity, ep.self_movement_id, ep.look,
                           spec=spec)
            for ep in episodes[e_lo:e_hi]], axis=0)
        if use_lags:
            chunk_starts = episode_starts[e_lo:e_hi + 1] - episode_starts[e_lo]
            X = add_velocity_lags(X, chunk_starts,
                                  offsets=tuple(meta["vel_lag_offsets"]),
                                  lag_cols=lag_cols)
        s, t = episode_starts[e_lo], episode_starts[e_hi]
        for axis, booster in boosters.items():
            log_probs[axis][s:t] = np.log(
                np.clip(booster.predict(X), 1e-12, None)).astype(np.float32)
    for axis in boosters:
        print(f"  [{axis}] predicted {n:,} native frames (chunked)", flush=True)
    return log_probs, episode_starts, ni_eps, demo_idxs, meta


def _viterbi_chunked(log_probs: np.ndarray, episode_starts: np.ndarray,
                     lam: float) -> np.ndarray:
    """viterbi_smooth over episode chunks — bit-identical (episodes are
    independent) with bounded padding memory."""
    out = np.empty(log_probs.shape[0], dtype=np.int64)
    for e_lo, e_hi in _episode_chunks(episode_starts, _CHUNK_FRAMES):
        s, t = episode_starts[e_lo], episode_starts[e_hi]
        chunk_starts = episode_starts[e_lo:e_hi + 1] - s
        out[s:t] = viterbi_smooth(log_probs[s:t], chunk_starts, lam)
    return out


def _decode_per_demo(
    log_probs: dict[str, np.ndarray],
    episode_starts: np.ndarray,
    ni_eps: list[np.ndarray],
    demo_idxs: list[int],
    lams: dict[str, float],
) -> dict[int, dict[str, np.ndarray]]:
    """Viterbi-decode at per-axis λ and slice back to per-demo streams."""
    decoded = {axis: _viterbi_chunked(log_probs[axis], episode_starts, lams[axis])
               for axis in ("fb", "lr")}
    per_demo: dict[int, dict[str, np.ndarray]] = {}
    for e, demo_idx in enumerate(demo_idxs):
        s, t = episode_starts[e], episode_starts[e + 1]
        per_demo[demo_idx] = {
            "native_index": ni_eps[e],
            "fb": decoded["fb"][s:t], "lr": decoded["lr"][s:t],
        }
    return per_demo


def _rewrite_move(
    move: np.ndarray,              # (T,) uint8 qobs press byte (one episode)
    qobs_ni: np.ndarray,           # (T,) int64 native_index per qobs frame
    pred: dict[str, np.ndarray],   # slim-side native predictions for the demo
) -> tuple[np.ndarray, int]:
    """Rewrite fb/lr bits from the native predictions (exact ni lookup).
    Returns (rewritten copy, n_frames with no exact native match — those
    keep the original bits)."""
    slim_ni = pred["native_index"]
    pos = np.searchsorted(slim_ni, qobs_ni)
    pos_c = np.clip(pos, 0, slim_ni.shape[0] - 1)
    hit = slim_ni[pos_c] == qobs_ni
    out = move.copy()
    for axis, (neg_bit, pos_bit) in _AXIS_BITS.items():
        cls = pred[axis][pos_c]                       # (T,) {0,1,2}
        clear = np.uint8(~((1 << neg_bit) | (1 << pos_bit)) & 0xFF)
        new = ((out & clear)
               | ((cls == 0).astype(np.uint8) << neg_bit)
               | ((cls == 2).astype(np.uint8) << pos_bit))
        out = np.where(hit, new, out).astype(np.uint8)
    return out, int((~hit).sum())


def _gather_scoring_arrays(
    qobs_split: Path,
    demo_to_ep: dict[int, int],
    episode_starts_slim: np.ndarray,
    ni_eps: list[np.ndarray],
) -> dict:
    """Flatten the qobs split into λ-independent scoring arrays.

    Returns truth class streams per axis, global slim-stream positions
    (``gpos``) + exact-hit mask for point-sampling any decoded native
    stream, and 20 Hz episode starts.
    """
    man = json.loads((qobs_split / "manifest.json").read_text())
    truth = {"fb": [], "lr": []}
    gpos_parts, hit_parts, ep_lengths = [], [], []
    for shard in man["shards"]:
        move = np.load(qobs_split / shard["actions"]["move"])
        ni = np.load(qobs_split / shard["obs"]["native_index"]).astype(np.int64)
        start = 0
        for length, demo_idx in zip(shard["episode_lengths"], shard["demo_idxs"]):
            stop = start + int(length)
            e = demo_to_ep.get(int(demo_idx))
            if e is not None:
                slim_ni = ni_eps[e]
                pos = np.searchsorted(slim_ni, ni[start:stop])
                pos_c = np.clip(pos, 0, slim_ni.shape[0] - 1)
                hit_parts.append(slim_ni[pos_c] == ni[start:stop])
                gpos_parts.append(episode_starts_slim[e] + pos_c)
                cls = decode_move_pressbyte(move[start:stop])
                truth["fb"].append(cls[:, 0].astype(np.int64))
                truth["lr"].append(cls[:, 1].astype(np.int64))
                ep_lengths.append(int(length))
            start = stop
    ep_starts = np.zeros(len(ep_lengths) + 1, dtype=np.int64)
    ep_starts[1:] = np.cumsum(ep_lengths)
    return {
        "truth": {a: np.concatenate(truth[a]) for a in truth},
        "gpos": np.concatenate(gpos_parts),
        "hit": np.concatenate(hit_parts),
        "ep_starts": ep_starts,
    }


def fit_apply_space(
    log_probs: dict[str, np.ndarray],
    episode_starts_slim: np.ndarray,
    scoring: dict,
    lam_grid: tuple[float, ...],
) -> dict:
    """Sweep λ per axis scored on the POINT-SAMPLED 20 Hz comparison — the
    training representation the relabel actually lands in.  The windowed
    gate fit is a proxy; this is the definitive operating point (the two
    differ: e.g. lr λ=1.0 by the windowed fit measured onset x0.92 end to
    end).  Same selection score as the gate fit: |log onset_x| + dur_tv.
    """
    out = {}
    for axis in ("fb", "lr"):
        truth = scoring["truth"][axis]
        frontier, best = [], None
        for lam in lam_grid:
            dec = _viterbi_chunked(log_probs[axis], episode_starts_slim, lam)
            synth = np.where(scoring["hit"], dec[scoring["gpos"]], truth)
            par = segment_parity(synth, truth, scoring["ep_starts"])
            ratio = par["onset_ratio"]
            score = ((abs(float(np.log(ratio))) if ratio else float("inf"))
                     + par["dur_tv"])
            row = {"lam": lam, "onset_ratio": ratio, "dur_tv": par["dur_tv"],
                   "agree": round(float((synth == truth).mean()), 6),
                   "score": round(score, 4)}
            frontier.append(row)
            if best is None or score < best[0]:
                best = (score, lam)
        out[axis] = {"lam": best[1], "frontier": frontier}
        chosen = next(r for r in frontier if r["lam"] == best[1])
        print(f"  [{axis}] apply-space fit: lam={best[1]}  "
              f"onset x{chosen['onset_ratio']}  durTV {chosen['dur_tv']:.3f}  "
              f"agree {chosen['agree'] * 100:.2f}%", flush=True)
    return out


def run_apply(matched_dir: Path, model_dir: Path, split: str,
              write: bool, fit: bool = False) -> dict:
    slim_split = matched_dir / "slim" / split
    qobs_split = matched_dir / "qobs" / split
    t0 = time.time()
    log_probs, ep_starts_slim, ni_eps, demo_idxs, model_meta = \
        _predict_log_probs(model_dir, slim_split)
    demo_to_ep = {d: e for e, d in enumerate(demo_idxs)}

    if fit:
        scoring = _gather_scoring_arrays(qobs_split, demo_to_ep,
                                         ep_starts_slim, ni_eps)
        from .decode import DEFAULT_LAM_GRID
        fits = fit_apply_space(log_probs, ep_starts_slim, scoring,
                               DEFAULT_LAM_GRID)
        model_meta["decode_fit_apply"] = {
            **fits, "fit_split": split, "matched_dir": str(matched_dir)}
        (model_dir / "meta.json").write_text(json.dumps(model_meta, indent=2))
        print(f"  wrote decode_fit_apply -> {model_dir / 'meta.json'}")

    fit_key = ("decode_fit_apply" if "decode_fit_apply" in model_meta
               else "decode_fit")
    lams = {a: float(model_meta[fit_key][a]["lam"]) for a in ("fb", "lr")}
    print(f"  decoding at {fit_key} lams: {lams}", flush=True)
    per_demo = _decode_per_demo(log_probs, ep_starts_slim, ni_eps, demo_idxs,
                                lams)

    man = json.loads((qobs_split / "manifest.json").read_text())
    truth_streams = {"fb": [], "lr": []}
    synth_streams = {"fb": [], "lr": []}
    ep_lengths = []
    total_miss = total_frames = 0
    sidecars = []

    for shard in man["shards"]:
        move = np.load(qobs_split / shard["actions"]["move"])
        ni = np.load(qobs_split / shard["obs"]["native_index"]).astype(np.int64)
        new_move = move.copy()
        start = 0
        for length, demo_idx in zip(shard["episode_lengths"], shard["demo_idxs"]):
            stop = start + int(length)
            pred = per_demo.get(int(demo_idx))
            if pred is not None:
                rew, miss = _rewrite_move(move[start:stop], ni[start:stop], pred)
                new_move[start:stop] = rew
                total_miss += miss
                # decode both press bytes for the parity comparison
                tr = decode_move_pressbyte(move[start:stop])
                sy = decode_move_pressbyte(rew)
                for ai, axis in enumerate(("fb", "lr")):
                    truth_streams[axis].append(tr[:, ai].astype(np.int64))
                    synth_streams[axis].append(sy[:, ai].astype(np.int64))
                ep_lengths.append(int(length))
                total_frames += int(length)
            start = stop
        if write:
            base = shard["actions"]["move"].replace("_act_move.npy",
                                                    "_act_move_mvdsynth.npy")
            np.save(qobs_split / base, new_move)
            sidecars.append(base)

    ep_starts = np.zeros(len(ep_lengths) + 1, dtype=np.int64)
    ep_starts[1:] = np.cumsum(ep_lengths)
    report = {"split": split, "model": str(model_dir),
              "frames": total_frames, "ni_miss": total_miss}
    for axis in ("fb", "lr"):
        truth = np.concatenate(truth_streams[axis])
        synth = np.concatenate(synth_streams[axis])
        par = segment_parity(synth, truth, ep_starts)   # stride 1: model frames
        par["agree"] = round(float((synth == truth).mean()), 6)
        report[axis] = par
        print(f"  [{axis}] END-TO-END 20Hz: agree {par['agree']*100:.2f}%  "
              f"onset x{par['onset_ratio']}  durTV {par['dur_tv']:.3f}", flush=True)
    print(f"  ni exact-miss frames: {total_miss:,} / {total_frames:,}")

    if write:
        report["sidecars"] = sidecars
        report["model_meta"] = {k: model_meta.get(k) for k in
                                ("tag", "decode_fit", "decode_fit_apply",
                                 "lag_look", "frame_acc")}
        out = matched_dir / "qobs" / f"relabel_meta_{split}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"  wrote {len(sidecars)} sidecars + {out}")
    print(f"  apply total {time.time() - t0:.1f}s")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-dir", type=Path, required=True,
                    help="Matched-collect root (contains slim/ + qobs/).")
    ap.add_argument("--model", type=Path, required=True,
                    help="GBT model dir (boosters + meta.json with decode_fit).")
    ap.add_argument("--split", default="precomputed_val",
                    choices=["precomputed_val", "precomputed_train"])
    ap.add_argument("--write", action="store_true",
                    help="Write *_act_move_mvdsynth.npy sidecars + relabel "
                         "meta. Default is measurement-only.")
    ap.add_argument("--fit", action="store_true",
                    help="Refit the per-axis switch penalty in APPLY space "
                         "(point-sampled 20 Hz comparison) before decoding; "
                         "stores decode_fit_apply into the model meta.json.")
    args = ap.parse_args()
    run_apply(args.matched_dir, args.model, args.split, args.write,
              fit=args.fit)


if __name__ == "__main__":
    main()
