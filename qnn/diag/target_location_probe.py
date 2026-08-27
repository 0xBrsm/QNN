"""Linear recoverability of target location from readout / target.feat / concat.

Hypothesis under test (Brian, 7/26): the heads consume
``cat(readout, target.feat)`` with NO explicit target position — only a
16-dim softmax-pooled feature from the target pointer. Is target location
(relative yaw, relative pitch, log-distance) linearly recoverable from:

  (a) the GRU readout (``policy.model.temporal`` forward output, ``.flat_out``)
  (b) target.feat (``policy.model.target_pointer`` forward output, ``.target_feat``)
  (c) their concat — EXACTLY what the canonical heads consume as
      ``features_base_flat`` in ``qnn.model.network.Network.forward``
      (``cat([readout_flat, target_feat], dim=-1)``; canonical heads then
      prefix-slice their own declared width off the front of this vector,
      per ``LookSegHead.forward``'s ``features[..., :self.in_dim]``)

...and does this differ between the look_seg model
(``runs/bc/bench/lookseg_w122s1b_seed43``) and the polar-look model
(``runs/head_probe/head_probe_atlas24x11_awposw_seed43``)?

Hook points (verified against ``qnn/model/network.py`` and ``qnn/model/{temporal,target}.py``):
  - ``policy.model.temporal`` is an ``nn.Module`` (``Temporal``); its forward
    returns a ``TemporalOutput`` dataclass with ``.flat_out`` — this becomes
    ``readout_flat`` in ``Network.forward`` with NO further transform before
    heads consume it. GRU d_gru=64 for both runs' shared ``full_5head`` base.
  - ``policy.model.target_pointer`` is an ``nn.Module`` (``TargetPointer``);
    its forward returns a ``TargetPointerOutput`` dataclass with
    ``.target_feat`` (16-dim, ``pointers.target.d_target`` in the base graph
    JSON) — used as-is, no further transform.
  These are unambiguous module boundaries (registered submodules, called
  exactly once per ``Network.forward``), so we hook them directly rather than
  slicing a head's concatenated input by (possibly-wrong) hardcoded widths.

Ground truth per frame is derived from the target slot's
``entity_scalars_raw[..., 3:6]`` (agent view-frame relative XYZ, ALREADY
dequantized by DIST_SCALE=1000 — see ``qnn.model.dequant.EntityDequantizer``):

  - ``QNN_RelativeFrame`` in ``src/engine/common/qnn.h:781-789`` computes
    ``out = (dot(delta,forward), dot(delta,right), dot(delta,up))`` from the
    view angles via ``AngleVectors`` (a full pitch+yaw rotation, not yaw-only)
    — so x=forward (boresight), y=right, z=up are already crosshair-relative.
  - ``qnn.engine_norm._F_REL`` (scale=DIST_SCALE=1000) confirms "Position
    relative to player, view-frame rotated."
  - Target slot selection matches ``qnn.diag.attack._compute_bearing``
    exactly: slot 0 of ``target_probs`` = NO_TARGET mass; entity token j
    <-> ``target_probs[:, j+1]``; target token = ``argmax(target_probs[:, 1:])``.
    (``_compute_bearing`` computes ``arccos(x/|rel|)`` — the TOTAL angle
    between boresight and target; our yaw/pitch below are exactly the
    orthogonal decomposition of that same angle: ``yaw = atan2(y, x)``,
    ``pitch = atan2(z, sqrt(x^2+y^2))``.)
  - log-distance uses ``|rel|`` in the same (DIST_SCALE-normalized) units;
    a monotonic transform, so its R^2 is invariant to the DIST_SCALE offset.

Data: qwd_v4 ``precomputed_val`` (courtesy of ``qnn.bc.supervised_loop.
make_resident_source_from_cache``, the SAME loader ``qnn.diag.attack`` uses
for entity_stream=full corpora — ``qnn.diag.data.load_val_episodes``, the
loader ``linear_probe.py`` uses, is NOT safe here: it slices ragged
entity_* shard arrays as if they were one-row-per-frame, which is wrong for
this entity_stream=full cache (verified: shard 0 of qwd_v4/precomputed_val
has 264,432 frames but 1,760,882 flat entity-token rows, keyed by
``entity_count``). ``make_resident_source_from_cache`` pads ragged tokens to
MAX_TOKEN_OBJECTS via indptr and applies the dequant chain, exactly what the
model needs). Per Brian's course-correction (7/28): this loads the WHOLE
precomputed_val split in ~4s on this machine — no shard truncation needed;
just load it all with ``segment_mask=ENGAGED_MASK`` (restricts to contiguous
runs where a target exists, matching the frame filter below) and run the
frozen forward passes.

Probe: ridge regression (scikit-learn), alpha chosen via 3-fold CV over a
small grid, train/val split by ORIGINAL episode (never frame) to avoid
temporal leakage. Metrics: R^2 for yaw (via sin/cos target pair), pitch,
log-distance; median absolute angular error in degrees (reconstructed from
the sin/cos prediction); a shuffled-label control (fit on permuted targets)
to confirm near-zero R^2 as a sanity floor.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


#: Segment mask restricting to contiguous frames where a live target exists
#: (target_probs[:, 0] != 1, i.e. mass on some real entity slot). Same
#: predicate qnn.diag.analyze's segment="engaged" uses.
ENGAGED_MASK: dict = {"act.target": {"$ne": 0}}

#: entity_scalars_raw actor-slice for view-frame relative XYZ (x=fwd,y=right,z=up).
_REL_BEGIN, _REL_END = 3, 6

#: Ridge alpha grid for the 3-fold CV sweep.
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

#: Representations probed, as (name, needs_readout, needs_target_feat).
REPRESENTATIONS = ("readout", "target_feat", "concat")


# ---------------------------------------------------------------------------
# Model hook plumbing
# ---------------------------------------------------------------------------

def _register_hooks(policy) -> tuple[dict[str, list], list]:
    """Hook ``policy.model.temporal`` (-> readout) and ``.target_pointer``
    (-> target.feat) separately. Fails loud if either boundary is absent —
    per the task's "no silent fallbacks" — since a silently-empty capture
    would look like a shape-0 probe rather than a real error.
    """
    model = policy.model
    temporal = getattr(model, "temporal", None)
    pointer = getattr(model, "target_pointer", None)
    if not getattr(model, "_has_temporal", False) or temporal is None:
        raise RuntimeError(
            "policy.model has no active temporal (GRU) module — this probe "
            "requires a GRU-readout graph (full_5head base); got "
            f"_has_temporal={getattr(model, '_has_temporal', None)!r}"
        )
    if not getattr(model, "_has_target_pointer", False) or pointer is None:
        raise RuntimeError(
            "policy.model has no active target_pointer module — this probe "
            "requires the pointer slot on (target.feat must exist); got "
            f"_has_target_pointer={getattr(model, '_has_target_pointer', None)!r}"
        )

    captures: dict[str, list] = {"readout": [], "target_feat": []}

    def hook_temporal(_m, _inp, out):
        captures["readout"].append(out.flat_out.detach().cpu().numpy())

    def hook_pointer(_m, _inp, out):
        captures["target_feat"].append(out.target_feat.detach().cpu().numpy())

    handles = [
        temporal.register_forward_hook(hook_temporal),
        pointer.register_forward_hook(hook_pointer),
    ]
    return captures, handles


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _compute_ground_truth(
    entity_scalars_raw: torch.Tensor,   # (T, N, 18) dequantized
    target_probs: torch.Tensor,         # (T, P) float
) -> dict[str, np.ndarray]:
    """Per-frame (yaw_rad, pitch_rad, logdist, has_target) at the argmax target slot.

    Mirrors qnn.diag.attack._compute_bearing's slot-selection exactly
    (argmax over target_probs[:, 1:], clamped to N-1 entity tokens), but
    decomposes the crosshair-relative vector into yaw/pitch instead of a
    single total bearing angle.
    """
    N = entity_scalars_raw.shape[1]
    best = torch.argmax(target_probs[:, 1:], dim=1).clamp(max=N - 1)   # (T,)
    fidx = torch.arange(entity_scalars_raw.shape[0], device=entity_scalars_raw.device)
    rel = entity_scalars_raw[fidx, best, _REL_BEGIN:_REL_END]           # (T, 3) fwd,right,up
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    horiz = torch.sqrt(x * x + y * y).clamp_min(1e-9)
    yaw = torch.atan2(y, x)
    pitch = torch.atan2(z, horiz)
    dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    logdist = torch.log(dist)
    has_target = target_probs[:, 0] < 1.0
    return {
        "yaw": yaw.detach().cpu().numpy().astype(np.float64),
        "pitch": pitch.detach().cpu().numpy().astype(np.float64),
        "logdist": logdist.detach().cpu().numpy().astype(np.float64),
        "has_target": has_target.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Frame extraction: frozen forward pass + aligned ground truth, per model
# ---------------------------------------------------------------------------

def extract_frames(
    policy,
    source,
    *,
    progress_every: int = 200,
) -> dict[str, np.ndarray]:
    """Run one frozen forward pass per resident-source segment (hidden reset
    at each segment start — segments are already contiguous engaged runs cut
    by ``ENGAGED_MASK`` at gather time), capturing readout/target_feat via
    hooks and the aligned ground truth from the same slice.

    Returns per-frame arrays: readout (F,64), target_feat (F,16), episode_id
    (F,) int (source segment index — used for the probe's episode-grouped
    split), yaw/pitch/logdist/has_target.
    """
    captures, handles = _register_hooks(policy)
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    n_segments = len(offs) - 1
    yaws: list[np.ndarray] = []
    pitches: list[np.ndarray] = []
    logdists: list[np.ndarray] = []
    has_targets: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    t0 = time.monotonic()
    kept_segments = 0
    with torch.inference_mode():
        for si in range(n_segments):
            lo, hi = int(offs[si]), int(offs[si + 1])
            if hi <= lo:
                continue
            obs_seq = {k: v[lo:hi].unsqueeze(1) for k, v in source.obs.items()}
            policy._forward_tensors(obs_seq, hidden=None)
            gt = _compute_ground_truth(
                source.obs["entity_scalars_raw"][lo:hi],
                source.actions["target_probs"][lo:hi],
            )
            n = hi - lo
            yaws.append(gt["yaw"]); pitches.append(gt["pitch"])
            logdists.append(gt["logdist"]); has_targets.append(gt["has_target"])
            episode_ids.append(np.full(n, si, dtype=np.int64))
            kept_segments += 1
            if kept_segments % progress_every == 0:
                done_frames = int(offs[si + 1])
                print(
                    f"[target-loc-probe]   segment {si+1}/{n_segments} "
                    f"({done_frames:,}/{int(offs[-1]):,} frames, "
                    f"{time.monotonic()-t0:.1f}s)", flush=True,
                )
    for h in handles:
        h.remove()

    if not captures["readout"]:
        raise RuntimeError("no segments produced any frames — check segment_mask / corpus")

    readout = np.concatenate(captures["readout"], axis=0)
    target_feat = np.concatenate(captures["target_feat"], axis=0)
    yaw = np.concatenate(yaws)
    pitch = np.concatenate(pitches)
    logdist = np.concatenate(logdists)
    has_target = np.concatenate(has_targets)
    episode_id = np.concatenate(episode_ids)

    if not (readout.shape[0] == target_feat.shape[0] == yaw.shape[0] == episode_id.shape[0]):
        raise RuntimeError(
            f"frame-count mismatch after concat: readout={readout.shape[0]} "
            f"target_feat={target_feat.shape[0]} yaw={yaw.shape[0]} "
            f"episode_id={episode_id.shape[0]}"
        )

    return {
        "readout": readout,
        "target_feat": target_feat,
        "yaw": yaw,
        "pitch": pitch,
        "logdist": logdist,
        "has_target": has_target,
        "episode_id": episode_id,
    }


# ---------------------------------------------------------------------------
# Ridge probe
# ---------------------------------------------------------------------------

def _episode_train_val_split(
    episode_id: np.ndarray, *, val_frac: float = 0.2, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, val_mask) splitting by EPISODE id, not frame."""
    uniq = np.unique(episode_id)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)
    n_val = max(1, int(round(len(perm) * val_frac)))
    val_eps = set(perm[:n_val].tolist())
    val_mask = np.isin(episode_id, list(val_eps))
    return ~val_mask, val_mask


def _fit_ridge_cv(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    """3-fold CV over ALPHA_GRID, refit best alpha on the full train set."""
    if not HAVE_SKLEARN:
        raise RuntimeError("sklearn not available — pip install scikit-learn")
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    best_alpha, best_score = ALPHA_GRID[0], -np.inf
    for alpha in ALPHA_GRID:
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            m = Ridge(alpha=alpha)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(m.score(X_train[va_idx], y_train[va_idx]))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_alpha = mean_score, alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    return model, best_alpha


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 via sklearn's r2_score — handles the (F, 2) sin/cos multi-output
    case correctly (per-column variance, uniform_average across outputs).
    A hand-rolled ``1 - ss_res/ss_tot`` using a bare ``.mean()`` on a 2-D
    array collapses to one scalar mean across BOTH columns instead of a
    per-column mean, which silently corrupts the yaw R^2 (caught during
    the shuffled-control sanity check: it should collapse to ~0 and did
    not until this was fixed to use r2_score)."""
    return float(r2_score(y_true, y_pred))


def probe_representation(
    X: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    logdist: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    shuffle_control: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit 3 independent ridge probes (yaw as sin/cos pair -> angle recon,
    pitch, logdist) on one representation X. Returns R^2 + angular error.
    """
    rng = np.random.RandomState(seed)
    y_sincos = np.stack([np.sin(yaw), np.cos(yaw)], axis=1)   # (F, 2)

    Xtr, Xva = X[train_mask], X[val_mask]
    if shuffle_control:
        perm = rng.permutation(Xtr.shape[0])
        y_sincos_tr = y_sincos[train_mask][perm]
        pitch_tr = pitch[train_mask][perm]
        logdist_tr = logdist[train_mask][perm]
    else:
        y_sincos_tr = y_sincos[train_mask]
        pitch_tr = pitch[train_mask]
        logdist_tr = logdist[train_mask]

    out: dict[str, Any] = {}

    # yaw via sin/cos regression
    m_yaw, alpha_yaw = _fit_ridge_cv(Xtr, y_sincos_tr)
    pred_sincos = m_yaw.predict(Xva)
    r2_yaw = _r2(y_sincos[val_mask], pred_sincos)
    pred_angle = np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1])
    true_angle = yaw[val_mask]
    ang_err = np.abs(np.angle(np.exp(1j * (pred_angle - true_angle))))
    out["yaw_r2"] = r2_yaw
    out["yaw_alpha"] = alpha_yaw
    out["median_angular_err_deg"] = float(np.degrees(np.median(ang_err)))

    # pitch
    m_pitch, alpha_pitch = _fit_ridge_cv(Xtr, pitch_tr)
    pred_pitch = m_pitch.predict(Xva)
    out["pitch_r2"] = _r2(pitch[val_mask], pred_pitch)
    out["pitch_alpha"] = alpha_pitch

    # log-distance
    m_logdist, alpha_logdist = _fit_ridge_cv(Xtr, logdist_tr)
    pred_logdist = m_logdist.predict(Xva)
    out["logdist_r2"] = _r2(logdist[val_mask], pred_logdist)
    out["logdist_alpha"] = alpha_logdist
    out["median_abs_logdist_err"] = float(np.median(np.abs(pred_logdist - logdist[val_mask])))

    return out


def run_target_location_probe(
    run_dirs: list[Path],
    cache_dir: Path,
    *,
    device: str | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """End-to-end: load corpus once, run per-model extraction + probes."""
    from qnn.bc.supervised_loop import make_resident_source_from_cache
    from qnn.diag.loader import load_policy

    if not HAVE_SKLEARN:
        raise RuntimeError("sklearn not available — pip install scikit-learn")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dir = Path(cache_dir) / "precomputed_val"

    print(f"[target-loc-probe] cache={val_dir} device={device}", flush=True)
    t0 = time.monotonic()
    source = make_resident_source_from_cache(
        val_dir, torch.device(device), segment_mask=ENGAGED_MASK,
    )
    n_frames = int(source.n_total_rows)
    n_episodes = max(0, len(source.episode_offsets) - 1)
    print(
        f"[target-loc-probe] corpus loaded: {n_episodes} engaged segments, "
        f"{n_frames:,} frames ({time.monotonic()-t0:.1f}s)", flush=True,
    )

    report: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "segment_mask": ENGAGED_MASK,
        "n_frames_loaded": n_frames,
        "n_segments_loaded": n_episodes,
        "models": {},
    }

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        run_name = run_dir.name
        print(f"[target-loc-probe] === model: {run_name} ===", flush=True)
        t1 = time.monotonic()
        policy, _probe = load_policy(run_dir, device=device)
        print(f"[target-loc-probe]   policy loaded ({time.monotonic()-t1:.1f}s)", flush=True)

        t2 = time.monotonic()
        frames = extract_frames(policy, source)
        print(
            f"[target-loc-probe]   extracted {frames['readout'].shape[0]:,} frames "
            f"({time.monotonic()-t2:.1f}s)", flush=True,
        )

        has_target = frames["has_target"]
        n_has_target = int(has_target.sum())
        print(f"[target-loc-probe]   frames with live target: {n_has_target:,}", flush=True)
        if n_has_target < 200:
            raise RuntimeError(
                f"{run_name}: only {n_has_target} frames with target_probs[:,0]<1 "
                "— too few to fit a probe"
            )

        readout = frames["readout"][has_target]
        target_feat = frames["target_feat"][has_target]
        yaw = frames["yaw"][has_target]
        pitch = frames["pitch"][has_target]
        logdist = frames["logdist"][has_target]
        episode_id = frames["episode_id"][has_target]

        train_mask, val_mask = _episode_train_val_split(episode_id, val_frac=val_frac, seed=seed)
        n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
        print(f"[target-loc-probe]   split: {n_train:,} train / {n_val:,} val frames "
              f"({len(np.unique(episode_id))} episodes)", flush=True)

        concat = np.concatenate([readout, target_feat], axis=1)
        reps = {"readout": readout, "target_feat": target_feat, "concat": concat}

        model_report: dict[str, Any] = {
            "n_frames_with_target": n_has_target,
            "n_train_frames": n_train,
            "n_val_frames": n_val,
            "n_episodes": int(len(np.unique(episode_id))),
            "representations": {},
        }
        for rep_name, X in reps.items():
            t3 = time.monotonic()
            result = probe_representation(
                X, yaw, pitch, logdist, train_mask, val_mask, shuffle_control=False, seed=seed,
            )
            shuffled = probe_representation(
                X, yaw, pitch, logdist, train_mask, val_mask, shuffle_control=True, seed=seed,
            )
            result["shuffled_control_r2"] = {
                "yaw": shuffled["yaw_r2"],
                "pitch": shuffled["pitch_r2"],
                "logdist": shuffled["logdist_r2"],
            }
            model_report["representations"][rep_name] = result
            print(
                f"[target-loc-probe]   {run_name}/{rep_name}: "
                f"yaw_r2={result['yaw_r2']:.3f} pitch_r2={result['pitch_r2']:.3f} "
                f"logdist_r2={result['logdist_r2']:.3f} "
                f"median_ang_err={result['median_angular_err_deg']:.1f}deg "
                f"shuffled_yaw_r2={shuffled['yaw_r2']:.3f} "
                f"({time.monotonic()-t3:.1f}s)", flush=True,
            )
        report["models"][run_name] = model_report

    return report


def render_table(report: dict[str, Any]) -> str:
    lines = []
    header = (
        f"{'model':<28} {'repr':<12} {'yaw R2':>8} {'pitch R2':>9} "
        f"{'logdist R2':>11} {'med ang err':>12} {'shuf yaw R2':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for model_name, mrep in report["models"].items():
        for rep_name, r in mrep["representations"].items():
            lines.append(
                f"{model_name:<28} {rep_name:<12} {r['yaw_r2']:>8.3f} "
                f"{r['pitch_r2']:>9.3f} {r['logdist_r2']:>11.3f} "
                f"{r['median_angular_err_deg']:>10.1f}deg "
                f"{r['shuffled_control_r2']['yaw']:>12.3f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_cli_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "target-location-probe",
        help="Linear recoverability of target location from readout/target.feat/concat.",
        description=__doc__,
    )
    p.add_argument("--run-dir", type=Path, action="append", required=True, dest="run_dirs",
                   help="Run directory (repeatable). Must contain config/probe.json + checkpoints/.")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Root cache directory; resident source built from <cache-dir>/precomputed_val.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="Write the JSON report to this path (required for file output).")


def run_cli(args) -> None:
    report = run_target_location_probe(
        args.run_dirs, args.cache_dir,
        device=args.device, val_frac=args.val_frac, seed=args.seed,
    )
    print()
    print(render_table(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[target-loc-probe] wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli_parser(sub)
    ns = parser.parse_args()
    run_cli(ns)
