"""LightGBM move labeler — a fast, robust GBT fit on the shared data layer.

Why this exists
---------------
The move GBT was historically slow/fragile only because of hand-rolled data
plumbing (full-corpus per-episode predict loops, RAM bugs, alignment hacks in
scripts/). The fit itself is ~38 s/axis. ``qnn.labeler.data`` already owns the
correct loading + featurization + op_input masking; this module reuses it and
treats the GBT as just a different fit on the same materialized matrix.

What it does
------------
  * materialize train + val via ``qnn.labeler.data.materialize_split`` (apply
    the op_input keep-mask; optionally cap train rows for speed),
  * fit one LightGBM multiclass booster per axis (fb / lr / ud), 300 rounds,
    num_threads capped (default 8), with a per-round timing log,
  * predict val on the FLAT (N, F) matrix in ONE batched call per axis (no
    per-episode Python loop — that loop was the ~200 s eval tail),
  * eval: per-axis frame accuracy AND a relabel-quality metric — predictions
    and truth downsampled to 20 Hz per episode (windowed-union, stride =
    round(native_hz / 20)) with per-axis 20 Hz agreement + switch-rate vs
    truth, respecting episode boundaries via the materializer's index,
  * fb/lr segment parity on the same 20 Hz streams (qnn.labeler.seg_stats):
    onset-rate ratio + duration-bucket TV distance vs truth — the a25
    move_seg head trains on segment (onset, duration) targets, so a labeler
    is judged on segment statistics, not just frame accuracy,
  * save the boosters + meta.json to artifacts/labeler/move_gbt_<tag>/.

CPU-only. No torch import on this path (the data layer is torch-free). Cap OMP
threads via --num-threads so a lightgbm fit can't stampede the box.

Usage:
    PYTHONPATH=src python -m qnn.labeler.gbt \\
        --data-dir artifacts/collect/qwd_labeler \\
        --tag qwd_v1 \\
        --max-train-frames 3000000 --num-threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import (
    FeatureSpec,
    N_CLASSES,
    matched_episode_strides,
    materialize_split,
)
from .decode import fit_switch_penalty
from .seg_stats import (
    downsample_axis,
    segment_parity,
    window_all_true,
    window_ids,
)

AXES = ("fb", "lr", "ud")

# Velocity-lag offsets for the optional temporal feature expansion.  The 9-dim
# single-frame feature set tops out at the grouped-CV ceiling (fb ~80% /
# lr ~75%) because move is momentum-dominated: a frame's press shows up in
# trailing velocity a few ticks LATER, so a frame-local GBT can't see it.
# These lags give the GBT the same trailing/leading velocity context the TCN
# gets from its convolutions — lifting fb/lr to the ~90% the diagnostic sees.
# Mirrors scripts/move_labeler_lib.VEL_OFFSETS (validated lag set).
VEL_LAG_OFFSETS = (-4, -2, -1, 1, 2, 3, 4, 6, 8)   # 0 already in base features


# ── temporal lag features (episode-boundary-aware) ─────────────────────────────

def add_velocity_lags(
    X: np.ndarray,                 # (N, F) base features (vel = first 3 cols)
    episode_starts: np.ndarray,    # (E+1,) row offsets
    offsets: tuple[int, ...] = VEL_LAG_OFFSETS,
    rows: np.ndarray | None = None,  # subset of row indices to materialize
    lag_cols: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """Append shifted copies of the ``lag_cols`` columns, edge-clamped within
    each episode (a lag never reads across an episode boundary).

    The materialized feature matrix puts body-frame velocity in columns 0..2
    and the per-emit view delta (look) in columns 6..8 (see build_features).
    Velocity lags are the default; adding the look columns gives the
    frame-local GBT the ROTATION rhythm too — air-strafe lr presses sync
    with the direction the view is swinging, a pattern invisible in any
    single frame.  Lagged neighbor indices are gathered from the FULL base
    ``X`` (so neighbors are real frames) but only the requested ``rows`` are
    emitted — pass the capped train subset to keep the output (and peak RAM)
    at ``len(rows)``, not ``N``.  ``rows=None`` emits all rows.  Returns
    ``(len(rows), F + len(lag_cols)*len(offsets))`` float32.
    """
    # Memory-frugal build: the dev container is cgroup-capped at 16 GB, and
    # the old parts-list + concatenate pattern doubled the output allocation
    # (12M rows x 63 cols OOM-killed at cap).  Preallocate the output, write
    # each lag block in place, and keep per-frame intermediates sel-local.
    n = X.shape[0]
    starts = episode_starts[:-1]
    lengths = np.diff(episode_starts)
    ep_of_frame = np.repeat(np.arange(starts.shape[0], dtype=np.int32), lengths)

    sel = np.arange(n) if rows is None else np.asarray(rows)
    ep_idx = ep_of_frame[sel].astype(np.int64)
    ep_start = starts[ep_idx]
    ep_end = ep_start + lengths[ep_idx] - 1                # inclusive last row

    F = X.shape[1]
    L = len(lag_cols)
    out = np.empty((sel.shape[0], F + L * len(offsets)), dtype=np.float32)
    out[:, :F] = X[sel]
    vel = X[:, list(lag_cols)]                             # one (n, L) lag source
    for i, k in enumerate(offsets):
        src = np.clip(sel + k, ep_start, ep_end)           # clamp within episode
        out[:, F + i * L: F + (i + 1) * L] = vel[src]
    return out


# 20 Hz windowing/downsample helpers (window_ids / downsample_axis /
# window_all_true) live in qnn.labeler.seg_stats — shared with the TCN
# trainer's segment-parity gate.


def _switch_rate(seq: np.ndarray) -> float:
    """Fraction of adjacent pairs that differ (per-axis class transitions)."""
    if seq.shape[0] < 2:
        return 0.0
    return float(np.mean(seq[1:] != seq[:-1]))


# ── fit ────────────────────────────────────────────────────────────────────────

def _fit_axis(
    lgb,
    X: np.ndarray,
    y: np.ndarray,
    *,
    axis: str,
    num_rounds: int,
    num_threads: int,
    seed: int,
    log_every: int = 50,
) -> "object":
    """Fit one multiclass booster on (X, y) with a per-round timing log."""
    params = {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "learning_rate": 0.1,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "num_threads": num_threads,
        "seed": seed,
        "verbosity": -1,
    }
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)

    t_axis = time.time()
    last = {"t": t_axis, "r": 0}

    def _cb(env) -> None:
        r = env.iteration + 1
        if r % log_every == 0 or r == num_rounds:
            now = time.time()
            dr = r - last["r"]
            rate = dr / max(1e-6, now - last["t"])
            print(f"    [{axis}] round {r:4d}/{num_rounds}  "
                  f"{now - t_axis:6.1f}s  ({rate:.1f} rounds/s)", flush=True)
            last["t"] = now
            last["r"] = r

    booster = lgb.train(
        params, dtrain, num_boost_round=num_rounds, callbacks=[_cb],
    )
    return booster


def _predict_proba(booster, X: np.ndarray) -> np.ndarray:
    """Batched class probabilities over the whole flat matrix (one call)."""
    return booster.predict(X)             # (N, num_class)


# ── main ────────────────────────────────────────────────────────────────────────

def run(
    data_dir: Path,
    tag: str,
    *,
    out_root: Path,
    max_train_frames: int,
    num_rounds: int,
    num_threads: int,
    native_hz: int,
    seed: int,
    temporal_lags: bool = True,
    lag_look: bool = False,
) -> None:
    # Cap OMP / BLAS threads as a belt-and-braces guard alongside lightgbm's
    # own num_threads, so a CPU fit can't stampede a shared box.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(num_threads))
    import lightgbm as lgb  # local import keeps module import cheap

    spec = FeatureSpec()    # core 9-dim single-frame features
    rng = np.random.default_rng(seed)
    stride = max(1, round(native_hz / 20))

    print(f"materializing splits from {data_dir}", flush=True)
    t0 = time.time()
    train = materialize_split(data_dir / "precomputed_train", spec)
    val   = materialize_split(data_dir / "precomputed_val", spec)
    print(f"  train: {train.n_episodes} eps  {train.n_frames:,} frames  "
          f"X{train.X.shape}", flush=True)
    print(f"  val:   {val.n_episodes} eps  {val.n_frames:,} frames  "
          f"X{val.X.shape}", flush=True)
    print(f"  materialize: {time.time() - t0:.1f}s  "
          f"(20Hz stride = round({native_hz}/20) = {stride})", flush=True)

    # Cap train rows for speed.  Sample row indices uniformly; the keep-mask
    # is applied per axis at fit time so a capped row set still drops no-op
    # frames axis-by-axis.  We cap FIRST, then build velocity lags only for
    # the capped rows (gathering neighbors from the full base X) so the
    # 3×-wider lag matrix is materialized at the capped size, not 78M rows.
    Ytr = train.Y
    Mtr = train.mask
    sel = None
    if max_train_frames and train.n_frames > max_train_frames:
        sel = rng.choice(train.n_frames, size=max_train_frames, replace=False)
        sel.sort()
        Ytr = Ytr[sel]
        Mtr = Mtr[sel]
        print(f"  capped train to {sel.shape[0]:,} frames "
              f"(--max-train-frames)", flush=True)

    # Velocity-lag expansion (episode-boundary-aware): gives the frame-local
    # GBT the trailing/leading momentum context the TCN gets from its
    # convolutions, lifting fb/lr from the ~80%/75% single-frame ceiling to
    # the ~90% the diagnostic reports.  Each lag is edge-clamped within its
    # episode (never reads across a boundary).  Train lags are built only for
    # the capped rows; val is materialized in full (we predict on all rows).
    if temporal_lags:
        # vel cols 0..2 always; look cols 6..8 opt-in (--lag-look) for the
        # rotation-rhythm context (air-strafe lr sync).
        lag_cols = (0, 1, 2, 6, 7, 8) if lag_look else (0, 1, 2)
        Xtr = add_velocity_lags(train.X, train.episode_starts, rows=sel,
                                lag_cols=lag_cols)
        val_X = add_velocity_lags(val.X, val.episode_starts, lag_cols=lag_cols)
        print(f"  +temporal lags (cols {lag_cols}): feat dim "
              f"{train.X.shape[1]} -> {Xtr.shape[1]} "
              f"(offsets {VEL_LAG_OFFSETS})", flush=True)
    else:
        Xtr = train.X if sel is None else train.X[sel]
        val_X = val.X

    # The base train matrix (34M x 9 on the full matched corpus) is dead
    # weight once Xtr exists — free it before the fits, or the per-axis
    # copy + LightGBM dataset construction busts the 16 GB container cap.
    train_frames_full = train.n_frames
    del train

    # Precompute episode-aware 20 Hz window ids for the val split once.
    # Matched corpora carry per-episode native rates (77 Hz / 60 Hz mixed) —
    # use the qobs-derived per-episode strides so windows are real 20 Hz
    # model frames; plain labeler corpora fall back to the global stride.
    val_strides = matched_episode_strides(data_dir / "precomputed_val", stride)
    if val_strides is not None and val_strides.shape[0] == val.n_episodes:
        win_stride: "int | np.ndarray" = val_strides
        dist = {int(v): int(c) for v, c in
                zip(*np.unique(val_strides, return_counts=True))}
        print(f"  per-episode 20Hz strides (matched qobs): {dist}", flush=True)
    else:
        win_stride = stride
        if val_strides is not None:
            print(f"  WARNING: stride/episode count mismatch "
                  f"({val_strides.shape[0]} vs {val.n_episodes}); "
                  f"using global stride {stride}", flush=True)
    val_win_id, val_n_windows, val_win_starts = window_ids(val.episode_starts,
                                                           win_stride)

    out_dir = out_root / f"move_gbt_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    boosters = {}
    fit_secs = {}
    eval_secs = {}
    frame_acc = {}
    hz20 = {}     # per-axis 20 Hz agreement + switch rates
    seg20 = {}    # fb/lr 20 Hz segment parity (move_seg gate: onset + duration)
    decode_fit = {}  # fb/lr Viterbi switch-penalty fit (frontier per axis)

    for ai, axis in enumerate(AXES):
        # Train rows where this axis is operative (keep-mask True).
        keep = Mtr[:, ai]
        Xa = Xtr[keep]
        ya = Ytr[keep, ai].astype(np.int64)
        print(f"  [{axis}] fitting on {Xa.shape[0]:,} operative frames "
              f"(class dist {np.bincount(ya, minlength=N_CLASSES).tolist()})",
              flush=True)

        t_fit = time.time()
        booster = _fit_axis(
            lgb, Xa, ya, axis=axis, num_rounds=num_rounds,
            num_threads=num_threads, seed=seed,
        )
        fit_secs[axis] = time.time() - t_fit
        boosters[axis] = booster

        # ── batched val predict (one call over the FLAT matrix) ──
        t_pred = time.time()
        proba_full = _predict_proba(booster, val_X)           # (Nval, 3) all rows
        pred_full = proba_full.argmax(axis=1).astype(np.int64)
        eval_secs[axis] = time.time() - t_pred

        truth_full = val.Y[:, ai].astype(np.int64)
        vkeep = val.mask[:, ai]
        # Frame accuracy on operative val frames (matches training masking).
        frame_acc[axis] = float(np.mean(pred_full[vkeep] == truth_full[vkeep]))

        # ── 20 Hz relabel-quality (boundary-aware, windowed-union) ──
        pred_20 = downsample_axis(pred_full, val_win_id, val_n_windows)
        truth_20 = downsample_axis(truth_full, val_win_id, val_n_windows)
        hz20[axis] = {
            "agree": float(np.mean(pred_20 == truth_20)),
            "switch_pred": _switch_rate(pred_20),
            "switch_truth": _switch_rate(truth_20),
            "n_windows": int(pred_20.shape[0]),
        }

        # ── 20 Hz segment parity (fb/lr — what move_seg trains on) ──
        # Onset-rate ratio + duration-bucket TV distance of pred vs truth;
        # windows containing a non-operative frame are treated as invalid
        # (segments break there, mirroring derive_segment_targets).
        if axis in ("fb", "lr"):
            valid_20 = window_all_true(vkeep, val_win_id, val_n_windows)
            seg20[axis] = segment_parity(pred_20, truth_20, val_win_starts,
                                         valid=valid_20)

            # ── decode fit: transition-penalized Viterbi (qnn.labeler.decode)
            # λ sweep scored on the same gate; frontier lands in meta.json.
            t_fit2 = time.time()
            log_probs = np.log(np.clip(proba_full, 1e-12, None))
            decode_fit[axis] = fit_switch_penalty(
                log_probs, truth_full, val.episode_starts, win_stride,
                valid_native=vkeep)
            best = next(r for r in decode_fit[axis]["frontier"]
                        if r["lam"] == decode_fit[axis]["lam"])
            print(f"  [{axis}] decode fit ({time.time() - t_fit2:.1f}s): "
                  f"lam={best['lam']}  onset x{best['onset_ratio']}  "
                  f"durTV {best['dur_tv']:.3f}  "
                  f"agree {best['agree_20hz'] * 100:.2f}%", flush=True)

        booster.save_model(str(out_dir / f"booster_{axis}.txt"))
        seg_msg = ""
        if axis in seg20:
            s = seg20[axis]
            ratio = s["onset_ratio"]
            seg_msg = (f"  seg onset x{ratio:.2f}  " if ratio is not None
                       else "  seg onset n/a  ") + f"durTV {s['dur_tv']:.3f}"
        print(f"  [{axis}] fit {fit_secs[axis]:.1f}s  "
              f"eval {eval_secs[axis]:.2f}s  "
              f"frame_acc {frame_acc[axis] * 100:.2f}%  "
              f"20Hz agree {hz20[axis]['agree'] * 100:.2f}%  "
              f"switch p/t {hz20[axis]['switch_pred']:.3f}/"
              f"{hz20[axis]['switch_truth']:.3f}{seg_msg}", flush=True)

    meta = {
        "tag": tag,
        "data_dir": str(data_dir),
        "feat_spec": {
            "base_dim": spec.dim,
            "use_weapon_id": spec.use_weapon_id,
            "use_baseline": spec.use_baseline,
            "clip_velocity": spec.clip_velocity,
        },
        "temporal_lags": bool(temporal_lags),
        "lag_look": bool(lag_look),
        "vel_lag_offsets": list(VEL_LAG_OFFSETS) if temporal_lags else [],
        "feat_dim": int(Xtr.shape[1]),
        "axes": list(AXES),
        "native_hz": native_hz,
        "downsample_stride": stride,
        "per_episode_strides": (
            {int(v): int(c) for v, c in zip(*np.unique(win_stride, return_counts=True))}
            if not np.isscalar(win_stride) else None),
        "num_rounds": num_rounds,
        "num_threads": num_threads,
        "seed": seed,
        "train_frames": int(Xtr.shape[0]),
        "train_frames_full": int(train_frames_full),
        "val_frames": int(val.n_frames),
        "val_episodes": int(val.n_episodes),
        "frame_acc": {a: frame_acc[a] for a in AXES},
        "hz20": hz20,
        "seg20": seg20,
        "decode_fit": decode_fit,
        "fit_secs": fit_secs,
        "eval_secs": eval_secs,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── summary table ──
    print("\n  ── summary ─────────────────────────────────────────────")
    print(f"  {'axis':4s}  {'frame_acc':>9s}  {'20Hz_agree':>10s}  "
          f"{'sw_pred':>7s}  {'sw_truth':>8s}  {'onset_x':>7s}  "
          f"{'dur_tv':>6s}  {'fit_s':>6s}  {'eval_s':>6s}")
    for a in AXES:
        s = seg20.get(a)
        onset_x = (f"{s['onset_ratio']:7.3f}" if s and s["onset_ratio"] is not None
                   else f"{'—':>7s}")
        dur_tv = f"{s['dur_tv']:6.3f}" if s else f"{'—':>6s}"
        print(f"  {a:4s}  {frame_acc[a] * 100:8.2f}%  "
              f"{hz20[a]['agree'] * 100:9.2f}%  "
              f"{hz20[a]['switch_pred']:7.3f}  {hz20[a]['switch_truth']:8.3f}  "
              f"{onset_x}  {dur_tv}  "
              f"{fit_secs[a]:6.1f}  {eval_secs[a]:6.2f}")
    print(f"\n  total fit {sum(fit_secs.values()):.1f}s  "
          f"total eval {sum(eval_secs.values()):.2f}s  ->  {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Labeler corpus with precomputed_train/ + precomputed_val/.")
    ap.add_argument("--tag", required=True,
                    help="Output suffix: artifacts/labeler/move_gbt_<tag>/.")
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/labeler"),
                    help="Parent dir for the move_gbt_<tag> output (default "
                         "artifacts/labeler).")
    ap.add_argument("--max-train-frames", type=int, default=3_000_000,
                    help="Cap train rows (uniform subsample) for fit speed. "
                         "0 = use all. Default 3M (~2 min total fit).")
    ap.add_argument("--num-rounds", type=int, default=300,
                    help="Boosting rounds per axis (default 300).")
    ap.add_argument("--num-threads", type=int, default=8,
                    help="LightGBM + OMP thread cap (default 8). Keep low on a "
                         "shared box.")
    ap.add_argument("--native-hz", type=int, default=77,
                    help="Native collect rate; 20 Hz downsample stride = "
                         "round(native_hz/20) (default 77 → stride 4).")
    ap.add_argument("--lag-look", action="store_true",
                    help="Also lag the 3 look (per-emit view delta) columns, "
                         "not just velocity — gives the frame-local GBT the "
                         "rotation rhythm (air-strafe lr presses sync with "
                         "yaw swing direction).")
    ap.add_argument("--no-temporal-lags", action="store_true",
                    help="Disable the episode-aware velocity-lag feature "
                         "expansion (fit on the 9-dim single-frame features "
                         "only).  With lags off, fb/lr sit at the ~80%%/75%% "
                         "frame-local ceiling; on (default) they reach ~90%%.")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    run(
        args.data_dir,
        args.tag,
        out_root=args.out_root,
        max_train_frames=args.max_train_frames,
        num_rounds=args.num_rounds,
        num_threads=args.num_threads,
        native_hz=args.native_hz,
        seed=args.seed,
        temporal_lags=not args.no_temporal_lags,
        lag_look=args.lag_look,
    )


if __name__ == "__main__":
    main()
