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
    materialize_split,
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
) -> np.ndarray:
    """Append shifted copies of the 3 velocity columns, edge-clamped within
    each episode (a lag never reads across an episode boundary).

    The materialized feature matrix puts body-frame velocity in columns 0..2
    (see build_features); we lag only those.  Lagged neighbor indices are
    gathered from the FULL base ``X`` (so neighbors are real frames) but only
    the requested ``rows`` are emitted — pass the capped train subset to keep
    the output (and peak RAM) at ``len(rows)``, not ``N``.  ``rows=None``
    emits all rows.  Returns ``(len(rows), F + 3*len(offsets))`` float32.
    """
    n = X.shape[0]
    vel = X[:, :3]
    starts = episode_starts[:-1]
    lengths = np.diff(episode_starts)
    # episode id + clamp bounds for every frame in the split
    ep_of_frame = np.repeat(np.arange(starts.shape[0]), lengths)
    ep_start_all = starts[ep_of_frame]
    ep_end_all = ep_start_all + lengths[ep_of_frame] - 1   # inclusive last row

    sel = np.arange(n) if rows is None else np.asarray(rows)
    ep_start = ep_start_all[sel]
    ep_end = ep_end_all[sel]
    cols = [X[sel]]
    for k in offsets:
        src = np.clip(sel + k, ep_start, ep_end)           # clamp within episode
        cols.append(vel[src])
    return np.concatenate(cols, axis=1).astype(np.float32)


# ── 20 Hz windowed-union downsample (boundary-aware) ───────────────────────────

def _window_ids(episode_starts: np.ndarray, stride: int) -> tuple[np.ndarray, int]:
    """Assign each frame a contiguous global window id, episode-aware.

    Window ``stride``-frame windows never straddle an episode boundary.
    Returns ``(win_id (N,), n_windows)``.
    """
    n = int(episode_starts[-1])
    starts = episode_starts[:-1]
    lengths = np.diff(episode_starts)
    # within-episode frame offset
    ep_of_frame = np.repeat(np.arange(starts.shape[0]), lengths)
    within = np.arange(n) - starts[ep_of_frame]
    win_within = within // stride                       # window idx inside episode
    n_win_per_ep = (lengths + stride - 1) // stride
    win_base = np.zeros(starts.shape[0] + 1, dtype=np.int64)
    win_base[1:] = np.cumsum(n_win_per_ep)
    win_id = win_base[ep_of_frame] + win_within
    return win_id.astype(np.int64), int(win_base[-1])


def _downsample_axis(
    labels: np.ndarray,            # (N,) int  per-axis class {0=neg,1=none,2=pos}
    win_id: np.ndarray,            # (N,) int  global episode-aware window id
    n_windows: int,
) -> np.ndarray:
    """Downsample a per-frame axis label stream to ~20 Hz, per episode.

    Windowed-union (vectorized): within each window the class is the
    most-common NON-none press if any frame pressed, else none.  Mirrors how
    a 20 Hz controller registers a button held for part of the window.
    Episode boundaries are respected via ``win_id`` (see ``_window_ids``).
    """
    # Per-window counts of neg (class 0) and pos (class 2) presses.
    out = np.ones(n_windows, dtype=np.int64)            # default 'none'
    neg = np.bincount(win_id[labels == 0], minlength=n_windows)
    pos = np.bincount(win_id[labels == 2], minlength=n_windows)
    pressed = (neg + pos) > 0
    # most-common pressed class; ties (neg==pos) resolve to pos (argmax-style,
    # matching np.bincount([0,2]).argmax() == 0 would pick neg, so mirror that)
    out[pressed] = np.where(neg[pressed] >= pos[pressed], 0, 2)
    return out


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


def _predict_classes(booster, X: np.ndarray) -> np.ndarray:
    """Batched argmax prediction over the whole flat matrix (one call)."""
    proba = booster.predict(X)            # (N, num_class)
    return proba.argmax(axis=1).astype(np.int64)


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
        Xtr = add_velocity_lags(train.X, train.episode_starts, rows=sel)
        val_X = add_velocity_lags(val.X, val.episode_starts)
        print(f"  +velocity lags: feat dim {train.X.shape[1]} -> "
              f"{Xtr.shape[1]} (offsets {VEL_LAG_OFFSETS})", flush=True)
    else:
        Xtr = train.X if sel is None else train.X[sel]
        val_X = val.X

    # Precompute episode-aware 20 Hz window ids for the val split once.
    val_win_id, val_n_windows = _window_ids(val.episode_starts, stride)

    out_dir = out_root / f"move_gbt_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    boosters = {}
    fit_secs = {}
    eval_secs = {}
    frame_acc = {}
    hz20 = {}     # per-axis 20 Hz agreement + switch rates

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
        pred_full = _predict_classes(booster, val_X)          # (Nval,) all rows
        eval_secs[axis] = time.time() - t_pred

        truth_full = val.Y[:, ai].astype(np.int64)
        vkeep = val.mask[:, ai]
        # Frame accuracy on operative val frames (matches training masking).
        frame_acc[axis] = float(np.mean(pred_full[vkeep] == truth_full[vkeep]))

        # ── 20 Hz relabel-quality (boundary-aware, windowed-union) ──
        pred_20 = _downsample_axis(pred_full, val_win_id, val_n_windows)
        truth_20 = _downsample_axis(truth_full, val_win_id, val_n_windows)
        hz20[axis] = {
            "agree": float(np.mean(pred_20 == truth_20)),
            "switch_pred": _switch_rate(pred_20),
            "switch_truth": _switch_rate(truth_20),
            "n_windows": int(pred_20.shape[0]),
        }

        booster.save_model(str(out_dir / f"booster_{axis}.txt"))
        print(f"  [{axis}] fit {fit_secs[axis]:.1f}s  "
              f"eval {eval_secs[axis]:.2f}s  "
              f"frame_acc {frame_acc[axis] * 100:.2f}%  "
              f"20Hz agree {hz20[axis]['agree'] * 100:.2f}%  "
              f"switch p/t {hz20[axis]['switch_pred']:.3f}/"
              f"{hz20[axis]['switch_truth']:.3f}", flush=True)

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
        "vel_lag_offsets": list(VEL_LAG_OFFSETS) if temporal_lags else [],
        "feat_dim": int(Xtr.shape[1]),
        "axes": list(AXES),
        "native_hz": native_hz,
        "downsample_stride": stride,
        "num_rounds": num_rounds,
        "num_threads": num_threads,
        "seed": seed,
        "train_frames": int(Xtr.shape[0]),
        "train_frames_full": int(train.n_frames),
        "val_frames": int(val.n_frames),
        "val_episodes": int(val.n_episodes),
        "frame_acc": {a: frame_acc[a] for a in AXES},
        "hz20": hz20,
        "fit_secs": fit_secs,
        "eval_secs": eval_secs,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── summary table ──
    print("\n  ── summary ─────────────────────────────────────────────")
    print(f"  {'axis':4s}  {'frame_acc':>9s}  {'20Hz_agree':>10s}  "
          f"{'sw_pred':>7s}  {'sw_truth':>8s}  {'fit_s':>6s}  {'eval_s':>6s}")
    for a in AXES:
        print(f"  {a:4s}  {frame_acc[a] * 100:8.2f}%  "
              f"{hz20[a]['agree'] * 100:9.2f}%  "
              f"{hz20[a]['switch_pred']:7.3f}  {hz20[a]['switch_truth']:8.3f}  "
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
    )


if __name__ == "__main__":
    main()
