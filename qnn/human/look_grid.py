"""Data-driven look turn-delta magnitude grid: fit, sufficient stats, drift.

The look head discretizes the per-frame view turn-delta magnitude θ (radians,
θ = angle between this frame's look direction and forward) into a hold bin plus
``N_MAG`` foveated magnitude bins. Historically those magnitude centers were a
hardcoded power law (``qnn.model.look_bins`` ``_FOVEA_POWER``); see that module.
A power law is not fit to any distribution — at our actual θ distribution it
leaves ~half the bins near-empty and carries an avoidable quantization floor,
and the right grid is *rate-dependent* (a 10 Hz collect turns ~2× as far per
frame as a 20 Hz one).

This module computes, from a collect's stored look labels:
  - a θ **histogram** — the durable sufficient statistic, so any grid can be
    refit later (or with a different algorithm) without re-walking demos,
  - the **hold fraction** (θ < hold_max), the single best drift scalar,
  - a **Lloyd-Max** (minimum-distortion) magnitude grid fit to that histogram.

The collect records this block in ``collect_metadata.json`` (observe + drift
monitor). A *run* pins a chosen grid into its own config (adopt) — analogous to
``collection_fingerprint``: the collect generates the candidate, the run copies
one in, and existing runs never change retroactively.

NumPy only — no torch. ``qnn.bc.collect`` and its import chain must stay
torch-free, so this module (imported by the collect) cannot pull torch.
``qnn.model.look_bins`` (torch) consumes the defaults/fit from here.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

# Architecture defaults (mirrored by qnn.model.look_bins, which imports these).
DEFAULT_N_MAG: int = 12
DEFAULT_FOVEA_POWER: float = 2.5
THETA_MAX: float = float(np.pi)          # turn angle ≤ π

# Label deadzone = the source resolution floor (QW angle16: 360/65536 deg).
# Since the 7/06 point-mass-hold scheme (research/look-head.md): hold is the
# TRUE stillness point mass (~19% exact zeros — zero mouse counts in a tick),
# not a small-angle region; below this floor a "turn" is physically
# meaningless in the recording format. Decoupled from grid geometry — the
# legacy rung1/2 rule deleted real drift as a side-effect of rung placement.
SOURCE_FLOOR_RAD: float = float(2.0 * np.pi / 65536.0)   # ≈ 9.59e-5 rad = 0.0055°

# θ histogram (sufficient stat) resolution. Uniform over [0, π].
# 7200 (0.025°/bin) since the 7/06 fp16 root-cause fix: the honest corpus has
# ~20% of its nonzero mass below 1° and the Lloyd-Max fovea rung lands at
# ~0.9° — 0.25° bins cannot place it. Metadata blocks written at 720 bins
# remain readable (theta_hist.nbins is stored); drift EMD comparisons require
# matching nbins on both sides.
HIST_NBINS: int = 7200                   # 0.025° per bin
HIST_EDGES: np.ndarray = np.linspace(0.0, THETA_MAX, HIST_NBINS + 1)
_HIST_CENTERS: np.ndarray = (HIST_EDGES[:-1] + HIST_EDGES[1:]) / 2.0


def default_mag_centers(
    n_mag: int = DEFAULT_N_MAG, fovea_power: float = DEFAULT_FOVEA_POWER
) -> np.ndarray:
    """Legacy power-law positive magnitude centers (radians), length ``n_mag``.

    The fallback grid when a run pins nothing. Excludes the hold bin (center 0).
    """
    k = np.arange(1, n_mag + 1, dtype=np.float64) / n_mag
    return (k ** fovea_power) * THETA_MAX


def hold_max_from_centers(mag_centers: np.ndarray) -> float:
    """Hold threshold: half the smallest magnitude center (matches look_bins)."""
    return float(np.min(mag_centers) * 0.5)


def default_dir_centers(n_dir: int = 16) -> np.ndarray:
    """Uniform direction-bin centers (radians) over [0, 2π). Rate-invariant."""
    return (np.arange(n_dir, dtype=np.float64) + 0.5) * (2.0 * np.pi / n_dir)


def default_cartesian_centers(
    n_bins: int = 25, fovea_power: float = DEFAULT_FOVEA_POWER
) -> np.ndarray:
    """Legacy per-axis Cartesian tangent bin centers (radians), length ``n_bins``
    (odd → a center at 0). Used by the binned look head + the look_dll density."""
    half = (n_bins - 1) // 2
    k = np.arange(1, half + 1, dtype=np.float64) / half
    pos = (k ** fovea_power) * THETA_MAX
    return np.concatenate([-pos[::-1], [0.0], pos])


def code_default_grid(
    n_bins: int = 25, n_mag: int = DEFAULT_N_MAG, n_dir: int = 16,
    fovea_power: float = DEFAULT_FOVEA_POWER,
) -> dict:
    """The pre-data-driven hardcoded grid, as an explicit pinnable block.

    Reproduces the ``qnn.model.look_bins`` power-law constants exactly (verified
    against them in the migration), but in plain numpy so this stays torch-free.
    This is the grid every model trained before data-driven grids actually used,
    so it's what gets pinned into those models' run dirs — NOT a runtime
    fallback. ``mag_centers_rad`` includes the leading hold center (0)."""
    mag_pos = default_mag_centers(n_mag, fovea_power)
    return {
        "schema": "look_grid_v1",
        "source": "code_default",
        "fovea_power": float(fovea_power),
        "theta_max_rad": float(THETA_MAX),
        "n_bins": int(n_bins),
        "n_mag": int(n_mag),
        "n_dir": int(n_dir),
        "hold_max_rad": hold_max_from_centers(mag_pos),
        "mag_centers_rad": np.concatenate([[0.0], mag_pos]).tolist(),
        "dir_centers_rad": default_dir_centers(n_dir).tolist(),
        "cartesian_centers_rad": default_cartesian_centers(n_bins, fovea_power).tolist(),
    }


def theta_from_look(look: np.ndarray) -> np.ndarray:
    """(N, 3) unit look vectors → (N,) turn angle θ in radians. forward=(1,0,0).

    ``atan2(|yz|, x)``, NOT ``arccos(x)`` — on fp16-cached vectors the arccos
    form combs θ onto √n·1.79° and zeroes every turn below ~1.27° (the 7/06
    root cause, research/look-head.md); the transverse components carry the
    magnitude linearly and survive the cast.
    """
    x = look[:, 0].astype(np.float64)
    n = np.linalg.norm(look[:, 1:3].astype(np.float64), axis=1)
    return np.arctan2(n, x)


def histogram_theta(theta: np.ndarray) -> np.ndarray:
    """(N,) θ radians → (HIST_NBINS,) int64 counts over [0, π]."""
    counts, _ = np.histogram(theta, bins=HIST_EDGES)
    return counts.astype(np.int64)


def _fit_from_weighted_points(
    pts: np.ndarray, wts: np.ndarray, n_mag: int, seed: int, iters: int
) -> np.ndarray:
    """Weighted 1-D Lloyd-Max on (pts, wts). Always returns ``n_mag`` sorted,
    distinct centers — empty clusters are reseeded to the max-distortion point,
    so ``N_MAG`` stays fixed even when the data supports fewer effective bins."""
    rng = np.random.default_rng(seed)
    total = wts.sum()
    if total <= 0 or len(pts) == 0:
        return default_mag_centers(n_mag)
    # Weighted-quantile init (stable, seed only breaks reseed ties).
    cdf = np.cumsum(wts) / total
    qs = (np.arange(n_mag) + 0.5) / n_mag
    centers = np.interp(qs, cdf, pts)
    for _ in range(iters):
        assign = np.argmin(np.abs(pts[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for j in range(n_mag):
            m = assign == j
            w = wts[m].sum()
            if w > 0:
                new[j] = (pts[m] * wts[m]).sum() / w
            else:  # empty cluster → reseed to the worst-quantized point
                dist = wts * (pts - centers[assign]) ** 2
                new[j] = pts[int(np.argmax(dist))] + rng.uniform(-1e-9, 1e-9)
        new.sort()
        if np.allclose(new, centers, atol=1e-7):
            centers = new
            break
        centers = new
    # De-duplicate any coincident centers (paranoia; reseed jitter usually avoids).
    for j in range(1, n_mag):
        if centers[j] <= centers[j - 1]:
            centers[j] = centers[j - 1] + 1e-6
    return centers


def fit_mag_centers(
    hist_counts: np.ndarray,
    hold_max: float,
    n_mag: int = DEFAULT_N_MAG,
    seed: int = 0,
    iters: int = 200,
) -> np.ndarray:
    """Fit ``n_mag`` magnitude centers (radians) to a θ histogram, excluding the
    hold region (bin centers < ``hold_max``). Minimum-distortion (Lloyd-Max)."""
    counts = np.asarray(hist_counts, dtype=np.float64)
    nz = _HIST_CENTERS >= hold_max
    return _fit_from_weighted_points(_HIST_CENTERS[nz], counts[nz], n_mag, seed, iters)


def rms_distortion_deg(hist_counts: np.ndarray, mag_centers: np.ndarray, hold_max: float) -> float:
    """Population RMS snap error (degrees) of ``mag_centers`` against a θ histogram.
    Sub-hold mass snaps to 0; the rest to the nearest magnitude center."""
    counts = np.asarray(hist_counts, dtype=np.float64)
    snapped = mag_centers[np.argmin(np.abs(_HIST_CENTERS[:, None] - mag_centers[None, :]), axis=1)]
    snapped = np.where(_HIST_CENTERS < hold_max, 0.0, snapped)
    total = counts.sum()
    if total <= 0:
        return 0.0
    mse = (counts * (snapped - _HIST_CENTERS) ** 2).sum() / total
    return float(np.degrees(np.sqrt(mse)))


def emd_theta(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """1-D earth-mover distance (degrees) between two θ histograms. Drift metric."""
    a = np.asarray(hist_a, dtype=np.float64); b = np.asarray(hist_b, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    ca = np.cumsum(a / a.sum()); cb = np.cumsum(b / b.sum())
    width = np.degrees(HIST_EDGES[1] - HIST_EDGES[0])
    return float(np.sum(np.abs(ca - cb)) * width)


def pinned_grid_from_collect(
    collect_dir: str | Path, n_dir: int = 16, n_bins: int = 25,
    fovea_power: float = DEFAULT_FOVEA_POWER,
) -> dict:
    """Build a pinnable grid block from a collect's recorded data-fit grid.

    Reads ``collect_metadata.json['look_grid']['fit']`` (the Lloyd-Max magnitude
    centers) and assembles the ``config/look_grid.json`` a run pins: the fitted
    magnitudes (hold-prefixed to ``N_MAG+1``, matching ``look_bins.MAG_CENTERS``),
    a uniform direction grid (rate-invariant, not fit), and the default Cartesian
    ruler (the look_dll density metric stays a fixed cross-run comparison scale).
    ``run.init`` calls this so every new run pins the grid of its own corpus."""
    collect_dir = Path(collect_dir)
    meta = json.loads((collect_dir / "collect_metadata.json").read_text())
    lg = meta.get("look_grid")
    if not lg or "fit" not in lg:
        raise ValueError(
            f"{collect_dir}/collect_metadata.json has no look_grid.fit — recollect "
            "(or backfill via qnn.human.look_grid.compute_from_collect) first.")
    n_mag = int(lg["n_mag"])
    mag_full = [0.0] + list(lg["fit"]["mag_centers_rad"])  # hold-prefix → N_MAG+1
    if len(mag_full) != n_mag + 1:
        raise ValueError(f"fit has {len(mag_full) - 1} mag centers, expected n_mag={n_mag}")
    return {
        "schema": "look_grid_v1",
        "source": "corpus_fit",
        "corpus": str(collect_dir),
        "tick_hz": lg.get("tick_hz"),
        "fovea_power": float(fovea_power),
        "theta_max_rad": float(THETA_MAX),
        "n_bins": int(n_bins),
        "n_mag": n_mag,
        "n_dir": int(n_dir),
        "hold_max_rad": float(lg["hold_max_rad"]),
        # Explicit label deadzone (point-mass hold, 7/06). Present → honored by
        # install_polar_grid; absent (legacy metadata) → derived rung1/2 rule.
        **({"deadzone_rad": float(lg["deadzone_rad"])} if "deadzone_rad" in lg else {}),
        "hold_frac": lg.get("hold_frac"),
        "mag_centers_rad": mag_full,
        "dir_centers_rad": default_dir_centers(n_dir).tolist(),
        "cartesian_centers_rad": default_cartesian_centers(n_bins, fovea_power).tolist(),
        "fit_rms_deg": lg["fit"].get("rms_deg"),
        "default_rms_deg": lg.get("default", {}).get("rms_deg"),
    }


def compute_from_collect(
    collect_dir: str | Path,
    n_mag: int = DEFAULT_N_MAG,
    fovea_power: float = DEFAULT_FOVEA_POWER,
    seed: int = 0,
    tick_hz: int | float | None = None,
) -> dict:
    """Walk a collect's train-split look labels and build the ``look_grid``
    metadata block: θ histogram (sufficient stat), exact hold fraction, the
    fitted Lloyd-Max grid, and the legacy default's distortion for context.

    Hold threshold is the legacy default's — behavioral, not fit. Fits only the
    magnitude center positions; ``n_mag`` is held fixed (changing it is an arch
    change, not a grid refit). ``tick_hz`` is stamped into the block; when None it
    is read from the collect's metadata (the orchestrator passes it to skip that read)."""
    collect_dir = Path(collect_dir)
    files = sorted(glob.glob(str(collect_dir / "precomputed_train" / "shard*_act_look.npy")))
    if not files:
        raise FileNotFoundError(f"no act_look shards under {collect_dir}/precomputed_train")
    # Point-mass hold: exclude only the true-stillness mass below the source
    # floor from the fit; every real drift is fittable mass (7/06 scheme).
    hold_max = SOURCE_FLOOR_RAD
    counts = np.zeros(HIST_NBINS, dtype=np.int64)
    n_frames = 0
    hold_n = 0
    for f in files:
        theta = theta_from_look(np.load(f).astype(np.float32))
        counts += histogram_theta(theta)
        n_frames += theta.size
        hold_n += int((theta < hold_max).sum())
    fitted = fit_mag_centers(counts, hold_max, n_mag=n_mag, seed=seed)
    default = default_mag_centers(n_mag, fovea_power)
    if tick_hz is None:
        meta_path = collect_dir / "collect_metadata.json"
        if meta_path.exists():
            tick_hz = json.loads(meta_path.read_text()).get("tick_hz")
    return {
        "schema": "look_grid_v1",
        "tick_hz": tick_hz,
        "n_frames": int(n_frames),
        "n_mag": int(n_mag),
        "seed": int(seed),
        "hold_max_rad": float(hold_max),
        "deadzone_rad": float(SOURCE_FLOOR_RAD),
        "hold_frac": float(hold_n / n_frames) if n_frames else 0.0,
        "theta_hist": {"nbins": HIST_NBINS, "max_deg": float(np.degrees(THETA_MAX)),
                       "counts": counts.tolist()},
        "fit": {
            "method": "lloyd_max_weighted",
            "mag_centers_rad": fitted.tolist(),
            "rms_deg": rms_distortion_deg(counts, fitted, hold_max),
        },
        "default": {
            "fovea_power": float(fovea_power),
            "mag_centers_rad": default.tolist(),
            "rms_deg": rms_distortion_deg(counts, default, hold_max),
        },
    }


def export_default_grid(run_dir: str | Path) -> Path:
    """Materialize the historical code-default polar grid into an OLD run's
    config/look_grid.json (source "code_default").

    Models trained before data-driven grids ran on the hardcoded power-law centers.
    Now that there is no runtime default (qnn.model.look_bins), those models must
    carry their grid explicitly like every new run. This writes the exact legacy
    grid so the run goes through the normal install_polar_grid path. Idempotent;
    refuses to overwrite a corpus_fit grid (only fills a missing/code_default one)."""
    run_dir = Path(run_dir)
    out = run_dir / "config" / "look_grid.json"
    if out.exists():
        existing = json.loads(out.read_text())
        if existing.get("source") == "corpus_fit":
            raise ValueError(f"{out} is a corpus_fit grid — refusing to overwrite with the default")
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = code_default_grid()
    grid["materialized_for"] = run_dir.name
    out.write_text(json.dumps(grid, indent=2) + "\n")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Look-grid tools (fit / drift / legacy export).")
    ap.add_argument("--export-default", type=Path, metavar="RUN_DIR",
                    help="write the historical code-default grid into RUN_DIR/config/look_grid.json")
    args = ap.parse_args()
    if args.export_default:
        path = export_default_grid(args.export_default)
        print(f"wrote code-default look grid -> {path}")
    else:
        ap.error("nothing to do (try --export-default <run_dir>)")


if __name__ == "__main__":
    main()
