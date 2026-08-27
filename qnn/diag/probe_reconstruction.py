"""Price the probe-grid atlas (rev 10) before any model or wire work.

The rev-10 direction precomputes the depth atlas at fixed map probes and
supplies nearby probes to the model, projected to the agent's frame.  The
open question is parallax: depth observed from a probe is not depth from
the agent.  This study answers "how dense must the probe grid be?" with
the same instrument and thresholds as the rev-8 gate.

Protocol (leave-one-out over dense schema-7 demo-frame sidecars, which
carry ``origin``/``view_yaw`` per record):

1. Poisson-disk subsample record positions at a target spacing — the
   kept records play the role of the load-time probe grid.
2. Every remaining record is a target: reconstruct its ego panorama from
   its K nearest probes and score against the record's own dense truth
   field with the gate metrics.

Arms:

- ``reproject``: donor hit cells become world points, scattered into the
  target's (elevation, yaw) ray bins, min-depth per bin — the
  information-availability bound for a model given K probes + offsets.
- ``shift``: single nearest probe, yaw-axis circular shift only, no
  parallax correction — the naive-lookup floor.
- ``ego``: the record's own wire payload (the per-tick atlas the probe
  grid would replace) — the reference ceiling.

Donor open cells are not scattered; uncovered target cells default to
open, so lost coverage surfaces as missed obstacles, never as silent
optimism.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from qnn.diag.spatial_reconstruction import (
    _metrics,
    _record_limits,
    load_records,
    reconstruct_record,
)
from qnn.utils.io import write_json

ARMS = ("reproject", "shift", "shift_corrected", "hybrid", "hybrid_corrected", "ego")


def _require_pose(record: dict[str, Any]) -> None:
    if "origin" not in record or "view_yaw" not in record:
        raise ValueError(
            "record lacks origin/view_yaw — probe studies need schema>=7 sidecars"
        )


def _hit_mask(record: dict[str, Any], layout: str) -> np.ndarray:
    if layout == "atlas":
        from qnn.engine_norm import ATLAS_MISS_CODE

        return np.asarray(record["atlas_code"], dtype=np.int64) != ATLAS_MISS_CODE
    return np.asarray(record["atlas_distance"], dtype=np.float64) >= 0.0


def _cell_directions(
    elevations: list[float], yaw_count: int, view_yaw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame unit vectors for every (elevation, yaw) cell."""
    yaw_step = 360.0 / yaw_count
    yaw_deg = view_yaw + np.arange(yaw_count) * yaw_step
    yaw_rad = np.radians(yaw_deg)[None, :]
    elev_rad = np.radians(np.asarray(elevations, dtype=np.float64))[:, None]
    cz = np.cos(elev_rad)
    return (
        np.cos(yaw_rad) * cz,
        np.sin(yaw_rad) * cz,
        np.broadcast_to(np.sin(elev_rad), (len(elevations), yaw_count)).copy(),
    )


def _assign_bands(
    point_elev_deg: np.ndarray, elevations: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Map point elevations to wire band rows; mask points outside every
    band's acceptance half-width."""
    centers = np.asarray(elevations, dtype=np.float64)
    order = np.argsort(centers)
    sorted_centers = centers[order]
    gaps = np.diff(sorted_centers)
    half_sorted = np.empty_like(sorted_centers)
    half_sorted[0] = gaps[0] / 2.0
    half_sorted[-1] = gaps[-1] / 2.0
    if len(gaps) > 1:
        half_sorted[1:-1] = np.minimum(gaps[:-1], gaps[1:]) / 2.0

    idx_sorted = np.abs(point_elev_deg[:, None] - sorted_centers[None, :]).argmin(axis=1)
    accept = (
        np.abs(point_elev_deg - sorted_centers[idx_sorted]) <= half_sorted[idx_sorted]
    )
    return order[idx_sorted], accept


def _donor_world_points(
    record: dict[str, Any], layout: str
) -> np.ndarray:
    """Hit cells of one donor record as an (N, 3) world-point cloud."""
    _require_pose(record)
    elevations, yaw_count, _ = _record_limits(record)
    depth = reconstruct_record(record, layout=layout)
    hit = _hit_mask(record, layout)
    dx, dy, dz = _cell_directions(elevations, yaw_count, float(record["view_yaw"]))
    origin = np.asarray(record["origin"], dtype=np.float64)
    return origin[None, :] + (
        np.stack([dx[hit], dy[hit], dz[hit]], axis=1) * depth[hit][:, None]
    )


def reconstruct_from_probes(
    target: dict[str, Any], donors: list[dict[str, Any]], *, layout: str = "atlas",
    return_coverage: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Scatter donor hit points into the target's ray bins, min per bin.

    Coverage is structurally poor in the steep bands: the floor/ceiling
    near the target subtends a huge solid angle from the target but a
    tiny one from any donor, so few donor rays land there (measured
    6-36%% coverage at |elev| >= 45 deg vs ~100%% at level).  Callers
    wanting a usable panorama should fall back where uncovered — see the
    ``hybrid`` arm."""
    _require_pose(target)
    elevations, yaw_count, limits = _record_limits(target)
    yaw_step = 360.0 / yaw_count
    prediction = limits.copy()
    covered = np.zeros(limits.shape, dtype=bool)

    points = (
        np.concatenate([_donor_world_points(d, layout) for d in donors])
        if donors else np.empty((0, 3))
    )
    if points.size:
        rel = points - np.asarray(target["origin"], dtype=np.float64)[None, :]
        dist = np.linalg.norm(rel, axis=1)
        keep = dist > 1e-6
        rel, dist = rel[keep], dist[keep]
        elev_deg = np.degrees(np.arcsin(np.clip(rel[:, 2] / dist, -1.0, 1.0)))
        band, accept = _assign_bands(elev_deg, elevations)
        yaw_rel = (
            np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) - float(target["view_yaw"])
        ) % 360.0
        cell = np.rint(yaw_rel / yaw_step).astype(np.int64) % yaw_count
        within = accept & (dist <= limits[band, cell])
        np.minimum.at(prediction, (band[within], cell[within]), dist[within])
        covered[band[within], cell[within]] = True
    if return_coverage:
        return prediction, covered
    return prediction


def reconstruct_shift(
    target: dict[str, Any], donor: dict[str, Any], *, layout: str = "atlas",
    corrected: bool = False,
) -> np.ndarray:
    """Nearest-probe panorama, yaw-shifted to the target's view frame.

    With ``corrected``, hit cells get the first-order parallax fix
    ``d_target = d_donor + (probe - agent) . ray`` — exact wherever the
    surface is locally flat across the offset, and computable by the
    model from the relative-offset encoding alone.  Open cells stay
    open: inventing a hit from a correction would manufacture false
    blocks out of misses."""
    _require_pose(target)
    _require_pose(donor)
    elevations, yaw_count, limits = _record_limits(target)
    yaw_step = 360.0 / yaw_count
    delta = int(round(
        ((float(target["view_yaw"]) - float(donor["view_yaw"])) % 360.0) / yaw_step
    )) % yaw_count
    rolled = np.roll(reconstruct_record(donor, layout=layout), -delta, axis=1)
    if corrected:
        rolled_hit = np.roll(_hit_mask(donor, layout), -delta, axis=1)
        dx, dy, dz = _cell_directions(
            elevations, yaw_count, float(target["view_yaw"])
        )
        offset = (
            np.asarray(donor["origin"], dtype=np.float64)
            - np.asarray(target["origin"], dtype=np.float64)
        )
        along = offset[0] * dx + offset[1] * dy + offset[2] * dz
        rolled = np.where(
            rolled_hit, np.maximum(rolled + along, 0.0), rolled
        )
    return np.minimum(rolled, limits)


def poisson_subsample(
    positions: np.ndarray, spacing: float, rng: np.random.Generator
) -> np.ndarray:
    """Greedy Poisson-disk thinning: indices of a probe set with pairwise
    distance >= spacing, in random visit order."""
    order = rng.permutation(len(positions))
    kept: list[int] = []
    kept_pos = np.empty((0, 3))
    for i in order:
        p = positions[i]
        if kept and np.min(np.linalg.norm(kept_pos - p[None, :], axis=1)) < spacing:
            continue
        kept.append(int(i))
        kept_pos = np.vstack([kept_pos, p[None, :]])
    return np.asarray(kept, dtype=np.int64)


def run_split(
    records: list[dict[str, Any]], *, spacing: float, k: int, seed: int,
    layout: str, arms: tuple[str, ...],
) -> dict[str, Any]:
    """One map, one spacing, one probe-thinning seed → per-arm arrays."""
    positions = np.asarray([r["origin"] for r in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    probe_idx = poisson_subsample(positions, spacing, rng)
    probe_set = set(probe_idx.tolist())
    target_idx = [i for i in range(len(records)) if i not in probe_set]
    if not target_idx:
        raise ValueError(f"spacing {spacing}: every record became a probe")

    probe_pos = positions[probe_idx]
    truths, limits_all = [], []
    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    nearest_dists = []
    for i in target_idx:
        target = records[i]
        elevations, yaw_count, limits = _record_limits(target)
        truth = np.asarray(target["truth"], dtype=np.float64).reshape(
            len(elevations), yaw_count
        )
        dists = np.linalg.norm(probe_pos - positions[i][None, :], axis=1)
        near = probe_idx[np.argsort(dists)[:k]]
        nearest_dists.append(float(dists.min()))
        truths.append(truth.reshape(-1))
        limits_all.append(limits.reshape(-1))
        for arm in arms:
            if arm == "reproject":
                pred = reconstruct_from_probes(
                    target, [records[int(j)] for j in near], layout=layout
                )
            elif arm in ("shift", "shift_corrected"):
                pred = reconstruct_shift(
                    target, records[int(near[0])], layout=layout,
                    corrected=arm == "shift_corrected",
                )
            elif arm in ("hybrid", "hybrid_corrected"):
                reproj, covered = reconstruct_from_probes(
                    target, [records[int(j)] for j in near], layout=layout,
                    return_coverage=True,
                )
                pred = np.where(
                    covered, reproj,
                    reconstruct_shift(
                        target, records[int(near[0])], layout=layout,
                        corrected=arm == "hybrid_corrected",
                    ),
                )
            elif arm == "ego":
                pred = reconstruct_record(target, layout=layout)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            predictions[arm].append(pred.reshape(-1))

    return {
        "probes": len(probe_idx),
        "targets": len(target_idx),
        "nearest_probe_mean": float(np.mean(nearest_dists)),
        "nearest_probe_p90": float(np.percentile(nearest_dists, 90)),
        "truth": np.concatenate(truths),
        "limits": np.concatenate(limits_all),
        "predictions": {arm: np.concatenate(p) for arm, p in predictions.items()},
    }


def run_sweep(
    sidecars: list[Path], *, spacings: list[float], k: int, seeds: list[int],
    layout: str, arms: tuple[str, ...],
) -> dict[str, Any]:
    by_map = {path.stem: load_records(path) for path in sidecars}
    for name, records in by_map.items():
        for record in records:
            _require_pose(record)
        print(f"{name}: {len(records)} records")

    out: dict[str, Any] = {"layout": layout, "k": k, "seeds": seeds, "spacings": {}}
    for spacing in spacings:
        pooled_truth, pooled_limits = [], []
        pooled_pred: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
        meta = []
        for name, records in by_map.items():
            for seed in seeds:
                split = run_split(
                    records, spacing=spacing, k=k, seed=seed,
                    layout=layout, arms=arms,
                )
                meta.append({
                    "map": name, "seed": seed, "probes": split["probes"],
                    "targets": split["targets"],
                    "nearest_probe_mean": split["nearest_probe_mean"],
                    "nearest_probe_p90": split["nearest_probe_p90"],
                })
                pooled_truth.append(split["truth"])
                pooled_limits.append(split["limits"])
                for arm in arms:
                    pooled_pred[arm].append(split["predictions"][arm])
        truth = np.concatenate(pooled_truth)
        limits = np.concatenate(pooled_limits)
        out["spacings"][f"{spacing:g}"] = {
            "splits": meta,
            "arms": {
                arm: _metrics(truth, np.concatenate(pooled_pred[arm]), limits)
                for arm in arms
            },
        }
        print(f"spacing {spacing:g}: scored {len(meta)} splits")
    return out


def _format_table(summary: dict[str, Any]) -> str:
    lines = [
        "| spacing | arm | MAE | missed | false blocks (all) | early >32u |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for spacing, block in summary["spacings"].items():
        for arm, m in block["arms"].items():
            lines.append(
                f"| {spacing} | {arm} | {m['mae']:.2f} "
                f"| {100 * m['missed_obstacle_rate']:.2f}% "
                f"| {100 * m['false_block_rate_all']:.2f}% "
                f"| {100 * m['blocked_early_gt_32_rate']:.2f}% |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecars", type=Path, nargs="+")
    parser.add_argument("--spacings", default="32,48,64,96,128")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seeds", default="17,18,19")
    parser.add_argument("--layout", choices=("atlas", "atlas_float"), default="atlas")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    summary = run_sweep(
        args.sidecars,
        spacings=[float(s) for s in args.spacings.split(",")],
        k=args.k,
        seeds=[int(s) for s in args.seeds.split(",")],
        layout=args.layout,
        arms=tuple(args.arms.split(",")),
    )
    if args.output is not None:
        write_json(args.output, summary)
    print(_format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
