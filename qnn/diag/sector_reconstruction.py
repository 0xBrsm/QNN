"""Score the v1 9-sector spatial representation as a depth codec.

Tier 0 of the spatial-substrate three-way (v1 rays vs rev-8 atlas vs
rev-10 probe-grid): how much of the dense oracle geometry does the v1
sector summary preserve, on the SAME query set and instrument as the
rev-8 atlas gate (qnn.diag.spatial_reconstruction) and the probe-grid
feasibility study (qnn.diag.probe_reconstruction)?

The v1 representation (qnn_spatial.c on main, pre-atlas) sampled 5 rays
per sector across 7 horizontal sectors (yaw 0/±40/±90/±150°, spans
40/40/30°, elevation 0 only, 1024u) plus vertical ground/ceiling probes
(128u), and kept only per-sector aggregates. This module simulates that
geometry on each record's oracle ``truth`` field — the same dense hull-1
trace grid the atlas is scored against — then reconstructs the full
(elevation, yaw) depth field from the sector aggregates alone:

- covered horizontal cells predict their sector's aggregate (two arms:
  ``sector_mean`` and ``sector_nearest``);
- the ∓75° bands predict a flat-floor/ceiling projection of the
  ground/ceiling probe (dist / sin 75°, capped at the band limit);
- everything else (bands ±15..±60°, uncovered yaw on the 0° band)
  predicts the band limit — the "open" prior, mirroring atlas miss
  semantics.

Simulating from the oracle field scores the representation's *geometry*
(sector granularity + coverage) under a best-case instrument; the real
v1 traced hull 0 per tick, so live v1 could only be worse. Metrics are
``spatial_reconstruction._metrics`` verbatim — directly comparable with
the gate table in wire.12.md — reported for the full grid and for the
covered subset.

Usage:
    python -m qnn.diag.sector_reconstruction <sidecar.jsonl ...> \
        [--output out.json]

Consumes the same QNN_SPATIAL_DIAG sidecars as the gate (schema >= 6).
The atlas layouts are scored on the same records alongside for a
side-by-side table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qnn.diag.spatial_reconstruction import (
    _metrics, _record_limits, load_records, score_records,
)
from qnn.utils.io import write_json

# v1 horizontal sector table: (name, yaw center°, span°). Rays at
# center + (i-2)·span/4 for i in 0..4 (QNN_BuildHorizontalSpatial).
_H_SECTORS: tuple[tuple[str, float, float], ...] = (
    ("fov_center",    0.0, 40.0),
    ("fov_left",     40.0, 40.0),
    ("fov_right",   -40.0, 40.0),
    ("flank_left",   90.0, 40.0),
    ("flank_right", -90.0, 40.0),
    ("rear_left",   150.0, 30.0),
    ("rear_right", -150.0, 30.0),
)
_V1_HORIZ_MAX = 1024.0
_V1_VERT_MAX = 128.0
ARMS = ("sector_mean", "sector_nearest")


def _ray_yaws(center: float, span: float) -> list[float]:
    return [center + (i - 2.0) * (span / 4.0) for i in range(5)]


def _covered_yaws(center: float, span: float, yaw_step: int) -> list[int]:
    """Yaw cells whose centers fall inside the sector span."""
    lo = center - span / 2.0
    hi = center + span / 2.0
    return sorted({
        int(round(deg)) % 360 // yaw_step
        for deg in np.arange(lo, hi + 1e-6, yaw_step)
    })


def reconstruct_sectors(
    record: dict[str, Any], *, arm: str = "sector_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """(prediction, covered_mask) on the record's (elevation, yaw) grid.

    Sector aggregates are computed from the record's oracle ``truth``
    field along v1's exact ray directions (best-case instrument).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; allowed: {ARMS}")
    elevations, yaw_count, limits = _record_limits(record)
    yaw_step = int(record["yaw_step"])
    truth = np.asarray(record["truth"], dtype=np.float64).reshape(
        len(elevations), yaw_count,
    )
    prediction = limits.copy()  # uncovered → open prior (band limit)
    covered = np.zeros_like(prediction, dtype=bool)

    def _band(elevation: float) -> int | None:
        for i, value in enumerate(elevations):
            if value == elevation:
                return i
        return None

    zero = _band(0.0)
    if zero is not None:
        for _, center, span in _H_SECTORS:
            rays = [
                min(truth[zero, int(round(yaw)) % 360 // yaw_step],
                    _V1_HORIZ_MAX)
                for yaw in _ray_yaws(center, span)
            ]
            value = float(np.mean(rays)) if arm == "sector_mean" \
                else float(np.min(rays))
            cells = _covered_yaws(center, span, yaw_step)
            prediction[zero, cells] = np.minimum(value, limits[zero, cells])
            covered[zero, cells] = True

    # Ground/ceiling: v1 probed straight down/up to 128u. Project a
    # flat-floor/ceiling assumption onto the steepest bands.
    sin75 = math.sin(math.radians(75.0))
    for elevation in (-75.0, 75.0):
        band = _band(elevation)
        if band is None:
            continue
        # v1's five bbox probes ~ the oracle's vertical neighborhood; the
        # closest grid analog of "straight down" is the band's own cells,
        # so probe depth = min over the band, capped at the v1 range.
        probe = min(float(truth[band].min()) * sin75, _V1_VERT_MAX)
        slant = min(probe / sin75, float(limits[band, 0]))
        prediction[band, :] = slant
        covered[band, :] = True
    return prediction, covered


def score_sector_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError("no spatial diag records")
    out: dict[str, Any] = {"records": len(records), "arms": {}}
    for arm in ARMS:
        truths, planes, limits_all, covers = [], [], [], []
        for record in records:
            elevations, yaw_count, limits = _record_limits(record)
            truth = np.asarray(record["truth"], dtype=np.float64).reshape(
                len(elevations), yaw_count,
            )
            prediction, covered = reconstruct_sectors(record, arm=arm)
            truths.append(truth.reshape(-1))
            planes.append(prediction.reshape(-1))
            limits_all.append(limits.reshape(-1))
            covers.append(covered.reshape(-1))
        truth_all = np.concatenate(truths)
        plane_all = np.concatenate(planes)
        limit_all = np.concatenate(limits_all)
        cover_all = np.concatenate(covers)
        out["arms"][arm] = {
            "coverage": float(np.mean(cover_all)),
            "full_grid": _metrics(truth_all, plane_all, limit_all),
            "covered_only": _metrics(
                truth_all[cover_all], plane_all[cover_all],
                limit_all[cover_all],
            ),
        }
    return out


def _row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {m['mae']:.2f} | {100 * m['missed_obstacle_rate']:.2f} "
        f"| {100 * m['false_block_rate_all']:.3f} "
        f"| {100 * m['blocked_early_gt_32_rate']:.2f} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecars", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    records: list[dict[str, Any]] = []
    for path in args.sidecars:
        records.extend(load_records(path))

    summary: dict[str, Any] = {
        "sidecars": [str(p) for p in args.sidecars],
        "atlas": score_records(records, layout="atlas")["plane"],
        "atlas_float": score_records(records, layout="atlas_float")["plane"],
        "sectors": score_sector_records(records),
    }

    lines = [
        "| layout | MAE (u) | missed % | false-block(all) % | early>32u % |",
        "|---|---:|---:|---:|---:|",
        _row("atlas (4-bit codes)", summary["atlas"]),
        _row("atlas_float (no quant)", summary["atlas_float"]),
    ]
    for arm, block in summary["sectors"]["arms"].items():
        lines.append(_row(f"v1 {arm} (full grid)", block["full_grid"]))
        lines.append(_row(
            f"v1 {arm} (covered {100 * block['coverage']:.1f}%)",
            block["covered_only"],
        ))
    table = "\n".join(lines)
    print(table)
    if args.output is not None:
        write_json(args.output, summary)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
