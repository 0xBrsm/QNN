"""Near-field substrate study: can a precomputed field replace the steep bands?

The probe-grid substrate's one residual is near-field/vertical geometry
(steep elevation bands starve from any probe — feasibility study, the
seed-robust jump gap, the dm4 edge-falls). Candidate fix: represent the
near field in POSITION space, where it is viewpoint-free — a layered
floor/ceiling height field precomputed at map load — instead of forcing
it through view-space ray bins.

This module prices that design with the gate instrument, before any
model work (the same move as the probe feasibility study): build a
leave-one-out layered height field from the OTHER records' steep-band
oracle hits (world-anchored via schema-7 pose), reconstruct each
held-out record's steep-band depths by marching its rays against the
field, and score with ``spatial_reconstruction._metrics`` against the
record's own dense truth — restricted to the steep bands, beside the
production atlas codes (the ceiling reference) on the same subset.

Leakage guard: a record never reads samples from its own file within
±``exclusion_s`` seconds (demo paths revisit poses; matches the probe
study's ±100-row rule at 20 Hz).

Height field: 2D grid, ``cell`` units, each column holding up to
``MAX_LEVELS`` distinct floor and ceiling z-levels (Quake maps are
multi-story; a single-level height field would be wrong). Query
selects the level relative to the ray's current z — side surfaces
(walls) are invisible to a height field by construction; the shallow
bands keep those. That approximation is exactly what this study prices.

Usage:
    python -m qnn.diag.nearfield_reconstruction <sidecar.jsonl ...> \
        --cells 8,16,32 --steep-min-deg 45 --output out.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from qnn.diag.spatial_reconstruction import (
    _metrics, _record_limits, load_records, reconstruct_record,
)
from qnn.utils.io import write_json

MAX_LEVELS = 4
_LEVEL_MERGE_U = 24.0   # samples closer than this merge into one level
_MARCH_STEP_U = 2.0
_EYE_EPS_U = 2.0


def _pack_cell(cx, cy):
    """Pack integer cell coords into one int64 key (vectorized-friendly)."""
    return (np.int64(cx) << np.int64(20)) ^ (np.int64(cy) & np.int64(0xFFFFF))


def _steep_band_indices(elevations: list[float], min_deg: float) -> list[int]:
    return [i for i, e in enumerate(elevations) if abs(e) >= min_deg]


def _world_hits(record: dict[str, Any], band_idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(floor_pts (n,3), ceil_pts (n,3)) — world-space oracle hit points."""
    elevations, yaw_count, limits = _record_limits(record)
    truth = np.asarray(record["truth"], dtype=np.float64).reshape(len(elevations), yaw_count)
    origin = np.asarray(record["origin"], dtype=np.float64)
    view_yaw = float(record["view_yaw"])
    yaw_step = 360.0 / yaw_count
    floor_pts, ceil_pts = [], []
    for bi in band_idxs:
        e = math.radians(elevations[bi])
        hits = truth[bi] < limits[bi] - 1.0
        if not hits.any():
            continue
        yaws = np.radians(view_yaw + yaw_step * np.arange(yaw_count)[hits])
        d = truth[bi][hits]
        pts = origin[None, :] + np.stack([
            d * math.cos(e) * np.cos(yaws),
            d * math.cos(e) * np.sin(yaws),
            d * math.sin(e) * np.ones_like(yaws),
        ], axis=1)
        (floor_pts if elevations[bi] < 0 else ceil_pts).append(pts)
    empty = np.zeros((0, 3))
    return (
        np.concatenate(floor_pts) if floor_pts else empty,
        np.concatenate(ceil_pts) if ceil_pts else empty,
    )


class LayeredField:
    """Per-column multi-level z samples on a 2D grid."""

    def __init__(self, cell: float, is_floor: bool) -> None:
        self.cell = float(cell)
        self.is_floor = is_floor
        self._cols: dict[tuple[int, int], list[float]] = defaultdict(list)

    def add(self, pts: np.ndarray) -> None:
        if not len(pts):
            return
        cx = np.floor(pts[:, 0] / self.cell).astype(np.int64)
        cy = np.floor(pts[:, 1] / self.cell).astype(np.int64)
        for x, y, z in zip(cx, cy, pts[:, 2]):
            self._cols[(int(x), int(y))].append(float(z))

    def finalize(self) -> None:
        """Cluster each column's samples into <= MAX_LEVELS z-levels."""
        for key, zs in self._cols.items():
            zs.sort()
            levels: list[list[float]] = [[zs[0]]]
            for z in zs[1:]:
                if z - levels[-1][-1] <= _LEVEL_MERGE_U:
                    levels[-1].append(z)
                else:
                    levels.append([z])
            # Floor surface = top of each cluster; ceiling = bottom.
            reps = [max(c) if self.is_floor else min(c) for c in levels]
            reps = reps[-MAX_LEVELS:] if self.is_floor else reps[:MAX_LEVELS]
            self._cols[key] = reps

    def to_arrays(self) -> tuple[dict[int, int], np.ndarray]:
        """Column index keyed by packed cell id ((cx << 32) ^ (cy & mask))."""
        index = {_pack_cell(kx, ky): i for i, (kx, ky) in enumerate(self._cols)}
        table = np.full((len(index), MAX_LEVELS), np.nan)
        for (kx, ky), reps in self._cols.items():
            table[index[_pack_cell(kx, ky)], : len(reps)] = reps
        return index, table


def _march(
    origin: np.ndarray, elev_deg: float, yaws_deg: np.ndarray, limit: float,
    index: dict, table: np.ndarray, cell: float, is_floor: bool,
) -> np.ndarray:
    """Predicted depth per yaw ray against a layered field (vectorized)."""
    n = len(yaws_deg)
    steps = max(2, int(math.ceil(limit / _MARCH_STEP_U)))
    t = np.linspace(0.0, limit, steps)                      # (T,)
    e = math.radians(elev_deg)
    yr = np.radians(yaws_deg)                               # (n,)
    x = origin[0] + np.outer(np.cos(yr), t) * math.cos(e)   # (n, T)
    y = origin[1] + np.outer(np.sin(yr), t) * math.cos(e)
    z = origin[2] + t[None, :] * math.sin(e)                # broadcast (n, T)
    z = np.broadcast_to(z, x.shape).copy()

    cx = np.floor(x / cell).astype(np.int64)
    cy = np.floor(y / cell).astype(np.int64)
    keys = _pack_cell(cx, cy)
    uniq, inv = np.unique(keys, return_inverse=True)
    col_of_uniq = np.asarray([index.get(int(k), -1) for k in uniq], dtype=np.int64)
    col = col_of_uniq[inv].reshape(x.shape)
    have = col >= 0
    levels = np.full((*x.shape, MAX_LEVELS), np.nan)
    levels[have] = table[col[have]]

    # Reference z for level selection: previous step's ray z (start: eye).
    z_prev = np.concatenate([z[:, :1] + _EYE_EPS_U, z[:, :-1]], axis=1)
    if is_floor:
        elig = levels <= (z_prev[..., None] + _EYE_EPS_U)
        cand = np.where(elig, levels, -np.inf).max(axis=-1)   # (n, T)
        crossed = z < cand
    else:
        elig = levels >= (z_prev[..., None] - _EYE_EPS_U)
        cand = np.where(elig, levels, np.inf).min(axis=-1)
        crossed = z > cand
    crossed &= np.isfinite(cand)

    out = np.full(n, limit)
    any_hit = crossed.any(axis=1)
    first = np.argmax(crossed, axis=1)
    for i in np.where(any_hit)[0]:
        j = int(first[i])
        if j == 0:
            out[i] = 0.0
            continue
        z0, z1, c = z[i, j - 1], z[i, j], cand[i, j]
        frac = (z0 - c) / (z0 - z1) if z0 != z1 else 1.0
        out[i] = t[j - 1] + float(np.clip(frac, 0.0, 1.0)) * (t[1] - t[0])
    return out


def run_study(
    records_by_file: dict[str, list[dict]], *, cell: float,
    steep_min_deg: float, exclusion_s: float,
) -> dict[str, Any]:
    # Group by map: fields are per-map.
    by_map: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for fname, records in records_by_file.items():
        for r in records:
            by_map[r["map"]].append((fname, r))

    truth_all, pred_hf, pred_atlas, limit_all = [], [], [], []
    total_queries = 0
    for map_name, tagged in by_map.items():
        # Per-record world hits, indexed so leave-one-out is cheap.
        hits = [_world_hits(r, _steep_band_indices(
            [float(v) for v in r["elevations"]], steep_min_deg)) for _, r in tagged]
        times = np.asarray([float(r.get("time", i)) for i, (_, r) in enumerate(tagged)])
        files = [f for f, _ in tagged]

        for qi, (fname, rec) in enumerate(tagged):
            elevations, yaw_count, limits = _record_limits(rec)
            band_idxs = _steep_band_indices(elevations, steep_min_deg)
            truth = np.asarray(rec["truth"], dtype=np.float64).reshape(
                len(elevations), yaw_count)
            atlas = reconstruct_record(rec, layout="atlas")

            floor_f = LayeredField(cell, is_floor=True)
            ceil_f = LayeredField(cell, is_floor=False)
            for di, (fpts, cpts) in enumerate(hits):
                if di == qi:
                    continue
                if files[di] == fname and abs(times[di] - times[qi]) < exclusion_s:
                    continue
                floor_f.add(fpts)
                ceil_f.add(cpts)
            floor_f.finalize()
            ceil_f.finalize()
            f_index, f_table = floor_f.to_arrays()
            c_index, c_table = ceil_f.to_arrays()

            origin = np.asarray(rec["origin"], dtype=np.float64)
            view_yaw = float(rec["view_yaw"])
            yaw_step = 360.0 / yaw_count
            yaws = view_yaw + yaw_step * np.arange(yaw_count)
            for bi in band_idxs:
                limit = float(limits[bi, 0])
                if elevations[bi] < 0:
                    pred = _march(origin, elevations[bi], yaws, limit,
                                  f_index, f_table, cell, is_floor=True)
                else:
                    pred = _march(origin, elevations[bi], yaws, limit,
                                  c_index, c_table, cell, is_floor=False)
                truth_all.append(truth[bi])
                pred_hf.append(pred)
                pred_atlas.append(atlas[bi])
                limit_all.append(limits[bi])
                total_queries += yaw_count

    truth_c = np.concatenate(truth_all)
    limit_c = np.concatenate(limit_all)
    return {
        "cell": cell,
        "steep_min_deg": steep_min_deg,
        "queries": int(total_queries),
        "atlas_steep": _metrics(truth_c, np.concatenate(pred_atlas), limit_c),
        "heightfield_steep": _metrics(truth_c, np.concatenate(pred_hf), limit_c),
    }


# Quake player hull 1: mins_z=-24, maxs_z=+32. The navmesh stores SURFACE
# z; the truth field is hull-1 clearance in CONFIG space (player-origin
# reachability), so mesh surfaces convert exactly: config floor = surface
# + 24 (origin rests 24 above the floor), config ceiling = surface - 32
# (origin can rise to 32 below the ceiling).
_HULL_FLOOR_OFF = 24.0
_HULL_CEIL_OFF = 32.0


def build_mesh_fields(
    polys: list[dict[str, Any]], cell: float,
) -> tuple[dict, np.ndarray, dict, np.ndarray]:
    """Navmesh poly dump → config-space layered floor + ceiling fields.

    Each poly's AABB footprint rasterizes into grid columns at its
    center z (+ hull offsets); the per-poly upward-trace ceiling
    annotation supplies the ceiling level over the same footprint.
    """
    floor_f = LayeredField(cell, is_floor=True)
    ceil_f = LayeredField(cell, is_floor=False)
    for p in polys:
        bmin, bmax = p["bmin"], p["bmax"]
        z = float(p["center"][2])
        xs = np.arange(bmin[0], bmax[0] + cell, cell)
        ys = np.arange(bmin[1], bmax[1] + cell, cell)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z + _HULL_FLOOR_OFF)], axis=1)
        floor_f.add(pts)
        ceiling = float(p.get("ceiling", 2048.0))
        if ceiling < 2048.0:
            cpts = pts.copy()
            cpts[:, 2] = z + ceiling + 2.0 - _HULL_CEIL_OFF
            ceil_f.add(cpts)
    floor_f.finalize()
    ceil_f.finalize()
    return (*floor_f.to_arrays(), *ceil_f.to_arrays())


def run_mesh_study(
    records_by_file: dict[str, list[dict]],
    polys_by_map: dict[str, list[dict]], *, cell: float, steep_min_deg: float,
) -> dict[str, Any]:
    """Score the navmesh-derived fields on the steep-band query set."""
    fields = {m: build_mesh_fields(p, cell) for m, p in polys_by_map.items()}
    truth_all, pred_mesh, limit_all = [], [], []
    floor_q = ceil_q = 0
    for records in records_by_file.values():
        for rec in records:
            if rec["map"] not in fields:
                continue
            f_index, f_table, c_index, c_table = fields[rec["map"]]
            elevations, yaw_count, limits = _record_limits(rec)
            band_idxs = _steep_band_indices(elevations, steep_min_deg)
            truth = np.asarray(rec["truth"], dtype=np.float64).reshape(
                len(elevations), yaw_count)
            origin = np.asarray(rec["origin"], dtype=np.float64)
            yaws = float(rec["view_yaw"]) + (360.0 / yaw_count) * np.arange(yaw_count)
            for bi in band_idxs:
                limit = float(limits[bi, 0])
                if elevations[bi] < 0:
                    pred = _march(origin, elevations[bi], yaws, limit,
                                  f_index, f_table, cell, is_floor=True)
                    floor_q += yaw_count
                else:
                    pred = _march(origin, elevations[bi], yaws, limit,
                                  c_index, c_table, cell, is_floor=False)
                    ceil_q += yaw_count
                truth_all.append(truth[bi])
                pred_mesh.append(pred)
                limit_all.append(limits[bi])
    return {
        "cell": cell,
        "floor_queries": floor_q,
        "ceiling_queries": ceil_q,
        "mesh_steep": _metrics(
            np.concatenate(truth_all), np.concatenate(pred_mesh),
            np.concatenate(limit_all)),
    }


def _row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {m['mae']:.2f} | {100 * m['missed_obstacle_rate']:.2f} "
        f"| {100 * m['false_block_rate_all']:.3f} "
        f"| {100 * m['blocked_early_gt_32_rate']:.2f} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecars", nargs="+", type=Path)
    parser.add_argument("--cells", default="8,16,32")
    parser.add_argument("--steep-min-deg", type=float, default=45.0)
    parser.add_argument("--exclusion-s", type=float, default=5.0)
    parser.add_argument(
        "--nav-polys", type=Path, nargs="*", default=None,
        help="navmesh poly dumps named navpolys_<map>.json (nav_query "
             "kind=polys); adds the mesh arm",
    )
    parser.add_argument(
        "--skip-corpus-field", action="store_true",
        help="skip the corpus-ray heightfield arm (coverage-limited proxy)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    records_by_file = {str(p): load_records(p) for p in args.sidecars}
    n = sum(len(v) for v in records_by_file.values())
    print(f"{n} records from {len(records_by_file)} sidecars")

    polys_by_map: dict[str, list[dict]] = {}
    if args.nav_polys:
        for p in args.nav_polys:
            map_name = p.stem.replace("navpolys_", "")
            polys_by_map[map_name] = json.loads(p.read_text())["polys"]
        # Poly dumps key by bare map name; records carry e.g. maps/dm2.bsp.
        sample = next(iter(records_by_file.values()))[0]["map"]
        if "/" in sample or sample.endswith(".bsp"):
            polys_by_map = {
                f"maps/{m}.bsp": v for m, v in polys_by_map.items()
            }

    results = []
    lines = [
        "| arm | MAE (u) | missed % | false-block(all) % | early>32u % |",
        "|---|---:|---:|---:|---:|",
    ]
    for ci, cell in enumerate(float(c) for c in args.cells.split(",")):
        if not args.skip_corpus_field:
            res = run_study(
                records_by_file, cell=cell,
                steep_min_deg=args.steep_min_deg, exclusion_s=args.exclusion_s,
            )
            results.append(res)
            if ci == 0:
                lines.append(_row("atlas codes (steep bands)", res["atlas_steep"]))
            lines.append(_row(f"corpus heightfield @{cell:g}u", res["heightfield_steep"]))
        if polys_by_map:
            mres = run_mesh_study(
                records_by_file, polys_by_map, cell=cell,
                steep_min_deg=args.steep_min_deg,
            )
            results.append(mres)
            lines.append(_row(f"navmesh fields @{cell:g}u", mres["mesh_steep"]))
    table = "\n".join(lines)
    print(table)
    if args.output is not None:
        write_json(args.output, {
            "sidecars": [str(p) for p in args.sidecars],
            "steep_min_deg": args.steep_min_deg,
            "exclusion_s": args.exclusion_s,
            "sweep": results,
        })
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
