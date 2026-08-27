"""Test a convex-cell cache of first-hit plane identities.

A distance panorama cached at a probe has parallax because its values belong
to the probe origin.  A plane identity does not: once a direction's first-hit
solid face is known, its distance from another origin is one plane equation.

This diagnostic stores one world-yaw first-hit face index per convex cell,
elevation band, and 5-degree yaw.  At query time it locates the current cell,
rotates into that immutable field, and evaluates the referenced plane at the
actual origin and direction.  A same-direction oracle separates residual
within-cell visibility changes from the fixed 5-degree angular sampling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from qnn.diag.static_cell_memory import (
    Cell, CellComplex, _score_arm, load_cell_dump,
)
from qnn.diag.static_map_memory import _record_rays, load_static_records
from qnn.engine_norm import ATLAS_YAWS
from qnn.utils.io import write_json


@dataclass(frozen=True)
class PlaneField:
    face_indices: np.ndarray
    center_distances: np.ndarray


def _world_field_directions(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    template = dict(record)
    template["view_yaw"] = 0.0
    return _record_rays(template)


def build_plane_field(
    complex_: CellComplex, cell_id: int, record: dict[str, Any],
    *, origin: np.ndarray | None = None,
) -> PlaneField:
    directions, limits = _world_field_directions(record)
    if origin is None:
        origin = complex_.cells[cell_id].center
    face_indices = np.full(len(directions), -1, dtype=np.int32)
    distances = limits.astype(np.float64, copy=True)
    for ray_i, (direction, limit) in enumerate(zip(directions, limits)):
        result = complex_.trace(
            origin, direction, float(limit), start_cell=cell_id,
        )
        if result.hit and result.face_index is not None:
            face_indices[ray_i] = result.face_index
            distances[ray_i] = result.distance
    return PlaneField(face_indices=face_indices, center_distances=distances)


def _inside_cell(cell: Cell, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    in_aabb = np.all(points >= cell.mins - 0.025, axis=1)
    in_aabb &= np.all(points <= cell.maxs + 0.025, axis=1)
    return in_aabb & np.all(
        points @ cell.normals.T <= cell.distances[None, :] + 0.025,
        axis=1,
    )


def nearest_grid_point(
    cell: Cell, origin: np.ndarray, spacing: float,
) -> tuple[np.ndarray, bool]:
    """Nearest valid globally anchored grid point, or the cell center."""
    base = np.rint(np.asarray(origin, dtype=np.float64) / spacing).astype(np.int64)
    offsets = np.arange(-2, 3, dtype=np.int64)
    indices = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1)
    points = (base[None, None, None, :] + indices).reshape(-1, 3) * spacing
    valid = points[_inside_cell(cell, points)]
    if not len(valid):
        return cell.center, True
    distances = np.linalg.norm(valid - origin, axis=1)
    return valid[int(np.argmin(distances))], False


def count_grid_points(complex_: CellComplex, spacing: float) -> tuple[int, int]:
    """Count map-load fields on a global 3D grid clipped to each convex cell."""
    total = 0
    center_fallbacks = 0
    for cell in complex_.cells:
        axes = [
            np.arange(
                math.ceil(cell.mins[axis] / spacing),
                math.floor(cell.maxs[axis] / spacing) + 1,
                dtype=np.int64,
            ) * spacing
            for axis in range(3)
        ]
        if any(not len(axis) for axis in axes):
            total += 1
            center_fallbacks += 1
            continue
        cell_count = 0
        xy = np.stack(np.meshgrid(axes[0], axes[1], indexing="ij"), axis=-1)
        xy = xy.reshape(-1, 2)
        for z in axes[2]:
            points = np.column_stack([xy, np.full(len(xy), z)])
            cell_count += int(np.count_nonzero(_inside_cell(cell, points)))
        if cell_count == 0:
            cell_count = 1
            center_fallbacks += 1
        total += cell_count
    return total, center_fallbacks


def _fixed_field_slots(directions: np.ndarray, yaw_count: int) -> np.ndarray:
    if yaw_count != ATLAS_YAWS:
        raise ValueError(f"expected {ATLAS_YAWS} yaw cells, got {yaw_count}")
    yaw = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi)
    world_cells = np.rint(yaw * ATLAS_YAWS / (2.0 * math.pi)).astype(np.int64)
    world_cells %= ATLAS_YAWS
    bands = np.arange(len(directions), dtype=np.int64) // yaw_count
    return bands * yaw_count + world_cells


def reproject_faces(
    complex_: CellComplex, face_indices: np.ndarray, origin: np.ndarray,
    directions: np.ndarray, limits: np.ndarray,
) -> np.ndarray:
    prediction = limits.astype(np.float64, copy=True)
    valid_indices = np.flatnonzero(face_indices >= 0)
    for ray_i in valid_indices:
        face = complex_.solid_faces[int(face_indices[ray_i])]
        denominator = float(np.dot(face.normal, directions[ray_i]))
        offset = float(face.dist - np.dot(face.normal, origin))
        if denominator <= 1e-5 or offset < -0.25:
            continue
        distance = max(offset / denominator, 0.0)
        if distance <= limits[ray_i] + 0.025:
            prediction[ray_i] = distance
    return prediction


def analyze(
    *, cells_dir: Path, sidecars: list[Path], output: Path,
    max_records_per_map: int, grid_spacings: list[float],
) -> dict[str, Any]:
    records = load_static_records(sidecars)
    complexes = {
        complex_.name: complex_
        for complex_ in map(load_cell_dump, sorted(cells_dir.glob("hullcells_*.json")))
    }
    missing = sorted(set(records) - complexes.keys())
    if missing:
        raise ValueError(f"sidecar maps lack cell dumps: {missing}")

    truth_all: list[np.ndarray] = []
    limits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    raw_fixed_all: list[np.ndarray] = []
    plane_fixed_all: list[np.ndarray] = []
    plane_same_all: list[np.ndarray] = []
    fallback_all: list[np.ndarray] = []
    grid_spacings = sorted(set(float(value) for value in grid_spacings), reverse=True)
    if any(value <= 0.0 for value in grid_spacings):
        raise ValueError("grid spacings must be positive")
    grid_all: dict[float, list[np.ndarray]] = {
        spacing: [] for spacing in grid_spacings
    }
    grid_pose_distances: dict[float, list[float]] = {
        spacing: [] for spacing in grid_spacings
    }
    grid_center_fallback_poses: dict[float, int] = {
        spacing: 0 for spacing in grid_spacings
    }
    spacing_pairs = list(zip(grid_spacings, grid_spacings[1:]))
    grid_change_totals: dict[tuple[float, float], list[int]] = {
        pair: [0, 0] for pair in spacing_pairs
    }
    build_ns = 0
    query_ns = 0
    map_reports: dict[str, Any] = {}
    fixed_identity: list[bool] = []
    same_identity: list[bool] = []
    actual_hit: list[bool] = []
    pose_center_distances: list[float] = []

    for map_name, map_records in records.items():
        if max_records_per_map > 0:
            map_records = map_records[:max_records_per_map]
        complex_ = complexes[map_name]
        fields: dict[int, PlaneField] = {}
        grid_fields: dict[float, dict[tuple[int, float, float, float], PlaneField]] = {
            spacing: {} for spacing in grid_spacings
        }
        map_grid_changes: dict[tuple[float, float], list[int]] = {
            pair: [0, 0] for pair in spacing_pairs
        }
        fallback_poses = 0
        print(f"{map_name}: {len(map_records)} records, {len(complex_.cells)} cells")
        for record_i, record in enumerate(map_records):
            origin = np.asarray(record["origin"], dtype=np.float64)
            directions, limits = _record_rays(record)
            yaw_count = np.asarray(record["static_atlas_code"]).shape[1]
            static_distance = np.asarray(
                record["static_atlas_distance"], dtype=np.float64,
            ).reshape(-1)
            truth = np.where(static_distance >= 0.0, static_distance, limits)
            labels = np.asarray(record["static_atlas_code"], dtype=np.int64).reshape(-1)
            cell_id, _ = complex_.locate(origin)

            started = time.perf_counter_ns()
            if cell_id is None:
                fallback_poses += 1
                exact = np.empty(len(directions), dtype=np.float64)
                for ray_i, (direction, limit) in enumerate(zip(directions, limits)):
                    exact[ray_i] = complex_.trace_global_faces(
                        origin, direction, float(limit),
                    ).distance
                raw_fixed = exact.copy()
                plane_fixed = exact.copy()
                plane_same = exact.copy()
                fallback = exact
                grid_predictions = {
                    spacing: exact.copy() for spacing in grid_spacings
                }
            else:
                pose_center_distances.append(float(np.linalg.norm(
                    origin - complex_.cells[cell_id].center,
                )))
                if cell_id not in fields:
                    build_started = time.perf_counter_ns()
                    fields[cell_id] = build_plane_field(complex_, cell_id, record)
                    build_ns += time.perf_counter_ns() - build_started
                field = fields[cell_id]
                slots = _fixed_field_slots(directions, yaw_count)
                fixed_faces = field.face_indices[slots]
                raw_fixed = field.center_distances[slots]
                plane_fixed = reproject_faces(
                    complex_, fixed_faces, origin, directions, limits,
                )

                same_faces = np.full(len(directions), -1, dtype=np.int32)
                for ray_i, (direction, limit) in enumerate(zip(directions, limits)):
                    result = complex_.trace(
                        complex_.cells[cell_id].center,
                        direction,
                        float(limit),
                        start_cell=cell_id,
                    )
                    if result.hit and result.face_index is not None:
                        same_faces[ray_i] = result.face_index
                plane_same = reproject_faces(
                    complex_, same_faces, origin, directions, limits,
                )

                grid_predictions: dict[float, np.ndarray] = {}
                grid_fixed_faces: dict[float, np.ndarray] = {}
                for spacing in grid_spacings:
                    sample, used_center = nearest_grid_point(
                        complex_.cells[cell_id], origin, spacing,
                    )
                    grid_pose_distances[spacing].append(float(np.linalg.norm(
                        sample - origin,
                    )))
                    if used_center:
                        grid_center_fallback_poses[spacing] += 1
                    key = (cell_id, float(sample[0]), float(sample[1]), float(sample[2]))
                    if key not in grid_fields[spacing]:
                        build_started = time.perf_counter_ns()
                        grid_fields[spacing][key] = build_plane_field(
                            complex_, cell_id, record, origin=sample,
                        )
                        build_ns += time.perf_counter_ns() - build_started
                    grid_field = grid_fields[spacing][key]
                    grid_faces = grid_field.face_indices[slots]
                    grid_fixed_faces[spacing] = grid_faces
                    grid_predictions[spacing] = reproject_faces(
                        complex_, grid_faces, origin, directions, limits,
                    )
                for pair in spacing_pairs:
                    changed = int(np.count_nonzero(
                        grid_fixed_faces[pair[0]] != grid_fixed_faces[pair[1]]
                    ))
                    total = len(directions)
                    map_grid_changes[pair][0] += changed
                    map_grid_changes[pair][1] += total
                    grid_change_totals[pair][0] += changed
                    grid_change_totals[pair][1] += total

                fallback = np.empty(len(directions), dtype=np.float64)
                for ray_i, (direction, limit) in enumerate(zip(directions, limits)):
                    actual = complex_.trace(
                        origin, direction, float(limit), start_cell=cell_id,
                    )
                    fallback[ray_i] = actual.distance
                    actual_face = (
                        actual.face_index
                        if actual.hit and actual.face_index is not None else -1
                    )
                    fixed_identity.append(int(fixed_faces[ray_i]) == actual_face)
                    same_identity.append(int(same_faces[ray_i]) == actual_face)
                    actual_hit.append(actual_face >= 0)
            query_ns += time.perf_counter_ns() - started
            truth_all.append(truth)
            limits_all.append(limits)
            labels_all.append(labels)
            raw_fixed_all.append(raw_fixed)
            plane_fixed_all.append(plane_fixed)
            plane_same_all.append(plane_same)
            fallback_all.append(fallback)
            for spacing in grid_spacings:
                grid_all[spacing].append(grid_predictions[spacing])
            if (record_i + 1) % 25 == 0:
                print(f"  {record_i + 1}/{len(map_records)}", flush=True)

        integrity = complex_.integrity()
        cell_count = len(complex_.cells)
        field_entries = cell_count * len(_world_field_directions(map_records[0])[0])
        point_location_bytes = integrity["planes"] * 16 + cell_count * 32
        solid_plane_bytes = integrity["face_kinds"].get("solid", 0) * 16
        field_bytes = field_entries * 2
        grid_reports: dict[str, Any] = {}
        grid_counts: dict[float, int] = {}
        for spacing in grid_spacings:
            sample_count, center_samples = count_grid_points(complex_, spacing)
            grid_counts[spacing] = sample_count
            sample_field_bytes = sample_count * len(_world_field_directions(map_records[0])[0]) * 2
            grid_reports[f"{spacing:g}"] = {
                "samples": sample_count,
                "center_samples_for_grid_empty_cells": center_samples,
                "fields_built_for_evaluation": len(grid_fields[spacing]),
                "map_load_ray_queries": sample_count * len(_world_field_directions(map_records[0])[0]),
                "estimated_bytes": {
                    "uint16_face_field": sample_field_bytes,
                    "sample_positions": sample_count * 12,
                    "point_location": point_location_bytes,
                    "solid_plane_table": solid_plane_bytes,
                    "total": (
                        sample_field_bytes + sample_count * 12
                        + point_location_bytes + solid_plane_bytes
                    ),
                },
            }
        sparse_reports: dict[str, Any] = {}
        for coarse, fine in spacing_pairs:
            changed, compared = map_grid_changes[(coarse, fine)]
            changed_rate = changed / compared if compared else 0.0
            ray_slots = len(_world_field_directions(map_records[0])[0])
            coarse_bytes = grid_counts[coarse] * ray_slots * 2
            coarse_uint12_bytes = math.ceil(
                grid_counts[coarse] * ray_slots * 12 / 8
            )
            pair_override_bytes = grid_counts[fine] * (
                4 + ray_slots * changed_rate * 4
            )
            bitmask_override_bytes = grid_counts[fine] * (
                4 + math.ceil(ray_slots / 8) + ray_slots * changed_rate * 2
            )
            bitmask_uint12_override_bytes = math.ceil(
                grid_counts[fine] * (4 + math.ceil(ray_slots / 8))
                + grid_counts[fine] * ray_slots * changed_rate * 12 / 8
            )
            sparse_reports[f"{coarse:g}_to_{fine:g}"] = {
                "evaluated_face_change_rate": changed_rate,
                "evaluated_rays": compared,
                "estimated_bytes": {
                    "coarse_uint16_face_field": coarse_bytes,
                    "coarse_uint12_face_field": coarse_uint12_bytes,
                    "fine_slot_face_pairs": int(round(pair_override_bytes)),
                    "fine_bitmask_and_face_values": int(round(bitmask_override_bytes)),
                    "fine_bitmask_and_uint12_face_values": bitmask_uint12_override_bytes,
                    "point_location": point_location_bytes,
                    "solid_plane_table": solid_plane_bytes,
                    "total": int(round(
                        coarse_uint12_bytes + bitmask_uint12_override_bytes
                        + point_location_bytes + solid_plane_bytes
                    )),
                },
                "caveat": "cost extrapolates retained-pose face-change rate",
            }
        map_reports[map_name] = {
            "records": len(map_records),
            "cells_used": len(fields),
            "cells_total": cell_count,
            "fallback_poses": fallback_poses,
            "field_entries": field_entries,
            "map_load_ray_queries": field_entries,
            "estimated_bytes": {
                "uint16_face_field": field_bytes,
                "point_location": point_location_bytes,
                "solid_plane_table": solid_plane_bytes,
                "total": field_bytes + point_location_bytes + solid_plane_bytes,
            },
            "tick_gathered_band_tokens": len(record["elevations"]),
            "grid_fields": grid_reports,
            "sparse_override_proxy": sparse_reports,
        }

    truth = np.concatenate(truth_all)
    limits = np.concatenate(limits_all)
    labels = np.concatenate(labels_all)
    arms = {
        "cell_center_depth_world_5deg": _score_arm(
            truth=truth, prediction_distance=np.concatenate(raw_fixed_all),
            limits=limits, labels=labels,
        ),
        "cell_plane_world_5deg": _score_arm(
            truth=truth, prediction_distance=np.concatenate(plane_fixed_all),
            limits=limits, labels=labels,
        ),
        "cell_plane_same_direction_oracle": _score_arm(
            truth=truth, prediction_distance=np.concatenate(plane_same_all),
            limits=limits, labels=labels,
        ),
        "full_portal_control": _score_arm(
            truth=truth, prediction_distance=np.concatenate(fallback_all),
            limits=limits, labels=labels,
        ),
    }
    for spacing in grid_spacings:
        arms[f"cell_plane_grid_{spacing:g}"] = _score_arm(
            truth=truth, prediction_distance=np.concatenate(grid_all[spacing]),
            limits=limits, labels=labels,
        )
    report = {
        "schema": 1,
        "representation": "convex_cell_first_hit_plane_field",
        "records": sum(item["records"] for item in map_reports.values()),
        "rays": len(truth),
        "map_build_seconds_python_lazy": build_ns / 1e9,
        "query_seconds_python": query_ns / 1e9,
        "visibility_stability": {
            "pose_to_cell_center_distance": {
                name: float(value)
                for name, value in zip(
                    ("p50", "p90", "p95", "p99", "max"),
                    np.percentile(pose_center_distances, (50, 90, 95, 99, 100)),
                )
            },
            "fixed_5deg_face_identity_rate": float(np.mean(fixed_identity)),
            "same_direction_face_identity_rate": float(np.mean(same_identity)),
            "fixed_5deg_identity_given_actual_hit": float(np.mean(
                np.asarray(fixed_identity)[np.asarray(actual_hit)]
            )),
            "same_direction_identity_given_actual_hit": float(np.mean(
                np.asarray(same_identity)[np.asarray(actual_hit)]
            )),
            "grid_pose_distance": {
                f"{spacing:g}": {
                    **{
                        name: float(value)
                        for name, value in zip(
                            ("p50", "p90", "p95", "p99", "max"),
                            np.percentile(
                                grid_pose_distances[spacing],
                                (50, 90, 95, 99, 100),
                            ),
                        )
                    },
                    "center_fallback_poses": grid_center_fallback_poses[spacing],
                }
                for spacing in grid_spacings
            },
            "sparse_override_proxy": {
                f"{coarse:g}_to_{fine:g}": {
                    "face_change_rate": (
                        grid_change_totals[(coarse, fine)][0]
                        / grid_change_totals[(coarse, fine)][1]
                    ),
                    "evaluated_rays": grid_change_totals[(coarse, fine)][1],
                    "prediction_equals_fine_arm": True,
                }
                for coarse, fine in spacing_pairs
            },
        },
        "maps": map_reports,
        "arms": arms,
    }
    write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-dir", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records-per-map", type=int, default=0)
    parser.add_argument(
        "--grid-spacings", type=float, nargs="+", default=[64.0, 32.0, 16.0],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze(
        cells_dir=args.cells_dir,
        sidecars=args.sidecars,
        output=args.output,
        max_records_per_map=args.max_records_per_map,
        grid_spacings=args.grid_spacings,
    )
    print(json.dumps({
        "output": str(args.output),
        "records": report["records"],
        "rays": report["rays"],
        "arms": {
            name: {
                "mae": arm["plane"]["mae"],
                "missed": arm["plane"]["missed_obstacle_rate"],
                "passed": arm["gate"]["passed"],
            }
            for name, arm in report["arms"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
