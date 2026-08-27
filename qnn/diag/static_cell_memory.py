"""Evaluate a map-load convex-cell/portal representation of static geometry.

The engine already constructs convex non-solid hull-1 leaves while carving the
world BSP.  The diagnostic export retains those leaves, their boundary
polygons, and exact portal adjacency.  This module tests whether that immutable
map-load representation can reproduce the static egocentric atlas without
running world traces or rebuilding map tokens each tick.

The full arm traverses portals until a solid boundary or the ray range.  The
bounded arms expose the useful budget curve: after ``H`` portal transitions,
``optimistic`` treats unresolved space as open while ``conservative`` treats
the next portal as blocked.  The accompanying certified-coverage rate makes
clear how often either approximation actually had enough geometry to decide.

Examples::

    python -m qnn.diag.static_cell_memory dump-cells \
      --worker assets/bin/qw_demo_worker \
      --demo-dir artifacts/corpus/qwd \
      --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
      --maps dm2 dm4 dm6 \
      --out-dir artifacts/diag/static_cell_memory_v1/cells

    python -m qnn.diag.static_cell_memory analyze \
      --cells-dir artifacts/diag/static_cell_memory_v1/cells \
      --sidecars artifacts/diag/static_map_memory_v1/sidecars/*.jsonl \
      --output runs/eval/_static_cell_memory_v1/report.json
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from qnn.bc.probe_atlas import DemoQueryWorker
from qnn.diag.spatial_reconstruction import _metrics, threshold_failures
from qnn.utils.io import write_json
from qnn.diag.static_map_memory import (
    DEFAULT_GATE,
    _quantized_prediction,
    _record_rays,
    canonical_map_name,
    first_demo_per_map,
    load_static_records,
)


CELL_DUMP_SCHEMA = 1
DEFAULT_HOP_BUDGETS = (0, 1, 2, 3, 4, 6, 8)
_PLANE_TOLERANCE = 0.025
_POINT_TOLERANCE = 0.05
_RAY_EPSILON = 1e-5


def dump_cells(
    *, worker_path: str, demo_dir: Path, manifest: Path, asset_root: Path,
    maps: list[str], out_dir: Path, tick_hz: int,
) -> list[Path]:
    """Dump exact world-hull empty cells from one representative demo/map."""
    demos = first_demo_per_map(manifest)
    missing = sorted(set(maps) - demos.keys())
    if missing:
        raise ValueError(f"manifest has no representative demo for maps {missing}")
    game_dir = os.path.relpath(demo_dir.absolute(), asset_root.absolute())
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for map_name in maps:
        worker = DemoQueryWorker(worker_path, game_dir, asset_root, tick_hz)
        try:
            result = worker.nav_query(demos[map_name], "hull_cells")
            if int(result["count"]) != len(result["cells"]):
                raise ValueError(
                    f"{map_name}: cell count {result['count']} != "
                    f"payload rows {len(result['cells'])}"
                )
            payload = {
                "schema": CELL_DUMP_SCHEMA,
                "map": map_name,
                "source": "exact_hull1_convex_leaves",
                "result": result,
            }
            path = out_dir / f"hullcells_{map_name}.json"
            path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
            written.append(path)
            print(
                f"{map_name}: {result['count']} cells, "
                f"{result['portal_faces']} portals, "
                f"{result['solid_faces']} solid faces -> {path}"
            )
        finally:
            worker.close()
    return written


@dataclass(frozen=True)
class CellFace:
    kind: str
    neighbor: int
    normal: np.ndarray
    dist: float
    vertices: np.ndarray


@dataclass(frozen=True)
class Cell:
    id: int
    center: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    normals: np.ndarray
    distances: np.ndarray
    faces: tuple[CellFace, ...]


@dataclass(frozen=True)
class RayResult:
    distance: float
    hit: bool
    portal_exits: tuple[float, ...]
    unresolved: bool
    reason: str
    face_index: int | None = None

    @property
    def portal_hops(self) -> int:
        return len(self.portal_exits)


class CellComplex:
    """Immutable convex cells with exact face-fragment portal routing."""

    def __init__(self, *, name: str, cells: list[Cell], source_bytes: int):
        if not cells:
            raise ValueError(f"{name}: empty cell complex")
        ids = [cell.id for cell in cells]
        if ids != list(range(len(cells))):
            raise ValueError(f"{name}: cell ids are not dense and ordered")
        self.name = name
        self.cells = tuple(cells)
        self.source_bytes = int(source_bytes)
        self.solid_faces = tuple(
            face for cell in cells for face in cell.faces if face.kind == "solid"
        )
        self._solid_normals = np.asarray(
            [face.normal for face in self.solid_faces], dtype=np.float64,
        )
        self._solid_distances = np.asarray(
            [face.dist for face in self.solid_faces], dtype=np.float64,
        )
        self._solid_face_indices = {
            id(face): index for index, face in enumerate(self.solid_faces)
        }

    def containing_cells(self, point: np.ndarray) -> list[int]:
        point = np.asarray(point, dtype=np.float64)
        result: list[int] = []
        for cell in self.cells:
            if np.any(point < cell.mins - _POINT_TOLERANCE):
                continue
            if np.any(point > cell.maxs + _POINT_TOLERANCE):
                continue
            if np.all(cell.normals @ point <= cell.distances + _PLANE_TOLERANCE):
                result.append(cell.id)
        return result

    def locate(self, point: np.ndarray) -> tuple[int | None, int]:
        candidates = self.containing_cells(point)
        if not candidates:
            return None, 0
        if len(candidates) == 1:
            return candidates[0], 1
        # A pose exactly on a portal is valid in both cells.  Prefer the cell
        # whose least plane slack places the point most deeply in its volume.
        point = np.asarray(point, dtype=np.float64)
        chosen = max(
            candidates,
            key=lambda cell_id: float(np.min(
                self.cells[cell_id].distances
                - self.cells[cell_id].normals @ point
            )),
        )
        return chosen, len(candidates)

    @staticmethod
    def _point_in_face(point: np.ndarray, face: CellFace) -> bool:
        vertices = face.vertices
        edges = np.roll(vertices, -1, axis=0) - vertices
        offsets = point - vertices
        signed = np.einsum("ij,j->i", np.cross(edges, offsets), face.normal)
        # The carve winding is outward, but accepting either consistent winding
        # also makes the diagnostic insensitive to imported BSP conventions.
        return bool(
            np.all(signed >= -_POINT_TOLERANCE)
            or np.all(signed <= _POINT_TOLERANCE)
        )

    @staticmethod
    def _exit_face(
        cell: Cell, point: np.ndarray, plane_indices: np.ndarray,
    ) -> CellFace | None:
        matches: list[CellFace] = []
        for face in cell.faces:
            on_plane = False
            for plane_i in plane_indices:
                if (
                    np.dot(face.normal, cell.normals[plane_i]) > 1.0 - 1e-5
                    and abs(face.dist - cell.distances[plane_i])
                    <= _PLANE_TOLERANCE
                ):
                    on_plane = True
                    break
            if on_plane and CellComplex._point_in_face(point, face):
                matches.append(face)
        if not matches:
            return None
        # At a polygon edge, the exact triangle query may see either adjacent
        # fragment.  Solid-first is the conservative and collision-safe tie.
        matches.sort(key=lambda face: face.kind != "solid")
        return matches[0]

    def trace(
        self, origin: np.ndarray, direction: np.ndarray, limit: float,
        *, start_cell: int | None = None, max_steps: int = 512,
    ) -> RayResult:
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        if start_cell is None:
            start_cell, _ = self.locate(origin)
        if start_cell is None:
            return self.trace_global_faces(origin, direction, limit)

        cell_id = int(start_cell)
        minimum_t = -_PLANE_TOLERANCE
        portal_exits: list[float] = []
        for _ in range(max_steps):
            cell = self.cells[cell_id]
            denominators = cell.normals @ direction
            forward = denominators > _RAY_EPSILON
            candidates = np.full(len(denominators), np.inf, dtype=np.float64)
            numerators = cell.distances - cell.normals @ origin
            np.divide(numerators, denominators, out=candidates, where=forward)
            candidates[(~forward) | (candidates < minimum_t)] = np.inf
            exit_t = float(np.min(candidates))
            if not np.isfinite(exit_t) or exit_t > limit + _PLANE_TOLERANCE:
                return RayResult(float(limit), False, tuple(portal_exits), False, "range")

            plane_indices = np.flatnonzero(
                np.abs(candidates - exit_t) <= _PLANE_TOLERANCE
            )
            point = origin + max(exit_t, 0.0) * direction
            face = self._exit_face(cell, point, plane_indices)
            if face is None:
                fallback = self.trace_global_faces(
                    origin, direction, limit,
                    reason="global_face_edge_fallback",
                )
                return fallback
            if face.kind == "solid":
                return RayResult(
                    max(exit_t, 0.0), True, tuple(portal_exits), False, "solid",
                    self._solid_face_indices[id(face)],
                )
            if face.kind != "portal" or not (0 <= face.neighbor < len(self.cells)):
                return RayResult(
                    float(limit), False, tuple(portal_exits), True, face.kind,
                )
            if face.neighbor == cell_id:
                return RayResult(
                    float(limit), False, tuple(portal_exits), True, "self_portal",
                )
            portal_exits.append(max(exit_t, 0.0))
            minimum_t = exit_t + _PLANE_TOLERANCE
            cell_id = face.neighbor

        return RayResult(
            float(limit), False, tuple(portal_exits), True, "step_limit",
        )

    def trace_global_faces(
        self, origin: np.ndarray, direction: np.ndarray, limit: float,
        *, reason: str = "global_face_fallback",
    ) -> RayResult:
        """Exact static-face fallback for origins outside non-solid cells.

        Cell solid-face normals point from empty space into solid.  This is the
        inverse of the production carve-face normal, so a production front-face
        hit has ``cell_normal · direction > 0``.  The 0.25-unit behind-plane
        tolerance matches ``QNN_CarveQueryRay``.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        denominators = self._solid_normals @ direction
        plane_offsets = self._solid_distances - self._solid_normals @ origin
        valid = (denominators > _RAY_EPSILON) & (plane_offsets >= -0.25)
        distances = np.full(len(self.solid_faces), np.inf, dtype=np.float64)
        np.divide(plane_offsets, denominators, out=distances, where=valid)
        distances = np.maximum(distances, 0.0)
        candidates = np.flatnonzero(valid & (distances <= limit))
        for face_i in candidates[np.argsort(distances[candidates])]:
            distance = float(distances[face_i])
            point = origin + distance * direction
            face = self.solid_faces[int(face_i)]
            if self._point_in_face(point, face):
                return RayResult(
                    distance, True, (), False, reason, int(face_i),
                )
        return RayResult(
            float(limit), False, (), False, reason,
        )

    def neighborhood_counts(self, start_cell: int, budget: int) -> dict[str, int]:
        visited = {int(start_cell)}
        queue = deque([(int(start_cell), 0)])
        while queue:
            cell_id, depth = queue.popleft()
            if depth >= budget:
                continue
            for face in self.cells[cell_id].faces:
                if face.kind != "portal" or face.neighbor in visited:
                    continue
                visited.add(face.neighbor)
                queue.append((face.neighbor, depth + 1))
        faces = [face for cell_id in visited for face in self.cells[cell_id].faces]
        return {
            "cells": len(visited),
            "solid_faces": sum(face.kind == "solid" for face in faces),
            "portal_faces": sum(face.kind == "portal" for face in faces),
            "face_fragments": len(faces),
            "vertices": sum(len(face.vertices) for face in faces),
        }

    def integrity(self) -> dict[str, Any]:
        portal_faces = [
            (cell.id, face)
            for cell in self.cells for face in cell.faces
            if face.kind == "portal"
        ]
        invalid_neighbors = sum(
            not (0 <= face.neighbor < len(self.cells))
            for _, face in portal_faces
        )
        reciprocal = 0
        for cell_id, face in portal_faces:
            if not (0 <= face.neighbor < len(self.cells)):
                continue
            neighbor = self.cells[face.neighbor]
            if any(
                other.kind == "portal" and other.neighbor == cell_id
                and np.dot(other.normal, face.normal) < -1.0 + 1e-5
                and abs(other.dist + face.dist) <= _PLANE_TOLERANCE
                for other in neighbor.faces
            ):
                reciprocal += 1
        kinds: dict[str, int] = {}
        plane_count = 0
        vertex_count = 0
        for cell in self.cells:
            plane_count += len(cell.normals)
            for face in cell.faces:
                kinds[face.kind] = kinds.get(face.kind, 0) + 1
                vertex_count += len(face.vertices)
        # Approximate compact runtime storage: float32 geometry plus int32
        # offsets/kinds/neighbors.  JSON bytes are also reported exactly.
        compact_bytes = (
            plane_count * 4 * 4
            + sum(kinds.values()) * 6 * 4
            + vertex_count * 3 * 4
            + len(self.cells) * 10 * 4
        )
        return {
            "cells": len(self.cells),
            "planes": plane_count,
            "face_fragments": sum(kinds.values()),
            "face_kinds": kinds,
            "vertices": vertex_count,
            "invalid_portal_neighbors": invalid_neighbors,
            "reciprocal_portal_rate": (
                reciprocal / len(portal_faces) if portal_faces else 1.0
            ),
            "source_json_bytes": self.source_bytes,
            "estimated_compact_bytes": compact_bytes,
        }


def load_cell_dump(path: Path) -> CellComplex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema", 0)) != CELL_DUMP_SCHEMA:
        raise ValueError(f"{path}: unsupported schema {payload.get('schema')}")
    result = payload["result"]
    cells: list[Cell] = []
    for row in result["cells"]:
        planes = row["planes"]
        faces = tuple(
            CellFace(
                kind=str(face["kind"]),
                neighbor=int(face["neighbor"]),
                normal=np.asarray(face["normal"], dtype=np.float64),
                dist=float(face["dist"]),
                vertices=np.asarray(face["verts"], dtype=np.float64),
            )
            for face in row["faces"]
        )
        cells.append(Cell(
            id=int(row["id"]),
            center=np.asarray(row["center"], dtype=np.float64),
            mins=np.asarray(row["mins"], dtype=np.float64),
            maxs=np.asarray(row["maxs"], dtype=np.float64),
            normals=np.asarray([plane["normal"] for plane in planes], dtype=np.float64),
            distances=np.asarray([plane["dist"] for plane in planes], dtype=np.float64),
            faces=faces,
        ))
    if int(result["count"]) != len(cells):
        raise ValueError(f"{path}: declared count does not match payload")
    return CellComplex(
        name=canonical_map_name(payload["map"]),
        cells=cells,
        source_bytes=path.stat().st_size,
    )


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _score_arm(
    *, truth: np.ndarray, prediction_distance: np.ndarray, limits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    codes, prediction = _quantized_prediction(prediction_distance, limits)
    plane = _metrics(truth, prediction, limits)
    failures = threshold_failures(
        {"plane": plane, "by_elevation": {"0": plane}}, **DEFAULT_GATE,
    )
    return {
        "code_accuracy": float(np.mean(codes == labels)),
        "plane": plane,
        "gate": {**DEFAULT_GATE, "passed": not failures, "failures": failures},
    }


def analyze(
    *, cells_dir: Path, sidecars: list[Path], output: Path,
    hop_budgets: list[int], max_records_per_map: int,
) -> dict[str, Any]:
    records = load_static_records(sidecars)
    complexes = {
        complex_.name: complex_
        for complex_ in map(load_cell_dump, sorted(cells_dir.glob("hullcells_*.json")))
    }
    missing = sorted(set(records) - complexes.keys())
    if missing:
        raise ValueError(f"sidecar maps lack cell dumps: {missing}")
    hop_budgets = sorted(set(int(value) for value in hop_budgets))
    if not hop_budgets or hop_budgets[0] < 0:
        raise ValueError("hop budgets must be non-negative")

    truths: list[np.ndarray] = []
    limits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    full_distances: list[np.ndarray] = []
    optimistic: dict[int, list[np.ndarray]] = {value: [] for value in hop_budgets}
    conservative: dict[int, list[np.ndarray]] = {value: [] for value in hop_budgets}
    certified: dict[int, int] = {value: 0 for value in hop_budgets}
    pose_candidates: list[int] = []
    fallback_poses = 0
    fallback_rays = 0
    ray_hops: list[int] = []
    unresolved_reasons: dict[str, int] = {}
    neighborhood: dict[int, dict[str, list[int]]] = {
        value: {name: [] for name in (
            "cells", "solid_faces", "portal_faces", "face_fragments", "vertices",
        )}
        for value in hop_budgets
    }
    map_reports: dict[str, Any] = {}
    elapsed_ns = 0

    for map_name, map_records in records.items():
        complex_ = complexes[map_name]
        if max_records_per_map > 0:
            map_records = map_records[:max_records_per_map]
        map_unresolved = 0
        map_fallback_poses = 0
        map_fallback_rays = 0
        map_rays = 0
        map_hops: list[int] = []
        print(f"{map_name}: {len(map_records)} records, {len(complex_.cells)} cells")
        for record_i, record in enumerate(map_records):
            origin = np.asarray(record["origin"], dtype=np.float64)
            start_cell, candidate_count = complex_.locate(origin)
            pose_candidates.append(candidate_count)
            uses_fallback = start_cell is None
            if uses_fallback:
                fallback_poses += 1
                map_fallback_poses += 1
            else:
                for budget in hop_budgets:
                    counts = complex_.neighborhood_counts(start_cell, budget)
                    for name, value in counts.items():
                        neighborhood[budget][name].append(value)

            directions, limits = _record_rays(record)
            target_dist = np.asarray(
                record["static_atlas_distance"], dtype=np.float64,
            ).reshape(-1)
            truth = np.where(target_dist >= 0.0, target_dist, limits)
            labels = np.asarray(record["static_atlas_code"], dtype=np.int64).reshape(-1)
            record_full = np.empty(len(directions), dtype=np.float64)
            record_opt = {
                budget: np.empty(len(directions), dtype=np.float64)
                for budget in hop_budgets
            }
            record_con = {
                budget: np.empty(len(directions), dtype=np.float64)
                for budget in hop_budgets
            }
            started = time.perf_counter_ns()
            for ray_i, (direction, limit) in enumerate(zip(directions, limits)):
                result = complex_.trace(
                    origin, direction, float(limit), start_cell=start_cell,
                )
                uses_ray_fallback = result.reason.startswith("global_face")
                if uses_ray_fallback:
                    fallback_rays += 1
                    map_fallback_rays += 1
                record_full[ray_i] = result.distance
                ray_hops.append(result.portal_hops)
                map_hops.append(result.portal_hops)
                map_rays += 1
                if result.unresolved:
                    map_unresolved += 1
                    unresolved_reasons[result.reason] = (
                        unresolved_reasons.get(result.reason, 0) + 1
                    )
                for budget in hop_budgets:
                    locally_certified = (
                        not uses_ray_fallback and not result.unresolved
                        and result.portal_hops <= budget
                    )
                    if locally_certified:
                        certified[budget] += 1
                    if locally_certified or uses_ray_fallback:
                        record_opt[budget][ray_i] = result.distance
                        record_con[budget][ray_i] = result.distance
                    else:
                        record_opt[budget][ray_i] = limit
                        record_con[budget][ray_i] = (
                            result.portal_exits[budget]
                            if len(result.portal_exits) > budget
                            else result.distance
                        )
            elapsed_ns += time.perf_counter_ns() - started
            truths.append(truth)
            limits_all.append(limits)
            labels_all.append(labels)
            full_distances.append(record_full)
            for budget in hop_budgets:
                optimistic[budget].append(record_opt[budget])
                conservative[budget].append(record_con[budget])
            if (record_i + 1) % 25 == 0:
                print(f"  {record_i + 1}/{len(map_records)}", flush=True)
        map_reports[map_name] = {
            "records": len(map_records),
            "rays": map_rays,
            "unresolved_rays": map_unresolved,
            "unresolved_rate": map_unresolved / map_rays if map_rays else 0.0,
            "fallback_poses": map_fallback_poses,
            "fallback_rays": map_fallback_rays,
            "portal_hops": _percentiles(map_hops),
            "integrity": complex_.integrity(),
        }

    truth = np.concatenate(truths)
    limits = np.concatenate(limits_all)
    labels = np.concatenate(labels_all)
    full = np.concatenate(full_distances)
    ray_count = len(truth)
    arms: dict[str, Any] = {
        "full": _score_arm(
            truth=truth, prediction_distance=full, limits=limits, labels=labels,
        ),
    }
    for budget in hop_budgets:
        opt = _score_arm(
            truth=truth,
            prediction_distance=np.concatenate(optimistic[budget]),
            limits=limits,
            labels=labels,
        )
        con = _score_arm(
            truth=truth,
            prediction_distance=np.concatenate(conservative[budget]),
            limits=limits,
            labels=labels,
        )
        coverage = certified[budget] / ray_count
        opt["certified_coverage"] = coverage
        con["certified_coverage"] = coverage
        arms[f"hop_{budget}_optimistic"] = opt
        arms[f"hop_{budget}_conservative"] = con

    report = {
        "schema": 1,
        "representation": "exact_hull1_convex_cells_and_portals",
        "records": sum(report["records"] for report in map_reports.values()),
        "rays": ray_count,
        "hop_budgets": hop_budgets,
        "pose_containment": {
            "located_rate": float(np.mean(np.asarray(pose_candidates) > 0)),
            "unique_rate": float(np.mean(np.asarray(pose_candidates) == 1)),
            "candidate_counts": _percentiles(pose_candidates),
        },
        "global_face_fallback": {
            "poses": fallback_poses,
            "pose_rate": fallback_poses / len(pose_candidates),
            "rays": fallback_rays,
            "ray_rate": fallback_rays / ray_count,
        },
        "portal_hops": _percentiles(ray_hops),
        "unresolved": {
            "rays": sum(unresolved_reasons.values()),
            "rate": sum(unresolved_reasons.values()) / ray_count,
            "reasons": unresolved_reasons,
        },
        "neighborhood_budget": {
            str(budget): {
                name: _percentiles(values)
                for name, values in buckets.items()
            }
            for budget, buckets in neighborhood.items()
        },
        "elapsed_seconds": elapsed_ns / 1e9,
        "maps": map_reports,
        "arms": arms,
    }
    write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump = subparsers.add_parser("dump-cells")
    dump.add_argument("--worker", required=True)
    dump.add_argument("--demo-dir", type=Path, required=True)
    dump.add_argument("--manifest", type=Path, required=True)
    dump.add_argument("--asset-root", type=Path, default=Path("assets"))
    dump.add_argument("--maps", nargs="+", required=True)
    dump.add_argument("--out-dir", type=Path, required=True)
    dump.add_argument("--tick-hz", type=int, default=20)

    score = subparsers.add_parser("analyze")
    score.add_argument("--cells-dir", type=Path, required=True)
    score.add_argument("--sidecars", type=Path, nargs="+", required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--hop-budgets", type=int, nargs="+", default=DEFAULT_HOP_BUDGETS)
    score.add_argument("--max-records-per-map", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dump-cells":
        dump_cells(
            worker_path=args.worker,
            demo_dir=args.demo_dir,
            manifest=args.manifest,
            asset_root=args.asset_root,
            maps=[canonical_map_name(value) for value in args.maps],
            out_dir=args.out_dir,
            tick_hz=args.tick_hz,
        )
        return 0
    report = analyze(
        cells_dir=args.cells_dir,
        sidecars=args.sidecars,
        output=args.output,
        hop_budgets=args.hop_budgets,
        max_records_per_map=args.max_records_per_map,
    )
    print(json.dumps({
        "output": str(args.output),
        "records": report["records"],
        "rays": report["rays"],
        "unresolved": report["unresolved"],
        "full": report["arms"]["full"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
