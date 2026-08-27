"""Test immutable map K/V as a teacher for the egocentric depth atlas.

This diagnostic isolates the central static-memory claim before policy work:
an exact hull-1 face table is encoded once per map, its attention keys and
values are cached, and moving ``(origin, ray direction)`` queries must recover
the static-world atlas.  Querying never rewrites or re-encodes map tokens.

Two positional arms share the same cached K/V:

``absolute``
    Static map-position Fourier features and dynamic query-position Fourier
    features interact only through dot-product attention.

``relative_bias``
    The same cached K/V plus a small query/key attention bias computed from
    relative center distance, ray alignment, and face orientation.  This is
    pose-dependent routing math, not map-token assembly.

The engine-side ``nav_query kind=hull_faces`` dump and schema-8 spatial
sidecars are diagnostic-only; neither changes the observation wire.

Examples::

    python -m qnn.diag.static_map_memory dump-faces \
      --worker assets/bin/qw_demo_worker \
      --demo-dir artifacts/corpus/qwd \
      --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
      --maps dm2 dm4 dm6 --out-dir artifacts/diag/static_map_memory/faces

    python -m qnn.diag.static_map_memory train \
      --faces-dir artifacts/diag/static_map_memory/faces \
      --sidecars artifacts/diag/static_map_memory/sidecars/*.jsonl \
      --position-mode relative_bias \
      --output-dir runs/eval/_static_map_memory_relative_v1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from qnn.bc.probe_atlas import DemoQueryWorker
from qnn.diag.spatial_reconstruction import (
    _metrics,
    _record_limits,
    threshold_failures,
)
from qnn.engine_norm import ATLAS_DEPTH_LEVELS, ATLAS_MISS_CODE
from qnn.utils.io import write_json


FACE_DUMP_SCHEMA = 1
POSITION_MODES = ("absolute", "relative_bias")
DEFAULT_GATE = {
    "max_mae": 9.43,
    "max_missed_obstacle_rate": 0.0204,
    "max_false_block_rate_all": 0.01,
    "max_blocked_early_gt_32_rate": 0.0288,
}
_LEVELS = np.asarray(ATLAS_DEPTH_LEVELS, dtype=np.float64)


def canonical_map_name(value: str) -> str:
    """``maps/dm2.bsp`` and ``dm2`` both become ``dm2``."""
    return Path(value).stem


def first_demo_per_map(manifest: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result.setdefault(canonical_map_name(row["map"]), row["file"])
    return result


def dump_faces(
    *, worker_path: str, demo_dir: Path, manifest: Path, asset_root: Path,
    maps: list[str], out_dir: Path, tick_hz: int,
) -> list[Path]:
    demos = first_demo_per_map(manifest)
    missing = sorted(set(maps) - demos.keys())
    if missing:
        raise ValueError(f"manifest has no representative demo for maps {missing}")
    game_dir = os.path.relpath(demo_dir.absolute(), asset_root.absolute())
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for map_name in maps:
        # A fresh process makes each bulk dump independent of demo-reset
        # state and keeps failures attributable to one map.
        query_worker = DemoQueryWorker(worker_path, game_dir, asset_root, tick_hz)
        try:
            result = query_worker.nav_query(demos[map_name], "hull_faces")
            if int(result["count"]) != len(result["faces"]):
                raise ValueError(
                    f"{map_name}: face count {result['count']} != "
                    f"payload rows {len(result['faces'])}"
                )
            payload = {
                "schema": FACE_DUMP_SCHEMA,
                "map": map_name,
                "source": "exact_hull1_boundary",
                "result": result,
            }
            path = out_dir / f"hullfaces_{map_name}.json"
            path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
            written.append(path)
            print(
                f"{map_name}: {result['count']} faces, "
                f"{result['vertex_count']} vertices -> {path}"
            )
        finally:
            query_worker.close()
    return written


@dataclass(frozen=True)
class MapGeometry:
    name: str
    centers: torch.Tensor       # (N, 3), world coordinates
    normals: torch.Tensor       # (N, 3)
    extents: torch.Tensor       # (N, 3), half AABB extent
    vertices: torch.Tensor      # (N, V, 3), world coordinates
    vertex_mask: torch.Tensor   # (N, V), True at real vertices
    bounds_min: torch.Tensor    # (3,)
    bounds_span: torch.Tensor   # (3,), never zero

    def to(self, device: torch.device) -> "MapGeometry":
        return MapGeometry(
            name=self.name,
            centers=self.centers.to(device),
            normals=self.normals.to(device),
            extents=self.extents.to(device),
            vertices=self.vertices.to(device),
            vertex_mask=self.vertex_mask.to(device),
            bounds_min=self.bounds_min.to(device),
            bounds_span=self.bounds_span.to(device),
        )


def load_face_dump(path: Path) -> MapGeometry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["schema"]) != FACE_DUMP_SCHEMA:
        raise ValueError(
            f"{path}: face schema {payload['schema']} != {FACE_DUMP_SCHEMA}"
        )
    result = payload["result"]
    faces = result["faces"]
    if not faces:
        raise ValueError(f"{path}: empty hull-face table")
    if int(result["count"]) != len(faces):
        raise ValueError(f"{path}: declared face count does not match payload")

    max_verts = max(len(face["verts"]) for face in faces)
    centers = np.empty((len(faces), 3), dtype=np.float32)
    normals = np.empty_like(centers)
    extents = np.empty_like(centers)
    vertices = np.zeros((len(faces), max_verts, 3), dtype=np.float32)
    vertex_mask = np.zeros((len(faces), max_verts), dtype=bool)
    for i, face in enumerate(faces):
        verts = np.asarray(face["verts"], dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
            raise ValueError(f"{path}: face {i} has invalid vertices {verts.shape}")
        normal = np.asarray(face["normal"], dtype=np.float32)
        if not np.isfinite(normal).all() or not np.isclose(
            np.linalg.norm(normal), 1.0, atol=2e-3,
        ):
            raise ValueError(f"{path}: face {i} has non-unit normal")
        mins = np.asarray(face["mins"], dtype=np.float32)
        maxs = np.asarray(face["maxs"], dtype=np.float32)
        centers[i] = verts.mean(axis=0)
        normals[i] = normal
        extents[i] = 0.5 * (maxs - mins)
        vertices[i, : len(verts)] = verts
        vertex_mask[i, : len(verts)] = True

    bounds_min_np = vertices[vertex_mask].reshape(-1, 3).min(axis=0)
    bounds_max_np = vertices[vertex_mask].reshape(-1, 3).max(axis=0)
    bounds_span_np = np.maximum(bounds_max_np - bounds_min_np, 1.0)
    return MapGeometry(
        name=canonical_map_name(payload["map"]),
        centers=torch.from_numpy(centers),
        normals=torch.from_numpy(normals),
        extents=torch.from_numpy(extents),
        vertices=torch.from_numpy(vertices),
        vertex_mask=torch.from_numpy(vertex_mask),
        bounds_min=torch.from_numpy(bounds_min_np),
        bounds_span=torch.from_numpy(bounds_span_np),
    )


def load_static_records(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    by_map: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("schema", 0)) < 8:
                    raise ValueError(
                        f"{path}: schema {row.get('schema')} lacks static atlas fields"
                    )
                for field in ("origin", "view_yaw", "static_atlas_code",
                              "static_atlas_distance"):
                    if field not in row:
                        raise ValueError(f"{path}: record lacks {field}")
                by_map.setdefault(canonical_map_name(row["map"]), []).append(row)
    if not by_map:
        raise ValueError("no static-memory records loaded")
    return by_map


def split_records(
    records: list[dict[str, Any]], *, block_size: int = 10, val_mod: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Block split keeps temporally adjacent records on the same side."""
    train, val = [], []
    for i, record in enumerate(records):
        (val if (i // block_size) % val_mod == 0 else train).append(record)
    if not train or not val:
        raise ValueError(
            f"block split produced train={len(train)} val={len(val)} records"
        )
    return train, val


def _fourier(x: torch.Tensor, bands: int = 4) -> torch.Tensor:
    frequencies = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype))
    phase = math.pi * x.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)


@dataclass(frozen=True)
class MapMemory:
    keys: tuple[torch.Tensor, ...]     # each (H, N, Dh)
    values: tuple[torch.Tensor, ...]   # each (H, N, Dh)
    centers: torch.Tensor              # (N, 3), world coordinates
    normals: torch.Tensor              # (N, 3)
    bounds_min: torch.Tensor           # (3,)
    bounds_span: torch.Tensor          # (3,)


class StaticCrossBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ffn: int, position_mode: str,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not divisible by {n_heads}")
        if position_mode not in POSITION_MODES:
            raise ValueError(
                f"position_mode {position_mode!r}, expected one of {POSITION_MODES}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.position_mode = position_mode
        self.query_norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model),
        )
        if position_mode == "relative_bias":
            self.bias_mlp = nn.Sequential(
                nn.Linear(5, 32), nn.GELU(), nn.Linear(32, n_heads),
            )

    def project_map(self, face_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = face_tokens.shape[0]
        key = self.key_proj(face_tokens).view(n, self.n_heads, self.d_head)
        value = self.value_proj(face_tokens).view(n, self.n_heads, self.d_head)
        return key.permute(1, 0, 2), value.permute(1, 0, 2)

    def _relative_bias(
        self, memory: MapMemory, origins: torch.Tensor, directions: torch.Tensor,
    ) -> torch.Tensor:
        scale = memory.bounds_span.max()
        rel = (memory.centers.unsqueeze(0) - origins.unsqueeze(1)) / scale
        along = torch.sum(rel * directions.unsqueeze(1), dim=-1)
        perp = torch.sqrt(torch.clamp(
            torch.sum(rel * rel, dim=-1) - along.square(), min=0.0,
        ) + 1e-8)
        distance = torch.linalg.vector_norm(rel, dim=-1)
        facing = -torch.sum(
            memory.normals.unsqueeze(0) * directions.unsqueeze(1), dim=-1,
        )
        in_front = torch.tanh(4.0 * along)
        features = torch.stack([along, perp, distance, facing, in_front], dim=-1)
        return self.bias_mlp(features).permute(2, 0, 1)

    def forward(
        self, query: torch.Tensor, memory: MapMemory, layer: int,
        origins: torch.Tensor, directions: torch.Tensor,
    ) -> torch.Tensor:
        qn = self.query_norm(query)
        q = self.query_proj(qn).view(-1, self.n_heads, self.d_head)
        q = q.permute(1, 0, 2)
        scores = torch.einsum("hqd,hnd->hqn", q, memory.keys[layer])
        scores = scores / math.sqrt(self.d_head)
        if self.position_mode == "relative_bias":
            scores = scores + self._relative_bias(memory, origins, directions)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("hqn,hnd->hqd", weights, memory.values[layer])
        attended = attended.permute(1, 0, 2).reshape(-1, self.d_model)
        query = query + self.out_proj(attended)
        return query + self.ffn(self.ffn_norm(query))


class StaticMapAtlasDecoder(nn.Module):
    """Face table -> cached K/V; pose/ray query -> one 4-bit atlas code."""

    def __init__(
        self, *, d_model: int = 96, n_heads: int = 4, n_layers: int = 2,
        d_ffn: int = 192, position_mode: str = "absolute",
    ) -> None:
        super().__init__()
        if position_mode not in POSITION_MODES:
            raise ValueError(
                f"position_mode {position_mode!r}, expected one of {POSITION_MODES}"
            )
        self.d_model = int(d_model)
        self.position_mode = position_mode
        vertex_in = 3 + 3 * 2 * 4
        self.vertex_encoder = nn.Sequential(
            nn.Linear(vertex_in, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        face_in = 3 * 2 * 4 + 3 + 3 + d_model
        self.face_encoder = nn.Sequential(
            nn.Linear(face_in, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        query_in = 3 + 3 * 2 * 4 + 3 + 1
        self.query_encoder = nn.Sequential(
            nn.Linear(query_in, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            StaticCrossBlock(d_model, n_heads, d_ffn, position_mode)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 16)

    def _normalize(self, points: torch.Tensor, geometry: MapGeometry) -> torch.Tensor:
        return 2.0 * (points - geometry.bounds_min) / geometry.bounds_span - 1.0

    def encode_map(self, geometry: MapGeometry) -> MapMemory:
        centers = self._normalize(geometry.centers, geometry)
        verts = self._normalize(geometry.vertices, geometry)
        vertex_input = torch.cat([verts, _fourier(verts)], dim=-1)
        vertex_encoded = self.vertex_encoder(vertex_input)
        mask = geometry.vertex_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1).to(vertex_encoded.dtype)
        vertex_mean = (vertex_encoded * mask).sum(dim=1) / denom
        neg_inf = torch.finfo(vertex_encoded.dtype).min
        vertex_max = vertex_encoded.masked_fill(~mask, neg_inf).max(dim=1).values
        extents = 2.0 * geometry.extents / geometry.bounds_span
        face_input = torch.cat([
            _fourier(centers), geometry.normals, extents,
            vertex_mean, vertex_max,
        ], dim=-1)
        face_tokens = self.face_encoder(face_input)
        projected = [block.project_map(face_tokens) for block in self.blocks]
        return MapMemory(
            keys=tuple(pair[0] for pair in projected),
            values=tuple(pair[1] for pair in projected),
            centers=geometry.centers,
            normals=geometry.normals,
            bounds_min=geometry.bounds_min,
            bounds_span=geometry.bounds_span,
        )

    def forward_queries(
        self, geometry: MapGeometry, memory: MapMemory, origins: torch.Tensor,
        directions: torch.Tensor, max_distances: torch.Tensor,
    ) -> torch.Tensor:
        origin_norm = self._normalize(origins, geometry)
        scale = geometry.bounds_span.max()
        query_input = torch.cat([
            origin_norm, _fourier(origin_norm), directions,
            (max_distances / scale).unsqueeze(-1),
        ], dim=-1)
        query = self.query_encoder(query_input)
        for layer, block in enumerate(self.blocks):
            query = block(query, memory, layer, origins, directions)
        return self.classifier(self.final_norm(query))


def _record_rays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    elevations, yaw_count, limits = _record_limits(record)
    view_yaw = float(record["view_yaw"])
    directions = np.empty((len(elevations), yaw_count, 3), dtype=np.float32)
    for ei, elevation in enumerate(elevations):
        elev = math.radians(elevation)
        yaws = np.radians(view_yaw + np.arange(yaw_count) * (360.0 / yaw_count))
        directions[ei, :, 0] = math.cos(elev) * np.cos(yaws)
        directions[ei, :, 1] = math.cos(elev) * np.sin(yaws)
        directions[ei, :, 2] = math.sin(elev)
    return directions.reshape(-1, 3), limits.reshape(-1).astype(np.float32)


def _triangulate(geometry: MapGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Fan-triangulate convex carve faces for the deterministic ceiling."""
    vertices = geometry.vertices.cpu().numpy()
    mask = geometry.vertex_mask.cpu().numpy()
    normals = geometry.normals.cpu().numpy()
    v0, v1, v2, tri_normals, tri_faces = [], [], [], [], []
    for face_i in range(len(vertices)):
        polygon = vertices[face_i, mask[face_i]]
        for j in range(1, len(polygon) - 1):
            v0.append(polygon[0])
            v1.append(polygon[j])
            v2.append(polygon[j + 1])
            tri_normals.append(normals[face_i])
            tri_faces.append(face_i)
    return tuple(np.asarray(values) for values in (v0, v1, v2, tri_normals,
                                                    tri_faces))


def _nearest_face_indices(geometry: MapGeometry, origin: np.ndarray, count: int) -> np.ndarray:
    vertices = geometry.vertices.cpu().numpy()
    mask = geometry.vertex_mask.cpu().numpy()
    mins = np.where(mask[..., None], vertices, np.inf).min(axis=1)
    maxs = np.where(mask[..., None], vertices, -np.inf).max(axis=1)
    delta = np.maximum(np.maximum(mins - origin, origin - maxs), 0.0)
    distance = np.linalg.norm(delta, axis=-1)
    return np.argsort(distance, kind="stable")[:count]


def raycast_faces(
    geometry: MapGeometry, origin: np.ndarray, directions: np.ndarray,
    limits: np.ndarray, *, face_limit: int | None, ray_chunk: int = 128,
) -> np.ndarray:
    """Exact triangle intersections using a static face table.

    ``face_limit`` applies one pose-level nearest-AABB gather before any ray
    queries.  This is the deterministic information ceiling for a cached
    local-token route; ``None`` uses the entire immutable map table.
    """
    v0, v1, v2, tri_normals, tri_faces = _triangulate(geometry)
    if face_limit is not None:
        if face_limit <= 0:
            raise ValueError(f"face_limit must be positive or None, got {face_limit}")
        chosen = _nearest_face_indices(geometry, origin, face_limit)
        keep = np.isin(tri_faces, chosen)
        v0, v1, v2 = v0[keep], v1[keep], v2[keep]
        tri_normals = tri_normals[keep]
    edge1 = v1 - v0
    edge2 = v2 - v0
    s = origin[None, :] - v0
    qvec = np.cross(s, edge1)
    t_numerator = np.sum(edge2 * qvec, axis=-1)
    out = np.asarray(limits, dtype=np.float64).copy()
    epsilon = 1e-5
    for start in range(0, len(directions), ray_chunk):
        end = min(start + ray_chunk, len(directions))
        direction = directions[start:end].astype(np.float64, copy=False)
        h = np.cross(direction[:, None, :], edge2[None, :, :])
        a = np.sum(edge1[None, :, :] * h, axis=-1)
        safe = np.abs(a) > epsilon
        inv_a = np.zeros_like(a)
        np.divide(1.0, a, out=inv_a, where=safe)
        u = inv_a * np.sum(s[None, :, :] * h, axis=-1)
        v = inv_a * np.sum(direction[:, None, :] * qvec[None, :, :], axis=-1)
        t = inv_a * t_numerator[None, :]
        front = direction @ tri_normals.T < -epsilon
        max_t = limits[start:end, None]
        valid = (
            safe & front & (u >= -epsilon) & (v >= -epsilon)
            & (u + v <= 1.0 + epsilon) & (t >= -epsilon) & (t <= max_t)
        )
        nearest = np.min(np.where(valid, np.maximum(t, 0.0), np.inf), axis=1)
        hit = np.isfinite(nearest)
        out[start:end][hit] = nearest[hit]
    return out


def _quantized_prediction(distances: np.ndarray, limits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hit = distances < limits - 1e-4
    midpoints = 0.5 * (_LEVELS[:-1] + _LEVELS[1:])
    codes = np.searchsorted(midpoints, distances, side="right").astype(np.int64)
    codes = np.where(hit, codes, ATLAS_MISS_CODE)
    decoded = np.minimum(_LEVELS[np.minimum(codes, len(_LEVELS) - 1)], limits)
    return codes, np.where(hit, decoded, limits)


def oracle_routing(
    *, faces_dir: Path, sidecars: list[Path], face_limits: list[int],
    output: Path, ray_chunk: int,
) -> dict[str, Any]:
    """Score exact static geometry after a bounded pose-level token gather."""
    records = load_static_records(sidecars)
    geometries = {
        geometry.name: geometry
        for geometry in map(load_face_dump, sorted(faces_dir.glob("hullfaces_*.json")))
    }
    missing = sorted(set(records) - geometries.keys())
    if missing:
        raise ValueError(f"sidecar maps lack face dumps: {missing}")
    arms: list[tuple[str, int | None]] = [(f"nearest_{n}", n) for n in face_limits]
    arms.append(("full_map", None))
    collected: dict[str, dict[str, list[np.ndarray] | int]] = {
        name: {"truth": [], "prediction": [], "limits": [],
               "labels": [], "codes": [], "elapsed_ns": 0}
        for name, _ in arms
    }
    for map_name, map_records in records.items():
        geometry = geometries[map_name]
        print(f"{map_name}: {len(map_records)} records, {len(geometry.centers)} faces")
        for record_i, record in enumerate(map_records):
            directions, limits = _record_rays(record)
            origin = np.asarray(record["origin"], dtype=np.float64)
            static_dist = np.asarray(record["static_atlas_distance"], dtype=np.float64).reshape(-1)
            truth = np.where(static_dist >= 0.0, static_dist, limits)
            labels = np.asarray(record["static_atlas_code"], dtype=np.int64).reshape(-1)
            for arm_name, face_limit in arms:
                started = time.perf_counter_ns()
                distance = raycast_faces(
                    geometry, origin, directions, limits,
                    face_limit=face_limit, ray_chunk=ray_chunk,
                )
                elapsed = time.perf_counter_ns() - started
                codes, prediction = _quantized_prediction(distance, limits)
                bucket = collected[arm_name]
                bucket["truth"].append(truth)
                bucket["prediction"].append(prediction)
                bucket["limits"].append(limits)
                bucket["labels"].append(labels)
                bucket["codes"].append(codes)
                bucket["elapsed_ns"] += elapsed
            if (record_i + 1) % 25 == 0:
                print(f"  {record_i + 1}/{len(map_records)}", flush=True)

    results: dict[str, Any] = {}
    for arm_name, _ in arms:
        bucket = collected[arm_name]
        truth = np.concatenate(bucket["truth"])
        prediction = np.concatenate(bucket["prediction"])
        limits = np.concatenate(bucket["limits"])
        labels = np.concatenate(bucket["labels"])
        codes = np.concatenate(bucket["codes"])
        plane = _metrics(truth, prediction, limits)
        summary = {"plane": plane, "by_elevation": {"0": plane}}
        failures = threshold_failures(summary, **DEFAULT_GATE)
        results[arm_name] = {
            "code_accuracy": float(np.mean(labels == codes)),
            "plane": plane,
            "elapsed_seconds": float(bucket["elapsed_ns"]) / 1e9,
            "gate": {**DEFAULT_GATE, "passed": not failures, "failures": failures},
        }
    report = {
        "schema": 1,
        "face_limits": face_limits,
        "records": sum(len(rows) for rows in records.values()),
        "arms": results,
    }
    write_json(output, report)
    return report


def sample_queries(
    records: list[dict[str, Any]], count: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_indices = rng.integers(0, len(records), size=count)
    first_codes = np.asarray(records[0]["static_atlas_code"])
    rays_per_record = int(first_codes.size)
    ray_indices = rng.integers(0, rays_per_record, size=count)
    origins = np.empty((count, 3), dtype=np.float32)
    directions = np.empty_like(origins)
    limits = np.empty(count, dtype=np.float32)
    labels = np.empty(count, dtype=np.int64)
    ray_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for out_i, (record_i, ray_i) in enumerate(zip(record_indices, ray_indices)):
        record_i = int(record_i)
        record = records[record_i]
        if record_i not in ray_cache:
            ray_cache[record_i] = _record_rays(record)
        ray_dirs, ray_limits = ray_cache[record_i]
        origins[out_i] = record["origin"]
        directions[out_i] = ray_dirs[ray_i]
        limits[out_i] = ray_limits[ray_i]
        labels[out_i] = np.asarray(record["static_atlas_code"]).reshape(-1)[ray_i]
    return origins, directions, limits, labels


def _tensor_digest(memory: MapMemory) -> str:
    digest = hashlib.sha256()
    tensors = (*memory.keys, *memory.values, memory.centers, memory.normals,
               memory.bounds_min, memory.bounds_span)
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(
    model: StaticMapAtlasDecoder, geometries: dict[str, MapGeometry],
    records: dict[str, list[dict[str, Any]]], *, device: torch.device,
    ray_chunk: int,
) -> dict[str, Any]:
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    limits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    predicted_codes_all: list[np.ndarray] = []
    per_elevation: dict[str, dict[str, list[np.ndarray]]] = {}
    map_reports: dict[str, Any] = {}

    with torch.no_grad():
        for map_name, map_records in records.items():
            geometry = geometries[map_name]
            _sync(device)
            started = time.perf_counter()
            memory = model.encode_map(geometry)
            _sync(device)
            encode_ms = 1000.0 * (time.perf_counter() - started)
            before = _tensor_digest(memory)
            query_times: list[float] = []
            map_correct = 0
            map_samples = 0
            for record in map_records:
                ray_dirs, ray_limits = _record_rays(record)
                origin = np.asarray(record["origin"], dtype=np.float32)
                chunks: list[np.ndarray] = []
                for start in range(0, len(ray_dirs), ray_chunk):
                    end = min(start + ray_chunk, len(ray_dirs))
                    origins = torch.from_numpy(
                        np.broadcast_to(origin, (end - start, 3)).copy()
                    ).to(device)
                    directions = torch.from_numpy(ray_dirs[start:end]).to(device)
                    max_distances = torch.from_numpy(ray_limits[start:end]).to(device)
                    _sync(device)
                    query_started = time.perf_counter()
                    logits = model.forward_queries(
                        geometry, memory, origins, directions, max_distances,
                    )
                    _sync(device)
                    query_times.append(1000.0 * (time.perf_counter() - query_started))
                    chunks.append(logits.argmax(dim=-1).cpu().numpy())
                predicted_codes = np.concatenate(chunks).reshape(
                    np.asarray(record["static_atlas_code"]).shape
                )
                target_codes = np.asarray(record["static_atlas_code"], dtype=np.int64)
                static_dist = np.asarray(
                    record["static_atlas_distance"], dtype=np.float64,
                )
                _, _, record_limits = _record_limits(record)
                target_truth = np.where(static_dist >= 0.0, static_dist, record_limits)
                hit = predicted_codes != ATLAS_MISS_CODE
                decoded = np.minimum(
                    _LEVELS[np.minimum(predicted_codes, len(_LEVELS) - 1)],
                    record_limits,
                )
                prediction = np.where(hit, decoded, record_limits)
                truths.append(target_truth.reshape(-1))
                predictions.append(prediction.reshape(-1))
                limits_all.append(record_limits.reshape(-1))
                labels_all.append(target_codes.reshape(-1))
                predicted_codes_all.append(predicted_codes.reshape(-1))
                map_correct += int(np.count_nonzero(predicted_codes == target_codes))
                map_samples += int(target_codes.size)
                for ei, elevation in enumerate(record["elevations"]):
                    bucket = per_elevation.setdefault(
                        f"{float(elevation):g}",
                        {"truth": [], "prediction": [], "limits": []},
                    )
                    bucket["truth"].append(target_truth[ei])
                    bucket["prediction"].append(prediction[ei])
                    bucket["limits"].append(record_limits[ei])
            after = _tensor_digest(memory)
            memory_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (*memory.keys, *memory.values)
            )
            map_reports[map_name] = {
                "faces": int(geometry.centers.shape[0]),
                "records": len(map_records),
                "code_accuracy": map_correct / map_samples,
                "map_encode_ms": encode_ms,
                "cached_kv_bytes": memory_bytes,
                "query_chunk": ray_chunk,
                "query_chunk_ms_p50": float(np.percentile(query_times, 50)),
                "query_chunk_ms_p95": float(np.percentile(query_times, 95)),
                "cache_digest": before,
                "cache_immutable": before == after,
            }

    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    limits = np.concatenate(limits_all)
    labels = np.concatenate(labels_all)
    predicted_codes = np.concatenate(predicted_codes_all)
    plane = _metrics(truth, prediction, limits)
    summary = {
        "records": sum(len(rows) for rows in records.values()),
        "code_accuracy": float(np.mean(labels == predicted_codes)),
        "plane": plane,
        "by_elevation": {
            elevation: _metrics(
                np.concatenate(bucket["truth"]),
                np.concatenate(bucket["prediction"]),
                np.concatenate(bucket["limits"]),
            )
            for elevation, bucket in per_elevation.items()
        },
        "maps": map_reports,
    }
    failures = threshold_failures(summary, **DEFAULT_GATE)
    summary["gate"] = {
        **DEFAULT_GATE,
        "passed": not failures,
        "failures": failures,
    }
    return summary


def _class_weights(records: Iterable[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = np.zeros(16, dtype=np.float64)
    for record in records:
        counts += np.bincount(
            np.asarray(record["static_atlas_code"]).reshape(-1), minlength=16,
        )
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"teacher corpus has no samples for atlas codes {missing}")
    weights = np.sqrt(counts.sum() / counts)
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def train(
    *, faces_dir: Path, sidecars: list[Path], output_dir: Path,
    position_mode: str, device_name: str, steps: int, batch_rays: int,
    ray_chunk: int, learning_rate: float, d_model: int, n_heads: int,
    n_layers: int, d_ffn: int, seed: int, eval_every: int,
) -> dict[str, Any]:
    if position_mode not in POSITION_MODES:
        raise ValueError(
            f"position_mode {position_mode!r}, expected one of {POSITION_MODES}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but torch.cuda is unavailable")

    all_records = load_static_records(sidecars)
    face_paths = sorted(faces_dir.glob("hullfaces_*.json"))
    if not face_paths:
        raise FileNotFoundError(f"no hullfaces_*.json under {faces_dir}")
    geometries_cpu = {g.name: g for g in map(load_face_dump, face_paths)}
    missing = sorted(set(all_records) - geometries_cpu.keys())
    if missing:
        raise ValueError(f"sidecar maps lack face dumps: {missing}")
    geometries = {name: geometries_cpu[name].to(device) for name in all_records}
    train_records: dict[str, list[dict[str, Any]]] = {}
    val_records: dict[str, list[dict[str, Any]]] = {}
    for name, rows in all_records.items():
        train_records[name], val_records[name] = split_records(rows)

    model = StaticMapAtlasDecoder(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ffn=d_ffn, position_mode=position_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(
        weight=_class_weights(
            (row for rows in train_records.values() for row in rows), device,
        )
    )
    rng = np.random.default_rng(seed)
    maps = sorted(train_records)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")

    for step in range(1, steps + 1):
        model.train()
        map_name = maps[(step - 1) % len(maps)]
        geometry = geometries[map_name]
        origins_np, directions_np, limits_np, labels_np = sample_queries(
            train_records[map_name], batch_rays, rng,
        )
        origins = torch.from_numpy(origins_np).to(device)
        directions = torch.from_numpy(directions_np).to(device)
        limits = torch.from_numpy(limits_np).to(device)
        labels = torch.from_numpy(labels_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        memory = model.encode_map(geometry)
        logits = model.forward_queries(
            geometry, memory, origins, directions, limits,
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            # A fixed, small validation slice is enough for checkpoint
            # selection; the winning state receives a complete final eval.
            preview = {name: rows[: min(12, len(rows))]
                       for name, rows in val_records.items()}
            report = evaluate(
                model, geometries, preview, device=device, ray_chunk=ray_chunk,
            )
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "val_mae": report["plane"]["mae"],
                "val_code_accuracy": report["code_accuracy"],
                "val_gate_passed": report["gate"]["passed"],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_mae"] < best_mae:
                best_mae = row["val_mae"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

    assert best_state is not None
    model.load_state_dict(best_state)
    final = evaluate(
        model, geometries, val_records, device=device, ray_chunk=ray_chunk,
    )
    config = {
        "position_mode": position_mode,
        "device": str(device),
        "steps": steps,
        "batch_rays": batch_rays,
        "ray_chunk": ray_chunk,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "seed": seed,
        "split": {"block_size": 10, "val_mod": 5},
        "maps": maps,
    }
    result = {"schema": 1, "config": config, "history": history, "final": final}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "report.json", result)
    torch.save({"config": config, "model": best_state}, output_dir / "model.pt")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dump_parser = sub.add_parser("dump-faces", help="export immutable hull-1 faces")
    dump_parser.add_argument("--worker", required=True)
    dump_parser.add_argument("--demo-dir", type=Path, required=True)
    dump_parser.add_argument("--manifest", type=Path, required=True)
    dump_parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    dump_parser.add_argument("--maps", nargs="+", required=True)
    dump_parser.add_argument("--out-dir", type=Path, required=True)
    dump_parser.add_argument("--tick-hz", type=int, default=20)

    train_parser = sub.add_parser("train", help="fit and score the static-memory teacher")
    train_parser.add_argument("--faces-dir", type=Path, required=True)
    train_parser.add_argument("--sidecars", type=Path, nargs="+", required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--position-mode", choices=POSITION_MODES, required=True)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--batch-rays", type=int, default=256)
    train_parser.add_argument("--ray-chunk", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--d-model", type=int, default=96)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=2)
    train_parser.add_argument("--d-ffn", type=int, default=192)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--eval-every", type=int, default=250)

    oracle_parser = sub.add_parser(
        "oracle-routing", help="score bounded nearest-face gathers with exact intersections",
    )
    oracle_parser.add_argument("--faces-dir", type=Path, required=True)
    oracle_parser.add_argument("--sidecars", type=Path, nargs="+", required=True)
    oracle_parser.add_argument("--face-limits", default="16,32,64,128,256,512")
    oracle_parser.add_argument("--ray-chunk", type=int, default=128)
    oracle_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "dump-faces":
        dump_faces(
            worker_path=args.worker, demo_dir=args.demo_dir,
            manifest=args.manifest, asset_root=args.asset_root,
            maps=[canonical_map_name(name) for name in args.maps],
            out_dir=args.out_dir, tick_hz=args.tick_hz,
        )
        return 0

    if args.command == "oracle-routing":
        limits = [int(value) for value in args.face_limits.split(",")]
        if not limits or any(value <= 0 for value in limits):
            raise ValueError(f"face limits must be positive, got {limits}")
        report = oracle_routing(
            faces_dir=args.faces_dir, sidecars=args.sidecars,
            face_limits=limits, output=args.output, ray_chunk=args.ray_chunk,
        )
        print(json.dumps({
            name: {
                "mae": arm["plane"]["mae"],
                "code_accuracy": arm["code_accuracy"],
                "gate": arm["gate"]["passed"],
            }
            for name, arm in report["arms"].items()
        }, indent=2))
        return 0

    result = train(
        faces_dir=args.faces_dir, sidecars=args.sidecars,
        output_dir=args.output_dir, position_mode=args.position_mode,
        device_name=args.device, steps=args.steps, batch_rays=args.batch_rays,
        ray_chunk=args.ray_chunk, learning_rate=args.learning_rate,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ffn=args.d_ffn, seed=args.seed, eval_every=args.eval_every,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "gate": result["final"]["gate"],
        "mae": result["final"]["plane"]["mae"],
        "code_accuracy": result["final"]["code_accuracy"],
    }, indent=2))
    return 0 if result["final"]["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
