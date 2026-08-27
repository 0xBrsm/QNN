"""Test a load-time directional map field with pose-time index routing.

Each navmesh probe owns eleven world-anchored panoramic band tokens.  A band
is stored as Fourier coefficients computed once at map load; the coefficients
are the immutable value memory.  A moving ray query selects K nearby probes,
selects the matching elevation band, evaluates the cached coefficients at the
world-space ray yaw, and learns only the selective fusion across donors.

The pose route therefore gathers indices into already-built map memory.  It
does not roll panoramas, fuse probes, or project map tokens each tick.  One
probe is one directional-function token containing all eleven bands; a ray
query evaluates one band and attends over K probe values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from qnn.bc.probe_atlas import decode_dump
from qnn.diag.spatial_reconstruction import (
    _metrics,
    _record_limits,
    threshold_failures,
)
from qnn.diag.probe_reconstruction import (
    reconstruct_from_probes,
    reconstruct_shift,
)
from qnn.diag.static_map_memory import (
    DEFAULT_GATE,
    _LEVELS,
    _fourier,
    _quantized_prediction,
    _record_rays,
    _sync,
    canonical_map_name,
    load_static_records,
    split_records,
)
from qnn.engine_norm import ATLAS_MISS_CODE
from qnn.utils.io import write_json


@dataclass(frozen=True)
class ProbeTable:
    name: str
    positions: np.ndarray       # (P, 3), world-space probe viewpoints
    codes: np.ndarray           # (P, 11, 72), world-yaw anchored


@dataclass(frozen=True)
class ProbeMemory:
    positions: torch.Tensor     # (P, 3)
    coefficients: torch.Tensor  # (P, 11, 2, F): depth ratio, hit
    bounds_min: torch.Tensor    # (3,)
    bounds_span: torch.Tensor   # (3,)
    harmonics: int


@dataclass(frozen=True)
class QuerySet:
    origins: np.ndarray
    directions: np.ndarray
    limits: np.ndarray
    labels: np.ndarray
    truth: np.ndarray
    band_indices: np.ndarray
    neighbors: np.ndarray

    def __len__(self) -> int:
        return int(len(self.labels))


def load_probe_table(path: Path) -> ProbeTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    positions, codes = decode_dump(payload)
    name = canonical_map_name(path.stem.removeprefix("navatlas_"))
    if int(payload["count"]) != len(positions):
        raise ValueError(f"{path}: declared probe count does not match payload")
    return ProbeTable(name=name, positions=positions, codes=codes)


def load_probe_tables(directory: Path) -> dict[str, ProbeTable]:
    paths = sorted(directory.glob("navatlas_*.json"))
    if not paths:
        raise FileNotFoundError(f"no navatlas_*.json under {directory}")
    return {table.name: table for table in map(load_probe_table, paths)}


def harmonic_basis_numpy(yaw_rad: np.ndarray, harmonics: int) -> np.ndarray:
    """Real periodic basis; H=36 spans every sample on a 72-cell ring."""
    if harmonics <= 0 or harmonics > 36:
        raise ValueError(f"harmonics must be in 1..36, got {harmonics}")
    yaw = np.asarray(yaw_rad, dtype=np.float64).reshape(-1)
    columns = [np.ones_like(yaw)]
    for frequency in range(1, harmonics + 1):
        columns.append(np.cos(frequency * yaw))
        # Nyquist sine is identically zero on the 72-cell training grid.
        if frequency < 36:
            columns.append(np.sin(frequency * yaw))
    return np.stack(columns, axis=-1)


def harmonic_basis_torch(yaw_rad: torch.Tensor, harmonics: int) -> torch.Tensor:
    columns = [torch.ones_like(yaw_rad)]
    for frequency in range(1, harmonics + 1):
        columns.append(torch.cos(frequency * yaw_rad))
        if frequency < 36:
            columns.append(torch.sin(frequency * yaw_rad))
    return torch.stack(columns, dim=-1)


def encode_probe_memory(
    table: ProbeTable,
    band_limits: np.ndarray,
    *,
    harmonics: int,
    device: torch.device,
) -> ProbeMemory:
    if table.codes.shape[1] != len(band_limits):
        raise ValueError(
            f"{table.name}: {table.codes.shape[1]} bands != {len(band_limits)} limits"
        )
    code_idx = np.minimum(table.codes.astype(np.int64), len(_LEVELS) - 1)
    decoded = np.minimum(_LEVELS[code_idx], band_limits[None, :, None])
    hit = table.codes != ATLAS_MISS_CODE
    depth_ratio = np.where(
        hit, decoded / band_limits[None, :, None], 1.0,
    )
    signal = np.stack([depth_ratio, hit.astype(np.float64)], axis=-1)
    grid_yaw = np.arange(table.codes.shape[-1], dtype=np.float64)
    grid_yaw *= 2.0 * math.pi / table.codes.shape[-1]
    basis = harmonic_basis_numpy(grid_yaw, harmonics)
    coefficients = np.einsum(
        "fy,pbyc->pbfc", np.linalg.pinv(basis), signal, optimize=True,
    ).transpose(0, 1, 3, 2)
    mins = table.positions.min(axis=0)
    span = np.maximum(table.positions.max(axis=0) - mins, 1.0)
    return ProbeMemory(
        positions=torch.from_numpy(table.positions.astype(np.float32)).to(device),
        coefficients=torch.from_numpy(coefficients.astype(np.float32)).to(device),
        bounds_min=torch.from_numpy(mins.astype(np.float32)).to(device),
        bounds_span=torch.from_numpy(span.astype(np.float32)).to(device),
        harmonics=harmonics,
    )


def memory_digest(memory: ProbeMemory) -> str:
    digest = hashlib.sha256()
    for tensor in (
        memory.positions, memory.coefficients, memory.bounds_min,
        memory.bounds_span,
    ):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_query_set(
    records: list[dict[str, Any]], table: ProbeTable, *, k: int,
) -> QuerySet:
    if k <= 0 or k > len(table.positions):
        raise ValueError(f"k={k} outside 1..{len(table.positions)}")
    tree = cKDTree(table.positions)
    record_origins = np.asarray([row["origin"] for row in records], dtype=np.float32)
    _, record_neighbors = tree.query(record_origins, k=k)
    if k == 1:
        record_neighbors = record_neighbors[:, None]

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    limits_all: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    neighbors: list[np.ndarray] = []
    for record_i, record in enumerate(records):
        ray_directions, limits = _record_rays(record)
        codes = np.asarray(record["static_atlas_code"], dtype=np.int64)
        static_distance = np.asarray(
            record["static_atlas_distance"], dtype=np.float64,
        )
        origins.append(np.broadcast_to(record_origins[record_i], ray_directions.shape))
        directions.append(ray_directions)
        limits_all.append(limits)
        labels.append(codes.reshape(-1))
        truth.append(np.where(
            static_distance.reshape(-1) >= 0.0,
            static_distance.reshape(-1), limits,
        ))
        band_count, yaw_count = codes.shape
        bands.append(np.repeat(np.arange(band_count), yaw_count))
        neighbors.append(np.broadcast_to(
            record_neighbors[record_i], (len(ray_directions), k),
        ))
    return QuerySet(
        origins=np.concatenate(origins).astype(np.float32, copy=False),
        directions=np.concatenate(directions).astype(np.float32, copy=False),
        limits=np.concatenate(limits_all).astype(np.float32, copy=False),
        labels=np.concatenate(labels),
        truth=np.concatenate(truth),
        band_indices=np.concatenate(bands).astype(np.int64, copy=False),
        neighbors=np.concatenate(neighbors).astype(np.int64, copy=False),
    )


class ProbeFusionDecoder(nn.Module):
    """Ray query cross-attention over K cached directional-field values."""

    def __init__(self, d_model: int = 96) -> None:
        super().__init__()
        query_dim = 3 + 3 * 2 * 4 + 3 + 1
        donor_dim = 10
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, 16),
        )

    def forward(
        self,
        memory: ProbeMemory,
        origins: torch.Tensor,
        directions: torch.Tensor,
        limits: torch.Tensor,
        band_indices: torch.Tensor,
        neighbors: torch.Tensor,
        *,
        return_router: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        yaw = torch.atan2(directions[:, 1], directions[:, 0])
        basis = harmonic_basis_torch(yaw, memory.harmonics)
        selected = memory.coefficients[neighbors, band_indices[:, None]]
        sampled = torch.einsum("qkcf,qf->qkc", selected, basis)
        depth_ratio = sampled[..., 0].clamp(0.0, 1.0)
        hit = sampled[..., 1].clamp(0.0, 1.0)

        probe_positions = memory.positions[neighbors]
        relative = probe_positions - origins[:, None, :]
        along_world = torch.sum(relative * directions[:, None, :], dim=-1)
        along = along_world / limits[:, None].clamp(min=1.0)
        relative_sq = torch.sum(relative.square(), dim=-1)
        perp = torch.sqrt(torch.clamp(
            relative_sq - along_world.square(), min=0.0,
        ) + 1e-8) / 512.0
        distance = torch.sqrt(relative_sq + 1e-8) / 512.0
        corrected = (depth_ratio + along).clamp(0.0, 1.0)
        rank = torch.arange(
            neighbors.shape[1], device=origins.device, dtype=origins.dtype,
        ) / max(neighbors.shape[1] - 1, 1)
        candidate_depth = torch.stack([depth_ratio, corrected], dim=2)
        candidate_hit = hit.unsqueeze(-1).expand(-1, -1, 2)
        variant = torch.tensor(
            [0.0, 1.0], device=origins.device, dtype=origins.dtype,
        ).view(1, 1, 2).expand_as(candidate_depth)
        candidate_count = 2 * neighbors.shape[1]
        donor_input = torch.cat([
            candidate_depth.unsqueeze(-1), candidate_hit.unsqueeze(-1),
            variant.unsqueeze(-1),
            along.unsqueeze(-1).unsqueeze(2).expand(-1, -1, 2, -1),
            perp.unsqueeze(-1).unsqueeze(2).expand(-1, -1, 2, -1),
            (relative / 512.0).unsqueeze(2).expand(-1, -1, 2, -1),
            distance.unsqueeze(-1).unsqueeze(2).expand(-1, -1, 2, -1),
            rank.view(1, -1, 1, 1).expand(len(origins), -1, 2, -1),
        ], dim=-1).reshape(len(origins), candidate_count, -1)
        donor = self.donor_encoder(donor_input)

        origin_norm = 2.0 * (
            origins - memory.bounds_min
        ) / memory.bounds_span - 1.0
        query_input = torch.cat([
            origin_norm, _fourier(origin_norm), directions,
            (limits / memory.bounds_span.max()).unsqueeze(-1),
        ], dim=-1)
        query = self.query_encoder(query_input)
        scores = torch.einsum(
            "qd,qkd->qk", self.query_proj(query), self.key_proj(donor),
        ) / math.sqrt(query.shape[-1])
        candidate_distance = distance.unsqueeze(-1).expand(-1, -1, 2)
        scores = scores - candidate_distance.reshape(len(origins), candidate_count)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("qk,qkd->qd", weights, self.value_proj(donor))
        logits = self.classifier(torch.cat([query, pooled], dim=-1))
        if return_router:
            return (
                logits,
                scores,
                candidate_depth.reshape(len(origins), candidate_count),
                candidate_hit.reshape(len(origins), candidate_count),
            )
        return logits


def _batch(
    query_set: QuerySet, indices: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(query_set.origins[indices]).to(device),
        torch.from_numpy(query_set.directions[indices]).to(device),
        torch.from_numpy(query_set.limits[indices]).to(device),
        torch.from_numpy(query_set.band_indices[indices]).to(device),
        torch.from_numpy(query_set.neighbors[indices]).to(device),
        torch.from_numpy(query_set.labels[indices]).to(device),
        torch.from_numpy(query_set.truth[indices].astype(np.float32)).to(device),
    )


def _class_weights(query_sets: Iterable[QuerySet], device: torch.device) -> torch.Tensor:
    counts = np.zeros(16, dtype=np.float64)
    for query_set in query_sets:
        counts += np.bincount(query_set.labels, minlength=16)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def evaluate(
    model: ProbeFusionDecoder,
    memories: dict[str, ProbeMemory],
    query_sets: dict[str, QuerySet],
    *,
    device: torch.device,
    chunk: int,
    max_per_map: int | None = None,
) -> dict[str, Any]:
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    limits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    codes_all: list[np.ndarray] = []
    band_all: list[np.ndarray] = []
    map_reports: dict[str, Any] = {}
    with torch.no_grad():
        for name, query_set in query_sets.items():
            memory = memories[name]
            count = len(query_set) if max_per_map is None else min(
                len(query_set), max_per_map,
            )
            before = memory_digest(memory)
            chunks: list[np.ndarray] = []
            elapsed: list[float] = []
            for start in range(0, count, chunk):
                indices = np.arange(start, min(start + chunk, count))
                origins, directions, limits, bands, neighbors, _, _ = _batch(
                    query_set, indices, device,
                )
                _sync(device)
                started = time.perf_counter()
                logits = model(
                    memory, origins, directions, limits, bands, neighbors,
                )
                _sync(device)
                elapsed.append(1000.0 * (time.perf_counter() - started))
                chunks.append(logits.argmax(dim=-1).cpu().numpy())
            codes = np.concatenate(chunks)
            limits_np = query_set.limits[:count]
            hit = codes != ATLAS_MISS_CODE
            decoded = np.minimum(
                _LEVELS[np.minimum(codes, len(_LEVELS) - 1)], limits_np,
            )
            prediction = np.where(hit, decoded, limits_np)
            truths.append(query_set.truth[:count])
            predictions.append(prediction)
            limits_all.append(limits_np)
            labels_all.append(query_set.labels[:count])
            codes_all.append(codes)
            band_all.append(query_set.band_indices[:count])
            after = memory_digest(memory)
            map_reports[name] = {
                "probes": int(memory.positions.shape[0]),
                "rays": count,
                "probe_tokens_per_pose": int(query_set.neighbors.shape[1]),
                "band_functions_per_pose": int(query_set.neighbors.shape[1] * 11),
                "cached_field_bytes": int(
                    memory.coefficients.numel() * memory.coefficients.element_size()
                ),
                "query_chunk": chunk,
                "query_chunk_ms_p50": float(np.percentile(elapsed, 50)),
                "query_chunk_ms_p95": float(np.percentile(elapsed, 95)),
                "cache_digest": before,
                "cache_immutable": before == after,
            }

    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    limits = np.concatenate(limits_all)
    labels = np.concatenate(labels_all)
    codes = np.concatenate(codes_all)
    bands = np.concatenate(band_all)
    plane = _metrics(truth, prediction, limits)
    by_elevation = {
        str(band): _metrics(
            truth[bands == band], prediction[bands == band], limits[bands == band],
        )
        for band in np.unique(bands)
    }
    summary = {
        "rays": int(len(truth)),
        "code_accuracy": float(np.mean(labels == codes)),
        "plane": plane,
        "by_elevation": by_elevation,
        "maps": map_reports,
    }
    failures = threshold_failures(summary, **DEFAULT_GATE)
    summary["gate"] = {
        **DEFAULT_GATE, "passed": not failures, "failures": failures,
    }
    return summary


def oracle_routing(
    *, probe_dir: Path, sidecars: list[Path], output: Path, k: int,
) -> dict[str, Any]:
    """Target-informed ceiling for K routed static probe panoramas.

    ``best_k`` may choose, per ray, the closest target depth among every raw
    donor and its analytic first-order parallax correction.  It is not a
    runtime rule; it answers whether the routed immutable values contain a
    gate-grade answer before another learned-fusion run is justified.
    """
    records = load_static_records(sidecars)
    tables = load_probe_tables(probe_dir)
    quantile_arms = {
        "corrected_min": 0.0,
        "corrected_p25": 0.25,
        "corrected_median": 0.5,
        "corrected_p75": 0.75,
        "corrected_max": 1.0,
    }
    reprojection_arms = ("reproject", "hybrid", "hybrid_corrected")
    arms = (
        "nearest_shift", "nearest_corrected", *quantile_arms,
        *reprojection_arms, "best_k",
    )
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        arm: {"truth": [], "prediction": [], "limits": []} for arm in arms
    }
    exact_available: list[np.ndarray] = []
    nearest_stats: dict[str, Any] = {}
    total_rays = 0
    for name, rows in records.items():
        _, val_rows = split_records(rows)
        table = tables[name]
        query_set = build_query_set(val_rows, table, k=k)
        world_yaw = np.mod(
            np.arctan2(query_set.directions[:, 1], query_set.directions[:, 0]),
            2.0 * math.pi,
        )
        cells = np.rint(world_yaw * table.codes.shape[-1] / (2.0 * math.pi))
        cells = cells.astype(np.int64) % table.codes.shape[-1]
        donor_codes = table.codes[
            query_set.neighbors,
            query_set.band_indices[:, None],
            cells[:, None],
        ].astype(np.int64)
        donor_hit = donor_codes != ATLAS_MISS_CODE
        donor_depth = np.minimum(
            _LEVELS[np.minimum(donor_codes, len(_LEVELS) - 1)],
            query_set.limits[:, None],
        )
        raw_prediction = np.where(
            donor_hit, donor_depth, query_set.limits[:, None],
        )
        relative = (
            table.positions[query_set.neighbors]
            - query_set.origins[:, None, :]
        )
        along = np.sum(relative * query_set.directions[:, None, :], axis=-1)
        corrected_distance = np.where(
            donor_hit,
            np.clip(donor_depth + along, 0.0, query_set.limits[:, None]),
            query_set.limits[:, None],
        )
        corrected_prediction = np.empty_like(corrected_distance)
        for donor_i in range(k):
            _, corrected_prediction[:, donor_i] = _quantized_prediction(
                corrected_distance[:, donor_i], query_set.limits,
            )
        candidates = np.concatenate(
            [raw_prediction, corrected_prediction], axis=1,
        )
        best = np.argmin(
            np.abs(candidates - query_set.truth[:, None]), axis=1,
        )
        predictions: dict[str, np.ndarray] = {
            "nearest_shift": raw_prediction[:, 0],
            "nearest_corrected": corrected_prediction[:, 0],
            "best_k": candidates[np.arange(len(candidates)), best],
        }
        for arm, quantile in quantile_arms.items():
            distance = np.quantile(corrected_prediction, quantile, axis=1)
            _, predictions[arm] = _quantized_prediction(
                distance, query_set.limits,
            )
        for arm, prediction in predictions.items():
            collected[arm]["truth"].append(query_set.truth)
            collected[arm]["prediction"].append(prediction)
            collected[arm]["limits"].append(query_set.limits)

        tree = cKDTree(table.positions)
        for target in val_rows:
            _, near = tree.query(
                np.asarray(target["origin"], dtype=np.float64), k=k,
            )
            near = np.atleast_1d(near)
            donors = [{
                "origin": table.positions[int(index)],
                "view_yaw": 0.0,
                "yaw_step": target["yaw_step"],
                "elevations": target["elevations"],
                "max_horiz": target["max_horiz"],
                "max_vert": target["max_vert"],
                "atlas_code": table.codes[int(index)],
            } for index in near]
            reprojection, covered = reconstruct_from_probes(
                target, donors, layout="atlas", return_coverage=True,
            )
            shift = reconstruct_shift(
                target, donors[0], layout="atlas", corrected=False,
            )
            shift_corrected = reconstruct_shift(
                target, donors[0], layout="atlas", corrected=True,
            )
            _, _, record_limits = _record_limits(target)
            static_distance = np.asarray(
                target["static_atlas_distance"], dtype=np.float64,
            )
            record_truth = np.where(
                static_distance >= 0.0, static_distance, record_limits,
            ).reshape(-1)
            reprojection_predictions = {
                "reproject": reprojection,
                "hybrid": np.where(covered, reprojection, shift),
                "hybrid_corrected": np.where(
                    covered, reprojection, shift_corrected,
                ),
            }
            for arm, prediction in reprojection_predictions.items():
                collected[arm]["truth"].append(record_truth)
                collected[arm]["prediction"].append(prediction.reshape(-1))
                collected[arm]["limits"].append(record_limits.reshape(-1))
        exact_available.append(np.any(donor_codes == query_set.labels[:, None], axis=1))
        distances = np.linalg.norm(relative[:, 0, :], axis=-1)
        nearest_stats[name] = {
            "probes": len(table.positions),
            "mean": float(np.mean(distances)),
            "p90": float(np.percentile(distances, 90)),
            "max": float(np.max(distances)),
        }
        total_rays += len(query_set)

    results: dict[str, Any] = {}
    for arm, bucket in collected.items():
        truth = np.concatenate(bucket["truth"])
        prediction = np.concatenate(bucket["prediction"])
        limits = np.concatenate(bucket["limits"])
        plane = _metrics(truth, prediction, limits)
        summary = {"plane": plane, "by_elevation": {"0": plane}}
        failures = threshold_failures(summary, **DEFAULT_GATE)
        results[arm] = {
            "plane": plane,
            "gate": {**DEFAULT_GATE, "passed": not failures, "failures": failures},
        }
    report = {
        "schema": 1,
        "k": k,
        "probe_tokens_per_pose": k,
        "band_functions_per_pose": k * 11,
        "rays": total_rays,
        "nearest_probe_distance": nearest_stats,
        "raw_exact_code_available_rate": float(np.mean(np.concatenate(exact_available))),
        "arms": results,
    }
    write_json(output, report)
    return report


def train(
    *,
    probe_dir: Path,
    sidecars: list[Path],
    output_dir: Path,
    k: int,
    harmonics: int,
    device_name: str,
    steps: int,
    batch_rays: int,
    eval_chunk: int,
    learning_rate: float,
    d_model: int,
    seed: int,
    eval_every: int,
    router_weight: float,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm requested but unavailable")

    records = load_static_records(sidecars)
    tables = load_probe_tables(probe_dir)
    missing = sorted(set(records) - tables.keys())
    if missing:
        raise ValueError(f"sidecar maps lack probe tables: {missing}")

    train_sets: dict[str, QuerySet] = {}
    val_sets: dict[str, QuerySet] = {}
    memories: dict[str, ProbeMemory] = {}
    nearest_stats: dict[str, Any] = {}
    for name, rows in records.items():
        train_rows, val_rows = split_records(rows)
        train_sets[name] = build_query_set(train_rows, tables[name], k=k)
        val_sets[name] = build_query_set(val_rows, tables[name], k=k)
        _, _, limit_grid = _record_limits(rows[0])
        memories[name] = encode_probe_memory(
            tables[name], limit_grid[:, 0], harmonics=harmonics, device=device,
        )
        tree = cKDTree(tables[name].positions)
        distances, _ = tree.query(
            np.asarray([row["origin"] for row in rows], dtype=np.float64), k=1,
        )
        nearest_stats[name] = {
            "mean": float(np.mean(distances)),
            "p90": float(np.percentile(distances, 90)),
            "max": float(np.max(distances)),
        }

    model = ProbeFusionDecoder(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(
        weight=_class_weights(train_sets.values(), device),
    )
    rng = np.random.default_rng(seed)
    maps = sorted(train_sets)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")

    for step in range(1, steps + 1):
        model.train()
        name = maps[(step - 1) % len(maps)]
        query_set = train_sets[name]
        indices = rng.integers(0, len(query_set), size=batch_rays)
        origins, directions, limits, bands, neighbors, labels, truth = _batch(
            query_set, indices, device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, route_scores, candidate_depth, candidate_hit = model(
            memories[name], origins, directions, limits, bands, neighbors,
            return_router=True,
        )
        classification_loss = loss_fn(logits, labels)
        candidate_prediction = torch.where(
            candidate_hit >= 0.5,
            candidate_depth * limits[:, None],
            limits[:, None],
        )
        best_candidate = torch.argmin(
            torch.abs(candidate_prediction - truth[:, None]), dim=1,
        )
        router_loss = nn.functional.cross_entropy(route_scores, best_candidate)
        loss = classification_loss + router_weight * router_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            preview = evaluate(
                model, memories, val_sets, device=device, chunk=eval_chunk,
                max_per_map=8192,
            )
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "train_classification_loss": float(classification_loss.detach().cpu()),
                "train_router_loss": float(router_loss.detach().cpu()),
                "val_mae": preview["plane"]["mae"],
                "val_code_accuracy": preview["code_accuracy"],
                "val_gate_passed": preview["gate"]["passed"],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_mae"] < best_mae:
                best_mae = row["val_mae"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    assert best_state is not None
    model.load_state_dict(best_state)
    final = evaluate(
        model, memories, val_sets, device=device, chunk=eval_chunk,
    )
    config = {
        "k": k,
        "probe_tokens_per_pose": k,
        "band_functions_per_pose": k * 11,
        "harmonics": harmonics,
        "device": str(device),
        "steps": steps,
        "batch_rays": batch_rays,
        "eval_chunk": eval_chunk,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "seed": seed,
        "router_weight": router_weight,
        "maps": maps,
        "nearest_probe_distance": nearest_stats,
        "routing": "pose_k_nearest_indices_into_load_time_band_memory",
    }
    result = {"schema": 1, "config": config, "history": history, "final": final}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "report.json", result)
    torch.save({"config": config, "model": best_state}, output_dir / "model.pt")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-output", type=Path, default=None,
        help="write the routed-information ceiling and skip learned training",
    )
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--harmonics", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-rays", type=int, default=1024)
    parser.add_argument("--eval-chunk", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--router-weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.oracle_output is not None:
        report = oracle_routing(
            probe_dir=args.probe_dir, sidecars=args.sidecars,
            output=args.oracle_output, k=args.k,
        )
        print(json.dumps({
            "output": str(args.oracle_output),
            "exact_code_available": report["raw_exact_code_available_rate"],
            "arms": {
                name: {"mae": arm["plane"]["mae"], "gate": arm["gate"]["passed"]}
                for name, arm in report["arms"].items()
            },
        }, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --oracle-output is used")
    result = train(
        probe_dir=args.probe_dir, sidecars=args.sidecars,
        output_dir=args.output_dir, k=args.k, harmonics=args.harmonics,
        device_name=args.device, steps=args.steps, batch_rays=args.batch_rays,
        eval_chunk=args.eval_chunk, learning_rate=args.learning_rate,
        d_model=args.d_model, seed=args.seed, eval_every=args.eval_every,
        router_weight=args.router_weight,
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
