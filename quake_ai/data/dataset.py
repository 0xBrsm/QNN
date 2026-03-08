"""Dataset builder for imitation and RL warm starts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from quake_ai.actions import ACTION_HEADS, ActionLabels
from quake_ai.navigation import build_observation, heading_from_yaw, load_navigation_map
from quake_ai.models.world_encoder import WorldObservationEncoder
from quake_ai.schemas import MapStateV2, TelemetryTickV1, WorldTickV2
from quake_ai.utils.io import read_json, read_ndjson, write_json


@dataclass(slots=True)
class Sample:
    episode_id: str
    tick: int
    obs: np.ndarray
    action: Dict[str, int]
    goal_progress: float
    done: bool
    map_id: str = ""
    mode: str = "unknown"
    source_path: str = ""


@dataclass(slots=True)
class SplitDataset:
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]


def build_samples(telemetry_path: str | Path, map_features_path: str | Path) -> List[Sample]:
    nav_map = load_navigation_map(map_features_path)
    out: List[Sample] = []

    for row in read_ndjson(telemetry_path):
        tick = TelemetryTickV1.from_dict(row)
        if tick.region_id not in nav_map.records_by_region:
            continue
        action = ActionLabels.from_dict(tick.action_label).to_dict()
        obs = build_observation(
            nav_map=nav_map,
            region_id=tick.region_id,
            heading=heading_from_yaw(tick.yaw),
            player_pos=tick.player_pos,
            player_vel=tick.player_vel,
            nearby_item_flags=tick.nearby_item_flags,
            goal_progress=tick.goal_progress,
        )
        out.append(
            Sample(
                episode_id=tick.episode_id,
                tick=tick.tick,
                obs=obs,
                action=action,
                goal_progress=tick.goal_progress,
                done=tick.done,
            )
        )
    return out


def build_world_samples(
    world_ticks_path: str | Path,
    map_state_path: str | Path,
    encoder: WorldObservationEncoder | None = None,
    metadata_index: Mapping[str, Mapping[str, object]] | None = None,
    mode_filter: Iterable[str] | None = None,
) -> List[Sample]:
    map_state = MapStateV2.from_dict(read_json(map_state_path))
    encoder = encoder or WorldObservationEncoder()
    allowed_modes = set(mode_filter) if mode_filter is not None else None

    samples: List[Sample] = []
    raw_distances = map_state.metadata.get("distance_to_goal", {})
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0

    for row in read_ndjson(world_ticks_path):
        world_tick = WorldTickV2.from_dict(row)
        if not world_tick.action_label:
            continue
        meta = metadata_index.get(world_tick.episode_id) if metadata_index is not None else None
        mode_label = "unknown"
        labels: set[str] = set()
        source_path = ""
        if isinstance(meta, Mapping):
            raw_mode = str(meta.get("mode", "")).strip()
            raw_classification = str(meta.get("classification", "")).strip()
            if raw_mode:
                labels.add(raw_mode)
                mode_label = raw_mode
            if raw_classification:
                labels.add(raw_classification)
                if mode_label == "unknown":
                    mode_label = raw_classification
            source_path = str(meta.get("source_path") or meta.get("local_path") or "")
        if allowed_modes is not None and not labels.intersection(allowed_modes):
            continue
        obs = encoder.encode(map_state, world_tick)
        action = ActionLabels.from_dict(world_tick.action_label).to_dict()
        goal_progress = 0.0
        if world_tick.current_region_id is None:
            goal_progress = float(world_tick.done and world_tick.done_reason == "goal_reached")
        elif isinstance(raw_distances, dict):
            distance = float(raw_distances.get(str(world_tick.current_region_id), raw_distances.get(world_tick.current_region_id, 0.0)))
            goal_progress = float(max(0.0, min(1.0, 1.0 - (distance / max_distance))))
        samples.append(
            Sample(
                episode_id=world_tick.episode_id,
                tick=world_tick.tick,
                obs=obs,
                action=action,
                goal_progress=goal_progress,
                done=world_tick.done,
                map_id=world_tick.map_id,
                mode=mode_label,
                source_path=source_path,
            )
        )
    return samples


def split_samples(samples: Sequence[Sample], train_ratio: float, val_ratio: float, seed: int) -> SplitDataset:
    by_episode: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_episode[sample.episode_id].append(sample)

    episode_ids = sorted(by_episode.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    total = len(episode_ids)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))

    train_ids = set(episode_ids[:train_cut])
    val_ids = set(episode_ids[train_cut:val_cut])
    test_ids = set(episode_ids[val_cut:])

    train = [s for s in samples if s.episode_id in train_ids]
    val = [s for s in samples if s.episode_id in val_ids]
    test = [s for s in samples if s.episode_id in test_ids]

    return SplitDataset(train=train, val=val, test=test)


def write_split_manifest(path: str | Path, split: SplitDataset) -> None:
    write_json(
        path,
        {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "train_episodes": sorted({s.episode_id for s in split.train}),
            "val_episodes": sorted({s.episode_id for s in split.val}),
            "test_episodes": sorted({s.episode_id for s in split.test}),
        },
    )


def class_weights(
    samples: Sequence[Sample],
    *,
    power: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    for head, size in ACTION_HEADS.items():
        counts = np.ones(size, dtype=np.float32)
        for sample in samples:
            counts[int(sample.action.get(head, 0))] += 1.0
        scaled = np.power(counts, -float(power)).astype(np.float32, copy=False)
        scaled = scaled / max(float(np.mean(scaled)), 1e-8)
        scaled = np.clip(scaled, float(min_weight), float(max_weight))
        weights[head] = scaled.astype(np.float32)
    return weights


def batch_iter(samples: Sequence[Sample], batch_size: int, rng: np.random.Generator) -> Iterable[List[Sample]]:
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield [samples[int(i)] for i in batch_idx]


def batch_index_iter(size: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    indices = np.arange(size)
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def stack_observations(samples: Sequence[Sample]) -> np.ndarray:
    return np.stack([s.obs for s in samples], axis=0)


def stack_actions(samples: Sequence[Sample]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for head in ACTION_HEADS:
        out[head] = np.array([int(s.action.get(head, 0)) for s in samples], dtype=np.int64)
    return out


def success_proxy(samples: Sequence[Sample]) -> float:
    if not samples:
        return 0.0
    by_episode: Dict[str, float] = {}
    for sample in samples:
        by_episode[sample.episode_id] = max(by_episode.get(sample.episode_id, 0.0), sample.goal_progress)
    values = list(by_episode.values())
    return float(np.mean(values)) if values else 0.0


def load_metadata_index(path: str | Path) -> Dict[str, Dict[str, object]]:
    from quake_ai.utils.io import read_ndjson

    index: Dict[str, Dict[str, object]] = {}
    for row in read_ndjson(path):
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            continue
        index[episode_id] = dict(row)
    return index
