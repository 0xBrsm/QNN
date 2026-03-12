"""Sequence-preserving dataset utilities for behavior cloning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from engine.token_protocol import TrustedTokenTick
from quake_ai.actions import ACTION_HEADS, ActionLabels
from quake_ai.model.observation import TokenObservationEncoder
from quake_ai.rl.schemas import MapState
from quake_ai.utils.io import write_json
from quake_ai.vocab import MAX_PLAYER_SLOTS


@dataclass(slots=True)
class Sample:
    episode_id: str
    tick: int
    obs: np.ndarray | Dict[str, np.ndarray]
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


def randomize_player_slots(
    ticks: List[TrustedTokenTick],
    rng: np.random.Generator,
) -> None:
    """Shuffle player_id assignments to prevent overfitting to slot numbers.

    Discovers all unique nonzero player_ids in the episode, maps them to a
    random permutation of 1..MAX_PLAYER_SLOTS, and applies the mapping
    in-place to every object token.
    """
    seen: Dict[int, None] = {}
    for tick in ticks:
        for obj in tick.object_tokens:
            if obj.player_id > 0 and obj.player_id not in seen:
                seen[obj.player_id] = None
    if not seen:
        return
    original_ids = list(seen.keys())
    slot_order = (rng.permutation(MAX_PLAYER_SLOTS) + 1).tolist()
    mapping = {
        player_id: int(slot_order[idx % MAX_PLAYER_SLOTS])
        for idx, player_id in enumerate(original_ids)
    }
    for tick in ticks:
        for obj in tick.object_tokens:
            if obj.player_id > 0:
                obj.player_id = mapping[obj.player_id]


def build_token_samples(
    ticks: Iterable[TrustedTokenTick],
    map_state: MapState,
    episode_id: str,
    *,
    encoder: TokenObservationEncoder | None = None,
    map_id: str = "",
    mode: str = "unknown",
    source_path: str = "",
    shuffle_player_slots: bool = True,
    rng: np.random.Generator | None = None,
) -> List[Sample]:
    encoder = encoder or TokenObservationEncoder()
    encoder.reset()
    raw_distances = map_state.metadata.get("distance_to_goal", {})
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0

    tick_list = list(ticks)
    if shuffle_player_slots:
        slot_rng = rng if rng is not None else np.random.default_rng()
        randomize_player_slots(tick_list, slot_rng)

    samples: List[Sample] = []
    for tick in tick_list:
        if not tick.action_label:
            continue
        action = ActionLabels.from_dict(tick.action_label).to_dict()
        obs = encoder.encode(tick)
        goal_progress = 0.0
        region_id = tick.current_region_id
        if region_id < 0:
            goal_progress = float(tick.done)
        elif isinstance(raw_distances, dict):
            distance = float(raw_distances.get(str(region_id), raw_distances.get(region_id, 0.0)))
            goal_progress = float(max(0.0, min(1.0, 1.0 - (distance / max_distance))))
        samples.append(
            Sample(
                episode_id=episode_id,
                tick=tick.tick,
                obs=obs,
                action=action,
                goal_progress=goal_progress,
                done=tick.done,
                map_id=map_id,
                mode=mode,
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


def success_proxy(samples: Sequence[Sample]) -> float:
    if not samples:
        return 0.0
    by_episode: Dict[str, float] = {}
    for sample in samples:
        by_episode[sample.episode_id] = max(by_episode.get(sample.episode_id, 0.0), sample.goal_progress)
    values = list(by_episode.values())
    return float(np.mean(values)) if values else 0.0
