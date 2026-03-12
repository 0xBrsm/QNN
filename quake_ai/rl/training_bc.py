"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Sequence

import numpy as np
import torch

from engine.token_protocol import TOKEN_BINARY_HEADER_SIZE, TrustedTokenTick, decode_binary_token_tick
from quake_ai.actions import ACTION_HEADS
from quake_ai.model.observation import TokenObservationEncoder
from quake_ai.rl.dataset_bc import (
    Sample,
    build_token_samples,
)
from quake_ai.model.policy import MLPGRUPolicy
from quake_ai.rl.schemas import MapState
from quake_ai.utils.io import read_json, read_ndjson, write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class BCConfig:
    map_id: str = ""
    output_dir: str = ""
    token_ticks_path: str = ""
    map_state_path: str = ""
    map_states_path: str = ""
    metadata_path: str = ""
    seed: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    sequence_length: int = 64
    epochs: int = 40
    lr: float = 0.01
    patience: int = 5
    use_gru: bool = False
    gru_hidden: int = 0
    trunk_hidden: int = 128
    class_weight_power: float = 0.5
    class_weight_min: float = 0.5
    class_weight_max: float = 2.0
    device: str = "auto"


@dataclass(slots=True)
class _EpisodeRecord:
    index: int
    start_offset: int
    end_offset: int
    episode_id: str
    map_id: str
    source_path: str
    map_state: MapState


@dataclass(slots=True)
class _EpisodeSplit:
    train: list[_EpisodeRecord]
    val: list[_EpisodeRecord]
    test: list[_EpisodeRecord]


def _stack_sequence_batch(chunks: Sequence[Sequence]) -> tuple[dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    if not chunks:
        raise ValueError("chunks must be non-empty")
    seq_len = len(chunks[0])
    batch_size = len(chunks)
    first_obs = chunks[0][0].obs
    if not isinstance(first_obs, dict):
        raise ValueError("Token BC expects dict observations")

    obs_batch: dict[str, np.ndarray] = {}
    for key, value in first_obs.items():
        obs_batch[key] = np.zeros((seq_len, batch_size, *value.shape), dtype=value.dtype)

    actions: Dict[str, np.ndarray] = {}
    first_action = chunks[0][0].action
    for head in first_action.keys():
        actions[head] = np.zeros((seq_len, batch_size), dtype=np.int64)

    for batch_idx, chunk in enumerate(chunks):
        if len(chunk) != seq_len:
            raise ValueError("All chunks in a batch must have the same length")
        for step_idx, sample in enumerate(chunk):
            if not isinstance(sample.obs, dict):
                raise ValueError("Token BC expects dict observations")
            for key, value in sample.obs.items():
                obs_batch[key][step_idx, batch_idx] = value
            for head, value in sample.action.items():
                actions[head][step_idx, batch_idx] = int(value)

    return obs_batch, actions, seq_len * batch_size


def _load_episode_rows(metadata_path: str, episode_count: int) -> list[Dict[str, Any]]:
    if not metadata_path:
        return [{} for _ in range(episode_count)]
    rows = [dict(row) for row in read_ndjson(metadata_path)]
    if len(rows) != episode_count:
        raise RuntimeError(
            f"metadata_path row count {len(rows)} does not match collected episodes {episode_count}: {metadata_path}"
        )
    return rows


def _load_map_states(
    map_state_path: str,
    map_states_path: str,
) -> tuple[Dict[str, MapState], MapState | None]:
    default_map_state: MapState | None = None
    map_states: Dict[str, MapState] = {}

    if map_state_path:
        payload = read_json(map_state_path)
        if isinstance(payload.get("map_states"), Mapping):
            map_states.update({str(key): MapState.from_dict(value) for key, value in payload["map_states"].items()})
        else:
            default_map_state = MapState.from_dict(payload)

    if map_states_path:
        payload = read_json(map_states_path)
        if not isinstance(payload.get("map_states"), Mapping):
            raise RuntimeError(f"map_states_path must define a map_states mapping: {map_states_path}")
        map_states.update({str(key): MapState.from_dict(value) for key, value in payload["map_states"].items()})

    return map_states, default_map_state


def _resolve_episode_map_state(
    episode_row: Mapping[str, Any],
    config: BCConfig,
    map_states: Mapping[str, MapState],
    default_map_state: MapState | None,
) -> tuple[str, MapState]:
    map_id = str(episode_row.get("map_id", "")).strip() or str(config.map_id)
    if map_id in map_states:
        return map_id, map_states[map_id]
    if default_map_state is not None:
        return map_id, default_map_state
    raise RuntimeError(f"No map state available for demo map_id={map_id}")


def _read_token_tick_from_file(handle: BinaryIO) -> TrustedTokenTick | None:
    header = handle.read(TOKEN_BINARY_HEADER_SIZE)
    if not header:
        return None
    if len(header) < TOKEN_BINARY_HEADER_SIZE:
        raise ValueError(f"Truncated header: {len(header)} bytes")

    def read_exact(size: int) -> bytes:
        data = handle.read(size)
        if len(data) != size:
            raise EOFError(f"Expected {size} bytes, got {len(data)}")
        return data

    return decode_binary_token_tick(header, read_exact)


def _scan_token_episode_ranges(path: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    with open(path, "rb") as handle:
        while True:
            start_offset = handle.tell()
            tick = _read_token_tick_from_file(handle)
            if tick is None:
                break
            end_offset = handle.tell()
            if current_start is None:
                current_start = start_offset
            elif tick.reset:
                ranges.append((current_start, current_end))
                current_start = start_offset
            current_end = end_offset
    if current_start is not None:
        ranges.append((current_start, current_end))
    return ranges


def _scan_episode_records(
    config: BCConfig,
    map_states: Mapping[str, MapState],
    default_map_state: MapState | None,
) -> list[_EpisodeRecord]:
    token_ticks_file = Path(config.token_ticks_path)
    episode_ranges = _scan_token_episode_ranges(str(token_ticks_file))
    episode_rows = _load_episode_rows(config.metadata_path, len(episode_ranges))
    records: list[_EpisodeRecord] = []
    for index, (start_offset, end_offset) in enumerate(episode_ranges):
        episode_row = episode_rows[index] if index < len(episode_rows) else {}
        map_id, map_state = _resolve_episode_map_state(episode_row, config, map_states, default_map_state)
        episode_id = str(episode_row.get("episode_id", "")).strip() or f"{token_ticks_file.stem}_{index:04d}"
        records.append(
            _EpisodeRecord(
                index=index,
                start_offset=start_offset,
                end_offset=end_offset,
                episode_id=episode_id,
                map_id=map_id,
                source_path=str(episode_row.get("source_path", "")),
                map_state=map_state,
            )
        )
    return records


def _split_episode_records(records: Sequence[_EpisodeRecord], train_ratio: float, val_ratio: float, seed: int) -> _EpisodeSplit:
    grouped: dict[str, list[_EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)

    episode_ids = sorted(grouped.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    total = len(episode_ids)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))

    train_ids = set(episode_ids[:train_cut])
    val_ids = set(episode_ids[train_cut:val_cut])
    test_ids = set(episode_ids[val_cut:])

    return _EpisodeSplit(
        train=[record for episode_id in episode_ids if episode_id in train_ids for record in grouped[episode_id]],
        val=[record for episode_id in episode_ids if episode_id in val_ids for record in grouped[episode_id]],
        test=[record for episode_id in episode_ids if episode_id in test_ids for record in grouped[episode_id]],
    )


def _read_episode_ticks_for_record(handle: BinaryIO, record: _EpisodeRecord) -> list[TrustedTokenTick]:
    handle.seek(record.start_offset)
    ticks: list[TrustedTokenTick] = []
    while handle.tell() < record.end_offset:
        tick = _read_token_tick_from_file(handle)
        if tick is None:
            raise EOFError(f"Unexpected EOF while reading episode {record.episode_id}")
        ticks.append(tick)
    if handle.tell() != record.end_offset:
        raise ValueError(f"Episode {record.episode_id} overran offset boundary")
    return ticks


def _episode_chunks(samples: Sequence[Sample], sequence_length: int) -> list[list[Sample]]:
    chunk_len = max(int(sequence_length), 1)
    return [list(samples[start : start + chunk_len]) for start in range(0, len(samples), chunk_len) if samples[start : start + chunk_len]]


def _normalized_goal_progress(tick: TrustedTokenTick, map_state: MapState) -> float:
    raw_distances = map_state.metadata.get("distance_to_goal", {})
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0
    region_id = tick.current_region_id
    if region_id < 0:
        return float(tick.done)
    if isinstance(raw_distances, dict):
        distance = float(raw_distances.get(str(region_id), raw_distances.get(region_id, 0.0)))
        return float(max(0.0, min(1.0, 1.0 - (distance / max_distance))))
    return 0.0


def _init_action_counts() -> dict[str, np.ndarray]:
    return {head: np.ones(size, dtype=np.float32) for head, size in ACTION_HEADS.items()}


def _accumulate_episode_stats(
    ticks: Sequence[TrustedTokenTick],
    map_state: MapState,
    action_counts: Mapping[str, np.ndarray] | None = None,
) -> tuple[int, float]:
    sample_count = 0
    goal_progress_max = 0.0
    for tick in ticks:
        if not tick.action_label:
            continue
        sample_count += 1
        goal_progress_max = max(goal_progress_max, _normalized_goal_progress(tick, map_state))
        if action_counts is not None:
            for head in ACTION_HEADS:
                action_counts[head][int(tick.action_label.get(head, 0))] += 1.0
    return sample_count, goal_progress_max


def _class_weights_from_counts(
    counts: Mapping[str, np.ndarray],
    *,
    power: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for head, values in counts.items():
        scaled = np.power(values.astype(np.float32, copy=False), -float(power)).astype(np.float32, copy=False)
        scaled = scaled / max(float(np.mean(scaled)), 1e-8)
        scaled = np.clip(scaled, float(min_weight), float(max_weight))
        weights[head] = scaled.astype(np.float32, copy=False)
    return weights


def _episode_slot_rng(base_seed: int, episode_index: int) -> np.random.Generator:
    seed_value = (int(base_seed) * 1_000_003 + int(episode_index) + 1) % (2**63 - 1)
    return np.random.default_rng(seed_value)


def _load_episode_samples(
    handle: BinaryIO,
    record: _EpisodeRecord,
    *,
    slot_seed: int,
) -> list[Sample]:
    ticks = _read_episode_ticks_for_record(handle, record)
    return build_token_samples(
        ticks,
        record.map_state,
        episode_id=record.episode_id,
        map_id=record.map_id,
        source_path=record.source_path,
        rng=_episode_slot_rng(slot_seed, record.index),
    )


def _write_episode_split_manifest(
    path: Path,
    sample_counts: Mapping[str, int],
    sample_episode_ids: Mapping[str, Sequence[str]],
) -> None:
    write_json(
        path,
        {
            "train": int(sample_counts.get("train", 0)),
            "val": int(sample_counts.get("val", 0)),
            "test": int(sample_counts.get("test", 0)),
            "train_episodes": sorted({str(value) for value in sample_episode_ids.get("train", ())}),
            "val_episodes": sorted({str(value) for value in sample_episode_ids.get("val", ())}),
            "test_episodes": sorted({str(value) for value in sample_episode_ids.get("test", ())}),
        },
    )


def _run_streaming_supervised(
    model: MLPGRUPolicy,
    token_ticks_path: str,
    records: Sequence[_EpisodeRecord],
    batch_size: int,
    sequence_length: int,
    *,
    slot_seed: int,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    if not records:
        return {"loss": 0.0, "accuracy": 0.0}

    total_rows = 0
    total_loss = 0.0
    total_accuracy = 0.0
    pending_chunks: dict[int, list[list[Sample]]] = defaultdict(list)
    ordered_records = list(records)
    if rng is not None and len(ordered_records) > 1:
        order = rng.permutation(len(ordered_records))
        ordered_records = [ordered_records[int(index)] for index in order]

    with open(token_ticks_path, "rb") as handle:
        for record in ordered_records:
            samples = _load_episode_samples(handle, record, slot_seed=slot_seed)
            if not samples:
                continue
            chunks = _episode_chunks(samples, sequence_length)
            if rng is not None and len(chunks) > 1:
                chunk_order = rng.permutation(len(chunks))
                chunks = [chunks[int(index)] for index in chunk_order]
            for chunk in chunks:
                length = len(chunk)
                pending_chunks[length].append(chunk)
                sequences_per_batch = max(1, int(batch_size) // max(length, 1))
                while len(pending_chunks[length]) >= sequences_per_batch:
                    batch_chunks = pending_chunks[length][:sequences_per_batch]
                    del pending_chunks[length][:sequences_per_batch]
                    obs_batch, action_batch, rows = _stack_sequence_batch(batch_chunks)
                    if class_weights is not None and lr is not None:
                        metrics = model.supervised_step(obs_batch, action_batch, class_weights, lr=lr)
                    else:
                        metrics = model.evaluate_supervised(obs_batch, action_batch)
                    total_rows += rows
                    total_loss += float(metrics["loss"]) * rows
                    total_accuracy += float(metrics["accuracy"]) * rows

    for length in sorted(pending_chunks.keys()):
        if not pending_chunks[length]:
            continue
        obs_batch, action_batch, rows = _stack_sequence_batch(pending_chunks[length])
        if class_weights is not None and lr is not None:
            metrics = model.supervised_step(obs_batch, action_batch, class_weights, lr=lr)
        else:
            metrics = model.evaluate_supervised(obs_batch, action_batch)
        total_rows += rows
        total_loss += float(metrics["loss"]) * rows
        total_accuracy += float(metrics["accuracy"]) * rows

    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": total_accuracy / max(total_rows, 1),
    }


def run_behavior_cloning(config: BCConfig) -> Dict[str, float]:
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if not str(config.output_dir).strip():
        raise RuntimeError("Behavior cloning requires output_dir")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not config.token_ticks_path or (not config.map_state_path and not config.map_states_path):
        raise RuntimeError("Behavior cloning requires token_ticks_path and at least one map state input")
    token_ticks_file = Path(config.token_ticks_path)
    if not token_ticks_file.exists():
        raise RuntimeError(f"token_ticks_path does not exist: {token_ticks_file}")
    if config.map_state_path and not Path(config.map_state_path).exists():
        raise RuntimeError(f"map_state_path does not exist: {config.map_state_path}")
    if config.map_states_path and not Path(config.map_states_path).exists():
        raise RuntimeError(f"map_states_path does not exist: {config.map_states_path}")
    map_states, default_map_state = _load_map_states(config.map_state_path, config.map_states_path)
    records = _scan_episode_records(config, map_states, default_map_state)
    split = _split_episode_records(records, config.train_ratio, config.val_ratio, config.seed)

    sample_counts = {"train": 0, "val": 0, "test": 0}
    sample_episode_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    class_counts = _init_action_counts()
    train_success_values: list[float] = []
    val_success_values: list[float] = []
    obs_dim = 0

    with open(config.token_ticks_path, "rb") as handle:
        for split_name, split_records in (("train", split.train), ("val", split.val), ("test", split.test)):
            for record in split_records:
                ticks = _read_episode_ticks_for_record(handle, record)
                sample_count, goal_progress_max = _accumulate_episode_stats(
                    ticks,
                    record.map_state,
                    class_counts if split_name == "train" else None,
                )
                sample_counts[split_name] += sample_count
                if sample_count > 0:
                    sample_episode_ids[split_name].add(record.episode_id)
                if split_name == "train" and sample_count > 0:
                    train_success_values.append(goal_progress_max)
                elif split_name == "val" and sample_count > 0:
                    val_success_values.append(goal_progress_max)
                if split_name == "train" and obs_dim <= 0 and sample_count > 0:
                    obs_dim = TokenObservationEncoder().obs_dim

    _write_episode_split_manifest(output / "split_manifest.json", sample_counts, sample_episode_ids)

    if sample_counts["train"] <= 0:
        raise RuntimeError("No training samples available after split")
    if obs_dim <= 0:
        obs_dim = TokenObservationEncoder().obs_dim

    model = MLPGRUPolicy(
        obs_dim=obs_dim,
        trunk_hidden=config.trunk_hidden,
        gru_hidden=config.gru_hidden,
        use_gru=config.use_gru,
        seed=config.seed,
        device=config.device,
    )

    weights = {
        head: torch.as_tensor(values, dtype=torch.float32, device=model.device)
        for head, values in _class_weights_from_counts(
            class_counts,
            power=config.class_weight_power,
            min_weight=config.class_weight_min,
            max_weight=config.class_weight_max,
        ).items()
    }
    train_success_proxy = float(np.mean(train_success_values)) if train_success_values else 0.0
    val_success_proxy = float(np.mean(val_success_values)) if val_success_values else 0.0

    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        train_metrics = _run_streaming_supervised(
            model,
            config.token_ticks_path,
            split.train,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            slot_seed=config.seed,
            class_weights=weights,
            lr=config.lr,
            rng=rng,
        )
        val_metrics = _run_streaming_supervised(
            model,
            config.token_ticks_path,
            split.val,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            slot_seed=config.seed,
        )

        proxy = val_success_proxy if split.val else train_success_proxy
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_success_proxy": float(proxy),
        }
        history.append(epoch_metrics)

        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save(output / "bc_best_model.npz")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_epoch < 0:
        model.save(output / "bc_best_model.npz")

    final_model = MLPGRUPolicy.load(output / "bc_best_model.npz", device=config.device)

    test_metrics = _run_streaming_supervised(
        final_model,
        config.token_ticks_path,
        split.test,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        slot_seed=config.seed,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "num_train_samples": int(sample_counts["train"]),
        "num_val_samples": int(sample_counts["val"]),
        "num_test_samples": int(sample_counts["test"]),
        "epochs_ran": len(history),
    }

    write_json(output / "bc_history.json", {"history": history})
    write_json(output / "bc_summary.json", summary)
    write_experiment_manifest(output / "bc_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
