"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, BinaryIO, Dict, List, Sequence

import numpy as np
import torch

from engine.token_protocol import TOKEN_BINARY_HEADER_SIZE, TrustedTokenTick, decode_binary_token_tick
from quake_ai.actions import ACTION_HEADS, ActionLabels
from quake_ai.model.observation import TokenObservationEncoder
from quake_ai.rl.dataset_bc import (
    Sample,
    build_token_samples,
    randomize_player_slots,
)
from quake_ai.model.policy import QNNPolicy
from quake_ai.rl.schemas import MapState
from quake_ai.utils.io import read_json, read_ndjson, write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest

_ACTION_HEAD_NAMES = list(ACTION_HEADS.keys())
_SENTINEL = None


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
    sequence_length: int = 128
    epochs: int = 40
    lr: float = 0.01
    patience: int = 5
    use_gru: bool = False
    gru_hidden: int = 0
    trunk_hidden: int = 128
    class_weight_power: float = 0.5
    class_weight_min: float = 0.5
    class_weight_max: float = 8.0
    action_tick_offset: int = 0  # shift action labels by N ticks (±1 to diagnose timing)
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


# ---------------------------------------------------------------------------
# Precomputed episode data — observations and actions stored as contiguous
# numpy arrays, encoded once and reused across all epochs.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _PrecomputedEpisode:
    """One episode's observations and actions as contiguous arrays."""
    obs: dict[str, np.ndarray]      # key → (n_samples, ...) arrays
    actions: dict[str, np.ndarray]   # head → (n_samples,) int64
    aim_target: np.ndarray | None   # (n_samples, 2) yaw/pitch to nearest enemy
    n_samples: int


_PLAYER_SUBJECT_ID = 1  # from vocab.py


def _compute_aim_targets(obs: dict[str, np.ndarray], n: int) -> np.ndarray | None:
    """Compute yaw/pitch angle to nearest enemy for each tick (vectorized).

    Returns (n, 2) array of [yaw, pitch] in [-1, 1] range (normalized by pi),
    or None if the observation lacks object tokens.
    """
    obj_ids = obs.get("object_ids")        # (n, max_obj, id_dim)
    obj_sc = obs.get("object_scalars")     # (n, max_obj, scalar_dim)
    obj_mask = obs.get("object_mask")      # (n, max_obj)
    if obj_ids is None or obj_sc is None or obj_mask is None:
        return None

    # Mask: visible player objects only
    is_player = (obj_ids[:, :, 0] == _PLAYER_SUBJECT_ID) & obj_mask  # (n, max_obj)

    rx = obj_sc[:, :, 0]  # (n, max_obj)
    ry = obj_sc[:, :, 1]
    rz = obj_sc[:, :, 2]
    dist_sq = rx * rx + ry * ry + rz * rz

    # Set non-player and empty slots to infinite distance
    dist_sq = np.where(is_player & (dist_sq > 1e-8), dist_sq, np.inf)

    # Find nearest player per tick
    nearest = np.argmin(dist_sq, axis=1)  # (n,)
    tick_idx = np.arange(n)

    best_rx = rx[tick_idx, nearest]
    best_ry = ry[tick_idx, nearest]
    best_rz = rz[tick_idx, nearest]
    has_target = np.isfinite(dist_sq[tick_idx, nearest])

    horiz = np.sqrt(best_rx * best_rx + best_ry * best_ry)
    yaw = np.where(has_target, np.arctan2(best_ry, best_rx) / np.pi, 0.0)
    pitch = np.where(has_target & (horiz > 1e-8), np.arctan2(best_rz, horiz) / np.pi, 0.0)

    aim = np.stack([yaw, pitch], axis=1).astype(np.float32)
    return aim


def _pack_samples(
    samples: List[Sample],
    action_tick_offset: int = 0,
) -> _PrecomputedEpisode | None:
    """Pack a list of Sample objects into contiguous arrays.

    If *action_tick_offset* is non-zero, actions are shifted by that many
    ticks relative to observations (positive = action comes from a later tick).
    This is a diagnostic tool for detecting timing misalignment in demo data.
    """
    if not samples:
        return None
    n = len(samples)
    first_obs = samples[0].obs
    if not isinstance(first_obs, dict):
        return None
    obs: dict[str, np.ndarray] = {}
    for key, value in first_obs.items():
        arr = np.empty((n, *value.shape), dtype=value.dtype)
        arr[0] = value
        obs[key] = arr
    actions: dict[str, np.ndarray] = {
        head: np.empty(n, dtype=np.int64)
        for head in _ACTION_HEAD_NAMES
    }
    for head in _ACTION_HEAD_NAMES:
        actions[head][0] = int(samples[0].action.get(head, 0))
    for i in range(1, n):
        s = samples[i]
        for key, value in s.obs.items():
            obs[key][i] = value
        for head in _ACTION_HEAD_NAMES:
            actions[head][i] = int(s.action.get(head, 0))
    # Temporal smoothing for look heads: average mouse delta over ±1 tick
    # window, then re-quantize to nearest bin.  Reduces label noise from
    # frame-to-frame jitter in human mouse input.
    _SMOOTH_HEADS = {"look_yaw", "look_pitch"}
    from quake_ai.actions import LOOK_MOUSE_BINS, look_label_from_mouse_count, mouse_count_from_look_label
    for head in _SMOOTH_HEADS:
        if head not in actions:
            continue
        labels = actions[head]
        mouse_vals = np.array([mouse_count_from_look_label(int(l)) for l in labels], dtype=np.float32)
        # ±1 tick moving average (3-tap)
        smoothed = mouse_vals.copy()
        if n >= 3:
            smoothed[1:-1] = (mouse_vals[:-2] + mouse_vals[1:-1] + mouse_vals[2:]) / 3.0
        actions[head] = np.array([look_label_from_mouse_count(int(round(v))) for v in smoothed], dtype=np.int64)

    if action_tick_offset != 0:
        for head in _ACTION_HEAD_NAMES:
            actions[head] = np.roll(actions[head], -action_tick_offset)
        # Trim edges where the shift wraps around.
        trim = abs(action_tick_offset)
        if n > 2 * trim:
            start = trim if action_tick_offset > 0 else 0
            end = n - trim if action_tick_offset < 0 else n
            for key in obs:
                obs[key] = obs[key][start:end]
            for head in _ACTION_HEAD_NAMES:
                actions[head] = actions[head][start:end]
            n = end - start
    # Compute auxiliary aim target: yaw/pitch angle to nearest enemy.
    aim_target = _compute_aim_targets(obs, n)
    return _PrecomputedEpisode(obs=obs, actions=actions, aim_target=aim_target, n_samples=n)


# ---------------------------------------------------------------------------
# Batch assembly from precomputed arrays — numpy slicing, no Python loops
# over individual samples.
# ---------------------------------------------------------------------------

def _build_batch_from_chunks(
    episodes: Sequence[_PrecomputedEpisode],
    chunk_indices: Sequence[tuple[int, int, int]],  # (episode_idx, start, length)
    seq_len: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Assemble a (seq_len, batch_size, ...) batch from precomputed episode slices."""
    batch_size = len(chunk_indices)
    first_ep = episodes[chunk_indices[0][0]]
    obs_batch: dict[str, np.ndarray] = {}
    for key, value in first_ep.obs.items():
        obs_batch[key] = np.zeros((seq_len, batch_size, *value.shape[1:]), dtype=value.dtype)
    actions: dict[str, np.ndarray] = {
        head: np.zeros((seq_len, batch_size), dtype=np.int64)
        for head in _ACTION_HEAD_NAMES
    }

    for batch_idx, (ep_idx, start, length) in enumerate(chunk_indices):
        ep = episodes[ep_idx]
        for key in obs_batch:
            obs_batch[key][:length, batch_idx] = ep.obs[key][start:start + length]
        for head in _ACTION_HEAD_NAMES:
            actions[head][:length, batch_idx] = ep.actions[head][start:start + length]

    return obs_batch, actions, seq_len * batch_size


# ---------------------------------------------------------------------------
# Legacy batch stacking for callers that still use Sample lists (dataset_bc).
# ---------------------------------------------------------------------------

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

    actions: Dict[str, np.ndarray] = {
        head: np.zeros((seq_len, batch_size), dtype=np.int64)
        for head in ACTION_HEADS
    }

    for batch_idx, chunk in enumerate(chunks):
        if len(chunk) != seq_len:
            raise ValueError("All chunks in a batch must have the same length")
        for step_idx, sample in enumerate(chunk):
            if not isinstance(sample.obs, dict):
                raise ValueError("Token BC expects dict observations")
            for key, value in sample.obs.items():
                obs_batch[key][step_idx, batch_idx] = value
            for head in ACTION_HEADS:
                actions[head][step_idx, batch_idx] = int(sample.action.get(head, 0))

    return obs_batch, actions, seq_len * batch_size


# ---------------------------------------------------------------------------
# File and metadata helpers (unchanged).
# ---------------------------------------------------------------------------

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


def _init_action_counts() -> dict[str, np.ndarray]:
    return {head: np.ones(size, dtype=np.float32) for head, size in ACTION_HEADS.items()}


def _accumulate_action_counts(
    ep: _PrecomputedEpisode,
    action_counts: dict[str, np.ndarray],
) -> None:
    for head in _ACTION_HEAD_NAMES:
        np.add.at(action_counts[head], ep.actions[head], 1.0)


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


# ---------------------------------------------------------------------------
# Precompute phase — read all episodes once, encode observations, store as
# contiguous numpy arrays. Amortizes I/O and encoding across all epochs.
# ---------------------------------------------------------------------------

def _precompute_split(
    token_ticks_path: str,
    records: Sequence[_EpisodeRecord],
    slot_seed: int,
    action_tick_offset: int = 0,
) -> list[_PrecomputedEpisode]:
    """Precompute all episodes in a split into memory."""
    episodes: list[_PrecomputedEpisode] = []
    with open(token_ticks_path, "rb") as handle:
        for record in records:
            ticks = _read_episode_ticks_for_record(handle, record)
            slot_rng = _episode_slot_rng(slot_seed, record.index)
            randomize_player_slots(ticks, slot_rng)
            samples = build_token_samples(
                ticks,
                record.episode_id,
                map_id=record.map_id,
                source_path=record.source_path,
            )
            ep = _pack_samples(samples, action_tick_offset=action_tick_offset)
            if ep is not None:
                episodes.append(ep)
    return episodes


# ---------------------------------------------------------------------------
# Training loop — operates on precomputed arrays, with background prefetch.
# ---------------------------------------------------------------------------

def _generate_chunk_indices(
    episodes: Sequence[_PrecomputedEpisode],
    sequence_length: int,
) -> list[tuple[int, int, int]]:
    """Generate (episode_idx, start_offset, chunk_length) for all chunks."""
    chunk_len = max(int(sequence_length), 1)
    indices: list[tuple[int, int, int]] = []
    for ep_idx, ep in enumerate(episodes):
        for start in range(0, ep.n_samples, chunk_len):
            length = min(chunk_len, ep.n_samples - start)
            indices.append((ep_idx, start, length))
    return indices


def _run_precomputed_supervised(
    model: QNNPolicy,
    episodes: Sequence[_PrecomputedEpisode],
    batch_size: int,
    sequence_length: int,
    *,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    _empty: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "fuzzy_accuracy": 0.0}
    if not episodes:
        return _empty

    # Process episodes sequentially with GRU hidden state carry-forward.
    # When sequence_length is 0, process each episode as one full sequence
    # (no truncation).  Otherwise, chunk into sequence_length windows for
    # truncated BPTT with hidden state carried across chunks.
    #
    # Gradients are accumulated across ACCUM_EPISODES episodes before each
    # optimizer step to stabilize training (batch_size=1 per chunk is too
    # noisy for convergence).
    training = class_weights is not None and lr is not None
    use_full_episode = int(sequence_length) <= 0
    _ACCUM_EPISODES = max(1, int(batch_size))  # reuse batch_size config as accum count

    # Shuffle episode order (not tick order within episodes).
    ep_order: list[int] = list(range(len(episodes)))
    if rng is not None:
        ep_order = [int(i) for i in rng.permutation(len(episodes))]

    total_rows = 0
    total_loss = 0.0
    total_accuracy = 0.0
    total_fuzzy = 0.0
    per_head_totals: Dict[str, float] = {}
    accum_count = 0

    if training:
        model.bc_zero_grad()

    for ep_idx in ep_order:
        ep = episodes[ep_idx]
        if ep.n_samples == 0:
            continue

        hidden = None  # Reset GRU at episode start.
        chunk_size = ep.n_samples if use_full_episode else max(int(sequence_length), 1)

        for start in range(0, ep.n_samples, chunk_size):
            end = min(start + chunk_size, ep.n_samples)
            length = end - start

            # Slice this chunk: shape (length, 1, ...) for batch_size=1
            obs_chunk: dict[str, np.ndarray] = {}
            for key, arr in ep.obs.items():
                obs_chunk[key] = arr[start:end].reshape(length, 1, *arr.shape[1:])
            act_chunk: dict[str, np.ndarray] = {
                head: ep.actions[head][start:end].reshape(length, 1)
                for head in _ACTION_HEAD_NAMES
            }

            # Slice aim target for this chunk if available.
            aim_chunk = None
            if ep.aim_target is not None:
                aim_chunk = ep.aim_target[start:end].reshape(length, 1, 2)

            if training:
                metrics = model.supervised_step(
                    obs_chunk, act_chunk, class_weights, lr=lr,
                    hidden=hidden,
                    aim_target=aim_chunk,
                    accumulate_only=True,
                )
            else:
                metrics = model.evaluate_supervised(
                    obs_chunk, act_chunk,
                    hidden=hidden,
                )

            # Carry hidden state forward (detached to truncate BPTT).
            next_h = metrics.pop("_next_hidden", None)
            if next_h is not None:
                hidden = next_h.detach() if hasattr(next_h, "detach") else next_h

            total_rows += length
            total_loss += float(metrics["loss"]) * length
            total_accuracy += float(metrics["accuracy"]) * length
            total_fuzzy += float(metrics.get("fuzzy_accuracy", 0.0)) * length
            for key, val in metrics.items():
                if key.startswith("acc_"):
                    per_head_totals[key] = per_head_totals.get(key, 0.0) + float(val) * length

        # Step optimizer after accumulating gradients across N episodes.
        if training:
            accum_count += 1
            if accum_count >= _ACCUM_EPISODES:
                model.bc_step()
                model.bc_zero_grad()
                accum_count = 0

    # Flush any remaining accumulated gradients.
    if training and accum_count > 0:
        model.bc_step()

    denom = max(total_rows, 1)
    result: Dict[str, float] = {
        "loss": total_loss / denom,
        "accuracy": total_accuracy / denom,
        "fuzzy_accuracy": total_fuzzy / denom,
    }
    for key, total in per_head_totals.items():
        result[key] = total / denom
    return result


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

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

    # Precompute all episodes once — observations encoded and stored as contiguous arrays.
    tick_offset = config.action_tick_offset
    if tick_offset:
        log.info("Action tick offset: %d (diagnostic mode)", tick_offset)
    train_episodes = _precompute_split(config.token_ticks_path, split.train, config.seed, tick_offset)
    val_episodes = _precompute_split(config.token_ticks_path, split.val, config.seed, tick_offset)
    test_episodes = _precompute_split(config.token_ticks_path, split.test, config.seed, tick_offset)

    sample_counts = {
        "train": sum(ep.n_samples for ep in train_episodes),
        "val": sum(ep.n_samples for ep in val_episodes),
        "test": sum(ep.n_samples for ep in test_episodes),
    }
    sample_episode_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for split_name, split_records, split_episodes in (
        ("train", split.train, train_episodes),
        ("val", split.val, val_episodes),
        ("test", split.test, test_episodes),
    ):
        ep_idx = 0
        for record in split_records:
            if ep_idx < len(split_episodes) and split_episodes[ep_idx].n_samples > 0:
                sample_episode_ids[split_name].add(record.episode_id)
                ep_idx += 1

    # Accumulate action class counts from precomputed arrays (vectorized)
    class_counts = _init_action_counts()
    for ep in train_episodes:
        _accumulate_action_counts(ep, class_counts)

    _write_episode_split_manifest(output / "split_manifest.json", sample_counts, sample_episode_ids)

    if sample_counts["train"] <= 0:
        raise RuntimeError("No training samples available after split")

    obs_dim = TokenObservationEncoder().obs_dim
    model = QNNPolicy(
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

    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        train_metrics = _run_precomputed_supervised(
            model,
            train_episodes,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            class_weights=weights,
            lr=config.lr,
            rng=rng,
        )
        val_metrics = _run_precomputed_supervised(
            model,
            val_episodes,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
        )

        epoch_metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_fuzzy_accuracy": float(train_metrics.get("fuzzy_accuracy", 0.0)),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_fuzzy_accuracy": float(val_metrics.get("fuzzy_accuracy", 0.0)),
        }
        for key in val_metrics:
            if key.startswith("acc_"):
                epoch_metrics[f"val_{key}"] = float(val_metrics[key])
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

    final_model = QNNPolicy.load(output / "bc_best_model.npz", device=config.device)

    test_metrics = _run_precomputed_supervised(
        final_model,
        test_episodes,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
    )

    summary: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_fuzzy_accuracy": float(test_metrics.get("fuzzy_accuracy", 0.0)),
        "num_train_samples": int(sample_counts["train"]),
        "num_val_samples": int(sample_counts["val"]),
        "num_test_samples": int(sample_counts["test"]),
        "epochs_ran": len(history),
    }
    for key in test_metrics:
        if key.startswith("acc_"):
            summary[f"test_{key}"] = float(test_metrics[key])

    write_json(output / "bc_history.json", {"history": history})
    write_json(output / "bc_summary.json", summary)
    write_experiment_manifest(output / "bc_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
