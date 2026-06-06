"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

import faulthandler as _faulthandler
import json
import os
import sys as _sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import time as _time

# Enable C-level traceback on signals (SEGV / FPE / ABRT). Chunked-prefetch
# BC path occasionally tickles platform-specific issues (WSL DXG pinned
# memory, dtype mismatches in index_copy_) that segfault below the Python
# layer; without this we get no traceback at all.
_faulthandler.enable(file=_sys.stderr, all_threads=True)

import numpy as np
import torch

from qnn import filter_dsl
from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR
from qnn.bc.class_weights import fire_class_weights
from qnn.schema import OBS_DIM
from qnn.model.policy import (
    HEAD_LOSS_WEIGHTS,
    ModelConfig,
    QNNPolicy,
)
from qnn.utils.io import write_json
from qnn.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class BCConfig:
    """Behavior cloning configuration.

    Model arch lives in ``model`` (a ``ModelConfig`` instance) — the sole
    source of truth for architecture. ``build_run_bc_config()`` constructs
    it from the frozen run's ``config/model.json``. Every field of this
    dataclass is required; no Python-level defaults.
    """
    output_dir: str
    bc_data_dir: str
    seed: int
    batch_size: int
    sequence_length: int  # 0 = full episode (no chunking)
    epochs: int
    lr: float
    # Model architecture (source of truth: config/model.json)
    model: ModelConfig
    max_grad_norm: float  # gradient clipping for BPTT stability
    tbptt_limit: int  # max ticks before detaching gradient graph (0 = no limit)
    fixed_tick_hz: int
    device: str
    head_loss_weights: str  # JSON string, e.g. '{"move":1.5,"weapon":0.0}'
    regression_threshold: float
    regression_patience: int
    lr_min: float
    warmup_epochs: int
    prometheus_pushgateway_url: str
    train_eval_interval: int
    train_eval_gap_threshold: float
    train_eval_val_regression_threshold: float
    train_eval_train_improve_threshold: float
    # Performance tuning (sourced from machine.json).
    pin_memory: bool
    prefetch: int
    microbatch_size: int
    snapshot_interval: int
    # When true (and the model has no recurrence), bypass the
    # prefetch + lane-packing pipeline and run training/val on
    # GPU-resident concatenated tensors. Eliminates the per-batch
    # mmap → collate → host→device copy that dominates wall time
    # for tiny memoryless head probes. No effect on full BC trunk
    # runs (use_gru / non-trivial sequence_length triggers fallback).
    preload_to_gpu: bool
    dtype: str                 # "fp32" | "bf16" | "fp16"
    step_report_interval_seconds: int
    fire_pos_weight_override: float  # >0 overrides auto-computed neg/pos ratio for fire BCE
    fire_focal_gamma: float          # >0 swaps fire BCE for focal BCE with this gamma; 0 disables
    # Lin et al. per-class focal prefactor: alpha on positives,
    # (1 - alpha) on negatives. Active only when fire_focal_gamma > 0;
    # 0.5 is neutral. See QNNPolicy.__init__.
    fire_focal_alpha: float
    # >0 enables distance-weighted BCE on the fire head: per-frame loss
    # weight = 1 - gaussian-of-distance-to-nearest-true-fire so wrong-by-
    # one-frame FPs cost a small fraction of wrong-by-100-frames FPs.
    # Tune from the FP timing histogram (~3 at 20 Hz is a sensible
    # starting point; see scripts/analysis/fire_fp_timing.py). 0 disables.
    fire_distance_sigma: float
    jump_pos_weight: float          # >1.0 upweights POS class on move ud-axis CE
    jump_pos_weight_end: float      # >0 linearly decays jump_pos_weight epoch-wise; -1 disables
    # Same Gaussian shoulder as fire_distance_sigma, applied to the
    # ud-axis (jump) CE. Jumps are also a sparse press-or-not decision
    # with human timing noise, so the same shaping is appropriate;
    # expect a different sigma than fire because jump bursts are shorter
    # and rarer. 0 disables.
    jump_distance_sigma: float
    # Per-frame predicate (MongoDB DSL, qnn.filter_dsl) over the stored
    # action/obs arrays plus derived scalars (see _flatten_episode_arrays).
    # None / empty = no masking. Example:
    # {"act.target": {"$ne": 0}} = combat-only training (drops frames
    # with zero target mass; act.target = 1 - target_dist[:, 0]).
    segment_mask: "dict | None"
    # Per-slot predicate over entity token fields. Slots where the
    # predicate evaluates to False have their entity arrays zeroed
    # (positions preserved). Target mass on hidden slots is folded
    # into NO_TARGET so the target head is never trained toward a
    # token the model cannot see.
    token_mask: "dict | None"
    # When true, augment each head's loss-keep with `(no press on axis) |
    # (op_input bit set)`, dropping frames where the demo held a press
    # but the engine didn't act on it (cooldown / dead-time /
    # weapon-switch hold). Requires the recollected corpus that carries
    # act_op_input.npy. See QNNPolicy._compute_head_losses_and_metrics.
    op_input_mask: bool
    # Expected collection identity (qnn.collection_fingerprint). Empty =
    # log-only mode.
    collection_fingerprint: str


from qnn.bc.loop import MidEpochState as _MidEpochState, PrecomputedEpisode as _PrecomputedEpisode, run_epoch as _run_precomputed_supervised
from qnn.bc.supervised_loop import preload_episodes_to_gpu as _preload_episodes_to_gpu, run_epoch_gpu_resident as _run_epoch_gpu_resident


def _selection_score(metrics: Mapping[str, float]) -> float:
    """Composite selection metric for combat-objective BC.

    Lower is better. Each head contributes additively; missing metrics default
    to a neutral value so runs with subsets of heads still produce monotonic
    improvement signals.
    """
    target_error = 1.0 - float(metrics.get("acc_target", 1.0))
    # Move: 3-axis macro-F1 (each axis macro-averages the 3 classes
    # neg/none/pos).  Scaled by 3 to match the magnitude of the historical
    # axis-sum-of-error form so selection scores line up with prior runs.
    move_err = 3.0 * (1.0 - float(metrics.get("f1_move", 1.0)))
    # Look: cos_sim ranges in [-1, 1]; convert to a "1 - cos" error in [0, 2].
    look_err = 1.0 - float(metrics.get("cos_sim_look", 1.0))
    # Fire: F1 ranges in [0, 1]; convert to a "1 - f1" error in [0, 1].
    fire_f1 = float(metrics.get("f1_fire_global", metrics.get("f1_fire", 1.0)))
    fire_err = 1.0 - fire_f1
    # Weapon: macro-F1 across 8 classes — equal weight regardless of
    # frequency so rare-weapon failures don't disappear into the dominant
    # rocket-launcher class.
    weapon_f1 = float(metrics.get("f1_weapon_global", metrics.get("f1_weapon", 1.0)))
    weapon_err = 1.0 - weapon_f1
    return target_error + move_err + look_err + fire_err + weapon_err


def _train_eval_schedule(
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    *,
    interval: int,
    gap_threshold: float,
    val_regression_threshold: float,
    train_improve_threshold: float,
) -> tuple[float, float, list[str]]:
    train_proxy_sum = _selection_score(train_metrics)
    val_sum = _selection_score(val_metrics)
    proxy_gap = val_sum - train_proxy_sum

    reasons: list[str] = []
    safe_interval = max(int(interval), 0)
    if safe_interval > 0 and (epoch + 1) % safe_interval == 0:
        reasons.append(f"interval/{safe_interval}")
    if proxy_gap > float(gap_threshold):
        reasons.append("proxy_gap")

    if history:
        prev = history[-1]
        prev_train_proxy_sum = float(
            prev.get(
                "train_proxy_sum",
                1.0 - float(prev.get("train_acc_target", 0.0)),
            )
        )
        prev_val_sum = 1.0 - float(prev.get("val_acc_target", 0.0))
        val_regression = val_sum - prev_val_sum
        train_delta = train_proxy_sum - prev_train_proxy_sum
        if (
            val_regression > float(val_regression_threshold)
            and train_delta < -float(train_improve_threshold)
        ):
            reasons.append("val_regressed_train_improved")

    return train_proxy_sum, proxy_gap, reasons





# --- Data loading ---

def _unpack_move_axes(packed: np.ndarray) -> np.ndarray:
    """Expand the on-disk packed move byte to (T, 3) uint8 axis class indices.

    The collector packs three 3-class axis indices (each in {0=neg, 1=none,
    2=pos}) into bits 0-1 (fb), 2-3 (lr), 4-5 (ud) of a single uint8.
    Bit 6 carries fire and is extracted separately by ``_unpack_fire_bit``.
    Materializes a fresh array (no longer mmap-backed) — fine because action
    labels are tiny relative to obs.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected (T,) packed move, got shape {arr.shape}")
    fb = (arr      ) & 0x3
    lr = (arr >> 2 ) & 0x3
    ud = (arr >> 4 ) & 0x3
    return np.ascontiguousarray(np.stack([fb, lr, ud], axis=-1))


def _unpack_fire_bit(packed: np.ndarray) -> np.ndarray:
    """Extract the fire bit (bit 6) from the packed move byte.

    Returns a (T,) uint8 in {0, 1}. The heads consume fire as a (T,)
    binary stream; this synthesizes it from the move byte that the
    collector packs in qnn.bc.collect._compact_action_arrays.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected (T,) packed move, got shape {arr.shape}")
    return np.ascontiguousarray((arr >> 6) & 0x1)


# Hoisted to avoid per-episode import-statement lookup × 44k calls.
from qnn.bc.target_labeler import (
    label_enemy_target_distribution as _LABEL_TARGETS,
    DEFAULT_LABELER_CONFIG as _LABELER_DEFAULT_CONFIG,
)

def _densify_obs_for_labeler(obs_padded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert padded native obs arrays to the float layout the target
    labeler reads.

    Numpy-only equivalent of running ``SelfDequantizer + EntityDequantizer``
    on CPU, but producing *only the fields the labeler actually touches*:

      - ``self_scalars[:, 0]`` (health, normalized by MAX_HEALTH) — for
        the dead-frame mask
      - ``entity_scalars_raw[:, :, {HALFEXT, REL, VEL, TEAM, RECENCY}]``
        at the actor-layout offsets
      - ``entity_ids[:, :, {1=modality, 2=player_id}]``
      - ``entity_types``

    Non-actor entity slots are left as zero — the labeler masks to
    ``entity_types == TOKEN_ACTOR`` before reading any scalar offset, so
    the projectile/item/mover branches of the full dequantizer would be
    discarded anyway. Skipping them here saves ~600s of load time on
    the production corpus vs. running the full model-side dequantizers.
    """
    from qnn import engine_norm as en

    health  = np.asarray(obs_padded["health"])
    T = health.shape[0]
    et = np.asarray(obs_padded["entity_types"]).astype(np.int64, copy=False)
    N = et.shape[1]

    # self_scalars: labeler reads only slot 0 (_SELF_HEALTH_OFFSET).
    self_scalars = np.zeros((T, 17), dtype=np.float32)
    self_scalars[:, 0] = health.astype(np.float32) / en.MAX_HEALTH

    # entity_scalars_raw at actor offsets. Mirrors the actor branch of
    # EntityDequantizer (qnn.model.dequant) exactly:
    #   [0:3]   half_extents / DIST_SCALE
    #   [3:6]   rel           / DIST_SCALE
    #   [7:10]  vel           / MAX_VELOCITY
    #   [16]    team
    #   [18]    recency       / TIME_SCALE
    # Non-actor slots are zeroed; labeler masks them out anyway.
    entity_scalars = np.zeros((T, N, 19), dtype=np.float32)
    half = np.asarray(obs_padded["entity_half_extents"]).astype(np.float32) / en.DIST_SCALE
    rel  = np.asarray(obs_padded["entity_rel"]).astype(np.float32) / en.DIST_SCALE
    vel  = np.asarray(obs_padded["entity_vel"]).astype(np.float32) / en.MAX_VELOCITY
    team = np.asarray(obs_padded["entity_team"]).astype(np.float32)
    recency = np.asarray(obs_padded["entity_recency"]).astype(np.float32) / en.TIME_SCALE
    actor_mask = (et == TOKEN_ACTOR)
    if actor_mask.any():
        mask3 = actor_mask[..., None]
        entity_scalars[..., 0:3]  = np.where(mask3, half, 0.0)
        entity_scalars[..., 3:6]  = np.where(mask3, rel,  0.0)
        entity_scalars[..., 7:10] = np.where(mask3, vel,  0.0)
        entity_scalars[..., 16]   = np.where(actor_mask, team,    0.0)
        entity_scalars[..., 18]   = np.where(actor_mask, recency, 0.0)

    # entity_ids: labeler reads slots 1 (modality) and 2 (player_id).
    entity_ids = np.stack([
        np.asarray(obs_padded["entity_subject_id"]).astype(np.int64, copy=False),
        np.asarray(obs_padded["entity_modality_id"]).astype(np.int64, copy=False),
        np.asarray(obs_padded["entity_player_id"]).astype(np.int64, copy=False),
    ], axis=-1)

    return {
        "self_scalars":       self_scalars,
        "entity_types":       et,
        "entity_scalars_raw": entity_scalars,
        "entity_ids":         entity_ids,
    }


def _compute_target_dist(
    obs_padded: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the target labeler on a padded-native episode.

    Returns ``(T, TARGET_DIST_CLASSES) float32`` — same output the
    collector used to bake into the cache. Recomputing at training
    start (a) decouples labeler config from the wire format
    (fingerprint stays stable when you tune LabelerConfig), and (b)
    is lossless: the model trains on the exact f32 distribution the
    labeler emits, no sparse-encoding truncation on multi-hot rows.

    Cost: ~3 µs/frame on CPU. For an 8M-frame corpus that's ~25s of
    one-time startup overhead, amortized across the whole run.
    """
    from qnn.bc.target_labeler import (
        label_enemy_target_distribution,
        DEFAULT_LABELER_CONFIG,
    )
    legacy_obs = _densify_obs_for_labeler(obs_padded)
    return label_enemy_target_distribution(
        legacy_obs, actions, config=DEFAULT_LABELER_CONFIG,
    )


def _madvise_sequential(arr: np.ndarray) -> None:
    """Hint the kernel to read-ahead and drop pages behind the cursor.

    Mmap'd training shards can be tens of GB.  Without this hint the
    page cache fills with every page ever touched, competing with WSL2
    VM memory.  MADV_SEQUENTIAL lets the kernel reclaim pages that the
    training loop has already consumed.
    """
    import mmap as mmap_mod
    mm = getattr(arr, '_mmap', None)
    if mm is not None and hasattr(mm, 'madvise'):
        mm.madvise(mmap_mod.MADV_SEQUENTIAL)


def _effective_head_loss_weights(raw: str) -> Dict[str, float]:
    weights = dict(HEAD_LOSS_WEIGHTS)
    if not raw:
        return weights
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"head_loss_weights must be a JSON object, got {type(parsed).__name__}")
    for head, value in parsed.items():
        weights[str(head)] = float(value)
    return weights


def _require_action_files(
    action_files: Mapping[str, str],
    required_actions: frozenset[str],
    *,
    cache_dir: Path,
) -> None:
    missing = sorted(required_actions.difference(action_files))
    if not missing:
        return
    raise RuntimeError(
        f"{cache_dir} is missing required action arrays {missing}. "
        "Recollect BC data on this branch before training."
    )


def _flatten_episode_arrays(obs: dict, actions: dict) -> dict[str, Any]:
    """Build a flat ``field_path -> np.ndarray`` view of an episode for
    qnn.filter_dsl predicate evaluation.

    Paths mirror the on-disk layout:
        act.<head>   →  action_arrays[head]
        obs.<chan>   →  obs_arrays[chan]
    """
    flat: dict[str, Any] = {}
    for head, arr in actions.items():
        flat[f"act.{head}"] = arr
        if head == "target_dist" and isinstance(arr, np.ndarray) and arr.ndim == 2:
            # Per-frame engagement scalar (= 1 - P(NO_TARGET)) so segment_mask
            # can express the no-engagement filter as `{"act.target":
            # {"$ne": 0}}` without column-indexing in filter_dsl.
            flat["act.target"] = 1.0 - arr[:, 0]
    for chan, arr in obs.items():
        flat[f"obs.{chan}"] = arr
    return flat


def _filter_referenced_keys(predicate: Any) -> set[str]:
    """Collect every leaf field path referenced by a filter predicate."""
    if not isinstance(predicate, dict):
        return set()
    keys: set[str] = set()
    for key, value in predicate.items():
        if key in ("$and", "$or"):
            if isinstance(value, list):
                for sub in value:
                    keys |= _filter_referenced_keys(sub)
        elif key == "$not":
            keys |= _filter_referenced_keys(value)
        elif key.startswith("$"):
            continue
        else:
            keys.add(key)
    return keys


def _flatten_token_arrays(obs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Build the per-token namespace used by train-time ``token_mask``."""
    return {
        "type": np.asarray(obs["entity_types"]),
        "modality": np.asarray(obs["entity_modality_id"]),
        "pid": np.asarray(obs["entity_player_id"]),
        "subject": np.asarray(obs["entity_subject_id"]),
        # Historical token masks used route_idx for slot-identity-like
        # filtering. Native v1 stores subject id in this position; keep
        # the alias so old configs fail less mysteriously.
        "route_idx": np.asarray(obs["entity_subject_id"]),
    }


def _token_keep_mask(
    obs: Mapping[str, np.ndarray],
    token_mask: Mapping[str, Any] | None,
) -> np.ndarray | None:
    if not token_mask:
        return None
    keep = np.asarray(
        filter_dsl.eval_filter(_flatten_token_arrays(obs), token_mask),
        dtype=bool,
    )
    expected = np.asarray(obs["entity_types"]).shape
    if keep.shape != expected:
        raise ValueError(
            f"token_mask predicate must produce a per-token bool array of "
            f"shape {expected}; got {keep.shape}"
        )
    return keep


def _mask_token_array(key: str, arr: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.asarray(arr).copy()
    fill = _ENTITY_TYPES_EMPTY_SENTINEL if key == "entity_types" else 0
    out[~keep] = fill
    return out


def _mask_target_dist_for_tokens(
    target_dist: np.ndarray,
    indptr: np.ndarray,
    keep: np.ndarray | None,
) -> np.ndarray:
    if keep is None:
        return np.asarray(target_dist)
    td = np.asarray(target_dist, dtype=np.float32).copy()
    rows = td.shape[0]
    counts = (indptr[1:rows + 1] - indptr[:rows]).astype(np.int64, copy=False)
    if counts.sum() == 0:
        return td
    row_idx = np.repeat(np.arange(rows, dtype=np.int64), counts)
    slot_idx = np.arange(keep.shape[0], dtype=np.int64) - np.repeat(indptr[:rows], counts)
    drop = (~keep) & (slot_idx < (td.shape[1] - 1))
    if not np.any(drop):
        return td
    moved_rows = row_idx[drop]
    moved_cols = slot_idx[drop] + 1
    moved = np.zeros(rows, dtype=td.dtype)
    np.add.at(moved, moved_rows, td[moved_rows, moved_cols])
    td[moved_rows, moved_cols] = 0.0
    td[:, 0] += moved
    return td


def _slice_native_episode(
    obs_arrays: Mapping[str, np.ndarray],
    row_start: int,
    row_end: int,
    indptr: np.ndarray,
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key, arr in obs_arrays.items():
        if key in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            tok_lo = int(indptr[row_start])
            tok_hi = int(indptr[row_end])
            obs[key] = arr[tok_lo:tok_hi]
        else:
            obs[key] = arr[row_start:row_end]
    return obs


# Native-format entity field categories. Self/spatial fields are
# row-indexed (axis 0 = frame). Entity per-token fields are
# token-indexed (axis 0 = token, leading dim varies per row). The
# scalar entity_count is row-indexed.
_NATIVE_ROW_INDEXED_OBS_FIELDS = frozenset({
    "health", "effective_armor",
    "ammo_shells", "ammo_nails", "ammo_rockets", "ammo_cells",
    "vel", "attack_finished",
    "self_weapon_id", "self_movement_id", "self_items",
    "spatial_dir",
    "spatial_nearest_dist", "spatial_mean_dist",
    "spatial_openness", "spatial_clearance", "spatial_traversable",
    "spatial_dropoff", "spatial_solid_frac", "spatial_water_frac",
    "spatial_slime_frac", "spatial_lava_frac",
    "entity_count",
})

_NATIVE_TOKEN_INDEXED_OBS_FIELDS = frozenset({
    "entity_types", "entity_subject_id", "entity_modality_id",
    "entity_player_id", "entity_event_count",
    "entity_event_actions", "entity_event_sources",
    "entity_half_extents", "entity_rel", "entity_vel",
    "entity_path", "entity_path_dist", "entity_eta", "entity_recency",
    "entity_facing", "entity_team", "entity_score",
    "entity_amount", "entity_regen", "entity_state",
})

# Sentinel for empty entity slots in the padded (T, MAX_TOKEN_OBJECTS,
# ...) materialization. -1 in entity_types matches the Tokenizer's
# `entity_mask = (entity_types == TOKEN_ACTOR)` semantics; the
# Tokenizer key-padding mask flips on non-actor types so the
# transformer simply ignores empty slots.
_ENTITY_TYPES_EMPTY_SENTINEL = -1


def _materialize_padded_entity(
    obs: dict[str, np.ndarray], n_max: int,
) -> dict[str, np.ndarray]:
    """Pad an episode's token-indexed entity fields to ``(T, n_max, ...)``.

    Required by the trainer's GPU-resident / chunked prefetch paths
    which both index batches along axis 0 (frame) and need a constant
    second-dim for tensor concatenation. The unpadded
    ``(total_tokens, ...)`` layout is preserved on disk per the
    engine_norm phase 2 spec — this pad is a load-time materialization,
    not a re-write of the shard.

    ``entity_count`` (T,) drives the per-row valid-prefix; trailing
    slots are zeroed (``entity_types`` gets -1 sentinels so the
    Tokenizer's actor-only mask works unchanged).
    """
    counts = obs.get("entity_count")
    if counts is None:
        return obs  # legacy already-padded layout (test path)
    counts_np = np.asarray(counts, dtype=np.int64)
    T = counts_np.shape[0]
    indptr_local = np.concatenate([[0], np.cumsum(counts_np)])
    # Build a single (T, n_max) gather index that's reused across every
    # token-indexed field. valid[t, j] iff slot j is occupied for row t.
    # gather_idx points at flat[0] for invalid slots — the result is
    # masked back out via np.where before being returned, so the bogus
    # read is harmless.
    slots = np.arange(n_max, dtype=np.int64)
    counts_clamped = np.minimum(counts_np, n_max)
    valid = slots[None, :] < counts_clamped[:, None]
    gather_idx = np.where(valid, indptr_local[:T, None] + slots[None, :], 0)

    out = dict(obs)
    for key in list(obs.keys()):
        if key not in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            continue
        flat = np.asarray(obs[key])
        fill = _ENTITY_TYPES_EMPTY_SENTINEL if key == "entity_types" else 0
        if flat.shape[0] == 0:
            # Empty episode (no tokens across any row). All slots
            # invalid; skip the gather, emit a fill-only tensor.
            out[key] = np.full((T, n_max) + flat.shape[1:], fill, dtype=flat.dtype)
            continue
        padded = flat[gather_idx]  # (T, n_max, *per_token_shape)
        # Broadcast the (T, n_max) mask up to padded's rank so np.where
        # zeros (or sentinels) the trailing invalid slots.
        if padded.ndim > 2:
            mask = valid.reshape(valid.shape + (1,) * (padded.ndim - 2))
        else:
            mask = valid
        out[key] = np.where(mask, padded, np.asarray(fill, dtype=padded.dtype))
    return out


def _pad_entity_batch(
    unpadded_obs: dict[str, np.ndarray],
    indptr: np.ndarray,
    row_start: int,
    row_end: int,
    n_max: int,
) -> dict[str, np.ndarray]:
    """Pad token-indexed entity fields for a contiguous row range.

    Vectorized batch-side equivalent of :func:`_materialize_padded_entity`.
    Operates on already-sliced row arrays whose token data lives in
    ``unpadded_obs[key]`` indexed by the per-episode ``indptr`` of
    length ``n_samples + 1``. Returns a dict with only the token-indexed
    keys padded to ``(row_end - row_start, n_max, *per_token_shape)``.

    Caller is responsible for stitching with row-indexed fields, which
    are sliced upstream. This split keeps the per-batch pad work tight:
    one vectorized ``flat[gather_idx]`` per token-indexed key over just
    the rows in this batch.
    """
    n_rows = row_end - row_start
    if n_rows <= 0:
        return {}
    # Local indptr restricted to the requested row range, offset so its
    # values index into the unpadded[key] arrays' contiguous token range.
    indptr_slice = indptr[row_start:row_end + 1]
    counts = (indptr_slice[1:] - indptr_slice[:-1]).astype(np.int64, copy=False)
    slots = np.arange(n_max, dtype=np.int64)
    counts_clamped = np.minimum(counts, n_max)
    valid = slots[None, :] < counts_clamped[:, None]
    # Absolute per-row token start into the unpadded[key] arrays.
    # ``indptr`` is the per-episode cumulative entity_count, so
    # indptr_slice[:-1] points at the first valid token for each row.
    row_starts = indptr_slice[:-1].astype(np.int64, copy=False)
    gather_idx = np.where(valid, row_starts[:, None] + slots[None, :], 0)

    out: dict[str, np.ndarray] = {}
    for key, flat in unpadded_obs.items():
        if key not in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            continue
        flat_arr = np.asarray(flat)
        fill = _ENTITY_TYPES_EMPTY_SENTINEL if key == "entity_types" else 0
        if flat_arr.shape[0] == 0:
            out[key] = np.full((n_rows, n_max) + flat_arr.shape[1:], fill, dtype=flat_arr.dtype)
            continue
        padded = flat_arr[gather_idx]
        if padded.ndim > 2:
            mask = valid.reshape(valid.shape + (1,) * (padded.ndim - 2))
        else:
            mask = valid
        out[key] = np.where(mask, padded, np.asarray(fill, dtype=padded.dtype))
    return out


def _load_precomputed(
    cache_dir: Path,
    *,
    required_actions: frozenset[str] = frozenset(),
    segment_mask: dict | None = None,
    token_mask: dict | None = None,
) -> list[_PrecomputedEpisode]:
    """Load precomputed episodes with real memory-mapped .npy arrays.

    Episodes are returned sorted globally by ``(demo_idx, episode_idx,
    segment_idx)``.  ``demo_idx`` is the position of each demo in the
    collector's canonical sorted demo list; ``episode_idx`` is the
    0-based ordinal of each surviving run when the collector segmented
    the demo on the filter config's ``drop_tick_labels`` mask;
    ``segment_idx`` is the 0-based ordinal of each surviving run inside
    that episode after applying the train-time ``segment_mask``
    predicate (or 0 if no mask is set).  This makes training-time
    shuffle a pure function of the seed, the dataset, and the mask —
    independent of which worker finished first during collection.

    No ``segment_mask`` keeps each episode as one ``segment_idx=0``
    trajectory. ``segment_mask`` and ``token_mask`` both use the
    shared training filter DSL; native-v1 caches supply cached
    ``target_dist`` so common action masks do not rerun the labeler.
    """
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != "sharded_v1":
        raise RuntimeError(
            f"{cache_dir}/manifest.json: expected sharded_v1 format. "
            "Recollect BC data with the current collector."
        )
    # engine_norm phase 2 explicit version gate. Legacy f16 shards have
    # no `format_version` key — refuse them loudly; in-place migration is
    # not supported (per the no-backcompat directive). Recollect via
    # `python -m qnn.bc.collect` to produce native_v1 shards.
    format_version = manifest.get("format_version")
    if format_version != "native_v1":
        raise RuntimeError(
            f"{cache_dir}/manifest.json: expected format_version='native_v1', "
            f"got {format_version!r}. Legacy f16 caches must be recollected "
            f"with the current collector — no silent migration."
        )

    import time as _time
    import os
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp

    shards = manifest.get("shards", [])

    required = set(required_actions)
    required.add("target_dist")
    _validate_loader_options(cache_dir, shards, segment_mask, token_mask, required)

    shard_args = []
    fallback_idx = 0
    for shard_idx, shard in enumerate(shards):
        n_eps = len(shard.get("episode_lengths", []))
        shard_args.append((
            str(cache_dir), shard_idx, shard, fallback_idx,
            frozenset(required), segment_mask, token_mask,
        ))
        fallback_idx += n_eps

    _t_load_start = _time.perf_counter()
    # Use os.cpu_count() capped to a reasonable max — labeler is python-
    # bound so we need one worker per logical core to actually parallelize
    # it. fork() context inherits the imports already done by the parent
    # so workers start near-instantly.
    n_workers = min(os.cpu_count() or 4, 30)
    ctx = _mp.get_context("fork")
    completed = 0
    total_shards = len(shard_args)
    _t_last_report = _t_load_start

    from concurrent.futures import as_completed

    pass1_results: list[tuple[int, int, list[_ShardEpisodeMeta]] | None] = [None] * total_shards
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        fut_to_idx = {
            ex.submit(_process_shard_work_count_only, shard_args[i]): i
            for i in range(total_shards)
        }
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            pass1_results[idx] = fut.result()
            completed += 1
            _now = _time.perf_counter()
            if _now - _t_last_report >= 5.0 or completed == total_shards:
                _elapsed = _now - _t_load_start
                eps_so_far = sum(
                    len(r[2]) for r in pass1_results if r is not None
                )
                print(
                    f"  [bc/load] {completed}/{total_shards} shards "
                    f"(counts)  {eps_so_far} eps  {_elapsed:.1f}s",
                    flush=True,
                )
                _t_last_report = _now

    keep_idxs = [i for i, r in enumerate(pass1_results) if r is not None]
    if not keep_idxs:
        print(f"  [bc/load] DONE 0 eps in "
              f"{_time.perf_counter() - _t_load_start:.1f}s "
              f"({total_shards} shards × {n_workers} workers)", flush=True)
        return []

    all_meta: list[tuple[tuple[int, int, int], int, int, int, int, np.ndarray | None]] = []
    shard_offsets: dict[int, tuple[int, int, int, int]] = {}
    # shard_offsets[idx] = (row_offset, row_count, tok_offset, tok_count)
    row_cursor = 0
    tok_cursor = 0
    for idx in keep_idxs:
        n_rows, n_toks, ep_metas = pass1_results[idx]  # type: ignore[misc]
        shard_offsets[idx] = (row_cursor, n_rows, tok_cursor, n_toks)
        for ep in ep_metas:
            all_meta.append((
                ep.sort_key,
                row_cursor + ep.row_start,
                row_cursor + ep.row_end,
                tok_cursor + ep.tok_start,
                tok_cursor + ep.tok_end,
                ep.ep_indptr,
            ))
        row_cursor += n_rows
        tok_cursor += n_toks
    total_rows = row_cursor
    total_toks = tok_cursor
    pass1_results = []  # drop refs; ep_metas now live in all_meta

    _t_pass1_done = _time.perf_counter()
    in_flight_cap = max(2 * n_workers, 8)
    global_obs_row: dict[str, np.ndarray] = {}
    global_obs_tok: dict[str, np.ndarray] = {}
    global_acts: dict[str, np.ndarray] = {}

    pass2_completed = 0
    pass2_total = len(keep_idxs)
    _t_last_report = _t_pass1_done
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        pending: dict[Any, int] = {}
        submit_iter = iter(keep_idxs)
        # Prime the pool.
        for _ in range(in_flight_cap):
            try:
                idx = next(submit_iter)
            except StopIteration:
                break
            fut = ex.submit(_process_shard_work, shard_args[idx])
            pending[fut] = idx
        while pending:
            fut = next(iter(as_completed(pending.keys())))
            shard_idx = pending.pop(fut)
            batch = fut.result()
            if batch is not None:
                r_off, n_rows, t_off, n_toks = shard_offsets[shard_idx]
                # First non-empty batch allocates globals.
                if not global_obs_row and not global_obs_tok and not global_acts:
                    for key, arr in batch.obs_row.items():
                        target_dtype = (
                            np.int32 if arr.dtype in (np.uint16, np.uint32) else arr.dtype
                        )
                        global_obs_row[key] = np.empty(
                            (total_rows,) + arr.shape[1:], dtype=target_dtype,
                        )
                    for key, arr in batch.obs_tok.items():
                        target_dtype = (
                            np.int32 if arr.dtype in (np.uint16, np.uint32) else arr.dtype
                        )
                        global_obs_tok[key] = np.empty(
                            (total_toks,) + arr.shape[1:], dtype=target_dtype,
                        )
                    for head_k, arr in batch.acts.items():
                        global_acts[head_k] = np.empty(
                            (total_rows,) + arr.shape[1:], dtype=arr.dtype,
                        )
                for key, buf in global_obs_row.items():
                    np.copyto(
                        buf[r_off:r_off + n_rows],
                        batch.obs_row[key], casting="same_kind",
                    )
                for head_k, buf in global_acts.items():
                    np.copyto(
                        buf[r_off:r_off + n_rows],
                        batch.acts[head_k], casting="same_kind",
                    )
                for key, buf in global_obs_tok.items():
                    np.copyto(
                        buf[t_off:t_off + n_toks],
                        batch.obs_tok[key], casting="same_kind",
                    )
                batch.obs_row.clear()
                batch.obs_tok.clear()
                batch.acts.clear()
            del batch
            pass2_completed += 1
            try:
                next_idx = next(submit_iter)
                fut2 = ex.submit(_process_shard_work, shard_args[next_idx])
                pending[fut2] = next_idx
            except StopIteration:
                pass
            _now = _time.perf_counter()
            if _now - _t_last_report >= 5.0 or pass2_completed == pass2_total:
                _elapsed = _now - _t_load_start
                print(
                    f"  [bc/load] {pass2_completed}/{pass2_total} shards "
                    f"(data)  {_elapsed:.1f}s",
                    flush=True,
                )
                _t_last_report = _now

    _t_global_alloc = _t_pass1_done
    indexed: list[tuple[tuple[int, int, int], _PrecomputedEpisode]] = []
    for sort_key, row_lo, row_hi, tok_lo, tok_hi, ep_indptr in all_meta:
        sub_obs: dict[str, np.ndarray] = {}
        for key, buf in global_obs_row.items():
            sub_obs[key] = buf[row_lo:row_hi]
        for key, buf in global_obs_tok.items():
            sub_obs[key] = buf[tok_lo:tok_hi]
        sub_act: dict[str, np.ndarray] = {}
        for head, buf in global_acts.items():
            sub_act[head] = buf[row_lo:row_hi]
        indexed.append((
            sort_key,
            _PrecomputedEpisode(
                obs=sub_obs,
                actions=sub_act,
                n_samples=row_hi - row_lo,
                sort_key=sort_key,
                entity_indptr=ep_indptr,
            ),
        ))

    indexed.sort(key=lambda item: item[0])
    _t_total = _time.perf_counter() - _t_load_start
    print(
        f"  [bc/load] DONE {len(indexed)} eps in {_t_total:.1f}s "
        f"({total_shards} shards × {n_workers} workers; "
        f"global_alloc={_t_global_alloc - _t_load_start:.1f}s)",
        flush=True,
    )
    return [ep for _, ep in indexed]


@dataclass(slots=True)
class _ShardEpisodeMeta:
    row_start: int
    row_end: int
    tok_start: int
    tok_end: int
    ep_indptr: np.ndarray
    sort_key: tuple[int, int, int]


@dataclass(slots=True)
class _ShardSegment:
    src_row_start: int
    src_row_end: int
    meta: _ShardEpisodeMeta


@dataclass(slots=True)
class _ShardBatch:
    obs_row: dict[str, np.ndarray]
    obs_tok: dict[str, np.ndarray]
    acts: dict[str, np.ndarray]
    episodes: list[_ShardEpisodeMeta]


def _validate_loader_options(
    cache_dir: Path,
    shards: Sequence[Mapping[str, Any]],
    segment_mask: dict | None,
    token_mask: dict | None,
    required_actions: frozenset[str],
) -> None:
    del segment_mask, token_mask
    for shard in shards:
        _require_action_files(shard["actions"], required_actions, cache_dir=cache_dir)
        if "entity_count" not in shard.get("obs", {}):
            raise RuntimeError(
                f"{cache_dir} contains a native_v1 shard without obs.entity_count; "
                "recollect with the current collector."
            )


def _episode_ids(
    shard: Mapping[str, Any],
    fallback_idx_start: int,
) -> tuple[list[int], list[int], list[int]]:
    lengths = [int(n) for n in shard.get("episode_lengths", [])]
    demo_idxs = shard.get("demo_idxs")
    if demo_idxs is None or len(demo_idxs) != len(lengths):
        demo_idxs = list(range(fallback_idx_start, fallback_idx_start + len(lengths)))
    episode_idxs = shard.get("episode_idxs")
    if episode_idxs is None or len(episode_idxs) != len(lengths):
        episode_idxs = [0] * len(lengths)
    return lengths, [int(v) for v in demo_idxs], [int(v) for v in episode_idxs]


def _build_indptr(entity_count: np.ndarray) -> np.ndarray:
    counts = np.asarray(entity_count, dtype=np.int64)
    indptr = np.empty(counts.shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr


def _target_runs(
    obs: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    segment_mask: dict | None,
) -> list[tuple[int, int]]:
    if not segment_mask:
        first = next(iter(actions.values()))
        return [(0, int(first.shape[0]))]
    flat = _flatten_episode_arrays(dict(obs), dict(actions))
    mask = np.asarray(filter_dsl.eval_filter(flat, segment_mask), dtype=bool)
    from qnn.bc.collect import _runs_from_mask
    return [(int(s), int(e)) for s, e in _runs_from_mask(mask)]


def _shard_segments(
    shard: Mapping[str, Any],
    fallback_idx_start: int,
    obs_arrays: Mapping[str, np.ndarray],
    action_arrays: Mapping[str, np.ndarray],
    shard_indptr: np.ndarray,
    segment_mask: dict | None,
) -> list[_ShardSegment]:
    lengths, demo_idxs, episode_idxs = _episode_ids(shard, fallback_idx_start)
    predicate_keys = _filter_referenced_keys(segment_mask)
    needs_token_obs = any(
        key.startswith("obs.") and key[4:] in _NATIVE_TOKEN_INDEXED_OBS_FIELDS
        for key in predicate_keys
    )
    segments: list[_ShardSegment] = []
    src_start = 0
    row_cursor = 0
    tok_cursor = 0
    for n_samples, demo_idx, episode_idx in zip(lengths, demo_idxs, episode_idxs):
        src_end = src_start + n_samples
        actions = {
            head: values[src_start:src_end]
            for head, values in action_arrays.items()
        }
        if segment_mask:
            if needs_token_obs:
                ep_indptr = (
                    shard_indptr[src_start:src_end + 1] - shard_indptr[src_start]
                ).astype(np.int64, copy=True)
                unpadded = _slice_native_episode(
                    obs_arrays, src_start, src_end, shard_indptr,
                )
                padded = _pad_entity_batch(
                    unpadded, ep_indptr, 0, int(n_samples), MAX_TOKEN_OBJECTS,
                )
                obs_for_filter = {
                    key: (padded[key] if key in padded else value)
                    for key, value in unpadded.items()
                }
            else:
                obs_for_filter = {
                    key[4:]: obs_arrays[key[4:]][src_start:src_end]
                    for key in predicate_keys
                    if key.startswith("obs.") and key[4:] in obs_arrays
                }
        else:
            obs_for_filter = {}
        runs = _target_runs(obs_for_filter, actions, segment_mask)
        for segment_idx, (local_start, local_end) in enumerate(runs):
            row_start = src_start + local_start
            row_end = src_start + local_end
            ep_indptr = (
                shard_indptr[row_start:row_end + 1] - shard_indptr[row_start]
            ).astype(np.int64, copy=True)
            n_rows = row_end - row_start
            n_toks = int(ep_indptr[-1])
            segments.append(_ShardSegment(
                src_row_start=row_start,
                src_row_end=row_end,
                meta=_ShardEpisodeMeta(
                    row_start=row_cursor,
                    row_end=row_cursor + n_rows,
                    tok_start=tok_cursor,
                    tok_end=tok_cursor + n_toks,
                    ep_indptr=ep_indptr,
                    sort_key=(demo_idx, episode_idx, segment_idx),
                ),
            ))
            row_cursor += n_rows
            tok_cursor += n_toks
        src_start = src_end
    return segments


def _process_shard_work_count_only(
    args: tuple,
) -> "tuple[int, int, list[_ShardEpisodeMeta]] | None":
    (cache_dir_str, shard_idx, shard, fallback_idx_start,
     required_actions, segment_mask, token_mask) = args
    cache_dir = Path(cache_dir_str)
    del shard_idx, required_actions
    obs_arrays = {
        key: np.load(cache_dir / fname, mmap_mode="r")
        for key, fname in shard["obs"].items()
    }
    action_arrays = {
        head: np.load(cache_dir / fname, mmap_mode="r")
        for head, fname in shard["actions"].items()
    }
    if "move" in action_arrays:
        refs = _filter_referenced_keys(segment_mask)
        if "act.move" in refs or "act.fire" in refs:
            move_packed = action_arrays["move"]
            action_arrays["move"] = _unpack_move_axes(move_packed)
            action_arrays["fire"] = _unpack_fire_bit(move_packed)
    shard_indptr = _build_indptr(obs_arrays["entity_count"])
    keep = _token_keep_mask(obs_arrays, token_mask)
    action_arrays["target_dist"] = _mask_target_dist_for_tokens(
        action_arrays["target_dist"], shard_indptr, keep,
    )
    segments = _shard_segments(
        shard, fallback_idx_start, obs_arrays, action_arrays, shard_indptr, segment_mask,
    )
    if not segments:
        return None
    return (
        segments[-1].meta.row_end,
        segments[-1].meta.tok_end,
        [seg.meta for seg in segments],
    )


def _process_shard_work(
    args: tuple,
):
    """Worker entrypoint: process one shard's episodes end-to-end.

    Returns one ``_ShardBatch`` (or ``None`` if the shard contributed
    no surviving rows). The shard batch carries the surviving rows /
    tokens as concatenated per-key arrays plus per-episode offset
    metadata; the parent merges all worker batches into global
    per-key buffers and materializes ``_PrecomputedEpisode`` records
    that view those globals.

    fork() context means imports / constants from the parent are
    inherited at zero startup cost.
    """
    (cache_dir_str, shard_idx, shard, fallback_idx_start,
     required_actions, segment_mask, token_mask) = args
    cache_dir = Path(cache_dir_str)
    del shard_idx, required_actions

    obs_arrays = {
        key: np.load(cache_dir / fname, mmap_mode="r")
        for key, fname in shard["obs"].items()
    }
    # PyTorch CPU lacks index_copy_ for uint16/uint32, which the
    # chunked-prefetch path uses for lane-packed batch staging. The
    # GPU-resident path is unaffected (it uses index_select). Upcast
    # u16/u32 fields to signed equivalents at the load boundary so the
    # downstream training code is dtype-agnostic.
    obs_arrays = {
        key: (np.asarray(arr).astype(np.int32, copy=False)
              if arr.dtype in (np.uint16, np.uint32)
              else arr)
        for key, arr in obs_arrays.items()
    }
    action_arrays = {
        head: np.load(cache_dir / fname, mmap_mode="r")
        for head, fname in shard["actions"].items()
    }
    if "move" in action_arrays:
        move_packed = action_arrays["move"]
        action_arrays["move"] = _unpack_move_axes(move_packed)
        action_arrays["fire"] = _unpack_fire_bit(move_packed)
    for arr in obs_arrays.values():
        _madvise_sequential(arr)
    for arr in action_arrays.values():
        if isinstance(arr, np.memmap):
            _madvise_sequential(arr)

    shard_indptr = _build_indptr(obs_arrays["entity_count"])
    keep = _token_keep_mask(obs_arrays, token_mask)
    if keep is not None:
        action_arrays["target_dist"] = _mask_target_dist_for_tokens(
            action_arrays["target_dist"], shard_indptr, keep,
        )
    segments = _shard_segments(
        shard, fallback_idx_start, obs_arrays, action_arrays, shard_indptr, segment_mask,
    )
    total_kept_rows = segments[-1].meta.row_end if segments else 0
    if total_kept_rows == 0:
        return None

    def _gather_rows(arr: np.ndarray) -> np.ndarray:
        out = np.empty((total_kept_rows,) + arr.shape[1:], dtype=arr.dtype)
        cursor = 0
        for seg in segments:
            n = seg.src_row_end - seg.src_row_start
            if n:
                out[cursor:cursor + n] = arr[seg.src_row_start:seg.src_row_end]
                cursor += n
        return out

    shard_obs_row: dict[str, np.ndarray] = {}
    for key, arr in obs_arrays.items():
        if key in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            continue
        shard_obs_row[key] = _gather_rows(arr)

    shard_acts: dict[str, np.ndarray] = {}
    for head, arr in action_arrays.items():
        shard_acts[head] = _gather_rows(arr)

    total_kept_toks = segments[-1].meta.tok_end

    def _gather_toks(arr: np.ndarray) -> np.ndarray:
        out = np.empty((total_kept_toks,) + arr.shape[1:], dtype=arr.dtype)
        cursor = 0
        for seg in segments:
            s = int(shard_indptr[seg.src_row_start])
            e = int(shard_indptr[seg.src_row_end])
            n = e - s
            if n:
                out[cursor:cursor + n] = arr[s:e]
                cursor += n
        return out

    shard_obs_tok: dict[str, np.ndarray] = {}
    for key, arr in obs_arrays.items():
        if key in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            source = _mask_token_array(key, arr, keep) if keep is not None else arr
            shard_obs_tok[key] = _gather_toks(source)

    return _ShardBatch(
        obs_row=shard_obs_row,
        obs_tok=shard_obs_tok,
        acts=shard_acts,
        episodes=[seg.meta for seg in segments],
    )



# ---------------------------------------------------------------------------
# Prometheus pushgateway integration (optional).
# ---------------------------------------------------------------------------

_PROM_METRICS_TO_PUSH = (
    "val_acc_target",
    "train_acc_target",
    "train_loss", "val_loss",
)


def _push_metrics_to_prometheus(
    gateway_url: str,
    epoch_metrics: Dict[str, float],
    epoch: int,
    variant: str,
    config: BCConfig,
    *,
    _warned: list[bool] = [False],  # noqa: B006 — mutable default for singleton state
) -> None:
    """Push selected epoch metrics to a Prometheus pushgateway.

    No-ops silently when prometheus_client is not installed or the push fails.
    Only prints a warning on the first failure to avoid log spam.
    """
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    except ImportError:
        if not _warned[0]:
            print("  [bc] prometheus_client not installed — skipping metrics push")
            _warned[0] = True
        return

    try:
        registry = CollectorRegistry()
        epoch_gauge = Gauge(
            "bc_epoch", "Current training epoch",
            labelnames=["variant", "lr", "batch_size"],
            registry=registry,
        )
        epoch_gauge.labels(variant=variant, lr=str(config.lr), batch_size=str(config.batch_size)).set(epoch)

        for metric_name in _PROM_METRICS_TO_PUSH:
            if metric_name not in epoch_metrics:
                continue
            safe_name = f"bc_{metric_name}"
            g = Gauge(
                safe_name, metric_name,
                labelnames=["variant", "lr", "batch_size"],
                registry=registry,
            )
            g.labels(variant=variant, lr=str(config.lr), batch_size=str(config.batch_size)).set(
                epoch_metrics[metric_name]
            )

        push_to_gateway(gateway_url, job="bc_training", registry=registry)
    except Exception as exc:
        if not _warned[0]:
            print(f"  [bc] WARNING: Prometheus push failed ({exc}); suppressing further warnings")
            _warned[0] = True


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def run_behavior_cloning(
    config: BCConfig,
    seed_checkpoint: str = "",
    *,
    model_factory: Callable[[int, ModelConfig], "torch.nn.Module"] | None = None,
) -> Dict[str, float]:
    """Run BC training.

    ``model_factory`` is the same hook ``QNNPolicy.__init__`` takes; pass
    one to swap in an ablation module (e.g. a per-head probe from
    ``qnn.bc.heads``) without forking the trainer. When ``None`` the
    canonical ``_CombatObjectiveNet`` is built from ``config.model``.

    Fine-tuning from a seed checkpoint ignores ``model_factory`` because
    ``QNNPolicy.load`` reconstructs the saved architecture; passing both
    is rejected to fail loud rather than silently dropping the factory.
    """
    set_global_seed(config.seed)
    # Episode shuffle uses a fixed seed (42) independent of the model init
    # seed, so all ablation runs see the same episode ordering per epoch.
    # This rng is saved/restored in checkpoints so resume produces the
    # same ordering as a continuous run.
    _SHUFFLE_SEED = 42
    rng = np.random.default_rng(_SHUFFLE_SEED)

    # Fall back to PUSHGATEWAY_URL env var if config doesn't specify one.
    if not config.prometheus_pushgateway_url:
        env_url = os.environ.get("PUSHGATEWAY_URL", "")
        if env_url:
            object.__setattr__(config, "prometheus_pushgateway_url", env_url)
            print(f"  [bc] Prometheus pushgateway: {env_url}")

    if not str(config.output_dir).strip():
        raise RuntimeError("Behavior cloning requires output_dir")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load precomputed .npy caches (produced by python -m qnn.bc.collect)
    bc_data_dir = Path(config.bc_data_dir) if hasattr(config, "bc_data_dir") else Path(config.output_dir).parent
    train_cache = bc_data_dir / "precomputed_train"
    val_cache = bc_data_dir / "precomputed_val"
    if not train_cache.exists():
        raise RuntimeError(f"BC training data not found at {train_cache}. Run python -m qnn.bc.collect first.")

    # Verify the dataset identity matches what the run expects. The check
    # is strict (raises FingerprintMismatch on mismatch or absent
    # fingerprint.json); empty fingerprint strings are rejected at config
    # load via _require_string in build_run_bc_config.
    from qnn import collection_fingerprint
    actual_fp = collection_fingerprint.verify(
        expected_fingerprint=config.collection_fingerprint,
        data_dir=bc_data_dir,
    )
    print(f"  [bc] collection fingerprint: {actual_fp['fingerprint']}")

    head_loss_weights = _effective_head_loss_weights(config.head_loss_weights)
    required_actions_set: set[str] = set()
    if config.model.use_weapon_head and head_loss_weights.get("weapon", 1.0) > 0.0:
        required_actions_set.add("weapon")
    # op_input is required only when the trainer-side mask is enabled;
    # corpora without it (pre-emission) still train cleanly when the
    # toggle is off.
    if config.op_input_mask:
        required_actions_set.add("op_input")
    required_actions = frozenset(required_actions_set)

    print(f"  [bc] Loading training data: {train_cache}")
    if config.segment_mask:
        print(f"  [bc] segment_mask: {config.segment_mask}")
    if config.token_mask:
        print(f"  [bc] token_mask: {config.token_mask}")
    train_episodes = _load_precomputed(
        train_cache, required_actions=required_actions,
        segment_mask=config.segment_mask,
        token_mask=config.token_mask,
    )
    val_episodes = _load_precomputed(
        val_cache, required_actions=required_actions,
        segment_mask=config.segment_mask,
        token_mask=config.token_mask,
    ) if val_cache.exists() else []

    sample_counts = {
        "train": sum(ep.n_samples for ep in train_episodes),
        "val": sum(ep.n_samples for ep in val_episodes),
    }

    if sample_counts["train"] <= 0:
        raise RuntimeError("No training samples available")

    # Configure mixed-precision autocast via the env var that QNNPolicy reads.
    os.environ["QNN_AUTOCAST_DTYPE"] = config.dtype
    print(f"  [bc] dtype={config.dtype}")

    obs_dim = OBS_DIM
    if seed_checkpoint and Path(seed_checkpoint).exists():
        if model_factory is not None:
            raise RuntimeError(
                "model_factory is incompatible with seed_checkpoint — "
                "QNNPolicy.load rebuilds the saved architecture itself."
            )
        print(f"  [bc] Fine-tuning from seed: {seed_checkpoint}")
        model = QNNPolicy.load(seed_checkpoint, device=config.device)
    else:
        model = QNNPolicy(
            obs_dim=obs_dim,
            model=config.model,
            jump_pos_weight=config.jump_pos_weight,
            fire_focal_gamma=config.fire_focal_gamma,
            fire_focal_alpha=config.fire_focal_alpha,
            fire_distance_sigma=config.fire_distance_sigma,
            jump_distance_sigma=config.jump_distance_sigma,
            seed=config.seed,
            device=config.device,
            model_factory=model_factory,
        )
    # op_input_mask is a training-time toggle, not a ModelConfig field —
    # set after construction so the same checkpoint can be retrained
    # either way (and so seed_checkpoint resumes pick up the run's
    # current config rather than the seed run's).
    model.op_input_mask = bool(config.op_input_mask)

    weights = fire_class_weights(
        train_episodes,
        head_loss_weights=head_loss_weights,
        override=float(config.fire_pos_weight_override),
        device=model.device,
    )

    # GPU-resident fast path: precompute concat tensors per obs/action key
    # once and feed them to run_epoch_gpu_resident. Only safe for memoryless
    # models (no GRU); falls back to the prefetch+plan pipeline otherwise.
    _gpu_train_obs: Dict[str, "torch.Tensor"] = {}
    _gpu_train_actions: Dict[str, "torch.Tensor"] = {}
    _gpu_val_obs: Dict[str, "torch.Tensor"] = {}
    _gpu_val_actions: Dict[str, "torch.Tensor"] = {}
    _use_gpu_resident = (
        bool(config.preload_to_gpu)
        and not bool(getattr(config.model, "use_gru", False))
        and int(config.sequence_length) == 0
    )
    if _use_gpu_resident:
        print(
            f"  [bc] preload_to_gpu=true: concatenating "
            f"{sum(ep.n_samples for ep in train_episodes)} train + "
            f"{sum(ep.n_samples for ep in val_episodes)} val frames to {model.device}"
        )
        _t0 = _time.monotonic()
        _gpu_train_obs, _gpu_train_actions = _preload_episodes_to_gpu(
            train_episodes, model.device
        )
        _gpu_val_obs, _gpu_val_actions = _preload_episodes_to_gpu(
            val_episodes, model.device
        )
        print(f"  [bc] preload done in {_time.monotonic() - _t0:.1f}s")
        # On Strix-Halo-class unified-memory APUs, CPU and "VRAM" share
        # the same pool. Holding the per-episode numpy arrays alongside
        # the GPU-resident tensors doubles the working set. Drop the
        # CPU side now that gpu_obs/gpu_actions own the data —
        # provided we won't need to re-run a CPU pass for train_eval.
        if config.train_eval_interval == 0:
            for _ep in train_episodes:
                _ep.obs = {}
                _ep.actions = {}
            for _ep in val_episodes:
                _ep.obs = {}
                _ep.actions = {}
            import gc as _gc
            _gc.collect()
            print(f"  [bc] released CPU episode arrays (train_eval_interval=0)")

    # Parse per-head loss weights from JSON string if provided.
    hlw: Dict[str, float] | None = None
    if config.head_loss_weights:
        hlw = dict(head_loss_weights)

    best_val_loss = float("inf")
    best_epoch = -1
    history: list[Dict[str, float]] = []
    start_epoch = 0

    # Regression-based stopping state.
    _best_move = float("inf")
    _best_look = float("inf")
    _best_max_reg = float("inf")  # for checkpoint selection: min of max(move_reg, look_reg)
    _best_reg_epoch = -1
    _reg_violations = 0

    # NAS archive: save every epoch checkpoint to SMB share for offsite backup.
    _NAS_CHECKPOINTS = r"\\pi.local\nqcorpus\bc_checkpoints"
    _smb_available = False
    try:
        import smbclient
        smbclient.ClientConfig(username="guest", password="", require_secure_negotiate=False)
        smbclient.register_session(
            "pi.local", username="guest", password="",
            auth_protocol="ntlm", require_signing=False,
        )
        _variant_name = output.parent.name or output.name
        _variant_dir = _NAS_CHECKPOINTS + "\\" + _variant_name
        smbclient.makedirs(_variant_dir, exist_ok=True)
        _smb_available = True
        print(f"  [bc] NAS archive available: {_variant_dir}")
    except Exception:
        _smb_available = False
        print("  [bc] NAS archive not available — skipping offsite backup")

    # Mid-epoch state: rolling file for deterministic resume within an epoch.
    mid_epoch_path = output / "snapshot.pt"
    _MID_EPOCH_SAVE_INTERVAL = config.snapshot_interval

    # Resume from checkpoint if available.
    checkpoint_path = output / "bc_training_checkpoint.pt"
    if checkpoint_path.exists():
        import torch as _torch_resume
        from qnn.utils.checkpoint_converter import migrate_entity_embed, migrate_self_scalars
        ckpt = _torch_resume.load(checkpoint_path, map_location=model.device, weights_only=False)
        migrate_entity_embed(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
        migrate_self_scalars(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
        model.model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        _best_move = ckpt.get("_best_move", float("inf"))
        _best_look = ckpt.get("_best_look", float("inf"))
        _best_max_reg = ckpt.get("_best_max_reg", float("inf"))
        _best_reg_epoch = ckpt.get("_best_reg_epoch", -1)
        _reg_violations = ckpt.get("_reg_violations", 0)
        # Optimizer state restored after first supervised step creates it.
        _resume_optimizer_state = ckpt.get("optimizer_state_dict")
        # Restore rng state so resume produces the same episode ordering
        # as a continuous run.
        _saved_rng_state = ckpt.get("rng_state")
        if _saved_rng_state is not None:
            rng.bit_generator.state = _saved_rng_state
        print(f"  [bc] Resuming from epoch {start_epoch} (best_val={best_val_loss:.4f} at epoch {best_epoch})")
    else:
        _resume_optimizer_state = None

    # Mid-epoch resume: if we have a mid-epoch state file, use it to
    # resume within the current epoch instead of restarting it.
    _mid_epoch_resume: _MidEpochState | None = None
    if mid_epoch_path.exists():
        import torch as _torch_mid
        try:
            _mid_ckpt = _torch_mid.load(mid_epoch_path, map_location=model.device, weights_only=False)
            if _mid_ckpt.get("epoch") == start_epoch:
                model.model.load_state_dict(_mid_ckpt["model_state_dict"])
                _resume_optimizer_state = _mid_ckpt.get("optimizer_state_dict")
                _mid_epoch_resume = _mid_ckpt["mid_epoch_state"]
                rng.bit_generator.state = _mid_ckpt["rng_state"]
                print(f"  [bc] Mid-epoch resume: epoch {start_epoch}, "
                      f"step {_mid_epoch_resume.opt_steps}, "
                      f"chunk {_mid_epoch_resume.next_episode}")
            else:
                mid_epoch_path.unlink()
        except Exception as exc:
            print(f"  [bc] Mid-epoch state load failed: {exc}")
            mid_epoch_path.unlink(missing_ok=True)

    # torch.compile: tested but net negative for this model size (189K params).
    # The fused kernels don't help when individual ops are already microseconds,
    # and the compile wrapper adds overhead (val: 100s → 120s per epoch).
    # Revisit if model size increases significantly.

    # Per-step reporting: aggregate every ~1024 samples, then wall-clock gate
    # actual logging/flushes so perf runs do not spend most of their time
    # printing and rewriting the step log.
    _report_every = max(1, 1024 // max(config.batch_size, 1)) if config.batch_size > 0 else 0
    _step_log: list[Dict[str, float]] = []
    _step_report_interval = max(int(config.step_report_interval_seconds), 0)
    _last_step_report_time = _time.monotonic() - _step_report_interval

    def _on_step(step_metrics: Dict[str, float]) -> None:
        nonlocal _last_step_report_time
        _now = _time.monotonic()
        if _step_report_interval > 0 and (_now - _last_step_report_time) < _step_report_interval:
            return
        _last_step_report_time = _now
        step_metrics["epoch"] = float(epoch)
        _step_log.append(step_metrics)
        mae_parts = [f"{k}={v:.4f}" for k, v in sorted(step_metrics.items()) if k.startswith("mae_")]
        print(f"  [bc]   step {int(step_metrics.get('opt_step', 0)):>5d}  "
              f"loss={step_metrics.get('loss', 0):.4f}  "
              f"{'  '.join(mae_parts)}")
        # Flush step log to disk every report interval for live monitoring.
        write_json(output / "bc_step_log.json", {"steps": _step_log})

    def _save_mid_epoch(state: _MidEpochState) -> None:
        bc_opt = model._optimizers.get("bc")
        mid_data = {
            "epoch": epoch,
            "model_state_dict": {
                k.replace("_orig_mod.", ""): v
                for k, v in model.model.state_dict().items()
            },
            "optimizer_state_dict": bc_opt.state_dict() if bc_opt else None,
            "mid_epoch_state": state,
            "rng_state": rng.bit_generator.state,
        }
        torch.save(mid_data, mid_epoch_path)

    _active_lr = config.lr
    _lr_override_path = output / "lr_override.json"

    import math as _math
    from datetime import datetime as _datetime, timezone as _tz

    import gc as _gc

    _prev_epoch_weights: Dict[str, torch.Tensor] | None = None

    for epoch in range(start_epoch, config.epochs):
        # Reclaim Python + CUDA allocator pool at each epoch boundary.
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optional linear decay of the ud-axis pos_weight across epochs.
        # Lets us start with high pos_weight (push recall hard while the head
        # is randomly initialized) and end with low pos_weight (let precision
        # recover as the head calibrates).  -1.0 sentinel disables decay.
        if config.jump_pos_weight_end > 0 and config.epochs > 1:
            alpha = float(epoch) / float(config.epochs - 1)
            current_pw = (1.0 - alpha) * float(config.jump_pos_weight) + alpha * float(config.jump_pos_weight_end)
            model.jump_pos_weight = current_pw
            print(f"  [bc] jump_pos_weight (decay): epoch {epoch}/{config.epochs - 1}  alpha={alpha:.3f}  pw={current_pw:.3f}")
        # Snapshot weights at the start of this epoch so we can compute
        # L2 drift from the end-of-last-epoch state as a "is the model still
        # actively changing?" signal.
        _epoch_start_weights = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
        # Hot-reload LR: drop {"lr": 0.001, "lr_min": 0.0003} into lr_override.json.
        _lr = config.lr
        _lr_min = config.lr_min
        if _lr_override_path.exists():
            try:
                _ovr = _json.loads(_lr_override_path.read_text())
                _lr = float(_ovr.get("lr", _lr))
                _lr_min = float(_ovr.get("lr_min", _lr_min))
                print(f"  [bc] lr_override.json: lr={_lr}, lr_min={_lr_min}")
            except Exception as exc:
                print(f"  [bc] lr_override.json parse error: {exc}")

        # LR schedule: optional linear warmup then optional cosine decay.
        _warmup = config.warmup_epochs
        if _warmup > 0 and epoch < _warmup:
            # Linear warmup from lr_min (or near-zero) to lr.
            _base = _lr_min if _lr_min > 0 else _lr * 0.01
            _active_lr = _base + (_lr - _base) * (epoch / _warmup)
        elif _lr_min > 0:
            # Cosine decay from lr to lr_min over post-warmup epochs.
            _post_warmup = epoch - _warmup
            _post_total = max(config.epochs - 1 - _warmup, 1)
            progress = _post_warmup / _post_total
            _active_lr = _lr_min + 0.5 * (_lr - _lr_min) * (1 + _math.cos(_math.pi * progress))
        else:
            _active_lr = _lr

        if epoch == start_epoch or epoch > start_epoch:
            print(f"  [bc] LR={_active_lr:.6f}")

        _t_train_start = _time.monotonic()
        if _use_gpu_resident:
            train_metrics = _run_epoch_gpu_resident(
                model,
                _gpu_train_obs,
                _gpu_train_actions,
                batch_size=config.batch_size,
                class_weights=weights,
                lr=_active_lr,
                rng=rng,
                max_grad_norm=config.max_grad_norm,
                head_loss_weights=hlw,
            )
        else:
            train_metrics = _run_precomputed_supervised(
                model,
                train_episodes,
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                class_weights=weights,
                lr=_active_lr,
                rng=rng,
                max_grad_norm=config.max_grad_norm,
                tbptt_limit=config.tbptt_limit,
                head_loss_weights=hlw,
                step_callback=_on_step,
                report_every=_report_every,
                report_interval_seconds=float(_step_report_interval),
                pin_memory=config.pin_memory,
                prefetch=config.prefetch,
                microbatch_size=config.microbatch_size,
                save_state_callback=_save_mid_epoch,
                snapshot_interval=_MID_EPOCH_SAVE_INTERVAL,
                resume_state=_mid_epoch_resume,
            )
        # Mid-epoch state consumed — don't reuse on next epoch.
        _mid_epoch_resume = None
        _t_train_end = _time.monotonic()
        # Restore optimizer state on first epoch after resume.
        if _resume_optimizer_state is not None:
            bc_opt = model._optimizers.get("bc")
            if bc_opt is not None:
                bc_opt.load_state_dict(_resume_optimizer_state)
                _resume_optimizer_state = None
        _t_val_start = _time.monotonic()
        if _use_gpu_resident:
            val_metrics = _run_epoch_gpu_resident(
                model,
                _gpu_val_obs,
                _gpu_val_actions,
                batch_size=config.batch_size,
                head_loss_weights=hlw,
            )
        else:
            val_metrics = _run_precomputed_supervised(
                model,
                val_episodes,
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                tbptt_limit=config.tbptt_limit,
                head_loss_weights=hlw,
                pin_memory=config.pin_memory,
                prefetch=config.prefetch,
                microbatch_size=config.microbatch_size,
            )
        _t_val_only_end = _time.monotonic()
        train_proxy_sum, train_proxy_gap, train_eval_reasons = _train_eval_schedule(
            epoch,
            history,
            train_metrics,
            val_metrics,
            interval=config.train_eval_interval,
            gap_threshold=config.train_eval_gap_threshold,
            val_regression_threshold=config.train_eval_val_regression_threshold,
            train_improve_threshold=config.train_eval_train_improve_threshold,
        )
        train_eval_metrics: Dict[str, float] = {}
        train_eval_sum: float | None = None
        _train_eval_secs = 0.0
        train_eval_ran = bool(val_episodes) and bool(train_eval_reasons)
        if train_eval_ran:
            # Clean train eval (model.eval mode, no dropout) on a train subset
            # only when scheduled or when proxy metrics suggest a gap issue.
            _t_train_eval_start = _time.monotonic()
            train_eval_metrics = _run_precomputed_supervised(
                model,
                train_episodes[:len(val_episodes)],
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                tbptt_limit=config.tbptt_limit,
                head_loss_weights=hlw,
                pin_memory=config.pin_memory,
                prefetch=config.prefetch,
            )
            _train_eval_secs = _time.monotonic() - _t_train_eval_start
            train_eval_sum = _selection_score(train_eval_metrics)
        _t_val_end = _time.monotonic()
        _train_secs = _t_train_end - _t_train_start
        _val_only_secs = _t_val_only_end - _t_val_start
        _val_secs = _t_val_end - _t_val_start
        train_rows = float(train_metrics.get("n_rows", sample_counts["train"]))
        val_rows = float(val_metrics.get("n_rows", sample_counts["val"]))
        train_eval_rows = float(train_eval_metrics.get("n_rows", 0.0)) if train_eval_ran else 0.0
        train_rows_per_sec = train_rows / _train_secs if _train_secs > 0 else 0.0
        val_rows_per_sec = val_rows / _val_only_secs if _val_only_secs > 0 else 0.0
        train_eval_rows_per_sec = train_eval_rows / _train_eval_secs if _train_eval_secs > 0 else 0.0
        _wall_clock = _datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if train_eval_ran:
            print(
                f"  [bc] timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  "
                f"train_eval={_train_eval_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]"
            )
        else:
            print(f"  [bc] timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]")
        # Headline per-head summary: one number per head (F1 where the
        # class imbalance makes accuracy misleading), plus per-axis move F1
        # so axis-specific regressions surface in the log.
        _headline_keys = (
            "acc_target",
            "f1_move", "f1_move_fb", "f1_move_lr", "f1_move_ud",
            "cos_sim_look",
            "f1_fire",
            "f1_weapon",
        )
        mae_str = "  ".join(
            f"{k}={float(val_metrics[k]):.4f}"
            for k in _headline_keys if k in val_metrics
        )

        # Composite selection: target acc + move/fire/weapon macro-F1 + look cos.
        val_selection_score = _selection_score(val_metrics)
        selection_metric = val_selection_score
        improved = selection_metric < best_val_loss

        # Weight drift: L2 of (weights now) - (weights at epoch start).
        # Non-zero drift in a plateau = model still reorganizing; zero = stuck.
        # Accumulate squared diffs on GPU, single host sync at the end.
        _cur_state = model.model.state_dict()
        _diffs = [((_cur_state[_k] - _start_v) ** 2).sum()
                  for _k, _start_v in _epoch_start_weights.items()
                  if _cur_state[_k].dtype.is_floating_point]
        _weight_drift_l2 = torch.stack(_diffs).sum().sqrt().item() if _diffs else 0.0

        _grad_mean = train_metrics.get("grad_norm_mean")
        _grad_max = train_metrics.get("grad_norm_max")

        epoch_line = (
            f"  [bc] Epoch {epoch + 1}/{config.epochs}  "
            f"train_proxy={train_proxy_sum:.4f}  "
            f"val={val_selection_score:.4f}  "
            f"proxy_gap={train_proxy_gap:+.4f}  "
        )
        if train_eval_ran and train_eval_sum is not None:
            epoch_line += (
                f"train_eval={train_eval_sum:.4f}  "
                f"gap={val_selection_score - train_eval_sum:+.4f}  "
                f"[{','.join(train_eval_reasons)}]  "
            )
        else:
            epoch_line += "train_eval=skipped  "
        epoch_line += f"{'*' if improved else ''}  "
        if _grad_mean is not None:
            epoch_line += (
                f"grad_mean={_grad_mean:.3f}  "
                f"grad_max={_grad_max:.3f}  "
            )
        epoch_line += f"drift={_weight_drift_l2:.3f}  "
        epoch_line += f"train_rps={train_rows_per_sec:.1f}  val_rps={val_rows_per_sec:.1f}  "
        epoch_line += mae_str
        print(epoch_line)

        # Assemble and record per-epoch metrics.
        epoch_metrics: Dict[str, Any] = {
            "epoch": float(epoch),
            "train_secs": _train_secs,
            "val_secs": _val_secs,
            "val_only_secs": _val_only_secs,
            "train_eval_secs": _train_eval_secs,
            "wall_clock": _wall_clock,
            "train_proxy_sum": train_proxy_sum,
            "train_proxy_gap": train_proxy_gap,
            "train_eval_ran": train_eval_ran,
            "train_eval_reason": ",".join(train_eval_reasons),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_eval_rows": train_eval_rows,
            "effective_train_rows_per_sec": train_rows_per_sec,
            "effective_val_rows_per_sec": val_rows_per_sec,
            "effective_train_eval_rows_per_sec": train_eval_rows_per_sec,
        }
        epoch_metrics["weight_drift_l2"] = _weight_drift_l2

        for key, value in train_metrics.items():
            if key == "_next_hidden":
                continue
            epoch_metrics[f"train_{key}"] = float(value)
        for key, value in train_eval_metrics.items():
            if key == "_next_hidden":
                continue
            epoch_metrics[f"train_eval_{key}"] = float(value)
        for key, value in val_metrics.items():
            if key == "_next_hidden":
                continue
            epoch_metrics[f"val_{key}"] = float(value)
        history.append(epoch_metrics)

        # Write history and step log incrementally so results survive crashes.
        write_json(output / "bc_history.json", {"history": history})
        if _step_log:
            write_json(output / "bc_step_log.json", {"steps": _step_log})

        # Epoch sentinel: external watchers can poll this file to detect
        # epoch completion across any training mode (BC, PPO, etc.).
        (output / "epoch_done").write_text(
            json.dumps({"epoch": epoch, "wall_clock": _wall_clock, "mode": "bc"}) + "\n"
        )

        # Push metrics to Prometheus pushgateway (no-op when URL is empty).
        if config.prometheus_pushgateway_url:
            _push_metrics_to_prometheus(
                config.prometheus_pushgateway_url,
                epoch_metrics,
                epoch,
                variant=Path(config.output_dir).name,
                config=config,
            )

        # Regression-based stopping: track per-head bests and regression.
        val_move = val_metrics.get("mae_move", float("inf"))
        val_look = val_metrics.get("mae_look", float("inf"))
        _best_move = min(_best_move, val_move)
        _best_look = min(_best_look, val_look)
        move_reg = val_move - _best_move
        look_reg = val_look - _best_look

        # Checkpoint selection: best val MAE sum.  Regression gate is purely
        # for stopping, not model selection.
        if selection_metric < best_val_loss:
            best_val_loss = selection_metric
            best_epoch = epoch
            model.save(output / "bc_best_model.pth")

        if move_reg > config.regression_threshold or look_reg > config.regression_threshold:
            _reg_violations += 1
        else:
            _reg_violations = 0

        print(f"  [bc]   regression: move={move_reg:+.4f} look={look_reg:+.4f} "
              f"violations={_reg_violations}/{config.regression_patience}")

        # Save resumable checkpoint every epoch (latest + epoch-stamped).
        bc_opt = model._optimizers.get("bc")
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": {
                k.replace("_orig_mod.", ""): v
                for k, v in model.model.state_dict().items()
            },
            "optimizer_state_dict": bc_opt.state_dict() if bc_opt else None,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "_best_move": _best_move,
            "_best_look": _best_look,
            "_best_max_reg": _best_max_reg,
            "_best_reg_epoch": _best_reg_epoch,
            "_reg_violations": _reg_violations,
            "rng_state": rng.bit_generator.state,
        }
        torch.save(ckpt_data, checkpoint_path)
        # Epoch completed cleanly — remove the rolling mid-epoch state.
        mid_epoch_path.unlink(missing_ok=True)
        # Epoch-stamped copy so we can resume from any epoch.
        epoch_ckpt_dir = output / "checkpoints"
        epoch_ckpt_dir.mkdir(exist_ok=True)
        torch.save(ckpt_data, epoch_ckpt_dir / f"bc_checkpoint_epoch{epoch:03d}.pt")

        # Archive checkpoint and best model to NAS.
        if _smb_available:
            try:
                import smbclient as _smb
                import shutil as _shutil
                for src in [checkpoint_path, output / "bc_best_model.pth"]:
                    if src.exists():
                        nas_dest = _variant_dir + "\\" + src.name
                        with open(src, "rb") as local_f:
                            with _smb.open_file(nas_dest, mode="wb") as remote_f:
                                _shutil.copyfileobj(local_f, remote_f)
            except Exception as exc:
                print(f"  [bc] NAS archive failed: {exc}")

        if _reg_violations >= config.regression_patience:
            print(f"  [bc] Regression stop: {config.regression_patience} consecutive epochs "
                  f"above threshold {config.regression_threshold}. Best epoch: {best_epoch + 1}")
            break

    if best_epoch < 0:
        model.save(output / "bc_best_model.pth")

    # Free the training model + optimizer state + the GPU-resident TRAIN
    # tensors before we load the best checkpoint for the final val pass.
    # Without this, we briefly hold (training model + final_model + train
    # data + val data) all on GPU, and the val forward's activation
    # allocation OOMs — especially when 2-3 trainers are sharing the GPU.
    # Keep _gpu_val_obs/_gpu_val_actions: they'll be reused below.
    del model
    if _use_gpu_resident:
        _gpu_train_obs.clear()
        _gpu_train_actions.clear()
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_model = QNNPolicy.load(
        output / "bc_best_model.pth", device=config.device, model_factory=model_factory,
    )

    # Route the final val through whichever path produced the training
    # data — if we preloaded to GPU, use the resident path (cheap reuse
    # of _gpu_val_obs; no second mmap+collate). Otherwise fall back to
    # the lane-packed val.
    if val_episodes:
        if _use_gpu_resident:
            final_val_metrics = _run_epoch_gpu_resident(
                final_model,
                _gpu_val_obs,
                _gpu_val_actions,
                batch_size=config.batch_size,
                head_loss_weights=hlw,
            )
        else:
            final_val_metrics = _run_precomputed_supervised(
                final_model,
                val_episodes,
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                tbptt_limit=config.tbptt_limit,
                pin_memory=config.pin_memory,
                prefetch=config.prefetch,
            )
    else:
        final_val_metrics = {"loss": 0.0}

    summary: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": float(final_val_metrics["loss"]),
        "num_train_samples": int(sample_counts["train"]),
        "num_val_samples": int(sample_counts["val"]),
        "epochs_ran": len(history),
    }
    if history:
        last = history[-1]
        for key in (
            "effective_train_rows_per_sec",
            "effective_val_rows_per_sec",
            "effective_train_eval_rows_per_sec",
            "train_rows",
            "val_rows",
            "train_eval_rows",
        ):
            if key in last:
                summary[f"final_{key}"] = float(last[key])
    if actual_fp is not None:
        summary["collection_fingerprint"] = actual_fp["fingerprint"]
    for key, value in final_val_metrics.items():
        if key == "_next_hidden":
            continue
        summary[f"final_val_{key}"] = float(value)

    write_json(output / "bc_history.json", {"history": history})
    write_json(output / "bc_summary.json", summary)
    write_experiment_manifest(output / "bc_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}


# ── Runner entry point (called by run.router) ──────────────────────

def run(ctx: "RunnerContext") -> dict[str, object]:
    """Run BC pipeline from a frozen run directory."""
    import dataclasses as _dc
    import time as _time

    from qnn.run.config import build_run_bc_config
    from qnn.run.common import RunnerContext, base_results, finalize_results, prepare_bc_run_outputs

    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    bc_cfg = build_run_bc_config(ctx.run_cfg, ctx.device)
    prepare_bc_run_outputs(ctx.run_cfg, resume=ctx.resume)

    bc_data_dir = Path(bc_cfg.get("bc_data_dir", ""))
    train_cache = bc_data_dir / "precomputed_train"
    if not train_cache.exists():
        raise RuntimeError(
            f"BC training data not found at {train_cache}. "
            f"Run python -m qnn.bc.collect first."
        )

    seed_checkpoint = str(ctx.run_cfg.get("checkpoint_path", ""))
    started = _time.monotonic()
    valid_keys = {f.name for f in _dc.fields(BCConfig)}
    unknown = sorted(set(bc_cfg) - valid_keys)
    if unknown:
        raise RuntimeError(
            f"BC config has {len(unknown)} unknown key(s) (typo or removed feature): {unknown}. "
            "Either remove them from the run's train.json/model.json or add them to BCConfig."
        )
    results["bc"] = run_behavior_cloning(BCConfig(**bc_cfg), seed_checkpoint=seed_checkpoint)
    stage_timings["bc"] = _time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)


# ── Standalone eval entry point ────────────────────────────────────────────────
# python -m qnn.bc.train --eval-only --run-dir runs/bc/<name> [--data-dir ...]

def _eval_only(run_dir: Path, data_dir: Path | None, device: str, batch_size: int) -> None:
    import json as _json
    checkpoint = run_dir / "checkpoints" / "bc_best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best-model checkpoint at {checkpoint}")

    if data_dir is None:
        machine_cfg = _json.loads((run_dir / "config" / "machine.json").read_text())
        data_dir = Path(machine_cfg["bc_data_dir"])

    val_cache = data_dir / "precomputed_val"
    if not val_cache.exists():
        raise FileNotFoundError(f"Val cache not found: {val_cache}")

    train_cfg = _json.loads((run_dir / "config" / "train.json").read_text())
    tbptt = int(train_cfg.get("tbptt_limit", 256))

    print(f"  checkpoint : {checkpoint}")
    print(f"  val data   : {val_cache}")
    print(f"  device     : {device}  batch_size: {batch_size}  tbptt: {tbptt}")

    model = QNNPolicy.load(str(checkpoint), device=device)
    model.model.eval()

    val_episodes = _load_precomputed(val_cache)
    print(f"  val episodes: {len(val_episodes)}")

    metrics = _run_precomputed_supervised(
        model,
        val_episodes,
        batch_size=batch_size,
        sequence_length=0,
        tbptt_limit=tbptt,
        pin_memory=False,
        prefetch=0,
    )

    print("\n--- val metrics ---")
    for k, v in sorted(metrics.items()):
        if k == "_next_hidden":
            continue
        print(f"  {k:<30s}  {v:.6f}")


if __name__ == "__main__":
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(description="Evaluate a BC best-model checkpoint on the val set.")
    _ap.add_argument("--eval-only", action="store_true", required=True)
    _ap.add_argument("--run-dir", type=Path, required=True, help="Run directory (contains config/ and checkpoints/)")
    _ap.add_argument("--data-dir", type=Path, default=None, help="Override bc_data_dir from machine.json")
    _ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    _ap.add_argument("--batch-size", type=int, default=256)
    _args = _ap.parse_args()
    _eval_only(_args.run_dir, _args.data_dir, _args.device, _args.batch_size)
