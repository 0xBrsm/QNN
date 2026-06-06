"""Supervised BC trainer.

Two data pipelines (resident vs streaming), one training platform.
:class:`Source` carries device-resident or lazy tensors plus episode
metadata; :func:`run_epoch` picks lane-packed batches when the model is
recurrent and frame-shuffled batches otherwise, and feeds both to
:func:`train_on_batches`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import heapq

import numpy as np
import torch

from qnn.model.policy import QNNPolicy

_RAW_SUM_METRIC_PREFIXES = (
    "n_", "correct_", "l1_sum_", "tp_", "fp_", "fn_", "target_pos_", "pred_pos_", "pred_target_",
)
_AVERAGED_METRIC_PREFIXES = (
    "acc_", "mae_", "cos_sim_", "f1_", "precision_", "recall_",
    "pos_rate_", "pred_rate_", "confidence_", "balanced_acc_",
    # Per-head soft-distribution diagnostics on the target head — NLL
    # (loss_target), present-weighted entropy / KL / Brier / top-1 mass.
    # Listed explicitly so we don't accidentally average raw-sum
    # target_* keys (those are caught above via the n_/tp_/fp_/fn_/etc.
    # prefixes that fire first in the dispatch order).
    "loss_", "target_present_", "target_entropy", "target_kl",
    "target_brier", "target_top1_",
)


@dataclass(slots=True)
class PrecomputedEpisode:
    """One episode's observations and actions as contiguous arrays.

    ``entity_indptr`` is the per-episode cumulative ``entity_count``
    (length ``n_samples + 1``) for the native-format token-indexed
    fields in ``obs`` (``entity_rel``, ``entity_types``, ...). When set,
    those fields are stored unpadded as ``(total_tokens_in_episode,
    ...)`` and consumers (chunked prefetch, GPU-resident preload)
    materialize the padded ``(n_rows, MAX_TOKEN_OBJECTS, ...)`` layout
    on demand. When ``None``, ``obs`` already holds the legacy padded
    layout (filter/mask path that needs padded semantics for predicate
    evaluation, plus synthetic fixtures without entity_count).
    """
    obs: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    n_samples: int
    sort_key: tuple[int, int, int] = (0, 0, 0)
    entity_indptr: np.ndarray | None = None

    def materialize_padded_obs(self, n_max: int) -> dict[str, np.ndarray]:
        """Return a fully-padded copy of ``obs`` for consumers that
        need the legacy ``(n_samples, n_max, ...)`` entity layout.

        Used by the GPU-resident preload and tests. Allocates one
        padded buffer per token-indexed field — the cost is amortized
        across the rest of training. Returns ``self.obs`` unchanged when
        ``entity_indptr`` is ``None`` (already padded / no token fields).
        """
        if self.entity_indptr is None:
            return self.obs
        # Local import to avoid a circular dependency: train imports
        # supervised_loop, but this method lives on the loop's data
        # carrier and needs the pad helper that lives in train.
        from qnn.bc.train import _pad_entity_batch
        padded_tok = _pad_entity_batch(
            self.obs, self.entity_indptr, 0, self.n_samples, n_max,
        )
        out: dict[str, np.ndarray] = {}
        for k, v in self.obs.items():
            out[k] = padded_tok[k] if k in padded_tok else v
        return out


def _flush_tensor_dict(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Single GPU→CPU sync for a dict of 0-d tensors via stack + tolist."""
    if not tensors:
        return {}
    keys = list(tensors.keys())
    vals = torch.stack([tensors[k] for k in keys]).tolist()
    return dict(zip(keys, vals))


def _stable_binary_metrics_from_counts(
    result: Dict[str, float],
    *,
    prefix: str,
) -> None:
    tp = float(result.get(f"tp_{prefix}", 0.0))
    fp = float(result.get(f"fp_{prefix}", 0.0))
    fn = float(result.get(f"fn_{prefix}", 0.0))
    tn = float(result.get(f"tn_{prefix}", 0.0))
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    acc = ((tp + tn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) > 0.0 else 0.0

    if f"precision_{prefix}" in result:
        result[f"precision_{prefix}_batch"] = result[f"precision_{prefix}"]
    if f"recall_{prefix}" in result:
        result[f"recall_{prefix}_batch"] = result[f"recall_{prefix}"]
    if f"f1_{prefix}" in result:
        result[f"f1_{prefix}_batch"] = result[f"f1_{prefix}"]
    if f"acc_{prefix}" in result:
        result[f"acc_{prefix}_batch"] = result[f"acc_{prefix}"]

    result[f"precision_{prefix}"] = precision
    result[f"recall_{prefix}"] = recall
    result[f"f1_{prefix}"] = f1
    result[f"acc_{prefix}"] = acc
    result[f"precision_{prefix}_global"] = precision
    result[f"recall_{prefix}_global"] = recall
    result[f"f1_{prefix}_global"] = f1
    result[f"acc_{prefix}_global"] = acc


def _stable_weapon_metrics_from_counts(result: Dict[str, float]) -> None:
    classes = sorted(k[len("tp_weapon_"):] for k in result if k.startswith("tp_weapon_"))
    if not classes:
        return

    per_prec: list[float] = []
    per_rec: list[float] = []
    per_f1: list[float] = []
    per_support: list[float] = []
    micro_tp = 0.0
    micro_fp = 0.0
    micro_fn = 0.0
    for cls_name in classes:
        tp = float(result.get(f"tp_weapon_{cls_name}", 0.0))
        fp = float(result.get(f"fp_weapon_{cls_name}", 0.0))
        fn = float(result.get(f"fn_weapon_{cls_name}", 0.0))
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / support if support > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

        p_key = f"precision_weapon_{cls_name}"
        r_key = f"recall_weapon_{cls_name}"
        f_key = f"f1_weapon_{cls_name}"
        if p_key in result:
            result[f"{p_key}_batch"] = result[p_key]
        if r_key in result:
            result[f"{r_key}_batch"] = result[r_key]
        if f_key in result:
            result[f"{f_key}_batch"] = result[f_key]
        result[p_key] = precision
        result[r_key] = recall
        result[f_key] = f1
        result[f"{p_key}_global"] = precision
        result[f"{r_key}_global"] = recall
        result[f"{f_key}_global"] = f1

        per_prec.append(precision)
        per_rec.append(recall)
        per_f1.append(f1)
        per_support.append(support)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_precision = float(sum(per_prec) / len(per_prec))
    macro_recall = float(sum(per_rec) / len(per_rec))
    macro_f1 = float(sum(per_f1) / len(per_f1))
    total_support = float(sum(per_support))
    weighted_f1 = (
        float(sum(f * s for f, s in zip(per_f1, per_support)) / total_support)
        if total_support > 0.0 else 0.0
    )
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0.0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0.0 else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0.0 else 0.0
    )

    if "f1_weapon" in result:
        result["f1_weapon_batch"] = result["f1_weapon"]
    result["f1_weapon"] = macro_f1
    result["f1_weapon_global"] = macro_f1
    result["precision_weapon_macro_global"] = macro_precision
    result["recall_weapon_macro_global"] = macro_recall
    result["f1_weapon_macro_global"] = macro_f1
    # Mirror balanced_acc_target naming: macro-averaged per-class recall.
    # Answers "for each weapon, what fraction of demonstrator-held frames
    # did the model match? Averaged across all 8 weapons equally."
    # Insensitive to the SG/RL frequency dominance that biases acc_weapon.
    result["balanced_acc_weapon"] = macro_recall
    result["f1_weapon_weighted_global"] = weighted_f1
    result["precision_weapon_micro_global"] = micro_precision
    result["recall_weapon_micro_global"] = micro_recall
    result["f1_weapon_micro_global"] = micro_f1
    result["acc_weapon_global"] = micro_recall
    if "acc_weapon" in result:
        result["acc_weapon_batch"] = result["acc_weapon"]
    result["acc_weapon"] = micro_recall


def _stable_target_metrics_from_counts(result: Dict[str, float]) -> None:
    classes = sorted(
        int(k[len("tp_target_idx_"):])
        for k in result
        if k.startswith("tp_target_idx_")
    )
    if not classes:
        return

    total = float(result.get("n_target_valid", 0.0))
    correct = float(result.get("correct_target", 0.0))
    if total > 0.0:
        result["acc_target"] = correct / total

    recalls: list[float] = []
    for idx in classes:
        tp = float(result.get(f"tp_target_idx_{idx}", 0.0))
        fp = float(result.get(f"fp_target_idx_{idx}", 0.0))
        fn = float(result.get(f"fn_target_idx_{idx}", 0.0))
        support = float(result.get(f"n_target_idx_{idx}", 0.0))
        pred_count = float(result.get(f"pred_target_idx_{idx}", 0.0))
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / support if support > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        if support > 0.0:
            recalls.append(recall)
        result[f"precision_target_idx_{idx}"] = precision
        result[f"recall_target_idx_{idx}"] = recall
        result[f"f1_target_idx_{idx}"] = f1
        result[f"pos_rate_target_idx_{idx}"] = support / total if total > 0.0 else 0.0
        result[f"pred_rate_target_idx_{idx}"] = pred_count / total if total > 0.0 else 0.0

    if recalls:
        result["balanced_acc_target"] = float(sum(recalls) / len(recalls))
    result["acc_target_idx0_baseline"] = (
        float(result.get("n_target_idx_0", 0.0)) / total if total > 0.0 else 0.0
    )
    _stable_binary_metrics_from_counts(result, prefix="target_nonzero")


def _apply_stable_epoch_metrics(result: Dict[str, float]) -> None:
    if "tp_attack" in result and "fp_attack" in result and "fn_attack" in result:
        _stable_binary_metrics_from_counts(result, prefix="attack")
    _stable_target_metrics_from_counts(result)
    _stable_weapon_metrics_from_counts(result)


def _apply_global_length_bucketing(
    ep_order: list[int],
    episodes: Sequence[PrecomputedEpisode],
) -> list[int]:
    """Deterministically bucket episodes by length across the full epoch order.

    Stable sort by descending ``n_samples`` so similarly-sized episodes are
    grouped together. Ties keep incoming order.
    """
    if len(ep_order) <= 1:
        return ep_order
    return sorted(ep_order, key=lambda idx: episodes[idx].n_samples, reverse=True)


@dataclass(slots=True)
class MidEpochState:
    """Snapshot of training loop state at an optimizer-step boundary.

    ``next_episode`` is the next packed chunk index.  ``active_hiddens`` stores
    lane hidden states as ``(lane_idx, 0, hidden)``; the field name stays stable
    so existing checkpoint plumbing can keep treating this as one opaque state.
    """
    next_episode: int            # next packed chunk index
    opt_steps: int               # optimizer steps completed this epoch
    active_hiddens: list[tuple[int, int, torch.Tensor | None]]
    total_rows: int
    total_loss: float
    ep_order: list[int] | None = None


@dataclass(slots=True)
class _LaneItem:
    ep_idx: int
    episode: PrecomputedEpisode
    lane: int
    lane_start: int


@dataclass(slots=True)
class _PackedSlice:
    item: _LaneItem
    src_start: int
    dst_start: int
    length: int
    reset: bool


@dataclass(slots=True)
class _PackedChunkPlan:
    slices: tuple[_PackedSlice, ...]
    length: int
    batch_size: int
    valid_rows: int
    active_lanes: int
    dst_indices: np.ndarray
    reset_indices: np.ndarray
    # Device-staged views, filled in once at plan-build time so per-batch
    # gather isn't paying a fresh H→D transfer for these tensors.
    dst_indices_d: torch.Tensor | None = None
    reset_indices_d: torch.Tensor | None = None


@dataclass(slots=True)
class Batch:
    """One trainer-ready batch handed to :func:`train_on_batches`.

    The producer owns where tensors come from (GPU-resident slice, prefetched
    pinned-host copy, etc.) and any per-batch state (hidden, masks, lane
    scaling). ``on_step`` is the producer's post-step hook: called with the
    metrics dict and ``stepped`` (True iff the trainer's accumulator just
    flushed). Producers use it to propagate ``_next_hidden`` and run
    optimizer-step-aligned bookkeeping (mid-epoch checkpoints, reporting).
    """
    obs: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    rows: int
    hidden: torch.Tensor | None = None
    masks: dict[str, torch.Tensor] | None = None
    loss_scale: float = 1.0
    compute_metrics: bool = True
    on_step: Any = None


class Source(Protocol):
    """Interface :func:`run_epoch` consumes.

    Two concrete implementations:
      - :class:`ResidentSource` (built by :func:`make_resident_source`):
        the whole corpus lives in device tensors; gather is ``index_select``.
      - :class:`StreamingSource` (built by :func:`make_streaming_source`):
        shards live on disk as mmaps; gather reads, pads, dequantizes, and
        transfers on demand.

    ``prefetch_depth`` is the in-flight batch queue cap; ``n_workers`` is
    the size of the parallel-gather thread pool. Resident leaves both at 0
    (gather is on-device ``index_select`` — nothing to parallelize across
    workers); streaming sets ``n_workers>=2`` so the host-bound shard read
    + token-pad work for batch N+k can overlap with the GPU step for
    batch N.
    """
    device: torch.device
    episodes: Sequence[Any]
    episode_offsets: np.ndarray
    prefetch_depth: int
    n_workers: int

    @property
    def n_total_rows(self) -> int: ...
    def gather(self, indices: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ...
    def attack_pos_neg_counts(self) -> tuple[int, int]: ...
    def release_device_tensors(self) -> None: ...
    def head(self, n_episodes: int) -> "Source": ...


@dataclass(slots=True)
class ResidentSource:
    """Device-resident concrete :class:`Source`.

    ``obs``/``actions`` are pre-concatenated, padded, dequantized device
    tensors. Built once at startup by :func:`make_resident_source`.
    """
    obs: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    episodes: list[PrecomputedEpisode]
    episode_offsets: np.ndarray
    device: torch.device
    prefetch_depth: int = 0
    n_workers: int = 0

    @property
    def n_total_rows(self) -> int:
        return int(self.episode_offsets[-1])

    def gather(self, indices: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {k: v.index_select(0, indices) for k, v in self.obs.items()},
            {k: v.index_select(0, indices) for k, v in self.actions.items()},
        )

    def attack_pos_neg_counts(self) -> tuple[int, int]:
        f = self.actions.get("attack")
        if f is None:
            return 0, 0
        pos = int((f > 0).sum().item())
        return pos, int(f.numel() - pos)

    def release_device_tensors(self) -> None:
        self.obs.clear()
        self.actions.clear()

    def head(self, n_episodes: int) -> "ResidentSource":
        """View of the first ``n_episodes`` episodes; shares device tensors."""
        return ResidentSource(
            obs=self.obs,
            actions=self.actions,
            episodes=self.episodes[:n_episodes],
            episode_offsets=self.episode_offsets[:n_episodes + 1],
            device=self.device,
            prefetch_depth=self.prefetch_depth,
            n_workers=self.n_workers,
        )


def make_resident_source(
    episodes: Sequence[PrecomputedEpisode],
    device: torch.device,
) -> ResidentSource:
    """Build a device-resident :class:`Source` from precomputed episodes.

    Concatenates per-key obs/action arrays onto ``device``, pads
    token-indexed obs fields to ``MAX_TOKEN_OBJECTS`` once globally,
    and pre-dequantizes legacy keys. Returns a :class:`ResidentSource`
    with ``episode_offsets`` computed alongside.
    """
    if not episodes:
        return ResidentSource(
            obs={}, actions={}, episodes=[],
            episode_offsets=np.zeros(1, dtype=np.int64),
            device=device,
        )
    from qnn.vocab import MAX_TOKEN_OBJECTS as _MAX_TOKEN_OBJECTS
    from qnn.bc.train import (
        _NATIVE_TOKEN_INDEXED_OBS_FIELDS as _TOK_FIELDS,
        _pad_entity_batch as _global_pad_entity_batch,
    )
    action_keys = list(episodes[0].actions.keys())

    def _concat_arrays(arr_list, dtype=None):
        np_arr = np.concatenate([np.asarray(a) for a in arr_list], axis=0)
        if dtype is not None and np_arr.dtype != dtype:
            np_arr = np_arr.astype(dtype, copy=False)
        return torch.from_numpy(np_arr).to(device)

    # Detect the global-buffer layout: every episode shares the same
    # ndarray .base for a given token-indexed key when they came from
    # the post-load global allocation. In that case we can skip the
    # per-episode concat (it would re-allocate the same data) and
    # build a single padded tensor from the global buffer + a global
    # token-indptr stitched from each sub-episode's ``entity_indptr``.
    first_obs = episodes[0].obs
    obs_keys = list(first_obs.keys())
    tok_keys = [k for k in obs_keys if k in _TOK_FIELDS]
    row_keys = [k for k in obs_keys if k not in _TOK_FIELDS]
    use_global_pad = (
        all(ep.entity_indptr is not None for ep in episodes)
        and bool(tok_keys)
    )

    gpu_obs: dict[str, torch.Tensor] = {}
    if use_global_pad:
        # Row-indexed obs and actions concatenate directly from the
        # sub-episode views.
        for k in row_keys:
            gpu_obs[k] = _concat_arrays([ep.obs[k] for ep in episodes])
        # Build a global token indptr by chaining each sub-episode's
        # rebased indptr. The token-indexed obs values are views into
        # a single underlying buffer per key (the _load_precomputed
        # ``global_obs_tok`` allocation), so we just iterate the
        # episodes in their stored order and accumulate.
        total_rows = sum(ep.n_samples for ep in episodes)
        global_indptr = np.empty(total_rows + 1, dtype=np.int64)
        global_indptr[0] = 0
        cursor_row = 0
        cursor_tok = 0
        tok_total = 0
        # First compute total token count to size the source buffer.
        for ep in episodes:
            ip = ep.entity_indptr
            tok_total += int(ip[-1])
        # Sanity: every token-indexed key's underlying buffer is the
        # same length per key (all sub-episodes view the same global
        # buffer). We can use one buffer per key for the pad.
        for ep in episodes:
            ip = ep.entity_indptr
            n = ep.n_samples
            ip_offset = cursor_tok
            global_indptr[cursor_row + 1:cursor_row + n + 1] = (
                ip[1:n + 1] + ip_offset
            )
            cursor_row += n
            cursor_tok += int(ip[-1])
        # Concatenate the unpadded token buffers (zero-copy if views
        # of a shared underlying ndarray; per-episode copy otherwise).
        unpadded_obs: dict[str, np.ndarray] = {}
        for k in tok_keys:
            unpadded_obs[k] = np.concatenate(
                [np.asarray(ep.obs[k]) for ep in episodes], axis=0,
            )
        # ONE global pad call per key.
        padded_tok = _global_pad_entity_batch(
            unpadded_obs, global_indptr, 0, total_rows, _MAX_TOKEN_OBJECTS,
        )
        for k in tok_keys:
            gpu_obs[k] = torch.from_numpy(padded_tok[k]).to(device)
    else:
        # Compatibility path for synthetic tests or any episode with
        # entity_indptr=None.
        padded_eps = [ep.materialize_padded_obs(_MAX_TOKEN_OBJECTS) for ep in episodes]
        for k in obs_keys:
            gpu_obs[k] = _concat_arrays([ep_obs[k] for ep_obs in padded_eps])
        del padded_eps

    gpu_actions = {
        k: _concat_arrays([ep.actions[k] for ep in episodes])
        for k in action_keys
    }

    # Pre-dequantize on-device: convert native widths to the legacy
    # float obs the policy / heads / target_labeler internals consume
    # (self_scalars, spatial_scalars, entity_scalars_raw, entity_ids,
    # plus the categorical id keys). Doing this ONCE at preload — vs.
    # every batch in the ObsEmbedding — eliminates per-batch dequant
    # latency and per-batch allocation churn.
    #
    # Trade-off: ~2× VRAM on the entity block (float32 vs native int)
    # in exchange for zero per-batch dequant work. With the smoke
    # corpus this is a few MB; on the full ~8M-frame corpus it's
    # ~6 GB additional VRAM, which still fits comfortably on the
    # APU's shared pool.
    if "health" in gpu_obs and "self_scalars" not in gpu_obs:
        from qnn.model.dequant import (
            SelfDequantizer, SpatialDequantizer, EntityDequantizer,
        )
        with torch.no_grad():
            gpu_obs = EntityDequantizer().to(device)(
                SpatialDequantizer().to(device)(
                    SelfDequantizer().to(device)(gpu_obs)
                )
            )

    # Precompute per-frame distance-to-nearest-positive WITHIN each
    # episode for the binary action streams that drive distance-
    # weighted BCE/CE. This is the flat-batch path's equivalent of
    # the Conv1d time-axis smoothing in distance_weighted_neg_weights:
    # since frame-shuffled SGD has no time axis, we amortize the
    # convolution to a one-time precompute per episode and store the
    # per-frame distance alongside the binary target. The training
    # step does a pointwise weight = 1 - exp(-d²/(2σ²)) at the sampled
    # frames (see qnn.bc.loss_shaping.flat_distance_weight).
    #
    # fire is a 0/1 byte; we read it directly.
    # jump-positive is derived from move (ud-axis == MOVE_CLASS_POS == 2);
    # we compute it on the fly per episode and store under a dedicated key
    # so policy.py can pick it up without touching MOVE_HEAD encoding.
    from qnn.bc.loss_shaping import per_frame_distance_to_pos

    if "attack" in episodes[0].actions:
        gpu_actions["attack_distance_to_pos"] = _concat_arrays([
            per_frame_distance_to_pos(np.asarray(ep.actions["attack"]).reshape(-1))
            for ep in episodes
        ])
    if "move" in episodes[0].actions:
        # move is (T, 3) uint8 with axes [fb, lr, ud]; ud=2 is the jump axis.
        # MOVE_CLASS_POS = 2 (neg/none/pos = 0/1/2). Build a 0/1 stream then
        # the same per-episode nearest-positive distance compute.
        jump_pos_arrays = []
        for ep in episodes:
            move = np.asarray(ep.actions["move"])
            ud = move[..., 2] if move.ndim >= 2 else move
            jump_pos = (ud == 2).astype(np.float32)
            jump_pos_arrays.append(per_frame_distance_to_pos(jump_pos))
        gpu_actions["jump_distance_to_pos"] = _concat_arrays(jump_pos_arrays)

    offsets = np.empty(len(episodes) + 1, dtype=np.int64)
    offsets[0] = 0
    for i, ep in enumerate(episodes):
        offsets[i + 1] = offsets[i] + ep.n_samples
    return ResidentSource(
        obs=gpu_obs,
        actions=gpu_actions,
        episodes=list(episodes),
        episode_offsets=offsets,
        device=device,
    )


class StreamingSource:
    """Disk-streaming concrete :class:`Source`.

    Holds open mmap views of shard files and reads only the rows each
    batch needs. Token-indexed obs fields are padded per-batch via the
    vectorized :func:`_pad_entity_batch` helper; legacy keys (``health``,
    ``vel``, ``self_items`` family) are dequantized on-device per batch.
    Per-frame distance arrays (used by distance-weighted BCE on fire/jump)
    are precomputed once at construction and kept on device — cheap
    (~30 MB each for an 8M-row corpus) and lets per-batch ``gather_actions``
    stay a single ``index_select``.

    Per-batch path (lane-packed batches → mostly-contiguous-per-shard
    indices):
      1. CPU: sort indices by (shard_idx, row_in_shard).
      2. CPU: per shard, vectorized fancy-index of row-indexed obs +
         vectorized token pad of token-indexed obs.
      3. GPU: ``to(device)`` + dequant chain.
    The shard-sort keeps mmap reads sequential per touched shard, so the
    OS page cache stays effective.
    """

    def __init__(
        self,
        ll: Any,  # qnn.bc.streaming_source.StreamingSource (avoid import cycle in annotation)
        device: torch.device,
        *,
        prefetch_depth: int = 4,
        n_workers: int = 1,
    ) -> None:
        self._ll = ll
        self.device = device
        self.episodes: list[Any] = list(ll.episodes)
        self.prefetch_depth = int(prefetch_depth)
        self.n_workers = int(n_workers)

        offsets = np.empty(len(self.episodes) + 1, dtype=np.int64)
        offsets[0] = 0
        for i, ep in enumerate(self.episodes):
            offsets[i + 1] = offsets[i] + ep.n_samples
        self.episode_offsets = offsets

        # Pre-extract shard_idx + row_start per episode for vectorized lookup
        # in _resolve_indices.
        self._ep_shard_idx = np.array([ep.shard_idx for ep in self.episodes], dtype=np.int64)
        self._ep_shard_row_start = np.array([ep.row_start for ep in self.episodes], dtype=np.int64)

        if not self.episodes:
            self._obs_keys: list[str] = []
            self._tok_obs_keys: list[str] = []
            self._row_obs_keys: list[str] = []
            self._action_keys: list[str] = []
            self._needs_dequant = False
        else:
            from qnn.bc.train import _NATIVE_TOKEN_INDEXED_OBS_FIELDS as _TOK_FIELDS
            view = self._open_shard(int(self.episodes[0].shard_idx))
            self._obs_keys = list(view.obs.keys())
            self._tok_obs_keys = [k for k in self._obs_keys if k in _TOK_FIELDS]
            self._row_obs_keys = [k for k in self._obs_keys if k not in _TOK_FIELDS]
            self._action_keys = list(view.actions.keys())
            self._needs_dequant = any(k in view.obs for k in ("health", "vel", "self_items"))
        self._dequant_chain: tuple = ()

        # Precompute per-frame distance arrays once at startup. These live
        # on device and avoid per-batch convolution work.
        self._attack_dist: torch.Tensor | None = None
        self._jump_dist: torch.Tensor | None = None
        if self.episodes:
            self._precompute_distances()

    def _open_shard(self, shard_idx: int):
        """``open_shard`` wrapper that lazily unpacks the packed ``move`` byte.

        The on-disk format stores ``move`` as a 1-D ``uint8`` array with the
        FB/LR/UD axes plus the fire bit packed together. Training needs the
        unpacked ``(T, 3)`` axis tensor and a separate ``fire`` byte; the
        resident path's loader does this at shard import. For streaming, we
        defer until first access — once per shard per thread, then cached
        in the (mutable) ``ShardView``.
        """
        view = self._ll.open_shard(shard_idx)
        if "move" in view.actions and "attack" not in view.actions:
            move_raw = view.actions["move"]
            if np.asarray(move_raw).ndim == 1:
                from qnn.bc.train import _unpack_move_axes, _unpack_attack_bit
                actions = dict(view.actions)
                actions["move"] = _unpack_move_axes(move_raw)
                actions["attack"] = _unpack_attack_bit(move_raw)
                view.actions = actions
        return view

    def _precompute_distances(self) -> None:
        from qnn.bc.loss_shaping import per_frame_distance_to_pos
        has_fire = "attack" in self._action_keys
        has_move = "move" in self._action_keys
        if not (has_fire or has_move):
            return
        fire_parts: list[np.ndarray] = []
        jump_parts: list[np.ndarray] = []
        any_fire = False
        any_move = False
        for ep in self.episodes:
            view = self._open_shard(int(ep.shard_idx))
            row_lo = int(ep.row_start)
            row_hi = int(ep.row_end)
            if "attack" in view.actions:
                any_fire = True
                fire = np.asarray(view.actions["attack"][row_lo:row_hi]).reshape(-1)
                fire_parts.append(per_frame_distance_to_pos(fire))
            if "move" in view.actions:
                any_move = True
                move = np.asarray(view.actions["move"][row_lo:row_hi])
                ud = move[..., 2] if move.ndim >= 2 else move
                jump_pos = (ud == 2).astype(np.float32)
                jump_parts.append(per_frame_distance_to_pos(jump_pos))
        if any_fire:
            self._attack_dist = torch.from_numpy(np.concatenate(fire_parts, axis=0)).to(self.device)
        if any_move:
            self._jump_dist = torch.from_numpy(np.concatenate(jump_parts, axis=0)).to(self.device)

    def _ensure_dequant(self) -> None:
        if not self._needs_dequant or self._dequant_chain:
            return
        from qnn.model.dequant import (
            SelfDequantizer, SpatialDequantizer, EntityDequantizer,
        )
        self._dequant_chain = (
            SelfDequantizer().to(self.device).eval(),
            SpatialDequantizer().to(self.device).eval(),
            EntityDequantizer().to(self.device).eval(),
        )

    @property
    def n_total_rows(self) -> int:
        return int(self.episode_offsets[-1])

    def attack_pos_neg_counts(self) -> tuple[int, int]:
        """One-pass scan of fire columns across shards. Returns ``(pos, neg)``.

        Equivalent to iterating ``ep.actions["attack"]`` in the resident path
        but reads directly from shard mmaps so streaming runs don't need to
        materialize the host-side episode arrays.
        """
        pos = 0
        neg = 0
        for ep in self.episodes:
            view = self._open_shard(int(ep.shard_idx))
            lo, hi = int(ep.row_start), int(ep.row_end)
            if "attack" in view.actions:
                arr = np.asarray(view.actions["attack"][lo:hi]).reshape(-1)
            elif "move" in view.actions:
                from qnn.bc.train import _unpack_attack_bit
                arr = _unpack_attack_bit(view.actions["move"][lo:hi]).reshape(-1)
            else:
                continue
            p = int((arr > 0).sum())
            pos += p
            neg += int(arr.shape[0]) - p
        return pos, neg

    def _resolve_indices(self, indices_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map global row indices to (shard, row_in_shard) and group by shard.

        Returns ``(shards_uniq, group_starts, rows_sorted, dst_order)``:
        ``rows_sorted`` is ``row_in_shard`` sorted by ``(shard_idx, row_in_shard)``
        for sequential mmap reads; ``dst_order`` is the argsort permutation
        back to original ordering.
        """
        ep_idx_per_row = np.searchsorted(self.episode_offsets[1:], indices_np, side="right")
        shards_per_row = self._ep_shard_idx[ep_idx_per_row]
        row_in_shard = (
            self._ep_shard_row_start[ep_idx_per_row]
            + (indices_np - self.episode_offsets[ep_idx_per_row])
        )
        sort_key = shards_per_row.astype(np.int64) * (np.int64(1) << 32) + row_in_shard
        order = np.argsort(sort_key, kind="stable")
        shards_sorted = shards_per_row[order]
        rows_sorted = row_in_shard[order]
        shards_uniq, group_starts = np.unique(shards_sorted, return_index=True)
        group_starts = np.append(group_starts, len(indices_np))
        return shards_uniq, group_starts, rows_sorted, order

    def gather(self, indices: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Gather obs + actions for ``indices`` in one pass over the layout.

        Resolves shard groups once (shared by obs and actions), then for each
        touched shard does one fancy-index read per key. Token-indexed obs
        are padded inline. Output keys for actions include the precomputed
        ``attack_distance_to_pos`` / ``jump_distance_to_pos`` slices when those
        signals were detected at construction.
        """
        indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        n_rows = len(indices_np)
        if n_rows == 0:
            return (
                {k: torch.empty(0, device=self.device) for k in self._obs_keys},
                {k: torch.empty(0, device=self.device) for k in self._action_keys},
            )

        from qnn.bc.train import (
            _ENTITY_TYPES_EMPTY_SENTINEL as _TOK_EMPTY,
        )
        from qnn.vocab import MAX_TOKEN_OBJECTS as _MAX_TOK

        shards_uniq, group_starts, rows_sorted, order = self._resolve_indices(indices_np)

        obs_out: dict[str, np.ndarray] = {}
        act_out: dict[str, np.ndarray] = {}
        for gi in range(len(shards_uniq)):
            shard_idx = int(shards_uniq[gi])
            s, e = int(group_starts[gi]), int(group_starts[gi + 1])
            shard_rows = rows_sorted[s:e]
            dst_positions = order[s:e]
            view = self._open_shard(shard_idx)
            for key in self._row_obs_keys:
                arr = view.obs[key]
                if key not in obs_out:
                    obs_out[key] = np.empty((n_rows, *arr.shape[1:]), dtype=arr.dtype)
                obs_out[key][dst_positions] = arr[shard_rows]
            if view.indptr is not None and self._tok_obs_keys:
                tok_layout = self._token_layout(view.indptr, shard_rows, _MAX_TOK)
                for key in self._tok_obs_keys:
                    arr = view.obs[key]
                    fill = _TOK_EMPTY if key == "entity_types" else 0
                    if key not in obs_out:
                        obs_out[key] = np.full((n_rows, _MAX_TOK, *arr.shape[1:]), fill, dtype=arr.dtype)
                    obs_out[key][dst_positions] = self._pad_token_block(arr, tok_layout, view.token_keep, fill)
            for key in self._action_keys:
                arr = view.actions[key]
                if key not in act_out:
                    act_out[key] = np.empty((n_rows, *arr.shape[1:]), dtype=arr.dtype)
                act_out[key][dst_positions] = arr[shard_rows]

        obs_t = {k: self._to_device(v) for k, v in obs_out.items()}
        act_t = {k: self._to_device(v) for k, v in act_out.items()}
        self._ensure_dequant()
        if self._dequant_chain:
            with torch.no_grad():
                for mod in self._dequant_chain:
                    obs_t = mod(obs_t)
        if self._attack_dist is not None:
            act_t["attack_distance_to_pos"] = self._attack_dist.index_select(0, indices)
        if self._jump_dist is not None:
            act_t["jump_distance_to_pos"] = self._jump_dist.index_select(0, indices)
        return obs_t, act_t

    def _to_device(self, v: np.ndarray) -> torch.Tensor:
        # uint16/uint32 don't have CPU index_copy_ in torch — upcast at the
        # device-transfer boundary (cheap, only on the per-batch slice).
        if v.dtype in (np.uint16, np.uint32):
            v = v.astype(np.int32, copy=False)
        return torch.from_numpy(v).to(self.device, non_blocking=True)

    @staticmethod
    def _token_layout(indptr: np.ndarray, shard_rows: np.ndarray, max_tok: int) -> tuple[np.ndarray, np.ndarray]:
        """Shared ``gather_idx`` + ``valid`` mask for token-pad of these shard rows."""
        rs = indptr[shard_rows].astype(np.int64, copy=False)
        re = indptr[shard_rows + 1].astype(np.int64, copy=False)
        counts = np.minimum(re - rs, max_tok).astype(np.int64, copy=False)
        indices = np.arange(max_tok, dtype=np.int64)
        valid = indices[None, :] < counts[:, None]
        gather_idx = np.where(valid, rs[:, None] + indices[None, :], 0)
        return gather_idx, valid

    @staticmethod
    def _pad_token_block(
        flat: np.ndarray,
        layout: tuple[np.ndarray, np.ndarray],
        token_keep: np.ndarray | None,
        fill: int,
    ) -> np.ndarray:
        gather_idx, valid = layout
        if flat.shape[0] == 0:
            return np.full((gather_idx.shape[0], gather_idx.shape[1], *flat.shape[1:]), fill, dtype=flat.dtype)
        if token_keep is not None:
            valid = valid & token_keep[gather_idx].astype(bool, copy=False)
        padded = flat[gather_idx]
        mask = valid.reshape(valid.shape + (1,) * (padded.ndim - 2)) if padded.ndim > 2 else valid
        return np.where(mask, padded, np.asarray(fill, dtype=padded.dtype))

    def release_device_tensors(self) -> None:
        """Free device tensors held by this streaming source.

        Mirrors :meth:`ResidentSource.release_device_tensors`. Called
        between training and final-val to keep the unified-memory pool
        from holding train-side state during a separate val pass.
        """
        self._attack_dist = None
        self._jump_dist = None
        self._dequant_chain = ()

    def head(self, n_episodes: int) -> "StreamingSource":
        """View of the first ``n_episodes`` episodes; shares shard mmaps,
        dequant chain, and (truncated views of) precomputed distance arrays.
        """
        view = StreamingSource.__new__(StreamingSource)
        view._ll = self._ll
        view.device = self.device
        view.episodes = list(self.episodes[:n_episodes])
        view.prefetch_depth = self.prefetch_depth
        view.n_workers = self.n_workers
        view.episode_offsets = self.episode_offsets[:n_episodes + 1].copy()
        view._ep_shard_idx = self._ep_shard_idx[:n_episodes]
        view._ep_shard_row_start = self._ep_shard_row_start[:n_episodes]
        view._obs_keys = self._obs_keys
        view._tok_obs_keys = self._tok_obs_keys
        view._row_obs_keys = self._row_obs_keys
        view._action_keys = self._action_keys
        view._needs_dequant = self._needs_dequant
        view._dequant_chain = self._dequant_chain
        n_rows_head = int(view.episode_offsets[-1])
        view._attack_dist = self._attack_dist[:n_rows_head] if self._attack_dist is not None else None
        view._jump_dist = self._jump_dist[:n_rows_head] if self._jump_dist is not None else None
        return view


def make_streaming_source(
    cache_dir: Any,
    device: torch.device,
    *,
    segment_mask: dict | None = None,
    token_mask: dict | None = None,
    prefetch_depth: int = 4,
    n_workers: int = 1,
) -> StreamingSource:
    """Build a :class:`StreamingSource` from a sharded BC cache directory."""
    from qnn.bc.streaming_source import StreamingSource as _LowLevel
    ll = _LowLevel.from_cache_dir(cache_dir, segment_mask=segment_mask, token_mask=token_mask)
    return StreamingSource(ll, device, prefetch_depth=prefetch_depth, n_workers=n_workers)


def parallel_prefetch_iter(
    prep_gen: Iterable[Callable[[], Any]],
    *,
    n_workers: int,
    depth: int,
) -> Iterable[Any]:
    """Run ``prep_gen`` callables in a thread pool; yield results in submission order.

    The batcher generators yield zero-arg closures (``Callable[[], Batch]``).
    This wraps the iteration so:
      - up to ``depth`` callables are in-flight at once,
      - ``n_workers`` threads share that in-flight set (so a slow gather
        doesn't block faster ones from making progress),
      - the consumer still sees Batches in original order.

    On a streaming source this is where the win lives: while the GPU runs
    forward+backward on batch N, ``n_workers`` threads are already running
    the mmap-read + token-pad + H→D for batches N+1 through N+depth. Most
    of that work releases the GIL (numpy + torch.from_numpy.to(device)),
    so threading actually parallelizes.

    Falls back to inline iteration when ``depth<=0`` or ``n_workers<=0``.
    """
    if depth <= 0 or n_workers <= 0:
        for prep in prep_gen:
            yield prep()
        return

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=n_workers)
    in_flight: deque = deque()

    def _submit_next() -> bool:
        try:
            prep = next(prep_iter)
        except StopIteration:
            return False
        in_flight.append(pool.submit(prep))
        return True

    prep_iter = iter(prep_gen)
    try:
        for _ in range(depth):
            if not _submit_next():
                break
        while in_flight:
            fut = in_flight.popleft()
            try:
                result = fut.result()
            except BaseException:
                # Cancel pending work and re-raise.
                for f in in_flight:
                    f.cancel()
                raise
            _submit_next()
            yield result
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def train_on_batches(
    model: "QNNPolicy",
    batches: Iterable[Batch],
    *,
    training: bool,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    max_grad_norm: float = 1.0,
    head_loss_weights: Mapping[str, float] | None = None,
    accum_target: float = 1.0,
    initial: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    """Train or eval over an iterable of :class:`Batch` objects.

    Data-source-agnostic: the caller owns where tensors live and how they
    are ordered/sequenced. This function handles forward+backward, grad
    clip, optimizer step (with optional accumulation), and metric
    aggregation. ``initial`` may seed ``total_rows``/``total_loss``/
    ``opt_steps`` for mid-epoch resume.
    """
    import time as _time

    model.model.train() if training else model.model.eval()
    if training:
        model.bc_zero_grad()
    _t_start = _time.monotonic()

    total_rows = int(initial["total_rows"]) if initial and "total_rows" in initial else 0
    opt_steps = int(initial["opt_steps"]) if initial and "opt_steps" in initial else 0
    total_loss_t: torch.Tensor | None = (
        torch.tensor(float(initial["total_loss"]), device=model.device)
        if initial and "total_loss" in initial else None
    )
    raw_sum_totals: Dict[str, torch.Tensor] = {}
    avg_totals: Dict[str, torch.Tensor] = {}
    metric_rows = 0
    accum_count = 0.0
    grad_norm_sum_t: torch.Tensor | None = None
    grad_norm_max_t: torch.Tensor | None = None
    grad_norm_n = 0

    for b in batches:
        if training:
            metrics = model.supervised_step(
                b.obs, b.actions, class_weights,
                lr=lr, accumulate_only=True,
                head_loss_weights=head_loss_weights,
                loss_scale=b.loss_scale,
                hidden=b.hidden, masks=b.masks,
                compute_metrics=b.compute_metrics,
            )
            accum_count += b.loss_scale
            stepped = accum_count >= accum_target
            if stepped:
                if max_grad_norm > 0:
                    _gn = torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(), max_grad_norm
                    ).detach()
                    if grad_norm_sum_t is None:
                        grad_norm_sum_t = torch.zeros_like(_gn)
                        grad_norm_max_t = torch.zeros_like(_gn)
                    grad_norm_sum_t.add_(_gn)
                    torch.maximum(grad_norm_max_t, _gn, out=grad_norm_max_t)
                    grad_norm_n += 1
                model.bc_step()
                model.bc_zero_grad()
                opt_steps += 1
                accum_count = 0.0
        else:
            metrics = model.evaluate_supervised(
                b.obs, b.actions,
                hidden=b.hidden, masks=b.masks,
                head_loss_weights=head_loss_weights,
            )
            stepped = False

        loss_t = metrics.get("loss")
        if isinstance(loss_t, torch.Tensor):
            if total_loss_t is None:
                total_loss_t = torch.zeros_like(loss_t)
            total_loss_t.add_(loss_t.detach() * b.rows)

        has_sampled = False
        for key, val in metrics.items():
            if key in ("loss", "_next_hidden") or not isinstance(val, torch.Tensor):
                continue
            if key.startswith(_RAW_SUM_METRIC_PREFIXES):
                if key not in raw_sum_totals:
                    raw_sum_totals[key] = torch.zeros_like(val)
                raw_sum_totals[key].add_(val.detach())
            elif key.startswith(_AVERAGED_METRIC_PREFIXES):
                has_sampled = True
                if key not in avg_totals:
                    avg_totals[key] = torch.zeros_like(val)
                avg_totals[key].add_(val.detach() * b.rows)
        if has_sampled:
            metric_rows += b.rows
        total_rows += b.rows

        if b.on_step is not None:
            b.on_step(
                metrics,
                stepped=stepped,
                opt_steps=opt_steps,
                total_rows=total_rows,
                total_loss=total_loss_t,
            )

    # Final partial-accumulation flush.
    if training and accum_count > 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
        model.bc_step()
        model.bc_zero_grad()
        opt_steps += 1

    elapsed = _time.monotonic() - _t_start
    denom = max(total_rows, 1)
    mdenom = max(metric_rows, 1)
    pending: Dict[str, torch.Tensor] = {}
    if total_loss_t is not None:
        pending["loss"] = total_loss_t
    pending.update(raw_sum_totals)
    pending.update(avg_totals)
    if grad_norm_sum_t is not None:
        pending["__grad_norm_sum"] = grad_norm_sum_t
        pending["__grad_norm_max"] = grad_norm_max_t
    synced = _flush_tensor_dict(pending)

    result: Dict[str, float] = {
        "loss": synced.get("loss", 0.0) / denom,
        "accuracy": 0.0,
        "n_rows": float(total_rows),
        "opt_steps": float(opt_steps),
    }
    for key in raw_sum_totals:
        result[key] = synced[key]
    for key in avg_totals:
        result[key] = synced[key] / mdenom
    if grad_norm_n > 0:
        result["grad_norm_mean"] = synced["__grad_norm_sum"] / grad_norm_n
        result["grad_norm_max"] = synced["__grad_norm_max"]
    rps_key = "effective_train_rows_per_sec" if training else "effective_val_rows_per_sec"
    result[rps_key] = total_rows / max(elapsed, 1e-9)
    _apply_stable_epoch_metrics(result)
    return result


def build_packed_plans(
    ordered_items: Sequence[tuple[int, PrecomputedEpisode]],
    chunk_size: int,
    n_lanes: int,
) -> list[_PackedChunkPlan]:
    """Bin-pack episodes into ``n_lanes`` lanes (shortest-lane heap),
    then slice each lane into chunks of ``chunk_size`` rows.
    """
    lane_lengths = [0] * n_lanes
    lane_items: list[list[_LaneItem]] = [[] for _ in range(n_lanes)]
    lane_heap = [(0, idx) for idx in range(n_lanes)]
    heapq.heapify(lane_heap)
    for ep_idx, episode in ordered_items:
        if episode.n_samples <= 0:
            continue
        lane_length, lane = heapq.heappop(lane_heap)
        item = _LaneItem(ep_idx=ep_idx, episode=episode, lane=lane, lane_start=lane_length)
        lane_items[lane].append(item)
        lane_lengths[lane] = lane_length + episode.n_samples
        heapq.heappush(lane_heap, (lane_lengths[lane], lane))

    total_length = max(lane_lengths, default=0)
    n_chunks = (total_length + chunk_size - 1) // chunk_size
    chunk_slices: list[list[_PackedSlice]] = [[] for _ in range(n_chunks)]
    chunk_rows = [0] * n_chunks
    for lane in range(n_lanes):
        for item in lane_items[lane]:
            item_start = item.lane_start
            item_end = item_start + item.episode.n_samples
            first_chunk = item_start // chunk_size
            last_chunk = (item_end - 1) // chunk_size
            for chunk_idx in range(first_chunk, last_chunk + 1):
                chunk_start = chunk_idx * chunk_size
                start = max(chunk_start, item_start)
                end = min(chunk_start + chunk_size, item_end)
                length = end - start
                chunk_rows[chunk_idx] += length
                chunk_slices[chunk_idx].append(_PackedSlice(
                    item=item,
                    src_start=start - item_start,
                    dst_start=start - chunk_start,
                    length=length,
                    reset=start == item_start,
                ))

    plans: list[_PackedChunkPlan] = []
    for chunk_idx, valid_rows in enumerate(chunk_rows):
        if not valid_rows:
            continue
        slices = tuple(chunk_slices[chunk_idx])
        dst_indices = np.empty((valid_rows,), dtype=np.int64)
        reset_indices: list[int] = []
        cursor = 0
        for sl in slices:
            base = sl.dst_start * n_lanes + sl.item.lane
            end = cursor + sl.length
            dst_indices[cursor:end] = base + (
                np.arange(sl.length, dtype=np.int64) * n_lanes
            )
            cursor = end
            if sl.reset:
                reset_indices.append(base)
        plans.append(_PackedChunkPlan(
            slices=slices,
            length=chunk_size,
            batch_size=n_lanes,
            valid_rows=valid_rows,
            active_lanes=len({sl.item.lane for sl in slices}),
            dst_indices=dst_indices,
            reset_indices=np.asarray(reset_indices, dtype=np.int64),
        ))
    return plans


def frame_shuffled_batches(
    source: Source,
    *,
    batch_size: int,
    training: bool,
    rng: np.random.Generator | None,
) -> Iterable[Callable[[], Batch]]:
    """Yield zero-arg closures that produce shuffled-frame batches.

    Yielding closures (not Batches) lets :func:`parallel_prefetch_iter`
    schedule them across multiple worker threads so gather work overlaps
    with the GPU step.
    """
    n_frames = source.n_total_rows
    if training and rng is not None:
        indices = torch.from_numpy(rng.permutation(n_frames)).to(source.device)
    else:
        indices = torch.arange(n_frames, device=source.device)
    bs = max(1, int(batch_size))

    def _prep(idx: torch.Tensor) -> Batch:
        obs, actions = source.gather(idx)
        return Batch(obs=obs, actions=actions, rows=int(idx.shape[0]))

    for start in range(0, n_frames, bs):
        idx = indices[start:start + bs]
        yield lambda idx=idx: _prep(idx)


def lane_packed_batches(
    model: "QNNPolicy",
    source: Source,
    *,
    chunk_size: int,
    batch_size: int,
    training: bool,
    rng: np.random.Generator | None,
    save_state_callback: Any = None,
    snapshot_interval: int = 0,
    step_callback: Any = None,
    report_every: int = 0,
    report_interval_seconds: float = 0.0,
    resume_state: "MidEpochState | None" = None,
) -> tuple[Iterable[Batch], "Mapping[str, Any] | None", float, int]:
    """Yield lane-packed sequence batches with per-batch hidden propagation.

    ``batch_size`` here means the number of **parallel lanes** (independent
    sequences) per gradient step — the recurrent analog of the per-step
    sample count in non-recurrent training. Each lane unrolls ``chunk_size``
    timesteps before backward.

    Returns ``(iterable, initial, accum_target, next_plan)``: the trainer
    consumes the iterable and uses ``initial`` to seed its accumulators on
    mid-epoch resume; ``accum_target`` is the gradient-accumulation target
    (active_lanes summed across batches); ``next_plan`` is unused externally
    and returned for diagnostics.
    """
    import time as _time

    device = source.device
    episodes = source.episodes
    n_lanes = max(1, int(batch_size))
    chunk_size = max(1, int(chunk_size))

    if resume_state is not None and resume_state.ep_order is not None:
        ep_order = [int(i) for i in resume_state.ep_order]
    else:
        ep_order = sorted(range(len(episodes)), key=lambda idx: episodes[idx].sort_key)
        if rng is not None:
            ep_order = [ep_order[int(i)] for i in rng.permutation(len(ep_order))]
        ep_order = _apply_global_length_bucketing(ep_order, episodes)

    ordered_items = [(idx, episodes[idx]) for idx in ep_order]
    plans = build_packed_plans(ordered_items, chunk_size, n_lanes)
    for p in plans:
        p.dst_indices_d = torch.from_numpy(p.dst_indices).to(device)
        if p.reset_indices.size:
            p.reset_indices_d = torch.from_numpy(p.reset_indices).to(device)

    next_plan = 0
    initial: Mapping[str, Any] | None = None
    if resume_state is not None:
        next_plan = int(resume_state.next_episode)
        initial = {
            "opt_steps": resume_state.opt_steps,
            "total_rows": resume_state.total_rows,
            "total_loss": resume_state.total_loss,
        }

    lane_hidden = (
        torch.zeros((n_lanes, model.gru_hidden), dtype=torch.float32, device=device)
        if model.use_gru else None
    )
    if resume_state is not None and lane_hidden is not None:
        for lane_idx, _cs, ch in resume_state.active_hiddens:
            if 0 <= lane_idx < n_lanes and ch is not None:
                lane_hidden[lane_idx].copy_(ch.to(device))

    offsets = source.episode_offsets
    _last_save_time = _time.monotonic()
    _last_report_time = _time.monotonic() - max(float(report_interval_seconds), 0.0)
    _report: Dict[str, Any] = {
        "rows": 0, "metric_rows": 0, "loss_t": None, "avg": {}, "raw": {},
    }

    def _gather_plan(plan: _PackedChunkPlan):
        flat_src = np.empty((plan.valid_rows,), dtype=np.int64)
        cursor = 0
        for sl in plan.slices:
            base = int(offsets[sl.item.ep_idx]) + sl.src_start
            flat_src[cursor:cursor + sl.length] = np.arange(base, base + sl.length, dtype=np.int64)
            cursor += sl.length
        src_idx = torch.from_numpy(flat_src).to(device)
        dst_idx = plan.dst_indices_d  # pre-staged on device at plan-build time
        T = plan.length
        B = plan.batch_size
        TB = T * B

        gathered_obs, gathered_acts = source.gather(src_idx)
        obs_batch: dict[str, torch.Tensor] = {}
        for k, v in gathered_obs.items():
            buf = torch.zeros((TB, *v.shape[1:]), dtype=v.dtype, device=device)
            buf.index_copy_(0, dst_idx, v)
            obs_batch[k] = buf.reshape(T, B, *v.shape[1:])
        act_batch: dict[str, torch.Tensor] = {}
        for k, v in gathered_acts.items():
            buf = torch.zeros((TB, *v.shape[1:]), dtype=v.dtype, device=device)
            buf.index_copy_(0, dst_idx, v)
            act_batch[k] = buf.reshape(T, B, *v.shape[1:])

        valid_flat = torch.zeros(TB, dtype=torch.bool, device=device)
        valid_flat.index_fill_(0, dst_idx, True)
        reset_flat = torch.zeros(TB, dtype=torch.bool, device=device)
        if plan.reset_indices_d is not None:
            reset_flat.index_fill_(0, plan.reset_indices_d, True)
        return obs_batch, act_batch, {
            "valid_mask": valid_flat.reshape(T, B),
            "reset_mask": reset_flat.reshape(T, B),
        }

    def _make_batch(plan: _PackedChunkPlan, plan_idx: int) -> Batch:
        obs_b, act_b, masks = _gather_plan(plan)
        if training:
            sample = (report_interval_seconds <= 0) or (
                _time.monotonic() - _last_report_time >= report_interval_seconds
            )
        else:
            sample = True

        def _on_step(metrics, *, stepped, opt_steps, total_rows, total_loss):
            nonlocal _last_save_time, _last_report_time
            nh = metrics.pop("_next_hidden", None)
            if nh is not None and lane_hidden is not None:
                lane_hidden.copy_(nh.detach())
            if not stepped:
                return
            if save_state_callback and snapshot_interval > 0:
                now = _time.monotonic()
                if now - _last_save_time >= snapshot_interval:
                    _last_save_time = now
                    active_hiddens = [
                        (lane_idx, 0, lane_hidden[lane_idx].clone())
                        for lane_idx in range(n_lanes)
                        if lane_hidden is not None
                    ]
                    total_loss_val = float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else 0.0
                    save_state_callback(MidEpochState(
                        next_episode=plan_idx + 1,
                        opt_steps=opt_steps,
                        active_hiddens=active_hiddens,
                        total_rows=total_rows,
                        total_loss=total_loss_val,
                        ep_order=ep_order,
                    ))
            if step_callback and report_every > 0:
                _report["rows"] += plan.valid_rows
                loss_t = metrics["loss"].detach()
                if _report["loss_t"] is None:
                    _report["loss_t"] = torch.zeros_like(loss_t)
                _report["loss_t"].add_(loss_t * plan.valid_rows)
                has_sampled = any(k.startswith(_AVERAGED_METRIC_PREFIXES) for k in metrics)
                if has_sampled:
                    _report["metric_rows"] += plan.valid_rows
                    for key, val in metrics.items():
                        if not isinstance(val, torch.Tensor):
                            continue
                        if key.startswith(_AVERAGED_METRIC_PREFIXES):
                            if key not in _report["avg"]:
                                _report["avg"][key] = torch.zeros_like(val)
                            _report["avg"][key].add_(val.detach() * plan.valid_rows)
                        elif key.startswith(_RAW_SUM_METRIC_PREFIXES):
                            if key not in _report["raw"]:
                                _report["raw"][key] = val.detach().clone()
                            else:
                                _report["raw"][key].add_(val.detach())
                step_ready = opt_steps % report_every == 0
                time_ready = (report_interval_seconds <= 0) or (
                    _time.monotonic() - _last_report_time >= report_interval_seconds
                )
                if step_ready and time_ready:
                    _last_report_time = _time.monotonic()
                    rd = max(_report["rows"], 1)
                    md = max(_report["metric_rows"], 1)
                    pending: Dict[str, torch.Tensor] = {"loss": _report["loss_t"]}
                    pending.update(_report["avg"])
                    pending.update(_report["raw"])
                    synced = _flush_tensor_dict(pending)
                    step_metrics: Dict[str, Any] = {
                        "n_rows": float(_report["rows"]),
                        "opt_step": opt_steps,
                        "loss": synced["loss"] / rd,
                    }
                    for key in _report["avg"]:
                        step_metrics[key] = synced[key] / md
                    for key in _report["raw"]:
                        step_metrics[key] = synced[key]
                    step_callback(step_metrics)
                    _report["rows"] = 0
                    _report["metric_rows"] = 0
                    _report["loss_t"] = None
                    _report["avg"].clear()
                    _report["raw"].clear()

        return Batch(
            obs=obs_b,
            actions=act_b,
            rows=plan.valid_rows,
            hidden=lane_hidden,
            masks=masks,
            loss_scale=float(plan.active_lanes),
            compute_metrics=sample,
            on_step=_on_step,
        )

    def _iter():
        for plan_idx in range(next_plan, len(plans)):
            yield lambda pi=plan_idx: _make_batch(plans[pi], pi)

    # accum_target = n_lanes (full lane sweep flushes the optimizer)
    return _iter(), initial, float(n_lanes), next_plan


def run_epoch(
    model: "QNNPolicy",
    source: Source,
    *,
    batch_size: int,
    sequence_length: int = 0,
    tbptt_limit: int = 0,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
    max_grad_norm: float = 1.0,
    head_loss_weights: Mapping[str, float] | None = None,
    save_state_callback: Any = None,
    snapshot_interval: int = 0,
    step_callback: Any = None,
    report_every: int = 0,
    report_interval_seconds: float = 0.0,
    resume_state: "MidEpochState | None" = None,
) -> Dict[str, float]:
    """Single supervised-epoch entry point.

    Picks lane-packed sequence batching when the model is recurrent
    (``model.use_gru``); frame-shuffled otherwise. Feeds the result to
    :func:`train_on_batches`. Source-agnostic — pass any :class:`Source`.
    """
    if source.n_total_rows == 0:
        return {"loss": 0.0, "accuracy": 0.0, "n_rows": 0.0}

    training = class_weights is not None and lr is not None
    accum_target: float = 1.0
    initial: Mapping[str, Any] | None = None

    if model.use_gru:
        if sequence_length <= 0:
            chunk_size = max(int(tbptt_limit), 1) if tbptt_limit > 0 else max(
                (ep.n_samples for ep in source.episodes), default=64,
            )
        else:
            chunk_size = max(int(sequence_length), 1)
        batches, initial, accum_target, _ = lane_packed_batches(
            model, source,
            chunk_size=chunk_size,
            batch_size=max(1, int(batch_size)),
            training=training,
            rng=rng,
            save_state_callback=save_state_callback,
            snapshot_interval=snapshot_interval,
            step_callback=step_callback,
            report_every=report_every,
            report_interval_seconds=report_interval_seconds,
            resume_state=resume_state,
        )
    else:
        batches = frame_shuffled_batches(
            source, batch_size=batch_size, training=training, rng=rng,
        )

    # Streaming sources hide disk-bound gather work behind the GPU step
    # by running multiple gather workers in parallel. Resident has
    # prefetch_depth=0 and runs the closures inline.
    actual_batches = parallel_prefetch_iter(
        batches,
        n_workers=int(getattr(source, "n_workers", 0)),
        depth=int(getattr(source, "prefetch_depth", 0)),
    )

    return train_on_batches(
        model, actual_batches,
        training=training,
        class_weights=class_weights, lr=lr,
        max_grad_norm=max_grad_norm,
        head_loss_weights=head_loss_weights,
        accum_target=accum_target,
        initial=initial,
    )
