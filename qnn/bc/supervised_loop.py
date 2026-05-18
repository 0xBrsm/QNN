"""Chunked supervised loop for BC.

The inner machinery behind :func:`qnn.bc.loop.run_epoch`: data carriers
(``PrecomputedEpisode``, ``MidEpochState``), the kernel-page-drop helpers
that keep mmap'd training data from blowing up the page cache, and the
batched, GRU-aware ``_run_batched`` driver that walks every episode.
``run_epoch`` itself stays in :mod:`qnn.bc.loop` as a thin orchestration
shim that picks ``chunk_size`` and dispatches.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict

import ctypes
import os

import numpy as np
import torch

from qnn.model.policy import QNNPolicy

# --- madvise page-drop for mmap'd training data ---
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.madvise.restype = ctypes.c_int
_libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
_libc.posix_fadvise.restype = ctypes.c_int
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_MADV_DONTNEED = 4
_POSIX_FADV_DONTNEED = 4


def _dontneed_pages(arr: np.ndarray) -> None:
    """Tell the kernel to immediately drop pages backing this array.

    Called after mmap'd training data has been copied into a pinned
    buffer, so the file-backed pages are no longer needed.  Without
    this, the page cache grows to the full corpus size (~71 GB) and
    starves the WSL2 host.
    """
    nbytes = arr.nbytes
    if nbytes == 0:
        return
    addr = arr.ctypes.data
    aligned = addr & ~(_PAGE_SIZE - 1)
    _libc.madvise(aligned, nbytes + (addr - aligned), _MADV_DONTNEED)


def _root_memmap(arr: np.ndarray) -> np.memmap | None:
    current = arr
    root: np.memmap | None = current if isinstance(current, np.memmap) else None
    while True:
        base = getattr(current, "base", None)
        if isinstance(base, np.memmap):
            root = base
            current = base
            continue
        break
    return root


def _drop_file_cache(
    arr: np.ndarray,
    fd_cache: dict[str, int],
) -> None:
    """Ask the kernel to evict the file-backed cache range for this memmap slice.

    On WSL2 + Docker Desktop, MADV_DONTNEED on the mapped pages alone can still
    leave the host-visible file cache ballooned near the corpus size.  Pair it
    with POSIX_FADV_DONTNEED on the backing file range so the kernel can drop
    those cache pages more directly.
    """
    filename = getattr(arr, "filename", None)
    if not filename:
        return
    root = _root_memmap(arr)
    if root is None:
        return
    root_addr = root.ctypes.data
    byte_delta = arr.ctypes.data - root_addr
    if byte_delta < 0:
        return
    file_offset = int(getattr(root, "offset", 0)) + int(byte_delta)
    length = int(arr.nbytes)
    if length <= 0:
        return
    fd = fd_cache.get(filename)
    if fd is None:
        try:
            fd = os.open(filename, os.O_RDONLY)
        except OSError:
            return
        fd_cache[filename] = fd
    _libc.posix_fadvise(fd, file_offset, length, _POSIX_FADV_DONTNEED)

_RAW_SUM_METRIC_PREFIXES = (
    "n_", "correct_", "l1_sum_", "tp_", "fp_", "fn_", "target_pos_", "pred_pos_", "pred_target_",
)
_AVERAGED_METRIC_PREFIXES = (
    "acc_", "mae_", "cos_sim_", "f1_", "precision_", "recall_",
    "pos_rate_", "pred_rate_", "confidence_", "balanced_acc_",
)


@dataclass(slots=True)
class PrecomputedEpisode:
    """One episode's observations and actions as contiguous arrays."""
    obs: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    n_samples: int
    sort_key: tuple[int, int, int] = (0, 0, 0)


# Batch madvise/fadvise per (cursor, key): accumulate consumed rows and
# fire one syscall per THRESHOLD bytes instead of one per chunk.
_ADVISE_BATCH_BYTES = 32 * 1024 * 1024


@dataclass(slots=True)
class _EpisodeCursor:
    episode: PrecomputedEpisode
    start: int = 0
    hidden: torch.Tensor | None = None
    advised_obs: dict[str, int] = field(default_factory=dict)
    advised_act: dict[str, int] = field(default_factory=dict)


def _maybe_advise_range(
    arr: np.ndarray,
    tracker: dict[str, int],
    key: str,
    consumed_end: int,
    fd_cache: dict[str, int],
    *,
    force: bool = False,
) -> None:
    prev = tracker.get(key, 0)
    if consumed_end <= prev:
        return
    pending = arr[prev:consumed_end]
    if not force and pending.nbytes < _ADVISE_BATCH_BYTES:
        return
    # Only issue page-drop hints for true file-backed memmaps. Dynamic labels
    # are ordinary heap ndarrays; MADV_DONTNEED
    # on anonymous pages can zero their contents and silently corrupt the
    # in-memory supervision signal across epochs.
    if _root_memmap(arr) is None:
        tracker[key] = consumed_end
        return
    _dontneed_pages(pending)
    _drop_file_cache(pending, fd_cache)
    tracker[key] = consumed_end


def _flush_tensor_dict(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Single GPU→CPU sync for a dict of 0-d tensors via stack + tolist."""
    if not tensors:
        return {}
    keys = list(tensors.keys())
    vals = torch.stack([tensors[k] for k in keys]).tolist()
    return dict(zip(keys, vals))


def _flush_cursor_advise(cursor: _EpisodeCursor, fd_cache: dict[str, int]) -> None:
    for key, arr in cursor.episode.obs.items():
        _maybe_advise_range(arr, cursor.advised_obs, key, arr.shape[0], fd_cache, force=True)
    for head, arr in cursor.episode.actions.items():
        _maybe_advise_range(arr, cursor.advised_act, head, arr.shape[0], fd_cache, force=True)


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
        int(k[len("tp_target_slot_"):])
        for k in result
        if k.startswith("tp_target_slot_")
    )
    if not classes:
        return

    total = float(result.get("n_target_valid", 0.0))
    correct = float(result.get("correct_target", 0.0))
    if total > 0.0:
        result["acc_target"] = correct / total

    recalls: list[float] = []
    for slot in classes:
        tp = float(result.get(f"tp_target_slot_{slot}", 0.0))
        fp = float(result.get(f"fp_target_slot_{slot}", 0.0))
        fn = float(result.get(f"fn_target_slot_{slot}", 0.0))
        support = float(result.get(f"n_target_slot_{slot}", 0.0))
        pred_count = float(result.get(f"pred_target_slot_{slot}", 0.0))
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / support if support > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        if support > 0.0:
            recalls.append(recall)
        result[f"precision_target_slot_{slot}"] = precision
        result[f"recall_target_slot_{slot}"] = recall
        result[f"f1_target_slot_{slot}"] = f1
        result[f"pos_rate_target_slot_{slot}"] = support / total if total > 0.0 else 0.0
        result[f"pred_rate_target_slot_{slot}"] = pred_count / total if total > 0.0 else 0.0

    if recalls:
        result["balanced_acc_target"] = float(sum(recalls) / len(recalls))
    result["acc_target_slot0_baseline"] = (
        float(result.get("n_target_slot_0", 0.0)) / total if total > 0.0 else 0.0
    )
    _stable_binary_metrics_from_counts(result, prefix="target_nonzero")


def _apply_stable_epoch_metrics(result: Dict[str, float]) -> None:
    if "tp_fire" in result and "fp_fire" in result and "fn_fire" in result:
        _stable_binary_metrics_from_counts(result, prefix="fire")
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
    cursor: _EpisodeCursor
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


@dataclass(slots=True)
class _PreparedChunkBatch:
    plan: _PackedChunkPlan
    obs: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


def _run_batched(
    model: QNNPolicy,
    episodes: Sequence[PrecomputedEpisode],
    batch_size: int,
    chunk_size: int,
    *,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
    max_grad_norm: float = 1.0,
    head_loss_weights: Mapping[str, float] | None = None,
    step_callback: Any | None = None,
    report_every: int = 0,
    report_interval_seconds: float = 0.0,
    pin_memory: bool = True,
    prefetch: int,
    microbatch_size: int = 0,
    save_state_callback: Any | None = None,
    snapshot_interval: int = 0,
    resume_state: MidEpochState | None = None,
) -> Dict[str, float]:
    _empty: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "n_rows": 0.0}
    if not episodes:
        return _empty

    training = class_weights is not None and lr is not None
    if training:
        model.model.train()
    else:
        model.model.eval()

    accum_target = max(1, int(batch_size))
    microbatch_target = min(accum_target, int(microbatch_size)) if microbatch_size > 0 else accum_target
    device_type = model.device.type if isinstance(model.device, torch.device) else str(model.device)
    use_pinned_host = device_type != "cpu" and bool(pin_memory)
    prefetch_depth = max(0, int(prefetch))
    use_prefetch = prefetch_depth > 0
    fd_cache: dict[str, int] = {}

    if resume_state is not None and resume_state.ep_order is not None:
        ep_order = [int(i) for i in resume_state.ep_order]
    else:
        ep_order = sorted(range(len(episodes)), key=lambda idx: episodes[idx].sort_key)
        if rng is not None:
            ep_order = [ep_order[int(i)] for i in rng.permutation(len(ep_order))]
        ep_order = _apply_global_length_bucketing(ep_order, episodes)

    total_rows = 0
    total_metric_rows = 0  # rows from sample steps only — denom for MAE/acc
    # Tensor-resident running sums; keep on GPU until epoch end / report boundary.
    total_loss_t: torch.Tensor | None = None
    total_accuracy_t: torch.Tensor | None = None
    raw_metric_totals_t: Dict[str, torch.Tensor] = {}
    averaged_metric_totals_t: Dict[str, torch.Tensor] = {}
    accum_count = 0.0

    opt_steps = 0
    grad_norm_sum_t: torch.Tensor | None = None
    grad_norm_max_t: torch.Tensor | None = None
    grad_norm_n = 0
    import time as _time
    _last_save_time = _time.monotonic()
    # Allow the first training batch in each epoch to compute/report metrics
    # immediately so short ablation runs do not end up with empty train stats.
    _last_report_time = _time.monotonic() - max(float(report_interval_seconds), 0.0)
    _report_rows = 0
    _report_metric_rows = 0
    _report_loss_t: torch.Tensor | None = None
    _report_avg_totals_t: Dict[str, torch.Tensor] = {}
    _report_raw_totals_t: Dict[str, torch.Tensor] = {}

    if training:
        model.bc_zero_grad()

    ordered_cursors = [_EpisodeCursor(episode=episodes[idx]) for idx in ep_order]
    action_names = list(episodes[0].actions.keys())
    lane_hidden = (
        torch.zeros((microbatch_target, model.gru_hidden), dtype=torch.float32, device=model.device)
        if model.use_gru
        else None
    )
    next_plan = 0

    if resume_state is not None:
        next_plan = resume_state.next_episode
        opt_steps = resume_state.opt_steps
        total_rows = resume_state.total_rows
        total_loss_t = torch.tensor(float(resume_state.total_loss), device=model.device)
        if lane_hidden is not None:
            for lane_idx, _cursor_start, cursor_hidden in resume_state.active_hiddens:
                if 0 <= lane_idx < microbatch_target and cursor_hidden is not None:
                    lane_hidden[lane_idx].copy_(cursor_hidden.to(model.device))

    def _build_packed_plans() -> list[_PackedChunkPlan]:
        lane_lengths = [0] * microbatch_target
        lane_items: list[list[_LaneItem]] = [[] for _ in range(microbatch_target)]
        for cursor in ordered_cursors:
            if cursor.episode.n_samples <= 0:
                continue
            lane = min(range(microbatch_target), key=lambda idx: (lane_lengths[idx], idx))
            item = _LaneItem(cursor=cursor, lane=lane, lane_start=lane_lengths[lane])
            lane_items[lane].append(item)
            lane_lengths[lane] += cursor.episode.n_samples

        total_length = max(lane_lengths, default=0)
        n_chunks = (total_length + chunk_size - 1) // chunk_size
        chunk_slices: list[list[_PackedSlice]] = [[] for _ in range(n_chunks)]
        chunk_rows = [0] * n_chunks
        for lane in range(microbatch_target):
            for item in lane_items[lane]:
                item_start = item.lane_start
                item_end = item_start + item.cursor.episode.n_samples
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
            if valid_rows:
                plans.append(_PackedChunkPlan(
                    slices=tuple(chunk_slices[chunk_idx]),
                    length=chunk_size,
                    batch_size=microbatch_target,
                    valid_rows=valid_rows,
                    active_lanes=len({sl.item.lane for sl in chunk_slices[chunk_idx]}),
                ))
        return plans

    packed_plans = _build_packed_plans()
    if next_plan >= len(packed_plans):
        return _empty

    # Pinned slots: one per in-flight batch plus one being consumed.
    # prefetch_depth batches can be staged at once.
    num_prefetch_slots = max(2, prefetch_depth + 1) if use_prefetch else 2
    obs_buffer_slots = []
    for _ in range(num_prefetch_slots):
        slot = {
            key: torch.empty(
                (chunk_size, microbatch_target, *episodes[0].obs[key].shape[1:]),
                dtype=torch.from_numpy(np.empty((), dtype=episodes[0].obs[key].dtype)).dtype,
                pin_memory=use_pinned_host,
            )
            for key in episodes[0].obs
        }
        obs_buffer_slots.append(slot)
    action_buffer_slots = []
    for _ in range(num_prefetch_slots):
        slot = {
            head: torch.empty(
                (chunk_size, microbatch_target, *episodes[0].actions[head].shape[1:]),
                dtype=torch.from_numpy(np.empty((), dtype=episodes[0].actions[head].dtype)).dtype,
                pin_memory=use_pinned_host,
            )
            for head in action_names
        }
        action_buffer_slots.append(slot)
    mask_buffer_slots = [
        {
            "valid_mask": torch.empty((chunk_size, microbatch_target), dtype=torch.bool, pin_memory=use_pinned_host),
            "reset_mask": torch.empty((chunk_size, microbatch_target), dtype=torch.bool, pin_memory=use_pinned_host),
        }
        for _ in range(num_prefetch_slots)
    ]

    def _prepare_prefetched_batch(slot_idx: int, plan: _PackedChunkPlan) -> _PreparedChunkBatch:
        for slot in obs_buffer_slots[slot_idx].values():
            slot.zero_()
        for slot in action_buffer_slots[slot_idx].values():
            slot.zero_()
        masks = mask_buffer_slots[slot_idx]
        masks["valid_mask"].zero_()
        masks["reset_mask"].zero_()

        obs_batch: dict[str, torch.Tensor] = {}
        for key in episodes[0].obs:
            dst = obs_buffer_slots[slot_idx][key][:plan.length, :plan.batch_size]
            for sl in plan.slices:
                arr = sl.item.cursor.episode.obs[key]
                chunk = arr[sl.src_start:sl.src_start + sl.length]
                dst[sl.dst_start:sl.dst_start + sl.length, sl.item.lane, ...].copy_(torch.from_numpy(np.asarray(chunk)))
                _maybe_advise_range(arr, sl.item.cursor.advised_obs, key, sl.src_start + sl.length, fd_cache)
            obs_batch[key] = dst
        act_batch: dict[str, torch.Tensor] = {}
        for head in action_names:
            dst = action_buffer_slots[slot_idx][head][:plan.length, :plan.batch_size]
            for sl in plan.slices:
                arr = sl.item.cursor.episode.actions[head]
                chunk = arr[sl.src_start:sl.src_start + sl.length]
                dst[sl.dst_start:sl.dst_start + sl.length, sl.item.lane, ...].copy_(torch.from_numpy(np.asarray(chunk)))
                _maybe_advise_range(arr, sl.item.cursor.advised_act, head, sl.src_start + sl.length, fd_cache)
            act_batch[head] = dst
        for sl in plan.slices:
            masks["valid_mask"][sl.dst_start:sl.dst_start + sl.length, sl.item.lane] = True
            if sl.reset:
                masks["reset_mask"][sl.dst_start, sl.item.lane] = True
        return _PreparedChunkBatch(
            plan=plan,
            obs=obs_batch,
            actions=act_batch,
            masks={key: value[:plan.length, :plan.batch_size] for key, value in masks.items()},
        )

    def _accumulate_sum(dct: Dict[str, torch.Tensor], key: str, val: torch.Tensor) -> None:
        v = val.detach()
        if key in dct:
            dct[key].add_(v)
        else:
            dct[key] = v.clone()

    def _accumulate_weighted(dct: Dict[str, torch.Tensor], key: str, val: torch.Tensor, rows: int) -> None:
        v = val.detach() * rows
        if key in dct:
            dct[key].add_(v)
        else:
            dct[key] = v.clone()

    def _record_metrics(metrics: Dict[str, Any], rows: int) -> None:
        nonlocal total_rows, total_metric_rows, total_loss_t, total_accuracy_t
        total_rows += rows
        loss_t = metrics["loss"].detach()
        if total_loss_t is None:
            total_loss_t = torch.zeros_like(loss_t)
        total_loss_t.add_(loss_t * rows)
        has_sampled_metrics = any(k.startswith(_AVERAGED_METRIC_PREFIXES) for k in metrics)
        if not has_sampled_metrics:
            return
        total_metric_rows += rows
        acc_t = metrics["accuracy"].detach() if isinstance(metrics["accuracy"], torch.Tensor) else torch.tensor(float(metrics["accuracy"]), device=loss_t.device)
        if total_accuracy_t is None:
            total_accuracy_t = torch.zeros_like(acc_t)
        total_accuracy_t.add_(acc_t * rows)
        for key, val in metrics.items():
            if key in {"loss", "accuracy", "_next_hidden"}:
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if key.startswith(_RAW_SUM_METRIC_PREFIXES):
                _accumulate_sum(raw_metric_totals_t, key, val)
            elif key.startswith(_AVERAGED_METRIC_PREFIXES):
                _accumulate_weighted(averaged_metric_totals_t, key, val, rows)

    def _maybe_step_optimizer(metrics: Dict[str, Any], rows: int, chunk_units: float, plan_index: int) -> None:
        nonlocal accum_count, opt_steps, _report_rows, _report_metric_rows, _report_loss_t, _last_save_time, _last_report_time
        nonlocal grad_norm_sum_t, grad_norm_max_t, grad_norm_n
        accum_count += chunk_units
        if not training or accum_count < accum_target:
            return
        if max_grad_norm > 0:
            # Keep the returned norm on-GPU — syncing it (float/.item) every
            # opt step drains the dispatch queue and starves the backward.
            _gn = torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm).detach()
            if grad_norm_sum_t is None:
                grad_norm_sum_t = torch.zeros_like(_gn)
                grad_norm_max_t = torch.zeros_like(_gn)
            grad_norm_sum_t.add_(_gn)
            torch.maximum(grad_norm_max_t, _gn, out=grad_norm_max_t)
            grad_norm_n += 1
        model.bc_step()
        model.bc_zero_grad()
        accum_count = 0.0
        opt_steps += 1

        # Mid-epoch state save at optimizer-step boundaries.
        # snapshot_interval is in seconds (wall clock).
        if save_state_callback and snapshot_interval > 0:
            _now = _time.monotonic()
            if _now - _last_save_time >= snapshot_interval:
                _last_save_time = _now
                active_hiddens = [
                    (lane_idx, 0, lane_hidden[lane_idx].clone())
                    for lane_idx in range(microbatch_target)
                    if lane_hidden is not None
                ]
                # One sync for snapshot persistence — every 15s by default.
                total_loss_float = float(total_loss_t.item()) if total_loss_t is not None else 0.0
                save_state_callback(MidEpochState(
                    next_episode=plan_index + 1,
                    opt_steps=opt_steps,
                    active_hiddens=active_hiddens,
                    total_rows=total_rows,
                    total_loss=total_loss_float,
                    ep_order=ep_order,
                ))

        if step_callback and report_every > 0:
            _report_rows += rows
            loss_t = metrics["loss"].detach()
            if _report_loss_t is None:
                _report_loss_t = torch.zeros_like(loss_t)
            _report_loss_t.add_(loss_t * rows)
            has_sampled_metrics = any(k.startswith(_AVERAGED_METRIC_PREFIXES) for k in metrics)
            if has_sampled_metrics:
                _report_metric_rows += rows
                for key, val in metrics.items():
                    if not isinstance(val, torch.Tensor):
                        continue
                    if key.startswith(_AVERAGED_METRIC_PREFIXES):
                        _accumulate_weighted(_report_avg_totals_t, key, val, rows)
                    elif key.startswith(_RAW_SUM_METRIC_PREFIXES):
                        _accumulate_sum(_report_raw_totals_t, key, val)

            # Gate sync on BOTH step count and wall-clock interval — lets us
            # keep accumulating on GPU without paying for a sync every step.
            step_ready = opt_steps % report_every == 0
            time_ready = (report_interval_seconds <= 0) or (_time.monotonic() - _last_report_time >= report_interval_seconds)
            if step_ready and time_ready:
                _last_report_time = _time.monotonic()
                rd = max(_report_rows, 1)
                md = max(_report_metric_rows, 1)
                # Single GPU→CPU sync for all report metrics.
                pending: Dict[str, torch.Tensor] = {"loss": _report_loss_t}
                pending.update({k: v for k, v in _report_avg_totals_t.items()})
                pending.update({k: v for k, v in _report_raw_totals_t.items()})
                synced = _flush_tensor_dict(pending)
                step_metrics: Dict[str, Any] = {"n_rows": float(_report_rows), "opt_step": opt_steps}
                step_metrics["loss"] = synced["loss"] / rd
                for key in _report_avg_totals_t:
                    step_metrics[key] = synced[key] / md
                for key in _report_raw_totals_t:
                    step_metrics[key] = synced[key]
                step_callback(step_metrics)
                _report_rows = 0
                _report_metric_rows = 0
                _report_loss_t = None
                _report_avg_totals_t.clear()
                _report_raw_totals_t.clear()

    def _apply_prepared_batch(prepared: _PreparedChunkBatch, plan_index: int) -> None:
        nonlocal lane_hidden
        plan = prepared.plan
        hidden_batch = lane_hidden
        # Sample full metrics once per report window during training — the
        # rest of the time we skip MAE/stat computation entirely to keep the
        # GPU dispatch queue deep and utilization high.  For eval we always
        # compute because val-epoch metrics are the output.
        if training:
            now = _time.monotonic()
            sample = (report_interval_seconds <= 0) or (now - _last_report_time >= report_interval_seconds)
            metrics = model.supervised_step(
                prepared.obs,
                prepared.actions,
                class_weights,
                lr=lr,
                hidden=hidden_batch,
                masks=prepared.masks,
                accumulate_only=True,
                head_loss_weights=head_loss_weights,
                loss_scale=float(plan.active_lanes),
                compute_metrics=sample,
            )
        else:
            metrics = model.evaluate_supervised(
                prepared.obs,
                prepared.actions,
                hidden=hidden_batch,
                masks=prepared.masks,
                head_loss_weights=head_loss_weights,
            )

        next_hidden = metrics.pop("_next_hidden", None)
        if next_hidden is not None and lane_hidden is not None:
            lane_hidden.copy_(next_hidden.detach())

        rows = plan.valid_rows
        _record_metrics(metrics, rows)
        _maybe_step_optimizer(metrics, rows, float(plan.active_lanes), plan_index)

    try:
        if use_prefetch:
            # N-ahead prefetch pipeline. Plans are already deterministic; the
            # worker only copies fixed ranges into fixed slots.
            with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                pending: deque[tuple[int, Future[_PreparedChunkBatch]]] = deque()
                slot_counter = 0
                submit_idx = next_plan

                # Prime the pipeline with up to prefetch_depth plans.
                while len(pending) < prefetch_depth and submit_idx < len(packed_plans):
                    slot = slot_counter % num_prefetch_slots
                    slot_counter += 1
                    pending.append((
                        submit_idx,
                        prefetch_executor.submit(_prepare_prefetched_batch, slot, packed_plans[submit_idx]),
                    ))
                    submit_idx += 1

                while pending:
                    plan_idx, future = pending.popleft()
                    prepared = future.result()
                    # Top up before applying so the loader stays busy while
                    # the main thread runs forward+backward on GPU.
                    if len(pending) < prefetch_depth and submit_idx < len(packed_plans):
                        slot = slot_counter % num_prefetch_slots
                        slot_counter += 1
                        pending.append((
                            submit_idx,
                            prefetch_executor.submit(_prepare_prefetched_batch, slot, packed_plans[submit_idx]),
                        ))
                        submit_idx += 1

                    _apply_prepared_batch(prepared, plan_idx)
        else:
            for plan_idx in range(next_plan, len(packed_plans)):
                prepared = _prepare_prefetched_batch(0, packed_plans[plan_idx])
                _apply_prepared_batch(prepared, plan_idx)

        if training and accum_count > 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
            model.bc_step()
            opt_steps += 1
            accum_count = 0.0

        denom = max(total_rows, 1)
        metric_denom = max(total_metric_rows, 1)
        # Single GPU→CPU sync at epoch end — everything that was accumulated on GPU.
        pending: Dict[str, torch.Tensor] = {}
        if total_loss_t is not None:
            pending["loss"] = total_loss_t
        if total_accuracy_t is not None:
            pending["accuracy"] = total_accuracy_t
        pending.update(raw_metric_totals_t)
        pending.update(averaged_metric_totals_t)
        if grad_norm_sum_t is not None:
            pending["__grad_norm_sum"] = grad_norm_sum_t
            pending["__grad_norm_max"] = grad_norm_max_t
        synced = _flush_tensor_dict(pending)
        result: Dict[str, float] = {
            "loss": synced.get("loss", 0.0) / denom,
            "accuracy": synced.get("accuracy", 0.0) / metric_denom,
            "n_rows": float(total_rows),
            "opt_steps": float(opt_steps),
        }
        for key in raw_metric_totals_t:
            result[key] = synced[key]
        for key in averaged_metric_totals_t:
            result[key] = synced[key] / metric_denom
        if grad_norm_n > 0:
            result["grad_norm_mean"] = synced["__grad_norm_sum"] / grad_norm_n
            result["grad_norm_max"] = synced["__grad_norm_max"]
        _apply_stable_epoch_metrics(result)
        return result
    finally:
        for cursor in ordered_cursors:
            _flush_cursor_advise(cursor, fd_cache)
        for fd in fd_cache.values():
            try:
                os.close(fd)
            except OSError:
                pass
