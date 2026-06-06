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
import heapq
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


# Page-drop hint machinery is opt-in. The default-off behavior is safe;
# enabling it requires running on a build where every episode array
# really is a file-backed memmap (and not, e.g., a heap copy returned
# by a ProcessPoolExecutor worker whose ``.base`` chain numpy may
# reconstruct inconsistently — that path has been observed to segfault
# inside ``_root_memmap`` during end-of-epoch flush). The optimization
# was designed for the load-time mmap walk; with the chunked-prefetch
# path operating on potentially-non-memmap arrays returned through
# pickle, leave it off until the underlying ``.base`` chain instability
# is root-caused (see project_bc_collect_status). The kernel reclaims
# clean file-cache pages under pressure regardless.
_PAGEDROP_ENABLED = bool(int(os.environ.get("QNN_BC_PAGEDROP", "0") or 0))


def _maybe_advise_range(
    arr: np.ndarray,
    tracker: dict[str, int],
    key: str,
    consumed_end: int,
    fd_cache: dict[str, int],
    *,
    force: bool = False,
) -> None:
    if not _PAGEDROP_ENABLED:
        return
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
    if not _PAGEDROP_ENABLED:
        return
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
    if (
        "tp_fire_masked" in result
        and "fp_fire_masked" in result
        and "fn_fire_masked" in result
    ):
        _stable_binary_metrics_from_counts(result, prefix="fire_masked")
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
    dst_indices: np.ndarray
    reset_indices: np.ndarray
    compact_order: np.ndarray


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
    import os as _os
    # QNN_PROFILE_STEPS=N → time data-wait vs apply (forward+backward+opt) for
    # the first N optimizer steps, print decomposition, then continue normally.
    # Synchronizes the GPU around `_apply_prepared_batch` so the GPU compute
    # actually finishes inside the measured window. Adds per-step sync overhead
    # only while profiling.
    _profile_steps_target = int(_os.environ.get("QNN_PROFILE_STEPS", "0") or 0)
    _profile_skip_target = int(_os.environ.get("QNN_PROFILE_SKIP", "100") or 0)
    _prof_active = _profile_steps_target > 0 and training
    if _prof_active:
        print(f"  [bc] profiling enabled: skip {_profile_skip_target} steps, measure next {_profile_steps_target}", flush=True)
    _prof_seen = 0          # total opt steps observed
    _prof_n = 0             # steps actually counted (after warmup skip)
    _prof_t_wait = 0.0
    _prof_t_apply = 0.0
    _prof_window_start = 0.0  # wall start of the measured window
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

    # Per-batch dequant on GPU. Chunked obs is staged on pinned host as
    # native engine widths (uint8 / int16 / uint32 / float16). The model
    # expects the legacy dequantized layout (``self_scalars``,
    # ``spatial_scalars``, ``entity_scalars_raw``, plus id tensors). The
    # GPU-resident path runs these once at preload because everything
    # fits in VRAM; the chunked path streams batches, so we run the
    # dequant lazily per batch on the (already-on-GPU) tensors. The
    # dequantizers are idempotent and short-circuit when the legacy
    # keys are present, so they're safe to call unconditionally.
    _native_to_dequant: tuple = ()
    if any(k in episodes[0].obs for k in ("health", "vel", "self_items")):
        from qnn.model.dequant import (
            SelfDequantizer, SpatialDequantizer, EntityDequantizer,
        )
        _native_to_dequant = (
            SelfDequantizer().to(model.device).eval(),
            SpatialDequantizer().to(model.device).eval(),
            EntityDequantizer().to(model.device).eval(),
        )

    def _move_and_dequant(obs_cpu: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pinned-host obs → on-GPU dequantized obs.

        Native fields arrive as (T, B, ...) for sequence chunks. The
        dequantizers operate on rank-1 batch shapes (or rank-2 like
        ``(B, K)``), so we flatten the leading (T, B) → (T*B,) before
        dequant and reshape back. New keys produced by the dequantizers
        are emitted at (T, B, ...) shape so downstream model code keys
        off the sequence rank correctly.
        """
        # Async H→D copy. Pinned host buffers + non_blocking=True lets
        # the next dispatch queue stage overlap with the transfer.
        gpu_obs = {k: v.to(model.device, non_blocking=True) for k, v in obs_cpu.items()}
        if not _native_to_dequant:
            return gpu_obs
        # Detect (T, B, ...) vs (B, ...) layout from a known row-indexed
        # field. ``health`` is (T, B) when chunked, (B,) when flat.
        ref = gpu_obs.get("health")
        if ref is not None and ref.ndim >= 2:
            T = int(ref.shape[0])
            B = int(ref.shape[1])
            TB = T * B
            flat: dict[str, torch.Tensor] = {}
            for k, v in gpu_obs.items():
                # Token-indexed and row-indexed fields share leading
                # (T, B) dims; flatten to (T*B, *tail).
                flat[k] = v.reshape(TB, *v.shape[2:])
            for mod in _native_to_dequant:
                flat = mod(flat)
            # Reshape back. Original native keys keep their (T, B, ...)
            # shape; newly-added dequant keys go from (T*B, *) → (T, B, *).
            out: dict[str, torch.Tensor] = {}
            for k, v in flat.items():
                if k in gpu_obs:
                    out[k] = v.reshape(gpu_obs[k].shape)
                else:
                    out[k] = v.reshape(T, B, *v.shape[1:])
            return out
        # Flat (B, ...) layout — dequant in place.
        for mod in _native_to_dequant:
            gpu_obs = mod(gpu_obs)
        return gpu_obs

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
        lane_heap = [(0, idx) for idx in range(microbatch_target)]
        heapq.heapify(lane_heap)
        for cursor in ordered_cursors:
            if cursor.episode.n_samples <= 0:
                continue
            lane_length, lane = heapq.heappop(lane_heap)
            item = _LaneItem(cursor=cursor, lane=lane, lane_start=lane_length)
            lane_items[lane].append(item)
            lane_lengths[lane] = lane_length + cursor.episode.n_samples
            heapq.heappush(lane_heap, (lane_lengths[lane], lane))

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
                slices = tuple(chunk_slices[chunk_idx])
                dst_indices = np.empty((valid_rows,), dtype=np.int64)
                reset_indices: list[int] = []
                cursor = 0
                for sl in slices:
                    base = sl.dst_start * microbatch_target + sl.item.lane
                    end = cursor + sl.length
                    dst_indices[cursor:end] = base + (
                        np.arange(sl.length, dtype=np.int64) * microbatch_target
                    )
                    cursor = end
                    if sl.reset:
                        reset_indices.append(base)
                plans.append(_PackedChunkPlan(
                    slices=slices,
                    length=chunk_size,
                    batch_size=microbatch_target,
                    valid_rows=valid_rows,
                    active_lanes=len({sl.item.lane for sl in slices}),
                    dst_indices=dst_indices,
                    reset_indices=np.asarray(reset_indices, dtype=np.int64),
                    compact_order=np.argsort(dst_indices, kind="stable"),
                ))
        return plans

    packed_plans = _build_packed_plans()
    if next_plan >= len(packed_plans):
        return _empty

    # Streaming-pad config for token-indexed entity fields. Episodes
    # stored in the native layout carry these as ``(total_tokens, ...)``
    # mmap views plus a per-episode ``entity_indptr``; the per-batch
    # buffer must still allocate the padded ``(L, B, n_max, ...)`` shape
    # the EntityDequantizer + tokenizer downstream consume. The pad
    # itself is run inside ``_prepare_prefetched_batch``.
    from qnn.bc.train import (
        _NATIVE_TOKEN_INDEXED_OBS_FIELDS as _TOK_FIELDS,
        _ENTITY_TYPES_EMPTY_SENTINEL as _TOK_EMPTY,
    )
    from qnn.vocab import MAX_TOKEN_OBJECTS as _MAX_TOK
    # Decision is per-load: a single load_precomputed call produces
    # either all-unpadded (entity_indptr set) or all-padded episodes,
    # never a mix. Inspect the first episode that has any obs to settle
    # the buffer shape.
    _streaming_pad = (
        episodes[0].entity_indptr is not None
        if episodes else False
    )

    def _padded_per_token_shape(key: str, arr: np.ndarray) -> tuple[int, ...]:
        # Token-indexed unpadded: arr.shape == (total_tokens, *tail).
        # Padded buffer needs (n_max, *tail).
        if _streaming_pad and key in _TOK_FIELDS:
            return (_MAX_TOK,) + tuple(arr.shape[1:])
        return tuple(arr.shape[1:])

    # Pinned slots: one per in-flight batch plus one being consumed.
    # prefetch_depth batches can be staged at once.
    num_prefetch_slots = max(2, prefetch_depth + 1) if use_prefetch else 2
    obs_buffer_slots = []
    for _slot_i in range(num_prefetch_slots):
        slot = {
            key: torch.empty(
                (chunk_size, microbatch_target,
                 *_padded_per_token_shape(key, episodes[0].obs[key])),
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
        dst_index = torch.from_numpy(plan.dst_indices)

        # Streaming-pad precompute: per slice, capture the token-axis
        # gather plan (row_starts into the episode's unpadded flat
        # array + the valid mask) so each token-indexed key reuses the
        # same indices across all 20 keys.
        slice_pad_plans: list[tuple[np.ndarray, np.ndarray] | None] = []
        if _streaming_pad:
            slots_v = np.arange(_MAX_TOK, dtype=np.int64)
            for sl in plan.slices:
                ep = sl.item.cursor.episode
                if ep.entity_indptr is None:
                    slice_pad_plans.append(None)
                    continue
                rs = sl.src_start
                re = sl.src_start + sl.length
                ip = ep.entity_indptr[rs:re + 1]
                counts = (ip[1:] - ip[:-1]).astype(np.int64, copy=False)
                counts_clamped = np.minimum(counts, _MAX_TOK)
                valid = slots_v[None, :] < counts_clamped[:, None]
                row_starts = ip[:-1].astype(np.int64, copy=False)
                gather_idx = np.where(valid, row_starts[:, None] + slots_v[None, :], 0)
                slice_pad_plans.append((gather_idx, valid))

        obs_batch: dict[str, torch.Tensor] = {}
        for key in episodes[0].obs:
            dst = obs_buffer_slots[slot_idx][key][:plan.length, :plan.batch_size]
            is_token = _streaming_pad and key in _TOK_FIELDS
            if is_token:
                # entity_types drives the EntityDequantizer's per-slot
                # mask (mask_actor = entity_types == TOKEN_ACTOR), so it
                # MUST carry the -1 sentinel for invalid slots. Every
                # other token-indexed field is gated by that mask on
                # the GPU side — invalid-slot garbage gets overwritten
                # with zero by torch.where. So we skip the per-key
                # np.where mask for non-entity_types fields and just
                # let flat[gather_idx] populate the buffer.
                needs_mask = (key == "entity_types")
                fill = _TOK_EMPTY if needs_mask else 0
                chunks = []
                for sl, pad_plan in zip(plan.slices, slice_pad_plans):
                    ep = sl.item.cursor.episode
                    flat = np.asarray(ep.obs[key])
                    if pad_plan is None:
                        chunks.append(flat[sl.src_start:sl.src_start + sl.length])
                        continue
                    gather_idx, valid = pad_plan
                    if flat.shape[0] == 0:
                        block = np.full((sl.length, _MAX_TOK) + flat.shape[1:], fill, dtype=flat.dtype)
                    elif needs_mask:
                        padded = flat[gather_idx]
                        if padded.ndim > 2:
                            mask = valid.reshape(valid.shape + (1,) * (padded.ndim - 2))
                        else:
                            mask = valid
                        block = np.where(mask, padded, np.asarray(fill, dtype=padded.dtype))
                    else:
                        # Garbage at invalid slots is acceptable — the
                        # EntityDequantizer's per-type torch.where on
                        # GPU overwrites with the scalars buffer's
                        # zero-init for non-actor / non-emit slots.
                        block = flat[gather_idx]
                    chunks.append(block)
            else:
                chunks = [
                    np.asarray(sl.item.cursor.episode.obs[key][sl.src_start:sl.src_start + sl.length])
                    for sl in plan.slices
                ]
            if chunks:
                src = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
                flat_dst = dst.reshape(plan.length * plan.batch_size, *dst.shape[2:])
                flat_dst.index_copy_(0, dst_index, torch.from_numpy(src))
            if not is_token:
                for sl in plan.slices:
                    arr = sl.item.cursor.episode.obs[key]
                    _maybe_advise_range(arr, sl.item.cursor.advised_obs, key, sl.src_start + sl.length, fd_cache)
            obs_batch[key] = dst
        act_batch: dict[str, torch.Tensor] = {}
        for head in action_names:
            dst = action_buffer_slots[slot_idx][head][:plan.length, :plan.batch_size]
            chunks = [
                np.asarray(sl.item.cursor.episode.actions[head][sl.src_start:sl.src_start + sl.length])
                for sl in plan.slices
            ]
            if chunks:
                src = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
                flat_dst = dst.reshape(plan.length * plan.batch_size, *dst.shape[2:])
                flat_dst.index_copy_(0, dst_index, torch.from_numpy(src))
            for sl in plan.slices:
                arr = sl.item.cursor.episode.actions[head]
                _maybe_advise_range(arr, sl.item.cursor.advised_act, head, sl.src_start + sl.length, fd_cache)
            act_batch[head] = dst
        flat_valid = masks["valid_mask"].reshape(plan.length * plan.batch_size)
        flat_valid.index_fill_(0, dst_index, True)
        if plan.reset_indices.size:
            flat_reset = masks["reset_mask"].reshape(plan.length * plan.batch_size)
            flat_reset.index_fill_(0, torch.from_numpy(plan.reset_indices), True)
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
        # Transfer pinned-host obs/actions/masks to GPU and dequantize
        # native fields in one pass. The dequant must run after the H→D
        # copy because Tokenizer / heads read the legacy keys directly
        # (e.g. ``self_scalars`` in qnn.bc.heads.fire_token.forward) and
        # would otherwise see only the native uint8 / int16 inputs.
        gpu_obs = _move_and_dequant(prepared.obs)
        gpu_actions = {
            k: v.to(model.device, non_blocking=True) for k, v in prepared.actions.items()
        }
        gpu_masks = {
            k: v.to(model.device, non_blocking=True) for k, v in prepared.masks.items()
        }
        # Sample full metrics once per report window during training — the
        # rest of the time we skip MAE/stat computation entirely to keep the
        # GPU dispatch queue deep and utilization high.  For eval we always
        # compute because val-epoch metrics are the output.
        if training:
            now = _time.monotonic()
            sample = (report_interval_seconds <= 0) or (now - _last_report_time >= report_interval_seconds)
            metrics = model.supervised_step(
                gpu_obs,
                gpu_actions,
                class_weights,
                lr=lr,
                hidden=hidden_batch,
                masks=gpu_masks,
                accumulate_only=True,
                head_loss_weights=head_loss_weights,
                loss_scale=float(plan.active_lanes),
                compute_metrics=sample,
            )
        else:
            metrics = model.evaluate_supervised(
                gpu_obs,
                gpu_actions,
                hidden=hidden_batch,
                masks=gpu_masks,
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
                    # Profiling: time wait + apply, but only AFTER skipping
                    # warmup steps (page faults, JIT, cache cold). Skip is
                    # measured in optimizer steps observed so far.
                    _measure = _prof_active and (_prof_seen >= _profile_skip_target) and (_prof_n < _profile_steps_target)
                    if _measure:
                        _t0 = _time.monotonic()
                        prepared = future.result()
                        _prof_t_wait += _time.monotonic() - _t0
                    else:
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

                    if _measure:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _t0 = _time.monotonic()
                        _apply_prepared_batch(prepared, plan_idx)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _prof_t_apply += _time.monotonic() - _t0
                        if _prof_n == 0:
                            _prof_window_start = _t0  # wall start of measured window
                        _prof_n += 1
                        if _prof_n == _profile_steps_target:
                            _total = _time.monotonic() - _prof_window_start
                            print(
                                f"  [bc] step timing (n={_prof_n} after {_profile_skip_target} warmup steps): "
                                f"wait_data={_prof_t_wait*1000/_prof_n:.3f}ms  "
                                f"apply={_prof_t_apply*1000/_prof_n:.3f}ms  "
                                f"wall_total={_total*1000/_prof_n:.3f}ms  "
                                f"wait_frac={_prof_t_wait/_total*100:.1f}%  "
                                f"apply_frac={_prof_t_apply/_total*100:.1f}%",
                                flush=True,
                            )
                    else:
                        _apply_prepared_batch(prepared, plan_idx)
                    _prof_seen += 1
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


# ── GPU-resident fast path ───────────────────────────────────────
#
# Bypasses the prefetch + lane-packing pipeline above. Used when
# preload_to_gpu=true is set in machine.json AND the model has no
# recurrence. Concatenates every episode's obs/action arrays into
# per-key GPU tensors once at startup, then each epoch loops over
# shuffled frame indices with index_select — no CPU collate, no
# host→device copy per batch. Produces the same metric dict shape
# as ``_run_batched`` so downstream history/checkpoint code is
# unchanged.


def preload_episodes_to_gpu(
    episodes: Sequence["PrecomputedEpisode"],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Concatenate all episodes' obs and action arrays onto ``device``.

    Returns ``(gpu_obs, gpu_actions)`` where each dict's tensors are
    contiguous along frame axis 0. Per-key dtype is preserved from the
    source npy arrays (already bf16/uint8/int8 in the cache).

    Token-indexed obs fields need a ``(N_total_frames, MAX_TOKEN_OBJECTS,
    ...)`` padded layout for the GPU-resident path. ``_load_precomputed``
    produces sub-episodes that already view a single global per-key
    buffer (post-filter rows / tokens, concatenated across all shards),
    so we can pad once globally — one ``_pad_entity_batch`` call per
    token-indexed key over the full corpus — instead of looping per
    episode. For episodes that pre-date the global-buffer layout
    (``entity_indptr is None``), the obs is already padded and the
    per-episode path stays the same.
    """
    if not episodes:
        return {}, {}
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
    # every batch in the Tokenizer — eliminates per-batch dequant
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
    # frames (see qnn.bc.heads.loss_shaping.flat_distance_weight).
    #
    # fire is a 0/1 byte; we read it directly.
    # jump-positive is derived from move (ud-axis == MOVE_CLASS_POS == 2);
    # we compute it on the fly per episode and store under a dedicated key
    # so policy.py can pick it up without touching MOVE_HEAD encoding.
    from qnn.bc.heads.loss_shaping import per_frame_distance_to_pos

    if "fire" in episodes[0].actions:
        gpu_actions["fire_distance_to_pos"] = _concat_arrays([
            per_frame_distance_to_pos(np.asarray(ep.actions["fire"]).reshape(-1))
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

    return gpu_obs, gpu_actions


def run_epoch_gpu_resident(
    model: "QNNPolicy",
    gpu_obs: dict[str, torch.Tensor],
    gpu_actions: dict[str, torch.Tensor],
    batch_size: int,
    *,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
    max_grad_norm: float = 1.0,
    head_loss_weights: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """One epoch over GPU-resident tensors.

    Equivalent semantics to ``_run_batched`` for memoryless models
    (no GRU, ``sequence_length`` ignored). Caller picks training vs
    eval by passing ``class_weights`` + ``lr`` or omitting both.
    """
    if not gpu_obs:
        return {"loss": 0.0, "accuracy": 0.0, "n_rows": 0.0}

    training = class_weights is not None and lr is not None
    if training:
        model.model.train()
    else:
        model.model.eval()

    # Shuffle frame indices for training; sequential for eval.
    n_frames = int(next(iter(gpu_obs.values())).shape[0])
    if training and rng is not None:
        perm = rng.permutation(n_frames)
        indices = torch.from_numpy(perm).to(model.device)
    else:
        indices = torch.arange(n_frames, device=model.device)

    import time as _time
    _t_start = _time.monotonic()

    bs = max(1, int(batch_size))
    total_rows = 0
    total_loss_t: torch.Tensor | None = None
    raw_sum_totals: Dict[str, torch.Tensor] = {}
    avg_totals: Dict[str, torch.Tensor] = {}
    metric_rows = 0
    opt_steps = 0
    grad_norm_sum_t: torch.Tensor | None = None
    grad_norm_max_t: torch.Tensor | None = None
    grad_norm_n = 0

    for start in range(0, n_frames, bs):
        end = min(start + bs, n_frames)
        idx = indices[start:end]
        rows = end - start

        batch_obs = {k: v.index_select(0, idx) for k, v in gpu_obs.items()}
        batch_actions = {k: v.index_select(0, idx) for k, v in gpu_actions.items()}

        if training:
            metrics = model.supervised_step(
                batch_obs,
                batch_actions,
                class_weights,
                lr=lr,
                accumulate_only=True,
                head_loss_weights=head_loss_weights,
                loss_scale=1.0,
                compute_metrics=True,
            )
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
        else:
            metrics = model.evaluate_supervised(
                batch_obs,
                batch_actions,
                head_loss_weights=head_loss_weights,
            )

        # Aggregate loss (weighted by rows) on GPU.
        loss_t = metrics.get("loss")
        if isinstance(loss_t, torch.Tensor):
            if total_loss_t is None:
                total_loss_t = torch.zeros_like(loss_t)
            total_loss_t.add_(loss_t.detach() * rows)

        # Raw-sum prefixes (tp_/fp_/fn_/tn_/n_/correct_/...) — accumulate as-is.
        # Averaged prefixes (acc_/f1_/precision_/recall_/loss_/cos_sim_/...) — weighted by rows.
        has_sampled = False
        for key, val in metrics.items():
            if key in ("loss", "_next_hidden"):
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if key.startswith(_RAW_SUM_METRIC_PREFIXES):
                if key not in raw_sum_totals:
                    raw_sum_totals[key] = torch.zeros_like(val)
                raw_sum_totals[key].add_(val.detach())
            elif key.startswith(_AVERAGED_METRIC_PREFIXES):
                has_sampled = True
                if key not in avg_totals:
                    avg_totals[key] = torch.zeros_like(val)
                avg_totals[key].add_(val.detach() * rows)
        if has_sampled:
            metric_rows += rows

        total_rows += rows

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
