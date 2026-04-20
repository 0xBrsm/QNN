"""Single-epoch supervised training loop for BC.

Processes precomputed episodes in batched chunks with GRU hidden state
carry-forward, gradient accumulation, TBPTT truncation, and optional
pinned-buffer prefetch.
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

from qnn.actions import ACTION_HEADS
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

_ACTION_HEAD_NAMES = list(ACTION_HEADS.keys())
_RAW_SUM_METRIC_PREFIXES = (
    "n_", "correct_", "l1_sum_", "tp_", "fp_", "fn_", "target_pos_", "pred_pos_",
)
_AVERAGED_METRIC_PREFIXES = ("acc_", "mae_")


@dataclass(slots=True)
class PrecomputedEpisode:
    """One episode's observations and actions as contiguous arrays."""
    obs: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    n_samples: int


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


@dataclass(slots=True)
class MidEpochState:
    """Snapshot of training loop state at an optimizer-step boundary.

    Sufficient to resume mid-epoch with bit-identical results:
    the RNG state + next_episode reconstructs the exact cursor order,
    and active_hiddens restores the GRU carry-forward for in-progress
    episodes.
    """
    next_episode: int            # index into the permuted episode list
    opt_steps: int               # optimizer steps completed this epoch
    active_hiddens: list[tuple[int, int, torch.Tensor | None]]
    # each entry: (ep_order_index, cursor.start, cursor.hidden)
    total_rows: int
    total_loss: float


@dataclass(slots=True)
class _ChunkBatchPlan:
    cursors: tuple[_EpisodeCursor, ...]
    starts: tuple[int, ...]
    length: int


@dataclass(slots=True)
class _PreparedChunkBatch:
    plan: _ChunkBatchPlan
    obs: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]


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
    focal_gamma: float = 0.0,
    sparse_discrete: bool = True,
    look_deadzone: float = 0.0,
    look_turn_alpha: float = 0.0,
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

    ep_order: list[int] = list(range(len(episodes)))
    if rng is not None:
        ep_order = [int(i) for i in rng.permutation(len(episodes))]

    total_rows = 0
    total_metric_rows = 0  # rows from sample steps only — denom for MAE/acc
    # Tensor-resident running sums; keep on GPU until epoch end / report boundary.
    total_loss_t: torch.Tensor | None = None
    total_accuracy_t: torch.Tensor | None = None
    raw_metric_totals_t: Dict[str, torch.Tensor] = {}
    averaged_metric_totals_t: Dict[str, torch.Tensor] = {}
    accum_count = 0

    opt_steps = 0
    grad_norm_sum_t: torch.Tensor | None = None
    grad_norm_max_t: torch.Tensor | None = None
    grad_norm_n = 0
    import time as _time
    _last_save_time = _time.monotonic()
    _last_report_time = _time.monotonic()
    _report_rows = 0
    _report_metric_rows = 0
    _report_loss_t: torch.Tensor | None = None
    _report_avg_totals_t: Dict[str, torch.Tensor] = {}
    _report_raw_totals_t: Dict[str, torch.Tensor] = {}

    if training:
        model.bc_zero_grad()

    ordered_cursors = [_EpisodeCursor(episode=episodes[idx]) for idx in ep_order]
    active: list[_EpisodeCursor] = []
    next_episode = 0

    # Resume mid-epoch: fast-forward to saved cursor positions.
    if resume_state is not None:
        next_episode = resume_state.next_episode
        opt_steps = resume_state.opt_steps
        total_rows = resume_state.total_rows
        total_loss_t = torch.tensor(float(resume_state.total_loss), device=model.device)
        # Restore active cursors with their positions and GRU hiddens.
        for ep_order_idx, cursor_start, cursor_hidden in resume_state.active_hiddens:
            if 0 <= ep_order_idx < len(ordered_cursors):
                cursor = ordered_cursors[ep_order_idx]
                cursor.start = cursor_start
                if cursor_hidden is not None:
                    cursor.hidden = cursor_hidden.to(model.device)
                active.append(cursor)
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
            for head in _ACTION_HEAD_NAMES
        }
        action_buffer_slots.append(slot)
    hidden_buffer = (
        torch.zeros((microbatch_target, model.gru_hidden), dtype=torch.float32, device=model.device)
        if model.use_gru
        else None
    )

    def _fill_active() -> None:
        nonlocal next_episode
        while len(active) < microbatch_target and next_episode < len(ordered_cursors):
            active.append(ordered_cursors[next_episode])
            next_episode += 1

    def _remaining(cursor: _EpisodeCursor) -> int:
        return cursor.episode.n_samples - cursor.start

    def _make_plan_from_specs(
        specs: Sequence[tuple[_EpisodeCursor, int]],
        accum_count_value: int,
    ) -> _ChunkBatchPlan | None:
        if not specs:
            return None
        tail_lengths = [cursor.episode.n_samples - start for cursor, start in specs if (cursor.episode.n_samples - start) < chunk_size]
        if tail_lengths:
            tail_length = min(tail_lengths)
            remaining_to_step = accum_target - accum_count_value if training else microbatch_target
            selected = [
                (cursor, start)
                for cursor, start in specs
                if (cursor.episode.n_samples - start) == tail_length
            ][:max(1, min(microbatch_target, remaining_to_step))]
            return _ChunkBatchPlan(
                cursors=tuple(cursor for cursor, _ in selected),
                starts=tuple(start for _, start in selected),
                length=tail_length,
            )
        remaining_to_step = accum_target - accum_count_value if training else microbatch_target
        group_size = min(len(specs), microbatch_target, remaining_to_step)
        selected = list(specs[:group_size])
        return _ChunkBatchPlan(
            cursors=tuple(cursor for cursor, _ in selected),
            starts=tuple(start for _, start in selected),
            length=chunk_size,
        )

    def _current_specs() -> list[tuple[_EpisodeCursor, int]]:
        return [(cursor, cursor.start) for cursor in active if _remaining(cursor) > 0]

    def _project_specs_after_plan(plan: _ChunkBatchPlan) -> list[tuple[_EpisodeCursor, int]]:
        next_starts = {id(cursor): start + plan.length for cursor, start in zip(plan.cursors, plan.starts)}
        return [
            (cursor, next_starts.get(id(cursor), cursor.start))
            for cursor in active
            if (cursor.episode.n_samples - next_starts.get(id(cursor), cursor.start)) > 0
        ]

    def _prepare_prefetched_batch(slot_idx: int, plan: _ChunkBatchPlan) -> _PreparedChunkBatch:
        group_size = len(plan.cursors)
        obs_batch: dict[str, torch.Tensor] = {}
        for key in plan.cursors[0].episode.obs:
            dst = obs_buffer_slots[slot_idx][key][:plan.length, :group_size]
            for idx, (cursor, start) in enumerate(zip(plan.cursors, plan.starts)):
                arr = cursor.episode.obs[key]
                chunk = arr[start:start + plan.length]
                dst[:, idx, ...].copy_(torch.from_numpy(np.asarray(chunk)))
                _maybe_advise_range(arr, cursor.advised_obs, key, start + plan.length, fd_cache)
            obs_batch[key] = dst
        act_batch: dict[str, torch.Tensor] = {}
        for head in _ACTION_HEAD_NAMES:
            dst = action_buffer_slots[slot_idx][head][:plan.length, :group_size]
            for idx, (cursor, start) in enumerate(zip(plan.cursors, plan.starts)):
                arr = cursor.episode.actions[head]
                chunk = arr[start:start + plan.length]
                dst[:, idx, ...].copy_(torch.from_numpy(np.asarray(chunk)))
                _maybe_advise_range(arr, cursor.advised_act, head, start + plan.length, fd_cache)
            act_batch[head] = dst
        return _PreparedChunkBatch(plan=plan, obs=obs_batch, actions=act_batch)

    def _hidden_batch_for_plan(plan: _ChunkBatchPlan) -> torch.Tensor | None:
        if hidden_buffer is None:
            return None
        group_size = len(plan.cursors)
        batch_hidden = hidden_buffer[:group_size]
        batch_hidden.zero_()
        for idx, cursor in enumerate(plan.cursors):
            if cursor.hidden is not None:
                batch_hidden[idx].copy_(cursor.hidden)
        return batch_hidden

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

    def _maybe_step_optimizer(metrics: Dict[str, Any], rows: int, chunk_count: int) -> None:
        nonlocal accum_count, opt_steps, _report_rows, _report_metric_rows, _report_loss_t, _last_save_time, _last_report_time
        nonlocal grad_norm_sum_t, grad_norm_max_t, grad_norm_n
        accum_count += chunk_count
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
        accum_count = 0
        opt_steps += 1

        # Mid-epoch state save at optimizer-step boundaries.
        # snapshot_interval is in seconds (wall clock).
        if save_state_callback and snapshot_interval > 0:
            _now = _time.monotonic()
            if _now - _last_save_time >= snapshot_interval:
                _last_save_time = _now
                cursor_to_ep_idx = {id(cursor): oi for oi, cursor in enumerate(ordered_cursors)}
                active_hiddens = [
                    (cursor_to_ep_idx.get(id(c), -1), c.start,
                     c.hidden.clone() if c.hidden is not None else None)
                    for c in active
                ]
                # One sync for snapshot persistence — every 15s by default.
                total_loss_float = float(total_loss_t.item()) if total_loss_t is not None else 0.0
                save_state_callback(MidEpochState(
                    next_episode=next_episode,
                    opt_steps=opt_steps,
                    active_hiddens=active_hiddens,
                    total_rows=total_rows,
                    total_loss=total_loss_float,
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

    def _apply_prepared_batch(prepared: _PreparedChunkBatch) -> None:
        plan = prepared.plan
        hidden_batch = _hidden_batch_for_plan(plan)
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
                accumulate_only=True,
                head_loss_weights=head_loss_weights,
                focal_gamma=focal_gamma,
                sparse_discrete=sparse_discrete,
                look_deadzone=look_deadzone,
                look_turn_alpha=look_turn_alpha,
                loss_scale=float(len(plan.cursors)),
                compute_metrics=sample,
            )
        else:
            metrics = model.evaluate_supervised(
                prepared.obs,
                prepared.actions,
                hidden=hidden_batch,
                head_loss_weights=head_loss_weights,
                focal_gamma=focal_gamma,
                sparse_discrete=sparse_discrete,
                look_deadzone=look_deadzone,
                look_turn_alpha=look_turn_alpha,
            )

        next_hidden = metrics.pop("_next_hidden", None)
        if next_hidden is not None:
            for idx, cursor in enumerate(plan.cursors):
                cursor.hidden = next_hidden[idx].detach()

        rows = plan.length * len(plan.cursors)
        _record_metrics(metrics, rows)
        _maybe_step_optimizer(metrics, rows, len(plan.cursors))

        for cursor in plan.cursors:
            cursor.start += plan.length

    try:
        _fill_active()
        current_specs = _current_specs()
        current_plan = _make_plan_from_specs(current_specs, accum_count)
        if current_plan is None:
            return _empty

        if use_prefetch:
            # N-ahead prefetch pipeline. Keeps `prefetch_depth` batches in
            # flight so main thread never waits on the loader. Projection
            # maintains its OWN cursor-start tracking (virt_starts) and its
            # OWN episode cursor (proj_next_episode), independent of main's
            # active/next_episode. Cursors still share identity with main —
            # main updates cursor.start when applying, and projection's
            # virt_starts for that cursor converges to the same value since
            # both advance by plan.length.
            with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                pending: deque[Future[_PreparedChunkBatch]] = deque()
                slot_counter = 0

                virt_starts: dict[int, int] = {id(c): c.start for c in active}
                virt_active: list[_EpisodeCursor] = list(active)
                proj_next_episode = next_episode
                proj_accum = accum_count

                def _plan_next() -> _ChunkBatchPlan | None:
                    nonlocal proj_accum, proj_next_episode
                    virt_active[:] = [
                        c for c in virt_active
                        if c.episode.n_samples - virt_starts[id(c)] > 0
                    ]
                    # Fill to microbatch_target from ordered_cursors.
                    while len(virt_active) < microbatch_target and proj_next_episode < len(ordered_cursors):
                        cursor = ordered_cursors[proj_next_episode]
                        virt_active.append(cursor)
                        virt_starts[id(cursor)] = cursor.start
                        proj_next_episode += 1
                    specs = [(c, virt_starts[id(c)]) for c in virt_active]
                    plan = _make_plan_from_specs(specs, proj_accum)
                    if plan is None:
                        return None
                    for c, s in zip(plan.cursors, plan.starts):
                        virt_starts[id(c)] = s + plan.length
                    proj_accum += len(plan.cursors)
                    if training and proj_accum >= accum_target:
                        proj_accum = 0
                    return plan

                # Prime the pipeline with up to prefetch_depth plans.
                while len(pending) < prefetch_depth:
                    plan = _plan_next()
                    if plan is None:
                        break
                    slot = slot_counter % num_prefetch_slots
                    slot_counter += 1
                    pending.append(prefetch_executor.submit(_prepare_prefetched_batch, slot, plan))

                while pending:
                    prepared = pending.popleft().result()
                    # Top up before applying so the loader stays busy while
                    # the main thread runs forward+backward on GPU.
                    if len(pending) < prefetch_depth:
                        plan = _plan_next()
                        if plan is not None:
                            slot = slot_counter % num_prefetch_slots
                            slot_counter += 1
                            pending.append(prefetch_executor.submit(_prepare_prefetched_batch, slot, plan))

                    _apply_prepared_batch(prepared)
                    for cursor in active:
                        if _remaining(cursor) <= 0:
                            _flush_cursor_advise(cursor, fd_cache)
                    active = [cursor for cursor in active if _remaining(cursor) > 0]
                    _fill_active()
        else:
            while active and current_plan is not None:
                prepared = _prepare_prefetched_batch(0, current_plan)
                _apply_prepared_batch(prepared)
                for cursor in active:
                    if _remaining(cursor) <= 0:
                        _flush_cursor_advise(cursor, fd_cache)
                active = [cursor for cursor in active if _remaining(cursor) > 0]
                _fill_active()
                current_plan = _make_plan_from_specs(_current_specs(), accum_count)

        if training and accum_count > 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
            model.bc_step()

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
        }
        for key in raw_metric_totals_t:
            result[key] = synced[key]
        for key in averaged_metric_totals_t:
            result[key] = synced[key] / metric_denom
        if grad_norm_n > 0:
            result["grad_norm_mean"] = synced["__grad_norm_sum"] / grad_norm_n
            result["grad_norm_max"] = synced["__grad_norm_max"]
        return result
    finally:
        for fd in fd_cache.values():
            try:
                os.close(fd)
            except OSError:
                pass


def run_epoch(
    model: QNNPolicy,
    episodes: Sequence[PrecomputedEpisode],
    batch_size: int,
    sequence_length: int,
    *,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
    max_grad_norm: float = 1.0,
    tbptt_limit: int = 256,
    head_loss_weights: Mapping[str, float] | None = None,
    focal_gamma: float = 0.0,
    sparse_discrete: bool = True,
    look_deadzone: float = 0.0,
    look_turn_alpha: float = 0.0,
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
    """Run one epoch of supervised training or evaluation over all episodes."""
    use_full_episode = int(sequence_length) <= 0
    if use_full_episode:
        if tbptt_limit > 0:
            chunk_size = max(int(tbptt_limit), 64)
        elif episodes:
            chunk_size = max(ep.n_samples for ep in episodes)
        else:
            chunk_size = 64
    else:
        chunk_size = max(int(sequence_length), 1)

    return _run_batched(
        model,
        episodes,
        max(1, int(batch_size)),
        chunk_size,
        class_weights=class_weights,
        lr=lr,
        rng=rng,
        max_grad_norm=max_grad_norm,
        head_loss_weights=head_loss_weights,
        focal_gamma=focal_gamma,
        sparse_discrete=sparse_discrete,
        look_deadzone=look_deadzone,
        look_turn_alpha=look_turn_alpha,
        step_callback=step_callback,
        report_every=report_every,
        report_interval_seconds=report_interval_seconds,
        pin_memory=pin_memory,
        prefetch=prefetch,
        microbatch_size=microbatch_size,
        save_state_callback=save_state_callback,
        snapshot_interval=snapshot_interval,
        resume_state=resume_state,
    )
