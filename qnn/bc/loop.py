"""Single-epoch supervised training loop for BC.

Processes precomputed episodes in batched chunks with GRU hidden state
carry-forward, gradient accumulation, TBPTT truncation, and optional
pinned-buffer prefetch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from qnn.actions import ACTION_HEADS
from qnn.model.policy import QNNPolicy

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


@dataclass(slots=True)
class _EpisodeCursor:
    episode: PrecomputedEpisode
    start: int = 0
    hidden: torch.Tensor | None = None


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
    pin_memory: bool = True,
    prefetch: bool = False,
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
    microbatch_target = accum_target
    device_type = model.device.type if isinstance(model.device, torch.device) else str(model.device)
    use_pinned_host = device_type != "cpu" and bool(pin_memory)
    use_prefetch = bool(prefetch)

    ep_order: list[int] = list(range(len(episodes)))
    if rng is not None:
        ep_order = [int(i) for i in rng.permutation(len(episodes))]

    total_rows = 0
    total_loss = 0.0
    total_accuracy = 0.0
    raw_metric_totals: Dict[str, float] = {}
    averaged_metric_totals: Dict[str, float] = {}
    accum_count = 0

    opt_steps = 0
    _report_rows = 0
    _report_loss = 0.0
    _report_avg_totals: Dict[str, float] = {}
    _report_raw_totals: Dict[str, float] = {}

    if training:
        model.bc_zero_grad()

    ordered_cursors = [_EpisodeCursor(episode=episodes[idx]) for idx in ep_order]
    active: list[_EpisodeCursor] = []
    next_episode = 0
    obs_buffer_slots = []
    for _ in range(2):
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
    for _ in range(2):
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
                dst[:, idx, ...].copy_(torch.from_numpy(np.asarray(cursor.episode.obs[key][start:start + plan.length])))
            obs_batch[key] = dst
        act_batch: dict[str, torch.Tensor] = {}
        for head in _ACTION_HEAD_NAMES:
            dst = action_buffer_slots[slot_idx][head][:plan.length, :group_size]
            for idx, (cursor, start) in enumerate(zip(plan.cursors, plan.starts)):
                dst[:, idx, ...].copy_(torch.from_numpy(np.asarray(cursor.episode.actions[head][start:start + plan.length])))
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

    def _record_metrics(metrics: Dict[str, float], rows: int) -> None:
        nonlocal total_rows, total_loss, total_accuracy
        total_rows += rows
        total_loss += float(metrics["loss"]) * rows
        total_accuracy += float(metrics["accuracy"]) * rows
        for key, val in metrics.items():
            if key in {"loss", "accuracy", "_next_hidden"}:
                continue
            if key.startswith(_RAW_SUM_METRIC_PREFIXES):
                raw_metric_totals[key] = raw_metric_totals.get(key, 0.0) + float(val)
            elif key.startswith(_AVERAGED_METRIC_PREFIXES):
                averaged_metric_totals[key] = averaged_metric_totals.get(key, 0.0) + float(val) * rows

    def _maybe_step_optimizer(metrics: Dict[str, float], rows: int, chunk_count: int) -> None:
        nonlocal accum_count, opt_steps, _report_rows, _report_loss
        accum_count += chunk_count
        if not training or accum_count < accum_target:
            return
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
        model.bc_step()
        model.bc_zero_grad()
        accum_count = 0
        opt_steps += 1

        if step_callback and report_every > 0:
            _report_rows += rows
            _report_loss += float(metrics["loss"]) * rows
            for key, val in metrics.items():
                if key.startswith(_AVERAGED_METRIC_PREFIXES):
                    _report_avg_totals[key] = _report_avg_totals.get(key, 0.0) + float(val) * rows
                elif key.startswith(_RAW_SUM_METRIC_PREFIXES):
                    _report_raw_totals[key] = _report_raw_totals.get(key, 0.0) + float(val)

            if opt_steps % report_every == 0:
                rd = max(_report_rows, 1)
                step_metrics = {"loss": _report_loss / rd, "n_rows": float(_report_rows), "opt_step": opt_steps}
                for key, total in _report_avg_totals.items():
                    step_metrics[key] = total / rd
                for key, total in _report_raw_totals.items():
                    step_metrics[key] = total
                step_callback(step_metrics)
                _report_rows = 0
                _report_loss = 0.0
                _report_avg_totals.clear()
                _report_raw_totals.clear()

    def _apply_prepared_batch(prepared: _PreparedChunkBatch) -> None:
        plan = prepared.plan
        hidden_batch = _hidden_batch_for_plan(plan)
        if training:
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

    _fill_active()
    current_specs = _current_specs()
    current_plan = _make_plan_from_specs(current_specs, accum_count)
    if current_plan is None:
        return _empty

    if use_prefetch:
        with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
            current_slot = 0
            current_future: Future[_PreparedChunkBatch] | None = prefetch_executor.submit(
                _prepare_prefetched_batch,
                current_slot,
                current_plan,
            )

            while active and current_plan is not None and current_future is not None:
                prepared = current_future.result()
                projected_specs = _project_specs_after_plan(prepared.plan)
                projected_accum = accum_count + len(prepared.plan.cursors)
                if training and projected_accum >= accum_target:
                    projected_accum = 0
                projected_next_episode = next_episode
                while len(projected_specs) < microbatch_target and projected_next_episode < len(ordered_cursors):
                    cursor = ordered_cursors[projected_next_episode]
                    projected_specs.append((cursor, cursor.start))
                    projected_next_episode += 1
                next_plan = _make_plan_from_specs(projected_specs, projected_accum)
                next_slot = 1 - current_slot
                next_future: Future[_PreparedChunkBatch] | None = None
                if next_plan is not None:
                    next_future = prefetch_executor.submit(
                        _prepare_prefetched_batch,
                        next_slot,
                        next_plan,
                    )

                _apply_prepared_batch(prepared)
                active = [cursor for cursor in active if _remaining(cursor) > 0]
                _fill_active()
                current_plan = next_plan
                current_future = next_future
                current_slot = next_slot
    else:
        while active and current_plan is not None:
            prepared = _prepare_prefetched_batch(0, current_plan)
            _apply_prepared_batch(prepared)
            active = [cursor for cursor in active if _remaining(cursor) > 0]
            _fill_active()
            current_plan = _make_plan_from_specs(_current_specs(), accum_count)

    if training and accum_count > 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
        model.bc_step()

    denom = max(total_rows, 1)
    result: Dict[str, float] = {
        "loss": total_loss / denom,
        "accuracy": total_accuracy / denom,
        "n_rows": float(total_rows),
    }
    for key, total in raw_metric_totals.items():
        result[key] = total
    for key, total in averaged_metric_totals.items():
        result[key] = total / denom
    return result


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
    pin_memory: bool = True,
    prefetch: bool = False,
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
        pin_memory=pin_memory,
        prefetch=prefetch,
    )
