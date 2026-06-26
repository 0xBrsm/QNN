"""Bench side-channel scopes — forward-scoped contexts entered by the trainer.

These carry per-batch signals that bench probes need but that are NOT part of
the canonical model's input: the prev-frame look vector, the engagement-EMA
scalar, and privileged target supervision. They are derived from the label
dict (``actions``), so only the trainer (``QNNPolicy.train_supervised`` /
``evaluate_supervised``) can build them — ``Network.forward`` never sees
``actions``.

This module owns that logic so it lives in ``bench``, not in the canonical
``qnn.model.policy``: the policy exposes one neutral ``side_channel_provider``
hook and the bench runner wires :func:`bench_side_channel_scope` into it. The
canonical model passes no provider and pays nothing.

The obs-derived accessor scope is entered separately, at the ``Network`` layer
where obs is already tensorized/flattened — see
``qnn.model.bench.inputs.obs_network.BenchObsNetwork``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any, ContextManager

import torch


def _engagement_ema_scope(actions: Any) -> ContextManager:
    """Enter an EngagementEMAContext if ``actions`` carries ``engagement_ema``."""
    if not isinstance(actions, Mapping) or "engagement_ema" not in actions:
        return contextlib.nullcontext()
    from qnn.model.bench.inputs.engagement_ema_context import (
        EngagementEMAContext, engagement_ema_context,
    )
    ee = actions["engagement_ema"]
    if not isinstance(ee, torch.Tensor):
        ee = torch.as_tensor(ee)
    # Flatten leading dims to match the head's batch axis (B* = batch · time).
    return engagement_ema_context(EngagementEMAContext(engagement_ema=ee.reshape(-1)))


def _target_supervision_scope(actions: Any, masks: Any) -> ContextManager:
    """Enter a TargetSupervisionContext carrying ``target_gt`` /
    ``target_probs_idx`` / ``prev_target_probs`` for bench pointer variants.

    No-op when ``actions`` carries neither ``target`` nor ``target_probs``;
    the canonical MLP pointer never reads from this context.
    """
    if not isinstance(actions, Mapping):
        return contextlib.nullcontext()
    has_target_gt = "target" in actions
    has_target_probs = "target_probs" in actions
    if not (has_target_gt or has_target_probs):
        return contextlib.nullcontext()
    from qnn.model.bench.inputs.target_supervision_context import (
        TargetSupervisionContext, target_supervision_context,
    )
    target_gt_flat: torch.Tensor | None = None
    if has_target_gt:
        tg = actions["target"]
        if not isinstance(tg, torch.Tensor):
            tg = torch.as_tensor(tg)
        target_gt_flat = tg.reshape(-1).long()
    target_probs_idx_flat: torch.Tensor | None = None
    prev_target_probs_flat: torch.Tensor | None = None
    if has_target_probs:
        td = actions["target_probs"]
        if not isinstance(td, torch.Tensor):
            td = torch.as_tensor(td, dtype=torch.float32)
        else:
            td = td.float()
        present = (1.0 - td[..., 0]).clamp(min=1e-6)
        idx_dist = td[..., 1:] / present.unsqueeze(-1)
        if td.ndim == 3:
            prev = torch.zeros_like(idx_dist)
            prev[1:] = idx_dist[:-1]
            if isinstance(masks, Mapping) and "reset_mask" in masks:
                rm = masks["reset_mask"]
                if not isinstance(rm, torch.Tensor):
                    rm = torch.as_tensor(rm)
                rm = rm.bool()
                if rm.ndim == 2:
                    prev = prev.masked_fill(rm.unsqueeze(-1), 0.0)
            prev_target_probs_flat = prev.reshape(-1, prev.shape[-1])
        target_probs_idx_flat = idx_dist.reshape(-1, idx_dist.shape[-1])
    return target_supervision_context(TargetSupervisionContext(
        target_gt=target_gt_flat,
        target_probs_idx=target_probs_idx_flat,
        prev_target_probs=prev_target_probs_flat,
    ))


def bench_side_channel_scope(actions: Any, masks: Any) -> ContextManager:
    """One context manager entering every bench label-derived side channel.

    Each sub-scope is a no-op when its column is absent, so this is safe to
    enter for any bench probe. Passed to ``QNNPolicy`` as the
    ``side_channel_provider``; the canonical model passes ``None``.
    """
    stack = contextlib.ExitStack()
    stack.enter_context(_engagement_ema_scope(actions))
    stack.enter_context(_target_supervision_scope(actions, masks))
    stack.enter_context(_weapon_switch_scope(actions, masks))
    return stack


def _weapon_switch_scope(actions: Any, masks: Any) -> ContextManager:
    """Enter the weapon-switch supervision context for the weapon_switch head.

    Prefers PRECOMPUTED per-frame label columns (``weapon_dwell`` /
    ``weapon_switch`` / ``weapon_newtgt``, generated per-episode at data-prep time
    by ``scripts/analysis/_gen_weapon_switch_labels.py``). These ride along through
    frame_shuffled batches, which is the ONLY correct source in the non-temporal
    bench path: that path has no temporal neighbours at train time, so batch-time
    derivation produces garbage (see src/docs/weapon-head.md). Falls back to
    on-the-fly derivation along the time axis when a temporal ``(T,B)`` batch +
    ``reset_mask`` is available. No-op when ``weapon`` is absent."""
    if not isinstance(actions, Mapping) or "weapon" not in actions:
        return contextlib.nullcontext()
    from qnn.model.bench.inputs.weapon_switch_context import (
        WeaponSwitchContext, derive_weapon_switch_labels, weapon_switch_context,
    )

    def _t(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    pre = ("weapon_dwell", "weapon_switch", "weapon_newtgt")
    if all(k in actions for k in pre):
        w = _t(actions["weapon"]).reshape(-1).long()
        dwell = _t(actions["weapon_dwell"]).reshape(-1).float()
        sw = _t(actions["weapon_switch"]).reshape(-1).float()
        nt = _t(actions["weapon_newtgt"]).reshape(-1).long()
        valid = torch.ones_like(sw, dtype=torch.bool)   # last-of-episode contamination ~0.2%, immaterial
        return weapon_switch_context(WeaponSwitchContext(
            dwell_age=dwell, held_weapon=w, switch_next=sw,
            new_weapon_target=nt, valid=valid,
        ))

    # Fallback: derive along the time axis of a temporal (T,B) batch.
    w = _t(actions["weapon"])
    rm = None
    if isinstance(masks, Mapping) and "reset_mask" in masks:
        rm = _t(masks["reset_mask"])
    return weapon_switch_context(derive_weapon_switch_labels(w, rm))
