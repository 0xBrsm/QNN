"""CanonicalTargetPointer — the legacy attention-style target head.

This is the pre-MLP target pointer, preserved here for bench
experimentation. The production model now uses the MLP pointer in
``qnn.model.target.TargetPointer``; this module keeps the older
behavior (query · entity_out dot product, optional weapon-id query
shift, hard-argmax / gt-dist / prev-target probes, learnable idx prior)
available for ablations and head probes.

Inputs flow:

* ``entity_outs`` / ``entity_mask`` / ``self_readout`` arrive on every
  forward via the standard :class:`TargetPointerInput`. ``self_readout``
  is the default query source; bench wrappers driving a temporal stack
  override it by calling ``stash(query=gru_flat)``.

* ``self_weapon_idx`` (for ``inject_weapon=True``) is stashed by the
  bench Network wrapper via ``stash(self_weapon_idx=...)``.

* Privileged target supervision (``target_gt``, ``target_probs_idx``,
  ``prev_target_probs``) is read from
  :func:`current_target_supervision_context` — the BC supervised loop
  enters that contextvar before model forward when ``actions`` carries
  the required labels.

The canonical pointer does NOT apply ``inp.enemy_mask`` — the historical
behavior is full ``entity_mask`` only, so the loss is responsible for
steering mass to enemy indices. Variants that want an enemy-only mask
(``EnemyMaskedTargetPointer``) post-mask the logits after this module
returns.

The legacy ``target_query`` output slot is gone — the new
:class:`TargetPointerOutput` exposes ``(target_logits, target_feat)``
only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qnn.model.bench.inputs.target_supervision_context import (
    current_target_supervision_context,
)
from qnn.model.target import TargetPointerInput, TargetPointerOutput


class CanonicalTargetPointer(nn.Module):
    """Legacy attention-style pointer (query · entity_out).

    All experimental modes from the pre-MLP era — hard-argmax pooling,
    gt-dist STE, prev-target query concat, weapon-id query shift,
    learnable idx prior — live here. The production model has none of
    these knobs.
    """

    def __init__(
        self,
        *,
        d_model: int,
        query_in_dim: int,
        inject_weapon: bool,
        weapon_vocab: int,
        hard_target: bool,
        linear_idx_prior: bool,
        gt_dist_target_feat: bool,
        prev_target_in_query: bool,
        prev_target_n_indices: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.query_in_dim = int(query_in_dim)
        self.inject_weapon = bool(inject_weapon)
        self.hard_target = bool(hard_target)
        self.linear_idx_prior = bool(linear_idx_prior)
        self.gt_dist_target_feat = bool(gt_dist_target_feat)
        self.prev_target_in_query = bool(prev_target_in_query)
        self.prev_target_n_indices = int(prev_target_n_indices)
        if self.hard_target and self.gt_dist_target_feat:
            raise ValueError(
                "hard_target and gt_dist_target_feat are mutually exclusive"
            )
        proj_in_dim = self.query_in_dim + (
            self.prev_target_n_indices if self.prev_target_in_query else 0
        )
        self.query_proj = nn.Linear(proj_in_dim, self.d_model)
        if self.inject_weapon:
            self.weapon_query_embed = nn.Embedding(int(weapon_vocab), self.d_model)
            nn.init.normal_(self.weapon_query_embed.weight, std=0.02)
        if self.linear_idx_prior:
            self.idx_prior_scale = nn.Parameter(torch.tensor(1.0))

        self._stashed_query: torch.Tensor | None = None
        self._stashed_self_weapon_idx: torch.Tensor | None = None

    def stash(
        self,
        *,
        query: torch.Tensor | None = None,
        self_weapon_idx: torch.Tensor | None = None,
    ) -> None:
        """Stash per-forward inputs the bench wrapper must compute outside
        the pointer (alternative query source, weapon index). Privileged
        target supervision flows through the contextvar instead."""
        self._stashed_query = query
        self._stashed_self_weapon_idx = self_weapon_idx

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        supervision = current_target_supervision_context()
        target_gt = supervision.target_gt if supervision is not None else None
        target_probs_idx = supervision.target_probs_idx if supervision is not None else None
        prev_target_probs = supervision.prev_target_probs if supervision is not None else None

        query_input = (
            self._stashed_query if self._stashed_query is not None else inp.self_readout
        )
        if self.prev_target_in_query:
            n = self.prev_target_n_indices
            prev = prev_target_probs
            if prev is None:
                prev = torch.zeros(
                    query_input.shape[0], n,
                    dtype=query_input.dtype, device=query_input.device,
                )
            elif prev.shape[-1] != n:
                raise ValueError(
                    f"prev_target_probs last dim {prev.shape[-1]} != {n}"
                )
            query_input = torch.cat([query_input, prev], dim=-1)
        query = self.query_proj(query_input)
        if self.inject_weapon:
            if self._stashed_self_weapon_idx is None:
                raise RuntimeError(
                    "CanonicalTargetPointer(inject_weapon=True) requires "
                    "stash(self_weapon_idx=...) before forward."
                )
            query = query + self.weapon_query_embed(self._stashed_self_weapon_idx.long())
        logits = (inp.entity_outs * query.unsqueeze(1)).sum(dim=-1)

        if self.linear_idx_prior:
            n_indices = inp.entity_outs.shape[1]
            offsets = -torch.arange(n_indices, dtype=logits.dtype, device=logits.device)
            logits = logits + self.idx_prior_scale * offsets

        mask_f = inp.entity_mask.to(logits.dtype)
        logits = logits.masked_fill(mask_f == 0, -1e9)

        has_any = (mask_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)

        if self.hard_target:
            n_indices = inp.entity_outs.shape[1]
            pred_idx = torch.argmax(logits, dim=-1)
            if target_gt is not None:
                gt = target_gt.long()
                use_gt = (gt >= 0) & (gt < n_indices)
                idx = torch.where(use_gt, gt, pred_idx)
            else:
                idx = pred_idx
            batch_idx = torch.arange(inp.entity_outs.shape[0], device=inp.entity_outs.device)
            target_feat = inp.entity_outs[batch_idx, idx] * has_any
        elif self.gt_dist_target_feat and self.training and target_probs_idx is not None:
            soft = F.softmax(logits, dim=-1)
            choice = target_probs_idx + (soft - soft.detach())
            target_feat = (choice.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any
        else:
            probs = F.softmax(logits, dim=-1)
            target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        return TargetPointerOutput(target_logits=logits, target_feat=target_feat)
