"""TargetPointer — supervised pointer head over entity tokens.

Single attention mechanism that serves two consumers:

  * ``target_logits`` — per-slot attention scores (pre-softmax), with invalid
    slots masked to a large negative value.  Supervised directly by BC labels
    (see ``qnn.bc.target_labeler``) via cross-entropy with ignore_index=-100.
    This is the "pointer" view over actor slots.

  * ``target_feat`` — pooled entity feature handed to downstream action heads
    so they can condition on the selected opponent.  Two modes:

      - **soft** (default): ``sum_i softmax(logits)[i] * entity_out[i]``.
        A weighted blend over the full entity set.
      - **hard** (``hard_target=True``): the entity vector at one chosen slot.
        Slot comes from ``target_gt`` (teacher forcing) when supplied with a
        valid index, otherwise ``argmax(logits)``. Decouples target-head loss
        tuning from motor-head training distribution so target weighting
        (focal/class) doesn't reshape what the motor heads see frame-to-frame.
        Motor gradient does NOT reach the pointer in this mode.
      - **gt-dist STE** (``gt_dist_target_feat=True``, training only): forward
        pools by the labeler's GT slot distribution; backward routes through
        ``softmax(logits)`` so motor gradient still trains the pointer.
        Gated on ``self.training`` — eval/PPO falls back to soft so the
        generalization metric isn't biased by privileged-input forward.
        Mutually exclusive with ``hard_target``.

Both outputs share one ``query_proj`` Linear and one attention pass; they
differ only in what the consumer reads.

Target is not an action — the engine doesn't consume a target label.  It is
an intermediary used to produce ``target_feat`` and to receive supervised
gradient from labels.  ``target_logits`` is never sampled as a PPO action.

The query source is configurable via ``query_in_dim``: defaults to ``d_model``
(self_readout), but set it to the GRU hidden width when the caller wants the
GRU output as the query (temporal commitment / hysteresis path).

If ``inject_weapon`` is set, an additive weapon-id embedding shifts the query
post-projection so target attention can condition on the currently held
weapon (e.g. RL pulls toward distant enemies, shotgun toward close ones).
The shift lives in the query/key dot-product space and stays out of the
weapon head's input — so it can't poison weapon-head training.

If ``linear_slot_prior`` is set, a logit prior linear in slot index is added
before masking: ``prior[k] = -alpha * k`` where ``alpha`` is a learnable
scalar.  Encodes the inductive bias that the slot ordering policy
(pool→recency→team→threat) already puts the most likely target at slot 0 by
construction; the residual attention scores only need to push slot k > slot 0
when there's evidence to override the ordering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetPointer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        query_in_dim: int,
        inject_weapon: bool,
        weapon_vocab: int,
        hard_target: bool,
        linear_slot_prior: bool,
        gt_dist_target_feat: bool,
        prev_target_in_query: bool,
        prev_target_n_slots: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.query_in_dim = int(query_in_dim)
        self.inject_weapon = bool(inject_weapon)
        self.hard_target = bool(hard_target)
        self.linear_slot_prior = bool(linear_slot_prior)
        self.gt_dist_target_feat = bool(gt_dist_target_feat)
        self.prev_target_in_query = bool(prev_target_in_query)
        self.prev_target_n_slots = int(prev_target_n_slots)
        if self.hard_target and self.gt_dist_target_feat:
            raise ValueError(
                "hard_target_feat and gt_dist_target_feat are mutually exclusive"
            )
        # When prev_target_in_query is True, query_proj sees
        # cat(query_input, prev_target_dist) as input. The caller passes
        # the previous frame's renormalized GT slot distribution at train
        # time and None (→ zeros) at eval time; the train/eval forward
        # signal differs by design — this is a privileged-input probe
        # like gt_dist_target_feat. See target docstring.
        proj_in_dim = self.query_in_dim + (self.prev_target_n_slots if self.prev_target_in_query else 0)
        self.query_proj = nn.Linear(proj_in_dim, self.d_model)
        if self.inject_weapon:
            self.weapon_query_embed = nn.Embedding(int(weapon_vocab), self.d_model)
            nn.init.normal_(self.weapon_query_embed.weight, std=0.02)
        if self.linear_slot_prior:
            # Learnable scalar slope; offsets (-0, -1, ..., -N+1) computed on the
            # fly per forward to avoid baking in a max-slot constant.
            self.slot_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        query_input: torch.Tensor,         # (B, Q)
        entity_outs: torch.Tensor,         # (B, N, D)
        entity_mask: torch.Tensor,         # (B, N) bool or 0/1
        *,
        self_weapon_slot: torch.Tensor | None = None,  # (B,) long, [0, weapon_vocab)
        target_gt: torch.Tensor | None = None,         # (B,) long, -100 = ignore
        target_dist_slot: torch.Tensor | None = None,  # (B, N) slot probs, sums to 1
        prev_target_dist: torch.Tensor | None = None,  # (B, N) prev-frame slot probs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (target_logits, target_feat, query).

        The ``query`` is exposed so callers can apply auxiliary supervision
        binding the prediction to a stable identity (e.g. ``target_pid``'s
        embedding) — slot labels alone don't push the model to encode
        identity in the query vector.
        """
        if self.prev_target_in_query:
            n = self.prev_target_n_slots
            if prev_target_dist is None:
                prev_target_dist = torch.zeros(
                    query_input.shape[0], n,
                    dtype=query_input.dtype, device=query_input.device,
                )
            elif prev_target_dist.shape[-1] != n:
                raise ValueError(
                    f"prev_target_dist last dim {prev_target_dist.shape[-1]} != {n}"
                )
            query_input = torch.cat([query_input, prev_target_dist], dim=-1)
        query = self.query_proj(query_input)                            # (B, D)
        if self.inject_weapon:
            assert self_weapon_slot is not None, "inject_weapon=True requires self_weapon_slot"
            query = query + self.weapon_query_embed(self_weapon_slot.long())
        logits = (entity_outs * query.unsqueeze(1)).sum(dim=-1)        # (B, N)

        if self.linear_slot_prior:
            n_slots = entity_outs.shape[1]
            offsets = -torch.arange(n_slots, dtype=logits.dtype, device=logits.device)
            logits = logits + self.slot_prior_scale * offsets

        mask_f = entity_mask.to(logits.dtype)
        logits = logits.masked_fill(mask_f == 0, -1e9)

        # Zero target_feat for empty scenes so a uniform softmax / argmax over
        # padding doesn't leak zero-padded entity features into downstream heads.
        has_any = (mask_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)

        if self.hard_target:
            n_slots = entity_outs.shape[1]
            pred_slot = torch.argmax(logits, dim=-1)                    # (B,)
            if target_gt is not None:
                gt = target_gt.long()
                use_gt = (gt >= 0) & (gt < n_slots)
                slot = torch.where(use_gt, gt, pred_slot)
            else:
                slot = pred_slot
            batch_idx = torch.arange(entity_outs.shape[0], device=entity_outs.device)
            target_feat = entity_outs[batch_idx, slot] * has_any
        elif self.gt_dist_target_feat and self.training and target_dist_slot is not None:
            # STE with the labeler's GT distribution: forward uses the GT slot
            # mass to give motors a clean target_feat blend, backward routes
            # through softmax(logits) so motor gradient still trains the
            # pointer.  Gated on self.training so val/eval/PPO take the soft
            # branch — that keeps the generalization metric honest (no
            # privileged input at eval time).
            soft = F.softmax(logits, dim=-1)
            choice = target_dist_slot + (soft - soft.detach())
            target_feat = (choice.unsqueeze(-1) * entity_outs).sum(dim=1) * has_any
        else:
            probs = F.softmax(logits, dim=-1)                           # (B, N)
            target_feat = (probs.unsqueeze(-1) * entity_outs).sum(dim=1) * has_any

        return logits, target_feat, query
