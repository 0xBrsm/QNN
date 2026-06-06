"""TargetPointer — supervised pointer head over entity tokens.

Single attention mechanism that serves two consumers:

  * ``target_logits`` — per-idx attention scores (pre-softmax), with invalid
    indices masked to a large negative value.  Supervised directly by BC labels
    (see ``qnn.bc.target_labeler``) via cross-entropy with ignore_index=-100.
    This is the "pointer" view over actor indices.

  * ``target_feat`` — pooled entity feature handed to downstream action heads
    so they can condition on the selected opponent.  Two modes:

      - **soft** (default): ``sum_i softmax(logits)[i] * entity_out[i]``.
        A weighted blend over the full entity set.
      - **hard** (``hard_target=True``): the entity vector at one chosen idx.
        Idx comes from ``target_gt`` (teacher forcing) when supplied with a
        valid index, otherwise ``argmax(logits)``. Decouples target-head loss
        tuning from motor-head training distribution so target weighting
        (focal/class) doesn't reshape what the motor heads see frame-to-frame.
        Motor gradient does NOT reach the pointer in this mode.
      - **gt-dist STE** (``gt_dist_target_feat=True``, training only): forward
        pools by the labeler's GT idx distribution; backward routes through
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
(cls_readout), but set it to the GRU hidden width when the caller wants the
GRU output as the query (temporal commitment / hysteresis path).

If ``inject_weapon`` is set, an additive weapon-id embedding shifts the query
post-projection so target attention can condition on the currently held
weapon (e.g. RL pulls toward distant enemies, shotgun toward close ones).
The shift lives in the query/key dot-product space and stays out of the
weapon head's input — so it can't poison weapon-head training.

If ``linear_idx_prior`` is set, a logit prior linear in idx index is added
before masking: ``prior[k] = -alpha * k`` where ``alpha`` is a learnable
scalar.  Encodes the inductive bias that the idx ordering policy
(pool→recency→team→threat) already puts the most likely target at idx 0 by
construction; the residual attention scores only need to push idx k > idx 0
when there's evidence to override the ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class TargetPointerInput:
    query: torch.Tensor                              # (B*, query_in_dim)
    entity_outs: torch.Tensor                        # (B*, N, D)
    entity_mask: torch.Tensor                        # (B*, N) bool
    self_weapon_idx: torch.Tensor | None = None      # (B*,) for inject_weapon
    target_gt: torch.Tensor | None = None            # (B*,) for hard_target with GT
    target_probs_idx: torch.Tensor | None = None     # (B*, N) for gt_dist_target_feat
    prev_target_probs: torch.Tensor | None = None    # (B*, N) for prev_target_in_query


@dataclass(frozen=True, slots=True)
class TargetPointerOutput:
    target_logits: torch.Tensor  # (B*, N)
    target_feat: torch.Tensor    # (B*, D)
    target_query: torch.Tensor   # (B*, D)


class TargetPointer(nn.Module):
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
                "hard_target_feat and gt_dist_target_feat are mutually exclusive"
            )
        # When prev_target_in_query is True, query_proj sees
        # cat(query_input, prev_target_probs) as input. The caller passes
        # the previous frame's renormalized GT idx distribution at train
        # time and None (→ zeros) at eval time; the train/eval forward
        # signal differs by design — this is a privileged-input probe
        # like gt_dist_target_feat. See target docstring.
        proj_in_dim = self.query_in_dim + (self.prev_target_n_indices if self.prev_target_in_query else 0)
        self.query_proj = nn.Linear(proj_in_dim, self.d_model)
        if self.inject_weapon:
            self.weapon_query_embed = nn.Embedding(int(weapon_vocab), self.d_model)
            nn.init.normal_(self.weapon_query_embed.weight, std=0.02)
        if self.linear_idx_prior:
            # Learnable scalar slope; offsets (-0, -1, ..., -N+1) computed on the
            # fly per forward to avoid baking in a max-idx constant.
            self.idx_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        """Return (target_logits, target_feat, target_query).

        ``target_query`` is exposed so callers can apply auxiliary supervision
        binding the prediction to a stable identity (e.g. ``target_pid``'s
        embedding) — idx labels alone don't push the model to encode
        identity in the query vector.
        """
        query_input = inp.query
        if self.prev_target_in_query:
            n = self.prev_target_n_indices
            prev = inp.prev_target_probs
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
        query = self.query_proj(query_input)                                  # (B*, D)
        if self.inject_weapon:
            assert inp.self_weapon_idx is not None, "inject_weapon=True requires self_weapon_idx"
            query = query + self.weapon_query_embed(inp.self_weapon_idx.long())
        logits = (inp.entity_outs * query.unsqueeze(1)).sum(dim=-1)           # (B*, N)

        if self.linear_idx_prior:
            n_indices = inp.entity_outs.shape[1]
            offsets = -torch.arange(n_indices, dtype=logits.dtype, device=logits.device)
            logits = logits + self.idx_prior_scale * offsets

        mask_f = inp.entity_mask.to(logits.dtype)
        logits = logits.masked_fill(mask_f == 0, -1e9)

        # Zero target_feat for empty scenes so a uniform softmax / argmax over
        # padding doesn't leak zero-padded entity features into downstream heads.
        has_any = (mask_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)

        if self.hard_target:
            n_indices = inp.entity_outs.shape[1]
            pred_idx = torch.argmax(logits, dim=-1)                           # (B*,)
            if inp.target_gt is not None:
                gt = inp.target_gt.long()
                use_gt = (gt >= 0) & (gt < n_indices)
                idx = torch.where(use_gt, gt, pred_idx)
            else:
                idx = pred_idx
            batch_idx = torch.arange(inp.entity_outs.shape[0], device=inp.entity_outs.device)
            target_feat = inp.entity_outs[batch_idx, idx] * has_any
        elif self.gt_dist_target_feat and self.training and inp.target_probs_idx is not None:
            # STE with the labeler's GT distribution: forward uses the GT idx
            # mass to give motors a clean target_feat blend, backward routes
            # through softmax(logits) so motor gradient still trains the
            # pointer.  Gated on self.training so val/eval/PPO take the soft
            # branch — that keeps the generalization metric honest (no
            # privileged input at eval time).
            soft = F.softmax(logits, dim=-1)
            choice = inp.target_probs_idx + (soft - soft.detach())
            target_feat = (choice.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any
        else:
            probs = F.softmax(logits, dim=-1)                                 # (B*, N)
            target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        return TargetPointerOutput(
            target_logits=logits, target_feat=target_feat, target_query=query,
        )
