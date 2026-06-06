"""TargetPointer — per-entity MLP scoring head over actor tokens.

Single supervised pointer that serves two downstream consumers:

* ``target_logits`` — per-idx pre-softmax scores, with non-enemy and
  padding indices masked to ``-1e9``. Supervised directly by BC target
  labels (see ``qnn.bc.target_labeler``) via cross-entropy on the
  renormalized GT idx distribution.

* ``target_feat`` — softmax-weighted blend of ``entity_outs`` using the
  same logits. Handed to the motor / weapon / attack heads so they can
  condition on the chosen opponent.

Architecture: two-layer MLP applied independently to each entity vector::

    logits_i = w2 · gelu(W1 · entity_outs_i + b1) + b2

``d_target`` is the MLP hidden width — the sole architectural knob. At
``d_target=1`` with the activation no-op'd the head degenerates to a
single linear scoring direction (the constant-query baseline); larger
values give per-entity non-linear scoring capacity. Activation is fixed
to GELU; alternative scoring shapes (cls/GRU query dot product,
weapon-conditioned query, oracle GT pool, hard-argmax pooling) live in
``qnn.model.bench`` for ablation and are not exposed to the canonical
training pipeline.

Target is not a sampled action — the engine doesn't consume a target
label. ``target_logits`` only receives supervised gradient from labels;
``target_feat`` is what propagates to motor heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class TargetPointerInput:
    entity_outs: torch.Tensor       # (B*, N, D)
    entity_mask: torch.Tensor       # (B*, N) bool — encoder valid-token mask
    enemy_mask: torch.Tensor        # (B*, N) bool — actor-and-not-teammate
    self_readout: torch.Tensor      # (B*, D) — cls/self readout; unused by
                                    # the canonical MLP, surfaced for bench
                                    # pointers that want a query source.


@dataclass(frozen=True, slots=True)
class TargetPointerOutput:
    target_logits: torch.Tensor     # (B*, N)
    target_feat: torch.Tensor       # (B*, D)


class TargetPointer(nn.Module):
    def __init__(self, *, d_model: int, d_target: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_target = int(d_target)
        self.score = nn.Sequential(
            nn.Linear(self.d_model, self.d_target),
            nn.GELU(),
            nn.Linear(self.d_target, 1),
        )

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        logits = self.score(inp.entity_outs).squeeze(-1)                # (B*, N)

        valid = inp.enemy_mask & inp.entity_mask
        valid_f = valid.to(logits.dtype)
        logits = logits.masked_fill(valid_f == 0, -1e9)

        # Zero target_feat for empty (no-enemy) scenes so a uniform softmax
        # over masked-out indices doesn't leak through downstream.
        has_any = (valid_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)
        probs = F.softmax(logits, dim=-1)
        target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        return TargetPointerOutput(target_logits=logits, target_feat=target_feat)
