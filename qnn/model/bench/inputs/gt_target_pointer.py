"""Oracle target pointer: pools entity tokens by the labeler's GT distribution.

Drop-in for the production ``TargetPointer`` in Network's
``target_pointer`` slot. Instead of learning attention from a query,
this pointer reads the BC labeler's renormalized GT idx distribution
from :func:`current_target_supervision_context` and soft-pools the
entity tokens by it.

Used for ablations that isolate downstream head capacity from pointer
error (e.g., the pre-attn head probes). ``target_logits`` is the
log-prob of the GT distribution (so downstream consumers that softmax
it recover the GT mass); ``target_feat`` is the GT-pooled feature.

Requires the BC supervised loop to have entered a
:class:`TargetSupervisionContext` carrying ``target_probs_idx`` —
without it, the oracle has no signal and raises on forward.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.model.bench.inputs.target_supervision_context import (
    current_target_supervision_context,
)
from qnn.model.target import TargetPointerInput, TargetPointerOutput


class GTTargetPointer(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # No learned parameters.

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        supervision = current_target_supervision_context()
        target_probs_idx = supervision.target_probs_idx if supervision is not None else None
        if target_probs_idx is None:
            raise RuntimeError(
                "GTTargetPointer requires target_probs_idx via "
                "target_supervision_context — the BC supervised loop must "
                "enter the scope before calling the model."
            )
        mask_f = inp.entity_mask.to(inp.entity_outs.dtype)
        has_any = (mask_f.sum(dim=-1, keepdim=True) > 0).to(inp.entity_outs.dtype)
        weights = target_probs_idx.to(inp.entity_outs.dtype) * mask_f
        target_feat = (weights.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        # target_logits: log-prob recovers the GT distribution under softmax.
        # eps-clamp keeps it finite at 0.
        log_probs = torch.log(weights.clamp(min=1e-9))
        return TargetPointerOutput(target_logits=log_probs, target_feat=target_feat)
