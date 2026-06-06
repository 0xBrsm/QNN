"""Oracle target pointer: uses the labeler's GT distribution directly.

Drop-in for ``TargetPointer`` in Network's target_pointer slot. Instead
of learning attention from a query, this pointer reads ``target_probs_idx``
(the labeler's renormalized GT idx distribution, shape ``(B*, N)``) and
soft-pools the entity tokens by it.

Use for ablations that isolate downstream head capacity from pointer
error — the canonical model's ``gt_dist_target_feat=True`` mode is the
training-time STE version of this idea, gated on ``self.training``; this
component is the unconditional oracle form, suitable for whole-network
ablations like the pre-attn head probes.

Contract matches ``TargetPointer``:
  Input:  ``TargetPointerInput`` — only ``entity_outs``, ``entity_mask``,
          and ``target_probs_idx`` are read. ``query`` and the other
          optional fields are accepted but ignored.
  Output: ``TargetPointerOutput`` — ``target_logits`` is the log-prob of
          the GT distribution (so downstream consumers that softmax it
          recover the GT mass), ``target_feat`` is the GT-pooled feature,
          ``target_query`` is zeros (no learned query). For an aux
          identity-supervision signal the canonical pointer is required.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.model.target import TargetPointerInput, TargetPointerOutput


class GTTargetPointer(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # No learned parameters — included as a buffer so Network's xavier
        # init and named_parameters() don't see anything to update.

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        if inp.target_probs_idx is None:
            raise RuntimeError(
                "GTTargetPointer requires target_probs_idx — the BC supervised "
                "loop passes it; eval/PPO paths would need to be adapted "
                "before using this pointer in those contexts."
            )
        mask_f = inp.entity_mask.to(inp.entity_outs.dtype)
        has_any = (mask_f.sum(dim=-1, keepdim=True) > 0).to(inp.entity_outs.dtype)
        weights = inp.target_probs_idx.to(inp.entity_outs.dtype) * mask_f
        target_feat = (weights.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        # target_logits: log-prob recovers the GT distribution under softmax.
        # eps-clamp to keep it finite at 0.
        log_probs = torch.log(weights.clamp(min=1e-9))
        target_query = torch.zeros(
            (inp.entity_outs.shape[0], self.d_model),
            dtype=inp.entity_outs.dtype, device=inp.entity_outs.device,
        )
        return TargetPointerOutput(
            target_logits=log_probs,
            target_feat=target_feat,
            target_query=target_query,
        )
