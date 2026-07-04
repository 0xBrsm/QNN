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

``detach_entity_grad=True`` (default) stops gradients from flowing back
through entity tokens into the entity embeddings. For bench probes that
test whether target_feat *carries signal* for a downstream head, this is
almost always correct: the entity embeddings should not receive gradient
from the probe head via the pointer. Without detach, backward through the
weighted pool is 3× more expensive than the forward pass because autograd
must differentiate through the full entity embedding construction.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.model.bench.inputs.target_supervision_context import (
    current_target_supervision_context,
)
from qnn.model.target import TargetPointerInput, TargetPointerOutput


class GTTargetPointer(nn.Module):
    def __init__(self, *, d_model: int, detach_entity_grad: bool = True) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.detach_entity_grad = bool(detach_entity_grad)
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
        entity_outs = inp.entity_outs.detach() if self.detach_entity_grad else inp.entity_outs
        mask_f = inp.entity_mask.to(entity_outs.dtype)
        has_any = (mask_f.sum(dim=-1, keepdim=True) > 0).to(entity_outs.dtype)
        weights = target_probs_idx.to(entity_outs.dtype) * mask_f
        # Fused soft-pool: einsum avoids materializing the (B, N, d_model)
        # product before reducing — one kernel, no large intermediate (matters
        # on ROCm where dispatch/alloc churn dominates this loop).
        target_feat = torch.einsum("bn,bnd->bd", weights, entity_outs) * has_any

        # target_logits: log-prob recovers the GT distribution under softmax.
        # eps-clamp keeps it finite at 0.
        log_probs = torch.log(weights.clamp(min=1e-9))
        return TargetPointerOutput(target_logits=log_probs, target_feat=target_feat)


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_pointer  # noqa: E402


@register_pointer("gt")
def _build_pointer_gt(pointer, d_model):
    return GTTargetPointer(d_model=d_model)
