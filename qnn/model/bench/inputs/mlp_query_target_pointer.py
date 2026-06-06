"""MLPQueryTargetPointer — non-linear per-entity scoring head.

Drop-in for ``TargetPointer`` in Network's ``target_pointer`` slot.
Replaces the constant-query pointer's single linear projection with a
small MLP applied independently to each entity vector::

    logits_i = w2 · act(W1 · entity_outs_i + b1) + b2

Same per-entity symmetry / mask handling / variable cardinality as
``ConstantQueryTargetPointer``; the only change is scoring capacity.
At ``d_target=1`` and a no-op activation this degenerates to the
constant-query baseline, so any improvement here isolates "linear
scoring head was the bottleneck" from "the entity tokens carry
insufficient signal."

Network supplies ``enemy_mask`` directly on :class:`TargetPointerInput`
— this module reads it from the input rather than via stash.

This module is the bench experimental copy that exposes the
``activation`` knob; the production target head
(``qnn.model.target.TargetPointer``) hard-codes GELU and exposes only
``d_target``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model.target import TargetPointerInput, TargetPointerOutput


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "none": nn.Identity,
}


class MLPQueryTargetPointer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_target: int,
        activation: str,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )
        self.d_model = int(d_model)
        self.d_target = int(d_target)
        self.score = nn.Sequential(
            nn.Linear(self.d_model, self.d_target),
            _ACTIVATIONS[activation](),
            nn.Linear(self.d_target, 1),
        )

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        logits = self.score(inp.entity_outs).squeeze(-1)                          # (B*, N)

        valid = inp.enemy_mask & inp.entity_mask
        valid_f = valid.to(logits.dtype)
        logits = logits.masked_fill(valid_f == 0, -1e9)

        has_any = (valid_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)
        probs = F.softmax(logits, dim=-1)
        target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        return TargetPointerOutput(target_logits=logits, target_feat=target_feat)
