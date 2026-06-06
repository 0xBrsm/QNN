"""ConstantQueryTargetPointer — query is a single learned vector.

Drop-in for ``TargetPointer`` in Network's ``target_pointer`` slot. The
"query" is a single ``(d_model,)`` learnable parameter shared across
all frames — equivalent to ``nn.Linear(d_model, 1)`` applied to each
entity vector and broadcast to a per-entity scalar logit.

This is the strict null-baseline for the pointer's query mechanism:
no conditioning on agent state at all (no cls_readout, no weapon spec,
no GRU). Used to answer "is the query doing per-frame work, or is it
just a per-entity scoring head with a learned readout direction?"

Logits get an enemy + entity mask: Network supplies ``enemy_mask`` via
:class:`TargetPointerInput` directly — no stash hook needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model.target import TargetPointerInput, TargetPointerOutput


class ConstantQueryTargetPointer(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # Linear(d_model, 1) = (learned query, learned bias). Same parameter
        # count as a constant query + scalar bias.
        self.score = nn.Linear(self.d_model, 1)

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        logits = self.score(inp.entity_outs).squeeze(-1)                          # (B*, N)

        valid = inp.enemy_mask & inp.entity_mask
        valid_f = valid.to(logits.dtype)
        logits = logits.masked_fill(valid_f == 0, -1e9)

        has_any = (valid_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)
        probs = F.softmax(logits, dim=-1)
        target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any

        return TargetPointerOutput(target_logits=logits, target_feat=target_feat)
