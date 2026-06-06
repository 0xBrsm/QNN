"""Look head: target-anchored prior + learned residual.

  base_look  = normalize(Σ_n softmax(target_logits)[n] * entity_rel[n])
  delta_look = mlp(features)
  pred_look  = normalize(base_look + delta_look)

base_look is zero (and pred_look defaults to delta_look's normalized
output) when no actor entity is present. base_look[..., 0] is the cosine
of the current aim against the soft target direction in view frame —
AttackHead consumes it as the alignment prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp

OUT_DIM = 3  # 3D direction vector


@dataclass(frozen=True, slots=True)
class LookHeadInput:
    features: torch.Tensor       # (B*, in_dim)
    target_logits: torch.Tensor  # (B*, N) — pre-masked by TargetPointer
    entity_rel: torch.Tensor     # (B*, N, 3) — relative XYZ from entity_scalars_raw
    actor_mask: torch.Tensor     # (B*, N) bool


@dataclass(frozen=True, slots=True)
class LookHeadOutput:
    pred_look: torch.Tensor   # (B*, 3) unit-normalized
    base_look: torch.Tensor   # (B*, 3) unit-normalized prior
    delta_look: torch.Tensor  # (B*, 3) raw residual


class LookHead(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        # Soft-attended target relative position; target_logits already has
        # -1e9 on non-actor indices so softmax is implicitly actor-only.
        probs = F.softmax(inp.target_logits, dim=-1)                            # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * inp.entity_rel).sum(dim=-2)    # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor

        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        base_look = soft_target_rel / soft_norm

        delta_look = self.mlp(inp.features)                                     # (B*, 3)
        unnormalized = base_look + delta_look
        out_norm = torch.linalg.vector_norm(unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)
        pred_look = unnormalized / out_norm
        return LookHeadOutput(pred_look=pred_look, base_look=base_look, delta_look=delta_look)
