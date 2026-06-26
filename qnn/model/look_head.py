"""Look head: target-anchored prior + learned residual.

  look_prior  = normalize(Σ_n softmax(target_logits)[n] * entity_rel[n])
  look_delta = mlp(features)
  look_predict  = normalize(look_prior + look_delta)

look_prior is zero (and look_predict defaults to look_delta's normalized
output) when no actor entity is present. look_prior[..., 0] is the cosine
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
    look_predict: torch.Tensor   # (B*, 3) unit-normalized
    look_prior: torch.Tensor   # (B*, 3) unit-normalized prior
    look_delta: torch.Tensor  # (B*, 3) raw residual
    # Binned (classification) look heads emit per-axis tangent bin logits
    # (B*, 2, N_BINS); the canonical look loss then uses cross-entropy instead
    # of smooth_l1. None for regression heads. See qnn.model.look_bins.
    look_bins: torch.Tensor | None = None
    # Polar (magnitude × direction) look heads emit a categorical over
    # {hold} ∪ magnitude bins (B*, N_MAG+1) and over direction bins (B*, N_DIR).
    # The hold mode is a single bin and a flick's yaw/pitch share one direction.
    # None for non-polar heads. See qnn.model.look_bins.
    look_mag_logits: torch.Tensor | None = None
    look_dir_logits: torch.Tensor | None = None


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
        look_prior = soft_target_rel / soft_norm

        look_delta = self.mlp(inp.features)                                     # (B*, 3)
        unnormalized = look_prior + look_delta
        out_norm = torch.linalg.vector_norm(unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)
        look_predict = unnormalized / out_norm
        return LookHeadOutput(look_predict=look_predict, look_prior=look_prior, look_delta=look_delta)
