"""AttackHead variant with hit-test physics prior + residual MLP.

  prior_logit = logit(Σ_n target_dist[n] * hit[n])
  attack_logit = prior_logit + delta_attack(features)

``hit[n]`` is a per-entity boolean from ``qnn.model.hit_test_torch.hit_test_torch``
(projectile lead solve, gravity ignored — matches the analytical labeler
truth). ``target_dist = softmax(target_logits)`` so the prior is the
attended hit feasibility, converted to logit space.

The residual MLP's final Linear is zero-initialized so training starts
at attack_logit == prior_logit exactly.

Requires ``inp.target_logits``, ``inp.entity_scalars``, ``inp.actor_mask``,
and ``inp.weapon_id``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput

# ACTOR scalar layout (dequant.py:290+): HALFEXT 0:3, REL 3:6, VEL 7:10.
_REL_BEGIN, _REL_END = 3, 6
_VEL_BEGIN, _VEL_END = 7, 10
_HALF_BEGIN, _HALF_END = 0, 3


class HitTestAttackHead(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int, activation: str) -> None:
        super().__init__()
        self.mlp = make_head_mlp(in_dim, OUT_DIM, bottleneck_dim, activation)
        # Zero-init final Linear so attack_logit starts at prior_logit.
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert (
            inp.target_logits is not None
            and inp.entity_scalars is not None
            and inp.actor_mask is not None
            and inp.weapon_id is not None
        ), "HitTestAttackHead requires target_logits, entity_scalars, actor_mask, weapon_id"

        # Lazy import to avoid loading the JIT-compiled physics module unless
        # this variant is actually instantiated.
        from qnn.model.hit_test_torch import hit_test_torch

        rel = inp.entity_scalars[..., _REL_BEGIN:_REL_END]
        vel = inp.entity_scalars[..., _VEL_BEGIN:_VEL_END]
        half = inp.entity_scalars[..., _HALF_BEGIN:_HALF_END]
        with torch.no_grad():
            # Boolean labeler-truth — detached so training only updates the MLP.
            hit = hit_test_torch(
                inp.weapon_id.reshape(-1).long(), rel, vel, half, inp.actor_mask,
            )                                                               # (B*, N) bool
        target_dist = F.softmax(inp.target_logits, dim=-1)                  # (B*, N)
        attended = (target_dist * hit.to(target_dist.dtype)).sum(dim=-1, keepdim=True)
        # logit(p) with eps clamp to keep finite at p ∈ {0, 1}.
        attended_clamped = attended.clamp(min=1e-4, max=1.0 - 1e-4)
        prior_logit = torch.log(attended_clamped / (1.0 - attended_clamped))
        prior_logit = prior_logit.to(inp.features.dtype)
        delta_attack = self.mlp(inp.features)                               # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
