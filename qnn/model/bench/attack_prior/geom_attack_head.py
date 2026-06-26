"""LookStyleAttackHead variant with explicit target-geometry scalars.

OFAT delta vs :class:`EngagedLookStyleAttackHead`: in addition to the
``engagement_ema`` scalar, the MLP also sees three soft-pooled geometry
scalars derived from the same ``target_logits`` softmax used to build
the geometric prior:

  probs    = softmax(target_logits)
  rel_soft = Σ probs · entity_rel         # (B*, 3) normalized
  vel_soft = Σ probs · entity_vel         # (B*, 3) normalized

  rel_w    = rel_soft · QNN_DIST_SCALE    # game units
  vel_w    = vel_soft · QNN_VEL_SCALE     # u/s
  dist     = |rel_w|              (clamp ≥ 1e-3)
  rel_u    = rel_w / dist
  radial   = rel_u · vel_w               # closing speed if negative
  tang     = sqrt(max(|vel_w|² − radial², 0))

  dist_norm   = dist   / 1000.0
  radial_norm = radial / 1000.0
  tang_norm   = tang   / 1000.0

The geometry scalars are zeroed when ``actor_mask`` has no live entity in
the frame, matching the prior's ``has_actor`` masking.

The geometric prior is unchanged from :class:`LookStyleAttackHead`:

  prior_logit  = aim_scale * look_prior[..., 0]
  features_aug = cat(features, engagement_ema, dist_norm, radial_norm, tang_norm)
  delta_attack = mlp(features_aug)        # in_dim grows by 4
  attack_logit = prior_logit + delta_attack

Motivation: the methodical MI sweep over conditional NLL flagged radial
velocity, tangential speed, and target distance as carrying small but
non-zero signal that the baseline head sees only implicitly through the
features token. This variant hands them to the MLP as dedicated input
columns so the gradient can address them directly.

``engagement_ema`` is sourced from the forward-scoped
:class:`EngagementEMAContext` set by the trainer
(:func:`qnn.model.policy._engagement_ema_scope`) — same plumbing pattern
as the other forward-scoped bench side channels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.weapon_physics import (
    ACTOR_REL_OFFSET, ACTOR_VEL_OFFSET, QNN_DIST_SCALE, QNN_VEL_SCALE,
)
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput
from qnn.model.bench.inputs.engagement_ema_context import current_engagement_ema_context

# entity_scalars_raw ACTOR layout — offsets owned by qnn.bc.weapon_physics.
_ESC_REL_BEGIN, _ESC_REL_END = ACTOR_REL_OFFSET, ACTOR_REL_OFFSET + 3
_ESC_VEL_BEGIN, _ESC_VEL_END = ACTOR_VEL_OFFSET, ACTOR_VEL_OFFSET + 3

_GEOM_NORM = 1000.0   # game-unit normalizer for dist / radial / tang


class GeomAttackHead(nn.Module):
    """LookStyleAttackHead + engagement_ema + (dist, radial_vel, tang_speed) on MLP input."""

    def __init__(
        self,
        in_dim: int,
        d_hidden: int,
        activation: str,
        *,
        scale_init: float = 5.0,
    ) -> None:
        super().__init__()
        self.scale_init = float(scale_init)
        # +1 engagement_ema, +3 geometry scalars (dist, radial, tangential).
        self.mlp = make_head_mlp(in_dim + 4, OUT_DIM, d_hidden, activation)
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "GeomAttackHead needs target_logits"
        assert inp.entity_scalars is not None, "GeomAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "GeomAttackHead needs actor_mask"

        entity_rel = inp.entity_scalars[..., _ESC_REL_BEGIN:_ESC_REL_END]    # (B*, N, 3)
        entity_vel = inp.entity_scalars[..., _ESC_VEL_BEGIN:_ESC_VEL_END]    # (B*, N, 3)
        probs = F.softmax(inp.target_logits, dim=-1)                          # (B*, N)
        probs_u = probs.unsqueeze(-1)
        soft_target_rel = (probs_u * entity_rel).sum(dim=-2)                  # (B*, 3)
        soft_target_vel = (probs_u * entity_vel).sum(dim=-2)                  # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_target_vel = soft_target_vel * has_actor

        # Geometric prior (unchanged from LookStyleAttackHead).
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        look_prior = soft_target_rel / soft_norm                               # (B*, 3)
        prior_logit = (self.scale_init * look_prior[..., 0:1]).to(inp.features.dtype)

        # Physical-units geometry from the soft-pooled rel/vel.
        rel_w = soft_target_rel * QNN_DIST_SCALE                              # (B*, 3)
        vel_w = soft_target_vel * QNN_VEL_SCALE                               # (B*, 3)
        dist = torch.linalg.vector_norm(rel_w, dim=-1, keepdim=True).clamp(min=1e-3)  # (B*, 1)
        rel_u = rel_w / dist
        radial = (rel_u * vel_w).sum(dim=-1, keepdim=True)                    # (B*, 1) closing < 0
        vel_sq = (vel_w * vel_w).sum(dim=-1, keepdim=True)                    # (B*, 1)
        tang_sq = (vel_sq - radial * radial).clamp(min=0.0)
        tang = torch.sqrt(tang_sq)                                            # (B*, 1)

        # Zero the geometry signals when no actor is present; the prior's
        # has_actor masking only zeros rel/vel, but dist clamps to 1e-3 so
        # dist_norm would still be a non-zero column without this.
        dist = dist * has_actor
        radial = radial * has_actor
        tang = tang * has_actor

        dist_norm = (dist / _GEOM_NORM).to(inp.features.dtype)
        radial_norm = (radial / _GEOM_NORM).to(inp.features.dtype)
        tang_norm = (tang / _GEOM_NORM).to(inp.features.dtype)

        engagement = current_engagement_ema_context().engagement_ema.to(inp.features.dtype)
        if engagement.dim() == 0:
            engagement = engagement.expand(inp.features.shape[0])
        engagement_col = engagement.unsqueeze(-1)                             # (B*, 1)

        features_aug = torch.cat(
            [inp.features, engagement_col, dist_norm, radial_norm, tang_norm],
            dim=-1,
        )                                                                     # (B*, in_dim+4)
        delta_attack = self.mlp(features_aug)                                 # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
