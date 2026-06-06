"""Bench attack head: (aim_vec, target_feat, noop_scalar) → fire logit.

Drop-in for canonical ``AttackHead``. Computes ``aim_vec`` internally
from the forward-scoped ``WeaponAimContext`` (entity_vel, weapon_id,
weapon_static, noop) plus the canonical ``AttackHeadInput`` fields.

Decision rule the head learns: fire when ``aim_vec[0] ≈ 1`` (predicted
hit if fired forward this tick) AND ``noop == 0``.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.bc.weapon_physics import ACTOR_REL_OFFSET
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import AttackHeadInput, AttackHeadOutput
from qnn.model.bench.weapon_aim.context import current_weapon_aim_context
from qnn.model.bench.weapon_aim.lead_aim import (
    compute_lead_aim,
    held_weapon_trajectory,
    pooled_aim_vec,
)

OUT_DIM = 1


class WeaponAimAttackHead(nn.Module):
    """Attack head fed by (aim_vec, target_feat, noop) — computed internally."""

    def __init__(self, in_dim: int, bottleneck_dim: int, activation: str) -> None:
        super().__init__()
        self._target_feat_dim = int(in_dim)
        mlp_in = 3 + self._target_feat_dim + 1
        self.mlp = make_head_mlp(mlp_in, OUT_DIM, bottleneck_dim, activation)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        ctx = current_weapon_aim_context()
        if inp.target_logits is None or inp.actor_mask is None:
            raise RuntimeError(
                "WeaponAimAttackHead requires the canonical Network to "
                "populate target_logits and actor_mask on AttackHeadInput."
            )

        # entity_rel from the canonical input dataclass; entity_vel from the
        # bench context (it isn't part of the canonical AttackHeadInput).
        entity_rel = inp.entity_scalars[..., ACTOR_REL_OFFSET:ACTOR_REL_OFFSET + 3] \
            if inp.entity_scalars is not None else ctx.entity_rel
        v_horiz, gravity = held_weapon_trajectory(ctx.weapon_static, ctx.weapon_id)
        per_entity_aim = compute_lead_aim(entity_rel, ctx.entity_vel, v_horiz, gravity)
        aim_vec = pooled_aim_vec(per_entity_aim, inp.target_logits, inp.actor_mask)

        target_feat = inp.features[..., -self._target_feat_dim:]
        mlp_in = torch.cat([aim_vec, target_feat, ctx.noop.unsqueeze(-1)], dim=-1)
        attack_logit = self.mlp(mlp_in)

        zeros = torch.zeros_like(attack_logit)
        return AttackHeadOutput(
            attack_logit=attack_logit,
            prior_logit=zeros,
            delta_attack=attack_logit,
        )
