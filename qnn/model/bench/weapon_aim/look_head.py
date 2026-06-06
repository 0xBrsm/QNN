"""Bench look head: weapon-aware aim_vec + target_feat-driven residual.

Drop-in for canonical ``LookHead`` — same ``LookHeadInput →
LookHeadOutput`` contract — but computes ``aim_vec`` (lead-corrected,
weapon-aware base direction) internally from its inputs + the
forward-scoped ``WeaponAimContext``.

  base_look = aim_vec
  delta_look = mlp(features)
  pred_look  = normalize(base_look + delta_look)

The residual path intentionally keeps the canonical feature contract:
``inp.features`` is passed directly to the MLP. The ablation changes the
geometric prior only.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.bench.weapon_aim.context import current_weapon_aim_context
from qnn.model.bench.weapon_aim.lead_aim import (
    compute_lead_aim,
    held_weapon_trajectory,
    pooled_aim_vec,
)
from qnn.model.look_head import LookHeadInput, LookHeadOutput

OUT_DIM = 3  # 3D direction vector


class WeaponAimLookHead(nn.Module):
    """Look head fed by (aim_vec, target_feat) — computed internally."""

    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        # MLP input is the canonical features (= cat(self_readout,
        # target_feat) when temporal Off). aim_vec goes into base_look,
        # the MLP learns delta_look the same way canonical does — keeping
        # the canonical input shape avoids ROCm/MIOpen issues we hit with
        # non-canonical MLP widths.
        self._motor_in = int(in_dim)
        self.mlp = make_head_mlp(self._motor_in, OUT_DIM, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        ctx = current_weapon_aim_context()

        v_horiz, gravity = held_weapon_trajectory(ctx.weapon_static, ctx.weapon_id)
        per_entity_aim = compute_lead_aim(
            inp.entity_rel, ctx.entity_vel, v_horiz, gravity,
        )                                                              # (B, N, 3)
        base_look = pooled_aim_vec(
            per_entity_aim, inp.target_logits, inp.actor_mask,
        )                                                              # (B, 3) unit

        delta_look = self.mlp(inp.features)                            # (B, 3)

        unnormalized = base_look + delta_look
        out_norm = torch.linalg.vector_norm(
            unnormalized, dim=-1, keepdim=True,
        ).clamp(min=1e-6)
        pred_look = unnormalized / out_norm

        return LookHeadOutput(
            pred_look=pred_look, base_look=base_look, delta_look=delta_look,
        )
