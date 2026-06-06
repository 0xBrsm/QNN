"""Look head ablation: aim_vec prior + MLP(cat(target_feat, prev_look)).

Combines the two corrections from the bench look-ablation series:

  * **aim_vec prior** — replaces the canonical "point at current target
    position" base_look with the analytical lead-corrected aim direction
    for the held weapon. Solves the ballistic equation given
    ``entity_rel``, ``entity_vel``, projectile ``v_horiz`` (or hitscan
    sentinel), and ``gravity``. Computed via the existing math in
    ``qnn.model.bench.weapon_aim.lead_aim`` (see ``compute_lead_aim`` and
    ``pooled_aim_vec``) and a forward-scoped ``WeaponAimContext`` stashed
    by ``WeaponAimNetwork``.

  * **prev_look feature** — passes the previous frame's demonstrator
    look direction (3D, zero at episode starts) as an explicit MLP
    input. Built at preload by ``_make_resident_source`` and read from
    ``PrevLookContext``.

  pred_look = normalize(aim_vec + delta_look)
  delta_look = mlp(cat(target_feat, prev_look))

Pair with ``TargetOnlyObsEmbedding`` so the self-half of features is
zero — only target_feat and prev_look feed the MLP. Attack head Off
(look-only).
"""
from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.prev_look_context import current_prev_look_context
from qnn.model.bench.weapon_aim.context import current_weapon_aim_context
from qnn.model.bench.weapon_aim.lead_aim import (
    compute_lead_aim, held_weapon_trajectory, pooled_aim_vec,
)
from qnn.model.look_head import LookHeadInput, LookHeadOutput

OUT_DIM = 3


class PrevLookAimVecHead(nn.Module):
    """Look head: aim_vec prior + MLP(cat(target_feat, prev_look))."""

    def __init__(self, *, d_model: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # MLP input: target_feat (d_model) + prev_look (3).
        self.mlp = make_head_mlp(self.d_model + 3, OUT_DIM, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        wctx = current_weapon_aim_context()
        v_horiz, gravity = held_weapon_trajectory(wctx.weapon_static, wctx.weapon_id)
        per_entity_aim = compute_lead_aim(
            inp.entity_rel, wctx.entity_vel, v_horiz, gravity,
        )                                                                       # (B*, N, 3)
        base_look = pooled_aim_vec(
            per_entity_aim, inp.target_logits, inp.actor_mask,
        )                                                                       # (B*, 3) unit

        # target_feat is the second half of the canonical features layout.
        target_feat = inp.features[..., self.d_model:2 * self.d_model]
        # Context is K_MAX-wide (5 prev frames stacked); this head uses K=1.
        prev_look = current_prev_look_context().prev_look[..., :3].to(target_feat.dtype)
        mlp_in = torch.cat([target_feat, prev_look], dim=-1)                    # (B*, d_model + 3)
        delta_look = self.mlp(mlp_in)                                           # (B*, 3)

        unnormalized = base_look + delta_look
        out_norm = torch.linalg.vector_norm(unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)
        pred_look = unnormalized / out_norm
        return LookHeadOutput(pred_look=pred_look, base_look=base_look, delta_look=delta_look)
