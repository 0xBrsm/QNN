"""Look head ablation: MLP over cat(target_feat, move_token), optional prior.

The **move token** is the exact attack_bundle motion token (a d_model
projection of ``cat(vel_xyz, view_pitch, look_delta)`` plus a movement-mode
embed and motion-powerup embed, where ``look_delta`` is the look-change
self-motion field), expressed via the shared :class:`TokenBuilder`
over named ``obs_fields`` so it stays identical to ``AttackBundleHead``'s motion bundle.

  move_token  = motion_proj(cat(vel, pitch, look_delta)) + movement_embed + powerup_embed
  look_delta  = mlp(cat(target_feat, move_token))
  look_predict = normalize(prior + look_delta)

``prior`` ∈ {"none", "aim_vec"}:
  * none     — no geometric prior (prior = 0); MLP regresses the full direction.
  * aim_vec  — lead-corrected weapon-aware aim direction (needs WeaponAimContext).

All obs scalars/embeds (including ``look_delta``) come from the forward-scoped
``ObsAccessor`` (entered by ``BenchObsNetwork`` for prior="none" or
``MoveAimNetwork`` for "aim_vec"). Pair with ``TargetOnlyObsEmbedding``; attack head Off.
"""
from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.tokens.obs_accessor import current_obs_accessor
from qnn.model.tokens.obs_fields import MOTION_FIELDS as _MOTION_FIELDS
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.look_head import LookHeadInput, LookHeadOutput

OUT_DIM = 3


class MoveTokenLookHead(nn.Module):
    """Look head: MLP(cat(target_feat, attack_bundle move_token)) + optional prior."""

    def __init__(
        self, *, d_model: int, d_hidden: int, activation: str,
        prior: str, entity_embed: nn.Embedding, movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        if prior not in ("none", "aim_vec"):
            raise ValueError(f"prior must be 'none' or 'aim_vec', got {prior!r}")
        self.d_model = int(d_model)
        self.prior = prior
        self.move_builder = TokenBuilder(
            self.d_model, _MOTION_FIELDS,
            entity_embed=entity_embed, movement_embed=movement_embed,
        )
        # MLP input: target_feat (d_model) + move_token (d_model).
        self.mlp = make_head_mlp(2 * self.d_model, OUT_DIM, d_hidden, activation)

    def _move_token(self, target_feat: torch.Tensor) -> torch.Tensor:
        return self.move_builder(current_obs_accessor(), dtype=target_feat.dtype)

    def _aim_vec_prior(self, inp: LookHeadInput) -> torch.Tensor:
        from qnn.model.bench.weapon_aim.context import current_weapon_aim_context
        from qnn.model.bench.weapon_aim.lead_aim import (
            compute_lead_aim, held_weapon_trajectory, pooled_aim_vec,
        )
        wctx = current_weapon_aim_context()
        v_horiz, gravity = held_weapon_trajectory(wctx.weapon_static, wctx.weapon_id)
        per_entity_aim = compute_lead_aim(inp.entity_rel, wctx.entity_vel, v_horiz, gravity)
        return pooled_aim_vec(per_entity_aim, inp.target_logits, inp.actor_mask)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        target_feat = inp.features[..., self.d_model:2 * self.d_model]
        move_token = self._move_token(target_feat)
        mlp_in = torch.cat([target_feat, move_token], dim=-1)                    # (B*, 2*d_model)
        look_delta = self.mlp(mlp_in)                                            # (B*, 3)

        if self.prior == "aim_vec":
            look_prior = self._aim_vec_prior(inp)
        else:
            look_prior = torch.zeros_like(look_delta)

        unnormalized = look_prior + look_delta
        out_norm = torch.linalg.vector_norm(unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)
        look_predict = unnormalized / out_norm
        return LookHeadOutput(look_predict=look_predict, look_prior=look_prior, look_delta=look_delta)
