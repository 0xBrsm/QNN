"""Full-stack OFAT winner + predicted target-presence scalar.

Extends :class:`EngagedGeomWeaponEmbedAttackHead` (the current OFAT
winner) by appending one additional scalar to the MLP input: a
**predicted target-presence** signal derived from the target head's
output. This tests whether the "target acquired" cue carries
fire-prediction information beyond what ``base_look`` (the geometric
direction to the soft target) already provides.

The predicted target-presence scalar is:

  target_presence = 1 - softmax(target_logits)[..., 0]

``target_logits`` has shape ``(B*, 1 + N)`` where index 0 is the
"no-target" mass and 1..N are entity-slot logits (matching how
:class:`LookStyleAttackHead.forward` reads it). The softmax converts to
probabilities over (no-target, entity_0, ..., entity_{N-1});
``[..., 0]`` is the no-target probability; subtracting from 1 gives the
"predicted probability that SOME entity is the target."

The geometric prior is unchanged from :class:`LookStyleAttackHead`:

  probs        = softmax(target_logits)
  rel_soft     = Σ probs · entity_rel
  base_look    = normalize(rel_soft * has_actor)
  prior_logit  = aim_scale · base_look[..., 0]

The MLP input is the concatenation:

  features_aug = cat(
      features,                                    # (B*, in_dim)
      engagement_ema.unsqueeze(-1),                # (B*, 1)
      weapon_embed[weapon_impulse],                # (B*, weapon_embed_dim)
      dist_norm, radial_norm, tang_norm,           # (B*, 3)
      target_presence.unsqueeze(-1),               # (B*, 1)  ← NEW
  )
  delta_attack = mlp(features_aug)                 # (B*, 1)
  attack_logit = prior_logit + delta_attack

Set ``scale_init = 0`` in ``probe.json`` (the ``alignment_scale`` field
maps to this constructor kwarg) to drop the prior entirely — the head
itself needs no special "no prior" branch.

The geometry compute mirrors :class:`GeomAttackHead` exactly: physical
units via ``QNN_DIST_SCALE`` / ``QNN_VEL_SCALE``, ``has_actor`` masking
applied to the geometry columns, /1000 normalization, float32 compute
cast back to ``inp.features.dtype`` before concatenation. The
``target_presence`` scalar is likewise computed in float32 and cast to
``inp.features.dtype`` before concat.

The per-weapon embedding uses default ``nn.Embedding`` init (N(0,1)) —
deliberately not zeroed so the MLP has a non-trivial weapon signal from
step 0, matching :class:`WeaponEmbedAttackHead`.

The residual MLP's final Linear is zero-initialized so step 0 would be
exactly the prior. (NB: ``Network._init_weights`` runs
``xavier_uniform_`` over every ``nn.Linear`` at construction end, which
overwrites the local zero-init — so step 0 is xavier-random, not exactly
the prior. OFAT comparison stays fair because every engaged variant
shares that behaviour.)

``engagement_ema`` is sourced from the forward-scoped
:class:`EngagementEMAContext` set by the trainer
(:func:`qnn.model.policy._engagement_ema_scope`) — same plumbing pattern
as ``prev_look``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.weapon_physics import QNN_DIST_SCALE, QNN_VEL_SCALE
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput
from qnn.model.bench.inputs.engagement_ema_context import current_engagement_ema_context
from qnn.vocab import self_weapon_id_to_impulse

_ESC_REL_BEGIN, _ESC_REL_END = 3, 6    # entity_scalars_raw ACTOR layout
_ESC_VEL_BEGIN, _ESC_VEL_END = 7, 10

_GEOM_NORM = 1000.0                    # game-unit normalizer for dist / radial / tang
_WEAPON_IMPULSE_COUNT = 9              # impulses 0..8 (0 = no weapon)


class FullStackTargetAttackHead(nn.Module):
    """OFAT-winner stack + predicted target-presence scalar."""

    def __init__(
        self,
        in_dim: int,
        d_hidden: int,
        activation: str,
        *,
        scale_init: float = 5.0,
        weapon_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.scale_init = float(scale_init)
        self.weapon_embed_dim = int(weapon_embed_dim)
        # Default init (N(0,1)) — deliberately not zero so the MLP sees a
        # learnable per-weapon signal from step 0.
        self.weapon_embed = nn.Embedding(_WEAPON_IMPULSE_COUNT, self.weapon_embed_dim)
        # +1 engagement_ema, +weapon_embed_dim weapon vector, +3 geometry scalars,
        # +1 predicted target-presence scalar.
        mlp_in_dim = in_dim + 1 + self.weapon_embed_dim + 3 + 1
        self.mlp = make_head_mlp(mlp_in_dim, OUT_DIM, d_hidden, activation)
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "FullStackTargetAttackHead needs target_logits"
        assert inp.entity_scalars is not None, "FullStackTargetAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "FullStackTargetAttackHead needs actor_mask"
        assert inp.weapon_id is not None, "FullStackTargetAttackHead needs weapon_id"

        # --- Soft-pooled target rel / vel (geometry compute in float32) ---
        entity_rel = inp.entity_scalars[..., _ESC_REL_BEGIN:_ESC_REL_END].float()    # (B*, N, 3)
        entity_vel = inp.entity_scalars[..., _ESC_VEL_BEGIN:_ESC_VEL_END].float()    # (B*, N, 3)
        probs = F.softmax(inp.target_logits.float(), dim=-1)                          # (B*, N)
        probs_u = probs.unsqueeze(-1)
        soft_target_rel = (probs_u * entity_rel).sum(dim=-2)                          # (B*, 3)
        soft_target_vel = (probs_u * entity_vel).sum(dim=-2)                          # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_target_vel = soft_target_vel * has_actor

        # --- Geometric prior (LookStyle) ---
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        base_look = soft_target_rel / soft_norm                                       # (B*, 3)
        prior_logit = (self.scale_init * base_look[..., 0:1]).to(inp.features.dtype)

        # --- Physical-units geometry scalars ---
        rel_w = soft_target_rel * QNN_DIST_SCALE                                      # (B*, 3) game units
        vel_w = soft_target_vel * QNN_VEL_SCALE                                       # (B*, 3) u/s
        dist = torch.linalg.vector_norm(rel_w, dim=-1, keepdim=True).clamp(min=1e-3)  # (B*, 1)
        rel_u = rel_w / dist
        radial = (rel_u * vel_w).sum(dim=-1, keepdim=True)                            # (B*, 1) closing < 0
        vel_sq = (vel_w * vel_w).sum(dim=-1, keepdim=True)                            # (B*, 1)
        tang_sq = (vel_sq - radial * radial).clamp(min=0.0)
        tang = torch.sqrt(tang_sq)                                                    # (B*, 1)

        # Zero the geometry signals when no actor is present; dist clamps to
        # 1e-3 so dist_norm would otherwise be a non-zero column.
        dist = dist * has_actor
        radial = radial * has_actor
        tang = tang * has_actor

        dist_norm = (dist / _GEOM_NORM).to(inp.features.dtype)
        radial_norm = (radial / _GEOM_NORM).to(inp.features.dtype)
        tang_norm = (tang / _GEOM_NORM).to(inp.features.dtype)

        # --- Predicted target-presence scalar ---
        # softmax(target_logits)[..., 0] is the "no-target" probability mass
        # (index 0 of the target head's output); 1 - that is the predicted
        # probability that SOME entity is the target. Computed in float32 and
        # cast to inp.features.dtype before concat — same pattern as the
        # other scalars in this stack.
        target_presence = (1.0 - probs[..., 0]).to(inp.features.dtype)                # (B*,)
        target_presence_col = target_presence.unsqueeze(-1)                            # (B*, 1)

        # --- engagement_ema scalar ---
        engagement = current_engagement_ema_context().engagement_ema.to(inp.features.dtype)
        if engagement.dim() == 0:
            engagement = engagement.expand(inp.features.shape[0])
        engagement_col = engagement.unsqueeze(-1)                                     # (B*, 1)

        # --- per-weapon embedding ---
        # weapon_id arrives as (B*, 1) raw obs ID; squeeze to (B*,) for embedding lookup.
        weapon_impulse = self_weapon_id_to_impulse(inp.weapon_id.long()).squeeze(-1)
        weapon_vec = self.weapon_embed(weapon_impulse).to(inp.features.dtype)         # (B*, weapon_embed_dim)

        features_aug = torch.cat(
            [
                inp.features,
                engagement_col,
                weapon_vec,
                dist_norm,
                radial_norm,
                tang_norm,
                target_presence_col,
            ],
            dim=-1,
        )                                                                             # (B*, in_dim+1+weapon_embed_dim+3+1)
        delta_attack = self.mlp(features_aug)                                         # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
