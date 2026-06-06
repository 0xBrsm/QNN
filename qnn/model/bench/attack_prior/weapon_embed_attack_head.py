"""LookStyleAttackHead variant with a per-weapon embedding on the MLP input.

OFAT delta vs :class:`EngagedLookStyleAttackHead`:

  weapon_impulse = self_weapon_id_to_impulse(weapon_id)        # 0..8
  weapon_vec     = self.weapon_embed(weapon_impulse)           # (B*, weapon_embed_dim)
  features_aug   = cat(features, engagement_ema, weapon_vec)
  delta_attack   = mlp(features_aug)   # in_dim grows by 1 + weapon_embed_dim
  attack_logit   = prior_logit + delta_attack

The geometric prior (``aim_scale * base_look[..., 0]``) is unchanged.
``engagement_ema`` is still appended (this variant builds on the engaged
baseline — it's NOT engaged-vs-not, it's engaged + weapon-embed-vs-not).

The current heads see the held weapon only indirectly through
``self_readout``; this variant exposes it directly so the residual MLP
can specialise per weapon (NG/SNG/LG sustain vs RL/GL/SSG burst, etc.).

``weapon_id`` is the raw ``obs.self_weapon_id`` (ENTITY_IDS-encoded)
delivered via :class:`AttackHeadInput`; it is mapped to the 0..8 impulse
index via :func:`qnn.vocab.self_weapon_id_to_impulse` before indexing
the learnable embedding table. The embedding uses default ``nn.Embedding``
initialisation (N(0, 1)) — we deliberately do NOT zero it, so the MLP
has a non-trivial weapon signal from step 0 to learn against.

``engagement_ema`` is sourced from the forward-scoped
:class:`EngagementEMAContext` set by the trainer
(:func:`qnn.model.policy._engagement_ema_scope`) — same plumbing pattern
as ``prev_look``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput
from qnn.model.bench.inputs.engagement_ema_context import current_engagement_ema_context
from qnn.vocab import self_weapon_id_to_impulse

_ESC_REL_BEGIN, _ESC_REL_END = 3, 6   # entity_scalars_raw ACTOR layout
_WEAPON_IMPULSE_COUNT = 9             # impulses 0..8 (0 = no weapon)


class WeaponEmbedAttackHead(nn.Module):
    """LookStyleAttackHead + engagement_ema + per-weapon embedding on the MLP input."""

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
        # +1 engagement_ema scalar, +weapon_embed_dim weapon vector.
        mlp_in_dim = in_dim + 1 + self.weapon_embed_dim
        self.mlp = make_head_mlp(mlp_in_dim, OUT_DIM, d_hidden, activation)
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "WeaponEmbedAttackHead needs target_logits"
        assert inp.entity_scalars is not None, "WeaponEmbedAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "WeaponEmbedAttackHead needs actor_mask"
        assert inp.weapon_id is not None, "WeaponEmbedAttackHead needs weapon_id"

        entity_rel = inp.entity_scalars[..., _ESC_REL_BEGIN:_ESC_REL_END]    # (B*, N, 3)
        probs = F.softmax(inp.target_logits, dim=-1)                        # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * entity_rel).sum(dim=-2)    # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        base_look = soft_target_rel / soft_norm                              # (B*, 3)
        prior_logit = (self.scale_init * base_look[..., 0:1]).to(inp.features.dtype)

        engagement = current_engagement_ema_context().engagement_ema.to(inp.features.dtype)
        if engagement.dim() == 0:
            engagement = engagement.expand(inp.features.shape[0])

        # weapon_id arrives as (B*, 1) raw obs ID; squeeze to (B*,) for embedding lookup.
        weapon_impulse = self_weapon_id_to_impulse(inp.weapon_id.long()).squeeze(-1)
        weapon_vec = self.weapon_embed(weapon_impulse).to(inp.features.dtype)  # (B*, weapon_embed_dim)

        features_aug = torch.cat(
            [inp.features, engagement.unsqueeze(-1), weapon_vec], dim=-1,
        )                                                                    # (B*, in_dim+1+weapon_embed_dim)
        delta_attack = self.mlp(features_aug)                                # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
