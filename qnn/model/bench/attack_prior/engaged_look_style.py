"""LookStyleAttackHead variant that consumes ``engagement_ema``.

OFAT delta vs :class:`LookStyleAttackHead`:

  features_aug = cat(features, engagement_ema.unsqueeze(-1))
  delta_attack = mlp(features_aug)   # in_dim grows by 1
  attack_logit = prior_logit + delta_attack

The geometric prior (``aim_scale * look_prior[..., 0]``) is unchanged.
The MLP gets one extra input column carrying the per-frame engagement
EMA (scalar in [0, 1]); its first linear learns its own weight on that
column. (NB: ``Network._init_weights`` runs ``xavier_uniform_`` over
every ``nn.Linear`` at construction end, which overwrites
LookStyleAttackHead's local zero-init of the final layer — so step 0 is
xavier-random, not exactly the prior, in both baseline and this variant.
OFAT comparison stays fair because both heads share that behaviour.)

``engagement_ema`` is sourced from the forward-scoped
:class:`EngagementEMAContext` set by the trainer
(:func:`qnn.model.policy._engagement_ema_scope`) — same plumbing pattern
as the other forward-scoped bench side channels.
"""
from __future__ import annotations

import torch

from qnn.model.attack_head import AttackHeadInput, AttackHeadOutput
from qnn.model.bench.attack_prior.look_style_attack_head import LookStyleAttackHead
from qnn.model.bench.inputs.engagement_ema_context import current_engagement_ema_context


class EngagedLookStyleAttackHead(LookStyleAttackHead):
    """LookStyleAttackHead + engagement_ema appended to MLP input."""

    def __init__(
        self,
        in_dim: int,
        d_hidden: int,
        activation: str,
        *,
        scale_init: float = 5.0,
    ) -> None:
        # +1 input dim for the engagement_ema scalar column.
        super().__init__(
            in_dim=in_dim + 1,
            d_hidden=d_hidden,
            activation=activation,
            scale_init=scale_init,
        )

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "EngagedLookStyleAttackHead needs target_logits"
        assert inp.entity_scalars is not None, "EngagedLookStyleAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "EngagedLookStyleAttackHead needs actor_mask"

        # Reuse the parent's geometric prior computation by routing through it
        # with no residual contribution, then recompute delta from augmented
        # features. We can't call super().forward() because that would use
        # the (in_dim+1)-shaped MLP against the original `features`.
        import torch.nn.functional as F  # local import: parity with parent module

        entity_rel = inp.entity_scalars[..., 3:6]                            # (B*, N, 3)
        probs = F.softmax(inp.target_logits, dim=-1)                         # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * entity_rel).sum(dim=-2)     # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        look_prior = soft_target_rel / soft_norm                              # (B*, 3)
        prior_logit = (self.scale_init * look_prior[..., 0:1]).to(inp.features.dtype)

        engagement = current_engagement_ema_context().engagement_ema.to(inp.features.dtype)
        if engagement.dim() == 0:
            engagement = engagement.expand(inp.features.shape[0])
        features_aug = torch.cat(
            [inp.features, engagement.unsqueeze(-1)], dim=-1,
        )                                                                    # (B*, in_dim+1)
        delta_attack = self.mlp(features_aug)                                # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
