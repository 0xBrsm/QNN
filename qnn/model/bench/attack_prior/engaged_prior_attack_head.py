"""LookStyleAttackHead variant with ``engagement_ema`` on the prior path.

This is the "engagement in the prior" variant — the architectural
alternative to :class:`EngagedLookStyleAttackHead`, which appends
``engagement_ema`` to the MLP input. Here we instead route the scalar
through a learnable bias on the prior:

  prior_logit  = aim_scale * look_prior[..., 0]
                 + engagement_scale * engagement_ema           # NEW term
  delta_attack = mlp(features)                                  # in_dim unchanged
  attack_logit = prior_logit + delta_attack

``aim_scale`` reuses the existing frozen ``scale_init`` constant from
``probe.json`` (default 5.0, supplied via the constructor). The new
``engagement_scale`` is a learnable scalar ``nn.Parameter`` initialised
to zero, so step-0 behaviour is identical to ``LookStyleAttackHead``
(OFAT discipline preserved). The MLP input dim is unchanged — features
only, with no augmentation — which is what distinguishes this variant
from :class:`EngagedLookStyleAttackHead`.

Tests the "learnable inductive bias" architectural pattern: rather than
let the MLP discover the engagement signal among its many input columns,
we hand it to the model as a dedicated additive prior term with a single
trainable gain. The pattern is not currently used elsewhere in the
codebase.

``engagement_ema`` is sourced from the forward-scoped
:class:`EngagementEMAContext` set by the trainer
(:func:`qnn.model.policy._engagement_ema_scope`) — same plumbing pattern
as the other forward-scoped bench side channels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.weapon_physics import ACTOR_REL_OFFSET
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput
from qnn.model.bench.inputs.engagement_ema_context import current_engagement_ema_context

# entity_scalars_raw ACTOR layout — offset owned by qnn.bc.weapon_physics.
_ESC_REL_BEGIN, _ESC_REL_END = ACTOR_REL_OFFSET, ACTOR_REL_OFFSET + 3


class EngagedPriorAttackHead(nn.Module):
    """LookStyleAttackHead + engagement_ema as a learnable additive prior term."""

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
        # Zero-init so step 0 matches LookStyleAttackHead exactly (no engagement
        # contribution on the prior path until training learns a non-zero gain).
        self.engagement_scale = nn.Parameter(torch.zeros(()))
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "EngagedPriorAttackHead needs target_logits"
        assert inp.entity_scalars is not None, "EngagedPriorAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "EngagedPriorAttackHead needs actor_mask"

        # Duplicated (not subclassed) from LookStyleAttackHead for clarity: the
        # geometric prior compute is short, and inlining it keeps the new
        # engagement term visible alongside the aim-alignment term below.
        entity_rel = inp.entity_scalars[..., _ESC_REL_BEGIN:_ESC_REL_END]    # (B*, N, 3)
        probs = F.softmax(inp.target_logits, dim=-1)                        # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * entity_rel).sum(dim=-2)    # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        look_prior = soft_target_rel / soft_norm                              # (B*, 3)

        aim_term = (self.scale_init * look_prior[..., 0:1]).to(inp.features.dtype)

        engagement = current_engagement_ema_context().engagement_ema.to(inp.features.dtype)
        if engagement.dim() == 0:
            engagement = engagement.expand(inp.features.shape[0])
        engagement_term = (self.engagement_scale.to(inp.features.dtype)
                           * engagement.unsqueeze(-1))                       # (B*, 1)

        prior_logit = aim_term + engagement_term
        delta_attack = self.mlp(inp.features)                                # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
