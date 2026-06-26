"""Attack head with LookHead-style inputs.

Mirrors :class:`qnn.model.look_head.LookHead`'s forward pass, but emits a
scalar attack logit instead of a 3-vector direction. The geometric prior
is self-contained — computed in-head from ``target_logits + entity_rel +
actor_mask`` — so the LookHead slot is not needed.

  look_prior   = normalize(Σ_n softmax(target_logits)[n] * entity_rel[n])
  prior_logit = scale * look_prior[..., 0]      # forward-axis alignment cosine
  delta       = mlp(features)                  # learned residual
  attack_logit = prior_logit + delta

Slot-friendly: takes :class:`AttackHeadInput` and reads ``target_logits``,
``entity_scalars`` (slicing the REL block), and ``actor_mask`` directly.
Pair with ``PreAttnEncoder`` + ``GTTargetPointer`` in the bench probe so
``target_logits`` is the GT-softmaxed-recovered distribution from the
``act_target_probs`` sidecar.

The residual MLP's final Linear is zero-initialized so training starts
at attack_logit == prior_logit exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.weapon_physics import ACTOR_REL_OFFSET
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput

# entity_scalars_raw ACTOR layout — offset owned by qnn.bc.weapon_physics.
_ESC_REL_BEGIN, _ESC_REL_END = ACTOR_REL_OFFSET, ACTOR_REL_OFFSET + 3


class LookStyleAttackHead(nn.Module):
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
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "LookStyleAttackHead needs target_logits"
        assert inp.entity_scalars is not None, "LookStyleAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "LookStyleAttackHead needs actor_mask"

        entity_rel = inp.entity_scalars[..., _ESC_REL_BEGIN:_ESC_REL_END]    # (B*, N, 3)
        probs = F.softmax(inp.target_logits, dim=-1)                        # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * entity_rel).sum(dim=-2)    # (B*, 3)

        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        look_prior = soft_target_rel / soft_norm                              # (B*, 3)

        prior_logit = (self.scale_init * look_prior[..., 0:1]).to(inp.features.dtype)
        delta_attack = self.mlp(inp.features)                                # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
