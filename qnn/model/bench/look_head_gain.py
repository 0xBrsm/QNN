"""Gain-parameterized look head (version B): an explicit scalar gain on the
target bearing plus a free tangent residual.

  e_tgt  = logmap(look_prior)            # (B*,2); ||e_tgt|| = remaining angle (bearing)
  g      = sigmoid(linear(features))     # (B*,1) scalar gain in (0,1)
  delta  = mlp(features)                 # (B*,2) free residual: scan / lead / dodge / no-target
  z_pred = g * e_tgt + delta             # tangent-space sum — NO renormalize
  look_predict = expmap(z_pred)

Contrast with the canonical LookHead, which does ``normalize(unit_prior +
delta_3d)``: there the prior is a unit *full-snap* (implied gain = 1, the
overshoot that drove look_r2 negative) and the output magnitude only emerges
from the trailing renormalize, so there is no clean gain knob. Here the
combination is additive in the look tangent space, so ``g`` is literally the
fraction of the remaining bearing turned this tick. ``e_tgt`` is zero when no
actor is present, so on no-target frames the head reduces to the residual —
the target term is one component, not a lock-to-target.

Same LookHeadInput as canonical LookHead (features, target_logits, entity_rel,
actor_mask), so look_r2 is a clean canonical-vs-gain A/B.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.look_bins import tangent_expmap, tangent_logmap
from qnn.model.look_head import LookHeadInput, LookHeadOutput


class GainLookHead(nn.Module):
    """Look head: g·logmap(target_prior) + residual, combined in tangent space."""

    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.gain = nn.Linear(in_dim, 1)
        self.delta = make_head_mlp(in_dim, 2, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        # Soft-attended target direction (same as canonical LookHead): target_logits
        # already has -1e9 on non-actor indices, so softmax is actor-only.
        probs = F.softmax(inp.target_logits, dim=-1)                            # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * inp.entity_rel).sum(dim=-2)    # (B*, 3)
        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        look_prior = soft_target_rel / soft_norm                               # (B*, 3) unit (0 if no actor)

        e_tgt = tangent_logmap(look_prior)                                     # (B*, 2); 0 if no actor
        g = torch.sigmoid(self.gain(inp.features))                             # (B*, 1) in (0,1)
        delta = self.delta(inp.features)                                       # (B*, 2)
        z_pred = g * e_tgt + delta                                             # (B*, 2)
        look_predict = tangent_expmap(z_pred)                                  # (B*, 3) unit

        # The canonical look regression loss supervises `_look_delta` toward
        # `look_label - look_prior` via smooth_l1, and smooth_l1 depends only on
        # the difference. Emitting look_delta = look_predict - look_prior makes
        # that loss collapse to smooth_l1(look_predict, look_label) — i.e. direct
        # supervision of the prediction, with gradients flowing through g and the
        # tangent residual. (Returning a constant here breaks the graph.)
        look_delta = look_predict - look_prior                                 # (B*, 3)
        return LookHeadOutput(
            look_predict=look_predict, look_prior=look_prior, look_delta=look_delta,
        )
