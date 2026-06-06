"""Look head ablation: base_look prior + MLP over (target_feat, prev_look).

Tests whether explicitly feeding the previous frame's demonstrator
look direction closes the ambient-tracking residual that the
canonical (frame-shuffled) look head leaves on the table.

Architecture:

  base_look  = canonical soft-pooled (softmax(target_logits) * entity_rel)
  target_feat = inp.features[..., d_model:2*d_model]  (sliced from canonical
                                                       cat(self_readout, target_feat))
  prev_look  = read from PrevLookContext (built at preload from
               actions["look"] shifted by one frame along the episode
               axis, zero at episode starts)
  delta_look = mlp(cat(target_feat, prev_look))
  pred_look  = normalize(base_look + delta_look)

The self_readout half of inp.features is ignored — pair with
``TargetOnlyObsEmbedding`` so it carries no signal anyway. The MLP
input width is ``d_model + 3``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.prev_look_context import current_prev_look_context
from qnn.model.look_head import LookHeadInput, LookHeadOutput

OUT_DIM = 3


class PrevLookHead(nn.Module):
    """Look head: base_look prior + MLP(cat(target_feat, prev_look[1..K])).

    ``num_prev_frames=K`` selects how many prior demonstrator-look frames
    enter the MLP. K=1 means look[t-1] only; K=5 stacks look[t-1..t-5].
    The preload at ``_make_resident_source`` always emits K_MAX=5 frames
    so any prefix is sliceable at zero extra cost.
    """

    def __init__(
        self, *, d_model: int, d_hidden: int, activation: str,
        num_prev_frames: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_prev_frames = int(num_prev_frames)
        if not 1 <= self.num_prev_frames <= 5:
            raise ValueError(f"num_prev_frames must be in [1, 5], got {num_prev_frames}")
        prev_dim = 3 * self.num_prev_frames
        self.mlp = make_head_mlp(self.d_model + prev_dim, OUT_DIM, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        # Canonical target-anchored prior (matches qnn.model.look_head.LookHead).
        probs = F.softmax(inp.target_logits, dim=-1)                              # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * inp.entity_rel).sum(dim=-2)      # (B*, 3)
        has_actor = inp.actor_mask.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor
        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        base_look = soft_target_rel / soft_norm

        target_feat = inp.features[..., self.d_model:2 * self.d_model]
        prev_stack = current_prev_look_context().prev_look.to(target_feat.dtype)
        # Slice the first 3*K dims (covers t-1, t-2, ..., t-K). The preload
        # zeroes the corresponding columns at the first K frames of each
        # episode, so no per-frame boundary handling needed here.
        prev_look = prev_stack[..., : 3 * self.num_prev_frames]

        mlp_in = torch.cat([target_feat, prev_look], dim=-1)                      # (B*, d_model + 3K)
        delta_look = self.mlp(mlp_in)                                             # (B*, 3)

        unnormalized = base_look + delta_look
        out_norm = torch.linalg.vector_norm(unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)
        pred_look = unnormalized / out_norm
        return LookHeadOutput(pred_look=pred_look, base_look=base_look, delta_look=delta_look)
