"""Pure-MLP look head — no target-anchored prior.

look_predict = expmap(MLP(features))   # features = whatever the Network routes
                                       # (here: cat(gru_flat, target_feat); with the
                                       # target pointer Off, target_feat is zeros, so
                                       # effectively MLP(GRU(CLS))).

No prior, no target term — the head predicts the turn purely from the (temporal,
CLS-aggregated) features, letting the full encoder + GRU decide the look direction
rather than anchoring to the combat target. Tests whether all-token attention +
PPO-safe temporal context (GRU over obs, not a fed previous-look input) recovers look signal beyond
the ~0.09 grounded ceiling.

Emits look_delta = look_predict and look_prior = 0 so the canonical look regression
loss — smooth_l1(look_delta, look_label - look_prior) — collapses to
smooth_l1(look_predict, look_label), i.e. direct supervision of the prediction.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.look_bins import N_BINS, decode, tangent_expmap
from qnn.model.look_head import LookHeadInput, LookHeadOutput

_THETA_CAP = math.pi - 1e-3  # max per-tick turn angle (rad)


class PureLookHead(nn.Module):
    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.mlp = make_head_mlp(in_dim, 2, d_hidden, activation)  # 2D tangent

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        z = self.mlp(inp.features)                              # (B*, 2) tangent
        # Cap the turn angle theta = ||z|| below pi so expmap can't WRAP — a
        # >pi angle makes cos/sin(theta) oscillate, producing a wild direction
        # and exploding gradients (the unbounded MLP->expmap diverged: grad~800,
        # look_r2=-7). Smooth tanh squash keeps small turns ~unchanged.
        theta = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp(min=1e-6)
        z = z * (_THETA_CAP * torch.tanh(theta / _THETA_CAP) / theta)
        look_predict = tangent_expmap(z)                        # (B*, 3) unit
        zero = torch.zeros_like(look_predict)
        return LookHeadOutput(
            look_predict=look_predict, look_prior=zero, look_delta=look_predict,
        )


class PureBinnedLookHead(nn.Module):
    """Same as PureLookHead but a BINNED (classification) output — per-axis foveated
    tangent bin logits, trained with cross-entropy (the canonical look loss switches
    to CE when look_bins is set). look_predict = expmap(expected tangent) drives look_r2;
    acc_look_turn / bin-NLL are the distributional metrics. Tests whether onset/multimodal
    structure (which regression's mean averages into momentum) is recoverable on the full
    model. No prior; look_delta unused on the CE path -> zero.
    """

    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.mlp = make_head_mlp(in_dim, 2 * N_BINS, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        logits = self.mlp(inp.features).reshape(-1, 2, N_BINS)   # (B*, 2, N_BINS)
        look_predict = tangent_expmap(decode(logits))           # (B*, 3) decoded mean
        zero = torch.zeros_like(look_predict)
        return LookHeadOutput(
            look_predict=look_predict, look_prior=zero, look_delta=zero, look_bins=logits,
        )
