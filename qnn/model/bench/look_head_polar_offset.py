"""Polar + stochastic-offset look head — the "discretized mixture" parameterization.

Extends the pure polar head (mag×dir categoricals; see look_head_polar) with a
continuous 2D Gaussian offset over the tangent residual WITHIN the chosen cell:

    mag ∈ {hold} ∪ {N_MAG foveated turn magnitudes}   (categorical)
    dir ∈ {N_DIR uniform directions in [0, 2π)}        (categorical)
    off ~ N(off_mean, diag(exp(off_logstd))^2)         (continuous, 2D)

  z = polar_to_tangent(mag, dir) + off

This keeps polar's multimodal structure (a single protected "hold" mode, correlated
yaw/pitch flicks) but makes the output resolution CONTINUOUS instead of snapping to
bin centers. The within-cell Gaussian replaces polar's flat −log(cell_area) density
term: the tangent density at a human turn z is

  log P(mag_bin(z)) + 𝟙[mag>0]·log P(dir_bin(z)) + log N(z − cell_center; mean, std)

For this to be a proper tangent density, the per-cell Gaussians must not overlap
much — off_logstd is clamped to a modest range and the harness ∫dens self-check
must come out ≈1. The hold cell (center 0) also gets the offset (held-frame jitter).

ACTING samples mag/dir, then adds a SAMPLED offset (not the mean) to preserve
within-mode jitter; ``look_predict`` is the deterministic argmax-cell + mean readout
for diagnostics only.
"""
from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.look_bins import (
    N_DIR, N_MAG, polar_to_tangent, tangent_expmap,
)
from qnn.model.look_head import LookHeadInput, LookHeadOutput

LOGSTD_MIN = -6.0
LOGSTD_MAX = 2.0


class PurePolarOffsetLookHead(nn.Module):
    """Look head: MLP(features) → mag + dir categoricals + 2D Gaussian within-cell offset."""

    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.n_mag1 = N_MAG + 1
        out_dim = self.n_mag1 + N_DIR + 2 + 2  # mag, dir, off_mean(2), off_logstd(2)
        self.mlp = make_head_mlp(in_dim, out_dim, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        h = self.mlp(inp.features)
        i = 0
        mag_logits = h[..., i:i + self.n_mag1]; i += self.n_mag1   # (B*, N_MAG+1) [0]=hold
        dir_logits = h[..., i:i + N_DIR]; i += N_DIR               # (B*, N_DIR)
        off_mean = h[..., i:i + 2]; i += 2                         # (B*, 2)
        off_logstd = h[..., i:i + 2].clamp(LOGSTD_MIN, LOGSTD_MAX)  # (B*, 2)

        # Deterministic readout for diagnostics only — ACTING samples (see harness).
        mag_bin = mag_logits.argmax(dim=-1)
        dir_bin = dir_logits.argmax(dim=-1)
        z = polar_to_tangent(mag_bin, dir_bin) + off_mean
        look_predict = tangent_expmap(z)
        zero = torch.zeros_like(look_predict)
        return LookHeadOutput(
            look_predict=look_predict, look_prior=zero, look_delta=zero,
            look_mag_logits=mag_logits, look_dir_logits=dir_logits,
            look_off_mean=off_mean, look_off_logstd=off_logstd,
        )
