"""Discretization of the look turn-delta for a binned (classification) look head.

The look label is a view-relative turn-delta unit vector. We work in its 2D
tangent (log-map at forward=(1,0,0)): ``z = (z0, z1)`` with ``||z||`` = the turn
angle. A binned look head classifies each tangent axis independently into
foveated bins (fine near 0, coarse in the tails — most ticks are small turns),
exactly the CSGO/VPT recipe, but in our tangent space so it ties to ``look_r2``.

Shared by the head (logits → decoded direction) and the canonical look loss
(label → bin targets, cross-entropy). Lives in the model layer so both
``qnn.model.policy`` (loss) and the bench head import it without a layering cycle.

  logmap:   3D unit vec  → (B,2) tangent (per-axis turn)
  expmap:   (B,2) tangent → 3D unit vec
  targets:  (B,2) tangent → (B,2) long bin indices (nearest center)
  decode:   (B,2,N) logits → (B,2) expected tangent (softmax·centers)
"""
from __future__ import annotations

import torch

_EPS = 1e-6
N_BINS = 25                       # per axis (odd → a center bin at 0)
_THETA_MAX = torch.pi             # bins span [-pi, pi] (turn angle ≤ pi)
_FOVEA_POWER = 2.5                # >1 ⇒ denser bins near zero


def _build_centers() -> torch.Tensor:
    half = (N_BINS - 1) // 2
    k = torch.arange(1, half + 1, dtype=torch.float64) / half
    pos = (k ** _FOVEA_POWER) * _THETA_MAX           # foveated positive centers
    centers = torch.cat([-pos.flip(0), torch.zeros(1, dtype=torch.float64), pos])
    return centers.to(torch.float32)                 # (N_BINS,)


CENTERS = _build_centers()        # (N_BINS,) CPU; callers .to(device)


def tangent_logmap(u: torch.Tensor) -> torch.Tensor:
    """3D unit vectors (..., 3) → 2D tangent (..., 2); ||result|| = turn angle."""
    theta = torch.arccos(u[..., 0].clamp(-1.0, 1.0))
    yz = u[..., 1:3]
    n = torch.linalg.vector_norm(yz, dim=-1)
    scale = torch.where(n > _EPS, theta / n.clamp(min=_EPS), torch.zeros_like(theta))
    return yz * scale[..., None]


def tangent_expmap(z: torch.Tensor) -> torch.Tensor:
    """2D tangent (..., 2) → 3D unit vector (..., 3). z=0 → (1,0,0)."""
    theta = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
    direction = torch.where(theta > _EPS, z / theta.clamp(min=_EPS), torch.zeros_like(z))
    fwd = torch.cos(theta)                            # (...,1)
    off = torch.sin(theta) * direction                # (...,2)
    return torch.cat([fwd, off], dim=-1)


def bin_targets(z: torch.Tensor) -> torch.Tensor:
    """2D tangent (..., 2) → (..., 2) long bin indices (nearest center per axis)."""
    centers = CENTERS.to(z.device)
    d = (z[..., None] - centers).abs()                # (..., 2, N_BINS)
    return d.argmin(dim=-1)


def soft_bin_targets(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """2D tangent (..., 2) → (..., 2, N_BINS) soft Gaussian bin targets.

    Distance-aware label smoothing in ANGLE space: each axis target is a
    Gaussian over the bin centers, ``p[k] ∝ exp(-(center_k - z)^2 / 2σ^2)``,
    normalized per axis. ``sigma`` is in radians (same units as the tangent /
    centers). Because the centers are foveated (fine near 0, coarse in the
    tails), a fixed angular σ naturally spreads mass over the many sub-degree
    center bins — whose hard-label distinctions aren't learnable — while
    leaving the coarse tail bins nearly one-hot. Pairs with a soft-target
    cross-entropy (``-(soft * log_softmax(logits)).sum(-1)``) in the look loss.
    """
    centers = CENTERS.to(z.device).to(z.dtype)            # (N_BINS,)
    d2 = (z[..., None] - centers) ** 2                     # (..., 2, N_BINS)
    return torch.softmax(-d2 / (2.0 * sigma * sigma), dim=-1)


def decode(logits: torch.Tensor) -> torch.Tensor:
    """Per-axis bin logits (..., 2, N_BINS) → expected tangent (..., 2)."""
    centers = CENTERS.to(logits.device).to(logits.dtype)
    p = torch.softmax(logits, dim=-1)
    return (p * centers).sum(dim=-1)


# ───────────────────────── polar binning ─────────────────────────
# Magnitude × direction parameterization of the turn-delta. Unlike the per-axis
# Cartesian bins above, this makes "hold" (no turn) a single protected bin and
# represents a flick's yaw/pitch as one direction (capturing their correlation).
#   mag_bin 0          → hold (z = 0)
#   mag_bin 1..N_MAG   → foveated turn magnitudes θ
#   dir_bin 0..N_DIR-1 → uniform turn directions φ ∈ [0, 2π)
# z = (θ cos φ, θ sin φ);  expmap(z) → the 3D unit vector the engine consumes.
N_MAG = 12                        # foveated magnitude bins (excludes hold)
N_DIR = 16                        # uniform direction bins over [0, 2π)


def _build_mag_centers() -> torch.Tensor:
    k = torch.arange(1, N_MAG + 1, dtype=torch.float64) / N_MAG
    pos = (k ** _FOVEA_POWER) * _THETA_MAX            # foveated, dense near 0
    return torch.cat([torch.zeros(1, dtype=torch.float64), pos]).to(torch.float32)  # (N_MAG+1,)


MAG_CENTERS = _build_mag_centers()                    # (N_MAG+1,); [0]=0 (hold)
DIR_CENTERS = ((torch.arange(N_DIR, dtype=torch.float32) + 0.5)
               * (2.0 * torch.pi / N_DIR))            # (N_DIR,) bin-center angles
_HOLD_MAX = float(MAG_CENTERS[1]) * 0.5               # θ below this → hold bin 0

# ── Precomputed geometry for tangent-density look_dll ──────────────────────
# Bin Voronoi widths (per-axis; same for both axes). Used to convert per-axis
# bin probability to tangent-space density (log p_bin − log width).
import numpy as _np_lb
_C64 = CENTERS.numpy().astype("float64")
_EDGES64 = _np_lb.concatenate([[-_np_lb.pi], (_C64[:-1] + _C64[1:]) / 2, [_np_lb.pi]])
BIN_LOG_WIDTH: torch.Tensor = torch.tensor(
    _np_lb.log(_np_lb.diff(_EDGES64)), dtype=torch.float32
)  # (N_BINS,)

# Polar cell log-areas in tangent space. Shape: (N_MAG+1,); index = mag_bin.
# hold (mag_bin=0): disk of radius _HOLD_MAX → area = π × r²
# turn (mag_bin k): annular sector → area = Δφ × (r_hi² − r_lo²) / 2
_MC64 = MAG_CENTERS.numpy().astype("float64")
_POLAR_MEDGE = _np_lb.concatenate(
    [[_HOLD_MAX], (_MC64[1:-1] + _MC64[2:]) / 2, [_np_lb.pi]]
)
_DPHI = 2.0 * _np_lb.pi / N_DIR
POLAR_LOG_CELL_AREA: torch.Tensor = torch.tensor(
    _np_lb.log(_np_lb.concatenate([
        [_np_lb.pi * _HOLD_MAX ** 2],
        _DPHI * (_POLAR_MEDGE[1:] ** 2 - _POLAR_MEDGE[:-1] ** 2) / 2,
    ])),
    dtype=torch.float32,
)  # (N_MAG+1,)


def polar_targets(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2D tangent (..., 2) → (mag_bin, dir_bin) long. mag_bin 0 = hold (no turn)."""
    theta = torch.linalg.vector_norm(z, dim=-1)                       # (...,)
    phi = torch.atan2(z[..., 1], z[..., 0]) % (2.0 * torch.pi)        # (...,)
    mc = MAG_CENTERS.to(z.device)
    mag_bin = (theta[..., None] - mc[1:]).abs().argmin(dim=-1) + 1     # nearest non-hold center
    mag_bin = torch.where(theta < _HOLD_MAX, torch.zeros_like(mag_bin), mag_bin)
    dc = DIR_CENTERS.to(z.device)
    ang = torch.atan2(torch.sin(phi[..., None] - dc), torch.cos(phi[..., None] - dc)).abs()
    dir_bin = ang.argmin(dim=-1)                                      # nearest direction (circular)
    return mag_bin, dir_bin


def polar_to_tangent(mag_bin: torch.Tensor, dir_bin: torch.Tensor) -> torch.Tensor:
    """(mag_bin, dir_bin) long → 2D tangent z (..., 2). hold → 0."""
    theta = MAG_CENTERS.to(mag_bin.device)[mag_bin]
    phi = DIR_CENTERS.to(dir_bin.device)[dir_bin]
    return torch.stack([theta * torch.cos(phi), theta * torch.sin(phi)], dim=-1)


def polar_sample(
    mag_logits: torch.Tensor, dir_logits: torch.Tensor, temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (mag_bin, dir_bin) and reconstruct the 3D unit turn vector.

    temperature → 0 collapses to argmax (deterministic/robotic); 1.0 matches the
    learned (human) distribution; >1 is more random. Returns (mag_bin, dir_bin, vec3d).
    """
    t = max(float(temperature), 1e-6)
    mag_bin = torch.distributions.Categorical(logits=mag_logits / t).sample()
    dir_bin = torch.distributions.Categorical(logits=dir_logits / t).sample()
    vec = tangent_expmap(polar_to_tangent(mag_bin, dir_bin))
    return mag_bin, dir_bin, vec


def polar_log_prob(
    mag_logits: torch.Tensor, dir_logits: torch.Tensor,
    mag_bin: torch.Tensor, dir_bin: torch.Tensor,
) -> torch.Tensor:
    """log P(action) = log P(mag) + 𝟙[mag≠hold]·log P(dir). For PPO / NLL."""
    lpm = torch.log_softmax(mag_logits, dim=-1).gather(-1, mag_bin[..., None]).squeeze(-1)
    lpd = torch.log_softmax(dir_logits, dim=-1).gather(-1, dir_bin[..., None]).squeeze(-1)
    return lpm + (mag_bin > 0).to(lpm.dtype) * lpd
