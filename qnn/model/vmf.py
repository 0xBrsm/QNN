"""von Mises–Fisher distribution on S² (p=3) and mixtures — for the look head.

The look turn-delta is a unit vector (a direction on the sphere). A mixture of vMF
is the continuous, multimodal distribution over directions:

    f(x; μ, κ) = C₃(κ) · exp(κ · μ·x),   C₃(κ) = κ / (4π sinh κ)

each component a mean direction μ (unit), a concentration κ (>0; tight ≈ high κ),
and a weight π. The mixture captures the flick/hold/track multimodality (one
component per mode — "hold" is a high-κ component at forward), with no
quantization, magnitude-spread tied to κ, and closed-form density + sampling
(exact for p=3, no rejection) + log-prob for PPO.

NOTE: vMF is ISOTROPIC (a single κ ⇒ circular spread around μ). It cannot express
the yaw≫pitch anisotropy of human aim within one component; see the Kent
distribution (Fisher–Bingham) if that turns out to matter.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LOG_4PI = math.log(4.0 * math.pi)
_LOG_2 = math.log(2.0)
KAPPA_MIN, KAPPA_MAX = 1e-2, 5.0e3      # clamp range (κ→0 uniform; κ huge ⇒ near-deterministic)


def log_sinh(kappa: torch.Tensor) -> torch.Tensor:
    """log sinh κ, stable for κ>0:  κ + log1p(−e^{−2κ}) − log 2."""
    return kappa + torch.log1p(-torch.exp(-2.0 * kappa)) - _LOG_2


def log_norm_const(kappa: torch.Tensor) -> torch.Tensor:
    """log C₃(κ) = log κ − log(4π) − log sinh κ.  → −log(4π) (uniform) as κ→0."""
    return torch.log(kappa) - _LOG_4PI - log_sinh(kappa)


def log_vmf(x: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """log f(x; μ, κ).  x,μ: (...,3) unit; κ: (...)."""
    return log_norm_const(kappa) + kappa * (mu * x).sum(-1)


def mixture_log_prob(
    x: torch.Tensor, mix_logits: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor,
) -> torch.Tensor:
    """log Σ_k π_k f(x; μ_k, κ_k).  x:(...,3); mix_logits,κ:(...,K); μ:(...,K,3).

    μ assumed unit, κ assumed >0 (the head emits them prepped); mix_logits raw.
    """
    mix_logp = torch.log_softmax(mix_logits, dim=-1)                 # (...,K)
    dot = (mu * x.unsqueeze(-2)).sum(-1)                             # (...,K)
    comp = log_norm_const(kappa) + kappa * dot                      # (...,K)
    return torch.logsumexp(mix_logp + comp, dim=-1)                 # (...)


def mixture_nll(
    x: torch.Tensor, mix_logits: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor,
) -> torch.Tensor:
    """Mean negative log-likelihood — the training loss."""
    return -mixture_log_prob(x, mix_logits, mu, kappa).mean()


def _tangent_basis(mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two orthonormal vectors spanning the tangent plane at μ (μ: (B,3))."""
    ref = torch.zeros_like(mu)
    along_x = mu[:, 0].abs() < 0.9
    ref[along_x, 0] = 1.0
    ref[~along_x, 1] = 1.0
    b1 = ref - (ref * mu).sum(-1, keepdim=True) * mu
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b2 = torch.cross(mu, b1, dim=-1)
    return b1, b2


def sample_vmf(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Sample x ~ vMF(μ, κ) on S² (exact inverse-CDF for p=3). μ:(B,3), κ:(B,) → (B,3)."""
    u = torch.rand_like(kappa)
    # W = cos(angle from μ) = 1 + (1/κ)·log(u + (1−u)e^{−2κ})
    w = 1.0 + torch.log(u + (1.0 - u) * torch.exp(-2.0 * kappa)) / kappa
    w = w.clamp(-1.0, 1.0)
    b1, b2 = _tangent_basis(mu)
    psi = (2.0 * math.pi) * torch.rand_like(kappa)
    t_dir = b1 * torch.cos(psi).unsqueeze(-1) + b2 * torch.sin(psi).unsqueeze(-1)
    x = w.unsqueeze(-1) * mu + torch.sqrt((1.0 - w * w).clamp(min=0.0)).unsqueeze(-1) * t_dir
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def mixture_sample(
    mix_logits: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x from the mixture (and return chosen component idx). (B,K) params → (B,3),(B,).

    temperature→0 ⇒ argmax component + κ→∞ (deterministic at the top μ); 1.0 matches
    the learned distribution; >1 jumpier.
    """
    t = max(float(temperature), 1e-6)
    comp = torch.distributions.Categorical(logits=mix_logits / t).sample()   # (B,)
    b = torch.arange(comp.shape[0], device=comp.device)
    mu_k = mu[b, comp]                                                        # (B,3)
    kappa_k = (kappa[b, comp] / t).clamp(KAPPA_MIN, KAPPA_MAX)                # sharpen with temp
    return sample_vmf(mu_k, kappa_k), comp
