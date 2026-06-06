"""Shared MLP helper for output heads.

Used by each head component (move/look/attack/weapon) so the bottleneck
+ activation contract stays in one place. ``head_activation`` is "none",
"gelu", or "relu" — relu kept for v22-era legacy checkpoint compatibility;
current training defaults to gelu.

Layout:
  (B>0, "none") → Linear(in, B) → Linear(B, out)
  (B>0, act)    → Linear(in, B) → act → Linear(B, out)
  (0,   "none") → Linear(in, out)
  (0,   act)    → Linear(in, in) → act → Linear(in, out)
"""

from __future__ import annotations

from torch import nn


def make_head_mlp(in_dim: int, out_dim: int, d_hidden: int, activation: str) -> nn.Module:
    hidden = d_hidden if d_hidden > 0 else in_dim
    activations = {"gelu": nn.GELU, "relu": nn.ReLU}
    has_activation = activation in activations
    if d_hidden > 0 or has_activation:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
        if has_activation:
            layers.append(activations[activation]())
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)
    return nn.Linear(in_dim, out_dim)
