"""Effective rank of weight matrices via SVD.

Cheap, deterministic, no data needed. Tells you how much of each Linear's
parameter budget is "real" vs slack. Linear(in, out) with effective rank ≪
min(in, out) is over-parameterized — could be replaced by a smaller layer
or a low-rank factorization with no loss.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def linear_effective_rank(
    weight: torch.Tensor,
    *,
    threshold_ratio: float = 0.01,
) -> tuple[int, int, float]:
    """Effective rank of a 2D Linear weight matrix.

    Returns (effective_rank, full_rank, frac). effective_rank counts singular
    values >= threshold_ratio * max(s). frac = effective / full.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    s = torch.linalg.svdvals(weight.float().detach().cpu())
    full = int(min(weight.shape))
    if s.numel() == 0:
        return 0, full, 0.0
    cutoff = float(threshold_ratio) * float(s.max().item())
    eff = int((s >= cutoff).sum().item())
    return eff, full, eff / max(full, 1)


def all_linear_ranks(
    model: nn.Module,
    *,
    threshold_ratio: float = 0.01,
    skip_prefixes: Iterable[str] = (),
) -> list[dict]:
    """Compute effective rank for every Linear in the model.

    Returns list of {name, shape, full_rank, effective_rank, frac, n_params}.
    Sorted by frac ascending — most-overparameterized layers first.
    """
    skip = tuple(skip_prefixes)
    rows: list[dict] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name.startswith(p) for p in skip):
            continue
        eff, full, frac = linear_effective_rank(mod.weight, threshold_ratio=threshold_ratio)
        rows.append({
            "name": name,
            "shape": tuple(mod.weight.shape),
            "full_rank": full,
            "effective_rank": eff,
            "frac": frac,
            "n_params": int(mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)),
        })
    rows.sort(key=lambda r: r["frac"])
    return rows


def singular_value_spectrum(
    weight: torch.Tensor,
    *,
    n_top: int = 20,
) -> list[float]:
    """Return the top n singular values of a Linear weight (sorted descending).

    Useful to plot the spectrum for any single Linear of interest.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    s = torch.linalg.svdvals(weight.float().detach().cpu())
    return s[: int(n_top)].tolist()
