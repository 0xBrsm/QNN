"""Per-layer gradient diagnostics during one supervised batch.

The "where is signal flowing" question. A submodule with near-zero gradient
norm during a representative backward pass is either (a) trivially solving
its sub-problem and not learning anything new, or (b) gradient-isolated from
the loss. Both are signals.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def per_parameter_grad_norms(
    policy,
    episode: dict,
    *,
    head_loss_weights: dict | None = None,
) -> list[dict]:
    """Run one supervised forward+backward on the episode, return per-parameter grad norms.

    Sorted by parameter name. Use this to identify layers receiving little
    gradient signal vs the dominant ones.
    """
    obs_t = {
        k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device)
        for k, v in episode["obs"].items()
    }
    act_t = {
        k: torch.from_numpy(np.ascontiguousarray(v)).to(policy.device)
        for k, v in episode["actions"].items()
    }

    # Zero existing grads
    for p in policy.model.parameters():
        if p.grad is not None:
            p.grad = None

    # Forward + loss + backward (no optimizer step; we just want gradients)
    _, logits, _, _, target_logits, target_query = policy._forward_tensors(
        obs_t, hidden=None,
    )
    losses, loss_is_real, _ = policy._compute_head_losses_and_metrics(
        logits,
        act_t,
        head_loss_weights=head_loss_weights,
        compute_metrics=False,
        target_logits=target_logits,
        target_query=target_query,
        obs=obs_t,
    )
    real = [l for l, r in zip(losses, loss_is_real) if r]
    if not real:
        return []
    loss = torch.stack(real).mean()
    loss.backward()

    rows: list[dict] = []
    for name, p in policy.model.named_parameters():
        if p.grad is None:
            rows.append({"name": name, "shape": tuple(p.shape), "grad_norm": 0.0, "param_norm": float(p.detach().norm().item())})
            continue
        rows.append({
            "name": name,
            "shape": tuple(p.shape),
            "grad_norm": float(p.grad.detach().norm().item()),
            "param_norm": float(p.detach().norm().item()),
        })
    return rows


def aggregate_by_module(
    rows: list[dict],
    *,
    depth: int = 2,
) -> list[dict]:
    """Sum grad-norm and param-norm by submodule prefix at the given depth.

    e.g. ``trunk.blocks.0.attn.q_proj.weight`` aggregated at depth 2 → ``trunk.blocks``.
    """
    grouped: dict[str, dict] = OrderedDict()
    for r in rows:
        parts = r["name"].split(".")
        key = ".".join(parts[:depth]) if len(parts) >= depth else r["name"]
        if key not in grouped:
            grouped[key] = {"name": key, "grad_norm_sq": 0.0, "param_norm_sq": 0.0, "n_params": 0}
        grouped[key]["grad_norm_sq"] += r["grad_norm"] ** 2
        grouped[key]["param_norm_sq"] += r["param_norm"] ** 2
        grouped[key]["n_params"] += int(np.prod(r["shape"]))

    out = []
    for v in grouped.values():
        out.append({
            "name": v["name"],
            "grad_norm": float(v["grad_norm_sq"] ** 0.5),
            "param_norm": float(v["param_norm_sq"] ** 0.5),
            "n_params": v["n_params"],
        })
    out.sort(key=lambda r: r["grad_norm"], reverse=True)
    return out


def gradient_health_summary(rows: list[dict]) -> dict:
    """High-level: max/min/median grad norm, fraction of layers with very small grads."""
    if not rows:
        return {}
    norms = [r["grad_norm"] for r in rows]
    norms.sort()
    n = len(norms)
    median = norms[n // 2]
    return {
        "n_layers": n,
        "max_grad_norm": max(norms),
        "min_grad_norm": min(norms),
        "median_grad_norm": median,
        "frac_below_1pct_of_max": sum(1 for x in norms if x < 0.01 * max(norms)) / n,
        "frac_zero": sum(1 for x in norms if x == 0.0) / n,
    }
