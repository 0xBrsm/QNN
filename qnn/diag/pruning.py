"""Neuron-level pruning sensitivity.

For each unit in the chosen submodule(s), zero its incoming weights and
measure the resulting val-loss delta. Sort by impact: top of the list is
"essential", bottom is "redundant — safely prunable". This is the most
direct measure of per-neuron contribution.

Cost note: O(n_neurons × val_episodes). For each head's 128-d bottleneck:
512 evaluations on 2-4 episodes is a few minutes on CPU. For encoder layers
(thousands of neurons), prefer ``top_k_pruning`` which prunes the K
smallest-weight-norm neurons at once and measures.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from qnn.diag.ablation import episode_val_loss


@contextmanager
def _zero_neuron_in_linear(linear: nn.Linear, idx: int) -> Iterator[None]:
    """Zero out the ``idx``-th output neuron of a Linear (its row in weight + bias)."""
    saved_w = linear.weight.data[idx].clone()
    saved_b = linear.bias.data[idx].clone() if linear.bias is not None else None
    linear.weight.data[idx].zero_()
    if linear.bias is not None:
        linear.bias.data[idx] = 0.0
    try:
        yield
    finally:
        linear.weight.data[idx].copy_(saved_w)
        if saved_b is not None:
            linear.bias.data[idx].copy_(saved_b)


def per_neuron_pruning_sensitivity(
    policy,
    val_episodes: list[dict],
    *,
    target: str,
    neurons: list[int] | None = None,
) -> list[dict]:
    """Zero one neuron of ``target`` Linear at a time, measure val-loss delta.

    ``target`` is the qualified name of an nn.Linear module — for the head
    bottleneck output use e.g. ``"move_head.0"`` (the first Linear).

    Returns list of {neuron, baseline, ablated, delta} sorted by delta desc.
    """
    parts = target.split(".")
    mod = policy.model
    for p in parts:
        mod = getattr(mod, p)
    if not isinstance(mod, nn.Linear):
        raise ValueError(f"target {target!r} is not a Linear (got {type(mod).__name__})")

    out_features = mod.weight.shape[0]
    if neurons is None:
        neurons = list(range(out_features))
    else:
        for i in neurons:
            if i < 0 or i >= out_features:
                raise ValueError(f"neuron index {i} out of range [0, {out_features})")

    baseline = episode_val_loss(policy, val_episodes)
    rows: list[dict] = []
    for i in neurons:
        with _zero_neuron_in_linear(mod, i):
            loss = episode_val_loss(policy, val_episodes)
        rows.append({
            "neuron": i,
            "baseline_loss": baseline,
            "ablated_loss": loss,
            "delta": loss - baseline,
        })
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def cumulative_pruning_curve(rows: list[dict]) -> list[dict]:
    """Given per-neuron sensitivity rows (sorted desc by delta), build a
    cumulative curve: how much of total impact is captured by top-K neurons.

    Useful to answer: "what fraction of capacity is in the top 10/25/50% of neurons?"
    """
    if not rows:
        return []
    deltas = [max(r["delta"], 0.0) for r in rows]  # negative deltas treated as 0 contribution
    total = sum(deltas)
    out = []
    running = 0.0
    for i, r in enumerate(rows):
        running += max(r["delta"], 0.0)
        out.append({
            "k": i + 1,
            "neuron": r["neuron"],
            "delta": r["delta"],
            "cumulative_delta": running,
            "cumulative_frac": (running / total) if total > 0 else 0.0,
        })
    return out


def top_k_weight_norm_neurons(
    linear: nn.Linear,
    *,
    k: int,
    largest: bool = False,
) -> list[int]:
    """Return indices of K neurons sorted by output weight L2 norm.

    ``largest=False`` returns the K smallest (likely-redundant) neurons.
    ``largest=True`` returns the K largest (likely-essential) for sanity-check
    pruning ("did I really break the model when I zeroed the most-important neurons?").
    """
    norms = linear.weight.detach().norm(dim=1)
    order = torch.argsort(norms, descending=largest)
    return order[:int(k)].tolist()


def head_bottleneck_pruning_summary(
    policy,
    val_episodes: list[dict],
) -> dict[str, dict]:
    """Run per-neuron pruning on each head's bottleneck (the first Linear's
    output dim). Returns per-head summary including the cumulative curve.
    """
    out: dict[str, dict] = {}
    for head_name in ("move_head", "look_head", "attack_head", "weapon_head"):
        component = getattr(policy.model, head_name, None)
        mlp = getattr(component, "mlp", None) if component is not None else None
        if not isinstance(mlp, nn.Sequential) or len(mlp) < 1 or not isinstance(mlp[0], nn.Linear):
            continue
        target = f"{head_name}.mlp.0"
        rows = per_neuron_pruning_sensitivity(policy, val_episodes, target=target)
        cum = cumulative_pruning_curve(rows)
        # Find K such that top-K explain 90% of impact
        k_at_90 = next((c["k"] for c in cum if c["cumulative_frac"] >= 0.9), len(cum))
        # Number of neurons with effectively zero impact (delta < 0.001)
        n_redundant = sum(1 for r in rows if abs(r["delta"]) < 0.001)
        out[head_name] = {
            "n_neurons": len(rows),
            "max_delta": rows[0]["delta"] if rows else 0.0,
            "median_delta": rows[len(rows) // 2]["delta"] if rows else 0.0,
            "k_at_90pct": k_at_90,
            "n_redundant_lt_0.001": n_redundant,
        }
    return out
