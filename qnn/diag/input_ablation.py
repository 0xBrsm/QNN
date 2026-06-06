"""Input-side ablation and saliency for a Linear's input.

Sibling to :mod:`qnn.diag.pruning`, which zeros *output* rows of a Linear
("which bottleneck neurons matter"). This module zeros *input* columns
("which input dims is the layer actually consuming") and adds an
input-gradient saliency that catches the same signal more cheaply for
large input dims.

Typical use, given a probe with ``LookStyleAttackHead`` and a head MLP
``Linear(motor_in=128, bottleneck=32) → GELU → Linear(32, 1)``:

  * ``per_input_dim_ablation(policy, val, target="attack_head.mlp.0")``
    zeros each of the 128 input columns one at a time and returns the
    val-loss delta. Sorted desc gives "which input dims is the trained
    head leaning on".
  * ``input_slice_ablation(...)`` zeros a *named slice* (e.g. ``"self"``
    = dims 0..d_model, ``"target"`` = dims d_model..2*d_model). Single
    eval per slice — cheap whole-half ablation.
  * ``input_saliency(...)`` runs one backward pass and reports
    mean |∂loss/∂input_i| per input dim. Order-of-magnitude faster than
    per-dim ablation; equivalent ranking in practice.

Cost: per-dim ablation is O(in_features × val_episodes). For a 128-dim
input × 4 episodes that's ~30s on the trainer GPU. Slice ablation is
constant. Saliency is one extra backward — sub-second.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn

from qnn.diag.ablation import episode_val_loss


def _resolve_linear(policy, qualified_name: str) -> nn.Linear:
    target = policy.model
    for part in qualified_name.split("."):
        target = getattr(target, part)
    if not isinstance(target, nn.Linear):
        raise ValueError(
            f"{qualified_name!r} is not a Linear (got {type(target).__name__})"
        )
    return target


@contextmanager
def _zero_input_columns(linear: nn.Linear, cols: Sequence[int]) -> Iterator[None]:
    """Zero the listed input columns of ``linear.weight`` (in-place, restored on exit).

    ``linear.weight`` shape is ``(out_features, in_features)`` — columns
    correspond to input dims. Bias is not touched (it's tied to outputs).
    """
    saved = linear.weight.data[:, cols].clone()
    linear.weight.data[:, cols] = 0.0
    try:
        yield
    finally:
        linear.weight.data[:, cols] = saved


def per_input_dim_ablation(
    policy,
    val_episodes: list[dict],
    *,
    target: str,
    dims: list[int] | None = None,
) -> list[dict]:
    """Zero one input column of ``target`` Linear at a time, measure val-loss delta.

    Returns list of ``{dim, baseline_loss, ablated_loss, delta}`` sorted by
    delta descending.
    """
    linear = _resolve_linear(policy, target)
    in_features = int(linear.weight.shape[1])
    if dims is None:
        dims = list(range(in_features))
    else:
        for d in dims:
            if d < 0 or d >= in_features:
                raise ValueError(f"dim {d} out of range [0, {in_features})")

    baseline = episode_val_loss(policy, val_episodes)
    rows: list[dict] = []
    for d in dims:
        with _zero_input_columns(linear, [d]):
            ablated = episode_val_loss(policy, val_episodes)
        rows.append({
            "dim": d,
            "baseline_loss": baseline,
            "ablated_loss": ablated,
            "delta": ablated - baseline,
        })
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def input_slice_ablation(
    policy,
    val_episodes: list[dict],
    *,
    target: str,
    slices: dict[str, tuple[int, int] | slice],
) -> dict[str, dict[str, float]]:
    """Zero a named slice of input dims, measure val-loss delta. One eval per slice.

    ``slices`` maps a name to a half-open ``(start, end)`` range or a
    ``slice`` object. Example for the look/attack parity probes where
    ``features = cat(self_readout, target_feat)`` at ``in_features=128``:

        slices={
            "self": (0, 64),     # self / weapon-token half
            "target": (64, 128), # target_feat half
        }

    Returns ``{name: {baseline_loss, ablated_loss, delta}}``.
    """
    linear = _resolve_linear(policy, target)
    in_features = int(linear.weight.shape[1])

    baseline = episode_val_loss(policy, val_episodes)
    out: dict[str, dict[str, float]] = {}
    for name, sl in slices.items():
        if isinstance(sl, slice):
            cols = list(range(*sl.indices(in_features)))
        else:
            start, end = sl
            if not (0 <= start < end <= in_features):
                raise ValueError(
                    f"slice {name!r} out of range: {start},{end} (in_features={in_features})"
                )
            cols = list(range(start, end))
        with _zero_input_columns(linear, cols):
            ablated = episode_val_loss(policy, val_episodes)
        out[name] = {
            "baseline_loss": baseline,
            "ablated_loss": ablated,
            "delta": ablated - baseline,
            "dims_zeroed": len(cols),
        }
    return out


def input_saliency(
    policy,
    val_episodes: list[dict],
    *,
    target: str,
    head_logit: str = "attack",
) -> dict[str, np.ndarray]:
    """Mean |∂head_logit / ∂Linear-input| per input dim, averaged over val.

    Runs one forward+backward per episode with the target Linear's input
    captured via a forward hook. The gradient flowing back into that
    input is the per-dim sensitivity of ``head_logit`` (default
    ``attack``) to each input column of the head's first Linear.

    Returns ``{"mean_abs_grad": (in_features,), "mean_input": (in_features,),
    "saliency": mean_abs_grad * mean_abs_input}``. ``saliency`` weights
    the gradient by typical input magnitude so dims that are large but
    flat don't dominate. Sort desc by ``saliency`` for "what is the
    head depending on".

    Order-of-magnitude faster than ``per_input_dim_ablation`` and
    correlates well in practice; use ablation when you want exact
    val-loss deltas, saliency for triage.
    """
    linear = _resolve_linear(policy, target)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, inp, _out):
        x = inp[0]
        x.requires_grad_(True)
        x.retain_grad()
        captured["x"] = x
        return None

    handle = linear.register_forward_hook(hook)
    abs_grad_sum = torch.zeros(int(linear.weight.shape[1]), dtype=torch.float32)
    abs_input_sum = torch.zeros_like(abs_grad_sum)
    n_seen = 0
    try:
        for ep in val_episodes:
            obs_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device)
                for k, v in ep["obs"].items()
            }
            policy.model.zero_grad(set_to_none=True)
            _, logits, _, _, _, _ = policy._forward_tensors(obs_t)
            if head_logit not in logits:
                raise RuntimeError(
                    f"head_logit={head_logit!r} not in forward logits "
                    f"(have {sorted(logits)})"
                )
            target_t = logits[head_logit]
            target_t.sum().backward()
            x = captured.get("x")
            if x is None or x.grad is None:
                continue
            g = x.grad.detach().reshape(-1, x.shape[-1]).float().cpu()
            xa = x.detach().reshape(-1, x.shape[-1]).float().cpu()
            abs_grad_sum += g.abs().sum(dim=0)
            abs_input_sum += xa.abs().sum(dim=0)
            n_seen += g.shape[0]
    finally:
        handle.remove()
        policy.model.zero_grad(set_to_none=True)

    denom = max(n_seen, 1)
    mean_abs_grad = (abs_grad_sum / denom).numpy()
    mean_abs_input = (abs_input_sum / denom).numpy()
    return {
        "mean_abs_grad": mean_abs_grad,
        "mean_abs_input": mean_abs_input,
        "saliency": mean_abs_grad * mean_abs_input,
        "n_frames": float(n_seen),
    }
