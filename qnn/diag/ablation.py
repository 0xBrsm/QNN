"""Layer / submodule ablation.

The most direct measure of "is this module doing real work?" — replace its
output with zeros (or its own training-time mean), and measure how much val
loss degrades. Big delta = essential. Small delta = candidate to cut.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn


@contextmanager
def replace_module_output(model: nn.Module, qualified_name: str, *, mode: str = "zero") -> Iterator[None]:
    """Patch a submodule's forward to emit zeros (or pass-through mean of input).

    ``mode``:
      - ``"zero"``: output replaced by torch.zeros_like(original_output)
      - ``"identity"``: output replaced by the *first* input tensor (skip-connect)
        — only meaningful for modules whose output shape equals input shape
    """
    target = model
    for part in qualified_name.split("."):
        target = getattr(target, part)
    original_forward = target.forward

    def patched_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if mode == "zero":
            if isinstance(out, torch.Tensor):
                return torch.zeros_like(out)
            if isinstance(out, tuple):
                return tuple(
                    torch.zeros_like(x) if isinstance(x, torch.Tensor) else x for x in out
                )
            return out
        elif mode == "identity":
            if isinstance(out, torch.Tensor) and args and isinstance(args[0], torch.Tensor):
                if out.shape == args[0].shape:
                    return args[0]
            return out
        raise ValueError(f"unknown ablation mode: {mode}")

    target.forward = patched_forward
    try:
        yield
    finally:
        target.forward = original_forward


@torch.inference_mode()
def episode_val_loss(policy, episodes: list[dict]) -> float:
    """Average val loss across the supplied episodes.

    Uses ``policy.evaluate_supervised`` directly — same path as training-time val.
    Per-episode losses are averaged weighted by episode length.
    """
    total = 0.0
    total_w = 0
    for ep in episodes:
        obs_t = {k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device) for k, v in ep["obs"].items()}
        act_t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(policy.device) for k, v in ep["actions"].items()}
        result = policy.evaluate_supervised(obs_t, act_t, compute_metrics=False)
        loss_t = result.get("loss")
        if loss_t is None:
            continue
        n = ep["n_samples"]
        total += float(loss_t.item()) * n
        total_w += n
    return total / max(total_w, 1)


def layer_ablation_table(
    policy,
    val_episodes: list[dict],
    *,
    targets: Iterable[str] | None = None,
    mode: str = "zero",
) -> list[dict]:
    """For each named submodule, ablate it and measure val-loss delta.

    Returns list of {name, baseline_loss, ablated_loss, delta} sorted by delta desc.
    Larger delta = more essential.
    """
    if targets is None:
        # Default: ablate each top-level head MLP and selected encoder components.
        targets = []
        for name, _ in policy.model.named_children():
            if name.endswith("_head") or name in ("gru", "target_pointer"):
                targets.append(name)
        # Plus each transformer layer if present
        if hasattr(policy.model, "encoder") and hasattr(policy.model.encoder, "blocks"):
            for i in range(len(policy.model.encoder.blocks)):
                targets.append(f"encoder.blocks.{i}")

    baseline = episode_val_loss(policy, val_episodes)
    rows: list[dict] = []
    for name in targets:
        try:
            with replace_module_output(policy.model, name, mode=mode):
                loss = episode_val_loss(policy, val_episodes)
            rows.append({
                "name": name,
                "baseline_loss": baseline,
                "ablated_loss": loss,
                "delta": loss - baseline,
            })
        except AttributeError:
            # Submodule path not present (e.g., no GRU when use_gru=False) — skip silently.
            continue
        except Exception as e:  # noqa: BLE001
            rows.append({
                "name": name,
                "baseline_loss": baseline,
                "ablated_loss": float("nan"),
                "delta": float("nan"),
                "error": str(e),
            })
    rows.sort(key=lambda r: r.get("delta", 0.0), reverse=True)
    return rows
