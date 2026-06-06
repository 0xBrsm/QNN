"""Participation ratio + dead-unit + activation-spread analysis.

Activation-side capacity diagnostics. Note: we deprioritize PR for ablation
decisions (it doesn't predict where to cut — see the bn=32 run with PR≈5
that still benefited from B=192). Useful as supporting context, not as a
primary signal.

Migrated from ``scripts/bc/bottleneck_diag.py``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

HEAD_NAMES = ("move_head", "look_head", "attack_head", "weapon_head")


def _resolve_bottleneck_heads(model: nn.Module) -> dict[str, nn.Sequential]:
    """Collect heads built as Linear → ReLU/GELU → Linear (3-element Sequential).

    Heads are wrapped in Component containers (``MoveHead``, ``LookHead``,
    ``AttackHead``, ``WeaponHead``); the underlying MLP lives at ``head.mlp``.
    """
    heads: dict[str, nn.Sequential] = {}
    for name in HEAD_NAMES:
        component = getattr(model, name, None)
        mlp = getattr(component, "mlp", None) if component is not None else None
        if isinstance(mlp, nn.Sequential) and len(mlp) == 3:
            heads[name] = mlp
    return heads


@contextmanager
def _capture_post_activation(heads: dict[str, nn.Sequential]) -> Iterator[dict[str, list[torch.Tensor]]]:
    """Hook each head's middle module (the ReLU/GELU) and accumulate its outputs."""
    buffers: dict[str, list[torch.Tensor]] = {n: [] for n in heads}
    handles: list = []

    def _make_hook(name: str):
        def hook(_m, _inp, out):
            t = out.detach().to(torch.float32)
            buffers[name].append(t.reshape(-1, t.shape[-1]).cpu())
        return hook

    for name, head in heads.items():
        handles.append(head[1].register_forward_hook(_make_hook(name)))
    try:
        yield buffers
    finally:
        for h in handles:
            h.remove()


def participation_ratio(activations: torch.Tensor, *, eps: float = 1e-8) -> float:
    """PR = (Σσ²)² / Σσ⁴ — heuristic effective dimensionality of the activation span."""
    centered = activations.float() - activations.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    s2 = s.pow(2)
    return float(s2.sum().pow(2) / s2.pow(2).sum().clamp(min=eps))


def effective_rank_at_threshold(activations: torch.Tensor, *, threshold_ratio: float = 0.01) -> int:
    """Number of singular values >= threshold_ratio * max(s)."""
    centered = activations.float() - activations.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    if s.numel() == 0:
        return 0
    return int((s >= float(threshold_ratio) * float(s.max().item())).sum().item())


def per_unit_fire_stats(activations: torch.Tensor) -> dict:
    """Fire rate (fraction of frames where unit > 0) and mean magnitude when active."""
    n_frames = activations.shape[0]
    fired = (activations > 0).float()
    fire_rate = fired.mean(dim=0)
    mag_when_active = (activations.abs() * fired).sum(dim=0) / fired.sum(dim=0).clamp(min=1.0)
    return {
        "fire_rates": fire_rate.tolist(),
        "mean_mag_when_active": mag_when_active.tolist(),
        "n_frames": int(n_frames),
        "dead_units_le_001": int((fire_rate <= 0.001).sum().item()),
        "rare_units_lt_01": int(((fire_rate > 0.001) & (fire_rate < 0.01)).sum().item()),
        "always_on_units_ge_999": int((fire_rate >= 0.999).sum().item()),
    }


def head_bottleneck_report(
    policy,
    val_episodes: list[dict],
    *,
    max_frames: int = 4000,
) -> dict[str, dict]:
    """Run forward passes on val episodes, collect bottleneck activations per head,
    and emit PR / dead-unit / second-Linear column-norm statistics.

    Returns dict mapping head name → metrics dict.
    """
    heads = _resolve_bottleneck_heads(policy.model)
    if not heads:
        return {}

    with torch.inference_mode(), _capture_post_activation(heads) as buffers:
        frames_seen = 0
        for ep in val_episodes:
            n = ep["n_samples"]
            obs_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device)
                for k, v in ep["obs"].items()
            }
            policy._forward_tensors(obs_t)
            frames_seen += n
            if frames_seen >= max_frames:
                break

    out: dict[str, dict] = {}
    for name, head in heads.items():
        if not buffers[name]:
            continue
        act = torch.cat(buffers[name], dim=0)[:max_frames]
        # Stats
        stats = per_unit_fire_stats(act)
        pr = participation_ratio(act)
        eff = effective_rank_at_threshold(act)

        # Static second-Linear column norms (per-unit contribution to outputs)
        second = head[2]
        col_norms = second.weight.detach().to(torch.float32).cpu().norm(dim=0)

        out[name] = {
            "input_dim": int(head[0].in_features),
            "d_hidden": int(head[0].out_features),
            "output_dim": int(head[2].out_features),
            "frames_analyzed": int(act.shape[0]),
            "participation_ratio": pr,
            "effective_rank_1pct": eff,
            "dead_units": stats["dead_units_le_001"],
            "rare_units": stats["rare_units_lt_01"],
            "always_on_units": stats["always_on_units_ge_999"],
            "weak_cols_lt_10pct_max": int((col_norms < 0.1 * col_norms.max()).sum().item()),
            "fire_rate_quartiles": [
                float(q) for q in torch.quantile(
                    torch.tensor(stats["fire_rates"]),
                    torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
                ).tolist()
            ],
        }
    return out
