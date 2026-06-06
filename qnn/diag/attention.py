"""Per-attention-head diagnostics: entropy, specialization, redundancy.

Forces the encoder's MultiheadAttention modules to return per-head weights via
a temporary forward override, runs a few val passes, and computes:

  - Per-head attention entropy (avg over queries) — high entropy means diffuse
    attention; low entropy means each query attends sharply to one key.
  - Cross-head similarity — cosine similarity of attention maps across heads.
    ~1.0 = redundant; ~0.0 = orthogonal patterns.
  - Per-head ablation — zero out one head's contribution to ``out_proj`` input
    and measure val-loss delta. The most direct "is this head pulling weight?"
    signal.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from qnn.diag.ablation import episode_val_loss


@contextmanager
def _force_per_head_weights(model: nn.Module) -> Iterator[dict[str, list[torch.Tensor]]]:
    """Patch every nn.MultiheadAttention.forward to capture per-head attention weights.

    Yields dict mapping qualified module name → list of (batch, num_heads, q, k) tensors.
    """
    captures: dict[str, list[torch.Tensor]] = {}
    saved: list[tuple[nn.Module, callable]] = []

    def make_patched(name: str, mha: nn.MultiheadAttention, original):
        def forward(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out, weights = original(*args, **kwargs)
            if weights is not None:
                captures.setdefault(name, []).append(weights.detach().cpu())
            return out, weights
        return forward

    for name, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            saved.append((mod, mod.forward))
            mod.forward = make_patched(name, mod, mod.forward)
    try:
        yield captures
    finally:
        for mod, original in saved:
            mod.forward = original


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Per-head, per-query entropy of attention distribution.

    Input ``weights`` shape: (batch, num_heads, q, k).
    Returns (num_heads,) tensor of mean entropy across batch and queries.
    """
    eps = 1e-12
    log_w = torch.log(weights.clamp(min=eps))
    ent = -(weights * log_w).sum(dim=-1)  # (batch, num_heads, q)
    return ent.mean(dim=(0, 2))  # (num_heads,)


def cross_head_similarity(weights: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between heads' flattened attention maps.

    Input shape: (batch, num_heads, q, k). Returns (num_heads, num_heads).
    """
    b, h, q, k = weights.shape
    flat = weights.reshape(b, h, q * k)
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # Average across batch
    flat_mean = flat.mean(dim=0)  # (num_heads, q*k)
    flat_mean = flat_mean / flat_mean.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return flat_mean @ flat_mean.T


def attention_pattern_summary(
    policy,
    val_episodes: list[dict],
    *,
    max_episodes: int = 4,
) -> dict[str, dict]:
    """For each MultiheadAttention in the model, capture attention weights on
    val episodes and return per-head entropy + cross-head similarity.

    Returns dict mapping module name → metrics.
    """
    out: dict[str, dict] = {}
    with torch.inference_mode(), _force_per_head_weights(policy.model) as captures:
        for ep in val_episodes[:max_episodes]:
            obs_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device)
                for k, v in ep["obs"].items()
            }
            policy._forward_tensors(obs_t)

    for name, weights_list in captures.items():
        if not weights_list:
            continue
        # Concatenate along batch dim
        weights = torch.cat(weights_list, dim=0)
        ent = attention_entropy(weights)
        sim = cross_head_similarity(weights)
        # Off-diagonal mean similarity (excluding the trivial 1.0 self)
        n_heads = sim.shape[0]
        if n_heads > 1:
            mask = ~torch.eye(n_heads, dtype=torch.bool)
            offdiag_sim = float(sim[mask].mean().item())
        else:
            offdiag_sim = float("nan")
        out[name] = {
            "n_heads": int(n_heads),
            "n_batches": int(weights.shape[0]),
            "seq_len": int(weights.shape[2]),
            "per_head_entropy": ent.tolist(),
            "mean_entropy": float(ent.mean().item()),
            "max_possible_entropy": float(np.log(weights.shape[3])),
            "cross_head_offdiag_sim_mean": offdiag_sim,
        }
    return out


@contextmanager
def _zero_attention_head(mha: nn.MultiheadAttention, head_idx: int) -> Iterator[None]:
    """Zero one head's contribution by zeroing its slice of in_proj rows for V
    and the corresponding rows of out_proj input.

    Strategy: zero the V projection rows for this head (so the head sees no
    value information) AND zero the out_proj input columns for this head
    (so its attention output doesn't reach the residual). Either alone would
    be a valid ablation; doing both is robust.
    """
    embed_dim = mha.embed_dim
    n_heads = mha.num_heads
    head_dim = embed_dim // n_heads
    start = head_idx * head_dim
    end = start + head_dim

    # in_proj_weight is (3*embed_dim, embed_dim) — Q rows 0..E, K rows E..2E, V rows 2E..3E
    saved_in_w = mha.in_proj_weight.data.clone()
    saved_out_w = mha.out_proj.weight.data.clone()
    saved_in_b = mha.in_proj_bias.data.clone() if mha.in_proj_bias is not None else None

    v_start = 2 * embed_dim + start
    v_end = 2 * embed_dim + end
    mha.in_proj_weight.data[v_start:v_end].zero_()
    if mha.in_proj_bias is not None:
        mha.in_proj_bias.data[v_start:v_end].zero_()
    # Zero out_proj input columns for this head
    mha.out_proj.weight.data[:, start:end].zero_()
    try:
        yield
    finally:
        mha.in_proj_weight.data.copy_(saved_in_w)
        mha.out_proj.weight.data.copy_(saved_out_w)
        if saved_in_b is not None:
            mha.in_proj_bias.data.copy_(saved_in_b)


def per_attention_head_ablation(
    policy,
    val_episodes: list[dict],
) -> list[dict]:
    """Zero out each attention head one at a time, measure val-loss delta.

    Returns list of {layer, head_idx, baseline, ablated, delta} sorted by delta desc.
    """
    baseline = episode_val_loss(policy, val_episodes)
    rows: list[dict] = []
    for name, mod in policy.model.named_modules():
        if not isinstance(mod, nn.MultiheadAttention):
            continue
        for h in range(mod.num_heads):
            with _zero_attention_head(mod, h):
                loss = episode_val_loss(policy, val_episodes)
            rows.append({
                "layer": name,
                "head_idx": h,
                "baseline_loss": baseline,
                "ablated_loss": loss,
                "delta": loss - baseline,
            })
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows
