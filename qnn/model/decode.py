"""Cross-gen-stable decode (post-#17 head era).

The ops here have been STABLE across the whole post-PR-#17 head era — every
generation's policy/export has needed them in the same form:

  * the sampling PRIMITIVES (:func:`categorical_sample`, :func:`bernoulli_sample`,
    :func:`gumbel_argmax`) — the stateless readout draws every head decode is
    built from;
  * the ATTACK bit decode (:func:`decode_attack_bit`) — sigmoid + 0.5 threshold
    (greedy) / temperature-bernoulli (sampled);
  * the per-axis MOVE decode (:func:`decode_move_axes`) — the basic
    "sample don't argmax" readout (argmax greedy / categorical-sample sampled).

What does NOT live here:
  * The gen-SPECIFIC decode — the polar-look hybrid, the sticky-weapon gate, the
    aim-prior blend — lives in :mod:`qnn.model.bench.a24.decode`. Those are a24-lineage
    CHOICES, not cross-gen invariants; a future generation replaces that module.
  * The a24-recent move LAYERS on top of the basic axis decode — sticky
    hysteresis, the semi-Markov hazard supplement, switch-back watermarking,
    stop-onset suppression — stay INLINE in :meth:`qnn.model.policy.QNNPolicy.act`.
    They are deeply entangled with per-episode instance state (held-class
    threading, row generators, caller-owned watermark buffers) and are a24-recent,
    not part of the cross-gen base.
  * The polar GEOMETRY (``tangent_expmap`` / ``MAG_CENTERS`` / ``DIR_CENTERS``)
    lives in :mod:`qnn.model.look_bins`; the lead/aim primitives in
    :mod:`qnn.model.bench.a24.lead_aim`. This module imports neither — it is pure readout.

EXPORT note: ``ExportWrapper`` (tools/export_onnx.py) DEFERS the move-axis and
attack-bit decode to the C engine (it emits raw fb/lr logits + gumbel-perturbed
jump, and the attack LOGIT), so :func:`decode_attack_bit` / :func:`decode_move_axes`
are called by ``QNNPolicy.act`` only. Base membership is about cross-gen
STABILITY, not about being called by both surfaces.

TRACE-SAFETY: keep this module dependency-light (torch only) and free of
``.item()``, data-dependent Python control flow, and advanced/boolean tensor
indexing, so anything reused by a traced graph stays ONNX-traceable.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


# ── Sampling primitives ─────────────────────────────────────────────
# Stateless readout draws shared by every head decode. ``row_generators`` (one
# torch.Generator per row) lets the eval pipeline advance each episode's RNG
# independently for reproducibility across batched envs; pass None for the
# default global RNG.

def categorical_sample(
    probs: torch.Tensor,
    row_generators: Any | None,
) -> torch.Tensor:
    """Sample one class index per row from probs (B, K).

    When row_generators is None, uses default RNG. When provided, draws
    one row at a time so each episode's RNG advances independently — the
    eval pipeline relies on this for reproducibility across batched envs.
    """
    if row_generators is None:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    out = torch.empty(probs.shape[0], dtype=torch.long, device=probs.device)
    for idx, gen in enumerate(row_generators):
        row_p = probs[idx:idx + 1]
        out[idx] = torch.multinomial(row_p, num_samples=1, generator=gen).squeeze()
    return out


def bernoulli_sample(
    prob: torch.Tensor,
    row_generators: Any | None,
) -> torch.Tensor:
    """Sample a 0/1 bit per row from a Bernoulli probability (B,)."""
    if row_generators is None:
        return torch.bernoulli(prob).long()
    out = torch.empty(prob.shape[0], dtype=torch.long, device=prob.device)
    for idx, gen in enumerate(row_generators):
        out[idx] = torch.bernoulli(prob[idx:idx + 1], generator=gen).long()
    return out


def gumbel_argmax(logits: torch.Tensor) -> torch.Tensor:
    """argmax over the last dim of (logits + Gumbel noise) → long indices.

    The in-graph sampler used by the ONNX export: ``torch.rand_like`` traces to
    RandomUniformLike, so this is a trace-safe Gumbel-max categorical sample that
    needs no Python-side generator. Equivalent in distribution to a categorical
    draw from softmax(logits).
    """
    u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
    gumbel = -torch.log(-torch.log(u))
    return (logits + gumbel).argmax(dim=-1)


# ── Attack bit decode ────────────────────────────────────────────────

def decode_attack_bit(
    attack_logit: torch.Tensor,
    *,
    sampled: bool,
    temperature: float = 1.0,
    threshold: float = 0.5,
    row_generators: Any | None = None,
) -> torch.Tensor:
    """Attack head logit → 0/1 fire bit (B,) int64.

    Greedy (``sampled=False``): sigmoid(logit) > ``threshold`` — the deterministic
    threshold readout. Sampled (``sampled=True``): temperature-modulate the logit
    so prob(fire) = sigmoid((logit − bias) / T), then a per-row Bernoulli draw.

    ``threshold`` is the fire operating point (default 0.5). It enters BOTH paths
    as the equivalent logit bias ``logit(threshold) = ln(θ/(1−θ))`` so greedy and
    sampled share one operating point; θ=0.5 → bias 0, identical to the historical
    decode. It is fit offline (``qnn.bc.decode_fit.fit_attack``) and stamped as
    ``decode.attack_threshold`` for the engine, which runs the same greedy cut.

    EXPORT defers this to the C engine (it emits the attack LOGIT and C runs the
    sigmoid threshold), so this is the policy path's readout only.
    """
    flat = attack_logit.reshape(-1)
    thr = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    bias = math.log(thr / (1.0 - thr))                       # θ=0.5 → 0.0
    if not sampled:
        return (flat > bias).long()                          # sigmoid(flat) > θ
    fire_prob = torch.sigmoid((flat - bias) / max(float(temperature), 1e-6))
    return bernoulli_sample(fire_prob, row_generators)


# ── Move per-axis decode (basic readout) ─────────────────────────────

def decode_move_axes(
    move_logits: torch.Tensor,
    *,
    sampled: bool,
    temperature: float = 1.0,
    row_generators: Any | None = None,
) -> torch.Tensor:
    """Per-axis move decode → class indices (n_rows, n_axes) int64.

    Each axis is a softmax over {neg, none, pos}. Greedy = per-axis argmax;
    sampled = per-axis categorical draw (the calibrated "sample don't argmax"
    readout). ``move_logits`` is (n_rows, n_axes, n_classes).

    This is the BASIC axis readout only. The a24-recent layers built on top of
    it — sticky hysteresis, the semi-Markov hazard supplement, switch-back
    watermarking, stop-onset suppression — stay inline in QNNPolicy.act (they
    thread per-episode instance state and are not cross-gen base). EXPORT defers
    the move decode to the C engine entirely.
    """
    n_rows = int(move_logits.shape[0])
    n_axes = int(move_logits.shape[1])
    n_classes = int(move_logits.shape[2])
    if not sampled:
        return torch.argmax(move_logits, dim=-1)               # (n_rows, n_axes)
    move_probs = F.softmax(move_logits / max(float(temperature), 1e-6), dim=-1)
    # Flatten axes into the batch dim so the row-wise sampler can run, reshape
    # back after.
    flat_probs = move_probs.reshape(-1, n_classes)             # (n_rows*n_axes, n_classes)
    if row_generators is None:
        flat_classes = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
    else:
        flat_classes = torch.empty(
            flat_probs.shape[0], dtype=torch.long, device=flat_probs.device)
        for axis_idx in range(flat_probs.shape[0]):
            row_idx = axis_idx // n_axes
            gen = row_generators[row_idx]
            flat_classes[axis_idx] = torch.multinomial(
                flat_probs[axis_idx:axis_idx + 1], num_samples=1, generator=gen,
            ).squeeze()
    return flat_classes.reshape(n_rows, n_axes)
