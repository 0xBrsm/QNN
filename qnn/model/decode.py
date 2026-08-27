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
  * The gen-SPECIFIC decode — the polar-look hybrid, the aim-prior blend, the
    a25 move commitment + attack_with joint decode — lives in the generation's
    OWN facade (:mod:`qnn.model.decode_actions`). Those are generation CHOICES,
    not cross-gen invariants; a future generation replaces that module. (The a24
    sticky/hazard/switch-back stack and its facade were retired with the a24 arch.)
  * The polar GEOMETRY (``tangent_expmap`` / ``MAG_CENTERS`` / ``DIR_CENTERS``)
    lives in :mod:`qnn.model.look_bins`; the lead/aim primitives in
    :mod:`qnn.model.lead_aim`. This module imports neither — it is pure readout.

EXPORT note: the export path decodes in-graph via the a25 facade;
:func:`decode_attack_bit` / :func:`decode_move_axes` are called by
``QNNPolicy.act`` only (the split-head fallback and the non-commitment move
readout). Base membership is about cross-gen STABILITY, not about being called
by both surfaces.

TRACE-SAFETY: keep this module dependency-light (torch only) and free of
``.item()``, data-dependent Python control flow, and advanced/boolean tensor
indexing, so anything reused by a traced graph stays ONNX-traceable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


# ── Sampling primitives ─────────────────────────────────────────────
# Stateless readout draws shared by every head decode. ``row_generators`` may
# be one torch.Generator per row (eval needs episode streams to stay invariant
# to batching), or one BatchedRNG (PPO needs the same independent draws but can
# generate the whole B×N block in one dispatcher call).


@dataclass(slots=True)
class BatchedRNG:
    """One generator producing vectorized draws for a fixed-size batch.

    PPO trajectories only need a deterministic stream for the fixed collector
    topology; unlike eval, they do not require each row's stream to survive a
    change in batching.  Keeping that distinction explicit avoids hundreds of
    tiny ``torch.rand`` dispatcher calls per policy tick.
    """

    generator: torch.Generator
    batch_size: int

    @classmethod
    def seeded(cls, seed: int, batch_size: int, device: torch.device | str) -> "BatchedRNG":
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(int(seed))
        return cls(generator=generator, batch_size=int(batch_size))

def row_uniforms(row_generators: Any, n: int, device) -> torch.Tensor:
    """(B, n) uniforms, row i drawn ONLY from row_generators[i].

    The one per-row loop the sampling path keeps: a bare ``torch.rand``
    per generator (~µs) instead of per-row multinomial machinery.
    Per-lane streams stay independent and seed-reproducible; downstream
    sampling is vectorized inverse-CDF over these uniforms.
    """
    if isinstance(row_generators, BatchedRNG):
        gdev = getattr(row_generators.generator, "device", None) or torch.device("cpu")
        return torch.rand(
            (row_generators.batch_size, n),
            generator=row_generators.generator,
            device=gdev,
        ).to(device)

    gdev = getattr(row_generators[0], "device", None) or torch.device("cpu")
    u = torch.empty(len(row_generators), n, device=gdev)
    for i, gen in enumerate(row_generators):
        torch.rand((n,), generator=gen, out=u[i])
    return u.to(device)


def inverse_cdf_sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Vectorized categorical draw: (B, K) probs × (B,) uniforms → (B,) idx."""
    cdf = probs.cumsum(-1)
    cdf = cdf / cdf[..., -1:].clamp_min(1e-12)
    idx = torch.searchsorted(cdf, u.reshape(-1, 1).contiguous()).squeeze(-1)
    return idx.clamp_(max=probs.shape[-1] - 1)


def categorical_sample(
    probs: torch.Tensor,
    row_generators: Any | None,
) -> torch.Tensor:
    """Sample one class index per row from probs (B, K).

    When row_generators is None, uses default RNG. When provided, each
    row's draw consumes ONLY that row's generator (independent per-episode
    streams, reproducible across batched envs) — one uniform per row, then
    a batched inverse-CDF instead of per-row multinomial calls. Same law
    and independence contract as the old per-row multinomial, but
    different draw values for a given seed: sampled trajectories are not
    comparable across this boundary (2026-07-11).
    """
    if row_generators is None:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    u = row_uniforms(row_generators, 1, probs.device)[:, 0]
    return inverse_cdf_sample(probs, u)


def bernoulli_sample(
    prob: torch.Tensor,
    row_generators: Any | None,
) -> torch.Tensor:
    """Sample a 0/1 bit per row from a Bernoulli probability (B,)."""
    if row_generators is None:
        return torch.bernoulli(prob).long()
    u = row_uniforms(row_generators, 1, prob.device)[:, 0]
    return (u < prob).long()


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
    decode. (The offline threshold FIT and its engine stamp were retired with the
    a24 arch — the a25 attack decode is the 9-way attack_with joint head; this
    readout remains for split-head models in tests/diagnostics.)
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

    This is the BASIC axis readout only — and, with the a24 sticky/hazard stack
    retired, the ONLY non-commitment move decode: QNNPolicy.act runs either the
    a25 commitment decode (move_seg models) or this plain per-axis readout.
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
    elif isinstance(row_generators, BatchedRNG):
        uniforms = row_uniforms(row_generators, n_axes, flat_probs.device)
        flat_classes = inverse_cdf_sample(flat_probs, uniforms.reshape(-1))
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
