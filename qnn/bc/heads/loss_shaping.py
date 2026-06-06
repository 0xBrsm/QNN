"""Shared per-frame loss-shaping helpers for binary-class heads.

Both the fire head and the jump (move ud-axis) head fit a binary
positive/negative decision against a label stream where the demonstrator's
button press doesn't always land on the geometrically-ideal frame: humans
react late, releases linger, and the labeling pipeline downsamples 70 Hz
native press events into 20 Hz training frames. The result is a label
stream with ~14% of false positives sitting one frame off a true positive
(see ``scripts/analysis/fire_fp_timing.py``).

Distance-weighted BCE smooths the loss landscape near label edges so the
gradient on a wrong-by-one-frame prediction is much smaller than the
gradient on a wrong-by-100-frames prediction, without changing the binary
labels themselves. Inference stays unchanged: same head, same threshold.

The same helper drives fire and jump because both are 0/1 streams with
the same human-timing-noise structure. Tune sigma per head.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def distance_weighted_neg_weights(
    target: torch.Tensor,
    valid: torch.Tensor | None,
    sigma_frames: float,
    kernel_radius: int | None = None,
) -> torch.Tensor:
    """Per-frame loss weight for binary BCE with a Gaussian negative shoulder.

    Given a (T, B) binary target stream, returns a (T, B) weight tensor:

      * Frames where ``target == 1``: weight 1.0 — positives are never
        down-weighted; the head still pays full BCE on missed presses.
      * Frames where ``target == 0`` and a positive sits ``d`` frames away:
        weight ``1 - exp(-d^2 / (2 * sigma^2))`` (multiple nearby
        positives sum then clamp to 1.0). Adjacent-to-press FPs get near-
        zero loss; FPs far from any press get full loss.

    The convolution treats the time axis (axis 0) as the sequence and
    runs independently per batch lane. Episode boundaries inside a BPTT
    window are handled by zeroing the target through ``valid``: a
    positive in an invalid frame does not contribute to a neighbor's
    weight. Cross-episode contamination within a sequence-length chunk
    is therefore upper-bounded by the kernel radius.

    Args:
        target: ``(T, B)`` 0/1 float or bool tensor.
        valid:  ``(T, B)`` bool tensor of frames inside the loss mask, or
                None for no masking.
        sigma_frames: Gaussian width in frames. Caller picks this from
                the timing-FP histogram (e.g. fire ~3 at 20 Hz).
        kernel_radius: half-width of the conv kernel. Defaults to
                ``ceil(3 * sigma_frames)`` which captures >99% of mass.

    Returns:
        ``(T, B)`` float weights in [0, 1], same device/dtype family as
        ``target``.
    """
    if target.ndim != 2:
        raise ValueError(
            f"distance_weighted_neg_weights expects (T, B) target; got shape {tuple(target.shape)}"
        )
    if sigma_frames <= 0.0:
        raise ValueError(f"sigma_frames must be > 0, got {sigma_frames}")
    if kernel_radius is None:
        kernel_radius = max(1, int(math.ceil(3.0 * sigma_frames)))
    T, B = target.shape
    device = target.device
    target_f = target.to(torch.float32)
    if valid is not None:
        target_f = target_f * valid.to(torch.float32)

    xs = torch.arange(-kernel_radius, kernel_radius + 1,
                      device=device, dtype=torch.float32)
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma_frames ** 2))

    # Conv1d expects (N, C, L). Treat each batch lane as a separate
    # signal: (B, 1, T) -> conv -> (B, 1, T).
    inp = target_f.transpose(0, 1).unsqueeze(1).contiguous()
    smoothed = F.conv1d(inp, kernel.view(1, 1, -1), padding=kernel_radius)
    smoothed = smoothed.squeeze(1).transpose(0, 1).clamp(0.0, 1.0)  # (T, B)

    neg_weight = 1.0 - smoothed
    # Positives always carry full weight — we never want to down-weight a
    # true positive's BCE just because it's adjacent to another positive.
    pos_mask = target > 0.5
    out = torch.where(pos_mask, torch.ones_like(neg_weight), neg_weight)
    return out.to(target.dtype if target.is_floating_point() else torch.float32)


# ── Flat-batch path ──────────────────────────────────────────────
#
# The GPU-resident supervised loop uses frame-shuffled SGD: each batch
# is N random frames from across the entire dataset, with no time
# axis. The conv-based distance_weighted_neg_weights helper above
# can't be used in that regime (it needs a (T, B) signal).
#
# Workaround: precompute the per-frame distance-to-nearest-positive
# ONCE at preload time (per episode, never crossing episode bounds),
# carry it alongside the binary target, and at training time do a
# pure pointwise weight = 1 - exp(-d²/(2σ²)) at the sampled frames.
# Same loss semantics as the conv path; no time axis required.


def per_frame_distance_to_pos(target: np.ndarray, large: float = 1e6) -> np.ndarray:
    """Per-frame distance (in frames) to the nearest positive within this episode.

    Args:
        target: 1-D array of 0/1 (or float) values for a single episode.
        large:  value to fill when the episode has no positives at all.
                Default is large enough that any reasonable sigma gives
                weight ≈ 1 (full BCE) on those frames.

    Returns:
        ``(n,)`` float32 array of non-negative distances. Positive frames
        themselves get distance 0.
    """
    arr = np.asarray(target).reshape(-1)
    pos_idx = np.flatnonzero(arr > 0.5)
    n = arr.shape[0]
    if pos_idx.size == 0:
        return np.full(n, large, dtype=np.float32)
    # For each frame, find the nearest entry in pos_idx via searchsorted.
    frames = np.arange(n)
    insert = np.searchsorted(pos_idx, frames)
    left  = pos_idx[np.clip(insert - 1, 0, pos_idx.size - 1)]
    right = pos_idx[np.clip(insert,     0, pos_idx.size - 1)]
    return np.minimum(np.abs(frames - left), np.abs(frames - right)).astype(np.float32)


def flat_distance_weight(
    distance: torch.Tensor,
    target: torch.Tensor,
    sigma_frames: float,
) -> torch.Tensor:
    """Per-frame BCE/CE weight for a flat (B,) batch given precomputed distance.

    weight = 1.0 on positives (target > 0.5)
    weight = 1 - exp(-d² / (2 σ²)) on negatives

    Equivalent to ``distance_weighted_neg_weights`` evaluated at the
    sampled frames, but with the time-axis convolution amortized into
    a one-time precompute (see ``per_frame_distance_to_pos``).
    """
    if sigma_frames <= 0.0:
        raise ValueError(f"sigma_frames must be > 0, got {sigma_frames}")
    d = distance.to(torch.float32)
    neg_w = 1.0 - torch.exp(-(d ** 2) / (2.0 * sigma_frames ** 2))
    pos_mask = target > 0.5
    out = torch.where(pos_mask, torch.ones_like(neg_w), neg_w)
    return out.to(target.dtype if target.is_floating_point() else torch.float32)
