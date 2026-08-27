"""Token-level masking for BC training obs.

Drops entity-token indices at train time via a MongoDB-style predicate,
parallel to ``segment_mask`` (which acts on per-frame fields).  This
replaces the deprecated collect-time ``--entity-filter pvs_actors`` flag.
A27 collects emit the two-pool actor/projectile combat stream, and any further
subsetting happens at train time via this mask.

Field paths in the per-token namespace:

    type      → entity_types[:, :]       int8, see qnn.vocab.TOKEN_*
    modality  → entity_ids[:, :, 1]      uint8, see qnn.vocab.MODALITY_IDS
    pid       → entity_ids[:, :, 2]      uint8, player id (0 = phantom)
    route_idx → entity_ids[:, :, 0]      uint8, multi-route alternative

Token type IDs (qnn.vocab):
    0 = PROJECTILE, 1 = ACTOR

Modality IDs (qnn_vocab.h):
    0 = SIGHT       — current view cone plus unobstructed trace
    1 = PROXIMITY   — current PVS state that did not qualify for SIGHT

Indices where the predicate evaluates to False are zeroed in place:

    entity_types[idx]       = -1   (the empty sentinel)
    entity_ids[idx, :]      = 0
    entity_scalars_raw[idx] = 0
    entity_event_*[idx]     = 0

Idx positions are preserved — no compaction — so target labels remain
valid.  A label pointing into a now-masked idx becomes the caller's
problem; the BC train loop sets the affected ``target`` row to -100 so
CE skips it.

Equivalent to the deprecated ``entity_filter=pvs_actors`` collect flag::

    token_mask = {
        "type":     1,                # TOKEN_ACTOR
        "pid":      {"$gt": 0},
        "modality": 0,                # SIGHT actors only
    }
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from qnn import filter_dsl


_TOKEN_IDX_KEYS = (
    "entity_types",
    "entity_ids",
    "entity_scalars_raw",
    "entity_event_actions",
    "entity_event_sources",
    "entity_event_counts",
)


def _flatten_per_token_arrays(obs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Build the flat ``field -> (T, N)`` dict consumed by filter_dsl."""
    types = obs["entity_types"]          # (T, N)     int8
    ids   = obs["entity_ids"]            # (T, N, 3)  uint8
    return {
        "type":      np.asarray(types),
        "modality":  np.asarray(ids[:, :, 1]),
        "pid":       np.asarray(ids[:, :, 2]),
        "route_idx": np.asarray(ids[:, :, 0]),
    }


def apply_token_mask(
    obs: Mapping[str, np.ndarray],
    predicate: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Return a copy of ``obs`` with non-matching token indices zeroed.

    ``predicate`` is a MongoDB-style filter_dsl predicate over per-token
    fields.  None or an empty predicate is a no-op (the obs dict is
    shallow-copied so callers can safely mutate the result).
    """
    obs_out = dict(obs)
    if not predicate:
        return obs_out

    flat = _flatten_per_token_arrays(obs)
    keep = np.asarray(filter_dsl.eval_filter(flat, predicate), dtype=bool)
    expected = obs["entity_types"].shape
    if keep.shape != expected:
        raise ValueError(
            f"token_mask predicate must produce a per-idx bool array of "
            f"shape {expected}; got {keep.shape}"
        )
    if keep.all():
        return obs_out

    drop = ~keep
    for key in _TOKEN_IDX_KEYS:
        if key not in obs:
            continue
        arr = np.asarray(obs[key]).copy()
        # entity_types uses -1 as the empty sentinel (TOKEN_PROJECTILE=0,
        # so plain 0 would silently relabel padded indices as projectiles).
        fill = -1 if key == "entity_types" else 0
        if arr.ndim == 2:
            arr[drop] = fill
        else:
            arr[drop, ...] = fill
        obs_out[key] = arr
    return obs_out


def clear_target_probs_on_masked_indices(
    target_probs: np.ndarray,
    obs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return a copy of ``target_probs`` with mass on masked indices moved
    to NO_TARGET (index 0).

    A idx is "masked" iff its ``entity_types`` is the -1 empty sentinel.
    Mass that would otherwise fall on a hidden idx is folded into
    NO_TARGET so row sums stay 1.0 and the target head doesn't get
    gradient pulling toward a idx the model can't see.
    """
    target_probs = np.asarray(target_probs, dtype=np.float32).copy()
    if target_probs.size == 0:
        return target_probs
    types = np.asarray(obs["entity_types"])         # (T, N) int8
    # target_probs[:, 0] = NO_TARGET; target_probs[:, 1:] = idx probabilities.
    masked_idx = (types == -1).astype(np.float32)  # (T, N) — 1 where the idx is gone
    moved = (target_probs[:, 1:] * masked_idx).sum(axis=1)
    target_probs[:, 1:] *= (1.0 - masked_idx)
    target_probs[:, 0] += moved
    return target_probs
