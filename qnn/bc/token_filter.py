"""Token-level masking for BC training obs.

Drops entity-token slots at train time via a MongoDB-style predicate,
parallel to ``segment_mask`` (which acts on per-frame fields).  This
replaces the deprecated collect-time ``--entity-filter pvs_actors`` flag:
collects always emit the full 4-pool token stream, and any subsetting
happens at train time via this mask.

Field paths in the per-token namespace:

    type      → entity_types[:, :]       int8, see qnn.vocab.TOKEN_*
    modality  → entity_ids[:, :, 1]      uint8, see qnn.vocab.MODALITY_IDS
    pid       → entity_ids[:, :, 2]      uint8, player id (0 = phantom)
    route_idx → entity_ids[:, :, 0]      uint8, multi-route alternative

Token type IDs (qnn.vocab):
    0 = PROJECTILE, 1 = ACTOR, 2 = ITEM, 3 = MOVER

Modality IDs (qnn_vocab.h):
    0 = SIGHT       — visible in PVS (actors, items, projectiles, movers)
    1 = PROXIMITY   — nearby & not LOS (items / projectiles / movers only;
                      actors never use this modality)
    2 = SOUND       — heard but not seen (actors only)
    3 = MEMORY      — last-known position after sight loss (actors only)

Slots where the predicate evaluates to False are zeroed in place:

    entity_types[slot]       = -1   (the empty sentinel)
    entity_ids[slot, :]      = 0
    entity_scalars_raw[slot] = 0
    entity_event_*[slot]     = 0

Slot positions are preserved — no compaction — so target labels remain
valid.  A label pointing into a now-masked slot becomes the caller's
problem; the BC train loop sets the affected ``target`` row to -100 so
CE skips it.

Equivalent to the deprecated ``entity_filter=pvs_actors`` collect flag::

    token_mask = {
        "type":     1,                # TOKEN_ACTOR
        "pid":      {"$gt": 0},
        "modality": 0,                # SIGHT — actors never get PROXIMITY,
                                      #   and we drop SOUND/MEMORY here
    }
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from qnn import filter_dsl


_TOKEN_SLOT_KEYS = (
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
    """Return a copy of ``obs`` with non-matching token slots zeroed.

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
            f"token_mask predicate must produce a per-slot bool array of "
            f"shape {expected}; got {keep.shape}"
        )
    if keep.all():
        return obs_out

    drop = ~keep
    for key in _TOKEN_SLOT_KEYS:
        if key not in obs:
            continue
        arr = np.asarray(obs[key]).copy()
        # entity_types uses -1 as the empty sentinel (TOKEN_PROJECTILE=0,
        # so plain 0 would silently relabel padded slots as projectiles).
        fill = -1 if key == "entity_types" else 0
        if arr.ndim == 2:
            arr[drop] = fill
        else:
            arr[drop, ...] = fill
        obs_out[key] = arr
    return obs_out


def clear_targets_on_masked_slots(
    target: np.ndarray,
    obs: Mapping[str, np.ndarray],
    ignore_index: int = -100,
) -> np.ndarray:
    """Return a copy of ``target`` with rows pointing to masked slots
    replaced with ``ignore_index``.

    A slot is "masked" iff its ``entity_types`` is the -1 empty sentinel.
    Useful right after ``apply_token_mask`` so the target head doesn't
    try to predict a slot that's been zeroed.
    """
    target = np.asarray(target).copy()
    valid = target != ignore_index
    if not valid.any():
        return target
    types = np.asarray(obs["entity_types"])  # (T, N) int8
    rows = np.arange(types.shape[0])
    # Bound the slot index so np.take doesn't IndexError when target == -100
    # (we mask those out below anyway).
    bounded = np.clip(target, 0, types.shape[1] - 1)
    slot_type = types[rows, bounded]
    target[valid & (slot_type == -1)] = ignore_index
    return target
