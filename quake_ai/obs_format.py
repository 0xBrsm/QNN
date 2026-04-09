"""Observation buffer format — v9 variable-length entity stream.

Single source of truth for the obs buffer wire format in Python.
Must match qnn_io.h on the C side.

Wire layout (all offsets fixed, little-endian):
  0..87    self section (scalars + embed IDs)
  88..555  spatial tokens (9 × 13 float32)
  556..811 action history (8 × 8 float32)
  812..    entity stream (variable-length, type-tagged tokens)

See OBS_SCHEMA in quake_ai/model/observation.py for the canonical
model-facing dict shape contract.
"""

import struct
from typing import Dict, List, Tuple

import numpy as np

from quake_ai.vocab import (
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    PROJECTILE_SCALAR_DIM, ACTOR_SCALAR_DIM, ITEM_SCALAR_DIM, MOVER_SCALAR_DIM,
    PROJECTILE_ID_DIM, ACTOR_ID_DIM, ITEM_ID_DIM, MOVER_ID_DIM,
    MAX_ENTITY_EVENTS, MAX_TOKEN_OBJECTS,
)

MAX_ENTITY_SCALAR_DIM = ACTOR_SCALAR_DIM  # largest per-type scalar count
MAX_ENTITY_ID_DIM = ACTOR_ID_DIM          # largest per-type ID count

OBS_BUFFER_SIZE = 4096
ACTION_SIZE = 48  # sizeof(qnn_action_t) — move[2] + look[3] + fire + jump + switch + recall[4]

# Per-tick header emitted by demo worker collect mode.
TICK_HEADER_SIZE = 16
TICK_HEADER_FORMAT = "<IIIHH"
TICK_MAGIC = b"QOBS"
TICK_MAGIC_SIZE = 4
FLAG_RESET = 0x01
FLAG_DONE = 0x02

# Self token: fixed layout at offset 0 (96 bytes)
SELF_SCALAR_DIM = 14
SPATIAL_TOKEN_COUNT = 9
SPATIAL_SCALAR_DIM = 13  # dir[3] + 10 measurement scalars
ACTION_HISTORY_LEN = 8
ACTION_HISTORY_DIM = 8  # move[2] + look[3] + fire + jump + switch

SELF_FIELDS = {
    "self_scalars":     (0,  np.float32, (14,)),
    "self_weapon_id":   (56, np.int32,   (1,)),
    "self_armor_type_id": (60, np.int32, (1,)),
    "self_powerup_ids": (64, np.int32,   (5,)),
    "self_movement_id": (84, np.int32,   (1,)),
}

SPATIAL_OFFSET = 88
SPATIAL_STRIDE = 52  # 13 float32
ACTION_HISTORY_OFFSET = 556
ENTITY_STREAM_OFFSET = 812

# Per-type wire layout: (n_ids, n_scalars)
TOKEN_LAYOUT = {
    TOKEN_PROJECTILE: (PROJECTILE_ID_DIM, PROJECTILE_SCALAR_DIM),
    TOKEN_ACTOR:      (ACTOR_ID_DIM,      ACTOR_SCALAR_DIM),
    TOKEN_ITEM:       (ITEM_ID_DIM,       ITEM_SCALAR_DIM),
    TOKEN_MOVER:      (MOVER_ID_DIM,      MOVER_SCALAR_DIM),
}


def unpack_entity_stream(raw: bytes, offset: int) -> Tuple[List[dict], int]:
    """Parse the variable-length entity token stream.

    Returns (tokens, end_offset) where tokens is a list of dicts with:
        type: int (TOKEN_*)
        ids: np.ndarray of int32
        scalars: np.ndarray of float32
        events: list of (action_id, source_id) tuples
    """
    pos = offset
    n_tokens = raw[pos]
    pos += 1
    tokens = []

    for _ in range(n_tokens):
        tok_type = raw[pos]
        pos += 1

        if tok_type not in TOKEN_LAYOUT:
            break  # unknown type, stop parsing

        n_ids, n_scalars = TOKEN_LAYOUT[tok_type]

        # Read IDs
        ids = np.frombuffer(raw, dtype=np.int32, offset=pos, count=n_ids).copy()
        pos += n_ids * 4

        # Read scalars
        scalars = np.frombuffer(raw, dtype=np.float32, offset=pos, count=n_scalars).copy()
        pos += n_scalars * 4

        # Read events
        n_events = raw[pos]
        pos += 1
        events = []
        for _ in range(n_events):
            action_id = struct.unpack_from("<i", raw, pos)[0]
            pos += 4
            source_id = struct.unpack_from("<i", raw, pos)[0]
            pos += 4
            events.append((action_id, source_id))

        tokens.append({
            "type": tok_type,
            "ids": ids,
            "scalars": scalars,
            "events": events,
        })

    return tokens, pos


def unpack_spatial(raw: bytes, offset: int) -> Tuple[np.ndarray, int]:
    """Parse spatial tokens (fixed: 9 sectors × 13 float32 scalars)."""
    pos = offset
    scalars = np.zeros((SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM), dtype=np.float32)

    for i in range(SPATIAL_TOKEN_COUNT):
        for j in range(SPATIAL_SCALAR_DIM):
            scalars[i, j] = struct.unpack_from("<f", raw, pos)[0]
            pos += 4

    return scalars, pos


def unpack_action_history(raw: bytes, offset: int, count: int) -> Tuple[np.ndarray, int]:
    """Parse action history (up to 8 × 8 floats)."""
    pos = offset
    history = np.zeros((ACTION_HISTORY_LEN, ACTION_HISTORY_DIM), dtype=np.float32)

    for i in range(min(count, ACTION_HISTORY_LEN)):
        for j in range(ACTION_HISTORY_DIM):
            history[i, j] = struct.unpack_from("<f", raw, pos)[0]
            pos += 4

    return history, pos


def densify_entity_tokens(tokens: list[dict]) -> dict[str, np.ndarray]:
    """Convert variable-length entity token list to fixed-size numpy arrays.

    Returns:
        entity_types: (16,) int32 — type tag per slot, -1 for empty
        entity_scalars_raw: (16, MAX_ENTITY_SCALAR_DIM) float32 — zero-padded raw scalars
        entity_ids: (16, MAX_ENTITY_ID_DIM) int32
        entity_event_actions: (16, MAX_ENTITY_EVENTS) int32
        entity_event_sources: (16, MAX_ENTITY_EVENTS) int32
        entity_event_counts: (16,) uint8
    """
    n = min(len(tokens), MAX_TOKEN_OBJECTS)

    types = np.full((MAX_TOKEN_OBJECTS,), -1, dtype=np.int32)
    scalars = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_SCALAR_DIM), dtype=np.float32)
    ids = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_ID_DIM), dtype=np.int32)
    evt_actions = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS), dtype=np.int32)
    evt_sources = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS), dtype=np.int32)
    evt_counts = np.zeros((MAX_TOKEN_OBJECTS,), dtype=np.uint8)

    for i in range(n):
        tok = tokens[i]
        types[i] = tok["type"]

        tok_scalars = tok["scalars"]
        sdim = len(tok_scalars)
        scalars[i, :sdim] = tok_scalars

        tok_ids = tok["ids"]
        idim = min(len(tok_ids), MAX_ENTITY_ID_DIM)
        ids[i, :idim] = tok_ids[:idim]

        events = tok["events"]
        ne = min(len(events), MAX_ENTITY_EVENTS)
        evt_counts[i] = ne
        for j in range(ne):
            evt_actions[i, j] = events[j][0]
            evt_sources[i, j] = events[j][1]

    return {
        "entity_types": types,
        "entity_scalars_raw": scalars,
        "entity_ids": ids,
        "entity_event_actions": evt_actions,
        "entity_event_sources": evt_sources,
        "entity_event_counts": evt_counts,
    }


def unpack_obs_buffer(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack a v9 obs buffer into a model-facing dict.

    Returns dict with:
        self_scalars, self_weapon_id, etc. (fixed fields)
        entity_types, entity_scalars_raw, entity_ids, entity_event_* (dense)
        spatial_scalars, action_history
    """
    obs: dict[str, np.ndarray] = {}

    # Self (fixed)
    for name, (offset, dtype, shape) in SELF_FIELDS.items():
        count = int(np.prod(shape))
        arr = np.frombuffer(raw, dtype=dtype, offset=offset, count=count).reshape(shape)
        obs[name] = arr.copy()

    # Spatial (fixed offset)
    spatial_scalars, _ = unpack_spatial(raw, SPATIAL_OFFSET)
    obs["spatial_scalars"] = spatial_scalars

    # Action history (fixed offset)
    action_history, _ = unpack_action_history(raw, ACTION_HISTORY_OFFSET, ACTION_HISTORY_LEN)
    obs["action_history"] = action_history

    # Entity stream → dense arrays
    tokens, _ = unpack_entity_stream(raw, ENTITY_STREAM_OFFSET)
    obs.update(densify_entity_tokens(tokens))

    return obs


# ---- Action struct (unchanged) ----

ACTION_FIELDS = {
    "move":     (0,  np.float32, (2,)),
    "look":     (8,  np.float32, (3,)),
    "fire":     (20, np.int32,   ()),
    "jump":     (24, np.int32,   ()),
    "switch":   (28, np.int32,   ()),
    "recall_0": (32, np.int32,   ()),
    "recall_1": (36, np.int32,   ()),
    "recall_2": (40, np.int32,   ()),
    "recall_3": (44, np.int32,   ()),
}
