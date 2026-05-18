"""Observation buffer format — v9 variable-length entity stream.

Single source of truth for the obs buffer wire format in Python.
Must match qnn_io.h on the C side.

Wire layout (all offsets fixed, little-endian):
  0..95    self section (scalars + embed IDs)
  96..563  spatial tokens (9 × 13 float32)
  564..819 action history (8 × 8 float32)
  820..    entity stream (variable-length, type-tagged tokens)

See OBS_SCHEMA in qnn/schema.py for the canonical
model-facing dict shape contract.
"""

import struct
from typing import Dict, List, Tuple

import numpy as np

from qnn.vocab import (
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    PROJECTILE_SCALAR_DIM, ACTOR_SCALAR_DIM, ITEM_SCALAR_DIM, MOVER_SCALAR_DIM,
    PROJECTILE_ID_DIM, ACTOR_ID_DIM, ITEM_ID_DIM, MOVER_ID_DIM,
    MAX_ENTITY_EVENTS, MAX_TOKEN_OBJECTS,
)

MAX_ENTITY_SCALAR_DIM = ACTOR_SCALAR_DIM  # largest per-type scalar count
MAX_ENTITY_ID_DIM = ACTOR_ID_DIM          # largest per-type ID count

OBS_BUFFER_SIZE = 4096
ACTION_SIZE = 32  # sizeof(qnn_action_t) — move[3] + look[3] + fire + switch

# Per-tick header emitted by demo worker collect mode.
TICK_HEADER_SIZE = 16
TICK_HEADER_FORMAT = "<IIIHH"
TICK_MAGIC = b"QOBS"
TICK_MAGIC_SIZE = 4
FLAG_RESET = 0x01
FLAG_DONE = 0x02

# LOBS — labeler-mode slim per-native-tick frame.  See qnn.h
# QNN_EmitLabelerTick docstring for the authoritative layout.
LABELER_MAGIC = b"LOBS"
LABELER_FRAME_SIZE = 33  # 4 magic + 10 header + 19 payload

# Self token: fixed layout at offset 0 (96 bytes)
SELF_SCALAR_DIM = 16
SPATIAL_TOKEN_COUNT = 9
SPATIAL_SCALAR_DIM = 13  # dir[3] + 10 measurement scalars

SELF_FIELDS = {
    "self_scalars":     (0,  np.float32, (16,)),
    "self_weapon_id":   (64, np.int32,   (1,)),
    "self_armor_type_id": (68, np.int32, (1,)),
    "self_powerup_ids": (72, np.int32,   (5,)),
    "self_movement_id": (92, np.int32,   (1,)),
}

SPATIAL_OFFSET = 96
SPATIAL_STRIDE = 52  # 13 float32
ENTITY_STREAM_OFFSET = 564

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
    count = SPATIAL_TOKEN_COUNT * SPATIAL_SCALAR_DIM
    scalars = np.frombuffer(raw, dtype=np.float32, offset=offset, count=count).reshape(
        SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM
    )
    return scalars.copy(), offset + count * 4


def unpack_entity_stream_dense(raw: bytes, offset: int) -> Tuple[dict[str, np.ndarray], int]:
    """Parse the entity stream directly into fixed-size dense arrays."""
    pos = offset
    if pos >= len(raw):
        return densify_entity_tokens([]), pos

    n_tokens = min(int(raw[pos]), MAX_TOKEN_OBJECTS)
    pos += 1

    types = np.full((MAX_TOKEN_OBJECTS,), -1, dtype=np.int32)
    scalars = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_SCALAR_DIM), dtype=np.float32)
    ids = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_ID_DIM), dtype=np.int32)
    evt_actions = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS), dtype=np.int32)
    evt_sources = np.zeros((MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS), dtype=np.int32)
    evt_counts = np.zeros((MAX_TOKEN_OBJECTS,), dtype=np.uint8)

    slot = 0
    while slot < n_tokens and pos < len(raw):
        tok_type = raw[pos]
        pos += 1
        if tok_type not in TOKEN_LAYOUT:
            break

        n_ids, n_scalars = TOKEN_LAYOUT[tok_type]
        types[slot] = tok_type

        tok_ids = np.frombuffer(raw, dtype=np.int32, offset=pos, count=n_ids)
        ids[slot, :n_ids] = tok_ids
        pos += n_ids * 4

        tok_scalars = np.frombuffer(raw, dtype=np.float32, offset=pos, count=n_scalars)
        scalars[slot, :n_scalars] = tok_scalars
        pos += n_scalars * 4

        if pos >= len(raw):
            break
        n_events = min(int(raw[pos]), MAX_ENTITY_EVENTS)
        pos += 1
        evt_counts[slot] = n_events
        if n_events > 0:
            event_values = np.frombuffer(raw, dtype=np.int32, offset=pos, count=n_events * 2).reshape(n_events, 2)
            evt_actions[slot, :n_events] = event_values[:, 0]
            evt_sources[slot, :n_events] = event_values[:, 1]
            pos += n_events * 8

        slot += 1

    return {
        "entity_types": types,
        "entity_scalars_raw": scalars,
        "entity_ids": ids,
        "entity_event_actions": evt_actions,
        "entity_event_sources": evt_sources,
        "entity_event_counts": evt_counts,
    }, pos


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


def unpack_labeler_buffer(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack an 18-byte LOBS payload (everything after the 4-byte magic
    and 10-byte header) into a model-facing dict.

    Layout (matches C-side QNN_EmitLabelerTick in qnn_collect_helpers.c):
        pos_delta_vel[3]     fp16   offset 0,  6 bytes (body-frame, normalized)
        movement_id          u8     offset 6,  1 byte
        view_delta[3]        fp16   offset 7,  6 bytes
        c_rule_fire          u8     offset 13, 1 byte (engine-side sound+ammo)
        c_rule_jump          u8     offset 14, 1 byte
        move_packed          u8     offset 15, 1 byte (fb | lr<<2 | ud<<4, target)
        target_valid_mask    u8     offset 16, 1 byte (bit0=fb, bit1=lr,
                                                       bit2=ud, bit3=fire,
                                                       bit4=weapon; bits5..7
                                                       reserved)
        usercmd_fire         u8     offset 17, 1 byte (press truth: usercmd
                                                       fire button OR'd
                                                       across the cmd window)
        weapon_id            u8     offset 18, 1 byte (held weapon 1..8,
                                                       0 = none)
    """
    if len(raw) < 19:
        raise ValueError(f"labeler payload too short: {len(raw)} bytes")
    pos_delta_vel = np.frombuffer(raw, dtype=np.float16, offset=0,  count=3).astype(np.float32)
    movement_id   = int(raw[6])
    view_delta    = np.frombuffer(raw, dtype=np.float16, offset=7,  count=3).astype(np.float32)
    c_rule_fire   = int(raw[13])
    c_rule_jump   = int(raw[14])
    move_packed   = int(raw[15])
    target_valid_mask = int(raw[16])
    usercmd_fire  = int(raw[17])
    weapon_id     = int(raw[18])
    return {
        "pos_delta_vel": pos_delta_vel,
        "movement_id":   movement_id,
        "view_delta":    view_delta,
        "c_rule_fire":   c_rule_fire,
        "c_rule_jump":   c_rule_jump,
        "move_packed":   move_packed,
        "target_valid_mask": target_valid_mask,
        "usercmd_fire":  usercmd_fire,
        "weapon_id":     weapon_id,
    }


def unpack_obs_buffer(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack a v9 obs buffer into a model-facing dict.

    Returns dict with:
        self_scalars, self_weapon_id, etc. (fixed fields)
        entity_types, entity_scalars_raw, entity_ids, entity_event_* (dense)
        spatial_scalars
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

    # Entity stream → dense arrays
    dense_entities, _ = unpack_entity_stream_dense(raw, ENTITY_STREAM_OFFSET)
    obs.update(dense_entities)

    return obs


# ---- Action struct: move[3] + look[3] + fire + weapon = 32 bytes ----
#
# The 4-byte int32 at offset 28 carries the raw engine weapon byte
# (action->switch_slot on the C side — still named for historical
# reasons; the field carries an impulse-form weapon id 0..8, where 0
# means no weapon held).  The Python-side key here is the new
# canonical name and what bc/collect writes to disk.

ACTION_FIELDS = {
    "move":     (0,  np.float32, (3,)),
    "look":     (12, np.float32, (3,)),
    "fire":     (24, np.int32,   ()),
    "weapon":   (28, np.int32,   ()),
}
