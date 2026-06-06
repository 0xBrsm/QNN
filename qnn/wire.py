"""Observation buffer wire format — engine_norm phase 2 (native).

Single source of truth for the obs buffer wire format in Python.
Must match qnn_io.h on the C side.

The native wire layout (per qnn.engine_norm) is the only format
emitted by the C engine and consumed by the BC collect / training and
the PPO / live bridges. Self/spatial blocks are fixed-offset native
widths; the entity stream is variable-length and type-tagged. See
``unpack_obs_buffer_native`` for the parser entry point.

See OBS_SCHEMA in qnn/schema.py for the model-facing dict shape
contract that downstream code consumes after the
SelfDequantizer/SpatialDequantizer/EntityDequantizer adapt the native
arrays at the ObsEmbedding boundary.
"""

import struct
from typing import Tuple

import numpy as np

from qnn.vocab import (
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    ACTOR_SCALAR_DIM, ACTOR_ID_DIM,
    MAX_ENTITY_EVENTS, MAX_TOKEN_OBJECTS,
)

MAX_ENTITY_SCALAR_DIM = ACTOR_SCALAR_DIM  # largest per-type scalar count
MAX_ENTITY_ID_DIM = ACTOR_ID_DIM          # largest per-type ID count

OBS_BUFFER_SIZE = 4096
# sizeof(qnn_action_t) — move (press byte) + weapon + input_mask + pad +
# look[3]. The press byte mirrors the input_mask bit layout (attack at
# bit 0, fb/lr/ud neg/pos in bits 1-6, jump at bit 7) so the engine's
# in-memory representation matches the on-disk compacted form emitted by
# qnn.bc.collect.
ACTION_SIZE = 16

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
LABELER_FRAME_SIZE = 39  # 4 magic + 10 header + 25 payload

# Legacy-shape constants kept for compat with qnn.schema. Used to
# express the model-facing dense layout after the dequantizers run;
# the wire itself no longer emits these (the native parser produces
# per-field arrays at native widths instead).
SELF_SCALAR_DIM = 17
SPATIAL_TOKEN_COUNT = 9
SPATIAL_SCALAR_DIM = 13  # dir[3] + 10 measurement scalars


def unpack_labeler_buffer(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack a LOBS payload (everything after the 4-byte magic and
    10-byte header) into a model-facing dict.  Returns only the fields
    in the shorter prefix (obs basics + usercmd + op_input); the full
    31-byte payload is unpacked in qnn.labeler.collect._unpack_labeler_episode.

    Layout (matches C-side QNN_EmitLabelerTick in qnn_labeler_collect.c):
        pos_delta_vel[3]     fp16   offset 0,  6 bytes (body-frame, normalized)
        movement_id          u8     offset 6,  1 byte
        cmd_angles[3]        int16  offset 7,  6 bytes (QW 65536/360 quantization)
        cmd_move[3]          int16  offset 13, 6 bytes (fb/lr/ud in QW units)
        cmd_buttons          u8     offset 19, 1 byte (raw button byte)
        cmd_impulse          u8     offset 20, 1 byte (last non-zero impulse)
        op_input             u8     offset 21, 1 byte (strict per-axis op mask:
                                                       bit0=fb, bit1=lr,
                                                       bit2=ud, bit3=fire,
                                                       bit4=impulse.  1 = press
                                                       AND engine acted on it
                                                       this tick.)
    """
    if len(raw) < 22:
        raise ValueError(f"labeler payload too short: {len(raw)} bytes")
    pos_delta_vel = np.frombuffer(raw, dtype=np.float16, offset=0,  count=3).astype(np.float32)
    movement_id   = int(raw[6])
    cmd_angles    = np.frombuffer(raw, dtype=np.int16,  offset=7,  count=3).copy()
    cmd_move      = np.frombuffer(raw, dtype=np.int16,  offset=13, count=3).copy()
    cmd_buttons   = int(raw[19])
    cmd_impulse   = int(raw[20])
    op_input      = int(raw[21])
    return {
        "pos_delta_vel": pos_delta_vel,
        "movement_id":   movement_id,
        "cmd_angles":    cmd_angles,
        "cmd_move":      cmd_move,
        "cmd_buttons":   cmd_buttons,
        "cmd_impulse":   cmd_impulse,
        "op_input":      op_input,
    }


# ─────────────────────────────────────────────────────────────────
# Native obs wire format (engine_norm phase 2 — variable-length entity)
# ─────────────────────────────────────────────────────────────────
#
# Per qnn.engine_norm. Layout (little-endian, all offsets fixed except
# the entity stream which starts at NATIVE_ENTITY_STREAM_OFFSET and
# extends variable bytes per frame).
#
#   off    field                    dtype       shape       bytes
#     0    health                   u8          ()             1
#     1    effective_armor          u8          ()             1
#     2..5 ammo {shells,nails,rk,c} u8          ()             4
#     6    vel                      i16         (3,)           6
#    12    attack_finished          f16         ()             2  (seconds)
#    14    self_weapon_id           u8          ()             1
#    15    self_movement_id         u8          ()             1
#    16    self_items               u32         ()             4
#    20    view_pitch               i8          ()             1  (deg / 90)
#    21    spatial_dir              i8          (9, 3)        27
#    47    spatial_nearest_dist     u16         (9,)          18
#    65    spatial_mean_dist        u16         (9,)          18
#    83    spatial_openness         u8          (9,)           9
#    92    spatial_clearance        u8          (9,)           9
#   101    spatial_traversable      u8          (9,)           9
#   110    spatial_dropoff          u8          (9,)           9
#   119    spatial_solid_frac       u8          (9,)           9
#   128    spatial_water_frac       u8          (9,)           9
#   137    spatial_slime_frac       u8          (9,)           9
#   146    spatial_lava_frac        u8          (9,)           9
#   155    entity_stream            variable
#
# Entity stream per frame:
#   u8  n_tokens
#   for each token:
#     u8  type tag (0=PROJ, 1=ACTOR, 2=ITEM, 3=MOVER)
#     u8  subject_id
#     u8  modality_id
#     u8  player_id              (actor only; absent otherwise)
#     u8  event_count
#     u8 × event_count × 2       (action_id, source_id) pairs
#     <per-type scalar bytes>    (per ENTITY_FIELDS in engine_norm)
#
# Per-type scalar layouts (engine_norm.py is authoritative):
#   PROJECTILE: i16×3 rel, i16×3 vel, f16 recency                          14 B
#   ACTOR:      u8×3 half_ext, i16×3 rel, i16×3 vel, i16×3 path,
#               u16 path_dist, f16 eta, u8 facing, u8 team, u8 score,
#               f16 recency                                                30 B
#   ITEM:       u8×3 half_ext, i16×3 rel, i16×3 path, u16 path_dist,
#               f16 eta, u8 amount, f16 regen, f16 recency                 24 B
#   MOVER:      u8×3 half_ext, i16×3 rel, i16×3 path, u16 path_dist,
#               f16 eta, u8 state, f16 recency                             22 B
#
# `dist` is not on the wire — recomputed as `|rel| / DIST_SCALE` by
# the EntityDequantizer at the model boundary.

NATIVE_SELF_OFFSET           = 0
NATIVE_SELF_BYTES            = 21

NATIVE_SPATIAL_OFFSET        = NATIVE_SELF_OFFSET + NATIVE_SELF_BYTES   # 21
NATIVE_SPATIAL_BYTES         = 135                                      # 9 sectors × 15 B
NATIVE_ENTITY_STREAM_OFFSET  = NATIVE_SPATIAL_OFFSET + NATIVE_SPATIAL_BYTES  # 156

# Token-type tags on the wire — mirror qnn.vocab TOKEN_* constants.
_TOK_PROJECTILE = TOKEN_PROJECTILE
_TOK_ACTOR      = TOKEN_ACTOR
_TOK_ITEM       = TOKEN_ITEM
_TOK_MOVER      = TOKEN_MOVER


def _u8(raw: bytes, off: int) -> int:
    return raw[off]


def _read_array(raw: bytes, off: int, dtype, count: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype, offset=off, count=count).copy()


def _read_scalar(raw: bytes, off: int, dtype) -> np.ndarray:
    """One scalar of `dtype` at `off`, returned as a 0-d ndarray."""
    return np.frombuffer(raw, dtype=dtype, offset=off, count=1).reshape(()).copy()


def _unpack_native_self(raw: bytes) -> dict[str, np.ndarray]:
    o = NATIVE_SELF_OFFSET
    return {
        "health":           _read_scalar(raw, o + 0,  np.uint8),
        "effective_armor":  _read_scalar(raw, o + 1,  np.uint8),
        "ammo_shells":      _read_scalar(raw, o + 2,  np.uint8),
        "ammo_nails":       _read_scalar(raw, o + 3,  np.uint8),
        "ammo_rockets":     _read_scalar(raw, o + 4,  np.uint8),
        "ammo_cells":       _read_scalar(raw, o + 5,  np.uint8),
        "vel":              _read_array(raw,  o + 6,  np.int16,   3),
        "attack_finished":  _read_scalar(raw, o + 12, np.float16),
        "self_weapon_id":   _read_scalar(raw, o + 14, np.uint8),
        "self_movement_id": _read_scalar(raw, o + 15, np.uint8),
        "self_items":       _read_scalar(raw, o + 16, np.int32),
        "view_pitch":       _read_scalar(raw, o + 20, np.int8),
    }


def _unpack_native_spatial(raw: bytes) -> dict[str, np.ndarray]:
    o = NATIVE_SPATIAL_OFFSET
    # Layout per-field-stride: 9 sectors, fields packed contiguously.
    # i.e. all 9 dirs first (9 × 3 × 1 B), then all 9 nearest_dist
    # (9 × 2 B), etc. Mirrors how engine_norm.SPATIAL_FIELDS lays out
    # per-sector data when packed by sector-major in qnn_io.c.
    #
    # Position tracker for byte offsets per field, building forward:
    p = o
    out: dict[str, np.ndarray] = {}
    out["spatial_dir"]          = _read_array(raw, p, np.int8,    SPATIAL_TOKEN_COUNT * 3).reshape(SPATIAL_TOKEN_COUNT, 3); p += SPATIAL_TOKEN_COUNT * 3
    out["spatial_nearest_dist"] = _read_array(raw, p, np.uint16,  SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT * 2
    out["spatial_mean_dist"]    = _read_array(raw, p, np.uint16,  SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT * 2
    out["spatial_openness"]     = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_clearance"]    = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_traversable"]  = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_dropoff"]      = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_solid_frac"]   = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_water_frac"]   = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_slime_frac"]   = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    out["spatial_lava_frac"]    = _read_array(raw, p, np.uint8,   SPATIAL_TOKEN_COUNT); p += SPATIAL_TOKEN_COUNT
    assert p - o == NATIVE_SPATIAL_BYTES, (
        f"spatial block consumed {p - o} B, expected {NATIVE_SPATIAL_BYTES}"
    )
    return out


def _unpack_native_entity_stream(
    raw: bytes, offset: int
) -> Tuple[dict[str, np.ndarray], int]:
    """Parse the variable-length native entity stream.

    Returns per-field arrays of length n_tokens. Empty stream
    (n_tokens=0) yields zero-length arrays so the dataloader can
    cleanly pad to batch-max at collation.
    """
    pos = offset
    n = _u8(raw, pos); pos += 1

    types       = np.zeros((n,),    dtype=np.int8)
    subjects    = np.zeros((n,),    dtype=np.uint8)
    modalities  = np.zeros((n,),    dtype=np.uint8)
    players     = np.zeros((n,),    dtype=np.uint8)
    evt_counts  = np.zeros((n,),    dtype=np.uint8)
    evt_actions = np.zeros((n, MAX_ENTITY_EVENTS), dtype=np.uint8)
    evt_sources = np.zeros((n, MAX_ENTITY_EVENTS), dtype=np.uint8)
    half_ext    = np.zeros((n, 3),  dtype=np.uint8)
    rel         = np.zeros((n, 3),  dtype=np.int16)
    vel         = np.zeros((n, 3),  dtype=np.int16)
    path        = np.zeros((n, 3),  dtype=np.int16)
    path_dist   = np.zeros((n,),    dtype=np.uint16)
    eta         = np.zeros((n,),    dtype=np.float16)
    recency     = np.zeros((n,),    dtype=np.float16)
    facing      = np.zeros((n,),    dtype=np.uint8)
    team        = np.zeros((n,),    dtype=np.uint8)
    score       = np.zeros((n,),    dtype=np.uint8)
    amount      = np.zeros((n,),    dtype=np.uint8)
    regen       = np.zeros((n,),    dtype=np.float16)
    state       = np.zeros((n,),    dtype=np.uint8)

    for i in range(n):
        tag = _u8(raw, pos); pos += 1
        if tag not in (_TOK_PROJECTILE, _TOK_ACTOR, _TOK_ITEM, _TOK_MOVER):
            break  # unknown — stop parsing defensively
        types[i] = tag

        subjects[i]   = _u8(raw, pos); pos += 1
        modalities[i] = _u8(raw, pos); pos += 1
        if tag == _TOK_ACTOR:
            players[i] = _u8(raw, pos); pos += 1
        # else: leave player_id = 0

        ne = _u8(raw, pos); pos += 1
        if ne > MAX_ENTITY_EVENTS:
            ne = MAX_ENTITY_EVENTS
        evt_counts[i] = ne
        for j in range(ne):
            evt_actions[i, j] = _u8(raw, pos); pos += 1
            evt_sources[i, j] = _u8(raw, pos); pos += 1

        # Per-type scalars — fixed-size for that type.
        if tag == _TOK_PROJECTILE:
            rel[i]     = _read_array(raw, pos, np.int16, 3);   pos += 6
            vel[i]     = _read_array(raw, pos, np.int16, 3);   pos += 6
            recency[i] = _read_scalar(raw, pos, np.float16);   pos += 2
        elif tag == _TOK_ACTOR:
            half_ext[i] = _read_array(raw, pos, np.uint8,   3); pos += 3
            rel[i]      = _read_array(raw, pos, np.int16,   3); pos += 6
            vel[i]      = _read_array(raw, pos, np.int16,   3); pos += 6
            path[i]     = _read_array(raw, pos, np.int16,   3); pos += 6
            path_dist[i] = _read_scalar(raw, pos, np.uint16);   pos += 2
            eta[i]      = _read_scalar(raw, pos, np.float16);   pos += 2
            facing[i]   = _u8(raw, pos); pos += 1
            team[i]     = _u8(raw, pos); pos += 1
            score[i]    = _u8(raw, pos); pos += 1
            recency[i]  = _read_scalar(raw, pos, np.float16);   pos += 2
        elif tag == _TOK_ITEM:
            half_ext[i] = _read_array(raw, pos, np.uint8,   3); pos += 3
            rel[i]      = _read_array(raw, pos, np.int16,   3); pos += 6
            path[i]     = _read_array(raw, pos, np.int16,   3); pos += 6
            path_dist[i] = _read_scalar(raw, pos, np.uint16);   pos += 2
            eta[i]      = _read_scalar(raw, pos, np.float16);   pos += 2
            amount[i]   = _u8(raw, pos); pos += 1
            regen[i]    = _read_scalar(raw, pos, np.float16);   pos += 2
            recency[i]  = _read_scalar(raw, pos, np.float16);   pos += 2
        elif tag == _TOK_MOVER:
            half_ext[i] = _read_array(raw, pos, np.uint8,   3); pos += 3
            rel[i]      = _read_array(raw, pos, np.int16,   3); pos += 6
            path[i]     = _read_array(raw, pos, np.int16,   3); pos += 6
            path_dist[i] = _read_scalar(raw, pos, np.uint16);   pos += 2
            eta[i]      = _read_scalar(raw, pos, np.float16);   pos += 2
            state[i]    = _u8(raw, pos); pos += 1
            recency[i]  = _read_scalar(raw, pos, np.float16);   pos += 2

    return {
        "entity_types":         types,
        "entity_subject_id":    subjects,
        "entity_modality_id":   modalities,
        "entity_player_id":     players,
        "entity_event_count":   evt_counts,
        "entity_event_actions": evt_actions,
        "entity_event_sources": evt_sources,
        "entity_half_extents":  half_ext,
        "entity_rel":           rel,
        "entity_vel":           vel,
        "entity_path":          path,
        "entity_path_dist":     path_dist,
        "entity_eta":           eta,
        "entity_recency":       recency,
        "entity_facing":        facing,
        "entity_team":          team,
        "entity_score":         score,
        "entity_amount":        amount,
        "entity_regen":         regen,
        "entity_state":         state,
        "entity_count":         np.array(n, dtype=np.uint8),
    }, pos


def unpack_obs_buffer_native(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack a native-format obs buffer into per-field arrays.

    Matches the layout produced by the new C-side QNN_IOPackObsBuffer
    (engine_norm phase 2). The returned dict feeds directly into the
    Self/Spatial/Entity dequantizers in qnn.model.dequant after the
    dataloader adds a leading batch dimension.

    Self and spatial fields are fixed-shape; entity fields have a
    leading length equal to the actual token count for this frame
    (0..MAX_TOKEN_OBJECTS). The dataloader pads to max-in-batch at
    collation.
    """
    out: dict[str, np.ndarray] = {}
    out.update(_unpack_native_self(raw))
    out.update(_unpack_native_spatial(raw))
    entities, _ = _unpack_native_entity_stream(raw, NATIVE_ENTITY_STREAM_OFFSET)
    out.update(entities)
    return out
