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

OBS_BUFFER_SIZE = 864
# sizeof(qnn_action_t) — move (press byte) + weapon + input_mask +
# op_input + look[3]. The press byte mirrors the input_mask bit layout
# (attack at bit 0, fb/lr/ud neg/pos in bits 1-6, jump at bit 7) so the
# engine's in-memory representation matches the on-disk compacted form
# emitted by qnn.bc.collect.  op_input (offset 3, formerly the _pad byte)
# is the strict per-axis OPERATIVENESS mask (press AND engine acted) —
# additive: the struct size is unchanged at 16 B, so every pre-existing
# field stays byte-identical.
ACTION_SIZE = 16

# Per-tick header emitted by demo worker collect mode.
#
# Header layout is "<IIIHH" = (tick, steps, tick_hz, flags, action_size).
# The `steps` field is unused by the QOBS parser in normal collects
# (_parse_qobs_frame reads only flags + obs + action).  In MATCHED-EMIT
# mode (qnn_runtime.matched_emit), the worker REUSES `steps` to carry the
# native frame index this 20 Hz QOBS frame was sampled at, so the slim
# native-rate labeler stream (MLOB) can be resampled to 20 Hz by exact
# index lookup.  No wire-size change — `steps` is repurposed, not added.
TICK_HEADER_SIZE = 16
TICK_HEADER_FORMAT = "<IIIHH"
TICK_MAGIC = b"QOBS"
TICK_MAGIC_SIZE = 4
FLAG_RESET = 0x01
FLAG_DONE = 0x02

# ── Slim MLOB record (matched-emit mode) ─────────────────────────────
#
# Emitted once per native frame alongside the 20 Hz QOBS stream.  Framing
# is "MLOB" magic + a fixed 24-byte payload (no obs buffer).  Must match
# qnn_mlob_record_t in qnn.h / QNN_EmitMlob in qnn_collect_helpers.c:
#
#   off  field             dtype     bytes
#     0  native_index      u32          4   = qnn_runtime.tick
#     4  flags             u16          2   FLAG_DONE / FLAG_RESET
#     6  vel[3]            i16 ×3        6   view-frame velocity, raw units
#    12  self_movement_id   u8          1   0=ground 1=air 2..4=water
#    13  self_weapon_id     u8          1   subject-form weapon id
#    14  move               u8          1   press byte (usercmd-truth)
#    15  weapon             u8          1   raw engine weapon byte
#    16  input_mask         u8          1   per-axis feasibility byte
#    17  op_input           u8          1   strict per-axis operativeness
#    18  look[3]           f16 ×3        6   view-relative look delta
# total                                24
MLOB_MAGIC = b"MLOB"
MLOB_RECORD_SIZE = 24


def parse_mlob_frame(raw: bytes) -> dict:
    """Parse one MLOB payload (the 24 bytes after the 4-byte magic).

    Returns a tick dict with the slim labeler fields and a ``done`` flag.
    ``vel`` is int16 (view-frame raw Quake units, ÷QNN_VELOCITY_SCALE to
    normalize); ``look`` is float16. ``native_index`` is the frame index
    that the matching 20 Hz QOBS frame stamps into its header ``steps``."""
    native_index, flags = struct.unpack_from("<IH", raw, 0)
    vel = np.frombuffer(raw, dtype=np.int16, offset=6, count=3).copy()
    move, weapon, input_mask, op_input = struct.unpack_from("<BBBB", raw, 14)
    look = np.frombuffer(raw, dtype=np.float16, offset=18, count=3).copy()
    return {
        "native_index":     int(native_index),
        "self_movement_id": raw[12],
        "self_weapon_id":   raw[13],
        "vel":              vel,
        "move":             move,
        "weapon":           weapon,
        "input_mask":       input_mask,
        "op_input":         op_input,
        "look":             look,
        "done":             bool(flags & FLAG_DONE),
    }

# Legacy-shape constants kept for compat with qnn.schema. Used to
# express the model-facing dense layout after the dequantizers run;
# the wire itself no longer emits these (the native parser produces
# per-field arrays at native widths instead).
SELF_SCALAR_DIM = 18
SPATIAL_TOKEN_COUNT = 11  # depth-atlas elevation bands (rev 8)
SPATIAL_SCALAR_DIM = 48  # per band: depth x 24 + hit x 24


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
#    21    look_delta               f16         (3,)           6
#    27    spatial_atlas            u8          (11, 12)     132  (two packed
#                                                                 4-bit codes/byte)
#   159    entity_stream            variable
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
NATIVE_SELF_BYTES            = 27   # 21 + look_delta (f16 × 3)

NATIVE_SPATIAL_OFFSET        = NATIVE_SELF_OFFSET + NATIVE_SELF_BYTES   # 27
NATIVE_SPATIAL_BYTES         = 132                                      # 11 bands × 12 packed bytes
NATIVE_ENTITY_STREAM_OFFSET  = NATIVE_SPATIAL_OFFSET + NATIVE_SPATIAL_BYTES  # 159

# Fixed-frame proof. Actor is the widest possible entity row:
# type + subject/modality/player + event count + four event pairs + 30 scalars.
NATIVE_MAX_ACTOR_ROW_BYTES = 1 + 3 + 1 + 2 * MAX_ENTITY_EVENTS + 30  # 43
NATIVE_MAX_PAYLOAD_BYTES = (
    NATIVE_ENTITY_STREAM_OFFSET + 1
    + MAX_TOKEN_OBJECTS * NATIVE_MAX_ACTOR_ROW_BYTES
)  # 848
POSE_TAIL_OFFSET = OBS_BUFFER_SIZE - 16
assert NATIVE_MAX_PAYLOAD_BYTES <= POSE_TAIL_OFFSET

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
        "look_delta":       _read_array(raw,  o + 21, np.float16, 3),
    }


def _unpack_native_spatial(raw: bytes) -> dict[str, np.ndarray]:
    # Final wire.12 atlas: elevation-major (11, 12) packed u8 bytes.
    # Each byte stores two yaw codes, low nibble first. SpatialDequantizer
    # expands them to 24 codes per row.
    atlas = _read_array(
        raw, NATIVE_SPATIAL_OFFSET, np.uint8, NATIVE_SPATIAL_BYTES,
    ).reshape(SPATIAL_TOKEN_COUNT, NATIVE_SPATIAL_BYTES // SPATIAL_TOKEN_COUNT)
    return {"spatial_atlas": atlas}


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

    pos = _walk_entity_stream(
        raw, pos, n,
        types, subjects, modalities, players, evt_counts, evt_actions,
        evt_sources, half_ext, rel, vel, path, path_dist, eta, recency,
        facing, team, score, amount, regen, state,
    )

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


def _walk_entity_stream(
    raw: bytes, pos: int, n: int,
    types, subjects, modalities, players, evt_counts, evt_actions,
    evt_sources, half_ext, rel, vel, path, path_dist, eta, recency,
    facing, team, score, amount, regen, state,
) -> int:
    """The one wire-layout walk, writing rows [0..n) into caller arrays.

    Callers pass either freshly allocated length-n arrays (per-lane
    unpack) or per-lane row views of padded (B, MAX_TOKEN_OBJECTS, …)
    batch arrays (``unpack_obs_buffer_native_batch``) — one
    implementation to keep in sync with QNN_IOPackObsBuffer.
    """
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

    return pos


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


# Batched-drain entity field specs: name → (dtype, per-token tail shape,
# pad fill). entity_types pads with -1 (empty-slot sentinel the model's
# mask reads); everything else zero-fills — mirrors vec_env.pad_entities.
_ENTITY_BATCH_FIELDS: tuple = (
    ("entity_types",         np.int8,    (),                   -1),
    ("entity_subject_id",    np.uint8,   (),                    0),
    ("entity_modality_id",   np.uint8,   (),                    0),
    ("entity_player_id",     np.uint8,   (),                    0),
    ("entity_event_count",   np.uint8,   (),                    0),
    ("entity_event_actions", np.uint8,   (MAX_ENTITY_EVENTS,),  0),
    ("entity_event_sources", np.uint8,   (MAX_ENTITY_EVENTS,),  0),
    ("entity_half_extents",  np.uint8,   (3,),                  0),
    ("entity_rel",           np.int16,   (3,),                  0),
    ("entity_vel",           np.int16,   (3,),                  0),
    ("entity_path",          np.int16,   (3,),                  0),
    ("entity_path_dist",     np.uint16,  (),                    0),
    ("entity_eta",           np.float16, (),                    0),
    ("entity_recency",       np.float16, (),                    0),
    ("entity_facing",        np.uint8,   (),                    0),
    ("entity_team",          np.uint8,   (),                    0),
    ("entity_score",         np.uint8,   (),                    0),
    ("entity_amount",        np.uint8,   (),                    0),
    ("entity_regen",         np.float16, (),                    0),
    ("entity_state",         np.uint8,   (),                    0),
)


def unpack_obs_buffer_native_batch(raws: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack B stacked obs buffers into batched per-field arrays.

    ``raws``: (B, OBS_BUFFER_SIZE) uint8. Self/spatial fields come out
    vectorized across lanes (column slice → dtype view); entity fields
    come out PADDED to ``MAX_TOKEN_OBJECTS`` (fill semantics identical to
    ``vec_env.pad_entities``), the walk writing straight into per-lane
    row views. Field set, dtypes and values match per-lane
    ``unpack_obs_buffer_native`` + padding exactly (tested).
    """
    B = int(raws.shape[0])
    out: dict[str, np.ndarray] = {}

    def col(off: int, nbytes: int, dtype, tail: tuple = ()) -> np.ndarray:
        block = np.ascontiguousarray(raws[:, off:off + nbytes]).view(dtype)
        return block.reshape((B, *tail)) if tail else block.reshape(B)

    o = NATIVE_SELF_OFFSET
    out["health"]           = col(o + 0,  1, np.uint8)
    out["effective_armor"]  = col(o + 1,  1, np.uint8)
    out["ammo_shells"]      = col(o + 2,  1, np.uint8)
    out["ammo_nails"]       = col(o + 3,  1, np.uint8)
    out["ammo_rockets"]     = col(o + 4,  1, np.uint8)
    out["ammo_cells"]       = col(o + 5,  1, np.uint8)
    out["vel"]              = col(o + 6,  6, np.int16, (3,))
    out["attack_finished"]  = col(o + 12, 2, np.float16)
    out["self_weapon_id"]   = col(o + 14, 1, np.uint8)
    out["self_movement_id"] = col(o + 15, 1, np.uint8)
    out["self_items"]       = col(o + 16, 4, np.int32)
    out["view_pitch"]       = col(o + 20, 1, np.int8)
    out["look_delta"]       = col(o + 21, 6, np.float16, (3,))

    out["spatial_atlas"] = col(
        NATIVE_SPATIAL_OFFSET, NATIVE_SPATIAL_BYTES, np.uint8,
        (SPATIAL_TOKEN_COUNT, NATIVE_SPATIAL_BYTES // SPATIAL_TOKEN_COUNT),
    )

    M = MAX_TOKEN_OBJECTS
    for name, dtype, tail, fill in _ENTITY_BATCH_FIELDS:
        arr = np.full((B, M, *tail), fill, dtype=dtype) if fill else \
            np.zeros((B, M, *tail), dtype=dtype)
        out[name] = arr
    counts = np.zeros((B,), dtype=np.uint8)
    for lane in range(B):
        raw = raws[lane].tobytes()
        pos = NATIVE_ENTITY_STREAM_OFFSET
        n = min(raw[pos], M); pos += 1  # row views are M long; engine caps at M
        counts[lane] = n
        if n == 0:
            continue
        _walk_entity_stream(
            raw, pos, n,
            *(out[name][lane] for name, _, _, _ in _ENTITY_BATCH_FIELDS),
        )
    out["entity_count"] = counts
    return out
