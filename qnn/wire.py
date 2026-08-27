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

import numpy as np

from qnn.obs_api import (
    DEFAULT_LAYOUT,
    ENTITY_POLICIES,
    Layout,
    LayoutField,
    MAX_PERCEPT_ROW_BYTES,
    POSE_TAIL_BYTES,
)
from qnn.vocab import (
    ACTOR_SCALAR_DIM, ACTOR_ID_DIM,
    MAX_ENTITY_EVENTS, MAX_TOKEN_OBJECTS,
)

MAX_ENTITY_SCALAR_DIM = ACTOR_SCALAR_DIM  # largest per-type scalar count
MAX_ENTITY_ID_DIM = ACTOR_ID_DIM          # largest per-type ID count

OBS_BUFFER_SIZE = 864
# sizeof(qnn_action_t) — move (press byte) + weapon + input_mask +
# op_input + look[3], mirroring qnn_action_t (src/engine/common/qnn.h):
# move press byte (input_mask bit layout), weapon (raw impulse, full-
# entity wires), attack (A27 combat 9-way categorical), input_mask,
# op_input, 3 pad bytes, look[3] f32 at offset 8 — 20 B total. Matches
# engine.bridge._ACTION_PACK_FORMAT ("<5B3x3f").
ACTION_SIZE = 20

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
# is "MLOB" magic + a fixed 23-byte payload (no obs buffer).  Must match
# qnn_mlob_record_t in qnn.h / QNN_EmitMlob in qnn_collect_helpers.c:
#
#   off  field             dtype     bytes
#     0  native_index      u32          4   = qnn_runtime.tick
#     4  flags             u16          2   FLAG_DONE / FLAG_RESET
#     6  vel[3]            i16 ×3        6   view-frame velocity, raw units
#    12  self_movement_id   u8          1   0=ground 1=air 2..4=water
#    13  move               u8          1   press byte (usercmd-truth)
#    14  weapon             u8          1   attack-with impulse (0 off attack)
#    15  input_mask         u8          1   per-axis feasibility byte
#    16  op_input           u8          1   strict per-axis operativeness
#    17  look[3]           f16 ×3        6   view-relative look delta
# total                                23
MLOB_MAGIC = b"MLOB"
MLOB_RECORD_SIZE = 23


def parse_mlob_frame(raw: bytes) -> dict:
    """Parse one MLOB payload (the 23 bytes after the 4-byte magic).

    Returns a tick dict with the slim labeler fields and a ``done`` flag.
    ``vel`` is int16 (view-frame raw Quake units, ÷QNN_VELOCITY_SCALE to
    normalize); ``look`` is float16. ``native_index`` is the frame index
    that the matching 20 Hz QOBS frame stamps into its header ``steps``."""
    native_index, flags = struct.unpack_from("<IH", raw, 0)
    vel = np.frombuffer(raw, dtype=np.int16, offset=6, count=3).copy()
    move, weapon, input_mask, op_input = struct.unpack_from("<BBBB", raw, 13)
    look = np.frombuffer(raw, dtype=np.float16, offset=17, count=3).copy()
    return {
        "native_index":     int(native_index),
        "self_movement_id": raw[12],
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
# Native obs wire format (obs_api v1 — layout-driven codec)
# ─────────────────────────────────────────────────────────────────
#
# The parser is generic over a qnn.obs_api.Layout: ``unpack_frame`` /
# ``unpack_frame_batch`` decode whatever field set a seat negotiated
# at attach time. The pinned offsets below describe the DEFAULT plan
# (no declaration sent = today's 864-byte frame, bit-identical) —
# they are derived from qnn.obs_api.DEFAULT_LAYOUT, no longer a
# hand-maintained table. Little-endian throughout; all offsets fixed
# except the entity stream, which starts at
# NATIVE_ENTITY_STREAM_OFFSET and extends variable bytes per frame.
#
#   off    field                    dtype       shape       bytes
#     0    health                   u8          ()             1
#     1    effective_armor          u8          ()             1
#     2..5 ammo {shells,nails,rk,c} u8          ()             4
#     6    vel                      i16         (3,)           6
#    12    attack_finished          f16         ()             2  (seconds)
#    14    self_weapon_id           u8          ()             1  (engine-written;
#                                                                 NOT parsed — no
#                                                                 engine-equipped-
#                                                                 weapon input
#                                                                 in the A27 obs)
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
#     u8  type tag (0=PROJECTILE, 1=ACTOR)
#     u8  subject_id
#     u8  modality_id
#     u8  player_id              (actor only; absent otherwise)
#     u8  event_count
#     u8 × event_count × 2       (action_id, source_id) pairs
#     <per-type scalar bytes>    (per ENTITY_FIELDS in engine_norm)
#
# Per-type scalar layouts (engine_norm.py is authoritative):
#   PROJECTILE: i16×3 rel, i16×3 vel                                      12 B
#   ACTOR:      u8×3 half_ext, i16×3 rel, i16×3 vel, i16×3 path,
#               u16 path_dist, f16 eta, u8 facing, u8 team, u8 score       28 B
#
# `dist` is not on the wire — recomputed as `|rel| / DIST_SCALE` by
# the EntityDequantizer at the model boundary.

# Default-plan constants, derived from the compiled default layout —
# kept because the C headers, tests and diagnostics cross-reference
# them by these names.
NATIVE_SELF_OFFSET           = 0
NATIVE_SELF_BYTES            = DEFAULT_LAYOUT.field("atlas").offset             # 27
NATIVE_SPATIAL_OFFSET        = NATIVE_SELF_BYTES                                # 27
NATIVE_SPATIAL_BYTES         = DEFAULT_LAYOUT.field("atlas").nbytes             # 132
NATIVE_ENTITY_STREAM_OFFSET  = DEFAULT_LAYOUT.field("entities").offset          # 159

# Fixed-frame proof. Actor is the widest possible entity row:
# type + subject/modality/player + event count + four event pairs + 30 scalars
# (the registry-wide budget — see qnn.obs_api.MAX_PERCEPT_ROW_BYTES).
NATIVE_MAX_ACTOR_ROW_BYTES = MAX_PERCEPT_ROW_BYTES  # 43
NATIVE_MAX_PAYLOAD_BYTES = (
    NATIVE_ENTITY_STREAM_OFFSET + DEFAULT_LAYOUT.field("entities").nbytes
)  # 848
POSE_TAIL_OFFSET = OBS_BUFFER_SIZE - POSE_TAIL_BYTES
assert NATIVE_MAX_PAYLOAD_BYTES <= POSE_TAIL_OFFSET
# The default plan IS today's frame — gate 1 of agents/plans/obs-api.md.
assert DEFAULT_LAYOUT.frame_bytes == OBS_BUFFER_SIZE


def _read_array(raw: bytes, off: int, dtype, count: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype, offset=off, count=count).copy()


def _read_scalar(raw: bytes, off: int, dtype) -> np.ndarray:
    """One scalar of `dtype` at `off`, returned as a 0-d ndarray."""
    return np.frombuffer(raw, dtype=dtype, offset=off, count=1).reshape(()).copy()


# Per-policy walk tables: tag → (has_player_id, ((key, np dtype,
# count, nbytes), …)), resolved once so the hot per-token loop never
# re-parses dtype names.
def _policy_walk_table(policy_name: str) -> dict[int, tuple[bool, tuple]]:
    table = _POLICY_WALK_CACHE.get(policy_name)
    if table is None:
        policy = ENTITY_POLICIES[policy_name]
        table = {
            tag: (
                token_type.has_player_id,
                tuple(
                    (key, np.dtype(dtype), count, np.dtype(dtype).itemsize * count)
                    for key, dtype, count in token_type.scalars
                ),
            )
            for tag, token_type in policy.token_types.items()
        }
        _POLICY_WALK_CACHE[policy_name] = table
    return table


_POLICY_WALK_CACHE: dict[str, dict[int, tuple[bool, tuple]]] = {}


def _walk_entity_stream(
    raw: bytes, pos: int, n: int,
    policy_name: str,
    arrays: dict[str, np.ndarray],
) -> int:
    """The one wire-layout walk, writing rows [0..n) into caller arrays.

    Callers pass either freshly allocated length-n arrays (per-lane
    unpack) or per-lane row views of padded (B, max_tokens, …) batch
    arrays (``unpack_frame_batch``) — one implementation to keep in
    sync with QNN_IOPackObsBuffer. The policy's token-type table
    drives the per-type scalar reads; unknown tags fail loud.
    """
    walk = _policy_walk_table(policy_name)
    label = ENTITY_POLICIES[policy_name].label
    types      = arrays["entity_types"]
    subjects   = arrays["entity_subject_id"]
    modalities = arrays["entity_modality_id"]
    players    = arrays["entity_player_id"]
    evt_counts  = arrays["entity_event_count"]
    evt_actions = arrays["entity_event_actions"]
    evt_sources = arrays["entity_event_sources"]
    for i in range(n):
        tag = raw[pos]; pos += 1
        row = walk.get(tag)
        if row is None:
            raise ValueError(f"invalid {label} token tag {tag}")
        has_player_id, scalars = row
        types[i] = tag

        subjects[i]   = raw[pos]; pos += 1
        modalities[i] = raw[pos]; pos += 1
        if has_player_id:
            players[i] = raw[pos]; pos += 1
        # else: leave player_id = 0

        ne = raw[pos]; pos += 1
        if ne > MAX_ENTITY_EVENTS:
            ne = MAX_ENTITY_EVENTS
        evt_counts[i] = ne
        for j in range(ne):
            evt_actions[i, j] = raw[pos]; pos += 1
            evt_sources[i, j] = raw[pos]; pos += 1

        # Per-type scalars — fixed-size for that type.
        for key, dtype, count, nbytes in scalars:
            if count == 1:
                arrays[key][i] = _read_scalar(raw, pos, dtype)
            else:
                arrays[key][i] = _read_array(raw, pos, dtype, count)
            pos += nbytes

    return pos


def _unpack_entity_stream(
    raw: bytes, field: LayoutField
) -> dict[str, np.ndarray]:
    """Parse the variable-length entity stream at its layout offset.

    Returns per-field arrays of length n_tokens. Empty stream
    (n_tokens=0) yields zero-length arrays so the dataloader can
    cleanly pad to batch-max at collation.
    """
    policy = ENTITY_POLICIES[field.params["policy"]]
    pos = field.offset
    n = raw[pos]; pos += 1
    arrays = {
        key: np.zeros((n, *tail), dtype=dtype)
        for key, dtype, tail, _fill in policy.fields
    }
    _walk_entity_stream(raw, pos, n, policy.name, arrays)
    arrays["entity_count"] = np.array(n, dtype=np.uint8)
    return arrays


def unpack_frame(raw: bytes, layout: Layout) -> dict[str, np.ndarray]:
    """Unpack one obs frame according to its negotiated layout.

    Matches the emit plan the engine compiled for this seat (obs_api
    v1). The returned dict feeds directly into the Self/Spatial/
    Entity dequantizers in qnn.model.dequant after the dataloader
    adds a leading batch dimension.

    State and sensor fields are fixed-shape (scalars come out as 0-d
    ndarrays); entity fields have a leading length equal to the
    actual token count for this frame. The dataloader pads to
    max-in-batch at collation.
    """
    if len(raw) < layout.frame_bytes:
        raise ValueError(
            f"obs frame is {len(raw)} bytes, layout needs {layout.frame_bytes}"
        )
    out: dict[str, np.ndarray] = {}
    for field in layout.fields:
        if field.kind == "state":
            dtype = field.np_dtype()
            if field.shape:
                out[field.name] = _read_array(
                    raw, field.offset, dtype, field.shape[0]
                )
            else:
                out[field.name] = _read_scalar(raw, field.offset, dtype)
        elif field.kind == "sensor":
            # The atlas: elevation-major u8 rows, nibble-packed (two
            # 4-bit codes per byte, low nibble first) or one code per
            # byte per the declared params. SpatialDequantizer expands
            # packed rows downstream.
            out["spatial_atlas"] = _read_array(
                raw, field.offset, field.np_dtype(), field.nbytes
            ).reshape(field.shape)
        elif field.kind == "percept":
            out.update(_unpack_entity_stream(raw, field))
        else:
            raise ValueError(f"unknown layout field kind {field.kind!r}")
    return out


def unpack_frame_batch(raws: np.ndarray, layout: Layout) -> dict[str, np.ndarray]:
    """Unpack B stacked obs frames into batched per-field arrays.

    ``raws``: (B, layout.frame_bytes) uint8. State/sensor fields come
    out vectorized across lanes (column slice → dtype view); entity
    fields come out PADDED to the layout's ``max_tokens`` (fill
    semantics identical to ``vec_env.pad_entities``), the walk
    writing straight into per-lane row views. Field set, dtypes and
    values match per-lane ``unpack_frame`` + padding exactly (tested).
    """
    B = int(raws.shape[0])
    if raws.shape[1] < layout.frame_bytes:
        raise ValueError(
            f"obs rows are {raws.shape[1]} bytes, layout needs {layout.frame_bytes}"
        )
    out: dict[str, np.ndarray] = {}

    def col(off: int, nbytes: int, dtype, tail: tuple = ()) -> np.ndarray:
        block = np.ascontiguousarray(raws[:, off:off + nbytes]).view(dtype)
        return block.reshape((B, *tail)) if tail else block.reshape(B)

    percept: LayoutField | None = None
    for field in layout.fields:
        if field.kind == "state":
            out[field.name] = col(
                field.offset, field.nbytes, field.np_dtype(), field.shape
            )
        elif field.kind == "sensor":
            out["spatial_atlas"] = col(
                field.offset, field.nbytes, field.np_dtype(), field.shape
            )
        elif field.kind == "percept":
            percept = field
        else:
            raise ValueError(f"unknown layout field kind {field.kind!r}")
    if percept is None:
        return out

    policy = ENTITY_POLICIES[percept.params["policy"]]
    M = int(percept.params["max_tokens"])
    for name, dtype, tail, fill in policy.fields:
        arr = np.full((B, M, *tail), fill, dtype=dtype) if fill else \
            np.zeros((B, M, *tail), dtype=dtype)
        out[name] = arr
    counts = np.zeros((B,), dtype=np.uint8)
    for lane in range(B):
        raw = raws[lane].tobytes()
        pos = percept.offset
        n = min(raw[pos], M); pos += 1  # row views are M long; engine caps at M
        counts[lane] = n
        if n == 0:
            continue
        _walk_entity_stream(
            raw, pos, n, policy.name,
            {name: out[name][lane] for name, _, _, _ in policy.fields},
        )
    out["entity_count"] = counts
    return out


# ── Default-plan compatibility wrappers ──────────────────────────
#
# The pre-obs-api entry points. Same names, same output dicts, same
# bytes — now thin wrappers over the generic codec at DEFAULT_LAYOUT.
# The one asymmetry: offset 14 (self_weapon_id) is WRITTEN by the
# engine but NOT part of the A27 obs contract (the 9-way attack head
# IS the select-and-fire decision), so these wrappers drop the key
# the layout necessarily carries for its byte.


def unpack_obs_buffer_native(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack one default-plan (864 B) obs buffer into per-field arrays."""
    out = unpack_frame(raw, DEFAULT_LAYOUT)
    del out["self_weapon_id"]  # occupies the frame byte; not consumed
    return out


def unpack_obs_buffer_native_batch(raws: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack B stacked default-plan obs buffers into batched arrays."""
    out = unpack_frame_batch(raws, DEFAULT_LAYOUT)
    del out["self_weapon_id"]  # occupies the frame byte; not consumed
    return out
