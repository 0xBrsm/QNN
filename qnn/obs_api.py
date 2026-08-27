"""Observation API — registry mirror, declarations, layout math (obs_api v1).

Read-only Python mirror of the C-side observation field registry
(WS1: src/engine/common/qnn_obs_registry.{c,h} — the C table is
authoritative; this module exists for validation and layout math on
the driver side). Normative design: agents/plans/obs-api.md.

Three contract kinds:

- **state**   — raw game scalars (the self block), individually
  requestable by name.
- **sensor**  — parameterized computed queries. One row here: the
  depth ``atlas`` (24×11-packed and 72×11-unpacked are two
  parameterizations of ONE sensor).
- **percept** — the entity stream at a pinned disclosure-policy
  version. Policy versions pin semantics; a served policy is never
  retired.

── THE LAYOUT ORDERING RULE ─────────────────────────────────────────

``compile_layout`` and the C plan compiler MUST derive the identical
byte layout from a declaration with no further negotiation. The rule:

1. Fields are laid out in **registry order** — the order rows are
   declared in this module's tables (mirroring qnn_obs_registry.c),
   NOT the order names appear in the declaration JSON. State rows
   come first (in ``STATE_FIELDS`` order), then sensors (``atlas``),
   then percepts (``entities``).
2. Offsets are dense: each requested field starts where the previous
   requested field ends. Unrequested fields occupy no bytes.
3. The percept stream is budgeted at ``1 + max_tokens ×
   MAX_PERCEPT_ROW_BYTES`` — the registry-wide widest token row
   (the QNN_OBS_MAX_ACTOR_ROW_BYTES sizing rule from qnn_io.h),
   not the declared policy's own widest row. The stream itself is
   variable-length inside that budget.
4. ``frame_bytes = payload_max + POSE_TAIL_BYTES`` (the env-gated
   pose tail stays reserved at the end of every frame, outside the
   declaration), unless the declaration carries an explicit
   ``frame_bytes`` override (≥ the computed minimum). The override
   exists ONLY for the wire-shim declarations of pre-obs-api frame
   generations whose buffer sizes were pinned constants (e.g. the
   f84c36cd^ 4096-byte frame).

This rule reproduces today's frame order bit-exactly: the default
declaration (all 13 state fields, atlas 24×11 packed, entities v3 ×
16 tokens) compiles to the current 864-byte frame — self block at 0,
atlas at 27, entity stream at 159, pose tail at 848. No discrepancy
between the spec's ordering rule and today's frame order was found.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qnn.vocab import (
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    MAX_ENTITY_EVENTS, MAX_TOKEN_OBJECTS,
)

OBS_API_VERSION = 1

# Env-gated pose tail (16 B) reserved at the end of every frame —
# outside the declaration, mirrors QNN_POSE_TAIL_OFF in qnn_io.h.
POSE_TAIL_BYTES = 16

# Registry-wide widest token row: type tag + subject/modality/player
# ids + event count + max event pairs + the widest per-type scalar
# block across ALL registered policies (the v1 full-stream actor at
# 30 B). Mirrors QNN_OBS_MAX_ACTOR_ROW_BYTES in qnn_io.h — the C
# buffer-sizing rule budgets for the widest supported mode, not the
# active one, so both generations share one constant.
MAX_PERCEPT_ROW_BYTES = 1 + 3 + 1 + 2 * MAX_ENTITY_EVENTS + 30  # 43

# Wire dtypes by registry name. numpy names are the serialization
# format in layout replies; everything is little-endian on the wire.
_DTYPES: dict[str, np.dtype] = {
    name: np.dtype(name)
    for name in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float16")
}


def _dtype(name: str) -> np.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown obs_api wire dtype {name!r}")
    return _DTYPES[name]


# ─────────────────────────────────────────────────────────────────
# State rows (kind="state") — registry order IS frame order
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StateField:
    """One state registry row: a named raw game scalar/vector."""

    name: str
    dtype: str          # numpy dtype name (little-endian on the wire)
    shape: tuple[int, ...]  # () = scalar (0-d ndarray at unpack)

    @property
    def nbytes(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count * _dtype(self.dtype).itemsize


# NOTE: self_items is written as a raw u32 bitfield by the engine but
# has always been parsed as int32 on the Python side (same bytes) —
# the registry mirrors the consumer dtype so unpack output is
# unchanged.
STATE_FIELDS: tuple[StateField, ...] = (
    StateField("health",           "uint8",   ()),
    StateField("effective_armor",  "uint8",   ()),
    StateField("ammo_shells",      "uint8",   ()),
    StateField("ammo_nails",       "uint8",   ()),
    StateField("ammo_rockets",     "uint8",   ()),
    StateField("ammo_cells",       "uint8",   ()),
    StateField("vel",              "int16",   (3,)),
    StateField("attack_finished",  "float16", ()),
    StateField("self_weapon_id",   "uint8",   ()),
    StateField("self_movement_id", "uint8",   ()),
    StateField("self_items",       "int32",   ()),
    StateField("view_pitch",       "int8",    ()),
    StateField("look_delta",       "float16", (3,)),
)
STATE_FIELDS_BY_NAME: dict[str, StateField] = {f.name: f for f in STATE_FIELDS}


# ─────────────────────────────────────────────────────────────────
# The atlas sensor (kind="sensor")
# ─────────────────────────────────────────────────────────────────

ATLAS_NAME = "atlas"
ATLAS_BANDS = 11            # elevation bands — the only registered value
ATLAS_YAW_CHOICES = (24, 72)  # QNN_OBS_ATLAS_YAWS / _YAWS_LEGACY
ATLAS_OUT_KEY = "spatial_atlas"


def atlas_shape(params: Mapping[str, Any]) -> tuple[int, int]:
    """Raw wire shape of the atlas block: (bands, bytes-per-band).

    Packed stores two 4-bit depth codes per byte (yaw/2 bytes per
    band, low nibble = even yaw cell); unpacked stores one u8 code
    per yaw cell. The dequantizer expands packed rows downstream.
    """
    yaw = int(params["yaw"])
    bands = int(params["bands"])
    return (bands, yaw // 2 if params["packed"] else yaw)


def atlas_size(params: Mapping[str, Any]) -> int:
    bands, per_band = atlas_shape(params)
    return bands * per_band


def _validate_atlas_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params) - {"yaw", "bands", "packed"}
    if unknown:
        raise ValueError(f"unknown atlas params: {sorted(unknown)}")
    missing = {"yaw", "bands", "packed"} - set(params)
    if missing:
        raise ValueError(f"missing atlas params: {sorted(missing)}")
    yaw, bands, packed = params["yaw"], params["bands"], params["packed"]
    if not isinstance(yaw, int) or yaw not in ATLAS_YAW_CHOICES:
        raise ValueError(f"atlas yaw must be one of {ATLAS_YAW_CHOICES}, got {yaw!r}")
    if not isinstance(bands, int) or bands != ATLAS_BANDS:
        raise ValueError(f"atlas bands must be {ATLAS_BANDS}, got {bands!r}")
    if not isinstance(packed, bool):
        raise ValueError(f"atlas packed must be a bool, got {packed!r}")
    if packed and yaw % 2:
        raise ValueError(f"packed atlas requires an even yaw count, got {yaw}")
    return {"yaw": yaw, "bands": bands, "packed": packed}


# ─────────────────────────────────────────────────────────────────
# The entities percept (kind="percept") — pinned disclosure policies
# ─────────────────────────────────────────────────────────────────

ENTITIES_NAME = "entities"


@dataclass(frozen=True)
class TokenType:
    """Wire walk spec for one token tag within a percept policy."""

    has_player_id: bool
    # Per-type scalar block, in wire order: (out_key, dtype, count).
    scalars: tuple[tuple[str, str, int], ...]

    @property
    def scalar_bytes(self) -> int:
        return sum(_dtype(d).itemsize * c for _, d, c in self.scalars)


@dataclass(frozen=True)
class PerceptPolicy:
    """One pinned entity-stream policy version (a registry row's param).

    ``fields`` is the full output-array spec in emission order:
    (out_key, dtype, per-token tail shape, batch pad fill). It is the
    union of every token type's columns — rows a type doesn't carry
    stay at the fill value, exactly like the pre-obs-api parsers.
    """

    name: str
    label: str  # names the stream in fail-loud parse errors
    token_types: dict[int, TokenType]
    fields: tuple[tuple[str, str, tuple[int, ...], int], ...]


# Policy "v3" — the current post-pvs-fix pure-combat stream
# (QNN_ENTITY_MODE_COMBAT): current-frame actor/projectile only,
# SIGHT/PROXIMITY, no recency, no item/mover. Pins documented in the
# registry header (visibility = hull-0 LOS ∪ PVS-audible sound
# events, 2.0 s memory tail, 4 event pairs/entity, Recast/Detour
# paths, priority-order token budget). Any semantic change = v4, a
# NEW row; v3 keeps serving.
_POLICY_V3 = PerceptPolicy(
    name="v3",
    label="A27 combat",
    token_types={
        TOKEN_PROJECTILE: TokenType(
            has_player_id=False,
            scalars=(
                ("entity_rel", "int16", 3),
                ("entity_vel", "int16", 3),
            ),
        ),
        TOKEN_ACTOR: TokenType(
            has_player_id=True,
            scalars=(
                ("entity_half_extents", "uint8",   3),
                ("entity_rel",          "int16",   3),
                ("entity_vel",          "int16",   3),
                ("entity_path",         "int16",   3),
                ("entity_path_dist",    "uint16",  1),
                ("entity_eta",          "float16", 1),
                ("entity_facing",       "uint8",   1),
                ("entity_team",         "uint8",   1),
                ("entity_score",        "uint8",   1),
            ),
        ),
    },
    fields=(
        # entity_types pads with -1 (empty-slot sentinel the model's
        # mask reads); everything else zero-fills — mirrors
        # vec_env.pad_entities.
        ("entity_types",         "int8",    (),                   -1),
        ("entity_subject_id",    "uint8",   (),                    0),
        ("entity_modality_id",   "uint8",   (),                    0),
        ("entity_player_id",     "uint8",   (),                    0),
        ("entity_event_count",   "uint8",   (),                    0),
        ("entity_event_actions", "uint8",   (MAX_ENTITY_EVENTS,),  0),
        ("entity_event_sources", "uint8",   (MAX_ENTITY_EVENTS,),  0),
        ("entity_half_extents",  "uint8",   (3,),                  0),
        ("entity_rel",           "int16",   (3,),                  0),
        ("entity_vel",           "int16",   (3,),                  0),
        ("entity_path",          "int16",   (3,),                  0),
        ("entity_path_dist",     "uint16",  (),                    0),
        ("entity_eta",           "float16", (),                    0),
        ("entity_facing",        "uint8",   (),                    0),
        ("entity_team",          "uint8",   (),                    0),
        ("entity_score",         "uint8",   (),                    0),
    ),
)

# Policy "v1" — the f84c36cd^ full stream (QNN_ENTITY_MODE_FULL):
# projectile/actor/item/mover with recency and the sight/proximity/
# sound/memory modality ladder. Wire-shim only — new declarations
# should request v3+. One deliberate deviation from the f84c36cd^
# parser: unknown token tags FAIL LOUD here where the legacy parser
# broke out of the walk defensively (fail-loud beats silent
# truncation; the engine never emits unknown tags).
_POLICY_V1 = PerceptPolicy(
    name="v1",
    label="full-stream",
    token_types={
        TOKEN_PROJECTILE: TokenType(
            has_player_id=False,
            scalars=(
                ("entity_rel",     "int16",   3),
                ("entity_vel",     "int16",   3),
                ("entity_recency", "float16", 1),
            ),
        ),
        TOKEN_ACTOR: TokenType(
            has_player_id=True,
            scalars=(
                ("entity_half_extents", "uint8",   3),
                ("entity_rel",          "int16",   3),
                ("entity_vel",          "int16",   3),
                ("entity_path",         "int16",   3),
                ("entity_path_dist",    "uint16",  1),
                ("entity_eta",          "float16", 1),
                ("entity_facing",       "uint8",   1),
                ("entity_team",         "uint8",   1),
                ("entity_score",        "uint8",   1),
                ("entity_recency",      "float16", 1),
            ),
        ),
        TOKEN_ITEM: TokenType(
            has_player_id=False,
            scalars=(
                ("entity_half_extents", "uint8",   3),
                ("entity_rel",          "int16",   3),
                ("entity_path",         "int16",   3),
                ("entity_path_dist",    "uint16",  1),
                ("entity_eta",          "float16", 1),
                ("entity_amount",       "uint8",   1),
                ("entity_regen",        "float16", 1),
                ("entity_recency",      "float16", 1),
            ),
        ),
        TOKEN_MOVER: TokenType(
            has_player_id=False,
            scalars=(
                ("entity_half_extents", "uint8",   3),
                ("entity_rel",          "int16",   3),
                ("entity_path",         "int16",   3),
                ("entity_path_dist",    "uint16",  1),
                ("entity_eta",          "float16", 1),
                ("entity_state",        "uint8",   1),
                ("entity_recency",      "float16", 1),
            ),
        ),
    },
    fields=(
        ("entity_types",         "int8",    (),                   -1),
        ("entity_subject_id",    "uint8",   (),                    0),
        ("entity_modality_id",   "uint8",   (),                    0),
        ("entity_player_id",     "uint8",   (),                    0),
        ("entity_event_count",   "uint8",   (),                    0),
        ("entity_event_actions", "uint8",   (MAX_ENTITY_EVENTS,),  0),
        ("entity_event_sources", "uint8",   (MAX_ENTITY_EVENTS,),  0),
        ("entity_half_extents",  "uint8",   (3,),                  0),
        ("entity_rel",           "int16",   (3,),                  0),
        ("entity_vel",           "int16",   (3,),                  0),
        ("entity_path",          "int16",   (3,),                  0),
        ("entity_path_dist",     "uint16",  (),                    0),
        ("entity_eta",           "float16", (),                    0),
        ("entity_recency",       "float16", (),                    0),
        ("entity_facing",        "uint8",   (),                    0),
        ("entity_team",          "uint8",   (),                    0),
        ("entity_score",         "uint8",   (),                    0),
        ("entity_amount",        "uint8",   (),                    0),
        ("entity_regen",         "float16", (),                    0),
        ("entity_state",         "uint8",   (),                    0),
    ),
)

ENTITY_POLICIES: dict[str, PerceptPolicy] = {
    _POLICY_V1.name: _POLICY_V1,
    _POLICY_V3.name: _POLICY_V3,
}

# Registry sanity: no policy row may outgrow the shared row budget.
for _policy in ENTITY_POLICIES.values():
    for _tag, _tt in _policy.token_types.items():
        _row = 1 + 2 + int(_tt.has_player_id) + 1 + 2 * MAX_ENTITY_EVENTS + _tt.scalar_bytes
        assert _row <= MAX_PERCEPT_ROW_BYTES, (_policy.name, _tag, _row)


def entities_size(params: Mapping[str, Any]) -> int:
    """Maximum stream bytes: count byte + budgeted worst-case rows."""
    return 1 + int(params["max_tokens"]) * MAX_PERCEPT_ROW_BYTES


def _validate_entities_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params) - {"policy", "max_tokens", "paths"}
    if unknown:
        raise ValueError(f"unknown entities params: {sorted(unknown)}")
    missing = {"policy", "max_tokens", "paths"} - set(params)
    if missing:
        raise ValueError(f"missing entities params: {sorted(missing)}")
    policy, max_tokens, paths = params["policy"], params["max_tokens"], params["paths"]
    if policy not in ENTITY_POLICIES:
        raise ValueError(
            f"unknown entities policy {policy!r} "
            f"(registered: {sorted(ENTITY_POLICIES)})"
        )
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_TOKEN_OBJECTS:
        raise ValueError(
            f"entities max_tokens must be an int in [1, {MAX_TOKEN_OBJECTS}], "
            f"got {max_tokens!r}"
        )
    if not isinstance(paths, bool):
        raise ValueError(f"entities paths must be a bool, got {paths!r}")
    return {"policy": str(policy), "max_tokens": max_tokens, "paths": paths}


# ─────────────────────────────────────────────────────────────────
# Declaration — the model's request, validated against the registry
# ─────────────────────────────────────────────────────────────────

_DECLARATION_KEYS = {"obs_api", "state", "atlas", "entities", "frame_bytes"}


@dataclass(frozen=True)
class Declaration:
    """A validated obs declaration (agents/plans/obs-api.md §declaration).

    ``frame_bytes`` is the wire-shim-only explicit frame-size
    override; ``None`` (every new declaration) means the compiled
    minimum. Construct via :meth:`from_dict` / :meth:`from_json` —
    the constructor itself trusts its inputs.
    """

    state: tuple[str, ...]
    atlas: dict[str, Any] | None
    entities: dict[str, Any] | None
    frame_bytes: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Declaration":
        if not isinstance(payload, Mapping):
            raise ValueError(f"declaration must be a JSON object, got {type(payload).__name__}")
        unknown = set(payload) - _DECLARATION_KEYS
        if unknown:
            raise ValueError(f"unknown declaration keys: {sorted(unknown)}")
        version = payload.get("obs_api")
        if version != OBS_API_VERSION:
            raise ValueError(
                f"unsupported obs_api version {version!r} (expected {OBS_API_VERSION})"
            )
        state = payload.get("state")
        if not isinstance(state, (list, tuple)) or not all(isinstance(n, str) for n in state):
            raise ValueError("declaration state must be a list of field names")
        unknown_state = [n for n in state if n not in STATE_FIELDS_BY_NAME]
        if unknown_state:
            raise ValueError(
                f"unknown state fields: {unknown_state} "
                f"(registered: {[f.name for f in STATE_FIELDS]})"
            )
        if len(set(state)) != len(state):
            dupes = sorted({n for n in state if state.count(n) > 1})
            raise ValueError(f"duplicate state fields: {dupes}")
        atlas = payload.get("atlas")
        if atlas is not None:
            if not isinstance(atlas, Mapping):
                raise ValueError("declaration atlas must be an object or null")
            atlas = _validate_atlas_params(atlas)
        entities = payload.get("entities")
        if entities is not None:
            if not isinstance(entities, Mapping):
                raise ValueError("declaration entities must be an object or null")
            entities = _validate_entities_params(entities)
        frame_bytes = payload.get("frame_bytes")
        if frame_bytes is not None and (not isinstance(frame_bytes, int) or frame_bytes <= 0):
            raise ValueError(f"declaration frame_bytes must be a positive int, got {frame_bytes!r}")
        return cls(
            state=tuple(state),
            atlas=atlas,
            entities=entities,
            frame_bytes=frame_bytes,
        )

    @classmethod
    def from_json(cls, blob: str | bytes) -> "Declaration":
        return cls.from_dict(json.loads(blob))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "obs_api": OBS_API_VERSION,
            "state": list(self.state),
            "atlas": dict(self.atlas) if self.atlas is not None else None,
            "entities": dict(self.entities) if self.entities is not None else None,
        }
        if self.frame_bytes is not None:
            payload["frame_bytes"] = int(self.frame_bytes)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# ─────────────────────────────────────────────────────────────────
# Layout — the compiled per-seat frame description
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LayoutField:
    """One placed field: where its bytes live in the frame.

    ``dtype``/``shape`` describe the raw wire block for state and
    sensor fields; percepts are variable-length streams, so both are
    ``None`` and ``nbytes`` is the stream's byte budget.
    """

    name: str
    kind: str  # "state" | "sensor" | "percept"
    params: dict[str, Any]
    offset: int
    nbytes: int
    dtype: str | None
    shape: tuple[int, ...] | None

    def np_dtype(self) -> np.dtype:
        if self.dtype is None:
            raise ValueError(f"field {self.name!r} has no wire dtype (kind={self.kind})")
        return _dtype(self.dtype)


@dataclass(frozen=True)
class Layout:
    """Ordered placed fields + total frame size for one seat."""

    frame_bytes: int
    fields: tuple[LayoutField, ...]

    def field(self, name: str) -> LayoutField:
        for placed in self.fields:
            if placed.name == name:
                return placed
        raise KeyError(f"layout has no field {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_bytes": int(self.frame_bytes),
            "fields": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "params": dict(f.params),
                    "offset": int(f.offset),
                    "bytes": int(f.nbytes),
                    "dtype": f.dtype,
                    "shape": list(f.shape) if f.shape is not None else None,
                }
                for f in self.fields
            ],
        }


def compile_layout(declaration: Declaration) -> Layout:
    """Compile a declaration into its frame layout.

    MUST stay offset-identical to the C plan compiler: the ordering
    rule at the top of this module is the whole contract — registry
    order (state → atlas sensor → entities percept), dense offsets,
    registry-wide percept row budget, pose-tail reserve.
    """
    fields: list[LayoutField] = []
    offset = 0
    requested = set(declaration.state)
    for state_field in STATE_FIELDS:  # registry order, NOT declaration order
        if state_field.name not in requested:
            continue
        fields.append(LayoutField(
            name=state_field.name,
            kind="state",
            params={},
            offset=offset,
            nbytes=state_field.nbytes,
            dtype=state_field.dtype,
            shape=state_field.shape,
        ))
        offset += state_field.nbytes

    if declaration.atlas is not None:
        nbytes = atlas_size(declaration.atlas)
        fields.append(LayoutField(
            name=ATLAS_NAME,
            kind="sensor",
            params=dict(declaration.atlas),
            offset=offset,
            nbytes=nbytes,
            dtype="uint8",
            shape=atlas_shape(declaration.atlas),
        ))
        offset += nbytes

    if declaration.entities is not None:
        nbytes = entities_size(declaration.entities)
        fields.append(LayoutField(
            name=ENTITIES_NAME,
            kind="percept",
            params=dict(declaration.entities),
            offset=offset,
            nbytes=nbytes,
            dtype=None,
            shape=None,
        ))
        offset += nbytes

    min_frame = offset + POSE_TAIL_BYTES
    if declaration.frame_bytes is None:
        frame_bytes = min_frame
    else:
        frame_bytes = int(declaration.frame_bytes)
        if frame_bytes < min_frame:
            raise ValueError(
                f"declared frame_bytes {frame_bytes} < computed minimum {min_frame} "
                f"(payload {offset} + pose tail {POSE_TAIL_BYTES})"
            )
    return Layout(frame_bytes=frame_bytes, fields=tuple(fields))


def parse_layout_reply(payload: Mapping[str, Any]) -> Layout:
    """Reconstruct a Layout from the engine's handshake reply dict.

    Fail-loud mirror of :meth:`Layout.to_dict`; the caller is
    expected to compare the result against its own
    ``compile_layout(declaration)`` and refuse any divergence.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"layout reply must be an object, got {type(payload).__name__}")
    unknown = set(payload) - {"frame_bytes", "fields"}
    if unknown:
        raise ValueError(f"unknown layout reply keys: {sorted(unknown)}")
    frame_bytes = payload.get("frame_bytes")
    if not isinstance(frame_bytes, int) or frame_bytes <= 0:
        raise ValueError(f"layout frame_bytes must be a positive int, got {frame_bytes!r}")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, (list, tuple)):
        raise ValueError("layout fields must be a list")
    fields: list[LayoutField] = []
    for entry in raw_fields:
        unknown = set(entry) - {"name", "kind", "params", "offset", "bytes", "dtype", "shape"}
        if unknown:
            raise ValueError(f"unknown layout field keys: {sorted(unknown)}")
        shape = entry.get("shape")
        fields.append(LayoutField(
            name=str(entry["name"]),
            kind=str(entry["kind"]),
            params=dict(entry.get("params", {})),
            offset=int(entry["offset"]),
            nbytes=int(entry["bytes"]),
            dtype=None if entry.get("dtype") is None else str(entry["dtype"]),
            shape=None if shape is None else tuple(int(d) for d in shape),
        ))
    return Layout(frame_bytes=frame_bytes, fields=tuple(fields))


# ─────────────────────────────────────────────────────────────────
# Canonical declarations
# ─────────────────────────────────────────────────────────────────

# The default plan (no declaration sent): today's packed 864-byte
# frame, bit-identical — gate 1 in agents/plans/obs-api.md. Legacy
# drivers that send nothing get exactly this.
DEFAULT_DECLARATION = Declaration.from_dict({
    "obs_api": OBS_API_VERSION,
    "state": [f.name for f in STATE_FIELDS],
    "atlas": {"yaw": 24, "bands": ATLAS_BANDS, "packed": True},
    "entities": {"policy": "v3", "max_tokens": MAX_TOKEN_OBJECTS, "paths": True},
})
DEFAULT_LAYOUT = compile_layout(DEFAULT_DECLARATION)

# The f84c36cd^ generation (a26 rc1 line / wire.13.1 shim): 72-wide
# unpacked atlas, full entity stream, frame pinned at the historical
# OBS_BUFFER_SIZE of 4096 — that generation sized the buffer as a
# generous fixed constant (payload variable, tail zero-padded), not
# from the payload budget, so the shim declaration carries the
# explicit override.
LEGACY_UNPACKED_DECLARATION = Declaration.from_dict({
    "obs_api": OBS_API_VERSION,
    "state": [f.name for f in STATE_FIELDS],
    "atlas": {"yaw": 72, "bands": ATLAS_BANDS, "packed": False},
    "entities": {"policy": "v1", "max_tokens": MAX_TOKEN_OBJECTS, "paths": True},
    "frame_bytes": 4096,
})
LEGACY_UNPACKED_LAYOUT = compile_layout(LEGACY_UNPACKED_DECLARATION)


# ─────────────────────────────────────────────────────────────────
# Handshake framing (OP_ATTACH_DECL) — WS2 implements the C side
# ─────────────────────────────────────────────────────────────────

# Bridge-protocol opcode, shared across the worker binary channel and
# the arena server/client channels (free in all three opcode spaces).
# Request framing: u8 opcode, u8 seat_index (0 on single-seat
# channels), u32le json length, declaration JSON. Reply: one JSON
# line on the existing response channel —
#   {"ok": true, "layout": {frame_bytes, fields: [...]}}
# with the usual {"ok": false, "error": ...} on validation failure
# (hard error naming the offending registry entry — never a silent
# default).
OP_ATTACH_DECL = 8


def encode_attach_decl(declaration: Declaration, seat_index: int = 0) -> bytes:
    """Encode one OP_ATTACH_DECL request for the given seat."""
    if not 0 <= int(seat_index) <= 0xFF:
        raise ValueError(f"seat_index out of range: {seat_index}")
    blob = declaration.to_json().encode("utf-8")
    return struct.pack("<BBI", OP_ATTACH_DECL, int(seat_index), len(blob)) + blob


def decode_attach_decl(payload: bytes) -> tuple[int, Declaration]:
    """Decode one OP_ATTACH_DECL request → (seat_index, declaration).

    The Python driver never receives this message; the decoder exists
    so the framing is testable end-to-end before WS2 lands the C
    parser it specifies.
    """
    if len(payload) < 6:
        raise ValueError(f"attach-decl request truncated at {len(payload)} bytes")
    opcode, seat_index, length = struct.unpack_from("<BBI", payload, 0)
    if opcode != OP_ATTACH_DECL:
        raise ValueError(f"not an OP_ATTACH_DECL request (opcode {opcode})")
    blob = payload[6:]
    if len(blob) != length:
        raise ValueError(f"attach-decl length {length} != payload {len(blob)}")
    return seat_index, Declaration.from_json(blob)


def coerce_declaration(value: "Declaration | Mapping[str, Any] | str | bytes | None") -> Declaration | None:
    """Normalize the bridges' optional ``declaration`` inputs.

    Accepts an already-validated Declaration, a raw dict (validated
    here), JSON text/bytes, or None (= legacy default plan, nothing
    sent on the wire).
    """
    if value is None or isinstance(value, Declaration):
        return value
    if isinstance(value, (str, bytes)):
        return Declaration.from_json(value)
    if isinstance(value, Mapping):
        return Declaration.from_dict(value)
    raise ValueError(f"cannot coerce {type(value).__name__} into an obs declaration")


# ─────────────────────────────────────────────────────────────────
# Wire-shim — stamped pre-obs-api generations → equivalent plans
# ─────────────────────────────────────────────────────────────────
#
# The deployed fleet carries monolithic wire-generation stamps
# (`wire_contract` = wire.12.1 / 12.2 / 13.1 / 13.2). Under the obs
# API those artifacts do not declare; this table maps each stamped id
# onto the equivalent declaration (agents/plans/obs-api.md §handshake)
# so existing models keep loading unchanged. Python mirror of the
# WS2 C-side shim; new models stamp `obs_declaration` and never
# consult this table.
#
# Every row pins its generation's historical buffer constant as the
# explicit ``frame_bytes`` override (the pre-obs-api emitters sized
# the frame as a pinned constant, not from the payload budget):
# 864 for the packed-atlas families, 4096 for the 72-unpacked ones.
#
# ┌──────────────────────────────────────────────────────────────┐
# │ UNRESOLVED BINDING — WS5 flips this in ONE place.            │
# │                                                              │
# │ The entity-policy names below are provisional: the live C    │
# │ emit writes a26 FULL entity rows (item/mover/recency — the   │
# │ "v1" walk) while the a27 python parse reads pure-combat rows │
# │ ("v3"), a pre-existing emit↔parse desync being reconciled at │
# │ WS5 integration. The verified corpus evidence (qwd_v3 AND    │
# │ qwd_v4 caches both store populated item/mover/recency/amount │
# │ /state fields) says both a26 lines trained on the full       │
# │ stream. Each shim row reads its policy from exactly one of   │
# │ the two constants below so integration can rebind either     │
# │ family without touching the rows.                            │
# └──────────────────────────────────────────────────────────────┘
WIRE12_ENTITIES_POLICY = "v1"   # a26 lines: full stream (matches their corpora)
WIRE13_ENTITIES_POLICY = "v3"   # a27 lines: pure-combat parse (current wire.py walk)


@dataclass(frozen=True)
class WireShimRow:
    """One stamped generation: its wire id, pinned semantics contract,
    and the equivalent obs declaration its artifacts route to."""

    wire_id: str
    semantics: str
    declaration: Declaration


def _shim_declaration(*, with_self_weapon_id: bool, yaw: int, packed: bool,
                      policy: str, frame_bytes: int) -> Declaration:
    state = [f.name for f in STATE_FIELDS
             if with_self_weapon_id or f.name != "self_weapon_id"]
    return Declaration.from_dict({
        "obs_api": OBS_API_VERSION,
        "state": state,
        "atlas": {"yaw": yaw, "bands": ATLAS_BANDS, "packed": packed},
        "entities": {"policy": policy, "max_tokens": MAX_TOKEN_OBJECTS, "paths": True},
        "frame_bytes": frame_bytes,
    })


# The 12.x rows carry the full 13-field state (self_weapon_id was a
# consumed input on the a26 line); the 13.x rows drop it (the a27
# 9-way attack-with head IS the select-and-fire decision — no
# engine-equipped-weapon input in the a27 obs contract).
WIRE_SHIM: dict[str, WireShimRow] = {
    row.wire_id: row for row in (
        WireShimRow("wire.12.1", "semantics.1", _shim_declaration(
            with_self_weapon_id=True, yaw=72, packed=False,
            policy=WIRE12_ENTITIES_POLICY, frame_bytes=4096)),
        WireShimRow("wire.12.2", "semantics.1", _shim_declaration(
            with_self_weapon_id=True, yaw=24, packed=True,
            policy=WIRE12_ENTITIES_POLICY, frame_bytes=864)),
        WireShimRow("wire.13.1", "semantics.2", _shim_declaration(
            with_self_weapon_id=False, yaw=72, packed=False,
            policy=WIRE13_ENTITIES_POLICY, frame_bytes=4096)),
        WireShimRow("wire.13.2", "semantics.2", _shim_declaration(
            with_self_weapon_id=False, yaw=24, packed=True,
            policy=WIRE13_ENTITIES_POLICY, frame_bytes=864)),
    )
}


# ─────────────────────────────────────────────────────────────────
# Frame packer — registry-mirror inverse of qnn.wire.unpack_frame
# ─────────────────────────────────────────────────────────────────

class UnrepresentableTokenError(ValueError):
    """A token tag the layout's declared entity policy cannot carry."""


def _pack_block(buf: bytearray, offset: int, value: Any,
                dtype: np.dtype, shape: tuple[int, ...], name: str) -> None:
    arr = np.asarray(value)
    if arr.dtype != dtype:
        raise ValueError(
            f"pack_frame: field {name!r} has dtype {arr.dtype}, wire wants {dtype}"
        )
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"pack_frame: field {name!r} has shape {tuple(arr.shape)}, "
            f"wire wants {tuple(shape)}"
        )
    raw = np.ascontiguousarray(arr).tobytes()
    buf[offset:offset + len(raw)] = raw


def _require(fields: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in fields:
        raise ValueError(f"pack_frame: missing field {key!r} required by {label}")
    return fields[key]


def _pack_entity_stream(buf: bytearray, field: LayoutField,
                        fields: Mapping[str, Any]) -> None:
    policy = ENTITY_POLICIES[field.params["policy"]]
    label = f"the {policy.label} entity stream (policy {policy.name!r})"
    types = np.asarray(_require(fields, "entity_types", label))
    n = int(types.shape[0])
    max_tokens = int(field.params["max_tokens"])
    if n > max_tokens:
        raise ValueError(
            f"pack_frame: {n} entity tokens exceed the declared "
            f"max_tokens {max_tokens}"
        )
    if "entity_count" in fields and int(np.asarray(fields["entity_count"])) != n:
        raise ValueError(
            f"pack_frame: entity_count {int(np.asarray(fields['entity_count']))} "
            f"!= len(entity_types) {n}"
        )
    subjects   = _require(fields, "entity_subject_id", label)
    modalities = _require(fields, "entity_modality_id", label)
    evt_counts  = _require(fields, "entity_event_count", label)
    evt_actions = _require(fields, "entity_event_actions", label)
    evt_sources = _require(fields, "entity_event_sources", label)
    pos = field.offset
    buf[pos] = n; pos += 1
    for i in range(n):
        tag = int(types[i])
        token_type = policy.token_types.get(tag)
        if token_type is None:
            raise UnrepresentableTokenError(
                f"pack_frame: token tag {tag} is not representable in {label} "
                f"(registered tags: {sorted(policy.token_types)})"
            )
        buf[pos] = tag; pos += 1
        buf[pos] = int(subjects[i]); pos += 1
        buf[pos] = int(modalities[i]); pos += 1
        if token_type.has_player_id:
            buf[pos] = int(_require(fields, "entity_player_id", label)[i]); pos += 1
        ne = int(evt_counts[i])
        if ne > MAX_ENTITY_EVENTS:
            raise ValueError(
                f"pack_frame: token {i} carries {ne} events "
                f"(max {MAX_ENTITY_EVENTS})"
            )
        buf[pos] = ne; pos += 1
        for j in range(ne):
            buf[pos] = int(np.asarray(evt_actions)[i, j]); pos += 1
            buf[pos] = int(np.asarray(evt_sources)[i, j]); pos += 1
        for key, dtype_name, count in token_type.scalars:
            dtype = _dtype(dtype_name)
            row = np.asarray(_require(fields, key, label)[i])
            if row.dtype != dtype:
                raise ValueError(
                    f"pack_frame: field {key!r} has dtype {row.dtype}, "
                    f"wire wants {dtype}"
                )
            if int(row.size) != count:
                raise ValueError(
                    f"pack_frame: field {key!r} token row has {int(row.size)} "
                    f"cells, wire wants {count}"
                )
            raw = np.ascontiguousarray(row).tobytes()
            buf[pos:pos + len(raw)] = raw
            pos += len(raw)
    if pos > field.offset + field.nbytes:
        raise AssertionError(
            f"pack_frame: entity stream overran its budget "
            f"({pos - field.offset} > {field.nbytes})"
        )


def pack_frame(fields: Mapping[str, Any], layout: Layout) -> bytes:
    """Serialize per-field arrays into one obs frame at `layout`.

    Registry-mirror INVERSE of :func:`qnn.wire.unpack_frame` — for any
    field dict that unpack_frame can produce, ``unpack_frame(
    pack_frame(d, layout), layout) == d`` bitwise. Exists for the
    gate-2 replay harness (qnn.obs_api_gate) and layout tests; the
    engine's emitters remain the only production frame writers.

    Fail loud, no silent defaults: a missing field, wrong dtype/shape,
    oversized token/event counts, or a token tag the declared entity
    policy cannot carry (:class:`UnrepresentableTokenError`) all
    raise. Bytes not covered by the layout (padding, the pose tail)
    are zero.
    """
    buf = bytearray(layout.frame_bytes)
    for field in layout.fields:
        if field.kind == "state":
            _pack_block(buf, field.offset, _require(fields, field.name, "the layout"),
                        field.np_dtype(), field.shape, field.name)
        elif field.kind == "sensor":
            _pack_block(buf, field.offset, _require(fields, ATLAS_OUT_KEY, "the layout"),
                        field.np_dtype(), field.shape, ATLAS_OUT_KEY)
        elif field.kind == "percept":
            _pack_entity_stream(buf, field, fields)
        else:
            raise ValueError(f"unknown layout field kind {field.kind!r}")
    return bytes(buf)


# ─────────────────────────────────────────────────────────────────
# Checkpoint-contract derivation — training-side declarations
# ─────────────────────────────────────────────────────────────────
#
# ``declaration_for_run`` answers "which obs declaration describes the
# frames this training run consumed", so eval/h2h can attach per-seat
# declarations for python-side models (agents/plans/obs-api.md §WS4).
# Two canonical answers exist today:
#
#   * packed 24×11 training frames  → DEFAULT_DECLARATION (the 864 B
#     default plan — a27 HEAD and the a26 rc2 recollect line),
#   * 72×11 unpacked frames         → LEGACY_UNPACKED_DECLARATION
#     (the f84c36cd^ 4096 B frame — the rc1 lines).
#
# NOTE the run's atlas width is NOT recorded in probe.json / config —
# it was an engine_norm code constant at the run's git commit (the two
# reference a26 runs carry byte-identical probe.json at both widths).
# Resolution therefore keys on, in order:
#   1. a split wire id on the checkpoint sidecar contract
#      (checkpoints/best_<run_id>.json → meta contract, e.g.
#      wire.13.2), which pins the atlas family directly;
#   2. for pre-split bare stamps (wire.12 / wire.13): the atlas block
#      width stored in the run's OWN training corpus cache
#      (bc_manifest.json → bc_data_dir → shard000000_obs_spatial_atlas
#      .npy header) — the training frames themselves are the truth.
# Anything else fails loud; there is no default declaration for an
# unrecognized run.

# a26 lines (wire.12.x) resolve to their SHIM declarations — full-stream
# entities (v1, the binding gate 2 proved against both corpora) with the
# self_weapon_id key exposed (a consumed a26 model input). The a27 rows keep
# the combat-walk declarations pending the C v3 combat serializer.
_TRAINING_DECLARATIONS: dict[str, Declaration] = {
    "wire.12.1": WIRE_SHIM["wire.12.1"].declaration,
    "wire.12.2": WIRE_SHIM["wire.12.2"].declaration,
    "wire.13.1": LEGACY_UNPACKED_DECLARATION,
    "wire.13.2": DEFAULT_DECLARATION,
}
_BARE_PRE_SPLIT_WIRES = ("wire.12", "wire.13")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"declaration_for_run: cannot read {path}: {exc}") from exc


def _run_checkpoint_sidecar(run_dir: Path) -> Path:
    """The run's checkpoint meta sidecar (best_<run_id>.json)."""
    ckpt_dir = run_dir / "checkpoints"
    run_json = run_dir / "run.json"
    if run_json.is_file():
        run_id = _read_json(run_json).get("run_id")
        if run_id:
            candidate = ckpt_dir / f"best_{run_id}.json"
            if candidate.is_file():
                return candidate
    candidates = sorted(ckpt_dir.glob("best_*.json"))
    if len(candidates) == 1:
        return candidates[0]
    # PPO run layout: the best checkpoint (and its meta sidecar) live one
    # level deeper, at checkpoints/best/best_model.json (qnn.ppo.train's
    # best-dir save). Additive fallback — BC layouts never have it.
    ppo_best = ckpt_dir / "best" / "best_model.json"
    if ppo_best.is_file():
        return ppo_best
    raise ValueError(
        f"declaration_for_run: cannot pick a checkpoint sidecar in {ckpt_dir} "
        f"(found {[c.name for c in candidates]}); need exactly one best_*.json, "
        "a run.json naming the run_id, or a PPO best/best_model.json"
    )


def training_corpus_for_run(run_dir: Path | str,
                            corpus_dir: Path | str | None = None) -> Path:
    """Resolve the corpus cache dir this run trained on.

    ``corpus_dir`` overrides; otherwise the bc_manifest's ``bc_data_dir``
    (usually repo-relative, e.g. ``artifacts/collect/qwd_v4``) is tried
    as given and then against every ancestor of ``run_dir`` (which
    finds the repo/workspace root from ``runs/<mode>/<name>``).
    """
    run_dir = Path(run_dir)
    if corpus_dir is not None:
        corpus = Path(corpus_dir)
        if not corpus.is_dir():
            raise ValueError(f"training_corpus_for_run: no such corpus dir {corpus}")
        return corpus
    manifest = run_dir / "checkpoints" / "bc_manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f"training_corpus_for_run: {manifest} not found and no corpus_dir given"
        )
    bc_data_dir = _read_json(manifest).get("config", {}).get("bc_data_dir")
    if not bc_data_dir:
        raise ValueError(
            f"training_corpus_for_run: {manifest} carries no config.bc_data_dir"
        )
    candidates = [Path(bc_data_dir)]
    candidates += [ancestor / bc_data_dir for ancestor in run_dir.resolve().parents]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ValueError(
        f"training_corpus_for_run: corpus {bc_data_dir!r} (from {manifest}) is not "
        "reachable from here — pass corpus_dir explicitly"
    )


def _corpus_atlas_bytes_per_band(corpus: Path) -> int:
    sample = corpus / "precomputed_train" / "shard000000_obs_spatial_atlas.npy"
    if not sample.is_file():
        raise ValueError(
            f"declaration_for_run: {sample} not found — cannot resolve the "
            "training atlas width for a pre-split wire stamp"
        )
    shape = np.load(sample, mmap_mode="r").shape
    if len(shape) != 3 or shape[1] != ATLAS_BANDS:
        raise ValueError(
            f"declaration_for_run: {sample} has shape {shape}, expected "
            f"(rows, {ATLAS_BANDS}, bytes_per_band)"
        )
    return int(shape[2])


def declaration_for_run(run_dir: Path | str,
                        corpus_dir: Path | str | None = None) -> Declaration:
    """Derive the training-side obs declaration for a BC run dir.

    Returns one of the two canonical declarations (see the section
    comment above for the resolution order and why probe.json alone
    cannot discriminate). The result describes the FRAME the run's
    collect/training parse consumed — it always carries the full
    13-field state block (the a27 parse drops the ``self_weapon_id``
    KEY after unpack, but the byte is in the frame).

    a26-line runs (split or bare wire.12 stamps) resolve to their
    WIRE_SHIM declarations: full-stream entities (v1 — the binding
    gate 2 proved bit-exact against qwd_v3 and qwd_v4) with
    ``self_weapon_id`` exposed. a27 rows keep the combat-walk
    declarations until the C combat-row serializer lands.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"declaration_for_run: no such run dir {run_dir}")
    sidecar = _run_checkpoint_sidecar(run_dir)
    contract = _read_json(sidecar).get("contract")
    if not isinstance(contract, Mapping) or "wire" not in contract:
        raise ValueError(
            f"declaration_for_run: {sidecar} carries no contract.wire "
            "(stamp the checkpoint first — tools/stamp_checkpoint.py)"
        )
    wire = str(contract["wire"])
    if wire in _TRAINING_DECLARATIONS:
        return _TRAINING_DECLARATIONS[wire]
    if wire not in _BARE_PRE_SPLIT_WIRES:
        raise ValueError(
            f"declaration_for_run: unrecognized wire id {wire!r} on {sidecar} "
            f"(known: {sorted(_TRAINING_DECLARATIONS)} + bare "
            f"{list(_BARE_PRE_SPLIT_WIRES)})"
        )
    # Pre-split bare stamp: the training corpus atlas block is the truth.
    corpus = training_corpus_for_run(run_dir, corpus_dir)
    per_band = _corpus_atlas_bytes_per_band(corpus)
    packed_bytes = atlas_shape(DEFAULT_DECLARATION.atlas)[1]        # 12
    unpacked_bytes = atlas_shape(LEGACY_UNPACKED_DECLARATION.atlas)[1]  # 72
    # Bare stamps only exist on pre-split checkpoints, which are a26-line by
    # construction — resolve to the wire.12.x shim declarations (v1 entities).
    if per_band == packed_bytes:
        return WIRE_SHIM["wire.12.2"].declaration
    if per_band == unpacked_bytes:
        return WIRE_SHIM["wire.12.1"].declaration
    raise ValueError(
        f"declaration_for_run: corpus {corpus} stores {per_band} atlas bytes "
        f"per band — expected {packed_bytes} (packed 24×11) or "
        f"{unpacked_bytes} (unpacked 72×11)"
    )
