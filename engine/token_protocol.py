"""Binary token-step protocol for the customized native client."""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import Any, Dict, List

from quake_ai.vocab import (
    ACTION_NAMES,
    MODALITY_NAMES,
    QUALIFIER_NAMES,
    SPATIAL_SECTOR_NAMES,
    SUBJECT_NAMES,
)

TOKEN_BINARY_MAGIC = b"QTOK"
TOKEN_BINARY_VERSION = 5

_HEADER_STRUCT = struct.Struct("<4sHHIIiHHHH")
TOKEN_BINARY_HEADER_SIZE = _HEADER_STRUCT.size

# v5 self token: 23 floats (scalars) + 3 int32s (embedding IDs)
_SELF_STRUCT = struct.Struct("<23f3i")
# v5 object token: u32 handle + 5 u16 IDs + 8 floats (scalars) + u16 event_count + u16 event_base + u16 route_cluster_count + 8 u16 route_cluster_ids
_OBJECT_STRUCT = struct.Struct("<I5H8f2H9H")
_EVENT_STRUCT = struct.Struct("<HHHH3f")
# v5 spatial: u16 sector_id + u16 reserved + 10 floats (reordered)
_SPATIAL_STRUCT = struct.Struct("<HH10f")

_FLAG_RESET = 1 << 0
_FLAG_DONE = 1 << 1
_FLAG_HAS_ACTION = 1 << 2

_ACTION_STRUCT = struct.Struct("<7H")


@dataclass(slots=True)
class TrustedSelfToken:
    """V5 self token — pre-normalized scalars + embedding IDs."""
    health: float
    armor: float
    armor_type: float
    weapon_bits: List[float]  # 7 bitmask bits
    weapon_super: float
    ammo: List[float]  # 4 values (shells, nails, rockets, cells)
    velocity: List[float]  # 3 values (x, y, z)
    yaw_sin: float
    yaw_cos: float
    pitch_sin: float
    pitch_cos: float
    dt: float
    weapon_id: int
    movement_id: int
    cluster_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health": float(self.health),
            "armor": float(self.armor),
            "armor_type": float(self.armor_type),
            "weapon_bits": list(self.weapon_bits),
            "weapon_super": float(self.weapon_super),
            "ammo": list(self.ammo),
            "velocity": list(self.velocity),
            "yaw_sin": float(self.yaw_sin),
            "yaw_cos": float(self.yaw_cos),
            "pitch_sin": float(self.pitch_sin),
            "pitch_cos": float(self.pitch_cos),
            "dt": float(self.dt),
            "weapon_id": int(self.weapon_id),
            "movement_id": int(self.movement_id),
            "cluster_id": int(self.cluster_id),
        }


@dataclass(slots=True)
class TrustedEventAtom:
    subject_id: int
    action_id: int
    qualifier_id: int
    modality_id: int
    recency: float
    confidence: float
    magnitude: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": int(self.subject_id),
            "subject": SUBJECT_NAMES.get(int(self.subject_id), "UNKNOWN"),
            "action_id": int(self.action_id),
            "action": ACTION_NAMES.get(int(self.action_id), "UNKNOWN"),
            "qualifier_id": int(self.qualifier_id),
            "qualifier": QUALIFIER_NAMES.get(int(self.qualifier_id), "UNKNOWN"),
            "modality_id": int(self.modality_id),
            "modality": MODALITY_NAMES.get(int(self.modality_id), "UNKNOWN"),
            "recency": float(self.recency),
            "confidence": float(self.confidence),
            "magnitude": float(self.magnitude),
        }


@dataclass(slots=True)
class TrustedObjectToken:
    handle: int
    subject_id: int
    qualifier_id: int
    modality_id: int
    player_id: int
    cluster_id: int
    rel_x: float
    rel_y: float
    rel_z: float
    route_cost: float
    recency: float
    confidence: float
    magnitude: float
    state: float
    route_cluster_ids: List[int] = field(default_factory=list)
    events: List[TrustedEventAtom] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handle": int(self.handle),
            "subject_id": int(self.subject_id),
            "subject": SUBJECT_NAMES.get(int(self.subject_id), "UNKNOWN"),
            "qualifier_id": int(self.qualifier_id),
            "qualifier": QUALIFIER_NAMES.get(int(self.qualifier_id), "UNKNOWN"),
            "modality_id": int(self.modality_id),
            "modality": MODALITY_NAMES.get(int(self.modality_id), "UNKNOWN"),
            "player_id": int(self.player_id),
            "cluster_id": int(self.cluster_id),
            "rel_x": float(self.rel_x),
            "rel_y": float(self.rel_y),
            "rel_z": float(self.rel_z),
            "route_cost": float(self.route_cost),
            "recency": float(self.recency),
            "confidence": float(self.confidence),
            "magnitude": float(self.magnitude),
            "state": float(self.state),
            "route_cluster_ids": list(self.route_cluster_ids),
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(slots=True)
class TrustedSpatialToken:
    sector_id: int
    nearest_dist: float
    mean_dist: float
    openness: float
    clearance: float
    traversable: float
    dropoff: float
    solid_frac: float
    water_frac: float
    slime_frac: float
    lava_frac: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_id": int(self.sector_id),
            "sector": SPATIAL_SECTOR_NAMES.get(int(self.sector_id), "UNKNOWN"),
            "nearest_dist": float(self.nearest_dist),
            "mean_dist": float(self.mean_dist),
            "openness": float(self.openness),
            "clearance": float(self.clearance),
            "traversable": float(self.traversable),
            "dropoff": float(self.dropoff),
            "solid_frac": float(self.solid_frac),
            "water_frac": float(self.water_frac),
            "slime_frac": float(self.slime_frac),
            "lava_frac": float(self.lava_frac),
        }


@dataclass(slots=True)
class TrustedTokenTick:
    tick: int
    steps: int
    current_region_id: int
    self_token: TrustedSelfToken
    object_tokens: List[TrustedObjectToken]
    spatial_tokens: List[TrustedSpatialToken]
    done: bool = False
    reset: bool = False
    tick_hz: int = 20
    action_label: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "tick": int(self.tick),
            "steps": int(self.steps),
            "current_region_id": int(self.current_region_id),
            "tick_hz": int(self.tick_hz),
            "self_token": self.self_token.to_dict(),
            "object_tokens": [row.to_dict() for row in self.object_tokens],
            "spatial_tokens": [row.to_dict() for row in self.spatial_tokens],
            "done": bool(self.done),
            "reset": bool(self.reset),
        }
        if self.action_label:
            d["action_label"] = dict(self.action_label)
        return d


def decode_binary_token_tick(header: bytes, read_exact) -> TrustedTokenTick:
    (
        magic,
        version,
        flags,
        tick,
        steps,
        current_region_id,
        object_count,
        event_count,
        spatial_count,
        tick_hz,
    ) = _HEADER_STRUCT.unpack(header)
    if magic != TOKEN_BINARY_MAGIC:
        raise ValueError(f"Unexpected token packet magic {magic!r}")
    if version != TOKEN_BINARY_VERSION:
        raise ValueError(f"Unsupported token packet version {version}")

    # v5 self token: 23 scalars + 3 IDs
    self_values = _SELF_STRUCT.unpack(read_exact(_SELF_STRUCT.size))
    self_token = TrustedSelfToken(
        health=float(self_values[0]),
        armor=float(self_values[1]),
        armor_type=float(self_values[2]),
        weapon_bits=[float(self_values[i]) for i in range(3, 10)],
        weapon_super=float(self_values[10]),
        ammo=[float(self_values[i]) for i in range(11, 15)],
        velocity=[float(self_values[i]) for i in range(15, 18)],
        yaw_sin=float(self_values[18]),
        yaw_cos=float(self_values[19]),
        pitch_sin=float(self_values[20]),
        pitch_cos=float(self_values[21]),
        dt=float(self_values[22]),
        weapon_id=int(self_values[23]),
        movement_id=int(self_values[24]),
        cluster_id=int(self_values[25]),
    )

    # v5 object tokens: u32 handle + 5 u16 IDs + 8 floats + u16 event_count + u16 event_base
    raw_objects = []
    for _ in range(object_count):
        raw_objects.append(_OBJECT_STRUCT.unpack(read_exact(_OBJECT_STRUCT.size)))

    raw_events: List[TrustedEventAtom] = []
    for _ in range(event_count):
        subject_id, action_id, qualifier_id, modality_id, recency, confidence, magnitude = _EVENT_STRUCT.unpack(
            read_exact(_EVENT_STRUCT.size)
        )
        raw_events.append(
            TrustedEventAtom(
                subject_id=int(subject_id),
                action_id=int(action_id),
                qualifier_id=int(qualifier_id),
                modality_id=int(modality_id),
                recency=float(recency),
                confidence=float(confidence),
                magnitude=float(magnitude),
            )
        )

    object_tokens: List[TrustedObjectToken] = []
    for row in raw_objects:
        (
            handle,
            subject_id,
            qualifier_id,
            modality_id,
            player_id,
            cluster_id,
            rel_x,
            rel_y,
            rel_z,
            route_cost,
            recency,
            confidence,
            magnitude,
            state,
            event_count_local,
            event_base,
            route_cluster_count,
            *route_cluster_raw,
        ) = row
        events = raw_events[event_base: event_base + event_count_local]
        n_route = min(int(route_cluster_count), len(route_cluster_raw))
        route_clusters = [int(route_cluster_raw[i]) for i in range(n_route)]
        object_tokens.append(
            TrustedObjectToken(
                handle=int(handle),
                subject_id=int(subject_id),
                qualifier_id=int(qualifier_id),
                modality_id=int(modality_id),
                player_id=int(player_id),
                cluster_id=int(cluster_id),
                rel_x=float(rel_x),
                rel_y=float(rel_y),
                rel_z=float(rel_z),
                route_cost=float(route_cost),
                recency=float(recency),
                confidence=float(confidence),
                magnitude=float(magnitude),
                state=float(state),
                route_cluster_ids=route_clusters,
                events=list(events),
            )
        )

    # v5 spatial: reordered geometry→traversability→content
    spatial_tokens: List[TrustedSpatialToken] = []
    for _ in range(spatial_count):
        (
            sector_id,
            _reserved,
            nearest_dist,
            mean_dist,
            openness,
            clearance,
            traversable,
            dropoff,
            solid_frac,
            water_frac,
            slime_frac,
            lava_frac,
        ) = _SPATIAL_STRUCT.unpack(read_exact(_SPATIAL_STRUCT.size))
        spatial_tokens.append(
            TrustedSpatialToken(
                sector_id=int(sector_id),
                nearest_dist=float(nearest_dist),
                mean_dist=float(mean_dist),
                openness=float(openness),
                clearance=float(clearance),
                traversable=float(traversable),
                dropoff=float(dropoff),
                solid_frac=float(solid_frac),
                water_frac=float(water_frac),
                slime_frac=float(slime_frac),
                lava_frac=float(lava_frac),
            )
        )

    action_label: Dict[str, int] = {}
    if flags & _FLAG_HAS_ACTION:
        move, strafe, look_yaw, look_pitch, fire, jump, weapon = _ACTION_STRUCT.unpack(
            read_exact(_ACTION_STRUCT.size)
        )
        action_label = {
            "move": int(move),
            "strafe": int(strafe),
            "look_yaw": int(look_yaw),
            "look_pitch": int(look_pitch),
            "fire": int(fire),
            "jump": int(jump),
            "weapon": int(weapon),
        }

    return TrustedTokenTick(
        tick=int(tick),
        steps=int(steps),
        current_region_id=int(current_region_id),
        self_token=self_token,
        object_tokens=object_tokens,
        spatial_tokens=spatial_tokens,
        done=bool(flags & _FLAG_DONE),
        reset=bool(flags & _FLAG_RESET),
        tick_hz=int(tick_hz),
        action_label=action_label,
    )


def encode_binary_token_tick(tick: TrustedTokenTick) -> bytes:
    """Encode a TrustedTokenTick back to the QTOK v5 binary wire format."""
    all_events: List[TrustedEventAtom] = []
    event_bases: List[int] = []
    for obj in tick.object_tokens:
        event_bases.append(len(all_events))
        all_events.extend(obj.events)

    flags = 0
    if tick.reset:
        flags |= _FLAG_RESET
    if tick.done:
        flags |= _FLAG_DONE
    if tick.action_label:
        flags |= _FLAG_HAS_ACTION

    st = tick.self_token
    parts = [
        _HEADER_STRUCT.pack(
            TOKEN_BINARY_MAGIC,
            TOKEN_BINARY_VERSION,
            flags,
            tick.tick,
            tick.steps,
            tick.current_region_id,
            len(tick.object_tokens),
            len(all_events),
            len(tick.spatial_tokens),
            tick.tick_hz,
        ),
        _SELF_STRUCT.pack(
            st.health,
            st.armor,
            st.armor_type,
            *st.weapon_bits,
            st.weapon_super,
            *st.ammo,
            *st.velocity,
            st.yaw_sin,
            st.yaw_cos,
            st.pitch_sin,
            st.pitch_cos,
            st.dt,
            st.weapon_id,
            st.movement_id,
            st.cluster_id,
        ),
    ]
    for idx, obj in enumerate(tick.object_tokens):
        rc = list(obj.route_cluster_ids or [])
        rc_count = len(rc)
        rc_padded = rc + [0] * (8 - len(rc))
        parts.append(
            _OBJECT_STRUCT.pack(
                obj.handle,
                obj.subject_id,
                obj.qualifier_id,
                obj.modality_id,
                obj.player_id,
                obj.cluster_id,
                obj.rel_x,
                obj.rel_y,
                obj.rel_z,
                obj.route_cost,
                obj.recency,
                obj.confidence,
                obj.magnitude,
                obj.state,
                len(obj.events),
                event_bases[idx],
                rc_count,
                *rc_padded,
            )
        )
    for ev in all_events:
        parts.append(
            _EVENT_STRUCT.pack(
                ev.subject_id,
                ev.action_id,
                ev.qualifier_id,
                ev.modality_id,
                ev.recency,
                ev.confidence,
                ev.magnitude,
            )
        )
    for sp in tick.spatial_tokens:
        parts.append(
            _SPATIAL_STRUCT.pack(
                sp.sector_id,
                0,
                sp.nearest_dist,
                sp.mean_dist,
                sp.openness,
                sp.clearance,
                sp.traversable,
                sp.dropoff,
                sp.solid_frac,
                sp.water_frac,
                sp.slime_frac,
                sp.lava_frac,
            )
        )
    if tick.action_label:
        parts.append(
            _ACTION_STRUCT.pack(
                tick.action_label.get("move", 0),
                tick.action_label.get("strafe", 0),
                tick.action_label.get("look_yaw", 0),
                tick.action_label.get("look_pitch", 0),
                tick.action_label.get("fire", 0),
                tick.action_label.get("jump", 0),
                tick.action_label.get("weapon", 0),
            )
        )
    return b"".join(parts)


def write_token_ticks_file(path: str, ticks: List[TrustedTokenTick]) -> None:
    """Write a sequence of TrustedTokenTick objects as concatenated QTOK packets."""
    with open(path, "wb") as f:
        for tick in ticks:
            f.write(encode_binary_token_tick(tick))


def read_token_ticks_file(path: str) -> List[TrustedTokenTick]:
    """Read a sequence of binary QTOK packets from a file."""
    ticks: List[TrustedTokenTick] = []
    with open(path, "rb") as f:
        def read_exact(n: int) -> bytes:
            data = f.read(n)
            if len(data) != n:
                raise EOFError(f"Expected {n} bytes, got {len(data)}")
            return data

        while True:
            header = f.read(TOKEN_BINARY_HEADER_SIZE)
            if not header:
                break
            if len(header) < TOKEN_BINARY_HEADER_SIZE:
                raise ValueError(f"Truncated header: {len(header)} bytes")
            ticks.append(decode_binary_token_tick(header, read_exact))
    return ticks
