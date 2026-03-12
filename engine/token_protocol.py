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
TOKEN_BINARY_VERSION = 3

_HEADER_STRUCT = struct.Struct("<4sHHIIiHHHH")
TOKEN_BINARY_HEADER_SIZE = _HEADER_STRUCT.size

_SELF_STRUCT = struct.Struct("<9fiif9i")
_OBJECT_STRUCT = struct.Struct("<IHHHH7fHH")
_EVENT_STRUCT = struct.Struct("<HHHH3f")
_SPATIAL_STRUCT = struct.Struct("<HH10f")

_FLAG_RESET = 1 << 0
_FLAG_DONE = 1 << 1
_FLAG_HAS_ACTION = 1 << 2

_ACTION_STRUCT = struct.Struct("<7H")


@dataclass(slots=True)
class TrustedSelfToken:
    origin: List[float]
    velocity: List[float]
    view_angles: List[float]
    health: int
    armor: int
    armor_type: float
    ammo_shells: int
    ammo_nails: int
    ammo_rockets: int
    ammo_cells: int
    weapon_id: int
    weapons_owned: int
    grounded: bool
    waterlevel: int
    current_region_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": list(self.origin),
            "velocity": list(self.velocity),
            "view_angles": list(self.view_angles),
            "health": int(self.health),
            "armor": int(self.armor),
            "armor_type": float(self.armor_type),
            "ammo_shells": int(self.ammo_shells),
            "ammo_nails": int(self.ammo_nails),
            "ammo_rockets": int(self.ammo_rockets),
            "ammo_cells": int(self.ammo_cells),
            "weapon_id": int(self.weapon_id),
            "weapons_owned": int(self.weapons_owned),
            "grounded": bool(self.grounded),
            "waterlevel": int(self.waterlevel),
            "current_region_id": int(self.current_region_id),
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
    rel_x: float
    rel_y: float
    rel_z: float
    recency: float
    confidence: float
    magnitude: float
    state: float
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
            "rel_x": float(self.rel_x),
            "rel_y": float(self.rel_y),
            "rel_z": float(self.rel_z),
            "recency": float(self.recency),
            "confidence": float(self.confidence),
            "magnitude": float(self.magnitude),
            "state": float(self.state),
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(slots=True)
class TrustedSpatialToken:
    sector_id: int
    nearest_dist: float
    mean_dist: float
    openness: float
    solid_frac: float
    water_frac: float
    slime_frac: float
    lava_frac: float
    traversable: float
    dropoff: float
    clearance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_id": int(self.sector_id),
            "sector": SPATIAL_SECTOR_NAMES.get(int(self.sector_id), "UNKNOWN"),
            "nearest_dist": float(self.nearest_dist),
            "mean_dist": float(self.mean_dist),
            "openness": float(self.openness),
            "solid_frac": float(self.solid_frac),
            "water_frac": float(self.water_frac),
            "slime_frac": float(self.slime_frac),
            "lava_frac": float(self.lava_frac),
            "traversable": float(self.traversable),
            "dropoff": float(self.dropoff),
            "clearance": float(self.clearance),
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

    self_values = _SELF_STRUCT.unpack(read_exact(_SELF_STRUCT.size))
    self_token = TrustedSelfToken(
        origin=[float(self_values[0]), float(self_values[1]), float(self_values[2])],
        velocity=[float(self_values[3]), float(self_values[4]), float(self_values[5])],
        view_angles=[float(self_values[6]), float(self_values[7]), float(self_values[8])],
        health=int(self_values[9]),
        armor=int(self_values[10]),
        armor_type=float(self_values[11]),
        ammo_shells=int(self_values[12]),
        ammo_nails=int(self_values[13]),
        ammo_rockets=int(self_values[14]),
        ammo_cells=int(self_values[15]),
        weapon_id=int(self_values[16]),
        weapons_owned=int(self_values[17]),
        grounded=bool(self_values[18]),
        waterlevel=int(self_values[19]),
        current_region_id=int(self_values[20]),
    )

    raw_objects: List[tuple[int, int, int, int, int, float, float, float, float, float, float, float, int, int]] = []
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
            rel_x,
            rel_y,
            rel_z,
            recency,
            confidence,
            magnitude,
            state,
            event_count_local,
            event_base,
        ) = row
        events = raw_events[event_base: event_base + event_count_local]
        object_tokens.append(
            TrustedObjectToken(
                handle=int(handle),
                subject_id=int(subject_id),
                qualifier_id=int(qualifier_id),
                modality_id=int(modality_id),
                player_id=int(player_id),
                rel_x=float(rel_x),
                rel_y=float(rel_y),
                rel_z=float(rel_z),
                recency=float(recency),
                confidence=float(confidence),
                magnitude=float(magnitude),
                state=float(state),
                events=list(events),
            )
        )

    spatial_tokens: List[TrustedSpatialToken] = []
    for _ in range(spatial_count):
        (
            sector_id,
            _reserved,
            nearest_dist,
            mean_dist,
            openness,
            solid_frac,
            water_frac,
            slime_frac,
            lava_frac,
            traversable,
            dropoff,
            clearance,
        ) = _SPATIAL_STRUCT.unpack(read_exact(_SPATIAL_STRUCT.size))
        spatial_tokens.append(
            TrustedSpatialToken(
                sector_id=int(sector_id),
                nearest_dist=float(nearest_dist),
                mean_dist=float(mean_dist),
                openness=float(openness),
                solid_frac=float(solid_frac),
                water_frac=float(water_frac),
                slime_frac=float(slime_frac),
                lava_frac=float(lava_frac),
                traversable=float(traversable),
                dropoff=float(dropoff),
                clearance=float(clearance),
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
    """Encode a TrustedTokenTick back to the QTOK binary wire format."""
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
            *tick.self_token.origin,
            *tick.self_token.velocity,
            *tick.self_token.view_angles,
            tick.self_token.health,
            tick.self_token.armor,
            tick.self_token.armor_type,
            tick.self_token.ammo_shells,
            tick.self_token.ammo_nails,
            tick.self_token.ammo_rockets,
            tick.self_token.ammo_cells,
            tick.self_token.weapon_id,
            tick.self_token.weapons_owned,
            int(tick.self_token.grounded),
            tick.self_token.waterlevel,
            tick.self_token.current_region_id,
        ),
    ]
    for idx, obj in enumerate(tick.object_tokens):
        parts.append(
            _OBJECT_STRUCT.pack(
                obj.handle,
                obj.subject_id,
                obj.qualifier_id,
                obj.modality_id,
                obj.player_id,
                obj.rel_x,
                obj.rel_y,
                obj.rel_z,
                obj.recency,
                obj.confidence,
                obj.magnitude,
                obj.state,
                len(obj.events),
                event_bases[idx],
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
                sp.solid_frac,
                sp.water_frac,
                sp.slime_frac,
                sp.lava_frac,
                sp.traversable,
                sp.dropoff,
                sp.clearance,
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
