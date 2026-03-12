"""Trusted live-worker protocol helpers.

This module keeps the hot-path parsing separate from the validated schema layer:

- `MapState` still uses the fully validated schema constructors.
- live `world_tick` payloads use lightweight trusted objects because the worker
  is a pinned local process we build and control.

The binary step-response format is negotiated via the worker `hello` response.
Resets remain JSON for debuggability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import Any, Dict, List, Mapping, Sequence


STEP_BINARY_MAGIC = b"QWLD"
STEP_BINARY_VERSION = 1

_HEADER_STRUCT = struct.Struct(
    "<4sHHf"  # magic, version, flags, reward
    "iii"  # tick, steps, current_region_id
    "13i"  # frags..grounded_raw
    "f"  # armor_type
    "9f"  # origin, velocity, view_angles
    "8i"  # damage metrics + done_reason_code
    "27i"  # weapon totals
    "7h"  # current action label
    "5H"  # action_history_count, visible_count, event_count, sound_count, string_count
)
STEP_BINARY_HEADER_SIZE = _HEADER_STRUCT.size

_ACTION_STRUCT = struct.Struct("<7h")
_ENTITY_STRUCT = struct.Struct("<iiiHHH3f")
_EVENT_STRUCT = struct.Struct("<HHiiiii")
_SOUND_STRUCT = struct.Struct("<HHi3fff")

_FLAG_DONE = 1 << 0
_FLAG_GOAL_REACHED = 1 << 1

_ENTITY_FLAG_STATIC_PROXY = 1 << 0

_EVENT_FLAG_HAS_DELTA = 1 << 0
_EVENT_FLAG_HAS_WEAPON_ID = 1 << 1

_DONE_REASON_BY_CODE = {
    0: "",
    1: "goal_reached",
    2: "player_died",
    3: "timeout",
}

_EVENT_TYPE_BY_CODE = {
    1: "damage_taken",
    2: "pickup_health",
    3: "pickup_armor",
    4: "pickup_ammo",
    5: "pickup_weapon",
    6: "pickup_item",
    7: "frag_gained",
    8: "frag_lost",
    9: "monster_kill",
    10: "damage_dealt",
    11: "hit_confirmed",
    12: "shots_fired",
    13: "goal_reached",
    14: "player_died",
}

_ACTION_KEYS = ("move", "strafe", "look_yaw", "look_pitch", "fire", "jump", "weapon")


def _entity_id_from_key(entity_key: int, entity_num: int, explicit: str = "") -> str:
    if explicit:
        return explicit
    if entity_num > 0:
        return f"entity_{entity_num:04d}"
    if entity_key < 0:
        return f"static_{(-entity_key) - 1:04d}"
    return ""


def _entity_num_from_id(entity_id: str) -> int:
    if entity_id.startswith("entity_"):
        try:
            return int(entity_id.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _entity_id_from_entity_num(entity_num: int) -> str:
    return f"entity_{entity_num:04d}" if entity_num > 0 else ""


def _action_dict(values: Sequence[int]) -> Dict[str, int]:
    return {key: int(values[idx]) for idx, key in enumerate(_ACTION_KEYS)}


@dataclass(slots=True)
class TrustedPlayerState:
    origin: List[float]
    velocity: List[float]
    view_angles: List[float]
    health: int
    armor: int
    ammo: int
    weapon_id: int
    grounded: bool | None = None
    armor_type: float = 0.0
    ammo_shells: int = 0
    ammo_nails: int = 0
    ammo_rockets: int = 0
    ammo_cells: int = 0
    weapons_owned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": list(self.origin),
            "velocity": list(self.velocity),
            "view_angles": list(self.view_angles),
            "health": int(self.health),
            "armor": int(self.armor),
            "ammo": int(self.ammo),
            "weapon_id": int(self.weapon_id),
            "grounded": self.grounded,
            "armor_type": float(self.armor_type),
            "ammo_shells": int(self.ammo_shells),
            "ammo_nails": int(self.ammo_nails),
            "ammo_rockets": int(self.ammo_rockets),
            "ammo_cells": int(self.ammo_cells),
            "weapons_owned": int(self.weapons_owned),
        }


@dataclass(slots=True)
class TrustedEntityState:
    entity_key: int
    entity_num: int
    classname: str
    region_id: int | None
    origin: List[float]
    velocity: List[float]
    angles: List[float]
    model_id: int = 0
    frame: int = 0
    visible: bool = True
    properties: Dict[str, Any] = field(default_factory=dict)
    _entity_id_text: str = ""

    @property
    def entity_id(self) -> str:
        return _entity_id_from_key(self.entity_key, self.entity_num, self._entity_id_text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_num": int(self.entity_num),
            "classname": str(self.classname),
            "region_id": self.region_id,
            "origin": list(self.origin),
            "velocity": list(self.velocity),
            "angles": list(self.angles),
            "model_id": int(self.model_id),
            "frame": int(self.frame),
            "visible": bool(self.visible),
            "properties": dict(self.properties),
        }


@dataclass(slots=True)
class TrustedSoundEvent:
    origin: List[float]
    volume: float
    attenuation: float
    entity_num: int
    category: int
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": list(self.origin),
            "volume": float(self.volume),
            "attenuation": float(self.attenuation),
            "entity_num": int(self.entity_num),
            "category": int(self.category),
            "name": str(self.name),
        }


@dataclass(slots=True)
class TrustedWorldEvent:
    event_type: str
    region_id: int | None
    payload: Dict[str, Any] = field(default_factory=dict)
    source_entity_num: int = 0
    target_entity_num: int = 0

    @property
    def source_id(self) -> str:
        return _entity_id_from_entity_num(self.source_entity_num)

    @property
    def target_id(self) -> str:
        return _entity_id_from_entity_num(self.target_entity_num)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": str(self.event_type),
            "region_id": self.region_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class TrustedWorldTick:
    episode_id: str
    map_id: str
    tick: int
    player: TrustedPlayerState
    current_region_id: int | None
    visible_entities: List[TrustedEntityState]
    events: List[TrustedWorldEvent]
    sound_events: List[TrustedSoundEvent] = field(default_factory=list)
    action_label: Dict[str, int] = field(default_factory=dict)
    action_history: List[Dict[str, int]] = field(default_factory=list)
    done: bool = False
    done_reason: str = ""
    reset: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": str(self.episode_id),
            "map_id": str(self.map_id),
            "tick": int(self.tick),
            "player": self.player.to_dict(),
            "current_region_id": self.current_region_id,
            "visible_entities": [entity.to_dict() for entity in self.visible_entities],
            "events": [event.to_dict() for event in self.events],
            "sounds": [sound.to_dict() for sound in self.sound_events],
            "action_label": dict(self.action_label),
            "action_history": [dict(action) for action in self.action_history],
            "done": bool(self.done),
            "done_reason": str(self.done_reason),
            "reset": bool(self.reset),
            "debug": dict(self.debug),
        }


def trusted_world_tick_from_mapping(data: Mapping[str, Any]) -> TrustedWorldTick:
    player_raw = data["player"]
    player = TrustedPlayerState(
        origin=[float(value) for value in player_raw.get("origin", [0.0, 0.0, 0.0])],
        velocity=[float(value) for value in player_raw.get("velocity", [0.0, 0.0, 0.0])],
        view_angles=[float(value) for value in player_raw.get("view_angles", [0.0, 0.0, 0.0])],
        health=int(player_raw.get("health", 0)),
        armor=int(player_raw.get("armor", 0)),
        ammo=int(player_raw.get("ammo", 0)),
        weapon_id=int(player_raw.get("weapon_id", 0)),
        grounded=None if player_raw.get("grounded") is None else bool(player_raw.get("grounded")),
        armor_type=float(player_raw.get("armor_type", 0.0)),
        ammo_shells=int(player_raw.get("ammo_shells", 0)),
        ammo_nails=int(player_raw.get("ammo_nails", 0)),
        ammo_rockets=int(player_raw.get("ammo_rockets", 0)),
        ammo_cells=int(player_raw.get("ammo_cells", 0)),
        weapons_owned=int(player_raw.get("weapons_owned", 0)),
    )

    visible_entities = [
        TrustedEntityState(
            entity_key=int(row.get("entity_key", _entity_num_from_id(str(row.get("entity_id", ""))))),
            entity_num=int(row.get("entity_num", 0)),
            classname=str(row.get("classname", "")),
            region_id=None if row.get("region_id") is None else int(row.get("region_id")),
            origin=[float(value) for value in row.get("origin", [0.0, 0.0, 0.0])],
            velocity=[float(value) for value in row.get("velocity", [0.0, 0.0, 0.0])],
            angles=[float(value) for value in row.get("angles", [0.0, 0.0, 0.0])],
            model_id=int(row.get("model_id", 0)),
            frame=int(row.get("frame", 0)),
            visible=bool(row.get("visible", True)),
            properties=dict(row.get("properties", {})),
            _entity_id_text=str(row.get("entity_id", "")),
        )
        for row in data.get("visible_entities", [])
    ]

    events = []
    for row in data.get("events", []):
        source_id = str(row.get("source_id", ""))
        target_id = str(row.get("target_id", ""))
        events.append(
            TrustedWorldEvent(
                event_type=str(row.get("event_type", "")),
                region_id=None if row.get("region_id") is None else int(row.get("region_id")),
                payload=dict(row.get("payload", {})),
                source_entity_num=_entity_num_from_id(source_id),
                target_entity_num=_entity_num_from_id(target_id),
            )
        )

    sound_rows = data.get("sounds", data.get("sound_events", []))
    sound_events = [
        TrustedSoundEvent(
            origin=[float(value) for value in row.get("origin", [0.0, 0.0, 0.0])],
            volume=float(row.get("volume", 1.0)),
            attenuation=float(row.get("attenuation", 1.0)),
            entity_num=int(row.get("entity_num", 0)),
            category=int(row.get("category", 0)),
            name=str(row.get("name", "")),
        )
        for row in sound_rows
    ]

    return TrustedWorldTick(
        episode_id=str(data.get("episode_id", "")),
        map_id=str(data.get("map_id", "")),
        tick=int(data.get("tick", 0)),
        player=player,
        current_region_id=None if data.get("current_region_id") is None else int(data.get("current_region_id")),
        visible_entities=visible_entities,
        events=events,
        sound_events=sound_events,
        action_label={str(key): int(value) for key, value in dict(data.get("action_label", {})).items()},
        action_history=[
            {str(key): int(value) for key, value in dict(row).items()}
            for row in data.get("action_history", [])
        ],
        done=bool(data.get("done", False)),
        done_reason=str(data.get("done_reason", "")),
        reset=bool(data.get("reset", False)),
        debug=dict(data.get("debug", {})),
    )


def decode_binary_step_tick(
    header: bytes,
    read_exact,
    *,
    episode_id: str,
    map_id: str,
) -> tuple[TrustedWorldTick, float, bool]:
    unpacked = _HEADER_STRUCT.unpack(header)
    magic = unpacked[0]
    version = unpacked[1]
    if magic != STEP_BINARY_MAGIC:
        raise ValueError(f"Unexpected step binary magic {magic!r}")
    if version != STEP_BINARY_VERSION:
        raise ValueError(f"Unsupported step binary version {version}")

    flags = unpacked[2]
    reward = float(unpacked[3])
    tick = int(unpacked[4])
    steps = int(unpacked[5])
    current_region_id = int(unpacked[6])
    frags = int(unpacked[7])
    monster_kills = int(unpacked[8])
    monster_total = int(unpacked[9])
    health = int(unpacked[10])
    armor = int(unpacked[11])
    ammo = int(unpacked[12])
    weapon_id = int(unpacked[13])
    weapons_owned = int(unpacked[14])
    ammo_shells = int(unpacked[15])
    ammo_nails = int(unpacked[16])
    ammo_rockets = int(unpacked[17])
    ammo_cells = int(unpacked[18])
    grounded_raw = int(unpacked[19])
    armor_type = float(unpacked[20])
    origin = [float(unpacked[21]), float(unpacked[22]), float(unpacked[23])]
    velocity = [float(unpacked[24]), float(unpacked[25]), float(unpacked[26])]
    view_angles = [float(unpacked[27]), float(unpacked[28]), float(unpacked[29])]
    damage_dealt = int(unpacked[30])
    damage_dealt_total = int(unpacked[31])
    damage_weapon_id = int(unpacked[32])
    hit_count = int(unpacked[33])
    hit_count_total = int(unpacked[34])
    shots_fired = int(unpacked[35])
    shots_fired_total = int(unpacked[36])
    done_reason_code = int(unpacked[37])
    weapon_damage_totals = [int(value) for value in unpacked[38:47]]
    weapon_hit_totals = [int(value) for value in unpacked[47:56]]
    weapon_shot_totals = [int(value) for value in unpacked[56:65]]
    current_action_values = unpacked[65:72]
    action_history_count = int(unpacked[72])
    visible_count = int(unpacked[73])
    event_count = int(unpacked[74])
    sound_count = int(unpacked[75])
    string_count = int(unpacked[76])

    action_label = _action_dict(current_action_values)

    action_history = []
    for _ in range(action_history_count):
        values = _ACTION_STRUCT.unpack(read_exact(_ACTION_STRUCT.size))
        action_history.append(_action_dict(values))

    entity_rows = [_ENTITY_STRUCT.unpack(read_exact(_ENTITY_STRUCT.size)) for _ in range(visible_count)]
    event_rows = [_EVENT_STRUCT.unpack(read_exact(_EVENT_STRUCT.size)) for _ in range(event_count)]
    sound_rows = [_SOUND_STRUCT.unpack(read_exact(_SOUND_STRUCT.size)) for _ in range(sound_count)]

    strings = [""] * (string_count + 1)
    for index in range(1, string_count + 1):
        length = struct.unpack("<H", read_exact(2))[0]
        strings[index] = read_exact(length).decode("utf-8")

    visible_entities = []
    for row in entity_rows:
        entity_key, entity_num, region_id_raw, class_idx, entity_id_idx, flags_value, ox, oy, oz = row
        properties: Dict[str, Any] = {}
        if flags_value & _ENTITY_FLAG_STATIC_PROXY:
            properties["source"] = "static_proxy"
        visible_entities.append(
            TrustedEntityState(
                entity_key=int(entity_key),
                entity_num=int(entity_num),
                classname=str(strings[int(class_idx)]),
                region_id=None if int(region_id_raw) < 0 else int(region_id_raw),
                origin=[float(ox), float(oy), float(oz)],
                velocity=[0.0, 0.0, 0.0],
                angles=[0.0, 0.0, 0.0],
                properties=properties,
                _entity_id_text=str(strings[int(entity_id_idx)]),
            )
        )

    events = []
    for row in event_rows:
        event_code, flags_value, region_id_raw, delta, weapon_id_raw, source_entity_num, target_entity_num = row
        payload: Dict[str, Any] = {}
        if flags_value & _EVENT_FLAG_HAS_DELTA:
            payload["delta"] = int(delta)
        if flags_value & _EVENT_FLAG_HAS_WEAPON_ID:
            payload["weapon_id"] = int(weapon_id_raw)
        events.append(
            TrustedWorldEvent(
                event_type=_EVENT_TYPE_BY_CODE.get(int(event_code), ""),
                region_id=None if int(region_id_raw) < 0 else int(region_id_raw),
                payload=payload,
                source_entity_num=int(source_entity_num),
                target_entity_num=int(target_entity_num),
            )
        )

    sound_events = []
    for row in sound_rows:
        name_idx, category, entity_num, ox, oy, oz, volume, attenuation = row
        sound_events.append(
            TrustedSoundEvent(
                origin=[float(ox), float(oy), float(oz)],
                volume=float(volume),
                attenuation=float(attenuation),
                entity_num=int(entity_num),
                category=int(category),
                name=str(strings[int(name_idx)]),
            )
        )

    tick_obj = TrustedWorldTick(
        episode_id=episode_id,
        map_id=map_id,
        tick=tick,
        player=TrustedPlayerState(
            origin=origin,
            velocity=velocity,
            view_angles=view_angles,
            health=health,
            armor=armor,
            ammo=ammo,
            weapon_id=weapon_id,
            grounded=None if grounded_raw < 0 else bool(grounded_raw),
            armor_type=armor_type,
            ammo_shells=ammo_shells,
            ammo_nails=ammo_nails,
            ammo_rockets=ammo_rockets,
            ammo_cells=ammo_cells,
            weapons_owned=weapons_owned,
        ),
        current_region_id=None if current_region_id < 0 else current_region_id,
        visible_entities=visible_entities,
        events=events,
        sound_events=sound_events,
        action_label=action_label,
        action_history=action_history,
        done=bool(flags & _FLAG_DONE),
        done_reason=_DONE_REASON_BY_CODE.get(done_reason_code, ""),
        reset=False,
        debug={
            "frags": frags,
            "monster_kills": monster_kills,
            "monster_total": monster_total,
            "damage_dealt": damage_dealt,
            "damage_dealt_total": damage_dealt_total,
            "damage_weapon_id": damage_weapon_id,
            "hit_count": hit_count,
            "hit_count_total": hit_count_total,
            "shots_fired": shots_fired,
            "shots_fired_total": shots_fired_total,
            "weapon_damage_dealt_total": weapon_damage_totals,
            "weapon_hits_landed_total": weapon_hit_totals,
            "weapon_shots_fired_total": weapon_shot_totals,
            "steps": steps,
        },
    )
    return tick_obj, reward, bool(flags & _FLAG_DONE)
