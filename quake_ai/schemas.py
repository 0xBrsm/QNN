"""Versioned data contracts for the Quake AI pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCHEMA_VERSION = "v1"
SCHEMA_VERSION_V2 = "v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_float_list(value: Iterable[Any], expected_len: int, field_name: str) -> List[float]:
    out = [float(x) for x in value]
    _require(len(out) == expected_len, f"{field_name} must have length {expected_len}")
    return out


def _as_int_list(value: Iterable[Any], field_name: str) -> List[int]:
    return [int(x) for x in value]


def _as_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), f"{field_name} must be a mapping")
    return {str(k): v for k, v in value.items()}


def _as_action_label(value: Any, field_name: str) -> Dict[str, int]:
    data = _as_mapping(value, field_name)
    return {str(k): int(v) for k, v in data.items()}


def _validate_action_label(action: Mapping[str, int], field_name: str) -> None:
    for key, value in action.items():
        _require(bool(str(key)), f"{field_name} keys must be non-empty")
        int(value)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float_vec3(value: Any, field_name: str) -> List[float] | None:
    if value is None:
        return None
    return _as_float_list(value, 3, field_name)


@dataclass(slots=True)
class TelemetryTickV1:
    episode_id: str
    tick: int
    player_pos: List[float]
    player_vel: List[float]
    yaw: float
    health: int
    armor: int
    ammo: int
    weapon_id: int
    nearby_item_flags: List[int]
    goal_progress: float
    action_label: Dict[str, int]
    done: bool
    done_reason: str
    region_id: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TelemetryTickV1":
        tick = cls(
            episode_id=str(data["episode_id"]),
            tick=int(data["tick"]),
            player_pos=_as_float_list(data["player_pos"], 3, "player_pos"),
            player_vel=_as_float_list(data["player_vel"], 3, "player_vel"),
            yaw=float(data["yaw"]),
            health=int(data["health"]),
            armor=int(data["armor"]),
            ammo=int(data["ammo"]),
            weapon_id=int(data["weapon_id"]),
            nearby_item_flags=_as_int_list(data.get("nearby_item_flags", []), "nearby_item_flags"),
            goal_progress=float(data["goal_progress"]),
            action_label={str(k): int(v) for k, v in dict(data["action_label"]).items()},
            done=bool(data["done"]),
            done_reason=str(data.get("done_reason", "")),
            region_id=int(data.get("region_id", 0)),
        )
        tick.validate()
        return tick

    def validate(self) -> None:
        _require(self.episode_id, "episode_id must be non-empty")
        _require(self.tick >= 0, "tick must be >= 0")
        _require(len(self.player_pos) == 3, "player_pos must have length 3")
        _require(len(self.player_vel) == 3, "player_vel must have length 3")
        _require(self.health >= 0, "health must be >= 0")
        _require(self.armor >= 0, "armor must be >= 0")
        _require(self.ammo >= 0, "ammo must be >= 0")
        _require(self.weapon_id >= 0, "weapon_id must be >= 0")
        _require(0.0 <= self.goal_progress <= 1.0, "goal_progress must be in [0, 1]")
        _require(self.done_reason == "" or isinstance(self.done_reason, str), "done_reason must be text")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class MapFeaturesV1:
    map_id: str
    region_id: int
    spawn_points: List[List[float]]
    item_nodes: List[Dict[str, Any]]
    connectivity_edges: List[List[int]]
    goal_nodes: List[int]
    distance_to_goal: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MapFeaturesV1":
        record = cls(
            map_id=str(data["map_id"]),
            region_id=int(data["region_id"]),
            spawn_points=[_as_float_list(p, 3, "spawn_point") for p in data.get("spawn_points", [])],
            item_nodes=[dict(node) for node in data.get("item_nodes", [])],
            connectivity_edges=[[int(edge[0]), int(edge[1])] for edge in data.get("connectivity_edges", [])],
            goal_nodes=[int(x) for x in data.get("goal_nodes", [])],
            distance_to_goal=float(data.get("distance_to_goal", 0.0)),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(self.map_id, "map_id must be non-empty")
        _require(self.region_id >= 0, "region_id must be >= 0")
        for edge in self.connectivity_edges:
            _require(len(edge) == 2, "each connectivity edge must have 2 ints")
            _require(edge[0] >= 0 and edge[1] >= 0, "edge ids must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class PacketEventV1:
    episode_id: str
    tick_estimate: int
    direction: str
    seq: int
    ack: int
    payload_type: str
    decoded_fields: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PacketEventV1":
        event = cls(
            episode_id=str(data["episode_id"]),
            tick_estimate=int(data["tick_estimate"]),
            direction=str(data["direction"]),
            seq=int(data["seq"]),
            ack=int(data["ack"]),
            payload_type=str(data["payload_type"]),
            decoded_fields=dict(data.get("decoded_fields", {})),
        )
        event.validate()
        return event

    def validate(self) -> None:
        _require(self.episode_id, "episode_id must be non-empty")
        _require(self.tick_estimate >= 0, "tick_estimate must be >= 0")
        _require(self.direction in {"client_to_server", "server_to_client"}, "invalid packet direction")
        _require(self.seq >= 0 and self.ack >= 0, "seq/ack must be >= 0")
        _require(self.payload_type, "payload_type must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class EpisodeSummaryV1:
    episode_id: str
    steps: int
    goal_reached: bool
    items_collected: int
    time_to_goal: float
    return_value: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeSummaryV1":
        summary = cls(
            episode_id=str(data["episode_id"]),
            steps=int(data["steps"]),
            goal_reached=bool(data["goal_reached"]),
            items_collected=int(data["items_collected"]),
            time_to_goal=float(data["time_to_goal"]),
            return_value=float(data.get("return", data.get("return_value", 0.0))),
        )
        summary.validate()
        return summary

    def validate(self) -> None:
        _require(self.episode_id, "episode_id must be non-empty")
        _require(self.steps >= 0, "steps must be >= 0")
        _require(self.items_collected >= 0, "items_collected must be >= 0")
        _require(self.time_to_goal >= 0.0, "time_to_goal must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["return"] = payload.pop("return_value")
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class RegionNodeV2:
    region_id: int
    center: List[float]
    neighbors: List[int]
    bounds_min: List[float]
    bounds_max: List[float]
    object_ids: List[str] = field(default_factory=list)
    visibility_hints: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionNodeV2":
        record = cls(
            region_id=int(data["region_id"]),
            center=_as_float_list(data["center"], 3, "center"),
            neighbors=_as_int_list(data.get("neighbors", []), "neighbors"),
            bounds_min=_as_float_list(data["bounds_min"], 3, "bounds_min"),
            bounds_max=_as_float_list(data["bounds_max"], 3, "bounds_max"),
            object_ids=[str(value) for value in data.get("object_ids", [])],
            visibility_hints=_as_int_list(data.get("visibility_hints", []), "visibility_hints"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(self.region_id >= 0, "region_id must be >= 0")
        _require(len(self.center) == 3, "center must have length 3")
        _require(len(self.bounds_min) == 3, "bounds_min must have length 3")
        _require(len(self.bounds_max) == 3, "bounds_max must have length 3")
        for axis in range(3):
            _require(self.bounds_min[axis] <= self.bounds_max[axis], "bounds_min must be <= bounds_max")
        for neighbor in self.neighbors:
            _require(neighbor >= 0, "neighbor ids must be >= 0")
        for object_id in self.object_ids:
            _require(bool(object_id), "object_ids must be non-empty")
        for region_id in self.visibility_hints:
            _require(region_id >= 0, "visibility_hints must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload


@dataclass(slots=True)
class StaticObjectV2:
    object_id: str
    category: str
    classname: str
    region_id: int
    origin: List[float]
    angles: List[float]
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaticObjectV2":
        record = cls(
            object_id=str(data["object_id"]),
            category=str(data["category"]),
            classname=str(data.get("classname", "")),
            region_id=int(data["region_id"]),
            origin=_as_float_list(data["origin"], 3, "origin"),
            angles=_as_float_list(data.get("angles", [0.0, 0.0, 0.0]), 3, "angles"),
            properties=_as_mapping(data.get("properties", {}), "properties"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.object_id), "object_id must be non-empty")
        _require(bool(self.category), "category must be non-empty")
        _require(self.region_id >= 0, "region_id must be >= 0")
        _require(len(self.origin) == 3, "origin must have length 3")
        _require(len(self.angles) == 3, "angles must have length 3")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload


@dataclass(slots=True)
class MapStateV2:
    map_id: str
    regions: List[RegionNodeV2]
    static_objects: List[StaticObjectV2]
    spawn_region_ids: List[int]
    goal_region_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MapStateV2":
        record = cls(
            map_id=str(data["map_id"]),
            regions=[RegionNodeV2.from_dict(row) for row in data.get("regions", [])],
            static_objects=[StaticObjectV2.from_dict(row) for row in data.get("static_objects", [])],
            spawn_region_ids=_as_int_list(data.get("spawn_region_ids", []), "spawn_region_ids"),
            goal_region_ids=_as_int_list(data.get("goal_region_ids", []), "goal_region_ids"),
            metadata=_as_mapping(data.get("metadata", {}), "metadata"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.map_id), "map_id must be non-empty")
        region_ids = [region.region_id for region in self.regions]
        _require(len(region_ids) == len(set(region_ids)), "region ids must be unique")
        known_regions = set(region_ids)
        object_ids = [obj.object_id for obj in self.static_objects]
        _require(len(object_ids) == len(set(object_ids)), "static object ids must be unique")

        for region in self.regions:
            region.validate()
            for neighbor in region.neighbors:
                _require(neighbor in known_regions, "region neighbors must reference known regions")
            for visibility_hint in region.visibility_hints:
                _require(visibility_hint in known_regions, "visibility_hints must reference known regions")

        known_object_ids = set(object_ids)
        for region in self.regions:
            for object_id in region.object_ids:
                _require(object_id in known_object_ids, "region object_ids must reference known static objects")

        for obj in self.static_objects:
            obj.validate()
            _require(obj.region_id in known_regions, "static objects must reference known regions")

        for region_id in self.spawn_region_ids:
            _require(region_id in known_regions, "spawn_region_ids must reference known regions")
        for region_id in self.goal_region_ids:
            _require(region_id in known_regions, "goal_region_ids must reference known regions")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload


@dataclass(slots=True)
class PlayerStateV2:
    origin: List[float]
    velocity: List[float]
    view_angles: List[float]
    health: int
    armor: int
    ammo: int
    weapon_id: int
    grounded: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlayerStateV2":
        record = cls(
            origin=_as_float_list(data["origin"], 3, "origin"),
            velocity=_as_float_list(data["velocity"], 3, "velocity"),
            view_angles=_as_float_list(data.get("view_angles", [0.0, 0.0, 0.0]), 3, "view_angles"),
            health=int(data["health"]),
            armor=int(data["armor"]),
            ammo=int(data["ammo"]),
            weapon_id=int(data["weapon_id"]),
            grounded=None if data.get("grounded") is None else bool(data.get("grounded")),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(len(self.origin) == 3, "origin must have length 3")
        _require(len(self.velocity) == 3, "velocity must have length 3")
        _require(len(self.view_angles) == 3, "view_angles must have length 3")
        # Quake can report negative player health after lethal damage or gib states.
        _require(self.armor >= 0, "armor must be >= 0")
        _require(self.ammo >= 0, "ammo must be >= 0")
        _require(self.weapon_id >= 0, "weapon_id must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload


@dataclass(slots=True)
class EntityStateV2:
    entity_id: str
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityStateV2":
        record = cls(
            entity_id=str(data["entity_id"]),
            entity_num=int(data.get("entity_num", 0)),
            classname=str(data.get("classname", "")),
            region_id=_optional_int(data.get("region_id"), "region_id"),
            origin=_as_float_list(data["origin"], 3, "origin"),
            velocity=_as_float_list(data.get("velocity", [0.0, 0.0, 0.0]), 3, "velocity"),
            angles=_as_float_list(data.get("angles", [0.0, 0.0, 0.0]), 3, "angles"),
            model_id=int(data.get("model_id", 0)),
            frame=int(data.get("frame", 0)),
            visible=bool(data.get("visible", True)),
            properties=_as_mapping(data.get("properties", {}), "properties"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.entity_id), "entity_id must be non-empty")
        _require(self.entity_num >= 0, "entity_num must be >= 0")
        if self.region_id is not None:
            _require(self.region_id >= 0, "region_id must be >= 0")
        _require(len(self.origin) == 3, "origin must have length 3")
        _require(len(self.velocity) == 3, "velocity must have length 3")
        _require(len(self.angles) == 3, "angles must have length 3")
        _require(self.model_id >= 0, "model_id must be >= 0")
        _require(self.frame >= 0, "frame must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload


@dataclass(slots=True)
class WorldEventV2:
    event_type: str
    region_id: int | None
    source_id: str = ""
    target_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldEventV2":
        record = cls(
            event_type=str(data["event_type"]),
            region_id=_optional_int(data.get("region_id"), "region_id"),
            source_id=str(data.get("source_id", "")),
            target_id=str(data.get("target_id", "")),
            payload=_as_mapping(data.get("payload", {}), "payload"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.event_type), "event_type must be non-empty")
        if self.region_id is not None:
            _require(self.region_id >= 0, "region_id must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload


@dataclass(slots=True)
class WorldTickV2:
    episode_id: str
    map_id: str
    tick: int
    player: PlayerStateV2
    current_region_id: int | None
    visible_entities: List[EntityStateV2]
    events: List[WorldEventV2]
    action_label: Dict[str, int] = field(default_factory=dict)
    action_history: List[Dict[str, int]] = field(default_factory=list)
    done: bool = False
    done_reason: str = ""
    reset: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldTickV2":
        record = cls(
            episode_id=str(data["episode_id"]),
            map_id=str(data["map_id"]),
            tick=int(data["tick"]),
            player=PlayerStateV2.from_dict(_as_mapping(data["player"], "player")),
            current_region_id=_optional_int(data.get("current_region_id"), "current_region_id"),
            visible_entities=[EntityStateV2.from_dict(row) for row in data.get("visible_entities", [])],
            events=[WorldEventV2.from_dict(row) for row in data.get("events", [])],
            action_label=_as_action_label(data.get("action_label", {}), "action_label"),
            action_history=[_as_action_label(row, "action_history") for row in data.get("action_history", [])],
            done=bool(data.get("done", False)),
            done_reason=str(data.get("done_reason", "")),
            reset=bool(data.get("reset", False)),
            debug=_as_mapping(data.get("debug", {}), "debug"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        _require(bool(self.episode_id), "episode_id must be non-empty")
        _require(bool(self.map_id), "map_id must be non-empty")
        _require(self.tick >= 0, "tick must be >= 0")
        self.player.validate()
        if self.current_region_id is not None:
            _require(self.current_region_id >= 0, "current_region_id must be >= 0")
        for entity in self.visible_entities:
            entity.validate()
        for event in self.events:
            event.validate()
        _validate_action_label(self.action_label, "action_label")
        for action in self.action_history:
            _validate_action_label(action, "action_history")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION_V2
        return payload
