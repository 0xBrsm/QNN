"""Versioned data contracts for the Quake AI pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping

SCHEMA_VERSION = "v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_float_list(value: Iterable[Any], expected_len: int, field_name: str) -> List[float]:
    out = [float(x) for x in value]
    _require(len(out) == expected_len, f"{field_name} must have length {expected_len}")
    return out


def _as_int_list(value: Iterable[Any], field_name: str) -> List[int]:
    return [int(x) for x in value]


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
