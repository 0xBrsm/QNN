"""Deterministic engine adapter and demo playback harness.

This module intentionally keeps the runtime interface lightweight so it can run
without a full Quake engine checkout during v0 development.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Tuple

from quake_ai.actions import ActionLabels
from quake_ai.data.netquake_demo import parse_netquake_demo
from quake_ai.maps.bsp_parser import region_for_point
from quake_ai.schemas import PacketEventV1, TelemetryTickV1

PAYLOAD_TYPE_MAP = {
    0x01: "move_cmd",
    0x02: "angle_update",
    0x03: "weapon",
    0x04: "event",
    0x05: "ack",
}


@dataclass(slots=True)
class DemoTick:
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
    packet: Dict[str, object]


@dataclass(slots=True)
class DemoEpisode:
    episode_id: str
    map_id: str
    ticks: List[DemoTick]


def decode_packet_hex(packet_hex: str) -> Tuple[str, Dict[str, int]]:
    payload = bytes.fromhex(packet_hex)
    packet_type = PAYLOAD_TYPE_MAP.get(payload[0] if payload else 0x00, "unknown")
    decoded = {
        "size": len(payload),
        "checksum": sum(payload) % 256,
        "first_byte": int(payload[0]) if payload else 0,
    }
    return packet_type, decoded


def _looks_like_json_demo(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(64).lstrip()
    return prefix.startswith(b"{") or prefix.startswith(b"[")


def _load_json_demo(path: Path) -> DemoEpisode:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ticks: List[DemoTick] = []
    for row in payload["ticks"]:
        action = ActionLabels.from_dict(row.get("action_label", {})).to_dict()
        ticks.append(
            DemoTick(
                tick=int(row["tick"]),
                player_pos=[float(v) for v in row["player_pos"]],
                player_vel=[float(v) for v in row["player_vel"]],
                yaw=float(row["yaw"]),
                health=int(row["health"]),
                armor=int(row["armor"]),
                ammo=int(row["ammo"]),
                weapon_id=int(row["weapon_id"]),
                nearby_item_flags=[int(v) for v in row.get("nearby_item_flags", [])],
                goal_progress=float(row.get("goal_progress", 0.0)),
                action_label=action,
                done=bool(row.get("done", False)),
                done_reason=str(row.get("done_reason", "")),
                packet=dict(row.get("packet", {})),
            )
        )

    return DemoEpisode(
        episode_id=str(payload["episode_id"]),
        map_id=str(payload.get("map_id", "E1M1")),
        ticks=ticks,
    )


def _load_binary_demo(path: Path, map_id: str) -> DemoEpisode:
    payload = parse_netquake_demo(path=path, map_id=map_id)
    ticks: List[DemoTick] = []
    for tick_idx, row in enumerate(payload.ticks):
        action = ActionLabels.from_dict(row.action_label).to_dict()
        ticks.append(
            DemoTick(
                tick=tick_idx,
                player_pos=[float(v) for v in row.player_pos],
                player_vel=[float(v) for v in row.player_vel],
                yaw=float(row.yaw),
                health=int(row.health),
                armor=int(row.armor),
                ammo=int(row.ammo),
                weapon_id=int(row.weapon_id),
                nearby_item_flags=[int(v) for v in row.nearby_item_flags],
                goal_progress=float(row.goal_progress),
                action_label=action,
                done=bool(row.done),
                done_reason=str(row.done_reason),
                packet=dict(row.packet),
            )
        )
    return DemoEpisode(
        episode_id=str(payload.episode_id),
        map_id=str(payload.map_id),
        ticks=ticks,
    )


def load_demo(path: str | Path, map_id: str = "E1M1") -> DemoEpisode:
    source = Path(path)
    if _looks_like_json_demo(source):
        return _load_json_demo(source)
    return _load_binary_demo(source, map_id=map_id)


class DemoPlaybackHarness:
    """Replays deterministic demos and yields aligned telemetry + packet events."""

    def __init__(self, map_id: str) -> None:
        self.map_id = map_id

    def replay(self, demo_path: str | Path) -> Iterator[Tuple[TelemetryTickV1, PacketEventV1]]:
        episode = load_demo(demo_path, map_id=self.map_id)
        for tick in episode.ticks:
            region = region_for_point(tuple(tick.player_pos))
            telemetry = TelemetryTickV1(
                episode_id=episode.episode_id,
                tick=tick.tick,
                player_pos=tick.player_pos,
                player_vel=tick.player_vel,
                yaw=tick.yaw,
                health=tick.health,
                armor=tick.armor,
                ammo=tick.ammo,
                weapon_id=tick.weapon_id,
                nearby_item_flags=tick.nearby_item_flags,
                goal_progress=tick.goal_progress,
                action_label=tick.action_label,
                done=tick.done,
                done_reason=tick.done_reason,
                region_id=region,
            )

            packet_info = tick.packet
            packet_hex = str(packet_info.get("payload_hex", ""))
            payload_type, decoded_fields = decode_packet_hex(packet_hex) if packet_hex else ("unknown", {"size": 0})
            packet = PacketEventV1(
                episode_id=episode.episode_id,
                tick_estimate=int(packet_info.get("tick_estimate", tick.tick)),
                direction=str(packet_info.get("direction", "client_to_server")),
                seq=int(packet_info.get("seq", tick.tick)),
                ack=int(packet_info.get("ack", max(0, tick.tick - 1))),
                payload_type=payload_type,
                decoded_fields=decoded_fields,
            )
            yield telemetry, packet


class SyntheticQuakeAdapter:
    """Deterministic fixed-tick adapter used for smoke training and RL rollouts."""

    def __init__(self, seed: int = 0, fixed_tick_hz: int = 20) -> None:
        self.seed = seed
        self.fixed_tick_hz = fixed_tick_hz

    def ticks_per_second(self) -> int:
        return self.fixed_tick_hz

    def replay_demos(self, demo_paths: Iterable[str | Path], map_id: str) -> Iterator[Tuple[TelemetryTickV1, PacketEventV1]]:
        harness = DemoPlaybackHarness(map_id=map_id)
        for path in sorted(str(p) for p in demo_paths):
            yield from harness.replay(path)
