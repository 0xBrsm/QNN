"""Engine-era world-stream helpers for demos and worker payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from quake_ai.actions import ActionLabels
from quake_ai.maps.world_model import nearest_region_id
from quake_ai.schemas import (
    EntityStateV2,
    MapStateV2,
    PlayerStateV2,
    TelemetryTickV1,
    WorldEventV2,
    WorldTickV2,
)
from quake_ai.utils.io import read_ndjson, write_ndjson

_PICKUP_EVENTS = ("pickup_health", "pickup_armor", "pickup_ammo", "pickup_weapon")


def _world_goal_progress(map_state: MapStateV2, region_id: int | None) -> float:
    if region_id is None:
        return 0.0
    raw_distances = map_state.metadata.get("distance_to_goal", {})
    if not isinstance(raw_distances, Mapping):
        return 0.0
    distance = float(raw_distances.get(str(region_id), raw_distances.get(region_id, 0.0)))
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))


def _event_payload_from_row(event: Mapping[str, object]) -> Dict[str, object]:
    return {
        str(key): value
        for key, value in event.items()
        if key not in {"type", "event_type", "region_id", "source_id", "target_id"}
    }


def _event_region_id(event: Mapping[str, object], fallback_region_id: int | None) -> int | None:
    region = event.get("region_id")
    if region is None:
        return fallback_region_id
    return int(region)


def _tick_events(tick: Any, current_region_id: int | None) -> List[WorldEventV2]:
    raw_events = [dict(event) for event in getattr(tick, "events", [])]
    if not raw_events:
        flags = list(getattr(tick, "nearby_item_flags", []))
        for idx, flag in enumerate(flags[:4]):
            if int(flag) != 1:
                continue
            raw_events.append({"type": _PICKUP_EVENTS[idx]})

    if bool(getattr(tick, "done", False)):
        done_reason = str(getattr(tick, "done_reason", ""))
        if done_reason:
            raw_events.append({"type": done_reason})

    return [
        WorldEventV2(
            event_type=str(event.get("event_type", event.get("type", "unknown"))),
            region_id=_event_region_id(event, current_region_id),
            source_id=str(event.get("source_id", "")),
            target_id=str(event.get("target_id", "")),
            payload=_event_payload_from_row(event),
        )
        for event in raw_events
    ]


def _dynamic_entities(tick: Any, map_state: MapStateV2, current_region_id: int | None, events: Sequence[WorldEventV2]) -> List[EntityStateV2]:
    entities: List[EntityStateV2] = []
    for row in getattr(tick, "visible_entities", []):
        origin = [float(value) for value in row.get("origin", [0.0, 0.0, 0.0])]
        region_id = nearest_region_id(map_state, origin)
        properties = {
            str(key): value
            for key, value in row.items()
            if key not in {"entity_num", "origin", "angles", "model_id", "frame", "classname", "velocity"}
        }
        entity_num = int(row.get("entity_num", 0))
        entities.append(
            EntityStateV2(
                entity_id=f"entity_{entity_num:04d}",
                entity_num=entity_num,
                classname=str(row.get("classname", "")),
                region_id=region_id,
                origin=origin,
                velocity=[float(value) for value in row.get("velocity", [0.0, 0.0, 0.0])],
                angles=[float(value) for value in row.get("angles", [0.0, 0.0, 0.0])],
                model_id=int(row.get("model_id", 0)),
                frame=int(row.get("frame", 0)),
                visible=bool(row.get("visible", True)),
                properties=properties,
            )
        )
    if entities:
        return entities

    relevant_regions = {current_region_id} if current_region_id is not None else set()
    for event in events:
        if event.region_id is not None:
            relevant_regions.add(event.region_id)

    for obj in map_state.static_objects:
        if obj.region_id not in relevant_regions:
            continue
        if obj.category not in {"item", "goal", "trigger"}:
            continue
        entities.append(
            EntityStateV2(
                entity_id=obj.object_id,
                entity_num=0,
                classname=obj.classname,
                region_id=obj.region_id,
                origin=list(obj.origin),
                velocity=[0.0, 0.0, 0.0],
                angles=list(obj.angles),
                model_id=0,
                frame=0,
                visible=True,
                properties={"source": "static_proxy"},
            )
        )
    return entities


def iter_world_ticks_from_demo_episode(episode: Any, map_state: MapStateV2, action_history_len: int = 2) -> Iterator[WorldTickV2]:
    history: List[Dict[str, int]] = []
    tick_index = 0

    for tick in getattr(episode, "ticks", []):
        player_origin = [float(value) for value in getattr(tick, "player_pos")]
        current_region_id = nearest_region_id(map_state, player_origin)
        action_label = ActionLabels.from_dict(dict(getattr(tick, "action_label", {}))).to_dict()
        events = _tick_events(tick, current_region_id)
        raw_view_angles = getattr(tick, "view_angles", [0.0, float(getattr(tick, "yaw")), 0.0])
        world_tick = WorldTickV2(
            episode_id=str(getattr(episode, "episode_id")),
            map_id=str(getattr(episode, "map_id")),
            tick=int(getattr(tick, "tick", tick_index)),
            player=PlayerStateV2(
                origin=player_origin,
                velocity=[float(value) for value in getattr(tick, "player_vel")],
                view_angles=[float(value) for value in raw_view_angles[:3]],
                health=int(getattr(tick, "health")),
                armor=int(getattr(tick, "armor")),
                ammo=int(getattr(tick, "ammo")),
                weapon_id=int(getattr(tick, "weapon_id")),
                grounded=None,
            ),
            current_region_id=current_region_id,
            visible_entities=_dynamic_entities(tick, map_state, current_region_id, events),
            events=events,
            action_label=action_label,
            action_history=[dict(action) for action in history[-action_history_len:]],
            done=bool(getattr(tick, "done", False)),
            done_reason=str(getattr(tick, "done_reason", "")),
            reset=tick_index == 0,
            debug={"packet": dict(getattr(tick, "packet", {}))},
        )
        world_tick.validate()
        history.append(action_label)
        tick_index += 1
        yield world_tick


def world_ticks_from_demo_episode(episode: Any, map_state: MapStateV2, action_history_len: int = 2) -> List[WorldTickV2]:
    return list(iter_world_ticks_from_demo_episode(episode, map_state, action_history_len=action_history_len))


def world_tick_to_telemetry_v1(world_tick: WorldTickV2, map_state: MapStateV2) -> TelemetryTickV1:
    pickup_flags = [0, 0, 0, 0]
    for event in world_tick.events:
        if event.event_type in _PICKUP_EVENTS:
            pickup_flags[_PICKUP_EVENTS.index(event.event_type)] = 1

    yaw = float(world_tick.player.view_angles[1]) if len(world_tick.player.view_angles) > 1 else 0.0
    region_id = 0 if world_tick.current_region_id is None else int(world_tick.current_region_id)
    return TelemetryTickV1(
        episode_id=world_tick.episode_id,
        tick=world_tick.tick,
        player_pos=list(world_tick.player.origin),
        player_vel=list(world_tick.player.velocity),
        yaw=yaw,
        health=int(world_tick.player.health),
        armor=int(world_tick.player.armor),
        ammo=int(world_tick.player.ammo),
        weapon_id=int(world_tick.player.weapon_id),
        nearby_item_flags=pickup_flags,
        goal_progress=_world_goal_progress(map_state, world_tick.current_region_id),
        action_label=dict(world_tick.action_label),
        done=bool(world_tick.done),
        done_reason=str(world_tick.done_reason),
        region_id=region_id,
    )


def load_world_ticks(path: str | Path) -> List[WorldTickV2]:
    return [WorldTickV2.from_dict(row) for row in read_ndjson(path)]


def write_world_ticks(path: str | Path, rows: Iterable[WorldTickV2]) -> None:
    write_ndjson(path, (row.to_dict() for row in rows))
