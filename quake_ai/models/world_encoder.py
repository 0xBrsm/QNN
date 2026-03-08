"""Fixed-size featurization for engine-era world-state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from quake_ai.actions import normalized_action_features
from quake_ai.schemas import MapStateV2, WorldTickV2

EVENT_BUCKETS = {
    "pickup": 0,
    "damage": 1,
    "trigger": 2,
    "temp": 3,
    "kill": 4,
    "misc": 5,
}


def _event_bucket(event_type: str) -> int:
    lowered = event_type.lower()
    if lowered.startswith("pickup"):
        return EVENT_BUCKETS["pickup"]
    if "damage" in lowered:
        return EVENT_BUCKETS["damage"]
    if any(token in lowered for token in ("trigger", "goal", "intermission", "secret")):
        return EVENT_BUCKETS["trigger"]
    if "temp" in lowered or "particle" in lowered or "sound" in lowered:
        return EVENT_BUCKETS["temp"]
    if "kill" in lowered or "monster" in lowered:
        return EVENT_BUCKETS["kill"]
    return EVENT_BUCKETS["misc"]


@dataclass(slots=True)
class WorldObservationEncoder:
    max_entities: int = 4
    max_events: int = 4
    action_history_len: int = 2

    @property
    def obs_dim(self) -> int:
        player_dim = 14
        region_dim = 7
        entity_dim = self.max_entities * 6
        event_dim = self.max_events * 3
        history_dim = self.action_history_len * 7
        return player_dim + region_dim + entity_dim + event_dim + history_dim

    def encode(self, map_state: MapStateV2, world_tick: WorldTickV2) -> np.ndarray:
        regions_by_id = {region.region_id: region for region in map_state.regions}
        objects_per_region: Dict[int, int] = {}
        for obj in map_state.static_objects:
            objects_per_region[obj.region_id] = objects_per_region.get(obj.region_id, 0) + 1

        current_region_id = 0 if world_tick.current_region_id is None else int(world_tick.current_region_id)
        region = regions_by_id.get(current_region_id)
        raw_distances = map_state.metadata.get("distance_to_goal", {})
        region_distance = float(raw_distances.get(str(current_region_id), raw_distances.get(current_region_id, 0.0))) if isinstance(raw_distances, dict) else 0.0
        max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
        if max_distance <= 0.0:
            max_distance = 1.0

        pitch = float(world_tick.player.view_angles[0]) if len(world_tick.player.view_angles) > 0 else 0.0
        yaw = float(world_tick.player.view_angles[1]) if len(world_tick.player.view_angles) > 1 else 0.0
        player_features = [
            float(world_tick.player.origin[0]) / 2048.0,
            float(world_tick.player.origin[1]) / 2048.0,
            float(world_tick.player.origin[2]) / 512.0,
            float(world_tick.player.velocity[0]) / 320.0,
            float(world_tick.player.velocity[1]) / 320.0,
            float(world_tick.player.velocity[2]) / 320.0,
            float(np.cos(np.deg2rad(yaw))),
            float(np.sin(np.deg2rad(yaw))),
            float(np.clip(pitch / 90.0, -1.0, 1.0)),
            float(world_tick.player.health) / 100.0,
            float(world_tick.player.armor) / 100.0,
            float(world_tick.player.ammo) / 100.0,
            float(world_tick.player.weapon_id) / 8.0,
            0.0 if world_tick.player.grounded is None else (1.0 if world_tick.player.grounded else -1.0),
        ]

        region_center = region.center if region is not None else [0.0, 0.0, 0.0]
        region_features = [
            float(region_center[0]) / 2048.0,
            float(region_center[1]) / 2048.0,
            float(region_center[2]) / 512.0,
            region_distance / max_distance,
            1.0 if current_region_id in map_state.goal_region_ids else 0.0,
            1.0 if current_region_id in map_state.spawn_region_ids else 0.0,
            float(objects_per_region.get(current_region_id, 0)) / 8.0,
        ]

        entity_features: List[float] = []
        for entity in sorted(world_tick.visible_entities, key=lambda row: (row.entity_num, row.entity_id))[: self.max_entities]:
            entity_features.extend(
                [
                    float(entity.origin[0] - world_tick.player.origin[0]) / 1024.0,
                    float(entity.origin[1] - world_tick.player.origin[1]) / 1024.0,
                    float(entity.origin[2] - world_tick.player.origin[2]) / 256.0,
                    1.0 if entity.region_id == world_tick.current_region_id else 0.0,
                    float(entity.model_id) / 256.0,
                    float(entity.frame) / 64.0,
                ]
            )
        entity_features.extend([0.0] * (self.max_entities * 6 - len(entity_features)))

        event_features: List[float] = []
        for event in world_tick.events[: self.max_events]:
            event_features.extend(
                [
                    float(_event_bucket(event.event_type)) / max(len(EVENT_BUCKETS) - 1, 1),
                    1.0 if event.region_id == world_tick.current_region_id else 0.0,
                    1.0 if (event.source_id or event.target_id) else 0.0,
                ]
            )
        event_features.extend([0.0] * (self.max_events * 3 - len(event_features)))

        history_features: List[float] = []
        history = world_tick.action_history[-self.action_history_len :]
        for action in history:
            history_features.extend(normalized_action_features(action))
        history_features.extend([0.0] * (self.action_history_len * 7 - len(history_features)))

        features = np.array(player_features + region_features + entity_features + event_features + history_features, dtype=np.float32)
        assert features.shape == (self.obs_dim,)
        return features
