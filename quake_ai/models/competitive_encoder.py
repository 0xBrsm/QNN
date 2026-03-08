"""Competitive observation encoder with compact combat semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from quake_ai.actions import normalized_action_features
from quake_ai.schemas import MapStateV2, WorldTickV2

_EVENT_BUCKETS = ("combat", "pickup", "movement", "meta", "other")
_PROJECTILE_TOKENS = ("rocket", "grenade", "spike", "bolt", "missile", "plasma")


def _event_bucket(event_type: str) -> int:
    lowered = event_type.lower()
    if "kill" in lowered or "damage" in lowered or "fire" in lowered or "hit" in lowered:
        return 0
    if lowered.startswith("pickup"):
        return 1
    if "jump" in lowered or "move" in lowered:
        return 2
    if "intermission" in lowered or "goal" in lowered or "player_died" in lowered:
        return 3
    return 4


def _hash_features(text: str) -> tuple[float, float]:
    token = str(text).strip().lower()
    if not token:
        return 0.0, 0.0
    h = hash(token) & 0xFFFF
    angle = (h / 65535.0) * 2.0 * math.pi
    return math.cos(angle), math.sin(angle)


def _map_hash_features(map_id: str) -> tuple[float, float]:
    return _hash_features(map_id)


def _entity_role_flags(classname: str, properties: Mapping[str, object]) -> tuple[float, float, float]:
    lowered = classname.lower()
    if lowered == "player":
        return 1.0, 0.0, 0.0
    if lowered.startswith("item_") or str(properties.get("source", "")).lower() == "static_proxy":
        return 0.0, 1.0, 0.0
    if any(token in lowered for token in _PROJECTILE_TOKENS):
        return 0.0, 0.0, 1.0
    return 0.0, 0.0, 0.0


def _property_value(properties: Mapping[str, object], key: str, scale: float) -> float:
    try:
        value = float(properties.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if scale <= 0.0:
        return value
    return float(np.clip(value / scale, -1.0, 1.0))


def _event_payload_value(payload: Mapping[str, object], key: str, scale: float) -> float:
    try:
        value = float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if scale <= 0.0:
        return value
    return float(np.clip(value / scale, -1.0, 1.0))


@dataclass(slots=True)
class CompetitiveObservationEncoder:
    max_entities: int = 8
    max_events: int = 8
    action_history_len: int = 4

    @property
    def obs_dim(self) -> int:
        player_dim = 14
        context_dim = 8
        entity_dim = self.max_entities * 15
        event_dim = self.max_events * 8
        history_dim = self.action_history_len * 7
        return player_dim + context_dim + entity_dim + event_dim + history_dim

    def _context_counts(self, visible_entities: Iterable[object]) -> tuple[float, float, float]:
        players = 0.0
        items = 0.0
        projectiles = 0.0
        for entity in visible_entities:
            player_like, item_like, projectile_like = _entity_role_flags(entity.classname, entity.properties)
            players += player_like
            items += item_like
            projectiles += projectile_like
        return players, items, projectiles

    def encode(self, map_state: MapStateV2, world_tick: WorldTickV2) -> np.ndarray:
        yaw = float(world_tick.player.view_angles[1]) if len(world_tick.player.view_angles) > 1 else 0.0
        pitch = float(world_tick.player.view_angles[0]) if len(world_tick.player.view_angles) > 0 else 0.0
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
            float(np.clip(world_tick.player.health / 100.0, -1.0, 1.0)),
            float(np.clip(world_tick.player.armor / 100.0, 0.0, 1.0)),
            float(np.clip(world_tick.player.ammo / 100.0, 0.0, 1.0)),
            float(np.clip(world_tick.player.weapon_id / 8.0, 0.0, 1.0)),
            1.0 if bool(world_tick.player.grounded) else 0.0,
        ]

        player_count, item_count, projectile_count = self._context_counts(world_tick.visible_entities)
        cos_h, sin_h = _map_hash_features(world_tick.map_id)
        context_features = [
            len(world_tick.visible_entities) / 16.0,
            len(world_tick.events) / 16.0,
            player_count / max(self.max_entities, 1),
            item_count / max(self.max_entities, 1),
            projectile_count / max(self.max_entities, 1),
            cos_h,
            sin_h,
            min(float(world_tick.tick) / 4096.0, 1.0),
        ]

        player_origin = np.asarray(world_tick.player.origin, dtype=np.float32)
        entity_features: List[float] = []
        sorted_entities = sorted(world_tick.visible_entities, key=lambda row: (row.entity_num, row.entity_id))
        for entity in sorted_entities[: self.max_entities]:
            rel = np.asarray(entity.origin, dtype=np.float32) - player_origin
            class_cos, class_sin = _hash_features(entity.classname)
            player_like, item_like, projectile_like = _entity_role_flags(entity.classname, entity.properties)
            entity_features.extend(
                [
                    float(rel[0]) / 1024.0,
                    float(rel[1]) / 1024.0,
                    float(rel[2]) / 256.0,
                    float(entity.velocity[0]) / 320.0,
                    float(entity.velocity[1]) / 320.0,
                    float(entity.velocity[2]) / 320.0,
                    1.0 if entity.region_id == world_tick.current_region_id else 0.0,
                    1.0 if getattr(entity, "visible", True) else 0.0,
                    player_like,
                    item_like,
                    projectile_like,
                    class_cos,
                    class_sin,
                    _property_value(entity.properties, "health", 100.0),
                    _property_value(entity.properties, "frags", 20.0),
                ]
            )
        entity_features.extend([0.0] * (self.max_entities * 15 - len(entity_features)))

        event_features: List[float] = []
        for event in world_tick.events[: self.max_events]:
            event_cos, event_sin = _hash_features(event.event_type)
            event_features.extend(
                [
                    float(_event_bucket(event.event_type)) / max(len(_EVENT_BUCKETS) - 1, 1),
                    1.0 if event.region_id == world_tick.current_region_id else 0.0,
                    1.0 if bool(event.source_id) else 0.0,
                    1.0 if bool(event.target_id) else 0.0,
                    _event_payload_value(event.payload, "delta", 100.0),
                    _event_payload_value(event.payload, "weapon_id", 8.0),
                    event_cos,
                    event_sin,
                ]
            )
        event_features.extend([0.0] * (self.max_events * 8 - len(event_features)))

        history_features: List[float] = []
        history = world_tick.action_history[-self.action_history_len :]
        for action in history:
            history_features.extend(normalized_action_features(action))
        history_features.extend([0.0] * (self.action_history_len * 7 - len(history_features)))

        features = np.array(player_features + context_features + entity_features + event_features + history_features, dtype=np.float32)
        assert features.shape == (self.obs_dim,)
        return features
