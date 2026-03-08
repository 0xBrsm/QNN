"""Deterministic symbolic and native-worker navigation environments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import dist, hypot
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from engine.native_bridge import NativeQuakeAdapter
from quake_ai.actions import ACTION_HEADS, ActionLabels, mouse_count_from_look_label
from quake_ai.combat_metrics import WEAPON_TOTAL_DEBUG_KEYS, flatten_weapon_metrics, weapon_metric_key
from quake_ai.models.competitive_encoder import CompetitiveObservationEncoder
from quake_ai.models.world_encoder import WorldObservationEncoder
from quake_ai.navigation import (
    build_observation,
    desired_motion_vector,
    load_navigation_map,
    region_to_point,
    select_neighbor,
)
from quake_ai.rl.reward import RewardWeights, reward_components, reward_mode_from_observation
from quake_ai.schemas import MapStateV2, WorldTickV2


@dataclass(slots=True)
class EnvState:
    region_id: int
    steps: int
    heading: int
    prev_distance: float
    items_collected: set[int]
    goal_reached: bool
    player_vel: Tuple[float, float, float]


@dataclass(slots=True)
class NativeEnvState:
    steps: int
    current_region_id: int | None
    prev_distance: float
    items_collected: int
    player_origin: Tuple[float, float, float]
    frags: int
    monster_kills: int
    weapon_id: int


def _distance_to_goal(map_state: MapStateV2, region_id: int | None) -> float:
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0

    if region_id is None:
        return max_distance

    raw_distances = map_state.metadata.get("distance_to_goal", {})
    if isinstance(raw_distances, dict):
        return float(raw_distances.get(str(region_id), raw_distances.get(region_id, max_distance)))
    return max_distance


def _goal_progress(map_state: MapStateV2, region_id: int | None, goal_reached: bool) -> float:
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0
    distance = 0.0 if goal_reached else _distance_to_goal(map_state, region_id)
    return _goal_progress_from_distance(map_state, distance)


def _goal_progress_from_distance(map_state: MapStateV2, distance: float) -> float:
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0
    return float(max(0.0, min(1.0, 1.0 - (distance / max_distance))))


def _pickup_count(world_tick: WorldTickV2) -> int:
    return sum(1 for event in world_tick.events if event.event_type.lower().startswith("pickup"))


def _goal_reached(world_tick: WorldTickV2, worker_info: Mapping[str, object]) -> bool:
    if bool(worker_info.get("goal_reached", False)):
        return True
    if world_tick.done_reason == "goal_reached":
        return True
    return any(token in event.event_type.lower() for event in world_tick.events for token in ("goal", "intermission"))


def _goal_region_centers(map_state: MapStateV2) -> Tuple[Tuple[float, float, float], ...]:
    centers_by_region = {region.region_id: tuple(float(value) for value in region.center) for region in map_state.regions}
    centers = [centers_by_region[region_id] for region_id in map_state.goal_region_ids if region_id in centers_by_region]
    return tuple(centers)


def _native_distance_to_goal(
    map_state: MapStateV2,
    goal_centers: Tuple[Tuple[float, float, float], ...],
    goal_distance_scale: float,
    origin: Tuple[float, float, float],
    region_id: int | None,
    *,
    goal_reached: bool,
) -> float:
    if goal_reached:
        return 0.0
    if goal_centers:
        return float(min(dist(origin, goal_center) for goal_center in goal_centers) / max(goal_distance_scale, 1.0))
    return _distance_to_goal(map_state, region_id)


def _planar_displacement(previous_origin: Tuple[float, float, float], current_origin: Tuple[float, float, float]) -> float:
    return float(hypot(current_origin[0] - previous_origin[0], current_origin[1] - previous_origin[1]))


def _debug_int(world_tick: WorldTickV2, key: str, fallback: int = 0) -> int:
    value = world_tick.debug.get(key, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _event_delta(world_tick: WorldTickV2, event_type: str) -> int:
    total = 0
    target = event_type.lower()
    for event in world_tick.events:
        if event.event_type.lower() != target:
            continue
        try:
            total += int(event.payload.get("delta", 1))
        except (TypeError, ValueError):
            total += 1
    return total


def _event_count(world_tick: WorldTickV2, event_type: str) -> int:
    target = event_type.lower()
    return sum(1 for event in world_tick.events if event.event_type.lower() == target)


def _event_weapon_metrics(world_tick: WorldTickV2, event_type: str) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    target = event_type.lower()
    for event in world_tick.events:
        if event.event_type.lower() != target:
            continue
        try:
            weapon_id = int(event.payload.get("weapon_id", 0))
        except (TypeError, ValueError):
            weapon_id = 0
        if weapon_id <= 0:
            continue
        try:
            delta = float(event.payload.get("delta", 1))
        except (TypeError, ValueError):
            delta = 1.0
        totals[weapon_id] = totals.get(weapon_id, 0.0) + delta
    return totals


def _debug_list(world_tick: WorldTickV2, key: str) -> list[float]:
    raw = world_tick.debug.get(key, [])
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _is_hostile_entity(classname: str, properties: Mapping[str, object]) -> bool:
    if str(properties.get("source", "")).lower() == "static_proxy":
        return False
    lowered = classname.lower()
    if not lowered:
        return True
    if lowered.startswith("item_") or lowered.startswith("trigger_") or lowered.startswith("info_"):
        return False
    if lowered in {"trigger_changelevel", "path_corner"}:
        return False
    return True


def _visible_threat_count(world_tick: WorldTickV2) -> int:
    return sum(
        1
        for entity in world_tick.visible_entities
        if _is_hostile_entity(entity.classname, entity.properties)
    )


def _combat_signals(
    *,
    world_tick: WorldTickV2,
    state: NativeEnvState,
    visible_threats: int,
    fire_pressed: int,
) -> Dict[str, float]:
    current_frags = _debug_int(world_tick, "frags", state.frags)
    current_monster_kills = _debug_int(world_tick, "monster_kills", state.monster_kills)
    frag_gain = max(0, current_frags - state.frags)
    frag_loss = max(0, state.frags - current_frags)
    monster_kills = max(0, current_monster_kills - state.monster_kills)
    weapon_switches = 1 if (world_tick.player.weapon_id > 0 and world_tick.player.weapon_id != state.weapon_id) else 0
    player_died = 1 if (world_tick.done_reason == "player_died" or _event_count(world_tick, "player_died") > 0) else 0
    damage_dealt = float(_event_delta(world_tick, "damage_dealt"))
    hit_count = float(_event_delta(world_tick, "hit_confirmed"))
    shots_fired = float(max(_event_delta(world_tick, "shots_fired"), fire_pressed))
    weapon_damage = flatten_weapon_metrics("weapon_damage_dealt", _event_weapon_metrics(world_tick, "damage_dealt"))
    weapon_hits = flatten_weapon_metrics("weapon_hits_landed", _event_weapon_metrics(world_tick, "hit_confirmed"))
    weapon_shots = flatten_weapon_metrics("weapon_shots_fired", _event_weapon_metrics(world_tick, "shots_fired"))

    if _event_count(world_tick, "frag_gained") > 0:
        frag_gain = max(frag_gain, _event_delta(world_tick, "frag_gained"))
    if _event_count(world_tick, "frag_lost") > 0:
        frag_loss = max(frag_loss, _event_delta(world_tick, "frag_lost"))
    if _event_count(world_tick, "monster_kill") > 0:
        monster_kills = max(monster_kills, _event_delta(world_tick, "monster_kill"))

    effective_fire = 1 if (shots_fired > 0 and visible_threats > 0) else 0
    blind_fire = 1 if (shots_fired > 0 and visible_threats == 0) else 0
    signals = {
        "frag_gain": float(frag_gain),
        "frag_loss": float(frag_loss),
        "monster_kills": float(monster_kills),
        "damage_taken": float(_event_delta(world_tick, "damage_taken")),
        "damage_dealt": damage_dealt,
        "hit_count": hit_count,
        "shots_fired": shots_fired,
        "health_gain": float(_event_delta(world_tick, "pickup_health")),
        "armor_gain": float(_event_delta(world_tick, "pickup_armor")),
        "ammo_gain": float(_event_delta(world_tick, "pickup_ammo")),
        "weapon_pickups": float(_event_count(world_tick, "pickup_weapon")),
        "weapon_switches": float(weapon_switches),
        "visible_threats": float(visible_threats),
        "fire_pressed": float(fire_pressed),
        "effective_fire": float(effective_fire),
        "blind_fire": float(blind_fire),
        "health_fraction": float(max(0.0, min(world_tick.player.health / 100.0, 1.0))),
        "armor_fraction": float(max(0.0, min(world_tick.player.armor / 100.0, 1.0))),
        "player_died": float(player_died),
        "episode_damage_dealt": float(_debug_int(world_tick, "damage_dealt_total", 0)),
        "episode_hit_count": float(_debug_int(world_tick, "hit_count_total", 0)),
        "episode_shots_fired": float(_debug_int(world_tick, "shots_fired_total", 0)),
    }
    signals.update(weapon_damage)
    signals.update(weapon_hits)
    signals.update(weapon_shots)
    for prefix, debug_key in WEAPON_TOTAL_DEBUG_KEYS.items():
        totals = _debug_list(world_tick, debug_key)
        for weapon_id, value in enumerate(totals):
            if weapon_id <= 0 or value == 0.0:
                continue
            signals[f"episode_{weapon_metric_key(prefix, weapon_id)}"] = float(value)
    return signals


class E1M1NavigationEnv:
    def __init__(
        self,
        map_features_path: str | Path,
        max_steps: int = 256,
        reward_weights: RewardWeights | None = None,
        seed: int = 7,
    ) -> None:
        self.map_features_path = str(map_features_path)
        self.max_steps = max_steps
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)
        self.nav_map = load_navigation_map(map_features_path)
        self.state: EnvState | None = None

    def _observation(self, state: EnvState) -> np.ndarray:
        player_pos = region_to_point(state.region_id)
        nearby = [1, 0, 0, 0] if state.region_id in self.nav_map.item_regions and state.region_id not in state.items_collected else [0, 0, 0, 0]
        return build_observation(
            nav_map=self.nav_map,
            region_id=state.region_id,
            heading=state.heading,
            player_pos=player_pos,
            player_vel=state.player_vel,
            nearby_item_flags=nearby,
            goal_progress=self.nav_map.goal_progress(state.region_id),
        )

    def reset(self, seed: int | None = None, start_variant: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if start_variant is None:
            start_region = int(self.rng.choice(self.nav_map.spawn_regions))
            heading = int(self.rng.integers(0, 8))
        else:
            start_region = self.nav_map.spawn_regions[start_variant % len(self.nav_map.spawn_regions)]
            heading = int(start_variant % 8)

        self.state = EnvState(
            region_id=start_region,
            steps=0,
            heading=heading,
            prev_distance=self.nav_map.distance(start_region),
            items_collected=set(),
            goal_reached=False,
            player_vel=(0.0, 0.0, 0.0),
        )
        return self._observation(self.state)

    def _select_next_region(self, region_id: int, heading: int, action: Mapping[str, int]) -> int:
        desired = desired_motion_vector(
            heading=heading,
            move=int(action.get("move", 0)),
            strafe=int(action.get("strafe", 0)),
        )
        return select_neighbor(self.nav_map, region_id, desired)

    def step(self, action: Mapping[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, float | int | bool]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        labels = ActionLabels.from_dict(action)
        state = self.state
        state.steps += 1
        start_region = state.region_id

        look_yaw_count = mouse_count_from_look_label(labels.look_yaw)
        if look_yaw_count < 0:
            state.heading = (state.heading + 1) % 8
        elif look_yaw_count > 0:
            state.heading = (state.heading - 1) % 8

        next_region = self._select_next_region(state.region_id, state.heading, labels.to_dict())
        state.region_id = next_region

        prev_point = np.array(region_to_point(start_region), dtype=np.float32)
        next_point = np.array(region_to_point(state.region_id), dtype=np.float32)
        velocity = (next_point - prev_point) / 5.0
        state.player_vel = (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        movement_delta = float(hypot(float(next_point[0] - prev_point[0]), float(next_point[1] - prev_point[1])))

        item_picked = False
        if state.region_id in self.nav_map.item_regions and state.region_id not in state.items_collected:
            state.items_collected.add(state.region_id)
            item_picked = True

        at_goal = state.region_id in self.nav_map.goal_regions
        state.goal_reached = at_goal

        timed_out = state.steps >= self.max_steps
        done = state.goal_reached or timed_out

        new_distance = self.nav_map.distance(state.region_id)
        stuck = state.region_id == start_region

        reward_breakdown = reward_components(
            previous_distance=state.prev_distance,
            new_distance=new_distance,
            item_picked=item_picked,
            goal_reached=state.goal_reached,
            timed_out=timed_out,
            stuck=stuck,
            weights=self.reward_weights,
        )
        reward = reward_breakdown["reward_total"]

        state.prev_distance = new_distance

        obs = self._observation(state)
        done_reason = "goal_reached" if state.goal_reached else ("timeout" if timed_out else "")
        info: Dict[str, float | int | bool] = {
            "at_goal": at_goal,
            "goal_reached": state.goal_reached,
            "items_collected": len(state.items_collected),
            "steps": state.steps,
            "distance_to_goal": new_distance,
            "goal_progress": self.nav_map.goal_progress(state.region_id),
            "stuck": stuck,
            "movement_delta": movement_delta,
            "done_reason": done_reason,
        }
        info.update(reward_breakdown)
        return obs, reward, done, info

    def close(self) -> None:
        return None


class NativeWorldEnv:
    """World-v2 environment backed by a native worker process."""

    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        max_steps: int = 256,
        fixed_tick_hz: int = 20,
        reward_weights: RewardWeights | None = None,
        reward_mode: str = "",
        observation_format: str = "world_v2",
        seed: int = 7,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        native_args: Sequence[str] | None = None,
        native_options: Mapping[str, object] | None = None,
        encoder: WorldObservationEncoder | CompetitiveObservationEncoder | None = None,
    ) -> None:
        self.max_steps = max_steps
        resolved_reward_mode = reward_mode_from_observation(observation_format, reward_mode)
        self.reward_weights = reward_weights or RewardWeights(mode=resolved_reward_mode)
        self.rng = np.random.default_rng(seed)
        self.observation_format = observation_format
        if encoder is not None:
            self.encoder = encoder
        elif observation_format == "world_v2_competitive":
            self.encoder = CompetitiveObservationEncoder()
        elif observation_format == "world_v2":
            self.encoder = WorldObservationEncoder()
        else:
            raise ValueError(f"Unsupported native observation_format {observation_format}")
        self.native_options = dict(native_options or {})
        self.adapter = NativeQuakeAdapter(
            executable=executable,
            map_id=map_id,
            fixed_tick_hz=fixed_tick_hz,
            workdir=workdir,
            env=env,
            extra_args=native_args,
            reset_options=self.native_options,
        )
        map_state = self.adapter.map_state_v2()
        if map_state is None:
            self.adapter.close()
            raise RuntimeError("Native worker did not return MapStateV2 in hello payload")
        self.map_state = map_state
        self.goal_centers = _goal_region_centers(map_state)
        self.goal_distance_scale = max(float(map_state.metadata.get("grid_size", 0.0)), 256.0)
        self.state: NativeEnvState | None = None

    def reset(self, seed: int | None = None, start_variant: int | None = None) -> np.ndarray:
        del start_variant
        reset_seed = seed if seed is not None else int(self.rng.integers(0, 2**31 - 1))
        world_tick = self.adapter.reset_world(seed=reset_seed)
        player_origin = tuple(float(value) for value in world_tick.player.origin)
        self.state = NativeEnvState(
            steps=0,
            current_region_id=world_tick.current_region_id,
            prev_distance=_native_distance_to_goal(
                self.map_state,
                self.goal_centers,
                self.goal_distance_scale,
                player_origin,
                world_tick.current_region_id,
                goal_reached=False,
            ),
            items_collected=_pickup_count(world_tick),
            player_origin=player_origin,
            frags=_debug_int(world_tick, "frags", 0),
            monster_kills=_debug_int(world_tick, "monster_kills", 0),
            weapon_id=int(world_tick.player.weapon_id),
        )
        return self.encoder.encode(self.map_state, world_tick)

    def step(self, action: Mapping[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        world_tick, worker_reward, worker_done, worker_info = self.adapter.step_world(action)

        steps = self.state.steps + 1
        previous_origin = self.state.player_origin
        current_region_id = world_tick.current_region_id
        current_origin = tuple(float(value) for value in world_tick.player.origin)
        pickup_count = _pickup_count(world_tick)
        goal_reached = _goal_reached(world_tick, worker_info)
        at_goal = goal_reached or (current_region_id in self.map_state.goal_region_ids if current_region_id is not None else False)
        timed_out = bool(world_tick.done_reason == "timeout" or (steps >= self.max_steps and not (worker_done or world_tick.done)))
        done = bool(worker_done or world_tick.done or timed_out)
        movement_delta = _planar_displacement(previous_origin, current_origin)
        stuck = movement_delta < 1.0
        new_distance = _native_distance_to_goal(
            self.map_state,
            self.goal_centers,
            self.goal_distance_scale,
            current_origin,
            current_region_id,
            goal_reached=goal_reached,
        )

        visible_threats = _visible_threat_count(world_tick)
        combat_signals = _combat_signals(
            world_tick=world_tick,
            state=self.state,
            visible_threats=visible_threats,
            fire_pressed=int(action.get("fire", 0)),
        )
        reward_breakdown = reward_components(
            previous_distance=self.state.prev_distance,
            new_distance=new_distance,
            item_picked=pickup_count > 0,
            goal_reached=goal_reached,
            timed_out=timed_out,
            stuck=stuck,
            weights=self.reward_weights,
            combat_signals=combat_signals,
        )
        reward = reward_breakdown["reward_total"]

        items_collected = self.state.items_collected + pickup_count
        current_frags = _debug_int(world_tick, "frags", self.state.frags)
        current_monster_kills = _debug_int(world_tick, "monster_kills", self.state.monster_kills)
        self.state = NativeEnvState(
            steps=steps,
            current_region_id=current_region_id,
            prev_distance=new_distance,
            items_collected=items_collected,
            player_origin=current_origin,
            frags=current_frags,
            monster_kills=current_monster_kills,
            weapon_id=int(world_tick.player.weapon_id),
        )

        info: Dict[str, object] = dict(worker_info)
        info.update(
            {
                "at_goal": at_goal,
                "goal_reached": goal_reached,
                "items_collected": items_collected,
                "steps": steps,
                "distance_to_goal": new_distance,
                "goal_progress": _goal_progress_from_distance(self.map_state, new_distance),
                "stuck": stuck,
                "movement_delta": movement_delta,
                "done_reason": world_tick.done_reason or ("timeout" if timed_out else ""),
                "worker_done": worker_done,
                "worker_reward": worker_reward,
                "health": int(world_tick.player.health),
                "armor": int(world_tick.player.armor),
                "ammo": int(world_tick.player.ammo),
                "weapon_id": int(world_tick.player.weapon_id),
                "frags": current_frags,
                "monster_kills": current_monster_kills,
                "monster_total": _debug_int(world_tick, "monster_total", 0),
                "frag_delta": float(combat_signals["frag_gain"]),
                "frag_loss": float(combat_signals["frag_loss"]),
                "monster_kill_delta": float(combat_signals["monster_kills"]),
                "damage_taken": float(combat_signals["damage_taken"]),
                "damage_dealt": float(combat_signals["damage_dealt"]),
                "hit_count": float(combat_signals["hit_count"]),
                "shots_fired": float(combat_signals["shots_fired"]),
                "health_gain": float(combat_signals["health_gain"]),
                "armor_gain": float(combat_signals["armor_gain"]),
                "ammo_gain": float(combat_signals["ammo_gain"]),
                "weapon_pickups": float(combat_signals["weapon_pickups"]),
                "weapon_switches": float(combat_signals["weapon_switches"]),
                "visible_threats": int(visible_threats),
                "fire_pressed": int(combat_signals["fire_pressed"]),
                "effective_fire": int(combat_signals["effective_fire"]),
                "blind_fire": int(combat_signals["blind_fire"]),
                "health_fraction": float(combat_signals["health_fraction"]),
                "armor_fraction": float(combat_signals["armor_fraction"]),
                "player_died": bool(combat_signals["player_died"]),
                "episode_damage_dealt": float(combat_signals["episode_damage_dealt"]),
                "episode_hit_count": float(combat_signals["episode_hit_count"]),
                "episode_shots_fired": float(combat_signals["episode_shots_fired"]),
            }
        )
        for key, value in combat_signals.items():
            if key.startswith("weapon_") or key.startswith("episode_weapon_"):
                info[key] = float(value)
        info.update(reward_breakdown)
        return self.encoder.encode(self.map_state, world_tick), reward, done, info

    def close(self) -> None:
        self.adapter.close()


class VectorNavigationEnv:
    """Simple synchronous vector wrapper (num_envs workers)."""

    def __init__(self, num_envs: int, map_features_path: str | Path, max_steps: int, seed: int) -> None:
        self.envs = [
            E1M1NavigationEnv(map_features_path=map_features_path, max_steps=max_steps, seed=seed + i)
            for i in range(num_envs)
        ]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs], axis=0)

    def step(self, action_batch: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float | int | bool]]]:
        next_obs = []
        rewards = []
        dones = []
        infos: List[Dict[str, float | int | bool]] = []

        for idx, env in enumerate(self.envs):
            raw_action = {head: int(values[idx]) for head, values in action_batch.items()}
            action = ActionLabels.from_dict(raw_action).to_dict()
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset(seed=None)
            next_obs.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)

        return (
            np.stack(next_obs, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        return None


class NativeVectorEnv:
    """Simple synchronous vector wrapper around native worker processes."""

    def __init__(
        self,
        num_envs: int,
        executable: str | Path,
        map_id: str,
        max_steps: int,
        seed: int,
        fixed_tick_hz: int = 20,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        reward_mode: str = "",
        observation_format: str = "world_v2",
        native_args: Sequence[str] | None = None,
        native_options: Mapping[str, object] | None = None,
    ) -> None:
        self.envs = [
            NativeWorldEnv(
                executable=executable,
                map_id=map_id,
                max_steps=max_steps,
                fixed_tick_hz=fixed_tick_hz,
                reward_mode=reward_mode,
                observation_format=observation_format,
                seed=seed + i,
                workdir=workdir,
                env=env,
                native_args=native_args,
                native_options=native_options,
            )
            for i in range(num_envs)
        ]
        self._executor = ThreadPoolExecutor(max_workers=max(num_envs, 1), thread_name_prefix="nq-native-env")

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> np.ndarray:
        futures = [self._executor.submit(env.reset) for env in self.envs]
        return np.stack([future.result() for future in futures], axis=0)

    @staticmethod
    def _step_env(
        env: NativeWorldEnv,
        action: Mapping[str, int],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset(seed=None)
        return obs, reward, done, info

    def step(self, action_batch: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
        actions = [
            ActionLabels.from_dict({head: int(values[idx]) for head, values in action_batch.items()}).to_dict()
            for idx in range(self.num_envs)
        ]
        futures = [
            self._executor.submit(self._step_env, env, action)
            for env, action in zip(self.envs, actions)
        ]
        results = [future.result() for future in futures]
        next_obs = [result[0] for result in results]
        rewards = [float(result[1]) for result in results]
        dones = [bool(result[2]) for result in results]
        infos = [result[3] for result in results]

        return (
            np.stack(next_obs, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()
