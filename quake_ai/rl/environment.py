"""Native-worker PvP environment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import dist, hypot
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from engine.bridge import NativeTokenAdapter
from engine.training_protocol import TrustedTrainingExtrasV1
from quake_ai.actions import ActionLabels
from quake_ai.rl.combat_metrics import WEAPON_TOTAL_DEBUG_KEYS, weapon_metric_key
from quake_ai.model.observation import TokenObservationEncoder, visible_threat_count
from quake_ai.rl.reward import WEAPON_TIER_VALUES, RewardWeights, effective_hp, reward_components
from quake_ai.rl.schemas import MapState


@dataclass(slots=True)
class NativeEnvState:
    steps: int
    current_region_id: int | None
    prev_distance: float
    items_collected: int
    player_origin: Tuple[float, float, float]
    frags: int
    weapon_id: int
    prev_ehp: float = 100.0


def _distance_to_goal(map_state: MapState, region_id: int | None) -> float:
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0

    if region_id is None:
        return max_distance

    raw_distances = map_state.metadata.get("distance_to_goal", {})
    if isinstance(raw_distances, dict):
        return float(raw_distances.get(str(region_id), raw_distances.get(region_id, max_distance)))
    return max_distance


def _goal_progress(map_state: MapState, region_id: int | None, goal_reached: bool) -> float:
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0
    distance = 0.0 if goal_reached else _distance_to_goal(map_state, region_id)
    return _goal_progress_from_distance(map_state, distance)


def _goal_progress_from_distance(map_state: MapState, distance: float) -> float:
    max_distance = float(map_state.metadata.get("max_distance_to_goal", 0.0))
    if max_distance <= 0.0:
        max_distance = 1.0
    return float(max(0.0, min(1.0, 1.0 - (distance / max_distance))))


def _goal_region_centers(map_state: MapState) -> Tuple[Tuple[float, float, float], ...]:
    centers_by_region = {region.region_id: tuple(float(value) for value in region.center) for region in map_state.regions}
    centers = [centers_by_region[region_id] for region_id in map_state.goal_region_ids if region_id in centers_by_region]
    return tuple(centers)


def _native_distance_to_goal(
    map_state: MapState,
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


def _combat_signals(
    *,
    state: NativeEnvState,
    health: int,
    armor: int,
    armor_type: float,
    weapon_id: int,
    visible_threats: int,
    fire_pressed: int,
) -> Dict[str, float]:
    shots_fired = float(fire_pressed)
    effective_fire = 1 if (shots_fired > 0 and visible_threats > 0) else 0
    blind_fire = 1 if (shots_fired > 0 and visible_threats == 0) else 0
    signals = {
        "frag_gain": 0.0,
        "frag_loss": 0.0,
        "monster_kills": 0.0,
        "damage_taken": 0.0,
        "damage_dealt": 0.0,
        "hit_count": 0.0,
        "shots_fired": shots_fired,
        "health_gain": 0.0,
        "armor_gain": 0.0,
        "ammo_gain": 0.0,
        "weapon_pickups": 0.0,
        "weapon_switches": float(1 if (weapon_id > 0 and weapon_id != state.weapon_id) else 0),
        "visible_threats": float(visible_threats),
        "fire_pressed": float(fire_pressed),
        "effective_fire": float(effective_fire),
        "blind_fire": float(blind_fire),
        "health": float(health),
        "armor": float(armor),
        "armor_type": float(armor_type),
        "prev_ehp": float(state.prev_ehp),
        "health_fraction": float(max(0.0, min(health / 100.0, 1.0))),
        "armor_fraction": float(max(0.0, min(armor / 100.0, 1.0))),
        "player_died": 0.0,
        "episode_damage_dealt": 0.0,
        "episode_hit_count": 0.0,
        "episode_shots_fired": 0.0,
    }
    return signals


def _apply_training_extras(
    combat_signals: Dict[str, float],
    *,
    training_extras: TrustedTrainingExtrasV1 | None,
) -> Dict[str, float]:
    if training_extras is None:
        return combat_signals

    weapon_pickup_value = 0.0
    for record in training_extras.item_records:
        if record.actor_entity_num != training_extras.self_entity_num:
            continue
        if record.event_kind != 1 or record.category != 4:
            continue
        weapon_pickup_value += WEAPON_TIER_VALUES.get(int(record.weapon_id), 1.0)

    combat_signals = dict(combat_signals)
    combat_signals.update(
        {
            "frag_gain": float(training_extras.frag_gain),
            "frag_loss": float(training_extras.frag_loss),
            "damage_taken": float(training_extras.damage_taken),
            "damage_dealt": float(training_extras.damage_dealt),
            "hit_count": float(training_extras.hit_count),
            "shots_fired": float(training_extras.shots_fired),
            "health_gain": float(training_extras.pickup_health),
            "armor_gain": float(training_extras.pickup_armor),
            "ammo_gain": float(training_extras.pickup_ammo),
            "weapon_pickups": float(weapon_pickup_value),
            "player_died": float(1.0 if training_extras.player_died else 0.0),
            "episode_damage_dealt": float(training_extras.episode_damage_dealt),
            "episode_hit_count": float(training_extras.episode_hit_count),
            "episode_shots_fired": float(training_extras.episode_shots_fired),
            "edp_raw": float(training_extras.edp_raw),
        }
    )
    return combat_signals


class NativeWorldEnv:
    """PvP environment backed by a native worker process."""

    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        max_steps: int = 256,
        fixed_tick_hz: int = 20,
        reward_weights: RewardWeights | None = None,
        mode: str = "",
        seed: int = 7,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        native_args: Sequence[str] | None = None,
        options: Mapping[str, object] | None = None,
        encoder: TokenObservationEncoder | None = None,
    ) -> None:
        self.max_steps = max_steps
        resolved_mode = mode.strip().lower() if mode else "pvp"
        if resolved_mode != "pvp":
            raise ValueError(f"Unsupported env mode {mode!r}; only 'pvp' is supported")
        if reward_weights is not None:
            self.reward_weights = reward_weights
        else:
            overrides = dict((options or {}).get("reward_overrides", {}))
            self.reward_weights = RewardWeights(mode=resolved_mode, **overrides)
        self.rng = np.random.default_rng(seed)
        self.encoder = encoder if encoder is not None else TokenObservationEncoder()
        self.options = dict(options or {})
        self.adapter = NativeTokenAdapter(
            executable=executable,
            map_id=map_id,
            fixed_tick_hz=fixed_tick_hz,
            workdir=workdir,
            env=env,
            extra_args=native_args,
            reset_options=self.options,
            training_format="binary_v1",
        )
        map_state = self.adapter.map_state_snapshot()
        if map_state is None:
            self.adapter.close()
            raise RuntimeError("Native worker did not return MapState in hello payload")
        self.map_state = map_state
        self.goal_centers = _goal_region_centers(map_state)
        self.goal_distance_scale = max(float(map_state.metadata.get("grid_size", 0.0)), 256.0)
        self.state: NativeEnvState | None = None

    def reset(self, seed: int | None = None, start_variant: int | None = None) -> Dict[str, np.ndarray]:
        del start_variant
        reset_seed = seed if seed is not None else int(self.rng.integers(0, 2**31 - 1))
        self.encoder.reset()
        token_tick, training_extras = self.adapter.reset_tokens_with_training(seed=reset_seed)
        player_origin = tuple(float(value) for value in token_tick.self_token.origin)
        frag_delta = int(training_extras.frag_gain) if training_extras is not None else 0
        self.state = NativeEnvState(
            steps=0,
            current_region_id=token_tick.current_region_id,
            prev_distance=_native_distance_to_goal(
                self.map_state,
                self.goal_centers,
                self.goal_distance_scale,
                player_origin,
                token_tick.current_region_id,
                goal_reached=False,
            ),
            items_collected=0,
            player_origin=player_origin,
            frags=frag_delta,
            weapon_id=int(token_tick.self_token.weapon_id),
            prev_ehp=effective_hp(
                float(token_tick.self_token.health),
                float(token_tick.self_token.armor),
                float(token_tick.self_token.armor_type),
            ),
        )
        return self.encoder.encode(token_tick)

    def step(self, action: Mapping[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        token_tick, training_extras = self.adapter.step_tokens_with_training(action)
        obs = self.encoder.encode(token_tick)

        steps = self.state.steps + 1
        previous_origin = self.state.player_origin
        current_region_id = token_tick.current_region_id
        current_origin = tuple(float(value) for value in token_tick.self_token.origin)
        pickup_count = int(sum(1 for row in (training_extras.item_records if training_extras is not None else ()) if row.event_kind == 1))
        goal_reached = False
        at_goal = goal_reached or (current_region_id in self.map_state.goal_region_ids if current_region_id is not None else False)
        timed_out = bool(steps >= self.max_steps and not token_tick.done)
        done = bool(token_tick.done or timed_out)
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

        current_frags = self.state.frags + (int(training_extras.frag_gain) if training_extras is not None else 0) - (int(training_extras.frag_loss) if training_extras is not None else 0)
        visible_threats = visible_threat_count(obs)
        combat_signals = _combat_signals(
            state=self.state,
            health=int(token_tick.self_token.health),
            armor=int(token_tick.self_token.armor),
            armor_type=float(token_tick.self_token.armor_type),
            weapon_id=int(token_tick.self_token.weapon_id),
            visible_threats=visible_threats,
            fire_pressed=int(action.get("fire", 0)),
        )
        combat_signals = _apply_training_extras(combat_signals, training_extras=training_extras)
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
        current_ehp = effective_hp(
            float(token_tick.self_token.health),
            float(token_tick.self_token.armor),
            float(token_tick.self_token.armor_type),
        )
        self.state = NativeEnvState(
            steps=steps,
            current_region_id=current_region_id,
            prev_distance=new_distance,
            items_collected=items_collected,
            player_origin=current_origin,
            frags=current_frags,
            weapon_id=int(token_tick.self_token.weapon_id),
            prev_ehp=current_ehp,
        )

        done_reason = ""
        if timed_out:
            done_reason = "timeout"
        elif training_extras is not None and training_extras.player_died:
            done_reason = "player_died"
        elif token_tick.done:
            done_reason = "done"
        info: Dict[str, object] = {}
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
                "done_reason": done_reason,
                "worker_done": bool(token_tick.done),
                "worker_reward": 0.0,
                "health": int(token_tick.self_token.health),
                "armor": int(token_tick.self_token.armor),
                "ammo": int(
                    token_tick.self_token.ammo_shells
                    + token_tick.self_token.ammo_nails
                    + token_tick.self_token.ammo_rockets
                    + token_tick.self_token.ammo_cells
                ),
                "weapon_id": int(token_tick.self_token.weapon_id),
                "scenario_id": str(self.map_state.map_id),
                "frags": current_frags,
                "monster_kills": 0,
                "monster_total": 0,
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
                "training_extras_tick": int(training_extras.tick) if training_extras is not None else -1,
            }
        )
        for prefix, _debug_key in WEAPON_TOTAL_DEBUG_KEYS.items():
            for weapon_id in range(1, 9):
                metric_key = f"episode_{weapon_metric_key(prefix, weapon_id)}"
                if metric_key in combat_signals:
                    info[metric_key] = float(combat_signals[metric_key])
        for key, value in combat_signals.items():
            if key.startswith("weapon_") and key not in info:
                info[key] = float(value)
        info.update(reward_breakdown)
        return obs, reward, done, info

    def close(self) -> None:
        self.adapter.close()


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
        mode: str = "",
        native_args: Sequence[str] | None = None,
        options: Mapping[str, object] | None = None,
    ) -> None:
        self.envs = [
            NativeWorldEnv(
                executable=executable,
                map_id=map_id,
                max_steps=max_steps,
                fixed_tick_hz=fixed_tick_hz,
                mode=mode,
                seed=seed + i,
                workdir=workdir,
                env=env,
                native_args=native_args,
                options=options,
            )
            for i in range(num_envs)
        ]
        self._executor = ThreadPoolExecutor(max_workers=max(num_envs, 1), thread_name_prefix="nq-native-env")

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @staticmethod
    def _stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = obs_list[0].keys()
        return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in keys}

    def reset(self) -> Dict[str, np.ndarray]:
        futures = [self._executor.submit(env.reset) for env in self.envs]
        return self._stack_obs([future.result() for future in futures])

    @staticmethod
    def _step_env(
        env: NativeWorldEnv,
        action: Mapping[str, int],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset(seed=None)
        return obs, reward, done, info

    def step(self, action_batch: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, object]]]:
        actions = [
            ActionLabels.from_dict({head: int(values[idx]) for head, values in action_batch.items()}).to_dict()
            for idx in range(self.num_envs)
        ]
        futures = [
            self._executor.submit(self._step_env, env, action)
            for env, action in zip(self.envs, actions)
        ]
        results = [future.result() for future in futures]
        next_obs = self._stack_obs([result[0] for result in results])
        rewards = [float(result[1]) for result in results]
        dones = [bool(result[2]) for result in results]
        infos = [result[3] for result in results]

        return (
            next_obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()
