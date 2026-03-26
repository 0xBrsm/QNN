"""Native-worker PvP environment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import logging

from engine.bridge import NativeEngineError, NativeObsBufferAdapter, NativeTokenAdapter
from engine.training_protocol import TrustedTrainingExtrasV1
from quake_ai.actions import ActionLabels, CONTINUOUS_ACTION_HEADS
from quake_ai.rl.combat_metrics import WEAPON_TOTAL_DEBUG_KEYS, weapon_metric_key
from quake_ai.model.observation import TokenObservationEncoder, visible_threat_count
from quake_ai.rl.reward import WEAPON_TIER_VALUES, RewardWeights, effective_hp, reward_components
from quake_ai.rl.schemas import MapState

if TYPE_CHECKING:
    from mapgen.pool import MapPool


_HEALTH_CAP = 250.0
_ARMOR_CAP = 200.0
_ARMOR_TYPE_CAP = 0.8
_VEL_CAP = 2000.0


@dataclass(frozen=True, slots=True)
class _SelfInfo:
    """Self-token fields extracted from the obs buffer for reward computation."""
    health: float       # normalized [0,1]
    armor: float        # normalized [0,1]
    armor_type: float   # normalized [0,1]
    velocity: list      # [vx, vy, vz] normalized
    weapon_id: int      # weapon embedding ID
    ammo: list          # [shells, nails, rockets, cells] normalized


def _self_info_from_obs(obs: Dict[str, np.ndarray]) -> _SelfInfo:
    """Extract self-token fields from the obs buffer numpy dict."""
    s = obs["self_scalars"]
    return _SelfInfo(
        health=float(s[0]),
        armor=float(s[1]),
        armor_type=float(s[2]),
        velocity=[float(s[15]), float(s[16]), float(s[17])],
        weapon_id=int(obs["self_weapon_id"][0]),
        ammo=[float(s[11]), float(s[12]), float(s[13]), float(s[14])],
    )


@dataclass(slots=True)
class NativeEnvState:
    steps: int
    current_region_id: int | None
    frags: int
    weapon_id: int
    prev_ehp: float = 100.0


def _velocity_magnitude(vel_norm: List[float]) -> float:
    """Planar speed from normalized velocity (`self_token.velocity / 2000`)."""
    return float(hypot(vel_norm[0] * _VEL_CAP, vel_norm[1] * _VEL_CAP))


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

    # Split damage into self (rocket splash etc.) vs other (enemy hits).
    self_ent = training_extras.self_entity_num
    damage_self = 0.0
    damage_other = 0.0
    for record in training_extras.damage_records:
        if record.attacker_entity_num != self_ent:
            continue
        delta = record.damage_health + record.damage_armor
        if record.target_entity_num == self_ent:
            damage_self += delta
        else:
            damage_other += delta

    combat_signals = dict(combat_signals)
    combat_signals.update(
        {
            "frag_gain": float(training_extras.frag_gain),
            "frag_loss": float(training_extras.frag_loss),
            "damage_taken": float(training_extras.damage_taken),
            "damage_dealt": float(training_extras.damage_dealt),
            "damage_dealt_self": float(damage_self),
            "damage_dealt_other": float(damage_other),
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
        max_steps: int,
        fixed_tick_hz: int,
        reward_weights: RewardWeights,
        mode: str,
        seed: int,
        env: Mapping[str, str],
        native_args: Sequence[str],
        options: Mapping[str, object],
        workdir: str | Path | None = None,
        encoder: TokenObservationEncoder | None = None,
        map_pool: MapPool | None = None,
        procgen: dict | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.reward_weights = reward_weights
        self.rng = np.random.default_rng(seed)
        self.encoder = encoder if encoder is not None else TokenObservationEncoder()  # kept for visible_threat_count shape inference
        self.options = dict(options)
        self.map_pool = map_pool
        self._procgen = procgen

        # Procgen: generate the first map inline (no background threads).
        self._maps_dir: Path | None = None
        self._current_map_id: str | None = None
        self._cleanup_generated_maps = bool(self._procgen["cleanup_generated_maps"]) if self._procgen is not None else True
        if self._procgen is not None:
            from mapgen.pool import generate_bsp
            self._maps_dir = Path(self._procgen["maps_dir"])
            seed_val = self.rng.integers(0, 2**31 - 1)
            map_id, _ = generate_bsp(
                int(seed_val), self._maps_dir,
                rooms=int(self._procgen["rooms"]),
                arena_size=int(self._procgen["arena_size"]),
            )
            self._current_map_id = map_id
        elif self.map_pool is not None:
            map_id = self.map_pool.get(timeout=120.0)
            self._current_map_id = map_id
            self._maps_dir = self.map_pool._maps_dir

        self.adapter = NativeObsBufferAdapter(
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
        self.state: NativeEnvState | None = None

    _MAX_PROCGEN_RETRIES = 3

    def reset(self, seed: int | None = None, start_variant: int | None = None) -> Dict[str, np.ndarray]:
        del start_variant
        reset_seed = seed if seed is not None else int(self.rng.integers(0, 2**31 - 1))

        # Swap to a fresh procgen map each episode.  Retry with a different map
        # if the engine crashes (e.g. malformed BSP from the generator).
        if self._procgen is not None or self.map_pool is not None:
            from mapgen.pool import generate_bsp
            old_map_id = self._current_map_id
            last_err: Exception | None = None
            for attempt in range(self._MAX_PROCGEN_RETRIES):
                if self._procgen is not None:
                    seed_val = int(reset_seed + attempt) if seed is not None else int(self.rng.integers(0, 2**31 - 1))
                    new_map_id, _ = generate_bsp(
                        seed_val, self._maps_dir,
                        rooms=int(self._procgen["rooms"]),
                        arena_size=int(self._procgen["arena_size"]),
                    )
                else:
                    new_map_id = self.map_pool.get(timeout=120.0)
                try:
                    new_map_state = self.adapter.change_map(new_map_id)
                    if new_map_state is not None:
                        self.map_state = new_map_state
                    obs, training_extras = self.adapter.reset_obs_with_training(seed=reset_seed)
                    self._current_map_id = new_map_id
                    last_err = None
                    break
                except (NativeEngineError, OSError) as exc:
                    logging.getLogger(__name__).warning(
                        "Procgen map %s failed (attempt %d/%d): %s",
                        new_map_id, attempt + 1, self._MAX_PROCGEN_RETRIES, exc,
                    )
                    # Clean up the failed map immediately.
                    if self._maps_dir and self._cleanup_generated_maps:
                        for ext in (".bsp", ".map", ".log", ".prt"):
                            (self._maps_dir / f"{new_map_id}{ext}").unlink(missing_ok=True)
                    last_err = exc
            if last_err is not None:
                raise last_err
            # Clean up old map files to avoid filling disk.
            if old_map_id and self._maps_dir and self._cleanup_generated_maps:
                for ext in (".bsp", ".map", ".log", ".prt"):
                    p = self._maps_dir / f"{old_map_id}{ext}"
                    p.unlink(missing_ok=True)
        else:
            obs, training_extras = self.adapter.reset_obs_with_training(seed=reset_seed)
        frag_delta = int(training_extras.frag_gain) if training_extras is not None else 0
        si = _self_info_from_obs(obs)
        self.state = NativeEnvState(
            steps=0,
            current_region_id=0,
            frags=frag_delta,
            weapon_id=si.weapon_id,
            prev_ehp=effective_hp(
                float(si.health * _HEALTH_CAP),
                float(si.armor * _ARMOR_CAP),
                float(si.armor_type * _ARMOR_TYPE_CAP),
            ),
        )
        return obs

    def step(self, action: Mapping[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        obs, training_extras = self.adapter.step_obs_with_training(action)

        si = _self_info_from_obs(obs)
        steps = self.state.steps + 1
        worker_done = training_extras is not None and training_extras.done
        timed_out = bool(steps >= self.max_steps and not worker_done)
        done = bool(worker_done or timed_out)
        speed = _velocity_magnitude(si.velocity)
        stuck = speed < 10.0

        raw_health = si.health * _HEALTH_CAP
        raw_armor = si.armor * _ARMOR_CAP
        raw_armor_type = si.armor_type * _ARMOR_TYPE_CAP

        current_frags = self.state.frags + (int(training_extras.frag_gain) if training_extras is not None else 0) - (int(training_extras.frag_loss) if training_extras is not None else 0)
        visible_threats = visible_threat_count(obs)
        combat_signals = _combat_signals(
            state=self.state,
            health=int(raw_health),
            armor=int(raw_armor),
            armor_type=float(raw_armor_type),
            weapon_id=si.weapon_id,
            visible_threats=visible_threats,
            fire_pressed=int(action.get("fire", 0)),
        )
        combat_signals = _apply_training_extras(combat_signals, training_extras=training_extras)
        reward_breakdown = reward_components(
            weights=self.reward_weights,
            combat_signals=combat_signals,
        )
        reward = reward_breakdown["reward_total"]

        current_ehp = effective_hp(
            float(raw_health),
            float(raw_armor),
            float(raw_armor_type),
        )
        # On death ticks, pretend prev_ehp is spawn health (100, no armor)
        # so the respawn tick doesn't get a free +2.3 ehp_delta reward.
        player_died = training_extras is not None and training_extras.player_died
        stored_ehp = 100.0 if player_died else current_ehp
        self.state = NativeEnvState(
            steps=steps,
            current_region_id=0,
            frags=current_frags,
            weapon_id=si.weapon_id,
            prev_ehp=stored_ehp,
        )

        done_reason = ""
        if timed_out:
            done_reason = "timeout"
        elif training_extras is not None and training_extras.player_died:
            done_reason = "player_died"
        elif worker_done:
            done_reason = "done"
        info: Dict[str, object] = {}
        info.update(
            {
                "steps": steps,
                "current_region_id": 0,
                "stuck": stuck,
                "speed": speed,
                "done_reason": done_reason,
                "worker_done": bool(worker_done),
                "worker_reward": 0.0,
                "health": int(raw_health),
                "armor": int(raw_armor),
                "ammo": float(sum(si.ammo)),
                "weapon_id": si.weapon_id,
                "scenario_id": str(self.map_state.map_id),
                "frags": current_frags,
                "monster_kills": 0,
                "monster_total": 0,
                "frag_delta": float(combat_signals["frag_gain"]),
                "frag_loss": float(combat_signals["frag_loss"]),
                "monster_kill_delta": float(combat_signals["monster_kills"]),
                "damage_taken": float(combat_signals["damage_taken"]),
                "damage_dealt": float(combat_signals["damage_dealt"]),
                "damage_dealt_self": float(combat_signals.get("damage_dealt_self", 0)),
                "damage_dealt_other": float(combat_signals.get("damage_dealt_other", 0)),
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
        fixed_tick_hz: int,
        reward_weights: RewardWeights,
        seed: int,
        env: Mapping[str, str],
        mode: str,
        native_args: Sequence[str],
        options: Mapping[str, object],
        workdir: str | Path | None = None,
    ) -> None:
        self.envs = [
            NativeWorldEnv(
                executable=executable,
                map_id=map_id,
                max_steps=max_steps,
                fixed_tick_hz=fixed_tick_hz,
                reward_weights=reward_weights,
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
        action: Mapping[str, object],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset(seed=None)
        return obs, reward, done, info

    def step(self, action_batch: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, object]]]:
        actions = []
        for idx in range(self.num_envs):
            payload: dict[str, object] = {}
            for head, values in action_batch.items():
                if head in CONTINUOUS_ACTION_HEADS:
                    payload[head] = np.asarray(values[idx], dtype=np.float32).tolist()
                else:
                    payload[head] = int(values[idx])
            actions.append(ActionLabels.from_dict(payload).to_dict())
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
