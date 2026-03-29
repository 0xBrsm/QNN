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
from quake_ai.vocab import MODALITY_IDS, SUBJECT_IDS

if TYPE_CHECKING:
    from mapgen.pool import MapPool


_PLAYER_SUBJECT_ID = SUBJECT_IDS["PLAYER"]
_VISUAL_MODALITY_ID = MODALITY_IDS["VISUAL"]


def _tracking_cosine(obs: Dict[str, np.ndarray]) -> float:
    """Cosine of angle between player aim and nearest enemy.

    Returns a value in [-1, +1]: +1 means enemy is dead-center on crosshair,
    -1 means enemy is directly behind. Returns 0.0 if no enemy is visible.

    Object token rel_x/y/z are already in the player's view frame
    (forward/right/up), computed by qnn_relative_frame() in the C worker.
    So rel_x/dist gives the cosine directly — no yaw/pitch needed.
    """
    obj_ids = obs["object_ids"]       # (N, 5) — subject_id at [:, 0]
    obj_sc = obs["object_scalars"]    # (N, 8) — rel_x/y/z at [:, 0:3]
    obj_mask = obs["object_mask"]     # (N,)

    best_cos = 0.0
    best_dist_sq = float("inf")

    for i in range(obj_ids.shape[0]):
        if not obj_mask[i]:
            break
        if int(obj_ids[i, 0]) != _PLAYER_SUBJECT_ID:
            continue

        rx = float(obj_sc[i, 0])  # forward component (view frame)
        ry = float(obj_sc[i, 1])  # right component
        # Offset up component to hitbox center (+28 Quake units / 1024 norm)
        rz = float(obj_sc[i, 2]) + 28.0 / 1024.0
        dist_sq = rx * rx + ry * ry + rz * rz
        if dist_sq < 1e-8:
            continue

        # In view frame, crosshair = forward axis, so cos = rx / dist
        cos = rx / (dist_sq ** 0.5)

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_cos = cos

    return best_cos


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
        self.options["reward_weights"] = {
            "frag_bonus": reward_weights.frag_bonus,
            "death_penalty": reward_weights.death_penalty,
            "ehp_delta_weight": reward_weights.ehp_delta_weight,
            "edp_delta_weight": reward_weights.edp_delta_weight,
            "fire_penalty": reward_weights.fire_penalty,
            "self_damage_penalty": reward_weights.self_damage_penalty,
            "tracking_weight": reward_weights.tracking_weight,
            "tracking_fov": reward_weights.tracking_fov,
            "tracking_penalty": reward_weights.tracking_penalty,
        }
        self._reward_weights = reward_weights
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

        te = training_extras
        steps = self.state.steps + 1
        worker_done = te is not None and te.done
        timed_out = bool(steps >= self.max_steps and not worker_done)
        done = bool(worker_done or timed_out)

        # Reward: use C-computed value from QTRN v2, require it.
        reward = te.computed_reward if te is not None else 0.0

        # Minimal state tracking for frags
        frag_gain = int(te.frag_gain) if te is not None else 0
        frag_loss = int(te.frag_loss) if te is not None else 0
        current_frags = self.state.frags + frag_gain - frag_loss
        self.state = NativeEnvState(
            steps=steps,
            current_region_id=0,
            frags=current_frags,
            weapon_id=0,
            prev_ehp=100.0,
        )

        # Done reason
        done_reason = ""
        if timed_out:
            done_reason = "timeout"
        elif te is not None and te.player_died:
            done_reason = "player_died"
        elif worker_done:
            done_reason = "done"

        # Split damage by type from per-record flags.
        _FLAG_SPLASH = 0x0004
        damage_direct = 0.0
        damage_splash = 0.0
        if te is not None:
            self_ent = te.self_entity_num
            for rec in te.damage_records:
                if rec.attacker_entity_num != self_ent or rec.target_entity_num == self_ent:
                    continue
                delta = rec.damage_health + rec.damage_armor
                if rec.flags & _FLAG_SPLASH:
                    damage_splash += delta
                else:
                    damage_direct += delta

        # Lean info dict: only what EpisodeStatAccumulator and SF need.
        info: Dict[str, object] = {
            "done_reason": done_reason,
            "scenario_id": str(self.map_state.map_id),
            "frag_delta": float(frag_gain),
            "frag_loss": float(frag_loss),
            "player_died": bool(te.player_died) if te is not None else False,
            "damage_dealt": float(te.damage_dealt) if te is not None else 0.0,
            "damage_dealt_self": float(te.damage_dealt_self) if te is not None else 0.0,
            "damage_dealt_other": float(te.damage_dealt - te.damage_dealt_self) if te is not None else 0.0,
            "damage_direct": float(damage_direct),
            "damage_splash": float(damage_splash),
            "damage_taken": float(te.damage_taken) if te is not None else 0.0,
            "hit_count": float(te.hit_count) if te is not None else 0.0,
            "shots_fired": float(te.shots_fired) if te is not None else 0.0,
            "health_gain": float(te.pickup_health) if te is not None else 0.0,
            "armor_gain": float(te.pickup_armor) if te is not None else 0.0,
            "weapon_pickups": float(te.weapon_pickups) if te is not None else 0.0,
            "tracking_cos": float(te.tracking_cos) if te is not None else 0.0,
            "blind_fire": 0,
            "stuck": False,
        }
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
