"""Native-worker PvP environment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import logging

from engine.bridge import NativeEngineError, NativeObsBufferAdapter
from qnn.actions import ActionLabels, CONTINUOUS_ACTION_HEADS
from qnn.env.reward import RewardWeights

if TYPE_CHECKING:
    from mapgen.pool import MapPool


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
        map_pool: MapPool | None = None,
        procgen: dict | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.reward_weights = reward_weights
        self.rng = np.random.default_rng(seed)
        self.options = dict(options)
        self.match_round_reset = self.options.get("round_reset") == "match"
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
        map_id = self.adapter.map_id_snapshot()
        if map_id is None:
            self.adapter.close()
            raise RuntimeError("Native worker did not return map_id in hello payload")
        self.map_id = map_id
        self._steps: int = -1
        self._frags: int = 0

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
                    new_map_id_result = self.adapter.change_map(new_map_id)
                    if new_map_id_result is not None:
                        self.map_id = new_map_id_result
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
        self._steps = 0
        self._frags = frag_delta
        return obs

    def step(self, action: Mapping[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        self.step_send(action)
        return self.step_recv()

    # Split-phase stepping (VecQuakeEnv): send every lane's action first so
    # all engines sim the tick concurrently, then drain the replies. One
    # in-flight step per env; step() == step_send()+step_recv().
    def step_send(self, action: Mapping[str, int]) -> None:
        if self._steps < 0:
            raise RuntimeError("Call reset() before step()")
        self.adapter.process.step_send(action)

    def pack_step_action(self, action: Mapping[str, object]) -> bytes:
        return self.adapter.process.pack_step_request(action)

    def step_send_packed(self, payload) -> None:
        if self._steps < 0:
            raise RuntimeError("Call reset() before step()")
        self.adapter.process.step_send_packed(payload)

    def step_recv(self) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        obs, training_extras = self.adapter.process.step_recv()
        return self._book_step(obs, training_extras)

    def step_recv_raw(self) -> Tuple[bytes, float, bool, Dict[str, object]]:
        """Drain with the obs left as raw wire bytes — the vectorized
        driver batch-unpacks all lanes at once. Bookkeeping identical."""
        raw, training_extras = self.adapter.process.step_recv_raw()
        return self._book_step(raw, training_extras)

    def round_reset_raw(self) -> bytes:
        """Reset this seat's whole 1v1 match without reloading the world."""
        if not self.match_round_reset:
            raise RuntimeError("round_reset_raw requires options.round_reset='match'")
        raw, training_extras = self.adapter.process.round_reset_raw()
        self._steps = 0
        self._frags = int(training_extras.frag_gain) if training_extras is not None else 0
        return raw

    def _book_step(self, obs, training_extras):
        te = training_extras
        self._steps += 1
        worker_done = te is not None and te.done
        timed_out = bool(self._steps >= self.max_steps and not worker_done)
        done = bool(worker_done or timed_out)

        # Reward: use C-computed value from QTRN v2, require it.
        reward = te.computed_reward if te is not None else 0.0

        # Minimal state tracking for frags
        frag_gain = int(te.frag_gain) if te is not None else 0
        frag_loss = int(te.frag_loss) if te is not None else 0
        self._frags += frag_gain - frag_loss

        # Done reason
        done_reason = ""
        if timed_out:
            done_reason = "timeout"
        elif te is not None and te.player_died:
            done_reason = "player_died"
        elif worker_done:
            done_reason = "done"

        # Death attribution: when the model died this frame, was it self-inflicted
        # (own rocket splash / environment-via-self -> attacker == self) or a kill
        # by the opponent? The engine emits one death record per death carrying the
        # victim and attacker entity nums; attacker == victim == self is a suicide.
        player_suicide = False
        if te is not None and te.player_died:
            self_ent = te.self_entity_num
            for d in te.death_records:
                if d.victim_entity_num == self_ent:
                    player_suicide = d.attacker_entity_num == self_ent
                    break

        # Split damage by type + SOURCE from per-record attacker/target. Dealt
        # (attacker==self, target!=self): direct vs splash. Taken (target==self):
        # self-inflicted (own splash / environment-via-self, attacker==self) vs
        # opponent (attacker!=self) — so "reckless self-splash" is separable from
        # "aggressive, taking opponent fire" (see the P1 A/B damage caveat).
        _FLAG_SPLASH = 0x0004
        damage_direct = 0.0
        damage_splash = 0.0
        damage_taken_self = 0.0
        damage_taken_other = 0.0
        if te is not None:
            self_ent = te.self_entity_num
            for rec in te.damage_records:
                delta = rec.damage_health + rec.damage_armor
                if rec.target_entity_num == self_ent:
                    if rec.attacker_entity_num == self_ent:
                        damage_taken_self += delta
                    else:
                        damage_taken_other += delta
                elif rec.attacker_entity_num == self_ent:
                    if rec.flags & _FLAG_SPLASH:
                        damage_splash += delta
                    else:
                        damage_direct += delta

        # Lean info dict: only what native PPO and episode statistics need.
        info: Dict[str, object] = {
            "done_reason": done_reason,
            # Prefer the eval's scenario spec id (options["scenario_id"]) when
            # set, so multiple scenarios sharing one map (e.g. the aim grid's
            # per-weapon cells) report distinct ids; fall back to the map id
            # (procgen / single-scenario evals leave it unset).
            "scenario_id": self.options.get("scenario_id", self.map_id),
            "frag_delta": float(frag_gain),
            "frag_loss": float(frag_loss),
            "player_died": bool(te.player_died) if te is not None else False,
            "player_suicide": bool(player_suicide),
            "damage_dealt": float(te.damage_dealt) if te is not None else 0.0,
            "damage_dealt_self": float(te.damage_dealt_self) if te is not None else 0.0,
            "damage_dealt_other": float(te.damage_dealt - te.damage_dealt_self) if te is not None else 0.0,
            "damage_direct": float(damage_direct),
            "damage_splash": float(damage_splash),
            "damage_taken": float(te.damage_taken) if te is not None else 0.0,
            "damage_taken_self": float(damage_taken_self),
            "damage_taken_other": float(damage_taken_other),
            "hit_count": float(te.hit_count) if te is not None else 0.0,
            "shots_fired": float(te.shots_fired) if te is not None else 0.0,
            "health_gain": float(te.pickup_health) if te is not None else 0.0,
            "armor_gain": float(te.pickup_armor) if te is not None else 0.0,
            "weapon_pickups": float(te.weapon_pickups) if te is not None else 0.0,
            "tracking_cos": float(te.tracking_cos) if te is not None else 0.0,
            # v3 lead-aim geometry (QTRN v3): view-frame rel pos + ABSOLUTE world
            # velocity (raw u / u·s⁻¹) of the SAME nearest in-LOS actor that
            # tracking_cos selects, plus the currently-held weapon id and a valid
            # flag. The eval recomputes a LEAD-POINT-referenced aim cosine from
            # these via the shared lead kernel (lead_aim.compute_lead_aim).
            "lead_rel": tuple(te.lead_rel) if te is not None else (0.0, 0.0, 0.0),
            "lead_vel": tuple(te.lead_vel) if te is not None else (0.0, 0.0, 0.0),
            "lead_weapon_id": int(te.lead_weapon_id) if te is not None else 0,
            "lead_valid": bool(te.lead_valid) if te is not None else False,
            "stuck": False,
        }
        # In match mode the worker already paired the terminal reward with a
        # live post-respawn observation.  Start the Python timeout counter for
        # the new bout immediately; VecQuakeEnv advances the episode id after
        # it books this terminal transition.
        if self.match_round_reset and te is not None and te.player_died:
            self._steps = 0
            self._frags = 0
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
