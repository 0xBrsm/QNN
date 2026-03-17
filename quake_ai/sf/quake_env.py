"""Gymnasium wrapper around NativeWorldEnv for Sample Factory."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import gymnasium
except ImportError as exc:
    raise ImportError("gymnasium is required: pip install gymnasium>=0.29.0") from exc

from quake_ai.actions import ACTION_HEADS
from quake_ai.rl.environment import NativeWorldEnv
from quake_ai.model.observation import ACTION_HISTORY_DIM, ACTION_HISTORY_LEN, SELF_SCALAR_DIM
from mapgen.pool import PROCGEN_SENTINEL

# Deterministic head ordering (Python 3.7+ dict preserves insertion order)
_HEAD_ORDER: list[str] = list(ACTION_HEADS.keys())
_HEAD_SIZES: list[int] = list(ACTION_HEADS.values())


def multi_discrete_to_heads(action: np.ndarray) -> Dict[str, int]:
    """Convert a SF MultiDiscrete action array to an action_heads dict."""
    return {head: int(action[i]) for i, head in enumerate(_HEAD_ORDER)}


def heads_to_multi_discrete(heads: Dict[str, int]) -> np.ndarray:
    """Convert an action_heads dict to a MultiDiscrete array."""
    return np.array([heads[head] for head in _HEAD_ORDER], dtype=np.int64)


class QuakeEnv(gymnasium.Env):
    """Gymnasium env wrapping NativeWorldEnv for Sample Factory APPO.

    Observation space: Dict of self/object/event/spatial token tensors from the inner encoder.
    Action space: Tuple[Discrete] matching ACTION_HEADS. SF 2.1.x requires
    Tuple[Discrete] rather than MultiDiscrete.

    SF passes env_config with at least "worker_index" and "env_id" keys.
    Per-worker scenario assignment uses env_id % len(scenarios) if a scenario
    list is configured; otherwise all workers use the single configured map.
    """

    metadata: Dict[str, Any] = {}

    def __init__(
        self,
        full_env_name: str,
        cfg: Any,
        env_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        env_config = env_config or {}
        env_id: int = int(env_config.get("env_id", 0))

        # Native args / options may be JSON-encoded strings from CLI or plain objects
        native_args: Sequence[str] | None = _parse_json_arg(cfg, "quake_native_args_json", list) or None
        options: Dict[str, object] | None = _parse_json_arg(cfg, "quake_options_json", dict) or None

        # Scenario rotation: if quake_scenario_config_json is provided it overrides map_id
        map_id = str(getattr(cfg, "quake_map_id", PROCGEN_SENTINEL))
        scenario_cfg_json = getattr(cfg, "quake_scenario_config_json", "") or ""
        scenario: Dict[str, Any] = {}
        if scenario_cfg_json:
            scenarios: list[Dict[str, Any]] = json.loads(scenario_cfg_json)
            scenario = scenarios[env_id % len(scenarios)]
            map_id = str(scenario.get("map_id", map_id))
            self.scenario_id = str(scenario.get("scenario_id", map_id))
            if "native_args" in scenario:
                native_args = list(scenario["native_args"])
            if "options" in scenario:
                # Merge scenario options over base options (preserves deathmatch, etc.)
                base = dict(options) if options else {}
                base.update(scenario["options"])
                options = base
        else:
            self.scenario_id = map_id

        basedir = getattr(cfg, "quake_basedir", "") or ""
        workdir = getattr(cfg, "quake_native_workdir", "") or ""

        # Procgen: generate maps inline on each reset() — no background threads.
        procgen_cfg: dict | None = None
        if map_id == PROCGEN_SENTINEL:
            maps_dir = Path(basedir) / "id1" / "maps" if basedir else Path("assets") / "id1" / "maps"
            procgen_opts = scenario.get("procgen", {})
            procgen_cfg = {
                "maps_dir": str(maps_dir),
                "arena_size": int(procgen_opts.get("arena_size", 3072)),
                "rooms": int(procgen_opts.get("rooms", 3)),
            }

        self.inner_env = NativeWorldEnv(
            executable=str(cfg.quake_executable),
            map_id=map_id,
            max_steps=int(getattr(cfg, "quake_max_steps_per_episode", 1024)),
            fixed_tick_hz=int(getattr(cfg, "quake_fixed_tick_hz", 20)),
            mode=str(getattr(cfg, "quake_mode", "")),
            seed=int(getattr(cfg, "quake_seed", 7)) + env_id,
            env={"QUAKE_BASEDIR": basedir} if basedir else None,
            native_args=native_args,
            options=options,
            workdir=workdir or None,
            procgen=procgen_cfg,
        )

        # Episode-level accumulators for SF custom metrics
        self._ep_frags = 0
        self._ep_deaths = 0
        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._ep_steps = 0
        self._ep_reward = 0.0
        # Skill metrics
        self._ep_hits = 0
        self._ep_shots = 0
        self._ep_health_pickups = 0.0
        self._ep_armor_pickups = 0.0
        self._ep_weapon_pickups = 0.0
        self._ep_blind_fires = 0
        self._ep_stuck_steps = 0
        # Reward decomposition
        self._ep_reward_frag = 0.0
        self._ep_reward_death = 0.0
        self._ep_reward_ehp = 0.0
        self._ep_reward_edp = 0.0

        # Demo recording: always record to the same file per env,
        # overwriting each episode.  Best demo archiving is handled by
        # the BestCheckpointArchiver observer (observer.py), not here.
        # Demos go in ``<com_gamedir>/demos/``.  When ``-game X`` is passed,
        # Quake sets com_gamedir to X (e.g. frikbotnex_train), otherwise id1.
        self._record_demos = bool(getattr(cfg, "quake_record_demos", False))
        num_policies = int(getattr(cfg, "num_policies", 1))
        self._policy_id = env_id % num_policies
        self._demo_name = f"train_p{self._policy_id}_w{env_id:03d}"
        game_subdir = "id1"
        if native_args:
            for i, arg in enumerate(native_args):
                if arg == "-game" and i + 1 < len(native_args):
                    game_subdir = native_args[i + 1]
                    break
        gamedir = Path(basedir or "assets") / game_subdir
        if workdir:
            gamedir = Path(workdir) / gamedir
        demos_dir = gamedir / "demos"
        self._demo_path = demos_dir / f"{self._demo_name}.dem"
        self._demo_last_path = demos_dir / f"{self._demo_name}_last.dem"
        if self._record_demos:
            demos_dir.mkdir(parents=True, exist_ok=True)
            self._inject_record_cmd()

        self.observation_space = gymnasium.spaces.Dict({
            "self_scalars": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(SELF_SCALAR_DIM,), dtype=np.float32,
            ),
            "self_weapon_id": gymnasium.spaces.Box(
                low=0, high=4, shape=(1,), dtype=np.int32,
            ),
            "self_movement_id": gymnasium.spaces.Box(
                low=0, high=4, shape=(1,), dtype=np.int32,
            ),
            "self_cluster_id": gymnasium.spaces.Box(
                low=0, high=255, shape=(1,), dtype=np.int32,
            ),
            "object_ids": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=self.inner_env.encoder.object_ids_shape,
                dtype=np.int32,
            ),
            "object_scalars": gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.inner_env.encoder.object_scalars_shape,
                dtype=np.float32,
            ),
            "object_mask": gymnasium.spaces.Box(
                low=0,
                high=1,
                shape=(self.inner_env.encoder.object_ids_shape[0],),
                dtype=np.uint8,
            ),
            "event_ids": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=self.inner_env.encoder.event_ids_shape,
                dtype=np.int32,
            ),
            "event_scalars": gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.inner_env.encoder.event_scalars_shape,
                dtype=np.float32,
            ),
            "event_owner": gymnasium.spaces.Box(
                low=0,
                high=self.inner_env.encoder.object_ids_shape[0] - 1,
                shape=(self.inner_env.encoder.event_ids_shape[0],),
                dtype=np.int32,
            ),
            "event_mask": gymnasium.spaces.Box(
                low=0,
                high=1,
                shape=(self.inner_env.encoder.event_ids_shape[0],),
                dtype=np.uint8,
            ),
            "spatial_ids": gymnasium.spaces.Box(
                low=0,
                high=8,
                shape=(self.inner_env.encoder.spatial_scalars_shape[0],),
                dtype=np.int32,
            ),
            "spatial_scalars": gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.inner_env.encoder.spatial_scalars_shape,
                dtype=np.float32,
            ),
            "action_history": gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(ACTION_HISTORY_LEN, ACTION_HISTORY_DIM),
                dtype=np.float32,
            ),
        })
        # SF 2.1.x does not support MultiDiscrete; use Tuple of Discrete instead.
        # Both produce a flat array matching ACTION_HEADS — the step() mapping is identical.
        self.action_space = gymnasium.spaces.Tuple(
            tuple(gymnasium.spaces.Discrete(n) for n in _HEAD_SIZES)
        )

    def _inject_record_cmd(self) -> None:
        """Add ``record <demo_name>`` to pre_map_commands so every episode is recorded."""
        opts = self.inner_env.adapter.reset_options
        base = str(opts.get("pre_map_commands", ""))
        record_name = f"demos/{self._demo_name}"
        opts["pre_map_commands"] = (
            f"{base}\nrecord {record_name}" if base else f"record {record_name}"
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        self._ep_frags = 0
        self._ep_deaths = 0
        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._ep_steps = 0
        self._ep_reward = 0.0
        self._ep_hits = 0
        self._ep_shots = 0
        self._ep_health_pickups = 0.0
        self._ep_armor_pickups = 0.0
        self._ep_weapon_pickups = 0.0
        self._ep_blind_fires = 0
        self._ep_stuck_steps = 0
        self._ep_reward_frag = 0.0
        self._ep_reward_death = 0.0
        self._ep_reward_ehp = 0.0
        self._ep_reward_edp = 0.0
        obs = self.inner_env.reset(seed=seed)
        return obs, {"scenario_id": self.scenario_id}

    def step(self, action: np.ndarray):
        action_dict = multi_discrete_to_heads(action)
        obs, reward, done, info = self.inner_env.step(action_dict)
        info = dict(info)
        info.setdefault("scenario_id", self.scenario_id)
        done_reason: str = str(info.get("done_reason", ""))
        truncated = bool(done_reason == "timeout" or info.get("timed_out", False))
        terminated = bool(done) and not truncated

        # Accumulate per-episode combat stats
        self._ep_frags += int(info.get("frag_delta", 0))
        self._ep_deaths += 1 if done_reason == "player_died" or info.get("frag_loss", 0) > 0 else 0
        self._ep_damage_dealt += float(info.get("damage_dealt", 0))
        self._ep_damage_taken += float(info.get("damage_taken", 0))
        self._ep_steps += 1
        self._ep_reward += float(reward)
        # Skill metrics
        self._ep_hits += int(info.get("hit_count", 0))
        self._ep_shots += int(info.get("shots_fired", 0))
        self._ep_health_pickups += float(info.get("health_gain", 0))
        self._ep_armor_pickups += float(info.get("armor_gain", 0))
        self._ep_weapon_pickups += float(info.get("weapon_pickups", 0))
        self._ep_blind_fires += int(info.get("blind_fire", 0))
        self._ep_stuck_steps += 1 if info.get("stuck") else 0
        # Reward decomposition
        self._ep_reward_frag += float(info.get("reward_frag_bonus", 0))
        self._ep_reward_death += float(info.get("reward_death_penalty", 0))
        self._ep_reward_ehp += float(info.get("reward_ehp_delta", 0))
        self._ep_reward_edp += float(info.get("reward_edp_delta", 0))

        # Preserve the completed episode's demo before reset overwrites it.
        if (terminated or truncated) and self._record_demos and self._demo_path.exists():
            import shutil
            shutil.copy2(self._demo_path, self._demo_last_path)

        # SF picks up episode_extra_stats_* from info on terminal/truncated steps
        if terminated or truncated:
            # --- Combat summary ---
            info["episode_extra_stats_frags"] = self._ep_frags
            info["episode_extra_stats_deaths"] = self._ep_deaths
            info["episode_extra_stats_kd_ratio"] = (
                self._ep_frags / max(self._ep_deaths, 1)
            )
            info["episode_extra_stats_damage_dealt"] = self._ep_damage_dealt
            info["episode_extra_stats_damage_taken"] = self._ep_damage_taken
            info["episode_extra_stats_steps"] = self._ep_steps

            # --- Skill metrics ---
            info["episode_extra_stats_accuracy"] = (
                self._ep_hits / max(self._ep_shots, 1)
            )
            info["episode_extra_stats_hits"] = self._ep_hits
            info["episode_extra_stats_shots_fired"] = self._ep_shots
            info["episode_extra_stats_damage_per_death"] = (
                self._ep_damage_dealt / max(self._ep_deaths, 1)
            )
            info["episode_extra_stats_health_pickups"] = self._ep_health_pickups
            info["episode_extra_stats_armor_pickups"] = self._ep_armor_pickups
            info["episode_extra_stats_weapon_pickups"] = self._ep_weapon_pickups
            info["episode_extra_stats_blind_fire_rate"] = (
                self._ep_blind_fires / max(self._ep_shots, 1)
            )
            info["episode_extra_stats_stuck_rate"] = (
                self._ep_stuck_steps / max(self._ep_steps, 1)
            )

            # --- Reward decomposition ---
            info["episode_extra_stats_reward_total"] = self._ep_reward
            info["episode_extra_stats_reward_frags"] = self._ep_reward_frag
            info["episode_extra_stats_reward_deaths"] = self._ep_reward_death
            info["episode_extra_stats_reward_ehp"] = self._ep_reward_ehp
            info["episode_extra_stats_reward_edp"] = self._ep_reward_edp

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.inner_env.close()

    def render(self) -> None:
        pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_json_arg(cfg: Any, attr: str, expected_type: type) -> Any:
    """Read a JSON-encoded CLI arg from cfg, returning None if absent/empty."""
    raw = getattr(cfg, attr, None) or ""
    if not raw:
        return None
    try:
        value = json.loads(raw)
        if isinstance(value, expected_type):
            return value
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def make_quake_env(
    full_env_name: str,
    cfg: Any,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: str | None = None,
) -> QuakeEnv:
    """Factory function registered with Sample Factory's register_env."""
    del render_mode
    return QuakeEnv(full_env_name, cfg, env_config)
