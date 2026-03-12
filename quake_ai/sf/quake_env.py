"""Gymnasium wrapper around NativeWorldEnv for Sample Factory."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import gymnasium
except ImportError as exc:
    raise ImportError("gymnasium is required: pip install gymnasium>=0.29.0") from exc

from quake_ai.actions import ACTION_HEADS
from quake_ai.rl.environment import NativeWorldEnv
from quake_ai.model.observation import ACTION_HISTORY_DIM, ACTION_HISTORY_LEN, SELF_SCALAR_DIM

# Deterministic head ordering (Python 3.7+ dict preserves insertion order)
_HEAD_ORDER: list[str] = list(ACTION_HEADS.keys())
_HEAD_SIZES: list[int] = list(ACTION_HEADS.values())  # [3, 3, 25, 25, 2, 2, 9]


def multi_discrete_to_heads(action: np.ndarray) -> Dict[str, int]:
    """Convert a SF MultiDiscrete action array to an action_heads dict."""
    return {head: int(action[i]) for i, head in enumerate(_HEAD_ORDER)}


def heads_to_multi_discrete(heads: Dict[str, int]) -> np.ndarray:
    """Convert an action_heads dict to a MultiDiscrete array."""
    return np.array([heads[head] for head in _HEAD_ORDER], dtype=np.int64)


class QuakeEnv(gymnasium.Env):
    """Gymnasium env wrapping NativeWorldEnv for Sample Factory APPO.

    Observation space: Dict of self/object/event/spatial token tensors from the inner encoder.
    Action space: Tuple(Discrete(3), Discrete(3), Discrete(25), Discrete(25), Discrete(2), Discrete(2), Discrete(9))
                  matching ACTION_HEADS.  SF 2.1.x requires Tuple[Discrete] rather than MultiDiscrete.

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
        map_id = str(getattr(cfg, "quake_map_id", "dm6"))
        scenario_cfg_json = getattr(cfg, "quake_scenario_config_json", "") or ""
        if scenario_cfg_json:
            scenarios: list[Dict[str, Any]] = json.loads(scenario_cfg_json)
            scenario = scenarios[env_id % len(scenarios)]
            map_id = str(scenario.get("map_id", map_id))
            self.scenario_id = str(scenario.get("scenario_id", map_id))
            if "native_args" in scenario:
                native_args = list(scenario["native_args"])
            if "options" in scenario:
                options = dict(scenario["options"])
        else:
            self.scenario_id = map_id

        basedir = getattr(cfg, "quake_basedir", "") or ""
        workdir = getattr(cfg, "quake_native_workdir", "") or ""

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
        )

        self.observation_space = gymnasium.spaces.Dict({
            "self_scalars": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(SELF_SCALAR_DIM,), dtype=np.float32,
            ),
            "self_weapon_id": gymnasium.spaces.Box(
                low=0, high=8, shape=(), dtype=np.int32,
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
        # Both produce a flat array of 7 integers — the step() mapping is identical.
        self.action_space = gymnasium.spaces.Tuple(
            tuple(gymnasium.spaces.Discrete(n) for n in _HEAD_SIZES)
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
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
