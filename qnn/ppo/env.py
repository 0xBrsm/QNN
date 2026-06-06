"""Gymnasium wrapper around NativeWorldEnv for PPO training."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import gymnasium
except ImportError as exc:
    raise ImportError("gymnasium is required: pip install gymnasium>=0.29.0") from exc

from qnn.actions import ACTION_HEADS, ActionLabels
from qnn.env.world import NativeWorldEnv
from qnn.run.metrics import EpisodeStatAccumulator
from qnn.schema import OBS_SCHEMA
from qnn.vocab import (
    ENTITY_VOCAB_SIZE, ACTION_VOCAB_SIZE, MAX_ENTITY_EVENTS,
)
from mapgen.pool import PROCGEN_SENTINEL

_MOVE_DIM = ACTION_HEADS["move"]
_LOOK_DIM = ACTION_HEADS["look"]
_DISCRETE_HEAD_ORDER = [
    "attack",
    "weapon",
]

_HEAD_NOOP_VALUES: Dict[str, object] = {
    "move": [0.0, 0.0, 0.0],
    "look": [0.0, 0.0, 0.0],
    "attack": 0,
    "weapon": 0,
}


def _parse_disabled_heads(head_loss_weights_json: str) -> frozenset[str]:
    """Return the set of head names whose weight is 0.0 (treated as disabled)."""
    raw = (head_loss_weights_json or "").strip()
    if not raw:
        return frozenset()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return frozenset()
    return frozenset(h for h, w in parsed.items() if float(w) == 0.0)


def tuple_action_to_heads(action) -> Dict[str, object]:
    """Convert an SF Tuple action to the canonical action dict.

    Handles two formats:
    - Non-batched: tuple/list of sub-arrays (Box(2), Box(2), Discrete, ...)
    - Batched (transposed): tuple of per-head values for one env,
      e.g. (tensor(2), tensor(2), tensor(1), ...) or a single tensor/array.
    """
    import torch as _torch

    # Unwrap single-element list/slice from SequentialVectorizeWrapper
    if isinstance(action, (list, tuple)) and len(action) == 1:
        action = action[0]

    # If it's a single tensor/ndarray, it may be the full flat action
    if isinstance(action, (_torch.Tensor, np.ndarray)):
        a = action.cpu().numpy() if isinstance(action, _torch.Tensor) else np.asarray(action)
        flat = a.reshape(-1).astype(np.float32)
    elif isinstance(action, (list, tuple)):
        parts = []
        for a in action:
            if isinstance(a, _torch.Tensor):
                a = a.cpu().numpy()
            parts.append(np.asarray(a, dtype=np.float32).reshape(-1))
        flat = np.concatenate(parts)
    else:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)

    expected = _MOVE_DIM + _LOOK_DIM + len(_DISCRETE_HEAD_ORDER)
    if flat.size != expected:
        raise ValueError(f"Expected {expected} action values, got {flat.size}")
    payload: Dict[str, object] = {
        "move": flat[0:_MOVE_DIM].astype(np.float32, copy=False).tolist(),
        "look": flat[_MOVE_DIM:_MOVE_DIM + _LOOK_DIM].astype(np.float32, copy=False).tolist(),
    }
    discrete_offset = _MOVE_DIM + _LOOK_DIM
    for head_index, head in enumerate(_DISCRETE_HEAD_ORDER):
        value = int(round(float(flat[discrete_offset + head_index])))
        if head == "weapon":
            # PPO action space is Discrete(8): class indices 0..7.
            # The engine consumes a Quake impulse byte 1..8; convert
            # at this boundary so ActionLabels.weapon and everything
            # downstream carries the impulse byte directly.
            value += 1
        payload[head] = value
    return ActionLabels.from_dict(payload).to_dict()


def heads_to_tuple_action(heads: Dict[str, object]) -> np.ndarray:
    """Convert a canonical action dict to the flat Tuple array expected by SF.

    Inverse of tuple_action_to_heads: labels.weapon is the engine impulse
    byte 1..8, SF Tuple wants the class index 0..7 in its Discrete idx.
    """
    labels = ActionLabels.from_dict(heads)
    return np.asarray(
        [
            float(labels.move[0]),
            float(labels.move[1]),
            float(labels.move[2]),
            float(labels.look[0]),
            float(labels.look[1]),
            float(labels.look[2]),
            int(labels.attack),
            int(labels.weapon) - 1,  # impulse 1..8 → class 0..7
        ],
        dtype=np.float32,
    )


_OBS_SPACE_SPEC: dict[str, tuple[np.dtype, float, float]] = {
    # (dtype, low, high) — float32 fields default to (-inf, inf)
    "self_weapon_id":            (np.dtype(np.int32), 0, ENTITY_VOCAB_SIZE - 1),
    "self_armor_type_id":        (np.dtype(np.int32), 0, ENTITY_VOCAB_SIZE - 1),
    "self_state_powerup_ids":    (np.dtype(np.int32), 0, ENTITY_VOCAB_SIZE - 1),
    "self_arsenal_powerup_ids":  (np.dtype(np.int32), 0, ENTITY_VOCAB_SIZE - 1),
    "self_motion_powerup_ids":   (np.dtype(np.int32), 0, ENTITY_VOCAB_SIZE - 1),
    "self_movement_id":          (np.dtype(np.int32), 0, 4),
    "entity_types":          (np.dtype(np.int32), -1, 3),
    "entity_ids":            (np.dtype(np.int32), 0, 255),
    "entity_event_actions":  (np.dtype(np.int32), 0, ACTION_VOCAB_SIZE - 1),
    "entity_event_sources":  (np.dtype(np.int32), 0, ENTITY_VOCAB_SIZE - 1),
    "entity_event_counts":   (np.dtype(np.uint8), 0, MAX_ENTITY_EVENTS),
}


def _build_observation_space() -> gymnasium.spaces.Dict:
    """Build observation space from canonical OBS_SCHEMA."""
    spaces = {}
    for key, shape in OBS_SCHEMA.items():
        spec = _OBS_SPACE_SPEC.get(key)
        if spec is not None:
            dtype, low, high = spec
        else:
            dtype, low, high = np.float32, -np.inf, np.inf
        spaces[key] = gymnasium.spaces.Box(
            low=low, high=high, shape=shape, dtype=dtype,
        )
    return gymnasium.spaces.Dict(spaces)


class QuakeEnv(gymnasium.Env):
    """Gymnasium env wrapping NativeWorldEnv for Sample Factory APPO.

    Observation space: Dict of self/object/event/spatial token tensors from the inner encoder.
    Action space: Tuple(Box(3), Box(3), Discrete...) for move/look vectors,
    and discrete fire/switch heads.

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
        native_args = _parse_json_arg(cfg, "quake_native_args_json", list)
        if native_args is None:
            raise RuntimeError("QuakeEnv requires quake_native_args_json")
        options = _parse_json_arg(cfg, "quake_options_json", dict)
        if options is None:
            raise RuntimeError("QuakeEnv requires quake_options_json")
        procgen_base = _parse_json_arg(cfg, "quake_procgen_json", dict)

        # Scenario rotation: if quake_scenario_config_json is provided it overrides map_id
        map_id = str(cfg.quake_map_id)
        scenario_cfg_json = getattr(cfg, "quake_scenario_config_json", None) or ""
        scenario: Dict[str, Any] = {}
        if scenario_cfg_json:
            scenarios: list[Dict[str, Any]] = json.loads(scenario_cfg_json)
            scenario = scenarios[env_id % len(scenarios)]
            map_id = str(scenario["map_id"])
            self.scenario_id = str(scenario["scenario_id"])
            if "native_args" in scenario:
                native_args = list(scenario["native_args"])
            if "options" in scenario:
                # Merge scenario options over base options (preserves deathmatch, etc.)
                base = dict(options) if options else {}
                base.update(scenario["options"])
                options = base
        else:
            self.scenario_id = map_id

        basedir = str(cfg.quake_basedir)
        workdir = str(cfg.quake_native_workdir)
        if not basedir.strip():
            raise RuntimeError("QuakeEnv requires quake_basedir")

        # Procgen: generate maps inline on each reset() — no background threads.
        procgen_cfg: dict | None = None
        if map_id == PROCGEN_SENTINEL:
            procgen_opts = scenario.get("procgen", procgen_base)
            if not isinstance(procgen_opts, dict):
                raise RuntimeError(
                    "Procgen training requires an explicit procgen config with arena_size, rooms, and cleanup_generated_maps"
                )
            procgen_cfg = {
                "maps_dir": str(Path(basedir) / "id1" / "maps"),
                "arena_size": int(procgen_opts["arena_size"]),
                "rooms": int(procgen_opts["rooms"]),
                "cleanup_generated_maps": bool(procgen_opts["cleanup_generated_maps"]),
            }

        # Reward weights from the run's config/reward.json.
        reward_json_path = str(getattr(cfg, "reward_json_path", "")).strip()
        if not reward_json_path:
            raise RuntimeError("QuakeEnv requires reward_json_path")
        from qnn.env.reward import RewardWeights
        reward_weights = RewardWeights.from_json(reward_json_path)

        self.inner_env = NativeWorldEnv(
            executable=str(cfg.quake_executable),
            map_id=map_id,
            max_steps=int(cfg.quake_max_steps_per_episode),
            fixed_tick_hz=int(cfg.quake_fixed_tick_hz),
            reward_weights=reward_weights,
            mode=str(cfg.quake_mode),
            seed=int(cfg.quake_seed) + env_id,
            env={"QUAKE_BASEDIR": basedir},
            native_args=native_args,
            options=options,
            workdir=workdir or None,
            procgen=procgen_cfg,
        )

        # Heads with head_loss_weight==0.0 are "disabled": their sampled
        # actions are overridden to no-ops here so the environment never sees
        # the random sample.  Gradient isolation for those heads is handled
        # upstream in TupleActionDistribution (see train.py).
        self._disabled_heads: frozenset[str] = _parse_disabled_heads(
            getattr(cfg, "head_loss_weights", "")
        )

        # Episode-level accumulators for SF custom metrics
        self._episode_stats = EpisodeStatAccumulator()

        self.observation_space = _build_observation_space()
        self.action_space = gymnasium.spaces.Tuple(
            (
                gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(_MOVE_DIM,), dtype=np.float32),
                gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(_LOOK_DIM,), dtype=np.float32),
                gymnasium.spaces.Discrete(ACTION_HEADS["attack"]),
                gymnasium.spaces.Discrete(ACTION_HEADS["weapon"]),
            )
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        self._episode_stats = EpisodeStatAccumulator()
        obs = self.inner_env.reset(seed=seed)
        return obs, {"scenario_id": self.scenario_id}

    def step(self, action):
        action_dict = tuple_action_to_heads(action)
        for head in self._disabled_heads:
            action_dict[head] = _HEAD_NOOP_VALUES[head]
        # Normalize look to the unit sphere before it hits the engine.
        # qnn_input.c maps look to view-angle deltas via atan2, which is
        # scale-invariant but ill-defined near the origin; the tanh clamp
        # at ±1 can also distort large-noise samples.  Normalizing keeps
        # the semantic "direction" intact, matching BC labels which are
        # unit vectors.  Skip when look is disabled (noop=[0,0,0]).
        if "look" not in self._disabled_heads:
            look = action_dict.get("look")
            if look is not None:
                look_arr = np.asarray(look, dtype=np.float32)
                norm = float(np.linalg.norm(look_arr))
                if norm > 1e-6:
                    action_dict["look"] = (look_arr / norm).tolist()
        obs, reward, done, info = self.inner_env.step(action_dict)
        info = dict(info)
        info.setdefault("scenario_id", self.scenario_id)
        done_reason: str = str(info.get("done_reason", ""))
        truncated = bool(done_reason == "timeout" or info.get("timed_out", False))
        terminated = bool(done) and not truncated

        self._episode_stats.add_step(
            reward=float(reward),
            info=info,
            terminal=terminated or truncated,
        )

        # Sample Factory expects custom episodic stats under info["episode_extra_stats"].
        if terminated or truncated:
            episode_stats = self._episode_stats.as_dict()
            info["episode_extra_stats"] = episode_stats
            for key, value in episode_stats.items():
                info[f"episode_extra_stats_{key}"] = value

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
