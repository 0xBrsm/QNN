"""Strict run-dir config loading and flat config assembly helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qnn.utils.io import read_json


def _load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{path} must contain a JSON object at the top level")
    return dict(payload)


def _require_key(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise RuntimeError(f"{context} must define {key}")
    return mapping[key]


def _require_mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _require_key(mapping, key, context)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}.{key} must be a JSON object")
    return dict(value)


def _require_list(mapping: Mapping[str, Any], key: str, context: str) -> list[Any]:
    value = _require_key(mapping, key, context)
    if not isinstance(value, list):
        raise RuntimeError(f"{context}.{key} must be a JSON array")
    return list(value)


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_key(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context}.{key} must be a non-empty string")
    return value


def _require_string_value(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_key(mapping, key, context)
    if not isinstance(value, str):
        raise RuntimeError(f"{context}.{key} must be a string")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = _require_key(mapping, key, context)
    if not isinstance(value, bool):
        raise RuntimeError(f"{context}.{key} must be a boolean")
    return value


def _key_with_prefix_fallback(mapping: Mapping[str, Any], key: str, prefix: str, context: str) -> Any:
    """Try ``{prefix}{key}`` first, then fall back to ``{key}``."""
    prefixed = f"{prefix}{key}"
    if prefixed in mapping:
        return mapping[prefixed]
    return _require_key(mapping, key, context)


def _optional_mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any] | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}.{key} must be a JSON object when present")
    return dict(value)


def _optional_note(scenario: Mapping[str, Any]) -> str:
    note = scenario.get("note")
    if isinstance(note, str):
        return note
    return ""


def _resolve_optional_path(raw_path: str) -> str:
    if not raw_path.strip():
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path)


def _scenario_surface(run_cfg: dict[str, Any]) -> dict[str, Any]:
    scenario = _require_mapping(run_cfg, "scenario", "run config")
    surface: dict[str, Any] = {
        "map_id": _require_string(scenario, "map_id", "scenario.json"),
        "native_args": [str(value) for value in _require_list(scenario, "native_args", "scenario.json")],
        "options": _require_mapping(scenario, "options", "scenario.json"),
        "procgen": _optional_mapping(scenario, "procgen", "scenario.json"),
        "scenario_config_path": "",
    }
    if "scenarios" in scenario:
        scenarios = _require_list(scenario, "scenarios", "scenario.json")
        if not scenarios:
            raise RuntimeError("scenario.json.scenarios must be non-empty when present")
        surface["scenario_config_path"] = str(run_cfg["config_paths"]["scenario"])
    return surface


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load the complete training configuration from a run directory."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        raise RuntimeError(f"run.json not found in run directory: {run_dir}")

    manifest = _load_config(manifest_path)
    config_section = _require_mapping(manifest, "config", "run.json")
    mode = _require_string(manifest, "mode", "run.json")
    runtime_scale = _require_string(manifest, "runtime_scale", "run.json")
    resume = _require_bool(manifest, "resume", "run.json")

    result: dict[str, Any] = {
        "manifest": manifest,
        "run_dir": run_dir,
        "config_paths": {},
        "mode": mode,
        "runtime_scale": runtime_scale,
        "resume": resume,
    }

    if mode == "head_probe":
        # Head probes train memoryless MLPs from frozen shards — no
        # encoder/GRU/pointer model.json, but they do carry a probe.json
        # with the MLP shape + feature list.
        required_keys = ["train", "machine", "probe"]
    else:
        required_keys = ["train", "machine", "model"]
    if mode in {"ppo", "pbt", "optuna", "eval"}:
        required_keys.extend(["scenario", "reward"])
    optional_keys = ["scenario", "reward"]

    for name in required_keys:
        rel_path = _require_string(config_section, name, "run.json.config")
        abs_path = run_dir / rel_path
        if not abs_path.exists():
            raise RuntimeError(
                f"run.json.config.{name} points to {rel_path}, but {abs_path} does not exist"
            )
        result[name] = _load_config(abs_path)
        result["config_paths"][name] = abs_path

    for name in optional_keys:
        if name in result or name not in config_section:
            continue
        rel_path = config_section[name]
        if not isinstance(rel_path, str) or not rel_path.strip():
            continue
        abs_path = run_dir / rel_path
        if abs_path.exists():
            result[name] = _load_config(abs_path)
            result["config_paths"][name] = abs_path

    # checkpoint_path: the starting checkpoint for this run (BC seed for PPO, model for eval).
    ckpt = manifest.get("checkpoint_path", "")
    if not isinstance(ckpt, str):
        raise RuntimeError("run.json.checkpoint_path must be a string")
    result["checkpoint_path"] = _resolve_optional_path(ckpt)

    # eval requires a checkpoint; ppo/pbt/optuna may omit it for random init.
    if mode == "eval" and not result["checkpoint_path"]:
        raise RuntimeError(f"run.json must define a non-empty checkpoint_path when mode is '{mode}'")

    result["output"] = _require_mapping(manifest, "output", "run.json")
    return result


def run_output_dirs(run_cfg: dict[str, Any]) -> dict[str, Path]:
    run_dir = Path(run_cfg["run_dir"])
    output = _require_mapping(run_cfg["manifest"], "output", "run.json")
    checkpoints_dir = run_dir / _require_string(output, "checkpoints", "run.json.output")
    metrics_dir = run_dir / _require_string(output, "metrics", "run.json.output")
    logs_dir = run_dir / _require_string(output, "logs", "run.json.output")
    return {
        "checkpoints": checkpoints_dir,
        "metrics": metrics_dir,
        "logs": logs_dir,
    }


def run_stage_dir(run_cfg: dict[str, Any], stage: str) -> Path:
    outputs = run_output_dirs(run_cfg)
    if stage in {"bc", "ppo"}:
        return outputs["checkpoints"]
    if stage in {"collect", "best"}:
        return outputs["checkpoints"] / stage
    if stage in {"eval", "eval_bc"}:
        return outputs["metrics"] / stage
    if stage == "logs":
        return outputs["logs"]
    raise RuntimeError(f"Unsupported run stage: {stage}")


def run_plan_path(run_cfg: dict[str, Any]) -> Path:
    return Path(run_cfg["run_dir"]) / "logs" / "live_training_plan.json"


def build_run_bc_config(
    run_cfg: dict[str, Any],
    requested_device: str,
) -> dict[str, Any]:
    """Build BC config from a flat run-dir config dict.

    Model arch lives under ``bc_cfg["model"]`` as a ``ModelConfig``
    instance built from ``config/model.json``. Train + machine knobs
    populate the remaining BCConfig fields.
    """
    from qnn.model.network import ModelConfig

    train = _require_mapping(run_cfg, "train", "run config")
    model = _require_mapping(run_cfg, "model", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")

    bc_cfg = dict(train)
    # Legacy no-op key: global length bucketing is unconditional now.
    bc_cfg.pop("length_bucket_window", None)
    bc_cfg["model"] = ModelConfig.from_dict(model)
    bc_cfg["run_id"] = str(_require_mapping(run_cfg, "manifest", "run config").get("run_id", ""))

    checkpoints_dir = run_output_dirs(run_cfg)["checkpoints"]
    bc_cfg["output_dir"] = str(checkpoints_dir)
    bc_cfg["bc_data_dir"] = _require_string(machine, "bc_data_dir", "machine.json")
    bc_cfg["device"] = requested_device
    # batch_size carries the gradient-step sample count for both training
    # paths: per-step frame count for frame-shuffled (non-recurrent), and
    # parallel-lane count for lane-packed (recurrent). The pre-rename
    # microbatch_size knob is gone — fail loudly so old configs surface.
    if "microbatch_size" in machine:
        raise ValueError(
            "machine.json: 'microbatch_size' was renamed — use 'batch_size'. "
            "For recurrent training, batch_size is the parallel lane count "
            "(see lane_packed_batches docstring)."
        )
    bc_cfg["batch_size"] = int(_require_key(machine, "batch_size", "machine.json"))
    bc_cfg["pin_memory"] = bool(_require_key(machine, "pin_memory", "machine.json"))
    bc_cfg["prefetch"] = int(_require_key(machine, "prefetch", "machine.json"))
    bc_cfg["snapshot_interval"] = int(_require_key(machine, "snapshot_interval", "machine.json"))
    bc_cfg["streaming"] = bool(_require_key(machine, "streaming", "machine.json"))
    bc_cfg["dtype"] = str(_require_string(train, "dtype", "train.json"))
    bc_cfg["collection_fingerprint"] = _require_string(
        train, "collection_fingerprint", "train.json"
    )

    return bc_cfg


def build_run_ppo_eval_config(
    run_cfg: dict[str, Any],
    requested_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build PPO and post-train eval configs from a flat run-dir config dict."""
    train = _require_mapping(run_cfg, "train", "run config")
    model = _require_mapping(run_cfg, "model", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")

    surface = _scenario_surface(run_cfg)
    outputs = run_output_dirs(run_cfg)

    ppo_cfg = dict(train)
    ppo_cfg.update(model)
    ppo_cfg.update(surface)
    ppo_cfg["run_id"] = str(_require_mapping(run_cfg, "manifest", "run config").get("run_id", ""))
    ppo_cfg["output_dir"] = str(outputs["checkpoints"])
    ppo_cfg["reward_json_path"] = str(run_cfg["config_paths"]["reward"])
    ppo_cfg["device"] = requested_device
    ppo_cfg["num_workers"] = int(_require_key(machine, "num_workers", "machine.json"))
    ppo_cfg["num_envs_per_worker"] = int(_require_key(machine, "num_envs_per_worker", "machine.json"))
    ppo_cfg["worker_num_splits"] = int(_require_key(machine, "worker_num_splits", "machine.json"))
    ppo_cfg["minibatch_size"] = int(_require_key(machine, "minibatch_size", "machine.json"))
    ppo_cfg["policy_workers_per_policy"] = int(_require_key(machine, "policy_workers_per_policy", "machine.json"))
    ppo_cfg["batched_sampling"] = bool(_require_key(machine, "batched_sampling", "machine.json"))
    ppo_cfg["worker_inference"] = bool(_require_key(machine, "worker_inference", "machine.json"))
    ppo_cfg["worker_inference_device"] = str(_require_key(machine, "worker_inference_device", "machine.json")).lower()
    ppo_cfg["num_envs"] = int(ppo_cfg["num_workers"]) * int(ppo_cfg["num_envs_per_worker"])

    # Validate minibatch alignment
    rollout_batch = int(ppo_cfg["num_envs"]) * int(ppo_cfg.get("rollout_steps", 0))
    mb = int(ppo_cfg["minibatch_size"])
    if rollout_batch > 0 and mb > 0 and rollout_batch % mb != 0:
        import warnings
        warnings.warn(
            f"Minibatch fragmentation: rollout_batch={rollout_batch} "
            f"(num_envs={ppo_cfg['num_envs']} x rollout_steps={ppo_cfg.get('rollout_steps')}) "
            f"is not evenly divisible by minibatch_size={mb}. "
            f"Last minibatch will be undersized ({rollout_batch % mb} vs {mb}). "
            f"Clean sizes for this config: "
            + ", ".join(str(d) for d in sorted(set(
                d for d in range(1, rollout_batch + 1)
                if rollout_batch % d == 0 and 1024 <= d <= rollout_batch
            ))[:8]),
            stacklevel=2,
        )

    eval_cfg = {
        "checkpoint_path": str(outputs["checkpoints"] / "best" / "best_model.pth"),
        "output_dir": str(outputs["metrics"] / "eval"),
        "map_id": str(surface["map_id"]),
        "native_executable": "",
        "native_workdir": _require_string_value(train, "native_workdir", "train.json"),
        "fixed_tick_hz": int(_require_key(train, "fixed_tick_hz", "train.json")),
        "native_env": {},
        "native_args": list(surface["native_args"]),
        "options": dict(surface["options"]),
        "mode": _require_string(train, "mode", "train.json"),
        "seed": int(_require_key(train, "eval_seed", "train.json")),
        "num_episodes": int(_require_key(machine, "eval_num_episodes", "machine.json")),
        "num_envs": int(_require_key(machine, "eval_num_envs", "machine.json")),
        "max_steps_per_episode": int(_require_key(train, "max_steps_per_episode", "train.json")),
        "policy_modes": [str(value) for value in _require_list(train, "eval_policy_modes", "train.json")],
        "start_mode": _require_string(train, "eval_start_mode", "train.json"),
        "holdout_seed_offset": int(_require_key(train, "eval_holdout_seed_offset", "train.json")),
        "sample_seed_offset": int(_require_key(train, "eval_sample_seed_offset", "train.json")),
        "map_features_path": _require_string_value(train, "eval_map_features_path", "train.json"),
        "procgen": surface["procgen"],
        "scenario_config_path": str(surface["scenario_config_path"]),
        "reward_json_path": str(run_cfg["config_paths"]["reward"]),
        "parallel_policy_modes": bool(_require_key(train, "eval_parallel_policy_modes", "train.json")),
        "device": requested_device,
    }
    return ppo_cfg, eval_cfg


def _parse_weapon_ban(raw: Any) -> tuple[int, ...]:
    """Normalize an ``eval_weapon_ban`` value to a tuple of impulses 1..8.

    Accepts a csv string ("2,3"), an int list/tuple, or None/"" (→ ()). Mirrors
    the ONNX export's --weapon-ban parsing so eval applies the same ban spec.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        items = list(raw)
    ban = tuple(int(x) for x in items)
    for imp in ban:
        if not 1 <= imp <= 8:
            raise ValueError(f"eval_weapon_ban: impulse {imp} out of range 1..8")
    return ban


def build_run_eval_config(
    run_cfg: dict[str, Any],
    requested_device: str,
) -> dict[str, Any]:
    """Build standalone eval config from a flat run-dir config dict.

    Reads ``eval_``-prefixed trainer/machine keys (shared with the PPO
    post-train eval path) and falls back to unprefixed keys when the
    prefixed variant is absent.  This lets PPO run dirs work without
    duplicating every key.
    """
    train = _require_mapping(run_cfg, "train", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")
    surface = _scenario_surface(run_cfg)
    outputs = run_output_dirs(run_cfg)

    def _t(key: str) -> Any:
        return _key_with_prefix_fallback(train, key, "eval_", "train.json")

    def _m(key: str) -> Any:
        return _key_with_prefix_fallback(machine, key, "eval_", "machine.json")

    return {
        "checkpoint_path": str(run_cfg["checkpoint_path"]),
        "output_dir": str(outputs["metrics"] / "eval"),
        "map_id": str(surface["map_id"]),
        "native_executable": "",
        "native_workdir": _require_string_value(train, "native_workdir", "train.json"),
        "fixed_tick_hz": int(_require_key(train, "fixed_tick_hz", "train.json")),
        "native_env": {},
        "native_args": list(surface["native_args"]),
        "options": dict(surface["options"]),
        "mode": _require_string(train, "mode", "train.json"),
        "seed": int(_t("seed")),
        "num_episodes": int(_m("num_episodes")),
        "num_envs": int(_m("num_envs")),
        "max_steps_per_episode": int(_require_key(train, "max_steps_per_episode", "train.json")),
        "policy_modes": [str(v) for v in _t("policy_modes")],
        "start_mode": str(_t("start_mode")),
        "holdout_seed_offset": int(_t("holdout_seed_offset")),
        "sample_seed_offset": int(_t("sample_seed_offset")),
        "map_features_path": str(_t("map_features_path")),
        "procgen": surface["procgen"],
        "scenario_config_path": str(surface["scenario_config_path"]),
        "reward_json_path": str(run_cfg["config_paths"]["reward"]),
        "parallel_policy_modes": bool(_t("parallel_policy_modes")),
        "device": requested_device,
        # Optional: aim-prior gain override (0.0 = control arm). Absent →
        # None → the model's baked decode-contract default.
        "look_aim_prior_gain": (
            float(train["eval_look_aim_prior_gain"])
            if "eval_look_aim_prior_gain" in train else None
        ),
        # Optional: per-model weapon ban (impulses 1..8) — the same spec the
        # ONNX export path applies, so eval predicts deployed behavior. Accepts
        # a csv string ("2,3") or a list ([2, 3]); absent → () → no ban.
        "weapon_ban": _parse_weapon_ban(train.get("eval_weapon_ban")),
        # Optional: engine-parity sticky move decode (set both for live
        # parity; absent → legacy per-frame sampling).
        "move_sticky_tau_fb": (
            float(train["eval_move_sticky_tau_fb"])
            if "eval_move_sticky_tau_fb" in train else None
        ),
        "move_sticky_tau_lr": (
            float(train["eval_move_sticky_tau_lr"])
            if "eval_move_sticky_tau_lr" in train else None
        ),
        # Optional: dump per-episode decoded move streams (diagnostics).
        "log_action_streams": bool(train.get("eval_log_action_streams", False)),
        # Optional: semi-Markov hazard decode tables (dict with edges +
        # per-axis fb/lr/ud release probabilities). Requires the sticky taus.
        "move_hazard": (
            dict(train["eval_move_hazard"])
            if isinstance(train.get("eval_move_hazard"), Mapping) else None
        ),
        # Optional: latency-agnostic switch-back suppression (watermark on the
        # abandoned class's softmax prob; see docs/move-head.md). Requires
        # the sticky taus.
        "move_switchback_eps": (
            float(train["eval_move_switchback_eps"])
            if "eval_move_switchback_eps" in train else None
        ),
        # Optional: stop-onset hazard symmetry — from a true stop (both held
        # fb/lr = none) gate presses are suppressed; onsets come from the
        # none-row hazard. Requires eval_move_hazard fb+lr tables.
        "move_stop_onset": bool(train.get("eval_move_stop_onset", False)),
        # Optional: engagement-gated sticky tau — tau=1 (table-only switching) on
        # disengaged (no-target) frames, sticky_tau on engaged frames. Pairs with a
        # non-combat baseline eval_move_hazard table (rc1o move scheme).
        "move_tau_engagement_gated": bool(train.get("eval_move_tau_engagement_gated", False)),
        # Optional: emulate the live client's obs latency — the policy sees
        # obs from N ticks ago while actions land in real time. 0 = bridge
        # semantics (every eval before 2026-06-11).
        "obs_lag_ticks": int(train.get("eval_obs_lag_ticks", 0)),
        # Optional: release-candidate decode regime for Python eval parity
        # with export-time in-graph decode (e.g. "a24rc1").
        "decode_regime": (
            str(train["eval_decode_regime"])
            if "eval_decode_regime" in train else None
        ),
        "look_aim_snap_thresholds_deg": train.get("eval_look_aim_snap_thresholds_deg"),
        "look_aim_snap_scales":         train.get("eval_look_aim_snap_scales"),
    }


def build_run_plan_values(run_cfg: dict[str, Any]) -> dict[str, int]:
    """Return derived plan values from the flat frozen run config."""
    mode = _require_string(run_cfg, "mode", "run config")
    train = _require_mapping(run_cfg, "train", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")

    if mode == "bc":
        return {
            "bc_batch_size": int(_require_key(machine, "batch_size", "machine.json")),
            "num_envs": 0,
            "rollout_steps": 0,
            "total_steps": 0,
            "minibatch_size": 0,
            "eval_episodes": 0,
        }

    if mode in {"ppo", "pbt", "optuna"}:
        num_workers = int(_require_key(machine, "num_workers", "machine.json"))
        num_envs_per_worker = int(_require_key(machine, "num_envs_per_worker", "machine.json"))
        return {
            "bc_batch_size": 0,
            "num_envs": num_workers * num_envs_per_worker,
            "rollout_steps": int(_require_key(train, "rollout_steps", "train.json")),
            "total_steps": int(_require_key(train, "total_steps", "train.json")),
            "minibatch_size": int(_require_key(machine, "minibatch_size", "machine.json")),
            "eval_episodes": int(_require_key(machine, "eval_num_episodes", "machine.json")),
        }

    if mode == "eval":
        return {
            "bc_batch_size": 0,
            "num_envs": int(_key_with_prefix_fallback(machine, "num_envs", "eval_", "machine.json")),
            "rollout_steps": 0,
            "total_steps": 0,
            "minibatch_size": 0,
            "eval_episodes": int(_key_with_prefix_fallback(machine, "num_episodes", "eval_", "machine.json")),
        }

    if mode == "head_probe":
        # head_probe goes through the canonical BC trainer, so its
        # planning surface mirrors BC's: batch_size is a machine knob.
        machine = _require_mapping(run_cfg, "machine", "run config")
        return {
            "bc_batch_size": int(_require_key(machine, "batch_size", "machine.json")),
            "num_envs": 0,
            "rollout_steps": 0,
            "total_steps": 0,
            "minibatch_size": 0,
            "eval_episodes": 0,
        }

    raise RuntimeError(f"Unsupported run mode in run.json: {mode}")


def scenario_note(run_cfg: dict[str, Any]) -> str:
    scenario = run_cfg.get("scenario")
    if not isinstance(scenario, Mapping):
        return ""
    return _optional_note(scenario)
