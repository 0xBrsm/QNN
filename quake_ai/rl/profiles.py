"""Strict run-dir config loading and flat config assembly helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quake_ai.utils.io import read_json


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

    for required_name in ("trainer", "scenario", "reward", "machine", "model"):
        rel_path = _require_string(config_section, required_name, "run.json.config")
        abs_path = run_dir / rel_path
        if not abs_path.exists():
            raise RuntimeError(
                f"run.json.config.{required_name} points to {rel_path}, but {abs_path} does not exist"
            )
        result[required_name] = _load_config(abs_path)
        result["config_paths"][required_name] = abs_path

    # checkpoint_path: the starting checkpoint for this run (BC seed for PPO, model for eval).
    # Accept legacy "seed_checkpoint" key as fallback.
    ckpt = manifest.get("checkpoint_path", "") or manifest.get("seed_checkpoint", "")
    if not isinstance(ckpt, str):
        raise RuntimeError("run.json.checkpoint_path must be a string")
    result["checkpoint_path"] = _resolve_optional_path(ckpt)

    if mode == "eval" and not result["checkpoint_path"]:
        raise RuntimeError("run.json must define a non-empty checkpoint_path when mode is 'eval'")

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
    variant_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build BC config from a flat run-dir config dict."""
    trainer = _require_mapping(run_cfg, "trainer", "run config")
    model = _require_mapping(run_cfg, "model", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")
    scenario = _require_mapping(run_cfg, "scenario", "run config")

    bc_cfg = dict(trainer)
    bc_cfg.update(model)

    checkpoints_dir = run_output_dirs(run_cfg)["checkpoints"]
    collect_dir = checkpoints_dir / "collect"
    bc_cfg["output_dir"] = str(checkpoints_dir)
    bc_cfg["token_ticks_path"] = str(collect_dir / "token_ticks.bin")
    bc_cfg["map_state_path"] = str(collect_dir / "world_map.json")
    bc_cfg["map_states_path"] = str(collect_dir / "map_states.json")
    bc_cfg["metadata_path"] = str(collect_dir / "demo_metadata.ndjson")
    bc_cfg["device"] = requested_device
    bc_cfg["batch_size"] = int(_require_key(machine, "batch_size", "machine.json"))
    bc_cfg["map_id"] = _require_string(scenario, "map_id", "scenario.json")

    if variant_overrides:
        bc_cfg.update(variant_overrides)
    return bc_cfg


def build_run_ppo_eval_config(
    run_cfg: dict[str, Any],
    requested_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build PPO and post-train eval configs from a flat run-dir config dict."""
    trainer = _require_mapping(run_cfg, "trainer", "run config")
    model = _require_mapping(run_cfg, "model", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")

    surface = _scenario_surface(run_cfg)
    outputs = run_output_dirs(run_cfg)

    ppo_cfg = dict(trainer)
    ppo_cfg.update(model)
    ppo_cfg.update(surface)
    ppo_cfg["output_dir"] = str(outputs["checkpoints"])
    ppo_cfg["reward_json_path"] = str(run_cfg["config_paths"]["reward"])
    ppo_cfg["device"] = requested_device
    ppo_cfg["num_workers"] = int(_require_key(machine, "num_workers", "machine.json"))
    ppo_cfg["num_envs_per_worker"] = int(_require_key(machine, "num_envs_per_worker", "machine.json"))
    ppo_cfg["worker_num_splits"] = int(_require_key(machine, "worker_num_splits", "machine.json"))
    ppo_cfg["minibatch_size"] = int(_require_key(machine, "minibatch_size", "machine.json"))
    ppo_cfg["policy_workers_per_policy"] = int(machine.get("policy_workers_per_policy", 1))
    ppo_cfg["num_envs"] = int(ppo_cfg["num_workers"]) * int(ppo_cfg["num_envs_per_worker"])

    eval_cfg = {
        "checkpoint_path": str(outputs["checkpoints"] / "best" / "best_model.pth"),
        "output_dir": str(outputs["metrics"] / "eval"),
        "map_id": str(surface["map_id"]),
        "native_executable": "",
        "native_workdir": _require_string_value(trainer, "native_workdir", "trainer.json"),
        "fixed_tick_hz": int(_require_key(trainer, "fixed_tick_hz", "trainer.json")),
        "native_env": {},
        "native_args": list(surface["native_args"]),
        "options": dict(surface["options"]),
        "mode": _require_string(trainer, "mode", "trainer.json"),
        "seed": int(_require_key(trainer, "eval_seed", "trainer.json")),
        "num_episodes": int(_require_key(machine, "eval_num_episodes", "machine.json")),
        "num_envs": int(_require_key(machine, "eval_num_envs", "machine.json")),
        "max_steps_per_episode": int(_require_key(trainer, "max_steps_per_episode", "trainer.json")),
        "policy_modes": [str(value) for value in _require_list(trainer, "eval_policy_modes", "trainer.json")],
        "start_mode": _require_string(trainer, "eval_start_mode", "trainer.json"),
        "holdout_seed_offset": int(_require_key(trainer, "eval_holdout_seed_offset", "trainer.json")),
        "sample_seed_offset": int(_require_key(trainer, "eval_sample_seed_offset", "trainer.json")),
        "map_features_path": _require_string_value(trainer, "eval_map_features_path", "trainer.json"),
        "procgen": surface["procgen"],
        "scenario_config_path": str(surface["scenario_config_path"]),
        "reward_json_path": str(run_cfg["config_paths"]["reward"]),
        "record_demos": bool(_require_key(trainer, "eval_record_demos", "trainer.json")),
        "parallel_policy_modes": bool(_require_key(trainer, "eval_parallel_policy_modes", "trainer.json")),
        "device": requested_device,
    }
    return ppo_cfg, eval_cfg


def build_run_eval_config(
    run_cfg: dict[str, Any],
    requested_device: str,
) -> dict[str, Any]:
    """Build standalone eval config from a flat run-dir config dict."""
    trainer = _require_mapping(run_cfg, "trainer", "run config")
    machine = _require_mapping(run_cfg, "machine", "run config")
    surface = _scenario_surface(run_cfg)
    outputs = run_output_dirs(run_cfg)

    return {
        "checkpoint_path": str(run_cfg["checkpoint_path"]),
        "output_dir": str(outputs["metrics"] / "eval"),
        "map_id": str(surface["map_id"]),
        "native_executable": "",
        "native_workdir": _require_string_value(trainer, "native_workdir", "trainer.json"),
        "fixed_tick_hz": int(_require_key(trainer, "fixed_tick_hz", "trainer.json")),
        "native_env": {},
        "native_args": list(surface["native_args"]),
        "options": dict(surface["options"]),
        "mode": _require_string(trainer, "mode", "trainer.json"),
        "seed": int(_require_key(trainer, "seed", "trainer.json")),
        "num_episodes": int(_require_key(machine, "num_episodes", "machine.json")),
        "num_envs": int(_require_key(machine, "num_envs", "machine.json")),
        "max_steps_per_episode": int(_require_key(trainer, "max_steps_per_episode", "trainer.json")),
        "policy_modes": [str(value) for value in _require_list(trainer, "policy_modes", "trainer.json")],
        "start_mode": _require_string(trainer, "start_mode", "trainer.json"),
        "holdout_seed_offset": int(_require_key(trainer, "holdout_seed_offset", "trainer.json")),
        "sample_seed_offset": int(_require_key(trainer, "sample_seed_offset", "trainer.json")),
        "map_features_path": _require_string_value(trainer, "map_features_path", "trainer.json"),
        "procgen": surface["procgen"],
        "scenario_config_path": str(surface["scenario_config_path"]),
        "reward_json_path": str(run_cfg["config_paths"]["reward"]),
        "record_demos": bool(_require_key(trainer, "record_demos", "trainer.json")),
        "parallel_policy_modes": bool(_require_key(trainer, "parallel_policy_modes", "trainer.json")),
        "device": requested_device,
    }


def build_run_check_surface(run_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the lightweight env surface needed by check runs."""
    trainer = _require_mapping(run_cfg, "trainer", "run config")
    surface = _scenario_surface(run_cfg)
    return {
        "map_id": str(surface["map_id"]),
        "tick_hz": int(_require_key(trainer, "fixed_tick_hz", "trainer.json")),
        "native_args": list(surface["native_args"]),
        "options": dict(surface["options"]),
    }


def build_run_plan_values(run_cfg: dict[str, Any]) -> dict[str, int]:
    """Return derived plan values from the flat frozen run config."""
    mode = _require_string(run_cfg, "mode", "run config")
    trainer = _require_mapping(run_cfg, "trainer", "run config")
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

    if mode == "ppo":
        num_workers = int(_require_key(machine, "num_workers", "machine.json"))
        num_envs_per_worker = int(_require_key(machine, "num_envs_per_worker", "machine.json"))
        return {
            "bc_batch_size": 0,
            "num_envs": num_workers * num_envs_per_worker,
            "rollout_steps": int(_require_key(trainer, "rollout_steps", "trainer.json")),
            "total_steps": int(_require_key(trainer, "total_steps", "trainer.json")),
            "minibatch_size": int(_require_key(machine, "minibatch_size", "machine.json")),
            "eval_episodes": int(_require_key(machine, "eval_num_episodes", "machine.json")),
        }

    if mode == "eval":
        return {
            "bc_batch_size": 0,
            "num_envs": int(_require_key(machine, "num_envs", "machine.json")),
            "rollout_steps": 0,
            "total_steps": 0,
            "minibatch_size": 0,
            "eval_episodes": int(_require_key(machine, "num_episodes", "machine.json")),
        }

    if mode == "check":
        return {
            "bc_batch_size": 0,
            "num_envs": 0,
            "rollout_steps": 0,
            "total_steps": 0,
            "minibatch_size": 0,
            "eval_episodes": 0,
        }

    raise RuntimeError(f"Unsupported run mode in run.json: {mode}")


def scenario_note(run_cfg: dict[str, Any]) -> str:
    scenario = _require_mapping(run_cfg, "scenario", "run config")
    return _optional_note(scenario)
