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


def _valid_policy_modes(modes: list[str]) -> list[str]:
    """Fail at config-build time, not after the training stage has already
    run: the eval driver only knows these modes, and a bad name would
    otherwise surface hours later at post-train eval."""
    known = {"greedy", "sampled"}
    bad = [m for m in modes if m not in known]
    if bad:
        raise RuntimeError(
            f"train.json eval_policy_modes {bad} unsupported — use {sorted(known)}")
    return modes


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

    # eval needs a model; native PPO is the FINE-TUNING stage and always
    # starts from a BC/RL seed (random-init RL died with Sample Factory).
    if mode in ("eval", "ppo") and not result["checkpoint_path"]:
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
    bc_cfg["resident_epoch_pack"] = bool(machine.get("resident_epoch_pack", False))
    bc_cfg["reset_aligned_lanes"] = bool(machine.get("reset_aligned_lanes", False))
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
    # Native single-process trainer topology: lanes are parallel engine
    # subprocesses in ONE process; the recurrent minibatch unit is a lane
    # subset over the full rollout window (see agents/plans/ppo-rebuild.md).
    ppo_cfg["num_lanes"] = int(_require_key(machine, "num_lanes", "machine.json"))
    ppo_cfg["minibatch_lanes"] = int(_require_key(machine, "minibatch_lanes", "machine.json"))
    # Collect-time forward device. Default cpu: the d64 act forward is
    # ~4× faster on CPU than ROCm eager (per-op launch overhead), the
    # worker-inference lesson carried into the native trainer — one
    # in-process replica, synced at each iteration boundary (exactly
    # on-policy; collection is synchronous). "" = same device as learner.
    ppo_cfg["collect_device"] = str(machine.get("collect_device", "cpu"))
    ppo_cfg["collector_num_threads"] = int(
        machine.get("collector_num_threads", 0)
    )
    if ppo_cfg["collector_num_threads"] < 0:
        raise ValueError("machine.json: collector_num_threads must be >= 0")
    env_backend = str(machine.get("env_backend", "process"))
    if env_backend not in {"process", "arena_grid"}:
        raise ValueError("machine.json: env_backend must be 'process' or 'arena_grid'")
    ppo_cfg["env_backend"] = env_backend
    ppo_cfg["matches_per_server"] = int(machine.get("matches_per_server", 8))
    ppo_cfg["seat_mode"] = str(machine.get("seat_mode", "bot"))
    ppo_cfg["arena_server_binary"] = str(
        machine.get("arena_server_binary", "assets/bin/ppo_arena_server")
    )
    ppo_cfg["arena_client_binary"] = str(
        machine.get("arena_client_binary", "assets/bin/ppo_arena_client")
    )
    ppo_cfg["arena_map_id"] = str(machine.get("arena_map_id", "qnn_arena8"))
    ppo_cfg["arena_base_port"] = int(machine.get("arena_base_port", 28000))
    ppo_cfg["arena_bot_skill"] = int(
        machine.get("arena_bot_skill", surface["options"].get("skill", 3))
    )
    if env_backend == "arena_grid":
        from qnn.ppo.arena import ArenaTopology

        # Build once during frozen-config validation so invalid client-slot or
        # lane divisibility choices fail before any engine process launches.
        topology = ArenaTopology.build(
            num_lanes=int(ppo_cfg["num_lanes"]),
            matches_per_server=int(ppo_cfg["matches_per_server"]),
            seat_mode=str(ppo_cfg["seat_mode"]),
        )
        first_port = int(ppo_cfg["arena_base_port"])
        last_port = first_port + topology.server_count - 1
        if first_port < 1024 or last_port > 65535:
            raise ValueError(
                "machine.json: arena_base_port plus the arena server count "
                "must remain in [1024, 65535]"
            )
        if not 0 <= int(ppo_cfg["arena_bot_skill"]) <= 3:
            raise ValueError("machine.json: arena_bot_skill must be in [0, 3]")
    ppo_cfg["num_envs"] = int(ppo_cfg["num_lanes"])  # legacy alias (reports/metrics)

    lanes, mb_lanes = int(ppo_cfg["num_lanes"]), int(ppo_cfg["minibatch_lanes"])
    if lanes > 0 and mb_lanes > 0 and lanes % mb_lanes != 0:
        import warnings
        warnings.warn(
            f"Minibatch fragmentation: num_lanes={lanes} is not evenly "
            f"divisible by minibatch_lanes={mb_lanes}; the last minibatch "
            f"will be undersized ({lanes % mb_lanes} lanes). Clean values: "
            + ", ".join(str(d) for d in range(1, lanes + 1) if lanes % d == 0),
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
        "policy_modes": _valid_policy_modes(
            [str(value) for value in _require_list(train, "eval_policy_modes", "train.json")]),
        "start_mode": _require_string(train, "eval_start_mode", "train.json"),
        "holdout_seed_offset": int(_require_key(train, "eval_holdout_seed_offset", "train.json")),
        "sample_seed_offset": int(_require_key(train, "eval_sample_seed_offset", "train.json")),
        "map_features_path": _require_string_value(train, "eval_map_features_path", "train.json"),
        "procgen": surface["procgen"],
        "scenario_config_path": str(surface["scenario_config_path"]),
        "reward_json_path": str(run_cfg["config_paths"]["reward"]),
        "parallel_policy_modes": bool(_require_key(train, "eval_parallel_policy_modes", "train.json")),
        "device": requested_device,
        "env_backend": str(machine.get("eval_env_backend", "process")),
        "arena_server_binary": str(machine.get("arena_server_binary", "assets/bin/ppo_arena_server")),
        "arena_client_binary": str(machine.get("arena_client_binary", "assets/bin/ppo_arena_client")),
        "arena_map_id": str(machine.get("arena_map_id", "qnn_arena8")),
        "arena_base_port": int(machine.get("eval_arena_base_port", 28900)),
        "arena_bot_skill": int(machine.get("arena_bot_skill", 3)),
        "arena_matches_per_server": int(machine.get("matches_per_server", 8)),
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


def _parse_per_env_decode_overrides(
    raw: Any,
) -> tuple[dict[str, float], ...] | None:
    """Normalize an ``eval_per_env_decode_overrides`` value into the tuple of
    per-lane decode dicts ``EvalConfig.per_env_decode_overrides`` expects.

    On disk this is a JSON list (one entry per ENV SLOT, aligned with the
    scenario order in scenario.json), each a mapping of a supported decode-config
    key (qnn.model.policy._PER_ROW_DECODE_KEYS) to that lane's scalar. It is the
    serialized form of the aim-grid batched widener: many (swept-value, scenario)
    cells packed into one ≤64-lane batched eval. None/absent → the scalar path
    (back-compat). Values are coerced to float so JSON ints ride through.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "eval_per_env_decode_overrides must be a list of {key: value} dicts")
    out: list[dict[str, float]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"eval_per_env_decode_overrides[{i}] must be a dict, got {type(item)}")
        out.append({str(k): float(v) for k, v in item.items()})
    return tuple(out)


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
        "env_backend": str(machine.get("eval_env_backend", "process")),
        "arena_server_binary": str(machine.get("arena_server_binary", "assets/bin/ppo_arena_server")),
        "arena_client_binary": str(machine.get("arena_client_binary", "assets/bin/ppo_arena_client")),
        "arena_map_id": str(machine.get("arena_map_id", "qnn_arena8")),
        "arena_base_port": int(machine.get("eval_arena_base_port", 28900)),
        "arena_bot_skill": int(machine.get("arena_bot_skill", 3)),
        "arena_matches_per_server": int(machine.get("matches_per_server", 8)),
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
        # Optional: dump per-episode decoded move streams (diagnostics).
        # NOTE: the a24 sticky-move eval keys (eval_move_sticky_tau_*,
        # eval_move_hazard, eval_move_switchback_eps, eval_move_stop_onset,
        # eval_move_tau_engagement_gated) were RETIRED with the a24 arch; old
        # train.json files still carrying them are simply not read here.
        "log_action_streams": bool(train.get("eval_log_action_streams", False)),
        # Optional: dump per-episode raw entity obs streams for the ACQUISITION
        # (Fitts-throughput) axis (acq_streams_<mode>.npz). See
        # EvalConfig.log_acq_streams.
        "log_acq_streams": bool(train.get("eval_log_acq_streams", False)),
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
        # Optional: opt-in batched-forward decode — stack all active envs into
        # one model.act(B=N) per macro-step so the CPU saturates instead of
        # idling in the per-env ping-pong. NOT bit-identical to the default B=1
        # path; only for evals robust to float-level batching differences (e.g.
        # the aim grid's intercept-hbw). See EvalConfig.batched_forward.
        "batched_forward": bool(train.get("eval_batched_forward", False)),
        # Optional: PER-LANE decode overrides — one {key: value} dict per env
        # slot (aligned with scenario order), so a single ≤64-lane batched eval
        # runs many (swept-value, scenario) cells instead of one cold subprocess
        # per swept value (the aim-grid widener). Requires batched_forward.
        # Absent → None → the model/decode scalar for every lane. See
        # EvalConfig.per_env_decode_overrides.
        "per_env_decode_overrides": _parse_per_env_decode_overrides(
            train.get("eval_per_env_decode_overrides")),
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

    if mode in {"ppo", "optuna"}:
        num_lanes = int(_require_key(machine, "num_lanes", "machine.json"))
        return {
            "bc_batch_size": 0,
            "num_envs": num_lanes,
            "rollout_steps": int(_require_key(train, "rollout_steps", "train.json")),
            "total_steps": int(_require_key(train, "total_env_steps", "train.json")),
            # Rows per PPO minibatch = lane subset × the full window.
            "minibatch_size": int(_require_key(machine, "minibatch_lanes", "machine.json"))
            * int(_require_key(train, "rollout_steps", "train.json")),
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
