"""ROCm-oriented live world-v2 training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from engine.native_bridge import NativeEngineProcess
from quake_ai.data.collector import collect_from_demos
from quake_ai.evaluation import EvalConfig, run_evaluation
from quake_ai.training_bc import BCConfig, run_behavior_cloning
from quake_ai.training_rl import PPOConfig, run_ppo
from quake_ai.actions import LOOK_NEUTRAL_LABEL
from quake_ai.utils.device import describe_torch_runtime
from quake_ai.utils.io import load_config, read_json, write_json

GIB = 1024**3
REPORT_STAGES = ("collect", "bc", "eval_bc", "ppo", "eval")


def _looks_like_quake_basedir(path: Path) -> bool:
    id1_dir = path / "id1"
    if not id1_dir.is_dir():
        return False
    pak_names = ("PAK0.PAK", "PAK1.PAK", "pak0.pak", "pak1.pak")
    return any((id1_dir / name).exists() for name in pak_names)


def _power_of_two_floor(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


@dataclass(frozen=True, slots=True)
class LiveProfile:
    name: str
    bc_config: str
    ppo_config: str
    eval_config: str
    default_demo_dir: str
    auto_demo_dirs: tuple[str, ...]
    collect_out: str
    plan_path: str
    runtime_scale: str = "corpus"
    profile_group: str = ""
    profile_note: str = ""
    scenario_id: str = ""
    retained_role: str = ""
    comparison_profiles: tuple[str, ...] = ()
    bc_overrides: Dict[str, Any] = field(default_factory=dict)
    ppo_overrides: Dict[str, Any] = field(default_factory=dict)
    eval_overrides: Dict[str, Any] = field(default_factory=dict)


_DEFAULT_BOT_NATIVE_ARGS = ["-game", "frikbotnex"]
_DEFAULT_BOT_NATIVE_OPTIONS = {
    "maxplayers": 2,
    "skill": 0,
    "deathmatch": 1,
    "coop": 0,
    "teamplay": 0,
    "fraglimit": 0,
    "timelimit": 0,
    "samelevel": 1,
    "pre_map_commands": "",
    "post_map_commands": "impulse 100",
}


def _deep_merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_config(existing, value)
            continue
        if isinstance(value, list):
            merged[key] = list(value)
            continue
        merged[key] = value
    return merged


def _build_scenario_profile(
    *,
    name: str,
    runtime_scale: str,
    bc_config: str,
    ppo_config: str,
    eval_config: str,
    default_demo_dir: str,
    auto_demo_dirs: tuple[str, ...],
    scenario: Mapping[str, Any],
    comparison_profiles: tuple[str, ...],
) -> LiveProfile:
    scenario_id = str(scenario["scenario_id"])
    output_root = str(scenario[f"{runtime_scale}_output_root"])
    map_id = str(scenario["map_id"])
    native_options = _deep_merge_config(_DEFAULT_BOT_NATIVE_OPTIONS, scenario.get("native_options", {}))
    use_gru = bool(scenario.get("use_gru", True))
    gru_hidden = int(scenario.get("gru_hidden", 64))
    bc_use_gru = bool(scenario.get("bc_use_gru", False))
    bc_gru_hidden = int(scenario.get("bc_gru_hidden", 0 if not bc_use_gru else gru_hidden))

    return LiveProfile(
        name=name,
        bc_config=bc_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        default_demo_dir=default_demo_dir,
        auto_demo_dirs=auto_demo_dirs,
        collect_out=f"{output_root}/collect",
        plan_path=f"{output_root}/live_training_plan.json",
        runtime_scale="verify" if runtime_scale == "verify" else "corpus",
        profile_group="combat_bot_ladder",
        profile_note=str(scenario.get("skill_note", "")).strip(),
        scenario_id=scenario_id,
        retained_role=str(scenario.get("retained_role", "")).strip(),
        comparison_profiles=comparison_profiles,
        bc_overrides={
            "map_id": map_id,
            "output_dir": f"{output_root}/bc",
            "use_gru": bc_use_gru,
            "gru_hidden": bc_gru_hidden,
        },
        ppo_overrides={
            "map_id": map_id,
            "output_dir": f"{output_root}/ppo",
            "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
            "native_options": native_options,
            "use_gru": use_gru,
            "gru_hidden": gru_hidden,
        },
        eval_overrides={
            "map_id": map_id,
            "checkpoint_path": f"{output_root}/ppo/ppo_model.npz",
            "output_dir": f"{output_root}/eval",
            "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
            "native_options": native_options,
        },
    )


def _load_bot_ladder_profiles() -> dict[str, LiveProfile]:
    scenarios_payload = load_config("configs/combat_bot_scenarios.json")
    scenarios = scenarios_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise RuntimeError("configs/combat_bot_scenarios.json must define a scenarios list")

    profiles: dict[str, LiveProfile] = {}
    for raw_scenario in scenarios:
        if not isinstance(raw_scenario, Mapping):
            continue
        scenario = dict(raw_scenario)
        base_name = f"combat-bot-{str(scenario['scenario_id']).strip()}"
        verify_name = f"{base_name}-verify"
        live_name = base_name
        comparison = tuple(str(value) for value in scenario.get("comparison_profiles", ("combat-verify",)) if str(value).strip())
        profiles[verify_name] = _build_scenario_profile(
            name=verify_name,
            runtime_scale="verify",
            bc_config="configs/bc_combat_bot_verify.yaml",
            ppo_config="configs/ppo_combat_bot_verify.yaml",
            eval_config="configs/eval_combat_bot_verify.yaml",
            default_demo_dir="tests/demo_data",
            auto_demo_dirs=("tests/demo_data",),
            scenario=scenario,
            comparison_profiles=comparison,
        )
        profiles[live_name] = _build_scenario_profile(
            name=live_name,
            runtime_scale="live",
            bc_config="configs/bc_combat_bot_live.yaml",
            ppo_config="configs/ppo_combat_bot_live.yaml",
            eval_config="configs/eval_combat_bot_live.yaml",
            default_demo_dir="../artifacts/corpus/netquake/materialized_competitive",
            auto_demo_dirs=("../artifacts/corpus/netquake/materialized_competitive", "tests/demo_data"),
            scenario=scenario,
            comparison_profiles=comparison,
        )
    return profiles


PROFILES: dict[str, LiveProfile] = {
    "verify": LiveProfile(
        name="verify",
        bc_config="configs/bc_e1m1_corpus_world_verify.yaml",
        ppo_config="configs/ppo_e1m1_corpus_world_verify.yaml",
        eval_config="configs/eval_e1m1_corpus_world_verify.yaml",
        default_demo_dir="tests/demo_data",
        auto_demo_dirs=("tests/demo_data",),
        collect_out="../artifacts/runs/e1m1_corpus_world_verify/collect",
        plan_path="../artifacts/runs/e1m1_corpus_world_verify/live_training_plan.json",
        runtime_scale="verify",
    ),
    "corpus": LiveProfile(
        name="corpus",
        bc_config="configs/bc_e1m1_corpus_world.yaml",
        ppo_config="configs/ppo_e1m1_corpus_world.yaml",
        eval_config="configs/eval_e1m1_corpus_world.yaml",
        default_demo_dir="../artifacts/runs/e1m1_corpus_world/demos",
        auto_demo_dirs=("../artifacts/runs/e1m1_corpus_world/demos", "../artifacts/runs/e1m1_corpus/demos"),
        collect_out="../artifacts/runs/e1m1_corpus_world/collect",
        plan_path="../artifacts/runs/e1m1_corpus_world/live_training_plan.json",
    ),
    "combat-verify": LiveProfile(
        name="combat-verify",
        bc_config="configs/bc_combat_bootstrap_verify.yaml",
        ppo_config="configs/ppo_campaign_combat_verify.yaml",
        eval_config="configs/eval_campaign_combat_verify.yaml",
        default_demo_dir="tests/demo_data",
        auto_demo_dirs=("tests/demo_data",),
        collect_out="../artifacts/runs/campaign_combat_verify/collect",
        plan_path="../artifacts/runs/campaign_combat_verify/live_training_plan.json",
        runtime_scale="verify",
    ),
    "combat": LiveProfile(
        name="combat",
        bc_config="configs/bc_combat_bootstrap_live.yaml",
        ppo_config="configs/ppo_campaign_combat_live.yaml",
        eval_config="configs/eval_campaign_combat_live.yaml",
        default_demo_dir="../artifacts/corpus/netquake/materialized_competitive",
        auto_demo_dirs=("../artifacts/corpus/netquake/materialized_competitive", "tests/demo_data"),
        collect_out="../artifacts/runs/campaign_combat_live/collect",
        plan_path="../artifacts/runs/campaign_combat_live/live_training_plan.json",
    ),
    "combat-bot-verify": LiveProfile(
        name="combat-bot-verify",
        bc_config="configs/bc_combat_bot_verify.yaml",
        ppo_config="configs/ppo_combat_bot_verify.yaml",
        eval_config="configs/eval_combat_bot_verify.yaml",
        default_demo_dir="tests/demo_data",
        auto_demo_dirs=("tests/demo_data",),
        collect_out="../artifacts/runs/competitive_bot_combat_verify/collect",
        plan_path="../artifacts/runs/competitive_bot_combat_verify/live_training_plan.json",
        runtime_scale="verify",
        profile_group="combat_bot_alias",
        profile_note="Alias preserved for compatibility. Prefer one of the named combat-bot-* ladder scenarios.",
    ),
    "combat-bot": LiveProfile(
        name="combat-bot",
        bc_config="configs/bc_combat_bot_live.yaml",
        ppo_config="configs/ppo_combat_bot_live.yaml",
        eval_config="configs/eval_combat_bot_live.yaml",
        default_demo_dir="../artifacts/corpus/netquake/materialized_competitive",
        auto_demo_dirs=("../artifacts/corpus/netquake/materialized_competitive", "tests/demo_data"),
        collect_out="../artifacts/runs/competitive_bot_combat_live/collect",
        plan_path="../artifacts/runs/competitive_bot_combat_live/live_training_plan.json",
        profile_group="combat_bot_alias",
        profile_note="Alias preserved for compatibility. Prefer one of the named combat-bot-* ladder scenarios.",
    ),
}
PROFILES.update(_load_bot_ladder_profiles())


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    requested_device: str
    resolved_device: str
    backend: str
    cpu_count: int
    cpu_affinity_count: int
    gpu_memory_bytes: int
    bc_batch_size: int
    num_envs: int
    rollout_steps: int
    total_steps: int
    minibatch_size: int
    eval_episodes: int

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gpu_memory_gib"] = round(self.gpu_memory_bytes / GIB, 2) if self.gpu_memory_bytes else 0.0
        return payload


def _runtime_plan(profile: LiveProfile, runtime: Mapping[str, Any]) -> RuntimePlan:
    cpu_count = max(int(runtime.get("cpu_affinity_count") or runtime.get("cpu_count") or 1), 1)
    gpu_memory_bytes = 0
    devices = runtime.get("devices")
    if isinstance(devices, list):
        gpu_memory_bytes = max(
            (int(device.get("total_memory", 0)) for device in devices if isinstance(device, Mapping)),
            default=0,
        )

    reserve = 2 if cpu_count > 8 else 1
    env_cap = 8 if profile.runtime_scale == "verify" else 30
    num_envs = max(2, min(cpu_count - reserve, env_cap))

    if profile.runtime_scale == "verify":
        rollout_steps = 64
        total_steps = max(4_096, num_envs * rollout_steps * 8)
        eval_episodes = 32
    else:
        # Corpus live runs need longer PPO windows to expose long-horizon behavior,
        # but evaluation stays bounded so retained runs complete in practical time.
        rollout_steps = 256 if gpu_memory_bytes >= 12 * GIB else 128
        total_steps = max(262_144, num_envs * rollout_steps * 64)
        eval_episodes = 32

    if gpu_memory_bytes >= 24 * GIB:
        bc_batch_size = 8_192 if profile.runtime_scale == "corpus" else 4_096
        minibatch_cap = 2_048
    elif gpu_memory_bytes >= 12 * GIB:
        bc_batch_size = 4_096 if profile.runtime_scale == "corpus" else 2_048
        minibatch_cap = 1_024
    elif gpu_memory_bytes >= 8 * GIB:
        bc_batch_size = 2_048 if profile.runtime_scale == "corpus" else 1_024
        minibatch_cap = 512
    else:
        bc_batch_size = 1_024 if profile.runtime_scale == "corpus" else 512
        minibatch_cap = 256

    minibatch_size = max(32, min(num_envs * rollout_steps // 2, minibatch_cap))
    minibatch_size = max(32, _power_of_two_floor(minibatch_size))

    return RuntimePlan(
        requested_device=str(runtime.get("requested_device", "auto")),
        resolved_device=str(runtime.get("resolved_device", "cpu")),
        backend=str(runtime.get("backend", "cpu")),
        cpu_count=max(int(runtime.get("cpu_count") or cpu_count), 1),
        cpu_affinity_count=cpu_count,
        gpu_memory_bytes=gpu_memory_bytes,
        bc_batch_size=bc_batch_size,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        total_steps=total_steps,
        minibatch_size=minibatch_size,
        eval_episodes=eval_episodes,
    )


def build_runtime_plan(profile_name: str, requested_device: str) -> tuple[LiveProfile, dict[str, Any], RuntimePlan]:
    profile = PROFILES[profile_name]
    runtime = describe_torch_runtime(requested_device)
    error = runtime.get("error")
    if error:
        raise RuntimeError(f"Accelerator runtime is unavailable: {error}")
    plan = _runtime_plan(profile, runtime)
    return profile, runtime, plan


def _resolve_asset_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_basedir = os.environ.get("QUAKE_BASEDIR", "").strip()
    if env_basedir:
        candidates.append(Path(env_basedir))
    candidates.extend([Path("/assets"), Path("assets"), Path("../assets")])

    for candidate in candidates:
        if _looks_like_quake_basedir(candidate):
            return candidate

    raise RuntimeError("Quake assets not available; mount a basedir with id1/PAK0.PAK under /assets or set QUAKE_BASEDIR")


def _required_gamedir(native_args: Sequence[str] | None) -> str | None:
    if not native_args:
        return None
    for index, value in enumerate(native_args):
        if str(value) != "-game":
            continue
        if index + 1 >= len(native_args):
            break
        gamedir = str(native_args[index + 1]).strip()
        if gamedir:
            return gamedir
    return None


def _validate_native_mod_assets(asset_root: Path, native_args: Sequence[str] | None) -> None:
    gamedir = _required_gamedir(native_args)
    if not gamedir:
        return
    mod_root = asset_root / gamedir
    if not mod_root.is_dir():
        raise RuntimeError(
            f"Configured native_args require {mod_root}, but it does not exist. "
            "Install the mod first with src/scripts/train-container.sh install-frikbotnex."
        )
    if not any((mod_root / name).exists() for name in ("progs.dat", "qwprogs.dat")):
        raise RuntimeError(
            f"Configured native_args require a compiled gamedir under {mod_root}, but no progs.dat or qwprogs.dat was found."
        )


def _resolve_demo_dir(profile: LiveProfile, explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else [Path(path) for path in profile.auto_demo_dirs]
    for candidate in candidates:
        if candidate.is_dir() and any(path.suffix.lower() == ".dem" for path in candidate.iterdir() if path.is_file()):
            return candidate
    if explicit:
        raise RuntimeError(f"Demo directory does not contain .dem files: {explicit}")
    raise RuntimeError(
        f"No demo directory available for the {profile.name} profile; set --demo-dir or materialize demos under one of: "
        + ", ".join(profile.auto_demo_dirs)
    )


def _iter_config_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_iter_config_strings(child))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for child in value:
            strings.extend(_iter_config_strings(child))
        return strings
    return []


def _references_stage_dir(config: Mapping[str, Any], stage_dir: str | Path) -> bool:
    target = Path(stage_dir)
    for candidate in _iter_config_strings(config):
        if not candidate.strip():
            continue
        candidate_path = Path(candidate)
        if candidate_path == target or target in candidate_path.parents:
            return True
    return False


def _all_action_requires_collect(profile: LiveProfile, *configs: Mapping[str, Any]) -> bool:
    return any(_references_stage_dir(config, profile.collect_out) for config in configs)


def _load_config_with_runtime(profile: LiveProfile, plan: RuntimePlan, requested_device: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bc_cfg = _deep_merge_config(load_config(profile.bc_config), profile.bc_overrides)
    ppo_cfg = _deep_merge_config(load_config(profile.ppo_config), profile.ppo_overrides)
    eval_cfg = _deep_merge_config(load_config(profile.eval_config), profile.eval_overrides)

    bc_cfg["device"] = requested_device
    ppo_cfg["device"] = requested_device
    eval_cfg["device"] = requested_device

    bc_cfg["batch_size"] = max(int(bc_cfg.get("batch_size", 1)), plan.bc_batch_size)
    ppo_cfg["num_envs"] = max(int(ppo_cfg.get("num_envs", 1)), plan.num_envs)
    ppo_cfg["rollout_steps"] = max(int(ppo_cfg.get("rollout_steps", 1)), plan.rollout_steps)
    ppo_cfg["total_steps"] = max(int(ppo_cfg.get("total_steps", 1)), plan.total_steps)
    ppo_cfg["minibatch_size"] = max(int(ppo_cfg.get("minibatch_size", 1)), plan.minibatch_size)
    eval_cfg["num_episodes"] = max(int(eval_cfg.get("num_episodes", 1)), plan.eval_episodes)
    eval_cfg["num_envs"] = max(int(eval_cfg.get("num_envs", 1)), min(plan.num_envs, int(eval_cfg["num_episodes"])))

    return bc_cfg, ppo_cfg, eval_cfg


def _write_plan(profile: LiveProfile, runtime: Mapping[str, Any], plan: RuntimePlan, demo_dir: Path, asset_root: Path) -> Path:
    target = Path(profile.plan_path)
    write_json(
        target,
        {
            "profile": profile.name,
            "profile_group": profile.profile_group,
            "profile_note": profile.profile_note,
            "scenario_id": profile.scenario_id,
            "retained_role": profile.retained_role,
            "comparison_profiles": list(profile.comparison_profiles),
            "demo_dir": str(demo_dir),
            "asset_root": str(asset_root),
            "runtime": dict(runtime),
            "plan": plan.to_dict(),
        },
    )
    return target


def _profile_output_root(profile: LiveProfile) -> Path:
    return Path(profile.plan_path).parent


def _stage_output_dir(profile: LiveProfile, stage: str) -> Path:
    if stage == "collect":
        return Path(profile.collect_out)
    return _profile_output_root(profile) / stage


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix:
        return checkpoint.with_suffix(".json")
    return checkpoint.with_name(f"{checkpoint.name}.json")


def _checkpoint_obs_dim(checkpoint_path: str | Path | None) -> int | None:
    if not checkpoint_path:
        return None
    metadata = _safe_read_json(_checkpoint_metadata_path(checkpoint_path))
    if not metadata:
        return None
    obs_dim = metadata.get("obs_dim")
    if obs_dim is None:
        return None
    try:
        return int(obs_dim)
    except (TypeError, ValueError):
        return None


def _resolve_ppo_init_checkpoint(ppo_cfg: Mapping[str, Any], bc_cfg: Mapping[str, Any]) -> tuple[str | None, str | None]:
    configured = str(ppo_cfg.get("init_ckpt", "")).strip() or None
    output_dir = str(bc_cfg.get("output_dir", "")).strip()
    if not output_dir:
        return configured, None

    local_bc_ckpt = Path(output_dir) / "bc_best_model.npz"
    if not local_bc_ckpt.exists():
        return configured, None

    local_bc_value = str(local_bc_ckpt)
    if configured == local_bc_value:
        return configured, None

    configured_obs_dim = _checkpoint_obs_dim(configured)
    local_bc_obs_dim = _checkpoint_obs_dim(local_bc_ckpt)
    if configured and configured_obs_dim is not None and local_bc_obs_dim is not None and configured_obs_dim != local_bc_obs_dim:
        return (
            local_bc_value,
            f"Switched PPO init_ckpt from {configured} (obs_dim={configured_obs_dim}) to {local_bc_value} (obs_dim={local_bc_obs_dim}).",
        )
    if configured and not Path(configured).exists():
        return local_bc_value, f"Configured PPO init_ckpt {configured} is missing; using {local_bc_value}."
    if not configured:
        return local_bc_value, f"No PPO init_ckpt configured; using {local_bc_value}."
    return configured, None


def _collect_existing_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(candidate for candidate in path.iterdir() if candidate.is_file())


def _mtime_window_seconds(paths: list[Path]) -> float | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    mtimes = [path.stat().st_mtime for path in existing]
    return float(max(mtimes) - min(mtimes))


def _first_device_summary(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    devices = runtime.get("devices")
    if not isinstance(devices, list) or not devices:
        return {}
    first = devices[0]
    if not isinstance(first, Mapping):
        return {}
    return {str(key): value for key, value in first.items()}


def _device_label(runtime: Mapping[str, Any]) -> str:
    device = _first_device_summary(runtime)
    if not device:
        return str(runtime.get("resolved_device", "unknown"))
    name = str(device.get("name", runtime.get("resolved_device", "unknown")))
    total_memory = int(device.get("total_memory", 0))
    if total_memory <= 0:
        return name
    return f"{name} ({round(total_memory / GIB, 2)} GiB)"


def _load_existing_runtime_context(profile: LiveProfile) -> tuple[Dict[str, Any], Dict[str, Any], Path]:
    plan_path = Path(profile.plan_path)
    payload = _safe_read_json(plan_path) or {}
    runtime = payload.get("runtime", {})
    plan = payload.get("plan", {})
    return (
        dict(runtime) if isinstance(runtime, Mapping) else {},
        dict(plan) if isinstance(plan, Mapping) else {},
        plan_path,
    )


def _stage_report(profile: LiveProfile, stage: str) -> Dict[str, Any]:
    stage_dir = _stage_output_dir(profile, stage)
    files = _collect_existing_files(stage_dir)

    report: Dict[str, Any] = {
        "stage": stage,
        "output_dir": str(stage_dir),
        "status": "present" if files else "missing",
        "files": [str(path) for path in files],
    }

    if stage == "collect":
        manifest = _safe_read_json(stage_dir / "collect_manifest.json")
        if manifest is not None:
            report["manifest"] = manifest
        return report

    if stage.startswith("eval"):
        manifest_name = "eval_manifest.json"
        summary_name = "eval_summary.json"
    else:
        manifest_name = f"{stage}_manifest.json"
        summary_name = f"{stage}_summary.json"
    manifest = _safe_read_json(stage_dir / manifest_name)
    summary = _safe_read_json(stage_dir / summary_name)
    if manifest is not None:
        report["manifest"] = manifest
    if summary is not None:
        report["summary"] = summary
    if stage.startswith("eval"):
        model_card = _safe_read_json(stage_dir / "model_card.json")
        if model_card is not None:
            report["model_card"] = model_card
    return report


def _metric(report: Mapping[str, Any], *path: str) -> float | None:
    current: Any = report
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _eval_metric(report: Mapping[str, Any], mode: str, key: str) -> float | None:
    return (
        _metric(report, "summary", "modes", mode, key)
        or _metric(report, "manifest", "metrics", "modes", mode, key)
        or _metric(report, "model_card", "evaluation", "modes", mode, key)
    )


def _stage_summary_metric(report: Mapping[str, Any], stage: str, key: str) -> float | None:
    if stage.startswith("eval"):
        return _eval_metric(report, "greedy", key) or _metric(report, "summary", key) or _metric(report, "manifest", "metrics", key)
    return _metric(report, "summary", key) or _metric(report, "manifest", "metrics", key)


def _comparison_metric_payload(current: float | None, baseline: float | None) -> Dict[str, float] | None:
    if current is None or baseline is None:
        return None
    return {
        "current": float(current),
        "baseline": float(baseline),
        "delta": float(current - baseline),
    }


def _profile_comparison(current_stage_reports: Mapping[str, Mapping[str, Any]], baseline_profile: LiveProfile) -> Dict[str, Any]:
    baseline_reports = {stage: _stage_report(baseline_profile, stage) for stage in REPORT_STAGES}
    metrics: Dict[str, Dict[str, float]] = {}
    for stage, key in (
        ("ppo", "completion_rate"),
        ("ppo", "death_rate"),
        ("ppo", "damage_dealt_mean"),
        ("ppo", "hit_count_mean"),
        ("ppo", "shots_fired_mean"),
        ("eval", "completion_rate"),
        ("eval", "death_rate"),
        ("eval", "mean_episode_return"),
        ("eval", "damage_dealt_mean"),
        ("eval", "hit_count_mean"),
        ("eval", "shots_fired_mean"),
    ):
        payload = _comparison_metric_payload(
            _stage_summary_metric(current_stage_reports.get(stage, {}), stage, key),
            _stage_summary_metric(baseline_reports.get(stage, {}), stage, key),
        )
        if payload is not None:
            metrics[f"{stage}_{key}"] = payload
    return {
        "profile": baseline_profile.name,
        "profile_note": baseline_profile.profile_note,
        "stage_status": {stage: baseline_reports[stage]["status"] for stage in REPORT_STAGES},
        "metrics": metrics,
    }


def _intra_profile_baseline_comparison(stage_reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Dict[str, float]] = {}
    for key in ("completion_rate", "death_rate", "mean_episode_return", "damage_dealt_mean", "hit_count_mean", "shots_fired_mean"):
        payload = _comparison_metric_payload(
            _stage_summary_metric(stage_reports.get("eval", {}), "eval", key),
            _stage_summary_metric(stage_reports.get("eval_bc", {}), "eval_bc", key),
        )
        if payload is not None:
            metrics[key] = payload
    return {
        "baseline": "eval_bc",
        "metrics": metrics,
    }


def _build_operational_note(
    *,
    profile: LiveProfile,
    action: str,
    runtime: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage_reports: Mapping[str, Mapping[str, Any]],
    stage_timings: Mapping[str, float],
) -> Dict[str, Any]:
    all_files: list[Path] = []
    for stage in REPORT_STAGES:
        all_files.extend(_collect_existing_files(_stage_output_dir(profile, stage)))

    elapsed_source = "measured" if stage_timings else "artifact_mtime_window"
    elapsed_seconds = sum(stage_timings.values()) if stage_timings else _mtime_window_seconds(all_files)

    worker_count = int(plan.get("num_envs", 0) or 0)
    rollout_steps = int(plan.get("rollout_steps", 0) or 0)
    bc_batch_size = int(plan.get("bc_batch_size", 0) or 0)
    eval_episodes = int(plan.get("eval_episodes", 0) or 0)

    utilization_parts = [f"backend={runtime.get('backend', 'unknown')}", f"device={_device_label(runtime)}"]
    if worker_count > 0:
        utilization_parts.append(f"ppo_workers={worker_count}")
    if rollout_steps > 0:
        utilization_parts.append(f"rollout_steps={rollout_steps}")
    if bc_batch_size > 0:
        utilization_parts.append(f"bc_batch_size={bc_batch_size}")
    if eval_episodes > 0:
        utilization_parts.append(f"eval_episodes={eval_episodes}")
    utilization_parts.append("direct accelerator utilization sampling not captured")

    ppo_report = stage_reports.get("ppo", {})
    eval_report = stage_reports.get("eval", {})
    eval_bc_report = stage_reports.get("eval_bc", {})
    bc_report = stage_reports.get("bc", {})
    baseline_compare = _intra_profile_baseline_comparison(stage_reports)

    instability_notes: list[str] = []
    if (ppo_report.get("status") == "present") and (Path(str(ppo_report.get("output_dir", ""))) / "ppo_model.npz").exists():
        instability_notes.append("PPO completed and wrote a checkpoint without a recorded worker crash.")
    if _metric(ppo_report, "summary", "completion_rate") == 0.0:
        instability_notes.append("Bounded PPO completed, but completion_rate stayed at 0.0 in this retained run.")
    if _metric(eval_report, "manifest", "metrics", "modes", "greedy", "stuck_rate") and _metric(
        eval_report, "manifest", "metrics", "modes", "greedy", "stuck_rate"
    ) >= 0.9:
        instability_notes.append("Evaluation reported a high greedy stuck_rate; this looks like policy quality, not a worker crash.")
    bc_greedy_completion = _metric(eval_bc_report, "manifest", "metrics", "modes", "greedy", "completion_rate")
    ppo_greedy_completion = _metric(eval_report, "manifest", "metrics", "modes", "greedy", "completion_rate")
    if bc_greedy_completion is not None and ppo_greedy_completion is not None:
        delta = ppo_greedy_completion - bc_greedy_completion
        if delta > 0.0:
            instability_notes.append(f"PPO improved greedy completion over the BC checkpoint by {delta:.3f}.")
        elif delta < 0.0:
            instability_notes.append(f"PPO regressed greedy completion relative to the BC checkpoint by {abs(delta):.3f}.")
        else:
            instability_notes.append("PPO matched the BC checkpoint on greedy completion in this retained run.")
    damage_delta = baseline_compare["metrics"].get("damage_dealt_mean")
    if isinstance(damage_delta, Mapping):
        instability_notes.append(
            f"Greedy evaluation damage_dealt_mean delta versus eval_bc: {float(damage_delta.get('delta', 0.0)):.3f}."
        )
    if not instability_notes:
        instability_notes.append("No instability was recorded in the retained artifacts.")

    note = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "profile_group": profile.profile_group,
        "profile_note": profile.profile_note,
        "scenario_id": profile.scenario_id,
        "retained_role": profile.retained_role,
        "action": action,
        "worker_count": worker_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": elapsed_source,
        "runtime_backend": str(runtime.get("backend", "unknown")),
        "device": _device_label(runtime),
        "rough_utilization": "; ".join(utilization_parts),
        "instability_observations": instability_notes,
        "determinism_drift": "This run did not record in-process drift metrics; explicit drift coverage lives in the asset-gated parity and churn tests.",
        "metrics": {
            "bc_test_accuracy": _metric(bc_report, "summary", "test_accuracy") or _metric(bc_report, "manifest", "metrics", "test_accuracy"),
            "eval_bc_greedy_completion_rate": _metric(eval_bc_report, "manifest", "metrics", "modes", "greedy", "completion_rate"),
            "eval_bc_greedy_stuck_rate": _metric(eval_bc_report, "manifest", "metrics", "modes", "greedy", "stuck_rate"),
            "eval_bc_sampled_completion_rate": _metric(eval_bc_report, "manifest", "metrics", "modes", "sampled", "completion_rate"),
            "eval_bc_sampled_stuck_rate": _metric(eval_bc_report, "manifest", "metrics", "modes", "sampled", "stuck_rate"),
            "ppo_steps_done": _metric(ppo_report, "summary", "steps_done") or _metric(ppo_report, "manifest", "metrics", "steps_done"),
            "ppo_completion_rate": _metric(ppo_report, "summary", "completion_rate")
            or _metric(ppo_report, "manifest", "metrics", "completion_rate"),
            "ppo_death_rate": _metric(ppo_report, "summary", "death_rate") or _metric(ppo_report, "manifest", "metrics", "death_rate"),
            "ppo_frag_delta_mean": _metric(ppo_report, "summary", "frag_delta_mean")
            or _metric(ppo_report, "manifest", "metrics", "frag_delta_mean"),
            "ppo_monster_kill_delta_mean": _metric(ppo_report, "summary", "monster_kill_delta_mean")
            or _metric(ppo_report, "manifest", "metrics", "monster_kill_delta_mean"),
            "ppo_episodes_completed": _metric(ppo_report, "summary", "episodes_completed")
            or _metric(ppo_report, "manifest", "metrics", "episodes_completed"),
            "eval_greedy_completion_rate": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "completion_rate"),
            "eval_greedy_stuck_rate": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "stuck_rate"),
            "eval_greedy_death_rate": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "death_rate"),
            "eval_greedy_frag_delta_mean": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "frag_delta_mean"),
            "eval_greedy_monster_kill_delta_mean": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "monster_kill_delta_mean"),
            "eval_sampled_completion_rate": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "completion_rate"),
            "eval_sampled_stuck_rate": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "stuck_rate"),
            "eval_sampled_death_rate": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "death_rate"),
            "eval_sampled_frag_delta_mean": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "frag_delta_mean"),
            "eval_sampled_monster_kill_delta_mean": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "monster_kill_delta_mean"),
        },
    }
    return note


def _format_operational_note(note: Mapping[str, Any]) -> str:
    elapsed_seconds = note.get("elapsed_seconds")
    elapsed_text = "unknown"
    if isinstance(elapsed_seconds, (int, float)):
        elapsed_text = f"{elapsed_seconds:.2f}s"

    lines = [
        "# Live Training Operational Note",
        "",
        f"- Profile: {note.get('profile', 'unknown')}",
        f"- Scenario: {note.get('scenario_id', 'n/a')}",
        f"- Note: {note.get('profile_note', '') or 'n/a'}",
        f"- Action: {note.get('action', 'unknown')}",
        f"- Worker count: {note.get('worker_count', 0)}",
        f"- Elapsed time: {elapsed_text} ({note.get('elapsed_source', 'unknown')})",
        f"- Runtime: {note.get('runtime_backend', 'unknown')} on {note.get('device', 'unknown')}",
        f"- Rough utilization: {note.get('rough_utilization', 'unknown')}",
        "- Stability observations:",
    ]
    for item in note.get("instability_observations", []):
        lines.append(f"  - {item}")
    lines.append(f"- Determinism drift: {note.get('determinism_drift', 'unknown')}")

    metrics = note.get("metrics", {})
    if isinstance(metrics, Mapping):
        rendered = [f"{key}={value}" for key, value in metrics.items() if value is not None]
        if rendered:
            lines.append(f"- Metrics: {', '.join(rendered)}")

    return "\n".join(lines) + "\n"


def _write_run_report(
    *,
    profile: LiveProfile,
    action: str,
    runtime: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    results: Mapping[str, Any],
    stage_timings: Mapping[str, float],
) -> Dict[str, Any]:
    output_root = _profile_output_root(profile)
    stage_reports = {stage: _stage_report(profile, stage) for stage in REPORT_STAGES}
    comparison_reports = {
        baseline_name: _profile_comparison(stage_reports, PROFILES[baseline_name])
        for baseline_name in profile.comparison_profiles
        if baseline_name in PROFILES
    }
    baseline_compare = _intra_profile_baseline_comparison(stage_reports)
    note = _build_operational_note(
        profile=profile,
        action=action,
        runtime=runtime,
        plan=plan,
        stage_reports=stage_reports,
        stage_timings=stage_timings,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "action": action,
        "plan_path": str(plan_path),
        "runtime": dict(runtime),
        "plan": dict(plan),
        "worker_regression_baseline": "artifacts/runs/e1m1_corpus_world/",
        "results": dict(results),
        "stage_timings_seconds": {str(key): float(value) for key, value in stage_timings.items()},
        "stages": stage_reports,
        "baseline_comparison": baseline_compare,
        "comparison_profiles": comparison_reports,
        "operational_note": note,
    }

    report_path = output_root / "live_run_report.json"
    note_json_path = output_root / "operational_note.json"
    note_md_path = output_root / "operational_note.md"
    write_json(report_path, report)
    write_json(note_json_path, note)
    note_md_path.write_text(_format_operational_note(note), encoding="utf-8")

    return {
        "report": report,
        "report_path": str(report_path),
        "operational_note_json_path": str(note_json_path),
        "operational_note_md_path": str(note_md_path),
    }


def _ensure_worker(worker_binary: Path, rebuild: bool) -> Path:
    if worker_binary.exists() and not rebuild:
        return worker_binary
    build_script = Path("engine/build/build_quake_worker.sh")
    subprocess.run(["bash", str(build_script), str(worker_binary)], check=True)
    return worker_binary


def _run_live_check(
    worker_binary: Path,
    asset_root: Path,
    map_id: str,
    tick_hz: int,
    *,
    native_args: list[str] | None = None,
    native_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = {"QUAKE_BASEDIR": str(asset_root)}
    with NativeEngineProcess(
        executable=worker_binary,
        map_id=map_id,
        fixed_tick_hz=tick_hz,
        env=env,
        extra_args=native_args,
    ) as proc:
        hello = proc.start()
        reset = proc.reset(seed=7, options=native_options)
        step = proc.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
    result = {
        "worker_binary": str(worker_binary),
        "asset_root": str(asset_root),
        "hello_server": hello["server"],
        "capabilities": list(hello.get("capabilities", [])),
        "native_args": list(native_args or []),
        "reset_info": dict(reset.get("info", {})),
        "reset_tick": int(reset["world_tick"]["tick"]),
        "step_tick": int(step["world_tick"]["tick"]),
        "step_done": bool(step.get("done", False)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_live_pipeline(
    *,
    profile_name: str,
    action: str,
    device: str,
    demo_dir: str | None,
    asset_root: str | None,
    worker_binary: str,
    rebuild_worker: bool,
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    if action == "report":
        runtime, plan_dict, plan_path = _load_existing_runtime_context(profile)
        report_artifacts = _write_run_report(
            profile=profile,
            action=action,
            runtime=runtime,
            plan=plan_dict,
            plan_path=plan_path,
            results={"profile": profile.name, "source": "existing_artifacts"},
            stage_timings={},
        )
        report = dict(report_artifacts["report"])
        report["report_path"] = report_artifacts["report_path"]
        report["operational_note_json_path"] = report_artifacts["operational_note_json_path"]
        report["operational_note_md_path"] = report_artifacts["operational_note_md_path"]
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    profile, runtime, plan = build_runtime_plan(profile_name, device)
    resolved_demo_dir = _resolve_demo_dir(profile, demo_dir)
    resolved_asset_root = _resolve_asset_root(asset_root) if action in {"check", "collect", "ppo", "eval", "eval-bc", "all"} else Path(
        asset_root or os.environ.get("QUAKE_BASEDIR", "/assets")
    )
    worker_path = Path(worker_binary)
    if action in {"check", "ppo", "eval", "eval-bc", "all"} or rebuild_worker:
        worker_path = _ensure_worker(worker_path, rebuild=rebuild_worker)
    bc_cfg, ppo_cfg, eval_cfg = _load_config_with_runtime(profile, plan, device)
    if action in {"check", "ppo", "eval", "eval-bc", "all"}:
        _validate_native_mod_assets(resolved_asset_root, ppo_cfg.get("native_args") if isinstance(ppo_cfg.get("native_args"), list) else None)
        _validate_native_mod_assets(resolved_asset_root, eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None)
    plan_path = _write_plan(profile, runtime, plan, resolved_demo_dir, resolved_asset_root)
    stage_timings: dict[str, float] = {}

    results: dict[str, Any] = {
        "profile": profile.name,
        "plan_path": str(plan_path),
        "runtime": runtime,
        "plan": plan.to_dict(),
        "worker_binary": str(worker_path),
        "asset_root": str(resolved_asset_root),
        "demo_dir": str(resolved_demo_dir),
    }

    if action == "plan":
        print(json.dumps(results, indent=2, sort_keys=True))
        return results

    if action == "check":
        results["check"] = _run_live_check(
            worker_binary=worker_path,
            asset_root=resolved_asset_root,
            map_id=str(ppo_cfg.get("map_id", "E1M1")),
            tick_hz=int(ppo_cfg.get("fixed_tick_hz", 20)),
            native_args=[str(value) for value in ppo_cfg.get("native_args", [])] if isinstance(ppo_cfg.get("native_args"), list) else None,
            native_options=dict(ppo_cfg.get("native_options", {})) if isinstance(ppo_cfg.get("native_options"), Mapping) else None,
        )
        return results

    if action in {"collect", "all"}:
        collect_required = action == "collect" or _all_action_requires_collect(profile, bc_cfg, ppo_cfg, eval_cfg)
        if not collect_required:
            results["collect"] = {
                "skipped": True,
                "reason": f"No downstream stage references {profile.collect_out}; using retained inputs instead.",
            }
        else:
            started = time.monotonic()
            collect_artifacts = collect_from_demos(
                map_id=str(bc_cfg["map_id"]),
                demo_dir=resolved_demo_dir,
                out_dir=profile.collect_out,
                map_path=resolved_asset_root,
            )
            manifest_path = Path(profile.collect_out) / "collect_manifest.json"
            write_json(manifest_path, collect_artifacts)
            results["collect"] = collect_artifacts
            stage_timings["collect"] = time.monotonic() - started

    if action in {"bc", "all"}:
        started = time.monotonic()
        results["bc"] = run_behavior_cloning(BCConfig(**bc_cfg))
        stage_timings["bc"] = time.monotonic() - started

    if action in {"eval-bc", "all"}:
        started = time.monotonic()
        eval_bc_cfg = dict(eval_cfg)
        eval_bc_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_bc_cfg["native_executable"] = str(worker_path)
        eval_bc_cfg["checkpoint_path"] = str(Path(bc_cfg["output_dir"]) / "bc_best_model.npz")
        eval_bc_cfg["output_dir"] = str(_profile_output_root(profile) / "eval_bc")
        results["eval_bc"] = run_evaluation(EvalConfig(**eval_bc_cfg))
        stage_timings["eval_bc"] = time.monotonic() - started

    if action in {"ppo", "all"}:
        selected_init_ckpt, init_ckpt_note = _resolve_ppo_init_checkpoint(ppo_cfg, bc_cfg)
        if selected_init_ckpt:
            ppo_cfg["init_ckpt"] = selected_init_ckpt
            results["ppo_init_ckpt"] = selected_init_ckpt
        if init_ckpt_note:
            results["ppo_init_ckpt_note"] = init_ckpt_note
        started = time.monotonic()
        ppo_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        ppo_cfg["native_executable"] = str(worker_path)
        results["ppo"] = run_ppo(PPOConfig(**ppo_cfg))
        stage_timings["ppo"] = time.monotonic() - started

    if action in {"eval", "all"}:
        started = time.monotonic()
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

    report_artifacts = _write_run_report(
        profile=profile,
        action=action,
        runtime=runtime,
        plan=plan.to_dict(),
        plan_path=plan_path,
        results=results,
        stage_timings=stage_timings,
    )
    results["report_path"] = report_artifacts["report_path"]
    results["operational_note_json_path"] = report_artifacts["operational_note_json_path"]
    results["operational_note_md_path"] = report_artifacts["operational_note_md_path"]

    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a retained live world-v2 training pipeline inside the trainer container")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="corpus")
    parser.add_argument("--action", choices=["plan", "report", "check", "collect", "bc", "eval-bc", "ppo", "eval", "all"], default="all")
    parser.add_argument("--device", default="gpu", help="Requested torch device override")
    parser.add_argument("--demo-dir", default=None, help="Override demo directory inside the trainer container")
    parser.add_argument("--asset-root", default=None, help="Override Quake asset root inside the trainer container")
    parser.add_argument("--worker-binary", default="../artifacts/bin/quake_worker", help="Path to the live worker binary")
    parser.add_argument("--rebuild-worker", action="store_true", help="Force a rebuild of the live worker binary")
    args = parser.parse_args()

    run_live_pipeline(
        profile_name=args.profile,
        action=args.action,
        device=args.device,
        demo_dir=args.demo_dir,
        asset_root=args.asset_root,
        worker_binary=args.worker_binary,
        rebuild_worker=args.rebuild_worker,
    )


if __name__ == "__main__":
    main()
