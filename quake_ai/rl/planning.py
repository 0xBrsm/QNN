"""Runtime planning — device detection, resource allocation, asset resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from quake_ai.rl.profiles import LiveProfile, PROFILES
from quake_ai.utils.device import describe_torch_runtime
from quake_ai.utils.io import read_json, write_json

GIB = 1024**3


def _power_of_two_floor(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


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
        # Live PvP runs need longer PPO windows to expose long-horizon behavior,
        # but evaluation stays bounded so retained runs complete in practical time.
        rollout_steps = 256 if gpu_memory_bytes >= 12 * GIB else 128
        total_steps = max(262_144, num_envs * rollout_steps * 64)
        eval_episodes = 32

    if gpu_memory_bytes >= 24 * GIB:
        bc_batch_size = 8_192 if profile.runtime_scale == "live" else 4_096
        minibatch_cap = 2_048
    elif gpu_memory_bytes >= 12 * GIB:
        bc_batch_size = 4_096 if profile.runtime_scale == "live" else 2_048
        minibatch_cap = 1_024
    elif gpu_memory_bytes >= 8 * GIB:
        bc_batch_size = 2_048 if profile.runtime_scale == "live" else 1_024
        minibatch_cap = 512
    else:
        bc_batch_size = 1_024 if profile.runtime_scale == "live" else 512
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


def _looks_like_quake_basedir(path: Path) -> bool:
    id1_dir = path / "id1"
    if not id1_dir.is_dir():
        return False
    pak_names = ("PAK0.PAK", "PAK1.PAK", "pak0.pak", "pak1.pak")
    return any((id1_dir / name).exists() for name in pak_names)


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
    install_hint = "scripts/build-mod.sh"
    if gamedir == "frikbotnex":
        install_hint = "scripts/build-mod.sh --mod-only"
    mod_root = asset_root / gamedir
    if not mod_root.is_dir():
        raise RuntimeError(
            f"Configured native_args require {mod_root}, but it does not exist. "
            f"Install the mod first with {install_hint}."
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
