"""Runtime planning, asset validation, and run-plan retention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from quake_ai.rl.run_config import (
    _require_mapping,
    _require_string,
    build_run_plan_values,
    run_plan_path,
    scenario_note,
)
from quake_ai.utils.device import describe_torch_runtime
from quake_ai.utils.io import write_json

GIB = 1024**3


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


def _runtime_plan(run_cfg: dict[str, Any], runtime: Mapping[str, Any]) -> RuntimePlan:
    cpu_count = max(int(runtime.get("cpu_affinity_count") or runtime.get("cpu_count") or 1), 1)
    gpu_memory_bytes = 0
    devices = runtime.get("devices")
    if isinstance(devices, list):
        gpu_memory_bytes = max(
            (int(device.get("total_memory", 0)) for device in devices if isinstance(device, Mapping)),
            default=0,
        )
    plan_values = build_run_plan_values(run_cfg)
    return RuntimePlan(
        requested_device=str(runtime.get("requested_device", "auto")),
        resolved_device=str(runtime.get("resolved_device", "cpu")),
        backend=str(runtime.get("backend", "cpu")),
        cpu_count=max(int(runtime.get("cpu_count") or cpu_count), 1),
        cpu_affinity_count=cpu_count,
        gpu_memory_bytes=gpu_memory_bytes,
        bc_batch_size=int(plan_values["bc_batch_size"]),
        num_envs=int(plan_values["num_envs"]),
        rollout_steps=int(plan_values["rollout_steps"]),
        total_steps=int(plan_values["total_steps"]),
        minibatch_size=int(plan_values["minibatch_size"]),
        eval_episodes=int(plan_values["eval_episodes"]),
    )


def build_runtime_plan_for_run(
    run_cfg: dict[str, Any],
    requested_device: str,
) -> tuple[dict[str, Any], RuntimePlan]:
    runtime = describe_torch_runtime(requested_device)
    error = runtime.get("error")
    if error:
        raise RuntimeError(f"Accelerator runtime is unavailable: {error}")
    return runtime, _runtime_plan(run_cfg, runtime)


def _looks_like_quake_basedir(path: Path) -> bool:
    id1_dir = path / "id1"
    if not id1_dir.is_dir():
        return False
    pak_names = ("PAK0.PAK", "PAK1.PAK", "pak0.pak", "pak1.pak")
    return any((id1_dir / name).exists() for name in pak_names)


def _resolve_asset_root(explicit: str | None) -> Path:
    if not explicit:
        raise RuntimeError("Training requires an explicit asset_root; define it in machine.json.launcher.")
    candidate = Path(explicit)
    if _looks_like_quake_basedir(candidate):
        return candidate
    raise RuntimeError(f"Configured asset_root is not a Quake basedir with id1/PAK0.PAK: {candidate}")


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
            f"Configured native_args require {mod_root}, but it does not exist. Install the mod first with {install_hint}."
        )
    if not any((mod_root / name).exists() for name in ("progs.dat", "qwprogs.dat")):
        raise RuntimeError(
            f"Configured native_args require a compiled gamedir under {mod_root}, but no progs.dat or qwprogs.dat was found."
        )


def resolve_demo_dir_from_run(run_cfg: dict[str, Any], explicit: str | None = None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir() and any(p.suffix.lower() == ".dem" for p in candidate.iterdir() if p.is_file()):
            return candidate
        raise RuntimeError(f"Demo directory does not contain .dem files: {explicit}")

    scenario = _require_mapping(run_cfg, "scenario", "run config")
    demo_dir = Path(_require_string(scenario, "demo_dir", "scenario.json"))
    if demo_dir.is_dir() and any(p.suffix.lower() == ".dem" for p in demo_dir.iterdir() if p.is_file()):
        return demo_dir
    raise RuntimeError(f"Configured scenario.json.demo_dir does not contain .dem files: {demo_dir}")


def write_run_plan(
    run_cfg: dict[str, Any],
    runtime_scale: str,
    runtime: Mapping[str, Any],
    plan: RuntimePlan,
    demo_dir: Path | None,
    asset_root: Path,
) -> Path:
    manifest = _require_mapping(run_cfg, "manifest", "run config")
    scenario = _require_mapping(run_cfg, "scenario", "run config")
    target = run_plan_path(run_cfg)
    write_json(
        target,
        {
            "run_name": _require_string(manifest, "name", "run.json"),
            "description": str(manifest.get("description", "")),
            "runtime_scale": runtime_scale,
            "scenario_note": scenario_note(run_cfg),
            "demo_dir": str(demo_dir) if demo_dir is not None else "",
            "asset_root": str(asset_root),
            "runtime": dict(runtime),
            "plan": plan.to_dict(),
        },
    )
    return target
