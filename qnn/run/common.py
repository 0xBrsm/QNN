"""Shared runner context and lifecycle helpers."""

from __future__ import annotations

import glob as _glob
import json
import re as _re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qnn.env.planning import (
    RuntimePlan,
    resolve_asset_root,
    build_runtime_plan_for_run,
    resolve_demo_dir_from_run,
    write_run_plan,
)
from qnn.run.config import load_run_config, run_output_dirs
from qnn.run.reporting import write_run_report
from qnn.utils.io import read_json, trusted_torch_load, write_json


@dataclass(slots=True)
class RunnerContext:
    run_cfg: dict[str, Any]
    run_dir: Path
    manifest: dict[str, Any]
    mode: str
    runtime_scale: str
    resume: bool
    device: str
    asset_root: Path
    worker_binary: Path | None
    runtime: dict[str, Any]
    plan: RuntimePlan
    plan_path: Path
    output_dirs: dict[str, Path]


def _reward_from_name(path: str) -> float:
    match = _re.search(r"reward_([-\d.]+)\.pth$", path)
    return float(match.group(1)) if match else float("-inf")


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def require_cfg_value(config: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in config:
        raise RuntimeError(f"{context} must define {key}")
    return config[key]


def require_cfg_string(config: Mapping[str, Any], key: str, context: str) -> str:
    value = require_cfg_value(config, key, context)
    if not isinstance(value, str):
        raise RuntimeError(f"{context}.{key} must be a string")
    return value


def require_cfg_list(config: Mapping[str, Any], key: str, context: str) -> list[Any]:
    value = require_cfg_value(config, key, context)
    if not isinstance(value, list):
        raise RuntimeError(f"{context}.{key} must be a list")
    return list(value)


def require_cfg_mapping(config: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = require_cfg_value(config, key, context)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}.{key} must be a mapping")
    return dict(value)


def build_runner_context(run_dir: Path) -> RunnerContext:
    run_cfg = load_run_config(run_dir)
    run_dir_resolved = Path(run_cfg["run_dir"])
    manifest = require_cfg_mapping(run_cfg, "manifest", "run config")
    mode = require_cfg_string(run_cfg, "mode", "run config")
    runtime_scale = require_cfg_string(run_cfg, "runtime_scale", "run config")
    resume = bool(require_cfg_value(run_cfg, "resume", "run config"))

    machine = require_cfg_mapping(run_cfg, "machine", "run config")
    resolved_device = require_cfg_string(machine, "device", "machine.json")
    resolved_asset_root = resolve_asset_root(require_cfg_string(machine, "asset_root", "machine.json"))
    worker_binary_str = machine.get("worker_binary", "")
    worker_path = Path(worker_binary_str) if worker_binary_str else None

    runtime, plan = build_runtime_plan_for_run(run_cfg, resolved_device)
    plan_path = write_run_plan(
        run_cfg,
        runtime_scale,
        runtime,
        plan,
        resolve_demo_dir_from_run(run_cfg),
        resolved_asset_root,
    )

    return RunnerContext(
        run_cfg=run_cfg,
        run_dir=run_dir_resolved,
        manifest=manifest,
        mode=mode,
        runtime_scale=runtime_scale,
        resume=resume,
        device=resolved_device,
        asset_root=resolved_asset_root,
        worker_binary=worker_path,
        runtime=dict(runtime),
        plan=plan,
        plan_path=plan_path,
        output_dirs=run_output_dirs(run_cfg),
    )


def base_results(ctx: RunnerContext) -> dict[str, Any]:
    return {
        "run_dir": str(ctx.run_dir),
        "run_name": require_cfg_string(ctx.manifest, "name", "run.json"),
        "mode": ctx.mode,
        "runtime_scale": ctx.runtime_scale,
        "resume": ctx.resume,
        "plan_path": str(ctx.plan_path),
        "runtime": ctx.runtime,
        "plan": ctx.plan.to_dict(),
    }


def finalize_results(
    ctx: RunnerContext,
    results: dict[str, Any],
    stage_timings: Mapping[str, float],
) -> dict[str, Any]:
    report = write_run_report(
        run_root=ctx.run_dir,
        action=ctx.mode,
        runtime_scale=ctx.runtime_scale,
        runtime=ctx.runtime,
        plan=ctx.plan.to_dict(),
        results=results,
        stage_timings=stage_timings,
    )
    results.update(
        {
            "report_path": report["report_path"],
            "operational_note_json_path": report["operational_note_json_path"],
            "operational_note_md_path": report["operational_note_md_path"],
        }
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def best_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the highest-reward checkpoint under *checkpoints_dir*."""
    best_files = _glob.glob(f"{checkpoints_dir}/checkpoint_p*/best_0*.pth")
    if best_files:
        return Path(max(best_files, key=_reward_from_name))
    regular = sorted(_glob.glob(f"{checkpoints_dir}/checkpoint_p*/checkpoint_*.pth"))
    return Path(regular[-1]) if regular else None


def is_sf_checkpoint_payload(payload: object) -> bool:
    return isinstance(payload, dict) and "model" in payload and ("train_step" in payload or "env_steps" in payload)


def prepare_eval_checkpoint(checkpoint_path: str, output_dir: str) -> str:
    path = Path(checkpoint_path)
    if not path.exists() or path.suffix != ".pth":
        return str(path)

    try:
        payload = trusted_torch_load(str(path), map_location="cpu")
    except Exception:
        return str(path)

    if not is_sf_checkpoint_payload(payload):
        return str(path)

    from qnn.utils.checkpoint_converter import sf_to_qnn
    from qnn.model.network import ModelConfig

    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise RuntimeError(f"SF checkpoint conversion requires architecture sidecar metadata: {sidecar_path}")
    meta = read_json(sidecar_path)
    if not isinstance(meta, Mapping):
        raise RuntimeError(f"SF checkpoint sidecar must be a JSON object: {sidecar_path}")
    if "obs_dim" not in meta:
        raise RuntimeError(f"SF checkpoint sidecar missing 'obs_dim': {sidecar_path}")
    converted_dir = Path(output_dir).parent / "_eval_ckpts"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted_path = converted_dir / f"{path.stem}_qnn.pth"
    converted_sidecar_path = converted_dir / f"{path.stem}_qnn.json"

    if not converted_path.exists() or converted_path.stat().st_mtime < path.stat().st_mtime:
        # Graph-described runs rebuild through the sidecar's model_graph —
        # a flat-built module cannot receive graph-shaped SF weights.
        graph = None
        if meta.get("model_graph") is not None:
            from qnn.model.graph import GraphSpec
            graph = GraphSpec.from_dict(meta["model_graph"])
        policy = sf_to_qnn(
            sf_checkpoint_path=path,
            obs_dim=int(meta["obs_dim"]),
            model=None if graph is not None else ModelConfig.from_flat_dict(meta),
            device="cpu",
            graph=graph,
        )
        policy.save(converted_path)
        converted_sidecar_path.write_text(
            json.dumps({"source_checkpoint": str(path), "converted_checkpoint": str(converted_path)}, indent=2),
            encoding="utf-8",
        )

    return str(converted_path)


def next_archive_path(path: Path) -> Path:
    attempt = 1
    while True:
        candidate = path.with_name(f"{path.name}_old{attempt:04d}")
        if not candidate.exists():
            return candidate
        attempt += 1


def archive_path_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    archived = next_archive_path(path)
    path.rename(archived)
    return archived


def assert_no_legacy_ppo_layout(run_cfg: dict[str, Any]) -> None:
    legacy_dir = run_output_dirs(run_cfg)["checkpoints"] / "ppo"
    if legacy_dir.exists():
        raise RuntimeError(
            f"Legacy PPO checkpoint layout is unsupported: {legacy_dir}. "
            "PPO artifacts now live directly under run.json.output.checkpoints."
        )


def ppo_checkpoint_paths(run_cfg: dict[str, Any]) -> list[Path]:
    checkpoints_dir = run_output_dirs(run_cfg)["checkpoints"]
    return sorted(checkpoints_dir.glob("checkpoint_p*/*.pth"))


def latest_ppo_checkpoint(run_cfg: dict[str, Any]) -> Path | None:
    checkpoints = ppo_checkpoint_paths(run_cfg)
    if not checkpoints:
        return None
    return checkpoints[-1]


def prepare_ppo_run_outputs(run_cfg: dict[str, Any], *, resume: bool) -> bool:
    assert_no_legacy_ppo_layout(run_cfg)
    outputs = run_output_dirs(run_cfg)
    existing = latest_ppo_checkpoint(run_cfg)
    if resume and existing is not None:
        return True

    for path in (
        outputs["checkpoints"] / ".summary",
        outputs["checkpoints"] / "best",
        outputs["metrics"] / "eval",
        outputs["checkpoints"] / "config.json",
        outputs["checkpoints"] / "sf_log.txt",
        outputs["checkpoints"] / "git.diff",
    ):
        archive_path_if_exists(path)
    for path in outputs["checkpoints"].glob("checkpoint_p*"):
        archive_path_if_exists(path)
    return False


def prepare_bc_run_outputs(run_cfg: dict[str, Any], *, resume: bool) -> None:
    outputs = run_output_dirs(run_cfg)
    checkpoint_path = outputs["checkpoints"] / "bc_training_checkpoint.pt"
    if resume:
        if checkpoint_path.exists():
            return
        # Nothing to resume from — start fresh.

    for path in (
        outputs["checkpoints"] / "bc_training_checkpoint.pt",
        outputs["checkpoints"] / "bc_best_model.pth",
        outputs["checkpoints"] / "bc_history.json",
        outputs["checkpoints"] / "bc_summary.json",
        outputs["checkpoints"] / "bc_manifest.json",
        outputs["checkpoints"] / "checkpoints",
    ):
        archive_path_if_exists(path)


def prepare_eval_outputs(run_cfg: dict[str, Any], *, resume: bool) -> None:
    if resume:
        return
    outputs = run_output_dirs(run_cfg)
    archive_path_if_exists(outputs["metrics"] / "eval")


def ensure_worker(worker_binary: Path, rebuild: bool) -> Path:
    if worker_binary.exists() and not rebuild:
        return worker_binary
    build_script = Path("src/engine/build/build_ppo_worker.sh")
    subprocess.run(["bash", str(build_script), str(worker_binary)], check=True)
    return worker_binary




def _merged_payload(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(base)
    if overrides:
        payload.update(dict(overrides))
    return payload


def sync_parent_machine_config(parent_run_cfg: dict[str, Any], child_run_dir: Path) -> None:
    """Copy the parent's machine.json into a child run dir.

    machine.json is the one mutable config (performance tuning) that should
    propagate to child runs on resume.  Call this before launching a child
    run that may already exist with a stale frozen machine.json.
    """
    import shutil

    parent_machine = parent_run_cfg.get("config_paths", {}).get("machine")
    child_machine = child_run_dir / "config" / "machine.json"
    if parent_machine and Path(parent_machine).exists() and child_machine.exists():
        shutil.copy2(str(parent_machine), str(child_machine))


def materialize_child_run(
    parent_run_cfg: dict[str, Any],
    run_dir: Path,
    *,
    name: str,
    mode: str,
    checkpoint_path: str,
    resume: bool,
    description: str,
    runtime_scale: str | None = None,
    trainer_overrides: Mapping[str, Any] | None = None,
    scenario_overrides: Mapping[str, Any] | None = None,
    reward_overrides: Mapping[str, Any] | None = None,
    machine_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    manifest_overrides: Mapping[str, Any] | None = None,
) -> Path:
    if run_dir.exists():
        raise RuntimeError(f"Child run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True)
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_payloads = {
        "train": _merged_payload(require_cfg_mapping(parent_run_cfg, "train", "run config"), trainer_overrides),
        "scenario": _merged_payload(require_cfg_mapping(parent_run_cfg, "scenario", "run config"), scenario_overrides),
        "reward": _merged_payload(require_cfg_mapping(parent_run_cfg, "reward", "run config"), reward_overrides),
        "machine": _merged_payload(require_cfg_mapping(parent_run_cfg, "machine", "run config"), machine_overrides),
        "model": _merged_payload(require_cfg_mapping(parent_run_cfg, "model", "run config"), model_overrides),
    }
    for config_name, payload in config_payloads.items():
        write_json(config_dir / f"{config_name}.json", payload)

    parent_manifest = require_cfg_mapping(parent_run_cfg, "manifest", "run config")
    manifest = dict(parent_manifest)
    manifest["name"] = name
    manifest["mode"] = mode
    manifest["runtime_scale"] = runtime_scale or require_cfg_string(parent_run_cfg, "runtime_scale", "run config")
    manifest["resume"] = resume
    manifest["description"] = description
    manifest["checkpoint_path"] = checkpoint_path
    manifest["config"] = {
        "train": "config/train.json",
        "scenario": "config/scenario.json",
        "reward": "config/reward.json",
        "machine": "config/machine.json",
        "model": "config/model.json",
    }
    manifest["created"] = datetime.now(timezone.utc).isoformat()
    manifest["git_commit"] = _git_commit_hash()
    if manifest_overrides:
        manifest.update(dict(manifest_overrides))

    write_json(run_dir / "run.json", manifest)
    return run_dir
