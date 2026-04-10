#!/usr/bin/env python3
"""Initialize a new training run directory under runs/.

Creates the run directory with frozen config copies, a run.json manifest,
and a run.md notes file from the template in run_templates/. CLI args are
overlaid onto the run.json template. No hardcoded paths — all defaults
come from the template.

Usage:
    python -m quake_ai.rl.init_run \
        --name arena_1v1_20260325 \
        --checkpoint-path runs/bc_final_gru/checkpoints/bc_best_model.pth \
        --scenario src/quake_ai/rl/run_templates/scenario.json \
        --reward src/quake_ai/rl/run_templates/reward.json \
        --trainer src/quake_ai/rl/run_templates/trainer.json \
        --machine src/quake_ai/rl/run_templates/machine.json \
        --model src/quake_ai/rl/run_templates/model.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_RUNS_DIR = Path("runs")
_TEMPLATE_DIR = Path("src/quake_ai/rl/run_templates")
_RUN_TEMPLATE = _TEMPLATE_DIR / "run.json"


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a training run directory")
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument("--mode", choices=["bc", "ppo", "pbt", "eval", "optuna"], help="Override run.json mode")
    parser.add_argument("--runtime-scale", choices=["live", "verify"], help="Override run.json runtime_scale")
    parser.add_argument("--resume", choices=["true", "false"], help="Override run.json resume")
    parser.add_argument("--description", help="Override run.json description")
    parser.add_argument("--checkpoint-path", help="Override run.json checkpoint_path")
    parser.add_argument("--trainer", default=str(_TEMPLATE_DIR / "trainer.json"), help="Path to trainer.json")
    parser.add_argument("--scenario", default=str(_TEMPLATE_DIR / "scenario.json"), help="Path to scenario.json")
    parser.add_argument("--reward", default=str(_TEMPLATE_DIR / "reward.json"), help="Path to reward.json")
    parser.add_argument("--machine", default=str(_TEMPLATE_DIR / "machine.json"), help="Path to machine.json")
    parser.add_argument("--model", default=str(_TEMPLATE_DIR / "model.json"), help="Path to model.json")
    parser.add_argument("--eval", default=str(_TEMPLATE_DIR / "eval.json"), help="Path to eval.json")
    args = parser.parse_args()

    run_dir = _RUNS_DIR / args.name
    if run_dir.exists():
        parser.error(f"Run directory already exists: {run_dir}")

    # Load run.json template
    manifest = json.loads(_RUN_TEMPLATE.read_text(encoding="utf-8"))

    # Overlay CLI args
    manifest["name"] = args.name
    if args.mode is not None:
        manifest["mode"] = args.mode

    if args.runtime_scale is not None:
        manifest["runtime_scale"] = args.runtime_scale
    if args.resume is not None:
        manifest["resume"] = args.resume == "true"
    if args.description is not None:
        manifest["description"] = args.description
    if args.checkpoint_path is not None:
        manifest["checkpoint_path"] = args.checkpoint_path
    manifest["created"] = datetime.now(timezone.utc).isoformat()
    manifest["git_commit"] = _git_commit_hash()

    mode = manifest.get("mode")
    if mode not in {"bc", "ppo", "pbt", "eval", "optuna"}:
        parser.error(f"run.json mode must be one of bc, ppo, pbt, eval, optuna; got {mode!r}")

    runtime_scale = manifest.get("runtime_scale")
    if runtime_scale not in {"live", "verify"}:
        parser.error(f"run.json runtime_scale must be 'live' or 'verify'; got {runtime_scale!r}")

    resume = manifest.get("resume")
    if not isinstance(resume, bool):
        parser.error(f"run.json resume must be a boolean; got {resume!r}")

    checkpoint_path = manifest.get("checkpoint_path", "")
    if not isinstance(checkpoint_path, str):
        parser.error(f"run.json checkpoint_path must be a string; got {checkpoint_path!r}")

    if mode in {"eval", "ppo", "pbt", "optuna"} and not checkpoint_path:
        parser.error(
            "--checkpoint-path is required unless the run.json template already defines one for "
            "eval, ppo, pbt, or optuna runs"
        )

    # Freeze config copies
    config_files = {
        "trainer": args.trainer,
        "scenario": args.scenario,
        "reward": args.reward,
        "machine": args.machine,
        "model": args.model,
        "eval": args.eval,
    }

    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True)

    # Copy seed checkpoint into the run dir so the run is self-contained.
    # run.json.checkpoint_path points to the local copy; the original
    # source path is recorded in run.json.checkpoint_source for provenance.
    if checkpoint_path:
        src_ckpt = Path(checkpoint_path)
        if not src_ckpt.exists():
            parser.error(f"Seed checkpoint not found: {src_ckpt}")
        seed_dir = run_dir / "seed"
        seed_dir.mkdir(parents=True)
        dest_ckpt = seed_dir / src_ckpt.name
        shutil.copy2(src_ckpt, dest_ckpt)
        # Also copy sidecar JSON if present (architecture metadata for SF checkpoints).
        sidecar = src_ckpt.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, seed_dir / sidecar.name)
        manifest["checkpoint_source"] = checkpoint_path
        manifest["checkpoint_path"] = str(dest_ckpt)
        checkpoint_path = manifest["checkpoint_path"]

    for name, src_path in config_files.items():
        src = Path(src_path)
        if not src.exists():
            parser.error(f"Config file not found: {src}")
        shutil.copy2(src, config_dir / f"{name}.json")

    # Write manifest
    with open(run_dir / "run.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Stamp run notes from template
    notes_template = _TEMPLATE_DIR / "run.md"
    if notes_template.exists():
        notes_text = notes_template.read_text(encoding="utf-8")
        notes_text = notes_text.replace("{{name}}", args.name)
        notes_text = notes_text.replace("{{checkpoint_path}}", checkpoint_path)
        (run_dir / "run.md").write_text(notes_text, encoding="utf-8")

    print(f"Initialized run: {run_dir}")
    print(f"  Config frozen from: {', '.join(config_files.values())}")
    print(f"  Checkpoint: {checkpoint_path or '(none)'}")
    print(f"  Commit: {manifest['git_commit']}")


if __name__ == "__main__":
    main()
