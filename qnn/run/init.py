#!/usr/bin/env python3
"""Initialize a new training run directory under runs/.

Creates the run directory with frozen config copies and a run.json manifest.
Templates live in each pipeline's package (bc/templates/, ppo/templates/).
CLI args are overlaid onto the template. No hardcoded defaults — all values
come from the template files.

Usage:
    python -m qnn.run.init \
        --name bc_v1 \
        --mode bc \
        --resume true

    python -m qnn.run.init \
        --name ppo_v1 \
        --mode ppo \
        --checkpoint-path runs/bc/bc_v1/checkpoints/bc_best.pth \
        --resume true
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from qnn.utils.artifacts import new_run_id

_RUNS_DIR = Path("runs")

_TEMPLATE_DIRS = {
    "bc": Path(__file__).resolve().parent.parent / "bc" / "templates",
    "ppo": Path(__file__).resolve().parent.parent / "ppo" / "templates",
    "pbt": Path(__file__).resolve().parent.parent / "ppo" / "templates",
    "optuna": Path(__file__).resolve().parent.parent / "ppo" / "templates",
    "head_probe": Path(__file__).resolve().parent.parent / "model" / "bench" / "templates",
}


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def _template_dir_for_mode(mode: str) -> Path:
    if mode not in _TEMPLATE_DIRS:
        raise ValueError(f"No templates for mode {mode!r}; expected one of {sorted(_TEMPLATE_DIRS)}")
    return _TEMPLATE_DIRS[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a training run directory")
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument("--mode", required=True, choices=["bc", "ppo", "pbt", "optuna", "head_probe"], help="Training mode")
    parser.add_argument("--resume", choices=["true", "false"], help="Override run.json resume")
    parser.add_argument("--description", help="Override run.json description")
    parser.add_argument("--checkpoint-path", help="Override run.json checkpoint_path")
    # Optional overrides — use a different file instead of the mode's template.
    parser.add_argument("--train", help="Path to train.json (overrides template)")
    parser.add_argument("--scenario", help="Path to scenario.json (overrides template)")
    parser.add_argument("--reward", help="Path to reward.json (overrides template)")
    parser.add_argument("--machine", help="Path to machine.json (overrides template)")
    parser.add_argument("--model", help="Path to model.json (overrides template)")
    parser.add_argument("--probe", help="Path to probe.json (overrides template; head_probe mode)")
    parser.add_argument("--decode", help="Path to decode.json (overrides template; e.g. a versioned templates/decode.*.json)")
    args = parser.parse_args()

    template_dir = _template_dir_for_mode(args.mode)
    run_dir = _RUNS_DIR / args.mode / args.name
    if run_dir.exists():
        parser.error(f"Run directory already exists: {run_dir}")

    # Load run.json template
    run_template = template_dir / "run.json"
    if not run_template.exists():
        parser.error(f"Run template not found: {run_template}")
    manifest = json.loads(run_template.read_text(encoding="utf-8"))

    # Overlay CLI args
    manifest["name"] = args.name
    manifest["run_id"] = new_run_id()
    manifest["mode"] = args.mode
    if args.resume is not None:
        manifest["resume"] = args.resume == "true"
    if args.description is not None:
        manifest["description"] = args.description
    if args.checkpoint_path is not None:
        manifest["checkpoint_path"] = args.checkpoint_path
    manifest["created"] = datetime.now(timezone.utc).isoformat()
    manifest["git_commit"] = _git_commit_hash()

    checkpoint_path = manifest.get("checkpoint_path", "")
    # Empty checkpoint_path on ppo/pbt/optuna means random init.
    # init.py accepts it; config.py accepts it; warm-start in train.py
    # no-ops when init_checkpoint is empty.

    # Resolve config files: CLI override or template default.
    # Each config key maps to its template filename.
    config_keys = list(manifest.get("config", {}).keys())
    config_files: dict[str, str] = {}
    for key in config_keys:
        cli_override = getattr(args, key, None)
        if cli_override:
            config_files[key] = cli_override
        else:
            default = template_dir / f"{key}.json"
            if default.exists():
                config_files[key] = str(default)
            else:
                parser.error(f"Template missing for config/{key}.json and no --{key} override provided")

    # Create run directory and freeze configs
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True)

    # Copy seed checkpoint into the run dir so the run is self-contained.
    if checkpoint_path:
        src_ckpt = Path(checkpoint_path)
        if not src_ckpt.exists():
            parser.error(f"Seed checkpoint not found: {src_ckpt}")
        seed_dir = run_dir / "seed"
        seed_dir.mkdir(parents=True)
        dest_ckpt = seed_dir / src_ckpt.name
        shutil.copy2(src_ckpt, dest_ckpt)
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

    # Pin the look turn-delta grid for look-bearing modes: copy the corpus's
    # data-fit grid into config/look_grid.json so it lives with the model (no
    # implicit code default). The trainer installs it at job start.
    if args.mode in ("bc", "head_probe"):
        from qnn.model import look_grid as _look_grid
        machine = json.loads((config_dir / "machine.json").read_text())
        bc_data_dir = machine.get("bc_data_dir")
        if not bc_data_dir:
            parser.error("machine.json missing bc_data_dir; cannot pin look grid")
        grid = _look_grid.pinned_grid_from_collect(bc_data_dir)
        grid["git_commit"] = manifest["git_commit"]
        grid["created"] = manifest["created"]
        (config_dir / "look_grid.json").write_text(json.dumps(grid, indent=2) + "\n")
        print(f"  Look grid pinned from corpus fit: {bc_data_dir} "
              f"(rms {grid.get('fit_rms_deg')} vs default {grid.get('default_rms_deg')})")

        # Pin the move-axis dwell-hazard release table from the same corpus, beside
        # the look grid. The move decode reads edges/fb/lr from this pinned table.
        from qnn.model import move_hazard as _move_hazard
        haz = _move_hazard.pinned_hazard_from_collect(bc_data_dir)
        haz["git_commit"] = manifest["git_commit"]
        haz["created"] = manifest["created"]
        (config_dir / "move_hazard.json").write_text(json.dumps(haz, indent=2) + "\n")
        print(f"  Move hazard pinned from corpus: {bc_data_dir} "
              f"(tick_hz {haz.get('tick_hz')}, edges {haz.get('edges')})")

        # Pin the weapon WHEN-hazard table from the same corpus. Best-effort: corpora
        # collected before weapon_hazard was added won't carry the block until
        # recollected/backfilled — don't fail run-init over it during the transition.
        try:
            from qnn.model import weapon_hazard as _weapon_hazard
            whaz = _weapon_hazard.pinned_hazard_from_collect(bc_data_dir)
            whaz["git_commit"] = manifest["git_commit"]
            whaz["created"] = manifest["created"]
            (config_dir / "weapon_hazard.json").write_text(json.dumps(whaz, indent=2) + "\n")
            print(f"  Weapon hazard pinned from corpus: {bc_data_dir} "
                  f"(tick_hz {whaz.get('tick_hz')}, axes {whaz.get('axes')})")
        except Exception as exc:  # noqa: BLE001 — transitional; recollect/backfill to enable
            print(f"  Weapon hazard pin skipped: {exc}")

    # PPO carries the SEED's pinned look grid: the trainer installs
    # config/look_grid.json before any forward (no code default), so a
    # look-bearing seed without its grid would fail — or silently decode on
    # the wrong grid after a resume. Resolve it from the seed's own run dir
    # (checkpoints/ or seed/ layouts both sit one level under the run root).
    if args.mode == "ppo" and manifest.get("checkpoint_source"):
        seed_run_cfg = Path(manifest["checkpoint_source"]).resolve().parent.parent / "config"
        seed_grid = seed_run_cfg / "look_grid.json"
        if seed_grid.exists():
            shutil.copy2(seed_grid, config_dir / "look_grid.json")
            print(f"  Look grid pinned from seed run: {seed_grid}")
        else:
            parser.error(
                f"Seed run has no pinned look grid ({seed_grid}). RL seeds "
                "must be a25+ run-dir checkpoints carrying config/look_grid.json."
            )

    # Write manifest
    with open(run_dir / "run.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Stamp run notes from template
    notes_template = template_dir / "run.md"
    if notes_template.exists():
        notes_text = notes_template.read_text(encoding="utf-8")
        notes_text = notes_text.replace("{{name}}", args.name)
        notes_text = notes_text.replace("{{checkpoint_path}}", checkpoint_path)
        (run_dir / "run.md").write_text(notes_text, encoding="utf-8")
    else:
        (run_dir / "run.md").write_text("", encoding="utf-8")

    print(f"Initialized run: {run_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Config frozen from: {', '.join(config_files.values())}")
    print(f"  Checkpoint: {checkpoint_path or '(none)'}")
    print(f"  Commit: {manifest['git_commit']}")


if __name__ == "__main__":
    main()
