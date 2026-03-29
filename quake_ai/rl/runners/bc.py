"""Behavior cloning runner."""

from __future__ import annotations

import time
from pathlib import Path

from quake_ai.rl.collect import CollectConfig, collect_demo_tokens
from quake_ai.rl.planning import resolve_demo_dir_from_run
from quake_ai.rl.run_config import build_run_bc_config
from quake_ai.rl.runners.common import (
    RunnerContext,
    base_results,
    ensure_demo_worker,
    finalize_results,
    prepare_bc_run_outputs,
    require_cfg_mapping,
    require_cfg_string,
    require_cfg_value,
)
from quake_ai.rl.training_bc import BCConfig, run_behavior_cloning


def run(ctx: RunnerContext) -> dict[str, object]:
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    bc_cfg = build_run_bc_config(ctx.run_cfg, ctx.device)
    prepare_bc_run_outputs(ctx.run_cfg, resume=ctx.resume)
    demo_dir = resolve_demo_dir_from_run(ctx.run_cfg)

    bc_ticks_path = Path(require_cfg_string(bc_cfg, "token_ticks_path", "BC config"))
    if not bc_ticks_path.exists():
        machine = require_cfg_mapping(ctx.run_cfg, "machine", "run config")
        started = time.monotonic()
        demo_worker_path = ensure_demo_worker(
            Path(require_cfg_string(machine, "demo_worker_binary", "machine.json")),
            rebuild=False,
        )
        collect_cfg = CollectConfig(
            demo_worker_binary=str(demo_worker_path),
            demo_dir=str(demo_dir),
            output_dir=str(bc_ticks_path.parent),
            map_id=require_cfg_string(bc_cfg, "map_id", "BC config"),
            fixed_tick_hz=int(require_cfg_value(bc_cfg, "fixed_tick_hz", "BC config")),
            asset_root=str(ctx.asset_root),
        )
        collect_result = collect_demo_tokens(collect_cfg)
        bc_cfg["token_ticks_path"] = collect_result.token_ticks_path
        bc_cfg["map_state_path"] = collect_result.map_state_path
        bc_cfg["map_states_path"] = collect_result.map_states_path
        bc_cfg["metadata_path"] = collect_result.metadata_path
        results["collect"] = {
            "token_ticks_path": collect_result.token_ticks_path,
            "map_state_path": collect_result.map_state_path,
            "map_states_path": collect_result.map_states_path,
            "metadata_path": collect_result.metadata_path,
            "missing_demos_path": collect_result.missing_demos_path,
            "source_summary_path": collect_result.source_summary_path,
            "demos_processed": collect_result.demos_processed,
            "demos_failed": collect_result.demos_failed,
            "total_ticks": collect_result.total_ticks,
        }
        stage_timings["collect"] = time.monotonic() - started
    else:
        results["collect"] = {"skipped": True, "reason": "BC data already available."}

    seed_checkpoint = str(ctx.run_cfg.get("checkpoint_path", ""))
    started = time.monotonic()
    results["bc"] = run_behavior_cloning(BCConfig(**bc_cfg), seed_checkpoint=seed_checkpoint)
    stage_timings["bc"] = time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
