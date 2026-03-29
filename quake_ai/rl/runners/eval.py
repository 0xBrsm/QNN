"""Standalone evaluation runner."""

from __future__ import annotations

import time

from quake_ai.rl.evaluation import EvalConfig, run_evaluation
from quake_ai.rl.planning import _validate_native_mod_assets
from quake_ai.rl.run_config import build_run_eval_config
from quake_ai.rl.runners.common import (
    RunnerContext,
    base_results,
    ensure_worker,
    finalize_results,
    prepare_eval_checkpoint,
    prepare_eval_outputs,
    require_cfg_string,
)


def run(ctx: RunnerContext) -> dict[str, object]:
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    prepare_eval_outputs(ctx.run_cfg, resume=ctx.resume)
    eval_cfg = build_run_eval_config(ctx.run_cfg, ctx.device)
    _validate_native_mod_assets(
        ctx.asset_root,
        eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None,
    )
    worker_path = ensure_worker(ctx.worker_binary, rebuild=False)
    eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(ctx.asset_root)}
    eval_cfg["native_executable"] = str(worker_path)
    eval_cfg["checkpoint_path"] = prepare_eval_checkpoint(
        require_cfg_string(ctx.run_cfg, "checkpoint_path", "run config"),
        str(eval_cfg["output_dir"]),
    )

    started = time.monotonic()
    results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
    stage_timings["eval"] = time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
