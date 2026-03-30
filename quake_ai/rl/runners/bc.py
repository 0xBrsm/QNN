"""Behavior cloning runner.

Expects precomputed .npy caches at {asset_root}/bc/precomputed_train/
and precomputed_val/. Run scripts/bc_collect.py first to generate them.
"""

from __future__ import annotations

import time
from pathlib import Path

from quake_ai.rl.run_config import build_run_bc_config
from quake_ai.rl.runners.common import (
    RunnerContext,
    base_results,
    finalize_results,
    prepare_bc_run_outputs,
)
from quake_ai.rl.bc_train import BCConfig, run_behavior_cloning


def run(ctx: RunnerContext) -> dict[str, object]:
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    bc_cfg = build_run_bc_config(ctx.run_cfg, ctx.device)
    prepare_bc_run_outputs(ctx.run_cfg, resume=ctx.resume)

    # Verify precomputed caches exist
    bc_data_dir = Path(bc_cfg.get("bc_data_dir", ""))
    train_cache = bc_data_dir / "precomputed_train"
    if not train_cache.exists():
        raise RuntimeError(
            f"BC training data not found at {train_cache}. "
            f"Run scripts/bc_collect.py first."
        )

    seed_checkpoint = str(ctx.run_cfg.get("checkpoint_path", ""))
    started = time.monotonic()
    # Filter to BCConfig fields only — bc_cfg may have extra keys from trainer/model merge
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(BCConfig)}
    filtered_cfg = {k: v for k, v in bc_cfg.items() if k in valid_keys}
    results["bc"] = run_behavior_cloning(BCConfig(**filtered_cfg), seed_checkpoint=seed_checkpoint)
    stage_timings["bc"] = time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
