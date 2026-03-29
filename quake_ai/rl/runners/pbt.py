"""Population-based-training orchestration over the shared PPO core."""

from __future__ import annotations

from quake_ai.rl.runners.common import RunnerContext, require_cfg_mapping, require_cfg_value
from quake_ai.rl.runners.ppo import run_pipeline


def run(ctx: RunnerContext) -> dict[str, object]:
    trainer = require_cfg_mapping(ctx.run_cfg, "trainer", "run config")
    with_pbt = bool(require_cfg_value(trainer, "with_pbt", "trainer.json"))
    num_policies = int(require_cfg_value(trainer, "num_policies", "trainer.json"))
    if not with_pbt and num_policies <= 1:
        raise RuntimeError("pbt mode requires trainer.json.with_pbt=true or trainer.json.num_policies > 1")
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
