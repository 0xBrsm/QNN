"""Population-based-training orchestration over the shared PPO core."""

from __future__ import annotations

from qnn.run.common import RunnerContext, require_cfg_mapping, require_cfg_value
from qnn.ppo.pipeline import run_pipeline


def run(ctx: RunnerContext) -> dict[str, object]:
    train = require_cfg_mapping(ctx.run_cfg, "train", "run config")
    with_pbt = bool(require_cfg_value(train, "with_pbt", "train.json"))
    num_policies = int(require_cfg_value(train, "num_policies", "train.json"))
    if not with_pbt and num_policies <= 1:
        raise RuntimeError("pbt mode requires train.json.with_pbt=true or train.json.num_policies > 1")
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
