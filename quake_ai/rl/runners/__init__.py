"""Runner registry for run-dir training modes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quake_ai.rl.runners.common import RunnerContext

RunnerFn = Callable[[RunnerContext], dict[str, Any]]


def get_runner(mode: str) -> RunnerFn:
    if mode == "bc":
        from quake_ai.rl.runners.bc import run

        return run
    if mode == "ppo":
        from quake_ai.rl.runners.ppo import run

        return run
    if mode == "pbt":
        from quake_ai.rl.runners.pbt import run

        return run
    if mode == "eval":
        from quake_ai.rl.runners.eval import run

        return run
    if mode == "optuna":
        from quake_ai.rl.runners.optuna import run

        return run
    raise RuntimeError(f"Unsupported run mode in run.json: {mode}")
