"""Training router driven entirely by a run directory manifest."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qnn.run.common import RunnerContext, best_checkpoint, build_runner_context


def get_runner(mode: str) -> Callable[[RunnerContext], dict[str, Any]]:
    if mode == "bc":
        from qnn.bc.train import run
        return run
    if mode == "ppo":
        from qnn.ppo.train import run
        return run
    if mode == "pbt":
        from qnn.ppo.pbt import run
        return run
    if mode == "eval":
        from qnn.eval.run import run
        return run
    if mode == "optuna":
        from qnn.ppo.optuna import run
        return run
    if mode == "head_probe":
        from qnn.model.bench.runner import run
        return run
    raise RuntimeError(f"Unsupported run mode in run.json: {mode}")


def run_context(ctx: RunnerContext) -> dict[str, Any]:
    runner = get_runner(ctx.mode)
    return runner(ctx)


def run_from_run_dir(run_dir: Path) -> dict[str, Any]:
    ctx = build_runner_context(run_dir)
    return run_context(ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training pipeline — run.json is the single entry point")
    parser.add_argument("--run-dir", required=True, help="Run directory containing run.json")
    args = parser.parse_args()
    run_from_run_dir(Path(args.run_dir))


if __name__ == "__main__":
    main()
