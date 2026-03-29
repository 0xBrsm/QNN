"""Training router driven entirely by a run directory manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from quake_ai.rl.runners import get_runner
from quake_ai.rl.runners.common import RunnerContext, best_checkpoint, build_runner_context


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
