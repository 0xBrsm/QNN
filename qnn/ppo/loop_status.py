"""Step-gated status for active PPO run directories.

Reads the native trainer's ``ppo_history.json`` (one row per PPO
iteration — see qnn.ppo.train). The step gates and recommended-action
ladder are unchanged from the SF era; only the telemetry source moved
(sf_log.txt parsing died with the Sample Factory backend).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qnn.run.metrics import effective_game_minutes_per_wall_minute
from qnn.utils.io import safe_read_json

DEFAULT_STEP_GATES: tuple[int, ...] = (1_000_000, 3_000_000, 5_000_000)

_safe_read_json = safe_read_json


def _assert_supported_layout(run_root: Path) -> None:
    legacy_dir = run_root / "checkpoints" / "ppo"
    if legacy_dir.exists():
        raise RuntimeError(
            f"Legacy PPO checkpoint layout is unsupported: {legacy_dir}. "
            "PPO artifacts now live directly under run.json.output.checkpoints."
        )


def _history_rows(run_root: Path) -> list[dict[str, Any]]:
    payload = _safe_read_json(run_root / "checkpoints" / "ppo_history.json")
    if not isinstance(payload, dict):
        return []
    rows = payload.get("history")
    return rows if isinstance(rows, list) else []


def _gate_rows(seed_env_steps: int, current_env_steps: int, gates: tuple[int, ...]) -> list[dict[str, Any]]:
    delta = max(0, current_env_steps - seed_env_steps)
    rows: list[dict[str, Any]] = []
    for target in gates:
        rows.append(
            {
                "delta_steps": int(target),
                "absolute_steps": int(seed_env_steps + target),
                "reached": delta >= target,
                "remaining_delta_steps": max(0, int(target - delta)),
            }
        )
    return rows


def _eval_is_stale(best_path: Path, eval_path: Path) -> bool:
    if not eval_path.exists():
        return True
    if not best_path.exists():
        return False
    return best_path.stat().st_mtime > eval_path.stat().st_mtime


def build_loop_status(
    root: str | Path,
    *,
    step_gates: tuple[int, ...] = DEFAULT_STEP_GATES,
    eval_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_root = Path(root)
    _assert_supported_layout(run_root)
    manifest = _safe_read_json(run_root / "run.json") or {}
    from qnn.run.router import best_checkpoint
    best_path = best_checkpoint(run_root / "checkpoints") or (run_root / "checkpoints" / "nonexistent")
    eval_summary_path = run_root / "metrics" / "eval" / "eval_summary.json"

    rows = _history_rows(run_root)
    last = rows[-1] if rows else {}
    # Native runs count RL env steps from 0 — the BC seed contributes no
    # env steps, so the gate ladder is the RL delta directly.
    seed_env_steps = 0
    current_env_steps = int(last["env_steps"]) if "env_steps" in last else None
    throughput = float(last["fps"]) if "fps" in last else None
    current_avg_reward = last.get("return_ema", last.get("ep_return_mean"))
    seed_checkpoint_raw = manifest.get("checkpoint_path")
    seed_checkpoint = seed_checkpoint_raw if isinstance(seed_checkpoint_raw, str) else None
    eval_summary = _safe_read_json(eval_summary_path)

    train_cfg = _safe_read_json(run_root / "config" / "train.json") or {}
    fixed_tick_hz = train_cfg.get("fixed_tick_hz")
    effective_minutes = effective_game_minutes_per_wall_minute(throughput, fixed_tick_hz)

    status: dict[str, Any] = {
        "root": str(run_root),
        "checkpoint_path": seed_checkpoint,
        "seed_env_steps": seed_env_steps,
        "current_env_steps": current_env_steps,
        "current_avg_episode_reward": current_avg_reward,
        "iteration": last.get("iteration"),
        "fixed_tick_hz": int(fixed_tick_hz) if isinstance(fixed_tick_hz, (int, float)) else None,
        "throughput_fps": throughput,
        "effective_game_minutes_per_wall_minute": effective_minutes,
        "eval_summary_path": str(eval_summary_path),
        "eval_present": eval_summary is not None,
    }

    if current_env_steps is None:
        status["recommended_action"] = "inspect_run_setup"
        status["reason"] = "No ppo_history.json rows yet — the run has not completed an iteration."
        return status

    delta_env_steps = max(0, current_env_steps - seed_env_steps)
    gates = _gate_rows(seed_env_steps, current_env_steps, step_gates)
    next_gate = next((gate for gate in gates if not gate["reached"]), None)
    eval_stale = _eval_is_stale(best_path, eval_summary_path)

    eta_seconds: float | None = None
    if next_gate is not None and throughput and throughput > 0.0:
        eta_seconds = next_gate["remaining_delta_steps"] / throughput

    status.update(
        {
            "delta_env_steps": int(delta_env_steps),
            "step_gates": gates,
            "next_gate_delta_steps": int(next_gate["delta_steps"]) if next_gate else None,
            "next_gate_absolute_steps": int(next_gate["absolute_steps"]) if next_gate else None,
            "remaining_to_next_gate": int(next_gate["remaining_delta_steps"]) if next_gate else 0,
            "eta_to_next_gate_seconds": None if eta_seconds is None or math.isnan(eta_seconds) else eta_seconds,
            "eval_is_stale": eval_stale,
        }
    )

    gate1 = step_gates[0]
    gate2 = step_gates[1] if len(step_gates) > 1 else gate1
    gate3 = step_gates[2] if len(step_gates) > 2 else gate2

    if delta_env_steps < gate1:
        status["recommended_action"] = "continue_training"
        status["reason"] = "The run has not reached the first checkpoint-review gate yet."
    elif delta_env_steps < gate2:
        status["recommended_action"] = "check_training_health"
        status["reason"] = "The run has crossed the first gate; review PPO telemetry and keep going unless it looks unhealthy."
    elif delta_env_steps < gate3:
        if eval_stale:
            status["recommended_action"] = "run_retained_eval"
            status["reason"] = "The run reached the first decision gate and needs a retained eval."
        else:
            status["recommended_action"] = "review_eval_and_decide"
            status["reason"] = "A retained eval exists for this gate; compare it against the seed baseline before continuing."
    else:
        if eval_stale:
            status["recommended_action"] = "stop_or_eval_now"
            status["reason"] = "The run has crossed the hard keep/kill gate without a fresh retained eval."
        else:
            status["recommended_action"] = "decide_next_change"
            status["reason"] = "The run has crossed the hard gate; use retained eval to either promote it or change one thing."

    if eval_run_dir:
        status["recommended_eval_command"] = f"docker compose -f src/docker/compose.yaml run --rm trainer agents/skills/train/scripts/train.sh {Path(eval_run_dir)}"
    else:
        status["recommended_eval_command"] = None
    status["recommended_summary_command"] = (
        f"python agents/skills/training-progress-loop/scripts/summarize_progress.py --root {run_root} --json"
    )
    return status
