"""Step-gated status for active PPO run directories."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from quake_ai.rl.metrics import effective_game_minutes_per_wall_minute
from quake_ai.utils.io import safe_read_json

DEFAULT_STEP_GATES: tuple[int, ...] = (1_000_000, 3_000_000, 5_000_000)

_LOAD_STATE_RE = re.compile(r"Loaded experiment state at self\.train_step=(\d+), self\.env_steps=(\d+)")
_TOTAL_FRAMES_RE = re.compile(r"Total num frames: (\d+)\. Throughput: (.*?)\. Samples:")
_AVG_REWARD_RE = re.compile(r"Avg episode reward: \[\(0, '([-\d.]+)'\)\]")
_CHECKPOINT_STEP_RE = re.compile(r"(?:best|checkpoint)_\d+_(\d+)(?:_reward_[\-\d.]+)?\.pth$")

_safe_read_json = safe_read_json


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _assert_supported_layout(run_root: Path) -> None:
    legacy_dir = run_root / "checkpoints" / "ppo"
    if legacy_dir.exists():
        raise RuntimeError(
            f"Legacy PPO checkpoint layout is unsupported: {legacy_dir}. "
            "PPO artifacts now live directly under run.json.output.checkpoints."
        )


def _seed_steps_from_checkpoint_name(path: str | Path | None) -> int | None:
    if not path:
        return None
    match = _CHECKPOINT_STEP_RE.search(Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def _loaded_state_steps(log_text: str) -> tuple[int | None, int | None]:
    match = _LOAD_STATE_RE.search(log_text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _latest_total_frames(log_text: str) -> tuple[int | None, float | None]:
    frames: int | None = None
    throughput: float | None = None
    for match in _TOTAL_FRAMES_RE.finditer(log_text):
        frames = int(match.group(1))
        values: list[float] = []
        for item in match.group(2).split(","):
            _, _, raw = item.partition(":")
            raw = raw.strip()
            if not raw or raw == "nan":
                continue
            values.append(float(raw))
        throughput = float(sum(values)) if values else None
    return frames, throughput


def _latest_avg_reward(log_text: str) -> float | None:
    reward: float | None = None
    for match in _AVG_REWARD_RE.finditer(log_text):
        reward = float(match.group(1))
    return reward


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
    ppo_dir = run_root / "checkpoints"
    config_path = ppo_dir / "config.json"
    log_path = ppo_dir / "sf_log.txt"
    from quake_ai.rl.training import best_checkpoint
    best_path = best_checkpoint(run_root / "checkpoints") or (run_root / "checkpoints" / "nonexistent")
    eval_summary_path = run_root / "metrics" / "eval" / "eval_summary.json"

    config = _safe_read_json(config_path) or {}
    log_text = _read_text(log_path)
    loaded_train_step, loaded_env_steps = _loaded_state_steps(log_text)
    seed_checkpoint_raw = manifest.get("checkpoint_path")
    seed_checkpoint = seed_checkpoint_raw if isinstance(seed_checkpoint_raw, str) else None
    checkpoint_seed_steps = _seed_steps_from_checkpoint_name(seed_checkpoint)
    seed_env_steps = loaded_env_steps if loaded_env_steps is not None else checkpoint_seed_steps
    current_env_steps, throughput = _latest_total_frames(log_text)
    current_avg_reward = _latest_avg_reward(log_text)
    eval_summary = _safe_read_json(eval_summary_path)
    fixed_tick_hz = config.get("quake_fixed_tick_hz")
    effective_minutes = effective_game_minutes_per_wall_minute(throughput, fixed_tick_hz)
    with_pbt_raw = config.get("with_pbt")
    num_policies_raw = config.get("num_policies")

    status: dict[str, Any] = {
        "root": str(run_root),
        "checkpoint_path": seed_checkpoint,
        "seed_train_step": loaded_train_step,
        "seed_env_steps": seed_env_steps,
        "current_env_steps": current_env_steps,
        "current_avg_episode_reward": current_avg_reward,
        "fixed_tick_hz": int(fixed_tick_hz) if isinstance(fixed_tick_hz, (int, float)) else None,
        "throughput_fps": throughput,
        "effective_game_minutes_per_wall_minute": effective_minutes,
        "with_pbt": bool(with_pbt_raw) if isinstance(with_pbt_raw, bool) else None,
        "num_policies": int(num_policies_raw) if isinstance(num_policies_raw, (int, float)) else None,
        "eval_summary_path": str(eval_summary_path),
        "eval_present": eval_summary is not None,
    }

    if seed_env_steps is None or current_env_steps is None:
        status["recommended_action"] = "inspect_run_setup"
        status["reason"] = "Could not determine seed or current env steps from the run artifacts."
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
