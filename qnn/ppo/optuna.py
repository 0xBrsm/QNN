"""Optuna orchestration over standard PPO child runs."""

from __future__ import annotations

import json
import socket
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qnn.eval.run import EvalConfig, run_evaluation
from qnn.run.common import (
    RunnerContext,
    archive_path_if_exists,
    base_results,
    best_checkpoint,
    build_runner_context,
    finalize_results,
    materialize_child_run,
    require_cfg_mapping,
    require_cfg_string,
    require_cfg_value,
    sync_parent_machine_config,
)
from qnn.ppo.pipeline import run_pipeline as run_ppo_pipeline
from qnn.utils.io import trusted_torch_load

_EVAL_SEEDS = (23,)
_EPISODES_PER_SEED = 24
_EVAL_MAX_STEPS = 1800


def _suggest_reward_weights(trial: Any, reward_config: dict[str, Any]) -> dict[str, float | bool]:
    """Suggest reward weights from the reward.json config.

    Array values [min, max] define search ranges for Optuna.
    Scalar values are fixed and passed through unchanged.
    """
    weights: dict[str, float | bool] = {}
    for key, value in reward_config.items():
        if isinstance(value, list) and len(value) == 2:
            weights[key] = trial.suggest_float(key, float(value[0]), float(value[1]))
        else:
            weights[key] = value
    return weights


def _seed_env_steps(seed_checkpoint: str) -> int:
    seed_path = Path(seed_checkpoint)
    if seed_path.suffix != ".pth" or not seed_path.exists():
        return 0
    try:
        payload = trusted_torch_load(str(seed_path), map_location="cpu")
    except Exception:
        return 0
    if isinstance(payload, dict):
        return int(payload.get("env_steps", 0))
    return 0


def _absolute_target_total_steps(run_cfg: dict[str, Any], seed_checkpoint: str, budget_steps: int) -> int:
    machine = dict(run_cfg["machine"])
    total_workers = int(machine["num_workers"])
    containers = int(machine.get("containers", 1))
    per_container_workers = total_workers // max(containers, 1)
    num_envs = per_container_workers * int(machine["num_envs_per_worker"])
    seed_total_steps = _seed_env_steps(seed_checkpoint) // max(num_envs, 1)
    budget_total_steps = int(budget_steps) // max(num_envs, 1)
    return seed_total_steps + budget_total_steps


def _prepare_optuna_outputs(ctx: RunnerContext) -> None:
    if ctx.resume:
        return
    archive_path_if_exists(ctx.run_dir / "trials")
    archive_path_if_exists(ctx.run_dir / "best_reward_weights.json")
    archive_path_if_exists(ctx.run_dir / "study_summary.json")


def _write_trial_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trial_wrapper_dir(trial_root: Path, trial_number: int) -> Path:
    return trial_root / f"trial_{trial_number}"


def _trial_run_dir(trial_dir: Path) -> Path:
    direct_run = trial_dir / "run.json"
    wrapped_run = trial_dir / "ppo" / "run.json"
    if wrapped_run.exists():
        return trial_dir / "ppo"
    if direct_run.exists():
        return trial_dir
    return trial_dir / "ppo"


def _prepare_trial_run(
    parent_run_cfg: dict[str, Any],
    trial_dir: Path,
    trial_number: int,
    seed_checkpoint: str,
    trial_budget_steps: int,
    reward_weights: Mapping[str, Any],
) -> Path:
    trial_dir.mkdir(parents=True, exist_ok=True)
    _write_trial_json(
        trial_dir / "params.json",
        {"trial_number": int(trial_number), "reward_weights": dict(reward_weights)},
    )
    _write_trial_json(trial_dir / "reward.json", dict(reward_weights))

    run_dir = _trial_run_dir(trial_dir)
    if (run_dir / "run.json").exists():
        sync_parent_machine_config(parent_run_cfg, run_dir)
        return run_dir

    absolute_target = _absolute_target_total_steps(parent_run_cfg, seed_checkpoint, trial_budget_steps)
    materialize_child_run(
        parent_run_cfg,
        run_dir,
        name="ppo",
        mode="ppo",
        checkpoint_path=seed_checkpoint,
        resume=True,
        description=f"optuna trial {trial_number}",
        trainer_overrides={"total_steps": absolute_target},
        reward_overrides=reward_weights,
        manifest_overrides={
            "parent_run": require_cfg_string(parent_run_cfg["manifest"], "name", "run.json"),
            "trial_number": int(trial_number),
        },
    )
    return run_dir


def _train_trial(trial_run_dir: Path) -> Path:
    trial_ctx = build_runner_context(trial_run_dir)
    run_ppo_pipeline(trial_ctx, post_train_eval=False, write_report=True)
    checkpoint = best_checkpoint(trial_ctx.output_dirs["checkpoints"])
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoints in {trial_ctx.output_dirs['checkpoints']}")
    return checkpoint


def _evaluate_trial(trial_ctx: RunnerContext, checkpoint_path: Path, output_root: Path) -> tuple[float, dict[str, float]]:
    scenario = require_cfg_mapping(trial_ctx.run_cfg, "scenario", "run config")
    trainer = require_cfg_mapping(trial_ctx.run_cfg, "train", "run config")
    machine = require_cfg_mapping(trial_ctx.run_cfg, "machine", "run config")
    model_config = require_cfg_mapping(trial_ctx.run_cfg, "model", "run config")

    native_args = list(scenario.get("native_args", []))
    options = dict(scenario.get("options", {}))
    scenarios = scenario.get("scenarios", [])
    if scenarios:
        selected = scenarios[0]
        if not isinstance(selected, Mapping):
            raise RuntimeError("scenario.json.scenarios entries must be JSON objects")
        map_id = str(selected.get("map_id", scenario.get("map_id", "")))
        if "options" in selected and isinstance(selected["options"], Mapping):
            options.update(dict(selected["options"]))
    else:
        map_id = str(scenario.get("map_id", ""))

    num_envs = int(machine.get("eval_num_envs", _EPISODES_PER_SEED))
    num_episodes = int(machine.get("eval_num_episodes", _EPISODES_PER_SEED))
    fixed_tick_hz = int(trainer.get("fixed_tick_hz", 20))
    max_steps = int(trainer.get("max_steps_per_episode", _EVAL_MAX_STEPS))

    per_seed: list[float] = []
    merged: dict[str, float] = {}
    for seed in _EVAL_SEEDS:
        output_dir = output_root / f"seed_{seed}"
        config = EvalConfig(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            map_id=map_id,
            native_executable=str(trial_ctx.worker_binary),
            native_workdir=str(trainer.get("native_workdir", "")),
            fixed_tick_hz=fixed_tick_hz,
            native_env={"QUAKE_BASEDIR": str(trial_ctx.asset_root)},
            native_args=native_args,
            options=options,
            mode=str(trainer.get("mode", "pvp")),
            seed=seed,
            num_episodes=num_episodes,
            num_envs=num_envs,
            max_steps_per_episode=max_steps,
            policy_modes=["greedy"],
            start_mode="sequential",
            holdout_seed_offset=10000,
            sample_seed_offset=20000,
            map_features_path="",
            procgen=None,
            scenario_config_path="",
            reward_json_path=str(trial_ctx.run_cfg["config_paths"]["reward"]),
            parallel_policy_modes=False,
            device="cpu",
        )
        summary = run_evaluation(config, model_config=model_config)
        # Frags per minute: frag_delta_mean (frags/step) × tick_hz × 60.
        # Normalizes across episode lengths and tick rates.
        frag_per_min = float(summary.get("frag_delta_mean", 0.0)) * fixed_tick_hz * 60
        per_seed.append(frag_per_min)
        merged.update({str(key): float(value) for key, value in summary.items() if isinstance(value, (int, float))})

    metric = statistics.median(per_seed)
    diagnostics = {
        "frags_per_minute": metric,
        "frag_delta_mean": merged.get("frag_delta_mean", 0.0),
        "death_rate": merged.get("death_rate", 0.0),
        "damage_dealt_mean": merged.get("damage_dealt_mean", 0.0),
        "stuck_rate": merged.get("stuck_rate", 0.0),
    }
    return metric, diagnostics


def run(ctx: RunnerContext) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna mode requires optuna: pip install optuna") from exc

    manifest = ctx.manifest
    seed_checkpoint = require_cfg_string(ctx.run_cfg, "checkpoint_path", "run config")
    if not seed_checkpoint:
        raise RuntimeError("run.json.checkpoint_path must be non-empty when mode is 'optuna'")

    trial_count = int(require_cfg_value(manifest, "trial_count", "run.json"))
    trial_budget_steps = int(require_cfg_value(manifest, "trial_budget_steps", "run.json"))
    study_name = require_cfg_string(manifest, "study_name", "run.json")
    storage = require_cfg_string(manifest, "storage", "run.json")
    if ctx.resume and not storage:
        raise RuntimeError("optuna resume requires run.json.storage so study state is stable across launches")

    _prepare_optuna_outputs(ctx)

    trial_root = ctx.run_dir / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    def _reload_run_cfg() -> dict[str, Any]:
        """Re-read all frozen configs from disk so mid-sweep changes take effect."""
        from qnn.run.config import load_run_config
        return load_run_config(ctx.run_dir)

    def objective(trial: Any) -> float:
        live_cfg = _reload_run_cfg()
        reward_weights = _suggest_reward_weights(trial, require_cfg_mapping(live_cfg, "reward", "run config"))
        trial_dir = _trial_wrapper_dir(trial_root, int(trial.number))
        trial_run_dir = _prepare_trial_run(
            live_cfg,
            trial_dir,
            int(trial.number),
            seed_checkpoint,
            trial_budget_steps,
            reward_weights,
        )

        train_started = time.monotonic()
        try:
            ppo_ckpt = _train_trial(trial_run_dir)
        except Exception as exc:
            _write_trial_json(
                trial_dir / "diagnostics.json",
                {"trial_number": int(trial.number), "status": "train_fail", "error": str(exc)[:200]},
            )
            return float("-inf")
        train_time = time.monotonic() - train_started

        try:
            metric, diagnostics = _evaluate_trial(
                build_runner_context(trial_run_dir),
                ppo_ckpt,
                trial_dir / "eval",
            )
        except Exception as exc:
            _write_trial_json(
                trial_dir / "diagnostics.json",
                {"trial_number": int(trial.number), "status": "eval_fail", "error": str(exc)[:200]},
            )
            return float("-inf")

        _write_trial_json(
            trial_dir / "diagnostics.json",
            {
                "trial_number": int(trial.number),
                "status": "ok",
                "metric": metric,
                "train_seconds": train_time,
                **diagnostics,
            },
        )
        try:
            trial.set_user_attr("train_seconds", train_time)
            for key, value in diagnostics.items():
                trial.set_user_attr(str(key), value)
        except Exception:
            pass  # Another container may have finished this trial already
        return float(metric)

    # The native trainer rewrites ppo_history.json once per PPO iteration
    # (seconds at production topologies) — 5 minutes of silence means dead.
    _STALE_SECONDS = 300

    def _is_trial_orphaned(trial_number: int) -> bool:
        """Check if a RUNNING trial's container is dead via history staleness."""
        trial_dir = _trial_wrapper_dir(trial_root, trial_number)
        run_dir = _trial_run_dir(trial_dir)
        history = run_dir / "checkpoints" / "ppo_history.json"
        if not history.exists():
            # No history yet — check if trial started long enough ago to have one
            return (time.time() - trial_dir.stat().st_mtime) > _STALE_SECONDS
        return (time.time() - history.stat().st_mtime) > _STALE_SECONDS

    def _find_orphan() -> int | None:
        """Find one orphaned RUNNING trial. Returns trial number or None."""
        for frozen in study.trials:
            if frozen.state != optuna.trial.TrialState.RUNNING:
                continue
            if not _is_trial_orphaned(int(frozen.number)):
                continue
            return int(frozen.number)
        return None

    def _run_orphan(trial_number: int) -> None:
        """Resume and complete an orphaned trial."""
        frozen = next(t for t in study.trials if t.number == trial_number)
        trial_id = getattr(frozen, "_trial_id", None)
        if trial_id is None:
            storage_obj = getattr(study, "_storage", None)
            study_id = getattr(study, "_study_id", None)
            if storage_obj and study_id and hasattr(storage_obj, "get_trial_id_from_study_id_trial_number"):
                trial_id = storage_obj.get_trial_id_from_study_id_trial_number(study_id, trial_number)
        if trial_id is None:
            print(f"[optuna] Could not resolve trial_id for orphan {trial_number}, skipping")
            return
        try:
            recovered_trial = optuna.trial.Trial(study, int(trial_id))
            value = objective(recovered_trial)
            study.tell(recovered_trial, value, skip_if_finished=True)
            print(f"[optuna] Recovered orphan trial {trial_number}: metric={value:.4f}")
        except Exception as exc:
            print(f"[optuna] Recovery of trial {trial_number} failed: {exc}")

    container_id = socket.gethostname()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage or None,
        direction="maximize",
        load_if_exists=bool(storage and ctx.resume),
    )

    machine = require_cfg_mapping(ctx.run_cfg, "machine", "run config")
    containers = int(machine.get("containers", 1))
    trials_per_container = max(1, int(trial_count) // max(containers, 1))

    started = time.monotonic()
    completed = 0

    print(f"[optuna] Container {container_id}: up to {trials_per_container} trials")
    for _ in range(trials_per_container):
        # Check for orphaned trials before starting a new one
        orphan = _find_orphan() if ctx.resume and storage else None
        if orphan is not None:
            print(f"[optuna] Recovering orphan trial {orphan} before starting new trial")
            _run_orphan(orphan)
        else:
            trial = study.ask()
            value = objective(trial)
            study.tell(trial, value)
        completed += 1
    stage_timings["optuna"] = time.monotonic() - started

    best_payload = {
        "study_name": study_name,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }
    _write_trial_json(ctx.run_dir / "best_reward_weights.json", best_payload)
    _write_trial_json(
        ctx.run_dir / "study_summary.json",
        {
            **best_payload,
            "trial_count": int(len(study.trials)),
            "storage": storage,
        },
    )

    results["optuna"] = {
        **best_payload,
        "trial_count": int(len(study.trials)),
        "trials_dir": str(trial_root),
    }
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
