from __future__ import annotations

import math
from pathlib import Path

from quake_ai.evaluation import EvalConfig, run_evaluation
from quake_ai.training_bc import BCConfig, run_behavior_cloning
from quake_ai.training_distill import DistillConfig, run_distillation
from quake_ai.training_rl import PPOConfig, run_ppo


def test_bc_smoke_run(collected_artifacts, tmp_path: Path) -> None:
    out = tmp_path / "bc"
    cfg = BCConfig(
        map_id="E1M1",
        telemetry_path=collected_artifacts["telemetry"],
        map_features_path=collected_artifacts["map_features"],
        output_dir=str(out),
        seed=3,
        epochs=1,
        patience=1,
        batch_size=32,
    )
    metrics = run_behavior_cloning(cfg)

    assert (out / "bc_best_model.npz").exists()
    assert math.isfinite(metrics["test_accuracy"])


def test_ppo_smoke_run(collected_artifacts, tmp_path: Path) -> None:
    bc_out = tmp_path / "bc"
    bc_cfg = BCConfig(
        map_id="E1M1",
        telemetry_path=collected_artifacts["telemetry"],
        map_features_path=collected_artifacts["map_features"],
        output_dir=str(bc_out),
        seed=5,
        epochs=1,
        patience=1,
        batch_size=32,
    )
    run_behavior_cloning(bc_cfg)

    ppo_out = tmp_path / "ppo"
    ppo_cfg = PPOConfig(
        map_features_path=collected_artifacts["map_features"],
        output_dir=str(ppo_out),
        init_ckpt=str(bc_out / "bc_best_model.npz"),
        total_steps=1024,
        rollout_steps=32,
        num_envs=4,
        minibatch_size=64,
        ppo_epochs=2,
        seed=13,
    )
    metrics = run_ppo(ppo_cfg)

    assert (ppo_out / "ppo_model.npz").exists()
    assert math.isfinite(metrics["completion_rate"])
    assert math.isfinite(metrics["value_loss"])


def test_eval_smoke_run(collected_artifacts, tmp_path: Path) -> None:
    bc_out = tmp_path / "bc"
    run_behavior_cloning(
        BCConfig(
            map_id="E1M1",
            telemetry_path=collected_artifacts["telemetry"],
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(bc_out),
            seed=7,
            epochs=1,
            patience=1,
            batch_size=32,
        )
    )

    ppo_out = tmp_path / "ppo"
    run_ppo(
        PPOConfig(
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(ppo_out),
            init_ckpt=str(bc_out / "bc_best_model.npz"),
            total_steps=512,
            rollout_steps=32,
            num_envs=4,
            minibatch_size=64,
            ppo_epochs=1,
            seed=17,
        )
    )

    eval_out = tmp_path / "eval"
    metrics = run_evaluation(
        EvalConfig(
            map_features_path=collected_artifacts["map_features"],
            checkpoint_path=str(ppo_out / "ppo_model.npz"),
            output_dir=str(eval_out),
            num_episodes=20,
            max_steps_per_episode=64,
            seed=23,
            policy_modes=["greedy", "sampled"],
            start_mode="randomized",
        )
    )

    assert (eval_out / "model_card.json").exists()
    assert math.isfinite(metrics["greedy_completion_rate"])
    assert math.isfinite(metrics["sampled_stuck_rate"])


def test_distill_smoke_run(collected_artifacts, tmp_path: Path) -> None:
    bc_out = tmp_path / "bc"
    run_behavior_cloning(
        BCConfig(
            map_id="E1M1",
            telemetry_path=collected_artifacts["telemetry"],
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(bc_out),
            seed=7,
            epochs=1,
            patience=1,
            batch_size=32,
        )
    )

    ppo_out = tmp_path / "ppo"
    run_ppo(
        PPOConfig(
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(ppo_out),
            init_ckpt=str(bc_out / "bc_best_model.npz"),
            total_steps=512,
            rollout_steps=32,
            num_envs=4,
            minibatch_size=64,
            ppo_epochs=1,
            seed=17,
        )
    )

    distill_out = tmp_path / "distill"
    metrics = run_distillation(
        DistillConfig(
            map_features_path=collected_artifacts["map_features"],
            teacher_ckpt=str(ppo_out / "ppo_model.npz"),
            output_dir=str(distill_out),
            num_episodes=32,
            epochs=1,
            patience=1,
            batch_size=64,
            seed=29,
        )
    )

    assert (distill_out / "distill_best_model.npz").exists()
    assert math.isfinite(metrics["teacher_completion_rate"])
    assert math.isfinite(metrics["test_accuracy"])
