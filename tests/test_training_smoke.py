from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.evaluation import EvalConfig, run_evaluation
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.training_bc import BCConfig, run_behavior_cloning
from quake_ai.training_distill import DistillConfig, run_distillation
from quake_ai.training_rl import PPOConfig, run_ppo
from quake_ai.utils.io import read_json


def _train_world_v2_bc(collected_artifacts, output_dir: Path, seed: int) -> Path:
    run_behavior_cloning(
        BCConfig(
            map_id="E1M1",
            telemetry_path=collected_artifacts["telemetry"],
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(output_dir),
            observation_format="world_v2",
            world_ticks_path=collected_artifacts["world_ticks"],
            map_state_path=collected_artifacts["world_map"],
            seed=seed,
            epochs=1,
            patience=1,
            batch_size=32,
            trunk_hidden=64,
        )
    )
    checkpoint = output_dir / "bc_best_model.npz"
    assert checkpoint.exists()
    return checkpoint


def _train_world_v2_competitive_bc(collected_artifacts, output_dir: Path, seed: int) -> Path:
    run_behavior_cloning(
        BCConfig(
            map_id="E1M1",
            output_dir=str(output_dir),
            observation_format="world_v2_competitive",
            world_ticks_path=collected_artifacts["world_ticks"],
            map_state_path=collected_artifacts["world_map"],
            seed=seed,
            epochs=1,
            patience=1,
            batch_size=32,
            trunk_hidden=64,
        )
    )
    checkpoint = output_dir / "bc_best_model.npz"
    assert checkpoint.exists()
    return checkpoint


def _run_world_v2_ppo_smoke(
    collected_artifacts,
    tmp_path: Path,
    *,
    worker_binary: Path,
    worker_env: Mapping[str, str] | None,
    bc_seed: int,
    ppo_seed: int,
    total_steps: int,
    rollout_steps: int,
    num_envs: int,
    minibatch_size: int,
    max_steps_per_episode: int = 64,
) -> tuple[Path, dict[str, float]]:
    bc_out = tmp_path / "bc_world"
    checkpoint = _train_world_v2_bc(collected_artifacts, bc_out, seed=bc_seed)

    ppo_out = tmp_path / "ppo_world"
    metrics = run_ppo(
        PPOConfig(
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(ppo_out),
            init_ckpt=str(checkpoint),
            observation_format="world_v2",
            native_executable=str(worker_binary),
            native_env=dict(worker_env or {}),
            map_id="E1M1",
            max_steps_per_episode=max_steps_per_episode,
            total_steps=total_steps,
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            minibatch_size=minibatch_size,
            ppo_epochs=1,
            seed=ppo_seed,
        )
    )

    assert (ppo_out / "ppo_model.npz").exists()
    return ppo_out, metrics


def _run_world_v2_competitive_ppo_smoke(
    collected_artifacts,
    tmp_path: Path,
    *,
    worker_binary: Path,
    worker_env: Mapping[str, str] | None,
    bc_seed: int,
    ppo_seed: int,
    total_steps: int,
    rollout_steps: int,
    num_envs: int,
    minibatch_size: int,
    max_steps_per_episode: int = 64,
    use_gru: bool = False,
    gru_hidden: int = 0,
) -> tuple[Path, dict[str, float]]:
    bc_out = tmp_path / "bc_world_competitive"
    checkpoint = _train_world_v2_competitive_bc(collected_artifacts, bc_out, seed=bc_seed)

    ppo_out = tmp_path / "ppo_world_competitive"
    metrics = run_ppo(
        PPOConfig(
            map_features_path=collected_artifacts["map_features"],
            output_dir=str(ppo_out),
            init_ckpt=str(checkpoint),
            observation_format="world_v2_competitive",
            reward_mode="combat_survival",
            native_executable=str(worker_binary),
            native_env=dict(worker_env or {}),
            native_options={"maxplayers": 2, "deathmatch": 1, "teamplay": 0},
            map_id="E1M1",
            max_steps_per_episode=max_steps_per_episode,
            total_steps=total_steps,
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            minibatch_size=minibatch_size,
            ppo_epochs=1,
            seed=ppo_seed,
            use_gru=use_gru,
            gru_hidden=gru_hidden,
        )
    )

    assert (ppo_out / "ppo_model.npz").exists()
    return ppo_out, metrics


def _assert_loop_verification(
    ppo_out: Path,
    metrics: Mapping[str, float],
    *,
    expected_steps: int,
    min_episodes_completed: int,
    min_history_rows: int,
) -> None:
    assert math.isclose(metrics["steps_done"], float(expected_steps))
    assert metrics["episodes_completed"] >= float(min_episodes_completed)

    history = read_json(ppo_out / "ppo_history.json")["history"]
    assert len(history) >= min_history_rows
    assert math.isclose(float(history[-1]["steps_done"]), float(expected_steps))
    assert "action_entropy" in history[-1]
    assert "reward_means" in history[-1]
    assert "stuck_rate" in history[-1]


def test_value_step_keeps_shared_trunk_and_policy_heads_frozen() -> None:
    model = MLPGRUPolicy(obs_dim=8, trunk_hidden=16, device="cpu")
    obs = np.random.randn(32, 8).astype(np.float32)
    returns = np.random.randn(32).astype(np.float32)

    trunk_before = [param.detach().clone() for param in model.model.trunk.parameters()]
    policy_before = [param.detach().clone() for param in model.model.policy_heads.parameters()]
    value_before = [param.detach().clone() for param in model.model.value_head.parameters()]

    model.value_step(obs=obs, returns=returns, lr=0.01)

    trunk_after = list(model.model.trunk.parameters())
    policy_after = list(model.model.policy_heads.parameters())
    value_after = list(model.model.value_head.parameters())

    assert all(torch.equal(before, after.detach()) for before, after in zip(trunk_before, trunk_after))
    assert all(torch.equal(before, after.detach()) for before, after in zip(policy_before, policy_after))
    assert any(not torch.equal(before, after.detach()) for before, after in zip(value_before, value_after))


def test_policy_act_matches_legacy_greedy_outputs() -> None:
    model = MLPGRUPolicy(obs_dim=8, trunk_hidden=16, device="cpu")
    obs = np.random.randn(6, 8).astype(np.float32)

    logits, values, _, _ = model.forward(obs)
    legacy_actions = model.greedy_actions(logits)
    legacy_log_probs = model.log_prob(logits, legacy_actions)

    action_batch = model.act(obs, mode="greedy")

    for head in ACTION_HEADS:
        assert np.array_equal(action_batch.actions[head], legacy_actions[head])
    assert np.allclose(action_batch.values.detach().cpu().numpy(), values)
    assert np.allclose(action_batch.log_probs.detach().cpu().numpy(), legacy_log_probs)


def test_policy_forward_padding_preserves_logits_and_values() -> None:
    model = MLPGRUPolicy(obs_dim=8, trunk_hidden=16, device="cpu")
    obs = np.random.randn(5, 8).astype(np.float32)

    logits, values, _, features = model.forward(obs)

    model._rocm_inference_pad_batch = 32
    padded_logits, padded_values, _, padded_features = model.forward(obs)

    for head in logits:
        assert np.allclose(padded_logits[head], logits[head], atol=1e-6)
    assert np.allclose(padded_values, values)
    assert np.allclose(padded_features, features)


def test_policy_act_sampled_row_generators_are_deterministic() -> None:
    model = MLPGRUPolicy(obs_dim=8, trunk_hidden=16, device="cpu")
    obs = np.random.randn(4, 8).astype(np.float32)

    def _row_generators() -> list[torch.Generator]:
        generators = []
        for seed in range(100, 104):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            generators.append(generator)
        return generators

    first = model.act(obs, mode="sampled", row_generators=_row_generators())
    second = model.act(obs, mode="sampled", row_generators=_row_generators())

    for head in ACTION_HEADS:
        assert np.array_equal(first.actions[head], second.actions[head])
    assert np.allclose(first.log_probs.detach().cpu().numpy(), second.log_probs.detach().cpu().numpy())


def test_policy_recurrent_hidden_round_trip_and_migration(tmp_path: Path) -> None:
    ff_model = MLPGRUPolicy(obs_dim=8, trunk_hidden=16, device="cpu")
    checkpoint = tmp_path / "ff_policy.npz"
    ff_model.save(checkpoint)

    recurrent = MLPGRUPolicy.load_for_finetune(checkpoint, use_gru=True, gru_hidden=12, device="cpu")
    obs = np.random.randn(3, 8).astype(np.float32)
    hidden = recurrent.zero_hidden(3)

    action_batch = recurrent.act(obs, mode="sampled", hidden=hidden, generator=torch.Generator(device="cpu").manual_seed(7))

    assert recurrent.use_gru
    assert recurrent.gru_hidden == 12
    assert action_batch.next_hidden.shape == (3, 12)
    assert not np.allclose(action_batch.next_hidden.detach().cpu().numpy(), 0.0)


def test_policy_sample_temperature_biases_sampling_toward_argmax() -> None:
    model = MLPGRUPolicy(obs_dim=8, trunk_hidden=16, device="cpu")
    obs = np.zeros((512, 8), dtype=np.float32)

    with torch.no_grad():
        weapon_head = model.model.policy_heads["weapon"]
        assert isinstance(weapon_head, torch.nn.Linear)
        weapon_head.weight.zero_()
        weapon_head.bias.zero_()
        weapon_head.bias[3] = 1.5

    warm_generator = torch.Generator(device="cpu")
    warm_generator.manual_seed(123)
    cool_generator = torch.Generator(device="cpu")
    cool_generator.manual_seed(123)

    warm = model.act(obs, mode="sampled", generator=warm_generator, sample_temperatures={"weapon": 1.0})
    cool = model.act(obs, mode="sampled", generator=cool_generator, sample_temperatures={"weapon": 0.1})

    warm_argmax_rate = float(np.mean(warm.actions["weapon"] == 3))
    cool_argmax_rate = float(np.mean(cool.actions["weapon"] == 3))

    assert cool_argmax_rate > warm_argmax_rate


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


def test_bc_world_v2_smoke_run(collected_artifacts, tmp_path: Path) -> None:
    out = tmp_path / "bc_world"
    cfg = BCConfig(
        map_id="E1M1",
        telemetry_path=collected_artifacts["telemetry"],
        map_features_path=collected_artifacts["map_features"],
        output_dir=str(out),
        observation_format="world_v2",
        world_ticks_path=collected_artifacts["world_ticks"],
        map_state_path=collected_artifacts["world_map"],
        seed=3,
        epochs=1,
        patience=1,
        batch_size=32,
        trunk_hidden=64,
    )
    metrics = run_behavior_cloning(cfg)

    assert (out / "bc_best_model.npz").exists()
    assert math.isfinite(metrics["test_accuracy"])


def test_bc_world_v2_competitive_smoke_run(collected_artifacts, tmp_path: Path) -> None:
    out = tmp_path / "bc_world_competitive"
    cfg = BCConfig(
        map_id="E1M1",
        output_dir=str(out),
        observation_format="world_v2_competitive",
        world_ticks_path=collected_artifacts["world_ticks"],
        map_state_path=collected_artifacts["world_map"],
        seed=5,
        epochs=1,
        patience=1,
        batch_size=32,
        trunk_hidden=64,
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


def test_ppo_world_v2_smoke_run(collected_artifacts, native_worker_binary: Path, tmp_path: Path) -> None:
    _, metrics = _run_world_v2_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=native_worker_binary,
        worker_env=None,
        bc_seed=31,
        ppo_seed=37,
        total_steps=256,
        rollout_steps=16,
        num_envs=2,
        minibatch_size=16,
    )

    assert math.isfinite(metrics["completion_rate"])
    assert math.isfinite(metrics["value_loss"])


def test_ppo_world_v2_competitive_smoke_run(collected_artifacts, native_worker_binary: Path, tmp_path: Path) -> None:
    _, metrics = _run_world_v2_competitive_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=native_worker_binary,
        worker_env=None,
        bc_seed=35,
        ppo_seed=41,
        total_steps=256,
        rollout_steps=16,
        num_envs=2,
        minibatch_size=16,
    )

    assert math.isfinite(metrics["death_rate"])
    assert math.isfinite(metrics["frag_delta_mean"])
    assert math.isfinite(metrics["value_loss"])


def test_ppo_world_v2_competitive_recurrent_smoke_run(collected_artifacts, native_worker_binary: Path, tmp_path: Path) -> None:
    ppo_out, metrics = _run_world_v2_competitive_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=native_worker_binary,
        worker_env=None,
        bc_seed=51,
        ppo_seed=52,
        total_steps=64,
        rollout_steps=8,
        num_envs=2,
        minibatch_size=16,
        use_gru=True,
        gru_hidden=32,
    )

    _assert_loop_verification(ppo_out, metrics, expected_steps=64, min_episodes_completed=2, min_history_rows=4)
    assert math.isfinite(metrics["damage_dealt_mean"])


def test_ppo_world_v2_real_worker_smoke_run(
    collected_artifacts,
    quake_worker_binary: Path,
    quake_assets_dir: Path,
    tmp_path: Path,
) -> None:
    ppo_out, metrics = _run_world_v2_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=quake_worker_binary,
        worker_env={"QUAKE_BASEDIR": str(quake_assets_dir)},
        bc_seed=33,
        ppo_seed=39,
        total_steps=64,
        rollout_steps=8,
        num_envs=2,
        minibatch_size=8,
        max_steps_per_episode=8,
    )

    _assert_loop_verification(ppo_out, metrics, expected_steps=64, min_episodes_completed=2, min_history_rows=4)
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


def test_eval_world_v2_smoke_run(collected_artifacts, native_worker_binary: Path, tmp_path: Path) -> None:
    ppo_out, _ = _run_world_v2_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=native_worker_binary,
        worker_env=None,
        bc_seed=41,
        ppo_seed=43,
        total_steps=256,
        rollout_steps=16,
        num_envs=2,
        minibatch_size=16,
    )

    eval_out = tmp_path / "eval_world"
    metrics = run_evaluation(
        EvalConfig(
            map_features_path=collected_artifacts["map_features"],
            checkpoint_path=str(ppo_out / "ppo_model.npz"),
            output_dir=str(eval_out),
            observation_format="world_v2",
            native_executable=str(native_worker_binary),
            map_id="E1M1",
            num_episodes=8,
            num_envs=2,
            max_steps_per_episode=8,
            seed=47,
            policy_modes=["greedy"],
            start_mode="randomized",
        )
    )

    assert (eval_out / "model_card.json").exists()
    assert math.isfinite(metrics["completion_rate"])
    assert math.isfinite(metrics["stuck_rate"])


def test_eval_world_v2_competitive_smoke_run(collected_artifacts, native_worker_binary: Path, tmp_path: Path) -> None:
    ppo_out, _ = _run_world_v2_competitive_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=native_worker_binary,
        worker_env=None,
        bc_seed=43,
        ppo_seed=47,
        total_steps=256,
        rollout_steps=16,
        num_envs=2,
        minibatch_size=16,
    )

    eval_out = tmp_path / "eval_world_competitive"
    metrics = run_evaluation(
        EvalConfig(
            map_features_path=collected_artifacts["map_features"],
            checkpoint_path=str(ppo_out / "ppo_model.npz"),
            output_dir=str(eval_out),
            observation_format="world_v2_competitive",
            reward_mode="combat_survival",
            native_executable=str(native_worker_binary),
            native_options={"maxplayers": 2, "deathmatch": 1, "teamplay": 0},
            map_id="E1M1",
            num_episodes=8,
            num_envs=2,
            max_steps_per_episode=8,
            seed=53,
            policy_modes=["greedy"],
            start_mode="randomized",
        )
    )

    assert (eval_out / "model_card.json").exists()
    assert math.isfinite(metrics["death_rate"])
    assert math.isfinite(metrics["frag_delta_mean"])


def test_eval_world_v2_real_worker_smoke_run(
    collected_artifacts,
    quake_worker_binary: Path,
    quake_assets_dir: Path,
    tmp_path: Path,
) -> None:
    ppo_out, _ = _run_world_v2_ppo_smoke(
        collected_artifacts,
        tmp_path,
        worker_binary=quake_worker_binary,
        worker_env={"QUAKE_BASEDIR": str(quake_assets_dir)},
        bc_seed=45,
        ppo_seed=49,
        total_steps=64,
        rollout_steps=8,
        num_envs=2,
        minibatch_size=8,
        max_steps_per_episode=8,
    )

    eval_out = tmp_path / "eval_world_real"
    metrics = run_evaluation(
        EvalConfig(
            map_features_path=collected_artifacts["map_features"],
            checkpoint_path=str(ppo_out / "ppo_model.npz"),
            output_dir=str(eval_out),
            observation_format="world_v2",
            native_executable=str(quake_worker_binary),
            native_env={"QUAKE_BASEDIR": str(quake_assets_dir)},
            map_id="E1M1",
            num_episodes=4,
            num_envs=2,
            max_steps_per_episode=8,
            seed=51,
            policy_modes=["greedy"],
            start_mode="randomized",
        )
    )

    assert (eval_out / "model_card.json").exists()
    assert math.isfinite(metrics["completion_rate"])
    assert math.isfinite(metrics["stuck_rate"])


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
