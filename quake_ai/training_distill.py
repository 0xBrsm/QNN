"""Policy distillation from PPO rollouts back into a supervised checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.data.dataset import (
    Sample,
    batch_index_iter,
    class_weights,
    split_samples,
    stack_actions,
    stack_observations,
    success_proxy,
    write_split_manifest,
)
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import E1M1NavigationEnv
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class DistillConfig:
    map_features_path: str
    teacher_ckpt: str
    output_dir: str
    seed: int = 31
    num_episodes: int = 2048
    max_steps_per_episode: int = 256
    teacher_mode: str = "sampled"
    only_successful: bool = True
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    batch_size: int = 1024
    epochs: int = 8
    lr: float = 0.005
    patience: int = 3
    trunk_hidden: int = 128
    holdout_seed_offset: int = 30_000
    device: str = "auto"


def _teacher_action(
    teacher: MLPGRUPolicy,
    obs: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> Dict[str, int]:
    logits, _, _, _ = teacher.forward(obs.reshape(1, -1))
    if mode == "greedy":
        selected = teacher.greedy_actions(logits)
    elif mode == "sampled":
        selected, _, _ = teacher.sample_actions(logits, rng)
    else:
        raise ValueError(f"Unsupported teacher_mode {mode}")
    action = {head: int(selected[head][0]) for head in ACTION_HEADS}
    if float(obs[-2]) > 0.5:
        action["use"] = 1
    return action


def _collect_rollouts(config: DistillConfig) -> tuple[List[Sample], Dict[str, float]]:
    teacher = MLPGRUPolicy.load(config.teacher_ckpt, device=config.device)
    env = E1M1NavigationEnv(
        map_features_path=config.map_features_path,
        max_steps=config.max_steps_per_episode,
        seed=config.seed,
    )

    reset_rng = np.random.default_rng(config.seed + config.holdout_seed_offset)
    action_rng = np.random.default_rng(config.seed + config.holdout_seed_offset + 1)

    samples: List[Sample] = []
    successful_episodes = 0
    completion_times: List[int] = []
    item_counts: List[int] = []
    retained_episodes = 0

    for episode in range(config.num_episodes):
        episode_seed = int(reset_rng.integers(0, 2**31 - 1))
        obs = env.reset(seed=episode_seed, start_variant=None)
        episode_id = f"distill_{episode:05d}"

        episode_samples: List[Sample] = []
        done = False
        step_count = 0
        last_info: Dict[str, float | int | bool] = {"goal_reached": False, "items_collected": 0}

        while not done and step_count < config.max_steps_per_episode:
            action = _teacher_action(teacher, obs, config.teacher_mode, action_rng)
            next_obs, _, done, info = env.step(action)
            episode_samples.append(
                Sample(
                    episode_id=episode_id,
                    tick=step_count,
                    obs=obs.copy(),
                    action=action,
                    goal_progress=float(obs[12]),
                    done=done,
                )
            )
            obs = next_obs
            step_count += 1
            last_info = info

        goal_reached = bool(last_info.get("goal_reached", False))
        if goal_reached:
            successful_episodes += 1
        if goal_reached or not config.only_successful:
            samples.extend(episode_samples)
            retained_episodes += 1

        completion_times.append(step_count)
        item_counts.append(int(last_info.get("items_collected", 0)))

    summary = {
        "teacher_completion_rate": float(successful_episodes / max(config.num_episodes, 1)),
        "teacher_median_time_to_goal": float(np.median(completion_times)) if completion_times else 0.0,
        "teacher_item_coverage": float(np.mean(item_counts)) if item_counts else 0.0,
        "retained_episodes": float(retained_episodes),
        "retained_samples": float(len(samples)),
    }
    return samples, summary


def run_distillation(config: DistillConfig) -> Dict[str, float]:
    set_global_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    samples, rollout_summary = _collect_rollouts(config)
    if not samples:
        raise RuntimeError("No distillation samples were produced from the teacher policy")

    split = split_samples(samples, config.train_ratio, config.val_ratio, config.seed)
    write_split_manifest(output / "split_manifest.json", split)

    if not split.train:
        raise RuntimeError("No training samples available after distillation split")

    obs_dim = split.train[0].obs.shape[0]
    student = MLPGRUPolicy(obs_dim=obs_dim, trunk_hidden=config.trunk_hidden, seed=config.seed, device=config.device)
    weights = {
        head: torch.as_tensor(values, dtype=torch.float32, device=student.device)
        for head, values in class_weights(split.train).items()
    }
    train_obs = torch.as_tensor(stack_observations(split.train), dtype=torch.float32, device=student.device)
    train_actions = {
        head: torch.as_tensor(values, dtype=torch.long, device=student.device)
        for head, values in stack_actions(split.train).items()
    }
    val_obs = None
    val_actions = None
    if split.val:
        val_obs = torch.as_tensor(stack_observations(split.val), dtype=torch.float32, device=student.device)
        val_actions = {
            head: torch.as_tensor(values, dtype=torch.long, device=student.device)
            for head, values in stack_actions(split.val).items()
        }
    test_obs = None
    test_actions = None
    if split.test:
        test_obs = torch.as_tensor(stack_observations(split.test), dtype=torch.float32, device=student.device)
        test_actions = {
            head: torch.as_tensor(values, dtype=torch.long, device=student.device)
            for head, values in stack_actions(split.test).items()
        }
    rng = np.random.default_rng(config.seed)

    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        train_losses: List[float] = []
        train_accs: List[float] = []

        for batch_idx in batch_index_iter(len(split.train), config.batch_size, rng):
            mb_idx = torch.as_tensor(batch_idx, dtype=torch.long, device=student.device)
            obs = train_obs.index_select(0, mb_idx)
            actions = {head: values.index_select(0, mb_idx) for head, values in train_actions.items()}
            metrics = student.supervised_step(obs, actions, weights, lr=config.lr)
            train_losses.append(metrics["loss"])
            train_accs.append(metrics["accuracy"])

        val_metrics = {"loss": 0.0, "accuracy": 0.0}
        if val_obs is not None and val_actions is not None:
            val_metrics = student.evaluate_supervised(val_obs, val_actions)

        proxy = success_proxy(split.val if split.val else split.train)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses) if train_losses else 0.0),
            "train_accuracy": float(np.mean(train_accs) if train_accs else 0.0),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_success_proxy": float(proxy),
        }
        history.append(epoch_metrics)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            student.save(output / "distill_best_model.npz")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_epoch < 0:
        student.save(output / "distill_best_model.npz")

    final_model = MLPGRUPolicy.load(output / "distill_best_model.npz", device=config.device)
    test_metrics = {"loss": 0.0, "accuracy": 0.0}
    if test_obs is not None and test_actions is not None:
        test_metrics = final_model.evaluate_supervised(test_obs, test_actions)

    summary = {
        "best_epoch": float(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "num_train_samples": float(len(split.train)),
        "num_val_samples": float(len(split.val)),
        "num_test_samples": float(len(split.test)),
        "epochs_ran": float(len(history)),
        **rollout_summary,
    }

    write_json(output / "distill_history.json", {"history": history})
    write_json(output / "distill_rollout_summary.json", rollout_summary)
    write_json(output / "distill_summary.json", summary)
    write_experiment_manifest(output / "distill_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
