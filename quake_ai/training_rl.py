"""PPO fine-tuning for the Quake v0 navigation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import VectorNavigationEnv
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class PPOConfig:
    map_features_path: str
    output_dir: str
    init_ckpt: str
    seed: int = 11
    num_envs: int = 8
    max_steps_per_episode: int = 256
    rollout_steps: int = 128
    total_steps: int = 50_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 0.01
    value_lr: float = 0.02
    ppo_epochs: int = 4
    minibatch_size: int = 256
    device: str = "auto"


def _gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, bootstrap: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_adv = np.zeros(num_envs, dtype=np.float32)
    next_value = bootstrap.astype(np.float32)

    for t in reversed(range(steps)):
        mask = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_value * mask - values[t]
        next_adv = delta + gamma * lam * mask * next_adv
        advantages[t] = next_adv
        next_value = values[t]

    return advantages


def run_ppo(config: PPOConfig) -> Dict[str, float]:
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    policy = MLPGRUPolicy.load(config.init_ckpt, device=config.device)
    env = VectorNavigationEnv(
        num_envs=config.num_envs,
        map_features_path=config.map_features_path,
        max_steps=config.max_steps_per_episode,
        seed=config.seed,
    )

    obs = env.reset()

    steps_done = 0
    history: List[Dict[str, float]] = []

    completed_episodes = 0
    completed_goals = 0
    completion_times: List[float] = []
    item_counts: List[int] = []

    while steps_done < config.total_steps:
        obs_roll = []
        rewards_roll = []
        dones_roll = []
        values_roll = []
        logp_roll = []
        actions_roll: Dict[str, List[np.ndarray]] = {head: [] for head in ACTION_HEADS}

        for _ in range(config.rollout_steps):
            logits, values, _, _ = policy.forward(obs)
            actions, logp, _ = policy.sample_actions(logits, rng)

            next_obs, rewards, dones, infos = env.step(actions)

            obs_roll.append(obs.copy())
            rewards_roll.append(rewards.copy())
            dones_roll.append(dones.copy())
            values_roll.append(values.astype(np.float32))
            logp_roll.append(logp.astype(np.float32))
            for head in ACTION_HEADS:
                actions_roll[head].append(actions[head].copy())

            for idx, done in enumerate(dones):
                if done:
                    completed_episodes += 1
                    goal = bool(infos[idx].get("goal_reached", False))
                    if goal:
                        completed_goals += 1
                    completion_times.append(float(infos[idx].get("steps", 0)))
                    item_counts.append(int(infos[idx].get("items_collected", 0)))

            obs = next_obs
            steps_done += config.num_envs

        _, bootstrap_values, _, _ = policy.forward(obs)

        rewards_arr = np.stack(rewards_roll, axis=0)
        dones_arr = np.stack(dones_roll, axis=0)
        values_arr = np.stack(values_roll, axis=0)
        old_logp_arr = np.stack(logp_roll, axis=0)

        advantages = _gae(
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            bootstrap=bootstrap_values,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )
        returns = advantages + values_arr

        obs_flat = np.reshape(np.stack(obs_roll, axis=0), (-1, obs.shape[1]))
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        old_logp_flat = old_logp_arr.reshape(-1)

        adv_mean = np.mean(adv_flat)
        adv_std = np.std(adv_flat) + 1e-6
        adv_flat = (adv_flat - adv_mean) / adv_std

        obs_flat_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=policy.device)
        adv_flat_t = torch.as_tensor(adv_flat, dtype=torch.float32, device=policy.device)
        ret_flat_t = torch.as_tensor(ret_flat, dtype=torch.float32, device=policy.device)
        old_logp_flat_t = torch.as_tensor(old_logp_flat, dtype=torch.float32, device=policy.device)
        act_flat_t = {
            head: torch.as_tensor(np.reshape(np.stack(actions_roll[head], axis=0), (-1,)), dtype=torch.long, device=policy.device)
            for head in ACTION_HEADS
        }

        batch_size = obs_flat.shape[0]
        policy_losses: List[float] = []
        value_losses: List[float] = []
        kls: List[float] = []
        clips: List[float] = []

        for _ in range(config.ppo_epochs):
            indices = np.arange(batch_size)
            rng.shuffle(indices)

            for start in range(0, batch_size, config.minibatch_size):
                mb_idx = torch.as_tensor(indices[start : start + config.minibatch_size], dtype=torch.long, device=policy.device)
                mb_obs = obs_flat_t.index_select(0, mb_idx)
                mb_actions = {head: values.index_select(0, mb_idx) for head, values in act_flat_t.items()}
                mb_old_logp = old_logp_flat_t.index_select(0, mb_idx)
                mb_adv = adv_flat_t.index_select(0, mb_idx)
                mb_ret = ret_flat_t.index_select(0, mb_idx)

                pol_metrics = policy.ppo_policy_step(
                    obs=mb_obs,
                    actions=mb_actions,
                    old_log_probs=mb_old_logp,
                    advantages=mb_adv,
                    clip_ratio=config.clip_ratio,
                    lr=config.policy_lr,
                )
                val_metrics = policy.value_step(obs=mb_obs, returns=mb_ret, lr=config.value_lr)

                policy_losses.append(pol_metrics["policy_loss"])
                value_losses.append(val_metrics["value_loss"])
                kls.append(pol_metrics["approx_kl"])
                clips.append(pol_metrics["clip_fraction"])

        completion_rate = (completed_goals / completed_episodes) if completed_episodes else 0.0
        median_time = float(np.median(completion_times[-200:])) if completion_times else 0.0
        avg_items = float(np.mean(item_counts[-200:])) if item_counts else 0.0

        step_metrics = {
            "steps_done": float(steps_done),
            "policy_loss": float(np.mean(policy_losses) if policy_losses else 0.0),
            "value_loss": float(np.mean(value_losses) if value_losses else 0.0),
            "approx_kl": float(np.mean(kls) if kls else 0.0),
            "clip_fraction": float(np.mean(clips) if clips else 0.0),
            "completion_rate": float(completion_rate),
            "median_time_to_goal": float(median_time),
            "item_coverage": float(avg_items),
        }
        history.append(step_metrics)

    policy.save(output / "ppo_model.npz")

    summary = {
        "steps_done": float(steps_done),
        "completion_rate": float(completed_goals / completed_episodes) if completed_episodes else 0.0,
        "episodes_completed": float(completed_episodes),
        "goals_completed": float(completed_goals),
        "median_time_to_goal": float(np.median(completion_times)) if completion_times else 0.0,
        "item_coverage": float(np.mean(item_counts)) if item_counts else 0.0,
        "policy_loss": float(np.mean([h["policy_loss"] for h in history])) if history else 0.0,
        "value_loss": float(np.mean([h["value_loss"] for h in history])) if history else 0.0,
    }

    write_json(output / "ppo_history.json", {"history": history})
    write_json(output / "ppo_summary.json", summary)
    write_experiment_manifest(output / "ppo_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
