"""PPO fine-tuning for the Quake v0 navigation policy."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.combat_metrics import iter_weapon_metric_keys
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import NativeVectorEnv, VectorNavigationEnv
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class PPOConfig:
    map_features_path: str
    output_dir: str
    init_ckpt: str
    observation_format: str = "symbolic"
    map_id: str = "E1M1"
    native_executable: str = ""
    native_workdir: str = ""
    fixed_tick_hz: int = 20
    native_env: Dict[str, str] = field(default_factory=dict)
    native_args: List[str] = field(default_factory=list)
    native_options: Dict[str, object] = field(default_factory=dict)
    reward_mode: str = ""
    seed: int = 11
    num_envs: int = 8
    max_steps_per_episode: int = 256
    rollout_steps: int = 128
    total_steps: int = 50_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 0.0005
    value_lr: float = 0.001
    ppo_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    bc_kl_coef: float = 0.0
    sample_temperatures: Dict[str, float] = field(default_factory=dict)
    use_gru: bool = False
    gru_hidden: int = 0
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


def _build_vector_env(config: PPOConfig) -> VectorNavigationEnv | NativeVectorEnv:
    if config.observation_format == "symbolic":
        return VectorNavigationEnv(
            num_envs=config.num_envs,
            map_features_path=config.map_features_path,
            max_steps=config.max_steps_per_episode,
            seed=config.seed,
        )

    if config.observation_format in {"world_v2", "world_v2_competitive"}:
        if not config.native_executable:
            raise RuntimeError(f"{config.observation_format} PPO requires native_executable")
        return NativeVectorEnv(
            num_envs=config.num_envs,
            executable=config.native_executable,
            map_id=config.map_id,
            max_steps=config.max_steps_per_episode,
            seed=config.seed,
            fixed_tick_hz=config.fixed_tick_hz,
            workdir=config.native_workdir or None,
            env=config.native_env,
            reward_mode=config.reward_mode,
            observation_format=config.observation_format,
            native_args=config.native_args,
            native_options=config.native_options,
        )

    raise ValueError(f"Unsupported observation_format {config.observation_format}")


def _make_action_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


_AUX_INFO_KEYS = (
    "frag_delta",
    "frag_loss",
    "monster_kill_delta",
    "damage_taken",
    "damage_dealt",
    "hit_count",
    "shots_fired",
    "health_gain",
    "armor_gain",
    "ammo_gain",
    "weapon_pickups",
    "weapon_switches",
    "visible_threats",
    "fire_pressed",
    "effective_fire",
    "blind_fire",
    "health_fraction",
    "armor_fraction",
)
_EPISODE_AUX_KEYS = (
    "episode_damage_dealt",
    "episode_hit_count",
    "episode_shots_fired",
)
_WEAPON_AUX_KEYS = tuple(iter_weapon_metric_keys())
_EPISODE_WEAPON_AUX_KEYS = tuple(
    iter_weapon_metric_keys(prefixes=(f"episode_{prefix}" for prefix in ("weapon_damage_dealt", "weapon_hits_landed", "weapon_shots_fired")))
)


def _iter_aux_metric_items(info: Dict[str, object], keys: tuple[str, ...]) -> List[tuple[str, float]]:
    pairs: List[tuple[str, float]] = []
    for key in keys:
        value = info.get(key)
        if isinstance(value, (int, float)):
            pairs.append((key, float(value)))
    return pairs


def run_ppo(config: PPOConfig) -> Dict[str, float]:
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    policy = MLPGRUPolicy.load_for_finetune(
        config.init_ckpt,
        use_gru=config.use_gru,
        gru_hidden=config.gru_hidden,
        device=config.device,
    )
    reference_policy = None
    if config.bc_kl_coef > 0.0:
        reference_policy = MLPGRUPolicy.load(config.init_ckpt, device=config.device)
    action_generator = _make_action_generator(policy.device, config.seed)
    env = _build_vector_env(config)
    try:
        obs = env.reset()
        hidden = policy.zero_hidden(config.num_envs)
        if obs.shape[1] != policy.obs_dim:
            raise RuntimeError(
                f"PPO checkpoint obs_dim={policy.obs_dim} does not match environment obs_dim={obs.shape[1]}"
            )
        obs_dim = obs.shape[1]

        steps_done = 0
        history: List[Dict[str, float]] = []

        completed_episodes = 0
        completed_goals = 0
        completion_times: List[float] = []
        item_counts: List[int] = []
        cumulative_reward_components: defaultdict[str, float] = defaultdict(float)
        cumulative_done_reasons: Counter[str] = Counter()
        cumulative_aux_metrics: defaultdict[str, float] = defaultdict(float)
        cumulative_episode_metrics: defaultdict[str, list[float]] = defaultdict(list)
        cumulative_stuck_steps = 0
        cumulative_env_steps = 0
        cumulative_movement_delta = 0.0
        cumulative_goal_progress = 0.0

        while steps_done < config.total_steps:
            rollout_initial_hidden = hidden.copy()
            obs_roll = np.empty((config.rollout_steps, config.num_envs, obs_dim), dtype=np.float32)
            rewards_roll = np.empty((config.rollout_steps, config.num_envs), dtype=np.float32)
            dones_roll = np.empty((config.rollout_steps, config.num_envs), dtype=bool)
            values_roll = np.empty((config.rollout_steps, config.num_envs), dtype=np.float32)
            logp_roll = np.empty((config.rollout_steps, config.num_envs), dtype=np.float32)
            actions_roll = {
                head: np.empty((config.rollout_steps, config.num_envs), dtype=np.int64)
                for head in ACTION_HEADS
            }
            rollout_action_counts = {head: np.zeros(size, dtype=np.int64) for head, size in ACTION_HEADS.items()}
            rollout_entropy_sums = {head: 0.0 for head in ACTION_HEADS}
            rollout_reward_components: defaultdict[str, float] = defaultdict(float)
            rollout_done_reasons: Counter[str] = Counter()
            rollout_aux_metrics: defaultdict[str, float] = defaultdict(float)
            rollout_episode_metrics: defaultdict[str, list[float]] = defaultdict(list)
            rollout_stuck_steps = 0
            rollout_movement_delta = 0.0
            rollout_goal_progress = 0.0
            rollout_env_steps = 0

            for step_idx in range(config.rollout_steps):
                action_batch = policy.act(
                    obs,
                    mode="sampled",
                    hidden=hidden,
                    generator=action_generator,
                    sample_temperatures=config.sample_temperatures,
                )
                actions = action_batch.actions
                values = action_batch.values.detach().cpu().numpy().astype(np.float32, copy=False)
                logp = action_batch.log_probs.detach().cpu().numpy().astype(np.float32, copy=False)
                next_hidden = action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)
                for head, size in ACTION_HEADS.items():
                    rollout_entropy_sums[head] += float(action_batch.entropies[head].sum().item())
                    rollout_action_counts[head] += np.bincount(actions[head], minlength=size)

                next_obs, rewards, dones, infos = env.step(actions)

                obs_roll[step_idx] = obs
                rewards_roll[step_idx] = rewards
                dones_roll[step_idx] = dones
                values_roll[step_idx] = values
                logp_roll[step_idx] = logp
                for head in ACTION_HEADS:
                    actions_roll[head][step_idx] = actions[head]

                rollout_env_steps += len(infos)
                for info in infos:
                    if bool(info.get("stuck", False)):
                        rollout_stuck_steps += 1
                    rollout_movement_delta += float(info.get("movement_delta", 0.0))
                    rollout_goal_progress += float(info.get("goal_progress", 0.0))
                    for key, value in _iter_aux_metric_items(info, _AUX_INFO_KEYS + _WEAPON_AUX_KEYS):
                        rollout_aux_metrics[key] += value
                    for key, value in info.items():
                        if key.startswith("reward_") and isinstance(value, (int, float)):
                            rollout_reward_components[key] += float(value)
                for idx, done in enumerate(dones):
                    if done:
                        completed_episodes += 1
                        goal = bool(infos[idx].get("goal_reached", False))
                        if goal:
                            completed_goals += 1
                        completion_times.append(float(infos[idx].get("steps", 0)))
                        item_counts.append(int(infos[idx].get("items_collected", 0)))
                        done_reason = str(infos[idx].get("done_reason", "")).strip() or ("goal_reached" if goal else "unknown")
                        rollout_done_reasons[done_reason] += 1
                        for key, value in _iter_aux_metric_items(
                            dict(infos[idx]),
                            _EPISODE_AUX_KEYS + _EPISODE_WEAPON_AUX_KEYS,
                        ):
                            rollout_episode_metrics[key].append(value)

                obs = next_obs
                if next_hidden.shape[0] == len(dones):
                    next_hidden[dones] = 0.0
                hidden = next_hidden
                steps_done += config.num_envs

            bootstrap_values = (
                policy.act(obs, mode="greedy", hidden=hidden).values.detach().cpu().numpy().astype(np.float32, copy=False)
            )

            advantages = _gae(
                rewards=rewards_roll,
                values=values_roll,
                dones=dones_roll,
                bootstrap=bootstrap_values,
                gamma=config.gamma,
                lam=config.gae_lambda,
            )
            returns = advantages + values_roll

            policy_losses: List[float] = []
            value_losses: List[float] = []
            total_losses: List[float] = []
            kls: List[float] = []
            clips: List[float] = []
            entropies: List[float] = []
            reference_kls: List[float] = []
            early_stop = False

            if policy.use_gru:
                adv_seq = advantages.astype(np.float32, copy=False)
                adv_mean = float(np.mean(adv_seq))
                adv_std = float(np.std(adv_seq) + 1e-6)
                adv_seq = (adv_seq - adv_mean) / adv_std
                mask_seq = np.ones((config.rollout_steps, config.num_envs), dtype=np.float32)
                if config.rollout_steps > 1:
                    mask_seq[1:] = 1.0 - dones_roll[:-1].astype(np.float32)

                obs_seq_t = torch.as_tensor(obs_roll, dtype=torch.float32, device=policy.device)
                adv_seq_t = torch.as_tensor(adv_seq, dtype=torch.float32, device=policy.device)
                ret_seq_t = torch.as_tensor(returns, dtype=torch.float32, device=policy.device)
                old_logp_seq_t = torch.as_tensor(logp_roll, dtype=torch.float32, device=policy.device)
                mask_seq_t = torch.as_tensor(mask_seq, dtype=torch.float32, device=policy.device)
                init_hidden_t = torch.as_tensor(rollout_initial_hidden, dtype=torch.float32, device=policy.device)
                act_seq_t = {
                    head: torch.as_tensor(actions_roll[head], dtype=torch.long, device=policy.device)
                    for head in ACTION_HEADS
                }
                env_batch_size = max(1, min(config.num_envs, max(1, config.minibatch_size // max(config.rollout_steps, 1))))

                for _ in range(config.ppo_epochs):
                    env_indices = np.arange(config.num_envs)
                    rng.shuffle(env_indices)
                    for start in range(0, config.num_envs, env_batch_size):
                        batch_envs = env_indices[start : start + env_batch_size]
                        mb_env_idx = torch.as_tensor(batch_envs, dtype=torch.long, device=policy.device)
                        mb_obs = obs_seq_t.index_select(1, mb_env_idx)
                        mb_actions = {head: values.index_select(1, mb_env_idx) for head, values in act_seq_t.items()}
                        mb_old_logp = old_logp_seq_t.index_select(1, mb_env_idx)
                        mb_adv = adv_seq_t.index_select(1, mb_env_idx)
                        mb_ret = ret_seq_t.index_select(1, mb_env_idx)
                        mb_masks = mask_seq_t.index_select(1, mb_env_idx)
                        mb_hidden = init_hidden_t.index_select(0, mb_env_idx)

                        step_metrics = policy.ppo_step(
                            obs=mb_obs,
                            actions=mb_actions,
                            old_log_probs=mb_old_logp,
                            advantages=mb_adv,
                            returns=mb_ret,
                            clip_ratio=config.clip_ratio,
                            policy_lr=config.policy_lr,
                            value_lr=config.value_lr,
                            value_coef=config.value_coef,
                            entropy_coef=config.entropy_coef,
                            max_grad_norm=config.max_grad_norm,
                            reference_policy=reference_policy,
                            reference_kl_coef=config.bc_kl_coef,
                            sample_temperatures=config.sample_temperatures,
                            hidden=mb_hidden,
                            masks=mb_masks,
                        )

                        policy_losses.append(step_metrics["policy_loss"])
                        value_losses.append(step_metrics["value_loss"])
                        total_losses.append(step_metrics["total_loss"])
                        entropies.append(step_metrics["entropy"])
                        reference_kls.append(step_metrics["reference_kl"])
                        kls.append(step_metrics["approx_kl"])
                        clips.append(step_metrics["clip_fraction"])
                        if config.target_kl > 0.0 and step_metrics["approx_kl"] > config.target_kl:
                            early_stop = True
                            break
                    if early_stop:
                        break
            else:
                obs_flat = obs_roll.reshape(-1, obs_dim)
                adv_flat = advantages.reshape(-1)
                ret_flat = returns.reshape(-1)
                old_logp_flat = logp_roll.reshape(-1)

                adv_mean = np.mean(adv_flat)
                adv_std = np.std(adv_flat) + 1e-6
                adv_flat = (adv_flat - adv_mean) / adv_std

                obs_flat_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=policy.device)
                adv_flat_t = torch.as_tensor(adv_flat, dtype=torch.float32, device=policy.device)
                ret_flat_t = torch.as_tensor(ret_flat, dtype=torch.float32, device=policy.device)
                old_logp_flat_t = torch.as_tensor(old_logp_flat, dtype=torch.float32, device=policy.device)
                act_flat_t = {
                    head: torch.as_tensor(actions_roll[head].reshape(-1), dtype=torch.long, device=policy.device)
                    for head in ACTION_HEADS
                }
                batch_size = obs_flat.shape[0]

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

                        step_metrics = policy.ppo_step(
                            obs=mb_obs,
                            actions=mb_actions,
                            old_log_probs=mb_old_logp,
                            advantages=mb_adv,
                            returns=mb_ret,
                            clip_ratio=config.clip_ratio,
                            policy_lr=config.policy_lr,
                            value_lr=config.value_lr,
                            value_coef=config.value_coef,
                            entropy_coef=config.entropy_coef,
                            max_grad_norm=config.max_grad_norm,
                            reference_policy=reference_policy,
                            reference_kl_coef=config.bc_kl_coef,
                            sample_temperatures=config.sample_temperatures,
                        )

                        policy_losses.append(step_metrics["policy_loss"])
                        value_losses.append(step_metrics["value_loss"])
                        total_losses.append(step_metrics["total_loss"])
                        entropies.append(step_metrics["entropy"])
                        reference_kls.append(step_metrics["reference_kl"])
                        kls.append(step_metrics["approx_kl"])
                        clips.append(step_metrics["clip_fraction"])
                        if config.target_kl > 0.0 and step_metrics["approx_kl"] > config.target_kl:
                            early_stop = True
                            break
                    if early_stop:
                        break

            cumulative_env_steps += rollout_env_steps
            cumulative_stuck_steps += rollout_stuck_steps
            cumulative_movement_delta += rollout_movement_delta
            cumulative_goal_progress += rollout_goal_progress
            cumulative_done_reasons.update(rollout_done_reasons)
            for key, value in rollout_reward_components.items():
                cumulative_reward_components[key] += value
            for key, value in rollout_aux_metrics.items():
                cumulative_aux_metrics[key] += value
            for key, values in rollout_episode_metrics.items():
                cumulative_episode_metrics[key].extend(values)

            completion_rate = (completed_goals / completed_episodes) if completed_episodes else 0.0
            death_rate = float(rollout_done_reasons.get("player_died", 0) / max(sum(rollout_done_reasons.values()), 1))
            median_time = float(np.median(completion_times[-200:])) if completion_times else 0.0
            avg_items = float(np.mean(item_counts[-200:])) if item_counts else 0.0
            rollout_steps_count = max(rollout_env_steps, 1)

            step_metrics = {
                "steps_done": float(steps_done),
                "policy_loss": float(np.mean(policy_losses) if policy_losses else 0.0),
                "value_loss": float(np.mean(value_losses) if value_losses else 0.0),
                "total_loss": float(np.mean(total_losses) if total_losses else 0.0),
                "entropy": float(np.mean(entropies) if entropies else 0.0),
                "reference_kl": float(np.mean(reference_kls) if reference_kls else 0.0),
                "approx_kl": float(np.mean(kls) if kls else 0.0),
                "clip_fraction": float(np.mean(clips) if clips else 0.0),
                "completion_rate": float(completion_rate),
                "death_rate": death_rate,
                "median_time_to_goal": float(median_time),
                "item_coverage": float(avg_items),
                "stuck_rate": float(rollout_stuck_steps / rollout_steps_count),
                "movement_delta_mean": float(rollout_movement_delta / rollout_steps_count),
                "goal_progress_mean": float(rollout_goal_progress / rollout_steps_count),
                "early_stop": bool(early_stop),
                "done_reasons": dict(sorted(rollout_done_reasons.items())),
                "reward_means": {
                    key: float(value / rollout_steps_count)
                    for key, value in sorted(rollout_reward_components.items())
                },
                "aux_metric_means": {
                    key: float(value / rollout_steps_count)
                    for key, value in sorted(rollout_aux_metrics.items())
                },
                "episode_metric_means": {
                    key: float(np.mean(values))
                    for key, values in sorted(rollout_episode_metrics.items())
                    if values
                },
                "action_entropy": {
                    head: float(rollout_entropy_sums[head] / rollout_steps_count)
                    for head in ACTION_HEADS
                },
                "action_frequencies": {
                    head: (rollout_action_counts[head] / rollout_steps_count).round(6).tolist()
                    for head in ACTION_HEADS
                },
            }
            history.append(step_metrics)

        policy.save(output / "ppo_model.npz")

        summary = {
            "steps_done": float(steps_done),
            "completion_rate": float(completed_goals / completed_episodes) if completed_episodes else 0.0,
            "episodes_completed": float(completed_episodes),
            "goals_completed": float(completed_goals),
            "death_rate": float(cumulative_done_reasons.get("player_died", 0) / max(completed_episodes, 1)),
            "median_time_to_goal": float(np.median(completion_times)) if completion_times else 0.0,
            "item_coverage": float(np.mean(item_counts)) if item_counts else 0.0,
            "policy_loss": float(np.mean([h["policy_loss"] for h in history])) if history else 0.0,
            "value_loss": float(np.mean([h["value_loss"] for h in history])) if history else 0.0,
            "entropy": float(np.mean([h["entropy"] for h in history])) if history else 0.0,
            "reference_kl": float(np.mean([h["reference_kl"] for h in history])) if history else 0.0,
            "approx_kl": float(np.mean([h["approx_kl"] for h in history])) if history else 0.0,
            "clip_fraction": float(np.mean([h["clip_fraction"] for h in history])) if history else 0.0,
            "stuck_rate": float(cumulative_stuck_steps / max(cumulative_env_steps, 1)),
            "movement_delta_mean": float(cumulative_movement_delta / max(cumulative_env_steps, 1)),
            "goal_progress_mean": float(cumulative_goal_progress / max(cumulative_env_steps, 1)),
            "done_reasons": dict(sorted(cumulative_done_reasons.items())),
            "reward_means": {
                key: float(value / max(cumulative_env_steps, 1))
                for key, value in sorted(cumulative_reward_components.items())
            },
            "aux_metric_means": {
                key: float(value / max(cumulative_env_steps, 1))
                for key, value in sorted(cumulative_aux_metrics.items())
            },
            "episode_metric_means": {
                key: float(np.mean(values))
                for key, values in sorted(cumulative_episode_metrics.items())
                if values
            },
        }
        for metric_key in (
            "frag_delta",
            "monster_kill_delta",
            "damage_taken",
            "damage_dealt",
            "hit_count",
            "shots_fired",
            "visible_threats",
            "fire_pressed",
            "effective_fire",
            "health_fraction",
            "armor_fraction",
        ):
            summary[f"{metric_key}_mean"] = float(cumulative_aux_metrics.get(metric_key, 0.0) / max(cumulative_env_steps, 1))

        write_json(output / "ppo_history.json", {"history": history})
        write_json(output / "ppo_summary.json", summary)
        write_experiment_manifest(output / "ppo_manifest.json", asdict(config), summary)

        return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
    finally:
        env.close()
