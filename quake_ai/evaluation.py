"""Policy evaluation and reporting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.combat_metrics import iter_weapon_metric_keys
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import E1M1NavigationEnv, NativeWorldEnv
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class EvalConfig:
    map_features_path: str
    checkpoint_path: str
    output_dir: str
    observation_format: str = "symbolic"
    map_id: str = "E1M1"
    native_executable: str = ""
    native_workdir: str = ""
    fixed_tick_hz: int = 20
    native_env: Dict[str, str] = field(default_factory=dict)
    native_args: List[str] = field(default_factory=list)
    native_options: Dict[str, object] = field(default_factory=dict)
    reward_mode: str = ""
    seed: int = 19
    num_episodes: int = 100
    num_envs: int = 1
    max_steps_per_episode: int = 256
    policy_modes: List[str] = field(default_factory=lambda: ["greedy"])
    start_mode: str = "sequential"
    holdout_seed_offset: int = 10_000
    sample_seed_offset: int = 20_000
    device: str = "auto"


def _episode_specs(config: EvalConfig) -> List[Tuple[int, int | None]]:
    if config.start_mode == "sequential":
        if config.observation_format in {"world_v2", "world_v2_competitive"}:
            return [(config.seed + episode, None) for episode in range(config.num_episodes)]
        return [(config.seed + episode, episode) for episode in range(config.num_episodes)]
    if config.start_mode == "randomized":
        rng = np.random.default_rng(config.seed + config.holdout_seed_offset)
        return [(int(rng.integers(0, 2**31 - 1)), None) for _ in range(config.num_episodes)]
    raise ValueError(f"Unsupported start_mode {config.start_mode}")


@dataclass(slots=True)
class _EpisodeState:
    episode_index: int
    obs: np.ndarray
    step_count: int = 0
    return_value: float = 0.0
    last_info: Mapping[str, object] = field(default_factory=lambda: {"goal_reached": False, "items_collected": 0})
    rng: torch.Generator | None = None
    hidden: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))


def _build_eval_env(config: EvalConfig) -> E1M1NavigationEnv | NativeWorldEnv:
    if config.observation_format == "symbolic":
        return E1M1NavigationEnv(
            map_features_path=config.map_features_path,
            max_steps=config.max_steps_per_episode,
            seed=config.seed,
        )

    if config.observation_format in {"world_v2", "world_v2_competitive"}:
        if not config.native_executable:
            raise RuntimeError(f"{config.observation_format} evaluation requires native_executable")
        return NativeWorldEnv(
            executable=config.native_executable,
            map_id=config.map_id,
            max_steps=config.max_steps_per_episode,
            fixed_tick_hz=config.fixed_tick_hz,
            reward_mode=config.reward_mode,
            observation_format=config.observation_format,
            seed=config.seed,
            workdir=config.native_workdir or None,
            env=config.native_env,
            native_args=config.native_args,
            native_options=config.native_options,
        )

    raise ValueError(f"Unsupported observation_format {config.observation_format}")


def _episode_rng(config: EvalConfig, mode: str, episode_index: int, device: torch.device) -> torch.Generator:
    offset = 0 if mode == "greedy" else config.sample_seed_offset
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.seed + offset + episode_index)
    return generator


def _select_actions_batch(
    model: MLPGRUPolicy,
    obs_batch: np.ndarray,
    mode: str,
    states: Sequence[_EpisodeState],
    ) -> tuple[List[Mapping[str, int]], np.ndarray]:
    hidden_batch = np.stack([state.hidden for state in states], axis=0) if states else model.zero_hidden(0)
    if mode == "greedy":
        action_batch = model.act(obs_batch, mode=mode, hidden=hidden_batch)
    elif mode == "sampled":
        row_generators = [state.rng for state in states]
        if any(generator is None for generator in row_generators):
            raise RuntimeError("Sampled evaluation requires a persistent per-episode RNG")
        action_batch = model.act(obs_batch, mode=mode, hidden=hidden_batch, row_generators=row_generators)
    else:
        raise ValueError(f"Unsupported policy mode {mode}")

    batch_size = obs_batch.shape[0]
    actions = [
        {head: int(action_batch.actions[head][idx]) for head in ACTION_HEADS}
        for idx in range(batch_size)
    ]
    next_hidden = action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)
    return actions, next_hidden


def _step_env(
    env: E1M1NavigationEnv | NativeWorldEnv,
    action: Mapping[str, int],
) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
    obs, reward, done, info = env.step(action)
    return obs, reward, done, dict(info)


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


def _iter_aux_metric_items(info: Mapping[str, object], keys: tuple[str, ...]) -> List[tuple[str, float]]:
    pairs: List[tuple[str, float]] = []
    for key in keys:
        value = info.get(key)
        if isinstance(value, (int, float)):
            pairs.append((key, float(value)))
    return pairs


def _evaluate_mode(
    config: EvalConfig,
    model: MLPGRUPolicy,
    mode: str,
    episode_specs: Sequence[Tuple[int, int | None]],
) -> Dict[str, float]:
    num_envs = max(1, min(config.num_envs, config.num_episodes))
    envs = [_build_eval_env(config) for _ in range(num_envs)]
    executor = ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="nq-eval") if num_envs > 1 else None

    completion = 0
    times: List[float] = []
    items: List[int] = []
    returns: List[float] = []
    end_health: List[int] = []
    end_armor: List[int] = []
    done_reasons: Dict[str, int] = {}
    aux_metric_sums: Dict[str, float] = {}
    episode_metric_values: Dict[str, List[float]] = {}
    stuck_steps = 0
    total_steps = 0
    checked_obs_dim = False

    try:
        active: Dict[int, _EpisodeState] = {}
        next_episode = 0

        while next_episode < len(episode_specs) and len(active) < len(envs):
            episode_seed, start_variant = episode_specs[next_episode]
            obs = envs[len(active)].reset(seed=episode_seed, start_variant=start_variant)
            if not checked_obs_dim:
                if obs.shape[0] != model.obs_dim:
                    raise RuntimeError(
                        f"Evaluation checkpoint obs_dim={model.obs_dim} does not match environment obs_dim={obs.shape[0]}"
                    )
                checked_obs_dim = True
            active[len(active)] = _EpisodeState(
                episode_index=next_episode,
                obs=obs,
                rng=None if mode == "greedy" else _episode_rng(config, mode, next_episode, model.device),
                hidden=model.zero_hidden(1)[0].copy(),
            )
            next_episode += 1

        while active:
            slot_ids = sorted(active.keys())
            states = [active[slot] for slot in slot_ids]
            obs_batch = np.stack([state.obs for state in states], axis=0)
            actions, next_hidden = _select_actions_batch(
                model=model,
                obs_batch=obs_batch,
                mode=mode,
                states=states,
            )

            if executor is None:
                results = [_step_env(envs[slot], action) for slot, action in zip(slot_ids, actions)]
            else:
                futures = [
                    executor.submit(_step_env, envs[slot], action)
                    for slot, action in zip(slot_ids, actions)
                ]
                results = [future.result() for future in futures]

            for batch_idx, (slot, result) in enumerate(zip(slot_ids, results)):
                obs, reward, done, info = result
                state = active[slot]
                state.hidden = next_hidden[batch_idx].copy()
                state.step_count += 1
                # Reward is useful in combat/survival eval even when completion stays at zero.
                state.return_value += float(reward)
                state.last_info = info
                total_steps += 1
                if bool(info.get("stuck", False)):
                    stuck_steps += 1
                for key, value in _iter_aux_metric_items(info, _AUX_INFO_KEYS + _WEAPON_AUX_KEYS):
                    aux_metric_sums[key] = aux_metric_sums.get(key, 0.0) + value

                if done or state.step_count >= config.max_steps_per_episode:
                    if bool(state.last_info.get("goal_reached", False)):
                        completion += 1
                    times.append(float(state.step_count))
                    items.append(int(state.last_info.get("items_collected", 0)))
                    returns.append(float(state.return_value))
                    end_health.append(int(state.last_info.get("health", 0)))
                    end_armor.append(int(state.last_info.get("armor", 0)))
                    done_reason = str(state.last_info.get("done_reason", "")).strip() or "unknown"
                    done_reasons[done_reason] = done_reasons.get(done_reason, 0) + 1
                    for key, value in _iter_aux_metric_items(
                        state.last_info,
                        _EPISODE_AUX_KEYS + _EPISODE_WEAPON_AUX_KEYS,
                    ):
                        episode_metric_values.setdefault(key, []).append(value)

                    if next_episode < len(episode_specs):
                        episode_seed, start_variant = episode_specs[next_episode]
                        next_obs = envs[slot].reset(seed=episode_seed, start_variant=start_variant)
                        active[slot] = _EpisodeState(
                            episode_index=next_episode,
                            obs=next_obs,
                            rng=None if mode == "greedy" else _episode_rng(config, mode, next_episode, model.device),
                            hidden=model.zero_hidden(1)[0].copy(),
                        )
                        next_episode += 1
                    else:
                        del active[slot]
                else:
                    state.obs = obs
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in envs:
            env.close()

    completion_rate = completion / max(config.num_episodes, 1)
    median_time = float(median(times)) if times else 0.0
    item_coverage = float(np.mean(items)) if items else 0.0
    stuck_rate = float(stuck_steps / max(total_steps, 1))

    summary = {
        "completion_rate": completion_rate,
        "death_rate": float(done_reasons.get("player_died", 0) / max(config.num_episodes, 1)),
        "median_time_to_goal": median_time,
        "item_coverage": item_coverage,
        "mean_episode_return": float(np.mean(returns)) if returns else 0.0,
        "mean_end_health": float(np.mean(end_health)) if end_health else 0.0,
        "mean_end_armor": float(np.mean(end_armor)) if end_armor else 0.0,
        "stuck_rate": stuck_rate,
        "num_episodes": config.num_episodes,
    }
    for metric_key in _AUX_INFO_KEYS + _WEAPON_AUX_KEYS:
        summary[f"{metric_key}_mean"] = float(aux_metric_sums.get(metric_key, 0.0) / max(total_steps, 1))
    summary["episode_metric_means"] = {
        key: float(np.mean(values))
        for key, values in sorted(episode_metric_values.items())
        if values
    }
    return summary


def run_evaluation(config: EvalConfig) -> Dict[str, float]:
    set_global_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = MLPGRUPolicy.load(config.checkpoint_path, device=config.device)
    episode_specs = _episode_specs(config)
    mode_summaries = {mode: _evaluate_mode(config, model, mode, episode_specs) for mode in config.policy_modes}

    if len(config.policy_modes) == 1:
        summary: Dict[str, object] = dict(mode_summaries[config.policy_modes[0]])
    else:
        summary = {
            "start_mode": config.start_mode,
            "policy_modes": list(config.policy_modes),
            "modes": mode_summaries,
        }

    write_json(output / "eval_summary.json", summary)
    write_experiment_manifest(output / "eval_manifest.json", asdict(config), summary)

    eval_notes = []
    if config.observation_format == "world_v2":
        eval_notes.append("Evaluation uses the native worker path with WorldTickV2 observations encoded by the shared world encoder.")
        eval_notes.append("Reward shaping and episode metrics remain in Python while the worker supplies deterministic world ticks.")
    elif config.observation_format == "world_v2_competitive":
        eval_notes.append("Evaluation uses the native worker path with the competitive world_v2 encoder and combat-survival metrics.")
        eval_notes.append("Goal-distance semantics are removed from observations while reward shaping remains in Python.")
    else:
        eval_notes.append("V0 uses deterministic symbolic environment derived from map features.")
        eval_notes.append("Packet traces are used for validation, not policy inputs.")
    if config.start_mode == "randomized":
        if config.observation_format in {"world_v2", "world_v2_competitive"}:
            eval_notes.append("Evaluation uses held-out randomized reset seeds.")
        else:
            eval_notes.append("Evaluation uses held-out randomized start seeds and headings.")
    else:
        if config.observation_format in {"world_v2", "world_v2_competitive"}:
            eval_notes.append("Evaluation uses fixed sequential reset seeds for regression tracking.")
        else:
            eval_notes.append("Evaluation uses fixed sequential seeds for regression tracking.")
    if set(config.policy_modes) == {"greedy"}:
        eval_notes.append("Evaluation uses greedy actions only.")
    elif set(config.policy_modes) == {"sampled"}:
        eval_notes.append("Evaluation uses stochastic action sampling only.")
    else:
        eval_notes.append("Evaluation reports both greedy and stochastic action-selection modes.")
    if config.num_envs > 1:
        eval_notes.append(f"Evaluation parallelizes episodes across {config.num_envs} environments.")

    model_card = {
        "model": {
            "checkpoint": str(config.checkpoint_path),
            "architecture": (
                f"2-layer MLP + GRU({model.gru_hidden}) actor-critic with shared trainable trunk"
                if model.use_gru
                else "2-layer MLP actor-critic with shared trainable trunk"
            ),
            "observation_modality": (
                "world_v2 competitive encoded state"
                if config.observation_format == "world_v2_competitive"
                else ("world_v2 encoded state" if config.observation_format == "world_v2" else "symbolic state features")
            ),
            "action_space": list(ACTION_HEADS.keys()),
        },
        "evaluation": summary,
        "notes": eval_notes,
    }
    write_json(output / "model_card.json", model_card)

    if len(config.policy_modes) == 1:
        return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}

    flattened: Dict[str, float] = {}
    for mode, metrics in mode_summaries.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flattened[f"{mode}_{key}"] = float(value)
    return flattened
