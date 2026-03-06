"""Policy evaluation and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from quake_ai.actions import ACTION_HEADS
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import E1M1NavigationEnv
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class EvalConfig:
    map_features_path: str
    checkpoint_path: str
    output_dir: str
    seed: int = 19
    num_episodes: int = 100
    max_steps_per_episode: int = 256
    policy_modes: List[str] = field(default_factory=lambda: ["greedy"])
    start_mode: str = "sequential"
    holdout_seed_offset: int = 10_000
    sample_seed_offset: int = 20_000
    device: str = "auto"


def _episode_specs(config: EvalConfig) -> List[Tuple[int, int | None]]:
    if config.start_mode == "sequential":
        return [(config.seed + episode, episode) for episode in range(config.num_episodes)]
    if config.start_mode == "randomized":
        rng = np.random.default_rng(config.seed + config.holdout_seed_offset)
        return [(int(rng.integers(0, 2**31 - 1)), None) for _ in range(config.num_episodes)]
    raise ValueError(f"Unsupported start_mode {config.start_mode}")


def _select_action(
    model: MLPGRUPolicy,
    obs: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> Mapping[str, int]:
    logits, _, _, _ = model.forward(obs.reshape(1, -1))
    if mode == "greedy":
        selected = model.greedy_actions(logits)
    elif mode == "sampled":
        selected, _, _ = model.sample_actions(logits, rng)
    else:
        raise ValueError(f"Unsupported policy mode {mode}")
    return {head: int(selected[head][0]) for head in ACTION_HEADS}


def _evaluate_mode(
    config: EvalConfig,
    model: MLPGRUPolicy,
    mode: str,
    episode_specs: Sequence[Tuple[int, int | None]],
) -> Dict[str, float]:
    env = E1M1NavigationEnv(
        map_features_path=config.map_features_path,
        max_steps=config.max_steps_per_episode,
        seed=config.seed,
    )
    action_rng = np.random.default_rng(config.seed + config.sample_seed_offset + (0 if mode == "greedy" else 1))

    completion = 0
    times: List[float] = []
    items: List[int] = []
    stuck_steps = 0
    total_steps = 0

    for episode_seed, start_variant in episode_specs:
        obs = env.reset(seed=episode_seed, start_variant=start_variant)
        done = False
        step_count = 0
        last_info = {"goal_reached": False, "items_collected": 0}

        while not done and step_count < config.max_steps_per_episode:
            action = _select_action(model=model, obs=obs, mode=mode, rng=action_rng)
            obs, _, done, info = env.step(action)
            step_count += 1
            total_steps += 1
            if bool(info.get("stuck", False)):
                stuck_steps += 1
            last_info = info

        if bool(last_info.get("goal_reached", False)):
            completion += 1
        times.append(float(step_count))
        items.append(int(last_info.get("items_collected", 0)))

    completion_rate = completion / max(config.num_episodes, 1)
    median_time = float(median(times)) if times else 0.0
    item_coverage = float(np.mean(items)) if items else 0.0
    stuck_rate = float(stuck_steps / max(total_steps, 1))

    return {
        "completion_rate": completion_rate,
        "median_time_to_goal": median_time,
        "item_coverage": item_coverage,
        "stuck_rate": stuck_rate,
        "num_episodes": config.num_episodes,
    }


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

    eval_notes = [
        "V0 uses deterministic symbolic environment derived from map features.",
        "Packet traces are used for validation, not policy inputs.",
    ]
    if config.start_mode == "randomized":
        eval_notes.append("Evaluation uses held-out randomized start seeds and headings.")
    else:
        eval_notes.append("Evaluation uses fixed sequential seeds for regression tracking.")
    if set(config.policy_modes) == {"greedy"}:
        eval_notes.append("Evaluation uses greedy actions only.")
    elif set(config.policy_modes) == {"sampled"}:
        eval_notes.append("Evaluation uses stochastic action sampling only.")
    else:
        eval_notes.append("Evaluation reports both greedy and stochastic action-selection modes.")

    model_card = {
        "model": {
            "checkpoint": str(config.checkpoint_path),
            "architecture": "2-layer MLP actor-critic with shared trainable trunk",
            "observation_modality": "symbolic state features",
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
