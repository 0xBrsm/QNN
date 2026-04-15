#!/usr/bin/env python3
"""Quick greedy evaluation — 24 episodes, one seed, just the numbers.

Usage:
    python -m qnn.eval.quick runs/tracking_arena_v6_20260327 \
        --checkpoint best/best_000000336_2752512_reward_12.921.pth

    # Dump per-tick metrics:
    python -m qnn.eval.quick runs/... --checkpoint ... --dump-ticks /tmp/ticks
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from qnn.actions import ACTION_HEADS, CONTINUOUS_ACTION_HEADS
from qnn.env.world import NativeWorldEnv
from qnn.eval.run import EvalConfig, run_evaluation, _load_checkpoint
from qnn.env.reward import RewardWeights
from qnn.run.config import (
    load_run_config,
    run_output_dirs,
    _require_mapping,
    _require_string,
)
from qnn.env.planning import resolve_asset_root, validate_native_mod_assets


# Keys to record per tick when --dump-ticks is set.
_TICK_KEYS = (
    "tracking_cos", "frag_delta", "player_died",
    "damage_dealt", "damage_dealt_other", "damage_dealt_self",
    "damage_direct", "damage_splash",
    "hit_count", "shots_fired",
    "health_gain", "armor_gain",
)


def _run_episode(
    policy: Any,
    env: NativeWorldEnv,
    max_steps: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Run one greedy episode. Returns (summary, per_tick_data)."""
    obs = env.reset()
    hidden = None
    total_reward = 0.0
    frags = 0
    deaths = 0
    steps = 0
    ticks: dict[str, list[float]] = {k: [] for k in _TICK_KEYS}

    for step in range(max_steps):
        if isinstance(obs, dict):
            batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        else:
            batched = np.expand_dims(obs, 0)
        result = policy.act(batched, mode="greedy", hidden=hidden)
        hidden = result.next_hidden

        action: dict[str, Any] = {}
        for head in ACTION_HEADS:
            value = result.actions[head][0]
            if head in CONTINUOUS_ACTION_HEADS:
                action[head] = value.astype(np.float32).tolist()
            else:
                action[head] = int(value)

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        frags += int(info.get("frag_delta", 0))
        deaths += 1 if info.get("player_died") else 0
        for key in _TICK_KEYS:
            ticks[key].append(float(info.get(key, 0.0)))
        if done:
            break

    summary = {
        "reward": total_reward,
        "frags": frags,
        "deaths": deaths,
        "steps": steps,
    }
    return summary, ticks


def _build_env(run_cfg: dict, trainer: dict, scenario: dict, machine: dict, seed: int | None = None) -> tuple[NativeWorldEnv, str, int]:
    """Build env from run config. Returns (env, map_id, max_steps)."""
    asset_root = resolve_asset_root(_require_string(machine, "asset_root", "machine.json"))
    worker_path = Path.cwd() / _require_string(machine, "worker_binary", "machine.json")
    if not worker_path.exists():
        raise FileNotFoundError(f"Worker binary not found: {worker_path}")

    native_args = list(scenario.get("native_args", []))
    validate_native_mod_assets(asset_root, native_args)

    scenarios = scenario.get("scenarios", [])
    surface = scenarios[0] if scenarios else None
    map_id = surface["map_id"] if surface else scenario["map_id"]
    options = dict(scenario.get("options", {}))
    if surface and "options" in surface:
        options.update(surface["options"])

    max_steps = int(trainer.get("max_steps_per_episode", 1800))
    reward_weights = RewardWeights.from_json(str(run_cfg["config_paths"]["reward"]))

    env_seed = seed if seed is not None else int(trainer.get("eval_seed", trainer.get("seed", 23)))
    env = NativeWorldEnv(
        executable=str(worker_path),
        map_id=map_id,
        max_steps=max_steps,
        fixed_tick_hz=int(trainer.get("fixed_tick_hz", 20)),
        reward_weights=reward_weights,
        mode=str(trainer.get("mode", "pvp")),
        seed=env_seed,
        env={"QUAKE_BASEDIR": str(asset_root)},
        native_args=native_args,
        options=options,
        workdir=str(trainer.get("native_workdir", "")) or None,
    )
    return env, map_id, max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick greedy eval — 24 episodes, one seed, just stats")
    parser.add_argument("run_dir", type=Path, help="Run directory containing run.json")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path relative to run's checkpoints dir")
    parser.add_argument("--num-episodes", type=int, default=24)
    parser.add_argument("--dump-ticks", type=Path, default=None,
                        help="Directory to save per-tick torch tensors (one .pth per seed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Env seeds to eval (default: single seed from trainer config)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_cfg = load_run_config(args.run_dir.resolve())
    machine = _require_mapping(run_cfg, "machine", "run config")
    model_config = _require_mapping(run_cfg, "model", "run config")
    trainer = _require_mapping(run_cfg, "train", "run config")
    scenario = _require_mapping(run_cfg, "scenario", "run config")

    outputs = run_output_dirs(run_cfg)
    checkpoint_path = outputs["checkpoints"] / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    num_eps = args.num_episodes

    default_seed = int(trainer.get("eval_seed", trainer.get("seed", 23)))
    seeds = args.seeds if args.seeds is not None else [default_seed]

    if args.dump_ticks is not None:
        # Per-seed episodes with per-tick recording.
        policy = _load_checkpoint(str(checkpoint_path), device=args.device, model_config=model_config)

        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"Seeds:      {seeds}  Mode: greedy  (tick dump)")

        args.dump_ticks.mkdir(parents=True, exist_ok=True)
        all_summaries = []

        for seed in seeds:
            env, map_id, max_steps = _build_env(run_cfg, trainer, scenario, machine, seed=seed)
            summary, ticks = _run_episode(policy, env, max_steps)
            env.close()
            all_summaries.append(summary)
            tick_arrays = {k: np.array(v, dtype=np.float32) for k, v in ticks.items()}
            torch.save(tick_arrays, args.dump_ticks / f"seed_{seed:04d}.pth")

        avg = lambda key: sum(s[key] for s in all_summaries) / len(all_summaries)
        print(f"\n  Reward:  {avg('reward'):>8.1f}")
        print(f"  Frags:   {avg('frags'):>8.2f}")
        print(f"  Deaths:  {avg('deaths'):>8.2f}")
        print(f"  Episodes saved to: {args.dump_ticks}")
    else:
        # Batch mode: use the full eval pipeline (faster, no per-tick data).
        config = EvalConfig(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(outputs["metrics"] / "quickeval"),
            map_id=(scenario.get("scenarios", [{}])[0].get("map_id") or scenario.get("map_id", "")),
            native_executable=str(Path.cwd() / _require_string(machine, "worker_binary", "machine.json")),
            native_workdir=str(trainer.get("native_workdir", "")),
            native_env={"QUAKE_BASEDIR": str(resolve_asset_root(_require_string(machine, "asset_root", "machine.json")))},
            native_args=list(scenario.get("native_args", [])),
            options=dict(scenario.get("options", {})) | dict((scenario.get("scenarios", [{}])[0]).get("options", {})),
            mode=str(trainer.get("mode", "pvp")),
            seed=int(trainer.get("eval_seed", trainer.get("seed", 23))),
            num_episodes=num_eps,
            num_envs=num_eps,
            max_steps_per_episode=int(trainer.get("max_steps_per_episode", 1800)),
            policy_modes=["greedy"],
            start_mode="randomized",
            holdout_seed_offset=int(trainer.get("eval_holdout_seed_offset", 10000)),
            sample_seed_offset=0,
            map_features_path="",
            procgen=None,
            scenario_config_path="",
            fixed_tick_hz=int(trainer.get("fixed_tick_hz", 20)),
            reward_json_path=str(run_cfg["config_paths"]["reward"]),
            record_demos=False,
            parallel_policy_modes=False,
            device=args.device,
        )

        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"Episodes:   {num_eps}  Mode: greedy  Map: {config.map_id}")

        results = run_evaluation(config, model_config=model_config)

        reward = results.get("mean_episode_return", 0)
        frags = results.get("episode_frag_delta_mean", 0)
        deaths = results.get("deaths_mean", 0)
        dmg = results.get("episode_damage_dealt_mean", 0)
        hits = results.get("episode_hit_count_mean", 0)
        shots = results.get("episode_shots_fired_mean", 0)
        stuck = results.get("stuck_rate", 0)

        print(f"\n  Reward:  {reward:>8.1f}")
        print(f"  Frags:   {frags:>8.2f}")
        print(f"  Deaths:  {deaths:>8.2f}")
        print(f"  Dmg:     {dmg:>8.0f}")
        print(f"  Hits:    {hits:>8.1f}")
        print(f"  Shots:   {shots:>8.0f}")
        print(f"  Stuck:   {stuck:>7.1%}")


if __name__ == "__main__":
    main()
