#!/usr/bin/env python3
"""Run a single episode for visual inspection and record a .dem file.

Usage:
    python -m quake_ai.rl.observe runs/tracking_arena_v6_20260327 \
        --checkpoint best/best_000000336_2752512_reward_12.921.pth

Records a Quake demo file for playback. Prints per-episode stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from quake_ai.actions import ACTION_HEADS, CONTINUOUS_ACTION_HEADS
from quake_ai.rl.environment import NativeWorldEnv
from quake_ai.rl.evaluation import _load_checkpoint, _is_sf_checkpoint
from quake_ai.rl.run_config import (
    load_run_config,
    run_output_dirs,
    _require_mapping,
    _require_string,
)
from quake_ai.rl.planning import _resolve_asset_root, _validate_native_mod_assets
from quake_ai.rl.reward import RewardWeights


def _gamedir(native_args: list[str]) -> str:
    for i, v in enumerate(native_args):
        if v == "-game" and i + 1 < len(native_args):
            return native_args[i + 1]
    return "id1"


def _run_episode(
    policy: Any,
    env: NativeWorldEnv,
    max_steps: int,
    policy_mode: str,
) -> dict[str, float]:
    obs = env.reset()
    hidden = None
    total_reward = 0.0
    frags = 0
    deaths = 0
    dmg = 0.0
    dmg_self = 0.0
    steps = 0

    for step in range(max_steps):
        if isinstance(obs, dict):
            batched = {k: np.expand_dims(v, 0) for k, v in obs.items()}
        else:
            batched = np.expand_dims(obs, 0)
        result = policy.act(batched, mode=policy_mode, hidden=hidden)
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
        dmg += float(info.get("damage_dealt_other", 0))
        dmg_self += float(info.get("damage_dealt_self", 0))

        if done:
            break

    return {
        "steps": steps,
        "reward": total_reward,
        "frags": frags,
        "deaths": deaths,
        "dmg_enemy": dmg,
        "dmg_self": dmg_self,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-episode visual observation with demo recording")
    parser.add_argument("run_dir", type=Path, help="Run directory containing run.json")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path relative to run's checkpoints dir")
    parser.add_argument("--scenario-id", default=None, help="Scenario id when scenario.json defines a ladder")
    parser.add_argument("--seed", type=int, default=None, help="Override env seed")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args()

    run_cfg = load_run_config(args.run_dir.resolve())
    machine = _require_mapping(run_cfg, "machine", "run config")
    model_config = _require_mapping(run_cfg, "model", "run config")
    trainer = _require_mapping(run_cfg, "trainer", "run config")
    scenario = _require_mapping(run_cfg, "scenario", "run config")
    reward_path = run_cfg["config_paths"]["reward"]

    # Resolve paths.
    outputs = run_output_dirs(run_cfg)
    checkpoint_path = outputs["checkpoints"] / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    asset_root = _resolve_asset_root(_require_string(machine, "asset_root", "machine.json"))
    worker_path = Path.cwd() / _require_string(machine, "worker_binary", "machine.json")
    if not worker_path.exists():
        raise FileNotFoundError(f"Worker binary not found: {worker_path}")

    native_args = list(scenario.get("native_args", []))
    _validate_native_mod_assets(asset_root, native_args)

    # Select scenario surface.
    scenarios = scenario.get("scenarios", [])
    surface = None
    if scenarios and args.scenario_id:
        surface = next((s for s in scenarios if s.get("scenario_id") == args.scenario_id), None)
    elif scenarios:
        surface = scenarios[0]

    map_id = surface["map_id"] if surface else scenario["map_id"]
    options = dict(scenario.get("options", {}))
    if surface and "options" in surface:
        options.update(surface["options"])

    # Demo recording: inject record command.
    run_name = Path(run_cfg["run_dir"]).name
    demo_stem = f"{run_name}-{checkpoint_path.stem}"
    demo_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in demo_stem)
    pre_cmds = str(options.get("pre_map_commands", ""))
    record_cmd = f"record {demo_stem}.dem"
    options["pre_map_commands"] = f"{pre_cmds}\n{record_cmd}".strip() if pre_cmds.strip() else record_cmd

    gamedir = _gamedir(native_args)
    demo_path = asset_root / gamedir / f"{demo_stem}.dem"

    seed = args.seed if args.seed is not None else int(trainer.get("eval_seed", trainer.get("seed", 0)))
    max_steps = int(trainer.get("max_steps_per_episode", 1800))
    mode_str = str(trainer.get("mode", "pvp"))
    policy_mode = str(trainer.get("demo_policy_mode", "greedy"))
    native_workdir = str(trainer.get("native_workdir", ""))

    print(f"Run:        {run_cfg['run_dir']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scenario:   {map_id}")
    print(f"Mode:       {policy_mode}")
    print(f"Device:     {args.device}")
    print(f"Demo:       {demo_path}")

    policy = _load_checkpoint(str(checkpoint_path), device=args.device, model_config=model_config)
    print(f"  arch: obs_dim={policy.obs_dim} d_model={policy.d_model} gru={policy.use_gru} layers={policy.n_layers}")

    env = NativeWorldEnv(
        executable=str(worker_path),
        map_id=map_id,
        max_steps=max_steps,
        fixed_tick_hz=int(trainer.get("fixed_tick_hz", 20)),
        reward_weights=RewardWeights.from_json(str(reward_path)),
        mode=mode_str,
        seed=seed,
        env={"QUAKE_BASEDIR": str(asset_root)},
        native_args=native_args,
        options=options,
        workdir=native_workdir or None,
    )

    print(f"Running {max_steps} steps...")
    try:
        result = _run_episode(policy, env, max_steps, policy_mode)
    finally:
        env.close()

    print()
    print("=== Observation ===")
    print(f"  Steps:        {result['steps']}")
    print(f"  Reward:       {result['reward']:.2f}")
    print(f"  Frags:        {result['frags']}")
    print(f"  Deaths:       {result['deaths']}")
    print(f"  Net frags:    {result['frags'] - result['deaths']}")
    print(f"  Dmg (enemy):  {result['dmg_enemy']:.0f}")
    print(f"  Dmg (self):   {result['dmg_self']:.0f}")
    print(f"  Demo:         {demo_path}")


if __name__ == "__main__":
    main()
