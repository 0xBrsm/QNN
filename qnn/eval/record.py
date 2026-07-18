#!/usr/bin/env python3
"""Run a single episode for visual inspection and record a .dem file.

Usage:
    python -m qnn.eval.record --checkpoint runs/bc_v7_prod_ah2/checkpoints/bc_best_model.pth
    python -m qnn.eval.record --checkpoint path/to/model.pth --scenario-id ffa-dm4-5bot
    python -m qnn.eval.record --checkpoint path/to/model.pth --scenario-id arena-1v1-box --steps 3600

Records a Quake demo file for playback.  Prints per-episode stats.
No run directory required — reads scenario.json template directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from qnn.actions import ACTION_HEADS, CONTINUOUS_ACTION_HEADS
from qnn.env.world import NativeWorldEnv
from qnn.env.reward import RewardWeights

_TEMPLATES = Path(__file__).resolve().parent.parent / "ppo" / "templates"
_DEFAULT_WORKER = "assets/bin/ppo_worker"
_DEFAULT_ASSET_ROOT = "assets"
_DEFAULT_STEPS = 1800
_DEFAULT_TICK_HZ = 20
_DEFAULT_SEED = 42


def _load_policy(path: str, device: str) -> Any:
    from qnn.model.policy import QNNPolicy
    return QNNPolicy.load(path, device=device)


def _run_episode(
    policy: Any,
    env: NativeWorldEnv,
    max_steps: int,
    policy_mode: str,
) -> dict[str, float]:
    obs = env.reset()
    hidden = None
    # act() state-threading contract (movearch commitment lanes) — same (1,D)
    # array each step = in-place carry; fresh per episode.
    act_state = policy.prepare_act_state(1)
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
        result = policy.act(batched, mode=policy_mode, hidden=hidden, **act_state)
        hidden = result.next_hidden

        action: dict[str, Any] = {}
        for head in ACTION_HEADS:
            value = result.actions[head][0]
            if head in CONTINUOUS_ACTION_HEADS:
                action[head] = value.astype(np.float32).tolist()
            elif np.ndim(value) == 0:
                action[head] = int(value)
            else:
                action[head] = value.astype(np.int64).tolist()

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


def _gamedir(native_args: list[str]) -> str:
    for i, v in enumerate(native_args):
        if v == "-game" and i + 1 < len(native_args):
            return native_args[i + 1]
    return "id1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one episode and record a .dem file for playback",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--scenario-id", default=None, help="Scenario id from scenario.json (default: first)")
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS, help=f"Max episode steps (default: {_DEFAULT_STEPS})")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help=f"Environment seed (default: {_DEFAULT_SEED})")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--mode", default="greedy", choices=["greedy", "sampled"], help="Policy mode (default: greedy)")
    parser.add_argument("--worker", default=_DEFAULT_WORKER, help=f"Worker binary (default: {_DEFAULT_WORKER})")
    parser.add_argument("--asset-root", default=_DEFAULT_ASSET_ROOT, help=f"Asset root (default: {_DEFAULT_ASSET_ROOT})")
    parser.add_argument("--tick-hz", type=int, default=_DEFAULT_TICK_HZ, help=f"Tick rate (default: {_DEFAULT_TICK_HZ})")
    parser.add_argument("--scenario-json", default="", help=f"Scenario file (default: {_TEMPLATES / 'scenario.json'})")
    parser.add_argument("--reward-json", default="", help=f"Reward JSON (default: {_TEMPLATES / 'reward.json'})")
    parser.add_argument("--demo-name", default="", help="Demo filename stem (default: auto from checkpoint)")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    worker = Path(args.worker)
    if not worker.exists():
        raise FileNotFoundError(f"Worker binary not found: {worker}")

    asset_root = Path(args.asset_root).resolve()

    # Load scenario from template.
    scenario_path = Path(args.scenario_json) if args.scenario_json else _TEMPLATES / "scenario.json"
    scenario = json.loads(scenario_path.read_text())

    # Select scenario surface.
    scenarios = scenario.get("scenarios", [])
    surface = None
    if scenarios and args.scenario_id:
        surface = next((s for s in scenarios if s.get("scenario_id") == args.scenario_id), None)
        if surface is None:
            avail = [s["scenario_id"] for s in scenarios]
            raise ValueError(f"Unknown scenario-id {args.scenario_id!r}. Available: {avail}")
    elif scenarios:
        surface = scenarios[0]

    map_id = surface["map_id"] if surface else scenario["map_id"]
    options: dict[str, object] = dict(scenario.get("options", {}))
    if surface and "options" in surface:
        options.update(surface["options"])

    # Inject demo recording into pre_map_commands.
    demo_stem = args.demo_name or f"observe_{checkpoint.stem}"
    demo_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in demo_stem)
    record_cmd = f"record {demo_stem}.dem"
    pre_cmds = str(options.get("pre_map_commands", ""))
    options["pre_map_commands"] = f"{pre_cmds}\n{record_cmd}".strip() if pre_cmds.strip() else record_cmd

    native_args = list(scenario.get("native_args", []))
    gamedir = _gamedir(native_args)

    # Reward weights.
    reward_json = args.reward_json
    if not reward_json or not Path(reward_json).exists():
        reward_json = str(_TEMPLATES / "reward.json")
    reward_weights = RewardWeights.from_json(reward_json)

    demo_path = asset_root / gamedir / f"{demo_stem}.dem"

    print(f"Checkpoint: {checkpoint}")
    print(f"Scenario:   {args.scenario_id or '(first)'} → {map_id}")
    print(f"Mode:       {args.mode}")
    print(f"Device:     {args.device}")
    print(f"Demo:       {demo_path}")

    policy = _load_policy(str(checkpoint), device=args.device)
    print(f"  arch: d_model={policy.d_model} gru={policy.use_gru} layers={policy.n_layers}")

    env = NativeWorldEnv(
        executable=str(worker),
        map_id=map_id,
        max_steps=args.steps,
        fixed_tick_hz=args.tick_hz,
        reward_weights=reward_weights,
        mode="pvp",
        seed=args.seed,
        env={"QUAKE_BASEDIR": str(asset_root)},
        native_args=native_args,
        options=options,
    )

    print(f"Running {args.steps} steps...")
    try:
        result = _run_episode(policy, env, args.steps, args.mode)
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

    # Copy demo to pi for review.
    if demo_path.exists():
        _copy_to_pi(demo_path, f"{demo_stem}.dem")


_NAS_DEMOS = r"\\pi.local\nqdev\game\id1\common"


def _copy_to_pi(local_path: Path, filename: str) -> None:
    try:
        import smbclient
        smbclient.ClientConfig(username="guest", password="", require_secure_negotiate=False)
        smbclient.register_session(
            "pi.local", username="guest", password="",
            auth_protocol="ntlm", require_signing=False,
        )
        smbclient.makedirs(_NAS_DEMOS, exist_ok=True)
        remote = _NAS_DEMOS + "\\" + filename
        with open(local_path, "rb") as src, smbclient.open_file(remote, mode="wb") as dst:
            dst.write(src.read())
        print(f"  Copied to:    {remote}")
    except Exception as exc:
        print(f"  Pi copy failed: {exc}")


if __name__ == "__main__":
    main()
