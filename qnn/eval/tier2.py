"""Tier-2 closed-loop geometry battery — substrate A/B at native decode.

Drives a checkpoint through solo freeplay episodes on the corpus maps
and scores GEOMETRY-MEDIATED behavior only (qnn.eval.geometry): no aim
or attack scoring, hence no decode-fit dependence — decode comes from a
bundled regime template (default ``a25base``, neutral operating point,
identical for every arm). Pose comes from the QNN_POSE_TAIL worker
channel; for probe-grid checkpoints the same pose drives sim-side
probe-grid obs assembly (qnn.eval.probe_table) from an exported per-map
table, so ego and probe-grid arms run through one harness.

Usage:
    python -m qnn.eval.tier2 --checkpoint <best.pth> \
        --maps dm2,dm6,dm4 --episodes 8 --steps 1200 \
        [--probe-tables probe_tables.npz --k 4] \
        --output tier2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from qnn.env.reward import RewardWeights
from qnn.env.world import NativeWorldEnv
from qnn.eval.geometry import GeometryAccumulator, aggregate
from qnn.utils.io import write_json
from qnn.wire import unpack_obs_buffer_native

_POSE_TAIL_BYTES = 16


def _zero_reward_weights() -> RewardWeights:
    return RewardWeights(
        death_penalty=0.0, ehp_delta_weight=0.0, edp_delta_weight=0.0,
        frag_bonus=0.0, fire_penalty=0.0, self_damage_penalty=0.0,
        tracking_weight=0.0, tracking_fov=0.0, tracking_penalty=0.0,
    )


def run_episode(
    model, env: NativeWorldEnv, *, steps: int, episode_seed: int,
    probes=None, k: int = 0, tick_hz: int = 20,
) -> dict[str, float]:
    from qnn.eval.run import (
        _commit_reset_lanes, _pad_entities_to_max, _seed_attack_rng,
        _stack_obs,
    )
    from qnn.eval.probe_table import assign_closed_loop
    import torch

    env.reset(seed=episode_seed)
    acc = GeometryAccumulator(tick_hz=tick_hz)
    hidden = model.zero_hidden(1)
    attack_state = np.zeros((1, 1), dtype=np.float32)
    attack_rng = _seed_attack_rng(episode_seed)
    move_commit = np.asarray(_commit_reset_lanes(), dtype=np.float32)
    rng = torch.Generator().manual_seed(episode_seed)

    # Prime: one no-op step to get the first raw obs + pose.
    env.step_send({"move": (0.0, 0.0, 0.0), "look": (0.0, 0.0, 0.0)})
    for _ in range(steps):
        try:
            raw, _reward, done, _info = env.step_recv_raw()
        except (RuntimeError, ValueError, UnicodeDecodeError) as exc:
            # Worker protocol desync (rare, episode-specific engine event
            # mid-stream). The pose ticks accumulated so far are valid;
            # end the episode loudly rather than killing the battery.
            print(f"  [tier2] episode truncated at tick {len(acc.poses)}: "
                  f"{type(exc).__name__}: {exc}")
            break
        pose = np.frombuffer(raw[-_POSE_TAIL_BYTES:], dtype="<f4").astype(np.float64)
        acc.add(pose)
        obs = _pad_entities_to_max(unpack_obs_buffer_native(raw))
        if probes is not None:
            pano, offsets = assign_closed_loop(probes, pose[:3], pose[3], k)
            obs["probe_atlas"] = pano
            obs["probe_offsets"] = offsets
        obs_b = _stack_obs([obs])
        sticky = {"attack_state": attack_state, "attack_rng": attack_rng}
        if getattr(model, "move_commitment", False):
            sticky["move_commit_state"] = move_commit[None, :]
        batch = model.act(
            obs_b, mode="sampled", hidden=np.stack([hidden], axis=0),
            row_generators=[rng], **sticky,
        )
        hidden = np.asarray(batch.next_hidden)[0].copy() if batch.next_hidden is not None else hidden
        action = {
            "move": batch.actions["move"][0].astype(np.float32, copy=False).tolist(),
            "look": batch.actions["look"][0].astype(np.float32, copy=False).tolist(),
            **{h: int(batch.actions[h][0])
               for h in ("attack", "jump", "weapon")
               if h in batch.actions},
        }
        if done:
            break
        env.step_send(action)
    return acc.finish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--worker", default="assets/bin/ppo_worker")
    parser.add_argument("--maps", default="dm2,dm6,dm4")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--decode-regime", default="a25base")
    parser.add_argument("--probe-tables", type=Path, default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--tick-hz", type=int, default=20)
    parser.add_argument("--look-grid", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    # Look polar grid: no code default — install from the checkpoint run's
    # frozen config (checkpoints/ sibling), same as every other entry point.
    grid_path = Path(args.checkpoint).parent.parent / "config" / "look_grid.json"
    if args.look_grid is not None:
        grid_path = args.look_grid
    if grid_path.exists():
        import torch
        from qnn.model.look_bins import install_polar_grid
        lg = json.loads(grid_path.read_text())
        install_polar_grid(
            torch.tensor(lg["mag_centers_rad"], dtype=torch.float32),
            torch.tensor(lg["dir_centers_rad"], dtype=torch.float32)
            if "dir_centers_rad" in lg else None,
            deadzone_rad=lg.get("deadzone_rad"),
        )
    else:
        raise SystemExit(f"look_grid.json not found at {grid_path} "
                         f"(pass --look-grid)")

    from types import SimpleNamespace

    from qnn.eval.run import (
        _apply_decode_config_params, _install_decode_regime, _load_checkpoint,
    )
    model = _load_checkpoint(args.checkpoint, device="cpu")
    resolved = _install_decode_regime(model, args.decode_regime)
    if resolved is not None:
        # Sources the move/look operating point from the regime onto the
        # model (move_commitment, aim gains, ...). The config shim absorbs
        # the one provenance field this harness doesn't keep.
        _apply_decode_config_params(SimpleNamespace(), model, resolved)

    tables = None
    if args.probe_tables is not None:
        from qnn.eval.probe_table import load_tables
        tables, meta = load_tables(args.probe_tables)
        print(f"probe tables: {args.probe_tables} "
              f"(spacing {meta['spacing']}, k={args.k})")

    env_vars = {
        "QUAKE_BASEDIR": os.path.abspath("assets"),
        "QNN_POSE_TAIL": "1",
    }
    summary: dict[str, dict] = {}
    for map_id in args.maps.split(","):
        per_map = []
        for ep in range(args.episodes):
            env = NativeWorldEnv(
                executable=args.worker, map_id=map_id, max_steps=args.steps + 8,
                fixed_tick_hz=args.tick_hz, reward_weights=_zero_reward_weights(),
                mode="freeplay", seed=args.seed + ep, env=env_vars,
                native_args=[], options={},
            )
            try:
                probes = tables[map_id] if tables is not None else None
                per_map.append(run_episode(
                    model, env, steps=args.steps,
                    episode_seed=args.seed + 1000 * ep,
                    probes=probes, k=args.k, tick_hz=args.tick_hz,
                ))
            finally:
                env.adapter.close()
            print(f"{map_id} ep{ep}: " + ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(per_map[-1].items())))
        summary[map_id] = aggregate(per_map)

    overall = aggregate([
        {**m, "ticks": m["ticks"]} for map_summary in summary.values()
        for m in [map_summary]
    ])
    result = {
        "checkpoint": str(args.checkpoint),
        "decode_regime": args.decode_regime,
        "probe_tables": str(args.probe_tables) if args.probe_tables else None,
        "k": args.k if args.probe_tables else None,
        "episodes_per_map": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "per_map": summary,
        "overall": overall,
    }
    print(json.dumps(result["overall"], indent=1))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
