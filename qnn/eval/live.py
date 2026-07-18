#!/usr/bin/env python3
"""Live network play — drive a trained policy against a real NQ server.

Sibling of qnn.eval.run / qnn.eval.quick.  Unlike those (which spawn an
in-process worker, simulate fixed-length episodes, and compute rewards),
this module connects to a real NetQuake server and feeds usercmds at
20Hz wall-clock for as long as the connection lasts.  Episode boundaries
are server-defined; reward is irrelevant; only the obs→policy→action
loop matters.

Usage:
    python -m qnn.eval.live \\
        --checkpoint runs/bc/<run>/checkpoints/<ckpt>.pth \\
        --server 127.0.0.1:26000

    # against a remote server, sampled with a custom temperature:
    python -m qnn.eval.live --checkpoint <ckpt> --server 10.0.0.5:26000 \\
        --mode sampled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Mapping

import numpy as np

from engine.bridge import NativeClientProcess, NativeEngineError
from qnn.actions import ACTION_HEADS
from qnn.model.policy import QNNPolicy

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTABLE = REPO_ROOT / "assets" / "bin" / "nq_client"


def _add_batch_dim(obs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.expand_dims(value, axis=0) for key, value in obs.items()}


def _select_action(
    model: QNNPolicy,
    obs: Mapping[str, np.ndarray],
    hidden: np.ndarray,
    mode: str,
    diag_log_path: str | Path | None = None,
    act_state: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Run one forward pass and convert the output into the action dict the
    bridge expects.  Mirrors the per-row extraction in qnn.eval.run."""
    action_batch = model.act(_add_batch_dim(obs), mode=mode, hidden=hidden,
                             diag_log_path=diag_log_path, **(act_state or {}))
    action: dict[str, object] = {
        "move": action_batch.actions["move"][0].astype(np.float32, copy=False).tolist(),
        "look": action_batch.actions["look"][0].astype(np.float32, copy=False).tolist(),
    }
    for head in ACTION_HEADS:
        if head in {"move", "look"}:
            continue
        action[head] = int(action_batch.actions[head][0])
    next_hidden = action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)
    return action, next_hidden


def _write_tick_record(fp: IO[str], tick: int, obs: Mapping[str, np.ndarray],
                       action: Mapping[str, object]) -> None:
    """Append a JSONL row dumping the obs we fed and the action we sent."""
    # engine_norm phase 2: bridge emits native-width keys per
    # qnn.engine_norm (no more (T, 17) self_scalars vector). We log
    # human-readable values directly from the native fields. Bit
    # flags from self_items are extracted on the fly using the
    # engine_norm bit constants so the log stays interpretable.
    from qnn import engine_norm as _en
    items = int(obs["self_items"].item() if hasattr(obs["self_items"], "item") else obs["self_items"])
    vel = obs["vel"].astype(float).tolist()
    rec = {
        "t": tick,
        "self": {
            "health":  int(obs["health"]),
            "armor":   int(obs["effective_armor"]),
            "wpn_sg":  int((items & _en.IT_SHOTGUN)         != 0),
            "wpn_ng":  int((items & _en.IT_NAILGUN)         != 0),
            "wpn_gl":  int((items & _en.IT_GRENADE_LAUNCHER)!= 0),
            "wpn_rl":  int((items & _en.IT_ROCKET_LAUNCHER) != 0),
            "wpn_lg":  int((items & _en.IT_LIGHTNING)       != 0),
            "ammo_sh": int(obs["ammo_shells"]),
            "ammo_n":  int(obs["ammo_nails"]),
            "ammo_r":  int(obs["ammo_rockets"]),
            "ammo_c":  int(obs["ammo_cells"]),
            "vel":     vel,
            "weapon_id": int(obs["self_weapon_id"]),
        },
        "n_entities": int(obs["entity_count"]),
        "action": {
            "move": list(action["move"]),
            "look": list(action["look"]),
            "attack": int(action["attack"]),
            "weapon": int(action["weapon"]),
        },
    }
    fp.write(json.dumps(rec) + "\n")
    fp.flush()


def play_live(
    *,
    checkpoint: Path,
    server: str,
    executable: Path = DEFAULT_EXECUTABLE,
    mode: str = "greedy",
    device: str = "cpu",
    asset_root: Path | None = None,
    game: str | None = None,
    tick_log_path: Path | None = None,
    model_diag_log_path: Path | None = None,
) -> int:
    """Connect, signon, and drive the policy until disconnect or interrupt.

    Returns 0 on clean shutdown (Ctrl-C), 1 on connection / runtime failure.
    """
    if not executable.exists():
        print(f"client binary not found: {executable}", file=sys.stderr)
        print("build it with: bash src/engine/build/build_nq_client.sh", file=sys.stderr)
        return 2

    model = QNNPolicy.load(checkpoint, device=device)
    hidden = model.zero_hidden(1).astype(np.float32, copy=False)
    # act() state-threading contract (movearch commitment lanes) — same (1,D)
    # array each step = in-place carry across the bridge session.
    act_state = model.prepare_act_state(1)

    env = {"QUAKE_BASEDIR": str(asset_root)} if asset_root else None
    extra_args: list[str] = []
    if game:
        # Stock NQ -game switches the gamedir search path; required for any
        # mod whose paks aren't in id1 (e.g. arena, ctf).  FFA needs nothing
        # since its only assets are server-side configs.
        extra_args.extend(["-game", game])
    client = NativeClientProcess(executable, server, env=env, extra_args=extra_args)
    print(f"connecting to {server}…", file=sys.stderr)
    tick_fp: IO[str] | None = None
    if tick_log_path is not None:
        tick_fp = tick_log_path.open("w", encoding="utf-8")
        print(f"per-tick log -> {tick_log_path}", file=sys.stderr)
    if model_diag_log_path is not None:
        print(f"model internals log -> {model_diag_log_path}", file=sys.stderr)
    try:
        obs = client.start()
        print("connected — entering step loop", file=sys.stderr)
        tick = 0
        while True:
            action, hidden = _select_action(
                model, obs, hidden, mode,
                diag_log_path=str(model_diag_log_path) if model_diag_log_path else None,
                act_state=act_state,
            )
            if tick_fp is not None:
                _write_tick_record(tick_fp, tick, obs, action)
            obs = client.step(action)
            tick += 1
    except NativeEngineError as exc:
        print(f"client exited: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted, closing client…", file=sys.stderr)
        return 0
    finally:
        client.close()
        if tick_fp is not None:
            tick_fp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to a QNNPolicy checkpoint (BC or PPO export).")
    parser.add_argument("--server", required=True,
                        help="NQ server address, e.g. 127.0.0.1:26000")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE,
                        help=f"Path to the nq_client binary (default: {DEFAULT_EXECUTABLE}).")
    parser.add_argument("--mode", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for inference (cpu, cuda, ...).")
    parser.add_argument("--asset-root", type=Path,
                        default=REPO_ROOT / "assets",
                        help="Quake basedir (containing id1/). Default: <repo>/assets.")
    parser.add_argument("--game", default=None,
                        help="Mod gamedir to load (e.g. 'arena'). Required for non-id1 mods.")
    parser.add_argument("--tick-log", type=Path, default=None,
                        help="Append a per-tick JSONL of obs+action to this path.")
    parser.add_argument("--model-diag-log", type=Path, default=None,
                        help="Append a per-tick JSONL of model head internals "
                             "(target/look/move/fire) — see QNNPolicy.act docstring.")
    args = parser.parse_args()

    raise SystemExit(play_live(
        checkpoint=args.checkpoint,
        server=args.server,
        executable=args.executable,
        mode=args.mode,
        device=args.device,
        asset_root=args.asset_root,
        game=args.game,
        tick_log_path=args.tick_log,
        model_diag_log_path=args.model_diag_log,
    ))


if __name__ == "__main__":
    main()
