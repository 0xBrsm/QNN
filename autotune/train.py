"""Autotune: one file, one metric, git ratchet.

Train APPO against FrikBots, evaluate net frag rate, log to TSV.
The tuning agent edits PARAMS/NOTES below, commits, runs, keeps or reverts.

Usage:
    cd /workspaces/dev-qnn/src && python -m autotune.train \
        --executable ../artifacts/bin/quake_worker \
        --basedir ../artifacts/quake \
        --bc-checkpoint ../artifacts/runs/competitive_materialized_competitive_all51/bc/bc_best_model.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# ── Mutable configuration (the ONLY thing the tuning agent edits) ────────

TRAINING_BUDGET_STEPS = 200_000

PARAMS = {
    # Architecture (LOCKED — changing these breaks BC warm-start)
    "d_model": 64,
    "n_heads": 2,
    "n_layers": 2,
    "ffn_dim": 256,
    "trunk_hidden": 128,
    "gru_hidden": 128,
    # PPO / APPO
    "learning_rate": 0.00025,
    "entropy_coef": 0.002,
    "clip_ratio": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "value_coef": 0.5,
    "bc_kl_coef": 0.05,
    "ppo_epochs": 2,
    "rollout": 64,
    "num_workers": 4,
    "max_policy_lag": 4,
    # Reward weights (metric is game-truth frags, not reward)
    "frag_bonus": 3.0,
    "death_penalty": -2.0,
    "ehp_delta_weight": 0.5,
    "edp_delta_weight": 0.6,
    # Per-head action temperatures
    "temp_move": 1.0,
    "temp_strafe": 1.0,
    "temp_look_yaw": 1.0,
    "temp_look_pitch": 0.7,
    "temp_fire": 0.4,
    "temp_jump": 0.35,
    "temp_weapon": 0.25,
}

NOTES = "Baseline: matches the combat-bot multi-scenario verify defaults."

# ── Fixed eval scenario ──────────────────────────────────────────────────

_EVAL_SEEDS = (19, 20, 21)
_EPISODES_PER_SEED = 32
_EVAL_MAX_STEPS = 1024
_EVAL_OPTIONS = {
    "maxplayers": 2, "skill": 0, "deathmatch": 1, "coop": 0,
    "teamplay": 0, "fraglimit": 0, "timelimit": 0, "samelevel": 1,
}

# ── TSV schema ───────────────────────────────────────────────────────────

_TSV_COLUMNS = [
    "iteration", "git_sha", "metric", "frag_delta_mean",
    "damage_dealt_mean", "death_rate", "stuck_rate",
    "mean_episode_return", "status", "notes",
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _iteration_number(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    with open(results_path) as f:
        return sum(1 for _ in csv.reader(f)) - 1


def _append_result(results_path: Path, row: Dict[str, Any]) -> None:
    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _TSV_COLUMNS})


# ── Training ─────────────────────────────────────────────────────────────

def _build_sf_cfg(
    params: Dict[str, Any],
    budget_steps: int,
    executable: str,
    basedir: str,
    bc_checkpoint: str,
    output_dir: str,
    iteration: int,
    device: str,
) -> Any:
    from quake_ai.sf.train import build_sf_cfg

    options: Dict[str, Any] = {
        "maxplayers": 2, "skill": 0, "deathmatch": 1, "coop": 0,
        "teamplay": 0, "fraglimit": 0, "timelimit": 0, "samelevel": 1,
    }
    reward_keys = ("frag_bonus", "death_penalty", "ehp_delta_weight", "edp_delta_weight")
    reward_overrides = {k: params[k] for k in reward_keys if k in params}
    if reward_overrides:
        options["reward_overrides"] = reward_overrides

    temp_argv = []
    for head in ("move", "strafe", "look_yaw", "look_pitch", "fire", "jump", "weapon"):
        key = f"temp_{head}"
        if key in params:
            temp_argv.append(f"--quake_temp_{head}={params[key]}")

    # Architecture params (d_model, n_heads, n_layers, ffn_dim, trunk_hidden, gru_hidden)
    # are LOCKED and match build_sf_cfg defaults — do not forward them.
    return build_sf_cfg(
        scenario="dm4",
        num_workers=int(params.get("num_workers", 4)),
        rollout=int(params.get("rollout", 64)),
        total_env_steps=budget_steps,
        output_dir=output_dir,
        experiment=f"autotune_iter_{iteration}",
        executable=executable,
        basedir=basedir,
        options_json=json.dumps(options),
        mode="pvp",
        max_steps_per_episode=256,
        seed=17,
        device=device,
        init_checkpoint=bc_checkpoint,
        use_gru=True,
        ppo_epochs=int(params.get("ppo_epochs", 2)),
        lr=float(params.get("learning_rate", 0.00025)),
        entropy_coef=float(params.get("entropy_coef", 0.002)),
        clip_ratio=float(params.get("clip_ratio", 0.2)),
        gamma=float(params.get("gamma", 0.99)),
        gae_lambda=float(params.get("gae_lambda", 0.95)),
        max_grad_norm=float(params.get("max_grad_norm", 0.5)),
        value_coef=float(params.get("value_coef", 0.5)),
        bc_kl_coef=float(params.get("bc_kl_coef", 0.05)),
        max_policy_lag=int(params.get("max_policy_lag", 4)),
        extra_argv=temp_argv or None,
    )


def _train(
    params: Dict[str, Any],
    budget_steps: int,
    executable: str,
    basedir: str,
    bc_checkpoint: str,
    output_dir: str,
    iteration: int,
    device: str,
) -> Path:
    """Run SF APPO training. Returns path to the SF checkpoint."""
    cfg = _build_sf_cfg(
        params, budget_steps, executable, basedir,
        bc_checkpoint, output_dir, iteration, device,
    )
    from quake_ai.sf.train import run_sf

    t0 = time.monotonic()
    run_sf(cfg)
    print(f"[autotune] training completed in {time.monotonic() - t0:.0f}s")

    ckpt_dir = Path(output_dir) / f"autotune_iter_{iteration}" / "checkpoint_p0"
    checkpoints = sorted(ckpt_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth files in {ckpt_dir}")
    return checkpoints[-1]


def _convert_checkpoint(sf_checkpoint: Path, params: Dict[str, Any], output_dir: Path) -> Path:
    """Convert SF .pth → BC .npz for evaluation."""
    from quake_ai.sf.checkpoint_converter import sf_to_bc

    bc_policy = sf_to_bc(
        sf_checkpoint_path=sf_checkpoint,
        obs_dim=145,
        trunk_hidden=int(params.get("trunk_hidden", 128)),
        gru_hidden=int(params.get("gru_hidden", 128)),
        use_gru=True,
    )
    bc_path = output_dir / "eval_model.npz"
    bc_policy.save(bc_path)
    return bc_path


# ── Evaluation ───────────────────────────────────────────────────────────

def _evaluate(
    checkpoint_path: str,
    executable: str,
    basedir: str,
    output_dir: str,
    num_envs: int = 2,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate against FrikBots. Returns (metric, diagnostics).

    Metric = median net-frag-rate across 3 seeds x 32 episodes each.
    """
    from quake_ai.rl.evaluation import EvalConfig, run_evaluation

    per_seed: list[float] = []
    merged: Dict[str, float] = {}

    for seed in _EVAL_SEEDS:
        seed_dir = str(Path(output_dir) / f"seed_{seed}")
        config = EvalConfig(
            map_features_path="",
            checkpoint_path=checkpoint_path,
            output_dir=seed_dir,
            mode="pvp",
            map_id="dm4",
            native_executable=executable,
            native_env={"QUAKE_BASEDIR": basedir} if basedir else {},
            native_args=["-game", "frikbotnex"],
            options=dict(_EVAL_OPTIONS),
            seed=seed,
            num_episodes=_EPISODES_PER_SEED,
            num_envs=num_envs,
            max_steps_per_episode=_EVAL_MAX_STEPS,
            policy_modes=["greedy"],
            start_mode="sequential",
        )
        summary = run_evaluation(config)
        per_seed.append(float(summary.get("frag_delta_mean", 0.0)))
        merged.update(summary)

    metric = statistics.median(per_seed)
    diagnostics = {
        "metric": metric,
        "frag_delta_mean": merged.get("frag_delta_mean", 0.0),
        "death_rate": merged.get("death_rate", 0.0),
        "damage_dealt_mean": merged.get("damage_dealt_mean", 0.0),
        "stuck_rate": merged.get("stuck_rate", 0.0),
        "mean_episode_return": merged.get("mean_episode_return", 0.0),
    }
    return metric, diagnostics


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune: train, evaluate, log")
    parser.add_argument("--executable", required=True)
    parser.add_argument("--basedir", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--output-dir", default="../artifacts/runs/autotune")
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.tsv"
    iteration = _iteration_number(results_path)
    sha = _git_sha()

    print(f"[autotune] iteration={iteration} sha={sha}")
    print(f"[autotune] budget={TRAINING_BUDGET_STEPS} steps")
    print(f"[autotune] params={json.dumps(PARAMS, indent=2)}")

    iter_dir = output_root / f"iter_{iteration}"
    train_dir = str(iter_dir / "sf")

    def _fail(stage: str, exc: Exception) -> None:
        print(f"[autotune] {stage} FAILED: {exc}", file=sys.stderr)
        _append_result(results_path, {
            "iteration": iteration, "git_sha": sha,
            "metric": "", "status": stage, "notes": str(exc)[:200],
        })
        sys.exit(1)

    # Train
    try:
        sf_ckpt = _train(
            PARAMS, TRAINING_BUDGET_STEPS, args.executable, args.basedir,
            args.bc_checkpoint, train_dir, iteration, args.device,
        )
    except Exception as exc:
        _fail("crash", exc)

    # Convert checkpoint
    try:
        bc_ckpt = _convert_checkpoint(sf_ckpt, PARAMS, iter_dir)
    except Exception as exc:
        _fail("convert_fail", exc)

    # Evaluate
    try:
        metric, diagnostics = _evaluate(
            str(bc_ckpt), args.executable, args.basedir, str(iter_dir / "eval"),
        )
        print(f"[autotune] metric={metric:.6f}")
    except Exception as exc:
        _fail("eval_fail", exc)

    # Log
    _append_result(results_path, {
        "iteration": iteration,
        "git_sha": sha,
        "metric": f"{metric:.6f}",
        "frag_delta_mean": f"{diagnostics.get('frag_delta_mean', 0.0):.6f}",
        "damage_dealt_mean": f"{diagnostics.get('damage_dealt_mean', 0.0):.6f}",
        "death_rate": f"{diagnostics.get('death_rate', 0.0):.4f}",
        "stuck_rate": f"{diagnostics.get('stuck_rate', 0.0):.4f}",
        "mean_episode_return": f"{diagnostics.get('mean_episode_return', 0.0):.4f}",
        "status": "ok",
        "notes": NOTES[:200],
    })

    print(f"[autotune] result logged to {results_path}")
    print(f"[autotune] metric = {metric:.6f}")


if __name__ == "__main__":
    main()
