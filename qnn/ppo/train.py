"""PPO APPO training entry point for the Quake combat bot.

Usage (standalone):
    python -m qnn.ppo.train \\
        --algo=APPO --env=quake_combat \\
        --quake_executable=assets/bin/ppo_worker \\
        --quake_basedir=assets \\
        --quake_map_id=dm4 \\
        --num_workers=30 --rollout=256

Usage (programmatic):
    from qnn.ppo.train import register_quake_components, build_ppo_cfg, run_ppo
    register_quake_components()
    cfg = build_ppo_cfg(scenario="dm4", num_workers=8, ...)
    run_ppo(cfg)

Sample Factory hyperparameter defaults are calibrated to match the existing
PPO config (ppo_combat_bot_live.yaml) so that comparisons stay meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import make_runner
    from sample_factory.algo.utils.context import global_model_factory
    # SF 2.1.x uses the model_factory pattern; 2.0.x had register_custom_encoder
    try:
        from sample_factory.model.encoder import register_custom_encoder as _register_custom_encoder
        _HAS_REGISTER_CUSTOM_ENCODER = True
    except ImportError:
        _HAS_REGISTER_CUSTOM_ENCODER = False
except ImportError as exc:
    raise ImportError("sample-factory is required: pip install sample-factory>=2.0.0") from exc

import torch

from qnn.ppo.core import make_quake_core
from qnn.ppo.encoder import QuakeTransformerEncoder, make_quake_encoder
from qnn.ppo.env import make_quake_env
from qnn.run.metrics import effective_game_minutes_per_wall_minute
from qnn.utils.io import write_json


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_quake_encoder(cfg: Any, obs_space: Any):
    return make_quake_encoder(cfg, obs_space)


def _make_quake_core(cfg: Any, core_input_size: int):
    return make_quake_core(cfg, core_input_size)


def _allow_numpy_in_torch_load() -> None:
    """Allowlist numpy globals so ``torch.load(weights_only=True)`` works.

    PyTorch 2.6+ defaults weights_only=True, but SF checkpoints contain
    numpy scalars.  This must run in every process (main + SF subprocesses).
    """
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])


# Run at import time so SF learner subprocesses (which import this module
# indirectly via the model factory) also get the allowlist.
_allow_numpy_in_torch_load()


def _patch_sample_factory_checkpoint_loading() -> None:
    """Ensure numpy globals are allowed for SF checkpoint loading."""
    _allow_numpy_in_torch_load()



def _experiment_dir(cfg: Any) -> Path:
    return Path(getattr(cfg, "train_dir", ".")) / str(getattr(cfg, "experiment", "quake_combat"))


def _summary_dir(cfg: Any) -> Path:
    return Path(getattr(cfg, "train_dir", ".")) / ".summary"


_PPO_SCALAR_TAGS = {
    "mean_episode_return": "reward/reward",
    "episode_len_mean": "len/len",
    "true_objective_mean": "policy_stats/avg_true_objective",
    "frags_mean": "policy_stats/avg_frags",
    "deaths_mean": "policy_stats/avg_deaths",
    "damage_dealt_mean": "policy_stats/avg_damage_dealt",
    "damage_taken_mean": "policy_stats/avg_damage_taken",
    "accuracy": "policy_stats/avg_accuracy",
    "hit_count_mean": "policy_stats/avg_hits",
    "shots_fired_mean": "policy_stats/avg_shots_fired",
    "damage_per_death_mean": "policy_stats/avg_damage_per_death",
    "health_pickups_mean": "policy_stats/avg_health_pickups",
    "armor_pickups_mean": "policy_stats/avg_armor_pickups",
    "weapon_pickups_mean": "policy_stats/avg_weapon_pickups",
    "blind_fire_rate": "policy_stats/avg_blind_fire_rate",
    "stuck_rate": "policy_stats/avg_stuck_rate",
    "reward_total_mean": "policy_stats/avg_reward_total",
    "reward_frags_mean": "policy_stats/avg_reward_frags",
    "reward_deaths_mean": "policy_stats/avg_reward_deaths",
    "reward_ehp_mean": "policy_stats/avg_reward_ehp",
    "reward_edp_mean": "policy_stats/avg_reward_edp",
    "reward_tracking_mean": "policy_stats/avg_reward_tracking",
    "tracking_cos_mean": "policy_stats/avg_tracking_cos_mean",
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "entropy": "train/entropy",
    "kl_divergence": "train/kl_divergence",
    "fraction_clipped": "train/fraction_clipped",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/lr",
    "actual_learning_rate": "train/actual_lr",
    "fps": "perf/_fps",
    "sample_throughput": "perf/_sample_throughput",
}


def _read_latest_scalars(summary_dir: Path) -> Dict[str, Dict[str, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    latest: Dict[str, Dict[str, float]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        tail = events[-1]
        latest[tag] = {
            "step": float(tail.step),
            "value": float(tail.value),
            "wall_time": float(tail.wall_time),
        }
    return latest


def _policy_summary_from_scalars(
    cfg: Any,
    policy_id: int,
    latest_scalars: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    metrics: Dict[str, float] = {}
    steps_done = 0
    for tag, payload in latest_scalars.items():
        steps_done = max(steps_done, int(payload["step"]))
    for metric_name, tag in _PPO_SCALAR_TAGS.items():
        payload = latest_scalars.get(tag)
        if payload is not None:
            metrics[metric_name] = float(payload["value"])

    effective_minutes = effective_game_minutes_per_wall_minute(
        metrics.get("fps"),
        getattr(cfg, "quake_fixed_tick_hz", None),
    )
    if effective_minutes is not None:
        metrics["effective_game_minutes_per_wall_minute"] = float(effective_minutes)

    frags_mean = metrics.get("frags_mean")
    deaths_mean = metrics.get("deaths_mean")
    if frags_mean is not None and deaths_mean is not None:
        metrics["frag_delta_mean"] = float(frags_mean - deaths_mean)

    return {
        "policy_id": policy_id,
        "steps_done": steps_done,
        "metrics": metrics,
        "raw_scalars": {tag: float(payload["value"]) for tag, payload in sorted(latest_scalars.items())},
    }


def _select_reference_policy(policy_summaries: Dict[int, Dict[str, Any]]) -> int | None:
    if not policy_summaries:
        return None

    def _score(item: tuple[int, Dict[str, Any]]) -> tuple[float, int, int]:
        policy_id, payload = item
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        reward = metrics.get("true_objective_mean")
        if reward is None:
            reward = metrics.get("mean_episode_return")
        score = float(reward) if isinstance(reward, (int, float)) else float("-inf")
        return (score, int(payload.get("steps_done", 0)), -policy_id)

    return max(policy_summaries.items(), key=_score)[0]


def write_ppo_stage_artifacts(cfg: Any, status: Any, runner: Any | None = None) -> Dict[str, Any]:
    """Retain a compact PPO summary next to the experiment checkpoints."""
    experiment_dir = _experiment_dir(cfg)
    summary_root = _summary_dir(cfg)
    policy_summaries: Dict[int, Dict[str, Any]] = {}
    errors: list[str] = []

    if summary_root.exists():
        for child in sorted(summary_root.iterdir()):
            if not child.is_dir() or not child.name.isdigit():
                continue
            policy_id = int(child.name)
            try:
                policy_summary = _policy_summary_from_scalars(cfg, policy_id, _read_latest_scalars(child))
                if runner is not None:
                    env_steps = getattr(runner, "env_steps", {})
                    if isinstance(env_steps, dict):
                        policy_summary["steps_done"] = max(
                            int(policy_summary.get("steps_done", 0)),
                            int(env_steps.get(policy_id, 0)),
                        )
                policy_summaries[policy_id] = policy_summary
            except Exception as exc:  # pragma: no cover - defensive artifact export
                errors.append(f"{child}: {exc}")

    selected_policy_id = _select_reference_policy(policy_summaries)
    selected = policy_summaries.get(selected_policy_id, {}) if selected_policy_id is not None else {}
    selected_metrics = selected.get("metrics", {}) if isinstance(selected.get("metrics"), dict) else {}
    aggregate_fps = sum(
        float(payload.get("metrics", {}).get("fps", 0.0))
        for payload in policy_summaries.values()
        if isinstance(payload.get("metrics"), dict) and isinstance(payload.get("metrics", {}).get("fps"), (int, float))
    )
    aggregate_effective_minutes = effective_game_minutes_per_wall_minute(
        aggregate_fps if aggregate_fps > 0.0 else None,
        getattr(cfg, "quake_fixed_tick_hz", None),
    )

    summary: Dict[str, Any] = {
        "status": str(status),
        "experiment": str(getattr(cfg, "experiment", "ppo")),
        "train_dir": str(getattr(cfg, "train_dir", "")),
        "experiment_dir": str(experiment_dir),
        "summary_dir": str(summary_root),
        "policy_count": len(policy_summaries),
        "selected_policy_id": selected_policy_id,
        "steps_done": selected.get("steps_done"),
        "policies": {str(policy_id): payload for policy_id, payload in policy_summaries.items()},
    }
    summary.update({key: value for key, value in selected_metrics.items() if isinstance(value, (int, float))})
    if aggregate_fps > 0.0:
        summary["aggregate_fps"] = float(aggregate_fps)
    if aggregate_effective_minutes is not None:
        summary["effective_game_minutes_per_wall_minute"] = float(aggregate_effective_minutes)
    if errors:
        summary["errors"] = errors

    manifest = {
        "stage": "ppo",
        "status": str(status),
        "experiment": str(getattr(cfg, "experiment", "ppo")),
        "output_dir": str(experiment_dir),
        "summary_dir": str(summary_root),
        "selected_policy_id": selected_policy_id,
        "metrics": {key: value for key, value in summary.items() if isinstance(value, (int, float))},
        "policy_count": len(policy_summaries),
        "policies": {
            str(policy_id): {
                "steps_done": int(payload.get("steps_done", 0)),
                "metrics": dict(payload.get("metrics", {})),
            }
            for policy_id, payload in policy_summaries.items()
        },
        "source_summary_dirs": [str(path) for path in sorted(summary_root.iterdir()) if path.is_dir()] if summary_root.exists() else [],
    }
    if errors:
        manifest["errors"] = errors

    summary_path = experiment_dir / "ppo_summary.json"
    manifest_path = experiment_dir / "ppo_manifest.json"
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)

    return {
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "selected_policy_id": selected_policy_id,
        "metrics": manifest["metrics"],
    }


def _scrub_numpy(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python types so torch.save
    produces a checkpoint loadable with weights_only=True."""
    import numpy as np

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _scrub_numpy(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _scrub_numpy(v)
    elif isinstance(obj, tuple):
        obj = tuple(_scrub_numpy(v) for v in obj)
    elif isinstance(obj, np.integer):
        obj = int(obj)
    elif isinstance(obj, np.floating):
        obj = float(obj)
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()
    return obj


def _warm_start_policy(
    pid: int,
    ckpt: str,
    exp_dir: Path,
) -> Path:
    """Seed a single policy dir from a checkpoint. Returns the dest path."""
    import shutil

    policy_dir = exp_dir / f"checkpoint_p{pid}"
    policy_dir.mkdir(parents=True, exist_ok=True)

    if ckpt.endswith(".pth"):
        # Name the seed as checkpoint_*, never best_*.  SF reserves
        # best_*.pth for its own save_best tracking — if the seed
        # occupies that name, SF's new-best writes silently collide
        # with the seed file and the checkpoint is lost.
        from qnn.utils.checkpoint_converter import migrate_modality_embed
        from qnn.utils.io import trusted_torch_load

        name = Path(ckpt).name.replace("best_", "checkpoint_")
        dest = policy_dir / name
        payload = trusted_torch_load(ckpt, map_location="cpu")
        migrated = False
        if "model" in payload:
            if migrate_modality_embed(payload["model"], optimizer=payload.get("optimizer")):
                print(f"[quake_ppo] Migrated modality_embed in {ckpt}")
                migrated = True
        if migrated:
            # Scrub numpy scalars so SF's weights_only=True torch.load works.
            _scrub_numpy(payload)
            torch.save(payload, dest)
        else:
            shutil.copy2(ckpt, dest)
        print(f"[quake_ppo] Policy {pid} warm-start copied: {ckpt} → {dest}")
        return dest
    else:
        from qnn.utils.checkpoint_converter import QNNPolicy, save_sf_format
        bc_policy = QNNPolicy.load(ckpt, device="cpu")
        dest = save_sf_format(bc_policy, policy_dir)
        print(f"[quake_ppo] Policy {pid} warm-start converted: {ckpt} → {dest}")
        return dest


def _ensure_warm_start_checkpoint(cfg: Any) -> Optional[Path]:
    """Seed SF checkpoint dirs from warm-start checkpoint(s).

    Supports both single-seed (all policies from one checkpoint) and
    multi-seed PBT (round-robin assignment from a list of checkpoints).

    Accepts either BC ``.pth`` (converted to SF format) or existing
    SF ``.pth`` (copied directly). The caller decides whether this launch
    is resuming an existing PPO experiment or seeding a fresh one.
    """
    # Multi-seed takes priority over single-seed.
    multi_raw = str(getattr(cfg, "quake_bc_checkpoints", "") or "").strip()
    multi_ckpts = [p.strip() for p in multi_raw.split(",") if p.strip()] if multi_raw else []
    single_ckpt = str(getattr(cfg, "quake_bc_checkpoint", "") or "").strip()

    if bool(getattr(cfg, "quake_resume", False)):
        print(f"[quake_ppo] Resume requested; using existing PPO checkpoints in {_experiment_dir(cfg)}")
        return None

    if not multi_ckpts and not single_ckpt:
        raise RuntimeError("Fresh PPO runs require an explicit warm-start checkpoint")

    cfg.load_checkpoint_kind = "latest"
    exp_dir = _experiment_dir(cfg)
    num_policies = int(getattr(cfg, "num_policies", 1))

    if multi_ckpts:
        print(f"[quake_ppo] Multi-seed warm-start: {len(multi_ckpts)} seed(s) across {num_policies} policies (round-robin)")
        first_path = None
        for pid in range(num_policies):
            ckpt = multi_ckpts[pid % len(multi_ckpts)]
            dest = _warm_start_policy(pid, ckpt, exp_dir)
            if pid == 0:
                first_path = dest
        return first_path
    else:
        print(f"[quake_ppo] Single-seed warm-start: {single_ckpt} → {num_policies} policies")
        first_path = None
        for pid in range(num_policies):
            dest = _warm_start_policy(pid, single_ckpt, exp_dir)
            if pid == 0:
                first_path = dest
        return first_path


def _patch_save_best_keep() -> None:
    """Fix SF's hardcoded keep=1 in save_best to use cfg.keep_checkpoints."""
    from sample_factory.algo.learning.learner import Learner

    _original_save_best = Learner.save_best

    def _save_best_keep_all(self, policy_id, metric, metric_value):
        if policy_id != self.policy_id:
            return False
        if metric_value - self.best_performance > 0.001:
            self.best_performance = metric_value
            name_suffix = f"_{metric}_{metric_value:.3f}"
            return self._save_impl("best", name_suffix, self.cfg.keep_checkpoints, verbose=False)
        return False

    Learner.save_best = _save_best_keep_all


def register_quake_components() -> None:
    """Register Quake env and encoder with Sample Factory (idempotent)."""
    _patch_save_best_keep()
    _patch_sample_factory_checkpoint_loading()
    register_env("quake_combat", make_quake_env)
    if _HAS_REGISTER_CUSTOM_ENCODER:
        # SF < 2.1: direct registration helper
        _register_custom_encoder("quake_trunk", QuakeTransformerEncoder)  # type: ignore[name-defined]
    else:
        # SF 2.1+: register via model factory
        global_model_factory().register_encoder_factory(_make_quake_encoder)  # type: ignore[name-defined]
    global_model_factory().register_model_core_factory(_make_quake_core)  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


def add_quake_cli_args(parser: Any) -> None:
    """Add Quake-specific CLI arguments to an SF argument parser."""
    # Environment
    parser.add_argument("--quake_executable", type=str, default=None,
                        help="Path to the Quake worker binary")
    parser.add_argument("--quake_basedir", type=str, default=None,
                        help="QUAKE_BASEDIR — root directory containing id1/")
    parser.add_argument("--quake_native_workdir", type=str, default=None,
                        help="Working directory for Quake processes")
    parser.add_argument("--quake_map_id", type=str, default=None,
                        help="Map name for all workers")
    parser.add_argument("--quake_max_steps_per_episode", type=int, default=None,
                        help="Episode step limit")
    parser.add_argument("--quake_fixed_tick_hz", type=int, default=None,
                        help="Quake tick rate (ticks per second)")
    parser.add_argument("--quake_mode", type=str, default=None,
                        help="Reward mode passed to NativeWorldEnv")
    parser.add_argument("--quake_seed", type=int, default=None,
                        help="Base RNG seed; each worker adds its env_id")
    # Native args / options encoded as JSON strings
    parser.add_argument("--quake_native_args_json", type=str, default=None,
                        help='JSON array of extra Quake CLI args, e.g. ["-game","frikbotnex_train"]')
    parser.add_argument("--quake_options_json", type=str, default=None,
                        help="JSON object of Quake server options, e.g. {\"skill\":0}")
    parser.add_argument("--quake_procgen_json", type=str, default=None,
                        help="JSON object of procgen settings for single-map procgen runs")
    # Multi-scenario support (Step 8)
    parser.add_argument("--quake_scenario_config_json", type=str, default=None,
                        help="JSON array of scenario dicts with map_id/native_args/options")
    parser.add_argument("--reward_json_path", type=str, default=None,
                        help="Path to flat reward.json with reward weights")
    # Encoder architecture
    parser.add_argument("--quake_d_model", type=int, default=None,
                        help="Transformer token dimension (transformer only)")
    parser.add_argument("--quake_n_heads", type=int, default=None,
                        help="Number of attention heads (transformer only)")
    parser.add_argument("--quake_readout", type=str, default=None,
                        help="Transformer readout token: cls or self")
    parser.add_argument("--quake_n_layers", type=int, default=None,
                        help="Number of transformer blocks (transformer only)")
    parser.add_argument("--quake_ffn_dim", type=int, default=None,
                        help="FFN inner dimension (transformer only)")
    parser.add_argument("--quake_attn_dropout", type=float, default=None,
                        help="Attention dropout rate (transformer only)")
    parser.add_argument("--quake_action_history_tokens", type=int, default=None,
                        help="Number of action history tokens fed to transformer")
    # BC warm-start
    parser.add_argument("--quake_bc_checkpoint", type=str, default=None,
                        help="Path to BC/PPO checkpoint for warm-start initialisation")
    parser.add_argument("--quake_bc_checkpoints", type=str, default=None,
                        help="Comma-separated BC/PPO checkpoints for multi-seed PBT warm-start")


def _validate_quake_cfg(cfg: Any) -> None:
    required_attrs = (
        "quake_executable",
        "quake_basedir",
        "quake_native_workdir",
        "quake_map_id",
        "quake_max_steps_per_episode",
        "quake_fixed_tick_hz",
        "quake_mode",
        "quake_seed",
        "quake_native_args_json",
        "quake_options_json",
        "quake_scenario_config_json",
        "quake_d_model",
        "quake_n_heads",
        "quake_readout",
        "quake_n_layers",
        "quake_ffn_dim",
        "quake_attn_dropout",
        "quake_action_history_tokens",
        "reward_json_path",
    )
    missing = [name for name in required_attrs if getattr(cfg, name, None) is None]
    if missing:
        raise RuntimeError("Missing required Quake PPO config fields: " + ", ".join(sorted(missing)))
    if not str(cfg.reward_json_path).strip():
        raise RuntimeError("Quake PPO runs require reward_json_path")
    if str(cfg.quake_map_id) == "procgen" and not (cfg.quake_scenario_config_json or cfg.quake_procgen_json):
        raise RuntimeError("Procgen PPO runs require either quake_scenario_config_json or quake_procgen_json")


# ---------------------------------------------------------------------------
# Programmatic cfg builder
# ---------------------------------------------------------------------------


def build_ppo_cfg(
    *,
    scenario: str,
    num_workers: int,
    num_envs_per_worker: int,
    worker_num_splits: int,
    rollout: int,
    total_env_steps: int,
    output_dir: str,
    experiment: str,
    executable: str,
    basedir: str,
    native_workdir: str,
    native_args_json: str,
    options_json: str,
    procgen_json: str,
    scenario_config_json: str,
    mode: str,
    max_steps_per_episode: int,
    fixed_tick_hz: int,
    seed: int,
    device: str,
    trunk_hidden: int,
    gru_hidden: int,
    use_gru: bool,
    d_model: int,
    n_heads: int,
    readout: str,
    n_layers: int,
    ffn_dim: int,
    action_history_tokens: int,
    attn_dropout: float,
    ppo_epochs: int,
    lr: float,
    entropy_coef: float,
    bc_kl_coef: float,
    clip_ratio: float,
    gamma: float,
    gae_lambda: float,
    max_grad_norm: float,
    value_coef: float,
    minibatch_size: int,
    policy_workers_per_policy: int,
    batched_sampling: bool,
    worker_inference: bool = False,
    max_policy_lag: int = 30,
    with_wandb: bool = False,
    # Population-Based Training
    with_pbt: bool,
    num_policies: int,
    pbt_period_env_steps: int,
    pbt_start_mutation: int,
    pbt_replace_fraction: float,
    pbt_mutation_rate: float,
    pbt_optimize_gamma: bool,
    reward_json_path: str,
    init_checkpoint: str = "",
    init_checkpoints: Optional[List[str]] = None,
    resume: bool = False,
    extra_argv: Optional[List[str]] = None,
) -> Any:
    """Build a PPO cfg namespace without command-line parsing.

    Maps explicit PPO config fields to their Sample Factory equivalents so
    the pipeline can call PPO programmatically with the same resolved config.
    """
    register_quake_components()

    batch_size = num_workers * num_envs_per_worker * rollout
    num_batches_per_epoch = max(1, batch_size // minibatch_size) if minibatch_size else 1

    argv: List[str] = [
        "--algo=APPO",
        f"--env=quake_combat",
        f"--use_rnn={'True' if use_gru else 'False'}",
        f"--rnn_type=gru",
        f"--rnn_size={gru_hidden}",
        f"--num_workers={num_workers}",
        f"--num_envs_per_worker={num_envs_per_worker}",
        f"--worker_num_splits={worker_num_splits}",
        f"--rollout={rollout}",
        f"--recurrence={rollout}",
        f"--batch_size={batch_size}",
        f"--num_batches_per_epoch={num_batches_per_epoch}",
        f"--num_epochs={ppo_epochs}",
        f"--ppo_clip_ratio={clip_ratio}",
        f"--learning_rate={lr}",
        f"--lr_schedule=constant",
        f"--gamma={gamma}",
        f"--gae_lambda={gae_lambda}",
        f"--max_grad_norm={max_grad_norm}",
        f"--value_loss_coeff={value_coef}",
        f"--exploration_loss_coeff={entropy_coef}",
        f"--kl_loss_coeff={bc_kl_coef}",
        "--adaptive_stddev=False",
        "--continuous_tanh_scale=1.0",
        "--initial_stddev=0.2",
        f"--policy_workers_per_policy={policy_workers_per_policy}",
        f"--batched_sampling={'True' if batched_sampling else 'False'}",
        f"--max_policy_lag={max_policy_lag}",
        "--keep_checkpoints=10",
        "--save_milestones_sec=600",
        f"--with_wandb={'True' if with_wandb else 'False'}",
        f"--experiment={experiment}",
        f"--train_dir={output_dir}",
        f"--device={'gpu' if device in ('gpu', 'cuda') else 'cpu'}",
        f"--train_for_env_steps={total_env_steps}",
        # Quake-specific
        f"--quake_executable={executable}",
        f"--quake_basedir={basedir}",
        f"--quake_native_workdir={native_workdir}",
        f"--quake_map_id={scenario}",
        f"--quake_max_steps_per_episode={max_steps_per_episode}",
        f"--quake_fixed_tick_hz={fixed_tick_hz}",
        f"--quake_mode={mode}",
        f"--quake_seed={seed}",
        f"--quake_native_args_json={native_args_json}",
        f"--quake_d_model={d_model}",
        f"--quake_n_heads={n_heads}",
        f"--quake_readout={readout}",
        f"--quake_n_layers={n_layers}",
        f"--quake_ffn_dim={ffn_dim}",
        f"--quake_attn_dropout={attn_dropout}",
        f"--quake_action_history_tokens={action_history_tokens}",
    ]

    argv.append(f"--quake_options_json={options_json}")
    argv.append(f"--quake_procgen_json={procgen_json}")
    argv.append(f"--quake_scenario_config_json={scenario_config_json}")

    # Reward weights
    argv.append(f"--reward_json_path={reward_json_path}")

    # Population-Based Training
    argv.append(f"--num_policies={num_policies}")
    if with_pbt:
        argv.extend([
            "--with_pbt=True",
            f"--pbt_period_env_steps={pbt_period_env_steps}",
            f"--pbt_start_mutation={pbt_start_mutation}",
            f"--pbt_replace_fraction={pbt_replace_fraction}",
            f"--pbt_mutation_rate={pbt_mutation_rate}",
            f"--pbt_optimize_gamma={'True' if pbt_optimize_gamma else 'False'}",
        ])

    if init_checkpoints:
        argv.append(f"--quake_bc_checkpoints={','.join(init_checkpoints)}")
    elif init_checkpoint:
        argv.append(f"--quake_bc_checkpoint={init_checkpoint}")

    if extra_argv:
        argv.extend(extra_argv)

    parser, _ = parse_sf_args(argv=argv)
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser, argv=argv)
    cfg.quake_resume = bool(resume)
    cfg.worker_inference = bool(worker_inference)
    _validate_quake_cfg(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------


def _set_ppo_report_interval(seconds: float = 60.0) -> None:
    """Increase Sample Factory's log spam interval from the hardcoded 5s default.

    Runner.__init__ hardcodes ``self.report_interval_sec = 5.0`` and then
    registers a periodic timer with that value.  We patch __init__ to
    overwrite both the attribute *and* the first timer's period after the
    original init completes.
    """
    try:
        from sample_factory.algo.runners import runner as runner_mod
        _orig_init = runner_mod.Runner.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.report_interval_sec = seconds
            # The first timer registered is the report timer.
            if self.timers:
                self.timers[0]._interval_sec = seconds

        if not getattr(_patched_init, "_quake_patched", False):
            _patched_init._quake_patched = True  # type: ignore[attr-defined]
            runner_mod.Runner.__init__ = _patched_init
    except Exception:
        pass


def run_ppo(cfg: Any) -> Dict[str, Any]:
    """Launch PPO training and retain a compact summary artifact."""
    from sample_factory.algo.utils.misc import ExperimentStatus
    from qnn.ppo.observer import BestCheckpointArchiver

    register_quake_components()
    _set_ppo_report_interval(60.0)
    _ensure_warm_start_checkpoint(cfg)

    # SF's argparse omits "linear_decay" from lr_schedule choices but the
    # scheduler code supports it.  Set it after argparse runs.
    cfg.lr_schedule = "linear_decay"

    cfg, runner = make_runner(cfg)

    # Worker inference: bypass centralized inference workers entirely.
    if getattr(cfg, "worker_inference", False):
        from qnn.ppo.worker_inference import WorkerInferenceSampler
        _orig_make_sampler = runner._make_sampler
        def _patched_make_sampler(sampler_cls, event_loop):
            print("[quake_ppo] Using WorkerInferenceSampler (bypassing inference workers)")
            return _orig_make_sampler(WorkerInferenceSampler, event_loop)
        runner._make_sampler = _patched_make_sampler
        print("[quake_ppo] Worker inference mode enabled")

    runner.register_observer(BestCheckpointArchiver(runner))

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    stage_artifacts = write_ppo_stage_artifacts(cfg, status, runner)
    return {
        "ppo_status": str(status),
        "train_dir": str(getattr(cfg, "train_dir", "")),
        "experiment_dir": str(_experiment_dir(cfg)),
        "ppo_summary_path": stage_artifacts["summary_path"],
        "ppo_manifest_path": stage_artifacts["manifest_path"],
        "selected_policy_id": stage_artifacts["selected_policy_id"],
        "metrics": stage_artifacts["metrics"],
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from sample_factory.algo.utils.misc import ExperimentStatus
    from qnn.ppo.observer import BestCheckpointArchiver

    register_quake_components()
    parser, _ = parse_sf_args()
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser)
    _validate_quake_cfg(cfg)
    _ensure_warm_start_checkpoint(cfg)

    cfg.lr_schedule = "linear_decay"

    cfg, runner = make_runner(cfg)
    runner.register_observer(BestCheckpointArchiver(runner))

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    write_ppo_stage_artifacts(cfg, status, runner)



if __name__ == "__main__":
    main()


# ── Runner entry point (called by run.router) ──────────────────────

def run(ctx) -> dict:
    """Runner entry point — delegates to ppo.pipeline."""
    from qnn.ppo.pipeline import run_pipeline
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
