"""Sample Factory APPO training entry point for Quake combat bot.

Usage (standalone):
    python -m quake_ai.sf.train \\
        --algo=APPO --env=quake_combat \\
        --quake_executable=../artifacts/bin/quake_worker \\
        --quake_basedir=../artifacts/quake \\
        --quake_map_id=dm4 \\
        --num_workers=30 --rollout=256

Usage (programmatic, from training.py):
    from quake_ai.sf.train import register_quake_components, build_sf_cfg, run_sf
    register_quake_components()
    cfg = build_sf_cfg(scenario="dm4", num_workers=8, ...)
    run_sf(cfg)

SF hyperparameter defaults are calibrated to match the existing PPO config
(ppo_combat_bot_live.yaml) so that a side-by-side comparison is meaningful.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import run_rl
    # SF 2.1.x uses the model_factory pattern; 2.0.x had register_custom_encoder
    try:
        from sample_factory.model.encoder import register_custom_encoder as _register_custom_encoder
        _HAS_REGISTER_CUSTOM_ENCODER = True
    except ImportError:
        from sample_factory.algo.utils.context import global_model_factory
        _HAS_REGISTER_CUSTOM_ENCODER = False
except ImportError as exc:
    raise ImportError("sample-factory is required: pip install sample-factory>=2.0.0") from exc

from quake_ai.sf.quake_encoder import QuakeTransformerEncoder, make_quake_encoder
from quake_ai.sf.quake_env import make_quake_env
from quake_ai.utils.io import trusted_torch_load


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_quake_encoder(cfg: Any, obs_space: Any):
    return make_quake_encoder(cfg, obs_space)


def _patch_sample_factory_checkpoint_loading() -> None:
    """Keep SF resume working under torch 2.6+'s weights_only default."""
    from sample_factory.algo.learning import learner as learner_mod

    if getattr(learner_mod.Learner.load_checkpoint, "_quake_trusted_patch", False):
        return

    def _load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            learner_mod.log.warning("No checkpoints found")
            return None

        latest_checkpoint = checkpoints[-1]
        num_attempts = 3
        for attempt in range(num_attempts):
            try:
                learner_mod.log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                return trusted_torch_load(latest_checkpoint, map_location=device)
            except Exception:
                learner_mod.log.exception(f"Could not load from checkpoint, attempt {attempt}")

        return None

    _load_checkpoint._quake_trusted_patch = True  # type: ignore[attr-defined]
    learner_mod.Learner.load_checkpoint = staticmethod(_load_checkpoint)


def _experiment_dir(cfg: Any) -> Path:
    return Path(getattr(cfg, "train_dir", ".")) / str(getattr(cfg, "experiment", "quake_combat"))


def _has_existing_sf_checkpoint(cfg: Any) -> bool:
    checkpoint_dir = _experiment_dir(cfg) / "checkpoint_p0"
    return any(checkpoint_dir.glob("*.pth"))


def _ensure_bc_warm_start_checkpoint(cfg: Any) -> Optional[Path]:
    bc_ckpt = str(getattr(cfg, "quake_bc_checkpoint", "") or "").strip()
    if not bc_ckpt:
        return None

    if _has_existing_sf_checkpoint(cfg):
        print(f"[quake_sf] Found existing SF checkpoints in {_experiment_dir(cfg) / 'checkpoint_p0'}; skipping BC warm-start conversion.")
        return None

    from quake_ai.sf.checkpoint_converter import MLPGRUPolicy, save_sf_format

    print(f"[quake_sf] Converting BC checkpoint {bc_ckpt} to SF warm-start format ...")
    bc_policy = MLPGRUPolicy.load(bc_ckpt, device="cpu")
    ckpt_path = save_sf_format(bc_policy, _experiment_dir(cfg) / "checkpoint_p0")
    print(f"[quake_sf] Warm-start checkpoint written: {ckpt_path}")
    print("[quake_sf] NOTE: verify head key mapping with inspect_sf_action_layout() first.")
    return ckpt_path


def register_quake_components() -> None:
    """Register Quake env and encoder with Sample Factory (idempotent)."""
    _patch_sample_factory_checkpoint_loading()
    register_env("quake_combat", make_quake_env)
    if _HAS_REGISTER_CUSTOM_ENCODER:
        # SF < 2.1: direct registration helper
        _register_custom_encoder("quake_trunk", QuakeTransformerEncoder)  # type: ignore[name-defined]
    else:
        # SF 2.1+: register via model factory
        global_model_factory().register_encoder_factory(_make_quake_encoder)  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


def add_quake_cli_args(parser: Any) -> None:
    """Add Quake-specific CLI arguments to an SF argument parser."""
    # Environment
    parser.add_argument("--quake_executable", type=str, default="../artifacts/bin/quake_worker",
                        help="Path to the Quake worker binary")
    parser.add_argument("--quake_basedir", type=str, default="",
                        help="QUAKE_BASEDIR — root directory containing id1/")
    parser.add_argument("--quake_native_workdir", type=str, default="",
                        help="Working directory for Quake processes")
    parser.add_argument("--quake_map_id", type=str, default="dm4",
                        help="Map name for all workers (overridden by quake_scenario_config_json)")
    parser.add_argument("--quake_max_steps_per_episode", type=int, default=1024,
                        help="Episode step limit")
    parser.add_argument("--quake_fixed_tick_hz", type=int, default=20,
                        help="Quake tick rate (ticks per second)")
    parser.add_argument("--quake_mode", type=str, default="pvp",
                        help="Reward mode passed to NativeWorldEnv")
    parser.add_argument("--quake_seed", type=int, default=17,
                        help="Base RNG seed; each worker adds its env_id")
    # Native args / options encoded as JSON strings
    parser.add_argument("--quake_native_args_json", type=str, default='["-game","frikbotnex_train"]',
                        help='JSON array of extra Quake CLI args, e.g. ["-game","frikbotnex_train"]')
    parser.add_argument("--quake_options_json", type=str, default="",
                        help="JSON object of Quake server options, e.g. {\"skill\":0}")
    # Multi-scenario support (Step 8)
    parser.add_argument("--quake_scenario_config_json", type=str, default="",
                        help="JSON array of scenario dicts with map_id/native_args/options")
    # Encoder architecture
    parser.add_argument("--quake_trunk_hidden", type=int, default=128,
                        help="Encoder output dimension (feeds GRU input)")
    parser.add_argument("--quake_d_model", type=int, default=64,
                        help="Transformer token dimension (transformer only)")
    parser.add_argument("--quake_n_heads", type=int, default=2,
                        help="Number of attention heads (transformer only)")
    parser.add_argument("--quake_n_layers", type=int, default=2,
                        help="Number of transformer blocks (transformer only)")
    parser.add_argument("--quake_ffn_dim", type=int, default=256,
                        help="FFN inner dimension (transformer only)")
    parser.add_argument("--quake_attn_dropout", type=float, default=0.0,
                        help="Attention dropout rate (transformer only)")
    # BC warm-start
    parser.add_argument("--quake_bc_checkpoint", type=str, default="",
                        help="Path to BC/PPO checkpoint for warm-start initialisation")


# ---------------------------------------------------------------------------
# Programmatic cfg builder
# ---------------------------------------------------------------------------


def build_sf_cfg(
    scenario: str = "dm4",
    num_workers: int = 8,
    rollout: int = 256,
    total_env_steps: int = 10_000_000,
    output_dir: str = "../artifacts/runs/sf_ppo",
    experiment: str = "quake_combat",
    executable: str = "../artifacts/bin/quake_worker",
    basedir: str = "",
    native_workdir: str = "",
    native_args_json: str = '["-game","frikbotnex_train"]',
    options_json: str = "",
    scenario_config_json: str = "",
    mode: str = "pvp",
    max_steps_per_episode: int = 1024,
    seed: int = 17,
    device: str = "gpu",
    init_checkpoint: str = "",
    trunk_hidden: int = 128,
    gru_hidden: int = 128,
    use_gru: bool = True,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    ffn_dim: int = 256,
    attn_dropout: float = 0.0,
    ppo_epochs: int = 2,
    lr: float = 0.00025,
    entropy_coef: float = 0.002,
    bc_kl_coef: float = 0.05,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
    value_coef: float = 0.5,
    max_policy_lag: int = 30,
    with_wandb: bool = False,
    extra_argv: Optional[List[str]] = None,
) -> Any:
    """Build an SF cfg namespace without command-line parsing.

    Maps PPOConfig fields to their SF equivalents so training.py can
    call SF programmatically with the same hyperparameters it used for run_ppo.
    """
    register_quake_components()

    batch_size = num_workers * rollout

    argv: List[str] = [
        "--algo=APPO",
        f"--env=quake_combat",
        f"--use_rnn={'True' if use_gru else 'False'}",
        f"--rnn_type=gru",
        f"--rnn_size={gru_hidden}",
        f"--num_workers={num_workers}",
        f"--num_envs_per_worker=1",
        "--worker_num_splits=1",
        f"--rollout={rollout}",
        f"--batch_size={batch_size}",
        f"--num_batches_per_epoch=1",
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
        f"--max_policy_lag={max_policy_lag}",
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
        f"--quake_mode={mode}",
        f"--quake_seed={seed}",
        f"--quake_native_args_json={native_args_json}",
        f"--quake_trunk_hidden={trunk_hidden}",
        f"--quake_d_model={d_model}",
        f"--quake_n_heads={n_heads}",
        f"--quake_n_layers={n_layers}",
        f"--quake_ffn_dim={ffn_dim}",
        f"--quake_attn_dropout={attn_dropout}",
    ]

    if options_json:
        argv.append(f"--quake_options_json={options_json}")
    if scenario_config_json:
        argv.append(f"--quake_scenario_config_json={scenario_config_json}")

    if init_checkpoint:
        # Warm-start runs materialize a seed checkpoint before launch if there
        # is no prior SF run to resume from.
        argv.append("--load_checkpoint_kind=latest")
        argv.append(f"--quake_bc_checkpoint={init_checkpoint}")

    if extra_argv:
        argv.extend(extra_argv)

    parser, _ = parse_sf_args(argv=argv)
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser, argv=argv)
    return cfg


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------


def run_sf(cfg: Any) -> Dict[str, Any]:
    """Launch SF APPO training and return a summary dict."""
    register_quake_components()
    _ensure_bc_warm_start_checkpoint(cfg)
    status = run_rl(cfg)
    return {"sf_status": str(status), "train_dir": str(getattr(cfg, "train_dir", ""))}


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    register_quake_components()
    parser, _ = parse_sf_args()
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser)
    _ensure_bc_warm_start_checkpoint(cfg)
    run_rl(cfg)


if __name__ == "__main__":
    main()
