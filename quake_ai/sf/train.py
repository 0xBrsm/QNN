"""Sample Factory APPO training entry point for Quake combat bot.

Usage (standalone):
    python -m quake_ai.sf.train \\
        --algo=APPO --env=quake_combat \\
        --quake_executable=assets/bin/quake_worker \\
        --quake_basedir=assets \\
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

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import make_runner
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


def _has_existing_sf_checkpoint(cfg: Any) -> bool:
    checkpoint_dir = _experiment_dir(cfg) / "checkpoint_p0"
    return any(checkpoint_dir.glob("*.pth"))


def _ensure_warm_start_checkpoint(cfg: Any) -> Optional[Path]:
    """Seed SF checkpoint dirs from a warm-start checkpoint.

    Accepts either a BC ``.npz`` (converted to SF format) or an existing
    SF ``.pth`` (copied directly).  Skipped when the experiment already
    has checkpoints (i.e. resuming a prior run).
    """
    ckpt = str(getattr(cfg, "quake_bc_checkpoint", "") or "").strip()
    if not ckpt:
        return None

    if _has_existing_sf_checkpoint(cfg):
        print(f"[quake_sf] Found existing SF checkpoints in {_experiment_dir(cfg) / 'checkpoint_p0'}; skipping warm-start.")
        return None

    # Tell SF to load from best_* (where we place the warm-start seed).
    cfg.load_checkpoint_kind = "best"

    exp_dir = _experiment_dir(cfg)
    num_policies = int(getattr(cfg, "num_policies", 1))

    if ckpt.endswith(".pth"):
        # Already SF format — copy directly into each policy dir.
        import shutil
        print(f"[quake_sf] Using SF checkpoint {ckpt} as warm-start ...")
        # Ensure the name starts with "best_" so load_checkpoint_kind=best finds it.
        name = Path(ckpt).name
        if not name.startswith("best_"):
            name = f"best_{name}"
        first_path = None
        for pid in range(num_policies):
            policy_dir = exp_dir / f"checkpoint_p{pid}"
            policy_dir.mkdir(parents=True, exist_ok=True)
            dest = policy_dir / name
            shutil.copy2(ckpt, dest)
            if pid == 0:
                first_path = dest
            print(f"[quake_sf] Warm-start checkpoint copied: {dest}")
        return first_path
    else:
        # BC .npz — convert to SF format.
        from quake_ai.sf.checkpoint_converter import QNNPolicy, save_sf_format
        print(f"[quake_sf] Converting BC checkpoint {ckpt} to SF warm-start format ...")
        bc_policy = QNNPolicy.load(ckpt, device="cpu")
        first_path = None
        for pid in range(num_policies):
            p = save_sf_format(bc_policy, exp_dir / f"checkpoint_p{pid}")
            if pid == 0:
                first_path = p
            print(f"[quake_sf] Warm-start checkpoint written: {p}")
        return first_path


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
    parser.add_argument("--quake_executable", type=str, default="assets/bin/quake_worker",
                        help="Path to the Quake worker binary")
    parser.add_argument("--quake_basedir", type=str, default="",
                        help="QUAKE_BASEDIR — root directory containing id1/")
    parser.add_argument("--quake_native_workdir", type=str, default="",
                        help="Working directory for Quake processes")
    parser.add_argument("--quake_map_id", type=str, default="procgen",
                        help="Map name for all workers — 'procgen' (default) generates a unique map each episode")
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
    parser.add_argument("--quake_record_demos", type=int, default=1,
                        help="Record .dem files during training (1=on, 0=off); best episodes saved as best.dem")
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
    scenario: str = "procgen",
    num_workers: int = 8,
    num_envs_per_worker: int = 1,
    worker_num_splits: int = 1,
    rollout: int = 256,
    total_env_steps: int = 10_000_000,
    output_dir: str = "assets/runs/sf_ppo",
    experiment: str = "quake_combat",
    executable: str = "assets/bin/quake_worker",
    basedir: str = "",
    native_workdir: str = "",
    native_args_json: str = '["-game","frikbotnex_train"]',
    options_json: str = "",
    scenario_config_json: str = "",
    mode: str = "pvp",
    max_steps_per_episode: int = 1024,
    fixed_tick_hz: int = 20,
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
    # Population-Based Training
    with_pbt: bool = False,
    num_policies: int = 1,
    pbt_period_env_steps: int = 5_000_000,
    pbt_start_mutation: int = 20_000_000,
    pbt_replace_fraction: float = 0.3,
    pbt_mutation_rate: float = 0.15,
    pbt_optimize_gamma: bool = False,
    record_demos: bool = True,
    extra_argv: Optional[List[str]] = None,
) -> Any:
    """Build an SF cfg namespace without command-line parsing.

    Maps PPOConfig fields to their SF equivalents so training.py can
    call SF programmatically with the same hyperparameters it used for run_ppo.
    """
    register_quake_components()

    batch_size = num_workers * num_envs_per_worker * rollout

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
        f"--quake_fixed_tick_hz={fixed_tick_hz}",
        f"--quake_mode={mode}",
        f"--quake_seed={seed}",
        f"--quake_native_args_json={native_args_json}",
        f"--quake_trunk_hidden={trunk_hidden}",
        f"--quake_d_model={d_model}",
        f"--quake_n_heads={n_heads}",
        f"--quake_n_layers={n_layers}",
        f"--quake_ffn_dim={ffn_dim}",
        f"--quake_attn_dropout={attn_dropout}",
        f"--quake_record_demos={1 if record_demos else 0}",
    ]

    if options_json:
        argv.append(f"--quake_options_json={options_json}")
    if scenario_config_json:
        argv.append(f"--quake_scenario_config_json={scenario_config_json}")

    # Population-Based Training
    if with_pbt:
        argv.extend([
            "--with_pbt=True",
            f"--num_policies={num_policies}",
            f"--pbt_period_env_steps={pbt_period_env_steps}",
            f"--pbt_start_mutation={pbt_start_mutation}",
            f"--pbt_replace_fraction={pbt_replace_fraction}",
            f"--pbt_mutation_rate={pbt_mutation_rate}",
            f"--pbt_optimize_gamma={'True' if pbt_optimize_gamma else 'False'}",
        ])

    if init_checkpoint:
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


def _set_sf_report_interval(seconds: float = 60.0) -> None:
    """Increase SF's log spam interval from the hardcoded 5s default.

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


def run_sf(cfg: Any) -> Dict[str, Any]:
    """Launch SF APPO training and return a summary dict."""
    from sample_factory.algo.utils.misc import ExperimentStatus
    from quake_ai.sf.observer import BestCheckpointArchiver

    register_quake_components()
    _set_sf_report_interval(60.0)
    _ensure_warm_start_checkpoint(cfg)

    # SF's argparse omits "linear_decay" from lr_schedule choices but the
    # scheduler code supports it.  Set it after argparse runs.
    cfg.lr_schedule = "linear_decay"

    cfg, runner = make_runner(cfg)
    runner.register_observer(BestCheckpointArchiver(runner))

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return {"sf_status": str(status), "train_dir": str(getattr(cfg, "train_dir", ""))}


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from sample_factory.algo.utils.misc import ExperimentStatus
    from quake_ai.sf.observer import BestCheckpointArchiver

    register_quake_components()
    parser, _ = parse_sf_args()
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser)
    _ensure_warm_start_checkpoint(cfg)

    cfg.lr_schedule = "linear_decay"

    cfg, runner = make_runner(cfg)
    runner.register_observer(BestCheckpointArchiver(runner))

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        runner.run()


if __name__ == "__main__":
    main()
