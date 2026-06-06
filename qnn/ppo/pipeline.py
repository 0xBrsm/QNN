"""PPO pipeline: run directory → training job → post-train eval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from qnn.env.planning import validate_native_mod_assets
from qnn.eval.run import EvalConfig, run_evaluation
from qnn.run.common import (
    base_results,
    best_checkpoint,
    ensure_worker,
    finalize_results,
    latest_ppo_checkpoint,
    prepare_eval_checkpoint,
    prepare_eval_outputs,
    prepare_ppo_run_outputs,
    require_cfg_list,
    require_cfg_mapping,
    require_cfg_string,
    require_cfg_value,
)
from qnn.run.config import build_run_ppo_eval_config
from qnn.utils.checkpoint_converter import sf_to_qnn
from qnn.utils.io import read_json, trusted_torch_load


def _scenario_config_json_from_cfg(config: dict[str, Any]) -> str:
    path_value = str(require_cfg_value(config, "scenario_config_path", "PPO config")).strip()
    if not path_value:
        return ""
    payload = read_json(path_value)
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"scenario_config_path must define a non-empty scenarios list: {path_value}")
    return json.dumps(scenarios)


def _validate_warm_start_arch(init_ckpt: str, ppo_cfg: dict[str, Any]) -> None:
    if not init_ckpt or not Path(init_ckpt).exists():
        return
    arch_keys = (
        "trunk_hidden", "gru_hidden", "use_gru", "d_model", "n_heads",
        "n_layers", "ffn_dim", "attn_dropout",
    )
    try:
        payload = trusted_torch_load(init_ckpt, map_location="cpu")
    except Exception:
        return
    meta = None
    if isinstance(payload, dict):
        if "meta" in payload and isinstance(payload["meta"], dict):
            meta = payload["meta"]
        elif "trunk_hidden" in payload:
            meta = payload
    if meta is None:
        return
    mismatches = []
    for key in arch_keys:
        ckpt_val = meta.get(key)
        if ckpt_val is None:
            continue
        cfg_val = ppo_cfg.get(key)
        if cfg_val is None:
            continue
        if type(ckpt_val) != type(cfg_val):
            try:
                ckpt_val = type(cfg_val)(ckpt_val)
            except (TypeError, ValueError):
                pass
        if ckpt_val != cfg_val:
            mismatches.append(f"  {key}: checkpoint={ckpt_val!r}  ppo_cfg={cfg_val!r}")
    if mismatches:
        raise RuntimeError(
            "Architecture mismatch between warm-start checkpoint and PPO config.\n"
            f"Checkpoint: {init_ckpt}\n"
            + "\n".join(mismatches)
            + "\n\nEnsure the run's frozen model config matches the warm-start checkpoint metadata."
        )


def _detect_obs_dim_from_checkpoint(ppo_cfg: dict[str, Any], checkpoint_path: str | Path | None = None) -> int:
    init_ckpt = str(checkpoint_path or require_cfg_string(ppo_cfg, "init_ckpt", "PPO config"))
    try:
        payload = trusted_torch_load(str(init_ckpt), map_location="cpu")
        if isinstance(payload, dict) and "meta" in payload:
            meta = payload["meta"]
            if isinstance(meta, dict) and "obs_dim" in meta:
                return int(meta["obs_dim"])
    except Exception as exc:
        raise RuntimeError(f"Unable to read obs_dim from warm-start checkpoint metadata: {init_ckpt}") from exc
    raise RuntimeError(f"Warm-start checkpoint metadata is missing obs_dim: {init_ckpt}")


def run_training_job(
    ppo_cfg: dict[str, Any],
    resolved_asset_root: Path,
    worker_path: Path,
    device: str,
) -> dict[str, Any]:
    """Execute a single PPO training job."""
    from qnn.ppo.train import build_ppo_cfg, register_quake_components, run_ppo

    register_quake_components()

    output_dir = str(Path(require_cfg_string(ppo_cfg, "output_dir", "PPO config")))
    scenario = require_cfg_string(ppo_cfg, "map_id", "PPO config")
    num_workers = int(require_cfg_value(ppo_cfg, "num_workers", "PPO config"))
    num_envs_per_worker = int(require_cfg_value(ppo_cfg, "num_envs_per_worker", "PPO config"))
    num_envs = int(require_cfg_value(ppo_cfg, "num_envs", "PPO config"))
    if num_envs != num_workers * num_envs_per_worker:
        raise RuntimeError(
            f"PPO config num_envs={num_envs} does not match num_workers*num_envs_per_worker="
            f"{num_workers * num_envs_per_worker}"
        )
    rollout = int(require_cfg_value(ppo_cfg, "rollout_steps", "PPO config"))
    total_steps = int(require_cfg_value(ppo_cfg, "total_steps", "PPO config"))
    total_env_steps = total_steps * num_envs

    native_args = require_cfg_list(ppo_cfg, "native_args", "PPO config")
    options = require_cfg_mapping(ppo_cfg, "options", "PPO config")
    procgen_cfg = require_cfg_value(ppo_cfg, "procgen", "PPO config")
    resume = bool(require_cfg_value(ppo_cfg, "resume", "PPO config"))
    init_checkpoint = str(ppo_cfg.get("init_ckpt", ""))

    cfg = build_ppo_cfg(
        scenario=scenario,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        worker_num_splits=int(require_cfg_value(ppo_cfg, "worker_num_splits", "PPO config")),
        rollout=rollout,
        total_env_steps=total_env_steps,
        output_dir=output_dir,
        experiment="",
        executable=str(worker_path),
        basedir=str(resolved_asset_root),
        native_workdir=require_cfg_string(ppo_cfg, "native_workdir", "PPO config"),
        native_args_json=json.dumps(native_args),
        options_json=json.dumps(options),
        procgen_json=json.dumps(procgen_cfg) if procgen_cfg is not None else "",
        scenario_config_json=_scenario_config_json_from_cfg(ppo_cfg),
        mode=require_cfg_string(ppo_cfg, "mode", "PPO config"),
        max_steps_per_episode=int(require_cfg_value(ppo_cfg, "max_steps_per_episode", "PPO config")),
        fixed_tick_hz=int(require_cfg_value(ppo_cfg, "fixed_tick_hz", "PPO config")),
        seed=int(require_cfg_value(ppo_cfg, "seed", "PPO config")),
        device=device,
        init_checkpoint=init_checkpoint,
        init_checkpoints=ppo_cfg.get("init_ckpts"),
        resume=resume,
        trunk_hidden=int(require_cfg_value(ppo_cfg, "trunk_hidden", "PPO config")),
        gru_hidden=int(require_cfg_value(ppo_cfg, "gru_hidden", "PPO config")),
        use_gru=bool(require_cfg_value(ppo_cfg, "use_gru", "PPO config")),
        d_model=int(require_cfg_value(ppo_cfg, "d_model", "PPO config")),
        n_heads=int(require_cfg_value(ppo_cfg, "n_heads", "PPO config")),
        n_layers=int(require_cfg_value(ppo_cfg, "n_layers", "PPO config")),
        ffn_dim=int(require_cfg_value(ppo_cfg, "ffn_dim", "PPO config")),
        attn_dropout=float(require_cfg_value(ppo_cfg, "attn_dropout", "PPO config")),
        ppo_epochs=int(require_cfg_value(ppo_cfg, "ppo_epochs", "PPO config")),
        lr=float(require_cfg_value(ppo_cfg, "policy_lr", "PPO config")),
        entropy_coef=float(require_cfg_value(ppo_cfg, "entropy_coef", "PPO config")),
        bc_kl_coef=float(require_cfg_value(ppo_cfg, "bc_kl_coef", "PPO config")),
        clip_ratio=float(require_cfg_value(ppo_cfg, "clip_ratio", "PPO config")),
        gamma=float(require_cfg_value(ppo_cfg, "gamma", "PPO config")),
        gae_lambda=float(require_cfg_value(ppo_cfg, "gae_lambda", "PPO config")),
        max_grad_norm=float(require_cfg_value(ppo_cfg, "max_grad_norm", "PPO config")),
        value_coef=float(require_cfg_value(ppo_cfg, "value_coef", "PPO config")),
        with_pbt=bool(require_cfg_value(ppo_cfg, "with_pbt", "PPO config")),
        num_policies=int(require_cfg_value(ppo_cfg, "num_policies", "PPO config")),
        pbt_period_env_steps=int(require_cfg_value(ppo_cfg, "pbt_period_env_steps", "PPO config")),
        pbt_start_mutation=int(require_cfg_value(ppo_cfg, "pbt_start_mutation", "PPO config")),
        pbt_replace_fraction=float(require_cfg_value(ppo_cfg, "pbt_replace_fraction", "PPO config")),
        pbt_mutation_rate=float(require_cfg_value(ppo_cfg, "pbt_mutation_rate", "PPO config")),
        pbt_optimize_gamma=bool(require_cfg_value(ppo_cfg, "pbt_optimize_gamma", "PPO config")),
        minibatch_size=int(require_cfg_value(ppo_cfg, "minibatch_size", "PPO config")),
        policy_workers_per_policy=int(require_cfg_value(ppo_cfg, "policy_workers_per_policy", "PPO config")),
        batched_sampling=bool(require_cfg_value(ppo_cfg, "batched_sampling", "PPO config")),
        worker_inference=bool(ppo_cfg.get("worker_inference", False)),
        worker_inference_device=str(ppo_cfg.get("worker_inference_device", "cpu")),
        reward_json_path=require_cfg_string(ppo_cfg, "reward_json_path", "PPO config"),
        head_loss_weights=str(ppo_cfg.get("head_loss_weights", "") or ""),
        initial_stddev=float(require_cfg_value(ppo_cfg, "initial_stddev", "PPO config")),
    )

    ppo_result = run_ppo(cfg)

    exp_dir = Path(output_dir)
    ppo_ckpt_path = best_checkpoint(exp_dir)
    ppo_ckpt = str(ppo_ckpt_path) if ppo_ckpt_path else None

    if ppo_ckpt:
        best_dir = Path(output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        qnn_ckpt_path = best_dir / "best_model.pth"
        try:
            from qnn.model.policy import ModelConfig
            obs_dim = (
                int(require_cfg_value(ppo_cfg, "obs_dim", "PPO config"))
                if "obs_dim" in ppo_cfg
                else _detect_obs_dim_from_checkpoint(ppo_cfg, ppo_ckpt)
            )
            qnn_policy = sf_to_qnn(
                sf_checkpoint_path=ppo_ckpt,
                obs_dim=obs_dim,
                model=ModelConfig.from_flat_dict(ppo_cfg),
                device="cpu",
            )
            qnn_policy.save(qnn_ckpt_path)
            ppo_result["best_model_path"] = str(qnn_ckpt_path)
            ppo_result["ppo_checkpoint_path"] = ppo_ckpt
            ppo_result["sf_checkpoint_path"] = ppo_ckpt
        except Exception as exc:
            ppo_result["checkpoint_convert_error"] = str(exc)

    return ppo_result


def run_pipeline(ctx: Any, *, post_train_eval: bool = True, write_report: bool = True) -> dict[str, Any]:
    """Execute PPO pipeline: train + optional post-train eval."""
    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    will_resume_ppo = prepare_ppo_run_outputs(ctx.run_cfg, resume=ctx.resume)
    ppo_cfg, eval_cfg = build_run_ppo_eval_config(ctx.run_cfg, ctx.device)
    seed_ckpt = str(ctx.run_cfg.get("checkpoint_path", "") or "")
    ppo_cfg["resume"] = will_resume_ppo
    if seed_ckpt and not will_resume_ppo and not Path(seed_ckpt).exists():
        raise FileNotFoundError(f"Seed checkpoint from run.json does not exist: {seed_ckpt}")
    if will_resume_ppo:
        latest_ckpt = latest_ppo_checkpoint(ctx.run_cfg)
        if latest_ckpt is None:
            raise RuntimeError("PPO resume requested, but no latest checkpoint could be located")
        results["ppo_resume_from"] = str(latest_ckpt)
    elif seed_ckpt:
        ppo_cfg["init_ckpt"] = seed_ckpt
        results["ppo_init_ckpt"] = seed_ckpt
        if ctx.resume:
            results["ppo_resume_fallback"] = "No existing PPO checkpoint found; started from seed checkpoint."
        _validate_warm_start_arch(seed_ckpt, ppo_cfg)
    else:
        results["ppo_init_ckpt"] = ""
        results["ppo_random_init"] = True

    validate_native_mod_assets(
        ctx.asset_root,
        ppo_cfg.get("native_args") if isinstance(ppo_cfg.get("native_args"), list) else None,
    )
    worker_path = ensure_worker(ctx.worker_binary, rebuild=False)

    started = time.monotonic()
    ppo_cfg["native_env"] = {"QUAKE_BASEDIR": str(ctx.asset_root)}
    ppo_cfg["native_executable"] = str(worker_path)
    results["ppo"] = run_training_job(ppo_cfg, ctx.asset_root, worker_path, ctx.device)
    stage_timings["ppo"] = time.monotonic() - started

    if post_train_eval:
        started = time.monotonic()
        prepare_eval_outputs(ctx.run_cfg, resume=False)
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(ctx.asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        eval_ckpt = best_checkpoint(ctx.output_dirs["checkpoints"])
        if eval_ckpt is None:
            print("[training] No checkpoints found for post-train eval, skipping.")
        else:
            eval_cfg["checkpoint_path"] = prepare_eval_checkpoint(
                str(eval_ckpt), str(eval_cfg["output_dir"]),
            )
            results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

    results["stage_timings"] = stage_timings
    if not write_report:
        return results
    return finalize_results(ctx, results, stage_timings)


def run(ctx: Any) -> dict[str, Any]:
    """Runner entry point called by run.router."""
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
