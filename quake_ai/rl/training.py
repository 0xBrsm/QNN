"""Training pipeline driven entirely by a run directory manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from engine.bridge import NativeTokenProcess
from quake_ai.actions import idle_action
from quake_ai.rl.collect import CollectConfig, collect_demo_tokens
from quake_ai.rl.evaluation import EvalConfig, run_evaluation
from quake_ai.rl.planning import (
    _resolve_asset_root,
    _validate_native_mod_assets,
    build_runtime_plan_for_run,
    resolve_demo_dir_from_run,
    write_run_plan,
)
from quake_ai.rl.profiles import (
    build_run_bc_config,
    build_run_check_surface,
    build_run_eval_config,
    build_run_ppo_eval_config,
    load_run_config,
    run_output_dirs,
)
from quake_ai.rl.reporting import write_run_report
from quake_ai.rl.training_bc import BCConfig, run_behavior_cloning
from quake_ai.utils.io import read_json, trusted_torch_load


def _require_cfg_value(config: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in config:
        raise RuntimeError(f"{context} must define {key}")
    return config[key]


def _require_cfg_string(config: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_cfg_value(config, key, context)
    if not isinstance(value, str):
        raise RuntimeError(f"{context}.{key} must be a string")
    return value


def _require_cfg_list(config: Mapping[str, Any], key: str, context: str) -> list[Any]:
    value = _require_cfg_value(config, key, context)
    if not isinstance(value, list):
        raise RuntimeError(f"{context}.{key} must be a list")
    return list(value)


def _require_cfg_mapping(config: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _require_cfg_value(config, key, context)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}.{key} must be a mapping")
    return dict(value)


def _is_sf_checkpoint_payload(payload: object) -> bool:
    return isinstance(payload, dict) and "model" in payload and ("train_step" in payload or "env_steps" in payload)


def _prepare_eval_checkpoint(checkpoint_path: str, output_dir: str) -> str:
    path = Path(checkpoint_path)
    if not path.exists() or path.suffix != ".pth":
        return str(path)

    try:
        payload = trusted_torch_load(str(path), map_location="cpu")
    except Exception:
        return str(path)

    if not _is_sf_checkpoint_payload(payload):
        return str(path)

    from quake_ai.ppo.checkpoint_converter import sf_to_qnn

    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise RuntimeError(f"SF checkpoint conversion requires architecture sidecar metadata: {sidecar_path}")
    meta = read_json(sidecar_path)
    if not isinstance(meta, Mapping):
        raise RuntimeError(f"SF checkpoint sidecar must be a JSON object: {sidecar_path}")
    required_keys = (
        "obs_dim",
        "trunk_hidden",
        "gru_hidden",
        "use_gru",
        "d_model",
        "n_heads",
        "n_layers",
        "ffn_dim",
        "action_history_tokens",
        "attn_dropout",
        "readout",
    )
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise RuntimeError(
            f"SF checkpoint sidecar is missing architecture fields ({', '.join(missing)}): {sidecar_path}"
        )
    converted_dir = Path(output_dir).parent / "_eval_ckpts"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted_path = converted_dir / f"{path.stem}_qnn.pth"
    converted_sidecar_path = converted_dir / f"{path.stem}_qnn.json"

    if not converted_path.exists() or converted_path.stat().st_mtime < path.stat().st_mtime:
        policy = sf_to_qnn(
            sf_checkpoint_path=path,
            obs_dim=int(meta["obs_dim"]),
            trunk_hidden=int(meta["trunk_hidden"]),
            gru_hidden=int(meta["gru_hidden"]),
            use_gru=bool(meta["use_gru"]),
            device="cpu",
            d_model=int(meta["d_model"]),
            n_heads=int(meta["n_heads"]),
            n_layers=int(meta["n_layers"]),
            ffn_dim=int(meta["ffn_dim"]),
            action_history_tokens=int(meta["action_history_tokens"]),
            attn_dropout=float(meta["attn_dropout"]),
            readout=str(meta["readout"]),
        )
        policy.save(converted_path)
        converted_sidecar_path.write_text(
            json.dumps({"source_checkpoint": str(path), "converted_checkpoint": str(converted_path)}, indent=2),
            encoding="utf-8",
        )

    return str(converted_path)


def _next_archive_path(path: Path) -> Path:
    attempt = 1
    while True:
        candidate = path.with_name(f"{path.name}_old{attempt:04d}")
        if not candidate.exists():
            return candidate
        attempt += 1


def _archive_path_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    archived = _next_archive_path(path)
    path.rename(archived)
    return archived


def _assert_no_legacy_ppo_layout(run_cfg: dict[str, Any]) -> None:
    legacy_dir = run_output_dirs(run_cfg)["checkpoints"] / "ppo"
    if legacy_dir.exists():
        raise RuntimeError(
            f"Legacy PPO checkpoint layout is unsupported: {legacy_dir}. "
            "PPO artifacts now live directly under run.json.output.checkpoints."
        )


def _ppo_checkpoint_paths(run_cfg: dict[str, Any]) -> list[Path]:
    checkpoints_dir = run_output_dirs(run_cfg)["checkpoints"]
    return sorted(checkpoints_dir.glob("checkpoint_p*/*.pth"))


def _latest_ppo_checkpoint(run_cfg: dict[str, Any]) -> Path | None:
    checkpoints = _ppo_checkpoint_paths(run_cfg)
    if not checkpoints:
        return None
    return checkpoints[-1]


def _prepare_ppo_run_outputs(run_cfg: dict[str, Any], *, resume: bool) -> bool:
    _assert_no_legacy_ppo_layout(run_cfg)
    outputs = run_output_dirs(run_cfg)
    existing = _latest_ppo_checkpoint(run_cfg)
    if resume:
        if existing is not None:
            return True

    for path in (
        outputs["checkpoints"] / ".summary",
        outputs["checkpoints"] / "best",
        outputs["metrics"] / "eval",
        outputs["checkpoints"] / "config.json",
        outputs["checkpoints"] / "sf_log.txt",
        outputs["checkpoints"] / "git.diff",
    ):
        _archive_path_if_exists(path)
    for path in outputs["checkpoints"].glob("checkpoint_p*"):
        _archive_path_if_exists(path)
    return False


def _prepare_bc_run_outputs(run_cfg: dict[str, Any], *, resume: bool) -> None:
    outputs = run_output_dirs(run_cfg)
    checkpoint_path = outputs["checkpoints"] / "bc_training_checkpoint.pt"
    if resume:
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"run.json.resume is true, but no BC checkpoint exists at {checkpoint_path}"
            )
        return

    for path in (
        outputs["checkpoints"] / "bc_training_checkpoint.pt",
        outputs["checkpoints"] / "bc_best_model.npz",
        outputs["checkpoints"] / "bc_history.json",
        outputs["checkpoints"] / "bc_summary.json",
        outputs["checkpoints"] / "bc_manifest.json",
        outputs["checkpoints"] / "checkpoints",
    ):
        _archive_path_if_exists(path)


def _prepare_eval_outputs(run_cfg: dict[str, Any], *, resume: bool) -> None:
    if resume:
        return
    outputs = run_output_dirs(run_cfg)
    _archive_path_if_exists(outputs["metrics"] / "eval")


def _ensure_worker(worker_binary: Path, rebuild: bool) -> Path:
    if worker_binary.exists() and not rebuild:
        return worker_binary
    build_script = Path("src/engine/build/build_quake_worker.sh")
    subprocess.run(["bash", str(build_script), str(worker_binary)], check=True)
    return worker_binary


def _ensure_demo_worker(demo_worker_binary: Path, rebuild: bool) -> Path:
    if demo_worker_binary.exists() and not rebuild:
        return demo_worker_binary
    build_script = Path("src/engine/build/build_quake_demo_worker.sh")
    subprocess.run(["bash", str(build_script), str(demo_worker_binary)], check=True)
    return demo_worker_binary


def _run_live_check(
    worker_binary: Path,
    asset_root: Path,
    map_id: str,
    tick_hz: int,
    *,
    native_args: list[str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = {"QUAKE_BASEDIR": str(asset_root)}
    with NativeTokenProcess(
        executable=worker_binary,
        map_id=map_id,
        fixed_tick_hz=tick_hz,
        env=env,
        extra_args=native_args,
    ) as proc:
        hello = proc.start()
        reset_tick = proc.reset(seed=7, options=options)
        step_tick = proc.step({**idle_action(), "move": [1.0, 0.0]})
    result = {
        "worker_binary": str(worker_binary),
        "asset_root": str(asset_root),
        "hello_server": hello["server"],
        "capabilities": list(hello.get("capabilities", [])),
        "native_args": list(native_args or []),
        "reset_tick": int(reset_tick.tick),
        "step_tick": int(step_tick.tick),
        "step_done": bool(step_tick.done),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _scenario_config_json(config: Mapping[str, Any]) -> str:
    path_value = str(_require_cfg_value(config, "scenario_config_path", "PPO config")).strip()
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
        "trunk_hidden",
        "gru_hidden",
        "use_gru",
        "d_model",
        "n_heads",
        "n_layers",
        "ffn_dim",
        "attn_dropout",
        "readout",
        "action_history_tokens",
    )

    try:
        payload = trusted_torch_load(init_ckpt, map_location="cpu")
    except Exception:
        return

    meta: dict[str, Any] | None = None
    if isinstance(payload, dict):
        if "meta" in payload and isinstance(payload["meta"], dict):
            meta = payload["meta"]
        elif "trunk_hidden" in payload:
            meta = payload

    if meta is None:
        return

    mismatches: list[str] = []
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
    init_ckpt = str(checkpoint_path or _require_cfg_string(ppo_cfg, "init_ckpt", "PPO config"))
    try:
        payload = trusted_torch_load(str(init_ckpt), map_location="cpu")
        if isinstance(payload, dict) and "meta" in payload:
            meta = payload["meta"]
            if isinstance(meta, dict) and "obs_dim" in meta:
                return int(meta["obs_dim"])
    except Exception as exc:
        raise RuntimeError(f"Unable to read obs_dim from warm-start checkpoint metadata: {init_ckpt}") from exc
    raise RuntimeError(f"Warm-start checkpoint metadata is missing obs_dim: {init_ckpt}")


def _run_ppo(
    ppo_cfg: dict[str, Any],
    resolved_asset_root: Path,
    worker_path: Path,
    device: str,
) -> dict[str, Any]:
    from quake_ai.ppo.checkpoint_converter import sf_to_qnn
    from quake_ai.ppo.train import build_ppo_cfg, register_quake_components, run_ppo

    register_quake_components()

    output_dir = str(Path(_require_cfg_string(ppo_cfg, "output_dir", "PPO config")))
    scenario = _require_cfg_string(ppo_cfg, "map_id", "PPO config")
    num_workers = int(_require_cfg_value(ppo_cfg, "num_workers", "PPO config"))
    num_envs_per_worker = int(_require_cfg_value(ppo_cfg, "num_envs_per_worker", "PPO config"))
    num_envs = int(_require_cfg_value(ppo_cfg, "num_envs", "PPO config"))
    if num_envs != num_workers * num_envs_per_worker:
        raise RuntimeError(
            f"PPO config num_envs={num_envs} does not match num_workers*num_envs_per_worker="
            f"{num_workers * num_envs_per_worker}"
        )
    rollout = int(_require_cfg_value(ppo_cfg, "rollout_steps", "PPO config"))
    total_steps = int(_require_cfg_value(ppo_cfg, "total_steps", "PPO config"))
    total_env_steps = total_steps * num_envs

    native_args = _require_cfg_list(ppo_cfg, "native_args", "PPO config")
    options = _require_cfg_mapping(ppo_cfg, "options", "PPO config")
    procgen_cfg = _require_cfg_value(ppo_cfg, "procgen", "PPO config")
    resume = bool(_require_cfg_value(ppo_cfg, "resume", "PPO config"))
    init_checkpoint = str(ppo_cfg.get("init_ckpt", ""))

    cfg = build_ppo_cfg(
        scenario=scenario,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        worker_num_splits=int(_require_cfg_value(ppo_cfg, "worker_num_splits", "PPO config")),
        rollout=rollout,
        total_env_steps=total_env_steps,
        output_dir=output_dir,
        experiment="",
        executable=str(worker_path),
        basedir=str(resolved_asset_root),
        native_workdir=_require_cfg_string(ppo_cfg, "native_workdir", "PPO config"),
        native_args_json=json.dumps(native_args),
        options_json=json.dumps(options),
        procgen_json=json.dumps(procgen_cfg) if procgen_cfg is not None else "",
        scenario_config_json=_scenario_config_json(ppo_cfg),
        mode=_require_cfg_string(ppo_cfg, "mode", "PPO config"),
        max_steps_per_episode=int(_require_cfg_value(ppo_cfg, "max_steps_per_episode", "PPO config")),
        fixed_tick_hz=int(_require_cfg_value(ppo_cfg, "fixed_tick_hz", "PPO config")),
        seed=int(_require_cfg_value(ppo_cfg, "seed", "PPO config")),
        device=device,
        init_checkpoint=init_checkpoint,
        init_checkpoints=ppo_cfg.get("init_ckpts"),
        resume=resume,
        trunk_hidden=int(_require_cfg_value(ppo_cfg, "trunk_hidden", "PPO config")),
        gru_hidden=int(_require_cfg_value(ppo_cfg, "gru_hidden", "PPO config")),
        use_gru=bool(_require_cfg_value(ppo_cfg, "use_gru", "PPO config")),
        d_model=int(_require_cfg_value(ppo_cfg, "d_model", "PPO config")),
        n_heads=int(_require_cfg_value(ppo_cfg, "n_heads", "PPO config")),
        readout=_require_cfg_string(ppo_cfg, "readout", "PPO config"),
        n_layers=int(_require_cfg_value(ppo_cfg, "n_layers", "PPO config")),
        ffn_dim=int(_require_cfg_value(ppo_cfg, "ffn_dim", "PPO config")),
        action_history_tokens=int(_require_cfg_value(ppo_cfg, "action_history_tokens", "PPO config")),
        attn_dropout=float(_require_cfg_value(ppo_cfg, "attn_dropout", "PPO config")),
        ppo_epochs=int(_require_cfg_value(ppo_cfg, "ppo_epochs", "PPO config")),
        lr=float(_require_cfg_value(ppo_cfg, "policy_lr", "PPO config")),
        entropy_coef=float(_require_cfg_value(ppo_cfg, "entropy_coef", "PPO config")),
        bc_kl_coef=float(_require_cfg_value(ppo_cfg, "bc_kl_coef", "PPO config")),
        clip_ratio=float(_require_cfg_value(ppo_cfg, "clip_ratio", "PPO config")),
        gamma=float(_require_cfg_value(ppo_cfg, "gamma", "PPO config")),
        gae_lambda=float(_require_cfg_value(ppo_cfg, "gae_lambda", "PPO config")),
        max_grad_norm=float(_require_cfg_value(ppo_cfg, "max_grad_norm", "PPO config")),
        value_coef=float(_require_cfg_value(ppo_cfg, "value_coef", "PPO config")),
        with_pbt=bool(_require_cfg_value(ppo_cfg, "with_pbt", "PPO config")),
        num_policies=int(_require_cfg_value(ppo_cfg, "num_policies", "PPO config")),
        pbt_period_env_steps=int(_require_cfg_value(ppo_cfg, "pbt_period_env_steps", "PPO config")),
        pbt_start_mutation=int(_require_cfg_value(ppo_cfg, "pbt_start_mutation", "PPO config")),
        pbt_replace_fraction=float(_require_cfg_value(ppo_cfg, "pbt_replace_fraction", "PPO config")),
        pbt_mutation_rate=float(_require_cfg_value(ppo_cfg, "pbt_mutation_rate", "PPO config")),
        pbt_optimize_gamma=bool(_require_cfg_value(ppo_cfg, "pbt_optimize_gamma", "PPO config")),
        minibatch_size=int(_require_cfg_value(ppo_cfg, "minibatch_size", "PPO config")),
        policy_workers_per_policy=int(ppo_cfg.get("policy_workers_per_policy", 1)),
        reward_json_path=_require_cfg_string(ppo_cfg, "reward_json_path", "PPO config"),
    )

    ppo_result = run_ppo(cfg)

    import glob as _glob
    import re as _re

    exp_dir = Path(output_dir)

    def _reward_from_name(path: str) -> float:
        match = _re.search(r"reward_([-\d.]+)\.pth$", path)
        return float(match.group(1)) if match else float("-inf")

    best_files = _glob.glob(f"{exp_dir}/checkpoint_p*/best_0*.pth")
    if best_files:
        ppo_ckpt: str | None = max(best_files, key=_reward_from_name)
    else:
        regular_files = sorted(_glob.glob(f"{exp_dir}/checkpoint_p*/checkpoint_*.pth"))
        ppo_ckpt = regular_files[-1] if regular_files else None

    if ppo_ckpt:
        best_dir = Path(output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        qnn_ckpt_path = best_dir / "best_model.pth"
        try:
            qnn_policy = sf_to_qnn(
                sf_checkpoint_path=ppo_ckpt,
                obs_dim=int(_require_cfg_value(ppo_cfg, "obs_dim", "PPO config")) if "obs_dim" in ppo_cfg else _detect_obs_dim_from_checkpoint(ppo_cfg, ppo_ckpt),
                trunk_hidden=int(_require_cfg_value(ppo_cfg, "trunk_hidden", "PPO config")),
                gru_hidden=int(_require_cfg_value(ppo_cfg, "gru_hidden", "PPO config")),
                use_gru=bool(_require_cfg_value(ppo_cfg, "use_gru", "PPO config")),
                device="cpu",
                d_model=int(_require_cfg_value(ppo_cfg, "d_model", "PPO config")),
                n_heads=int(_require_cfg_value(ppo_cfg, "n_heads", "PPO config")),
                n_layers=int(_require_cfg_value(ppo_cfg, "n_layers", "PPO config")),
                ffn_dim=int(_require_cfg_value(ppo_cfg, "ffn_dim", "PPO config")),
                action_history_tokens=int(_require_cfg_value(ppo_cfg, "action_history_tokens", "PPO config")),
                attn_dropout=float(_require_cfg_value(ppo_cfg, "attn_dropout", "PPO config")),
                readout=_require_cfg_string(ppo_cfg, "readout", "PPO config"),
            )
            qnn_policy.save(qnn_ckpt_path)
            ppo_result["best_model_path"] = str(qnn_ckpt_path)
            ppo_result["ppo_checkpoint_path"] = ppo_ckpt
            ppo_result["sf_checkpoint_path"] = ppo_ckpt
        except Exception as exc:
            ppo_result["checkpoint_convert_error"] = str(exc)

    return ppo_result


def run_from_run_dir(run_dir: Path) -> dict[str, Any]:
    run_cfg = load_run_config(run_dir)
    run_dir_resolved = Path(run_cfg["run_dir"])
    manifest = _require_cfg_mapping(run_cfg, "manifest", "run config")
    action = _require_cfg_string(run_cfg, "mode", "run config")
    runtime_scale = _require_cfg_string(run_cfg, "runtime_scale", "run config")
    resume = bool(_require_cfg_value(run_cfg, "resume", "run config"))

    machine = _require_cfg_mapping(run_cfg, "machine", "run config")
    resolved_device = _require_cfg_string(machine, "device", "machine.json")
    resolved_asset_root = _resolve_asset_root(_require_cfg_string(machine, "asset_root", "machine.json"))
    worker_path = Path(_require_cfg_string(machine, "worker_binary", "machine.json"))

    runtime, plan = build_runtime_plan_for_run(run_cfg, resolved_device)
    plan_path = write_run_plan(
        run_cfg,
        runtime_scale,
        runtime,
        plan,
        resolve_demo_dir_from_run(run_cfg) if action == "bc" else None,
        resolved_asset_root,
    )

    results: dict[str, Any] = {
        "run_dir": str(run_dir_resolved),
        "run_name": _require_cfg_string(manifest, "name", "run.json"),
        "mode": action,
        "runtime_scale": runtime_scale,
        "resume": resume,
        "plan_path": str(plan_path),
        "runtime": runtime,
        "plan": plan.to_dict(),
    }
    stage_timings: dict[str, float] = {}

    if action == "check":
        check_cfg = build_run_check_surface(run_cfg)
        _validate_native_mod_assets(
            resolved_asset_root,
            check_cfg.get("native_args") if isinstance(check_cfg.get("native_args"), list) else None,
        )
        worker_path = _ensure_worker(worker_path, rebuild=False)
        results["check"] = _run_live_check(
            worker_binary=worker_path,
            asset_root=resolved_asset_root,
            map_id=_require_cfg_string(check_cfg, "map_id", "check config"),
            tick_hz=int(_require_cfg_value(check_cfg, "tick_hz", "check config")),
            native_args=_require_cfg_list(check_cfg, "native_args", "check config"),
            options=_require_cfg_mapping(check_cfg, "options", "check config"),
        )
        report = write_run_report(
            run_root=run_dir_resolved,
            action=action,
            runtime_scale=runtime_scale,
            runtime=runtime,
            plan=plan.to_dict(),
            results=results,
            stage_timings=stage_timings,
        )
        results.update({
            "report_path": report["report_path"],
            "operational_note_json_path": report["operational_note_json_path"],
            "operational_note_md_path": report["operational_note_md_path"],
        })
        print(json.dumps(results, indent=2, sort_keys=True))
        return results

    if action == "bc":
        bc_cfg = build_run_bc_config(run_cfg, resolved_device)
        _prepare_bc_run_outputs(run_cfg, resume=resume)
        demo_dir = resolve_demo_dir_from_run(run_cfg)

        bc_ticks_path = Path(_require_cfg_string(bc_cfg, "token_ticks_path", "BC config"))
        if not bc_ticks_path.exists():
            started = time.monotonic()
            demo_worker_path = _ensure_demo_worker(Path(_require_cfg_string(machine, "demo_worker_binary", "machine.json")), rebuild=False)
            collect_cfg = CollectConfig(
                demo_worker_binary=str(demo_worker_path),
                demo_dir=str(demo_dir),
                output_dir=str(Path(_require_cfg_string(bc_cfg, "output_dir", "BC config")) / "collect"),
                map_id=_require_cfg_string(bc_cfg, "map_id", "BC config"),
                fixed_tick_hz=int(_require_cfg_value(bc_cfg, "fixed_tick_hz", "BC config")),
                asset_root=str(resolved_asset_root),
            )
            collect_result = collect_demo_tokens(collect_cfg)
            bc_cfg["token_ticks_path"] = collect_result.token_ticks_path
            bc_cfg["map_state_path"] = collect_result.map_state_path
            bc_cfg["map_states_path"] = collect_result.map_states_path
            bc_cfg["metadata_path"] = collect_result.metadata_path
            results["collect"] = {
                "token_ticks_path": collect_result.token_ticks_path,
                "map_state_path": collect_result.map_state_path,
                "map_states_path": collect_result.map_states_path,
                "metadata_path": collect_result.metadata_path,
                "missing_demos_path": collect_result.missing_demos_path,
                "source_summary_path": collect_result.source_summary_path,
                "demos_processed": collect_result.demos_processed,
                "demos_failed": collect_result.demos_failed,
                "total_ticks": collect_result.total_ticks,
            }
            stage_timings["collect"] = time.monotonic() - started
        else:
            results["collect"] = {"skipped": True, "reason": "BC data already available."}

        started = time.monotonic()
        results["bc"] = run_behavior_cloning(BCConfig(**bc_cfg))
        stage_timings["bc"] = time.monotonic() - started
        results["stage_timings"] = stage_timings
        report = write_run_report(
            run_root=run_dir_resolved,
            action=action,
            runtime_scale=runtime_scale,
            runtime=runtime,
            plan=plan.to_dict(),
            results=results,
            stage_timings=stage_timings,
        )
        results.update({
            "report_path": report["report_path"],
            "operational_note_json_path": report["operational_note_json_path"],
            "operational_note_md_path": report["operational_note_md_path"],
        })
        print(json.dumps(results, indent=2, sort_keys=True))
        return results

    if action == "ppo":
        will_resume_ppo = _prepare_ppo_run_outputs(run_cfg, resume=resume)
        ppo_cfg, eval_cfg = build_run_ppo_eval_config(run_cfg, resolved_device)
        seed_ckpt = _require_cfg_string(run_cfg, "checkpoint_path", "run config")
        if not seed_ckpt:
            raise RuntimeError("run.json.checkpoint_path must be non-empty when mode is 'ppo'")
        if not Path(seed_ckpt).exists():
            raise FileNotFoundError(f"Seed checkpoint from run.json does not exist: {seed_ckpt}")
        ppo_cfg["resume"] = will_resume_ppo
        if will_resume_ppo:
            latest_ckpt = _latest_ppo_checkpoint(run_cfg)
            if latest_ckpt is None:
                raise RuntimeError("PPO resume requested, but no latest checkpoint could be located")
            results["ppo_resume_from"] = str(latest_ckpt)
        else:
            ppo_cfg["init_ckpt"] = seed_ckpt
            results["ppo_init_ckpt"] = seed_ckpt
            if resume:
                results["ppo_resume_fallback"] = "No existing PPO checkpoint found; started from seed checkpoint."
            _validate_warm_start_arch(seed_ckpt, ppo_cfg)
        _validate_native_mod_assets(
            resolved_asset_root,
            ppo_cfg.get("native_args") if isinstance(ppo_cfg.get("native_args"), list) else None,
        )
        worker_path = _ensure_worker(worker_path, rebuild=False)

        started = time.monotonic()
        ppo_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        ppo_cfg["native_executable"] = str(worker_path)
        results["ppo"] = _run_ppo(ppo_cfg, resolved_asset_root, worker_path, resolved_device)
        stage_timings["ppo"] = time.monotonic() - started

        started = time.monotonic()
        _prepare_eval_outputs(run_cfg, resume=False)
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        eval_cfg["checkpoint_path"] = _prepare_eval_checkpoint(
            str(eval_cfg["checkpoint_path"]),
            str(eval_cfg["output_dir"]),
        )
        results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

        results["stage_timings"] = stage_timings
        report = write_run_report(
            run_root=run_dir_resolved,
            action=action,
            runtime_scale=runtime_scale,
            runtime=runtime,
            plan=plan.to_dict(),
            results=results,
            stage_timings=stage_timings,
        )
        results.update({
            "report_path": report["report_path"],
            "operational_note_json_path": report["operational_note_json_path"],
            "operational_note_md_path": report["operational_note_md_path"],
        })
        print(json.dumps(results, indent=2, sort_keys=True))
        return results

    if action == "eval":
        _prepare_eval_outputs(run_cfg, resume=resume)
        eval_cfg = build_run_eval_config(run_cfg, resolved_device)
        _validate_native_mod_assets(
            resolved_asset_root,
            eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None,
        )
        worker_path = _ensure_worker(worker_path, rebuild=False)
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        eval_cfg["checkpoint_path"] = _prepare_eval_checkpoint(
            _require_cfg_string(run_cfg, "checkpoint_path", "run config"),
            str(eval_cfg["output_dir"]),
        )

        started = time.monotonic()
        results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started
        results["stage_timings"] = stage_timings
        report = write_run_report(
            run_root=run_dir_resolved,
            action=action,
            runtime_scale=runtime_scale,
            runtime=runtime,
            plan=plan.to_dict(),
            results=results,
            stage_timings=stage_timings,
        )
        results.update({
            "report_path": report["report_path"],
            "operational_note_json_path": report["operational_note_json_path"],
            "operational_note_md_path": report["operational_note_md_path"],
        })
        print(json.dumps(results, indent=2, sort_keys=True))
        return results

    raise RuntimeError(f"Unsupported run mode in {run_dir_resolved / 'run.json'}: {action}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training pipeline — run.json is the single entry point")
    parser.add_argument("--run-dir", required=True, help="Run directory containing run.json")
    args = parser.parse_args()
    run_from_run_dir(Path(args.run_dir))


if __name__ == "__main__":
    main()
