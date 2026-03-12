"""Training pipeline — BC bootstrap and PPO as separate phases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

from engine.bridge import NativeWorldProcess as NativeEngineProcess
from quake_ai.actions import LOOK_NEUTRAL_LABEL
from quake_ai.rl.evaluation import EvalConfig, run_evaluation
from quake_ai.rl.planning import (
    RuntimePlan,
    _profile_output_root,
    _resolve_asset_root,
    _resolve_demo_dir,
    _resolve_ppo_init_checkpoint,
    _validate_native_mod_assets,
    _write_plan,
    build_runtime_plan,
)
from quake_ai.rl.collect import CollectConfig, collect_demo_tokens
from quake_ai.rl.profiles import PROFILES, LiveProfile, load_config_with_runtime
from quake_ai.rl.reporting import REPORT_STAGES, _load_existing_runtime_context, _write_run_report
from quake_ai.rl.training_bc import BCConfig, run_behavior_cloning
from quake_ai.utils.io import read_json, trusted_torch_load


def _ensure_worker(worker_binary: Path, rebuild: bool) -> Path:
    if worker_binary.exists() and not rebuild:
        return worker_binary
    build_script = Path("engine/build/build_quake_worker.sh")
    subprocess.run(["bash", str(build_script), str(worker_binary)], check=True)
    return worker_binary


def _ensure_demo_worker(demo_worker_binary: Path, rebuild: bool) -> Path:
    if demo_worker_binary.exists() and not rebuild:
        return demo_worker_binary
    build_script = Path("engine/build/build_quake_demo_worker.sh")
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
    with NativeEngineProcess(
        executable=worker_binary,
        map_id=map_id,
        fixed_tick_hz=tick_hz,
        env=env,
        extra_args=native_args,
    ) as proc:
        hello = proc.start()
        reset = proc.reset(seed=7, options=options)
        step = proc.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
    result = {
        "worker_binary": str(worker_binary),
        "asset_root": str(asset_root),
        "hello_server": hello["server"],
        "capabilities": list(hello.get("capabilities", [])),
        "native_args": list(native_args or []),
        "reset_info": dict(reset.get("info", {})),
        "reset_tick": int(reset["world_tick"]["tick"]),
        "step_tick": int(step["world_tick"]["tick"]),
        "step_done": bool(step.get("done", False)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _scenario_config_json(config: Mapping[str, Any]) -> str:
    path_value = str(config.get("scenario_config_path", "")).strip()
    if not path_value:
        return ""
    payload = read_json(path_value)
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"scenario_config_path must define a non-empty scenarios list: {path_value}")
    return json.dumps(scenarios)


def _seed_bc_collect_paths(profile: LiveProfile, bc_cfg: dict[str, Any]) -> None:
    collect_dir = Path(profile.collect_out)
    if not str(bc_cfg.get("token_ticks_path", "")).strip():
        token_ticks_path = collect_dir / "token_ticks.bin"
        if token_ticks_path.exists():
            bc_cfg["token_ticks_path"] = str(token_ticks_path)
    if not str(bc_cfg.get("map_state_path", "")).strip():
        map_state_path = collect_dir / "world_map.json"
        if map_state_path.exists():
            bc_cfg["map_state_path"] = str(map_state_path)
    if not str(bc_cfg.get("map_states_path", "")).strip():
        map_states_path = collect_dir / "map_states.json"
        if map_states_path.exists():
            bc_cfg["map_states_path"] = str(map_states_path)
    if not str(bc_cfg.get("metadata_path", "")).strip():
        metadata_path = collect_dir / "demo_metadata.ndjson"
        if metadata_path.exists():
            bc_cfg["metadata_path"] = str(metadata_path)


def _detect_obs_dim(ppo_cfg: dict[str, Any]) -> int:
    """Detect obs_dim from the init_ckpt meta, falling back to 0."""
    init_ckpt = ppo_cfg.get("init_ckpt", "")
    if not init_ckpt:
        return 0
    try:
        payload = trusted_torch_load(str(init_ckpt), map_location="cpu")
        if isinstance(payload, dict) and "meta" in payload:
            return int(payload["meta"].get("obs_dim", 0))
    except Exception:
        pass
    return 0


def _run_sf_ppo(
    profile: Any,
    ppo_cfg: dict[str, Any],
    resolved_asset_root: Path,
    worker_path: Path,
    device: str,
) -> dict[str, Any]:
    """Launch Sample Factory APPO as the PPO stage.

    After training, the best SF checkpoint is converted to MLPGRUPolicy format
    so that evaluation.py can load it without changes.
    """
    from quake_ai.sf.train import register_quake_components, build_sf_cfg, run_sf
    from quake_ai.sf.checkpoint_converter import sf_to_bc

    register_quake_components()

    output_dir = str(Path(ppo_cfg.get("output_dir", "../artifacts/runs/sf_ppo")))
    scenario = str(ppo_cfg.get("map_id", "dm4"))
    num_workers = int(ppo_cfg.get("num_envs", 8))
    rollout = int(ppo_cfg.get("rollout_steps", 256))
    total_steps = int(ppo_cfg.get("total_steps", 1_000_000))
    total_env_steps = total_steps * num_workers

    native_args = ppo_cfg.get("native_args", [])
    options = ppo_cfg.get("options", {})

    native_args_json = json.dumps(list(native_args)) if native_args else '["-game","frikbotnex_train"]'
    options_json = json.dumps(dict(options)) if options else ""
    scenario_config_json = _scenario_config_json(ppo_cfg)

    cfg = build_sf_cfg(
        scenario=scenario,
        num_workers=num_workers,
        rollout=rollout,
        total_env_steps=total_env_steps,
        output_dir=output_dir,
        experiment=f"quake_{profile.scenario_id or scenario}",
        executable=str(worker_path),
        basedir=str(resolved_asset_root),
        native_workdir=str(ppo_cfg.get("native_workdir", "") or ""),
        native_args_json=native_args_json,
        options_json=options_json,
        scenario_config_json=scenario_config_json,
        mode=str(ppo_cfg.get("mode", "pvp")),
        max_steps_per_episode=int(ppo_cfg.get("max_steps_per_episode", 1024)),
        seed=int(ppo_cfg.get("seed", 17)),
        device=device,
        init_checkpoint=str(ppo_cfg.get("init_ckpt", "")),
        trunk_hidden=int(ppo_cfg.get("trunk_hidden", 128)),
        gru_hidden=int(ppo_cfg.get("gru_hidden", 128)),
        use_gru=bool(ppo_cfg.get("use_gru", True)),
        d_model=int(ppo_cfg.get("d_model", 64)),
        n_heads=int(ppo_cfg.get("n_heads", 2)),
        ffn_dim=int(ppo_cfg.get("ffn_dim", 256)),
        ppo_epochs=int(ppo_cfg.get("ppo_epochs", 2)),
        lr=float(ppo_cfg.get("policy_lr", 0.00025)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.002)),
        bc_kl_coef=float(ppo_cfg.get("bc_kl_coef", 0.05)),
        clip_ratio=float(ppo_cfg.get("clip_ratio", 0.2)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
    )

    sf_result = run_sf(cfg)

    import glob as _glob
    train_dir = Path(output_dir)
    exp_name = f"quake_{profile.scenario_id or scenario}"
    ckpt_pattern = str(train_dir / exp_name / "checkpoint_p0" / "checkpoint_*.pth")
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    sf_ckpt = ckpt_files[-1] if ckpt_files else None

    if sf_ckpt:
        bc_ckpt_path = train_dir / exp_name / "best_model.pth"
        try:
            bc_policy = sf_to_bc(
                sf_checkpoint_path=sf_ckpt,
                obs_dim=ppo_cfg.get("obs_dim", 0) or _detect_obs_dim(ppo_cfg),
                trunk_hidden=int(ppo_cfg.get("trunk_hidden", 128)),
                gru_hidden=int(ppo_cfg.get("gru_hidden", 128)),
                use_gru=bool(ppo_cfg.get("use_gru", True)),
                device="cpu",
            )
            bc_policy.save(bc_ckpt_path)
            sf_result["best_model_path"] = str(bc_ckpt_path)
            sf_result["sf_checkpoint_path"] = sf_ckpt
        except Exception as exc:
            sf_result["checkpoint_convert_error"] = str(exc)

    return sf_result


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

def _setup_pipeline(
    profile_name: str,
    action: str,
    eval_bc: bool,
    device: str,
    demo_dir: str | None,
    asset_root: str | None,
    worker_binary: str,
    rebuild_worker: bool,
) -> tuple[LiveProfile, dict[str, Any], RuntimePlan, Path, Path, Path, dict, dict, dict, dict]:
    """Common setup for both BC and PPO phases."""
    profile, runtime, plan = build_runtime_plan(profile_name, device)
    resolved_demo_dir = _resolve_demo_dir(profile, demo_dir)
    resolved_asset_root = _resolve_asset_root(asset_root)
    worker_path = Path(worker_binary)
    needs_live_worker = action in {"check", "ppo", "eval", "eval-bc"} or (action == "bc" and eval_bc)
    if needs_live_worker or rebuild_worker:
        worker_path = _ensure_worker(worker_path, rebuild=rebuild_worker)
    bc_cfg, ppo_cfg, eval_cfg = load_config_with_runtime(profile, plan, device)
    _seed_bc_collect_paths(profile, bc_cfg)
    if needs_live_worker:
        _validate_native_mod_assets(resolved_asset_root, ppo_cfg.get("native_args") if isinstance(ppo_cfg.get("native_args"), list) else None)
        _validate_native_mod_assets(resolved_asset_root, eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None)
    plan_path = _write_plan(profile, runtime, plan, resolved_demo_dir, resolved_asset_root)

    results: dict[str, Any] = {
        "profile": profile.name,
        "plan_path": str(plan_path),
        "runtime": runtime,
        "plan": plan.to_dict(),
        "worker_binary": str(worker_path),
        "asset_root": str(resolved_asset_root),
        "demo_dir": str(resolved_demo_dir),
    }

    return profile, runtime, plan, resolved_demo_dir, resolved_asset_root, worker_path, bc_cfg, ppo_cfg, eval_cfg, results


def run_live_pipeline(
    *,
    profile_name: str,
    action: str,
    eval_bc: bool,
    device: str,
    demo_dir: str | None,
    asset_root: str | None,
    worker_binary: str,
    rebuild_worker: bool,
) -> dict[str, Any]:
    profile = PROFILES[profile_name]

    # Report mode: load existing artifacts, no execution.
    if action == "report":
        runtime, plan_dict, plan_path = _load_existing_runtime_context(profile)
        report_artifacts = _write_run_report(
            profile=profile,
            action=action,
            runtime=runtime,
            plan=plan_dict,
            plan_path=plan_path,
            results={"profile": profile.name, "source": "existing_artifacts"},
            stage_timings={},
        )
        report = dict(report_artifacts["report"])
        report["report_path"] = report_artifacts["report_path"]
        report["operational_note_json_path"] = report_artifacts["operational_note_json_path"]
        report["operational_note_md_path"] = report_artifacts["operational_note_md_path"]
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    profile, runtime, plan, resolved_demo_dir, resolved_asset_root, worker_path, bc_cfg, ppo_cfg, eval_cfg, results = _setup_pipeline(
        profile_name, action, eval_bc, device, demo_dir, asset_root, worker_binary, rebuild_worker,
    )
    stage_timings: dict[str, float] = {}

    if action == "plan":
        print(json.dumps(results, indent=2, sort_keys=True))
        return results

    if action == "check":
        results["check"] = _run_live_check(
            worker_binary=worker_path,
            asset_root=resolved_asset_root,
            map_id=str(ppo_cfg.get("map_id", "dm4")),
            tick_hz=int(ppo_cfg.get("fixed_tick_hz", 20)),
            native_args=[str(value) for value in ppo_cfg.get("native_args", [])] if isinstance(ppo_cfg.get("native_args"), list) else None,
            options=dict(ppo_cfg.get("options", {})) if isinstance(ppo_cfg.get("options"), Mapping) else None,
        )
        return results

    # ------------------------------------------------------------------
    # BC phase: collect → BC → eval-bc
    # ------------------------------------------------------------------
    if action == "bc":
        # Collect demo tokens if not already available.
        bc_ticks_path = bc_cfg.get("token_ticks_path", "")
        bc_ticks_exist = bool(bc_ticks_path) and Path(str(bc_ticks_path)).exists()
        if not bc_ticks_exist:
            started = time.monotonic()
            demo_worker_path = _ensure_demo_worker(
                Path(str(worker_path).replace("quake_worker", "quake_demo_worker")),
                rebuild=rebuild_worker,
            )
            collect_cfg = CollectConfig(
                demo_worker_binary=str(demo_worker_path),
                demo_dir=str(resolved_demo_dir),
                output_dir=str(Path(profile.collect_out)),
                map_id=str(bc_cfg.get("map_id", "dm4")),
                fixed_tick_hz=int(bc_cfg.get("fixed_tick_hz", 0)),
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

        # Behavior cloning.
        started = time.monotonic()
        results["bc"] = run_behavior_cloning(BCConfig(**bc_cfg))
        stage_timings["bc"] = time.monotonic() - started

        if eval_bc:
            started = time.monotonic()
            eval_bc_cfg = dict(eval_cfg)
            eval_bc_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
            eval_bc_cfg["native_executable"] = str(worker_path)
            eval_bc_cfg["checkpoint_path"] = str(Path(bc_cfg["output_dir"]) / "bc_best_model.npz")
            eval_bc_cfg["output_dir"] = str(_profile_output_root(profile) / "eval_bc")
            results["eval_bc"] = run_evaluation(EvalConfig(**eval_bc_cfg))
            stage_timings["eval_bc"] = time.monotonic() - started
        else:
            results["eval_bc"] = {"skipped": True, "reason": "--eval-bc not set"}

    # ------------------------------------------------------------------
    # PPO phase: PPO → eval
    # ------------------------------------------------------------------
    elif action == "ppo":
        selected_init_ckpt, init_ckpt_note = _resolve_ppo_init_checkpoint(ppo_cfg, bc_cfg)
        if selected_init_ckpt:
            ppo_cfg["init_ckpt"] = selected_init_ckpt
            results["ppo_init_ckpt"] = selected_init_ckpt
        if init_ckpt_note:
            results["ppo_init_ckpt_note"] = init_ckpt_note

        started = time.monotonic()
        ppo_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        ppo_cfg["native_executable"] = str(worker_path)
        results["ppo"] = _run_sf_ppo(profile, ppo_cfg, resolved_asset_root, worker_path, device)
        stage_timings["ppo"] = time.monotonic() - started

        started = time.monotonic()
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

    # ------------------------------------------------------------------
    # Individual stage actions (collect, eval-bc, eval) for debugging.
    # ------------------------------------------------------------------
    elif action == "collect":
        started = time.monotonic()
        demo_worker_path = _ensure_demo_worker(
            Path(str(worker_path).replace("quake_worker", "quake_demo_worker")),
            rebuild=rebuild_worker,
        )
        collect_cfg = CollectConfig(
            demo_worker_binary=str(demo_worker_path),
            demo_dir=str(resolved_demo_dir),
            output_dir=str(Path(profile.collect_out)),
            map_id=str(bc_cfg.get("map_id", "dm4")),
            fixed_tick_hz=int(bc_cfg.get("fixed_tick_hz", 0)),
            asset_root=str(resolved_asset_root),
        )
        collect_result = collect_demo_tokens(collect_cfg)
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

    elif action == "eval-bc":
        started = time.monotonic()
        eval_bc_cfg = dict(eval_cfg)
        eval_bc_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_bc_cfg["native_executable"] = str(worker_path)
        eval_bc_cfg["checkpoint_path"] = str(Path(bc_cfg["output_dir"]) / "bc_best_model.npz")
        eval_bc_cfg["output_dir"] = str(_profile_output_root(profile) / "eval_bc")
        results["eval_bc"] = run_evaluation(EvalConfig(**eval_bc_cfg))
        stage_timings["eval_bc"] = time.monotonic() - started

    elif action == "eval":
        started = time.monotonic()
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

    # Write report.
    report_artifacts = _write_run_report(
        profile=profile,
        action=action,
        runtime=runtime,
        plan=plan.to_dict(),
        plan_path=Path(results["plan_path"]),
        results=results,
        stage_timings=stage_timings,
    )
    results["report_path"] = report_artifacts["report_path"]
    results["operational_note_json_path"] = report_artifacts["operational_note_json_path"]
    results["operational_note_md_path"] = report_artifacts["operational_note_md_path"]

    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Training pipeline — BC bootstrap or PPO training")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="combat-bot-multi")
    parser.add_argument(
        "--action",
        choices=["plan", "report", "check", "collect", "bc", "eval-bc", "ppo", "eval"],
        default="bc",
        help="Phase to run: 'bc' (collect+BC; add --eval-bc to run eval-bc), 'ppo' (PPO+eval), or individual stages",
    )
    parser.add_argument("--eval-bc", action="store_true", help="When used with --action bc, run eval-bc after behavior cloning")
    parser.add_argument("--device", default="gpu", help="Requested torch device override")
    parser.add_argument("--demo-dir", default=None, help="Override demo directory")
    parser.add_argument("--asset-root", default=None, help="Override Quake asset root")
    parser.add_argument("--worker-binary", default="../artifacts/bin/quake_worker", help="Path to the live worker binary")
    parser.add_argument("--rebuild-worker", action="store_true", help="Force a rebuild of the live worker binary")
    args = parser.parse_args()

    run_live_pipeline(
        profile_name=args.profile,
        action=args.action,
        eval_bc=args.eval_bc,
        device=args.device,
        demo_dir=args.demo_dir,
        asset_root=args.asset_root,
        worker_binary=args.worker_binary,
        rebuild_worker=args.rebuild_worker,
    )


if __name__ == "__main__":
    main()
