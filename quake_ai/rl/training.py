"""Training pipeline — BC bootstrap and PPO as separate phases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from engine.bridge import NativeTokenProcess
from quake_ai.actions import LOOK_NEUTRAL_LABEL
from quake_ai.rl.evaluation import EvalConfig, run_evaluation
from quake_ai.rl.planning import (
    RuntimePlan,
    _profile_output_root,
    _resolve_asset_root,
    _resolve_demo_dir,
    _validate_native_mod_assets,
    _write_plan,
    build_runtime_plan,
)
from quake_ai.rl.collect import CollectConfig, collect_demo_tokens
from quake_ai.rl.profiles import (
    BC_CHECKPOINT, BC_COLLECT_DIR, BC_OUTPUT_DIR,
    PROFILES, LiveProfile, build_bc_config, load_config_with_runtime,
)
from quake_ai.rl.reporting import REPORT_STAGES, _load_existing_runtime_context, _write_run_report
from quake_ai.rl.training_bc import BCConfig, run_behavior_cloning
from quake_ai.utils.io import read_json, trusted_torch_load


def _is_sf_checkpoint_payload(payload: object) -> bool:
    return isinstance(payload, dict) and "model" in payload and ("train_step" in payload or "env_steps" in payload)


def _reward_from_checkpoint_name(path: Path) -> float:
    import re as _re

    match = _re.search(r"reward_([-\d.]+)\.pth$", path.name)
    return float(match.group(1)) if match else float("-inf")


def _best_sf_checkpoint(best_dir: Path) -> Path | None:
    candidates = [path for path in best_dir.glob("best_*.pth") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_reward_from_checkpoint_name(path), path.stat().st_mtime))


def _prepare_eval_checkpoint(profile: LiveProfile, checkpoint_path: str, output_dir: str) -> str:
    path = Path(checkpoint_path)
    if not path.exists() and path.name == "best_model.pth":
        fallback = _best_sf_checkpoint(path.parent)
        if fallback is not None:
            path = fallback

    if not path.exists() or path.suffix != ".pth":
        return str(path)

    try:
        payload = trusted_torch_load(str(path), map_location="cpu")
    except Exception:
        return str(path)

    if not _is_sf_checkpoint_payload(payload):
        return str(path)

    from quake_ai.model.policy import QNNPolicy
    from quake_ai.ppo.checkpoint_converter import sf_to_qnn

    bc_policy = QNNPolicy.load(str(profile.bc_checkpoint), device="cpu")
    converted_dir = Path(output_dir).parent / "_eval_ckpts"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted_path = converted_dir / f"{path.stem}_qnn.pth"
    sidecar_path = converted_dir / f"{path.stem}_qnn.json"

    if not converted_path.exists() or converted_path.stat().st_mtime < path.stat().st_mtime:
        policy = sf_to_qnn(
            sf_checkpoint_path=path,
            obs_dim=bc_policy.obs_dim,
            trunk_hidden=bc_policy.trunk_hidden,
            gru_hidden=bc_policy.gru_hidden,
            use_gru=bc_policy.use_gru,
            device="cpu",
        )
        policy.save(converted_path)
        sidecar_path.write_text(
            json.dumps({"source_checkpoint": str(path), "converted_checkpoint": str(converted_path)}, indent=2),
            encoding="utf-8",
        )

    return str(converted_path)


def _profile_with_output_root(profile: LiveProfile, output_root: str | None) -> LiveProfile:
    """Clone a profile so one run can target an explicit retained root."""
    if not output_root:
        return profile

    root = Path(output_root)
    ppo_overrides = dict(profile.ppo_overrides)
    eval_overrides = dict(profile.eval_overrides)

    ppo_overrides["output_dir"] = str(root)
    eval_overrides["output_dir"] = str(root / "eval")
    eval_checkpoint_name = Path(str(eval_overrides.get("checkpoint_path", "best_model.pth"))).name
    eval_overrides["checkpoint_path"] = str(root / "best" / eval_checkpoint_name)

    return replace(
        profile,
        plan_path=str(root / "live_training_plan.json"),
        ppo_overrides=ppo_overrides,
        eval_overrides=eval_overrides,
    )


def _assert_fresh_output_root(profile: LiveProfile) -> None:
    """Guard the canonical fresh-run path from silently resuming old state."""
    output_root = _profile_output_root(profile)
    if not output_root.exists():
        return

    ignorable = {
        "decision_hook.json",
        "loop_status.json",
        "loop_watch.log",
        "loop_watch.pid",
        "loop_watch_state.json",
        "loop_watch_stdout.log",
    }
    existing = sorted(path.name for path in output_root.iterdir() if path.name not in ignorable)
    if not existing:
        return

    preview = ", ".join(existing[:5])
    if len(existing) > 5:
        preview += ", ..."
    raise RuntimeError(
        f"Fresh run requested, but {output_root} already contains artifacts ({preview}). "
        "Choose a new --output-root for the run instead of reusing an existing root."
    )


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
        step_tick = proc.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
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
    path_value = str(config.get("scenario_config_path", "")).strip()
    if not path_value:
        return ""
    payload = read_json(path_value)
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"scenario_config_path must define a non-empty scenarios list: {path_value}")
    return json.dumps(scenarios)


def _seed_bc_collect_paths(bc_cfg: dict[str, Any]) -> None:
    """Auto-discover collected token data in the shared BC collect dir."""
    collect_dir = Path(BC_COLLECT_DIR)
    for key, filename in [
        ("token_ticks_path", "token_ticks.bin"),
        ("map_state_path", "world_map.json"),
        ("map_states_path", "map_states.json"),
        ("metadata_path", "demo_metadata.ndjson"),
    ]:
        if not str(bc_cfg.get(key, "")).strip():
            path = collect_dir / filename
            if path.exists():
                bc_cfg[key] = str(path)


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


def _run_ppo(
    profile: Any,
    ppo_cfg: dict[str, Any],
    resolved_asset_root: Path,
    worker_path: Path,
    device: str,
) -> dict[str, Any]:
    """Launch APPO as the PPO stage.

    After training, the best PPO checkpoint is converted to QNNPolicy format
    so that evaluation.py can load it without changes.
    """
    from quake_ai.ppo.train import register_quake_components, build_ppo_cfg, run_ppo
    from quake_ai.ppo.checkpoint_converter import sf_to_qnn

    register_quake_components()

    output_dir = str(Path(ppo_cfg.get("output_dir", "assets/runs/ppo")))
    scenario = str(ppo_cfg.get("map_id", "procgen"))
    num_envs = int(ppo_cfg.get("num_envs", 8))
    num_envs_per_worker = int(ppo_cfg.get("num_envs_per_worker", 1))
    num_workers = num_envs // num_envs_per_worker
    rollout = int(ppo_cfg.get("rollout_steps", 256))
    total_steps = int(ppo_cfg.get("total_steps", 1_000_000))
    total_env_steps = total_steps * num_envs

    native_args = ppo_cfg.get("native_args", [])
    options = ppo_cfg.get("options", {})

    native_args_json = json.dumps(list(native_args)) if native_args else '["-game","frikbotnex_train"]'
    options_json = json.dumps(dict(options)) if options else ""
    scenario_config_json = _scenario_config_json(ppo_cfg)

    cfg = build_ppo_cfg(
        scenario=scenario,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        worker_num_splits=int(ppo_cfg.get("worker_num_splits", 1)),
        rollout=rollout,
        total_env_steps=total_env_steps,
        output_dir=output_dir,
        experiment="ppo",
        executable=str(worker_path),
        basedir=str(resolved_asset_root),
        native_workdir=str(ppo_cfg.get("native_workdir", "") or ""),
        native_args_json=native_args_json,
        options_json=options_json,
        scenario_config_json=scenario_config_json,
        mode=str(ppo_cfg.get("mode", "pvp")),
        max_steps_per_episode=int(ppo_cfg.get("max_steps_per_episode", 1024)),
        fixed_tick_hz=int(ppo_cfg.get("fixed_tick_hz", 20)),
        seed=int(ppo_cfg.get("seed", 17)),
        device=device,
        init_checkpoint=str(ppo_cfg.get("init_ckpt", "")),
        init_checkpoints=ppo_cfg.get("init_ckpts"),
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
        with_pbt=bool(ppo_cfg.get("with_pbt", False)),
        num_policies=int(ppo_cfg.get("num_policies", 1)),
        pbt_period_env_steps=int(ppo_cfg.get("pbt_period_env_steps", 5_000_000)),
        pbt_start_mutation=int(ppo_cfg.get("pbt_start_mutation", 20_000_000)),
        pbt_replace_fraction=float(ppo_cfg.get("pbt_replace_fraction", 0.3)),
        pbt_mutation_rate=float(ppo_cfg.get("pbt_mutation_rate", 0.15)),
        pbt_optimize_gamma=bool(ppo_cfg.get("pbt_optimize_gamma", False)),
        minibatch_size=int(ppo_cfg.get("minibatch_size", 0)),
        record_demos=bool(ppo_cfg.get("record_demos", True)),
    )

    ppo_result = run_ppo(cfg)

    import glob as _glob
    import re as _re
    train_dir = Path(output_dir)
    exp_dir = train_dir / "ppo"

    def _reward_from_name(path: str) -> float:
        m = _re.search(r"reward_([-\d.]+)\.pth$", path)
        return float(m.group(1)) if m else float("-inf")

    # Find the best checkpoint across all policies (PBT has checkpoint_p0..pN).
    best_files = _glob.glob(f"{exp_dir}/checkpoint_p*/best_0*.pth")
    if best_files:
        ppo_ckpt: str | None = max(best_files, key=_reward_from_name)
    else:
        regular_files = sorted(_glob.glob(f"{exp_dir}/checkpoint_p*/checkpoint_*.pth"))
        ppo_ckpt = regular_files[-1] if regular_files else None

    if ppo_ckpt:
        # Write converted model to {profile_root}/best/best_model.pth
        best_dir = Path(output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        bc_ckpt_path = best_dir / "best_model.pth"
        try:
            bc_policy = sf_to_qnn(
                sf_checkpoint_path=ppo_ckpt,
                obs_dim=ppo_cfg.get("obs_dim", 0) or _detect_obs_dim(ppo_cfg),
                trunk_hidden=int(ppo_cfg.get("trunk_hidden", 128)),
                gru_hidden=int(ppo_cfg.get("gru_hidden", 128)),
                use_gru=bool(ppo_cfg.get("use_gru", True)),
                device="cpu",
            )
            bc_policy.save(bc_ckpt_path)
            ppo_result["best_model_path"] = str(bc_ckpt_path)
            ppo_result["ppo_checkpoint_path"] = ppo_ckpt
            ppo_result["sf_checkpoint_path"] = ppo_ckpt
        except Exception as exc:
            ppo_result["checkpoint_convert_error"] = str(exc)

    return ppo_result


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
    output_root: str | None = None,
    fresh_output_root: bool = False,
) -> tuple[LiveProfile, dict[str, Any], RuntimePlan, Path, Path, Path, dict, dict, dict]:
    """Common setup for both BC and PPO phases."""
    profile, runtime, plan = build_runtime_plan(profile_name, device)
    profile = _profile_with_output_root(profile, output_root)
    resolved_asset_root = _resolve_asset_root(asset_root)
    resolved_demo_dir = _resolve_demo_dir(profile, demo_dir)
    worker_path = Path(worker_binary)
    needs_live_worker = action in {"check", "ppo", "eval", "eval-bc"} or (action == "bc" and eval_bc)
    if needs_live_worker or rebuild_worker:
        worker_path = _ensure_worker(worker_path, rebuild=rebuild_worker)
    ppo_cfg, eval_cfg = load_config_with_runtime(profile, plan, device)
    if needs_live_worker:
        _validate_native_mod_assets(resolved_asset_root, ppo_cfg.get("native_args") if isinstance(ppo_cfg.get("native_args"), list) else None)
        _validate_native_mod_assets(resolved_asset_root, eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None)
    if fresh_output_root:
        _assert_fresh_output_root(profile)
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

    return profile, runtime, plan, resolved_demo_dir, resolved_asset_root, worker_path, ppo_cfg, eval_cfg, results


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
    checkpoint_override: str | None = None,
    seed_checkpoint: str | None = None,
    seed_checkpoints: list[str] | None = None,
    output_root: str | None = None,
    fresh: bool = False,
    disable_pbt: bool = False,
    ppo_fixed_tick_hz: int | None = None,
    ppo_max_steps_per_episode: int | None = None,
    record_demos: bool = False,
) -> dict[str, Any]:
    profile = _profile_with_output_root(PROFILES[profile_name], output_root)

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

    profile, runtime, plan, resolved_demo_dir, resolved_asset_root, worker_path, ppo_cfg, eval_cfg, results = _setup_pipeline(
        profile_name, action, eval_bc, device, demo_dir, asset_root, worker_binary, rebuild_worker,
        output_root=output_root,
        fresh_output_root=bool(fresh and action == "ppo"),
    )
    if record_demos:
        eval_cfg["record_demos"] = True
    if action == "ppo" and disable_pbt:
        ppo_cfg["with_pbt"] = False
        ppo_cfg["num_policies"] = 1
        results["ppo_disable_pbt"] = True
    if action == "ppo" and ppo_fixed_tick_hz is not None:
        ppo_cfg["fixed_tick_hz"] = int(ppo_fixed_tick_hz)
        results["ppo_fixed_tick_hz_override"] = int(ppo_fixed_tick_hz)
    if action == "ppo" and ppo_max_steps_per_episode is not None:
        ppo_cfg["max_steps_per_episode"] = int(ppo_max_steps_per_episode)
        results["ppo_max_steps_per_episode_override"] = int(ppo_max_steps_per_episode)
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
    # BC phase: collect → BC → eval-bc (profile-independent)
    # ------------------------------------------------------------------
    if action == "bc":
        bc_cfg = build_bc_config(profile.runtime_scale, device)
        _seed_bc_collect_paths(bc_cfg)

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
                output_dir=BC_COLLECT_DIR,
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
            eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
            eval_cfg["native_executable"] = str(worker_path)
            eval_cfg["checkpoint_path"] = BC_CHECKPOINT
            eval_cfg["output_dir"] = str(_profile_output_root(profile) / "eval_bc")
            eval_cfg["checkpoint_path"] = _prepare_eval_checkpoint(profile, str(eval_cfg["checkpoint_path"]), str(eval_cfg["output_dir"]))
            results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
            stage_timings["eval"] = time.monotonic() - started

    # ------------------------------------------------------------------
    # PPO phase: PPO → eval
    # ------------------------------------------------------------------
    elif action == "ppo":
        # Multi-seed PBT: multiple seed checkpoints, one per policy (round-robin).
        if seed_checkpoints:
            init_ckpts: list[str] = []
            for ckpt in seed_checkpoints:
                if not Path(ckpt).exists():
                    raise FileNotFoundError(f"Seed checkpoint does not exist: {ckpt}")
                init_ckpts.append(ckpt)
            ppo_cfg["init_ckpts"] = init_ckpts
            results["ppo_init_ckpts"] = init_ckpts
            # First checkpoint is also recorded as init_ckpt for backward compat.
            ppo_cfg["init_ckpt"] = init_ckpts[0]
            results["ppo_init_ckpt"] = init_ckpts[0]
        elif seed_checkpoint:
            init_ckpt = seed_checkpoint
            if not Path(init_ckpt).exists():
                raise FileNotFoundError(f"Explicit seed checkpoint does not exist: {init_ckpt}")
            if init_ckpt and Path(init_ckpt).exists():
                ppo_cfg["init_ckpt"] = init_ckpt
                results["ppo_init_ckpt"] = init_ckpt
        else:
            # Prefer latest best SF checkpoint over BC for warm-start when resuming an existing root.
            import re as _re

            profile_root = _profile_output_root(profile)
            best_dir = profile_root / "best"
            sf_checkpoints = sorted(best_dir.glob("*.pth")) if best_dir.exists() else []
            if sf_checkpoints:
                def _reward_from_name(p: Path) -> float:
                    m = _re.search(r"reward_([-\d.]+)\.pth$", p.name)
                    return float(m.group(1)) if m else float("-inf")
                # Prefer best_* (have reward in name), fall back to latest checkpoint_*
                best_files = [f for f in sf_checkpoints if f.name.startswith("best_0")]
                if best_files:
                    init_ckpt = str(max(best_files, key=_reward_from_name))
                else:
                    # No best_* files; use the most recent .pth by mtime
                    init_ckpt = str(max(sf_checkpoints, key=lambda p: p.stat().st_mtime))
            else:
                init_ckpt = profile.bc_checkpoint
            if init_ckpt and Path(init_ckpt).exists():
                ppo_cfg["init_ckpt"] = init_ckpt
                results["ppo_init_ckpt"] = init_ckpt

        started = time.monotonic()
        ppo_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        ppo_cfg["native_executable"] = str(worker_path)
        results["ppo"] = _run_ppo(profile, ppo_cfg, resolved_asset_root, worker_path, device)
        stage_timings["ppo"] = time.monotonic() - started

        started = time.monotonic()
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        eval_cfg["checkpoint_path"] = _prepare_eval_checkpoint(profile, str(eval_cfg["checkpoint_path"]), str(eval_cfg["output_dir"]))
        results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
        stage_timings["eval"] = time.monotonic() - started

    # ------------------------------------------------------------------
    # Individual stage actions for debugging.
    # ------------------------------------------------------------------
    elif action in ("eval", "eval-bc"):
        started = time.monotonic()
        eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(resolved_asset_root)}
        eval_cfg["native_executable"] = str(worker_path)
        if checkpoint_override:
            eval_cfg["checkpoint_path"] = checkpoint_override
        elif action == "eval-bc":
            eval_cfg["checkpoint_path"] = BC_CHECKPOINT
            eval_cfg["output_dir"] = str(_profile_output_root(profile) / "eval_bc")
        eval_cfg["checkpoint_path"] = _prepare_eval_checkpoint(profile, str(eval_cfg["checkpoint_path"]), str(eval_cfg["output_dir"]))
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
        choices=["plan", "report", "check", "bc", "eval-bc", "ppo", "eval"],
        default="bc",
        help="Phase to run: 'bc' (collect+BC), 'ppo' (PPO+eval), 'eval' (standalone evaluation)",
    )
    parser.add_argument("--eval-bc", action="store_true", help="When used with --action bc, run eval after behavior cloning")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path for eval")
    parser.add_argument("--seed-checkpoint", default=None, help="Explicit SF or BC warm-start checkpoint for PPO")
    parser.add_argument("--seed-checkpoints", default=None, help="Comma-separated seed checkpoints for multi-seed PBT")
    parser.add_argument("--output-root", default=None, help="Override retained output root for this run")
    parser.add_argument("--fresh", action="store_true", help="Require an empty output root for PPO instead of resuming")
    parser.add_argument("--no-pbt", action="store_true", help="Disable PBT for this PPO run")
    parser.add_argument("--ppo-fixed-tick-hz", type=int, default=None, help="Override PPO fixed tick rate for this run")
    parser.add_argument(
        "--ppo-max-steps-per-episode",
        type=int,
        default=None,
        help="Override PPO max steps per episode for this run",
    )
    parser.add_argument("--device", default="gpu", help="Requested torch device override")
    parser.add_argument("--demo-dir", default=None, help="Override demo directory")
    parser.add_argument("--asset-root", default=None, help="Override Quake asset root")
    parser.add_argument("--worker-binary", default="assets/bin/quake_worker", help="Path to the live worker binary")
    parser.add_argument("--rebuild-worker", action="store_true", help="Force a rebuild of the live worker binary")
    parser.add_argument("--record-demos", action="store_true", help="Record .dem files during evaluation")
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
        checkpoint_override=args.checkpoint,
        seed_checkpoint=args.seed_checkpoint,
        seed_checkpoints=[p.strip() for p in args.seed_checkpoints.split(",") if p.strip()] if args.seed_checkpoints else None,
        output_root=args.output_root,
        fresh=args.fresh,
        disable_pbt=args.no_pbt,
        ppo_fixed_tick_hz=args.ppo_fixed_tick_hz,
        ppo_max_steps_per_episode=args.ppo_max_steps_per_episode,
        record_demos=args.record_demos,
    )


if __name__ == "__main__":
    main()
