"""Run reporting, operational notes, and metric comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from quake_ai.rl.planning import (
    GIB,
    _profile_output_root,
    _safe_read_json,
    _stage_output_dir,
)
from quake_ai.rl.profiles import PROFILES, LiveProfile
from quake_ai.utils.io import write_json

REPORT_STAGES = ("bc", "eval_bc", "sf", "best", "eval")


def _collect_existing_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(candidate for candidate in path.iterdir() if candidate.is_file())


def _mtime_window_seconds(paths: list[Path]) -> float | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    mtimes = [path.stat().st_mtime for path in existing]
    return float(max(mtimes) - min(mtimes))


def _first_device_summary(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    devices = runtime.get("devices")
    if not isinstance(devices, list) or not devices:
        return {}
    first = devices[0]
    if not isinstance(first, Mapping):
        return {}
    return {str(key): value for key, value in first.items()}


def _device_label(runtime: Mapping[str, Any]) -> str:
    device = _first_device_summary(runtime)
    if not device:
        return str(runtime.get("resolved_device", "unknown"))
    name = str(device.get("name", runtime.get("resolved_device", "unknown")))
    total_memory = int(device.get("total_memory", 0))
    if total_memory <= 0:
        return name
    return f"{name} ({round(total_memory / GIB, 2)} GiB)"


def _load_existing_runtime_context(profile: LiveProfile) -> tuple[Dict[str, Any], Dict[str, Any], Path]:
    plan_path = Path(profile.plan_path)
    payload = _safe_read_json(plan_path) or {}
    runtime = payload.get("runtime", {})
    plan = payload.get("plan", {})
    return (
        dict(runtime) if isinstance(runtime, Mapping) else {},
        dict(plan) if isinstance(plan, Mapping) else {},
        plan_path,
    )


def _stage_report(profile: LiveProfile, stage: str) -> Dict[str, Any]:
    stage_dir = _stage_output_dir(profile, stage)
    files = _collect_existing_files(stage_dir)

    report: Dict[str, Any] = {
        "stage": stage,
        "output_dir": str(stage_dir),
        "status": "present" if files else "missing",
        "files": [str(path) for path in files],
    }

    if stage == "collect":
        manifest = _safe_read_json(stage_dir / "collect_manifest.json")
        if manifest is not None:
            report["manifest"] = manifest
        return report

    if stage.startswith("eval"):
        manifest_name = "eval_manifest.json"
        summary_name = "eval_summary.json"
    else:
        manifest_name = f"{stage}_manifest.json"
        summary_name = f"{stage}_summary.json"
    manifest = _safe_read_json(stage_dir / manifest_name)
    summary = _safe_read_json(stage_dir / summary_name)
    if manifest is not None:
        report["manifest"] = manifest
    if summary is not None:
        report["summary"] = summary
    if stage.startswith("eval"):
        model_card = _safe_read_json(stage_dir / "model_card.json")
        if model_card is not None:
            report["model_card"] = model_card
    return report


def _metric(report: Mapping[str, Any], *path: str) -> float | None:
    current: Any = report
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _eval_metric(report: Mapping[str, Any], mode: str, key: str) -> float | None:
    return (
        _metric(report, "summary", "modes", mode, key)
        or _metric(report, "manifest", "metrics", "modes", mode, key)
        or _metric(report, "model_card", "evaluation", "modes", mode, key)
    )


def _stage_summary_metric(report: Mapping[str, Any], stage: str, key: str) -> float | None:
    if stage.startswith("eval"):
        return _eval_metric(report, "greedy", key) or _metric(report, "summary", key) or _metric(report, "manifest", "metrics", key)
    return _metric(report, "summary", key) or _metric(report, "manifest", "metrics", key)


def _comparison_metric_payload(current: float | None, baseline: float | None) -> Dict[str, float] | None:
    if current is None or baseline is None:
        return None
    return {
        "current": float(current),
        "baseline": float(baseline),
        "delta": float(current - baseline),
    }


def _profile_comparison(current_stage_reports: Mapping[str, Mapping[str, Any]], baseline_profile: LiveProfile) -> Dict[str, Any]:
    baseline_reports = {stage: _stage_report(baseline_profile, stage) for stage in REPORT_STAGES}
    metrics: Dict[str, Dict[str, float]] = {}
    for stage, key in (
        ("ppo", "death_rate"),
        ("ppo", "frag_delta_mean"),
        ("ppo", "damage_dealt_mean"),
        ("ppo", "hit_count_mean"),
        ("ppo", "shots_fired_mean"),
        ("eval", "death_rate"),
        ("eval", "frag_delta_mean"),
        ("eval", "mean_episode_return"),
        ("eval", "damage_dealt_mean"),
        ("eval", "hit_count_mean"),
        ("eval", "shots_fired_mean"),
    ):
        payload = _comparison_metric_payload(
            _stage_summary_metric(current_stage_reports.get(stage, {}), stage, key),
            _stage_summary_metric(baseline_reports.get(stage, {}), stage, key),
        )
        if payload is not None:
            metrics[f"{stage}_{key}"] = payload
    return {
        "profile": baseline_profile.name,
        "profile_note": baseline_profile.profile_note,
        "stage_status": {stage: baseline_reports[stage]["status"] for stage in REPORT_STAGES},
        "metrics": metrics,
    }


def _intra_profile_baseline_comparison(stage_reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Dict[str, float]] = {}
    for key in ("death_rate", "frag_delta_mean", "mean_episode_return", "damage_dealt_mean", "hit_count_mean", "shots_fired_mean"):
        payload = _comparison_metric_payload(
            _stage_summary_metric(stage_reports.get("eval", {}), "eval", key),
            _stage_summary_metric(stage_reports.get("eval_bc", {}), "eval_bc", key),
        )
        if payload is not None:
            metrics[key] = payload
    return {
        "baseline": "eval_bc",
        "metrics": metrics,
    }


def _build_operational_note(
    *,
    profile: LiveProfile,
    action: str,
    runtime: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage_reports: Mapping[str, Mapping[str, Any]],
    stage_timings: Mapping[str, float],
) -> Dict[str, Any]:
    all_files: list[Path] = []
    for stage in REPORT_STAGES:
        all_files.extend(_collect_existing_files(_stage_output_dir(profile, stage)))

    elapsed_source = "measured" if stage_timings else "artifact_mtime_window"
    elapsed_seconds = sum(stage_timings.values()) if stage_timings else _mtime_window_seconds(all_files)

    worker_count = int(plan.get("num_envs", 0) or 0)
    rollout_steps = int(plan.get("rollout_steps", 0) or 0)
    bc_batch_size = int(plan.get("bc_batch_size", 0) or 0)
    eval_episodes = int(plan.get("eval_episodes", 0) or 0)

    utilization_parts = [f"backend={runtime.get('backend', 'unknown')}", f"device={_device_label(runtime)}"]
    if worker_count > 0:
        utilization_parts.append(f"ppo_workers={worker_count}")
    if rollout_steps > 0:
        utilization_parts.append(f"rollout_steps={rollout_steps}")
    if bc_batch_size > 0:
        utilization_parts.append(f"bc_batch_size={bc_batch_size}")
    if eval_episodes > 0:
        utilization_parts.append(f"eval_episodes={eval_episodes}")
    utilization_parts.append("direct accelerator utilization sampling not captured")

    sf_report = stage_reports.get("sf", {})
    best_report = stage_reports.get("best", {})
    eval_report = stage_reports.get("eval", {})
    eval_bc_report = stage_reports.get("eval_bc", {})
    bc_report = stage_reports.get("bc", {})
    baseline_compare = _intra_profile_baseline_comparison(stage_reports)

    instability_notes: list[str] = []
    if best_report.get("status") == "present":
        instability_notes.append("PPO completed and wrote a checkpoint without a recorded worker crash.")
    if _metric(eval_report, "manifest", "metrics", "modes", "greedy", "stuck_rate") and _metric(
        eval_report, "manifest", "metrics", "modes", "greedy", "stuck_rate"
    ) >= 0.9:
        instability_notes.append("Evaluation reported a high greedy stuck_rate; this looks like policy quality, not a worker crash.")
    bc_greedy_return = _metric(eval_bc_report, "manifest", "metrics", "modes", "greedy", "mean_episode_return")
    ppo_greedy_return = _metric(eval_report, "manifest", "metrics", "modes", "greedy", "mean_episode_return")
    if bc_greedy_return is not None and ppo_greedy_return is not None:
        delta = ppo_greedy_return - bc_greedy_return
        if delta > 0.0:
            instability_notes.append(f"PPO improved greedy return over the BC checkpoint by {delta:.3f}.")
        elif delta < 0.0:
            instability_notes.append(f"PPO regressed greedy return relative to the BC checkpoint by {abs(delta):.3f}.")
        else:
            instability_notes.append("PPO matched the BC checkpoint on greedy return in this retained run.")
    damage_delta = baseline_compare["metrics"].get("damage_dealt_mean")
    if isinstance(damage_delta, Mapping):
        instability_notes.append(
            f"Greedy evaluation damage_dealt_mean delta versus eval_bc: {float(damage_delta.get('delta', 0.0)):.3f}."
        )
    if not instability_notes:
        instability_notes.append("No instability was recorded in the retained artifacts.")

    note = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "profile_group": profile.profile_group,
        "profile_note": profile.profile_note,
        "scenario_id": profile.scenario_id,
        "retained_role": profile.retained_role,
        "action": action,
        "worker_count": worker_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": elapsed_source,
        "runtime_backend": str(runtime.get("backend", "unknown")),
        "device": _device_label(runtime),
        "rough_utilization": "; ".join(utilization_parts),
        "instability_observations": instability_notes,
        "determinism_drift": "This run did not record in-process drift metrics; explicit drift coverage lives in the asset-gated parity and churn tests.",
        "metrics": {
            "bc_test_accuracy": _metric(bc_report, "summary", "test_accuracy") or _metric(bc_report, "manifest", "metrics", "test_accuracy"),
            "eval_bc_greedy_return": _metric(eval_bc_report, "manifest", "metrics", "modes", "greedy", "mean_episode_return"),
            "eval_bc_greedy_stuck_rate": _metric(eval_bc_report, "manifest", "metrics", "modes", "greedy", "stuck_rate"),
            "eval_bc_sampled_return": _metric(eval_bc_report, "manifest", "metrics", "modes", "sampled", "mean_episode_return"),
            "eval_bc_sampled_stuck_rate": _metric(eval_bc_report, "manifest", "metrics", "modes", "sampled", "stuck_rate"),
            "ppo_steps_done": _metric(sf_report, "summary", "steps_done") or _metric(sf_report, "manifest", "metrics", "steps_done"),
            "ppo_death_rate": _metric(sf_report, "summary", "death_rate") or _metric(sf_report, "manifest", "metrics", "death_rate"),
            "ppo_frag_delta_mean": _metric(sf_report, "summary", "frag_delta_mean")
            or _metric(sf_report, "manifest", "metrics", "frag_delta_mean"),
            "ppo_episodes_completed": _metric(sf_report, "summary", "episodes_completed")
            or _metric(sf_report, "manifest", "metrics", "episodes_completed"),
            "eval_greedy_return": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "mean_episode_return"),
            "eval_greedy_stuck_rate": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "stuck_rate"),
            "eval_greedy_death_rate": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "death_rate"),
            "eval_greedy_frag_delta_mean": _metric(eval_report, "manifest", "metrics", "modes", "greedy", "frag_delta_mean"),
            "eval_sampled_return": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "mean_episode_return"),
            "eval_sampled_stuck_rate": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "stuck_rate"),
            "eval_sampled_death_rate": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "death_rate"),
            "eval_sampled_frag_delta_mean": _metric(eval_report, "manifest", "metrics", "modes", "sampled", "frag_delta_mean"),
        },
    }
    return note


def _format_operational_note(note: Mapping[str, Any]) -> str:
    elapsed_seconds = note.get("elapsed_seconds")
    elapsed_text = "unknown"
    if isinstance(elapsed_seconds, (int, float)):
        elapsed_text = f"{elapsed_seconds:.2f}s"

    lines = [
        "# Live Training Operational Note",
        "",
        f"- Profile: {note.get('profile', 'unknown')}",
        f"- Scenario: {note.get('scenario_id', 'n/a')}",
        f"- Note: {note.get('profile_note', '') or 'n/a'}",
        f"- Action: {note.get('action', 'unknown')}",
        f"- Worker count: {note.get('worker_count', 0)}",
        f"- Elapsed time: {elapsed_text} ({note.get('elapsed_source', 'unknown')})",
        f"- Runtime: {note.get('runtime_backend', 'unknown')} on {note.get('device', 'unknown')}",
        f"- Rough utilization: {note.get('rough_utilization', 'unknown')}",
        "- Stability observations:",
    ]
    for item in note.get("instability_observations", []):
        lines.append(f"  - {item}")
    lines.append(f"- Determinism drift: {note.get('determinism_drift', 'unknown')}")

    metrics = note.get("metrics", {})
    if isinstance(metrics, Mapping):
        rendered = [f"{key}={value}" for key, value in metrics.items() if value is not None]
        if rendered:
            lines.append(f"- Metrics: {', '.join(rendered)}")

    return "\n".join(lines) + "\n"


def _write_run_report(
    *,
    profile: LiveProfile,
    action: str,
    runtime: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    results: Mapping[str, Any],
    stage_timings: Mapping[str, float],
) -> Dict[str, Any]:
    output_root = _profile_output_root(profile)
    stage_reports = {stage: _stage_report(profile, stage) for stage in REPORT_STAGES}
    comparison_reports = {
        baseline_name: _profile_comparison(stage_reports, PROFILES[baseline_name])
        for baseline_name in profile.comparison_profiles
        if baseline_name in PROFILES
    }
    baseline_compare = _intra_profile_baseline_comparison(stage_reports)
    note = _build_operational_note(
        profile=profile,
        action=action,
        runtime=runtime,
        plan=plan,
        stage_reports=stage_reports,
        stage_timings=stage_timings,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "action": action,
        "plan_path": str(plan_path),
        "runtime": dict(runtime),
        "plan": dict(plan),
        "training_focus": "pvp_transformer_sf",
        "results": dict(results),
        "stage_timings_seconds": {str(key): float(value) for key, value in stage_timings.items()},
        "stages": stage_reports,
        "baseline_comparison": baseline_compare,
        "comparison_profiles": comparison_reports,
        "operational_note": note,
    }

    report_path = output_root / "live_run_report.json"
    note_json_path = output_root / "operational_note.json"
    note_md_path = output_root / "operational_note.md"
    write_json(report_path, report)
    write_json(note_json_path, note)
    note_md_path.write_text(_format_operational_note(note), encoding="utf-8")

    return {
        "report": report,
        "report_path": str(report_path),
        "operational_note_json_path": str(note_json_path),
        "operational_note_md_path": str(note_md_path),
    }
