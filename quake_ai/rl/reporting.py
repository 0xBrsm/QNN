"""Run reporting helpers for the run-dir training workflow."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quake_ai.rl.metrics import EVAL_REPORT_METRICS, PPO_REPORT_METRICS, report_metric_key
from quake_ai.utils.io import safe_read_json as _safe_read_json
from quake_ai.utils.io import write_json

REPORT_STAGES = ("collect", "bc", "ppo", "best", "eval", "eval_bc")


def _assert_supported_layout(run_root: Path) -> None:
    legacy_dir = run_root / "checkpoints" / "ppo"
    if legacy_dir.exists():
        raise RuntimeError(
            f"Legacy PPO checkpoint layout is unsupported: {legacy_dir}. "
            "PPO artifacts now live directly under run.json.output.checkpoints."
        )


def _stage_dir(run_root: Path, stage: str) -> Path:
    if stage in {"bc", "ppo"}:
        return run_root / "checkpoints"
    if stage in {"collect", "best"}:
        return run_root / "checkpoints" / stage
    if stage in {"eval", "eval_bc"}:
        return run_root / "metrics" / stage
    raise RuntimeError(f"Unsupported report stage: {stage}")


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


def _first_device_summary(runtime: Mapping[str, Any]) -> dict[str, Any]:
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
    return str(device.get("name", runtime.get("resolved_device", "unknown")))


def _stage_report(run_root: Path, stage: str) -> dict[str, Any]:
    stage_dir = _stage_dir(run_root, stage)
    files = _collect_existing_files(stage_dir)

    report: dict[str, Any] = {
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


def _first_metric(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _eval_metric(report: Mapping[str, Any], mode: str, key: str) -> float | None:
    return _first_metric(
        _metric(report, "summary", "modes", mode, key),
        _metric(report, "manifest", "metrics", "modes", mode, key),
        _metric(report, "model_card", "evaluation", "modes", mode, key),
    )


def _stage_summary_metric(report: Mapping[str, Any], stage: str, key: str) -> float | None:
    resolved_key = report_metric_key(stage, key)
    if stage.startswith("eval"):
        return _first_metric(
            _eval_metric(report, "greedy", resolved_key),
            _metric(report, "summary", resolved_key),
            _metric(report, "manifest", "metrics", resolved_key),
        )
    return _first_metric(
        _metric(report, "summary", resolved_key),
        _metric(report, "manifest", "metrics", resolved_key),
    )


def _format_operational_note(note: Mapping[str, Any]) -> str:
    lines = [
        f"generated_at_utc: {note.get('generated_at_utc', '')}",
        f"run_root: {note.get('run_root', '')}",
        f"action: {note.get('action', '')}",
        f"runtime_scale: {note.get('runtime_scale', '')}",
        f"device: {note.get('device', '')}",
        f"elapsed_seconds: {note.get('elapsed_seconds', '')}",
        "",
        "summary:",
        f"  {note.get('summary', '')}",
    ]
    metrics = note.get("metrics", {})
    if isinstance(metrics, Mapping) and metrics:
        lines.append("")
        lines.append("metrics:")
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def write_run_report(
    *,
    run_root: str | Path,
    action: str,
    runtime_scale: str,
    runtime: Mapping[str, Any],
    plan: Mapping[str, Any],
    results: Mapping[str, Any],
    stage_timings: Mapping[str, float],
) -> dict[str, Any]:
    root = Path(run_root)
    _assert_supported_layout(root)
    stage_reports = {stage: _stage_report(root, stage) for stage in REPORT_STAGES}

    all_files: list[Path] = []
    for stage in REPORT_STAGES:
        all_files.extend(_collect_existing_files(_stage_dir(root, stage)))

    elapsed_source = "measured" if stage_timings else "artifact_mtime_window"
    elapsed_seconds = sum(stage_timings.values()) if stage_timings else _mtime_window_seconds(all_files)

    metrics: dict[str, float] = {}
    for stage, keys in (("ppo", PPO_REPORT_METRICS), ("eval", EVAL_REPORT_METRICS)):
        for key in keys:
            value = _stage_summary_metric(stage_reports.get(stage, {}), stage, key)
            if value is not None:
                metrics[f"{stage}_{key}"] = value

    note = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "action": action,
        "runtime_scale": runtime_scale,
        "device": _device_label(runtime),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": elapsed_source,
        "summary": "Run report generated from retained artifacts.",
        "metrics": metrics,
        "plan": dict(plan),
    }

    report = {
        "generated_at_utc": note["generated_at_utc"],
        "run_root": str(root),
        "action": action,
        "runtime_scale": runtime_scale,
        "runtime": dict(runtime),
        "plan": dict(plan),
        "results": dict(results),
        "stage_reports": stage_reports,
        "metrics": metrics,
        "operational_note": note,
    }

    report_path = root / "live_run_report.json"
    note_json_path = root / "operational_note.json"
    note_md_path = root / "operational_note.md"
    write_json(report_path, report)
    write_json(note_json_path, note)
    note_md_path.write_text(_format_operational_note(note), encoding="utf-8")
    return {
        "report": report,
        "report_path": str(report_path),
        "operational_note_json_path": str(note_json_path),
        "operational_note_md_path": str(note_md_path),
    }
