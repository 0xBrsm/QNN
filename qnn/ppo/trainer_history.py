"""Trainer-config history ledger for the autonomous training loop."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qnn.utils.io import read_json

TRAINER_LEDGER_DEFAULT_PATH = Path("runs/trainer_history.jsonl")
TRAINER_CONFIG_DEFAULT_PATH = Path(__file__).resolve().parent / "templates" / "train.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def trainer_config_hash(trainer_config: dict[str, Any]) -> str:
    """Deterministic short hash of a trainer config dict."""
    blob = json.dumps(trainer_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_trainer_config(path: Path | None = None) -> dict[str, Any]:
    """Load the current trainer config JSON."""
    return dict(read_json(path or TRAINER_CONFIG_DEFAULT_PATH))


def save_trainer_config(trainer_config: dict[str, Any], path: Path | None = None) -> None:
    """Write updated trainer config back to the JSON file."""
    target = path or TRAINER_CONFIG_DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(trainer_config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all history entries from the JSONL ledger."""
    target = path or TRAINER_LEDGER_DEFAULT_PATH
    if not target.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def record_launch(
    *,
    trainer_config: dict[str, Any],
    run_root: str,
    checkpoint_path: str,
    notes: str = "",
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Append a launch entry when a new run starts. Returns the entry."""
    entry = {
        "timestamp": _utc_now(),
        "event": "launch",
        "trainer_config_hash": trainer_config_hash(trainer_config),
        "trainer_config": trainer_config,
        "run_root": run_root,
        "checkpoint_path": checkpoint_path,
        "notes": notes,
    }
    _append(entry, ledger_path)
    return entry


def record_outcome(
    *,
    run_root: str,
    trainer_config_hash_value: str,
    status: str,
    diagnoses: list[str],
    metrics: dict[str, Any],
    recommendations: list[dict[str, Any]],
    notes: str = "",
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Append an outcome entry when a run is judged. Returns the entry."""
    entry = {
        "timestamp": _utc_now(),
        "event": "outcome",
        "trainer_config_hash": trainer_config_hash_value,
        "run_root": run_root,
        "status": status,
        "diagnoses": diagnoses,
        "metrics": metrics,
        "recommendations": recommendations,
        "notes": notes,
    }
    _append(entry, ledger_path)
    return entry


def find_prior_attempts(
    trainer_config: dict[str, Any],
    ledger_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all outcome entries that used the same trainer config hash."""
    target_hash = trainer_config_hash(trainer_config)
    return [
        entry
        for entry in load_ledger(ledger_path)
        if entry.get("event") == "outcome" and entry.get("trainer_config_hash") == target_hash
    ]


def find_failed_hashes(ledger_path: Path | None = None) -> set[str]:
    """Return trainer config hashes whose most recent outcome was not 'promising'."""
    outcomes: dict[str, str] = {}
    for entry in load_ledger(ledger_path):
        if entry.get("event") == "outcome":
            outcomes[entry["trainer_config_hash"]] = entry.get("status", "")
    return {h for h, status in outcomes.items() if status != "promising"}


def _append(entry: dict[str, Any], ledger_path: Path | None) -> None:
    target = ledger_path or TRAINER_LEDGER_DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
