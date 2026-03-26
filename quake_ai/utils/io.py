"""Shared I/O helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping


def read_ndjson(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_ndjson(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def safe_read_json(path: str | Path) -> Dict[str, Any] | None:
    """Read JSON if the file exists, otherwise return None."""
    p = Path(path)
    if not p.exists():
        return None
    return read_json(p)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load config from JSON-compatible YAML file.

    V0 stores configs as JSON text with .yaml extension to keep runtime deps minimal.
    """
    return read_json(path)


def trusted_torch_load(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load a trusted local PyTorch checkpoint across torch 2.5/2.6+ defaults."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256_dict(data: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(data), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
