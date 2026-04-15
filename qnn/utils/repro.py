"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from qnn.utils.io import sha256_dict, write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    torch = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def git_sha(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def write_experiment_manifest(path: str | Path, config: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "git_sha": git_sha(Path(path).parent),
        "config_hash": sha256_dict(config),
        "config": dict(config),
        "metrics": dict(metrics),
    }
    write_json(path, payload)
