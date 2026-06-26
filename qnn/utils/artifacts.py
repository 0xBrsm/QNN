"""Run identity and checkpoint artifact naming.

One place defines how runs are identified and how checkpoint files are
named, so nothing else in the tree keys off directory names.

Identity
    ``run_id`` = ``YYYYMMDD-xxxxxx`` — creation date plus a 6-char random
    suffix (lowercase Crockford base32: no i/l/o/u). Sortable by creation,
    collision-safe across same-day launches, short enough to grep and to
    embed in filenames. Generated once at run init, stored in ``run.json``,
    immutable thereafter. Directory names remain human labels only.

Checkpoint names (BC and PPO)
    ``ckpt_e{epoch:03d}_{run_id}.pt``  rolling resume checkpoint (model +
        optimizer state). Exactly one exists per run while training; each
        epoch's atomic write supersedes and removes the previous epoch's
        file. The epoch in the name is provenance, not an archive series.
    ``best_{run_id}.pth``              best model (weights + meta sidecar).

Legacy names (``bc_training_checkpoint.pt`` / ``bc_best_model.pth``) are
still discovered by the ``find_*`` helpers so pre-rename run dirs keep
loading; new runs never write them.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Crockford base32 lowercase: unambiguous in filenames and logs.
_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
_RUN_ID_RE = re.compile(r"^\d{8}-[0-9abcdefghjkmnpqrstvwxyz]{6}$")

_CKPT_RE = re.compile(r"^ckpt_e(\d+)_(.+)\.pt$")

LEGACY_RESUME_NAME = "bc_training_checkpoint.pt"
LEGACY_BEST_NAME = "bc_best_model.pth"


def new_run_id(created: datetime | None = None) -> str:
    """Mint a run id. ``created`` defaults to now (UTC); pass the run's
    original creation time when backfilling."""
    stamp = (created or datetime.now(timezone.utc)).strftime("%Y%m%d")
    suffix = "".join(secrets.choice(_ID_ALPHABET) for _ in range(6))
    return f"{stamp}-{suffix}"


def is_run_id(value: object) -> bool:
    return isinstance(value, str) and bool(_RUN_ID_RE.match(value))


def ckpt_name(epoch: int, run_id: str) -> str:
    return f"ckpt_e{epoch:03d}_{run_id}.pt"


def best_name(run_id: str) -> str:
    return f"best_{run_id}.pth"


def parse_ckpt_name(name: str) -> tuple[int, str] | None:
    """Return ``(epoch, run_id)`` for a ``ckpt_e*`` filename, else None."""
    m = _CKPT_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def find_resume_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Newest resume checkpoint in *checkpoints_dir*: highest-epoch
    ``ckpt_e*`` file, falling back to the legacy fixed name."""
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.is_dir():
        for p in checkpoints_dir.glob("ckpt_e*.pt"):
            parsed = parse_ckpt_name(p.name)
            if parsed is not None:
                candidates.append((parsed[0], p))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    legacy = checkpoints_dir / LEGACY_RESUME_NAME
    return legacy if legacy.exists() else None


def find_best_model(checkpoints_dir: Path) -> Path | None:
    """The run's best model: ``best_<run_id>.pth``, falling back to the
    legacy fixed name."""
    if checkpoints_dir.is_dir():
        matches = sorted(checkpoints_dir.glob("best_*.pth"))
        # Exclude SF-style archived names (best_000123..._reward_*.pth):
        # ours have exactly one underscore-separated id token.
        ours = [p for p in matches if is_run_id(p.stem[len("best_"):])]
        if ours:
            return ours[-1]
    legacy = checkpoints_dir / LEGACY_BEST_NAME
    return legacy if legacy.exists() else None


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Durable atomic checkpoint write: tmp sibling + fsync + rename.

    A crash at any instant leaves either the complete previous file or the
    complete new one on disk — never a torn write. This is what makes a
    single rolling resume checkpoint safe without an epoch-stamp archive.
    """
    import torch

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
