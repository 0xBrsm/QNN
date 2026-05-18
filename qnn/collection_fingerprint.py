"""Deterministic identity fingerprint for a collected BC dataset.

A collection's identity is the composite sha256 over every input that
influences what's on disk:

  - filter      — sha256 of the filter.json (user intent)
  - manifest    — sha256 of the input manifest.ndjson (data available)
  - done_log    — sha256 of canonical sorted done.log lines (demos that survived BSP/error)
  - worker      — sha256 of the demo worker binary (extraction code)
  - code        — git commit of qnn.bc.collect at collect time
  - training_view — sha256 of canonical episode index seen by trainer
    (split + demo_idx + episode_idx + n_samples), independent of shard layout

Any byte-level change to any layer flips the composite, which lets
training runs link deterministically to a specific collection and
fail loud when the dataset they think they're training on doesn't
match what's on disk.

Reconstruction workflow.  The filter.json and fingerprint.json for
every active collection are tracked in git (the corpus manifest at
``artifacts/corpus/*_manifest.ndjson`` is also tracked).  To
reconstruct a historical collection exactly:

  1. ``git checkout <commit>``   — fetch that day's source, manifest,
     filter.json, and fingerprint.json
  2. Rebuild the worker binary from src/ (deterministic per machine)
  3. ``python -m qnn.bc.collect --filter-config artifacts/collect/<dir>/filter.json …``
  4. Verify the new fingerprint.json matches the committed one
     (component-by-component, or just the composite)

If any component differs the reconstruction is not exact and the
divergent component points at what changed (likely worker binary on a
different host — fixable with a Docker-based reproducible build).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FingerprintMismatch(RuntimeError):
    """Raised when a run's recorded fingerprint disagrees with the
    data dir's current fingerprint.json. """


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_sorted_lines(path: Path) -> str:
    """Hash text file contents after canonical line sorting.

    Used for append-order-sensitive files such as done.log where write order is
    intentionally non-deterministic (parallel collect completion) but line set
    identity is what matters.
    """
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    canonical = ("\n".join(sorted(lines)) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def _training_view_signature(data_dir: Path) -> str | None:
    """Hash the canonical per-episode index consumed by training.

    Canonical episode rows are tuples:
      (split, demo_idx, episode_idx, n_samples)
    sorted by (split, demo_idx, episode_idx, n_samples).  This is stable across
    recollects even when shard packing order differs.
    """

    def _rows_for_split(split: str) -> list[tuple[str, int, int, int]]:
        split_dir = data_dir / split
        manifest_path = split_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or manifest.get("format") != "sharded_v1":
            return []

        rows: list[tuple[str, int, int, int]] = []
        fallback_idx = 0
        for shard in manifest.get("shards", []):
            episode_lengths = [int(x) for x in shard.get("episode_lengths", [])]
            demo_idxs = shard.get("demo_idxs")
            if demo_idxs is None or len(demo_idxs) != len(episode_lengths):
                demo_idxs = list(range(fallback_idx, fallback_idx + len(episode_lengths)))
            fallback_idx += len(episode_lengths)
            episode_idxs = shard.get("episode_idxs")
            if episode_idxs is None or len(episode_idxs) != len(episode_lengths):
                episode_idxs = [0] * len(episode_lengths)
            for n_samples, demo_idx, episode_idx in zip(episode_lengths, demo_idxs, episode_idxs):
                rows.append((split, int(demo_idx), int(episode_idx), int(n_samples)))
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
        return rows

    all_rows = _rows_for_split("precomputed_train") + _rows_for_split("precomputed_val")
    if not all_rows:
        return None
    canonical = json.dumps(all_rows, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _git_commit(cwd: Path | None = None) -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or None
    except Exception:
        return None


def compute(
    *,
    filter_path: Path | None,
    manifest_path: Path,
    done_log_path: Path,
    worker_binary_path: Path,
    data_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Build the fingerprint dict for a finalized collection.

    Missing optional inputs (no filter.json, no done.log because the
    run was a one-shot, etc.) record as None so the fingerprint stays
    well-formed.  Callers are responsible for ensuring required inputs
    (manifest, worker binary) exist before calling. """
    components: dict[str, str | None] = {}
    components["filter"] = (
        "sha256:" + _sha256_file(filter_path)
        if filter_path is not None and filter_path.exists() else None
    )
    components["manifest"] = (
        "sha256:" + _sha256_file(manifest_path)
        if manifest_path.exists() else None
    )
    components["done_log"] = (
        "sha256:" + _sha256_sorted_lines(done_log_path)
        if done_log_path.exists() else None
    )
    components["worker"] = (
        "sha256:" + _sha256_file(worker_binary_path)
        if worker_binary_path.exists() else None
    )
    training_view = _training_view_signature(data_dir) if data_dir is not None else None
    components["training_view"] = f"sha256:{training_view}" if training_view else None
    git = _git_commit(repo_root)
    components["code"] = f"git:{git}" if git else None

    canonical = json.dumps(components, sort_keys=True).encode()
    composite = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return {
        "fingerprint": composite,
        "components": components,
        "created": datetime.now(timezone.utc).isoformat(),
    }


def load(data_dir: Path) -> dict | None:
    """Read fingerprint.json from a collection dir; return None if absent."""
    fp = data_dir / "fingerprint.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def write(fingerprint: dict, data_dir: Path) -> None:
    """Persist the fingerprint dict to data_dir/fingerprint.json."""
    (data_dir / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2)
    )


_ALLOW_ENV = "QNN_ALLOW_FINGERPRINT_MISMATCH"


def verify(*, expected_fingerprint: str | None, data_dir: Path) -> dict | None:
    """Compare `expected_fingerprint` (from a run's recorded identity)
    against the data dir's current fingerprint.

    Returns the current fingerprint dict (or None if data_dir has no
    fingerprint.json).  Raises ``FingerprintMismatch`` when expected and
    actual disagree, unless ``QNN_ALLOW_FINGERPRINT_MISMATCH=1`` is set
    (in which case a warning prints and execution continues). """
    actual = load(data_dir)
    if expected_fingerprint is None:
        return actual
    if actual is None:
        msg = (
            f"run config expects collection_fingerprint={expected_fingerprint!r} "
            f"but {data_dir}/fingerprint.json is absent"
        )
        if os.environ.get(_ALLOW_ENV):
            print(f"  [warn] {msg} (override via {_ALLOW_ENV})")
            return None
        raise FingerprintMismatch(msg)
    if actual["fingerprint"] != expected_fingerprint:
        diff_lines: list[str] = []
        actual_components = actual.get("components") or {}
        # Component-level diff is best-effort: caller may not have stored
        # the expected components, only the composite.  We diff what we have.
        if isinstance(actual_components, dict):
            for k, v in actual_components.items():
                diff_lines.append(f"    {k}: {v}")
        msg = (
            f"collection fingerprint mismatch:\n"
            f"  expected: {expected_fingerprint}\n"
            f"  actual:   {actual['fingerprint']}\n"
        )
        if diff_lines:
            msg += "  actual components:\n" + "\n".join(diff_lines)
        msg += f"\n  (set {_ALLOW_ENV}=1 to override)"
        if os.environ.get(_ALLOW_ENV):
            print(f"  [warn] {msg}")
            return actual
        raise FingerprintMismatch(msg)
    return actual
