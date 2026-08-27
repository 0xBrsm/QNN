"""Deterministic identity fingerprint for a collected BC dataset.

A collection's identity is the composite sha256 over every input that
influences what's on disk:

  - filter      — sha256 of the filter.json (user intent)
  - manifest    — sha256 of the input manifest.ndjson (data available)
  - done_log    — sha256 of canonical sorted done.log lines (demos that survived BSP/error)
  - worker      — sha256 of the demo worker binary (extraction code)
  - code        — git commit of qnn.bc.collect at collect time, with a
    ``-dirty`` suffix when tracked files differ from HEAD (schema v2+)
  - training_view — sha256 of canonical episode index seen by trainer
    (split + demo_idx + episode_idx + n_samples), independent of shard layout
  - perception  — the RESOLVED qnn_los_clearance/qnn_fov regime, read back
    from the demo worker's own hello response rather than guessed from the
    launcher's environment (schema v2+; see SCHEMA VERSIONING below and
    ``agents/plans/a26-superiority-decomposition.md`` E6 — an unrecorded
    perception regime cost a month of confusion when two corpora collected
    under different LOS-veto settings were compared as if identical)

Any byte-level change to any layer flips the composite, which lets
training runs link deterministically to a specific collection and
fail loud when the dataset they think they're training on doesn't
match what's on disk.

SCHEMA VERSIONING.  ``compute()`` always produces the current schema
(``FINGERPRINT_SCHEMA_VERSION``, top-level ``"schema"`` key). Fingerprints
written before the "perception" component and the dirty-tree flag existed
have no "schema" key at all — :func:`schema_of` reads that as schema 1.
Old fingerprint.json files on disk are NEVER rewritten retroactively by
this module, so a corpus that hasn't been recollected keeps comparing
identically (composite hash unchanged) forever. Adding fields to
``compute()``'s components necessarily changes the composite hash for any
*freshly computed* fingerprint, so a straight composite-string comparison
between an old (schema 1) and new (schema 2) fingerprint of the very same
corpus content would misreport "different corpus". :func:`compare` exists
for exactly this: it detects a schema mismatch and falls back to comparing
only the components both schemas share (the original six above), so a
schema bump alone never silently reads as drift.

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


class PerceptionRegimeMismatch(RuntimeError):
    """Raised when collect workers report disagreeing resolved
    perception regimes (qnn_los_clearance / qnn_fov) within one collect.
    Env is fixed for the lifetime of a collect's worker processes, so
    disagreement means a worker defect, not a legitimate difference —
    see agents/plans/a26-superiority-decomposition.md E6. """


# Fingerprints written before "perception" and the dirty-tree flag existed
# carry no "schema" key at all. schema_of() reads that absence as 1. Bump
# this only when compute()'s component set changes again.
FINGERPRINT_SCHEMA_VERSION = 2

# Components present in every schema-1 fingerprint (i.e. every fingerprint
# ever written before this module gained a "schema" key). compare() falls
# back to these when comparing across a schema boundary, since the
# composite hash of two different schemas is never meaningfully comparable
# (different inputs feed it) even when the underlying corpus is identical.
V1_COMPONENT_KEYS: tuple[str, ...] = (
    "filter", "manifest", "done_log", "worker", "code", "training_view",
)


def schema_of(fingerprint: dict) -> int:
    """Return a fingerprint dict's schema version. Absent = 1 (every
    fingerprint written before schema versioning existed)."""
    return int(fingerprint.get("schema", 1))


def _format_perception(regime: tuple[int, float] | None) -> str:
    """Canonical string form of a resolved perception regime for the
    fingerprint's ``perception`` component.

    ``None`` means no collect worker reported a regime at all — either
    every worker predates the hello ``"perception"`` block (pre-E6
    binary) or the collect never got as far as a worker hello. Recorded
    as the literal ``"unreported"``, never guessed from the launcher's
    ``QNN_LOS_CLEARANCE``/``qnn_fov`` env — a guess is exactly the
    ambient-env failure mode this fix closes. """
    if regime is None:
        return "unreported"
    los_clearance, fov = regime
    return f"los_clearance={int(los_clearance)},fov={float(fov):g}"


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


def _git_dirty(cwd: Path | None = None) -> bool:
    """True iff any TRACKED file differs from HEAD (staged or unstaged).

    Untracked files (e.g. the fingerprint.json this call is about to
    write, or scratch output under the same tree) deliberately do NOT
    count — ``--untracked-files=no`` — so writing this file's own
    output never flips its own dirty flag. """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _git_commit(cwd: Path | None = None) -> str | None:
    """Git identity of the code that ran this collect: HEAD sha, plus a
    ``-dirty`` suffix when tracked files differ from HEAD. Bare
    ``git rev-parse HEAD`` alone can't distinguish "ran exactly this
    commit" from "ran this commit plus uncommitted changes" — the
    dirty suffix closes that gap (schema v2+). """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
    if not sha:
        return None
    return f"{sha}-dirty" if _git_dirty(cwd) else sha


def compute(
    *,
    filter_path: Path | None,
    manifest_path: Path,
    done_log_path: Path,
    worker_binary_path: Path,
    data_dir: Path | None = None,
    repo_root: Path | None = None,
    perception_regime: tuple[int, float] | None = None,
) -> dict:
    """Build the fingerprint dict for a finalized collection.

    Missing optional inputs (no filter.json, no done.log because the
    run was a one-shot, etc.) record as None so the fingerprint stays
    well-formed.  Callers are responsible for ensuring required inputs
    (manifest, worker binary) exist before calling.

    ``perception_regime``: the resolved ``(los_clearance, fov)`` the
    collect worker(s) reported (see ``qnn.bc.collect``'s hello-response
    parsing).  ``None`` means no worker reported one (old binary predating
    the E6 fix, or the collect has no worker output to read) — recorded
    as ``"unreported"``, never guessed from the launcher's env. Always
    produces the current schema (``FINGERPRINT_SCHEMA_VERSION``); older
    fingerprints on disk are schema 1 and are never rewritten by this
    function — see the module docstring's SCHEMA VERSIONING section and
    :func:`compare`. """
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
    components["perception"] = _format_perception(perception_regime)

    canonical = json.dumps(components, sort_keys=True).encode()
    composite = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return {
        "fingerprint": composite,
        "components": components,
        "created": datetime.now(timezone.utc).isoformat(),
        "schema": FINGERPRINT_SCHEMA_VERSION,
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


def compare(a: dict, b: dict) -> dict:
    """Schema-aware comparison of two fingerprint dicts.

    Plain ``a["fingerprint"] == b["fingerprint"]`` composite-hash equality
    is only meaningful when both fingerprints were computed under the
    same schema — the composite is a hash of the *components dict*, and a
    schema bump changes which components feed it, so two fingerprints of
    the identical underlying corpus captured before/after a schema bump
    have different composites even though nothing about the corpus
    changed. Comparing those composites directly would misreport
    "different corpus" (exactly the E6 confusion this module exists to
    prevent).

    Returns a dict:
      ``equivalent``          — the real verdict callers should act on:
                                 True iff there's no evidence the corpus
                                 differs.
      ``schema_mismatch``     — True iff `a` and `b` have different schemas.
      ``schema_a``, ``schema_b`` — each fingerprint's schema version.
      ``differing_components`` — sorted component keys whose values differ
                                 (compared over both components' union when
                                 schemas match, over :data:`V1_COMPONENT_KEYS`
                                 present in both when they don't).

    Same schema: ``equivalent`` is exact composite equality (fast path,
    identical to the historical behavior), with a component-level diff
    for diagnostics.  Different schema: falls back to comparing only the
    v1-common components — a schema-only difference (no common component
    changed) is ``equivalent`` True with ``schema_mismatch`` True, so
    callers can surface "the fingerprint format changed" separately from
    "the corpus changed" instead of conflating them. """
    schema_a, schema_b = schema_of(a), schema_of(b)
    comps_a = a.get("components") or {}
    comps_b = b.get("components") or {}

    if schema_a == schema_b:
        equal = a.get("fingerprint") == b.get("fingerprint")
        differing = sorted(
            k for k in set(comps_a) | set(comps_b)
            if comps_a.get(k) != comps_b.get(k)
        )
        return {
            "equivalent": equal,
            "schema_mismatch": False,
            "schema_a": schema_a,
            "schema_b": schema_b,
            "differing_components": differing,
        }

    common_keys = [k for k in V1_COMPONENT_KEYS if k in comps_a and k in comps_b]
    differing = sorted(k for k in common_keys if comps_a.get(k) != comps_b.get(k))
    return {
        "equivalent": len(differing) == 0,
        "schema_mismatch": True,
        "schema_a": schema_a,
        "schema_b": schema_b,
        "differing_components": differing,
        "compared_components": common_keys,
    }


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
