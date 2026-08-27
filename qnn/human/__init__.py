"""Per-collect HUMAN baseline creators — one home for the logic that produces the
model-AGNOSTIC human distribution artifacts a collect yields, consumed by whatever
needs them (decode-fit, diag, analysis).

Domains (Brian's framing — look / move / attack / jump):
  look   — turn-delta GRID (qnn.human.look_grid), INTERCEPT skill ladder + frozen
           per-corpus placement anchors (qnn.human.intercept; skill-curves §16.3),
           ACQUISITION Fitts band (qnn.human.acquisition),
           engagement RANGE prior (qnn.human.aim_range)
  move   — move-axis + weapon WHEN-hazard release tables (qnn.human.move_hazard /
           qnn.human.weapon_hazard)
  attack — per-weapon OP-ATTACK rate + intervals (qnn.human.op_attack)
  jump   — no per-collect baseline artifact yet (qnn.human.jump; contextual, no rate
           calibration — feedback_jump_no_rate_calibration)

Every one of these depends ONLY on the raw collect corpus on disk (plus torch-free
primitives) — this package is the SELF-CONTAINED producer of every corpus-derived human
baseline, runnable against any collection at any time with no dependency on the model
graph, the training/collect pipeline, decode-fit, or a checkpoint. They are computed once
per collect and cached: compute once, adopt, never change retroactively.

Artifact sinks, all per-collect:
  * the four corpus-walk baselines (intercept / acquisition / range / op_attack) →
    standalone docs under ``<collect_dir>/human_baseline/<name>.json`` + a
    ``human_baseline`` provenance block (schema + git + per-file sha256) in
    ``collect_metadata.json`` (``_ARTIFACTS``);
  * the three cheap decode TABLES (look_grid / move_hazard / weapon_hazard) → top-level
    blocks in ``collect_metadata.json``, the keys ``run.init`` pins into a run's config
    (``_TABLES``).
The one exception is the human-band window bank (``qnn.human.band_bank``): also a
per-collect ``human_baseline/`` artifact, but built LAZILY on demand by the eval-time
scorer (``qnn.eval.humanlikeness.human_band``), so it is not in ``_ARTIFACTS`` /
``ensure_from_collect`` and not covered by the provenance block.

  from qnn.human import ensure_from_collect, ensure_collect_tables
  ensure_collect_tables(collect_dir)                # cheap: the 3 decode tables (collect calls this)
  paths = ensure_from_collect(collect_dir)          # all 7; compute-if-missing, cheap when cached

  PYTHONPATH=src python -m qnn.human artifacts/collect/qwd [--force]   # backfill CLI (all 7)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
BASELINE_SUBDIR = "human_baseline"
# v2: the intercept artifact carries the frozen per-corpus placement_anchors node
# (+ hygiene census) consumed by qnn.decode_fit.human_refs (skill-curves §16.3).
SCHEMA = "human_baseline_v2"
_META_KEY = "human_baseline"


@dataclass(frozen=True)
class _Artifact:
    key: str
    filename: str
    module: str            # `python -m <module>` create entry (qnn.human.*)
    split: str             # each creator's own default split (full-corpus behavior)
    extra_args: tuple[str, ...] = ()   # creator-specific required CLI args


# The corpus-derived, model-agnostic human reference set. Splits mirror each creator's
# own main() default (intercept/acq/range = both splits; op-attack = val).
# intercept: the creator REFUSES to build without an explicit lead-geometry choice
# (no silent default) — the pinned choice is the deployed hazard lead caps
# (a25 = 4.0/5.0 frames, same values stamped in the artifact's config block).
_ARTIFACTS: tuple[_Artifact, ...] = (
    _Artifact("intercept", "_aim_intercept_skill.json", "qnn.human.intercept", "both",
              ("--lead-hold-cap-frames", "4.0", "--lead-hold-cap-radial-frames", "5.0")),
    # window-sampled tracking (the trigger-free aim statistic; same ruler +
    # lead law as intercept, sampled over ±k windows around discharges) —
    # the decode-fit LADDER rides this; intercept stays the at-discharge
    # report card + crest-capture reference (decode-fit-v2 addendum 7/18)
    _Artifact("tracking", "_aim_tracking_window.json", "qnn.human.tracking", "both",
              ("--lead-hold-cap-frames", "4.0", "--lead-hold-cap-radial-frames", "5.0")),
    _Artifact("acquisition", "_acq_submovement.json", "qnn.human.acquisition", "both"),
    _Artifact("op_attack", "_op_attack_rate_byweapon.json", "qnn.human.op_attack", "val"),
    _Artifact("range", "_aim_range_byweapon.json", "qnn.human.aim_range", "both"),
)


@dataclass(frozen=True)
class _Table:
    key: str               # collect_metadata.json block key == config/<key>.json run.init pins
    module: str            # qnn.human submodule holding the compute entry
    func: str              # compute callable; takes (collect_dir, tick_hz=…) -> block


# The three cheap decode TABLES, parallel to _ARTIFACTS: each computes in-process from
# the corpus and lands as a top-level collect_metadata.json block (a different sink from
# the human_baseline/ docs — these are the keys run.init pins into a run's config/*.json).
_TABLES: tuple[_Table, ...] = (
    _Table("look_grid", "qnn.human.look_grid", "compute_from_collect"),
    _Table("move_hazard", "qnn.human.move_hazard", "compute_hazard_from_collect"),
    _Table("weapon_hazard", "qnn.human.weapon_hazard", "compute_hazard_from_collect"),
)


def _default_workers() -> int:
    # Cap at 8: other CPU evals share this box (feedback_no_collect_alongside_trainer).
    return min(8, max(1, (os.cpu_count() or 2) - 1))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return ""


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def baseline_dir(collect_dir: str | Path) -> Path:
    return Path(collect_dir) / BASELINE_SUBDIR


def baseline_paths(collect_dir: str | Path) -> dict[str, Path]:
    """The expected cache paths for a collect (no compute, no existence check)."""
    d = baseline_dir(collect_dir)
    return {a.key: d / a.filename for a in _ARTIFACTS}


def collect_from_run_dir(run_dir: str | Path) -> Path:
    """Resolve a bench run-dir's collect corpus from its config/machine.json
    (bc_data_dir), the same way the decode-fit pipeline does."""
    run_dir = Path(run_dir)
    machine = json.loads((run_dir / "config" / "machine.json").read_text())
    cd = Path(machine["bc_data_dir"])
    return cd if cd.is_absolute() else (_REPO / cd)


def baseline_paths_for_run(run_dir: str | Path) -> dict[str, Path]:
    """The collect-cached baseline paths for the corpus a run-dir was trained on — the
    self-contained accessor the aim-grid / degrade-sweep subprocess scripts use so they
    read the SAME collect cache as the pipeline (never the old global copy)."""
    return baseline_paths(collect_from_run_dir(run_dir))


def _compute_one(collect_dir: Path, art: _Artifact, out_path: Path, workers: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", art.module,
           "--collect-dir", str(collect_dir),
           "--split", art.split,
           "--workers", str(workers),
           "--out", str(out_path), *art.extra_args]
    subprocess.run(
        cmd, check=True, cwd=_REPO,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
             "PYTHONPATH": "src"})


def _write_provenance(collect_dir: Path, paths: dict[str, Path]) -> None:
    """Record the baseline provenance block in collect_metadata.json (same home as
    look_grid / move_hazard). File existence is the cache; this is the audit trail +
    staleness signal (git_commit)."""
    meta_path = Path(collect_dir) / "collect_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[_META_KEY] = {
        "schema": SCHEMA,
        "git_commit": _git_sha(),
        "subdir": BASELINE_SUBDIR,
        "artifacts": {
            a.key: {"file": f"{BASELINE_SUBDIR}/{a.filename}",
                    "sha256": _sha256(paths[a.key])}
            for a in _ARTIFACTS
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def ensure_collect_tables(
    collect_dir: str | Path, *, force: bool = False,
) -> dict:
    """Compute-if-missing the three cheap decode TABLES (look_grid / move_hazard /
    weapon_hazard) and record them as top-level blocks in the collect's
    ``collect_metadata.json`` — the keys ``run.init`` pins into a run's config.

    In-process, NumPy-only, best-effort per table: a label-less/empty corpus skips that
    table with a note rather than aborting (mirrors the old collect-loop behavior). This
    is the post-collect step the collect loop calls; ``run.init`` reads the result."""
    collect_dir = Path(collect_dir)
    meta_path = collect_dir / "collect_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tick_hz = meta.get("tick_hz")
    for t in _TABLES:
        if meta.get(t.key) is not None and not force:
            print(f"[human-baseline] cached: {t.key} (collect_metadata.json)", flush=True)
            continue
        try:
            print(f"[human-baseline] computing {t.key} (tick_hz={tick_hz}) …", flush=True)
            compute = getattr(importlib.import_module(t.module), t.func)
            meta[t.key] = compute(collect_dir, tick_hz=tick_hz)
        except Exception as exc:  # noqa: BLE001 — best-effort; never abort over a table
            print(f"[human-baseline] {t.key} skipped: {exc}", flush=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _artifact_stale(art: _Artifact, out: Path) -> str | None:
    """Cache-validity check for an existing doc — the reason it must be
    recomputed under the CURRENT schema, or None when the cache is good.
    v2: the intercept/tracking docs must carry the frozen
    ``placement_anchors`` node at the current ``ANCHORS_VERSION``
    (skill-curves §16.3); a pre-anchors doc is stale and recomputed here,
    never legacy-read (human_refs fails loud). The tracking doc must also
    carry the ``spread_of_median`` compat node (reachable_band reads it)."""
    if art.key not in ("intercept", "tracking"):
        return None
    try:
        doc = json.loads(out.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return f"unreadable ({e.__class__.__name__})"
    from qnn.human.intercept import ANCHORS_VERSION
    got = (doc.get("placement_anchors") or {}).get("anchors_version")
    if got != ANCHORS_VERSION:
        return f"placement_anchors version {got!r} != {ANCHORS_VERSION}"
    if art.key == "tracking" and not doc.get("spread_of_median"):
        return "missing spread_of_median compat node"
    return None


def ensure_from_collect(
    collect_dir: str | Path, *, force: bool = False, workers: int | None = None,
    include_tables: bool = True,
) -> dict[str, Path]:
    """Compute-if-missing ALL corpus-derived human baselines for a collect and return the
    ``human_baseline/`` doc paths. Cheap when already cached (existence check + skip).
    ``force`` recomputes every artifact. Produces the three decode tables
    (``include_tables``; into collect_metadata.json) then the four corpus-walk baselines
    (into human_baseline/). The decode-fit pipeline's stage-0 entry point and the
    ``python -m qnn.human`` backfill CLI both land here — runs once per collect and every
    subsequent model fit reuses the cache."""
    collect_dir = Path(collect_dir)
    if not collect_dir.is_dir():
        raise FileNotFoundError(f"collect dir not found: {collect_dir}")
    if include_tables:
        ensure_collect_tables(collect_dir, force=force)
    workers = workers or _default_workers()
    paths = baseline_paths(collect_dir)
    for art in _ARTIFACTS:
        out = paths[art.key]
        if out.exists() and not force:
            stale = _artifact_stale(art, out)
            if stale is None:
                print(f"[human-baseline] cached: {art.key} ({out.relative_to(collect_dir)})",
                      flush=True)
                continue
            print(f"[human-baseline] stale: {art.key} ({stale}) — recomputing",
                  flush=True)
        print(f"[human-baseline] computing {art.key} → {out.relative_to(collect_dir)} "
              f"(split={art.split}, workers={workers}) …", flush=True)
        _compute_one(collect_dir, art, out, workers)
        if not out.exists():
            raise RuntimeError(f"{art.module} did not produce {out}")
    _write_provenance(collect_dir, paths)
    return paths


def pinned_from_collect(collect_dir: str | Path) -> dict[str, Path]:
    """Read-only accessor: the cached baseline paths, failing loud with a backfill hint
    if any is missing (mirrors move_hazard.pinned_hazard_from_collect)."""
    collect_dir = Path(collect_dir)
    paths = baseline_paths(collect_dir)
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{collect_dir}: human baselines missing {missing} — backfill first: "
            f"`PYTHONPATH=src python -m qnn.human {collect_dir}`")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute/backfill the corpus-derived per-collect human baselines.")
    ap.add_argument("collect_dir", type=Path)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    paths = ensure_from_collect(args.collect_dir, force=args.force, workers=args.workers)
    print(f"\n[human-baseline] ready ({len(paths)} artifacts) under "
          f"{baseline_dir(args.collect_dir)}")
