"""Unified per-head analysis dispatcher for the Phase-2 ``qnn.diag analyze`` CLI.

Runs each requested head's ``analyze(policy, source, ...)`` function and
collects results into the canonical schema::

    {
        "run":     "<run_id>",
        "cache":   "<cache_dir>",
        "segment": "engaged" | "all",
        "heads":   { "attack": {...}, "look": {...}, "move": {...} },
        "meta":    { "n_episodes": N, "n_frames": M }
    }

The attack-with output is analyzed as part of the attack slice; there is no
separate equipped-weapon diagnostic.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

import torch


# Segment mask for "engaged" frames (target present).
ENGAGED_MASK: dict = {"act.target": {"$ne": 0}}

# Ordered list of heads to attempt.  Each entry: (head_name, module_path)
_HEAD_MODULES = [
    ("attack", "qnn.diag.attack"),
    ("look",   "qnn.diag.look"),
    ("move",   "qnn.diag.move"),
]


def _build_segment_mask(segment: str) -> dict | None:
    """Return the segment_mask dict for the given --segment value."""
    if segment == "engaged":
        return ENGAGED_MASK
    if segment == "all":
        return None
    raise ValueError(f"Unknown --segment value {segment!r}; expected 'engaged' or 'all'")


def run_analyze(
    run_dir: Path,
    cache_dir: Path,
    *,
    heads: list[str] | None = None,
    segment: str = "engaged",
    device: str | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """Load policy + source, dispatch per-head analyze(), collect results.

    Parameters
    ----------
    run_dir:
        Path to a bench run directory (must contain ``config/probe.json``
        and a checkpoint under ``checkpoints/``).
    cache_dir:
        Root cache directory.  The resident source is built from
        ``<cache_dir>/precomputed_val``.
    heads:
        List of head names to run (e.g. ``["attack", "move"]``).
        Defaults to all available heads: ``["attack", "look", "move"]``
        plus ``"weapon"`` if present.
    segment:
        ``"engaged"`` → ``segment_mask={"act.target": {"$ne": 0}}``;
        ``"all"``      → ``segment_mask=None``.
    device:
        Torch device string.  Defaults to ``"cuda"`` if available else
        ``"cpu"``.
    out:
        Optional JSON output path.  Written as a side-effect; the dict is
        also returned.

    Returns
    -------
    Report dict matching the Phase-2 schema.
    """
    from qnn.diag.loader import load_policy
    from qnn.bc.supervised_loop import make_resident_source_from_cache

    run_dir  = Path(run_dir)
    cache_dir = Path(cache_dir)
    val_dir   = cache_dir / "precomputed_val"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── run_id ──────────────────────────────────────────────────────────────
    run_json = run_dir / "run.json"
    if run_json.exists():
        run_id = json.loads(run_json.read_text()).get("run_id", run_dir.name)
    else:
        run_id = run_dir.name

    print(f"[analyze] run_id={run_id}  segment={segment}  device={device}")
    print(f"[analyze] run_dir  = {run_dir}")
    print(f"[analyze] cache    = {cache_dir}")
    print(f"[analyze] val_dir  = {val_dir}")

    # ── load policy ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    print("[analyze] loading policy …", flush=True)
    policy, _probe = load_policy(run_dir, device=device)
    print(f"[analyze] policy loaded in {time.monotonic()-t0:.1f}s", flush=True)

    # ── build resident source ────────────────────────────────────────────────
    segment_mask = _build_segment_mask(segment)
    print(f"[analyze] building resident source (segment_mask={segment_mask}) …",
          flush=True)
    t1 = time.monotonic()
    source = make_resident_source_from_cache(
        val_dir,
        torch.device(device),
        segment_mask=segment_mask,
    )
    n_frames   = int(source.n_total_rows)
    n_episodes = max(0, len(source.episode_offsets) - 1)
    print(f"[analyze] source ready: {n_episodes} episodes, {n_frames:,} frames "
          f"({time.monotonic()-t1:.1f}s)", flush=True)

    # ── resolve heads ────────────────────────────────────────────────────────
    if heads is None:
        requested = [h for h, _ in _HEAD_MODULES]
    else:
        requested = list(heads)

    # ── dispatch per-head ────────────────────────────────────────────────────
    results: dict[str, Any] = {}
    for head_name, module_path in _HEAD_MODULES:
        if head_name not in requested:
            continue

        # Optional heads (weapon): skip gracefully if module absent.
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            print(f"[analyze] {head_name}: module {module_path!r} not found — skipping",
                  flush=True)
            continue

        if not hasattr(mod, "analyze"):
            print(f"[analyze] {head_name}: {module_path}.analyze not found — skipping",
                  flush=True)
            continue

        print(f"[analyze] running {head_name}.analyze() …", flush=True)
        t2 = time.monotonic()
        try:
            head_result = mod.analyze(policy, source, segment=segment)
        except Exception as exc:
            print(f"[analyze] {head_name}: ERROR — {exc}", flush=True)
            head_result = {"error": str(exc)}
        elapsed = time.monotonic() - t2
        print(f"[analyze] {head_name}: done in {elapsed:.1f}s", flush=True)
        results[head_name] = head_result

    # ── assemble report ──────────────────────────────────────────────────────
    report: dict[str, Any] = {
        "run":     run_id,
        "cache":   str(cache_dir),
        "segment": segment,
        "heads":   results,
        "meta": {
            "n_episodes": n_episodes,
            "n_frames":   n_frames,
        },
    }

    # ── write output ─────────────────────────────────────────────────────────
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[analyze] wrote {out}", flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    _print_summary(report)

    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a short human-readable summary after the run."""
    meta    = report.get("meta", {})
    heads   = report.get("heads", {})
    segment = report.get("segment", "?")
    run_id  = report.get("run",     "?")

    print()
    print("=" * 60)
    print(f"  diag analyze summary")
    print(f"  run:     {run_id}")
    print(f"  segment: {segment}")
    print(f"  frames:  {meta.get('n_frames', '?'):,}  "
          f"episodes: {meta.get('n_episodes', '?')}")
    print(f"  heads:   {list(heads.keys())}")
    print()
    for head, result in heads.items():
        if "error" in result:
            print(f"  [{head}]  ERROR: {result['error']}")
            continue
        # Print a small per-head digest.
        note = result.get("note")
        if note:
            print(f"  [{head}]  (stub) {note[:80]}")
            continue
        keys = [k for k in result if k != "segment"]
        print(f"  [{head}]  segment={result.get('segment', '?')}  "
              f"keys={keys}")
        # Attack: show void ratio if present.
        od = result.get("offset_distribution", {})
        if od:
            print(f"         n_pred={od.get('n_pred', '?')}  "
                  f"n_true={od.get('n_true', '?')}  "
                  f"threshold={od.get('threshold', '?')}")
            ptrue = od.get("pred_to_true", {})
            if ptrue:
                print(f"         pred→true  median={ptrue.get('median', '?')}  "
                      f"lead%={ptrue.get('lead_pct', '?'):.1f}")
        # Move: show jump discrim summary if present.
        jd = result.get("jump_discrim", {})
        if jd:
            print(f"         jump_discrim  auc_all={jd.get('auc_all', '?'):.4f}  "
                  f"human_jump_rate={jd.get('human_jump_rate', '?'):.4f}  "
                  f"n_total={jd.get('n_total', '?'):,}")
    print("=" * 60)
