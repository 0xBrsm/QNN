"""Demo classification for BC collection — thin wrapper around the C
qw_classifier binary.

Parsing (DEM_* records, svc_* opcode dispatch, match-text scan,
signon spec bit, fullserverinfo extraction) lives in
``src/demo/qw_classifier.c`` and was validated 1:1 against the
previous pure-Python implementation:

    QWD .qwd  — 4686 / 4686 demos match
    NQ  .dem  — 63   / 63   demos match
    MVD .mvd  — 4    / 4    testdata demos match

Build:
    bash src/engine/build/build_qw_classifier.sh

Policy (mode classification from filename + hostname + maxclients,
filename trick-prefix gate) stays in Python — easy to evolve
without rebuilds.  The classifier produces FACTS only (mode,
gamedir, recorder, label intervals, server-config fields); demo
inclusion/exclusion is the consumer's job (see src/qnn/bc/collect.py
filter config).

Usage:
    python -m demo.classify --demo-dir assets/corpus/qwd \\
        --manifest assets/corpus/qwd_manifest.ndjson
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


# ── QW mode classification ──────────────────────────────────────────

_FNAME_DUEL_RE = re.compile(r"^(duel|1on1|1v1)[_\[]", re.I)
_FNAME_4ON4_RE = re.compile(r"^(4on4|4v4|tdm)[_\[]", re.I)
_FNAME_2ON2_RE = re.compile(r"^(2on2|2v2)[_\[]", re.I)
_FNAME_CTF_RE = re.compile(r"^(ctf)[_\[]", re.I)
_FNAME_FFA_RE = re.compile(r"^(ffa)[_\[]", re.I)
_MAP_TRICK_RE = re.compile(
    r'^(slide\d*|trick\d*|ztricks?\d*|endif|speed|race|surf|bhop|'
    r'p_freestyle|jqfs\d*|jq\d*)$',
    re.I,
)

# Single source of truth for trick-style detection: feeds
# `mode="trick"` classification.  Substrings matched anywhere in
# filename or hostname (unanchored), covering common naming
# conventions for trick/practice/movie recordings and trick-server
# hostnames.  `vb[-_]` matches both `vb_` and `vb-` (dash-variant
# from some sources).
_TRICK_TERM_RE = re.compile(
    r'(vb[-_]|bogo|JSS[-_]|jss[-_]|rj\d|rj_|run\d|slide\d?|trick\d*|race\d?|'
    r'jump|speed|bhop|surf|stunt|defrag|hook|telebug|skinny|powtest|fastest|'
    r'fragmovie|wankspeed|hijump|hi-jump|2bfree|way2ez|jq[-_\d]|jqfs|'
    r'freestyle|trickserver|trick\s*server)',
    re.IGNORECASE,
)


def _int_or(info: dict, key: str, default: int = -1) -> int:
    try:
        return int(info[key])
    except (KeyError, ValueError):
        return default


def _classify_qw_mode(info: dict[str, str], filename: str) -> str:
    """Classify QW demo mode from serverinfo + filename.

    Trick / practice detection runs FIRST — a demo recorded on a trick
    server (freestyle hostname, vb-/jq-style filename, dedicated trick
    map) is "not a real match" even if the server's KTX `mode` cvar
    says "ffa". """
    # Trick / non-match detection (overrides game-mode classification).
    if _TRICK_TERM_RE.search(filename):
        return "trick"
    if _MAP_TRICK_RE.match(info.get("map", "")):
        return "trick"
    if _TRICK_TERM_RE.search(info.get("hostname", "")):
        return "trick"

    core = filename.split("_", 1)[1] if "_" in filename else filename

    ktx_mode = info.get("mode", "").lower().strip()
    if ktx_mode:
        for pat, mode in [("duel", "duel"), ("1on1", "duel"),
                          ("2on2", "2on2"), ("2v2", "2on2"),
                          ("4on4", "4on4"), ("4v4", "4on4"),
                          ("ctf", "ctf"), ("ffa", "ffa")]:
            if pat in ktx_mode:
                return mode

    serverdemo = info.get("serverdemo", "").lower()
    if serverdemo:
        for prefix, mode in [("duel", "duel"), ("1on1", "duel"),
                             ("4on4", "4on4"), ("4v4", "4on4"),
                             ("2on2", "2on2"), ("2v2", "2on2"),
                             ("ctf", "ctf"), ("ffa", "ffa")]:
            if serverdemo.startswith(prefix):
                return mode

    teamplay = _int_or(info, "teamplay")
    maxclients = _int_or(info, "maxclients")

    if teamplay > 0:
        if _FNAME_2ON2_RE.match(core):
            return "2on2"
        if _FNAME_CTF_RE.match(core):
            return "ctf"
        return "2on2" if maxclients <= 4 else "4on4"

    if teamplay == 0:
        if maxclients == 2 or _FNAME_DUEL_RE.match(core):
            return "duel"
        if _FNAME_FFA_RE.match(core) or maxclients > 2:
            if "vs" in core.lower() or "1on1" in core.lower():
                return "duel"
            return "ffa"
        if "vs" in core.lower():
            return "duel"
        return "ffa"

    for pat, mode in [(_FNAME_DUEL_RE, "duel"), (_FNAME_4ON4_RE, "4on4"),
                      (_FNAME_2ON2_RE, "2on2"), (_FNAME_CTF_RE, "ctf"),
                      (_FNAME_FFA_RE, "ffa")]:
        if pat.match(core):
            return mode
    return "unknown"


# ── Result types ─────────────────────────────────────────────────────

class MatchBounds(NamedTuple):
    """Demo-level bounds.  play_start/play_end/match_found and the two
    match_*_text booleans were dropped: that information now lives in
    the `labels.match` interval (empty interval = no match detected)."""
    total_frames: int


class ClassifyResult(NamedTuple):
    bounds: MatchBounds
    gamedir: str
    mode: str
    labels: dict[str, list[list[int]]] = {}
    recorder: str = "player"
    # Per-frame active-input tallies; populated literally on QWD, all-zero
    # on MVD/NQ until inference adapters ship for those formats.
    active_input: dict[str, int] = {}
    # Per-frame inventory/state delta tallies (health/armor/ammo/items/frags).
    active_state: dict[str, int] = {}
    # Frame at which the walker bailed on corrupt / truncated data.
    # None when the walker reached EOF cleanly.
    error_frame: int | None = None
    # Server-config facts extracted from the demo's signon serverinfo.
    # None when the demo lacks fullserverinfo (e.g. NQ).
    teamplay: int | None = None
    maxclients: int | None = None
    deathmatch: int | None = None
    hostname: str | None = None
    map: str | None = None
    # Mean svc_updateping value for the recorder's slot (QWD only).
    # None when the demo has no usable ping samples.
    avg_ping_ms: float | None = None


# Backwards-compat alias used by callers.
AnalysisResult = ClassifyResult


# ── C classifier subprocess wrapper ──────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_BINARY = _REPO_ROOT / "assets" / "bin" / "qw_classifier"

_PROC: subprocess.Popen | None = None
_PROC_PID: int | None = None


def _classifier_binary() -> Path:
    if not _CLASSIFIER_BINARY.exists():
        raise RuntimeError(
            f"qw_classifier binary not found at {_CLASSIFIER_BINARY}. "
            f"Build with: bash src/engine/build/build_qw_classifier.sh"
        )
    return _CLASSIFIER_BINARY


def _get_proc() -> subprocess.Popen:
    """Per-process persistent qw_classifier subprocess.

    Lazy-initialized; respawned across fork boundaries so each
    multiprocessing worker gets its own subprocess (the parent's
    pipe handles aren't safe to share).
    """
    global _PROC, _PROC_PID
    cur_pid = os.getpid()
    if _PROC is None or _PROC_PID != cur_pid or _PROC.poll() is not None:
        binary = _classifier_binary()
        _PROC = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        _PROC_PID = cur_pid
    return _PROC


def _shutdown_proc() -> None:
    global _PROC
    proc = _PROC
    if proc is None or proc.poll() is not None:
        _PROC = None
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _PROC = None


atexit.register(_shutdown_proc)


def _call_classifier(path: Path) -> dict:
    """Send one demo path to the C classifier, return its parsed JSON."""
    proc = _get_proc()
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(str(path).encode() + b"\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"qw_classifier closed stdout for {path}")
    return json.loads(line.decode("utf-8", errors="replace"))


def _result_from_classifier(obj: dict, path: Path) -> ClassifyResult:
    """Build a ClassifyResult from one JSON response."""
    bounds = MatchBounds(total_frames=obj["total_frames"])

    fmt = obj.get("format", "qwd")
    info = obj.get("info") or {}
    if fmt == "dem":
        gamedir = "id1"
        mode = "unknown"
    else:
        gamedir = info.get("gamedir", "qw")
        mode = _classify_qw_mode(info, path.name)

    # Optional duration sanity warning, matches old per-format behavior.
    if bounds.total_frames > 0:
        native_hz = 72.0 if fmt == "dem" else 77.0
        seconds = bounds.total_frames / native_hz
        if seconds < 300:
            print(f"[classify] short demo ({seconds:.0f}s): "
                  f"{path.name} — review for trick/snippet",
                  file=sys.stderr)

    labels = obj.get("labels") or {}
    recorder = obj.get("recorder", "player")
    active_input = obj.get("active_input") or {}
    active_state = obj.get("active_state") or {}
    error_frame = obj.get("error_frame")
    # Pull server-config facts from the same info dict that drives gamedir
    # and mode classification — keep all serverinfo-derived metadata flowing
    # through the classifier as the single source of truth.
    def _opt_int(key: str) -> int | None:
        try:
            return int(info[key])
        except (KeyError, ValueError, TypeError):
            return None
    teamplay = _opt_int("teamplay")
    maxclients = _opt_int("maxclients")
    deathmatch = _opt_int("deathmatch")
    hostname = info.get("hostname") or None
    map_name = info.get("map") or None
    avg_ping_ms = obj.get("avg_ping_ms")
    return ClassifyResult(bounds, gamedir, mode, labels, recorder,
                          active_input, active_state, error_frame,
                          teamplay, maxclients, deathmatch, hostname, map_name,
                          avg_ping_ms)


def classify_demo(path: Path) -> ClassifyResult:
    """Classify any supported demo format for BC collection.

    Classification produces FACTS only.  Trick recognition is conveyed
    via `mode="trick"`; gamedir is reported verbatim.  Consumers
    filter on those fields directly in src/qnn/bc/collect.py. """
    obj = _call_classifier(path)
    if not obj.get("ok"):
        return ClassifyResult(
            MatchBounds(0), "qw", "unknown", {}, "player", {}, {}, None,
        )
    return _result_from_classifier(obj, path)


# Backwards compat aliases
analyze_demo = classify_demo
analyze_qw_demo = classify_demo


# ── CLI ──────────────────────────────────────────────────────────────

def _classify_one(args: tuple) -> tuple[str, ClassifyResult | None]:
    demo_path_str, = args
    try:
        return (demo_path_str, classify_demo(Path(demo_path_str)))
    except Exception:
        return (demo_path_str, None)


def main() -> None:
    import argparse
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description="Classify demos: per-frame labels + structural facts"
    )
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = CPU count)")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    manifest_path = Path(args.manifest)
    entries = [json.loads(l) for l in
               manifest_path.read_text().strip().splitlines()]

    work: list[tuple[str, int]] = []
    for i, entry in enumerate(entries):
        demo_path = demo_dir / entry["file"]
        if demo_path.exists():
            work.append((str(demo_path), i))

    n_workers = args.workers or os.cpu_count() or 4
    print(f"Classifying {len(work)} demos with {n_workers} workers...",
          file=sys.stderr)

    results: dict[str, ClassifyResult | None] = {}
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_classify_one, (path,)): path
            for path, _ in work
        }
        done_count = 0
        for future in as_completed(futures):
            path_str, result = future.result()
            results[path_str] = result
            done_count += 1
            if done_count % 500 == 0:
                elapsed = time.monotonic() - t0
                print(f"  {done_count}/{len(work)} "
                      f"({done_count/elapsed:.0f} demos/s)...",
                      file=sys.stderr)

    elapsed = time.monotonic() - t0
    match_count = no_match = parse_errors = 0

    # Each pass rebuilds the entry from scratch — only the
    # corpus-management fields below are carried over from any prior
    # manifest state.  Everything else (mode, gamedir, labels, etc.)
    # is sourced from this run's classifier.  This is the single
    # source of truth for what columns the manifest contains; older
    # schema fields are dropped on rewrite.
    _CORPUS_KEYS = ("file", "source", "sha256", "bytes", "format")

    for path_str, entry_idx in work:
        ar = results.get(path_str)
        old = entries[entry_idx]
        entry = {k: old[k] for k in _CORPUS_KEYS if k in old}
        entries[entry_idx] = entry
        if ar is None:
            parse_errors += 1
            entry["parse_error"] = True
            continue

        entry["parse_error"] = False
        entry["total_frames"] = ar.bounds.total_frames
        entry["gamedir"] = ar.gamedir
        entry["mode"] = ar.mode
        entry["labels"] = ar.labels
        entry["recorder"] = ar.recorder
        entry["active_input"] = ar.active_input
        entry["active_state"] = ar.active_state
        if ar.error_frame is not None:
            entry["error_frame"] = ar.error_frame
        else:
            entry.pop("error_frame", None)
        # Server-config facts from signon serverinfo — classifier is the
        # source of truth.  Only write when the classifier actually
        # extracted a value (None for NQ demos that lack fullserverinfo).
        for k, v in (("teamplay", ar.teamplay),
                     ("maxclients", ar.maxclients),
                     ("deathmatch", ar.deathmatch),
                     ("hostname", ar.hostname),
                     ("map", ar.map),
                     ("avg_ping_ms", ar.avg_ping_ms)):
            if v is not None:
                entry[k] = v

        if "match" in ar.labels:
            match_count += 1
        else:
            no_match += 1

    print(f"Classified {len(work)} demos in {elapsed:.1f}s "
          f"({len(work)/elapsed:.0f} demos/s)")
    print(f"  match: {match_count}  no-match: {no_match}  "
          f"parse_errors: {parse_errors}")

    if not args.dry_run:
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
