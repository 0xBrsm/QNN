"""Demo classification for BC collection.

Labels each demo with everything the collector needs:

  play_start / play_end — frame range to collect
  match_found           — whether tournament start+end text was found
  total_frames          — total frames in the demo
  mode                  — duel, 2on2, 4on4, ffa, ctf, trick, unknown
  gamedir               — server mod directory (qw, ktx, id1, ...)
  bc_exclude / reason   — spectator/proxy heuristics

Supports .dem (NQ), .qwd (QW client), .mvd (QW multi-view server).

Usage:
    python -m demo.classify --demo-dir assets/corpus/qwd \\
        --manifest assets/corpus/qwd_manifest.ndjson
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, NamedTuple

# ── Match text patterns (mirror qnn_match.c) ────────────────────────

MATCH_START_RE = re.compile(
    r"match has begun|match started|match is \d+v\d+"
    r"|game is starting|match has started",
    re.IGNORECASE,
)

MATCH_END_RE = re.compile(
    r"match is over|match over|game over|has won over",
    re.IGNORECASE,
)


def _strip_highbit(raw: bytes) -> str:
    """Strip Quake's high-bit colored chars and decode to plain text."""
    return bytes(b & 0x7F for b in raw).decode("latin-1", errors="replace")


# ── QW mode classification ──────────────────────────────────────────

_FNAME_DUEL_RE = re.compile(r"^(duel|1on1|1v1)[_\[]", re.I)
_FNAME_4ON4_RE = re.compile(r"^(4on4|4v4|tdm)[_\[]", re.I)
_FNAME_2ON2_RE = re.compile(r"^(2on2|2v2)[_\[]", re.I)
_FNAME_CTF_RE = re.compile(r"^(ctf)[_\[]", re.I)
_FNAME_FFA_RE = re.compile(r"^(ffa)[_\[]", re.I)
_FNAME_TRICK_RE = re.compile(r"^(vb_|jss_)", re.I)
_MAP_TRICK_RE = re.compile(r"^(slide\d*|trick\d*|ztricks?\d*|endif|speed|race|surf|bhop)$", re.I)


def _int_or(info: dict, key: str, default: int = -1) -> int:
    try:
        return int(info[key])
    except (KeyError, ValueError):
        return default


def _classify_qw_mode(info: dict[str, str], filename: str) -> str:
    """Classify QW demo mode from serverinfo + filename."""
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

    if _FNAME_TRICK_RE.match(core):
        return "trick"
    if _MAP_TRICK_RE.match(info.get("map", "")):
        return "trick"

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
                      (_FNAME_FFA_RE, "ffa"), (_FNAME_TRICK_RE, "trick")]:
        if pat.match(core):
            return mode
    return "unknown"


# ── Result types ─────────────────────────────────────────────────────

class MatchBounds(NamedTuple):
    play_start: int
    play_end: int
    match_found: bool
    total_frames: int


class ClassifyResult(NamedTuple):
    bounds: MatchBounds
    bc_exclude: bool
    bc_exclude_reason: str | None
    gamedir: str
    mode: str

# Backwards compat alias
AnalysisResult = ClassifyResult


def _scan_payload_for_match(data: bytes, start: int, end: int
                            ) -> tuple[bool, bool]:
    """Check a message payload for match-start/end text patterns."""
    text = _strip_highbit(data[start:end])
    return (bool(MATCH_START_RE.search(text)),
            bool(MATCH_END_RE.search(text)))


# Serverinfo extraction for gamedir
_FULLINFO_RE = re.compile(
    rb'fullserverinfo\s+"?(\\[^\x00"]{10,4096})', re.DOTALL)
_KV_RE = re.compile(r"\\([^\\]+)\\([^\\]*)")


def _parse_qw_serverinfo(data: bytes, head: int = 65536) -> dict[str, str]:
    """Extract key/value pairs from the first fullserverinfo."""
    m = _FULLINFO_RE.search(data[:head])
    if not m:
        return {}
    text = m.group(1).decode("latin-1", errors="replace")
    return {k.lower().strip("*"): v for k, v in _KV_RE.findall(text)}


def _bounds_from_frames(match_start: int | None, match_end: int | None,
                        total: int) -> MatchBounds:
    if match_start is not None and match_end is not None:
        return MatchBounds(match_start, match_end, True, total)
    return MatchBounds(0, max(0, total - 1), False, total)


# ── NQ .dem format ───────────────────────────────────────────────────

# Import NQ protocol helpers — only needed for the deep-parse path
from .protocol import (
    Reader,
    STAT_HEALTH,
    SVC_CDTRACK,
    SVC_CENTERPRINT,
    SVC_CLIENTDATA,
    SVC_CUTSCENE,
    SVC_DAMAGE,
    SVC_DISCONNECT,
    SVC_FINALE,
    SVC_FOUNDSECRET,
    SVC_INTERMISSION,
    SVC_KILLEDMONSTER,
    SVC_LIGHTSTYLE,
    SVC_NOP,
    SVC_PARTICLE,
    SVC_PRINT,
    SVC_SELLSCREEN,
    SVC_SERVERINFO,
    SVC_SETANGLE,
    SVC_SETPAUSE,
    SVC_SETVIEW,
    SVC_SIGNONNUM,
    SVC_SOUND,
    SVC_SPAWNBASELINE,
    SVC_SPAWNSTATIC,
    SVC_SPAWNSTATICSOUND,
    SVC_STOPSOUND,
    SVC_STUFFTEXT,
    SVC_TEMP_ENTITY,
    SVC_TIME,
    SVC_UPDATECOLORS,
    SVC_UPDATEFRAGS,
    SVC_UPDATENAME,
    SVC_UPDATESTAT,
    SVC_VERSION,
    read_clientdata,
    skip_baseline,
    skip_damage,
    skip_entity_update,
    skip_particle,
    skip_serverinfo,
    skip_sound,
    skip_temp_entity,
)


def _classify_nq(data: bytes) -> ClassifyResult:
    """Parse an NQ .dem file — match boundaries + bc_exclude heuristics."""
    nl = data.find(b"\n")
    if nl < 0:
        return ClassifyResult(MatchBounds(0, 0, False, 0), True, "no_header", "id1", "unknown")
    pos = nl + 1

    frame = 0
    match_start_frame: int | None = None
    match_end_frame: int | None = None
    health_min = 999999
    health_max = -999999
    health_seen = False
    ammo_ever_decreased = False
    prev_shells = prev_nails = prev_rockets = prev_cells = -1

    while pos + 16 <= len(data):
        msg_len = struct.unpack_from("<i", data, pos)[0]
        msg_start = pos + 16
        msg_end = msg_start + msg_len
        if msg_end > len(data) or msg_len < 0 or msg_len > 65536:
            break

        # Deep parse for health/ammo/text
        r = Reader(data, msg_start)
        try:
            while r.pos < msg_end:
                cmd = r.byte()
                if cmd == 255:
                    break
                if cmd & 128:
                    skip_entity_update(r, cmd & 127)
                    continue
                if cmd == SVC_NOP:
                    continue
                if cmd == SVC_DISCONNECT:
                    break
                if cmd == SVC_UPDATESTAT:
                    stat_id = r.byte()
                    val = r.long()
                    if stat_id == STAT_HEALTH:
                        health_seen = True
                        health_min = min(health_min, val)
                        health_max = max(health_max, val)
                    continue
                if cmd == SVC_VERSION:
                    r.long()
                    continue
                if cmd == SVC_SETVIEW:
                    r.short()
                    continue
                if cmd == SVC_SOUND:
                    skip_sound(r)
                    continue
                if cmd == SVC_TIME:
                    r.float()
                    continue
                if cmd in (SVC_PRINT, SVC_STUFFTEXT, SVC_CENTERPRINT,
                           SVC_FINALE, SVC_CUTSCENE):
                    text = r.string()
                    if match_start_frame is None and MATCH_START_RE.search(text):
                        match_start_frame = frame
                    if (match_start_frame is not None
                            and match_end_frame is None
                            and MATCH_END_RE.search(text)):
                        match_end_frame = frame
                    continue
                if cmd == SVC_SETANGLE:
                    r.angle(); r.angle(); r.angle()
                    continue
                if cmd == SVC_SERVERINFO:
                    skip_serverinfo(r)
                    continue
                if cmd == SVC_LIGHTSTYLE:
                    r.byte(); r.string()
                    continue
                if cmd == SVC_UPDATENAME:
                    r.byte(); r.string()
                    continue
                if cmd == SVC_UPDATEFRAGS:
                    r.byte(); r.short()
                    continue
                if cmd == SVC_CLIENTDATA:
                    health, ammo, shells, nails, rockets, cells = read_clientdata(r)
                    health_seen = True
                    health_min = min(health_min, health)
                    health_max = max(health_max, health)
                    if prev_shells >= 0:
                        if (shells < prev_shells or nails < prev_nails
                                or rockets < prev_rockets or cells < prev_cells):
                            ammo_ever_decreased = True
                    prev_shells, prev_nails = shells, nails
                    prev_rockets, prev_cells = rockets, cells
                    continue
                if cmd == SVC_STOPSOUND:
                    r.short()
                    continue
                if cmd == SVC_UPDATECOLORS:
                    r.byte(); r.byte()
                    continue
                if cmd == SVC_PARTICLE:
                    skip_particle(r)
                    continue
                if cmd == SVC_DAMAGE:
                    skip_damage(r)
                    continue
                if cmd == SVC_SPAWNSTATIC:
                    skip_baseline(r)
                    continue
                if cmd == SVC_SPAWNBASELINE:
                    r.short(); skip_baseline(r)
                    continue
                if cmd == SVC_TEMP_ENTITY:
                    skip_temp_entity(r)
                    continue
                if cmd == SVC_SETPAUSE:
                    r.byte()
                    continue
                if cmd == SVC_SIGNONNUM:
                    r.byte()
                    continue
                if cmd in (SVC_KILLEDMONSTER, SVC_FOUNDSECRET, SVC_SELLSCREEN,
                           SVC_INTERMISSION):
                    continue
                if cmd == SVC_SPAWNSTATICSOUND:
                    r.coord(); r.coord(); r.coord()
                    r.byte(); r.byte(); r.byte()
                    continue
                if cmd == SVC_CDTRACK:
                    r.byte(); r.byte()
                    continue
                break
        except (IndexError, struct.error):
            pass

        pos = msg_end
        frame += 1

    if not health_seen:
        health_min = health_max = 0

    bounds = _bounds_from_frames(match_start_frame, match_end_frame, frame)

    exclude = False
    reason = None
    if health_min >= 100:
        exclude, reason = True, "health_never_below_100"
    if not ammo_ever_decreased:
        exclude, reason = True, "never_fired"

    return ClassifyResult(bounds, exclude, reason, "id1", "unknown")


# ── QWD / MVD format ─────────────────────────────────────────────────

DEM_CMD = 0
DEM_READ = 1
DEM_SET = 2
DEM_MULTIPLE = 3
DEM_SINGLE = 4
DEM_STATS = 5
DEM_ALL = 6
DEM_MASK = 7
DEM_CMD_PAYLOAD = 36


def _classify_qw(data: bytes, path: Path, is_mvd: bool) -> ClassifyResult:
    """Parse a QWD or MVD demo — match boundaries + spectator detection."""
    pos = 0
    frame = 0
    match_start_frame: int | None = None
    match_end_frame: int | None = None
    dem_cmd_count = 0
    has_forwardmove = False

    while pos + 5 <= len(data):
        type_byte = data[pos + 4]
        pos += 5
        dem_type = type_byte & DEM_MASK

        if dem_type == DEM_CMD:
            if is_mvd:
                break
            # Check forwardmove at usercmd offset 16 (short)
            if not has_forwardmove and pos + DEM_CMD_PAYLOAD <= len(data):
                fwd = struct.unpack_from("<h", data, pos + 16)[0]
                if fwd != 0:
                    has_forwardmove = True
            dem_cmd_count += 1
            pos += DEM_CMD_PAYLOAD
            frame += 1
            continue

        if dem_type == DEM_SET:
            pos += 8
            continue

        if dem_type == DEM_MULTIPLE:
            pos += 4

        if dem_type in (DEM_READ, DEM_MULTIPLE, DEM_SINGLE,
                        DEM_STATS, DEM_ALL):
            if pos + 4 > len(data):
                break
            msg_len = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            if msg_len < 0 or msg_len > 65536:
                break
            msg_start = pos
            msg_end = min(pos + msg_len, len(data))
            pos = msg_end

            has_start, has_end = _scan_payload_for_match(
                data, msg_start, msg_end)
            if match_start_frame is None and has_start:
                match_start_frame = frame
            if (match_start_frame is not None
                    and match_end_frame is None and has_end):
                match_end_frame = frame

            frame += 1
            continue

        continue

    bounds = _bounds_from_frames(match_start_frame, match_end_frame, frame)
    info = _parse_qw_serverinfo(data)
    gamedir = info.get("gamedir", "qw")
    mode = _classify_qw_mode(info, path.name)

    # QWD spectator detection: dem_cmd messages present but forwardmove
    # is always zero — the recording client was spectating, not playing.
    # The usercmd only contains camera angles, no actionable input.
    exclude = False
    reason = None
    if not is_mvd and dem_cmd_count > 0 and not has_forwardmove:
        exclude = True
        reason = "spectator_no_forwardmove"
    if not is_mvd and dem_cmd_count == 0:
        exclude = True
        reason = "no_usercmd"

    return ClassifyResult(bounds, exclude, reason, gamedir, mode)


# ── Unified entry point ──────────────────────────────────────────────

def classify_demo(path: Path) -> ClassifyResult:
    """Classify any supported demo format for BC collection."""
    data = path.read_bytes()
    ext = path.suffix.lower()
    if ext == ".dem":
        return _classify_nq(data)
    elif ext == ".mvd":
        return _classify_qw(data, path, is_mvd=True)
    else:
        return _classify_qw(data, path, is_mvd=False)


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
    import json
    import os
    import sys
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description="Analyze demos for match boundaries and BC exclusion"
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
    print(f"Analyzing {len(work)} demos with {n_workers} workers...",
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
    match_count = no_match = errors = excludes = 0

    for path_str, entry_idx in work:
        ar = results.get(path_str)
        entry = entries[entry_idx]
        if ar is None:
            errors += 1
            entry["bc_exclude"] = True
            entry["bc_exclude_reason"] = "parse_error"
            continue

        entry["play_start"] = ar.bounds.play_start
        entry["play_end"] = ar.bounds.play_end
        entry["match_found"] = ar.bounds.match_found
        entry["total_frames"] = ar.bounds.total_frames
        entry["gamedir"] = ar.gamedir
        entry["mode"] = ar.mode

        if ar.bc_exclude:
            entry["bc_exclude"] = True
            entry["bc_exclude_reason"] = ar.bc_exclude_reason
            excludes += 1
        elif "bc_exclude_reason" in entry and not entry.get("bc_exclude"):
            del entry["bc_exclude_reason"]

        if ar.bounds.match_found:
            match_count += 1
        else:
            no_match += 1

    print(f"Analyzed {len(work)} demos in {elapsed:.1f}s "
          f"({len(work)/elapsed:.0f} demos/s)")
    print(f"  match: {match_count}  no-match: {no_match}  "
          f"errors: {errors}  bc_exclude: {excludes}")

    if not args.dry_run:
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
