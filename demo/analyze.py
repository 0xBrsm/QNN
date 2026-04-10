"""Analyze raw demo files and update manifest with BC exclusion flags.

Parses each .dem file directly (no npy needed) to extract:
  - Health stats from svc_clientdata (STAT_HEALTH)
  - Match start/end from svc_print text
  - Ammo changes as proxy for firing

Exclusion rules:
  1. Health never drops below 100 during the demo -> spectator/proxy
  2. Ammo never decreases during the demo -> no firing (spectator/proxy)

Usage:
    python -m demo.analyze
    python -m demo.analyze --demo-dir assets/demos --manifest assets/demos/manifest.ndjson
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Dict

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

# Match text patterns
MATCH_START_RE = re.compile(
    r"match has begun|match started|match is \d+v\d+|game is starting|match has started",
    re.IGNORECASE,
)
MATCH_END_RE = re.compile(
    r"match is over|match over|game over|has won over",
    re.IGNORECASE,
)


def analyze_demo(path: Path) -> Dict[str, object]:
    """Parse a .dem file and extract BC-relevant metadata."""
    data = path.read_bytes()
    nl = data.index(b"\n")
    pos = nl + 1

    health_min = 999999
    health_max = -999999
    health_seen = False
    ammo_ever_decreased = False
    prev_shells = prev_nails = prev_rockets = prev_cells = -1
    match_start_text = False
    match_end_text = False
    saw_intermission = False

    while pos + 16 <= len(data):
        msg_len = struct.unpack_from("<i", data, pos)[0]
        msg_start = pos + 16
        msg_end = msg_start + msg_len
        if msg_end > len(data) or msg_len < 0 or msg_len > 65536:
            break

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
                if cmd in (SVC_PRINT, SVC_STUFFTEXT, SVC_CENTERPRINT, SVC_FINALE, SVC_CUTSCENE):
                    text = r.string()
                    if MATCH_START_RE.search(text):
                        match_start_text = True
                    if MATCH_END_RE.search(text):
                        match_end_text = True
                    continue
                if cmd == SVC_SETANGLE:
                    r.angle()
                    r.angle()
                    r.angle()
                    continue
                if cmd == SVC_SERVERINFO:
                    skip_serverinfo(r)
                    continue
                if cmd == SVC_LIGHTSTYLE:
                    r.byte()
                    r.string()
                    continue
                if cmd == SVC_UPDATENAME:
                    r.byte()
                    r.string()
                    continue
                if cmd == SVC_UPDATEFRAGS:
                    r.byte()
                    r.short()
                    continue
                if cmd == SVC_CLIENTDATA:
                    health, ammo, shells, nails, rockets, cells = read_clientdata(r)
                    health_seen = True
                    health_min = min(health_min, health)
                    health_max = max(health_max, health)
                    if prev_shells >= 0:
                        if (shells < prev_shells or nails < prev_nails or
                                rockets < prev_rockets or cells < prev_cells):
                            ammo_ever_decreased = True
                    prev_shells = shells
                    prev_nails = nails
                    prev_rockets = rockets
                    prev_cells = cells
                    continue
                if cmd == SVC_STOPSOUND:
                    r.short()
                    continue
                if cmd == SVC_UPDATECOLORS:
                    r.byte()
                    r.byte()
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
                    r.short()
                    skip_baseline(r)
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
                if cmd in (SVC_KILLEDMONSTER, SVC_FOUNDSECRET, SVC_SELLSCREEN):
                    continue
                if cmd == SVC_INTERMISSION:
                    saw_intermission = True
                    continue
                if cmd == SVC_SPAWNSTATICSOUND:
                    r.coord()
                    r.coord()
                    r.coord()
                    r.byte()
                    r.byte()
                    r.byte()
                    continue
                if cmd == SVC_CDTRACK:
                    r.byte()
                    r.byte()
                    continue
                break  # unknown cmd, stop parsing this block
        except (IndexError, struct.error):
            pass  # truncated message, move on

        pos = msg_end

    if not health_seen:
        health_min = 0
        health_max = 0

    return {
        "health_min": health_min,
        "health_max": health_max,
        "ammo_ever_decreased": ammo_ever_decreased,
        "match_start_text": match_start_text,
        "match_end_text": match_end_text,
        "saw_intermission": saw_intermission,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BC demos and update manifest")
    parser.add_argument("--demo-dir", default="assets/demos")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    manifest_path = Path(args.manifest) if args.manifest else demo_dir / "manifest.ndjson"

    if manifest_path.exists():
        entries = [json.loads(line) for line in manifest_path.read_text().strip().split("\n") if line.strip()]
        by_name: Dict[str, Dict] = {e["file"]: e for e in entries}
    else:
        by_name = {}

    new_entries = []
    for dem in sorted(demo_dir.glob("*.dem")):
        name = dem.name
        entry = by_name.get(name, {"file": name})
        entry["file"] = name

        print(f"  {name[:60]}...", end=" ", flush=True)
        try:
            analysis = analyze_demo(dem)
        except Exception as exc:
            print(f"ERROR: {exc}")
            entry["bc_exclude"] = True
            entry["bc_exclude_reason"] = "parse_error"
            new_entries.append(entry)
            continue

        entry["match_start_text"] = analysis["match_start_text"]
        entry["match_end_text"] = analysis["match_end_text"]
        entry["saw_intermission"] = analysis["saw_intermission"]
        entry["health_min"] = analysis["health_min"]
        entry["health_max"] = analysis["health_max"]
        entry["ammo_ever_decreased"] = analysis["ammo_ever_decreased"]

        exclude = False
        reason = None

        if analysis["health_min"] >= 100:
            exclude = True
            reason = "health_never_below_100"

        if not analysis["ammo_ever_decreased"]:
            exclude = True
            reason = "never_fired"

        entry["bc_exclude"] = exclude
        if reason:
            entry["bc_exclude_reason"] = reason
        elif "bc_exclude_reason" in entry:
            del entry["bc_exclude_reason"]

        status = f"EXCLUDE ({reason})" if exclude else "ok"
        print(f"hp=[{analysis['health_min']},{analysis['health_max']}] fired={analysis['ammo_ever_decreased']} {status}")

        new_entries.append(entry)

    manifest_path.write_text(
        "\n".join(json.dumps(e) for e in new_entries) + "\n"
    )

    excluded = [e for e in new_entries if e.get("bc_exclude")]
    included = [e for e in new_entries if not e.get("bc_exclude")]
    print(f"\n{len(new_entries)} demos analyzed")
    print(f"  Excluded: {len(excluded)}")
    for e in excluded:
        print(f"    {e['file'][:60]}  reason={e.get('bc_exclude_reason', '?')}")
    print(f"  Included: {len(included)}")


if __name__ == "__main__":
    main()
