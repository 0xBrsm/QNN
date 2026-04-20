"""Post-hoc analysis of demo files and collected data.

Helpers for understanding corpus characteristics, debugging collection
quality, and investigating demo format variations.  This is NOT used by
the collection pipeline — see demo.classify for that.
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path
from typing import Dict, NamedTuple


class QWDProfile(NamedTuple):
    """Per-demo structural profile of a QWD file."""
    dem_cmd_count: int        # number of dem_cmd (usercmd) messages
    dem_read_count: int       # number of dem_read (server) messages
    dem_set_count: int        # dem_set (sequence sync) messages
    has_usercmd: bool         # dem_cmd_count > 0
    impulse_nonzero: int      # frames where impulse != 0
    impulse_unique: int       # distinct nonzero impulse values seen
    impulse_changes: int      # frames where impulse changed from previous
    forwardmove_nonzero: int  # frames with nonzero forwardmove
    total_frames: int         # dem_cmd + dem_read


class QWDStats(NamedTuple):
    """Detailed per-frame key/button/impulse statistics for a QWD file."""
    dem_cmd_count: int
    # Button counts (bits in the buttons byte)
    attack_frames: int        # BUTTON_ATTACK (bit 0)
    jump_frames: int          # BUTTON_JUMP (bit 1)
    # Movement
    forwardmove_pos: int      # forwardmove > 0
    forwardmove_neg: int      # forwardmove < 0
    sidemove_pos: int         # sidemove > 0 (strafe right)
    sidemove_neg: int         # sidemove < 0 (strafe left)
    upmove_pos: int           # upmove > 0 (jump/swim up)
    upmove_neg: int           # upmove < 0 (swim down)
    # Impulse distribution
    impulse_counts: Dict[int, int]   # {impulse_value: frame_count}
    impulse_changes: int      # frames where impulse value transitioned
    # Run lengths for impulse (how long each non-zero value persists)
    impulse_run_lengths: list[int]


# dem_cmd payload layout (after the 5-byte frame header):
#   usercmd_t (24 bytes, C struct with natural alignment):
#     byte msec            offset 0
#     (3 bytes padding)
#     vec3_t angles        offset 4   (3 floats, 12 bytes)
#     short forwardmove    offset 16
#     short sidemove       offset 18
#     short upmove         offset 20
#     byte buttons         offset 22
#     byte impulse         offset 23
#   3 float viewangles     offset 24  (cl.viewangles, 12 bytes)
# Total payload = 36 bytes
#
# Verified by cross-checking buttons (should be 0-7) and impulse
# (should be sparse 0 with occasional 1-12) across 2,653 QWD files.
# Layout A (usercmd first) is universal — no client variant found.

_DEM_CMD_PAYLOAD = 36
_DEM_MASK = 7

# QW server message types we parse from dem_read payloads.
_SVC_SERVERDATA = 11     # protocol, servercount, gamedir, playernum, levelname
_SVC_MAXSPEED = 49       # float maxspeed
_SVC_SERVERINFO = 52     # key\0value\0 (serverinfo strings)


class QWDServerInfo(NamedTuple):
    """Server-side metadata extracted from the demo's server messages."""
    gamedir: str             # from svc_serverdata (e.g. "qw", "ctf", "ktx")
    levelname: str           # map title from svc_serverdata
    protocol: int            # QW protocol version (typically 28)
    maxspeed: float          # from svc_maxspeed (0.0 if not found)
    serverinfo: Dict[str, str]  # key-value pairs from svc_serverinfo


def _read_null_string(data: bytes, pos: int, limit: int) -> tuple[str, int]:
    """Read a null-terminated string. Returns (string, position after null)."""
    end = data.find(b'\0', pos, limit)
    if end < 0:
        return '', limit
    return data[pos:end].decode('latin-1', errors='replace'), end + 1


def serverinfo_qwd(path: Path) -> QWDServerInfo:
    """Extract server metadata from a QWD file's server messages.

    Scans dem_read packets for svc_serverdata (gamedir, level, protocol),
    svc_maxspeed, and svc_serverinfo key-value pairs.
    """
    data = path.read_bytes()
    pos = 0
    gamedir = ''
    levelname = ''
    protocol = 0
    maxspeed = 0.0
    serverinfo: Dict[str, str] = {}

    while pos + 5 <= len(data):
        type_byte = data[pos + 4]
        pos += 5
        dem_type = type_byte & _DEM_MASK

        if dem_type == 0:  # dem_cmd — skip
            if pos + _DEM_CMD_PAYLOAD > len(data):
                break
            pos += _DEM_CMD_PAYLOAD
            continue

        if dem_type == 2:  # dem_set
            pos += 8
            continue

        if dem_type == 1:  # dem_read — scan for server messages
            if pos + 4 > len(data):
                break
            msg_len = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            if msg_len < 0 or msg_len > 65536:
                break
            msg_end = pos + msg_len

            # Skip 8-byte sequence header
            mp = pos + 8

            # Parse svc_serverdata if this is the first message.
            # It appears once, at the start, and contains gamedir/level.
            if not gamedir and mp < msg_end and data[mp] == _SVC_SERVERDATA:
                mp += 1
                if mp + 8 <= msg_end:
                    protocol = struct.unpack_from('<i', data, mp)[0]
                    mp += 8  # skip protocol + servercount
                    gamedir, mp = _read_null_string(data, mp, msg_end)
                    if mp < msg_end:
                        mp += 1  # playernum byte
                        levelname, mp = _read_null_string(data, mp, msg_end)

            # Scan for svc_maxspeed (byte 0x31 followed by a round
            # float in [128,512]).  Common values: 320, 400, 200.
            # Round floats have 0x00 0x00 in the mantissa low bytes and
            # 0x43 as the exponent byte (byte 3), so we match that
            # pattern to avoid false positives from random data.
            if maxspeed == 0.0:
                scan = pos + 8
                scan_limit = min(msg_end - 4, len(data) - 5)
                while scan < scan_limit:
                    if (data[scan] == _SVC_MAXSPEED
                            and data[scan + 1] == 0x00
                            and data[scan + 2] == 0x00
                            and data[scan + 4] == 0x43):
                        val = struct.unpack_from('<f', data, scan + 1)[0]
                        if 128.0 <= val <= 512.0:
                            maxspeed = val
                            break
                    scan += 1

            pos = msg_end
            continue

        # Unknown dem type
        continue

    return QWDServerInfo(
        gamedir=gamedir,
        levelname=levelname,
        protocol=protocol,
        maxspeed=maxspeed,
        serverinfo=serverinfo,
    )


def profile_qwd(path: Path) -> QWDProfile:
    """Scan a QWD file and profile its usercmd structure."""
    data = path.read_bytes()
    pos = 0
    cmd_count = read_count = set_count = 0
    impulse_nonzero = 0
    impulse_values: set[int] = set()
    impulse_changes = 0
    fwd_nonzero = 0
    prev_impulse = -1

    while pos + 5 <= len(data):
        type_byte = data[pos + 4]
        pos += 5
        dem_type = type_byte & _DEM_MASK

        if dem_type == 0:  # dem_cmd
            if pos + _DEM_CMD_PAYLOAD > len(data):
                break
            # Extract fields from usercmd_t (starts at payload offset 0)
            fwd = struct.unpack_from("<h", data, pos + 16)[0]
            impulse = data[pos + 23]

            if impulse != 0:
                impulse_nonzero += 1
                impulse_values.add(impulse)
            if impulse != prev_impulse and prev_impulse >= 0:
                impulse_changes += 1
            prev_impulse = impulse
            if fwd != 0:
                fwd_nonzero += 1

            cmd_count += 1
            pos += _DEM_CMD_PAYLOAD
            continue

        if dem_type == 2:  # dem_set
            pos += 8
            set_count += 1
            continue

        if dem_type == 1:  # dem_read
            if pos + 4 > len(data):
                break
            msg_len = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            if msg_len < 0 or msg_len > 65536:
                break
            pos += msg_len
            read_count += 1
            continue

        # Unknown type — try to continue
        continue

    total = cmd_count + read_count
    return QWDProfile(
        dem_cmd_count=cmd_count,
        dem_read_count=read_count,
        dem_set_count=set_count,
        has_usercmd=cmd_count > 0,
        impulse_nonzero=impulse_nonzero,
        impulse_unique=len(impulse_values),
        impulse_changes=impulse_changes,
        forwardmove_nonzero=fwd_nonzero,
        total_frames=total,
    )


def stats_qwd(path: Path) -> QWDStats:
    """Extract detailed per-frame key/button/impulse statistics from a QWD."""
    data = path.read_bytes()
    pos = 0
    cmd_count = 0
    attack = jump = 0
    fwd_pos = fwd_neg = side_pos = side_neg = up_pos = up_neg = 0
    impulse_counts: Dict[int, int] = Counter()
    impulse_changes = 0
    impulse_runs: list[int] = []
    prev_impulse = -1
    cur_run_val = -1
    cur_run_len = 0

    while pos + 5 <= len(data):
        type_byte = data[pos + 4]
        pos += 5
        dem_type = type_byte & _DEM_MASK

        if dem_type == 0:  # dem_cmd
            if pos + _DEM_CMD_PAYLOAD > len(data):
                break
            fwd = struct.unpack_from("<h", data, pos + 16)[0]
            side = struct.unpack_from("<h", data, pos + 18)[0]
            up = struct.unpack_from("<h", data, pos + 20)[0]
            buttons = data[pos + 22]
            impulse = data[pos + 23]

            if buttons & 1: attack += 1
            if buttons & 2: jump += 1
            if fwd > 0: fwd_pos += 1
            elif fwd < 0: fwd_neg += 1
            if side > 0: side_pos += 1
            elif side < 0: side_neg += 1
            if up > 0: up_pos += 1
            elif up < 0: up_neg += 1

            impulse_counts[impulse] += 1
            if impulse != prev_impulse and prev_impulse >= 0:
                impulse_changes += 1
            # Track run lengths for non-zero impulse
            if impulse == cur_run_val:
                cur_run_len += 1
            else:
                if cur_run_val > 0 and cur_run_len > 0:
                    impulse_runs.append(cur_run_len)
                cur_run_val = impulse
                cur_run_len = 1
            prev_impulse = impulse

            cmd_count += 1
            pos += _DEM_CMD_PAYLOAD
            continue

        if dem_type == 2:  # dem_set
            pos += 8
            continue

        if dem_type == 1:  # dem_read
            if pos + 4 > len(data):
                break
            msg_len = struct.unpack_from("<i", data, pos)[0]
            pos += 4
            if msg_len < 0 or msg_len > 65536:
                break
            pos += msg_len
            continue

        continue

    # Flush last run
    if cur_run_val > 0 and cur_run_len > 0:
        impulse_runs.append(cur_run_len)

    return QWDStats(
        dem_cmd_count=cmd_count,
        attack_frames=attack,
        jump_frames=jump,
        forwardmove_pos=fwd_pos,
        forwardmove_neg=fwd_neg,
        sidemove_pos=side_pos,
        sidemove_neg=side_neg,
        upmove_pos=up_pos,
        upmove_neg=up_neg,
        impulse_counts=dict(impulse_counts),
        impulse_changes=impulse_changes,
        impulse_run_lengths=impulse_runs,
    )


def dump_stats(path: Path) -> None:
    """Print human-readable stats for a QWD file."""
    s = stats_qwd(path)
    si = serverinfo_qwd(path)
    n = max(s.dem_cmd_count, 1)
    print(f"File: {path.name}")
    print(f"Frames: {s.dem_cmd_count}")
    print()
    print("=== Server ===")
    print(f"  gamedir:    {si.gamedir or '(unknown)'}")
    print(f"  level:      {si.levelname or '(unknown)'}")
    print(f"  protocol:   {si.protocol}")
    print(f"  maxspeed:   {si.maxspeed if si.maxspeed > 0 else '(not found)'}")
    if si.serverinfo:
        for k, v in sorted(si.serverinfo.items()):
            print(f"  info {k}: {v}")
    print()
    print("=== Buttons ===")
    print(f"  attack:  {s.attack_frames:6d} ({100*s.attack_frames/n:5.1f}%)")
    print(f"  jump:    {s.jump_frames:6d} ({100*s.jump_frames/n:5.1f}%)")
    print()
    print("=== Movement ===")
    print(f"  forward: {s.forwardmove_pos:6d} ({100*s.forwardmove_pos/n:5.1f}%)  back: {s.forwardmove_neg:6d} ({100*s.forwardmove_neg/n:5.1f}%)")
    print(f"  right:   {s.sidemove_pos:6d} ({100*s.sidemove_pos/n:5.1f}%)  left: {s.sidemove_neg:6d} ({100*s.sidemove_neg/n:5.1f}%)")
    print(f"  up:      {s.upmove_pos:6d} ({100*s.upmove_pos/n:5.1f}%)  down: {s.upmove_neg:6d} ({100*s.upmove_neg/n:5.1f}%)")
    print()
    print("=== Impulses ===")
    total_imp = sum(c for v, c in s.impulse_counts.items() if v != 0)
    print(f"  non-zero frames: {total_imp} ({100*total_imp/n:.1f}%)")
    print(f"  transitions: {s.impulse_changes}")
    if s.impulse_run_lengths:
        import numpy as np
        runs = np.array(s.impulse_run_lengths)
        print(f"  non-zero runs: {len(runs)}  mean={runs.mean():.1f}  median={int(np.median(runs))}  max={runs.max()}")
    print(f"  value distribution:")
    for v in sorted(s.impulse_counts):
        c = s.impulse_counts[v]
        label = {0: "none", 1: "axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
                 6: "GL", 7: "RL", 8: "LG", 9: "cheat", 10: "next_wpn",
                 12: "prev_wpn", 22: "hook"}.get(v, f"mod/{v}")
        print(f"    {v:3d} ({label:>8}): {c:6d} ({100*c/n:5.1f}%)")
