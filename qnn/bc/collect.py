#!/usr/bin/env python3
"""Collect BC training data by replaying demos through the demo worker.

Replays .dem files in parallel, reads obs buffer + action labels directly
from the demo worker, shards directly into train/val .npy files. No
intermediate per-episode staging — each worker accumulates rows and
flushes complete shards.

Supports resume via append-only done log.

Usage:
    python -m qnn.bc.collect \
        --demo-dir artifacts/corpus/qwd \
        --workers 30
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import numpy as np

from qnn import filter_dsl, engine_norm as en
from qnn.wire import (
    OBS_BUFFER_SIZE, ACTION_SIZE, TICK_HEADER_SIZE,
    FLAG_DONE,
    unpack_obs_buffer_native,
)
from qnn.actions import MOVE_AXIS_THRESHOLD, MOVE_CLASS_NEG, MOVE_CLASS_NONE, MOVE_CLASS_POS
from qnn.bc.target_labeler import (
    DEFAULT_LABELER_CONFIG,
    NO_TARGET_INDEX,
    TARGET_PROBS_CLASSES,
    label_enemy_target_probs,
)
from qnn.vocab import (
    TOKEN_ACTOR, TOKEN_PROJECTILE, TOKEN_ITEM, TOKEN_MOVER,
    MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS, ACTOR_SCALAR_DIM,
)


# Engine action struct: 16 bytes. Layout matches qnn_action_t on the C side.
#
#   offset 0  : move       u8   — press byte:
#                                  bit 0    = attack press
#                                  bits 1-2 = forward neg / pos
#                                  bits 3-4 = side neg / pos
#                                  bits 5-6 = up neg / pos
#                                  bit 7    = jump press (explicit)
#   offset 1  : weapon     u8   — raw engine weapon byte (0..8)
#   offset 2  : input_mask u8   — 8-bit per-axis feasibility, same layout as
#                                  the press byte. Packed by QNN_PackInputMask.
#   offset 3  : _pad       u8   — alignment for the trailing float[3]
#   offset 4..15 : look[3] f32  — view-relative look-delta unit vector
#
# Kept as a private record dtype since the new wire pipeline no longer
# routes through qnn.wire.ACTION_FIELDS (deleted with the legacy parser).
_ACTION_FIELDS_LAYOUT = {
    "move":       (0, np.uint8,   ()),
    "weapon":     (1, np.uint8,   ()),
    "input_mask": (2, np.uint8,   ()),
    "look":       (4, np.float32, (3,)),
}



TICK_TOTAL_SIZE = TICK_HEADER_SIZE + OBS_BUFFER_SIZE + ACTION_SIZE

# ── Filter spec (MongoDB-style query DSL) ───────────────────────────
#
# The filter is a JSON config file (--filter-config) with schema:
#
#   {
#     "demos":    { "keep": {<predicate>}, "drop": {<predicate>} },
#     "segments": { "drop": ["intermission", "paused", ...] },
#     "tokens":   { "keep": [<field>...], "drop": [<field>...] },
#     "actions":  { "keep": [<field>...], "drop": [<field>...] }
#   }
#
# demos.keep must match for a demo to be included; ANY demos.drop match
# excludes.  segments.drop names manifest label intervals (signon, dead,
# intermission) whose frames get masked out, with each surviving
# contiguous run becoming its own sub-episode.  tokens / actions
# keep/drop list obs / act field names (uses field-level whitelist or
# blacklist; mutually exclusive).  Every axis is optional; an empty
# config collects every demo, every frame, every field.
#
# Predicate dicts use MongoDB query syntax: each key is a field path,
# each value is either a bare scalar (implicit $eq) or an operator
# dict.  Logical operators ($and / $or / $not) take sub-predicates.
# Field paths can be top-level manifest fields or
# `labels.<name>.<aggregate>` where aggregate is one of:
#   coverage  — sum(intervals)/total_frames  (float 0..1)
#   count     — number of intervals          (int)
#   duration  — sum of interval widths       (int frames)
#   exists    — at least one interval        (bool)
#
# Unknown fields, unknown operators, malformed predicates all fail
# loud — never silent.  Per agents/conventions.md.

# Top-level scalar / list manifest fields the filter understands.
_VALID_MANIFEST_FIELDS = frozenset((
    "format", "source", "gamedir", "mode", "map", "hostname",
    "teamplay", "maxclients", "deathmatch", "total_frames",
    "recorder", "parse_error", "error_frame", "avg_ping_ms",
))

# Label-derived aggregates accessible as `labels.<name>.<agg>`.
_LABEL_AGGREGATES = frozenset(("coverage", "count", "duration", "exists"))

# Channels emitted by the classifier in the `active_input` dict
# (qw_classifier.c active_input_t).  Accessible via
# `active_input.<channel>` (raw frame count) or
# `active_input.<channel>.coverage` (count / total_frames).
_VALID_ACTIVE_INPUT_CHANNELS = frozenset((
    "forwardmove", "sidemove", "upmove",
    "pitch", "yaw", "roll",
    "attack", "jump", "use",
    "weaponswitch", "none",
))

# Channels emitted in the `active_state` dict (qw_classifier.c
# active_state_t) — per-frame inventory/state delta counters.
_VALID_ACTIVE_STATE_CHANNELS = frozenset((
    "health_up", "health_down",
    "armor_up",  "armor_down",
    "ammo_up",   "ammo_down",
    "frag_up", "weapon_up", "special_up",
))

# Label names the classifier emits — used to pre-populate label aggregates
# in the flattened entry so missing labels resolve to natural zero/false
# defaults (e.g., a demo without a `dead` interval gets
# `labels.dead.coverage = 0.0`).  Drift-resistant: every label-open call
# site in qw_classifier.c must appear here.
_KNOWN_LABEL_NAMES = frozenset((
    "dead", "intermission", "paused", "impossible_health", "signon", "match",
))

# Direct labels that can appear in `segments.drop`.
_VALID_TICK_DROP_LABELS = frozenset((
    "dead", "intermission", "paused", "impossible_health", "signon",
))


def _flatten_manifest_entry(entry: dict) -> dict[str, Any]:
    """Expand a manifest entry into a flat ``field_path -> value`` dict
    consumed by ``qnn.filter_dsl.eval_filter``.

    The field-path namespace matches the filter DSL spec:
        <top-level-field>           — _VALID_MANIFEST_FIELDS scalars
        labels.<name>.<aggregate>   — coverage / count / duration / exists
        active_input.<channel>      — raw frame count
        active_input.<channel>.coverage — count / total_frames
        active_state.<channel>[.coverage] — same shape

    Aggregates are computed on-the-fly from interval lists, mirroring
    the old `_resolve_field` semantics.  Known labels are pre-populated
    with zero defaults so a demo without (e.g.) a `dead` interval still
    has `labels.dead.coverage = 0.0` available to predicates. """
    flat: dict[str, Any] = {}
    for k in _VALID_MANIFEST_FIELDS:
        flat[k] = entry.get(k)
    total = int(entry.get("total_frames") or 0)
    # Pre-populate every known label's four aggregates with zero defaults.
    for name in _KNOWN_LABEL_NAMES:
        flat[f"labels.{name}.coverage"] = 0.0
        flat[f"labels.{name}.count"] = 0
        flat[f"labels.{name}.duration"] = 0
        flat[f"labels.{name}.exists"] = False
    labels = entry.get("labels") or {}
    for name, intervals in labels.items():
        intervals = intervals or []
        duration = sum(int(e) - int(s) for s, e in intervals)
        flat[f"labels.{name}.exists"] = bool(intervals)
        flat[f"labels.{name}.count"] = len(intervals)
        flat[f"labels.{name}.duration"] = duration
        flat[f"labels.{name}.coverage"] = (duration / total) if total > 0 else 0.0
    for block, valid in (("active_input", _VALID_ACTIVE_INPUT_CHANNELS),
                          ("active_state", _VALID_ACTIVE_STATE_CHANNELS)):
        block_data = entry.get(block) or {}
        for chan in valid:
            count = int(block_data.get(chan, 0))
            flat[f"{block}.{chan}"] = count
            flat[f"{block}.{chan}.coverage"] = (count / total) if total > 0 else 0.0
    return flat


def _is_valid_manifest_path(path: str) -> bool:
    """Path-shape validator for filter predicates over manifest entries."""
    parts = path.split(".")
    if len(parts) == 1:
        return parts[0] in _VALID_MANIFEST_FIELDS
    if parts[0] == "labels":
        return (len(parts) == 3
                and parts[1] in _KNOWN_LABEL_NAMES
                and parts[2] in _LABEL_AGGREGATES)
    if parts[0] == "active_input":
        if len(parts) == 2:
            return parts[1] in _VALID_ACTIVE_INPUT_CHANNELS
        if len(parts) == 3:
            return parts[1] in _VALID_ACTIVE_INPUT_CHANNELS and parts[2] == "coverage"
        return False
    if parts[0] == "active_state":
        if len(parts) == 2:
            return parts[1] in _VALID_ACTIVE_STATE_CHANNELS
        if len(parts) == 3:
            return parts[1] in _VALID_ACTIVE_STATE_CHANNELS and parts[2] == "coverage"
        return False
    return False


def _eval_filter(entry: dict, predicate: dict) -> bool:
    """Apply a filter predicate to a manifest entry; True = matches."""
    return bool(filter_dsl.eval_filter(_flatten_manifest_entry(entry), predicate))


# Native obs field names (engine_norm phase 2). Mirrors the keys produced
# by qnn.wire.unpack_obs_buffer_native — see that module for the
# authoritative list. Used to validate filter `tokens.{keep,drop}` lists.
_VALID_TOKEN_FIELDS = frozenset({
    # Self block (native widths per engine_norm.SELF_FIELDS).
    "health", "effective_armor",
    "ammo_shells", "ammo_nails", "ammo_rockets", "ammo_cells",
    "vel", "attack_finished",
    "self_weapon_id", "self_movement_id", "self_items",
    # Spatial block (per-field arrays).
    "spatial_dir", "spatial_nearest_dist", "spatial_mean_dist",
    "spatial_openness", "spatial_clearance", "spatial_traversable",
    "spatial_dropoff", "spatial_solid_frac", "spatial_water_frac",
    "spatial_slime_frac", "spatial_lava_frac",
    # Entity block (variable-length on disk, see _ShardWriter docstring).
    "entity_types", "entity_subject_id", "entity_modality_id",
    "entity_player_id", "entity_event_count",
    "entity_event_actions", "entity_event_sources",
    "entity_half_extents", "entity_rel", "entity_vel",
    "entity_path", "entity_path_dist", "entity_eta", "entity_recency",
    "entity_facing", "entity_team", "entity_score",
    "entity_amount", "entity_regen", "entity_state",
    "entity_count",
})
_VALID_ACTION_FIELDS = frozenset({"move", "look", "attack", "weapon", "target_probs"})


def _validate_filter_schema(spec: dict) -> None:
    """Validate filter config at startup; surfaces unknown fields,
    operators, label names, and channel names loudly before collection
    runs.  Per agents/conventions.md.

    Schema:
        {
          "demos":    { "keep": {<predicate>}, "drop": {<predicate>} },
          "segments": { "drop": ["signon", "dead", "intermission"] },
          "tokens":   { "keep": [<field>...], "drop": [<field>...] },
          "actions":  { "keep": [<field>...], "drop": [<field>...] }
        }

    Each axis is optional.  Within each axis, keep/drop are mutually
    exclusive for tokens and actions (whitelist vs blacklist).  demos
    accepts both since they compose (drop wins).
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"filter config must be a JSON object, got "
            f"{type(spec).__name__}"
        )
    allowed = {"demos", "segments", "tokens", "actions"}
    extras = set(spec.keys()) - allowed
    if extras:
        raise ValueError(
            f"unknown top-level keys in filter config: {sorted(extras)}.  "
            f"Allowed: {sorted(allowed)}"
        )

    # demos: predicate dicts (MongoDB DSL).
    demos = spec.get("demos") or {}
    if not isinstance(demos, dict):
        raise ValueError(f"'demos' must be an object, got {type(demos).__name__}")
    demos_allowed = {"keep", "drop"}
    bad = set(demos.keys()) - demos_allowed
    if bad:
        raise ValueError(
            f"unknown keys in 'demos': {sorted(bad)}.  "
            f"Allowed: {sorted(demos_allowed)}"
        )
    for which in ("keep", "drop"):
        sub = demos.get(which) or {}
        if not isinstance(sub, dict):
            raise ValueError(
                f"'demos.{which}' must be a predicate dict, "
                f"got {type(sub).__name__}"
            )
        filter_dsl.validate_predicate(sub, _is_valid_manifest_path)

    # segments: drop-only, list of named tick-label intervals.
    segments = spec.get("segments") or {}
    if not isinstance(segments, dict):
        raise ValueError(f"'segments' must be an object, got {type(segments).__name__}")
    segments_allowed = {"drop"}
    bad = set(segments.keys()) - segments_allowed
    if bad:
        raise ValueError(
            f"unknown keys in 'segments': {sorted(bad)}.  "
            f"Allowed: {sorted(segments_allowed)} (no whitelist — "
            f"omit to drop nothing)"
        )
    seg_drop = segments.get("drop") or []
    if not isinstance(seg_drop, list):
        raise ValueError("'segments.drop' must be a list of label names")
    bad = [n for n in seg_drop if n not in _VALID_TICK_DROP_LABELS]
    if bad:
        raise ValueError(
            f"unknown 'segments.drop' labels: {bad}.  Valid: "
            f"{sorted(_VALID_TICK_DROP_LABELS)}"
        )

    # tokens / actions: keep XOR drop, lists of valid field names.
    for axis, valid in (("tokens", _VALID_TOKEN_FIELDS),
                         ("actions", _VALID_ACTION_FIELDS)):
        block = spec.get(axis) or {}
        if not isinstance(block, dict):
            raise ValueError(f"{axis!r} must be an object, got {type(block).__name__}")
        block_allowed = {"keep", "drop"}
        bad = set(block.keys()) - block_allowed
        if bad:
            raise ValueError(
                f"unknown keys in {axis!r}: {sorted(bad)}.  "
                f"Allowed: {sorted(block_allowed)}"
            )
        for which in ("keep", "drop"):
            names = block.get(which) or []
            if not isinstance(names, list):
                raise ValueError(
                    f"{axis!r}.{which!r} must be a list of field names"
                )
            bad = [n for n in names if n not in valid]
            if bad:
                raise ValueError(
                    f"unknown {axis!r}.{which!r}: {bad}.  Valid: "
                    f"{sorted(valid)}"
                )
        keep = block.get("keep") or []
        drop = block.get("drop") or []
        if keep and drop:
            raise ValueError(
                f"{axis!r} cannot set both keep and drop "
                f"(pick one — keep is whitelist, drop is blacklist)"
            )


def _load_and_pin_filter(output: Path, cli_path: Path | None) -> dict:
    """Establish the canonical filter for this collection.

    The filter is part of the collection's on-disk identity: it always
    lives at ``<output>/filter.json``. The CLI ``--filter-config`` flag
    seeds it on a fresh collect; a re-run finds it in the cache dir
    automatically with no flag needed. Mismatch between an existing
    pinned filter and a CLI override fails loud.

    Behavior:
      - Neither exists: write ``{}`` to ``<output>/filter.json``
        (explicit "no filter" record) and return an empty spec.
      - Only CLI path: load, validate, write canonical JSON to
        ``<output>/filter.json``.
      - Only pinned: load, validate, return as-is.
      - Both: fail loud if they disagree; otherwise use the pinned copy.
    """
    output.mkdir(parents=True, exist_ok=True)
    pinned_path = output / "filter.json"
    cli_spec: dict | None = None
    if cli_path is not None:
        try:
            cli_spec = json.loads(cli_path.read_text())
            _validate_filter_schema(cli_spec)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            raise SystemExit(f"--filter-config {str(cli_path)!r}: {exc}")
    if pinned_path.exists():
        try:
            pinned_spec = json.loads(pinned_path.read_text())
            _validate_filter_schema(pinned_spec)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            raise SystemExit(f"{pinned_path}: {exc}")
        if cli_spec is not None and cli_spec != pinned_spec:
            raise SystemExit(
                f"--filter-config {str(cli_path)!r} differs from pinned "
                f"filter at {pinned_path}. To change the filter for this "
                f"output, delete the pinned file or use a different --output."
            )
        return pinned_spec
    spec = cli_spec if cli_spec is not None else {}
    pinned_path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def _apply_field_filter(
    arrays: dict[str, np.ndarray],
    keep: tuple[str, ...] | list[str],
    drop: tuple[str, ...] | list[str],
) -> None:
    """Restrict ``arrays`` in place to the configured field set.  ``keep``
    wins if non-empty (whitelist).  Otherwise ``drop`` excludes the listed
    keys (blacklist).  Empty/None on both → no-op (default, all fields
    survive). """
    if keep:
        keep_set = frozenset(keep)
        for k in list(arrays.keys()):
            if k not in keep_set:
                del arrays[k]
    elif drop:
        drop_set = frozenset(drop)
        for k in list(arrays.keys()):
            if k in drop_set:
                del arrays[k]


def _label_keep_mask(labels: dict,
                      drop_label_names: tuple[str, ...],
                      total_frames: int,
                      n_emitted: int) -> np.ndarray:
    """Per-emitted-tick boolean keep mask composed from caller-named
    label-interval kinds.  A tick is kept iff every named label says
    keep.  Empty list → all-True mask.

    The classifier records intervals in classifier-frame indices in
    [0, total_frames).  Emitted ticks come from the C worker after
    optional resampling.  We map each drop interval into emitted-tick
    space by linear scaling — the mapping is approximate (the worker
    may skip engine-filtered frames), bounded only by total_frames. """
    if n_emitted <= 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(n_emitted, dtype=bool)
    if total_frames <= 0 or not labels or not drop_label_names:
        return keep
    for name in drop_label_names:
        for s, e in labels.get(name) or []:
            i0 = int(int(s) * n_emitted / total_frames)
            i1 = int(int(e) * n_emitted / total_frames)
            i0 = max(0, min(n_emitted, i0))
            i1 = max(0, min(n_emitted, i1))
            if i1 > i0:
                keep[i0:i1] = False
    return keep


def _runs_from_mask(keep: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True runs in a boolean mask.

    Returns a list of [start, end) half-open intervals.  Used by
    `_unpack_episode` to split a single demo into one sub-episode per
    surviving run of frames after `segments.drop` carves intervals
    out, so each run carries causal continuity through the trainer's
    GRU hidden state rather than splicing across dropped frames. """
    if keep.size == 0 or not keep.any():
        return []
    padded = np.concatenate(([False], keep, [False])).astype(np.int8)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))
_WORKER_DEMO_PROC: subprocess.Popen | None = None
_WORKER_DEMO_ARGS: tuple[str, str, int, str] | None = None
# Collect-time tokens/actions field filter, set per-run via
# _init_collect_worker.  Tuple positions: (tokens_keep, tokens_drop,
# actions_keep, actions_drop).  Within each axis, keep wins if non-empty
# (whitelist), else drop blacklists, else all fields pass through.  Read
# by _unpack_episode right after compaction, before episodes get handed
# to the shard writer.
_WORKER_FIELD_FILTER: tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
] = ((), (), (), ())
_WORKER_DEMO_STDERR_FILE = None
_WORKER_DEMO_STDERR_PATH: Path | None = None
_ACTION_RECORD_DTYPE = np.dtype(
    {
        "names": list(_ACTION_FIELDS_LAYOUT.keys()),
        "formats": [dtype if not shape else (dtype, shape) for _, dtype, shape in _ACTION_FIELDS_LAYOUT.values()],
        "offsets": [offset for offset, _, _ in _ACTION_FIELDS_LAYOUT.values()],
        "itemsize": ACTION_SIZE,
    }
)


# ── Worker subprocess management ─────────────────────────────────────

def _cleanup_worker_stderr() -> None:
    global _WORKER_DEMO_STDERR_FILE, _WORKER_DEMO_STDERR_PATH
    if _WORKER_DEMO_STDERR_FILE is not None:
        try:
            _WORKER_DEMO_STDERR_FILE.close()
        except Exception:
            pass
        _WORKER_DEMO_STDERR_FILE = None
    if _WORKER_DEMO_STDERR_PATH is not None:
        try:
            _WORKER_DEMO_STDERR_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        _WORKER_DEMO_STDERR_PATH = None


def _open_worker_stderr() -> tuple[object, Path]:
    fd, path = tempfile.mkstemp(prefix=f"qnn_collect_worker_{os.getpid()}_", suffix=".stderr")
    return os.fdopen(fd, "w+b"), Path(path)


def _start_worker(demo_worker: str, asset_root: str, tick_hz: int, game_dir: str) -> subprocess.Popen:
    global _WORKER_DEMO_STDERR_FILE, _WORKER_DEMO_STDERR_PATH
    env = {**os.environ, "QUAKE_BASEDIR": str(Path(asset_root).resolve())}
    _cleanup_worker_stderr()
    stderr_file, stderr_path = _open_worker_stderr()
    _WORKER_DEMO_STDERR_FILE = stderr_file
    _WORKER_DEMO_STDERR_PATH = stderr_path
    proc = subprocess.Popen(
        [demo_worker, "-game", game_dir],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file,
        env=env,
    )
    hello = json.dumps({"op": "hello", "map_id": "start", "tick_hz": tick_hz}) + "\n"
    proc.stdin.write(hello.encode())
    proc.stdin.flush()
    resp = proc.stdout.readline()
    if not resp or b'"ok":true' not in resp:
        err = _worker_stderr_tail(proc, limit=500)
        _cleanup_worker_stderr()
        raise RuntimeError(f"Demo worker hello failed: {err}")
    return proc


def _shutdown_worker() -> None:
    global _WORKER_DEMO_PROC
    proc = _WORKER_DEMO_PROC
    if proc is None:
        return
    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write(json.dumps({"op": "shutdown"}).encode() + b"\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _WORKER_DEMO_PROC = None
        _cleanup_worker_stderr()


def _init_collect_worker(
    demo_worker: str, asset_root: str, tick_hz: int, game_dir: str,
    tokens_keep:  tuple[str, ...] = (),
    tokens_drop:  tuple[str, ...] = (),
    actions_keep: tuple[str, ...] = (),
    actions_drop: tuple[str, ...] = (),
) -> None:
    global _WORKER_DEMO_ARGS, _WORKER_FIELD_FILTER
    _WORKER_DEMO_ARGS = (demo_worker, asset_root, int(tick_hz), game_dir)
    _WORKER_FIELD_FILTER = (
        tuple(tokens_keep),  tuple(tokens_drop),
        tuple(actions_keep), tuple(actions_drop),
    )
    _shutdown_worker()
    atexit.register(_shutdown_worker)


def _get_collect_worker() -> subprocess.Popen:
    global _WORKER_DEMO_PROC
    if _WORKER_DEMO_ARGS is None:
        raise RuntimeError("collect worker not initialized")
    if _WORKER_DEMO_PROC is not None and _WORKER_DEMO_PROC.poll() is None:
        return _WORKER_DEMO_PROC
    demo_worker, asset_root, tick_hz, game_dir = _WORKER_DEMO_ARGS
    _WORKER_DEMO_PROC = _start_worker(demo_worker, asset_root, tick_hz, game_dir)
    return _WORKER_DEMO_PROC


# ── Demo playback ────────────────────────────────────────────────────

def _read_collect_frames(
    proc: subprocess.Popen,
    op: dict,
    magic: bytes,
    frame_after_magic: int,
    parse_frame,
) -> list[dict] | None:
    """Dispatch ``op`` to the worker and read its framed output stream.
    ``parse_frame`` takes the bytes after the magic and returns a tick
    dict that must include ``"done": bool``.  Returns the list of ticks
    on success, ``None`` on worker death (the caller fetches stderr
    tail and decides whether it's a watchdog stall or a crash).

    Shared by the BC (QOBS) and labeler (LOBS) wire formats.  Each
    caller supplies its own magic bytes, frame size, and parse
    callback; the op-dispatch / error-line intercept / bad-magic guard /
    partial-read termination logic is identical and lives here. """
    proc.stdin.write((json.dumps(op) + "\n").encode())
    proc.stdin.flush()
    ticks: list[dict] = []
    while True:
        if proc.poll() is not None:
            return None
        m = proc.stdout.read(4)
        if not m or len(m) < 4:
            return None
        if m[0:1] == b'{':
            rest = proc.stdout.readline()
            try:
                err = json.loads(m + rest)
                raise RuntimeError(err.get("error", "unknown error"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(f"Worker error: {(m + rest)[:200]!r}")
        if m != magic:
            raise RuntimeError(f"Bad magic: {m!r}")
        raw = proc.stdout.read(frame_after_magic) if proc.poll() is None else b""
        if len(raw) < frame_after_magic:
            return None
        tick = parse_frame(raw)
        ticks.append(tick)
        if tick.get("done"):
            break
    return ticks


def _parse_qobs_frame(raw: bytes) -> dict:
    """Parse one QOBS frame (after the 4-byte magic).  Returns a tick
    dict with separate obs / action byte-views and a done flag. """
    header = struct.unpack_from("<IIIHH", raw)
    flags = header[3]
    return {
        "obs":    raw[TICK_HEADER_SIZE:TICK_HEADER_SIZE + OBS_BUFFER_SIZE],
        "action": raw[TICK_HEADER_SIZE + OBS_BUFFER_SIZE:],
        "done":   bool(flags & FLAG_DONE),
    }


def _collect_one_demo(proc: subprocess.Popen, demo_name: str,
                      play_start: int = 0, play_end: int = 999999999,
                      force_mvd_emit: bool = False,
                      ) -> list[dict] | None:
    op = {
        "op": "collect", "demo_path": demo_name, "seed": 0,
        "play_start": play_start, "play_end": play_end,
        "force_mvd_emit": 1 if force_mvd_emit else 0,
    }
    return _read_collect_frames(
        proc, op, b"QOBS", TICK_TOTAL_SIZE, _parse_qobs_frame
    )


def _worker_stderr_tail(proc: subprocess.Popen | None, limit: int = 4000) -> str:
    """Return a worker's stderr tail without back-pressuring the worker."""
    if proc is None or _WORKER_DEMO_STDERR_PATH is None:
        return ""
    try:
        if _WORKER_DEMO_STDERR_FILE is not None:
            _WORKER_DEMO_STDERR_FILE.flush()
        with open(_WORKER_DEMO_STDERR_PATH, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit), os.SEEK_SET)
            data = f.read()
    except Exception:
        return ""
    if not data:
        return ""
    text = data.decode(errors="replace").strip()
    return text[-limit:]


# ── Labeler-input densifier (native → legacy normalized) ────────────
#
# The native obs format stores per-field entity arrays at native widths
# concatenated across all tokens in an episode. The target labeler
# (qnn.bc.target_labeler) was written against the legacy dense layout —
# (T, 16, 19) float32 entity_scalars_raw, (T, 16, 3) int32 entity_ids,
# (T, 16) int32 entity_types, (T, 17) float32 self_scalars — and reads
# values at fixed idx offsets matching qnn_onnx.c:emit_actor (see
# _ACTOR_*_OFFSET constants in target_labeler.py).
#
# We construct that dense view here from per-frame native arrays just
# for the labeler call. The result is discarded after target_probs is
# computed; only the native arrays go to disk.

# Legacy actor scalar idx layout (mirrors qnn.model.dequant entity idx
# comments — kept in sync there):
#   [hx,hy,hz, rx,ry,rz, dist, vx,vy,vz, px,py,pz, pd, eta, fac,team,score, rec]
#                          ^ idx 6 = dist, recomputed from rel
_ACTOR_IDX_HALFEXT = 0
_ACTOR_IDX_REL     = 3
_ACTOR_IDX_DIST    = 6
_ACTOR_IDX_VEL     = 7
_ACTOR_IDX_PATH    = 10
_ACTOR_IDX_PD      = 13
_ACTOR_IDX_ETA     = 14
_ACTOR_IDX_FACING  = 15
_ACTOR_IDX_TEAM    = 16
_ACTOR_IDX_SCORE   = 17
_ACTOR_IDX_RECENCY = 18

# Self scalars (T, 17) legacy idx layout matches qnn.model.dequant
# SelfDequantizer _IDX_* constants — kept in sync with that module.
_SELF_IDX_HEALTH      = 0
_SELF_IDX_ARMOR       = 1
_SELF_IDX_WEAPON_SG   = 2
_SELF_IDX_WEAPON_SSG  = 3
_SELF_IDX_WEAPON_NG   = 4
_SELF_IDX_WEAPON_SNG  = 5
_SELF_IDX_WEAPON_GL   = 6
_SELF_IDX_WEAPON_RL   = 7
_SELF_IDX_WEAPON_LG   = 8
_SELF_IDX_AMMO_SH     = 9
_SELF_IDX_AMMO_NA     = 10
_SELF_IDX_AMMO_RK     = 11
_SELF_IDX_AMMO_CE     = 12
_SELF_IDX_VEL         = 13   # 13..15
_SELF_IDX_ATTACK_FIN  = 16


def _densify_native_obs_for_labeler(
    native_obs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reconstruct the legacy normalized-float dense view the labeler
    expects from a per-tick stack of native obs fields.

    Input ``native_obs`` carries the per-tick native arrays after
    stacking across the episode:
      - Self/spatial fields: ``(T, ...)`` at native dtypes.
      - Entity fields: ``(T, ...)`` at the **per-tick variable-length**
        shape — entity_types ``(T, N_t)``, entity_rel ``(T, N_t, 3)``,
        etc. (Variable across t — handled by the caller stacking with
        np.array(..., dtype=object) or by passing a list of dicts; here
        we accept the per-tick concatenation produced by
        ``_native_obs_to_dense_arrays``.)

    Returns ``{entity_types, entity_ids, entity_scalars_raw,
    self_scalars}`` — exactly the keys
    ``label_enemy_target_probs`` reads from. Other native fields
    are dropped (not needed by the labeler).
    """
    T = native_obs["health"].shape[0]
    N = MAX_TOKEN_OBJECTS  # legacy 16-idx dense layout

    # ---- Self scalars (T, 17) ----
    self_scalars = np.zeros((T, 17), dtype=np.float32)
    self_scalars[:, _SELF_IDX_HEALTH]     = native_obs["health"]          .astype(np.float32) / en.MAX_HEALTH
    self_scalars[:, _SELF_IDX_ARMOR]      = native_obs["effective_armor"] .astype(np.float32) / en.MAX_ARMOR_EFFECT
    self_scalars[:, _SELF_IDX_AMMO_SH]    = native_obs["ammo_shells"]     .astype(np.float32) / en.MAX_SHELLS
    self_scalars[:, _SELF_IDX_AMMO_NA]    = native_obs["ammo_nails"]      .astype(np.float32) / en.MAX_NAILS
    self_scalars[:, _SELF_IDX_AMMO_RK]    = native_obs["ammo_rockets"]    .astype(np.float32) / en.MAX_ROCKETS
    self_scalars[:, _SELF_IDX_AMMO_CE]    = native_obs["ammo_cells"]      .astype(np.float32) / en.MAX_CELLS
    self_scalars[:, _SELF_IDX_VEL:_SELF_IDX_VEL + 3] = (
        native_obs["vel"].astype(np.float32) / en.MAX_VELOCITY
    )
    self_scalars[:, _SELF_IDX_ATTACK_FIN] = native_obs["attack_finished"] .astype(np.float32) / en.TIME_SCALE
    # cl.items bit-extracted weapon flags (indices 2..8).
    items_i64 = native_obs["self_items"].astype(np.int64)
    self_scalars[:, _SELF_IDX_WEAPON_SG ] = ((items_i64 & en.IT_SHOTGUN)          != 0).astype(np.float32)
    self_scalars[:, _SELF_IDX_WEAPON_SSG] = ((items_i64 & en.IT_SUPER_SHOTGUN)    != 0).astype(np.float32)
    self_scalars[:, _SELF_IDX_WEAPON_NG ] = ((items_i64 & en.IT_NAILGUN)          != 0).astype(np.float32)
    self_scalars[:, _SELF_IDX_WEAPON_SNG] = ((items_i64 & en.IT_SUPER_NAILGUN)    != 0).astype(np.float32)
    self_scalars[:, _SELF_IDX_WEAPON_GL ] = ((items_i64 & en.IT_GRENADE_LAUNCHER) != 0).astype(np.float32)
    self_scalars[:, _SELF_IDX_WEAPON_RL ] = ((items_i64 & en.IT_ROCKET_LAUNCHER)  != 0).astype(np.float32)
    self_scalars[:, _SELF_IDX_WEAPON_LG ] = ((items_i64 & en.IT_LIGHTNING)        != 0).astype(np.float32)

    # ---- Entity dense (T, 16, ...) ----
    # Native entity fields are stored per-tick as 1D Python lists in
    # ``native_obs["_per_tick_entities"]`` — a list of length T where
    # each item is the per-tick dict from unpack_obs_buffer_native.
    # Fast path: zero out and fill the actor indices used by the labeler.
    entity_types        = np.full((T, N),  -1, dtype=np.int32)
    entity_ids          = np.zeros((T, N, 3),   dtype=np.int32)
    entity_scalars_raw  = np.zeros((T, N, ACTOR_SCALAR_DIM), dtype=np.float32)

    per_tick = native_obs.get("_per_tick_entities")
    if per_tick is None:
        # Caller forgot to attach. Empty fallback so the labeler sees
        # "no targets" rather than crash, but this is an error path.
        raise RuntimeError(
            "_densify_native_obs_for_labeler requires "
            "'_per_tick_entities' (list of per-tick parsed entity dicts)"
        )

    for t, tick_ents in enumerate(per_tick):
        n_tokens = int(tick_ents["entity_count"])
        if n_tokens <= 0:
            continue
        # Hard cap at N=16 — legacy dense layout has fixed idx count;
        # native worker emits ≤ N anyway, but be defensive.
        nt = min(n_tokens, N)
        types_t = tick_ents["entity_types"][:nt]
        entity_types[t, :nt] = types_t.astype(np.int32)
        entity_ids[t, :nt, 0] = tick_ents["entity_subject_id"][:nt].astype(np.int32)
        entity_ids[t, :nt, 1] = tick_ents["entity_modality_id"][:nt].astype(np.int32)
        entity_ids[t, :nt, 2] = tick_ents["entity_player_id"][:nt].astype(np.int32)

        # Common normalized scalars per actor idx. Tokens of other
        # types still get rel/recency filled (the labeler treats them
        # via mask later), but only actors contribute to target.
        # half_ext
        entity_scalars_raw[t, :nt, _ACTOR_IDX_HALFEXT:_ACTOR_IDX_HALFEXT + 3] = (
            tick_ents["entity_half_extents"][:nt].astype(np.float32) / en.DIST_SCALE
        )
        # rel + derived dist
        rel = tick_ents["entity_rel"][:nt].astype(np.float32) / en.DIST_SCALE
        entity_scalars_raw[t, :nt, _ACTOR_IDX_REL:_ACTOR_IDX_REL + 3] = rel
        entity_scalars_raw[t, :nt, _ACTOR_IDX_DIST] = np.linalg.norm(rel, axis=-1)
        # vel (proj+actor only have vel; item/mover have zero — safe to write)
        entity_scalars_raw[t, :nt, _ACTOR_IDX_VEL:_ACTOR_IDX_VEL + 3] = (
            tick_ents["entity_vel"][:nt].astype(np.float32) / en.MAX_VELOCITY
        )
        # path / path_dist / eta — proj has none on the wire; native
        # parser returns zeros for those indices, which is the right
        # fallback for the dense layout.
        entity_scalars_raw[t, :nt, _ACTOR_IDX_PATH:_ACTOR_IDX_PATH + 3] = (
            tick_ents["entity_path"][:nt].astype(np.float32) / en.DIST_SCALE
        )
        entity_scalars_raw[t, :nt, _ACTOR_IDX_PD] = (
            tick_ents["entity_path_dist"][:nt].astype(np.float32) / en.DIST_SCALE
        )
        entity_scalars_raw[t, :nt, _ACTOR_IDX_ETA] = (
            tick_ents["entity_eta"][:nt].astype(np.float32) / en.TIME_SCALE
        )
        # Actor-only fields (facing/team/score) — set where applicable.
        actor_mask = types_t == TOKEN_ACTOR
        if actor_mask.any():
            entity_scalars_raw[t, :nt, _ACTOR_IDX_FACING] = np.where(
                actor_mask,
                tick_ents["entity_facing"][:nt].astype(np.float32) / 255.0,
                0.0,
            )
            entity_scalars_raw[t, :nt, _ACTOR_IDX_TEAM] = np.where(
                actor_mask,
                tick_ents["entity_team"][:nt].astype(np.float32),
                0.0,
            )
            entity_scalars_raw[t, :nt, _ACTOR_IDX_SCORE] = np.where(
                actor_mask,
                tick_ents["entity_score"][:nt].astype(np.float32) / 255.0,
                0.0,
            )
        # Recency is present for every token type.
        entity_scalars_raw[t, :nt, _ACTOR_IDX_RECENCY] = (
            tick_ents["entity_recency"][:nt].astype(np.float32) / en.TIME_SCALE
        )

    return {
        "entity_types":       entity_types,
        "entity_ids":         entity_ids,
        "entity_scalars_raw": entity_scalars_raw,
        "self_scalars":       self_scalars,
    }


# target_probs was previously encoded as sparse (T, 3) u8 here and
# expanded back at training time. Both the encoder and decoder have
# been removed: target_probs is now cached as a dense per-shard sidecar
# written by collect and consumed directly by BC training.


def _compact_action_arrays(act_arrays: dict[str, np.ndarray]) -> None:
    """Convert raw wire labels to the compact on-disk format.

    Mutates `act_arrays` in place.  See qnn.actions docstring for the spec.

      move          uint8[T]       — press byte from the engine; same layout
                                      as input_mask. Passed through; load
                                      time unpackers in qnn.bc.train rebuild
                                      the per-axis class streams.
      look          float32[T, 3]  → float16[T, 3].
      weapon        uint8[T]       — raw engine weapon byte (0..8): 0 = no
                                      weapon held, 1..8 = Quake weapon id
                                      (axe..thunderbolt). No-weapon frames
                                      are kept on disk so downstream signals
                                      still train; the model masks them out
                                      of the weapon-head CE loss via
                                      ignore_index.
      input_mask    uint8[T]       — engine-act feasibility byte (same layout
                                      as the press byte).
      target_probs   float32[T, 17] → uint8[T, 3] sparse
                                      ``(idx, idx2, w2_u8)`` per engine_norm.
                                      ActionDequantizer expands at the model
                                      boundary; the dataloader expands at
                                      load time for the legacy float code
                                      path. See _encode_sparse_target_probs.

    The engine-facing `switch` idx is derived from the weapon head's
    argmax at inference time (model.policy._weapon_switch_indices_from_choices);
    no `switch` array is written to disk — it would be redundant since
    the idx is a pure function of the weapon class.
    """
    move = act_arrays.get("move")
    if move is not None and not (
        isinstance(move, np.ndarray) and move.dtype == np.uint8
    ):
        raise ValueError(
            f"action 'move' must be uint8 on the wire, got dtype "
            f"{getattr(move, 'dtype', type(move))}"
        )

    look = act_arrays.get("look")
    if look is not None and look.dtype != np.float16:
        act_arrays["look"] = np.ascontiguousarray(look, dtype=np.float16)

    weapon = act_arrays.get("weapon")
    if weapon is not None:
        weapon_arr = np.asarray(weapon)
        if weapon_arr.dtype != np.uint8:
            raise ValueError(
                f"action 'weapon' must be uint8 on the wire, got dtype {weapon_arr.dtype}"
            )
        invalid = weapon_arr[weapon_arr > 8]
        if invalid.size:
            sample = sorted({int(v) for v in invalid[:8]})
            raise ValueError(
                f"weapon bytes must be in 0..8 (0=no weapon, 1..8=axe..LG), "
                f"got {sample}"
            )
        act_arrays["weapon"] = np.ascontiguousarray(weapon_arr)

    # input_mask: uint8 on the wire (bit layout shared with the press
    # byte; packed by QNN_PackInputMask).
    input_mask = act_arrays.get("input_mask")
    if input_mask is not None:
        im_arr = np.asarray(input_mask)
        if im_arr.dtype != np.uint8:
            raise ValueError(
                f"action 'input_mask' must be uint8 on the wire, got dtype {im_arr.dtype}"
            )
        act_arrays["input_mask"] = np.ascontiguousarray(im_arr)

    # target_probs: cast to f16 for on-disk caching, or drop if the
    # caller hasn't computed it. The labeler is ~95% of BC load time
    # when run live; caching here amortizes it across every subsequent
    # training launch. f16 max quantization error measured at 2.4e-4
    # on real labeler output (values in [0, 1], min nonzero ~7e-3);
    # well below noise. Train-time loader auto-detects the cache file
    # via shard manifest and falls back to live labeler if absent.
    td = act_arrays.get("target_probs")
    if td is not None and td.dtype != np.float16:
        act_arrays["target_probs"] = np.ascontiguousarray(td, dtype=np.float16)


# Field categories for native obs handling. Self/spatial are fixed-
# shape per frame and stack along axis 0 like the legacy code path.
# Entity fields carry the per-frame variable token count; they
# concatenate along axis 0 across ticks and the on-disk shape becomes
# (total_tokens_in_shard, ...).
_NATIVE_SELF_FIELDS = (
    "health", "effective_armor",
    "ammo_shells", "ammo_nails", "ammo_rockets", "ammo_cells",
    "vel", "attack_finished",
    "self_weapon_id", "self_movement_id", "self_items",
    "view_pitch",
)
_NATIVE_SPATIAL_FIELDS = (
    "spatial_dir",
    "spatial_nearest_dist", "spatial_mean_dist",
    "spatial_openness", "spatial_clearance", "spatial_traversable",
    "spatial_dropoff", "spatial_solid_frac", "spatial_water_frac",
    "spatial_slime_frac", "spatial_lava_frac",
)
_NATIVE_ENTITY_FIELDS = (
    "entity_types", "entity_subject_id", "entity_modality_id",
    "entity_player_id", "entity_event_count",
    "entity_event_actions", "entity_event_sources",
    "entity_half_extents", "entity_rel", "entity_vel",
    "entity_path", "entity_path_dist", "entity_eta", "entity_recency",
    "entity_facing", "entity_team", "entity_score",
    "entity_amount", "entity_regen", "entity_state",
)


def _unpack_episode(
    ticks: list[dict],
    combat_only: bool = False,
    labels: dict | None = None,
    drop_label_names: tuple[str, ...] = (),
    total_frames: int = 0,
    sight_only: bool = False,
    target_probs_cache: bool = True,
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Unpack raw ticks into one or more (obs, action) sub-episodes.

    Native wire format (engine_norm phase 2). Per-frame self/spatial
    fields stack at native dtypes; entity fields are stored as
    concatenated per-token arrays plus a per-row ``entity_count``
    array so the loader can recover slice boundaries via cumsum without
    on-disk padding.

    When `segments.drop` carves intervals out of the demo, each
    surviving contiguous run of frames is emitted as its own sub-
    episode — never concatenated across dropped intervals — so the
    trainer can reset its GRU hidden state at each sub-episode
    boundary instead of carrying recurrent context across a gap.
    Returns sub-episodes in source-frame order (deterministic by demo).

    Empty list means the demo contributes no rows (all frames dropped
    or `combat_only` removed everything). """
    n = len(ticks)
    if n == 0:
        return []

    # Parse each tick into per-field numpy arrays. self / spatial are
    # fixed-shape; entity fields are (n_tokens_t, ...) where n_tokens_t
    # is variable per tick. Keep the per-tick dicts around for the
    # labeler densifier (it needs (T, 16, 19) dense entity scalars).
    per_tick: list[dict[str, np.ndarray]] = [
        unpack_obs_buffer_native(t["obs"]) for t in ticks
    ]

    obs_arrays: dict[str, np.ndarray] = {}
    # Fixed-shape self / spatial: stack along the new T axis.
    for key in _NATIVE_SELF_FIELDS + _NATIVE_SPATIAL_FIELDS:
        obs_arrays[key] = np.stack([tick[key] for tick in per_tick], axis=0)
    # Variable-length entity fields: concatenate along the existing
    # first axis (each per-tick array is already (n_tokens_t, ...)).
    entity_counts = np.array(
        [int(tick["entity_count"]) for tick in per_tick],
        dtype=np.uint8,
    )
    obs_arrays["entity_count"] = entity_counts
    if entity_counts.sum() == 0:
        # No tokens anywhere — emit zero-length arrays with the right
        # shape so downstream concat / slice paths stay typed.
        for key in _NATIVE_ENTITY_FIELDS:
            tmpl = per_tick[0][key]
            obs_arrays[key] = np.zeros((0,) + tmpl.shape[1:], dtype=tmpl.dtype)
    else:
        for key in _NATIVE_ENTITY_FIELDS:
            obs_arrays[key] = np.concatenate(
                [tick[key] for tick in per_tick], axis=0,
            )

    # Action block: decode the per-tick 32-byte action struct into a
    # contiguous record array, then split into per-field arrays at
    # native (float32 / int32) widths. _compact_action_arrays
    # downcasts later.
    action_blob = b"".join(t["action"] for t in ticks)
    records = np.frombuffer(action_blob, dtype=_ACTION_RECORD_DTYPE, count=n)
    act_arrays: dict[str, np.ndarray] = {}
    for name, (_, _, shape) in _ACTION_FIELDS_LAYOUT.items():
        field = np.asarray(records[name])
        act_arrays[name] = field.copy().reshape(n, *shape) if shape else field.copy()

    # Label-derived keep mask: composed from caller-named drop labels,
    # each mapped from classifier-frame intervals (in [0, total_frames))
    # into the emitted-tick range by linear scaling.  Each surviving
    # contiguous run of frames becomes its own sub-episode so the
    # trainer gets a clean GRU reset between runs.  An empty
    # drop_label_names yields a single full-length run.
    if labels and drop_label_names:
        keep = _label_keep_mask(labels, drop_label_names, total_frames, n)
        runs = _runs_from_mask(keep)
    else:
        runs = [(0, n)]
    if not runs:
        return []

    episodes: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for s, e in runs:
        # Self / spatial / entity_count slice cleanly along axis 0.
        sub_obs: dict[str, np.ndarray] = {}
        for key in _NATIVE_SELF_FIELDS + _NATIVE_SPATIAL_FIELDS:
            sub_obs[key] = obs_arrays[key][s:e]
        sub_counts = obs_arrays["entity_count"][s:e]
        sub_obs["entity_count"] = sub_counts
        # Entity fields use a token-axis slice. Compute the [start_tok,
        # end_tok) range on the concatenated arrays from cumsum.
        cs = np.cumsum(obs_arrays["entity_count"].astype(np.int64))
        tok_start = int(cs[s - 1]) if s > 0 else 0
        tok_end   = int(cs[e - 1]) if e > 0 else 0
        for key in _NATIVE_ENTITY_FIELDS:
            sub_obs[key] = obs_arrays[key][tok_start:tok_end]
        sub_per_tick = per_tick[s:e]

        sub_act = {key: values[s:e] for key, values in act_arrays.items()}

        # Labeler runs at collect time when either:
        #   - combat_only=True: needed to derive the `present` mask for
        #     non-engagement frame filtering below.
        #   - target_probs_cache=True: result is written to disk as an
        #     extra action array so BC training can mmap it instead of
        #     recomputing on every load (~95% of load time saved).
        if combat_only or target_probs_cache:
            labeler_view = _densify_native_obs_for_labeler({
                **{k: sub_obs[k] for k in _NATIVE_SELF_FIELDS},
                "_per_tick_entities": sub_per_tick,
            })
            sub_act["target_probs"] = label_enemy_target_probs(
                labeler_view, sub_act, config=DEFAULT_LABELER_CONFIG,
                sight_only=sight_only)
            if combat_only:
                present = 1.0 - sub_act["target_probs"][:, NO_TARGET_INDEX]
                keep2 = present >= 0.25
                if not bool(np.any(keep2)):
                    continue
                if not bool(np.all(keep2)):
                    sub_obs, sub_counts = _apply_frame_mask(sub_obs, keep2)
                    sub_act = {key: values[keep2] for key, values in sub_act.items()}

        _compact_action_arrays(sub_act)
        # No obs compaction step: the native parser already returns
        # arrays at the on-disk dtype. Just enforce dtype/shape
        # contracts via _validate_native_obs to catch drift early.
        _validate_native_obs(sub_obs)
        # Field filter (set per-run by _init_collect_worker).  Default is
        # no-op — empty keep/drop means every field passes through.  Same
        # keep/drop shape as the demos predicate, applied to dict keys.
        tokens_keep, tokens_drop, actions_keep, actions_drop = _WORKER_FIELD_FILTER
        _apply_field_filter(sub_obs, tokens_keep,  tokens_drop)
        _apply_field_filter(sub_act, actions_keep, actions_drop)
        episodes.append((sub_obs, sub_act))

    return episodes


def _apply_frame_mask(
    obs: dict[str, np.ndarray], mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Filter a native obs dict by a per-frame boolean ``mask``.

    Self/spatial/entity_count slice along axis 0; entity per-token
    fields slice along the token-axis range computed from the kept
    rows' entity_count via cumsum. Returns ``(new_obs, new_count)``.
    """
    new_obs: dict[str, np.ndarray] = {}
    for key in _NATIVE_SELF_FIELDS + _NATIVE_SPATIAL_FIELDS:
        if key in obs:
            new_obs[key] = obs[key][mask]
    counts = obs["entity_count"]
    new_counts = counts[mask]
    new_obs["entity_count"] = new_counts
    # Build a per-token mask from the per-frame mask by replicating
    # mask[i] count_i times in the original concatenated order.
    tok_mask = np.repeat(mask, counts.astype(np.int64))
    for key in _NATIVE_ENTITY_FIELDS:
        if key in obs:
            new_obs[key] = obs[key][tok_mask]
    return new_obs, new_counts


def _validate_native_obs(obs: dict[str, np.ndarray]) -> None:
    """Assert per-field dtypes match engine_norm; surfaces drift loudly.

    No silent fallback — see feedback_no_silent_drops. Each field's
    dtype must equal what unpack_obs_buffer_native produced for the
    native wire format. Entity-count consistency is also checked:
    sum(entity_count) must equal the leading dim of every entity
    per-token array.
    """
    expected = {
        # Self
        "health":           np.uint8,   "effective_armor":  np.uint8,
        "ammo_shells":      np.uint8,   "ammo_nails":       np.uint8,
        "ammo_rockets":     np.uint8,   "ammo_cells":       np.uint8,
        "vel":              np.int16,   "attack_finished":  np.float16,
        "self_weapon_id":   np.uint8,   "self_movement_id": np.uint8,
        "self_items":       np.int32,
        # Spatial
        "spatial_dir":          np.int8,
        "spatial_nearest_dist": np.uint16, "spatial_mean_dist": np.uint16,
        "spatial_openness":    np.uint8,  "spatial_clearance":  np.uint8,
        "spatial_traversable": np.uint8,  "spatial_dropoff":    np.uint8,
        "spatial_solid_frac":  np.uint8,  "spatial_water_frac": np.uint8,
        "spatial_slime_frac":  np.uint8,  "spatial_lava_frac":  np.uint8,
        # Entity
        "entity_count":         np.uint8,
        "entity_types":         np.int8,
        "entity_subject_id":    np.uint8,
        "entity_modality_id":   np.uint8,
        "entity_player_id":     np.uint8,
        "entity_event_count":   np.uint8,
        "entity_event_actions": np.uint8,
        "entity_event_sources": np.uint8,
        "entity_half_extents":  np.uint8,
        "entity_rel":  np.int16,  "entity_vel":  np.int16,
        "entity_path": np.int16,
        "entity_path_dist": np.uint16,
        "entity_eta":     np.float16, "entity_recency": np.float16,
        "entity_facing":  np.uint8,   "entity_team":    np.uint8,
        "entity_score":   np.uint8,   "entity_amount":  np.uint8,
        "entity_regen":   np.float16, "entity_state":   np.uint8,
    }
    for key, dt in expected.items():
        arr = obs.get(key)
        if arr is None:
            continue  # field may be dropped by tokens.{keep,drop}
        if arr.dtype != dt:
            raise RuntimeError(
                f"native obs field {key!r}: expected dtype {dt.__name__}, "
                f"got {arr.dtype}. The native wire parser must produce "
                f"native widths — silent downcast would be a regression."
            )
    counts = obs.get("entity_count")
    if counts is not None:
        total = int(counts.astype(np.int64).sum())
        for key in _NATIVE_ENTITY_FIELDS:
            arr = obs.get(key)
            if arr is None:
                continue
            if arr.shape[0] != total:
                raise RuntimeError(
                    f"native obs entity field {key!r}: leading dim "
                    f"{arr.shape[0]} != sum(entity_count)={total}. "
                    f"Cumsum-based slicing would mis-index the loader."
                )


# ── Train/val split assignment ───────────────────────────────────────

def _split_for_demo(demo_name: str, train_ratio: float, seed: int) -> str:
    """Deterministic train/val assignment from demo name hash."""
    h = hashlib.sha256(f"{seed}:{demo_name}".encode()).digest()
    frac = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    return "train" if frac < train_ratio else "val"


# ── Shard writer ─────────────────────────────────────────────────────

class _ShardWriter:
    """Accumulates episodes and flushes shards to disk.

    On resume, loads the existing manifest and continues numbering
    shards from where the previous run left off so we never overwrite.

    Ordering contract: ``add_episode`` is called in submit-order from
    the orchestration loop (see :func:`collect` — workers run in
    parallel via ``ProcessPoolExecutor``, but completed results are
    drained from a small ``pending`` dict in submit-order before
    reaching the writer). The submit order is the ``sorted(demos)``
    list — alphabetical by demo filename, NOT manifest position
    (the source ndjson manifest may itself be in any order). The
    on-disk shard sequence is therefore deterministic and reproducible
    across collect runs of the same corpus: shard ``N+1`` only ever
    contains demos whose filename sorts after shard ``N``'s demos,
    and within each shard episodes are in arrival order. The trainer
    relies on this — it can iterate shards sequentially without
    re-sorting or random-access reshuffling. The per-episode
    ``demo_idx`` field still refers to the source manifest position,
    so it is not monotonic across shards — but it isn't used for
    ordering, only as a per-episode identifier.
    """

    def __init__(self, split_dir: Path, shard_rows: int):
        self.split_dir = split_dir
        self.shard_rows = max(1, shard_rows)
        self._obs_bufs: dict[str, list[np.ndarray]] = {}
        self._act_bufs: dict[str, list[np.ndarray]] = {}
        self._episode_lengths: list[int] = []
        self._episode_demo_idxs: list[int] = []
        self._episode_idxs: list[int] = []
        self._current_rows = 0
        split_dir.mkdir(parents=True, exist_ok=True)

        # Resume: load existing manifest and continue from last shard
        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists():
            try:
                prev = json.loads(manifest_path.read_text())
                self.shards: list[dict] = prev.get("shards", [])
                self.shard_idx = len(self.shards)
            except (json.JSONDecodeError, KeyError):
                self.shards = []
                self.shard_idx = 0
        else:
            self.shards = []
            self.shard_idx = 0

    def add_episode(
        self,
        obs: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        n_samples: int,
        demo_idx: int,
        episode_idx: int = 0,
    ) -> None:
        for key, arr in obs.items():
            # look_delta is a wire-only inference field — the BC preload
            # re-derives it from the look column, so don't bloat the cache.
            if key == "look_delta":
                continue
            self._obs_bufs.setdefault(key, []).append(arr)
        for key, arr in actions.items():
            self._act_bufs.setdefault(key, []).append(arr)
        self._episode_lengths.append(n_samples)
        self._episode_demo_idxs.append(int(demo_idx))
        self._episode_idxs.append(int(episode_idx))
        self._current_rows += n_samples
        if self._current_rows >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self._episode_lengths:
            return
        prefix = f"shard{self.shard_idx:06d}"
        obs_files: dict[str, str] = {}
        for key, arrays in self._obs_bufs.items():
            fname = f"{prefix}_obs_{key}.npy"
            np.save(self.split_dir / fname, np.concatenate(arrays, axis=0))
            obs_files[key] = fname
        act_files: dict[str, str] = {}
        for key, arrays in self._act_bufs.items():
            fname = f"{prefix}_act_{key}.npy"
            np.save(self.split_dir / fname, np.concatenate(arrays, axis=0))
            act_files[key] = fname
        self.shards.append({
            "rows": self._current_rows,
            "episode_lengths": self._episode_lengths,
            "demo_idxs": self._episode_demo_idxs,
            "episode_idxs": self._episode_idxs,
            "obs": obs_files,
            "actions": act_files,
        })
        self.shard_idx += 1
        self._obs_bufs.clear()
        self._act_bufs.clear()
        self._episode_lengths = []
        self._episode_demo_idxs = []
        self._episode_idxs = []
        self._current_rows = 0
        # Write manifest after every shard so a killed process doesn't
        # lose all progress.  On resume we reload this.
        self._write_manifest()

    def _write_manifest(self) -> None:
        total_episodes = sum(len(s["episode_lengths"]) for s in self.shards)
        manifest = {
            "format": "sharded_v1",
            # engine_norm phase 2 native wire format. The loader checks
            # for this string and refuses caches without it — legacy f16
            # shards have no format_version and must be recollected
            # (no in-place migration support, per the no-backcompat
            # directive). See the native-v1 cache validation in qnn.bc.train.
            "format_version": "native_v1",
            "episodes": total_episodes,
            "shard_rows": self.shard_rows,
            "target_labeler_version": "v3-simple",
            "target_probs_classes": TARGET_PROBS_CLASSES,
            # On-disk layout note: per-shard entity_* .npy files are
            # variable-length along axis 0 (one row per token, not per
            # frame). The companion obs/entity_count.npy is row-indexed
            # and is the loader's cumsum key for recovering per-frame
            # token slices. See qnn.engine_norm entity-block docstring.
            "shards": self.shards,
        }
        (self.split_dir / "manifest.json").write_text(json.dumps(manifest))

    def write_manifest(self) -> None:
        self.flush()
        self._write_manifest()


# ── Per-worker collect function ──────────────────────────────────────

# Stall detection lives in the C worker (qnn_watchdog) and produces a
# clean exit with code QNN_WATCHDOG_EXIT_CODE (77) plus a stderr marker
# when the main loop stops advancing.  The Python side simply reads EOF,
# marks the demo as a permanent error, and moves on — no retries, no
# wall-clock timeout here.
_WATCHDOG_EXIT_CODE = 77


def _infer_demo_type(demo_dir: Path) -> str:
    """Infer the collect output namespace from the corpus directory."""
    name = demo_dir.name.lower()
    if name in {"qwd", "qw", "quakeworld"}:
        return "qwd"
    if name in {"dem", "nq", "netquake"}:
        return "dem"
    if any(demo_dir.glob("*.qwd")) or any(demo_dir.glob("*.mvd")):
        return "qwd"
    if any(demo_dir.glob("*.dem")) or any(demo_dir.glob("*.DEM")):
        return "dem"
    return name or "demo"


def _default_demo_worker(demo_type: str) -> str:
    return "assets/bin/qw_demo_worker" if demo_type == "qwd" else "assets/bin/nq_demo_worker"


def _game_dir_for_demo_dir(demo_dir: Path, asset_root: Path) -> str:
    """Return a game dir usable by the Quake filesystem.

    The worker joins -game with QUAKE_BASEDIR internally, so artifact
    corpora outside assets need a relative path such as ../artifacts/corpus/qwd.
    Use absolute() rather than resolve() to preserve repo-local symlinks.
    """
    return os.path.relpath(demo_dir.absolute(), asset_root.absolute())


def _run_per_demo_collect(
    demo_name: str,
    collect_fn,
    unpack_fn,
    min_ticks: int = 10,
) -> dict:
    """Shared boilerplate for a per-demo pool-worker entry: spawn the
    worker, dispatch the collect, classify errors, unpack episodes,
    return a status dict.

    Callers supply two callbacks:
      ``collect_fn(proc) -> list[dict] | None``
          Sends an op to the worker and reads the framed output stream
          (typically built on top of :func:`_read_collect_frames`).
      ``unpack_fn(ticks) -> list[tuple[obs, act]]``
          Turns the raw tick stream into the per-shard (obs, act)
          sub-episode tuples that ``run_collect`` consumes.

    The "always shut down after a demo" discipline lives here: static
    engine state (oldrealtime, cl_maxfps, cls.latency, ...) leaks
    across demos and truncates the second demo's detected tick rate, so
    a fresh process is the only reliable way to start clean.  Cost is
    ~50 ms per demo, dwarfed by demo I/O. """
    proc = None
    try:
        proc = _get_collect_worker()
        ticks = collect_fn(proc)
    except Exception as exc:
        err_tail = _worker_stderr_tail(proc)
        _shutdown_worker()
        msg = f"{exc}\n{err_tail}" if err_tail else str(exc)
        return {"demo": demo_name, "status": "error", "msg": msg[-4000:]}

    if ticks is None:
        err_tail = _worker_stderr_tail(proc)
        rc = proc.poll() if proc is not None else None
        _shutdown_worker()
        if rc == _WATCHDOG_EXIT_CODE:
            return {"demo": demo_name, "status": "error",
                    "msg": f"watchdog stall\n{err_tail}"[-4000:]}
        return {"demo": demo_name, "status": "crash",
                "msg": err_tail or "worker crash"}

    _shutdown_worker()

    if len(ticks) < min_ticks:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}

    episodes = unpack_fn(ticks)
    # Drop empty sub-episodes; preserve order so episode_idx is
    # deterministic from (manifest_order, run_index).
    sized = [(obs, act, int(next(iter(act.values())).shape[0]) if act else 0)
             for obs, act in episodes]
    sized = [(obs, act, rows) for obs, act, rows in sized if rows > 0]
    if not sized:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}
    return {
        "demo": demo_name, "status": "ok", "ticks": len(ticks),
        "episodes": [{"obs": obs, "actions": act, "rows": rows}
                     for obs, act, rows in sized],
    }


def _collect_demo(args: tuple) -> dict:
    """Collect one demo, return unpacked arrays. Runs in worker process."""
    (demo_name, force_mvd_emit, combat_only, labels,
     total_frames, drop_label_names, sight_only, target_probs_cache) = args
    # Worker always walks the full demo; any clipping happens via the
    # tick mask (segments.drop in the filter config).
    return _run_per_demo_collect(
        demo_name,
        collect_fn=lambda proc: _collect_one_demo(
            proc, demo_name, force_mvd_emit=force_mvd_emit),
        unpack_fn=lambda ticks: _unpack_episode(
            ticks, combat_only=combat_only, labels=labels,
            drop_label_names=drop_label_names, total_frames=total_frames,
            sight_only=sight_only,
            target_probs_cache=target_probs_cache),
    )


# ── Done tracking (append-only) ─────────────────────────────────────

def _load_done_set(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text().splitlines() if line.strip()}


def _append_done(done_path: Path, demo_name: str) -> None:
    with open(done_path, "a") as f:
        f.write(demo_name + "\n")


# ── Manifest loading ─────────────────────────────────────────────────

def _load_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    for line in manifest_path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _bsp_header_valid(path: Path) -> bool:
    """Header-level sanity check on a Q1 BSP file.

    Mod_LoadBrushModel in the engine dereferences each lump's
    ``[ofs, ofs+len]`` window unconditionally; a single lump with bounds
    past EOF segfaults the worker during signon precache.  Validate the
    fixed-size header (15 lumps × 8 bytes, version=29) against the file
    size before declaring the map "available", so demos for corrupt BSPs
    take the same "missing BSP" path as missing files.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(4 + 15 * 8)
            file_size = path.stat().st_size
    except OSError:
        return False
    if len(header) < 4 + 15 * 8:
        return False
    version = struct.unpack_from("<i", header, 0)[0]
    if version != 29:                # Q1 BSP version (HL=23, others unsupported)
        return False
    for i in range(15):
        ofs, ln = struct.unpack_from("<ii", header, 4 + i * 8)
        if ofs < 0 or ln < 0:
            return False
        if ofs + ln > file_size:
            return False
    return True


def _available_maps(asset_root: Path) -> set[str]:
    """Return the set of map names (lowercase, no extension) whose BSPs
    are reachable by the demo worker — either as loose .bsp files under
    */maps/ or packed inside */pak*.pak.  Loose BSPs are header-validated
    so corrupt files don't crash the worker mid-signon."""
    maps: set[str] = set()
    if not asset_root.exists():
        return maps
    for bsp in asset_root.rglob("maps/*.bsp"):
        if _bsp_header_valid(bsp):
            maps.add(bsp.stem.lower())
        else:
            print(f"[collect] skipping invalid BSP: {bsp}", file=sys.stderr)
    for pak in list(asset_root.rglob("pak*.pak")) + list(asset_root.rglob("PAK*.PAK")):
        try:
            with open(pak, "rb") as f:
                head = f.read(12)
                if head[:4] != b"PACK":
                    continue
                dir_ofs, dir_len = struct.unpack("<II", head[4:12])
                f.seek(dir_ofs)
                directory = f.read(dir_len)
            for i in range(dir_len // 64):
                name = directory[i * 64 : i * 64 + 56].rstrip(b"\x00").decode("latin-1", "ignore")
                if name.startswith("maps/") and name.endswith(".bsp"):
                    # Pak-packed BSPs ship with the engine and are trusted;
                    # validation would require slicing into the pak buffer.
                    maps.add(name[5:-4].lower())
        except (OSError, struct.error):
            continue
    return maps


# ── Shared collect runner ────────────────────────────────────────────
#
# Filter / manifest / pool / shard / metadata orchestration shared by the
# BC collect (this module's main()) and the labeler collect
# (qnn.labeler.collect.main()).  Each caller passes a per-demo strategy
# (a Pool-callable that returns {"status", "ticks", "episodes": [...]}
# dicts) plus a work-args builder so the orchestration stays format-
# agnostic — adding a third collect path means writing a per-demo
# function and calling this, not duplicating ~200 lines of pool +
# shard + metadata + fingerprint glue.

def run_collect(
    *,
    output: Path,
    demo_dir: Path,
    manifest_path: Path,
    asset_root: Path,
    demo_worker: str,
    game_dir: str,
    tick_hz: int,
    workers: int,
    shard_rows: int,
    train_ratio: float,
    seed: int,
    keep_pred: dict,
    drop_pred: dict,
    drop_label_names: tuple[str, ...],
    per_demo_fn,
    build_work_args,
    shard_kind: str,
    extra_demo_filter=None,
    extra_metadata: dict | None = None,
    filter_path: Path | None = None,
    tokens_keep:  tuple[str, ...] = (),
    tokens_drop:  tuple[str, ...] = (),
    actions_keep: tuple[str, ...] = (),
    actions_drop: tuple[str, ...] = (),
) -> None:
    """Orchestrate filter → pool → shard → metadata → fingerprint.

    Parameters
    ----------
    per_demo_fn : Callable[[tuple], dict]
        Top-level function (must be picklable) the worker Pool calls
        per demo.  Returns ``{"demo", "status", "ticks", "episodes":
        [{"obs", "actions", "rows"}, ...]}`` for status == "ok",
        ``{"status": "skipped"|"error"|"crash"}`` otherwise.
    build_work_args : Callable[[dict, dict, int, tuple[str, ...]], tuple]
        Builds the per-demo work tuple from
        ``(manifest_entry, labels, total_frames, drop_label_names)``.
        The returned tuple is what ``per_demo_fn`` receives.
    extra_demo_filter : Callable[[dict], str | None] | None
        Optional per-entry filter applied AFTER the keep/drop
        predicates.  Returns a non-empty reason string to exclude the
        entry, ``None`` to keep.  Used by callers that need
        additional gates not expressible in the JSON filter DSL
        (e.g., the labeler's native-rate floor).
    shard_kind : str
        Free-form tag stored in ``collect_metadata.json`` for downstream
        callers to identify the corpus.  BC passes ``"bc"``; labeler
        passes ``"labeler"``.
    extra_metadata : dict | None
        Merged into ``collect_metadata.json`` on top of the runner's
        base metadata.  Use for caller-specific fields (e.g.,
        ``force_mvd_emit``, ``min_hz``).
    filter_path : Path | None
        Path to the filter-config JSON, recorded in the fingerprint.
    """
    available_maps = _available_maps(asset_root)

    labels_lookup: dict[str, dict] = {}
    total_frames_lookup: dict[str, int] = {}
    demo_idx_map: dict[str, int] = {}
    selected_entries: list[dict] = []
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        excluded: list[str] = []
        missing_map = 0
        extra_excluded = 0
        for manifest_pos, e in enumerate(manifest):
            demo_idx_map[e["file"]] = manifest_pos
            labels_lookup[e["file"]] = e.get("labels") or {}
            total_frames_lookup[e["file"]] = int(e.get("total_frames") or 0)
            if drop_pred and _eval_filter(e, drop_pred):
                excluded.append(e["file"]); continue
            if keep_pred and not _eval_filter(e, keep_pred):
                excluded.append(e["file"]); continue
            mp = (e.get("map") or "").lower()
            if mp and mp not in available_maps:
                excluded.append(e["file"]); missing_map += 1; continue
            if extra_demo_filter is not None:
                reason = extra_demo_filter(e)
                if reason:
                    excluded.append(e["file"]); extra_excluded += 1; continue
            selected_entries.append(e)
        demos = sorted([demo_dir / e["file"] for e in selected_entries
                        if (demo_dir / e["file"]).exists()])
        tail_parts: list[str] = []
        if missing_map:
            tail_parts.append(f"{missing_map} missing BSP")
        if extra_excluded:
            tail_parts.append(f"{extra_excluded} extra-filter")
        tail = f" ({', '.join(tail_parts)})" if tail_parts else ""
        print(f"Manifest: {len(manifest)} entries, {len(excluded)} excluded{tail}, "
              f"{len(selected_entries)} included")
    else:
        demos = sorted(
            list(demo_dir.glob("*.dem")) + list(demo_dir.glob("*.DEM"))
            + list(demo_dir.glob("*.qwd")) + list(demo_dir.glob("*.mvd"))
        )
        for idx, demo in enumerate(demos):
            demo_idx_map[demo.name] = idx
            labels_lookup[demo.name] = {}
            total_frames_lookup[demo.name] = 0
        selected_entries = [{"file": d.name} for d in demos]
        print(f"No manifest found, using all {len(demos)} demo files")

    if not demos:
        print("No demos to collect")
        sys.exit(1)

    done_path = output / "done.log"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_demos = _load_done_set(done_path)

    selected_by_name = {e["file"]: e for e in selected_entries}
    work: list[tuple] = []
    for demo in demos:
        if demo.name in done_demos:
            continue
        entry = selected_by_name.get(demo.name, {"file": demo.name})
        labels = labels_lookup.get(demo.name, {})
        total_frames = total_frames_lookup.get(demo.name, 0)
        work.append(build_work_args(entry, labels, total_frames, drop_label_names))

    n_workers = max(1, workers)
    print(f"Output: {output}")
    print(f"Game dir: {game_dir}")
    print(f"Shard kind: {shard_kind}")
    print(f"Demos: {len(demos)} total, {len(done_demos)} cached, {len(work)} to collect")
    print(f"Workers: {n_workers}")

    if not work:
        train_manifest = output / "precomputed_train" / "manifest.json"
        val_manifest = output / "precomputed_val" / "manifest.json"
        if train_manifest.exists() and val_manifest.exists():
            print("Done (no new data).")
            return
        sys.exit("done.log marks demos complete but shards are missing; "
                 "clear done.log to force re-collect.")

    train_writer = _ShardWriter(output / "precomputed_train", shard_rows)
    val_writer = _ShardWriter(output / "precomputed_val", shard_rows)

    import time as _time
    collected = 0
    skipped = 0
    errors = 0
    total_ticks = 0
    t_start = _time.monotonic()
    total_work = len(work)
    progress_step = max(1, total_work // 20)

    # Workers fan out in parallel via the pool, but the writer side
    # drains results in *submit order* so the on-disk shards preserve
    # the ``sorted(demos)`` ordering — alphabetical by filename, not
    # manifest position. The trainer can then iterate shards
    # sequentially without re-sorting or random access.
    #
    # Mechanics: each future carries its submit index. As futures
    # complete (in arbitrary order), the result is stashed in
    # ``pending`` keyed by submit index. After every completion we
    # drain a contiguous prefix from ``pending`` starting at
    # ``next_idx``. Worst-case memory overhead is bounded by
    # ``n_workers - 1`` results held while a slow demo at the front
    # of the queue finishes — small relative to the per-worker
    # working set, and the historical measurement was ~20% wall-time
    # cost.
    splits = [_split_for_demo(w[0], train_ratio, seed) for w in work]

    def _consume(idx: int, payload: object) -> None:
        nonlocal collected, skipped, errors, total_ticks
        work_item = work[idx]
        demo_name = work_item[0]
        if isinstance(payload, BaseException):
            print(f"  {demo_name}... EXCEPTION: {payload}")
            errors += 1
            return
        result = payload  # type: ignore[assignment]
        status = result["status"]
        if status == "ok":
            writer = train_writer if splits[idx] == "train" else val_writer
            for episode_idx, ep in enumerate(result["episodes"]):
                rows = int(ep["rows"])
                writer.add_episode(ep["obs"], ep["actions"], rows,
                                    demo_idx_map[demo_name], episode_idx)
                total_ticks += rows
            _append_done(done_path, demo_name)
            collected += 1
        elif status == "skipped":
            _append_done(done_path, demo_name)
            skipped += 1
        elif status in ("crash", "error"):
            msg = result.get("msg") or "worker crash"
            tag = "FAILED (worker crash)" if status == "crash" else "ERROR"
            print(f"  {demo_name}... {tag}: {msg}")
            errors += 1
        else:
            print(f"  {demo_name}... unexpected status={status!r}")
            errors += 1

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_collect_worker,
        initargs=(
            demo_worker, str(asset_root), tick_hz, game_dir,
            tuple(tokens_keep),  tuple(tokens_drop),
            tuple(actions_keep), tuple(actions_drop),
        ),
    ) as pool:
        futures = {pool.submit(per_demo_fn, w): i for i, w in enumerate(work)}
        pending: dict[int, object] = {}
        next_idx = 0
        for future in as_completed(futures):
            # ``futures.pop`` is load-bearing: a ``Future`` caches its
            # own ``.result()`` internally, so leaving completed
            # futures in the dict keeps every demo's per-episode
            # arrays alive until the whole as_completed loop ends.
            # Pre-crash measurement (82f6f39a) was 18 GB peak with
            # ``futures[future]`` here, 0.71 GB peak after switching
            # to ``pop``. The result moves into ``pending`` (and out
            # of it via _consume below), so the only remaining strong
            # refs to the future are the loop variable (replaced next
            # iter) and as_completed's internal bookkeeping (released
            # on yield).
            idx = futures.pop(future)
            try:
                pending[idx] = future.result()
            except Exception as exc:
                pending[idx] = exc
            while next_idx in pending:
                _consume(next_idx, pending.pop(next_idx))
                next_idx += 1

                done_count = collected + skipped + errors
                if done_count % progress_step == 0 or done_count == total_work:
                    elapsed = _time.monotonic() - t_start
                    rate = done_count / max(elapsed, 0.01)
                    ticks_rate = total_ticks / max(elapsed, 0.01)
                    remaining = (total_work - done_count) / max(rate, 0.01)
                    mins, secs = divmod(int(remaining), 60)
                    hrs, mins = divmod(mins, 60)
                    eta = f"{hrs}h{mins:02d}m" if hrs else f"{mins}m{secs:02d}s"
                    print(f"  [{done_count}/{total_work}] {rate:.1f} demos/s, "
                          f"{ticks_rate/1000:.0f}K ticks/s, "
                          f"{total_ticks/1e6:.1f}M ticks total, "
                          f"queue={len(pending)}, ETA {eta}")

    train_writer.write_manifest()
    val_writer.write_manifest()

    train_eps = sum(len(s["episode_lengths"]) for s in train_writer.shards)
    val_eps = sum(len(s["episode_lengths"]) for s in val_writer.shards)

    metadata = {
        "episodes": train_eps + val_eps,
        "train": train_eps,
        "val": val_eps,
        "collected": collected,
        "skipped": skipped,
        "errors": errors,
        "tick_hz": tick_hz,
        "seed": seed,
        "shard_kind": shard_kind,
        "tokens_keep":  list(tokens_keep),
        "tokens_drop":  list(tokens_drop),
        "actions_keep": list(actions_keep),
        "actions_drop": list(actions_drop),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    (output / "collect_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Deterministic identity fingerprint of every input that influenced
    # what's on disk (filter, manifest, done log, worker, code).  The
    # trainer reads this to verify it's loading the exact collection it
    # was configured against.
    from qnn import collection_fingerprint
    fp = collection_fingerprint.compute(
        filter_path=filter_path,
        manifest_path=manifest_path,
        done_log_path=done_path,
        worker_binary_path=Path(demo_worker),
        data_dir=output,
        repo_root=Path(__file__).resolve().parents[3],
    )
    collection_fingerprint.write(fp, output)

    print(f"\nDone: {collected} collected, {skipped} skipped, {errors} errors")
    print(f"  train: {train_eps} episodes in {len(train_writer.shards)} shards")
    print(f"  val: {val_eps} episodes in {len(val_writer.shards)} shards")
    print(f"  fingerprint: {fp['fingerprint']}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    # Line-buffer stdout so progress/error lines appear immediately in
    # redirected log files.  Without this, Python fully buffers stdout
    # when it's not a TTY and a multi-hour collect shows no output until
    # the buffer fills or the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Collect BC training data from demo files")
    parser.add_argument("--demo-dir", required=True, help="Directory containing .dem/.qwd/.mvd files")
    parser.add_argument(
        "--output",
        default="",
        help="Output directory for sharded .npy caches (default: artifacts/collect/<demo-type>)",
    )
    parser.add_argument("--manifest", default="", help="Path to manifest.ndjson (default: auto-detected)")
    parser.add_argument("--demo-worker", default="", help="Demo worker binary (default: inferred from demo type)")
    parser.add_argument("--asset-root", default="assets")
    parser.add_argument("--tick-hz", type=int, default=20, help="Engine tick rate = emit rate (workers tick at this rate)")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=30, help="Parallel workers")
    parser.add_argument("--shard-rows", type=int, default=262144, help="Rows per shard")
    parser.add_argument(
        "--force-mvd-emit",
        action="store_true",
        help="QW only: derive action labels via the MVD inference path even on QWD demos with"
             " usercmd_t available.  Used to validate label drift between recorded inputs and"
             " 9-candidate physics reconstruction; should never be set for production training.",
    )
    parser.add_argument(
        "--combat-only",
        action="store_true",
        help="Drop every frame where the engagement labeler returned TARGET_IGNORE"
             " (i.e., demonstrator wasn't shooting at a tracked enemy).  Default OFF —"
             " keep all frames so non-combat movement/positioning signal is preserved"
             " and the corpus isn't gated by a fragile post-processing derivation.",
    )
    parser.add_argument(
        "--sight",
        action="store_true",
        help="Restrict the target labeler to modality 0 (SIGHT) actors only;"
             " engagements break when the target pid enters SOUND or MEMORY"
             " modality. Pair with a matching train-time token_mask when"
             " reproducing sight-only entity ablations.",
    )
    parser.add_argument(
        "--no-target-dist-cache",
        dest="target_probs_cache",
        action="store_false",
        default=True,
        help="Disable writing the per-shard target_probs (.npy f16) sidecar."
             " Default: enabled. Caching the labeler output amortizes the"
             " ~95%% of BC load time the labeler accounts for (run once at"
             " collect, mmap thereafter). Use --no-target-dist-cache when"
             " iterating on the labeler itself — without the cache, train"
             " always re-runs the labeler, picking up your config edits.",
    )
    parser.add_argument(
        "--filter-config",
        default=None,
        help="Path to a JSON filter config (MongoDB query syntax).  Schema:"
             " {demos: {keep, drop}, segments: {drop}, tokens: {keep, drop},"
             " actions: {keep, drop}}.  demos.{keep,drop} are predicate dicts"
             " evaluated per manifest entry (demo kept iff demos.keep matches"
             " AND no demos.drop matches); segments.drop is a list of label"
             " names whose intervals are masked out of the kept frames;"
             " tokens / actions keep/drop list obs / act field names to"
             " include or exclude (mutually exclusive within an axis)."
             " Field paths: manifest fields or `labels.<name>.<aggregate>`"
             " where aggregate is one of coverage, count, duration, exists."
             " Operators: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, plus"
             " $and/$or/$not.  Unknown fields/operators/labels fail loud at"
             " startup.  Default: no filtering (every manifest entry"
             " collected, every field on disk).",
    )
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    demo_type = _infer_demo_type(demo_dir)
    output = Path(args.output) if args.output else Path("artifacts/collect") / demo_type

    # Pin the filter to <output>/filter.json so the spec used to produce
    # this cache always travels with it. See _load_and_pin_filter.
    filter_spec = _load_and_pin_filter(
        output, Path(args.filter_config) if args.filter_config else None)
    demos_block    = filter_spec.get("demos") or {}
    segments_block = filter_spec.get("segments") or {}
    tokens_block   = filter_spec.get("tokens") or {}
    actions_block  = filter_spec.get("actions") or {}
    keep_pred = demos_block.get("keep") or {}
    drop_pred = demos_block.get("drop") or {}
    drop_label_names = tuple(segments_block.get("drop") or ())
    tokens_keep  = tuple(tokens_block.get("keep") or ())
    tokens_drop  = tuple(tokens_block.get("drop") or ())
    actions_keep = tuple(actions_block.get("keep") or ())
    actions_drop = tuple(actions_block.get("drop") or ())

    demo_worker_path = args.demo_worker or _default_demo_worker(demo_type)
    demo_worker = str(Path(demo_worker_path).resolve())
    game_dir = _game_dir_for_demo_dir(demo_dir, Path(args.asset_root))

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        corpus_manifest = demo_dir.parent / f"{demo_dir.name}_manifest.ndjson"
        legacy_manifest = demo_dir / "manifest.ndjson"
        manifest_path = corpus_manifest if corpus_manifest.exists() else legacy_manifest

    print(f"Demo type: {demo_type}")

    # Per-demo work tuple: (demo_name, force_mvd_emit,
    # combat_only, labels, total_frames, drop_label_names, sight_only,
    # target_probs_cache) — the shape _collect_demo unpacks.
    def build_work_args(entry, labels, total_frames, drop_labels):
        return (entry["file"], args.force_mvd_emit,
                args.combat_only, labels, total_frames, drop_labels,
                args.sight,
                args.target_probs_cache)

    run_collect(
        output=output,
        demo_dir=demo_dir,
        manifest_path=manifest_path,
        asset_root=Path(args.asset_root),
        demo_worker=demo_worker,
        game_dir=game_dir,
        tick_hz=args.tick_hz,
        workers=args.workers,
        shard_rows=args.shard_rows,
        train_ratio=args.train_ratio,
        seed=args.seed,
        keep_pred=keep_pred,
        drop_pred=drop_pred,
        drop_label_names=drop_label_names,
        per_demo_fn=_collect_demo,
        build_work_args=build_work_args,
        shard_kind="bc",
        filter_path=output / "filter.json",
        tokens_keep=tokens_keep,
        tokens_drop=tokens_drop,
        actions_keep=actions_keep,
        actions_drop=actions_drop,
    )


if __name__ == "__main__":
    main()
