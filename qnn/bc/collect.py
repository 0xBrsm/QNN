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

from qnn import filter_dsl
from qnn.wire import (
    OBS_BUFFER_SIZE, ACTION_SIZE, TICK_HEADER_SIZE,
    ACTION_FIELDS, FLAG_DONE,
    unpack_obs_buffer,
)
from qnn.actions import MOVE_AXIS_THRESHOLD, MOVE_CLASS_NEG, MOVE_CLASS_NONE, MOVE_CLASS_POS
from qnn.bc.target_labeler import TARGET_IGNORE, label_enemy_target
from qnn.vocab import TOKEN_ACTOR  # noqa: F401 — kept for back-compat imports



TICK_TOTAL_SIZE = TICK_HEADER_SIZE + OBS_BUFFER_SIZE + ACTION_SIZE

# ── Filter spec (MongoDB-style query DSL) ───────────────────────────
#
# The filter is a JSON config file (--filter-config) with schema:
#
#   {
#     "keep": {<predicate-dict>},
#     "drop": {<predicate-dict>},
#     "drop_tick_labels": ["intermission", "paused", ...]
#   }
#
# The "keep" predicate must match for a demo to be included.  ANY
# "drop" predicate matching causes the demo to be excluded.  Both
# default to empty (no constraint).
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

# Direct labels that can appear in `drop_tick_labels`.
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


def _validate_filter_schema(spec: dict) -> None:
    """Validate filter config at startup; surfaces unknown fields,
    operators, label names, and channel names loudly before collection
    runs.  Per agents/conventions.md. """
    if not isinstance(spec, dict):
        raise ValueError(
            f"filter config must be a JSON object, got "
            f"{type(spec).__name__}"
        )
    allowed = {"keep", "drop", "drop_tick_labels"}
    extras = set(spec.keys()) - allowed
    if extras:
        raise ValueError(
            f"unknown top-level keys in filter config: {sorted(extras)}.  "
            f"Allowed: {sorted(allowed)}"
        )
    for which in ("keep", "drop"):
        sub = spec.get(which) or {}
        if not isinstance(sub, dict):
            raise ValueError(f"{which!r} must be a dict, got {type(sub).__name__}")
        filter_dsl.validate_predicate(sub, _is_valid_manifest_path)
    tick_labels = spec.get("drop_tick_labels") or []
    if not isinstance(tick_labels, list):
        raise ValueError("drop_tick_labels must be a list of label names")
    bad = [n for n in tick_labels if n not in _VALID_TICK_DROP_LABELS]
    if bad:
        raise ValueError(
            f"unknown drop_tick_labels: {bad}.  Valid: "
            f"{sorted(_VALID_TICK_DROP_LABELS)}"
        )


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
    surviving run of frames after `drop_tick_labels` carves intervals
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
_WORKER_DEMO_STDERR_FILE = None
_WORKER_DEMO_STDERR_PATH: Path | None = None
_ACTION_RECORD_DTYPE = np.dtype(
    {
        "names": list(ACTION_FIELDS.keys()),
        "formats": [dtype if not shape else (dtype, shape) for _, dtype, shape in ACTION_FIELDS.values()],
        "offsets": [offset for offset, _, _ in ACTION_FIELDS.values()],
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


def _init_collect_worker(demo_worker: str, asset_root: str, tick_hz: int, game_dir: str) -> None:
    global _WORKER_DEMO_ARGS
    _WORKER_DEMO_ARGS = (demo_worker, asset_root, int(tick_hz), game_dir)
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

def _collect_one_demo(proc: subprocess.Popen, demo_name: str,
                      play_start: int = 0, play_end: int = 999999999,
                      force_mvd_emit: bool = False,
                      ) -> list[dict] | None:
    cmd = json.dumps({
        "op": "collect", "demo_path": demo_name, "seed": 0,
        "play_start": play_start, "play_end": play_end,
        "force_mvd_emit": 1 if force_mvd_emit else 0,
    }) + "\n"
    proc.stdin.write(cmd.encode())
    proc.stdin.flush()

    ticks = []
    while True:
        if proc.poll() is not None:
            return None
        magic = proc.stdout.read(4)
        if not magic or len(magic) < 4:
            return None
        if magic[0:1] == b'{':
            rest = proc.stdout.readline()
            try:
                err = json.loads(magic + rest)
                raise RuntimeError(err.get("error", "unknown error"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(f"Worker error: {(magic + rest)[:200]!r}")
        if magic != b'QOBS':
            raise RuntimeError(f"Bad magic: {magic!r}")
        raw = proc.stdout.read(TICK_TOTAL_SIZE) if proc.poll() is None else b""
        if len(raw) < TICK_TOTAL_SIZE:
            return None

        header = struct.unpack_from("<IIIHH", raw)
        flags = header[3]
        ticks.append({
            "obs": raw[TICK_HEADER_SIZE:TICK_HEADER_SIZE + OBS_BUFFER_SIZE],
            "action": raw[TICK_HEADER_SIZE + OBS_BUFFER_SIZE:],
            "done": bool(flags & FLAG_DONE),
        })
        if flags & FLAG_DONE:
            break
    return ticks


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


def _compact_obs_arrays(obs_arrays: dict[str, np.ndarray]) -> None:
    """Downcast obs to compact on-disk dtypes.  Mutates in place.

    Float scalars are bounded under abs(3.4) per a corpus scan, so fp16
    (max ~65k) is lossless within signal floor.  Int columns are bounded by
    qnn.vocab sizes (action vocab=20, entity vocab=42), all fit in uint8.
    entity_types uses -1 as the empty sentinel — int8 keeps the sign cleanly
    instead of remapping to a uint sentinel.
    """
    _f16 = ("self_scalars", "spatial_scalars", "entity_scalars_raw")
    for key in _f16:
        arr = obs_arrays.get(key)
        if arr is not None and arr.dtype != np.float16:
            obs_arrays[key] = np.ascontiguousarray(arr, dtype=np.float16)

    _u8 = ("entity_event_actions", "entity_event_sources", "entity_ids")
    for key in _u8:
        arr = obs_arrays.get(key)
        if arr is not None and arr.dtype != np.uint8:
            obs_arrays[key] = np.ascontiguousarray(arr, dtype=np.uint8)

    types = obs_arrays.get("entity_types")
    if types is not None and types.dtype != np.int8:
        obs_arrays["entity_types"] = np.ascontiguousarray(types, dtype=np.int8)


def _compact_action_arrays(act_arrays: dict[str, np.ndarray]) -> None:
    """Convert raw float/int wire labels to the compact on-disk format.

    Mutates `act_arrays` in place.  See qnn.actions docstring for the spec.

      move    float32[T, 3]  → uint8[T] packed: bits 0-1=fb, 2-3=lr, 4-5=ud,
                                each holding a class index in {0=neg, 1=none, 2=pos}.
      look    float32[T, 3]  → float16[T, 3]
      fire    int32[T]       → uint8[T]
      weapon  int32[T] raw engine weapon byte → uint8[T] same value:
                0 = no weapon held (pre-spawn, dead, transitional),
                1..8 = Quake weapon id in impulse order (axe..thunderbolt).
              No-weapon frames are kept on disk so downstream signals
              (move, fire, look) still train; the model side masks them
              out of the weapon-head CE loss via ignore_index.
      target  int64[T]       (left as-is — token slot indices need full range)

    Each axis is thresholded independently against MOVE_AXIS_THRESHOLD = 0.1.
    The C-side cmd values (forwardmove/maxspeed etc., in [-1, 1]) are clean
    multiples of 200/320 ≈ 0.625 in standard QW play, so 0.1 separates
    "pressed" from "released" without dropping any real press.

    The engine-facing `switch` slot is derived from the weapon head's
    argmax at inference time (model.policy._weapon_switch_slots_from_choices);
    no `switch` array is written to disk — it would be redundant since
    the slot is a pure function of the weapon class.
    """
    move = np.asarray(act_arrays.get("move"))
    if move is not None and move.dtype != np.uint8:
        # move shape: (T, 3) float — fb, lr, ud
        t = MOVE_AXIS_THRESHOLD
        # Build per-axis class index in {0, 1, 2} via threshold arithmetic.
        #   class = NONE + 1*(v >  t) - 1*(v < -t)
        # which yields NONE-1=NEG when v<-t, NONE+1=POS when v>+t, else NONE.
        none = np.full(move.shape[0], MOVE_CLASS_NONE, dtype=np.uint8)
        fb_cls = none + (move[:, 0] >  t).astype(np.uint8) - (move[:, 0] < -t).astype(np.uint8)
        lr_cls = none + (move[:, 1] >  t).astype(np.uint8) - (move[:, 1] < -t).astype(np.uint8)
        ud_cls = none + (move[:, 2] >  t).astype(np.uint8) - (move[:, 2] < -t).astype(np.uint8)
        # Pack bits 0-1=fb, 2-3=lr, 4-5=ud (2 bits per axis × 3 axes = 6 bits).
        packed = (fb_cls & 0x3) | ((lr_cls & 0x3) << 2) | ((ud_cls & 0x3) << 4)
        act_arrays["move"] = packed.astype(np.uint8)

    look = act_arrays.get("look")
    if look is not None and look.dtype != np.float16:
        act_arrays["look"] = np.ascontiguousarray(look, dtype=np.float16)

    fire = act_arrays.get("fire")
    if fire is not None and fire.dtype != np.uint8:
        act_arrays["fire"] = np.ascontiguousarray(fire, dtype=np.uint8)

    weapon = act_arrays.get("weapon")
    if weapon is not None:
        weapon_bytes = np.asarray(weapon, dtype=np.int32).reshape(-1)
        invalid = weapon_bytes[(weapon_bytes < 0) | (weapon_bytes > 8)]
        if invalid.size:
            sample = sorted({int(v) for v in invalid[:8]})
            raise ValueError(
                f"weapon bytes must be in 0..8 (0=no weapon, 1..8=axe..LG), "
                f"got {sample}"
            )
        act_arrays["weapon"] = np.ascontiguousarray(weapon_bytes.astype(np.uint8))


def _unpack_episode(
    ticks: list[dict],
    combat_only: bool = False,
    labels: dict | None = None,
    drop_label_names: tuple[str, ...] = (),
    total_frames: int = 0,
    sight_only: bool = False,
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Unpack raw ticks into one or more (obs, action) sub-episodes.

    When `drop_tick_labels` carves intervals out of the demo, each
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

    obs_list = [unpack_obs_buffer(t["obs"]) for t in ticks]
    obs_arrays: dict[str, np.ndarray] = {}
    for key in obs_list[0]:
        obs_arrays[key] = np.stack([obs[key] for obs in obs_list], axis=0)

    action_blob = b"".join(t["action"] for t in ticks)
    records = np.frombuffer(action_blob, dtype=_ACTION_RECORD_DTYPE, count=n)
    act_arrays: dict[str, np.ndarray] = {}
    for name, (_, _, shape) in ACTION_FIELDS.items():
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
        sub_obs = {key: values[s:e] for key, values in obs_arrays.items()}
        sub_act = {key: values[s:e] for key, values in act_arrays.items()}

        # Derive combat target labels per sub-episode so the labeler
        # never reads across a dropped interval.
        sub_act["target"] = label_enemy_target(sub_obs, sub_act,
                                                sight_only=sight_only)

        # Combat-only further filters within a sub-episode.  It still
        # concatenates surviving frames inside the sub-episode (no
        # second-level segmentation here — the filter-config keep mask
        # is the segmentation boundary).
        if combat_only:
            keep2 = np.asarray(sub_act["target"]).reshape(-1) != TARGET_IGNORE
            if not bool(np.any(keep2)):
                continue
            if not bool(np.all(keep2)):
                sub_obs = {key: values[keep2] for key, values in sub_obs.items()}
                sub_act = {key: values[keep2] for key, values in sub_act.items()}

        _compact_action_arrays(sub_act)
        _compact_obs_arrays(sub_obs)
        episodes.append((sub_obs, sub_act))

    return episodes


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
            "episodes": total_episodes,
            "shard_rows": self.shard_rows,
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


def _collect_demo(args: tuple) -> dict:
    """Collect one demo, return unpacked arrays. Runs in worker process."""
    (demo_name, force_mvd_emit, combat_only, labels,
     total_frames, drop_label_names, sight_only) = args
    # Worker always walks the full demo; any clipping happens via the
    # tick mask (drop_tick_labels in the filter config).
    proc = None

    try:
        proc = _get_collect_worker()
        ticks = _collect_one_demo(proc, demo_name,
                                  play_start=0, play_end=999999999,
                                  force_mvd_emit=force_mvd_emit)
    except Exception as exc:
        err_tail = _worker_stderr_tail(proc)
        _shutdown_worker()
        msg = str(exc)
        if err_tail:
            msg = f"{msg}\n{err_tail}"
        return {"demo": demo_name, "status": "error", "msg": msg[-4000:]}

    if ticks is None:
        err_tail = _worker_stderr_tail(proc)
        rc = proc.poll() if proc is not None else None
        _shutdown_worker()
        if rc == _WATCHDOG_EXIT_CODE:
            return {"demo": demo_name, "status": "error",
                    "msg": f"watchdog stall\n{err_tail}"[-4000:]}
        return {"demo": demo_name, "status": "crash", "msg": err_tail or "worker crash"}

    # Always shut down the worker after a successful demo: static engine
    # state (oldrealtime in Host_Frame, cl_maxfps / rate cvars, cls.latency,
    # and so on) leaks across demos in ways that truncate the second demo's
    # detected tick rate and cut emission short on multi-map replays.  The
    # per-demo Host_Init cost is small (~50ms) next to the second-plus of
    # actual demo processing, and a fresh process is the only reliable way
    # to start from a known-clean state.
    _shutdown_worker()

    if len(ticks) < 10:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}

    episodes = _unpack_episode(ticks,
                                combat_only=combat_only,
                                labels=labels,
                                drop_label_names=drop_label_names,
                                total_frames=total_frames,
                                sight_only=sight_only)
    # Drop empty episodes; preserve original order so episode_idx is
    # deterministic from (manifest_order, run_index).
    sized = [(obs, act, int(next(iter(act.values())).shape[0]) if act else 0)
             for obs, act in episodes]
    sized = [(obs, act, rows) for obs, act, rows in sized if rows > 0]
    if not sized:
        return {"demo": demo_name, "status": "skipped", "ticks": 0}
    return {
        "demo": demo_name,
        "status": "ok",
        "ticks": len(ticks),
        "episodes": [{"obs": obs, "actions": act, "rows": rows}
                     for obs, act, rows in sized],
    }


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
    for pak in asset_root.rglob("pak*.pak"):
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

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_collect_worker,
        initargs=(demo_worker, str(asset_root), tick_hz, game_dir),
    ) as pool:
        futures = {pool.submit(per_demo_fn, w): w for w in work}
        for future in as_completed(futures):
            work_item = futures[future]
            demo_name = work_item[0]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  {demo_name}... EXCEPTION: {exc}")
                errors += 1
            else:
                status = result["status"]
                if status == "ok":
                    split = _split_for_demo(demo_name, train_ratio, seed)
                    writer = train_writer if split == "train" else val_writer
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
                          f"{total_ticks/1e6:.1f}M ticks total, ETA {eta}")

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
             " modality.  Combined with a train-time token_mask that drops the"
             " same modalities, this reproduces the deprecated"
             " ``--entity-filter pvs_actors`` behavior end-to-end.",
    )
    parser.add_argument(
        "--filter-config",
        default=None,
        help="Path to a JSON filter config (MongoDB query syntax).  Schema:"
             " {keep, drop} are predicate dicts evaluated per manifest entry"
             " (demo kept iff keep matches AND no drop matches);"
             " drop_tick_labels is a list of label names whose intervals are"
             " masked out of the kept frames.  Field paths: manifest fields"
             " or `labels.<name>.<aggregate>` where aggregate is one of"
             " coverage, count, duration, exists.  Operators: $eq, $ne,"
             " $in, $nin, $gt, $gte, $lt, $lte, plus $and/$or/$not.  Unknown"
             " fields/operators/labels fail loud at startup.  Default: no"
             " filtering (every manifest entry collected).",
    )
    args = parser.parse_args()

    if args.filter_config:
        try:
            with open(args.filter_config) as f:
                filter_spec = json.load(f)
            _validate_filter_schema(filter_spec)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            parser.error(f"--filter-config {args.filter_config!r}: {exc}")
    else:
        filter_spec = {}
    keep_pred = filter_spec.get("keep") or {}
    drop_pred = filter_spec.get("drop") or {}
    drop_label_names = tuple(filter_spec.get("drop_tick_labels") or ())

    demo_dir = Path(args.demo_dir)
    demo_type = _infer_demo_type(demo_dir)
    output = Path(args.output) if args.output else Path("artifacts/collect") / demo_type
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
    # combat_only, labels, total_frames, drop_label_names, sight_only) —
    # the shape _collect_demo unpacks.
    def build_work_args(entry, labels, total_frames, drop_labels):
        return (entry["file"], args.force_mvd_emit,
                args.combat_only, labels, total_frames, drop_labels,
                args.sight)

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
        filter_path=Path(args.filter_config) if args.filter_config else None,
    )


if __name__ == "__main__":
    main()
