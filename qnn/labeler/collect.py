"""Native-rate slim collect for the move labeler.

Walks a QWD corpus, drives qw_demo_worker at native rate (tick_hz=0),
and writes only the slim per-tick arrays the labeler consumes:

  obs/self_velocity        fp16 (T, 3)    body-frame, pre-normalized in C
  obs/self_movement_id     uint8 (T,)
  obs/look                 fp16 (T, 3)    per-native-frame view delta
  obs/c_rule_fire          uint8 (T,)     engine sound+ammo fire detection
  obs/c_rule_jump          uint8 (T,)     engine ground→air jump detection
  obs/target_valid_mask    uint8 (T,)     per-axis engine-effective bits
                                          (bit0=fb..bit4=weapon)
  obs/usercmd_fire         uint8 (T,)     true fire press (cmd-window OR'd)
  obs/weapon_id            uint8 (T,)     held weapon 1..8 (0 = none)
  actions/move             uint8 (T,)     packed fb|lr<<2|ud<<4

Demo-level filtering goes through `--filter-config` (same JSON schema as
qnn.bc.collect, with `keep` / `drop` MongoDB-style predicates plus
`drop_tick_labels`).  Pointing it at `artifacts/collect/qwd/filter.json`
selects the same demo set the BC corpus uses.  `drop_tick_labels`
intervals (signon / dead / intermission) carve sub-episodes out of each
demo so the labeler's bidirectional context never bridges a dropped
interval — same semantics as bc.collect.

Native-rate constraint: `--min-hz` (default 70) drops demos detected
below this recording rate.  Labeler-specific, on top of the
filter-config.

Usage:
    PYTHONPATH=src python -m qnn.labeler.collect \\
        --demo-dir       artifacts/corpus/qwd \\
        --manifest       artifacts/corpus/qwd_manifest.ndjson \\
        --rate-tsv       artifacts/qwd_rate_histogram.tsv \\
        --filter-config  artifacts/collect/qwd/filter.json \\
        --output         artifacts/collect/qwd_labeler \\
        --workers        30
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from qnn.wire import FLAG_DONE
from qnn.bc.collect import (
    _WATCHDOG_EXIT_CODE,
    _default_demo_worker,
    _game_dir_for_demo_dir,
    _get_collect_worker,
    _label_keep_mask,
    _runs_from_mask,
    _shutdown_worker,
    _validate_filter_schema,
    _worker_stderr_tail,
    run_collect,
)


# ── LOBS stream IO ────────────────────────────────────────────────────────────
#
# LOBS framing (matches QNN_EmitLabelerTick in qnn_collect_helpers.c):
#   "LOBS"                4 bytes magic
#   tick u32              4
#   tick_hz u32           4
#   flags u16             2
#   payload               19  (see qnn.wire.unpack_labeler_buffer)
# Total: 33 bytes per native tick.
#
# Kept local to this module so the BC collect doesn't have to know
# about the labeler-only wire format.

_LOBS_HEADER_SIZE = 10
_LOBS_PAYLOAD_SIZE = 19
_LOBS_FRAME_AFTER_MAGIC = _LOBS_HEADER_SIZE + _LOBS_PAYLOAD_SIZE


def _collect_one_demo_labeler(
    proc: subprocess.Popen,
    demo_name: str,
    play_start: int = 0,
    play_end: int = 999999999,
    force_mvd_emit: bool = False,
) -> list[dict] | None:
    """Dispatch one demo to the worker in labeler_mode and read its LOBS
    stream.  Mirrors qnn.bc.collect._collect_one_demo but for the slim
    labeler wire format — the BC code never has to handle LOBS."""
    cmd = json.dumps({
        "op": "collect", "demo_path": demo_name, "seed": 0,
        "play_start": play_start, "play_end": play_end,
        "force_mvd_emit": 1 if force_mvd_emit else 0,
        "labeler_mode":   1,
    }) + "\n"
    proc.stdin.write(cmd.encode())
    proc.stdin.flush()

    ticks: list[dict] = []
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
        if magic != b'LOBS':
            raise RuntimeError(f"Bad magic: {magic!r}")
        raw = proc.stdout.read(_LOBS_FRAME_AFTER_MAGIC) if proc.poll() is None else b""
        if len(raw) < _LOBS_FRAME_AFTER_MAGIC:
            return None
        tick_idx, tick_hz, flags = struct.unpack_from("<IIH", raw, 0)
        ticks.append({
            "tick":    int(tick_idx),
            "tick_hz": int(tick_hz),
            "flags":   int(flags),
            "lobs":    raw[_LOBS_HEADER_SIZE:],
            "done":    bool(flags & FLAG_DONE),
        })
        if flags & FLAG_DONE:
            break
    return ticks


# ── slim unpack ───────────────────────────────────────────────────────────────

def _unpack_labeler_episode(
    ticks: list[dict],
    labels: dict | None = None,
    drop_label_names: tuple[str, ...] = (),
    total_frames: int = 0,
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Slim unpack: directly read fields out of the LOBS per-tick payload
    and split into sub-episodes wherever ``drop_tick_labels`` carves a
    gap.  Mirrors qnn.bc.collect._unpack_episode's segmentation
    semantics so the labeler's bidirectional context never bridges a
    dropped interval.

    Returns a list of (obs, action) sub-episodes in source order.  Empty
    list means the demo contributes no rows.
    """
    n = len(ticks)
    if n == 0:
        return []

    payload = np.frombuffer(b"".join(t["lobs"] for t in ticks), dtype=np.uint8)
    if payload.size != n * 19:
        return []
    payload = payload.reshape(n, 19)

    # Layout (matches QNN_EmitLabelerTick): bytes 0..5 = vel fp16[3];
    # 6 = movement_id u8; 7..12 = view fp16[3]; 13 = c_rule_fire u8;
    # 14 = c_rule_jump u8; 15 = move_packed u8;
    # 16 = target_valid_mask u8; 17 = usercmd_fire u8; 18 = weapon_id u8.
    self_velocity = payload[:, 0:6].copy().view(np.float16).reshape(n, 3)
    movement_id   = payload[:, 6].copy()
    look          = payload[:, 7:13].copy().view(np.float16).reshape(n, 3)
    c_rule_fire   = payload[:, 13].copy()
    c_rule_jump   = payload[:, 14].copy()
    move_packed   = payload[:, 15].copy()
    target_valid_mask = payload[:, 16].copy()
    usercmd_fire  = payload[:, 17].copy()
    weapon_id     = payload[:, 18].copy()

    obs_full = {
        "self_velocity":    self_velocity,
        "self_movement_id": movement_id.astype(np.uint8, copy=False),
        "look":             look,
        "c_rule_fire":      c_rule_fire.astype(np.uint8, copy=False),
        "c_rule_jump":      c_rule_jump.astype(np.uint8, copy=False),
        # Per-axis engine-effective bitmask (bit0=fb, bit1=lr, bit2=ud,
        # bit3=fire, bit4=weapon).  Stored alongside obs so the trainer
        # can opt into sanitizing targets via --sanitize-targets; ignored
        # otherwise.
        "target_valid_mask": target_valid_mask.astype(np.uint8, copy=False),
        # True press signal (usercmd.fire OR'd across the cmd window).
        # Used to verify the mask's first-press cooldown gate against
        # truth; also useful as apply-time obs.
        "usercmd_fire":    usercmd_fire.astype(np.uint8, copy=False),
        # Currently-held weapon byte (1..8 axe..LG; 0 if no weapon).
        # Needed for weapon-keyed cooldown gating in
        # first_shot_per_cooldown; also the dense per-frame weapon target.
        "weapon_id":       weapon_id.astype(np.uint8, copy=False),
    }
    act_full = {"move": move_packed.astype(np.uint8, copy=False)}

    if labels and drop_label_names:
        keep = _label_keep_mask(labels, drop_label_names, total_frames, n)
        runs = _runs_from_mask(keep)
    else:
        runs = [(0, n)]
    if not runs:
        return []

    episodes: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for s, e in runs:
        sub_obs = {key: values[s:e] for key, values in obs_full.items()}
        sub_act = {key: values[s:e] for key, values in act_full.items()}
        episodes.append((sub_obs, sub_act))
    return episodes


# ── per-worker collect ────────────────────────────────────────────────────────

def _collect_demo_slim(
    args: tuple[str, int, int, bool, dict, int, tuple[str, ...]],
) -> dict:
    """Pool-worker entry: collect one demo into slim labeler sub-episodes.

    Returns the same ``{"demo", "status", "ticks", "episodes":
    [{"obs", "actions", "rows"}, ...]}`` shape as
    qnn.bc.collect._collect_demo, so qnn.bc.collect.run_collect can
    dispatch this strategy interchangeably with the BC one.
    """
    (demo_name, play_start, play_end, force_mvd_emit,
     labels, total_frames, drop_label_names) = args
    proc = None

    try:
        proc = _get_collect_worker()
        ticks = _collect_one_demo_labeler(
            proc, demo_name,
            play_start=play_start, play_end=play_end,
            force_mvd_emit=force_mvd_emit)
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
        return {"demo": demo_name, "status": "crash", "msg": err_tail or "worker crash"}

    # Match bc_collect's per-demo worker shutdown to avoid engine state leaks.
    _shutdown_worker()

    if len(ticks) < 10:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}

    episodes = _unpack_labeler_episode(
        ticks,
        labels=labels,
        drop_label_names=drop_label_names,
        total_frames=total_frames,
    )
    # Translate (obs, acts) tuples to the shape run_collect expects,
    # dropping empty sub-episodes; order preserved so episode_idx is
    # stable across collects given the same labels + drop_label_names.
    sized: list[dict] = []
    for obs, acts in episodes:
        rows = int(np.asarray(acts["move"]).shape[0])
        if rows <= 0:
            continue
        sized.append({"obs": obs, "actions": acts, "rows": rows})
    if not sized:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}
    return {
        "demo": demo_name, "status": "ok", "ticks": len(ticks),
        "episodes": sized,
    }


# ── corpus filter ─────────────────────────────────────────────────────────────

def _load_rate_lookup(rate_tsv: Path) -> dict[str, int]:
    rates: dict[str, int] = {}
    with open(rate_tsv) as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                try:
                    rates[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return rates


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir",   type=Path, required=True)
    ap.add_argument("--manifest",   type=Path, default=None,
                    help="Corpus manifest .ndjson (default: <demo-dir>/../<demo-dir.name>_manifest.ndjson)")
    ap.add_argument("--rate-tsv",   type=Path,
                    default=Path("artifacts/qwd_rate_histogram.tsv"))
    ap.add_argument("--output",     type=Path, required=True)
    ap.add_argument("--asset-root", type=Path, default=Path("assets"))
    ap.add_argument("--demo-worker", type=Path, default=None,
                    help="qw_demo_worker binary (default: assets/bin/qw_demo_worker)")
    ap.add_argument("--workers",    type=int, default=30)
    ap.add_argument("--shard-rows", type=int, default=200_000,
                    help="Rows per shard. At native ~77Hz, 200k rows ~= 43 minutes of demo time.")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed",       type=int, default=17)
    ap.add_argument("--min-hz",     type=int, default=70,
                    help="Drop demos detected below this native rate "
                         "(labeler-specific, on top of --filter-config)")
    ap.add_argument("--force-mvd-emit", action="store_true",
                    help="Run QWD demos through the MVD inference path so "
                         "action.fire is C-rule (sound+ammo) rather than "
                         "button-truth — matches the apply-time distribution.")
    ap.add_argument("--filter-config", type=Path, default=None,
                    help="Path to a JSON filter config (same schema as "
                         "qnn.bc.collect --filter-config).  Demo-level "
                         "keep / drop predicates gate inclusion; "
                         "drop_tick_labels carves sub-episodes out of each "
                         "kept demo.  Point at artifacts/collect/qwd/"
                         "filter.json to mirror the BC corpus.")
    args = ap.parse_args()

    if args.filter_config is not None:
        try:
            filter_spec = json.loads(Path(args.filter_config).read_text())
            _validate_filter_schema(filter_spec)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            sys.exit(f"--filter-config {args.filter_config!r}: {exc}")
    else:
        filter_spec = {}
    keep_pred = filter_spec.get("keep") or {}
    drop_pred = filter_spec.get("drop") or {}
    drop_label_names = tuple(filter_spec.get("drop_tick_labels") or ())

    asset_root = args.asset_root.resolve()
    demo_dir = args.demo_dir.resolve()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = (args.manifest if args.manifest
                     else demo_dir.parent / f"{demo_dir.name}_manifest.ndjson")
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")
    if not args.rate_tsv.exists():
        sys.exit(f"rate TSV not found: {args.rate_tsv} (run scripts/qwd_rate_histogram.py)")

    demo_worker = args.demo_worker or Path(_default_demo_worker("qwd"))
    if not Path(demo_worker).is_absolute():
        demo_worker = Path.cwd() / demo_worker
    if not demo_worker.exists():
        sys.exit(f"demo worker not found: {demo_worker}")
    game_dir = _game_dir_for_demo_dir(demo_dir, asset_root)

    # Labeler-specific native-rate gate, applied on top of --filter-config.
    # Loaded once and closed over so run_collect can call it per-entry.
    rate_lookup = _load_rate_lookup(args.rate_tsv)
    min_hz = args.min_hz
    def rate_gate(entry: dict) -> str | None:
        hz = rate_lookup.get(entry["file"])
        if hz is None:
            return "rate_unknown"
        if hz < min_hz:
            return f"rate_below_min:{hz}"
        return None

    force_mvd_emit = args.force_mvd_emit
    def build_work_args(entry, labels, total_frames, drop_labels):
        ps = int(entry.get("play_start", 0))
        pe = int(entry.get("play_end", 999_999_999))
        return (entry["file"], ps, pe, force_mvd_emit,
                labels, total_frames, drop_labels)

    run_collect(
        output=output,
        demo_dir=demo_dir,
        manifest_path=manifest_path,
        asset_root=asset_root,
        demo_worker=str(demo_worker),
        game_dir=game_dir,
        # tick_hz=0 → engine resolves to detected native rate per demo
        # (see qnn_collect_main.c).
        tick_hz=0,
        workers=args.workers,
        shard_rows=args.shard_rows,
        train_ratio=args.train_ratio,
        seed=args.seed,
        keep_pred=keep_pred,
        drop_pred=drop_pred,
        drop_label_names=drop_label_names,
        per_demo_fn=_collect_demo_slim,
        build_work_args=build_work_args,
        shard_kind="labeler",
        extra_demo_filter=rate_gate,
        extra_metadata={
            "format": "labeler_v2",
            "force_mvd_emit": force_mvd_emit,
            "min_hz": min_hz,
            "filter_config": (str(args.filter_config)
                              if args.filter_config else None),
            "drop_tick_labels": list(drop_label_names),
        },
        filter_path=Path(args.filter_config) if args.filter_config else None,
    )


if __name__ == "__main__":
    main()
