"""Native-rate slim collect for the move labeler.

Walks a QWD corpus, drives qw_demo_worker at native rate (tick_hz=0),
and writes per-tick arrays for labeler training:

  obs/self_velocity        fp16  (T, 3)   body-frame, pre-normalized in C
  obs/self_movement_id     uint8 (T,)
  obs/cmd_angles           int16 (T, 3)   usercmd view angles, QW 65536/360
                                          quantization (pitch/yaw/roll)
  obs/cmd_move             int16 (T, 3)   usercmd fb/lr/ud aggregated across
                                          cmd window (mean fb/lr; jump-or-
                                          most-negative ud), raw QW units
  obs/cmd_buttons          uint8 (T,)     usercmd buttons OR'd across cmd
                                          window (bit0=fire, bit1=button2)
  obs/cmd_impulse          uint8 (T,)     usercmd last non-zero impulse in
                                          cmd window (matches engine)
  obs/op_input             uint8 (T,)     strict per-axis op mask of the
                                          cmd_* fields (bit0=fb..bit4=
                                          impulse).  1 = press AND engine
                                          acted on it this tick.  QC VM is
                                          source of truth for bits 2/3/4.
  obs/weapon_id            uint8 (T,)     server-held weapon byte (1..8;
                                          0 = none).  Lags impulses by
                                          the cmd-pipeline delay.
  obs/c_rule_fire          uint8 (T,)     sound-derived self weapon-fire
                                          bit (PHS multicast).  Sparse
                                          one-tick-per-event — same
                                          generator MVD inference uses
                                          at apply time.
  obs/c_rule_jump          uint8 (T,)     sound-derived self plyrjmp8
                                          bit.  Same role as c_rule_fire.
  obs/look                 fp16  (T, 3)   per-tick view delta — derived
                                          here from cmd_angles + prev-tick
                                          anchor basis.  Trainer-facing
                                          feature (Quake AngleVectors port).
  actions/move             uint8 (T,)     3-class fb|lr<<2|ud<<4 derived
                                          from cmd_move via threshold.
                                          Trainer-facing target.

Demo-level filtering goes through `--filter-config` (same JSON schema as
qnn.bc.collect: nested `demos` / `segments` / `tokens` / `actions` axes
with `keep` / `drop` sub-keys).  Pointing it at
`artifacts/collect/qwd/filter.json` selects the same demo set the BC
corpus uses.  `segments.drop` intervals (signon / dead / intermission)
carve sub-episodes out of each demo so the labeler's bidirectional
context never bridges a dropped interval — same semantics as bc.collect.

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
    _default_demo_worker,
    _game_dir_for_demo_dir,
    _label_keep_mask,
    _load_and_pin_filter,
    _read_collect_frames,
    _run_per_demo_collect,
    _runs_from_mask,
    run_collect,
)


# ── LOBS stream IO ────────────────────────────────────────────────────────────
#
# LOBS framing (matches QNN_EmitLabelerTick in qnn_labeler_collect.c):
#   "LOBS"                4 bytes magic
#   tick u32              4
#   tick_hz u32           4
#   flags u16             2
#   payload              25  (see qnn.wire.unpack_labeler_buffer)
# Total: 39 bytes per native tick.

_LOBS_HEADER_SIZE = 10
_LOBS_PAYLOAD_SIZE = 25
_LOBS_FRAME_AFTER_MAGIC = _LOBS_HEADER_SIZE + _LOBS_PAYLOAD_SIZE


# ── derivations from raw usercmd ───────────────────────────────────────────────

# Threshold for "is there a press on this axis" (in raw QW move units).
# QW's cl_forwardspeed default is ~200 (run forward unmoded); +speed activation
# doubles to 400.  Anything within ±_MOVE_PRESS_THRESHOLD counts as "none"
# (no press) — keeps a small dead zone around zero for averaged values to
# settle on, matching the prior 3-class derivation's 0.1 normalized threshold
# (= 0.1 × QNN_SV_MAXSPEED = ~32 in raw units).
_MOVE_PRESS_THRESHOLD = 32


def _decode_angles(cmd_angles_int16: np.ndarray) -> np.ndarray:
    """Decode int16 QW-quantized angles to degrees in [0, 360).  Treats the
    int16 pattern as unsigned 16-bit (matching MSG_ReadAngle16's modular
    encoding) then scales by 360/65536."""
    return (cmd_angles_int16.astype(np.int32) & 0xFFFF).astype(
        np.float32) * (360.0 / 65536.0)


def _angles_to_basis(angles_deg: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quake's AngleVectors, vectorized over (T, 3) input.  Returns
    (forward, right, up) each shape (T, 3)."""
    pitch = np.deg2rad(angles_deg[:, 0])
    yaw   = np.deg2rad(angles_deg[:, 1])
    roll  = np.deg2rad(angles_deg[:, 2])
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    sr, cr = np.sin(roll), np.cos(roll)
    forward = np.stack([cp * cy, cp * sy, -sp], axis=1)
    right = np.stack([
        -sr * sp * cy + cr * sy,
        -sr * sp * sy - cr * cy,
        -sr * cp,
    ], axis=1)
    up = np.stack([
        cr * sp * cy + sr * sy,
        cr * sp * sy - sr * cy,
        cr * cp,
    ], axis=1)
    return forward, right, up


def _derive_view_delta(cmd_angles_int16: np.ndarray) -> np.ndarray:
    """Per-tick body-frame view delta from cmd_angles.  Mirrors
    QNN_FillLookAndSwitch in C: look = cur_forward · prev-anchor-basis,
    where the anchor is the previous emit's angles (= previous tick in
    labeler mode, since every native tick emits).

    Tick 0 has no anchor → (0, 0, 0)."""
    if cmd_angles_int16.shape[0] < 2:
        return np.zeros((cmd_angles_int16.shape[0], 3), dtype=np.float16)
    angles_deg = _decode_angles(cmd_angles_int16)
    forward, right, up = _angles_to_basis(angles_deg)
    T = forward.shape[0]
    out = np.zeros((T, 3), dtype=np.float32)
    out[1:, 0] = np.einsum("ij,ij->i", forward[1:], forward[:-1])
    out[1:, 1] = np.einsum("ij,ij->i", forward[1:], right[:-1])
    out[1:, 2] = np.einsum("ij,ij->i", forward[1:], up[:-1])
    return out.astype(np.float16)


def _derive_move_packed(cmd_move_int16: np.ndarray) -> np.ndarray:
    """3-class fb|lr<<2|ud<<4 from raw int16 cmd_move (mean-aggregated
    QW units).  Class 0=neg, 1=none, 2=pos with a ±_MOVE_PRESS_THRESHOLD
    dead zone around zero (matches the prior 0.1-normalized threshold)."""
    def classify(v: np.ndarray) -> np.ndarray:
        return np.where(v >  _MOVE_PRESS_THRESHOLD, 2,
               np.where(v < -_MOVE_PRESS_THRESHOLD, 0, 1)).astype(np.uint8)
    fb = classify(cmd_move_int16[:, 0])
    lr = classify(cmd_move_int16[:, 1])
    ud = classify(cmd_move_int16[:, 2])
    return ((fb & 0x3) | ((lr & 0x3) << 2) | ((ud & 0x3) << 4)).astype(np.uint8)


def _parse_lobs_frame(raw: bytes) -> dict:
    """Parse one LOBS frame (after the 4-byte magic).  Returns a tick
    dict with the native-rate header fields and the raw payload bytes. """
    tick_idx, tick_hz, flags = struct.unpack_from("<IIH", raw, 0)
    return {
        "tick":    int(tick_idx),
        "tick_hz": int(tick_hz),
        "flags":   int(flags),
        "lobs":    raw[_LOBS_HEADER_SIZE:],
        "done":    bool(flags & FLAG_DONE),
    }


def _collect_one_demo_labeler(
    proc: subprocess.Popen,
    demo_name: str,
    play_start: int = 0,
    play_end: int = 999999999,
    force_mvd_emit: bool = False,
) -> list[dict] | None:
    """Dispatch one demo to the worker in labeler_mode and read its LOBS
    stream.  Wraps qnn.bc.collect._read_collect_frames with the
    labeler-specific op (labeler_mode=1), magic (LOBS), and payload
    layout (see ``_parse_lobs_frame``).  The worker always runs the QC
    VM predicates to fill op_input."""
    op = {
        "op": "collect", "demo_path": demo_name, "seed": 0,
        "play_start": play_start, "play_end": play_end,
        "force_mvd_emit":  1 if force_mvd_emit  else 0,
        "labeler_mode":    1,
    }
    return _read_collect_frames(
        proc, op, b"LOBS", _LOBS_FRAME_AFTER_MAGIC, _parse_lobs_frame
    )


# ── slim unpack ───────────────────────────────────────────────────────────────

def _unpack_labeler_episode(
    ticks: list[dict],
    labels: dict | None = None,
    drop_label_names: tuple[str, ...] = (),
    total_frames: int = 0,
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Slim unpack: directly read fields out of the LOBS per-tick payload
    and split into sub-episodes wherever ``segments.drop`` carves a
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
    if payload.size != n * _LOBS_PAYLOAD_SIZE:
        return []
    payload = payload.reshape(n, _LOBS_PAYLOAD_SIZE)

    # Layout (matches QNN_EmitLabelerTick): bytes 0..5 = vel fp16[3];
    # 6 = movement_id u8; 7..12 = cmd_angles int16[3];
    # 13..18 = cmd_move int16[3]; 19 = cmd_buttons u8;
    # 20 = cmd_impulse u8; 21 = op_input u8; 22 = weapon_id u8;
    # 23 = c_rule_fire u8 (sound-derived self weapon-fire bit);
    # 24 = c_rule_jump u8 (sound-derived self plyrjmp8 bit).
    self_velocity = payload[:, 0:6].copy().view(np.float16).reshape(n, 3)
    movement_id   = payload[:, 6].copy()
    cmd_angles    = payload[:, 7:13].copy().view(np.int16).reshape(n, 3)
    cmd_move      = payload[:, 13:19].copy().view(np.int16).reshape(n, 3)
    cmd_buttons   = payload[:, 19].copy()
    cmd_impulse   = payload[:, 20].copy()
    op_input      = payload[:, 21].copy()
    weapon_id     = payload[:, 22].copy()
    c_rule_fire   = payload[:, 23].copy()
    c_rule_jump   = payload[:, 24].copy()

    # Derived trainer-facing fields.  `look` is the per-tick body-frame
    # view delta the labeler model consumes as a feature; `move_packed`
    # is the 3-class fb/lr/ud target.  Both are pure functions of
    # cmd_angles / cmd_move respectively — derived once at collect time
    # so the trainer doesn't have to know about the wire format.
    look          = _derive_view_delta(cmd_angles)
    move_packed   = _derive_move_packed(cmd_move)

    obs_full = {
        "self_velocity":    self_velocity,
        "self_movement_id": movement_id.astype(np.uint8, copy=False),
        # Per-tick body-frame view delta — derived from cmd_angles.
        # Labeler model feature.
        "look":             look,
        # Raw usercmd block — exact QW wire-format precision.
        "cmd_angles":       cmd_angles.astype(np.int16, copy=False),
        "cmd_move":         cmd_move.astype(np.int16, copy=False),
        "cmd_buttons":      cmd_buttons.astype(np.uint8, copy=False),
        "cmd_impulse":      cmd_impulse.astype(np.uint8, copy=False),
        # Strict per-axis op mask of the cmd_* fields.  1 = press AND
        # engine acted on it.  Trainer derives CE-keep mask per axis
        # as (no_press_axis) | (op_input_bit_axis).
        "op_input":         op_input.astype(np.uint8, copy=False),
        # Server-held weapon byte (1..8 axe..LG; 0 if no weapon).  Lags
        # impulses by the cmd-pipeline delay.
        "weapon_id":        weapon_id.astype(np.uint8, copy=False),
        # Sound-derived discrete-cmd reconstructions — same generators
        # MVD inference uses at apply time.  Sparse one-tick-per-event;
        # no hold/chain-fill (those live behind the MVD inference path
        # for BC policy labels, not labeler training features).
        "c_rule_fire":      c_rule_fire.astype(np.uint8, copy=False),
        "c_rule_jump":      c_rule_jump.astype(np.uint8, copy=False),
    }
    act_full = {
        # 3-class fb|lr<<2|ud<<4 derived from cmd_move via threshold.
        # Trainer-facing target.  Sparse weapon-impulse target lives
        # implicitly in obs/cmd_impulse + obs/op_input bit 4.
        "move":   move_packed.astype(np.uint8, copy=False),
    }

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

    Thin wrapper around ``_run_per_demo_collect`` (shared with BC) that
    supplies the labeler-specific collect (LOBS) and unpack (slim
    self-only obs)."""
    (demo_name, play_start, play_end, force_mvd_emit,
     labels, total_frames, drop_label_names) = args
    return _run_per_demo_collect(
        demo_name,
        collect_fn=lambda proc: _collect_one_demo_labeler(
            proc, demo_name,
            play_start=play_start, play_end=play_end,
            force_mvd_emit=force_mvd_emit),
        unpack_fn=lambda ticks: _unpack_labeler_episode(
            ticks, labels=labels,
            drop_label_names=drop_label_names,
            total_frames=total_frames),
    )


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
                         "qnn.bc.collect --filter-config).  demos.keep / "
                         "demos.drop predicates gate inclusion; "
                         "segments.drop carves sub-episodes out of each "
                         "kept demo.  Point at artifacts/collect/qwd/"
                         "filter.json to mirror the BC corpus.")
    args = ap.parse_args()

    asset_root = args.asset_root.resolve()
    demo_dir = args.demo_dir.resolve()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    # Pin the filter to <output>/filter.json so the spec used to produce
    # this cache always travels with it. See _load_and_pin_filter.
    filter_spec = _load_and_pin_filter(
        output, Path(args.filter_config) if args.filter_config else None)
    demos_block    = filter_spec.get("demos") or {}
    segments_block = filter_spec.get("segments") or {}
    keep_pred = demos_block.get("keep") or {}
    drop_pred = demos_block.get("drop") or {}
    drop_label_names = tuple(segments_block.get("drop") or ())

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
            "format": "labeler_v9",
            "force_mvd_emit": force_mvd_emit,
            "min_hz": min_hz,
            "segments_drop": list(drop_label_names),
        },
        filter_path=output / "filter.json",
    )


if __name__ == "__main__":
    main()
