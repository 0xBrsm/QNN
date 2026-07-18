#!/usr/bin/env python3
"""Deliverable 2 — HUMAN reference fingerprints for all control channels.

Computes the human (demo-label) temporal-statistic fingerprints — dwell-time,
switch-rate, inter-event-interval, turn magnitude — for every channel
(move fb/lr/ud, target, weapon, attack, look), in-distribution and
episode-boundary-respecting. Pure numpy, no model, no torch — runs locally.

In-distribution loading mirrors scripts/analysis/move_momentum_baseline.py:
per-shard, per-episode (episode_lengths), keep = (1 - target_probs[:,0]) != 0,
the ud feasibility rewrite from the input_mask bits, and the inlined packed-move
decode (bit 0 = attack; fb bits 1-2; lr bits 3-4; ud_neg bit 5, ud_pos bit 6|7).

Action label formats:
  move    packed byte -> per-axis class {0=neg,1=none,2=pos} (decode above)
  attack  packed byte bit 0 -> {0,1}
  target  argmax over act_target_probs (17-dim; index 0 = no-target)
  weapon  act_weapon impulse byte (1..8: AXE,SHOTGUN,SSG,NG,SNG,GL,RL,LG)
  look    act_look view-relative unit vector (3,); turn magnitude =
          degrees(arccos(clip(unit[0],-1,1))); turn bout = run of frames >=15deg.

Writes runs/head_probe/_humanlikeness_human_reference.json and prints a summary
table (mean/median dwell + switch-rate per channel).

Usage:
  PYTHONPATH=src python scripts/analysis/humanlikeness_human_reference.py \
      --data-dir artifacts/collect/qwd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from qnn.eval.humanlikeness.core import (  # noqa: E402
    dwell_times,
    switch_rate,
    inter_event_intervals,
    onset_intervals,
    describe,
)

# Canonical output path for the human reference fingerprint JSON.  Defined once
# here so shell harnesses and sibling scripts can reference this constant instead
# of hard-coding the path string independently.
DEFAULT_HUMAN_REFERENCE = Path("runs/head_probe/_humanlikeness_human_reference.json")

AXES = ("fb", "lr", "ud")
FLICK_DEG = 15.0
WEAPON_NAMES = {1: "AXE", 2: "SHOTGUN", 3: "SSG", 4: "NG", 5: "SNG",
                6: "GL", 7: "RL", 8: "LG"}


def _unpack_move(packed: np.ndarray) -> np.ndarray:
    """Packed move byte -> (T,3) class indices [fb,lr,ud] (mirror _unpack_move_axes)."""
    a = np.asarray(packed, dtype=np.uint8).reshape(-1)
    fb = 1 + ((a >> 2) & 1).astype(np.int64) - ((a >> 1) & 1).astype(np.int64)
    lr = 1 + ((a >> 4) & 1).astype(np.int64) - ((a >> 3) & 1).astype(np.int64)
    ud_pos = (((a >> 6) & 1) | ((a >> 7) & 1)).astype(np.int64)
    ud = 1 + ud_pos - ((a >> 5) & 1).astype(np.int64)
    return np.stack([fb, lr, ud], axis=-1)


def _unpack_attack(packed: np.ndarray) -> np.ndarray:
    return (np.asarray(packed, dtype=np.uint8).reshape(-1) & 0x1).astype(np.int64)


def _ud_rewrite(move: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).reshape(-1).astype(np.int64)
    out = move.copy()
    up_neg = ((m >> 5) & 1) != 0
    up_pos = ((m >> 6) & 1) != 0
    jump = ((m >> 7) & 1) != 0
    ud = move[:, 2]
    out[:, 2] = np.where((ud == 2) & (jump | up_pos), 2,
                         np.where((ud == 0) & up_neg, 0, 1))
    return out


def _turn_deg(look: np.ndarray) -> np.ndarray:
    """View-relative unit vector (T,3) -> per-frame turn magnitude in degrees."""
    look = np.asarray(look, dtype=np.float32)
    n = np.linalg.norm(look, axis=1)
    u0 = np.where(n > 1e-8, look[:, 0] / np.maximum(n, 1e-8), 1.0)
    return np.degrees(np.arccos(np.clip(u0, -1.0, 1.0)))


def _episodes(root: Path, split: str, *, with_demo: bool = False):
    """Yield per-episode dict of in-distribution channel labels (already rewritten).

    ``with_demo=True`` adds the episode's source ``demo`` index (from the shard's
    ``demo_idxs``) so callers can group/resample at the demo level (e.g. the
    human_band.py split-half null).
    """
    manifest = json.loads((root / split / "manifest.json").read_text())
    for shard in manifest["shards"]:
        a = shard["actions"]
        packed = np.asarray(np.load(root / split / a["move"], mmap_mode="r"))
        imask = np.asarray(np.load(root / split / a["input_mask"], mmap_mode="r")).reshape(-1)
        move = _ud_rewrite(_unpack_move(packed), imask)
        attack = _unpack_attack(packed)
        weapon = np.asarray(np.load(root / split / a["weapon"], mmap_mode="r"), dtype=np.int64).reshape(-1)
        tp = np.asarray(np.load(root / split / a["target_probs"], mmap_mode="r"), dtype=np.float32)
        target = np.argmax(tp, axis=1).astype(np.int64)
        keep = (1.0 - tp[:, 0]) != 0.0
        turn = _turn_deg(np.asarray(np.load(root / split / a["look"], mmap_mode="r")))
        # attack operativeness (input_mask bit0): raw attack is held-button
        # STATE; only op frames are decisions. Model<->human attack rate
        # comparisons must condition on this (op-filter doctrine).
        atk_op = (imask & 1) != 0
        demo_idxs = shard.get("demo_idxs") or [None] * len(shard["episode_lengths"])
        start = 0
        for ei, length in enumerate(shard["episode_lengths"]):
            stop = start + int(length)
            sl = slice(start, stop)
            ep = {
                "move": move[sl], "attack": attack[sl], "weapon": weapon[sl],
                "target": target[sl], "turn": turn[sl], "keep": keep[sl],
                "attack_op": atk_op[sl],
            }
            if with_demo:
                ep["demo"] = demo_idxs[ei]
            yield ep
            start = stop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="precomputed_val")
    ap.add_argument("--out", type=Path, default=DEFAULT_HUMAN_REFERENCE)
    args = ap.parse_args()

    # Accumulators ---------------------------------------------------------
    move_dwell = {ax: {0: [], 1: [], 2: []} for ax in AXES}  # dwell per held class
    move_dwell_all = {ax: [] for ax in AXES}                  # any-class dwell
    move_iei = {ax: [] for ax in AXES}
    move_sw = {ax: [0, 0] for ax in AXES}                     # [n_switch, n_trans]

    tgt_dwell_locked = []     # dwell of a specific locked target id (id!=0)
    tgt_dwell_engaged = []    # dwell of any engaged (id!=0) run, regardless of id
    tgt_id_iei = []           # re-target interval (between target-id switches)
    tgt_sw = [0, 0]           # re-target switch-rate over target-id stream

    wpn_dwell = {}            # dwell per equipped weapon id
    wpn_dwell_all = []
    wpn_switch_iei = []       # weapon-switch interval
    wpn_sw = [0, 0]

    atk_runs = []             # sustained-fire run lengths (attack==1)
    atk_onset_iei = []        # fire-onset interval
    atk_sw = [0, 0]

    turn_all = []             # per-frame turn magnitude (deg), in-dist
    bout_dur = []             # turn-bout (>=15deg) durations

    n_frames = 0
    n_episodes = 0

    for ep in _episodes(args.data_dir, args.split):
        keep = ep["keep"]
        n_frames += int(keep.sum())
        n_episodes += 1

        # move (per axis) --------------------------------------------------
        for ai, ax in enumerate(AXES):
            lab = ep["move"][:, ai]
            for cls in (0, 1, 2):
                d = dwell_times(lab, keep, only_value=cls)
                if d.size:
                    move_dwell[ax][cls].append(d)
            da = dwell_times(lab, keep)
            if da.size:
                move_dwell_all[ax].append(da)
            iei = inter_event_intervals(lab, keep)
            if iei.size:
                move_iei[ax].append(iei)
            _, ns, nt = switch_rate(lab, keep)
            move_sw[ax][0] += ns
            move_sw[ax][1] += nt

        # target -----------------------------------------------------------
        tgt = ep["target"]
        # locked-target dwell: runs of a constant id, engaged only (id != 0) —
        # how long a SPECIFIC enemy is held before a re-target or disengage.
        d_lock = dwell_times(tgt, keep, exclude_value=0)
        if d_lock.size:
            tgt_dwell_locked.append(d_lock)
        # target-present dwell: how long ANY engagement lasts (any id != 0),
        # merging consecutive id-switches that never pass through no-target.
        engaged = (tgt != 0).astype(np.int64)
        d_eng = dwell_times(engaged, keep, only_value=1)
        if d_eng.size:
            tgt_dwell_engaged.append(d_eng)
        iei = inter_event_intervals(tgt, keep)
        if iei.size:
            tgt_id_iei.append(iei)
        _, ns, nt = switch_rate(tgt, keep)
        tgt_sw[0] += ns
        tgt_sw[1] += nt

        # weapon -----------------------------------------------------------
        wpn = ep["weapon"]
        for w in np.unique(wpn[keep]):
            d = dwell_times(wpn, keep, only_value=int(w))
            if d.size:
                wpn_dwell.setdefault(int(w), []).append(d)
        da = dwell_times(wpn, keep)
        if da.size:
            wpn_dwell_all.append(da)
        iei = inter_event_intervals(wpn, keep)
        if iei.size:
            wpn_switch_iei.append(iei)
        _, ns, nt = switch_rate(wpn, keep)
        wpn_sw[0] += ns
        wpn_sw[1] += nt

        # attack -----------------------------------------------------------
        atk = ep["attack"]
        r = dwell_times(atk, keep, only_value=1)   # sustained-fire run lengths
        if r.size:
            atk_runs.append(r)
        oi = onset_intervals(atk, keep, onset_value=1)
        if oi.size:
            atk_onset_iei.append(oi)
        _, ns, nt = switch_rate(atk, keep)
        atk_sw[0] += ns
        atk_sw[1] += nt

        # look -------------------------------------------------------------
        turn = ep["turn"]
        turn_all.append(turn[keep])
        flick = (turn >= FLICK_DEG).astype(np.int64)
        bd = dwell_times(flick, keep, only_value=1)  # bout = run of >=15deg frames
        if bd.size:
            bout_dur.append(bd)

    cat = lambda L: np.concatenate(L) if L else np.empty(0, dtype=np.int64)  # noqa: E731

    # Assemble fingerprint -------------------------------------------------
    def rate(pair):
        return round(pair[0] / pair[1], 6) if pair[1] else None

    fp: dict = {
        "_meta": {
            "split": args.split,
            "data_dir": str(args.data_dir),
            "n_episodes": n_episodes,
            "in_dist_frames": n_frames,
            "flick_deg": FLICK_DEG,
            "note": "in-distribution (1-target_probs[:,0]!=0), episode-boundary-respecting; "
                    "dwell/iei units = frames; turn = degrees.",
        },
        "move": {},
        "target": {},
        "weapon": {},
        "attack": {},
        "look": {},
    }

    for ax in AXES:
        fp["move"][ax] = {
            "switch_rate": rate(move_sw[ax]),
            "n_switches": move_sw[ax][0],
            "n_transitions": move_sw[ax][1],
            "dwell_all": describe(cat(move_dwell_all[ax])),
            "dwell_by_class": {
                str(c): describe(cat(move_dwell[ax][c])) for c in (0, 1, 2)
            },
            "inter_event_interval": describe(cat(move_iei[ax])),
        }

    fp["target"] = {
        "retarget_switch_rate": rate(tgt_sw),
        "n_switches": tgt_sw[0],
        "n_transitions": tgt_sw[1],
        "locked_target_dwell": describe(cat(tgt_dwell_locked)),
        "target_present_dwell": describe(cat(tgt_dwell_engaged)),
        "retarget_interval": describe(cat(tgt_id_iei)),
    }

    fp["weapon"] = {
        "switch_rate": rate(wpn_sw),
        "n_switches": wpn_sw[0],
        "n_transitions": wpn_sw[1],
        "dwell_all": describe(cat(wpn_dwell_all)),
        "dwell_by_weapon": {
            WEAPON_NAMES.get(w, str(w)): describe(cat(d)) for w, d in sorted(wpn_dwell.items())
        },
        "weapon_switch_interval": describe(cat(wpn_switch_iei)),
    }

    fp["attack"] = {
        "switch_rate": rate(atk_sw),
        "n_switches": atk_sw[0],
        "n_transitions": atk_sw[1],
        "sustained_fire_run": describe(cat(atk_runs)),
        "fire_onset_interval": describe(cat(atk_onset_iei)),
    }

    turn_cat = cat(turn_all).astype(np.float64)
    fp["look"] = {
        "turn_magnitude_deg": describe(turn_cat),
        "flick_rate_ge15deg": round(float((turn_cat >= FLICK_DEG).mean()), 6) if turn_cat.size else None,
        "turn_bout_duration": describe(cat(bout_dur)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fp, indent=2))

    # Summary table --------------------------------------------------------
    print(f"\nHUMAN reference fingerprint  ({args.split})")
    print(f"  episodes={n_episodes:,}  in-dist frames={n_frames:,}\n")
    print(f"{'channel':<22} {'switch_rate':>12} {'dwell_mean':>11} {'dwell_med':>10}")
    print("-" * 58)

    def row(name, sw, dw):
        sr = f"{sw:.5f}" if sw is not None else "n/a"
        dm = f"{dw['mean']:.2f}" if dw["mean"] is not None else "n/a"
        dd = f"{dw['median']:.1f}" if dw["median"] is not None else "n/a"
        print(f"{name:<22} {sr:>12} {dm:>11} {dd:>10}")

    for ax in AXES:
        row(f"move.{ax}", fp["move"][ax]["switch_rate"], fp["move"][ax]["dwell_all"])
    row("target (id)", fp["target"]["retarget_switch_rate"], fp["target"]["locked_target_dwell"])
    row("weapon", fp["weapon"]["switch_rate"], fp["weapon"]["dwell_all"])
    row("attack (fire run)", fp["attack"]["switch_rate"], fp["attack"]["sustained_fire_run"])
    row("look (turn-bout)", None, fp["look"]["turn_bout_duration"])
    print(f"\nlook turn-magnitude (deg): mean={fp['look']['turn_magnitude_deg']['mean']}  "
          f"p90={fp['look']['turn_magnitude_deg']['p90']}  "
          f"flick_rate>=15deg={fp['look']['flick_rate_ge15deg']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
