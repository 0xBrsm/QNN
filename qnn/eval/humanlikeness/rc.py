"""Closed-loop human-likeness comparison for rc1/rc2/rc3 using the BC metrics.

Applies the project's distributional human-likeness kernel (scripts/analysis/
humanlikeness.py: dwell-time / switch-rate / inter-event distributions compared
via Wasserstein-EMD + KS + RBF-MMD) to the closed-loop action streams dumped by
rc_3way_frikbot_eval.py, against the human reference computed on-the-fly from the
val corpus (the same decode as humanlikeness_human_reference.py).

Tick-rate handling: the human corpus is labeled at ~20 Hz; rc1/rc3 run at 20 Hz,
rc2 at 10 Hz. All dwell/interval/turn stats are converted to PHYSICAL units
(seconds, deg/sec) so the three RCs are commensurable; switch rate is reported
both per-frame (vs the documented human per-frame values) and per-second.

Usage (inside the trainer container):
  PYTHONPATH=src python scripts/analysis/rc_humanlikeness.py \
      --streams-dir runs/eval/rc_3way_arena_box --data-dir artifacts/collect/qwd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from qnn.eval.humanlikeness.core import (  # noqa: E402
    dwell_times, switch_rate, onset_intervals, compare_samples,
    mmd2_rbf_permutation_test, describe,
)
from qnn.eval.humanlikeness.human_reference import _episodes, AXES, FLICK_DEG  # noqa: E402

HUMAN_HZ = 20.0  # qwd corpus is labeled/downsampled to 20 Hz (labeler GBT + relabel table)


def collect_human(data_dir: Path, split: str) -> dict:
    """Raw human samples per channel (engaged frames), units = frames."""
    move_dwell = {ax: [] for ax in AXES}
    move_sw = {ax: [0, 0] for ax in AXES}
    wpn_dwell, wpn_sw = [], [0, 0]
    atk_run, atk_onset, atk_sw = [], [], [0, 0]
    turn_all = []
    for ep in _episodes(data_dir, split):
        keep = ep["keep"]
        for ai, ax in enumerate(AXES):
            lab = ep["move"][:, ai]
            d = dwell_times(lab, keep)
            if d.size: move_dwell[ax].append(d)
            _, ns, nt = switch_rate(lab, keep); move_sw[ax][0] += ns; move_sw[ax][1] += nt
        d = dwell_times(ep["attack_context"], keep)
        if d.size: wpn_dwell.append(d)
        _, ns, nt = switch_rate(ep["attack_context"], keep); wpn_sw[0] += ns; wpn_sw[1] += nt
        r = dwell_times(ep["attack"], keep, only_value=1)
        if r.size: atk_run.append(r)
        oi = onset_intervals(ep["attack"], keep, onset_value=1)
        if oi.size: atk_onset.append(oi)
        _, ns, nt = switch_rate(ep["attack"], keep); atk_sw[0] += ns; atk_sw[1] += nt
        turn_all.append(ep["turn"][keep])
    cat = lambda L: np.concatenate(L) if L else np.empty(0)  # noqa: E731
    return {
        "hz": HUMAN_HZ,
        "move_dwell": {ax: cat(move_dwell[ax]) for ax in AXES},
        "move_switch": {ax: (move_sw[ax][0] / move_sw[ax][1] if move_sw[ax][1] else 0.0) for ax in AXES},
        "weapon_dwell": cat(wpn_dwell),
        "weapon_switch": wpn_sw[0] / wpn_sw[1] if wpn_sw[1] else 0.0,
        "attack_run": cat(atk_run), "attack_onset": cat(atk_onset),
        "attack_switch": atk_sw[0] / atk_sw[1] if atk_sw[1] else 0.0,
        "turn": cat(turn_all).astype(np.float64),
        "flick_ge15": float((cat(turn_all) >= FLICK_DEG).mean()) if turn_all else 0.0,
    }


def collect_bot(npz_path: Path) -> dict:
    """Raw bot samples per channel (engaged frames = actor visible), units = frames."""
    z = np.load(npz_path)
    hz = float(z["tick_hz"][0])
    offs = z["episode_offsets"]
    move = {ax: z[ax] for ax in AXES}
    wpn, atk, turn, keep = z["weapon"], z["attack"], z["turn_deg"], z["keep"]
    md = {ax: [] for ax in AXES}; msw = {ax: [0, 0] for ax in AXES}
    wd, wsw = [], [0, 0]
    ar, ao, asw = [], [], [0, 0]
    turn_keep = []
    for i in range(len(offs) - 1):
        sl = slice(int(offs[i]), int(offs[i + 1]))
        k = keep[sl]
        for ax in AXES:
            lab = move[ax][sl]
            d = dwell_times(lab, k)
            if d.size: md[ax].append(d)
            _, ns, nt = switch_rate(lab, k); msw[ax][0] += ns; msw[ax][1] += nt
        d = dwell_times(wpn[sl], k)
        if d.size: wd.append(d)
        _, ns, nt = switch_rate(wpn[sl], k); wsw[0] += ns; wsw[1] += nt
        r = dwell_times(atk[sl], k, only_value=1)
        if r.size: ar.append(r)
        oi = onset_intervals(atk[sl], k, onset_value=1)
        if oi.size: ao.append(oi)
        _, ns, nt = switch_rate(atk[sl], k); asw[0] += ns; asw[1] += nt
        turn_keep.append(turn[sl][k])
    cat = lambda L: np.concatenate(L) if L else np.empty(0)  # noqa: E731
    return {
        "hz": hz,
        "move_dwell": {ax: cat(md[ax]) for ax in AXES},
        "move_switch": {ax: (msw[ax][0] / msw[ax][1] if msw[ax][1] else 0.0) for ax in AXES},
        "weapon_dwell": cat(wd),
        "weapon_switch": wsw[0] / wsw[1] if wsw[1] else 0.0,
        "attack_run": cat(ar), "attack_onset": cat(ao),
        "attack_switch": asw[0] / asw[1] if asw[1] else 0.0,
        "turn": cat(turn_keep).astype(np.float64),
        "flick_ge15": float((cat(turn_keep) >= FLICK_DEG).mean()) if turn_keep else 0.0,
    }


def cmp_dist(human_frames, human_hz, bot_frames, bot_hz, mmd=True):
    """EMD/KS/MMD on a dwell/interval distribution in SECONDS (rate-fair)."""
    h = np.asarray(human_frames, float) / human_hz
    b = np.asarray(bot_frames, float) / bot_hz
    c = compare_samples(h, b)
    out = {"emd_sec": round(c.emd, 4), "ks": round(c.ks_stat, 4),
           "human": describe(h), "bot": describe(b)}
    if mmd and h.size >= 2 and b.size >= 2:
        m = mmd2_rbf_permutation_test(h, b, n_perm=300)
        out["mmd2"] = round(m["mmd2"], 5); out["mmd_p"] = round(m["p_value"], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="precomputed_val")
    args = ap.parse_args()

    print("collecting human reference samples ...", flush=True)
    H = collect_human(args.data_dir, args.split)
    Hhz = H["hz"]

    rcs = {}
    for rc in ("rc1", "rc2", "rc3"):
        p = args.streams_dir / f"rc_{rc}_streams.npz"
        if p.exists():
            print(f"collecting {rc} ...", flush=True)
            rcs[rc] = collect_bot(p)
    order = [r for r in ("rc1", "rc2", "rc3") if r in rcs]

    result = {"human_hz": Hhz, "channels": {}}
    for rc in order:
        B = rcs[rc]
        ch = {
            "tick_hz": B["hz"],
            "move": {ax: {
                "switch_per_frame": {"human": round(H["move_switch"][ax], 4), "bot": round(B["move_switch"][ax], 4)},
                "switch_per_sec": {"human": round(H["move_switch"][ax] * Hhz, 3), "bot": round(B["move_switch"][ax] * B["hz"], 3)},
                "dwell": cmp_dist(H["move_dwell"][ax], Hhz, B["move_dwell"][ax], B["hz"]),
            } for ax in AXES},
            "weapon": {
                "switch_per_frame": {"human": round(H["weapon_switch"], 4), "bot": round(B["weapon_switch"], 4)},
                "switch_per_sec": {"human": round(H["weapon_switch"] * Hhz, 3), "bot": round(B["weapon_switch"] * B["hz"], 3)},
                "dwell": cmp_dist(H["weapon_dwell"], Hhz, B["weapon_dwell"], B["hz"]),
            },
            "attack": {
                "switch_per_frame": {"human": round(H["attack_switch"], 4), "bot": round(B["attack_switch"], 4)},
                "fire_run": cmp_dist(H["attack_run"], Hhz, B["attack_run"], B["hz"]),
                "fire_onset_interval": cmp_dist(H["attack_onset"], Hhz, B["attack_onset"], B["hz"]),
            },
            "look": {
                "flick_ge15deg": {"human": round(H["flick_ge15"], 4), "bot": round(B["flick_ge15"], 4)},
                "turn_deg_per_sec": cmp_dist(H["turn"], Hhz, B["turn"], B["hz"]),
            },
        }
        result["channels"][rc] = ch

    (args.streams_dir / "humanlikeness.json").write_text(json.dumps(result, indent=2))

    # ---- markdown ----
    L = ["# Closed-loop human-likeness — rc1/rc2/rc3 vs human (BC distributional metrics)\n"]
    L.append(f"Human reference: {args.split} corpus @ {Hhz:.0f} Hz. "
             "EMD on dwell/interval distributions is in **seconds** (rate-fair across the 10/20 Hz split). "
             "MMD p-value: small = distinguishable from human; large = indistinguishable.\n")
    L.append(f"Engaged-frame keep: human = target-present; bot = actor-token visible.\n")

    def emd(rc, *path):
        d = result["channels"][rc]
        for k in path: d = d[k]
        return d
    hdr = "| metric | " + " | ".join(f"{r} ({rcs[r]['hz']:.0f}Hz)" for r in order) + " |"
    sep = "|" + "---|" * (len(order) + 1)

    L.append("## Switch rate vs human (per-frame; human in parens)\n")
    L.append(hdr); L.append(sep)
    for ax in AXES:
        hv = emd(order[0], 'move', ax, 'switch_per_frame')['human']
        cells = [f"{emd(r,'move',ax,'switch_per_frame')['bot']:.3f}" for r in order]
        L.append(f"| move.{ax} (human {hv:.3f}) | " + " | ".join(cells) + " |")
    hv = emd(order[0], 'weapon', 'switch_per_frame')['human']
    L.append(f"| weapon (human {hv:.3f}) | " + " | ".join(f"{emd(r,'weapon','switch_per_frame')['bot']:.3f}" for r in order) + " |")
    hv = emd(order[0], 'attack', 'switch_per_frame')['human']
    L.append(f"| attack (human {hv:.3f}) | " + " | ".join(f"{emd(r,'attack','switch_per_frame')['bot']:.3f}" for r in order) + " |")
    hv = emd(order[0], 'look', 'flick_ge15deg')['human']
    L.append(f"| look flick>=15° (human {hv:.3f}) | " + " | ".join(f"{emd(r,'look','flick_ge15deg')['bot']:.3f}" for r in order) + " |")
    L.append("")

    L.append("## Distribution distance to human — EMD (seconds, lower=better) / MMD p (higher=better)\n")
    L.append(hdr); L.append(sep)
    rows = [("move.fb dwell", ('move','fb','dwell')), ("move.lr dwell", ('move','lr','dwell')),
            ("move.ud dwell", ('move','ud','dwell')), ("weapon dwell", ('weapon','dwell')),
            ("attack fire-run", ('attack','fire_run')), ("attack fire-onset iei", ('attack','fire_onset_interval')),
            ("look turn deg/sec", ('look','turn_deg_per_sec'))]
    for name, path in rows:
        cells = []
        for r in order:
            d = emd(r, *path)
            mp = d.get('mmd_p')
            cells.append(f"{d['emd_sec']:.3f} / {mp if mp is not None else '—'}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Median dwell / hold — human vs bot (seconds)\n")
    L.append(hdr); L.append(sep)
    for name, path in [("move.fb dwell med", ('move','fb','dwell')), ("move.lr dwell med", ('move','lr','dwell')),
                       ("move.ud dwell med", ('move','ud','dwell')), ("weapon dwell med", ('weapon','dwell')),
                       ("fire-run med", ('attack','fire_run'))]:
        hmed = emd(order[0], *path)['human']['median']
        cells = [f"{emd(r,*path)['bot']['median']}" for r in order]
        L.append(f"| {name} (human {hmed}) | " + " | ".join(cells) + " |")
    L.append("")
    L.append("> turn deg/sec mean (human {:.1f}): ".format(emd(order[0],'look','turn_deg_per_sec')['human']['mean'])
             + ", ".join(f"{r}={emd(r,'look','turn_deg_per_sec')['bot']['mean']:.1f}" for r in order))

    (args.streams_dir / "REPORT_humanlikeness.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {args.streams_dir/'REPORT_humanlikeness.md'} + humanlikeness.json")


if __name__ == "__main__":
    main()
