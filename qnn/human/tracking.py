"""TRACKING distribution (window-sampled alignment around discharges).

The trigger-free half of the intercept split (design: decode-fit-v2 addendum
2026-07-18 + discharge-quality-gate). ``qnn.human.intercept`` measures
alignment AT the fired tick — a crest-conditioned statistic for humans (they
crest-fire) but a random-phase sample for an alignment-blind model, so fitting
model at-discharge hbw to human at-discharge anchors silently demands
inhuman time-average tracking (the TOT trap). This baseline measures the
SAME ruler (``aim_kernel._lead_aim_angle_deg_live``, same lead law, same
op/LOS conditioning) sampled UNIFORMLY over engagement windows — every tick
within ±k of a discharge, each tick assigned to its NEAREST discharge (whose
weapon labels it; overlapping burst windows never double-count a tick).

Emits, per weapon (norm = hitbox-half-widths, the construct-valid form):

  * ``tracking_dist``      — pooled window-tick hbw distribution + percentiles
                             per k in ``K_TICKS`` (window-size sensitivity is
                             the first sanity check).
  * ``placement_anchors``  — the SAME frozen §16.3 selection procedure as the
                             intercept baseline, run on per-demo WINDOW-median
                             populations at ``K_ANCHOR``.
  * ``crest_capture``      — per-demo median(at-discharge) / median(window):
                             humans are crest-firers, so this should sit
                             BELOW 1; an alignment-blind sampler reads ≈ 1.
                             The second sanity check.
  * ``crest_offset``       — histogram of (best-tick − fired-tick) within ±k:
                             where the alignment crest sits relative to the
                             trigger. Humans should concentrate at 0. The
                             directionality answer (recoverable-by-hold vs
                             fired-past-optimal) for the model comes from the
                             same histogram on eval windows.

HUMAN qwd only, offline, no model. Same explicit lead-cap contract as the
intercept baseline (caps OR --linear, never a silent default).

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.tracking \\
      --collect-dir artifacts/collect/qwd_v3 \\
      --lead-hold-cap-frames 4.0 --lead-hold-cap-radial-frames 5.0 \\
      [--split both] [--workers N] [--min-ticks 200] \\
      [--out <collect>/human_baseline/_aim_tracking_window.json]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from qnn.eval import aim_kernel as A
from qnn.human import intercept as I

DEG = 180.0 / math.pi

# Window half-widths (20Hz ticks) reported for sensitivity; anchors ride
# K_ANCHOR. ±4 ticks = ±200 ms ≈ one tracking-oscillation period (~5 Hz).
K_TICKS = (2, 4, 8)
K_ANCHOR = 4
K_MAX = max(K_TICKS)

NN = len(I.NORM_EDGES) - 1
N_OFF = 2 * K_MAX + 1                      # crest-offset bins: −K_MAX..+K_MAX


def _episode(cnt, rel, vel, typ, rec, wid, hx, af):
    """Per (weapon, half) window/discharge hbw histograms + crest-offset
    counts for one episode. Returns
    ``{"win": {k: {w: (2, NN)}}, "dis": {w: (2, NN)}, "off": {w: (N_OFF,)}}``
    (w keys: 0 = all weapons + I.ALL_IMP)."""
    T = len(cnt)
    ar, av, lo, halfw = A.densify_entities(cnt, rel, vel, typ, rec, aux=[hx])
    dist_u = np.linalg.norm(ar, axis=2)
    ar *= (1.0 / A._DIST_SCALE)
    av *= (1.0 / A._VEL_SCALE)

    af = np.asarray(af, np.float32)
    if af.ndim == 2:
        af = af[:, 0]
    discharge = np.zeros(T, bool)
    if T > 1:
        discharge[1:] = af[1:] > (af[:-1] + 1e-4)
    has = lo.any(1)
    di = np.where(discharge & has)[0]

    win = {k: {w: np.zeros((2, NN), np.float64) for w in [0] + I.ALL_IMP}
           for k in K_TICKS}
    dis = {w: np.zeros((2, NN), np.float64) for w in [0] + I.ALL_IMP}
    off = {w: np.zeros(N_OFF, np.float64) for w in [0] + I.ALL_IMP}
    if not len(di):
        return {"win": win, "dis": dis, "off": off}

    imp_d = I._WI[wid[di]]                       # weapon impulse per discharge
    # per-weapon discharge ordinal parity (EVEN/ODD split-half, as intercept)
    half_d = np.zeros(len(di), np.int64)
    for w in I.ALL_IMP:
        m = np.where(imp_d == w)[0]
        half_d[m] = np.arange(len(m)) % 2

    # nearest-discharge assignment for every tick within ±K_MAX of any
    # discharge (ties → earlier discharge; bursts never double-count a tick)
    ins = np.searchsorted(di, np.arange(T))
    prev_i = np.clip(ins - 1, 0, len(di) - 1)
    next_i = np.clip(ins, 0, len(di) - 1)
    d_prev = np.abs(np.arange(T) - di[prev_i])
    d_next = np.abs(di[next_i] - np.arange(T))
    nearest = np.where(d_prev <= d_next, prev_i, next_i)
    gap = np.minimum(d_prev, d_next)
    tick_mask = (gap <= K_MAX) & has
    ti = np.where(tick_mask)[0]
    if not len(ti):
        return {"win": win, "dis": dis, "off": off}
    anchor = nearest[ti]                          # discharge index per tick
    offset = ti - di[anchor]                      # signed ticks from ITS anchor

    # one vectorized ruler pass over all window ticks; each tick scored under
    # its ANCHOR discharge's weapon (burst-stable; ticks near a switch follow
    # the shot they belong to)
    imp_t = imp_d[anchor]
    ang = A._lead_aim_angle_deg_live(ar[ti], av[ti], imp_t, I._WN,
                                     I._LEAD_CAP, I._LEAD_CAP_RAD)
    ang = np.where(lo[ti], ang, np.inf)
    slot = np.argmin(ang, axis=1)
    rows = np.arange(len(ti))
    best = ang[rows, slot]
    dsel = dist_u[ti][rows, slot]
    hsel = halfw[ti][rows, slot]
    with np.errstate(divide="ignore", invalid="ignore"):
        ang_radius = np.arctan(hsel / np.maximum(dsel, 1e-3)) * DEG
        norm = best / np.maximum(ang_radius, 1e-6)
    fin = np.isfinite(norm)
    bins = np.clip(np.searchsorted(I.NORM_EDGES, norm, side="right") - 1,
                   0, NN - 1)

    is_fired = offset == 0
    for w in [0] + I.ALL_IMP:
        wm = fin if w == 0 else (fin & (imp_t == w))
        if not wm.any():
            continue
        h_t = half_d[anchor]
        for k in K_TICKS:
            m = wm & (np.abs(offset) <= k)
            np.add.at(win[k][w], (h_t[m], bins[m]), 1.0)
        m = wm & is_fired
        np.add.at(dis[w], (h_t[m], bins[m]), 1.0)
        # crest offset per discharge: argmin of norm over ITS finite window
        # ticks (fired tick must be finite — it is, by the `has` gate on di)
        dm = np.where(wm)[0]
        if len(dm):
            order = np.lexsort((norm[dm], anchor[dm]))
            sa = anchor[dm][order]
            first = np.ones(len(sa), bool)
            first[1:] = sa[1:] != sa[:-1]
            best_rows = dm[order[first]]
            np.add.at(off[w], offset[best_rows] + K_MAX, 1.0)
    return {"win": win, "dis": dis, "off": off}


def _worker(args):
    sh, dd = args
    res: dict[int, dict] = {}
    for _ei, dmi, fsl, esl, arr in A.iter_shard_episodes(
            sh, dd,
            obs=("entity_rel", "entity_vel", "entity_types", "entity_recency",
                 "self_weapon_id", "entity_half_extents", "attack_finished")):
        hx = np.asarray(arr["entity_half_extents"][esl])
        hx_h = hx[:, 0] if hx.ndim == 2 else hx
        out = _episode(np.asarray(arr["entity_count"][fsl], np.int64),
                       np.asarray(arr["entity_rel"][esl]),
                       np.asarray(arr["entity_vel"][esl]),
                       np.asarray(arr["entity_types"][esl]),
                       np.asarray(arr["entity_recency"][esl]),
                       np.asarray(arr["self_weapon_id"][fsl]),
                       np.asarray(hx_h, np.float32),
                       np.asarray(arr["attack_finished"][fsl]))
        b = res.setdefault(int(dmi), {"win": {k: {} for k in K_TICKS},
                                      "dis": {}, "off": {}})
        for k in K_TICKS:
            for w, v in out["win"][k].items():
                b["win"][k][w] = (b["win"][k][w] + v) if w in b["win"][k] \
                    else v.copy()
        for key in ("dis", "off"):
            for w, v in out[key].items():
                b[key][w] = (b[key][w] + v) if w in b[key] else v.copy()
    return res


def _pool_hist(per: dict, demos: list, w, *, k=None) -> np.ndarray:
    tot = np.zeros(NN, np.float64)
    for d in demos:
        v = per[d]["win"][k].get(w) if k is not None else per[d]["dis"].get(w)
        if v is not None:
            tot += v.sum(0)
    return tot


def run(collect_dir: Path, splits, out_path: Path, n_workers: int,
        min_ticks: int, cap_frames=None, cap_rad_frames=None) -> None:
    I._LEAD_CAP = (float(cap_frames) * A.TICK_DT_MODULE) if cap_frames else None
    I._LEAD_CAP_RAD = (float(cap_rad_frames) * A.TICK_DT_MODULE) \
        if cap_rad_frames else None
    hazard_aware = I._LEAD_CAP is not None or I._LEAD_CAP_RAD is not None

    per: dict[int, dict] = {}
    for split in splits:
        dd = collect_dir / f"precomputed_{split}"
        man = dd / "manifest.json"
        if not man.exists():
            continue
        shards = json.loads(man.read_text())["shards"]
        tasks = [(sh, str(dd)) for sh in shards]
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
                for dmi, b in r.items():
                    t = per.setdefault(dmi, {"win": {k: {} for k in K_TICKS},
                                             "dis": {}, "off": {}})
                    for k in K_TICKS:
                        for w, v in b["win"][k].items():
                            t["win"][k][w] = (t["win"][k][w] + v) \
                                if w in t["win"][k] else v.copy()
                    for key in ("dis", "off"):
                        for w, v in b[key].items():
                            t[key][w] = (t[key][w] + v) if w in t[key] \
                                else v.copy()
                print(f"  [{split}] {i+1}/{len(tasks)} shards", flush=True)

    # family aggregates (same convention as intercept)
    for d in per:
        for fam, members in I.FAMILIES.items():
            for k in K_TICKS:
                acc = None
                for w in members:
                    v = per[d]["win"][k].get(w)
                    if v is not None:
                        acc = v.copy() if acc is None else acc + v
                if acc is not None:
                    per[d]["win"][k][fam] = acc
            for key in ("dis", "off"):
                acc = None
                for w in members:
                    v = per[d][key].get(w)
                    if v is not None:
                        acc = v.copy() if acc is None else acc + v
                if acc is not None:
                    per[d][key][fam] = acc

    demos = sorted(per)
    keys = ([("all", 0)] + [(I.IMPULSE_NAME[w], w) for w in I.ALL_IMP]
            + [(fam, fam) for fam in I.FAMILIES])

    # per-demo medians at the anchor window; the anchor machinery reads
    # per[d]["norm"] — present it that view
    per_anchor = {d: {"norm": dict(per[d]["win"][K_ANCHOR])} for d in demos}
    dup_dropped, dup_examples = I._duplicate_census(per_anchor, demos)
    census, pops = I._anchor_populations(per_anchor, demos, keys, min_ticks,
                                         dup_dropped)
    census["duplicates_dropped"] = len(dup_dropped)
    census["duplicate_examples"] = dup_examples[:8]

    reliability = {}
    spread = {}
    spread_compat = {}       # "norm:<name>" keys — human_refs.reachable_band
    half_min = max(1, min_ticks // 2)
    for name, w in keys:
        h0, h1, meds = [], [], []
        for d in demos:
            v = per[d]["win"][K_ANCHOR].get(w)
            if v is None or v.sum() < min_ticks:
                continue
            m = I._median_from_hist(v.sum(0), I.NORM_EDGES)
            if np.isfinite(m):
                meds.append(m)
            if v[0].sum() >= half_min and v[1].sum() >= half_min:
                m0 = I._median_from_hist(v[0], I.NORM_EDGES)
                m1 = I._median_from_hist(v[1], I.NORM_EDGES)
                if np.isfinite(m0) and np.isfinite(m1):
                    h0.append(m0)
                    h1.append(m1)
        r, n = I._spearman(h0, h1)
        sb = (2 * r / (1 + r)) if (np.isfinite(r) and r > -1) else None
        reliability[name] = {
            "split_half_spearman_of_median": (round(r, 4) if np.isfinite(r)
                                              else None),
            "spearman_brown": (round(sb, 4) if sb is not None else None),
            "n_players": n}
        if len(meds) >= 5:
            p10, p50, p90 = np.percentile(meds, [10, 50, 90])
            spread[name] = {"n_players": len(meds),
                            "median_p10": round(float(p10), 3),
                            "median_p50": round(float(p50), 3),
                            "median_p90": round(float(p90), 3)}
            spread_compat[f"norm:{name}"] = dict(spread[name])

    rel_sb = {name: reliability[name]["spearman_brown"] for name, _w in keys}
    anchor_weapons = I._select_placement_anchors(pops, rel_sb)
    placement_anchors = {
        "anchors_version": I.ANCHORS_VERSION,
        "statistic": (f"per-demo median hbw over ±{K_ANCHOR}-tick windows "
                      "around discharges (window-sampled tracking, NOT "
                      "at-discharge)"),
        "procedure": I.ANCHOR_PROCEDURE,
        "config": {"k_ticks": K_ANCHOR, "min_ticks": min_ticks,
                   "boot_resamples": I.ANCHOR_BOOT, "seed": I.ANCHOR_SEED},
        "weapons": anchor_weapons,
    }

    # pooled distributions per k + at-discharge, with medians for sensitivity
    PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def _pcts(counts):
        tot = counts.sum()
        if tot <= 0:
            return None
        cum = np.cumsum(counts)
        out = {}
        for p in PCTS:
            tgt = p / 100.0 * tot
            j = min(int(np.searchsorted(cum, tgt)), NN - 1)
            lo, hi = I.NORM_EDGES[j], I.NORM_EDGES[j + 1]
            prev = cum[j - 1] if j > 0 else 0.0
            out[f"p{p}"] = round(float(lo if (not np.isfinite(hi)
                                              or counts[j] <= 0)
                                       else lo + (tgt - prev) / counts[j]
                                       * (hi - lo)), 3)
        return out

    tracking_dist = {}
    for name, w in keys:
        entry = {}
        for k in K_TICKS:
            tot = _pool_hist(per, demos, w, k=k)
            entry[f"k{k}"] = {"n_ticks": int(tot.sum()),
                              "percentiles": _pcts(tot)}
        tot = _pool_hist(per, demos, w)
        entry["at_discharge"] = {"n_attacks": int(tot.sum()),
                                 "percentiles": _pcts(tot)}
        tracking_dist[name] = entry

    # crest capture: per-demo median(at-discharge)/median(window @ K_ANCHOR)
    crest_capture = {}
    for name, w in keys:
        ratios = []
        for d in demos:
            vw = per[d]["win"][K_ANCHOR].get(w)
            vd = per[d]["dis"].get(w)
            if vw is None or vd is None or vw.sum() < min_ticks \
                    or vd.sum() < max(1, min_ticks // (2 * K_ANCHOR + 1)):
                continue
            mw = I._median_from_hist(vw.sum(0), I.NORM_EDGES)
            md = I._median_from_hist(vd.sum(0), I.NORM_EDGES)
            if np.isfinite(mw) and np.isfinite(md) and mw > 0:
                ratios.append(md / mw)
        if len(ratios) >= 5:
            p10, p50, p90 = np.percentile(ratios, [10, 50, 90])
            crest_capture[name] = {
                "n_demos": len(ratios),
                "ratio_p10": round(float(p10), 3),
                "ratio_p50": round(float(p50), 3),
                "ratio_p90": round(float(p90), 3),
                "note": "median(at-discharge hbw)/median(window hbw); < 1 = "
                        "crest-firing (fires at better-than-typical "
                        "alignment); ≈ 1 = alignment-blind trigger"}

    crest_offset = {}
    for name, w in keys:
        tot = np.zeros(N_OFF, np.float64)
        for d in demos:
            v = per[d]["off"].get(w)
            if v is not None:
                tot += v
        s = tot.sum()
        if s > 0:
            crest_offset[name] = {
                "offsets_ticks": list(range(-K_MAX, K_MAX + 1)),
                "frac": [round(float(x / s), 4) for x in tot],
                "n_discharges": int(s),
                "frac_best_at_fire": round(float(tot[K_MAX] / s), 4),
                "frac_best_before": round(float(tot[:K_MAX].sum() / s), 4),
                "frac_best_after": round(float(tot[K_MAX + 1:].sum() / s), 4)}

    out = {
        "title": "TRACKING distribution (window-sampled alignment around "
                 "discharges) — the trigger-free aim statistic",
        "defs": {
            "window": f"ticks within ±k of a discharge, each tick assigned "
                      f"to its NEAREST discharge (weapon + split-half parity "
                      f"follow the anchor; k ∈ {list(K_TICKS)}, anchors at "
                      f"k={K_ANCHOR})",
            "metric": "same ruler as _aim_intercept_skill (lead-corrected "
                      "angle to most-aligned visible enemy, hbw-normalized), "
                      "sampled uniformly over the window instead of at the "
                      "fired tick",
        },
        "config": {"splits": splits, "n_players": len(demos),
                   "min_ticks": min_ticks,
                   "lead": ("hazard_aware" if hazard_aware else "linear"),
                   "lead_hold_cap_frames": cap_frames,
                   "lead_hold_cap_radial_frames": cap_rad_frames},
        "reliability_of_window_median": reliability,
        "spread_of_window_median": spread,
        # human_refs.reachable_band compat (same key shape as intercept)
        "spread_of_median": spread_compat,
        "census": census,
        "placement_anchors": placement_anchors,
        "tracking_dist": tracking_dist,
        "crest_capture": crest_capture,
        "crest_offset": crest_offset,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    # ── sanity console report ────────────────────────────────────────────
    print(f"\n{len(demos)} players")
    print("\n==== window-median spread + reliability (hbw, k="
          f"{K_ANCHOR}) vs k-sensitivity ====")
    print(f"{'wpn':>7} {'SB':>7} {'n':>5} | {'p10':>7} {'p50':>7} {'p90':>7} "
          f"| " + "  ".join(f"{'k'+str(k)+' p50':>8}" for k in K_TICKS)
          + f" {'@fire p50':>10}")
    for name, w in keys:
        rel = reliability[name]
        sp = spread.get(name)
        td = tracking_dist[name]
        row_k = "  ".join(
            f"{(td[f'k{k}']['percentiles'] or {}).get('p50', ''):>8}"
            for k in K_TICKS)
        fire = (td["at_discharge"]["percentiles"] or {}).get("p50", "")
        sps = (f"{sp['median_p10']:>7} {sp['median_p50']:>7} "
               f"{sp['median_p90']:>7}" if sp else f"{'':>7} {'':>7} {'':>7}")
        print(f"{name:>7} {str(rel['spearman_brown']):>7} "
              f"{rel['n_players']:>5} | {sps} | {row_k} {fire:>10}")
    print("\n==== crest capture (at-discharge / window median; <1 = "
          "crest-firing) ====")
    for name, _w in keys:
        cc = crest_capture.get(name)
        if cc:
            print(f"  {name:>7}: p50 {cc['ratio_p50']}  "
                  f"[p10 {cc['ratio_p10']} .. p90 {cc['ratio_p90']}]  "
                  f"(n={cc['n_demos']})")
    print("\n==== crest offset (best tick − fired tick; humans should "
          "concentrate at 0) ====")
    for name, _w in keys:
        co = crest_offset.get(name)
        if co:
            print(f"  {name:>7}: at-fire {co['frac_best_at_fire']}  "
                  f"before {co['frac_best_before']}  "
                  f"after {co['frac_best_after']}  "
                  f"(n={co['n_discharges']})")
    print("\n==== placement anchors (window statistic, k="
          f"{K_ANCHOR}) ====")
    for name, _w in keys:
        a = anchor_weapons.get(name)
        if a is None:
            continue
        flags = ",".join(f for f in ("unvalidated", "family_borrowed",
                                     "shrunk") if a[f]) or "-"
        print(f"  {name:>7} elite {a['elite_hbw']:>8} ({a['elite_depth']})  "
              f"floor {a['floor_hbw']:>8} ({a['floor_depth']})  "
              f"SB {str(a['reliability_sb']):>7}  n {a['n_demos']:>4}  {flags}")
    print(f"\nWritten -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-dir", type=Path,
                    default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="both",
                    choices=["val", "train", "both"])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--min-ticks", type=int, default=200,
                    help="min window ticks per demo per weapon (≈ the "
                         "intercept baseline's 40 discharges × a ±4 window's "
                         "yield after burst overlap)")
    ap.add_argument("--lead-hold-cap-frames", type=float, default=None)
    ap.add_argument("--lead-hold-cap-radial-frames", type=float, default=None)
    ap.add_argument("--linear", action="store_true",
                    help="EXPLICITLY build the un-capped (linear-lead) band "
                         "(same no-silent-default contract as intercept)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_aim_tracking_window.json"
    if args.lead_hold_cap_frames is None \
            and args.lead_hold_cap_radial_frames is None and not args.linear:
        ap.error("must pass the deployed hazard lead caps "
                 "(--lead-hold-cap-frames / --lead-hold-cap-radial-frames; "
                 "a25 = 4.0 / 5.0) OR --linear")
    splits = ["train", "val"] if args.split == "both" else [args.split]
    print(f"Collect: {args.collect_dir}  splits={splits}  "
          f"workers={args.workers}  k={list(K_TICKS)} (anchor k={K_ANCHOR})")
    run(args.collect_dir, splits, out, args.workers, args.min_ticks,
        args.lead_hold_cap_frames, args.lead_hold_cap_radial_frames)


if __name__ == "__main__":
    main()
