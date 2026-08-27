"""INTERCEPTION distribution (discharge-anchored alignment-at-attack).

One of the two live aim axes (acquisition ⊥ interception; §9). INTERCEPTION = how
aligned the crosshair is to the target AT THE MOMENT A SHOT IS FIRED.

  Discharge = attack_finished rising edge (a shot left the muzzle this tick;
  verified vs the raw attack button, per-weapon shares match within ~2 pts).
  At each discharge with an enemy in LOS: lead-corrected angle to the most-aligned
  visible enemy (aim_skill._lead_aim_angle_deg_live — deployed lead geometry).

The DECODE-FIT reference is `interception_dist`: the pooled distribution of the
metric (percentiles + EXACT per-weapon min/max), the export point rides it. The
trustworthy anchor is [min, max] — the genuine best/worst human action, robust to the
corpus NOT being a representative population and to demo != player (the extremes don't
move when one person records many demos). The interior percentiles are the observed
metric distribution (duplication-biased, named honestly — not a population/skill
ranking). Per-demo aggregates quantify RANGES only (within_player_consistency: a rough
sense of a single recording's own spread), NEVER skill. Measured two ways:
  deg   — raw lead-corrected angle (deg).
  norm  — distance-normalized to HITBOX-HALF-WIDTHS (angle / atan(halfw/dist)),
          range-invariant: <1 means the crosshair is on the enemy's body at any
          range. The construct-valid form.

Per-weapon (no g-factor, §9) + overall; split-half reliability (EVEN/ODD discharge
parity) of the per-demo median.

PLACEMENT ANCHORS (skill-curves §16.3): the decode-fit skill coordinate no longer
rides fixed p10/p90 depths — each collect runs the FROZEN per-corpus selection
procedure (hygiene census → candidate depths → bootstrap-CI + out-of-sample-holdout +
breakdown criteria → reliability shrinkage → family fallback) and emits a validated
``placement_anchors`` node consumed by ``qnn.decode_fit.human_refs``. The existing
``spread_of_median`` node is unchanged (other consumers read it).

HUMAN qwd only, offline, no model, deployed decode READ-only.

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.intercept \\
      --collect-dir artifacts/collect/qwd [--split both] [--workers N] \\
      [--min-shots 40] [--out <collect>/human_baseline/_aim_intercept_skill.json]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from qnn.eval import aim_kernel as A

IMPULSE_NAME = {2: "SG", 3: "SSG", 4: "NG", 5: "SNG", 6: "GL", 7: "RL", 8: "LG"}
DIRECT_FIRE_IMP = [2, 3, 4, 5, 8]
ALL_IMP = DIRECT_FIRE_IMP + [7, 6]           # + RL, GL
DEG = 180.0 / math.pi

# Same-physics families: identical lead geometry within each, so their alignment
# distributions must match — a sanity check (run separately AND pooled).
FAMILIES = {"SG+SSG": [2, 3], "NG+SNG": [4, 5]}

# Hazard-discounted lead caps in MODULE units (frames × A.TICK_DT_MODULE); set from
# --lead-hold-cap-frames before the Pool fork so workers inherit. None = linear lead
# (bit-identical to the un-capped band). Caps bite only projectile weapons (NG/SNG/
# GL/RL); hitscan (SG/SSG/LG) is boosted ×100 → lead term ~0 → caps inert.
_LEAD_CAP: "float | None" = None
_LEAD_CAP_RAD: "float | None" = None

# Fine-near-zero bin edges for the alignment-error histogram (median + buckets).
DEG_EDGES = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15,
                      20, 25, 30, 40, 60, 90, np.inf])
# Tail extended well past 10 hbw so the loose end is RESOLVED, not censored at the
# last finite edge. The interior percentiles read off this histogram; the true
# min/max bounds are tracked EXACTLY (running per-weapon, not histogram-pinned).
NORM_EDGES = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 10, 15, 20,
                       30, 45, 65, 90, 130, np.inf])
ND, NN = len(DEG_EDGES) - 1, len(NORM_EDGES) - 1


def _median_from_hist(counts, edges):
    tot = counts.sum()
    if tot <= 0:
        return float("nan")
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, 0.5 * tot))
    k = min(k, len(counts) - 1)
    lo, hi = edges[k], edges[k + 1]
    if not np.isfinite(hi) or counts[k] <= 0:
        return float(lo)
    prev = cum[k - 1] if k > 0 else 0.0
    return float(lo + (0.5 * tot - prev) / counts[k] * (hi - lo))


def _episode(cnt, rel, vel, typ, rec, weapon, hx, af):
    """Per (weapon, half) discharge-alignment histograms (deg + norm).
    half = EVEN/ODD parity of this weapon's discharge index. key 0 = ALL weapons.
    Returns {'deg': {w:(2,ND)}, 'norm': {w:(2,NN)}}."""
    T = len(cnt)
    # Shared densify (raw padded arrays; halfw via aux) — see qnn.eval.aim_kernel.
    ar, av, lo, halfw = A.densify_entities(cnt, rel, vel, typ, rec, aux=[hx])
    dist_u = np.linalg.norm(ar, axis=2)               # raw units
    ar *= (1.0 / A._DIST_SCALE); av *= (1.0 / A._VEL_SCALE)

    # Discharge = the attack-with INTENT (the act_attack label), NOT the
    # attack_finished cooldown edge. In the a27 attack-with encoding act_attack is
    # nonzero (1..8 = fired impulse) ONLY on the discharge frame, so it is at once
    # the discharge signal, the fired weapon, and the frame the human COMMITTED the
    # aim. The legacy attack_finished-rising-edge discharge read the weapon at the
    # cooldown-reset frame — a DIFFERENT tick at 10 Hz — where act_attack is already
    # 0, so imp=0 for EVERY shot: no per-weapon attribution AND the pooled 'all'
    # ruler ran the lead-aim ballistic with imp=0. Intent-keying fixes both and
    # matches the model-side operative-fire discharge (eval/run.py) +
    # aim_kernel.action_attack_context. (af is retained in the signature for the
    # caller contract but is no longer the discharge source.)
    wv = np.asarray(weapon, dtype=np.int64).reshape(-1)
    discharge = (wv >= 1) & (wv <= 8)
    has = lo.any(1)
    di = np.where(discharge & has)[0]
    deg = {w: np.zeros((2, ND), np.float64) for w in [0] + ALL_IMP}
    nrm = {w: np.zeros((2, NN), np.float64) for w in [0] + ALL_IMP}
    if not len(di):
        return {"deg": deg, "norm": nrm, "bounds": {}}
    rel_d, vel_d, los_d = ar[di], av[di], lo[di]
    imp = np.asarray(weapon, dtype=np.int64)[di].clip(0, 8)
    ang = A._lead_aim_angle_deg_live(rel_d, vel_d, imp, _WN,
                                     _LEAD_CAP, _LEAD_CAP_RAD)     # (M,N) deg
    ang = np.where(los_d, ang, np.inf)
    slot = np.argmin(ang, axis=1)
    mrow = np.arange(len(di))
    best = ang[mrow, slot]                                        # (M,) deg to most-aligned
    # normalized to hitbox-half-widths at that slot
    dsel = dist_u[di][mrow, slot]; hsel = halfw[di][mrow, slot]
    with np.errstate(divide="ignore", invalid="ignore"):
        ang_radius = np.arctan(hsel / np.maximum(dsel, 1e-3)) * DEG
        best_norm = best / np.maximum(ang_radius, 1e-6)
    fin = np.isfinite(best)
    di_deg = np.clip(np.searchsorted(DEG_EDGES, best, side="right") - 1, 0, ND - 1)
    di_nrm = np.clip(np.searchsorted(NORM_EDGES, best_norm, side="right") - 1, 0, NN - 1)

    def fill(w, mask):
        idx = np.where(mask)[0]
        for pos, k in enumerate(idx):
            h = pos % 2
            deg[w][h, di_deg[k]] += 1.0
            nrm[w][h, di_nrm[k]] += 1.0
    # EXACT per-weapon norm (hbw) bounds — the genuine best/worst human action, not
    # the last histogram edge. {w: (min, max)} over finite discharges (w=0 = all).
    bounds: dict[int, tuple[float, float]] = {}

    def bd(w, mask):
        v = best_norm[mask]
        v = v[np.isfinite(v)]
        if len(v):
            bounds[w] = (float(v.min()), float(v.max()))
    fill(0, fin); bd(0, fin)
    for w in ALL_IMP:
        m = fin & (imp == w)
        if m.any():
            fill(w, m); bd(w, m)
    return {"deg": deg, "norm": nrm, "bounds": bounds}


def _worker(args):
    sh, dd = args
    res: dict[int, dict] = {}
    for _ei, dmi, fsl, esl, arr in A.iter_shard_episodes(
            sh, dd,
            obs=("entity_rel", "entity_vel", "entity_types", "entity_recency",
                 "entity_half_extents", "attack_finished"),
            acts=("attack",)):
        hx = np.asarray(arr["entity_half_extents"][esl])
        hx_h = hx[:, 0] if hx.ndim == 2 else hx
        out = _episode(np.asarray(arr["entity_count"][fsl], np.int64),
                       np.asarray(arr["entity_rel"][esl]), np.asarray(arr["entity_vel"][esl]),
                       np.asarray(arr["entity_types"][esl]), np.asarray(arr["entity_recency"][esl]),
                       np.asarray(arr["attack"][fsl]),
                       np.asarray(hx_h, np.float32), np.asarray(arr["attack_finished"][fsl]))
        b = res.setdefault(int(dmi), {"deg": {}, "norm": {}, "bounds": {}})
        for kind in ("deg", "norm"):
            for w, v in out[kind].items():
                b[kind][w] = (b[kind][w] + v) if w in b[kind] else v.copy()
        for w, (blo, bhi) in out["bounds"].items():
            if w in b["bounds"]:
                o0, o1 = b["bounds"][w]
                b["bounds"][w] = (min(o0, blo), max(o1, bhi))
            else:
                b["bounds"][w] = (blo, bhi)
    return res


_WN, _ZD = A._build_physics_tables()


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), int(len(x))
    def rk(a):
        o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); s = a[o]; i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and s[j + 1] == s[i]:
                j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
        return r
    rx, ry = rk(x), rk(y); rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return (float((rx * ry).sum() / d) if d > 0 else float("nan")), int(len(x))


# ── PLACEMENT ANCHORS: frozen per-corpus selection procedure (§16.3) ─────────────
# The decode-fit band coordinate anchors (p0 = floor, p100 = elite) are OUTPUTS of
# this procedure, run per collect on the hygiened per-demo-median distribution. The
# anchor-depth analysis (runs/head_probe/_anchor_depth_analysis.json) proved the safe
# depth varies by weapon and corpus, so the depth is SELECTED, never assumed. The
# criteria below are FROZEN — anti-cycling: they change only on a failed holdout
# validation, never to move an anchor somewhere convenient.
ANCHORS_VERSION = 1
ANCHOR_ELITE_DEPTHS = (1.0, 2.0, 5.0, 10.0)   # candidate elite depths, deepest first
ANCHOR_FLOOR_DEPTHS = (95.0, 90.0)            # candidate floor depths, deepest first
ANCHOR_DEFAULT = {"elite": 10.0, "floor": 90.0}   # analysis-blessed resting depths
ANCHOR_BOOT = 2000                            # demo-bootstrap resamples (criterion a)
ANCHOR_CI = (2.5, 97.5)                       # 95% percentile CI over the bootstrap
ANCHOR_CI_FRAC_MAX = 0.10                     # (a) CI width ≤ 10% of the p10→p90 band
ANCHOR_MIN_SEL = 5                            # (b) top-k holdout needs ≥ 5 demos
ANCHOR_CONTAM_LO_HBW = 1.0                    # (c) elite-side contaminant: median < 1 hbw
ANCHOR_CONTAM_HI_HBW = 20.0                   # (c) floor-side contaminant: median > 20 hbw
ANCHOR_RELIABILITY_MIN = 0.7                  # SB below → shrink toward the log-mean
ANCHOR_DUP_MED_RTOL = 1e-3                    # duplicate demos: medians within 0.1%
ANCHOR_SEED = 20260716                        # frozen; per-key streams via crc32(name)

# Same-physics family a failed weapon borrows its anchors from (builder aggregates).
ANCHOR_FAMILY_OF = {"SG": "SG+SSG", "SSG": "SG+SSG", "NG": "NG+SNG", "SNG": "NG+SNG"}

ANCHOR_PROCEDURE = (
    "Frozen per-corpus placement-anchor selection (skill-curves §16.3). Hygiene "
    "(censused, never silent): duplicate demos (identical per-weapon shot-count "
    "vectors + per-weapon medians within 0.1%) dropped keeping the first; demos whose "
    "median pins the top histogram edge excluded from anchor populations (they stay "
    "in the pooled distribution/census). Candidates: elite {p1,p2,p5,p10}, floor "
    "{p95,p90} over the hygiened per-demo-median distribution; p10/p90 are the "
    "analysis-blessed DEFAULT depths, deeper placement must be EARNED. A depth is "
    "accepted iff (a) demo-bootstrap (2000 resamples) 95% CI width <= 10% of the "
    "p10->p90 band width; (b) out-of-sample RTM-consistency: the top-2k% demos "
    "selected on EVEN-half medians (>= min_shots per half; 2k so the selection's "
    "median RANKS at the candidate depth) have an ODD-half aggregate median within "
    "the criterion-(a) CI width of its regression-to-the-mean PREDICTION (the "
    "log-space odd~even regression line evaluated at the selection median) — this "
    "catches tails that shrink MORE than measured reliability explains "
    "(contamination/noise-domination), while honest low reliability is handled by "
    "shrinkage, never depth rejection (the analysis's RL conclusion); (c) quantile "
    "breakdown point >= measured contamination fraction (< 1 hbw elite side, > 20 "
    "hbw floor side), with >= 5 selected demos to run (b) at all. Deepest passing "
    "depth wins per side; when no candidate passes but the DEFAULT depth is at "
    "least testable (n_sel >= 5), the anchor rests at the default (validated=False, "
    "not a failure). Spearman-Brown reliability < 0.7 shrinks own-population "
    "anchors toward the log-median mean by the reliability coefficient. A side too "
    "THIN to test even the default borrows the same-physics family aggregate "
    "(SG+SSG / NG+SNG); if the family is also untestable, p10/p90 unvalidated "
    "(loud flag). Anti-cycling: criteria change only on a failed holdout "
    "validation.")


def _duplicate_census(per: dict, demos: list) -> tuple[set, list]:
    """Duplicate-demo detection: identical per-weapon shot-count vectors AND
    identical (or near-identical, ≤ ``ANCHOR_DUP_MED_RTOL``) per-weapon medians.
    The analysis found exact pairs (e.g. 374/4637). Keeps the lowest demo id,
    censuses the rest. Returns ``(dropped_ids, [[kept, dropped], …])``."""
    groups: dict[tuple, list] = {}
    for d in demos:
        vec = tuple(int(per[d]["norm"][w].sum()) if w in per[d]["norm"] else 0
                    for w in ALL_IMP)
        if sum(vec) == 0:
            continue
        groups.setdefault(vec, []).append(d)
    dropped: set = set()
    examples: list = []
    for ds in groups.values():
        if len(ds) < 2:
            continue
        ds = sorted(ds)
        kept = ds[0]
        for d in ds[1:]:
            same = True
            for w in ALL_IMP:
                v1 = per[kept]["norm"].get(w)
                if v1 is None or v1.sum() == 0:
                    continue
                m1 = _median_from_hist(v1.sum(0), NORM_EDGES)
                m2 = _median_from_hist(per[d]["norm"][w].sum(0), NORM_EDGES)
                if np.isfinite(m1) != np.isfinite(m2):
                    same = False; break
                if np.isfinite(m1) and abs(m1 - m2) > ANCHOR_DUP_MED_RTOL * max(
                        abs(m1), abs(m2), 1e-9):
                    same = False; break
            if same:
                dropped.add(d); examples.append([int(kept), int(d)])
    return dropped, examples


def _anchor_populations(per: dict, demos: list, keys: list, min_shots: int,
                        dropped: set) -> tuple[dict, dict]:
    """Hygiened anchor populations per bucket key. Returns ``(census, pops)``:
    ``pops[name] = {meds, even, odd, contam_lo, contam_hi}`` where ``meds`` is the
    hygiened per-demo full-median population (duplicates dropped, cap-pinned
    excluded), ``even``/``odd`` the STRICT-half medians (≥ min_shots per half) of
    the same demos, and ``contam_*`` the measured contamination fractions on the
    post-duplicate population (cap-pinned demos count as contaminants)."""
    cap_edge = float(NORM_EDGES[-2])
    pops: dict[str, dict] = {}
    cap_excluded: dict[str, int] = {}
    n_before: dict[str, int] = {}
    n_after: dict[str, int] = {}
    for name, w in keys:
        meds, even, odd = [], [], []
        n_bef = n_cap = n_lo = n_hi = n_postdup = 0
        for d in demos:
            v = per[d]["norm"].get(w)
            if v is None or v.sum() < min_shots:
                continue
            m = _median_from_hist(v.sum(0), NORM_EDGES)
            if not np.isfinite(m):
                continue
            n_bef += 1
            if d in dropped:
                continue
            n_postdup += 1
            if m < ANCHOR_CONTAM_LO_HBW:
                n_lo += 1
            if m > ANCHOR_CONTAM_HI_HBW:
                n_hi += 1
            if m >= cap_edge:               # broken recording — pinned at the cap
                n_cap += 1
                continue
            meds.append(m)
            if v[0].sum() >= min_shots and v[1].sum() >= min_shots:
                m0 = _median_from_hist(v[0], NORM_EDGES)
                m1 = _median_from_hist(v[1], NORM_EDGES)
                if np.isfinite(m0) and np.isfinite(m1):
                    even.append(m0); odd.append(m1)
        pops[name] = {
            "meds": np.asarray(meds, float),
            "even": np.asarray(even, float), "odd": np.asarray(odd, float),
            "contam_lo": (n_lo / n_postdup if n_postdup else 0.0),
            "contam_hi": (n_hi / n_postdup if n_postdup else 0.0)}
        cap_excluded[name] = n_cap; n_before[name] = n_bef; n_after[name] = len(meds)
    census = {
        "cap_pinned_excluded": cap_excluded,
        "n_before": n_before, "n_after": n_after,
        "note": "hygiene for ANCHOR computation only — the pooled distribution / "
                "spread_of_median are untouched. n_before = demos meeting min_shots "
                "(pre-hygiene); n_after = anchor population (duplicates dropped, "
                f"cap-pinned >= {cap_edge:g} hbw excluded)"}
    return census, pops


def _select_placement_anchors(pops: dict[str, dict],
                              reliability_sb: dict[str, Any]) -> dict[str, dict]:
    """The frozen selection procedure (``ANCHOR_PROCEDURE``) over the hygiened
    populations. Returns the per-bucket ``weapons`` node: validated elite/floor
    anchors + depth/CI/holdout diagnostics + shrinkage / family-fallback /
    unvalidated flags. Deterministic (frozen seed, per-key streams)."""
    # pass 1 — evaluate every candidate depth on every bucket's own population
    raw: dict[str, dict] = {}
    for name, pop in pops.items():
        meds = pop["meds"]
        n = len(meds)
        if n == 0:
            continue
        rng = np.random.default_rng([ANCHOR_SEED, zlib.crc32(str(name).encode())])
        p10, p90 = np.percentile(meds, [10, 90])
        band_w = float(p90 - p10)
        depths = list(ANCHOR_ELITE_DEPTHS) + list(ANCHOR_FLOOR_DEPTHS)
        samples = meds[rng.integers(0, n, size=(ANCHOR_BOOT, n))]
        qs = np.percentile(samples, depths, axis=1)              # (n_depths, BOOT)
        ci_w = {d: float(np.percentile(qs[i], ANCHOR_CI[1])
                         - np.percentile(qs[i], ANCHOR_CI[0]))
                for i, d in enumerate(depths)}
        even, odd = pop["even"], pop["odd"]
        order = np.argsort(even, kind="mergesort")               # best (low) first
        # log-space odd~even regression line over the strict-half population —
        # the RTM prediction line criterion (b) checks selections against
        le, lo = np.log(np.maximum(even, 1e-9)), np.log(np.maximum(odd, 1e-9))
        if len(le) >= 3 and float(np.std(le)) > 0 and float(np.std(lo)) > 0:
            r_half = float(np.corrcoef(le, lo)[0, 1])
            rtm_b = r_half * float(np.std(lo)) / float(np.std(le))
            rtm = (float(np.mean(le)), float(np.mean(lo)), rtm_b)
        else:
            r_half, rtm = float("nan"), None
        sides: dict[str, dict] = {}
        for side, cands in (("elite", ANCHOR_ELITE_DEPTHS),
                            ("floor", ANCHOR_FLOOR_DEPTHS)):
            chosen = None; default = None; trail = []
            contam = pop["contam_lo"] if side == "elite" else pop["contam_hi"]
            for dep in cands:
                anchor = float(np.percentile(meds, dep))
                cw = ci_w[dep]
                ci_frac = (cw / band_w) if band_w > 0 else float("inf")
                k_pct = dep if side == "elite" else 100.0 - dep
                # holdout selects a 2k% tail: its median RANKS at the candidate
                # depth (rank-matched, not k%), so the RTM prediction below is
                # directly comparable to the candidate anchor
                nk = int(round(2.0 * k_pct / 100.0 * len(even)))
                gap = None; pred = None; ok_b = False
                testable = nk >= ANCHOR_MIN_SEL and rtm is not None
                if testable:
                    sel = order[:nk] if side == "elite" else order[-nk:]
                    ev_med = float(np.median(even[sel]))
                    mu_e, mu_o, b = rtm
                    pred = float(math.exp(
                        mu_o + b * (math.log(max(ev_med, 1e-9)) - mu_e)))
                    gap = float(np.median(odd[sel]) - pred)
                    ok_b = abs(gap) <= cw
                ok = (ci_frac <= ANCHOR_CI_FRAC_MAX and ok_b
                      and (k_pct / 100.0) >= contam)
                row = {"depth": f"p{dep:g}", "value": round(anchor, 4),
                       "ci_frac": round(ci_frac, 4), "n_sel": nk,
                       "rtm_pred": (round(pred, 4) if pred is not None else None),
                       "holdout_gap": (round(gap, 4) if gap is not None else None),
                       "pass": bool(ok)}
                trail.append(row)
                if ok and chosen is None:                        # deepest passing wins
                    chosen = {"value": anchor, "depth": f"p{dep:g}",
                              "ci_frac": ci_frac, "holdout_gap": gap}
                if dep == ANCHOR_DEFAULT[side]:
                    default = {"value": anchor, "depth": f"p{dep:g}",
                               "ci_frac": ci_frac, "holdout_gap": gap,
                               "testable": bool(testable)}
            sides[side] = {"chosen": chosen, "default": default, "trail": trail}
        raw[name] = {"sides": sides, "p10": float(p10), "p90": float(p90),
                     "half_log_r": (round(r_half, 4) if np.isfinite(r_half) else None),
                     "mean_log": float(np.mean(np.log(np.maximum(meds, 1e-9))))}

    # pass 2 — assemble: default resting depth, family fallback for THIN sides,
    # unvalidated fallback, reliability shrinkage
    def _resolve(rr: dict | None, side: str) -> tuple[dict | None, bool]:
        """A bucket's own resolution for one side: the deepest EARNED candidate
        (validated) → else the testable DEFAULT depth (not validated, not a
        failure) → else None (population too thin to test at all)."""
        if rr is None:
            return None, False
        s = rr["sides"][side]
        if s["chosen"] is not None:
            return s["chosen"], True
        d = s["default"]
        if d is not None and d["testable"]:
            return d, False
        return None, False

    weapons: dict[str, dict] = {}
    for name, pop in pops.items():
        r = raw.get(name)
        if r is None:
            continue
        sb = reliability_sb.get(name)
        sb_f = float(sb) if sb is not None and np.isfinite(float(sb)) else None
        entry: dict[str, Any] = {
            "n_demos": int(len(pop["meds"])), "n_strict_half": int(len(pop["even"])),
            "reliability_sb": (round(sb_f, 4) if sb_f is not None else None),
            "half_log_r": r["half_log_r"],
            "contam_frac_lo": round(pop["contam_lo"], 5),
            "contam_frac_hi": round(pop["contam_hi"], 5),
            "family_borrowed": False, "unvalidated": False, "shrunk": False}
        fam = ANCHOR_FAMILY_OF.get(name)
        vals: dict[str, float] = {}
        for side in ("elite", "floor"):
            ch, validated = _resolve(r, side)
            borrowed = False
            if ch is None and fam is not None:   # own side too thin — borrow family
                ch, validated = _resolve(raw.get(fam), side)
                borrowed = ch is not None
            if ch is None:                   # family untestable too — p10/p90 loud
                dep = ANCHOR_DEFAULT[side]
                ch = {"value": r["p10"] if side == "elite" else r["p90"],
                      "depth": f"p{dep:g}", "ci_frac": None, "holdout_gap": None}
                entry["unvalidated"] = True
            if borrowed:
                entry["family_borrowed"] = True
            entry[f"{side}_validated"] = bool(validated)
            raw_v = float(ch["value"]); v = raw_v
            # reliability shrinkage: own-population anchors only (a borrowed value
            # already carries its family's treatment)
            if not borrowed and sb_f is not None and sb_f < ANCHOR_RELIABILITY_MIN:
                rr = min(max(sb_f, 0.0), 1.0)
                v = float(math.exp(r["mean_log"]
                                   + rr * (math.log(max(raw_v, 1e-9)) - r["mean_log"])))
                entry["shrunk"] = True
            entry[f"{side}_hbw"] = round(v, 4)
            entry[f"{side}_depth"] = ch["depth"]
            entry[f"ci_frac_{side}"] = (round(ch["ci_frac"], 4)
                                        if ch["ci_frac"] is not None else None)
            entry[f"holdout_gap_{side}"] = (round(ch["holdout_gap"], 4)
                                            if ch["holdout_gap"] is not None else None)
            entry[f"shrinkage_gap_{side}"] = round(v - raw_v, 4)
            vals[side] = v
        if not (0 < vals["elite"] < vals["floor"]):   # degenerate — never expected
            entry["unvalidated"] = True
            entry["elite_hbw"], entry["floor_hbw"] = round(r["p10"], 4), round(r["p90"], 4)
            entry["elite_depth"], entry["floor_depth"] = "p10", "p90"
        entry["selection_trail"] = {s: r["sides"][s]["trail"] for s in ("elite", "floor")}
        weapons[name] = entry
    return weapons


def run(collect_dir, splits, out_path, n_workers, min_shots,
        cap_frames=None, cap_rad_frames=None):
    # Convert frame caps → MODULE units (same as policy.py) and publish to the
    # module globals BEFORE the Pool fork so workers inherit them. The tick→module
    # scale lives in the aim kernel that owns _DIST_SCALE/_VEL_SCALE (A.TICK_DT_MODULE),
    # not a bench module.
    global _LEAD_CAP, _LEAD_CAP_RAD
    _LEAD_CAP = (float(cap_frames) * A.TICK_DT_MODULE) if cap_frames else None
    _LEAD_CAP_RAD = (float(cap_rad_frames) * A.TICK_DT_MODULE) if cap_rad_frames else None
    hazard_aware = _LEAD_CAP is not None or _LEAD_CAP_RAD is not None

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
                    t = per.setdefault(dmi, {"deg": {}, "norm": {}, "bounds": {}})
                    for kind in ("deg", "norm"):
                        for w, v in b[kind].items():
                            t[kind][w] = (t[kind][w] + v) if w in t[kind] else v.copy()
                    for w, (blo, bhi) in b.get("bounds", {}).items():
                        if w in t["bounds"]:
                            o0, o1 = t["bounds"][w]
                            t["bounds"][w] = (min(o0, blo), max(o1, bhi))
                        else:
                            t["bounds"][w] = (blo, bhi)
                print(f"  [{split}] {i+1}/{len(tasks)} shards", flush=True)

    # Family aggregates: sum member-weapon histograms per player so the pooled
    # distribution/reliability flow through every downstream stat unchanged. Keyed
    # by the family name string (per[d][kind] is a plain dict → string keys are fine).
    for d in per:
        for kind in ("deg", "norm"):
            for fam, members in FAMILIES.items():
                acc = None
                for w in members:
                    v = per[d][kind].get(w)
                    if v is not None:
                        acc = v.copy() if acc is None else acc + v
                if acc is not None:
                    per[d][kind][fam] = acc
        for fam, members in FAMILIES.items():
            mb = [per[d]["bounds"][w] for w in members if w in per[d].get("bounds", {})]
            if mb:
                per[d]["bounds"][fam] = (min(x[0] for x in mb), max(x[1] for x in mb))

    demos = sorted(per)
    # GLOBAL per-weapon EXACT norm bounds (best/worst human action, pooled over all
    # demos) — the duplication-robust anchor for interception_dist. Keyed by weapon id
    # / family string, matching `keys` below.
    global_bounds: dict = {}
    for d in demos:
        for w, (blo, bhi) in per[d].get("bounds", {}).items():
            if w in global_bounds:
                o0, o1 = global_bounds[w]
                global_bounds[w] = (min(o0, blo), max(o1, bhi))
            else:
                global_bounds[w] = (blo, bhi)
    half_min = max(1, min_shots // 2)
    keys = ([("all", 0)] + [(IMPULSE_NAME[w], w) for w in ALL_IMP]
            + [(fam, fam) for fam in FAMILIES])
    EDGES = {"deg": DEG_EDGES, "norm": NORM_EDGES}

    def full_median(d, w, kind):
        v = per[d][kind].get(w)
        if v is None or v.sum() < min_shots:
            return None
        return _median_from_hist(v.sum(0), EDGES[kind])

    def half_medians(d, w, kind):
        v = per[d][kind].get(w)
        if v is None or v[0].sum() < half_min or v[1].sum() < half_min:
            return None
        return [_median_from_hist(v[h], EDGES[kind]) for h in (0, 1)]

    reliability = {}; spread = {}; corpus_hist = {}
    for kind in ("deg", "norm"):
        edges = EDGES[kind]
        for name, w in keys:
            h0, h1 = [], []
            for d in demos:
                hv = half_medians(d, w, kind)
                if hv and np.isfinite(hv[0]) and np.isfinite(hv[1]):
                    h0.append(hv[0]); h1.append(hv[1])
            r, n = _spearman(h0, h1)
            sb = (2 * r / (1 + r)) if (np.isfinite(r) and r > -1) else None
            reliability[f"{kind}:{name}"] = {
                "split_half_spearman_of_median": (round(r, 4) if np.isfinite(r) else None),
                "spearman_brown": (round(sb, 4) if sb is not None else None), "n_players": n}
            meds = np.array([m for m in (full_median(d, w, kind) for d in demos)
                             if m is not None and np.isfinite(m)], float)
            if len(meds) >= 5:
                p10, p50, p90 = np.percentile(meds, [10, 50, 90])
                spread[f"{kind}:{name}"] = {
                    "n_players": int(len(meds)), "median_p10": round(float(p10), 3),
                    "median_p50": round(float(p50), 3), "median_p90": round(float(p90), 3),
                    "median_min": round(float(meds.min()), 3),
                    "median_max": round(float(meds.max()), 3),
                    "note": "lower error = better; per-player-median RANGE (min..max = the "
                            "human sustained-aim band; skill itself is quantified by the POOLED "
                            "event curve, not these medians)"}
        # corpus-wide alignment-bucket distribution (ALL attacks, overall)
        tot = np.zeros(len(edges) - 1, np.float64)
        for d in demos:
            v = per[d][kind].get(0)
            if v is not None:
                tot += v.sum(0)
        s = tot.sum()
        corpus_hist[kind] = {
            "edges": [float(e) if np.isfinite(e) else None for e in edges],
            "frac_per_bucket": [round(float(x / s), 4) for x in tot] if s > 0 else [],
            "n_attacks": int(s)}

    # ── interception_dist: the pooled distribution of the INTERCEPTION metric ──
    # For each weapon, pool ALL discharges (all demos, both halves) into one
    # distribution and report its percentiles + EXACT min/max of the alignment to the
    # interception point (origin for direct weapons, feet for RL — the lead kernel's
    # per-weapon z-anchor). This is the decode-fit reference (perweapon_human_ladder):
    # the export point rides this curve. Its trustworthy anchor is [min, max] — the
    # genuine best/worst human action, which is robust to the corpus NOT being a
    # representative population and to demo≠player duplication (extremes don't move
    # when a player records many demos). The interior percentiles are the observed
    # metric distribution (duplication-biased, named honestly — NOT a population/skill
    # ranking). p1 included so the loose end (skill p99 → metric p1) resolves.
    ADIST_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def _pctiles_from_hist(counts, edges):
        tot = counts.sum()
        if tot <= 0:
            return None
        cum = np.cumsum(counts)
        out = {}
        for p in ADIST_PCTS:
            tgt = p / 100.0 * tot
            k = min(int(np.searchsorted(cum, tgt)), len(counts) - 1)
            lo, hi = edges[k], edges[k + 1]
            prev = cum[k - 1] if k > 0 else 0.0
            if not np.isfinite(hi) or counts[k] <= 0:
                out[f"p{p}"] = round(float(lo), 2)
            else:
                out[f"p{p}"] = round(float(lo + (tgt - prev) / counts[k] * (hi - lo)), 2)
        return out

    interception_dist = {}
    for name, w in keys:
        d = {}
        for kind in ("deg", "norm"):
            tot = np.zeros(len(EDGES[kind]) - 1, np.float64)
            for dm in demos:
                v = per[dm][kind].get(w)
                if v is not None:
                    tot += v.sum(0)
            entry = {"n_attacks": int(tot.sum()),
                     "percentiles": _pctiles_from_hist(tot, EDGES[kind])}
            if kind == "norm" and w in global_bounds:
                # EXACT best/worst human action (not histogram-pinned) — the
                # duplication-robust anchor the export point rides between.
                blo, bhi = global_bounds[w]
                entry["min"] = round(float(blo), 3)
                entry["max"] = round(float(bhi), 3)
            d[kind] = entry
        interception_dist[name] = d

    # ── WITHIN-DEMO CONSISTENCY (hbw) — a ROUGH sense of the human range within the
    # interception_dist bounds ──
    # interception_dist gives the metric distribution (bounded by exact min/max). This
    # gives a rough sense of how wide a SINGLE recording's own spread is — the human
    # range within those bounds. We do NOT build a smooth per-player ability curve:
    # the corpus is not a representative population sample and demo≠player (one person
    # may have recorded many demos), so per-demo aggregates quantify RANGES only, never
    # skill. Reports, per weapon (norm/hbw):
    #   * dispersion_across_players : min/max + p10/p50/p90 of the per-demo IQR.
    #   * consistency_is_skill : Spearman(per-demo median, per-demo QCD) — SCALE-FREE,
    #     so it is not a level artifact; ~0 = tightness independent of skill.
    #   * variance_split : between-demo SD of medians vs typical within-demo SD — context
    #     only (neither is a population parameter). Ported from _aim_coh_gate_sweep.
    def _quartiles_from_hist(counts, edges):
        tot = counts.sum()
        if tot <= 0:
            return None
        cum = np.cumsum(counts)
        q = []
        for frac in (0.25, 0.50, 0.75):
            tgt = frac * tot
            k = min(int(np.searchsorted(cum, tgt)), len(counts) - 1)
            lo, hi = edges[k], edges[k + 1]
            prev = cum[k - 1] if k > 0 else 0.0
            if not np.isfinite(hi) or counts[k] <= 0:
                q.append(float(lo))
            else:
                q.append(float(lo + (tgt - prev) / counts[k] * (hi - lo)))
        return q  # [p25, p50, p75]

    def _logsd_from_hist(counts, edges):
        """Robust within-player SD in ln(hbw): std of log bin-centres weighted by
        counts (finite, positive bins only; the inf tail is dropped)."""
        c = np.asarray(counts, float)
        centres = np.array([(edges[i] + edges[i + 1]) / 2.0 if np.isfinite(edges[i + 1])
                            else np.nan for i in range(len(c))])
        m = np.isfinite(centres) & (centres > 0) & (c > 0)
        if m.sum() < 2 or c[m].sum() < 2:
            return float("nan")
        lc = np.log(centres[m]); wt = c[m]
        mu = float(np.average(lc, weights=wt))
        var = float(np.average((lc - mu) ** 2, weights=wt))
        return math.sqrt(max(0.0, var))

    within_consistency = {}
    for name, w in keys:
        meds, iqrs, qcds, logsds = [], [], [], []
        for dm in demos:
            v = per[dm]["norm"].get(w)
            if v is None or v.sum() < min_shots:
                continue
            counts = v.sum(0)
            q = _quartiles_from_hist(counts, NORM_EDGES)
            if q is None or not np.isfinite(q[1]):
                continue
            p25, p50, p75 = q
            meds.append(p50); iqrs.append(p75 - p25)
            # scale-free dispersion (quartile coefficient of dispersion) — used for the
            # consistency-vs-skill correlation so it is NOT a level artifact (raw IQR
            # scales mechanically with the median for right-skewed data).
            if (p75 + p25) > 1e-9:
                qcds.append((p75 - p25) / (p75 + p25))
            else:
                qcds.append(float("nan"))
            sd = _logsd_from_hist(counts, NORM_EDGES)
            logsds.append(sd if np.isfinite(sd) else float("nan"))
        if len(meds) < 8:
            continue
        meds = np.array(meds); iqrs = np.array(iqrs); qcds = np.array(qcds)
        # consistency-vs-skill on the SCALE-FREE dispersion (median vs QCD). Raw
        # median-vs-IQR is confounded by level, so it is not reported.
        r_skill, n_skill = _spearman(meds, qcds)
        di = np.percentile(iqrs, [10, 50, 90])
        between_sd = float(np.log(meds[meds > 0]).std(ddof=0)) if (meds > 0).sum() > 1 else float("nan")
        lg = np.asarray(logsds, float); lg = lg[np.isfinite(lg)]
        typ_within_sd = float(np.median(lg)) if len(lg) else float("nan")
        within_consistency[name] = {
            "n_players": int(len(meds)),
            "dispersion_across_players": {
                "iqr_hbw_min": round(float(iqrs.min()), 3), "iqr_hbw_max": round(float(iqrs.max()), 3),
                "iqr_hbw_p10": round(float(di[0]), 3), "iqr_hbw_p50": round(float(di[1]), 3),
                "iqr_hbw_p90": round(float(di[2]), 3),
                "note": "within-demo spread (p75-p25 of a player's own shots, hbw); min..max = "
                        "the human RANGE; p50 = the typical player. A rough sense of human "
                        "spread — not a smooth population distribution"},
            "consistency_is_skill": {
                "spearman_median_vs_qcd": (round(r_skill, 4) if np.isfinite(r_skill) else None),
                "n": n_skill,
                "note": "scale-free: median vs quartile-coeff-of-dispersion. ~0 = tightness is "
                        "independent of skill (raw median-vs-IQR would be a level artifact)"},
            "variance_split_ln_hbw": {
                "between_demo_sd": (round(between_sd, 4) if np.isfinite(between_sd) else None),
                "typical_within_demo_sd": (round(typ_within_sd, 4) if np.isfinite(typ_within_sd) else None),
                "note": "SD of ln(hbw): between = spread of per-demo medians; within = median of "
                        "per-demo own SDs. Context only — demo≠player, so neither is a population "
                        "parameter; the trustworthy anchor is interception_dist min/max"},
        }

    # ── CROSS-WEAPON TRANSFER (norm): does intercept skill transfer, or is it
    # per-weapon like coh (§9 g-factor r≈0.04)? Correlate per-player medians. ──
    medn = {w: {d: m for d in demos
                for m in [full_median(d, w, "norm")] if m is not None and np.isfinite(m)}
            for w in ALL_IMP}
    # Min shared-player count for a cell to enter the summary. Low-N Spearman is
    # noise (a handful of dual-mains can read ±1.0), so the raw mean-off-diagonal
    # is N-fragile; we report a floored mean and an N-weighted mean instead.
    XW_N_FLOOR = 30
    xw = {}; offdiag_raw = []; offdiag_floor = []
    same_phys = [(4, 5), (2, 3)]        # (NG,SNG), (SG,SSG)
    for i, wi in enumerate(ALL_IMP):
        xw[IMPULSE_NAME[wi]] = {}
        for j, wj in enumerate(ALL_IMP):
            common = [d for d in medn[wi] if d in medn[wj]]
            r, n = _spearman([medn[wi][d] for d in common], [medn[wj][d] for d in common])
            rr = round(r, 4) if np.isfinite(r) else None
            xw[IMPULSE_NAME[wi]][IMPULSE_NAME[wj]] = {"r": rr, "n": n}
            if j > i and rr is not None:
                offdiag_raw.append((rr, n))
                if n >= XW_N_FLOOR:
                    offdiag_floor.append((rr, n))
    mean_off_raw = round(float(np.mean([r for r, _ in offdiag_raw])), 4) if offdiag_raw else None
    mean_off_floor = (round(float(np.mean([r for r, _ in offdiag_floor])), 4)
                      if offdiag_floor else None)
    wsum = sum(n for _, n in offdiag_floor)
    mean_off_nw = (round(float(sum(r * n for r, n in offdiag_floor) / wsum), 4)
                   if wsum else None)
    same_phys_r = {f"{IMPULSE_NAME[a]}-{IMPULSE_NAME[b]}":
                   xw[IMPULSE_NAME[a]][IMPULSE_NAME[b]] for a, b in same_phys}
    cross_weapon = {"spearman_matrix_norm": xw,
                    "mean_offdiag_r_raw": mean_off_raw,
                    "mean_offdiag_r_nfloor": mean_off_floor,
                    "mean_offdiag_r_nweighted": mean_off_nw,
                    "n_floor": XW_N_FLOOR,
                    "n_cells_total": len(offdiag_raw), "n_cells_kept": len(offdiag_floor),
                    "same_physics_pairs": same_phys_r,
                    "coh_reference_mean_pairwise_r": 0.0434,
                    "note": "per-player median intercept (hbw) correlated across weapons; "
                            "raw mean is N-fragile (low-N cells = noise) — trust nfloor/nweighted "
                            f"(cells with >= {XW_N_FLOOR} shared players). high -> general factor; "
                            "~0 -> per-weapon like coh"}

    # ── PLACEMENT ANCHORS: frozen per-corpus selection (skill-curves §16.3) ──
    dup_dropped, dup_examples = _duplicate_census(per, demos)
    census, pops = _anchor_populations(per, demos, keys, min_shots, dup_dropped)
    census["duplicates_dropped"] = len(dup_dropped)
    census["duplicate_examples"] = dup_examples[:8]
    rel_sb = {name: reliability[f"norm:{name}"]["spearman_brown"] for name, _w in keys}
    anchor_weapons = _select_placement_anchors(pops, rel_sb)
    placement_anchors = {
        "anchors_version": ANCHORS_VERSION,
        "procedure": ANCHOR_PROCEDURE,
        "config": {
            "elite_depths": list(ANCHOR_ELITE_DEPTHS),
            "floor_depths": list(ANCHOR_FLOOR_DEPTHS),
            "boot_resamples": ANCHOR_BOOT, "ci_pcts": list(ANCHOR_CI),
            "ci_frac_max": ANCHOR_CI_FRAC_MAX, "min_sel_demos": ANCHOR_MIN_SEL,
            "contam_lo_hbw": ANCHOR_CONTAM_LO_HBW, "contam_hi_hbw": ANCHOR_CONTAM_HI_HBW,
            "reliability_min_sb": ANCHOR_RELIABILITY_MIN,
            "dup_median_rtol": ANCHOR_DUP_MED_RTOL,
            "half_min_shots": min_shots, "seed": ANCHOR_SEED},
        "weapons": anchor_weapons,
    }

    out = {
        "title": "INTERCEPTION distribution (alignment-at-attack) — pooled metric distribution "
                 "anchored by exact min/max",
        "defs": {"discharge": "attack_finished rising edge",
                 "metric": "lead-corrected angle to most-aligned visible enemy at discharge; "
                           "decode-fit reference = interception_dist (pooled metric percentiles + "
                           "exact min/max bounds). per-demo aggregates are RANGES only, never skill "
                           "(corpus not a representative population; demo != player)",
                 "deg": "raw degrees", "norm": "hitbox-half-widths (range-invariant; <1 = on body)",
                 "split_half": "EVEN/ODD discharge parity; reliability of the per-demo median"},
        "config": {"splits": splits, "n_players": len(demos), "min_shots": min_shots,
                   "lead": ("hazard_aware" if hazard_aware else "linear"),
                   "lead_hold_cap_frames": cap_frames, "lead_hold_cap_radial_frames": cap_rad_frames,
                   "note_families": "SG+SSG and NG+SNG share identical lead geometry → "
                                    "distributions must match (sanity); caps bite NG/SNG/GL/RL only"},
        "reliability_of_median": reliability,
        "spread_of_median": spread,
        "census": census,
        "placement_anchors": placement_anchors,
        "corpus_alignment_distribution": corpus_hist,
        "interception_dist": interception_dist,
        "within_player_consistency": within_consistency,
        "cross_weapon_transfer": cross_weapon,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{len(demos)} players")
    for kind in ("deg", "norm"):
        unit = "deg" if kind == "deg" else "hbw"
        print(f"\n==== ALIGNMENT-AT-ATTACK [{kind}] — reliability of per-player MEDIAN ====")
        print(f"{'bucket':>8} {'r(median)':>10} {'SB':>8} {'n':>5} | {'med p10':>8} {'p50':>8} {'p90':>8}")
        for name, w in keys:
            rel = reliability[f"{kind}:{name}"]; sp = spread.get(f"{kind}:{name}")
            sps = (f"{sp['median_p10']:>8} {sp['median_p50']:>8} {sp['median_p90']:>8}"
                   if sp else f"{'':>8} {'':>8} {'':>8}")
            print(f"{name:>8} {str(rel['split_half_spearman_of_median']):>10} "
                  f"{str(rel['spearman_brown']):>8} {rel['n_players']:>5} | {sps}  ({unit})")
        ch = corpus_hist[kind]
        print(f"  corpus distribution over {ch['n_attacks']} attacks (frac per bucket):")
        edges = ch["edges"]
        cells = [f"{edges[i]}-{edges[i+1] if edges[i+1] is not None else 'inf'}:{ch['frac_per_bucket'][i]}"
                 for i in range(len(ch['frac_per_bucket']))]
        print("    " + "  ".join(cells))
    print("\n==== interception_dist (to interception point; origin, RL=feet) — pctiles + exact min/max ====")
    for unit, kind in (("deg", "deg"), ("hbw", "norm")):
        mm = "  min..max" if kind == "norm" else ""
        print(f"  [{unit}]  {'wpn':>5} {'nAtk':>7}  " + "  ".join(f"{('p'+str(p)):>6}" for p in ADIST_PCTS) + mm)
        for name, w in keys:
            e = interception_dist[name][kind]; pc = e["percentiles"]
            if pc is None:
                continue
            mms = f"  {e['min']}..{e['max']}" if kind == "norm" and "min" in e else ""
            print(f"        {name:>5} {e['n_attacks']:>7}  " + "  ".join(f"{pc['p'+str(p)]:>6}" for p in ADIST_PCTS) + mms)
    print("\n==== WITHIN-DEMO CONSISTENCY (hbw) — rough human range + scale-free is-it-skill ====")
    print(f"  {'wpn':>6} {'n':>5} | {'iqr min':>8} {'p50':>7} {'max':>7} | "
          f"{'r(med,qcd)':>11} | {'btwn_sd':>8} {'wthn_sd':>8}  (ln hbw)")
    for name, _w in keys:
        wc = within_consistency.get(name)
        if wc is None:
            continue
        di = wc["dispersion_across_players"]; sk = wc["consistency_is_skill"]
        vs = wc["variance_split_ln_hbw"]
        print(f"  {name:>6} {wc['n_players']:>5} | {di['iqr_hbw_min']:>8} {di['iqr_hbw_p50']:>7} "
              f"{di['iqr_hbw_max']:>7} | {str(sk['spearman_median_vs_qcd']):>11} | "
              f"{str(vs['between_demo_sd']):>8} {str(vs['typical_within_demo_sd']):>8}")
    print("\n==== CROSS-WEAPON TRANSFER of intercept (per-player median, hbw) ====")
    nm = [IMPULSE_NAME[w] for w in ALL_IMP]
    print(f"{'':>5}" + "".join(f"{n:>11}" for n in nm))
    for a in nm:
        row = f"{a:>5}"
        for b in nm:
            c = cross_weapon["spearman_matrix_norm"][a][b]
            cell = f"{c['r']:+.2f}/{c['n']}" if c["r"] is not None else "na"
            row += f"{cell:>11}"
        print(row)
    print(f"  mean off-diagonal r: raw={cross_weapon['mean_offdiag_r_raw']}  "
          f"n>={cross_weapon['n_floor']}={cross_weapon['mean_offdiag_r_nfloor']}  "
          f"n-weighted={cross_weapon['mean_offdiag_r_nweighted']}  "
          f"({cross_weapon['n_cells_kept']}/{cross_weapon['n_cells_total']} cells kept; "
          f"coh ref 0.0434)")
    print(f"  same-physics: " + "  ".join(f"{k}: r={v['r']} (n={v['n']})"
                                          for k, v in cross_weapon["same_physics_pairs"].items()))
    print("\n==== PLACEMENT ANCHORS (frozen per-corpus selection, §16.3; hbw) ====")
    caps = {k: v for k, v in census["cap_pinned_excluded"].items() if v}
    print(f"  census: {census['duplicates_dropped']} duplicate demos dropped "
          f"{census['duplicate_examples']}; cap-pinned excluded {caps}")
    print(f"  {'wpn':>7} {'elite':>8} {'depth':>6} {'floor':>8} {'depth':>6} "
          f"{'SB':>7} {'n':>5} {'nHalf':>6}  flags")
    for name, _w in keys:
        a = anchor_weapons.get(name)
        if a is None:
            continue
        flags = ",".join(f for f in ("unvalidated", "family_borrowed", "shrunk")
                         if a[f]) or "-"
        print(f"  {name:>7} {a['elite_hbw']:>8} {a['elite_depth']:>6} "
              f"{a['floor_hbw']:>8} {a['floor_depth']:>6} "
              f"{str(a['reliability_sb']):>7} {a['n_demos']:>5} {a['n_strict_half']:>6}  {flags}")
    print(f"\nWritten -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="both", choices=["val", "train", "both"])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--min-shots", type=int, default=40)
    ap.add_argument("--lead-hold-cap-frames", type=float, default=None,
                    help="TANGENTIAL hazard lead cap in 20Hz frames (deployed a25 = 4.0)")
    ap.add_argument("--lead-hold-cap-radial-frames", type=float, default=None,
                    help="RADIAL hazard lead cap in 20Hz frames (deployed a25 = 5.0)")
    ap.add_argument("--linear", action="store_true",
                    help="EXPLICITLY build the un-capped (linear-lead) band. Required to run "
                         "without lead caps — there is no silent default: forgetting the caps "
                         "would otherwise build the deployed artifact with the WRONG lead law.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <collect>/human_baseline/_aim_intercept_skill.json)")
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_aim_intercept_skill.json"
    # NO SILENT DEFAULT: the deployed decode uses hazard lead caps (a25 = 4.0/5.0), and
    # this artifact is the single source of truth for every downstream intercept gate.
    # Refuse to build without an explicit choice — caps OR --linear.
    if args.lead_hold_cap_frames is None and args.lead_hold_cap_radial_frames is None and not args.linear:
        ap.error("must pass the deployed hazard lead caps "
                 "(--lead-hold-cap-frames / --lead-hold-cap-radial-frames; a25 = 4.0 / 5.0) "
                 "OR --linear to explicitly opt into the un-capped band")
    splits = ["train", "val"] if args.split == "both" else [args.split]
    lead = ("hazard-aware" if (args.lead_hold_cap_frames or args.lead_hold_cap_radial_frames)
            else "linear")
    print(f"Collect: {args.collect_dir}  splits={splits}  workers={args.workers}  lead={lead} "
          f"(cap={args.lead_hold_cap_frames}/{args.lead_hold_cap_radial_frames} frames)")
    run(args.collect_dir, splits, out, args.workers, args.min_shots,
        args.lead_hold_cap_frames, args.lead_hold_cap_radial_frames)


if __name__ == "__main__":
    main()
