"""ACQUISITION as a TWO-COMPONENT process — where the per-player skill SPREAD lives.

The prior CLEAN decomposition (runs/head_probe/_aim_primitive_decomp_clean.json)
measured acquisition only as the BALLISTIC PRIMARY sweep — endpoint accuracy
(p90/p10 ≈ 1.28×) and peak corrective velocity (≈1.53×) — and concluded
acquisition is a "weak" skill axis. But the motor-control literature
(Woodworth's two-component model; Fitts' law; Meyer's optimized-submovement
model) says discrete-targeting skill lives in THROUGHPUT (speed×accuracy jointly)
and in the FEEDBACK-DRIVEN CORRECTIVE/HOMING phase — neither of which the prior
test measured. This script adds those, re-using the SAME flick detection +
LOS/origin-angle geometry kernels (no new physics).

Two-component segmentation of each FLICK EVENT (20 Hz, 50 ms/frame):
  * A FLICK EVENT opens when the origin bearing to the best LOS actor jumps to
    >= FLICK_ONSET_DEG having been below it (identical to the clean decomp's
    onset rule). Its window runs until the angle settles < SETTLE_DEG (into the
    hit cone W), window cap, or contact loss.
  * PRIMARY phase = onset → end of the initial ballistic sweep. The primary end
    is the first local angular-VELOCITY MINIMUM after the initial peak closure
    (the deceleration trough that ends the open-loop sweep). Measure:
      - primary endpoint LOS angle (deg at primary end)
      - primary peak angular velocity (deg/frame, max closure in the primary)
  * CORRECTIVE phase = primary-end → cone settle (< SETTLE_DEG). Measure:
      - SUBMOVEMENT COUNT: number of velocity RE-ACCELERATIONS (secondary
        closure peaks) after the primary, i.e. additional feedback corrections.
      - HOMING TIME: frames from primary-end to cone settle (NaN if never
        settles inside the window → a failed/uncounted homing).
  * THROUGHPUT (ISO 9241-9 Shannon effective): IDe = log2(D / We + 1), D = initial-
    onset LOS angle (deg), We = 4.133*SD(settle endpoints) = the EFFECTIVE width the
    movement actually used (accuracy-adjusted; not the nominal SETTLE_DEG cone). MT =
    total flick time onset→settle (seconds, frames/20). TP = IDe / MT (bits/s). We is
    a population quantity, so throughput is computed after the pass (run(); the model
    side uses the same human We). Only flicks that SETTLE get a TP.
  * VELOCITY-NORMALIZED PRIMARY ERROR = primary endpoint error (deg) ÷ primary
    peak velocity (deg/frame) — the Fitts-relevant "did you land close GIVEN
    your speed" (low = good).

Per-player traits (median over that player's flick events, with an event floor):
  throughput, submovement_count, homing_time_frames, velnorm_primary_err,
  primary_endpoint (continuity w/ prior), primary_peakvel (continuity).

KEY ANALYSES
  1. WHICH COMPONENT SPREADS? per-player p10/p50/p90 + p90/p10 ratio + split-half
     (even/odd event parity) reliability for each metric. Compare to the prior
     primary-endpoint 1.28× / peakvel 1.53× and to pursuit (3.4–7.1×) /
     interception (3.86×). Hypothesis: throughput and/or corrective metrics
     (submovement count, homing time) spread WIDER than the primary endpoint.
  2. INDEPENDENCE: correlate acquisition-throughput against pursuit (coh) and
     interception (lead) across players — separate 3-skill axis or coupled?
  3. VERDICT: does acquisition become a real spreading skill once measured the
     literature-correct way (throughput / corrective efficiency), or genuinely
     flat? Weighted toward throughput + homing-time (more robust to the coarse
     20 Hz sampling than fine submovement counting).

CAVEAT (stated in output): 20 Hz = 50 ms/frame; a flick+correction is ~2–6
frames, so submovement segmentation is COARSE — we detect a primary + 1–2
corrections only and report that resolution limit; we do NOT claim fine
submovement microstructure. Verdict weighted toward throughput + homing time.

All geometry reuses the canonical kernels from aim_skill.py
(_lead_aim_angle_deg_live / _build_physics_tables / _DIST_SCALE / _VEL_SCALE /
MAX_ENTITY_SLOTS / TOKEN_ACTOR) and the flick-onset rule + origin-angle helper
matching aim_primitive_decomp_clean.py verbatim. Engaged-LOS frames only
(target_probs argmax>0 AND actor recency==0). No op filter. HUMAN qwd, offline.

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.acquisition \\
      --collect-dir artifacts/collect/qwd [--split both] [--workers N] \\
      [--min-frames 100] \\
      [--out <collect>/human_baseline/_acq_submovement.json]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from qnn.eval import aim_kernel as A

IMPULSE_NAME = {1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
                6: "GL", 7: "RL", 8: "LG"}
DIRECT_FIRE = ["SG", "SSG", "NG", "SNG", "LG"]   # pursuit coh-valid hitscan
PROJECTILE_NAILS = {4, 5}                        # NG, SNG — genuine lead

# ── Phase-segmentation thresholds (degrees / frames at 20 Hz) — matched to the
#    clean decomp so the flick population is identical. ────────────────────────
FLICK_ONSET_DEG   = 18.0   # origin bearing jump that opens a flick event (= D, approx)
SETTLE_DEG        = 5.0    # hit cone W; angle considered settled
FLICK_WIN_FRAMES  = 30     # ballistic-correction window cap (1.5 s)
LEAD_AVEL_DEG     = 3.0    # fast-moving threshold for the interception metric (nails)
TOT_AVEL_DEG      = 3.0    # moving target for pursuit context
TOT_CONE_DEG      = 5.0
MIN_ENGAGED_FRAMES_EP = 30
FPS               = 20.0   # 20 Hz, 50 ms / frame

# Two-component detection knobs.
#  - A "re-acceleration" (submovement) is a closure-rate local max that exceeds a
#    floor (so noise-level wiggles below this don't count as corrections).
SUBMOVE_VEL_FLOOR = 1.0    # deg/frame; a corrective peak must close at least this fast

# Minimums for a player to receive a trait value.
MIN_FLICK_EVENTS  = 5      # discrete-targeting events with a measurable primary
MIN_SETTLED_FLICKS = 5     # events that reach the cone (for TP / homing)
MIN_LEAD_FRAMES   = 50     # fast-moving nail frames (interception axis)


def _origin_angle_deg(rel: np.ndarray) -> np.ndarray:
    """Forward-axis (+x) bearing angle to the target's CURRENT position, deg."""
    norms = np.maximum(np.linalg.norm(rel, axis=-1), 1e-9)
    cos_a = np.clip(rel[..., 0] / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _decompose_flick(win: np.ndarray) -> dict[str, Any] | None:
    """Two-component decomposition of one flick angle profile `win` (deg, dense
    contiguous frames from onset). Returns None if too short to decompose.

    win[0] = onset angle (~D). closure[k] = win[k]-win[k+1] (positive = closing).
    PRIMARY = onset → first velocity minimum after the initial peak closure.
    """
    if len(win) < 2:
        return None
    closure = -np.diff(win)                       # (L-1,) per-frame angle closure
    L = len(closure)
    if L == 0:
        return None

    # Primary peak = index of the maximum closure rate (the ballistic sweep peak).
    pk = int(np.argmax(closure))
    primary_peakvel = float(closure[pk])

    # Primary END = first velocity MINIMUM (deceleration trough) AFTER the peak.
    # Search for the first local min of closure at index j>pk (closure[j] <=
    # closure[j-1] and < closure[j+1], or the first non-positive closure). The
    # primary endpoint angle is win[primary_end_frame].
    primary_end_k = L                              # default: sweep runs to window end
    for j in range(pk + 1, L):
        # trough: closure stops decreasing (re-acceleration begins) or goes <=0
        if closure[j] <= 0.0:
            primary_end_k = j                      # angle frame index = j (post-decel)
            break
        if j + 1 < L and closure[j] < closure[j + 1] and closure[j] <= closure[j - 1]:
            primary_end_k = j
            break
    else:
        # never found a trough → the whole window is one ballistic sweep
        primary_end_k = L
    # primary_end_k is a CLOSURE index; the corresponding ANGLE frame is the
    # frame AFTER applying primary_end_k closures, i.e. win[primary_end_k].
    pe_frame = min(primary_end_k, len(win) - 1)
    primary_endpoint = float(win[pe_frame])

    # CORRECTIVE phase = frames after the primary end.
    corr_closure = closure[primary_end_k:]         # remaining closures
    # SUBMOVEMENT COUNT = number of re-acceleration peaks (local maxima of the
    # closure rate above the floor) in the corrective phase. Coarse at 20 Hz:
    # this resolves a primary + ~1–2 corrections only.
    submoves = 0
    if len(corr_closure) >= 1:
        c = corr_closure
        for j in range(len(c)):
            if c[j] < SUBMOVE_VEL_FLOOR:
                continue
            left_ok = (j == 0) or (c[j] >= c[j - 1])
            right_ok = (j == len(c) - 1) or (c[j] > c[j + 1])
            # require a genuine re-acceleration: a positive rise into this peak
            rises = (j == 0 and c[j] >= SUBMOVE_VEL_FLOOR) or (j > 0 and c[j] > c[j - 1])
            if left_ok and right_ok and rises:
                submoves += 1

    # HOMING: frames from primary end to cone settle (< SETTLE_DEG). NaN if the
    # window never settles after the primary end.
    settle_k = None
    for k in range(pe_frame, len(win)):
        if win[k] < SETTLE_DEG:
            settle_k = k
            break
    homing_frames = float(settle_k - pe_frame) if settle_k is not None else float("nan")

    # TOTAL flick movement time (onset → settle), for Fitts MT. NaN if never settles.
    total_settle_k = None
    for k in range(len(win)):
        if win[k] < SETTLE_DEG:
            total_settle_k = k
            break
    mt_frames = float(total_settle_k) if total_settle_k is not None else float("nan")
    # Angular error at the settle frame (within the cone). Its population SD gives the
    # ISO 9241-9 EFFECTIVE width We = 4.133*SD — the accuracy the movement actually
    # used. NaN if never settles.
    settle_angle = float(win[total_settle_k]) if total_settle_k is not None else float("nan")

    onset_D = float(win[0])
    # velocity-normalized primary error (Fitts-relevant). primary endpoint error
    # = how far from cone center the ballistic sweep landed; normalize by speed.
    primary_err = max(0.0, primary_endpoint)       # deg from forward axis (~0 ideal)
    velnorm_primary_err = (primary_err / primary_peakvel
                           if primary_peakvel > 1e-6 else float("nan"))

    # Throughput is NOT computed here — it needs the ISO effective width We, a
    # population quantity derived from all settle endpoints. We store (onset_D,
    # mt_frames, settle_angle) per flick and compute the Shannon effective throughput
    # `shannon_throughput()` once We is known (run() for human; cell_* for the model,
    # using the shared human We so both ride the same instrument).
    return {
        "onset_D": onset_D,
        "primary_endpoint": primary_endpoint,
        "primary_peakvel": primary_peakvel,
        "submoves": float(submoves),
        "homing_frames": homing_frames,
        "mt_frames": mt_frames,
        "settle_angle": settle_angle,
        "velnorm_primary_err": velnorm_primary_err,
        "settled": settle_k is not None,
    }


def effective_width(settle_angles) -> float:
    """ISO 9241-9 effective target width We = 4.133 * SD(endpoints). Our endpoints are
    unsigned angular errors at settle (in [0, SETTLE_DEG)); we use their SD as the
    endpoint-scatter proxy (documented approximation — no signed over/undershoot at
    20 Hz). Falls back to the nominal cone if too few settled flicks."""
    a = np.asarray([x for x in settle_angles if np.isfinite(x)], float)
    if len(a) < 30:
        return float(SETTLE_DEG)
    return float(4.133 * a.std(ddof=0))


def shannon_throughput(onset_D, mt_frames, we: float) -> float:
    """ISO Shannon effective throughput TP = IDe / MT, IDe = log2(D/We + 1), MT in
    seconds (frames/FPS). NaN unless the flick settled (mt_frames finite) with D>0."""
    if not (np.isfinite(mt_frames) and mt_frames > 0 and onset_D > 0 and we > 0):
        return float("nan")
    ide = math.log2(onset_D / we + 1.0)
    return ide / (mt_frames / FPS)


def _episode_metrics(ep_cnt, ep_rel, ep_vel, ep_typ, ep_rec, ep_weapon, ep_engaged):
    """Per-episode flick events (two-component) + per-weapon pursuit coh + lead."""
    T = len(ep_cnt)
    # Shared densify (raw padded arrays; caller applies the unit scale) — aim_kernel.
    all_rel, all_vel, all_los = A.densify_entities(ep_cnt, ep_rel, ep_vel, ep_typ, ep_rec)
    all_rel *= (1.0 / A._DIST_SCALE)
    all_vel *= (1.0 / A._VEL_SCALE)

    has_los = all_los.any(axis=1)
    eng = ep_engaged & has_los
    eng_idx = np.where(eng)[0]

    res: dict[str, Any] = {
        # per-flick-event trait lists (+ parity for split-half). throughput is derived
        # later from (onset_D, mt_frames) + the shared effective width, so we store the
        # settle endpoint instead of a throughput here.
        "settle_angle": [], "submoves": [], "homing": [], "velnorm": [],
        "primary_endpoint": [], "primary_peakvel": [], "parity": [],
        "mt_frames": [], "onset_D": [],
        "n_flick_events": 0, "n_settled": 0,
        # interception (lead) — NAILS only, fast-moving
        "lead_n_moving": 0, "lead_n_better": 0,
        # pursuit (per-weapon coh, lead-angle deployed metric)
        "weapon_coh": {},
    }
    if not len(eng_idx):
        return res

    rel = all_rel[eng_idx]
    vel = all_vel[eng_idx]
    los = all_los[eng_idx]
    imp = A.action_attack_context(ep_weapon)[eng_idx]

    lead_ang = A._lead_aim_angle_deg_live(rel, vel, imp, _WN)   # (M,16)
    orig_ang = _origin_angle_deg(rel)                          # (M,16)
    lead_m = np.where(los, lead_ang, np.inf)
    orig_m = np.where(los, orig_ang, np.inf)

    best_slot = np.argmin(lead_m, axis=1)
    rows = np.arange(len(eng_idx))
    min_lead = lead_m[rows, best_slot]
    min_orig = orig_m[rows, best_slot]
    finite = np.isfinite(min_lead) & np.isfinite(min_orig)
    best_rel = rel[rows, best_slot]

    # target bearing angular velocity (deg/frame), contiguous in time
    bu = best_rel / np.maximum(np.linalg.norm(best_rel, axis=-1, keepdims=True), 1e-9)
    avel = np.full(len(eng_idx), np.nan, dtype=np.float64)
    if len(eng_idx) > 1:
        dt = np.diff(eng_idx)
        cont = dt == 1
        dots = np.clip((bu[1:] * bu[:-1]).sum(axis=1), -1.0, 1.0)
        a = np.degrees(np.arccos(dots))
        avel[1:][cont] = a[cont]

    # ── INTERCEPTION (lead) primitive — NAILS only, fast-moving ──
    is_nail = np.array([int(w) in PROJECTILE_NAILS for w in imp])
    lead_mask = finite & is_nail & (avel >= LEAD_AVEL_DEG) & np.isfinite(avel)
    if lead_mask.any():
        better = (min_lead[lead_mask] < min_orig[lead_mask])
        res["lead_n_moving"] = int(lead_mask.sum())
        res["lead_n_better"] = int(better.sum())

    # ── FLICK detection (origin-angle track), identical onset rule ──
    track = np.full(T, np.nan, dtype=np.float64)
    track[eng_idx[finite]] = min_orig[finite]

    onsets: list[int] = []
    last_val = np.nan
    for t in range(T):
        v = track[t]
        if np.isnan(v):
            last_val = np.nan
            continue
        if (v >= FLICK_ONSET_DEG) and (np.isnan(last_val) or last_val < FLICK_ONSET_DEG):
            onsets.append(t)
        last_val = v

    ev_i = 0
    for ot in onsets:
        win_vals = [track[ot]]
        for k in range(1, min(FLICK_WIN_FRAMES, T - ot)):
            v = track[ot + k]
            if np.isnan(v):
                break
            win_vals.append(v)
            if v < SETTLE_DEG:
                break
        wv = np.array(win_vals, dtype=np.float64)
        dec = _decompose_flick(wv)
        if dec is None:
            continue
        res["n_flick_events"] += 1
        res["parity"].append(ev_i % 2)
        res["primary_endpoint"].append(dec["primary_endpoint"])
        res["primary_peakvel"].append(dec["primary_peakvel"])
        res["submoves"].append(dec["submoves"])
        res["onset_D"].append(dec["onset_D"])
        res["velnorm"].append(dec["velnorm_primary_err"])
        # homing / settle_angle / mt only defined when the flick settles
        res["homing"].append(dec["homing_frames"])
        res["settle_angle"].append(dec["settle_angle"])
        res["mt_frames"].append(dec["mt_frames"])
        if dec["settled"]:
            res["n_settled"] += 1
        ev_i += 1

    # ── pursuit per-weapon coh (lead-angle, deployed metric) ──
    fin_imp = imp[finite]
    fin_lead = min_lead[finite]
    within = fin_lead < 5.0
    for w in np.unique(fin_imp):
        m = fin_imp == w
        res["weapon_coh"][int(w)] = np.array(
            [int(m.sum()), int((m & within).sum())], dtype=np.int64)
    return res


_WN, _ZD = A._build_physics_tables()


# ── accumulators ─────────────────────────────────────────────────────────────

def _new_acc() -> dict[str, Any]:
    return {
        "settle_angle": [], "submoves": [], "homing": [], "velnorm": [],
        "primary_endpoint": [], "primary_peakvel": [], "parity": [],
        "mt_frames": [], "onset_D": [],
        "n_flick_events": 0, "n_settled": 0,
        "lead_n_moving": 0, "lead_n_better": 0,
        "weapon_coh": {},
    }


def _merge(b: dict[str, Any], m: dict[str, Any]) -> None:
    for k in ("settle_angle", "submoves", "homing", "velnorm",
              "primary_endpoint", "primary_peakvel", "parity",
              "mt_frames", "onset_D"):
        b[k].extend(m[k])
    for k in ("n_flick_events", "n_settled", "lead_n_moving", "lead_n_better"):
        b[k] += m[k]
    for w, pair in m["weapon_coh"].items():
        acc = b["weapon_coh"].setdefault(w, np.zeros(2, dtype=np.int64))
        acc += pair


def _worker(args: tuple) -> dict[int, dict[str, Any]]:
    sh, data_dir_str = args
    out: dict[int, dict[str, Any]] = {}
    for _ei, demo_idx, fsl, esl, arr in A.iter_shard_episodes(
            sh, data_dir_str,
            obs=("entity_rel", "entity_vel", "entity_types", "entity_recency"),
            acts=("target_probs", "attack")):
        ep_engaged = np.asarray(arr["target_probs"][fsl]).argmax(axis=1) > 0
        if int(ep_engaged.sum()) < MIN_ENGAGED_FRAMES_EP:
            continue
        m = _episode_metrics(
            ep_cnt=np.asarray(arr["entity_count"][fsl], dtype=np.int32),
            ep_rel=np.asarray(arr["entity_rel"][esl]),
            ep_vel=np.asarray(arr["entity_vel"][esl]),
            ep_typ=np.asarray(arr["entity_types"][esl]),
            ep_rec=np.asarray(arr["entity_recency"][esl]),
            ep_weapon=np.asarray(arr["attack"][fsl]),
            ep_engaged=ep_engaged,
        )
        b = out.setdefault(int(demo_idx), _new_acc())
        _merge(b, m)
    return out


# ── stats helpers ────────────────────────────────────────────────────────────

def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return float("nan"), n
    from scipy.stats import spearmanr
    r, _ = spearmanr(x[mask], y[mask])
    return float(r), n


def _nanmedian(vals: list[float]) -> float:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=np.float64)
    return float(np.median(v)) if len(v) else float("nan")


def _spread(x: np.ndarray) -> dict[str, Any]:
    """Per-player dispersion. p90/p10 ratio is the headline (comparable to the
    program's pursuit/interception/prior-endpoint ratios), BUT when a metric is
    floored at 0 (submovement count, homing time at 20 Hz) p10≈0 makes the ratio
    blow up — so we ALSO report a robust quartile-dispersion (IQR/median) and a
    p90/p50 ratio, which stay finite for zero-floored metrics. The verdict reads
    the right column per metric (ratio for the continuous metrics; QCD/p90-p50
    for the zero-floored count/time)."""
    v = x[np.isfinite(x)]
    if len(v) < 10:
        return {"n": int(len(v)), "note": "too few players"}
    p10 = float(np.percentile(v, 10))
    p25 = float(np.percentile(v, 25))
    p50 = float(np.percentile(v, 50))
    p75 = float(np.percentile(v, 75))
    p90 = float(np.percentile(v, 90))
    p99 = float(np.percentile(v, 99))
    a10, a90, a99 = abs(p10), abs(p90), abs(p99)
    lo, hi = min(a10, a90), max(a10, a90)
    ratio = round(hi / lo, 3) if lo > 1e-9 else float("inf")
    lo9, hi9 = min(a10, a99), max(a10, a99)
    ratio99 = round(hi9 / lo9, 3) if lo9 > 1e-9 else float("inf")
    # robust dispersion that survives a 0 floor:
    #   quartile coefficient of dispersion = (p75-p25)/(p75+p25)
    #   p90/p50 ratio (finite when p50>0)
    qcd = (round((p75 - p25) / (p75 + p25), 3)
           if (p75 + p25) > 1e-9 else float("inf"))
    r9050 = round(abs(p90) / abs(p50), 3) if abs(p50) > 1e-9 else float("inf")
    return {"n": int(len(v)), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "p10": round(p10, 4), "p25": round(p25, 4),
            "p50": round(p50, 4), "p75": round(p75, 4),
            "p90": round(p90, 4), "p99": round(p99, 4),
            "ratio_p90_p10": ratio, "ratio_p99_p10": ratio99,
            "ratio_p90_p50": r9050, "qcd_iqr": qcd}


# ── runner ─────────────────────────────────────────────────────────────────

def run(collect_dir: Path, splits: list[str], out_path: Path,
        n_workers: int, min_frames: int) -> None:
    per: dict[int, dict[str, Any]] = {}
    for split in splits:
        dd = collect_dir / f"precomputed_{split}"
        man = dd / "manifest.json"
        if not man.exists():
            print(f"  [{split}] no manifest, skip")
            continue
        shards = json.loads(man.read_text())["shards"]
        tasks = [(sh, str(dd)) for sh in shards]
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
                for dmi, acc in r.items():
                    if dmi not in per:
                        per[dmi] = _new_acc()
                    _merge(per[dmi], acc)
                print(f"  [{split}] {i+1}/{len(tasks)} shards done", flush=True)

    # ── ISO effective width We = 4.133*SD(settle endpoints), pooled over ALL settled
    # flicks — the shared instrument. Human throughput below AND the closed-loop model
    # throughput (cell_acquisition_throughput) both use this SAME We so they are
    # comparable. Then reconstitute each demo's per-flick throughput from its stored
    # (onset_D, mt_frames) via the Shannon effective form. ──
    WE = effective_width([x for acc in per.values() for x in acc["settle_angle"]])
    for acc in per.values():
        acc["throughput"] = [shannon_throughput(D, mt, WE)
                             for D, mt in zip(acc["onset_D"], acc["mt_frames"])]

    # ── per-player traits (median over events) + split-half halves ──
    pid: list[int] = []
    # all oriented so HIGHER = better skill where it has a sign:
    #   throughput   : bits/s, higher = better (raw)
    #   submoves     : fewer corrections = better → store -median (higher=better)
    #   homing       : fewer frames = better → store -median (higher=better)
    #   velnorm      : lower error/speed = better → store -median (higher=better)
    #   primary_endpoint : closer-to-cone (smaller deg) better → store -median
    #   primary_peakvel  : faster slew, raw median (continuity w/ prior)
    tp: list[float] = []; sm: list[float] = []; hm: list[float] = []
    vn: list[float] = []; pe: list[float] = []; pv: list[float] = []
    # RAW (unsigned) medians kept for spread reporting in natural units
    tp_raw: list[float] = []; sm_raw: list[float] = []; hm_raw: list[float] = []
    vn_raw: list[float] = []; pe_raw: list[float] = []; pv_raw: list[float] = []
    mt_raw: list[float] = []; D_raw: list[float] = []
    # the median submove count is floored at 0 at 20 Hz (most flicks need 0–1
    # corrections), so it carries no spread by construction. The MEAN submove
    # count and the CORRECTION RATE (fraction of flicks with >=1 submove) are
    # continuous and discriminate players even when the median collapses.
    sm_mean_raw: list[float] = []; corr_rate_raw: list[float] = []
    sm_mean: list[float] = []; corr_rate: list[float] = []
    hm_mean_raw: list[float] = []        # mean homing (continuous-r robust)
    # split-half (even/odd event parity) — RAW per-player stats, for reliability
    halves: dict[str, tuple[list[float], list[float]]] = {
        k: ([], []) for k in
        ("throughput", "submoves", "submove_mean", "correction_rate",
         "homing", "homing_mean", "velnorm",
         "primary_endpoint", "primary_peakvel")}
    lead_score: list[float] = []
    n_events: list[int] = []; n_settled: list[int] = []
    wcoh: dict[str, dict[int, float]] = {n: {} for n in IMPULSE_NAME.values()}

    for dmi, acc in per.items():
        pid.append(dmi)
        n_events.append(acc["n_flick_events"])
        n_settled.append(acc["n_settled"])
        par = np.array(acc["parity"], dtype=np.int64)

        def trait(key, settle_floor=False):
            vals = np.array(acc[key], dtype=np.float64)
            ne = acc["n_flick_events"]
            ok_events = (acc["n_settled"] if settle_floor else ne)
            floor = MIN_SETTLED_FLICKS if settle_floor else MIN_FLICK_EVENTS
            if ok_events < floor:
                return float("nan"), float("nan"), float("nan")
            med = _nanmedian(list(vals))
            ev = _nanmedian(list(vals[par == 0])) if len(par) else float("nan")
            od = _nanmedian(list(vals[par == 1])) if len(par) else float("nan")
            return med, ev, od

        def trait_mean(key, settle_floor=False, ge1=False):
            """Per-player MEAN (or, if ge1, fraction with value>=1) + even/odd."""
            vals = np.array(acc[key], dtype=np.float64)
            par_k = par
            ok_events = (acc["n_settled"] if settle_floor else acc["n_flick_events"])
            floor = MIN_SETTLED_FLICKS if settle_floor else MIN_FLICK_EVENTS
            if ok_events < floor:
                return float("nan"), float("nan"), float("nan")
            def agg(a):
                a = a[np.isfinite(a)]
                if not len(a):
                    return float("nan")
                return float((a >= 1.0).mean()) if ge1 else float(a.mean())
            return (agg(vals), agg(vals[par_k == 0]) if len(par_k) else float("nan"),
                    agg(vals[par_k == 1]) if len(par_k) else float("nan"))

        tpr, tpe, tpo = trait("throughput", settle_floor=True)
        smr, sme, smo = trait("submoves")
        hmr, hme, hmo = trait("homing", settle_floor=True)
        vnr, vne, vno = trait("velnorm")
        per_, pee, peo = trait("primary_endpoint")
        pvr, pve, pvo = trait("primary_peakvel")
        mtr, _, _ = trait("mt_frames", settle_floor=True)
        Dr, _, _ = trait("onset_D")
        smm, smme, smmo = trait_mean("submoves")
        crr, cre, cro = trait_mean("submoves", ge1=True)
        hmm, hmme, hmmo = trait_mean("homing", settle_floor=True)

        tp_raw.append(tpr); sm_raw.append(smr); hm_raw.append(hmr)
        vn_raw.append(vnr); pe_raw.append(per_); pv_raw.append(pvr)
        mt_raw.append(mtr); D_raw.append(Dr)
        sm_mean_raw.append(smm); corr_rate_raw.append(crr); hm_mean_raw.append(hmm)
        # signed (higher=better)
        tp.append(tpr)
        sm.append(-smr if np.isfinite(smr) else float("nan"))
        hm.append(-hmr if np.isfinite(hmr) else float("nan"))
        vn.append(-vnr if np.isfinite(vnr) else float("nan"))
        pe.append(-per_ if np.isfinite(per_) else float("nan"))
        pv.append(pvr)
        sm_mean.append(-smm if np.isfinite(smm) else float("nan"))
        corr_rate.append(-crr if np.isfinite(crr) else float("nan"))
        for key, e, o in (("throughput", tpe, tpo), ("submoves", sme, smo),
                          ("submove_mean", smme, smmo),
                          ("correction_rate", cre, cro),
                          ("homing", hme, hmo), ("homing_mean", hmme, hmmo),
                          ("velnorm", vne, vno),
                          ("primary_endpoint", pee, peo),
                          ("primary_peakvel", pve, pvo)):
            halves[key][0].append(e); halves[key][1].append(o)

        # interception (lead)
        if acc["lead_n_moving"] >= MIN_LEAD_FRAMES:
            lead_score.append(acc["lead_n_better"] / acc["lead_n_moving"])
        else:
            lead_score.append(float("nan"))
        # pursuit per-weapon coh
        for imp, name in IMPULSE_NAME.items():
            pair = acc["weapon_coh"].get(imp)
            if pair is not None and int(pair[0]) >= min_frames:
                wcoh[name][dmi] = int(pair[1]) / int(pair[0])

    pid_arr = np.array(pid)
    n_events = np.array(n_events); n_settled = np.array(n_settled)

    # General pursuit-accuracy index = per-player mean direct-fire coh.
    gacc = np.full(len(pid_arr), np.nan, dtype=np.float64)
    for k, p in enumerate(pid_arr):
        vals = [wcoh[n].get(int(p)) for n in DIRECT_FIRE]
        vals = [v for v in vals if v is not None]
        if vals:
            gacc[k] = float(np.mean(vals))
    coh5 = gacc                               # pursuit axis proxy
    lead = np.array(lead_score)               # interception axis

    # ── TEST 1: which component spreads + split-half reliability ──
    spreads = {
        "throughput_bits_per_s":   _spread(np.array(tp_raw)),
        "submovement_count_median": _spread(np.array(sm_raw)),
        "submovement_count_mean":  _spread(np.array(sm_mean_raw)),
        "correction_rate":         _spread(np.array(corr_rate_raw)),
        "homing_time_median_frames": _spread(np.array(hm_raw)),
        "homing_time_mean_frames": _spread(np.array(hm_mean_raw)),
        "velnorm_primary_err":     _spread(np.array(vn_raw)),
        "primary_endpoint_deg":    _spread(np.array(pe_raw)),
        "primary_peakvel_degf":    _spread(np.array(pv_raw)),
        "flick_MT_frames":         _spread(np.array(mt_raw)),
        "onset_D_deg":             _spread(np.array(D_raw)),
    }

    def _sh(pair):
        e, o = pair
        r, n = _spearman(np.array(e), np.array(o))
        return {"split_half_spearman": round(r, 4), "n": n}
    reliability = {
        "throughput":            _sh(halves["throughput"]),
        "submovement_count_median": _sh(halves["submoves"]),
        "submovement_count_mean": _sh(halves["submove_mean"]),
        "correction_rate":       _sh(halves["correction_rate"]),
        "homing_time_median":    _sh(halves["homing"]),
        "homing_time_mean":      _sh(halves["homing_mean"]),
        "velnorm_primary_err":   _sh(halves["velnorm"]),
        "primary_endpoint":      _sh(halves["primary_endpoint"]),
        "primary_peakvel":       _sh(halves["primary_peakvel"]),
    }

    # ── TEST 2: independence (acquisition-throughput vs pursuit & interception) ──
    tp_arr = np.array(tp); sm_arr = np.array(sm_mean); hm_arr = np.array(hm_mean_raw)
    cr_arr = np.array(corr_rate); vn_arr = np.array(vn)
    indep = {}
    for nm, ax in (("pursuit_coh5", coh5), ("interception_lead", lead)):
        r, n = _spearman(tp_arr, ax)
        indep[f"throughput__{nm}"] = {"spearman": round(r, 4), "n": n}
    # corrective axes vs pursuit/interception (mean homing + correction rate —
    # the discriminating continuous forms, not the 0-floored medians)
    for nm, ax in (("pursuit_coh5", coh5), ("interception_lead", lead)):
        r, n = _spearman(hm_arr, ax)
        indep[f"homing_mean__{nm}"] = {"spearman": round(r, 4), "n": n}
        r, n = _spearman(cr_arr, ax)
        indep[f"correction_rate__{nm}"] = {"spearman": round(r, 4), "n": n}
    # within-acquisition coupling: does throughput ride the primary or the homing?
    pe_arr = np.array(pe); pv_arr = np.array(pv)
    couple = {}
    for a_nm, a in (("primary_endpoint", pe_arr), ("primary_peakvel", pv_arr),
                    ("homing_mean", hm_arr), ("correction_rate", cr_arr),
                    ("velnorm", vn_arr)):
        r, n = _spearman(tp_arr, a)
        couple[f"throughput__{a_nm}"] = {"spearman": round(r, 4), "n": n}
    r, n = _spearman(hm_arr, pe_arr)
    couple["homing_mean__primary_endpoint"] = {"spearman": round(r, 4), "n": n}

    # ── acquisition_dist: the pooled distribution of the ACQUISITION metric ──
    # The decode-fit reference (skill_vector.acquisition_human_band): every settled
    # flick across every demo is ONE sample. Anchored by EXACT min/max — the genuine
    # best/worst human flick, robust to the corpus not being a representative population
    # and to demo != player. The interior percentiles are the observed metric
    # distribution (duplication-biased, named honestly — not a population/skill ranking).
    POOL_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    all_tp = (np.concatenate([np.asarray(acc["throughput"], float) for acc in per.values()])
              if per else np.array([], float))
    all_tp = all_tp[np.isfinite(all_tp) & (all_tp > 0)]
    acquisition_dist = {
        "metric": "throughput_bits_per_s",
        "id_form": "shannon_effective: IDe=log2(D/We+1), We=4.133*SD(settle endpoints)",
        "effective_width_deg": round(WE, 4),
        "unit_note": "pooled over ALL settled flicks, ALL demos (one sample per event); "
                     "higher = better",
        "n_events": int(len(all_tp)),
        "min": round(float(all_tp.min()), 4) if len(all_tp) else None,
        "max": round(float(all_tp.max()), 4) if len(all_tp) else None,
        "percentiles": ({f"p{p}": round(float(np.percentile(all_tp, p)), 4) for p in POOL_PCTS}
                        if len(all_tp) >= 50 else {}),
    }

    # ── WITHIN-DEMO CONSISTENCY (throughput) — a ROUGH sense of a single recording's
    # own throughput spread within the acquisition_dist bounds ──
    # Per-demo aggregates quantify RANGES only (corpus not a representative population;
    # demo != player). consistency-vs-skill is on the SCALE-FREE QCD so it is not a
    # level artifact.
    cons_med, cons_iqr, cons_qcd, cons_logsd = [], [], [], []
    for acc in per.values():
        v = np.asarray(acc["throughput"], float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) < MIN_SETTLED_FLICKS:
            continue
        p25, p50, p75 = np.percentile(v, [25, 50, 75])
        cons_med.append(p50); cons_iqr.append(p75 - p25)
        cons_qcd.append((p75 - p25) / (p75 + p25) if (p75 + p25) > 1e-9 else float("nan"))
        cons_logsd.append(float(np.std(np.log(v), ddof=0)))
    within_consistency: dict[str, Any] = {"n_players": len(cons_med)}
    if len(cons_med) >= 8:
        cm = np.array(cons_med); ci = np.array(cons_iqr); cq = np.array(cons_qcd)
        r_skill, n_skill = _spearman(cm, cq)   # scale-free
        di = np.percentile(ci, [10, 50, 90])
        within_consistency = {
            "n_players": int(len(cm)),
            "dispersion_across_players": {
                "iqr_bits_min": round(float(ci.min()), 4), "iqr_bits_max": round(float(ci.max()), 4),
                "iqr_bits_p10": round(float(di[0]), 4), "iqr_bits_p50": round(float(di[1]), 4),
                "iqr_bits_p90": round(float(di[2]), 4),
                "note": "within-demo throughput spread (p75-p25 of a demo's own settled flicks, "
                        "bits/s); min..max = the human RANGE; p50 = typical. Rough sense, not a "
                        "smooth population distribution"},
            "consistency_is_skill": {
                "spearman_median_vs_qcd": (round(r_skill, 4) if np.isfinite(r_skill) else None),
                "n": n_skill,
                "note": "scale-free: median vs quartile-coeff-of-dispersion. ~0 = tightness "
                        "independent of skill (raw median-vs-IQR would be a level artifact)"},
            "variance_split_ln_throughput": {
                "between_demo_sd": round(float(np.log(cm[cm > 0]).std(ddof=0)), 4)
                                   if (cm > 0).sum() > 1 else None,
                "typical_within_demo_sd": round(float(np.median(cons_logsd)), 4)
                                          if cons_logsd else None,
                "note": "SD of ln(throughput): between = spread of per-demo medians; within = "
                        "median of per-demo own SDs. Context only — demo != player, neither is a "
                        "population parameter; the trustworthy anchor is acquisition_dist min/max"},
        }

    def _summ(x):
        v = np.array(x)[np.isfinite(x)]
        if not len(v):
            return None
        return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 4)}

    # reference spreads (anchors from the program)
    ref = {
        "prior_primary_endpoint_p90_p10": 1.28,    # this script reproduces in spreads
        "prior_primary_peakvel_p90_p10": 1.53,
        "pursuit_direct_fire_p90_p10": [2.75, 4.24],    # skill-curves §1.3 per-weapon
        "pursuit_direct_fire_p99_p10": [3.4, 7.1],
        "interception_lead_p99_p10_program": 3.86,
        "note": "prior primary-endpoint/peakvel ratios are reproduced in spreads "
                "(primary_endpoint_deg / primary_peakvel_degf) under THIS run's "
                "primary-end definition; small deltas vs 1.28/1.53 reflect the "
                "new velocity-minimum primary cut vs the prior window-min endpoint.",
    }

    out = {
        "title": "ACQUISITION two-component (primary sweep + corrective homing) — "
                 "where the per-player skill SPREAD lives",
        "splits": splits, "collect_dir": str(collect_dir),
        "n_players": len(pid),
        "resolution_caveat":
            "20 Hz = 50 ms/frame; a flick+correction is ~2–6 frames. Submovement "
            "segmentation is COARSE — we detect a primary + ~1–2 corrections only "
            "(velocity re-acceleration peaks above SUBMOVE_VEL_FLOOR). Fine "
            "submovement microstructure is NOT claimed. Throughput (ID/MT) and "
            "homing-time are more robust to the coarse sampling than the "
            "submovement count, so the verdict is weighted toward those.",
        "thresholds": {
            "FLICK_ONSET_DEG": FLICK_ONSET_DEG, "SETTLE_DEG_W": SETTLE_DEG,
            "FLICK_WIN_FRAMES": FLICK_WIN_FRAMES, "FPS": FPS,
            "SUBMOVE_VEL_FLOOR_degf": SUBMOVE_VEL_FLOOR,
            "MIN_FLICK_EVENTS": MIN_FLICK_EVENTS,
            "MIN_SETTLED_FLICKS": MIN_SETTLED_FLICKS,
            "MIN_LEAD_FRAMES": MIN_LEAD_FRAMES,
            "min_frames_per_weapon": min_frames,
        },
        "metric_defs": {
            "throughput": "ISO 9241-9 Shannon EFFECTIVE throughput TP = IDe/MT bits/s; "
                          "IDe=log2(D/We+1), D=onset LOS angle deg, We=4.133*SD(settle "
                          "endpoints) (effective width, not the nominal cone); MT=onset→"
                          "settle seconds (frames/20). Settled flicks only.",
            "submovement_count": "# velocity re-acceleration peaks (>floor) in the "
                                 "corrective phase (after primary-end). Median.",
            "homing_time_frames": "frames from primary-end to cone settle. Settled "
                                  "flicks only. Median.",
            "velnorm_primary_err": "primary endpoint error deg / primary peak vel "
                                   "(deg/frame). Lower=better. Median.",
            "primary_endpoint_deg": "LOS angle deg at primary-end (first velocity "
                                    "minimum after peak closure). Continuity w/ prior.",
            "primary_peakvel_degf": "max per-frame closure in primary. Continuity w/ prior.",
        },
        "event_counts": {
            "n_players": len(pid),
            "median_flick_events_per_player": int(np.median(n_events)) if len(n_events) else 0,
            "median_settled_flicks_per_player": int(np.median(n_settled)) if len(n_settled) else 0,
            "total_flick_events": int(n_events.sum()),
            "total_settled": int(n_settled.sum()),
        },
        "metric_distributions": {
            "throughput_bits_per_s": _summ(tp_raw),
            "submovement_count_median": _summ(sm_raw),
            "submovement_count_mean": _summ(sm_mean_raw),
            "correction_rate": _summ(corr_rate_raw),
            "homing_time_median_frames": _summ(hm_raw),
            "homing_time_mean_frames": _summ(hm_mean_raw),
            "velnorm_primary_err": _summ(vn_raw),
            "primary_endpoint_deg": _summ(pe_raw),
            "primary_peakvel_degf": _summ(pv_raw),
            "flick_MT_frames": _summ(mt_raw),
            "onset_D_deg": _summ(D_raw),
            "pursuit_coh5_index": _summ(coh5),
            "interception_lead": _summ(lead),
        },
        "reference_spreads": ref,
        "acquisition_dist": acquisition_dist,
        "within_player_consistency": within_consistency,
        "TEST_1_which_component_spreads": {
            "per_component_spread": spreads,
            "split_half_reliability": reliability,
        },
        "TEST_2_independence": {
            "throughput_vs_other_axes": indep,
            "within_acquisition_coupling": couple,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    # ── console ──
    print("\n=== EVENT COUNTS ===")
    print(f"  {out['event_counts']}")
    print("\n=== TEST 1: WHICH COMPONENT SPREADS (p90/p10 ratio) ===")
    print(f"  {'metric':26s} {'n':>5s} {'p10':>8s} {'p50':>8s} {'p90':>8s} "
          f"{'p90/p10':>8s} {'p90/p50':>8s} {'QCD':>6s}")
    for k, v in spreads.items():
        if "ratio_p90_p10" not in v:
            print(f"  {k:26s} {v.get('n','-'):>5} (too few)"); continue
        rp = v['ratio_p90_p10']
        rps = f"{rp:>8.3f}" if np.isfinite(rp) else f"{'inf':>8s}"
        print(f"  {k:26s} {v['n']:>5d} {v['p10']:>8.3f} {v['p50']:>8.3f} "
              f"{v['p90']:>8.3f} {rps} {v['ratio_p90_p50']:>8.3f} {v['qcd_iqr']:>6.3f}")
    print("  -- reference: prior primary-endpoint 1.28× / peakvel 1.53× ; "
          "pursuit 2.75–4.24× (p99/p10 3.4–7.1×) ; interception ~3.86×")
    print("\n  SPLIT-HALF reliability (even/odd event parity):")
    for k, v in reliability.items():
        print(f"    {k:20s} r={v['split_half_spearman']:+.3f} n={v['n']}")
    print("\n=== TEST 2: INDEPENDENCE (acquisition throughput vs other axes) ===")
    for k, v in indep.items():
        print(f"    {k:34s} r={v['spearman']:+.3f} n={v['n']}")
    print("  within-acquisition coupling:")
    for k, v in couple.items():
        print(f"    {k:34s} r={v['spearman']:+.3f} n={v['n']}")
    print("\n=== acquisition_dist (all settled flicks, all demos — decode-fit reference) ===")
    pp = acquisition_dist["percentiles"]
    if pp:
        print(f"  We={acquisition_dist['effective_width_deg']}deg  n_events={acquisition_dist['n_events']:,}  "
              f"min={acquisition_dist['min']} max={acquisition_dist['max']}  " +
              "  ".join(f"{k}={pp[k]}" for k in ("p10", "p50", "p90")))
    wc = within_consistency
    if wc.get("n_players", 0) >= 8:
        di = wc["dispersion_across_players"]; sk = wc["consistency_is_skill"]
        print(f"  within-demo IQR (bits/s): min={di['iqr_bits_min']} p50={di['iqr_bits_p50']} "
              f"max={di['iqr_bits_max']}  | consistency-is-skill (scale-free) r(med,qcd)="
              f"{sk['spearman_median_vs_qcd']} (n={sk['n']})")
    print(f"\nWritten -> {out_path}")


# ── closed-loop (model-side) acquisition from emitted eval streams ───────────
#
# The aim-grid closed-loop eval (qnn.eval.run, eval_log_acq_streams) emits the
# SAME per-frame entity obs the human collect cache stores, scenario-tagged, to
# acq_streams_<mode>.npz. These helpers run THIS module's flick kernel on those
# emitted streams so a single grid campaign yields a per-cell ACQUISITION
# throughput alongside the per-cell intercept ruler — mirroring
# aim_grid_closedloop._score_cell's intercept_hbw. No physics is duplicated: the
# geometry and flick segmentation are _episode_metrics / _decompose_flick above.
#
# One difference vs the human path: the closed-loop model has NO labeler target
# pointer, so the target_probs engagement gate is unavailable. We pass ep_engaged
# all-True and let _episode_metrics' internal has_los gate (actor token present,
# recency 0) define engagement — the obs-derivable, model-side analog. In the
# bot-pin 1v1 instrument the two are near-identical (the model is always facing
# its single pinned opponent), so this is immaterial to the throughput measure.

def episode_metrics_from_streams(
    cnt: np.ndarray, rel: np.ndarray, vel: np.ndarray,
    typ: np.ndarray, rec: np.ndarray, weapon: np.ndarray,
) -> dict[str, Any]:
    """Run the shared _episode_metrics kernel on ONE emitted closed-loop episode's
    entity streams (collect-cache layout: (T,) count + concatenated entities)."""
    ep_engaged = np.ones(len(cnt), dtype=bool)
    return _episode_metrics(
        ep_cnt=np.asarray(cnt, dtype=np.int32),
        ep_rel=np.asarray(rel),
        ep_vel=np.asarray(vel),
        ep_typ=np.asarray(typ),
        ep_rec=np.asarray(rec),
        ep_weapon=np.asarray(weapon),
        ep_engaged=ep_engaged,
    )


def cell_acquisition_throughput(npz_path: Path, scenario_id: str,
                                effective_width_deg: float) -> dict | None:
    """Model ACQUISITION (Shannon effective throughput, bits/s; higher = better) for
    ONE closed-loop grid cell, from its emitted acq_streams npz. ``effective_width_deg``
    is the HUMAN We (acquisition_dist.effective_width_deg) — the model rides the SAME
    instrument as the human reference, so the two are comparable. Gathers the cell's
    episodes, runs the shared flick kernel, and returns the per-cell median settled-
    flick throughput + counts. None if the cell emitted no episodes; ``throughput`` is
    None (with counts) below the settled-flick floor (MIN_SETTLED_FLICKS)."""
    z = np.load(npz_path, allow_pickle=False)
    keys = [str(k) for k in z["acq_episode_keys"]]
    scen = [str(s) for s in z["acq_episode_scenarios"]]
    acc = _new_acc()
    n_eps = 0
    for k, s in zip(keys, scen):
        if s != scenario_id:
            continue
        n_eps += 1
        _merge(acc, episode_metrics_from_streams(
            z[f"acq_cnt_{k}"], z[f"acq_rel_{k}"], z[f"acq_vel_{k}"],
            z[f"acq_typ_{k}"], z[f"acq_rec_{k}"], z[f"acq_weapon_{k}"]))
    if not n_eps:
        return None
    tps = [shannon_throughput(D, mt, effective_width_deg)
           for D, mt in zip(acc["onset_D"], acc["mt_frames"])]
    tp = _nanmedian(tps) if acc["n_settled"] >= MIN_SETTLED_FLICKS else None
    return {
        "throughput": round(tp, 4) if tp is not None and np.isfinite(tp) else None,
        "n_flick_events": int(acc["n_flick_events"]),
        "n_settled": int(acc["n_settled"]),
        "n_episodes": int(n_eps),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="both", choices=["val", "train", "both"])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--min-frames", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <collect>/human_baseline/_acq_submovement.json)")
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_acq_submovement.json"
    splits = ["train", "val"] if args.split == "both" else [args.split]
    print(f"Collect: {args.collect_dir}  splits={splits}  workers={args.workers}  "
          f"min_frames={args.min_frames}")
    run(args.collect_dir, splits, out, args.workers, args.min_frames)


if __name__ == "__main__":
    main()
