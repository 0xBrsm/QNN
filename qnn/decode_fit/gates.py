"""Gates — confirmation, secant correction, style gate, attack trim.

Plan §P3 (decode-fit-v2.md): stage 6 splits into two honest halves.

* CONFIRMATION (gated, corrective): one batched eval on the SAME botpin
  instrument at the planned operating point. PASS per weapon = the measured
  percentile CI overlaps the promised percentile ±5 (decision 3; refused
  plans promised the FRONTIER, so they confirm against ``frontier_pct``).
  On a miss: ONE damped secant correction off the fitted curve's local
  slope, then the caller re-confirms once.
* DEPLOYMENT REPORT (world-gated, style-flagged): free-play arena eval.
  Gated arms = world-results invariants ONLY (op-attack ruler,
  threat-reactivity hazard, weapon-switch rate). Fitted-vs-native style
  spend on band-v5 anchored ratios is FLAGGED (training-target register,
  stamped into promoted provenance), never gating — skill placement is
  never capped for style (Brian 2026-07-16). Band MEMBERSHIP never gates
  either (research/human-band.md axioms: it is the research tracker).
  Per-weapon free-play hbw is a REPORT CARD, never gated (decision 4 /
  skill-curves §1.7b) — the fit→free-play domain gap is tracked as a
  ``transfer_coeff`` per weapon instead of manufacturing FAILs.

Everything here is PURE decision logic: no eval is ever launched from this
module. ``attack_trim`` takes a ``launch_eval`` callback injected by the CLI
(instruments.py owns launching); the other gates score artifacts they are
handed. The world-results / band / intercept readers are ported from the v1
pipeline (``qnn.eval.decode_fit_pipeline``) with their line references cited
inline — v1 is deleted in the same change that lands this package (Phase 3).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qnn.decode_fit.context import (INSTRUMENT_WEAPONS, MODELNAME_TO_ABBR,
                                    TRANSFER_ALIAS, WEAPON_IMPULSE, read_json)
from qnn.decode_fit.events import EventTable
from qnn.decode_fit.human_refs import hbw_to_pct
from qnn.decode_fit.response import (AlphaResponse, GainResponse, WeaponPlan)

# the 4 weapons the botpin instrument measures directly (SSG/SNG confirm
# transitively through their alias — same response, own ladder).
INSTRUMENT_ABBRS = tuple(MODELNAME_TO_ABBR[w] for w in INSTRUMENT_WEAPONS)

# Below this effective discharge mass a confirmation card is ``low_n``:
# neither pass nor fail, counted separately and flagged loud (spec: the
# gate must never adjudicate a weapon it barely observed).
CONFIRM_MIN_DISCHARGES = 50

# ── attack-trim constants (ported verbatim from v1 decode_fit_pipeline.py
# ~995-1011; semantics unchanged — see that file's comments for derivations) ──
TRIM_MAX_ITERS = 6                 # 3 knobs (attack, stick, threat-break)
TRIM_DAMPING = 0.6                 # step = damping × log-ratio (anti-oscillation)
TRIM_TOL = 0.15
TRIM_ATT_STEP_CLAMP = (-0.75, 0.75)
# Threat-break hazard trim: additive step toward the human reactivity hazard;
# Δratio→λ slope ≈ 0.35. Tolerance is on the LIFT (ratio − 1), NOT the ratio
# (±15% of a 1.143 ratio contains "no reactivity at all" — the v1 it0 bug).
TRIM_REACT_STEP_SLOPE = 0.35
TRIM_REACT_STEP_CLAMP = (-0.03, 0.03)
TRIM_REACT_HAZARD_MAX = 0.15
TRIM_REACT_LIFT_TOL = 0.25
TRIM_STICK_STEP_CLAMP = (-0.75, 1.0)
# Per-weapon attack trim (a25rc3c redesign): each weapon's bias_vec entry
# steps toward ITS human rate; below the tick floor a weapon's per-eval rate
# is too noisy to step on (it stays on the aggregate ruler only).
TRIM_VEC_STEP_CLAMP = (-0.75, 0.75)
TRIM_PERWEAPON_MIN_TICKS = 2000

# Pinned human threat-reactivity reference (own-fire-gated, skill-stable
# all-humans pin) — ported from v1 decode_fit_pipeline.py:1256.
HUMAN_REACTIVITY_HAZARD = 1.143

# Fitted-vs-native style budget (band v5, Axiom 3): the fitted operating
# point may not lift any channel's anchored ratio more than this above the
# native-substrate reference — the decode must not make the model less human
# than the head it decorates. Provisional at a quarter of the scorer's
# verdict bar (ANCHOR_RATIO_MAX = 0.2); finalize once several v5-scored fits
# exist.
STYLE_REGRESSION_MARGIN = 0.05

_IMPULSE_NAME = {1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
                 6: "GL", 7: "RL", 8: "LG"}


def _log(msg: str) -> None:
    print(f"[decode-fit] {msg}", flush=True)


# ══ 1. confirmation gate (decision 3) ═════════════════════════════════════════

def _median_ci(hbw: np.ndarray, clusters: np.ndarray, n_boot: int,
               rng: np.random.Generator) -> tuple[float, tuple[float, float]]:
    """Median + CLUSTER-bootstrap CI: resample unique ``cluster`` ids with
    replacement and recompute the median (episodes are the independence unit —
    consecutive shots at one opponent trajectory are correlated)."""
    med = float(np.median(hbw))
    uniq = np.unique(clusters)
    if n_boot <= 0 or len(uniq) < 2:
        return med, (med, med)
    idx_by_c = {c: np.flatnonzero(clusters == c) for c in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_c[c] for c in pick])
        boots[b] = np.median(hbw[rows])
    return med, (float(np.percentile(boots, 2.5)),
                 float(np.percentile(boots, 97.5)))


def _pct_interval(hbw_ci: tuple[float, float],
                  ladder: dict[float, float]) -> tuple[float, float]:
    """hbw CI → SKILL-percentile CI on one weapon's ladder (hbw_to_pct is
    INVERTED — lower hbw = higher pct — so the interval endpoints swap)."""
    a = hbw_to_pct(hbw_ci[0], ladder)
    b = hbw_to_pct(hbw_ci[1], ladder)
    return (min(a, b), max(a, b))


def confirmation_gate(table: EventTable, plans: dict[str, WeaponPlan],
                      ladder: dict[str, dict[float, float]], *,
                      tol_pct: float = 5.0, n_boot: int = 2000,
                      seed: int = 0) -> dict[str, Any]:
    """Score the confirmation waves (plan §P3, decision 3).

    Per instrument weapon: measured median hbw + cluster-bootstrap CI, mapped
    through the weapon's OWN human ladder to a percentile CI. PASS = that CI
    overlaps [center − tol_pct, center + tol_pct] where center is the plan's
    ``target_pct`` — or ``frontier_pct`` for refused plans (the plan promised
    the frontier, so the frontier is what must be confirmed). Weapons under
    ``CONFIRM_MIN_DISCHARGES`` effective discharges are ``low_n`` (neither
    pass nor fail; listed separately, flagged loud). SSG/SNG confirm
    transitively via their alias and are reported with their OWN ladder
    placement of the alias's measured hbw. Also reports ``fit_calibration``
    per weapon: did the fit predict itself (measured median inside
    ``plan.pred_hbw_ci``)?"""
    rng = np.random.default_rng(seed)
    cards: dict[str, dict[str, Any]] = {}
    ladders_out: dict[str, dict[float, float]] = {}
    measured: dict[str, dict[str, Any]] = {}      # instrument abbr → readings
    passed, failed, low_n, transitive = [], [], [], []

    # ── measured (instrument) weapons ─────────────────────────────────────
    for abbr in INSTRUMENT_ABBRS:
        plan = plans.get(abbr)
        if plan is None:
            continue
        lad = ladder.get(abbr)
        if not lad:
            raise ValueError(f"confirmation_gate: no human ladder for {abbr}")
        ladders_out[abbr] = dict(lad)
        rows = table.where(weapon=abbr) if len(table) else table
        n_eff = float(rows["weight"].sum()) if len(rows) else 0.0
        center = float(plan.frontier_pct if plan.refused else plan.target_pct)
        card: dict[str, Any] = {
            "target_pct": float(plan.target_pct),
            "refused": bool(plan.refused),
            "center_pct": center,
            "tol_band_pct": [round(center - tol_pct, 2), round(center + tol_pct, 2)],
            "n_events": int(n_eff),
        }
        if n_eff < CONFIRM_MIN_DISCHARGES:
            card["verdict"] = "low_n"
            card["note"] = (f"only {int(n_eff)} effective discharges "
                            f"(< {CONFIRM_MIN_DISCHARGES}) — neither pass nor "
                            "fail; extend the confirmation wave")
            cards[abbr] = card
            low_n.append(abbr)
            continue
        med, hbw_ci = _median_ci(rows["hbw"], rows["cluster"], n_boot, rng)
        pct = hbw_to_pct(med, lad)
        pct_ci = _pct_interval(hbw_ci, lad)
        overlap = (pct_ci[0] <= center + tol_pct) and (pct_ci[1] >= center - tol_pct)
        tms_vals = rows["tms"][np.isfinite(rows["tms"])]
        tms = float(np.median(tms_vals)) if len(tms_vals) else None
        card.update({
            "verdict": "pass" if overlap else "fail",
            "n_clusters": int(len(np.unique(rows["cluster"]))),
            "measured_hbw_median": round(med, 4),
            "measured_hbw_ci": [round(hbw_ci[0], 4), round(hbw_ci[1], 4)],
            "measured_pct": round(pct, 1),
            "measured_pct_ci": [round(pct_ci[0], 1), round(pct_ci[1], 1)],
            "tms": tms,
            # did the fit predict itself? (measured median vs the plan's
            # prediction CI — calibration information, not the verdict)
            "fit_calibration": {
                "pred_hbw": plan.pred_hbw,
                "pred_hbw_ci": list(plan.pred_hbw_ci),
                "measured_in_pred_ci": bool(
                    plan.pred_hbw_ci[0] - 1e-9 <= med <= plan.pred_hbw_ci[1] + 1e-9),
            },
        })
        cards[abbr] = card
        measured[abbr] = {"median": med, "ci": hbw_ci, "tms": tms}
        (passed if overlap else failed).append(abbr)

    # ── transitive weapons (SSG/SNG ride SG/NG; own ladder placement) ─────
    for abbr, plan in plans.items():
        if abbr in cards:
            continue
        alias = plan.alias_of or TRANSFER_ALIAS.get(abbr)
        lad = ladder.get(abbr)
        center = float(plan.frontier_pct if plan.refused else plan.target_pct)
        card = {
            "verdict": "transitive",
            "alias_of": alias,
            "target_pct": float(plan.target_pct),
            "refused": bool(plan.refused),
            "center_pct": center,
            "tol_band_pct": [round(center - tol_pct, 2), round(center + tol_pct, 2)],
        }
        m = measured.get(alias or "")
        if m is None:
            card["note"] = (f"alias {alias!r} unmeasured "
                            f"({cards.get(alias, {}).get('verdict', 'absent')}) — "
                            "no transitive placement")
        elif lad:
            ladders_out[abbr] = dict(lad)
            pct = hbw_to_pct(m["median"], lad)
            pct_ci = _pct_interval(m["ci"], lad)
            card.update({
                "measured_hbw_median": round(m["median"], 4),
                "measured_hbw_ci": [round(m["ci"][0], 4), round(m["ci"][1], 4)],
                "measured_pct": round(pct, 1),
                "measured_pct_ci": [round(pct_ci[0], 1), round(pct_ci[1], 1)],
                # informational only — a transitive weapon never fails the gate
                "overlaps_target": bool((pct_ci[0] <= center + tol_pct)
                                        and (pct_ci[1] >= center - tol_pct)),
            })
        cards[abbr] = card
        transitive.append(abbr)

    status = "PASS" if not failed else "FAIL"
    out: dict[str, Any] = {
        "status": status,
        "criterion": ("percentile CI overlap at ±{:g} (decision 3); refused "
                      "plans confirm against frontier_pct — the plan promised "
                      "the frontier".format(tol_pct)),
        "tol_pct": float(tol_pct),
        "n_boot": int(n_boot),
        "weapons": cards,
        "passed": passed, "failed": failed, "low_n": low_n,
        "transitive": transitive,
        # ladders travel with the gate so secant_correction can re-place
        # corrected predictions without re-reading the human baseline.
        "ladders": ladders_out,
    }
    if low_n:
        out["note"] = (f"LOW-N (loud): {low_n} under {CONFIRM_MIN_DISCHARGES} "
                       "effective discharges — unadjudicated; the confirmation "
                       "wave must be extended before these placements are trusted")
    return out


# ══ 2. one damped secant correction (plan §P3) ════════════════════════════════

def secant_correction(plans: dict[str, WeaponPlan], gate: dict[str, Any],
                      gain_fits: dict[str, GainResponse],
                      alpha_fits: dict[str, AlphaResponse | None], *,
                      tremor_fit=None, support_elite: dict[str, float] | None = None,
                      damping: float = 0.5) -> dict[str, WeaponPlan] | None:
    """One damped secant step for every FAILed confirmation weapon, off the
    fitted curve's local slope: Δ = target_hbw_effective − measured;
    new_lever = lever + damping·Δ/grad (grad ≤ 0, so a too-loose measurement
    — Δ < 0 — RAISES the lever). Gain is clipped to [0, knee(0.95)·1.5];
    super-band plans correct α instead via ``alpha_grad`` (same clip form on
    the α ray).

    TOO-GOOD + UNTRUSTED-FIT routing (the a25rc3c RL failure class): when the
    weapon measured BETTER than the target (Δ > 0) and the measurement fell
    outside the fit's prediction CI (the local curve is wrong there — a
    degenerate zero-width CI counts), walking the gain/α lever off that curve
    is extrapolating a model reality just falsified. Instead the correction
    routes through the TREMOR down band, anchored on the MEASURED value:
    ``mag = ln(target/measured) / slope`` — performance must come back inside
    the human band, and tremor is the designed degradation lever (skill-curves
    §15.4). Returns the FULL plan dict with corrected entries, or None when
    nothing failed. ONE correction only — the caller re-confirms once."""
    cards = gate.get("weapons") or {}
    failed = [w for w, c in cards.items() if c.get("verdict") == "fail"]
    if not failed:
        return None
    out = dict(plans)
    for abbr in failed:
        plan = plans[abbr]
        card = cards[abbr]
        measured = float(card["measured_hbw_median"])
        tms = card.get("tms")
        # refused plans promised the frontier — correct toward it, never the wish
        eff_target = max(plan.target_hbw, plan.frontier_hbw)
        delta = eff_target - measured
        src = plan.alias_of or abbr
        lad_raw = (gate.get("ladders") or {}).get(abbr) or {}
        lad = {float(k): float(v) for k, v in lad_raw.items()}

        fit_cal = card.get("fit_calibration") or {}
        _ci = fit_cal.get("pred_hbw_ci") or plan.pred_hbw_ci or (None, None)
        _degenerate = (_ci[0] is not None and _ci[1] is not None
                       and abs(float(_ci[1]) - float(_ci[0])) < 1e-9)
        fit_untrusted = (fit_cal.get("measured_in_pred_ci") is False
                         or _degenerate)

        if plan.band == "frontier-measured":
            # a measured-frontier promise came from ≤2-pin cell evidence; the
            # 4-pin confirmation is the better instrument in BOTH directions —
            # re-promise at its measurement, levers untouched (walking a curve
            # the cells already refuted is never an option here)
            pct = round(hbw_to_pct(measured, lad), 1) if lad \
                else plan.achieved_pct
            out[abbr] = dataclasses.replace(
                plan, pred_hbw=round(measured, 4),
                pred_hbw_ci=(round(measured, 4), round(measured, 4)),
                frontier_hbw=round(measured, 4), frontier_pct=pct,
                achieved_pct=pct,
                refused=bool(measured > plan.target_hbw),
                notes=(plan.notes + f"; frontier-measured re-promise: 4-pin "
                       f"confirmation measured {measured:.3f} vs promised "
                       f"{plan.pred_hbw:.3f} — the confirm instrument wins "
                       "both directions").lstrip("; "))
            continue
        if delta > 0 and fit_untrusted:
            # Measured TIGHTER than promised with the local fit falsified.
            # Two distinct cases (the a25rc3c SG mis-route): beating a REFUSED
            # plan's frontier while still short of the wish is GOOD NEWS about
            # a wrong frontier — re-promise at the measurement, never degrade
            # real capability toward a falsified pessimistic prediction. Only
            # overshooting the WISH (the human-band placement itself) warrants
            # the tremor DOWN band.
            if plan.refused and measured > plan.target_hbw:
                pct = round(hbw_to_pct(measured, lad), 1) if lad \
                    else plan.frontier_pct
                out[abbr] = dataclasses.replace(
                    plan, band="frontier-corrected",
                    pred_hbw=round(measured, 4),
                    pred_hbw_ci=(round(measured, 4), round(measured, 4)),
                    frontier_hbw=round(measured, 4), frontier_pct=pct,
                    achieved_pct=pct,
                    notes=(plan.notes + f"; frontier correction: measured "
                           f"{measured:.3f} beats the fitted frontier "
                           f"{plan.frontier_hbw:.3f} (fit falsified) while "
                           f"short of the {plan.target_hbw:.3f} wish — "
                           "re-promising at the measurement; levers untouched"
                           ).lstrip("; "))
                continue
            sup = (support_elite or {}).get(plan.alias_of or abbr)
            if sup is not None and measured >= sup - 1e-9:
                # TWO-BOUNDARY RULE (§16.3): the measurement beat the wish but
                # sits INSIDE observed human support (looser than the raw
                # un-shrunk elite) — degradation may not fire. The plan becomes
                # a REFUSAL-TO-DEGRADE: it re-promises at the measurement as
                # its frontier (confirmation then centers there), flagged for
                # the report as beyond-defensible-band / within-support.
                pct = round(hbw_to_pct(measured, lad), 1) if lad \
                    else plan.achieved_pct
                out[abbr] = dataclasses.replace(
                    plan, refused=True, band="support-refused",
                    pred_hbw=round(measured, 4),
                    pred_hbw_ci=(round(measured, 4), round(measured, 4)),
                    frontier_hbw=round(measured, 4), frontier_pct=pct,
                    achieved_pct=pct,
                    notes=(plan.notes + f"; refusal-to-degrade: measured "
                           f"{measured:.3f} beats the {plan.target_hbw:.3f} "
                           f"wish but is within observed human support "
                           f"(raw elite {sup:.3f}) — placement rides the "
                           "measurement, FLAGGED beyond the defensible "
                           "sustained band, no degradation"
                           ).lstrip("; "))
                continue
            back_target = plan.target_hbw if measured < plan.target_hbw \
                else eff_target
            # LEVER-CHANNEL correction first (Brian 2026-07-17): a hot gain/α
            # is corrected on its OWN ray, never masked with tremor — stacking
            # a degradation lever on a mis-set skill lever leaves two opposing
            # knobs at the operating point. The falsified fit is not walked:
            # the step rides the MEASURED ray — (0, ray base) → (lever,
            # measured) are both world evidence (the base is pinned by the
            # lever-0 sweep cells), same conditioning guard as the empirical
            # correction.
            lever = "alpha" if plan.band == "super" else "gain"
            lev_val = float(getattr(plan, lever))
            ray_base = lev_cap = None
            if lever == "gain":
                _gf = gain_fits.get(src)
                if _gf is not None:
                    ray_base = float(_gf.native_at(tms))
                    lev_cap = _gf.knee(0.95)[0] * 1.5
            else:
                _af = alpha_fits.get(src)
                if _af is not None:
                    ray_base = float(_af.start)
                    lev_cap = _af.k * math.log(1.0 / (1.0 - 0.95)) * 1.5
            if lev_val > 1e-9 and ray_base is not None and ray_base > 0:
                ray_dlog = math.log(measured / ray_base)
                s_emp = ray_dlog / lev_val
                if s_emp < -1e-9 and abs(ray_dlog) >= EMPIRICAL_MIN_DLOG:
                    new_lev = float(np.clip(
                        lev_val + damping * math.log(back_target
                                                     / max(measured, 1e-9))
                        / s_emp,
                        0.0, lev_cap))
                    pred = float(measured * math.exp(s_emp * (new_lev - lev_val)))
                    out[abbr] = dataclasses.replace(
                        plan, **{lever: round(new_lev, 4)},
                        pred_hbw=round(pred, 4),
                        pred_hbw_ci=(round(pred, 4), round(pred, 4)),
                        achieved_pct=(round(hbw_to_pct(pred, lad), 1) if lad
                                      else plan.achieved_pct),
                        notes=(plan.notes + f"; empirical lever correction: "
                               f"{lever} {lev_val}→{round(new_lev, 4)} on the "
                               f"measured ray (base {ray_base:.3f} → "
                               f"{measured:.3f} at {lever} {lev_val}, realized "
                               f"slope {s_emp:.3f}) toward {back_target:.3f} — "
                               "a hot lever corrects in its own channel, "
                               "never via tremor").lstrip("; "))
                    continue
            if tremor_fit is not None:
                # BEYOND LEVER REACH ONLY: the lever is already at 0 or its
                # measured ray is flat/ill-conditioned (the lever cannot
                # express the placement) while the model still beats the wish
                # from outside human support — tremor is the designed
                # degradation channel for exactly this class.
                slope = float(getattr(tremor_fit, "slope", 0.0))
                if slope > 1e-9:
                    mag = float(np.clip(
                        damping * math.log(back_target / max(measured, 1e-9))
                        / slope,
                        0.0, getattr(tremor_fit, "mag_max", 1.0)))
                    pred = float(measured * math.exp(slope * mag))
                    out[abbr] = dataclasses.replace(
                        plan, tremor=round(mag, 4), band="down-corrected",
                        pred_hbw=round(pred, 4),
                        pred_hbw_ci=(round(pred, 4), round(pred, 4)),
                        achieved_pct=(round(hbw_to_pct(pred, lad), 1) if lad
                                      else plan.achieved_pct),
                        notes=(plan.notes + f"; tremor correction: measured "
                               f"{measured:.3f} < target {back_target:.3f} "
                               f"with the local {plan.band}-band fit falsified "
                               "(measured outside pred CI) — degrading via the "
                               "DOWN band from the measured baseline, "
                               f"mag={round(mag, 4)}").lstrip("; "))
                    continue

        if plan.band == "super":
            afit = alpha_fits.get(src)
            if afit is None:
                out[abbr] = dataclasses.replace(
                    plan, notes=(plan.notes + "; secant correction skipped: "
                                 "super-band plan with no α fit").lstrip("; "))
                continue
            # (empirical follow-up lives in empirical_correction — this path
            # remains the curve-based first correction)
            grad = afit.alpha_grad(plan.alpha)
            if grad >= -1e-12:
                out[abbr] = dataclasses.replace(
                    plan, notes=(plan.notes + "; secant correction skipped: "
                                 "flat α gradient at the operating point").lstrip("; "))
                continue
            cap = afit.k * math.log(1.0 / (1.0 - 0.95)) * 1.5
            new_alpha = float(np.clip(plan.alpha + damping * delta / grad, 0.0, cap))
            pred = afit.predict_hbw(new_alpha)
            ci = (pred, pred)
            new = dataclasses.replace(
                plan, alpha=round(new_alpha, 4), pred_hbw=round(pred, 4),
                pred_hbw_ci=(round(ci[0], 4), round(ci[1], 4)),
                achieved_pct=(round(hbw_to_pct(pred, lad), 1) if lad
                              else plan.achieved_pct),
                notes=(plan.notes + f"; secant correction: alpha {plan.alpha}"
                       f"→{round(new_alpha, 4)} (measured {measured:.3f} hbw vs "
                       f"target {eff_target:.3f})").lstrip("; "))
        else:
            fit = gain_fits[src]
            grad = fit.gain_grad(plan.gain, tms)      # d hbw / d gain, ≤ 0
            if grad >= -1e-12:
                out[abbr] = dataclasses.replace(
                    plan, notes=(plan.notes + "; secant correction skipped: "
                                 "flat gain gradient at the operating point").lstrip("; "))
                continue
            # Δ/grad: Δ<0 (too loose) over grad<0 → positive step (more gain);
            # Δ>0 (too tight) → negative step. Moves TOWARD the target.
            cap = fit.knee(0.95)[0] * 1.5
            new_gain = float(np.clip(plan.gain + damping * delta / grad, 0.0, cap))
            pred = fit.predict_hbw(new_gain, tms)
            ci = fit.predict_ci(new_gain, tms)
            new = dataclasses.replace(
                plan, gain=round(new_gain, 4), pred_hbw=round(pred, 4),
                pred_hbw_ci=(round(ci[0], 4), round(ci[1], 4)),
                achieved_pct=(round(hbw_to_pct(pred, lad), 1) if lad
                              else plan.achieved_pct),
                notes=(plan.notes + f"; secant correction: gain {plan.gain}"
                       f"→{round(new_gain, 4)} (measured {measured:.3f} hbw vs "
                       f"target {eff_target:.3f})").lstrip("; "))
        out[abbr] = new
    return out


EMPIRICAL_MIN_DLOG = 0.05    # two points closer than ~5% in log-hbw carry no slope


def empirical_correction(prev_plans: dict[str, WeaponPlan],
                         prev_gate: dict[str, Any],
                         plans: dict[str, WeaponPlan],
                         gate: dict[str, Any], *,
                         tremor_fit=None) -> dict[str, WeaponPlan] | None:
    """A TRUE secant step from two MEASURED points — no fitted-curve trust.

    After the one curve-based correction, a weapon can fail re-confirmation
    because the fitted response slope mispredicts the correction's effect
    (the a25rc3c RL case: fitted tremor slope 3× the realized one, so the
    half-step landed short and the re-measure failed against the wish).
    At that point the ray holds two measurements: (lever₁, m₁) from the
    first confirmation and (lever₂, m₂) from the second. The realized slope
    ``s = ln(m₂/m₁)/(lever₂−lever₁)`` supports one evidence-based full step
    ``lever₃ = lever₂ + ln(target/m₂)/s`` — extrapolating the WORLD, not a
    falsified fit. Applies to whichever single lever the first correction
    moved (tremor, α, or gain).

    CONDITIONING (the a25rc3c RL overshoot, tremor 0.176 → 5.72 hbw): a
    slope from two points closer than ``EMPIRICAL_MIN_DLOG`` in log-hbw is
    noise, not evidence — RL's pair (3.120, 3.244) implied slope 1.32 while
    the fitted 3.95 was correct at range. An ill-conditioned TREMOR pair
    falls back to the FITTED slope anchored at the latest measurement with
    a FULL (undamped) step; other levers without conditioning refuse.
    Returns the corrected FULL plan dict, or None when any failed weapon
    lacks a usable step (no lever moved, ill-conditioned with no fallback,
    or slope with the wrong sign) — a third attempt would deterministically
    re-fail, so the caller must stop."""
    cards = gate.get("weapons") or {}
    prev_cards = prev_gate.get("weapons") or {}
    failed = [w for w, c in cards.items() if c.get("verdict") == "fail"]
    if not failed:
        return None
    out = dict(plans)
    for abbr in failed:
        p1, p2 = prev_plans.get(abbr), plans[abbr]
        c1, c2 = prev_cards.get(abbr) or {}, cards[abbr]
        m1, m2 = c1.get("measured_hbw_median"), c2.get("measured_hbw_median")
        if p1 is None or m1 is None or m2 is None:
            return None
        moved = [(lv, getattr(p1, lv), getattr(p2, lv))
                 for lv in ("tremor", "alpha", "gain")
                 if abs(getattr(p2, lv) - getattr(p1, lv)) > 1e-9]
        if len(moved) != 1:                  # none or ambiguous — no clean ray
            return None
        lever, l1, l2 = moved[0]
        m1, m2 = float(m1), float(m2)
        dlog = math.log(m2 / m1)
        s = dlog / (l2 - l1)
        note_slope = f"realized slope {s:.3f}"
        if abs(dlog) < EMPIRICAL_MIN_DLOG:
            # pair too close — the slope is noise. Tremor rays fall back to
            # the FITTED slope anchored at the latest measurement (full step);
            # anything else has no trustworthy step.
            if lever == "tremor" and tremor_fit is not None \
                    and float(getattr(tremor_fit, "slope", 0.0)) > 1e-9:
                s = float(tremor_fit.slope)
                note_slope = (f"pair ill-conditioned (|Δlog|={abs(dlog):.3f} < "
                              f"{EMPIRICAL_MIN_DLOG}) — fitted slope {s:.3f} "
                              "anchored at the measurement")
            else:
                return None
        # sanity: tremor/α loosen (s>0 along +lever), gain tightens (s<0)
        if lever == "gain":
            ok_dir = s < -1e-6
        else:
            ok_dir = s > 1e-6
        if not ok_dir:
            return None
        eff_target = max(p2.target_hbw, p2.frontier_hbw) if p2.refused \
            else p2.target_hbw
        l3 = max(0.0, l2 + math.log(eff_target / max(m2, 1e-9)) / s)
        pred = float(m2 * math.exp(s * (l3 - l2)))
        lad_raw = (gate.get("ladders") or {}).get(abbr) or {}
        lad = {float(k): float(v) for k, v in lad_raw.items()}
        out[abbr] = dataclasses.replace(
            p2, **{lever: round(l3, 4)},
            pred_hbw=round(pred, 4),
            pred_hbw_ci=(round(pred, 4), round(pred, 4)),
            achieved_pct=(round(hbw_to_pct(pred, lad), 1) if lad
                          else p2.achieved_pct),
            notes=(p2.notes + f"; empirical correction: {lever} "
                   f"{round(l2, 4)}→{round(l3, 4)} off the MEASURED ray "
                   f"({round(l1, 4)}:{m1:.3f} → {round(l2, 4)}:{m2:.3f}, "
                   f"{note_slope}) toward {eff_target:.3f}").lstrip("; "))
    return out


# ══ world-results / free-play readers (ported from v1) ════════════════════════

def _load_op_attack_targets(op_attack_path: Path) -> dict[str, Any]:
    """Human per-weapon op-attack targets from the collect-cached baseline
    (qnn.human). Fails loud with a backfill hint if absent — the cache owns
    computation, not this reader. Port of v1 decode_fit_pipeline.py:1186."""
    if not Path(op_attack_path).exists():
        raise FileNotFoundError(
            f"op-attack targets missing: {op_attack_path} — backfill the "
            "collect's decode-fit human baselines first "
            "(python -m qnn.human <collect_dir>).")
    return read_json(Path(op_attack_path))


def _op_shot_rates(npz: Path,
                   targets: dict[str, Any] | None = None,
                   op_attack_path: Path | None = None) -> dict[str, Any] | None:
    """WORLD-RESULTS attack pair for the trim/gate (owner directive: match
    what firing delivers, not button hygiene). Port of v1
    decode_fit_pipeline.py:1202, semantics unchanged.

    Bot: attack pulses per engaged second per held weapon from the eval action
    streams (a25 attack is single-tick, so pulses ARE shots at decision
    granularity). Human: corpus op-attack rate per held weapon evaluated at
    the BOT's weapon mix, so weapon preference cannot confound the attack
    trim. Weapons without human coverage drop out of both sides (shares
    renormalized over the covered mass). Returns {h_att, b_att} in fires/s
    plus per-weapon detail, or None when the npz is unreadable."""
    try:
        z = np.load(npz)
        hz = float(z["tick_hz"][0])
        att = z["attack"].astype(bool)
        wpn = z["weapon"].astype(np.int64)
        keep = z["keep"].astype(bool)
    except Exception:
        return None
    t = (targets or _load_op_attack_targets(op_attack_path))["weapons"]
    per: dict[str, Any] = {}
    eng_tot = fire_tot = h_weighted = covered = all_eng = 0.0
    for w in range(1, 9):
        m = keep & (wpn == w)
        n = float(m.sum())
        if not n:
            continue
        all_eng += n
        f = float((att & m).sum())
        name = _IMPULSE_NAME[w]
        h_rate = (t.get(name) or {}).get("rate_per_s")
        per[name] = {"engaged_ticks": int(n),
                     "bot_fire_per_s": round(f / n * hz, 3),
                     "human_fire_per_s": h_rate}
        if h_rate is not None:
            eng_tot += n
            fire_tot += f
            h_weighted += n * float(h_rate)
            covered += n
    if not eng_tot:
        return None
    return {"h_att": h_weighted / eng_tot, "b_att": fire_tot / eng_tot * hz,
            "per_weapon": per, "engaged_ticks": int(eng_tot),
            "coverage": round(covered / max(all_eng, 1.0), 3),
            "units": "fires/s while engaged, bot-mix-weighted human target"}


def _reactivity_hazard(npz: Path) -> dict[str, Any] | None:
    """Closed-loop threat reactivity vs the pinned human hazard, from the
    per-episode streams (threat_trace bit2). None on pre-bit2 streams.
    Port of v1 decode_fit_pipeline.py:1259."""
    try:
        z = np.load(npz)
    except Exception:
        return None
    cp_t = n_t = cp_c = n_c = 0
    for key in z.files:
        if not key.startswith("ep_"):
            continue
        tkey = "threat_" + key
        if tkey not in z.files:
            continue
        mv = z[key]                      # (T, 3) int8 move classes
        th = z[tkey].astype(np.uint8)    # (T,) flags
        if not (th & 4).any():
            continue                     # pre-bit2 stream (or no threat)
        cp = np.zeros(len(mv), bool)
        cp[1:] = (mv[1:, 0] != mv[:-1, 0]) | (mv[1:, 1] != mv[:-1, 1])
        threat = (th & 4).astype(bool)
        cp_t += int(cp[threat].sum()); n_t += int(threat.sum())
        cp_c += int(cp[~threat].sum()); n_c += int((~threat).sum())
    if n_t < 500:
        return None
    rt, rc_ = cp_t / n_t, cp_c / max(n_c, 1)
    return {"cp_rate_threat": round(rt, 4), "cp_rate_calm": round(rc_, 4),
            "hazard_ratio": round(rt / max(rc_, 1e-9), 4),
            "human_ref": HUMAN_REACTIVITY_HAZARD,
            "threat_frames": n_t}


def _rc_rates(ctx, npz: Path,
              human_cache: dict[str, float] | None = None) -> dict[str, float]:
    """Commensurate rate pair for the trim loop: engaged attack on/off switch
    rate + weapon selection switch rate, human and bot computed by the SAME
    rc_humanlikeness code path. Human side cached across iterations.
    Port of v1 decode_fit_pipeline.py:1345."""
    from qnn.eval.humanlikeness import rc
    bot = rc.collect_bot(npz)
    if human_cache is not None:
        h_att, h_wsw = human_cache["h_att"], human_cache["h_wsw"]
    else:
        human = rc.collect_human(ctx.corpus_dir, "precomputed_val")
        h_att = float(human.get("attack_switch", 0.0))
        h_wsw = float(human.get("weapon_switch", 0.0))
    return {"h_att": h_att, "b_att": float(bot.get("attack_switch", 0.0)),
            "h_wsw": h_wsw, "b_wsw": float(bot.get("weapon_switch", 0.0))}


def _score_human_band(ctx, npz: Path) -> dict[str, Any]:
    """Human-band membership (research/human-band.md): windowed per-channel
    behavior features vs the demo-split null, RBF-MMD² per channel, family
    verdict via max-rank. Port of v1 decode_fit_pipeline.py:1410 with ONE v2
    change: the verdict counts toward the gate only when the collect's band
    bank artifact already exists — a missing bank returns ``unscored``
    (does not fail) instead of triggering a full corpus featurization from
    inside a gate."""
    try:
        from qnn.eval.humanlikeness import human_band as hb
    except Exception as e:  # pragma: no cover
        return {"status": "unscored", "error": f"human_band import failed: {e!r}"}
    try:
        hz, eps = hb.load_rc_episodes(Path(npz))
        bank_path = hb.bank_cache_path(ctx.corpus_dir, hz, 15.0)
        if not bank_path.exists():
            return {"status": "unscored",
                    "note": (f"band bank artifact missing: {bank_path} — the "
                             "band arm is unscored (does not fail); build the "
                             "bank via qnn.human.band_bank.load_or_build_bank")}
        subj = hb.featurize(eps, hz, 15.0)
        bank, _ = hb.load_or_build_bank(ctx.corpus_dir, "precomputed_val", hz, 15.0)
        bctx = hb.band_context(bank, 17)
        # Null cohort split is RANDOM-only (the per-player skill map was a coh
        # heuristic, retired) — a valid, unbiased null (v1 comment carried).
        skill_map: dict[int, float] = {}
        counts = [int(subj[ch]["X"].shape[0]) for ch in hb.CHANNELS
                  if subj[ch]["X"].shape[0] >= 8]
        if not counts:
            return {"status": "unscored",
                    "error": "no usable behavior windows in npz"}
        n_use = min(256, min(counts))
        null = hb.build_null(bank, bctx, n_use, 300, 17, skill=skill_map)
        res = hb.score_subject(subj, bank, bctx, null, n_use, 30, 17)
        res["hz"] = hz
        res["status"] = "scored"
        res["contract"] = ("research/human-band.md: v5 discharge attack, "
                           "engaged conditioning; verdict = anchored ratio "
                           "≤ ANCHOR_RATIO_MAX (null pct report-only); "
                           "family = worst channel")
        return res
    except Exception as e:
        return {"status": "unscored", "error": f"human_band scoring failed: {e!r}"}


def _occupancy_shares(per_w: dict[str, dict]) -> dict[str, float]:
    """Per-weapon discharge-occupancy shares over the eval (discharges / Σ).
    Port of v1 decode_fit_pipeline.py:1666."""
    tot = sum(int(v.get("discharges", 0)) for v in per_w.values())
    if not tot:
        return {}
    return {w: round(int(v.get("discharges", 0)) / tot, 4)
            for w, v in sorted(per_w.items())}


def _expected_aggregate_intercept(
    per_w: dict[str, dict], per_weapon_target_hbw: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Occupancy-weighted expected aggregate INTERCEPT (hbw): Σ_w (eval
    discharge share of w × target hbw for w), over weapons that have BOTH a
    target and observed discharges (shares renormalized over that set). The
    free-play eval mixes weapons at their own discharge shares, so the placed
    aggregate hbw is a mixture — comparing it to a single scalar mis-reads.
    Port of v1 decode_fit_pipeline.py:1675."""
    if not per_weapon_target_hbw:
        return None
    shares: dict[str, tuple[int, float]] = {}
    for w, hbw in per_weapon_target_hbw.items():
        n = int(per_w.get(w, {}).get("discharges", 0))
        if n > 0 and hbw is not None:
            shares[w] = (n, float(hbw))
    tot = sum(n for n, _ in shares.values())
    if not tot:
        return None
    expected = sum((n / tot) * hbw for n, hbw in shares.values())
    total = sum(int(v.get("discharges", 0)) for v in per_w.values())
    return {
        "expected": expected,
        "weights": {w: round(n / tot, 4) for w, (n, _) in shares.items()},
        "per_weapon_target_hbw": {w: round(h, 4) for w, (_, h) in shares.items()},
        # coverage < 1 ⇒ some observed discharge occupancy has no target
        # (e.g. GL/Axe, intercept-invalid) and is excluded from the expectation.
        "coverage": round(tot / total, 4) if total else None,
    }


def _eval_summary_intercept_per_weapon(npz: Path) -> dict[str, dict]:
    """Per-weapon closed-loop INTERCEPT (median hbw + discharge count) from the
    eval's ``engine_intercept_hbw_per_weapon`` (RL's row is the feet-anchored
    pitch fixture). Port of v1 decode_fit_pipeline.py:1753."""
    summary = Path(npz).parent / "eval_summary.json"
    if not summary.exists():
        return {}
    s = json.loads(summary.read_text())
    per_w = s.get("engine_intercept_hbw_per_weapon") or {}
    out = {}
    for w, blk in per_w.items():
        if isinstance(blk, dict) and blk.get("median_hbw") is not None:
            out[w] = {"intercept_hbw": round(float(blk["median_hbw"]), 4),
                      "discharges": int(blk.get("n_attacks", 0))}
    return out


def _eval_summary_intercept(npz: Path) -> float | None:
    """Aggregate INTERCEPT (median hbw) from the eval_summary.json beside the
    streams npz. Port of v1 decode_fit_pipeline.py:1770."""
    summary = Path(npz).parent / "eval_summary.json"
    if not summary.exists():
        return None
    s = json.loads(summary.read_text())
    blk = s.get("engine_intercept_hbw")
    if not isinstance(blk, dict) or blk.get("median_hbw") is None:
        return None
    return float(blk["median_hbw"])


def _weapon_switch_diag(ctx, npz: Path) -> dict[str, Any]:
    """Weapon-selection switch rate diag (per second, both sides via
    rc_humanlikeness — port of the v1 _score_eval_npz weapon_switch block,
    decode_fit_pipeline.py:1630). Never blocks the gate."""
    try:
        from qnn.eval.humanlikeness import rc
        bot = rc.collect_bot(npz)
        human = rc.collect_human(ctx.corpus_dir, "precomputed_val")
        return {
            "human": round(float(human.get("weapon_switch", 0.0)) * human["hz"], 3),
            "bot": round(float(bot.get("weapon_switch", 0.0)) * bot["hz"], 3),
        }
    except Exception as e:  # pragma: no cover — diagnostic row only
        return {"error": repr(e)}


# ══ 3. style gate (decision 4) ════════════════════════════════════════════════

def style_gate(ctx, npz_path: Path, plans: dict[str, WeaponPlan], *,
               rel_tol: float = 0.25,
               native_npz: Path | None = None) -> dict[str, Any]:
    """The free-play DEPLOYMENT report (plan §P3, decision 4; band-v5 axioms).

    GATED arms — world-results invariants only (ALL must be True for PASS;
    an unmeasurable gated arm is ``None`` and cannot PASS — fail loud):
      * op-attack WORLD-RESULTS ruler — bot operative fires/s while engaged vs
        the bot-mix-weighted human target (ported ``_op_shot_rates`` /
        ``_load_op_attack_targets``; op-filter semantics unchanged),
        rel-delta ≤ ``rel_tol``.
      * threat-reactivity hazard — the LIFT criterion the trim converged on,
        re-checked at the deployable operating point.
      * weapon-switch rate — per-second both sides via rc_humanlikeness,
        rel-delta ≤ ``rel_tol`` (the LG-park detector).

    FLAGGED, never gating (doctrine, Brian 2026-07-16: skill placement is
    never capped for style — style spend is the TRAINING-TARGET register):
      * fitted-vs-native style spend (Axiom 3) — per channel, band-v5
        anchored ratio at the operating point vs the native reference
        (``native_npz``, the last styletrim wave: same style values, aim
        knobs zeroed); channels over native + ``STYLE_REGRESSION_MARGIN``
        are flagged loudly and stamped into promoted provenance.
      * band MEMBERSHIP (research/human-band.md: the research tracker).

    REPORT-ONLY (decision 4 / skill-curves §1.7b):
      * the v5 anchored band scores (subject + native reference);
      * per-weapon free-play intercept hbw beside each plan's ``pred_hbw``,
        with ``transfer_coeff = freeplay_hbw / pred_hbw`` — the fit→free-play
        domain gap, TRACKED per model instead of gated;
      * the occupancy-weighted aggregate intercept vs its plan expectation;
      * previous-rc non-regression placeholder (first v5-scored fit is the
        baseline; wire a reference report when one exists)."""
    npz = Path(npz_path)
    report: dict[str, Any] = {
        "npz": str(npz),
        "rel_tol": float(rel_tol),
        "contract": ("gated = world invariants ONLY (op-attack, reactivity, "
                     "weapon-switch); style spend + band membership are "
                     "FLAGGED/tracked, never gating — skill is never capped "
                     "for style (human-band.md axioms + Brian 2026-07-16); "
                     "per-weapon free-play hbw is a REPORT CARD (decision 4 "
                     "/ skill-curves §1.7b)"),
    }

    # ── GATED arm 1: op-attack world-results ruler ────────────────────────
    targets = _load_op_attack_targets(ctx.op_attack_path)
    op = _op_shot_rates(npz, targets=targets)
    if op is None:
        opattack_ok: bool | None = None
        report["op_attack"] = {
            "ok": None,
            "note": ("op-shot ruler unmeasurable (npz unreadable or no engaged "
                     "human-covered weapon mass) — a gated arm that cannot be "
                     "measured cannot PASS"),
        }
    else:
        rel = (abs(op["b_att"] - op["h_att"]) / op["h_att"]
               if op["h_att"] else float("inf"))
        opattack_ok = bool(rel <= rel_tol)
        report["op_attack"] = {
            "ok": opattack_ok,
            "bot_fire_per_s": round(op["b_att"], 4),
            "human_fire_per_s": round(op["h_att"], 4),
            "rel_delta": round(rel, 3) if np.isfinite(rel) else None,
            "rel_tol": float(rel_tol),
            "per_weapon": op["per_weapon"],
            "coverage": op["coverage"],
            "units": op["units"],
        }

    # ── GATED arm 2: threat-reactivity hazard at the operating point ──────
    react = _reactivity_hazard(npz)
    if react is None:
        react_ok: bool | None = None
        report["reactivity"] = {
            "ok": None,
            "note": ("hazard unmeasurable (<500 threat frames or pre-bit2 "
                     "stream) — a gated arm that cannot be measured cannot "
                     "PASS"),
        }
    else:
        _h_lift = HUMAN_REACTIVITY_HAZARD - 1.0
        _lift = float(react["hazard_ratio"]) - 1.0
        react_ok = bool(abs(_lift - _h_lift) <= TRIM_REACT_LIFT_TOL * _h_lift)
        report["reactivity"] = {"ok": react_ok, **react,
                                "lift_tol": TRIM_REACT_LIFT_TOL}

    # ── GATED arm 3: weapon-switch rate (the LG-park detector) ────────────
    wsw = _weapon_switch_diag(ctx, npz)
    if "error" in wsw or not wsw.get("human"):
        wsw_ok: bool | None = None
        report["weapon_switch_per_sec"] = {
            **wsw, "ok": None,
            "note": "switch rate unmeasurable — cannot PASS",
        }
    else:
        _rel = abs(wsw["bot"] - wsw["human"]) / wsw["human"]
        wsw_ok = bool(_rel <= rel_tol)
        report["weapon_switch_per_sec"] = {
            **wsw, "ok": wsw_ok,
            "rel_delta": round(_rel, 3), "rel_tol": float(rel_tol),
        }

    # ── FLAGGED (never gates): fitted-vs-native style spend (Axiom 3) ─────
    # Doctrine (Brian 2026-07-16): skill placement is NEVER capped for
    # style. Style spend outside the human envelope is measured, FLAGGED
    # loudly (stamped into promoted provenance), and registered as a
    # TRAINING TARGET — it does not block promotion. Band scores are
    # tracked on both sides; membership never gates either.
    band = _score_human_band(ctx, npz)
    report["human_band"] = band
    if native_npz is not None:
        native_band = _score_human_band(ctx, Path(native_npz))
    else:
        native_band = {"status": "unscored",
                       "note": "no native reference npz provided"}
    report["human_band_native"] = native_band
    judged: dict[str, dict[str, Any]] = {}
    for ch, c in (band.get("channels") or {}).items():
        rf = c.get("anchored_ratio")
        rn = ((native_band.get("channels") or {}).get(ch) or {}) \
            .get("anchored_ratio")
        if rf is not None and rn is not None:
            judged[ch] = {"fitted": rf, "native": rn,
                          "flagged": bool(rf > rn + STYLE_REGRESSION_MARGIN)}
    style_flags = sorted(ch for ch, v in judged.items() if v["flagged"])
    report["style_spend"] = {
        "gated": False,
        "flags": style_flags,
        "margin": STYLE_REGRESSION_MARGIN,
        "channels": judged,
        "measured": bool(judged),
        "contract": ("fitted-vs-native anchored-ratio spend per channel "
                     "(Axiom 3): flagged > native + margin. FLAG ONLY — "
                     "training-target register, never a skill cap; stamped "
                     "into promoted provenance. Unmeasured (pre-v5 npz) is "
                     "reported, not blocking."),
    }
    if style_flags:
        _log(f"style spend FLAGGED (training targets, not gating): "
             f"{style_flags} — fitted vs native "
             + str({ch: (judged[ch]['fitted'], judged[ch]['native'])
                    for ch in style_flags}))
    report["prev_rc_regression"] = {
        "ok": None, "gated": False,
        "note": ("no v5-scored prior rc on record — first v5 fit is the "
                 "baseline; wire a reference report when one exists"),
    }

    # ── REPORT-ONLY: per-weapon free-play hbw + transfer coefficient ──────
    per_w = _eval_summary_intercept_per_weapon(npz)
    card: dict[str, dict[str, Any]] = {}
    for w, plan in sorted(plans.items()):
        obs = per_w.get(w) or {}
        fp = obs.get("intercept_hbw")
        card[w] = {
            "freeplay_hbw": fp,
            "discharges": int(obs.get("discharges", 0)),
            "pred_hbw": plan.pred_hbw,
            # the fit→free-play domain-gap number we TRACK per model
            # (decision 4 / §1.7b) instead of gating on
            "transfer_coeff": (round(fp / plan.pred_hbw, 4)
                               if fp is not None and plan.pred_hbw else None),
            "target_pct": plan.target_pct,
            "achieved_pct_fit": plan.achieved_pct,
            "refused": plan.refused,
            "band": plan.band,
        }
    for w, obs in sorted(per_w.items()):      # observed but unplanned (GL/Axe)
        if w not in card:
            card[w] = {"freeplay_hbw": obs.get("intercept_hbw"),
                       "discharges": int(obs.get("discharges", 0)),
                       "pred_hbw": None, "transfer_coeff": None}
    report["freeplay_intercept"] = {
        "gated": False,
        "note": "report card only — never gates (decision 4 / §1.7b)",
        "per_weapon": card,
    }
    report["occupancy_shares"] = _occupancy_shares(per_w)

    agg = _eval_summary_intercept(npz)
    exp = _expected_aggregate_intercept(
        per_w, {w: p.pred_hbw for w, p in plans.items()})
    agg_row: dict[str, Any] = {"gated": False, "measured_hbw": agg}
    if exp is not None:
        agg_row.update({
            "expected_hbw": round(exp["expected"], 4),
            "weights": exp["weights"],
            "per_weapon_expected_hbw": exp["per_weapon_target_hbw"],
            "coverage": exp["coverage"],
            "rel_delta": (round(abs(agg - exp["expected"]) / exp["expected"], 3)
                          if agg is not None and exp["expected"] else None),
        })
    report["aggregate_intercept"] = agg_row

    gated = {"opattack_ok": opattack_ok, "reactivity_ok": react_ok,
             "wsw_ok": wsw_ok}
    report["gated"] = gated
    report["status"] = ("PASS" if all(v is True for v in gated.values())
                        else "FAIL")
    if style_flags and report["status"] == "PASS":
        report["status"] = "PASS"          # flags never demote the verdict
        report["status_note"] = f"PASS with style flags: {style_flags}"
    return report


# ══ 4. attack trim (world-results fire-rate calibration) ══════════════════════

def attack_trim(ctx, config_path: Path,
                launch_eval: Callable[[Path, str], Path | None], *,
                max_iters: int = TRIM_MAX_ITERS) -> dict[str, Any]:
    """Closed-loop attack-rate trim, redesigned after the a25rc3c gate FAILs
    (per-weapon imbalance + native→fitted transfer breach). Eval launch is
    injected: ``launch_eval(config_path, tag) → npz path | None``.

    MEASURED AT THE DEPLOYABLE OPERATING POINT. The v1 directive ("two fits,
    not one" — trim style at native so a style knob can't absorb a skill
    effect) is preserved in ORDER, not in venue: skill is placed and LOCKED
    by the confirmation gate before this trim runs, so there is no co-fit
    circularity — and the a25rc3c evidence killed the invariance premise
    (a native-converged trim measured 0.632 off on the fitted config; wsw
    collapsed 5× native→fitted under α-heavy aim). The gate scores the
    deployable, so the trim measures the deployable. One aim-zeroed eval at
    the converged style values is launched at the END as the NATIVE
    reference for the style-spend flags.

    PER-WEAPON: the attack ruler steps each weapon's ``attack.bias_vec``
    entry toward ITS human rate (the global ``attack.bias`` knob is not
    touched — a global step just moves the LG/SG imbalance around, the
    rc3d failure). Weapons under ``TRIM_PERWEAPON_MIN_TICKS`` engaged ticks
    ride the aggregate criterion only. ``attack.stick_bias`` (weapon-switch)
    and ``move.threat_break_hazard`` (reactivity) step as before.

    BACKTRACKING: every knob keeps a step scale that HALVES when its
    measured log-ratio flips sign (overshoot) — the stick_bias cliff:
    undamped chasing walked 1.45→3.49 and collapsed switching 5×.
    An iteration that fails criteria but has nothing to step returns
    STALLED (loud) instead of burning evals on an unchanged config.
    Missing npz / missing per-weapon engine table → abort with a note."""
    config_path = Path(config_path)
    report: dict[str, Any] = {
        "config": str(config_path),
        "tol": TRIM_TOL, "max_iters": int(max_iters),
        "measured_at": ("deployable operating point (skill locked by the "
                        "confirmation gate; per-weapon bias_vec + backtracking)"),
    }
    # working clone at the REQUESTED operating point (aim knobs kept)
    style_cfg = Path(str(config_path) + ".styletrim.json")
    style_cfg.write_text(json.dumps(read_json(config_path), indent=2) + "\n")

    back: dict[str, dict[str, float]] = {}

    def _step(key: str, raw: float, clamp: tuple[float, float]) -> float:
        """Damped step with per-knob backtracking: halve the knob's scale
        whenever its raw log-ratio flips sign (it overshot the target)."""
        st = back.setdefault(key, {"scale": 1.0, "sign": 0.0})
        sign = math.copysign(1.0, raw) if raw else 0.0
        if st["sign"] and sign and sign != st["sign"]:
            st["scale"] *= 0.5
        if sign:
            st["sign"] = sign
        return float(np.clip(st["scale"] * TRIM_DAMPING * raw, *clamp))

    human_rates: dict[str, float] | None = None
    op_targets = _load_op_attack_targets(ctx.op_attack_path)
    iters: list[dict[str, Any]] = []
    npz: Path | None = None
    converged = False
    for it in range(max_iters):
        npz = launch_eval(style_cfg, f"styletrim{it}")
        if npz is None:
            report.update(status="EVAL-FAILED", iterations=iters,
                          note=f"iteration {it} eval failed — trim aborted")
            _log(f"attack trim: iteration {it} eval failed — aborting")
            return report
        npz = Path(npz)
        rc_pair = _rc_rates(ctx, npz, human_cache=human_rates)
        human_rates = {"h_att": rc_pair["h_att"], "h_wsw": rc_pair["h_wsw"]}
        # ATTACK on the WORLD-RESULTS ruler (fires/s while engaged). Button-
        # transition rates equate a bot pulse (one shot) with a human
        # press-episode (N shots via auto-refire) and starve the fire rate —
        # the a25rc2a bug. Transitions stay as a diagnostic row (v1:1097-1121).
        op = _op_shot_rates(npz, targets=op_targets)
        if op is None:
            report.update(status="EVAL-FAILED", iterations=iters,
                          note=(f"iteration {it} eval lacks the per-weapon "
                                "engine_los_attack table — cannot measure the "
                                "op-shot ruler"))
            _log(f"attack trim: iteration {it} lacks the op-shot table — aborting")
            return report
        rates = {"h_att": op["h_att"], "b_att": op["b_att"],
                 "h_wsw": rc_pair["h_wsw"], "b_wsw": rc_pair["b_wsw"]}
        att_ratio = rates["h_att"] / max(rates["b_att"], 1e-6)
        wsw_ratio = rates["b_wsw"] / max(rates["h_wsw"], 1e-6)
        react = _reactivity_hazard(npz)
        # per-weapon ratios (human/bot, > 1 ⇒ under-firing) on the same ruler
        per_w = op.get("per_weapon") or {}
        w_ratio = {
            w: float(v["human_fire_per_s"]) / max(float(v["bot_fire_per_s"]), 1e-6)
            for w, v in per_w.items()
            if int(v.get("engaged_ticks", 0)) >= TRIM_PERWEAPON_MIN_TICKS
            and v.get("human_fire_per_s") and w in WEAPON_IMPULSE}
        _tol_log = float(np.log(1.0 + TRIM_TOL))
        w_off = {w: r for w, r in w_ratio.items()
                 if abs(math.log(r)) > _tol_log}
        row = {"iter": it, **{k: round(v, 5) for k, v in rates.items()},
               "att_ratio": round(att_ratio, 3), "wsw_ratio": round(wsw_ratio, 3),
               "att_ratio_per_weapon": {w: round(r, 3) for w, r in w_ratio.items()},
               "reactivity": react,
               "att_units": op["units"], "att_per_weapon": op["per_weapon"],
               "att_transitions_diag": {"h": round(rc_pair["h_att"], 5),
                                        "b": round(rc_pair["b_att"], 5)}}
        att_ok = (abs(float(np.log(att_ratio))) <= _tol_log) and not w_off
        wsw_ok = abs(float(np.log(wsw_ratio))) <= _tol_log
        react_ok = True
        r_meas = None
        if react is not None:
            r_meas = max(float(react["hazard_ratio"]), 1e-3)
            _h_lift = HUMAN_REACTIVITY_HAZARD - 1.0
            react_ok = (abs((r_meas - 1.0) - _h_lift)
                        <= TRIM_REACT_LIFT_TOL * _h_lift)
        if att_ok and wsw_ok and react_ok:
            iters.append({**row, "action": "converged"})
            converged = True
            break
        cfg = read_json(style_cfg)
        p = cfg["params"]
        updates: dict[str, Any] = {}
        if w_off:
            vec = list(p.get("attack.bias_vec") or [0.0] * 8)
            stepped: dict[str, float] = {}
            for w, r in sorted(w_off.items()):
                idx = WEAPON_IMPULSE[w] - 1
                if not 0 <= idx < len(vec):
                    continue
                d = _step(f"bias_vec.{w}", math.log(r), TRIM_VEC_STEP_CLAMP)
                vec[idx] = round(float(vec[idx]) + d, 4)
                stepped[w] = vec[idx]
            if stepped:
                p["attack.bias_vec"] = vec
                updates["attack.bias_vec"] = stepped
        if not wsw_ok:
            d = _step("stick_bias", float(np.log(wsw_ratio)),
                      TRIM_STICK_STEP_CLAMP)
            p["attack.stick_bias"] = round(
                max(0.0, float(p.get("attack.stick_bias", 0.0)) + d), 4)
            updates["attack.stick_bias"] = p["attack.stick_bias"]
        if react is not None and not react_ok:
            d = _step("threat_break",
                      (HUMAN_REACTIVITY_HAZARD - r_meas) * TRIM_REACT_STEP_SLOPE,
                      TRIM_REACT_STEP_CLAMP)
            p["move.threat_break_hazard"] = round(
                min(max(float(p.get("move.threat_break_hazard", 0.0)) + d, 0.0),
                    TRIM_REACT_HAZARD_MAX), 4)
            updates["move.threat_break_hazard"] = p["move.threat_break_hazard"]
        if not updates:
            # off-criteria but nothing steppable (e.g. aggregate residual with
            # every eligible weapon in-tol) — repeating the eval is waste
            iters.append({**row, "action": "stalled"})
            report.update(status="STALLED", iterations=iters,
                          note=("criteria unmet but no knob has a steppable "
                                "signal — trim stopped loud"))
            _log("attack trim: STALLED — criteria unmet, nothing to step")
            break
        style_cfg.write_text(json.dumps(cfg, indent=2) + "\n")
        iters.append({**row, "action": updates})
        _log(f"attack trim it{it}: att {rates['b_att']:.3f}/{rates['h_att']:.3f} "
             f"wsw {rates['b_wsw']:.3f}/{rates['h_wsw']:.3f} "
             f"react {(react or {}).get('hazard_ratio', '-')}"
             f"/{HUMAN_REACTIVITY_HAZARD} → {updates}")

    # freeze the trimmed style values into the deployable config
    _style = read_json(style_cfg)["params"]
    cfg = read_json(config_path)
    for k in ("attack.bias_vec", "attack.stick_bias",
              "move.threat_break_hazard"):
        if k in _style:
            cfg["params"][k] = _style[k]
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    style_cfg.unlink(missing_ok=True)
    if "status" not in report:
        report["status"] = "CONVERGED" if converged else "MAX-ITERS"
    report.update({
        "iterations": iters,
        "converged": converged,
        "style_values": {k: cfg["params"].get(k)
                         for k in ("attack.bias_vec", "attack.stick_bias",
                                   "move.threat_break_hazard")},
    })
    # NATIVE reference (aim knobs zeroed, converged style values) for the
    # style-spend flags — the one venue where native still matters.
    native_cfg = Path(str(config_path) + ".nativeref.json")
    _n = read_json(config_path)
    for k in ("look.aim_prior_gain", "look.aim_mag_gain",
              "look.aim_degrade_tremor_mag"):
        v = _n["params"].get(k)
        _n["params"][k] = [0.0] * len(v) if isinstance(v, list) else 0.0
    native_cfg.write_text(json.dumps(_n, indent=2) + "\n")
    native_npz = launch_eval(native_cfg, "native")
    native_cfg.unlink(missing_ok=True)
    report["native_npz"] = str(native_npz) if native_npz else None
    # the deployable npz for style_gate: on convergence the LAST trim eval
    # already measured the frozen values; otherwise launch a final eval so
    # the scored npz matches the emitted config.
    if converged and npz is not None:
        report["final_npz"] = str(npz)
    else:
        final_npz = launch_eval(config_path, "final")
        report["final_npz"] = str(final_npz) if final_npz else None
    reuse = report["final_npz"] or (str(npz) if npz else None)
    report["npz"] = reuse
    return report
