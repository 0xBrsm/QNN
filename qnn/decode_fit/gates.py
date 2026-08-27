"""Gates — placement (fit-CI), placement measurement, style gate, attack trim.

The gating confirmation round is GONE (Brian 2026-07-18): a confirm on the
fit's own substrate passes by construction, and one on a different substrate
fails whenever the fit is unrepresentative — so the fit must BE
representative (all four opponent pins, human range-mix weighting, per-wave
content-derived seeds) and is adjudicated on its own bootstrap CIs by
``placement_gate``. ``measure_placement`` remains as the report-only scorer
for ``--seed-replicates`` (refused plans promised the FRONTIER, so they
score against ``frontier_pct``). The style gate + attack trim run on free
play as before (a different estimand: natural play)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qnn.decode_fit.context import (CALIBRATION_FAMILY_KEY, IMPULSE_NAME,
                                    INSTRUMENT_WEAPONS, MODELNAME_TO_ABBR,
                                    TRANSFER_ALIAS, calibration_members,
                                    read_json)
from qnn.decode_fit.events import EventTable
from qnn.decode_fit.human_refs import hbw_to_pct
from qnn.decode_fit.occupancy import (RC1B_REFIRE_SEC,  # noqa: F401
                                      selection_profile as _selection_profile)
from qnn.decode_fit.response import GainResponse, WeaponPlan

# the 4 family representatives the botpin instrument measures directly.
INSTRUMENT_ABBRS = tuple(MODELNAME_TO_ABBR[w] for w in INSTRUMENT_WEAPONS)

# Below this effective discharge mass a measurement card is ``low_n``:
# neither pass nor fail, counted separately and flagged loud (spec: the
# gate must never adjudicate a weapon it barely observed).
CONFIRM_MIN_DISCHARGES = 50

# ── attack-trim constants (ported verbatim from v1 decode_fit_pipeline.py
# ~995-1011; semantics unchanged — see that file's comments for derivations) ──
# rc1b used 8 rounds for occupancy at a fixed switch margin. The reconciled
# controller fits fire, selection, switch, and threat-break jointly; preference
# corrections can perturb switching and therefore need verification headroom.
TRIM_MAX_ITERS = 12
TRIM_DAMPING = 0.6                 # step = damping × log-ratio (anti-oscillation)
TRIM_TOL = 0.15
# Threat-break hazard trim: additive step toward the human reactivity hazard;
# Δratio→λ slope ≈ 0.35. Tolerance is on the LIFT (ratio − 1), NOT the ratio
# (±15% of a 1.143 ratio contains "no reactivity at all" — the v1 it0 bug).
TRIM_REACT_STEP_SLOPE = 0.35
TRIM_REACT_STEP_CLAMP = (-0.03, 0.03)
TRIM_REACT_HAZARD_MAX = 0.15
TRIM_REACT_LIFT_TOL = 0.25
TRIM_STICK_STEP_CLAMP = (-0.75, 1.0)
# Conditional family cadence is owned by the four forced-weapon pin fit.
# Natural free play cannot identify a family the selector does not choose.
# Selection calibration: switch hysteresis is allowed to reduce jitter, but it
# may not rewrite the model's learned final-aim active-fire weapon mix. The
# zero-margin profile is measured per fit with the fixed a26rc1b estimator:
# operative discharges weighted by each weapon's refire duration. This avoids
# treating human weapon-script holds as genuine occupancy. A selection-only
# vector restores the reference. Shares below 1% on both sides are too noisy
# to steer; the 0.3% floor and 0.25 log tolerance are the rc1b values.
TRIM_PREF_STEP_CLAMP = (-0.75, 0.75)
TRIM_SELECTION_MIN_SHARE = 0.01
TRIM_SELECTION_SHARE_FLOOR = 0.003
TRIM_SELECTION_LOG_TOL = 0.25

# Frozen a26rc1b active-fire dwell weights (raw weapon id 1..8) and the
# occupancy estimand they weight. Provenance: the a26rc1b hand repair
# (awposw seed43) established both; they live in ``qnn.decode_fit.occupancy``
# beside the bias controllers that produced the rc1b config and are re-exported
# here for the trim and the style gate. Re-exported, never re-declared — the
# gate and the controllers must never drift into two occupancy rulers.

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

_IMPULSE_NAME = IMPULSE_NAME


def _log(msg: str) -> None:
    print(f"[decode-fit] {msg}", flush=True)


# ══ 1. placement measurement (report-only) ═════════════════════════════════════════

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


def measure_placement(table: EventTable, plans: dict[str, WeaponPlan],
                      ladder: dict[str, dict[float, float]], *,
                      tol_pct: float = 5.0, n_boot: int = 2000,
                      seed: int = 0) -> dict[str, Any]:
    """Score placement waves against the plans — MEASUREMENT ONLY.

    The gating confirmation round is gone (a confirm on the fit's own
    substrate passes by construction; the gate is now
    ``placement_gate`` on the fit's own CIs). This scorer remains for the
    report-only ``--seed-replicates`` re-measurement: per instrument weapon,
    measured median hbw + cluster-bootstrap CI mapped through the weapon's
    calibration-family human ladder; ``status``/verdicts are informational.
    Weapons under
    ``CONFIRM_MIN_DISCHARGES`` effective discharges are ``low_n``. SSG/NG
    ride the identical SG/SNG family plan. ``fit_calibration`` reports
    whether the fit predicted the measurement (measured median inside
    ``plan.pred_hbw_ci``)."""
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
            raise ValueError(f"measure_placement: no human ladder for {abbr}")
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
                            "fail; extend the placement wave")
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

    # ── transitive weapons (SSG/NG ride SG/SNG; own ladder placement) ─────
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
        "ladders": ladders_out,
    }
    if low_n:
        out["note"] = (f"LOW-N (loud): {low_n} under {CONFIRM_MIN_DISCHARGES} "
                       "effective discharges — unadjudicated; the placement "
                       "wave must be extended before these placements are trusted")
    return out


# ══ 2. placement gate — the fit adjudicates itself (no confirm round) ═════════

# a plan CI narrower than this relative width is DEGENERATE — some code path
# recorded a point promise as certainty (the a26 first-fit RL/SG/NG class)
DEGENERATE_CI_REL = 1e-6
# a representative fit needs this much effective discharge mass per weapon
PLACEMENT_MIN_EVENTS = 400


def placement_gate(plans: dict[str, WeaponPlan],
                   gain_fits: dict[str, GainResponse],
                   ladder: dict[str, dict[float, float]], *,
                   tol_pct: float = 5.0) -> dict[str, Any]:
    """Gate the placement on the FIT'S OWN uncertainty (Brian 2026-07-18:
    a confirmation round on the fit's substrate passes by construction, one
    on a different substrate fails whenever the fit is unrepresentative —
    the fix is a representative multi-seed fit whose CIs are trustworthy,
    gated here).

    Per plan (aliases included — their plans carry real CIs from the alias
    response on their calibration-family ladder): the prediction CI mapped to
    percentile
    must overlap [center − tol, center + tol] (center = ``frontier_pct`` for
    refused plans). HARD failures, loud, per weapon:

    * degenerate prediction CI (relative width < ``DEGENERATE_CI_REL``) —
      a point promise is never gate-able;
    * knee undetermined on the source gain fit after the extension round —
      the curve cannot place anything;
    * effective discharge mass under ``PLACEMENT_MIN_EVENTS`` — the fit is
      not the representative sample the gate's trust rests on.
    """
    cards: dict[str, dict[str, Any]] = {}
    passed, failed = [], []
    for abbr, plan in sorted(plans.items()):
        lad = ladder.get(abbr)
        if not lad:
            raise ValueError(f"placement_gate: no human ladder for {abbr}")
        src = plan.alias_of or abbr
        fit = gain_fits.get(src)
        center = float(plan.frontier_pct if plan.refused else plan.target_pct)
        lo, hi = float(plan.pred_hbw_ci[0]), float(plan.pred_hbw_ci[1])
        pct_ci = _pct_interval((lo, hi), lad)
        overlap = (pct_ci[0] <= center + tol_pct) and \
                  (pct_ci[1] >= center - tol_pct)
        reasons: list[str] = []
        if (hi - lo) < DEGENERATE_CI_REL * max(abs(plan.pred_hbw), 1e-9):
            reasons.append(
                f"degenerate prediction CI [{lo}, {hi}] — a point promise "
                "reached the gate (fix the CI path, never widen the tol)")
        if fit is None:
            reasons.append(f"no gain response for source {src!r}")
        else:
            if fit.knee_undetermined:
                reasons.append("knee undetermined after the extension round "
                               "— the curve cannot place a target")
            if fit.n_events < PLACEMENT_MIN_EVENTS:
                reasons.append(
                    f"fit mass {fit.n_events} < {PLACEMENT_MIN_EVENTS} "
                    "effective discharges — not a representative sample")
        if not overlap and not reasons:
            reasons.append(
                f"predicted placement p{pct_ci[0]:.1f}..p{pct_ci[1]:.1f} "
                f"misses the promise band "
                f"p{center - tol_pct:g}..p{center + tol_pct:g}")
        verdict = "pass" if (overlap and not reasons) else "fail"
        cards[abbr] = {
            "verdict": verdict,
            "target_pct": float(plan.target_pct),
            "refused": bool(plan.refused),
            "center_pct": center,
            "tol_band_pct": [round(center - tol_pct, 2),
                             round(center + tol_pct, 2)],
            "band": plan.band,
            "alias_of": plan.alias_of,
            "pred_hbw": plan.pred_hbw,
            "pred_hbw_ci": [round(lo, 4), round(hi, 4)],
            "pred_pct_ci": [round(pct_ci[0], 1), round(pct_ci[1], 1)],
            **({"fit_mass": fit.n_events, "fit_clusters": fit.n_clusters,
                "n_boot_ok": (fit.diagnostics or {}).get("n_boot_ok")}
               if fit is not None else {}),
            **({"fail_reasons": reasons} if reasons else {}),
        }
        (passed if verdict == "pass" else failed).append(abbr)
    return {
        "status": "PASS" if not failed else "FAIL",
        "criterion": (f"fit-CI placement at ±{tol_pct:g}: the plan's own "
                      "prediction CI must land in the promise band; "
                      "degenerate CI / undetermined knee / thin mass fail "
                      "hard (no confirmation round — the fit adjudicates "
                      "itself, so its uncertainty must be honest)"),
        "tol_pct": float(tol_pct),
        "weapons": cards,
        "passed": passed, "failed": failed,
    }


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
                   targets: dict[str, Any]) -> dict[str, Any] | None:
    """WORLD-RESULTS attack pair for the trim/gate (owner directive: match
    what firing delivers, not button hygiene). Port of v1
    decode_fit_pipeline.py:1202, semantics unchanged.

    Bot: attack pulses per engaged second per held weapon from the eval action
    streams (a25 attack is single-tick, so pulses ARE shots at decision
    granularity). SG/SSG and NG/SNG are pooled before rate and coverage
    decisions. Human: corpus op-attack rate per calibration family evaluated
    at the BOT's family mix, so weapon preference cannot confound the attack
    trim. Weapons without human coverage drop out of both sides. Returns
    {h_att, b_att} in fires/s plus descriptive per-weapon and load-bearing
    per-family detail, or None when the npz is unreadable."""
    try:
        z = np.load(npz)
        hz = float(z["tick_hz"][0])
        att = z["attack"].astype(bool)
        wpn = z["weapon"].astype(np.int64)
        keep = z["keep"].astype(bool)
    except Exception:
        return None
    t = targets["weapons"]
    per: dict[str, Any] = {}
    family_obs: dict[str, dict[str, Any]] = {}
    all_eng = 0.0
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
        family = CALIBRATION_FAMILY_KEY.get(name, name)
        acc = family_obs.setdefault(
            family, {"engaged_ticks": 0.0, "fires": 0.0, "members": []})
        acc["engaged_ticks"] += n
        acc["fires"] += f
        acc["members"].append(name)

    per_family: dict[str, Any] = {}
    eng_tot = fire_tot = h_weighted = covered = 0.0
    for family, obs in family_obs.items():
        members = calibration_members(family)
        human_rows = [(t.get(member) or {}) for member in members]
        weighted = [(float(row["rate_per_s"]),
                     float(row.get("engaged_los_ticks") or 1.0))
                    for row in human_rows if row.get("rate_per_s") is not None]
        h_rate = (sum(rate * weight for rate, weight in weighted)
                  / sum(weight for _, weight in weighted)) if weighted else None
        n = float(obs["engaged_ticks"])
        f = float(obs["fires"])
        per_family[family] = {
            "members": list(members),
            "engaged_ticks": int(n),
            "bot_fire_per_s": round(f / n * hz, 3),
            "human_fire_per_s": round(h_rate, 4) if h_rate is not None else None,
        }
        if h_rate is not None:
            eng_tot += n
            fire_tot += f
            h_weighted += n * h_rate
            covered += n
    if not eng_tot:
        return None
    return {"h_att": h_weighted / eng_tot, "b_att": fire_tot / eng_tot * hz,
            "per_weapon": per, "per_family": per_family,
            "engaged_ticks": int(eng_tot),
            "coverage": round(covered / max(all_eng, 1.0), 3),
            "units": ("fires/s while engaged, bot-family-mix-weighted human "
                      "target; SG+SSG and NG+SNG pooled")}


def _selection_log_ratios(target_shares: dict[str, float],
                          observed_shares: dict[str, float]) -> dict[str, float]:
    """rc1b log-ratio ruler for dynamically meaningful weapons."""
    return {
        w: math.log(max(float(target_shares.get(w, 0.0)),
                        TRIM_SELECTION_SHARE_FLOOR)
                    / max(float(observed_shares.get(w, 0.0)),
                          TRIM_SELECTION_SHARE_FLOOR))
        for w in target_shares
        if max(float(target_shares.get(w, 0.0)),
               float(observed_shares.get(w, 0.0)))
        >= TRIM_SELECTION_MIN_SHARE
    }


def _reactivity_hazard(npz: Path, *, n_boot: int = 0,
                       seed: int = 0) -> dict[str, Any] | None:
    """Closed-loop threat reactivity vs the pinned human hazard, from the
    per-episode streams (threat_trace bit2). None on pre-bit2 streams.
    Port of v1 decode_fit_pipeline.py:1259.

    ``n_boot`` > 0 adds a cluster-bootstrap CI on ``hazard_ratio`` (episodes
    are the independence unit — ticks within one episode's threat/calm
    windows are correlated). Off by default: the attack-trim loop calls
    this every iteration and doesn't consult the CI, so it stays cheap;
    ``style_gate``'s GATED measurement turns it on. Motivated by a real
    awposw-3 case (2026-07-19): one FROZEN style config measured
    hazard_ratio 1.083-1.144 across separately-launched freeplay waves —
    noise wider than TRIM_REACT_LIFT_TOL — so a bare point estimate at the
    gate boundary is not trustworthy (the same single-wave-noise failure
    mode the no-confirm redesign already fixed for placement_gate)."""
    try:
        z = np.load(npz)
    except Exception:
        return None
    ep_counts: list[tuple[int, int, int, int]] = []
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
        ep_counts.append((int(cp[threat].sum()), int(threat.sum()),
                          int(cp[~threat].sum()), int((~threat).sum())))
    n_t = sum(c[1] for c in ep_counts)
    if n_t < 500:
        return None
    cp_t = sum(c[0] for c in ep_counts)
    cp_c = sum(c[2] for c in ep_counts)
    n_c = sum(c[3] for c in ep_counts)
    rt, rc_ = cp_t / n_t, cp_c / max(n_c, 1)
    result = {"cp_rate_threat": round(rt, 4), "cp_rate_calm": round(rc_, 4),
              "hazard_ratio": round(rt / max(rc_, 1e-9), 4),
              "human_ref": HUMAN_REACTIVITY_HAZARD,
              "threat_frames": n_t}
    if n_boot > 0 and len(ep_counts) >= 4:
        arr = np.array(ep_counts, dtype=np.float64)   # (E, 4): cp_t/n_t/cp_c/n_c
        rng = np.random.default_rng(seed)
        n_ep = len(arr)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, n_ep, size=n_ep)
            s = arr[pick].sum(axis=0)
            b_rt = s[0] / s[1] if s[1] > 0 else 0.0
            b_rc = s[2] / max(s[3], 1)
            boots[b] = b_rt / max(b_rc, 1e-9)
        result["hazard_ratio_ci"] = [round(float(np.percentile(boots, 2.5)), 4),
                                      round(float(np.percentile(boots, 97.5)), 4)]
        result["n_episodes"] = int(n_ep)
    return result


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
               native_npz: Path | None = None,
               selection_target: dict[str, Any] | None = None) -> dict[str, Any]:
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
        rel-delta ≤ ``rel_tol``;
      * rc1b active-fire occupancy (operative discharges weighted by refire
        duration) remains within the rc1b log tolerance per meaningful weapon
        of the model's own zero-margin final-aim reference.

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
            "per_family": op.get("per_family", op["per_weapon"]),
            "per_weapon": op["per_weapon"],
            "coverage": op["coverage"],
            "units": op["units"],
        }

    # ── GATED arm 2: threat-reactivity hazard at the operating point ──────
    # CI, not a bare point estimate: a frozen config's hazard_ratio swung
    # 1.083-1.144 across separately-launched freeplay waves (awposw-3,
    # 2026-07-19) — noise wider than the tolerance band below. Verdict is
    # CI-overlap (placement_gate's pattern), same estimand the trim loop's
    # own iterations already show is this noisy.
    react = _reactivity_hazard(npz, n_boot=2000)
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
        _lift_lo = _h_lift - TRIM_REACT_LIFT_TOL * _h_lift
        _lift_hi = _h_lift + TRIM_REACT_LIFT_TOL * _h_lift
        ci = react.get("hazard_ratio_ci")
        if ci is not None:
            react_ok = bool((ci[1] - 1.0) >= _lift_lo and (ci[0] - 1.0) <= _lift_hi)
        else:
            _lift = float(react["hazard_ratio"]) - 1.0
            react_ok = bool(_lift_lo <= _lift <= _lift_hi)
        report["reactivity"] = {"ok": react_ok, **react,
                                "lift_tol": TRIM_REACT_LIFT_TOL,
                                "lift_band": [round(_lift_lo, 4), round(_lift_hi, 4)]}

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

    # ── GATED arm 4: hysteresis may not rewrite learned weapon occupancy ──
    selection = _selection_profile(npz)
    if selection is None or selection_target is None:
        selection_ok: bool | None = None
        report["weapon_occupancy"] = {
            "ok": None,
            "note": "selection profile/reference unmeasurable — cannot PASS",
        }
    else:
        target_shares = selection_target.get("shares") or {}
        observed_shares = selection.get("shares") or {}
        ratios = _selection_log_ratios(target_shares, observed_shares)
        worst = max((abs(v) for v in ratios.values()), default=0.0)
        selection_ok = bool(worst <= TRIM_SELECTION_LOG_TOL)
        report["weapon_occupancy"] = {
            "ok": selection_ok,
            "metric": "refire-weighted operative-discharge share (a26rc1b)",
            "target_shares": {w: round(float(v), 4)
                              for w, v in target_shares.items()},
            "observed_shares": {w: round(float(v), 4)
                                for w, v in observed_shares.items()},
            "log_ratio": {w: round(v, 3) for w, v in ratios.items()},
            "max_abs_log_ratio": round(worst, 3),
            "tol": round(TRIM_SELECTION_LOG_TOL, 4),
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
             "wsw_ok": wsw_ok, "selection_ok": selection_ok}
    report["gated"] = gated
    report["status"] = ("PASS" if all(v is True for v in gated.values())
                        else "FAIL")
    if style_flags and report["status"] == "PASS":
        # flags never demote the verdict; they only annotate it
        report["status_note"] = f"PASS with style flags: {style_flags}"
    return report


# ══ 4. attack trim (world-results fire-rate calibration) ══════════════════════

def attack_trim(ctx, config_path: Path,
                launch_eval: Callable[[Path, str], Path | None], *,
                max_iters: int = TRIM_MAX_ITERS) -> dict[str, Any]:
    """Closed-loop final-behavior calibration at the deployable aim point.

    The a26/a27 reconciliation gives each estimand one explicit control:

    * ``attack.fire_bias_vec`` is fixed by the forced-family cadence pins and
      is never rewritten from selection-starved free play;
    * ``weapon.switch_margin`` fits human switch rate;
    * ``weapon.preference_bias_vec`` restores the model's own zero-margin,
      final-aim refire-weighted discharge occupancy after hysteresis is added;
    * ``move.threat_break_hazard`` fits threat reactivity.

    A mandatory reference wave measures that active-fire occupancy before the
    loop with rc1b's discharge×refire estimator. This operationalizes the
    successful rc1b hand repair, avoids the human weapon-scripting confound,
    and prevents
    rate correction from becoming an LG/RL preference optimizer. All controls
    are then iterated jointly because changing the selected weapon legitimately
    changes its conditional fire response. Eval launch is injected as
    ``launch_eval(config_path, tag) → npz path | None``.

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
        "measured_at": ("deployable final-aim operating point; forced-pin "
                        "cadence fixed, selection/switch/reactivity backtracking"),
    }
    requested = read_json(config_path)
    p0 = requested.get("params") or {}
    legacy = [float(x) for x in (p0.get("attack.bias_vec") or [0.0] * 8)]
    if p0.get("attack.vector_semantics") != "split_v1" or any(abs(x) > 1e-9 for x in legacy):
        report.update(
            status="INVALID-SEMANTICS", converged=False,
            note=("reconciled trim requires attack.vector_semantics=split_v1 "
                  "and zero legacy attack.bias_vec; refusing to reinterpret an "
                  "a26/a27 artifact whose vector had branch-specific meaning"))
        return report

    # Reference = this model at its FINAL aim operating point, with selection
    # controls neutral. Fire pins stay active because they are part of the
    # observable substrate whose learned weapon mix we are preserving.
    reference_cfg = Path(str(config_path) + ".selectionref.json")
    ref = json.loads(json.dumps(requested))
    ref["params"]["weapon.switch_margin"] = 0.0
    ref["params"]["weapon.preference_bias_vec"] = [0.0] * 8
    reference_cfg.write_text(json.dumps(ref, indent=2) + "\n")
    reference_npz = launch_eval(reference_cfg, "selectionref")
    reference_cfg.unlink(missing_ok=True)
    selection_target = (_selection_profile(Path(reference_npz))
                        if reference_npz is not None else None)
    if selection_target is None:
        report.update(
            status="EVAL-FAILED", converged=False,
            selection_reference_npz=(str(reference_npz) if reference_npz else None),
            note="zero-margin rc1b discharge-occupancy reference was unmeasurable")
        return report
    report["selection_reference_npz"] = str(reference_npz)
    report["selection_target"] = selection_target
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
        selection = _selection_profile(npz)
        if selection is None:
            report.update(status="EVAL-FAILED", iterations=iters,
                          note=(f"iteration {it} lacks the rc1b operative-"
                                "discharge occupancy stream"))
            return report
        target_shares = selection_target["shares"]
        observed_shares = selection["shares"]
        selection_log_ratio = _selection_log_ratios(
            target_shares, observed_shares)
        selection_max_log = max((abs(v) for v in selection_log_ratio.values()),
                                default=0.0)
        # Per-family ratios are report-only here. The four forced-weapon pins
        # own conditional cadence because natural selection can starve a family.
        per_w = op.get("per_family") or op.get("per_weapon") or {}
        w_ratio = {
            w: float(v["human_fire_per_s"]) / max(float(v["bot_fire_per_s"]), 1e-6)
            for w, v in per_w.items()
            if int(v.get("engaged_ticks", 0)) > 0 and v.get("human_fire_per_s")}
        _tol_log = float(np.log(1.0 + TRIM_TOL))
        aggregate_att_ok = abs(float(np.log(att_ratio))) <= _tol_log
        row = {"iter": it, **{k: round(v, 5) for k, v in rates.items()},
               "att_ratio": round(att_ratio, 3), "wsw_ratio": round(wsw_ratio, 3),
               "att_ratio_per_family": {w: round(r, 3) for w, r in w_ratio.items()},
               "aggregate_attack_gate_ok": aggregate_att_ok,
               "reactivity": react,
               "selection_shares": {w: round(v, 4)
                                    for w, v in observed_shares.items()},
               "selection_target_shares": {w: round(v, 4)
                                           for w, v in target_shares.items()},
               "selection_log_ratio": {w: round(v, 3)
                                       for w, v in selection_log_ratio.items()},
               "selection_max_abs_log_ratio": round(selection_max_log, 3),
               "att_units": op["units"],
               "att_per_family": op.get("per_family", per_w),
               "att_per_weapon": op["per_weapon"],
               "att_transitions_diag": {"h": round(rc_pair["h_att"], 5),
                                        "b": round(rc_pair["b_att"], 5)}}
        wsw_ok = abs(float(np.log(wsw_ratio))) <= _tol_log
        selection_ok = selection_max_log <= TRIM_SELECTION_LOG_TOL
        react_ok = True
        r_meas = None
        if react is not None:
            r_meas = max(float(react["hazard_ratio"]), 1e-3)
            _h_lift = HUMAN_REACTIVITY_HAZARD - 1.0
            react_ok = (abs((r_meas - 1.0) - _h_lift)
                        <= TRIM_REACT_LIFT_TOL * _h_lift)
        if wsw_ok and selection_ok and react_ok:
            iters.append({**row, "action": "converged"})
            converged = True
            break
        cfg = read_json(style_cfg)
        p = cfg["params"]
        updates: dict[str, Any] = {}
        if not wsw_ok:
            d = _step("switch_margin", float(np.log(wsw_ratio)),
                      TRIM_STICK_STEP_CLAMP)
            p["weapon.switch_margin"] = round(
                max(0.0, float(p.get("weapon.switch_margin", 0.0)) + d), 4)
            updates["weapon.switch_margin"] = p["weapon.switch_margin"]
        if not selection_ok:
            pref = list(p.get("weapon.preference_bias_vec") or [0.0] * 8)
            stepped_pref: dict[str, float] = {}
            for w, raw in sorted(selection_log_ratio.items()):
                impulse = next((i for i, name in _IMPULSE_NAME.items()
                                if name == w), None)
                if impulse is None:
                    continue
                d = _step(f"preference_bias_vec.{w}", raw,
                          TRIM_PREF_STEP_CLAMP)
                pref[impulse - 1] = float(pref[impulse - 1]) + d
            # A constant offset is selection-invariant; center it so the vector
            # has a unique, auditable representation and clip runaway attractors.
            center = float(np.mean(pref))
            pref = [round(float(np.clip(v - center, -8.0, 8.0)), 4)
                    for v in pref]
            for impulse, name in _IMPULSE_NAME.items():
                if name in selection_log_ratio:
                    stepped_pref[name] = pref[impulse - 1]
            p["weapon.preference_bias_vec"] = pref
            updates["weapon.preference_bias_vec"] = stepped_pref
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
    for k in ("attack.fire_bias_vec", "weapon.preference_bias_vec",
              "weapon.switch_margin",
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
                         for k in ("attack.fire_bias_vec",
                                   "weapon.preference_bias_vec",
                                   "weapon.switch_margin",
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
