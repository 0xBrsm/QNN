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
from qnn.decode_fit.occupancy import (
    CHOICE_PROB_TOL,
    CONTINUE_PROB_TOL,
    RC1B_REFIRE_SEC,  # noqa: F401
    human_weapon_behavior_reference,
    selection_profile as _selection_profile,
    weapon_behavior_report,
)
from qnn.decode_fit.response import GainResponse, WeaponPlan
from qnn.human import attack_conditional as attack_ref

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
TRIM_CONTINUE_STEP_CLAMP = (-0.75, 0.75)
TRIM_TRANSITION_BIAS_CLAMP = (-8.0, 8.0)
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

# ── ATTACK-GATE RESPEC (Brian 2026-08-15) ───────────────────────────────────
# op_attack (marginal fires/s-while-engaged) is a SKILL x STYLE product: a
# model with human-conditional trigger behavior (fires more readily exactly
# when well-aligned, same as a human) plus ELITE aim availability (well
# aligned far more often than a human) is CORRECTLY expected to exceed the
# human marginal ARITHMETICALLY — E12/E13
# (agents/plans/a26-superiority-decomposition.md) caught exactly this case:
# a28p90floor refused on op_attack (LG 6.89 vs human 2.92 fires/s engaged)
# while decisively beating a26rc1b and every human-band channel it was
# scored against. The marginal is demoted to REPORT-ONLY below; the
# CONDITIONAL trigger shape (P(discharge | aim-error bin, op-ready), the E7
# instrument) and HOLD TEXTURE (discharge run-length / inter-burst-gap
# shape, engaged-conditioned) carry the humanness content instead — style
# (how the trigger behaves given aim quality) separated from skill (how good
# the aim is, never capped — feedback_aim_above_human_ceiling).
# Thin-data floor shared with the human reference builder (qnn.human.
# attack_conditional.THIN_DATA_MIN_TICKS) — a family under this many bot
# op-ready ticks falls back to the POOLED comparison; pooled itself under the
# floor is UNSCORED (never FAIL — a gate arm must never adjudicate a weapon
# it barely observed, the same doctrine as CONFIRM_MIN_DISCHARGES above).
ATTACK_ARM_MIN_TICKS = attack_ref.THIN_DATA_MIN_TICKS

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
    decode_fit_pipeline.py:1202, LOS-gated 2026-08-07 (blind-fire-cadence.md).

    Bot: attack pulses per engaged second per equipped weapon from the eval action
    streams (a25 attack is single-tick, so pulses ARE shots at decision
    granularity). SG/SSG and NG/SNG are pooled before rate and coverage
    decisions. Human: corpus op-attack rate per calibration family evaluated
    at the BOT's family mix, so weapon preference cannot confound the attack
    trim. Weapons without human coverage drop out of both sides. Returns
    {h_att, b_att} in fires/s plus descriptive per-weapon and load-bearing
    per-family detail, or None when the npz is unreadable.

    DENOMINATOR, matched to the human side: ``keep`` alone (any actor-type
    token present, ANY modality — SIGHT, SOUND, MEMORY, or PROXIMITY;
    ``qnn.eval.run``'s ``_keep = (entity_types == ACTOR).any()``) is a strict
    SUPERSET of the ``target_rate_per_s`` this compares against
    (``qnn.human.op_attack``, gated on an in-LOS actor at recency 0). Scoring
    ``keep`` alone against an LOS-gated human target is the asymmetric-gate
    bug agents/plans/blind-fire-cadence.md flags: the bot side counts ticks
    (remembered/heard/PVS-only targets) the human side structurally cannot.
    ``engaged`` (band-v5's strict in-LOS flag, the same underlying engine fact
    ``AlignHbw.los`` reads) is ANDed in so both sides require an actor being
    actively SIGHTed this exact tick."""
    try:
        z = np.load(npz)
        hz = float(z["tick_hz"][0])
        att = z["attack"].astype(bool)
        # Equipped weapon (schema 6 `self_weapon_id`; schema-5 name
        # `weapon_held`) — the denominator stream: engaged time attributed
        # to the weapon that would fire, a state attribution, not a
        # behavioral signal. The legacy `weapon` column is the attack-with
        # DECISION class (0 off fire ticks): reading it here makes
        # fires/held ≡ hz for every weapon (the a28rc1 trim stall).
        # Old-schema npz are unmeasurable, which re-produces the wave
        # rather than passing a broken ruler.
        if "self_weapon_id" in z:
            wpn = z["self_weapon_id"].astype(np.int64)
        elif "weapon_held" in z:
            wpn = z["weapon_held"].astype(np.int64)
        else:
            return None
        keep = z["keep"].astype(bool) & z["engaged"].astype(bool)
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
    rate + weapon-PREFERENCE switch rate (switches per consecutive-discharge
    pair; core.preference_pairs), human and bot computed by the SAME
    rc_humanlikeness code path. Human side cached across iterations.
    Port of v1 decode_fit_pipeline.py:1345; weapon ruler re-based 2026-08-09
    (held-weapon churn retired — weapon scripting fallacy)."""
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
        # v6 ENCOUNTER unit is the ADOPTED verdict instrument (Brian
        # 2026-08-15, research/human-band.md §v6, bar ENCOUNTER_RATIO_MAX);
        # the v5 15s row rides along as `unit_15s` for continuity with
        # pre-8/15 subjects. Same missing-bank guard for both units.
        def _score_unit(encounter: bool) -> dict:
            if encounter:
                bank_path = hb.bank_cache_path_encounters(ctx.corpus_dir, hz,
                                                          2.2)
            else:
                bank_path = hb.bank_cache_path(ctx.corpus_dir, hz, 15.0)
            if not bank_path.exists():
                return {"status": "unscored",
                        "note": (f"band bank artifact missing: {bank_path} — "
                                 "unscored (does not fail); build via "
                                 "qnn.human.band_bank")}
            if encounter:
                subj = hb.featurize_encounters(eps, hz)
                bank, _ = hb.load_or_build_bank_encounters(
                    ctx.corpus_dir, "precomputed_val", hz)
                bar = hb.ENCOUNTER_RATIO_MAX
            else:
                subj = hb.featurize(eps, hz, 15.0)
                bank, _ = hb.load_or_build_bank(
                    ctx.corpus_dir, "precomputed_val", hz, 15.0)
                bar = hb.ANCHOR_RATIO_MAX
            bctx = hb.band_context(bank, 17)
            # Null cohort split is RANDOM-only (the per-player skill map was
            # a coh heuristic, retired) — a valid, unbiased null (v1 carried).
            counts = [int(subj[ch]["X"].shape[0]) for ch in hb.CHANNELS
                      if subj[ch]["X"].shape[0] >= 8]
            if not counts:
                return {"status": "unscored",
                        "error": "no usable behavior windows in npz"}
            n_use = min(256, min(counts))
            null = hb.build_null(bank, bctx, n_use, 300, 17, skill={})
            res = hb.score_subject(subj, bank, bctx, null, n_use, 30, 17,
                                   ratio_max=bar)
            res["hz"] = hz
            res["status"] = "scored"
            return res

        res = _score_unit(encounter=True)
        res["contract"] = ("research/human-band.md §v6 (ADOPTED 8/15): "
                           "ENCOUNTER-unit scoring, verdict = anchored ratio "
                           "≤ ENCOUNTER_RATIO_MAX 0.364 (null pct "
                           "report-only); family = worst channel; 15s v5 row "
                           "under unit_15s for continuity")
        res["unit_15s"] = _score_unit(encounter=False)
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
    """Weapon-PREFERENCE switch rate diag: switches per counted
    consecutive-discharge pair, both sides via rc_humanlikeness
    (core.preference_pairs — equip-state churn is retired; weapon scripting
    makes it a fallacy). Pairs are gated to STATIONARY-MENU windows by the
    SAME rule on both sides (rc._invalidated_pairs: weapon-feasibility-mask
    change or death in the pair window — a menu change says nothing about
    preference): human from the corpus items/ammo/health streams, bot from
    the schema-6 ``weapon_feas``/``health`` columns. On pre-6 npz the bot
    side has no gate streams and reports ungated pairs (upper bound) — the
    ``feas_gated`` flags say which ruler ran. Never blocks the gate."""
    try:
        from qnn.eval.humanlikeness import rc
        bot = rc.collect_bot(npz)
        human = rc.collect_human(ctx.corpus_dir, "precomputed_val")
        return {
            "human": round(float(human.get("weapon_switch", 0.0)), 4),
            "bot": round(float(bot.get("weapon_switch", 0.0)), 4),
            "pairs": {"human": int(human.get("weapon_pairs", 0)),
                      "bot": int(bot.get("weapon_pairs", 0))},
            "feas_gated": {
                "human": bool(human.get("weapon_feas_gated", False)),
                "bot": bool(bot.get("weapon_feas_gated", False))},
            "units": ("preference switches per consecutive-discharge pair "
                      "(stationary-menu gated)"),
        }
    except Exception as e:  # pragma: no cover — diagnostic row only
        return {"error": repr(e)}


# ══ 2.5 ATTACK-GATE RESPEC arms (Brian 2026-08-15) ═══════════════════════════

def _load_attack_streams(npz: Path) -> dict[str, Any] | None:
    """The RESPEC's per-tick streams from a wave npz: ``aim_err_deg`` +
    ``op_ready`` (both new, qnn.eval.run._log_streams / qnn.eval.h2h) plus the
    pre-existing ``discharge`` / ``engaged`` / weapon-identity columns. None
    when the npz is unreadable OR predates the respec (missing aim_err_deg or
    op_ready) — a PRE-RESPEC WAVE, the case both new arms must report
    unscored (never fail) rather than silently score a zero-filled stream.

    ``discharge_collapsed`` (2026-08-15 hold-texture fix #2, coordinator
    iteration 3): the bot's raw ``discharge`` lane fires on the engine's
    LITERAL per-shot re-entry cadence for a held continuous weapon, not the
    EFFECTIVE cadence the human corpus's op-attack convention already
    collapses to (qnn.human.attack_conditional.collapse_to_effective_events)
    — computed ONCE here (needs ``imp`` + ``tick_hz``) so
    ``attack_hold_texture_arm``'s per-family loop never repeats the O(events)
    collapse; ``None`` when unresolvable (no weapon-identity stream). Used
    ONLY by the hold-texture arm — ``attack_conditional_arm`` still reads
    the raw ``discharge`` lane (out of this fix's scope)."""
    try:
        z = np.load(npz)
    except Exception:
        return None
    if "aim_err_deg" not in z or "op_ready" not in z:
        return None
    imp = None
    if "self_weapon_id" in z:
        imp = z["self_weapon_id"].astype(np.int64)
    elif "weapon_imp" in z:
        imp = z["weapon_imp"].astype(np.int64)
    tick_hz = float(z["tick_hz"][0]) if "tick_hz" in z else 20.0
    discharge = z["discharge"].astype(bool) if "discharge" in z else None
    discharge_collapsed = None
    if discharge is not None and imp is not None:
        discharge_collapsed = attack_ref.collapse_to_effective_events(
            discharge, imp, tick_hz)
    return {
        "err": z["aim_err_deg"].astype(np.float64),
        "ready": z["op_ready"].astype(bool),
        "engaged": (z["engaged"].astype(bool) if "engaged" in z
                    else np.zeros(0, bool)),
        "discharge": discharge,
        "discharge_collapsed": discharge_collapsed,
        "imp": imp,
        "tick_hz": tick_hz,
    }


def _bot_family_curve(streams: dict[str, Any],
                      family_impulses: "tuple[int, ...] | None",
                      ) -> dict[str, np.ndarray] | None:
    """Bot (ready, fire) counts per ``attack_ref.EDGES`` bin for one family
    (``None`` = pooled, no weapon-identity stream required). ``None`` when a
    per-family curve is requested but no weapon-identity stream exists (the
    caller falls back to pooled)."""
    err, ready, disch = streams["err"], streams["ready"], streams["discharge"]
    if disch is None:
        return None
    if family_impulses is None:
        fam_mask = np.ones_like(ready, dtype=bool)
    else:
        if streams["imp"] is None:
            return None
        fam_mask = np.isin(streams["imp"], family_impulses)
    nb = attack_ref.NBINS
    sel = ready & fam_mask & np.isfinite(err)
    if not sel.any():
        return {"ready": np.zeros(nb, np.int64), "fire": np.zeros(nb, np.int64)}
    idx = np.digitize(err[sel], attack_ref.EDGES) - 1
    good = (idx >= 0) & (idx < nb)
    idx = idx[good]
    f = disch[sel][good]
    r = np.bincount(idx, minlength=nb)
    fi = np.bincount(idx[f], minlength=nb) if f.any() else np.zeros(nb, np.int64)
    return {"ready": r, "fire": fi}


def _bot_family_hold(streams: dict[str, Any],
                     family_impulses: "tuple[int, ...] | None",
                     ) -> dict[str, Any] | None:
    """Bot discharge run-length / inter-burst GAP-EXCESS samples (ticks over
    the exact weapon's own effective refire cooldown), engaged-conditioned —
    the hold-texture population, unlike the curve above NOT gated on
    op_ready (matches qnn.human.attack_conditional's own engaged-only
    hold-texture conditioning). Uses the PRE-COLLAPSED discharge stream
    (``_load_attack_streams``'s ``discharge_collapsed`` — fix #2) and the
    truncation-safe ``attack_ref.hold_texture_samples`` extractor (fix #1);
    both new 2026-08-15 (coordinator iteration 3). ``imp`` is required even
    in POOLED mode now (the gap-excess computation needs each tick's exact
    weapon to look up its own cooldown, not just family membership) — a
    wave with no weapon-identity stream is unresolvable for hold-texture
    entirely, pooled included."""
    engaged, disch = streams["engaged"], streams["discharge_collapsed"]
    if disch is None or engaged.size == 0 or streams["imp"] is None:
        return None
    imp = streams["imp"]
    if family_impulses is None:
        fam_mask = np.ones_like(engaged, dtype=bool)
    else:
        fam_mask = np.isin(imp, family_impulses)
    keep = engaged & fam_mask
    n_ticks = int(keep.sum())
    if not n_ticks:
        return {"runs": np.zeros(0, np.int64),
                "gaps": np.zeros(0, np.float64), "n_ticks": 0}
    runs, gap_excess = attack_ref.hold_texture_samples(
        disch, imp, keep, streams["tick_hz"])
    return {"runs": runs, "gaps": gap_excess, "n_ticks": n_ticks}


_ATTACK_PRE_RESPEC_NOTE = (
    "wave predates the ATTACK-GATE RESPEC (no aim_err_deg/op_ready streams "
    "in the npz) — re-run the freeplay leg on the current stream writer; a "
    "pre-respec wave is unscored, never a fail")


def attack_conditional_arm(npz: Path, ctx) -> dict[str, Any]:
    """GATED arm: bot P(discharge | aim-error bin, op-ready) vs the human
    corpus reference's NORMALIZED curve (qnn.human.attack_conditional),
    per family (SG+SSG/NG+SNG/GL/RL/LG) — the conditional-trigger half of
    the ATTACK-GATE RESPEC. Gated on ``attack_ref.curve_deviation`` (MASS-
    WEIGHTED over mutually-populated bins) <= the human reference's own
    BETWEEN-DEMO p95 deviation (v2, 2026-08-15: replaces v1's split-half-of-
    the-pool deviation, an Axiom-3 violation that shrank with corpus size).
    A family under ``ATTACK_ARM_MIN_TICKS`` bot op-ready ticks falls back to
    the pooled comparison; pooled itself too thin, or a comparison spanning
    fewer than ``attack_ref.MIN_BINS_FOR_VERDICT`` populated bins (a curve
    concentrated into 1-2 bins — the alignment-law weapons' steady-tracking
    failure mode, 2026-08-15 bin-collapse fix — cannot characterize a SHAPE
    regardless of tick mass), is UNSCORED (never fails)."""
    streams = _load_attack_streams(npz)
    if streams is None:
        return {"ok": None, "note": _ATTACK_PRE_RESPEC_NOTE}
    human = attack_ref.load_or_build_reference(ctx.corpus_dir)["families"]
    families_out: dict[str, Any] = {}
    for fam, imps in attack_ref.FAMILY_IMPULSES.items():
        bot = _bot_family_curve(streams, imps)
        used_fallback = False
        if bot is None:
            families_out[fam] = {
                "ok": None,
                "note": ("no per-tick weapon-identity stream (self_weapon_id/"
                         "weapon_imp) — family membership unresolvable")}
            continue
        n_ready = int(bot["ready"].sum())
        href = human.get(fam)
        if n_ready < ATTACK_ARM_MIN_TICKS:
            pooled_bot = _bot_family_curve(streams, None)
            if (pooled_bot is None
                    or int(pooled_bot["ready"].sum()) < ATTACK_ARM_MIN_TICKS):
                families_out[fam] = {
                    "ok": None, "n_ready_ticks": n_ready,
                    "note": (f"< {ATTACK_ARM_MIN_TICKS} op-ready ticks in "
                            "family AND pooled — unscored")}
                continue
            bot, used_fallback = pooled_bot, True
            href = human.get(attack_ref.POOLED)
        if not href:
            families_out[fam] = {"ok": None, "n_ready_ticks": n_ready,
                                 "note": "no human reference for this family"}
            continue
        card = attack_ref.score_curve(bot["ready"], bot["fire"], href)
        families_out[fam] = {**card, "n_ready_ticks": n_ready,
                             "used_pooled_fallback": used_fallback}
    scored_ok = [v["ok"] for v in families_out.values() if v.get("ok") is not None]
    return {"ok": (all(scored_ok) if scored_ok else None),
            "n_families_scored": len(scored_ok), "families": families_out}


def attack_hold_texture_arm(npz: Path, ctx) -> dict[str, Any]:
    """GATED arm: bot discharge-run-length / inter-burst-gap p50/p90
    (ticks, engaged-conditioned) vs the human corpus reference, per family —
    the hold-texture half of the ATTACK-GATE RESPEC. A metronomic hold at
    the right marginal rate and a human-shaped burst pattern read
    differently here even when attack_conditional's curve matches. Gated on
    the human reference's BETWEEN-DEMO p95 deviation (v2, 2026-08-15 — see
    ``attack_conditional_arm``'s docstring). Same thin-data / pooled-
    fallback rule as ``attack_conditional_arm``."""
    streams = _load_attack_streams(npz)
    if streams is None:
        return {"ok": None, "note": _ATTACK_PRE_RESPEC_NOTE}
    human = attack_ref.load_or_build_reference(ctx.corpus_dir)["families"]
    families_out: dict[str, Any] = {}
    for fam, imps in attack_ref.FAMILY_IMPULSES.items():
        bot = _bot_family_hold(streams, imps)
        used_fallback = False
        if bot is None:
            families_out[fam] = {
                "ok": None,
                "note": ("no per-tick weapon-identity stream (self_weapon_id/"
                         "weapon_imp) — family membership unresolvable")}
            continue
        href = human.get(fam)
        if bot["n_ticks"] < ATTACK_ARM_MIN_TICKS:
            pooled_bot = _bot_family_hold(streams, None)
            if pooled_bot is None or pooled_bot["n_ticks"] < ATTACK_ARM_MIN_TICKS:
                families_out[fam] = {
                    "ok": None, "n_engaged_ticks": bot["n_ticks"],
                    "note": (f"< {ATTACK_ARM_MIN_TICKS} engaged ticks in "
                            "family AND pooled — unscored")}
                continue
            bot, used_fallback = pooled_bot, True
            href = human.get(attack_ref.POOLED)
        if not href:
            families_out[fam] = {"ok": None, "n_engaged_ticks": bot["n_ticks"],
                                 "note": "no human reference for this family"}
            continue
        card = attack_ref.score_hold(bot["runs"], bot["gaps"], href)
        families_out[fam] = {**card, "n_engaged_ticks": bot["n_ticks"],
                             "used_pooled_fallback": used_fallback}
    scored = [v["ok"] for v in families_out.values() if v.get("ok") is not None]
    return {"ok": (all(scored) if scored else None),
            "n_families_scored": len(scored), "families": families_out}


# ══ 3. style gate (decision 4) ════════════════════════════════════════════════

def style_gate(ctx, npz_path: Path, plans: dict[str, WeaponPlan], *,
               rel_tol: float = 0.25,
               native_npz: Path | None = None,
               selection_target: dict[str, Any] | None = None) -> dict[str, Any]:
    """The free-play DEPLOYMENT report (plan §P3, decision 4; band-v5 axioms).

    GATED arms — world-results invariants only (ALL must be True for PASS;
    an unmeasurable gated arm is ``None`` and cannot PASS — fail loud):
      * ATTACK-CONDITIONAL trigger shape (``attack_conditional_arm``) — bot
        P(discharge | aim-error bin, op-ready) vs the human corpus reference's
        NORMALIZED curve (qnn.human.attack_conditional), per weapon family;
        gated on max per-bin deviation <= the human reference's own
        split-half p95 deviation (ATTACK-GATE RESPEC, Brian 2026-08-15 —
        replaces the demoted ``op_attack`` marginal below).
      * ATTACK HOLD-TEXTURE (``attack_hold_texture_arm``) — bot discharge
        run-length / inter-burst-gap p50/p90 (ticks, engaged-conditioned) vs
        the same human reference, per family; gated the same way. Distinct
        from the curve above: a metronomic hold at the right marginal rate
        and a human-shaped burst pattern can share the same conditional
        curve but not the same hold texture.
      * threat-reactivity hazard — the LIFT criterion the trim converged on,
        re-checked at the deployable operating point.
      * weapon CHOICE — destination probability conditional on the previous
        exact discharge weapon after leaving it, matched against humans;
      * weapon CONTINUATION — per-exact-weapon probability that the next
        stationary-menu discharge uses the same weapon, matched against
        humans.  These two conditional decisions replace the pooled switch
        rate and model-self occupancy, whose aggregate could pass while SG/GL
        under-continued and LG over-continued.

    REPORT-ONLY, EXCLUDED from the gate verdict (ATTACK-GATE RESPEC,
    Brian 2026-08-15): op-attack WORLD-RESULTS ruler — bot operative fires/s
    while engaged vs the bot-mix-weighted human target (``_op_shot_rates`` /
    ``_load_op_attack_targets``; op-filter semantics unchanged). This
    MARGINAL is a skill x style PRODUCT: a model with human-conditional
    trigger behavior (fires more readily exactly when well-aligned) plus
    elite aim availability (well aligned far more often than a human) is
    CORRECTLY expected to exceed the human marginal arithmetically
    (agents/plans/a26-superiority-decomposition.md E12/E13). Still computed
    and reported every fit (the report card is useful), but ``gated: false``
    and never enters ``status``.

    FLAGGED, never gating (doctrine, Brian 2026-07-16: skill placement is
    never capped for style — style spend is the TRAINING-TARGET register):
      * occupancy vs the HUMAN corpus profile — the rc1b anti-camp
        comparison (``occupancy.human_occupancy_report``), in every fit
        report as of 2026-08-09 (it previously only ran by hand);
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
        "contract": ("gated = world invariants ONLY (attack_conditional, "
                     "attack_hold_texture, reactivity, human weapon-choice, "
                     "per-weapon continuation); style spend + band membership are "
                     "FLAGGED/tracked, never gating — skill is never capped "
                     "for style (human-band.md axioms + Brian 2026-07-16); "
                     "op_attack (the marginal) is REPORT-ONLY since "
                     "2026-08-15 — demoted per Brian's ATTACK-GATE RESPEC "
                     "(skill x style product; conditional+hold arms carry "
                     "the humanness content now); per-weapon free-play hbw "
                     "is a REPORT CARD (decision 4 / skill-curves §1.7b)"),
    }

    # ── GATED arm: attack-conditional trigger shape + hold texture ────────
    # (ATTACK-GATE RESPEC, Brian 2026-08-15 — see qnn.human.attack_conditional
    # and the two functions above this one.)
    cond = attack_conditional_arm(npz, ctx)
    hold = attack_hold_texture_arm(npz, ctx)
    report["attack_conditional"] = cond
    report["attack_hold_texture"] = hold

    # ── REPORT-ONLY (excluded from the gate verdict): op-attack marginal ──
    targets = _load_op_attack_targets(ctx.op_attack_path)
    op = _op_shot_rates(npz, targets=targets)
    _op_would_fail = False
    if op is None:
        report["op_attack"] = {
            "ok": None, "gated": False,
            "report_only_since": "2026-08-15",
            "note": ("op-shot ruler unmeasurable (npz unreadable or no engaged "
                     "human-covered weapon mass); marginal = skill x style "
                     "product — demoted per Brian 8/15, conditional+hold arms "
                     "carry the humanness content"),
        }
    else:
        rel = (abs(op["b_att"] - op["h_att"]) / op["h_att"]
               if op["h_att"] else float("inf"))
        opattack_ok = bool(rel <= rel_tol)
        _op_would_fail = not opattack_ok
        report["op_attack"] = {
            "ok": opattack_ok, "gated": False,
            "report_only_since": "2026-08-15",
            "note": ("marginal = skill x style product; demoted per Brian "
                     "8/15 — conditional+hold arms carry the humanness "
                     "content"),
            "bot_fire_per_s": round(op["b_att"], 4),
            "human_fire_per_s": round(op["h_att"], 4),
            "rel_delta": round(rel, 3) if np.isfinite(rel) else None,
            "rel_tol": float(rel_tol),
            "per_family": op.get("per_family", op["per_weapon"]),
            "per_weapon": op["per_weapon"],
            "coverage": op["coverage"],
            "units": op["units"],
        }
    if cond["ok"] is None and hold["ok"] is None and _op_would_fail:
        report["attack_arms_warning"] = (
            "attack arms unscorable on this wave generation — the demoted "
            "op_attack marginal would have FAILED (bot vs human fires/s "
            "while engaged) but neither replacement arm "
            "(attack_conditional/attack_hold_texture) has data on this wave "
            "(pre-respec npz or unresolvable weapon identity); re-run the "
            "freeplay leg on the current stream writer before trusting this "
            "deploy")
        _log(f"WARNING: {report['attack_arms_warning']}")

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

    # ── GATED arms 3/4: the two actual weapon decisions ───────────────────
    href = human_weapon_behavior_reference(ctx.corpus_dir)
    behavior = (weapon_behavior_report(npz, href)
                if href is not None else None)
    if behavior is None:
        choice_ok: bool | None = None
        continue_ok: bool | None = None
        report["weapon_choice"] = {
            "ok": None, "note": "human/bot choice matrix unmeasurable — cannot PASS"}
        report["weapon_continuation"] = {
            "ok": None, "note": "human/bot continuation matrix unmeasurable — cannot PASS"}
    else:
        report["weapon_choice"] = behavior["choice"]
        report["weapon_continuation"] = behavior["continuation"]
        choice_ok = bool(behavior["choice"]["ok"])
        continue_ok = bool(behavior["continuation"]["ok"])

    # Retain the two superseded aggregate rulers as diagnostics so before/after
    # reports show the cancellation explicitly.  Neither enters the verdict.
    report["weapon_pref_switch_aggregate"] = {
        **_weapon_switch_diag(ctx, npz), "gated": False,
        "note": ("diagnostic only — pooled continuation can hide opposite "
                 "per-weapon errors")}
    selection = _selection_profile(npz)
    report["weapon_occupancy_legacy"] = {
        "gated": False,
        "selection_target": selection_target,
        "observed": selection,
        "note": ("legacy model-self active-fire occupancy; superseded by "
                 "human switch-choice + per-weapon continuation")}

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

    # op_attack (opattack_ok) is DELIBERATELY EXCLUDED — report-only since
    # 2026-08-15 (ATTACK-GATE RESPEC); attack_conditional/attack_hold_texture
    # carry the humanness content the marginal used to gate on.
    #
    # attack_conditional_ok / attack_hold_texture_ok get a DIFFERENT
    # unmeasurable rule than the other three gated arms (spec, 2026-08-15):
    # "too-thin data = unscored, never FAIL" / "a pre-respec wave ... does
    # not fail the gate". So None (unscored: no aim_err/op_ready streams, or
    # every family+pooled too thin) does NOT block PASS — only an explicit
    # measured False (a real deviation) does. The pre-existing three keep
    # the original "unmeasurable cannot PASS" contract unchanged.
    gated = {"attack_conditional_ok": cond["ok"],
             "attack_hold_texture_ok": hold["ok"],
             "reactivity_ok": react_ok,
             "weapon_choice_ok": choice_ok,
             "weapon_continuation_ok": continue_ok}
    report["gated"] = gated
    _new_arms_ok = (cond["ok"] is not False and hold["ok"] is not False)
    _rest_ok = all(v is True for v in
                   (react_ok, choice_ok, continue_ok))
    report["status"] = "PASS" if (_new_arms_ok and _rest_ok) else "FAIL"
    if style_flags and report["status"] == "PASS":
        # flags never demote the verdict; they only annotate it
        report["status_note"] = f"PASS with style flags: {style_flags}"
    return report


# ══ 4. attack trim (world-results fire-rate calibration) ══════════════════════

def attack_trim(ctx, config_path: Path,
                launch_eval: Callable[[Path, str], Path | None], *,
                max_iters: int = TRIM_MAX_ITERS) -> dict[str, Any]:
    """Closed-loop final-behavior calibration at the deployable aim point.

    Each estimand has one explicit control:

    * ``attack.fire_bias_vec`` is fixed by the forced-family cadence pins and
      is never rewritten from selection-starved free play;
    * ``move.threat_break_hazard`` fits threat reactivity.

    The human weapon-TRANSITION estimands (``weapon.choice_prob_matrix`` /
    ``weapon.continue_prob_vec``) were removed 2026-08-26 with the decode law
    they fitted. Weapon selection belongs to the network, and fitted preference
    vectors stay rejected (bias-opt conclusion), so threat reactivity is the
    only estimand this trim still owns.

    The old scalar switch-rate + model-self occupancy targets are diagnostics
    only: a28rc1b matched their aggregate while SG/GL and LG were wrong in
    opposite directions.  The two human conditional decisions are iterated
    jointly because preference and continuation affect one another in closed
    loop. Eval launch is injected as
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

    human_weapon = human_weapon_behavior_reference(ctx.corpus_dir)
    if human_weapon is None:
        report.update(
            status="EVAL-FAILED", converged=False,
            note="human weapon choice/continuation reference was unmeasurable")
        return report
    _ht = np.asarray(human_weapon["transitions"])
    report["human_weapon_reference"] = {
        "split": human_weapon.get("split"),
        "n_episodes": human_weapon.get("n_episodes"),
        "continuation": {
            _IMPULSE_NAME[w]: round(float(_ht[w, w]) /
                                    max(float(_ht[w, 1:].sum()), 1.0), 4)
            for w in range(1, 9) if _ht[w, 1:].sum() > 0},
    }
    # working clone at the REQUESTED operating point (aim knobs kept)
    style_cfg = Path(str(config_path) + ".styletrim.json")
    working = json.loads(json.dumps(requested))
    wp = working["params"]
    style_cfg.write_text(json.dumps(working, indent=2) + "\n")

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
        react = _reactivity_hazard(npz)
        behavior = weapon_behavior_report(npz, human_weapon)
        if behavior is None:
            report.update(status="EVAL-FAILED", iterations=iters,
                          note=(f"iteration {it} lacks the weapon preference/"
                                "feasibility transition streams"))
            return report
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
               "att_ratio": round(att_ratio, 3),
               "att_ratio_per_family": {w: round(r, 3) for w, r in w_ratio.items()},
               "aggregate_attack_gate_ok": aggregate_att_ok,
               "reactivity": react,
               "weapon_choice": behavior["choice"],
               "weapon_continuation": behavior["continuation"],
               "att_units": op["units"],
               "att_per_family": op.get("per_family", per_w),
               "att_per_weapon": op["per_weapon"],
               "att_transitions_diag": {"h": round(rc_pair["h_att"], 5),
                                        "b": round(rc_pair["b_att"], 5)}}
        react_ok = True
        r_meas = None
        if react is not None:
            r_meas = max(float(react["hazard_ratio"]), 1e-3)
            _h_lift = HUMAN_REACTIVITY_HAZARD - 1.0
            react_ok = (abs((r_meas - 1.0) - _h_lift)
                        <= TRIM_REACT_LIFT_TOL * _h_lift)
        if react_ok:
            iters.append({**row, "action": "converged"})
            converged = True
            break
        cfg = read_json(style_cfg)
        p = cfg["params"]
        updates: dict[str, Any] = {}
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
             f"choiceΔ {behavior['choice']['max_abs_prob_delta']:.3f} "
             f"continueΔ {behavior['continuation']['max_abs_prob_delta']:.3f} "
             f"react {(react or {}).get('hazard_ratio', '-')}"
             f"/{HUMAN_REACTIVITY_HAZARD} → {updates}")

    # freeze the trimmed style values into the deployable config
    _style = read_json(style_cfg)["params"]
    cfg = read_json(config_path)
    for k in ("attack.fire_bias_vec", "weapon.preference_bias_vec",
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
