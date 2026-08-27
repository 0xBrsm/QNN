"""Round planning — spend the episode budget where it buys decision-relevant
information, not grid coverage.

The rounds (plan §P2, budget box ≤1500 episodes with auto-extend only on CI
rejection — decision 2):

  * ACQ      : face-away tms sweep (the acquisition instrument), 4 diverse
               matchup cells per tms — throughput is target-free and global,
               so the (mw×fw) matrix buys nothing (v1 burned 20 cells/tms).
  * SCREEN   : per weapon, log-spaced gains on ALL FOUR opponent pins (each
               pin is a distinct target-kinematics condition — FrikBot orbits
               at its pinned weapon's sweet-spot range: SG 128u, NG/RL 180u,
               LG 350u — so a curve fit on a pin subset estimates a different
               aggregate than the one the skill claims; the a26 first-fit
               confirmation offset). Rows are mix-weighted to the human
               engagement-range mass, so the fitted shared curve IS the human
               range-mix aggregate over the full condition set. Explicit
               COUPLING cells (gain {0, mid} at tms_ref vs tms*) identify the
               tms coefficient.
  * EXTEND   : after the screening fit — the α ray at each weapon's fitted
               knee, the tremor arm, and refinement gains at each weapon's
               target/knee where the CI is widest. Same four-pin coverage.

There is NO confirmation round: a confirm on the fit's own substrate passes
by construction, and one on a different substrate fails whenever the fit is
unrepresentative — either way it never gated the right thing (Brian,
2026-07-18). Placement is gated on the fit's OWN bootstrap CIs
(gates.placement_gate); per-wave content-derived eval seeds make the fit
multi-seed by construction, and ``--seed-replicates`` re-measures the placed
operating point under fresh seeds (report-only).

Planners return plain cell dicts ``{model_weapon, frikbot_pin, op}`` (op =
all four decode keys); the CLI maps them onto ``instruments.Cell``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qnn.decode_fit.context import (ABBR_TO_MODELNAME, FRIKBOT_TO_PIN,
                                    INSTRUMENT_WEAPONS, MODELNAME_TO_ABBR)

_PIN_TO_FRIKBOT = {v: k for k, v in FRIKBOT_TO_PIN.items()}
# the 4 instrument representatives (SG/SSG and NG/SNG collapse end to end)
_INSTRUMENT_ABBRS = tuple(MODELNAME_TO_ABBR[w] for w in INSTRUMENT_WEAPONS)
# screening gain levels: 0 anchors native; log-spaced span covers every knee
# the v1-era fits ever produced (0.015…0.68) with headroom.
SCREEN_GAINS = (0.0, 0.02, 0.05, 0.12, 0.3, 0.7)
# Knee-undetermined = the curve may still be DESCENDING at the span edge. The
# a27 sweep established that real p60/p90 operating points can require ~1.5;
# a26's old 1.2 ceiling therefore extrapolated a candidate it never measured.
# Keep the full response inside the measured domain, with a broad 5.0 refusal
# frontier. This is the valid a27 fix reconciled into the representative a26
# all-pin estimator.
MAX_GAIN = 5.0
EDGE_GAINS = (0.85, 1.0, 1.2, 1.6, 2.2, 3.2, MAX_GAIN)
COUPLING_GAINS = (0.0, 0.12)         # measured at tms_ref AND tms*
ALPHA_RAY = (0.1, 0.25, 0.5)
TREMOR_ARM = (0.02, 0.05, 0.1)
ACQ_TMS = (0.5, 0.8, 1.1, 1.4, 1.7, 2.0)
# one diverse matchup per pin for the acquisition sweep
ACQ_MATCHUPS = (("lightning", "shotgun"), ("shotgun", "super_nailgun"),
                ("super_nailgun", "rocket_launcher"),
                ("rocket_launcher", "lightning"))

# 4 eps/cell × 4 pins = 16 episodes per (weapon, gain) point — the same
# information mass the old 8 × top-2-pins design spent, redistributed over
# the full condition set.
EPISODES_PER_CELL = 4
ACQ_EPISODES_PER_CELL = 12
# seed-replicate re-measurement of the placed operating point (report-only)
PLACEMENT_EPISODES_PER_CELL = 12
MAX_EPISODES = 1500                  # decision 2; extend only on CI rejection


def _op(gain: float = 0.0, alpha: float = 0.0, tremor: float = 0.0,
        tms: float = 1.0) -> dict[str, float]:
    return {"gain": float(gain), "alpha": float(alpha),
            "tremor": float(tremor), "tms": float(tms)}


def top_pins(pin_weights: dict[str, dict[str, float]], abbr: str,
             n: int = 2) -> list[str]:
    """The n frikbot pins carrying the most human engagement-range mass for
    this weapon (fallback: all four, uniform)."""
    pw = (pin_weights or {}).get(abbr) or {}
    if not pw:
        return list(_PIN_TO_FRIKBOT)[:n] if n < 4 else list(_PIN_TO_FRIKBOT)
    ranked = sorted(pw, key=pw.get, reverse=True)
    return ranked[:n]


@dataclass
class BudgetLedger:
    """Episode accounting against the box. The box constrains the PLANNED
    spend; CI-rejection extensions (decision 2: the only allowed overrun) are
    recorded and totaled but do NOT consume box headroom — otherwise a
    legitimate extension would starve the rounds that follow it (the
    a25rc3d first-flight failure)."""
    max_episodes: int = MAX_EPISODES
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def spent(self) -> int:
        return int(sum(r["episodes"] for r in self.rounds))

    def box_spent(self) -> int:
        return int(sum(r["episodes"] for r in self.rounds
                       if not r["ci_extension"]))

    def charge(self, name: str, n_cells: int, episodes_per_cell: int,
               *, ci_extension: bool = False) -> int:
        eps = int(n_cells * episodes_per_cell)
        if not ci_extension and self.box_spent() + eps > self.max_episodes:
            raise RuntimeError(
                f"budget box: round {name!r} ({eps} eps) would exceed "
                f"{self.max_episodes} (planned spend {self.box_spent()}). Only "
                "CI-rejection extensions may exceed the box (decision 2).")
        self.rounds.append({"round": name, "cells": int(n_cells),
                            "episodes": eps, "ci_extension": bool(ci_extension)})
        return eps

    def as_dict(self) -> dict[str, Any]:
        return {"max_episodes": self.max_episodes, "spent": self.spent(),
                "box_spent": self.box_spent(), "rounds": self.rounds}


def plan_acq_round(tms_values: tuple[float, ...] = ACQ_TMS) -> list[dict]:
    return [{"model_weapon": mw, "frikbot_pin": fw, "op": _op(tms=t)}
            for t in tms_values for mw, fw in ACQ_MATCHUPS]


def plan_screening_round(pin_weights: dict[str, dict[str, float]],
                         tms_star: float, *, tms_ref: float = 1.0,
                         gains: tuple[float, ...] = SCREEN_GAINS) -> list[dict]:
    """Gain screening at the fitted dampener + the coupling cross at tms_ref.
    The coupling cells repeat {0, mid} gains at tms_ref so the likelihood sees
    tms variation and ``c`` is identified."""
    cells: list[dict] = []
    for abbr in _INSTRUMENT_ABBRS:
        mw = ABBR_TO_MODELNAME[abbr]
        pins = [_PIN_TO_FRIKBOT[p] for p in top_pins(pin_weights, abbr, n=4)]
        for fw in pins:
            for g in gains:
                cells.append({"model_weapon": mw, "frikbot_pin": fw,
                              "op": _op(gain=g, tms=tms_star)})
        # coupling cross on the TOP pin only (c is a single coefficient per
        # weapon — one pin's contrast identifies it and it is pin-invariant;
        # box: 416 screen + ~416 extend + 288 acq ≤ 1500, confirm deleted)
        if abs(tms_star - tms_ref) > 1e-6:
            for g in COUPLING_GAINS:
                cells.append({"model_weapon": mw, "frikbot_pin": pins[0],
                              "op": _op(gain=g, tms=tms_ref)})
    return cells


def plan_edge_screen(pin_weights: dict[str, dict[str, float]],
                     tms_star: float) -> list[dict]:
    """EDGE_GAINS cells for every instrument weapon, launched WITH the screen
    (one wall-time round-trip). Historically the CI-rejection extension fired
    on every fit (SG/SSG/NG knees undetermined at the base grid), costing a
    full extra round-trip — pre-spending the edge cells up front determines
    the knees in the screening fit and edge-verifies refusals immediately.
    Charged under the CI allowance at 2× episodes (the same spend the
    extension round made); ``plan_ci_extension`` remains the backstop for the
    rare genuinely-unbracketed knee."""
    cells: list[dict] = []
    for abbr in _INSTRUMENT_ABBRS:
        mw = ABBR_TO_MODELNAME[abbr]
        pins = [_PIN_TO_FRIKBOT[p] for p in top_pins(pin_weights, abbr, n=4)]
        for fw in pins:
            for g in EDGE_GAINS:
                cells.append({"model_weapon": mw, "frikbot_pin": fw,
                              "op": _op(gain=g, tms=tms_star)})
    return cells


def plan_extend_round(gain_fits: dict[str, Any], plans: dict[str, Any],
                      pin_weights: dict[str, dict[str, float]],
                      tms_star: float, *,
                      alphas: tuple[float, ...] = ALPHA_RAY,
                      tremors: tuple[float, ...] = TREMOR_ARM) -> list[dict]:
    """After the screening fit: the α ray at each weapon's fitted knee, the
    tremor arm at gain 0, and refinement gains where the inversion's CI is
    widest (the plan's own gain + the knee point, deduped vs the screen)."""
    cells: list[dict] = []
    for abbr, fit in gain_fits.items():
        mw = ABBR_TO_MODELNAME[abbr]
        pins = [_PIN_TO_FRIKBOT[p] for p in top_pins(pin_weights, abbr, n=4)]
        knee_g = float(np.clip(fit.knee(0.95)[0], 0.005, MAX_GAIN))
        for fw in pins:
            for a in alphas:
                cells.append({"model_weapon": mw, "frikbot_pin": fw,
                              "op": _op(gain=knee_g, alpha=a, tms=tms_star)})
            for t in tremors:
                cells.append({"model_weapon": mw, "frikbot_pin": fw,
                              "op": _op(tremor=t, tms=tms_star)})
            # refinement gains: the resolved plan gain + the ρ=0.9 knee point
            plan = plans.get(abbr) or plans.get(
                next((k for k, v in plans.items()
                      if getattr(v, "alias_of", None) == abbr), ""), None)
            refine = {float(np.clip(fit.knee(0.9)[0], 0.005, MAX_GAIN))}
            if plan is not None and plan.gain > 0:
                refine.add(float(np.clip(plan.gain, 0.005, MAX_GAIN)))
            for g in sorted(refine):
                if all(abs(g - s) / max(s, 1e-9) > 0.15 for s in SCREEN_GAINS if s > 0):
                    cells.append({"model_weapon": mw, "frikbot_pin": fw,
                                  "op": _op(gain=g, tms=tms_star)})
    return cells


def plan_ci_extension(gain_fits: dict[str, Any], plans: dict[str, Any],
                      pin_weights: dict[str, dict[str, float]],
                      tms_star: float) -> list[dict]:
    """The one CI-rejection round for knee-UNDETERMINED weapons. When the
    swept span hasn't reached the legal gain ceiling, the knee is undetermined
    because the sweep never bracketed it — extend with ``EDGE_GAINS`` ABOVE
    the span edge (the a25rc3c SG fix). Only when the span already covers the
    ceiling does midrange refinement (the old behavior) make sense."""
    cells: list[dict] = []
    for abbr, fit in gain_fits.items():
        span_hi = float(fit.swept_gain_span[1])
        past_edge = [g for g in EDGE_GAINS if g > span_hi * 1.05]
        if not past_edge:
            cells += plan_extend_round({abbr: fit},
                                       {k: v for k, v in plans.items()},
                                       pin_weights, tms_star,
                                       alphas=(), tremors=())
            continue
        mw = ABBR_TO_MODELNAME[abbr]
        pins = [_PIN_TO_FRIKBOT[p] for p in top_pins(pin_weights, abbr, n=4)]
        for fw in pins:
            for g in past_edge:
                cells.append({"model_weapon": mw, "frikbot_pin": fw,
                              "op": _op(gain=g, tms=tms_star)})
    return cells


# α×gain interaction guard: a super-band plan whose α evidence was swept at a
# gain more than this relative distance from the RESOLVED plan gain re-sweeps
# the ray there (the LG overshoot class: α fitted at the screening knee,
# applied at the plan gain — the interaction is unmodeled, inversions
# overshoot, and the pullback then costs tremor texture).
ALPHA_REANCHOR_REL = 0.25


def plan_alpha_reanchor(plans: dict[str, Any], table: Any,
                        pin_weights: dict[str, dict[str, float]],
                        tms_star: float) -> tuple[list[dict], list[str]]:
    """``(cells, abbrs)`` — a bracketing α ray {0.5, 1.0, 1.5}×plan.α AT the
    resolved plan gain, top-2 pins, for every super-band plan whose swept α
    rows all sit further than ``ALPHA_REANCHOR_REL`` (relative) from the plan
    gain. Empty when every α plan is already anchored near its gain."""
    cells: list[dict] = []
    abbrs: list[str] = []
    for abbr, p in sorted(plans.items()):
        if p.alias_of or p.band != "super" or p.alpha <= 0 or p.gain <= 0:
            continue
        t = table.where(weapon=abbr)
        m = t["alpha"] > 0
        if m.any():
            g_alpha = np.unique(np.round(t["gain"][m].astype(float), 4))
            if float(np.min(np.abs(g_alpha - p.gain))) \
                    <= ALPHA_REANCHOR_REL * max(p.gain, 1e-6):
                continue                # ray already anchored near the plan gain
        mw = ABBR_TO_MODELNAME[abbr]
        pins = [_PIN_TO_FRIKBOT[q] for q in top_pins(pin_weights, abbr, n=4)]
        for fw in pins:
            for f in (0.5, 1.0, 1.5):
                cells.append({"model_weapon": mw, "frikbot_pin": fw,
                              "op": _op(gain=p.gain, alpha=round(p.alpha * f, 4),
                                        tms=tms_star)})
        abbrs.append(abbr)
    return cells, abbrs


def plan_placement_round(plans: dict[str, Any], tms_star: float
                         ) -> list[dict]:
    """The placed operating point per instrument weapon, all four pins —
    used ONLY by the report-only ``--seed-replicates`` re-measurement (the
    gating confirmation round is gone: placement is gated on the fit's own
    CIs). SSG/NG ride the identical SG/SNG family plan."""
    cells: list[dict] = []
    for abbr in _INSTRUMENT_ABBRS:
        p = plans.get(abbr)
        if p is None:
            continue
        mw = ABBR_TO_MODELNAME[abbr]
        for fw in _PIN_TO_FRIKBOT.values():
            cells.append({"model_weapon": mw, "frikbot_pin": fw,
                          "op": _op(gain=p.gain, alpha=p.alpha,
                                    tremor=p.tremor, tms=tms_star)})
    return cells
