"""The v2 decode-fit orchestrator.

    python -m qnn.decode_fit --run-dir runs/head_probe/<run> \
        [--skill native|p90|SG=p90,RL=p75,...] [--write] [--validate]
        [--eval-template <dir>] [--version aXXrcN] [--offline-retrofit]

Flow (plan §P1-P3; every closed-loop artifact manifest-cached under
``runs/decode_fit/<model>/``):

  0. context + human baselines (collect-cached, qnn.human)
  1. invariant pins (qnn.bc.decode_fit.fit — fire-rate first, baked into
     every wave substrate; invariants-before-skills)
  2. ACQ round     → turn_mag_scale fit (extend the sweep when target is
     sweep-bound; NO CLAMP)
  3. SCREEN round  → per-weapon gain responses (+ measured tms coupling)
  4. EXTEND round  → α rays at fitted knees, tremor arm, refinement gains
     (+ one CI-rejection extension when a decision quantity is undetermined)
  5. plan          → refusal/frontier semantics (decision 1)
  6. CONFIRM round → CI-overlap gate at percentile ±5 (decision 3), one
     damped secant correction on a miss, re-confirm once
  7. attack trim + free-play style gate (per-weapon free-play hbw is a
     report card — decision 4)
  8. promote the staged config ONLY on gate PASS (no emit-despite-FAIL),
     sidecar + fit report

``--offline-retrofit`` fits from the legacy v1 grid JSONs (no evals) and
emits the report only — the Phase-1 acceptance path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qnn.decode_fit import design, events, human_refs, response
from qnn.decode_fit.context import (INTERCEPT_WEAPONS, WEAPON_IMPULSE,
                                    FitContext, read_json, rel_to_repo,
                                    resolve_fit_context)

_log = lambda m: print(f"[decode-fit] {m}", flush=True)  # noqa: E731

TMS_EXTEND_STEP = 0.3
TMS_RANGE = (0.1, 3.0)
ACQ_TARGET_PCT = 50.0


def parse_skill_vector(spec: str) -> dict[str, float] | str:
    """``native`` | ``p90`` (uniform) | ``SG=p90,RL=p75,...`` (per-weapon)."""
    spec = str(spec).strip()
    if spec.lower() == "native":
        return "native"

    def _pct(s: str) -> float:
        v = float(str(s).strip().lower().lstrip("p"))
        if 0.0 <= v <= 1.0:
            v *= 100.0
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"percentile out of range: {s!r}")
        return v

    if "=" not in spec:
        return {w: _pct(spec) for w in INTERCEPT_WEAPONS}
    out: dict[str, float] = {}
    for tok in spec.split(","):
        k, _, v = tok.partition("=")
        if k.strip() not in WEAPON_IMPULSE:
            raise ValueError(f"unknown weapon in --skill spec: {k!r}")
        out[k.strip()] = _pct(v)
    return out


# ── stage 1: invariant pins (fire-rate before any skill sweep) ───────────────

def move_commit_pins(ctx: FitContext) -> dict[str, Any]:
    """The move-commit dur_tilt calibration (teacher-forced CPU fit, cached by
    the fitter under the run's decode_fit dir keyed on run_dir). Emitted as
    ``move.commit_dur_tilt`` — its absence shipped [0,0] in the first a25rc3d
    flight and contributed to the MOVE band miss."""
    cache = ctx.out_dir / "move_commit_calibration.json"
    if cache.exists():
        doc = read_json(cache)
        if str(doc.get("run_dir")) == str(rel_to_repo(ctx.run_dir)) \
                and doc.get("dur_tilt"):
            return {"move.commit_dur_tilt": [float(x) for x in doc["dur_tilt"]]}
        _log(f"WARNING: {cache.name} keyed to {doc.get('run_dir')!r}, not this "
             "run — ignoring")
    from qnn.eval.move_commit_fit import fit_dur_tilt
    _log("move-commit dur_tilt cache miss — running the teacher-forced fit "
         "(~10-15 min)")
    doc = fit_dur_tilt(ctx.run_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(doc, indent=1) + "\n")
    return {"move.commit_dur_tilt": [float(x) for x in doc["dur_tilt"]]}


def invariant_pins(ctx: FitContext) -> dict[str, Any]:
    """qnn.bc.decode_fit.fit's pin dict, manifest-cached. The attack.* subset
    is baked into every wave substrate so the closed-loop discharge sample is
    fired at a human-calibrated rate (invariants-before-skills, d46104d7)."""
    key = ctx.content_key(stage="pins")
    cached = ctx.manifest_get("pins", key)
    if cached is not None:
        return read_json(cached)
    from qnn.bc import decode_fit as bc_fit
    template = Path(__file__).resolve().parents[2] / "qnn" / "model" / "bench" \
        / "templates" / "decode.a25base.json"
    res = bc_fit.fit(ctx.run_dir, template)
    fit = res.get("fit") if isinstance(res, dict) else None
    if not fit:
        _log("WARNING: pin fit unavailable — waves run on the bare-head rate "
             "(under-fired weapons will rest on a thin sample)")
        return {}
    out = ctx.out_dir / "_pins_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fit, indent=2) + "\n")
    ctx.manifest_put("pins", out, key)
    return fit


def _wave_substrate(ctx: FitContext, pins: dict[str, Any]) -> dict[str, Any]:
    """The shared wave substrate: a25 base template, aim levers OFF (per-lane
    overrides carry the full operating point), fire-rate invariant baked."""
    template = Path(__file__).resolve().parents[2] / "qnn" / "model" / "bench" \
        / "templates" / "decode.a25base.json"
    base = read_json(template)
    base.setdefault("params", {})
    base["params"]["look.aim_prior_gain"] = 0.0
    base["params"]["look.aim_mag_gain"] = 0.0
    base["params"].update({k: v for k, v in pins.items() if k.startswith("attack.")})
    base["version"] = f"decodefit-v2-substrate-{ctx.model_id[:24]}"
    return base


# ── stage 2: acquisition (tms) ────────────────────────────────────────────────

def fit_tms(ctx: FitContext, substrate: dict, ledger: design.BudgetLedger,
            *, target_pct: float = ACQ_TARGET_PCT) -> dict[str, Any]:
    from qnn.decode_fit import instruments
    band = human_refs.acquisition_band(ctx.acq_path)
    tms_values = list(design.ACQ_TMS)
    charged: set[float] = set()
    for _round in range(4):          # bracket-extension loop (NO CLAMP)
        # charge only the NEW tms levels — resume-skip reuses done waves
        new = [v for v in tms_values if v not in charged]
        ledger.charge(f"acq[{len(tms_values)}tms]",
                      len(design.plan_acq_round(tuple(new))),
                      design.ACQ_EPISODES_PER_CELL,
                      ci_extension=_round > 0)
        charged.update(new)
        dirs = instruments.run_acq_waves(
            ctx, tms_values, substrate,
            episodes_per_cell=design.ACQ_EPISODES_PER_CELL)
        rows = instruments.collect_acq_throughput(ctx, dirs)
        fit = response.fit_acquisition(rows, band, target_pct=target_pct)
        if fit["unfittable"]:
            raise RuntimeError(
                f"acquisition NOT fittable: tms↔throughput corr "
                f"{fit['tms_throughput_corr']} — the instrument produced no "
                "target-free flicks (face-away spawn missing?)")
        if fit["sweep_floor_bound"] and min(tms_values) > TMS_RANGE[0] + 1e-9:
            lo = max(TMS_RANGE[0], min(tms_values) - TMS_EXTEND_STEP)
            _log(f"acq sweep floor-bound; extending down to {lo}")
            tms_values = sorted({lo, *tms_values})
            continue
        if fit["sweep_ceil_bound"] and max(tms_values) < TMS_RANGE[1] - 1e-9:
            hi = min(TMS_RANGE[1], max(tms_values) + TMS_EXTEND_STEP)
            _log(f"acq sweep ceil-bound; extending up to {hi}")
            tms_values = sorted({*tms_values, hi})
            continue
        return fit
    return fit


# ── stages 3-5: intercept rounds + fits + plan ───────────────────────────────

def _fit_all_gains(table: events.EventTable, pinw: dict, tms_ref: float
                   ) -> dict[str, response.GainResponse]:
    fits = {}
    for abbr in ("SG", "NG", "RL", "LG"):
        fits[abbr] = response.fit_gain_response(
            table, abbr, pin_weights=pinw.get(abbr), tms_ref=tms_ref)
    return fits


def run_intercept_rounds(ctx: FitContext, substrate: dict, tms_star: float,
                         targets: dict[str, float],
                         ledger: design.BudgetLedger,
                         alpha_cap: float | None = None,
                         ) -> dict[str, Any]:
    from qnn.decode_fit import instruments
    ladder = human_refs.perweapon_human_ladder(ctx.intercept_path)
    reach = human_refs.reachable_band(ctx.intercept_path)
    pinw = human_refs.range_pin_weights(ctx.range_path)

    def _cells(plan_rows: list[dict]) -> list:
        return [instruments.Cell(model_weapon=c["model_weapon"],
                                 frikbot_pin=c["frikbot_pin"], op=c["op"])
                for c in plan_rows]

    # SCREEN + pre-spent edge gains, ONE pool (the CI-rejection extension
    # fired on every fit — folding the EDGE_GAINS cells into the screen
    # round-trip determines the knees in the screening fit; the extension
    # stays as backstop for a genuinely-unbracketed knee)
    screen = design.plan_screening_round(pinw, tms_star)
    edge = design.plan_edge_screen(pinw, tms_star)
    ledger.charge("screen", len(screen), design.EPISODES_PER_CELL)
    ledger.charge("edge-screen", len(edge), design.EPISODES_PER_CELL * 2,
                  ci_extension=True)
    grouped = instruments.run_botpin_wave_groups(
        ctx, [{"cells": _cells(screen),
               "episodes_per_cell": design.EPISODES_PER_CELL, "tag": "screen"},
              {"cells": _cells(edge),
               "episodes_per_cell": design.EPISODES_PER_CELL * 2,
               "tag": "edgescreen"}],
        substrate)
    dirs = grouped["screen"] + grouped["edgescreen"]
    table = instruments.collect_events(dirs)
    gain_fits = _fit_all_gains(table, pinw, tms_ref=1.0)
    plans0 = response.build_plan(gain_fits, {}, None, ladder, reach, targets,
                                 tms=tms_star)

    # EXTEND (α rays at fitted knees + tremor + refinement)
    extend = design.plan_extend_round(gain_fits, plans0, pinw, tms_star)
    ledger.charge("extend", len(extend), design.EPISODES_PER_CELL)
    dirs += instruments.run_botpin_waves(
        ctx, _cells(extend), substrate,
        episodes_per_cell=design.EPISODES_PER_CELL, tag="extend")
    table = instruments.collect_events(dirs)
    gain_fits = _fit_all_gains(table, pinw, tms_ref=1.0)
    alpha_fits = {a: response.fit_alpha_response(
        table, a, pinned_gain=gain_fits[a].knee(0.95)[0],
        pin_weights=pinw.get(a)) for a in gain_fits}
    tremor_fit = response.fit_tremor_response(table)

    # ONE CI-rejection extension: weapons whose knee stayed undetermined get
    # gains PAST the swept span edge (or midrange refinement only when the
    # span already covers the legal ceiling — design.plan_ci_extension;
    # decision 2 allows exceeding the box here, flagged in the ledger).
    # REFUSALS MUST BE EDGE-VERIFIED (the SG plateau-then-drop bimodality):
    # a plateau-shaped response gives the exponential a confident wrong basin
    # (small k, tight CI, "determined") whose floor forces refusal — no
    # weapon may refuse inside an UNSWEPT span, so a refused plan joins the
    # edge round even with a determined knee.
    _edge_max = max(design.EDGE_GAINS)
    undet = [a for a, f in gain_fits.items()
             if f.knee_undetermined
             or (getattr(plans0.get(a), "refused", False)
                 and f.swept_gain_span[1] < _edge_max)]
    if undet:
        _log(f"CI rejection: knee undetermined / unverified refusal for "
             f"{undet} — one extension round")
        extra = design.plan_ci_extension(
            {a: gain_fits[a] for a in undet},
            {a: p for a, p in plans0.items() if a in undet},
            pinw, tms_star)
        if extra:
            ledger.charge("ci-extension", len(extra),
                          design.EPISODES_PER_CELL * 2, ci_extension=True)
            dirs += instruments.run_botpin_waves(
                ctx, _cells(extra), substrate,
                episodes_per_cell=design.EPISODES_PER_CELL * 2, tag="ciext")
            table = instruments.collect_events(dirs)
            gain_fits = _fit_all_gains(table, pinw, tms_ref=1.0)
            # the knee may have moved past the old α anchor — re-pin the rays
            alpha_fits = {a: response.fit_alpha_response(
                table, a, pinned_gain=gain_fits[a].knee(0.95)[0],
                pin_weights=pinw.get(a)) for a in gain_fits}

    plans = response.build_plan(gain_fits, alpha_fits, tremor_fit, ladder,
                                reach, targets, tms=tms_star,
                                alpha_style_cap=alpha_cap)
    # measurements outrank curves at the PLAN stage too: a refusal cannot
    # stand on a parametric floor that well-measured cells beat (the SG
    # plateau-basin class) — re-anchor such plans on the best measured cell.
    plans = response.apply_measured_frontier(plans, table, gain_fits, ladder,
                                             pinw)

    # α×gain interaction guard (the LG overshoot class): super-band plans
    # whose α evidence sits far from the resolved gain re-sweep the ray THERE,
    # refit α on rows AT that gain only, and rebuild.
    re_cells, re_abbrs = design.plan_alpha_reanchor(plans, table, pinw, tms_star)
    if re_cells:
        _log(f"α re-anchor for {re_abbrs}: ray re-swept at the resolved gain")
        ledger.charge("alpha-reanchor", len(re_cells), design.EPISODES_PER_CELL)
        dirs += instruments.run_botpin_waves(
            ctx, _cells(re_cells), substrate,
            episodes_per_cell=design.EPISODES_PER_CELL, tag="alphaanchor")
        table = instruments.collect_events(dirs)
        for abbr in re_abbrs:
            g_star = plans[abbr].gain
            near = table.filter(
                (np.abs(table["gain"] - g_star) < 1e-6)
                & (table["weapon"] == abbr))
            afit = response.fit_alpha_response(
                near, abbr, pinned_gain=g_star, pin_weights=pinw.get(abbr))
            if afit is not None:
                alpha_fits[abbr] = afit
        plans = response.build_plan(gain_fits, alpha_fits, tremor_fit, ladder,
                                    reach, targets, tms=tms_star,
                                    alpha_style_cap=alpha_cap)
        plans = response.apply_measured_frontier(plans, table, gain_fits,
                                                 ladder, pinw)
    return {"table": table, "gain_fits": gain_fits, "alpha_fits": alpha_fits,
            "tremor_fit": tremor_fit, "plans": plans, "ladder": ladder,
            "reachable": reach, "pin_weights": pinw, "wave_dirs": dirs,
            "alpha_cap": alpha_cap}


def run_seed_replicates(ctx: FitContext, substrate: dict, tms_star: float,
                        plans: dict[str, Any], state: dict[str, Any],
                        ledger: design.BudgetLedger, *, n: int
                        ) -> dict[str, Any]:
    """CHEAP-TIER multi-seed rigor (Brian 2026-07-17): re-measure the
    CONFIRMED placement under ``n`` fresh base seeds and report the spread —
    never gating. The fit itself is deterministic (content-keyed waves,
    derived seeds), so its OWN reruns cannot expose sampling sensitivity;
    each replicate here re-runs the confirmation cells with a distinct seed
    salt (``seed_extra`` → +r·104729 on every wave's eval seed) and scores
    them against the promoted plans on the confirmation instrument. All
    replicates ride ONE worker pool (one wall-time round-trip)."""
    from qnn.decode_fit import gates, instruments
    cells_raw = design.plan_confirmation_round(plans, tms_star,
                                               state["pin_weights"])
    cells = [instruments.Cell(model_weapon=c["model_weapon"],
                              frikbot_pin=c["frikbot_pin"], op=c["op"])
             for c in cells_raw]
    ledger.charge(f"seed-replicates x{n}", len(cells) * n,
                  design.CONFIRM_EPISODES_PER_CELL, ci_extension=True)
    grouped = instruments.run_botpin_wave_groups(
        ctx, [{"cells": cells,
               "episodes_per_cell": design.CONFIRM_EPISODES_PER_CELL,
               "tag": f"seedrep{r}", "seed_extra": r}
              for r in range(1, n + 1)],
        substrate)
    rows: list[dict[str, Any]] = []
    for r in range(1, n + 1):
        table = instruments.collect_events(grouped[f"seedrep{r}"])
        g = gates.confirmation_gate(table, plans, state["ladder"])
        rows.append({"replicate": r, "status": g["status"],
                     "weapons": {w: {"measured_hbw_median":
                                     c.get("measured_hbw_median"),
                                     "measured_pct": c.get("measured_pct"),
                                     "verdict": c.get("verdict")}
                                 for w, c in (g.get("weapons") or {}).items()}})
    spread: dict[str, dict[str, float]] = {}
    for w in plans:
        vals = [row["weapons"][w]["measured_hbw_median"] for row in rows
                if row["weapons"].get(w, {}).get("measured_hbw_median")
                is not None]
        if vals:
            spread[w] = {"min": round(min(vals), 4), "max": round(max(vals), 4),
                         "n": len(vals)}
    statuses = [row["status"] for row in rows]
    _log(f"seed replicates ({n}): {statuses} — per-weapon hbw spread "
         + " ".join(f"{w}[{s['min']}..{s['max']}]"
                    for w, s in sorted(spread.items())))
    return {"n": n, "seed_salt_stride": 104729, "rows": rows,
            "hbw_spread": spread,
            "note": ("report-only sampling-robustness replicates of the "
                     "confirmation instrument — never gating")}


# ── stage 6: confirmation (+ one secant correction) ──────────────────────────

def run_confirmation(ctx: FitContext, substrate: dict, tms_star: float,
                     state: dict[str, Any], ledger: design.BudgetLedger
                     ) -> dict[str, Any]:
    from qnn.decode_fit import gates, instruments
    plans = state["plans"]
    prev_plans = prev_gate = None
    for attempt in (1, 2, 3):
        cells_raw = design.plan_confirmation_round(plans, tms_star,
                                                   state["pin_weights"])
        cells = [instruments.Cell(model_weapon=c["model_weapon"],
                                  frikbot_pin=c["frikbot_pin"], op=c["op"])
                 for c in cells_raw]
        ledger.charge(f"confirm#{attempt}", len(cells),
                      design.CONFIRM_EPISODES_PER_CELL,
                      ci_extension=attempt > 1)
        dirs = instruments.run_botpin_waves(
            ctx, cells, substrate,
            episodes_per_cell=design.CONFIRM_EPISODES_PER_CELL,
            tag=f"confirm{attempt}")
        table = instruments.collect_events(dirs)
        gate = gates.confirmation_gate(table, plans, state["ladder"])
        if gate["status"] == "PASS" or attempt == 3:
            return {"gate": gate, "plans": plans, "attempt": attempt}
        if attempt == 1:
            corrected = gates.secant_correction(
                plans, gate, state["gain_fits"], state["alpha_fits"],
                tremor_fit=state["tremor_fit"],
                support_elite=human_refs.support_elite_bounds(
                    ctx.intercept_path))
            note = "one damped secant correction"
        else:
            # third attempt only off two MEASURED points on the same ray —
            # unchanged weapons' waves are content-keyed cache hits, so the
            # extra round costs only the failed weapons' cells
            corrected = gates.empirical_correction(
                prev_plans, prev_gate, plans, gate,
                tremor_fit=state["tremor_fit"])
            note = "one empirical (measured-ray) correction"
        if corrected is None:
            return {"gate": gate, "plans": plans, "attempt": attempt}
        _log(f"confirmation miss — applying {note}")
        prev_plans, prev_gate = plans, gate
        plans = corrected
    return {"gate": gate, "plans": plans, "attempt": attempt}


# ── main ─────────────────────────────────────────────────────────────────────

def _anchor_stamp(ctx: FitContext) -> dict[str, Any]:
    """Compact report stamp of the collect's placement anchors (skill-curves
    §16.3): version + per-weapon values, selected depths, and the loud flags
    (unvalidated / family_borrowed / shrunk) a report reader must see before
    trusting any band-coordinate number in this fit."""
    node = human_refs.placement_anchors(ctx.intercept_path)
    keep = ("elite_hbw", "floor_hbw", "elite_depth", "floor_depth",
            "elite_validated", "floor_validated", "reliability_sb",
            "half_log_r", "unvalidated", "family_borrowed", "shrunk")
    return {"anchors_version": node.get("anchors_version"),
            "weapons": {w: {k: e.get(k) for k in keep}
                        for w, e in (node.get("weapons") or {}).items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.decode_fit", description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--skill", default="native")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="run the free-play attack-trim + style gate "
                         "(needs --eval-template)")
    ap.add_argument("--eval-template", type=Path, default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--acq-target", default=f"p{ACQ_TARGET_PCT:.0f}")
    ap.add_argument("--budget", type=int, default=design.MAX_EPISODES)
    ap.add_argument("--offline-retrofit", action="store_true",
                    help="fit from legacy v1 grid JSONs; report only, no evals")
    ap.add_argument("--alpha-cap", type=float, default=None,
                    help="optional hard cap on look.aim_mag_gain (ablations "
                         "only). Default UNCAPPED (Brian 2026-07-16): skill "
                         "placement is never capped for style — α's style "
                         "spend (hold destruction) is measured by the "
                         "fitted-vs-native flag and becomes a training "
                         "target, not a skill ceiling.")
    ap.add_argument("--seed-replicates", type=int, default=0,
                    help="after a PASSing confirmation, re-measure the "
                         "confirmed placement under N fresh base seeds "
                         "(report-only sampling-robustness check; one "
                         "worker-pool round-trip)")
    ap.add_argument("--force", action="store_true",
                    help="promote the config even on gate FAIL (debug only; "
                         "stamped into provenance)")
    args = ap.parse_args(argv)

    ctx = resolve_fit_context(args.run_dir)
    from qnn.human import ensure_from_collect
    ensure_from_collect(ctx.corpus_dir)
    _log(f"run={ctx.model_id} ckpt={ctx.checkpoint.name} skill={args.skill!r}")

    if args.offline_retrofit:
        return _offline_retrofit(ctx, args)

    from qnn.decode_fit import emit, gates, instruments
    ledger = design.BudgetLedger(max_episodes=args.budget)
    pins = invariant_pins(ctx)
    substrate = _wave_substrate(ctx, pins)

    acq_pct = parse_skill_vector(args.acq_target)
    acq_pct = ACQ_TARGET_PCT if isinstance(acq_pct, str) else \
        float(np.mean(list(acq_pct.values())))
    tms_fit = fit_tms(ctx, substrate, ledger, target_pct=acq_pct)
    tms_star = float(tms_fit["turn_mag_scale"])
    _log(f"turn_mag_scale = {tms_star} (ci {tms_fit['tms_ci']}, "
         f"acq p{tms_fit['achieved_acq_percentile']}, in_band={tms_fit['in_band']})")

    sk = parse_skill_vector(args.skill)
    alpha_cap = args.alpha_cap if args.alpha_cap and args.alpha_cap > 0 else None
    # native targets are only known after the fits — run the rounds with a
    # placeholder, then re-target at the measured native placements.
    targets0 = sk if isinstance(sk, dict) else \
        {w: 50.0 for w in INTERCEPT_WEAPONS}
    state = run_intercept_rounds(ctx, substrate, tms_star, targets0, ledger,
                                 alpha_cap=alpha_cap)
    if not isinstance(sk, dict):    # --skill native
        ladder = state["ladder"]
        natives = {}
        for w in INTERCEPT_WEAPONS:
            src = w if w in state["gain_fits"] else \
                {"SSG": "SG", "SNG": "NG"}.get(w, w)
            natives[w] = round(human_refs.hbw_to_pct(
                state["gain_fits"][src].native_at(tms_star), ladder[w]), 1)
        state["plans"] = response.build_plan(
            state["gain_fits"], state["alpha_fits"], state["tremor_fit"],
            state["ladder"], state["reachable"], natives, tms=tms_star,
            alpha_style_cap=alpha_cap)

    for w, p in state["plans"].items():
        _log(f"  {w}: band={p.band} gain={p.gain} α={p.alpha} pred={p.pred_hbw} "
             f"→ p{p.achieved_pct}{' REFUSED→frontier p%.1f' % p.frontier_pct if p.refused else ''}")

    conf = run_confirmation(ctx, substrate, tms_star, state, ledger)
    plans = conf["plans"]
    gate = conf["gate"]
    _log(f"confirmation: {gate['status']} (attempt {conf['attempt']})")

    seed_reps: dict[str, Any] | None = None
    if args.seed_replicates > 0 and gate["status"] == "PASS":
        seed_reps = run_seed_replicates(ctx, substrate, tms_star, plans,
                                        state, ledger,
                                        n=int(args.seed_replicates))

    result: dict[str, Any] = {
        "skill": args.skill,
        "pct_basis": ("human sustained-band coordinate: log-hbw-linear "
                      "between the collect's validated placement anchors "
                      "(floor = p0, elite = p100; frozen selection procedure, "
                      "skill-curves §16.3), per weapon — NOT a population "
                      "percentile and NOT comparable across corpora"),
        "placement_anchors": _anchor_stamp(ctx),
        "tms_fit": tms_fit, "budget": ledger.as_dict(),
        "confirmation": gate,
        **({"seed_replicates": seed_reps} if seed_reps else {}),
        "plans": {w: vars(p) for w, p in plans.items()},
        "provenance": ctx.provenance(),
    }
    if not args.write:
        _report(ctx, result, "REPORT-ONLY")
        return 0

    version = args.version or f"{ctx.model_id}-v2-{str(args.skill).replace('=', '').replace(',', '_')}"
    vectors = response.build_vectors(plans)
    emit_pins = {**pins, **move_commit_pins(ctx)}   # + move-commit dur_tilt
    staged = emit.stage_decode_config(ctx, plans, vectors, emit_pins, tms_star,
                                      version)
    result["staged"] = rel_to_repo(staged)
    # verify-first trim: seed the style knobs from the previous promoted fit
    # so a rate-stable checkpoint converges at iteration 0 (one eval)
    _warm = emit.warm_start_style(ctx, staged)
    if _warm:
        result["trim_warm_start"] = rel_to_repo(_warm)

    style: dict[str, Any] = {"status": "SKIPPED"}
    if args.validate and args.eval_template is not None:
        trim = gates.attack_trim(
            ctx, staged,
            lambda cfg, tag: instruments.run_freeplay(
                ctx, cfg, args.eval_template, tag=tag))
        result["attack_trim"] = trim
        npz = trim.get("npz")            # trim's final-eval npz, reused (dedup)
        if npz:
            _native = trim.get("native_npz")
            style = gates.style_gate(
                ctx, Path(npz), plans,
                native_npz=Path(_native) if _native else None)
    result["style_gate"] = style

    passed = gate["status"] == "PASS" and style.get("status") in ("PASS", "SKIPPED")
    _flags = (style.get("style_spend") or {}).get("flags") or []
    try:
        promoted = emit.promote_decode_config(staged, gate_passed=passed,
                                              force=args.force,
                                              style_flags=_flags)
        result["config"] = rel_to_repo(promoted)
        label = "PASS" if passed else "FORCED"
        if style.get("status") == "SKIPPED":
            label = "PASS-PENDING-VALIDATION" if passed else label
    except RuntimeError as e:
        result["promote_refused"] = str(e)
        label = "FAIL"
    _report(ctx, result, label)
    return 0 if label != "FAIL" else 1


def _offline_retrofit(ctx: FitContext, args: argparse.Namespace) -> int:
    """Phase-1 path: fit from the legacy v1 grids, report only."""
    head_probe = Path("runs/head_probe")
    tables = []
    for stem in ("_aim_grid_lead", "_aim_alpha_grid", "_aim_grid_tremor"):
        p = head_probe / f"{stem}_{ctx.model_id}.json"
        if p.exists():
            tables.append(events.legacy_grid_table(p))
    if not tables:
        raise FileNotFoundError(f"no legacy grids for {ctx.model_id}")
    table = events.EventTable.concat(tables)
    ladder = human_refs.perweapon_human_ladder(ctx.intercept_path)
    reach = human_refs.reachable_band(ctx.intercept_path)
    pinw = human_refs.range_pin_weights(ctx.range_path)
    lead = read_json(head_probe / f"_aim_grid_lead_{ctx.model_id}.json")
    sub = lead.get("substrate_decode")
    tms = float(read_json(Path(sub))["params"].get("look.turn_mag_scale", 1.0)) \
        if sub and Path(sub).exists() else 1.0
    gain_fits = _fit_all_gains(table, pinw, tms_ref=tms)
    sk = parse_skill_vector(args.skill)
    targets = sk if isinstance(sk, dict) else {
        w: round(human_refs.hbw_to_pct(
            gain_fits.get(w, gain_fits.get({"SSG": "SG", "SNG": "NG"}.get(w, w))).native,
            ladder[w]), 1) for w in INTERCEPT_WEAPONS}
    plans = response.build_plan(gain_fits, {}, None, ladder, reach, targets, tms=tms)
    result = {
        "mode": "offline-retrofit", "skill": args.skill, "tms": tms,
        "fits": {w: {"floor": f.floor, "floor_ci": f.floor_ci(),
                     "native": f.native, "k": f.k, "knee": f.knee(),
                     "knee_undetermined": f.knee_undetermined,
                     "monotone_p": f.monotone_p, "n_events": f.n_events}
                 for w, f in gain_fits.items()},
        "plans": {w: vars(p) for w, p in plans.items()},
        "provenance": ctx.provenance(),
    }
    _report(ctx, result, "RETROFIT")
    return 0


def _report(ctx: FitContext, result: dict[str, Any], label: str) -> None:
    result["result"] = label
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.out_dir / "decode_fit_v2_report.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    _log(f"wrote {rel_to_repo(out)} → {label}")


if __name__ == "__main__":
    raise SystemExit(main())
