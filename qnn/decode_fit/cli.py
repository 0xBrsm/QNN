"""The v2 decode-fit orchestrator.

    python -m qnn.decode_fit --run-dir runs/head_probe/<run> \
        [--skill native|p90|SG=p90,RL=p75,...] [--write] [--validate]
        [--eval-template <dir>] [--offline-retrofit]
    python -m qnn.decode_fit --run-dir runs/head_probe/<run> \
        --assign-rc aXXrcNx [--config <promoted.json>] [--replace]
    python -m qnn.decode_fit --run-dir runs/head_probe/<run> \
        --trim-config <existing decode config> --eval-template <dir>
        (re-run the free-play attack-trim + style gate on an EXISTING config
        at its own substrate — a copy is staged/trimmed, the input is never
        mutated; use case: a deployed config was hand-edited post-promotion
        and its selection layer is off the trim's converged equilibrium)

Fits stage + promote under a PROVISIONAL version derived from the run id —
rc names are EARNED, never pre-assigned (src/docs/model-versioning.md): the
rc numeral by the passed fit, the deploy letter by the deploy slot. After a
PASS, ``--assign-rc`` marks the promoted config with its rc name (a copy
with ``version`` re-stamped + assignment provenance); deploy consumes the
rc-named file.

Flow (plan §P1-P3; every closed-loop artifact lives under
``runs/decode_fit/<model>/`` — waves resume-skip off their content-hashed
done dirs + the substrate/env staleness check):

  0. context + human baselines (collect-cached, qnn.human)
  1. ACQ round     → turn_mag_scale fit on the NATIVE substrate (throughput
     is target-free — no discharge in the ruler; extend the sweep when
     target is sweep-bound; NO CLAMP)
  2. LIVE PINS round 0 (qnn.decode_fit.live_pins) → conditional family
     cadence at tms*, with SG and SNG as the class representatives; the fitted
     fire-only attack.fire_bias_vec is baked into every later wave substrate
     (invariants-before-skills, now measured in-regime — the deleted
     offline corpus-forward pins mis-signed across the offline↔live gap)
  3. SCREEN round  → per-weapon gain responses (+ measured tms coupling)
  4. EXTEND round  → α rays at fitted knees, tremor arm, refinement gains
     (+ one CI-rejection extension when a decision quantity is undetermined)
  5. plan          → refusal/frontier semantics (decision 1)
  6. PLACEMENT gate → the fit adjudicates itself on its OWN bootstrap CIs
     at percentile ±5 (no confirmation round — a confirm on the fit's
     substrate passes by construction, one on a different substrate fails
     whenever the fit is unrepresentative; representativeness is designed
     in: all four opponent pins, human range-mix weighting, per-wave
     content-derived seeds). Degenerate CI / undetermined knee / thin
     mass FAIL hard. ``--seed-replicates`` re-measures the placed point
     under fresh seeds (report-only).
  7. attack trim + free-play style gate (per-weapon free-play hbw is a
     report card — decision 4)
  8. promote the staged config ONLY on gate PASS (no emit-despite-FAIL),
     sidecar + fit report

``--offline-retrofit`` fits from the legacy v1 grid JSONs (no evals) and
emits the report only — the Phase-1 acceptance path.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qnn.decode_fit import design, events, human_refs, live_pins, response
from qnn.decode_fit.context import (INSTRUMENT_WEAPONS, INTERCEPT_WEAPONS,
                                    MODELNAME_TO_ABBR, TRANSFER_ALIAS,
                                    WEAPON_IMPULSE,
                                    FitContext, read_json, rel_to_repo,
                                    calibration_members, resolve_fit_context)

_log = lambda m: print(f"[decode-fit] {m}", flush=True)  # noqa: E731

TMS_EXTEND_STEP = 0.3
TMS_RANGE = (0.1, 3.0)
ACQ_TARGET_PCT = 50.0
# The transition gate adjudicates eight previous-weapon rows separately. The
# historical 32-episode free-play sample could expose fewer than 20 leave
# events for Axe/NG and turn a correct conditional law into LOW_N. Four arena
# batches retain enough tail support without changing any forced-pin budget.
TRANSITION_TRIM_EPISODES = 128


def parse_skill_vector(spec: str) -> dict[str, float] | str:
    """``native`` | ``p90`` | explicit family coordinates.

    SG/SSG and NG/SNG are indivisible fit coordinates. Naming either member
    expands to both; contradictory coordinates for a family fail loud.
    """
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
        weapon, pct = k.strip(), _pct(v)
        for member in calibration_members(weapon):
            if member in out and abs(out[member] - pct) > 1e-9:
                family = "+".join(calibration_members(weapon))
                raise ValueError(
                    f"conflicting --skill coordinates for {family}: "
                    f"p{out[member]:g} vs p{pct:g}")
            out[member] = pct
    return out


# ── non-lever pins (emit-time calibrations that are not wave levers) ─────────

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
    doc = fit_dur_tilt(ctx.run_dir, cache_dir=ctx.corpus_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(doc, indent=1) + "\n")
    return {"move.commit_dur_tilt": [float(x) for x in doc["dur_tilt"]]}


def _wave_substrate(ctx: FitContext) -> dict[str, Any]:
    """The shared wave substrate: a25 base template, aim levers OFF (per-lane
    overrides carry the full operating point), attack operating point NATIVE —
    the template's zero legacy/joint and explicit fire/preference vectors
    stand until the live-pins round bakes its fitted bias vector (no offline
    pins: qnn.decode_fit.live_pins)."""
    template = Path(__file__).resolve().parents[2] / "qnn" / "model" / "bench" \
        / "templates" / "decode.base.json"
    base = read_json(template)
    base.setdefault("params", {})
    base["params"]["look.aim_prior_gain"] = 0.0
    base["params"]["look.aim_mag_gain"] = 0.0
    base["version"] = f"decodefit-v2-substrate-{ctx.model_id[:24]}"
    return base


# ── stage 1: acquisition (tms) ────────────────────────────────────────────────

def fit_tms(ctx: FitContext, substrate: dict, ledger: design.BudgetLedger,
            *, target_pct: float = ACQ_TARGET_PCT) -> dict[str, Any]:
    from qnn.decode_fit import instruments
    band = human_refs.acquisition_band(ctx.acq_path)
    tms_values = list(design.ACQ_TMS)
    charged: set[float] = set()
    rep_dirs: list[Path] = []        # one seed-replicate extension, ever
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
            episodes_per_cell=design.ACQ_EPISODES_PER_CELL) + rep_dirs
        rows = instruments.collect_acq_throughput(ctx, dirs)
        fit = response.fit_acquisition(rows, band, target_pct=target_pct)
        if fit["unfittable"] and fit["acq_marginal"] and not rep_dirs:
            # marginal ≠ dead: the lever moved the axis but not decisively.
            # Spend ONE seed-replicate extension (never a guess) and refit.
            _log(f"acq corr marginal ({fit['tms_throughput_corr']}, raw "
                 f"{fit['tms_throughput_corr_raw']}) — one seed-replicate "
                 "extension")
            ledger.charge(f"acq-seedrep[{len(tms_values)}tms]",
                          len(design.plan_acq_round(tuple(tms_values))),
                          design.ACQ_EPISODES_PER_CELL, ci_extension=True)
            rep_dirs = instruments.run_acq_waves(
                ctx, tms_values, substrate,
                episodes_per_cell=design.ACQ_EPISODES_PER_CELL,
                tag="acqr1", seed_extra=1)
            rows = instruments.collect_acq_throughput(ctx, dirs + rep_dirs)
            fit = response.fit_acquisition(rows, band, target_pct=target_pct)
        if fit["unfittable"]:
            raise RuntimeError(
                "acquisition NOT fittable: settled-weighted tms↔throughput "
                f"corr {fit['tms_throughput_corr']} (raw "
                f"{fit['tms_throughput_corr_raw']}"
                + (", after a seed-replicate extension" if rep_dirs else "")
                + ") — the lever does not move the axis "
                "(face-away spawn missing?)")
        floor_dead = min(tms_values) in {
            d["turn_mag_scale"] for d in fit.get("dropped_tms", [])}
        if fit["sweep_floor_bound"] and min(tms_values) > TMS_RANGE[0] + 1e-9:
            if floor_dead:
                # the current floor already settled-collapsed (response.
                # ACQ_SETTLED_COLLAPSE_FRAC) — slower turning only completes
                # FEWER acquisitions, never more, so a level below it cannot
                # be informative either. Walking on toward TMS_RANGE[0]
                # would only spend episodes on more dead cells; accept the
                # reliable floor instead (fit["reliable_tms_range"]).
                _log(f"acq sweep floor-bound at tms={min(tms_values)}, but "
                     "that level is already settled-collapsed — not "
                     "extending further into the dead regime")
            else:
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
    for abbr in (MODELNAME_TO_ABBR[w] for w in INSTRUMENT_WEAPONS):
        fits[abbr] = response.fit_gain_response(
            table, abbr, pin_weights=pinw.get(abbr), tms_ref=tms_ref)
    return fits


def run_intercept_rounds(ctx: FitContext, substrate: dict, tms_star: float,
                         targets: dict[str, float],
                         ledger: design.BudgetLedger,
                         alpha_cap: float | None = None,
                         no_demote: bool = False,
                         ) -> dict[str, Any]:
    from qnn.decode_fit import instruments
    # THE aim ladder rides the window-sampled TRACKING statistic (trigger-
    # free; decode-fit-v2 addendum 7/18) — at-discharge intercept remains the
    # report card + crest-capture reference, never the placement target.
    ladder = human_refs.perweapon_human_ladder(ctx.tracking_path)
    reach = human_refs.reachable_band(ctx.tracking_path)
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
    table = instruments.collect_tracking(dirs)
    gain_fits = _fit_all_gains(table, pinw, tms_ref=1.0)
    plans0 = response.build_plan(gain_fits, {}, None, ladder, reach, targets,
                                 tms=tms_star)

    # EXTEND (α rays at fitted knees + tremor + refinement)
    extend = design.plan_extend_round(gain_fits, plans0, pinw, tms_star)
    ledger.charge("extend", len(extend), design.EPISODES_PER_CELL)
    dirs += instruments.run_botpin_waves(
        ctx, _cells(extend), substrate,
        episodes_per_cell=design.EPISODES_PER_CELL, tag="extend")
    table = instruments.collect_tracking(dirs)
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
            table = instruments.collect_tracking(dirs)
            gain_fits = _fit_all_gains(table, pinw, tms_ref=1.0)
            # the knee may have moved past the old α anchor — re-pin the rays
            alpha_fits = {a: response.fit_alpha_response(
                table, a, pinned_gain=gain_fits[a].knee(0.95)[0],
                pin_weights=pinw.get(a)) for a in gain_fits}

    plans = response.build_plan(gain_fits, alpha_fits, tremor_fit, ladder,
                                reach, targets, tms=tms_star,
                                alpha_style_cap=alpha_cap,
                                no_demote=no_demote)
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
        table = instruments.collect_tracking(dirs)
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
                                    alpha_style_cap=alpha_cap,
                                    no_demote=no_demote)
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
    """Report-only multi-seed re-measurement of the PLACED operating point
    (Brian 2026-07-17/18): the fit is multi-seed by construction (per-wave
    content-derived eval seeds), so these replicates are the out-of-sample
    spread check — each re-runs the placement cells with a distinct seed
    salt (``seed_extra`` → +r·104729 on every wave's eval seed) and scores
    them against the plans' promises. Never gating. All replicates ride ONE
    worker pool (one wall-time round-trip)."""
    from qnn.decode_fit import gates, instruments
    cells_raw = design.plan_placement_round(plans, tms_star)
    cells = [instruments.Cell(model_weapon=c["model_weapon"],
                              frikbot_pin=c["frikbot_pin"], op=c["op"])
             for c in cells_raw]
    ledger.charge(f"seed-replicates x{n}", len(cells) * n,
                  design.PLACEMENT_EPISODES_PER_CELL, ci_extension=True)
    grouped = instruments.run_botpin_wave_groups(
        ctx, [{"cells": cells,
               "episodes_per_cell": design.PLACEMENT_EPISODES_PER_CELL,
               "tag": f"seedrep{r}", "seed_extra": r}
              for r in range(1, n + 1)],
        substrate)
    rows: list[dict[str, Any]] = []
    for r in range(1, n + 1):
        table = instruments.collect_tracking(grouped[f"seedrep{r}"])
        g = gates.measure_placement(table, plans, state["ladder"])
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
                     "placement instrument — never gating")}


# ── main ─────────────────────────────────────────────────────────────────────

def _anchor_stamp(ctx: FitContext) -> dict[str, Any]:
    """Compact report stamp of the collect's placement anchors (skill-curves
    §16.3): version + per-weapon values, selected depths, and the loud flags
    (unvalidated / family_borrowed / shrunk) a report reader must see before
    trusting any band-coordinate number in this fit."""
    node = human_refs.placement_anchors(ctx.tracking_path)
    keep = ("elite_hbw", "floor_hbw", "elite_depth", "floor_depth",
            "elite_validated", "floor_validated", "reliability_sb",
            "half_log_r", "unvalidated", "family_borrowed", "shrunk")
    return {"anchors_version": node.get("anchors_version"),
            "weapons": {w: {k: e.get(k) for k in keep}
                        for w, e in (node.get("weapons") or {}).items()}}


_REPLAN_MANIFEST_KEYS = ("model_id", "checkpoint_sha256", "corpus_dir",
                        "corpus_fingerprint", "look_grid_sha256")


def _instrument_contract() -> dict[str, Any]:
    """Fingerprint of the wave/scoring contract a ``--replan-from`` sidecar
    must still match: the frozen aim-statistic geometry + look-applier regime
    baked into every wave's env signature (``instruments._env_sig``) plus the
    scored-table ``events.SCORER_VERSION`` — a code change to either
    invalidates a saved fit exactly like a live wave's substrate/env
    staleness check (``instruments.build_wave_dir``)."""
    from qnn.decode_fit import events, instruments
    sig = instruments._env_sig(0, 0, 1)
    sig.pop("episodes_per_cell", None)   # a replan-policy knob, not a lever
    sig["scorer_version"] = events.SCORER_VERSION
    return sig


def _replan_manifest(ctx: FitContext) -> dict[str, Any]:
    prov = ctx.provenance()
    return {**{k: prov.get(k) for k in _REPLAN_MANIFEST_KEYS},
            "instrument_contract": _instrument_contract()}


def _guard_replan(ctx: FitContext, saved_manifest: dict[str, Any]) -> None:
    """Refuse ``--replan-from`` when the saved fit-state was not fit against
    THIS checkpoint/corpus/look-grid/instrument contract — replan must never
    cross a model or instrument change (a stale sidecar's fitted responses
    describe a different physics/geometry)."""
    current = _replan_manifest(ctx)
    mismatched = [k for k in current if current[k] != saved_manifest.get(k)]
    if mismatched:
        detail = "; ".join(
            f"{k}: saved={saved_manifest.get(k)!r} current={current[k]!r}"
            for k in mismatched)
        raise RuntimeError(
            "--replan-from refused: saved fit-state manifest does not match "
            f"this run — {detail}. Replan never crosses a model, corpus, "
            "look-grid, or instrument change; run the full fit instead.")


def _resolve_fit_state_path(p: Path) -> Path:
    """``--replan-from`` accepts the sidecar itself, its owning run dir
    (``runs/decode_fit/<model>/``), or that run's ``decode_fit_v2_report.json``
    (the report does not itself carry bootstrap draws — see the module
    docstring in response.py's fit-state section — so this always resolves to
    the ``.npz`` sidecar written alongside it)."""
    p = Path(p)
    if p.is_dir():
        return p / response.FIT_STATE_NPZ
    if p.suffix == ".json":
        return p.parent / response.FIT_STATE_NPZ
    return p


def _assign_rc_cmd(args) -> int:
    """``--assign-rc``: mark a gate-promoted provisional config with its
    earned rc name (emit.assign_rc) and exit. Source = ``--config``, else the
    run's newest promoted config that has no rc assignment yet."""
    from qnn.decode_fit import emit
    from qnn.decode_fit.context import _REPO
    src = args.config
    if src is None:
        out_dir = _REPO / "runs" / "decode_fit" / Path(args.run_dir).name
        cands = []
        for p in sorted(out_dir.glob("decode.*.json")):
            if p.name.endswith(".staged.json"):
                continue
            prov = (read_json(p).get("provenance") or {})
            if "promoted_utc" in prov and not prov.get("staged", True) \
                    and not prov.get("rc_assigned") \
                    and "rc_assigned_utc" not in prov:
                cands.append(p)
        if not cands:
            _log(f"no unassigned promoted decode config under "
                 f"{rel_to_repo(out_dir)} — a gate PASS promotes first, "
                 f"then earns its rc name")
            return 1
        src = max(cands, key=lambda p: p.stat().st_mtime)
    try:
        target = emit.assign_rc(src, args.assign_rc, replace=args.replace,
                                force=args.force)
    except (ValueError, RuntimeError) as e:
        _log(f"assign-rc refused: {e}")
        return 1
    _log(f"deploy consumes {rel_to_repo(target)} (deploy letter = this "
         f"slot; it only advances once this config actually ships)")
    return 0


def _load_plans_for_trim(ctx, cfg: dict[str, Any]
                         ) -> tuple[dict[str, "response.WeaponPlan"] | None, str]:
    """Best-effort ``WeaponPlan`` reconstruction for style_gate's plan-keyed
    report-card sections (``freeplay_intercept``/``aggregate_intercept``) on
    a ``--trim-config`` invocation — which runs no fitting stage of its own,
    so there is no live ``plans`` dict to hand it. Never blocks the trim:
    style_gate's gated arms never touch ``plans`` (only report cards do), so
    the worst case is an honestly-unscored report, not a refusal.

    Preference order:
      1. this model's ``decode_fit_v2_report.json`` ``plans`` block — EXACT:
         it is ``vars(p) for w, p in plans.items()`` (this module's
         ``_report``) and already self-carries its ``weapon`` key, so
         ``WeaponPlan(**d)`` losslessly reconstructs it, including post-build
         overrides a bare refit would miss (measured-frontier reanchor, α
         reanchor — response.apply_measured_frontier / design.
         plan_alpha_reanchor). Used only
         when its provenance checkpoint matches THIS run (never a stale or
         foreign fit); a skill-target mismatch against the trim-config's own
         ``skill_vector`` is logged loud but does not block.
      2. the fit-state sidecar (``response.FIT_STATE_NPZ``), rebuilt via
         ``response.build_plan`` at the trim-config's own
         ``skill_vector.targets`` — coarser (no measured-frontier / α-reanchor
         post-processing) but still a real fit.
      3. ``None`` — the plan-keyed report cards go UNSCORED.
    """
    report_path = ctx.out_dir / "decode_fit_v2_report.json"
    if report_path.exists():
        doc = read_json(report_path)
        raw = doc.get("plans")
        prov = doc.get("provenance") or {}
        if raw and prov.get("checkpoint_sha256") == ctx.checkpoint_sha:
            try:
                # report plans are ``vars(p)`` (this module's ``_report``) —
                # already carry their own ``weapon`` key, unlike a config's
                # ``skill_vector.placed`` table (emit._placed_table, keyed by
                # weapon but not self-carrying it).
                plans = {w: response.WeaponPlan(**d) for w, d in raw.items()}
            except TypeError as e:
                _log(f"decode_fit_v2_report.json plans block malformed "
                     f"({e}) — falling back to the fit-state sidecar")
            else:
                cfg_targets = (cfg.get("skill_vector") or {}).get("targets") or {}
                if cfg_targets and any(
                        abs(float(cfg_targets.get(w, p.target_pct))
                            - p.target_pct) > 1e-6
                        for w, p in plans.items()):
                    _log("WARNING: decode_fit_v2_report.json plans targets "
                         "differ from this config's skill_vector.targets — "
                         "using the report's plans anyway (report cards "
                         "only, never gates)")
                return plans, f"reconstructed from {rel_to_repo(report_path)}"
        elif raw:
            _log(f"decode_fit_v2_report.json provenance checkpoint does not "
                 f"match this run — not this run's fit; falling back to the "
                 f"fit-state sidecar")

    sidecar = ctx.out_dir / response.FIT_STATE_NPZ
    if sidecar.exists():
        try:
            state = response.load_fit_state(sidecar)
            _guard_replan(ctx, state["manifest"])
        except Exception as e:
            _log(f"fit-state sidecar unusable for retrim plans: {e}")
        else:
            targets = (cfg.get("skill_vector") or {}).get("targets")
            if targets:
                plans = response.build_plan(
                    state["gain_fits"], state["alpha_fits"],
                    state["tremor_fit"], state["ladder"], state["reachable"],
                    targets, tms=state["tms_star"])
                return plans, (
                    f"rebuilt from fit-state sidecar {rel_to_repo(sidecar)} "
                    "at this config's skill_vector.targets (no "
                    "measured-frontier / α-reanchor post-processing)")
            _log("fit-state sidecar present but the trim-config has no "
                 "skill_vector.targets to rebuild plans from")

    return None, ("no usable plans source (no checkpoint-matching "
                  "decode_fit_v2_report.json, no fit-state sidecar) — "
                  "style_gate's plan-keyed report cards are UNSCORED")


def _trim_config_cmd(ctx: FitContext, args) -> int:
    """``--trim-config``: re-run the free-play attack-trim + style gate on an
    EXISTING decode config at its own substrate, without re-fitting anything.

    Use case: a deployed config was hand-edited post-promotion (guard/idle/
    jump keys added) and its conditional weapon transition layer is off the
    trim's converged equilibrium for the edited substrate — decode/guards
    don't transfer across checkpoints, but a
    style re-trim on the SAME checkpoint's own substrate is exactly what
    ``gates.attack_trim`` already does; this mode just points it at an
    already-promoted config instead of a freshly-staged one.

    ``attack_trim`` WRITES its converged style values back into the config
    path it is handed, so this operates on a COPY, never the input file:
    ``decode.<version>.retrim.staged.json`` under ``ctx.out_dir``, ``version``
    re-stamped ``f"{orig_version}-retrim"`` with a provenance note recording
    the source path, invocation UTC, and reason. No promote/assign step here
    — rc letters are earned separately via ``--assign-rc`` once a human has
    looked at the retrim report."""
    from qnn.decode_fit import gates, instruments
    from qnn.model.decode_config import resolve_decode_config

    src = Path(args.trim_config)
    if not src.is_file():
        raise FileNotFoundError(f"--trim-config: no such file: {src}")
    cfg = read_json(src)

    # decode fits (and their guard/style layers) never transfer across
    # checkpoints — refuse loudly rather than re-trim a config fit against a
    # different model (feedback_decode_guards_dont_transfer_across_arch.md).
    cfg_sha = (cfg.get("provenance") or {}).get("checkpoint_sha256")
    if cfg_sha != ctx.checkpoint_sha:
        raise RuntimeError(
            f"--trim-config refused: {rel_to_repo(src)} provenance."
            f"checkpoint_sha256={cfg_sha!r} does not match --run-dir "
            f"{rel_to_repo(ctx.run_dir)}'s checkpoint "
            f"({ctx.checkpoint_sha!r}, {rel_to_repo(ctx.checkpoint)}) — "
            "decode fits never transfer across checkpoints; point "
            "--run-dir at the run that actually produced this config")
    # attack_trim's own INVALID-SEMANTICS check covers split_v1 / zero legacy
    # attack.bias_vec — no need to duplicate it here (gates.py:1358-1366).

    orig_version = cfg.get("version")
    if not orig_version:
        raise RuntimeError(f"{rel_to_repo(src)} has no 'version' field — "
                            "cannot stage a retrim")
    retrim_version = f"{orig_version}-retrim"
    staged_cfg = json.loads(json.dumps(cfg))     # deep copy — NEVER mutate src
    staged_cfg["version"] = retrim_version
    staged_cfg.setdefault("provenance", {})["retrim"] = {
        "source_path": rel_to_repo(src),
        "invocation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": "post-promotion substrate edit re-trim",
    }
    staged_path = ctx.out_dir / f"decode.{orig_version}.retrim.staged.json"
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    staged_path.write_text(json.dumps(staged_cfg, indent=2) + "\n")
    try:
        resolve_decode_config(staged_path)
    except Exception as e:
        staged_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"staged retrim copy failed resolve_decode_config and was "
            f"removed ({staged_path.name}): {e!r}") from e
    _log(f"retrim staged copy ← {rel_to_repo(src)} → "
         f"{rel_to_repo(staged_path)} (version={retrim_version})")

    trim = gates.attack_trim(
        ctx, staged_path,
        lambda cfg_path, tag: instruments.run_freeplay(
            ctx, cfg_path, args.eval_template, tag=tag,
            episodes=TRANSITION_TRIM_EPISODES))
    _log(f"attack trim: {trim.get('status')}")

    plans, plans_note = _load_plans_for_trim(ctx, cfg)
    _log(f"style-gate plans source: {plans_note}")
    style: dict[str, Any]
    npz = trim.get("npz")
    if npz is None:
        style = {"status": "SKIPPED",
                 "note": "attack_trim produced no scoreable npz "
                         f"(trim status {trim.get('status')})"}
    elif plans is None:
        style = {"status": "UNSCORED", "note": plans_note}
    else:
        native_npz = trim.get("native_npz")
        style = gates.style_gate(
            ctx, Path(npz), plans,
            native_npz=Path(native_npz) if native_npz else None,
            selection_target=trim.get("selection_target"))
    _log(f"style gate: {style.get('status')}")

    trim_status = trim.get("status")
    result = {
        "mode": "trim-config",
        "source_config": rel_to_repo(src),
        "staged_path": rel_to_repo(staged_path),
        "plans_source": plans_note,
        "attack_trim": trim,
        "style_gate": style,
        "status": trim_status,
    }
    out = ctx.out_dir / f"retrim_{orig_version}_report.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    _log(f"wrote {rel_to_repo(out)} — attack_trim={trim_status} "
         f"style_gate={style.get('status')}")
    return 0 if trim_status == "CONVERGED" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.decode_fit", description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--skill", default="native")
    ap.add_argument("--no-demote", action="store_true",
                    help="pXX targets are FLOORS: a weapon whose post-fix "
                         "NATIVE landing already meets/beats its target is "
                         "placed at native (no tremor demotion). The "
                         "demotion arm stays available without the flag.")
    ap.add_argument("--replan-from", type=Path, default=None, metavar="PATH",
                    help="skip ACQ/live-pins/collection/fitting entirely: "
                         "load a previous run's fitted responses + ladder/"
                         "reachable state from its _fit_state.npz sidecar "
                         "(PATH = the sidecar itself, its run dir, or its "
                         "decode_fit_v2_report.json), rebuild plans fresh "
                         "against THIS invocation's --skill/--no-demote, run "
                         "the placement gate, then proceed straight to trim/"
                         "validate as normal (those stay live evals). "
                         "Refuses on any checkpoint/corpus/look-grid/"
                         "instrument mismatch — replan never crosses a model "
                         "or instrument change. The sidecar is written "
                         "unconditionally at the end of every normal fit.")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="run the free-play attack-trim + style gate "
                         "(needs --eval-template)")
    ap.add_argument("--eval-template", type=Path, default=None)
    ap.add_argument("--pins", choices=("fit", "report"), default="fit",
                    help="live-pins stage mode: 'fit' = damped-secant cadence "
                         "repair (default); 'report' = round-0 observation "
                         "only, fire_bias_vec stays ZERO — for checkpoints "
                         "whose cadence is model-owned (crest-line occupancy "
                         "regulator; agents/plans/decode-fit-pins-crest-"
                         "mismatch.md option A). Cadence stays gated by the "
                         "stage-5 free-play arms.")
    ap.add_argument("--version", default=None,
                    help="override the staged config's PROVISIONAL version "
                         "(ablation arms etc.). rc / bare-tier names are "
                         "refused — they are earned post-pass via "
                         "--assign-rc (src/docs/model-versioning.md)")
    ap.add_argument("--assign-rc", default=None, metavar="aXXrcNx",
                    help="assign an earned rc name to a gate-promoted "
                         "config and exit (no fitting). Source = --config, "
                         "or the run's latest unassigned promoted config")
    ap.add_argument("--config", type=Path, default=None,
                    help="explicit promoted decode config for --assign-rc")
    ap.add_argument("--replace", action="store_true",
                    help="with --assign-rc: overwrite an existing "
                         "decode.<rc>.json — ONLY for re-emitting a "
                         "superseded NEVER-deployed artifact")
    ap.add_argument("--trim-config", type=Path, default=None, metavar="PATH",
                    help="re-run the free-play attack-trim + style gate on "
                         "an EXISTING decode config at its own substrate "
                         "(post-promotion hand-edit case: guard/idle/jump "
                         "keys added and the conditional weapon transition "
                         "layer is off the trim's converged equilibrium). PATH "
                         "is copied to decode.<version>.retrim.staged.json "
                         "under this run's decode_fit dir and re-trimmed "
                         "there — the input file is NEVER mutated. Requires "
                         "--eval-template. No fitting, no promote/assign; "
                         "exit 0 only on attack_trim CONVERGED")
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
                    help="after a PASSing placement gate, re-measure the "
                         "placed operating point under N fresh base seeds "
                         "(report-only sampling-robustness check; one "
                         "worker-pool round-trip)")
    ap.add_argument("--force", action="store_true",
                    help="promote the config even on gate FAIL (debug only; "
                         "stamped into provenance)")
    args = ap.parse_args(argv)

    if args.trim_config is not None:
        if args.eval_template is None:
            ap.error("--trim-config requires --eval-template: the free-play "
                     "attack-trim launches evals against a template run-dir")
        if args.write or args.offline_retrofit or args.replan_from \
                or args.assign_rc:
            ap.error("--trim-config is mutually exclusive with --write, "
                     "--offline-retrofit, --replan-from, and --assign-rc — "
                     "it re-trims an EXISTING config at its own substrate, "
                     "never the fit/emit/assign pipeline")

    from qnn.decode_fit import emit
    if args.assign_rc:
        return _assign_rc_cmd(args)
    if args.write and not args.force and (not args.validate or args.eval_template is None):
        ap.error(
            "--write requires --validate and --eval-template: a decode config "
            "cannot be promoted as playable without measuring its final-aim "
            "fire rate, weapon occupancy/switching, and reactivity. Use --force "
            "only for a stamped debug artifact.")
    if args.version and emit.RESERVED_VERSION_RE.fullmatch(args.version):
        ap.error(
            f"--version {args.version}: rc / bare-tier names are EARNED, "
            f"never pre-assigned — the fit stages under a provisional "
            f"version; on gate PASS assign the name with --assign-rc "
            f"(src/docs/model-versioning.md)")

    if args.offline_retrofit and args.replan_from:
        ap.error("--offline-retrofit and --replan-from are mutually "
                 "exclusive fitting paths")

    ctx = resolve_fit_context(args.run_dir)
    from qnn.human import ensure_from_collect
    ensure_from_collect(ctx.corpus_dir)
    _log(f"run={ctx.model_id} ckpt={ctx.checkpoint.name} skill={args.skill!r}")

    if args.trim_config is not None:
        return _trim_config_cmd(ctx, args)

    if args.offline_retrofit:
        return _offline_retrofit(ctx, args)

    from qnn.decode_fit import gates, instruments
    no_demote = bool(getattr(args, "no_demote", False))
    alpha_cap = args.alpha_cap if args.alpha_cap and args.alpha_cap > 0 else None
    sk = parse_skill_vector(args.skill)
    targets0 = sk if isinstance(sk, dict) else \
        {w: 50.0 for w in INTERCEPT_WEAPONS}

    if args.replan_from is not None:
        # ── replan-only: skip ACQ/live-pins/collection/fitting entirely ────
        sidecar = _resolve_fit_state_path(args.replan_from)
        if not sidecar.exists():
            ap.error(f"--replan-from: no fit-state sidecar at {sidecar} — "
                     "it is written at the end of every normal (non-replan) "
                     "fit; run one first")
        replan = response.load_fit_state(sidecar)
        _guard_replan(ctx, replan["manifest"])
        tms_star = replan["tms_star"]
        tms_fit = replan["tms_fit"]
        live = replan["live"]
        ledger = design.BudgetLedger(max_episodes=args.budget)
        substrate = _wave_substrate(ctx)
        substrate = {**substrate,
                     "params": {**substrate["params"],
                                "attack.fire_bias_vec": live["fire_bias_vec"]}}
        _log(f"--replan-from {rel_to_repo(sidecar)}: manifest verified — "
             f"skipping collection/fit (tms*={tms_star}); rebuilding plans "
             f"for skill={args.skill!r} no_demote={no_demote}")
        plans = response.build_plan(
            replan["gain_fits"], replan["alpha_fits"], replan["tremor_fit"],
            replan["ladder"], replan["reachable"], targets0, tms=tms_star,
            alpha_style_cap=alpha_cap, no_demote=no_demote)
        state: dict[str, Any] = {
            "table": None, "gain_fits": replan["gain_fits"],
            "alpha_fits": replan["alpha_fits"],
            "tremor_fit": replan["tremor_fit"], "plans": plans,
            "ladder": replan["ladder"], "reachable": replan["reachable"],
            "pin_weights": {}, "wave_dirs": [], "alpha_cap": alpha_cap}
        if not isinstance(sk, dict):    # --skill native
            ladder = state["ladder"]
            natives = {}
            for w in INTERCEPT_WEAPONS:
                src = w if w in state["gain_fits"] else TRANSFER_ALIAS.get(w, w)
                natives[w] = round(human_refs.hbw_to_pct(
                    state["gain_fits"][src].native_at(tms_star), ladder[w]), 1)
            state["plans"] = response.build_plan(
                state["gain_fits"], state["alpha_fits"], state["tremor_fit"],
                state["ladder"], state["reachable"], natives, tms=tms_star,
                alpha_style_cap=alpha_cap)
    else:
        ledger = design.BudgetLedger(max_episodes=args.budget)
        substrate = _wave_substrate(ctx)

        # ACQ first, on the NATIVE substrate: acquisition throughput is
        # target-free (flick/settle geometry off the obs streams — no
        # discharge in the ruler), so the attack operating point cannot
        # starve it.
        acq_pct = parse_skill_vector(args.acq_target)
        acq_pct = ACQ_TARGET_PCT if isinstance(acq_pct, str) else \
            float(np.mean(list(acq_pct.values())))
        tms_fit = fit_tms(ctx, substrate, ledger, target_pct=acq_pct)
        tms_star = float(tms_fit["turn_mag_scale"])
        _log(f"turn_mag_scale = {tms_star} (ci {tms_fit['tms_ci']}, "
             f"acq p{tms_fit['achieved_acq_percentile']}, in_band={tms_fit['in_band']})")

        # LIVE PINS round 0 at the fitted dampener — the exact regime SCREEN
        # runs at — before any gain arm is swept (invariants-before-skills,
        # d46104d7; the invariant is now MEASURED in-regime instead of
        # corpus-forward).
        live = live_pins.fit_live_pins(ctx, substrate, tms_star, ledger,
                                       mode=args.pins)
        _log(f"live pins ({live['status']}): fire_bias_vec="
             + str(live["fire_bias_vec"]) + "  " + " ".join(
            f"{w}[{v['native_rate_per_s']:.2f}"
            + (f"→{v['fitted_rate_per_s']:.2f}@{v['bias']:+.2f}"
               f"/{v['secant_steps']}s]" if v["secant_steps"] else "]")
            for w, v in sorted(live["weapons"].items())))
        substrate = {**substrate,
                     "params": {**substrate["params"],
                                "attack.fire_bias_vec": live["fire_bias_vec"]}}

        # native targets are only known after the fits — run the rounds with
        # a placeholder, then re-target at the measured native placements.
        state = run_intercept_rounds(ctx, substrate, tms_star, targets0, ledger,
                                     alpha_cap=alpha_cap, no_demote=no_demote)
        if not isinstance(sk, dict):    # --skill native
            ladder = state["ladder"]
            natives = {}
            for w in INTERCEPT_WEAPONS:
                src = w if w in state["gain_fits"] else TRANSFER_ALIAS.get(w, w)
                natives[w] = round(human_refs.hbw_to_pct(
                    state["gain_fits"][src].native_at(tms_star), ladder[w]), 1)
            state["plans"] = response.build_plan(
                state["gain_fits"], state["alpha_fits"], state["tremor_fit"],
                state["ladder"], state["reachable"], natives, tms=tms_star,
                alpha_style_cap=alpha_cap)

        # fit-state sidecar: written unconditionally so a LATER invocation
        # that only changes a planning flag (--skill, --no-demote) can
        # --replan-from this run instead of re-collecting/re-fitting.
        sidecar = ctx.out_dir / response.FIT_STATE_NPZ
        response.save_fit_state(
            sidecar, manifest=_replan_manifest(ctx), tms_star=tms_star,
            tms_fit=tms_fit, live=live, gain_fits=state["gain_fits"],
            alpha_fits=state["alpha_fits"], tremor_fit=state["tremor_fit"],
            ladder=state["ladder"], reachable=state["reachable"])
        _log(f"fit-state sidecar written: {rel_to_repo(sidecar)}")

    for w, p in state["plans"].items():
        _log(f"  {w}: band={p.band} gain={p.gain} α={p.alpha} pred={p.pred_hbw} "
             f"→ p{p.achieved_pct}{' REFUSED→frontier p%.1f' % p.frontier_pct if p.refused else ''}")

    # No confirmation round: the placement gate rides the fit's OWN bootstrap
    # CIs — the fit must be representative (all-pin, mix-weighted, per-wave
    # seeds) and honest about uncertainty, or it fails here, loud.
    plans = state["plans"]
    gate = gates.placement_gate(plans, state["gain_fits"], state["ladder"])
    _log(f"placement gate: {gate['status']}"
         + (f" — failed {gate['failed']}: " + "; ".join(
             f"{w}[{', '.join((gate['weapons'][w].get('fail_reasons') or ['band miss']))}]"
             for w in gate["failed"]) if gate["failed"] else ""))

    seed_reps: dict[str, Any] | None = None
    if args.seed_replicates > 0 and gate["status"] == "PASS":
        seed_reps = run_seed_replicates(ctx, substrate, tms_star, plans,
                                        state, ledger,
                                        n=int(args.seed_replicates))

    # crest capture (model): at-discharge vs window-tracking medians over the
    # SAME fit waves — the trigger-timing tax per weapon, pooled over all
    # swept operating points (report card + the attack-head RL target; the
    # human reference lives in _aim_tracking_window.json crest_capture:
    # SG/RL ≈ 0.83-0.86, spray weapons ≈ 0.94-0.98; an alignment-blind
    # trigger reads ≈ 1). Replan mode has no wave dirs (nothing was
    # collected this invocation) — the report card is skipped, not faked.
    crest: dict[str, Any] = {}
    if state["table"] is not None:
        _disch = instruments.collect_events(state["wave_dirs"])
        for _w in sorted(set(p.alias_of or w for w, p in plans.items())):
            _td = state["table"].where(weapon=_w)
            _dd = _disch.where(weapon=_w)
            if len(_td) and len(_dd):
                _mt = float(np.median(_td["hbw"]))
                _md = float(np.median(_dd["hbw"]))
                crest[_w] = {"window_median_hbw": round(_mt, 4),
                             "discharge_median_hbw": round(_md, 4),
                             "capture_ratio": round(_md / max(_mt, 1e-9), 4)}
        if crest:
            _log("crest capture (model, pooled): " + "  ".join(
                f"{w}={v['capture_ratio']}" for w, v in sorted(crest.items())))
    else:
        _log("crest capture skipped (replan mode: no wave dirs collected "
             "this invocation)")

    def _fit_stamp(f) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(f).items() if k != "_boot"}

    result: dict[str, Any] = {
        "skill": args.skill,
        "pct_basis": ("human sustained-band coordinate on the WINDOW-SAMPLED "
                      "TRACKING statistic (±k ticks around discharges, "
                      "trigger-free; _aim_tracking_window.json): log-hbw-"
                      "linear between the collect's validated placement "
                      "anchors (floor = p0, elite = p100; frozen selection "
                      "procedure, skill-curves §16.3), per calibration family "
                      "(SG+SSG, NG+SNG, RL, LG) — NOT a "
                      "population percentile, NOT comparable across corpora, "
                      "and NOT comparable to pre-tracking (at-discharge) "
                      "fits"),
        "placement_anchors": _anchor_stamp(ctx),
        "live_pins": live,
        "tms_fit": tms_fit, "budget": ledger.as_dict(),
        "placement_gate": gate,
        # full fit diagnostics (n_boot_ok, knee CIs, pin offsets) — the gate
        # rides these, so the report must carry them (the a26 first fit had
        # no way to see its own degenerate CIs)
        "fits": {
            "gain": {a: _fit_stamp(f)
                     for a, f in state["gain_fits"].items()},
            "alpha": {a: _fit_stamp(f)
                      for a, f in state["alpha_fits"].items() if f},
        },
        **({"seed_replicates": seed_reps} if seed_reps else {}),
        **({"crest_capture": crest} if crest else {}),
        **({"replanned_from": rel_to_repo(_resolve_fit_state_path(args.replan_from))}
           if args.replan_from is not None else {}),
        "plans": {w: vars(p) for w, p in plans.items()},
        "provenance": ctx.provenance(),
    }
    if not args.write:
        _report(ctx, result, "REPORT-ONLY")
        return 0

    version = args.version or emit.provisional_version(ctx.model_id, args.skill)
    vectors = response.build_vectors(plans)
    # Non-lever pins: the live-fitted fire-only vector (the exact vector every
    # wave ran on) + the move-commit dur_tilt. Switch-rate parity is owned by
    # the free-play attack trim (gates.attack_trim) + its warm start.
    emit_pins = {
        "attack.fire_bias_vec": live["fire_bias_vec"],
        "weapon.preference_bias_vec": [0.0] * 8,
        **move_commit_pins(ctx),
    }
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
                ctx, cfg, args.eval_template, tag=tag,
                episodes=TRANSITION_TRIM_EPISODES))
        result["attack_trim"] = trim
        npz = trim.get("npz")            # trim's final-eval npz, reused (dedup)
        if npz:
            _native = trim.get("native_npz")
            style = gates.style_gate(
                ctx, Path(npz), plans,
                native_npz=Path(_native) if _native else None,
                selection_target=trim.get("selection_target"))
    result["style_gate"] = style

    trim_ok = result.get("attack_trim", {}).get("status") == "CONVERGED"
    passed = (gate["status"] == "PASS" and trim_ok
              and style.get("status") == "PASS")
    _flags = (style.get("style_spend") or {}).get("flags") or []
    try:
        promoted = emit.promote_decode_config(staged, gate_passed=passed,
                                              force=args.force,
                                              style_flags=_flags)
        result["config"] = rel_to_repo(promoted)
        label = "PASS" if passed else "FORCED"
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
    # legacy v1 grids are AT-DISCHARGE cell medians — the retrofit keeps the
    # at-discharge ladder (statistic consistency; report-only legacy path)
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
            gain_fits.get(w, gain_fits.get(TRANSFER_ALIAS.get(w, w))).native,
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
