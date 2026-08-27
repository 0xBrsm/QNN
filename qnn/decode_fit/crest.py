"""The crest (discharge-quality gate) θ fit — steps 3-4 of
``agents/plans/discharge-quality-gate.md``.

    python -m qnn.decode_fit.crest --run-dir runs/head_probe/<run> \
        --base runs/decode_fit/<model>/decode.<rc>.json [--emit <out.json>]

The gain/α arms need waves. This one does not: every operative discharge the
fit already ran carries a ±k-tick alignment window, and the gated
counterfactual over that window is deterministic, so the whole θ → capture
curve comes out of the sweep's existing ``intercept_windows.npz`` files by
offline replay (plan §"The window-replay estimator").

What the driver adds on top of ``response.fit_crest_response`` /
``response.build_crest_plan``:

  * **cell selection.** θ is fit retroactively, so the sweep's grid rarely
    lands exactly on the plan's inverted (g, α). Cells are taken within an
    explicit tolerance of the base config's placed operating point, and the
    cells actually used — with their distance from that point — are printed and
    stamped, never assumed away.
  * **H choice.** Both admissible holds are fit and the SMALLEST H that arms
    with a real effect wins (plan open-question 2, hard cap 2 ticks / 100 ms).
  * **emission.** A copy of the base config with exactly two params added.

SCOPE: this ends at a fitted, emitted, PROVISIONAL config. Step 5 — the
closed-loop confirmation botpin at (g, α, θ) — is the deploy gate and is not
run here; the replay estimator's obs-vs-engine ruler gap and its
feed-forward-vs-realized divergence residual are exactly what that round
exists to catch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qnn.decode_fit import events, human_refs, response
from qnn.decode_fit.context import (CALIBRATION_FAMILY_KEY, WEAPON_IMPULSE,
                                    read_json, rel_to_repo,
                                    resolve_fit_context)

_log = lambda m: print(f"[crest-fit] {m}", flush=True)  # noqa: E731

# Cell tolerance around the base config's placed operating point. Wide enough
# to pool the neighbouring swept α rays (which is what makes the bootstrap
# honest — a single exact cell carries too few episodes), tight enough that the
# crest geometry is still the placed point's.
GAIN_TOL = 0.12
ALPHA_TOL = 0.30
# Wave tags whose windows are ON the mechanical botpin instrument. Free-play
# and acquisition waves run a different scenario mix and do not carry a swept
# operating point, so they are not comparable evidence for a per-weapon θ.
BOTPIN_TAG_PREFIXES = ("screen", "edgescreen", "extend", "ciext",
                       "alphaanchor", "confirm", "seedrep", "livepin")


def plans_from_config(cfg: dict[str, Any]) -> dict[str, response.WeaponPlan]:
    """Rebuild the resolved per-weapon plans from a promoted decode config's
    ``skill_vector.placed`` block. The crest arm composes ONTO a placement, so
    it must ride the exact placement the config it augments actually shipped —
    not a re-derivation that could drift from it."""
    placed = (cfg.get("skill_vector") or {}).get("placed") or {}
    if not placed:
        raise ValueError("base config carries no skill_vector.placed block — "
                         "the crest arm has no placement to compose onto")
    out: dict[str, response.WeaponPlan] = {}
    for abbr, p in placed.items():
        out[abbr] = response.WeaponPlan(
            weapon=abbr, impulse=int(p.get("impulse", WEAPON_IMPULSE[abbr])),
            target_pct=float(p["target_pct"]), target_hbw=float(p["target_hbw"]),
            gain=float(p["gain"]), alpha=float(p["alpha"]),
            pred_hbw=float(p["pred_hbw"]),
            pred_hbw_ci=tuple(p["pred_hbw_ci"]),
            achieved_pct=float(p["achieved_pct"]), refused=bool(p["refused"]),
            frontier_pct=float(p["frontier_pct"]),
            frontier_hbw=float(p["frontier_hbw"]), band=str(p["band"]),
            tremor=float(p.get("tremor", 0.0)), alias_of=p.get("alias_of"),
            notes=str(p.get("notes", "")))
    return out


def human_crest_capture(tracking_path: Path) -> dict[str, float]:
    """``{abbr: ratio_p50}`` — the human trigger-timing target per calibration
    family (``_aim_tracking_window.json`` ``crest_capture``: per-demo
    median(at-discharge hbw) / median(window hbw); < 1 = crest-firing, ≈ 1 =
    alignment-blind trigger). Fails loud on a family with no reference: an
    unreferenced θ is an invented style target."""
    doc = json.loads(Path(tracking_path).read_text())
    node = doc.get("crest_capture")
    if not node:
        raise ValueError(
            f"{tracking_path}: no crest_capture node — the corpus baselines "
            "predate the trigger-timing reference; rebuild via "
            "`python -m qnn.human <collect> --force`")
    out: dict[str, float] = {}
    for abbr in WEAPON_IMPULSE:
        for key in (CALIBRATION_FAMILY_KEY.get(abbr, abbr), abbr):
            row = node.get(key)
            if row and row.get("ratio_p50") is not None:
                out[abbr] = float(row["ratio_p50"])
                break
    return out


def botpin_wave_dirs(waves_dir: Path) -> list[Path]:
    """Every mechanical botpin wave under a fit's ``waves/`` that carries a
    window npz. Waves without one predate the instrument and are skipped (the
    arm reports how many, so a thin fit is visible)."""
    out = []
    for d in sorted(Path(waves_dir).iterdir()):
        if not d.is_dir() or not d.name.startswith(BOTPIN_TAG_PREFIXES):
            continue
        if (d / "metrics" / "eval" / events.TRACKING_WINDOWS_NPZ).exists():
            out.append(d)
    return out


def fit_crest(run_dir: Path, base_config: Path, *,
              gain_tol: float = GAIN_TOL, alpha_tol: float = ALPHA_TOL,
              hold_choices: tuple[int, ...] = response.CREST_H_CHOICES,
              min_effect: float = response.CREST_MIN_EFFECT,
              n_boot: int = 200, seed: int = 0) -> dict[str, Any]:
    """Run the replay fit + plan inversion for one model's rc line."""
    ctx = resolve_fit_context(run_dir)
    cfg = read_json(base_config)
    plans = plans_from_config(cfg)
    # AT-DISCHARGE references: θ moves the tick the trigger lands on, so its
    # coordinate is the intercept ladder, not the tracking ladder the gain/α
    # arms are placed on.
    ladder = human_refs.perweapon_human_ladder(ctx.intercept_path)
    reach = human_refs.reachable_band(ctx.intercept_path)
    pinw = human_refs.range_pin_weights(ctx.range_path)
    hcap = human_crest_capture(ctx.tracking_path)

    dirs = botpin_wave_dirs(ctx.waves_dir)
    if not dirs:
        raise FileNotFoundError(
            f"{ctx.waves_dir}: no botpin wave carries {events.TRACKING_WINDOWS_NPZ} "
            "— the crest arm has nothing to replay; re-run the waves with "
            f"QNN_EVAL_INTERCEPT_WINDOW={events.TRACKING_K}")
    _log(f"replaying {len(dirs)} botpin waves under {rel_to_repo(ctx.waves_dir)}")
    windows = events.load_waves_crest(dirs)
    tracking = events.load_waves_tracking(dirs)
    _log(f"{len(windows)} discharge windows / {len(tracking)} window-tick samples")

    sources = sorted({p.alias_of or w for w, p in plans.items()})
    fits: dict[int, dict[str, response.CrestResponse]] = {}
    for H in hold_choices:
        fits[H] = {}
        for src in sources:
            plan = plans[src]
            mw = response.select_operating_cells(
                windows, src, plan.gain, plan.alpha,
                gain_tol=gain_tol, alpha_tol=alpha_tol)
            mt = response.select_operating_cells(
                tracking, src, plan.gain, plan.alpha,
                gain_tol=gain_tol, alpha_tol=alpha_tol)
            fits[H][src] = response.fit_crest_response(
                windows.filter(mw), tracking.filter(mt), src, hold_ticks=H,
                pin_weights=pinw.get(src), n_boot=n_boot, seed=seed)

    # SMALLEST H that arms with a real effect wins (plan open-question 2).
    candidates: dict[int, dict[str, response.CrestPlan]] = {
        H: response.build_crest_plan(plans, fits[H], ladder, reach, hcap,
                                     min_effect=min_effect)
        for H in hold_choices}
    chosen_H = 0
    for H in sorted(hold_choices):
        if any(p.armed for p in candidates[H].values()):
            chosen_H = H
            break
    crest_plans = candidates[chosen_H] if chosen_H else \
        candidates[min(hold_choices)]

    report = {
        "stage": "crest θ replay fit (discharge-quality gate, plan steps 3-4)",
        "status": "PROVISIONAL — replay estimator only; the closed-loop "
                  "confirmation round (plan step 5) and the attack trim's "
                  "rate compensation (step 6) have NOT been run",
        "base_config": rel_to_repo(base_config),
        "base_version": cfg.get("version"),
        "hold_ticks": chosen_H,
        "hold_choices": list(hold_choices),
        "cell_tolerance": {"gain": gain_tol, "alpha": alpha_tol},
        "min_effect": min_effect,
        "n_waves": len(dirs), "n_discharge_windows": int(len(windows)),
        "n_tracking_rows": int(len(tracking)),
        "human_crest_capture_p50": {w: hcap[w] for w in sorted(hcap)
                                    if w in plans},
        "at_discharge_elite_hbw": {w: round(reach[w][0], 4)
                                   for w in sorted(plans) if w in reach},
        "arms": {str(H): {src: _arm_stamp(f) for src, f in fits[H].items()}
                 for H in hold_choices},
        "plans": {w: vars(p) for w, p in sorted(crest_plans.items())},
        "vectors": response.build_crest_vectors(crest_plans),
        "provenance": ctx.provenance(),
    }
    return {"ctx": ctx, "cfg": cfg, "crest_plans": crest_plans,
            "fits": fits, "report": report}


def _arm_stamp(f: response.CrestResponse) -> dict[str, Any]:
    """Report stamp of one arm: the anchors, the diagnostics, and the curve
    sampled at display knots (the full grid is an implementation detail)."""
    knots = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    return {
        "window_hbw": round(f.window_hbw, 4),
        "native_at_discharge_hbw": round(f.native_hbw, 4),
        "native_capture": round(f.native_capture, 4),
        "native_capture_ci": [round(x, 4) for x in f.native_capture_ci],
        "n_events": f.n_events, "n_clusters": f.n_clusters,
        "cells": [list(c) for c in f.cells],
        "diagnostics": f.diagnostics,
        "curve": [{"theta": t, "capture": round(f.capture_at(t), 4),
                   "capture_ci": [round(x, 4) for x in f.capture_ci_at(t)],
                   **f.flags_at(t)} for t in knots],
    }


def _description(cfg: dict[str, Any], crest_plans, vectors,
                 report: dict[str, Any]) -> str:
    armed = sorted(w for w, p in crest_plans.items() if p.armed)
    off = sorted(w for w, p in crest_plans.items() if not p.armed)
    return (
        f"{cfg.get('version')} base, CREST ARMED. Every other param is "
        f"byte-identical to {cfg.get('version')} — no re-fit; the only change "
        f"is the two discharge-quality-gate keys "
        f"(attack.crest_theta_vec / attack.crest_hold_ticks), fitted by the "
        f"θ replay arm over the rc1 line's EXISTING intercept_windows.npz "
        f"(agents/plans/discharge-quality-gate.md steps 3-4; no new eval "
        f"episodes were spent). H={vectors['attack.crest_hold_ticks']} tick. "
        f"ARMED: {', '.join(f'{w} θ={crest_plans[w].theta}' for w in armed) or 'none'}. "
        f"OFF (explicit 0.0): {', '.join(off) or 'none'} — each refused by the "
        f"p100 overshoot clamp or by having no crest gap; see "
        f"provenance.crest_fit.plans for the per-weapon reason. "
        f"PROVISIONAL: the closed-loop confirmation round (plan step 5) and "
        f"the attack trim's rate re-compensation (step 6) have NOT been run, "
        f"so this config is UNCONFIRMED and not deploy-blessed.")


def emit_config(base_config: Path, out_path: Path, cfg: dict[str, Any],
                crest_plans: dict[str, response.CrestPlan],
                report: dict[str, Any], version: str) -> Path:
    """Write ``base`` + exactly the two crest params, with a provenance note.

    Everything else is copied verbatim: this is a lever addition on a frozen
    placement, not a re-fit, so any other numeric drift would make the A/B
    uninterpretable. The base's own provenance (including any rc-assignment
    fields) is kept as-is and describes the LINEAGE — the crest block below is
    the only thing this emission is responsible for."""
    vectors = response.build_crest_vectors(crest_plans)
    out = json.loads(json.dumps(cfg))          # deep copy
    out["version"] = version
    out["description"] = _description(cfg, crest_plans, vectors, report)
    out["params"]["attack.crest_theta_vec"] = vectors["attack.crest_theta_vec"]
    out["params"]["attack.crest_hold_ticks"] = vectors["attack.crest_hold_ticks"]
    prov = out.setdefault("provenance", {})
    prov["crest_fit"] = {
        "base_config": rel_to_repo(base_config),
        "base_version": cfg.get("version"),
        "status": report["status"],
        "hold_ticks": report["hold_ticks"],
        "cell_tolerance": report["cell_tolerance"],
        "min_effect": report["min_effect"],
        "n_waves": report["n_waves"],
        "n_discharge_windows": report["n_discharge_windows"],
        "human_crest_capture_p50": report["human_crest_capture_p50"],
        "at_discharge_elite_hbw": report["at_discharge_elite_hbw"],
        "plans": report["plans"],
        "arms": report["arms"],
        "estimator": (
            "offline replay of attack_crest_gate_step over the fit's existing "
            "per-discharge ±4-tick alignment windows: for a discharge at t0 "
            "the gated counterfactual is deterministic, so the whole θ curve "
            "comes out of waves already run. Known residuals the closed-loop "
            "confirmation exists to catch: (1) the replay reads the REALIZED "
            "next-tick hbw where the decode reads the feed-forward z_rate "
            "estimate of it; (2) the eval's engine-side hbw ruler vs the "
            "decode's obs-side z_err ruler; (3) the estimator assumes the look "
            "stream is policy-invariant to a ≤1-tick held trigger."),
        "clamp": (
            "θ is capped so the replayed at-discharge level's bootstrap LOWER "
            "CI stays at or above the at-discharge p100 (= the elite anchor "
            "exactly, human_refs §16). An over-elite ask is REFUSED and the "
            "plan rides the clamp, the same semantics as a gain refusal riding "
            "the achievable frontier."),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.decode_fit.crest",
                                 description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path,
                    help="the promoted decode config the crest params are "
                         "added to (params otherwise copied verbatim)")
    ap.add_argument("--emit", type=Path, default=None,
                    help="write the augmented config here")
    ap.add_argument("--version", default=None,
                    help="version string for the emitted config")
    ap.add_argument("--report", type=Path, default=None,
                    help="write the full fit report JSON here (default: "
                         "summary to stdout only)")
    ap.add_argument("--gain-tol", type=float, default=GAIN_TOL)
    ap.add_argument("--alpha-tol", type=float, default=ALPHA_TOL)
    ap.add_argument("--min-effect", type=float,
                    default=response.CREST_MIN_EFFECT)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)

    res = fit_crest(args.run_dir, args.base, gain_tol=args.gain_tol,
                    alpha_tol=args.alpha_tol, min_effect=args.min_effect,
                    n_boot=args.n_boot)
    report, crest_plans = res["report"], res["crest_plans"]
    for H, arms in sorted(res["fits"].items()):
        for src, f in sorted(arms.items()):
            d = f.diagnostics
            _log(f"arm H={H} {src}: window {f.window_hbw:.3f} native_disch "
                 f"{f.native_hbw:.3f} capture {f.native_capture:.3f}"
                 f"{tuple(round(x, 3) for x in f.native_capture_ci)} "
                 f"n={f.n_events}/{f.n_clusters}cl cells={f.cells} "
                 f"monotone={d['monotone_frac']} span={d['capture_span']}")
    _log(f"chosen H = {report['hold_ticks']}")
    for w, p in sorted(crest_plans.items()):
        _log(f"  {w}: θ={p.theta} armed={p.armed} refused={p.refused} "
             f"clamped={p.clamped} capture {p.native_capture}→{p.placed_capture} "
             f"(human {p.target_capture}) at-discharge "
             f"{p.native_at_discharge_hbw}→{p.at_discharge_hbw} hbw "
             f"(p{p.native_at_discharge_pct}→p{p.at_discharge_pct}, "
             f"p100={p.elite_hbw})"
             + (f" — {p.notes}" if p.notes else ""))
    _log("vectors: " + json.dumps(report["vectors"]))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        _log(f"wrote {args.report}")
    if args.emit:
        version = args.version or Path(args.emit).stem.replace("decode.", "")
        p = emit_config(args.base, args.emit, res["cfg"], crest_plans,
                        report, version)
        _log(f"emitted {rel_to_repo(p)} (version {version}) — PROVISIONAL: "
             "closed-loop confirmation NOT run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
