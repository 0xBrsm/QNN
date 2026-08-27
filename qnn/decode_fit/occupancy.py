"""a26rc1b active-fire OCCUPANCY — the estimand and its two bias controllers.

The estimand (:func:`selection_profile`) is the rc1b repair's preference
ruler: operative discharges weighted by each weapon's refire duration. Human
weapon scripts make held-weapon frame mass an invalid preference ruler, so
dwell is estimated as ``discharge count x refire seconds`` instead of counted
off held frames.

Two closed-loop controllers step ``attack.bias_vec`` until that occupancy
tracks a named target profile at a FIXED ``weapon.switch_margin``:

* :func:`calibrate_occupancy_to_human` — the anti-camp pass. qnn_arena8 is
  full-loadout (no pickups) so weapon share is pure preference; pull the
  profile toward the human corpus at a modest margin (rc1b: 1.0).
* :func:`calibrate_occupancy_to_native` — the jitter-fix pass. A margin big
  enough to reproduce the human SWITCH rate (rc1b: 2.3) is sticky enough to
  park the bot on LG; step the bias so the sticky attractor becomes the
  model's own zero-margin (native) occupancy — i.e. break the LG-vs-RL tie
  toward RL. Low switch rate (margin) AND native occupancy (bias).

Both were run by hand to produce the deployed a26rc1b decode config;
``scripts/analysis/_occ_calib_a26rc1b.py`` and ``_bias_to_native_a26rc1b.py``
are thin drivers over them. For NEW fits prefer ``gates.attack_trim``, the
reconciled controller that fits fire cadence, selection, switch rate and
threat-break jointly on ``weapon.preference_bias_vec``; these two stay
importable and versioned because a shipped artifact was produced by exactly
this logic and must remain reproducible.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qnn.decode_fit.context import IMPULSE_NAME, read_json

# Frozen a26rc1b active-fire dwell weights in seconds, indexed by raw weapon
# id 1..8 (slot 0 = "no weapon", unused). Provenance: the a26rc1b hand repair
# (awposw seed43) — nominal per-weapon refire durations, deliberately NOT
# inferred from held-weapon frames or weapon-script state. ``gates``
# re-exports this; there is no second copy anywhere.
RC1B_REFIRE_SEC = np.asarray(
    [0.0, 0.5, 0.5, 0.7, 0.1, 0.1, 0.6, 0.8, 0.1], dtype=np.float64)

# ── controller constants (rc1b values; both passes share these) ──────────────
OCC_DAMPING = 0.6                  # step = damping x log-ratio (anti-oscillation)
OCC_BIAS_CLAMP = (-8.0, 4.0)
OCC_MAX_ITERS = 8

# ── pass 1: occupancy → HUMAN corpus profile ─────────────────────────────────
# human corpus active-fire share (%) — the target profile
HUMAN_OCC_SHARE_PCT = {1: 3.4, 2: 43.8, 3: 4.0, 4: 0.6, 5: 1.0,
                       6: 4.6, 7: 40.3, 8: 2.4}
HUMAN_OCC_SWITCH_MARGIN = 1.0
HUMAN_OCC_TOL = 0.30
HUMAN_OCC_SHARE_FLOOR_PCT = 0.05
# only weapons carrying real share on either side adjudicate convergence
HUMAN_OCC_CONVERGE_MIN_PCT = 3.0

# ── pass 2: occupancy → the model's own NATIVE (zero-margin) profile ─────────
# measured at weapon.switch_margin = 0 for awposw seed43
A26RC1B_NATIVE_OCC_SHARE_PCT = {1: 0.0, 2: 2.5, 3: 0.3, 4: 0.0, 5: 0.4,
                                6: 0.2, 7: 81.5, 8: 15.2}
NATIVE_OCC_SWITCH_MARGIN = 2.3
NATIVE_OCC_TOL = 0.25
NATIVE_OCC_SHARE_FLOOR_PCT = 0.3
# below 1% on BOTH sides a weapon is too noisy to steer — leave it alone
NATIVE_OCC_MIN_SHARE_PCT = 1.0

# ``launch_eval(config_path, tag) -> npz path | None`` (injected free-play).
LaunchEval = Callable[[Path, str], Path | None]


def _log(msg: str) -> None:
    print(f"[occupancy] {msg}", flush=True)


# ══ 1. estimand ═══════════════════════════════════════════════════════════════

def selection_profile(npz: Path) -> dict[str, Any] | None:
    """a26rc1b active-fire occupancy from refire-weighted discharges.

    Human weapon scripts make held-weapon frame mass an invalid preference
    ruler. The fixed rc1b repair instead estimated active-fire dwell as
    ``operative discharge count * weapon refire duration`` and preserved the
    zero-margin distribution of that quantity. Keep that exact estimand while
    using the reconciled selection-only preference vector as its control.
    """
    try:
        z = np.load(npz)
        discharge = np.asarray(z["discharge"]).reshape(-1).astype(bool)
        weapon = np.asarray(z["weapon_imp"]).reshape(-1).astype(np.int64)
    except Exception:
        return None
    if len(weapon) != len(discharge) or not discharge.any():
        return None
    counts = {name: int(((weapon == impulse) & discharge).sum())
              for impulse, name in IMPULSE_NAME.items()}
    weighted = {
        name: float(counts[name] * RC1B_REFIRE_SEC[impulse])
        for impulse, name in IMPULSE_NAME.items()
    }
    total = sum(weighted.values())
    if total <= 0.0:
        return None
    return {
        "metric": "refire-weighted operative-discharge share (a26rc1b)",
        "operative_discharges": sum(counts.values()),
        "counts": counts,
        "refire_seconds": {
            name: float(RC1B_REFIRE_SEC[impulse])
            for impulse, name in IMPULSE_NAME.items()
        },
        "weighted_seconds": weighted,
        "shares": {w: weighted[w] / total for w in weighted},
    }


def occupancy_share_pct(npz: Path) -> dict[int, float] | None:
    """:func:`selection_profile` shares as PERCENT keyed by raw weapon id.

    The controllers below think in the rc1b percent units their targets are
    written in; the gate thinks in fractions. One estimand, two renderings —
    never a second estimator.
    """
    profile = selection_profile(Path(npz))
    if profile is None:
        return None
    shares = profile["shares"]
    return {impulse: 100.0 * shares[name]
            for impulse, name in IMPULSE_NAME.items()}


def raw_weapon_switch_per_frame(npz: Path) -> float | None:
    """UNFILTERED within-episode weapon-switch fraction — a progress row only.

    Counts ``weapon_imp`` changes inside each episode over ALL frames and
    divides by the total frame count. This is NOT the gate's switch ruler:
    ``gates._weapon_switch_diag`` measures engaged (``keep``) frames only,
    per SECOND, against a matched human reference. The two numbers are not
    interchangeable and only the gate's version may adjudicate anything.
    """
    try:
        z = np.load(npz, allow_pickle=True)
        weapon = np.asarray(z["weapon_imp"]).reshape(-1)
        offsets = (np.asarray(z["episode_offsets"]).reshape(-1)
                   if "episode_offsets" in z.files else None)
    except Exception:
        return None
    if not len(weapon):
        return None
    switches = 0
    if offsets is not None and len(offsets) > 1:
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            if b - a > 1:
                switches += int((weapon[a + 1:b] != weapon[a:b - 1]).sum())
    else:
        switches = int((weapon[1:] != weapon[:-1]).sum())
    return switches / len(weapon)


# ══ 2. controller ═════════════════════════════════════════════════════════════

def occupancy_bias_loop(config_path: Path, launch_eval: LaunchEval, *,
                        target_pct: dict[int, float],
                        switch_margin: float,
                        tol: float,
                        share_floor_pct: float,
                        converge_on: Callable[[int, float, float], bool],
                        step_on: Callable[[int, float, float], bool],
                        tag_prefix: str,
                        label: str,
                        max_iters: int = OCC_MAX_ITERS,
                        damping: float = OCC_DAMPING,
                        bias_clamp: tuple[float, float] = OCC_BIAS_CLAMP,
                        ) -> dict[str, Any]:
    """Damped log-ratio bias controller on the rc1b occupancy estimand.

    Each iteration writes ``config_path + '.occtrim.json'`` at the current
    bias, runs one free-play wave through ``launch_eval``, measures occupancy,
    and steps ``attack.bias_vec[k-1] += damping * log(target/observed)``
    (both sides floored at ``share_floor_pct`` so an empty weapon cannot
    produce an infinite step, result clamped to ``bias_clamp``).

    ``converge_on(impulse, target_pct, observed_pct)`` selects the weapons
    that adjudicate convergence — a profile is converged when every selected
    weapon's ``|log-ratio|`` is within ``tol``. ``step_on`` selects the
    weapons that get a correction; weapons too small on both sides are left
    alone rather than chased through noise.

    The bias frozen back into ``config_path`` is always the one that PRODUCED
    the last MEASURED occupancy, never the stepped-but-unevaluated vector.
    """
    config_path = Path(config_path)
    cfg = read_json(config_path)
    cfg["params"]["weapon.switch_margin"] = float(switch_margin)
    bias = [float(b) for b in (cfg["params"].get("attack.bias_vec")
                               or [0.0] * 8)]
    work = Path(str(config_path) + ".occtrim.json")

    history: list[dict[str, Any]] = []
    frozen = [round(b, 4) for b in bias]
    converged = False
    status = "MAX-ITERS"
    note: str | None = None
    for it in range(int(max_iters)):
        used = [round(b, 4) for b in bias]
        cfg["params"]["attack.bias_vec"] = used
        work.write_text(json.dumps(cfg, indent=2) + "\n")
        npz = launch_eval(work, f"{tag_prefix}{it}")
        share = occupancy_share_pct(Path(npz)) if npz is not None else None
        if share is None:
            status = "EVAL-FAILED"
            note = (f"iteration {it} produced no measurable occupancy — "
                    "loop aborted, last measured bias kept")
            _log(f"{label}: it{it} eval FAILED")
            break
        frozen = used
        worst = 0.0
        for impulse in sorted(target_pct):
            t_pct = float(target_pct[impulse])
            o_pct = float(share[impulse])
            log_ratio = math.log(max(t_pct, share_floor_pct)
                                 / max(o_pct, share_floor_pct))
            if converge_on(impulse, t_pct, o_pct):
                worst = max(worst, abs(log_ratio))
            if step_on(impulse, t_pct, o_pct):
                bias[impulse - 1] = float(np.clip(
                    bias[impulse - 1] + damping * log_ratio, *bias_clamp))
        history.append({
            "iter": it,
            "bias_used": used,
            "occupancy_pct": {IMPULSE_NAME[k]: round(share[k], 1)
                              for k in sorted(share)},
            "target_pct": {IMPULSE_NAME[k]: float(target_pct[k])
                           for k in sorted(target_pct)},
            "weapon_switch_per_frame": raw_weapon_switch_per_frame(Path(npz)),
            "worst_abs_log_ratio": round(worst, 3),
        })
        _log(f"{label}: it{it} worst|log-ratio|={worst:.2f} "
             + " ".join(f"{IMPULSE_NAME[k]} {share[k]:.1f}/{target_pct[k]}"
                        for k in sorted(target_pct)
                        if converge_on(k, float(target_pct[k]),
                                       float(share[k]))))
        if worst <= tol:
            converged = True
            status = "CONVERGED"
            _log(f"{label}: CONVERGED at it{it}")
            break

    # freeze the bias that PRODUCED the last measured occupancy
    cfg["params"]["attack.bias_vec"] = frozen
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    work.unlink(missing_ok=True)
    report: dict[str, Any] = {
        "controller": label,
        "config": str(config_path),
        "status": status,
        "converged": converged,
        "switch_margin": float(switch_margin),
        "tol": float(tol),
        "max_iters": int(max_iters),
        "target_pct": {IMPULSE_NAME[k]: float(target_pct[k])
                       for k in sorted(target_pct)},
        "final_bias_vec": frozen,
        "history": history,
    }
    if note:
        report["note"] = note
    return report


def calibrate_occupancy_to_human(config_path: Path, launch_eval: LaunchEval, *,
                                 max_iters: int = OCC_MAX_ITERS,
                                 ) -> dict[str, Any]:
    """rc1b anti-camp pass: occupancy → the human corpus profile.

    Runs at a modest ``weapon.switch_margin`` (1.0, not the trim's sticky
    2.40) so the bias — not hysteresis — is doing the work. Every weapon is
    stepped; only weapons above 3% on either side adjudicate convergence.
    """
    return occupancy_bias_loop(
        config_path, launch_eval,
        target_pct=HUMAN_OCC_SHARE_PCT,
        switch_margin=HUMAN_OCC_SWITCH_MARGIN,
        tol=HUMAN_OCC_TOL,
        share_floor_pct=HUMAN_OCC_SHARE_FLOOR_PCT,
        converge_on=lambda _k, t, o: max(t, o) > HUMAN_OCC_CONVERGE_MIN_PCT,
        step_on=lambda _k, _t, _o: True,
        tag_prefix="occ",
        label="occupancy->human",
        max_iters=max_iters)


def calibrate_occupancy_to_native(config_path: Path, launch_eval: LaunchEval, *,
                                  target_pct: dict[int, float] | None = None,
                                  max_iters: int = OCC_MAX_ITERS,
                                  ) -> dict[str, Any]:
    """rc1b jitter-fix pass: occupancy → the model's own zero-margin profile.

    Holds ``weapon.switch_margin`` at the human-switch-rate value (2.3) and
    steps the bias until the sticky attractor is the native occupancy, so the
    bot sticks to RL rather than parking on LG. ``target_pct`` defaults to the
    measured awposw-seed43 native profile; pass a freshly measured zero-margin
    profile for any other checkpoint (native occupancy is per-model).
    """
    return occupancy_bias_loop(
        config_path, launch_eval,
        target_pct=(A26RC1B_NATIVE_OCC_SHARE_PCT if target_pct is None
                    else target_pct),
        switch_margin=NATIVE_OCC_SWITCH_MARGIN,
        tol=NATIVE_OCC_TOL,
        share_floor_pct=NATIVE_OCC_SHARE_FLOOR_PCT,
        converge_on=lambda _k, t, _o: t >= NATIVE_OCC_MIN_SHARE_PCT,
        step_on=lambda _k, t, o: max(t, o) >= NATIVE_OCC_MIN_SHARE_PCT,
        tag_prefix="bn",
        label="occupancy->native",
        max_iters=max_iters)
