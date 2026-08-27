"""Round 0: forced-weapon conditional cadence calibration.

One cell each forces SG, SNG, RL, and LG at the representative's top human
range pin with aim levers off and the fitted turn dampener active. The cells
fit ``attack.fire_bias_vec`` to human conditional fires/s while engaged. SG
carries SG+SSG; SNG carries NG+SNG. This avoids the selection starvation that
makes natural free play unable to identify low-preference family cadence.

The same cells also enforce enough discharge mass for the later response fit.
Every correction is measured live; a family that cannot reach its cadence or
evidence contract within the bounded secant schedule fails loud.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable

from qnn.decode_fit import design, human_refs, instruments
from qnn.decode_fit.context import (ABBR_TO_MODELNAME, FRIKBOT_TO_PIN,
                                    INSTRUMENT_WEAPONS, MODELNAME_TO_ABBR,
                                    WEAPON_IMPULSE, calibration_members)

_log = lambda m: print(f"[decode-fit] {m}", flush=True)  # noqa: E731
_PIN_TO_FRIKBOT = {v: k for k, v in FRIKBOT_TO_PIN.items()}

EPISODES_PER_CELL = design.EPISODES_PER_CELL
MASS_FLOOR = 6.0
MIN_ENGAGED_TICKS = 2000
RATE_REL_TOL = 0.15
RATE_DAMPING = 0.6
RATE_EPS = 0.01
BIAS_KICK = 3.0
BIAS_STEP_CLAMP = 3.0
BIAS_MIN_STEP = 0.05
MAX_SECANT_STEPS = 6


def _bias_vec(bias: dict[str, float]) -> list[float]:
    """Impulse-indexed fire vector with family values mirrored."""
    vec = [0.0] * 8
    for abbr, value in bias.items():
        for member in calibration_members(abbr):
            vec[WEAPON_IMPULSE[member] - 1] = round(float(value), 4)
    return vec


def _bake(substrate: dict[str, Any], bias: dict[str, float]) -> dict[str, Any]:
    sub = json.loads(json.dumps(substrate))
    sub["params"]["attack.fire_bias_vec"] = _bias_vec(bias)
    return sub


def _within_contract(row: dict[str, float], target: float) -> bool:
    rate_ok = abs(math.log(max(row["rate_per_s"], RATE_EPS) / target)) \
        <= math.log1p(RATE_REL_TOL)
    return (rate_ok and row["mass"] >= MASS_FLOOR
            and row["engaged_ticks"] >= MIN_ENGAGED_TICKS)


def _next_bias(history: list[dict[str, float]], target: float) -> float:
    """Damped secant on ``bias -> log(conditional fire rate)``."""
    b1, r1 = history[-1]["bias"], history[-1]["rate_per_s"]
    y1 = math.log(max(r1, RATE_EPS))
    goal = math.log(target)
    step: float | None = None
    if len(history) >= 2:
        b0, r0 = history[-2]["bias"], history[-2]["rate_per_s"]
        y0 = math.log(max(r0, RATE_EPS))
        if abs(b1 - b0) > 1e-9 and abs(y1 - y0) > 1e-9:
            step = RATE_DAMPING * (goal - y1) * (b1 - b0) / (y1 - y0)
    if step is None or not math.isfinite(step):
        step = BIAS_KICK if r1 <= 0 else RATE_DAMPING * (goal - y1)
    direction = 1.0 if target > r1 else -1.0
    if step * direction <= 0:
        step = direction * BIAS_MIN_STEP
    step = direction * min(max(abs(step), BIAS_MIN_STEP), BIAS_STEP_CLAMP)
    return b1 + step


def _run_probe_waves(ctx, substrate: dict[str, Any],
                     cells: dict[str, instruments.Cell],
                     step: int) -> dict[str, dict[str, float]]:
    tags = {abbr: f"livepin{step}_{abbr.lower()}" for abbr in cells}
    grouped = instruments.run_botpin_wave_groups(
        ctx, [{"cells": [cells[abbr]], "episodes_per_cell": EPISODES_PER_CELL,
               "tag": tags[abbr]} for abbr in sorted(cells)], substrate)
    observations: dict[str, dict[str, float]] = {}
    for abbr in cells:
        dirs = grouped[tags[abbr]]
        table = instruments.collect_events(dirs)
        mass = (0.0 if len(table) == 0
                else float(table.where(weapon=abbr)["weight"].sum()))
        observations[abbr] = {
            **instruments.collect_forced_attack_rate(dirs, abbr),
            "mass": mass,
        }
    return observations


def fit_live_pins(ctx, substrate: dict[str, Any], tms: float,
                  ledger: design.BudgetLedger, *,
                  wave_runner: Callable[..., dict[str, dict[str, float]]]
                  | None = None
                  ) -> dict[str, Any]:
    """Fit family cadence in the four forced-weapon pin cells."""
    runner = wave_runner or (
        lambda sub, cells, step: _run_probe_waves(ctx, sub, cells, step))
    pin_weights = human_refs.range_pin_weights(ctx.range_path)
    targets = human_refs.family_attack_rates(ctx.op_attack_path)
    abbrs = [MODELNAME_TO_ABBR[w] for w in INSTRUMENT_WEAPONS]
    matchups = {
        abbr: (ABBR_TO_MODELNAME[abbr],
               _PIN_TO_FRIKBOT[design.top_pins(pin_weights, abbr, n=1)[0]])
        for abbr in abbrs
    }
    bias = {abbr: 0.0 for abbr in abbrs}
    history: dict[str, list[dict[str, float]]] = {abbr: [] for abbr in abbrs}
    active = set(abbrs)
    for step in range(MAX_SECANT_STEPS + 1):
        cells = {abbr: instruments.Cell(
            model_weapon=matchups[abbr][0], frikbot_pin=matchups[abbr][1],
            op={"gain": 0.0, "alpha": 0.0, "tremor": 0.0, "tms": float(tms)})
            for abbr in active}
        ledger.charge(f"live-pins#{step}[{','.join(sorted(active))}]",
                      len(cells), EPISODES_PER_CELL, ci_extension=step > 0)
        observations = runner(_bake(substrate, bias), cells, step)
        for abbr in sorted(active):
            obs = observations[abbr]
            row = {
                "bias": round(bias[abbr], 4),
                "mass": float(obs["mass"]),
                "fires": int(obs["fires"]),
                "engaged_ticks": int(obs["engaged_ticks"]),
                "tick_hz": float(obs["tick_hz"]),
                "rate_per_s": float(obs["rate_per_s"]),
                "target_rate_per_s": float(targets[abbr]),
            }
            history[abbr].append(row)
            _log(f"live-pins#{step} {abbr}: bias {bias[abbr]:+.3f} -> "
                 f"{row['rate_per_s']:.3f}/{targets[abbr]:.3f} fires/s, "
                 f"mass {row['mass']:.0f}, ticks {row['engaged_ticks']} "
                 f"({'pass' if _within_contract(row, targets[abbr]) else 'retry'})")
        active = {abbr for abbr in active
                  if not _within_contract(history[abbr][-1], targets[abbr])}
        if not active:
            break
        if step == MAX_SECANT_STEPS:
            detail = {abbr: history[abbr] for abbr in sorted(active)}
            raise RuntimeError(
                f"live-pins: {', '.join(sorted(active))} failed forced cadence "
                f"after {MAX_SECANT_STEPS} secant steps: {json.dumps(detail)} "
                f"(targets {targets}, tolerance {RATE_REL_TOL:.0%}, mass floor "
                f"{MASS_FLOOR:g}, tick floor {MIN_ENGAGED_TICKS}; honest refusal)")
        for abbr in active:
            bias[abbr] = _next_bias(history[abbr], targets[abbr])

    return {
        "status": "PASS",
        "floor": MASS_FLOOR,
        "min_engaged_ticks": MIN_ENGAGED_TICKS,
        "rate_rel_tol": RATE_REL_TOL,
        "target_basis": ("human conditional fires/s while engaged, pooled by "
                         "same-physics family; forced pins own cadence"),
        "episodes_per_cell": EPISODES_PER_CELL,
        "tms": float(tms),
        "fire_bias_vec": _bias_vec(bias),
        "weapons": {
            abbr: {
                "impulse": WEAPON_IMPULSE[abbr],
                "family": list(calibration_members(abbr)),
                "matchup": {"model_weapon": matchups[abbr][0],
                            "frikbot_pin": matchups[abbr][1]},
                "target_rate_per_s": targets[abbr],
                "native_mass": history[abbr][0]["mass"],
                "native_rate_per_s": history[abbr][0]["rate_per_s"],
                "fitted_rate_per_s": history[abbr][-1]["rate_per_s"],
                "steps": history[abbr],
                "secant_steps": len(history[abbr]) - 1,
                "bias": bias[abbr],
            }
            for abbr in abbrs
        },
        "unprobed_impulses": sorted(
            set(WEAPON_IMPULSE.values())
            - {WEAPON_IMPULSE[m]
               for abbr in abbrs for m in calibration_members(abbr)}),
        "unprobed_note": ("Axe/GL are not held by the cadence instrument; "
                          "SSG/NG inherit their SG/SNG family bias"),
    }
