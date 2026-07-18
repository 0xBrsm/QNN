"""Per-weapon LOS-zone-conditioned turn-EMD on the REPAIRED reference.

Report-card ruler (retired from every gate — acquisition-band membership
replaced it there) shared by the decode-fit style report and the
turn-fidelity analysis. Moved verbatim from the retired
``qnn.eval.skill_vector`` (decode-fit v2). NG/SNG score against POOLED nail
reference mass (kinematics-identical and individually thin). Held-keyed both
sides — never compare to pre-repair (pre-atan2-comb, 7/08) EMD numbers.
"""
from __future__ import annotations

import numpy as np

# the intercept-valid weapons the ruler reports (mirrors qnn.decode_fit)
INTERCEPT_WEAPONS = ("SG", "SSG", "NG", "SNG", "LG", "RL")
NAILS = ("NG", "SNG")


def _emd_ordinal(p: list[float], q: list[float]) -> float:
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def perweapon_turn_emd(turn_by_los_w: dict, ref_skilled: dict,
                       turn_bins: list[str], los_bins: list[str]) -> float | None:
    """LOS-zone-conditioned turn-EMD vs one weapon's skilled reference (same
    zone-mass weighting as look_aim_factorial._turn_emd). ``turn_by_los_w`` is
    the eval's raw per-zone counts for the weapon; normalized here."""
    total_mass = sum(ref_skilled[lz]["n_ticks"] for lz in los_bins) or 1
    wsum, any_mass = 0.0, False
    for lz in los_bins:
        skz = ref_skilled.get(lz)
        if not skz:
            continue
        w = skz["n_ticks"] / total_mass
        raw = turn_by_los_w.get(lz, {})
        tot = sum(float(raw.get(tb, 0)) for tb in turn_bins)
        bp = [float(raw.get(tb, 0)) / tot for tb in turn_bins] if tot \
            else [0.0] * len(turn_bins)
        sp = [skz["turn_dist"][tb] for tb in turn_bins]
        if tot > 0:
            any_mass = True
        wsum += w * _emd_ordinal(bp, sp)
    return round(wsum, 4) if any_mass else None


def _pooled_nail_ref(reference: dict) -> dict:
    """Pool NG+SNG skilled reference mass per zone (kinematics-identical, thin)."""
    los_bins = reference["los_bins"]
    turn_bins = reference["turn_bins"]
    pw = reference["per_weapon"]
    out = {}
    for lz in los_bins:
        n = 0.0
        acc = {tb: 0.0 for tb in turn_bins}
        for w in NAILS:
            z = pw.get(w, {}).get("skilled", {}).get(lz)
            if not z:
                continue
            nz = z["n_ticks"]
            n += nz
            for tb in turn_bins:
                acc[tb] += z["turn_dist"][tb] * nz
        out[lz] = {"n_ticks": n,
                   "turn_dist": {tb: (acc[tb] / n if n else 0.0) for tb in turn_bins}}
    return out


def perweapon_emd_from_eval(summary: dict, reference: dict) -> dict[str, float | None]:
    """Score per-weapon turn-EMD (repaired ruler) from an eval_summary's
    ``engine_turn_by_los_angle_per_weapon``. NG/SNG scored against the POOLED
    nail reference mass. Held-keyed both sides (the ruler stays held)."""
    per_w = summary.get("engine_turn_by_los_angle_per_weapon") or {}
    turn_bins = reference["turn_bins"]
    los_bins = reference["los_bins"]
    pooled_nail = _pooled_nail_ref(reference)
    out: dict[str, float | None] = {}
    for abbr in INTERCEPT_WEAPONS:
        tw = per_w.get(abbr)
        if not tw:
            continue
        ref_sk = pooled_nail if abbr in NAILS else \
            reference["per_weapon"].get(abbr, {}).get("skilled")
        if not ref_sk:
            continue
        out[abbr] = perweapon_turn_emd(tw, ref_sk, turn_bins, los_bins)
    return out
