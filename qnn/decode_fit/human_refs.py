"""Readers for the corpus-derived HUMAN references (collect-keyed, path REQUIRED).

The single reader of the ``qnn.human`` baseline artifact schemas the fit
consumes: the pooled per-weapon INTERCEPT ladder, the per-demo-median
REACHABLE band, and the ACQUISITION (Fitts-throughput) band. Ports the v1
``skill_vector`` readers with one deliberate change: **every function takes
the collect-keyed artifact path** — there are no cwd-relative module defaults
(the v1 ``_fit_turn_mag_scale`` bug silently read a retired global copy
because the default argument existed).

Skill-coordinate semantics (settled 2026-07-16, skill-curves §16): a SKILL
percentile ``s`` is a **band coordinate** — position ``s/100`` along the
human SUSTAINED band, log-hbw-linear between two contamination-robust
anchors: p0 = the FLOOR anchor (worst-typical session) and p100 = the ELITE
anchor (elite-typical session). It is NOT a population percentile and NOT a
pooled-event quantile: the corpus is a convenience sample of sessions, so
distributional claims are unidentifiable (support statements survive
arbitrary reweighting; density statements do not — skill-curves §16). The
old pooled-event ladder (every shot one sample) is retired as a
coordinate — its p90 was the top-10% of individual shots, a level no human
sustains; it survives only as report context via ``interception_pctiles``.
SG/SSG and NG/SNG pool within their same-physics calibration families; the
family coordinate is copied onto both impulses. Coordinates never pool across
those families or with LG/RL. The frontier still rides ``max(fit floor
upper-CI, elite anchor)``; since p100 = the elite anchor exactly,
refusals are now always model-floor refusals, never dishonest-target
refusals.

Anchor provenance (skill-curves §16.3, 2026-07-16): the anchors are no
longer fixed p10/p90 depths on ``spread_of_median`` — they are per-corpus
OUTPUTS of the frozen selection procedure run by the ``qnn.human``
post-collect intercept builder (bootstrap-CI stability + out-of-sample
holdout + breakdown-vs-contamination criteria, reliability shrinkage,
same-physics family fallback), emitted as the artifact's
``placement_anchors`` node. This module reads ONLY that node and FAILS
LOUD on artifacts that predate it (no fallback — repo doctrine: no legacy
paths). Band coordinates are therefore NOT comparable across corpora;
cross-corpus comparisons are made in hbw.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from qnn.decode_fit.context import (CALIBRATION_FAMILIES,
                                    CALIBRATION_FAMILY_KEY,
                                    INTERCEPT_WEAPONS)

# Display knots the band ladder is sampled at for reports (interpolation is
# exact log-linear between the two anchors regardless of knots).
LADDER_PCTS = (0, 10, 25, 50, 75, 90, 100)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ── INTERCEPT (per-weapon pooled event ladder) ────────────────────────────────

def interception_pctiles(doc: dict, weapon: str) -> dict[str, float] | None:
    """``{p1..p99: hbw}`` for one weapon from a loaded ``_aim_intercept_skill``
    doc (``interception_dist[weapon].norm.percentiles``). The single reader of
    that schema. None if the weapon/node is absent."""
    node = (((doc.get("interception_dist") or {}).get(weapon) or {}).get("norm") or {})
    pct = node.get("percentiles")
    return {k: float(v) for k, v in pct.items()} if pct else None


def placement_anchors(intercept_path: Path) -> dict[str, Any]:
    """The artifact's ``placement_anchors`` node — validated per-corpus anchors
    from the frozen selection procedure (``qnn.human.intercept``, skill-curves
    §16.3): ``{anchors_version, procedure, config, weapons: {abbr: {elite_hbw,
    floor_hbw, elite_depth, floor_depth, reliability_sb, shrunk,
    family_borrowed, unvalidated, …}}}``. FAILS LOUD when the node is absent —
    the artifact predates the procedure and is stale; rebuild via
    ``python -m qnn.human <collect> --force``. No fallback to fixed
    spread-of-median depths (repo doctrine: no legacy paths). The fit report
    stamps this node's flags via ``cli._anchor_stamp``."""
    path = Path(intercept_path)
    if not path.exists():
        raise FileNotFoundError(
            f"human intercept baseline not found: {path} — run "
            "`python -m qnn.human <collect>` (collect-cached; no global copy)")
    doc = _read(path)
    node = doc.get("placement_anchors")
    if not isinstance(node, dict) or not node.get("weapons"):
        raise ValueError(
            f"placement_anchors missing from {path} — the artifact predates the "
            "frozen per-corpus anchor-selection procedure (skill-curves §16.3); "
            "rebuild via `python -m qnn.human <collect> --force`")
    return node



def perweapon_human_ladder(intercept_path: Path) -> dict[str, dict[float, float]]:
    """``{abbr: {band_pct: hbw}}`` — each calibration family's skill ladder,
    copied onto its member impulses and log-hbw-linear between the validated
    sustained-band anchors (p0 =
    ``placement_anchors`` floor, p100 = elite), sampled at the display knots.
    Raises if the artifact or its ``placement_anchors`` node is missing (no
    fallback; rebuild via ``python -m qnn.human <collect> --force``)."""
    wnode = placement_anchors(intercept_path)["weapons"]
    out: dict[str, dict[float, float]] = {}
    for abbr in INTERCEPT_WEAPONS:
        family = CALIBRATION_FAMILY_KEY.get(abbr, abbr)
        e = wnode.get(family) or {}
        elite, worst = e.get("elite_hbw"), e.get("floor_hbw")
        if elite is None or worst is None or not (0 < elite < worst):
            continue
        lw, le = math.log(float(worst)), math.log(float(elite))
        out[abbr] = {float(s): float(math.exp(lw + (s / 100.0) * (le - lw)))
                     for s in LADDER_PCTS}
    missing = [w for w in INTERCEPT_WEAPONS if w not in out]
    if missing:
        raise ValueError(
            f"validated placement anchors missing weapons {missing} in "
            f"{intercept_path} (placement_anchors.weapons.<family> "
            "elite_hbw/floor_hbw)"
            " — rebuild via `python -m qnn.human <collect> --force`")
    return out


def hbw_to_pct(hbw: float, ladder: dict[float, float]) -> float:
    """Model hbw → band coordinate (log-hbw-linear; smaller hbw → higher pct;
    clamped to [0, 100] — below-band reads 0.0, beyond-elite reads 100.0)."""
    pairs = sorted((math.log(h), p) for p, h in ladder.items())  # log-hbw asc
    hs = np.array([h for h, _ in pairs], float)
    ps = np.array([p for _, p in pairs], float)                  # descending
    return float(np.interp(math.log(max(hbw, 1e-9)), hs, ps))


def pct_to_hbw(pct: float, ladder: dict[float, float]) -> float:
    """Band coordinate → target hbw (log-hbw-linear, clamped to the anchors)."""
    pairs = sorted(ladder.items())                               # by pct asc
    ps = np.array([p for p, _ in pairs], float)
    hs = np.array([math.log(h) for _, h in pairs], float)
    return float(math.exp(np.interp(pct, ps, hs)))


def reachable_band(intercept_path: Path) -> dict[str, tuple[float, float]]:
    """``{abbr: (elite_hbw, median_p50)}`` — the per-demo-median REACHABLE band:
    the validated ELITE anchor (``placement_anchors``; the frontier rides it)
    plus the typical-session median (``spread_of_median[norm:W].median_p50`` —
    a descriptive midpoint, not a placement anchor). Frontier/stop criteria
    ride this, never the pooled-event tail (a median never reaches the pooled
    p10). Raises when ``placement_anchors`` is absent (stale artifact)."""
    wnode = placement_anchors(intercept_path)["weapons"]
    sm = _read(Path(intercept_path)).get("spread_of_median") or {}
    out: dict[str, tuple[float, float]] = {}
    for w in INTERCEPT_WEAPONS:
        family = CALIBRATION_FAMILY_KEY.get(w, w)
        elite = (wnode.get(family) or {}).get("elite_hbw")
        p50 = (sm.get(f"norm:{family}") or {}).get("median_p50")
        if elite is not None and p50 is not None:
            out[w] = (float(elite), float(p50))
    return out


# ── ACQUISITION (global Fitts-throughput band) ────────────────────────────────

def acquisition_band(acq_path: Path) -> dict[str, Any]:
    """The GLOBAL human acquisition band: ``{pct: throughput}`` ladder + the
    per-demo-median min/max membership envelope + the shared ISO effective
    width. Raises if the artifact or any required node is absent."""
    p = Path(acq_path)
    if not p.exists():
        raise FileNotFoundError(
            f"human acquisition baseline not found: {p} — run "
            "`python -m qnn.human <collect>`")
    doc = _read(p)
    pooled = ((doc.get("acquisition_dist") or {}).get("percentiles") or {})
    ladder = {float(pk): float(pooled[k]) for pk, k in
              ((10.0, "p10"), (25.0, "p25"), (50.0, "p50"),
               (75.0, "p75"), (90.0, "p90"), (99.0, "p99"))
              if pooled.get(k) is not None}
    if len(ladder) < 2:
        raise ValueError(f"acquisition_dist.percentiles missing from {p}")
    med = (((doc.get("TEST_1_which_component_spreads") or {})
            .get("per_component_spread") or {}).get("throughput_bits_per_s") or {})
    m_min, m_max = med.get("min"), med.get("max")
    if m_min is None or m_max is None:
        raise ValueError(f"acquisition per-demo-median range missing from {p}")
    we = (doc.get("acquisition_dist") or {}).get("effective_width_deg")
    if we is None:
        raise ValueError(f"acquisition_dist.effective_width_deg missing from {p} "
                         "— the model side must ride the same ISO instrument")
    rel = (((doc.get("TEST_1_which_component_spreads") or {})
            .get("split_half_reliability") or {}).get("throughput") or {})
    return {
        "metric": "throughput_bits_per_s",
        "scope": "global",
        "ladder": ladder,
        "band_basis": "per_demo_median_min_max",
        "band": [round(float(m_min), 4), round(float(m_max), 4)],
        "median_bits_per_s": med.get("p50"),
        "effective_width_deg": float(we),
        "split_half_reliability": rel.get("split_half_spearman"),
        "n_players": doc.get("n_players"),
        "artifact": str(p),
    }


def throughput_to_pct(tp: float, ladder: dict[float, float]) -> float:
    """Model throughput (bits/s) → human acquisition SKILL percentile.
    ASCENDING (higher throughput → higher pct); clamped at the ends."""
    pairs = sorted((float(p), float(t)) for p, t in ladder.items())
    ps = np.array([p for p, _ in pairs], float)
    ts = np.array([t for _, t in pairs], float)
    return float(np.interp(tp, ts, ps))


def pct_to_throughput(pct: float, ladder: dict[float, float]) -> float:
    pairs = sorted((float(p), float(t)) for p, t in ladder.items())
    ps = np.array([p for p, _ in pairs], float)
    ts = np.array([t for _, t in pairs], float)
    return float(np.interp(pct, ps, ts))


# ── engagement-range pin weights ──────────────────────────────────────────────

def range_pin_weights(range_path: Path) -> dict[str, dict[str, float]]:
    """``{abbr: {fsg,fng,frl,flg: mass}}`` — the human per-weapon engagement-
    range mass at each frikbot pin archetype (``_aim_range_byweapon.json``
    ``weapons[w].pin_weights``). Empty dict if the artifact carries none
    (callers fall back to uniform pin pooling, logged by the caller)."""
    p = Path(range_path)
    if not p.exists():
        return {}
    doc = _read(p)
    nodes = doc.get("weapons") or {}
    out = {str(w): {str(k): float(v) for k, v in blk["pin_weights"].items()}
           for w, blk in nodes.items()
           if isinstance(blk, dict) and blk.get("pin_weights")}
    # Pool underlying frame mass, not normalized member weights, so the
    # calibration curve rides the actual human family range mixture.
    for source, members in (("SG", ("SG", "SSG")),
                            ("SNG", ("NG", "SNG"))):
        mass: dict[str, float] = {}
        for member in members:
            blk = nodes.get(member) or {}
            n = float(blk.get("n_frames") or 0.0)
            for pin, weight in (blk.get("pin_weights") or {}).items():
                mass[str(pin)] = mass.get(str(pin), 0.0) + n * float(weight)
        total = sum(mass.values())
        if total > 0.0:
            family_weights = {pin: value / total for pin, value in mass.items()}
            for member in members:
                out[member] = dict(family_weights)
            out[source] = dict(family_weights)
    return out


def family_attack_rates(op_attack_path: Path) -> dict[str, float]:
    """Human conditional fire rate for each forced-weapon representative.

    Same-physics members are pooled by human engaged-LOS mass. Missing family
    evidence is fatal: the four-pin cadence fit may not silently substitute a
    singleton or skip a family.
    """
    p = Path(op_attack_path)
    if not p.exists():
        raise FileNotFoundError(
            f"op-attack targets missing: {p} — rebuild the collect's "
            "decode-fit human baselines (python -m qnn.human <collect_dir>)")
    weapons = (_read(p).get("weapons") or {})
    out: dict[str, float] = {}
    for source, members in CALIBRATION_FAMILIES.items():
        rows = []
        for member in members:
            row = weapons.get(member) or {}
            if row.get("rate_per_s") is not None:
                rows.append((float(row["rate_per_s"]),
                             float(row.get("engaged_los_ticks") or 1.0)))
        if not rows:
            raise ValueError(
                f"op-attack targets {p} lack cadence evidence for "
                f"{'+'.join(members)}")
        out[source] = sum(rate * weight for rate, weight in rows) / sum(
            weight for _, weight in rows)
    return out
