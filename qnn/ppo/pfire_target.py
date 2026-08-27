"""Human p_fire(alignment) target — the rung-3 trigger objective.

The attack head's "when to attack" problem, stated as a distribution match:
at every tick where firing is FEASIBLE and an actor is in LOS, the policy's
probability of attacking at all should equal the probability a human of the
target skill band attacks at that alignment.

Why this shape rather than a reward:

* A per-discharge reward pays per trigger-pull, so its optimum is to fire
  more. That is exactly what sank the 8M fine-tune (rung3a_ae2_ft8m: +33.6%
  discharge rate, SG crest .9139 -> .9626). A distribution match has no such
  gradient — over-firing and under-firing are both penalised.
* crest_capture, the gate, is a RATIO whose denominator moves with the
  policy's own ambient tracking, so optimising it directly rewards making
  ambient aim worse. p_fire has no denominator to game.
* It is per-tick and differentiable, so credit lands on the action that
  caused it, with no sampling noise.

AXIS. Alignment is keyed on hbw (hitbox-half-width-normalised angular error),
NOT raw degrees. Degrees are range-dependent and measurably wash out the
signal — the same elite cohort reads slope 3.48 on degrees and 18.98 on hbw.

DENOMINATOR. Only refire-ready ticks count. The attack head carries
feasibility_mask=true from the core base graph, so the policy is FORCED to
no_attack during cooldown; ~25% of corpus ticks are cooldown-locked, and the
dilution concentrates in the tight-alignment buckets (only 22% of SG [0,0.5)
ticks are ready vs 64% of [8,inf)). Both sides must share the gate.

SCOPE. Hitscan families only (SG/SSG/LG). RL is excluded twice over: the
all-LOS-tick geometry is weapon-free only for hitscan (RL carries z-drop),
and RL trigger selectivity is nearly flat across the top four skill quintiles
(16.52 -> 14.48 vs SG's 23.35 -> 15.89), so it is not where human RL skill
lives.

Targets are LOADED from the measured artifact, never hardcoded, so a run's
config records exactly which curve it was fit against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PFireTarget:
    """Piecewise-constant human p_fire over alignment bands.

    ``edges`` are the upper hbw bounds of bands 0..n-2 (band n-1 is the tail),
    ``p`` the per-band human fire probability. Both come from the skill-binned
    corpus artifact.
    """

    edges: np.ndarray     # (n-1,) ascending hbw upper bounds
    p: np.ndarray         # (n,) target probabilities
    family: str
    skill_label: str
    source: str

    def __post_init__(self) -> None:
        if self.p.shape[0] != self.edges.shape[0] + 1:
            raise ValueError(
                f"p_fire target needs len(p) == len(edges)+1, got "
                f"{self.p.shape[0]} and {self.edges.shape[0]}"
            )
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError("p_fire target edges must be strictly ascending")
        if not np.all((self.p >= 0.0) & (self.p <= 1.0)):
            raise ValueError("p_fire target probabilities must lie in [0, 1]")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PFireTarget":
        """Build from a run-config block.

        Required keys: ``artifact`` (path to _human_crest_by_skill*.json),
        ``family`` (e.g. "SG+SSG"), ``skill_bin`` (0 = elite). No defaults —
        the target curve is a pre-registered choice, not an implicit one.
        """
        missing = [k for k in ("artifact", "family", "skill_bin") if k not in cfg]
        if missing:
            raise RuntimeError(
                f"pfire_target config is missing {missing} — the trigger "
                "objective's curve must be named explicitly"
            )
        path = Path(str(cfg["artifact"]))
        family = str(cfg["family"])
        skill_bin = int(cfg["skill_bin"])
        doc = json.loads(path.read_text())
        fam = doc.get("families", {}).get(family)
        if fam is None:
            raise RuntimeError(
                f"{path}: no family {family!r} (have "
                f"{sorted(doc.get('families', {}))})"
            )
        bins = [b for b in fam["quartiles"] if int(b["bin"]) == skill_bin]
        if not bins:
            raise RuntimeError(f"{path}: family {family!r} has no bin {skill_bin}")
        band = bins[0]
        deciles = band.get("p_fire_by_ambient_decile")
        if not deciles:
            raise RuntimeError(
                f"{path}: family {family!r} bin {skill_bin} has no "
                "p_fire_by_ambient_decile — re-run crest_by_skill.py"
            )
        p = np.array([float(d["p_fire"]) for d in deciles], dtype=np.float64)
        # band i covers (prev_hi, hi]; the last band's hi is the histogram top
        # and is not a real edge.
        edges = np.array([float(d["hbw_hi"]) for d in deciles[:-1]], dtype=np.float64)
        return cls(
            edges=edges, p=p, family=family,
            skill_label=str(band.get("skill_label", f"bin{skill_bin}")),
            source=str(path),
        )

    def lookup(self, hbw: np.ndarray) -> np.ndarray:
        """Target p_fire for each alignment value (NaN in -> NaN out)."""
        hbw = np.asarray(hbw, dtype=np.float64)
        idx = np.digitize(hbw, self.edges, right=True)
        out = self.p[np.clip(idx, 0, self.p.shape[0] - 1)]
        return np.where(np.isfinite(hbw), out, np.nan)

    def provenance(self) -> dict[str, Any]:
        return {
            "artifact": self.source,
            "family": self.family,
            "skill_label": self.skill_label,
            "edges": [float(x) for x in self.edges],
            "p_fire": [float(x) for x in self.p],
        }


@dataclass(frozen=True)
class FireOccupancyTarget:
    """Human marginal fire occupancy on ALL engaged (pure-LOS) ticks.

    The target is a fires/second human rate; PPO runs at a fixed tick rate, so
    the exact per-decision occupancy is rate/tick_hz. Loaded straight from the
    PRIMARY human artifacts (see :meth:`from_config`) — never hand-copied
    constants, and since 2026-08-08 never a per-checkpoint decode-fit report
    either (the report carried human-side numbers it did not produce, which
    put the pins fit on the training launch's critical path for nothing).

    POPULATION, checked at load time by the KEY this reads, not a docstring:
    ``qnn.ppo.learner._fire_occupancy_loss``'s mask is
    ``torch.isfinite(align_hbw) & decision_mask`` — every tick with an in-LOS
    actor, no ``target_probs`` engagement label required. The live-pins block's
    ``target_rate_per_s`` is the WRONG number for that mask: it is fit from
    ``qnn.human.op_attack``, whose population additionally requires a
    ``target_probs``-labeled engagement (1.72x narrower on SG+SSG,
    agents/plans/blind-fire-cadence.md §3) — using it here overstates the
    target by roughly that same factor. This reads
    ``target_rate_per_s_los_engaged`` instead, sourced from
    ``qnn.human.blind_fire`` / ``qnn.decode_fit.human_refs.family_aimed_rates_los``,
    which shares this loss's exact population.

    CONTINUOUS WEAPONS (NG/SNG/LG, the family event correction —
    agents/plans/crest-finetune-allweapons.md "The objective"): the bolt
    rate above is the WRONG number for these three. Their human cadence
    ruler is the hold-train ONSET rate — how often a human STARTS firing,
    not how often the think-chain streams a bolt — sourced via
    ``qnn.decode_fit.human_refs.family_onset_rates_los``. ``basis`` records
    which population a given instance actually loaded, so a report or
    config diff never has to re-derive it from the weapon name. NOTE: the
    per-tick occupancy loss cannot consume an onset-basis target (it would
    train holds away); the learner fails loud on ``basis != "bolt"`` until
    the onset-basis loss exists.
    """

    probability: float
    rate_per_s: float
    tick_hz: float
    weapon: str
    family: tuple[str, ...]
    source: str
    basis: str = "bolt"   # "bolt" (per-discharge) or "onset" (hold-train start)

    # Continuous-fire weapons whose occupancy target is the hold-train onset
    # rate, not the bolt rate — mirrors
    # qnn.ppo.crest_reward.CONTINUOUS_ONSET_IMPULSES by abbreviation (kept as
    # a local literal, not an import, the same way this module already
    # hardcodes its scope elsewhere — see PFireTarget's SCOPE docstring).
    _ONSET_WEAPONS = frozenset({"NG", "SNG", "LG"})

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "FireOccupancyTarget":
        """Load the target from the PRIMARY human artifacts, not a fit report.

        Every number this class carries is human-side (a human rate on the
        loss's own population); routing it through a per-checkpoint decode-fit
        report put the wedge-prone pins fit on the training launch's critical
        path for no informational gain — the report was a carrier, never a
        source (2026-08-08, crest-finetune-allweapons.md). Config names the
        artifact matching the weapon's basis:

        * bolt (discrete weapons): ``blind_fire_artifact`` — the collect's
          ``_blind_fire_byweapon.json``; rate via ``family_aimed_rates_los``
          (mass-weighted same-physics pooling, e.g. SG+SSG).
        * onset (NG/SNG/LG): ``corrected_events_artifact`` —
          ``_human_crest_by_skill_corrected_events.json``; rate via
          ``family_onset_rates_los`` (per-weapon, no pooling).
        """
        missing = [k for k in ("weapon", "tick_hz") if k not in cfg]
        if missing:
            raise RuntimeError(
                f"fire_occupancy_target config is missing {missing} — the "
                "human cadence ruler must be named explicitly"
            )
        weapon = str(cfg["weapon"])
        tick_hz = float(cfg["tick_hz"])
        if tick_hz <= 0.0:
            raise ValueError(f"fire occupancy tick_hz must be > 0, got {tick_hz}")

        from qnn.decode_fit.human_refs import (
            family_aimed_rates_los,
            family_onset_rates_los,
        )

        if weapon in cls._ONSET_WEAPONS:
            basis = "onset"
            key = "corrected_events_artifact"
            if key not in cfg:
                raise RuntimeError(
                    f"fire_occupancy_target for {weapon!r} (continuous) "
                    f"requires {key!r} — the hold-train ONSET rate source "
                    "(_human_crest_by_skill_corrected_events.json); the "
                    "bolt rate is the wrong population (the family event "
                    "correction, agents/plans/crest-finetune-allweapons.md)."
                )
            path = Path(str(cfg[key]))
            rates = family_onset_rates_los(path)
            family: tuple[str, ...] = (weapon,)
        else:
            basis = "bolt"
            key = "blind_fire_artifact"
            if key not in cfg:
                raise RuntimeError(
                    f"fire_occupancy_target for {weapon!r} (discrete) "
                    f"requires {key!r} — the pure-LOS aimed-rate source "
                    "(_blind_fire_byweapon.json); qnn.human.op_attack's "
                    "LOS+labeled-engaged rate is ~1.72x too narrow for the "
                    "occupancy loss's mask (blind-fire-cadence.md)."
                )
            path = Path(str(cfg[key]))
            rates = family_aimed_rates_los(path)
            family = ("SG", "SSG") if weapon == "SG" else (weapon,)
        if weapon not in rates:
            raise RuntimeError(
                f"{path}: no {basis} rate for weapon {weapon!r} "
                f"(available: {sorted(rates)})"
            )
        rate = float(rates[weapon])

        probability = rate / tick_hz
        if not 0.0 < probability < 1.0:
            raise ValueError(
                f"fire occupancy rate/tick_hz must lie in (0, 1), got "
                f"{rate}/{tick_hz}={probability}"
            )
        return cls(
            probability=probability,
            rate_per_s=rate,
            tick_hz=tick_hz,
            weapon=weapon,
            family=family,
            source=str(path),
            basis=basis,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "artifact": self.source,
            "weapon": self.weapon,
            "family": list(self.family),
            "tick_hz": self.tick_hz,
            "basis": self.basis,
            "target_rate_per_s_los_engaged": self.rate_per_s,
            "target_probability": self.probability,
        }


# ── per-weapon PROJECTION targets (crest-finetune-allweapons iteration 2) ──
#
# Distinct from FireOccupancyTarget above: that class feeds the per-tick BCE
# LOSS (_fire_occupancy_loss) and fails loud on anything but a "bolt" basis
# because a per-tick loss cannot consume an onset rate without training
# holds away. FireOccupancyProjectionTarget instead feeds
# qnn.ppo.learner._project_fire_occupancy_per_weapon's BISECTION regulator —
# the general 9-way adapter's per-weapon analogue of _project_fire_occupancy
# (which only ever worked for a fixed-weapon PPO arm). A bisection has no
# such restriction: it matches a measured mean P(attack) on a tick
# population directly, so a continuous weapon's target here is a HELD
# FRACTION (see below), not an onset rate.

# Continuous-fire weapons: same partition FireOccupancyTarget uses
# (_ONSET_WEAPONS above) — kept as its own literal here rather than a cross
# reference so this dataclass reads standalone; it is a physics fact (which
# weapons stream via a hold-train), not a hand-copied number.
_PROJECTION_CONTINUOUS_WEAPONS = frozenset({"NG", "SNG", "LG"})
_PROJECTION_DISCRETE_WEAPONS = frozenset({"SG", "SSG", "GL", "RL"})
_PROJECTION_WEAPONS = _PROJECTION_CONTINUOUS_WEAPONS | _PROJECTION_DISCRETE_WEAPONS

# QC think-chain bolt rate while attack stays held (src/docs/mvd-fire-audit.md):
# the engine streams ~10 discharges/second for the whole hold, so a measured
# bolt rate divided by this constant is the FRACTION OF ENGAGED TICKS the
# button is down — exactly the per-tick Bernoulli quantity the projection's
# bisection solves for a continuous weapon (not the onset/train-start rate).
_HOLD_BOLTS_PER_SECOND = 10.0


@dataclass(frozen=True)
class FireOccupancyProjectionTarget:
    """One weapon's per-tick P(attack) target for the general-adapter
    fire-occupancy PROJECTION.

    ``basis``:
      * ``"aimed_family"`` — discrete weapon, rate pooled across its
        same-physics family by ``qnn.decode_fit.human_refs.family_aimed_rates_los``
        (SG's target is fit on SG+SSG-pooled mass; applies to SG only).
      * ``"aimed_own"`` — discrete weapon with NO family-pool entry (SSG is
        a pooled MEMBER, not the family key; GL has no
        ``CALIBRATION_FAMILIES`` entry at all) — its own raw
        ``aimed_rate_per_s`` from the blind-fire artifact, undivided.
      * ``"held_fraction"`` — continuous weapon (NG/SNG/LG): own raw
        ``aimed_rate_per_s`` (never pooled — each measured separately) divided
        by the QC hold-train bolt rate, giving the fraction of engaged ticks
        the button is held down.
    """

    weapon: str
    probability: float
    rate_per_s: float
    tick_hz: float
    basis: str
    source: str

    def provenance(self) -> dict[str, Any]:
        return {
            "weapon": self.weapon,
            "basis": self.basis,
            "rate_per_s": self.rate_per_s,
            "tick_hz": self.tick_hz,
            "probability": self.probability,
            "source": self.source,
        }


def _raw_aimed_rate_per_s(doc: Mapping[str, Any], weapon: str, path: Path) -> float:
    row = (doc.get("weapons") or {}).get(weapon) or {}
    rate = row.get("aimed_rate_per_s")
    if rate is None:
        raise RuntimeError(
            f"{path}: weapon {weapon!r} has no aimed_rate_per_s -- rebuild "
            "the collect's decode-fit human baselines "
            "(python -m qnn.human <collect_dir>)"
        )
    return float(rate)


def load_fire_occupancy_projection_targets(
    cfg: Mapping[str, Any],
) -> dict[str, "FireOccupancyProjectionTarget"]:
    """Per-weapon projection targets for the general 9-way PPO adapter
    (``qnn.ppo.learner._project_fire_occupancy_per_weapon``).

    Required config keys: ``blind_fire_artifact`` (the collect's
    ``_blind_fire_byweapon.json``) and ``tick_hz``. Optional ``weapons``
    (default: all 7 non-melee weapons) restricts which slots get a target
    this run. ``continuous_basis`` selects the continuous-weapon (NG/SNG/LG)
    ruler: ``"held_fraction"`` (default — blind-fire aimed rate over the QC
    bolt rate) or ``"onset"`` (hold-train starts per engaged-LOS tick from
    ``corrected_events_artifact`` via ``family_onset_rates_los``; the
    learner's projection bisects the EXPECTED ONSET RATE for these, see
    ``_project_fire_occupancy_per_weapon``). Discrete weapons are unaffected
    by ``continuous_basis``.

    Every rate is read from the artifact, never hand-copied. Same-physics
    families are pooled ONLY where ``qnn.decode_fit.human_refs.family_aimed_rates_los``
    actually has a key for the weapon (SG, RL); SSG/GL fall back to their own
    raw per-weapon rate (see ``FireOccupancyProjectionTarget.basis``) because
    each keeps a SEPARATE ``fire_bias`` slot, unlike the single-target
    forced-cadence fit this loader's sibling feeds.
    """
    missing = [k for k in ("blind_fire_artifact", "tick_hz") if k not in cfg]
    if missing:
        raise RuntimeError(
            "fire_occupancy_projection_targets config is missing "
            f"{missing} -- the per-weapon human cadence rulers must be "
            "named explicitly"
        )
    continuous_basis = str(cfg.get("continuous_basis", "held_fraction"))
    if continuous_basis not in ("held_fraction", "onset"):
        raise ValueError(
            "fire_occupancy_projection_targets.continuous_basis must be "
            f"'held_fraction' or 'onset', got {continuous_basis!r}"
        )
    onset_rates: dict[str, float] | None = None
    if continuous_basis == "onset":
        ce_path = cfg.get("corrected_events_artifact")
        if not ce_path:
            raise RuntimeError(
                "continuous_basis='onset' requires corrected_events_artifact "
                "(runs/head_probe/_human_crest_by_skill_corrected_events.json "
                "or a successor) -- the onset-rate rulers are never hand-typed"
            )
        from qnn.decode_fit.human_refs import family_onset_rates_los
        onset_rates = family_onset_rates_los(Path(str(ce_path)))
    tick_hz = float(cfg["tick_hz"])
    if tick_hz <= 0.0:
        raise ValueError(f"fire occupancy tick_hz must be > 0, got {tick_hz}")
    path = Path(str(cfg["blind_fire_artifact"]))
    if not path.exists():
        raise FileNotFoundError(
            f"blind-fire targets missing: {path} -- rebuild the collect's "
            "decode-fit human baselines (python -m qnn.human <collect_dir>)"
        )
    doc = json.loads(path.read_text())

    from qnn.decode_fit.human_refs import family_aimed_rates_los
    family_rates = family_aimed_rates_los(path)

    weapons_cfg = cfg.get("weapons")
    weapons = (
        [str(w) for w in weapons_cfg] if weapons_cfg
        else sorted(_PROJECTION_WEAPONS)
    )
    out: dict[str, FireOccupancyProjectionTarget] = {}
    for w in weapons:
        if w not in _PROJECTION_WEAPONS:
            raise ValueError(
                f"fire_occupancy_projection_targets: {w!r} is not one of "
                f"the non-melee weapons {sorted(_PROJECTION_WEAPONS)}"
            )
        if w in _PROJECTION_CONTINUOUS_WEAPONS and onset_rates is not None:
            if w not in onset_rates:
                raise RuntimeError(
                    f"continuous_basis='onset': no onset rate for {w!r} in "
                    f"{cfg.get('corrected_events_artifact')} "
                    f"(have {sorted(onset_rates)})"
                )
            rate = float(onset_rates[w])
            basis = "onset"
            probability = rate / tick_hz
        elif w in _PROJECTION_CONTINUOUS_WEAPONS:
            rate = _raw_aimed_rate_per_s(doc, w, path)
            basis = "held_fraction"
            probability = rate / _HOLD_BOLTS_PER_SECOND
        else:
            if w in family_rates:
                rate = float(family_rates[w])
                basis = "aimed_family"
            else:
                rate = _raw_aimed_rate_per_s(doc, w, path)
                basis = "aimed_own"
            probability = rate / tick_hz
        if not 0.0 < probability < 1.0:
            raise ValueError(
                f"{path}: weapon {w!r} projection probability out of "
                f"(0, 1): {probability} (rate_per_s={rate}, basis={basis})"
            )
        out[w] = FireOccupancyProjectionTarget(
            weapon=w, probability=probability, rate_per_s=rate,
            tick_hz=tick_hz, basis=basis, source=str(path),
        )
    return out
