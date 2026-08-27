"""Live crest-law reward shaping — fire-at-alignment rung 3.

Scores each DISCHARGE tick by the alignment of the shot it just took:

    bonus = weight * (exp(-gamma * hbw) - exp(-gamma * baseline_hbw))

where ``hbw`` is the lead-corrected angular error to the most-aligned
in-LOS actor's intercept point, hitbox-half-width normalized — the ONE law
shared with the offline sidecar (``qnn.bc.cache_align_hbw``), the
closed-loop ruler (``qnn.eval.run._intercept_hbw``), and the human baseline
(``scripts/analysis/human_crest_baseline.py``): the lead geometry is
``aim_kernel._lead_aim_angle_deg_live`` (→ ``qnn.model.lead_aim``) and the
normalizer is ``degrees(atan2(ACTOR_HALFW_U, raw_range))``. Parity with the
cache pass is test-pinned (tests/test_ppo_rung3.py::
test_crest_shaper_matches_offline_sidecar_law).

Live conventions (vs the offline label):

* Scored on the PRE-step obs — the state the shot was taken from — with
  THIS tick's attack class as the weapon (a live discharge needs no
  nearest-discharge attribution; the fired weapon is the action itself).
* Gate = in-LOS actor present (``entity_types == TOKEN_ACTOR &
  entity_modality_id == 0``). The offline label's additional
  ``target_probs`` engagement gate has no live analog; ``has_los`` alone is
  the precedented relaxation (``human_crest_baseline``'s wider ``has``
  gate).
* ``baseline_hbw`` centers the law so baseline-placement fires pay ~zero:
  better-than-baseline placement earns positive reward, wild fires pay
  negative — the anchor KL (learner-side) bounds the marginal excursion.

Config-keyed under ``ppo`` config ``crest_reward`` (absent = shaping off,
byte-identical rollout rewards). Process backend (``VecQuakeEnv``) only.

FAMILY EVENT CORRECTION (2026-08-08, the all-weapons crest fine-tune prep;
agents/plans/crest-finetune-allweapons.md "The objective") — WHICH discharge
events earn the crest term, never how alignment is scored (the hbw law
itself, ``qnn.ppo.align_hbw.AlignHbw``, is untouched and stays one-law
parity-pinned with the offline sidecar):

* SG/SSG/GL/RL (discrete): every operative discharge scores. GL's ~70%
  no-target ground-control fire scores zero with NO penalty — the same
  NaN-hbw gate every blind discharge already hits (no separate code path).
* NG/SNG/LG (continuous): ONSET-gated. Only the first discharge of a
  hold-train earns the crest term; the engine's think-chain/hold-tail
  streams the rest at short, mechanical gaps that carry no fresh aiming
  decision, so they score exactly zero — never a penalty. Onset detection
  (``OnsetGate``) is causal and online, per (lane, weapon): a discharge is
  an onset iff the tick-gap since that weapon's previous discharge in the
  SAME episode exceeds a configurable threshold (default 6 ticks == 0.3s at
  20 Hz), or there was no previous discharge this episode at all. State
  resets at episode boundaries and never crosses weapons — an ammo-out
  fallback discharge on a different weapon (the pinned-weapon-keying
  finding: LG-pinned cells fell back to SG 6.6% of the time) cannot corrupt
  another weapon's hold-train clock.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from qnn.eval import aim_kernel as A
from qnn.ppo.align_hbw import (
    ALIGNMENT_SCORED_IMPULSES,
    AlignHbw,
)

# Continuous-fire weapons whose human aiming decision is the hold-train
# ONSET, not every discharge tick (NG, SNG, LG raw impulses). Default gap
# threshold: 6 ticks (0.3s at 20 Hz) — the engine's think-chain re-fires at
# ~2-tick gaps and the hold-tail runs 5 ticks past release, so a gap past
# tail expiry is a fresh decision, not a continuation. Configurable per
# weapon via ``crest_reward.onset_gap_ticks`` (impulse-int keys).
CONTINUOUS_ONSET_IMPULSES = frozenset({4, 5, 8})   # NG, SNG, LG
DEFAULT_ONSET_GAP_TICKS: Dict[int, int] = {imp: 6 for imp in CONTINUOUS_ONSET_IMPULSES}


class OnsetGate:
    """Causal, per-(lane, weapon) onset detector for continuous-weapon
    hold-trains (NG/SNG/LG).

    A discharge on weapon ``w`` is an ONSET iff the number of ticks elapsed
    since the previous ``w`` discharge in the SAME episode exceeds
    ``gap_ticks[w]`` — or there was no previous ``w`` discharge this episode
    at all (episode start, or ``w``'s first-ever discharge). State is one
    int counter per (lane, weapon) — "ticks since last discharge", -1
    meaning "none yet this episode" — advanced ONE TICK PER ``step()`` CALL.
    ``step()`` is the only interface: it must be called exactly once per
    environment tick (even on ticks with no gated-weapon discharge), or the
    gap clock desyncs from wall-clock time.

    Episode boundaries are detected causally from the caller-supplied
    per-lane episode id: a lane whose id differs from the one last seen has
    every weapon's counter reset to "none yet" — the first discharge of a
    new episode is always an onset, regardless of how the previous episode
    ended. Weapons are tracked independently: an intervening discharge on a
    DIFFERENT weapon (e.g. an ammo-out fallback) never resets or advances
    another weapon's counter beyond the ordinary one-tick clock advance.
    """

    def __init__(self, gap_ticks: Mapping[int, int]) -> None:
        if not gap_ticks:
            raise ValueError("OnsetGate needs at least one weapon's gap_ticks")
        for imp, gap in gap_ticks.items():
            if int(gap) <= 0:
                raise ValueError(
                    f"onset gap_ticks[{imp}] must be > 0, got {gap}")
        self._gap_ticks: Dict[int, int] = {
            int(k): int(v) for k, v in gap_ticks.items()
        }
        self.impulses = tuple(sorted(self._gap_ticks))
        self._slot = {imp: i for i, imp in enumerate(self.impulses)}
        self._since_last: np.ndarray | None = None       # (lanes, weapons)
        self._last_episode_id: np.ndarray | None = None  # (lanes,)

    def _ensure(self, num_lanes: int) -> None:
        if self._since_last is not None and self._since_last.shape[0] == num_lanes:
            return
        self._since_last = np.full(
            (num_lanes, len(self.impulses)), -1, dtype=np.int64)
        self._last_episode_id = np.full(num_lanes, -1, dtype=np.int64)

    def step(
        self,
        attack: np.ndarray,
        active: np.ndarray,
        episode_ids: np.ndarray,
    ) -> np.ndarray:
        """Advance one tick; return the (B,) onset mask.

        True only where ``attack`` this tick is a gated weapon on an ACTIVE
        lane AND it is an onset; False everywhere else (including
        non-discharge and non-gated-weapon lanes) — callers must AND this
        with their own alignment-scored mask, never read False as "blocked".
        """
        attack = np.asarray(attack).reshape(-1).astype(np.int64)
        active = np.asarray(active, dtype=bool).reshape(-1)
        episode_ids = np.asarray(episode_ids).reshape(-1).astype(np.int64)
        b = attack.shape[0]
        if active.shape[0] != b or episode_ids.shape[0] != b:
            raise ValueError(
                f"OnsetGate.step shape mismatch: attack={b}, "
                f"active={active.shape[0]}, episode_ids={episode_ids.shape[0]}"
            )
        self._ensure(b)
        assert self._since_last is not None and self._last_episode_id is not None

        # Episode boundary: an active lane whose id moved has no discharge
        # history this episode yet — its first gated discharge is an onset
        # unconditionally, whatever the raw tick-gap says.
        boundary = active & (episode_ids != self._last_episode_id)
        if boundary.any():
            self._since_last[boundary, :] = -1
        self._last_episode_id = np.where(
            active, episode_ids, self._last_episode_id)

        # Wall-clock advance: one more tick has passed since each weapon's
        # last discharge, for every ACTIVE lane (a held/replayed lane never
        # advances). Slots with no discharge yet this episode (-1) stay -1.
        if active.any():
            live = self._since_last[active]
            started = live >= 0
            live[started] += 1
            self._since_last[active] = live

        onset = np.zeros(b, dtype=bool)
        for imp, slot in self._slot.items():
            fired = active & (attack == imp)
            if not fired.any():
                continue
            gap = self._since_last[fired, slot]
            onset[fired] = (gap < 0) | (gap > self._gap_ticks[imp])
            self._since_last[fired, slot] = 0
        return onset


class CrestRewardShaper:
    """Per-tick discharge-alignment bonuses over a stacked lane batch."""

    def __init__(
        self,
        *,
        weight: float,
        gamma: float,
        baseline_hbw: float,
        hit_damage_weight: float = 0.0,
        onset_gap_ticks: Mapping[int, int] | None = None,
    ) -> None:
        self.weight = float(weight)
        self.gamma = float(gamma)
        self.baseline_hbw = float(baseline_hbw)
        # Actual-contact term (Brian 8/5, the ToT-PPO lesson: alignment is
        # the CURVE, a landed hit is the much larger reward): per tick,
        # + hit_damage_weight × damage_dealt_other. Damage-scaled so SG
        # pellet fractions and splash pay proportionally; GAE carries
        # projectile flight-time credit back to the firing decision.
        self.hit_damage_weight = float(hit_damage_weight)
        self._baseline_term = float(np.exp(-self.gamma * self.baseline_hbw))
        self._align = AlignHbw()
        # Family event correction: NG/SNG/LG are onset-gated (see module
        # docstring). ``onset_gap_ticks`` overrides the default per-weapon
        # threshold (impulse-int keys, e.g. {8: 8} to loosen LG only) — keys
        # outside CONTINUOUS_ONSET_IMPULSES are a config error, not a silent
        # no-op.
        gap_ticks = dict(DEFAULT_ONSET_GAP_TICKS)
        if onset_gap_ticks:
            unknown = {int(k) for k in onset_gap_ticks} - CONTINUOUS_ONSET_IMPULSES
            if unknown:
                raise ValueError(
                    f"onset_gap_ticks has non-continuous impulses "
                    f"{sorted(unknown)} — only "
                    f"{sorted(CONTINUOUS_ONSET_IMPULSES)} (NG/SNG/LG) are "
                    "onset-gated"
                )
            gap_ticks.update({int(k): int(v) for k, v in onset_gap_ticks.items()})
        self._onset_gate = OnsetGate(gap_ticks)
        # Cumulative telemetry, surfaced via snapshot() into collect stats.
        self._n_scored = 0
        self._hbw_sum = 0.0
        self._bonus_sum = 0.0
        self._hit_damage_sum = 0.0
        self._n_onset_gated = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CrestRewardShaper":
        """Build from the run config block — every knob explicit, no defaults
        (the law's constants are pre-registered in the rung-3 spec)."""
        missing = [k for k in ("weight", "gamma", "baseline_hbw") if k not in cfg]
        if missing:
            raise RuntimeError(
                f"crest_reward config is missing {missing} — the rung-3 law "
                "requires weight, gamma, and baseline_hbw explicitly"
            )
        onset_cfg = cfg.get("onset_gap_ticks")
        onset_gap_ticks = (
            {int(k): int(v) for k, v in onset_cfg.items()}
            if onset_cfg else None
        )
        return cls(
            weight=float(cfg["weight"]),
            gamma=float(cfg["gamma"]),
            baseline_hbw=float(cfg["baseline_hbw"]),
            hit_damage_weight=float(cfg.get("hit_damage_weight", 0.0)),
            onset_gap_ticks=onset_gap_ticks,
        )

    @property
    def onset_gap_ticks(self) -> Dict[int, int]:
        """The effective per-weapon onset gap thresholds (impulse -> ticks),
        defaults merged with any config override — read-only."""
        return dict(self._onset_gate._gap_ticks)

    def hit_bonus(self, info: Mapping[str, Any]) -> float:
        """Contact reward for one lane-tick from the engine's damage records
        (damage_dealt_other excludes self-damage — rocket jumps stay free)."""
        if self.hit_damage_weight == 0.0:
            return 0.0
        dmg = float(info.get("damage_dealt_other", 0.0))
        if dmg <= 0.0:
            return 0.0
        self._hit_damage_sum += dmg
        return self.hit_damage_weight * dmg

    def hbw(
        self,
        obs: Mapping[str, np.ndarray],
        attack: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Per-lane align-hbw for discharge lanes; NaN where unscored.

        Delegates to qnn.ppo.align_hbw.AlignHbw — the geometry lives there so
        the trigger objective's all-LOS-tick denominator and this scorer can
        never be two different laws."""
        return self._align.at_discharge(obs, attack, active)

    def hbw_all_los(
        self,
        obs: Mapping[str, np.ndarray],
        active: np.ndarray,
    ) -> np.ndarray:
        """Per-lane align-hbw for EVERY LOS lane — the p_fire denominator.
        Hitscan geometry: SG/SSG/LG only, not RL. See AlignHbw.all_los."""
        return self._align.all_los(obs, active)

    def bonuses(
        self,
        obs: Mapping[str, np.ndarray],
        attack: np.ndarray,
        active: np.ndarray,
        episode_ids: np.ndarray,
    ) -> np.ndarray:
        """(B,) shaping bonuses; zero on non-discharge/unscored lanes AND on
        continuous-weapon (NG/SNG/LG) hold-train CONTINUATION discharges —
        only the onset earns the crest term (family event correction).

        ``episode_ids`` (per-lane episode id) is REQUIRED and must be
        supplied on EVERY call this shaper makes for a given rollout,
        including ticks with no continuous-weapon discharge at all: the
        onset gate's per-(lane, weapon) gap clock is wall-clock state, not
        conditional on this tick's action, and a skipped call desyncs it.
        """
        hbw = self.hbw(obs, attack, active)
        attack_arr = np.asarray(attack).reshape(-1).astype(np.int64)
        active_arr = np.asarray(active, dtype=bool).reshape(-1)
        onset = self._onset_gate.step(attack_arr, active_arr, episode_ids)
        continuous = np.isin(attack_arr, self._onset_gate.impulses)
        scored = np.isfinite(hbw) & (~continuous | onset)
        bonus = np.zeros(hbw.shape[0], dtype=np.float32)
        if scored.any():
            h = hbw[scored]
            bonus[scored] = self.weight * (
                np.exp(-self.gamma * h) - self._baseline_term
            )
            self._n_scored += int(scored.sum())
            self._hbw_sum += float(h.sum())
            self._bonus_sum += float(bonus[scored].sum())
        # Telemetry only — these lanes are unscored, never penalized.
        gated_away = continuous & active_arr & np.isfinite(hbw) & ~onset
        self._n_onset_gated += int(gated_away.sum())
        return bonus

    def snapshot(self) -> Dict[str, float]:
        """Cumulative shaping telemetry (merged into collect stats)."""
        return {
            "crest_scored": float(self._n_scored),
            "crest_hbw_mean": (
                self._hbw_sum / self._n_scored if self._n_scored else 0.0
            ),
            "crest_bonus_sum": self._bonus_sum,
            "crest_hit_damage_sum": self._hit_damage_sum,
            "crest_onset_gated": float(self._n_onset_gated),
        }
