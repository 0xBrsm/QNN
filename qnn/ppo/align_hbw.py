"""Per-tick alignment hbw — ONE law, shared by the reward and the trigger objective.

Extracted from ``qnn.ppo.crest_reward`` (2026-08-06) so the geometry outlives
the crest reward itself: the rung-3 trigger objective needs alignment on EVERY
LOS tick (the p_fire denominator), not only on discharge ticks, and it must be
the same law the offline sidecar (``qnn.bc.cache_align_hbw``), the closed-loop
ruler (``qnn.eval.run._intercept_hbw``) and the human baseline
(``scripts/analysis/human_crest_baseline.py``) use — lead geometry from
``aim_kernel._lead_aim_angle_deg_live`` (-> ``qnn.model.lead_aim``), normalizer
``degrees(atan2(ACTOR_HALFW_U, raw_range))``.

Computed rollout-side in numpy rather than re-derived in torch learner-side,
specifically so a second implementation of the law cannot exist.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from qnn.eval import aim_kernel as A

# Weapon families where target-alignment at discharge is a HUMAN aiming
# signal (corpus-validated taxonomy, family-event-corrected 2026-08-08 —
# agents/plans/crest-finetune-allweapons.md "The objective"): every ranged
# weapon except melee. Discrete weapons (SG/SSG/GL/RL) are scored on EVERY
# operative discharge; GL's ~70% no-target ground-control fire naturally
# scores zero (no penalty) via the NaN-hbw gate below, same mechanism as any
# other blind fire. Continuous weapons (NG/SNG/LG) are scored HERE (hbw is
# computed for them) but the caller (``qnn.ppo.crest_reward.CrestRewardShaper``)
# additionally onset-gates them: only a hold-train's first discharge earns
# the crest term, since the engine's think-chain/hold-tail streams the rest
# at short, mechanical gaps that carry no fresh aiming decision. Axe (melee)
# is the one true category error for a ranged-alignment law and stays out.
ALIGNMENT_SCORED_IMPULSES = frozenset({2, 3, 4, 5, 6, 7, 8})  # every non-melee weapon

# Representative impulse for the all-LOS-tick geometry. Under the rev-E
# current-position anchor the velocity term is zeroed and the only per-weapon
# input left is ballistic z-drop over flight time, which the hitscan x100 boost
# drives to nothing — SG/SSG/LG agree to max|diff| = 0 exactly. RL does NOT
# (z-drop, max 16.9 deg), so it cannot use the all-tick path.
HITSCAN_IMPULSE = 2   # SG


class AlignHbw:
    """Per-lane alignment-hbw over a stacked lane batch."""

    def __init__(self) -> None:
        self._weapon_np, _ = A._build_physics_tables()

    @staticmethod
    def los(obs: Mapping[str, np.ndarray]) -> np.ndarray:
        types = np.asarray(obs["entity_types"])
        # SIGHT-visible actor: modality == 0 (percept v3) or recency <= 0
        # (legacy v1 streams — the a26-line wire-shim declarations carry
        # recency and no modality column; same dual read as
        # qnn.eval.h2h._log_lane_tick / qnn.eval.run._log_streams).
        if "entity_modality_id" in obs:
            vis = np.asarray(obs["entity_modality_id"]) == 0
        else:
            vis = np.asarray(obs["entity_recency"], dtype=np.float64) <= 0.0
        return (types == A.TOKEN_ACTOR) & vis                      # (B, N)

    def _for(
        self,
        obs: Mapping[str, np.ndarray],
        imp: np.ndarray,
        cand: np.ndarray,
        los: np.ndarray,
    ) -> np.ndarray:
        """Per-lane hbw over the ``cand`` lanes, keyed on the impulses in
        ``imp``. NaN off ``cand`` or where no LOS actor resolves."""
        out = np.full(cand.shape[0], np.nan, dtype=np.float32)
        idx = np.flatnonzero(cand)
        if not len(idx):
            return out

        rel = np.asarray(obs["entity_rel"])[idx].astype(np.float32)   # raw u
        vel = np.asarray(obs["entity_vel"])[idx].astype(np.float32)   # raw u/s
        dist_u = np.linalg.norm(rel, axis=2)                          # (M, N)
        angles = A._lead_aim_angle_deg_live(
            rel * (1.0 / A._DIST_SCALE),
            vel * (1.0 / A._VEL_SCALE),
            imp[idx],
            self._weapon_np,
        )
        angles = np.where(los[idx], angles, np.inf)
        slot = np.argmin(angles, axis=1)
        rows = np.arange(len(idx))
        best_ang = angles[rows, slot]
        best_range = dist_u[rows, slot]

        finite = np.isfinite(best_ang)
        ang_radius = np.degrees(
            np.arctan2(A.ACTOR_HALFW_U, np.maximum(best_range, 1e-3))
        )
        hbw = best_ang / np.maximum(ang_radius, 1e-6)
        out[idx[finite]] = hbw[finite].astype(np.float32)
        return out

    def at_discharge(
        self,
        obs: Mapping[str, np.ndarray],
        attack: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """hbw for discharge lanes only, keyed on the FIRED weapon — the
        p_fire numerator and the crest scorer's input."""
        attack = np.asarray(attack).reshape(-1).astype(np.int64)
        los = self.los(obs)
        cand = (
            np.asarray(active, dtype=bool)
            & np.isin(attack, list(ALIGNMENT_SCORED_IMPULSES))
            & los.any(axis=1)
        )
        return self._for(obs, attack, cand, los)

    def all_los(
        self,
        obs: Mapping[str, np.ndarray],
        active: np.ndarray,
    ) -> np.ndarray:
        """hbw for EVERY LOS lane, fired or not — the p_fire DENOMINATOR.

        Hitscan geometry (see HITSCAN_IMPULSE): valid for SG/SSG/LG, NOT RL.
        """
        los = self.los(obs)
        cand = np.asarray(active, dtype=bool) & los.any(axis=1)
        imp = np.full(cand.shape[0], HITSCAN_IMPULSE, dtype=np.int64)
        return self._for(obs, imp, cand, los)
