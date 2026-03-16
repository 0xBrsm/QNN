"""PvP reward shaping for transformer/SF training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

# Relative pickup value by Quake weapon_id (1=Axe … 8=LG).
# SSG=1.0 is the baseline; RL is the most valuable DM weapon.
WEAPON_TIER_VALUES: Dict[int, float] = {
    1: 0.0,   # Axe — no DM value
    2: 0.5,   # Shotgun
    3: 1.0,   # Super Shotgun (baseline)
    4: 0.75,  # Nailgun
    5: 1.25,  # Super Nailgun
    6: 1.5,   # Grenade Launcher
    7: 2.0,   # Rocket Launcher
    8: 1.75,  # Lightning Gun
}

def effective_hp(health: float, armor: float, armor_type: float) -> float:
    """True raw-damage budget before death, accounting for armor absorption.

    Quake armor absorbs `armor_type` fraction of each damage point until depleted.
    Two cases:
      - Armor depletes first: ehp = health + armor  (always holds when armor_type > 0)
      - Health depletes first: ehp = health / (1 - armor_type)
    Result: min of the two, floored at 1.0 to keep log defined.
    """
    if armor_type <= 0.0 or armor <= 0.0:
        return max(health, 1.0)
    ehp_armor_first = health + armor
    ehp_health_first = health / (1.0 - armor_type)
    return max(min(ehp_armor_first, ehp_health_first), 1.0)



@dataclass(slots=True)
class RewardWeights:
    mode: str = "pvp"
    death_penalty: float = -2.0
    # PvP EHP delta signal: weight * log(ehp_now / ehp_prev) per step.
    # Picking up health/armor → positive; taking damage → exactly inverse negative.
    # Diminishing returns at high EHP, amplified near death (log derivative = 1/ehp).
    # Skipped on death ticks (death_penalty covers those).
    ehp_delta_weight: float = 0.5
    # PvP EDP delta signal: weight * Σ log(opp_prev / opp_now) per step.
    # Positive when dealing damage. Slightly higher weight than ehp_delta to
    # encourage trades (equal damage exchanges are net-positive).
    # Skipped per-opponent on their death tick (frag_bonus covers those).
    # Requires custom progs.dat with T_Damage hook for opponent EHP tracking.
    edp_delta_weight: float = 0.6
    frag_bonus: float = 3.0

    def __post_init__(self) -> None:
        resolved_mode = self.mode.strip().lower() if self.mode else "pvp"
        if resolved_mode != "pvp":
            raise ValueError(f"Unsupported reward mode {self.mode!r}; only 'pvp' is supported")
        self.mode = "pvp"


def _combat_signal(combat_signals: Mapping[str, float], key: str) -> float:
    return float(combat_signals.get(key, 0.0))


def _pvp_reward_components(
    weights: RewardWeights,
    combat_signals: Mapping[str, float],
) -> Dict[str, float]:
    frag_gain = max(_combat_signal(combat_signals, "frag_gain"), 0.0)
    player_died = 1.0 if _combat_signal(combat_signals, "player_died") > 0.0 else 0.0

    health = max(_combat_signal(combat_signals, "health"), 1.0)
    armor = max(_combat_signal(combat_signals, "armor"), 0.0)
    armor_type = max(0.0, min(_combat_signal(combat_signals, "armor_type"), 0.8))
    ehp_now = effective_hp(health, armor, armor_type)
    prev_ehp = max(_combat_signal(combat_signals, "prev_ehp"), 1.0)
    # Skip EHP delta on death ticks — the jump back to spawn health is not a real gain.
    ehp_delta = 0.0 if player_died > 0.0 else weights.ehp_delta_weight * math.log(ehp_now / prev_ehp)

    # EDP delta: Σ log(opp_prev / opp_now) pre-computed by custom progs.dat.
    # 0.0 when progs.dat doesn't provide the signal (graceful fallback).
    edp_raw = _combat_signal(combat_signals, "edp_raw")
    edp_delta = weights.edp_delta_weight * edp_raw

    components = {
        "reward_frag_bonus": float(weights.frag_bonus * frag_gain),
        "reward_death_penalty": float(weights.death_penalty * player_died),
        "reward_ehp_delta": float(ehp_delta),
        "reward_edp_delta": float(edp_delta),
    }
    components["reward_total"] = float(sum(components.values()))
    return components


def reward_components(
    weights: RewardWeights,
    *,
    combat_signals: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    return _pvp_reward_components(
        weights=weights,
        combat_signals=combat_signals or {},
    )


def shaped_reward(
    weights: RewardWeights,
    *,
    combat_signals: Mapping[str, float] | None = None,
) -> float:
    return reward_components(
        weights=weights,
        combat_signals=combat_signals,
    )["reward_total"]
