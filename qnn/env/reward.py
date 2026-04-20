"""Reward shaping for training. Weights come from the run's config/reward.json."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

# Relative pickup value by Quake weapon_id (1=Axe … 8=LG).
WEAPON_TIER_VALUES: Dict[int, float] = {
    1: 0.0,   # Axe
    2: 0.5,   # Shotgun
    3: 1.0,   # Super Shotgun
    4: 0.75,  # Nailgun
    5: 1.25,  # Super Nailgun
    6: 1.5,   # Grenade Launcher
    7: 2.0,   # Rocket Launcher
    8: 1.75,  # Lightning Gun
}


@dataclass(slots=True)
class RewardWeights:
    death_penalty: float
    ehp_delta_weight: float
    edp_delta_weight: float
    frag_bonus: float
    fire_penalty: float
    self_damage_penalty: float
    tracking_weight: float
    tracking_fov: float      # full FOV in degrees; cosine boundary = cos(fov/2)
    tracking_penalty: bool    # if False, clamp tracking to non-negative (reward only)

    @classmethod
    def from_json(cls, path: str | Path) -> "RewardWeights":
        """Load weights from a flat reward.json file."""
        with open(path) as f:
            cfg = json.load(f)
        if not isinstance(cfg, Mapping):
            raise RuntimeError(f"reward config must be a JSON object: {path}")
        return cls(
            death_penalty=float(cfg["death_penalty"]),
            ehp_delta_weight=float(cfg["ehp_delta_weight"]),
            edp_delta_weight=float(cfg["edp_delta_weight"]),
            frag_bonus=float(cfg["frag_bonus"]),
            fire_penalty=float(cfg["fire_penalty"]),
            self_damage_penalty=float(cfg["self_damage_penalty"]),
            tracking_weight=float(cfg["tracking_weight"]),
            tracking_fov=float(cfg["tracking_fov"]),
            tracking_penalty=bool(cfg["tracking_penalty"]),
        )


def effective_hp(health: float, armor: float, armor_type: float) -> float:
    """True raw-damage budget before death, accounting for armor absorption."""
    if armor_type <= 0.0 or armor <= 0.0:
        return max(health, 1.0)
    ehp_armor_first = health + armor
    ehp_health_first = health / (1.0 - armor_type)
    return max(min(ehp_armor_first, ehp_health_first), 1.0)


def _combat_signal(combat_signals: Mapping[str, float], key: str) -> float:
    return float(combat_signals.get(key, 0.0))


def _remap_tracking(cos: float, fov_deg: float) -> float:
    """Remap raw tracking cosine through an FOV-aware linear curve.

    Returns +1 at dead center, 0 at the FOV boundary, -1 at 180° behind.
    ``fov_deg`` is the full field-of-view in degrees (e.g. 90 means ±45°).
    A value of 360 means the entire sphere is rewarded: +1 at dead center,
    0 only at 180° behind, and never negative unless ``fov_deg < 360`` and
    ``tracking_penalty`` is enabled.
    """
    half_rad = math.radians(min(fov_deg, 360.0) * 0.5)
    boundary = math.cos(half_rad)  # cosine at the FOV edge
    if cos >= boundary:
        # Inside FOV: remap [boundary, 1] → [0, +1]
        return (cos - boundary) / max(1.0 - boundary, 1e-8)
    else:
        # Outside FOV: remap [-1, boundary] → [-1, 0]
        return -(boundary - cos) / max(boundary + 1.0, 1e-8)


def reward_components(
    weights: RewardWeights,
    *,
    combat_signals: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    signals = combat_signals or {}
    frag_gain = max(_combat_signal(signals, "frag_gain"), 0.0)
    player_died = 1.0 if _combat_signal(signals, "player_died") > 0.0 else 0.0

    health = max(_combat_signal(signals, "health"), 1.0)
    armor = max(_combat_signal(signals, "armor"), 0.0)
    armor_type = max(0.0, min(_combat_signal(signals, "armor_type"), 0.8))
    ehp_now = effective_hp(health, armor, armor_type)
    prev_ehp = max(_combat_signal(signals, "prev_ehp"), 1.0)
    ehp_delta = 0.0 if player_died > 0.0 else weights.ehp_delta_weight * math.log(ehp_now / prev_ehp)

    edp_raw = _combat_signal(signals, "edp_raw")
    edp_delta = weights.edp_delta_weight * edp_raw

    fired = 1.0 if _combat_signal(signals, "shots_fired") > 0.0 else 0.0
    fire_pen = weights.fire_penalty * fired

    self_damage = _combat_signal(signals, "damage_dealt_self")
    self_damage_pen = weights.self_damage_penalty * self_damage

    # Tracking: cosine of angle between player aim and nearest enemy.
    # Remapped so the FOV boundary maps to 0: positive inside, negative outside.
    tracking_cos = _combat_signal(signals, "tracking_cos")
    remapped = _remap_tracking(tracking_cos, weights.tracking_fov)
    if not weights.tracking_penalty:
        remapped = max(remapped, 0.0)
    tracking = weights.tracking_weight * remapped

    components = {
        "reward_frag_bonus": float(weights.frag_bonus * frag_gain),
        "reward_death_penalty": float(weights.death_penalty * player_died),
        "reward_ehp_delta": float(ehp_delta),
        "reward_edp_delta": float(edp_delta),
        "reward_fire_penalty": float(fire_pen),
        "reward_self_damage_penalty": float(self_damage_pen),
        "reward_tracking": float(tracking),
    }
    components["reward_total"] = float(sum(components.values()))
    return components


def shaped_reward(
    weights: RewardWeights,
    *,
    combat_signals: Mapping[str, float] | None = None,
) -> float:
    return reward_components(weights=weights, combat_signals=combat_signals)["reward_total"]
