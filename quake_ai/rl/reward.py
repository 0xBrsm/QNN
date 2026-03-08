"""Reward shaping for live navigation and combat-survival training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(slots=True)
class RewardWeights:
    mode: str = "navigation"
    progress_delta: float = 1.0
    combat_progress_delta: float = 0.05
    item_pickup: float = 0.05
    completion_bonus: float = 3.0
    step_penalty: float = -0.02
    timeout_penalty: float = -1.0
    stuck_penalty: float = -0.05
    survival_step_bonus: float = 0.01
    visible_threat_bonus: float = 0.002
    fire_threat_bonus: float = 0.03
    blind_fire_penalty: float = -0.01
    damage_taken_penalty: float = -0.02
    death_penalty: float = -2.0
    frag_bonus: float = 3.0
    frag_loss_penalty: float = -1.0
    monster_kill_bonus: float = 1.5
    health_gain_bonus: float = 0.01
    armor_gain_bonus: float = 0.008
    ammo_gain_bonus: float = 0.002
    weapon_pickup_bonus: float = 0.05
    weapon_switch_bonus: float = 0.01
    health_stability_bonus: float = 0.004
    armor_stability_bonus: float = 0.002


def reward_mode_from_observation(observation_format: str, explicit_mode: str = "") -> str:
    mode = explicit_mode.strip().lower()
    if mode:
        return mode
    if observation_format == "world_v2_competitive":
        return "combat_survival"
    return "navigation"


def _navigation_reward_components(
    previous_distance: float,
    new_distance: float,
    item_picked: bool,
    goal_reached: bool,
    timed_out: bool,
    stuck: bool,
    weights: RewardWeights,
) -> Dict[str, float]:
    components = {
        "reward_step_penalty": float(weights.step_penalty),
        "reward_progress": float(weights.progress_delta * (previous_distance - new_distance)),
        "reward_item_pickup": float(weights.item_pickup if item_picked else 0.0),
        "reward_completion_bonus": float(weights.completion_bonus if goal_reached else 0.0),
        "reward_timeout_penalty": float(weights.timeout_penalty if timed_out else 0.0),
        "reward_stuck_penalty": float(weights.stuck_penalty if stuck else 0.0),
    }
    components["reward_total"] = float(sum(components.values()))
    return components


def _combat_signal(combat_signals: Mapping[str, float], key: str) -> float:
    return float(combat_signals.get(key, 0.0))


def _combat_reward_components(
    previous_distance: float,
    new_distance: float,
    goal_reached: bool,
    timed_out: bool,
    stuck: bool,
    weights: RewardWeights,
    combat_signals: Mapping[str, float],
) -> Dict[str, float]:
    progress = float(previous_distance - new_distance)
    damage_taken = max(_combat_signal(combat_signals, "damage_taken"), 0.0)
    frag_gain = max(_combat_signal(combat_signals, "frag_gain"), 0.0)
    frag_loss = max(_combat_signal(combat_signals, "frag_loss"), 0.0)
    monster_kills = max(_combat_signal(combat_signals, "monster_kills"), 0.0)
    health_gain = max(_combat_signal(combat_signals, "health_gain"), 0.0)
    armor_gain = max(_combat_signal(combat_signals, "armor_gain"), 0.0)
    ammo_gain = max(_combat_signal(combat_signals, "ammo_gain"), 0.0)
    weapon_pickups = max(_combat_signal(combat_signals, "weapon_pickups"), 0.0)
    weapon_switches = max(_combat_signal(combat_signals, "weapon_switches"), 0.0)
    visible_threats = max(_combat_signal(combat_signals, "visible_threats"), 0.0)
    health_fraction = max(0.0, min(_combat_signal(combat_signals, "health_fraction"), 1.0))
    armor_fraction = max(0.0, min(_combat_signal(combat_signals, "armor_fraction"), 1.0))
    effective_fire = 1.0 if _combat_signal(combat_signals, "effective_fire") > 0.0 else 0.0
    blind_fire = 1.0 if _combat_signal(combat_signals, "blind_fire") > 0.0 else 0.0
    player_died = 1.0 if _combat_signal(combat_signals, "player_died") > 0.0 else 0.0

    components = {
        "reward_survival_bonus": float(weights.survival_step_bonus),
        "reward_progress": float(weights.combat_progress_delta * progress),
        "reward_visible_threat_bonus": float(weights.visible_threat_bonus * visible_threats),
        "reward_fire_threat_bonus": float(weights.fire_threat_bonus * effective_fire),
        "reward_blind_fire_penalty": float(weights.blind_fire_penalty * blind_fire),
        "reward_damage_taken_penalty": float(weights.damage_taken_penalty * damage_taken),
        "reward_frag_bonus": float(weights.frag_bonus * frag_gain),
        "reward_frag_loss_penalty": float(weights.frag_loss_penalty * frag_loss),
        "reward_monster_kill_bonus": float(weights.monster_kill_bonus * monster_kills),
        "reward_health_gain_bonus": float(weights.health_gain_bonus * health_gain),
        "reward_armor_gain_bonus": float(weights.armor_gain_bonus * armor_gain),
        "reward_ammo_gain_bonus": float(weights.ammo_gain_bonus * ammo_gain),
        "reward_weapon_pickup_bonus": float(weights.weapon_pickup_bonus * weapon_pickups),
        "reward_weapon_switch_bonus": float(weights.weapon_switch_bonus * weapon_switches),
        "reward_health_stability_bonus": float(weights.health_stability_bonus * health_fraction),
        "reward_armor_stability_bonus": float(weights.armor_stability_bonus * armor_fraction),
        "reward_completion_bonus": float(weights.completion_bonus if goal_reached else 0.0),
        "reward_timeout_penalty": float(weights.timeout_penalty if timed_out else 0.0),
        "reward_stuck_penalty": float(weights.stuck_penalty if stuck else 0.0),
        "reward_death_penalty": float(weights.death_penalty * player_died),
    }
    components["reward_total"] = float(sum(components.values()))
    return components


def reward_components(
    previous_distance: float,
    new_distance: float,
    item_picked: bool,
    goal_reached: bool,
    timed_out: bool,
    stuck: bool,
    weights: RewardWeights,
    *,
    combat_signals: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    if weights.mode == "combat_survival":
        return _combat_reward_components(
            previous_distance=previous_distance,
            new_distance=new_distance,
            goal_reached=goal_reached,
            timed_out=timed_out,
            stuck=stuck,
            weights=weights,
            combat_signals=combat_signals or {},
        )
    return _navigation_reward_components(
        previous_distance=previous_distance,
        new_distance=new_distance,
        item_picked=item_picked,
        goal_reached=goal_reached,
        timed_out=timed_out,
        stuck=stuck,
        weights=weights,
    )


def shaped_reward(
    previous_distance: float,
    new_distance: float,
    item_picked: bool,
    goal_reached: bool,
    timed_out: bool,
    stuck: bool,
    weights: RewardWeights,
    *,
    combat_signals: Mapping[str, float] | None = None,
) -> float:
    return reward_components(
        previous_distance=previous_distance,
        new_distance=new_distance,
        item_picked=item_picked,
        goal_reached=goal_reached,
        timed_out=timed_out,
        stuck=stuck,
        weights=weights,
        combat_signals=combat_signals,
    )["reward_total"]
