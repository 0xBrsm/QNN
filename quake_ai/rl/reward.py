"""Reward shaping for E1M1 navigation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardWeights:
    progress_delta: float = 1.0
    item_pickup: float = 0.05
    completion_bonus: float = 3.0
    step_penalty: float = -0.02
    timeout_penalty: float = -1.0
    stuck_penalty: float = -0.05
    invalid_use_penalty: float = -0.1


def shaped_reward(
    previous_distance: float,
    new_distance: float,
    item_picked: bool,
    goal_reached: bool,
    timed_out: bool,
    stuck: bool,
    used_wrong: bool,
    weights: RewardWeights,
) -> float:
    reward = weights.step_penalty
    reward += weights.progress_delta * (previous_distance - new_distance)
    if item_picked:
        reward += weights.item_pickup
    if goal_reached:
        reward += weights.completion_bonus
    if timed_out:
        reward += weights.timeout_penalty
    if stuck:
        reward += weights.stuck_penalty
    if used_wrong:
        reward += weights.invalid_use_penalty
    return float(reward)
