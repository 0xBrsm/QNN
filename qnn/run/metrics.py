"""Shared training and evaluation metric helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict

EPISODE_MEAN_ALIASES = {
    "frags": "frags_mean",
    "deaths": "deaths_mean",
    "suicides": "suicides_mean",
    "deaths_by_opponent": "deaths_by_opponent_mean",
    "suicide_rate": "suicide_rate_mean",
    "kd_ratio": "kd_ratio_mean",
    "damage_dealt": "episode_damage_dealt_mean",
    "damage_taken": "episode_damage_taken_mean",
    "damage_taken_self": "episode_damage_taken_self_mean",
    "damage_taken_other": "episode_damage_taken_other_mean",
    "accuracy": "accuracy",
    "hits": "episode_hit_count_mean",
    "shots_fired": "episode_shots_fired_mean",
    "damage_per_death": "damage_per_death_mean",
    "health_pickups": "health_pickups_mean",
    "armor_pickups": "armor_pickups_mean",
    "weapon_pickups": "weapon_pickups_mean",
    "blind_fire_rate": "blind_fire_rate",
    "stuck_rate": "episode_stuck_rate_mean",
    "reward_total": "reward_total_mean",
    "reward_frags": "reward_frags_mean",
    "reward_deaths": "reward_deaths_mean",
    "reward_ehp": "reward_ehp_mean",
    "reward_edp": "reward_edp_mean",
    "reward_tracking": "reward_tracking_mean",
    "tracking_cos_mean": "tracking_cos_mean",
}

EVAL_REPORT_ALIASES = {
    "frag_delta_mean": "episode_frag_delta_mean",
    "damage_dealt_mean": "episode_damage_dealt_mean",
    "hit_count_mean": "episode_hit_count_mean",
    "shots_fired_mean": "episode_shots_fired_mean",
    "stuck_rate": "episode_stuck_rate_mean",
}

PPO_REPORT_METRICS = (
    "mean_episode_return",
    "effective_game_minutes_per_wall_minute",
    "deaths_mean",
    "suicides_mean",
    "suicide_rate_mean",
    "frag_delta_mean",
    "damage_dealt_mean",
    "hit_count_mean",
    "shots_fired_mean",
    "accuracy",
    "damage_per_death_mean",
    "blind_fire_rate",
    "stuck_rate",
    "reward_total_mean",
    "reward_frags_mean",
    "reward_deaths_mean",
    "reward_ehp_mean",
    "reward_edp_mean",
    "reward_tracking_mean",
    "tracking_cos_mean",
)

EVAL_REPORT_METRICS = (
    "mean_episode_return",
    "death_rate",
    "suicides_mean",
    "deaths_by_opponent_mean",
    "suicide_rate_mean",
    "frag_delta_mean",
    "damage_dealt_mean",
    "hit_count_mean",
    "shots_fired_mean",
    "accuracy",
    "damage_per_death_mean",
    "blind_fire_rate",
    "stuck_rate",
    "reward_total_mean",
    "reward_frags_mean",
    "reward_deaths_mean",
    "reward_ehp_mean",
    "reward_edp_mean",
    "reward_tracking_mean",
    "tracking_cos_mean",
)


def effective_game_minutes_per_wall_minute(
    frames_per_second: float | int | None,
    fixed_tick_hz: float | int | None,
) -> float | None:
    """Convert training throughput into simulated game minutes per wall minute.

    Numerically this is equivalent to game-seconds-per-wall-second:
    env frames/sec divided by game ticks/sec.
    """
    if frames_per_second is None or fixed_tick_hz is None:
        return None
    fps = float(frames_per_second)
    tick_hz = float(fixed_tick_hz)
    if fps <= 0.0 or tick_hz <= 0.0:
        return None
    return fps / tick_hz


def build_episode_extra_stats(
    *,
    frags: int,
    deaths: int,
    suicides: int,
    damage_dealt: float,
    damage_taken: float,
    steps: int,
    reward_total: float,
    damage_taken_self: float = 0.0,
    damage_taken_other: float = 0.0,
    hits: int,
    shots_fired: int,
    health_pickups: float,
    armor_pickups: float,
    weapon_pickups: float,
    blind_fires: int,
    stuck_steps: int,
    reward_frags: float,
    reward_deaths: float,
    reward_ehp: float,
    reward_edp: float,
    reward_tracking: float,
    tracking_cos_sum: float,
) -> Dict[str, float]:
    """Build the canonical terminal episode statistics."""
    max_deaths = max(deaths, 1)
    max_shots = max(shots_fired, 1)
    max_steps = max(steps, 1)
    return {
        "frags": float(frags),
        "deaths": float(deaths),
        "suicides": float(suicides),
        "deaths_by_opponent": float(deaths - suicides),
        "suicide_rate": float(suicides / max_deaths),
        "kd_ratio": float(frags / max_deaths),
        "damage_dealt": float(damage_dealt),
        "damage_taken": float(damage_taken),
        "damage_taken_self": float(damage_taken_self),
        "damage_taken_other": float(damage_taken_other),
        "steps": float(steps),
        "accuracy": float(hits / max_shots),
        "hits": float(hits),
        "shots_fired": float(shots_fired),
        "damage_per_death": float(damage_dealt / max_deaths),
        "health_pickups": float(health_pickups),
        "armor_pickups": float(armor_pickups),
        "weapon_pickups": float(weapon_pickups),
        "blind_fire_rate": float(blind_fires / max_shots),
        "stuck_rate": float(stuck_steps / max_steps),
        "reward_total": float(reward_total),
        "reward_frags": float(reward_frags),
        "reward_deaths": float(reward_deaths),
        "reward_ehp": float(reward_ehp),
        "reward_edp": float(reward_edp),
        "reward_tracking": float(reward_tracking),
        "tracking_cos_mean": float(tracking_cos_sum / max_steps),
    }


@dataclass(slots=True)
class EpisodeStatAccumulator:
    """Accumulate one episode's combat and reward totals from NativeWorldEnv info."""

    frags: int = 0
    deaths: int = 0
    suicides: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    damage_taken_self: float = 0.0
    damage_taken_other: float = 0.0
    steps: int = 0
    reward_total: float = 0.0
    hits: int = 0
    shots_fired: int = 0
    health_pickups: float = 0.0
    armor_pickups: float = 0.0
    weapon_pickups: float = 0.0
    blind_fires: int = 0
    stuck_steps: int = 0
    reward_frags: float = 0.0
    reward_deaths: float = 0.0
    reward_ehp: float = 0.0
    reward_edp: float = 0.0
    reward_tracking: float = 0.0
    tracking_cos_sum: float = 0.0

    def add_step(self, *, reward: float, info: Mapping[str, object], terminal: bool) -> None:
        done_reason = str(info.get("done_reason", ""))
        self.frags += int(info.get("frag_delta", 0))
        del terminal
        player_died = (
            bool(info.get("player_died"))
            or done_reason == "player_died"
            or float(info.get("frag_loss", 0.0)) > 0.0
        )
        if player_died:
            self.deaths += 1
            if bool(info.get("player_suicide")):
                self.suicides += 1
        self.damage_dealt += float(info.get("damage_dealt", 0.0))
        self.damage_taken += float(info.get("damage_taken", 0.0))
        self.damage_taken_self += float(info.get("damage_taken_self", 0.0))
        self.damage_taken_other += float(info.get("damage_taken_other", 0.0))
        self.steps += 1
        self.reward_total += float(reward)
        self.hits += int(info.get("hit_count", 0))
        self.shots_fired += int(info.get("shots_fired", 0))
        self.health_pickups += float(info.get("health_gain", 0.0))
        self.armor_pickups += float(info.get("armor_gain", 0.0))
        self.weapon_pickups += float(info.get("weapon_pickups", 0.0))
        self.blind_fires += int(info.get("blind_fire", 0))
        self.stuck_steps += 1 if info.get("stuck") else 0
        self.reward_frags += float(info.get("reward_frag_bonus", 0.0))
        self.reward_deaths += float(info.get("reward_death_penalty", 0.0))
        self.reward_ehp += float(info.get("reward_ehp_delta", 0.0))
        self.reward_edp += float(info.get("reward_edp_delta", 0.0))
        self.reward_tracking += float(info.get("reward_tracking", 0.0))
        self.tracking_cos_sum += float(info.get("tracking_cos", 0.0))

    def as_dict(self) -> Dict[str, float]:
        return build_episode_extra_stats(
            frags=self.frags,
            deaths=self.deaths,
            suicides=self.suicides,
            damage_dealt=self.damage_dealt,
            damage_taken=self.damage_taken,
            damage_taken_self=self.damage_taken_self,
            damage_taken_other=self.damage_taken_other,
            steps=self.steps,
            reward_total=self.reward_total,
            hits=self.hits,
            shots_fired=self.shots_fired,
            health_pickups=self.health_pickups,
            armor_pickups=self.armor_pickups,
            weapon_pickups=self.weapon_pickups,
            blind_fires=self.blind_fires,
            stuck_steps=self.stuck_steps,
            reward_frags=self.reward_frags,
            reward_deaths=self.reward_deaths,
            reward_ehp=self.reward_ehp,
            reward_edp=self.reward_edp,
            reward_tracking=self.reward_tracking,
            tracking_cos_sum=self.tracking_cos_sum,
        )


def append_metric_values(store: dict[str, list[float]], metrics: Mapping[str, float]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            store.setdefault(str(key), []).append(float(value))


def mean_metric_values(store: Mapping[str, list[float]]) -> Dict[str, float]:
    return {
        key: float(sum(values) / len(values))
        for key, values in sorted(store.items())
        if values
    }


def build_eval_summary_aliases(episode_metric_means: Mapping[str, float]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for source_key, target_key in EPISODE_MEAN_ALIASES.items():
        value = episode_metric_means.get(source_key)
        if isinstance(value, (int, float)):
            summary[target_key] = float(value)
    frags_mean = episode_metric_means.get("frags")
    deaths_mean = episode_metric_means.get("deaths")
    if isinstance(frags_mean, (int, float)) and isinstance(deaths_mean, (int, float)):
        summary["episode_frag_delta_mean"] = float(frags_mean - deaths_mean)
    return summary


def report_metric_key(stage: str, key: str) -> str:
    if stage.startswith("eval"):
        return EVAL_REPORT_ALIASES.get(key, key)
    return key
