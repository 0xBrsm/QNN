"""Deterministic symbolic navigation environment derived from map features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

from quake_ai.actions import ACTION_HEADS
from quake_ai.navigation import (
    build_observation,
    desired_motion_vector,
    load_navigation_map,
    region_to_point,
    select_neighbor,
)
from quake_ai.rl.reward import RewardWeights, shaped_reward


@dataclass(slots=True)
class EnvState:
    region_id: int
    steps: int
    heading: int
    prev_distance: float
    items_collected: set[int]
    goal_reached: bool
    player_vel: Tuple[float, float, float]


class E1M1NavigationEnv:
    def __init__(
        self,
        map_features_path: str | Path,
        max_steps: int = 256,
        reward_weights: RewardWeights | None = None,
        seed: int = 7,
    ) -> None:
        self.map_features_path = str(map_features_path)
        self.max_steps = max_steps
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)
        self.nav_map = load_navigation_map(map_features_path)
        self.state: EnvState | None = None

    def _observation(self, state: EnvState) -> np.ndarray:
        player_pos = region_to_point(state.region_id)
        nearby = [1, 0, 0, 0] if state.region_id in self.nav_map.item_regions and state.region_id not in state.items_collected else [0, 0, 0, 0]
        return build_observation(
            nav_map=self.nav_map,
            region_id=state.region_id,
            heading=state.heading,
            player_pos=player_pos,
            player_vel=state.player_vel,
            nearby_item_flags=nearby,
            goal_progress=self.nav_map.goal_progress(state.region_id),
        )

    def reset(self, seed: int | None = None, start_variant: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if start_variant is None:
            start_region = int(self.rng.choice(self.nav_map.spawn_regions))
            heading = int(self.rng.integers(0, 8))
        else:
            start_region = self.nav_map.spawn_regions[start_variant % len(self.nav_map.spawn_regions)]
            heading = int(start_variant % 8)

        self.state = EnvState(
            region_id=start_region,
            steps=0,
            heading=heading,
            prev_distance=self.nav_map.distance(start_region),
            items_collected=set(),
            goal_reached=False,
            player_vel=(0.0, 0.0, 0.0),
        )
        return self._observation(self.state)

    def _select_next_region(self, region_id: int, heading: int, action: Mapping[str, int]) -> int:
        desired = desired_motion_vector(
            heading=heading,
            move=int(action.get("move", 0)),
            strafe=int(action.get("strafe", 0)),
        )
        return select_neighbor(self.nav_map, region_id, desired)

    def step(self, action: Mapping[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, float | int | bool]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        state = self.state
        state.steps += 1
        start_region = state.region_id

        turn = int(action.get("turn", 0))
        if turn == 1:
            state.heading = (state.heading + 1) % 8
        elif turn == 2:
            state.heading = (state.heading - 1) % 8

        next_region = self._select_next_region(state.region_id, state.heading, action)
        state.region_id = next_region

        prev_point = np.array(region_to_point(start_region), dtype=np.float32)
        next_point = np.array(region_to_point(state.region_id), dtype=np.float32)
        velocity = (next_point - prev_point) / 5.0
        state.player_vel = (float(velocity[0]), float(velocity[1]), float(velocity[2]))

        item_picked = False
        if state.region_id in self.nav_map.item_regions and state.region_id not in state.items_collected:
            state.items_collected.add(state.region_id)
            item_picked = True

        at_goal = state.region_id in self.nav_map.goal_regions
        used_goal = at_goal and int(action.get("use", 0)) == 1
        used_wrong = (not at_goal) and int(action.get("use", 0)) == 1
        state.goal_reached = used_goal

        timed_out = state.steps >= self.max_steps
        done = state.goal_reached or timed_out

        new_distance = self.nav_map.distance(state.region_id)
        stuck = state.region_id == start_region

        reward = shaped_reward(
            previous_distance=state.prev_distance,
            new_distance=new_distance,
            item_picked=item_picked,
            goal_reached=state.goal_reached,
            timed_out=timed_out,
            stuck=stuck,
            used_wrong=used_wrong,
            weights=self.reward_weights,
        )

        state.prev_distance = new_distance

        obs = self._observation(state)
        info: Dict[str, float | int | bool] = {
            "at_goal": at_goal,
            "goal_reached": state.goal_reached,
            "items_collected": len(state.items_collected),
            "steps": state.steps,
            "distance_to_goal": new_distance,
            "goal_progress": self.nav_map.goal_progress(state.region_id),
            "stuck": stuck,
            "used_goal": used_goal,
            "used_wrong": used_wrong,
        }
        return obs, reward, done, info


class VectorNavigationEnv:
    """Simple synchronous vector wrapper (num_envs workers)."""

    def __init__(self, num_envs: int, map_features_path: str | Path, max_steps: int, seed: int) -> None:
        self.envs = [
            E1M1NavigationEnv(map_features_path=map_features_path, max_steps=max_steps, seed=seed + i)
            for i in range(num_envs)
        ]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs], axis=0)

    def step(self, action_batch: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float | int | bool]]]:
        next_obs = []
        rewards = []
        dones = []
        infos: List[Dict[str, float | int | bool]] = []

        for idx, env in enumerate(self.envs):
            action = {head: int(action_batch[head][idx]) for head in ACTION_HEADS}
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset(seed=None)
            next_obs.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)

        return (
            np.stack(next_obs, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )
