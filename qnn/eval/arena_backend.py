"""Synchronous arena-grid adapter for batched policy evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from qnn.env.reward import RewardWeights
from qnn.ppo.arena_backend import ArenaGridBackend


class ArenaEvalPool:
    """Present grouped ``step_many`` results as eval-runner lane results.

    This backend intentionally targets the homogeneous ``qnn_arena8`` eval.
    General seeded/procgen and multi-map evaluation stays on isolated workers.
    """

    def __init__(
        self,
        *,
        num_lanes: int,
        server_executable: str | Path,
        client_executable: str | Path,
        basedir: str | Path,
        workdir: str | Path | None,
        map_id: str,
        matches_per_server: int,
        base_port: int,
        bot_skill: int,
        max_steps_per_episode: int,
        fixed_tick_hz: int,
        reward_weights: RewardWeights,
        scenario_id: str,
        scenario_ids: Sequence[str] | None = None,
        weapon_config: Mapping[str, object] | None = None,
        declarations: Sequence[object] | None = None,
    ) -> None:
        self.num_lanes = int(num_lanes)
        self._backend = ArenaGridBackend(
            num_lanes=self.num_lanes,
            server_executable=server_executable,
            client_executable=client_executable,
            basedir=basedir,
            workdir=workdir,
            map_id=map_id,
            matches_per_server=matches_per_server,
            seat_mode="bot",
            base_port=base_port,
            bot_skill=bot_skill,
            max_steps_per_episode=max_steps_per_episode,
            fixed_tick_hz=fixed_tick_hz,
            reward_weights=reward_weights,
            direct_actions=True,
            observer_mode="virtual",
            scenario_id=scenario_id,
            scenario_ids=scenario_ids,
            weapon_config=weapon_config,
            declarations=declarations,
        )
        self._obs: dict[str, np.ndarray] | None = None
        self._fresh_lanes: set[int] = set()

    @staticmethod
    def _row(obs: Mapping[str, np.ndarray], lane: int) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(value[lane]).copy()
            for key, value in obs.items()
        }

    def reset_lane(self, lane: int) -> dict[str, np.ndarray]:
        lane = int(lane)
        if not 0 <= lane < self.num_lanes:
            raise ValueError(f"eval arena lane out of range: {lane}")
        if self._obs is None:
            self._obs = self._backend.reset()
            self._fresh_lanes = set(range(self.num_lanes))
        elif lane not in self._fresh_lanes:
            self._obs = self._backend.reset_lanes((lane,))
        self._fresh_lanes.discard(lane)
        return self._row(self._obs, lane)

    def step_many(
        self,
        lanes: Sequence[int],
        actions: Sequence[Mapping[str, object]],
    ) -> list[tuple[dict[str, np.ndarray], float, bool, dict[str, object]]]:
        if self._obs is None:
            raise RuntimeError("reset at least one arena eval lane before stepping")
        if len(lanes) != len(actions):
            raise ValueError("one arena eval action is required per active lane")
        move = np.zeros((self.num_lanes, 3), dtype=np.float32)
        look = np.zeros((self.num_lanes, 3), dtype=np.float32)
        look[:, 0] = 1.0
        attack = np.zeros(self.num_lanes, dtype=np.int64)
        for lane, action in zip(lanes, actions, strict=True):
            lane = int(lane)
            move[lane] = np.asarray(action.get("move", (0.0, 0.0, 0.0)), dtype=np.float32)
            look[lane] = np.asarray(action.get("look", (1.0, 0.0, 0.0)), dtype=np.float32)
            attack[lane] = int(action.get("attack", 0))
        batch = self._backend.step_many(
            {"move": move, "look": look, "attack": attack}
        )
        self._obs = batch.obs
        return [
            (
                self._row(batch.obs, int(lane)),
                float(batch.rewards[int(lane)]),
                bool(batch.terminal[int(lane)] or batch.truncated[int(lane)]),
                dict(batch.infos[int(lane)]),
            )
            for lane in lanes
        ]

    def close(self) -> None:
        self._backend.close()
