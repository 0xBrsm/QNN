"""Grouped eight-match environment backend for native PPO."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engine.arena_bridge import ArenaGroupProcess
from engine.bridge import NativeObsBufferProcess
from engine.training_protocol import TrustedTrainingExtrasV1
from qnn.env.reward import RewardWeights
from qnn.env.world import NativeWorldEnv
from qnn.ppo.arena import ArenaTopology
from qnn.ppo.env_backend import EnvStepBatch, EpisodeResult
from qnn.run.metrics import EpisodeStatAccumulator
from qnn.wire import (
    OBS_BUFFER_SIZE,
    unpack_obs_buffer_native,
    unpack_obs_buffer_native_batch,
)


@dataclass
class _ArenaLaneState:
    """Duck-typed state for NativeWorldEnv's canonical QTRN bookkeeping."""

    scenario_id: str
    max_steps: int
    map_id: str
    options: dict[str, object]
    match_round_reset: bool = True
    _steps: int = 0
    _frags: int = 0
    stats: EpisodeStatAccumulator = field(default_factory=EpisodeStatAccumulator)
    length: int = 0
    return_value: float = 0.0

    def book(
        self,
        raw: bytes,
        extras: TrustedTrainingExtrasV1,
    ) -> tuple[bytes, float, bool, dict[str, object]]:
        # Keep reward attribution and episode metrics identical to process
        # lanes without spawning a throwaway NativeWorldEnv.
        result = NativeWorldEnv._book_step(self, raw, extras)  # type: ignore[arg-type]
        if result[2] and self.match_round_reset:
            self._steps = 0
            self._frags = 0
        return result

    def reset_episode(self) -> None:
        self._steps = 0
        self._frags = 0


class ArenaGridBackend:
    """Dense PPO backend that shares one Quake world across 8× 1v1s.

    Bot mode exposes one learner trajectory per match.  Self-play mode exposes
    both seats, preserving their roles in :attr:`topology` so a later opponent
    policy router can replace seat-1 actions without changing engine transport.
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
        seat_mode: str,
        base_port: int,
        bot_skill: int,
        max_steps_per_episode: int,
        fixed_tick_hz: int,
        reward_weights: RewardWeights,
        direct_actions: bool,
        observer_mode: str = "external",
        scenario_id: str = "arena-grid-1v1",
        scenario_ids: Sequence[str] | None = None,
        weapon_config: Mapping[str, object] | None = None,
    ) -> None:
        if int(fixed_tick_hz) != 20:
            raise ValueError("arena_grid currently requires fixed_tick_hz=20")
        self.topology = ArenaTopology.build(
            num_lanes=int(num_lanes),
            matches_per_server=int(matches_per_server),
            seat_mode=str(seat_mode),
        )
        self.num_lanes = self.topology.num_lanes
        self._timings: dict[str, float] = {}
        self._started = False
        self._closed = False
        self._inflight: dict[int, Future[list[tuple[bytes, TrustedTrainingExtrasV1]]]] | None = None
        self._inflight_episode_ids: np.ndarray | None = None
        self._latest_raws: np.ndarray | None = None

        reward_json = json.dumps(
            {"reward_weights": asdict(reward_weights)}, separators=(",", ":")
        )
        process_env = {"QUAKE_BASEDIR": str(Path(basedir).resolve())}
        self._server_env_ids: list[np.ndarray] = []
        self._groups: list[ArenaGroupProcess] = []
        for server_id in range(self.topology.server_count):
            seats = tuple(
                seat
                for seat in self.topology.external_seats
                if seat.server_id == server_id
            )
            env_ids = np.asarray([int(seat.env_id) for seat in seats], dtype=np.int64)
            port = int(base_port) + server_id
            if not 1024 <= port <= 65535:
                raise ValueError(f"arena server port out of range: {port}")
            self._server_env_ids.append(env_ids)
            self._groups.append(ArenaGroupProcess(
                server_executable=server_executable,
                client_executable=client_executable,
                port=port,
                map_id=str(map_id),
                external_seats=[(seat.match_id, seat.seat_id) for seat in seats],
                bot_count=0 if seat_mode == "self_play" else int(matches_per_server),
                bot_skill=int(bot_skill),
                self_play=seat_mode == "self_play",
                reward_json=reward_json,
                direct_actions=bool(direct_actions),
                observer_mode=str(observer_mode),
                env=process_env,
                workdir=workdir,
                weapon_config=weapon_config,
            ))

        self._executor = ThreadPoolExecutor(
            max_workers=self.topology.server_count,
            thread_name_prefix="qnn-arena-group",
        )
        # Per-lane scenario ids let a single homogeneous arena bucket its lanes
        # into distinct eval cells (the aim-grid packs one swept decode value per
        # lane, all sharing one weapon/map). PPO leaves scenario_ids None and every
        # lane shares the broadcast scenario_id.
        if scenario_ids is None:
            lane_scenario_ids = [str(scenario_id)] * self.num_lanes
        else:
            lane_scenario_ids = [str(sid) for sid in scenario_ids]
            if len(lane_scenario_ids) != self.num_lanes:
                raise ValueError(
                    f"scenario_ids has {len(lane_scenario_ids)} entries but the "
                    f"arena has {self.num_lanes} lanes")
        self._states = [
            _ArenaLaneState(
                scenario_id=lane_scenario_ids[lane],
                max_steps=int(max_steps_per_episode),
                map_id=str(map_id),
                options={"scenario_id": lane_scenario_ids[lane]},
            )
            for lane in range(self.num_lanes)
        ]
        self._episode_ids = np.full(self.num_lanes, -1, dtype=np.int64)

    def reset_timings(self) -> None:
        self._timings.clear()

    def timing_snapshot(self) -> dict[str, float]:
        return dict(self._timings)

    def _time_add(self, key: str, started: float) -> None:
        self._timings[key] = self._timings.get(key, 0.0) + (time.perf_counter() - started)

    @staticmethod
    def _raw_batch(raws: np.ndarray) -> dict[str, np.ndarray]:
        return unpack_obs_buffer_native_batch(raws)

    def _scatter_group_transitions(
        self,
        transitions_by_server: Mapping[
            int, Sequence[tuple[bytes, TrustedTrainingExtrasV1]]
        ],
    ) -> tuple[np.ndarray, list[TrustedTrainingExtrasV1]]:
        """Restore server-grouped transitions to dense environment order."""
        raws = np.empty((self.num_lanes, OBS_BUFFER_SIZE), dtype=np.uint8)
        extras: list[TrustedTrainingExtrasV1 | None] = [None] * self.num_lanes
        for server_id, transitions in transitions_by_server.items():
            env_ids = self._server_env_ids[server_id]
            if len(transitions) != len(env_ids):
                raise RuntimeError("arena group returned the wrong number of seats")
            for env_id, (raw, training) in zip(env_ids, transitions, strict=True):
                raws[int(env_id)] = np.frombuffer(raw, dtype=np.uint8)
                extras[int(env_id)] = training
        if any(value is None for value in extras):
            raise RuntimeError("arena group omitted a trajectory")
        return raws, [value for value in extras if value is not None]

    def _collect_group_transitions(
        self,
        futures: Mapping[int, Future[list[tuple[bytes, TrustedTrainingExtrasV1]]]],
    ) -> tuple[np.ndarray, list[TrustedTrainingExtrasV1]]:
        return self._scatter_group_transitions(
            {server_id: future.result() for server_id, future in futures.items()}
        )

    def reset(self) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("arena backend is closed")
        if self._inflight is not None:
            raise RuntimeError("cannot reset while an arena step is in flight")
        started = time.perf_counter()
        if not self._started:
            # Stock NetQuake's reliable sign-on stream is serialized within a
            # world, and concurrently filling multiple 16-client self-play
            # worlds can starve one handshake until its 120 s deadline.  This
            # is one-time startup, not rollout work: admit complete groups in
            # deterministic order, then retain the thread pool for parallel
            # steady-state stepping and match resets.
            initial = {
                server_id: group.start()
                for server_id, group in enumerate(self._groups)
            }
            self._started = True
            raws, _ = self._scatter_group_transitions(initial)
        else:
            full_mask = (1 << self.topology.matches_per_server) - 1
            futures = {
                server_id: self._executor.submit(group.reset_matches, full_mask)
                for server_id, group in enumerate(self._groups)
            }
            raws, _ = self._collect_group_transitions(futures)
        self._latest_raws = raws.copy()
        for state in self._states:
            state.reset_episode()
            state.stats = EpisodeStatAccumulator()
            state.length = 0
            state.return_value = 0.0
        self._episode_ids += 1
        self._time_add("reset_s", started)
        return self._raw_batch(raws)

    def reset_lanes(self, env_ids: Sequence[int]) -> dict[str, np.ndarray]:
        """Reset the matches owning ``env_ids`` and return the dense live state.

        This is primarily the evaluation adapter's episode-boundary primitive.
        A self-play pair maps to one match bit and is reset atomically.
        """
        if not self._started or self._latest_raws is None:
            raise RuntimeError("call reset() before reset_lanes()")
        if self._inflight is not None:
            raise RuntimeError("cannot reset lanes while an arena step is in flight")
        masks: dict[int, int] = {}
        selected_matches: set[tuple[int, int]] = set()
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            if not 0 <= env_id < self.num_lanes:
                raise ValueError(f"arena env id out of range: {env_id}")
            seat = self.topology.seat_for_env(env_id)
            masks[seat.server_id] = masks.get(seat.server_id, 0) | (1 << seat.match_id)
            selected_matches.add((seat.server_id, seat.match_id))
        if not masks:
            return self._raw_batch(self._latest_raws.copy())

        futures = {
            server_id: self._executor.submit(
                self._groups[server_id].reset_matches, mask
            )
            for server_id, mask in masks.items()
        }
        for server_id, future in futures.items():
            transitions = future.result()
            for env_id, (raw, _training) in zip(
                self._server_env_ids[server_id], transitions, strict=True
            ):
                self._latest_raws[int(env_id)] = np.frombuffer(raw, dtype=np.uint8)
        for env_id, state in enumerate(self._states):
            seat = self.topology.seat_for_env(env_id)
            if (seat.server_id, seat.match_id) in selected_matches:
                state.reset_episode()
                state.stats = EpisodeStatAccumulator()
                state.length = 0
                state.return_value = 0.0
        return self._raw_batch(self._latest_raws.copy())

    def submit(self, action_batch: Mapping[str, np.ndarray]) -> None:
        if not self._started:
            raise RuntimeError("call reset() before submit()")
        if self._inflight is not None:
            raise RuntimeError("an arena step is already in flight")
        started = time.perf_counter()
        packets = NativeObsBufferProcess.pack_step_batch(
            action_batch,
            num_rows=self.num_lanes,
            normalize_look=True,
        )
        self._time_add("action_pack_s", started)
        started = time.perf_counter()
        self._inflight = {
            server_id: self._executor.submit(
                group.step_many,
                [packets[int(env_id)] for env_id in self._server_env_ids[server_id]],
            )
            for server_id, group in enumerate(self._groups)
        }
        self._inflight_episode_ids = self._episode_ids.copy()
        self._time_add("submit_s", started)

    def step_many(self, action_batch: Mapping[str, np.ndarray]) -> EnvStepBatch:
        """Synchronous grouped step for evaluation and simple rollout drivers."""
        self.submit(action_batch)
        return self.receive()

    def receive(self) -> EnvStepBatch:
        futures = self._inflight
        episode_ids = self._inflight_episode_ids
        if futures is None or episode_ids is None:
            raise RuntimeError("receive() called without a submitted arena step")

        started = time.perf_counter()
        raws, extras = self._collect_group_transitions(futures)
        self._time_add("drain_s", started)
        self._inflight = None
        self._inflight_episode_ids = None

        rewards = np.zeros(self.num_lanes, dtype=np.float32)
        terminal = np.zeros(self.num_lanes, dtype=bool)
        truncated = np.zeros(self.num_lanes, dtype=bool)
        infos: list[dict[str, object]] = []
        reset_masks: dict[int, int] = {}
        final_raws: dict[int, bytes] = {}
        started = time.perf_counter()
        for env_id, (raw_row, training) in enumerate(zip(raws, extras, strict=True)):
            _, reward, done, info = self._states[env_id].book(
                raw_row.tobytes(), training
            )
            is_truncated = bool(done and info.get("done_reason") == "timeout")
            rewards[env_id] = float(reward)
            terminal[env_id] = bool(done and not is_truncated)
            truncated[env_id] = is_truncated
            infos.append(info)
            if is_truncated:
                seat = self.topology.seat_for_env(env_id)
                reset_masks[seat.server_id] = (
                    reset_masks.get(seat.server_id, 0) | (1 << seat.match_id)
                )
                final_raws[env_id] = raw_row.tobytes()

        # Timeout is a match boundary, so reset both seats exactly once.  The
        # engine returns a live spawn observation for every external seat.
        if reset_masks:
            reset_futures = {
                server_id: self._executor.submit(
                    self._groups[server_id].reset_matches, mask
                )
                for server_id, mask in reset_masks.items()
            }
            for server_id, future in reset_futures.items():
                transitions = future.result()
                for env_id, (raw, _) in zip(
                    self._server_env_ids[server_id], transitions, strict=True
                ):
                    raws[int(env_id)] = np.frombuffer(raw, dtype=np.uint8)
        self._time_add("book_s", started)

        started = time.perf_counter()
        obs = self._raw_batch(raws)
        self._latest_raws = raws.copy()
        final_obs_rows = [
            (env_id, unpack_obs_buffer_native(raw))
            for env_id, raw in final_raws.items()
        ]
        self._time_add("unpack_s", started)

        episodes: list[EpisodeResult] = []
        started = time.perf_counter()
        for env_id, info in enumerate(infos):
            state = self._states[env_id]
            done = bool(terminal[env_id] or truncated[env_id])
            state.stats.add_step(
                reward=float(rewards[env_id]), info=info, terminal=done
            )
            state.length += 1
            state.return_value += float(rewards[env_id])
            if done:
                episodes.append(EpisodeResult(
                    lane=env_id,
                    episode_id=int(episode_ids[env_id]),
                    scenario_id=str(info.get("scenario_id", state.scenario_id)),
                    stats=dict(state.stats.as_dict()),
                    length=state.length,
                    return_value=state.return_value,
                ))
                state.stats = EpisodeStatAccumulator()
                state.length = 0
                state.return_value = 0.0
                state.reset_episode()
                self._episode_ids[env_id] += 1
        self._time_add("stats_s", started)

        return EnvStepBatch(
            env_ids=np.arange(self.num_lanes, dtype=np.int64),
            episode_ids=episode_ids,
            obs=obs,
            rewards=rewards,
            terminal=terminal,
            truncated=truncated,
            valid=np.ones(self.num_lanes, dtype=bool),
            final_obs_rows=final_obs_rows,
            episodes=episodes,
            infos=infos,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inflight is not None:
            for future in self._inflight.values():
                with suppress(Exception):
                    future.result()
            self._inflight = None
        for group in self._groups:
            group.close()
        self._executor.shutdown(wait=True)
