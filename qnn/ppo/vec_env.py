"""Vectorized Quake env for the native PPO trainer.

N ``NativeWorldEnv`` engine subprocesses driven from ONE Python process
by a thread pool — each ``step()`` blocks on pipe I/O and releases the
GIL, so the fan-out is genuinely concurrent (the proven pattern from
``NativeVectorEnv`` / ``eval_batched_forward``). No gymnasium, no
Sample Factory: observations are the native per-field dicts, actions
are the canonical engine dicts.

Differences from the retired SF ``QuakeEnv`` wrapper, all PPO-driven:
  - auto-reset returns per-lane reset flags (GRU/decode state zeroing);
  - terminal (engine done) vs truncated (driver timeout) are split, and
    the PRE-reset final obs is returned for truncation bootstrapping;
  - per-lane ``EpisodeStatAccumulator`` books episode stats internally
    and emits them on episode end.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from engine.bridge import NativeObsBufferProcess
from qnn.env.reward import RewardWeights
from qnn.env.world import NativeWorldEnv
from qnn.ppo.env_backend import EnvStepBatch, EpisodeResult
from qnn.run.metrics import EpisodeStatAccumulator
from qnn.wire import (
    OBS_BUFFER_SIZE,
    unpack_obs_buffer_native,
    unpack_obs_buffer_native_batch,
)
from qnn.schema import SPATIAL_TOKEN_COUNT  # noqa: F401  (obs contract anchor)
from qnn.vocab import MAX_TOKEN_OBJECTS

try:  # optional: only procgen scenarios need mapgen
    from mapgen.pool import PROCGEN_SENTINEL
except ImportError:  # pragma: no cover
    PROCGEN_SENTINEL = "__procgen__"

# Entity keys arrive from the wire with a variable leading dim (sized to
# the live entity_count) and must be padded to MAX_TOKEN_OBJECTS so lanes
# stack. entity_types pads with -1 (the empty-slot sentinel the model's
# mask reads); everything else zero-fills. This mirrors the eval driver's
# _pad_entities_to_max — the single obs-shape contract for batched act().
_ENTITY_PAD_FILL = {"entity_types": -1}


def pad_entities(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Pad variable-length entity_* fields to MAX_TOKEN_OBJECTS, in place."""
    if "entity_count" not in obs:
        return obs
    n = int(obs["entity_count"])
    assert 0 <= n <= MAX_TOKEN_OBJECTS, f"entity_count={n} out of range"
    for key, arr in obs.items():
        if not key.startswith("entity_") or key == "entity_count":
            continue
        if arr.ndim == 0 or arr.shape[0] == MAX_TOKEN_OBJECTS:
            continue
        fill = _ENTITY_PAD_FILL.get(key, 0)
        padded = np.full((MAX_TOKEN_OBJECTS, *arr.shape[1:]), fill, dtype=arr.dtype)
        if n > 0:
            padded[:n] = arr[:n]
        obs[key] = padded
    return obs


@dataclass
class LaneStep:
    """One lane's step result, post auto-reset."""
    obs: Dict[str, np.ndarray] | None
    reward: float
    terminal: bool
    truncated: bool
    info: Dict[str, Any]
    final_obs: Dict[str, np.ndarray] | None  # pre-reset obs when truncated


@dataclass
class _LaneState:
    scenario_id: str
    stats: EpisodeStatAccumulator = field(default_factory=EpisodeStatAccumulator)
    length: int = 0
    return_value: float = 0.0


class VecQuakeEnv:
    """Synchronous vector of N engine lanes with scenario rotation."""

    def __init__(
        self,
        *,
        num_lanes: int,
        executable: str | Path,
        basedir: str,
        workdir: str | None,
        map_id: str,
        native_args: Sequence[str],
        options: Mapping[str, Any],
        scenarios: Sequence[Mapping[str, Any]] | None,
        procgen: Mapping[str, Any] | None,
        max_steps_per_episode: int,
        fixed_tick_hz: int,
        reward_weights: RewardWeights,
        mode: str,
        seed: int,
    ) -> None:
        if num_lanes < 1:
            raise ValueError("num_lanes must be >= 1")
        self.num_lanes = int(num_lanes)
        self._lane_states: List[_LaneState] = []
        self.envs: List[NativeWorldEnv] = []

        for lane in range(self.num_lanes):
            lane_map, lane_args, lane_opts, scenario_id = self._resolve_scenario(
                lane, map_id, native_args, options, scenarios,
            )
            procgen_cfg = None
            if lane_map == PROCGEN_SENTINEL:
                if not isinstance(procgen, Mapping):
                    raise RuntimeError(
                        "Procgen training requires an explicit procgen config "
                        "with arena_size, rooms, and cleanup_generated_maps"
                    )
                procgen_cfg = {
                    "maps_dir": str(Path(basedir) / "id1" / "maps"),
                    "arena_size": int(procgen["arena_size"]),
                    "rooms": int(procgen["rooms"]),
                    "cleanup_generated_maps": bool(procgen["cleanup_generated_maps"]),
                }
            self.envs.append(NativeWorldEnv(
                executable=str(executable),
                map_id=lane_map,
                max_steps=int(max_steps_per_episode),
                fixed_tick_hz=int(fixed_tick_hz),
                reward_weights=reward_weights,
                mode=str(mode),
                seed=int(seed) + lane,
                env={"QUAKE_BASEDIR": str(basedir)},
                native_args=list(lane_args),
                options=dict(lane_opts),
                workdir=workdir or None,
                procgen=procgen_cfg,
            ))
            self._lane_states.append(_LaneState(scenario_id=scenario_id))

        self._executor = ThreadPoolExecutor(
            max_workers=self.num_lanes, thread_name_prefix="qnn-vec-env",
        )
        # lane → in-flight async reset (see step()); rows replay from
        # _prev_obs while a lane is held.
        self._pending_resets: Dict[int, Any] = {}
        self._prev_obs: Dict[str, np.ndarray] | None = None
        self._timings: Dict[str, float] = {}
        # Incremented after every successful reset. Starting at -1 makes the
        # first full reset generation 0 while preserving monotonic identity if
        # the backend is explicitly reset again later.
        self._episode_ids = np.full(self.num_lanes, -1, dtype=np.int64)
        self._inflight_sent: np.ndarray | None = None
        self._inflight_episode_ids: np.ndarray | None = None

    def reset_timings(self) -> None:
        self._timings.clear()

    def timing_snapshot(self) -> Dict[str, float]:
        return dict(self._timings)

    def _time_add(self, key: str, started: float) -> None:
        self._timings[key] = self._timings.get(key, 0.0) + (time.perf_counter() - started)

    @staticmethod
    def _resolve_scenario(
        lane: int,
        map_id: str,
        native_args: Sequence[str],
        options: Mapping[str, Any],
        scenarios: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[str, Sequence[str], Mapping[str, Any], str]:
        """Round-robin scenario assignment (lane % len), merging scenario
        options over the base — same semantics the SF wrapper had."""
        if not scenarios:
            return map_id, native_args, options, map_id
        scenario = scenarios[lane % len(scenarios)]
        lane_map = str(scenario["map_id"])
        lane_args = list(scenario.get("native_args", native_args))
        lane_opts = dict(options)
        lane_opts.update(scenario.get("options", {}))
        return lane_map, lane_args, lane_opts, str(scenario.get("scenario_id", lane_map))

    # ── stepping ──────────────────────────────────────────────────────

    @staticmethod
    def _stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = obs_list[0].keys()
        return {k: np.stack([np.asarray(o[k]) for o in obs_list], axis=0) for k in keys}

    def reset(self) -> Dict[str, np.ndarray]:
        if self._inflight_sent is not None:
            raise RuntimeError("cannot reset while an environment step is in flight")
        for fut in self._pending_resets.values():
            fut.result()  # drain in-flight lane resets before the full reset
        self._pending_resets.clear()
        self._prev_obs = None
        futures = [self._executor.submit(env.reset) for env in self.envs]
        obs_list = [pad_entities(f.result()) for f in futures]
        for state in self._lane_states:
            state.stats = EpisodeStatAccumulator()
            state.length = 0
            state.return_value = 0.0
        self._episode_ids += 1
        return self._stack_obs(obs_list)

    @staticmethod
    def _finish_lane(
        obs: Dict[str, np.ndarray] | None,
        reward: float,
        done: bool,
        info: Dict[str, object],
    ) -> LaneStep:
        # Death IS terminal (plan: GAE must cut at the death, not bootstrap
        # value through the respawn). The deathmatch engine respawns without
        # setting te.done, so enforce the episode boundary here — NOT in
        # NativeWorldEnv, where eval counts deaths as within-episode stats.
        # obs is None on the batched-drain path (lanes unpack together and
        # truncation stashes final_obs from the batch rows).
        if info.get("player_died") and not done:
            done = True
            info["done_reason"] = "player_died"
        truncated = bool(info.get("done_reason") == "timeout")
        terminal = bool(done and not truncated)
        return LaneStep(
            obs=obs,
            reward=float(reward),
            terminal=terminal,
            truncated=truncated,
            info=dict(info),
            final_obs=None,
        )

    def submit(
        self,
        action_batch: Mapping[str, np.ndarray],
    ) -> None:
        """Submit one stacked action batch without waiting for observations.

        ``action_batch`` is the ``PolicyActionBatch.actions`` dict:
        move (B,3) float, look (B,3) float, attack (B,), weapon (B,).
        Look is unit-normalized before the engine (atan2 view-delta
        mapping is scale-invariant but ill-defined near the origin —
        same rationale as the retired wrapper).
        """
        if self._inflight_sent is not None:
            raise RuntimeError("an environment step is already in flight")
        _stage = time.perf_counter()
        pending = self._pending_resets
        action_packets = NativeObsBufferProcess.pack_step_batch(
            action_batch,
            num_rows=self.num_lanes,
            normalize_look=True,
        )
        self._time_add("action_pack_s", _stage)

        # Split-phase step: send every action first so all engines sim the
        # tick CONCURRENTLY, then drain replies as RAW wire bytes and
        # unpack every lane at once (vectorized self/spatial column
        # slices; entity walk writes straight into padded (B, MAX, …)
        # rows) — no per-lane obs dicts, no pad_entities, no _stack_obs.
        #
        # Resets are ASYNCHRONOUS: an engine reset is a ~41 ms JSON round
        # trip and this step is synchronous across lanes, so a blocking
        # reset stalls every lane (~200 deaths/window made this the
        # dominant collect cost). A done lane instead submits its reset
        # to the pool and is HELD: skipped in send/drain, its row replayed
        # from the previous tick, reported hold=True + terminal=True with
        # reward 0 so the caller masks the ~1–2 filler frames per death
        # out of every policy-gradient term.
        sent = np.ones(self.num_lanes, dtype=bool)
        if pending:
            sent[np.fromiter(pending, dtype=np.int64)] = False
        _stage = time.perf_counter()
        for lane, env in enumerate(self.envs):
            if sent[lane]:
                env.step_send_packed(action_packets[lane])
        self._time_add("send_s", _stage)

        self._inflight_sent = sent
        self._inflight_episode_ids = self._episode_ids.copy()

    def receive(self) -> EnvStepBatch:
        """Drain the submitted step and return a backend-neutral batch."""
        sent = self._inflight_sent
        episode_ids = self._inflight_episode_ids
        if sent is None or episode_ids is None:
            raise RuntimeError("receive() called without a submitted environment step")

        pending = self._pending_resets
        _stage = time.perf_counter()
        raws = np.empty((self.num_lanes, OBS_BUFFER_SIZE), dtype=np.uint8)
        results: List[LaneStep | None] = [None] * self.num_lanes
        for lane, env in enumerate(self.envs):
            if not sent[lane]:
                continue
            raw, reward, done, info = env.step_recv_raw()
            raws[lane] = np.frombuffer(raw, dtype=np.uint8)
            results[lane] = self._finish_lane(None, reward, done, info)
        self._time_add("drain_s", _stage)

        # Match-scoped arenas never enter the expensive map/sign-on reset
        # path. Death replies already carry a live paired-respawn observation
        # from C. Time-limit truncations need one explicit round-reset request;
        # preserve their pre-reset row for value bootstrapping, issue all such
        # resets concurrently, then replace only those next-observation rows.
        fast_round_done = np.zeros(self.num_lanes, dtype=bool)
        truncation_raw: Dict[int, bytes] = {}
        round_reset_futures: Dict[int, Any] = {}
        for lane, result in enumerate(results):
            if result is None or not (result.terminal or result.truncated):
                continue
            if not self.envs[lane].match_round_reset:
                continue
            fast_round_done[lane] = True
            if result.truncated:
                truncation_raw[lane] = raws[lane].tobytes()
                round_reset_futures[lane] = self._executor.submit(
                    self.envs[lane].round_reset_raw
                )
            elif not result.info.get("player_died"):
                round_reset_futures[lane] = self._executor.submit(
                    self.envs[lane].round_reset_raw
                )
        for lane, future in round_reset_futures.items():
            raws[lane] = np.frombuffer(future.result(), dtype=np.uint8)

        _stage = time.perf_counter()
        obs_b = unpack_obs_buffer_native_batch(raws)
        self._time_add("unpack_s", _stage)
        _stage = time.perf_counter()
        hold = np.zeros(self.num_lanes, dtype=bool)
        # Held lanes: replay last obs; harvest any completed reset into
        # the row — the fresh obs is what the policy acts on NEXT tick.
        for lane in list(pending):
            fut = pending[lane]
            if fut.done():
                fresh = pad_entities(fut.result())
                for k, v in obs_b.items():
                    v[lane] = fresh[k]
                del pending[lane]
                self._episode_ids[lane] += 1
            elif self._prev_obs is not None:
                for k, v in obs_b.items():
                    v[lane] = self._prev_obs[k][lane]
            hold[lane] = True
        # Truncated lanes stash their pre-reset terminal obs (bootstrap
        # target) BEFORE anything overwrites those rows.
        for lane, r in enumerate(results):
            if r is not None and r.truncated:
                if lane in truncation_raw:
                    r.final_obs = pad_entities(
                        unpack_obs_buffer_native(truncation_raw[lane])
                    )
                else:
                    r.final_obs = {k: v[lane].copy() for k, v in obs_b.items()}
        # Newly-done lanes: fire the reset, don't wait for it.
        for lane, r in enumerate(results):
            if r is not None and (r.terminal or r.truncated):
                if fast_round_done[lane]:
                    self._episode_ids[lane] += 1
                else:
                    pending[lane] = self._executor.submit(self.envs[lane].reset)
        self._time_add("reset_s", _stage)

        _stage = time.perf_counter()
        episodes: List[EpisodeResult] = []
        final_rows: List[tuple[int, Dict[str, np.ndarray]]] = []
        rewards = np.zeros(self.num_lanes, dtype=np.float32)
        terminal = np.zeros(self.num_lanes, dtype=bool)
        truncated = np.zeros(self.num_lanes, dtype=bool)
        # Held filler frames: terminal (GAE cuts, nothing carries), zero
        # reward, no stats booked, no episode emitted.
        terminal[hold] = True
        for lane, r in enumerate(results):
            if r is None:
                continue
            state = self._lane_states[lane]
            done = r.terminal or r.truncated
            state.stats.add_step(reward=r.reward, info=r.info, terminal=done)
            state.length += 1
            state.return_value += r.reward
            rewards[lane] = r.reward
            terminal[lane] = r.terminal
            truncated[lane] = r.truncated
            if r.final_obs is not None:
                final_rows.append((lane, r.final_obs))
            if done:
                episodes.append(EpisodeResult(
                    lane=lane,
                    episode_id=int(episode_ids[lane]),
                    scenario_id=str(r.info.get("scenario_id", state.scenario_id)),
                    stats=dict(state.stats.as_dict()),
                    length=state.length,
                    return_value=state.return_value,
                ))
                state.stats = EpisodeStatAccumulator()
                state.length = 0
                state.return_value = 0.0
        self._time_add("stats_s", _stage)

        self._prev_obs = obs_b
        self._inflight_sent = None
        self._inflight_episode_ids = None
        return EnvStepBatch(
            env_ids=np.arange(self.num_lanes, dtype=np.int64),
            episode_ids=episode_ids,
            obs=obs_b,
            rewards=rewards,
            terminal=terminal,
            truncated=truncated,
            valid=~hold,
            final_obs_rows=final_rows,
            episodes=episodes,
        )

    def step(
        self,
        action_batch: Mapping[str, np.ndarray],
    ) -> tuple[
        Dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[tuple[int, Dict[str, np.ndarray]]],
        List[EpisodeResult],
        np.ndarray,
    ]:
        """Compatibility wrapper for synchronous non-PPO callers."""
        self.submit(action_batch)
        batch = self.receive().require_dense_lane_order(self.num_lanes)
        return (
            batch.obs,
            batch.rewards,
            batch.terminal,
            batch.truncated,
            batch.final_obs_rows,
            batch.episodes,
            batch.hold,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()
