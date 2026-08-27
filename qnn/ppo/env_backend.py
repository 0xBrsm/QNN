"""Backend-neutral environment contract for native PPO rollout collection.

The collector deliberately knows nothing about subprocesses, pipes, or the
eventual in-process multi-world engine.  Backends expose a split phase:

``submit(actions)``
    Accept one action for each environment and start advancing them.

``receive()``
    Return the completed transitions as an :class:`EnvStepBatch`.

The current collector still consumes one dense, lane-ordered batch per tick.
Keeping environment and episode-generation IDs in the result makes that
assumption explicit and gives the future ready-queue collector enough identity
to accept sparse/out-of-order completions without changing the rollout ABI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Protocol, runtime_checkable

import numpy as np


ObsBatch = Dict[str, np.ndarray]
FinalObsRows = list[tuple[int, Dict[str, np.ndarray]]]


@dataclass
class EpisodeResult:
    """Completed episode metadata emitted by an environment backend."""

    lane: int
    episode_id: int
    scenario_id: str
    stats: Dict[str, float]
    length: int
    return_value: float


@dataclass
class EnvStepBatch:
    """Transitions completed by an environment backend.

    ``env_ids`` identifies the recurrent/decode-state row to update.
    ``episode_ids`` is a monotonically increasing generation per environment;
    together they prevent late completions from being attached to a reset
    episode. ``valid`` is false for compatibility filler rows while an engine
    reset is in flight.
    """

    env_ids: np.ndarray
    episode_ids: np.ndarray
    obs: ObsBatch
    rewards: np.ndarray
    terminal: np.ndarray
    truncated: np.ndarray
    valid: np.ndarray
    final_obs_rows: FinalObsRows
    episodes: list[EpisodeResult]
    # Per-lane alignment hbw on the PRE-step obs for EVERY LOS lane (NaN
    # where no LOS actor resolves) — the p_fire denominator the rung-3 trigger
    # objective is fit against. Computed rollout-side by qnn.ppo.align_hbw so
    # the learner never re-derives the law. None = backend does not supply it.
    align_hbw: np.ndarray | None = None
    infos: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.env_ids = np.asarray(self.env_ids, dtype=np.int64)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.terminal = np.asarray(self.terminal, dtype=bool)
        self.truncated = np.asarray(self.truncated, dtype=bool)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.align_hbw is not None:
            self.align_hbw = np.asarray(self.align_hbw, dtype=np.float32)

        if self.env_ids.ndim != 1:
            raise ValueError("env_ids must be a 1-D array")
        count = int(self.env_ids.shape[0])
        for name, values in (
            ("episode_ids", self.episode_ids),
            ("rewards", self.rewards),
            ("terminal", self.terminal),
            ("truncated", self.truncated),
            ("valid", self.valid),
        ) + ((("align_hbw", self.align_hbw),) if self.align_hbw is not None else ()):
            if values.shape != (count,):
                raise ValueError(
                    f"{name} must have shape ({count},), got {values.shape}"
                )
        if isinstance(self.obs, list):
            # Heterogeneous-seat form (per-lane obs dicts — cross-arch eval):
            # one dict per lane, no dense field-major stacking possible.
            if len(self.obs) != count:
                raise ValueError(
                    f"obs list must contain {count} lane dicts, got {len(self.obs)}")
        else:
            for name, values in self.obs.items():
                if np.asarray(values).ndim < 1 or int(np.asarray(values).shape[0]) != count:
                    raise ValueError(
                        f"obs[{name!r}] must have leading dimension {count}, "
                        f"got {np.asarray(values).shape}"
                    )
        if self.infos and len(self.infos) != count:
            raise ValueError(f"infos must contain {count} rows, got {len(self.infos)}")

    @property
    def hold(self) -> np.ndarray:
        """Compatibility view for the current reset-filler mask."""
        return ~self.valid

    def require_dense_lane_order(self, num_lanes: int) -> "EnvStepBatch":
        """Validate the current collector's one-result-per-lane contract.

        The backend result itself supports arbitrary environment IDs.  Only
        the synchronous collector imposes this temporary dense-order gate.
        """
        expected = np.arange(int(num_lanes), dtype=np.int64)
        if not np.array_equal(self.env_ids, expected):
            raise RuntimeError(
                "synchronous rollout collector requires one transition per "
                "lane in lane order"
            )
        return self


@runtime_checkable
class RolloutEnvBackend(Protocol):
    """Split-phase backend consumed by :class:`RolloutCollector`."""

    num_lanes: int

    def reset(self) -> ObsBatch:
        """Reset every environment and return one observation per lane."""

    def submit(self, action_batch: Mapping[str, np.ndarray]) -> None:
        """Start one environment step for the submitted action rows."""

    def receive(self) -> EnvStepBatch:
        """Wait for and return the transitions from the submitted step."""

    def close(self) -> None:
        """Release backend resources."""

    def reset_timings(self) -> None:
        """Reset optional backend profiling counters."""

    def timing_snapshot(self) -> Dict[str, float]:
        """Return optional backend profiling counters."""
