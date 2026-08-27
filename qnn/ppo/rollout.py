"""Device-resident rollout buffer + GAE for the native PPO trainer.

One synchronous on-policy window: T ticks × B lanes. Obs fields are
pre-allocated ``(T, B, …)`` tensors on the training device, filled tick
by tick from the vec env's stacked numpy obs — the learner then reruns
the model over the whole window through the SAME sequence path BC
trains through (``reset_mask``/``reset_ts`` GRU segmentation), so no
re-stacking or host↔device churn happens between collection and update.

Episode-boundary semantics (mirrors the BC lane-packed layout):
  - ``reset_mask[t, b]`` marks a frame that STARTS a new episode — the
    GRU hidden for lane b is zeroed before consuming tick t.
  - ``terminal``: engine-done (death/frag limit). No bootstrap.
  - ``truncated``: driver timeout. Bootstraps V(final_obs) — the vec env
    hands back the pre-reset final obs so the value is computed on the
    state the episode actually ended in, not the auto-reset spawn.
"""

from __future__ import annotations

from typing import Dict, Iterator, Mapping

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-shape (T, B) storage for one PPO window."""

    def __init__(
        self,
        rollout_steps: int,
        num_lanes: int,
        *,
        heads: Mapping[str, tuple[int, ...]],
        device: torch.device | str,
        hidden_dim: int,
    ) -> None:
        T, B = int(rollout_steps), int(num_lanes)
        self.rollout_steps = T
        self.num_lanes = B
        self.device = torch.device(device)

        self.obs: Dict[str, torch.Tensor] = {}
        self._obs_dtypes: Dict[str, torch.dtype] = {}

        # Per trained head: executed action, behavior log-prob, decision mask.
        self.actions = {
            h: torch.zeros((T, B, *shape), dtype=torch.long, device=self.device)
            for h, shape in heads.items()
        }
        self.log_probs = {
            h: torch.zeros((T, B), dtype=torch.float32, device=self.device)
            for h in heads
        }
        self.decision_mask = {
            h: torch.zeros((T, B), dtype=torch.bool, device=self.device)
            for h in heads
        }

        # Per-tick alignment hbw (NaN off-LOS) — input to the trigger
        # objective's p_fire matching term. NaN, not 0, so "no LOS actor"
        # cannot be confused with "perfectly aligned".
        self.align_hbw = torch.full(
            (T, B), float("nan"), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.terminal = torch.zeros((T, B), dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros((T, B), dtype=torch.bool, device=self.device)
        self.reset_mask = torch.zeros((T, B), dtype=torch.bool, device=self.device)
        self._reset_np = np.zeros((T, B), dtype=bool)
        # V(final_obs) for truncated ticks, scattered in by the collector at
        # window end; 0 elsewhere (terminal ticks bootstrap nothing).
        self.final_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)

        # GRU hidden at window start (already zeroed for lanes whose first
        # frame starts an episode). (B, hidden_dim).
        self.hidden0 = torch.zeros((B, max(hidden_dim, 0)), dtype=torch.float32, device=self.device)

        # Filled by compute_gae().
        self.advantages = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((T, B), dtype=torch.float32, device=self.device)

        self._t = 0

    # ── collection ────────────────────────────────────────────────────

    @staticmethod
    def _torch_dtype(arr: np.ndarray) -> torch.dtype:
        # Storage dtype: keep integer/quantized fields compact; float16
        # wire fields upcast to float32 once (model dequant expects floats).
        if arr.dtype == np.float16:
            return torch.float32
        return torch.from_numpy(arr[:0].copy()).dtype

    def add(
        self,
        obs: Mapping[str, np.ndarray],
        *,
        reset_mask: np.ndarray,
    ) -> int:
        """Book tick t's obs (stacked (B, …) numpy) + episode starts.

        Returns t. Actions/log-probs/values/rewards for the tick are
        filled by the collector via the public tensors (`actions[h][t]`
        etc.) — they are produced as device tensors already and a
        copy-through-numpy would be waste.
        """
        t = self._t
        if t >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer window is full; call reset()")
        for key, arr in obs.items():
            buf = self.obs.get(key)
            if buf is None:
                dtype = self._torch_dtype(arr)
                buf = torch.zeros(
                    (self.rollout_steps, *arr.shape), dtype=dtype, device=self.device
                )
                self.obs[key] = buf
            buf[t].copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(buf.dtype), non_blocking=True)
        rm = reset_mask.astype(bool)
        self.reset_mask[t].copy_(torch.from_numpy(rm), non_blocking=True)
        self._reset_np[t] = rm  # host mirror: sync-free per-minibatch reset_ts
        self._t = t + 1
        return t

    def reset(self) -> None:
        self._t = 0

    def copy_from(self, source: "RolloutBuffer") -> None:
        """Bulk-copy a complete compatible window into this buffer.

        The bounded pipeline collects into host-resident buffers so per-tick
        bookkeeping never queues behind learner kernels on the shared APU.
        Once the learner releases its single device buffer, this method stages
        the next immutable unroll in a small number of contiguous copies.
        """
        if (
            self.rollout_steps != source.rollout_steps
            or self.num_lanes != source.num_lanes
            or self.actions.keys() != source.actions.keys()
        ):
            raise ValueError("rollout buffers are not shape-compatible")
        if not source.full:
            raise RuntimeError("cannot stage an incomplete rollout buffer")

        for key, source_tensor in source.obs.items():
            target = self.obs.get(key)
            if target is None:
                target = torch.empty(
                    source_tensor.shape,
                    dtype=source_tensor.dtype,
                    device=self.device,
                )
                self.obs[key] = target
            elif target.shape != source_tensor.shape or target.dtype != source_tensor.dtype:
                raise ValueError(f"observation field {key!r} is not shape-compatible")
            target.copy_(source_tensor)
        if self.obs.keys() != source.obs.keys():
            raise ValueError("rollout buffers have different observation fields")

        for target_group, source_group in (
            (self.actions, source.actions),
            (self.log_probs, source.log_probs),
            (self.decision_mask, source.decision_mask),
        ):
            for key, target in target_group.items():
                target.copy_(source_group[key])

        for name in (
            "align_hbw",
            "values",
            "rewards",
            "terminal",
            "truncated",
            "reset_mask",
            "final_values",
            "hidden0",
            "advantages",
            "returns",
        ):
            getattr(self, name).copy_(getattr(source, name))
        self._reset_np[...] = source._reset_np
        self._t = source._t

    @property
    def full(self) -> bool:
        return self._t >= self.rollout_steps

    def reset_ts(self, lanes: torch.Tensor | None = None) -> tuple[int, ...]:
        """Host-side tick indices with any episode start — the GRU
        segmentation boundaries (Temporal's device-sync-free contract).

        ``lanes`` restricts to a minibatch's lane subset: with dense
        episode boundaries (death-terminal), the ALL-lanes boundary set
        approaches every timestep and the fused-segment GRU degenerates
        to per-tick launches — a 32-lane subset has ~4× fewer cuts.
        """
        rm = self._reset_np
        if lanes is not None:
            rm = rm[:, np.asarray(lanes.cpu())]
        return tuple(int(t) for t in np.nonzero(rm.any(axis=1))[0])

    # ── GAE ───────────────────────────────────────────────────────────

    def compute_gae(
        self,
        bootstrap_value: torch.Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """GAE(λ) over the window.

        ``bootstrap_value``: V(s_T) per lane — the value of the obs the
        NEXT window starts with (0 where the last tick was terminal;
        the final-obs value where it was truncated — both already folded
        below, callers pass the plain next-obs values).

        Next-value per tick t:
          terminal[t]  → 0
          truncated[t] → final_values[t]  (V of the pre-reset obs)
          else         → values[t+1] (or bootstrap_value at t = T−1)
        The recursion cuts at every episode end (terminal OR truncated).
        """
        T = self.rollout_steps
        values, rewards = self.values, self.rewards
        done = (self.terminal | self.truncated).float()
        adv = torch.zeros_like(rewards)
        last_adv = torch.zeros_like(bootstrap_value)
        for t in range(T - 1, -1, -1):
            next_values = values[t + 1] if t + 1 < T else bootstrap_value
            next_values = torch.where(self.truncated[t], self.final_values[t], next_values)
            next_values = torch.where(self.terminal[t], torch.zeros_like(next_values), next_values)
            nonterm = 1.0 - done[t]
            delta = rewards[t] + gamma * next_values - values[t]
            last_adv = delta + gamma * gae_lambda * nonterm * last_adv
            adv[t] = last_adv
        self.advantages = adv
        self.returns = adv + values

    # ── learner access ────────────────────────────────────────────────

    def lane_minibatches(
        self,
        minibatch_lanes: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Yield shuffled lane-index tensors (T stays contiguous per lane —
        the recurrent minibatch unit is the full window of a lane subset)."""
        B = self.num_lanes
        perm = torch.randperm(B, generator=generator)
        mb = max(1, min(int(minibatch_lanes), B))
        for start in range(0, B, mb):
            yield perm[start:start + mb].to(self.device)

    def lane_slice_obs(self, lanes: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v.index_select(1, lanes) for k, v in self.obs.items()}
