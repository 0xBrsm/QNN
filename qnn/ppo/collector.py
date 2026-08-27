"""Synchronous rollout collection with adapter-owned raw-head sampling.

``policy.act`` supplies the ordinary deploy-decoded action plus raw logits.
Active PPO adapters replace only the engine fields they own; every frozen
field keeps the policy's normal sampled decode.  The collector has no model
generation or head-structure branches.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from qnn.model.decode import BatchedRNG
from qnn.ppo.distributions import HeadDistribution
from qnn.ppo.env_backend import EpisodeResult, RolloutEnvBackend
from qnn.ppo.learner import ValueHead, value_features
from qnn.ppo.rollout import RolloutBuffer

class RolloutCollector:
    """Owns the recurrent/decode state across windows and fills a
    RolloutBuffer per collect() call."""

    def __init__(
        self,
        policy: Any,
        value_head: ValueHead,
        vec_env: RolloutEnvBackend,
        adapters: Mapping[str, HeadDistribution],
        *,
        device: torch.device | str,
        seed: int,
        initial_obs: Dict[str, np.ndarray] | None = None,
        temperatures: Mapping[str, float] | None = None,
        sample_temperatures: Mapping[str, float] | None = None,
        value_af: bool = False,
    ) -> None:
        self.policy = policy
        self.value_head = value_head
        self.value_af = bool(value_af)
        self.vec_env = vec_env
        self.adapters = dict(adapters)
        self.device = torch.device(device)
        # τ for the trained heads' raw sampling (the RL policy definition —
        # the learner recomputes with the SAME τ).
        self.rl_temperatures = {h: float((temperatures or {}).get(h, 1.0)) for h in adapters}
        # τ for frozen fields' ordinary deploy decode.
        self.sample_temperatures = dict(sample_temperatures or {})

        B = vec_env.num_lanes
        # PPO keeps one fixed-width collector topology for the run, so one
        # batched RNG stream is sufficient for deterministic replay. Eval's
        # per-row generators remain supported by policy.act.
        self.row_generators = BatchedRNG.seeded(seed, B, self.device)

        self.hidden: torch.Tensor | None = None
        # Commitment-decode state threading (a25/a28): prepare_act_state is
        # the single-source contract — per-lane arrays act() mutates IN
        # PLACE, re-initialized to their reset lanes on episode boundaries
        # (same lifecycle as the GRU hidden zeroing below). Older graphs
        # return {} and act() runs stateless, unchanged.
        self._act_state = policy.prepare_act_state(B)
        self._act_state_reset = {
            k: v[0].copy() for k, v in self._act_state.items()
        }
        # initial_obs lets the orchestrator reset once, size the value head
        # from a probe forward, and hand the same obs here.
        self.obs = initial_obs if initial_obs is not None else vec_env.reset()
        self.reset_flags = np.ones(B, dtype=bool)
        self.total_env_steps = 0

    # ── helpers ───────────────────────────────────────────────────────

    def _values_for(
        self, obs: Mapping[str, np.ndarray], hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        """V(s) for a stacked obs batch — forward + value head, grad-free."""
        with torch.inference_mode(), self.policy._autocast():
            # Truncation stashes form variable-width batches. The collection
            # model is compiled for fixed B=num_lanes action inference; route
            # bootstrap forwards through the original eager Module.forward so
            # rare batch widths do not trigger dynamic compile storms.
            features, _, _, _, _ = self.policy._forward_tensors(
                dict(obs), hidden=hidden, eager_model=True,
            )
            return self.value_head(
                value_features(features, obs, self.value_af)
            ).float()

    def _ensure_hidden(self, B: int) -> torch.Tensor | None:
        if not self.policy.use_gru:
            return None
        if self.hidden is None:
            d_gru = int(getattr(self.policy.model, "d_gru", 0))
            self.hidden = torch.zeros((B, d_gru), dtype=torch.float32, device=self.device)
        return self.hidden

    # ── collection ────────────────────────────────────────────────────

    def collect(self, buffer: RolloutBuffer) -> Dict[str, Any]:
        """Fill one T×B window; returns collection stats."""
        B = self.vec_env.num_lanes
        T = buffer.rollout_steps
        buffer.reset()
        reset_env_timings = getattr(self.vec_env, "reset_timings", None)
        if reset_env_timings is not None:
            reset_env_timings()
        episodes: List[EpisodeResult] = []
        # (t, lane) → (final_obs, hidden_row) for truncation bootstraps.
        final_stash: List[tuple[int, int, Dict[str, np.ndarray], torch.Tensor | None]] = []

        hidden = self._ensure_hidden(B)
        buffer.hidden0.copy_(
            hidden if hidden is not None
            else torch.zeros((B, 0), device=self.device)
        )

        act_s = 0.0  # forward+decode+value+booking (the GPU-side slice)
        env_s = 0.0  # thread-pooled engine stepping
        buffer_s = 0.0
        book_s = 0.0
        for t in range(T):
            _t0 = time.perf_counter()
            buffer.add(self.obs, reset_mask=self.reset_flags)
            buffer_s += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            if self._act_state and self.reset_flags.any():
                # Fresh episodes restart their commitment machines exactly
                # like the ONNX state_loopback memset / eval row re-init.
                for lane in np.flatnonzero(self.reset_flags):
                    for k, template in self._act_state_reset.items():
                        self._act_state[k][lane] = template
            with self.policy._autocast():
                batch, extras = self.policy.act(
                    dict(self.obs),
                    mode="sampled",
                    hidden=hidden,
                    masks={"reset_mask": self.reset_flags},
                    row_generators=self.row_generators,
                    sample_temperatures=self.sample_temperatures,
                    rl_extras=True,
                    **self._act_state,
                )
            with torch.inference_mode():
                buffer.values[t].copy_(self.value_head(
                    value_features(extras["features"], self.obs, self.value_af)
                ).float())

            self._sample_heads(buffer, t, batch, extras)

            act_s += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            self.vec_env.submit(batch.actions)
            step = self.vec_env.receive().require_dense_lane_order(B)
            env_s += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            # Held lanes (async engine reset in flight) booked a filler
            # frame: replayed obs, zero reward, terminal. Mask it out of
            # every head's surrogate — only the value loss sees it.
            if step.hold.any():
                not_held = ~torch.from_numpy(step.hold).to(self.device)
                for head in self.adapters:
                    buffer.decision_mask[head][t] &= not_held
            if step.align_hbw is not None:
                buffer.align_hbw[t].copy_(torch.from_numpy(step.align_hbw))
            buffer.rewards[t].copy_(torch.from_numpy(step.rewards))
            buffer.terminal[t].copy_(torch.from_numpy(step.terminal))
            buffer.truncated[t].copy_(torch.from_numpy(step.truncated))
            episodes.extend(step.episodes)

            next_hidden = batch.next_hidden
            for lane, fobs in step.final_obs_rows:
                final_stash.append((
                    t, lane, fobs,
                    None if next_hidden is None else next_hidden[lane].clone(),
                ))

            done = step.terminal | step.truncated
            if done.any():
                done_t = torch.from_numpy(done).to(self.device)
                if next_hidden is not None and next_hidden.numel():
                    next_hidden = next_hidden.clone()
                    next_hidden[done_t] = 0.0
            hidden = next_hidden if self.policy.use_gru else None
            self.hidden = hidden
            self.reset_flags = done
            self.obs = step.obs
            self.total_env_steps += B
            book_s += time.perf_counter() - _t0

        _bootstrap_t0 = time.perf_counter()
        # Truncation bootstraps: V(final_obs) with the hidden the episode
        # actually ended with, one batched forward for the whole window.
        if final_stash:
            stacked = {
                k: np.stack([f[2][k] for f in final_stash], axis=0)
                for k in final_stash[0][2]
            }
            h = (
                torch.stack([f[3] for f in final_stash], dim=0)
                if final_stash[0][3] is not None else None
            )
            v_final = self._values_for(stacked, h)
            for i, (t, lane, _, _) in enumerate(final_stash):
                buffer.final_values[t, lane] = v_final[i]

        # Next-window bootstrap: V of the obs the next window starts with.
        bootstrap = self._values_for(self.obs, hidden)
        # Lanes that just reset start a fresh episode — GAE already cut the
        # recursion at the done tick, so what remains multiplies by zero;
        # still, zero them for cleanliness.
        bootstrap = bootstrap * (
            ~torch.from_numpy(self.reset_flags).to(bootstrap.device)
        ).float()
        bootstrap_s = time.perf_counter() - _bootstrap_t0

        stats: Dict[str, Any] = {
            "episodes": episodes,
            # GAE runs on the buffer's (learner) device; collection may be
            # on a CPU replica (collect_device) — land the bootstrap there.
            "bootstrap_value": bootstrap.to(buffer.rewards.device),
            "env_steps": T * B,
            "act_s": act_s,
            "env_s": env_s,
            "buffer_s": buffer_s,
            "book_s": book_s,
            "bootstrap_s": bootstrap_s,
        }
        env_timing = getattr(self.vec_env, "timing_snapshot", None)
        if env_timing is not None:
            stats.update({f"env_{k}": v for k, v in env_timing().items()})
        return stats

    def _sample_heads(
        self,
        buffer: RolloutBuffer,
        t: int,
        batch: Any,
        extras: Mapping[str, Any],
    ) -> None:
        logits = extras["logits"]
        for head, dist in self.adapters.items():
            temperature = self.rl_temperatures.get(head, 1.0)
            actions, mask = dist.collect(
                logits,
                batch.actions,
                temperature=temperature,
                row_generators=self.row_generators,
            )
            lp, _ = dist.log_prob_entropy(
                logits, actions, temperature=temperature,
            )
            buffer.actions[head][t].copy_(actions)
            buffer.log_probs[head][t].copy_(lp.float())
            buffer.decision_mask[head][t].copy_(mask)
