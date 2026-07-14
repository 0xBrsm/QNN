"""Recurrent PPO update on the native model — the learner half of the
single-process trainer.

The recompute runs the model over the whole (T, B) window through the
SAME sequence path BC trains through — ``reset_mask``/``reset_ts`` GRU
segmentation, bf16 autocast, and (when enabled by the orchestrator)
``compile_bc_hot_path``'s compiled regions — so learner throughput is
BC-step throughput, not a new code path.

Per-head clipped surrogate, masked by each head's decision ticks (the
op-filter analog); one shared value function (trainer-owned MLP over
``features``, never part of the deploy graph); approx-KL early stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from qnn.ppo.distributions import HeadDistribution


class ValueHead(nn.Module):
    """V(s) over the model's motor feature vector (``features`` — the
    first return of ``Network.forward``). Trainer-owned: lives only in
    the RL resume checkpoint, never in deploy checkpoints or ONNX."""

    def __init__(self, in_dim: int, d_hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(d_hidden)),
            nn.GELU(),
            nn.Linear(int(d_hidden), 1),
        )
        # Near-zero init on the output layer: the first updates should be
        # advantage-driven, not dominated by a randomly-initialized V.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


@dataclass
class PPOUpdateConfig:
    clip_ratio: float = 0.2
    ppo_epochs: int = 3
    minibatch_lanes: int = 16
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    kl_target: float = 0.02          # early-stop threshold on approx KL
    normalize_advantage: bool = True
    rl_head_weights: Dict[str, float] = field(default_factory=dict)
    entropy_coef: Dict[str, float] = field(default_factory=dict)
    temperatures: Dict[str, float] = field(default_factory=dict)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum()
    if int(n) == 0:
        return x.sum() * 0.0
    return (x * mask.float()).sum() / n.float()


def ppo_update(
    policy: Any,
    value_head: ValueHead,
    buffer: Any,
    adapters: Mapping[str, HeadDistribution],
    optimizer: torch.optim.Optimizer,
    cfg: PPOUpdateConfig,
    *,
    mb_generator: torch.Generator | None = None,
) -> Dict[str, float]:
    """One PPO update over a full rollout window. Returns flat metrics."""
    model = policy.model
    was_training = model.training
    model.train()

    # Advantage normalization over the union of decision ticks — the rows
    # that actually contribute surrogate terms.
    adv = buffer.advantages
    if cfg.normalize_advantage:
        union = torch.zeros_like(buffer.terminal)
        for h in adapters:
            union |= buffer.decision_mask[h]
        mean = _masked_mean(adv, union)
        var = _masked_mean((adv - mean) ** 2, union)
        adv = (adv - mean) / (var.sqrt() + 1e-8)

    stats: Dict[str, list[float]] = {}

    def _log(key: str, value: torch.Tensor | float) -> None:
        if torch.is_tensor(value):
            value = value.detach()
        stats.setdefault(key, []).append(float(value))

    stop = False
    epochs_run = 0
    for _epoch in range(int(cfg.ppo_epochs)):
        if stop:
            break
        epochs_run += 1
        for lanes in buffer.lane_minibatches(cfg.minibatch_lanes, mb_generator):
            obs_mb = buffer.lane_slice_obs(lanes)
            hidden0 = buffer.hidden0.index_select(0, lanes)
            reset_mb = buffer.reset_mask.index_select(1, lanes)
            reset_ts = buffer.reset_ts(lanes)
            with policy._autocast():
                # The dequant adapters are written for flat (B, …) obs (BC
                # pre-dequantizes at preload; act() runs flat) — flatten the
                # (T, b) window through them, then restore the seq layout the
                # model's own _flatten_obs keys on.
                T = buffer.rollout_steps
                b = int(lanes.shape[0])
                flat = {
                    k: v.reshape(T * b, *v.shape[2:]) for k, v in obs_mb.items()
                }
                deq = policy._obs_tensors_dequant(flat)
                obs_t = {k: v.reshape(T, b, *v.shape[1:]) for k, v in deq.items()}
                features, logits, _, _, _ = model(
                    obs_t, hidden0, reset_mask=reset_mb, reset_ts=reset_ts,
                )
            values = value_head(features)

            adv_mb = adv.index_select(1, lanes)
            ret_mb = buffer.returns.index_select(1, lanes)
            val_old = buffer.values.index_select(1, lanes)

            policy_loss = values.new_zeros(())
            entropy_loss = values.new_zeros(())
            kl_max = 0.0
            for h, dist in adapters.items():
                actions = buffer.actions[h].index_select(1, lanes)
                lp_old = buffer.log_probs[h].index_select(1, lanes)
                mask = buffer.decision_mask[h].index_select(1, lanes)
                lp_new, entropy = dist.log_prob_entropy(
                    logits, actions,
                    temperature=float(cfg.temperatures.get(h, 1.0)),
                )
                log_ratio = lp_new - lp_old
                ratio = log_ratio.exp()
                clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                surrogate = torch.min(ratio * adv_mb, clipped * adv_mb)
                w = float(cfg.rl_head_weights.get(h, 1.0))
                policy_loss = policy_loss - w * _masked_mean(surrogate, mask)
                ent_c = float(cfg.entropy_coef.get(h, 0.0))
                if ent_c:
                    entropy_loss = entropy_loss - ent_c * _masked_mean(entropy, mask)
                with torch.no_grad():
                    approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)
                    clipfrac = _masked_mean(
                        ((ratio - 1.0).abs() > cfg.clip_ratio).float(), mask,
                    )
                kl_max = max(kl_max, float(approx_kl))
                _log(f"kl/{h}", approx_kl)
                _log(f"clipfrac/{h}", clipfrac)
                _log(f"entropy/{h}", _masked_mean(entropy, mask))

            # Clipped value loss (PPO2 form) against the collection values.
            v_clip = val_old + (values - val_old).clamp(-cfg.clip_ratio, cfg.clip_ratio)
            value_loss = 0.5 * torch.max(
                (values - ret_mb) ** 2, (v_clip - ret_mb) ** 2,
            ).mean()

            loss = policy_loss + cfg.value_coef * value_loss + entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for g in optimizer.param_groups for p in g["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            _log("loss/policy", policy_loss)
            _log("loss/value", value_loss)
            _log("loss/total", loss)
            _log("grad_norm", grad_norm)

            if cfg.kl_target > 0 and kl_max > cfg.kl_target:
                stop = True
                break

    if not was_training:
        model.eval()

    out = {k: sum(v) / len(v) for k, v in stats.items() if v}
    out["ppo_epochs_run"] = float(epochs_run)
    with torch.no_grad():
        var_ret = buffer.returns.var()
        out["explained_variance"] = float(
            1.0 - (buffer.returns - buffer.values).var() / (var_ret + 1e-8)
        )
    return out
