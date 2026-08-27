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

import numpy as np
import torch
import torch.nn as nn

from qnn.ppo.distributions import HeadDistribution
from qnn.vocab import self_weapon_id_to_impulse


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


def value_features(
    features: torch.Tensor,
    obs: Mapping[str, Any] | None,
    value_af: bool,
) -> torch.Tensor:
    """Assemble the value head's input vector.

    With ``value_af``, append the remaining-refire scalar
    (``self_arsenal_scalars[..., 0:1]``, /TIME_SCALE — the same field
    ``Network._weapon_feasibility_mask`` reads) so V(s) can value a
    cooldown tick below a ready tick. This is the value-function half of
    the crest-ceiling two-blindness diagnosis
    (agents/plans/crest-ceiling-handoff.md "Constraint"): ``af`` may reach
    the VALUE head (trainer-owned, never exported) but must never become a
    policy input — the a25 closed-loop regression stands.
    """
    if not value_af:
        return features
    if obs is None:
        raise RuntimeError(
            "value_af=True but no obs were provided to build the value "
            "head input — every V(s) call site must pass its obs batch"
        )
    if "self_arsenal_scalars" in obs:
        # Dequantized model obs (the learner's _dequant_window path).
        af = obs["self_arsenal_scalars"][..., 0:1]
        if not torch.is_tensor(af):
            af = torch.as_tensor(np.asarray(af))
    elif "attack_finished" in obs:
        # Native wire obs (the collector's per-step and bootstrap paths):
        # attack_finished is seconds on the wire; apply the SAME /TIME_SCALE
        # normalization dequant applies so both paths feed V(s) one unit.
        from qnn import engine_norm as en
        af = obs["attack_finished"]
        if not torch.is_tensor(af):
            af = torch.as_tensor(np.asarray(af))
        af = af.float()[..., None] / float(en.TIME_SCALE)
    else:
        raise RuntimeError(
            "value_af=True but the obs batch carries neither "
            "'self_arsenal_scalars' (dequantized) nor 'attack_finished' "
            "(native wire) — cannot build the value head's refire input"
        )
    return torch.cat(
        [features, af.to(device=features.device, dtype=features.dtype)],
        dim=-1,
    )


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
    # Anchor-KL fine-tune (fire-at-alignment rung 3): per-head coefficient on
    # KL(π_θ ‖ π_anchor) where the anchor is the FROZEN seed checkpoint —
    # a different object from kl_target's behavior-policy early stop.
    anchor_kl_coef: Dict[str, float] = field(default_factory=dict)
    # Give V(s) the remaining-refire scalar (value_features). Trainer-only:
    # widens the ValueHead input by 1; the policy never sees it.
    value_af: bool = False
    # Rung-3 trigger objective: match P(attack | alignment) to the human
    # curve. coef 0 (or target None) leaves the loss byte-identical.
    pfire_coef: float = 0.0
    pfire_target: Any = None
    # Marginal human fire occupancy on engaged ticks.  This is a constraint
    # loss on the same action probability PPO samples, never reward shaping.
    fire_occupancy_coef: float = 0.0
    fire_occupancy_target: Any = None
    fire_occupancy_temperature: float = 1.0
    fire_occupancy_project: bool = False
    fire_occupancy_project_max_delta: float = 0.0
    # Per-weapon analogue of fire_occupancy_target/project, for the general
    # 9-way PPO adapter (crest-finetune-allweapons iteration 2). Mapping of
    # weapon abbreviation -> qnn.ppo.pfire_target.FireOccupancyProjectionTarget.
    # Consumed only when the active "attack" adapter has no fixed impulse
    # (see _project_fire_occupancy_per_weapon); ignored for fixed-weapon runs,
    # which keep using fire_occupancy_target/_project_fire_occupancy.
    fire_occupancy_projection_targets: Any = None


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum()
    if int(n) == 0:
        return x.sum() * 0.0
    return (x * mask.float()).sum() / n.float()



# Refire-ready threshold — the SAME constant the model's own feasibility mask
# uses (qnn.model.network.Network._FEAS_AF_READY) and the same one the human
# curve was measured with. One number, three consumers.
_FEAS_AF_READY = 1e-4


def _pfire_loss(
    attack_dist: HeadDistribution,
    logits: Mapping[str, torch.Tensor],
    obs_t: Dict[str, torch.Tensor],
    target: torch.Tensor,             # (T, b) human p_fire, NaN off-LOS
    decision_mask: torch.Tensor,      # (T, b)
    coef: float,
    temperature: float,
    log: Any,
) -> torch.Tensor:
    """Binary cross-entropy between P(attack | s) and the human target.

    Probability comes from the active attack adapter.  In a fixed-weapon run
    this is the exact model-bias + decode-offset Bernoulli PPO samples, rather
    than the unrelated nine-way marginal used by the original rung-3 loss.

    Masked to ticks that are in LOS (finite target), refire-READY, and a real
    decision. The readiness gate matters: the head's feasibility mask forces
    no_attack during cooldown, so scoring those ticks would ask the policy for
    behaviour it is structurally unable to produce.
    """
    q = attack_dist.fire_probability(logits, temperature).float().clamp(
        1e-6, 1.0 - 1e-6,
    )

    af = obs_t["self_arsenal_scalars"][..., 0]          # (T, b) cooldown left
    ready = af <= _FEAS_AF_READY
    in_los = torch.isfinite(target)
    mask = in_los & ready & decision_mask
    # Split the mask so an empty one names its own cause in one run instead of
    # costing a smoke cycle per hypothesis.
    log("pfire/frac_los", in_los.float().mean())
    log("pfire/frac_ready", ready.float().mean())
    log("pfire/frac_decision", decision_mask.float().mean())
    if not bool(mask.any()):
        # Log the zero rather than returning silently: an all-NaN alignment
        # stream (backend not emitting align_hbw) is otherwise indistinguishable
        # from "objective is on and converged", and cost a smoke cycle once.
        log("pfire/frac_scored", 0.0)
        return q.new_zeros(())

    t = torch.nan_to_num(target, nan=0.0)
    bce = -(t * torch.log(q) + (1.0 - t) * torch.log1p(-q))
    loss = _masked_mean(bce, mask)
    log("pfire/loss", loss.detach())
    log("pfire/q_mean", _masked_mean(q.detach(), mask))
    log("pfire/target_mean", _masked_mean(t, mask))
    log("pfire/frac_scored", mask.float().mean())
    return coef * loss


def _fire_occupancy_loss(
    attack_dist: HeadDistribution,
    logits: Mapping[str, torch.Tensor],
    align_hbw: torch.Tensor,
    decision_mask: torch.Tensor,
    target_probability: float,
    coef: float,
    temperature: float,
    log: Any,
) -> torch.Tensor:
    """Match marginal fire occupancy on engaged decision ticks.

    The model feasibility mask already drives fire probability to zero during
    cooldown, so scoring all finite-alignment ticks reproduces decode-fit's
    fires/engaged-ticks denominator.  BCE on the minibatch mean is minimized
    exactly at the human occupancy while leaving the conditional alignment
    shape to ``_pfire_loss`` and PPO's crest objective.

    POPULATION CONTRACT: ``mask`` below is ``isfinite(align_hbw)`` — every
    PURE-LOS tick, no ``target_probs`` engagement label. ``target_probability``
    MUST be measured on that identical population
    (``qnn.ppo.pfire_target.FireOccupancyTarget``'s
    ``target_rate_per_s_los_engaged`` field, sourced from
    ``qnn.decode_fit.human_refs.family_aimed_rates_los`` /
    ``qnn.human.blind_fire`` — never ``qnn.human.op_attack``'s narrower
    LOS+labeled-engaged rate, which is ~1.72x smaller on SG+SSG and was
    historically fed to this exact mask; agents/plans/blind-fire-cadence.md).
    This function cannot assert the caller's population choice — it only
    sees a float — so the check lives at the target's construction site.
    """
    q = attack_dist.fire_probability(logits, temperature).float()
    mask = torch.isfinite(align_hbw) & decision_mask
    log("fire_occupancy/frac_engaged", mask.float().mean())
    if not bool(mask.any()):
        log("fire_occupancy/frac_scored", 0.0)
        return q.new_zeros(())
    q_mean = _masked_mean(q, mask).clamp(1e-6, 1.0 - 1e-6)
    target = q_mean.new_tensor(float(target_probability)).clamp(
        1e-6, 1.0 - 1e-6,
    )
    loss = -(target * torch.log(q_mean) + (1.0 - target) * torch.log1p(-q_mean))
    log("fire_occupancy/loss", loss.detach())
    log("fire_occupancy/q_mean", q_mean.detach())
    log("fire_occupancy/target", target.detach())
    log("fire_occupancy/error", (q_mean - target).detach())
    log("fire_occupancy/frac_scored", mask.float().mean())
    return float(coef) * loss


def _dequant_window(
    policy: Any, buffer: Any, lanes: torch.Tensor,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Any]:
    """Dequantized (T, b) obs window + recurrent state for a lane subset.

    The dequant adapters are written for flat (B, …) obs (BC pre-dequantizes
    at preload; act() runs flat) — flatten the (T, b) window through them,
    then restore the seq layout the model's own _flatten_obs keys on."""
    obs_mb = buffer.lane_slice_obs(lanes)
    hidden0 = buffer.hidden0.index_select(0, lanes)
    reset_mb = buffer.reset_mask.index_select(1, lanes)
    reset_ts = buffer.reset_ts(lanes)
    T = buffer.rollout_steps
    b = int(lanes.shape[0])
    flat = {k: v.reshape(T * b, *v.shape[2:]) for k, v in obs_mb.items()}
    deq = policy._obs_tensors_dequant(flat)
    obs_t = {k: v.reshape(T, b, *v.shape[1:]) for k, v in deq.items()}
    return obs_t, hidden0, reset_mb, reset_ts


def _anchor_window_logits(
    policy: Any,
    anchor_model: Any,
    buffer: Any,
    keys: tuple[str, ...],
    chunk_lanes: int,
) -> Dict[str, torch.Tensor]:
    """Frozen-anchor logits for the whole (T, B) window, keyed by logit slot.

    The anchor never changes within an update, so one grad-free pass per
    window (in minibatch-sized lane chunks) replaces a per-epoch recompute.
    Runs from the SAME hidden0/reset segmentation as the learner recompute —
    exact for frozen-trunk mode; for full-network mode the anchor's true
    hidden trajectory is unknowable and window-start state is the standard
    approximation."""
    B = buffer.num_lanes
    store: Dict[str, torch.Tensor] = {}
    was_training = anchor_model.training
    anchor_model.eval()
    with torch.no_grad(), policy._autocast():
        for start in range(0, B, max(int(chunk_lanes), 1)):
            lanes = torch.arange(
                start, min(start + max(int(chunk_lanes), 1), B),
                device=buffer.rewards.device,
            )
            obs_t, hidden0, reset_mb, reset_ts = _dequant_window(
                policy, buffer, lanes,
            )
            _, logits, _, _, _ = anchor_model(
                obs_t, hidden0, reset_mask=reset_mb, reset_ts=reset_ts,
            )
            for k in keys:
                if k not in logits:
                    raise RuntimeError(
                        f"anchor forward produced no {k!r} logits — the anchor "
                        f"checkpoint's graph must match the trained policy's"
                    )
                dst = store.setdefault(
                    k,
                    torch.empty(
                        (buffer.rollout_steps, B, *logits[k].shape[2:]),
                        dtype=torch.float32,
                        device=logits[k].device,
                    ),
                )
                dst.index_copy_(1, lanes, logits[k].float())
    if was_training:
        anchor_model.train()
    return store


def _project_fire_occupancy(
    policy: Any,
    buffer: Any,
    attack_dist: HeadDistribution,
    target_probability: float,
    temperature: float,
    chunk_lanes: int,
    max_delta: float,
) -> dict[str, float]:
    """Project the model-owned fixed-weapon intercept onto human occupancy.

    For fixed conditional logits, marginal occupancy is monotone in the one
    family intercept.  A bisection projection solves that constraint directly
    after PPO reshapes timing, avoiding coefficient tuning and grad clipping.
    """
    impulse = getattr(attack_dist, "impulse", None)
    fire_bias = getattr(getattr(policy.model, "attack_head", None),
                        "fire_bias", None)
    if impulse is None or not isinstance(fire_bias, nn.Parameter):
        raise RuntimeError(
            "fire occupancy projection requires fixed-weapon PPO and "
            "attack_head.fire_bias"
        )
    model = policy.model
    was_training = model.training
    model.eval()
    margins: list[torch.Tensor] = []
    with torch.no_grad(), policy._autocast():
        for start in range(0, buffer.num_lanes, max(int(chunk_lanes), 1)):
            lanes = torch.arange(
                start, min(start + max(int(chunk_lanes), 1), buffer.num_lanes),
                device=buffer.rewards.device,
            )
            obs_t, hidden0, reset_mb, reset_ts = _dequant_window(
                policy, buffer, lanes,
            )
            _, logits, _, _, _ = model(
                obs_t, hidden0, reset_mask=reset_mb, reset_ts=reset_ts,
            )
            mask = (
                torch.isfinite(buffer.align_hbw.index_select(1, lanes))
                & buffer.decision_mask["attack"].index_select(1, lanes)
            )
            if bool(mask.any()):
                margins.append(attack_dist.fire_logit(logits).float()[mask])
    if was_training:
        model.train()
    if not margins:
        return {"fire_occupancy/project_frac_scored": 0.0}
    margin = torch.cat(margins)
    tau = max(float(temperature), 1e-6)
    before = torch.sigmoid(margin / tau).mean()
    lo, hi = margin.new_tensor(-20.0), margin.new_tensor(20.0)
    target = margin.new_tensor(float(target_probability))
    for _ in range(48):
        mid = (lo + hi) * 0.5
        q = torch.sigmoid((margin + mid) / tau).mean()
        lo, hi = (mid, hi) if q < target else (lo, mid)
    solved_delta = (lo + hi) * 0.5
    delta = solved_delta
    if float(max_delta) > 0.0:
        delta = delta.clamp(-float(max_delta), float(max_delta))
    with torch.no_grad():
        fire_bias[int(impulse) - 1].add_(delta.to(fire_bias.dtype))
    after = torch.sigmoid((margin + delta) / tau).mean()
    return {
        "fire_occupancy/q_mean": float(after),
        "fire_occupancy/target": float(target),
        "fire_occupancy/project_before": float(before),
        "fire_occupancy/project_delta": float(delta),
        "fire_occupancy/project_solved_delta": float(solved_delta),
        "fire_occupancy/project_frac_scored": float(
            margin.numel() / buffer.align_hbw.numel()
        ),
    }


def _project_fire_occupancy_per_weapon(
    policy: Any,
    buffer: Any,
    attack_dist: HeadDistribution,
    targets: Mapping[str, Any],
    temperature: float,
    chunk_lanes: int,
    max_delta: float,
) -> dict[str, float]:
    """Per-weapon analogue of :func:`_project_fire_occupancy` for the
    general 9-way PPO adapter (crest-finetune-allweapons iteration 2's
    regulator).

    Iteration 1's crest fine-tune reward-hacked because rung3k's regulator
    only ever bisected ONE ``fire_bias`` slot (a fixed-weapon PPO arm's
    pinned impulse) and was disabled for the multi-weapon arm entirely. This
    generalizes it: for each weapon named in ``targets``, bisect
    ``attack_head.fire_bias[impulse - 1]`` so mean P(attack) over that
    weapon's own EQUIPPED, engaged decision ticks matches the weapon's target —
    the exact same bounded bisection as the fixed-weapon case, just run once
    per weapon over its own tick population instead of once over the whole
    buffer.

    EQUIPPED WEAPON comes from the rollout buffer's ``self_weapon_id`` obs,
    converted via ``qnn.vocab.self_weapon_id_to_impulse`` (never a raw
    arithmetic offset — the ENTITY_IDS encoding is NOT the impulse byte;
    see that function's docstring for the historical bug this guards).
    ``self_weapon_id`` is trainer-side bookkeeping the PPO env backend
    carries alongside the model's real inputs (qnn.ppo.vec_env keeps the
    byte the engine already writes instead of stripping it the way the
    a27+ combat model's own obs contract does) — a buffer that predates
    this plumbing, or a backend that never carries it (e.g. arena_grid),
    fails loud below rather than silently scoring the wrong ticks.

    QUANTITY: ``attack_dist.fire_probability``'s marginal P(any attack) =
    1 - P(no-attack), expressed here as ``sigmoid(margin)`` where
    ``margin`` is weapon w's class logit (biased by w's CURRENT
    ``fire_bias`` slot, mirroring ``_project_fire_occupancy``'s ``margin``
    already including ``model_bias``) minus the no-attack class logit —
    algebraically identical to feeding a logits dict with only column w
    perturbed through ``fire_probability``, but without rebuilding a (N, 9)
    softmax every bisection step. This equals P(class w) exactly whenever
    every OTHER weapon class is infeasible on w's equipped ticks. Iteration-2's
    ammo-isolated cells make that true by construction (feasibility_mask
    forces every non-pinned weapon's logit to -inf), so P(any attack) ==
    P(class w) here — but that is an assumption ABOUT THE CELL DESIGN, not
    something these tensors prove on their own (an ammo-out fallback fire
    to a different owned weapon, e.g. the ~6.6% SG fallback the LG cell
    measured, would attribute that tick's fire to whichever weapon it was
    actually EQUIPPED on, which is correct, but a cell that leaks feasibility
    more broadly would silently overcount P(any attack) as P(class w)).
    Callers should watch ``project/<w>/frac_scored`` and world-results
    fallback-rate diagnostics, not just this loss.

    Respects ``max_delta`` per weapon per call (same bounded-step contract
    as the fixed-weapon projection). Weapons in ``targets`` with zero
    matching ticks this iteration are skipped: logged at
    ``frac_scored=0``, ``fire_bias`` left untouched.
    """
    fire_bias = getattr(getattr(policy.model, "attack_head", None),
                        "fire_bias", None)
    if not isinstance(fire_bias, nn.Parameter):
        raise RuntimeError(
            "per-weapon fire occupancy projection requires "
            "attack_head.fire_bias"
        )
    if not targets:
        raise RuntimeError(
            "fire_occupancy_projection_targets is empty — the per-weapon "
            "human cadence rulers must be named explicitly"
        )
    head_key = attack_dist.logits_keys[0]
    model = policy.model
    was_training = model.training
    model.eval()
    raw_rows: list[torch.Tensor] = []
    held_impulses: list[torch.Tensor] = []
    onset_candidates: list[torch.Tensor] = []
    with torch.no_grad(), policy._autocast():
        for start in range(0, buffer.num_lanes, max(int(chunk_lanes), 1)):
            lanes = torch.arange(
                start, min(start + max(int(chunk_lanes), 1), buffer.num_lanes),
                device=buffer.rewards.device,
            )
            obs_t, hidden0, reset_mb, reset_ts = _dequant_window(
                policy, buffer, lanes,
            )
            if "self_weapon_id" not in obs_t:
                raise RuntimeError(
                    "per-weapon fire occupancy projection requires "
                    "'self_weapon_id' in the rollout obs to identify the "
                    "equipped weapon per tick — this PPO env backend does not "
                    "carry it (qnn.ppo.vec_env's unpack_frame_batch at "
                    "DEFAULT_LAYOUT is the process backend's source; "
                    "arena_grid never supplies it)"
                )
            _, logits, _, _, _ = model(
                obs_t, hidden0, reset_mask=reset_mb, reset_ts=reset_ts,
            )
            mask = (
                torch.isfinite(buffer.align_hbw.index_select(1, lanes))
                & buffer.decision_mask["attack"].index_select(1, lanes)
            )
            if not bool(mask.any()):
                continue
            impulse = self_weapon_id_to_impulse(obs_t["self_weapon_id"].long())
            raw_rows.append(logits[head_key][mask].float())
            held_impulses.append(impulse[mask])
            # Onset candidacy: a tick can START a hold-train only if the
            # REALIZED previous tick (same lane, same episode) did not
            # attack. Episode-start frames have no previous tick.
            attacked = buffer.actions["attack"].index_select(1, lanes) > 0
            prev_attacked = torch.zeros_like(attacked)
            prev_attacked[1:] = attacked[:-1]
            prev_attacked[reset_mb] = False
            onset_candidates.append((~prev_attacked)[mask])
    if was_training:
        model.train()

    stats: dict[str, float] = {}
    if not raw_rows:
        for weapon in targets:
            stats[f"fire_occupancy/project/{weapon}/frac_scored"] = 0.0
        return stats

    raw = torch.cat(raw_rows, dim=0)        # (N, 9) unbiased selector logits
    held = torch.cat(held_impulses, dim=0)  # (N,) impulse 1..8 (0 = none)
    candidates = torch.cat(onset_candidates, dim=0)  # (N,) realized prev != attack
    total = int(raw.shape[0])
    tau = max(float(temperature), 1e-6)
    current_bias = fire_bias.detach()

    from qnn.decode_fit.context import WEAPON_IMPULSE

    for weapon, target in targets.items():
        impulse = WEAPON_IMPULSE.get(weapon)
        if impulse is None or not 1 <= impulse <= 8:
            raise ValueError(
                f"fire_occupancy_projection_targets: unknown weapon "
                f"{weapon!r} (have {sorted(WEAPON_IMPULSE)})"
            )
        w_mask = held == impulse
        n = int(w_mask.sum())
        stats[f"fire_occupancy/project/{weapon}/frac_scored"] = n / total
        if n == 0:
            continue
        onset_basis = getattr(target, "basis", "") == "onset"
        if onset_basis:
            # Onset-basis: match the RATE OF HOLD-TRAIN STARTS per engaged
            # held tick, not the per-tick held fraction. Numerator = summed
            # attack probability over the ONSET-CANDIDATE ticks (realized
            # previous tick not attacking — first-order surrogate holding
            # the sampled trajectory fixed, the same fixed-trajectory
            # assumption the held-fraction bisection already makes);
            # denominator = ALL engaged held decision ticks, matching
            # family_onset_rates_los's per-engaged-LOS-tick population.
            # Hold-continuation ticks contribute nothing, so pushing the
            # onset rate down cannot train holds away — the failure mode
            # that keeps the per-tick LOSS refusing onset targets.
            cand = w_mask & candidates
            n_cand = int(cand.sum())
            stats[f"fire_occupancy/project/{weapon}/frac_onset_candidates"] = (
                n_cand / n
            )
            if n_cand == 0:
                continue
            rows = raw[cand]
        else:
            rows = raw[w_mask]
        # Margin already includes weapon w's CURRENT fire_bias, mirroring
        # _project_fire_occupancy's margin/model_bias contract: the solved
        # delta below is an INCREMENT to add, not a replacement value.
        margin = (
            rows[..., impulse] + current_bias[impulse - 1] - rows[..., 0]
        )
        denom = float(n) if onset_basis else float(margin.numel())

        def _q(shift: torch.Tensor | float) -> torch.Tensor:
            return torch.sigmoid((margin + shift) / tau).sum() / denom

        before = _q(0.0)
        lo, hi = margin.new_tensor(-20.0), margin.new_tensor(20.0)
        tgt = margin.new_tensor(float(target.probability))
        for _ in range(48):
            mid = (lo + hi) * 0.5
            lo, hi = (mid, hi) if _q(mid) < tgt else (lo, mid)
        solved_delta = (lo + hi) * 0.5
        delta = solved_delta
        if float(max_delta) > 0.0:
            delta = delta.clamp(-float(max_delta), float(max_delta))
        with torch.no_grad():
            fire_bias[impulse - 1].add_(delta.to(fire_bias.dtype))
        after = _q(delta)
        stats.update({
            f"fire_occupancy/project/{weapon}/q_mean": float(after),
            f"fire_occupancy/project/{weapon}/target": float(target.probability),
            f"fire_occupancy/project/{weapon}/before": float(before),
            f"fire_occupancy/project/{weapon}/delta": float(delta),
            f"fire_occupancy/project/{weapon}/solved_delta": float(solved_delta),
        })
    return stats


def ppo_update(
    policy: Any,
    value_head: ValueHead,
    buffer: Any,
    adapters: Mapping[str, HeadDistribution],
    optimizer: torch.optim.Optimizer,
    cfg: PPOUpdateConfig,
    *,
    mb_generator: torch.Generator | None = None,
    anchor_model: Any = None,
) -> Dict[str, float]:
    """One PPO update over a full rollout window. Returns flat metrics."""
    anchor_heads = {
        h: float(c) for h, c in cfg.anchor_kl_coef.items()
        if float(c) != 0.0 and h in adapters
    }
    if anchor_heads and anchor_model is None:
        raise RuntimeError(
            "anchor_kl_coef enables heads "
            f"{sorted(anchor_heads)} but no anchor model was provided"
        )

    model = policy.model
    was_training = model.training
    model.train()

    anchor_store: Dict[str, torch.Tensor] | None = None
    if anchor_heads:
        anchor_keys = tuple(dict.fromkeys(
            k for h in anchor_heads for k in adapters[h].logits_keys
        ))
        anchor_store = _anchor_window_logits(
            policy, anchor_model, buffer, anchor_keys, cfg.minibatch_lanes,
        )

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

    # Piecewise-constant human target per tick, built ONCE from the rollout's
    # alignment stream (NaN off-LOS propagates through as NaN and is masked).
    pfire_tgt = None
    if cfg.pfire_target is not None and float(cfg.pfire_coef) != 0.0:
        if "attack" not in adapters:
            raise RuntimeError(
                "pfire_coef is set but the attack head is not trained — the "
                "trigger objective has nothing to act on"
            )
        pfire_tgt = torch.from_numpy(
            cfg.pfire_target.lookup(buffer.align_hbw.detach().cpu().numpy())
        ).to(buffer.align_hbw.device, torch.float32)
    occupancy_target = None
    if (cfg.fire_occupancy_target is not None
            and (float(cfg.fire_occupancy_coef) != 0.0
                 or bool(cfg.fire_occupancy_project))):
        if "attack" not in adapters:
            raise RuntimeError(
                "fire_occupancy_coef is set but the attack head is not trained"
            )
        if getattr(cfg.fire_occupancy_target, "basis", "bolt") != "bolt":
            raise RuntimeError(
                "fire_occupancy_target has basis="
                f"{cfg.fire_occupancy_target.basis!r}: _fire_occupancy_loss "
                "measures per-tick P(attack), which for continuous weapons "
                "includes hold-continuation ticks — matching it to an ONSET "
                "rate would suppress holds entirely. The onset-basis "
                "occupancy loss is not implemented "
                "(agents/plans/crest-finetune-allweapons.md); disable the "
                "occupancy term for this weapon or implement the onset loss."
            )
        occupancy_target = float(cfg.fire_occupancy_target.probability)

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
            with policy._autocast():
                obs_t, hidden0, reset_mb, reset_ts = _dequant_window(
                    policy, buffer, lanes,
                )
                features, logits, _, _, _ = model(
                    obs_t, hidden0, reset_mask=reset_mb, reset_ts=reset_ts,
                )
            values = value_head(value_features(features, obs_t, cfg.value_af))
            anchor_mb = (
                {k: v.index_select(1, lanes) for k, v in anchor_store.items()}
                if anchor_store is not None else None
            )

            adv_mb = adv.index_select(1, lanes)
            ret_mb = buffer.returns.index_select(1, lanes)
            val_old = buffer.values.index_select(1, lanes)

            policy_loss = values.new_zeros(())
            entropy_loss = values.new_zeros(())
            kl_max = 0.0
            attack_dist = adapters.get("attack")
            attack_temp = float(cfg.temperatures.get("attack", 1.0))
            if pfire_tgt is not None:
                assert attack_dist is not None
                policy_loss = policy_loss + _pfire_loss(
                    attack_dist, logits, obs_t,
                    pfire_tgt.index_select(1, lanes),
                    buffer.decision_mask["attack"].index_select(1, lanes),
                    float(cfg.pfire_coef), attack_temp, _log,
                )
            if occupancy_target is not None and float(cfg.fire_occupancy_coef) != 0.0:
                assert attack_dist is not None
                policy_loss = policy_loss + _fire_occupancy_loss(
                    attack_dist, logits,
                    buffer.align_hbw.index_select(1, lanes),
                    buffer.decision_mask["attack"].index_select(1, lanes),
                    occupancy_target, float(cfg.fire_occupancy_coef),
                    float(cfg.fire_occupancy_temperature), _log,
                )
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
                if anchor_mb is not None and h in anchor_heads:
                    kl_anchor = dist.kl_divergence(
                        logits, anchor_mb,
                        temperature=float(cfg.temperatures.get(h, 1.0)),
                    )
                    policy_loss = policy_loss + anchor_heads[h] * _masked_mean(
                        kl_anchor, mask,
                    )
                    _log(f"kl_anchor/{h}", _masked_mean(kl_anchor.detach(), mask))
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

    projection_stats: dict[str, float] = {}
    if cfg.fire_occupancy_project:
        attack_dist = adapters.get("attack")
        if attack_dist is None:
            raise RuntimeError(
                "fire_occupancy_project is set but the attack head is not "
                "trained"
            )
        if getattr(attack_dist, "impulse", None) is not None:
            # Fixed-weapon PPO: one pinned impulse, the original rung3k
            # regulator (unchanged; still the only path that adapter needs).
            if occupancy_target is None:
                raise RuntimeError(
                    "fire_occupancy_project is set for the fixed-weapon "
                    "attack adapter but fire_occupancy_target is missing"
                )
            projection_stats = _project_fire_occupancy(
                policy, buffer, attack_dist, occupancy_target,
                cfg.fire_occupancy_temperature, cfg.minibatch_lanes,
                cfg.fire_occupancy_project_max_delta,
            )
        else:
            # General 9-way multi-weapon PPO (crest-finetune-allweapons
            # iteration 2): one bisection per weapon in the targets mapping.
            if not cfg.fire_occupancy_projection_targets:
                raise RuntimeError(
                    "fire_occupancy_project is set for the general attack "
                    "adapter but fire_occupancy_projection_targets is "
                    "missing — the per-weapon human cadence rulers must be "
                    "named explicitly"
                )
            projection_stats = _project_fire_occupancy_per_weapon(
                policy, buffer, attack_dist,
                cfg.fire_occupancy_projection_targets,
                cfg.fire_occupancy_temperature, cfg.minibatch_lanes,
                cfg.fire_occupancy_project_max_delta,
            )

    if not was_training:
        model.eval()

    out = {k: sum(v) / len(v) for k, v in stats.items() if v}
    out.update(projection_stats)
    out["ppo_epochs_run"] = float(epochs_run)
    with torch.no_grad():
        var_ret = buffer.returns.var()
        out["explained_variance"] = float(
            1.0 - (buffer.returns - buffer.values).var() / (var_ret + 1e-8)
        )
    return out
