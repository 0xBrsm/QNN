"""Model-agnostic PPO action-head adapters.

The trainer deals only in named adapters.  Each adapter owns the complete
boundary between a model logit slot and the engine action contract:

* raw-policy sampling during collection;
* engine action encoding;
* rollout-buffer shape;
* learner log-probability and entropy recomputation; and
* the model module trained in heads-only mode.

Adding a head therefore changes this registry, not the collector, learner, or
policy wrapper.  Deploy-time decoding remains the policy's default for every
engine action field not claimed by an active PPO adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np
import torch
import torch.nn.functional as F

from qnn.model.decode import bernoulli_sample, categorical_sample
from qnn.actions import ATTACK_ACTION_SIZE
from qnn.model.network import ATTACK_FIRE_BIAS, ATTACK_HEAD, MOVE_HEAD


def categorical_log_prob_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log pi(a) and entropy for a temperature-scaled categorical."""
    logp = F.log_softmax(logits / max(float(temperature), 1e-6), dim=-1)
    lp = logp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    return lp, ent


def categorical_kl(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(P ‖ Q) between temperature-scaled categoricals, per row."""
    tau = max(float(temperature), 1e-6)
    logp = F.log_softmax(logits_p / tau, dim=-1)
    logq = F.log_softmax(logits_q / tau, dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def bernoulli_log_prob_entropy(
    logit: torch.Tensor,
    actions: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log pi(a) and entropy for Bernoulli(sigmoid(logit / tau))."""
    scaled = logit / max(float(temperature), 1e-6)
    action = actions.to(scaled.dtype)
    lp = -(action * F.softplus(-scaled) + (1.0 - action) * F.softplus(scaled))
    prob = torch.sigmoid(scaled)
    ent = F.softplus(scaled) - scaled * prob
    return lp, ent


def bernoulli_kl(
    logit_p: torch.Tensor,
    logit_q: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(Bernoulli(P) ‖ Bernoulli(Q)) from logits, per element."""
    tau = max(float(temperature), 1e-6)
    p_logit = logit_p / tau
    q_logit = logit_q / tau
    p = torch.sigmoid(p_logit)
    return (
        p * (F.logsigmoid(p_logit) - F.logsigmoid(q_logit))
        + (1.0 - p)
        * (F.logsigmoid(-p_logit) - F.logsigmoid(-q_logit))
    )


class HeadDistribution(ABC):
    """One trainable raw-head policy and its engine action encoding."""

    name: str
    action_shape: tuple[int, ...]
    module_name: str
    engine_fields: frozenset[str]
    # Model logit-dict keys this adapter reads — the learner snapshots exactly
    # these slots from the frozen anchor forward for the anchor-KL term.
    logits_keys: tuple[str, ...]

    @abstractmethod
    def sample(
        self,
        logits: Mapping[str, torch.Tensor],
        *,
        temperature: float,
        row_generators: Any,
    ) -> torch.Tensor:
        """Sample buffer-encoded actions for one flat collector batch."""

    @abstractmethod
    def apply(
        self,
        engine_actions: MutableMapping[str, np.ndarray],
        sampled: torch.Tensor,
    ) -> None:
        """Replace this adapter's fields in the decoded engine action batch."""

    @abstractmethod
    def log_prob_entropy(
        self,
        logits: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute the raw policy for flat or windowed stored actions."""

    @abstractmethod
    def kl_divergence(
        self,
        logits: Mapping[str, torch.Tensor],
        anchor_logits: Mapping[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Per-element KL(π_current ‖ π_anchor) at this head's temperature.

        Both logit dicts use the model's head-keyed layout; the leading
        dims (flat B or windowed T×B) pass through unchanged."""

    def collect(
        self,
        logits: Mapping[str, torch.Tensor],
        engine_actions: MutableMapping[str, np.ndarray],
        *,
        temperature: float,
        row_generators: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled = self.sample(
            logits, temperature=temperature, row_generators=row_generators,
        )
        self.apply(engine_actions, sampled)
        mask = torch.ones(sampled.shape[0], dtype=torch.bool, device=sampled.device)
        return sampled, mask

    def fire_probability(
        self,
        logits: Mapping[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Probability of the attack action this adapter actually samples."""
        raise RuntimeError(f"{type(self).__name__} has no fire/no-fire action law")


class AttackDistribution(HeadDistribution):
    """Nine-class no-attack/attack-with-impulse distribution."""

    name = "attack"
    action_shape = ()
    module_name = "attack_head"
    engine_fields = frozenset({"attack"})
    logits_keys = (ATTACK_HEAD,)

    @staticmethod
    def _logits(logits: Mapping[str, torch.Tensor], shape: torch.Size) -> torch.Tensor:
        attack_logits = logits[ATTACK_HEAD]
        if attack_logits.shape[-1] != ATTACK_ACTION_SIZE:
            raise RuntimeError(
                f"attack adapter requires {ATTACK_ACTION_SIZE} logits, got "
                f"{attack_logits.shape[-1]}"
            )
        return attack_logits.reshape(*shape, ATTACK_ACTION_SIZE)

    @classmethod
    def _biased_logits(
        cls, logits: Mapping[str, torch.Tensor], shape: torch.Size,
    ) -> torch.Tensor:
        """9-way logits with the model-owned fire-only intercept
        (``attack_head.fire_bias``, published as ``ATTACK_FIRE_BIAS``) folded
        into classes 1..8; class 0 (no-attack) is untouched.

        ``fire_bias`` is deliberately NOT added inside the head's own forward
        (attack_with_head.py: "those logits also own weapon choice, so
        shifting them would silently change the selected weapon") — every
        fire consumer adds it back for its own purpose instead.
        ``FixedWeaponAttackDistribution.fire_logit`` is the single-weapon
        case; this is the general 9-way case, needed so
        ``qnn.ppo.learner``'s per-weapon fire-occupancy projection can
        actually move future rollouts by bisecting ``fire_bias`` (before
        this existed, ``sample``/``fire_probability``/``log_prob_entropy``
        never read ``ATTACK_FIRE_BIAS`` at all, so projecting it against the
        general adapter was a no-op). Zero bias is an exact identity.
        Deliberately excluded from ``kl_divergence`` for the same reason
        the fixed-weapon adapter excludes it there: a calibration
        intercept, not the trained conditional shape, so it should not
        fight the frozen anchor.
        """
        attack_logits = cls._logits(logits, shape)
        bias = logits.get(ATTACK_FIRE_BIAS)
        if bias is None:
            return attack_logits
        if bias.shape[-1] != ATTACK_ACTION_SIZE - 1:
            raise RuntimeError(
                f"attack adapter requires {ATTACK_ACTION_SIZE - 1} fire-bias "
                f"entries, got {bias.shape[-1]}"
            )
        bias = bias.reshape(*shape, ATTACK_ACTION_SIZE - 1).to(attack_logits.dtype)
        return torch.cat(
            [attack_logits[..., :1], attack_logits[..., 1:] + bias], dim=-1,
        )

    def sample(self, logits, *, temperature, row_generators):
        attack_logits = self._biased_logits(
            logits, torch.Size((logits[ATTACK_HEAD].shape[0],))
        )
        probs = F.softmax(attack_logits / max(float(temperature), 1e-6), dim=-1)
        return categorical_sample(probs, row_generators)

    def apply(self, engine_actions, sampled):
        engine_actions["attack"] = sampled.detach().cpu().numpy().astype(np.int64)

    def log_prob_entropy(self, logits, actions, temperature=1.0):
        attack_logits = self._biased_logits(logits, actions.shape)
        return categorical_log_prob_entropy(attack_logits, actions, temperature)

    def kl_divergence(self, logits, anchor_logits, temperature=1.0):
        shape = logits[ATTACK_HEAD].shape[:-1]
        return categorical_kl(
            self._logits(logits, shape),
            self._logits(anchor_logits, shape),
            temperature,
        )

    def fire_probability(self, logits, temperature=1.0):
        shape = logits[ATTACK_HEAD].shape[:-1]
        attack_logits = self._biased_logits(logits, shape)
        probs = F.softmax(
            attack_logits / max(float(temperature), 1e-6), dim=-1,
        )
        return 1.0 - probs[..., 0]


class FixedWeaponAttackDistribution(HeadDistribution):
    """Stochastic twin of the deployed fixed-weapon attack-with threshold.

    A one-weapon PPO arena does not need to relearn weapon selection. Its only
    operative decision is whether the pinned weapon's score beats class 0 after
    the run's decode operating-point offsets are applied. Sampling a Bernoulli
    from that exact margin gives PPO a stochastic policy whose 0.5 boundary is
    the greedy eval/export boundary; storing 0 or ``impulse`` preserves the A27
    single-lane engine contract.

    This deliberately excludes guard terms. Fixed SG/SNG/LG cells have no
    weapon-specific guard; RL cells must keep using the full deploy decode until
    the self-splash veto is represented in learner recomputation.
    """

    name = "attack"
    action_shape = ()
    module_name = "attack_head"
    engine_fields = frozenset({"attack"})
    logits_keys = (ATTACK_HEAD,)

    def __init__(
        self,
        impulse: int,
        *,
        attack_bias: float = 0.0,
        bias_vec: list[float] | tuple[float, ...] | None = None,
        fire_bias_vec: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        impulse = int(impulse)
        if not 1 <= impulse <= 8:
            raise ValueError(f"fixed attack impulse must be in [1, 8], got {impulse}")
        if impulse == 7:
            raise ValueError(
                "fixed rocket attack is not supported: learner recomputation "
                "does not yet carry the deployed self-splash guard"
            )
        self.impulse = impulse
        idx = impulse - 1
        legacy = 0.0 if bias_vec is None else float(bias_vec[idx])
        fire = 0.0 if fire_bias_vec is None else float(fire_bias_vec[idx])
        # attack_with_decode_step compares
        #   l_weapon + legacy + fire > l0 - attack_bias.
        self.decode_offset = float(attack_bias) + legacy + fire

    @staticmethod
    def _logits9(logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
        attack_logits = logits[ATTACK_HEAD]
        if attack_logits.shape[-1] != ATTACK_ACTION_SIZE:
            raise RuntimeError(
                f"fixed attack adapter requires {ATTACK_ACTION_SIZE} logits, got "
                f"{attack_logits.shape[-1]}"
            )
        return attack_logits

    def _base_fire_logit(self, logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits9 = self._logits9(logits)
        return logits9[..., self.impulse] - logits9[..., 0]

    def fire_logit(self, logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Exact selected-weapon Bernoulli logit used for collection/update."""
        out = self._base_fire_logit(logits) + self.decode_offset
        model_bias = logits.get(ATTACK_FIRE_BIAS)
        if model_bias is not None:
            if model_bias.shape[-1] != ATTACK_ACTION_SIZE - 1:
                raise RuntimeError(
                    "model fire bias must have 8 entries, got "
                    f"{model_bias.shape[-1]}"
                )
            out = out + model_bias[..., self.impulse - 1].to(out.dtype)
        return out

    def fire_probability(self, logits, temperature=1.0):
        return torch.sigmoid(
            self.fire_logit(logits) / max(float(temperature), 1e-6),
        )

    def sample(self, logits, *, temperature, row_generators):
        prob = self.fire_probability(logits, temperature)
        return bernoulli_sample(prob, row_generators) * self.impulse

    def apply(self, engine_actions, sampled):
        engine_actions["attack"] = sampled.detach().cpu().numpy().astype(np.int64)

    def log_prob_entropy(self, logits, actions, temperature=1.0):
        fire = (actions == self.impulse).to(actions.dtype)
        return bernoulli_log_prob_entropy(
            self.fire_logit(logits), fire, temperature,
        )

    def kl_divergence(self, logits, anchor_logits, temperature=1.0):
        # Anchor only the learned conditional timing shape.  The model-owned
        # family intercept is calibrated by the human occupancy constraint;
        # putting it in this KL would make the frozen zero-bias seed fight that
        # calibration.  External decode offsets are constants and omitted too.
        return bernoulli_kl(
            self._base_fire_logit(logits),
            self._base_fire_logit(anchor_logits),
            temperature,
        )


class MoveAxesDistribution(HeadDistribution):
    """Three independent three-class move axes; engine encoding is -1/0/+1."""

    name = "move"
    action_shape = (3,)
    module_name = "move_head"
    engine_fields = frozenset({"move"})
    logits_keys = (MOVE_HEAD,)

    def sample(self, logits, *, temperature, row_generators):
        move_logits = logits[MOVE_HEAD].reshape(-1, 3, 3)
        probs = F.softmax(move_logits / max(float(temperature), 1e-6), dim=-1)
        return torch.stack(
            [categorical_sample(probs[:, axis], row_generators) for axis in range(3)],
            dim=-1,
        )

    def apply(self, engine_actions, sampled):
        engine_actions["move"] = (
            sampled.detach().cpu().numpy().astype(np.float32) - 1.0
        )

    def log_prob_entropy(self, logits, actions, temperature=1.0):
        move_logits = logits[MOVE_HEAD].reshape(*actions.shape[:-1], 3, 3)
        lp, ent = categorical_log_prob_entropy(move_logits, actions, temperature)
        return lp.sum(dim=-1), ent.sum(dim=-1)

    def kl_divergence(self, logits, anchor_logits, temperature=1.0):
        shape = logits[MOVE_HEAD].shape[:-1]
        cur = logits[MOVE_HEAD].reshape(*shape, 3, 3)
        anc = anchor_logits[MOVE_HEAD].reshape(*shape, 3, 3)
        return categorical_kl(cur, anc, temperature).sum(dim=-1)


_ADAPTERS: dict[str, Callable[[], HeadDistribution]] = {
    "attack": AttackDistribution,
    "move": MoveAxesDistribution,
}


def build_adapters(
    rl_head_weights: Mapping[str, float],
    *,
    attack_impulse: int | None = None,
    attack_bias: float = 0.0,
    attack_bias_vec: list[float] | tuple[float, ...] | None = None,
    attack_fire_bias_vec: list[float] | tuple[float, ...] | None = None,
) -> dict[str, HeadDistribution]:
    """Build enabled adapters and reject unknown or overlapping action fields."""
    adapters: dict[str, HeadDistribution] = {}
    claimed_fields: dict[str, str] = {}
    for name, weight in rl_head_weights.items():
        if float(weight) == 0.0:
            continue
        factory = _ADAPTERS.get(name)
        if factory is None:
            raise ValueError(
                f"rl_head_weights[{name!r}] > 0 but no RL distribution adapter "
                f"exists for that head; known: {sorted(_ADAPTERS)}"
            )
        adapter = (
            FixedWeaponAttackDistribution(
                attack_impulse,
                attack_bias=attack_bias,
                bias_vec=attack_bias_vec,
                fire_bias_vec=attack_fire_bias_vec,
            )
            if name == "attack" and attack_impulse is not None
            else factory()
        )
        overlap = adapter.engine_fields & claimed_fields.keys()
        if overlap:
            field = sorted(overlap)[0]
            raise ValueError(
                f"RL adapters {claimed_fields[field]!r} and {name!r} both own "
                f"engine action field {field!r}"
            )
        adapters[name] = adapter
        claimed_fields.update({field: name for field in adapter.engine_fields})
    return adapters
