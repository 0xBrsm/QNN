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
from qnn.model.network import ATTACK_HEAD, MOVE_HEAD


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


class HeadDistribution(ABC):
    """One trainable raw-head policy and its engine action encoding."""

    name: str
    action_shape: tuple[int, ...]
    module_name: str
    engine_fields: frozenset[str]

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


class AttackDistribution(HeadDistribution):
    """Nine-class no-attack/attack-with-impulse distribution."""

    name = "attack"
    action_shape = ()
    module_name = "attack_head"
    engine_fields = frozenset({"attack"})

    @staticmethod
    def _logits(logits: Mapping[str, torch.Tensor], shape: torch.Size) -> torch.Tensor:
        attack_logits = logits[ATTACK_HEAD]
        if attack_logits.shape[-1] != ATTACK_ACTION_SIZE:
            raise RuntimeError(
                f"attack adapter requires {ATTACK_ACTION_SIZE} logits, got "
                f"{attack_logits.shape[-1]}"
            )
        return attack_logits.reshape(*shape, ATTACK_ACTION_SIZE)

    def sample(self, logits, *, temperature, row_generators):
        attack_logits = self._logits(
            logits, torch.Size((logits[ATTACK_HEAD].shape[0],))
        )
        probs = F.softmax(attack_logits / max(float(temperature), 1e-6), dim=-1)
        return categorical_sample(probs, row_generators)

    def apply(self, engine_actions, sampled):
        engine_actions["attack"] = sampled.detach().cpu().numpy().astype(np.int64)

    def log_prob_entropy(self, logits, actions, temperature=1.0):
        attack_logits = self._logits(logits, actions.shape)
        return categorical_log_prob_entropy(attack_logits, actions, temperature)


class MoveAxesDistribution(HeadDistribution):
    """Three independent three-class move axes; engine encoding is -1/0/+1."""

    name = "move"
    action_shape = (3,)
    module_name = "move_head"
    engine_fields = frozenset({"move"})

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


_ADAPTERS: dict[str, Callable[[], HeadDistribution]] = {
    "attack": AttackDistribution,
    "move": MoveAxesDistribution,
}


def build_adapters(rl_head_weights: Mapping[str, float]) -> dict[str, HeadDistribution]:
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
        adapter = factory()
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
