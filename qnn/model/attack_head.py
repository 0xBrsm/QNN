"""Canonical attack head: features → MLP → binary logit.

Intentionally minimal — no prior, no extra signals. Variants that add a
prior (aim alignment, hit-test physics, feasibility gating, etc.) were
concluded bench ablations (findings in ``src/docs/fire-discrimination.md``);
any new variant is a standalone ``nn.Module`` implementing the same
``AttackHeadInput → AttackHeadOutput`` contract, slotted in via
``Network(attack_head=<variant>)``.

``AttackHeadInput`` carries optional fields that variants may need
(``look_prior``, ``weapon_id``, ``target_logits``, ``entity_scalars``,
``actor_mask``, ``self_scalars``). The canonical head reads only
``features``; variants pick what they need.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp

OUT_DIM = 1  # binary logit


@dataclass(frozen=True, slots=True)
class AttackHeadInput:
    features: torch.Tensor                              # (B*, in_dim)
    # Optional fields consumed by bench variants. Canonical AttackHead
    # ignores them. Network.forward populates all of them so any variant
    # slotted in receives a uniform input dataclass.
    look_prior: torch.Tensor | None = None               # (B*, 3) unit vec to target
    weapon_id: torch.Tensor | None = None               # (B*, 1) raw obs ID
    target_logits: torch.Tensor | None = None           # (B*, N)
    entity_scalars: torch.Tensor | None = None          # (B*, N, ACTOR_SCALAR_DIM)
    actor_mask: torch.Tensor | None = None              # (B*, N) bool
    self_scalars: torch.Tensor | None = None            # (B*, SELF_SCALAR_DIM)


@dataclass(frozen=True, slots=True)
class AttackHeadOutput:
    attack_logit: torch.Tensor   # (B*, 1) fed to BCE
    prior_logit: torch.Tensor    # (B*, 1) prior alone — canonical returns zeros
    delta_attack: torch.Tensor   # (B*, 1) raw MLP output


class AttackHead(nn.Module):
    """Minimal attack head. features → MLP → logit."""

    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        delta_attack = self.mlp(inp.features)
        prior_logit = torch.zeros_like(delta_attack)
        return AttackHeadOutput(
            attack_logit=delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
