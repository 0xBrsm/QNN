"""Shared attack-selector output contract and the retired 8-way head.

Two output streams:

  ``logits``  — raw classifier output, trained via the registered head's
                owned loss and consumed by its direct action decoder.
  ``context`` — (B*, d_model) softmax(logits) @ embed.weight, fed to
                the motor heads. A27's 9-way attack-with head shares these
                dataclasses but not the canonical 8-way implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import AttackSelectorInput, AttackSelectorOutput
from qnn.schema import WEAPON_HEAD_SIZE

# Import-compatible aliases for retired checkpoints and bench modules.
WeaponHeadInput = AttackSelectorInput
WeaponHeadOutput = AttackSelectorOutput


class WeaponHead(nn.Module):
    def __init__(
        self,
        selector_dim: int,
        d_model: int,
        d_hidden: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.mlp = make_head_mlp(selector_dim, WEAPON_HEAD_SIZE, d_hidden, activation)
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, self.d_model)

    def forward(self, inp: AttackSelectorInput) -> AttackSelectorOutput:
        logits = self.mlp(inp.selector)                                          # (B*, WEAPON_HEAD_SIZE)
        probs = F.softmax(logits, dim=-1)
        context = probs @ self.embed.weight
        return AttackSelectorOutput(logits=logits, context=context)


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("weapon", "canonical")
def _build_weapon_canonical(head, dims, d_model):
    return WeaponHead(
        selector_dim=dims["weapon_in"], d_model=d_model, d_hidden=head.d_hidden,
        activation=head.activation)
