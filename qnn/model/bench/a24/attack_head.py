"""a24 attack head — CLS-readout MLP (graph attack type ``"cls"``).

Override of the canonical :class:`qnn.model.attack_head.AttackHead`: a plain MLP
over the leading ``in_dim`` features (CLS or GRU(CLS)). Registered in
``qnn.model.graph.build.HEAD_TYPES`` as attack type ``"cls"``.

Mirrors :class:`qnn.model.bench.a24.move_head.CLSMoveHead`: slices
``inp.features[..., :in_dim]`` so the zeroed ``target_feat`` half (pointer Off)
is dropped rather than fed as dead dims.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM as ATTACK_OUT_DIM
from qnn.model.attack_head import AttackHeadInput, AttackHeadOutput


class CLSAttackHead(nn.Module):
    """Attack head: MLP over the leading ``in_dim`` features (CLS or GRU(CLS))."""

    def __init__(self, *, in_dim: int, d_attack: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, ATTACK_OUT_DIM, d_attack, activation)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        feats = inp.features[..., : self.in_dim]
        delta_attack = self.mlp(feats)
        return AttackHeadOutput(
            attack_logit=delta_attack,
            prior_logit=torch.zeros_like(delta_attack),
            delta_attack=delta_attack,
        )


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("attack", "cls")
def _build_attack_cls(head, dims, d_model):
    return CLSAttackHead(in_dim=dims["motor_in"], d_attack=head.d_hidden, activation=head.activation)
