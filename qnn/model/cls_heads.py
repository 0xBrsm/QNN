"""CLS-readout action heads — the validated full_4head/full_5head winners.

Promoted from ``qnn.model.bench`` after the full_4head promotion to
canonical BC v24: each head is a plain MLP over the leading ``in_dim``
features (the CLS self-readout, or the GRU-integrated CLS when a
temporal node is active). They are registered in the model-graph head
tables (``qnn.model.graph.build.HEAD_TYPES``) as type ``"cls"``.

The ``[..., :in_dim]`` slice exists so a head sized for the readout
alone drops a zeroed ``target_feat`` tail (pointer Off) instead of
reading dead dims; with a pointer active the caller sizes ``in_dim`` to
include it and the slice is a no-op.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM as ATTACK_OUT_DIM
from qnn.model.attack_head import AttackHeadInput, AttackHeadOutput
from qnn.model.move_head import MoveHeadInput, MoveHeadOutput
from qnn.model.weapon_head import WeaponHeadInput, WeaponHeadOutput
from qnn.schema import WEAPON_HEAD_SIZE

_MOVE_OUT_DIM = MOVE_AXES * MOVE_AXIS_CLASSES


class CLSMoveHead(nn.Module):
    """Move head: MLP over the leading ``in_dim`` features.

    No-GRU: in_dim=d_model reads the CLS self_readout (target_feat half is zeros).
    GRU:    in_dim=d_gru reads gru_flat (the GRU-integrated CLS; target_feat half zeros).
    """

    def __init__(self, *, in_dim: int, d_move: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, _MOVE_OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        feats = inp.features[..., : self.in_dim]
        logits = self.mlp(feats).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


class CLSAttackHead(nn.Module):
    """Attack head: MLP over the leading ``in_dim`` features (CLS or GRU(CLS)).

    Mirrors ``CLSMoveHead``: slices ``inp.features[..., :in_dim]`` so the zeroed
    target_feat half (pointer Off) is dropped rather than fed as dead dims.
    """

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


class CLSWeaponHead(nn.Module):
    """Weapon head: MLP over the leading ``in_dim`` features (CLS or GRU(CLS)).

    Mirrors ``CLSAttackHead``: slices ``inp.selector[..., :in_dim]`` so the zeroed
    target_feat half (pointer Off) is dropped rather than fed as dead dims.
    """

    def __init__(self, *, in_dim: int, d_model: int, d_weapon: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.mlp = make_head_mlp(in_dim, WEAPON_HEAD_SIZE, d_weapon, activation)
        # Soft-mix context for motor heads (kept even when they're Off — the contract).
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, d_model)

    def forward(self, inp: WeaponHeadInput) -> WeaponHeadOutput:
        feats = inp.selector[..., : self.in_dim]
        logits = self.mlp(feats)
        context = F.softmax(logits, dim=-1) @ self.embed.weight
        return WeaponHeadOutput(logits=logits, context=context)
