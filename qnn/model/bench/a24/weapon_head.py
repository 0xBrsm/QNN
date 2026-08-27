"""a24 weapon head — CLS-readout MLP (graph weapon type ``"cls"``).

Override of the canonical :class:`qnn.model.weapon_head.WeaponHead`: a plain MLP
over the leading ``in_dim`` features (CLS or GRU(CLS)) plus the soft-mix context
embedding for the motor heads. Registered in
``qnn.model.graph.build.HEAD_TYPES`` as weapon type ``"cls"``.

Mirrors :class:`qnn.model.bench.a24.attack_head.CLSAttackHead`: slices
``inp.selector[..., :in_dim]`` so the zeroed ``target_feat`` half (pointer Off)
is dropped rather than fed as dead dims.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.weapon_head import WeaponHeadInput, WeaponHeadOutput
from qnn.schema import WEAPON_HEAD_SIZE


class CLSWeaponHead(nn.Module):
    """Weapon head: MLP over the leading ``in_dim`` features (CLS or GRU(CLS))."""

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


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("weapon", "cls")
def _build_weapon_cls(head, dims, d_model):
    return CLSWeaponHead(
        in_dim=dims["weapon_in"], d_model=d_model, d_weapon=head.d_hidden,
        activation=head.activation)
