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

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.weapon_head import WeaponHeadInput, WeaponHeadOutput, weapon_index_from_id
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


_PRIOR_PATH = Path(__file__).with_name("weapon_transition_prior.json")


def _load_log_prior() -> torch.Tensor:
    """Fixed held->act log-prior table, (8,8) rows=held class, cols=act class."""
    data = json.loads(_PRIOR_PATH.read_text())
    t = torch.tensor(data["log_prior"], dtype=torch.float32)
    assert t.shape == (WEAPON_HEAD_SIZE, WEAPON_HEAD_SIZE), f"bad prior shape {t.shape}"
    return t


class PriorResidualWeaponHead(nn.Module):
    """Weapon head = fixed held->act transition log-prior + state-grounded residual.

        logits = residual_mlp(selector) + temperature * log_prior[held_idx]

    The held weapon enters ONLY through the fixed (counted, non-learnable) prior,
    indexed by ``obs_weapon_id``; the residual MLP is blind to it (the graph drops
    the ``held_weapon`` token), so the residual learns the state-conditioned
    "when/what to pull away" while the prior carries steady-state reconstruction.
    ``temperature`` is the single learnable knob (the prior strength). Requires the
    weapon head spec to set ``context_from_obs: true`` so the network passes
    ``obs_weapon_id``. Graph weapon type ``"cls_prior"``.
    """

    def __init__(self, *, in_dim: int, d_model: int, d_weapon: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.mlp = make_head_mlp(in_dim, WEAPON_HEAD_SIZE, d_weapon, activation)
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, d_model)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("log_prior", _load_log_prior())  # (8, 8), fixed

    def forward(self, inp: WeaponHeadInput) -> WeaponHeadOutput:
        feats = inp.selector[..., : self.in_dim]
        residual = self.mlp(feats)                                  # (B*, 8)
        assert inp.obs_weapon_id is not None, (
            "PriorResidualWeaponHead requires context_from_obs=true (obs_weapon_id)"
        )
        held = weapon_index_from_id(inp.obs_weapon_id.reshape(-1)).clamp(0, WEAPON_HEAD_SIZE - 1)
        prior = self.log_prior[held].reshape(residual.shape)        # (B*, 8)
        logits = residual + self.temperature * prior
        context = F.softmax(logits, dim=-1) @ self.embed.weight
        return WeaponHeadOutput(logits=logits, context=context)


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("weapon", "cls")
def _build_weapon_cls(head, dims, d_model):
    return CLSWeaponHead(
        in_dim=dims["weapon_in"], d_model=d_model, d_weapon=head.d_hidden,
        activation=head.activation)


@register_head("weapon", "cls_prior")
def _build_weapon_cls_prior(head, dims, d_model):
    return PriorResidualWeaponHead(
        in_dim=dims["weapon_in"], d_model=d_model, d_weapon=head.d_hidden,
        activation=head.activation)
