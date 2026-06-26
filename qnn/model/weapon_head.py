"""Weapon head: 8-way weapon classifier + weapon context for motor heads.

Two output streams:

  ``logits``  — (B*, WEAPON_HEAD_SIZE) raw classifier output, trained
                via CE loss; consumed by the weapon switch heuristic at
                inference.
  ``context`` — (B*, d_model) embedding fed to the motor heads
                (move/look/attack). Two sources controlled by
                ``context_from_obs``:
                  True  → embed(currently-held weapon from obs).
                  False → softmax(logits) @ embed.weight (soft mix).
                The weapon_head still trains via its own CE loss in either
                mode. The obs path lets the motor heads condition on the
                ground-truth held weapon without depending on the
                classifier — useful for ablations of classifier quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.schema import WEAPON_HEAD_SIZE
from qnn.vocab import ENTITY_IDS


def weapon_index_from_id(weapon_ids: torch.Tensor) -> torch.Tensor:
    """Map raw entity IDs to the 0..7 weapon-head class index.

    Non-weapon IDs map to 0 (axe). Caller should clamp to
    [0, WEAPON_HEAD_SIZE - 1] if downstream consumers require it.
    """
    wid = weapon_ids.long()
    indices = torch.zeros_like(wid)
    indices = torch.where(wid == ENTITY_IDS["AXE"], torch.full_like(indices, 0), indices)
    indices = torch.where(wid == ENTITY_IDS["SHOTGUN"], torch.full_like(indices, 1), indices)
    indices = torch.where(wid == ENTITY_IDS["SUPER_SHOTGUN"], torch.full_like(indices, 2), indices)
    indices = torch.where(wid == ENTITY_IDS["NAILGUN"], torch.full_like(indices, 3), indices)
    indices = torch.where(wid == ENTITY_IDS["SUPER_NAILGUN"], torch.full_like(indices, 4), indices)
    indices = torch.where(wid == ENTITY_IDS["GRENADE_LAUNCHER"], torch.full_like(indices, 5), indices)
    indices = torch.where(wid == ENTITY_IDS["ROCKET_LAUNCHER"], torch.full_like(indices, 6), indices)
    indices = torch.where(wid == ENTITY_IDS["THUNDERBOLT"], torch.full_like(indices, 7), indices)
    return indices


@dataclass(frozen=True, slots=True)
class WeaponHeadInput:
    selector: torch.Tensor                  # (B*, selector_dim) — features for the classifier
    obs_weapon_id: torch.Tensor | None      # (B*,) raw obs ID; required iff context_from_obs


@dataclass(frozen=True, slots=True)
class WeaponHeadOutput:
    logits: torch.Tensor   # (B*, WEAPON_HEAD_SIZE)
    context: torch.Tensor  # (B*, d_model)
    # Optional loss-only output forwarded generically by Network as an underscored
    # key (not used for inference/sampling) so a bench head's LOSS can live entirely
    # in bench — mirrors LookHeadOutput's distributional fields.
    when_logit: torch.Tensor | None = None  # (B*, 1) switch hazard, for weapon_switch


class WeaponHead(nn.Module):
    def __init__(
        self,
        selector_dim: int,
        d_model: int,
        d_hidden: int,
        activation: str,
        *,
        context_from_obs: bool,
    ) -> None:
        super().__init__()
        self.context_from_obs = bool(context_from_obs)
        self.d_model = int(d_model)
        self.mlp = make_head_mlp(selector_dim, WEAPON_HEAD_SIZE, d_hidden, activation)
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, self.d_model)

    def forward(self, inp: WeaponHeadInput) -> WeaponHeadOutput:
        logits = self.mlp(inp.selector)                                          # (B*, WEAPON_HEAD_SIZE)
        if self.context_from_obs:
            assert inp.obs_weapon_id is not None, (
                "context_from_obs=True requires obs_weapon_id"
            )
            idx = weapon_index_from_id(inp.obs_weapon_id.reshape(-1)).clamp(0, WEAPON_HEAD_SIZE - 1)
            context = self.embed(idx)
        else:
            probs = F.softmax(logits, dim=-1)
            context = probs @ self.embed.weight
        return WeaponHeadOutput(logits=logits, context=context)
