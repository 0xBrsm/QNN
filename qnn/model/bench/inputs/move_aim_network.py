"""Network wrapper that enters BOTH the obs-accessor and weapon_aim scopes.

Used by the move-token look-head ablation when the prior is ``aim_vec``:
the move token reads self scalars / vocab embeds via the ObsAccessor, and
the aim_vec prior needs the WeaponAimContext (entity geometry + weapon).
Subclasses ``WeaponAimNetwork`` (which enters the weapon_aim scope around
``Network.forward``) and additionally enters the obs-accessor scope around
that — so the head sees both via ``current_obs_accessor`` /
``current_weapon_aim_context``.
"""
from __future__ import annotations

from typing import Dict

import torch

from qnn.model.tokens.obs_accessor import obs_accessor_scope_from_obs
from qnn.model.bench.weapon_aim.network import WeaponAimNetwork


class MoveAimNetwork(WeaponAimNetwork):
    def forward(  # type: ignore[override]
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ):
        with obs_accessor_scope_from_obs(obs):
            return super().forward(obs=obs, hidden=hidden, reset_mask=reset_mask)
