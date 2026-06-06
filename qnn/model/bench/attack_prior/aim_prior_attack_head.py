"""AttackHead variant with aim-alignment prior + residual MLP.

  prior_logit = scale * base_look[..., 0]   # alignment cosine in [-1, +1]
  attack_logit = prior_logit + delta_attack(features)

Three scale parameterizations selected at construction:
  ``fixed``     — scale is a Python constant
  ``scalar``    — scale is a single learned nn.Parameter
  ``perweapon`` — scale[weapon_id] is a learned (11, 1) Embedding

The residual MLP's final Linear is zero-initialized so training starts
at attack_logit == prior_logit exactly; the MLP learns only deviations.

Slots into Network via ``attack_head=AimPriorAttackHead(...)``.
Requires ``inp.base_look`` (and ``inp.weapon_id`` for the perweapon
variant) — both populated by Network.forward in canonical configs.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput

_VALID_SCALE_MODES = frozenset({"fixed", "scalar", "perweapon"})

# Per-weapon scale table size — covers ENTITY_IDS weapon range
# (NONE=0, AXE=3, ..., LG=10). 11 rows.
_N_WEAPON_SLOTS = 11


class AimPriorAttackHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_hidden: int,
        activation: str,
        *,
        scale_mode: str = "fixed",
        scale_init: float = 5.0,
    ) -> None:
        super().__init__()
        if scale_mode not in _VALID_SCALE_MODES:
            raise ValueError(
                f"scale_mode must be one of {sorted(_VALID_SCALE_MODES)}, got {scale_mode!r}"
            )
        self.scale_mode = scale_mode
        self.scale_init = float(scale_init)
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)
        if scale_mode == "scalar":
            self.scale = nn.Parameter(torch.tensor(self.scale_init, dtype=torch.float32))
        elif scale_mode == "perweapon":
            self.scale_emb = nn.Embedding(_N_WEAPON_SLOTS, 1)
            nn.init.constant_(self.scale_emb.weight, self.scale_init)
        # Zero-init final Linear so attack_logit starts at prior_logit.
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.base_look is not None, "AimPriorAttackHead requires base_look"
        alignment = inp.base_look[..., 0:1].to(inp.features.dtype)          # (B*, 1)
        if self.scale_mode == "fixed":
            prior_logit = self.scale_init * alignment
        elif self.scale_mode == "scalar":
            prior_logit = self.scale.to(alignment.dtype) * alignment
        else:  # perweapon
            assert inp.weapon_id is not None, "perweapon mode requires weapon_id"
            wid = inp.weapon_id.long().reshape(-1).clamp(0, _N_WEAPON_SLOTS - 1)
            scale = self.scale_emb(wid).to(alignment.dtype)                 # (B*, 1)
            prior_logit = scale * alignment
        delta_attack = self.mlp(inp.features)                               # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
