"""AttackHead variant with analytical projectile-lead oracle + residual MLP.

  base_logit  = Σ_n target_dist[n] · aim_scale · required_look_alignment[n]
  attack_logit = base_logit + delta_attack(features)

``required_look_alignment[n]`` is the cosine of the angle between the
forward axis and the unit vector the player would need to face to hit
entity n given the current weapon:

  hitscan      → rel[n] / |rel[n]|
  projectile   → constant-velocity intercept solve (gravity ignored)

The intercept solve mirrors ``scripts/analysis/attack_precision_offset.py``.
No cooldown / ammo / weapon-ready gate — train-time and eval-time filter
on the `input_mask` bit 0 already cover engine no-op frames.

The residual MLP's final Linear is zero-initialized so training starts
at the oracle's decision; the MLP learns only deviations.

Requires ``inp.target_logits``, ``inp.entity_scalars``, ``inp.actor_mask``,
``inp.weapon_id``. Pair with ``GTTargetPointer``
(``softmax(target_logits)`` recovers the GT distribution under that pointer).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.weapon_physics import QNN_DIST_SCALE, QNN_VEL_SCALE, WEAPON_PHYSICS
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput
from qnn.vocab import self_weapon_id_to_impulse

_ESC_REL_BEGIN, _ESC_REL_END = 3, 6   # entity_scalars_raw ACTOR layout
_ESC_VEL_BEGIN, _ESC_VEL_END = 7, 10
_MAX_WEAPON_ID = 9


def _build_weapon_luts() -> tuple[torch.Tensor, torch.Tensor]:
    hitscan = torch.zeros(_MAX_WEAPON_ID, dtype=torch.bool)
    speed = torch.zeros(_MAX_WEAPON_ID, dtype=torch.float32)
    for wid, phys in WEAPON_PHYSICS.items():
        if phys["hitscan"]:
            hitscan[wid] = True
        else:
            speed[wid] = float(phys["speed"])
    return hitscan, speed


class OracleAttackHead(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        d_hidden: int,
        activation: str,
        aim_scale: float,
    ) -> None:
        super().__init__()
        self.aim_scale = float(aim_scale)
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)
        # Zero-init the residual head's final Linear so training starts
        # at the oracle's decision. Matches look_head's convention.
        final = self.mlp[-1] if isinstance(self.mlp, nn.Sequential) else self.mlp
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        hitscan, speed = _build_weapon_luts()
        self.register_buffer("_weapon_hitscan", hitscan, persistent=False)
        self.register_buffer("_weapon_speed", speed, persistent=False)

    def _required_look_alignment(
        self,
        rel: torch.Tensor,        # (B*, N, 3) view-frame, dequantized to [-1, 1]
        vel: torch.Tensor,        # (B*, N, 3) view-frame, dequantized to [-1, 1]
        weapon_id: torch.Tensor,  # (B*,) ENTITY_IDS-encoded
    ) -> torch.Tensor:
        rel_w = rel * QNN_DIST_SCALE
        vel_w = vel * QNN_VEL_SCALE

        rel_norm = torch.linalg.vector_norm(rel_w, dim=-1).clamp(min=1e-3)
        align_hit = rel_w[..., 0] / rel_norm                                # (B*, N)

        impulse_id = self_weapon_id_to_impulse(weapon_id.long())            # (B*,) in 0..8
        speed = self._weapon_speed.to(rel.device)[impulse_id]               # (B*,)
        speed_sq = (speed * speed).unsqueeze(-1)                            # (B*, 1)
        a = (vel_w * vel_w).sum(dim=-1) - speed_sq                          # (B*, N)
        b = 2.0 * (rel_w * vel_w).sum(dim=-1)
        c = (rel_w * rel_w).sum(dim=-1)
        disc = b * b - 4.0 * a * c
        a_safe = torch.where(a.abs() < 1e-6, torch.full_like(a, 1e-6), a)
        sq = torch.sqrt(disc.clamp(min=0.0))
        t1 = (-b - sq) / (2.0 * a_safe)
        t2 = (-b + sq) / (2.0 * a_safe)
        inf = torch.full_like(t1, float("inf"))
        t = torch.minimum(
            torch.where(t1 > 0, t1, inf),
            torch.where(t2 > 0, t2, inf),
        )
        valid = (disc >= 0) & (a.abs() > 1e-6) & torch.isfinite(t)
        t_safe = torch.where(valid, t, torch.zeros_like(t))
        intercept = rel_w + vel_w * t_safe.unsqueeze(-1)
        inter_norm = torch.linalg.vector_norm(intercept, dim=-1).clamp(min=1e-3)
        align_proj = intercept[..., 0] / inter_norm
        align_proj = torch.where(valid, align_proj, align_hit)

        is_hitscan = self._weapon_hitscan.to(rel.device)[impulse_id]
        return torch.where(is_hitscan.unsqueeze(-1), align_hit, align_proj)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        assert inp.target_logits is not None, "OracleAttackHead needs target_logits (use GTTargetPointer)"
        assert inp.entity_scalars is not None, "OracleAttackHead needs entity_scalars"
        assert inp.actor_mask is not None, "OracleAttackHead needs actor_mask"
        assert inp.weapon_id is not None, "OracleAttackHead needs weapon_id"

        rel = inp.entity_scalars[..., _ESC_REL_BEGIN:_ESC_REL_END]            # (B*, N, 3)
        vel = inp.entity_scalars[..., _ESC_VEL_BEGIN:_ESC_VEL_END]            # (B*, N, 3)
        weapon_id_flat = inp.weapon_id.long().reshape(-1)                     # (B*,)
        aim_alignment = self._required_look_alignment(rel, vel, weapon_id_flat)  # (B*, N)

        actor_mask = inp.actor_mask.to(aim_alignment.dtype)
        oracle_per_idx = self.aim_scale * aim_alignment * actor_mask          # (B*, N)

        # GTTargetPointer logits are log-probs; softmax recovers the GT mass.
        idx_dist = F.softmax(inp.target_logits, dim=-1).to(oracle_per_idx.dtype)
        prior_logit = (idx_dist * oracle_per_idx).sum(dim=-1, keepdim=True).to(
            inp.features.dtype
        )

        delta_attack = self.mlp(inp.features)                                 # (B*, 1)
        return AttackHeadOutput(
            attack_logit=prior_logit + delta_attack,
            prior_logit=prior_logit,
            delta_attack=delta_attack,
        )
