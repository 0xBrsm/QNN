"""WeaponHeadObsEmbedding — replace the self token with a held-weapon token.

Bench obs embedding for the look/attack parity ablation that pairs
``LookHead`` + ``LookStyleAttackHead`` over
``features = cat(weapon_token, target_feat)`` instead of the canonical
``cat(self_token, target_feat)``.

The token at ``self_slice.start`` becomes:

    weapon_token = weapon_proj(cat(weapon_static[impulse(weapon_id)],
                                   attack_finished,
                                   held_weapon_readiness))
                 + entity_embed(weapon_id) * (weapon_id > 0)
                 + kind_embed(KIND_SELF)

Static 7 scalars from ``build_model_weapon_scalars`` (damage, cooldown,
v_horiz, v_vert_0, gravity, max_dist, radius) + 2 dynamic fire-gate
scalars (current cooldown countdown ``attack_finished`` from
``self_scalars[IDX_ATTACK_FINISHED]``, and the held weapon's ammo
readiness sliced from ``self_weapon_readiness[impulse-1]``) get
projected to d_model together. The ENTITY_IDS-vocab embedding for the
held weapon is summed in, then the canonical KIND_SELF kind tag so
encoders that slice ``self_slice.start`` see this token as the
readout-slot.

Entity / spatial / event / mask plumbing inherits unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.bc.weapon_physics import (
    MODEL_TOKEN_SCALAR_DIM, build_model_weapon_scalars,
)
from qnn.model.dequant import IDX_ATTACK_FINISHED
from qnn.model.transformer import ObsEmbedding, _TOKEN_KIND_SELF
from qnn.vocab import self_weapon_id_to_impulse


class WeaponHeadObsEmbedding(ObsEmbedding):
    """1-self-token variant — slot 0 is the held-weapon token.

    The token combines the 7 static weapon-physics scalars with any
    subset of the two dynamic fire-gate scalars selected at construction:

      * ``include_attack_finished`` — adds ``self_scalars[IDX_ATTACK_FINISHED]``
        (per-tick cooldown countdown).
      * ``include_ammo`` — adds the held weapon's row of
        ``self_weapon_readiness`` (ownership-masked ammo fraction).

    Project-input width is ``7 + include_attack_finished + include_ammo``.
    """

    _N_SELF_TOKENS = 1

    def __init__(
        self,
        d_model: int,
        *,
        self_weapon_embed_in_self: bool,
        include_spatial: bool = True,
        include_attack_finished: bool = True,
        include_ammo: bool = True,
    ) -> None:
        self.include_attack_finished = bool(include_attack_finished)
        self.include_ammo = bool(include_ammo)
        super().__init__(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon_embed_in_self,
            include_spatial=include_spatial,
        )

    def _init_self_components(self) -> None:
        proj_dim = MODEL_TOKEN_SCALAR_DIM + int(self.include_attack_finished) + int(self.include_ammo)
        self.weapon_proj = nn.Linear(proj_dim, self.d_model)
        weapon_static = torch.from_numpy(build_model_weapon_scalars())
        self.register_buffer("_weapon_static", weapon_static, persistent=False)

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        # obs.self_weapon_id is ENTITY_IDS-encoded (NONE=0, …, AXE=3, …, LG=10).
        # weapon_static is impulse-indexed (0..8); self_weapon_readiness is
        # axe-first (impulse-1). Convert once, keep the ENTITY_IDS encoding
        # for entity_embed.
        wid_entity = obs_dict["self_weapon_id"].long().squeeze(-1).clamp(0, vocab_max)
        wid_impulse = self_weapon_id_to_impulse(wid_entity).clamp(
            0, self._weapon_static.shape[0] - 1,
        )
        proj_parts = [self._weapon_static[wid_impulse]]                          # (B, 7)
        if self.include_attack_finished:
            proj_parts.append(
                obs_dict["self_scalars"][:, IDX_ATTACK_FINISHED:IDX_ATTACK_FINISHED + 1]
            )
        if self.include_ammo:
            # readiness layout is (B, 8) in axe-first order; held = impulse - 1.
            # No-weapon (impulse 0) → readiness 0 via the (wid_impulse >= 1) gate.
            readiness_8 = obs_dict["self_weapon_readiness"]                      # (B, 8)
            held_idx = (wid_impulse - 1).clamp_min(0).unsqueeze(-1)              # (B, 1)
            held_readiness = readiness_8.gather(1, held_idx)                     # (B, 1)
            has_weapon = (wid_impulse >= 1).to(held_readiness.dtype).unsqueeze(-1)
            proj_parts.append(held_readiness * has_weapon)

        proj_in = torch.cat(proj_parts, dim=-1)
        weapon_token = self.weapon_proj(proj_in).unsqueeze(1)                    # (B, 1, d_model)

        # Sum the held weapon's ENTITY_IDS embedding when one is held.
        wmask = (wid_entity > 0).float().unsqueeze(-1).unsqueeze(-1)
        weapon_token = weapon_token + self.entity_embed(wid_entity).unsqueeze(1) * wmask

        # Match the canonical self-block kind tag so encoders that slice
        # self_slice.start see this token as the head-slot readout.
        kind_self = torch.full((batch, 1), _TOKEN_KIND_SELF, dtype=torch.long, device=device)
        weapon_token = weapon_token + self.kind_embed(kind_self)

        return weapon_token


class AttackFinishedOnlyObsEmbedding(ObsEmbedding):
    """1-self-token variant: slot 0 = Linear(1, d_model)(attack_finished) + kind.

    No weapon static, no weapon ID embed, no readiness — strict
    minimum to test whether the cooldown countdown alone is the
    fire-frame discriminator the attack head needs. Pair with the
    canonical LookHead + LookStyleAttackHead so the only thing
    differing from ``target_only`` is the single attack_finished scalar
    injected into the head's feature half.
    """

    _N_SELF_TOKENS = 1

    def _init_self_components(self) -> None:
        self.attack_finished_proj = nn.Linear(1, self.d_model)

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        af = obs_dict["self_scalars"][:, IDX_ATTACK_FINISHED:IDX_ATTACK_FINISHED + 1]
        tok = self.attack_finished_proj(af).unsqueeze(1)                          # (B, 1, d_model)
        kind_self = torch.full((batch, 1), _TOKEN_KIND_SELF, dtype=torch.long, device=device)
        tok = tok + self.kind_embed(kind_self)
        return tok


class TargetOnlyObsEmbedding(ObsEmbedding):
    """1-self-token variant where slot 0 is a constant zero token.

    For the ablation that drops the self/weapon half of the head input
    entirely — downstream encoder still slices a d_model vector at
    ``self_slice.start``, but it carries no signal. The MLP's first
    Linear sees ``cat(zeros, target_feat)``, so only target_feat
    contributes (modulo the bias term).
    """

    _N_SELF_TOKENS = 1

    def _init_self_components(self) -> None:
        # No projection / buffers: the token is identically zero.
        pass

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        # Match the float dtype the encoder consumes downstream (bf16 under
        # autocast, float32 outside). self_scalars is always float.
        ref = obs_dict["self_scalars"]
        return torch.zeros((batch, 1, self.d_model), dtype=ref.dtype, device=device)
