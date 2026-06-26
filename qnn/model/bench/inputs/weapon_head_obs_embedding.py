"""WeaponHeadObsEmbedding — replace the self token with a held-weapon token.

Bench obs embeddings for the look/attack parity ablation that pairs
``LookHead`` + ``LookStyleAttackHead`` over ``features = cat(self_token,
target_feat)``. Each variant overrides only the self block; entity / spatial
/ event / mask plumbing inherits from ``ObsEmbedding`` unchanged.

The self tokens are now expressed as :class:`TokenBuilder` configs over named
``obs_fields`` (weapon static-table lookup, held-weapon readiness gather,
held-weapon embed, kind tag) instead of bespoke slicing + impulse conversion.
"""

from __future__ import annotations

import torch

from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import (
    KIND_SELF, ScalarGroup, VocabEmbed, KindTag,
)
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.transformer import ObsEmbedding


class WeaponHeadObsEmbedding(ObsEmbedding):
    """1-self-token variant — slot 0 is the held-weapon token.

    The token combines the 7 static weapon-physics scalars with the
    ``attack_finished`` per-tick cooldown countdown (first-class and
    unconditional — the held-weapon token is its single home, there is no
    flag to remove it) and, optionally, the held weapon's ammo readiness:

      * ``include_ammo`` — adds the held weapon's ownership-masked ammo
        readiness fraction.

    Project-input width is ``8 + include_ammo`` (7 static + attack_finished).
    """

    _N_SELF_TOKENS = 1

    def __init__(
        self,
        d_model: int,
        *,
        self_weapon_embed_in_self: bool,
        include_spatial: bool = True,
        include_ammo: bool = True,
    ) -> None:
        self.include_ammo = bool(include_ammo)
        super().__init__(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon_embed_in_self,
            include_spatial=include_spatial,
        )

    def _init_self_components(self) -> None:
        # attack_finished is unconditional — its single home is the held-weapon token.
        scalar_names = ["weapon_static", "attack_finished"]
        if self.include_ammo:
            scalar_names.append("held_readiness")
        self.weapon_builder = TokenBuilder(
            self.d_model,
            [ScalarGroup(scalar_names), VocabEmbed("weapon_id"), KindTag(KIND_SELF)],
            entity_embed=self.entity_embed,
            movement_embed=self.movement_embed,
            kind_embed=self.kind_embed,
        )

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        return self.weapon_builder(ObsAccessor(obs_dict)).unsqueeze(1)


class AttackFinishedOnlyObsEmbedding(ObsEmbedding):
    """1-self-token variant: slot 0 = Linear(1, d_model)(attack_finished) + kind.

    No weapon static, no weapon ID embed, no readiness — strict minimum to
    test whether the cooldown countdown alone is the fire-frame discriminator
    the attack head needs.
    """

    _N_SELF_TOKENS = 1

    def _init_self_components(self) -> None:
        self.builder = TokenBuilder(
            self.d_model,
            [ScalarGroup(["attack_finished"]), KindTag(KIND_SELF)],
            entity_embed=self.entity_embed,
            movement_embed=self.movement_embed,
            kind_embed=self.kind_embed,
        )

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        return self.builder(ObsAccessor(obs_dict)).unsqueeze(1)


class TargetOnlyObsEmbedding(ObsEmbedding):
    """1-self-token variant where slot 0 is a constant zero token.

    For the ablation that drops the self/weapon half of the head input
    entirely — downstream encoder still slices a d_model vector at
    ``self_slice.start``, but it carries no signal.
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
