"""SplitSelfObsEmbedding — bench obs embedding that splits self into subtokens.

Subclass of the canonical ``ObsEmbedding``. Where the canonical produces
a single self token, this variant produces four:

    [CLS, state, arsenal, motion, [spatial_0..8,] entity_0..N-1]

The three self subtokens route disjoint scalar bundles + ID embeds, each
expressed as a :class:`TokenBuilder` over named ``obs_fields``:

* ``state``    health, effective_armor, armor_type, state powerups
                (PENT / RING / MEGAHEALTH)
* ``arsenal``  per-weapon readiness×subject sum (inventory), arsenal
                powerups (QUAD). Pure inventory — what the weapon head reads.
                Equipped-weapon id (incumbent leak) and attack_finished (equipped-weapon
                refire state → weapon token) are deliberately NOT here.
* ``motion``   vel_xyz, view_pitch, look_delta, movement_id, motion powerups
                (SUIT) — the canonical ``MOTION_FIELDS`` list.

CLS is a learnable parameter at slot 0; the attention pools the rest
of the stream into it. Encoders slice ``self_readout`` from CLS.

Everything outside the self block (entity / spatial / embeddings /
event handling / kind tags) inherits from ``ObsEmbedding`` unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import (
    MOTION_FIELDS, SELF_STATE_FIELDS, VocabSum, WeaponReadiness,
)
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.transformer import ObsEmbedding


class SplitSelfObsEmbedding(ObsEmbedding):
    """4-self-token variant — CLS + state + arsenal + motion."""

    _N_SELF_TOKENS = 4

    def _init_self_components(self) -> None:
        # CLS — learnable trunk-readout token at slot 0.
        self.cls_embed = nn.Parameter(torch.zeros(self.d_model))

        def _build(fields):
            return TokenBuilder(
                self.d_model, fields,
                entity_embed=self.entity_embed,
                movement_embed=self.movement_embed,
            )

        self.state_builder = _build(list(SELF_STATE_FIELDS))
        # Arsenal = pure INVENTORY: per-weapon readiness (ownership×ammo) +
        # arsenal powerup (QUAD). This is what the weapon head reads.
        # Deliberately excluded:
        #   - equipped-weapon id  → incumbent leak for the weapon head.
        #   - attack_finished     → first-class on the equipped-weapon token and
        #                           nowhere else (its single home); never in
        #                           the arsenal/inventory token.
        arsenal_fields = [WeaponReadiness(), VocabSum("powerup_arsenal")]
        self.arsenal_builder = _build(arsenal_fields)
        self.motion_builder = _build(list(MOTION_FIELDS))

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        acc = ObsAccessor(obs_dict)
        cls_token = self.cls_embed.view(1, 1, -1).expand(batch, 1, -1)
        state_token = self.state_builder(acc).unsqueeze(1)
        arsenal_token = self.arsenal_builder(acc).unsqueeze(1)
        motion_token = self.motion_builder(acc).unsqueeze(1)
        return torch.cat([cls_token, state_token, arsenal_token, motion_token], dim=1)
