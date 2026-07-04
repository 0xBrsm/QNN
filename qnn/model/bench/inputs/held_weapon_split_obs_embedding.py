"""HeldWeaponSplitObsEmbedding — SplitSelf + a dedicated held-weapon token.

Adds a 5th self subtoken to SplitSelfObsEmbedding:

    [CLS, state, arsenal, motion, weapon, [spatial..,] entities..]

The weapon token = the held weapon's STATIC stats scalars (the (9,7)
build_model_weapon_scalars row — v_horiz / gravity / etc., gathered by the held
weapon id) + the ``attack_finished`` refire cooldown + the SAME weapon-id embed
from the shared entity vocab. This is the held-weapon identity + trajectory the
canonical SplitSelf never exposed (its arsenal token only carried per-weapon
readiness; the optional weapon_id embed was off).

``attack_finished`` is first-class and unconditional here — the held-weapon token
is its single home (the attack head's dominant cooldown signal). It is not gated
by any flag and does not appear in the base state/arsenal tokens.

The token is a declarative ``TokenBuilder`` over named ``obs_fields`` — the
``weapon_static`` computed field (the static-table gather) pushes through one
Linear; ``weapon_id`` is the shared entity-vocab embed. No bespoke
Linear / static-table buffer / gather lives here.
"""
from __future__ import annotations

import torch

from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import ScalarGroup, VocabEmbed
from qnn.model.bench.inputs.split_self_obs_embedding import SplitSelfObsEmbedding
from qnn.model.tokens.token_builder import TokenBuilder

# Held-weapon identity token: static per-weapon stats (impulse-gathered) +
# the attack_finished refire cooldown → one Linear, plus the held-weapon id
# embed from the shared entity vocab. attack_finished is a FIRST-CLASS,
# UNCONDITIONAL member of the held-weapon token (it is the attack head's
# dominant signal) and lives nowhere else — there is no flag to remove it.
HELD_WEAPON_FIELDS = (
    ScalarGroup(["weapon_static", "attack_finished"]),
    VocabEmbed("weapon_id"),
)


class HeldWeaponSplitObsEmbedding(SplitSelfObsEmbedding):
    """SplitSelf with an extra held-weapon token (stats scalars + id embed)."""

    _N_SELF_TOKENS = 5

    def _init_self_components(self) -> None:
        super()._init_self_components()  # CLS + state/arsenal/motion builders
        self.weapon_builder = TokenBuilder(
            self.d_model, HELD_WEAPON_FIELDS,
            entity_embed=self.entity_embed,
            movement_embed=self.movement_embed,
        )

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        base = super()._build_self_block(obs_dict, batch, device, vocab_max)  # (B, 4, D)
        weapon_token = self.weapon_builder(ObsAccessor(obs_dict)).unsqueeze(1)  # (B, 1, D)
        return torch.cat([base, weapon_token], dim=1)                          # (B, 5, D)
