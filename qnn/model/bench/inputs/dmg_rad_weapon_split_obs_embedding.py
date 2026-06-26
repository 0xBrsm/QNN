"""DmgRadWeaponSplitObsEmbedding — SplitSelf + held-weapon token.

Adds a 5th self subtoken to SplitSelfObsEmbedding:

    [CLS, state, arsenal, motion, weapon, [spatial..,] entities..]

The weapon token bundles everything about the CURRENTLY HELD weapon: damage +
splash radius (WT_DAMAGE, WT_RADIUS from the per-weapon static table), the
attack_finished refire cooldown, and the held-weapon id. This is the
held-weapon/incumbent token. ``attack_finished`` is a first-class, unconditional
member of every held-weapon token (its single home — the attack head's dominant
cooldown signal; see HeldWeaponSplitObsEmbedding). Inventory (which weapons are
owned/loaded) lives separately in the arsenal token's WeaponReadiness.

The token is a declarative ``TokenBuilder`` over named ``obs_fields`` — the
``weapon_dmg_rad`` computed field (the static-table damage+radius gather) and
``attack_finished`` scalar push through one Linear; ``weapon_id`` is the shared
entity-vocab embed. No bespoke Linear / gather lives here.
"""
from __future__ import annotations

import torch

from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import ScalarGroup, VocabEmbed
from qnn.model.bench.inputs.split_self_obs_embedding import SplitSelfObsEmbedding
from qnn.model.tokens.token_builder import TokenBuilder

# Held-weapon "incumbent" token fields: damage+radius (static-table gather) +
# attack_finished cooldown → one Linear, plus the held-weapon id embed.
DMG_RAD_WEAPON_FIELDS = (
    ScalarGroup(["weapon_dmg_rad", "attack_finished"]),
    VocabEmbed("weapon_id"),
)


class DmgRadWeaponSplitObsEmbedding(SplitSelfObsEmbedding):
    """SplitSelf + held-weapon token built from damage+radius scalars + id embed."""

    _N_SELF_TOKENS = 5

    def _init_self_components(self) -> None:
        super()._init_self_components()
        self.weapon_builder = TokenBuilder(
            self.d_model, DMG_RAD_WEAPON_FIELDS,
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
