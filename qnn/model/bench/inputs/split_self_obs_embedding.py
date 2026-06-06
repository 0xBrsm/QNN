"""SplitSelfObsEmbedding — bench obs embedding that splits self into subtokens.

Subclass of the canonical ``ObsEmbedding``. Where the canonical produces
a single self token, this variant produces four:

    [CLS, state, arsenal, motion, [spatial_0..8,] entity_0..N-1]

The three self subtokens route disjoint scalar bundles + ID embeds:

* ``state``    health, effective_armor, armor_type, state powerups
                (PENT / RING / MEGAHEALTH)
* ``arsenal``  attack_finished, per-weapon readiness×subject sum,
                optional held-weapon embed, arsenal powerups (QUAD)
* ``motion``   vel_xyz, view_pitch, movement_id, motion powerups (SUIT)

CLS is a learnable parameter at slot 0; the attention pools the rest
of the stream into it. Encoders slice ``self_readout`` from CLS. Use
this obs embedding to ablate "does splitting the self bundle into
specialized subtokens help the attention?" — the canonical single-self
design is the baseline.

Everything outside the self block (entity / spatial / embeddings /
event handling / kind tags) inherits from ``ObsEmbedding`` unchanged.

Pair with ``Network(obs_embedding=SplitSelfObsEmbedding(...))`` (and
optionally ``encoder=PreAttnEncoder()`` for no-attention probes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qnn.schema import (
    SELF_ARSENAL_SCALAR_DIM, SELF_MOTION_SCALAR_DIM, SELF_STATE_SCALAR_DIM,
)
from qnn.vocab import ENTITY_IDS
from qnn.model.transformer import ObsEmbedding


# Weapon vocab IDs in axe-first impulse order, paired with the (B, 8)
# per-weapon readiness vector the dequant emits. The arsenal subtoken
# sums entity_embed[weapon_id] × readiness across all 8.
_WEAPON_SUBJECT_IDS = (
    ENTITY_IDS["AXE"],
    ENTITY_IDS["SHOTGUN"],
    ENTITY_IDS["SUPER_SHOTGUN"],
    ENTITY_IDS["NAILGUN"],
    ENTITY_IDS["SUPER_NAILGUN"],
    ENTITY_IDS["GRENADE_LAUNCHER"],
    ENTITY_IDS["ROCKET_LAUNCHER"],
    ENTITY_IDS["THUNDERBOLT"],
)


class SplitSelfObsEmbedding(ObsEmbedding):
    """4-self-token variant — CLS + state + arsenal + motion."""

    _N_SELF_TOKENS = 4

    def _init_self_components(self) -> None:
        # CLS — learnable trunk-readout token at slot 0.
        self.cls_embed = nn.Parameter(torch.zeros(self.d_model))
        # Self subtoken projections.
        self.self_proj_state   = nn.Linear(SELF_STATE_SCALAR_DIM,   self.d_model)
        self.self_proj_arsenal = nn.Linear(SELF_ARSENAL_SCALAR_DIM, self.d_model)
        self.self_proj_motion  = nn.Linear(SELF_MOTION_SCALAR_DIM,  self.d_model)
        self.register_buffer(
            "_weapon_subject_ids",
            torch.tensor(_WEAPON_SUBJECT_IDS, dtype=torch.long),
            persistent=False,
        )

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        cls_token = self.cls_embed.view(1, 1, -1).expand(batch, 1, -1)

        # ---- State subtoken ----
        state_token = self.self_proj_state(obs_dict["self_state_scalars"]).unsqueeze(1)
        if "self_armor_type_id" in obs_dict:
            aid = obs_dict["self_armor_type_id"].long().squeeze(-1).clamp(0, vocab_max)
            amask = (aid > 0).float().unsqueeze(-1).unsqueeze(-1)
            state_token = state_token + self.entity_embed(aid).unsqueeze(1) * amask
        if "self_state_powerup_ids" in obs_dict:
            pids = obs_dict["self_state_powerup_ids"].long().clamp(0, vocab_max)
            pmask = (pids > 0).float().unsqueeze(-1)
            state_token = state_token + (self.entity_embed(pids) * pmask).sum(dim=1, keepdim=True)

        # ---- Arsenal subtoken ----
        arsenal_token = self.self_proj_arsenal(obs_dict["self_arsenal_scalars"]).unsqueeze(1)
        # Per-weapon readiness × entity_embed sum. self_weapon_readiness is
        # (B, 8) axe-first; absent weapons have readiness=0 so their
        # contribution drops out without a mask.
        readiness = obs_dict["self_weapon_readiness"]
        weapon_embeds = self.entity_embed(self._weapon_subject_ids)
        arsenal_token = arsenal_token + torch.einsum(
            "bw,wd->bd", readiness.to(weapon_embeds.dtype), weapon_embeds,
        ).unsqueeze(1)
        if self.self_weapon_embed_in_self and "self_weapon_id" in obs_dict:
            wid = obs_dict["self_weapon_id"].long().squeeze(-1).clamp(0, vocab_max)
            wmask = (wid > 0).float().unsqueeze(-1).unsqueeze(-1)
            arsenal_token = arsenal_token + self.entity_embed(wid).unsqueeze(1) * wmask
        if "self_arsenal_powerup_ids" in obs_dict:
            pids = obs_dict["self_arsenal_powerup_ids"].long().clamp(0, vocab_max)
            pmask = (pids > 0).float().unsqueeze(-1)
            arsenal_token = arsenal_token + (self.entity_embed(pids) * pmask).sum(dim=1, keepdim=True)

        # ---- Motion subtoken ----
        motion_token = self.self_proj_motion(obs_dict["self_motion_scalars"]).unsqueeze(1)
        if "self_movement_id" in obs_dict:
            mid = obs_dict["self_movement_id"].long().squeeze(-1).clamp(0, 4)
            motion_token = motion_token + self.movement_embed(mid).unsqueeze(1)
        if "self_motion_powerup_ids" in obs_dict:
            pids = obs_dict["self_motion_powerup_ids"].long().clamp(0, vocab_max)
            pmask = (pids > 0).float().unsqueeze(-1)
            motion_token = motion_token + (self.entity_embed(pids) * pmask).sum(dim=1, keepdim=True)

        return torch.cat([cls_token, state_token, arsenal_token, motion_token], dim=1)
