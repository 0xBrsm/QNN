"""Trunk-style projected features for head probes (no transformer, no GRU).

The full BC model's head input is built from trunk-projected self / actor
tokens (linear scalar projections + learned subject / modality / player /
event / movement / weapon / kind embeddings) further mixed by a
self-attention trunk and (optionally) a recurrent self-readout. Head
probes that bypass the trunk + GRU but want to test head architecture
under *otherwise* trunk-faithful inputs need the same projections and
embeddings — just without the attention mixing or temporal carry.

This module exposes the shared encoder that produces those inputs.
Each head probe (``fire_token``, eventually ``move_token`` etc.) reuses
``TokenizedFeatureEncoder``: it runs ``qnn.model.transformer.Tokenizer``
to get the pre-attention token stack, takes the self-token at row 0,
soft-pools the actor entity tokens by the labeler's GT slot
distribution, and returns ``cat(self_token, target_feat)``.

Privileged input: the GT target distribution is consumed via the
``target_dist_slot`` parameter that the BC supervised loop already
passes alongside the obs (see ``QNNPolicy.supervised_step``). This is
the same renormalized 16-slot tensor the canonical model's
``gt_dist_target_feat`` mode receives; the probe just consumes it
unconditionally rather than gating on a learned pointer's softmax.
The probe is therefore an oracle-pointer ablation — it isolates head
capacity from pointer error.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qnn.model.transformer import Tokenizer
from qnn.schema import SELF_SCALAR_DIM, WEAPON_HEAD_SIZE
from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR


class TokenizedFeatureEncoder(nn.Module):
    """Trunk tokenizer + GT-soft-pooled target_feat. No attention, no GRU.

    Forward consumes an obs dict in the canonical BC layout
    (``entity_scalars_raw``, ``entity_event_*``, ``self_*``,
    ``spatial_scalars``, ``entity_types``, ``entity_ids``) plus a
    pre-renormalized GT slot distribution ``target_dist_slot`` of shape
    ``(B, MAX_TOKEN_OBJECTS)``. Returns ``cat(self_token, [weapon_embed,]
    target_feat)`` of shape ``((2 + weapon_embed_concat) * d_model,)``.

    The encoder asks Tokenizer for the self/entity-only layout because the
    fire ablation only cares about self state + the pointed-at actor, not
    arena geometry.

    Knobs:
      * ``self_weapon_embed_in_self``: when True, Tokenizer adds the
        current-weapon embed onto the self token (additive).
      * ``strip_self_embeds``: when True, ignore the Tokenizer's
        self-token output and use a fresh ``Linear(SELF_SCALAR_DIM, D)``
        on raw ``self_scalars`` — no kind/armor/movement/powerup/weapon
        adds. Useful for probing how much of the self token's info is
        in the raw scalars vs the learned embeds.
      * ``weapon_embed_concat``: when True, allocate a dedicated
        ``Embedding(WEAPON_HEAD_SIZE+1, D)`` and concatenate
        ``weapon_embed(self_weapon_id)`` between the self token and
        target_feat. Mutually exclusive with
        ``self_weapon_embed_in_self`` (would otherwise double-count).
    """

    def __init__(
        self,
        *,
        d_model: int,
        self_weapon_embed_in_self: bool,
        strip_self_embeds: bool,
        weapon_embed_concat: bool,
    ) -> None:
        super().__init__()
        if weapon_embed_concat and self_weapon_embed_in_self:
            raise ValueError(
                "weapon_embed_concat=True and self_weapon_embed_in_self=True "
                "would double-count current weapon. Set the additive one False."
            )
        self.d_model = int(d_model)
        self.strip_self_embeds = bool(strip_self_embeds)
        self.tokenizer = Tokenizer(
            d_model=self.d_model,
            self_weapon_embed_in_self=bool(self_weapon_embed_in_self),
        )
        if self.strip_self_embeds:
            # Independent projection — no kind/armor/movement/powerup/weapon adds.
            self.self_proj_stripped = nn.Linear(SELF_SCALAR_DIM, self.d_model)
        if weapon_embed_concat:
            self.weapon_embed_concat = nn.Embedding(WEAPON_HEAD_SIZE + 1, self.d_model)
            nn.init.normal_(self.weapon_embed_concat.weight, std=0.02)
        else:
            self.weapon_embed_concat = None

    @property
    def output_dim(self) -> int:
        n_blocks = 2 + (1 if self.weapon_embed_concat is not None else 0)
        return n_blocks * self.d_model

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        target_dist_slot: torch.Tensor,
    ) -> torch.Tensor:
        tokens, _ = self.tokenizer(obs, include_spatial=False)
        # Token layout without spatial: self (1) + entities (MAX_TOKEN_OBJECTS).
        if self.strip_self_embeds:
            self_token = self.self_proj_stripped(obs["self_scalars"])            # (B, D)
        else:
            self_token = tokens[:, 0, :]                                         # (B, D)
        entity_start = 1
        entity_tokens = tokens[:, entity_start:entity_start + MAX_TOKEN_OBJECTS, :]  # (B, N, D)

        slot_dist = target_dist_slot.to(entity_tokens.dtype)                    # (B, N)
        # Zero out non-actor slots so a stale GT mass on a non-actor slot
        # (shouldn't happen — labeler builds dist over actor slots — but
        # cheap to enforce) can't leak into target_feat. Also zero target_feat
        # for actor-empty scenes, mirroring TargetPointer.forward.
        entity_types = obs["entity_types"].long()
        actor_mask = (entity_types == TOKEN_ACTOR).to(slot_dist.dtype)           # (B, N)
        has_actor = (actor_mask.sum(dim=-1, keepdim=True) > 0).to(slot_dist.dtype) # (B, 1)
        weights = slot_dist * actor_mask                                          # (B, N)
        target_feat = (weights.unsqueeze(-1) * entity_tokens).sum(dim=1) * has_actor  # (B, D)

        parts = [self_token]
        if self.weapon_embed_concat is not None:
            # weapon_embed_concat is impulse-indexed (size WEAPON_HEAD_SIZE+1).
            # obs.self_weapon_id is ENTITY_IDS-encoded (0=NONE, 3..10=axe..LG):
            # impulse = max(0, eid - 2). Inlined for ROCm kernel-launch
            # efficiency — calling out to a helper added ~40% overhead.
            wid = (obs["self_weapon_id"].long().squeeze(-1) - 2).clamp(
                0, WEAPON_HEAD_SIZE,
            )
            parts.append(self.weapon_embed_concat(wid).to(self_token.dtype))
        parts.append(target_feat)
        return torch.cat(parts, dim=-1)
