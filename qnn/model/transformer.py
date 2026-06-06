"""Transformer trunk for the token observation contract.

Tokenizer converts packed observations into d_model token sequences.
TransformerTrunk runs self-attention over those tokens and returns:

  * ``self_readout`` — the self-token at position 0.
  * ``target_feat`` — attention-pooled actor feature from ``TargetPointer``:
    ``sum_i softmax(logits)[i] * entity_out[i]`` over actor slots only.
    Concatenated into the fused feature vector so action heads condition
    on the selected opponent.
  * ``target_logits`` — ``(B, MAX_TOKEN_OBJECTS)`` raw attention scores,
    supervised directly by BC labels via an auxiliary CE loss.  Not sampled
    as an action.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qnn.vocab import (
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    ENTITY_VOCAB_SIZE, ACTION_VOCAB_SIZE, MODALITY_VOCAB_SIZE,
    MAX_PLAYER_SLOTS, MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS,
    PROJECTILE_SCALAR_DIM, ACTOR_SCALAR_DIM, ITEM_SCALAR_DIM, MOVER_SCALAR_DIM,
)
from qnn.schema import (
    SELF_SCALAR_DIM, SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM,
)
from qnn.model.target import TargetPointer

_TOKEN_KIND_SELF = 0
_TOKEN_KIND_ENTITY = 1
_TOKEN_KIND_SPATIAL = 2


# ── Tokenizer ────────────────────────────────────────────────────


class Tokenizer(nn.Module):
    """Convert packed observations into d_model transformer tokens."""

    def __init__(
        self,
        d_model: int,
        *,
        self_weapon_embed_in_self: bool,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_weapon_embed_in_self = bool(self_weapon_embed_in_self)

        # Native-width → float adapters. Each is a no-op pass-through
        # when the obs dict already carries the legacy float tensors
        # (self_scalars / spatial_scalars / entity_scalars_raw), so
        # checkpoints trained on the f16 cache load and run unchanged.
        # The dataloader migration decides which format the model sees.
        from qnn.model.dequant import (
            SelfDequantizer, SpatialDequantizer, EntityDequantizer,
        )
        self.self_dequant    = SelfDequantizer()
        self.spatial_dequant = SpatialDequantizer()
        self.entity_dequant  = EntityDequantizer()

        # Self token
        self.self_proj = nn.Linear(SELF_SCALAR_DIM, d_model)
        # Current weapon as an additive contribution on the self token uses
        # the shared entity_embed (weapons live at ENTITY_IDS rows 3..10 by
        # design — see qnn.vocab). Pre-v22 used this path; v22 removed
        # weapon-from-self because the weapon head trivially predicts the
        # current weapon. The flag is retained for ablation.

        # Per-type entity projections
        self.proj_projectile = nn.Linear(PROJECTILE_SCALAR_DIM, d_model)
        self.proj_actor = nn.Linear(ACTOR_SCALAR_DIM, d_model)
        self.proj_item = nn.Linear(ITEM_SCALAR_DIM, d_model)
        self.proj_mover = nn.Linear(MOVER_SCALAR_DIM, d_model)

        # Spatial
        self.spatial_proj = nn.Linear(SPATIAL_SCALAR_DIM, d_model)

        # Shared embeddings
        self.entity_embed = nn.Embedding(ENTITY_VOCAB_SIZE, d_model)
        self.action_embed = nn.Embedding(ACTION_VOCAB_SIZE, d_model)
        self.modality_embed = nn.Embedding(MODALITY_VOCAB_SIZE, d_model)
        self.player_embed = nn.Embedding(MAX_PLAYER_SLOTS + 1, d_model)

        # Kind embeddings (self, entity, spatial)
        self.kind_embed = nn.Embedding(3, d_model)

        # Self-specific embeddings
        self.movement_embed = nn.Embedding(5, d_model)

        self.n_tokens = 1 + MAX_TOKEN_OBJECTS + SPATIAL_TOKEN_COUNT

    def _project_entity_scalars(
        self,
        entity_types: torch.Tensor,
        entity_scalars_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Project raw per-type scalars through type-specific layers.

        Args:
            entity_types: (batch, 16) long — token type tags, -1 for empty
            entity_scalars_raw: (batch, 16, max_scalar_dim) float — zero-padded raw scalars

        Returns:
            (batch, 16, d_model) — projected entity representations
        """
        batch, n_slots, _ = entity_scalars_raw.shape
        device = entity_scalars_raw.device
        result = torch.zeros(batch, n_slots, self.d_model, device=device, dtype=entity_scalars_raw.dtype)
        proj_map = {
            TOKEN_PROJECTILE: (self.proj_projectile, PROJECTILE_SCALAR_DIM),
            TOKEN_ACTOR: (self.proj_actor, ACTOR_SCALAR_DIM),
            TOKEN_ITEM: (self.proj_item, ITEM_SCALAR_DIM),
            TOKEN_MOVER: (self.proj_mover, MOVER_SCALAR_DIM),
        }
        for tok_type, (proj, sdim) in proj_map.items():
            mask = (entity_types == tok_type)  # (batch, 16)
            if mask.any():
                raw = entity_scalars_raw[mask][:, :sdim]  # (k, sdim)
                # Cast back to result.dtype so autocast (e.g. bf16) doesn't
                # mismatch the fp32 destination buffer.
                result[mask] = proj(raw).to(result.dtype)
        return result

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        *,
        include_spatial: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build transformer input tokens from parsed observation dict.

        obs_dict must contain:
            self_scalars: (batch, 16)
            self_weapon_id, self_armor_type_id, self_movement_id: (batch, 1)
            self_powerup_ids: (batch, 5)
            entity_types: (batch, 16) — token type tags, -1 for empty
            entity_scalars_proj: (batch, 16, d_model) — pre-projected by type
              OR entity_scalars_raw: (batch, 16, MAX_ENTITY_SCALAR_DIM) — raw scalars (projected here)
            entity_ids: (batch, 16, max_ids) — [subject, modality, player_id]
            entity_event_actions: (batch, 16, 4) — action IDs
            entity_event_sources: (batch, 16, 4) — source IDs
            entity_event_counts: (batch, 16) — valid event count
            spatial_scalars: (batch, 9, 13)
        """
        # Convert any engine-native fields to the legacy float tensors
        # the rest of this function expects. The three dequantizers
        # short-circuit if the obs already carries the legacy floats.
        obs_dict = self.entity_dequant(
            self.spatial_dequant(
                self.self_dequant(obs_dict)
            )
        )

        device = obs_dict["self_scalars"].device
        batch = obs_dict["self_scalars"].shape[0]

        # ---- Self token ----
        # Build all additive contributions as a list of (batch, 1, D) tensors,
        # then stack + sum in one reduction. The previous version did 4-5
        # sequential `self_token = self_token + …` adds — each a separate
        # kernel launch. Stacking lets the GPU do one reduction kernel,
        # cutting ROCm dispatch-queue churn that dominated this loop.
        contribs: list[torch.Tensor] = [self.self_proj(obs_dict["self_scalars"]).unsqueeze(1)]

        kind_self = torch.full((batch, 1), _TOKEN_KIND_SELF, dtype=torch.long, device=device)
        contribs.append(self.kind_embed(kind_self))

        # Self ID embeds
        if "self_armor_type_id" in obs_dict:
            aid = obs_dict["self_armor_type_id"].long().squeeze(-1).clamp(0, self.entity_embed.num_embeddings - 1)
            amask = (aid > 0).float().unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            contribs.append(self.entity_embed(aid).unsqueeze(1) * amask)
        if "self_movement_id" in obs_dict:
            mid = obs_dict["self_movement_id"].long().squeeze(-1).clamp(0, 4)
            contribs.append(self.movement_embed(mid).unsqueeze(1))
        if self.self_weapon_embed_in_self and "self_weapon_id" in obs_dict:
            wid = obs_dict["self_weapon_id"].long().squeeze(-1).clamp(
                0, self.entity_embed.num_embeddings - 1,
            )
            wmask = (wid > 0).float().unsqueeze(-1).unsqueeze(-1)
            contribs.append(self.entity_embed(wid).unsqueeze(1) * wmask)
        if "self_powerup_ids" in obs_dict:
            pids = obs_dict["self_powerup_ids"].long().clamp(0, self.entity_embed.num_embeddings - 1)  # (batch, 5)
            pmask = (pids > 0).float().unsqueeze(-1)  # (batch, 5, 1)
            contribs.append((self.entity_embed(pids) * pmask).sum(dim=1, keepdim=True))

        # One reduction kernel instead of N-1 adds. Stack to
        # (N, batch, 1, D) then sum dim 0 → (batch, 1, D).
        if len(contribs) > 1:
            self_token = torch.stack(contribs, dim=0).sum(dim=0)
        else:
            self_token = contribs[0]
        # ---- Entity tokens ----
        entity_types = obs_dict["entity_types"].long()  # (batch, 16)
        if "entity_scalars_proj" in obs_dict:
            entity_repr = obs_dict["entity_scalars_proj"]  # (batch, 16, d_model) — pre-projected
        else:
            entity_repr = self._project_entity_scalars(
                entity_types, obs_dict["entity_scalars_raw"],
            )
        # Phase 1 combat-only model: non-actor tokens are masked out of the
        # sequence entirely so the trunk only reasons over self, spatial, and actors.
        entity_mask = (entity_types == TOKEN_ACTOR)  # (batch, 16)

        # Subject embedding
        entity_subject = obs_dict["entity_ids"][:, :, 0].long().clamp(0, self.entity_embed.num_embeddings - 1)
        entity_repr = entity_repr + self.entity_embed(entity_subject)

        # Modality embedding
        entity_modality = obs_dict["entity_ids"][:, :, 1].long().clamp(0, self.modality_embed.num_embeddings - 1)
        entity_repr = entity_repr + self.modality_embed(entity_modality)

        # Player ID embedding (actors only)
        if obs_dict["entity_ids"].shape[2] >= 3:
            entity_player = obs_dict["entity_ids"][:, :, 2].long().clamp(0, self.player_embed.num_embeddings - 1)
            player_mask = (entity_types == TOKEN_ACTOR).unsqueeze(-1)
            entity_repr = entity_repr + self.player_embed(entity_player) * player_mask.float()

        # Events
        evt_actions = obs_dict["entity_event_actions"].long().clamp(0, self.action_embed.num_embeddings - 1)
        evt_sources = obs_dict["entity_event_sources"].long().clamp(0, self.entity_embed.num_embeddings - 1)
        evt_counts = obs_dict["entity_event_counts"].long()

        # (batch, 16, 4, d_model)
        evt_embed = self.action_embed(evt_actions) + self.entity_embed(evt_sources)
        evt_range = torch.arange(MAX_ENTITY_EVENTS, device=device).view(1, 1, MAX_ENTITY_EVENTS)
        evt_mask = (evt_range < evt_counts.unsqueeze(-1)).unsqueeze(-1).float()
        entity_repr = entity_repr + (evt_embed * evt_mask).sum(dim=2)

        # Kind embedding. n_entity_tokens is data-dependent (variable
        # per batch under the native variable-length entity wire), so
        # take the actual second-dim of entity_types rather than the
        # historical MAX_TOKEN_OBJECTS constant.
        n_entity_tokens = entity_types.shape[1]
        kind_entity = torch.full(
            (batch, n_entity_tokens), _TOKEN_KIND_ENTITY,
            dtype=torch.long, device=device,
        )
        entity_repr = entity_repr + self.kind_embed(kind_entity)

        if include_spatial:
            # ---- Spatial tokens ----
            spatial_scalars = obs_dict["spatial_scalars"]
            spatial_token = self.spatial_proj(spatial_scalars)
            kind_spatial = torch.full((batch, SPATIAL_TOKEN_COUNT), _TOKEN_KIND_SPATIAL, dtype=torch.long, device=device)
            spatial_token = spatial_token + self.kind_embed(kind_spatial)
            tokens = torch.cat([self_token, spatial_token, entity_repr], dim=1)
        else:
            tokens = torch.cat([self_token, entity_repr], dim=1)
        self_valid = torch.zeros((batch, 1), dtype=torch.bool, device=device)
        if include_spatial:
            spatial_valid = torch.zeros((batch, SPATIAL_TOKEN_COUNT), dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([self_valid, spatial_valid, ~entity_mask], dim=1)
        else:
            key_padding_mask = torch.cat([self_valid, ~entity_mask], dim=1)

        return tokens, key_padding_mask


# ── Transformer ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(attn_out)
        normed = self.ln2(x)
        return x + self.drop(self.ffn(normed))


class TransformerTrunk(nn.Module):
    """Token transformer readout for the token observation contract."""

    def __init__(
        self,
        *,
        obs_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
        self_weapon_embed_in_self: bool,
    ) -> None:
        super().__init__()
        del obs_dim
        self.d_model = int(d_model)
        self.output_dim = self.d_model
        self.tokenizer = Tokenizer(
            d_model=self.d_model,
            self_weapon_embed_in_self=self_weapon_embed_in_self,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.final_ln = nn.LayerNorm(self.d_model)
        # Plain TargetPointer for the no-GRU path. The GRU path in
        # _CombatObjectiveNet replaces this attribute with a richer
        # variant (gru_target_query / hard_target / inject_weapon / slot prior).
        self.target_pointer = TargetPointer(
            d_model=self.d_model,
            query_in_dim=self.d_model,
            inject_weapon=False,
            weapon_vocab=8,
            hard_target=False,
            linear_slot_prior=False,
            gt_dist_target_feat=False,
            prev_target_in_query=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self_readout, entity_outs, entity_mask = self.forward_raw(obs_dict)
        target_logits, target_feat, target_query = self.target_pointer(self_readout, entity_outs, entity_mask)
        return self_readout, target_feat, target_logits, target_query

    def forward_raw(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as ``forward`` but stops short of the target pointer.

        Returns ``(self_readout, entity_outs, entity_mask)`` so callers can
        run a recurrent layer over ``self_readout`` first and then point with
        the recurrent feature.
        """
        tokens, key_padding_mask = self.tokenizer(obs_dict)
        transformed = tokens
        for block in self.blocks:
            transformed = block(transformed, key_padding_mask=key_padding_mask)
        transformed = self.final_ln(transformed)

        self_readout = transformed[:, 0, :]  # (B, d_model)

        # Entity tokens follow self (1) + spatial (SPATIAL_TOKEN_COUNT).
        entity_start = 1 + SPATIAL_TOKEN_COUNT
        entity_outs = transformed[:, entity_start:entity_start + MAX_TOKEN_OBJECTS, :]
        entity_types = obs_dict["entity_types"].long()

        entity_mask = (entity_types == TOKEN_ACTOR)  # (B, N)
        return self_readout, entity_outs, entity_mask
