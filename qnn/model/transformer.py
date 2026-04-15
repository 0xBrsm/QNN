"""Transformer trunk for the token observation contract.

Tokenizer converts packed observations into d_model token sequences.
TransformerTrunk runs self-attention over those tokens and reads out
a summary vector from the self-token at position 0.
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
    ACTION_HISTORY_LEN, ACTION_HISTORY_DIM,
)

_TOKEN_KIND_SELF = 0
_TOKEN_KIND_ENTITY = 1
_TOKEN_KIND_SPATIAL = 2
_TOKEN_KIND_ACTION = 3


# ── Tokenizer ────────────────────────────────────────────────────


class Tokenizer(nn.Module):
    """Convert packed observations into d_model transformer tokens."""

    def __init__(self, d_model: int, action_history_tokens: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_history_tokens = min(int(action_history_tokens), ACTION_HISTORY_LEN)

        # Self token
        self.self_proj = nn.Linear(SELF_SCALAR_DIM, d_model)

        # Per-type entity projections
        self.proj_projectile = nn.Linear(PROJECTILE_SCALAR_DIM, d_model)
        self.proj_actor = nn.Linear(ACTOR_SCALAR_DIM, d_model)
        self.proj_item = nn.Linear(ITEM_SCALAR_DIM, d_model)
        self.proj_mover = nn.Linear(MOVER_SCALAR_DIM, d_model)

        # Spatial
        self.spatial_proj = nn.Linear(SPATIAL_SCALAR_DIM, d_model)

        # Action history
        if self.action_history_tokens > 0:
            self.action_proj = nn.Linear(ACTION_HISTORY_DIM, d_model)
            self.action_pos_embed = nn.Embedding(ACTION_HISTORY_LEN, d_model)

        # Shared embeddings
        self.entity_embed = nn.Embedding(ENTITY_VOCAB_SIZE, d_model)
        self.action_embed = nn.Embedding(ACTION_VOCAB_SIZE, d_model)
        self.modality_embed = nn.Embedding(MODALITY_VOCAB_SIZE, d_model)
        self.player_embed = nn.Embedding(MAX_PLAYER_SLOTS + 1, d_model)

        # Kind embeddings (self, entity, spatial, action)
        self.kind_embed = nn.Embedding(4, d_model)

        # Self-specific embeddings
        self.movement_embed = nn.Embedding(5, d_model)

        self.n_tokens = 1 + MAX_TOKEN_OBJECTS + SPATIAL_TOKEN_COUNT + self.action_history_tokens

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
                result[mask] = proj(raw)
        return result

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build transformer input tokens from parsed observation dict.

        obs_dict must contain:
            self_scalars: (batch, 14)
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
            action_history: (batch, 8, 8) [optional]
        """
        device = obs_dict["self_scalars"].device
        batch = obs_dict["self_scalars"].shape[0]

        # ---- Self token ----
        self_token = self.self_proj(obs_dict["self_scalars"])

        # Kind embedding
        kind_self = torch.full((batch, 1), _TOKEN_KIND_SELF, dtype=torch.long, device=device)
        self_token = self_token.unsqueeze(1) + self.kind_embed(kind_self)

        # Self ID embeds
        if "self_weapon_id" in obs_dict:
            wid = obs_dict["self_weapon_id"].long().squeeze(-1).clamp(0, self.entity_embed.num_embeddings - 1)
            self_token = self_token + self.entity_embed(wid).unsqueeze(1)
        if "self_armor_type_id" in obs_dict:
            aid = obs_dict["self_armor_type_id"].long().squeeze(-1).clamp(0, self.entity_embed.num_embeddings - 1)
            amask = (aid > 0).float().unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            self_token = self_token + self.entity_embed(aid).unsqueeze(1) * amask
        if "self_movement_id" in obs_dict:
            mid = obs_dict["self_movement_id"].long().squeeze(-1).clamp(0, 4)
            self_token = self_token + self.movement_embed(mid).unsqueeze(1)
        if "self_powerup_ids" in obs_dict:
            pids = obs_dict["self_powerup_ids"].long().clamp(0, self.entity_embed.num_embeddings - 1)  # (batch, 5)
            pmask = (pids > 0).float().unsqueeze(-1)  # (batch, 5, 1)
            self_token = self_token + (self.entity_embed(pids) * pmask).sum(dim=1, keepdim=True)
        # ---- Entity tokens ----
        entity_types = obs_dict["entity_types"].long()  # (batch, 16)
        if "entity_scalars_proj" in obs_dict:
            entity_repr = obs_dict["entity_scalars_proj"]  # (batch, 16, d_model) — pre-projected
        else:
            entity_repr = self._project_entity_scalars(
                entity_types, obs_dict["entity_scalars_raw"],
            )
        entity_mask = (entity_types >= 0)  # (batch, 16)

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

        # Kind embedding
        kind_entity = torch.full((batch, MAX_TOKEN_OBJECTS), _TOKEN_KIND_ENTITY, dtype=torch.long, device=device)
        entity_repr = entity_repr + self.kind_embed(kind_entity)

        # ---- Spatial tokens ----
        spatial_scalars = obs_dict["spatial_scalars"]
        spatial_token = self.spatial_proj(spatial_scalars)
        kind_spatial = torch.full((batch, SPATIAL_TOKEN_COUNT), _TOKEN_KIND_SPATIAL, dtype=torch.long, device=device)
        spatial_token = spatial_token + self.kind_embed(kind_spatial)

        # ---- Action history ----
        if self.action_history_tokens > 0 and "action_history" in obs_dict:
            n = self.action_history_tokens
            action_slice = obs_dict["action_history"][:, -n:, :]
            action_token = self.action_proj(action_slice)
            kind_action = torch.full((batch, n), _TOKEN_KIND_ACTION, dtype=torch.long, device=device)
            action_token = action_token + self.kind_embed(kind_action)
            pos_ids = torch.arange(ACTION_HISTORY_LEN - n, ACTION_HISTORY_LEN, dtype=torch.long, device=device)
            action_token = action_token + self.action_pos_embed(pos_ids).unsqueeze(0)
            action_valid = action_slice.abs().sum(dim=-1) > 0

            tokens = torch.cat([self_token, spatial_token, action_token, entity_repr], dim=1)
            self_valid = torch.zeros((batch, 1), dtype=torch.bool, device=device)
            spatial_valid = torch.zeros((batch, SPATIAL_TOKEN_COUNT), dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([self_valid, spatial_valid, ~action_valid, ~entity_mask], dim=1)
        else:
            tokens = torch.cat([self_token, spatial_token, entity_repr], dim=1)
            self_valid = torch.zeros((batch, 1), dtype=torch.bool, device=device)
            spatial_valid = torch.zeros((batch, SPATIAL_TOKEN_COUNT), dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([self_valid, spatial_valid, ~entity_mask], dim=1)

        return tokens, key_padding_mask


# ── Transformer ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        obs_dim: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        readout: str = "self",
        action_history_tokens: int = 0,
    ) -> None:
        super().__init__()
        del obs_dim, readout
        self.output_dim = int(d_model)
        self.tokenizer = Tokenizer(d_model=d_model, action_history_tokens=action_history_tokens)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, key_padding_mask = self.tokenizer(obs_dict)
        transformed = tokens
        for block in self.blocks:
            transformed = block(transformed, key_padding_mask=key_padding_mask)
        return self.final_ln(transformed[:, 0, :])
