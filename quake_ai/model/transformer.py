"""Transformer trunk for the native token observation contract.

Token attention reads self, object, event, and spatial groups and emits a
current-frame summary vector from the configured readout token.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quake_ai.model.observation import (
    ACTION_HISTORY_DIM,
    ACTION_HISTORY_LEN,
    MAX_OBJECT_TOKENS,
    OBJECT_ID_DIM,
    OBJECT_SCALAR_DIM,
    SELF_SCALAR_DIM,
    SPATIAL_SCALAR_DIM,
    SPATIAL_TOKEN_COUNT,
)
from quake_ai.vocab import ACTION_IDS, MAX_PLAYER_SLOTS, MODALITY_IDS, QUALIFIER_IDS, SPATIAL_SECTOR_IDS, SUBJECT_IDS

_TOKEN_KIND_SELF = 0
_TOKEN_KIND_OBJECT = 1
_TOKEN_KIND_SPATIAL = 2
_TOKEN_KIND_ACTION = 3


class TokenObservationTokenizer(nn.Module):
    """Convert packed token observations into d_model transformer tokens."""

    def __init__(self, d_model: int, action_history_tokens: int = 0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.action_history_tokens = min(int(action_history_tokens), ACTION_HISTORY_LEN)

        self.self_proj = nn.Linear(SELF_SCALAR_DIM, d_model)
        self.object_proj = nn.Linear(OBJECT_SCALAR_DIM, d_model)
        self.spatial_proj = nn.Linear(SPATIAL_SCALAR_DIM, d_model)
        if self.action_history_tokens > 0:
            self.action_proj = nn.Linear(ACTION_HISTORY_DIM, d_model)
            self.action_pos_embed = nn.Embedding(ACTION_HISTORY_LEN, d_model)

        self.kind_embed = nn.Embedding(4, d_model)  # self, object, spatial, action
        self.movement_embed = nn.Embedding(5, d_model)
        self.cluster_embed = nn.Embedding(256, d_model)
        self.subject_embed = nn.Embedding(32, d_model)  # v8: IDs 0-31
        self.action_embed = nn.Embedding(max(ACTION_IDS.values()) + 1, d_model)
        self.qualifier_embed = nn.Embedding(max(QUALIFIER_IDS.values()) + 1, d_model)
        self.modality_embed = nn.Embedding(max(MODALITY_IDS.values()) + 1, d_model)
        self.player_embed = nn.Embedding(MAX_PLAYER_SLOTS + 1, d_model)
        self.spatial_sector_embed = nn.Embedding(max(SPATIAL_SECTOR_IDS.values()) + 1, d_model)

        # Constant tensors registered as buffers so they move with .to(device)
        # and are not recreated every forward pass.
        self.register_buffer('_kind_self', torch.tensor([_TOKEN_KIND_SELF], dtype=torch.long))
        self.register_buffer('_kind_object', torch.full((MAX_OBJECT_TOKENS,), _TOKEN_KIND_OBJECT, dtype=torch.long))
        self.register_buffer('_kind_spatial', torch.full((SPATIAL_TOKEN_COUNT,), _TOKEN_KIND_SPATIAL, dtype=torch.long))
        self.register_buffer('_pu_range', torch.arange(5, dtype=torch.long))

        # v8: no CLS token — self token is position 0
        # 1 (self) + SPATIAL + action_history + MAX_OBJECT_TOKENS
        self.n_tokens = 1 + MAX_OBJECT_TOKENS + SPATIAL_TOKEN_COUNT + self.action_history_tokens

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        self_scalars = obs_dict["self_scalars"]
        self_weapon_id = obs_dict["self_weapon_id"].long().squeeze(-1)
        self_movement_id = obs_dict.get("self_movement_id")
        self_cluster_id = obs_dict.get("self_cluster_id")
        self_armor_type_id = obs_dict.get("self_armor_type_id")
        self_powerup_ids = obs_dict.get("self_powerup_ids")
        self_powerup_count = obs_dict.get("self_powerup_count")
        object_ids = obs_dict["object_ids"].long()
        object_scalars = obs_dict["object_scalars"]
        object_mask = obs_dict["object_mask"].bool()
        spatial_ids = obs_dict["spatial_ids"].long()
        spatial_scalars = obs_dict["spatial_scalars"]

        batch = int(self_scalars.shape[0])
        device = self_scalars.device
        _se_max = self.subject_embed.num_embeddings - 1

        if self_movement_id is None:
            self_movement_id = torch.zeros((batch,), dtype=torch.long, device=device)
        else:
            self_movement_id = self_movement_id.long().squeeze(-1)
        if self_cluster_id is None:
            self_cluster_id = torch.zeros((batch,), dtype=torch.long, device=device)
        else:
            self_cluster_id = self_cluster_id.long().squeeze(-1)
        if self_armor_type_id is None:
            self_armor_type_id = torch.zeros((batch,), dtype=torch.long, device=device)
        else:
            self_armor_type_id = self_armor_type_id.long().squeeze(-1)

        # v8: self token — subject_embed for weapon, armor_type, and powerups
        self_token = (
            self.self_proj(self_scalars).unsqueeze(1)
            + self.kind_embed(self._kind_self.expand(batch, -1))
            + self.subject_embed(self_weapon_id.clamp(min=0, max=_se_max)).unsqueeze(1)
            + self.subject_embed(self_armor_type_id.clamp(min=0, max=_se_max)).unsqueeze(1)
            + self.movement_embed(self_movement_id.clamp(min=0, max=self.movement_embed.num_embeddings - 1)).unsqueeze(1)
            + self.cluster_embed(self_cluster_id.clamp(min=0, max=self.cluster_embed.num_embeddings - 1)).unsqueeze(1)
        )

        # v8: variable powerup stacking — sum subject_embed(p) for active powerups
        if self_powerup_ids is not None and self_powerup_count is not None:
            pu_ids = self_powerup_ids.long().clamp(min=0, max=_se_max)  # (batch, 5)
            pu_count = self_powerup_count.long().squeeze(-1)  # (batch,)
            pu_embeds = self.subject_embed(pu_ids)  # (batch, 5, d_model)
            # Mask: only sum entries up to powerup_count
            pu_range = self._pu_range.unsqueeze(0)  # (1, 5)
            pu_mask = pu_range < pu_count.unsqueeze(1)  # (batch, 5)
            pu_sum = (pu_embeds * pu_mask.unsqueeze(-1).to(dtype=pu_embeds.dtype)).sum(dim=1, keepdim=True)  # (batch, 1, d_model)
            self_token = self_token + pu_sum

        # --- object tokens ---
        object_token = self.object_proj(object_scalars)
        object_token = object_token + self.kind_embed(
            self._kind_object.expand(batch, -1)
        )
        object_token = object_token + self.subject_embed(object_ids[:, :, 0].clamp(min=0, max=_se_max))
        object_token = object_token + self.qualifier_embed(object_ids[:, :, 1].clamp(min=0, max=self.qualifier_embed.num_embeddings - 1))
        object_token = object_token + self.modality_embed(object_ids[:, :, 2].clamp(min=0, max=self.modality_embed.num_embeddings - 1))
        object_token = object_token + self.player_embed(object_ids[:, :, 3].clamp(min=0, max=self.player_embed.num_embeddings - 1))
        # Column 4: cluster_id
        object_cluster_ids = object_ids[:, :, 4]
        object_token = object_token + self.cluster_embed(
            object_cluster_ids.clamp(min=0, max=self.cluster_embed.num_embeddings - 1)
        )
        # v8 columns 5-6: powerup_id and weapon_id (subject_embed additions)
        if int(object_ids.shape[-1]) > 5:
            obj_powerup_ids = object_ids[:, :, 5].clamp(min=0, max=_se_max)
            # Only add powerup embed where ID > 0 (NONE=0 would add noise)
            obj_pu_mask = (object_ids[:, :, 5] > 0).unsqueeze(-1).to(dtype=object_token.dtype)
            object_token = object_token + self.subject_embed(obj_powerup_ids) * obj_pu_mask
        if int(object_ids.shape[-1]) > 6:
            obj_weapon_ids = object_ids[:, :, 6].clamp(min=0, max=_se_max)
            obj_wep_mask = (object_ids[:, :, 6] > 0).unsqueeze(-1).to(dtype=object_token.dtype)
            object_token = object_token + self.subject_embed(obj_weapon_ids) * obj_wep_mask

        # Route embedding
        route_cluster_ids = obs_dict.get("object_route_cluster_ids")
        if route_cluster_ids is not None:
            route_raw = route_cluster_ids.long()
            route_mask = route_raw >= 0
            route_cluster_ids = route_raw.clamp(min=0, max=self.cluster_embed.num_embeddings - 1)
            route_embeds = self.cluster_embed(route_cluster_ids)
            route_embed = (route_embeds * route_mask.unsqueeze(-1).to(dtype=route_embeds.dtype)).sum(dim=2)
        else:
            route_embed = torch.zeros((batch, MAX_OBJECT_TOKENS, self.d_model), dtype=object_token.dtype, device=device)
        object_token = object_token + route_embed

        # --- per-entity events: embed and sum onto owner object token ---
        obj_event_ids = obs_dict.get("object_event_ids")
        obj_event_scalars = obs_dict.get("object_event_scalars")
        obj_event_counts = obs_dict.get("object_event_counts")
        if obj_event_ids is not None and obj_event_scalars is not None and obj_event_counts is not None:
            # obj_event_ids: (batch, 16, 4, 3) — subject, action, qualifier
            # obj_event_scalars: (batch, 16, 4) — recency per slot
            # obj_event_counts: (batch, 16) — valid slot count
            oe_ids = obj_event_ids.long()  # (batch, 16, 4, 3)
            oe_rec = obj_event_scalars.unsqueeze(-1)  # (batch, 16, 4, 1) — recency as weight
            oe_counts = obj_event_counts.long()  # (batch, 16)
            # Build mask: slot < count
            oe_range = torch.arange(4, device=device).view(1, 1, 4)  # (1, 1, 4)
            oe_mask = (oe_range < oe_counts.unsqueeze(-1)).unsqueeze(-1).to(dtype=object_token.dtype)  # (batch, 16, 4, 1)
            # Embed each event slot: subject + action + qualifier, scaled by recency
            oe_embed = (
                self.subject_embed(oe_ids[:, :, :, 0].clamp(min=0, max=_se_max))
                + self.action_embed(oe_ids[:, :, :, 1].clamp(min=0, max=self.action_embed.num_embeddings - 1))
                + self.qualifier_embed(oe_ids[:, :, :, 2].clamp(min=0, max=self.qualifier_embed.num_embeddings - 1))
            )  # (batch, 16, 4, d_model)
            oe_embed = oe_embed * oe_mask * oe_rec.to(dtype=oe_embed.dtype)
            object_token = object_token + oe_embed.sum(dim=2)  # sum over 4 event slots → (batch, 16, d_model)

        # --- spatial tokens ---
        spatial_token = self.spatial_proj(spatial_scalars)
        spatial_token = spatial_token + self.kind_embed(
            self._kind_spatial.expand(batch, -1)
        )
        spatial_token = spatial_token + self.spatial_sector_embed(
            spatial_ids.clamp(min=0, max=self.spatial_sector_embed.num_embeddings - 1)
        )

        # v8: no CLS token — self is position 0
        # Sequence: self(1) + spatial(9) + action_history(N) + objects(16)
        if self.action_history_tokens > 0:
            action_history = obs_dict["action_history"]  # (batch, ACTION_HISTORY_LEN, ACTION_HISTORY_DIM)
            n = self.action_history_tokens
            action_slice = action_history[:, -n:, :]  # (batch, n, ACTION_HISTORY_DIM)
            action_token = self.action_proj(action_slice)  # (batch, n, d_model)
            action_token = action_token + self.kind_embed(
                torch.full((batch, n), _TOKEN_KIND_ACTION, dtype=torch.long, device=device)
            )
            pos_ids = torch.arange(ACTION_HISTORY_LEN - n, ACTION_HISTORY_LEN, dtype=torch.long, device=device)
            action_token = action_token + self.action_pos_embed(pos_ids).unsqueeze(0)
            action_valid = action_slice.abs().sum(dim=-1) > 0  # (batch, n)
            tokens = torch.cat([self_token, spatial_token, action_token, object_token], dim=1)
            # self(1) always valid, spatial(9) always valid
            self_valid = torch.zeros((batch, 1), dtype=torch.bool, device=device)  # False = not masked
            spatial_valid = torch.zeros((batch, SPATIAL_TOKEN_COUNT), dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([self_valid, spatial_valid, ~action_valid, ~object_mask], dim=1)
        else:
            tokens = torch.cat([self_token, spatial_token, object_token], dim=1)
            self_valid = torch.zeros((batch, 1), dtype=torch.bool, device=device)
            spatial_valid = torch.zeros((batch, SPATIAL_TOKEN_COUNT), dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([self_valid, spatial_valid, ~object_mask], dim=1)

        return tokens, key_padding_mask


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
    """Token transformer readout for the native token contract."""

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        readout: str = "self",  # accepted for compatibility, ignored (always "self")
        action_history_tokens: int = 0,
    ) -> None:
        super().__init__()
        del obs_dim, readout
        self.output_dim = int(d_model)
        self.tokenizer = TokenObservationTokenizer(d_model=d_model, action_history_tokens=action_history_tokens)
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
        # v8: self token is position 0 (no CLS token).
        return self.final_ln(transformed[:, 0, :])
