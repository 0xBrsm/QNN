"""Transformer trunk for the native token observation contract.

Token attention reads self, object, event, and spatial groups. A rolling action
history is concatenated after the [CLS] readout before the final projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quake_ai.model.observation import (
    ACTION_HISTORY_DIM,
    ACTION_HISTORY_LEN,
    EVENT_ID_DIM,
    EVENT_SCALAR_DIM,
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
_ACTION_HISTORY_FEATURES = ACTION_HISTORY_LEN * ACTION_HISTORY_DIM


class TokenObservationTokenizer(nn.Module):
    """Convert packed token observations into d_model transformer tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)

        self.self_proj = nn.Linear(SELF_SCALAR_DIM, d_model)
        self.object_proj = nn.Linear(OBJECT_SCALAR_DIM, d_model)
        self.event_proj = nn.Linear(EVENT_SCALAR_DIM, d_model)
        self.spatial_proj = nn.Linear(SPATIAL_SCALAR_DIM, d_model)

        self.kind_embed = nn.Embedding(3, d_model)
        self.weapon_embed = nn.Embedding(9, d_model)
        self.subject_embed = nn.Embedding(max(SUBJECT_IDS.values()) + 1, d_model)
        self.action_embed = nn.Embedding(max(ACTION_IDS.values()) + 1, d_model)
        self.qualifier_embed = nn.Embedding(max(QUALIFIER_IDS.values()) + 1, d_model)
        self.modality_embed = nn.Embedding(max(MODALITY_IDS.values()) + 1, d_model)
        self.player_embed = nn.Embedding(MAX_PLAYER_SLOTS + 1, d_model)
        self.spatial_sector_embed = nn.Embedding(max(SPATIAL_SECTOR_IDS.values()) + 1, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.n_tokens = 1 + 1 + MAX_OBJECT_TOKENS + SPATIAL_TOKEN_COUNT

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        self_scalars = obs_dict["self_scalars"]
        self_weapon_id = obs_dict["self_weapon_id"].long()
        object_ids = obs_dict["object_ids"].long()
        object_scalars = obs_dict["object_scalars"]
        object_mask = obs_dict["object_mask"].bool()
        event_ids = obs_dict["event_ids"].long()
        event_scalars = obs_dict["event_scalars"]
        event_owner = obs_dict["event_owner"].long()
        event_mask = obs_dict["event_mask"].bool()
        spatial_ids = obs_dict["spatial_ids"].long()
        spatial_scalars = obs_dict["spatial_scalars"]

        batch = int(self_scalars.shape[0])
        device = self_scalars.device

        self_token = (
            self.self_proj(self_scalars).unsqueeze(1)
            + self.kind_embed(torch.full((batch, 1), _TOKEN_KIND_SELF, dtype=torch.long, device=device))
            + self.weapon_embed(self_weapon_id.clamp(min=0, max=self.weapon_embed.num_embeddings - 1)).unsqueeze(1)
        )

        object_token = self.object_proj(object_scalars)
        object_token = object_token + self.kind_embed(
            torch.full((batch, MAX_OBJECT_TOKENS), _TOKEN_KIND_OBJECT, dtype=torch.long, device=device)
        )
        object_token = object_token + self.subject_embed(object_ids[:, :, 0].clamp(min=0, max=self.subject_embed.num_embeddings - 1))
        object_token = object_token + self.qualifier_embed(object_ids[:, :, 1].clamp(min=0, max=self.qualifier_embed.num_embeddings - 1))
        object_token = object_token + self.modality_embed(object_ids[:, :, 2].clamp(min=0, max=self.modality_embed.num_embeddings - 1))
        object_token = object_token + self.player_embed(object_ids[:, :, 3].clamp(min=0, max=self.player_embed.num_embeddings - 1))

        event_token = self.event_proj(event_scalars)
        event_token = event_token + self.subject_embed(event_ids[:, :, 0].clamp(min=0, max=self.subject_embed.num_embeddings - 1))
        event_token = event_token + self.action_embed(event_ids[:, :, 1].clamp(min=0, max=self.action_embed.num_embeddings - 1))
        event_token = event_token + self.qualifier_embed(event_ids[:, :, 2].clamp(min=0, max=self.qualifier_embed.num_embeddings - 1))
        event_token = event_token + self.modality_embed(event_ids[:, :, 3].clamp(min=0, max=self.modality_embed.num_embeddings - 1))
        event_token = event_token * event_mask.unsqueeze(-1).to(dtype=event_token.dtype)

        object_event_trace = torch.zeros((batch, MAX_OBJECT_TOKENS, self.d_model), dtype=event_token.dtype, device=device)
        owner_index = event_owner.clamp(min=0, max=MAX_OBJECT_TOKENS - 1).unsqueeze(-1).expand(-1, -1, self.d_model)
        object_event_trace.scatter_add_(1, owner_index, event_token)
        object_token = object_token + object_event_trace

        spatial_token = self.spatial_proj(spatial_scalars)
        spatial_token = spatial_token + self.kind_embed(
            torch.full((batch, SPATIAL_TOKEN_COUNT), _TOKEN_KIND_SPATIAL, dtype=torch.long, device=device)
        )
        spatial_token = spatial_token + self.spatial_sector_embed(
            spatial_ids.clamp(min=0, max=self.spatial_sector_embed.num_embeddings - 1)
        )

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, self_token, object_token, spatial_token], dim=1)

        prefix = torch.zeros((batch, 2), dtype=torch.bool, device=device)
        spatial_valid = torch.ones((batch, SPATIAL_TOKEN_COUNT), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([prefix, ~object_mask, ~spatial_valid], dim=1)
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
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
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
        output_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del obs_dim
        self.output_dim = int(output_dim)
        self.tokenizer = TokenObservationTokenizer(d_model=d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model + _ACTION_HISTORY_FEATURES, output_dim)
        self.output_act = nn.Tanh()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        action_history = obs_dict["action_history"]
        tokens, key_padding_mask = self.tokenizer(obs_dict)
        transformed = tokens
        for block in self.blocks:
            transformed = block(transformed, key_padding_mask=key_padding_mask)
        cls_output = self.final_ln(transformed[:, 0, :])
        history_features = action_history.reshape(action_history.shape[0], _ACTION_HISTORY_FEATURES).to(dtype=cls_output.dtype)
        return self.output_act(self.output_proj(torch.cat([cls_output, history_features], dim=1)))
