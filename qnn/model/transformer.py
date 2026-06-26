"""Observation embedding + transformer encoder for the token observation contract.

Two slots cooperate to convert packed observations into per-token
representations the heads consume:

* **ObsEmbedding** — input-embedding layer. Projects raw
  obs scalars + discrete IDs into ``d_model`` token vectors with an
  explicit layout (CLS, self subtokens, [spatial,] entities) and emits
  an ``EncoderInput`` carrying the tokens, the key-padding mask, the
  ``self_slice`` / ``entity_slice`` slot ranges, and the actor mask.

* **Encoder** (``TransformerEncoder`` or any drop-in) — runs attention
  (or not) over those tokens and emits an ``EncoderOutput``:

    * ``self_readout`` — tokens[:, self_slice.start, :] (CLS by
      convention; the attention pools the self subtokens, spatial, and
      entities into it).
    * ``entity_outs`` — tokens[:, entity_slice, :].
    * ``entity_mask`` — actor-only validity mask over the N entity slots.

Splitting tokenization from attention lets ablations swap one half
without touching the other — e.g. different self-token layouts share
the same TransformerEncoder, or different attention/no-attention
encoders share the same ObsEmbedding.

Pointing into the entity set is the TargetPointer component's job, not
the encoder's. Pre-refactor versions of this class carried an internal
``self.target_pointer`` for the no-GRU code path; that's gone — Network
constructs its own TargetPointer slot unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import canonical_self_fields
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.vocab import (
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    ENTITY_VOCAB_SIZE, ACTION_VOCAB_SIZE, MODALITY_VOCAB_SIZE,
    MAX_PLAYER_INDICES, MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS,
    PROJECTILE_SCALAR_DIM, ACTOR_SCALAR_DIM, ITEM_SCALAR_DIM, MOVER_SCALAR_DIM,
)
from qnn.schema import (
    SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM,
)

# Token-kind embedding rows. ObsEmbedding tags each output token with
# one of these so the attention can distinguish self / entity / spatial
# rows without relying on positional encoding alone.
_TOKEN_KIND_SELF    = 0
_TOKEN_KIND_ENTITY  = 1
_TOKEN_KIND_SPATIAL = 2

# Canonical self-block layout:
#   slot 0                  monolithic self token (carries all self info)
#   slot 1..9               spatial tokens (when include_spatial=True)
#   slot 1+spatial..        entity tokens (MAX_TOKEN_OBJECTS)
# Subclasses (see ``ObsEmbedding._N_SELF_TOKENS``) can change the self-block
# width.


# ── ObsEmbedding ─────────────────────────────────────────────────


class ObsEmbedding(nn.Module):
    """Input-embedding layer: packed observations → d_model token vectors.

    Monolithic-self design: every self-related scalar bundle and ID
    embed sums into a single self token at slot 0. Per-type Linear
    projections on entity scalars + nn.Embedding lookups for discrete
    IDs (entity/action/modality/player/kind/movement), summed into
    per-slot d_model vectors. No attention — that lives in
    TransformerEncoder.

    Token layout:
        [self, [spatial_0..8,] entity_0..N-1]
        ↑                                    ↑
        self_slice.start (= 0)               entity_slice

    Subclasses can change the self-block shape by overriding
    ``_N_SELF_TOKENS`` + ``_init_self_components`` + ``_build_self_block``.
    The entity / spatial / embedding / mask plumbing is shared. See
    ``qnn.model.bench.inputs.split_self_obs_embedding`` for the
    self-splitting variant (CLS + state + arsenal + motion).
    """

    # Number of self-block tokens this obs embedding emits at the head of
    # the sequence. Subclasses override.
    _N_SELF_TOKENS: int = 1

    def __init__(
        self,
        d_model: int,
        *,
        self_weapon_embed_in_self: bool,
        include_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # out_dim — width of every token vector this embedding emits.
        self.out_dim = self.d_model
        self.self_weapon_embed_in_self = bool(self_weapon_embed_in_self)
        self.include_spatial = bool(include_spatial)

        # Native-width → float adapters. Each is a no-op pass-through
        # when the obs dict already carries the dequanted floats, so
        # the dataloader format choice is transparent to the model.
        from qnn.model.dequant import (
            SelfDequantizer, SpatialDequantizer, EntityDequantizer,
        )
        self.self_dequant    = SelfDequantizer()
        self.spatial_dequant = SpatialDequantizer()
        self.entity_dequant  = EntityDequantizer()

        # Per-type entity projections.
        self.proj_projectile = nn.Linear(PROJECTILE_SCALAR_DIM, self.d_model)
        self.proj_actor      = nn.Linear(ACTOR_SCALAR_DIM,      self.d_model)
        self.proj_item       = nn.Linear(ITEM_SCALAR_DIM,       self.d_model)
        self.proj_mover      = nn.Linear(MOVER_SCALAR_DIM,      self.d_model)

        self.spatial_proj = nn.Linear(SPATIAL_SCALAR_DIM, self.d_model)

        # Shared embeddings.
        self.entity_embed   = nn.Embedding(ENTITY_VOCAB_SIZE,    self.d_model)
        self.action_embed   = nn.Embedding(ACTION_VOCAB_SIZE,    self.d_model)
        self.modality_embed = nn.Embedding(MODALITY_VOCAB_SIZE,  self.d_model)
        self.player_embed   = nn.Embedding(MAX_PLAYER_INDICES + 1, self.d_model)
        # Self / entity / spatial token-kind tags.
        self.kind_embed     = nn.Embedding(3, self.d_model)
        self.movement_embed = nn.Embedding(5, self.d_model)

        # Self-block submodules — subclass hook so split-self can declare
        # its own projections without inheriting the monolithic one.
        self._init_self_components()

        # Token-layout invariants. Computed from _N_SELF_TOKENS so
        # subclasses inherit the right slices for free.
        spatial_count = SPATIAL_TOKEN_COUNT if self.include_spatial else 0
        self.n_tokens = self._N_SELF_TOKENS + spatial_count + MAX_TOKEN_OBJECTS
        self.self_slice = slice(0, self._N_SELF_TOKENS)
        entity_start = self._N_SELF_TOKENS + spatial_count
        self.entity_slice = slice(entity_start, entity_start + MAX_TOKEN_OBJECTS)

    def _init_self_components(self) -> None:
        """Declare self-block submodules. Override to change the self design."""
        self.self_token_builder = TokenBuilder(
            self.d_model,
            canonical_self_fields(self.self_weapon_embed_in_self),
            entity_embed=self.entity_embed,
            movement_embed=self.movement_embed,
            kind_embed=self.kind_embed,
        )

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
        batch, n_indices, _ = entity_scalars_raw.shape
        device = entity_scalars_raw.device
        result = torch.zeros(batch, n_indices, self.d_model, device=device, dtype=entity_scalars_raw.dtype)
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

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        """Build the monolithic self token. Shape (B, 1, d_model).

        The declarative TokenBuilder preserves the old stack+sum reduction
        shape for the monolithic token to avoid ROCm dispatch-queue churn.

        Subclasses override to emit a different (B, _N_SELF_TOKENS, D)
        block.
        """
        del batch, device, vocab_max
        return self.self_token_builder(ObsAccessor(obs_dict)).unsqueeze(1)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> "EncoderInput":
        """Build transformer input tokens from parsed observation dict.

        obs_dict must contain:
            self_scalars: (batch, SELF_SCALAR_DIM) — monolithic self bundle
            self_weapon_id, self_armor_type_id, self_movement_id: (batch, 1)
            self_powerup_ids: (batch, 5) — QUAD/PENT/RING/SUIT/MEGAHEALTH
                              (composed from the split fields if absent)
            entity_types: (batch, N) — token type tags, -1 for empty
            entity_scalars_proj: (batch, N, d_model)
              OR entity_scalars_raw: (batch, N, MAX_ENTITY_SCALAR_DIM)
            entity_ids: (batch, N, max_ids) — [subject, modality, player_id]
            entity_event_actions / _sources / _counts
            spatial_scalars: (batch, 9, 13)
        """
        obs_dict = self.entity_dequant(
            self.spatial_dequant(
                self.self_dequant(obs_dict)
            )
        )
        device = obs_dict["self_scalars"].device
        batch = obs_dict["self_scalars"].shape[0]
        vocab_max = self.entity_embed.num_embeddings - 1

        self_block = self._build_self_block(obs_dict, batch, device, vocab_max)

        # ---- Entity tokens ----
        entity_types = obs_dict["entity_types"].long()
        if "entity_scalars_proj" in obs_dict:
            entity_repr = obs_dict["entity_scalars_proj"]
        else:
            entity_repr = self._project_entity_scalars(
                entity_types, obs_dict["entity_scalars_raw"],
            )
        entity_mask = (entity_types == TOKEN_ACTOR)

        entity_subject = obs_dict["entity_ids"][:, :, 0].long().clamp(0, vocab_max)
        entity_repr = entity_repr + self.entity_embed(entity_subject)

        entity_modality = obs_dict["entity_ids"][:, :, 1].long().clamp(0, self.modality_embed.num_embeddings - 1)
        entity_repr = entity_repr + self.modality_embed(entity_modality)

        if obs_dict["entity_ids"].shape[2] >= 3:
            entity_player = obs_dict["entity_ids"][:, :, 2].long().clamp(0, self.player_embed.num_embeddings - 1)
            player_mask = (entity_types == TOKEN_ACTOR).unsqueeze(-1)
            entity_repr = entity_repr + self.player_embed(entity_player) * player_mask.float()

        evt_actions = obs_dict["entity_event_actions"].long().clamp(0, self.action_embed.num_embeddings - 1)
        evt_sources = obs_dict["entity_event_sources"].long().clamp(0, vocab_max)
        evt_counts  = obs_dict["entity_event_counts"].long()
        evt_embed = self.action_embed(evt_actions) + self.entity_embed(evt_sources)
        evt_range = torch.arange(MAX_ENTITY_EVENTS, device=device).view(1, 1, MAX_ENTITY_EVENTS)
        evt_mask = (evt_range < evt_counts.unsqueeze(-1)).unsqueeze(-1).float()
        entity_repr = entity_repr + (evt_embed * evt_mask).sum(dim=2)

        # Kind tag on entity tokens — second-dim is data-dependent under
        # the variable-length entity wire, so read it from the input.
        kind_entity = torch.full(
            (batch, entity_types.shape[1]), _TOKEN_KIND_ENTITY,
            dtype=torch.long, device=device,
        )
        entity_repr = entity_repr + self.kind_embed(kind_entity)

        if self.include_spatial:
            spatial_token = self.spatial_proj(obs_dict["spatial_scalars"])
            kind_spatial = torch.full(
                (batch, SPATIAL_TOKEN_COUNT), _TOKEN_KIND_SPATIAL,
                dtype=torch.long, device=device,
            )
            spatial_token = spatial_token + self.kind_embed(kind_spatial)
            tokens = torch.cat([self_block, spatial_token, entity_repr], dim=1)
            non_entity_valid = torch.zeros(
                (batch, self._N_SELF_TOKENS + SPATIAL_TOKEN_COUNT),
                dtype=torch.bool, device=device,
            )
        else:
            tokens = torch.cat([self_block, entity_repr], dim=1)
            non_entity_valid = torch.zeros(
                (batch, self._N_SELF_TOKENS), dtype=torch.bool, device=device,
            )
        key_padding_mask = torch.cat([non_entity_valid, ~entity_mask], dim=1)

        return EncoderInput(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            self_slice=self.self_slice,
            entity_slice=self.entity_slice,
            entity_mask=entity_mask,
        )


# ── Transformer ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
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


@dataclass(frozen=True, slots=True)
class EncoderInput:
    """ObsEmbedding → Encoder contract.

    Carries pre-built token vectors plus the layout the encoder needs to
    recover ``self_readout`` and ``entity_outs`` after (optionally)
    mixing the tokens with attention. ``self_slice.start`` is the
    readout position (CLS-by-convention); ``entity_slice`` covers
    exactly ``MAX_TOKEN_OBJECTS`` rows.
    """
    tokens: torch.Tensor             # (B*, T, D)
    key_padding_mask: torch.Tensor   # (B*, T) bool — True at padded positions
    self_slice: slice                # tokens[:, self_slice, :] is the self block
    entity_slice: slice              # tokens[:, entity_slice, :] is the entity block
    entity_mask: torch.Tensor        # (B*, N) bool — True at valid actor slots


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    self_readout: torch.Tensor   # (B*, D)
    entity_outs: torch.Tensor    # (B*, N, D)
    entity_mask: torch.Tensor    # (B*, N) bool — True at actor slots
    # Per-self-token outputs (CLS at index 0), in GraphSpec self-token order, so a
    # head can read one token as its readout via a ``token.<name>`` edge. Optional:
    # encoders/scaffolds that don't expose it leave it None (only token edges need it).
    self_block: torch.Tensor | None = None  # (B*, N_self, D)


class TransformerEncoder(nn.Module):
    """Self-attention encoder over pre-built tokens.

    Input: an ``EncoderInput`` produced by an obs-embedding slot (e.g.
    ``ObsEmbedding``). Output: ``EncoderOutput`` with the
    ``(self_readout, entity_outs, entity_mask)`` triple sliced from the
    attended token stream. ``self_readout`` comes from
    ``tokens[:, self_slice.start, :]`` — by convention this is the CLS
    slot that the attention pools the rest of the stream into.

    Downstream pointing is the caller's responsibility (typically
    ``TargetPointer`` as a separate component slot in ``Network``).
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ffn: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # out_dim — the width of self_readout / entity_outs this encoder emits.
        # Single dim contract every temporal/target/head slot sizes against.
        self.out_dim = self.d_model
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    d_ffn=int(d_ffn),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.final_ln = nn.LayerNorm(self.d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inp: EncoderInput) -> EncoderOutput:
        transformed = inp.tokens
        for block in self.blocks:
            transformed = block(transformed, key_padding_mask=inp.key_padding_mask)
        transformed = self.final_ln(transformed)
        return EncoderOutput(
            self_readout=transformed[:, inp.self_slice.start, :],
            entity_outs=transformed[:, inp.entity_slice, :],
            entity_mask=inp.entity_mask,
            self_block=transformed[:, inp.self_slice, :],
        )


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_encoder  # noqa: E402


@register_encoder("transformer")
def _build_encoder_transformer(encoder):
    return TransformerEncoder(
        d_model=encoder.d_model, n_heads=encoder.n_heads, n_layers=encoder.n_layers,
        d_ffn=encoder.d_ffn, dropout=encoder.attn_dropout)
