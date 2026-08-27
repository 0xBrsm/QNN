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
    ENTITY_VOCAB_SIZE, ACTION_VOCAB_SIZE, COMBAT_MODALITY_VOCAB_SIZE,
    MODALITY_VOCAB_SIZE,
    MAX_PLAYER_INDICES, MAX_TOKEN_OBJECTS,
    PROJECTILE_SCALAR_DIM, ACTOR_SCALAR_DIM,
    FULL_PROJECTILE_SCALAR_DIM, FULL_ACTOR_SCALAR_DIM,
    ITEM_SCALAR_DIM, MOVER_SCALAR_DIM,
    ENTITY_STREAMS, ENTITY_STREAM_COMBAT, ENTITY_STREAM_FULL,
)
from qnn.schema import (
    SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM, PROBE_OFFSET_DIM,
    PROBE_SPATIAL_SOURCES, SPATIAL_SOURCES, SPATIAL_SOURCE_EGO,
    SPATIAL_SOURCE_POOLED9, SPATIAL_SOURCE_PROBE_GRID_NF,
)
from qnn.model.spatial_pool import SectorPool9

# Token-kind embedding rows. ObsEmbedding tags each output token with
# one of these so the attention can distinguish self / entity / spatial
# rows without relying on positional encoding alone.
_TOKEN_KIND_SELF    = 0
_TOKEN_KIND_ENTITY  = 1
_TOKEN_KIND_SPATIAL = 2

# Near-field floor ring (probe_grid_nf source, rev-11): the agent's own
# steep downward atlas bands, emitted as extra egocentric spatial tokens
# beside the k probe tokens. Bands {-75,-60,-45} are elevation-row indices
# 0,1,2 in the atlas (SPATIAL_FIELDS order); yaw is sub-sampled to 12 cells
# (every other one of 24), the offline-priced knee where floor MAE holds
# near the full-atlas reference at ~18 wire bytes.
_NF_FLOOR_BANDS = (0, 1, 2)
_NF_YAW_STRIDE  = 2

# Canonical self-block layout:
#   slot 0                  monolithic self token (carries all self info)
#   slot 1..11              spatial band tokens (when include_spatial=True;
#                           9 sector tokens for the pooled9 source)
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
        [self, [spatial_0..10,] entity_0..N-1]
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
        include_spatial: bool = True,
        spatial_source: str = SPATIAL_SOURCE_EGO,
        spatial_k: int = 0,
        probe_bands: tuple[int, ...] = (),
        entity_stream: str = ENTITY_STREAM_COMBAT,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # Entity stream: "combat" (default) is the A27 actor/projectile
        # stream; "full" rebuilds the a26-line entity wiring (recency dims,
        # live item/mover projections, 4-way modality vocab, actor-only
        # attention mask) so a26 checkpoints load and forward bit-faithfully.
        self.entity_stream = str(entity_stream)
        if self.entity_stream not in ENTITY_STREAMS:
            raise ValueError(
                f"unknown entity_stream {self.entity_stream!r}; "
                f"expected one of {ENTITY_STREAMS}"
            )
        # out_dim — width of every token vector this embedding emits.
        self.out_dim = self.d_model
        self.include_spatial = bool(include_spatial)
        self.spatial_source = str(spatial_source)
        if self.spatial_source not in SPATIAL_SOURCES:
            raise ValueError(
                f"unknown spatial_source {self.spatial_source!r}; "
                f"expected one of {SPATIAL_SOURCES}"
            )
        self.spatial_k = int(spatial_k)
        # Probe-source band prune: () keeps all 11 atlas bands; a tuple keeps
        # only those band rows as probe tokens (the rest are supplied by the
        # ego ring or dropped). Applied before band-major fusion.
        self._probe_bands = tuple(int(b) for b in probe_bands) or None
        self._n_probe_tokens = (
            len(self._probe_bands) if self._probe_bands else SPATIAL_TOKEN_COUNT
        )

        # Native-width → float adapters. Each is a no-op pass-through
        # when the obs dict already carries the dequanted floats, so
        # the dataloader format choice is transparent to the model.
        from qnn.model.dequant import (
            SelfDequantizer, SpatialDequantizer, EntityDequantizer,
        )
        self.self_dequant    = SelfDequantizer()
        self.spatial_dequant = SpatialDequantizer()
        self.entity_dequant  = EntityDequantizer(entity_stream=self.entity_stream)

        # Per-type entity projections. The full (a26) stream widens
        # projectile/actor by the trailing recency scalar and adds the
        # item/mover projections the combat stream deleted.
        if self.entity_stream == ENTITY_STREAM_FULL:
            self.proj_projectile = nn.Linear(FULL_PROJECTILE_SCALAR_DIM, self.d_model)
            self.proj_actor      = nn.Linear(FULL_ACTOR_SCALAR_DIM,      self.d_model)
            self.proj_item       = nn.Linear(ITEM_SCALAR_DIM,            self.d_model)
            self.proj_mover      = nn.Linear(MOVER_SCALAR_DIM,           self.d_model)
        else:
            self.proj_projectile = nn.Linear(PROJECTILE_SCALAR_DIM, self.d_model)
            self.proj_actor      = nn.Linear(ACTOR_SCALAR_DIM,      self.d_model)

        # Ego source projects the row's own band panorama; probe_grid
        # projects the k nearest map probes' same-band panoramas plus
        # their relative-pose encodings (rev-10 probe-grid direction);
        # pooled9 reduces the ego atlas to the v1 9-sector depth summary
        # (capacity-class bench arm) before projecting. Ego/probe_grid
        # emit 11 band tokens; pooled9 emits 9 sector tokens.
        if self.spatial_source in PROBE_SPATIAL_SOURCES:
            self.spatial_proj = nn.Linear(
                self.spatial_k * (SPATIAL_SCALAR_DIM + PROBE_OFFSET_DIM),
                self.d_model,
            )
        elif self.spatial_source == SPATIAL_SOURCE_POOLED9:
            self.sector_pool = SectorPool9()
            self.spatial_proj = nn.Linear(SectorPool9.OUT_DIM, self.d_model)
        else:
            self.spatial_proj = nn.Linear(SPATIAL_SCALAR_DIM, self.d_model)
        # probe_grid_nf adds one token per near-field floor band; each
        # projects that band's sub-sampled [depth, hit] ring (12 yaw cells
        # → 24 scalars) into d_model, beside the k probe band tokens.
        self._nf_bands = (
            len(_NF_FLOOR_BANDS)
            if self.include_spatial
            and self.spatial_source == SPATIAL_SOURCE_PROBE_GRID_NF
            else 0
        )
        if self._nf_bands:
            half = SPATIAL_SCALAR_DIM // 2               # 24 yaw depth + 24 hit
            self._nf_yaws = len(range(0, half, _NF_YAW_STRIDE))
            self.nf_proj = nn.Linear(2 * self._nf_yaws, self.d_model)
            # Ring gather indices are compile-time constants — register them
            # once instead of rebuilding them on every forward.
            self.register_buffer(
                "_nf_yaw_idx", torch.arange(0, half, _NF_YAW_STRIDE),
                persistent=False,
            )
            self.register_buffer(
                "_nf_band_idx", torch.tensor(_NF_FLOOR_BANDS), persistent=False,
            )
        self._spatial_count = (
            0 if not self.include_spatial
            else SectorPool9.N_SECTORS
            if self.spatial_source == SPATIAL_SOURCE_POOLED9
            else self._n_probe_tokens + self._nf_bands
        )
        if self._probe_bands is not None:
            self.register_buffer(
                "_probe_band_idx", torch.tensor(self._probe_bands), persistent=False,
            )

        # Shared embeddings.
        self.entity_embed   = nn.Embedding(ENTITY_VOCAB_SIZE,    self.d_model)
        self.action_embed   = nn.Embedding(ACTION_VOCAB_SIZE,    self.d_model)
        # Combat exposes SIGHT/PROXIMITY only; the full stream keeps the
        # 4-way engine vocab (SOUND/MEMORY rows are trained a26 weights).
        self.modality_embed = nn.Embedding(
            MODALITY_VOCAB_SIZE
            if self.entity_stream == ENTITY_STREAM_FULL
            else COMBAT_MODALITY_VOCAB_SIZE,
            self.d_model,
        )
        self.player_embed   = nn.Embedding(MAX_PLAYER_INDICES + 1, self.d_model)
        # Fixed row order is a wire convention, not something a transformer
        # without positional encoding can observe. Give each atlas band
        # (or pooled9 sector) an explicit learned identity.
        self.band_embed = (
            nn.Embedding(self._spatial_count, self.d_model)
            if self.include_spatial else None
        )
        # Self / entity / spatial token-kind tags.
        self.kind_embed     = nn.Embedding(3, self.d_model)
        self.movement_embed = nn.Embedding(5, self.d_model)

        # Self-block submodules — subclass hook so split-self can declare
        # its own projections without inheriting the monolithic one.
        self._init_self_components()

        # Token-layout invariants. Computed from _N_SELF_TOKENS so
        # subclasses inherit the right slices for free.
        spatial_count = self._spatial_count
        self.n_tokens = self._N_SELF_TOKENS + spatial_count + MAX_TOKEN_OBJECTS
        self.self_slice = slice(0, self._N_SELF_TOKENS)
        entity_start = self._N_SELF_TOKENS + spatial_count
        self.entity_slice = slice(entity_start, entity_start + MAX_TOKEN_OBJECTS)

    def _init_self_components(self) -> None:
        """Declare self-block submodules. Override to change the self design."""
        self.self_token_builder = TokenBuilder(
            self.d_model,
            canonical_self_fields(),
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
        if self.entity_stream == ENTITY_STREAM_FULL:
            proj_map = {
                TOKEN_PROJECTILE: (self.proj_projectile, FULL_PROJECTILE_SCALAR_DIM),
                TOKEN_ACTOR: (self.proj_actor, FULL_ACTOR_SCALAR_DIM),
                TOKEN_ITEM: (self.proj_item, ITEM_SCALAR_DIM),
                TOKEN_MOVER: (self.proj_mover, MOVER_SCALAR_DIM),
            }
        else:
            proj_map = {
                TOKEN_PROJECTILE: (self.proj_projectile, PROJECTILE_SCALAR_DIM),
                TOKEN_ACTOR: (self.proj_actor, ACTOR_SCALAR_DIM),
            }
        out_dtype = entity_scalars_raw.dtype

        if self.training:
            # One fused GEMM over the per-type projections, then a per-slot
            # gather on the type tag. The weights stay the checkpoint
            # Linears — they are zero-padded to the max scalar width and
            # concatenated per forward (tiny tensors; autograd splits the
            # grads back). Active TOKEN_* values are contiguous from 0, so
            # the tag doubles as the gather index; invalid/empty slots are
            # zeroed. Eval/export keep the per-type form so
            # the traced inference graph is unchanged.
            max_dim = int(entity_scalars_raw.shape[-1])
            weights = torch.cat([
                nn.functional.pad(proj.weight, (0, max_dim - proj.weight.shape[1]))
                for proj, _ in proj_map.values()
            ])                                                    # (n·d, max_dim)
            bias = torch.cat([proj.bias for proj, _ in proj_map.values()])
            fused = nn.functional.linear(entity_scalars_raw, weights, bias)
            fused = fused.view(*entity_types.shape, len(proj_map), self.d_model)
            if self.entity_stream == ENTITY_STREAM_FULL:
                # a26: all four TOKEN_* tags are live; empty slots (-1)
                # clamp to 0 and are zeroed by the mask.
                valid = entity_types >= 0
                idx = entity_types.clamp(min=0).view(*entity_types.shape, 1, 1)
            else:
                valid = ((entity_types == TOKEN_PROJECTILE) | (entity_types == TOKEN_ACTOR))
                idx = entity_types.clamp(min=0, max=TOKEN_ACTOR).view(*entity_types.shape, 1, 1)
            picked = fused.gather(2, idx.expand(*entity_types.shape, 1, self.d_model))
            picked = picked.squeeze(2) * valid.unsqueeze(-1).to(fused.dtype)
            return picked.to(out_dtype)

        # Eval/export: project every slot through every type's Linear and
        # select by type mask. Branch-free and sync-free (boolean-mask
        # gather/scatter would call nonzero(), a blocking device→host copy);
        # the dense GEMMs over 16 slots are microseconds.
        result: torch.Tensor | None = None
        for tok_type, (proj, sdim) in proj_map.items():
            mask = (entity_types == tok_type).unsqueeze(-1)  # (batch, 16, 1)
            out = proj(entity_scalars_raw[..., :sdim]).to(out_dtype)
            typed = out * mask.to(out_dtype)
            result = typed if result is None else result + typed
        assert result is not None
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
            self_armor_type_id, self_movement_id: (batch, 1)
            self_powerup_ids: (batch, 5) — QUAD/PENT/RING/SUIT/MEGAHEALTH
                              (composed from the split fields if absent)
            entity_types: (batch, N) — token type tags, -1 for empty
            entity_scalars_proj: (batch, N, d_model)
              OR entity_scalars_raw: (batch, N, MAX_ENTITY_SCALAR_DIM)
            entity_ids: (batch, N, max_ids) — [subject, modality, player_id]
            entity_event_actions / _sources / _counts
            spatial_scalars: (batch, 11, 48)
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
        # Attention validity: combat attends actor + projectile tokens; the
        # full (a26) stream attended ONLY actor tokens (parity-pinned — the
        # projectile/item/mover rows were computed but key-padded out).
        if self.entity_stream == ENTITY_STREAM_FULL:
            entity_valid = entity_mask
        else:
            entity_valid = entity_mask | (entity_types == TOKEN_PROJECTILE)

        entity_subject = obs_dict["entity_ids"][:, :, 0].long().clamp(0, vocab_max)
        entity_repr = entity_repr + self.entity_embed(entity_subject)

        entity_modality = obs_dict["entity_ids"][:, :, 1].long()
        if self.entity_stream == ENTITY_STREAM_FULL:
            # a26 clamped the modality id into its 4-row table.
            entity_modality = entity_modality.clamp(0, self.modality_embed.num_embeddings - 1)
        entity_repr = entity_repr + self.modality_embed(entity_modality)

        if obs_dict["entity_ids"].shape[2] >= 3:
            entity_player = obs_dict["entity_ids"][:, :, 2].long().clamp(0, self.player_embed.num_embeddings - 1)
            player_mask = (entity_types == TOKEN_ACTOR).unsqueeze(-1)
            entity_repr = entity_repr + self.player_embed(entity_player) * player_mask.float()

        evt_actions = obs_dict["entity_event_actions"].long().clamp(0, self.action_embed.num_embeddings - 1)
        evt_sources = obs_dict["entity_event_sources"].long().clamp(0, vocab_max)
        evt_counts  = obs_dict["entity_event_counts"].long()
        evt_embed = self.action_embed(evt_actions) + self.entity_embed(evt_sources)
        # Resident corpora compact these tensors to their corpus-wide used
        # width (often one slot rather than the four-slot wire maximum).
        # Inference retains the full wire width.
        n_event_slots = evt_actions.shape[-1]
        evt_range = torch.arange(n_event_slots, device=device).view(1, 1, n_event_slots)
        evt_mask = (evt_range < evt_counts.unsqueeze(-1)).unsqueeze(-1).float()
        entity_repr = entity_repr + (evt_embed * evt_mask).sum(dim=2)

        # Kind tag on entity tokens — constant for the whole block, so add the
        # table row broadcast instead of materializing an index tensor and an
        # embedding lookup (whose backward is an atomic scatter).
        entity_repr = entity_repr + self.kind_embed.weight[_TOKEN_KIND_ENTITY]

        if self.include_spatial:
            if self.spatial_source in PROBE_SPATIAL_SOURCES:
                # (B, K, 11, 144) probe panoramas, already rolled into
                # the agent's view frame loader-side; band-major fusion:
                # each band token sees all K probes' rows plus each
                # probe's relative-pose encoding.
                probe = obs_dict["probe_scalars"]
                if self._probe_bands is not None:
                    probe = probe.index_select(2, self._probe_band_idx)
                b, k, bands, dim = probe.shape
                per_band = probe.permute(0, 2, 1, 3).reshape(b, bands, k * dim)
                offsets = obs_dict["probe_offsets"].to(per_band.dtype)
                offsets = offsets.reshape(b, 1, k * PROBE_OFFSET_DIM)
                spatial_in = torch.cat(
                    [per_band, offsets.expand(b, bands, -1)], dim=-1,
                )
                spatial_token = self.spatial_proj(spatial_in)
                if self._nf_bands:
                    # Egocentric near-field ring from the agent's own atlas
                    # (dequant produced spatial_scalars above): steep floor
                    # bands, sub-sampled in yaw, [depth, hit] concatenated.
                    ss = obs_dict["spatial_scalars"].to(spatial_token.dtype)
                    half = ss.shape[-1] // 2
                    depth = ss[..., :half]
                    hit = ss[..., half:]
                    bandsel, yaw = self._nf_band_idx, self._nf_yaw_idx
                    ring = torch.cat(
                        [depth[:, bandsel][..., yaw], hit[:, bandsel][..., yaw]],
                        dim=-1,
                    )                                    # (B, nf_bands, 2*nf_yaws)
                    ring_token = self.nf_proj(ring)      # (B, nf_bands, d_model)
                    spatial_token = torch.cat(
                        [spatial_token, ring_token], dim=1,
                    )
            elif self.spatial_source == SPATIAL_SOURCE_POOLED9:
                spatial_token = self.spatial_proj(
                    self.sector_pool(obs_dict["spatial_scalars"])
                )
            else:                                    # SPATIAL_SOURCE_EGO
                spatial_token = self.spatial_proj(obs_dict["spatial_scalars"])
            assert self.band_embed is not None
            # One row per spatial token, in order — that is the whole table,
            # so add it broadcast instead of gathering through arange (whose
            # backward is an atomic scatter). Same form as kind_embed below.
            spatial_token = spatial_token + self.band_embed.weight.unsqueeze(0)
            spatial_token = spatial_token + self.kind_embed.weight[_TOKEN_KIND_SPATIAL]
            tokens = torch.cat([self_block, spatial_token, entity_repr], dim=1)
            non_entity_valid = torch.zeros(
                (batch, self._N_SELF_TOKENS + self._spatial_count),
                dtype=torch.bool, device=device,
            )
        else:
            tokens = torch.cat([self_block, entity_repr], dim=1)
            non_entity_valid = torch.zeros(
                (batch, self._N_SELF_TOKENS), dtype=torch.bool, device=device,
            )
        key_padding_mask = torch.cat([non_entity_valid, ~entity_valid], dim=1)

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
