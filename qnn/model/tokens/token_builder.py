"""TokenBuilder — compose obs fields into one d_model token.

Given a declarative ``fields`` list (``obs_fields.FieldSource`` primitives)
and the shared embedding tables, ``TokenBuilder`` produces a ``(B*, d_model)``
token from an :class:`ObsAccessor`. It owns one ``Linear`` per
``ScalarGroup``; embedding tables are *tied* (passed in from the encoder, not
re-declared) so equipped-weapon/entity-token rockets share one vocab.

This single module expresses every bench self-token and attack/look bundle
that used to be a bespoke ``_build_self_block`` override or ``_*_token``
method: the canonical monolithic self token, the split-self subtokens, the
weapon-head token, attack-finished-only, and the engaged/weapon/motion
bundles. A new ablation is a new ``fields`` tuple — no new module.

Dtype: scalar groups go through ``Linear`` (autocast may emit bf16); embedding
lookups stay fp32. All contributions are cast to a single reference dtype
before summing — the ``dtype`` arg when given (heads pass ``features.dtype``),
else the scalar-projection output dtype, else fp32. This centralizes the
``.to(features.dtype)`` casts the old per-probe code scattered everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import (
    NUM_AMMO_POOLS, VOCAB_FIELDS, WEAPON_SUBJECT_IDS,
    AmmoPools, ScalarGroup, VocabEmbed, VocabSum, WeaponReadiness, KindTag,
)


class TokenBuilder(nn.Module):
    def __init__(
        self,
        d_model: int,
        fields,
        *,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
        kind_embed: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.fields = tuple(fields)
        # Tied tables are owned by ObsEmbedding and must keep their original
        # state_dict names. Store refs without registering duplicate modules.
        object.__setattr__(self, "entity_embed", entity_embed)
        object.__setattr__(self, "movement_embed", movement_embed)
        object.__setattr__(self, "kind_embed", kind_embed)

        # One Linear per ScalarGroup, keyed by its position in ``fields`` so
        # the parameter layout is deterministic w.r.t. the spec order.
        self.projs = nn.ModuleDict()
        needs_readiness = False
        n_ammo_pools = 0
        for i, src in enumerate(self.fields):
            if isinstance(src, ScalarGroup):
                self.projs[str(i)] = nn.Linear(src.width, self.d_model)
            elif isinstance(src, WeaponReadiness):
                needs_readiness = True
            elif isinstance(src, AmmoPools):
                n_ammo_pools += 1
            elif isinstance(src, KindTag) and self.kind_embed is None:
                raise ValueError("KindTag field needs a kind_embed table")

        if needs_readiness:
            self.register_buffer(
                "_weapon_subject_ids",
                torch.tensor(WEAPON_SUBJECT_IDS, dtype=torch.long),
                persistent=False,
            )
        # Ammo-type embeds are a NEW owned table (ammo types are not entities,
        # so nothing to tie). One table per token; at most one AmmoPools field.
        if n_ammo_pools > 1:
            raise ValueError("a token may declare at most one AmmoPools field")
        self.ammo_embed = (
            nn.Embedding(NUM_AMMO_POOLS, self.d_model) if n_ammo_pools else None
        )

    def _table(self, selector: str) -> nn.Embedding:
        return self.entity_embed if selector == "entity" else self.movement_embed

    def _vocab_embed(self, acc: ObsAccessor, name: str) -> torch.Tensor:
        spec = VOCAB_FIELDS[name]
        emb = self._table(spec.table)
        ids = acc.vocab_ids(name).clamp(0, emb.num_embeddings - 1)   # (B*,)
        out = emb(ids)
        if spec.masked:
            out = out * (ids > 0).to(out.dtype).unsqueeze(-1)
        return out

    def _vocab_sum(self, acc: ObsAccessor, name: str) -> torch.Tensor:
        spec = VOCAB_FIELDS[name]
        emb = self._table(spec.table)
        ids = acc.vocab_ids(name).clamp(0, emb.num_embeddings - 1)   # (B*, P)
        mask = (ids > 0).unsqueeze(-1)
        out = emb(ids)
        return (out * mask.to(out.dtype)).sum(dim=1)

    def _weapon_readiness(self, acc: ObsAccessor) -> torch.Tensor:
        weapon_embeds = self.entity_embed(self._weapon_subject_ids)  # (8, d_model)
        readiness = acc.readiness().to(weapon_embeds.dtype)          # (B*, 8)
        return torch.einsum("bw,wd->bd", readiness, weapon_embeds)

    def _ammo_pools(self, acc: ObsAccessor) -> torch.Tensor:
        ammo_embeds = self.ammo_embed.weight                        # (4, d_model)
        pools = acc.ammo_pools().to(ammo_embeds.dtype)              # (B*, 4)
        return torch.einsum("bp,pd->bd", pools, ammo_embeds)

    def _kind_tag(self, acc: ObsAccessor, kind: int) -> torch.Tensor:
        # The tag is constant for the whole token. Broadcasting the table row
        # avoids allocating an index tensor and launching an embedding lookup.
        # Backward sums over the expanded dim instead of scatter-adding B rows
        # — same gradient, different accumulation order, so training runs are
        # not trajectory-comparable across this change (adopted 2026-07-11).
        return self.kind_embed.weight[kind].unsqueeze(0).expand(acc.batch, -1)

    def forward(self, acc: ObsAccessor, dtype: torch.dtype | None = None) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i, src in enumerate(self.fields):
            if isinstance(src, ScalarGroup):
                cols = [acc.scalar(n) for n in src.names]
                x = cols[0] if len(cols) == 1 else torch.cat(cols, dim=-1)
                parts.append(self.projs[str(i)](x))
            elif isinstance(src, VocabEmbed):
                parts.append(self._vocab_embed(acc, src.name))
            elif isinstance(src, VocabSum):
                parts.append(self._vocab_sum(acc, src.name))
            elif isinstance(src, WeaponReadiness):
                parts.append(self._weapon_readiness(acc))
            elif isinstance(src, AmmoPools):
                parts.append(self._ammo_pools(acc))
            elif isinstance(src, KindTag):
                parts.append(self._kind_tag(acc, src.kind))
            else:  # pragma: no cover - guards spec typos
                raise TypeError(f"unknown FieldSource {type(src).__name__}")

        # Keep the reduction as one stack+sum kernel. The production
        # ObsEmbedding self token used this shape to avoid ROCm dispatch-queue
        # churn, and the same reducer is fine for bench token bundles.
        if dtype is not None:
            parts = [p.to(dtype) for p in parts]
        if len(parts) == 1:
            return parts[0]
        return torch.stack(parts, dim=0).sum(dim=0)
