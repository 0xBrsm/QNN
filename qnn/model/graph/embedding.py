"""GraphObsEmbedding — obs embedding assembled from a GraphSpec token dict.

Replaces the per-layout ``ObsEmbedding`` subclass zoo (monolithic-self,
``SplitSelfObsEmbedding``, ``HeldWeaponSplitObsEmbedding``): the self
block is whatever the spec's ``cls``/``fields`` tokens declare, each
fields token a :class:`TokenBuilder` over the shared field catalog. The
entity / spatial / shared-embedding plumbing is inherited from
``ObsEmbedding`` unchanged.

Field order inside a token is fixed (scalars → kind_tag → vocab →
readiness → ammo_pools → vocab_sum) so the parameter layout is
deterministic for a given spec. State-dict names are keyed by token name
(``self_builders.<token>.projs.<i>...``), so renaming a token in the
spec renames its parameters — legacy checkpoints map in through the
loader migrations, not by accident of naming.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qnn.model.graph.spec import (
    GraphSpec, TokenSpec, TOKEN_KIND_CLS, TOKEN_KIND_FIELDS,
    TOKEN_KIND_SPATIAL,
)
from qnn.model.tokens.obs_accessor import ObsAccessor
from qnn.model.tokens.obs_fields import (
    KIND_SELF, AmmoPools, KindTag, ScalarGroup, VocabEmbed, VocabSum,
    WeaponReadiness,
)
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.transformer import ObsEmbedding


def token_fields(token: TokenSpec) -> tuple:
    """FieldSource tuple for one ``fields`` token, in the canonical order."""
    fields: list[object] = []
    if token.scalars:
        fields.append(ScalarGroup(token.scalars))
    if token.kind_tag:
        fields.append(KindTag(KIND_SELF))
    fields.extend(VocabEmbed(n) for n in token.vocab)
    if token.readiness:
        fields.append(WeaponReadiness())
    if token.ammo_pools:
        fields.append(AmmoPools())
    fields.extend(VocabSum(n) for n in token.vocab_sum)
    return tuple(fields)


class GraphObsEmbedding(ObsEmbedding):
    """N-self-token obs embedding declared by a GraphSpec."""

    def __init__(self, spec: GraphSpec) -> None:
        self_specs = spec.self_tokens
        # Plain attrs are safe before nn.Module.__init__ (no params/modules
        # yet); the base __init__ reads _N_SELF_TOKENS and calls
        # _init_self_components, which needs the token specs.
        self._self_token_specs = self_specs
        self._N_SELF_TOKENS = len(self_specs)
        # Ordered self-token names: index i ↔ encoder ``self_block[:, i]`` (CLS at
        # 0). Lets a head read one token as its readout via a ``token.<name>`` edge.
        self.self_token_names = tuple(t.name for t in self_specs)
        spatial = next(
            (t for t in spec.tokens if t.kind == TOKEN_KIND_SPATIAL), None
        )
        super().__init__(
            d_model=spec.encoder.d_model,
            include_spatial=spec.has_spatial,
            spatial_source=spatial.source if spatial is not None else "ego",
            spatial_k=spatial.k if spatial is not None else 0,
            probe_bands=spatial.probe_bands if spatial is not None else (),
            entity_stream=spec.entity_stream,
        )

    def _init_self_components(self) -> None:
        builders: dict[str, TokenBuilder] = {}
        for token in self._self_token_specs:
            if token.kind == TOKEN_KIND_CLS:
                self.cls_embed = nn.Parameter(torch.zeros(self.d_model))
                continue
            builders[token.name] = TokenBuilder(
                self.d_model,
                token_fields(token),
                entity_embed=self.entity_embed,
                movement_embed=self.movement_embed,
                kind_embed=self.kind_embed if token.kind_tag else None,
            )
        self.self_builders = nn.ModuleDict(builders)

    def _build_self_block(
        self,
        obs_dict: dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
        vocab_max: int,
    ) -> torch.Tensor:
        del device, vocab_max
        acc = ObsAccessor(obs_dict)
        parts: list[torch.Tensor] = []
        for token in self._self_token_specs:
            if token.kind == TOKEN_KIND_CLS:
                parts.append(self.cls_embed.view(1, 1, -1).expand(batch, 1, -1))
            else:
                parts.append(self.self_builders[token.name](acc).unsqueeze(1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
