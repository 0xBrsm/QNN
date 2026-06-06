"""Weapon-head probe with trunk-style token projections, no transformer or GRU.

Mirror of ``qnn.bc.heads.fire_token`` but for the weapon classification
head (8-class CE on ``act.weapon``). The encoder is shared
(``TokenizedFeatureEncoder``) so we can run the same encoder ablations
(strip / concat / additive-only) against the weapon target.

The forward signature matches ``_CombatObjectiveNet.forward`` so the
canonical BC supervised loop drives it without modification.
``logits`` only carries ``weapon`` — other heads are skipped because
their keys aren't present, and the canonical CE loss path keys off
``head in logits``. ``target_logits`` / ``target_query`` are returned
as zero placeholders; the runner sets ``head_loss_weights["target"]``
to 0 so the target loss path runs but contributes zero gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from qnn.bc.heads.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.bc.heads.tokenized import TokenizedFeatureEncoder
from qnn.model.policy import ModelConfig, WEAPON_HEAD, WEAPON_HEAD_SIZE
from qnn.vocab import MAX_TOKEN_OBJECTS


class WeaponTokenHead(nn.Module):
    """Tokenized encoder + configurable MLP weapon head.

    Forward returns the same six-tuple ``_CombatObjectiveNet`` does;
    only ``weapon`` is populated in ``logits``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        self_weapon_embed_in_self: bool,
        strip_self_embeds: bool,
        weapon_embed_concat: bool,
        hidden: int,
        n_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = TokenizedFeatureEncoder(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon_embed_in_self,
            strip_self_embeds=strip_self_embeds,
            weapon_embed_concat=weapon_embed_concat,
        )
        self.d_model = self.encoder.d_model
        layers: list[nn.Module] = []
        prev = self.encoder.output_dim
        for _ in range(int(n_hidden_layers)):
            layers += [nn.Linear(prev, int(hidden)), nn.GELU(), nn.Dropout(float(dropout))]
            prev = int(hidden)
        layers.append(nn.Linear(prev, WEAPON_HEAD_SIZE))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        target_gt: torch.Tensor | None = None,
        target_dist_slot: torch.Tensor | None = None,
        prev_target_dist: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Memoryless probe — hidden / reset_mask / target_gt /
        # prev_target_dist are accepted to match the BC trainer's call
        # signature; ignored here.
        del hidden, reset_mask, target_gt, prev_target_dist

        if target_dist_slot is None:
            raise RuntimeError(
                "WeaponTokenHead requires target_dist_slot — the canonical "
                "BC supervised loop derives it from actions.target_dist."
            )

        sample = obs["self_scalars"]
        is_seq = sample.ndim == 3
        if is_seq:
            T = int(sample.shape[0])
            B = int(sample.shape[1])
            flat_obs: Dict[str, torch.Tensor] = {
                k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()
            }
            tds_flat = target_dist_slot.reshape(T * B, target_dist_slot.shape[-1])
            BB = T * B
        else:
            B = int(sample.shape[0])
            T = 0
            flat_obs = obs
            tds_flat = target_dist_slot
            BB = B

        features_flat = self.encoder(flat_obs, tds_flat)               # (BB, encoder_dim)
        weapon_flat = self.head(features_flat)                          # (BB, WEAPON_HEAD_SIZE)

        device = features_flat.device
        dtype = features_flat.dtype
        target_logits_flat = torch.zeros((BB, MAX_TOKEN_OBJECTS), dtype=dtype, device=device)
        target_query_flat = torch.zeros((BB, self.d_model), dtype=dtype, device=device)
        values_flat = torch.zeros((BB,), dtype=dtype, device=device)

        if is_seq:
            features = features_flat.reshape(T, B, -1)
            weapon_logits = weapon_flat.reshape(T, B, WEAPON_HEAD_SIZE)
            target_logits = target_logits_flat.reshape(T, B, MAX_TOKEN_OBJECTS)
            target_query = target_query_flat.reshape(T, B, self.d_model)
            values = values_flat.reshape(T, B)
            next_hidden = torch.zeros((B, 0), dtype=dtype, device=device)
        else:
            features = features_flat
            weapon_logits = weapon_flat
            target_logits = target_logits_flat
            target_query = target_query_flat
            values = values_flat
            next_hidden = torch.zeros((B, 0), dtype=dtype, device=device)

        logits: Dict[str, torch.Tensor] = {WEAPON_HEAD: weapon_logits}
        return features, logits, values, next_hidden, target_logits, target_query


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=weapon_token "
            "(no Python-level defaults — see qnn.bc.heads.templates)."
        )
    return probe[key]


def _build_weapon_token(probe: Mapping[str, Any]) -> HeadBuildResult:
    """HeadBuilder for ``weapon_token`` — every key required in probe.json."""
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    strip_self_embeds = bool(_required(probe, "strip_self_embeds"))
    weapon_embed_concat = bool(_required(probe, "weapon_embed_concat"))
    hidden = int(_required(probe, "hidden"))
    n_hidden_layers = int(_required(probe, "n_hidden_layers"))
    dropout = float(_required(probe, "dropout"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> WeaponTokenHead:
        del obs_dim, model_cfg
        return WeaponTokenHead(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon,
            strip_self_embeds=strip_self_embeds,
            weapon_embed_concat=weapon_embed_concat,
            hidden=hidden,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
        )

    return model_config, factory


WEAPON_TOKEN = HeadSpec(
    name="weapon",
    loss=HeadLossSpec(
        # The canonical weapon CE + per-class metrics live in policy.py;
        # these fields are schema-only here — the runner doesn't dispatch
        # through them. Kept for symmetry with fire_token's HeadLossSpec.
        loss_fn=lambda *_a, **_k: None,  # not called
        metrics_fn=lambda *_a, **_k: {},  # not called
        label_key="weapon",
        output_dim=WEAPON_HEAD_SIZE,
        selection_metric="f1_weapon_global",
        selection_lower_is_better=False,
    ),
    build=_build_weapon_token,
)
