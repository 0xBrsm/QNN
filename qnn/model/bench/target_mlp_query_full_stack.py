"""HeadSpec for the target-pointer MLP-per-entity head with full encoder.

Variant of ``target_mlp_query`` that swaps the no-attention
``PreAttnEncoder`` for the canonical ``TransformerEncoder`` and turns
spatial tokens on in the ``ObsEmbedding``. Same MLPQueryTargetPointer,
no GRU, motor heads Off, canonical BC soft-CE on ``target_probs``.

Tests whether attention over the full token stream (self + 9 spatial +
entity tokens) lets the per-entity MLP scoring head extract sharper
target signal than the pre-attention baseline.

Required probe.json keys::

    {
      "head": "target_mlp_query_full_stack",
      "d_model": 64,
      "self_weapon_embed_in_self": false,
      "d_target": 64,
      "activation": "gelu",
      "n_heads": 2,
      "n_layers": 2,
      "d_ffn": 256,
      "attn_dropout": 0.0
    }
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from torch import nn

from qnn.model.bench.inputs.mlp_query_target_pointer import MLPQueryTargetPointer
from qnn.model.bench.spec import (
    HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config,
)
from qnn.model.bench.target import target_metrics, target_soft_ce_loss
from qnn.model.network import ModelConfig, Network, Off
from qnn.model.transformer import ObsEmbedding, TransformerEncoder


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=target_mlp_query_full_stack"
        )
    return probe[key]


def _build_target_mlp_query_full_stack(
    probe: Mapping[str, Any],
) -> HeadBuildResult:
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    d_target = int(_required(probe, "d_target"))
    activation = str(_required(probe, "activation"))
    n_heads = int(_required(probe, "n_heads"))
    n_layers = int(_required(probe, "n_layers"))
    d_ffn = int(_required(probe, "d_ffn"))
    attn_dropout = float(_required(probe, "attn_dropout"))

    model_config = dataclasses.replace(
        neutral_model_config(
            d_model=d_model, self_weapon_embed_in_self=self_weapon,
        ),
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        attn_dropout=attn_dropout,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=self_weapon,
                include_spatial=True,
            ),
            encoder=TransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ffn=d_ffn,
                dropout=attn_dropout,
            ),
            target_pointer=MLPQueryTargetPointer(
                d_model=d_model, d_target=d_target, activation=activation,
            ),
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


TARGET_MLP_QUERY_FULL_STACK = HeadSpec(
    name="target_mlp_query_full_stack",
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_skill",
        selection_lower_is_better=False,
    ),
    build=_build_target_mlp_query_full_stack,
)
