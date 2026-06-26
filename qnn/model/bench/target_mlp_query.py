"""HeadSpec for the target-pointer MLP-per-entity scoring head.

Mirrors ``target_constant_query`` (no agent-state conditioning, no
cross-entity context, enemy + entity mask on logits, motor heads Off,
PreAttnEncoder scaffolding, canonical BC soft-CE on ``target_probs``)
but replaces the pointer's ``Linear(d_model, 1)`` scoring with a small
MLP applied independently to each entity vector.

Tests whether the linear scoring head was the bottleneck for the
~0.911 plateau the other variants share, holding everything else
constant (same encoder, same mask, same loss path, no GRU).

Required probe.json keys::

    {
      "head": "target_mlp_query",
      "d_model": 64,
      "self_weapon_embed_in_self": false,
      "d_target": 64,
      "activation": "gelu"
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from qnn.model.bench.inputs.mlp_query_target_pointer import MLPQueryTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.spec import (
    HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config,
)
from qnn.model.bench.target import target_metrics, target_soft_ce_loss
from qnn.model.network import ModelConfig, Network, Off
from qnn.model.transformer import ObsEmbedding


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=target_mlp_query"
        )
    return probe[key]


def _build_target_mlp_query(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    d_target = int(_required(probe, "d_target"))
    activation = str(_required(probe, "activation"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=self_weapon,
                include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
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


TARGET_MLP_QUERY = HeadSpec(
    name="target_mlp_query",
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_skill",
        selection_lower_is_better=False,
    ),
    build=_build_target_mlp_query,
)
