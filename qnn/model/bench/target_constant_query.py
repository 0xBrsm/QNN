"""HeadSpec for the target-pointer constant-query null baseline.

Mirrors ``target_weapon_query`` / ``target_self_query_enemy`` scaffolding
exactly (ObsEmbedding + PreAttnEncoder + enemy-masked logits + motor
heads Off + canonical BC soft-CE on ``target_probs``) but replaces the
pointer's query with a single learned ``(d_model,)`` parameter shared
across all frames. The pointer reduces to ``nn.Linear(d_model, 1)``
applied per entity — no agent-state conditioning at all.

Tests the hypothesis that the query in the existing variants is
decorative: if this null baseline lands at the same ~0.911 accuracy as
weapon/cls queries under PreAttnEncoder + no GRU, then the entity
tokens carry sufficient signal and the query mechanism is doing no
per-frame work.

Required probe.json keys::

    {
      "head": "target_constant_query",
      "d_model": 64,
      "self_weapon_embed_in_self": false
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from qnn.model.bench.inputs.constant_query_target_pointer import (
    ConstantQueryTargetPointer,
)
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
            f"probe.json must define {key!r} for head=target_constant_query"
        )
    return probe[key]


def _build_target_constant_query(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        # Network supplies enemy_mask via TargetPointerInput; the pointer
        # reads it directly — no wrapper needed.
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=self_weapon,
                include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=ConstantQueryTargetPointer(d_model=d_model),
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


TARGET_CONSTANT_QUERY = HeadSpec(
    name="target_constant_query",
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_kl",
        selection_lower_is_better=True,
    ),
    build=_build_target_constant_query,
)
