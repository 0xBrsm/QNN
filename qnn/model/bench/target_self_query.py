"""HeadSpecs for the target-pointer self-query baseline.

Two variants, both apples-to-apples siblings of ``target_weapon_query``
(same ObsEmbedding + PreAttnEncoder + motor heads Off scaffolding,
same canonical BC soft-CE on ``target_probs``). The only thing they
change is the pointer:

  * ``target_self_query``         — :class:`CanonicalTargetPointer`
                                    with all flags False. Query =
                                    ``self_readout`` (post-encoder
                                    cls). ``entity_mask`` only — non-
                                    enemy entities are NOT masked out
                                    of the logits; the loss is
                                    responsible for steering mass to
                                    enemy indices.

  * ``target_self_query_enemy``   — same canonical pointer, but the
                                    bench pointer wrapper post-masks
                                    non-enemy indices to ``-1e9`` after
                                    the dot product (matching the
                                    ``target_weapon_query`` mask). Tests
                                    whether the enemy-only restriction
                                    is the source of any
                                    self-vs-weapon-query gap.

Required probe.json keys (both)::

    {
      "head": "target_self_query" | "target_self_query_enemy",
      "d_model": 64,
      "self_weapon_embed_in_self": false
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model.bench.inputs.canonical_target_pointer import CanonicalTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.spec import (
    HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config,
)
from qnn.model.bench.target import target_metrics, target_soft_ce_loss
from qnn.model.network import ModelConfig, Network, Off
from qnn.model.target import TargetPointerInput, TargetPointerOutput
from qnn.model.transformer import ObsEmbedding


class EnemyMaskedTargetPointer(CanonicalTargetPointer):
    """CanonicalTargetPointer + enemy-only post-mask on the logits.

    Runs the canonical forward, then re-applies ``inp.enemy_mask`` to
    the logits and recomputes ``target_feat`` from the re-masked
    softmax. ``enemy_mask`` is supplied by Network on every forward —
    no stash needed.
    """

    def forward(self, inp: TargetPointerInput) -> TargetPointerOutput:
        out = super().forward(inp)
        valid = inp.enemy_mask & inp.entity_mask
        valid_f = valid.to(out.target_logits.dtype)
        logits = out.target_logits.masked_fill(valid_f == 0, -1e9)
        has_any = (valid_f.sum(dim=-1, keepdim=True) > 0).to(logits.dtype)
        probs = F.softmax(logits, dim=-1)
        target_feat = (probs.unsqueeze(-1) * inp.entity_outs).sum(dim=1) * has_any
        return TargetPointerOutput(target_logits=logits, target_feat=target_feat)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=target_self_query[_enemy]"
        )
    return probe[key]


def _build_target_self_query(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        pointer = CanonicalTargetPointer(
            d_model=d_model,
            query_in_dim=d_model,
            inject_weapon=False,
            weapon_vocab=1,           # inert (inject_weapon=False)
            hard_target=False,
            linear_idx_prior=False,
            gt_dist_target_feat=False,
            prev_target_in_query=False,
        )
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=self_weapon,
                include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=pointer,
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


def _build_target_self_query_enemy(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        pointer = EnemyMaskedTargetPointer(
            d_model=d_model,
            query_in_dim=d_model,
            inject_weapon=False,
            weapon_vocab=1,           # inert (inject_weapon=False)
            hard_target=False,
            linear_idx_prior=False,
            gt_dist_target_feat=False,
            prev_target_in_query=False,
        )
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=self_weapon,
                include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=pointer,
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


TARGET_SELF_QUERY = HeadSpec(
    name="target_self_query",
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_skill",
        selection_lower_is_better=False,
    ),
    build=_build_target_self_query,
)


TARGET_SELF_QUERY_ENEMY = HeadSpec(
    name="target_self_query_enemy",
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_skill",
        selection_lower_is_better=False,
    ),
    build=_build_target_self_query_enemy,
)
