"""full_4head — the assembled 4-action-head bench model.

Brings the bench winners into ONE shared-trunk model (not the canonical heads):

    HeldWeaponSplitObsEmbedding  (split-self subtokens + held-weapon token)
      -> TransformerEncoder (CLS)
      -> GRU (feeds ALL heads; no bypass)
      -> { move:   CLSMoveHead
           look:   PurePolarLookHead   (the held-out Δloglik winner)
           attack: CLSAttackHead
           weapon: CLSWeaponHead       (8-way; held-weapon token via the CLS stream) }

No target pointer / no target_feat — nothing downstream uses it (attack ≈ dead on it,
weapon aggregate wash; see src/docs/{attack,weapon,target}-head.md). Each head reads the
GRU readout at the canonical selector dims (motor_in for move/look/attack, weapon_in for
weapon). Per-head losses are dispatched by QNNPolicy._compute_head_losses_and_metrics
(canonical move multi-axis CE / attack BCE / weapon 8-way CE; look carries its own
PurePolarLookHead.look_loss). The HeadLossSpec.loss_fn is a no-op stub (multi-head).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.bench.inputs.held_weapon_split_obs_embedding import HeldWeaponSplitObsEmbedding
from qnn.model.bench.move_cls_transformer import CLSMoveHead
from qnn.model.bench.attack_cls_transformer import CLSAttackHead
from qnn.model.bench.weapon_cls_transformer import CLSWeaponHead
from qnn.model.bench.look_head_polar import PurePolarLookHead
from qnn.model.network import Network, ModelConfig, Off, slot_dims
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=full_4head")
    return probe[key]


def _build_full_4head(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model    = int(_required(probe, "d_model"))
    n_heads    = int(_required(probe, "n_heads"))
    n_layers   = int(_required(probe, "n_layers"))
    d_ffn      = int(_required(probe, "d_ffn"))
    d_gru      = int(_required(probe, "d_gru"))
    d_move     = int(probe.get("d_move", 64))
    d_look     = int(probe.get("d_look", 64))
    d_attack   = int(probe.get("d_attack", 64))
    d_weapon   = int(probe.get("d_weapon", 64))
    activation = str(probe.get("activation", "gelu"))
    attn_dropout = float(probe.get("attn_dropout", 0.0))

    # GRU-to-all, weapon head on, weapon reads ONLY the GRU readout (no target_feat,
    # no self_readout). Target pointer is Off.
    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=True, d_gru=d_gru, look_bypass_gru=False,
        use_weapon_head=True, weapon_sources=("gru",),
        d_move=d_move, d_look=d_look, d_attack=d_attack, d_weapon=d_weapon,
    )
    dims = slot_dims(
        d_model=d_model, d_gru=d_gru, has_temporal=True,
        has_target_pointer=False, has_weapon_head=True,
        weapon_sources=model_config.weapon_sources,
    )
    # No pointer → no target_feat block. motor_in = gru readout (d_gru) + weapon
    # context (d_model); weapon_in = gru readout (d_gru). No dead zero pad.
    motor_in = dims["motor_in"]    # move/look/attack selector
    weapon_in = dims["weapon_in"]  # weapon selector

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=HeldWeaponSplitObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=False, include_spatial=True,
            ),
            encoder=TransformerEncoder(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                d_ffn=d_ffn, dropout=attn_dropout,
            ),
            temporal=Temporal(d_model, d_gru),
            target_pointer=Off,
            move_head=CLSMoveHead(in_dim=motor_in, d_move=d_move, activation=activation),
            look_head=PurePolarLookHead(motor_in, d_look, activation),
            attack_head=CLSAttackHead(in_dim=motor_in, d_attack=d_attack, activation=activation),
            weapon_head=CLSWeaponHead(in_dim=weapon_in, d_model=d_model, d_weapon=d_weapon, activation=activation),
        )

    return model_config, factory


def _stub(*_a: Any, **_k: Any) -> torch.Tensor:
    return torch.zeros(())


FULL_4HEAD = HeadSpec(
    name="full_4head",
    loss=HeadLossSpec(
        # Multi-head: QNNPolicy computes every head's loss; this stub is never
        # dispatched (mirrors weapon_aim). Best-epoch uses the composite
        # _selection_score, not this field.
        loss_fn=_stub,
        metrics_fn=lambda *_a, **_k: {},
        label_key="look",
        output_dim=0,
        selection_metric="loss",
        selection_lower_is_better=True,
    ),
    build=_build_full_4head,
)
