"""move_cls_transformer — move head driven by CLS of a full transformer encoder.

Token stream (via DmgRadWeaponSplitObsEmbedding + include_spatial=True):
    [CLS, state, arsenal, motion, weapon(dmg+rad), spatial_0..8, entity_0..N-1]

Encoder: TransformerEncoder(d_model, n_heads, n_layers, d_ffn)
Move head: MLP over the CLS readout.
  - no GRU:  inp.features = cat(self_readout=CLS, target_feat=0) → head reads CLS (d_model).
  - use_gru: Temporal(d_model, d_gru) integrates CLS over time;
             inp.features = cat(gru_flat, target_feat=0) → head reads gru_flat (d_gru).
No target pointer.

probe.json required keys: d_model, n_heads, n_layers, d_ffn, d_move, activation
Optional: attn_dropout (default 0.0), use_gru (default false), d_gru (required if use_gru)
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.dmg_rad_weapon_split_obs_embedding import (
    DmgRadWeaponSplitObsEmbedding,
)
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.move_head import MoveHeadInput, MoveHeadOutput
from qnn.model.network import ModelConfig, Network, Off
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder

_OUT_DIM = MOVE_AXES * MOVE_AXIS_CLASSES  # 9


class CLSMoveHead(nn.Module):
    """Move head: MLP over the leading ``in_dim`` features.

    No-GRU: in_dim=d_model reads the CLS self_readout (target_feat half is zeros).
    GRU:    in_dim=d_gru reads gru_flat (the GRU-integrated CLS; target_feat half zeros).
    """

    def __init__(self, *, in_dim: int, d_move: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        feats = inp.features[..., : self.in_dim]
        logits = self.mlp(feats).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=move_cls_transformer")
    return probe[key]


def _build_move_cls_transformer(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model      = int(_required(probe, "d_model"))
    n_heads      = int(_required(probe, "n_heads"))
    n_layers     = int(_required(probe, "n_layers"))
    d_ffn        = int(_required(probe, "d_ffn"))
    d_move       = int(_required(probe, "d_move"))
    activation   = str(_required(probe, "activation"))
    attn_dropout = float(probe.get("attn_dropout", 0.0))
    use_gru      = bool(probe.get("use_gru", False))
    d_gru        = int(_required(probe, "d_gru")) if use_gru else 0

    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=use_gru, d_gru=d_gru, look_bypass_gru=False,
    )
    # No-GRU: features = cat(self_readout=CLS, target_feat=0) → read CLS (d_model).
    # GRU:    features = cat(gru_flat, target_feat=0)         → read gru_flat (d_gru).
    in_dim = d_gru if use_gru else d_model

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=DmgRadWeaponSplitObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=False,
                include_spatial=True,
            ),
            encoder=TransformerEncoder(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                d_ffn=d_ffn, dropout=attn_dropout,
            ),
            temporal=(Temporal(d_model, d_gru) if use_gru else Off),
            target_pointer=Off,
            move_head=CLSMoveHead(in_dim=in_dim, d_move=d_move, activation=activation),
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


def _stub_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {}


MOVE_CLS_TRANSFORMER = HeadSpec(
    name="move_cls_transformer",
    loss=HeadLossSpec(
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="move",
        output_dim=_OUT_DIM,
        # Declarative only — the live selection is the _selection_score composite
        # in qnn.bc.train, which prefers move_dll (distributional Δloglik,
        # human-likeness) over f1_move's argmax point-accuracy. Kept in sync here
        # for documentation.
        selection_metric="move_dll",
        selection_lower_is_better=False,
    ),
    build=_build_move_cls_transformer,
)
