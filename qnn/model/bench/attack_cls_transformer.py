"""attack_cls_transformer — attack head driven by CLS of a full transformer encoder.

1:1 clone of ``move_cls_transformer`` with a binary attack output instead of the
3×3 move output. The point is to test attack in the SAME clean regime move/look
were tested in — full attention over the whole token stream pooled into CLS,
with/without GRU temporal recurrence — rather than the PreAttn + GT-oracle
scaffolds every prior attack probe used.

Token stream (DmgRadWeaponSplitObsEmbedding + include_spatial=True):
    [CLS, state, arsenal, motion, weapon(dmg+rad), spatial_0..8, entity_0..N-1]

Encoder: TransformerEncoder(d_model, n_heads, n_layers, d_ffn).
Attack head: MLP over the CLS readout → 1 logit (BCE, canonical AttackHead contract).
  - no GRU:  inp.features = cat(self_readout=CLS, target_feat=0) → head reads CLS (d_model).
  - use_gru: Temporal(d_model, d_gru) integrates CLS over time;
             inp.features = cat(gru_flat, target_feat=0) → head reads gru_flat (d_gru).
No target pointer (target_feat half is zeros); all other heads Off. The held-weapon
embed is off (self_weapon_embed_in_self=False) — dead for attack and the weapon-head
leak control point.

probe.json required keys: d_model, n_heads, n_layers, d_ffn, d_attack, activation
Optional: attn_dropout (default 0.0), use_gru (default false), d_gru (required if use_gru)
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import OUT_DIM, AttackHeadInput, AttackHeadOutput
from qnn.model.bench.inputs.dmg_rad_weapon_split_obs_embedding import (
    DmgRadWeaponSplitObsEmbedding,
)
from qnn.model.bench.inputs.mlp_query_target_pointer import MLPQueryTargetPointer
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.network import ModelConfig, Network, Off
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder


class CLSAttackHead(nn.Module):
    """Attack head: MLP over the leading ``in_dim`` features (CLS or GRU(CLS)).

    Mirrors ``CLSMoveHead``: slices ``inp.features[..., :in_dim]`` so the zeroed
    target_feat half (pointer Off) is dropped rather than fed as dead dims.
    """

    def __init__(self, *, in_dim: int, d_attack: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_attack, activation)

    def forward(self, inp: AttackHeadInput) -> AttackHeadOutput:
        feats = inp.features[..., : self.in_dim]
        delta_attack = self.mlp(feats)
        return AttackHeadOutput(
            attack_logit=delta_attack,
            prior_logit=torch.zeros_like(delta_attack),
            delta_attack=delta_attack,
        )


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=attack_cls_transformer")
    return probe[key]


def _build_attack_cls_transformer(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model      = int(_required(probe, "d_model"))
    n_heads      = int(_required(probe, "n_heads"))
    n_layers     = int(_required(probe, "n_layers"))
    d_ffn        = int(_required(probe, "d_ffn"))
    d_attack     = int(_required(probe, "d_attack"))
    activation   = str(_required(probe, "activation"))
    attn_dropout = float(probe.get("attn_dropout", 0.0))
    use_gru      = bool(probe.get("use_gru", False))
    d_gru        = int(_required(probe, "d_gru")) if use_gru else 0
    # Optional real target pointer: feeds target_feat alongside the CLS/GRU
    # readout. Tests whether explicit target localization substitutes for what
    # the GRU provides (CLS+target vs GRU comparison).
    use_target_pointer = bool(probe.get("use_target_pointer", False))
    d_target           = int(probe.get("d_target", 64))

    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=use_gru, d_gru=d_gru, look_bypass_gru=False,
    )
    # features_base_flat = cat(readout, target_feat), readout = gru_flat (use_gru)
    # else self_readout=CLS. With the pointer off, target_feat is zeros and we
    # slice it off; with it on, the head reads both halves.
    readout_dim = d_gru if use_gru else d_model
    in_dim = readout_dim + (d_model if use_target_pointer else 0)

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
            target_pointer=(
                MLPQueryTargetPointer(d_model=d_model, d_target=d_target, activation=activation)
                if use_target_pointer else Off
            ),
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=CLSAttackHead(in_dim=in_dim, d_attack=d_attack, activation=activation),
        )

    return model_config, factory


def _stub_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {}


ATTACK_CLS_TRANSFORMER = HeadSpec(
    name="attack_cls_transformer",
    loss=HeadLossSpec(
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="attack",
        output_dim=OUT_DIM,
        selection_metric="attack_skill",
        selection_lower_is_better=False,
    ),
    build=_build_attack_cls_transformer,
)
