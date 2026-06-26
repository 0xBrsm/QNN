"""weapon_cls_transformer — weapon head driven by CLS of a full transformer encoder.

Weapon-head analogue of ``attack_cls_transformer``. Tests the weapon classifier
in the full canonical regime — attention over the whole token stream pooled into
CLS, with/without GRU temporal recurrence — rather than the PreAttn + GT-oracle
scaffold every prior weapon probe used. This is the first weapon-selection probe
on the real encoder stack the canonical model actually uses.

Token stream (SplitSelfObsEmbedding + include_spatial=True):
    [CLS, state, arsenal, motion, spatial_0..8, entity_0..N-1]

The WEAPON TOKEN IS DROPPED: ``attack_cls_transformer`` uses
``DmgRadWeaponSplitObsEmbedding`` which adds a held-weapon (dmg+rad+readiness+
attack_finished+weapon_id) subtoken — the incumbent signal. We use plain
``SplitSelfObsEmbedding`` (no weapon subtoken) + ``self_weapon_embed_in_self=
False`` so the head cannot see which weapon it is currently holding. Keeps the
no-incumbent property of the locked arsenal+motion form while letting attention/
temporal supply context.

Encoder: TransformerEncoder(d_model, n_heads, n_layers, d_ffn).
Weapon head: MLP over the readout → WEAPON_HEAD_SIZE logits (canonical weapon CE
in policy.py; HeadLossSpec is a stub, mirroring weapon_arsenal/weapon_preattn).
  - no GRU:  selector = cat(self_readout=CLS, target_feat=0); head reads CLS (d_model).
  - use_gru: Temporal integrates CLS over time; selector = cat(gru_flat, target_feat=0);
             head reads gru_flat (d_gru). (weapon_sources=("gru","target_feat").)
No target pointer (target_feat half is zeros; CLS attends entities for target context).

probe.json required keys: d_model, n_heads, n_layers, d_ffn, d_weapon, activation
Optional: attn_dropout (default 0.0), use_gru (default false), d_gru (required if use_gru),
include_weapon_token (default false — when true, swap SplitSelfObsEmbedding for
HeldWeaponSplitObsEmbedding so the CLS stream carries a dedicated held-weapon token
[weapon_static stats + weapon_id embed]; the explicit-token held-weapon input on the
full attention+GRU encoder, the prod-relevant delivery from §5/§6).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.held_weapon_split_obs_embedding import HeldWeaponSplitObsEmbedding
from qnn.model.bench.inputs.split_self_obs_embedding import SplitSelfObsEmbedding
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.network import ModelConfig, Network, Off
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder
from qnn.model.weapon_head import WeaponHeadInput, WeaponHeadOutput
from qnn.schema import WEAPON_HEAD_SIZE


class CLSWeaponHead(nn.Module):
    """Weapon head: MLP over the leading ``in_dim`` features (CLS or GRU(CLS)).

    Mirrors ``CLSAttackHead``: slices ``inp.selector[..., :in_dim]`` so the zeroed
    target_feat half (pointer Off) is dropped rather than fed as dead dims.
    """

    def __init__(self, *, in_dim: int, d_model: int, d_weapon: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.mlp = make_head_mlp(in_dim, WEAPON_HEAD_SIZE, d_weapon, activation)
        # Soft-mix context for motor heads (all Off here, but keep the contract).
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, d_model)

    def forward(self, inp: WeaponHeadInput) -> WeaponHeadOutput:
        feats = inp.selector[..., : self.in_dim]
        logits = self.mlp(feats)
        context = F.softmax(logits, dim=-1) @ self.embed.weight
        return WeaponHeadOutput(logits=logits, context=context)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=weapon_cls_transformer")
    return probe[key]


def _build_weapon_cls_transformer(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model      = int(_required(probe, "d_model"))
    n_heads      = int(_required(probe, "n_heads"))
    n_layers     = int(_required(probe, "n_layers"))
    d_ffn        = int(_required(probe, "d_ffn"))
    d_weapon     = int(_required(probe, "d_weapon"))
    activation   = str(_required(probe, "activation"))
    attn_dropout = float(probe.get("attn_dropout", 0.0))
    use_gru      = bool(probe.get("use_gru", False))
    d_gru        = int(_required(probe, "d_gru")) if use_gru else 0
    # include_weapon_token: add a dedicated held-weapon token to the CLS stream
    # (HeldWeaponSplitObsEmbedding's 5th self subtoken = weapon_static stats +
    # weapon_id embed) instead of dropping the incumbent. The explicit-token form
    # of the held-weapon input — same lever as weapon_arsenal use_weapon_token,
    # but on the full attention encoder (+ GRU). See src/docs/weapon-head.md §5/§6.
    include_weapon_token = bool(probe.get("include_weapon_token", False))

    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=use_gru, d_gru=d_gru, look_bypass_gru=False,
        use_weapon_head=True,
        # Readout-only selector: gru_flat when temporal, else self_readout=CLS.
        # The readout source leads so CLSWeaponHead's selector[..., :in_dim]
        # slice picks it up and drops the zeroed target_feat tail.
        weapon_sources=(("gru",) if use_gru else ("self_readout",)) + ("target_feat",),
    )
    in_dim = d_gru if use_gru else d_model

    emb_cls = HeldWeaponSplitObsEmbedding if include_weapon_token else SplitSelfObsEmbedding

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=emb_cls(
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
            move_head=Off,
            look_head=Off,
            attack_head=Off,
            weapon_head=CLSWeaponHead(
                in_dim=in_dim, d_model=d_model, d_weapon=d_weapon, activation=activation,
            ),
        )

    return model_config, factory


def _stub_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {}


WEAPON_CLS_TRANSFORMER = HeadSpec(
    name="weapon_cls_transformer",
    loss=HeadLossSpec(
        # Canonical weapon CE + per-class metrics live in policy.py; the runner
        # doesn't dispatch through these for weapon (mirrors weapon_preattn).
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="weapon",
        output_dim=WEAPON_HEAD_SIZE,
        selection_metric="weapon_skill",
        selection_lower_is_better=False,
    ),
    build=_build_weapon_cls_transformer,
)
