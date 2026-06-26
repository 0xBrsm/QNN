"""look_cls — "pull out all the stops" look probe.

Full transformer encoder over ALL tokens with a dedicated CLS (SplitSelfObsEmbedding:
CLS + state + arsenal + motion + spatial + entities), GRU temporal recurrence over the
CLS readout, and a pure-MLP look head — NO target pointer, NO target-anchored prior,
all other heads Off.

Rationale: every prior look probe constrained the head to the combat target
(target_feat / look_prior), which structurally can't represent looking at non-target
things (~91% of the per-tick turn we can't explain). Here the encoder attends over all
tokens and pools them into CLS; the GRU integrates CLS over time (PPO-safe momentum —
the GRU input is the obs-derived CLS, not a fed previous-look input). The look head then predicts the
turn purely from cat(gru_flat, target_feat=zeros) = MLP(GRU(CLS)). Tests whether full
capacity + all information + PPO-safe temporal context beats the ~0.09 grounded ceiling.

Single-head probe: the runner computes the canonical look regression loss / look_r2 via
QNNPolicy._compute_head_losses_and_metrics, gated by head_loss_weights (look=1, rest 0).
The HeadSpec.loss below is a stub.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qnn.model.bench.inputs.split_self_obs_embedding import SplitSelfObsEmbedding
from qnn.model.bench.inputs.held_weapon_split_obs_embedding import HeldWeaponSplitObsEmbedding
from qnn.model.bench.look_head_pure import PureLookHead, PureBinnedLookHead
from qnn.model.bench.look_head_polar import PurePolarLookHead
from qnn.model.bench.look_head_vmf import PureVMFLookHead
from qnn.model.bench.spec import (
    HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config,
)
from qnn.model.network import ModelConfig, Network, Off, slot_dims
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=look_cls")
    return probe[key]


def _build_look_cls(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model      = int(_required(probe, "d_model"))
    d_gru        = int(_required(probe, "d_gru"))
    d_hidden     = int(_required(probe, "d_hidden"))
    activation   = str(_required(probe, "activation"))
    n_heads      = int(_required(probe, "n_heads"))
    n_layers     = int(_required(probe, "n_layers"))
    d_ffn        = int(_required(probe, "d_ffn"))
    attn_dropout = float(_required(probe, "attn_dropout"))
    self_weapon     = bool(probe.get("self_weapon_embed_in_self", False))
    held_weapon     = bool(probe.get("held_weapon", False))  # add dedicated held-weapon token
    binned          = bool(probe.get("binned", False))       # legacy: binned (CE) vs regression
    # look_dist selects the look output distribution. Each non-regression head
    # carries its own loss (the look_loss hook), so adding one needs no canonical
    # change. Default preserves the legacy `binned` bool.
    look_dist       = str(probe.get("look_dist", "binned" if binned else "regression"))
    n_components    = int(probe.get("n_components", 3))       # vMF mixture size
    # use_temporal=False removes the GRU entirely (temporal=Off). The look head then
    # sees cat(self_readout, target_feat=0) from a single frame — no obs-derived
    # momentum reconstruction is possible.
    use_temporal = bool(probe.get("use_temporal", True))
    _Emb = HeldWeaponSplitObsEmbedding if held_weapon else SplitSelfObsEmbedding

    def _build_look_head(in_dim: int) -> nn.Module:
        if look_dist == "regression":
            return PureLookHead(in_dim, d_hidden, activation)
        if look_dist == "binned":
            return PureBinnedLookHead(in_dim, d_hidden, activation)
        if look_dist == "polar":
            return PurePolarLookHead(in_dim, d_hidden, activation)
        if look_dist == "vmf":
            return PureVMFLookHead(in_dim, d_hidden, activation, n_components=n_components)
        raise RuntimeError(f"probe.json look_dist={look_dist!r} not in regression|binned|polar|vmf")

    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=self_weapon),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=use_temporal, d_gru=(d_gru if use_temporal else 0), look_bypass_gru=False,
    )
    dims = slot_dims(
        d_model=model_config.d_model, d_gru=model_config.d_gru,
        has_temporal=use_temporal, has_target_pointer=False, has_weapon_head=False,
        weapon_sources=model_config.weapon_sources,
    )
    # temporal on: (d_gru + d_model); off: (2 * d_model). target_feat half is zeros (pointer Off).
    motor_in = dims["motor_in"]

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=_Emb(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=True,
            ),
            encoder=TransformerEncoder(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                d_ffn=d_ffn, dropout=attn_dropout,
            ),
            temporal=(Temporal(d_model, d_gru) if use_temporal else Off),
            target_pointer=Off,
            move_head=Off,
            look_head=_build_look_head(motor_in),
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


def _stub_loss(*args, **kwargs) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args, **kwargs) -> dict[str, float]:
    return {}


LOOK_CLS = HeadSpec(
    name="look_cls",
    loss=HeadLossSpec(
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="look",
        output_dim=3,
        # Distributional heads select on look_dll (Δloglik, human-likeness);
        # regression on look_r2. Metadata only — actual best-epoch tracking is the
        # _selection_score composite in qnn.bc.train, which prefers look_dll with a
        # look_r2 fallback.
        selection_metric="look_dll",
        selection_lower_is_better=False,
    ),
    build=_build_look_cls,
)
