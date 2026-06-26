"""temporal_probe — GRU vs separable-TCN in the temporal slot, full network.

A parity ablation for the temporal component. Unlike the per-head probes, this
keeps the **entire canonical network** (ObsEmbedding + TransformerEncoder +
TargetPointer + all four heads) and swaps **only** the ``temporal`` slot:

  * ``variant="gru"`` — canonical ``Temporal`` (single-layer GRU, ``d_gru``).
  * ``variant="tcn"`` — :class:`qnn.model.temporal_tcn.SeparableTCN`, a causal
    dilated depthwise-separable stack with the same I/O contract and output
    width (``d_gru``), sized to a receptive field that matches the TBPTT
    window (default RF 95 ~ 96 frames, ~26k params at ``C=64`` = GRU parity).

Everything downstream (head MLP widths, target pointer, loss weights) is
identical across variants, so a paired run isolates the temporal architecture.

Losses/metrics are the canonical multi-head pipeline (``head_loss_weights`` in
train.json drives them); the ``HeadLossSpec`` here is a stub, same as the other
full-network probes (e.g. ``weapon_aim``).

Required probe.json keys::

    {
      "head": "temporal", "variant": "tcn",
      "d_model": 64, "n_heads": 2, "n_layers": 2, "d_ffn": 256,
      "attn_dropout": 0.0, "d_gru": 64, "d_target": 64,
      "self_weapon_embed_in_self": false,
      "use_weapon_head": true,
      "weapon_sources": ["gru", "self_readout", "target_feat"],
      "weapon_context_from_obs": false,
      "weapon_switch_confidence": 0.65, "weapon_switch_margin": 0.15,
      "d_move": 128, "d_look": 128, "d_attack": 128, "d_weapon": 128,
      "head_activation": "gelu"
    }

Optional TCN knobs (used only when ``variant="tcn"``)::

    "tcn_kernel": 3, "tcn_dilations": [1, 2, 4, 8, 16, 16],
    "tcn_channels": 0, "tcn_separable": true, "tcn_activation": "gelu",
    "tcn_dropout": 0.0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec
from qnn.model.network import ModelConfig, Network
from qnn.model.temporal import Temporal
from qnn.model.temporal_tcn import SeparableTCN, TCNConfig

_VARIANTS = ("gru", "tcn")


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=temporal")
    return probe[key]


def _build_temporal(probe: Mapping[str, Any]) -> HeadBuildResult:
    variant = str(_required(probe, "variant"))
    if variant not in _VARIANTS:
        raise RuntimeError(f"variant must be one of {_VARIANTS}, got {variant!r}")

    d_model = int(_required(probe, "d_model"))
    d_gru = int(_required(probe, "d_gru"))

    # Full canonical ModelConfig — identical across variants so the only
    # difference is the temporal module the factory injects.
    model_config = ModelConfig(
        d_model=d_model,
        n_heads=int(_required(probe, "n_heads")),
        n_layers=int(_required(probe, "n_layers")),
        d_ffn=int(_required(probe, "d_ffn")),
        attn_dropout=float(_required(probe, "attn_dropout")),
        use_gru=True,  # always on: sizes motor heads on d_gru + allocates the carry buffer
        d_gru=d_gru,
        use_weapon_head=bool(_required(probe, "use_weapon_head")),
        weapon_switch_confidence=float(_required(probe, "weapon_switch_confidence")),
        weapon_switch_margin=float(_required(probe, "weapon_switch_margin")),
        weapon_sources=tuple(_required(probe, "weapon_sources")),
        weapon_context_from_obs=bool(_required(probe, "weapon_context_from_obs")),
        look_bypass_gru=False,
        d_target=int(_required(probe, "d_target")),
        self_weapon_embed_in_self=bool(_required(probe, "self_weapon_embed_in_self")),
        d_move=int(_required(probe, "d_move")),
        d_look=int(_required(probe, "d_look")),
        d_attack=int(_required(probe, "d_attack")),
        d_weapon=int(_required(probe, "d_weapon")),
        head_activation=str(_required(probe, "head_activation")),
    )

    tcn_cfg = TCNConfig(
        kernel_size=int(probe.get("tcn_kernel", 3)),
        dilations=tuple(int(x) for x in probe.get("tcn_dilations", (1, 2, 4, 8, 16, 16))),
        channels=int(probe.get("tcn_channels", 0)),
        separable=bool(probe.get("tcn_separable", True)),
        activation=str(probe.get("tcn_activation", "gelu")),
        dropout=float(probe.get("tcn_dropout", 0.0)),
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        # temporal=None -> Network builds the canonical GRU (use_gru=True).
        # An override injects the TCN. All other slots None -> canonical.
        temporal = (
            SeparableTCN(d_model=model_cfg.d_model, hidden_dim=model_cfg.d_gru, cfg=tcn_cfg)
            if variant == "tcn"
            else None
        )
        return Network(obs_dim=obs_dim, model=model_cfg, temporal=temporal)

    return model_config, factory


def _stub_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {}


TEMPORAL_PROBE = HeadSpec(
    name="temporal",
    loss=HeadLossSpec(
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="look",
        output_dim=3,
        # Declarative only — live selection is the _selection_score composite in
        # qnn.bc.train (all heads active). Look is the most temporal-sensitive
        # head, so look_dll is the metric to watch for a GRU-vs-TCN delta.
        selection_metric="look_dll",
        selection_lower_is_better=False,
    ),
    build=_build_temporal,
)
