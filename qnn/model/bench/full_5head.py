"""full_5head — full_4head + the canonical target pointer.

Identical trunk and heads to full_4head (HeldWeaponSplitObsEmbedding ->
TransformerEncoder(CLS) -> GRU -> CLS move/attack/weapon + PurePolarLookHead),
plus the canonical ``TargetPointer`` (per-entity MLP scorer, the bench KL
winner — see src/docs/target-head.md §2) in the pointer slot:

    target_logits — supervised by the v3 distribution labeler (soft-CE,
                    present-weighted; selection on val_target_kl).
    target_feat   — pointer-blended entity vector, concatenated into the
                    motor feature base, so move/look/attack condition on a
                    persistent pooled (rel, vel) view of the engaged enemy.

The weapon head keeps weapon_sources=("gru",) by default — the validated
full_4head winner. Probe key ``weapon_target_feat: true`` switches it to
("gru", "target_feat") for the second ablation: the original exclusion rested
on un-gated bench-form macro-F1 (arsenal+motion 0.4925 vs +target 0.4825,
weapon-head.md §1) and was never graded on the current primary weapon metrics
(threshold-swept gate operating point + dwell/switch distributional fidelity);
per-class, target range is a real LG-viability signal (LG f1 +32%).

Rationale: offline BC deltas are expected ~flat by construction (the labels
derive from human aim, so target_feat is in-distribution redundant —
fire-discrimination.md §3). The hypothesis is OOD/closed-loop: an explicit
pointer gives look/attack a target representation that stays stable under the
bot's own drifting trajectory (tracking/lead substrate). Judge this run on
live eval metrics (obs_blind_fire_*, engine_fire_tracking_cos, time-to-frag),
not per-frame val metrics.
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
from qnn.model.network import Network, ModelConfig, slot_dims
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=full_5head")
    return probe[key]


def _build_full_5head(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model    = int(_required(probe, "d_model"))
    n_heads    = int(_required(probe, "n_heads"))
    n_layers   = int(_required(probe, "n_layers"))
    d_ffn      = int(_required(probe, "d_ffn"))
    d_gru      = int(_required(probe, "d_gru"))
    d_target   = int(_required(probe, "d_target"))
    d_move     = int(probe.get("d_move", 64))
    d_look     = int(probe.get("d_look", 64))
    d_attack   = int(probe.get("d_attack", 64))
    d_weapon   = int(probe.get("d_weapon", 64))
    activation = str(probe.get("activation", "gelu"))
    attn_dropout = float(probe.get("attn_dropout", 0.0))
    weapon_sources = (
        ("gru", "target_feat") if bool(probe.get("weapon_target_feat", False))
        else ("gru",)
    )

    model_config = dataclasses.replace(
        neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False),
        n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn, attn_dropout=attn_dropout,
        use_gru=True, d_gru=d_gru, look_bypass_gru=False,
        use_weapon_head=True, weapon_sources=weapon_sources,
        d_target=d_target,
        d_move=d_move, d_look=d_look, d_attack=d_attack, d_weapon=d_weapon,
    )
    dims = slot_dims(
        d_model=d_model, d_gru=d_gru, has_temporal=True,
        has_target_pointer=True, has_weapon_head=True,
        weapon_sources=model_config.weapon_sources,
    )
    # Pointer present -> motor feature base = cat(gru readout, target_feat);
    # motor_in grows by d_model vs full_4head. weapon_in follows weapon_sources.
    motor_in = dims["motor_in"]
    weapon_in = dims["weapon_in"]

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
            # target_pointer omitted (None) -> Network builds the canonical
            # TargetPointer(d_model, d_target) and cats target_feat into the
            # motor feature base.
            move_head=CLSMoveHead(in_dim=motor_in, d_move=d_move, activation=activation),
            look_head=PurePolarLookHead(motor_in, d_look, activation),
            attack_head=CLSAttackHead(in_dim=motor_in, d_attack=d_attack, activation=activation),
            weapon_head=CLSWeaponHead(in_dim=weapon_in, d_model=d_model, d_weapon=d_weapon, activation=activation),
        )

    return model_config, factory


def _stub(*_a: Any, **_k: Any) -> torch.Tensor:
    return torch.zeros(())


FULL_5HEAD = HeadSpec(
    name="full_5head",
    loss=HeadLossSpec(
        # Multi-head: QNNPolicy computes every head's loss (incl. the target
        # soft-CE / target_kl path); this stub is never dispatched.
        loss_fn=_stub,
        metrics_fn=lambda *_a, **_k: {},
        label_key="look",
        output_dim=0,
        selection_metric="loss",
        selection_lower_is_better=True,
    ),
    build=_build_full_5head,
)
