"""build_network — assemble a Network from a GraphSpec.

The single model factory: BC, bench probes, eval, PPO, and ONNX export
all construct the model here. Nodes are resolved from small string
registries (encoder types, pointer types, per-head type tables); edge
widths come from ``slot_dims`` fed with the spec's resolved node
widths; ``Network`` stays the executor of the (fixed) dataflow.

``model_config_from_graph`` bridges a GraphSpec onto the flat
``ModelConfig`` that ``Network`` / ``QNNPolicy`` still consume for
policy-layer flags; ``graph_from_model_config`` is the reverse
migration for legacy flat checkpoints (v20+ canonical layouts — the
v17 ``look_bypass_gru`` layout is not expressible as a graph and keeps
loading through the legacy path).
"""

from __future__ import annotations

import torch.nn as nn

from qnn.model.attack_head import AttackHead
from qnn.model.graph.embedding import GraphObsEmbedding
from qnn.model.graph.spec import (
    EDGE_READOUT, EDGE_TARGET_FEAT,
    WEAPON_EDGE_TO_SOURCE, WEAPON_SOURCE_TO_EDGE,
    GraphSpec, GraphSpecError, HeadNodeSpec, TokenSpec,
    TOKEN_KIND_ENTITIES, TOKEN_KIND_SPATIAL,
    EncoderSpec, PointerSpec, TemporalSpec,
    monolithic_self_token,
)
from qnn.model.look_head import LookHead
from qnn.model.move_head import MoveHead
from qnn.model.network import ModelConfig, Network, Off, slot_dims
from qnn.model.target import TargetPointer
from qnn.model.temporal import Temporal
from qnn.model.transformer import TransformerEncoder
from qnn.model.weapon_head import WeaponHead

# Validated bench winners — promoted into the graph head registry.
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.cls_heads import CLSAttackHead, CLSMoveHead, CLSWeaponHead
from qnn.model.look_head_polar import PurePolarLookHead


# ── Head type registry ───────────────────────────────────────────────
# Keyed [head name][type]. Builders receive the head spec, the resolved
# dim contract from slot_dims, and d_model.

HEAD_TYPES: dict[str, dict[str, object]] = {
    "move": {
        "canonical": lambda h, dims, d_model: MoveHead(
            in_dim=dims["motor_in"], d_hidden=h.d_hidden, activation=h.activation),
        "cls": lambda h, dims, d_model: CLSMoveHead(
            in_dim=dims["motor_in"], d_move=h.d_hidden, activation=h.activation),
    },
    "look": {
        "canonical": lambda h, dims, d_model: LookHead(
            in_dim=dims["motor_in"], d_hidden=h.d_hidden, activation=h.activation),
        "polar": lambda h, dims, d_model: PurePolarLookHead(
            dims["motor_in"], h.d_hidden, h.activation),
    },
    "attack": {
        "canonical": lambda h, dims, d_model: AttackHead(
            in_dim=dims["motor_in"], d_hidden=h.d_hidden, activation=h.activation),
        "cls": lambda h, dims, d_model: CLSAttackHead(
            in_dim=dims["motor_in"], d_attack=h.d_hidden, activation=h.activation),
    },
    "weapon": {
        "canonical": lambda h, dims, d_model: WeaponHead(
            selector_dim=dims["weapon_in"], d_model=d_model, d_hidden=h.d_hidden,
            activation=h.activation, context_from_obs=h.context_from_obs),
        "cls": lambda h, dims, d_model: CLSWeaponHead(
            in_dim=dims["weapon_in"], d_model=d_model, d_weapon=h.d_hidden,
            activation=h.activation),
    },
}


def _build_head(head: HeadNodeSpec, dims: dict[str, int], d_model: int) -> nn.Module:
    table = HEAD_TYPES[head.name]
    if head.type not in table:
        raise GraphSpecError(
            f"heads[{head.name!r}]: unknown type {head.type!r}; "
            f"registered: {sorted(table)}"
        )
    return table[head.type](head, dims, d_model)


def _weapon_sources(spec: GraphSpec) -> tuple[str, ...]:
    """ModelConfig.weapon_sources from the weapon head's edges.

    When no weapon head is present, return the neutral placeholder the
    flat config schema requires (it is never consumed — slot_dims gets
    has_weapon_head=False).
    """
    weapon = spec.head("weapon")
    if weapon is None:
        return ("self_readout", "target_feat")
    return tuple(WEAPON_EDGE_TO_SOURCE[e] for e in weapon.inputs)


def model_config_from_graph(spec: GraphSpec) -> ModelConfig:
    """The flat ModelConfig bridge — policy-layer flags for QNNPolicy/Network."""
    enc = spec.encoder
    weapon = spec.head("weapon")
    decode = dict(weapon.decode) if weapon else {}
    activation = spec.heads[0].activation if spec.heads else "none"
    pointer = spec.pointer

    def d_hidden(name: str) -> int:
        h = spec.head(name)
        return h.d_hidden if h else 0

    return ModelConfig(
        d_model=enc.d_model,
        n_heads=enc.n_heads if enc.type == "transformer" else 1,
        n_layers=enc.n_layers,
        d_ffn=enc.d_ffn,
        attn_dropout=enc.attn_dropout,
        use_gru=spec.temporal is not None,
        d_gru=spec.temporal.d_gru if spec.temporal else 0,
        use_weapon_head=weapon is not None,
        weapon_switch_confidence=float(decode.get("sticky_confidence", 0.0)),
        weapon_switch_margin=float(decode.get("sticky_margin", 0.0)),
        weapon_sources=_weapon_sources(spec),
        weapon_context_from_obs=bool(weapon.context_from_obs) if weapon else False,
        look_bypass_gru=False,
        d_target=pointer.d_target if (pointer and pointer.type == "mlp") else enc.d_model,
        self_weapon_embed_in_self=any(
            "weapon_id" in t.vocab for t in spec.self_tokens
        ),
        d_move=d_hidden("move"),
        d_look=d_hidden("look"),
        d_attack=d_hidden("attack"),
        d_weapon=d_hidden("weapon"),
        head_activation=activation,
    )


def build_network(obs_dim: int, spec: GraphSpec) -> Network:
    """The single model factory: GraphSpec → wired Network."""
    model = model_config_from_graph(spec)
    enc = spec.encoder

    obs_embedding = GraphObsEmbedding(spec)
    if enc.type == "transformer":
        encoder: nn.Module = TransformerEncoder(
            d_model=enc.d_model, n_heads=enc.n_heads, n_layers=enc.n_layers,
            d_ffn=enc.d_ffn, dropout=enc.attn_dropout,
        )
    else:
        encoder = PreAttnEncoder()

    temporal = Temporal(enc.d_model, spec.temporal.d_gru) if spec.temporal else Off

    pointer = spec.pointer
    if pointer is None:
        target_pointer: object = Off
    elif pointer.type == "mlp":
        target_pointer = TargetPointer(d_model=enc.d_model, d_target=pointer.d_target)
    else:
        target_pointer = GTTargetPointer(d_model=enc.d_model)

    dims = slot_dims(
        d_model=enc.d_model,
        d_gru=spec.temporal.d_gru if spec.temporal else 0,
        has_temporal=spec.temporal is not None,
        has_target_pointer=pointer is not None,
        has_weapon_head=spec.head("weapon") is not None,
        weapon_sources=model.weapon_sources,
    )

    def head_or_off(name: str) -> object:
        h = spec.head(name)
        return _build_head(h, dims, enc.d_model) if h else Off

    return Network(
        obs_dim=obs_dim,
        model=model,
        obs_embedding=obs_embedding,
        encoder=encoder,
        temporal=temporal,
        target_pointer=target_pointer,
        move_head=head_or_off("move"),
        look_head=head_or_off("look"),
        attack_head=head_or_off("attack"),
        weapon_head=head_or_off("weapon"),
    )


# ── Legacy flat-config migration ─────────────────────────────────────


def graph_from_model_config(cfg: ModelConfig) -> GraphSpec:
    """Translate a flat v20+ canonical ModelConfig into the equivalent graph.

    Used by the checkpoint loader to migrate legacy flat-meta
    checkpoints. The canonical flat layout is the monolithic-self
    ObsEmbedding + TransformerEncoder + (GRU) + MLP TargetPointer +
    canonical heads; anything else (bench factories, v17 look bypass)
    is out of scope and must load through its own path.
    """
    if cfg.look_bypass_gru:
        raise GraphSpecError(
            "look_bypass_gru (v17 layout) is not expressible as a graph; "
            "load this checkpoint through the legacy flat path"
        )
    has_temporal = bool(cfg.use_gru and cfg.d_gru > 0)
    tokens = (
        monolithic_self_token(cfg.self_weapon_embed_in_self),
        TokenSpec(name="spatial", kind=TOKEN_KIND_SPATIAL),
        TokenSpec(name="entities", kind=TOKEN_KIND_ENTITIES),
    )
    motor_inputs = (EDGE_READOUT, EDGE_TARGET_FEAT)
    heads = [
        HeadNodeSpec(name="move", type="canonical", inputs=motor_inputs,
                     d_hidden=cfg.d_move, activation=cfg.head_activation),
        HeadNodeSpec(name="look", type="canonical", inputs=motor_inputs,
                     d_hidden=cfg.d_look, activation=cfg.head_activation),
        HeadNodeSpec(name="attack", type="canonical", inputs=motor_inputs,
                     d_hidden=cfg.d_attack, activation=cfg.head_activation),
    ]
    if cfg.use_weapon_head:
        # Network silently drops a "gru" source when no temporal slot is
        # active; the graph forbids dangling edges, so drop it here.
        # .get(s, s): unknown sources pass through so spec.validate()
        # reports them as unknown edges (not a KeyError here).
        inputs = tuple(
            WEAPON_SOURCE_TO_EDGE.get(s, s)
            for s in cfg.weapon_sources
            if not (s == "gru" and not has_temporal)
        )
        decode = {}
        if cfg.weapon_switch_confidence or cfg.weapon_switch_margin:
            decode = {
                "sticky_confidence": float(cfg.weapon_switch_confidence),
                "sticky_margin": float(cfg.weapon_switch_margin),
            }
        heads.append(HeadNodeSpec(
            name="weapon", type="canonical", inputs=inputs,
            d_hidden=cfg.d_weapon, activation=cfg.head_activation,
            context_from_obs=cfg.weapon_context_from_obs,
            decode=decode,
        ))
    spec = GraphSpec(
        tokens=tokens,
        encoder=EncoderSpec(
            type="transformer", d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, d_ffn=cfg.d_ffn, attn_dropout=cfg.attn_dropout,
        ),
        temporal=TemporalSpec(type="gru", d_gru=cfg.d_gru) if has_temporal else None,
        pointers=(PointerSpec(name="target", type="mlp", d_target=cfg.d_target),),
        heads=tuple(heads),
    )
    spec.validate()
    return spec
