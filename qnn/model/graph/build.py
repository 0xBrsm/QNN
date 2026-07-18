"""build_network — assemble a Network from a GraphSpec.

The single model factory: BC, bench probes, eval, PPO, and ONNX export
all construct the model here. Node builders self-register from their own
modules into :mod:`qnn.model.node_registry` (one file per node owns its
builder); this module imports those modules for their registrations and
dispatches by spec discriminator — no central type table. Edge widths
come from ``slot_dims`` fed with the spec's resolved node widths;
``Network`` stays the executor of the (fixed) dataflow.

``model_config_from_graph`` bridges a GraphSpec onto the flat
``ModelConfig`` that ``Network`` / ``QNNPolicy`` still consume for
policy-layer flags; ``graph_from_model_config`` is the reverse
migration for legacy flat checkpoints (v20+ canonical layouts — the
v17 ``look_bypass_gru`` layout is not expressible as a graph and keeps
loading through the legacy path).
"""

from __future__ import annotations

import torch.nn as nn

from qnn.model import node_registry as registry
from qnn.model.graph.embedding import GraphObsEmbedding
from qnn.model.graph.spec import (
    EDGE_READOUT, EDGE_TARGET_FEAT,
    WEAPON_EDGE_TO_SOURCE, WEAPON_SOURCE_TO_EDGE,
    _is_token_edge, _token_edge_name,
    _is_scalar_edge, _scalar_edge_name,
    GraphSpec, GraphSpecError, HeadNodeSpec, TokenSpec,
    TOKEN_KIND_ENTITIES, TOKEN_KIND_SPATIAL,
    EncoderSpec, PointerSpec, TemporalSpec,
    monolithic_self_token,
)
from qnn.model.network import ModelConfig, Network, Off, slot_dims

# Importing these node modules registers their builders into node_registry
# (one file per node owns its own builder, beside the class). build_network
# then dispatches by spec discriminator — there is no central type table here.
# The list enumerates which node modules participate; the wiring lives in them.
import qnn.model.move_head            # noqa: F401  move "canonical"
import qnn.model.look_head            # noqa: F401  look "canonical"
import qnn.model.attack_head          # noqa: F401  attack "canonical"
import qnn.model.weapon_head          # noqa: F401  weapon "canonical"
import qnn.model.temporal             # noqa: F401  temporal "gru"
import qnn.model.target               # noqa: F401  pointer "mlp"
import qnn.model.transformer          # noqa: F401  encoder "transformer"
import qnn.model.bench.a24.move_head     # noqa: F401  move "cls"
import qnn.model.bench.a24.look_head     # noqa: F401  look "polar"
import qnn.model.bench.a24.attack_head   # noqa: F401  attack "cls"
import qnn.model.bench.a24.weapon_head   # noqa: F401  weapon "cls" / "cls_prior"
import qnn.model.bench.a25.move_hazard_head  # noqa: F401  move_hazard "canonical"
import qnn.model.bench.a25.move_seg_head     # noqa: F401  move_seg "canonical"
import qnn.model.bench.a25.jump_head         # noqa: F401  jump "canonical"
import qnn.model.bench.a25.attack_with_head  # noqa: F401  weapon "attack_with"
import qnn.model.bench.inputs.preattn_encoder    # noqa: F401  encoder "passthrough"
import qnn.model.bench.inputs.gt_target_pointer  # noqa: F401  pointer "gt"
# Base-graph compositions — each generation registers its own (arch lives with
# the generation, not here). base_graph_dict resolves names from the registry.
import qnn.model.bench.a24.graphs  # noqa: F401  full_4head / full_5head
import qnn.model.bench.a25.graphs  # noqa: F401  full_6head


def _build_head(head: HeadNodeSpec, dims: dict[str, int], d_model: int) -> nn.Module:
    builder = registry.head_builder(head.name, head.type)
    if builder is None:
        raise GraphSpecError(
            f"heads[{head.name!r}]: unknown type {head.type!r}; "
            f"registered: {registry.registered_head_types(head.name)}"
        )
    return builder(head, dims, d_model)


# Legacy ``[head name][type] -> builder`` view, materialized from the registry
# after the node modules above register. Kept for introspection / back-compat
# (re-exported by qnn.model.graph); the registry is the source of truth.
HEAD_TYPES: dict[str, dict[str, object]] = registry.head_type_table()


def _weapon_sources(spec: GraphSpec) -> tuple[str, ...]:
    """ModelConfig.weapon_sources from the weapon head's edges.

    When no weapon head is present, return the neutral placeholder the
    flat config schema requires (it is never consumed — slot_dims gets
    has_weapon_head=False).
    """
    weapon = spec.head("weapon")
    if weapon is None:
        return ("self_readout", "target_feat")

    def _edge_to_source(edge: str) -> str:
        # token.<name> → token:<name> (read that self-token as the readout);
        # scalar.<name> → scalar:<name> (raw obs scalar straight into the cat);
        # otherwise the fixed gru/self_readout/target_feat mapping.
        if _is_token_edge(edge):
            return "token:" + _token_edge_name(edge)
        if _is_scalar_edge(edge):
            return "scalar:" + _scalar_edge_name(edge)
        return WEAPON_EDGE_TO_SOURCE[edge]

    return tuple(_edge_to_source(e) for e in weapon.inputs)


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
    enc_builder = registry.encoder_builder(enc.type)
    if enc_builder is None:
        raise GraphSpecError(
            f"encoder: unknown type {enc.type!r}; "
            f"registered: {registry.registered_encoder_types()}")
    encoder: nn.Module = enc_builder(enc)

    if spec.temporal is None:
        temporal: object = Off
    else:
        temporal_builder = registry.temporal_builder(spec.temporal.type)
        if temporal_builder is None:
            raise GraphSpecError(
                f"temporal: unknown type {spec.temporal.type!r}; "
                f"registered: {registry.registered_temporal_types()}")
        temporal = temporal_builder(spec.temporal, enc.d_model)

    pointer = spec.pointer
    if pointer is None:
        target_pointer: object = Off
    else:
        pointer_builder = registry.pointer_builder(pointer.type)
        if pointer_builder is None:
            raise GraphSpecError(
                f"pointers[{pointer.name!r}]: unknown type {pointer.type!r}; "
                f"registered: {registry.registered_pointer_types()}")
        target_pointer = pointer_builder(pointer, enc.d_model)

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
        move_hazard_head=head_or_off("move_hazard"),
        move_seg_head=head_or_off("move_seg"),
        jump_head=head_or_off("jump"),
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
