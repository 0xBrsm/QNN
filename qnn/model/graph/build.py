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
import qnn.model.move_hazard_head  # noqa: F401  move_hazard "canonical"
import qnn.model.move_seg_head     # noqa: F401  move_seg "canonical"
import qnn.model.look_seg_head     # noqa: F401  look_seg "canonical"
import qnn.model.jump_head         # noqa: F401  jump "canonical"
import qnn.model.attack_with_head  # noqa: F401  weapon "attack_with"
import qnn.model.attack_future_head  # noqa: F401  attack_future "canonical"
# a24 is a retired arch kept ONLY for legacy-checkpoint reload — its node
# types and base graphs (full_4head/full_5head) stay bench-scoped in
# qnn.model.bench.a24, never imported from cross-gen code beyond this
# registration bootstrap.
import qnn.model.bench.a24.move_head     # noqa: F401  move "cls"
import qnn.model.bench.a24.look_head     # noqa: F401  look "polar"
import qnn.model.bench.a24.attack_head   # noqa: F401  attack "cls"
import qnn.model.bench.a24.weapon_head   # noqa: F401  weapon "cls" / "cls_prior"
import qnn.model.bench.a24.graphs  # noqa: F401  full_4head / full_5head
import qnn.model.bench.inputs.preattn_encoder    # noqa: F401  encoder "passthrough"
import qnn.model.bench.inputs.gt_target_pointer  # noqa: F401  pointer "gt"
# Base-graph compositions — full_6head/full_movearch outlived a25 and are
# promoted (qnn.model.graph.base_graphs); a24's retired-arch bases stay
# bench-scoped above. base_graph_dict resolves names from the registry.
import qnn.model.graph.base_graphs  # noqa: F401  full_6head / full_movearch


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
    """ModelConfig selector sources from the categorical attack head's edges.

    When no weapon head is present, return the neutral placeholder the
    flat config schema requires (it is never consumed — slot_dims gets
    has_weapon_head=False).
    """
    selector = spec.head("attack")
    if selector is None or selector.type != "attack_with":
        selector = spec.head("weapon")
    if selector is None:
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

    return tuple(_edge_to_source(e) for e in selector.inputs)


def model_config_from_graph(spec: GraphSpec) -> ModelConfig:
    """The flat ModelConfig bridge — policy-layer flags for QNNPolicy/Network."""
    enc = spec.encoder
    attack = spec.head("attack")
    selector = attack if attack is not None and attack.type == "attack_with" else spec.head("weapon")
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
        use_weapon_head=selector is not None,
        weapon_sources=_weapon_sources(spec),
        look_bypass_gru=False,
        d_target=pointer.d_target if (pointer and pointer.type == "mlp") else enc.d_model,
        d_move=d_hidden("move"),
        d_look=d_hidden("look"),
        d_attack=d_hidden("attack"),
        d_weapon=selector.d_hidden if selector else 0,
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
        has_weapon_head=(spec.head("attack") is not None and spec.head("attack").type == "attack_with")
                        or spec.head("weapon") is not None,
        weapon_sources=model.weapon_sources,
    )

    def head_or_off(name: str) -> object:
        h = spec.head(name)
        return _build_head(h, dims, enc.d_model) if h else Off

    attack_spec = spec.head("attack")
    attack_is_selector = attack_spec is not None and attack_spec.type == "attack_with"
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
        weapon_head=Off if attack_is_selector else head_or_off("weapon"),
        move_hazard_head=head_or_off("move_hazard"),
        move_seg_head=head_or_off("move_seg"),
        look_seg_head=head_or_off("look_seg"),
        jump_head=head_or_off("jump"),
        # LAST, deliberately: Network assigns modules in argument order and
        # _init_weights walks self.modules() in registration order, so keeping
        # the aux head last leaves every other module's xavier draw
        # bit-identical to the control arm's. (The head's builder also restores
        # the RNG across its own constructor draw — see network.py and
        # qnn.model.attack_future_head.)
        attack_future_head=head_or_off("attack_future"),
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
        monolithic_self_token(),
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
        heads.append(HeadNodeSpec(
            name="weapon", type="canonical", inputs=inputs,
            d_hidden=cfg.d_weapon, activation=cfg.head_activation,
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
