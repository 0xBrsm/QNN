"""build_network — assemble a Network from a GraphSpec.

The single model factory: BC, bench probes, eval, PPO, and ONNX export
all construct the model here. Node builders self-register from their own
modules into :mod:`qnn.model.node_registry` (one file per node owns its
builder); this module imports those modules for their registrations and
dispatches by spec discriminator — no central type table. Edge widths
come from ``slot_dims`` fed with the spec's resolved node widths;
``Network`` stays the executor of the (fixed) dataflow.

a28: graphs are the ONLY way to build a model. There is no legacy
flat-``ModelConfig`` migration and no pre-a28 node support — pre-a28
checkpoints load from their own branches.
"""

from __future__ import annotations

import torch.nn as nn

from qnn.model import node_registry as registry
from qnn.model.graph.embedding import GraphObsEmbedding
from qnn.model.graph.spec import (
    AIM_DIM,
    AIM2_DIM,
    EDGE_AIM,
    EDGE_AIM2,
    INTENT_DIM,
    WEAPON_EDGE_TO_SOURCE,
    GraphSpec, GraphSpecError, HeadNodeSpec,
)
from qnn.model.network import ModelConfig, Network, Off, slot_dims

# Importing these node modules registers their builders into node_registry
# (one file per node owns its own builder, beside the class). build_network
# then dispatches by spec discriminator — there is no central type table here.
# The list enumerates which node modules participate; the wiring lives in them.
import qnn.model.look_head            # noqa: F401  look "polar"
import qnn.model.look_head_xm         # noqa: F401  look "xm_tangent"
import qnn.model.temporal             # noqa: F401  temporal "gru"
import qnn.model.target               # noqa: F401  pointer "mlp"
import qnn.model.transformer          # noqa: F401  encoder "transformer"
import qnn.model.move_seg_head     # noqa: F401  move_seg "canonical"
import qnn.model.look_seg_head     # noqa: F401  look_seg "canonical"
import qnn.model.jump_head         # noqa: F401  jump "canonical"
import qnn.model.attack_with_head  # noqa: F401  attack "attack_with"
import qnn.model.attack_future_head  # noqa: F401  attack_future "canonical"
# BENCH head slot: the revived per-tick move head (cell C3 of
# agents/plans/seg-vs-frame-decision.md). Never in bases/core.json.
import qnn.model.move_tick_head  # noqa: F401  move_tick "per_tick"
# Bench input mechanisms (current-gen probe machinery, not retired arch).
import qnn.model.bench.inputs.preattn_encoder    # noqa: F401  encoder "passthrough"
import qnn.model.bench.inputs.gt_target_pointer  # noqa: F401  pointer "gt"
# The canonical base graph — base_graph_dict resolves names from the registry.
import qnn.model.graph.base_graphs  # noqa: F401  core


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
    """ModelConfig selector sources from the attack selector's edges.

    Empty when no selector head is present (never consumed — the
    selector cat is only assembled when the attack slot is live).
    ``EDGE_AIM`` / ``EDGE_AIM2`` are NOT weapon sources: they are computed
    tail blocks, not node outputs, and Network appends them after these
    sources (spec.aim_edge / spec.aim2_edge carry them instead).
    """
    selector = spec.head("attack")
    if selector is None or selector.type != "attack_with":
        return ()
    return tuple(
        WEAPON_EDGE_TO_SOURCE[e] for e in selector.inputs
        if e not in (EDGE_AIM, EDGE_AIM2)
    )


def model_config_from_graph(spec: GraphSpec) -> ModelConfig:
    """The flat ModelConfig bridge — policy-layer scalars for QNNPolicy/Network."""
    enc = spec.encoder
    activation = spec.heads[0].activation if spec.heads else "none"
    pointer = spec.pointer

    return ModelConfig(
        d_model=enc.d_model,
        n_heads=enc.n_heads if enc.type == "transformer" else 1,
        n_layers=enc.n_layers,
        d_ffn=enc.d_ffn,
        attn_dropout=enc.attn_dropout,
        use_gru=spec.temporal is not None,
        d_gru=spec.temporal.d_gru if spec.temporal else 0,
        weapon_sources=_weapon_sources(spec),
        d_target=pointer.d_target if (pointer and pointer.type == "mlp") else enc.d_model,
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
        weapon_sources=model.weapon_sources,
        intent_dim=INTENT_DIM if spec.intent is not None else 0,
        aim_dim=(AIM2_DIM if spec.aim2_edge else AIM_DIM if spec.aim_edge else 0),
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
        look_head=head_or_off("look"),
        attack_head=head_or_off("attack"),
        move_seg_head=head_or_off("move_seg"),
        look_seg_head=head_or_off("look_seg"),
        jump_head=head_or_off("jump"),
        intent_source=spec.intent.source if spec.intent is not None else None,
        aim_edge=spec.aim_edge,
        aim2_edge=spec.aim2_edge,
        # LAST, deliberately: Network assigns modules in argument order and
        # _init_weights walks self.modules() in registration order, so keeping
        # the aux head last leaves every other module's xavier draw
        # bit-identical to the control arm's. (The head's builder also restores
        # the RNG across its own constructor draw — see network.py and
        # qnn.model.attack_future_head.)
        attack_future_head=head_or_off("attack_future"),
        # LAST for the same reason (see network.py): the bench per-tick move
        # head must not shift any other module's init draw.
        move_tick_head=head_or_off("move_tick"),
    )
