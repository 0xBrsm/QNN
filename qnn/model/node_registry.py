"""Node-builder registry — one file per node self-registers its builder.

Decentralizes the node→constructor mapping that ``build.py`` used to hold as a
central ``HEAD_TYPES`` table plus an inline if/else for the encoder / pointer /
temporal slots. Each node module declares its own builder *beside the class it
builds* via the ``register_*`` decorators here; :func:`build_network` dispatches
through the registry. Adding a node type is then a single new file plus its
decorator — no edit to ``build.py``.

Builder signatures (stable contracts the build dispatch calls with):

* head     — ``(head_spec, dims, d_model) -> nn.Module``  where ``dims`` is the
             :func:`qnn.model.network.slot_dims` contract.
* encoder  — ``(encoder_spec) -> nn.Module``
* pointer  — ``(pointer_spec, d_model) -> nn.Module``
* temporal — ``(temporal_spec, d_model) -> nn.Module``

Leaf module, and deliberately *not* under :mod:`qnn.model.graph`: it imports
nothing from :mod:`qnn` at runtime, and a node module reaching it
(``from qnn.model.node_registry import register_head``) must not run the graph
package ``__init__`` (which imports ``build``, which imports the node modules —
a cycle). Hosting it directly under :mod:`qnn.model` (a namespace package, no
heavy ``__init__``) keeps that import path cheap and acyclic. Spec types are
referenced only under ``TYPE_CHECKING``. Lookup misses return ``None`` —
``build.py`` owns the ``GraphSpecError`` wording so the spec layer keeps a
single voice for invalid graphs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qnn.model.graph.spec import (
        EncoderSpec, HeadNodeSpec, PointerSpec, TemporalSpec,
    )

HeadBuilder = Callable[["HeadNodeSpec", "dict[str, int]", int], nn.Module]
EncoderBuilder = Callable[["EncoderSpec"], nn.Module]
PointerBuilder = Callable[["PointerSpec", int], nn.Module]
TemporalBuilder = Callable[["TemporalSpec", int], nn.Module]

# Builders, keyed by the spec's discriminators. Heads are keyed by
# (head name, type) because the same type token (e.g. "canonical") means a
# different module per head; the other node kinds are keyed by type alone.
_HEADS: dict[tuple[str, str], HeadBuilder] = {}
_ENCODERS: dict[str, EncoderBuilder] = {}
_POINTERS: dict[str, PointerBuilder] = {}
_TEMPORALS: dict[str, TemporalBuilder] = {}

# Named base-graph compositions (raw spec dicts), registered by each generation
# (e.g. qnn.model.bench.a24.graphs registers "full_5head"). The arch composition
# lives with the generation, not in the generation-agnostic graph package.
_BASE_GRAPHS: dict[str, dict] = {}


def _register(table: dict, key, fn):
    if key in table and table[key] is not fn:
        raise RuntimeError(f"duplicate node builder for {key!r} in {table}")
    table[key] = fn
    return fn


def register_head(head_name: str, type_name: str):
    """Register a head builder for ``heads[<head_name>].type == <type_name>``."""
    return lambda fn: _register(_HEADS, (head_name, type_name), fn)


def register_encoder(type_name: str):
    return lambda fn: _register(_ENCODERS, type_name, fn)


def register_pointer(type_name: str):
    return lambda fn: _register(_POINTERS, type_name, fn)


def register_temporal(type_name: str):
    return lambda fn: _register(_TEMPORALS, type_name, fn)


def register_base_graph(name: str, graph: dict) -> None:
    """Register a named base-graph composition (raw spec dict).

    Called by each generation's ``graphs`` module so the arch composition is
    owned by the generation (e.g. ``qnn.model.bench.a24.graphs``), not the
    graph machinery."""
    if name in _BASE_GRAPHS and _BASE_GRAPHS[name] != graph:
        raise RuntimeError(f"conflicting base graph registered for {name!r}")
    _BASE_GRAPHS[name] = graph


# -- lookups (None on miss; caller raises the spec-level error) -----------

def head_builder(head_name: str, type_name: str) -> "HeadBuilder | None":
    return _HEADS.get((head_name, type_name))


def encoder_builder(type_name: str) -> "EncoderBuilder | None":
    return _ENCODERS.get(type_name)


def pointer_builder(type_name: str) -> "PointerBuilder | None":
    return _POINTERS.get(type_name)


def temporal_builder(type_name: str) -> "TemporalBuilder | None":
    return _TEMPORALS.get(type_name)


def registered_head_types(head_name: str) -> list[str]:
    return sorted(t for (n, t) in _HEADS if n == head_name)


def registered_encoder_types() -> list[str]:
    return sorted(_ENCODERS)


def registered_pointer_types() -> list[str]:
    return sorted(_POINTERS)


def registered_temporal_types() -> list[str]:
    return sorted(_TEMPORALS)


def base_graph(name: str) -> "dict | None":
    return _BASE_GRAPHS.get(name)


def registered_base_graphs() -> list[str]:
    return sorted(_BASE_GRAPHS)


def head_type_table() -> dict[str, dict[str, "HeadBuilder"]]:
    """Materialize the legacy ``[head name][type] -> builder`` view (for
    introspection / back-compat; the registry is the source of truth)."""
    out: dict[str, dict[str, HeadBuilder]] = {}
    for (name, type_name), fn in _HEADS.items():
        out.setdefault(name, {})[type_name] = fn
    return out
