"""qnn.model.graph — declarative model assembly.

See ``src/docs/model-graph.md``. The spec is pure data
(:class:`GraphSpec`), the builder (:func:`build_network`) is the single
model factory shared by BC, bench probes, eval, PPO, and export.

The canonical base graph is ``core`` (:mod:`qnn.model.graph.base_graphs`);
bench probes are overrides merged onto a base via :func:`merge_overrides`.
a28: there is no legacy flat-config migration — pre-a28 checkpoints load
from their own branches.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from qnn.model.graph.build import (
    HEAD_TYPES,
    build_network,
    model_config_from_graph,
)
from qnn.model.graph.embedding import GraphObsEmbedding
from qnn.model.graph.spec import GraphSpec, GraphSpecError

__all__ = [
    "GraphSpec", "GraphSpecError", "GraphObsEmbedding", "HEAD_TYPES",
    "build_network", "model_config_from_graph",
    "base_graph_dict", "load_base_graph", "merge_overrides",
]


def base_graph_dict(name: str) -> dict[str, Any]:
    """Raw dict of a registered base graph (a fresh copy, safe to mutate).

    Base graphs are registered by each generation's ``graphs`` module (imported
    by ``qnn.model.graph.build``'s bootstrap), so importing this package has
    populated the registry by the time this is called."""
    from qnn.model import node_registry
    graph = node_registry.base_graph(name)
    if graph is None:
        raise GraphSpecError(
            f"unknown base graph {name!r}; available: "
            f"{node_registry.registered_base_graphs()}")
    return json.loads(json.dumps(graph))  # deep copy — callers merge/mutate


def load_base_graph(name: str) -> GraphSpec:
    return GraphSpec.from_dict(base_graph_dict(name))


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto a base graph dict.

    Dicts merge recursively; an explicit ``null`` deletes the key (drop a
    token / head / pointer / the temporal node); scalars and lists
    replace wholesale. Returns a new dict; inputs are not mutated.

    A ``null`` for a key the base does not have raises: a typo'd delete
    (``"temproal": null``) would otherwise no-op silently and train the
    un-ablated model — recording a false null result.
    """
    out: dict[str, Any] = {k: v for k, v in base.items()}
    for key, value in overrides.items():
        if value is None:
            if key not in out:
                raise GraphSpecError(
                    f"override deletes unknown key {key!r} (not in base; "
                    f"base keys: {sorted(out)})"
                )
            del out[key]
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_overrides(out[key], value)
        else:
            # Wholesale replacement: nulls nested inside it have no delete
            # semantics (there is no base subtree to delete from) — a
            # literal None would survive into the merged graph and explode
            # deep inside spec parsing instead of failing loud here.
            _reject_nested_null(value, key)
            out[key] = value
    return out


def _reject_nested_null(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if v is None:
                raise GraphSpecError(
                    f"override sets {path}.{k} to null inside a replaced "
                    f"subtree — null-deletes only apply where the base has "
                    f"a matching mapping"
                )
            _reject_nested_null(v, f"{path}.{k}")
