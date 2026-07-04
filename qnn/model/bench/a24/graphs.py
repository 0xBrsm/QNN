"""a24 base-graph compositions — registered into the node registry.

The a24 arch lives with the generation (these ``bases/*.json`` files), not in
the generation-agnostic graph package. Importing this module registers the a24
base graphs so ``qnn.model.graph.base_graph_dict("full_5head")`` resolves them;
``qnn.model.graph.build`` imports it in its node-registration bootstrap.
"""
from __future__ import annotations

import json
from importlib import resources

from qnn.model.node_registry import register_base_graph

_BASES = ("full_4head", "full_5head")

for _name in _BASES:
    _text = (resources.files("qnn.model.bench.a24") / "bases" / f"{_name}.json").read_text(
        encoding="utf-8")
    register_base_graph(_name, json.loads(_text))
