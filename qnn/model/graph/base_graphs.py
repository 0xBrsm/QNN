"""The canonical base graph — registered into the node registry.

``core`` IS the canonical a28 architecture: six tokens → transformer
(d64/2L/2H/ffn256) → GRU (d64) + MLP target pointer → four heads (look
polar, attack_with selector, move_seg +water_ud, jump), every input edge
explicit (``gru`` + ``target.feat``), no weapon_ctx, no implicit inputs.
Plan: agents/plans/a28-core-graph.md.

Bench probes are overrides merged onto this base (``qnn.model.graph
.merge_overrides``); a bench arm that outlives its generation gets
promoted INTO this base, not registered beside it. Importing this module
registers it so ``qnn.model.graph.base_graph_dict("core")`` resolves;
``qnn.model.graph.build`` imports it in its node-registration bootstrap.
The JSON body lives in ``bases/`` (packaged via
``[tool.setuptools.package-data]`` "qnn.model.graph" = ["bases/*.json"]).
"""
from __future__ import annotations

import json
from importlib import resources

from qnn.model.node_registry import register_base_graph

_BASES = ("core",)

for _name in _BASES:
    _text = (resources.files("qnn.model.graph") / "bases" / f"{_name}.json").read_text(
        encoding="utf-8")
    register_base_graph(_name, json.loads(_text))
