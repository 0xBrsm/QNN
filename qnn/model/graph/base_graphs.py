"""Base-graph compositions promoted from a25 — registered into the node registry.

full_6head (full_5head + the retired move_hazard WHEN head, kept for reload)
and full_movearch — the promoted move-arch consolidation shape (two-seed p3c
pass, 2026-07-10): per-tick move head dropped, jump head (bit7-gated 2-class),
move_seg with the water-ud swim axis, attack_with 9-way in the weapon slot,
arsenal ammo-pool token. Importing this module registers it so
``qnn.model.graph.base_graph_dict("full_6head")`` resolves it;
``qnn.model.graph.build`` imports it in its node-registration bootstrap.

Promoted 2026-07-27 out of ``qnn.model.bench.a25.graphs`` (both base graphs
outlived the generation — full_movearch is the current base — so cross-gen
code importing this module is no longer a bench-isolation violation). The
JSON bodies live alongside it in ``bases/`` (packaged via
``[tool.setuptools.package-data]`` "qnn.model.graph" = ["bases/*.json"]).
"""
from __future__ import annotations

import json
from importlib import resources

from qnn.model.node_registry import register_base_graph

_BASES = ("full_6head", "full_movearch")

for _name in _BASES:
    _text = (resources.files("qnn.model.graph") / "bases" / f"{_name}.json").read_text(
        encoding="utf-8")
    register_base_graph(_name, json.loads(_text))
