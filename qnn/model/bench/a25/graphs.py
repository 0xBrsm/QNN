"""a25 base-graph compositions — registered into the node registry.

The a25 arch bases live with the generation: full_6head (full_5head + the
retired move_hazard WHEN head, kept for reload) and full_movearch — the
promoted move-arch consolidation shape (two-seed p3c pass, 2026-07-10):
per-tick move head dropped, jump head (bit7-gated 2-class), move_seg with
the water-ud swim axis, attack_with 9-way in the weapon slot, arsenal
ammo-pool token. Importing this module registers it so
``qnn.model.graph.base_graph_dict("full_6head")`` resolves it;
``qnn.model.graph.build`` imports it in its node-registration bootstrap.
"""
from __future__ import annotations

import json
from importlib import resources

from qnn.model.node_registry import register_base_graph

_BASES = ("full_6head", "full_movearch")

for _name in _BASES:
    _text = (resources.files("qnn.model.bench.a25") / "bases" / f"{_name}.json").read_text(
        encoding="utf-8")
    register_base_graph(_name, json.loads(_text))
