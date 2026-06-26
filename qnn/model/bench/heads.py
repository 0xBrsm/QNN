"""Head registry — legacy-checkpoint-reload only.

``HEADS`` exists so retained run-dirs (``runs/head_probe/*``) can reload
their checkpoints through ``HEADS[head].build(probe)``. It is frozen to
the promoted full-model assemblies; do NOT add new entries. New probes
are graph deltas — a ``probe.json`` names a base graph plus overrides
(see ``qnn.model.graph`` and ``runner.py``).
"""

from __future__ import annotations

from qnn.model.bench.full_4head import FULL_4HEAD
from qnn.model.bench.full_5head import FULL_5HEAD
from qnn.model.bench.full_multitrunk import FULL_MULTITRUNK
from qnn.model.bench.spec import HeadSpec

HEADS: dict[str, HeadSpec] = {
    FULL_4HEAD.name:      FULL_4HEAD,
    FULL_5HEAD.name:      FULL_5HEAD,
    FULL_MULTITRUNK.name: FULL_MULTITRUNK,
}
