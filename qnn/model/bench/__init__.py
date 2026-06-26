"""Component-ablation library — uses the canonical BC training pipeline.

Per-component ablations are alternate Network configurations plugged into
the BC trainer via ``QNNPolicy.__init__(..., model_factory=...)``. New
probes are graph deltas: ``probe.json`` names a base graph plus overrides
(see ``qnn.model.graph``), and the runner builds the model with
``build_network``. The legacy ``HEADS`` registry is retained only so old
run-dirs can reload their checkpoints — it is frozen to the promoted
full-model assemblies (full_4head / full_5head / full_multitrunk).

The runner translates a head_probe run-dir into a ``BCConfig`` + model
factory + head_loss_weights and hands it to ``run_behavior_cloning``.
Shard loading, segment_mask filtering, fingerprint verification, episode
shuffling, BPTT batching, eval cadence, checkpointing, and
``bc_history.json`` are all the canonical BC implementations.

Layout:

  spec.py                    HeadSpec / HeadBuilder dataclasses
  heads.py                   HEADS dict — frozen legacy-reload registry
  full_4head.py              promoted 4-head full-model assembly
  full_5head.py              full_4head + canonical TargetPointer (5th head)
  full_multitrunk.py         per-head trunk variant of the full assembly
  inputs/                    slot components used by the kept assemblies
  side_channels.py           label-derived forward-scoped contexts
  runner.py                  run-dir / qnn.run.router entry point
  templates/                 run.json + train.json + machine.json + probe.json + run.md

Concluded single-head experiments were removed; their findings live in
``src/docs`` (see model-graph.md "Legacy Ablation Paths").
"""

from qnn.model.bench.heads import HEADS
from qnn.model.bench.spec import (
    HeadBuilder,
    HeadBuildResult,
    HeadSpec,
    neutral_model_config,
)

__all__ = [
    "HEADS",
    "HeadBuildResult",
    "HeadBuilder",
    "HeadSpec",
    "neutral_model_config",
]
