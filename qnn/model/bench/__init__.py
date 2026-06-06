"""Component-ablation library — uses the canonical BC training pipeline.

Per-component ablations are alternate Network configurations plugged into
the BC trainer via ``QNNPolicy.__init__(..., model_factory=...)``. Each
probe builds a ``Network`` with the right slot overrides (encoder /
temporal / target_pointer / move_head / look_head / attack_head /
weapon_head) and returns it from its ``HeadSpec.build`` factory; the
canonical BC supervised loop drives it unchanged.

The runner translates a head_probe run-dir into a ``BCConfig`` + model
factory + head_loss_weights and hands it to ``run_behavior_cloning``.
Shard loading, segment_mask filtering, fingerprint verification, episode
shuffling, BPTT batching, eval cadence, checkpointing, and
``bc_history.json`` are all the canonical BC implementations.

Layout:

  spec.py                    HeadSpec / HeadLossSpec / HeadBuilder dataclasses
  features.py                FeatureBuilder registry (used by flat-feature probes)
  heads.py                   HEADS dict — central registry of per-probe specs
  target.py                  target_soft_ce_loss + target_metrics + TARGET HeadSpec
  attack.py                  attack_bce_loss + attack_metrics + ATTACK HeadSpec (flat features)
  flat.py                    FlatFeatureHead (generic MLP over named per-frame features)
  attack_preattn.py          PreAttn encoder + GT pointer + canonical AttackHead
  weapon_preattn.py          PreAttn encoder + GT pointer + canonical WeaponHead
  attack_preattn_oracle.py   PreAttn encoder + GT pointer + OracleAttackHead
  loss_shaping.py            Gaussian-shouldered BCE helpers (canonical loss path)
  runner.py                  run-dir / qnn.run.router entry point
  templates/                 run.json + train.json + machine.json + probe.json + run.md

Slot-configurable Network components (PreAttnEncoder, GTTargetPointer,
Off sentinel) live under ``qnn.model``; this package only holds the
probe configurations and the loss/metrics specs.
"""

from qnn.model.bench.features import FEATURE_REGISTRY, FeatureBuilder, register_feature
from qnn.model.bench.flat import FlatFeatureHead
from qnn.model.bench.heads import HEADS
from qnn.model.bench.spec import (
    HeadBuilder,
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureBuilder",
    "FlatFeatureHead",
    "HEADS",
    "HeadBuildResult",
    "HeadBuilder",
    "HeadLossSpec",
    "HeadSpec",
    "neutral_model_config",
    "register_feature",
]
