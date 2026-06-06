"""Per-head ablation library — uses the canonical BC training pipeline.

Head probes are alternate ``nn.Module``s plugged into the BC trainer via
``QNNPolicy.__init__(..., model_factory=...)``. The runner here just
translates a head_probe run-dir into a ``BCConfig`` + model factory +
head_loss_weights and hands it to ``run_behavior_cloning``. Shard
loading, segment_mask filtering, fingerprint verification, episode
shuffling, BPTT batching, eval cadence, checkpointing, and
``bc_history.json`` are all the canonical BC implementations — no
parallel pipeline.

Layout:

  spec.py        HeadSpec / HeadLossSpec / HeadBuilder dataclasses
  features.py    FeatureBuilder registry (still used by flat-feature heads)
  heads.py       HEADS dict — central registry of per-head specs
  target.py      target_soft_ce_loss + target_metrics + TARGET HeadSpec (flat — Step 4)
  fire.py        fire_bce_loss + fire_metrics + FIRE HeadSpec     (flat — Step 4)
  fire_token.py  FireTokenHead + FIRE_TOKEN HeadSpec (trunk-projected tokens)
  tokenized.py   Shared trunk-projected encoder (Tokenizer + GT soft-pool)
  runner.py      run-dir / qnn.run.router entry point
  templates/     run.json + train.json + machine.json + probe.json + run.md

Single entry point — versioned + frozen configs + fingerprint enforced::

    python -m qnn.run.init --mode head_probe --name <run> \\
        [--probe path/to/custom_probe.json]
    docker compose -f src/docker/compose.yaml run --rm trainer \\
        agents/skills/train/scripts/train.sh runs/head_probe/<run>

There is no ad-hoc CLI by design — every probe gets a run-dir so the
result is reproducible and git-archived. Iteration on a head's spec
happens by editing the per-head file and committing. To probe a
non-default feature set, override probe.json at init time with
``--probe``.
"""

from qnn.bc.heads.features import FeatureBuilder, register_feature, FEATURE_REGISTRY
from qnn.bc.heads.fire_token import FireTokenHead
from qnn.bc.heads.flat import FlatFeatureHead
from qnn.bc.heads.heads import HEADS
from qnn.bc.heads.spec import (
    HeadBuilder,
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.bc.heads.tokenized import TokenizedFeatureEncoder

__all__ = [
    "FeatureBuilder", "register_feature", "FEATURE_REGISTRY",
    "HeadSpec", "HeadLossSpec", "HeadBuilder", "HeadBuildResult", "HEADS",
    "neutral_model_config",
    "TokenizedFeatureEncoder",
    "FireTokenHead", "FlatFeatureHead",
]
