"""HeadSpec dataclass — shared types for per-head modules.

Each kept assembly file (``qnn.model.bench.full_4head``,
``qnn.model.bench.full_5head``, ``qnn.model.bench.full_multitrunk``)
constructs its own ``HeadSpec`` and
registers it in ``qnn.model.bench.heads.HEADS``. The runner in
``qnn.model.bench.runner`` reads the spec to wire the head into the
canonical BC training pipeline via ``QNNPolicy``'s ``model_factory``
hook — head probes share BC's shard loader, supervised loop,
checkpointing, and history. No parallel pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn

from qnn.model.network import ModelConfig


# A head builder takes the probe-side knobs (everything from probe.json)
# and returns the two things the canonical BC trainer needs to drive a
# probe model: a ``ModelConfig`` (used to populate QNNPolicy's policy-
# layer flags) and an ``nn.Module`` factory for QNNPolicy's
# ``model_factory`` hook. ``head_loss_weights`` is *not* part of this
# return — it lives in ``train.json`` like every other BC training knob,
# so each probe run-dir owns the loss-weight choice.
HeadBuildResult = tuple[
    ModelConfig,
    Callable[[int, ModelConfig], nn.Module],
]
HeadBuilder = Callable[[Mapping[str, Any]], HeadBuildResult]


def neutral_model_config(
    *,
    d_model: int,
    self_weapon_embed_in_self: bool,
) -> ModelConfig:
    """A ``ModelConfig`` with every encoder/GRU/head BC flag disabled.

    Head probes call ``QNNPolicy`` via ``model_factory=...``, which
    means ``Network.__init__`` — the only consumer of
    encoder/GRU dimensions — is never invoked. The policy layer still
    reads policy-relevant flags (``use_gru``, ``use_weapon_head``, …),
    so those need to be False for the probe to make sense. Numeric
    fields (``d_ffn``, ``n_layers``, …) are inert under the factory
    path and set to safe placeholders.

    ``d_model`` and ``self_weapon_embed_in_self`` are required so each
    head's build callable (which reads probe.json) is the only place
    those values come from — no implicit defaults in this helper.
    """
    return ModelConfig(
        d_model=int(d_model),
        n_heads=1,
        n_layers=0,
        d_ffn=0,
        attn_dropout=0.0,
        use_gru=False,
        d_gru=0,
        use_weapon_head=False,
        weapon_switch_confidence=0.0,
        weapon_switch_margin=0.0,
        weapon_sources=("self_readout", "target_feat"),
        weapon_context_from_obs=False,
        look_bypass_gru=False,
        d_target=int(d_model),
        self_weapon_embed_in_self=bool(self_weapon_embed_in_self),
        d_move=0,
        d_look=0,
        d_attack=0,
        d_weapon=0,
        head_activation="none",
    )


@dataclass(frozen=True, slots=True)
class HeadSpec:
    """One head's complete probe configuration.

    ``build`` is the per-head factory that translates probe.json into
    the (model_config, model_factory) pair the runner feeds to
    ``run_behavior_cloning``. All losses/metrics are computed by
    ``QNNPolicy._compute_head_losses_and_metrics`` — the spec carries
    no loss plumbing. Each head module owns its own ``build`` so a
    probe is a single file + one entry in ``HEADS``.
    """
    name: str
    build: HeadBuilder
