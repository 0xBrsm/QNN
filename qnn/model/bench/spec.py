"""HeadSpec / HeadLossSpec dataclasses — shared types for per-head modules.

Each per-head file (``qnn.model.bench.target``, ``qnn.model.bench.attack``,
``qnn.model.bench.attack_preattn``, …) constructs its own ``HeadSpec`` and
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

import torch
import torch.nn as nn

from qnn.model.network import ModelConfig


@dataclass(frozen=True, slots=True)
class HeadLossSpec:
    """Loss + metrics bundle — same shape regardless of model architecture.

    ``loss_fn(logits, labels) -> scalar`` — head-specific loss
    (BCE for fire, soft-CE for target, multi-axis CE for move, …).
    ``metrics_fn(logits, labels) -> dict[str, float]`` — returns
    metrics under the same key names BC training uses
    (``target_kl``, ``f1_attack`` …) so probe JSON diffs cleanly
    against BC ``bc_history.json``.
    """
    loss_fn: Callable[..., torch.Tensor]
    metrics_fn: Callable[..., dict[str, float]]
    label_key: str           # which array in the loaded shard dict is the label
    output_dim: int          # MLP output width
    selection_metric: str    # which metric drives best-epoch tracking
    selection_lower_is_better: bool = True


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
        weapon_use_gru=False,
        weapon_use_self_readout=True,
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
    the (model_config, model_factory, head_loss_weights) triple the
    runner feeds to ``run_behavior_cloning``. Each head module owns its
    own ``build`` so adding a new probe is a single file + one entry in
    ``HEADS``.

    ``default_features`` is a legacy hook retained for the flat-feature
    heads (``fire`` / ``target``) so their builders can look up the
    default per-frame feature list when probe.json doesn't override it.
    Pre-attention heads (``attack_preattn``) leave it empty.
    """
    name: str
    loss: HeadLossSpec
    build: HeadBuilder
    default_features: tuple[str, ...] = ()
