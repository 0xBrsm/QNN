"""HeadSpec / HeadLossSpec dataclasses — shared types for per-head modules.

Each per-head file (``qnn.bc.heads.target``, ``qnn.bc.heads.fire``,
``qnn.bc.heads.fire_token``, …) constructs its own ``HeadSpec`` and
registers it in ``qnn.bc.heads.heads.HEADS``. The runner in
``qnn.bc.heads.runner`` reads the spec to wire the head into the
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

from qnn.model.policy import ModelConfig


@dataclass(frozen=True, slots=True)
class HeadLossSpec:
    """Loss + metrics bundle — same shape regardless of model architecture.

    ``loss_fn(logits, labels) -> scalar`` — head-specific loss
    (BCE for fire, soft-CE for target, multi-axis CE for move, …).
    ``metrics_fn(logits, labels) -> dict[str, float]`` — returns
    metrics under the same key names BC training uses
    (``acc_target``, ``balanced_acc_target``, ``f1_fire`` …) so probe
    JSON diffs cleanly against BC ``bc_history.json``.
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
    """A ``ModelConfig`` with every trunk/GRU/head BC flag disabled.

    Head probes call ``QNNPolicy`` via ``model_factory=...``, which
    means ``_CombatObjectiveNet.__init__`` — the only consumer of
    trunk/GRU dimensions — is never invoked. The policy layer still
    reads policy-relevant flags (``use_gru``, ``use_weapon_head``, …),
    so those need to be False for the probe to make sense. Numeric
    fields (``ffn_dim``, ``n_layers``, …) are inert under the factory
    path and set to safe placeholders.

    ``d_model`` and ``self_weapon_embed_in_self`` are required so each
    head's build callable (which reads probe.json) is the only place
    those values come from — no implicit defaults in this helper.
    """
    return ModelConfig(
        d_model=int(d_model),
        n_heads=1,
        n_layers=0,
        ffn_dim=0,
        attn_dropout=0.0,
        use_gru=False,
        gru_hidden=0,
        use_weapon_head=False,
        weapon_switch_confidence=0.0,
        weapon_switch_margin=0.0,
        weapon_use_gru=False,
        weapon_use_self_readout=True,
        weapon_context_from_obs=False,
        look_bypass_gru=False,
        gru_target_query=False,
        hard_target_feat=False,
        weapon_in_target_query=False,
        linear_slot_prior=False,
        gt_dist_target_feat=False,
        prev_target_in_query=False,
        self_weapon_embed_in_self=bool(self_weapon_embed_in_self),
        head_bottleneck_dim={"move": 0, "look": 0, "fire": 0, "weapon": 0},
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
    Tokenized heads (``fire_token``) leave it empty.
    """
    name: str
    loss: HeadLossSpec
    build: HeadBuilder
    default_features: tuple[str, ...] = ()
