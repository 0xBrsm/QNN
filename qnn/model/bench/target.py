"""Target-head probe: flat-feature MLP loss / metrics / HeadSpec.

Soft-CE on the labeler's 17-dim ``target_probs`` (NO_TARGET in column 0
+ 16 idx probabilities). Same math as the target branch in
``qnn.model.policy._compute_head_losses_and_metrics``, written
standalone here so the probe doesn't drag in the BC policy module.

The model is a generic ``FlatFeatureHead`` whose 16-dim output flows
through the BC forward contract's ``target_logits`` idx — the
canonical target soft-CE path then computes the loss against
``actions["target_probs"]`` unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from qnn.model.bench.flat import FlatFeatureHead
from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.bc.target_labeler import NO_TARGET_INDEX
from qnn.model.network import ModelConfig


def target_soft_ce_loss(
    logits: torch.Tensor,       # (B, 16)
    target_probs: torch.Tensor,  # (B, 17) full GT with NO_TARGET col 0
) -> torch.Tensor:
    """Present-weighted soft-CE — mirror of qnn.model.policy soft-CE branch.

    Renormalize per-frame idx mass to 1, present-weight the per-frame
    cross-entropy, no in-policy gate (segment_mask is the only filter).
    """
    present = (1.0 - target_probs[:, NO_TARGET_INDEX]).clamp(min=0.0)
    idx_dist = target_probs[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    idx_target = idx_dist / present.clamp(min=1e-6).unsqueeze(-1)
    per_frame_ce = -(idx_target * log_probs).sum(dim=-1)
    return (present * per_frame_ce).sum() / present.sum().clamp(min=1e-6)


def target_metrics(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> dict[str, float]:
    """Returns BC's target_* metric keys + acc_target / balanced_acc_target."""
    with torch.no_grad():
        present = (1.0 - target_probs[:, NO_TARGET_INDEX]).clamp(min=0.0)
        idx_dist = target_probs[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        soft = F.softmax(logits, dim=-1)
        idx_target = idx_dist / present.clamp(min=1e-6).unsqueeze(-1)

        per_frame_ce = -(idx_target * log_probs).sum(dim=-1)
        nll = (present * per_frame_ce).sum() / present.sum().clamp(min=1e-6)
        ent_per_frame = -(idx_target.clamp(min=1e-8) * idx_target.clamp(min=1e-8).log()).sum(dim=-1)
        entropy = (present * ent_per_frame).sum() / present.sum().clamp(min=1e-6)
        brier_per_frame = ((soft - idx_target) ** 2).sum(dim=-1)
        brier = (present * brier_per_frame).sum() / present.sum().clamp(min=1e-6)

        pred = logits.argmax(dim=-1)
        target_label = idx_dist.argmax(dim=-1)
        batch_idx = torch.arange(pred.shape[0], device=pred.device)
        top1_mass = (present * idx_target[batch_idx, pred]).sum() / present.sum().clamp(min=1e-6)

        acc = (pred == target_label).float().mean()
        recalls: list[float] = []
        n_indices = logits.shape[-1]
        for s in range(n_indices):
            true_s = (target_label == s)
            if bool(true_s.any().item()):
                tp = ((pred == s) & true_s).float().sum()
                recalls.append((tp / true_s.float().sum()).item())
        bal = float(sum(recalls) / len(recalls)) if recalls else 0.0

    return {
        "loss_target": float(nll.item()),
        "target_entropy": float(entropy.item()),
        "target_kl": float((nll - entropy).item()),
        "target_brier": float(brier.item()),
        "target_top1_mass": float(top1_mass.item()),
        "target_present_mean": float(present.mean().item()),
        "acc_target": float(acc.item()),
        "balanced_acc_target": bal,
    }


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=target "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


def _build_target(probe: Mapping[str, Any]) -> HeadBuildResult:
    """Build the flat-feature target probe from probe.json. All keys required.

    Reads:
      hidden (int), n_hidden_layers (int), dropout (float),
      feature_names (list of str), d_model (int — inert under the
      flat-feature path but required so probe.json reflects the full
      probe surface), self_weapon_embed_in_self (bool — likewise).
    """
    hidden = int(_required(probe, "hidden"))
    n_hidden_layers = int(_required(probe, "n_hidden_layers"))
    dropout = float(_required(probe, "dropout"))
    raw_names = _required(probe, "feature_names")
    if not isinstance(raw_names, list) or len(raw_names) == 0:
        raise RuntimeError(
            "probe.json.feature_names must be a non-empty list for head=target"
        )
    feature_names = tuple(str(s) for s in raw_names)
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> FlatFeatureHead:
        del obs_dim, model_cfg
        return FlatFeatureHead(
            feature_names=feature_names,
            output_route="target",
            hidden=hidden,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
        )

    return model_config, factory


TARGET = HeadSpec(
    name="target",
    # Per-frame inputs only; target IS the label so target_probs features
    # are excluded. The pre-refactor default included ``look`` (the
    # next-frame action label), which the canonical BC supervised loop
    # doesn't pass to the model. Substituted with ``self_velocity``
    # here — a closely related motion proxy already in the obs dict.
    # default_features is documentation; the template's probe.json
    # owns the actual feature list.
    default_features=(
        "entity_scalars_flat",
        "self_velocity",
        "self_weapon_one_hot",
    ),
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_kl",
        selection_lower_is_better=True,
    ),
    build=_build_target,
)
