"""Fire-head probe: flat-feature MLP loss / metrics / HeadSpec.

Binary BCE on ``act.attack``. Default features include privileged
``target_probs_indices`` (the 16-idx GT distribution the BC trainer
already passes alongside the obs) and target-pooled actor rel/vel —
answers "how well can fire be predicted from per-frame state when the
right target is known."

The model is a generic ``FlatFeatureHead`` driven by the canonical BC
supervised loop via ``QNNPolicy(model_factory=...)``; no parallel
pipeline.
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
from qnn.model.network import ModelConfig


def attack_bce_loss(
    logits: torch.Tensor,       # (B, 1)
    attack_target: torch.Tensor,  # (B,) {0, 1}
    pos_weight: float = 1.0,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), attack_target.float(), pos_weight=pw,
    )


def attack_metrics(
    logits: torch.Tensor,
    attack_target: torch.Tensor,
) -> dict[str, float]:
    """f1 / precision / recall / pos_rate / acc / confidence, BC-style keys."""
    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze(-1))
        pred = (probs >= 0.5).long()
        target = attack_target.long()
        tp = ((pred == 1) & (target == 1)).float().sum()
        fp = ((pred == 1) & (target == 0)).float().sum()
        fn = ((pred == 0) & (target == 1)).float().sum()
        prec = (tp / (tp + fp).clamp(min=1.0)).item()
        rec = (tp / (tp + fn).clamp(min=1.0)).item()
        f1 = (2.0 * prec * rec / max(prec + rec, 1e-6))
        pos_rate = float(target.float().mean().item())
        acc = float((pred == target).float().mean().item())
        conf = float(probs.mean().item())
    return {
        "f1_attack": f1,
        "precision_attack": prec,
        "recall_attack": rec,
        "pos_rate_attack": pos_rate,
        "acc_attack": acc,
        "confidence_attack": conf,
    }


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=attack "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


def _build_attack(probe: Mapping[str, Any]) -> HeadBuildResult:
    """Build the flat-feature attack probe from probe.json. All keys required.

    Reads:
      hidden (int), n_hidden_layers (int), dropout (float),
      feature_names (list of str), d_model (int — neutral ModelConfig
      width, inert here but required so probe.json reflects the full
      surface area), self_weapon_embed_in_self (bool — likewise inert
      under the flat-feature path but kept on the probe schema for
      symmetry with attack_preattn).
    """
    hidden = int(_required(probe, "hidden"))
    n_hidden_layers = int(_required(probe, "n_hidden_layers"))
    dropout = float(_required(probe, "dropout"))
    raw_names = _required(probe, "feature_names")
    if not isinstance(raw_names, list) or len(raw_names) == 0:
        raise RuntimeError(
            "probe.json.feature_names must be a non-empty list for head=attack"
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
            output_route="attack",
            hidden=hidden,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
        )

    return model_config, factory


ATTACK = HeadSpec(
    name="attack_flat",
    # Principled "now" feature set per the user's framing: enemy
    # position (privileged via target_probs_indices), current weapon,
    # ammo, and cooldown — the four things fire actually needs. Look is
    # omitted because target_pooled_rel is in view-relative coords (engine
    # transforms rel via player_view_angles), so (1,0,0) already means
    # "on crosshair" — look is the next-frame intent, redundant for
    # the now-decision.
    #
    # default_features is documentation only — the template's probe.json
    # is the single source of truth; the user copies these names there.
    default_features=(
        "self_health_armor",
        "self_weapon_one_hot",
        "self_ammo",
        "self_attack_finished",
        "target_probs_indices",
        "target_pooled_rel",
        "target_pooled_vel",
    ),
    loss=HeadLossSpec(
        loss_fn=attack_bce_loss,
        metrics_fn=attack_metrics,
        label_key="attack",
        output_dim=1,
        selection_metric="f1_attack",
        selection_lower_is_better=False,
    ),
    build=_build_attack,
)
