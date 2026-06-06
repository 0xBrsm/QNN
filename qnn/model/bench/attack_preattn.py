"""Attack-head probe: PreAttn encoder + GT-pool target + attack-head variant.

Builds a Network with the encoder swapped for ``PreAttnEncoder`` (no
transformer attention), target_pointer swapped for ``GTTargetPointer``
(uses GT idx distribution directly, no learned pointer), temporal off,
and every head except attack disabled.

The ``prior_mode`` key in probe.json selects which attack-head module
fills the slot:

  none                              → canonical ``AttackHead``
  geometric_fixed                   → ``AimPriorAttackHead(scale_mode="fixed")``
  geometric_learnable_scalar        → ``AimPriorAttackHead(scale_mode="scalar")``
  geometric_learnable_perweapon     → ``AimPriorAttackHead(scale_mode="perweapon")``
  hit_test                          → ``HitTestAttackHead``

Privileged input: the GT target distribution flows through the
``GTTargetPointer``. The probe is an *oracle-pointer* setup — it
isolates head capacity from pointer error and is not usable at
inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qnn.model.bench.attack import attack_bce_loss, attack_metrics
from qnn.model.bench.attack_prior.aim_prior_attack_head import AimPriorAttackHead
from qnn.model.bench.attack_prior.hit_test_attack_head import HitTestAttackHead
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.transformer import ObsEmbedding
from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.model.attack_head import AttackHead
from qnn.model.network import Network, Off, compute_slot_dims


_GEOM_SCALE_MODES = {
    "geometric_fixed": "fixed",
    "geometric_learnable_scalar": "scalar",
    "geometric_learnable_perweapon": "perweapon",
}


def _build_attack_head(
    prior_mode: str,
    *,
    in_dim: int,
    bottleneck: int,
    alignment_scale: float,
):
    if prior_mode == "none":
        return AttackHead(in_dim=in_dim, bottleneck_dim=bottleneck, activation="gelu")
    if prior_mode in _GEOM_SCALE_MODES:
        return AimPriorAttackHead(
            in_dim=in_dim,
            bottleneck_dim=bottleneck,
            activation="gelu",
            scale_mode=_GEOM_SCALE_MODES[prior_mode],
            scale_init=alignment_scale,
        )
    if prior_mode == "hit_test":
        return HitTestAttackHead(
            in_dim=in_dim, bottleneck_dim=bottleneck, activation="gelu",
        )
    raise ValueError(
        f"unknown prior_mode {prior_mode!r}; expected one of "
        f"'none', {sorted(_GEOM_SCALE_MODES)}, 'hit_test'"
    )


def _build_attack_preattn(probe: Mapping[str, Any]) -> HeadBuildResult:
    """HeadBuilder for ``attack`` — required keys in probe.json.

    Required:

      d_model (int): token width (PreAttnEncoder + AttackHead share it).
      self_weapon_embed_in_self (bool): pass through to ObsEmbedding.
      hidden (int): residual MLP bottleneck width.

    Optional:

      prior_mode (str): one of {"none", "geometric_fixed",
        "geometric_learnable_scalar", "geometric_learnable_perweapon",
        "hit_test"}. Default "none". Geometric modes need ``base_look``
        so ``LookHead`` stays in the slot for those (its ``base_look``
        is a pure function of target_logits + entity_rel with no
        learnable params).
      alignment_scale (float): seed value for the geometric scale
        (fixed for ``geometric_fixed``; initial value for the learnable
        variants). Default 5.0.
    """
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    hidden = int(_required(probe, "hidden"))
    prior_mode = str(probe.get("prior_mode", "none"))
    alignment_scale = float(probe.get("alignment_scale", 5.0))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )
    dims = compute_slot_dims(
        model_config, has_temporal=False, has_weapon_head=False,
    )

    needs_look_head = prior_mode in _GEOM_SCALE_MODES

    def factory(obs_dim: int, model_cfg) -> Network:
        return Network(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=GTTargetPointer(d_model=d_model),
            temporal=Off,
            move_head=Off,
            look_head=None if needs_look_head else Off,
            weapon_head=Off,
            attack_head=_build_attack_head(
                prior_mode,
                in_dim=dims["motor_in"],
                bottleneck=hidden,
                alignment_scale=alignment_scale,
            ),
        )

    return model_config, factory


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=attack "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


ATTACK_PREATTN = HeadSpec(
    name="attack",
    loss=HeadLossSpec(
        loss_fn=attack_bce_loss,
        metrics_fn=attack_metrics,
        label_key="attack",
        output_dim=1,
        selection_metric="f1_attack",
        selection_lower_is_better=False,
    ),
    build=_build_attack_preattn,
)
