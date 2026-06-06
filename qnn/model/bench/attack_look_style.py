"""Attack-head probe: feed the attack head LookHead's inputs.

Bench probe that pairs ``LookStyleAttackHead`` with the canonical bench
input bypasses (``PreAttnEncoder`` for the self token, ``GTTargetPointer``
for the GT-softmaxed target distribution read from the ``act_target_probs``
sidecar). The attack head's inputs match :class:`LookHeadInput` exactly:

  features      = cat(self_readout, target_feat)        # dense MLP input
  target_logits = log-probs from GTTargetPointer        # per-entity logits
  entity_rel    = entity_scalars[..., REL]              # via AttackHeadInput
  actor_mask    = entity_types == ACTOR                 # via AttackHeadInput

Architecture inside the head mirrors LookHead (geometric prior + learned
residual), but emits a scalar attack logit. Tests whether the dense+
geometric input pattern that works for look transfers to attack.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qnn.model.bench.attack import attack_bce_loss, attack_metrics
from qnn.model.bench.attack_prior.look_style_attack_head import LookStyleAttackHead
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.model.network import Network, Off, compute_slot_dims
from qnn.model.transformer import ObsEmbedding


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=attack_look_style "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


def _build_attack_look_style(probe: Mapping[str, Any]) -> HeadBuildResult:
    """Required keys in probe.json:

      d_model (int).
      self_weapon_embed_in_self (bool).
      hidden (int): residual MLP bottleneck width.
      alignment_scale (float): fixed prior scale (default 5.0).
    """
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    hidden = int(_required(probe, "hidden"))
    alignment_scale = float(probe.get("alignment_scale", 5.0))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )
    dims = compute_slot_dims(model_config, has_temporal=False, has_weapon_head=False)

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
            look_head=Off,
            weapon_head=Off,
            attack_head=LookStyleAttackHead(
                in_dim=dims["motor_in"],
                bottleneck_dim=hidden,
                activation="gelu",
                scale_init=alignment_scale,
            ),
        )

    return model_config, factory


ATTACK_LOOK_STYLE = HeadSpec(
    name="attack_look_style",
    loss=HeadLossSpec(
        loss_fn=attack_bce_loss,
        metrics_fn=attack_metrics,
        label_key="attack",
        output_dim=1,
        selection_metric="f1_attack",
        selection_lower_is_better=False,
    ),
    build=_build_attack_look_style,
)
