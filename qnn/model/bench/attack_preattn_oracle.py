"""Attack-head probe with an analytical oracle prior + residual MLP.

Slot-friendly variant: builds a Network with ``PreAttnEncoder`` in the
encoder slot, ``GTTargetPointer`` in the target_pointer slot, and
``OracleAttackHead`` (from ``bench.attack_prior``) in the attack_head
slot — everything else is Off.

The oracle ``required_look`` computation mirrors
``scripts/analysis/attack_precision_offset.py`` (hitscan → rel/|rel|;
projectile → constant-velocity intercept solve, gravity ignored).
No engine-readiness gate inside the head — ``input_mask`` filtering
applied by the trainer covers that.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qnn.model.bench.attack import attack_bce_loss, attack_metrics
from qnn.model.bench.attack_prior.oracle_attack_head import OracleAttackHead
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.transformer import ObsEmbedding
from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.model.network import Network, Off, compute_slot_dims


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=attack_preattn_oracle "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


def _build_attack_oracle(probe: Mapping[str, Any]) -> HeadBuildResult:
    """Required keys in probe.json:

      d_model (int).
      self_weapon_embed_in_self (bool).
      d_hidden (int): residual MLP bottleneck width.
      aim_scale (float): prior magnitude per unit aim cosine.
    """
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    d_hidden = int(_required(probe, "d_hidden"))
    aim_scale = float(_required(probe, "aim_scale"))

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
            attack_head=OracleAttackHead(
                in_dim=dims["motor_in"],
                d_hidden=d_hidden,
                activation="gelu",
                aim_scale=aim_scale,
            ),
        )

    return model_config, factory


ATTACK_PREATTN_ORACLE = HeadSpec(
    name="attack_preattn_oracle",
    loss=HeadLossSpec(
        loss_fn=attack_bce_loss,
        metrics_fn=attack_metrics,
        label_key="attack",
        output_dim=1,
        selection_metric="f1_attack",
        selection_lower_is_better=False,
    ),
    build=_build_attack_oracle,
)
