"""HeadSpec for the weapon-aim ablation — joint look + attack heads.

Both heads consume the same bench inputs: ``aim_vec`` (weapon-aware
lead-corrected direction) and ``target_feat`` (soft-pooled entity
feature). Attack also gets ``noop`` (binary fire-feasibility gate).

The variant in probe.json selects:

  ``canonical``    — canonical LookHead + canonical AttackHead, no
                     weapon-awareness. Baseline.
  ``weapon_aim``   — WeaponAimLookHead + canonical AttackHead. Attack
                     stays canonical while the look prior is isolated.

Both variants run through the same ``WeaponAimNetwork`` wrapper for
construction symmetry; in ``canonical`` the stash is a no-op (canonical
heads ignore the side-channel attrs).

Loss/metrics: this spec is for joint look+attack training, so the
``loss_fn`` field of the spec is a no-op stub — canonical
``QNNPolicy._compute_head_losses_and_metrics`` computes both head
losses end-to-end via the standard BC pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.model.attack_head import AttackHead
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.look_head import LookHead
from qnn.model.network import Network, Off, compute_slot_dims
from qnn.model.transformer import ObsEmbedding

from qnn.model.bench.weapon_aim.look_head import WeaponAimLookHead
from qnn.model.bench.weapon_aim.network import WeaponAimNetwork


_VARIANTS = ("canonical", "weapon_aim")


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=weapon_aim "
            "(no Python-level defaults — see "
            "qnn.model.bench.weapon_aim.probe.json)."
        )
    return probe[key]


def _build_weapon_aim(probe: Mapping[str, Any]) -> HeadBuildResult:
    """Build a probe Network for the joint look+attack ablation.

    Required probe.json keys:

      d_model (int)                       — encoder + head width.
      self_weapon_embed_in_self (bool)    — ObsEmbedding passthrough.
      bottleneck (int)                    — head MLP bottleneck width.
      activation (str)                    — "none" / "gelu" / "relu".
      variant (str)                       — "canonical" or "weapon_aim".

    Scaffolding: canonical TransformerEncoder + canonical TargetPointer +
    temporal Off + move/weapon Off. Both look and attack heads active.
    """
    d_model     = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    bottleneck  = int(_required(probe, "bottleneck"))
    activation  = str(_required(probe, "activation"))
    variant     = str(_required(probe, "variant"))
    if variant not in _VARIANTS:
        raise RuntimeError(
            f"variant must be one of {_VARIANTS}, got {variant!r}"
        )

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )
    dims = compute_slot_dims(
        model_config, has_temporal=False, has_weapon_head=False,
    )
    motor_in = dims["motor_in"]   # 2 * d_model (no GRU, no weapon head)

    def factory(obs_dim: int, model_cfg) -> Network:
        # The bench wrapper exists to install the WeaponAimContext for
        # forward() — only needed when at least one bench head reads it.
        # Canonical variant uses bare Network and pays no overhead.
        net_cls = WeaponAimNetwork if variant == "weapon_aim" else Network
        look_head = (
            WeaponAimLookHead(in_dim=motor_in, bottleneck_dim=bottleneck, activation=activation)
            if variant == "weapon_aim"
            else LookHead(in_dim=motor_in, bottleneck_dim=bottleneck, activation=activation)
        )
        attack_head = AttackHead(
            in_dim=motor_in, bottleneck_dim=bottleneck, activation=activation,
        )

        return net_cls(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=GTTargetPointer(d_model=d_model),
            temporal=Off,
            move_head=Off,
            weapon_head=Off,
            look_head=look_head,
            attack_head=attack_head,
        )

    return model_config, factory


# ---- loss/metrics: stubs ----
#
# Joint look+attack training uses the canonical BC pipeline's per-head
# losses (look = smooth_l1 residual; attack = BCE) via
# QNNPolicy._compute_head_losses_and_metrics. The HeadSpec.loss field
# here is a placeholder — the runner won't invoke it for this
# multi-head probe. We pick "look" as the nominal label_key and
# "cos_sim_look" as the selection metric since look is the primary
# downstream consumer of aim_vec.


def _stub_loss(*args, **kwargs) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args, **kwargs) -> dict[str, float]:
    return {}


WEAPON_AIM = HeadSpec(
    name="weapon_aim",
    loss=HeadLossSpec(
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="look",
        output_dim=3,
        selection_metric="cos_sim_look",
        selection_lower_is_better=False,
    ),
    build=_build_weapon_aim,
)
