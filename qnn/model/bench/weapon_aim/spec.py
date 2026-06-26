"""HeadSpec for the weapon-aim ablation — joint look + attack heads.

Both heads consume the **same LookHead-style inputs**: features =
``cat(self_readout, target_feat)``, target_logits, entity_rel,
actor_mask. The look head uses them for the canonical
target-anchored prior; the attack head (LookStyleAttackHead) uses
them to derive the same prior + a scalar fire logit. Holding attack
constant across variants isolates the look-prior change.

The variant in probe.json selects:

  ``canonical``    — canonical LookHead + LookStyleAttackHead. Baseline.
  ``weapon_aim``   — WeaponAimLookHead + LookStyleAttackHead. Attack
                     stays identical to canonical; only the look prior
                     swaps to the weapon-aware aim_vec.

The ``weapon_aim`` variant runs through ``WeaponAimNetwork`` so its
look head can read the forward-scoped weapon context; ``canonical``
uses bare ``Network``.

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
from qnn.model.bench.attack_prior.engaged_geom_weapon_embed_attack_head import (
    EngagedGeomWeaponEmbedAttackHead,
)
from qnn.model.bench.attack_prior.engaged_look_style import EngagedLookStyleAttackHead
from qnn.model.bench.attack_prior.engaged_prior_attack_head import EngagedPriorAttackHead
from qnn.model.bench.attack_prior.geom_attack_head import GeomAttackHead
from qnn.model.bench.attack_prior.look_style_attack_head import LookStyleAttackHead
from qnn.model.bench.attack_prior.weapon_embed_attack_head import WeaponEmbedAttackHead
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.inputs.weapon_head_obs_embedding import (
    AttackFinishedOnlyObsEmbedding, TargetOnlyObsEmbedding, WeaponHeadObsEmbedding,
)
from qnn.model.bench.look_head_move_token import MoveTokenLookHead
from qnn.model.bench.look_head_gain import GainLookHead
from qnn.model.bench.inputs.obs_network import BenchObsNetwork
from qnn.model.bench.inputs.move_aim_network import MoveAimNetwork
from qnn.model.look_head import LookHead
from qnn.model.network import Network, Off, slot_dims
from qnn.model.transformer import ObsEmbedding

from qnn.model.bench.weapon_aim.look_head import WeaponAimLookHead
from qnn.model.bench.weapon_aim.network import WeaponAimNetwork


_VARIANTS = ("canonical", "weapon_aim")
_FEATURE_TOKENS = (
    "self",
    "weapon",                 # weapon_static + attack_finished + held-weapon ammo
    "weapon_ammo",            # weapon_static + held-weapon ammo (no cooldown)
    "weapon_cooldown",        # weapon_static + attack_finished (no ammo)
    "target_only",            # zero self/weapon half — only target_feat carries signal
    "attack_finished_only",   # Linear(1, d_model)(attack_finished) — no weapon static / ID
    "move_token",             # look-only: MLP(cat(target_feat, attack_bundle move_token)),
                              #   no prior. BenchObsNetwork wrapper; attack head Off.
    "aimvec_move_token",      # look-only: aim_vec prior + MLP(cat(target_feat, move_token)).
                              #   MoveAimNetwork wrapper (both contexts); attack head Off.
    "gain",                   # look-only: GainLookHead — g·logmap(target_prior) + residual
                              #   in tangent space. Explicit scalar gain on the bearing (vs
                              #   canonical's full-snap unit prior). Bare Network; attack Off.
    "canonical_look",         # look-only: plain LookHead (normalize(unit_prior + delta)) on
                              #   target_feat only — the A/B baseline for "gain". Bare Network;
                              #   attack Off. Falls through to the LookHead else-branch.
)


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
      d_hidden (int)                      — head MLP hidden width.
      activation (str)                    — "none" / "gelu" / "relu".
      variant (str)                       — "canonical" or "weapon_aim".

    Scaffolding: canonical TransformerEncoder + canonical TargetPointer +
    temporal Off + move/weapon Off. Both look and attack heads active.
    """
    d_model         = int(_required(probe, "d_model"))
    self_weapon     = bool(_required(probe, "self_weapon_embed_in_self"))
    d_hidden        = int(_required(probe, "d_hidden"))
    activation      = str(_required(probe, "activation"))
    variant         = str(_required(probe, "variant"))
    alignment_scale = float(probe.get("alignment_scale", 5.0))
    feature_token   = str(probe.get("feature_token", "self"))
    # OFAT knob for the attack-head prior. "look_style" = canonical
    # LookStyleAttackHead (geometric prior only). "engaged_look_style" =
    # same prior + engagement_ema concatenated to the residual MLP input.
    attack_head_kind = str(probe.get("attack_head", "look_style"))
    if variant not in _VARIANTS:
        raise RuntimeError(
            f"variant must be one of {_VARIANTS}, got {variant!r}"
        )
    if feature_token not in _FEATURE_TOKENS:
        raise RuntimeError(
            f"feature_token must be one of {_FEATURE_TOKENS}, got {feature_token!r}"
        )
    _ATTACK_HEAD_KINDS = (
        "look_style", "engaged_look_style", "engaged_prior",
        "geom", "weapon_embed", "engaged_geom_weapon_embed",
    )
    weapon_embed_dim = int(probe.get("weapon_embed_dim", 8))
    if attack_head_kind not in _ATTACK_HEAD_KINDS:
        raise RuntimeError(
            f"attack_head must be one of {_ATTACK_HEAD_KINDS}, got {attack_head_kind!r}"
        )

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )
    dims = slot_dims(
        d_model=model_config.d_model, d_gru=model_config.d_gru,
        has_temporal=False, has_target_pointer=True, has_weapon_head=False,
        weapon_sources=model_config.weapon_sources,
    )
    motor_in = dims["motor_in"]   # 2 * d_model (no GRU, no weapon head)

    def factory(obs_dim: int, model_cfg) -> Network:
        # Build obs_embedding FIRST so heads that tie embeds (move_token) can
        # reference its entity_embed / movement_embed.
        if feature_token == "self":
            obs_embedding = ObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=False,
            )
        elif feature_token in ("target_only", "move_token", "aimvec_move_token",
                               "gain", "canonical_look"):
            obs_embedding = TargetOnlyObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=False,
            )
        elif feature_token == "attack_finished_only":
            obs_embedding = AttackFinishedOnlyObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=False,
            )
        else:
            # weapon / weapon_ammo / weapon_cooldown — attack_finished is now
            # unconditional on the weapon token (no flag); only ammo still toggles.
            obs_embedding = WeaponHeadObsEmbedding(
                d_model=d_model, self_weapon_embed_in_self=self_weapon, include_spatial=False,
                include_ammo=(feature_token in ("weapon", "weapon_ammo")),
            )

        # Network wrapper installs the forward-scoped context(s) heads read.
        # move_token needs the ObsAccessor scope; aimvec_move_token needs both
        # that and WeaponAimContext; aim_vec/weapon_aim heads need WeaponAim.
        if feature_token == "aimvec_move_token":
            net_cls = MoveAimNetwork
        elif feature_token == "move_token":
            net_cls = BenchObsNetwork
        elif variant == "weapon_aim":
            net_cls = WeaponAimNetwork
        else:
            net_cls = Network

        if feature_token in ("move_token", "aimvec_move_token"):
            look_head = MoveTokenLookHead(
                d_model=d_model, d_hidden=d_hidden, activation=activation,
                prior=("aim_vec" if feature_token == "aimvec_move_token" else "none"),
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
        elif feature_token == "gain":
            look_head = GainLookHead(
                in_dim=motor_in, d_hidden=d_hidden, activation=activation,
            )
        elif variant == "weapon_aim":
            look_head = WeaponAimLookHead(
                in_dim=motor_in, d_hidden=d_hidden, activation=activation,
            )
        else:
            look_head = LookHead(
                in_dim=motor_in, d_hidden=d_hidden, activation=activation,
            )
        # Look-only ablations skip the attack head entirely.
        if feature_token in ("move_token", "aimvec_move_token", "gain",
                             "canonical_look"):
            attack_head = Off
        elif attack_head_kind == "engaged_look_style":
            attack_head = EngagedLookStyleAttackHead(
                in_dim=motor_in,
                d_hidden=d_hidden,
                activation=activation,
                scale_init=alignment_scale,
            )
        elif attack_head_kind == "engaged_prior":
            attack_head = EngagedPriorAttackHead(
                in_dim=motor_in,
                d_hidden=d_hidden,
                activation=activation,
                scale_init=alignment_scale,
            )
        elif attack_head_kind == "geom":
            attack_head = GeomAttackHead(
                in_dim=motor_in,
                d_hidden=d_hidden,
                activation=activation,
                scale_init=alignment_scale,
            )
        elif attack_head_kind == "weapon_embed":
            attack_head = WeaponEmbedAttackHead(
                in_dim=motor_in,
                d_hidden=d_hidden,
                activation=activation,
                scale_init=alignment_scale,
                weapon_embed_dim=weapon_embed_dim,
            )
        elif attack_head_kind == "engaged_geom_weapon_embed":
            attack_head = EngagedGeomWeaponEmbedAttackHead(
                in_dim=motor_in,
                d_hidden=d_hidden,
                activation=activation,
                scale_init=alignment_scale,
                weapon_embed_dim=weapon_embed_dim,
            )
        else:
            attack_head = LookStyleAttackHead(
                in_dim=motor_in,
                d_hidden=d_hidden,
                activation=activation,
                scale_init=alignment_scale,
            )

        return net_cls(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=obs_embedding,
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
# "look_r2" as the selection metric since look is the primary
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
        selection_metric="look_skill",
        selection_lower_is_better=False,
    ),
    build=_build_weapon_aim,
)
