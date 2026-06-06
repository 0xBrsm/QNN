"""Weapon-head probe: PreAttn encoder + GT-pool target + canonical WeaponHead.

Mirrors ``attack_preattn`` for the 8-way weapon classifier. Network with
``PreAttnEncoder`` in the encoder slot, ``GTTargetPointer`` in the
target_pointer slot, temporal off, every head except weapon disabled.
The weapon head is the canonical ``WeaponHead`` with the historical
``hidden`` bottleneck and ``context_from_obs=False`` (softmax-pooled
weapon embed feeds non-existent motor heads in this probe — but the
classifier itself runs the same way).

Privileged input: GT target distribution via ``target_probs_idx``. The
canonical weapon CE + per-class metrics live in policy.py; the
HeadLossSpec carries the schema knobs but the runner doesn't dispatch
through ``loss_fn`` for weapon.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qnn.model.bench.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.network import Network, compute_slot_dims
from qnn.model.network import Off
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.transformer import ObsEmbedding
from qnn.model.weapon_head import WeaponHead
from qnn.schema import WEAPON_HEAD_SIZE


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=weapon_preattn "
            "(no Python-level defaults — see qnn.model.bench.templates)."
        )
    return probe[key]


def _build_weapon_preattn(probe: Mapping[str, Any]) -> HeadBuildResult:
    """HeadBuilder for ``weapon_preattn`` — required keys in probe.json.

    Reads:

      d_model (int).
      self_weapon_embed_in_self (bool).
      hidden (int): WeaponHead bottleneck width.

    Earlier knobs ``strip_self_embeds``, ``weapon_embed_concat``,
    ``n_hidden_layers``, ``dropout`` were used at defaults in every
    retained run; the canonical WeaponHead reproduces them.
    """
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    hidden = int(_required(probe, "hidden"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )
    dims = compute_slot_dims(
        model_config, has_temporal=False, has_weapon_head=True,
    )

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
            attack_head=Off,
            weapon_head=WeaponHead(
                selector_dim=dims["weapon_in"],
                d_model=d_model,
                bottleneck_dim=hidden,
                activation="gelu",
                context_from_obs=False,
            ),
        )

    return model_config, factory


WEAPON_PREATTN = HeadSpec(
    name="weapon",
    loss=HeadLossSpec(
        # The canonical weapon CE + per-class metrics live in policy.py;
        # these fields are schema-only here — the runner doesn't dispatch
        # through them. Kept for symmetry with attack_preattn's HeadLossSpec.
        loss_fn=lambda *_a, **_k: None,  # not called
        metrics_fn=lambda *_a, **_k: {},  # not called
        label_key="weapon",
        output_dim=WEAPON_HEAD_SIZE,
        selection_metric="f1_weapon_global",
        selection_lower_is_better=False,
    ),
    build=_build_weapon_preattn,
)
