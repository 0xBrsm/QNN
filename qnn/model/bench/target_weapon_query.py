"""HeadSpec for the target-pointer weapon-query ablation.

Replaces the canonical ``TargetPointer``'s MLP scoring with a query
assembled from the held weapon's physical specs + impulse-keyed vocab
embed. The rest of the stack stays canonical: ``ObsEmbedding`` →
``PreAttnEncoder`` → ``WeaponQueryTargetPointer`` over entity tokens,
with motor heads Off. Trained against ``actions["target_probs"]`` via
the canonical BC soft-CE branch in
``QNNPolicy._compute_head_losses_and_metrics`` — set
``head_loss_weights = {"target": 1.0, "move": 0.0, "look": 0.0, "attack": 0.0, "weapon": 0.0}``
in train.json.

Required probe.json keys::

    {
      "head": "target_weapon_query",
      "d_model": 64,
      "self_weapon_embed_in_self": false
    }

Crib the surrounding ``train.json`` / ``machine.json`` / ``run.json``
from any existing weapon_aim head_probe run-dir; this head has no other
knobs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import torch
from torch import nn

from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.bench.inputs.weapon_query_target_pointer import (
    WeaponQueryTargetPointer,
)
from qnn.model.bench.spec import (
    HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config,
)
from qnn.model.bench.target import target_metrics, target_soft_ce_loss
from qnn.model.network import ModelConfig, Network, Off, _flatten_obs
from qnn.model.transformer import ObsEmbedding
from qnn.vocab import self_weapon_id_to_impulse


class WeaponQueryTargetNetwork(Network):
    """Network subclass that stashes per-frame weapon impulse on the
    target_pointer before delegating to the canonical forward.

    Enemy mask is computed by Network itself and supplied via
    :class:`TargetPointerInput.enemy_mask`; this wrapper only handles
    the weapon-impulse side-channel that the pointer needs to build its
    weapon-conditioned query.
    """

    def forward(  # type: ignore[override]
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ):
        _, flat_obs = _flatten_obs(obs)
        wid_impulse = self_weapon_id_to_impulse(
            flat_obs["self_weapon_id"].long().squeeze(-1),
        ).long()
        self.target_pointer.stash(weapon_impulse=wid_impulse)
        return super().forward(obs=obs, hidden=hidden, reset_mask=reset_mask)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=target_weapon_query"
        )
    return probe[key]


def _build_target_weapon_query(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    # Optional: subset of the 7 static weapon scalars
    # (damage=0, cooldown=1, v_horiz=2, v_vert_0=3, gravity=4, max_dist=5, radius=6).
    # Defaults to all 7.
    raw_indices = probe.get("static_scalar_indices")
    static_scalar_indices = (
        tuple(int(i) for i in raw_indices) if raw_indices is not None else None
    )

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        return WeaponQueryTargetNetwork(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=ObsEmbedding(
                d_model=d_model,
                self_weapon_embed_in_self=self_weapon,
                include_spatial=False,
            ),
            encoder=PreAttnEncoder(),
            target_pointer=WeaponQueryTargetPointer(
                d_model=d_model,
                static_scalar_indices=static_scalar_indices,
            ),
            temporal=Off,
            move_head=Off,
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


TARGET_WEAPON_QUERY = HeadSpec(
    name="target_weapon_query",
    loss=HeadLossSpec(
        loss_fn=target_soft_ce_loss,
        metrics_fn=target_metrics,
        label_key="target_probs",
        output_dim=16,
        selection_metric="target_skill",
        selection_lower_is_better=False,
    ),
    build=_build_target_weapon_query,
)
