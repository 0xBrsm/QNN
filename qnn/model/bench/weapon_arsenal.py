"""weapon_arsenal — weapon head fed the explicit ARSENAL inventory token + target_feat.

First split-token-style ablation of the weapon classifier. Every prior weapon
head read the *monolithic* self readout (state+arsenal+motion bundled) — e.g.
``weapon_dense_noembed`` at acc 0.626 / macro-f1 0.48. This head reads ONLY the
``arsenal`` inventory token — per-weapon ``WeaponReadiness`` (ownership×ammo) +
arsenal powerup (QUAD), the exact contents of the split ``arsenal`` subtoken —
plus the GT-pooled ``target_feat``. No held-weapon embed (no incumbent leak),
no temporal, no state/motion.

Question it answers: does inventory + target alone drive "right weapon for the
job", and how far below the full-self-readout baseline (0.626) does
arsenal-only land?

Scaffold (same pattern as weapon_preattn / attack_geom_bundle):
    ObsEmbedding(monolithic, include_spatial=False)
      → PreAttnEncoder (passthrough; no attention)
      → GTTargetPointer (oracle target_feat)
      → WeaponArsenalHead
All other heads Off. The network's weapon selector is
``cat(self_readout, target_feat)``; this head IGNORES the self_readout half and
slices ``target_feat`` (the last d_model) — the arsenal block comes from the
ObsAccessor instead, so "self readout" plays no role.

probe.json: head=weapon_arsenal, d_model, d_hidden, use_target_feat (bool,
default true — set false for the inventory-only / no-target ablation),
use_motion_token (bool, default false — concat the canonical motion subtoken
[vel/pitch + look_delta + movement_id + motion powerup] for the arsenal+motion
ablation), use_state_token (bool, default false — concat the canonical lean
state subtoken [health/armor + armor_type + state powerups]; no weapon_id
incumbent leak, no arsenal-powerup double-count), use_weapon_token (bool,
default false — concat a held-weapon-identity token [masked VocabEmbed on
weapon_id], the modern split-token form of self_weapon_embed_in_self; this is
the incumbent, for the weapon+arsenal+motion held-weapon-input variant).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.obs_network import BenchObsNetwork
from qnn.model.tokens.obs_accessor import current_obs_accessor
from qnn.model.tokens.obs_fields import (
    MOTION_FIELDS,
    SELF_STATE_FIELDS,
    VocabEmbed,
    VocabSum,
    WeaponReadiness,
)
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.network import Off
from qnn.model.transformer import ObsEmbedding
from qnn.model.weapon_head import WeaponHeadInput, WeaponHeadOutput
from qnn.schema import WEAPON_HEAD_SIZE


class WeaponArsenalHead(nn.Module):
    """P(weapon | arsenal inventory token [+ target_feat]); no incumbent, no temporal."""

    def __init__(
        self,
        *,
        d_model: int,
        d_hidden: int,
        use_target_feat: bool,
        use_motion_token: bool,
        use_state_token: bool,
        use_weapon_token: bool,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_target_feat = bool(use_target_feat)
        self.use_motion_token = bool(use_motion_token)
        self.use_state_token = bool(use_state_token)
        self.use_weapon_token = bool(use_weapon_token)
        # Arsenal = pure INVENTORY, matching the split arsenal subtoken:
        # per-weapon readiness (ownership×ammo) + arsenal powerup (QUAD).
        self.arsenal_builder = TokenBuilder(
            self.d_model,
            [WeaponReadiness(), VocabSum("powerup_arsenal")],
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        # Weapon token = held-weapon identity (masked VocabEmbed on weapon_id),
        # the modern split-token equivalent of self_weapon_embed_in_self. This is
        # the INCUMBENT — deliberately excluded by the base arsenal probe — added
        # here for the weapon+arsenal+motion variant (the held-weapon-input dense
        # in the modern token stack; see src/docs/weapon-head.md §6).
        self.weapon_builder = (
            TokenBuilder(
                self.d_model,
                [VocabEmbed("weapon_id")],
                entity_embed=entity_embed,
                movement_embed=movement_embed,
            )
            if self.use_weapon_token
            else None
        )
        # Motion = the canonical MOTION_FIELDS (vel/pitch + look_delta + movement_id
        # + motion powerup), the same list the split motion subtoken builds.
        self.motion_builder = (
            TokenBuilder(
                self.d_model,
                list(MOTION_FIELDS),
                entity_embed=entity_embed,
                movement_embed=movement_embed,
            )
            if self.use_motion_token
            else None
        )
        # State = the canonical lean SELF_STATE_FIELDS (health/armor + armor_type
        # + state powerups). No weapon_id (incumbent leak) and no powerup_arsenal
        # (already in the arsenal token).
        self.state_builder = (
            TokenBuilder(
                self.d_model,
                list(SELF_STATE_FIELDS),
                entity_embed=entity_embed,
                movement_embed=movement_embed,
            )
            if self.use_state_token
            else None
        )
        d_in = self.d_model * (
            1 + int(self.use_weapon_token) + int(self.use_target_feat)
            + int(self.use_motion_token) + int(self.use_state_token)
        )
        self.mlp = make_head_mlp(d_in, WEAPON_HEAD_SIZE, d_hidden, "gelu")
        # Soft-mix context for motor heads (all Off here, but keep the API).
        self.embed = nn.Embedding(WEAPON_HEAD_SIZE, self.d_model)

    def forward(self, inp: WeaponHeadInput) -> WeaponHeadOutput:
        dtype = inp.selector.dtype
        acc = current_obs_accessor()
        arsenal = self.arsenal_builder(acc, dtype=dtype)   # (B*, d_model)
        parts = [arsenal]
        if self.weapon_builder is not None:
            parts.append(self.weapon_builder(acc, dtype=dtype))
        if self.state_builder is not None:
            parts.append(self.state_builder(acc, dtype=dtype))
        if self.motion_builder is not None:
            parts.append(self.motion_builder(acc, dtype=dtype))
        if self.use_target_feat:
            # selector = cat(self_readout, target_feat); take target_feat (last d_model).
            # Detach the slice: target_feat is already grad-detached at the source
            # (GTTargetPointer detach_entity_grad=True), but slicing a grad-requiring
            # cat([self_readout, target_feat]) leaves SliceBackward->CatBackward, which
            # routes a zero cotangent back through the ENTIRE ObsEmbedding entity/self
            # projection graph every step (~3x backward cost on ROCm). Detaching cuts
            # that dead traversal; logits and head gradients are bit-identical.
            parts.append(inp.selector[..., -self.d_model:].detach())
        sel = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        logits = self.mlp(sel)
        context = F.softmax(logits, dim=-1) @ self.embed.weight
        return WeaponHeadOutput(logits=logits, context=context)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=weapon_arsenal")
    return probe[key]


def _build_weapon_arsenal(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model          = int(_required(probe, "d_model"))
    d_hidden         = int(_required(probe, "d_hidden"))
    use_target_feat  = bool(probe.get("use_target_feat", True))
    use_motion_token = bool(probe.get("use_motion_token", False))
    use_state_token  = bool(probe.get("use_state_token", False))
    use_weapon_token = bool(probe.get("use_weapon_token", False))

    model_config = neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False)

    def factory(obs_dim: int, model_cfg) -> BenchObsNetwork:
        obs_embedding = ObsEmbedding(
            d_model=d_model, self_weapon_embed_in_self=False, include_spatial=False,
        )
        # Only build the GT pointer when the head actually reads target_feat.
        # With use_target_feat=False the head ignores the selector entirely, so a
        # pointer would soft-pool entity tokens every step and throw the result
        # away — pure overhead. Off skips the pool, the log, and the target
        # supervision-context entity work.
        target_pointer = GTTargetPointer(d_model=d_model) if use_target_feat else Off
        return BenchObsNetwork(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=obs_embedding,
            encoder=PreAttnEncoder(),
            target_pointer=target_pointer,
            temporal=Off,
            move_head=Off,
            look_head=Off,
            attack_head=Off,
            weapon_head=WeaponArsenalHead(
                d_model=d_model, d_hidden=d_hidden, use_target_feat=use_target_feat,
                use_motion_token=use_motion_token, use_state_token=use_state_token,
                use_weapon_token=use_weapon_token,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            ),
        )

    return model_config, factory


WEAPON_ARSENAL = HeadSpec(
    name="weapon_arsenal",
    loss=HeadLossSpec(
        # Canonical weapon CE + per-class metrics live in policy.py; the runner
        # doesn't dispatch through these for weapon (mirrors weapon_preattn).
        loss_fn=lambda *_a, **_k: None,
        metrics_fn=lambda *_a, **_k: {},
        label_key="weapon",
        output_dim=WEAPON_HEAD_SIZE,
        selection_metric="weapon_skill",
        selection_lower_is_better=False,
    ),
    build=_build_weapon_arsenal,
)
