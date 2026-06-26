"""Move-head ablation: MLP over the motion token, with optional second token.

Variants selected by probe.json (mutually exclusive second-token knobs):

  (default) — motion token only
  use_target_feat  — cat(GT-pooled entity token, motion_token)
  use_state_token  — cat(health/armor/weapon/powerup state token, motion_token)
  use_spatial_token — cat(spatial-geometry token, motion_token)
    spatial_token = Linear(9×13, d_model)(spatial_scalars.flatten())

Scaffold: ObsEmbedding → PreAttnEncoder (passthrough) → no temporal.
Only move head active; set head_loss_weights move=1, rest 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model._mlp import make_head_mlp
from qnn.model.bench.inputs.gt_target_pointer import GTTargetPointer
from qnn.model.bench.inputs.obs_network import BenchObsNetwork
from qnn.model.tokens.obs_accessor import current_obs_accessor
from qnn.model.tokens.obs_fields import MOTION_FIELDS, ScalarGroup, VocabEmbed, VocabSum
from qnn.model.bench.inputs.preattn_encoder import PreAttnEncoder
from qnn.model.tokens.token_builder import TokenBuilder
from qnn.model.bench.spec import HeadBuildResult, HeadLossSpec, HeadSpec, neutral_model_config
from qnn.model.move_head import MoveHeadInput, MoveHeadOutput
from qnn.model.network import ModelConfig, Off
from qnn.model.transformer import ObsEmbedding

_SPATIAL_IN = 9 * 13   # SPATIAL_TOKEN_COUNT × SPATIAL_SCALAR_DIM

# Canonical motion-token fields (incl. look_delta) — single source of truth.
_MOTION_FIELDS = MOTION_FIELDS

_STATE_FIELDS = [
    ScalarGroup(["health_armor", "attack_finished"]),
    VocabEmbed("weapon_id"),
    VocabEmbed("armor_type"),
    VocabSum("powerup_state"),
    VocabSum("powerup_arsenal"),
]

_STATE_NO_WEAPON_FIELDS = [
    ScalarGroup(["health_armor", "attack_finished"]),
    VocabEmbed("armor_type"),
    VocabSum("powerup_state"),
    VocabSum("powerup_arsenal"),
]

# weapon_static: per-weapon static properties (MODEL_TOKEN_SCALAR_DIM dims, impulse-indexed)
# held_readiness: current weapon readiness fraction (1 dim)
# attack_finished: cooldown remaining (1 dim)
# weapon_id embed: weapon identity
_WEAPON_FIELDS = [
    ScalarGroup(["weapon_static", "held_readiness", "attack_finished"]),
    VocabEmbed("weapon_id"),
]

_OUT_DIM = MOVE_AXES * MOVE_AXIS_CLASSES  # 9


class MotionTokenMoveHead(nn.Module):
    """Move head driven by the motion token; ignores encoder features."""

    def __init__(
        self,
        *,
        d_model: int,
        d_move: int,
        activation: str,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.motion_builder = TokenBuilder(
            d_model, _MOTION_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.mlp = make_head_mlp(d_model, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        motion = self.motion_builder(current_obs_accessor(), dtype=inp.features.dtype)
        logits = self.mlp(motion).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


class MotionStateTokenMoveHead(nn.Module):
    """Move head: MLP over cat(state_token, motion_token); no encoder features."""

    def __init__(
        self,
        *,
        d_model: int,
        d_move: int,
        activation: str,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
        state_fields=_STATE_FIELDS,
    ) -> None:
        super().__init__()
        self.state_builder = TokenBuilder(
            d_model, state_fields,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.motion_builder = TokenBuilder(
            d_model, _MOTION_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.mlp = make_head_mlp(2 * d_model, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        acc = current_obs_accessor()
        dtype = inp.features.dtype
        state  = self.state_builder(acc, dtype=dtype)
        motion = self.motion_builder(acc, dtype=dtype)
        logits = self.mlp(torch.cat([state, motion], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


class MotionSpatialStateTokenMoveHead(nn.Module):
    """Move head: MLP over cat(spatial_token, state_token, motion_token).

    Combines map-geometry (spatial) and self-state (health/armor/weapon/powerup)
    with motion. MLP input: 3 × d_model.
    No inp.features used — backward is fast.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_move: int,
        activation: str,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Linear(_SPATIAL_IN, d_model)
        self.state_builder = TokenBuilder(
            d_model, _STATE_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.motion_builder = TokenBuilder(
            d_model, _MOTION_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.mlp = make_head_mlp(3 * d_model, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        acc = current_obs_accessor()
        dtype = inp.features.dtype
        spatial_raw = acc.dq["spatial_scalars"].to(dtype)
        spatial = self.spatial_proj(spatial_raw.reshape(spatial_raw.shape[0], -1))
        state  = self.state_builder(acc, dtype=dtype)
        motion = self.motion_builder(acc, dtype=dtype)
        logits = self.mlp(torch.cat([spatial, state, motion], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


class MotionWeaponTokenMoveHead(nn.Module):
    """Move head: MLP over cat(weapon_token, motion_token).

    weapon_token = Linear(weapon_static + held_readiness + attack_finished)(scalars)
                 + entity_embed(weapon_id)
    Captures held-weapon identity and readiness state.
    No inp.features used — backward is fast.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_move: int,
        activation: str,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        self.weapon_builder = TokenBuilder(
            d_model, _WEAPON_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.motion_builder = TokenBuilder(
            d_model, _MOTION_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.mlp = make_head_mlp(2 * d_model, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        acc = current_obs_accessor()
        dtype = inp.features.dtype
        weapon = self.weapon_builder(acc, dtype=dtype)
        motion = self.motion_builder(acc, dtype=dtype)
        logits = self.mlp(torch.cat([weapon, motion], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


class MotionSpatialTokenMoveHead(nn.Module):
    """Move head: MLP over cat(spatial_token, motion_token).

    spatial_token = Linear(9×13, d_model)(spatial_scalars.flatten())
    Captures map-geometry signal (clearance, openness, traversability).
    No inp.features used — backward is fast.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_move: int,
        activation: str,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Linear(_SPATIAL_IN, d_model)
        self.motion_builder = TokenBuilder(
            d_model, _MOTION_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        self.mlp = make_head_mlp(2 * d_model, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        acc = current_obs_accessor()
        dtype = inp.features.dtype
        spatial_raw = acc.dq["spatial_scalars"].to(dtype)          # (B*, 9, 13)
        spatial_token = self.spatial_proj(spatial_raw.reshape(spatial_raw.shape[0], -1))
        motion = self.motion_builder(acc, dtype=dtype)
        logits = self.mlp(torch.cat([spatial_token, motion], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


class MotionTokenTargetMoveHead(nn.Module):
    """Move head: MLP over cat(target_feat, motion_token)."""

    def __init__(
        self,
        *,
        d_model: int,
        d_move: int,
        activation: str,
        entity_embed: nn.Embedding,
        movement_embed: nn.Embedding,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.motion_builder = TokenBuilder(
            d_model, _MOTION_FIELDS,
            entity_embed=entity_embed,
            movement_embed=movement_embed,
        )
        # target_feat (d_model) + motion_token (d_model)
        self.mlp = make_head_mlp(2 * d_model, _OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        # inp.features layout (temporal=Off, target_pointer=GT):
        #   [self_readout (d_model) | target_feat (d_model)]
        # Detach: features_base_flat includes self_readout (grad), which would
        # pull backward through ObsEmbedding even though the gradient is zero.
        target_feat = inp.features[..., self.d_model:2 * self.d_model].detach()
        motion = self.motion_builder(current_obs_accessor(), dtype=inp.features.dtype)
        logits = self.mlp(torch.cat([target_feat, motion], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(f"probe.json must define {key!r} for head=move_motion_token")
    return probe[key]


def _build_move_motion_token(probe: Mapping[str, Any]) -> HeadBuildResult:
    d_model          = int(_required(probe, "d_model"))
    d_move           = int(_required(probe, "d_move"))
    activation       = str(_required(probe, "activation"))
    use_target_feat      = bool(probe.get("use_target_feat", False))
    use_state_token      = bool(probe.get("use_state_token", False))
    use_spatial_token    = bool(probe.get("use_spatial_token", False))
    use_weapon_token        = bool(probe.get("use_weapon_token", False))
    use_spatial_state       = bool(probe.get("use_spatial_state", False))
    use_state_no_weapon     = bool(probe.get("use_state_no_weapon", False))
    if sum([use_target_feat, use_state_token, use_spatial_token, use_weapon_token, use_spatial_state, use_state_no_weapon]) > 1:
        raise RuntimeError("probe.json: second-token knobs are mutually exclusive")

    model_config = neutral_model_config(d_model=d_model, self_weapon_embed_in_self=False)

    def factory(obs_dim: int, model_cfg: ModelConfig) -> nn.Module:
        obs_embedding = ObsEmbedding(
            d_model=d_model,
            self_weapon_embed_in_self=False,
            include_spatial=False,
        )
        if use_target_feat:
            move_head = MotionTokenTargetMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
            target_pointer = GTTargetPointer(d_model=d_model)
        elif use_state_token:
            move_head = MotionStateTokenMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
            target_pointer = Off
        elif use_state_no_weapon:
            move_head = MotionStateTokenMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
                state_fields=_STATE_NO_WEAPON_FIELDS,
            )
            target_pointer = Off
        elif use_weapon_token:
            move_head = MotionWeaponTokenMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
            target_pointer = Off
        elif use_spatial_token:
            move_head = MotionSpatialTokenMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
            target_pointer = Off
        elif use_spatial_state:
            move_head = MotionSpatialStateTokenMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
            target_pointer = Off
        else:
            move_head = MotionTokenMoveHead(
                d_model=d_model,
                d_move=d_move,
                activation=activation,
                entity_embed=obs_embedding.entity_embed,
                movement_embed=obs_embedding.movement_embed,
            )
            target_pointer = Off

        return BenchObsNetwork(
            obs_dim=obs_dim,
            model=model_cfg,
            obs_embedding=obs_embedding,
            encoder=PreAttnEncoder(),
            temporal=Off,
            target_pointer=target_pointer,
            move_head=move_head,
            look_head=Off,
            weapon_head=Off,
            attack_head=Off,
        )

    return model_config, factory


def _stub_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.zeros(())


def _stub_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {}


MOVE_MOTION_TOKEN = HeadSpec(
    name="move_motion_token",
    loss=HeadLossSpec(
        loss_fn=_stub_loss,
        metrics_fn=_stub_metrics,
        label_key="move",
        output_dim=_OUT_DIM,
        # Declarative only — live selection is _selection_score (qnn.bc.train),
        # which prefers move_dll (distributional, human-likeness) over f1_move.
        selection_metric="move_dll",
        selection_lower_is_better=False,
    ),
    build=_build_move_motion_token,
)
