"""Combat-objective BC model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import (
    MOVE_AXES,
    MOVE_AXIS_NAMES,
    MOVE_AXIS_CLASSES,
    MOVE_CLASS_NEG,
    MOVE_CLASS_NONE,
    MOVE_CLASS_POS,
)
from qnn.model.target import TargetPointer
from qnn.model.transformer import TransformerTrunk
from qnn.utils.device import configure_torch_runtime, resolve_torch_device
from qnn.utils.io import trusted_torch_load
from qnn.vocab import ENTITY_IDS, TOKEN_ACTOR


HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "target": 1.0,
    "move": 1.0,
    "look": 1.0,
    "fire": 1.0,
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Canonical model architecture config — sole source of truth for arch.

    All fields are required; defaults live only in model.json. The
    dataclass is frozen so a constructed config can't drift from its
    serialized form. ``head_activation`` is "none" or "gelu" (ReLU was
    removed). ``head_bottleneck_dim`` is per-head ({move,look,fire,weapon}
    → int); ``0`` disables the bottleneck for that head.
    """
    d_model: int
    n_heads: int
    n_layers: int
    ffn_dim: int
    attn_dropout: float
    use_gru: bool
    gru_hidden: int
    use_weapon_head: bool
    weapon_switch_confidence: float
    weapon_switch_margin: float
    weapon_use_gru: bool
    weapon_context_from_obs: bool
    look_bypass_gru: bool
    gru_target_query: bool
    hard_target_feat: bool
    weapon_in_target_query: bool
    linear_slot_prior: bool
    gt_dist_target_feat: bool
    prev_target_in_query: bool
    weapon_use_self_readout: bool
    self_weapon_embed_in_self: bool
    head_bottleneck_dim: "dict[str, int]"
    head_activation: str

    @classmethod
    def from_dict(cls, raw: "Mapping[str, Any]") -> "ModelConfig":
        """Build from a model.json-style mapping.

        Accepts ``head_bottleneck_dim`` as either an int (broadcast to all
        four heads) or a per-head dict. Strips the legacy ``trunk_hidden``
        alias of ``d_model``. Any other unknown key or any missing
        required field raises TypeError — every architectural flag must
        be set explicitly in model.json.
        """
        data = dict(raw)
        data.pop("trunk_hidden", None)
        hbd = data.get("head_bottleneck_dim")
        if isinstance(hbd, int):
            v = int(hbd)
            data["head_bottleneck_dim"] = {"move": v, "look": v, "fire": v, "weapon": v}
        elif isinstance(hbd, Mapping):
            data["head_bottleneck_dim"] = {k: int(v) for k, v in hbd.items()}
        if data.get("head_activation") not in ("none", "gelu", "relu"):
            raise ValueError(
                f"head_activation must be 'none', 'gelu', or 'relu', got {data.get('head_activation')!r}"
            )
        return cls(**data)

    @classmethod
    def from_flat_dict(cls, raw: "Mapping[str, Any]") -> "ModelConfig":
        """Like ``from_dict`` but extracts the model fields from a larger
        flat config dict (e.g. a PPO config that merges train + model
        keys). Missing required model fields still raise TypeError.
        """
        keys = {f.name for f in fields(cls)} | {"trunk_hidden"}
        subset = {k: v for k, v in raw.items() if k in keys}
        return cls.from_dict(subset)

    def to_dict(self) -> "dict[str, Any]":
        return asdict(self)

# Move head: 3 independent categorical axes (fb, lr, ud), each a 3-class
# softmax over {neg, none, pos}.  Implemented as one Linear(features, 9) and
# reshaped to (B, 3 axes, 3 classes); three cross-entropies, three argmax/
# sample decodes.  No priors.
MOVE_HEAD = "move"
MOVE_HEAD_SIZE = MOVE_AXES * MOVE_AXIS_CLASSES  # 9 logits
LOOK_HEAD = "look"
LOOK_HEAD_SIZE = 3  # 3D direction vector
FIRE_HEAD = "fire"
FIRE_HEAD_SIZE = 1  # binary logit
WEAPON_HEAD = "weapon"
# WEAPON_HEAD_SIZE = 8 lives in qnn.schema (canonical home alongside
# SELF_SCALAR_DIM); re-exported here for back-compat with callers that
# import it from qnn.model.policy.
from qnn.schema import WEAPON_HEAD_SIZE  # noqa: F401  re-export
WEAPON_HEAD_CLASS_NAMES: Tuple[Tuple[int, str], ...] = (
    (0, "axe"),
    (1, "shotgun"),
    (2, "super_shotgun"),
    (3, "nailgun"),
    (4, "super_nailgun"),
    (5, "grenade_launcher"),
    (6, "rocket_launcher"),
    (7, "thunderbolt"),
)
# Offset of the relative-XYZ block inside an actor's per-token scalar vector.
# Mirrors qnn.bc.target_labeler._ACTOR_REL_OFFSET; duplicated here so the model
# layer doesn't import from BC.
_ACTOR_REL_OFFSET = 3


@dataclass(slots=True)
class PolicyActionBatch:
    actions: Dict[str, np.ndarray]
    log_probs: torch.Tensor
    values: torch.Tensor
    entropies: Dict[str, torch.Tensor]
    next_hidden: torch.Tensor


class _CombatObjectiveNet(nn.Module):
    def __init__(self, obs_dim: int, model: ModelConfig) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.config = model
        self.d_model = int(model.d_model)
        self.trunk = TransformerTrunk(
            obs_dim=obs_dim,
            d_model=int(model.d_model),
            n_heads=int(model.n_heads),
            n_layers=int(model.n_layers),
            ffn_dim=int(model.ffn_dim),
            dropout=float(model.attn_dropout),
            self_weapon_embed_in_self=bool(model.self_weapon_embed_in_self),
        )
        self.use_gru = bool(model.use_gru and model.gru_hidden > 0)
        self.gru_hidden = int(model.gru_hidden) if self.use_gru else 0
        self.use_weapon_head = bool(model.use_weapon_head)
        # look_bypass_gru is a v17-fidelity load-time flag.  v20+ always sets
        # this False — when True (only via QNNPolicy.load on a v17 checkpoint)
        # the look head is fed cat(self_readout, target_feat) instead of
        # cat(gru_flat, target_feat), matching the features it was trained on.
        self.look_bypass_gru = bool(model.look_bypass_gru and self.use_gru)
        self.weapon_use_gru = bool(model.weapon_use_gru and self.use_gru)
        # Weapon head selector composition is per-flag: at least one of
        # {gru_flat, self_readout} must be present (target_feat is always
        # included). Default (true, true) preserves the historical
        # [gru_flat, self_readout, target_feat] layout.
        self.weapon_use_self_readout = bool(model.weapon_use_self_readout)
        if not (self.weapon_use_gru or self.weapon_use_self_readout):
            raise ValueError(
                "weapon head needs at least one of weapon_use_gru or "
                "weapon_use_self_readout — got both False (target_feat "
                "alone is too thin)"
            )
        self.weapon_context_from_obs = bool(model.weapon_context_from_obs)
        self.gru_target_query = bool(model.gru_target_query and self.use_gru)
        # hard_target_feat: target_feat is the entity vector at a single chosen
        # slot (BC GT during training, argmax at eval) instead of a soft pool.
        # Decouples target-head loss tuning from motor-head training distribution.
        self.hard_target_feat = bool(model.hard_target_feat)
        self.weapon_in_target_query = bool(model.weapon_in_target_query and self.use_gru)
        self.linear_slot_prior = bool(model.linear_slot_prior and self.use_gru)
        # gt_dist_target_feat: training-time STE that pools target_feat by the
        # labeler's GT slot distribution in the forward and routes gradient
        # through softmax(logits) on the backward. Motors see clean target
        # context; the pointer still trains from motor gradient. Gated on
        # self.training inside TargetPointer so val/eval/PPO use soft.
        self.gt_dist_target_feat = bool(model.gt_dist_target_feat and self.use_gru)
        # prev_target_in_query: concat previous-frame renormalized slot
        # distribution (16 floats) to the target pointer's query input. At
        # BC train the caller passes the GT prev dist (privileged); at
        # eval the caller passes None and TargetPointer substitutes zeros
        # (probe-style train/eval forward mismatch — see target docstring).
        self.prev_target_in_query = bool(model.prev_target_in_query and self.use_gru)
        self.head_bottleneck_dims = dict(model.head_bottleneck_dim)
        self.head_activation = model.head_activation
        weapon_ctx_dim = self.d_model if self.use_weapon_head else 0
        # v20-settled GRU layout:
        #   GRU input  = self_readout (no mean-actors pool projection)
        #   target_pointer queries with self_readout directly
        #   move/look/fire features = cat(gru_flat, target_feat, weapon_context)
        #   look features have v17 (look_bypass_gru) variant that keeps
        #     cat(self_readout, target_feat) for inference fidelity.
        base_features_dim = (self.gru_hidden + self.d_model) if self.use_gru else (2 * self.d_model)
        move_in = base_features_dim + weapon_ctx_dim
        look_in = base_features_dim + weapon_ctx_dim
        fire_in = base_features_dim + weapon_ctx_dim
        weapon_in = (
            (self.gru_hidden if self.weapon_use_gru else 0)
            + (self.d_model if self.weapon_use_self_readout else 0)
            + self.d_model  # target_feat is always present
        )
        if self.use_gru:
            self.gru = nn.GRU(self.d_model, self.gru_hidden, batch_first=False)
            self.target_pointer = TargetPointer(
                d_model=self.d_model,
                query_in_dim=self.gru_hidden if self.gru_target_query else self.d_model,
                inject_weapon=self.weapon_in_target_query,
                weapon_vocab=WEAPON_HEAD_SIZE,
                hard_target=self.hard_target_feat,
                linear_slot_prior=self.linear_slot_prior,
                gt_dist_target_feat=self.gt_dist_target_feat,
                prev_target_in_query=self.prev_target_in_query,
            )
        if self.use_weapon_head:
            self.weapon_head = self._make_head(weapon_in, WEAPON_HEAD_SIZE, self.head_bottleneck_dims["weapon"])
            self.weapon_embed = nn.Embedding(WEAPON_HEAD_SIZE, self.d_model)
        self.move_head = self._make_head(move_in, MOVE_HEAD_SIZE, self.head_bottleneck_dims["move"])
        # Look head outputs a residual added to a target-anchored prior:
        #   base_look  = normalize(soft_target_rel)
        #   delta_look = look_head(features)
        #   pred_look  = normalize(base_look + delta_look)
        self.look_head = self._make_head(look_in, LOOK_HEAD_SIZE, self.head_bottleneck_dims["look"])
        self.fire_head = self._make_head(fire_in, FIRE_HEAD_SIZE, self.head_bottleneck_dims["fire"])
        self._init_weights()

    def _make_head(self, in_dim: int, out_dim: int, bottleneck_dim: int) -> nn.Module:
        """Build an output head.

        head_activation is "none", "gelu", or "relu" (relu kept for v22-era
        legacy checkpoint compatibility — current training defaults to gelu).
          (B>0, "none") → Linear(in, B) → Linear(B, out)
          (B>0, act)    → Linear(in, B) → act → Linear(B, out)
          (0,   "none") → Linear(in, out)
          (0,   act)    → Linear(in, in) → act → Linear(in, out)
        """
        hidden = bottleneck_dim if bottleneck_dim > 0 else in_dim
        activations = {"gelu": nn.GELU, "relu": nn.ReLU}
        has_activation = self.head_activation in activations
        if bottleneck_dim > 0 or has_activation:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
            if has_activation:
                layers.append(activations[self.head_activation]())
            layers.append(nn.Linear(hidden, out_dim))
            return nn.Sequential(*layers)
        return nn.Linear(in_dim, out_dim)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _compose_look(
        features_flat: torch.Tensor,
        target_logits_flat: torch.Tensor,
        entity_rel_flat: torch.Tensor,
        entity_actor_mask_flat: torch.Tensor,
        look_head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (pred_look, base_look, delta_look) per frame.

        Args are all flattened to 2-D batches:
          features_flat:           (B*, head_in)
          target_logits_flat:      (B*, N)         -- pre-masked by TargetPointer
          entity_rel_flat:         (B*, N, 3)      -- relative XYZ from entity_scalars_raw
          entity_actor_mask_flat:  (B*, N) bool

        Returns:
          pred_look (B*, 3) unit-normalized — used at inference / PPO
          base_look (B*, 3) unit-normalized prior (zero for empty scenes)
          delta_look (B*, 3) raw head output — used for the regression loss
        """
        # Soft-attended target relative position; target_logits already has -1e9
        # on non-actor slots so softmax is implicitly actor-only.
        probs = F.softmax(target_logits_flat, dim=-1)                    # (B*, N)
        soft_target_rel = (probs.unsqueeze(-1) * entity_rel_flat).sum(dim=-2)  # (B*, 3)

        has_actor = entity_actor_mask_flat.any(dim=-1, keepdim=True).to(soft_target_rel.dtype)
        soft_target_rel = soft_target_rel * has_actor

        soft_norm = torch.linalg.vector_norm(soft_target_rel, dim=-1, keepdim=True).clamp(min=1e-6)
        base_look = soft_target_rel / soft_norm

        delta_look = look_head(features_flat)                             # (B*, 3)
        unnormalized = base_look + delta_look
        out_norm = torch.linalg.vector_norm(unnormalized, dim=-1, keepdim=True).clamp(min=1e-6)
        pred_look = unnormalized / out_norm
        return pred_look, base_look, delta_look

    @staticmethod
    def _compose_move_categorical(
        features_flat: torch.Tensor,
        move_head: nn.Module,
    ) -> torch.Tensor:
        """Compose categorical move logits (3 axes × 3 classes).

        Returns logits of shape (B*, 3 axes, 3 classes).
        Axis order: 0=fb, 1=lr, 2=ud.  Class order: 0=neg, 1=none, 2=pos.
        """
        return move_head(features_flat).reshape(
            -1, MOVE_AXES, MOVE_AXIS_CLASSES
        )

    @staticmethod
    def _weapon_choices_from_ids(weapon_ids: torch.Tensor) -> torch.Tensor:
        wid = weapon_ids.long()
        slots = torch.zeros_like(wid)
        slots = torch.where(wid == ENTITY_IDS["AXE"], torch.full_like(slots, 0), slots)
        slots = torch.where(wid == ENTITY_IDS["SHOTGUN"], torch.full_like(slots, 1), slots)
        slots = torch.where(wid == ENTITY_IDS["SUPER_SHOTGUN"], torch.full_like(slots, 2), slots)
        slots = torch.where(wid == ENTITY_IDS["NAILGUN"], torch.full_like(slots, 3), slots)
        slots = torch.where(wid == ENTITY_IDS["SUPER_NAILGUN"], torch.full_like(slots, 4), slots)
        slots = torch.where(wid == ENTITY_IDS["GRENADE_LAUNCHER"], torch.full_like(slots, 5), slots)
        slots = torch.where(wid == ENTITY_IDS["ROCKET_LAUNCHER"], torch.full_like(slots, 6), slots)
        slots = torch.where(wid == ENTITY_IDS["THUNDERBOLT"], torch.full_like(slots, 7), slots)
        return slots

    def _weapon_context(
        self,
        weapon_logits: torch.Tensor | None,
        obs_weapon_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_weapon_head:
            return None
        if self.weapon_context_from_obs:
            assert obs_weapon_ids is not None, "weapon_context_from_obs=True requires self_weapon_id in obs"
            slots = self._weapon_choices_from_ids(obs_weapon_ids.reshape(-1)).clamp(0, WEAPON_HEAD_SIZE - 1)
            return self.weapon_embed(slots)
        assert weapon_logits is not None
        probs = F.softmax(weapon_logits, dim=-1)
        return probs @ self.weapon_embed.weight

    @staticmethod
    def _with_weapon_context(
        features: torch.Tensor,
        weapon_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if weapon_context is None:
            return features
        return torch.cat([features, weapon_context], dim=-1)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        target_gt: torch.Tensor | None = None,
        target_dist_slot: torch.Tensor | None = None,
        prev_target_dist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use `vel` to detect (B, 3) flat vs (T, B, 3) sequence — same
        # ndim semantics as the legacy self_scalars (B, 17) / (T, B, 17).
        sample = obs.get("vel")
        if sample is None:
            sample = obs["self_scalars"]  # legacy fallback
        input_is_sequence = sample.ndim == 3
        if input_is_sequence:
            seq_len = int(sample.shape[0])
            batch_size = int(sample.shape[1])
            flat_obs = {
                key: value.reshape(seq_len * batch_size, *value.shape[2:])
                for key, value in obs.items()
            }
            if self.use_gru:
                self_readout, entity_outs, entity_mask = self.trunk.forward_raw(flat_obs)
                pool_seq = self_readout.reshape(seq_len, batch_size, self.d_model)
                h0 = self._initial_hidden(hidden, batch_size, dtype=pool_seq.dtype, device=pool_seq.device)
                if reset_mask is None:
                    gru_out_seq, h_final = self.gru(pool_seq, h0)
                else:
                    reset_seq = reset_mask.to(device=pool_seq.device, dtype=torch.bool).reshape(seq_len, batch_size)
                    h = h0
                    outs = []
                    for t in range(seq_len):
                        reset_t = reset_seq[t].view(1, batch_size, 1)
                        h = h.masked_fill(reset_t, 0.0)
                        out_t, h = self.gru(pool_seq[t:t + 1], h)
                        outs.append(out_t)
                    gru_out_seq = torch.cat(outs, dim=0)
                    h_final = h
                gru_flat = gru_out_seq.reshape(seq_len * batch_size, self.gru_hidden)
                tp_query_input = gru_flat if self.gru_target_query else self_readout
                tp_weapon_slot = None
                if self.weapon_in_target_query:
                    tp_weapon_slot = self._weapon_choices_from_ids(
                        flat_obs["self_weapon_id"].reshape(-1)
                    ).clamp(0, WEAPON_HEAD_SIZE - 1)
                tp_target_gt = target_gt.reshape(-1) if target_gt is not None else None
                tp_target_dist_slot = (
                    target_dist_slot.reshape(-1, target_dist_slot.shape[-1])
                    if target_dist_slot is not None else None
                )
                tp_prev_target_dist = (
                    prev_target_dist.reshape(-1, prev_target_dist.shape[-1])
                    if prev_target_dist is not None else None
                )
                target_logits, target_feat, target_query = self.target_pointer(
                    tp_query_input,
                    entity_outs,
                    entity_mask,
                    self_weapon_slot=tp_weapon_slot,
                    target_gt=tp_target_gt,
                    target_dist_slot=tp_target_dist_slot,
                    prev_target_dist=tp_prev_target_dist,
                )
                features_base_flat = torch.cat([gru_flat, target_feat], dim=-1)
                # Weapon selector composition gated by weapon_use_gru and
                # weapon_use_self_readout independently. target_feat always
                # present; at least one of gru_flat/self_readout (guarded
                # in __init__).
                _ws_parts = []
                if self.weapon_use_gru:
                    _ws_parts.append(gru_flat)
                if self.weapon_use_self_readout:
                    _ws_parts.append(self_readout)
                _ws_parts.append(target_feat)
                weapon_selector_flat = torch.cat(_ws_parts, dim=-1)
                next_hidden = h_final.squeeze(0)
            else:
                self_readout, target_feat, target_logits, target_query = self.trunk(flat_obs)
                features_base_flat = torch.cat([self_readout, target_feat], dim=-1)
                weapon_selector_flat = features_base_flat
                next_hidden = torch.zeros((batch_size, 0), dtype=sample.dtype, device=sample.device)
            n_targets = int(target_logits.shape[-1])
            # weapon_context source for the motor heads:
            #   weapon_context_from_obs=True  → embed(currently-held weapon from obs)
            #   weapon_context_from_obs=False → softmax(weapon_logits) @ weapon_embed.weight
            # The weapon_head still trains via its own CE loss in either mode.
            obs_weapon_ids_flat = (
                flat_obs["self_weapon_id"] if self.use_weapon_head and self.weapon_context_from_obs else None
            )
            weapon_logits = self.weapon_head(weapon_selector_flat) if self.use_weapon_head else None
            weapon_context = self._weapon_context(weapon_logits, obs_weapon_ids_flat)
            move_features_flat = self._with_weapon_context(features_base_flat, weapon_context)
            entity_scalars_flat = flat_obs["entity_scalars_raw"]            # (T*B, N, ACTOR_SCALAR_DIM)
            entity_rel_flat = entity_scalars_flat[..., _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET + 3]
            entity_types_flat = flat_obs["entity_types"].long()
            actor_mask_flat = (entity_types_flat == TOKEN_ACTOR)
            move_flat = self._compose_move_categorical(move_features_flat, self.move_head)
            # Look = target-anchored prior + learned residual.  v17 checkpoints
            # set look_bypass_gru=True so the look head sees the same features
            # it was trained on (cat(self_readout, target_feat)).
            if self.use_gru and self.look_bypass_gru:
                look_features_flat = self._with_weapon_context(
                    torch.cat([self_readout, target_feat], dim=-1), weapon_context,
                )
            else:
                look_features_flat = self._with_weapon_context(features_base_flat, weapon_context)
            look_flat, base_look_flat, delta_look_flat = self._compose_look(
                look_features_flat, target_logits, entity_rel_flat, actor_mask_flat, self.look_head,
            )
            fire_features_flat = self._with_weapon_context(features_base_flat, weapon_context)
            fire_flat = self.fire_head(fire_features_flat)
            features_flat = move_features_flat  # for return signature; downstream consumers ignore weapon dim
            features = features_flat.reshape(seq_len, batch_size, -1)
            logits = {
                MOVE_HEAD: move_flat.reshape(seq_len, batch_size, MOVE_AXES, MOVE_AXIS_CLASSES),
                LOOK_HEAD: look_flat.reshape(seq_len, batch_size, LOOK_HEAD_SIZE),
                # Underscored keys are loss-only; not used for inference / sampling.
                "_look_base":   base_look_flat.reshape(seq_len, batch_size, LOOK_HEAD_SIZE),
                "_look_delta":  delta_look_flat.reshape(seq_len, batch_size, LOOK_HEAD_SIZE),
                FIRE_HEAD: fire_flat.reshape(seq_len, batch_size, FIRE_HEAD_SIZE),
            }
            if weapon_logits is not None:
                logits[WEAPON_HEAD] = weapon_logits.reshape(seq_len, batch_size, WEAPON_HEAD_SIZE)
            values = torch.zeros((seq_len, batch_size), dtype=sample.dtype, device=sample.device)
            target_logits = target_logits.reshape(seq_len, batch_size, n_targets)
            target_query = target_query.reshape(seq_len, batch_size, self.d_model)
            return features, logits, values, next_hidden, target_logits, target_query

        batch_size = int(sample.shape[0])
        if self.use_gru:
            self_readout, entity_outs, entity_mask = self.trunk.forward_raw(obs)
            h0 = self._initial_hidden(hidden, batch_size, dtype=self_readout.dtype, device=self_readout.device)
            # Single-step GRU: feed (1, batch, D), strip the seq dim out.
            gru_step, h_final = self.gru(self_readout.unsqueeze(0), h0)
            recurrent = gru_step.squeeze(0)
            tp_query_input = recurrent if self.gru_target_query else self_readout
            tp_weapon_slot = None
            if self.weapon_in_target_query:
                tp_weapon_slot = self._weapon_choices_from_ids(
                    obs["self_weapon_id"].reshape(-1)
                ).clamp(0, WEAPON_HEAD_SIZE - 1)
            tp_target_gt = target_gt.reshape(-1) if target_gt is not None else None
            tp_target_dist_slot = (
                target_dist_slot.reshape(-1, target_dist_slot.shape[-1])
                if target_dist_slot is not None else None
            )
            tp_prev_target_dist = (
                prev_target_dist.reshape(-1, prev_target_dist.shape[-1])
                if prev_target_dist is not None else None
            )
            target_logits, target_feat, target_query = self.target_pointer(
                tp_query_input,
                entity_outs,
                entity_mask,
                self_weapon_slot=tp_weapon_slot,
                target_gt=tp_target_gt,
                target_dist_slot=tp_target_dist_slot,
                prev_target_dist=tp_prev_target_dist,
            )
            features_base = torch.cat([recurrent, target_feat], dim=-1)
            _ws_parts = []
            if self.weapon_use_gru:
                _ws_parts.append(recurrent)
            if self.weapon_use_self_readout:
                _ws_parts.append(self_readout)
            _ws_parts.append(target_feat)
            weapon_selector = torch.cat(_ws_parts, dim=-1)
            next_hidden = h_final.squeeze(0)
        else:
            self_readout, target_feat, target_logits, target_query = self.trunk(obs)
            features_base = torch.cat([self_readout, target_feat], dim=-1)
            weapon_selector = features_base
            next_hidden = torch.zeros((batch_size, 0), dtype=sample.dtype, device=sample.device)
        obs_weapon_ids = (
            obs["self_weapon_id"] if self.use_weapon_head and self.weapon_context_from_obs else None
        )
        weapon_logits = self.weapon_head(weapon_selector) if self.use_weapon_head else None
        weapon_context = self._weapon_context(weapon_logits, obs_weapon_ids)
        move_features = self._with_weapon_context(features_base, weapon_context)
        # Look = target-anchored prior + learned residual.  v17 fidelity
        # path mirrors the seq case above.
        entity_rel = obs["entity_scalars_raw"][..., _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET + 3]
        actor_mask = (obs["entity_types"].long() == TOKEN_ACTOR)
        if self.use_gru and self.look_bypass_gru:
            look_features = self._with_weapon_context(
                torch.cat([self_readout, target_feat], dim=-1), weapon_context,
            )
        else:
            look_features = self._with_weapon_context(features_base, weapon_context)
        pred_look, base_look, delta_look = self._compose_look(
            look_features, target_logits, entity_rel, actor_mask, self.look_head,
        )
        fire_features = self._with_weapon_context(features_base, weapon_context)
        fire_logit = self.fire_head(fire_features)
        # Move = 3 categorical axes (no priors).
        move_logits = self._compose_move_categorical(move_features, self.move_head)
        features = move_features  # for return signature
        logits = {
            MOVE_HEAD: move_logits,
            LOOK_HEAD: pred_look,
            "_look_base":    base_look,
            "_look_delta":   delta_look,
            FIRE_HEAD: fire_logit,
        }
        if weapon_logits is not None:
            logits[WEAPON_HEAD] = weapon_logits
        values = torch.zeros((batch_size,), dtype=sample.dtype, device=sample.device)
        return features, logits, values, next_hidden, target_logits, target_query

    def _initial_hidden(
        self,
        hidden: torch.Tensor | None,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if hidden is None:
            return torch.zeros((1, batch_size, self.gru_hidden), dtype=dtype, device=device)
        h = hidden.to(device=device, dtype=dtype)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        return h.contiguous()


class QNNPolicy:
    """Feed-forward combat-objective model for BC."""

    def __init__(
        self,
        *,
        obs_dim: int,
        model: ModelConfig,
        jump_pos_weight: float,
        fire_focal_gamma: float,
        fire_focal_alpha: float,
        fire_distance_sigma: float,
        jump_distance_sigma: float,
        seed: int,
        device: str,
        model_factory: Callable[[int, ModelConfig], nn.Module] | None = None,
    ) -> None:
        """Construct a BC policy.

        ``model_factory`` is an optional override for the inner ``nn.Module``:
        when ``None`` (the default, used by all production BC training) the
        canonical ``_CombatObjectiveNet`` is built from ``model``. Ablation
        runners (e.g. ``qnn.bc.heads``) pass a factory that builds an
        alternate module — typically one that drops the trunk or GRU — but
        the factory must respect ``_CombatObjectiveNet``'s forward contract
        so the canonical BC supervised loop can drive it unchanged.

        The injected module's flags should still be consistent with
        ``model`` (use_gru / use_weapon_head / etc.) since QNNPolicy's
        policy-layer logic — hidden-state shaping, weapon-switch heuristics,
        head-loss gating — reads from ``model``, not from the module.
        """
        self.obs_dim = int(obs_dim)
        self.config = model
        self.d_model = int(model.d_model)
        self.use_gru = bool(model.use_gru and model.gru_hidden > 0)
        self.gru_hidden = int(model.gru_hidden) if self.use_gru else 0
        self.use_weapon_head = bool(model.use_weapon_head)
        self.look_bypass_gru = bool(model.look_bypass_gru and self.use_gru)
        self.weapon_switch_confidence = float(model.weapon_switch_confidence)
        self.weapon_switch_margin = float(model.weapon_switch_margin)
        self.weapon_use_gru = bool(model.weapon_use_gru)
        self.weapon_use_self_readout = bool(model.weapon_use_self_readout)
        self.weapon_context_from_obs = bool(model.weapon_context_from_obs)
        self.gru_target_query = bool(model.gru_target_query and self.use_gru)
        self.hard_target_feat = bool(model.hard_target_feat)
        self.weapon_in_target_query = bool(model.weapon_in_target_query and self.use_gru)
        self.linear_slot_prior = bool(model.linear_slot_prior and self.use_gru)
        self.gt_dist_target_feat = bool(model.gt_dist_target_feat and self.use_gru)
        self.prev_target_in_query = bool(model.prev_target_in_query and self.use_gru)
        self.head_bottleneck_dims = dict(model.head_bottleneck_dim)
        self.head_activation = model.head_activation
        # jump_pos_weight > 1.0 upweights the POS class on the move ud-axis CE
        # — direct imbalance fix for the rare jump-positive case (~4% pos rate).
        # Inverse-frequency reference: ~24× for 4% positive rate.
        self.jump_pos_weight = float(jump_pos_weight)
        # fire_focal_gamma > 0 swaps the fire-head BCE for focal BCE
        # (Lin et al. 2017): each frame's BCE is multiplied by
        # (1 - p_t)^gamma so easy examples contribute less gradient and
        # capacity flows to the borderline ready-frame "fire or wait?"
        # decisions. 0 = standard BCE.
        self.fire_focal_gamma = float(fire_focal_gamma)
        # fire_focal_alpha is Lin's per-class prefactor on the focal weight:
        # alpha_t = alpha on positives, (1 - alpha) on negatives. Active
        # only when fire_focal_gamma > 0. Default 0.5 is neutral (both
        # classes weighted equally up to a global scale). To run the Lin
        # recipe end-to-end set fire_pos_weight_override=1.0 alongside —
        # otherwise pos_weight stacks multiplicatively on the positive
        # branch and alpha loses its canonical class-fraction meaning.
        self.fire_focal_alpha = float(fire_focal_alpha)
        # op_input_mask is a training-time attribute (NOT a ModelConfig
        # field — checkpoint meta stays clean and the same ckpt can be
        # retrained either way). Trainer sets this to True after
        # construction when train.json.op_input_mask is true. Read by
        # ``_compute_head_losses_and_metrics`` to drop frames where the
        # demo held a press but the engine ignored it.
        self.op_input_mask: bool = False
        # fire_distance_sigma > 0 enables Gaussian-shouldered BCE on the
        # fire head: per-frame BCE is multiplied by 1 at positives and by
        # 1 - exp(-d^2/(2*sigma^2)) at negatives, where d is distance (in
        # frames) to the nearest positive. Adjacent-to-press FPs cost
        # near-zero loss; far-from-press FPs cost full loss. Inference is
        # unchanged. See src/qnn/bc/heads/loss_shaping.py. 0 = standard BCE.
        self.fire_distance_sigma = float(fire_distance_sigma)
        # Same shoulder applied to the move ud-axis (jump) CE. Tuned
        # independently of fire because jump-press timing noise has a
        # different scale.
        self.jump_distance_sigma = float(jump_distance_sigma)
        self.n_heads = int(model.n_heads)
        self.n_layers = int(model.n_layers)
        self.ffn_dim = int(model.ffn_dim)
        self.attn_dropout = float(model.attn_dropout)
        self.head_hidden = (self.gru_hidden + self.d_model) if self.use_gru else (2 * self.d_model)
        self.seed = int(seed)
        self.device_spec = resolve_torch_device(device)
        configure_torch_runtime(self.device_spec)
        self.device = self.device_spec.device
        self._rocm_inference_pad_batch = 0
        if self.device_spec.backend == "rocm":
            raw_pad_batch = os.environ.get("QNN_ROCM_INFERENCE_PAD_BATCH", "32").strip()
            try:
                self._rocm_inference_pad_batch = max(int(raw_pad_batch), 0)
            except ValueError:
                self._rocm_inference_pad_batch = 32

        torch.manual_seed(self.seed)
        if model_factory is None:
            self.model = _CombatObjectiveNet(obs_dim=self.obs_dim, model=model).to(self.device)
        else:
            built = model_factory(self.obs_dim, model)
            if not isinstance(built, nn.Module):
                raise TypeError(
                    f"model_factory must return nn.Module, got {type(built).__name__}"
                )
            self.model = built.to(self.device)
        self.model.train()
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}

    def zero_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.gru_hidden), dtype=np.float32)

    def _tensor(self, value: np.ndarray | torch.Tensor | Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device == self.device and value.dtype == dtype:
                return value
            non_blocking = value.device.type == "cpu" and (
                not isinstance(self.device, torch.device) or self.device.type != "cpu"
            )
            return value.to(device=self.device, dtype=dtype, non_blocking=non_blocking)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _autocast(self):
        dtype_name = os.environ.get("QNN_AUTOCAST_DTYPE", "fp32").lower()
        if dtype_name == "fp32" or self.device.type != "cuda":
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_name)
        if dtype is None:
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        return torch.amp.autocast(device_type=self.device.type, dtype=dtype, enabled=True)

    def _class_weights_for_head(
        self,
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        head: str,
        size: int,
    ) -> torch.Tensor:
        source = class_weights.get(head)
        if source is None:
            return torch.ones((size,), dtype=torch.float32, device=self.device)
        cache = getattr(self, "_class_weights_cache", None)
        if cache is None:
            cache = {}
            self._class_weights_cache = cache
        key = (head, id(source))
        cached = cache.get(key)
        if cached is not None:
            return cached
        tensor = self._tensor(source, dtype=torch.float32)
        cache[key] = tensor
        return tensor

    @staticmethod
    def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            return logits.reshape(-1, logits.shape[-1])
        return logits

    @staticmethod
    def _flatten_targets(target: torch.Tensor) -> torch.Tensor:
        if target.ndim > 1:
            return target.reshape(-1)
        return target

    def _maybe_pad_obs_batch(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], int]:
        # `vel` matches the old `self_scalars` ndim semantics: (B, 3)
        # flat, (B, T, 3) sequence. Native obs replaced self_scalars
        # as a single-field key with per-field arrays.
        sample = obs_dict.get("vel")
        if sample is None:
            sample = obs_dict["self_scalars"]  # legacy fallback
        if sample.ndim != 2 or self._rocm_inference_pad_batch <= 0:
            return obs_dict, 0

        batch_size = int(sample.shape[0])
        target_batch = self._rocm_inference_pad_batch
        if batch_size == 0 or batch_size >= target_batch:
            return obs_dict, 0

        pad_rows = target_batch - batch_size
        padded_obs: Dict[str, torch.Tensor] = {}
        for key, value in obs_dict.items():
            pad_shape = (pad_rows, *value.shape[1:])
            pad_value = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            padded_obs[key] = torch.cat([value, pad_value], dim=0)
        return padded_obs, pad_rows

    @staticmethod
    def _pad_companion(
        tensor: torch.Tensor | None, pad_rows: int,
    ) -> torch.Tensor | None:
        """Zero-pad a companion tensor along dim 0 to match obs padding.

        Used after ``_maybe_pad_obs_batch`` to extend hidden state and
        per-frame supervision tensors (target_gt, target_dist_slot,
        prev_target_dist) so callers that pass any of them on ROCm with
        small batches don't hit a B mismatch inside heads that consume
        them as features (e.g. fire-token head probe's
        ``target_dist_slots`` feature builder). Pass-through when the
        tensor is None or no padding was applied.
        """
        if tensor is None or pad_rows <= 0:
            return tensor
        pad_shape = (pad_rows, *tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

    def _forward_tensors(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        target_gt: np.ndarray | torch.Tensor | None = None,
        target_dist_slot: np.ndarray | torch.Tensor | None = None,
        prev_target_dist: np.ndarray | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(obs, dict):
            raise ValueError("Token policy expects dict observations")

        obs_tensors: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            dtype = torch.float32
            if key.endswith("_id") or key.endswith("_ids"):
                dtype = torch.long
            elif key.endswith("_mask"):
                dtype = torch.bool
            obs_tensors[key] = self._tensor(value, dtype=dtype)

        hidden_tensor: torch.Tensor | None = None
        if self.use_gru and hidden is not None:
            hidden_tensor = self._tensor(hidden, dtype=torch.float32)

        target_gt_tensor: torch.Tensor | None = None
        if target_gt is not None:
            target_gt_tensor = self._tensor(target_gt, dtype=torch.long)

        target_dist_slot_tensor: torch.Tensor | None = None
        if target_dist_slot is not None:
            target_dist_slot_tensor = self._tensor(target_dist_slot, dtype=torch.float32)

        prev_target_dist_tensor: torch.Tensor | None = None
        if prev_target_dist is not None:
            prev_target_dist_tensor = self._tensor(prev_target_dist, dtype=torch.float32)

        # Use `vel` to detect flat-batch (B, 3) vs sequence (B, T, 3).
        # The legacy obs carried `self_scalars` (B, 17) here; the native
        # obs has per-field arrays, with vel matching the same ndim
        # semantics (2D flat, 3D sequence).
        sample = obs_tensors.get("vel")
        if sample is None:
            sample = obs_tensors["self_scalars"]  # legacy fallback
        if sample.ndim == 2:
            batch_size = int(sample.shape[0])
            padded_obs, pad_rows = self._maybe_pad_obs_batch(obs_tensors)
            padded_hidden = (
                self._pad_companion(hidden_tensor, pad_rows)
                if self.use_gru else hidden_tensor
            )
            features, logits, values, next_hidden, target_logits, target_query = self.model(
                padded_obs,
                padded_hidden,
                target_gt=self._pad_companion(target_gt_tensor, pad_rows),
                target_dist_slot=self._pad_companion(target_dist_slot_tensor, pad_rows),
                prev_target_dist=self._pad_companion(prev_target_dist_tensor, pad_rows),
            )
            if pad_rows == 0:
                return features, logits, values, next_hidden, target_logits, target_query
            return (
                features[:batch_size],
                {head: tensor[:batch_size] for head, tensor in logits.items()},
                values[:batch_size],
                next_hidden[:batch_size],
                target_logits[:batch_size],
                target_query[:batch_size],
            )

        if sample.ndim != 3:
            raise ValueError("obs must be rank-2 or rank-3")
        reset_mask_tensor = None
        if isinstance(masks, Mapping) and "reset_mask" in masks:
            reset_mask_tensor = self._tensor(masks["reset_mask"], dtype=torch.bool)
        return self.model(
            obs_tensors,
            hidden_tensor,
            reset_mask=reset_mask_tensor,
            target_gt=target_gt_tensor,
            target_dist_slot=target_dist_slot_tensor,
            prev_target_dist=prev_target_dist_tensor,
        )

    def _optimizer(self, name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer = self._optimizers.get(name)
        if optimizer is None:
            optimizer = torch.optim.Adam(list(params), lr=lr, fused=True)
            self._optimizers[name] = optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        return optimizer

    def bc_zero_grad(self) -> None:
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.zero_grad()

    def bc_step(self) -> None:
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.step()

    def encode(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, _, _, next_hidden, _, _ = self._forward_tensors(obs, hidden=hidden)
        return (
            features.detach().cpu().numpy().astype(np.float32),
            next_hidden.detach().cpu().numpy().astype(np.float32),
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, logits_t, values_t, next_hidden, _, _ = self._forward_tensors(obs, hidden=hidden)
        logits = {head: tensor.detach().cpu().numpy().astype(np.float32) for head, tensor in logits_t.items()}
        values = values_t.detach().cpu().numpy().astype(np.float32)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        return logits, values, next_hidden.detach().cpu().numpy().astype(np.float32), features_np

    def act(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        mode: str,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        row_generators: Any | None = None,
        sample_temperatures: Mapping[str, float] | None = None,
        diag_log_path: str | Path | None = None,
    ) -> PolicyActionBatch:
        """Emit engine actions from a forward pass.

        Output dict shape matches the engine's action contract (see
        qnn.actions.ActionLabels):
          move   : (B, 3) float in [-1, 1] — view-relative wishvel/maxspeed.
                   Categorical mode argmaxes/samples each axis to {-1, 0, +1};
                   continuous mode passes the regression output through clamp.
                   Up axis is 0 (no jump head).
          look   : (B, 3) float — pred_look unit vector from the head.
          fire   : (B,)   int   — 0/1 from sigmoid(logit) threshold or bernoulli.
          weapon : (B,)   int   — engine weapon byte 1..8 (or 0 = no switch).

        log_probs / values / entropies are placeholders for shape compatibility
        with the action batch consumer; greedy/sampled eval doesn't read them.

        When *diag_log_path* is set, append one JSONL record per call with
        target/look/move/fire internals — for distribution-shift debugging in
        live eval.
        """
        del masks, generator
        with torch.inference_mode():
            _, logits, _, next_hidden, target_logits, _ = self._forward_tensors(obs, hidden=hidden)

        sample_mode = str(mode).lower()
        if sample_mode not in ("greedy", "sampled"):
            raise ValueError(f"Unsupported policy mode: {mode}")
        temps = dict(sample_temperatures or {})

        # ---- move ----
        # 3 categorical axes (fb, lr, ud), each a 3-class softmax over
        # {neg, none, pos}.  Greedy = argmax per axis; sampled = categorical
        # per axis.  Decoded engine value per axis = class - 1, i.e. {-1, 0, +1}.
        move_logits = logits[MOVE_HEAD].reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        n_rows = int(move_logits.shape[0])
        if sample_mode == "greedy":
            move_classes = torch.argmax(move_logits, dim=-1)               # (n_rows, 3)
        else:
            t_move = float(temps.get("move", 1.0))
            move_probs = F.softmax(move_logits / max(t_move, 1e-6), dim=-1)  # (n_rows, 3, 3)
            # Sample one class per axis.  Flatten axes into the batch dim
            # so _categorical_sample can run row-wise; reshape back after.
            flat_probs = move_probs.reshape(-1, MOVE_AXIS_CLASSES)         # (n_rows*3, 3)
            if row_generators is None:
                flat_classes = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            else:
                flat_classes = torch.empty(flat_probs.shape[0], dtype=torch.long, device=flat_probs.device)
                for axis_idx in range(flat_probs.shape[0]):
                    row_idx = axis_idx // MOVE_AXES
                    gen = row_generators[row_idx]
                    flat_classes[axis_idx] = torch.multinomial(
                        flat_probs[axis_idx:axis_idx + 1], num_samples=1, generator=gen,
                    ).squeeze()
            move_classes = flat_classes.reshape(n_rows, MOVE_AXES)
        move = (move_classes.float() - float(MOVE_CLASS_NONE))             # {-1, 0, +1} per axis

        # ---- look ----
        # pred_look is already unit-normalized inside the model; clamp guards
        # against any fp noise that pushes a component slightly out of range.
        look = torch.clamp(logits[LOOK_HEAD].reshape(-1, LOOK_HEAD_SIZE), -1.0, 1.0)

        # ---- fire ----
        fire_logit = logits[FIRE_HEAD].reshape(-1)
        fire_prob = torch.sigmoid(fire_logit)
        if sample_mode == "greedy":
            fire = (fire_prob > 0.5).long()
        else:
            t_fire = float(temps.get("fire", 1.0))
            # Temperature-modulate the logit: prob(class=1) = sigmoid(logit/T).
            fire_prob_t = torch.sigmoid(fire_logit / max(t_fire, 1e-6))
            fire = self._bernoulli_sample(fire_prob_t, row_generators)

        # ---- weapon ----
        # Sticky weapon controller: pick the desired weapon class when
        # the head is both confident and separated from the runner-up;
        # otherwise emit the currently-held weapon so the resulting
        # impulse is a no-op.  The action dict carries the engine
        # impulse byte 1..8 directly (class index + 1); the bridge
        # writes it to the binary wire unchanged.
        weapon_impulse = torch.ones(int(move.shape[0]), dtype=torch.long, device=move.device)
        if self.use_weapon_head and WEAPON_HEAD in logits and isinstance(obs, Mapping) and "self_weapon_id" in obs:
            weapon_logits = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            weapon_probs = F.softmax(weapon_logits, dim=-1)
            top2 = torch.topk(weapon_probs, k=2, dim=-1)
            desired_class = top2.indices[:, 0]
            confidence = top2.values[:, 0]
            margin = top2.values[:, 0] - top2.values[:, 1]
            current_ids = self._tensor(obs["self_weapon_id"], dtype=torch.long).reshape(-1)
            current_class = self.model._weapon_choices_from_ids(current_ids)
            should_switch = (
                (desired_class != current_class)
                & (confidence >= self.weapon_switch_confidence)
                & (margin >= self.weapon_switch_margin)
            )
            chosen_class = torch.where(should_switch, desired_class, current_class)
            weapon_impulse = chosen_class + 1   # class 0..7 → impulse 1..8

        actions = {
            "move":   move.detach().cpu().numpy().astype(np.float32),
            "look":   look.detach().cpu().numpy().astype(np.float32),
            "fire":   fire.detach().cpu().numpy().astype(np.int64),
            "weapon": weapon_impulse.detach().cpu().numpy().astype(np.int64),
        }

        if diag_log_path is not None:
            self._append_act_diagnostics(
                diag_log_path, obs, logits, target_logits, fire_logit, fire_prob,
                actions,
            )

        zero = torch.zeros(int(move.shape[0]), dtype=torch.float32, device=move.device)
        entropies = {
            "move":   zero.clone(),
            "look":   zero.clone(),
            "fire":   zero.clone(),
            "weapon": zero.clone(),
        }
        return PolicyActionBatch(
            actions=actions,
            log_probs=zero.clone(),
            values=zero.clone(),
            entropies=entropies,
            next_hidden=next_hidden.detach(),
        )

    @staticmethod
    def _append_act_diagnostics(
        path: str | Path,
        obs: Any,
        logits: Dict[str, torch.Tensor],
        target_logits: torch.Tensor,
        fire_logit: torch.Tensor,
        fire_prob: torch.Tensor,
        actions: Dict[str, np.ndarray],
    ) -> None:
        """Append per-row JSONL records with target/look/move/fire internals."""
        import json as _json
        from qnn.vocab import TOKEN_ACTOR  # local import to keep top-level clean

        def _np(x: torch.Tensor) -> np.ndarray:
            return x.detach().cpu().numpy()

        # Mask invalid target slots (TargetPointer pre-masks with -1e9 — pick
        # them up so we can tell "no actor present" from "low confidence".
        tl = _np(target_logits.reshape(target_logits.shape[0], -1))  # (B, N)
        # entity_types lets us see how many actor slots actually contain a bot
        et = obs.get("entity_types") if isinstance(obs, dict) else None
        actor_counts = None
        if et is not None:
            et_np = np.asarray(et)
            actor_counts = (et_np == TOKEN_ACTOR).sum(axis=-1).reshape(-1).tolist()

        # Soft attention probs (with masked slots ~0 thanks to -1e9 logit)
        tl_t = target_logits.reshape(target_logits.shape[0], -1)
        probs = torch.softmax(tl_t, dim=-1).detach().cpu().numpy()
        argmax_slot = probs.argmax(axis=-1).tolist()
        max_prob = probs.max(axis=-1).tolist()
        # Entropy in nats
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)

        base_look = _np(logits["_look_base"]).reshape(-1, 3) if "_look_base" in logits else None
        delta_look = _np(logits["_look_delta"]).reshape(-1, 3) if "_look_delta" in logits else None
        pred_look = _np(logits[LOOK_HEAD]).reshape(-1, 3)

        # Alignment scalar that feeds the fire head
        if base_look is not None:
            align = (pred_look * base_look).sum(axis=-1)
        else:
            align = np.full(pred_look.shape[0], np.nan, dtype=np.float32)

        move_logits_np = (
            _np(logits[MOVE_HEAD]).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES).tolist()
            if MOVE_HEAD in logits else None
        )
        move_prob_np = (
            _np(F.softmax(logits[MOVE_HEAD], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES).tolist()
            if MOVE_HEAD in logits else None
        )
        fire_logit_np = _np(fire_logit).reshape(-1).tolist()
        fire_prob_np = _np(fire_prob).reshape(-1).tolist()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for i in range(pred_look.shape[0]):
                rec = {
                    "row": i,
                    "actor_count": (actor_counts[i] if actor_counts is not None else None),
                    "target": {
                        "argmax_slot": int(argmax_slot[i]),
                        "max_prob": float(max_prob[i]),
                        "entropy_nats": float(ent[i]),
                        "logits": tl[i].tolist(),
                    },
                    "look": {
                        "pred": pred_look[i].tolist(),
                        "base": (base_look[i].tolist() if base_look is not None else None),
                        "base_mag": (
                            float(np.linalg.norm(base_look[i])) if base_look is not None else None
                        ),
                        "delta": (delta_look[i].tolist() if delta_look is not None else None),
                        "delta_mag": (
                            float(np.linalg.norm(delta_look[i])) if delta_look is not None else None
                        ),
                        "alignment": float(align[i]),
                    },
                    "move": {
                        "axes":   list(MOVE_AXIS_NAMES),
                        "logits": (move_logits_np[i] if move_logits_np is not None else None),
                        "prob":   (move_prob_np[i]   if move_prob_np   is not None else None),
                        "action": actions["move"][i].tolist(),
                    },
                    "fire": {
                        "logit": float(fire_logit_np[i]),
                        "prob":  float(fire_prob_np[i]),
                        "action": int(actions["fire"][i]),
                    },
                }
                f.write(_json.dumps(rec) + "\n")

    @staticmethod
    def _categorical_sample(
        probs: torch.Tensor,
        row_generators: Any | None,
    ) -> torch.Tensor:
        """Sample one class index per row from probs (B, K).

        When row_generators is None, uses default RNG. When provided, draws
        one row at a time so each episode's RNG advances independently — the
        eval pipeline relies on this for reproducibility across batched envs.
        """
        if row_generators is None:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        out = torch.empty(probs.shape[0], dtype=torch.long, device=probs.device)
        for idx, gen in enumerate(row_generators):
            row_p = probs[idx:idx + 1]
            out[idx] = torch.multinomial(row_p, num_samples=1, generator=gen).squeeze()
        return out

    @staticmethod
    def _bernoulli_sample(
        prob: torch.Tensor,
        row_generators: Any | None,
    ) -> torch.Tensor:
        if row_generators is None:
            return torch.bernoulli(prob).long()
        out = torch.empty(prob.shape[0], dtype=torch.long, device=prob.device)
        for idx, gen in enumerate(row_generators):
            out[idx] = torch.bernoulli(prob[idx:idx + 1], generator=gen).long()
        return out

    def _compute_head_losses_and_metrics(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        compute_metrics: bool = True,
        target_logits: torch.Tensor | None = None,
        target_query: torch.Tensor | None = None,
        obs: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[bool], Dict[str, torch.Tensor | int | float]]:
        weights_map = head_loss_weights or HEAD_LOSS_WEIGHTS
        losses: list[torch.Tensor] = []
        loss_is_real: list[bool] = []
        metrics: Dict[str, torch.Tensor | int | float] = {}
        accuracy_components: list[torch.Tensor] = []
        valid_flat = valid_mask.reshape(-1).bool() if valid_mask is not None else None

        # Optional per-axis loss-keep mask sourced from the engine's
        # op_input bits — bit i set ⇔ engine acted on the corresponding
        # axis press this tick. The mask is ``(no press on that axis) |
        # (op_input bit set)``, mirroring the labeler recipe at
        # qnn.labeler.train.SequenceDataset.__getitem__. Bits:
        #   0=fb, 1=lr, 2=ud, 3=fire, 4=impulse (weapon is held-derived,
        #   not masked here).
        # Off by default; trainer sets self.op_input_mask = True from
        # BCConfig when the toggle is enabled. Requires the recollected
        # corpus that carries act_op_input.npy — hard-fails if missing.
        op_mask_on = bool(self.op_input_mask)
        if op_mask_on and "op_input" not in actions:
            raise RuntimeError(
                "op_input_mask=True but actions['op_input'] is absent. "
                "Recollect the corpus on a post-bit-3-fix branch (the "
                "engine emits act_op_input.npy as part of every shard "
                "via the BC QWD + labeler paths)."
            )
        op_input_flat: torch.Tensor | None = None
        if op_mask_on:
            op_input_flat = self._tensor(actions["op_input"], dtype=torch.long).reshape(-1)

        target_loss_weight = float(weights_map.get("target", 1.0))
        target_pid_aux_weight = float(weights_map.get("target_pid_aux", 0.0))
        if (
            target_logits is not None
            and "target_dist" in actions
            and (target_loss_weight != 0.0 or target_pid_aux_weight != 0.0)
        ):
            target_flat = self._flatten_logits(target_logits)
            dist_t = self._tensor(actions["target_dist"], dtype=torch.float32)
            if dist_t.ndim == 3:
                dist_t = dist_t.reshape(-1, dist_t.shape[-1])
            # dist_t[:, 0] = NO_TARGET; dist_t[:, 1:] = slot probabilities.
            present = (1.0 - dist_t[:, 0]).clamp(min=0.0)
            slot_dist = dist_t[:, 1:]
            # No in-policy gate. Engagement filtering is the caller's job
            # via segment_mask (e.g. `{"act.target": {"$ne": 0}}`). Frames
            # with present=0 are already dropped at the dataset level; what
            # remains contributes to target loss and metrics in proportion
            # to its present value, with the clamp(min=1e-6) in the
            # renormalize keeping the divide numerically safe.
            valid = (
                valid_flat
                if valid_flat is not None
                else torch.ones_like(present, dtype=torch.bool)
            )
            aux_is_real = bool(valid.any().item())
            # Argmax of slot_dist serves as the hard label for the existing
            # accuracy / per-slot recall diagnostics.
            target_label = slot_dist.argmax(dim=-1)
            if aux_is_real:
                # Present-weighted soft CE: -sum_s p_slot * log_softmax(logits).
                # slot_dist sums to `present`; renormalize so each frame's
                # target probability mass sums to 1 before computing CE, then
                # weight the per-frame term by `present`.
                log_probs = F.log_softmax(target_flat[valid], dim=-1)
                present_v = present[valid]
                slot_target = slot_dist[valid] / present_v.clamp(min=1e-6).unsqueeze(-1)
                per_frame_ce = -(slot_target * log_probs).sum(dim=-1)
                aux_ce = (present_v * per_frame_ce).sum() / present_v.sum().clamp(min=1e-6)
            else:
                aux_ce = torch.zeros((), dtype=target_flat.dtype, device=target_flat.device)
            losses.append(aux_ce * target_loss_weight)
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_target"] = aux_ce.detach()
                metrics["target_present_mean"] = present.mean().detach()
                if aux_is_real:
                    # Present-weighted soft-distribution diagnostics, all
                    # computed at frames passing segment_mask (valid_flat).
                    # The renormalized slot_target sums to 1 per row so
                    # these are real probability-distribution quantities.
                    soft = F.softmax(target_flat[valid], dim=-1)
                    # Entropy of the (renormalized) label distribution.
                    ent_per_frame = -(slot_target.clamp(min=1e-8) * slot_target.clamp(min=1e-8).log()).sum(dim=-1)
                    target_entropy = (present_v * ent_per_frame).sum() / present_v.sum().clamp(min=1e-6)
                    metrics["target_entropy"] = target_entropy.detach()
                    # KL(label || model) = NLL - entropy(label).
                    metrics["target_kl"] = (aux_ce - target_entropy).detach()
                    # Brier: present-weighted squared error between predicted
                    # and renormalized label distributions.
                    brier_per_frame = ((soft - slot_target) ** 2).sum(dim=-1)
                    metrics["target_brier"] = (
                        (present_v * brier_per_frame).sum() / present_v.sum().clamp(min=1e-6)
                    ).detach()
                    # Top-1 mass: label mass at the model's argmax slot.
                    pred = torch.argmax(target_flat, dim=1)
                    pred_v = pred[valid]
                    target_v = target_label[valid]
                    batch_idx = torch.arange(pred_v.shape[0], device=pred_v.device)
                    top1_mass_per_frame = slot_target[batch_idx, pred_v]
                    metrics["target_top1_mass"] = (
                        (present_v * top1_mass_per_frame).sum() / present_v.sum().clamp(min=1e-6)
                    ).detach()
                    acc = (pred_v == target_v).float().mean()
                    metrics["acc_target"] = acc
                    accuracy_components.append(acc)
                    metrics["n_target_valid"] = torch.as_tensor(
                        float(target_v.numel()), dtype=target_flat.dtype, device=target_flat.device,
                    )
                    metrics["correct_target"] = (pred_v == target_v).sum().to(target_flat.dtype).detach()

                    true_nonzero = target_v != 0
                    pred_nonzero = pred_v != 0
                    tp_nz = (pred_nonzero & true_nonzero).sum().to(target_flat.dtype)
                    fp_nz = (pred_nonzero & ~true_nonzero).sum().to(target_flat.dtype)
                    fn_nz = (~pred_nonzero & true_nonzero).sum().to(target_flat.dtype)
                    metrics["tp_target_nonzero"] = tp_nz.detach()
                    metrics["fp_target_nonzero"] = fp_nz.detach()
                    metrics["fn_target_nonzero"] = fn_nz.detach()
                    metrics["n_target_nonzero"] = true_nonzero.sum().to(target_flat.dtype).detach()
                    metrics["acc_target_slot0_baseline"] = (target_v == 0).float().mean().detach()

                    recalls = []
                    for slot in range(target_flat.shape[1]):
                        pred_slot = pred_v == slot
                        true_slot = target_v == slot
                        tp = (pred_slot & true_slot).sum().to(target_flat.dtype)
                        fp = (pred_slot & ~true_slot).sum().to(target_flat.dtype)
                        fn = (~pred_slot & true_slot).sum().to(target_flat.dtype)
                        support = true_slot.sum().to(target_flat.dtype)
                        pred_count = pred_slot.sum().to(target_flat.dtype)
                        metrics[f"tp_target_slot_{slot}"] = tp.detach()
                        metrics[f"fp_target_slot_{slot}"] = fp.detach()
                        metrics[f"fn_target_slot_{slot}"] = fn.detach()
                        metrics[f"n_target_slot_{slot}"] = support.detach()
                        metrics[f"pred_target_slot_{slot}"] = pred_count.detach()
                        if bool((support > 0).item()):
                            recalls.append(tp / support.clamp(min=1.0))
                    if recalls:
                        metrics["balanced_acc_target"] = torch.stack(recalls).mean().detach()

            # Auxiliary loss: bind the predicted query to the target pid's
            # embedding identity.  Slot labels alone don't push the model to
            # encode "I'm engaging this specific pid" because slot ordering
            # shuffles within an engagement (a former slot 0 becomes slot 1
            # ~half the time within a second).  Cosine pull between the query
            # and the static pid embedding gives identity-stable supervision.
            pid_aux_weight = target_pid_aux_weight
            if (
                pid_aux_weight > 0.0
                and target_query is not None
                and obs is not None
                and "entity_ids" in obs
                and aux_is_real
            ):
                query_flat = target_query.reshape(-1, target_query.shape[-1])
                entity_ids = self._tensor(obs["entity_ids"], dtype=torch.long)
                # Flatten leading dims to match target_label.
                eids_flat = entity_ids.reshape(-1, entity_ids.shape[-2], entity_ids.shape[-1])
                slot_idx = target_label[valid]
                # Gather pid for the target slot of each valid frame.
                row_idx = torch.arange(eids_flat.shape[0], device=eids_flat.device)[valid]
                target_pid = eids_flat[row_idx, slot_idx, 2]
                # Drop frames where target_pid resolves to 0 (no-pid sentinel).
                pid_mask = target_pid > 0
                if bool(pid_mask.any().item()):
                    q = query_flat[valid][pid_mask]
                    p = self.model.trunk.tokenizer.player_embed(target_pid[pid_mask])
                    cos = F.cosine_similarity(q, p, dim=-1)
                    aux_pid = -(cos.mean())
                else:
                    aux_pid = torch.zeros((), dtype=query_flat.dtype, device=query_flat.device)
                losses.append(aux_pid * pid_aux_weight)
                loss_is_real.append(bool(pid_mask.any().item()))
                if compute_metrics:
                    metrics["loss_target_pid_aux"] = aux_pid.detach()

        if WEAPON_HEAD in logits and WEAPON_HEAD in actions:
            weapon_logits = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            weapon_target = self._weapon_target_from_actions(actions)
            # No-weapon frames carry target=-100; F.cross_entropy with
            # ignore_index=-100 skips them on-GPU. Avoid the
            # ``valid.any().item()`` host sync that used to gate the call —
            # syncing per microbatch stalled the ROCm dispatch queue and
            # cost ~10ms per step on the head-probe loop.
            if valid_flat is not None:
                weapon_target = torch.where(
                    valid_flat, weapon_target, torch.full_like(weapon_target, -100)
                )
            valid_weapon = weapon_target >= 0
            weapon_loss = F.cross_entropy(
                weapon_logits, weapon_target, ignore_index=-100, reduction="mean",
            )
            losses.append(weapon_loss * weights_map.get(WEAPON_HEAD, 1.0))
            # Engaged training always has at least one valid weapon frame
            # per microbatch — skip the per-step host sync that previously
            # checked `valid.any().item()`. If you ever train on a corpus
            # where a microbatch could be all-no-weapon, restore the sync
            # or switch to a reduction='sum' / clamped-divisor scheme to
            # avoid the 0/0 → NaN in F.cross_entropy(reduction='mean').
            loss_is_real.append(True)
            if compute_metrics:
                metrics["loss_weapon"] = weapon_loss.detach()
                with torch.no_grad():
                    # Vectorized 8-class confusion matrix: 1 scatter_add
                    # instead of an 8-iteration Python loop with ~10 tensor
                    # ops per iteration. Cuts per-batch weapon-metric kernel
                    # count from ~80 to ~5 — measured ~5-8s/epoch saved at
                    # bs=4096 on this head-probe loop.
                    weapon_probs = F.softmax(weapon_logits, dim=-1)
                    weapon_pred = torch.argmax(weapon_probs, dim=-1)
                    # Map invalid frames to a sentinel out-of-range index
                    # so they don't land in any of the WEAPON_HEAD_SIZE rows.
                    safe_target = torch.where(
                        valid_weapon, weapon_target,
                        torch.full_like(weapon_target, WEAPON_HEAD_SIZE),
                    )
                    safe_pred = torch.where(
                        valid_weapon, weapon_pred,
                        torch.full_like(weapon_pred, WEAPON_HEAD_SIZE),
                    )
                    # Confusion matrix: rows=pred, cols=target, size (K+1)^2.
                    # Last row/col is the "invalid" bucket and is discarded.
                    K = WEAPON_HEAD_SIZE
                    flat_idx = (safe_pred * (K + 1) + safe_target).long()
                    conf = torch.zeros(
                        (K + 1) * (K + 1), dtype=torch.float32, device=weapon_logits.device,
                    )
                    conf.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                    conf = conf.view(K + 1, K + 1)[:K, :K]  # (K, K), drop invalid bucket
                    # Per-class tp/fp/fn: tp = diag; row sum - tp = fp; col sum - tp = fn.
                    tp_all = conf.diagonal()
                    fp_all = conf.sum(dim=1) - tp_all
                    fn_all = conf.sum(dim=0) - tp_all
                    valid_count = conf.sum()
                    metrics["n_weapon_valid"] = valid_count.detach().to(weapon_logits.dtype)
                    metrics["acc_weapon"] = (tp_all.sum() / valid_count.clamp(min=1.0)).detach()
                    metrics["confidence_weapon"] = weapon_probs.max(dim=-1).values.mean().detach()
                    # Per-class precision / recall / F1 + base rate so the
                    # rare classes (axe / GL / NG / SNG together <10% of
                    # frames) don't disappear into the headline number.
                    class_f1s = []
                    for cls_idx, cls_name in WEAPON_HEAD_CLASS_NAMES:
                        tp = tp_all[cls_idx]
                        fp = fp_all[cls_idx]
                        fn = fn_all[cls_idx]
                        metrics[f"tp_weapon_{cls_name}"] = tp.detach()
                        metrics[f"fp_weapon_{cls_name}"] = fp.detach()
                        metrics[f"fn_weapon_{cls_name}"] = fn.detach()
                        prec = tp / (tp + fp).clamp(min=1.0)
                        rec = tp / (tp + fn).clamp(min=1.0)
                        f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                        metrics[f"precision_weapon_{cls_name}"] = prec.detach()
                        metrics[f"recall_weapon_{cls_name}"] = rec.detach()
                        metrics[f"f1_weapon_{cls_name}"] = f1.detach()
                        metrics[f"pos_rate_weapon_{cls_name}"] = (
                            (tp + fn) / valid_count.clamp(min=1.0)
                        ).detach()
                        class_f1s.append(f1)
                    metrics["f1_weapon"] = torch.stack(class_f1s).mean().detach()

        if MOVE_HEAD in logits and MOVE_HEAD in actions:
            # Move = 3 categorical axes (fb, lr, ud) × 3 classes {neg, none,
            # pos}.  Logits already include the target-anchored prior on
            # fb/lr from _compose_move_categorical (zero on ud).  Labels
            # are uint8[T, 3] axis class indices from the corpus loader.
            move_logits = logits[MOVE_HEAD]
            move_pred = move_logits.reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
            move_target_t = self._tensor(actions[MOVE_HEAD], dtype=torch.long)
            # Distance-weighted shoulder on the ud (jump) axis. Only sensible
            # when targets arrive in (T, B, 3) form so the conv sees a real
            # time axis; single-step inference falls back to plain CE.
            jump_dist_weight_flat: torch.Tensor | None = None
            if self.jump_distance_sigma > 0.0:
                ud_idx = MOVE_AXIS_NAMES.index("ud")
                if move_target_t.ndim == 3:
                    from qnn.bc.heads.loss_shaping import distance_weighted_neg_weights
                    jump_pos_2d = (move_target_t[..., ud_idx] == MOVE_CLASS_POS).to(torch.float32)
                    valid_2d = valid_mask.bool() if valid_mask is not None else None
                    w_2d = distance_weighted_neg_weights(
                        jump_pos_2d, valid_2d, self.jump_distance_sigma,
                    )
                    jump_dist_weight_flat = w_2d.reshape(-1)
                elif move_target_t.ndim == 2 and "jump_distance_to_pos" in actions:
                    # Flat batch (frame-shuffled SGD). The jump-positive mask
                    # is derived from move[..., ud_idx] == MOVE_CLASS_POS;
                    # the per-frame distance was precomputed at preload time.
                    from qnn.bc.heads.loss_shaping import flat_distance_weight
                    jump_pos_1d = (move_target_t[..., ud_idx] == MOVE_CLASS_POS).to(torch.float32)
                    jump_d = self._tensor(actions["jump_distance_to_pos"], dtype=torch.float32).reshape(-1)
                    jump_dist_weight_flat = flat_distance_weight(
                        jump_d, jump_pos_1d, self.jump_distance_sigma,
                    )
            move_target = move_target_t.reshape(-1, MOVE_AXES)
            base_move_valid = valid_flat if valid_flat is not None else torch.ones(
                (move_target.shape[0],), dtype=torch.bool, device=move_target.device,
            )
            # Per-axis loss-keep mask. When op_input_mask is off, every
            # axis sees the same base_move_valid (no behavior change vs
            # the original single-mask form). When on, axis i additionally
            # drops frames where the press was held but the engine didn't
            # act on it: keep ⇔ (target class == NONE) | (op_input bit i set).
            move_valid_per_axis: list[torch.Tensor] = []
            for axis_i in range(MOVE_AXES):
                axis_valid = base_move_valid
                if op_mask_on and op_input_flat is not None:
                    no_press = move_target[:, axis_i] == MOVE_CLASS_NONE
                    op_kept = ((op_input_flat >> axis_i) & 1) != 0
                    axis_valid = axis_valid & (no_press | op_kept)
                move_valid_per_axis.append(axis_valid)
            move_is_real = bool(base_move_valid.any().item())
            # ud (jump) axis is heavily imbalanced (~4% pos rate); upweight
            # the POS class via jump_pos_weight when set above 1.0.  fb/lr
            # are balanced enough that plain CE works.
            ud_class_weight = None
            if self.jump_pos_weight != 1.0:
                ud_class_weight = torch.tensor(
                    [1.0, 1.0, float(self.jump_pos_weight)],
                    dtype=move_pred.dtype, device=move_pred.device,
                )
            ce_per_axis = []
            for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                axis_valid = move_valid_per_axis[axis_i]
                axis_pred = move_pred[axis_valid, axis_i, :]
                axis_target = move_target[axis_valid, axis_i]
                axis_is_real = axis_pred.shape[0] > 0
                if not axis_is_real:
                    ce_axis = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                elif axis_name == "ud":
                    if jump_dist_weight_flat is not None:
                        # Per-frame CE then multiplicative distance weight,
                        # matching the fire-head .mean() reduction so both
                        # heads' loss magnitudes scale the same way.
                        ce_pf = F.cross_entropy(
                            axis_pred, axis_target,
                            weight=ud_class_weight, reduction="none",
                        )
                        ce_axis = (ce_pf * jump_dist_weight_flat[axis_valid]).mean()
                    else:
                        ce_axis = F.cross_entropy(
                            axis_pred, axis_target, weight=ud_class_weight, reduction="mean",
                        )
                else:
                    ce_axis = F.cross_entropy(axis_pred, axis_target, reduction="mean")
                ce_per_axis.append(ce_axis)
            move_loss = torch.stack(ce_per_axis).mean()  # equal-weight axes
            losses.append(move_loss * weights_map.get(MOVE_HEAD, 1.0))
            loss_is_real.append(move_is_real)
            if compute_metrics:
                metrics["loss_move"] = move_loss.detach()
                if move_is_real:
                    with torch.no_grad():
                        # Per-axis argmax computed once; per-axis indexing
                        # below selects each axis's valid frames separately
                        # so op_input-masked axes drop their stale frames.
                        move_argmax_all = torch.argmax(move_pred, dim=-1)  # (B, 3)
                        per_axis_acc: list[torch.Tensor] = []
                        per_axis_macro_f1 = []
                        for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                            axis_valid = move_valid_per_axis[axis_i]
                            metrics[f"loss_move_{axis_name}"] = ce_per_axis[axis_i].detach()
                            pred_axis = move_argmax_all[axis_valid, axis_i]
                            true_axis = move_target[axis_valid, axis_i]
                            if pred_axis.numel() > 0:
                                axis_acc = (pred_axis == true_axis).float().mean()
                            else:
                                axis_acc = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                            metrics[f"acc_move_{axis_name}"] = axis_acc.detach()
                            per_axis_acc.append(axis_acc)
                            # Per-class precision / recall / F1 across all three
                            # classes (neg/none/pos).  Macro-F1 per axis is the
                            # honest single-axis summary that doesn't hide the
                            # rare-class failure modes (jump under ud, backpedal
                            # under fb) behind the dominant "none" class.
                            class_f1s = []
                            for cls_idx, cls_name in ((MOVE_CLASS_NEG, "neg"),
                                                      (MOVE_CLASS_NONE, "none"),
                                                      (MOVE_CLASS_POS, "pos")):
                                pred_cls = pred_axis == cls_idx
                                true_cls = true_axis == cls_idx
                                tp = (pred_cls & true_cls).sum().float()
                                fp = (pred_cls & ~true_cls).sum().float()
                                fn = (~pred_cls & true_cls).sum().float()
                                prec = tp / (tp + fp).clamp(min=1.0)
                                rec = tp / (tp + fn).clamp(min=1.0)
                                f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                                metrics[f"precision_move_{axis_name}_{cls_name}"] = prec.detach()
                                metrics[f"recall_move_{axis_name}_{cls_name}"] = rec.detach()
                                metrics[f"f1_move_{axis_name}_{cls_name}"] = f1.detach()
                                if true_cls.numel() > 0:
                                    metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = true_cls.float().mean().detach()
                                else:
                                    metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = torch.zeros(
                                        (), dtype=move_pred.dtype, device=move_pred.device,
                                    )
                                class_f1s.append(f1)
                            macro = torch.stack(class_f1s).mean()
                            metrics[f"f1_move_{axis_name}"] = macro.detach()
                            per_axis_macro_f1.append(macro)
                        # Equal-axes overall acc/F1 — the mean-of-per-axis
                        # form is identical to the original ``argmax ==
                        # target`` mean when all axes share one valid mask
                        # (i.e. op_input_mask off), and remains
                        # well-defined when per-axis valids differ.
                        metrics["acc_move"] = torch.stack(per_axis_acc).mean().detach()
                        metrics["f1_move"] = torch.stack(per_axis_macro_f1).mean().detach()

        if FIRE_HEAD in logits and FIRE_HEAD in actions:
            fire_logits = logits[FIRE_HEAD]
            fire_target_t = self._tensor(actions[FIRE_HEAD], dtype=torch.float32)
            # Two distance-shoulder paths exist:
            #
            # 1. Sequence path (ndim==2, lane-packed pipeline): compute
            #    weights via Conv1d on the (T, B) target stream so each
            #    frame sees its time-axis neighbors.
            # 2. Flat path (ndim==1, GPU-resident frame-shuffled SGD):
            #    no time axis exists in the batch, so we use a
            #    per-frame "distance to nearest positive in same episode"
            #    that was precomputed at preload time and shipped via
            #    actions["fire_distance_to_pos"].
            #
            # Both produce the same loss semantics; only the
            # convolution/precompute boundary moves.
            distance_weight_flat: torch.Tensor | None = None
            if self.fire_distance_sigma > 0.0:
                if fire_target_t.ndim == 2:
                    from qnn.bc.heads.loss_shaping import distance_weighted_neg_weights
                    valid_2d = valid_mask.bool() if valid_mask is not None else None
                    w_2d = distance_weighted_neg_weights(
                        fire_target_t, valid_2d, self.fire_distance_sigma,
                    )
                    distance_weight_flat = w_2d.reshape(-1)
                elif fire_target_t.ndim == 1 and "fire_distance_to_pos" in actions:
                    from qnn.bc.heads.loss_shaping import flat_distance_weight
                    fire_d = self._tensor(actions["fire_distance_to_pos"], dtype=torch.float32)
                    distance_weight_flat = flat_distance_weight(
                        fire_d.reshape(-1), fire_target_t.reshape(-1),
                        self.fire_distance_sigma,
                    )

            fire_pred_full = fire_logits.reshape(-1)
            fire_target_full = fire_target_t.reshape(-1)
            fire_dw_full = distance_weight_flat
            if valid_flat is not None:
                fire_pred_full = fire_pred_full[valid_flat]
                fire_target_full = fire_target_full[valid_flat]
                if fire_dw_full is not None:
                    fire_dw_full = fire_dw_full[valid_flat]
            # Loss-keep mask. Off: kept == full. On: drop frames where
            # fire was held but the engine didn't act on it (op_input
            # bit 3 == 0 and fire_target == 1) — those are
            # cooldown/dead-time false positives in the demo label.
            if op_mask_on and op_input_flat is not None:
                op_input_full = op_input_flat
                if valid_flat is not None:
                    op_input_full = op_input_full[valid_flat]
                fire_keep = (fire_target_full == 0) | (((op_input_full >> 3) & 1) != 0)
                fire_pred = fire_pred_full[fire_keep]
                fire_target = fire_target_full[fire_keep]
                fire_dw = fire_dw_full[fire_keep] if fire_dw_full is not None else None
            else:
                fire_pred = fire_pred_full
                fire_target = fire_target_full
                fire_dw = fire_dw_full
            fire_is_real = fire_target.numel() > 0
            # pos_weight conventionally lives in class_weights[FIRE_HEAD] (set
            # at training startup from corpus statistics: neg_count/pos_count).
            pos_weight: torch.Tensor | None = None
            if class_weights is not None and FIRE_HEAD in class_weights:
                cw = class_weights[FIRE_HEAD]
                pos_weight = cw if isinstance(cw, torch.Tensor) else torch.as_tensor(cw, device=fire_pred.device)
            if fire_is_real:
                # Unified path: per-frame BCE, then multiplicative weighting
                # by (focal? * distance?). When both gamma and sigma are 0
                # the product is all-ones and the reduction matches the
                # original ``F.binary_cross_entropy_with_logits(..., reduction="mean")``.
                if self.fire_focal_gamma > 0.0 or fire_dw is not None:
                    bce = F.binary_cross_entropy_with_logits(
                        fire_pred, fire_target, pos_weight=pos_weight, reduction="none",
                    )
                    weight = torch.ones_like(bce)
                    if self.fire_focal_gamma > 0.0:
                        # Focal BCE: down-weight easy examples by (1 - p_t)^gamma.
                        # Optional per-class alpha (Lin et al.): alpha on
                        # positives, (1 - alpha) on negatives.
                        p = torch.sigmoid(fire_pred)
                        pt = torch.where(fire_target > 0.5, p, 1.0 - p)
                        alpha_t = torch.where(
                            fire_target > 0.5,
                            torch.full_like(p, self.fire_focal_alpha),
                            torch.full_like(p, 1.0 - self.fire_focal_alpha),
                        )
                        weight = weight * alpha_t * (1.0 - pt).clamp(min=1e-6) ** self.fire_focal_gamma
                    if fire_dw is not None:
                        weight = weight * fire_dw
                    fire_loss = (weight * bce).mean()
                else:
                    fire_loss = F.binary_cross_entropy_with_logits(
                        fire_pred, fire_target, pos_weight=pos_weight, reduction="mean",
                    )
            else:
                fire_loss = torch.zeros((), dtype=fire_logits.dtype, device=fire_logits.device)
            losses.append(fire_loss * weights_map.get(FIRE_HEAD, 1.0))
            loss_is_real.append(fire_is_real)
            if compute_metrics:
                metrics["loss_fire"] = fire_loss.detach()
                # Always emit headline metrics on the full subset
                # (segment_mask only; no op_input filter). Comparable
                # apples-to-apples against baseline runs that ran with
                # op_input_mask=False.
                if fire_target_full.numel() > 0:
                    with torch.no_grad():
                        pred_pos = (torch.sigmoid(fire_pred_full) > 0.5)
                        target_pos = fire_target_full > 0.5
                        tp = (pred_pos & target_pos).sum()
                        fp = (pred_pos & ~target_pos).sum()
                        fn = (~pred_pos & target_pos).sum()
                        tn = (~pred_pos & ~target_pos).sum()
                        metrics["tp_fire"] = tp.detach()
                        metrics["fp_fire"] = fp.detach()
                        metrics["fn_fire"] = fn.detach()
                        metrics["tn_fire"] = tn.detach()
                        n_total = tp + fp + fn + tn
                        metrics["acc_fire"] = ((tp + tn).float() / n_total.clamp(min=1)).detach()
                        prec_denom = (tp + fp).clamp(min=1)
                        rec_denom = (tp + fn).clamp(min=1)
                        prec = tp.float() / prec_denom
                        rec = tp.float() / rec_denom
                        f1_denom = (prec + rec).clamp(min=1e-6)
                        metrics["precision_fire"] = prec.detach()
                        metrics["recall_fire"] = rec.detach()
                        metrics["f1_fire"] = (2.0 * prec * rec / f1_denom).detach()
                # When op_input_mask is on, also emit kept-only metrics
                # (``*_fire_masked``). These reflect the subset the loss
                # actually trains on. Both are reported so the ablation
                # shows the label-set change AND the model behavior on
                # the cleaner subset.
                if op_mask_on and fire_is_real:
                    with torch.no_grad():
                        pred_pos_m = (torch.sigmoid(fire_pred) > 0.5)
                        target_pos_m = fire_target > 0.5
                        tp_m = (pred_pos_m & target_pos_m).sum()
                        fp_m = (pred_pos_m & ~target_pos_m).sum()
                        fn_m = (~pred_pos_m & target_pos_m).sum()
                        tn_m = (~pred_pos_m & ~target_pos_m).sum()
                        metrics["tp_fire_masked"] = tp_m.detach()
                        metrics["fp_fire_masked"] = fp_m.detach()
                        metrics["fn_fire_masked"] = fn_m.detach()
                        metrics["tn_fire_masked"] = tn_m.detach()
                        n_total_m = tp_m + fp_m + fn_m + tn_m
                        metrics["acc_fire_masked"] = ((tp_m + tn_m).float() / n_total_m.clamp(min=1)).detach()
                        prec_denom_m = (tp_m + fp_m).clamp(min=1)
                        rec_denom_m = (tp_m + fn_m).clamp(min=1)
                        prec_m = tp_m.float() / prec_denom_m
                        rec_m = tp_m.float() / rec_denom_m
                        f1_denom_m = (prec_m + rec_m).clamp(min=1e-6)
                        metrics["precision_fire_masked"] = prec_m.detach()
                        metrics["recall_fire_masked"] = rec_m.detach()
                        metrics["f1_fire_masked"] = (2.0 * prec_m * rec_m / f1_denom_m).detach()

        if LOOK_HEAD in logits and LOOK_HEAD in actions:
            # Magnitude-sensitive supervision: regress the raw delta_look
            # output against the geometric residual (demo_unit - base_look).
            # The residual has bounded magnitude (≤ 2 for unit vectors);
            # forces the head to "pay" the right magnitude for whatever
            # direction it expresses, instead of growing delta arbitrarily
            # large to override the prior.  cos_sim is reported as a metric
            # but no longer drives the loss.
            look_pred = logits[LOOK_HEAD].reshape(-1, LOOK_HEAD_SIZE)
            base_look = logits["_look_base"].reshape(-1, LOOK_HEAD_SIZE)
            delta_look = logits["_look_delta"].reshape(-1, LOOK_HEAD_SIZE)

            look_target_t = self._tensor(actions[LOOK_HEAD], dtype=torch.float32)
            look_target = look_target_t.reshape(-1, look_target_t.shape[-1])
            target_norm = torch.linalg.vector_norm(look_target, dim=-1, keepdim=True)
            valid = target_norm.squeeze(-1) > 1e-6
            if valid_flat is not None:
                valid = valid & valid_flat
            aux_is_real = bool(valid.any().item())
            if aux_is_real:
                unit_target = look_target[valid] / target_norm[valid].clamp(min=1e-6)
                # Target residual: what delta should be to make
                # normalize(base + delta) = unit_target.
                target_residual = unit_target - base_look[valid]
                look_loss = F.smooth_l1_loss(
                    delta_look[valid], target_residual, beta=0.05, reduction="mean",
                )
            else:
                look_loss = torch.zeros((), dtype=look_pred.dtype, device=look_pred.device)
            losses.append(look_loss * weights_map.get(LOOK_HEAD, 1.0))
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_look"] = look_loss.detach()
                if aux_is_real:
                    with torch.no_grad():
                        cos = (look_pred[valid] * unit_target).sum(dim=-1)
                        metrics["cos_sim_look"] = cos.mean().detach()
                        # Track delta magnitude so we can confirm the head
                        # is no longer growing it unbounded.
                        metrics["mag_delta_look"] = (
                            torch.linalg.vector_norm(delta_look[valid], dim=-1).mean().detach()
                        )
                    accuracy_components.append(cos.mean().detach())

        if compute_metrics:
            metrics["accuracy"] = (
                torch.stack(accuracy_components).mean()
                if accuracy_components
                else torch.zeros((), device=self.device)
            )
        else:
            metrics["accuracy"] = torch.zeros((), device=self.device)

        return losses, loss_is_real, metrics

    def _weapon_target_from_actions(
        self,
        actions: Mapping[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """Return dense desired-weapon targets from collected BC labels.

        The collector stores `weapon` as the raw engine weapon byte:
          0 = no weapon held (pre-spawn / dead / transitional),
          1..8 = Quake weapon id in impulse order (axe..thunderbolt).
        The 8-class weapon head trains on weapons only; no-weapon frames
        map to -100 so F.cross_entropy(..., ignore_index=-100) skips
        them while their move/fire/look labels still train.
        """
        weapon = self._tensor(actions[WEAPON_HEAD], dtype=torch.long).reshape(-1)
        bad = (weapon < 0) | (weapon > WEAPON_HEAD_SIZE)
        if bool(bad.any().item()):
            sample = weapon[bad][:8].detach().cpu().tolist()
            raise ValueError(
                f"weapon bytes must be in 0..{WEAPON_HEAD_SIZE}, got {sample}"
            )
        # 1..8 → class 0..7; 0 (no weapon) → -100 ignore.
        target = weapon - 1
        target = target.masked_fill(weapon == 0, -100)
        return target

    def supervised_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        lr: float,
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        accumulate_only: bool = False,
        head_loss_weights: Mapping[str, float] | None = None,
        loss_scale: float = 1.0,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        optimizer = self._optimizer("bc", self.model.parameters(), lr)
        if not accumulate_only:
            optimizer.zero_grad()

        # Teacher-force the hard-target gather with the BC GT slot so motor
        # heads always see the correctly-paired enemy vector during training.
        # No-op when hard_target_feat is off (TargetPointer ignores target_gt
        # in soft-pool mode).
        target_gt_arr = actions.get("target") if isinstance(actions, Mapping) else None
        # GT-distribution STE: derive a (T*B, N) renormalized slot distribution
        # from actions["target_dist"] (T*B, 17) when present. TargetPointer
        # uses it as the STE forward signal in training mode only; eval falls
        # back to the soft path even when target_dist_slot is supplied (gated
        # on self.training). Only supplied here in the training step.
        target_dist_slot_arr = None
        prev_target_dist_arr = None
        if isinstance(actions, Mapping) and "target_dist" in actions:
            td = self._tensor(actions["target_dist"], dtype=torch.float32)
            present = (1.0 - td[..., 0]).clamp(min=1e-6)
            target_dist_slot_arr = td[..., 1:] / present.unsqueeze(-1)
            # prev_target_dist: shift slot_target by one along the time axis
            # (only meaningful for sequence inputs). Zero at episode starts
            # via reset_mask if provided; the t=0 row is also zeroed.
            if td.ndim == 3:
                prev_st = torch.zeros_like(target_dist_slot_arr)
                prev_st[1:] = target_dist_slot_arr[:-1]
                if isinstance(masks, Mapping) and "reset_mask" in masks:
                    rm = self._tensor(masks["reset_mask"], dtype=torch.bool)
                    if rm.ndim == 2:
                        prev_st = prev_st.masked_fill(rm.unsqueeze(-1), 0.0)
                prev_target_dist_arr = prev_st
        with self._autocast():
            _, logits, _, next_hidden, target_logits, target_query = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
                target_gt=target_gt_arr,
                target_dist_slot=target_dist_slot_arr,
                prev_target_dist=prev_target_dist_arr,
            )
            valid_mask = (
                self._tensor(masks["valid_mask"], dtype=torch.bool)
                if isinstance(masks, Mapping) and "valid_mask" in masks
                else None
            )
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits,
                actions,
                class_weights=class_weights,
                head_loss_weights=head_loss_weights,
                compute_metrics=compute_metrics,
                target_logits=target_logits,
                target_query=target_query,
                obs=obs,
                valid_mask=valid_mask,
            )
            real = [l for l, r in zip(losses, loss_is_real) if r]
            loss = torch.stack(real).mean() if real else torch.zeros((), device=self.device)
        (loss * float(loss_scale)).backward()
        if not accumulate_only:
            optimizer.step()

        metrics["loss"] = loss.detach()
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def evaluate_supervised(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        # Mirror the privileged inputs supervised_step derives from
        # ``actions``: target_gt and a renormalized 16-slot target_dist.
        # The canonical model is no-op for these in eval (TargetPointer
        # gates STE on self.training) — but a model_factory-injected
        # ablation may need them (e.g., a probe whose entire encoder
        # pools by GT slot mass). Passing them keeps eval symmetric with
        # training across both code paths.
        target_gt_arr = actions.get("target") if isinstance(actions, Mapping) else None
        target_dist_slot_arr = None
        if isinstance(actions, Mapping) and "target_dist" in actions:
            td = self._tensor(actions["target_dist"], dtype=torch.float32)
            present = (1.0 - td[..., 0]).clamp(min=1e-6)
            target_dist_slot_arr = td[..., 1:] / present.unsqueeze(-1)
        with torch.inference_mode(), self._autocast():
            _, logits, _, next_hidden, target_logits, target_query = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
                target_gt=target_gt_arr,
                target_dist_slot=target_dist_slot_arr,
            )
            valid_mask = (
                self._tensor(masks["valid_mask"], dtype=torch.bool)
                if isinstance(masks, Mapping) and "valid_mask" in masks
                else None
            )
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits,
                actions,
                head_loss_weights=head_loss_weights,
                compute_metrics=compute_metrics,
                target_logits=target_logits,
                target_query=target_query,
                obs=obs,
                valid_mask=valid_mask,
            )
        real = [l for l, r in zip(losses, loss_is_real) if r]
        metrics["loss"] = torch.stack(real).mean() if real else torch.zeros((), device=self.device)
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def ppo_step(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        del args, kwargs
        raise RuntimeError("Combat-objective phase 1 does not support PPO")

    def value_step(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        del args, kwargs
        raise RuntimeError("Combat-objective phase 1 has no value head")

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "obs_dim": self.obs_dim,
            "model": self.config.to_dict(),
            "jump_pos_weight": self.jump_pos_weight,
            "fire_focal_gamma": self.fire_focal_gamma,
            "fire_focal_alpha": self.fire_focal_alpha,
            "fire_distance_sigma": self.fire_distance_sigma,
            "jump_distance_sigma": self.jump_distance_sigma,
            "backend": "pytorch",
            "requested_device": self.device_spec.requested,
            "resolved_device": self.device_spec.resolved,
            "accelerator_backend": self.device_spec.backend,
        }
        raw_sd = self.model.state_dict()
        clean_sd = {
            key.replace("_orig_mod.", ""): value.detach().cpu()
            for key, value in raw_sd.items()
        }
        payload = {
            "meta": meta,
            "state_dict": clean_sd,
        }
        torch.save(payload, target)
        target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str,
        model_factory: Callable[[int, ModelConfig], nn.Module] | None = None,
    ) -> "QNNPolicy":
        """Load a saved checkpoint.

        ``model_factory`` mirrors the constructor hook: when None, the
        canonical ``_CombatObjectiveNet`` is built from the saved
        ModelConfig and strict-loaded; when set, the factory builds the
        alternate module (e.g. a head-probe model) and the state_dict
        is loaded into it. The caller is responsible for passing the
        same factory used to train the checkpoint — checkpoints don't
        embed the factory identity.
        """
        source = Path(path)
        payload = trusted_torch_load(source, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
            raise ValueError(f"Unrecognised checkpoint format: {source}")
        meta = dict(payload["meta"])
        if "model" not in meta:
            from qnn.utils.checkpoint_converter import migrate_legacy_flat_meta
            migrated = migrate_legacy_flat_meta(meta)
            if migrated is None:
                raise ValueError(
                    f"Checkpoint {source} is missing the 'model' arch block "
                    "and migrate_legacy_flat_meta did not recognize the schema."
                )
            meta = migrated
        model_cfg = ModelConfig.from_dict(meta["model"])
        policy = cls(
            obs_dim=int(meta["obs_dim"]),
            model=model_cfg,
            jump_pos_weight=float(meta["jump_pos_weight"]),
            fire_focal_gamma=float(meta["fire_focal_gamma"]),
            fire_focal_alpha=float(meta["fire_focal_alpha"]),
            fire_distance_sigma=float(meta["fire_distance_sigma"]),
            jump_distance_sigma=float(meta["jump_distance_sigma"]),
            seed=0,
            device=device,
            model_factory=model_factory,
        )
        if model_factory is None:
            from qnn.utils.checkpoint_converter import (
                migrate_drop_action_history,
                migrate_drop_fire_align_scalar,
                migrate_drop_weapon_embed_self,
                migrate_entity_embed,
                migrate_self_attack_finished_scalar,
                migrate_self_scalars,
                migrate_v17_move_heads,
            )

            migrate_entity_embed(payload["state_dict"])
            migrate_self_scalars(payload["state_dict"])
            migrate_self_attack_finished_scalar(payload["state_dict"])
            migrate_v17_move_heads(payload["state_dict"])
            migrate_drop_action_history(payload["state_dict"])
            migrate_drop_fire_align_scalar(payload["state_dict"])
            migrate_drop_weapon_embed_self(payload["state_dict"])
        # When model_factory is set the saved state_dict is for the
        # injected module (e.g. FireTokenHead), not _CombatObjectiveNet —
        # canonical migrations don't apply and the strict-load below will
        # use empty allow-prefixes.
        try:
            # strict=False so v17 checkpoints still load:
            #  - migrate_v17_move_heads packs split fb/lr into the unified
            #    move_head and bias-locks the ud axis (no random init)
            #  - migrate_drop_action_history strips the pre-rip-out
            #    action_proj / action_pos_embed weights and truncates
            #    kind_embed from 4 -> 3 rows
            #  - migrate_drop_fire_align_scalar trims the trailing
            #    alignment-scalar column from v17/v20-era fire heads
            #    (settled-null in ablation; the column is dead weight)
            #  - weapon_head / weapon_embed start fresh on v17/v20-pre-v21
            #  - trunk.gru_input_proj weight (pre-v20 mean-actors pool) is
            #    silently dropped
            missing, unexpected = policy.model.load_state_dict(payload["state_dict"], strict=False)
            if model_factory is None:
                allowed_missing_prefixes: tuple[str, ...] = (
                    "weapon_head.", "weapon_embed.",
                )
                allowed_unexpected_prefixes: tuple[str, ...] = (
                    "trunk.gru_input_proj.",  # pre-v20: mean-actors pool projection
                )
            else:
                # No legacy migrations for factory-built modules — they
                # save and load their own state_dict shape.
                allowed_missing_prefixes = ()
                allowed_unexpected_prefixes = ()
            missing_keep = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
            unexpected_keep = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
            if missing_keep or unexpected_keep:
                raise RuntimeError(
                    f"state_dict mismatch: missing={missing_keep}, unexpected={unexpected_keep}"
                )
        except RuntimeError as exc:
            raise ValueError(
                f"Incompatible checkpoint architecture for {source}. "
                "This code expects the combat-objective BC policy layout."
            ) from exc
        policy.model.to(policy.device)
        return policy

    @classmethod
    def load_for_finetune(
        cls,
        path: str | Path,
        *,
        use_gru: bool,
        gru_hidden: int,
        device: str,
    ) -> "QNNPolicy":
        del use_gru, gru_hidden
        return cls.load(path, device=device)
