"""Combat-objective BC model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

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

def _resolve_bottleneck_dims(raw: "int | dict") -> "dict[str, int]":
    """Normalise head_bottleneck_dim (int or per-head dict) to a full dict."""
    if isinstance(raw, dict):
        return {k: int(v) for k, v in raw.items()}
    val = int(raw)
    return {"move": val, "look": val, "fire": val, "weapon": val}


HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "target": 1.0,
    "move": 1.0,
    "look": 1.0,
    "fire": 1.0,
}

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
WEAPON_HEAD_SIZE = 8
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
    def __init__(
        self,
        obs_dim: int,
        trunk_hidden: int,
        d_model: int | None = None,
        n_heads: int = 1,
        n_layers: int = 2,
        ffn_dim: int = 256,
        attn_dropout: float = 0.0,
        use_gru: bool = False,
        gru_hidden: int = 0,
        use_weapon_head: bool = False,
        look_bypass_gru: bool = False,
        weapon_use_gru: bool = True,
        weapon_context_from_obs: bool = True,
        head_bottleneck_dim: "int | dict" = 32,
        head_use_relu: bool = True,
        head_activation: "str | None" = None,
        gru_target_query: bool = False,
        hard_target_feat: bool = False,
        weapon_in_target_query: bool = False,
        linear_slot_prior: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.d_model = int(d_model if d_model is not None else trunk_hidden)
        self.trunk = TransformerTrunk(
            obs_dim=obs_dim,
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=attn_dropout,
        )
        self.use_gru = bool(use_gru and gru_hidden > 0)
        self.gru_hidden = int(gru_hidden) if self.use_gru else 0
        self.use_weapon_head = bool(use_weapon_head)
        # look_bypass_gru is a v17-fidelity load-time flag.  v20+ always sets
        # this False — when True (only via QNNPolicy.load on a v17 checkpoint)
        # the look head is fed cat(self_readout, target_feat) instead of
        # cat(gru_flat, target_feat), matching the features it was trained on.
        self.look_bypass_gru = bool(look_bypass_gru and self.use_gru)
        self.weapon_use_gru = bool(weapon_use_gru and self.use_gru)
        self.weapon_context_from_obs = bool(weapon_context_from_obs)
        # gru_target_query: route the GRU output (instead of self_readout) into
        # the target attention query. Only meaningful when use_gru is True.
        self.gru_target_query = bool(gru_target_query and self.use_gru)
        # hard_target_feat couples two changes:
        #   - target_feat switches from soft attention pool to a single chosen
        #     entity vector (hard pick) — heads see "one real enemy", not a blend.
        #   - During training the chosen slot is the BC GT (teacher forcing) so
        #     the motor heads always see the correctly-paired enemy regardless
        #     of pointer confidence. At eval the chosen slot is argmax(logits).
        # Decoupling means target-head loss tuning (focal / class weights)
        # stops reshaping motor-head training distribution.
        self.hard_target_feat = bool(hard_target_feat)
        # weapon_in_target_query: add an additive currently-held weapon embedding
        # to the target query (post-projection) so target attention can condition
        # on weapon (e.g. RL pulls toward distant enemies, shotgun toward close).
        # Lives in the query/key dot-product space, stays out of the weapon head's
        # input — can't poison weapon-head training. Independent of hard_target_feat.
        self.weapon_in_target_query = bool(weapon_in_target_query and self.use_gru)
        # linear_slot_prior: add an additive logit prior linear in slot index
        # (-alpha * k) to target_logits before masking. Encodes the slot ordering
        # policy (slot 0 most likely) directly so the residual attention scores
        # only need to override when a non-slot-0 target is better.
        self.linear_slot_prior = bool(linear_slot_prior and self.use_gru)
        self.head_bottleneck_dims = _resolve_bottleneck_dims(head_bottleneck_dim)
        self.head_use_relu = bool(head_use_relu)
        self.head_activation = (
            str(head_activation).lower()
            if head_activation is not None
            else ("relu" if self.head_use_relu else "none")
        )
        if self.head_activation not in ("none", "relu", "gelu"):
            raise ValueError(f"Unknown head_activation: {self.head_activation}")
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
        # Weapon selector: gru_flat (optional) + self_readout + target_feat.
        weapon_in = (self.gru_hidden if self.weapon_use_gru else 0) + (2 * self.d_model)
        if self.use_gru:
            self.gru = nn.GRU(self.d_model, self.gru_hidden, batch_first=False)
            self.target_pointer = TargetPointer(
                d_model=self.d_model,
                query_in_dim=self.gru_hidden if self.gru_target_query else self.d_model,
                inject_weapon=self.weapon_in_target_query,
                weapon_vocab=WEAPON_HEAD_SIZE,
                hard_target=self.hard_target_feat,
                linear_slot_prior=self.linear_slot_prior,
            )
        if self.use_weapon_head:
            self.weapon_head = self._make_head(weapon_in, WEAPON_HEAD_SIZE, self.head_bottleneck_dims.get("weapon", 0))
            self.weapon_embed = nn.Embedding(WEAPON_HEAD_SIZE, self.d_model)
        # Move head: 3 categorical axes × 3 classes = 9 logits, reshaped at
        # forward time.
        self.move_head = self._make_head(move_in, MOVE_HEAD_SIZE, self.head_bottleneck_dims.get("move", 0))
        # Look head outputs a residual added to a target-anchored prior:
        #   base_look  = normalize(soft_target_rel)   -- "look at your target"
        #   delta_look = look_head(features)          -- learned deviation
        #   pred_look  = normalize(base_look + delta_look)
        self.look_head = self._make_head(look_in, LOOK_HEAD_SIZE, self.head_bottleneck_dims.get("look", 0))
        # Fire head consumes the fused features.  The alignment scalar
        # (cosine between pred_look and base_look) was settled-null in
        # earlier ablation and removed.
        self.fire_head = self._make_head(fire_in, FIRE_HEAD_SIZE, self.head_bottleneck_dims.get("fire", 0))
        self._init_weights()

    def _make_head(self, in_dim: int, out_dim: int, bottleneck_dim: int) -> nn.Module:
        """Build an output head.

        Activation controlled by self.head_activation ("none" | "relu" | "gelu").
        Shape:
          (B>0, "none") → Linear(in, B) → Linear(B, out)            — low-rank, no activation
          (B>0, act≠none) → Linear(in, B) → act → Linear(B, out)    — 2-layer MLP with activation
          (0,   "none") → Linear(in, out)                            — v21 baseline (single linear)
          (0,   act≠none) → Linear(in, in) → act → Linear(in, out)  — full-width MLP
        """
        hidden = bottleneck_dim if bottleneck_dim > 0 else in_dim
        has_activation = self.head_activation != "none"
        if bottleneck_dim > 0 or has_activation:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
            if has_activation:
                if self.head_activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU(inplace=True))
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
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = obs["self_scalars"]
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
                target_logits, target_feat, target_query = self.target_pointer(
                    tp_query_input,
                    entity_outs,
                    entity_mask,
                    self_weapon_slot=tp_weapon_slot,
                    target_gt=tp_target_gt,
                )
                features_base_flat = torch.cat([gru_flat, target_feat], dim=-1)
                # Weapon selector composition is gated by weapon_use_gru: drop
                # gru_flat to test whether weapon choice needs recurrence.
                if self.weapon_use_gru:
                    weapon_selector_flat = torch.cat([gru_flat, self_readout, target_feat], dim=-1)
                else:
                    weapon_selector_flat = torch.cat([self_readout, target_feat], dim=-1)
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
            target_logits, target_feat, target_query = self.target_pointer(
                tp_query_input,
                entity_outs,
                entity_mask,
                self_weapon_slot=tp_weapon_slot,
                target_gt=tp_target_gt,
            )
            features_base = torch.cat([recurrent, target_feat], dim=-1)
            if self.weapon_use_gru:
                weapon_selector = torch.cat([recurrent, self_readout, target_feat], dim=-1)
            else:
                weapon_selector = torch.cat([self_readout, target_feat], dim=-1)
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
        obs_dim: int,
        trunk_hidden: int = 64,
        gru_hidden: int = 64,
        use_gru: bool = True,
        seed: int = 0,
        device: str = "auto",
        d_model: int | None = None,
        n_heads: int = 1,
        n_layers: int = 2,
        ffn_dim: int = 256,
        attn_dropout: float = 0.0,
        use_weapon_head: bool = False,
        weapon_switch_confidence: float = 0.65,
        weapon_switch_margin: float = 0.15,
        jump_pos_weight: float = 1.0,
        target_focal_gamma: float = 0.0,
        look_bypass_gru: bool = False,
        weapon_use_gru: bool = True,
        weapon_context_from_obs: bool = True,
        head_bottleneck_dim: "int | dict" = 32,
        head_use_relu: bool = True,
        head_activation: "str | None" = None,
        gru_target_query: bool = False,
        hard_target_feat: bool = False,
        weapon_in_target_query: bool = False,
        linear_slot_prior: bool = False,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.d_model = int(d_model if d_model is not None else trunk_hidden)
        self.trunk_hidden = self.d_model
        self.use_gru = bool(use_gru and gru_hidden > 0)
        self.gru_hidden = int(gru_hidden) if self.use_gru else 0
        self.use_weapon_head = bool(use_weapon_head)
        self.look_bypass_gru = bool(look_bypass_gru and self.use_gru)
        self.weapon_switch_confidence = float(weapon_switch_confidence)
        self.weapon_switch_margin = float(weapon_switch_margin)
        self.weapon_use_gru = bool(weapon_use_gru)
        self.weapon_context_from_obs = bool(weapon_context_from_obs)
        self.gru_target_query = bool(gru_target_query and self.use_gru)
        self.hard_target_feat = bool(hard_target_feat)
        self.weapon_in_target_query = bool(weapon_in_target_query and self.use_gru)
        self.linear_slot_prior = bool(linear_slot_prior and self.use_gru)
        self.head_bottleneck_dims = _resolve_bottleneck_dims(head_bottleneck_dim)
        self.head_use_relu = bool(head_use_relu)
        self.head_activation = (
            str(head_activation).lower()
            if head_activation is not None
            else ("relu" if self.head_use_relu else "none")
        )
        # jump_pos_weight > 1.0 upweights the POS class on the move ud-axis CE
        # — direct imbalance fix for the rare jump-positive case (~4% pos rate).
        # Inverse-frequency reference: ~24× for 4% positive rate.
        self.jump_pos_weight = float(jump_pos_weight)
        # target_focal_gamma > 0 modulates target-slot CE by (1 - p_t)^gamma so
        # the loss concentrates on exception frames (true target != slot 0)
        # instead of saturating on the 96% slot-0 majority. 2.0 is the
        # standard focal-loss exponent.
        self.target_focal_gamma = float(target_focal_gamma)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ffn_dim = int(ffn_dim)
        self.attn_dropout = float(attn_dropout)
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
        self.model = _CombatObjectiveNet(
            obs_dim=self.obs_dim,
            trunk_hidden=self.trunk_hidden,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_dim=self.ffn_dim,
            attn_dropout=self.attn_dropout,
            use_gru=self.use_gru,
            gru_hidden=self.gru_hidden,
            use_weapon_head=self.use_weapon_head,
            look_bypass_gru=self.look_bypass_gru,
            weapon_use_gru=self.weapon_use_gru,
            weapon_context_from_obs=self.weapon_context_from_obs,
            head_bottleneck_dim=self.head_bottleneck_dims,
            head_use_relu=self.head_use_relu,
            head_activation=self.head_activation,
            gru_target_query=self.gru_target_query,
            hard_target_feat=self.hard_target_feat,
            weapon_in_target_query=self.weapon_in_target_query,
            linear_slot_prior=self.linear_slot_prior,
        ).to(self.device)
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
        sample = obs_dict["self_scalars"]
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

    def _forward_tensors(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        target_gt: np.ndarray | torch.Tensor | None = None,
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

        sample = obs_tensors["self_scalars"]
        if sample.ndim == 2:
            batch_size = int(sample.shape[0])
            padded_obs, pad_rows = self._maybe_pad_obs_batch(obs_tensors)
            padded_hidden = hidden_tensor
            if self.use_gru and padded_hidden is not None and pad_rows > 0:
                pad = torch.zeros(
                    (pad_rows, padded_hidden.shape[-1]),
                    dtype=padded_hidden.dtype,
                    device=padded_hidden.device,
                )
                padded_hidden = torch.cat([padded_hidden, pad], dim=0)
            features, logits, values, next_hidden, target_logits, target_query = self.model(
                padded_obs,
                padded_hidden,
                target_gt=target_gt_tensor,
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

        if target_logits is not None and "target" in actions:
            target_flat = self._flatten_logits(target_logits)
            target_label = self._flatten_targets(self._tensor(actions["target"], dtype=torch.long))
            valid = target_label != -100
            if valid_flat is not None:
                valid = valid & valid_flat
            aux_is_real = bool(valid.any().item())
            if aux_is_real:
                # Focal modulation: (1 - p_t)^gamma weights hard frames (true
                # slot != 0) higher than the 96% slot-0 majority where p_t
                # saturates. gamma=0 reduces to vanilla CE.
                if self.target_focal_gamma > 0.0:
                    log_probs = F.log_softmax(target_flat[valid], dim=-1)
                    gathered = log_probs.gather(1, target_label[valid].unsqueeze(1)).squeeze(1)
                    p_t = gathered.exp()
                    focal_w = (1.0 - p_t).pow(self.target_focal_gamma)
                    aux_ce = -(focal_w * gathered).mean()
                else:
                    aux_ce = F.cross_entropy(target_flat[valid], target_label[valid], reduction="mean")
            else:
                aux_ce = torch.zeros((), dtype=target_flat.dtype, device=target_flat.device)
            losses.append(aux_ce * weights_map.get("target", 1.0))
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_target"] = aux_ce.detach()
                if aux_is_real:
                    pred = torch.argmax(target_flat, dim=1)
                    pred_v = pred[valid]
                    target_v = target_label[valid]
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
            pid_aux_weight = float(weights_map.get("target_pid_aux", 0.0))
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
            # No-weapon frames carry target=-100 and are skipped by CE +
            # per-class metrics; their move/fire/look labels still train.
            valid_weapon = weapon_target >= 0
            if valid_flat is not None:
                valid_weapon = valid_weapon & valid_flat
            if bool(valid_weapon.any().item()):
                weapon_loss = F.cross_entropy(weapon_logits[valid_weapon], weapon_target[valid_weapon], reduction="mean")
            else:
                weapon_loss = torch.zeros((), dtype=weapon_logits.dtype, device=weapon_logits.device)
            losses.append(weapon_loss * weights_map.get(WEAPON_HEAD, 1.0))
            loss_is_real.append(bool(valid_weapon.any().item()))
            if compute_metrics:
                metrics["loss_weapon"] = weapon_loss.detach()
                with torch.no_grad():
                    weapon_probs = F.softmax(weapon_logits, dim=-1)
                    weapon_pred = torch.argmax(weapon_probs, dim=-1)
                    pred_v = weapon_pred[valid_weapon]
                    target_v = weapon_target[valid_weapon]
                    if target_v.numel() > 0:
                        metrics["acc_weapon"] = (pred_v == target_v).float().mean().detach()
                    metrics["confidence_weapon"] = weapon_probs.max(dim=-1).values.mean().detach()
                    metrics["n_weapon_valid"] = torch.as_tensor(
                        float(target_v.numel()), dtype=weapon_logits.dtype, device=weapon_logits.device,
                    )
                    # Per-class precision / recall / F1 + base rate so the
                    # rare classes (axe / GL / NG / SNG together <10% of
                    # frames) don't disappear into the headline number.
                    # Macro-F1 averages the eight per-class F1s with equal
                    # weight regardless of frequency.
                    class_f1s = []
                    for cls_idx, cls_name in WEAPON_HEAD_CLASS_NAMES:
                        pred_cls = pred_v == cls_idx
                        true_cls = target_v == cls_idx
                        tp = (pred_cls & true_cls).sum().float()
                        fp = (pred_cls & ~true_cls).sum().float()
                        fn = (~pred_cls & true_cls).sum().float()
                        metrics[f"tp_weapon_{cls_name}"] = tp.detach()
                        metrics[f"fp_weapon_{cls_name}"] = fp.detach()
                        metrics[f"fn_weapon_{cls_name}"] = fn.detach()
                        prec = tp / (tp + fp).clamp(min=1.0)
                        rec = tp / (tp + fn).clamp(min=1.0)
                        f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                        metrics[f"precision_weapon_{cls_name}"] = prec.detach()
                        metrics[f"recall_weapon_{cls_name}"] = rec.detach()
                        metrics[f"f1_weapon_{cls_name}"] = f1.detach()
                        metrics[f"pos_rate_weapon_{cls_name}"] = true_cls.float().mean().detach()
                        class_f1s.append(f1)
                    metrics["f1_weapon"] = torch.stack(class_f1s).mean().detach()

        if MOVE_HEAD in logits and MOVE_HEAD in actions:
            # Move = 3 categorical axes (fb, lr, ud) × 3 classes {neg, none,
            # pos}.  Logits already include the target-anchored prior on
            # fb/lr from _compose_move_categorical (zero on ud).  Labels
            # are uint8[T, 3] axis class indices from the corpus loader.
            move_logits = logits[MOVE_HEAD]
            move_pred = move_logits.reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
            move_target = self._tensor(
                actions[MOVE_HEAD], dtype=torch.long
            ).reshape(-1, MOVE_AXES)
            move_valid = valid_flat if valid_flat is not None else torch.ones(
                (move_target.shape[0],), dtype=torch.bool, device=move_target.device,
            )
            move_is_real = bool(move_valid.any().item())
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
                axis_pred = move_pred[move_valid, axis_i, :]
                axis_target = move_target[move_valid, axis_i]
                if not move_is_real:
                    ce_axis = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                elif axis_name == "ud":
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
                        move_argmax = torch.argmax(move_pred, dim=-1)[move_valid]  # (B, 3)
                        move_target_v = move_target[move_valid]
                        metrics["acc_move"] = (move_argmax == move_target_v).float().mean().detach()
                        per_axis_macro_f1 = []
                        for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                            metrics[f"loss_move_{axis_name}"] = ce_per_axis[axis_i].detach()
                            pred_axis = move_argmax[:, axis_i]
                            true_axis = move_target_v[:, axis_i]
                            metrics[f"acc_move_{axis_name}"] = (pred_axis == true_axis).float().mean().detach()
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
                                metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = true_cls.float().mean().detach()
                                class_f1s.append(f1)
                            macro = torch.stack(class_f1s).mean()
                            metrics[f"f1_move_{axis_name}"] = macro.detach()
                            per_axis_macro_f1.append(macro)
                        metrics["f1_move"] = torch.stack(per_axis_macro_f1).mean().detach()

        if FIRE_HEAD in logits and FIRE_HEAD in actions:
            fire_logits = logits[FIRE_HEAD]
            fire_pred = fire_logits.reshape(-1)
            fire_target = self._tensor(actions[FIRE_HEAD], dtype=torch.float32).reshape(-1)
            if valid_flat is not None:
                fire_pred = fire_pred[valid_flat]
                fire_target = fire_target[valid_flat]
            fire_is_real = fire_target.numel() > 0
            # pos_weight conventionally lives in class_weights[FIRE_HEAD] (set
            # at training startup from corpus statistics: neg_count/pos_count).
            pos_weight: torch.Tensor | None = None
            if class_weights is not None and FIRE_HEAD in class_weights:
                cw = class_weights[FIRE_HEAD]
                pos_weight = cw if isinstance(cw, torch.Tensor) else torch.as_tensor(cw, device=fire_pred.device)
            if fire_is_real:
                fire_loss = F.binary_cross_entropy_with_logits(
                    fire_pred, fire_target, pos_weight=pos_weight, reduction="mean",
                )
            else:
                fire_loss = torch.zeros((), dtype=fire_logits.dtype, device=fire_logits.device)
            losses.append(fire_loss * weights_map.get(FIRE_HEAD, 1.0))
            loss_is_real.append(fire_is_real)
            if compute_metrics:
                metrics["loss_fire"] = fire_loss.detach()
                if fire_is_real:
                    with torch.no_grad():
                        pred_pos = (torch.sigmoid(fire_pred) > 0.5)
                        target_pos = fire_target > 0.5
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
        with self._autocast():
            _, logits, _, next_hidden, target_logits, target_query = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
                target_gt=target_gt_arr,
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
        with torch.inference_mode(), self._autocast():
            _, logits, _, next_hidden, target_logits, target_query = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
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
            "trunk_hidden": self.trunk_hidden,
            "gru_hidden": self.gru_hidden,
            "use_gru": self.use_gru,
            "use_weapon_head": self.use_weapon_head,
            "look_bypass_gru": self.look_bypass_gru,
            "weapon_switch_confidence": self.weapon_switch_confidence,
            "weapon_switch_margin": self.weapon_switch_margin,
            "jump_pos_weight": self.jump_pos_weight,
            "target_focal_gamma": self.target_focal_gamma,
            "weapon_use_gru": self.weapon_use_gru,
            "weapon_context_from_obs": self.weapon_context_from_obs,
            "gru_target_query": self.gru_target_query,
            "hard_target_feat": self.hard_target_feat,
            "weapon_in_target_query": self.weapon_in_target_query,
            "linear_slot_prior": self.linear_slot_prior,
            "head_bottleneck_dims": self.head_bottleneck_dims,
            "head_use_relu": self.head_use_relu,
            "head_activation": self.head_activation,
            "d_model": self.d_model,
            "head_hidden": self.head_hidden,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "attn_dropout": self.attn_dropout,
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
    def load(cls, path: str | Path, device: str = "auto") -> "QNNPolicy":
        source = Path(path)
        payload = trusted_torch_load(source, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
            raise ValueError(f"Unrecognised checkpoint format: {source}")
        meta = dict(payload["meta"])
        model = cls(
            obs_dim=int(meta.get("obs_dim", 0)),
            trunk_hidden=int(meta.get("d_model", meta["trunk_hidden"])),
            gru_hidden=int(meta.get("gru_hidden", 0)),
            use_gru=bool(meta.get("use_gru", False)),
            use_weapon_head=bool(meta.get("use_weapon_head", False)),
            look_bypass_gru=bool(meta.get("look_bypass_gru", False)),
            weapon_switch_confidence=float(meta.get("weapon_switch_confidence", 0.65)),
            weapon_switch_margin=float(meta.get("weapon_switch_margin", 0.15)),
            jump_pos_weight=float(meta.get("jump_pos_weight", 1.0)),
            target_focal_gamma=float(meta.get("target_focal_gamma", 0.0)),
            weapon_use_gru=bool(meta.get("weapon_use_gru", True)),
            # Legacy mapping: pre-refactor checkpoints used weapon_teacher_force
            # to gate label-vs-prediction at BC training. Under the new
            # semantics, motor heads read currently-held weapon from obs.
            #   teacher_force=True  → trained with embed(true_slot); obs at
            #     inference returns the same slot → context_from_obs=True
            #   teacher_force=False → trained with softmax-blended prediction;
            #     preserve inference behavior via context_from_obs=False
            weapon_context_from_obs=bool(
                meta["weapon_context_from_obs"]
                if "weapon_context_from_obs" in meta
                else meta.get("weapon_teacher_force", True)
            ),
            head_bottleneck_dim=(
                meta["head_bottleneck_dims"]
                if "head_bottleneck_dims" in meta
                else int(meta.get("head_bottleneck_dim", 0))
            ),
            head_use_relu=bool(meta.get("head_use_relu", False)),
            head_activation=meta.get("head_activation"),
            gru_target_query=bool(meta.get("gru_target_query", False)),
            hard_target_feat=bool(meta.get("hard_target_feat", False)),
            weapon_in_target_query=bool(meta.get("weapon_in_target_query", False)),
            linear_slot_prior=bool(meta.get("linear_slot_prior", False)),
            seed=0,
            device=device,
            d_model=int(meta.get("d_model", meta["trunk_hidden"])),
            n_heads=int(meta.get("n_heads", 2)),
            n_layers=int(meta.get("n_layers", 2)),
            ffn_dim=int(meta.get("ffn_dim", 256)),
            attn_dropout=float(meta.get("attn_dropout", 0.0)),
        )
        from qnn.utils.checkpoint_converter import (
            migrate_drop_action_history,
            migrate_drop_fire_align_scalar,
            migrate_entity_embed,
            migrate_self_scalars,
            migrate_v17_move_heads,
        )

        migrate_entity_embed(payload["state_dict"])
        migrate_self_scalars(payload["state_dict"])
        migrate_v17_move_heads(payload["state_dict"])
        migrate_drop_action_history(payload["state_dict"])
        migrate_drop_fire_align_scalar(payload["state_dict"])
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
            missing, unexpected = model.model.load_state_dict(payload["state_dict"], strict=False)
            allowed_missing_prefixes = (
                "weapon_head.", "weapon_embed.",
            )
            allowed_unexpected_prefixes = (
                "trunk.gru_input_proj.",  # pre-v20: mean-actors pool projection
            )
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
        model.model.to(model.device)
        return model

    @classmethod
    def load_for_finetune(
        cls,
        path: str | Path,
        *,
        use_gru: bool,
        gru_hidden: int,
        device: str = "auto",
    ) -> "QNNPolicy":
        del use_gru, gru_hidden
        return cls.load(path, device=device)
