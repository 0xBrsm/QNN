"""Per-frame feature builders for flat-feature head-probe ablations.

Each builder takes a torch ``(obs_dict, target_dist_slot)`` pair where
``obs_dict`` carries per-frame tensors keyed under the canonical BC
naming (``self_scalars``, ``self_weapon_id``, ``entity_scalars_raw``,
…) and returns a ``(N, dim)`` float32 tensor. ``N`` is the flat batch
size (``T*B`` if the BC supervised loop is operating on a sequence
chunk); the model's forward flattens before calling these builders.

Builders are registered by name; head specs compose features by naming
them and the runner builds an MLP of the resulting concat width. The
oracle-pointer probes pool actor tokens by ``target_dist_slot`` (a
16-slot renormalized distribution supplied by the BC trainer from
``actions.target_dist``); features that don't need it accept ``None``.

This registry is consumed by ``qnn.bc.heads.flat.FlatFeatureHead``,
which conforms to ``_CombatObjectiveNet.forward`` so the canonical BC
trainer drives it through ``QNNPolicy(model_factory=...)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from qnn.bc.target_labeler import (
    _ACTOR_REL_OFFSET,
    _ACTOR_VEL_OFFSET,
)
from qnn.model.policy import WEAPON_HEAD_SIZE


_N_SLOTS = 16
_FeatureFn = Callable[[Mapping[str, torch.Tensor], torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True, slots=True)
class FeatureBuilder:
    """One per-frame feature in the registry.

    ``dim`` is the static output width — probes use it to size the MLP
    input. ``fn`` accepts the canonical BC obs dict + the
    GT-renormalized 16-slot target distribution (or ``None`` if the
    feature doesn't need it) and returns a ``(N, dim)`` float32 tensor.
    """
    name: str
    dim: int
    fn: _FeatureFn


FEATURE_REGISTRY: dict[str, FeatureBuilder] = {}


def register_feature(name: str, dim: int) -> Callable[[_FeatureFn], _FeatureFn]:
    """Decorator: register a feature builder by name."""
    def _wrap(fn: _FeatureFn) -> _FeatureFn:
        if name in FEATURE_REGISTRY:
            raise ValueError(f"feature {name!r} already registered")
        FEATURE_REGISTRY[name] = FeatureBuilder(name=name, dim=int(dim), fn=fn)
        return fn
    return _wrap


def build_feature_vector(
    feature_names: tuple[str, ...],
    obs: Mapping[str, torch.Tensor],
    target_dist_slot: torch.Tensor | None,
) -> torch.Tensor:
    """Concatenate the named features along the last axis into a (N, F) tensor."""
    parts: list[torch.Tensor] = []
    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            raise KeyError(
                f"unknown feature {name!r}; registered: {sorted(FEATURE_REGISTRY)}"
            )
        builder = FEATURE_REGISTRY[name]
        x = builder.fn(obs, target_dist_slot)
        if x.ndim != 2 or x.shape[1] != builder.dim:
            raise RuntimeError(
                f"feature {name!r}: expected (N, {builder.dim}) got {tuple(x.shape)}"
            )
        parts.append(x.to(dtype=torch.float32))
    if not parts:
        # Fallback: empty feature set yields zero-width tensor with the
        # right leading dim. Pick any obs entry to read N from.
        any_t = next(iter(obs.values()))
        return torch.zeros((any_t.shape[0], 0), dtype=torch.float32, device=any_t.device)
    return torch.cat(parts, dim=1)


def feature_vector_dim(feature_names: tuple[str, ...]) -> int:
    """Static input dimension given a list of feature names."""
    return sum(FEATURE_REGISTRY[n].dim for n in feature_names)


def _as_float(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float32) if t.dtype != torch.float32 else t


# ── self-state features ─────────────────────────────────────────────────────

@register_feature("self_scalars", dim=17)
def _self_scalars(obs, _tds):
    return _as_float(obs["self_scalars"])


@register_feature("self_health_armor", dim=2)
def _self_health_armor(obs, _tds):
    return _as_float(obs["self_scalars"])[:, 0:2]


@register_feature("self_weapon_owned", dim=7)
def _self_weapon_owned(obs, _tds):
    """Which weapons the player has in inventory (separate from currently held)."""
    return _as_float(obs["self_scalars"])[:, 2:9]


@register_feature("self_ammo", dim=4)
def _self_ammo(obs, _tds):
    """Per-pool ammo: shells, nails, rockets, cells (each normalized to its cap)."""
    return _as_float(obs["self_scalars"])[:, 9:13]


@register_feature("self_velocity", dim=3)
def _self_velocity(obs, _tds):
    """View-relative player velocity, normalized by QNN_VELOCITY_SCALE."""
    return _as_float(obs["self_scalars"])[:, 13:16]


@register_feature("self_attack_finished", dim=1)
def _self_attack_finished(obs, _tds):
    """Cooldown remaining in seconds, normalized by QNN_TIME_SCALE=60s
    (canonical time-scalar normalization shared with recency / eta /
    regen). 0 = ready; engine blocks both fire AND weapon switch while
    > 0. Single global timer set by the last fired weapon's W_Attack
    — see weapons.qc:1367."""
    return _as_float(obs["self_scalars"])[:, 16:17]


@register_feature("self_movement_one_hot", dim=3)
def _self_movement_one_hot(obs, _tds):
    mid = obs["self_movement_id"].long().reshape(-1)
    out = torch.zeros((mid.shape[0], 3), dtype=torch.float32, device=mid.device)
    out[mid == 0, 0] = 1.0
    out[mid == 1, 1] = 1.0
    out[mid >= 2, 2] = 1.0
    return out


@register_feature("self_weapon_one_hot", dim=WEAPON_HEAD_SIZE + 1)
def _self_weapon_one_hot(obs, _tds):
    """One-hot over [no_weapon, axe..LG]. Width WEAPON_HEAD_SIZE+1 so
    the 'no weapon held' state is its own column (vs squashed onto axe).

    obs.self_weapon_id is ENTITY_IDS-encoded (0=NONE, 3..10=axe..LG):
    impulse = max(0, eid - 2). Inlined here, not via helper — see
    qnn.vocab for the encoding reference and qnn.model.transformer for
    why we inline (per-batch helper-call overhead on ROCm is real).
    """
    wid = (obs["self_weapon_id"].long().reshape(-1) - 2).clamp(
        0, WEAPON_HEAD_SIZE,
    )
    return F.one_hot(wid, num_classes=WEAPON_HEAD_SIZE + 1).to(torch.float32)


# ── target distribution as input (privileged) ───────────────────────────────

@register_feature("target_dist_slots", dim=_N_SLOTS)
def _target_dist_slots(_obs, target_dist_slot):
    """16-slot GT distribution renormalized by present, as supplied by the
    BC supervised loop. Privileged: oracle-pointer feature."""
    if target_dist_slot is None:
        raise RuntimeError(
            "feature 'target_dist_slots' requires target_dist_slot — "
            "ensure BC trainer is passing it (actions.target_dist driven)."
        )
    return _as_float(target_dist_slot)


# ── target-pooled actor features (privileged via target_dist_slot) ─────────

def _pool_actor_block(
    obs: Mapping[str, torch.Tensor],
    target_dist_slot: torch.Tensor | None,
    offset: int,
    n: int,
) -> torch.Tensor:
    """Soft-pool entity_scalars_raw[..., offset:offset+n] by renormalized target_dist."""
    if target_dist_slot is None:
        raise RuntimeError("target_pooled_* needs target_dist_slot from the BC trainer")
    es = _as_float(obs["entity_scalars_raw"])                         # (N, 16, 19)
    block = es[:, :, offset:offset + n]                                # (N, 16, n)
    slot = _as_float(target_dist_slot)                                # (N, 16)
    return (slot.unsqueeze(-1) * block).sum(dim=1)                     # (N, n)


@register_feature("target_pooled_rel", dim=3)
def _target_pooled_rel(obs, target_dist_slot):
    return _pool_actor_block(obs, target_dist_slot, _ACTOR_REL_OFFSET, 3)


@register_feature("target_pooled_vel", dim=3)
def _target_pooled_vel(obs, target_dist_slot):
    return _pool_actor_block(obs, target_dist_slot, _ACTOR_VEL_OFFSET, 3)


@register_feature("target_pooled_scalars", dim=19)
def _target_pooled_scalars(obs, target_dist_slot):
    """Full 19-dim soft-pooled actor scalar vector (rel + vel + dist +
    eta + facing + recency + etc.). The maximum-information target-aware
    feature; expensive but a clean upper bound for probes."""
    return _pool_actor_block(obs, target_dist_slot, 0, 19)


# ── target-head-style flat features (no privileged target_dist) ─────────────

@register_feature("entity_scalars_flat", dim=_N_SLOTS * 19)
def _entity_scalars_flat(obs, _tds):
    """Per-slot entity scalars flattened. Used by the target-head probe
    where the model has to identify the target without privileged info."""
    es = _as_float(obs["entity_scalars_raw"])
    return es.reshape(es.shape[0], _N_SLOTS * 19)
