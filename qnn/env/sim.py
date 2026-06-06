"""Offline training sim for a static token scene.

Reads a tokens.json (start from src/qnn/env/templates/sim_tokens.json), builds
a full production-shape obs dict for the described scene, and trains the look
head via REINFORCE — no engine, no SF, no docker.

Each tick:
  1. Model emits look action (move/fire/switch held at no-op).
  2. Action → pitch/yaw deltas via atan2 (mirrors src/engine/nq/qnn_input.c).
  3. Every kind="entity" token's rel/dist is recomputed in the new view frame,
     keeping the world position frozen.
  4. Reward = cos(view_forward, unit_rel_to_target), matching QNN_TrackingCosine.
  5. REINFORCE policy-gradient update.

Target defaults to the nearest actor-type entity (same rule as QNN_TrackingCosine),
or you can mark one actor token with ``"target": true`` in the scene JSON.

Usage:
    python -m qnn.env.sim --tokens path/to/tokens.json --episodes 3000
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from qnn.model.policy import QNNPolicy
from qnn.schema import OBS_SCHEMA, SPATIAL_TOKEN_COUNT
from qnn.vocab import (
    ACTION_IDS, ENTITY_IDS, MAX_TOKEN_OBJECTS,
    MODALITY_IDS, SPATIAL_SECTOR_IDS,
    TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER, TOKEN_PROJECTILE,
)


# ── Per-token scalar field layouts ───────────────────────────────────
# Matches src/engine/common/qnn_io.c's wire-format packing.
# Each entry: field_name -> (offset, length) inside that token's scalar idx.

_ACTOR_LAYOUT: Dict[str, Tuple[int, int]] = {
    "half_extents": (0, 3),
    "rel":         (3, 3),
    "dist":        (6, 1),
    "vel":         (7, 3),
    "path":        (10, 3),
    "path_dist":   (13, 1),
    "eta":         (14, 1),
    "facing":      (15, 1),
    "team":        (16, 1),
    "score":       (17, 1),
    "recency":     (18, 1),
}
_ITEM_LAYOUT: Dict[str, Tuple[int, int]] = {
    "half_extents": (0, 3),
    "rel":         (3, 3),
    "dist":        (6, 1),
    "path":        (7, 3),
    "path_dist":   (10, 1),
    "eta":         (11, 1),
    "amount":      (12, 1),
    "regen":       (13, 1),
    "recency":     (14, 1),
}
_MOVER_LAYOUT: Dict[str, Tuple[int, int]] = {
    "half_extents": (0, 3),
    "rel":         (3, 3),
    "dist":        (6, 1),
    "path":        (7, 3),
    "path_dist":   (10, 1),
    "eta":         (11, 1),
    "state":       (12, 1),
    "recency":     (13, 1),
}
_PROJECTILE_LAYOUT: Dict[str, Tuple[int, int]] = {
    "rel":     (0, 3),
    "dist":    (3, 1),
    "vel":     (4, 3),
    "recency": (7, 1),
}
_ENTITY_TYPE_NAMES = {
    "projectile": TOKEN_PROJECTILE,
    "actor":      TOKEN_ACTOR,
    "item":       TOKEN_ITEM,
    "mover":      TOKEN_MOVER,
}
_ENTITY_LAYOUTS = {
    TOKEN_PROJECTILE: _PROJECTILE_LAYOUT,
    TOKEN_ACTOR:      _ACTOR_LAYOUT,
    TOKEN_ITEM:       _ITEM_LAYOUT,
    TOKEN_MOVER:      _MOVER_LAYOUT,
}

_SELF_SCALAR_LAYOUT: Dict[str, int] = {
    "health": 0, "armor": 1,
    "weapon_sg": 2, "weapon_ssg": 3, "weapon_ng": 4, "weapon_sng": 5,
    "weapon_gl": 6, "weapon_rl": 7, "weapon_lg": 8,
    "ammo_shells": 9, "ammo_nails": 10, "ammo_rockets": 11, "ammo_cells": 12,
    # "vel" handled separately — maps to indices 13, 14, 15.
    "attack_finished": 16,  # normalized seconds remaining over QNN_TIME_SCALE (60s)
}
_SPATIAL_LAYOUT: Dict[str, Tuple[int, int]] = {
    "dir":          (0, 3),
    "nearest_dist": (3, 1),
    "mean_dist":    (4, 1),
    "openness":     (5, 1),
    "clearance":    (6, 1),
    "traversable":  (7, 1),
    "dropoff":      (8, 1),
    "solid_frac":   (9, 1),
    "water_frac":   (10, 1),
    "slime_frac":   (11, 1),
    "lava_frac":    (12, 1),
}


# ── Vocab name lookups ───────────────────────────────────────────────

def _lookup_vocab(name: Any, table: Dict[str, int], what: str) -> int:
    if name is None:
        return 0
    if not isinstance(name, str):
        raise ValueError(
            f"{what} must be a vocab name (string), got {type(name).__name__} {name!r}"
        )
    if name not in table:
        raise ValueError(
            f"Unknown {what} name {name!r}. Valid names: {sorted(table)}"
        )
    return int(table[name])


# ── Quake math (roll=0) — ports qnn.h AngleVectors + RelativeFrame ───

def _angle_basis(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    cp, sp = float(np.cos(p)), float(np.sin(p))
    cy, sy = float(np.cos(y)), float(np.sin(y))
    forward = np.array([cp * cy, cp * sy, -sp], dtype=np.float32)
    right   = np.array([sy, -cy, 0.0], dtype=np.float32)
    up      = np.array([sp * cy, sp * sy, cp], dtype=np.float32)
    return np.stack([forward, right, up], axis=0)


def _action_to_angle_delta(look: np.ndarray) -> Tuple[float, float]:
    fwd        = float(np.clip(look[0], -1.0, 1.0))
    yaw_comp   = float(np.clip(look[1], -1.0, 1.0))
    pitch_comp = float(np.clip(-look[2], -1.0, 1.0))
    yaw_deg   = float(np.degrees(np.arctan2(yaw_comp,   fwd)))
    pitch_deg = float(np.degrees(np.arctan2(pitch_comp, fwd)))
    return pitch_deg, yaw_deg


# ── Obs builder ──────────────────────────────────────────────────────

def _zero_obs() -> Dict[str, np.ndarray]:
    obs: Dict[str, np.ndarray] = {}
    for key, shape in OBS_SCHEMA.items():
        is_int = (
            key.endswith("_id") or key.endswith("_ids") or key == "entity_types"
            or key.startswith("entity_event_")
        )
        obs[key] = np.zeros(shape, dtype=np.int32 if is_int else np.float32)
    # entity_types: "no token" sentinel is -1 (see transformer.py:154).
    obs["entity_types"][:] = -1
    return obs


def _write_scalar_field(
    dest: np.ndarray, value: Any, offset: int, length: int, field_name: str,
) -> None:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float32))
    if arr.size != length:
        raise ValueError(
            f"Field {field_name!r} expects {length} value(s), got {arr.size}"
        )
    dest[offset:offset + length] = arr.reshape(-1)


_POWERUP_ROUTING = {
    "QUAD":       ("self_arsenal_powerup_ids", 0),
    "PENT":       ("self_state_powerup_ids",   0),
    "RING":       ("self_state_powerup_ids",   1),
    "MEGAHEALTH": ("self_state_powerup_ids",   2),
    "SUIT":       ("self_motion_powerup_ids",  0),
}


def _write_self_token(obs: Dict[str, np.ndarray], tok: Dict[str, Any]) -> None:
    for field, idx in _SELF_SCALAR_LAYOUT.items():
        if field in tok:
            obs["self_scalars"][idx] = float(tok[field])
    if "vel" in tok:
        vel = np.asarray(tok["vel"], dtype=np.float32)
        if vel.size != 3:
            raise ValueError(f"self.vel must have 3 values, got {vel.size}")
        obs["self_scalars"][13:16] = vel
    if "weapon_id" in tok:
        obs["self_weapon_id"][0] = _lookup_vocab(tok["weapon_id"], ENTITY_IDS, "self.weapon_id")
    if "armor_type_id" in tok:
        obs["self_armor_type_id"][0] = _lookup_vocab(tok["armor_type_id"], ENTITY_IDS, "self.armor_type_id")
    if "powerup_ids" in tok:
        ids = tok["powerup_ids"]
        if not isinstance(ids, list):
            raise ValueError("self.powerup_ids must be a list")
        for i, name in enumerate(ids):
            if name not in _POWERUP_ROUTING:
                raise ValueError(
                    f"self.powerup_ids[{i}] = {name!r} unknown; valid: "
                    f"{sorted(_POWERUP_ROUTING)}"
                )
            key, slot = _POWERUP_ROUTING[name]
            obs[key][slot] = _lookup_vocab(name, ENTITY_IDS, f"self.powerup_ids[{i}]")
    if "movement_id" in tok:
        obs["self_movement_id"][0] = int(tok["movement_id"])

    # Derive subtoken scalar tensors from the legacy slot layout so the
    # dequant's "self_state_scalars in obs" short-circuit sees real
    # values rather than zeros from _zero_obs. view_pitch defaults to 0
    # — probes don't currently expose pitch as a separate field.
    obs["self_state_scalars"][:]   = obs["self_scalars"][0:2]
    obs["self_arsenal_scalars"][:] = obs["self_scalars"][16:17]
    obs["self_motion_scalars"][:3] = obs["self_scalars"][13:16]
    obs["self_motion_scalars"][3]  = float(tok.get("view_pitch", 0.0))


def _write_spatial_token(obs: Dict[str, np.ndarray], tok: Dict[str, Any], index: int) -> None:
    sector_idx = index - 1  # production: spatial tokens at stream indices 1..9
    if not 0 <= sector_idx < SPATIAL_TOKEN_COUNT:
        raise ValueError(
            f"spatial token index {index} out of range [1, {SPATIAL_TOKEN_COUNT}]"
        )
    # sector name is a label, not written to the obs tensor — but we still
    # validate it for sanity.
    if "sector" in tok:
        expected = {v: k for k, v in SPATIAL_SECTOR_IDS.items()}[sector_idx]
        if tok["sector"] != expected:
            raise ValueError(
                f"spatial token at index {index} has sector={tok['sector']!r}; "
                f"expected {expected!r} for that stream position"
            )
    dest = obs["spatial_scalars"][sector_idx]
    for field, (off, length) in _SPATIAL_LAYOUT.items():
        if field in tok:
            _write_scalar_field(dest, tok[field], off, length, f"spatial.{field}")


def _write_entity_token(obs: Dict[str, np.ndarray], tok: Dict[str, Any], index: int) -> int:
    idx = index - (1 + SPATIAL_TOKEN_COUNT)  # 10..25 → 0..15
    if not 0 <= idx < MAX_TOKEN_OBJECTS:
        raise ValueError(
            f"entity token index {index} out of range [12, {12 + MAX_TOKEN_OBJECTS - 1}]"
        )
    type_name = tok.get("type")
    if type_name not in _ENTITY_TYPE_NAMES:
        raise ValueError(
            f"entity token at index {index} has unknown type {type_name!r}. "
            f"Valid: {sorted(_ENTITY_TYPE_NAMES)}"
        )
    tok_type = _ENTITY_TYPE_NAMES[type_name]
    layout = _ENTITY_LAYOUTS[tok_type]

    obs["entity_types"][idx] = tok_type
    obs["entity_ids"][idx, 0] = _lookup_vocab(tok.get("subject"),  ENTITY_IDS,   f"entity[{index}].subject")
    obs["entity_ids"][idx, 1] = _lookup_vocab(tok.get("modality"), MODALITY_IDS, f"entity[{index}].modality")
    if tok_type == TOKEN_ACTOR:
        obs["entity_ids"][idx, 2] = int(tok.get("player_id", 0))

    dest = obs["entity_scalars_raw"][idx]
    for field, (off, length) in layout.items():
        if field in tok:
            _write_scalar_field(dest, tok[field], off, length, f"entity[{index}].{field}")

    events = tok.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"entity[{index}].events must be a list")
    n_evt = min(len(events), obs["entity_event_actions"].shape[1])
    for i, ev in enumerate(events[:n_evt]):
        obs["entity_event_actions"][idx, i] = _lookup_vocab(ev.get("action"), ACTION_IDS,  f"entity[{index}].events[{i}].action")
        obs["entity_event_sources"][idx, i] = _lookup_vocab(ev.get("source"), ENTITY_IDS,  f"entity[{index}].events[{i}].source")
    obs["entity_event_counts"][idx] = n_evt
    return idx


# ── Scene loader ─────────────────────────────────────────────────────

def load_scene(
    path: Path,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int, np.ndarray]], np.ndarray, int]:
    """Parse a tokens.json.

    Returns ``(zero_view_obs, [(idx, tok_type, world_pos)], target_world_pos, target_idx)``.
    """
    payload = json.loads(Path(path).read_text())
    tokens = payload.get("tokens", [])
    if not isinstance(tokens, list):
        raise ValueError(f"{path}: 'tokens' must be a list")

    obs = _zero_obs()
    entity_worlds: List[Tuple[int, int, np.ndarray]] = []
    designated_target_idx: int | None = None
    designated_target_world: np.ndarray | None = None

    for tok in tokens:
        if not isinstance(tok, dict):
            raise ValueError(f"{path}: token entries must be objects, got {type(tok).__name__}")
        if "index" not in tok or "kind" not in tok:
            raise ValueError(f"{path}: token missing 'index' or 'kind': {tok!r}")
        kind = tok["kind"]
        index = int(tok["index"])

        if kind == "self":
            if index != 0:
                raise ValueError(f"self token must be at index 0, got {index}")
            _write_self_token(obs, tok)
        elif kind == "spatial":
            _write_spatial_token(obs, tok, index)
        elif kind == "entity":
            idx = _write_entity_token(obs, tok, index)
            tok_type = int(obs["entity_types"][idx])
            layout = _ENTITY_LAYOUTS[tok_type]
            rel_off, _ = layout["rel"]
            world_pos = obs["entity_scalars_raw"][idx, rel_off:rel_off + 3].copy()
            entity_worlds.append((idx, tok_type, world_pos))
            if bool(tok.get("target", False)):
                if tok_type != TOKEN_ACTOR:
                    raise ValueError(f"{path}: target=true is only valid for actor tokens")
                designated_target_idx = idx
                designated_target_world = world_pos.copy()
        else:
            raise ValueError(f"Unknown kind {kind!r} at index {index}. Valid: self/spatial/entity")

    if designated_target_idx is not None and designated_target_world is not None:
        return obs, entity_worlds, designated_target_world, designated_target_idx

    # Target = nearest actor (matches QNN_TrackingCosine).
    actors = [(s, t, wp) for (s, t, wp) in entity_worlds if t == TOKEN_ACTOR and np.linalg.norm(wp) > 1e-6]
    if not actors:
        raise RuntimeError(
            f"{path}: no actor-type entity with non-zero rel; can't pick a tracking target"
        )
    actors.sort(key=lambda e: float(np.linalg.norm(e[2])))
    target_idx = int(actors[0][0])
    target_world = actors[0][2].copy()
    return obs, entity_worlds, target_world, target_idx


def _active_actor_indices(
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
) -> List[int]:
    actor_indices: List[int] = []
    for idx, tok_type, world_pos in entity_worlds:
        if tok_type != TOKEN_ACTOR:
            continue
        if np.linalg.norm(world_pos) <= 1e-6:
            continue
        if int(obs_template["entity_ids"][idx, 2]) <= 0:
            continue
        actor_indices.append(int(idx))
    return actor_indices


def _shuffle_actor_indices(
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_idx: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int, np.ndarray]], int]:
    actor_indices = _active_actor_indices(obs_template, entity_worlds)
    if len(actor_indices) <= 1:
        return obs_template, entity_worlds, target_idx

    permuted_sources = list(rng.permutation(actor_indices))
    if permuted_sources == actor_indices:
        permuted_sources = actor_indices[1:] + actor_indices[:1]

    shuffled = {key: value.copy() for key, value in obs_template.items()}
    world_by_idx = {int(idx): (int(tok_type), world_pos.copy()) for idx, tok_type, world_pos in entity_worlds}
    src_to_dest: Dict[int, int] = {}

    for dest_idx, src_idx in zip(actor_indices, permuted_sources):
        shuffled["entity_types"][dest_idx] = obs_template["entity_types"][src_idx]
        shuffled["entity_ids"][dest_idx] = obs_template["entity_ids"][src_idx]
        shuffled["entity_scalars_raw"][dest_idx] = obs_template["entity_scalars_raw"][src_idx]
        shuffled["entity_event_actions"][dest_idx] = obs_template["entity_event_actions"][src_idx]
        shuffled["entity_event_sources"][dest_idx] = obs_template["entity_event_sources"][src_idx]
        shuffled["entity_event_counts"][dest_idx] = obs_template["entity_event_counts"][src_idx]
        src_to_dest[int(src_idx)] = int(dest_idx)

    shuffled_worlds: List[Tuple[int, int, np.ndarray]] = []
    for idx, tok_type, world_pos in entity_worlds:
        new_idx = src_to_dest.get(int(idx), int(idx))
        if int(tok_type) == TOKEN_ACTOR and int(idx) in world_by_idx:
            _, mapped_world = world_by_idx[int(idx)]
            shuffled_worlds.append((new_idx, int(tok_type), mapped_world.copy()))
        else:
            shuffled_worlds.append((new_idx, int(tok_type), world_pos.copy()))

    return shuffled, shuffled_worlds, src_to_dest.get(int(target_idx), int(target_idx))


# ── Sim dynamics ─────────────────────────────────────────────────────

def _apply_view(
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    view_angles: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    """Recompute rel/dist for every live entity token under new view angles."""
    basis = _angle_basis(view_angles[0], view_angles[1])
    obs = {k: v.copy() for k, v in obs_template.items()}
    scalars = obs["entity_scalars_raw"]
    for idx, tok_type, world_pos in entity_worlds:
        layout = _ENTITY_LAYOUTS[tok_type]
        rel_off, _ = layout["rel"]
        dist_off, _ = layout["dist"]
        rel_view = basis @ world_pos
        scalars[idx, rel_off:rel_off + 3] = rel_view
        scalars[idx, dist_off] = float(np.linalg.norm(world_pos))
    return obs


def _tracking_cos(view_angles: Tuple[float, float], target_world: np.ndarray) -> float:
    basis = _angle_basis(view_angles[0], view_angles[1])
    forward = basis[0]
    norm = float(np.linalg.norm(target_world))
    if norm < 1e-6:
        return 0.0
    return float(np.dot(forward, target_world) / norm)


# ── Model forward + sampling ─────────────────────────────────────────

_INT_KEYS = {
    "entity_types", "entity_ids",
    "entity_event_actions", "entity_event_sources", "entity_event_counts",
    "self_weapon_id", "self_armor_type_id", "self_movement_id",
    "self_state_powerup_ids", "self_arsenal_powerup_ids", "self_motion_powerup_ids",
}


def _to_tensor_batch(frames: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in OBS_SCHEMA:
        stacked = np.stack([f[key] for f in frames], axis=0)
        dtype = torch.int32 if key in _INT_KEYS else torch.float32
        out[key] = torch.as_tensor(stacked, dtype=dtype, device=device)
    return out


def _sample_look(
    policy: QNNPolicy, obs_batch: Dict[str, torch.Tensor], stddev: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, logits, _, _, _ = policy._forward_tensors(obs_batch)
    mean = F.normalize(logits["look"], dim=-1)  # look_cosine semantics
    dist = torch.distributions.Normal(mean, stddev)
    sampled = dist.sample()  # detached → gradient flows through mean in log_prob
    log_prob = dist.log_prob(sampled).sum(dim=-1)
    return sampled, log_prob


def _target_probe_step(
    policy: QNNPolicy,
    obs_batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    _, _, _, _, target_logits = policy._forward_tensors(obs_batch)
    probs = F.softmax(target_logits, dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1)[0].item())
    return target_logits[0], probs[0], pred_idx


# ── Training loop ────────────────────────────────────────────────────

def _run_episode(
    policy: QNNPolicy,
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_world: np.ndarray,
    steps: int, stddev: float, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (log_probs [T], rewards [T], values [T], entropies [T])."""
    view_angles = [0.0, 0.0]
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    values: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    for _ in range(steps):
        obs_frame = _apply_view(obs_template, entity_worlds, tuple(view_angles))
        obs_batch = _to_tensor_batch([obs_frame], device)
        _, logits_dict, values_t, _, _ = policy._forward_tensors(obs_batch)
        mean = F.normalize(logits_dict["look"], dim=-1)  # look_cosine
        dist = torch.distributions.Normal(mean, stddev)
        sampled = dist.sample()  # detached — gradient via mean in log_prob
        log_prob = dist.log_prob(sampled).sum(dim=-1)  # (1,)
        entropy = dist.entropy().sum(dim=-1)           # (1,)

        look = sampled[0].detach().cpu().numpy()
        pitch_d, yaw_d = _action_to_angle_delta(look)
        view_angles[1] = (view_angles[1] - yaw_d) % 360.0
        view_angles[0] = max(-70.0, min(80.0, view_angles[0] + pitch_d))
        rewards.append(_tracking_cos(tuple(view_angles), target_world))
        log_probs.append(log_prob[0])
        values.append(values_t.reshape(-1)[0])
        entropies.append(entropy[0])
    return (
        torch.stack(log_probs),
        torch.as_tensor(rewards, dtype=torch.float32, device=device),
        torch.stack(values),
        torch.stack(entropies),
    )


def _target_topk_summary(
    obs_frame: Dict[str, np.ndarray],
    probs: torch.Tensor,
    *,
    topk: int,
) -> str:
    probs_np = probs.detach().cpu().numpy()
    order = np.argsort(-probs_np)[:topk]
    parts: List[str] = []
    for idx in order:
        prob = float(probs_np[idx])
        pid = int(obs_frame["entity_ids"][idx, 2])
        rel = obs_frame["entity_scalars_raw"][idx, _ACTOR_LAYOUT["rel"][0]:_ACTOR_LAYOUT["rel"][0] + 3]
        parts.append(f"s{int(idx)}:p{pid}@{prob:.3f} rel={np.round(rel, 3).tolist()}")
    return " | ".join(parts)


def run_target_probe(
    policy: QNNPolicy,
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_world: np.ndarray,
    target_idx: int,
    *,
    steps: int,
    log_every: int,
    device: torch.device,
    driver: str = "oracle",
    shuffle_actors: bool = False,
    rng: np.random.Generator | None = None,
    topk: int = 3,
) -> None:
    """Inspect TargetPointer predictions over a short rollout."""
    if shuffle_actors:
        if rng is None:
            rng = np.random.default_rng(0)
        obs_template, entity_worlds, target_idx = _shuffle_actor_indices(
            obs_template, entity_worlds, target_idx, rng,
        )

    view_angles = [0.0, 0.0]
    correct = 0
    for step in range(steps):
        obs_frame = _apply_view(obs_template, entity_worlds, tuple(view_angles))
        obs_batch = _to_tensor_batch([obs_frame], device)
        _target_logits, probs, pred_idx = _target_probe_step(policy, obs_batch)
        target_prob = float(probs[target_idx].item())
        pred_prob = float(probs[pred_idx].item())
        is_correct = pred_idx == target_idx
        correct += int(is_correct)

        if step % log_every == 0 or step == steps - 1:
            target_pid = int(obs_frame["entity_ids"][target_idx, 2])
            pred_pid = int(obs_frame["entity_ids"][pred_idx, 2])
            topk_summary = _target_topk_summary(obs_frame, probs, topk=topk)
            print(
                f"step {step:3d}  target_idx={target_idx} pid={target_pid}  "
                f"pred_idx={pred_idx} pid={pred_pid}  "
                f"target_p={target_prob:.3f} pred_p={pred_prob:.3f}  "
                f"correct={is_correct}  {topk_summary}",
                flush=True,
            )

        if driver == "oracle":
            basis = _angle_basis(view_angles[0], view_angles[1])
            rel_view = basis @ target_world
            rel_norm = float(np.linalg.norm(rel_view))
            if rel_norm < 1e-6:
                continue
            look_np = rel_view / rel_norm
        else:
            _, logits, _, _, _ = policy._forward_tensors(obs_batch)
            look_np = F.normalize(logits["look"], dim=-1)[0].detach().cpu().numpy()
        pitch_d, yaw_d = _action_to_angle_delta(look_np)
        view_angles[1] = (view_angles[1] - yaw_d) % 360.0
        view_angles[0] = max(-70.0, min(80.0, view_angles[0] + pitch_d))

    print(
        f"target probe summary: acc={correct / max(steps, 1):.3f} "
        f"({correct}/{steps})",
        flush=True,
    )


def train_target_sl(
    policy: QNNPolicy,
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_world: np.ndarray,
    target_idx: int,
    *,
    steps: int,
    lr: float,
    log_every: int,
    device: torch.device,
    shuffle_actors: bool = False,
    rng: np.random.Generator | None = None,
) -> None:
    """Supervised CE training on TargetPointer logits with optional actor-idx shuffling."""
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=lr)
    if rng is None:
        rng = np.random.default_rng(0)

    for step in range(steps):
        step_obs = obs_template
        step_worlds = entity_worlds
        step_target_idx = target_idx
        if shuffle_actors:
            step_obs, step_worlds, step_target_idx = _shuffle_actor_indices(
                obs_template, entity_worlds, target_idx, rng,
            )
        obs_frame = _apply_view(step_obs, step_worlds, (0.0, 0.0))
        obs_batch = _to_tensor_batch([obs_frame], device)
        target_logits, probs, pred_idx = _target_probe_step(policy, obs_batch)
        target_tensor = torch.tensor([step_target_idx], dtype=torch.long, device=device)
        loss = F.cross_entropy(target_logits.unsqueeze(0), target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            target_prob = float(probs[step_target_idx].item())
            print(
                f"step {step:6d}  loss={loss.item():+.4f}  "
                f"target_idx={step_target_idx} pred_idx={pred_idx}  "
                f"target_p={target_prob:.3f} correct={pred_idx == step_target_idx}",
                flush=True,
            )


def train_sl(
    policy: QNNPolicy,
    obs_template: Dict[str, np.ndarray],
    target_world: np.ndarray,
    *,
    steps: int, lr: float, log_every: int,
) -> None:
    """Supervised look-direction regression on the static token scene.

    No sim dynamics, no sampling — the model sees the same obs every step,
    and the loss is cosine between its look output and the unit vector toward
    the target (which equals the target's world_pos / |world_pos| since the
    obs is captured at view_angles=(0, 0)).
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=lr)

    device = policy.device
    target_unit = torch.as_tensor(
        target_world / max(float(np.linalg.norm(target_world)), 1e-6),
        dtype=torch.float32, device=device,
    )
    obs_batch = _to_tensor_batch([obs_template], device)

    for step in range(steps):
        _, logits, _, _, _ = policy._forward_tensors(obs_batch)
        pred = F.normalize(logits["look"], dim=-1)[0]
        cos = torch.dot(pred, target_unit)
        loss = 1.0 - cos

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(
                f"step {step:6d}  loss={loss.item():+.4f}  "
                f"cos_sim={cos.item():+.4f}",
                flush=True,
            )


def run_oracle(
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_world: np.ndarray,
    *,
    steps: int, log_every: int,
) -> None:
    """No model, no training — apply the ideal action each tick and log cos.

    If this doesn't reach cos ≈ 1 the sim math is wrong, because the
    oracle output IS the correct look action by construction.
    """
    view_angles = [0.0, 0.0]
    for step in range(steps):
        basis = _angle_basis(view_angles[0], view_angles[1])
        rel_view = basis @ target_world
        rel_norm = float(np.linalg.norm(rel_view))
        if rel_norm < 1e-6:
            print("oracle: rel is near zero, aligned", flush=True)
            return
        unit_rel = rel_view / rel_norm
        pitch_d, yaw_d = _action_to_angle_delta(unit_rel)
        view_angles[1] = (view_angles[1] - yaw_d) % 360.0
        view_angles[0] = max(-70.0, min(80.0, view_angles[0] + pitch_d))
        cos = _tracking_cos(tuple(view_angles), target_world)
        if step % log_every == 0 or step == steps - 1:
            print(
                f"step {step:3d}  view=({view_angles[0]:+7.2f}, {view_angles[1]:+7.2f})  "
                f"unit_rel={unit_rel.round(4).tolist()}  "
                f"yaw_d={yaw_d:+7.2f}  pitch_d={pitch_d:+7.2f}  "
                f"cos={cos:+.6f}",
                flush=True,
            )


def train_sl_episodic(
    policy: QNNPolicy,
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_world: np.ndarray,
    *,
    episodes: int, steps_per_episode: int, lr: float, log_every: int,
    driver: str = "model",
) -> None:
    """DAgger-style on-policy SL: model drives view each tick, ideal look supervises each step.

    At each tick: compute rel_view for the target under current view angles, use
    its unit vector as the supervised target for look.  Model's own (deterministic,
    normalized) output is applied via atan2 to update the view for the next tick —
    so the obs distribution is on-policy, but supervision is oracle.
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=lr)

    device = policy.device
    for ep in range(episodes):
        view_angles = [0.0, 0.0]
        optimizer.zero_grad()
        loss_acc = torch.zeros((), device=device)
        cos_acc = 0.0
        for _ in range(steps_per_episode):
            obs_frame = _apply_view(obs_template, entity_worlds, tuple(view_angles))
            obs_batch = _to_tensor_batch([obs_frame], device)
            _, logits, _, _, _ = policy._forward_tensors(obs_batch)
            raw = logits["look"][0]               # 3-dim, unnormalized
            pred = F.normalize(raw, dim=-1)        # unit vector for env / cos metric

            basis = _angle_basis(view_angles[0], view_angles[1])
            rel_view = basis @ target_world
            rel_norm = float(np.linalg.norm(rel_view))
            if rel_norm < 1e-6:
                break
            target_unit = torch.as_tensor(
                rel_view / rel_norm, dtype=torch.float32, device=device,
            )

            # Smooth L1 on raw logits against the unit target — avoids the
            # vanishing-gradient trap of cos(F.normalize(x), target) at x=target.
            loss_acc = loss_acc + F.smooth_l1_loss(raw, target_unit)
            cos_acc += float(torch.dot(pred, target_unit).item())

            # Drive view: "model" uses the model's output; "oracle" uses the
            # ideal action (which by construction is target_unit).  Oracle
            # driving gives a clean straight-shot trajectory (stable after
            # one tick), so the model is supervised on a narrow obs
            # distribution — an easier learning problem than self-driving.
            if driver == "oracle":
                look_np = target_unit.detach().cpu().numpy()
            else:
                look_np = pred.detach().cpu().numpy()
            pitch_d, yaw_d = _action_to_angle_delta(look_np)
            view_angles[1] = (view_angles[1] - yaw_d) % 360.0
            view_angles[0] = max(-70.0, min(80.0, view_angles[0] + pitch_d))

        loss_acc = loss_acc / steps_per_episode
        loss_acc.backward()
        optimizer.step()

        if ep % log_every == 0 or ep == episodes - 1:
            final_cos = _tracking_cos(tuple(view_angles), target_world)
            print(
                f"ep {ep:6d}  loss={loss_acc.item():+.4f}  "
                f"cos_mean={cos_acc / steps_per_episode:+.4f}  "
                f"final_cos={final_cos:+.4f}",
                flush=True,
            )


def train_rl(
    policy: QNNPolicy,
    obs_template: Dict[str, np.ndarray],
    entity_worlds: List[Tuple[int, int, np.ndarray]],
    target_world: np.ndarray,
    *,
    episodes: int, steps_per_episode: int, batch_episodes: int,
    lr: float, stddev: float, gamma: float, log_every: int,
    value_coef: float = 0.5, entropy_coef: float = 0.01,
    normalize_adv: bool = False,
) -> None:
    """Actor-critic policy gradient with value baseline.

    * Value head (from QNNPolicy._forward_tensors's returned `values`) provides
      a learned baseline, so advantage = returns - V(s) has lower variance
      than the naive "batch mean" baseline we used before.
    * Value loss (MSE on discounted returns) is added to the policy loss.
    * Entropy bonus on the Normal keeps exploration from collapsing.
    * Advantage normalization is OFF by default — it was the main cause of
      sign-flipping updates in the earlier REINFORCE attempts on this scene.
    """
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=lr)

    device = policy.device
    for ep in range(0, episodes, batch_episodes):
        batch_log_probs: List[torch.Tensor] = []
        batch_returns: List[torch.Tensor] = []
        batch_values: List[torch.Tensor] = []
        batch_entropies: List[torch.Tensor] = []
        ep_rewards: List[float] = []
        last_rewards: torch.Tensor | None = None
        for _ in range(batch_episodes):
            log_probs, rewards, values, entropies = _run_episode(
                policy, obs_template, entity_worlds, target_world,
                steps_per_episode, stddev, device,
            )
            returns = torch.zeros_like(rewards)
            acc = 0.0
            for t in reversed(range(rewards.shape[0])):
                acc = rewards[t].item() + gamma * acc
                returns[t] = acc
            batch_log_probs.append(log_probs)
            batch_returns.append(returns)
            batch_values.append(values)
            batch_entropies.append(entropies)
            ep_rewards.append(float(rewards.mean().item()))
            last_rewards = rewards

        all_lp = torch.cat(batch_log_probs)
        all_ret = torch.cat(batch_returns)
        all_val = torch.cat(batch_values)
        all_ent = torch.cat(batch_entropies)

        # Advantage = returns − V(s).  The value head learns the baseline.
        adv = (all_ret - all_val).detach()
        if normalize_adv and adv.std().item() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        policy_loss = -(all_lp * adv).mean()
        value_loss = F.mse_loss(all_val, all_ret)
        entropy_mean = all_ent.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep // batch_episodes) % log_every == 0:
            print(
                f"ep {ep:6d}  mean_reward={np.mean(ep_rewards):+.4f}  "
                f"min={float(last_rewards.min().item()):+.4f}  "
                f"max={float(last_rewards.max().item()):+.4f}  "
                f"V={all_val.mean().item():+.4f}  "
                f"pi_loss={policy_loss.item():+.4f}  "
                f"v_loss={value_loss.item():+.4f}  "
                f"H={entropy_mean.item():+.3f}",
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline training sim for a static token scene.")
    ap.add_argument("--tokens", type=Path, required=True, help="Path to tokens.json.")
    ap.add_argument(
        "--mode", choices=["oracle", "sl", "sl-episodic", "target-probe", "target-sl", "rl"], default="rl",
        help="oracle      = no model, apply ideal look each tick (sanity-check the sim math). "
             "sl          = static supervised cosine regression (same obs every step). "
             "target-probe= inspect TargetPointer predictions on a static scene. "
             "target-sl   = supervised CE training on TargetPointer logits. "
             "sl-episodic = on-policy SL: model drives view each tick, oracle supervises. "
             "rl          = REINFORCE with atan2 action → viewangle dynamics.",
    )
    ap.add_argument("--steps", type=int, default=400,
                    help="SL mode: number of SGD steps.")
    ap.add_argument("--episodes", type=int, default=3000,
                    help="RL mode: total number of episodes.")
    ap.add_argument("--steps-per-episode", type=int, default=64,
                    help="RL mode: ticks per episode.")
    ap.add_argument("--batch-episodes", type=int, default=16,
                    help="RL mode: episodes per gradient update.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stddev", type=float, default=0.2, help="RL mode only.")
    ap.add_argument("--gamma", type=float, default=0.99, help="RL mode only.")
    ap.add_argument(
        "--driver", choices=["model", "oracle"], default="model",
        help="sl-episodic only: who drives the view each tick. "
             "'oracle' = apply the ideal look (fast trajectory convergence); "
             "'model'  = apply the model's output (on-policy imitation).",
    )
    ap.add_argument(
        "--rl-curriculum", type=str, default=None,
        help="RL mode only.  Comma-separated list of (episodes, steps-per-episode) "
             "pairs; episode length grows as policy masters shorter horizons. "
             "Example: '800:1,400:2,400:4,400:8,400:16,400:32' — 2800 episodes "
             "total, episode length 1→32.  Overrides --episodes / --steps-per-episode.",
    )
    ap.add_argument("--value-coef", type=float, default=0.1, help="RL mode only.")
    ap.add_argument("--entropy-coef", type=float, default=0.01, help="RL mode only.")
    ap.add_argument("--normalize-adv", action="store_true", default=False, help="RL mode only.")
    ap.add_argument("--log-every", type=int, default=2)
    ap.add_argument("--topk", type=int, default=3, help="target-probe only: how many indices to print per step.")
    ap.add_argument("--shuffle-actors", action="store_true", default=False,
                    help="target-probe / target-sl: permute active actor indices so the target is not tied to a fixed token position.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    # Architecture knobs — defaults match runs/ppo/ppo_look_sanity/config/model.json.
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--gru-hidden", type=int, default=64)
    ap.add_argument(
        "--no-gru", dest="use_gru", action="store_false", default=True,
        help="Disable the GRU hidden-state core (stateless forward each tick).",
    )
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ffn-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_template, entity_worlds, target_world, target_idx = load_scene(args.tokens)
    print(f"scene: {args.tokens}")
    print(f"  entity tokens loaded: {len(entity_worlds)}")
    print(
        f"  target idx: {target_idx}  world_pos: {target_world}  "
        f"|target|={np.linalg.norm(target_world):.3f}"
    )

    from qnn.model.network import ModelConfig
    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        attn_dropout=0.0,
        use_gru=args.use_gru,
        gru_hidden=args.gru_hidden,
        use_weapon_head=False,
        weapon_switch_confidence=0.65,
        weapon_switch_margin=0.15,
        weapon_use_gru=False,
        weapon_use_self_readout=True,
        weapon_context_from_obs=False,
        look_bypass_gru=False,
        gru_target_query=False,
        hard_target_feat=False,
        weapon_in_target_query=False,
        linear_idx_prior=False,
        gt_dist_target_feat=False,
        prev_target_in_query=False,
        self_weapon_embed_in_self=False,
        head_bottleneck_dim={"move": 0, "look": 0, "attack": 0, "weapon": 0},
        head_activation="none",
    )
    policy = QNNPolicy(
        obs_dim=0,
        model=model_cfg,
        jump_pos_weight=1.0,
        seed=args.seed,
        device=args.device,
    )
    print(
        f"policy: d_model={policy.d_model} gru={policy.use_gru} "
        f"layers={policy.n_layers} device={policy.device}"
    )
    if args.mode == "oracle":
        print(f"mode=oracle steps={args.steps}")
        run_oracle(
            obs_template, entity_worlds, target_world,
            steps=args.steps, log_every=args.log_every,
        )
        return 0
    if args.mode == "sl":
        print(f"training: mode=sl steps={args.steps} lr={args.lr}")
        train_sl(
            policy, obs_template, target_world,
            steps=args.steps, lr=args.lr, log_every=args.log_every,
        )
    elif args.mode == "target-probe":
        print(
            f"mode=target-probe steps={args.steps} driver={args.driver} "
            f"shuffle_actors={args.shuffle_actors}"
        )
        run_target_probe(
            policy, obs_template, entity_worlds, target_world, target_idx,
            steps=args.steps, log_every=args.log_every, device=policy.device,
            driver=args.driver, shuffle_actors=args.shuffle_actors,
            rng=np.random.default_rng(args.seed), topk=args.topk,
        )
    elif args.mode == "target-sl":
        print(
            f"training: mode=target-sl steps={args.steps} lr={args.lr} "
            f"shuffle_actors={args.shuffle_actors}"
        )
        train_target_sl(
            policy, obs_template, entity_worlds, target_world, target_idx,
            steps=args.steps, lr=args.lr, log_every=args.log_every,
            device=policy.device, shuffle_actors=args.shuffle_actors,
            rng=np.random.default_rng(args.seed),
        )
        run_target_probe(
            policy, obs_template, entity_worlds, target_world, target_idx,
            steps=min(args.steps_per_episode, 8), log_every=1, device=policy.device,
            driver="oracle", shuffle_actors=args.shuffle_actors,
            rng=np.random.default_rng(args.seed + 1), topk=args.topk,
        )
    elif args.mode == "sl-episodic":
        print(
            f"training: mode=sl-episodic episodes={args.episodes} "
            f"steps-per-episode={args.steps_per_episode} lr={args.lr}"
        )
        train_sl_episodic(
            policy, obs_template, entity_worlds, target_world,
            episodes=args.episodes, steps_per_episode=args.steps_per_episode,
            lr=args.lr, log_every=args.log_every, driver=args.driver,
        )
    else:
        if args.rl_curriculum:
            phases = []
            for chunk in args.rl_curriculum.split(","):
                eps_str, spe_str = chunk.split(":")
                phases.append((int(eps_str), int(spe_str)))
            print(f"training: mode=rl curriculum={phases} (episodes:steps_per_episode)")
            for i, (phase_eps, phase_spe) in enumerate(phases):
                print(f"\n== Phase {i+1}/{len(phases)}: {phase_eps} episodes × {phase_spe} ticks ==",
                      flush=True)
                train_rl(
                    policy, obs_template, entity_worlds, target_world,
                    episodes=phase_eps, steps_per_episode=phase_spe,
                    batch_episodes=args.batch_episodes, lr=args.lr, stddev=args.stddev,
                    gamma=args.gamma, log_every=args.log_every,
                    value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                    normalize_adv=args.normalize_adv,
                )
        else:
            print(
                f"training: mode=rl episodes={args.episodes} "
                f"steps={args.steps_per_episode} batch={args.batch_episodes} "
                f"lr={args.lr} stddev={args.stddev} gamma={args.gamma}"
            )
            train_rl(
                policy, obs_template, entity_worlds, target_world,
                episodes=args.episodes, steps_per_episode=args.steps_per_episode,
                batch_episodes=args.batch_episodes, lr=args.lr, stddev=args.stddev,
                gamma=args.gamma, log_every=args.log_every,
                value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                normalize_adv=args.normalize_adv,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
