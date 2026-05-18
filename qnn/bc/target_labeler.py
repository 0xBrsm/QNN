"""Combat target label generator for BC.

Produces a per-tick ``target`` integer label (0..MAX_TOKEN_OBJECTS-1, or
-100 for "no target, skip loss") from obs + action arrays.

Algorithm — fire-anchored sticky with adaptive acquire/release cones
(Schmitt-trigger hysteresis):

  Pass 1: walk fire ticks causally with a sticky current_pid.
    - If current_pid is in stream AND passes the (wide) release cone:
        keep current_pid; attribute this fire to it.
    - Else: argmax cos over in-cone enemies (per-enemy adaptive acquire
        cone) and acquire/transfer to that pid.

  Acquire cone(d) = clamp(atan(208/d), 5°, 30°)   (transverse 208u, capped)
  Release cone(d) = clamp(atan(416/d), 5°, 45°)   (transverse 416u, K=2 ratio)

  Pass 2 (unchanged): group consecutive valid_shots into engagements by
    same pid + continuous token-stream presence.

  Pass 3 (unchanged): extend each engagement's label backward toward the
    previous engagement's end and forward toward the next engagement's
    start, stopping at token-stream loss.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR

TARGET_IGNORE = -100

# Adaptive cone parameters. Transverse offsets are in Quake units; obs rel
# vectors are scaled by 1/QNN_DIST_SCALE (=1/1000) so we rescale before
# applying. Acquire cone admits enemies within 208u perpendicular of aim,
# capped at 30° at close range and floored at 5° at extreme range. Release
# cone is twice that (K=2 Schmitt-trigger ratio), capped at 45°.
QNN_DIST_SCALE = 1000.0
ACQUIRE_TRANSVERSE_U = 208.0
RELEASE_TRANSVERSE_U = 416.0
_ACQUIRE_CAP_COS = math.cos(math.radians(30.0))
_ACQUIRE_FLOOR_COS = math.cos(math.radians(5.0))
_RELEASE_CAP_COS = math.cos(math.radians(45.0))
_RELEASE_FLOOR_COS = math.cos(math.radians(5.0))


def _adaptive_cone_cos(dist_qu: np.ndarray, transverse_u: float,
                       cap_cos: float, floor_cos: float) -> np.ndarray:
    """Per-element adaptive cone threshold. dist_qu in Quake units."""
    safe_d = np.maximum(dist_qu, 1e-3)
    c = np.cos(np.arctan(transverse_u / safe_d))
    return np.clip(c, cap_cos, floor_cos)


# Actor scalar layout: rel is at offset 3, length 3.  See _ACTOR_LAYOUT in sim.py
# / qnn_actor_token_t in qnn_object.h.
_ACTOR_REL_OFFSET = 3
# team scalar is at offset 16 (after half_extents/rel/dist/vel/path/path_dist/eta/facing).
# Value == 1.0 means "same team as demonstrator" — these actors must never be
# eligible target candidates.
_ACTOR_TEAM_OFFSET = 16
_TEAM_TEAMMATE_VALUE = 1.0


_MODALITY_SIGHT = 0  # qnn_vocab.h: QNN_MODALITY_SIGHT


def label_enemy_target(
    obs: Dict[str, np.ndarray],
    actions: Dict[str, np.ndarray],
    sight_only: bool = False,
) -> np.ndarray:
    """Return a (T,) int64 array of target labels.

    Values are valid slot indices (0..MAX_TOKEN_OBJECTS-1) where a target is
    tracked in the token stream, or TARGET_IGNORE (-100) elsewhere.

    Algorithm:
      1. Sticky fire-anchored attribution with adaptive acquire/release cones.
         Maintain a current_pid across fire ticks. At each fire frame:
            - If current_pid is still in stream AND cos(look, pid) >= release
              cone threshold → keep current_pid (sticky hold).
            - Else → cone-argmax over enemies passing the acquire cone;
              acquire/transfer to that pid.
      2. Group consecutive fires into engagements: same pid AND the pid
         remained continuously in the token stream between the two fires.
         A pid switch or a token-stream break splits the engagement.
      3. For each engagement, walk backward from start_t and forward from
         end_t through the continuous in-stream run, bounded by adjacent
         engagements' fire times. Apply slot labels for the resulting
         contiguous range; engagements never overlap by construction.

    Visibility is already gated by the engine's token stream (recency keeps
    an entity present for ~2s after FOV loss); no additional angular cone
    is applied to the backward/forward walks.

    ``sight_only`` restricts the labeler's enemy-actor mask to slots in
    modality 0 (SIGHT).  Engagements break when the pid leaves SIGHT, so
    SOUND/MEMORY-tracked frames go unlabeled — reproduces the old
    collect-time ``--entity-filter pvs_actors`` behavior where the
    labeler couldn't see non-PVS modalities.
    """
    # ── Setup and Vector Math ────────────────────────────────────────
    entity_types = np.asarray(obs["entity_types"])
    entity_ids = np.asarray(obs["entity_ids"])
    entity_scalars = np.asarray(obs["entity_scalars_raw"])
    look = np.asarray(actions["look"])
    fire = np.asarray(actions["fire"]).reshape(-1)

    T = look.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=np.int64)

    actor_mask = (entity_types == TOKEN_ACTOR)
    teammate_mask = entity_scalars[:, :, _ACTOR_TEAM_OFFSET] == _TEAM_TEAMMATE_VALUE
    enemy_actor_mask = actor_mask & ~teammate_mask
    if sight_only:
        modality = entity_ids[:, :, 1]
        enemy_actor_mask &= (modality == _MODALITY_SIGHT)
    player_ids = entity_ids[:, :, 2]

    rel = entity_scalars[:, :, _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET + 3]
    rel_norm = np.linalg.norm(rel, axis=-1)                       # (T, 16) scaled
    dist_qu = rel_norm * QNN_DIST_SCALE                            # (T, 16) Quake units
    unit_rel = rel / np.maximum(rel_norm[..., None], 1e-6)

    look_norm = np.linalg.norm(look, axis=-1, keepdims=True)
    unit_look = look / np.maximum(look_norm, 1e-6)

    cos_tr = np.einsum("tij,tj->ti", unit_rel, unit_look)
    cos_actor = np.where(enemy_actor_mask, cos_tr, -np.inf)

    # Per-frame, per-enemy cone thresholds (adaptive to distance).
    acquire_thr = _adaptive_cone_cos(dist_qu, ACQUIRE_TRANSVERSE_U,
                                     _ACQUIRE_CAP_COS, _ACQUIRE_FLOOR_COS)
    release_thr = _adaptive_cone_cos(dist_qu, RELEASE_TRANSVERSE_U,
                                     _RELEASE_CAP_COS, _RELEASE_FLOOR_COS)

    # ── Pass 1: Sticky fire-anchored attribution ─────────────────────
    # Mirror the engine's causal release on stream loss: sticky is released
    # as soon as current_pid leaves the obs entity pool, even between fires.
    # That way a transient out-of-obs window during a fire-to-fire gap forces
    # a fresh acquire at the next fire (instead of a stale sticky-keep).
    fire_ticks = np.flatnonzero(fire == 1)
    valid_shots: list[tuple[int, int]] = []
    current_pid = 0
    prev_fire_t = -1

    for t in fire_ticks:
        t = int(t)
        # Stream-loss release: if current_pid was set at the previous fire,
        # check that it was in obs at EVERY frame between then and now.
        # A single out-of-obs frame in the gap releases the sticky.
        if current_pid > 0 and prev_fire_t >= 0 and t - prev_fire_t > 1:
            gap_in_stream = (enemy_actor_mask[prev_fire_t + 1:t] &
                             (player_ids[prev_fire_t + 1:t] == current_pid)).any(axis=1)
            if not gap_in_stream.all():
                current_pid = 0
        if not enemy_actor_mask[t].any():
            current_pid = 0
            prev_fire_t = t
            continue
        # Sticky-keep test: is current_pid in stream AND in release cone?
        kept = False
        if current_pid > 0:
            pid_mask = enemy_actor_mask[t] & (player_ids[t] == current_pid)
            if pid_mask.any():
                slot = int(np.flatnonzero(pid_mask)[0])
                if cos_tr[t, slot] >= release_thr[t, slot]:
                    valid_shots.append((t, current_pid))
                    kept = True
            else:
                current_pid = 0
        if kept:
            prev_fire_t = t
            continue
        # Acquire: argmax cos over enemies passing per-enemy acquire cone.
        admit = enemy_actor_mask[t] & (cos_tr[t] >= acquire_thr[t])
        if not admit.any():
            current_pid = 0
            prev_fire_t = t
            continue
        cos_admitted = np.where(admit, cos_tr[t], -np.inf)
        best_slot = int(np.argmax(cos_admitted))
        pid = int(player_ids[t, best_slot])
        if pid > 0:
            valid_shots.append((t, pid))
            current_pid = pid
        prev_fire_t = t

    target = np.full(T, TARGET_IGNORE, dtype=np.int64)
    if not valid_shots:
        return target

    # ── Setup Slot Tracking Helper ───────────────────────────────────
    pid_slots_cache: dict[int, np.ndarray] = {}

    def get_slots(pid: int) -> np.ndarray:
        """Return (T,) array: slot index of pid per tick among enemy actor slots, -1 if absent."""
        if pid not in pid_slots_cache:
            has_pid = enemy_actor_mask & (player_ids == pid)
            any_pid = has_pid.any(axis=1)
            first_slot = has_pid.argmax(axis=1)
            pid_slots_cache[pid] = np.where(any_pid, first_slot, -1).astype(np.int64)
        return pid_slots_cache[pid]

    # ── Pass 2: Group shots by pid and token stream continuity ───────
    engagements: list[tuple[int, int, int]] = []
    current_start, current_end, current_pid = valid_shots[0][0], valid_shots[0][0], valid_shots[0][1]

    for t, pid in valid_shots[1:]:
        same_target = (pid == current_pid)

        # Verify the entity remained continuously in the token stream since the last shot.
        if same_target:
            slots = get_slots(pid)
            continuous_stream = bool((slots[current_end:t + 1] >= 0).all())
        else:
            continuous_stream = False

        if same_target and continuous_stream:
            # Target is identical and never dropped out of the engine state.
            current_end = t
        else:
            # Sequence broke: target switched OR target dropped out of token stream.
            engagements.append((current_start, current_end, current_pid))
            current_start, current_end, current_pid = t, t, pid

    engagements.append((current_start, current_end, current_pid))

    # ── Pass 3: Label the timeline using explicit boundaries ─────────
    for i, (start_t, end_t, pid) in enumerate(engagements):
        slots = get_slots(pid)
        in_stream = slots >= 0

        # Define hard boundaries to prevent timeline overlap between engagements.
        prev_eng_end = engagements[i - 1][1] if i > 0 else -1
        next_eng_start = engagements[i + 1][0] if i + 1 < len(engagements) else T

        # Trace backward (stops at token-stream loss or previous engagement).
        back_bound = start_t
        while back_bound > prev_eng_end + 1 and in_stream[back_bound - 1]:
            back_bound -= 1

        # Trace forward (stops at token-stream loss or next engagement).
        forward_bound = end_t
        while forward_bound < next_eng_start - 1 and in_stream[forward_bound + 1]:
            forward_bound += 1

        # Apply dynamic slot labels for the entire contiguous valid block.
        # Pass 2's continuity check guarantees in_stream is True throughout
        # this range, so slots[range] has no -1 entries.
        valid_slice = slice(back_bound, forward_bound + 1)
        target[valid_slice] = slots[valid_slice]

    return target
