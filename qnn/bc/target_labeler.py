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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from qnn.bc.weapon_physics import all_hits_at_fire
from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR

TARGET_IGNORE = -100

# Distribution-labeler output layout: index 0 is NO_TARGET, indices 1..N
# map to obs slots 0..N-1.
NO_TARGET_INDEX = 0
TARGET_DIST_CLASSES = MAX_TOKEN_OBJECTS + 1

# Adaptive cone parameters. Transverse offsets are in Quake units; obs rel
# vectors are scaled by 1/QNN_DIST_SCALE (=1/1000) so we rescale before
# applying. Acquire cone admits enemies within 208u perpendicular of aim,
# capped at 30° at close range and floored at 5° at extreme range. Release
# cone is twice that (K=2 Schmitt-trigger ratio), capped at 45°.
QNN_DIST_SCALE = 1000.0
QNN_VEL_SCALE  = 2000.0  # obs vel = world_vel / 2000 (entity moving 2000u/s → 1.0)
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
# vel is at offset 7 (after half_extents/rel/dist). 3 components.
_ACTOR_VEL_OFFSET = 7
# team scalar is at offset 16 (after half_extents/rel/dist/vel/path/path_dist/eta/facing).
# Value == 1.0 means "same team as demonstrator" — these actors must never be
# eligible target candidates.
_ACTOR_TEAM_OFFSET = 16
_TEAM_TEAMMATE_VALUE = 1.0


_MODALITY_SIGHT = 0  # qnn_vocab.h: QNN_MODALITY_SIGHT


# Weapon-aware lead-corrected cone (see src/docs/labeler_weapon_lead_spec.md).
# Per-weapon projectile speed in Quake u/s (np.inf = hitscan).
# Source: vendor/quake/QW/progs/weapons.qc.
_WEAPON_SPEED = {
    0: math.inf,   # no weapon — defensive default
    1: math.inf,   # Axe (melee, treated as hitscan range-gated to 64u)
    2: math.inf,   # Shotgun (hitscan, pellet spread)
    3: math.inf,   # Super Shotgun (hitscan, wider spread)
    4: 1000.0,     # Nailgun
    5: 1000.0,     # Super Nailgun
    6: 600.0,      # Grenade Launcher (straight-line approx)
    7: 1000.0,     # Rocket Launcher (no self-vel inheritance in vanilla QW)
    8: math.inf,   # Lightning Gun (hitscan, range-gated to 600u)
}
_WEAPON_MAX_RANGE = {
    0: 0.0,        # no weapon — admits nothing
    1: 64.0,       # Axe melee range
    2: math.inf,
    3: math.inf,
    4: math.inf,
    5: math.inf,
    6: math.inf,
    7: math.inf,
    8: 600.0,      # LG hardcoded engine max
}
# v2 (analytic decomposition) — see src/docs/labeler_weapon_lead_spec.md.
# cone_half = max(skill, bbox, spread, splash)  where
#   skill   = sqrt(lead² + (σ[w] · K(p))²)
#   bbox    = atan(target_half_extent / dist)
#   spread  = SPREAD_HALF[w]                              # fixed per-weapon
#   splash  = atan(SPLASH_RADIUS[w] / dist)               # 0 for non-explosive
# Skill knob: p ∈ [0.5, 0.99].  K(p) interpolated from empirical-quantile fit.
_WEAPON_SIGMA_DEG = {
    0: 0.0,        # no weapon
    1: 18.0,       # Axe
    2: 11.0,       # Shotgun
    3: 11.0,       # Super Shotgun
    4: 12.0,       # Nailgun
    5: 11.0,       # Super Nailgun
    6: 16.0,       # Grenade Launcher
    7: 14.0,       # Rocket Launcher
    8:  7.0,       # Lightning Gun
}
_WEAPON_SPREAD_HALF_DEG = {
    0: 0.0, 1: 0.0,
    2: 2.3,        # SG: 6 pellets, spread 0.04
    3: 8.0,        # SSG: spread 0.14 horiz / 0.08 vert
    4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0,
}
_WEAPON_SPLASH_RADIUS_U = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 120.0,      # GL: T_RadiusDamage 120
    7: 120.0,      # RL: T_RadiusDamage 120
    8: 0.0,
}
_ACQUIRE_HARD_CAP_DEG = 30.0   # match opt3's 30° acquire cap
_RELEASE_HARD_CAP_DEG = 45.0   # match opt3's 45° release cap

# K(p) — skill-knob to σ-multiplier curve.  Anchored to empirical
# aim-noise quantiles (val corpus, 2026-05-18).  Piecewise-linear between
# anchors, clamped outside.
_K_TABLE = (
    (0.50, 1.00),
    (0.75, 1.70),
    (0.90, 2.70),
    (0.95, 3.50),
    (0.99, 6.50),
)


def _k_of_p(p: float) -> float:
    """Map skill knob p ∈ [0, 1] to σ-multiplier K via interpolation."""
    if p <= _K_TABLE[0][0]:
        return _K_TABLE[0][1]
    if p >= _K_TABLE[-1][0]:
        return _K_TABLE[-1][1]
    for (p0, k0), (p1, k1) in zip(_K_TABLE[:-1], _K_TABLE[1:]):
        if p0 <= p <= p1:
            t = (p - p0) / (p1 - p0)
            return k0 + t * (k1 - k0)
    return _K_TABLE[-1][1]


# Actor half-extent layout (offset 0..2 in entity_scalars, scaled by 1/QNN_DIST_SCALE).
_ACTOR_HALFEXT_OFFSET = 0


def _weapon_aware_cone_thresholds(
    weapon_id: np.ndarray,         # (T,)
    dist_qu: np.ndarray,           # (T, N) Quake units
    vel_qu: np.ndarray,            # (T, N, 3) world u/s
    rel_unit: np.ndarray,          # (T, N, 3) unit
    half_extent_qu: np.ndarray,    # (T, N, 3) Quake units
    p_accept: float,
    p_release: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame, per-slot (acquire_cos, release_cos) using the v2 analytic
    decomposition: cone_half = max(skill, bbox, spread, splash).

    Skill = sqrt(lead² + (σ[w] · K(p))²) where
      lead   = atan(V_perp / projectile_speed)
      σ[w]   = per-weapon empirical aim-noise scale
      K(p)   = skill-quantile curve (acquire uses p_accept, release p_release)
    """
    T, N = dist_qu.shape
    rad = np.radians

    # Per-frame weapon constants — broadcast to (T, 1).
    speed   = np.asarray([_WEAPON_SPEED.get(int(w), math.inf) for w in weapon_id],
                         dtype=np.float32)[:, None]
    sigma   = np.asarray([_WEAPON_SIGMA_DEG.get(int(w), 0.0) for w in weapon_id],
                         dtype=np.float32)[:, None]
    spread  = np.asarray([_WEAPON_SPREAD_HALF_DEG.get(int(w), 0.0) for w in weapon_id],
                         dtype=np.float32)[:, None]
    splash  = np.asarray([_WEAPON_SPLASH_RADIUS_U.get(int(w), 0.0) for w in weapon_id],
                         dtype=np.float32)[:, None]
    is_hits = np.isinf(speed)

    # Lead error — per-target V_perp (component of vel ⟂ to LoS).
    v_par_mag = np.einsum("tij,tij->ti", vel_qu, rel_unit)         # (T, N)
    v_par = v_par_mag[..., None] * rel_unit                         # (T, N, 3)
    v_perp = np.linalg.norm(vel_qu - v_par, axis=-1)                # (T, N) u/s
    safe_speed = np.where(is_hits, 1.0, speed)                      # avoid /inf
    lead_rad = np.where(is_hits,
                        0.0,
                        np.arctan(v_perp / safe_speed))             # (T, N)

    # Skill components for acquire / release.
    k_acc = _k_of_p(p_accept)
    k_rel = _k_of_p(p_release)
    aim_acc = rad(sigma * k_acc)                                    # (T, 1)
    aim_rel = rad(sigma * k_rel)                                    # (T, 1)
    skill_acc = np.sqrt(lead_rad**2 + aim_acc**2)                   # (T, N)
    skill_rel = np.sqrt(lead_rad**2 + aim_rel**2)                   # (T, N)

    # Geometric tolerances — same for acquire and release.
    safe_dist = np.maximum(dist_qu, 1e-3)
    # bbox half-extent: take max of x/y/z (most generous direction).
    bbox_half_qu = np.max(half_extent_qu, axis=-1)                  # (T, N) Quake u
    bbox_rad = np.arctan(bbox_half_qu / safe_dist)                  # (T, N)
    splash_rad = np.where(splash > 0, np.arctan(splash / safe_dist), 0.0)  # (T, N)
    spread_rad = rad(spread)                                        # (T, 1)

    # Cone half-angle: use opt3's per-distance adaptive cone as the
    # primary mechanism (proven low-jitter geometry), with the v2
    # geometric tolerances (bbox / spread / splash) as a *lower* bound to
    # ensure close-range and explosive engagements get the right minimum
    # tolerance.  The σ-aim-noise term is dropped — the empirical-skill
    # analysis showed it was over-widening cones at distance and creating
    # co-angular jitter.  The lead-corrected aim center (computed in the
    # caller) is the only piece v2 retains from the σ-K(p) approach.
    spread_full = np.broadcast_to(spread_rad, (T, N))
    acq_floor_rad = rad(5.0)
    rel_floor_rad = rad(5.0)
    acq_cap_rad   = rad(_ACQUIRE_HARD_CAP_DEG)
    rel_cap_rad   = rad(_RELEASE_HARD_CAP_DEG)
    acq_adapt = np.clip(np.arctan(208.0 / safe_dist), acq_floor_rad, acq_cap_rad)
    rel_adapt = np.clip(np.arctan(416.0 / safe_dist), rel_floor_rad, rel_cap_rad)
    # Adaptive cone as the base; bbox/spread/splash widen it only when
    # they would naturally exceed the adaptive value (close-range bbox,
    # mid-range splash).
    geom_floor = np.maximum(bbox_rad, np.maximum(spread_full, splash_rad))
    acq_half = np.maximum(acq_adapt, geom_floor)
    rel_half = np.maximum(rel_adapt, geom_floor)
    # Reference the unused σ × K(p) skill quantities so they remain a
    # documented part of the API; not used in the final cone but the
    # analysis script can read them.
    _ = (skill_acc, skill_rel, k_acc, k_rel, aim_acc, aim_rel)
    return np.cos(acq_half), np.cos(rel_half)


def label_enemy_target(
    obs: Dict[str, np.ndarray],
    actions: Dict[str, np.ndarray],
    sight_only: bool = False,
    weapon_aware: bool = False,
    p_accept: float = 0.9,
    p_release: float = 0.95,
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

    ``weapon_aware`` replaces the weapon-agnostic adaptive cone with a
    physics-derived lead-corrected cone keyed on the demonstrator's
    weapon (actions["weapon"], 0..8).  The cone center becomes the
    lead-adjusted aim point ``rel + target_velocity * (dist /
    projectile_speed)``, and the cone half-width becomes
    ``atan(V_perp_max / projectile_speed)`` for projectile weapons
    (distance-invariant) or a floor for hitscan weapons.  Per-weapon
    max-range gates kill candidates beyond the weapon's effective reach
    (axe 64u, LG 600u, others unbounded).  See
    ``src/docs/labeler_weapon_lead_spec.md``.
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

    look_norm = np.linalg.norm(look, axis=-1, keepdims=True)
    unit_look = look / np.maximum(look_norm, 1e-6)

    if weapon_aware:
        # Lead-corrected aim: aim point = rel + target_velocity * T_flight.
        # For hitscan (projectile_speed = inf), T_flight = 0 → aim = rel.
        weapon = np.asarray(actions.get("weapon",
                                         np.full(T, 7, dtype=np.uint8))).reshape(-1)
        vel = entity_scalars[:, :, _ACTOR_VEL_OFFSET:_ACTOR_VEL_OFFSET + 3]
        half_ext = entity_scalars[:, :,
            _ACTOR_HALFEXT_OFFSET:_ACTOR_HALFEXT_OFFSET + 3]
        rel_qu = rel * QNN_DIST_SCALE                              # (T, 16, 3) Quake u
        vel_qu = vel * QNN_VEL_SCALE                               # (T, 16, 3) Quake u/s
        half_ext_qu = half_ext * QNN_DIST_SCALE                    # (T, 16, 3) Quake u
        rel_unit = rel_qu / np.maximum(np.linalg.norm(rel_qu, axis=-1, keepdims=True), 1e-3)

        # Lead-corrected aim direction per slot.
        speed = np.asarray([_WEAPON_SPEED.get(int(w), math.inf) for w in weapon],
                           dtype=np.float32)                       # (T,)
        is_hitscan = np.isinf(speed)
        safe_speed = np.where(is_hitscan, 1.0, speed)
        t_flight = np.where(is_hitscan[:, None], 0.0,
                            dist_qu / safe_speed[:, None])         # (T, 16)
        lead_qu = vel_qu * t_flight[..., None]                     # (T, 16, 3)
        aim_qu = rel_qu + lead_qu
        aim_norm = np.linalg.norm(aim_qu, axis=-1)
        unit_aim = aim_qu / np.maximum(aim_norm[..., None], 1e-6)
        # The cone test admits if the demonstrator's look is consistent
        # with engaging this pid either hitscan-style (aim at current
        # position) or projectile-style (aim at lead point).  Without
        # this max, sticky-keep fails on moving targets when the
        # demonstrator under-leads — even though they're clearly still
        # engaging the same enemy.  See switch-jitter analysis at
        # scripts/analysis/v2_jitter_decompose.py.
        cos_lead    = np.einsum("tij,tj->ti", unit_aim, unit_look)
        cos_current = np.einsum("tij,tj->ti", rel_unit, unit_look)
        cos_tr = np.maximum(cos_lead, cos_current)

        # Per-weapon max-range gate.
        max_range = np.asarray([_WEAPON_MAX_RANGE.get(int(w), math.inf)
                                for w in weapon], dtype=np.float32)
        in_range = dist_qu <= max_range[:, None]
        cos_tr = np.where(in_range, cos_tr, -np.inf)

        # Per-frame, per-slot cone thresholds (v2 analytic decomposition).
        acquire_thr, release_thr = _weapon_aware_cone_thresholds(
            weapon_id=weapon,
            dist_qu=dist_qu,
            vel_qu=vel_qu,
            rel_unit=rel_unit,
            half_extent_qu=half_ext_qu,
            p_accept=p_accept,
            p_release=p_release,
        )
    else:
        unit_rel = rel / np.maximum(rel_norm[..., None], 1e-6)
        cos_tr = np.einsum("tij,tj->ti", unit_rel, unit_look)
        # Per-frame, per-enemy cone thresholds (adaptive to distance).
        acquire_thr = _adaptive_cone_cos(dist_qu, ACQUIRE_TRANSVERSE_U,
                                         _ACQUIRE_CAP_COS, _ACQUIRE_FLOOR_COS)
        release_thr = _adaptive_cone_cos(dist_qu, RELEASE_TRANSVERSE_U,
                                         _RELEASE_CAP_COS, _RELEASE_FLOOR_COS)

    cos_actor = np.where(enemy_actor_mask, cos_tr, -np.inf)

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


# ─── V3: confidence-distributed target labels ────────────────────────────
#
# See ``src/docs/labeler_v3_simple.md`` for the design. Per-fire, every
# enemy pid that has cone evidence OR physics-hit evidence becomes a
# candidate; mass is distributed proportionally, never argmaxed. Per-pid
# streams group fire anchors through continuous in-stream presence (Pass
# 2) and extend backward/forward (Pass 3). Multiple pid streams may
# overlap — the soft distribution captures the ambiguity rather than
# forcing a one-hot switch.
#
# Constants below are PRE-CALIBRATION defaults. Production collections
# should freeze a calibrated config (see the calibration recipe in the
# design doc) and record it in the manifest.

_ACTOR_RECENCY_OFFSET = 18   # matches qnn.bc.weapon_physics.ACTOR_RECENCY_OFFSET
_SELF_HEALTH_OFFSET = 0      # obs_self_scalars[:, 0] normalized health


@dataclass(frozen=True)
class LabelerConfig:
    """Calibration knobs for ``label_enemy_target_distribution``.

    Defaults are the production-fit values from a 6-shard QWD val audit
    (see ``scripts/analysis/labeler_v3_eng_conf_audit.py`` and
    ``scripts/analysis/labeler_v3_aggregation_audit.py``).  Re-fit per
    collection if labelers, weapons, or demo sources change.
    """
    # Candidate admission
    cone_admit:           float = 0.25   # min cone score for cone-only admission
    physics_hit_base:     float = 0.95   # base evidence weight for a physics hit
    theta_reject_deg:     float = 45.0   # hard reject if cone-only and angle > this

    # Visibility / recency decay (recency is in seconds; SIGHT max = 2.0)
    recency_tau:          float = 1.00

    # Anchor mass cap
    present_cap:          float = 0.98

    # Engagement confidence (logistic regression on per-stream features)
    # σ(intercept + w·x), x = (mean_anchor, fire_count_conf, max_anchor,
    #                         log1p(duration), log1p(n_fires)).
    # Coefficients fitted on QWD val shards 0..5 (7,345 physics-confirmed
    # streams) against the "v2-and-physics consensus support" target.
    fire_count_tau:       float = 2.0
    eng_logistic_intercept: float = -4.9824
    eng_logistic_w_mean_anchor:     float =  5.3106
    eng_logistic_w_fire_count_conf: float =  4.5381
    eng_logistic_w_max_anchor:      float =  1.0199
    eng_logistic_w_log_duration:    float = -0.7043
    eng_logistic_w_log_n_fires:     float =  0.6047

    # Death penalty
    dead_health_threshold: float = 0.05
    death_window:          int   = 10    # frames; default matches POST_K
    death_penalty:         float = 0.65

    # Frame extension (time_conf = floor + (1-floor) * exp(-dt / tau))
    # Retuned on the dt-bucket confirmation rates from the boundary audit:
    # current floor=0.50/tau=35 systematically under-predicts the dt>=20 tail.
    time_floor:            float = 0.65
    extension_tau:         float = 40.0  # frames


DEFAULT_LABELER_CONFIG = LabelerConfig()


def _pid_slots_per_frame(enemy_mask: np.ndarray, player_ids: np.ndarray,
                         pid: int) -> np.ndarray:
    """Per-frame slot of ``pid`` among enemy-actor slots, or -1 if absent."""
    has = enemy_mask & (player_ids == pid)
    any_pid = has.any(axis=1)
    first_slot = has.argmax(axis=1)
    return np.where(any_pid, first_slot, -1).astype(np.int64)


def _bad_end_window(dead: np.ndarray, death_edge: np.ndarray,
                    back: int, fwd: int, death_window: int, T: int) -> bool:
    """True if the demonstrator died inside the window or within
    ``death_window`` frames after it."""
    if dead[back:fwd + 1].any():
        return True
    post_lo = fwd + 1
    post_hi = min(T, fwd + 1 + death_window)
    if post_hi <= post_lo:
        return False
    return bool(death_edge[post_lo:post_hi].any())


def label_enemy_target_distribution(
    obs: Dict[str, np.ndarray],
    actions: Dict[str, np.ndarray],
    *,
    config: LabelerConfig = DEFAULT_LABELER_CONFIG,
    sight_only: bool = False,
) -> np.ndarray:
    """Return a ``(T, TARGET_DIST_CLASSES)`` float32 confidence distribution
    over ``(NO_TARGET, slot_0, ..., slot_{N-1})``.

    Row sums are 1.0. ``p(NO_TARGET) = 1 - sum(p_slots)``. Frames with no
    candidate evidence collapse to ``p(NO_TARGET) = 1.0``.

    Algorithm (see ``src/docs/labeler_v3_simple.md``):
      1. For each fire, admit every enemy pid that has cone evidence
         (lead-corrected angle below opt3 adaptive width) OR physics-hit
         evidence (recency-0 ray/projectile test). Distribute anchor mass
         proportionally; never argmax.
      2. Group anchors per-pid into streams while the pid stays in token
         stream. Multiple pid streams may overlap.
      3. Extend each stream backward/forward through continuous presence.
         Accumulate per-slot scores with engagement-confidence × time-decay ×
         visibility. Normalize to 17 classes.
    """
    entity_types = np.asarray(obs["entity_types"])
    entity_ids = np.asarray(obs["entity_ids"])
    entity_scalars = np.asarray(obs["entity_scalars_raw"])
    self_scalars = np.asarray(obs["self_scalars"])
    look = np.asarray(actions["look"], dtype=np.float32)
    fire = np.asarray(actions["fire"]).reshape(-1)

    T = look.shape[0]
    N = MAX_TOKEN_OBJECTS
    out = np.zeros((T, TARGET_DIST_CLASSES), dtype=np.float32)
    if T == 0:
        return out

    # Default to "no target" everywhere; valid evidence overwrites below.
    out[:, NO_TARGET_INDEX] = 1.0

    actor_mask = entity_types == TOKEN_ACTOR
    teammate_mask = entity_scalars[:, :, _ACTOR_TEAM_OFFSET] == _TEAM_TEAMMATE_VALUE
    enemy = actor_mask & ~teammate_mask
    if sight_only:
        modality = entity_ids[:, :, 1]
        enemy &= (modality == _MODALITY_SIGHT)
    player_ids = entity_ids[:, :, 2]
    recency = entity_scalars[:, :, _ACTOR_RECENCY_OFFSET]

    # Lead-corrected aim geometry (same as v2 weapon_aware path).
    rel = entity_scalars[:, :, _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET + 3]
    rel_qu = rel * QNN_DIST_SCALE
    rel_norm_qu = np.linalg.norm(rel_qu, axis=-1)
    dist_qu = rel_norm_qu

    vel = entity_scalars[:, :, _ACTOR_VEL_OFFSET:_ACTOR_VEL_OFFSET + 3]
    vel_qu = vel * QNN_VEL_SCALE

    look_norm = np.linalg.norm(look, axis=-1, keepdims=True)
    unit_look = look / np.maximum(look_norm, 1e-6)

    weapon = np.asarray(actions.get("weapon",
                                    np.full(T, 7, dtype=np.uint8))).reshape(-1)
    speed = np.asarray([_WEAPON_SPEED.get(int(w), math.inf) for w in weapon],
                       dtype=np.float32)
    is_hitscan = np.isinf(speed)
    safe_speed = np.where(is_hitscan, 1.0, speed)
    t_flight = np.where(is_hitscan[:, None], 0.0,
                        dist_qu / safe_speed[:, None])
    aim_qu = rel_qu + vel_qu * t_flight[..., None]
    aim_norm_qu = np.linalg.norm(aim_qu, axis=-1, keepdims=True)
    unit_aim = aim_qu / np.maximum(aim_norm_qu, 1e-6)
    unit_rel = rel_qu / np.maximum(rel_norm_qu[..., None], 1e-6)

    cos_lead = np.einsum("tij,tj->ti", unit_aim, unit_look)
    cos_cur  = np.einsum("tij,tj->ti", unit_rel, unit_look)
    cos_tr   = np.maximum(cos_lead, cos_cur)
    theta    = np.arccos(np.clip(cos_tr, -1.0, 1.0))

    safe_dist = np.maximum(dist_qu, 1e-3)
    theta_acq = np.clip(np.arctan(208.0 / safe_dist),
                        math.radians(5.0), math.radians(30.0))
    cone = np.exp(-0.5 * (theta / theta_acq) ** 2).astype(np.float32)

    theta_reject_rad = math.radians(config.theta_reject_deg)

    # ── Pass 1: per-fire anchor evidence ─────────────────────────────
    anchors_by_pid: dict[int, list[tuple[int, float]]] = defaultdict(list)
    fire_ticks = np.flatnonzero(fire == 1)
    for tt in fire_ticks:
        t = int(tt)
        w = int(weapon[t])
        physics_pids = all_hits_at_fire(w, look[t], entity_scalars,
                                        entity_ids, t, enemy)
        cand: list[tuple[int, int]] = []
        evidence: list[float] = []
        for s in range(N):
            if not enemy[t, s]:
                continue
            pid = int(player_ids[t, s])
            if pid <= 0:
                continue
            cone_val = float(cone[t, s])
            cone_ok = cone_val >= config.cone_admit
            hit = pid in physics_pids and recency[t, s] == 0.0
            if not (cone_ok or hit):
                continue
            if theta[t, s] > theta_reject_rad and not hit:
                continue
            # Noisy-OR combination of cone and physics evidence.  Captures
            # the agreement boost that max() throws away (see
            # scripts/analysis/labeler_v3_aggregation_audit.py).
            cone_e = cone_val if cone_ok else 0.0
            phys_e = config.physics_hit_base if hit else 0.0
            base = 1.0 - (1.0 - cone_e) * (1.0 - phys_e)
            vis = math.exp(-float(recency[t, s]) / config.recency_tau)
            cand.append((pid, s))
            evidence.append(base * vis)
        if not evidence:
            continue
        total = float(sum(evidence))
        if total <= 0.0:
            continue
        present = min(total, config.present_cap)
        for (pid, _), e in zip(cand, evidence):
            anchors_by_pid[pid].append((t, present * e / total))

    if not anchors_by_pid:
        return out

    # ── Pass 2: per-pid stream grouping ──────────────────────────────
    # Streams: (pid, fire_times, window_start, window_end, eng_conf)
    streams: list[tuple[int, list[int], int, int, float]] = []
    pid_slots_cache: dict[int, np.ndarray] = {}

    def slots_of(pid: int) -> np.ndarray:
        cached = pid_slots_cache.get(pid)
        if cached is None:
            cached = _pid_slots_per_frame(enemy, player_ids, pid)
            pid_slots_cache[pid] = cached
        return cached

    dead = (self_scalars[:, _SELF_HEALTH_OFFSET]
            < config.dead_health_threshold)
    death_edge = np.zeros(T, dtype=bool)
    death_edge[1:] = dead[1:] & ~dead[:-1]

    for pid, anchors in anchors_by_pid.items():
        slots = slots_of(pid)
        in_stream = slots >= 0
        # Split into groups by stream-presence continuity between anchors.
        groups: list[list[tuple[int, float]]] = []
        cur: list[tuple[int, float]] = [anchors[0]]
        for prev_a, this_a in zip(anchors, anchors[1:]):
            t_prev, _ = prev_a
            t_this, _ = this_a
            if bool(in_stream[t_prev:t_this + 1].all()):
                cur.append(this_a)
            else:
                groups.append(cur)
                cur = [this_a]
        groups.append(cur)

        for group in groups:
            n_fire = len(group)
            anchor_vals = np.array([a for _, a in group], dtype=np.float64)
            mean_anchor = float(anchor_vals.mean())
            max_anchor = float(anchor_vals.max())
            fire_conf = 1.0 - math.exp(-n_fire / config.fire_count_tau)
            # Extension window through continuous in-stream presence.
            start_t = group[0][0]
            end_t = group[-1][0]
            back = start_t
            while back > 0 and in_stream[back - 1]:
                back -= 1
            fwd = end_t
            while fwd < T - 1 and in_stream[fwd + 1]:
                fwd += 1
            duration = fwd - back + 1
            bad_end = _bad_end_window(dead, death_edge, back, fwd,
                                      config.death_window, T)
            death_pen = config.death_penalty if bad_end else 1.0
            # Logistic regression on per-stream features.  Coefficients fitted
            # on 6 shards of QWD val data against v2-and-physics consensus
            # support (see scripts/analysis/labeler_v3_eng_conf_audit.py).
            logit = (
                config.eng_logistic_intercept
                + config.eng_logistic_w_mean_anchor     * mean_anchor
                + config.eng_logistic_w_fire_count_conf * fire_conf
                + config.eng_logistic_w_max_anchor      * max_anchor
                + config.eng_logistic_w_log_duration    * math.log1p(duration)
                + config.eng_logistic_w_log_n_fires     * math.log1p(n_fire)
            )
            eng_conf = float(death_pen / (1.0 + math.exp(-logit)))
            streams.append((pid, [t for t, _ in group], back, fwd, eng_conf))

    # ── Pass 3: accumulate per-slot scores ───────────────────────────
    slot_scores = np.zeros((T, N), dtype=np.float32)
    for pid, fire_ts, back, fwd, eng_conf in streams:
        slots = slots_of(pid)
        for t in range(back, fwd + 1):
            s = int(slots[t])
            if s < 0:
                continue
            dt = min(abs(t - ft) for ft in fire_ts)
            time_conf = (config.time_floor
                         + (1.0 - config.time_floor)
                         * math.exp(-dt / config.extension_tau))
            vis = math.exp(-float(recency[t, s]) / config.recency_tau)
            slot_scores[t, s] += eng_conf * time_conf * vis

    # ── Normalize per frame ──────────────────────────────────────────
    S = slot_scores.sum(axis=1)
    active = S > 0.0
    if active.any():
        present = np.minimum(S, config.present_cap)
        # Where active: NO_TARGET = 1 - present; slots distributed.
        scale = np.zeros(T, dtype=np.float32)
        scale[active] = present[active] / S[active]
        out[active, NO_TARGET_INDEX] = (1.0 - present[active]).astype(np.float32)
        out[:, 1:] = (slot_scores * scale[:, None]).astype(np.float32)
    return out
