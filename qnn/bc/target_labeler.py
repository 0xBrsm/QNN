"""Combat target label generator for BC.

The live labeler is ``label_enemy_target_probs`` (v3), which emits a
per-tick ``(MAX_TOKEN_OBJECTS + 1,)`` float32 distribution over
``(NO_TARGET, idx_0, ..., idx_{N-1})``. See ``src/docs/labeler_v3_simple.md``
for the design.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np

from qnn.bc.weapon_physics import all_hits_at_fire
from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR

TARGET_IGNORE = -100

# Distribution-labeler output layout: index 0 is NO_TARGET, indices 1..N
# map to obs indices 0..N-1.
NO_TARGET_INDEX = 0
TARGET_PROBS_CLASSES = MAX_TOKEN_OBJECTS + 1

# Adaptive cone parameters (used by v3 candidate admission and by analysis
# scripts under scripts/analysis/). Transverse offsets are in Quake units;
# obs rel vectors are scaled by 1/QNN_DIST_SCALE (=1/1000) so we rescale
# before applying. Acquire cone admits enemies within 208u perpendicular
# of aim, capped at 30° at close range and floored at 5° at extreme range.
# Release cone (twice as wide, capped at 45°) is used by hybrid/analysis
# labelers, not by the live v3 path.
QNN_DIST_SCALE = 1000.0
QNN_VEL_SCALE  = 2000.0  # obs vel = world_vel / 2000 (entity moving 2000u/s → 1.0)
ACQUIRE_TRANSVERSE_U = 208.0
RELEASE_TRANSVERSE_U = 416.0
_ACQUIRE_CAP_COS = math.cos(math.radians(30.0))
_ACQUIRE_FLOOR_COS = math.cos(math.radians(5.0))
_RELEASE_CAP_COS = math.cos(math.radians(45.0))
_RELEASE_FLOOR_COS = math.cos(math.radians(5.0))


def _extract_attack(actions) -> np.ndarray:
    """Return the per-frame attack stream from an actions dict.

    The on-disk action layout packs attack into bit 0 of the ``move``
    byte (mirroring the input_mask layout); there is no standalone
    ``attack`` field after the engine collector. For BC-time call sites
    that still pre-unpack ``attack`` as a (T,) uint8 (e.g. the streaming
    source slice path), use that directly. Otherwise derive from
    ``move`` bit 0.
    """
    if "attack" in actions:
        return np.asarray(actions["attack"])
    move = np.asarray(actions["move"], dtype=np.uint8)
    return (move & 1).astype(np.uint8)


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


# Per-weapon projectile speed in Quake u/s (np.inf = hitscan). Used by the
# v3 lead-corrected aim calculation. Source: vendor/quake/QW/progs/weapons.qc.
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
# Actor half-extent layout (offset 0..2 in entity_scalars, scaled by 1/QNN_DIST_SCALE).
_ACTOR_HALFEXT_OFFSET = 0


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
    """Calibration knobs for ``label_enemy_target_probs``.

    Defaults are the production-fit values from a 6-shard QWD val audit
    (see ``scripts/analysis/labeler_v3_eng_conf_audit.py``).  Re-fit per
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


def _pid_indices_per_frame(enemy_mask: np.ndarray, player_ids: np.ndarray,
                         pid: int) -> np.ndarray:
    """Per-frame idx of ``pid`` among enemy-actor indices, or -1 if absent."""
    has = enemy_mask & (player_ids == pid)
    any_pid = has.any(axis=1)
    first_idx = has.argmax(axis=1)
    return np.where(any_pid, first_idx, -1).astype(np.int64)


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


def label_enemy_target_probs(
    obs: Dict[str, np.ndarray],
    actions: Dict[str, np.ndarray],
    *,
    config: LabelerConfig = DEFAULT_LABELER_CONFIG,
    sight_only: bool = False,
) -> np.ndarray:
    """Return a ``(T, TARGET_PROBS_CLASSES)`` float32 confidence distribution
    over ``(NO_TARGET, idx_0, ..., idx_{N-1})``.

    Row sums are 1.0. ``p(NO_TARGET) = 1 - sum(p_indices)``. Frames with no
    candidate evidence collapse to ``p(NO_TARGET) = 1.0``.

    Algorithm (see ``src/docs/labeler_v3_simple.md``):
      1. For each fire, admit every enemy pid that has cone evidence
         (lead-corrected angle below opt3 adaptive width) OR physics-hit
         evidence (recency-0 ray/projectile test). Distribute anchor mass
         proportionally; never argmax.
      2. Group anchors per-pid into streams while the pid stays in token
         stream. Multiple pid streams may overlap.
      3. Extend each stream backward/forward through continuous presence.
         Accumulate per-idx scores with engagement-confidence × time-decay ×
         visibility. Normalize to 17 classes.
    """
    entity_types = np.asarray(obs["entity_types"])
    entity_ids = np.asarray(obs["entity_ids"])
    entity_scalars = np.asarray(obs["entity_scalars_raw"])
    self_scalars = np.asarray(obs["self_scalars"])
    look = np.asarray(actions["look"], dtype=np.float32)
    fire = np.asarray(_extract_attack(actions)).reshape(-1)

    T = look.shape[0]
    N = MAX_TOKEN_OBJECTS
    out = np.zeros((T, TARGET_PROBS_CLASSES), dtype=np.float32)
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

    # Lead-corrected aim geometry.
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
            # the agreement boost that max() throws away.
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
    pid_indices_cache: dict[int, np.ndarray] = {}

    def indices_of(pid: int) -> np.ndarray:
        cached = pid_indices_cache.get(pid)
        if cached is None:
            cached = _pid_indices_per_frame(enemy, player_ids, pid)
            pid_indices_cache[pid] = cached
        return cached

    dead = (self_scalars[:, _SELF_HEALTH_OFFSET]
            < config.dead_health_threshold)
    death_edge = np.zeros(T, dtype=bool)
    death_edge[1:] = dead[1:] & ~dead[:-1]

    for pid, anchors in anchors_by_pid.items():
        indices = indices_of(pid)
        in_stream = indices >= 0
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

    # ── Pass 3: accumulate per-idx scores ───────────────────────────
    idx_scores = np.zeros((T, N), dtype=np.float32)
    for pid, fire_ts, back, fwd, eng_conf in streams:
        indices = indices_of(pid)
        for t in range(back, fwd + 1):
            s = int(indices[t])
            if s < 0:
                continue
            dt = min(abs(t - ft) for ft in fire_ts)
            time_conf = (config.time_floor
                         + (1.0 - config.time_floor)
                         * math.exp(-dt / config.extension_tau))
            vis = math.exp(-float(recency[t, s]) / config.recency_tau)
            idx_scores[t, s] += eng_conf * time_conf * vis

    # ── Normalize per frame ──────────────────────────────────────────
    S = idx_scores.sum(axis=1)
    active = S > 0.0
    if active.any():
        present = np.minimum(S, config.present_cap)
        # Where active: NO_TARGET = 1 - present; indices distributed.
        scale = np.zeros(T, dtype=np.float32)
        scale[active] = present[active] / S[active]
        out[active, NO_TARGET_INDEX] = (1.0 - present[active]).astype(np.float32)
        out[:, 1:] = (idx_scores * scale[:, None]).astype(np.float32)
    return out
