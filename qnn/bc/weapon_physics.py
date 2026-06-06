"""Shared weapon-physics primitives for BC labelers.

The cone-anchored labeler (``target_labeler.py``) uses ``all_hits_at_fire``
to admit physics-hit pids as co-primary candidates. The analysis scripts
(``scripts/analysis/hit_labeler.py``, ``hit_streams.py``) reuse the same
ray-AABB / projectile-sim primitives. Keeping the implementation in one
place avoids the drift that would otherwise creep in between the labeler
and its diagnostic scripts.

All distance/velocity scales follow the obs-buffer contract in
``qnn_io.c``: rel/half-extent are scaled by 1/QNN_DIST_SCALE (=1/1000)
and velocity by 1/QNN_VEL_SCALE (=1/2000) before being packed into
``entity_scalars_raw``. Callers pass the raw scalar array; we unscale
internally.

Recency is in *seconds* (float32, max 2.0 for SIGHT modality per
``qnn_vocab.h``). The physics path requires recency == 0 — anything else
is treated as occluded.
"""
from __future__ import annotations

import math

import numpy as np

# Obs-buffer scaling (matches qnn_io.c packing).
QNN_DIST_SCALE = 1000.0
QNN_VEL_SCALE  = 2000.0

# Actor scalar offsets (matches _ACTOR_LAYOUT in sim.py / qnn_actor_token_t).
ACTOR_HALFEXT_OFFSET = 0   # length 3
ACTOR_REL_OFFSET     = 3   # length 3
ACTOR_VEL_OFFSET     = 7   # length 3
ACTOR_TEAM_OFFSET    = 16
ACTOR_RECENCY_OFFSET = 18

TEAM_TEAMMATE_VALUE = 1.0

# Per-weapon physics. Source: vendor/quake/QW/progs/weapons.qc.
#   hitscan=True with `range` u and `spread_deg` half-angle.
#   hitscan=False with `speed` u/s, optional `splash` u, `gravity` u/s²,
#     `max_t` flight cap in seconds.
WEAPON_PHYSICS: dict[int, dict] = {
    1: {"hitscan": True,  "range": 64.0,    "spread_deg": 0.0,  "splash": 0.0},   # Axe
    2: {"hitscan": True,  "range": 1500.0,  "spread_deg": 2.3,  "splash": 0.0},   # SG
    3: {"hitscan": True,  "range": 1500.0,  "spread_deg": 8.0,  "splash": 0.0},   # SSG
    4: {"hitscan": False, "speed": 1000.0,  "splash": 0.0, "max_t": 3.0},         # NG
    5: {"hitscan": False, "speed": 1000.0,  "splash": 0.0, "max_t": 3.0},         # SNG
    6: {"hitscan": False, "speed": 600.0,   "splash": 120.0, "max_t": 2.5,        # GL
        "gravity": 800.0},
    7: {"hitscan": False, "speed": 1000.0,  "splash": 120.0, "max_t": 4.0},       # RL
    8: {"hitscan": True,  "range": 600.0,   "spread_deg": 0.0,  "splash": 0.0},   # LG
}


def _ray_aabb_distance(origin: np.ndarray, direction_unit: np.ndarray,
                       box_center: np.ndarray, box_half: np.ndarray) -> float:
    """Slab method ray-AABB intersection. Returns the entry-point distance
    along the ray (>=0) or +inf if no hit. Origin is the player; box is
    the target."""
    p = -box_center  # ray origin relative to box (assumes origin == zeros)
    d = direction_unit
    eps = 1e-6
    tmin, tmax = -math.inf, math.inf
    for i in range(3):
        if abs(d[i]) < eps:
            if p[i] < -box_half[i] or p[i] > box_half[i]:
                return math.inf
        else:
            t1 = (-box_half[i] - p[i]) / d[i]
            t2 = ( box_half[i] - p[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return math.inf
    if tmax < 0:
        return math.inf
    return max(tmin, 0.0)


def _hitscan_test(look_u: np.ndarray, rel_qu: np.ndarray, half_qu: np.ndarray,
                  max_range: float, spread_rad: float) -> tuple[bool, float]:
    """Returns (hit, dist). dist is ray-bbox entry distance (smaller=closer)."""
    dist = _ray_aabb_distance(np.zeros(3), look_u, rel_qu, half_qu)
    if dist <= max_range:
        return True, dist
    if spread_rad > 0:
        rel_norm = float(np.linalg.norm(rel_qu))
        if rel_norm < 1e-3:
            return False, math.inf
        unit_rel = rel_qu / rel_norm
        cos_off = float(np.clip(np.dot(unit_rel, look_u), -1.0, 1.0))
        ang_off = math.acos(cos_off)
        if ang_off <= spread_rad and rel_norm <= max_range:
            return True, rel_norm
    return False, math.inf


def _projectile_test(look_u: np.ndarray, rel_qu: np.ndarray, vel_qu: np.ndarray,
                     half_qu: np.ndarray, speed: float, splash: float,
                     max_t: float, gravity: float = 0.0,
                     dt: float = 0.05) -> tuple[bool, float]:
    """Returns (hit, t_hit) in seconds. Simulates constant-velocity target."""
    half_ext = np.maximum(half_qu, 4.0).astype(np.float32)
    g_vec = np.array([0.0, 0.0, -gravity], dtype=np.float32)
    n_steps = int(max_t / dt)
    for step in range(n_steps):
        t = step * dt
        proj = look_u * speed * t + 0.5 * g_vec * (t ** 2)
        targ = rel_qu + vel_qu * t
        diff = proj - targ
        if splash > 0:
            gap = np.maximum(np.abs(diff) - half_ext, 0)
            if float(np.linalg.norm(gap)) <= splash:
                return True, float(t)
        if np.all(np.abs(diff) <= half_ext):
            return True, float(t)
    return False, math.inf


def _decode_slot(esc: np.ndarray, t: int, slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (rel_qu, vel_qu, half_qu) for one slot at one frame from the raw
    scalar array. Returns Quake-unit-scaled vectors."""
    rel  = esc[t, slot, ACTOR_REL_OFFSET:ACTOR_REL_OFFSET + 3].astype(np.float32) * QNN_DIST_SCALE
    vel  = esc[t, slot, ACTOR_VEL_OFFSET:ACTOR_VEL_OFFSET + 3].astype(np.float32) * QNN_VEL_SCALE
    half = esc[t, slot, ACTOR_HALFEXT_OFFSET:ACTOR_HALFEXT_OFFSET + 3].astype(np.float32) * QNN_DIST_SCALE
    return rel, vel, half


def _physics_hit_for_slot(weapon: int, look_u: np.ndarray, esc: np.ndarray,
                          t: int, slot: int) -> tuple[bool, float]:
    """Run the appropriate hit test for ``weapon`` against actor ``slot``.
    Returns (hit, metric). Metric is ray-distance for hitscan, time-of-hit
    for projectile (lower = closer). Recency gate is applied by the caller."""
    phys = WEAPON_PHYSICS[weapon]
    rel, vel, half = _decode_slot(esc, t, slot)
    spread_rad = math.radians(phys.get("spread_deg", 0.0))
    if phys["hitscan"]:
        return _hitscan_test(look_u, rel, half, phys["range"], spread_rad)
    return _projectile_test(
        look_u, rel, vel, half, phys["speed"],
        phys.get("splash", 0.0), phys.get("max_t", 3.0),
        phys.get("gravity", 0.0),
    )


def _look_unit(look_t: np.ndarray) -> np.ndarray | None:
    """Return a unit-norm look vector or None if the input is degenerate."""
    look_t = np.asarray(look_t, dtype=np.float32)
    ln = float(np.linalg.norm(look_t))
    if ln < 1e-3:
        return None
    return look_t / ln


def slot_would_be_hit(weapon: int, look_t: np.ndarray, esc: np.ndarray,
                      t: int, slot: int) -> bool:
    """Recency-gated boolean: would a shot with ``weapon`` at frame ``t``
    hit the actor at ``slot``? Used by the cone labeler's sticky-keep
    physics check (see ``hit_labeler.label_hit_anchored``)."""
    if weapon not in WEAPON_PHYSICS or weapon == 0:
        return False
    if esc[t, slot, ACTOR_RECENCY_OFFSET] > 0.0:
        return False
    look_u = _look_unit(look_t)
    if look_u is None:
        return False
    hit, _ = _physics_hit_for_slot(weapon, look_u, esc, t, slot)
    return hit


def find_hit_slot(weapon: int, look: np.ndarray, esc: np.ndarray, t: int,
                  enemy_mask: np.ndarray) -> int:
    """Argmin-metric hit slot at frame ``t``, or -1 if no enemy gets hit.
    Hitscan uses ray-distance, projectile uses time-of-hit. Recency-gated."""
    if weapon not in WEAPON_PHYSICS or weapon == 0:
        return -1
    look_u = _look_unit(look[t])
    if look_u is None:
        return -1
    best_slot = -1
    best_metric = math.inf
    for s in range(esc.shape[1]):
        if not enemy_mask[t, s]:
            continue
        if esc[t, s, ACTOR_RECENCY_OFFSET] > 0.0:
            continue
        hit, metric = _physics_hit_for_slot(weapon, look_u, esc, t, s)
        if hit and metric < best_metric:
            best_metric = metric
            best_slot = s
    return best_slot


def all_hits_at_fire(weapon: int, look_t: np.ndarray, esc: np.ndarray,
                     eids: np.ndarray, t: int, enemy_mask: np.ndarray) -> set[int]:
    """Set of enemy pids the projectile/ray would hit at frame ``t``.
    Used by the v3 distribution labeler to admit physics-hit candidates
    alongside cone candidates (cone-OR-physics co-primary). Recency-gated;
    pids <= 0 are skipped (no-pid sentinel)."""
    if weapon not in WEAPON_PHYSICS or weapon == 0:
        return set()
    look_u = _look_unit(look_t)
    if look_u is None:
        return set()
    hits: set[int] = set()
    for s in range(esc.shape[1]):
        if not enemy_mask[t, s]:
            continue
        if esc[t, s, ACTOR_RECENCY_OFFSET] > 0.0:
            continue
        hit, _ = _physics_hit_for_slot(weapon, look_u, esc, t, s)
        if hit:
            pid = int(eids[t, s, 2])
            if pid > 0:
                hits.add(pid)
    return hits
