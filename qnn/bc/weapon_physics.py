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

# Per-weapon physics. Source: vendor/quakec/qc/weapons.qc (NetQuake QC).
#
# Labeler-facing keys (existing; consumed by `hit_test` and cone gates):
#   hitscan=True with `range` u and `spread_deg` half-angle.
#   hitscan=False with `speed` u/s, optional `splash` u, `gravity` u/s²,
#     `max_t` flight cap in seconds.
#
# Model-token keys (new; consumed by `build_model_weapon_scalars`):
#   damage   — per-fire expected damage. SG/SSG: pellets × per-pellet
#              dmg. GL: T_RadiusDamage center value (no direct-hit term).
#   cooldown — literal attack_finished delay in seconds.
#   v_horiz  — initial along-view speed in u/s. Hitscan = QNN_VEL_SCALE
#              (= 2000, sentinel "instant propagation").
#   v_vert_0 — initial up-velocity component in u/s (GL only).
#   max_dist — hard cap on weapon reach in u. Only meaningful for axe
#              (64) and LG (600); other weapons get 4096 (world-axis
#              cap). Projectile lifetime × speed exceeds map size in
#              every case so we don't encode literal lifetime here.
WEAPON_PHYSICS: dict[int, dict] = {
    1: {  # Axe
        "hitscan": True, "range": 64.0, "spread_deg": 0.0, "splash": 0.0,
        "damage": 20.0, "cooldown": 0.5,
        "v_horiz": 2000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 64.0,
    },
    2: {  # SG: 6 pellets × 4 dmg = 24 expected
        "hitscan": True, "range": 1500.0, "spread_deg": 2.3, "splash": 0.0,
        "damage": 24.0, "cooldown": 0.5,
        "v_horiz": 2000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 2048.0,
    },
    3: {  # SSG: 14 pellets × 4 dmg = 56 expected
        "hitscan": True, "range": 1500.0, "spread_deg": 8.0, "splash": 0.0,
        "damage": 56.0, "cooldown": 0.7,
        "v_horiz": 2000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 2048.0,
    },
    4: {  # NG
        "hitscan": False, "speed": 1000.0, "splash": 0.0, "max_t": 3.0,
        "damage": 9.0, "cooldown": 0.2,
        "v_horiz": 1000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 4096.0,
    },
    5: {  # SNG
        "hitscan": False, "speed": 1000.0, "splash": 0.0, "max_t": 3.0,
        "damage": 18.0, "cooldown": 0.2,
        "v_horiz": 1000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 4096.0,
    },
    6: {  # GL: T_RadiusDamage(120); no direct-hit component
        "hitscan": False, "speed": 600.0, "splash": 120.0, "max_t": 2.5,
        "gravity": 800.0,
        "damage": 120.0, "cooldown": 0.6,
        "v_horiz": 600.0, "v_vert_0": 200.0, "max_dist": 4096.0,
    },
    7: {  # RL: 100 direct (+ splash to others)
        "hitscan": False, "speed": 1000.0, "splash": 120.0, "max_t": 4.0,
        "damage": 100.0, "cooldown": 0.8,
        "v_horiz": 1000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 4096.0,
    },
    8: {  # LG: 30 per beam frame
        "hitscan": True, "range": 600.0, "spread_deg": 0.0, "splash": 0.0,
        "damage": 30.0, "cooldown": 0.1,
        "v_horiz": 2000.0, "v_vert_0": 0.0, "gravity": 0.0, "max_dist": 600.0,
    },
}


# Universal normalization scales — mirror qnn.engine_norm. Inlined to
# keep this module's import surface narrow (no torch / qnn.engine_norm).
_MAX_HEALTH = 100.0
_TIME_SCALE = 60.0
_DIST_SCALE = QNN_DIST_SCALE   # 1000.0
_VEL_SCALE  = QNN_VEL_SCALE    # 2000.0

# Model weapon-token scalar layout. Row 0 = "no weapon" sentinel; rows
# 1..8 = weapons in impulse order, matching `self_weapon_id` indexing.
MODEL_TOKEN_SCALAR_DIM = 7
WT_DAMAGE   = 0
WT_COOLDOWN = 1
WT_V_HORIZ  = 2
WT_V_VERT_0 = 3
WT_GRAVITY  = 4
WT_MAX_DIST = 5
WT_RADIUS   = 6


def build_model_weapon_scalars() -> np.ndarray:
    """(9, 7) normalized static-scalar table for the weapon token.

    Row 0 is the "no weapon" sentinel (all zeros). Normalization mirrors
    the universal scales the rest of the obs uses: distances /1000,
    velocities /2000, time /60, damage /100.
    """
    out = np.zeros((9, MODEL_TOKEN_SCALAR_DIM), dtype=np.float32)
    for w_id, p in WEAPON_PHYSICS.items():
        out[w_id, WT_DAMAGE]   = p["damage"]   / _MAX_HEALTH
        out[w_id, WT_COOLDOWN] = p["cooldown"] / _TIME_SCALE
        out[w_id, WT_V_HORIZ]  = p["v_horiz"]  / _VEL_SCALE
        out[w_id, WT_V_VERT_0] = p["v_vert_0"] / _VEL_SCALE
        out[w_id, WT_GRAVITY]  = p["gravity"]  / _VEL_SCALE
        # max_dist /DIST_SCALE then clipped to [0, 1]. Practical
        # engagement ranges differ within ≤1000u (axe=64, LG=600);
        # anything beyond is "full-range" from the model's perspective.
        out[w_id, WT_MAX_DIST] = min(p["max_dist"] / _DIST_SCALE, 1.0)
        out[w_id, WT_RADIUS]   = p["splash"]   / _DIST_SCALE
    return out


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


def _decode_idx(esc: np.ndarray, t: int, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (rel_qu, vel_qu, half_qu) for one idx at one frame from the raw
    scalar array. Returns Quake-unit-scaled vectors."""
    rel  = esc[t, idx, ACTOR_REL_OFFSET:ACTOR_REL_OFFSET + 3].astype(np.float32) * QNN_DIST_SCALE
    vel  = esc[t, idx, ACTOR_VEL_OFFSET:ACTOR_VEL_OFFSET + 3].astype(np.float32) * QNN_VEL_SCALE
    half = esc[t, idx, ACTOR_HALFEXT_OFFSET:ACTOR_HALFEXT_OFFSET + 3].astype(np.float32) * QNN_DIST_SCALE
    return rel, vel, half


def _physics_hit_for_idx(weapon: int, look_u: np.ndarray, esc: np.ndarray,
                          t: int, idx: int) -> tuple[bool, float]:
    """Run the appropriate hit test for ``weapon`` against actor ``idx``.
    Returns (hit, metric). Metric is ray-distance for hitscan, time-of-hit
    for projectile (lower = closer). Recency gate is applied by the caller."""
    phys = WEAPON_PHYSICS[weapon]
    rel, vel, half = _decode_idx(esc, t, idx)
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


def idx_would_be_hit(weapon: int, look_t: np.ndarray, esc: np.ndarray,
                      t: int, idx: int) -> bool:
    """Recency-gated boolean: would a shot with ``weapon`` at frame ``t``
    hit the actor at ``idx``? Used by the cone labeler's sticky-keep
    physics check (see ``hit_labeler.label_hit_anchored``)."""
    if weapon not in WEAPON_PHYSICS or weapon == 0:
        return False
    if esc[t, idx, ACTOR_RECENCY_OFFSET] > 0.0:
        return False
    look_u = _look_unit(look_t)
    if look_u is None:
        return False
    hit, _ = _physics_hit_for_idx(weapon, look_u, esc, t, idx)
    return hit


def find_hit_idx(weapon: int, look: np.ndarray, esc: np.ndarray, t: int,
                  enemy_mask: np.ndarray) -> int:
    """Argmin-metric hit idx at frame ``t``, or -1 if no enemy gets hit.
    Hitscan uses ray-distance, projectile uses time-of-hit. Recency-gated."""
    if weapon not in WEAPON_PHYSICS or weapon == 0:
        return -1
    look_u = _look_unit(look[t])
    if look_u is None:
        return -1
    best_idx = -1
    best_metric = math.inf
    for s in range(esc.shape[1]):
        if not enemy_mask[t, s]:
            continue
        if esc[t, s, ACTOR_RECENCY_OFFSET] > 0.0:
            continue
        hit, metric = _physics_hit_for_idx(weapon, look_u, esc, t, s)
        if hit and metric < best_metric:
            best_metric = metric
            best_idx = s
    return best_idx


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
        hit, _ = _physics_hit_for_idx(weapon, look_u, esc, t, s)
        if hit:
            pid = int(eids[t, s, 2])
            if pid > 0:
                hits.add(pid)
    return hits
