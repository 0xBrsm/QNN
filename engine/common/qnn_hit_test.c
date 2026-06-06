/* qnn_hit_test.c — verbatim port of qnn.bc.weapon_physics.
 *
 * Identical algorithms to the Python labeler: slab method ray-AABB for
 * hitscan with spread-cone admission, step-loop projectile sim with
 * gravity for projectile weapons. dt=0.05s and the WEAPON_PHYSICS
 * table both match weapon_physics.py exactly.
 */
#include "qnn_hit_test.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Per-weapon physics — mirrors qnn.bc.weapon_physics.WEAPON_PHYSICS.
 * Indexed by impulse id (0..8). Entry 0 (none) has all-zero fields so
 * QNN_HitTest returns false for it via the explicit weapon-id check. */
typedef struct {
    bool  hitscan;
    float range;       /* max ray-distance for hitscan */
    float spread_deg;  /* half-angle spread cone, hitscan only */
    float speed;       /* projectile muzzle speed */
    float splash;      /* splash radius (0 = no splash) */
    float max_t;       /* projectile flight cap, seconds */
    float gravity;     /* projectile gravity, world units/s^2 */
} WeaponPhys;

static const WeaponPhys WEAPON_PHYSICS[9] = {
    /* 0: none — never hits */
    { false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    /* 1: axe   — hitscan, 64u, no spread */
    { true,  64.0f,   0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    /* 2: sg    — hitscan, 1500u, 2.3° spread */
    { true,  1500.0f, 2.3f, 0.0f, 0.0f, 0.0f, 0.0f },
    /* 3: ssg   — hitscan, 1500u, 8°   spread */
    { true,  1500.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    /* 4: ng    — projectile, 1000u/s, no splash, 3s cap */
    { false, 0.0f, 0.0f, 1000.0f,   0.0f, 3.0f, 0.0f },
    /* 5: sng   — same as ng */
    { false, 0.0f, 0.0f, 1000.0f,   0.0f, 3.0f, 0.0f },
    /* 6: gl    — projectile, 600u/s, 120u splash, 2.5s cap, gravity 800 */
    { false, 0.0f, 0.0f, 600.0f,  120.0f, 2.5f, 800.0f },
    /* 7: rl    — projectile, 1000u/s, 120u splash, 4s cap, no gravity */
    { false, 0.0f, 0.0f, 1000.0f, 120.0f, 4.0f, 0.0f },
    /* 8: lg    — hitscan, 600u, no spread */
    { true,  600.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
};

static inline float dot3(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline float norm3(const float v[3]) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/* Slab method ray-AABB. ray origin = (0,0,0), box centered at
 * ``box_center`` with half-extents ``box_half``. Returns entry distance
 * along the ray (>=0) or +INF on miss. Mirrors
 * weapon_physics._ray_aabb_distance. */
static float ray_aabb_distance(const float direction_unit[3],
                                const float box_center[3],
                                const float box_half[3]) {
    /* In the Python helper: p = -box_center; d = direction_unit.
     * We follow the same sign convention exactly. */
    float p[3] = { -box_center[0], -box_center[1], -box_center[2] };
    const float eps = 1e-6f;
    float tmin = -INFINITY;
    float tmax = +INFINITY;
    for (int i = 0; i < 3; ++i) {
        float d = direction_unit[i];
        if (fabsf(d) < eps) {
            if (p[i] < -box_half[i] || p[i] > box_half[i]) return INFINITY;
        } else {
            float t1 = (-box_half[i] - p[i]) / d;
            float t2 = ( box_half[i] - p[i]) / d;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return INFINITY;
        }
    }
    if (tmax < 0.0f) return INFINITY;
    return tmin > 0.0f ? tmin : 0.0f;
}

/* Hitscan test — mirrors weapon_physics._hitscan_test. */
static bool hitscan_test(const float look_unit[3],
                          const float rel[3],
                          const float half[3],
                          float max_range,
                          float spread_rad) {
    float dist = ray_aabb_distance(look_unit, rel, half);
    if (dist <= max_range) return true;
    if (spread_rad > 0.0f) {
        float rel_norm = norm3(rel);
        if (rel_norm < 1e-3f) return false;
        float unit_rel[3] = { rel[0]/rel_norm, rel[1]/rel_norm, rel[2]/rel_norm };
        float cos_off = dot3(unit_rel, look_unit);
        if (cos_off < -1.0f) cos_off = -1.0f;
        if (cos_off >  1.0f) cos_off =  1.0f;
        float ang_off = acosf(cos_off);
        if (ang_off <= spread_rad && rel_norm <= max_range) return true;
    }
    return false;
}

/* Projectile test — mirrors weapon_physics._projectile_test (dt=0.05s).
 * Splash radius is the closest-point distance from the projectile to
 * the bbox (axis-aligned slab clamp), matching the Python helper. */
static bool projectile_test(const float look_unit[3],
                             const float rel[3],
                             const float vel[3],
                             const float half[3],
                             float speed,
                             float splash,
                             float max_t,
                             float gravity) {
    const float dt = 0.05f;
    float half_ext[3] = {
        half[0] > 4.0f ? half[0] : 4.0f,
        half[1] > 4.0f ? half[1] : 4.0f,
        half[2] > 4.0f ? half[2] : 4.0f,
    };
    int n_steps = (int)(max_t / dt);
    for (int step = 0; step < n_steps; ++step) {
        float t = step * dt;
        float proj[3] = {
            look_unit[0] * speed * t,
            look_unit[1] * speed * t,
            look_unit[2] * speed * t - 0.5f * gravity * t * t,
        };
        float targ[3] = {
            rel[0] + vel[0] * t,
            rel[1] + vel[1] * t,
            rel[2] + vel[2] * t,
        };
        float diff[3] = {
            proj[0] - targ[0],
            proj[1] - targ[1],
            proj[2] - targ[2],
        };
        if (splash > 0.0f) {
            float gap[3] = {
                fabsf(diff[0]) - half_ext[0],
                fabsf(diff[1]) - half_ext[1],
                fabsf(diff[2]) - half_ext[2],
            };
            if (gap[0] < 0.0f) gap[0] = 0.0f;
            if (gap[1] < 0.0f) gap[1] = 0.0f;
            if (gap[2] < 0.0f) gap[2] = 0.0f;
            float gap_norm = norm3(gap);
            if (gap_norm <= splash) return true;
        }
        if (fabsf(diff[0]) <= half_ext[0] &&
            fabsf(diff[1]) <= half_ext[1] &&
            fabsf(diff[2]) <= half_ext[2]) {
            return true;
        }
    }
    return false;
}

bool QNN_HitTest(int weapon_id_impulse,
                 const float look_unit[3],
                 const float rel[3],
                 const float vel[3],
                 const float half_extents[3]) {
    if (weapon_id_impulse <= 0 || weapon_id_impulse > 8) return false;
    /* Reject degenerate look — caller should pre-normalize; we still
     * defend so a zero look doesn't trigger acosf domain issues. */
    if (norm3(look_unit) < 1e-3f) return false;
    const WeaponPhys *p = &WEAPON_PHYSICS[weapon_id_impulse];
    if (p->hitscan) {
        float spread_rad = p->spread_deg * (float)(M_PI / 180.0);
        return hitscan_test(look_unit, rel, half_extents, p->range, spread_rad);
    }
    return projectile_test(look_unit, rel, vel, half_extents,
                            p->speed, p->splash, p->max_t, p->gravity);
}
