/* qnn_hit_test.h — "would this weapon hit this slot, given current aim?"
 *
 * Verbatim port of qnn.bc.weapon_physics. Single source of truth for
 * hit/no-hit physics shared by:
 *   - the v3 distribution labeler (called from Python via ctypes)
 *   - the BC dataloader's per-frame entity_hit_test precompute (ctypes)
 *   - the engine at live-play decision time (linked directly)
 *
 * All inputs in Quake world units (the same scale weapons.qc operates
 * in). Callers that have view-frame-scaled obs values must un-scale by
 * QNN_DIST_SCALE / QNN_VEL_SCALE first. No allocations, no globals;
 * pure function of inputs.
 *
 * Weapon ids are impulse-encoded (0..8): 0=none, 1=axe, 2=sg, 3=ssg,
 * 4=ng, 5=sng, 6=gl, 7=rl, 8=lg. The WEAPON_PHYSICS table embedded in
 * the .c file mirrors qnn.bc.weapon_physics.WEAPON_PHYSICS exactly.
 */
#ifndef QNN_HIT_TEST_H
#define QNN_HIT_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/* Returns true if a shot fired with ``weapon_id_impulse`` (0..8) and
 * aim direction ``look_unit`` would hit the actor at relative position
 * ``rel`` with velocity ``vel`` and half-extents ``half_extents``.
 *
 * Inputs:
 *   weapon_id_impulse: impulse-encoded weapon id (0..8). 0 returns false.
 *   look_unit[3]:      player's current aim direction (must be unit-norm).
 *   rel[3]:            target position relative to player, world units.
 *   vel[3]:            target velocity relative to player, world units/s.
 *   half_extents[3]:   target bbox half-extents, world units.
 *
 * Returns false on unknown weapon id, degenerate look vector, or no hit. */
bool QNN_HitTest(int weapon_id_impulse,
                 const float look_unit[3],
                 const float rel[3],
                 const float vel[3],
                 const float half_extents[3]);

#ifdef __cplusplus
}
#endif

#endif /* QNN_HIT_TEST_H */
