/*
 * qnn_io.h — Unified tick IO: action in, tokens out.
 *
 * All token types, buffer layout constants, and the public tick API.
 * This is the single header that consumers (collect, trainer, client)
 * include.  Also the wire format contract — buffer offsets must match
 * Python obs_format.py.
 */

#ifndef QNN_IO_H
#define QNN_IO_H

#include "qnn.h"
#include "qnn_vocab.h"
#include "qnn_object.h"

#include <stdint.h>

/* ── Observation dimensions (must match obs_format.py) ─────────── */
/* Entity-level capacities: QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS
   from qnn_vocab.h.  Per-type scalar/ID dims from qnn_object.h. */

#define QNN_OBS_SELF_SCALAR_DIM      14
#define QNN_OBS_SPATIAL_COUNT         9
#define QNN_OBS_SPATIAL_SCALAR_DIM   13
#define QNN_OBS_ACTION_HISTORY_LEN    8
#define QNN_OBS_ACTION_HISTORY_DIM    8  /* move[3] + look[3] + fire + switch */
#define QNN_OBS_SELF_POWERUP_SLOTS    5

/* ── Normalization constants ───────────────────────────────────── */

#define QNN_DIST_SCALE      1000.0f
#define QNN_VELOCITY_SCALE  2000.0f
#define QNN_TIME_SCALE        60.0f
/* Engine physics constants moved to qnn.h (shared with snap/inference). */

/* ── Action history ────────────────────────────────────────────── */

#define QNN_ACTION_SWITCH_SLOTS 5

/* ── Obs buffer wire format (little-endian) ───────────────────── */

/*  Offset  Field                        Type       Shape             Bytes */
/*       0  self_scalars                 float32    [14]                 56 */
/*      56  self_weapon_id               int32      [1]                   4 */
/*      60  self_armor_type_id           int32      [1]                   4 */
/*      64  self_powerup_ids             int32      [5]                  20 */
/*      84  self_movement_id             int32      [1]                   4 */
/*      88  spatial_scalars              float32    [9,13]              468 */
/*     556  action_history               float32    [8,8]               256 */
/*     812  entity_stream                variable   (see token_spec_v9) ~1825 max */
/*   Total (max):                                                     2637 */
/*   Buffer oversized for safety. */

/* QNN_OBS_BUFFER_SIZE is defined in qnn.h (shared with tick-emit state). */

#define QNN_OBS_OFF_SELF_SCALARS            0
#define QNN_OBS_OFF_SELF_WEAPON_ID         56
#define QNN_OBS_OFF_SELF_ARMOR_TYPE_ID     60
#define QNN_OBS_OFF_SELF_POWERUP_IDS       64
#define QNN_OBS_OFF_SELF_MOVEMENT_ID       84
#define QNN_OBS_OFF_SPATIAL                88
#define QNN_OBS_OFF_ACTION_HISTORY        556
#define QNN_OBS_OFF_ENTITY_STREAM         812

/* ── Buffer write helpers (little-endian) ──────────────────────── */

static inline void QNN_BufWriteF32(uint8_t *buf, int offset, float value)
{
	union { float f; uint32_t u; } bits;
	bits.f = value;
	buf[offset + 0] = (uint8_t)(bits.u & 0xffu);
	buf[offset + 1] = (uint8_t)((bits.u >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((bits.u >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((bits.u >> 24) & 0xffu);
}

static inline void QNN_BufWriteI32(uint8_t *buf, int offset, int32_t value)
{
	uint32_t u = (uint32_t)value;
	buf[offset + 0] = (uint8_t)(u & 0xffu);
	buf[offset + 1] = (uint8_t)((u >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((u >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((u >> 24) & 0xffu);
}

/* ── Token types ───────────────────────────────────────────────── */

typedef struct
{
	float health;
	float armor;
	float weapon_sg;
	float weapon_ng;
	float weapon_gl;
	float weapon_rl;
	float weapon_lg;
	float ammo_shells;
	float ammo_nails;
	float ammo_rockets;
	float ammo_cells;
	float vel[3];
	int weapon_id;
	int armor_type_id;
	int movement_id;
	int powerup_ids[QNN_OBS_SELF_POWERUP_SLOTS];
} qnn_self_token_t;

typedef struct
{
	float	dir[3];
	float	nearest_dist;
	float	mean_dist;
	float	openness;
	float	solid_frac;
	float	water_frac;
	float	slime_frac;
	float	lava_frac;
	float	traversable;
	float	dropoff;
	float	clearance;
} qnn_spatial_token_t;

typedef struct
{
	float move[3];
	float look[3];
	float fire;
	float switch_norm;
} qnn_action_token_t;

/* ── Tick result ───────────────────────────────────────────────── */

typedef struct
{
	qnn_self_token_t	self;
	qnn_tagged_token_t	entities[QNN_MAX_TOKEN_OBJECTS];
	int			entity_count;
	qnn_spatial_token_t	spatial[QNN_SPATIAL_TOKEN_COUNT];
	qnn_action_token_t	action_history[QNN_OBS_ACTION_HISTORY_LEN];
	int			action_history_count;
} qnn_tick_result_t;

/* ── Public API ────────────────────────────────────────────────── */

void QNN_IOInit(const qnn_map_state_t *map_state);
void QNN_IOUpdate(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag);
void QNN_IOEmit(const qnn_snapshot_t *snapshot, qnn_tick_result_t *out);
void QNN_IOPushAction(const qnn_action_t *action);
void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *result);

/* ── Internal module functions (called by qnn_io.c) ──────────── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot);
void QNN_SpatialEmitTokens(const qnn_snapshot_t *snapshot, qnn_spatial_token_t tokens[QNN_SPATIAL_TOKEN_COUNT]);

#endif /* QNN_IO_H */
