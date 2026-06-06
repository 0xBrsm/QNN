/*
 * qnn_io.h — Unified tick IO: action in, tokens out.
 *
 * All token types, buffer layout constants, and the public tick API.
 * This is the single header that consumers (collect, trainer, client)
 * include.  Also the wire format contract — buffer offsets must match
 * Python qnn/wire.py and the per-field native widths in
 * src/qnn/engine_norm.py.
 */

#ifndef QNN_IO_H
#define QNN_IO_H

#include "qnn.h"
#include "qnn_vocab.h"
#include "qnn_object.h"

#include <stdint.h>

/* ── Observation dimensions (must match qnn/engine_norm.py) ────── */
/* Entity-level capacities: QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS
   from qnn_vocab.h. */

#define QNN_OBS_SPATIAL_COUNT         9

/* ── Normalization constants ───────────────────────────────────── */

#define QNN_DIST_SCALE      1000.0f
#define QNN_VELOCITY_SCALE  2000.0f
#define QNN_TIME_SCALE        60.0f
/* Engine physics constants moved to qnn.h (shared with snap/inference). */

/* ── Native-width obs buffer (little-endian, native dtypes) ───────
 *
 *  Replaces the legacy float32-everywhere layout.  Per-field native
 *  widths follow src/qnn/engine_norm.py exactly; the model-side
 *  dequantizers (qnn.model.dequant) divide by the scale factors below.
 *
 *  SELF BLOCK (20 B):
 *     0   1   health             u8
 *     1   1   effective_armor    u8   (= round(raw_armor × type_factor))
 *     2   1   ammo_shells        u8
 *     3   1   ammo_nails         u8
 *     4   1   ammo_rockets       u8
 *     5   1   ammo_cells         u8
 *     6   6   vel[3]             i16 × 3   clamped ±VELOCITY_SCALE
 *    12   2   attack_finished    f16        seconds (raw, dequant /60)
 *    14   1   self_weapon_id     u8
 *    15   1   self_movement_id   u8
 *    16   4   self_items         u32        raw cl.items bitfield
 *
 *  SPATIAL BLOCK (135 B = field-major across 9 sectors):
 *    Layout: for each field, all 9 sectors' values laid out
 *    contiguously, then the next field, etc.  Mirrors
 *    src/qnn/wire.py:_unpack_native_spatial.
 *     0    27  dir[9, 3]          i8 × 27   raw round(value × 127)
 *    27    18  nearest_dist[9]    u16 × 9   raw Quake units
 *    45    18  mean_dist[9]       u16 × 9   raw Quake units
 *    63     9  openness[9]        u8  × 9   raw round(value × 255)
 *    72     9  clearance[9]       u8  × 9
 *    81     9  traversable[9]     u8  × 9
 *    90     9  dropoff[9]         u8  × 9
 *    99     9  solid_frac[9]      u8  × 9
 *   108     9  water_frac[9]      u8  × 9
 *   117     9  slime_frac[9]      u8  × 9
 *   126     9  lava_frac[9]       u8  × 9
 *
 *  ENTITY STREAM (variable-length, starts at offset 155):
 *     0   1   n_tokens           u8
 *     Per token (variable):
 *       1 type_tag (u8)         QNN_TOKEN_* (0=PROJ, 1=ACTOR, 2=ITEM, 3=MOVER)
 *       1 subject_id (u8)
 *       1 modality_id (u8)
 *      [1 player_id (u8)        ACTOR only — omitted for other types]
 *       1 event_count (u8)
 *       event_count × 2  interleaved (action, source) u8 pairs
 *       … per-type scalars (see below)
 *
 *     PROJECTILE per-type (14 B):
 *         rel[3]  i16 × 3      raw Quake units
 *         vel[3]  i16 × 3      clamped ±VELOCITY_SCALE
 *         recency f16          raw seconds
 *     ACTOR per-type (30 B):
 *         half_extents[3] u8 × 3  (saturating)
 *         rel[3]   i16 × 3
 *         vel[3]   i16 × 3
 *         path[3]  i16 × 3
 *         path_dist u16
 *         eta      f16          raw seconds
 *         facing   u8           raw round(value × 255)
 *         team     u8
 *         score    u8           raw round(value × 255)
 *         recency  f16
 *     ITEM per-type (24 B):
 *         half_extents[3] u8 × 3
 *         rel[3]   i16 × 3
 *         path[3]  i16 × 3
 *         path_dist u16
 *         eta      f16
 *         amount   u8           raw round(value × 255)
 *         regen    f16
 *         recency  f16
 *     MOVER per-type (22 B):
 *         half_extents[3] u8 × 3
 *         rel[3]   i16 × 3
 *         path[3]  i16 × 3
 *         path_dist u16
 *         eta      f16
 *         state    u8
 *         recency  f16
 *
 *   `dist` is NOT emitted on the wire — the model dequantizer
 *   recomputes |rel| / DIST_SCALE.
 */

#define QNN_OBS_SELF_BLOCK_BYTES         20
#define QNN_OBS_SPATIAL_PER_SECTOR_BYTES 15
#define QNN_OBS_SPATIAL_BLOCK_BYTES (QNN_OBS_SPATIAL_COUNT * QNN_OBS_SPATIAL_PER_SECTOR_BYTES)

#define QNN_OBS_OFF_SELF                 0
#define QNN_OBS_OFF_SPATIAL              QNN_OBS_SELF_BLOCK_BYTES
#define QNN_OBS_OFF_ENTITY_STREAM        (QNN_OBS_OFF_SPATIAL + QNN_OBS_SPATIAL_BLOCK_BYTES)
/* Offsets of fields within the self block. */
#define QNN_OBS_OFF_SELF_HEALTH          0
#define QNN_OBS_OFF_SELF_EFF_ARMOR       1
#define QNN_OBS_OFF_SELF_AMMO_SHELLS     2
#define QNN_OBS_OFF_SELF_AMMO_NAILS      3
#define QNN_OBS_OFF_SELF_AMMO_ROCKETS    4
#define QNN_OBS_OFF_SELF_AMMO_CELLS      5
#define QNN_OBS_OFF_SELF_VEL             6
#define QNN_OBS_OFF_SELF_ATTACK_FIN      12
#define QNN_OBS_OFF_SELF_WEAPON_ID       14
#define QNN_OBS_OFF_SELF_MOVEMENT_ID     15
#define QNN_OBS_OFF_SELF_ITEMS           16

/* QNN_OBS_BUFFER_SIZE (4096) is defined in qnn.h (shared with tick-emit
 * state). The new layout occupies far less than 4096 B even with
 * QNN_MAX_TOKEN_OBJECTS=16 fully populated. */

/* ── Native-width little-endian write helpers ───────────────────── */

static inline void QNN_BufWriteU8(uint8_t *buf, int offset, uint8_t value)
{
	buf[offset] = value;
}

static inline void QNN_BufWriteI8(uint8_t *buf, int offset, int8_t value)
{
	buf[offset] = (uint8_t)value;
}

static inline void QNN_BufWriteU16(uint8_t *buf, int offset, uint16_t value)
{
	buf[offset + 0] = (uint8_t)(value & 0xffu);
	buf[offset + 1] = (uint8_t)((value >> 8) & 0xffu);
}

static inline void QNN_BufWriteI16(uint8_t *buf, int offset, int16_t value)
{
	uint16_t u = (uint16_t)value;
	QNN_BufWriteU16(buf, offset, u);
}

static inline void QNN_BufWriteU32(uint8_t *buf, int offset, uint32_t value)
{
	buf[offset + 0] = (uint8_t)(value & 0xffu);
	buf[offset + 1] = (uint8_t)((value >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((value >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((value >> 24) & 0xffu);
}

static inline void QNN_BufWriteF32(uint8_t *buf, int offset, float value)
{
	union { float f; uint32_t u; } bits;
	bits.f = value;
	QNN_BufWriteU32(buf, offset, bits.u);
}

static inline void QNN_BufWriteI32(uint8_t *buf, int offset, int32_t value)
{
	QNN_BufWriteU32(buf, offset, (uint32_t)value);
}

/* IEEE-754 binary32 → binary16 with round-to-nearest-even.  Subnormals
 * and ±inf/NaN are preserved; values outside the f16 normal range
 * saturate to ±inf (typical for clamped-positive seconds; the model
 * promotes f16 → f32 at dequant time). */
static inline uint16_t QNN_FloatToHalf(float value)
{
	union { float f; uint32_t u; } bits;
	uint32_t f, sign, exp, mant;
	uint16_t h;
	int e;

	bits.f = value;
	f = bits.u;
	sign = (f >> 31) & 0x1u;
	exp  = (f >> 23) & 0xffu;
	mant = f & 0x7fffffu;

	if (exp == 0xff) {
		/* Inf / NaN */
		h = (uint16_t)((sign << 15) | (0x1fu << 10) | (mant ? 0x200u : 0u));
		return h;
	}

	e = (int)exp - 127 + 15;
	if (e >= 0x1f) {
		/* Overflow → ±inf */
		return (uint16_t)((sign << 15) | (0x1fu << 10));
	}
	if (e <= 0) {
		/* Subnormal or underflow */
		if (e < -10) {
			return (uint16_t)(sign << 15);
		}
		/* Shift mantissa by (1 - e), include implicit leading 1. */
		mant = (mant | 0x800000u);
		{
			int shift = 14 - e;   /* 23 - 9 - e? compute: need >> (1 - e + 13) */
			uint32_t shifted, half_lsb;
			/* Right-shift to fit into 10 bits at position 0, but
			 * with round-to-nearest-even.  Target: 10-bit mantissa
			 * for a subnormal half. */
			shift = 14 - e;       /* equivalent to (23 - 10 + 1 - e) */
			shifted = mant >> shift;
			half_lsb = 1u << (shift - 1);
			if ((mant & ((1u << shift) - 1u)) > half_lsb
				|| (((mant & ((1u << shift) - 1u)) == half_lsb) && (shifted & 1u))) {
				shifted += 1u;
			}
			return (uint16_t)((sign << 15) | (shifted & 0x3ffu));
		}
	}
	{
		/* Normal — round mantissa to 10 bits with RNE. */
		uint32_t shifted = mant >> 13;
		uint32_t low = mant & 0x1fffu;
		if (low > 0x1000u || (low == 0x1000u && (shifted & 1u))) {
			shifted += 1u;
			if (shifted == 0x400u) {
				/* Rounded up into exponent. */
				shifted = 0u;
				e += 1;
				if (e >= 0x1f)
					return (uint16_t)((sign << 15) | (0x1fu << 10));
			}
		}
		return (uint16_t)((sign << 15) | ((uint16_t)e << 10) | (uint16_t)(shifted & 0x3ffu));
	}
}

static inline void QNN_BufWriteF16(uint8_t *buf, int offset, float value)
{
	QNN_BufWriteU16(buf, offset, QNN_FloatToHalf(value));
}

/* Helpers to convert clamped [-1, 1] / [0, 1] floats to their native
 * fixed-point widths. */
static inline int8_t QNN_QuantizeI8(float value)
{
	float v = value * 127.0f;
	if (v >  127.0f) v =  127.0f;
	if (v < -127.0f) v = -127.0f;
	return (int8_t)(v < 0.0f ? (int)(v - 0.5f) : (int)(v + 0.5f));
}

static inline uint8_t QNN_QuantizeU8Unit(float value)
{
	float v = value * 255.0f;
	if (v > 255.0f) v = 255.0f;
	if (v <   0.0f) v =   0.0f;
	return (uint8_t)(v + 0.5f);
}

static inline uint8_t QNN_QuantizeU8Saturating(float raw)
{
	/* Clamp a raw unsigned magnitude to u8 with saturation. */
	if (raw > 255.0f) raw = 255.0f;
	if (raw <   0.0f) raw =   0.0f;
	return (uint8_t)(raw + 0.5f);
}

static inline uint16_t QNN_QuantizeU16Saturating(float raw)
{
	if (raw > 65535.0f) raw = 65535.0f;
	if (raw <     0.0f) raw =     0.0f;
	return (uint16_t)(raw + 0.5f);
}

static inline int16_t QNN_QuantizeI16Clamped(float raw, float limit)
{
	if (raw >  limit) raw =  limit;
	if (raw < -limit) raw = -limit;
	return (int16_t)(raw < 0.0f ? (int)(raw - 0.5f) : (int)(raw + 0.5f));
}

/* ── Token types ───────────────────────────────────────────────── */
/* The internal token structs stay float-typed and are converted to
 * native widths in QNN_IOPackObsBuffer.  This keeps the emit helpers
 * (qnn_self_common.c, qnn_oracle.c, qnn_spatial.c) producing the same
 * floats they always have, while the new wire format only affects
 * serialization. */

typedef struct
{
	/* Raw engine values — packed to native widths at wire time. */
	int      health;             /* engine byte 0..255 */
	int      raw_armor;          /* STAT_ARMOR before type-factor */
	float    armor_type;         /* 0.0 / 0.3 / 0.6 / 0.8 */
	int      ammo_shells;
	int      ammo_nails;
	int      ammo_rockets;
	int      ammo_cells;
	float    vel[3];             /* view-frame, raw Quake units */
	/* QC ``self.attack_finished`` cooldown remaining in seconds (raw —
	 * model divides by QNN_TIME_SCALE).  0 = engine will process the
	 * next fire and any queued weapon-switch impulse this tick. */
	float    attack_finished;
	int      weapon_id;          /* 0..8 impulse-form */
	int      movement_id;        /* 0=ground/1=air/2..4=water */
	int32_t  items;              /* raw cl.items bitfield (Quake's `int items`) */
} qnn_self_token_t;

typedef struct
{
	float	dir[3];              /* in [-1, 1] */
	float	nearest_dist;        /* raw Quake units */
	float	mean_dist;
	float	openness;            /* in [0, 1] */
	float	solid_frac;
	float	water_frac;
	float	slime_frac;
	float	lava_frac;
	float	traversable;
	float	dropoff;
	float	clearance;
} qnn_spatial_token_t;

/* ── Tick result ───────────────────────────────────────────────── */

typedef struct
{
	qnn_self_token_t	self;
	qnn_tagged_token_t	entities[QNN_MAX_TOKEN_OBJECTS];
	int			entity_count;
	qnn_spatial_token_t	spatial[QNN_SPATIAL_TOKEN_COUNT];
} qnn_tick_result_t;

/* ── Public API ────────────────────────────────────────────────── */

void QNN_IOInit(const qnn_map_state_t *map_state);
void QNN_IOUpdate(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag);
void QNN_IOEmit(const qnn_snapshot_t *snapshot, qnn_tick_result_t *out);
void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *result);

/* ── Internal module functions (called by qnn_io.c) ──────────── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot);
void QNN_SpatialEmitTokens(const qnn_snapshot_t *snapshot, qnn_spatial_token_t tokens[QNN_SPATIAL_TOKEN_COUNT]);

#endif /* QNN_IO_H */
