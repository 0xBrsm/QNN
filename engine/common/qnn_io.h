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

#include <math.h>
#include <stdio.h>
#include <stdint.h>

/* ── Observation dimensions (must match qnn/engine_norm.py) ────── */
/* Entity-level capacities: QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS
   from qnn_vocab.h. */

/* Spatial depth atlas (final wire.12): elevation-major code grid,
 * nibble-packed on the wire. Must match qnn/engine_norm.py
 * (ATLAS_ELEV_DEG / ATLAS_YAWS / ATLAS_DEPTH_LEVELS / ATLAS_MISS_CODE). */
#define QNN_OBS_ATLAS_ELEVS          11
#define QNN_OBS_ATLAS_YAWS           24
#define QNN_OBS_ATLAS_PACKED_BYTES   (QNN_OBS_ATLAS_YAWS / 2)
#define QNN_OBS_ATLAS_YAWS_LEGACY    72
#define QNN_OBS_ATLAS_YAWS_MAX       QNN_OBS_ATLAS_YAWS_LEGACY
#define QNN_OBS_ATLAS_MISS_CODE      15

/* Which spatial obs block the emit path produces per tick. Exactly ONE
 * is emitted — the engine obeys the loaded model's codec, never computes
 * both. The ONNX load path pushes the active codec's declared mode via
 * QNN_IOSetSpatialMode; the default (no model loaded — e.g. corpus
 * collect, which packs the flat atlas obs buffer) is the v2 atlas. */
typedef enum {
	QNN_SPATIAL_MODE_ATLAS = 0,   /* wire.12 depth atlas (default) */
	QNN_SPATIAL_MODE_ATLAS_LEGACY,/* pre-final a26 rc1: 72 unpacked cells */
	QNN_SPATIAL_MODE_RAYCAST_V1,  /* wire.11 raycast scalars */
} qnn_spatial_mode_t;

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
 *  SELF BLOCK (27 B):
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
 *    20   1   view_pitch         i8         pitch_deg / 90, range ~[-1, 1]
 *                                            (Quake pitch engine-clamped ±70°)
 *    21   6   look_delta[3]      f16 × 3    look[t-1]-look[t-2] (realized
 *                                            look-vec change; ~0 steady turn)
 *
 *  SPATIAL BLOCK (132 B — center-ray depth atlas, elevation-major):
 *    atlas[11][12]  u8 × 132   two 4-bit depth codes per byte; low nibble
 *                              is the even yaw cell, high nibble the odd.
 *                              Codes 0..14 index the log depth ladder;
 *                              15 = no hit within the band's range.
 *    Band b centers at elevation -75°+15°·b; yaw cell y centers at
 *    15°·y counter-clockwise from view yaw (cell 0 = forward).
 *    Mirrors src/qnn/wire.py:_unpack_native_spatial.
 *
 *  ENTITY STREAM (variable-length, starts at offset 159):
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

#define QNN_OBS_SELF_BLOCK_BYTES         27
#define QNN_OBS_SPATIAL_BLOCK_BYTES \
	(QNN_OBS_ATLAS_ELEVS * QNN_OBS_ATLAS_PACKED_BYTES)

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
#define QNN_OBS_OFF_SELF_VIEW_PITCH      20
#define QNN_OBS_OFF_SELF_LOOK_DELTA      21   /* f16 × 3 = 6 B (21..26) */

/* QNN_OBS_BUFFER_SIZE (864) is defined in qnn.h (shared with tick-emit
 * state). Bytes 0..847 cover the maximum payload; 848..863 are the optional
 * pose tail. Actor is the maximum-width entity row:
 * type 1 + ids 3 + event count 1 + events 8 + scalars 30 = 43 B. */
#define QNN_OBS_MAX_ACTOR_ROW_BYTES \
	(1 + 3 + 1 + 2 * QNN_MAX_ENTITY_EVENTS + 30)
#define QNN_OBS_MAX_PAYLOAD_BYTES \
	(QNN_OBS_OFF_ENTITY_STREAM + 1 + \
	 QNN_MAX_TOKEN_OBJECTS * QNN_OBS_MAX_ACTOR_ROW_BYTES)

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

/* Nibble-pack one atlas band row: QNN_OBS_ATLAS_YAWS unpacked 4-bit
 * codes -> QNN_OBS_ATLAS_PACKED_BYTES bytes, low nibble = even yaw cell.
 * This IS the wire's bit layout, so it lives here in one place: the flat
 * obs packer (qnn_io.c), the ONNX scratch packer (qnn_onnx.c), and
 * qnn/wire.py's _unpack_native_spatial must agree byte for byte. */
static inline void QNN_AtlasPackRow(uint8_t *dst, const uint8_t *codes)
{
	int j;

	for (j = 0; j < QNN_OBS_ATLAS_PACKED_BYTES; ++j)
		dst[j] = (uint8_t)((codes[2 * j] & 0x0fu) |
			((codes[2 * j + 1] & 0x0fu) << 4));
}

/* Nearest-level 4-bit atlas depth code.  Levels mirror
 * engine_norm.ATLAS_DEPTH_LEVELS: 0, 8, 16, 24, 36, 52, 72, 100, 136,
 * 184, 248, 336, 456, 620, 1016; thresholds are the midpoints.  Miss
 * (no hit within the band's range) is QNN_OBS_ATLAS_MISS_CODE, encoded
 * by the caller — this helper only maps hit distances. */
static inline uint8_t QNN_AtlasQuantizeDepth(float dist)
{
	static const float mid[14] = {
		4.0f, 12.0f, 20.0f, 30.0f, 44.0f, 62.0f, 86.0f, 118.0f,
		160.0f, 216.0f, 292.0f, 396.0f, 538.0f, 818.0f,
	};
	uint8_t code = 0;

	while (code < 14 && dist >= mid[code])
		code++;
	return code;
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
	float    view_pitch;         /* pitch normalized to ~[-1, 1] (deg / 90) */
	float    look_delta[3];      /* look[t-1]-look[t-2] (realized look-vec change);
	                              * ~0 under steady turn. See QNN_ComputeLookDelta. */
} qnn_self_token_t;

/* v1 raycast-scalar spatial sector (wire.11 obs block). Emitted every
 * tick alongside the v2 depth atlas so the single bin serves both
 * wire.11 (a24/a25) and wire.12 (a26) models. */
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
	/* v1 raycast-scalar spatial tokens (wire.11). Emitted every tick
	 * beside spatial_atlas; the codec selected by the model's
	 * wire_contract stamp binds one or the other. */
	qnn_spatial_token_t	spatial[QNN_SPATIAL_TOKEN_COUNT];
	/* Center-ray depth atlas (final wire.12): elevation-major unpacked
	 * 4-bit codes — 0..14 index the log depth
	 * ladder, QNN_OBS_ATLAS_MISS_CODE = no hit within the band's range.
	 * Tick scratch stays unpacked; flat-wire and ONNX packers combine
	 * adjacent yaw codes into nibbles. Band identity is implicit in fixed
	 * wire order and receives a learned band-id embedding model-side. */
	/* Sized to the largest still-supported atlas. Current 24-cell rows use
	 * columns 0..23; the a26 rc1 compatibility codec uses all 72. */
	uint8_t	spatial_atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX];
} qnn_tick_result_t;

/* ── Public API ────────────────────────────────────────────────── */

void QNN_IOInit(const qnn_map_state_t *map_state);
void QNN_IOUpdate(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag);
void QNN_IOEmit(const qnn_snapshot_t *snapshot, qnn_tick_result_t *out);
void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *result);

/* Select which spatial block QNN_IOEmit produces. Set once at model load
 * from the selected codec (qnn_onnx.c) so the emit path computes EXACTLY
 * the loaded contract's spatial obs and skips the other. */
void QNN_IOSetSpatialMode(qnn_spatial_mode_t mode);

/* Pose tail: capture-time (origin_xyz, view_yaw) as 4 f32 in the obs
 * buffer's guaranteed-zero tail (the entity stream tops out well below
 * it). Not part of the wire contract — consumers opt in per channel:
 * collect stashes when QNN_POSE_DIAG is set (pose sidecars), the
 * trainer/arena workers when QNN_POSE_TAIL=1 (closed-loop pose for
 * probe-grid obs assembly + geometry metrics). Off = tail stays zero. */
#define QNN_POSE_TAIL_OFF (QNN_OBS_BUFFER_SIZE - 16)
typedef char qnn_obs_payload_must_not_overlap_pose_tail[
	QNN_OBS_MAX_PAYLOAD_BYTES <= QNN_POSE_TAIL_OFF ? 1 : -1];
void QNN_IOStashPoseTail(uint8_t *obs, const qnn_snapshot_t *snapshot);
int QNN_IOPoseTailEnabled(void);

/* ── Internal module functions (called by qnn_io.c) ──────────── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot);
/* v1 raycast-scalar spatial tokens (wire.11) — emitted every tick
 * beside the v2 atlas below. */
void QNN_SpatialEmitTokens(const qnn_snapshot_t *snapshot,
	qnn_spatial_token_t tokens[QNN_SPATIAL_TOKEN_COUNT]);
void QNN_SpatialEmitAtlas(const qnn_snapshot_t *snapshot,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX],
	int yaw_count);
/* Load-time, world-anchored, static-world-only probe carve (movers
 * excluded) — feeds the nav_query probe_atlas dump. */
int QNN_SpatialCarveProbeAtlas(const float origin[3], float yaw_deg,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS]);
/* Diagnostic world-only cost sweep for 72/36/24 yaw cells. */
int QNN_SpatialBenchmarkWorldAtlas(const float origin[3], int yaw_count,
	int iterations, double *microseconds_per_atlas,
	double *nanoseconds_per_ray, unsigned int *checksum);
/* Diagnostic-only access to the immutable, map-load hull-1 boundary.
 * `hull_faces` nav queries use this to build static-memory experiments;
 * neither function participates in the observation wire. */
int QNN_SpatialWorldFaceCount(void);
void QNN_SpatialWriteWorldFacesJson(FILE *out);
int QNN_SpatialWriteWorldCellsJson(FILE *out);

#endif /* QNN_IO_H */
