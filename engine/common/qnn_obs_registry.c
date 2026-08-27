/*
 * qnn_obs_registry.c — The observation field registry + emit-plan compiler.
 *
 * See qnn_obs_registry.h for the contract-kind model and the percept
 * policy-v3 pins.  The entity-stream and atlas serializers here are the
 * former QNN_IOPackObsBuffer bodies moved behind registry rows; the
 * default plan's bytes are gate-tested against the pre-refactor packer
 * (src/engine/tests/qnn_obs_registry_test.c).
 */

#include "qnn_obs_registry.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Fail-loud helper for programming errors on the per-tick path (short
 * buffers, emitter overruns).  These are never data-dependent, so an
 * abort — caught by the qnn_fault SIGABRT handler with a trace — beats
 * a silently truncated frame. */
static void QNN_ObsFatal(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	fprintf(stderr, "qnn_obs_registry: FATAL: ");
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

/* ── Shared shape / size helpers ───────────────────────────────── */

static void QNN_ObsShapeScalar(const qnn_obs_params_t *params,
	int shape[QNN_OBS_MAX_SHAPE_DIMS], int *ndim)
{
	(void)params;
	memset(shape, 0, QNN_OBS_MAX_SHAPE_DIMS * sizeof(int));
	*ndim = 0;
}

static void QNN_ObsShapeVec3(const qnn_obs_params_t *params,
	int shape[QNN_OBS_MAX_SHAPE_DIMS], int *ndim)
{
	(void)params;
	memset(shape, 0, QNN_OBS_MAX_SHAPE_DIMS * sizeof(int));
	shape[0] = 3;
	*ndim = 1;
}

static int QNN_ObsSizeBytes1(const qnn_obs_params_t *params)
{
	(void)params;
	return 1;
}

static int QNN_ObsSizeBytes2(const qnn_obs_params_t *params)
{
	(void)params;
	return 2;
}

static int QNN_ObsSizeBytes4(const qnn_obs_params_t *params)
{
	(void)params;
	return 4;
}

static int QNN_ObsSizeBytes6(const qnn_obs_params_t *params)
{
	(void)params;
	return 6;
}

/* ── State field emitters (the self block, one row per field) ─────
 * Byte-for-byte the former QNN_IOPackObsBuffer self-block writes; the
 * quantizers and native widths must keep matching qnn/engine_norm.py. */

static int QNN_ObsEmitHealth(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0,
		QNN_QuantizeU8Saturating((float)ctx->result->self.health));
	return 1;
}

static int QNN_ObsEmitEffectiveArmor(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	const qnn_self_token_t *tok = &ctx->result->self;
	float eff_armor = (float)tok->raw_armor * tok->armor_type;

	(void)params;
	QNN_BufWriteU8(out, 0, QNN_QuantizeU8Saturating(eff_armor));
	return 1;
}

static int QNN_ObsEmitAmmoShells(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0,
		QNN_QuantizeU8Saturating((float)ctx->result->self.ammo_shells));
	return 1;
}

static int QNN_ObsEmitAmmoNails(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0,
		QNN_QuantizeU8Saturating((float)ctx->result->self.ammo_nails));
	return 1;
}

static int QNN_ObsEmitAmmoRockets(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0,
		QNN_QuantizeU8Saturating((float)ctx->result->self.ammo_rockets));
	return 1;
}

static int QNN_ObsEmitAmmoCells(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0,
		QNN_QuantizeU8Saturating((float)ctx->result->self.ammo_cells));
	return 1;
}

static int QNN_ObsEmitVel(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	const qnn_self_token_t *tok = &ctx->result->self;

	(void)params;
	QNN_BufWriteI16(out, 0,
		QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE));
	QNN_BufWriteI16(out, 2,
		QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE));
	QNN_BufWriteI16(out, 4,
		QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE));
	return 6;
}

static int QNN_ObsEmitAttackFinished(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteF16(out, 0, ctx->result->self.attack_finished);
	return 2;
}

static int QNN_ObsEmitWeaponId(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0, (uint8_t)ctx->result->self.weapon_id);
	return 1;
}

static int QNN_ObsEmitMovementId(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteU8(out, 0, (uint8_t)ctx->result->self.movement_id);
	return 1;
}

static int QNN_ObsEmitItems(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteI32(out, 0, ctx->result->self.items);
	return 4;
}

static int QNN_ObsEmitViewPitch(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	(void)params;
	QNN_BufWriteI8(out, 0, QNN_QuantizeI8(ctx->result->self.view_pitch));
	return 1;
}

static int QNN_ObsEmitLookDelta(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	const qnn_self_token_t *tok = &ctx->result->self;

	(void)params;
	QNN_BufWriteF16(out, 0, tok->look_delta[0]);
	QNN_BufWriteF16(out, 2, tok->look_delta[1]);
	QNN_BufWriteF16(out, 4, tok->look_delta[2]);
	return 6;
}

/* ── Atlas sensor ─────────────────────────────────────────────────
 * One sensor, two supported parameterizations (see header).  Both
 * serialize the elevation-major tick scratch (unpacked 4-bit codes);
 * only the wire packing differs. */

static qboolean QNN_ObsValidateAtlasParams(const qnn_obs_params_t *params,
	char *error, size_t error_size)
{
	const qnn_obs_atlas_params_t *p = &params->atlas;

	if (p->bands != QNN_OBS_ATLAS_ELEVS)
	{
		snprintf(error, error_size,
			"atlas: bands must be %d (got %d)",
			QNN_OBS_ATLAS_ELEVS, p->bands);
		return false;
	}
	if (p->yaw == QNN_OBS_ATLAS_YAWS && p->packed)
		return true;
	if (p->yaw == QNN_OBS_ATLAS_YAWS_LEGACY && !p->packed)
		return true;
	snprintf(error, error_size,
		"atlas: unsupported parameterization {yaw: %d, packed: %s} — "
		"supported: {yaw: %d, packed: true}, {yaw: %d, packed: false}",
		p->yaw, p->packed ? "true" : "false",
		QNN_OBS_ATLAS_YAWS, QNN_OBS_ATLAS_YAWS_LEGACY);
	return false;
}

static void QNN_ObsShapeAtlas(const qnn_obs_params_t *params,
	int shape[QNN_OBS_MAX_SHAPE_DIMS], int *ndim)
{
	const qnn_obs_atlas_params_t *p = &params->atlas;

	memset(shape, 0, QNN_OBS_MAX_SHAPE_DIMS * sizeof(int));
	shape[0] = p->bands;
	/* Packed rows carry two 4-bit codes per byte. */
	shape[1] = p->packed ? p->yaw / 2 : p->yaw;
	*ndim = 2;
}

static int QNN_ObsSizeAtlas(const qnn_obs_params_t *params)
{
	const qnn_obs_atlas_params_t *p = &params->atlas;

	return p->bands * (p->packed ? p->yaw / 2 : p->yaw);
}

static int QNN_ObsEmitAtlas(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	const qnn_obs_atlas_params_t *p = &params->atlas;
	int i;

	if (p->packed)
	{
		/* Nibble-packed (current wire): low nibble = even yaw cell,
		 * high nibble = odd.  QNN_AtlasPackRow is the one bit-layout
		 * authority shared with the ONNX scratch packer. */
		for (i = 0; i < p->bands; ++i)
			QNN_AtlasPackRow(out + i * QNN_OBS_ATLAS_PACKED_BYTES,
				ctx->result->spatial_atlas[i]);
		return p->bands * QNN_OBS_ATLAS_PACKED_BYTES;
	}

	/* Flat emit (restored from f84c36cd^): one code per byte,
	 * elevation-major.  The tick scratch rows are QNN_OBS_ATLAS_YAWS_MAX
	 * wide, so copy per band, never as one block. */
	for (i = 0; i < p->bands; ++i)
		memcpy(out + i * p->yaw, ctx->result->spatial_atlas[i],
			(size_t)p->yaw);
	return p->bands * p->yaw;
}

/* ── Entity percept (policies v1 / v3) ─────────────────────────────
 * Two row writers, one per disclosure policy:
 *   V1 — the f84c36cd^-era FULL stream, moved verbatim from the
 *        original QNN_IOPackObsBuffer: all four token types, recency
 *        in every row.  Wire-shim only (wire.9/.11/.12.x).
 *   V3 — the A27 pure-combat stream: actor/projectile only, no
 *        recency.  Ported from bb27a296 ("feat(a27): pure-combat
 *        substrate refactor") — the last commit where the native
 *        emit still wrote this shape, before the obs-api WS1 merge
 *        (5471d10e) folded the two policies onto one (v1-shaped)
 *        writer.  This is the fix for that fold: v3 rows are back to
 *        their own layout, matching qnn/obs_api.py's _POLICY_V3.
 * Both share event-slot and count packing; only the per-type scalar
 * block differs. */

static qboolean QNN_ObsValidateEntityParams(const qnn_obs_params_t *params,
	char *error, size_t error_size)
{
	const qnn_obs_entity_params_t *p = &params->entities;

	/* v3 = the A27 pure-combat disclosure (the default); v1 = the
	 * f84c36cd^-era FULL disclosure kept for the wire-shim
	 * generations.  Pins for both in the header. */
	if (strcmp(p->policy, "v3") != 0 && strcmp(p->policy, "v1") != 0)
	{
		snprintf(error, error_size,
			"entities: unknown disclosure policy \"%s\" (registered: "
			"v1, v3)",
			p->policy);
		return false;
	}
	if (p->max_tokens < 1 || p->max_tokens > QNN_MAX_TOKEN_OBJECTS)
	{
		snprintf(error, error_size,
			"entities: max_tokens must be 1..%d (got %d)",
			QNN_MAX_TOKEN_OBJECTS, p->max_tokens);
		return false;
	}
	return true;
}

static void QNN_ObsShapeEntities(const qnn_obs_params_t *params,
	int shape[QNN_OBS_MAX_SHAPE_DIMS], int *ndim)
{
	/* Variable-length byte stream — shape is the reserved size. */
	memset(shape, 0, QNN_OBS_MAX_SHAPE_DIMS * sizeof(int));
	shape[0] = 1 + params->entities.max_tokens * QNN_OBS_MAX_ACTOR_ROW_BYTES;
	*ndim = 1;
}

static int QNN_ObsSizeEntities(const qnn_obs_params_t *params)
{
	/* u8 n_tokens + max_tokens maximum-width (actor) rows. */
	return 1 + params->entities.max_tokens * QNN_OBS_MAX_ACTOR_ROW_BYTES;
}

/* Pack the event block: u8 count followed by `count` interleaved
 * (action, source) u8 pairs.  Matches the Python wire parser in
 * src/qnn/wire.py:_unpack_native_entity_stream. */
static int QNN_PackEventSlots(uint8_t *obs, int pos,
	const qnn_token_event_t *events, int event_count)
{
	int j;
	int cap = event_count;
	if (cap > QNN_MAX_ENTITY_EVENTS) cap = QNN_MAX_ENTITY_EVENTS;
	if (cap < 0) cap = 0;

	obs[pos++] = (uint8_t)cap;
	for (j = 0; j < cap; ++j)
	{
		obs[pos++] = (uint8_t)events[j].action_id;
		obs[pos++] = (uint8_t)events[j].source_id;
	}
	return pos;
}

/* Policy v1 (FULL) — all four token types, recency in every row.
 * Byte-for-byte the original QNN_IOPackObsBuffer switch (wire-shim
 * generations only: wire.9/.11/.12.x). */
static int QNN_ObsEmitEntityRowV1(uint8_t *out, int pos,
	const qnn_tagged_token_t *tt, int paths)
{
	switch (tt->type)
	{
	case QNN_TOKEN_PROJECTILE:
		{
			const qnn_projectile_token_t *tok = &tt->projectile;
			/* Common IDs — projectile has no player_id. */
			out[pos++] = (uint8_t)tok->subject_id;
			out[pos++] = (uint8_t)tok->modality_id;
			/* Events */
			pos = QNN_PackEventSlots(out, pos, tok->events, tok->event_count);
			/* Per-type scalars (14 B): rel i16×3, vel i16×3, recency f16 */
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteF16(out, pos, tok->recency); pos += 2;
		}
		break;

	case QNN_TOKEN_ACTOR:
		{
			const qnn_actor_token_t *tok = &tt->actor;
			out[pos++] = (uint8_t)tok->subject_id;
			out[pos++] = (uint8_t)tok->modality_id;
			out[pos++] = (uint8_t)tok->player_id;
			pos = QNN_PackEventSlots(out, pos, tok->events, tok->event_count);
			/* Per-type (30 B): half u8×3, rel i16×3, vel i16×3, path i16×3,
			 * path_dist u16, eta f16, facing u8, team u8, score u8, recency f16 */
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[0], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[1], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[2], 32767.0f) : 0); pos += 2;
			QNN_BufWriteU16(out, pos, paths ? QNN_QuantizeU16Saturating(tok->path_dist) : 0); pos += 2;
			QNN_BufWriteF16(out, pos, paths ? tok->eta : 0.0f); pos += 2;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Unit(tok->facing)); pos += 1;
			QNN_BufWriteU8 (out, pos, (uint8_t)(tok->team > 0.5f ? 1 : 0)); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Unit(tok->score)); pos += 1;
			QNN_BufWriteF16(out, pos, tok->recency); pos += 2;
		}
		break;

	case QNN_TOKEN_ITEM:
		{
			const qnn_item_token_t *tok = &tt->item;
			out[pos++] = (uint8_t)tok->subject_id;
			out[pos++] = (uint8_t)tok->modality_id;
			pos = QNN_PackEventSlots(out, pos, tok->events, tok->event_count);
			/* Per-type (24 B): half u8×3, rel i16×3, path i16×3,
			 * path_dist u16, eta f16, amount u8, regen f16, recency f16 */
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[0], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[1], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[2], 32767.0f) : 0); pos += 2;
			QNN_BufWriteU16(out, pos, paths ? QNN_QuantizeU16Saturating(tok->path_dist) : 0); pos += 2;
			QNN_BufWriteF16(out, pos, paths ? tok->eta : 0.0f); pos += 2;
			/* Raw engine pickup amount as u8 saturating; model
			 * applies per-subject normalization via
			 * qnn.engine_norm.ITEM_AMOUNT_MULT/CONST. */
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->amount)); pos += 1;
			QNN_BufWriteF16(out, pos, tok->regen); pos += 2;
			QNN_BufWriteF16(out, pos, tok->recency); pos += 2;
		}
		break;

	case QNN_TOKEN_MOVER:
		{
			const qnn_mover_token_t *tok = &tt->mover;
			out[pos++] = (uint8_t)tok->subject_id;
			out[pos++] = (uint8_t)tok->modality_id;
			pos = QNN_PackEventSlots(out, pos, tok->events, tok->event_count);
			/* Per-type (22 B): half u8×3, rel i16×3, path i16×3,
			 * path_dist u16, eta f16, state u8, recency f16 */
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[0], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[1], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[2], 32767.0f) : 0); pos += 2;
			QNN_BufWriteU16(out, pos, paths ? QNN_QuantizeU16Saturating(tok->path_dist) : 0); pos += 2;
			QNN_BufWriteF16(out, pos, paths ? tok->eta : 0.0f); pos += 2;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Unit(tok->state)); pos += 1;
			QNN_BufWriteF16(out, pos, tok->recency); pos += 2;
		}
		break;

	default:
		QNN_ObsFatal("entities v1: unrecognized token type %d",
			(int)tt->type);
	}
	return pos;
}

/* Policy v3 (COMBAT) — actor/projectile only, no recency.  Ported
 * verbatim from bb27a296's QNN_IOPackObsBuffer (the last commit where
 * the native emit still wrote this shape).  QNN_QualifyCombatEntity
 * (qnn_oracle.c) only ever qualifies ACTOR/PROJECTILE candidates for
 * COMBAT mode, so item/mover reaching here is a programming error
 * upstream — fail loud rather than emit a silently-wrong row. */
static int QNN_ObsEmitEntityRowV3(uint8_t *out, int pos,
	const qnn_tagged_token_t *tt, int paths)
{
	switch (tt->type)
	{
	case QNN_TOKEN_PROJECTILE:
		{
			const qnn_projectile_token_t *tok = &tt->projectile;
			out[pos++] = (uint8_t)tok->subject_id;
			out[pos++] = (uint8_t)tok->modality_id;
			pos = QNN_PackEventSlots(out, pos, tok->events, tok->event_count);
			/* Per-type scalars (12 B): rel i16×3, vel i16×3 — no recency. */
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
		}
		break;

	case QNN_TOKEN_ACTOR:
		{
			const qnn_actor_token_t *tok = &tt->actor;
			out[pos++] = (uint8_t)tok->subject_id;
			out[pos++] = (uint8_t)tok->modality_id;
			out[pos++] = (uint8_t)tok->player_id;
			pos = QNN_PackEventSlots(out, pos, tok->events, tok->event_count);
			/* Per-type (28 B): half u8×3, rel i16×3, vel i16×3, path i16×3,
			 * path_dist u16, eta f16, facing u8, team u8, score u8 — no
			 * recency.  paths=false zeroes the path-derived fields (the
			 * flag's real effect is compute-side: no pathfinder queries
			 * for this seat), same as the v1 writer. */
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[0], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[1], 32767.0f) : 0); pos += 2;
			QNN_BufWriteI16(out, pos, paths ? QNN_QuantizeI16Clamped(tok->path[2], 32767.0f) : 0); pos += 2;
			QNN_BufWriteU16(out, pos, paths ? QNN_QuantizeU16Saturating(tok->path_dist) : 0); pos += 2;
			QNN_BufWriteF16(out, pos, paths ? tok->eta : 0.0f); pos += 2;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Unit(tok->facing)); pos += 1;
			QNN_BufWriteU8 (out, pos, (uint8_t)(tok->team > 0.5f ? 1 : 0)); pos += 1;
			QNN_BufWriteU8 (out, pos, QNN_QuantizeU8Unit(tok->score)); pos += 1;
		}
		break;

	default:
		QNN_ObsFatal("entities v3: unexpected token type %d (the combat "
			"policy only carries actor/projectile — QNN_QualifyCombatEntity "
			"in qnn_oracle.c should never qualify anything else)",
			(int)tt->type);
	}
	return pos;
}

static int QNN_ObsEmitEntities(const qnn_obs_seat_ctx_t *ctx,
	const qnn_obs_params_t *params, uint8_t *out)
{
	const qnn_tick_result_t *r = ctx->result;
	/* paths=false keeps the row layout but zeroes the path-derived
	 * fields (the flag's real effect is compute-side: no pathfinder
	 * queries for this seat). */
	int paths = params->entities.paths;
	qboolean v3 = strcmp(params->entities.policy, "v3") == 0;
	int i;
	int pos;
	int pack_count = r->entity_count < params->entities.max_tokens
		? r->entity_count : params->entities.max_tokens;

	pos = 0;
	out[pos++] = (uint8_t)pack_count;

	for (i = 0; i < pack_count; ++i)
	{
		const qnn_tagged_token_t *tt = &r->entities[i];
		out[pos++] = (uint8_t)tt->type;
		pos = v3 ? QNN_ObsEmitEntityRowV3(out, pos, tt, paths)
			: QNN_ObsEmitEntityRowV1(out, pos, tt, paths);
	}
	return pos;
}

/* ── The registry table ───────────────────────────────────────────
 * Row order for the 13 state fields is the canonical self-block wire
 * order — QNN_ObsDeclDefault requests them in table order. */

static const qnn_obs_registry_entry_t qnn_obs_registry[] = {
	{ "health",           QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitHealth },
	{ "effective_armor",  QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitEffectiveArmor },
	{ "ammo_shells",      QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitAmmoShells },
	{ "ammo_nails",       QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitAmmoNails },
	{ "ammo_rockets",     QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitAmmoRockets },
	{ "ammo_cells",       QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitAmmoCells },
	{ "vel",              QNN_OBS_KIND_STATE, "(no params)", "int16",
	  NULL, QNN_ObsShapeVec3,   QNN_ObsSizeBytes6, QNN_ObsEmitVel },
	{ "attack_finished",  QNN_OBS_KIND_STATE, "(no params)", "float16",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes2, QNN_ObsEmitAttackFinished },
	{ "self_weapon_id",   QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitWeaponId },
	{ "self_movement_id", QNN_OBS_KIND_STATE, "(no params)", "uint8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitMovementId },
	/* self_items is written as a raw u32 bitfield; the python mirror
	 * has always parsed it as int32 (same bytes), and the dtype here
	 * is the CONSUMER name quoted in layout replies. */
	{ "self_items",       QNN_OBS_KIND_STATE, "(no params)", "int32",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes4, QNN_ObsEmitItems },
	{ "view_pitch",       QNN_OBS_KIND_STATE, "(no params)", "int8",
	  NULL, QNN_ObsShapeScalar, QNN_ObsSizeBytes1, QNN_ObsEmitViewPitch },
	{ "look_delta",       QNN_OBS_KIND_STATE, "(no params)", "float16",
	  NULL, QNN_ObsShapeVec3,   QNN_ObsSizeBytes6, QNN_ObsEmitLookDelta },
	{ "atlas",            QNN_OBS_KIND_SENSOR,
	  "{yaw in {24, 72}, bands: 11, packed: bool} — supported: "
	  "(24, 11, packed), (72, 11, unpacked)", "uint8",
	  QNN_ObsValidateAtlasParams, QNN_ObsShapeAtlas,
	  QNN_ObsSizeAtlas, QNN_ObsEmitAtlas },
	{ "entities",         QNN_OBS_KIND_PERCEPT,
	  "{policy: \"v1\"|\"v3\", max_tokens: 1..16, paths: bool}", NULL,
	  QNN_ObsValidateEntityParams, QNN_ObsShapeEntities,
	  QNN_ObsSizeEntities, QNN_ObsEmitEntities },
};

#define QNN_OBS_REGISTRY_COUNT \
	((int)(sizeof(qnn_obs_registry) / sizeof(qnn_obs_registry[0])))

int QNN_ObsRegistryCount(void)
{
	return QNN_OBS_REGISTRY_COUNT;
}

const qnn_obs_registry_entry_t *QNN_ObsRegistryAt(int index)
{
	if (index < 0 || index >= QNN_OBS_REGISTRY_COUNT)
		return NULL;
	return &qnn_obs_registry[index];
}

const qnn_obs_registry_entry_t *QNN_ObsRegistryFind(const char *name)
{
	int i;

	for (i = 0; i < QNN_OBS_REGISTRY_COUNT; ++i)
		if (strcmp(qnn_obs_registry[i].name, name) == 0)
			return &qnn_obs_registry[i];
	return NULL;
}

/* ── The default declaration ────────────────────────────────────── */

void QNN_ObsDeclDefault(qnn_obs_decl_t *out)
{
	int i;
	int state_count = 0;

	memset(out, 0, sizeof(*out));
	out->obs_api = QNN_OBS_API_VERSION;

	/* All state rows in table (= wire) order. */
	for (i = 0; i < QNN_OBS_REGISTRY_COUNT; ++i)
	{
		if (qnn_obs_registry[i].kind != QNN_OBS_KIND_STATE)
			continue;
		if (state_count >= QNN_OBS_MAX_STATE_FIELDS)
			QNN_ObsFatal("default decl overflows state capacity");
		strncpy(out->state[state_count], qnn_obs_registry[i].name,
			QNN_OBS_MAX_FIELD_NAME - 1);
		state_count++;
	}
	out->state_count = state_count;

	out->atlas_requested = true;
	out->atlas.yaw = QNN_OBS_ATLAS_YAWS;
	out->atlas.bands = QNN_OBS_ATLAS_ELEVS;
	out->atlas.packed = true;

	out->entities_requested = true;
	strncpy(out->entities.policy, "v3", QNN_OBS_MAX_POLICY_NAME - 1);
	out->entities.max_tokens = QNN_MAX_TOKEN_OBJECTS;
	out->entities.paths = true;
}

/* ── The emit-plan compiler ─────────────────────────────────────── */

static qboolean QNN_ObsPlanAppend(qnn_obs_plan_t *plan,
	const qnn_obs_registry_entry_t *entry, const qnn_obs_params_t *params,
	char *error, size_t error_size)
{
	qnn_obs_plan_step_t *step;

	if (plan->step_count >= QNN_OBS_MAX_PLAN_STEPS)
	{
		snprintf(error, error_size,
			"plan overflow at \"%s\" (max %d steps)",
			entry->name, QNN_OBS_MAX_PLAN_STEPS);
		return false;
	}
	if (entry->validate_params != NULL)
	{
		if (!entry->validate_params(params, error, error_size))
			return false;
	}

	step = &plan->steps[plan->step_count++];
	step->entry = entry;
	if (params != NULL)
		step->params = *params;
	else
		memset(&step->params, 0, sizeof(step->params));
	step->offset = plan->payload_bytes;
	step->bytes = entry->size_fn(&step->params);
	if (step->bytes <= 0)
		QNN_ObsFatal("registry row \"%s\" reports non-positive size %d",
			entry->name, step->bytes);
	plan->payload_bytes += step->bytes;
	return true;
}

qboolean QNN_ObsPlanCompile(const qnn_obs_decl_t *decl, qnn_obs_plan_t *plan,
	char *error, size_t error_size)
{
	int i, j;

	memset(plan, 0, sizeof(*plan));

	if (decl->obs_api != QNN_OBS_API_VERSION)
	{
		snprintf(error, error_size,
			"obs_api %d not supported (this engine speaks %d)",
			decl->obs_api, QNN_OBS_API_VERSION);
		return false;
	}
	if (decl->state_count < 0 || decl->state_count > QNN_OBS_MAX_STATE_FIELDS)
	{
		snprintf(error, error_size,
			"state list length %d out of range (0..%d)",
			decl->state_count, QNN_OBS_MAX_STATE_FIELDS);
		return false;
	}

	/* State fields first, in declaration order. */
	for (i = 0; i < decl->state_count; ++i)
	{
		const qnn_obs_registry_entry_t *entry =
			QNN_ObsRegistryFind(decl->state[i]);

		if (entry == NULL)
		{
			snprintf(error, error_size,
				"unknown obs field \"%s\"", decl->state[i]);
			return false;
		}
		if (entry->kind != QNN_OBS_KIND_STATE)
		{
			snprintf(error, error_size,
				"\"%s\" is not a state field — request it via its "
				"own declaration key", entry->name);
			return false;
		}
		for (j = 0; j < i; ++j)
		{
			if (strcmp(decl->state[j], decl->state[i]) == 0)
			{
				snprintf(error, error_size,
					"duplicate state field \"%s\"", decl->state[i]);
				return false;
			}
		}
		if (!QNN_ObsPlanAppend(plan, entry, NULL, error, error_size))
			return false;
		plan->wants_state = true;
	}

	/* Sensor: the depth atlas. */
	if (decl->atlas_requested)
	{
		const qnn_obs_registry_entry_t *entry = QNN_ObsRegistryFind("atlas");
		qnn_obs_params_t params;

		if (entry == NULL)
			QNN_ObsFatal("registry is missing the atlas row");
		memset(&params, 0, sizeof(params));
		params.atlas = decl->atlas;
		if (!QNN_ObsPlanAppend(plan, entry, &params, error, error_size))
			return false;
		plan->wants_atlas = true;
		plan->atlas = decl->atlas;
	}

	/* Percept: the entity stream. */
	if (decl->entities_requested)
	{
		const qnn_obs_registry_entry_t *entry = QNN_ObsRegistryFind("entities");
		qnn_obs_params_t params;

		if (entry == NULL)
			QNN_ObsFatal("registry is missing the entities row");
		memset(&params, 0, sizeof(params));
		params.entities = decl->entities;
		if (!QNN_ObsPlanAppend(plan, entry, &params, error, error_size))
			return false;
		plan->wants_entities = true;
		plan->entities = decl->entities;
	}

	if (plan->step_count == 0)
	{
		snprintf(error, error_size, "declaration requests no fields");
		return false;
	}

	plan->frame_bytes = plan->payload_bytes + QNN_OBS_POSE_TAIL_BYTES;
	if (decl->frame_bytes_override != 0)
	{
		if (decl->frame_bytes_override < plan->frame_bytes)
		{
			snprintf(error, error_size,
				"frame_bytes override %d < compiled minimum %d "
				"(payload %d + pose tail %d)",
				decl->frame_bytes_override, plan->frame_bytes,
				plan->payload_bytes, QNN_OBS_POSE_TAIL_BYTES);
			return false;
		}
		if (decl->frame_bytes_override > QNN_OBS_FRAME_BYTES_MAX)
		{
			snprintf(error, error_size,
				"frame_bytes override %d exceeds the %d-byte ceiling",
				decl->frame_bytes_override, QNN_OBS_FRAME_BYTES_MAX);
			return false;
		}
		plan->frame_bytes = decl->frame_bytes_override;
	}
	return true;
}

/* ── Plan-driven compute stage ──────────────────────────────────── */

void QNN_ObsPlanCompute(const qnn_obs_plan_t *plan,
	const qnn_obs_compute_fns_t *fns, const qnn_snapshot_t *snapshot,
	qnn_tick_result_t *out)
{
	memset(out, 0, sizeof(*out));

	/* A skipped / never-demanded atlas must read all-miss, not
	 * hit-at-zero, so prefill the scratch with the miss code. */
	memset(out->spatial_atlas, QNN_OBS_ATLAS_MISS_CODE,
		sizeof(out->spatial_atlas));

	/* Demand-driven: run a provider ONLY for kinds this plan requests.
	 * Provider order matches the pre-plan QNN_IOEmit (entities, self,
	 * spatial). */
	if (plan->wants_entities)
	{
		if (fns->entities == NULL)
			QNN_ObsFatal("plan demands entities but no provider is wired");
		out->entity_count = fns->entities(snapshot, &plan->entities,
			out->entities, plan->entities.max_tokens);
	}
	if (plan->wants_state)
	{
		if (fns->self == NULL)
			QNN_ObsFatal("plan demands state but no provider is wired");
		fns->self(snapshot, &out->self);
	}
	if (plan->wants_atlas)
	{
		if (fns->atlas == NULL)
			QNN_ObsFatal("plan demands the atlas but no provider is wired");
		fns->atlas(snapshot, &plan->atlas, out->spatial_atlas);
	}
}

/* ── Plan-driven serialization stage ────────────────────────────── */

void QNN_ObsPlanPack(const qnn_obs_plan_t *plan, uint8_t *obs,
	int obs_bytes, const qnn_tick_result_t *result)
{
	qnn_obs_seat_ctx_t ctx;
	int i;

	if (obs_bytes < plan->frame_bytes)
		QNN_ObsFatal("obs buffer %d B < plan frame %d B",
			obs_bytes, plan->frame_bytes);

	memset(obs, 0, (size_t)obs_bytes);
	ctx.snapshot = NULL;
	ctx.result = result;

	for (i = 0; i < plan->step_count; ++i)
	{
		const qnn_obs_plan_step_t *step = &plan->steps[i];
		int wrote = step->entry->emit_fn(&ctx, &step->params,
			obs + step->offset);

		if (wrote < 0 || wrote > step->bytes)
			QNN_ObsFatal("\"%s\" wrote %d B into a %d B slot",
				step->entry->name, wrote, step->bytes);
	}
}
