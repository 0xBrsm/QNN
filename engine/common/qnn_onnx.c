/*
 * qnn_onnx.c — ORT session lifecycle + per-tick inference + action
 * decode for the live NQ client. Direct from engine types
 * (qnn_tick_result_t in, qnn_action_t out) — no parallel obs/action
 * structs, no transcoder.
 *
 * Native-width policy: scratch buffers carry the same native dtypes as
 * the wire format (see qnn_io.h header comment / src/qnn/engine_norm.py).
 * The ONNX model's input dtypes match these, and the model's own
 * dequantizer modules (qnn.model.dequant) reproduce the normalization
 * the C side used to do inline.
 *
 * Sections:
 *   - Thread-local error buffer
 *   - Context struct + lifecycle (Init / Reset / Free)
 *   - Tick-result → ORT input scratch packing
 *   - Session run + output extraction
 *   - Action decode (move argmax, look clamp, fire sigmoid, sticky weapon)
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "onnxruntime_c_api.h"

#include "qnn.h"
#include "qnn_io.h"
#include "qnn_object.h"
#include "qnn_onnx.h"


/* ── Schema (mirrors the engine_norm wire layout) ───────────────── */

#define QNN_ONNX_GRU_HIDDEN_DIM         64
#define QNN_ONNX_WEAPON_CLASSES          8

/* Sticky-weapon controller thresholds — v23 ModelConfig values. */
#define QNN_ONNX_WEAPON_SWITCH_CONFIDENCE  0.65f
#define QNN_ONNX_WEAPON_SWITCH_MARGIN      0.15f

/* Per-frame ONNX input count: 10 self + 11 spatial + 17 entity + 1 hidden = 39.
 *
 * Self (10): health, effective_armor, ammo_shells, ammo_nails,
 *            ammo_rockets, ammo_cells, vel, attack_finished,
 *            self_weapon_id, self_movement_id, self_items.
 *            Actually 11 (vel counted once). Recount: health 1 +
 *            armor 1 + 4 ammos + vel 1 + af 1 + weapon_id 1 +
 *            movement_id 1 + items 1 = 11.
 * Spatial (11): dir, nearest_dist, mean_dist, openness, clearance,
 *               traversable, dropoff, solid_frac, water_frac,
 *               slime_frac, lava_frac.
 * Entity (17): types, subject_id, modality_id, player_id,
 *              event_count, event_actions, event_sources,
 *              half_extents, rel, vel, path, path_dist,
 *              eta, recency, facing, team, score, amount,
 *              regen, state. Recount: types 1 + subject 1 +
 *              modality 1 + player 1 + count 1 + actions 1 +
 *              sources 1 + half 1 + rel 1 + vel 1 + path 1 +
 *              path_dist 1 + eta 1 + recency 1 + facing 1 +
 *              team 1 + score 1 + amount 1 + regen 1 + state 1
 *              = 20.
 * Hidden (1): hidden.
 *
 * Total: 11 + 11 + 20 + 1 = 43. */
#define QNN_ONNX_N_OBS_INPUTS  43
#define QNN_ONNX_N_INPUTS      (QNN_ONNX_N_OBS_INPUTS + 1)
#define QNN_ONNX_N_OUTPUTS      5


/* ── Thread-local error buffer ──────────────────────────────────── */

static __thread char qnn_onnx_error[512] = "";

static void qnn_onnx_set_error(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(qnn_onnx_error, sizeof(qnn_onnx_error), fmt, ap);
	va_end(ap);
}

static int qnn_onnx_set_error_from_ort(const OrtApi *ort, OrtStatus *status, const char *where)
{
	const char *msg;

	if (status == NULL)
		return 0;
	msg = ort->GetErrorMessage(status);
	qnn_onnx_set_error("%s: %s", where, msg ? msg : "(no message)");
	ort->ReleaseStatus(status);
	return 1;
}

const char *QNN_OnnxLastError(void)
{
	return qnn_onnx_error;
}


/* ── Context struct (native-width scratch buffers) ─────────────── */

struct qnn_onnx_ctx
{
	const OrtApi       *ort;
	OrtEnv             *env;
	OrtSessionOptions  *opts;
	OrtSession         *session;
	OrtMemoryInfo      *meminfo;

	/* GRU hidden state, carried across steps. */
	float    hidden[QNN_ONNX_GRU_HIDDEN_DIM];

	/* ── Self block scratch (per-field native widths) ──────── */
	uint8_t  self_health;
	uint8_t  self_effective_armor;
	uint8_t  self_ammo_shells;
	uint8_t  self_ammo_nails;
	uint8_t  self_ammo_rockets;
	uint8_t  self_ammo_cells;
	int16_t  self_vel[3];
	uint16_t self_attack_finished;     /* binary16 bit pattern */
	uint8_t  self_weapon_id;
	uint8_t  self_movement_id;
	int32_t  self_items;
	int8_t   self_view_pitch;

	/* ── Spatial block scratch (B=1, N=9, per-field native) ── */
	int8_t   spatial_dir          [QNN_SPATIAL_TOKEN_COUNT][3];
	uint16_t spatial_nearest_dist [QNN_SPATIAL_TOKEN_COUNT];
	uint16_t spatial_mean_dist    [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_openness     [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_clearance    [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_traversable  [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_dropoff      [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_solid_frac   [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_water_frac   [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_slime_frac   [QNN_SPATIAL_TOKEN_COUNT];
	uint8_t  spatial_lava_frac    [QNN_SPATIAL_TOKEN_COUNT];

	/* ── Entity block scratch (B=1, N=QNN_MAX_TOKEN_OBJECTS) ── */
	int8_t   entity_types          [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_subject_id     [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_modality_id    [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_player_id      [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_event_count    [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_event_actions  [QNN_MAX_TOKEN_OBJECTS][QNN_MAX_ENTITY_EVENTS];
	uint8_t  entity_event_sources  [QNN_MAX_TOKEN_OBJECTS][QNN_MAX_ENTITY_EVENTS];
	uint8_t  entity_half_extents   [QNN_MAX_TOKEN_OBJECTS][3];
	int16_t  entity_rel            [QNN_MAX_TOKEN_OBJECTS][3];
	int16_t  entity_vel            [QNN_MAX_TOKEN_OBJECTS][3];
	int16_t  entity_path           [QNN_MAX_TOKEN_OBJECTS][3];
	uint16_t entity_path_dist      [QNN_MAX_TOKEN_OBJECTS];
	uint16_t entity_eta            [QNN_MAX_TOKEN_OBJECTS];    /* f16 */
	uint16_t entity_recency        [QNN_MAX_TOKEN_OBJECTS];    /* f16 */
	uint8_t  entity_facing         [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_team           [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_score          [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_amount         [QNN_MAX_TOKEN_OBJECTS];
	uint16_t entity_regen          [QNN_MAX_TOKEN_OBJECTS];    /* f16 */
	uint8_t  entity_state          [QNN_MAX_TOKEN_OBJECTS];

	/* Output scratch. */
	float    move_logits[3 * 3];
	float    look[3];
	float    fire_logit;
	float    weapon_logits[QNN_ONNX_WEAPON_CLASSES];
	float    next_hidden[QNN_ONNX_GRU_HIDDEN_DIM];
};


/* ── Lifecycle ──────────────────────────────────────────────────── */

#define QNN_ONNX_CHECK_OR_FAIL(call, where)                                 \
	do {                                                                \
		OrtStatus *_s = (call);                                     \
		if (qnn_onnx_set_error_from_ort(ort, _s, (where)))          \
			goto fail;                                          \
	} while (0)

qnn_onnx_ctx_t *QNN_OnnxInit(const char *onnx_path)
{
	qnn_onnx_ctx_t *ctx;
	const OrtApi *ort;

	if (onnx_path == NULL || onnx_path[0] == '\0') {
		qnn_onnx_set_error("QNN_OnnxInit: onnx_path is NULL or empty");
		return NULL;
	}

	ctx = (qnn_onnx_ctx_t *)calloc(1, sizeof(*ctx));
	if (ctx == NULL) {
		qnn_onnx_set_error("QNN_OnnxInit: out of memory");
		return NULL;
	}

	ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) {
		qnn_onnx_set_error("QNN_OnnxInit: OrtGetApiBase returned NULL (ABI mismatch?)");
		free(ctx);
		return NULL;
	}
	ctx->ort = ort;

	QNN_ONNX_CHECK_OR_FAIL(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qnn_onnx", &ctx->env), "CreateEnv");
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateSessionOptions(&ctx->opts), "CreateSessionOptions");
	QNN_ONNX_CHECK_OR_FAIL(ort->SetIntraOpNumThreads(ctx->opts, 1), "SetIntraOpNumThreads");
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateSession(ctx->env, onnx_path, ctx->opts, &ctx->session), "CreateSession");
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->meminfo), "CreateCpuMemoryInfo");

	QNN_OnnxReset(ctx);
	return ctx;

fail:
	QNN_OnnxFree(ctx);
	return NULL;
}

void QNN_OnnxReset(qnn_onnx_ctx_t *ctx)
{
	if (ctx == NULL) return;
	memset(ctx->hidden, 0, sizeof(ctx->hidden));
}

void QNN_OnnxFree(qnn_onnx_ctx_t *ctx)
{
	if (ctx == NULL) return;
	if (ctx->ort != NULL) {
		if (ctx->meminfo) ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
		if (ctx->session) ctx->ort->ReleaseSession(ctx->session);
		if (ctx->opts)    ctx->ort->ReleaseSessionOptions(ctx->opts);
		if (ctx->env)     ctx->ort->ReleaseEnv(ctx->env);
	}
	free(ctx);
}


/* ── Pack: qnn_tick_result_t → ORT scratch buffers ──────────────── */

/* Reuse the wire packer quantizers so both paths emit identical
 * native values from the same float internals. */

static void emit_actor(qnn_onnx_ctx_t *ctx, const qnn_actor_token_t *src, int slot)
{
	int j, n_evt;

	ctx->entity_types        [slot] = (int8_t)QNN_TOKEN_ACTOR;
	ctx->entity_subject_id   [slot] = (uint8_t)src->subject_id;
	ctx->entity_modality_id  [slot] = (uint8_t)src->modality_id;
	ctx->entity_player_id    [slot] = (uint8_t)src->player_id;

	ctx->entity_half_extents [slot][0] = QNN_QuantizeU8Saturating(src->half_extents[0]);
	ctx->entity_half_extents [slot][1] = QNN_QuantizeU8Saturating(src->half_extents[1]);
	ctx->entity_half_extents [slot][2] = QNN_QuantizeU8Saturating(src->half_extents[2]);
	ctx->entity_rel          [slot][0] = QNN_QuantizeI16Clamped(src->rel[0], 32767.0f);
	ctx->entity_rel          [slot][1] = QNN_QuantizeI16Clamped(src->rel[1], 32767.0f);
	ctx->entity_rel          [slot][2] = QNN_QuantizeI16Clamped(src->rel[2], 32767.0f);
	ctx->entity_vel          [slot][0] = QNN_QuantizeI16Clamped(src->vel[0], QNN_VELOCITY_SCALE);
	ctx->entity_vel          [slot][1] = QNN_QuantizeI16Clamped(src->vel[1], QNN_VELOCITY_SCALE);
	ctx->entity_vel          [slot][2] = QNN_QuantizeI16Clamped(src->vel[2], QNN_VELOCITY_SCALE);
	ctx->entity_path         [slot][0] = QNN_QuantizeI16Clamped(src->path[0], 32767.0f);
	ctx->entity_path         [slot][1] = QNN_QuantizeI16Clamped(src->path[1], 32767.0f);
	ctx->entity_path         [slot][2] = QNN_QuantizeI16Clamped(src->path[2], 32767.0f);
	ctx->entity_path_dist    [slot]    = QNN_QuantizeU16Saturating(src->path_dist);
	ctx->entity_eta          [slot]    = QNN_FloatToHalf(src->eta);
	ctx->entity_facing       [slot]    = QNN_QuantizeU8Unit(src->facing);
	ctx->entity_team         [slot]    = (uint8_t)(src->team > 0.5f ? 1 : 0);
	ctx->entity_score        [slot]    = QNN_QuantizeU8Unit(src->score);
	ctx->entity_recency      [slot]    = QNN_FloatToHalf(src->recency);

	n_evt = src->event_count < QNN_MAX_ENTITY_EVENTS ? src->event_count : QNN_MAX_ENTITY_EVENTS;
	ctx->entity_event_count[slot] = (uint8_t)n_evt;
	for (j = 0; j < n_evt; ++j) {
		ctx->entity_event_actions[slot][j] = (uint8_t)src->events[j].action_id;
		ctx->entity_event_sources[slot][j] = (uint8_t)src->events[j].source_id;
	}
}

static void emit_projectile(qnn_onnx_ctx_t *ctx, const qnn_projectile_token_t *src, int slot)
{
	int j, n_evt;

	ctx->entity_types        [slot] = (int8_t)QNN_TOKEN_PROJECTILE;
	ctx->entity_subject_id   [slot] = (uint8_t)src->subject_id;
	ctx->entity_modality_id  [slot] = (uint8_t)src->modality_id;
	/* projectile has no player_id; slot stays at zero. */

	ctx->entity_rel          [slot][0] = QNN_QuantizeI16Clamped(src->rel[0], 32767.0f);
	ctx->entity_rel          [slot][1] = QNN_QuantizeI16Clamped(src->rel[1], 32767.0f);
	ctx->entity_rel          [slot][2] = QNN_QuantizeI16Clamped(src->rel[2], 32767.0f);
	ctx->entity_vel          [slot][0] = QNN_QuantizeI16Clamped(src->vel[0], QNN_VELOCITY_SCALE);
	ctx->entity_vel          [slot][1] = QNN_QuantizeI16Clamped(src->vel[1], QNN_VELOCITY_SCALE);
	ctx->entity_vel          [slot][2] = QNN_QuantizeI16Clamped(src->vel[2], QNN_VELOCITY_SCALE);
	ctx->entity_recency      [slot]    = QNN_FloatToHalf(src->recency);

	n_evt = src->event_count < QNN_MAX_ENTITY_EVENTS ? src->event_count : QNN_MAX_ENTITY_EVENTS;
	ctx->entity_event_count[slot] = (uint8_t)n_evt;
	for (j = 0; j < n_evt; ++j) {
		ctx->entity_event_actions[slot][j] = (uint8_t)src->events[j].action_id;
		ctx->entity_event_sources[slot][j] = (uint8_t)src->events[j].source_id;
	}
}

static void emit_item(qnn_onnx_ctx_t *ctx, const qnn_item_token_t *src, int slot)
{
	int j, n_evt;

	ctx->entity_types        [slot] = (int8_t)QNN_TOKEN_ITEM;
	ctx->entity_subject_id   [slot] = (uint8_t)src->subject_id;
	ctx->entity_modality_id  [slot] = (uint8_t)src->modality_id;

	ctx->entity_half_extents [slot][0] = QNN_QuantizeU8Saturating(src->half_extents[0]);
	ctx->entity_half_extents [slot][1] = QNN_QuantizeU8Saturating(src->half_extents[1]);
	ctx->entity_half_extents [slot][2] = QNN_QuantizeU8Saturating(src->half_extents[2]);
	ctx->entity_rel          [slot][0] = QNN_QuantizeI16Clamped(src->rel[0], 32767.0f);
	ctx->entity_rel          [slot][1] = QNN_QuantizeI16Clamped(src->rel[1], 32767.0f);
	ctx->entity_rel          [slot][2] = QNN_QuantizeI16Clamped(src->rel[2], 32767.0f);
	ctx->entity_path         [slot][0] = QNN_QuantizeI16Clamped(src->path[0], 32767.0f);
	ctx->entity_path         [slot][1] = QNN_QuantizeI16Clamped(src->path[1], 32767.0f);
	ctx->entity_path         [slot][2] = QNN_QuantizeI16Clamped(src->path[2], 32767.0f);
	ctx->entity_path_dist    [slot]    = QNN_QuantizeU16Saturating(src->path_dist);
	ctx->entity_eta          [slot]    = QNN_FloatToHalf(src->eta);
	/* Raw engine pickup amount as u8 saturating; model normalizes
	 * per subject via qnn.engine_norm.ITEM_AMOUNT_MULT/CONST. */
	ctx->entity_amount       [slot]    = QNN_QuantizeU8Saturating(src->amount);
	ctx->entity_regen        [slot]    = QNN_FloatToHalf(src->regen);
	ctx->entity_recency      [slot]    = QNN_FloatToHalf(src->recency);

	n_evt = src->event_count < QNN_MAX_ENTITY_EVENTS ? src->event_count : QNN_MAX_ENTITY_EVENTS;
	ctx->entity_event_count[slot] = (uint8_t)n_evt;
	for (j = 0; j < n_evt; ++j) {
		ctx->entity_event_actions[slot][j] = (uint8_t)src->events[j].action_id;
		ctx->entity_event_sources[slot][j] = (uint8_t)src->events[j].source_id;
	}
}

static void emit_mover(qnn_onnx_ctx_t *ctx, const qnn_mover_token_t *src, int slot)
{
	int j, n_evt;

	ctx->entity_types        [slot] = (int8_t)QNN_TOKEN_MOVER;
	ctx->entity_subject_id   [slot] = (uint8_t)src->subject_id;
	ctx->entity_modality_id  [slot] = (uint8_t)src->modality_id;

	ctx->entity_half_extents [slot][0] = QNN_QuantizeU8Saturating(src->half_extents[0]);
	ctx->entity_half_extents [slot][1] = QNN_QuantizeU8Saturating(src->half_extents[1]);
	ctx->entity_half_extents [slot][2] = QNN_QuantizeU8Saturating(src->half_extents[2]);
	ctx->entity_rel          [slot][0] = QNN_QuantizeI16Clamped(src->rel[0], 32767.0f);
	ctx->entity_rel          [slot][1] = QNN_QuantizeI16Clamped(src->rel[1], 32767.0f);
	ctx->entity_rel          [slot][2] = QNN_QuantizeI16Clamped(src->rel[2], 32767.0f);
	ctx->entity_path         [slot][0] = QNN_QuantizeI16Clamped(src->path[0], 32767.0f);
	ctx->entity_path         [slot][1] = QNN_QuantizeI16Clamped(src->path[1], 32767.0f);
	ctx->entity_path         [slot][2] = QNN_QuantizeI16Clamped(src->path[2], 32767.0f);
	ctx->entity_path_dist    [slot]    = QNN_QuantizeU16Saturating(src->path_dist);
	ctx->entity_eta          [slot]    = QNN_FloatToHalf(src->eta);
	ctx->entity_state        [slot]    = QNN_QuantizeU8Unit(src->state);
	ctx->entity_recency      [slot]    = QNN_FloatToHalf(src->recency);

	n_evt = src->event_count < QNN_MAX_ENTITY_EVENTS ? src->event_count : QNN_MAX_ENTITY_EVENTS;
	ctx->entity_event_count[slot] = (uint8_t)n_evt;
	for (j = 0; j < n_evt; ++j) {
		ctx->entity_event_actions[slot][j] = (uint8_t)src->events[j].action_id;
		ctx->entity_event_sources[slot][j] = (uint8_t)src->events[j].source_id;
	}
}

static void pack_scratch(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *r)
{
	const qnn_self_token_t *self;
	int i, n;
	float eff_armor;

	/* Wipe entity-stream scratch so previous-frame data doesn't bleed
	 * into empty slots; entity_types defaults to -1 (empty). */
	memset(ctx->entity_subject_id,    0, sizeof(ctx->entity_subject_id));
	memset(ctx->entity_modality_id,   0, sizeof(ctx->entity_modality_id));
	memset(ctx->entity_player_id,     0, sizeof(ctx->entity_player_id));
	memset(ctx->entity_event_count,   0, sizeof(ctx->entity_event_count));
	memset(ctx->entity_event_actions, 0, sizeof(ctx->entity_event_actions));
	memset(ctx->entity_event_sources, 0, sizeof(ctx->entity_event_sources));
	memset(ctx->entity_half_extents,  0, sizeof(ctx->entity_half_extents));
	memset(ctx->entity_rel,           0, sizeof(ctx->entity_rel));
	memset(ctx->entity_vel,           0, sizeof(ctx->entity_vel));
	memset(ctx->entity_path,          0, sizeof(ctx->entity_path));
	memset(ctx->entity_path_dist,     0, sizeof(ctx->entity_path_dist));
	memset(ctx->entity_eta,           0, sizeof(ctx->entity_eta));
	memset(ctx->entity_recency,       0, sizeof(ctx->entity_recency));
	memset(ctx->entity_facing,        0, sizeof(ctx->entity_facing));
	memset(ctx->entity_team,          0, sizeof(ctx->entity_team));
	memset(ctx->entity_score,         0, sizeof(ctx->entity_score));
	memset(ctx->entity_amount,        0, sizeof(ctx->entity_amount));
	memset(ctx->entity_regen,         0, sizeof(ctx->entity_regen));
	memset(ctx->entity_state,         0, sizeof(ctx->entity_state));
	for (i = 0; i < QNN_MAX_TOKEN_OBJECTS; ++i)
		ctx->entity_types[i] = -1;

	/* ---- Self block ---- */
	self = &r->self;
	eff_armor = (float)self->raw_armor * self->armor_type;

	ctx->self_health          = QNN_QuantizeU8Saturating((float)self->health);
	ctx->self_effective_armor = QNN_QuantizeU8Saturating(eff_armor);
	ctx->self_ammo_shells     = QNN_QuantizeU8Saturating((float)self->ammo_shells);
	ctx->self_ammo_nails      = QNN_QuantizeU8Saturating((float)self->ammo_nails);
	ctx->self_ammo_rockets    = QNN_QuantizeU8Saturating((float)self->ammo_rockets);
	ctx->self_ammo_cells      = QNN_QuantizeU8Saturating((float)self->ammo_cells);
	ctx->self_vel[0]          = QNN_QuantizeI16Clamped(self->vel[0], QNN_VELOCITY_SCALE);
	ctx->self_vel[1]          = QNN_QuantizeI16Clamped(self->vel[1], QNN_VELOCITY_SCALE);
	ctx->self_vel[2]          = QNN_QuantizeI16Clamped(self->vel[2], QNN_VELOCITY_SCALE);
	ctx->self_attack_finished = QNN_FloatToHalf(self->attack_finished);
	ctx->self_weapon_id       = (uint8_t)self->weapon_id;
	ctx->self_movement_id     = (uint8_t)self->movement_id;
	ctx->self_items           = self->items;
	ctx->self_view_pitch      = QNN_QuantizeI8(self->view_pitch);

	/* ---- Spatial block ---- */
	for (i = 0; i < QNN_SPATIAL_TOKEN_COUNT; ++i) {
		const qnn_spatial_token_t *t = &r->spatial[i];
		ctx->spatial_dir[i][0]      = QNN_QuantizeI8(t->dir[0]);
		ctx->spatial_dir[i][1]      = QNN_QuantizeI8(t->dir[1]);
		ctx->spatial_dir[i][2]      = QNN_QuantizeI8(t->dir[2]);
		ctx->spatial_nearest_dist[i] = QNN_QuantizeU16Saturating(t->nearest_dist);
		ctx->spatial_mean_dist[i]    = QNN_QuantizeU16Saturating(t->mean_dist);
		ctx->spatial_openness[i]    = QNN_QuantizeU8Unit(t->openness);
		ctx->spatial_clearance[i]   = QNN_QuantizeU8Unit(t->clearance);
		ctx->spatial_traversable[i] = QNN_QuantizeU8Unit(t->traversable);
		ctx->spatial_dropoff[i]     = QNN_QuantizeU8Unit(t->dropoff);
		ctx->spatial_solid_frac[i]  = QNN_QuantizeU8Unit(t->solid_frac);
		ctx->spatial_water_frac[i]  = QNN_QuantizeU8Unit(t->water_frac);
		ctx->spatial_slime_frac[i]  = QNN_QuantizeU8Unit(t->slime_frac);
		ctx->spatial_lava_frac[i]   = QNN_QuantizeU8Unit(t->lava_frac);
	}

	/* ---- Entities ---- */
	n = r->entity_count < QNN_MAX_TOKEN_OBJECTS ? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
	for (i = 0; i < n; ++i) {
		const qnn_tagged_token_t *tt = &r->entities[i];
		switch (tt->type) {
		case QNN_TOKEN_ACTOR:      emit_actor     (ctx, &tt->actor,      i); break;
		case QNN_TOKEN_PROJECTILE: emit_projectile(ctx, &tt->projectile, i); break;
		case QNN_TOKEN_ITEM:       emit_item      (ctx, &tt->item,       i); break;
		case QNN_TOKEN_MOVER:      emit_mover     (ctx, &tt->mover,      i); break;
		default: break;
		}
	}
}


/* ── ORT input table ────────────────────────────────────────────── */

typedef struct {
	const char *name;
	size_t      ctx_offset;       /* offsetof(qnn_onnx_ctx_t, field) */
	int64_t     shape[4];         /* with leading batch=1 */
	size_t      n_dims;
	size_t      byte_count;
	ONNXTensorElementDataType dtype;
} qnn_onnx_input_def_t;

#define _OFFS(field)     offsetof(qnn_onnx_ctx_t, field)
#define _SIZE_OF(field)  sizeof(((qnn_onnx_ctx_t *)0)->field)

static const qnn_onnx_input_def_t QNN_ONNX_INPUTS[QNN_ONNX_N_OBS_INPUTS] = {
	/* ── Self block (11 inputs) ───────────────────────────── */
	{ "self_health",          _OFFS(self_health),          {1}, 1, _SIZE_OF(self_health),          ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_effective_armor", _OFFS(self_effective_armor), {1}, 1, _SIZE_OF(self_effective_armor), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_ammo_shells",     _OFFS(self_ammo_shells),     {1}, 1, _SIZE_OF(self_ammo_shells),     ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_ammo_nails",      _OFFS(self_ammo_nails),      {1}, 1, _SIZE_OF(self_ammo_nails),      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_ammo_rockets",    _OFFS(self_ammo_rockets),    {1}, 1, _SIZE_OF(self_ammo_rockets),    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_ammo_cells",      _OFFS(self_ammo_cells),      {1}, 1, _SIZE_OF(self_ammo_cells),      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_vel",             _OFFS(self_vel),             {1, 3}, 2, _SIZE_OF(self_vel),          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 },
	{ "self_attack_finished", _OFFS(self_attack_finished), {1}, 1, _SIZE_OF(self_attack_finished), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 },
	{ "self_weapon_id",       _OFFS(self_weapon_id),       {1}, 1, _SIZE_OF(self_weapon_id),       ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_movement_id",     _OFFS(self_movement_id),     {1}, 1, _SIZE_OF(self_movement_id),     ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "self_items",           _OFFS(self_items),           {1}, 1, _SIZE_OF(self_items),           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 },
	{ "view_pitch",           _OFFS(self_view_pitch),      {1}, 1, _SIZE_OF(self_view_pitch),      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 },

	/* ── Spatial block (11 inputs, all shape (1, 9, …)) ───── */
	{ "spatial_dir",          _OFFS(spatial_dir),          {1, QNN_SPATIAL_TOKEN_COUNT, 3}, 3, _SIZE_OF(spatial_dir),          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 },
	{ "spatial_nearest_dist", _OFFS(spatial_nearest_dist), {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_nearest_dist), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 },
	{ "spatial_mean_dist",    _OFFS(spatial_mean_dist),    {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_mean_dist),    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 },
	{ "spatial_openness",     _OFFS(spatial_openness),     {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_openness),     ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_clearance",    _OFFS(spatial_clearance),    {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_clearance),    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_traversable",  _OFFS(spatial_traversable),  {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_traversable),  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_dropoff",      _OFFS(spatial_dropoff),      {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_dropoff),      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_solid_frac",   _OFFS(spatial_solid_frac),   {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_solid_frac),   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_water_frac",   _OFFS(spatial_water_frac),   {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_water_frac),   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_slime_frac",   _OFFS(spatial_slime_frac),   {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_slime_frac),   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "spatial_lava_frac",    _OFFS(spatial_lava_frac),    {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_lava_frac),    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },

	/* ── Entity block (20 inputs, all shape (1, N, …)) ────── */
	{ "entity_types",          _OFFS(entity_types),          {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_types),          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 },
	{ "entity_subject_id",     _OFFS(entity_subject_id),     {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_subject_id),     ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_modality_id",    _OFFS(entity_modality_id),    {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_modality_id),    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_player_id",      _OFFS(entity_player_id),      {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_player_id),      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_event_count",    _OFFS(entity_event_count),    {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_event_count),    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_event_actions",  _OFFS(entity_event_actions),  {1, QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS}, 3, _SIZE_OF(entity_event_actions),  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_event_sources",  _OFFS(entity_event_sources),  {1, QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS}, 3, _SIZE_OF(entity_event_sources),  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_half_extents",   _OFFS(entity_half_extents),   {1, QNN_MAX_TOKEN_OBJECTS, 3},                  3, _SIZE_OF(entity_half_extents),   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_rel",            _OFFS(entity_rel),            {1, QNN_MAX_TOKEN_OBJECTS, 3},                  3, _SIZE_OF(entity_rel),            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 },
	{ "entity_vel",            _OFFS(entity_vel),            {1, QNN_MAX_TOKEN_OBJECTS, 3},                  3, _SIZE_OF(entity_vel),            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 },
	{ "entity_path",           _OFFS(entity_path),           {1, QNN_MAX_TOKEN_OBJECTS, 3},                  3, _SIZE_OF(entity_path),           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 },
	{ "entity_path_dist",      _OFFS(entity_path_dist),      {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_path_dist),      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 },
	{ "entity_eta",            _OFFS(entity_eta),            {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_eta),            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 },
	{ "entity_recency",        _OFFS(entity_recency),        {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_recency),        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 },
	{ "entity_facing",         _OFFS(entity_facing),         {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_facing),         ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_team",           _OFFS(entity_team),           {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_team),           ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_score",          _OFFS(entity_score),          {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_score),          ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_amount",         _OFFS(entity_amount),         {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_amount),         ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
	{ "entity_regen",          _OFFS(entity_regen),          {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_regen),          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 },
	{ "entity_state",          _OFFS(entity_state),          {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_state),          ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 },
};

/* Names array — last entry is `hidden` (handled separately because its
 * tensor lives in ctx->hidden, not via the table). */
static const char *QNN_ONNX_INPUT_NAMES[QNN_ONNX_N_INPUTS] = {
	"self_health", "self_effective_armor",
	"self_ammo_shells", "self_ammo_nails", "self_ammo_rockets", "self_ammo_cells",
	"self_vel", "self_attack_finished",
	"self_weapon_id", "self_movement_id", "self_items", "view_pitch",
	"spatial_dir", "spatial_nearest_dist", "spatial_mean_dist",
	"spatial_openness", "spatial_clearance", "spatial_traversable", "spatial_dropoff",
	"spatial_solid_frac", "spatial_water_frac", "spatial_slime_frac", "spatial_lava_frac",
	"entity_types", "entity_subject_id", "entity_modality_id", "entity_player_id",
	"entity_event_count", "entity_event_actions", "entity_event_sources",
	"entity_half_extents", "entity_rel", "entity_vel", "entity_path",
	"entity_path_dist", "entity_eta", "entity_recency",
	"entity_facing", "entity_team", "entity_score",
	"entity_amount", "entity_regen", "entity_state",
	"hidden",
};

static const char *QNN_ONNX_OUTPUT_NAMES[QNN_ONNX_N_OUTPUTS] = {
	"move_logits", "look", "fire_logit", "weapon_logits", "next_hidden",
};


/* ── Action decode ──────────────────────────────────────────────── */

/* Engine impulse byte (1..8 = AXE..LG) → weapon class (0..7).
 * Mirrors _CombatObjectiveNet._weapon_choices_from_ids. */
static int qnn_onnx_weapon_id_to_class(int weapon_id)
{
	if (weapon_id >= 1 && weapon_id <= 8) return weapon_id - 1;
	return 0;
}

static float qnn_onnx_clampf(float v, float lo, float hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

static void qnn_onnx_top2(const float *xs, int n, int *i1, float *v1, int *i2, float *v2)
{
	int i;

	*i1 = *i2 = -1;
	*v1 = *v2 = -INFINITY;
	for (i = 0; i < n; ++i) {
		float x = xs[i];
		if (x > *v1) {
			*i2 = *i1; *v2 = *v1;
			*i1 = i;   *v1 = x;
		} else if (x > *v2) {
			*i2 = i;   *v2 = x;
		}
	}
}

static void qnn_onnx_decode(const qnn_onnx_ctx_t *ctx, int self_weapon_id, qnn_action_t *out)
{
	int axis, c, best, i;
	float best_v, p_fire;
	int top1_idx, top2_idx;
	float top1_val, top2_val;
	float maxv, sumexp, top1_prob, top2_prob, confidence, margin;
	int current_class, desired_class, chosen_class;

	/* ---- move: 3 axes × 3 classes; argmax per axis; encoded as the
	 * press byte's per-axis neg/pos bits.  attack bit OR'd in below. ---- */
	{
		int axis_signs[3] = {0, 0, 0};
		int fb_neg, fb_pos, lr_neg, lr_pos, up_neg, up_pos, attack_bit;
		for (axis = 0; axis < 3; ++axis) {
			const float *row = &ctx->move_logits[axis * 3];
			best = 0;
			best_v = row[0];
			for (c = 1; c < 3; ++c) {
				if (row[c] > best_v) { best_v = row[c]; best = c; }
			}
			axis_signs[axis] = best - 1;   /* class 0,1,2 → -1, 0, +1 */
		}
		fb_neg = axis_signs[0] < 0 ? 1 : 0;
		fb_pos = axis_signs[0] > 0 ? 1 : 0;
		lr_neg = axis_signs[1] < 0 ? 1 : 0;
		lr_pos = axis_signs[1] > 0 ? 1 : 0;
		up_neg = axis_signs[2] < 0 ? 1 : 0;
		up_pos = axis_signs[2] > 0 ? 1 : 0;
		p_fire = 1.0f / (1.0f + expf(-ctx->fire_logit));
		attack_bit = (p_fire > 0.5f) ? 1 : 0;
		out->move = QNN_PackInputMask(
			/*alive=*/1,
			fb_neg, fb_pos, lr_neg, lr_pos,
			up_neg, up_pos,
			/*jump_act=*/up_pos,
			/*attack_act=*/attack_bit);
	}

	/* ---- look: already a unit vector; clamp for fp noise ---- */
	for (i = 0; i < 3; ++i)
		out->look[i] = qnn_onnx_clampf(ctx->look[i], -1.0f, 1.0f);

	/* ---- weapon: sticky controller (softmax top-2 + conf/margin gate) ---- */
	qnn_onnx_top2(ctx->weapon_logits, QNN_ONNX_WEAPON_CLASSES,
	              &top1_idx, &top1_val, &top2_idx, &top2_val);

	maxv = top1_val;
	sumexp = 0.0f;
	for (i = 0; i < QNN_ONNX_WEAPON_CLASSES; ++i)
		sumexp += expf(ctx->weapon_logits[i] - maxv);
	top1_prob = expf(top1_val - maxv) / sumexp;
	top2_prob = (top2_idx >= 0) ? expf(top2_val - maxv) / sumexp : 0.0f;

	current_class = qnn_onnx_weapon_id_to_class(self_weapon_id);
	desired_class = top1_idx;
	confidence = top1_prob;
	margin = top1_prob - top2_prob;

	chosen_class = current_class;
	if (desired_class != current_class
	    && confidence >= QNN_ONNX_WEAPON_SWITCH_CONFIDENCE
	    && margin     >= QNN_ONNX_WEAPON_SWITCH_MARGIN) {
		chosen_class = desired_class;
	}
	out->weapon = chosen_class + 1;   /* class 0..7 → impulse 1..8 */
}


/* ── Step ───────────────────────────────────────────────────────── */

int QNN_OnnxStep(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result, qnn_action_t *out)
{
	const OrtApi *ort;
	OrtValue *in_values[QNN_ONNX_N_INPUTS];
	OrtValue *out_values[QNN_ONNX_N_OUTPUTS];
	int rc = 1;
	size_t i;
	void *p;
	int64_t hidden_shape[2];

	if (ctx == NULL || result == NULL || out == NULL) {
		qnn_onnx_set_error("QNN_OnnxStep: NULL argument");
		return 1;
	}
	ort = ctx->ort;
	memset(in_values,  0, sizeof(in_values));
	memset(out_values, 0, sizeof(out_values));

	pack_scratch(ctx, result);

	for (i = 0; i < QNN_ONNX_N_OBS_INPUTS; ++i) {
		const qnn_onnx_input_def_t *def = &QNN_ONNX_INPUTS[i];
		void *buf = (void *)((char *)ctx + def->ctx_offset);
		QNN_ONNX_CHECK_OR_FAIL(
			ort->CreateTensorWithDataAsOrtValue(
				ctx->meminfo, buf, def->byte_count,
				def->shape, def->n_dims, def->dtype, &in_values[i]),
			"CreateTensorWithDataAsOrtValue(obs)");
	}

	/* Hidden state input — separate because it lives in ctx->hidden, not via the table. */
	hidden_shape[0] = 1;
	hidden_shape[1] = QNN_ONNX_GRU_HIDDEN_DIM;
	QNN_ONNX_CHECK_OR_FAIL(
		ort->CreateTensorWithDataAsOrtValue(
			ctx->meminfo, ctx->hidden, sizeof(ctx->hidden),
			hidden_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_values[QNN_ONNX_N_OBS_INPUTS]),
		"CreateTensorWithDataAsOrtValue(hidden)");

	QNN_ONNX_CHECK_OR_FAIL(
		ort->Run(
			ctx->session, NULL,
			QNN_ONNX_INPUT_NAMES, (const OrtValue *const *)in_values, QNN_ONNX_N_INPUTS,
			QNN_ONNX_OUTPUT_NAMES, QNN_ONNX_N_OUTPUTS, out_values),
		"Session.Run");

	QNN_ONNX_CHECK_OR_FAIL(ort->GetTensorMutableData(out_values[0], &p), "Get(move_logits)");
	memcpy(ctx->move_logits, p, sizeof(ctx->move_logits));
	QNN_ONNX_CHECK_OR_FAIL(ort->GetTensorMutableData(out_values[1], &p), "Get(look)");
	memcpy(ctx->look, p, sizeof(ctx->look));
	QNN_ONNX_CHECK_OR_FAIL(ort->GetTensorMutableData(out_values[2], &p), "Get(fire_logit)");
	ctx->fire_logit = ((float *)p)[0];
	QNN_ONNX_CHECK_OR_FAIL(ort->GetTensorMutableData(out_values[3], &p), "Get(weapon_logits)");
	memcpy(ctx->weapon_logits, p, sizeof(ctx->weapon_logits));
	QNN_ONNX_CHECK_OR_FAIL(ort->GetTensorMutableData(out_values[4], &p), "Get(next_hidden)");
	memcpy(ctx->next_hidden, p, sizeof(ctx->next_hidden));

	memcpy(ctx->hidden, ctx->next_hidden, sizeof(ctx->hidden));
	qnn_onnx_decode(ctx, result->self.weapon_id, out);
	rc = 0;

fail:
	for (i = 0; i < QNN_ONNX_N_INPUTS;  ++i) if (in_values[i])  ort->ReleaseValue(in_values[i]);
	for (i = 0; i < QNN_ONNX_N_OUTPUTS; ++i) if (out_values[i]) ort->ReleaseValue(out_values[i]);
	return rc;
}
