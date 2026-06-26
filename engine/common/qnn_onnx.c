/*
 * qnn_onnx.c — ORT session lifecycle + per-tick inference + action
 * decode for the live NQ client. Direct from engine types
 * (qnn_tick_result_t in, qnn_action_t out) — no parallel obs/action
 * structs, no transcoder.
 *
 * Per-model obs codec
 * -------------------
 * Everything between the stable seam (qnn_tick_result_t in / qnn_action_t
 * out) — token emit, ORT tensor packing, output decode — is a single
 * SELECTABLE codec (qnn_obs_codec_t). A codec is the bin-side
 * implementation of ONE wire contract (see src/docs/contracts/README.md:
 * "wire vs codec" — the model declares its format, a codec implements
 * it). Exactly one codec is selected per loaded model (QNN_OnnxInit →
 * qnn_onnx_select_codec) and QNN_OnnxStep dispatches every tick through
 * ctx->codec. To add a contract, write its codec and add it to
 * QNN_CODECS[] — no change to Init/Step is needed.
 *
 * Native-width policy (wire.9): scratch buffers carry the same native
 * dtypes as the wire format (see qnn_io.h header comment /
 * src/qnn/engine_norm.py). The ONNX model's input dtypes match these,
 * and the model's own dequantizer modules (qnn.model.dequant) reproduce
 * the normalization the C side used to do inline. The wire.7 codec
 * (legacy v17/v22) instead emits already-normalized packed-float32
 * tensors (semantics.1 scales applied codec-side), since that model
 * generation has no dequantizer.
 *
 * Sections:
 *   - Thread-local error buffer
 *   - Codec interface (qnn_obs_codec_t)
 *   - Context struct (per-codec scratch) + lifecycle (Init / Reset / Free)
 *   - wire.9 codec: scratch packing, input bind, output decode
 *   - wire.7 codec: legacy packed-float32 emit (shares wire.9 decode)
 *   - Codec registry + load-time selection
 *   - Step (dispatch through ctx->codec->emit / ->decode)
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

/* The sticky-weapon gate (top-2 + confidence/margin thresholds) now runs
 * IN-GRAPH (export_onnx.ExportWrapper, Pattern A): the `weapon` output is the
 * DECIDED impulse byte, and its thresholds travel with the model (stamped under
 * the `decode.` metadata namespace) instead of being hardcoded here. */

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
 * Total: 11 + 11 + 20 + 1 = 43, plus look_delta (self) = 44 obs inputs. */
#define QNN_ONNX_N_OBS_INPUTS  44
#define QNN_ONNX_N_INPUTS      (QNN_ONNX_N_OBS_INPUTS + 1)
#define QNN_ONNX_N_OUTPUTS      5

/* Canonical output-head slot order (the slots in ctx->out_present and the
 * index decode reads each head from). Mirrors QNN_ONNX_OUTPUT_NAMES. */
#define QNN_ONNX_OUT_MOVE     0
#define QNN_ONNX_OUT_LOOK     1
#define QNN_ONNX_OUT_FIRE     2
#define QNN_ONNX_OUT_WEAPON   3
#define QNN_ONNX_OUT_HIDDEN   4

/* Wire-contract metadata keys (mirror tools/export_onnx.py stamping +
 * src/qnn/engine_norm.py WIRE_CONTRACT_ID / SEMANTICS_CONTRACT_ID). */
#define QNN_WIRE_CONTRACT_KEY       "wire_contract"
#define QNN_SEMANTICS_CONTRACT_KEY  "semantics_contract"
#define QNN_ARCH_KEY                "arch"
#define QNN_VERSION_KEY             "version"   /* compact a{arch}.s{sem}.w{wire} */
#define QNN_SEMANTICS_CONTRACT_ID   "semantics.1"

/* Upper bounds for the ORT bind arrays in QNN_OnnxStep — the max over
 * every registered codec. Today wire.9 (45) is the widest input set;
 * all codecs share the 5 wire outputs. */
#define QNN_ONNX_MAX_INPUTS    QNN_ONNX_N_INPUTS
#define QNN_ONNX_MAX_OUTPUTS   QNN_ONNX_N_OUTPUTS


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


/* ── Codec interface (qnn_obs_codec_t) ──────────────────────────────
 *
 * One codec = the bin-side implementation of one wire contract. The
 * codec owns: the ORT input/output name lists, the emit() that builds
 * this contract's input tensors from a qnn_tick_result_t, and the
 * decode() that turns the model's output tensors into a qnn_action_t.
 *
 *   emit():   fill in_values[0 .. n_inputs-1] (including the GRU hidden
 *             input). Tensors alias ctx-owned scratch — they live until
 *             QNN_OnnxStep releases them after Run. Returns 0 on success.
 *   decode(): read out_values[0 .. n_outputs-1] into *out and refresh
 *             ctx->hidden from next_hidden. Returns 0 on success.
 *
 * id            = the wire-contract this codec implements ("wire.9").
 * semantics_contract = the semantics contract the codec assumes
 *                 ("semantics.1"); checked against the model's stamp.
 * input_names/n_inputs + output_names/n_outputs are the literal name
 * arrays passed to OrtApi::Run; n_inputs counts obs + hidden. Codec
 * selection matches `id` (preferred) or the full input_names set against
 * the loaded session. */
typedef struct qnn_obs_codec {
	const char         *id;                 /* wire-contract id */
	const char         *semantics_contract; /* assumed semantics id */
	const char *const  *input_names;        /* ORT input set this codec binds */
	size_t              n_inputs;           /* obs + hidden */
	const char *const  *output_names;
	size_t              n_outputs;
	int (*emit)(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result,
	            OrtValue **in_values);
	int (*decode)(qnn_onnx_ctx_t *ctx, OrtValue *const *out_values,
	              const qnn_tick_result_t *result, qnn_action_t *out);
} qnn_obs_codec_t;


/* ── Context struct (per-codec scratch buffers) ─────────────────── */

struct qnn_onnx_ctx
{
	const OrtApi       *ort;
	OrtEnv             *env;
	OrtSessionOptions  *opts;
	OrtSession         *session;
	OrtMemoryInfo      *meminfo;

	/* Selected obs codec — bound once at load (QNN_OnnxInit),
	 * dispatched every tick by QNN_OnnxStep. */
	const qnn_obs_codec_t *codec;

	/* HEAD-AGNOSTIC presence — which of the codec's known OUTPUT heads
	 * (QNN_ONNX_OUT_* slots) this loaded graph actually declares. Set at
	 * load by enumerating the session's real outputs (qnn_onnx_select_codec);
	 * NO head is required. Step requests only present heads; decode drives
	 * present heads and leaves absent ones uncommanded (see qnn_onnx_decode).
	 * A graph output the codec doesn't know is never requested → ignored. */
	int out_present[QNN_ONNX_N_OUTPUTS];

	/* Whether the graph declares the `hidden` GRU INPUT. The codec can
	 * produce it, but a stateless (GRU-off) graph omits it (and its paired
	 * `next_hidden` output) — feed it only when declared, carry state only
	 * when next_hidden is present (out_present[QNN_ONNX_OUT_HIDDEN]). */
	int has_hidden_input;

	/* arch id from the model's `arch` metadata stamp — provenance only
	 * (shown in the load line; not a selection/validation key). */
	char arch[40];
	/* compact a{arch}.s{sem}.w{wire} render from the `version` stamp (provenance;
	 * empty when the model predates the stamp). */
	char version[32];

	/* GRU hidden state, carried across steps. Shared by all codecs
	 * (the GRU width is an arch property, identical across the load
	 * set). */
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
	uint16_t self_look_delta[3];       /* binary16 bit pattern × 3 */

	/* ── Spatial block scratch (B=1, N=9, per-field native) ── */
	int8_t   spatial_dir          [QNN_SPATIAL_TOKEN_COUNT][3];
	/* nearest/mean dist widened to i32: the ONNX graph takes int32 here
	 * (torch's ONNX tracer rejects UInt16 input tensors). Lossless — the
	 * quantizer's u16 range fits in i32. */
	int32_t  spatial_nearest_dist [QNN_SPATIAL_TOKEN_COUNT];
	int32_t  spatial_mean_dist    [QNN_SPATIAL_TOKEN_COUNT];
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
	int32_t  entity_path_dist      [QNN_MAX_TOKEN_OBJECTS];    /* i32 (was u16; see spatial_*_dist) */
	uint16_t entity_eta            [QNN_MAX_TOKEN_OBJECTS];    /* f16 */
	uint16_t entity_recency        [QNN_MAX_TOKEN_OBJECTS];    /* f16 */
	uint8_t  entity_facing         [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_team           [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_score          [QNN_MAX_TOKEN_OBJECTS];
	uint8_t  entity_amount         [QNN_MAX_TOKEN_OBJECTS];
	uint16_t entity_regen          [QNN_MAX_TOKEN_OBJECTS];    /* f16 */
	uint8_t  entity_state          [QNN_MAX_TOKEN_OBJECTS];

	/* ── wire.7 (legacy v17/v22) scratch ────────────────────────
	 * Packed, already-normalized float32 tensors + int64 IDs. Only
	 * populated when the wire.7 codec is selected. Layout per
	 * src/docs/contracts/wire/wire.7.md (constants below). */
	float    w7_self_scalars       [17];
	int64_t  w7_self_weapon_id;
	int64_t  w7_self_armor_type_id;
	int64_t  w7_self_movement_id;
	int64_t  w7_self_powerup_ids   [5];
	int64_t  w7_entity_types       [QNN_MAX_TOKEN_OBJECTS];
	float    w7_entity_scalars_raw [QNN_MAX_TOKEN_OBJECTS][19];
	int64_t  w7_entity_ids         [QNN_MAX_TOKEN_OBJECTS][3];
	int64_t  w7_entity_event_actions[QNN_MAX_TOKEN_OBJECTS][QNN_MAX_ENTITY_EVENTS];
	int64_t  w7_entity_event_sources[QNN_MAX_TOKEN_OBJECTS][QNN_MAX_ENTITY_EVENTS];
	int64_t  w7_entity_event_counts[QNN_MAX_TOKEN_OBJECTS];
	float    w7_spatial_scalars    [QNN_SPATIAL_TOKEN_COUNT][13];

	/* Output scratch. Weapon has two forms, one per wire generation:
	 *   wire.9 → `weapon_decided` (int64 impulse; sticky gate ran in-graph)
	 *   wire.7 → `weapon_logits`  (float[8]; engine runs the sticky controller)
	 * Only the slot the loaded codec declares is populated. */
	float    move_logits[3 * 3];
	float    look[3];
	float    fire_logit;
	int64_t  weapon_decided;                          /* wire.9 (Pattern A) */
	float    weapon_logits[QNN_ONNX_WEAPON_CLASSES];  /* wire.7 (engine gate) */
	float    next_hidden[QNN_ONNX_GRU_HIDDEN_DIM];
	/* wire.7 sticky-gate thresholds: read from `decode.*` ONNX metadata at
	 * load (Pattern B — params in the model), default to the historical
	 * 0.65/0.15 for un-stamped legacy graphs. Unused by wire.9 (in-graph). */
	float    weapon_switch_confidence;
	float    weapon_switch_margin;
	/* Move sticky decode (Pattern B): fb/lr hold the previous emitted class
	 * unless the argmax-class softmax prob >= move_sticky_tau (read from
	 * decode.move_sticky_tau, default 0.6). prev_move is per-axis class
	 * {0:neg,1:none,2:pos}, carried across ticks (reset with the GRU hidden).
	 * The jump axis (2) is sampled in-graph (gumbel) and not held. */
	int      prev_move[3];
	float    move_sticky_tau;
};


/* ── Lifecycle ──────────────────────────────────────────────────── */

#define QNN_ONNX_CHECK_OR_FAIL(call, where)                                 \
	do {                                                                \
		OrtStatus *_s = (call);                                     \
		if (qnn_onnx_set_error_from_ort(ort, _s, (where)))          \
			goto fail;                                          \
	} while (0)

/* Pick the obs codec for the open session and store it on ctx->codec;
 * validate the session's signature + semantics. Returns 0 on success,
 * non-zero (with qnn_onnx_error set) if no codec handles the model.
 * Defined after the codec registry. */
static int qnn_onnx_select_codec(qnn_onnx_ctx_t *ctx);

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

	/* ERROR level: ORT 1.26 logs W-level GPU device-discovery probes
	 * (/sys/class/drm/...) that always "fail" on the CPU-only pi — noise. */
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "qnn_onnx", &ctx->env), "CreateEnv");
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateSessionOptions(&ctx->opts), "CreateSessionOptions");
	QNN_ONNX_CHECK_OR_FAIL(ort->SetIntraOpNumThreads(ctx->opts, 1), "SetIntraOpNumThreads");
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateSession(ctx->env, onnx_path, ctx->opts, &ctx->session), "CreateSession");
	QNN_ONNX_CHECK_OR_FAIL(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->meminfo), "CreateCpuMemoryInfo");

	/* Choose + validate the obs codec for this model's wire contract. */
	if (qnn_onnx_select_codec(ctx) != 0)
		goto fail;

	/* One clean load line: path + the compact a{arch}.s{sem}.w{wire} version.
	 * Falls back to the raw codec/arch ids for models exported before the
	 * `version` stamp (those carry no combined string). */
	if (ctx->version[0] != '\0')
		fprintf(stderr, "model: loaded %s  %s\n", onnx_path, ctx->version);
	else
		fprintf(stderr, "model: loaded %s  %s/%s (unversioned)\n",
			onnx_path, ctx->codec->id, ctx->arch);

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
	/* episode start: no previous move → sticky decode falls back to argmax
	 * (none held; first confident frame sets it). class 1 = none. */
	ctx->prev_move[0] = ctx->prev_move[1] = ctx->prev_move[2] = 1;
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


/* ════════════════════════════════════════════════════════════════════
 *  wire.9 codec — native split, 44 inputs (current; v24 / full_4head)
 *
 *  The current native contract: per-field native dtypes, model-side
 *  dequant. See src/docs/contracts/wire/wire.9.md. This is the table +
 *  pack + bind + decode that the bin shipped before the codec seam,
 *  moved verbatim behind qnn_obs_codec_t — byte-for-byte identical ORT
 *  inputs and decoded action for a v24 model.
 *
 *  APPEND-ONLY INVARIANT on qnn_tick_result_t / qnn_snapshot_t: every
 *  codec derives its inputs from the shared tick result, so that struct
 *  must stay a SUPERSET of every contract's raw fields. Fields may be
 *  added but never removed/repurposed without retiring the codec(s) that
 *  read them (see the support-band floor in contracts/README.md).
 * ════════════════════════════════════════════════════════════════════ */

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
	/* look_delta = look[t-1]-look[t-2] (engine-computed; QNN_SelfEmitToken).
	 * f16 on the wire — matches the schema's self field at offset 21. */
	ctx->self_look_delta[0]   = QNN_FloatToHalf(self->look_delta[0]);
	ctx->self_look_delta[1]   = QNN_FloatToHalf(self->look_delta[1]);
	ctx->self_look_delta[2]   = QNN_FloatToHalf(self->look_delta[2]);

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
	{ "look_delta",           _OFFS(self_look_delta),      {1, 3}, 2, _SIZE_OF(self_look_delta),   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 },

	/* ── Spatial block (11 inputs, all shape (1, 9, …)) ───── */
	{ "spatial_dir",          _OFFS(spatial_dir),          {1, QNN_SPATIAL_TOKEN_COUNT, 3}, 3, _SIZE_OF(spatial_dir),          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 },
	{ "spatial_nearest_dist", _OFFS(spatial_nearest_dist), {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_nearest_dist), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 },
	{ "spatial_mean_dist",    _OFFS(spatial_mean_dist),    {1, QNN_SPATIAL_TOKEN_COUNT},    2, _SIZE_OF(spatial_mean_dist),    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 },
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
	{ "entity_path_dist",      _OFFS(entity_path_dist),      {1, QNN_MAX_TOKEN_OBJECTS},                     2, _SIZE_OF(entity_path_dist),      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 },
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
	"look_delta",
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

/* The FULL possible output set the wire.9/wire.7 codecs understand (the
 * superset). A loaded graph is SELF-DECLARING and fully HEAD-AGNOSTIC: it
 * carries a tensor only for the heads the model actually has, and NO head is
 * required. Load enumerates the session's real outputs and records per-slot
 * presence on ctx->out_present[]; an output name the codec doesn't know is
 * simply not requested by Step → harmlessly ignored. Step requests only the
 * present heads; decode drives present heads and leaves absent ones
 * uncommanded. Index order here is the canonical slot order (== QNN_ONNX_OUT_*)
 * that decode reads; absent slots stay NULL in out_values.
 *
 * The two wire generations differ ONLY in the weapon slot, and are otherwise
 * independent (own table + own decode fn): wire.9 emits a DECIDED `weapon`
 * impulse (int64; sticky gate in-graph), wire.7 emits raw `weapon_logits`
 * (float[8]; engine runs the controller). */
static const char *QNN_ONNX_OUTPUT_NAMES_W9[QNN_ONNX_N_OUTPUTS] = {
	"move_logits", "look", "fire_logit", "weapon", "next_hidden",
};
static const char *QNN_ONNX_OUTPUT_NAMES_W7[QNN_ONNX_N_OUTPUTS] = {
	"move_logits", "look", "fire_logit", "weapon_logits", "next_hidden",
};


/* ── Action decode ──────────────────────────────────────────────── */

static float qnn_onnx_clampf(float v, float lo, float hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

/* Engine impulse byte (1..8 = AXE..LG) → weapon class (0..7). wire.7 only —
 * the held weapon for that generation's engine-side sticky controller. */
static int qnn_onnx_weapon_id_to_class(int weapon_id)
{
	if (weapon_id >= 1 && weapon_id <= 8) return weapon_id - 1;
	return 0;
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

/* wire.7 sticky-weapon controller: softmax top-2 over weapon_logits, switch to
 * the top class only when conf>=C and margin>=M (C,M from ctx, read from the
 * model's stamped decode metadata or the legacy default), else hold the current
 * weapon. Returns the engine impulse byte 1..8. Mirrors QNNPolicy.emit_actions
 * — kept engine-side for wire.7 because that generation emits raw logits, not a
 * decided weapon (wire.9 bakes this in-graph). */
static int qnn_onnx_weapon_from_logits(const qnn_onnx_ctx_t *ctx, int self_weapon_id)
{
	int top1_idx, top2_idx, i, current_class, chosen_class;
	float top1_val, top2_val, maxv, sumexp, top1_prob, top2_prob, confidence, margin;

	qnn_onnx_top2(ctx->weapon_logits, QNN_ONNX_WEAPON_CLASSES,
	              &top1_idx, &top1_val, &top2_idx, &top2_val);
	maxv = top1_val;
	sumexp = 0.0f;
	for (i = 0; i < QNN_ONNX_WEAPON_CLASSES; ++i)
		sumexp += expf(ctx->weapon_logits[i] - maxv);
	top1_prob = expf(top1_val - maxv) / sumexp;
	top2_prob = (top2_idx >= 0) ? expf(top2_val - maxv) / sumexp : 0.0f;

	current_class = qnn_onnx_weapon_id_to_class(self_weapon_id);
	confidence = top1_prob;
	margin = top1_prob - top2_prob;
	chosen_class = current_class;
	if (top1_idx != current_class
	    && confidence >= ctx->weapon_switch_confidence
	    && margin     >= ctx->weapon_switch_margin)
		chosen_class = top1_idx;
	return chosen_class + 1;   /* class 0..7 → impulse 1..8 */
}

/* Shared move/look/fire decode (identical across wire generations). Weapon is
 * decoded per-codec by the caller (wire.9 passthrough int vs wire.7 controller),
 * so it is NOT touched here. ctx is mutable: the sticky move decode updates
 * ctx->prev_move across ticks. */
static void qnn_onnx_decode_core(qnn_onnx_ctx_t *ctx, qnn_action_t *out)
{
	int axis, c, best, i;
	float best_v, p_fire;

	/* HEAD-AGNOSTIC decode: each head drives its part of the action only
	 * when the graph declares it; an absent head leaves that part
	 * uncommanded. move + fire share the one packed input mask (alive is
	 * always set), so they are decoded together: an absent move yields zero
	 * movement bits, an absent fire yields no attack. */
	{
		int axis_signs[3] = {0, 0, 0};
		int fb_neg, fb_pos, lr_neg, lr_pos, up_neg, up_pos, attack_bit;

		/* ---- move: 3 axes × 3 classes (neg/none/pos).
		 *   fb/lr (axes 0,1): STICKY — hold the previous emitted class unless
		 *     the argmax-class softmax prob >= move_sticky_tau (kills the
		 *     per-frame strafe jitter; the raw logits arrive un-sampled).
		 *   jump (axis 2): plain argmax of the graph's gumbel-perturbed row,
		 *     i.e. a SAMPLE (preserves the calibrated jump mass).
		 * Encoded as the press byte's per-axis neg/pos bits. Absent → no move. */
		if (ctx->out_present[QNN_ONNX_OUT_MOVE]) {
			for (axis = 0; axis < 3; ++axis) {
				const float *row = &ctx->move_logits[axis * 3];
				best = 0;
				best_v = row[0];
				for (c = 1; c < 3; ++c) {
					if (row[c] > best_v) { best_v = row[c]; best = c; }
				}
				if (axis < 2) {
					/* sticky: switch to argmax only if confident enough. */
					float sm = 0.0f, conf;
					for (c = 0; c < 3; ++c) sm += expf(row[c] - best_v);
					conf = 1.0f / sm;   /* softmax prob of the argmax class */
					if (conf < ctx->move_sticky_tau)
						best = ctx->prev_move[axis];   /* hold previous */
					ctx->prev_move[axis] = best;
				}
				axis_signs[axis] = best - 1;   /* class 0,1,2 → -1, 0, +1 */
			}
		}
		fb_neg = axis_signs[0] < 0 ? 1 : 0;
		fb_pos = axis_signs[0] > 0 ? 1 : 0;
		lr_neg = axis_signs[1] < 0 ? 1 : 0;
		lr_pos = axis_signs[1] > 0 ? 1 : 0;
		up_neg = axis_signs[2] < 0 ? 1 : 0;
		up_pos = axis_signs[2] > 0 ? 1 : 0;
		/* ---- attack: fire sigmoid > 0.5. Absent → no attack. ---- */
		attack_bit = 0;
		if (ctx->out_present[QNN_ONNX_OUT_FIRE]) {
			p_fire = 1.0f / (1.0f + expf(-ctx->fire_logit));
			attack_bit = (p_fire > 0.5f) ? 1 : 0;
		}
		out->move = QNN_PackInputMask(
			/*alive=*/1,
			fb_neg, fb_pos, lr_neg, lr_pos,
			up_neg, up_pos,
			/*jump_act=*/up_pos,
			/*attack_act=*/attack_bit);
	}

	/* ---- look: already a unit vector; clamp for fp noise. Absent → leave
	 * look uncommanded (don't write — stays at the caller's neutral). ---- */
	if (ctx->out_present[QNN_ONNX_OUT_LOOK]) {
		for (i = 0; i < 3; ++i)
			out->look[i] = qnn_onnx_clampf(ctx->look[i], -1.0f, 1.0f);
	}
}


/* ── wire.9 emit / decode (codec entry points) ──────────────────── */

/* Bind the 44 obs tensors (via QNN_ONNX_INPUTS) + the GRU hidden input.
 * in_values[0 .. QNN_ONNX_N_INPUTS-1]; tensors alias ctx scratch. */
static int wire9_emit(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result,
                      OrtValue **in_values)
{
	const OrtApi *ort = ctx->ort;
	size_t i;
	int64_t hidden_shape[2];

	pack_scratch(ctx, result);

	for (i = 0; i < QNN_ONNX_N_OBS_INPUTS; ++i) {
		const qnn_onnx_input_def_t *def = &QNN_ONNX_INPUTS[i];
		void *buf = (void *)((char *)ctx + def->ctx_offset);
		OrtStatus *s = ort->CreateTensorWithDataAsOrtValue(
			ctx->meminfo, buf, def->byte_count,
			def->shape, def->n_dims, def->dtype, &in_values[i]);
		if (qnn_onnx_set_error_from_ort(ort, s, "CreateTensorWithDataAsOrtValue(obs)"))
			return 1;
	}

	/* Hidden state input — lives in ctx->hidden, not via the table. Feed it
	 * ONLY when the graph declares it (a GRU-off model has no `hidden` input);
	 * Step then runs with codec->n_inputs-1 (hidden is the last name). */
	if (ctx->has_hidden_input) {
		OrtStatus *s;
		hidden_shape[0] = 1;
		hidden_shape[1] = QNN_ONNX_GRU_HIDDEN_DIM;
		s = ort->CreateTensorWithDataAsOrtValue(
			ctx->meminfo, ctx->hidden, sizeof(ctx->hidden),
			hidden_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			&in_values[QNN_ONNX_N_OBS_INPUTS]);
		if (qnn_onnx_set_error_from_ort(ort, s, "CreateTensorWithDataAsOrtValue(hidden)"))
			return 1;
	}
	return 0;
}

/* Read the shared outputs (move/look/fire/next_hidden) into ctx scratch and
 * refresh hidden. Weapon is read per-codec by the caller (its tensor differs:
 * wire.9 `weapon` int64 vs wire.7 `weapon_logits` float[8]). HEAD-AGNOSTIC:
 * read a head only when this graph declares it. */
static int qnn_onnx_extract_core(qnn_onnx_ctx_t *ctx, OrtValue *const *out_values)
{
	const OrtApi *ort = ctx->ort;
	void *p;
	OrtStatus *s;

	if (ctx->out_present[QNN_ONNX_OUT_MOVE]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_MOVE], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(move_logits)")) return 1;
		memcpy(ctx->move_logits, p, sizeof(ctx->move_logits));
	}
	if (ctx->out_present[QNN_ONNX_OUT_LOOK]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_LOOK], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(look)")) return 1;
		memcpy(ctx->look, p, sizeof(ctx->look));
	}
	if (ctx->out_present[QNN_ONNX_OUT_FIRE]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_FIRE], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(fire_logit)")) return 1;
		ctx->fire_logit = ((float *)p)[0];
	}
	/* next_hidden → carry GRU state across steps. Absent → stateless model;
	 * leave ctx->hidden untouched (it stays zero — see QNN_OnnxReset). */
	if (ctx->out_present[QNN_ONNX_OUT_HIDDEN]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_HIDDEN], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(next_hidden)")) return 1;
		memcpy(ctx->next_hidden, p, sizeof(ctx->next_hidden));
		memcpy(ctx->hidden, ctx->next_hidden, sizeof(ctx->hidden));
	}
	return 0;
}

/* wire.9 decode: weapon is the DECIDED impulse int (sticky gate ran in-graph,
 * Pattern A) — pass it through. Independent of wire.7. */
static int wire9_decode(qnn_onnx_ctx_t *ctx, OrtValue *const *out_values,
                        const qnn_tick_result_t *result, qnn_action_t *out)
{
	const OrtApi *ort = ctx->ort;
	void *p;
	OrtStatus *s;

	(void)result;   /* held weapon not needed: the gate decided in-graph */
	if (qnn_onnx_extract_core(ctx, out_values)) return 1;
	if (ctx->out_present[QNN_ONNX_OUT_WEAPON]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_WEAPON], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(weapon)")) return 1;
		ctx->weapon_decided = ((int64_t *)p)[0];   /* decided impulse 1..8 */
	}
	qnn_onnx_decode_core(ctx, out);
	out->weapon = ctx->out_present[QNN_ONNX_OUT_WEAPON] ? (int)ctx->weapon_decided : 0;
	return 0;
}

/* wire.7 decode: weapon is raw `weapon_logits` float[8] — run the engine-side
 * sticky controller (thresholds from ctx, stamped or legacy default). Fully
 * independent of wire.9 (own output table + own weapon path). */
static int wire7_decode(qnn_onnx_ctx_t *ctx, OrtValue *const *out_values,
                        const qnn_tick_result_t *result, qnn_action_t *out)
{
	const OrtApi *ort = ctx->ort;
	void *p;
	OrtStatus *s;

	if (qnn_onnx_extract_core(ctx, out_values)) return 1;
	if (ctx->out_present[QNN_ONNX_OUT_WEAPON]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_WEAPON], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(weapon_logits)")) return 1;
		memcpy(ctx->weapon_logits, p, sizeof(ctx->weapon_logits));
	}
	qnn_onnx_decode_core(ctx, out);
	out->weapon = ctx->out_present[QNN_ONNX_OUT_WEAPON]
		? qnn_onnx_weapon_from_logits(ctx, result->self.weapon_id) : 0;
	return 0;
}

static const qnn_obs_codec_t QNN_CODEC_WIRE_9 = {
	"wire.9",
	QNN_SEMANTICS_CONTRACT_ID,
	QNN_ONNX_INPUT_NAMES,  QNN_ONNX_N_INPUTS,
	QNN_ONNX_OUTPUT_NAMES_W9, QNN_ONNX_N_OUTPUTS,
	wire9_emit,
	wire9_decode,
};


/* ════════════════════════════════════════════════════════════════════
 *  wire.7 codec — legacy packed scalars, 13 inputs (v17 / v22)
 *
 *  The packed-float32 contract (token-spec ~v11). Unlike wire.9, this
 *  model generation has NO model-side dequantizer: the C side baked the
 *  semantics.1 scales into the packed float tensors. So this codec emits
 *  ALREADY-NORMALIZED float32 (health/100, rel/1000, eta/60, …) — the
 *  same normalization the v17-era C packer (commit c38a5a26 qnn_io.c /
 *  qnn_self_common.c / qnn_oracle.c) applied. Layout cross-checked
 *  against /tmp/qnn_v17.onnx, docs/contracts/wire/wire.7.md, the v11
 *  archive (docs/archive/token-spec-v11.md), and git show
 *  b0f75210^:src/qnn/wire.py.
 *
 *  ID PROMOTION: the packed wire stored IDs as int32; the ONNX graph
 *  inputs are int64. This codec promotes (the w7_* scratch is int64).
 *
 *  Outputs match wire.9 EXCEPT the weapon slot: wire.7 emits raw
 *  `weapon_logits` (float[8]) and the engine runs the sticky controller
 *  (wire7_decode), whereas wire.9 emits a decided `weapon` int (gate
 *  in-graph). The two are otherwise independent — own output table + decode.
 *  v17's `weapon_logits` second dim is a symbolic export artifact;
 *  semantically 8 (== QNN_ONNX_WEAPON_CLASSES) either way.
 * ════════════════════════════════════════════════════════════════════ */

/* cl.items bit positions (semantics.1 / engine_norm.py; native Quake
 * layout). Local copies — the engine's IT_* live in per-game TUs the
 * client build drops. */
#define W7_IT_SHOTGUN          (1 <<  0)
#define W7_IT_SUPER_SHOTGUN    (1 <<  1)
#define W7_IT_NAILGUN          (1 <<  2)
#define W7_IT_SUPER_NAILGUN    (1 <<  3)
#define W7_IT_GRENADE_LAUNCHER (1 <<  4)
#define W7_IT_ROCKET_LAUNCHER  (1 <<  5)
#define W7_IT_LIGHTNING        (1 <<  6)
#define W7_IT_ARMOR1           (1 << 13)   /* green */
#define W7_IT_ARMOR2           (1 << 14)   /* yellow */
#define W7_IT_ARMOR3           (1 << 15)   /* red */
#define W7_IT_INVISIBILITY     (1 << 19)   /* ring */
#define W7_IT_INVULNERABILITY  (1 << 20)   /* pent */
#define W7_IT_SUIT             (1 << 21)
#define W7_IT_QUAD             (1 << 22)

/* Largest v17 weapon cooldown ceiling used to normalize attack_finished
 * (QNN_ATTACK_FINISHED_MAX_SEC at c38a5a26 = 1.0s). */
#define W7_ATTACK_FINISHED_MAX_SEC 1.0f

static const char *QNN_WIRE7_INPUT_NAMES[] = {
	"self_scalars", "self_weapon_id", "self_armor_type_id", "self_movement_id",
	"self_powerup_ids", "entity_types", "entity_scalars_raw", "entity_ids",
	"entity_event_actions", "entity_event_sources", "entity_event_counts",
	"spatial_scalars", "hidden",
};
#define QNN_WIRE7_N_INPUTS  (sizeof(QNN_WIRE7_INPUT_NAMES) / sizeof(QNN_WIRE7_INPUT_NAMES[0]))

static float w7_clampf(float v, float lo, float hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

/* Per-subject item-amount normalization — mirrors v17 oracle
 * QNN_NormalizeItemAmount (c38a5a26). */
static float w7_item_amount(int subject_id, float amount)
{
	switch (subject_id) {
	case QNN_SUBJECT_HEALTH:
	case QNN_SUBJECT_MEGAHEALTH:      return amount / QNN_MAX_HEALTH;
	case QNN_SUBJECT_ARMOR_GREEN:     return (amount * 0.3f) / QNN_MAX_ARMOR;
	case QNN_SUBJECT_ARMOR_YELLOW:    return (amount * 0.6f) / QNN_MAX_ARMOR;
	case QNN_SUBJECT_ARMOR_RED:       return (amount * 0.8f) / QNN_MAX_ARMOR;
	case QNN_SUBJECT_SHELLS:          return amount / QNN_MAX_SHELLS;
	case QNN_SUBJECT_NAILS:           return amount / QNN_MAX_NAILS;
	case QNN_SUBJECT_ROCKETS:         return amount / QNN_MAX_ROCKETS;
	case QNN_SUBJECT_CELLS:           return amount / QNN_MAX_CELLS;
	case QNN_SUBJECT_QUAD:
	case QNN_SUBJECT_PENT:
	case QNN_SUBJECT_RING:
	case QNN_SUBJECT_SUIT:            return 1.0f;
	case QNN_SUBJECT_SHOTGUN:
	case QNN_SUBJECT_SUPER_SHOTGUN:   return 5.0f / QNN_MAX_SHELLS;
	case QNN_SUBJECT_NAILGUN:
	case QNN_SUBJECT_SUPER_NAILGUN:   return 30.0f / QNN_MAX_NAILS;
	case QNN_SUBJECT_GRENADE_LAUNCHER:return 5.0f / QNN_MAX_ROCKETS;
	case QNN_SUBJECT_ROCKET_LAUNCHER: return 5.0f / QNN_MAX_ROCKETS;
	case QNN_SUBJECT_THUNDERBOLT:     return 15.0f / QNN_MAX_CELLS;
	default:                          return 0.0f;
	}
}

/* Fill the wire.7 ACTOR-width [19] entity scalar row for one slot. The
 * 19 columns are the ACTOR superset; ITEM/MOVER/PROJECTILE fill their
 * own subset and leave the rest zero (matching the densified zero-pad).
 * Index map (ACTOR): 0-2 half/1000, 3-5 rel/1000, 6 dist/1000,
 * 7-9 vel/2000, 10-12 path/1000, 13 path_dist/1000, 14 eta/60,
 * 15 facing, 16 team, 17 score, 18 recency. */
static void w7_pack_entity(qnn_onnx_ctx_t *ctx, const qnn_tagged_token_t *tt, int slot)
{
	float *s = ctx->w7_entity_scalars_raw[slot];
	int64_t *ids = ctx->w7_entity_ids[slot];
	const qnn_token_event_t *evs;
	int n_evt, j, k;

	switch (tt->type) {
	case QNN_TOKEN_PROJECTILE: {
		const qnn_projectile_token_t *t = &tt->projectile;
		/* PROJECTILE [8]: rel/1000, dist/1000, vel/2000, recency/60. */
		s[0] = t->rel[0] / QNN_DIST_SCALE;
		s[1] = t->rel[1] / QNN_DIST_SCALE;
		s[2] = t->rel[2] / QNN_DIST_SCALE;
		s[3] = t->dist   / QNN_DIST_SCALE;
		s[4] = t->vel[0] / QNN_VELOCITY_SCALE;
		s[5] = t->vel[1] / QNN_VELOCITY_SCALE;
		s[6] = t->vel[2] / QNN_VELOCITY_SCALE;
		s[7] = t->recency / QNN_TIME_SCALE;
		ids[0] = t->subject_id; ids[1] = t->modality_id; /* ids[2] player: 0 */
		evs = t->events; n_evt = t->event_count;
		break;
	}
	case QNN_TOKEN_ACTOR: {
		const qnn_actor_token_t *t = &tt->actor;
		s[0]  = t->half_extents[0] / QNN_DIST_SCALE;
		s[1]  = t->half_extents[1] / QNN_DIST_SCALE;
		s[2]  = t->half_extents[2] / QNN_DIST_SCALE;
		s[3]  = t->rel[0] / QNN_DIST_SCALE;
		s[4]  = t->rel[1] / QNN_DIST_SCALE;
		s[5]  = t->rel[2] / QNN_DIST_SCALE;
		s[6]  = t->dist   / QNN_DIST_SCALE;
		s[7]  = t->vel[0] / QNN_VELOCITY_SCALE;
		s[8]  = t->vel[1] / QNN_VELOCITY_SCALE;
		s[9]  = t->vel[2] / QNN_VELOCITY_SCALE;
		s[10] = t->path[0] / QNN_DIST_SCALE;
		s[11] = t->path[1] / QNN_DIST_SCALE;
		s[12] = t->path[2] / QNN_DIST_SCALE;
		s[13] = t->path_dist / QNN_DIST_SCALE;
		s[14] = t->eta / QNN_TIME_SCALE;
		s[15] = t->facing;                  /* already [-1,1] */
		s[16] = (t->team > 0.5f) ? 1.0f : 0.0f;
		s[17] = t->score;                   /* already [0,1] */
		s[18] = t->recency / QNN_TIME_SCALE;
		ids[0] = t->subject_id; ids[1] = t->modality_id; ids[2] = t->player_id;
		evs = t->events; n_evt = t->event_count;
		break;
	}
	case QNN_TOKEN_ITEM: {
		const qnn_item_token_t *t = &tt->item;
		/* ITEM [15]: half/1000, rel/1000, dist/1000, path/1000,
		 * path_dist/1000, eta/60, amount(per-subject), regen/60,
		 * recency/60. */
		s[0]  = t->half_extents[0] / QNN_DIST_SCALE;
		s[1]  = t->half_extents[1] / QNN_DIST_SCALE;
		s[2]  = t->half_extents[2] / QNN_DIST_SCALE;
		s[3]  = t->rel[0] / QNN_DIST_SCALE;
		s[4]  = t->rel[1] / QNN_DIST_SCALE;
		s[5]  = t->rel[2] / QNN_DIST_SCALE;
		s[6]  = t->dist   / QNN_DIST_SCALE;
		s[7]  = t->path[0] / QNN_DIST_SCALE;
		s[8]  = t->path[1] / QNN_DIST_SCALE;
		s[9]  = t->path[2] / QNN_DIST_SCALE;
		s[10] = t->path_dist / QNN_DIST_SCALE;
		s[11] = t->eta / QNN_TIME_SCALE;
		s[12] = w7_item_amount(t->subject_id, t->amount);
		s[13] = t->regen / QNN_TIME_SCALE;
		s[14] = t->recency / QNN_TIME_SCALE;
		ids[0] = t->subject_id; ids[1] = t->modality_id;
		evs = t->events; n_evt = t->event_count;
		break;
	}
	case QNN_TOKEN_MOVER: {
		const qnn_mover_token_t *t = &tt->mover;
		/* MOVER [14]: half/1000, rel/1000, dist/1000, path/1000,
		 * path_dist/1000, eta/60, state, recency/60. */
		s[0]  = t->half_extents[0] / QNN_DIST_SCALE;
		s[1]  = t->half_extents[1] / QNN_DIST_SCALE;
		s[2]  = t->half_extents[2] / QNN_DIST_SCALE;
		s[3]  = t->rel[0] / QNN_DIST_SCALE;
		s[4]  = t->rel[1] / QNN_DIST_SCALE;
		s[5]  = t->rel[2] / QNN_DIST_SCALE;
		s[6]  = t->dist   / QNN_DIST_SCALE;
		s[7]  = t->path[0] / QNN_DIST_SCALE;
		s[8]  = t->path[1] / QNN_DIST_SCALE;
		s[9]  = t->path[2] / QNN_DIST_SCALE;
		s[10] = t->path_dist / QNN_DIST_SCALE;
		s[11] = t->eta / QNN_TIME_SCALE;
		s[12] = t->state;
		s[13] = t->recency / QNN_TIME_SCALE;
		ids[0] = t->subject_id; ids[1] = t->modality_id;
		evs = t->events; n_evt = t->event_count;
		break;
	}
	default:
		return;
	}

	if (n_evt > QNN_MAX_ENTITY_EVENTS) n_evt = QNN_MAX_ENTITY_EVENTS;
	ctx->w7_entity_event_counts[slot] = n_evt;
	for (j = 0, k = 0; j < n_evt; ++j) {
		ctx->w7_entity_event_actions[slot][k] = evs[j].action_id;
		ctx->w7_entity_event_sources[slot][k] = evs[j].source_id;
		++k;
	}
}

static void wire7_pack_scratch(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *r)
{
	const qnn_self_token_t *self = &r->self;
	float eff_armor;
	int items = self->items;
	int i, n, pu;

	memset(ctx->w7_self_scalars,        0, sizeof(ctx->w7_self_scalars));
	memset(ctx->w7_self_powerup_ids,    0, sizeof(ctx->w7_self_powerup_ids));
	memset(ctx->w7_entity_scalars_raw,  0, sizeof(ctx->w7_entity_scalars_raw));
	memset(ctx->w7_entity_ids,          0, sizeof(ctx->w7_entity_ids));
	memset(ctx->w7_entity_event_actions,0, sizeof(ctx->w7_entity_event_actions));
	memset(ctx->w7_entity_event_sources,0, sizeof(ctx->w7_entity_event_sources));
	memset(ctx->w7_entity_event_counts, 0, sizeof(ctx->w7_entity_event_counts));
	for (i = 0; i < QNN_MAX_TOKEN_OBJECTS; ++i)
		ctx->w7_entity_types[i] = -1;

	/* ---- self_scalars[17] (already-normalized) ---- */
	eff_armor = (float)self->raw_armor * self->armor_type;
	ctx->w7_self_scalars[0]  = (float)self->health / QNN_MAX_HEALTH;
	ctx->w7_self_scalars[1]  = eff_armor / QNN_MAX_ARMOR;
	ctx->w7_self_scalars[2]  = (items & W7_IT_SHOTGUN)          ? 1.0f : 0.0f;
	ctx->w7_self_scalars[3]  = (items & W7_IT_SUPER_SHOTGUN)    ? 1.0f : 0.0f;
	ctx->w7_self_scalars[4]  = (items & W7_IT_NAILGUN)          ? 1.0f : 0.0f;
	ctx->w7_self_scalars[5]  = (items & W7_IT_SUPER_NAILGUN)    ? 1.0f : 0.0f;
	ctx->w7_self_scalars[6]  = (items & W7_IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
	ctx->w7_self_scalars[7]  = (items & W7_IT_ROCKET_LAUNCHER)  ? 1.0f : 0.0f;
	ctx->w7_self_scalars[8]  = (items & W7_IT_LIGHTNING)        ? 1.0f : 0.0f;
	ctx->w7_self_scalars[9]  = w7_clampf((float)self->ammo_shells  / QNN_MAX_SHELLS,  0.0f, 1.0f);
	ctx->w7_self_scalars[10] = w7_clampf((float)self->ammo_nails   / QNN_MAX_NAILS,   0.0f, 1.0f);
	ctx->w7_self_scalars[11] = w7_clampf((float)self->ammo_rockets / QNN_MAX_ROCKETS, 0.0f, 1.0f);
	ctx->w7_self_scalars[12] = w7_clampf((float)self->ammo_cells   / QNN_MAX_CELLS,   0.0f, 1.0f);
	ctx->w7_self_scalars[13] = self->vel[0] / QNN_VELOCITY_SCALE;
	ctx->w7_self_scalars[14] = self->vel[1] / QNN_VELOCITY_SCALE;
	ctx->w7_self_scalars[15] = self->vel[2] / QNN_VELOCITY_SCALE;
	ctx->w7_self_scalars[16] = w7_clampf(self->attack_finished / W7_ATTACK_FINISHED_MAX_SEC, 0.0f, 1.0f);

	/* ---- self ids (int32 → int64 promotion) ---- */
	ctx->w7_self_weapon_id   = self->weapon_id;      /* already ENTITY_IDS-encoded */
	ctx->w7_self_movement_id = self->movement_id;
	ctx->w7_self_armor_type_id = 0;
	if (items & W7_IT_ARMOR3)      ctx->w7_self_armor_type_id = QNN_SUBJECT_ARMOR_RED;
	else if (items & W7_IT_ARMOR2) ctx->w7_self_armor_type_id = QNN_SUBJECT_ARMOR_YELLOW;
	else if (items & W7_IT_ARMOR1) ctx->w7_self_armor_type_id = QNN_SUBJECT_ARMOR_GREEN;

	/* powerup_ids[5], zero-padded — order matches v17 emit. */
	pu = 0;
	if (items & W7_IT_QUAD)            ctx->w7_self_powerup_ids[pu++] = QNN_SUBJECT_QUAD;
	if (items & W7_IT_INVULNERABILITY) ctx->w7_self_powerup_ids[pu++] = QNN_SUBJECT_PENT;
	if (items & W7_IT_INVISIBILITY)    ctx->w7_self_powerup_ids[pu++] = QNN_SUBJECT_RING;
	if (items & W7_IT_SUIT)            ctx->w7_self_powerup_ids[pu++] = QNN_SUBJECT_SUIT;
	if (self->health > 100 && pu < 5) ctx->w7_self_powerup_ids[pu++] = QNN_SUBJECT_MEGAHEALTH;

	/* ---- spatial_scalars[9, 13] (already-normalized) ----
	 * Order: dir[3], nearest/1000, mean/1000, openness, clearance,
	 * traversable, dropoff, solid, water, slime, lava. Matches the
	 * v17 packer (c38a5a26 qnn_io.c) + wire.7.md. */
	for (i = 0; i < QNN_SPATIAL_TOKEN_COUNT; ++i) {
		const qnn_spatial_token_t *t = &r->spatial[i];
		float *row = ctx->w7_spatial_scalars[i];
		row[0]  = t->dir[0];
		row[1]  = t->dir[1];
		row[2]  = t->dir[2];
		row[3]  = t->nearest_dist / QNN_DIST_SCALE;
		row[4]  = t->mean_dist    / QNN_DIST_SCALE;
		row[5]  = t->openness;
		row[6]  = t->clearance;
		row[7]  = t->traversable;
		row[8]  = t->dropoff;
		row[9]  = t->solid_frac;
		row[10] = t->water_frac;
		row[11] = t->slime_frac;
		row[12] = t->lava_frac;
	}

	/* ---- entity stream → densified [16,19] + ids/events ---- */
	n = r->entity_count < QNN_MAX_TOKEN_OBJECTS ? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
	for (i = 0; i < n; ++i) {
		const qnn_tagged_token_t *tt = &r->entities[i];
		if (tt->type < 0 || tt->type > QNN_TOKEN_MOVER) continue;
		ctx->w7_entity_types[i] = tt->type;
		w7_pack_entity(ctx, tt, i);
	}
}

static int wire7_emit(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result,
                      OrtValue **in_values)
{
	const OrtApi *ort = ctx->ort;
	OrtStatus *s;
	int64_t hidden_shape[2];

	/* (name, ptr, byte_count, {shape}, n_dims, dtype) per wire.7 input. */
	const struct {
		void   *buf;
		size_t  bytes;
		int64_t shape[3];
		size_t  n_dims;
		ONNXTensorElementDataType dtype;
	} binds[] = {
		{ ctx->w7_self_scalars,        sizeof(ctx->w7_self_scalars),        {1, 17},     2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT },
		{ &ctx->w7_self_weapon_id,     sizeof(ctx->w7_self_weapon_id),      {1, 1},      2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ &ctx->w7_self_armor_type_id, sizeof(ctx->w7_self_armor_type_id),  {1, 1},      2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ &ctx->w7_self_movement_id,   sizeof(ctx->w7_self_movement_id),    {1, 1},      2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_self_powerup_ids,    sizeof(ctx->w7_self_powerup_ids),    {1, 5},      2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_entity_types,        sizeof(ctx->w7_entity_types),        {1, QNN_MAX_TOKEN_OBJECTS},                        2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_entity_scalars_raw,  sizeof(ctx->w7_entity_scalars_raw),  {1, QNN_MAX_TOKEN_OBJECTS, 19},                    3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT },
		{ ctx->w7_entity_ids,          sizeof(ctx->w7_entity_ids),          {1, QNN_MAX_TOKEN_OBJECTS, 3},                     3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_entity_event_actions,sizeof(ctx->w7_entity_event_actions),{1, QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS}, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_entity_event_sources,sizeof(ctx->w7_entity_event_sources),{1, QNN_MAX_TOKEN_OBJECTS, QNN_MAX_ENTITY_EVENTS}, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_entity_event_counts, sizeof(ctx->w7_entity_event_counts), {1, QNN_MAX_TOKEN_OBJECTS},                        2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
		{ ctx->w7_spatial_scalars,     sizeof(ctx->w7_spatial_scalars),     {1, QNN_SPATIAL_TOKEN_COUNT, 13},                  3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT },
	};
	size_t n_binds = sizeof(binds) / sizeof(binds[0]);
	size_t i;

	wire7_pack_scratch(ctx, result);

	for (i = 0; i < n_binds; ++i) {
		s = ort->CreateTensorWithDataAsOrtValue(
			ctx->meminfo, binds[i].buf, binds[i].bytes,
			binds[i].shape, binds[i].n_dims, binds[i].dtype, &in_values[i]);
		if (qnn_onnx_set_error_from_ort(ort, s, "CreateTensorWithDataAsOrtValue(wire.7)"))
			return 1;
	}

	/* hidden — last input (index 12). Fed only when the graph declares it
	 * (uniform with wire.9); v17/v22 always do, so this is byte-identical. */
	if (ctx->has_hidden_input) {
		hidden_shape[0] = 1;
		hidden_shape[1] = QNN_ONNX_GRU_HIDDEN_DIM;
		s = ort->CreateTensorWithDataAsOrtValue(
			ctx->meminfo, ctx->hidden, sizeof(ctx->hidden),
			hidden_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_values[n_binds]);
		if (qnn_onnx_set_error_from_ort(ort, s, "CreateTensorWithDataAsOrtValue(hidden)"))
			return 1;
	}
	return 0;
}

static const qnn_obs_codec_t QNN_CODEC_WIRE_7 = {
	"wire.7",
	QNN_SEMANTICS_CONTRACT_ID,
	QNN_WIRE7_INPUT_NAMES, QNN_WIRE7_N_INPUTS,
	QNN_ONNX_OUTPUT_NAMES_W7, QNN_ONNX_N_OUTPUTS,   /* weapon_logits, not weapon */
	wire7_emit,
	wire7_decode,                                   /* engine-side weapon controller */
};


/* ════════════════════════════════════════════════════════════════════
 *  Codec registry + load-time selection
 *
 *  To add a wire contract: write its emit()/decode() + name arrays +
 *  a `static const qnn_obs_codec_t QNN_CODEC_<NAME> = {...}`, then add
 *  &QNN_CODEC_<NAME> here. Init/Step need no change — selection resolves
 *  the codec by the model's REQUIRED `wire_contract` stamp (an unstamped
 *  model is a hard load error — no name-match guessing), then validates the
 *  ORT input set is a SUBSET of it and records which known output heads the
 *  graph declares (head-agnostic; nothing required); dispatch goes through
 *  ctx->codec.
 * ════════════════════════════════════════════════════════════════════ */

static const qnn_obs_codec_t *const QNN_CODECS[] = {
	&QNN_CODEC_WIRE_9,
	&QNN_CODEC_WIRE_7,
};
#define QNN_N_CODECS (sizeof(QNN_CODECS) / sizeof(QNN_CODECS[0]))

static const qnn_obs_codec_t *qnn_codec_by_id(const char *id)
{
	size_t i;
	if (id == NULL) return NULL;
	for (i = 0; i < QNN_N_CODECS; ++i)
		if (strcmp(QNN_CODECS[i]->id, id) == 0)
			return QNN_CODECS[i];
	return NULL;
}

/* True iff `names` (size n) is a SUBSET of the codec's input_names set:
 * every input the graph declares must be one this codec can produce. The
 * codec need NOT supply every input it knows (optional inputs like `hidden`
 * may be absent on a GRU-off graph). An input the codec can't produce →
 * not a subset → refuse (stale/foreign stamp guard). */
static int qnn_codec_inputs_subset(const qnn_obs_codec_t *codec,
                                   const char *const *names, size_t n)
{
	size_t i, j;
	for (j = 0; j < n; ++j) {
		int found = 0;
		for (i = 0; i < codec->n_inputs; ++i)
			if (strcmp(codec->input_names[i], names[j]) == 0) { found = 1; break; }
		if (!found) return 0;
	}
	return 1;
}

/* Look up a custom-metadata-map string value. Returns a malloc'd copy
 * (caller frees via ort allocator) or NULL if absent. */
static char *qnn_onnx_metadata_lookup(qnn_onnx_ctx_t *ctx, OrtAllocator *alloc,
                                      const char *key)
{
	const OrtApi *ort = ctx->ort;
	OrtModelMetadata *meta = NULL;
	char *value = NULL;
	OrtStatus *s;

	s = ort->SessionGetModelMetadata(ctx->session, &meta);
	if (s != NULL) { ort->ReleaseStatus(s); return NULL; }
	s = ort->ModelMetadataLookupCustomMetadataMap(meta, alloc, key, &value);
	if (s != NULL) { ort->ReleaseStatus(s); value = NULL; }
	ort->ReleaseModelMetadata(meta);
	/* ORT returns "" for some absent keys; treat empty as absent. */
	if (value != NULL && value[0] == '\0') { (void)ort->AllocatorFree(alloc, value); value = NULL; }
	return value;
}

/* Read a `decode.<name>` float param stamped by export_onnx (the model's decode
 * regime travels with its weights). Returns `dflt` when the key is absent or
 * unparseable. The decode for the look/weapon heads runs IN-GRAPH (Pattern A),
 * so these are PROVENANCE today (logged at load); an engine-applied (Pattern B)
 * head reads them here instead of a hardcoded constant. */
static float qnn_onnx_decode_param(qnn_onnx_ctx_t *ctx, OrtAllocator *alloc,
                                   const char *name, float dflt)
{
	char key[96];
	char *val;
	float out = dflt;
	snprintf(key, sizeof(key), "decode.%s", name);
	val = qnn_onnx_metadata_lookup(ctx, alloc, key);
	if (val != NULL) {
		out = (float)atof(val);
		(void)ctx->ort->AllocatorFree(alloc, val);
	}
	return out;
}

static int qnn_onnx_select_codec(qnn_onnx_ctx_t *ctx)
{
	const OrtApi *ort = ctx->ort;
	OrtAllocator *alloc = NULL;
	const qnn_obs_codec_t *codec = NULL;
	char *wire_stamp = NULL, *sem_stamp = NULL;
	const char **in_names = NULL;
	const char **out_names = NULL;
	size_t n_in = 0, n_out = 0, i;
	int rc = 1;
	OrtStatus *s;

	s = ort->GetAllocatorWithDefaultOptions(&alloc);
	if (qnn_onnx_set_error_from_ort(ort, s, "GetAllocatorWithDefaultOptions"))
		return 1;

	/* Read the contract stamps (required — see the refusal paths below). */
	wire_stamp = qnn_onnx_metadata_lookup(ctx, alloc, QNN_WIRE_CONTRACT_KEY);
	sem_stamp  = qnn_onnx_metadata_lookup(ctx, alloc, QNN_SEMANTICS_CONTRACT_KEY);

	/* arch + version stamps — provenance only (shown in the load line); optional.
	 * `version` is the compact a{arch}.s{sem}.w{wire} render; empty for models
	 * exported before the stamp (the load line then falls back to the raw ids). */
	{
		char *arch_stamp = qnn_onnx_metadata_lookup(ctx, alloc, QNN_ARCH_KEY);
		char *ver_stamp  = qnn_onnx_metadata_lookup(ctx, alloc, QNN_VERSION_KEY);
		snprintf(ctx->arch, sizeof(ctx->arch), "%s", arch_stamp ? arch_stamp : "?");
		snprintf(ctx->version, sizeof(ctx->version), "%s", ver_stamp ? ver_stamp : "");
		if (arch_stamp) (void)ort->AllocatorFree(alloc, arch_stamp);
		if (ver_stamp)  (void)ort->AllocatorFree(alloc, ver_stamp);
	}

	/* decode.* params — the model's decode regime, stamped by export_onnx so it
	 * travels with the weights. wire.7 (engine-applied controller, Pattern B)
	 * uses these at runtime; wire.9 bakes the gate in-graph (Pattern A) so for
	 * it they are provenance. Default to the historical 0.65/0.15 for un-stamped
	 * legacy graphs. */
	ctx->weapon_switch_confidence =
		qnn_onnx_decode_param(ctx, alloc, "weapon_switch_confidence", 0.65f);
	ctx->weapon_switch_margin =
		qnn_onnx_decode_param(ctx, alloc, "weapon_switch_margin", 0.15f);
	/* fb/lr sticky-decode confidence threshold (engine-applied, Pattern B). */
	ctx->move_sticky_tau =
		qnn_onnx_decode_param(ctx, alloc, "move_sticky_tau", 0.6f);

	/* Enumerate the session's actual input names (for the signature
	 * validation against the selected codec below). */
	s = ort->SessionGetInputCount(ctx->session, &n_in);
	if (qnn_onnx_set_error_from_ort(ort, s, "SessionGetInputCount"))
		goto cleanup;
	in_names = (const char **)calloc(n_in, sizeof(*in_names));
	if (in_names == NULL) {
		qnn_onnx_set_error("qnn_onnx_select_codec: out of memory");
		goto cleanup;
	}
	for (i = 0; i < n_in; ++i) {
		char *name = NULL;
		s = ort->SessionGetInputName(ctx->session, i, alloc, &name);
		if (qnn_onnx_set_error_from_ort(ort, s, "SessionGetInputName"))
			goto cleanup;
		in_names[i] = name;   /* ort-allocated; freed below */
	}

	/* (1) The model MUST declare its wire contract — no name-match guessing.
	 * An unstamped model is a hard load error (stamp it: tools/stamp_onnx.py
	 * for an existing .onnx, or stamp_checkpoint.py + re-export). */
	if (wire_stamp == NULL) {
		char buf[1024];
		size_t off = 0;
		off += (size_t)snprintf(buf + off, sizeof(buf) - off,
			"model has no `wire_contract` stamp — refusing (no codec guessing); "
			"stamp it with tools/stamp_onnx.py or stamp_checkpoint.py. inputs=[");
		for (i = 0; i < n_in && off < sizeof(buf) - 2; ++i)
			off += (size_t)snprintf(buf + off, sizeof(buf) - off, "%s%s",
				i ? ", " : "", in_names[i]);
		if (off < sizeof(buf) - 2) snprintf(buf + off, sizeof(buf) - off, "]");
		qnn_onnx_set_error("%s", buf);
		goto cleanup;
	}
	codec = qnn_codec_by_id(wire_stamp);
	if (codec == NULL) {
		qnn_onnx_set_error(
			"wire_contract=%s: no codec for this contract in this bin", wire_stamp);
		goto cleanup;
	}

	/* (2) Validate the session signature is a SUBSET of the codec's input
	 * set: every input the graph wants must be one this codec can produce
	 * (an input the codec can't supply is a stale/foreign stamp → refuse).
	 * The codec MAY know inputs the graph omits — e.g. a GRU-off graph drops
	 * `hidden`. Record whether `hidden` is declared (feed it only if so). */
	if (!qnn_codec_inputs_subset(codec, in_names, n_in)) {
		qnn_onnx_set_error(
			"wire_contract=%s declares codec '%s' but the model declares an "
			"input that codec can't produce (stale/foreign stamp)",
			wire_stamp ? wire_stamp : "(none)", codec->id);
		goto cleanup;
	}
	ctx->has_hidden_input = 0;
	for (i = 0; i < n_in; ++i)
		if (strcmp(in_names[i], "hidden") == 0) { ctx->has_hidden_input = 1; break; }

	/* (3) Semantics check: a mismatch is silent miscalibration → load error.
	 * A missing stamp is also refused — no assuming (every model we ship is
	 * stamped via export_onnx / stamp_onnx / stamp_checkpoint). */
	if (sem_stamp == NULL) {
		qnn_onnx_set_error(
			"model has no `semantics_contract` stamp — refusing (no guessing); "
			"codec '%s' assumes %s", codec->id, codec->semantics_contract);
		goto cleanup;
	}
	if (strcmp(sem_stamp, codec->semantics_contract) != 0) {
		qnn_onnx_set_error(
			"semantics mismatch: model semantics_contract=%s but codec '%s' "
			"assumes %s (silent miscalibration — refusing)",
			sem_stamp, codec->id, codec->semantics_contract);
		goto cleanup;
	}

	/* (4) The graph is SELF-DECLARING and HEAD-AGNOSTIC about its output
	 * heads. Enumerate the session's REAL outputs (mirrors the input
	 * enumeration above) and record per-slot presence on ctx->out_present[].
	 * NO head is required (any may be absent → uncommanded). An output the
	 * codec doesn't know (e.g. a future `target_*`) is simply never requested
	 * by Step → harmlessly ignored, NOT a refusal. */
	s = ort->SessionGetOutputCount(ctx->session, &n_out);
	if (qnn_onnx_set_error_from_ort(ort, s, "SessionGetOutputCount"))
		goto cleanup;
	out_names = (const char **)calloc(n_out, sizeof(*out_names));
	if (out_names == NULL) {
		qnn_onnx_set_error("qnn_onnx_select_codec: out of memory (outputs)");
		goto cleanup;
	}
	for (i = 0; i < n_out; ++i) {
		char *name = NULL;
		s = ort->SessionGetOutputName(ctx->session, i, alloc, &name);
		if (qnn_onnx_set_error_from_ort(ort, s, "SessionGetOutputName"))
			goto cleanup;
		out_names[i] = name;   /* ort-allocated; freed in cleanup */
	}
	{
		size_t slot;
		/* Mark each known head the graph declares as present. An output name
		 * the codec doesn't recognize is left unmarked (and so never
		 * requested by Step) — ignored, not an error. NO head is required. */
		for (i = 0; i < n_out; ++i)
			for (slot = 0; slot < codec->n_outputs; ++slot)
				if (strcmp(out_names[i], codec->output_names[slot]) == 0) {
					ctx->out_present[slot] = 1;
					break;
				}
	}

	ctx->codec = codec;
	rc = 0;

cleanup:
	if (in_names != NULL) {
		for (i = 0; i < n_in; ++i)
			if (in_names[i]) (void)ort->AllocatorFree(alloc, (void *)in_names[i]);
		free(in_names);
	}
	if (out_names != NULL) {
		for (i = 0; i < n_out; ++i)
			if (out_names[i]) (void)ort->AllocatorFree(alloc, (void *)out_names[i]);
		free(out_names);
	}
	if (wire_stamp) (void)ort->AllocatorFree(alloc, wire_stamp);
	if (sem_stamp)  (void)ort->AllocatorFree(alloc, sem_stamp);
	return rc;
}


/* ── Step (dispatch through ctx->codec) ─────────────────────────── */

int QNN_OnnxStep(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result, qnn_action_t *out)
{
	const OrtApi *ort;
	const qnn_obs_codec_t *codec;
	OrtValue *in_values[QNN_ONNX_MAX_INPUTS];
	OrtValue *out_values[QNN_ONNX_MAX_OUTPUTS];
	/* Bound (present-only) output set passed to ort->Run, plus the map
	 * from each bound slot back to its canonical out_values index. */
	const char *bound_names[QNN_ONNX_MAX_OUTPUTS];
	OrtValue   *bound_values[QNN_ONNX_MAX_OUTPUTS];
	size_t      bound_slot[QNN_ONNX_MAX_OUTPUTS];
	size_t      n_bound = 0;
	int rc = 1;
	size_t i;

	if (ctx == NULL || result == NULL || out == NULL) {
		qnn_onnx_set_error("QNN_OnnxStep: NULL argument");
		return 1;
	}
	codec = ctx->codec;
	if (codec == NULL) {
		qnn_onnx_set_error("QNN_OnnxStep: no codec selected");
		return 1;
	}
	if (codec->n_inputs > QNN_ONNX_MAX_INPUTS || codec->n_outputs > QNN_ONNX_MAX_OUTPUTS) {
		qnn_onnx_set_error("QNN_OnnxStep: codec '%s' exceeds bind capacity", codec->id);
		return 1;
	}
	ort = ctx->ort;
	memset(in_values,  0, sizeof(in_values));
	memset(out_values, 0, sizeof(out_values));

	if (codec->emit(ctx, result, in_values) != 0)
		goto fail;

	/* Inputs fed to Run: every input the codec produced. `hidden` is the
	 * LAST codec input and is fed only when the graph declares it (emit
	 * skipped it otherwise) — so a GRU-off graph runs with n_inputs-1. */
	{
		size_t n_in_run = codec->n_inputs - (ctx->has_hidden_input ? 0 : 1);

		/* Request only the known heads THIS graph declares (ctx->out_present).
		 * codec->output_names is the canonical superset; we run a compact set
		 * and scatter the results back into out_values[canonical slot] (absent
		 * slots stay NULL — decode reads each head only when present). An
		 * output the codec doesn't know is never in this set → ignored. */
		for (i = 0; i < codec->n_outputs; ++i) {
			if (!ctx->out_present[i])
				continue;
			bound_names[n_bound] = codec->output_names[i];
			bound_slot[n_bound]  = i;
			++n_bound;
		}
		memset(bound_values, 0, sizeof(bound_values));

		{
			OrtStatus *s = ort->Run(
				ctx->session, NULL,
				codec->input_names, (const OrtValue *const *)in_values, n_in_run,
				bound_names, n_bound, bound_values);
			if (qnn_onnx_set_error_from_ort(ort, s, "Session.Run"))
				goto fail;
		}
	}

	/* Scatter the run results back into the canonical slot array so decode
	 * can read each head by its fixed QNN_ONNX_OUT_* index. */
	for (i = 0; i < n_bound; ++i)
		out_values[bound_slot[i]] = bound_values[i];

	if (codec->decode(ctx, out_values, result, out) != 0)
		goto fail;
	rc = 0;

fail:
	for (i = 0; i < codec->n_inputs;  ++i) if (in_values[i])  ort->ReleaseValue(in_values[i]);
	for (i = 0; i < codec->n_outputs; ++i) if (out_values[i]) ort->ReleaseValue(out_values[i]);
	return rc;
}
