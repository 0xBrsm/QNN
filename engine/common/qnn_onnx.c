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
 *   - wire.9 codec: scratch packing, input bind, output decode (in-graph MOVE)
 *   - wire.7 codec: legacy packed-float32 emit (shares wire.9 decode)
 *   - Codec registry + load-time selection
 *   - Step (dispatch through ctx->codec->emit / ->decode)
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
/* The codec binds ONLY the obs inputs. Every recurrent state input (the GRU
 * `hidden`, and on wire.9 the MOVE-decode `move_state` / `move_state_rng`) is
 * carried GENERICALLY by the loop-back engine (see qnn_loopback_t below): the
 * engine has zero semantic knowledge of any state tensor — it learns the
 * complete set from the model's `state.loopback` metadata declaration at load.
 * So this is the single per-codec input count: the obs block. */
#define QNN_ONNX_N_INPUTS      QNN_ONNX_N_OBS_INPUTS

/* ACTION output heads only (move/look/fire/weapon). The recurrent-state OUTPUTS
 * (next_hidden / move_state_out / move_state_rng_out) are NOT action heads — the
 * loop-back engine reads them by name and threads them back generically, so they
 * never appear in a codec's output_names. */
#define QNN_ONNX_N_OUTPUTS   4

/* Canonical action-head slot order (the slots in ctx->out_present and the index
 * decode reads each head from). Mirrors a codec's output_names array. */
#define QNN_ONNX_OUT_MOVE          0
#define QNN_ONNX_OUT_LOOK          1
#define QNN_ONNX_OUT_FIRE          2
#define QNN_ONNX_OUT_WEAPON        3

/* ── Loop-back (recurrent state) engine ──────────────────────────────
 *
 * State carrying is GENERIC, OPAQUE, and CONTRACT-DECLARED. The engine does
 * NOT branch on which state tensor is which (no has_hidden / has_move_state),
 * and has no per-tensor handling. The model stamps a `state.loopback`
 * declaration into ONNX metadata; the engine parses it into this table at load.
 *
 * Each entry pairs a recurrent INPUT (in_name) with the OUTPUT that produces its
 * next-tick value (out_name) — exactly the hidden→next_hidden pattern, but for
 * an arbitrary, opaque set of tensors. The engine:
 *   - at load: allocates `buf` by the ORT-reported shape/dtype of `in_name`,
 *     applies `init`;
 *   - each tick: binds `buf` as the `in_name` input, runs, copies the `out_name`
 *     result back into `buf` for the next tick;
 *   - on episode reset: re-applies `init` ONLY where reset == EPISODE.
 *
 * Adding a future state tensor is an export/contract change with ZERO engine
 * change. The init/reset POLICIES are declared per-entry in the metadata:
 *   init  = "zeros" | "<csv of float lane values>" | "entropy"
 *   reset = "episode" | "persist"
 * `entropy` seeds the buffer once at load from wall-clock entropy and (with
 * reset=persist) keeps it across episodes — the RNG-stream case. */
typedef enum { QNN_LB_INIT_ZEROS, QNN_LB_INIT_CSV, QNN_LB_INIT_ENTROPY } qnn_lb_init_t;
typedef enum { QNN_LB_RESET_EPISODE, QNN_LB_RESET_PERSIST } qnn_lb_reset_t;

#define QNN_LB_MAX_ENTRIES   8
#define QNN_LB_NAME_MAX      48
#define QNN_LB_MAX_CSV_LANES 16

typedef struct {
	char          in_name[QNN_LB_NAME_MAX];
	char          out_name[QNN_LB_NAME_MAX];
	qnn_lb_init_t init;
	qnn_lb_reset_t reset;
	/* CSV init lanes (only when init == CSV). The lanes are written as the
	 * native dtype's element values (cast at apply time). */
	float         csv[QNN_LB_MAX_CSV_LANES];
	size_t        n_csv;
	/* ORT-reported tensor signature for in_name (resolved at load). */
	ONNXTensorElementDataType dtype;
	int64_t       shape[4];
	size_t        n_dims;
	size_t        byte_count;
	size_t        n_elems;
	void         *buf;            /* heap-allocated, byte_count bytes */
} qnn_loopback_t;

/* Wire-contract metadata keys (mirror tools/export_onnx.py stamping +
 * src/qnn/engine_norm.py WIRE_CONTRACT_ID / SEMANTICS_CONTRACT_ID). */
#define QNN_WIRE_CONTRACT_KEY       "wire_contract"
#define QNN_SEMANTICS_CONTRACT_KEY  "semantics_contract"
#define QNN_ARCH_KEY                "arch"
#define QNN_VERSION_KEY             "version"   /* compact a{arch}.s{sem}.w{wire} */
#define QNN_STATE_LOOPBACK_KEY      "state.loopback"  /* recurrent-state declaration */
#define QNN_TICK_HZ_KEY             "tick_hz"   /* REQUIRED policy decision cadence (Hz) */
#define QNN_SEMANTICS_CONTRACT_ID   "semantics.1"

/* Upper bounds for the ORT bind arrays in QNN_OnnxStep. Inputs = the widest
 * obs block (wire.9, 44) + the loop-back inputs (≤ QNN_LB_MAX_ENTRIES).
 * Outputs = the action heads + the loop-back outputs (one per entry). */
#define QNN_ONNX_MAX_INPUTS    (QNN_ONNX_N_INPUTS + QNN_LB_MAX_ENTRIES)
#define QNN_ONNX_MAX_OUTPUTS   (QNN_ONNX_N_OUTPUTS + QNN_LB_MAX_ENTRIES)


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
 *   emit():   fill in_values[0 .. n_inputs-1] with the OBS tensors only
 *             (aliasing ctx-owned scratch; they live until QNN_OnnxStep
 *             releases them after Run). The recurrent STATE inputs are bound
 *             generically by QNN_OnnxStep from the loop-back table — emit knows
 *             nothing about them. Returns 0 on success.
 *   decode(): read the ACTION out_values[0 .. n_outputs-1] into *out. The
 *             recurrent STATE outputs are threaded back generically by
 *             QNN_OnnxStep — decode never sees them. Returns 0 on success.
 *
 * id            = the wire-contract this codec implements ("wire.9").
 * semantics_contract = the semantics contract the codec assumes
 *                 ("semantics.1"); checked against the model's stamp.
 * input_names/n_inputs are the OBS input names emit() binds; output_names/
 * n_outputs are the ACTION-head names. The recurrent STATE I/O (hidden,
 * move_state, …) is NOT in either array — it is carried generically by the
 * loop-back engine. Codec selection matches `id` (the model's wire stamp). */
typedef struct qnn_obs_codec {
	const char         *id;                 /* wire-contract id */
	const char         *semantics_contract; /* assumed semantics id */
	const char *const  *input_names;        /* OBS input set this codec binds */
	size_t              n_inputs;           /* obs only */
	const char *const  *output_names;       /* ACTION-head names */
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

	/* HEAD-AGNOSTIC presence — which of the codec's known ACTION OUTPUT heads
	 * (QNN_ONNX_OUT_* slots) this loaded graph actually declares. Set at
	 * load by enumerating the session's real outputs (qnn_onnx_select_codec);
	 * NO head is required. Step requests only present heads; decode drives
	 * present heads and leaves absent ones uncommanded (see qnn_onnx_decode).
	 * A graph output the codec doesn't know is never requested → ignored. */
	int out_present[QNN_ONNX_N_OUTPUTS];

	/* Resolved WIRE-VERSION major (7 or 9) from the `wire_contract` stamp.
	 * This is the ONE thing the wire version gates: how the `move` output is
	 * INTERPRETED — wire.9 reads a DECIDED 3-axis class (the a24 stateful decode
	 * ran IN-GRAPH), wire.7 takes per-axis argmax of raw move_logits. It does NOT
	 * gate state carrying (that is generic / contract-declared — see the loop-back
	 * table). (The in-graph shape reclaimed wire.9 during a24 dev; there is no
	 * wire.10 — see src/docs/contracts/wire/wire.9.md.) */
	int wire_major;

	/* Policy decision cadence (Hz) from the model's REQUIRED `tick_hz` stamp.
	 * The live client runs inference at this rate (sets qnn_client_fixed_dt =
	 * 1/tick_hz). No default: a model without the stamp is refused at load
	 * (the rate the weights were trained at must travel with them). */
	int tick_hz;

	/* Generic recurrent-state loop-back table — the COMPLETE, OPAQUE set of
	 * state tensors this graph carries, learned from the model's
	 * `state.loopback` metadata at load (NOT from input-name sniffing). The
	 * engine binds each entry's buffer as its in_name input every tick, copies
	 * the paired out_name result back, and re-applies init on reset where the
	 * entry's reset policy says so. Zero entries → stateless graph. */
	qnn_loopback_t loopbacks[QNN_LB_MAX_ENTRIES];
	size_t         n_loopbacks;

	/* arch id from the model's `arch` metadata stamp — provenance only
	 * (shown in the load line; not a selection/validation key). */
	char arch[40];
	/* compact a{arch}.s{sem}.w{wire} render from the `version` stamp (provenance;
	 * empty when the model predates the stamp). */
	char version[32];

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
	 * Only the slot the loaded codec declares is populated.
	 *
	 * MOVE has two forms too:
	 *   wire.7 → `move_logits` (float[3][3]); the engine runs plain per-axis
	 *            argmax (legacy a17/a22 behavior — predate the sticky gate).
	 *   wire.9 → `move_decided` (int64[3]); the a24 stateful decode (sticky gate /
	 *            watermark / hazard / stop-onset + the continuous-fire hold-tail)
	 *            ran IN-GRAPH, so the engine just packs the classes and carries the
	 *            recurrent move_state pair. */
	float    move_logits[3 * 3];                      /* wire.7 (engine argmax) */
	int64_t  move_decided[3];                         /* wire.9 (decided fb/lr/jump class) */
	float    look[3];
	float    fire_logit;                              /* wire.7/.9 (engine decodes attack) —
	                                                     * name FROZEN: mirrors the legacy
	                                                     * `"fire_logit"` output tensor it reads;
	                                                     * wire.11 uses attack_decided below */
	int64_t  attack_decided;                          /* wire.11 (decided attack bit, in-graph) */
	int64_t  weapon_decided;                          /* wire.9 (Pattern A, in-graph gate) */
	float    weapon_logits[QNN_ONNX_WEAPON_CLASSES];  /* wire.7 (engine gate) */
	/* wire.7 sticky-gate thresholds: read from `decode.*` ONNX metadata at load
	 * (Pattern B — params in the model). REQUIRED for wire.7 (no default; a
	 * missing stamp refuses the load). Unused by wire.9 (gate in-graph). */
	float    weapon_switch_confidence;
	float    weapon_switch_margin;
	/* Attack fire operating point: fire when sigmoid(fire_logit) > this. Read from
	 * `decode.attack_threshold` at load (all wire generations); default 0.5 when the
	 * stamp is absent (pre-attack-threshold exports unchanged). Fit offline by
	 * qnn.bc.decode_fit.fit_attack. */
	float    attack_threshold;
	/* Continuous-weapon fire hold-tail (all wire generations; the attack
	 * sigmoid+hold-tail is decoded engine-side from `fire_logit` regardless of
	 * move format). NG/SNG/LG fire from the 0.1s player_nail/player_light QC
	 * think-chain while button0 stays held, but the policy is trained on the
	 * ~0.2s op-fire (W_Attack re-entry) cadence. Armed to QNN_FIRE_HOLD_SEC worth
	 * of decision ticks (round(QNN_FIRE_HOLD_SEC * tick_hz)) on each model fire of
	 * a continuous weapon and counted down per tick; while >0
	 * the attack bit is forced so button0 stays pressed and the server
	 * think-chain streams the in-between nails/bolts. Keyed to the MODEL's fire
	 * (NOT attack_finished) so it cannot feed back into an unbounded hold, and
	 * self-limits ~one op-cadence after the model stops choosing fire. See
	 * src/docs/mvd-fire-audit.md (QWD think-chain). Fixed in f687cb1d.
	 * (wire.9's in-graph decode computes an equivalent hold-tail in move_state but
	 * the export discards that attack bit and still emits fire_logit, so the engine
	 * remains the single attack-decode site for every wire format.) */
	int      fire_hold_ticks;
};


/* ── Lifecycle ──────────────────────────────────────────────────── */

#define QNN_ONNX_CHECK_OR_FAIL(call, where)                                 \
	do {                                                                \
		OrtStatus *_s = (call);                                     \
		if (qnn_onnx_set_error_from_ort(ort, _s, (where)))          \
			goto fail;                                          \
	} while (0)

/* The OrtEnv is process-global: ORT intends one long-lived env per process, so
 * recreating it on every model load (incl. each hot-swap) re-pays runtime init
 * for nothing. Created once, never released (reclaimed at process exit). */
static OrtEnv *g_qnn_env = NULL;

static double qnn_onnx_now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

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

	{
		double t_env0 = qnn_onnx_now_ms();
		double t_env1;
		double t_sess1;

		/* ERROR level: ORT 1.26 logs W-level GPU device-discovery probes
		 * (/sys/class/drm/...) that always "fail" on the CPU-only pi — noise.
		 * Env is process-global (g_qnn_env): created once, reused on reload. */
		if (g_qnn_env == NULL)
			QNN_ONNX_CHECK_OR_FAIL(ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "qnn_onnx", &g_qnn_env), "CreateEnv");
		ctx->env = g_qnn_env;
		t_env1 = qnn_onnx_now_ms();

		QNN_ONNX_CHECK_OR_FAIL(ort->CreateSessionOptions(&ctx->opts), "CreateSessionOptions");
		QNN_ONNX_CHECK_OR_FAIL(ort->SetIntraOpNumThreads(ctx->opts, 1), "SetIntraOpNumThreads");
		QNN_ONNX_CHECK_OR_FAIL(ort->CreateSession(ctx->env, onnx_path, ctx->opts, &ctx->session), "CreateSession");
		t_sess1 = qnn_onnx_now_ms();

		QNN_ONNX_CHECK_OR_FAIL(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->meminfo), "CreateCpuMemoryInfo");

		/* Diagnostic only — to the debug log (stderr), never the console. */
		fprintf(stderr, "model: load timing env=%.0fms session=%.0fms\n",
			t_env1 - t_env0, t_sess1 - t_env1);
	}

	/* Choose + validate the obs codec for this model's wire contract. */
	if (qnn_onnx_select_codec(ctx) != 0)
		goto fail;

	/* One clean load line: path + the compact a{arch}.s{sem}.w{wire} version.
	 * Falls back to the raw codec/arch ids for models exported before the
	 * `version` stamp (those carry no combined string). */
	if (ctx->version[0] != '\0')
		Con_Printf("model: loaded %s  %s\n", onnx_path, ctx->version);
	else
		Con_Printf("model: loaded %s  %s/%s (unversioned)\n",
			onnx_path, ctx->codec->id, ctx->arch);

	QNN_OnnxReset(ctx);
	return ctx;

fail:
	QNN_OnnxFree(ctx);
	return NULL;
}

/* Number of elements of a tensor element type, for CSV/zero init. */
static size_t qnn_lb_dtype_bytes(ONNXTensorElementDataType dt)
{
	switch (dt) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 4;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return 4;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return 8;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return 2;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return 2;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return 1;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return 1;
	default:                                    return 0;
	}
}

/* Write one element value `v` (as the entry's native dtype) at element index i. */
static void qnn_lb_write_elem(const qnn_loopback_t *lb, size_t i, double v)
{
	void *base = lb->buf;
	switch (lb->dtype) {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: ((float *)base)[i]   = (float)v;   break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: ((int64_t *)base)[i] = (int64_t)v; break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: ((int32_t *)base)[i] = (int32_t)v; break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: ((int16_t *)base)[i] = (int16_t)v; break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:((uint16_t *)base)[i]= (uint16_t)v;break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:  ((int8_t *)base)[i]  = (int8_t)v;  break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: ((uint8_t *)base)[i] = (uint8_t)v; break;
	default: break;
	}
}

/* Apply a loop-back entry's INIT policy to its buffer. `seed_entropy` chooses
 * whether ENTROPY entries are (re)seeded — true once at load, false on episode
 * reset (ENTROPY rides with reset=persist, so the engine never re-seeds it).
 *   ZEROS   — memset 0.
 *   CSV     — write the declared lane values (broadcast/tiled across the buffer
 *             if the buffer holds more elements than declared lanes).
 *   ENTROPY — seed each element from a wall-clock-mixed nonzero value (only when
 *             seed_entropy); a 0 draw is bumped to a fixed nonzero constant so an
 *             in-graph xorshift never starts dead. */
static void qnn_lb_apply_init(qnn_loopback_t *lb, int seed_entropy)
{
	size_t i;
	if (lb->buf == NULL) return;
	switch (lb->init) {
	case QNN_LB_INIT_ZEROS:
		memset(lb->buf, 0, lb->byte_count);
		break;
	case QNN_LB_INIT_CSV:
		for (i = 0; i < lb->n_elems; ++i) {
			double v = (lb->n_csv > 0) ? lb->csv[i % lb->n_csv] : 0.0;
			qnn_lb_write_elem(lb, i, v);
		}
		break;
	case QNN_LB_INIT_ENTROPY:
		if (!seed_entropy)
			break;   /* persist: keep the load-time seed across reset */
		for (i = 0; i < lb->n_elems; ++i) {
			uint64_t mix = ((uint64_t)(unsigned)time(NULL) ^ 0xA5A5A5A5u)
			             + (uint64_t)(i + 1) * 0x9E3779B9u;
			uint32_t s = (uint32_t)(mix & 0xFFFFFFFFu);
			if (s == 0) s = 0x9E3779B9u;
			qnn_lb_write_elem(lb, i, (double)s);
		}
		break;
	}
}

void QNN_OnnxReset(qnn_onnx_ctx_t *ctx)
{
	size_t k;
	if (ctx == NULL) return;
	/* Re-apply init to every EPISODE-reset loop-back buffer; PERSIST entries
	 * (e.g. an RNG stream) are left untouched. The engine has NO idea what any
	 * of these tensors mean — it just honors the declared policy. */
	for (k = 0; k < ctx->n_loopbacks; ++k)
		if (ctx->loopbacks[k].reset == QNN_LB_RESET_EPISODE)
			qnn_lb_apply_init(&ctx->loopbacks[k], /*seed_entropy=*/0);
	ctx->fire_hold_ticks = 0;   /* no in-flight continuous-fire burst */
}

void QNN_OnnxFree(qnn_onnx_ctx_t *ctx)
{
	size_t k;
	if (ctx == NULL) return;
	for (k = 0; k < QNN_LB_MAX_ENTRIES; ++k)
		free(ctx->loopbacks[k].buf);
	if (ctx->ort != NULL) {
		if (ctx->meminfo) ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
		if (ctx->session) ctx->ort->ReleaseSession(ctx->session);
		if (ctx->opts)    ctx->ort->ReleaseSessionOptions(ctx->opts);
		/* ctx->env aliases the process-global g_qnn_env — never released here. */
	}
	free(ctx);
}

int QNN_OnnxTickHz(const qnn_onnx_ctx_t *ctx)
{
	return ctx ? ctx->tick_hz : 0;
}


/* ════════════════════════════════════════════════════════════════════
 *  wire.9 codec — native split, 44 obs + in-graph MOVE decode (current;
 *  v24 / full_4head, HEAD)
 *
 *  The current native contract: per-field native dtypes, model-side
 *  dequant, and the a24 stateful MOVE decode baked IN-GRAPH (so `move` is
 *  the DECIDED 3-axis class and the recurrent move_state pair is threaded
 *  generically by the loop-back engine). See src/docs/contracts/wire/wire.9.md.
 *  This is the table + pack + bind + decode behind qnn_obs_codec_t — the
 *  obs block is byte-for-byte the native 44-input split.
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

/* Input names — the OBS block only (mirrors QNN_ONNX_INPUTS). The recurrent
 * state inputs (`hidden`, and on wire.9 `move_state` / `move_state_rng`) are
 * NOT here: they are carried generically by the loop-back engine, which learns
 * the full set from the `state.loopback` metadata at load and binds each by name
 * (qnn_onnx_select_codec / QNN_OnnxStep). */
static const char *QNN_ONNX_INPUT_NAMES[QNN_ONNX_N_OBS_INPUTS] = {
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
};

/* ACTION-head output names — SELF-DECLARING + HEAD-AGNOSTIC: a loaded graph
 * carries a tensor only for the heads it has, NO head is required. Load
 * enumerates the session's real outputs and records per-slot presence on
 * ctx->out_present[]; an output name the codec doesn't know is simply not
 * requested by Step → harmlessly ignored. Index order here is the canonical
 * slot order (== QNN_ONNX_OUT_*) that decode reads; absent slots stay NULL.
 *
 * The recurrent-state OUTPUTS (next_hidden / move_state_out / move_state_rng_out)
 * are NOT here — they are loop-back outputs, threaded back generically by the
 * loop-back engine, never an action head.
 *
 * The wire generations differ in two slots:
 *   - move:   wire.9 emits a DECIDED `move` (int64 (B,3) fb/lr/jump class — the
 *             a24 stateful decode ran in-graph); wire.7 emits raw `move_logits`
 *             (float (B,3,3)) and the engine runs plain argmax (gated on
 *             ctx->wire_major — action interpretation only).
 *   - weapon: wire.9 emits a DECIDED `weapon` impulse (int64; sticky gate
 *             in-graph), wire.7 emits raw `weapon_logits` (float[8]; engine runs
 *             the controller). */
/* FROZEN LEGACY NAMES — do NOT rename to chase the `attack` (not `fire`)
 * convention. `"fire_logit"` here is the ACTUAL output-tensor name that pre-wire.11
 * models export; the engine binds outputs by name, so this string must keep
 * matching what those archived models (rc1v, milestone-demo bots, …) emit. The
 * convention rename already landed on the wire at wire.11 (the FIRE slot became
 * the decided `attack` bit — see _W11 below). These W7/W9 entries are not tech
 * debt; they are the correct historical names. Leaving them frozen is the fix. */
static const char *QNN_ONNX_OUTPUT_NAMES_W9[QNN_ONNX_N_OUTPUTS] = {
	"move", "look", "fire_logit", "weapon",
};
/* wire.11 = wire.9 + the in-graph ATTACK decode: the FIRE slot is the DECIDED
 * `attack` bit (int64) rather than raw `fire_logit` (float). The engine reads the
 * bit and ORs it into the press byte; no engine-side sigmoid/threshold/hold-tail. */
static const char *QNN_ONNX_OUTPUT_NAMES_W11[QNN_ONNX_N_OUTPUTS] = {
	"move", "look", "attack", "weapon",
};
static const char *QNN_ONNX_OUTPUT_NAMES_W7[QNN_ONNX_N_OUTPUTS] = {
	"move_logits", "look", "fire_logit", "weapon_logits",
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

/* Continuous-weapon (NG/SNG/LG) hold-tail, as a WALL-CLOCK duration: bridges the
 * model's ~0.2s op-fire cadence plus a little sampling jitter so a sustained
 * burst never drops button0 between the model's fires (which would stall the
 * server's player_nail/player_light think-chain). The tick COUNT is derived
 * from the model's stamped decision cadence (ctx->tick_hz) so it's the same
 * wall-clock hold at any rate — 5 ticks @20Hz, 2-3 @10Hz — instead of a
 * tick-rate-varying hardcoded constant. Over-extension on disengage is bounded
 * to this duration. */
#define QNN_FIRE_HOLD_SEC 0.25f

/* Shared move/look/fire decode. The MOVE source depends on the wire generation
 * (a24 move-decode migration, STEP 3 — the stateful gate now lives in the ONNX
 * graph, no longer here):
 *   wire.9 — `move` is the DECIDED 3-axis class (int64[3], fb/lr/jump) the
 *             in-graph a24 decode produced (sticky gate + switch-back watermark
 *             + hazard + stop-onset, threaded through the recurrent move-decode
 *             state the loop-back engine carries). The engine just packs classes.
 *   wire.7 (legacy logit-move) — `move` is raw move_logits[3][3]; the engine
 *             runs PLAIN per-axis argmax (all 3 axes), the original
 *             pre-2026-06-10 behavior (a17/a22 predate the sticky gate).
 * This is the ONE behavior the wire version gates (ctx->wire_major) — ACTION
 * INTERPRETATION only, never state carrying.
 * ATTACK + look are identical across generations: the attack sigmoid+threshold
 * and the continuous-weapon hold-tail are decoded engine-side from `fire_logit`
 * regardless of move format, and look is just clamped. Weapon is decoded
 * per-codec by the caller (wire.9/.10 passthrough int vs wire.7 controller), so
 * it is NOT touched here. ctx is mutable: the hold-tail updates fire_hold_ticks
 * across ticks. */
static void qnn_onnx_decode_core(qnn_onnx_ctx_t *ctx, qnn_action_t *out)
{
	int axis, c, best, i;
	float p_fire;

	/* HEAD-AGNOSTIC decode: each head drives its part of the action only
	 * when the graph declares it; an absent head leaves that part
	 * uncommanded. move + fire share the one packed input mask (alive is
	 * always set), so they are decoded together: an absent move yields zero
	 * movement bits, an absent fire yields no attack. */
	{
		int axis_signs[3] = {0, 0, 0};
		int fb_neg, fb_pos, lr_neg, lr_pos, up_neg, up_pos, attack_bit;

		/* ---- move: 3 axes × 3 classes (neg/none/pos). Absent → no move. */
		if (ctx->out_present[QNN_ONNX_OUT_MOVE]) {
			if (ctx->wire_major >= 9) {
				/* wire.9: classes already decided in-graph; just read. */
				for (axis = 0; axis < 3; ++axis) {
					int cls = (int)ctx->move_decided[axis];   /* 0,1,2 */
					axis_signs[axis] = cls - 1;               /* → -1,0,+1 */
				}
			} else {
				/* wire.7: plain per-axis argmax of the raw logits (legacy;
				 * all 3 axes). Ties → first max (c scans 1,2 with
				 * strict '>' starting best=0). */
				for (axis = 0; axis < 3; ++axis) {
					const float *row = &ctx->move_logits[axis * 3];
					float best_v = row[0];
					best = 0;
					for (c = 1; c < 3; ++c) {
						if (row[c] > best_v) { best_v = row[c]; best = c; }
					}
					axis_signs[axis] = best - 1;   /* class 0,1,2 → -1,0,+1 */
				}
			}
		}
		fb_neg = axis_signs[0] < 0 ? 1 : 0;
		fb_pos = axis_signs[0] > 0 ? 1 : 0;
		lr_neg = axis_signs[1] < 0 ? 1 : 0;
		lr_pos = axis_signs[1] > 0 ? 1 : 0;
		up_neg = axis_signs[2] < 0 ? 1 : 0;
		up_pos = axis_signs[2] > 0 ? 1 : 0;
		/* ---- attack: fire sigmoid > ctx->attack_threshold (stamped
		 * decode.attack_threshold; default 0.5). Absent head → no attack. ---- */
		attack_bit = 0;
		if (ctx->out_present[QNN_ONNX_OUT_FIRE] && ctx->wire_major >= 11) {
			/* wire.11: attack is DECIDED in-graph (its own decode + hold-tail
			 * baked, attack_bias applied) — just read the bit. No engine-side
			 * sigmoid/threshold/hold-tail. */
			attack_bit = (int)ctx->attack_decided ? 1 : 0;
		} else if (ctx->out_present[QNN_ONNX_OUT_FIRE]) {
			/* Held weapon (ENTITY_IDS-encoded). Read the format-neutral
			 * self_weapon_id, which BOTH wire packers set every tick — NOT the
			 * wire.7-only model-input buffer w7_self_weapon_id, which is never
			 * populated on wire.9 (the deployed format) and so left this
			 * continuous-fire hold-tail permanently disarmed there. */
			int wid = ctx->self_weapon_id;
			/* Hold-tail covers all three continuous weapons.  Each one's
			 * think-chain (player_nail* / player_light*) re-fires every 0.1s while
			 * button0 is held, but the model's op-fire tracks the ~0.2s W_Attack
			 * re-entry cadence, so the engine holds button0 to let the chain
			 * stream the in-between shots:
			 *   NG/SNG: think 0.1s; W_Fire{Super,}Spikes set attack_finished +0.2.
			 *   LG:     think 0.1s; W_Attack literally sets +0.1, BUT the measured
			 *           op-fire cadence is 0.2s (QWD LG op-attack lands every ~4
			 *           frames @20Hz -- the player_light think-chain owns the
			 *           0.1s bolts).  Same gap as the nailguns, so LG needs the
			 *           hold-tail too.  (Restores THUNDERBOLT; the bfdb04d7
			 *           removal was premised on the 0.1s literal, which the
			 *           cadence data disproved.) */
			int is_continuous =
				(wid == QNN_SUBJECT_NAILGUN
				 || wid == QNN_SUBJECT_SUPER_NAILGUN
				 || wid == QNN_SUBJECT_THUNDERBOLT);
			p_fire = 1.0f / (1.0f + expf(-ctx->fire_logit));
			attack_bit = (p_fire > ctx->attack_threshold) ? 1 : 0;
			/* Continuous-weapon hold-tail: keep button0 pressed across the
			 * model's op-fire gaps so the server think-chain keeps streaming
			 * nails. Armed by the model's own fire; self-limiting. */
			if (is_continuous && attack_bit) {
				/* hold-tail length = wall-clock QNN_FIRE_HOLD_SEC at the model's
				 * decision cadence (≥1 tick); tick_hz is the stamped, validated
				 * cadence (no default) read at load. */
				int hold = (int)lroundf(QNN_FIRE_HOLD_SEC * (float)ctx->tick_hz);
				ctx->fire_hold_ticks = hold > 0 ? hold : 1;
			} else if (is_continuous && ctx->fire_hold_ticks > 0) {
				attack_bit = 1;
				ctx->fire_hold_ticks--;
			} else
				ctx->fire_hold_ticks = 0;
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

/* Bind the 44 obs tensors (via QNN_ONNX_INPUTS) into in_values[0 ..
 * QNN_ONNX_N_OBS_INPUTS-1]; tensors alias ctx scratch. The recurrent STATE
 * inputs (hidden / move_state / move_state_rng / …) are bound generically by
 * QNN_OnnxStep from the loop-back table — emit knows nothing about them. */
static int wire9_emit(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result,
                      OrtValue **in_values)
{
	const OrtApi *ort = ctx->ort;
	size_t i;

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
	return 0;
}

/* Read the shared ACTION outputs (move/look/fire) into ctx scratch. Weapon is
 * read per-codec by the caller (its tensor differs: wire.9 `weapon` int64 vs
 * wire.7 `weapon_logits` float[8]). HEAD-AGNOSTIC: read a head only when this
 * graph declares it. The recurrent STATE outputs (next_hidden / move_state_out
 * / …) are threaded back generically by QNN_OnnxStep from the loop-back table —
 * this function never touches them. */
static int qnn_onnx_extract_core(qnn_onnx_ctx_t *ctx, OrtValue *const *out_values)
{
	const OrtApi *ort = ctx->ort;
	void *p;
	OrtStatus *s;

	if (ctx->out_present[QNN_ONNX_OUT_MOVE]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_MOVE], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(move)")) return 1;
		if (ctx->wire_major >= 9)
			/* wire.9: `move` is the DECIDED 3-axis class (int64 (B,3)). */
			memcpy(ctx->move_decided, p, sizeof(ctx->move_decided));
		else
			/* wire.7: raw move_logits (float (B,3,3)). */
			memcpy(ctx->move_logits, p, sizeof(ctx->move_logits));
	}
	if (ctx->out_present[QNN_ONNX_OUT_LOOK]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_LOOK], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(look)")) return 1;
		memcpy(ctx->look, p, sizeof(ctx->look));
	}
	if (ctx->out_present[QNN_ONNX_OUT_FIRE]) {
		s = ort->GetTensorMutableData(out_values[QNN_ONNX_OUT_FIRE], &p);
		if (qnn_onnx_set_error_from_ort(ort, s, "Get(attack/fire_logit)")) return 1;
		if (ctx->wire_major >= 11)
			ctx->attack_decided = ((int64_t *)p)[0];   /* wire.11: decided bit */
		else
			ctx->fire_logit = ((float *)p)[0];         /* wire.7/.9: raw logit */
	}
	return 0;
}

/* Decided-weapon decode for wire.9: `weapon` is the DECIDED impulse int (sticky
 * gate ran in-graph, Pattern A) — pass it through. The MOVE handling (in-graph
 * decided classes) lives in qnn_onnx_extract_core / qnn_onnx_decode_core, keyed
 * on ctx->wire_major. Independent of wire.7 (which emits raw logits and runs the
 * engine-side weapon controller). */
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

/* wire.9 — the CURRENT contract: native 44-obs split, declaring the DECIDED
 * `move` class (the a24 stateful move decode runs IN-GRAPH) + the decided
 * `weapon` impulse (sticky gate in-graph, Pattern A). Its recurrent MOVE-decode
 * state pair (move_state / move_state_rng) is carried GENERICALLY by the
 * loop-back engine from the model's `state.loopback` metadata — NOT special-cased
 * here. The move difference vs the legacy logit-move path (wire.7) is keyed on
 * ctx->wire_major in extract/decode_core — ACTION interpretation only.
 *
 * (The in-graph shape reclaimed the wire.9 number during active a24 development
 * — a wire.10 was distinguished briefly but never finalized as a release, and
 * the old engine-side-argmax wire.9 has no surviving artifact, so there is no
 * wire.10 codec. See src/docs/contracts/wire/wire.9.md.) */
static const qnn_obs_codec_t QNN_CODEC_WIRE_9 = {
	"wire.9",
	QNN_SEMANTICS_CONTRACT_ID,
	QNN_ONNX_INPUT_NAMES,  QNN_ONNX_N_OBS_INPUTS,
	QNN_ONNX_OUTPUT_NAMES_W9, QNN_ONNX_N_OUTPUTS,
	wire9_emit,
	wire9_decode,
};

/* wire.11 = wire.9 + the in-graph ATTACK decode (REPLACES wire.9 for the a24 gen).
 * Same obs pack + emit + recurrent-state plumbing as wire.9 (so it reuses
 * wire9_emit / wire9_decode); the ONLY differences are the FIRE output slot — the
 * DECIDED `attack` bit (int64) instead of `fire_logit` (float), keyed on
 * ctx->wire_major >= 11 in wire9_decode + qnn_onnx_decode_core — and the
 * attack_state recurrent tensor (carried generically via state.loopback, no codec
 * knowledge). With this every action is decided in-graph; the engine runs no
 * attack sigmoid/threshold/hold-tail at wire.11. */
static const qnn_obs_codec_t QNN_CODEC_WIRE_11 = {
	"wire.11",
	QNN_SEMANTICS_CONTRACT_ID,
	QNN_ONNX_INPUT_NAMES,  QNN_ONNX_N_OBS_INPUTS,
	QNN_ONNX_OUTPUT_NAMES_W11, QNN_ONNX_N_OUTPUTS,
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
 *  archive (research/archive/token-spec-v11.md), and git show
 *  b0f75210^:src/qnn/wire.py.
 *
 *  ID PROMOTION: the packed wire stored IDs as int32; the ONNX graph
 *  inputs are int64. This codec promotes (the w7_* scratch is int64).
 *
 *  Outputs match wire.9 on the look/fire slots but differ on move + weapon:
 *  wire.7 emits raw `move_logits` (float[3][3]; engine runs per-axis argmax)
 *  and raw `weapon_logits` (float[8]; engine runs the sticky controller,
 *  wire7_decode), whereas wire.9 emits a DECIDED `move` class (a24 decode
 *  in-graph) and a decided `weapon` int (gate in-graph). The two are otherwise
 *  independent — own output table + decode. v17's `weapon_logits` second dim is
 *  a symbolic export artifact; semantically 8 (== QNN_ONNX_WEAPON_CLASSES)
 *  either way.
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

/* wire.7 OBS inputs only — `hidden` is carried generically by the loop-back
 * engine (declared in the model's `state.loopback` metadata), not here. */
static const char *QNN_WIRE7_INPUT_NAMES[] = {
	"self_scalars", "self_weapon_id", "self_armor_type_id", "self_movement_id",
	"self_powerup_ids", "entity_types", "entity_scalars_raw", "entity_ids",
	"entity_event_actions", "entity_event_sources", "entity_event_counts",
	"spatial_scalars",
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
	/* Also fill the format-neutral copy the shared decode reads for the
	 * continuous-fire hold-tail (wire.9's pack_scratch sets it too). */
	ctx->self_weapon_id      = (uint8_t)self->weapon_id;
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

	/* (name, ptr, byte_count, {shape}, n_dims, dtype) per wire.7 OBS input.
	 * `hidden` is bound generically by QNN_OnnxStep (loop-back table). */
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
	&QNN_CODEC_WIRE_11,
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

/* Resolve the wire-version MAJOR (7 or 9) from the `wire.N` contract id.
 * Returns 0 on an unrecognized form. Used ONLY to gate ACTION interpretation
 * (the move output: wire.9 decided classes vs wire.7 raw logits) — never state
 * carrying. */
static int qnn_onnx_wire_major(const char *wire_id)
{
	if (wire_id == NULL) return 0;
	if (strncmp(wire_id, "wire.", 5) == 0)
		return atoi(wire_id + 5);
	return 0;
}

/* Copy a trimmed token [b,e) into dst (NUL-terminated, capped). */
static void qnn_lb_copy_token(char *dst, size_t cap, const char *b, const char *e)
{
	size_t n;
	while (b < e && (*b == ' ' || *b == '\t')) ++b;
	while (e > b && (e[-1] == ' ' || e[-1] == '\t')) --e;
	n = (size_t)(e - b);
	if (n >= cap) n = cap - 1;
	memcpy(dst, b, n);
	dst[n] = '\0';
}

/* Parse the `state.loopback` declaration string into ctx->loopbacks[]. Grammar:
 *   entry := field (',' field)*            entries separated by ';'
 *   field := key '=' value
 *   keys  := in | out | init | reset
 *   init  := "zeros" | "entropy" | "<space-separated float lanes>"
 *   reset := "episode" | "persist"
 * Returns 0 on success, non-zero (error set) on a malformed declaration or
 * overflow. An empty/NULL declaration is valid → zero loop-back entries
 * (a stateless graph). */
static int qnn_loopback_parse(qnn_onnx_ctx_t *ctx, const char *decl)
{
	const char *p;
	ctx->n_loopbacks = 0;
	if (decl == NULL || decl[0] == '\0')
		return 0;
	p = decl;
	while (*p) {
		const char *entry_end = strchr(p, ';');
		const char *fp;
		qnn_loopback_t lb;
		int have_in = 0, have_out = 0;
		if (entry_end == NULL) entry_end = p + strlen(p);
		memset(&lb, 0, sizeof(lb));
		lb.init = QNN_LB_INIT_ZEROS;
		lb.reset = QNN_LB_RESET_EPISODE;
		fp = p;
		while (fp < entry_end) {
			const char *field_end = fp;
			const char *eq;
			while (field_end < entry_end && *field_end != ',') ++field_end;
			eq = fp;
			while (eq < field_end && *eq != '=') ++eq;
			if (eq < field_end) {
				char key[16];
				qnn_lb_copy_token(key, sizeof(key), fp, eq);
				const char *vb = eq + 1, *ve = field_end;
				if (strcmp(key, "in") == 0) {
					qnn_lb_copy_token(lb.in_name, sizeof(lb.in_name), vb, ve);
					have_in = lb.in_name[0] != '\0';
				} else if (strcmp(key, "out") == 0) {
					qnn_lb_copy_token(lb.out_name, sizeof(lb.out_name), vb, ve);
					have_out = lb.out_name[0] != '\0';
				} else if (strcmp(key, "reset") == 0) {
					char v[16];
					qnn_lb_copy_token(v, sizeof(v), vb, ve);
					lb.reset = (strcmp(v, "persist") == 0)
						? QNN_LB_RESET_PERSIST : QNN_LB_RESET_EPISODE;
				} else if (strcmp(key, "init") == 0) {
					char v[256];
					qnn_lb_copy_token(v, sizeof(v), vb, ve);
					if (strcmp(v, "zeros") == 0) {
						lb.init = QNN_LB_INIT_ZEROS;
					} else if (strcmp(v, "entropy") == 0) {
						lb.init = QNN_LB_INIT_ENTROPY;
					} else {
						/* CSV of space-separated float lanes. */
						char *tok, *save = NULL;
						lb.init = QNN_LB_INIT_CSV;
						lb.n_csv = 0;
						for (tok = strtok_r(v, " ", &save); tok != NULL;
						     tok = strtok_r(NULL, " ", &save)) {
							if (lb.n_csv >= QNN_LB_MAX_CSV_LANES) {
								qnn_onnx_set_error(
									"state.loopback: too many init lanes for '%s' (max %d)",
									lb.in_name, QNN_LB_MAX_CSV_LANES);
								return 1;
							}
							lb.csv[lb.n_csv++] = (float)atof(tok);
						}
					}
				}
				/* unknown keys are ignored (forward-compat) */
			}
			fp = field_end + 1;
		}
		if (!have_in || !have_out) {
			qnn_onnx_set_error("state.loopback: entry missing in= or out=");
			return 1;
		}
		if (ctx->n_loopbacks >= QNN_LB_MAX_ENTRIES) {
			qnn_onnx_set_error("state.loopback: too many entries (max %d)",
				QNN_LB_MAX_ENTRIES);
			return 1;
		}
		ctx->loopbacks[ctx->n_loopbacks++] = lb;
		p = (*entry_end == ';') ? entry_end + 1 : entry_end;
	}
	return 0;
}

/* Resolve each loop-back entry's in_name against the session's REAL inputs,
 * record its ORT-reported dtype + shape + byte size, allocate its buffer, and
 * apply its INIT (entropy seeded ONCE here at load). Also validates that the
 * paired out_name is a real graph OUTPUT. Returns 0 on success, non-zero (error
 * set) if a declared loop-back tensor is missing from the graph I/O. */
static int qnn_loopback_resolve(qnn_onnx_ctx_t *ctx, OrtAllocator *alloc,
                                const char *const *in_names, size_t n_in,
                                const char *const *out_names, size_t n_out)
{
	const OrtApi *ort = ctx->ort;
	size_t k, i;
	for (k = 0; k < ctx->n_loopbacks; ++k) {
		qnn_loopback_t *lb = &ctx->loopbacks[k];
		int in_idx = -1;
		int out_found = 0;
		OrtTypeInfo *ti = NULL;
		const OrtTensorTypeAndShapeInfo *tinfo = NULL;
		OrtStatus *s;
		size_t bytes;

		for (i = 0; i < n_in; ++i)
			if (strcmp(in_names[i], lb->in_name) == 0) { in_idx = (int)i; break; }
		if (in_idx < 0) {
			qnn_onnx_set_error(
				"state.loopback declares input '%s' but the graph has no such input "
				"(stamp/graph mismatch — refusing)", lb->in_name);
			return 1;
		}
		for (i = 0; i < n_out; ++i)
			if (strcmp(out_names[i], lb->out_name) == 0) { out_found = 1; break; }
		if (!out_found) {
			qnn_onnx_set_error(
				"state.loopback declares output '%s' but the graph has no such output "
				"(stamp/graph mismatch — refusing)", lb->out_name);
			return 1;
		}

		/* ORT-reported shape/dtype for in_name → buffer size. */
		s = ort->SessionGetInputTypeInfo(ctx->session, (size_t)in_idx, &ti);
		if (qnn_onnx_set_error_from_ort(ort, s, "SessionGetInputTypeInfo(loopback)"))
			return 1;
		s = ort->CastTypeInfoToTensorInfo(ti, &tinfo);
		if (qnn_onnx_set_error_from_ort(ort, s, "CastTypeInfoToTensorInfo(loopback)")) {
			ort->ReleaseTypeInfo(ti); return 1;
		}
		s = ort->GetTensorElementType(tinfo, &lb->dtype);
		if (qnn_onnx_set_error_from_ort(ort, s, "GetTensorElementType(loopback)")) {
			ort->ReleaseTypeInfo(ti); return 1;
		}
		s = ort->GetDimensionsCount(tinfo, &lb->n_dims);
		if (qnn_onnx_set_error_from_ort(ort, s, "GetDimensionsCount(loopback)")) {
			ort->ReleaseTypeInfo(ti); return 1;
		}
		if (lb->n_dims > 4) lb->n_dims = 4;
		s = ort->GetDimensions(tinfo, lb->shape, lb->n_dims);
		if (qnn_onnx_set_error_from_ort(ort, s, "GetDimensions(loopback)")) {
			ort->ReleaseTypeInfo(ti); return 1;
		}
		ort->ReleaseTypeInfo(ti);

		/* Element count from the resolved shape (a symbolic/-1 batch dim is
		 * pinned to 1 — the live client always runs batch=1). */
		lb->n_elems = 1;
		for (i = 0; i < lb->n_dims; ++i) {
			int64_t d = lb->shape[i];
			if (d <= 0) d = 1;
			lb->shape[i] = d;
			lb->n_elems *= (size_t)d;
		}
		bytes = qnn_lb_dtype_bytes(lb->dtype);
		if (bytes == 0) {
			qnn_onnx_set_error("state.loopback: unsupported dtype for '%s'", lb->in_name);
			return 1;
		}
		lb->byte_count = lb->n_elems * bytes;
		lb->buf = calloc(1, lb->byte_count);
		if (lb->buf == NULL) {
			qnn_onnx_set_error("state.loopback: out of memory allocating '%s'", lb->in_name);
			return 1;
		}
		qnn_lb_apply_init(lb, /*seed_entropy=*/1);
	}
	return 0;
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
	 * travels with the weights.
	 *
	 * weapon_switch_confidence/margin are consumed ONLY by the wire.7
	 * engine-applied weapon controller (Pattern B). wire.9 bakes the weapon gate
	 * in-graph (Pattern A), so for it these are unused provenance. There is NO
	 * hardcoded default: a model-varying decode param must travel in the ONNX, so
	 * a wire.7 model missing the stamp is REFUSED (the tick_hz precedent) rather
	 * than silently miscalibrated by a worker-side constant. For wire.9 the field
	 * is irrelevant (gate in-graph); record the stamp if present, else 0.
	 *
	 * The MOVE decode params (move_sticky_tau / move_switchback_eps / the hazard
	 * tables / move_stop_onset) are NOT read here: on wire.9 the a24 stateful MOVE
	 * decode runs ENTIRELY IN-GRAPH (params bake in as graph constants —
	 * tools/export_onnx.py ExportWrapper), so the engine has no move state machine
	 * to parameterize. They are STAMPED for provenance but the engine ignores
	 * them. The legacy wire.7 logit-move path is plain per-axis argmax (no
	 * tunable), the pre-2026-06-10 behavior. */
	if (qnn_onnx_wire_major(wire_stamp) == 7) {
		char *wsc = qnn_onnx_metadata_lookup(ctx, alloc, "decode.weapon_switch_confidence");
		char *wsm = qnn_onnx_metadata_lookup(ctx, alloc, "decode.weapon_switch_margin");
		if (wsc == NULL || wsm == NULL) {
			if (wsc) (void)ort->AllocatorFree(alloc, wsc);
			if (wsm) (void)ort->AllocatorFree(alloc, wsm);
			qnn_onnx_set_error(
				"wire.7 model missing decode.weapon_switch_confidence/margin — "
				"refusing (the weapon-controller thresholds must travel with the "
				"weights, no default); re-export or stamp them.");
			goto cleanup;
		}
		ctx->weapon_switch_confidence = (float)atof(wsc);
		ctx->weapon_switch_margin     = (float)atof(wsm);
		(void)ort->AllocatorFree(alloc, wsc);
		(void)ort->AllocatorFree(alloc, wsm);
	} else {
		/* wire.9: in-graph gate — these are unused provenance (0 if unstamped). */
		ctx->weapon_switch_confidence =
			qnn_onnx_decode_param(ctx, alloc, "weapon_switch_confidence", 0.0f);
		ctx->weapon_switch_margin =
			qnn_onnx_decode_param(ctx, alloc, "weapon_switch_margin", 0.0f);
	}

	/* Attack fire operating point — ALL wire generations (attack is decoded
	 * engine-side from fire_logit regardless of move format). Default 0.5 (the
	 * historical cut) when `decode.attack_threshold` is absent, so pre-attack-
	 * threshold exports are byte-unchanged. Fit via qnn.bc.decode_fit.fit_attack. */
	ctx->attack_threshold = qnn_onnx_decode_param(ctx, alloc, "attack_threshold", 0.5f);

	/* The recurrent-state RNG (move_state_rng) is no longer seeded by a named
	 * special case here: it is one OPAQUE loop-back entry declared with
	 * init=entropy / reset=persist in the model's `state.loopback` metadata, and
	 * qnn_loopback_resolve seeds it once at load (see below). */

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

	/* Resolve the wire-version MAJOR — the ONE thing the wire version gates is
	 * the ACTION (move) interpretation; state carrying is contract-declared. */
	ctx->wire_major = qnn_onnx_wire_major(wire_stamp);

	/* (1b) The model MUST declare its decision cadence — the rate the weights
	 * were trained at. No default: an unstamped model is refused (a wrong
	 * cadence is silent miscalibration of every prediction/lead computation).
	 * Stamp it with tools/stamp_onnx.py --tick-hz <hz> (or re-export). The live
	 * client runs inference at this rate (qnn_client_fixed_dt = 1/tick_hz). */
	{
		char *hz_stamp = qnn_onnx_metadata_lookup(ctx, alloc, QNN_TICK_HZ_KEY);
		int hz = hz_stamp ? atoi(hz_stamp) : 0;
		if (hz_stamp) (void)ort->AllocatorFree(alloc, hz_stamp);
		if (hz <= 0) {
			qnn_onnx_set_error(
				"model has no valid `tick_hz` stamp — refusing (the decision "
				"cadence must travel with the weights, no default); stamp it "
				"with tools/stamp_onnx.py --tick-hz <hz>.");
			goto cleanup;
		}
		ctx->tick_hz = hz;
	}

	/* Parse the `state.loopback` declaration — the COMPLETE set of recurrent
	 * state tensors this graph carries, fully OPAQUE to the engine. Absent →
	 * zero entries (a stateless graph). */
	{
		char *lb_stamp = qnn_onnx_metadata_lookup(ctx, alloc, QNN_STATE_LOOPBACK_KEY);
		int prc = qnn_loopback_parse(ctx, lb_stamp);
		if (lb_stamp) (void)ort->AllocatorFree(alloc, lb_stamp);
		if (prc != 0) goto cleanup;
	}

	/* (2) Validate the session inputs partition cleanly into (a) the codec's
	 * obs inputs and (b) the declared loop-back inputs: every graph input must
	 * be EITHER an obs input the codec produces OR a declared loop-back in_name.
	 * A graph input that is neither is a stale/foreign stamp → refuse. (An obs
	 * input the codec knows but the graph omits is fine — optional obs.) */
	for (i = 0; i < n_in; ++i) {
		size_t j;
		int is_obs = 0, is_lb = 0;
		for (j = 0; j < codec->n_inputs; ++j)
			if (strcmp(codec->input_names[j], in_names[i]) == 0) { is_obs = 1; break; }
		for (j = 0; j < ctx->n_loopbacks; ++j)
			if (strcmp(ctx->loopbacks[j].in_name, in_names[i]) == 0) { is_lb = 1; break; }
		if (!is_obs && !is_lb) {
			qnn_onnx_set_error(
				"wire_contract=%s codec '%s': graph input '%s' is neither an obs "
				"input nor a declared state.loopback input (stale/foreign stamp or "
				"undeclared state tensor — refusing)",
				wire_stamp ? wire_stamp : "(none)", codec->id, in_names[i]);
			goto cleanup;
		}
	}

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

	/* (5) ACTION-I/O version validation: the `move` output the wire version
	 * expects MUST be the one the graph actually declares. wire.9 reads a
	 * DECIDED `move`; wire.7 reads raw `move_logits`. The codec's move
	 * slot name (output_names[QNN_ONNX_OUT_MOVE]) already encodes which the
	 * wire id implies; out_present requires that exact name in the graph. If
	 * the move head is declared at all, it must match — a wire.9 stamp on a
	 * move_logits graph (or vice versa) is a stale stamp → refuse. The move
	 * head MAY be absent entirely (head-agnostic), so only check when some move
	 * output exists in the graph. */
	{
		int graph_has_move = 0, graph_has_move_logits = 0;
		for (i = 0; i < n_out; ++i) {
			if (strcmp(out_names[i], "move") == 0)        graph_has_move = 1;
			if (strcmp(out_names[i], "move_logits") == 0) graph_has_move_logits = 1;
		}
		if (ctx->wire_major >= 9) {
			if (graph_has_move_logits && !graph_has_move) {
				qnn_onnx_set_error(
					"wire_contract=%s expects a DECIDED `move` output but the graph "
					"declares raw `move_logits` (stale stamp — refusing)", wire_stamp);
				goto cleanup;
			}
		} else {
			if (graph_has_move && !graph_has_move_logits) {
				qnn_onnx_set_error(
					"wire_contract=%s expects raw `move_logits` but the graph declares "
					"a DECIDED `move` output (stale stamp — refusing)", wire_stamp);
				goto cleanup;
			}
		}
	}

	/* (6) Resolve the loop-back table against the graph I/O: confirm each
	 * declared in_name/out_name is a real graph input/output, size + allocate
	 * each buffer by its ORT-reported shape/dtype, and apply init (entropy
	 * seeded once here). A declared loop-back tensor missing from the graph is
	 * a hard load error. */
	if (qnn_loopback_resolve(ctx, alloc, in_names, n_in, out_names, n_out) != 0)
		goto cleanup;

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
	/* INPUT side: obs (from emit) followed by the generic loop-back inputs. */
	OrtValue   *in_values[QNN_ONNX_MAX_INPUTS];
	const char *run_in_names[QNN_ONNX_MAX_INPUTS];
	size_t      n_in_run = 0;
	/* ACTION-head outputs scattered back into canonical slots for decode. */
	OrtValue   *out_values[QNN_ONNX_N_OUTPUTS];
	/* The compact set passed to ort->Run: present action heads + every
	 * loop-back output, plus a back-map so we can route each result. */
	const char *bound_names[QNN_ONNX_MAX_OUTPUTS];
	OrtValue   *bound_values[QNN_ONNX_MAX_OUTPUTS];
	/* For each bound output: which action slot it is (or -1 for a loop-back
	 * output, in which case lb_idx names the loop-back entry to copy into). */
	int         bound_action_slot[QNN_ONNX_MAX_OUTPUTS];
	int         bound_lb_idx[QNN_ONNX_MAX_OUTPUTS];
	size_t      n_bound = 0;
	int rc = 1;
	size_t i, k;

	if (ctx == NULL || result == NULL || out == NULL) {
		qnn_onnx_set_error("QNN_OnnxStep: NULL argument");
		return 1;
	}
	codec = ctx->codec;
	if (codec == NULL) {
		qnn_onnx_set_error("QNN_OnnxStep: no codec selected");
		return 1;
	}
	if (codec->n_inputs + ctx->n_loopbacks > QNN_ONNX_MAX_INPUTS
	    || codec->n_outputs + ctx->n_loopbacks > QNN_ONNX_MAX_OUTPUTS) {
		qnn_onnx_set_error("QNN_OnnxStep: codec '%s' exceeds bind capacity", codec->id);
		return 1;
	}
	ort = ctx->ort;
	memset(in_values,  0, sizeof(in_values));
	memset(out_values, 0, sizeof(out_values));

	/* emit binds the OBS inputs as a contiguous prefix of in_values; their
	 * names are the codec's input_names (obs only). */
	if (codec->emit(ctx, result, in_values) != 0)
		goto fail;
	while (n_in_run < codec->n_inputs && in_values[n_in_run] != NULL) {
		run_in_names[n_in_run] = codec->input_names[n_in_run];
		++n_in_run;
	}

	/* GENERIC state carry-in: bind each loop-back buffer as its in_name input.
	 * The engine has no idea what any of these tensors mean — it just feeds the
	 * buffer it has been carrying frame-to-frame. */
	for (k = 0; k < ctx->n_loopbacks; ++k) {
		qnn_loopback_t *lb = &ctx->loopbacks[k];
		OrtStatus *s = ort->CreateTensorWithDataAsOrtValue(
			ctx->meminfo, lb->buf, lb->byte_count,
			lb->shape, lb->n_dims, lb->dtype, &in_values[n_in_run]);
		if (qnn_onnx_set_error_from_ort(ort, s, "CreateTensorWithDataAsOrtValue(loopback)"))
			goto fail;
		run_in_names[n_in_run] = lb->in_name;
		++n_in_run;
	}

	/* OUTPUTS to request: present ACTION heads (scattered to canonical slots
	 * for decode) + every loop-back output (copied back into its buffer). */
	for (i = 0; i < codec->n_outputs; ++i) {
		if (!ctx->out_present[i]) continue;
		bound_names[n_bound] = codec->output_names[i];
		bound_action_slot[n_bound] = (int)i;
		bound_lb_idx[n_bound] = -1;
		++n_bound;
	}
	for (k = 0; k < ctx->n_loopbacks; ++k) {
		bound_names[n_bound] = ctx->loopbacks[k].out_name;
		bound_action_slot[n_bound] = -1;
		bound_lb_idx[n_bound] = (int)k;
		++n_bound;
	}
	memset(bound_values, 0, sizeof(bound_values));

	{
		OrtStatus *s = ort->Run(
			ctx->session, NULL,
			run_in_names, (const OrtValue *const *)in_values, n_in_run,
			bound_names, n_bound, bound_values);
		if (qnn_onnx_set_error_from_ort(ort, s, "Session.Run"))
			goto fail;
	}

	/* Route results: action heads → canonical slot for decode; loop-back
	 * outputs → copy the result bytes into the paired in-buffer for next tick
	 * (the generic hidden→next_hidden carry, for an opaque tensor set). */
	for (i = 0; i < n_bound; ++i) {
		if (bound_action_slot[i] >= 0) {
			out_values[bound_action_slot[i]] = bound_values[i];
		} else {
			qnn_loopback_t *lb = &ctx->loopbacks[bound_lb_idx[i]];
			void *p = NULL;
			OrtStatus *s = ort->GetTensorMutableData(bound_values[i], &p);
			if (qnn_onnx_set_error_from_ort(ort, s, "Get(loopback out)"))
				goto fail;
			if (p != NULL) memcpy(lb->buf, p, lb->byte_count);
		}
	}

	if (codec->decode(ctx, out_values, result, out) != 0)
		goto fail;
	rc = 0;

fail:
	for (i = 0; i < n_in_run; ++i) if (in_values[i]) ort->ReleaseValue(in_values[i]);
	for (i = 0; i < n_bound;  ++i) if (bound_values[i]) ort->ReleaseValue(bound_values[i]);
	return rc;
}
