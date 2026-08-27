/*
 * qnn_obs_registry.h — The observation field registry + emit-plan compiler.
 *
 * Implements the demand-driven obs API (agents/plans/obs-api.md, LOCKED
 * 2026-07-26).  One C-side authoritative table maps field names to
 * {kind, param schema, dtype, shape, size, emit}; a model's *declaration*
 * (which fields, with which parameters) compiles into an ordered *emit
 * plan* with computed offsets and a total frame size.  The engine then
 * computes and serializes EXACTLY what the plan demands — the request
 * list IS the compute plan.
 *
 * Three contract kinds:
 *   state   — raw game scalars (the self block).  Named, individually
 *             requestable, version-free.
 *   sensor  — parameterized computed queries (the depth atlas).
 *             Requested WITH parameters; 24×11-packed and 72×11-unpacked
 *             are two parameterizations of ONE sensor, not two fields.
 *   percept — the entity stream: oracle state × disclosure policy.
 *             Requested AT a policy version.  Policy versions pin
 *             semantics, not byte layout.
 *
 * Adding a field = adding a row (the decode-param registry pattern).
 * Unknown field / bad params at compile time = hard error naming the
 * entry — fail loud, no silent defaults.
 *
 * ── Percept policy "v3" pins (any change to ANY of these = policy v4,
 *    a NEW registry row; v3 keeps serving — never retire a served
 *    policy) ──────────────────────────────────────────────────────────
 *   visibility    = hull-0 LOS ∪ PVS-audible sound events (own-fire
 *                   gated at OWN_FIRE_DIST_U)
 *   memory tail   = 2.0 s recency
 *   event window  = 4 pairs/entity (QNN_MAX_ENTITY_EVENTS)
 *   path provider = Recast/Detour navmesh
 *   token budget  = priority order as implemented today (qnn_oracle.c)
 * The v3 wire row layout is the current QNN_IOPackObsBuffer entity
 * stream (see qnn_io.h "ENTITY STREAM"): u8 n_tokens, then per token a
 * type tag, ids, event pairs, and per-type native-width scalars.  The
 * `paths` parameter gates COMPUTE only (pathfinder demand); paths=false
 * keeps the v3 row layout with the path-derived fields (path[3],
 * path_dist, eta) zeroed, so layout stays policy-pinned.
 *
 * ── Percept policy "v1" (WS2; wire-shim generations) ────────────────
 * The f84c36cd^-era FULL disclosure policy, kept for the stamped
 * wire.9/.11/.12.x fleet: all four token types (projectile / actor /
 * item / mover), the sight/proximity/sound/memory modality ladder and
 * the 2.0 s recency memory tail (QNN_ENTITY_MODE_FULL oracle
 * qualification).  Policy "v3" is the A27 pure-combat disclosure
 * (current-frame actor/projectile only, SIGHT/PROXIMITY —
 * QNN_ENTITY_MODE_COMBAT), which is also the no-declaration default.
 * The compute stage derives the oracle qualification mode from the
 * plan's policy (qnn_io.c provider); v1 and v3 now serialize with
 * their OWN row writer (QNN_ObsEmitEntityRowV1 / V3 in the .c file) —
 * they only share event-slot packing.
 *
 * RESOLVED (was a KNOWN CONFLICT through the WS1/WS2 landing): both
 * policies used to serialize the FULL row layout (recency in every
 * row), which matched the Gate-1 goldens but disagreed with the
 * python mirror's (qnn/obs_api.py) v3 walk — a fresh collect desynced
 * inside the entity section.  The v3 writer is now ported from
 * bb27a296 ("feat(a27): pure-combat substrate refactor" — the last
 * commit where the native emit still wrote this shape): actor/
 * projectile only, no recency, matching _POLICY_V3 field-for-field.
 * Gate 1 (qnn_obs_registry_test.c) now pins v1 and v3 against their
 * OWN reference packers.
 */

#ifndef QNN_OBS_REGISTRY_H
#define QNN_OBS_REGISTRY_H

#include "qnn_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Declaration schema version this registry implements. */
#define QNN_OBS_API_VERSION           1

/* Capacity bounds.  The seed registry holds 13 state fields + the atlas
 * sensor + the entity percept; leave headroom for added rows. */
#define QNN_OBS_MAX_STATE_FIELDS      16
#define QNN_OBS_MAX_PLAN_STEPS        (QNN_OBS_MAX_STATE_FIELDS + 2)
#define QNN_OBS_MAX_FIELD_NAME        32
#define QNN_OBS_MAX_POLICY_NAME       8
#define QNN_OBS_MAX_SHAPE_DIMS        4

/* Every compiled frame reserves an env-gated pose tail after the
 * payload (see QNN_IOStashPoseTail) — outside the declaration, never a
 * requestable field.  Default plan: 848 payload + 16 tail = 864. */
#define QNN_OBS_POSE_TAIL_BYTES       16

/* Hard ceiling on a declaration's explicit frame_bytes override (the
 * wire-shim escape hatch for fixed-frame legacy generations, e.g. the
 * pre-packing 4096-byte buffer).  Anything larger is a corrupt or
 * hostile declaration — refuse rather than size buffers from it. */
#define QNN_OBS_FRAME_BYTES_MAX       65536

/* ── Contract kinds ─────────────────────────────────────────────── */

typedef enum {
	QNN_OBS_KIND_STATE = 0,   /* raw game scalar (self block) */
	QNN_OBS_KIND_SENSOR,      /* parameterized computed query (atlas) */
	QNN_OBS_KIND_PERCEPT,     /* disclosure-policy entity stream */
} qnn_obs_kind_t;

/* ── Field parameters ───────────────────────────────────────────── */

/* Atlas sensor parameters.  Supported parameterizations:
 *   {yaw: 24, bands: 11, packed: true}   — current wire (two 4-bit
 *                                          codes/byte, 132 B)
 *   {yaw: 72, bands: 11, packed: false}  — the restored f84c36cd^ flat
 *                                          emit (one code/byte, 792 B)
 * Anything else fails compilation. */
typedef struct {
	int       yaw;            /* yaw cells per elevation band */
	int       bands;          /* elevation bands — must be 11 */
	qboolean  packed;         /* true = nibble-packed pairs */
} qnn_obs_atlas_params_t;

/* Entity percept parameters. */
typedef struct {
	char      policy[QNN_OBS_MAX_POLICY_NAME];  /* "v3" */
	int       max_tokens;     /* 1..QNN_MAX_TOKEN_OBJECTS */
	qboolean  paths;          /* false = zero path fields, skip pathfinder */
} qnn_obs_entity_params_t;

/* One params slot per plan step — which member is live follows the
 * entry's kind (state entries carry no parameters). */
typedef union {
	qnn_obs_atlas_params_t   atlas;
	qnn_obs_entity_params_t  entities;
} qnn_obs_params_t;

/* ── The declaration (parsed form) ──────────────────────────────────
 * The C-struct equivalent of the JSON `obs_declaration` blob; JSON
 * parsing lives at the protocol layer (WS2).  Field order in `state`
 * is wire order. */

typedef struct {
	int       obs_api;        /* must be QNN_OBS_API_VERSION */
	int       state_count;
	char      state[QNN_OBS_MAX_STATE_FIELDS][QNN_OBS_MAX_FIELD_NAME];
	qboolean  atlas_requested;      /* false = not requested = not computed */
	qnn_obs_atlas_params_t   atlas;
	qboolean  entities_requested;
	qnn_obs_entity_params_t  entities;
	/* Explicit frame-size override (0 = none → payload + pose tail).
	 * Wire-shim only: pre-obs-api generations whose flat frame was a
	 * pinned constant (the pre-packing 4096-byte buffer) rather than
	 * the payload budget.  Must cover the compiled minimum and stay
	 * under QNN_OBS_FRAME_BYTES_MAX, else compilation fails. */
	int       frame_bytes_override;
} qnn_obs_decl_t;

/* ── Registry entry ─────────────────────────────────────────────── */

/* Serialization context handed to emit_fn — the per-seat view of one
 * tick.  Pack-stage emitters read only `result`; `snapshot` is carried
 * for future rows that serialize straight from snapshot state. */
typedef struct {
	const qnn_snapshot_t     *snapshot;   /* may be NULL at pack time */
	const qnn_tick_result_t  *result;
} qnn_obs_seat_ctx_t;

typedef struct qnn_obs_registry_entry_s {
	const char      *name;
	qnn_obs_kind_t   kind;
	/* Human-readable parameter schema, quoted in compile errors. */
	const char      *param_schema;
	/* Wire dtype of the field's cells, as the numpy dtype NAME the
	 * python mirror (qnn/obs_api.py) uses — the layout reply quotes
	 * it verbatim and the drivers compare dicts ("uint8", "int8",
	 * "int16", "float16", "int32"; "uint8" for the packed-nibble
	 * atlas bytes).  NULL for the opaque variable-length entity
	 * stream (serialized as JSON null in the reply). */
	const char      *dtype;
	/* Validate params against the schema.  NULL = entry takes no
	 * parameters.  Returns false + fills `error` on rejection. */
	qboolean (*validate_params)(const qnn_obs_params_t *params,
		char *error, size_t error_size);
	/* Wire shape for the given params (ndim 0 = scalar). */
	void (*shape_fn)(const qnn_obs_params_t *params,
		int shape[QNN_OBS_MAX_SHAPE_DIMS], int *ndim);
	/* Bytes reserved on the wire (maximum for variable-length). */
	int (*size_fn)(const qnn_obs_params_t *params);
	/* Serialize the field at `out` (already offset into the frame).
	 * Returns bytes written (≤ size_fn; less for variable-length). */
	int (*emit_fn)(const qnn_obs_seat_ctx_t *ctx,
		const qnn_obs_params_t *params, uint8_t *out);
} qnn_obs_registry_entry_t;

/* ── Compiled emit plan ─────────────────────────────────────────── */

typedef struct {
	const qnn_obs_registry_entry_t  *entry;
	qnn_obs_params_t                 params;
	int                              offset;   /* into the frame */
	int                              bytes;    /* reserved (max) */
} qnn_obs_plan_step_t;

typedef struct qnn_obs_plan_s {
	qnn_obs_plan_step_t  steps[QNN_OBS_MAX_PLAN_STEPS];
	int                  step_count;
	int                  payload_bytes;   /* end of the last field */
	int                  frame_bytes;     /* payload + pose-tail reserve */
	/* Demand flags for the compute stage — derived from the steps so
	 * the tick path never re-scans the table. */
	qboolean                 wants_state;
	qboolean                 wants_atlas;
	qnn_obs_atlas_params_t   atlas;       /* valid iff wants_atlas */
	qboolean                 wants_entities;
	qnn_obs_entity_params_t  entities;    /* valid iff wants_entities */
} qnn_obs_plan_t;

/* ── Compute providers ──────────────────────────────────────────────
 * The compute stage is demand-driven: QNN_ObsPlanCompute invokes a
 * provider ONLY for kinds the plan requests.  qnn_io.c wires the real
 * emitters (QNN_SelfEmitToken / QNN_OracleEmitTokens /
 * QNN_SpatialEmitAtlas); tests wire counting stubs to prove the
 * demand-driven property (a plan without entities never reaches the
 * oracle, hence never the pathfinder). */

typedef struct {
	void (*self)(const qnn_snapshot_t *snapshot, qnn_self_token_t *out);
	int (*entities)(const qnn_snapshot_t *snapshot,
		const qnn_obs_entity_params_t *params,
		qnn_tagged_token_t *out, int max_tokens);
	void (*atlas)(const qnn_snapshot_t *snapshot,
		const qnn_obs_atlas_params_t *params,
		uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX]);
} qnn_obs_compute_fns_t;

/* ── API ────────────────────────────────────────────────────────── */

/* Table introspection (layout replies, python mirror validation). */
int QNN_ObsRegistryCount(void);
const qnn_obs_registry_entry_t *QNN_ObsRegistryAt(int index);
const qnn_obs_registry_entry_t *QNN_ObsRegistryFind(const char *name);

/* The default declaration — today's packed 864-byte frame: all 13
 * state fields in wire order, atlas {24, 11, packed}, entities
 * {v3, QNN_MAX_TOKEN_OBJECTS, paths on}.  Legacy drivers that send no
 * declaration get exactly this. */
void QNN_ObsDeclDefault(qnn_obs_decl_t *out);

/* Compile a declaration into an ordered emit plan (offsets + total
 * frame_bytes).  Unknown field / bad params / unknown policy → returns
 * false with `error` naming the offending entry.  Never partially
 * succeeds: on failure the plan contents are undefined. */
qboolean QNN_ObsPlanCompile(const qnn_obs_decl_t *decl, qnn_obs_plan_t *plan,
	char *error, size_t error_size);

/* Compute stage: reset `out`, prefill the atlas scratch with the miss
 * code, then run providers for exactly the kinds the plan demands. */
void QNN_ObsPlanCompute(const qnn_obs_plan_t *plan,
	const qnn_obs_compute_fns_t *fns, const qnn_snapshot_t *snapshot,
	qnn_tick_result_t *out);

/* Serialization stage: zero `obs` and walk the plan, emitting each
 * field at its compiled offset.  `obs_bytes` must cover
 * plan->frame_bytes — a short buffer is a programming error and
 * aborts (fail loud). */
void QNN_ObsPlanPack(const qnn_obs_plan_t *plan, uint8_t *obs,
	int obs_bytes, const qnn_tick_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* QNN_OBS_REGISTRY_H */
