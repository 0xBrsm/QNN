/*
 * qnn_object.h — World model types and API.
 *
 * Everything related to objects: semantic entity/event types, oracle
 * store API, entity token output, perception API (entity + event
 * classification), snapshot capture.
 */

#ifndef QNN_OBJECT_H
#define QNN_OBJECT_H

#include "qnn.h"
#include "qnn_vocab.h"

/* ── Observed runtime entity (entity.c → object.c, no FOV filter) ── */

#define QNN_MAX_ENTITY_UPDATES 128
#define QNN_MAX_PVS_ITEMS 128

typedef struct {
	int entity_num;
	int subject_id;
	int qualifier_id;
	float magnitude;
	vec3_t origin;
	vec3_t velocity;
	vec3_t angles;
	float half_extents[3];
	int effects;
	int health;
	int frags;
	qboolean is_item;
	qboolean in_fov;
	qboolean is_brush;     /* true for *N brush models (movers) */
} qnn_entity_update_t;

typedef struct {
	int entity_num;
	int subject_id;
	vec3_t origin;
	float magnitude;
	qboolean in_fov;
} qnn_pvs_item_t;

/* ── Event classification output (event.c → object.c) ─────────── */

#define QNN_MAX_EVENT_RECORDS 32

typedef struct {
	int entity_num;
	int action_id;
	int source_id;
	vec3_t origin;
	int pickup_category;
	qboolean is_respawn;
	int weapon_subject_id;
	int powerup_subject_id;
	int match_subject_id;  /* for mover spatial matching — DOOR, PLATFORM, etc. */
} qnn_event_record_t;

/* ── Static entity init (entity.c → object.c at map load) ──────── */
/* QNN_MAX_STATIC_PROPERTIES defined in qnn.h */

typedef struct {
	int entity_num;
	int subject_id;
	int qualifier_id;
	float magnitude;
	qboolean is_item;
	float respawn_s;
	vec3_t origin;
	vec3_t angles;
	char classname[QNN_MAX_CLASSNAME];
	char category[QNN_MAX_CATEGORY];
	qnn_property_t properties[QNN_MAX_STATIC_PROPERTIES];
	int property_count;
} qnn_static_entity_t;

/* ── Internal types ────────────────────────────────────────────── */

typedef struct
{
	float dist_sq;
	int ent_idx;
	int obj_idx;
} qnn_match_candidate_t;

/* ── Capacity constants ────────────────────────────────────────── */

#define QNN_MAX_TOKEN_OBJECTS        16
#define QNN_MAX_ROUTE_PATHS           3
#define QNN_MAX_ROUTE_CLUSTERS        8
#define QNN_MAX_ENTITY_EVENTS         4
#define QNN_ENTITY_EVENT_ID_DIM       2

/* Old qnn_entity_core_t / qnn_world_object_t / qnn_actor_t removed.
   The oracle reads directly from store entries in qnn_store.h. */

/* ── Semantic event atom ───────────────────────────────────────── */

typedef struct
{
	qboolean active;
	int owner_index;
	int next_for_owner;
	int action_id;
	int source_id;
	float timestamp;       /* cl.mtime[0] when event was created */
} qnn_semantic_event_atom_t;

#define QNN_EVENT_HEAD_CAPACITY (MAX_EDICTS + 1024)

/* ── Per-type entity tokens ────────────────────────────────────── */

/* Event attachment (shared by all token types) */
typedef struct
{
	int action_id;
	int source_id;
} qnn_token_event_t;

/* Projectile: 8 scalars, 2 IDs */
typedef struct
{
	int subject_id;
	int modality_id;
	float rel[3];
	float dist;
	float vel[3];
	float recency;
	qnn_token_event_t events[QNN_MAX_ENTITY_EVENTS];
	int event_count;
} qnn_projectile_token_t;

/* Actor: 19 scalars, 3 IDs */
typedef struct
{
	int subject_id;
	int modality_id;
	int player_id;
	float half_extents[3];
	float rel[3];
	float dist;
	float vel[3];
	float path[3];
	float path_dist;
	float eta;
	float facing;
	float team;
	float score;
	float recency;
	qnn_token_event_t events[QNN_MAX_ENTITY_EVENTS];
	int event_count;
} qnn_actor_token_t;

/* Item: 15 scalars, 2 IDs */
typedef struct
{
	int subject_id;
	int modality_id;
	float half_extents[3];
	float rel[3];
	float dist;
	float path[3];
	float path_dist;
	float eta;
	float amount;
	float regen;
	float recency;
	qnn_token_event_t events[QNN_MAX_ENTITY_EVENTS];
	int event_count;
} qnn_item_token_t;

/* Mover: 14 scalars, 2 IDs */
typedef struct
{
	int subject_id;
	int modality_id;
	float half_extents[3];
	float rel[3];
	float dist;
	float path[3];
	float path_dist;
	float eta;
	float state;
	float recency;
	qnn_token_event_t events[QNN_MAX_ENTITY_EVENTS];
	int event_count;
} qnn_mover_token_t;

/* Tagged token for the variable-length stream */
#define QNN_TOKEN_PROJECTILE  0
#define QNN_TOKEN_ACTOR       1
#define QNN_TOKEN_ITEM        2
#define QNN_TOKEN_MOVER       3

typedef struct
{
	int type; /* QNN_TOKEN_* */
	union {
		qnn_projectile_token_t projectile;
		qnn_actor_token_t actor;
		qnn_item_token_t item;
		qnn_mover_token_t mover;
	};
} qnn_tagged_token_t;

/* ── Event atom globals (defined in qnn_event.c) ──────────────── */

extern qnn_semantic_event_atom_t qnn_semantic_events[QNN_MAX_EVENT_ATOMS];
extern int qnn_event_head[QNN_EVENT_HEAD_CAPACITY];

/* ── Static property helpers (qnn_map.c) ──────────────────────── */

const char *QNN_ObjectStaticProperty(const qnn_static_object_t *obj, const char *key);
int QNN_ObjectStaticPropertyInt(const qnn_static_object_t *obj, const char *key, int fallback);

/* ── Oracle token emission (qnn_oracle.c) ─────────────────────── */

int QNN_OracleEmitTokens(qnn_tagged_token_t *out_tokens, int max_tokens,
	const qnn_snapshot_t *snapshot, const qnn_map_state_t *map_state,
	int *out_player_cluster_id);

/* Episode-boundary hook for the oracle.  Currently a no-op; reserved for
 * future cross-episode state that needs clearing on reset. */
void QNN_OracleResetState(void);

/* ── Sound rule type (used by qnn_event.c) ─────────────────────── */

typedef struct
{
	const char *name;
	int action_id;
	int source_id;
	int match_subject_id;  /* entity subject to match for spatial lookup (movers) */
} qnn_sound_rule_t;

/* ── Event system (qnn_event.c) ────────────────────────────────── */

void QNN_EventInit(const qnn_map_state_t *map_state);
void QNN_EventTick(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag);

/* ── Entity perception (qnn_entity.c) ─────────────────────────── */

/* BSP line trace from start to end against the world hull.  Per-engine
 * implementation: NQ forwards to upstream's SV_RecursiveHullCheck, QW
 * adapts upstream's PM_RecursiveHullCheck.  Caller's trace_t is filled
 * with at minimum fraction (1.0 = no obstruction) and endpos. */
void QNN_TraceLine(const vec3_t start, const vec3_t end, trace_t *trace);

qboolean QNN_InFov(const vec3_t player_origin, const vec3_t view_angles, const vec3_t target);
qboolean QNN_EntityInPvs(const vec3_t viewer, const vec3_t target);

/* Forward declaration — full definition in qnn_map.h */
struct qnn_raw_entity_s;
typedef struct qnn_raw_entity_s qnn_raw_entity_t;

int QNN_EntityClassifyStatic(const qnn_raw_entity_t *raw, int raw_count,
	qnn_static_entity_t *out, int max);
int QNN_EntityClassifyKnown(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int max_entities,
	qnn_pvs_item_t *out_pvs, int max_pvs, int *out_pvs_count);
int QNN_AppendPlayerEntityUpdates(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int start_count, int max_entities);

/* Returns true iff slot ``entity_num`` (1-indexed) currently holds a
 * live first-person player on this engine — not the recorder, not an
 * empty slot, not a spectator (QW), and has a real player body.  Both
 * stores and the event handler must consult this before promoting a
 * slot to an ACTOR entity, otherwise spectators in QWD demos and
 * non-player edicts in NQ demos leak into the actor token stream. */
qboolean QNN_IsLivePlayerSlot(int entity_num);

/* Team detection lives in per-game qnn_players.c.  NQ uses pants color
 * (the engine's own team field); QW uses userinfo "team" + cl.teamplay. */
void QNN_PlayersResetTeamCache(void);
float QNN_IsSameTeam(int entity_num);
float QNN_FragFraction(int entity_frags);
qboolean QNN_ClassifyStaticSubject(const qnn_static_object_t *obj, int *subject_id, int *qualifier_id, float *magnitude, qboolean *is_item, float *respawn_s);
float QNN_ItemRespawnS(const qnn_static_object_t *obj, int subject_id);
int QNN_SubjectPickupCategory(int subject_id);

/* ── Event perception (qnn_event.c) ───────────────────────────── */

int QNN_EventClassifySounds(const qnn_snapshot_t *snapshot, qnn_event_record_t *out_records, int max_records);

/* ── Self state capture (qnn_self.c) ──────────────────────────── */

/* QNN_CaptureBaseSnapshot is per-game (origin/velocity acquisition
 * differs).  QNN_CurrentFrags is per-game (frag source differs).
 * QNN_WeaponId, QNN_CurrentArmortype, and QNN_SelfEmitToken are shared
 * (read cl.stats[*] / snapshot only) and live in common/qnn_self.c. */
void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot);
int QNN_WeaponId(void);
int QNN_ItemFlagFromImpulse(int impulse);
int QNN_NextWeaponId(int reverse);
float QNN_LatencySeconds(void);
int QNN_LatencyFrames(int emit_hz);

/* MVD-side label inference helper: how many emit frames to shift
 * backward from a server-state-change observation to estimate the
 * player's actual press time.  Returns ping_ms / (1000/emit_hz),
 * integer floor — see qnn_self.c for the derivation.  Caller invokes
 * QNN_ObservePings() each emit so the per-demo running median (used
 * to clamp garbage svc_updateping values) stays fresh.  Returns 0 on
 * NQ (no MVD inference) or before any svc_updateping has been seen. */
void QNN_ObservePings(void);
int QNN_PressBackShiftFrames(int player_slot, int emit_hz);
/* Validated per-player ping in ms / sec.  Median-fallback + outlier
 * reject — single source of truth for ping used by all back-shift
 * paths (fire/jump per-sound, weapon/move per-emit-frame). */
int QNN_PressPingMs(int player_slot);
float QNN_PressPingSec(int player_slot);
/* Recording-client's own ping in ms.  Engine-agnostic wrapper:
 * QW returns QNN_PressPingMs(cl.playernum); NQ returns 0 (no
 * MVD-style ping broadcast). */
int QNN_SelfPingMs(void);
/* Reset per-demo running ping median.  Call at demo boundaries
 * before the new demo's first QNN_ObservePings(). */
void QNN_ResetPingEstimator(void);
float QNN_CurrentArmortype(void);
int QNN_CurrentFrags(void);

/* ── Sound drain (qnn_event.c) ───────────────────────────────── */

void QNN_DrainSounds(qnn_snapshot_t *snapshot);

/* ── Utility (qnn_sys.c) ─────────────────────────────────────── */

const char *QNN_ProgString(string_t value);

#endif /* QNN_OBJECT_H */
