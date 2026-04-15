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
extern int qnn_prev_object_indices[QNN_MAX_TOKEN_OBJECTS];
extern int qnn_prev_object_count;

/* ── Static property helpers (qnn_map.c) ──────────────────────── */

const char *QNN_ObjectStaticProperty(const qnn_static_object_t *obj, const char *key);
int QNN_ObjectStaticPropertyInt(const qnn_static_object_t *obj, const char *key, int fallback);

/* ── Oracle token emission (qnn_oracle.c) ─────────────────────── */

int QNN_OracleEmitTokens(qnn_tagged_token_t *out_tokens, int max_tokens,
	const qnn_snapshot_t *snapshot, const qnn_map_state_t *map_state,
	int *out_player_cluster_id);

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

qboolean QNN_InFov(const vec3_t player_origin, const vec3_t view_angles, const vec3_t target);

/* Forward declaration — full definition in qnn_map.h */
struct qnn_raw_entity_s;
typedef struct qnn_raw_entity_s qnn_raw_entity_t;

int QNN_EntityClassifyStatic(const qnn_raw_entity_t *raw, int raw_count,
	qnn_static_entity_t *out, int max);
int QNN_EntityClassifyKnown(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int max_entities,
	qnn_pvs_item_t *out_pvs, int max_pvs, int *out_pvs_count);
void QNN_EntityResetTeamCache(void);
float QNN_IsSameTeam(int entity_num);
float QNN_FragFraction(int entity_frags);
qboolean QNN_ClassifyStaticSubject(const qnn_static_object_t *obj, int *subject_id, int *qualifier_id, float *magnitude, qboolean *is_item, float *respawn_s);
qboolean QNN_ClassifyKnownSubject(const qnn_known_entity_t *ent, int *subject_id, int *qualifier_id, float *magnitude);
float QNN_ItemRespawnS(const qnn_static_object_t *obj, int subject_id);
int QNN_SubjectPickupCategory(int subject_id);

/* ── Event perception (qnn_event.c) ───────────────────────────── */

int QNN_EventClassifySounds(const qnn_snapshot_t *snapshot, qnn_event_record_t *out_records, int max_records);

/* ── Self state capture (qnn_self.c) ──────────────────────────── */

void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot);
int QNN_WeaponId(void);
int QNN_CurrentFrags(void);

/* ── Sound drain (qnn_event.c) ───────────────────────────────── */

void QNN_DrainSounds(qnn_snapshot_t *snapshot);

/* ── Utility (qnn_sys.c) ─────────────────────────────────────── */

const char *QNN_ProgString(string_t value);

#endif /* QNN_OBJECT_H */
