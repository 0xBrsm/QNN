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
	int subject_id;
	int action_id;
	int qualifier_id;
	float magnitude;
	vec3_t origin;
	int pickup_category;
	qboolean is_respawn;
	int weapon_subject_id;
	int powerup_subject_id;
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
#define QNN_ENTITY_EVENT_ID_DIM       3

/* Old qnn_entity_core_t / qnn_world_object_t / qnn_actor_t removed.
   The oracle reads directly from store entries in qnn_store.h. */

/* ── Semantic event atom ───────────────────────────────────────── */

typedef struct
{
	qboolean active;
	int owner_index;
	int next_for_owner;
	int subject_id;
	int action_id;
	int qualifier_id;
	int modality_id;
	float recency;
	float confidence;
	float magnitude;
} qnn_semantic_event_atom_t;

#define QNN_EVENT_HEAD_CAPACITY (MAX_EDICTS + 1024)

/* ── Entity token (ready-to-consume output) ────────────────────── */

typedef struct
{
	int subject_id;
	int qualifier_id;
	int modality_id;
	int player_id;
	int cluster_id;
	int powerup_subject_id;
	int weapon_subject_id;

	float rel[3];
	float distance;
	float route_cost;
	float vel[3];
	float rel_yaw;
	float rel_pitch;
	float half_extents[3];
	float recency;
	float confidence;
	float magnitude;
	float state;

	int route_cluster_ids[QNN_MAX_ROUTE_CLUSTERS];
	int route_cluster_count;

	int event_subject[QNN_MAX_ENTITY_EVENTS];
	int event_action[QNN_MAX_ENTITY_EVENTS];
	int event_qualifier[QNN_MAX_ENTITY_EVENTS];
	float event_recency[QNN_MAX_ENTITY_EVENTS];
	int event_count;
} qnn_entity_token_t;

/* ── Event atom globals (defined in qnn_event.c) ──────────────── */

extern qnn_semantic_event_atom_t qnn_semantic_events[QNN_MAX_EVENT_ATOMS];
extern int qnn_event_head[QNN_EVENT_HEAD_CAPACITY];
extern int qnn_prev_object_indices[QNN_MAX_TOKEN_OBJECTS];
extern int qnn_prev_object_count;

/* ── Static property helpers (qnn_map.c) ──────────────────────── */

const char *QNN_ObjectStaticProperty(const qnn_static_object_t *obj, const char *key);
int QNN_ObjectStaticPropertyInt(const qnn_static_object_t *obj, const char *key, int fallback);

/* ── Oracle token emission (qnn_oracle.c) ─────────────────────── */

int QNN_OracleEmitTokens(qnn_entity_token_t *out_tokens, int max_tokens,
	const qnn_snapshot_t *snapshot, const qnn_map_state_t *map_state,
	int *out_player_cluster_id);

/* ── Sound rule type (used by qnn_event.c) ─────────────────────── */

typedef struct
{
	const char *name;
	int subject_id;
	int action_id;
	int qualifier_id;
	float magnitude;
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
