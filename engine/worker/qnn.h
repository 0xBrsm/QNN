#ifndef QNN_H
#define QNN_H

#include <stdio.h>

#include "qnn_navmesh.h"
#include "quakedef.h"

#ifndef QNN_NAV_ORACLE_RUNTIME_FWD
#define QNN_NAV_ORACLE_RUNTIME_FWD
typedef struct qnn_nav_oracle_runtime_s qnn_nav_oracle_runtime_t;
#endif

#define QNN_ACTION_HISTORY 2
#define QNN_MAX_PROPERTY_KEY 64
#define QNN_MAX_PROPERTY_VALUE 256
#define QNN_MAX_CLASSNAME 64
#define QNN_MAX_CATEGORY 16
#define QNN_MAX_OBJECT_ID 32
#define QNN_MAX_MAP_ID 64
#define QNN_MAX_MODEL_NAME 64
#define QNN_MAX_SOUNDS 16
#define QNN_MAX_SOUND_NAME 64
#define QNN_MAX_VISIBLE 64
#define QNN_MAX_EVENTS 16
#define QNN_MAX_DYNAMIC_OBJECTS 128
#define QNN_MAX_TOKEN_OBJECTS 64
#define QNN_MAX_EVENT_ATOMS 256
#define QNN_SPATIAL_TOKEN_COUNT 9
#define QNN_MAX_TRAIN_DAMAGE 64
#define QNN_MAX_TRAIN_ITEMS 64
#define QNN_MAX_TRAIN_DEATHS 16
#define QNN_MAX_TRAIN_SPAWNS 16

/* Sound category IDs (matches Python SOUND_CATEGORIES) */
#define QNN_SND_CAT_WEAPON   0
#define QNN_SND_CAT_PAIN     1
#define QNN_SND_CAT_PICKUP   2
#define QNN_SND_CAT_FOOTSTEP 3
#define QNN_SND_CAT_AMBIENT  4

typedef struct
{
	vec3_t	origin;
	float	volume;
	float	attenuation;
	int	entity_num;
	int	category;
	char	name[QNN_MAX_SOUND_NAME];
} qnn_sound_event_t;

typedef struct
{
	char	key[QNN_MAX_PROPERTY_KEY];
	char	value[QNN_MAX_PROPERTY_VALUE];
} qnn_property_t;

typedef struct
{
	char	object_id[QNN_MAX_OBJECT_ID];
	char	category[QNN_MAX_CATEGORY];
	char	classname[QNN_MAX_CLASSNAME];
	int	region_id;
	vec3_t	origin;
	vec3_t	angles;
	qnn_property_t *properties;
	int	property_count;
} qnn_static_object_t;

typedef struct
{
	int	region_id;
	vec3_t	center;
	vec3_t	bounds_min;
	vec3_t	bounds_max;
} qnn_region_t;

typedef struct
{
	char	requested_map_id[QNN_MAX_MAP_ID];
	char	map_name[QNN_MAX_MAP_ID];
	char	source[32];
	char	navmesh_status[16];
	char	navmesh_error[256];
	char	nav_oracle_status[16];
	char	nav_oracle_error[256];
	qnn_navmesh_build_config_t navmesh_config;
	qnn_navmesh_summary_t navmesh_summary;
	qnn_navmesh_runtime_t *navmesh;
	qnn_nav_oracle_runtime_t *nav_oracle;
	int	nav_area_count;
	int	nav_cluster_count;
	int	nav_min_cluster_area_count;
	int	nav_max_cluster_area_count;
	float	nav_avg_cluster_area_count;
	int	nav_walk_link_count;
	int	nav_teleport_link_count;
	int	nav_lift_link_count;
	int	nav_push_link_count;
	int	nav_drop_link_count;
	qnn_region_t *regions;
	int	region_count;
	qnn_static_object_t *static_objects;
	int	static_object_count;
} qnn_map_state_t;

typedef struct
{
	float	move[2];
	float	look[2];
	int	fire;
	int	jump;
	int	switch_slot;
	int	recall[4];
} qnn_action_t;

typedef struct
{
	int	entity_key;
	char	entity_id[32];
	int	entity_num;
	char	classname[QNN_MAX_CLASSNAME];
	char	model_name[QNN_MAX_MODEL_NAME];
	int	region_id;
	vec3_t	origin;
	vec3_t	velocity;
	vec3_t	angles;
	int	model_id;
	int	frame;
	int	effects;
	int	skin;
	int	health;
	int	frags;
	float	half_extents[3];
	qboolean static_proxy;
} qnn_visible_entity_t;

typedef struct
{
	char	event_type[32];
	int	region_id;
	int	has_delta;
	int	delta;
	int	has_weapon_id;
	int	weapon_id;
	int	source_entity_num;
	int	target_entity_num;
} qnn_event_t;

typedef struct
{
	vec3_t	player_origin;
	vec3_t	player_velocity;
	vec3_t	player_view_angles;
	int	health;
	int	armor;
	float	armor_type;
	int	ammo;
	int	ammo_shells;
	int	ammo_nails;
	int	ammo_rockets;
	int	ammo_cells;
	int	weapons_owned;
	int	weapon_id;
	int	waterlevel;
	qboolean grounded;
	int	current_region_id;
	qboolean done;
	qnn_visible_entity_t visible[QNN_MAX_VISIBLE];
	int	visible_count;
	qnn_event_t events[QNN_MAX_EVENTS];
	int	event_count;
	int	damage_dealt;
	int	hit_count;
	int	shots_fired;
	int	damage_weapon_id;
	qnn_sound_event_t sounds[QNN_MAX_SOUNDS];
	int	sound_count;
	qnn_action_t action_label;
} qnn_snapshot_t;

extern qnn_action_t qnn_pending_action;

/* Binary step protocol: 1 opcode byte + action struct.
 * Replaces JSON for the hot step path.  Hello/reset/shutdown stay JSON. */
#define QNN_BINARY_OP_STEP 0x01
#define QNN_BINARY_ACTION_SIZE ((int)sizeof(qnn_action_t))

/* ── Tick resampling gate ─────────────────────────────────────────
 * Accumulates engine frames and emits at a fixed target Hz.
 * Call QNN_ResampleInit() once, then QNN_ResampleShouldEmit()
 * every frame.  When it returns true, emit the token tick.
 * Action labels are merged across the window: fire/jump use OR,
 * look accumulates mouse-count deltas, move uses latest value.
 */
typedef struct
{
	int	target_hz;		/* 0 = disabled (emit every frame) */
	float	target_dt;		/* 1.0 / target_hz */
	float	accumulated_dt;		/* time since last emission */
	int	fire_any;		/* OR accumulator for fire */
	int	jump_any;		/* OR accumulator for jump */
	float	look_yaw_degrees;	/* accumulated yaw delta (degrees) across window */
	float	look_pitch_degrees;	/* accumulated pitch delta (degrees) across window */
} qnn_resample_state_t;

extern qnn_resample_state_t qnn_resample;

#define QNN_DEMO_MOUSE_DEGREES_PER_COUNT 0.066f

void QNN_ResampleInit(int target_hz);
void QNN_ResampleAccumulate(const qnn_snapshot_t *snapshot, float frame_dt);
void QNN_ResampleAccumulateLook(float yaw_degrees, float pitch_degrees);
qboolean QNN_ResampleShouldEmit(void);
void QNN_ResampleApplyActionMerge(qnn_snapshot_t *snapshot);
extern qnn_sound_event_t qnn_sound_buffer[QNN_MAX_SOUNDS];
extern int qnn_sound_count;

/* Shared globals (defined in qnn_sys.c) */
extern qnn_map_state_t qnn_map_state;
extern char qnn_basedir_storage[MAX_OSPATH];
extern char *basedir;
extern char *cachedir;

/* Common utilities (qnn_sys.c) */
void QNN_ResolveBasedir(char *out, size_t out_size);
int QNN_JsonExtractInt(const char *line, const char *key, int fallback);
float QNN_JsonExtractFloat(const char *line, const char *key, float fallback);
qboolean QNN_JsonExtractBool(const char *line, const char *key, qboolean fallback);
qboolean QNN_JsonExtractString(const char *line, const char *key, char *out, size_t out_size);
qboolean QNN_JsonExtractVec2(const char *line, const char *key, float out[2]);
qboolean QNN_JsonExtractVec3(const char *line, const char *key, vec3_t out);
float QNN_LookAxisFromMouseCount(int mouse_count);
int QNN_MouseCountFromLookAxis(float axis);
int QNN_SwitchSlotFromWeaponId(int weapon_id);
int QNN_SwitchImpulseFromSlot(int switch_slot, int weapons_owned);
qboolean QNN_PrepareMap(const char *requested_map_id, char *error, size_t error_size);
void QNN_WriteJsonString(FILE *out, const char *text);
void QNN_WriteError(const char *message);
const char *QNN_ProgString(string_t value);
int QNN_WeaponId(void);
int QNN_CurrentFrags(void);
void QNN_CaptureVisibleEntities(qnn_snapshot_t *snapshot, float fixed_dt);
void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot);
void QNN_DrainSounds(qnn_snapshot_t *snapshot);

void QNN_ClearAction(qnn_action_t *action);
qboolean QNN_BuildMapState(qnn_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size);
void QNN_FreeMapState(qnn_map_state_t *map_state);
void QNN_WriteMapStateJson(FILE *out, const qnn_map_state_t *map_state);
int QNN_NearestRegionId(const qnn_map_state_t *map_state, const vec3_t point);
void QNN_SemanticReset(const qnn_map_state_t *map_state);
void QNN_SemanticUpdate(const qnn_map_state_t *map_state, const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag);
void QNN_WriteTokenStepBinary(FILE *out, const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag);
void QNN_WriteObsBuffer(FILE *out, const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag);
void QNN_TrainingResetEpisode(void);
void QNN_TrainingResetTick(void);
void QNN_TrainingParseRewardWeights(const char *line);
void QNN_WriteTrainingExtrasBinary(FILE *out, const qnn_snapshot_t *snapshot, int tick, int steps, qboolean reset_flag);
int QNN_HandleNavQuery(const char *line);

/* Map Quake weapon_id (1-8) to the canonical weapon class embedding (0-4).
   Axe/SG/SSG → 0 (Shotgun), NG/SNG → 1, GL → 2, RL → 3, LG → 4. */
/* Weapon classes from weapon_id (bit-position of IT_ flag):
   1=SG, 2=SSG, 3=NG, 4=SNG, 5=GL, 6=RL, 7=LG, 13=Axe.
   Classes: 0=axe, 1=hitscan(SG/SSG), 2=nails(NG/SNG),
            3=explosive(GL), 4=rocket(RL), 5=lightning(LG). */
/* ── Common math macros ──────────────────────────────────────────── */

#define QNN_Clamp(v, lo, hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))
#define QNN_Normalize(v, scale) (QNN_Clamp((v) / (scale), -1.0f, 1.0f))
#define QNN_AngleSinDeg(d) (sinf((d) * ((float)M_PI / 180.0f)))
#define QNN_AngleCosDeg(d) (cosf((d) * ((float)M_PI / 180.0f)))
#define QNN_DistSq(a, b) (((a)[0]-(b)[0])*((a)[0]-(b)[0]) + ((a)[1]-(b)[1])*((a)[1]-(b)[1]) + ((a)[2]-(b)[2])*((a)[2]-(b)[2]))
#define QNN_VecLength(v) ((float)sqrt((double)((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])))

/* Build a forward direction vector from Quake view angles (pitch, yaw). */
static inline void QNN_ForwardFromAngles(const vec3_t angles, vec3_t out)
{
	float cp = cosf(angles[0] * ((float)M_PI / 180.0f));
	float sp = sinf(angles[0] * ((float)M_PI / 180.0f));
	float cy = cosf(angles[1] * ((float)M_PI / 180.0f));
	float sy = sinf(angles[1] * ((float)M_PI / 180.0f));
	out[0] = cp * cy;
	out[1] = cp * sy;
	out[2] = -sp;
}

/* ── Binary write helpers (little-endian) ────────────────────────── */

void QNN_WriteU16LE(FILE *out, uint16_t value);
void QNN_WriteU32LE(FILE *out, uint32_t value);
void QNN_WriteI16LE(FILE *out, int value);
void QNN_WriteI32LE(FILE *out, int32_t value);
void QNN_WriteF32LE(FILE *out, float value);

/* ── Weapon class mapping ────────────────────────────────────────── */

#define QNN_WEAPON_CLASS_COUNT 6
static inline int qnn_weapon_class_from_id(int weapon_id)
{
	switch (weapon_id)
	{
	case 1: case 2: return 1;  /* SG, SSG */
	case 3: case 4: return 2;  /* NG, SNG */
	case 5: return 3;          /* GL */
	case 6: return 4;          /* RL */
	case 7: return 5;          /* LG */
	default: return 0;         /* Axe */
	}
}

#endif
