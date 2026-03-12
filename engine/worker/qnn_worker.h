#ifndef QNN_WORKER_H
#define QNN_WORKER_H

#include <stdio.h>

#include "quakedef.h"

#define QNN_WORKER_ACTION_HISTORY 2
#define QNN_WORKER_MAX_PROPERTY_KEY 64
#define QNN_WORKER_MAX_PROPERTY_VALUE 256
#define QNN_WORKER_MAX_CLASSNAME 64
#define QNN_WORKER_MAX_CATEGORY 16
#define QNN_WORKER_MAX_OBJECT_ID 32
#define QNN_WORKER_MAX_MAP_ID 64
#define QNN_WORKER_MAX_MODEL_NAME 64
#define QNN_WORKER_LOOK_NEUTRAL_LABEL 12
#define QNN_WORKER_MAX_SOUNDS 16
#define QNN_WORKER_MAX_SOUND_NAME 64
#define QNN_WORKER_MAX_VISIBLE 64
#define QNN_WORKER_MAX_EVENTS 16
#define QNN_WORKER_MAX_DYNAMIC_OBJECTS 128
#define QNN_WORKER_MAX_TOKEN_OBJECTS 128
#define QNN_WORKER_MAX_EVENT_ATOMS 256
#define QNN_WORKER_SPATIAL_TOKEN_COUNT 9
#define QNN_WORKER_MAX_TRAIN_DAMAGE 64
#define QNN_WORKER_MAX_TRAIN_ITEMS 64
#define QNN_WORKER_MAX_TRAIN_DEATHS 16
#define QNN_WORKER_MAX_TRAIN_SPAWNS 16

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
	char	name[QNN_WORKER_MAX_SOUND_NAME];
} qnn_worker_sound_event_t;

typedef struct
{
	char	key[QNN_WORKER_MAX_PROPERTY_KEY];
	char	value[QNN_WORKER_MAX_PROPERTY_VALUE];
} qnn_worker_property_t;

typedef struct
{
	char	object_id[QNN_WORKER_MAX_OBJECT_ID];
	char	category[QNN_WORKER_MAX_CATEGORY];
	char	classname[QNN_WORKER_MAX_CLASSNAME];
	int	region_id;
	vec3_t	origin;
	vec3_t	angles;
	qnn_worker_property_t *properties;
	int	property_count;
} qnn_worker_static_object_t;

typedef struct
{
	int	region_id;
	vec3_t	center;
	vec3_t	bounds_min;
	vec3_t	bounds_max;
	int	*neighbors;
	int	neighbor_count;
	int	*object_indices;
	int	object_count;
	float	distance_to_goal;
} qnn_worker_region_t;

typedef struct
{
	char	requested_map_id[QNN_WORKER_MAX_MAP_ID];
	char	map_name[QNN_WORKER_MAX_MAP_ID];
	char	source[32];
	qnn_worker_region_t *regions;
	int	region_count;
	qnn_worker_static_object_t *static_objects;
	int	static_object_count;
	int	*spawn_region_ids;
	int	spawn_region_count;
	int	*goal_region_ids;
	int	goal_region_count;
	float	max_distance_to_goal;
} qnn_worker_map_state_t;

typedef struct
{
	int	move;
	int	strafe;
	int	look_yaw;
	int	look_pitch;
	int	look_yaw_count;
	int	look_pitch_count;
	int	fire;
	int	jump;
	int	weapon;
} qnn_worker_action_t;

typedef struct
{
	int	entity_key;
	char	entity_id[32];
	int	entity_num;
	char	classname[QNN_WORKER_MAX_CLASSNAME];
	char	model_name[QNN_WORKER_MAX_MODEL_NAME];
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
	qboolean static_proxy;
} qnn_worker_visible_entity_t;

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
} qnn_worker_event_t;

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
	qboolean goal_reached;
	qboolean done;
	char	done_reason[32];
	qnn_worker_visible_entity_t visible[QNN_WORKER_MAX_VISIBLE];
	int	visible_count;
	qnn_worker_event_t events[QNN_WORKER_MAX_EVENTS];
	int	event_count;
	int	damage_dealt;
	int	hit_count;
	int	shots_fired;
	int	damage_weapon_id;
	qnn_worker_sound_event_t sounds[QNN_WORKER_MAX_SOUNDS];
	int	sound_count;
	qnn_worker_action_t action_label;
} qnn_worker_snapshot_t;

extern qnn_worker_action_t qnn_worker_pending_action;
extern qnn_worker_sound_event_t qnn_worker_sound_buffer[QNN_WORKER_MAX_SOUNDS];
extern int qnn_worker_sound_count;

/* Shared globals (defined in qnn_worker_common.c) */
extern qnn_worker_map_state_t qnn_worker_map_state;
extern char qnn_worker_basedir_storage[MAX_OSPATH];
extern char *basedir;
extern char *cachedir;

/* Common utilities (qnn_worker_common.c) */
void qnn_worker_resolve_basedir(char *out, size_t out_size);
int qnn_json_extract_int(const char *line, const char *key, int fallback);
qboolean qnn_json_extract_string(const char *line, const char *key, char *out, size_t out_size);
qboolean qnn_worker_prepare_map(const char *requested_map_id, char *error, size_t error_size);
void qnn_worker_write_json_string(FILE *out, const char *text);
void qnn_worker_write_error(const char *message);
const char *qnn_worker_prog_string(string_t value);
int qnn_worker_weapon_id(void);
int qnn_worker_current_frags(void);
void qnn_worker_capture_visible_entities(qnn_worker_snapshot_t *snapshot, float fixed_dt);
void qnn_worker_capture_base_snapshot(qnn_worker_snapshot_t *snapshot);
void qnn_worker_drain_sounds(qnn_worker_snapshot_t *snapshot);

void qnn_worker_clear_action(qnn_worker_action_t *action);
qboolean qnn_worker_build_map_state(qnn_worker_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size);
void qnn_worker_free_map_state(qnn_worker_map_state_t *map_state);
void qnn_worker_write_map_state_json(FILE *out, const qnn_worker_map_state_t *map_state);
int qnn_worker_nearest_region_id(const qnn_worker_map_state_t *map_state, const vec3_t point);
float qnn_worker_goal_progress(const qnn_worker_map_state_t *map_state, int region_id);
qboolean qnn_worker_is_goal_region(const qnn_worker_map_state_t *map_state, int region_id);
void qnn_worker_semantic_reset(const qnn_worker_map_state_t *map_state);
void qnn_worker_semantic_update(const qnn_worker_map_state_t *map_state, const qnn_worker_snapshot_t *snapshot, float dt, qboolean reset_flag);
void qnn_worker_write_token_step_binary(FILE *out, const qnn_worker_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag);
void qnn_worker_training_reset_episode(void);
void qnn_worker_training_reset_tick(void);
void qnn_worker_write_training_extras_binary(FILE *out, const qnn_worker_snapshot_t *snapshot, int tick, int steps, qboolean reset_flag);

#endif
