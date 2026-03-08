#ifndef NQ_WORKER_H
#define NQ_WORKER_H

#include <stdio.h>

#include "quakedef.h"

#define NQ_WORKER_ACTION_HISTORY 2
#define NQ_WORKER_MAX_PROPERTY_KEY 64
#define NQ_WORKER_MAX_PROPERTY_VALUE 256
#define NQ_WORKER_MAX_CLASSNAME 64
#define NQ_WORKER_MAX_CATEGORY 16
#define NQ_WORKER_MAX_OBJECT_ID 32
#define NQ_WORKER_MAX_MAP_ID 64
#define NQ_WORKER_LOOK_NEUTRAL_LABEL 12

typedef struct
{
	char	key[NQ_WORKER_MAX_PROPERTY_KEY];
	char	value[NQ_WORKER_MAX_PROPERTY_VALUE];
} nq_worker_property_t;

typedef struct
{
	char	object_id[NQ_WORKER_MAX_OBJECT_ID];
	char	category[NQ_WORKER_MAX_CATEGORY];
	char	classname[NQ_WORKER_MAX_CLASSNAME];
	int	region_id;
	vec3_t	origin;
	vec3_t	angles;
	nq_worker_property_t *properties;
	int	property_count;
} nq_worker_static_object_t;

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
} nq_worker_region_t;

typedef struct
{
	char	requested_map_id[NQ_WORKER_MAX_MAP_ID];
	char	map_name[NQ_WORKER_MAX_MAP_ID];
	char	source[32];
	nq_worker_region_t *regions;
	int	region_count;
	nq_worker_static_object_t *static_objects;
	int	static_object_count;
	int	*spawn_region_ids;
	int	spawn_region_count;
	int	*goal_region_ids;
	int	goal_region_count;
	float	max_distance_to_goal;
} nq_worker_map_state_t;

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
} nq_worker_action_t;

extern nq_worker_action_t nq_worker_pending_action;

void nq_worker_clear_action(nq_worker_action_t *action);
qboolean nq_worker_build_map_state(nq_worker_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size);
void nq_worker_free_map_state(nq_worker_map_state_t *map_state);
void nq_worker_write_map_state_json(FILE *out, const nq_worker_map_state_t *map_state);
int nq_worker_nearest_region_id(const nq_worker_map_state_t *map_state, const vec3_t point);
float nq_worker_goal_progress(const nq_worker_map_state_t *map_state, int region_id);
qboolean nq_worker_is_goal_region(const nq_worker_map_state_t *map_state, int region_id);

#endif
