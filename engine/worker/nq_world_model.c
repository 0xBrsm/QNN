#include "nq_worker.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define NQ_WORKER_GRID_SIZE 256.0f
#define NQ_WORKER_GRID_HALF 128.0f
#define NQ_WORKER_Z_HALF 256.0f
#define NQ_WORKER_OBJECT_PADDING 16.0f

typedef struct
{
	int	input_index;
	char	classname[NQ_WORKER_MAX_CLASSNAME];
	char	category[NQ_WORKER_MAX_CATEGORY];
	vec3_t	origin;
	vec3_t	angles;
	int	region_id;
	nq_worker_property_t *properties;
	int	property_count;
} nq_parsed_entity_t;

static int nq_worker_region_id_from_point(const vec3_t point)
{
	int gx;
	int gy;

	gx = (int)floor((point[0] / NQ_WORKER_GRID_SIZE) + 0.5f);
	gy = (int)floor((point[1] / NQ_WORKER_GRID_SIZE) + 0.5f);
	return (gx + 1024) * 2048 + (gy + 1024);
}

static void nq_worker_region_center(int region_id, vec3_t out)
{
	int gx;
	int gy;

	gx = region_id / 2048 - 1024;
	gy = region_id % 2048 - 1024;
	out[0] = gx * NQ_WORKER_GRID_SIZE;
	out[1] = gy * NQ_WORKER_GRID_SIZE;
	out[2] = 0.0f;
}

static int nq_worker_category_order(const char *category)
{
	if (!strcmp(category, "spawn"))
		return 0;
	if (!strcmp(category, "goal"))
		return 1;
	if (!strcmp(category, "item"))
		return 2;
	if (!strcmp(category, "trigger"))
		return 3;
	if (!strcmp(category, "door"))
		return 4;
	if (!strcmp(category, "lift"))
		return 5;
	if (!strcmp(category, "mover"))
		return 6;
	if (!strcmp(category, "monster"))
		return 7;
	return 8;
}

static const char *nq_worker_classify(const char *classname)
{
	if (!strcasecmp(classname, "info_player_start")
		|| !strcasecmp(classname, "info_player_coop")
		|| !strcasecmp(classname, "info_player_deathmatch"))
		return "spawn";
	if (!strcasecmp(classname, "trigger_changelevel"))
		return "goal";
	if (!strncasecmp(classname, "item_", 5))
		return "item";
	if (!strncasecmp(classname, "trigger_", 8))
		return "trigger";
	if (!strncasecmp(classname, "func_door", 9))
		return "door";
	if (!strncasecmp(classname, "func_plat", 9)
		|| !strncasecmp(classname, "func_train", 10)
		|| !strncasecmp(classname, "func_button", 11))
		return "lift";
	if (!strncasecmp(classname, "func_", 5))
		return "mover";
	if (!strncasecmp(classname, "monster_", 8))
		return "monster";
	return "misc";
}

static void nq_worker_json_string(FILE *out, const char *text)
{
	const unsigned char *cursor;

	fputc('"', out);
	for (cursor = (const unsigned char *)text; *cursor; ++cursor)
	{
		if (*cursor == '\\' || *cursor == '"')
		{
			fputc('\\', out);
			fputc(*cursor, out);
			continue;
		}
		if (*cursor == '\n')
		{
			fputs("\\n", out);
			continue;
		}
		if (*cursor == '\r')
		{
			fputs("\\r", out);
			continue;
		}
		if (*cursor == '\t')
		{
			fputs("\\t", out);
			continue;
		}
		if (*cursor < 32)
		{
			fprintf(out, "\\u%04x", (unsigned int)*cursor);
			continue;
		}
		fputc(*cursor, out);
	}
	fputc('"', out);
}

static void nq_worker_free_parsed_entities(nq_parsed_entity_t *entities, int entity_count)
{
	int i;

	for (i = 0; i < entity_count; ++i)
	{
		free(entities[i].properties);
	}
	free(entities);
}

static qboolean nq_worker_append_property(nq_parsed_entity_t *entity, const char *key, const char *value)
{
	nq_worker_property_t *next;

	next = (nq_worker_property_t *)realloc(entity->properties, (entity->property_count + 1) * sizeof(*next));
	if (next == NULL)
		return false;
	entity->properties = next;
	memset(&entity->properties[entity->property_count], 0, sizeof(entity->properties[entity->property_count]));
	strncpy(entity->properties[entity->property_count].key, key, NQ_WORKER_MAX_PROPERTY_KEY - 1);
	strncpy(entity->properties[entity->property_count].value, value, NQ_WORKER_MAX_PROPERTY_VALUE - 1);
	entity->property_count += 1;
	return true;
}

static void nq_worker_skip_space(char **cursor)
{
	while (**cursor && isspace((unsigned char)**cursor))
	{
		*cursor += 1;
	}
}

static qboolean nq_worker_parse_quoted(char **cursor, char *out, size_t out_size)
{
	size_t index;

	nq_worker_skip_space(cursor);
	if (**cursor != '"')
		return false;
	*cursor += 1;
	index = 0;
	while (**cursor && **cursor != '"')
	{
		char ch;

		ch = **cursor;
		if (ch == '\\' && (*cursor)[1])
		{
			*cursor += 1;
			ch = **cursor;
		}
		if (index + 1 < out_size)
			out[index++] = ch;
		*cursor += 1;
	}
	if (**cursor != '"')
		return false;
	out[index] = 0;
	*cursor += 1;
	return true;
}

static qboolean nq_worker_parse_origin(const char *value, vec3_t out)
{
	float x;
	float y;
	float z;

	if (sscanf(value, "%f %f %f", &x, &y, &z) != 3)
		return false;
	out[0] = x;
	out[1] = y;
	out[2] = z;
	return true;
}

static void nq_worker_parse_angle_value(const char *value, vec3_t out)
{
	float yaw;

	yaw = 0.0f;
	out[0] = 0.0f;
	out[1] = 0.0f;
	out[2] = 0.0f;
	if (sscanf(value, "%f", &yaw) == 1)
		out[1] = yaw;
}

static void nq_worker_parse_angles_value(const char *value, vec3_t out)
{
	float pitch;
	float yaw;
	float roll;

	pitch = 0.0f;
	yaw = 0.0f;
	roll = 0.0f;
	if (sscanf(value, "%f %f %f", &pitch, &yaw, &roll) == 3)
	{
		out[0] = pitch;
		out[1] = yaw;
		out[2] = roll;
		return;
	}
	nq_worker_parse_angle_value(value, out);
}

static int nq_worker_compare_entities(const void *lhs_ptr, const void *rhs_ptr)
{
	const nq_parsed_entity_t *lhs;
	const nq_parsed_entity_t *rhs;
	int order_cmp;
	int classname_cmp;
	int axis;

	lhs = (const nq_parsed_entity_t *)lhs_ptr;
	rhs = (const nq_parsed_entity_t *)rhs_ptr;

	order_cmp = nq_worker_category_order(lhs->category) - nq_worker_category_order(rhs->category);
	if (order_cmp != 0)
		return order_cmp;

	classname_cmp = strcmp(lhs->classname, rhs->classname);
	if (classname_cmp != 0)
		return classname_cmp;

	for (axis = 0; axis < 3; ++axis)
	{
		if (lhs->origin[axis] < rhs->origin[axis])
			return -1;
		if (lhs->origin[axis] > rhs->origin[axis])
			return 1;
	}

	return lhs->input_index - rhs->input_index;
}

static int nq_worker_compare_regions(const void *lhs_ptr, const void *rhs_ptr)
{
	const nq_worker_region_t *lhs;
	const nq_worker_region_t *rhs;

	lhs = (const nq_worker_region_t *)lhs_ptr;
	rhs = (const nq_worker_region_t *)rhs_ptr;
	return lhs->region_id - rhs->region_id;
}

static int nq_worker_compare_ints(const void *lhs_ptr, const void *rhs_ptr)
{
	const int *lhs;
	const int *rhs;

	lhs = (const int *)lhs_ptr;
	rhs = (const int *)rhs_ptr;
	return *lhs - *rhs;
}

static qboolean nq_worker_push_unique_int(int **values, int *count, int value)
{
	int *next;
	int i;

	for (i = 0; i < *count; ++i)
	{
		if ((*values)[i] == value)
			return true;
	}

	next = (int *)realloc(*values, (*count + 1) * sizeof(*next));
	if (next == NULL)
		return false;
	*values = next;
	(*values)[*count] = value;
	*count += 1;
	return true;
}

static qboolean nq_worker_push_object_index(nq_worker_region_t *region, int object_index)
{
	int *next;

	next = (int *)realloc(region->object_indices, (region->object_count + 1) * sizeof(*next));
	if (next == NULL)
		return false;
	region->object_indices = next;
	region->object_indices[region->object_count] = object_index;
	region->object_count += 1;
	return true;
}

static nq_worker_region_t *nq_worker_get_or_add_region(nq_worker_map_state_t *map_state, int region_id)
{
	nq_worker_region_t *region;
	nq_worker_region_t *next;
	int i;

	for (i = 0; i < map_state->region_count; ++i)
	{
		if (map_state->regions[i].region_id == region_id)
			return &map_state->regions[i];
	}

	next = (nq_worker_region_t *)realloc(map_state->regions, (map_state->region_count + 1) * sizeof(*next));
	if (next == NULL)
		return NULL;
	map_state->regions = next;
	region = &map_state->regions[map_state->region_count];
	memset(region, 0, sizeof(*region));
	region->region_id = region_id;
	nq_worker_region_center(region_id, region->center);
	region->bounds_min[0] = region->center[0] - NQ_WORKER_GRID_HALF;
	region->bounds_min[1] = region->center[1] - NQ_WORKER_GRID_HALF;
	region->bounds_min[2] = -NQ_WORKER_Z_HALF;
	region->bounds_max[0] = region->center[0] + NQ_WORKER_GRID_HALF;
	region->bounds_max[1] = region->center[1] + NQ_WORKER_GRID_HALF;
	region->bounds_max[2] = NQ_WORKER_Z_HALF;
	map_state->region_count += 1;
	return region;
}

static void nq_worker_expand_region_bounds(nq_worker_region_t *region, const vec3_t origin)
{
	if (origin[0] - NQ_WORKER_OBJECT_PADDING < region->bounds_min[0])
		region->bounds_min[0] = origin[0] - NQ_WORKER_OBJECT_PADDING;
	if (origin[1] - NQ_WORKER_OBJECT_PADDING < region->bounds_min[1])
		region->bounds_min[1] = origin[1] - NQ_WORKER_OBJECT_PADDING;
	if (origin[2] - NQ_WORKER_OBJECT_PADDING < region->bounds_min[2])
		region->bounds_min[2] = origin[2] - NQ_WORKER_OBJECT_PADDING;
	if (origin[0] + NQ_WORKER_OBJECT_PADDING > region->bounds_max[0])
		region->bounds_max[0] = origin[0] + NQ_WORKER_OBJECT_PADDING;
	if (origin[1] + NQ_WORKER_OBJECT_PADDING > region->bounds_max[1])
		region->bounds_max[1] = origin[1] + NQ_WORKER_OBJECT_PADDING;
	if (origin[2] + NQ_WORKER_OBJECT_PADDING > region->bounds_max[2])
		region->bounds_max[2] = origin[2] + NQ_WORKER_OBJECT_PADDING;
}

static qboolean nq_worker_parse_entities(char *text, nq_parsed_entity_t **out_entities, int *out_count, char *error, size_t error_size)
{
	nq_parsed_entity_t *entities;
	int entity_count;
	char *cursor;
	int input_index;

	entities = NULL;
	entity_count = 0;
	cursor = text;
	input_index = 0;

	while (*cursor)
	{
		nq_parsed_entity_t entity;
		char key[NQ_WORKER_MAX_PROPERTY_KEY];
		char value[NQ_WORKER_MAX_PROPERTY_VALUE];
		qboolean has_origin;
		qboolean has_angles;
		nq_parsed_entity_t *next_entities;

		nq_worker_skip_space(&cursor);
		if (!*cursor)
			break;
		if (*cursor != '{')
		{
			cursor += 1;
			continue;
		}
		cursor += 1;

		memset(&entity, 0, sizeof(entity));
		entity.input_index = input_index;
		has_origin = false;
		has_angles = false;

		while (*cursor)
		{
			nq_worker_skip_space(&cursor);
			if (*cursor == '}')
			{
				cursor += 1;
				break;
			}
			if (!nq_worker_parse_quoted(&cursor, key, sizeof(key)))
			{
				snprintf(error, error_size, "Failed to parse entity key");
				nq_worker_free_parsed_entities(entities, entity_count);
				free(entity.properties);
				return false;
			}
			if (!nq_worker_parse_quoted(&cursor, value, sizeof(value)))
			{
				snprintf(error, error_size, "Failed to parse entity value");
				nq_worker_free_parsed_entities(entities, entity_count);
				free(entity.properties);
				return false;
			}

			if (!strcmp(key, "classname"))
			{
				strncpy(entity.classname, value, sizeof(entity.classname) - 1);
			}
			else if (!strcmp(key, "origin"))
			{
				has_origin = nq_worker_parse_origin(value, entity.origin);
			}
			else if (!strcmp(key, "angles"))
			{
				nq_worker_parse_angles_value(value, entity.angles);
				has_angles = true;
			}
			else if (!strcmp(key, "angle"))
			{
				if (!has_angles)
					nq_worker_parse_angle_value(value, entity.angles);
			}
			else if (!nq_worker_append_property(&entity, key, value))
			{
				snprintf(error, error_size, "Out of memory while parsing entity properties");
				nq_worker_free_parsed_entities(entities, entity_count);
				free(entity.properties);
				return false;
			}
		}

		input_index += 1;
		if (!has_origin)
		{
			free(entity.properties);
			continue;
		}

		strncpy(entity.category, nq_worker_classify(entity.classname), sizeof(entity.category) - 1);
		entity.region_id = nq_worker_region_id_from_point(entity.origin);
		next_entities = (nq_parsed_entity_t *)realloc(entities, (entity_count + 1) * sizeof(*next_entities));
		if (next_entities == NULL)
		{
			snprintf(error, error_size, "Out of memory while parsing entity list");
			nq_worker_free_parsed_entities(entities, entity_count);
			free(entity.properties);
			return false;
		}
		entities = next_entities;
		entities[entity_count] = entity;
		entity_count += 1;
	}

	*out_entities = entities;
	*out_count = entity_count;
	return true;
}

static qboolean nq_worker_add_neighbors(nq_worker_region_t *regions, int region_count)
{
	int src_index;

	for (src_index = 0; src_index < region_count; ++src_index)
	{
		int best_ids[3];
		int best_dist[3];
		int dst_index;
		int slot;
		int rank;
		int src_gx;
		int src_gy;

		best_ids[0] = -1;
		best_ids[1] = -1;
		best_ids[2] = -1;
		best_dist[0] = 0x7fffffff;
		best_dist[1] = 0x7fffffff;
		best_dist[2] = 0x7fffffff;

		src_gx = regions[src_index].region_id / 2048 - 1024;
		src_gy = regions[src_index].region_id % 2048 - 1024;

		for (dst_index = 0; dst_index < region_count; ++dst_index)
		{
			int dst_gx;
			int dst_gy;
			int manhattan;

			if (src_index == dst_index)
				continue;
			dst_gx = regions[dst_index].region_id / 2048 - 1024;
			dst_gy = regions[dst_index].region_id % 2048 - 1024;
			manhattan = abs(src_gx - dst_gx) + abs(src_gy - dst_gy);

			slot = -1;
			for (rank = 0; rank < 3; ++rank)
			{
				if (manhattan < best_dist[rank]
					|| (manhattan == best_dist[rank] && regions[dst_index].region_id < best_ids[rank]))
				{
					slot = rank;
					break;
				}
			}
			if (slot < 0)
				continue;
			for (rank = 2; rank > slot; --rank)
			{
				best_dist[rank] = best_dist[rank - 1];
				best_ids[rank] = best_ids[rank - 1];
			}
			best_dist[slot] = manhattan;
			best_ids[slot] = regions[dst_index].region_id;
		}

		for (rank = 0; rank < 3; ++rank)
		{
			int dst_region_index;

			if (best_ids[rank] < 0)
				continue;
			if (!nq_worker_push_unique_int(&regions[src_index].neighbors, &regions[src_index].neighbor_count, best_ids[rank]))
				return false;
			for (dst_region_index = 0; dst_region_index < region_count; ++dst_region_index)
			{
				if (regions[dst_region_index].region_id == best_ids[rank])
				{
					if (!nq_worker_push_unique_int(&regions[dst_region_index].neighbors, &regions[dst_region_index].neighbor_count, regions[src_index].region_id))
						return false;
					break;
				}
			}
		}
	}

	for (src_index = 0; src_index < region_count; ++src_index)
	{
		if (regions[src_index].neighbor_count > 1)
			qsort(regions[src_index].neighbors, regions[src_index].neighbor_count, sizeof(int), nq_worker_compare_ints);
	}
	return true;
}

static void nq_worker_compute_distances(nq_worker_map_state_t *map_state)
{
	int *queue;
	int queue_head;
	int queue_tail;
	int region_index;
	float max_distance;

	queue = NULL;
	queue_head = 0;
	queue_tail = 0;
	max_distance = 0.0f;

	for (region_index = 0; region_index < map_state->region_count; ++region_index)
	{
		map_state->regions[region_index].distance_to_goal = 999999.0f;
	}

	queue = (int *)malloc(map_state->region_count * sizeof(*queue));
	if (queue == NULL)
	{
		for (region_index = 0; region_index < map_state->region_count; ++region_index)
			map_state->regions[region_index].distance_to_goal = 0.0f;
		map_state->max_distance_to_goal = 0.0f;
		return;
	}

	for (region_index = 0; region_index < map_state->goal_region_count; ++region_index)
	{
		int goal_id;
		int i;

		goal_id = map_state->goal_region_ids[region_index];
		for (i = 0; i < map_state->region_count; ++i)
		{
			if (map_state->regions[i].region_id == goal_id)
			{
				map_state->regions[i].distance_to_goal = 0.0f;
				queue[queue_tail++] = i;
				break;
			}
		}
	}

	while (queue_head < queue_tail)
	{
		int current_index;
		nq_worker_region_t *current_region;
		int neighbor_pos;

		current_index = queue[queue_head++];
		current_region = &map_state->regions[current_index];

		for (neighbor_pos = 0; neighbor_pos < current_region->neighbor_count; ++neighbor_pos)
		{
			int neighbor_id;
			int neighbor_index;

			neighbor_id = current_region->neighbors[neighbor_pos];
			for (neighbor_index = 0; neighbor_index < map_state->region_count; ++neighbor_index)
			{
				if (map_state->regions[neighbor_index].region_id != neighbor_id)
					continue;
				if (map_state->regions[neighbor_index].distance_to_goal > current_region->distance_to_goal + 1.0f)
				{
					map_state->regions[neighbor_index].distance_to_goal = current_region->distance_to_goal + 1.0f;
					queue[queue_tail++] = neighbor_index;
				}
				break;
			}
		}
	}

	for (region_index = 0; region_index < map_state->region_count; ++region_index)
	{
		if (map_state->regions[region_index].distance_to_goal < 999999.0f
			&& map_state->regions[region_index].distance_to_goal > max_distance)
			max_distance = map_state->regions[region_index].distance_to_goal;
	}
	for (region_index = 0; region_index < map_state->region_count; ++region_index)
	{
		if (map_state->regions[region_index].distance_to_goal >= 999999.0f)
			map_state->regions[region_index].distance_to_goal = max_distance + 1.0f;
	}
	map_state->max_distance_to_goal = map_state->region_count > 0 ? max_distance : 0.0f;
	if (map_state->max_distance_to_goal <= 0.0f && map_state->region_count > 0)
		map_state->max_distance_to_goal = 1.0f;

	free(queue);
}

void nq_worker_clear_action(nq_worker_action_t *action)
{
	action->move = 0;
	action->strafe = 0;
	action->look_yaw = NQ_WORKER_LOOK_NEUTRAL_LABEL;
	action->look_pitch = NQ_WORKER_LOOK_NEUTRAL_LABEL;
	action->look_yaw_count = 0;
	action->look_pitch_count = 0;
	action->fire = 0;
	action->jump = 0;
	action->weapon = 0;
}

void nq_worker_free_map_state(nq_worker_map_state_t *map_state)
{
	int i;

	if (map_state == NULL)
		return;

	for (i = 0; i < map_state->region_count; ++i)
	{
		free(map_state->regions[i].neighbors);
		free(map_state->regions[i].object_indices);
	}
	for (i = 0; i < map_state->static_object_count; ++i)
	{
		free(map_state->static_objects[i].properties);
	}

	free(map_state->regions);
	free(map_state->static_objects);
	free(map_state->spawn_region_ids);
	free(map_state->goal_region_ids);
	memset(map_state, 0, sizeof(*map_state));
}

qboolean nq_worker_build_map_state(nq_worker_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size)
{
	char path[MAX_QPATH];
	byte *raw;
	dheader_t *header;
	int entity_ofs;
	int entity_len;
	char *entity_text;
	nq_parsed_entity_t *entities;
	int entity_count;
	int stable_index;

	memset(out, 0, sizeof(*out));
	strncpy(out->requested_map_id, requested_map_id, sizeof(out->requested_map_id) - 1);
	strncpy(out->map_name, map_name, sizeof(out->map_name) - 1);
	strncpy(out->source, "bsp_entities", sizeof(out->source) - 1);

	snprintf(path, sizeof(path), "maps/%s.bsp", map_name);
	raw = COM_LoadTempFile(path);
	if (raw == NULL)
	{
		snprintf(error, error_size, "Could not load %s from Quake data path", path);
		return false;
	}

	header = (dheader_t *)raw;
	if (LittleLong(header->version) != BSPVERSION)
	{
		snprintf(error, error_size, "Unsupported BSP version %d", LittleLong(header->version));
		return false;
	}

	entity_ofs = LittleLong(header->lumps[LUMP_ENTITIES].fileofs);
	entity_len = LittleLong(header->lumps[LUMP_ENTITIES].filelen);
	entity_text = (char *)(raw + entity_ofs);
	entities = NULL;
	entity_count = 0;

	if (!nq_worker_parse_entities(entity_text, &entities, &entity_count, error, error_size))
		return false;

	if (entity_count > 1)
		qsort(entities, entity_count, sizeof(*entities), nq_worker_compare_entities);

	for (stable_index = 0; stable_index < entity_count; ++stable_index)
	{
		nq_worker_static_object_t *object;
		nq_worker_static_object_t *next_objects;
		nq_worker_region_t *region;

		next_objects = (nq_worker_static_object_t *)realloc(out->static_objects, (out->static_object_count + 1) * sizeof(*next_objects));
		if (next_objects == NULL)
		{
			snprintf(error, error_size, "Out of memory while building static object list");
			nq_worker_free_parsed_entities(entities, entity_count);
			nq_worker_free_map_state(out);
			return false;
		}
		out->static_objects = next_objects;
		object = &out->static_objects[out->static_object_count];
		memset(object, 0, sizeof(*object));
		strncpy(object->category, entities[stable_index].category, sizeof(object->category) - 1);
		strncpy(object->classname, entities[stable_index].classname, sizeof(object->classname) - 1);
		snprintf(object->object_id, sizeof(object->object_id), "%s_%04d", entities[stable_index].category, stable_index);
		object->region_id = entities[stable_index].region_id;
		VectorCopy(entities[stable_index].origin, object->origin);
		VectorCopy(entities[stable_index].angles, object->angles);
		object->properties = entities[stable_index].properties;
		object->property_count = entities[stable_index].property_count;
		entities[stable_index].properties = NULL;
		entities[stable_index].property_count = 0;
		out->static_object_count += 1;

		region = nq_worker_get_or_add_region(out, object->region_id);
		if (region == NULL)
		{
			snprintf(error, error_size, "Out of memory while building region list");
			nq_worker_free_parsed_entities(entities, entity_count);
			nq_worker_free_map_state(out);
			return false;
		}
		if (!nq_worker_push_object_index(region, out->static_object_count - 1))
		{
			snprintf(error, error_size, "Out of memory while attaching region objects");
			nq_worker_free_parsed_entities(entities, entity_count);
			nq_worker_free_map_state(out);
			return false;
		}
		nq_worker_expand_region_bounds(region, object->origin);

		if (!strcmp(object->category, "spawn"))
		{
			if (!nq_worker_push_unique_int(&out->spawn_region_ids, &out->spawn_region_count, object->region_id))
			{
				snprintf(error, error_size, "Out of memory while collecting spawn regions");
				nq_worker_free_parsed_entities(entities, entity_count);
				nq_worker_free_map_state(out);
				return false;
			}
		}
		if (!strcmp(object->category, "goal"))
		{
			if (!nq_worker_push_unique_int(&out->goal_region_ids, &out->goal_region_count, object->region_id))
			{
				snprintf(error, error_size, "Out of memory while collecting goal regions");
				nq_worker_free_parsed_entities(entities, entity_count);
				nq_worker_free_map_state(out);
				return false;
			}
		}
	}

	nq_worker_free_parsed_entities(entities, entity_count);

	if (out->region_count == 0)
	{
		nq_worker_region_t *region;

		region = nq_worker_get_or_add_region(out, 0);
		if (region == NULL)
		{
			snprintf(error, error_size, "Out of memory while creating fallback region");
			nq_worker_free_map_state(out);
			return false;
		}
	}

	if (out->spawn_region_count == 0)
	{
		if (!nq_worker_push_unique_int(&out->spawn_region_ids, &out->spawn_region_count, out->regions[0].region_id))
		{
			snprintf(error, error_size, "Out of memory while setting fallback spawn region");
			nq_worker_free_map_state(out);
			return false;
		}
	}
	if (out->goal_region_count == 0)
	{
		int fallback_goal;

		fallback_goal = out->regions[out->region_count - 1].region_id;
		if (!nq_worker_push_unique_int(&out->goal_region_ids, &out->goal_region_count, fallback_goal))
		{
			snprintf(error, error_size, "Out of memory while setting fallback goal region");
			nq_worker_free_map_state(out);
			return false;
		}
	}

	if (out->region_count > 1)
		qsort(out->regions, out->region_count, sizeof(*out->regions), nq_worker_compare_regions);
	if (out->spawn_region_count > 1)
		qsort(out->spawn_region_ids, out->spawn_region_count, sizeof(int), nq_worker_compare_ints);
	if (out->goal_region_count > 1)
		qsort(out->goal_region_ids, out->goal_region_count, sizeof(int), nq_worker_compare_ints);

	if (!nq_worker_add_neighbors(out->regions, out->region_count))
	{
		snprintf(error, error_size, "Out of memory while computing region graph");
		nq_worker_free_map_state(out);
		return false;
	}

	nq_worker_compute_distances(out);
	return true;
}

void nq_worker_write_map_state_json(FILE *out, const nq_worker_map_state_t *map_state)
{
	int i;

	fprintf(out, "{\"goal_region_ids\":[");
	for (i = 0; i < map_state->goal_region_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "%d", map_state->goal_region_ids[i]);
	}
	fprintf(out, "],\"map_id\":");
	nq_worker_json_string(out, map_state->requested_map_id);
	fprintf(out, ",\"metadata\":{\"distance_to_goal\":{");
	for (i = 0; i < map_state->region_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		nq_worker_json_string(out, va("%d", map_state->regions[i].region_id));
		fprintf(out, ":%.3f", map_state->regions[i].distance_to_goal);
	}
	fprintf(out, "},\"grid_size\":%.1f,\"max_distance_to_goal\":%.3f,\"region_count\":%d,\"source\":",
		NQ_WORKER_GRID_SIZE,
		map_state->max_distance_to_goal,
		map_state->region_count);
	nq_worker_json_string(out, map_state->source);
	fprintf(out, ",\"static_object_count\":%d},\"regions\":[", map_state->static_object_count);

	for (i = 0; i < map_state->region_count; ++i)
	{
		int j;
		const nq_worker_region_t *region;

		region = &map_state->regions[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out,
			"{\"bounds_max\":[%.1f,%.1f,%.1f],\"bounds_min\":[%.1f,%.1f,%.1f],\"center\":[%.1f,%.1f,%.1f],\"neighbors\":[",
			region->bounds_max[0], region->bounds_max[1], region->bounds_max[2],
			region->bounds_min[0], region->bounds_min[1], region->bounds_min[2],
			region->center[0], region->center[1], region->center[2]);
		for (j = 0; j < region->neighbor_count; ++j)
		{
			if (j > 0)
				fputc(',', out);
			fprintf(out, "%d", region->neighbors[j]);
		}
		fprintf(out, "],\"object_ids\":[");
		for (j = 0; j < region->object_count; ++j)
		{
			if (j > 0)
				fputc(',', out);
			nq_worker_json_string(out, map_state->static_objects[region->object_indices[j]].object_id);
		}
		fprintf(out, "],\"region_id\":%d,\"visibility_hints\":[", region->region_id);
		for (j = 0; j < region->neighbor_count; ++j)
		{
			if (j > 0)
				fputc(',', out);
			fprintf(out, "%d", region->neighbors[j]);
		}
		fprintf(out, "]}");
	}

	fprintf(out, "],\"spawn_region_ids\":[");
	for (i = 0; i < map_state->spawn_region_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "%d", map_state->spawn_region_ids[i]);
	}
	fprintf(out, "],\"static_objects\":[");

	for (i = 0; i < map_state->static_object_count; ++i)
	{
		int j;
		const nq_worker_static_object_t *object;

		object = &map_state->static_objects[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"angles\":[%.1f,%.1f,%.1f],\"category\":",
			object->angles[0], object->angles[1], object->angles[2]);
		nq_worker_json_string(out, object->category);
		fprintf(out, ",\"classname\":");
		nq_worker_json_string(out, object->classname);
		fprintf(out, ",\"object_id\":");
		nq_worker_json_string(out, object->object_id);
		fprintf(out, ",\"origin\":[%.1f,%.1f,%.1f],\"properties\":{",
			object->origin[0], object->origin[1], object->origin[2]);
		for (j = 0; j < object->property_count; ++j)
		{
			if (j > 0)
				fputc(',', out);
			nq_worker_json_string(out, object->properties[j].key);
			fputc(':', out);
			nq_worker_json_string(out, object->properties[j].value);
		}
		fprintf(out, "},\"region_id\":%d}", object->region_id);
	}

	fprintf(out, "]}");
}

int nq_worker_nearest_region_id(const nq_worker_map_state_t *map_state, const vec3_t point)
{
	int candidate;
	int i;
	int best_region_id;
	float best_distance;

	candidate = nq_worker_region_id_from_point(point);
	for (i = 0; i < map_state->region_count; ++i)
	{
		if (map_state->regions[i].region_id == candidate)
			return candidate;
	}

	best_region_id = map_state->region_count > 0 ? map_state->regions[0].region_id : 0;
	best_distance = 999999999.0f;
	for (i = 0; i < map_state->region_count; ++i)
	{
		float dx;
		float dy;
		float distance;

		dx = point[0] - map_state->regions[i].center[0];
		dy = point[1] - map_state->regions[i].center[1];
		distance = dx * dx + dy * dy;
		if (distance < best_distance)
		{
			best_distance = distance;
			best_region_id = map_state->regions[i].region_id;
		}
	}

	return best_region_id;
}

float nq_worker_goal_progress(const nq_worker_map_state_t *map_state, int region_id)
{
	int i;
	float distance;
	float max_distance;

	max_distance = map_state->max_distance_to_goal;
	if (max_distance <= 0.0f)
		return 0.0f;

	distance = max_distance;
	for (i = 0; i < map_state->region_count; ++i)
	{
		if (map_state->regions[i].region_id == region_id)
		{
			distance = map_state->regions[i].distance_to_goal;
			break;
		}
	}

	if (distance < 0.0f)
		distance = 0.0f;
	if (distance > max_distance)
		distance = max_distance;
	return 1.0f - (distance / max_distance);
}

qboolean nq_worker_is_goal_region(const nq_worker_map_state_t *map_state, int region_id)
{
	int i;

	for (i = 0; i < map_state->goal_region_count; ++i)
	{
		if (map_state->goal_region_ids[i] == region_id)
			return true;
	}
	return false;
}
