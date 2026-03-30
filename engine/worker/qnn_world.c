#include "qnn.h"
#include "qnn_nav_oracle.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define QNN_WORKER_GRID_SIZE 256.0f
#define QNN_WORKER_GRID_HALF 128.0f
#define QNN_WORKER_Z_HALF 256.0f
#define QNN_WORKER_OBJECT_PADDING 16.0f
#define QNN_NAV_CELL_SIZE 16.0f
#define QNN_NAV_CELL_HEIGHT 8.0f
#define QNN_NAV_WALKABLE_SLOPE_ANGLE 45.0f
#define QNN_NAV_WALKABLE_HEIGHT 56.0f
#define QNN_NAV_WALKABLE_CLIMB 18.0f
#define QNN_NAV_WALKABLE_RADIUS 16.0f
#define QNN_NAV_MAX_EDGE_LEN 192.0f
#define QNN_NAV_MAX_SIMPLIFICATION_ERROR 1.3f
#define QNN_NAV_MIN_REGION_SIZE 8
#define QNN_NAV_MERGE_REGION_SIZE 20
#define QNN_NAV_MAX_VERTS_PER_POLY 6
#define QNN_NAV_DETAIL_SAMPLE_DISTANCE 6.0f
#define QNN_NAV_DETAIL_SAMPLE_MAX_ERROR 1.0f

typedef struct
{
	int	input_index;
	char	classname[QNN_MAX_CLASSNAME];
	char	category[QNN_MAX_CATEGORY];
	vec3_t	origin;
	vec3_t	angles;
	int	region_id;
	qnn_property_t *properties;
	int	property_count;
} qnn_parsed_entity_t;

static void QNN_DefaultNavmeshConfig(qnn_navmesh_build_config_t *config)
{
	memset(config, 0, sizeof(*config));
	config->cell_size = QNN_NAV_CELL_SIZE;
	config->cell_height = QNN_NAV_CELL_HEIGHT;
	config->walkable_slope_angle = QNN_NAV_WALKABLE_SLOPE_ANGLE;
	config->walkable_height = QNN_NAV_WALKABLE_HEIGHT;
	config->walkable_climb = QNN_NAV_WALKABLE_CLIMB;
	config->walkable_radius = QNN_NAV_WALKABLE_RADIUS;
	config->max_edge_len = QNN_NAV_MAX_EDGE_LEN;
	config->max_simplification_error = QNN_NAV_MAX_SIMPLIFICATION_ERROR;
	config->min_region_size = QNN_NAV_MIN_REGION_SIZE;
	config->merge_region_size = QNN_NAV_MERGE_REGION_SIZE;
	config->max_verts_per_poly = QNN_NAV_MAX_VERTS_PER_POLY;
	config->detail_sample_distance = QNN_NAV_DETAIL_SAMPLE_DISTANCE;
	config->detail_sample_max_error = QNN_NAV_DETAIL_SAMPLE_MAX_ERROR;
}

static qboolean QNN_BspLumpView(
	const byte *raw,
	const dheader_t *header,
	int lump_index,
	size_t element_size,
	const void **out_data,
	int *out_count,
	char *error,
	size_t error_size)
{
	int offset;
	int length;

	offset = LittleLong(header->lumps[lump_index].fileofs);
	length = LittleLong(header->lumps[lump_index].filelen);
	if (offset < 0 || length < 0 || (element_size > 0 && (length % (int)element_size) != 0))
	{
		snprintf(error, error_size, "Invalid BSP lump %d", lump_index);
		return false;
	}
	*out_data = raw + offset;
	*out_count = element_size > 0 ? (length / (int)element_size) : 0;
	return true;
}

static int QNN_CountNavmeshTriangles(const dmodel_t *models, int model_count, const dface_t *faces, int face_count, const texinfo_t *texinfo, int texinfo_count)
{
	const dmodel_t *world_model;
	int first_face;
	int num_faces;
	int face_index;
	int triangle_count;

	if (model_count <= 0)
		return 0;

	world_model = &models[0];
	first_face = LittleLong(world_model->firstface);
	num_faces = LittleLong(world_model->numfaces);
	if (first_face < 0 || num_faces < 0 || first_face + num_faces > face_count)
		return 0;

	triangle_count = 0;
	for (face_index = 0; face_index < num_faces; ++face_index)
	{
		const dface_t *face;
		int edge_count;
		int texinfo_index;

		face = &faces[first_face + face_index];
		edge_count = LittleShort(face->numedges);
		texinfo_index = LittleShort(face->texinfo);
		if (edge_count < 3 || texinfo_index < 0 || texinfo_index >= texinfo_count)
			continue;
		if (LittleLong(texinfo[texinfo_index].flags) & TEX_SPECIAL)
			continue;
		triangle_count += edge_count - 2;
	}
	return triangle_count;
}

static qboolean QNN_ExtractNavmeshGeometry(
	const byte *raw,
	const dheader_t *header,
	float **out_vertices,
	int *out_vertex_count,
	int **out_triangles,
	int *out_triangle_count,
	char *error,
	size_t error_size)
{
	const dmodel_t *models;
	const dvertex_t *vertices;
	const dedge_t *edges;
	const int *surfedges;
	const dface_t *faces;
	const texinfo_t *texinfo;
	int model_count;
	int vertex_count;
	int edge_count;
	int surfedge_count;
	int face_count;
	int texinfo_count;
	const dmodel_t *world_model;
	int first_face;
	int num_faces;
	float *nav_vertices;
	int *nav_triangles;
	int triangle_count;
	int *face_vertices;
	int face_capacity;
	int write_index;
	int face_index;

	*out_vertices = NULL;
	*out_vertex_count = 0;
	*out_triangles = NULL;
	*out_triangle_count = 0;

	if (!QNN_BspLumpView(raw, header, LUMP_MODELS, sizeof(dmodel_t), (const void **)&models, &model_count, error, error_size)
		|| !QNN_BspLumpView(raw, header, LUMP_VERTEXES, sizeof(dvertex_t), (const void **)&vertices, &vertex_count, error, error_size)
		|| !QNN_BspLumpView(raw, header, LUMP_EDGES, sizeof(dedge_t), (const void **)&edges, &edge_count, error, error_size)
		|| !QNN_BspLumpView(raw, header, LUMP_SURFEDGES, sizeof(int), (const void **)&surfedges, &surfedge_count, error, error_size)
		|| !QNN_BspLumpView(raw, header, LUMP_FACES, sizeof(dface_t), (const void **)&faces, &face_count, error, error_size)
		|| !QNN_BspLumpView(raw, header, LUMP_TEXINFO, sizeof(texinfo_t), (const void **)&texinfo, &texinfo_count, error, error_size))
		return false;

	if (model_count <= 0 || vertex_count <= 0)
	{
		snprintf(error, error_size, "BSP is missing world model geometry");
		return false;
	}

	triangle_count = QNN_CountNavmeshTriangles(models, model_count, faces, face_count, texinfo, texinfo_count);
	if (triangle_count <= 0)
	{
		snprintf(error, error_size, "BSP did not yield any walkable world triangles");
		return false;
	}

	nav_vertices = (float *)malloc((size_t)vertex_count * 3u * sizeof(*nav_vertices));
	nav_triangles = (int *)malloc((size_t)triangle_count * 3u * sizeof(*nav_triangles));
	if (nav_vertices == NULL || nav_triangles == NULL)
	{
		snprintf(error, error_size, "Out of memory while extracting navmesh geometry");
		goto cleanup_fail;
	}

	for (face_index = 0; face_index < vertex_count; ++face_index)
	{
		nav_vertices[face_index * 3 + 0] = LittleFloat(vertices[face_index].point[0]);
		nav_vertices[face_index * 3 + 1] = LittleFloat(vertices[face_index].point[1]);
		nav_vertices[face_index * 3 + 2] = LittleFloat(vertices[face_index].point[2]);
	}

	world_model = &models[0];
	first_face = LittleLong(world_model->firstface);
	num_faces = LittleLong(world_model->numfaces);
	face_vertices = NULL;
	face_capacity = 0;
	write_index = 0;

	for (face_index = 0; face_index < num_faces; ++face_index)
	{
		const dface_t *face;
		int edge_count_local;
		int texinfo_index;
		int first_edge;
		int edge_index;

		face = &faces[first_face + face_index];
		edge_count_local = LittleShort(face->numedges);
		texinfo_index = LittleShort(face->texinfo);
		if (edge_count_local < 3 || texinfo_index < 0 || texinfo_index >= texinfo_count)
			continue;
		if (LittleLong(texinfo[texinfo_index].flags) & TEX_SPECIAL)
			continue;

		if (edge_count_local > face_capacity)
		{
			int *next_face_vertices;

			next_face_vertices = (int *)realloc(face_vertices, (size_t)edge_count_local * sizeof(*next_face_vertices));
			if (next_face_vertices == NULL)
			{
				snprintf(error, error_size, "Out of memory while triangulating BSP faces");
				goto cleanup_fail;
			}
			face_vertices = next_face_vertices;
			face_capacity = edge_count_local;
		}

		first_edge = LittleLong(face->firstedge);
		if (first_edge < 0 || first_edge + edge_count_local > surfedge_count)
		{
			snprintf(error, error_size, "BSP face references invalid surfedge range");
			goto cleanup_fail;
		}

		for (edge_index = 0; edge_index < edge_count_local; ++edge_index)
		{
			int surfedge_index;
			int edge_lookup;
			int vertex_index;

			surfedge_index = LittleLong(surfedges[first_edge + edge_index]);
			edge_lookup = surfedge_index >= 0 ? surfedge_index : -surfedge_index;
			if (edge_lookup < 0 || edge_lookup >= edge_count)
			{
				snprintf(error, error_size, "BSP face references invalid edge index");
				goto cleanup_fail;
			}
			vertex_index = surfedge_index >= 0 ? LittleShort(edges[edge_lookup].v[0]) : LittleShort(edges[edge_lookup].v[1]);
			if (vertex_index < 0 || vertex_index >= vertex_count)
			{
				snprintf(error, error_size, "BSP face references invalid vertex index");
				goto cleanup_fail;
			}
			face_vertices[edge_index] = vertex_index;
		}

		for (edge_index = 1; edge_index < edge_count_local - 1; ++edge_index)
		{
			nav_triangles[write_index++] = face_vertices[0];
			nav_triangles[write_index++] = face_vertices[edge_index];
			nav_triangles[write_index++] = face_vertices[edge_index + 1];
		}
	}

	free(face_vertices);
	*out_vertices = nav_vertices;
	*out_vertex_count = vertex_count;
	*out_triangles = nav_triangles;
	*out_triangle_count = triangle_count;
	return true;

cleanup_fail:
	free(face_vertices);
	free(nav_vertices);
	free(nav_triangles);
	return false;
}

static int QNN_RegionIdFromPoint(const vec3_t point)
{
	int gx;
	int gy;

	gx = (int)floor((point[0] / QNN_WORKER_GRID_SIZE) + 0.5f);
	gy = (int)floor((point[1] / QNN_WORKER_GRID_SIZE) + 0.5f);
	return (gx + 1024) * 2048 + (gy + 1024);
}

static void QNN_RegionCenter(int region_id, vec3_t out)
{
	int gx;
	int gy;

	gx = region_id / 2048 - 1024;
	gy = region_id % 2048 - 1024;
	out[0] = gx * QNN_WORKER_GRID_SIZE;
	out[1] = gy * QNN_WORKER_GRID_SIZE;
	out[2] = 0.0f;
}

static int QNN_CategoryOrder(const char *category)
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

static const char *QNN_Classify(const char *classname)
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

static void QNN_JsonStringLocal(FILE *out, const char *text)
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

static void QNN_FreeParsedEntities(qnn_parsed_entity_t *entities, int entity_count)
{
	int i;

	for (i = 0; i < entity_count; ++i)
	{
		free(entities[i].properties);
	}
	free(entities);
}

static qboolean QNN_AppendProperty(qnn_parsed_entity_t *entity, const char *key, const char *value)
{
	qnn_property_t *next;

	next = (qnn_property_t *)realloc(entity->properties, (entity->property_count + 1) * sizeof(*next));
	if (next == NULL)
		return false;
	entity->properties = next;
	memset(&entity->properties[entity->property_count], 0, sizeof(entity->properties[entity->property_count]));
	strncpy(entity->properties[entity->property_count].key, key, QNN_MAX_PROPERTY_KEY - 1);
	strncpy(entity->properties[entity->property_count].value, value, QNN_MAX_PROPERTY_VALUE - 1);
	entity->property_count += 1;
	return true;
}

static void QNN_SkipSpace(char **cursor)
{
	while (**cursor && isspace((unsigned char)**cursor))
	{
		*cursor += 1;
	}
}

static qboolean QNN_ParseQuoted(char **cursor, char *out, size_t out_size)
{
	size_t index;

	QNN_SkipSpace(cursor);
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

static qboolean QNN_ParseOrigin(const char *value, vec3_t out)
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

static void QNN_FormatVec3(const vec3_t value, char *out, size_t out_size)
{
	snprintf(out, out_size, "%.1f %.1f %.1f", value[0], value[1], value[2]);
}

static qboolean QNN_ModelBounds(
	const dmodel_t *models,
	int model_count,
	const char *model_name,
	vec3_t bounds_min,
	vec3_t bounds_max,
	vec3_t center)
{
	int model_index;

	if (models == NULL || model_name == NULL || model_name[0] != '*')
		return false;
	model_index = atoi(model_name + 1);
	if (model_index <= 0 || model_index >= model_count)
		return false;

	bounds_min[0] = LittleFloat(models[model_index].mins[0]);
	bounds_min[1] = LittleFloat(models[model_index].mins[1]);
	bounds_min[2] = LittleFloat(models[model_index].mins[2]);
	bounds_max[0] = LittleFloat(models[model_index].maxs[0]);
	bounds_max[1] = LittleFloat(models[model_index].maxs[1]);
	bounds_max[2] = LittleFloat(models[model_index].maxs[2]);
	center[0] = (bounds_min[0] + bounds_max[0]) * 0.5f;
	center[1] = (bounds_min[1] + bounds_max[1]) * 0.5f;
	center[2] = (bounds_min[2] + bounds_max[2]) * 0.5f;
	return true;
}

static qboolean QNN_AppendModelProperties(
	qnn_parsed_entity_t *entity,
	const dmodel_t *models,
	int model_count,
	const char *model_name)
{
	vec3_t bounds_min;
	vec3_t bounds_max;
	vec3_t center;
	char buffer[96];

	if (!QNN_ModelBounds(models, model_count, model_name, bounds_min, bounds_max, center))
		return true;
	QNN_FormatVec3(bounds_min, buffer, sizeof(buffer));
	if (!QNN_AppendProperty(entity, "QNN_ModelBounds_min", buffer))
		return false;
	QNN_FormatVec3(bounds_max, buffer, sizeof(buffer));
	if (!QNN_AppendProperty(entity, "QNN_ModelBounds_max", buffer))
		return false;
	return true;
}

static void QNN_ParseAngleValue(const char *value, vec3_t out)
{
	float yaw;

	yaw = 0.0f;
	out[0] = 0.0f;
	out[1] = 0.0f;
	out[2] = 0.0f;
	if (sscanf(value, "%f", &yaw) == 1)
		out[1] = yaw;
}

static void QNN_ParseAnglesValue(const char *value, vec3_t out)
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
	QNN_ParseAngleValue(value, out);
}

static int QNN_CompareEntities(const void *lhs_ptr, const void *rhs_ptr)
{
	const qnn_parsed_entity_t *lhs;
	const qnn_parsed_entity_t *rhs;
	int order_cmp;
	int classname_cmp;
	int axis;

	lhs = (const qnn_parsed_entity_t *)lhs_ptr;
	rhs = (const qnn_parsed_entity_t *)rhs_ptr;

	order_cmp = QNN_CategoryOrder(lhs->category) - QNN_CategoryOrder(rhs->category);
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

static int QNN_CompareRegions(const void *lhs_ptr, const void *rhs_ptr)
{
	const qnn_region_t *lhs;
	const qnn_region_t *rhs;

	lhs = (const qnn_region_t *)lhs_ptr;
	rhs = (const qnn_region_t *)rhs_ptr;
	return lhs->region_id - rhs->region_id;
}

static int QNN_CompareInts(const void *lhs_ptr, const void *rhs_ptr)
{
	const int *lhs;
	const int *rhs;

	lhs = (const int *)lhs_ptr;
	rhs = (const int *)rhs_ptr;
	return *lhs - *rhs;
}

static qnn_region_t *QNN_GetOrAddRegion(qnn_map_state_t *map_state, int region_id)
{
	qnn_region_t *region;
	qnn_region_t *next;
	int i;

	for (i = 0; i < map_state->region_count; ++i)
	{
		if (map_state->regions[i].region_id == region_id)
			return &map_state->regions[i];
	}

	next = (qnn_region_t *)realloc(map_state->regions, (map_state->region_count + 1) * sizeof(*next));
	if (next == NULL)
		return NULL;
	map_state->regions = next;
	region = &map_state->regions[map_state->region_count];
	memset(region, 0, sizeof(*region));
	region->region_id = region_id;
	QNN_RegionCenter(region_id, region->center);
	region->bounds_min[0] = region->center[0] - QNN_WORKER_GRID_HALF;
	region->bounds_min[1] = region->center[1] - QNN_WORKER_GRID_HALF;
	region->bounds_min[2] = -QNN_WORKER_Z_HALF;
	region->bounds_max[0] = region->center[0] + QNN_WORKER_GRID_HALF;
	region->bounds_max[1] = region->center[1] + QNN_WORKER_GRID_HALF;
	region->bounds_max[2] = QNN_WORKER_Z_HALF;
	map_state->region_count += 1;
	return region;
}

static void QNN_ExpandRegionBounds(qnn_region_t *region, const vec3_t origin)
{
	if (origin[0] - QNN_WORKER_OBJECT_PADDING < region->bounds_min[0])
		region->bounds_min[0] = origin[0] - QNN_WORKER_OBJECT_PADDING;
	if (origin[1] - QNN_WORKER_OBJECT_PADDING < region->bounds_min[1])
		region->bounds_min[1] = origin[1] - QNN_WORKER_OBJECT_PADDING;
	if (origin[2] - QNN_WORKER_OBJECT_PADDING < region->bounds_min[2])
		region->bounds_min[2] = origin[2] - QNN_WORKER_OBJECT_PADDING;
	if (origin[0] + QNN_WORKER_OBJECT_PADDING > region->bounds_max[0])
		region->bounds_max[0] = origin[0] + QNN_WORKER_OBJECT_PADDING;
	if (origin[1] + QNN_WORKER_OBJECT_PADDING > region->bounds_max[1])
		region->bounds_max[1] = origin[1] + QNN_WORKER_OBJECT_PADDING;
	if (origin[2] + QNN_WORKER_OBJECT_PADDING > region->bounds_max[2])
		region->bounds_max[2] = origin[2] + QNN_WORKER_OBJECT_PADDING;
}

static qboolean QNN_ParseEntities(
	char *text,
	const dmodel_t *models,
	int model_count,
	qnn_parsed_entity_t **out_entities,
	int *out_count,
	char *error,
	size_t error_size)
{
	qnn_parsed_entity_t *entities;
	int entity_count;
	char *cursor;
	int input_index;

	entities = NULL;
	entity_count = 0;
	cursor = text;
	input_index = 0;

	while (*cursor)
	{
		qnn_parsed_entity_t entity;
		char key[QNN_MAX_PROPERTY_KEY];
		char value[QNN_MAX_PROPERTY_VALUE];
		char model_name[QNN_MAX_MODEL_NAME];
		qboolean has_origin;
		qboolean has_angles;
		qboolean has_model;
		qnn_parsed_entity_t *next_entities;

		QNN_SkipSpace(&cursor);
		if (!*cursor)
			break;
		if (*cursor != '{')
		{
			cursor += 1;
			continue;
		}
		cursor += 1;

		memset(&entity, 0, sizeof(entity));
		memset(model_name, 0, sizeof(model_name));
		entity.input_index = input_index;
		has_origin = false;
		has_angles = false;
		has_model = false;

		while (*cursor)
		{
			QNN_SkipSpace(&cursor);
			if (*cursor == '}')
			{
				cursor += 1;
				break;
			}
			if (!QNN_ParseQuoted(&cursor, key, sizeof(key)))
			{
				snprintf(error, error_size, "Failed to parse entity key");
				QNN_FreeParsedEntities(entities, entity_count);
				free(entity.properties);
				return false;
			}
			if (!QNN_ParseQuoted(&cursor, value, sizeof(value)))
			{
				snprintf(error, error_size, "Failed to parse entity value");
				QNN_FreeParsedEntities(entities, entity_count);
				free(entity.properties);
				return false;
			}

			if (!strcmp(key, "classname"))
			{
				strncpy(entity.classname, value, sizeof(entity.classname) - 1);
			}
			else if (!strcmp(key, "model"))
			{
				strncpy(model_name, value, sizeof(model_name) - 1);
				has_model = true;
				if (!QNN_AppendProperty(&entity, key, value))
				{
					snprintf(error, error_size, "Out of memory while parsing entity properties");
					QNN_FreeParsedEntities(entities, entity_count);
					free(entity.properties);
					return false;
				}
			}
			else if (!strcmp(key, "origin"))
			{
				has_origin = QNN_ParseOrigin(value, entity.origin);
			}
			else if (!strcmp(key, "angles"))
			{
				QNN_ParseAnglesValue(value, entity.angles);
				has_angles = true;
			}
			else if (!strcmp(key, "angle"))
			{
				if (!has_angles)
					QNN_ParseAngleValue(value, entity.angles);
			}
			else if (!QNN_AppendProperty(&entity, key, value))
			{
				snprintf(error, error_size, "Out of memory while parsing entity properties");
				QNN_FreeParsedEntities(entities, entity_count);
				free(entity.properties);
				return false;
			}
		}

		input_index += 1;
		if (has_model)
		{
			vec3_t model_center;
			vec3_t model_bounds_min;
			vec3_t model_bounds_max;

			if (!QNN_AppendModelProperties(&entity, models, model_count, model_name))
			{
				snprintf(error, error_size, "Out of memory while annotating BSP model bounds");
				QNN_FreeParsedEntities(entities, entity_count);
				free(entity.properties);
				return false;
			}
			if (!has_origin && QNN_ModelBounds(models, model_count, model_name, model_bounds_min, model_bounds_max, model_center))
			{
				VectorCopy(model_center, entity.origin);
				has_origin = true;
			}
		}
		if (!has_origin)
		{
			free(entity.properties);
			continue;
		}

		strncpy(entity.category, QNN_Classify(entity.classname), sizeof(entity.category) - 1);
		entity.region_id = QNN_RegionIdFromPoint(entity.origin);
		next_entities = (qnn_parsed_entity_t *)realloc(entities, (entity_count + 1) * sizeof(*next_entities));
		if (next_entities == NULL)
		{
			snprintf(error, error_size, "Out of memory while parsing entity list");
			QNN_FreeParsedEntities(entities, entity_count);
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

void QNN_ClearAction(qnn_action_t *action)
{
	action->move[0] = 0.0f;
	action->move[1] = 0.0f;
	action->look[0] = 0.0f;
	action->look[1] = 0.0f;
	action->fire = 0;
	action->jump = 0;
	action->switch_slot = 0;
	action->recall[0] = 0;
	action->recall[1] = 0;
	action->recall[2] = 0;
	action->recall[3] = 0;
}

void QNN_FreeMapState(qnn_map_state_t *map_state)
{
	int i;

	if (map_state == NULL)
		return;

	for (i = 0; i < map_state->static_object_count; ++i)
	{
		free(map_state->static_objects[i].properties);
	}

	qnn_nav_oracle_destroy(map_state->nav_oracle);
	qnn_navmesh_destroy(map_state->navmesh);
	free(map_state->regions);
	free(map_state->static_objects);
	memset(map_state, 0, sizeof(*map_state));
}

qboolean QNN_BuildMapState(qnn_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size)
{
	char path[MAX_QPATH];
	byte *raw;
	dheader_t *header;
	int entity_ofs;
	int entity_len;
	char *entity_text;
	float *nav_vertices;
	int nav_vertex_count;
	int *nav_triangles;
	int nav_triangle_count;
	char nav_error[256];
	qnn_parsed_entity_t *entities;
	int entity_count;
	int stable_index;

	memset(out, 0, sizeof(*out));
	strncpy(out->requested_map_id, requested_map_id, sizeof(out->requested_map_id) - 1);
	strncpy(out->map_name, map_name, sizeof(out->map_name) - 1);
	strncpy(out->source, "bsp_entities", sizeof(out->source) - 1);
	strncpy(out->navmesh_status, "missing", sizeof(out->navmesh_status) - 1);
	strncpy(out->nav_oracle_status, "missing", sizeof(out->nav_oracle_status) - 1);
	QNN_DefaultNavmeshConfig(&out->navmesh_config);

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
	{
		const dmodel_t *models;
		int model_count;

		if (!QNN_BspLumpView(raw, header, LUMP_MODELS, sizeof(dmodel_t), (const void **)&models, &model_count, error, error_size))
			return false;
		entities = NULL;
		entity_count = 0;
		nav_vertices = NULL;
		nav_vertex_count = 0;
		nav_triangles = NULL;
		nav_triangle_count = 0;
		memset(nav_error, 0, sizeof(nav_error));

		if (!QNN_ParseEntities(entity_text, models, model_count, &entities, &entity_count, error, error_size))
			return false;
	}

	if (entity_count > 1)
		qsort(entities, entity_count, sizeof(*entities), QNN_CompareEntities);

	for (stable_index = 0; stable_index < entity_count; ++stable_index)
	{
		qnn_static_object_t *object;
		qnn_static_object_t *next_objects;
		qnn_region_t *region;

		next_objects = (qnn_static_object_t *)realloc(out->static_objects, (out->static_object_count + 1) * sizeof(*next_objects));
		if (next_objects == NULL)
		{
			snprintf(error, error_size, "Out of memory while building static object list");
			QNN_FreeParsedEntities(entities, entity_count);
			QNN_FreeMapState(out);
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

		region = QNN_GetOrAddRegion(out, object->region_id);
		if (region == NULL)
		{
			snprintf(error, error_size, "Out of memory while building region list");
			QNN_FreeParsedEntities(entities, entity_count);
			QNN_FreeMapState(out);
			return false;
		}
		QNN_ExpandRegionBounds(region, object->origin);
	}

	QNN_FreeParsedEntities(entities, entity_count);

	if (out->region_count == 0)
	{
		if (QNN_GetOrAddRegion(out, 0) == NULL)
		{
			snprintf(error, error_size, "Out of memory while creating fallback region");
			QNN_FreeMapState(out);
			return false;
		}
	}

	if (out->region_count > 1)
		qsort(out->regions, out->region_count, sizeof(*out->regions), QNN_CompareRegions);

	if (QNN_ExtractNavmeshGeometry(raw, header, &nav_vertices, &nav_vertex_count, &nav_triangles, &nav_triangle_count, nav_error, sizeof(nav_error)))
	{
		out->navmesh = qnn_navmesh_build(
			nav_vertices,
			nav_vertex_count,
			nav_triangles,
			nav_triangle_count,
			&out->navmesh_config,
			&out->navmesh_summary,
			nav_error,
			sizeof(nav_error));
		if (out->navmesh != NULL)
			strncpy(out->navmesh_status, "ready", sizeof(out->navmesh_status) - 1);
		else
			strncpy(out->navmesh_status, "error", sizeof(out->navmesh_status) - 1);
	}
	else
	{
		strncpy(out->navmesh_status, "error", sizeof(out->navmesh_status) - 1);
	}
	if (nav_error[0] != 0)
		strncpy(out->navmesh_error, nav_error, sizeof(out->navmesh_error) - 1);
	memset(nav_error, 0, sizeof(nav_error));
	if (out->navmesh != NULL)
	{
		qnn_nav_oracle_static_object_view_t *oracle_objects;
		qnn_nav_oracle_property_view_t *oracle_properties;
		qnn_nav_oracle_summary_t oracle_summary;
		int total_properties;
		int property_offset;
		int object_index;

		memset(&oracle_summary, 0, sizeof(oracle_summary));
		oracle_objects = NULL;
		oracle_properties = NULL;
		total_properties = 0;
		property_offset = 0;
		for (object_index = 0; object_index < out->static_object_count; ++object_index)
			total_properties += out->static_objects[object_index].property_count;

		if (out->static_object_count > 0)
		{
			oracle_objects = (qnn_nav_oracle_static_object_view_t *)calloc((size_t)out->static_object_count, sizeof(*oracle_objects));
			if (oracle_objects == NULL)
			{
				snprintf(error, error_size, "Out of memory while preparing nav oracle objects");
				QNN_FreeMapState(out);
				free(nav_vertices);
				free(nav_triangles);
				return false;
			}
		}
		if (total_properties > 0)
		{
			oracle_properties = (qnn_nav_oracle_property_view_t *)calloc((size_t)total_properties, sizeof(*oracle_properties));
			if (oracle_properties == NULL)
			{
				free(oracle_objects);
				snprintf(error, error_size, "Out of memory while preparing nav oracle properties");
				QNN_FreeMapState(out);
				free(nav_vertices);
				free(nav_triangles);
				return false;
			}
		}

		for (object_index = 0; object_index < out->static_object_count; ++object_index)
		{
			int property_index;

			oracle_objects[object_index].object_id = out->static_objects[object_index].object_id;
			oracle_objects[object_index].category = out->static_objects[object_index].category;
			oracle_objects[object_index].classname = out->static_objects[object_index].classname;
			oracle_objects[object_index].origin = out->static_objects[object_index].origin;
			oracle_objects[object_index].angles = out->static_objects[object_index].angles;
			oracle_objects[object_index].property_count = out->static_objects[object_index].property_count;
			oracle_objects[object_index].properties = oracle_properties + property_offset;
			for (property_index = 0; property_index < out->static_objects[object_index].property_count; ++property_index)
			{
				oracle_properties[property_offset + property_index].key = out->static_objects[object_index].properties[property_index].key;
				oracle_properties[property_offset + property_index].value = out->static_objects[object_index].properties[property_index].value;
			}
			property_offset += out->static_objects[object_index].property_count;
		}

		out->nav_oracle = qnn_nav_oracle_build(
			out->navmesh,
			oracle_objects,
			out->static_object_count,
			&oracle_summary,
			nav_error,
			sizeof(nav_error));
		free(oracle_objects);
		free(oracle_properties);
		if (out->nav_oracle != NULL)
		{
			strncpy(out->nav_oracle_status, "ready", sizeof(out->nav_oracle_status) - 1);
			out->nav_area_count = oracle_summary.area_count;
			out->nav_cluster_count = oracle_summary.cluster_count;
			out->nav_min_cluster_area_count = oracle_summary.min_cluster_area_count;
			out->nav_max_cluster_area_count = oracle_summary.max_cluster_area_count;
			out->nav_avg_cluster_area_count = oracle_summary.avg_cluster_area_count;
			out->nav_walk_link_count = oracle_summary.walk_link_count;
			out->nav_teleport_link_count = oracle_summary.teleport_link_count;
			out->nav_lift_link_count = oracle_summary.lift_link_count;
			out->nav_push_link_count = oracle_summary.push_link_count;
			out->nav_drop_link_count = oracle_summary.drop_link_count;
		}
		else
		{
			strncpy(out->nav_oracle_status, "error", sizeof(out->nav_oracle_status) - 1);
		}
	}
	if (nav_error[0] != 0)
		strncpy(out->nav_oracle_error, nav_error, sizeof(out->nav_oracle_error) - 1);
	free(nav_vertices);
	free(nav_triangles);
	return true;
}

void QNN_WriteMapStateJson(FILE *out, const qnn_map_state_t *map_state)
{
	int i;
	qnn_nav_oracle_summary_t oracle_summary;

	memset(&oracle_summary, 0, sizeof(oracle_summary));
	oracle_summary.area_count = map_state->nav_area_count;
	oracle_summary.cluster_count = map_state->nav_cluster_count;
	oracle_summary.min_cluster_area_count = map_state->nav_min_cluster_area_count;
	oracle_summary.max_cluster_area_count = map_state->nav_max_cluster_area_count;
	oracle_summary.avg_cluster_area_count = map_state->nav_avg_cluster_area_count;
	oracle_summary.walk_link_count = map_state->nav_walk_link_count;
	oracle_summary.teleport_link_count = map_state->nav_teleport_link_count;
	oracle_summary.lift_link_count = map_state->nav_lift_link_count;
	oracle_summary.push_link_count = map_state->nav_push_link_count;
	oracle_summary.drop_link_count = map_state->nav_drop_link_count;
	oracle_summary.total_link_count = map_state->nav_walk_link_count
		+ map_state->nav_teleport_link_count
		+ map_state->nav_lift_link_count
		+ map_state->nav_push_link_count
		+ map_state->nav_drop_link_count;

	fprintf(out, "{\"map_id\":");
	QNN_JsonStringLocal(out, map_state->requested_map_id);
	fprintf(out, ",\"metadata\":{\"grid_size\":%.1f,\"region_count\":%d,\"source\":",
		QNN_WORKER_GRID_SIZE,
		map_state->region_count);
	QNN_JsonStringLocal(out, map_state->source);
	fprintf(out, ",\"static_object_count\":%d,\"navmesh\":{\"backend\":\"recast_detour\",\"cell_height\":%.1f,\"cell_size\":%.1f,",
		map_state->static_object_count,
		map_state->navmesh_config.cell_height,
		map_state->navmesh_config.cell_size);
	fprintf(out, "\"status\":");
	QNN_JsonStringLocal(out, map_state->navmesh_status);
	fprintf(out, ",\"walkable_climb\":%.1f,\"walkable_height\":%.1f,\"walkable_radius\":%.1f,\"walkable_slope_angle\":%.1f",
		map_state->navmesh_config.walkable_climb,
		map_state->navmesh_config.walkable_height,
		map_state->navmesh_config.walkable_radius,
		map_state->navmesh_config.walkable_slope_angle);
	if (map_state->navmesh_error[0] != 0)
	{
		fprintf(out, ",\"error\":");
		QNN_JsonStringLocal(out, map_state->navmesh_error);
	}
	fprintf(out, ",\"summary\":");
	qnn_navmesh_write_summary_json(out, &map_state->navmesh_summary);
	fprintf(out, ",\"oracle\":{\"status\":");
	QNN_JsonStringLocal(out, map_state->nav_oracle_status);
	if (map_state->nav_oracle_error[0] != 0)
	{
		fprintf(out, ",\"error\":");
		QNN_JsonStringLocal(out, map_state->nav_oracle_error);
	}
	fprintf(out, ",\"summary\":");
	qnn_nav_oracle_write_summary_json(out, &oracle_summary);
	fprintf(out, "}");
	fprintf(out, "}},\"regions\":[");

	for (i = 0; i < map_state->region_count; ++i)
	{
		int j;
		const qnn_region_t *region;

		region = &map_state->regions[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out,
			"{\"bounds_max\":[%.1f,%.1f,%.1f],\"bounds_min\":[%.1f,%.1f,%.1f],\"center\":[%.1f,%.1f,%.1f],\"region_id\":%d}",
			region->bounds_max[0], region->bounds_max[1], region->bounds_max[2],
			region->bounds_min[0], region->bounds_min[1], region->bounds_min[2],
			region->center[0], region->center[1], region->center[2],
			region->region_id);
	}
	fprintf(out, "],\"static_objects\":[");

	for (i = 0; i < map_state->static_object_count; ++i)
	{
		int j;
		const qnn_static_object_t *object;

		object = &map_state->static_objects[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"angles\":[%.1f,%.1f,%.1f],\"category\":",
			object->angles[0], object->angles[1], object->angles[2]);
		QNN_JsonStringLocal(out, object->category);
		fprintf(out, ",\"classname\":");
		QNN_JsonStringLocal(out, object->classname);
		fprintf(out, ",\"object_id\":");
		QNN_JsonStringLocal(out, object->object_id);
		fprintf(out, ",\"origin\":[%.1f,%.1f,%.1f],\"properties\":{",
			object->origin[0], object->origin[1], object->origin[2]);
		for (j = 0; j < object->property_count; ++j)
		{
			if (j > 0)
				fputc(',', out);
			QNN_JsonStringLocal(out, object->properties[j].key);
			fputc(':', out);
			QNN_JsonStringLocal(out, object->properties[j].value);
		}
		fprintf(out, "},\"region_id\":%d}", object->region_id);
	}

	fprintf(out, "]}");
}

int QNN_NearestRegionId(const qnn_map_state_t *map_state, const vec3_t point)
{
	int candidate;
	int i;
	int best_region_id;
	float best_distance;

	candidate = QNN_RegionIdFromPoint(point);
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
