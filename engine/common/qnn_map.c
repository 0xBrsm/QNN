/*
 * qnn_map.c — BSP entity extraction from cl.worldmodel.
 *
 * Parses the entity lump text into raw entity records with classname,
 * origin, angles, and all key-value properties.  No classification —
 * that lives in qnn_entity.c.
 *
 * Uses a custom quoted-string parser (not COM_Parse) because the
 * entity lump contains quoted key-value pairs and COM_Parse strips
 * quotes.  This preserves whitespace inside values (e.g. origins).
 */

#include "qnn_map.h"
#include "qnn_navmesh.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Text parsing helpers ─────────────────────────────────────────── */

static void QNN_SkipSpaceEntity(char **cursor)
{
	while (**cursor && (**cursor == ' ' || **cursor == '\t'
		|| **cursor == '\r' || **cursor == '\n'))
		*cursor += 1;
}

static qboolean QNN_ParseQuotedEntity(char **cursor, char *out, size_t out_size)
{
	size_t index;

	QNN_SkipSpaceEntity(cursor);
	if (**cursor != '"')
		return false;
	*cursor += 1;
	index = 0;
	while (**cursor && **cursor != '"')
	{
		char ch = **cursor;
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
	out[index] = '\0';
	*cursor += 1;
	return true;
}

static qboolean QNN_ParseOriginStr(const char *str, vec3_t out)
{
	return sscanf(str, "%f %f %f", &out[0], &out[1], &out[2]) == 3;
}

static void QNN_ParseAnglesStr(const char *str, vec3_t out)
{
	if (sscanf(str, "%f %f %f", &out[0], &out[1], &out[2]) != 3)
	{
		out[0] = 0.0f;
		out[1] = 0.0f;
		out[2] = 0.0f;
	}
}

static void QNN_ParseAngleStr(const char *str, vec3_t out)
{
	float yaw = 0.0f;
	sscanf(str, "%f", &yaw);
	out[0] = 0.0f;
	out[1] = yaw;
	out[2] = 0.0f;
}

static void QNN_FormatVec3Entity(const vec3_t v, char *out, size_t out_size)
{
	snprintf(out, out_size, "%.1f %.1f %.1f", v[0], v[1], v[2]);
}

/*
 * Read BSP submodel bounds and center.  Values are already byte-swapped
 * by the engine loader.
 */
static qboolean QNN_ModelBoundsEntity(
	const dmodel_t *models, int model_count,
	const char *model_name,
	vec3_t bounds_min, vec3_t bounds_max, vec3_t center)
{
	int model_index;

	if (models == NULL || model_name == NULL || model_name[0] != '*')
		return false;
	model_index = atoi(model_name + 1);
	if (model_index <= 0 || model_index >= model_count)
		return false;

	VectorCopy(models[model_index].mins, bounds_min);
	VectorCopy(models[model_index].maxs, bounds_max);
	center[0] = (bounds_min[0] + bounds_max[0]) * 0.5f;
	center[1] = (bounds_min[1] + bounds_max[1]) * 0.5f;
	center[2] = (bounds_min[2] + bounds_max[2]) * 0.5f;
	return true;
}

/* ── Main parser ──────────────────────────────────────────────────── */

int QNN_MapParseEntities(qnn_raw_entity_t *out, int max)
{
	char *cursor;
	int input_index;
	int count;
	dmodel_t *models;
	int model_count;

	if (cl.worldmodel == NULL || cl.worldmodel->entities == NULL)
		return 0;

	cursor = cl.worldmodel->entities;
	models = cl.worldmodel->submodels;
	model_count = cl.worldmodel->numsubmodels;
	input_index = 0;
	count = 0;

	while (*cursor && count < max)
	{
		char classname[QNN_MAX_CLASSNAME];
		char key[QNN_MAX_PROPERTY_KEY];
		char value[QNN_MAX_PROPERTY_VALUE];
		char model_name[QNN_MAX_MODEL_NAME];
		vec3_t origin, angles;
		int spawnflags;
		qboolean has_origin, has_angles, has_model;
		qnn_property_t props[QNN_MAX_STATIC_PROPERTIES];
		int prop_count;

		QNN_SkipSpaceEntity(&cursor);
		if (!*cursor)
			break;
		if (*cursor != '{')
		{
			cursor += 1;
			continue;
		}
		cursor += 1;

		classname[0] = '\0';
		model_name[0] = '\0';
		VectorCopy(vec3_origin, origin);
		VectorCopy(vec3_origin, angles);
		spawnflags = 0;
		has_origin = false;
		has_angles = false;
		has_model = false;
		prop_count = 0;

		while (*cursor)
		{
			QNN_SkipSpaceEntity(&cursor);
			if (*cursor == '}')
			{
				cursor += 1;
				break;
			}
			if (!QNN_ParseQuotedEntity(&cursor, key, sizeof(key)))
				break;
			if (!QNN_ParseQuotedEntity(&cursor, value, sizeof(value)))
				break;

			if (!strcmp(key, "classname"))
			{
				strncpy(classname, value, sizeof(classname) - 1);
				classname[sizeof(classname) - 1] = '\0';
			}
			else if (!strcmp(key, "model"))
			{
				strncpy(model_name, value, sizeof(model_name) - 1);
				model_name[sizeof(model_name) - 1] = '\0';
				has_model = true;
				if (prop_count < QNN_MAX_STATIC_PROPERTIES)
				{
					strncpy(props[prop_count].key, key, QNN_MAX_PROPERTY_KEY - 1);
					props[prop_count].key[QNN_MAX_PROPERTY_KEY - 1] = '\0';
					strncpy(props[prop_count].value, value, QNN_MAX_PROPERTY_VALUE - 1);
					props[prop_count].value[QNN_MAX_PROPERTY_VALUE - 1] = '\0';
					prop_count++;
				}
			}
			else if (!strcmp(key, "origin"))
			{
				has_origin = QNN_ParseOriginStr(value, origin);
			}
			else if (!strcmp(key, "angles"))
			{
				QNN_ParseAnglesStr(value, angles);
				has_angles = true;
			}
			else if (!strcmp(key, "angle"))
			{
				if (!has_angles)
					QNN_ParseAngleStr(value, angles);
			}
			else if (!strcmp(key, "spawnflags"))
			{
				spawnflags = atoi(value);
				/* Also store spawnflags as a property for nav */
				if (prop_count < QNN_MAX_STATIC_PROPERTIES)
				{
					strncpy(props[prop_count].key, key, QNN_MAX_PROPERTY_KEY - 1);
					props[prop_count].key[QNN_MAX_PROPERTY_KEY - 1] = '\0';
					strncpy(props[prop_count].value, value, QNN_MAX_PROPERTY_VALUE - 1);
					props[prop_count].value[QNN_MAX_PROPERTY_VALUE - 1] = '\0';
					prop_count++;
				}
			}
			else
			{
				/* Store all other key-value pairs as properties */
				if (prop_count < QNN_MAX_STATIC_PROPERTIES)
				{
					strncpy(props[prop_count].key, key, QNN_MAX_PROPERTY_KEY - 1);
					props[prop_count].key[QNN_MAX_PROPERTY_KEY - 1] = '\0';
					strncpy(props[prop_count].value, value, QNN_MAX_PROPERTY_VALUE - 1);
					props[prop_count].value[QNN_MAX_PROPERTY_VALUE - 1] = '\0';
					prop_count++;
				}
			}
		}

		input_index += 1;

		/* Annotate BSP model bounds as properties */
		if (has_model)
		{
			vec3_t bounds_min, bounds_max, center;
			char buffer[96];

			if (QNN_ModelBoundsEntity(models, model_count, model_name,
				bounds_min, bounds_max, center))
			{
				if (prop_count < QNN_MAX_STATIC_PROPERTIES)
				{
					QNN_FormatVec3Entity(bounds_min, buffer, sizeof(buffer));
					strncpy(props[prop_count].key, "qnn_model_bounds_min", QNN_MAX_PROPERTY_KEY - 1);
					strncpy(props[prop_count].value, buffer, QNN_MAX_PROPERTY_VALUE - 1);
					prop_count++;
				}
				if (prop_count < QNN_MAX_STATIC_PROPERTIES)
				{
					QNN_FormatVec3Entity(bounds_max, buffer, sizeof(buffer));
					strncpy(props[prop_count].key, "qnn_model_bounds_max", QNN_MAX_PROPERTY_KEY - 1);
					strncpy(props[prop_count].value, buffer, QNN_MAX_PROPERTY_VALUE - 1);
					prop_count++;
				}
				if (!has_origin)
				{
					VectorCopy(center, origin);
					has_origin = true;
				}
			}
		}

		if (!has_origin && !has_model)
			continue;

		/* Populate output */
		memset(&out[count], 0, sizeof(out[count]));
		out[count].entity_num = input_index;
		strncpy(out[count].classname, classname, sizeof(out[count].classname) - 1);
		strncpy(out[count].model_name, model_name, sizeof(out[count].model_name) - 1);
		VectorCopy(origin, out[count].origin);
		VectorCopy(angles, out[count].angles);
		out[count].spawnflags = spawnflags;
		out[count].has_origin = has_origin;
		out[count].has_model = has_model;
		out[count].property_count = prop_count;
		if (prop_count > 0)
			memcpy(out[count].properties, props, (size_t)prop_count * sizeof(qnn_property_t));
		count++;
	}
	return count;
}

/* ══════════════════════════════════════════════════════════════════
 * Baseline-driven entity list
 *
 * Builds the unified entity list from cl_entities[] baselines (server
 * edict numbers) joined with BSP entity lump data.  Brush models (*N)
 * are matched by model key to get classname and properties.  Point
 * models are classified by QNN_ClassifyByModel.
 * ══════════════════════════════════════════════════════════════════ */

int QNN_MapBuildFromBaselines(qnn_raw_entity_t *out, int max)
{
	qnn_raw_entity_t *bsp_entities;
	int bsp_count;
	int edict_num;
	int count;

	if (cl.worldmodel == NULL || cl.num_entities <= 0 || out == NULL || max <= 0)
		return 0;

	/* Parse BSP entity lump for brush model metadata lookup */
	bsp_entities = (qnn_raw_entity_t *)calloc(QNN_MAX_RAW_ENTITIES, sizeof(*bsp_entities));
	if (bsp_entities == NULL)
		return 0;
	bsp_count = QNN_MapParseEntities(bsp_entities, QNN_MAX_RAW_ENTITIES);

	count = 0;
	for (edict_num = 1; edict_num < cl.num_entities && count < max; ++edict_num)
	{
		entity_t *ent = &cl_entities[edict_num];
		int modelindex = ent->baseline.modelindex;
		model_t *model;
		const char *model_name;

		if (modelindex <= 0)
			continue;
		if (modelindex >= MAX_MODELS)
			continue;
		model = cl.model_precache[modelindex];
		if (model == NULL)
			continue;
		model_name = model->name;
		if (model_name == NULL || model_name[0] == '\0')
			continue;

		memset(&out[count], 0, sizeof(out[count]));
		out[count].entity_num = edict_num;
		VectorCopy(ent->baseline.origin, out[count].origin);
		VectorCopy(ent->baseline.angles, out[count].angles);
		strncpy(out[count].model_name, model_name, sizeof(out[count].model_name) - 1);

		if (model_name[0] == '*')
		{
			/* Brush model: join with BSP entity lump by model key
			   to get classname, spawnflags, and properties. */
			int bi;
			qboolean found = false;

			for (bi = 0; bi < bsp_count; ++bi)
			{
				if (!bsp_entities[bi].has_model)
					continue;
				if (strcmp(bsp_entities[bi].model_name, model_name) != 0)
					continue;

				/* Copy classname, spawnflags, properties from BSP */
				strncpy(out[count].classname, bsp_entities[bi].classname,
					sizeof(out[count].classname) - 1);
				out[count].spawnflags = bsp_entities[bi].spawnflags;
				out[count].has_model = true;
				out[count].property_count = bsp_entities[bi].property_count;
				if (bsp_entities[bi].property_count > 0)
					memcpy(out[count].properties, bsp_entities[bi].properties,
						(size_t)bsp_entities[bi].property_count * sizeof(qnn_property_t));

				/* Brush models often have origin 0,0,0 — use BSP bounds center */
				if (fabsf(out[count].origin[0]) < 0.1f
					&& fabsf(out[count].origin[1]) < 0.1f
					&& fabsf(out[count].origin[2]) < 0.1f
					&& bsp_entities[bi].has_origin)
				{
					VectorCopy(bsp_entities[bi].origin, out[count].origin);
				}
				out[count].has_origin = true;
				found = true;
				break;
			}
			if (!found)
				continue; /* brush model not in BSP lump — skip */
		}
		else
		{
			/* Point model: classify by model name + skin */
			int subject_id, qualifier_id;
			float magnitude;
			int bi;

			if (!QNN_ClassifyByModel(model_name, ent->baseline.skin,
				&subject_id, &qualifier_id, &magnitude))
				continue; /* unknown model (flames, gibs, etc.) — skip */

			/* Find the BSP entity at this origin to get the real classname
			   (e.g., "item_armor1" instead of "progs/armor.mdl"). */
			for (bi = 0; bi < bsp_count; ++bi)
			{
				float dx = bsp_entities[bi].origin[0] - out[count].origin[0];
				float dy = bsp_entities[bi].origin[1] - out[count].origin[1];
				float dz = bsp_entities[bi].origin[2] - out[count].origin[2];
				if (dx * dx + dy * dy + dz * dz < 1.0f)
				{
					strncpy(out[count].classname, bsp_entities[bi].classname,
						sizeof(out[count].classname) - 1);
					out[count].spawnflags = bsp_entities[bi].spawnflags;
					out[count].property_count = bsp_entities[bi].property_count;
					if (bsp_entities[bi].property_count > 0)
						memcpy(out[count].properties, bsp_entities[bi].properties,
							(size_t)bsp_entities[bi].property_count * sizeof(qnn_property_t));
					break;
				}
			}
			if (out[count].classname[0] == '\0')
				strncpy(out[count].classname, model_name, sizeof(out[count].classname) - 1);
			out[count].has_origin = true;
			out[count].has_model = true;
		}

		count++;
	}

	free(bsp_entities);
	return count;
}

/* ══════════════════════════════════════════════════════════════════
 * Geometry extraction + navmesh build
 * ══════════════════════════════════════════════════════════════════ */

/* cell_size = agentRadius / 3..4 per Recast author for indoor maps.
   Quake agent radius is 16, so cs=4 gives 4-cell erosion margin.
   cell_height=2 matches the finer horizontal resolution. */
#define QNN_NAV_CELL_SIZE 4.0f
#define QNN_NAV_CELL_HEIGHT 2.0f
#define QNN_NAV_WALKABLE_SLOPE_ANGLE 45.0f
#define QNN_NAV_WALKABLE_HEIGHT 56.0f
#define QNN_NAV_WALKABLE_CLIMB 18.0f
#define QNN_NAV_WALKABLE_RADIUS 16.0f
#define QNN_NAV_MAX_EDGE_LEN 192.0f
#define QNN_NAV_MAX_SIMPLIFICATION_ERROR 1.3f
/* At cs=4, small regions appear more often.  Keep min small to avoid
   dropping narrow walkable strips (staircases, ledges). */
#define QNN_NAV_MIN_REGION_SIZE 2
#define QNN_NAV_MERGE_REGION_SIZE 20
#define QNN_NAV_MAX_VERTS_PER_POLY 6
#define QNN_NAV_DETAIL_SAMPLE_DISTANCE 6.0f
#define QNN_NAV_DETAIL_SAMPLE_MAX_ERROR 1.0f

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

static qboolean QNN_ExtractGeometry(
	float **out_vertices, int *out_vertex_count,
	int **out_triangles, int *out_triangle_count,
	char *error, size_t error_size)
{
	model_t *worldmodel;
	mvertex_t *vertices;
	medge_t *edges;
	int *surfedges;
	msurface_t *surfaces;
	dmodel_t *submodels;
	int vertex_count, surface_count, submodel_count;
	int first_face, num_faces, triangle_count;
	int face_index, vertex_index, write_index;
	float *verts_out;
	int *tris_out;
	int *face_verts;
	int face_verts_cap;

	*out_vertices = NULL;
	*out_vertex_count = 0;
	*out_triangles = NULL;
	*out_triangle_count = 0;

	worldmodel = cl.worldmodel;
	if (worldmodel == NULL)
	{
		snprintf(error, error_size, "cl.worldmodel is NULL");
		return false;
	}

	vertices = worldmodel->vertexes;
	vertex_count = worldmodel->numvertexes;
	edges = worldmodel->edges;
	surfedges = worldmodel->surfedges;
	surfaces = worldmodel->surfaces;
	surface_count = worldmodel->numsurfaces;
	submodels = worldmodel->submodels;
	submodel_count = worldmodel->numsubmodels;

	if (submodel_count <= 0 || vertex_count <= 0)
	{
		snprintf(error, error_size, "Worldmodel is missing geometry");
		return false;
	}

	first_face = submodels[0].firstface;
	num_faces = submodels[0].numfaces;
	if (first_face < 0 || num_faces < 0 || first_face + num_faces > surface_count)
	{
		snprintf(error, error_size, "Worldmodel submodel 0 has invalid face range");
		return false;
	}

	/* Count triangles */
	triangle_count = 0;
	for (face_index = 0; face_index < num_faces; ++face_index)
	{
		msurface_t *face = &surfaces[first_face + face_index];
		if (face->numedges < 3 || face->texinfo == NULL)
			continue;
		if (face->texinfo->flags & TEX_SPECIAL)
			continue;
		triangle_count += face->numedges - 2;
	}

	if (triangle_count <= 0)
	{
		snprintf(error, error_size, "No walkable triangles in worldmodel");
		return false;
	}

	/* Allocate output */
	verts_out = (float *)malloc((size_t)vertex_count * 3 * sizeof(float));
	tris_out = (int *)malloc((size_t)triangle_count * 3 * sizeof(int));
	if (verts_out == NULL || tris_out == NULL)
	{
		free(verts_out);
		free(tris_out);
		snprintf(error, error_size, "Out of memory for geometry extraction");
		return false;
	}

	/* Copy vertices */
	for (vertex_index = 0; vertex_index < vertex_count; ++vertex_index)
	{
		verts_out[vertex_index * 3 + 0] = vertices[vertex_index].position[0];
		verts_out[vertex_index * 3 + 1] = vertices[vertex_index].position[1];
		verts_out[vertex_index * 3 + 2] = vertices[vertex_index].position[2];
	}

	/* Fan-triangulate faces */
	face_verts = NULL;
	face_verts_cap = 0;
	write_index = 0;

	for (face_index = 0; face_index < num_faces; ++face_index)
	{
		msurface_t *face = &surfaces[first_face + face_index];
		int edge_count_local = face->numedges;
		int first_edge, edge_index;

		if (edge_count_local < 3 || face->texinfo == NULL)
			continue;
		if (face->texinfo->flags & TEX_SPECIAL)
			continue;

		if (edge_count_local > face_verts_cap)
		{
			int *tmp = (int *)realloc(face_verts, (size_t)edge_count_local * sizeof(int));
			if (tmp == NULL) { free(face_verts); free(verts_out); free(tris_out); snprintf(error, error_size, "OOM"); return false; }
			face_verts = tmp;
			face_verts_cap = edge_count_local;
		}

		first_edge = face->firstedge;
		for (edge_index = 0; edge_index < edge_count_local; ++edge_index)
		{
			int se = surfedges[first_edge + edge_index];
			int ei = se >= 0 ? se : -se;
			face_verts[edge_index] = se >= 0 ? edges[ei].v[0] : edges[ei].v[1];
		}

		for (edge_index = 1; edge_index < edge_count_local - 1; ++edge_index)
		{
			tris_out[write_index++] = face_verts[0];
			tris_out[write_index++] = face_verts[edge_index];
			tris_out[write_index++] = face_verts[edge_index + 1];
		}
	}
	free(face_verts);

	*out_vertices = verts_out;
	*out_vertex_count = vertex_count;
	*out_triangles = tris_out;
	*out_triangle_count = triangle_count;
	return true;
}

qnn_navmesh_runtime_t *QNN_MapBuildNavmesh(
	qnn_navmesh_build_config_t *out_config,
	qnn_navmesh_summary_t *out_summary,
	char *error, size_t error_size)
{
	float *nav_vertices = NULL;
	int nav_vertex_count = 0;
	int *nav_triangles = NULL;
	int nav_triangle_count = 0;
	qnn_navmesh_build_config_t config;
	qnn_navmesh_runtime_t *result;

	QNN_DefaultNavmeshConfig(&config);
	if (out_config != NULL)
		*out_config = config;

	if (!QNN_ExtractGeometry(&nav_vertices, &nav_vertex_count,
		&nav_triangles, &nav_triangle_count, error, error_size))
		return NULL;

	result = qnn_navmesh_build(
		nav_vertices, nav_vertex_count,
		nav_triangles, nav_triangle_count,
		&config, out_summary, error, error_size);

	free(nav_vertices);
	free(nav_triangles);
	return result;
}

/* ══════════════════════════════════════════════════════════════════
 * Static property helpers (used by qnn_entity.c)
 * ══════════════════════════════════════════════════════════════════ */

const char *QNN_ObjectStaticProperty(const qnn_static_object_t *obj, const char *key)
{
	int i;
	for (i = 0; i < obj->property_count; ++i)
	{
		if (!strcmp(obj->properties[i].key, key))
			return obj->properties[i].value;
	}
	return NULL;
}

int QNN_ObjectStaticPropertyInt(const qnn_static_object_t *obj, const char *key, int fallback)
{
	const char *value = QNN_ObjectStaticProperty(obj, key);
	if (value == NULL || value[0] == 0)
		return fallback;
	return atoi(value);
}
