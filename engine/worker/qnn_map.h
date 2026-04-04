/*
 * qnn_map.h — Engine BSP interface.
 *
 * Single point of contact with cl.worldmodel.  Extracts:
 *   - Raw entity list (classname, origin, angles, properties) → entity.c
 *   - Geometry (triangle soup) → navmesh.cpp via qnn_navmesh_build
 * No other module reads cl.worldmodel directly.
 */

#ifndef QNN_MAP_H
#define QNN_MAP_H

#include "qnn.h"

#define QNN_MAX_RAW_ENTITIES 2048

typedef struct qnn_raw_entity_s {
	int entity_num;
	char classname[QNN_MAX_CLASSNAME];
	char model_name[QNN_MAX_MODEL_NAME];
	vec3_t origin;
	vec3_t angles;
	int spawnflags;
	qboolean has_origin;
	qboolean has_model;
	qnn_property_t properties[QNN_MAX_STATIC_PROPERTIES];
	int property_count;
} qnn_raw_entity_t;

/* Parse the BSP entity lump (cl.worldmodel->entities) into raw entity
   records.  Returns the number of entities written to out[].
   Caller provides the output array with max capacity. */
int QNN_MapParseEntities(qnn_raw_entity_t *out, int max);

/* Build the entity list from cl_entities[] baselines (server edict numbers)
   joined with BSP entity lump data (classname, properties).  This is the
   authoritative entity source — edict numbers match the server's runtime
   numbering, so sound events and PVS updates map correctly.
   Entities with no baseline (lights, spawn points, DM-inhibited) are
   excluded.  Returns the number of entities written to out[]. */
int QNN_MapBuildFromBaselines(qnn_raw_entity_t *out, int max);

/* Classify a client entity by model name and skin (defined in qnn_entity.c). */
qboolean QNN_ClassifyByModel(const char *model_name, int skin, int *subject_id, int *qualifier_id, float *magnitude);

/* Extract BSP geometry and build the navmesh.  Reads cl.worldmodel
   for vertices/edges/surfaces.  Returns the built navmesh runtime,
   or NULL on failure.  Caller owns the returned runtime. */
qnn_navmesh_runtime_t *QNN_MapBuildNavmesh(
	qnn_navmesh_build_config_t *out_config,
	qnn_navmesh_summary_t *out_summary,
	char *error, size_t error_size);

#endif /* QNN_MAP_H */
