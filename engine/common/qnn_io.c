/*
 * qnn_io.c — Unified tick IO: action in, tokens out.
 *
 * Encapsulates the per-tick pipeline: world model update, token
 * emission, and obs buffer serialization.
 */

#include "qnn_io.h"
#include "qnn_object.h"
#include "qnn_store.h"
#include "qnn_map.h"
#include "qnn_route.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Lifecycle ─────────────────────────────────────────────────── */

void QNN_IOInit(const qnn_map_state_t *map_state)
{
	QNN_PlayersResetTeamCache();
	QNN_EventInit(map_state);
}

/* ── Per-frame update ──────────────────────────────────────────── */

void QNN_IOUpdate(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag)
{
	/* Stamp the entity store (e->pvs / origin / velocity) BEFORE
	 * EventTick runs.  EventTick's freshness checks (dimlight, mover
	 * toggle, player promote) compare e->pvs against now=cl.mtime[0],
	 * and want a within-this-HF stamp.  When StoreUpdate ran later
	 * (in IOEmit), e->pvs was always one HF stale → checks failed at
	 * any tick rate where cl.mtime advanced per HF (always true at
	 * 20 Hz host = engine tick = emit tick). */
	QNN_StoreUpdate(snapshot, dt);
	QNN_EventTick(snapshot, dt, reset_flag);
	if (reset_flag)
		QNN_OracleResetState();
}

/* ── Token emission ──────────────────────────────────────────────
 * StoreUpdate is now in IOUpdate so it runs before EventTick.  Emit
 * just reads the already-stamped store. */

void QNN_IOEmit(const qnn_snapshot_t *snapshot, qnn_tick_result_t *out)
{
	memset(out, 0, sizeof(*out));

	{
		int player_cluster_id;
		out->entity_count = QNN_OracleEmitTokens(out->entities,
			QNN_MAX_TOKEN_OBJECTS,
			snapshot, &qnn_map_state, &player_cluster_id);
	}
	QNN_SelfEmitToken(&out->self, snapshot);
	QNN_SpatialEmitTokens(snapshot, out->spatial);
}

/* ── Obs buffer serialization ─────────────────────────────────── */

void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *r)
{
	int i, j;
	int pos;

	memset(obs, 0, QNN_OBS_BUFFER_SIZE);

	/* Self */
	{
		const qnn_self_token_t *tok = &r->self;
		float scalars[QNN_OBS_SELF_SCALAR_DIM];

		scalars[0] = tok->health;
		scalars[1] = tok->armor;
		scalars[2] = tok->weapon_sg;
		scalars[3] = tok->weapon_ssg;
		scalars[4] = tok->weapon_ng;
		scalars[5] = tok->weapon_sng;
		scalars[6] = tok->weapon_gl;
		scalars[7] = tok->weapon_rl;
		scalars[8] = tok->weapon_lg;
		scalars[9] = tok->ammo_shells;
		scalars[10] = tok->ammo_nails;
		scalars[11] = tok->ammo_rockets;
		scalars[12] = tok->ammo_cells;
		scalars[13] = tok->vel[0];
		scalars[14] = tok->vel[1];
		scalars[15] = tok->vel[2];

		for (i = 0; i < QNN_OBS_SELF_SCALAR_DIM; ++i)
			QNN_BufWriteF32(obs, QNN_OBS_OFF_SELF_SCALARS + i * 4, scalars[i]);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_WEAPON_ID, tok->weapon_id);
		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_ARMOR_TYPE_ID, tok->armor_type_id);

		for (i = 0; i < QNN_OBS_SELF_POWERUP_SLOTS; ++i)
			QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_POWERUP_IDS + i * 4, tok->powerup_ids[i]);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_MOVEMENT_ID, tok->movement_id);
	}

	/* Spatial — fixed position after self: 9 × 13 float32 */
	for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
	{
		const qnn_spatial_token_t *tok = &r->spatial[i];
		int off = QNN_OBS_OFF_SPATIAL + i * QNN_OBS_SPATIAL_SCALAR_DIM * 4;
		QNN_BufWriteF32(obs, off, tok->dir[0]); off += 4;
		QNN_BufWriteF32(obs, off, tok->dir[1]); off += 4;
		QNN_BufWriteF32(obs, off, tok->dir[2]); off += 4;
		QNN_BufWriteF32(obs, off, QNN_Normalize(tok->nearest_dist, QNN_DIST_SCALE)); off += 4;
		QNN_BufWriteF32(obs, off, QNN_Normalize(tok->mean_dist, QNN_DIST_SCALE)); off += 4;
		QNN_BufWriteF32(obs, off, tok->openness); off += 4;
		QNN_BufWriteF32(obs, off, tok->clearance); off += 4;
		QNN_BufWriteF32(obs, off, tok->traversable); off += 4;
		QNN_BufWriteF32(obs, off, tok->dropoff); off += 4;
		QNN_BufWriteF32(obs, off, tok->solid_frac); off += 4;
		QNN_BufWriteF32(obs, off, tok->water_frac); off += 4;
		QNN_BufWriteF32(obs, off, tok->slime_frac); off += 4;
		QNN_BufWriteF32(obs, off, tok->lava_frac); off += 4;
	}

	/* Entities — variable-length tagged tokens at end of buffer.
	   Format: [n_tokens: u8] per token: [type: u8] [ids...] [scalars...] [n_events: u8] [events...] */
	{
	int pack_count = r->entity_count < QNN_MAX_TOKEN_OBJECTS ? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
	pos = QNN_OBS_OFF_ENTITY_STREAM;
	obs[pos++] = (uint8_t)pack_count;
	for (i = 0; i < pack_count; ++i)
	{
		const qnn_tagged_token_t *tt = &r->entities[i];
		obs[pos++] = (uint8_t)tt->type;

		switch (tt->type)
		{
		case QNN_TOKEN_PROJECTILE:
			{
				const qnn_projectile_token_t *tok = &tt->projectile;
				QNN_BufWriteI32(obs, pos, tok->subject_id); pos += 4;
				QNN_BufWriteI32(obs, pos, tok->modality_id); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->vel[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->vel[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->vel[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->recency); pos += 4;
				obs[pos++] = (uint8_t)tok->event_count;
				for (j = 0; j < tok->event_count; ++j)
				{
					QNN_BufWriteI32(obs, pos, tok->events[j].action_id); pos += 4;
					QNN_BufWriteI32(obs, pos, tok->events[j].source_id); pos += 4;
				}
			}
			break;
		case QNN_TOKEN_ACTOR:
			{
				const qnn_actor_token_t *tok = &tt->actor;
				QNN_BufWriteI32(obs, pos, tok->subject_id); pos += 4;
				QNN_BufWriteI32(obs, pos, tok->modality_id); pos += 4;
				QNN_BufWriteI32(obs, pos, tok->player_id); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->vel[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->vel[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->vel[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path_dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->eta); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->facing); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->team); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->score); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->recency); pos += 4;
				obs[pos++] = (uint8_t)tok->event_count;
				for (j = 0; j < tok->event_count; ++j)
				{
					QNN_BufWriteI32(obs, pos, tok->events[j].action_id); pos += 4;
					QNN_BufWriteI32(obs, pos, tok->events[j].source_id); pos += 4;
				}
			}
			break;
		case QNN_TOKEN_ITEM:
			{
				const qnn_item_token_t *tok = &tt->item;
				QNN_BufWriteI32(obs, pos, tok->subject_id); pos += 4;
				QNN_BufWriteI32(obs, pos, tok->modality_id); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path_dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->eta); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->amount); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->regen); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->recency); pos += 4;
				obs[pos++] = (uint8_t)tok->event_count;
				for (j = 0; j < tok->event_count; ++j)
				{
					QNN_BufWriteI32(obs, pos, tok->events[j].action_id); pos += 4;
					QNN_BufWriteI32(obs, pos, tok->events[j].source_id); pos += 4;
				}
			}
			break;
		case QNN_TOKEN_MOVER:
			{
				const qnn_mover_token_t *tok = &tt->mover;
				QNN_BufWriteI32(obs, pos, tok->subject_id); pos += 4;
				QNN_BufWriteI32(obs, pos, tok->modality_id); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->half_extents[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->rel[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[0]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[1]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path[2]); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->path_dist); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->eta); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->state); pos += 4;
				QNN_BufWriteF32(obs, pos, tok->recency); pos += 4;
				obs[pos++] = (uint8_t)tok->event_count;
				for (j = 0; j < tok->event_count; ++j)
				{
					QNN_BufWriteI32(obs, pos, tok->events[j].action_id); pos += 4;
					QNN_BufWriteI32(obs, pos, tok->events[j].source_id); pos += 4;
				}
			}
			break;
		}
	}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Map lifecycle (moved from qnn_map.cpp)
 * ══════════════════════════════════════════════════════════════════ */

void QNN_FreeMapState(qnn_map_state_t *map_state)
{
	int i;

	if (map_state == NULL)
		return;

	for (i = 0; i < map_state->static_object_count; ++i)
		free(map_state->static_objects[i].properties);

	QNN_RouteDestroy(map_state->route);
	qnn_navmesh_destroy(map_state->navmesh);
	free(map_state->static_objects);
	memset(map_state, 0, sizeof(*map_state));
}

static int QNN_StaticEntitySortCompare(const void *lhs_ptr, const void *rhs_ptr)
{
	const qnn_static_entity_t *lhs = (const qnn_static_entity_t *)lhs_ptr;
	const qnn_static_entity_t *rhs = (const qnn_static_entity_t *)rhs_ptr;
	int order_cmp;
	int classname_cmp;
	int axis;

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
	return lhs->entity_num - rhs->entity_num;
}

qboolean QNN_BuildMapState(qnn_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size)
{
	qnn_raw_entity_t *raw_entities;
	qnn_static_entity_t *entities;
	int raw_count;
	int entity_count;
	int i;

	memset(out, 0, sizeof(*out));
	strncpy(out->requested_map_id, requested_map_id, sizeof(out->requested_map_id) - 1);
	strncpy(out->map_name, map_name, sizeof(out->map_name) - 1);
	strncpy(out->source, "bsp_entities", sizeof(out->source) - 1);
	strncpy(out->navmesh_status, "missing", sizeof(out->navmesh_status) - 1);
	strncpy(out->route_status, "missing", sizeof(out->route_status) - 1);

	if (cl.worldmodel == NULL)
		return true;

	if (cl.worldmodel->entities == NULL)
	{
		snprintf(error, error_size, "Worldmodel has no entity text");
		return false;
	}

	/* 1. Build entity list from server baselines joined with BSP metadata */
	raw_entities = (qnn_raw_entity_t *)calloc(QNN_MAX_RAW_ENTITIES, sizeof(*raw_entities));
	if (raw_entities == NULL)
	{
		snprintf(error, error_size, "Out of memory allocating raw entities");
		return false;
	}
	raw_count = QNN_MapBuildFromBaselines(raw_entities, QNN_MAX_RAW_ENTITIES);

	/* 2. Classify raw entities into semantic entities */
	entities = (qnn_static_entity_t *)calloc(2048, sizeof(*entities));
	if (entities == NULL)
	{
		free(raw_entities);
		snprintf(error, error_size, "Out of memory allocating classified entities");
		return false;
	}
	entity_count = QNN_EntityClassifyStatic(raw_entities, raw_count, entities, 2048);
	if (entity_count > 1)
		qsort(entities, (size_t)entity_count, sizeof(entities[0]), QNN_StaticEntitySortCompare);

	for (i = 0; i < entity_count; ++i)
	{
		qnn_static_object_t *object;
		qnn_static_object_t *next_objects;
		int j;

		next_objects = (qnn_static_object_t *)realloc(out->static_objects,
			(size_t)(out->static_object_count + 1) * sizeof(*next_objects));
		if (next_objects == NULL)
		{
			snprintf(error, error_size, "Out of memory while building static object list");
			free(raw_entities);
			free(entities);
			QNN_FreeMapState(out);
			return false;
		}
		out->static_objects = next_objects;
		object = &out->static_objects[out->static_object_count];
		memset(object, 0, sizeof(*object));
		strncpy(object->category, entities[i].category, sizeof(object->category) - 1);
		strncpy(object->classname, entities[i].classname, sizeof(object->classname) - 1);
		snprintf(object->object_id, sizeof(object->object_id), "%s_%04d", entities[i].category, i);
		object->entity_num = entities[i].entity_num;
		VectorCopy(entities[i].origin, object->origin);
		VectorCopy(entities[i].angles, object->angles);

		/* Copy properties: malloc'd array for qnn_static_object_t */
		object->property_count = entities[i].property_count;
		object->properties = NULL;
		if (object->property_count > 0)
		{
			object->properties = (qnn_property_t *)malloc(
				(size_t)object->property_count * sizeof(qnn_property_t));
			if (object->properties == NULL)
			{
				snprintf(error, error_size, "Out of memory while copying entity properties");
				free(raw_entities);
				free(entities);
				QNN_FreeMapState(out);
				return false;
			}
			for (j = 0; j < object->property_count; ++j)
				object->properties[j] = entities[i].properties[j];
		}
		out->static_object_count += 1;
	}

	free(raw_entities);
	free(entities);

	/* Route build is optional -- don't fail the whole map load if it fails. */
	QNN_RouteBuildFromWorldmodel(out, error, error_size);
	return true;
}

static void QNN_CanonicalizeMap(char *out, size_t out_size, const char *requested)
{
	size_t i;

	snprintf(out, out_size, "%s", requested);
	for (i = 0; i < strlen(out); ++i)
		out[i] = (char)tolower((unsigned char)out[i]);
}

qboolean QNN_PrepareMap(const char *requested_map_id, char *error, size_t error_size)
{
	char saved_map_id[QNN_MAX_MAP_ID];
	char map_name[QNN_MAX_MAP_ID];

	if (!requested_map_id || !requested_map_id[0])
	{
		snprintf(error, error_size, "map_id is required");
		return false;
	}

	/* Defensive copy: the caller (e.g. QNN_HandleReset) may pass a pointer
	   that aliases qnn_map_state.requested_map_id, which QNN_FreeMapState
	   zeroes below. Without this copy, the subsequent strncpy in
	   QNN_BuildMapState reads from zeroed memory and silently leaves
	   qnn_map_state.requested_map_id empty, failing the next reset. */
	strncpy(saved_map_id, requested_map_id, sizeof(saved_map_id) - 1);
	saved_map_id[sizeof(saved_map_id) - 1] = '\0';
	requested_map_id = saved_map_id;

	QNN_CanonicalizeMap(map_name, sizeof(map_name), requested_map_id);

	/* Cache hit: same map, navmesh built, worldmodel unchanged. */
	if (!strcmp(qnn_map_state.requested_map_id, requested_map_id)
		&& !strcmp(qnn_map_state.map_name, map_name)
		&& qnn_map_state.navmesh != NULL
		&& qnn_map_state.cached_worldmodel == cl.worldmodel)
		return true;

	QNN_FreeMapState(&qnn_map_state);
	if (!QNN_BuildMapState(&qnn_map_state, requested_map_id, map_name, error, error_size))
		return false;
	qnn_map_state.cached_worldmodel = cl.worldmodel;
	return true;
}
