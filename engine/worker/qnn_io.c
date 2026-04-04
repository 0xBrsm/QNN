/*
 * qnn_io.c — Unified tick IO: action in, tokens out.
 *
 * Encapsulates the per-tick pipeline: world model update, token emission,
 * action history, and obs buffer serialization.
 */

#include "qnn_io.h"
#include "qnn_object.h"
#include "qnn_map.h"
#include "qnn_route.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Action history (internal) ─────────────────────────────────── */

static float qnn_action_history[QNN_OBS_ACTION_HISTORY_LEN][QNN_OBS_ACTION_HISTORY_DIM];
static int qnn_action_history_count = 0;

static qboolean QNN_ActionHasSignal(const qnn_action_t *action)
{
	return (action->move[0] != 0.0f || action->move[1] != 0.0f
		|| action->look[0] != 0.0f || action->look[1] != 0.0f
		|| action->fire || action->jump || action->switch_slot) ? true : false;
}

static void QNN_ActionReset(void)
{
	memset(qnn_action_history, 0, sizeof(qnn_action_history));
	qnn_action_history_count = 0;
}

static void QNN_ActionPush(const qnn_action_t *action)
{
	float features[QNN_OBS_ACTION_HISTORY_DIM];
	int i;

	features[0] = action->move[0];
	features[1] = action->move[1];
	features[2] = action->look[0];
	features[3] = action->look[1];
	features[4] = (float)action->fire;
	features[5] = (float)action->jump;
	features[6] = (float)(action->switch_slot < 0 ? 0 : action->switch_slot > QNN_ACTION_SWITCH_SLOTS ? QNN_ACTION_SWITCH_SLOTS : action->switch_slot)
		/ (float)QNN_ACTION_SWITCH_SLOTS;

	if (qnn_action_history_count >= QNN_OBS_ACTION_HISTORY_LEN)
	{
		memmove(qnn_action_history[0], qnn_action_history[1],
			(QNN_OBS_ACTION_HISTORY_LEN - 1) * sizeof(qnn_action_history[0]));
		qnn_action_history_count = QNN_OBS_ACTION_HISTORY_LEN - 1;
	}
	for (i = 0; i < QNN_OBS_ACTION_HISTORY_DIM; ++i)
		qnn_action_history[qnn_action_history_count][i] = features[i];
	qnn_action_history_count++;
}

static int QNN_ActionEmit(qnn_action_token_t *out, int max_tokens)
{
	int n, i;

	n = qnn_action_history_count < max_tokens ? qnn_action_history_count : max_tokens;
	if (n > QNN_OBS_ACTION_HISTORY_LEN)
		n = QNN_OBS_ACTION_HISTORY_LEN;

	for (i = 0; i < n; ++i)
	{
		out[i].move[0]     = qnn_action_history[i][0];
		out[i].move[1]     = qnn_action_history[i][1];
		out[i].look[0]     = qnn_action_history[i][2];
		out[i].look[1]     = qnn_action_history[i][3];
		out[i].fire        = qnn_action_history[i][4];
		out[i].jump        = qnn_action_history[i][5];
		out[i].switch_norm = qnn_action_history[i][6];
	}
	return n;
}

/* ── Lifecycle ─────────────────────────────────────────────────── */

void QNN_IOInit(const qnn_map_state_t *map_state)
{
	QNN_EntityResetTeamCache();
	QNN_EventInit(map_state);
	QNN_ActionReset();
}

/* ── Per-frame update ──────────────────────────────────────────── */

void QNN_IOUpdate(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag)
{
	if (reset_flag)
		QNN_ActionReset();
	QNN_EventTick(snapshot, dt, reset_flag);
}

/* ── Token emission ────────────────────────────────────────────── */

void QNN_IOEmit(const qnn_snapshot_t *snapshot, qnn_tick_result_t *out)
{
	memset(out, 0, sizeof(*out));

	out->entity_count = QNN_OracleEmitTokens(out->entities,
		QNN_MAX_TOKEN_OBJECTS,
		snapshot, &qnn_map_state, &out->player_cluster_id);
	QNN_SelfEmitToken(&out->self, snapshot, out->player_cluster_id);
	QNN_SpatialEmitTokens(snapshot, out->spatial);
	out->action_history_count = QNN_ActionEmit(out->action_history, QNN_OBS_ACTION_HISTORY_LEN);
}

void QNN_IOPushAction(const qnn_snapshot_t *snapshot)
{
	if (QNN_ActionHasSignal(&snapshot->action_label))
		QNN_ActionPush(&snapshot->action_label);
}

/* ── Obs buffer serialization ─────────────────────────────────── */

void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *r)
{
	int i, j;

	memset(obs, 0, QNN_OBS_BUFFER_SIZE);

	/* Self */
	{
		const qnn_self_token_t *tok = &r->self;
		float scalars[QNN_OBS_SELF_SCALAR_DIM];

		scalars[0] = tok->health;
		scalars[1] = tok->armor;
		scalars[2] = tok->weapon_sg;
		scalars[3] = tok->weapon_ng;
		scalars[4] = tok->weapon_gl;
		scalars[5] = tok->weapon_rl;
		scalars[6] = tok->weapon_lg;
		scalars[7] = tok->ammo_shells;
		scalars[8] = tok->ammo_nails;
		scalars[9] = tok->ammo_rockets;
		scalars[10] = tok->ammo_cells;
		scalars[11] = tok->vel[0];
		scalars[12] = tok->vel[1];
		scalars[13] = tok->vel[2];

		for (i = 0; i < QNN_OBS_SELF_SCALAR_DIM; ++i)
			QNN_BufWriteF32(obs, QNN_OBS_OFF_SELF_SCALARS + i * 4, scalars[i]);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_WEAPON_ID, tok->weapon_id);
		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_ARMOR_TYPE_ID, tok->armor_type_id);

		for (i = 0; i < QNN_OBS_SELF_POWERUP_SLOTS; ++i)
			QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_POWERUP_IDS + i * 4, tok->powerup_ids[i]);
		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_POWERUP_COUNT, tok->powerup_count);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_MOVEMENT_ID, tok->movement_id);
		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_CLUSTER_ID, tok->cluster_id);
	}

	/* Entities — obs buffer has room for QNN_MAX_TOKEN_OBJECTS slots */
	{
	int pack_count = r->entity_count < QNN_MAX_TOKEN_OBJECTS ? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
	for (i = 0; i < pack_count; ++i)
	{
		const qnn_entity_token_t *tok = &r->entities[i];
		int ids_off = QNN_OBS_OFF_OBJECT_IDS + i * QNN_OBS_OBJECT_ID_DIM * 4;
		int sc_off = QNN_OBS_OFF_OBJECT_SCALARS + i * QNN_OBS_OBJECT_SCALAR_DIM * 4;
		int rc_off = QNN_OBS_OFF_OBJECT_ROUTE_IDS + i * QNN_MAX_ROUTE_CLUSTERS * 4;
		int ev_ids_off = QNN_OBS_OFF_OBJECT_EVENT_IDS + i * QNN_MAX_ENTITY_EVENTS * QNN_ENTITY_EVENT_ID_DIM * 4;
		int ev_sc_off = QNN_OBS_OFF_OBJECT_EVENT_SCALARS + i * QNN_MAX_ENTITY_EVENTS * 4;

		QNN_BufWriteI32(obs, ids_off +  0, tok->subject_id);
		QNN_BufWriteI32(obs, ids_off +  4, tok->qualifier_id);
		QNN_BufWriteI32(obs, ids_off +  8, tok->modality_id);
		QNN_BufWriteI32(obs, ids_off + 12, tok->player_id);
		QNN_BufWriteI32(obs, ids_off + 16, tok->cluster_id);
		QNN_BufWriteI32(obs, ids_off + 20, tok->powerup_subject_id);
		QNN_BufWriteI32(obs, ids_off + 24, tok->weapon_subject_id);

		QNN_BufWriteF32(obs, sc_off +  0, tok->rel[0]);
		QNN_BufWriteF32(obs, sc_off +  4, tok->rel[1]);
		QNN_BufWriteF32(obs, sc_off +  8, tok->rel[2]);
		QNN_BufWriteF32(obs, sc_off + 12, tok->distance);
		QNN_BufWriteF32(obs, sc_off + 16, tok->route_cost);
		QNN_BufWriteF32(obs, sc_off + 20, tok->vel[0]);
		QNN_BufWriteF32(obs, sc_off + 24, tok->vel[1]);
		QNN_BufWriteF32(obs, sc_off + 28, tok->vel[2]);
		QNN_BufWriteF32(obs, sc_off + 32, tok->rel_yaw);
		QNN_BufWriteF32(obs, sc_off + 36, tok->rel_pitch);
		QNN_BufWriteF32(obs, sc_off + 40, tok->half_extents[0]);
		QNN_BufWriteF32(obs, sc_off + 44, tok->half_extents[1]);
		QNN_BufWriteF32(obs, sc_off + 48, tok->half_extents[2]);
		QNN_BufWriteF32(obs, sc_off + 52, tok->recency);
		QNN_BufWriteF32(obs, sc_off + 56, tok->confidence);
		QNN_BufWriteF32(obs, sc_off + 60, tok->magnitude);
		QNN_BufWriteF32(obs, sc_off + 64, tok->state);

		obs[QNN_OBS_OFF_OBJECT_MASK + i] = 1;

		for (j = 0; j < QNN_MAX_ROUTE_CLUSTERS; ++j)
			QNN_BufWriteI32(obs, rc_off + j * 4, tok->route_cluster_ids[j]);

		for (j = 0; j < tok->event_count; ++j)
		{
			int slot_off = ev_ids_off + j * QNN_ENTITY_EVENT_ID_DIM * 4;
			QNN_BufWriteI32(obs, slot_off + 0, tok->event_subject[j]);
			QNN_BufWriteI32(obs, slot_off + 4, tok->event_action[j]);
			QNN_BufWriteI32(obs, slot_off + 8, tok->event_qualifier[j]);
			QNN_BufWriteF32(obs, ev_sc_off + j * 4, tok->event_recency[j]);
		}
		obs[QNN_OBS_OFF_OBJECT_EVENT_COUNTS + i] = (uint8_t)tok->event_count;
	}
	}

	/* Spatial */
	for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
	{
		const qnn_spatial_token_t *tok = &r->spatial[i];
		int sid_off = QNN_OBS_OFF_SPATIAL_IDS + i * 4;
		int ssc_off = QNN_OBS_OFF_SPATIAL_SCALARS + i * QNN_OBS_SPATIAL_SCALAR_DIM * 4;

		QNN_BufWriteI32(obs, sid_off, tok->sector_id);
		QNN_BufWriteF32(obs, ssc_off +  0, QNN_Normalize(tok->nearest_dist, QNN_DIST_SCALE));
		QNN_BufWriteF32(obs, ssc_off +  4, QNN_Normalize(tok->mean_dist, QNN_DIST_SCALE));
		QNN_BufWriteF32(obs, ssc_off +  8, tok->openness);
		QNN_BufWriteF32(obs, ssc_off + 12, tok->clearance);
		QNN_BufWriteF32(obs, ssc_off + 16, tok->traversable);
		QNN_BufWriteF32(obs, ssc_off + 20, tok->dropoff);
		QNN_BufWriteF32(obs, ssc_off + 24, tok->solid_frac);
		QNN_BufWriteF32(obs, ssc_off + 28, tok->water_frac);
		QNN_BufWriteF32(obs, ssc_off + 32, tok->slime_frac);
		QNN_BufWriteF32(obs, ssc_off + 36, tok->lava_frac);
	}

	/* Action history */
	for (i = 0; i < r->action_history_count; ++i)
	{
		const qnn_action_token_t *tok = &r->action_history[i];
		int off = QNN_OBS_OFF_ACTION_HISTORY + i * QNN_OBS_ACTION_HISTORY_DIM * 4;

		QNN_BufWriteF32(obs, off +  0, tok->move[0]);
		QNN_BufWriteF32(obs, off +  4, tok->move[1]);
		QNN_BufWriteF32(obs, off +  8, tok->look[0]);
		QNN_BufWriteF32(obs, off + 12, tok->look[1]);
		QNN_BufWriteF32(obs, off + 16, tok->fire);
		QNN_BufWriteF32(obs, off + 20, tok->jump);
		QNN_BufWriteF32(obs, off + 24, tok->switch_norm);
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
	char map_name[QNN_MAX_MAP_ID];

	if (!requested_map_id || !requested_map_id[0])
	{
		snprintf(error, error_size, "map_id is required");
		return false;
	}

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
