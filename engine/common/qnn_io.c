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
#include "qnn_collect_helpers.h"  /* qnn_runtime — look_delta carry reset */

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Lifecycle ─────────────────────────────────────────────────── */

void QNN_IOInit(const qnn_map_state_t *map_state)
{
	/* Register the perception cone cvar once. IOInit re-runs per map
	 * load; the guard keeps Cvar_RegisterVariable from warning on the
	 * repeats and preserves any console-set value across map changes. */
	if (Cvar_FindVar("qnn_fov") == NULL)
		Cvar_RegisterVariable(&qnn_fov);

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
	{
		QNN_OracleResetState();
		/* Drop the look_delta carry so the first frame(s) of a new
		 * episode don't difference across the boundary. */
		qnn_runtime.ld_has_prev_view = false;
		qnn_runtime.ld_has_prev_realized = false;
	}
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

/* Pack the event block: u8 count followed by `count` interleaved
 * (action, source) u8 pairs.  Matches the Python wire parser in
 * src/qnn/wire.py:_unpack_native_entity_stream. */
static int QNN_PackEventSlots(uint8_t *obs, int pos,
	const qnn_token_event_t *events, int event_count)
{
	int j;
	int cap = event_count;
	if (cap > QNN_MAX_ENTITY_EVENTS) cap = QNN_MAX_ENTITY_EVENTS;
	if (cap < 0) cap = 0;

	obs[pos++] = (uint8_t)cap;
	for (j = 0; j < cap; ++j)
	{
		obs[pos++] = (uint8_t)events[j].action_id;
		obs[pos++] = (uint8_t)events[j].source_id;
	}
	return pos;
}

void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *r)
{
	int i;
	int pos;

	memset(obs, 0, QNN_OBS_BUFFER_SIZE);

	/* ── Self block (27 B) ───────────────────────────────────── */
	{
		const qnn_self_token_t *tok = &r->self;
		float eff_armor = (float)tok->raw_armor * tok->armor_type;

		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_HEALTH,
			QNN_QuantizeU8Saturating((float)tok->health));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_EFF_ARMOR,
			QNN_QuantizeU8Saturating(eff_armor));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_SHELLS,
			QNN_QuantizeU8Saturating((float)tok->ammo_shells));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_NAILS,
			QNN_QuantizeU8Saturating((float)tok->ammo_nails));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_ROCKETS,
			QNN_QuantizeU8Saturating((float)tok->ammo_rockets));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_CELLS,
			QNN_QuantizeU8Saturating((float)tok->ammo_cells));

		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 0,
			QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE));
		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 2,
			QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE));
		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 4,
			QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE));

		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_ATTACK_FIN, tok->attack_finished);

		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_WEAPON_ID,
			(uint8_t)tok->weapon_id);
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_MOVEMENT_ID,
			(uint8_t)tok->movement_id);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_ITEMS, tok->items);

		QNN_BufWriteI8 (obs, QNN_OBS_OFF_SELF_VIEW_PITCH,
			QNN_QuantizeI8(tok->view_pitch));

		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 0, tok->look_delta[0]);
		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 2, tok->look_delta[1]);
		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 4, tok->look_delta[2]);
	}

	/* ── Spatial block (135 B = field-major across 9 sectors) ─
	 * Layout: 9 sectors × dir (27 B), 9 × nearest_dist (18 B),
	 *         9 × mean_dist (18 B), then 8 × (9 × u8) for the
	 *         clamped [0, 1] fractions.  Matches qnn/wire.py's
	 *         _unpack_native_spatial. */
	{
		int p = QNN_OBS_OFF_SPATIAL;
		/* spatial_dir [9, 3] i8 */
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
		{
			const qnn_spatial_token_t *tok = &r->spatial[i];
			QNN_BufWriteI8(obs, p++, QNN_QuantizeI8(tok->dir[0]));
			QNN_BufWriteI8(obs, p++, QNN_QuantizeI8(tok->dir[1]));
			QNN_BufWriteI8(obs, p++, QNN_QuantizeI8(tok->dir[2]));
		}
		/* spatial_nearest_dist [9] u16 */
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
		{
			QNN_BufWriteU16(obs, p, QNN_QuantizeU16Saturating(r->spatial[i].nearest_dist));
			p += 2;
		}
		/* spatial_mean_dist [9] u16 */
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
		{
			QNN_BufWriteU16(obs, p, QNN_QuantizeU16Saturating(r->spatial[i].mean_dist));
			p += 2;
		}
		/* 8 × (9 × u8) — openness, clearance, traversable, dropoff,
		 * solid_frac, water_frac, slime_frac, lava_frac. */
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].openness);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].clearance);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].traversable);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].dropoff);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].solid_frac);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].water_frac);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].slime_frac);
		for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i) obs[p++] = QNN_QuantizeU8Unit(r->spatial[i].lava_frac);
	}

	/* ── Entity stream (variable-length, native widths) ──────── */
	{
		int pack_count = r->entity_count < QNN_MAX_TOKEN_OBJECTS
			? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
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
					/* Common IDs — projectile has no player_id. */
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					/* Events */
					pos = QNN_PackEventSlots(obs, pos, tok->events, tok->event_count);
					/* Per-type scalars (14 B): rel i16×3, vel i16×3, recency f16 */
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;

			case QNN_TOKEN_ACTOR:
				{
					const qnn_actor_token_t *tok = &tt->actor;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					obs[pos++] = (uint8_t)tok->player_id;
					pos = QNN_PackEventSlots(obs, pos, tok->events, tok->event_count);
					/* Per-type (30 B): half u8×3, rel i16×3, vel i16×3, path i16×3,
					 * path_dist u16, eta f16, facing u8, team u8, score u8, recency f16 */
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[2], 32767.0f)); pos += 2;
					QNN_BufWriteU16(obs, pos, QNN_QuantizeU16Saturating(tok->path_dist)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->eta); pos += 2;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->facing)); pos += 1;
					QNN_BufWriteU8 (obs, pos, (uint8_t)(tok->team > 0.5f ? 1 : 0)); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->score)); pos += 1;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;

			case QNN_TOKEN_ITEM:
				{
					const qnn_item_token_t *tok = &tt->item;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					pos = QNN_PackEventSlots(obs, pos, tok->events, tok->event_count);
					/* Per-type (24 B): half u8×3, rel i16×3, path i16×3,
					 * path_dist u16, eta f16, amount u8, regen f16, recency f16 */
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[2], 32767.0f)); pos += 2;
					QNN_BufWriteU16(obs, pos, QNN_QuantizeU16Saturating(tok->path_dist)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->eta); pos += 2;
					/* Raw engine pickup amount as u8 saturating; model
					 * applies per-subject normalization via
					 * qnn.engine_norm.ITEM_AMOUNT_MULT/CONST. */
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->amount)); pos += 1;
					QNN_BufWriteF16(obs, pos, tok->regen); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;

			case QNN_TOKEN_MOVER:
				{
					const qnn_mover_token_t *tok = &tt->mover;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					pos = QNN_PackEventSlots(obs, pos, tok->events, tok->event_count);
					/* Per-type (22 B): half u8×3, rel i16×3, path i16×3,
					 * path_dist u16, eta f16, state u8, recency f16 */
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[2], 32767.0f)); pos += 2;
					QNN_BufWriteU16(obs, pos, QNN_QuantizeU16Saturating(tok->path_dist)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->eta); pos += 2;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->state)); pos += 1;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
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
