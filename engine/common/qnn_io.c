/*
 * qnn_io.c — Unified tick IO: action in, tokens out.
 *
 * Encapsulates the per-tick pipeline: world model update, token
 * emission, and obs buffer serialization.
 */

#include "qnn_io.h"
#include "qnn_obs_registry.h"
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
	/* Register the perception cone cvar once (idempotent — IOInit
	 * re-runs per map load; a console-set value survives map changes). */
	QNN_RegisterPerceptionCvars();

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

/* WS2 residue of the codec spatial-mode shim: whether QNN_IOEmitPlan
 * additionally emits the v1 raycast scalars (wire.7/.9/.11 — not a
 * registry field). The atlas parameterization rides the plan now. */
static qnn_spatial_mode_t qnn_io_spatial_mode = QNN_SPATIAL_MODE_ATLAS;

void QNN_IOSetSpatialMode(qnn_spatial_mode_t mode)
{
	qnn_io_spatial_mode = mode;
}

/* Entity-stream contract the ACTION path assumes (qnn_input.c's
 * weapon-vs-attack impulse branch). Default = COMBAT (the A27 tree's
 * native contract); QNN_OnnxInit overrides it from the loaded model's
 * codec (FULL for wire.9/.11/.12.x). The oracle's per-tick token
 * qualification is derived from the emit plan's entity policy in
 * QNN_IOProvideEntities below — set around the oracle call so
 * heterogeneous per-seat plans qualify independently. */
static qnn_entity_mode_t qnn_io_entity_mode = QNN_ENTITY_MODE_COMBAT;

void QNN_IOSetEntityMode(qnn_entity_mode_t mode)
{
	qnn_io_entity_mode = mode;
}

qnn_entity_mode_t QNN_IOGetEntityMode(void)
{
	return qnn_io_entity_mode;
}

/* ── Emit plan (obs API) ─────────────────────────────────────────
 * QNN_IOEmit / QNN_IOPackObsBuffer walk a compiled emit plan
 * (qnn_obs_registry.{c,h}).  Until WS2 lands per-seat declarations,
 * every consumer runs the DEFAULT plan — today's packed 864-byte
 * frame; the plan machinery is gate-tested to reproduce it
 * bit-identically (src/engine/tests/qnn_obs_registry_test.c). */

const qnn_obs_plan_t *QNN_IODefaultObsPlan(void)
{
	static qnn_obs_plan_t plan;
	static int compiled;

	if (!compiled)
	{
		qnn_obs_decl_t decl;
		char error[256];

		QNN_ObsDeclDefault(&decl);
		if (!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)))
		{
			fprintf(stderr, "qnn_io: default obs plan failed to "
				"compile: %s\n", error);
			abort();
		}
		/* The default plan IS today's frame — payload 848 + 16-byte
		 * pose tail.  Any drift here is a wire break; die loudly. */
		if (plan.frame_bytes != QNN_OBS_BUFFER_SIZE)
		{
			fprintf(stderr, "qnn_io: default obs plan frame is %d B, "
				"expected %d B\n", plan.frame_bytes, QNN_OBS_BUFFER_SIZE);
			abort();
		}
		compiled = 1;
	}
	return &plan;
}

/* Engine-side compute providers.  QNN_ObsPlanCompute calls a provider
 * ONLY when the plan demands its kind; the qnn_runtime skip flags (a
 * per-collect compute gate, both default true — set in
 * QNN_HandleCollect) layer on top. */

static void QNN_IOProvideSelf(const qnn_snapshot_t *snapshot,
	qnn_self_token_t *out)
{
	QNN_SelfEmitToken(out, snapshot);
}

static int QNN_IOProvideEntities(const qnn_snapshot_t *snapshot,
	const qnn_obs_entity_params_t *params, qnn_tagged_token_t *out,
	int max_tokens)
{
	int player_cluster_id;
	int count;
	qnn_entity_mode_t saved;

	if (qnn_runtime.skip_entities)
		return 0;
	/* Oracle disclosure follows the PLAN's policy, not the process
	 * global: v1 = the FULL ladder (all token types, memory tail),
	 * v3 = COMBAT (current-frame actor/projectile).  Save/restore so
	 * heterogeneous per-seat plans in one world qualify independently
	 * and the steady global keeps serving the action path. */
	saved = qnn_io_entity_mode;
	qnn_io_entity_mode = (strcmp(params->policy, "v1") == 0)
		? QNN_ENTITY_MODE_FULL : QNN_ENTITY_MODE_COMBAT;
	count = QNN_OracleEmitTokens(out, max_tokens,
		snapshot, &qnn_map_state, &player_cluster_id);
	qnn_io_entity_mode = saved;
	return count;
}

static void QNN_IOProvideAtlas(const qnn_snapshot_t *snapshot,
	const qnn_obs_atlas_params_t *params,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX])
{
	if (qnn_runtime.skip_spatial)
		return;
	/* Parameterization is plan-carried: 24 for the packed frontier,
	 * 72 for the a26/a27 rc1 lines.  (The old ATLAS_LEGACY global
	 * override is retired; raycast contracts simply do not request
	 * the atlas.) */
	QNN_SpatialEmitAtlas(snapshot, atlas, params->yaw);
}

static const qnn_obs_compute_fns_t qnn_io_compute_fns = {
	QNN_IOProvideSelf,
	QNN_IOProvideEntities,
	QNN_IOProvideAtlas,
};

void QNN_IOEmitPlan(const qnn_obs_plan_t *plan,
	const qnn_snapshot_t *snapshot, qnn_tick_result_t *out)
{
	QNN_ObsPlanCompute(plan, &qnn_io_compute_fns, snapshot, out);

	/* wire.7/.9/.11 raycast residue: the v1 raycast scalars are not a
	 * registry field (they live outside the flat obs buffer, consumed
	 * by the ONNX scratch packer); keep emitting them when the loaded
	 * codec selected that contract. */
	if (qnn_io_spatial_mode == QNN_SPATIAL_MODE_RAYCAST_V1
		&& !qnn_runtime.skip_spatial)
		QNN_SpatialEmitTokens(snapshot, out->spatial);
}

void QNN_IOEmit(const qnn_snapshot_t *snapshot, qnn_tick_result_t *out)
{
	QNN_IOEmitPlan(QNN_IODefaultObsPlan(), snapshot, out);
}

/* ── Obs buffer serialization ─────────────────────────────────── */

int QNN_IOPoseTailEnabled(void)
{
	static int checked, enabled;

	if (!checked)
	{
		const char *flag = getenv("QNN_POSE_TAIL");
		enabled = (flag != NULL && flag[0] == '1');
		checked = 1;
	}
	return enabled;
}

void QNN_IOStashPoseTail(uint8_t *obs, int frame_bytes,
	const qnn_snapshot_t *snapshot)
{
	float pose[4];

	/* Every compiled plan reserves QNN_OBS_POSE_TAIL_BYTES at the end
	 * of its frame; a shorter buffer here is a programming error. */
	if (frame_bytes < (int)sizeof(pose))
	{
		fprintf(stderr, "qnn_io: pose tail into a %d-byte frame\n",
			frame_bytes);
		abort();
	}
	pose[0] = snapshot->player_origin[0];
	pose[1] = snapshot->player_origin[1];
	pose[2] = snapshot->player_origin[2];
	pose[3] = snapshot->player_view_angles[1];
	memcpy(obs + frame_bytes - (int)sizeof(pose), pose, sizeof(pose));
}

void QNN_IOPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *r)
{
	/* Legacy fixed-frame entry point: serialize the DEFAULT plan into
	 * the classic 864-byte buffer.  Per-seat plans go through
	 * QNN_ObsPlanPack directly (WS2). */
	QNN_ObsPlanPack(QNN_IODefaultObsPlan(), obs, QNN_OBS_BUFFER_SIZE, r);
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
