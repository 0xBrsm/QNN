#include "qnn_arena_observer.h"

#include "qnn_collect_helpers.h"
#include "qnn_io.h"
#include "qnn_obs_registry.h"
#include "qnn_predict.h"

#include <string.h>
#include <stdlib.h>

static char qnn_arena_map_id[QNN_MAX_MAP_ID];

void QNN_ArenaNewMap(void)
{
	/* Rendering is intentionally absent from arena observers.  The stock
	 * R_NewMap path clears particle/surface state that a dedicated host never
	 * initialized; parsing and observation construction need only worldmodel
	 * and the entity baselines already installed by CL_ParseServerInfo. */
	if (sv.active)
		cl_numvisedicts = 0;
	else
		R_NewMap();
}

void QNN_ArenaNewTranslation(int slot)
{
	if (!sv.active)
		CL_NewTranslation(slot);
}

/* Parsing already filled msg_origins/msg_angles; PPO needs current transforms
 * and self velocity, not renderer efrags, trails, particles, dynamic lights,
 * or temporary-entity draw lists. */
void QNN_ArenaRelinkEntities(void)
{
	entity_t *entity;
	int entity_num;

	VectorCopy(cl.mvelocity[0], cl.velocity);
	for (entity_num = 1; entity_num < cl.num_entities; ++entity_num)
	{
		entity = &cl_entities[entity_num];
		if (entity->model == NULL)
			continue;
		if (entity->msgtime != cl.mtime[0])
		{
			entity->model = NULL;
			continue;
		}
		VectorCopy(entity->msg_origins[0], entity->origin);
		VectorCopy(entity->msg_angles[0], entity->angles);
		entity->forcelink = false;
	}
}

qboolean QNN_ArenaObserverReady(void)
{
	return cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS;
}

static void QNN_ArenaDeriveMapId(char *out, size_t out_size)
{
	const char *name;
	const char *base;
	const char *dot;
	size_t length;

	out[0] = 0;
	if (cl.worldmodel == NULL)
		return;
	name = cl.worldmodel->name;
	base = strrchr(name, '/');
	base = base ? base + 1 : name;
	dot = strrchr(base, '.');
	length = dot ? (size_t)(dot - base) : strlen(base);
	if (length >= out_size)
		length = out_size - 1;
	memcpy(out, base, length);
	out[length] = 0;
}

void QNN_ArenaObserverPrepare(void)
{
	char error[256];

	QNN_ArenaDeriveMapId(qnn_arena_map_id, sizeof(qnn_arena_map_id));
	if (qnn_arena_map_id[0] == 0
		|| !QNN_PrepareMap(qnn_arena_map_id, error, sizeof(error)))
		Sys_Error("Arena observer map preparation failed: %s", error);
	QNN_IOInit(&qnn_map_state);
}

void QNN_ArenaObserverWrite(FILE *out, const struct qnn_obs_plan_s *plan,
	const qnn_action_t *previous_action,
	int tick, int steps, qboolean reset_flag, float dt)
{
	qnn_snapshot_t snapshot;
	qnn_tick_result_t result;
	/* Frame buffer sized for the largest plan seen — per-seat plans
	 * (OP_ATTACH_DECL) may exceed the default 864 (the legacy 4096
	 * fixed-frame generations).  Single-threaded, like all emit state. */
	static uint8_t *obs;
	static int obs_cap;

	if (plan == NULL)
		plan = QNN_IODefaultObsPlan();
	if (plan->frame_bytes > obs_cap)
	{
		obs = (uint8_t *)realloc(obs, (size_t)plan->frame_bytes);
		if (obs == NULL)
			Sys_Error("Arena observer: out of memory for a %d-byte frame",
				plan->frame_bytes);
		obs_cap = plan->frame_bytes;
	}

	QNN_CaptureBaseSnapshot(&snapshot);
	if (getenv("QNN_ARENA_OBSERVER_DEBUG") != NULL)
		fprintf(stderr,
			"arena_observer view=%d health=%d mtime=%.3f origin=%.1f,%.1f,%.1f reset=%d\n",
			cl.viewentity, snapshot.health, cl.mtime[0],
			snapshot.player_origin[0], snapshot.player_origin[1],
			snapshot.player_origin[2], reset_flag ? 1 : 0);
	if (reset_flag)
		QNN_PredictReset();
	QNN_PredictTick(dt);
	QNN_PredictSelfVelocity(snapshot.player_velocity);
	snapshot.action_label = *previous_action;
	snapshot.done = QNN_TrainingNetworkRoundReset();
	QNN_DrainSounds(&snapshot);
	QNN_IOUpdate(&snapshot, dt, reset_flag);
	QNN_IOEmitPlan(plan, &snapshot, &result);
	QNN_ObsPlanPack(plan, obs, plan->frame_bytes, &result);
	if (QNN_IOPoseTailEnabled())
		QNN_IOStashPoseTail(obs, plan->frame_bytes, &snapshot);
	fwrite(obs, 1, (size_t)plan->frame_bytes, out);
	QNN_WriteTrainingExtrasBinary(out, &snapshot, tick, steps, reset_flag);
	fflush(out);
}
