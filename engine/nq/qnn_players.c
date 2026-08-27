/*
 * qnn_players.c (nq) -- NetQuake runtime player transport.
 *
 * NetQuake client playback materializes visible player edicts in cl_entities[].
 * Keep that rule local to NQ so QuakeWorld cannot accidentally treat baseline
 * shim entities as live players.
 */

#include "qnn_object.h"
#include "qnn_map.h"
#include "qnn_context.h"

/* ── Team detection (NQ): pants color is the engine's own team field ────
 *   Matches engine logic: ent->v.team = (colors & 15) + 1  (host_cmd.c).
 *   Same pants color = same team.  Works for teamplay (teammates share
 *   color) and FFA (all different = all enemies).
 *
 *   Cached pants colors for all players, latched once when cl.scores
 *   first has nonzero entries, so that mid-demo level transitions
 *   (which zero cl.scores via CL_ClearState) don't flip the team signal.
 */
#define QNN_MAX_CACHED_PLAYERS 32
static int qnn_cached_pants[QNN_MAX_CACHED_PLAYERS];
static int qnn_cached_pants_count = 0;
static int qnn_self_pants_cached = -1;

void QNN_PlayersRegisterContext(void)
{
	QNN_ContextRegister(qnn_cached_pants, sizeof(qnn_cached_pants));
	QNN_ContextRegister(&qnn_cached_pants_count, sizeof(qnn_cached_pants_count));
	QNN_ContextRegister(&qnn_self_pants_cached, sizeof(qnn_self_pants_cached));
}

static void QNN_LatchTeamColors(void)
{
	int i, self_slot;

	if (cl.scores == NULL || cl.viewentity <= 0)
		return;
	self_slot = cl.viewentity - 1;
	if (self_slot < 0 || self_slot >= cl.maxclients)
		return;
	if (cl.scores[self_slot].colors == 0)
		return;
	qnn_self_pants_cached = cl.scores[self_slot].colors & 15;
	qnn_cached_pants_count = cl.maxclients < QNN_MAX_CACHED_PLAYERS ? cl.maxclients : QNN_MAX_CACHED_PLAYERS;
	for (i = 0; i < qnn_cached_pants_count; ++i)
		qnn_cached_pants[i] = cl.scores[i].colors & 15;
}

float QNN_IsSameTeam(int entity_num)
{
	int other_slot;

	if (qnn_self_pants_cached < 0)
		QNN_LatchTeamColors();
	if (qnn_self_pants_cached < 0)
		return 0.0f;

	other_slot = entity_num - 1;
	if (other_slot < 0 || other_slot >= qnn_cached_pants_count)
		return 0.0f;
	return (qnn_self_pants_cached == qnn_cached_pants[other_slot]) ? 1.0f : 0.0f;
}

void QNN_PlayersResetTeamCache(void)
{
	qnn_self_pants_cached = -1;
	qnn_cached_pants_count = 0;
}

qboolean QNN_IsLivePlayerSlot(int entity_num)
{
	entity_t *entity;
	int subject_id, qualifier_id;
	float magnitude;

	if (entity_num <= 0 || entity_num >= cl.num_entities)
		return false;
	if (entity_num == cl.viewentity)
		return false;
	entity = &cl_entities[entity_num];
	if (entity->model == NULL)
		return false;
	if (!QNN_ClassifyByModel(entity->model->name, entity->skinnum,
		&subject_id, &qualifier_id, &magnitude))
		return false;
	return subject_id == QNN_SUBJECT_PLAYER;
}

int QNN_AppendPlayerEntityUpdates(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int start_count, int max_entities)
{
	int entity_num;
	int count = start_count;

	for (entity_num = 1; entity_num <= cl.maxclients && count < max_entities; ++entity_num)
	{
		entity_t *entity;
		int qualifier_id, subject_id_unused;
		float magnitude;
		vec3_t anchor_origin;
		float half_extents[3];
		vec3_t delta;
		qnn_entity_update_t *eu;
		float server_dt = (float)(cl.mtime[0] - cl.mtime[1]);

		if (!QNN_IsLivePlayerSlot(entity_num))
			continue;
		entity = &cl_entities[entity_num];
		/* IsLivePlayerSlot already verified the model classifies as a
		 * player; re-call to recover qualifier/magnitude for the update. */
		(void)QNN_ClassifyByModel(entity->model->name, entity->skinnum,
			&subject_id_unused, &qualifier_id, &magnitude);

		if (server_dt < 0.001f || server_dt > 0.5f)
			server_dt = 1.0f / 20.0f;

		QNN_EntityAnchorFromModel(entity_num, entity->origin,
			anchor_origin, half_extents);
		if (!QNN_EntityInPvs(snapshot->player_origin, anchor_origin))
			continue;

		eu = &out_entities[count];
		memset(eu, 0, sizeof(*eu));
		eu->entity_num = entity_num;
		eu->subject_id = QNN_SUBJECT_PLAYER;
		eu->qualifier_id = qualifier_id;
		eu->magnitude = magnitude;
		VectorCopy(anchor_origin, eu->origin);
		VectorSubtract(entity->msg_origins[0], entity->msg_origins[1], delta);
		if (DotProduct(delta, delta) <= 500.0f * 500.0f)
			VectorScale(delta, 1.0f / server_dt, eu->velocity);
		VectorCopy(entity->angles, eu->angles);
		eu->effects = entity->effects;
		eu->half_extents[0] = half_extents[0];
		eu->half_extents[1] = half_extents[1];
		eu->half_extents[2] = half_extents[2];
		if (cl.scores != NULL && entity_num - 1 < cl.maxclients)
			eu->frags = cl.scores[entity_num - 1].frags;
		eu->in_fov = QNN_InFov(snapshot->player_origin,
			snapshot->player_view_angles, anchor_origin, half_extents[2]);
		count++;
	}

	return count;
}
