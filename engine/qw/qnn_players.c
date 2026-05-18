/*
 * qnn_players.c (qw) -- QuakeWorld runtime player transport.
 *
 * QW active players are not packet/baseline entities.  They arrive via
 * svc_playerinfo into player_state_t, with identity in cl.players[].
 * QWD uses the current frame ring buffer; MVD uses the latest-state array
 * maintained by the worker's MVD parser.
 */

#include "qnn_object.h"

#include <string.h>

extern player_state_t qnn_mvd_latest_playerstate[MAX_CLIENTS];

/* ── Team detection (QW): explicit userinfo "team" string ──────────────
 *   QW players set their team via the "team" userinfo key.  When the
 *   server's "teamplay" serverinfo is nonzero, identical team strings
 *   mean teammates.  This is the authoritative team signal — far more
 *   reliable than pants color (which collides randomly in non-team modes).
 *   In teamplay==0 modes there are no teams, so always return 0.
 *   No cache needed: userinfo is read live each call.
 */
float QNN_IsSameTeam(int entity_num)
{
	int self_slot, other_slot;
	const char *self_team;
	const char *other_team;

	/* Server runs teamplay only when the serverinfo "teamplay" key is
	 * a positive integer.  In FFA/duel it's 0 and we never have teams. */
	if (atoi(Info_ValueForKey(cl.serverinfo, "teamplay")) <= 0)
		return 0.0f;

	self_slot = cl.playernum;
	other_slot = entity_num - 1;
	if (self_slot < 0 || self_slot >= MAX_CLIENTS)
		return 0.0f;
	if (other_slot < 0 || other_slot >= MAX_CLIENTS)
		return 0.0f;
	if (other_slot == self_slot)
		return 0.0f;

	self_team = Info_ValueForKey(cl.players[self_slot].userinfo, "team");
	other_team = Info_ValueForKey(cl.players[other_slot].userinfo, "team");
	if (self_team[0] == '\0' || other_team[0] == '\0')
		return 0.0f;
	return (strcmp(self_team, other_team) == 0) ? 1.0f : 0.0f;
}

void QNN_PlayersResetTeamCache(void)
{
	/* QW reads userinfo live; nothing to reset. */
}

static player_state_t *QNN_QWPlayerStateForSlot(int slot)
{
	if (slot < 0 || slot >= MAX_CLIENTS)
		return NULL;

	if (cls.mvdplayback)
	{
		player_state_t *state = &qnn_mvd_latest_playerstate[slot];
		return state->messagenum > 0 ? state : NULL;
	}

	if (cl.validsequence <= 0)
		return NULL;

	{
		frame_t *frame = &cl.frames[cl.parsecount & UPDATE_MASK];
		player_state_t *state = &frame->playerstate[slot];
		return state->messagenum == cl.parsecount ? state : NULL;
	}
}

qboolean QNN_IsLivePlayerSlot(int entity_num)
{
	int slot = entity_num - 1;
	player_info_t *info;
	player_state_t *state;

	if (slot < 0 || slot >= MAX_CLIENTS)
		return false;
	if (slot == cl.playernum)
		return false;
	info = &cl.players[slot];
	if (info->name[0] == '\0')
		return false;
	if (info->spectator)
		return false;
	state = QNN_QWPlayerStateForSlot(slot);
	if (state == NULL)
		return false;
	if (!state->modelindex)
		return false;
	return true;
}

int QNN_AppendPlayerEntityUpdates(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int start_count, int max_entities)
{
	int slot;
	int count = start_count;

	for (slot = 0; slot < MAX_CLIENTS && count < max_entities; ++slot)
	{
		player_info_t *info = &cl.players[slot];
		player_state_t *state;
		int player_entnum = slot + 1;
		vec3_t anchor_origin;
		float half_extents[3];
		qnn_entity_update_t *eu;

		if (!QNN_IsLivePlayerSlot(player_entnum))
			continue;
		/* IsLivePlayerSlot guarantees state != NULL — re-query for the
		 * actual update.  Cheap (single array lookup); avoids passing
		 * state out of the predicate which would couple it to QW. */
		state = QNN_QWPlayerStateForSlot(slot);

		QNN_EntityAnchorFromModel(player_entnum, state->origin,
			anchor_origin, half_extents);

		/* MVD carries global server state.  Cull player tokens to the
		 * tracked client's PVS so BC does not learn omniscient enemies. */
		if (cls.mvdplayback && !QNN_EntityInPvs(snapshot->player_origin, anchor_origin))
			continue;

		eu = &out_entities[count];
		memset(eu, 0, sizeof(*eu));
		eu->entity_num = player_entnum;
		eu->subject_id = QNN_SUBJECT_PLAYER;
		VectorCopy(anchor_origin, eu->origin);
		VectorCopy(state->velocity, eu->velocity);
		eu->angles[0] = -state->viewangles[0] / 3.0f;
		eu->angles[1] = state->viewangles[1];
		eu->angles[2] = 0.0f;
		eu->effects = state->effects;
		eu->half_extents[0] = half_extents[0];
		eu->half_extents[1] = half_extents[1];
		eu->half_extents[2] = half_extents[2];
		eu->frags = (cl.scores != NULL) ? cl.scores[slot].frags : info->frags;
		eu->in_fov = QNN_InFov(snapshot->player_origin,
			snapshot->player_view_angles, anchor_origin);
		count++;
	}

	return count;
}
