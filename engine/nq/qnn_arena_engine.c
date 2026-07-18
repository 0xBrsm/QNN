#include "qnn.h"

#include <stdio.h>
#include <string.h>

extern dfunction_t *ED_FindFunction(char *name);
extern ddef_t *ED_FindField(char *name);

static int qnn_match_field_offset = -2;
static int qnn_seat_field_offset = -2;
static qboolean qnn_arena_selfplay;

typedef struct
{
	qnn_action_t action;
	vec3_t viewangles;
	int respawn_tick;
	qboolean staged;
	qboolean view_initialized;
} qnn_arena_action_state_t;

static qnn_arena_action_state_t qnn_arena_actions[16];

static int QNN_ArenaFieldOffset(const char *name, int *cache)
{
	if (*cache == -2)
	{
		ddef_t *def = ED_FindField((char *)name);
		*cache = def ? (int)def->ofs : -1;
	}
	return *cache;
}

static void QNN_ArenaSetField(edict_t *entity, const char *name, int *cache, float value)
{
	int offset = QNN_ArenaFieldOffset(name, cache);
	if (entity != NULL && offset >= 0)
		((eval_t *)((char *)&entity->v + offset * 4))->_float = value;
}

static qboolean QNN_ArenaCall(const char *name, edict_t *self_entity)
{
	dfunction_t *function;
	func_t old_self;
	edict_t *call_self;

	if (!sv.active || progs == NULL || pr_global_struct == NULL)
		return false;
	function = ED_FindFunction((char *)name);
	if (function == NULL)
		return false;
	call_self = self_entity != NULL ? self_entity : sv.edicts;
	old_self = pr_global_struct->self;
	pr_global_struct->self = EDICT_TO_PROG(call_self);
	PR_ExecuteProgram((func_t)(function - pr_functions));
	pr_global_struct->self = old_self;
	return true;
}

static qboolean QNN_ArenaClientSeat(
	client_t *client, int *match_id_out, int *seat_id_out)
{
	edict_t *player;
	const char *name;
	int match_id, seat_id;

	if (client == NULL || !client->active || !client->spawned
		|| client->edict == NULL)
		return false;
	player = client->edict;
	name = player->v.netname ? pr_strings + player->v.netname : "";
	if (sscanf(name, "qnn_%d_%d", &match_id, &seat_id) != 2)
		return false;
	if (match_id < 0 || match_id >= 8 || seat_id < 0 || seat_id > 1)
		return false;
	*match_id_out = match_id;
	*seat_id_out = seat_id;
	return true;
}

static int QNN_ArenaActionSlot(int match_id, int seat_id)
{
	return match_id * 2 + seat_id;
}

static float QNN_ArenaNetworkAngle(float angle)
{
	int encoded = ((int)angle * 256 / 360) & 255;

	/* MSG_ReadAngle consumes the byte through MSG_ReadChar, so the server sees
	   the upper half of the byte range as negative angles. */
	if (encoded >= 128)
		encoded -= 256;
	return (float)encoded * (360.0f / 256.0f);
}

static void QNN_ArenaSyncClientView(client_t *client)
{
	qnn_arena_action_state_t *state;
	int match_id, seat_id, axis;

	if (!QNN_ArenaClientSeat(client, &match_id, &seat_id))
		return;
	state = &qnn_arena_actions[QNN_ArenaActionSlot(match_id, seat_id)];
	/* svc_setangle is encoded from edict.angles, not edict.v_angle.  Seed the
	   server's unquantized accumulator from the exact angle the observer sees. */
	for (axis = 0; axis < 3; ++axis)
		state->viewangles[axis] = QNN_ArenaNetworkAngle(client->edict->v.angles[axis]);
	state->view_initialized = true;
}

static void QNN_ArenaSyncMatchViews(int match_id)
{
	int client_index;

	for (client_index = 0; client_index < svs.maxclients; ++client_index)
	{
		client_t *client = &svs.clients[client_index];
		int client_match_id, seat_id;
		if (QNN_ArenaClientSeat(client, &client_match_id, &seat_id)
			&& client_match_id == match_id)
			QNN_ArenaSyncClientView(client);
	}
}

void QNN_ArenaConfigureActionSeats(qboolean selfplay)
{
	qnn_arena_selfplay = selfplay ? true : false;
	memset(qnn_arena_actions, 0, sizeof(qnn_arena_actions));
}

qboolean QNN_ArenaStageActions(const qnn_action_t *actions, int action_count)
{
	int action_index;

	if (actions == NULL || action_count < 1 || action_count > 16)
		return false;
	for (action_index = 0; action_index < 16; ++action_index)
		qnn_arena_actions[action_index].staged = false;
	for (action_index = 0; action_index < action_count; ++action_index)
	{
		int slot = qnn_arena_selfplay ? action_index : action_index * 2;
		if (slot < 0 || slot >= 16)
			return false;
		qnn_arena_actions[slot].action = actions[action_index];
		qnn_arena_actions[slot].staged = true;
	}
	return true;
}

void QNN_ArenaApplyStagedClient(client_t *client)
{
	qnn_arena_action_state_t *state;
	qnn_action_t *action;
	edict_t *player;
	int match_id, seat_id, slot, axis;
	int held_weapon;

	if (!QNN_ArenaClientSeat(client, &match_id, &seat_id))
		return;
	if (!qnn_arena_selfplay && seat_id != 0)
		return;
	slot = QNN_ArenaActionSlot(match_id, seat_id);
	state = &qnn_arena_actions[slot];
	if (!state->staged)
		return;
	action = &state->action;
	player = client->edict;

	if (!state->view_initialized)
	{
		for (axis = 0; axis < 3; ++axis)
			state->viewangles[axis] = QNN_ArenaNetworkAngle(player->v.v_angle[axis]);
		state->view_initialized = true;
	}
	QNN_ApplyActionLook(action, state->viewangles);
	for (axis = 0; axis < 3; ++axis)
		player->v.v_angle[axis] = QNN_ArenaNetworkAngle(state->viewangles[axis]);

	client->cmd.forwardmove = (float)QNN_ActionAxisSign(action->move, 0)
		* QNN_SV_MAXSPEED;
	client->cmd.sidemove = (float)QNN_ActionAxisSign(action->move, 1)
		* QNN_SV_MAXSPEED;
	client->cmd.upmove = 0.0f;

	if (player->v.health <= 0.0f)
	{
		state->respawn_tick += 1;
		player->v.button0 = (state->respawn_tick & 1) ? 1.0f : 0.0f;
		player->v.button2 = (state->respawn_tick & 1) ? 1.0f : 0.0f;
	}
	else
	{
		player->v.button0 = QNN_ActionAttack(action->move) ? 1.0f : 0.0f;
		player->v.button2 = QNN_ActionAxisSign(action->move, 2) > 0 ? 1.0f : 0.0f;
	}

	held_weapon = QNN_ImpulseFromItemFlag((int)player->v.weapon);
	if (action->weapon > 0 && action->weapon != held_weapon)
		player->v.impulse = (float)action->weapon;
	state->staged = false;
}

int QNN_ArenaPendingActionCount(void)
{
	int slot;
	int count = 0;

	for (slot = 0; slot < 16; ++slot)
		if (qnn_arena_actions[slot].staged)
			count += 1;
	return count;
}

void QNN_ArenaProcessPending(void)
{
	int client_index;

	QNN_ArenaCall("qnn_arena_process_resets", sv.edicts);
	/* PutClientInServer raises fixangle for every player it respawns.  This is
	   the point after QuakeC resets and before SV_SendClientMessages consumes
	   the flag, so capture the exact view sent to each observer here. */
	for (client_index = 0; client_index < svs.maxclients; ++client_index)
	{
		client_t *client = &svs.clients[client_index];
		int match_id, seat_id;
		if (QNN_ArenaClientSeat(client, &match_id, &seat_id)
			&& client->edict->v.fixangle)
			QNN_ArenaSyncClientView(client);
	}
}

qboolean QNN_ArenaResetMatch(int match_id)
{
	dfunction_t *function;
	func_t old_self;
	float old_parm0;

	if (!sv.active || progs == NULL || pr_global_struct == NULL)
		return false;
	function = ED_FindFunction((char *)"qnn_arena_reset_match");
	if (function == NULL)
		return false;
	old_self = pr_global_struct->self;
	old_parm0 = G_FLOAT(OFS_PARM0);
	pr_global_struct->self = EDICT_TO_PROG(sv.edicts);
	G_FLOAT(OFS_PARM0) = (float)match_id;
	PR_ExecuteProgram((func_t)(function - pr_functions));
	G_FLOAT(OFS_PARM0) = old_parm0;
	pr_global_struct->self = old_self;
	if (G_FLOAT(OFS_RETURN) != 0.0f)
	{
		QNN_ArenaSyncMatchViews(match_id);
		return true;
	}
	return false;
}

qboolean QNN_ArenaAddBot(float skill)
{
	dfunction_t *function;
	func_t old_self;
	float old_parm0, old_parm1, old_parm2;

	if (!sv.active || progs == NULL || pr_global_struct == NULL)
		return false;
	function = ED_FindFunction((char *)"BotConnect");
	if (function == NULL)
		return false;
	old_self = pr_global_struct->self;
	old_parm0 = G_FLOAT(OFS_PARM0);
	old_parm1 = G_FLOAT(OFS_PARM1);
	old_parm2 = G_FLOAT(OFS_PARM2);
	pr_global_struct->self = EDICT_TO_PROG(sv.edicts);
	G_FLOAT(OFS_PARM0) = 0.0f;
	G_FLOAT(OFS_PARM1) = 0.0f;
	G_FLOAT(OFS_PARM2) = skill;
	PR_ExecuteProgram((func_t)(function - pr_functions));
	G_FLOAT(OFS_PARM0) = old_parm0;
	G_FLOAT(OFS_PARM1) = old_parm1;
	G_FLOAT(OFS_PARM2) = old_parm2;
	pr_global_struct->self = old_self;
	return true;
}

int QNN_ArenaAssignNamedSeats(void)
{
	int client_index;
	int assigned = 0;

	for (client_index = 0; client_index < svs.maxclients; ++client_index)
	{
		client_t *client = &svs.clients[client_index];
		int match_id, seat_id;

		/* A connection has an edict and netname before the prespawn/spawn/begin
		   exchange finishes.  Do not freeze the server command loop at that
		   halfway point: the remote observer still needs server frames to
		   complete sign-on and receive the arena-ready training message. */
		if (!QNN_ArenaClientSeat(client, &match_id, &seat_id))
			continue;
		QNN_ArenaSetField(client->edict, "qnn_match_id", &qnn_match_field_offset, (float)match_id);
		QNN_ArenaSetField(client->edict, "qnn_seat_id", &qnn_seat_field_offset, (float)seat_id);
		assigned += 1;
	}
	return assigned;
}
