#include "qnn.h"

#include <stdio.h>
#include <string.h>

extern dfunction_t *ED_FindFunction(char *name);
extern ddef_t *ED_FindField(char *name);

static int qnn_match_field_offset = -2;
static int qnn_seat_field_offset = -2;

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

void QNN_ArenaProcessPending(void)
{
	QNN_ArenaCall("qnn_arena_process_resets", sv.edicts);
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
	return G_FLOAT(OFS_RETURN) != 0.0f ? true : false;
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
		edict_t *player;
		const char *name;
		int match_id, seat_id;

		/* A connection has an edict and netname before the prespawn/spawn/begin
		   exchange finishes.  Do not freeze the server command loop at that
		   halfway point: the remote observer still needs server frames to
		   complete sign-on and receive the arena-ready training message. */
		if (!client->active || !client->spawned || client->edict == NULL)
			continue;
		player = client->edict;
		name = player->v.netname ? pr_strings + player->v.netname : "";
		if (sscanf(name, "qnn_%d_%d", &match_id, &seat_id) != 2)
			continue;
		if (match_id < 0 || match_id >= 8 || seat_id < 0 || seat_id > 1)
			continue;
		QNN_ArenaSetField(player, "qnn_match_id", &qnn_match_field_offset, (float)match_id);
		QNN_ArenaSetField(player, "qnn_seat_id", &qnn_seat_field_offset, (float)seat_id);
		assigned += 1;
	}
	return assigned;
}
