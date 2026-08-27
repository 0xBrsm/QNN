#include "qnn_arena_virtual.h"

#include "qnn_arena_observer.h"
#include "qnn_collect_helpers.h"
#include "qnn_context.h"
#include "qnn_obs_registry.h"
#include "qnn_predict.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QNN_VIRTUAL_MAX_CLIENTS 16
#define QNN_VIRTUAL_QUEUE_DEPTH 128

extern double oldrealtime;
extern int cl_numvisedicts;
extern entity_t *cl_visedicts[MAX_VISEDICTS];

typedef struct
{
	int type;
	int size;
	byte data[NET_MAXMESSAGE];
} qnn_virtual_message_t;

typedef struct
{
	qnn_context_t state;
	client_t *server_client;
	qnn_action_t previous_action;
	qnn_virtual_message_t queue[QNN_VIRTUAL_QUEUE_DEPTH];
	int queue_head;
	int queue_count;
	int match_id;
	int seat_id;
	int action_index;
	int tick;
	int steps;
	qboolean shadow;
	qboolean attached;
	qboolean prepared;
	char name[32];
} qnn_virtual_client_t;

static qnn_context_t qnn_server_context;
static qnn_virtual_client_t qnn_virtual_clients[QNN_VIRTUAL_MAX_CLIENTS];
static qnn_virtual_client_t *qnn_active_virtual_client;
static int qnn_virtual_client_count;
static qboolean qnn_virtual_selfplay;
static qboolean qnn_virtual_shadow;
static qboolean qnn_virtual_registered;
static qboolean qnn_virtual_frames_emitted;

/* Per-seat compiled obs plans (OP_ATTACH_DECL, WS2), keyed by ACTION
 * index — the fixed position python's driver reads each seat's frames
 * at, established before name-based seat assignment finishes.  No
 * attach = NULL = the default plan (864-byte legacy frame). */
static qnn_obs_plan_t qnn_virtual_seat_plans[QNN_VIRTUAL_MAX_CLIENTS];
static qboolean qnn_virtual_seat_has_plan[QNN_VIRTUAL_MAX_CLIENTS];

qboolean QNN_ArenaVirtualAttachSeatPlan(int seat_index,
	const struct qnn_obs_plan_s *plan, char *error, size_t error_size)
{
	if (qnn_virtual_client_count == 0)
	{
		snprintf(error, error_size,
			"arena runs external observers (no virtual/shadow seats) — "
			"server-side obs declarations have no seat to bind");
		return false;
	}
	if (seat_index < 0 || seat_index >= qnn_virtual_client_count)
	{
		snprintf(error, error_size,
			"obs declaration seat %d out of range (0..%d)",
			seat_index, qnn_virtual_client_count - 1);
		return false;
	}
	if (qnn_virtual_frames_emitted)
	{
		snprintf(error, error_size,
			"obs declaration for seat %d arrived after the first frame "
			"was emitted — attach before the arena is ready", seat_index);
		return false;
	}
	if (qnn_virtual_seat_has_plan[seat_index])
	{
		snprintf(error, error_size,
			"obs declaration for seat %d was already attached", seat_index);
		return false;
	}
	qnn_virtual_seat_plans[seat_index] = *plan;
	qnn_virtual_seat_has_plan[seat_index] = true;
	return true;
}

static const qnn_obs_plan_t *QNN_ArenaVirtualSeatPlan(int action_index)
{
	if (action_index >= 0 && action_index < QNN_VIRTUAL_MAX_CLIENTS
		&& qnn_virtual_seat_has_plan[action_index])
		return &qnn_virtual_seat_plans[action_index];
	return NULL;
}

static void QNN_ArenaRegisterObserverState(void)
{
	if (qnn_virtual_registered)
		return;
	QNN_ContextRegister(&cls, sizeof(cls));
	QNN_ContextRegister(&cl, sizeof(cl));
	QNN_ContextRegister(cl_efrags, sizeof(cl_efrags));
	QNN_ContextRegister(cl_entities, sizeof(cl_entities));
	QNN_ContextRegister(cl_static_entities, sizeof(cl_static_entities));
	QNN_ContextRegister(cl_lightstyle, sizeof(cl_lightstyle));
	QNN_ContextRegister(cl_dlights, sizeof(cl_dlights));
	QNN_ContextRegister(cl_temp_entities, sizeof(cl_temp_entities));
	QNN_ContextRegister(cl_beams, sizeof(cl_beams));
	QNN_ContextRegister(&cl_numvisedicts, sizeof(cl_numvisedicts));
	QNN_ContextRegister(cl_visedicts, sizeof(cl_visedicts));
	QNN_ContextRegister(cls.message.data, (size_t)cls.message.maxsize);
	QNN_ContextRegister(&host_frametime, sizeof(host_frametime));
	QNN_ContextRegister(&host_time, sizeof(host_time));
	QNN_ContextRegister(&realtime, sizeof(realtime));
	QNN_ContextRegister(&oldrealtime, sizeof(oldrealtime));
	QNN_ContextRegister(&host_framecount, sizeof(host_framecount));
	QNN_ContextRegister(&qnn_pending_action, sizeof(qnn_pending_action));
	QNN_ContextRegister(&qnn_runtime, sizeof(qnn_runtime));
	QNN_StoreRegisterContext();
	QNN_EventRegisterContext();
	QNN_SoundRegisterContext();
	QNN_PlayersRegisterContext();
	QNN_PredictRegisterContext();
	QNN_TrainingRegisterContext();
	qnn_virtual_registered = true;
}

static void QNN_ArenaActivate(qnn_virtual_client_t *client)
{
	if (qnn_active_virtual_client != NULL)
		Sys_Error("Nested arena virtual-client activation");
	QNN_ContextCapture(&qnn_server_context);
	QNN_ContextRestore(&client->state);
	qnn_active_virtual_client = client;
	qnn_training_client_context = true;
}

static void QNN_ArenaDeactivate(void)
{
	qnn_virtual_client_t *client = qnn_active_virtual_client;
	if (client == NULL)
		Sys_Error("Arena virtual-client state is not active");
	QNN_ContextCapture(&client->state);
	QNN_ContextRestore(&qnn_server_context);
	qnn_active_virtual_client = NULL;
	qnn_training_client_context = false;
}

static void QNN_ArenaInitClientState(qnn_virtual_client_t *client,
	const char *reward_json)
{
	byte *message_data = cls.message.data;
	int message_maxsize = cls.message.maxsize;

	memset(&cls, 0, sizeof(cls));
	cls.state = client->shadow ? ca_connected : ca_disconnected;
	cls.demonum = -1;
	cls.message.data = message_data;
	cls.message.maxsize = message_maxsize;
	SZ_Clear(&cls.message);
	CL_ClearState();
	memset(&qnn_runtime, 0, sizeof(qnn_runtime));
	QNN_ClearAction(&qnn_pending_action);
	QNN_TrainingResetEpisode();
	QNN_PredictReset();
	QNN_PlayersResetTeamCache();
	S_StopAllSounds(true);
	if (reward_json != NULL && reward_json[0] != 0)
		QNN_TrainingParseRewardWeights(reward_json);
	if (!client->shadow)
		CL_EstablishConnection("local");
}

void QNN_ArenaVirtualConfigure(int client_count, qboolean selfplay,
	qboolean shadow, const char *reward_json)
{
	int index;

	if (client_count < 1 || client_count > QNN_VIRTUAL_MAX_CLIENTS)
		Sys_Error("Invalid arena virtual-client count %d", client_count);
	/* Dedicated Host_Init skips the client and sound initializers.  Virtual
	 * observers use those parsers without enabling rendering, so initialize
	 * their message buffer, commands, and sound-name table explicitly. */
	if (cls.message.data == NULL)
	{
		S_Init();
		CL_Init();
		R_InitParticles();
	}
	QNN_PredictInit();
	QNN_ArenaRegisterObserverState();
	QNN_ContextInit(&qnn_server_context);
	qnn_virtual_client_count = client_count;
	qnn_virtual_selfplay = selfplay;
	qnn_virtual_shadow = shadow;
	qnn_virtual_frames_emitted = false;
	memset(qnn_virtual_seat_has_plan, 0, sizeof(qnn_virtual_seat_has_plan));

	for (index = 0; index < client_count; ++index)
	{
		qnn_virtual_client_t *client = &qnn_virtual_clients[index];
		memset(client, 0, sizeof(*client));
		client->shadow = shadow;
		client->match_id = selfplay ? index / 2 : index;
		client->seat_id = selfplay ? index % 2 : 0;
		client->action_index = index;
		snprintf(client->name, sizeof(client->name), "qnn_%d_%d",
			client->match_id, client->seat_id);
		QNN_ContextRestore(&qnn_server_context);
		QNN_ArenaInitClientState(client, reward_json);
		QNN_ContextInit(&client->state);
		QNN_ContextRestore(&qnn_server_context);
	}
}

void QNN_ArenaVirtualAttach(client_t *server_client)
{
	int index;

	if (!qnn_virtual_shadow || server_client == NULL)
		return;
	for (index = 0; index < qnn_virtual_client_count; ++index)
	{
		qnn_virtual_client_t *client = &qnn_virtual_clients[index];
		if (client->attached)
			continue;
		client->server_client = server_client;
		client->attached = true;
		return;
	}
	Sys_Error("More arena shadow clients connected than configured");
}

static qnn_virtual_client_t *QNN_ArenaClientForServer(client_t *server_client)
{
	int index;
	for (index = 0; index < qnn_virtual_client_count; ++index)
		if (qnn_virtual_clients[index].server_client == server_client)
			return &qnn_virtual_clients[index];
	return NULL;
}

void QNN_ArenaVirtualMirrorMessage(client_t *server_client,
	const sizebuf_t *message, int message_type)
{
	qnn_virtual_client_t *client;
	qnn_virtual_message_t *slot;
	int tail;

	if (!qnn_virtual_shadow || message == NULL || message->cursize <= 0)
		return;
	client = QNN_ArenaClientForServer(server_client);
	if (client == NULL)
		return;
	if (message->cursize > NET_MAXMESSAGE
		|| client->queue_count >= QNN_VIRTUAL_QUEUE_DEPTH)
		Sys_Error("Arena shadow message queue overflow");
	tail = (client->queue_head + client->queue_count) % QNN_VIRTUAL_QUEUE_DEPTH;
	slot = &client->queue[tail];
	slot->type = message_type;
	slot->size = message->cursize;
	memcpy(slot->data, message->data, (size_t)message->cursize);
	client->queue_count += 1;
}

qboolean QNN_ArenaVirtualMirrorActive(void)
{
	return qnn_active_virtual_client != NULL
		&& qnn_active_virtual_client->shadow;
}

int QNN_ArenaVirtualGetMessage(void)
{
	qnn_virtual_client_t *client = qnn_active_virtual_client;
	qnn_virtual_message_t *slot;

	if (client == NULL || !client->shadow || client->queue_count == 0)
		return 0;
	slot = &client->queue[client->queue_head];
	SZ_Clear(&net_message);
	SZ_Write(&net_message, slot->data, slot->size);
	client->queue_head = (client->queue_head + 1) % QNN_VIRTUAL_QUEUE_DEPTH;
	client->queue_count -= 1;
	return slot->type;
}

static void QNN_ArenaSetClientName(qnn_virtual_client_t *client)
{
	Cvar_Set("_cl_name", client->name);
}

void QNN_ArenaVirtualPumpSignon(float dt)
{
	int index;
	for (index = 0; index < qnn_virtual_client_count; ++index)
	{
		qnn_virtual_client_t *client = &qnn_virtual_clients[index];
		if (client->shadow && !client->attached)
			continue;
		QNN_ArenaActivate(client);
		QNN_ArenaSetClientName(client);
		/* The external observer parks as soon as it parses the first ready
		 * snapshot.  Preserve that exact comparison point in shadow mode; later
		 * datagrams remain queued and both parsers drain them on the first step. */
		if (!client->shadow || !QNN_TrainingNetworkArenaReady())
		{
			host_frametime = dt;
			realtime += dt;
			host_time += dt;
			QNN_TrainingResetTick();
			CL_ReadFromServer();
		}
		if (client->shadow)
			SZ_Clear(&cls.message);
		else if (cls.signon < SIGNONS || cls.message.cursize > 0)
			CL_SendCmd();
		QNN_ArenaDeactivate();
	}
}

int QNN_ArenaVirtualAssignSeats(void)
{
	int index;
	int assigned = 0;

	for (index = 0; index < qnn_virtual_client_count; ++index)
	{
		qnn_virtual_client_t *client = &qnn_virtual_clients[index];
		client_t *server_client = client->server_client;
		int match_id, seat_id;
		const char *name;

		if (!client->shadow && server_client == NULL)
		{
			int server_index;
			for (server_index = 0; server_index < svs.maxclients; ++server_index)
			{
				client_t *candidate = &svs.clients[server_index];
				if (!candidate->active || candidate->edict == NULL)
					continue;
				name = candidate->edict->v.netname
					? pr_strings + candidate->edict->v.netname : "";
				if (!strcmp(name, client->name))
				{
					client->server_client = candidate;
					client->attached = true;
					server_client = candidate;
					break;
				}
			}
		}
		if (server_client == NULL || server_client->edict == NULL)
			continue;
		name = server_client->edict->v.netname
			? pr_strings + server_client->edict->v.netname : "";
		if (sscanf(name, "qnn_%d_%d", &match_id, &seat_id) != 2)
			continue;
		client->match_id = match_id;
		client->seat_id = seat_id;
		client->action_index = qnn_virtual_selfplay
			? match_id * 2 + seat_id : match_id;
		assigned += 1;
	}
	return assigned;
}

qboolean QNN_ArenaVirtualReady(void)
{
	int index;
	if (QNN_ArenaVirtualAssignSeats() != qnn_virtual_client_count)
		return false;
	for (index = 0; index < qnn_virtual_client_count; ++index)
	{
		qboolean ready;
		QNN_ArenaActivate(&qnn_virtual_clients[index]);
		ready = QNN_ArenaObserverReady();
		QNN_ArenaDeactivate();
		if (!ready)
			return false;
	}
	return true;
}

void QNN_ArenaVirtualPrepare(void)
{
	int index;
	for (index = 0; index < qnn_virtual_client_count; ++index)
	{
		qnn_virtual_client_t *client = &qnn_virtual_clients[index];
		QNN_ArenaActivate(client);
		if (!QNN_ArenaObserverReady() || !QNN_TrainingNetworkArenaReady())
			Sys_Error("Arena virtual client %s is not ready", client->name);
		QNN_ArenaObserverPrepare();
		client->prepared = true;
		QNN_ArenaDeactivate();
	}
}

static qnn_virtual_client_t *QNN_ArenaClientForAction(int action_index)
{
	int index;
	for (index = 0; index < qnn_virtual_client_count; ++index)
		if (qnn_virtual_clients[index].action_index == action_index)
			return &qnn_virtual_clients[index];
	return NULL;
}

void QNN_ArenaVirtualStageActions(const qnn_action_t *actions, int action_count)
{
	int action_index;
	if (actions == NULL || action_count != qnn_virtual_client_count)
		Sys_Error("Invalid arena virtual action batch");
	for (action_index = 0; action_index < action_count; ++action_index)
	{
		qnn_virtual_client_t *client = QNN_ArenaClientForAction(action_index);
		if (client == NULL)
			Sys_Error("Arena virtual action %d has no seat", action_index);
		QNN_ArenaActivate(client);
		QNN_ArenaApplyLocalAction(&actions[action_index]);
		client->previous_action = actions[action_index];
		if (!client->shadow)
		{
			sizebuf_t message;
			byte data[1];
			message.data = data;
			message.maxsize = sizeof(data);
			message.cursize = 0;
			MSG_WriteByte(&message, clc_nop);
			if (NET_SendUnreliableMessage(cls.netcon, &message) == -1)
				Sys_Error("Arena virtual client lost its loopback connection");
		}
		QNN_ArenaDeactivate();
	}
}

void QNN_ArenaVirtualWriteInitial(FILE *out, float dt)
{
	int action_index;

	/* Frames are flowing at negotiated sizes from here on — a late
	 * OP_ATTACH_DECL would desync the driver's reads, so refuse it
	 * (see QNN_ArenaVirtualAttachSeatPlan). */
	qnn_virtual_frames_emitted = true;
	for (action_index = 0; action_index < qnn_virtual_client_count; ++action_index)
	{
		qnn_virtual_client_t *client = QNN_ArenaClientForAction(action_index);
		QNN_ArenaActivate(client);
		QNN_ArenaObserverWrite(out, QNN_ArenaVirtualSeatPlan(action_index),
			&client->previous_action,
			client->tick, client->steps, true, dt);
		QNN_ArenaDeactivate();
	}
}

void QNN_ArenaVirtualReceive(FILE *out, float dt, qboolean reset_receive)
{
	int action_index;
	for (action_index = 0; action_index < qnn_virtual_client_count; ++action_index)
	{
		qnn_virtual_client_t *client = QNN_ArenaClientForAction(action_index);
		double previous_server_time;
		qboolean reset_flag;

		QNN_ArenaActivate(client);
		previous_server_time = cl.mtime[0];
		host_frametime = reset_receive ? 0.0 : dt;
		if (!reset_receive)
		{
			realtime += dt;
			host_time += dt;
		}
		QNN_TrainingResetTick();
		CL_ReadFromServer();
		if (!reset_receive && cl.mtime[0] == previous_server_time)
			Sys_Error("Arena virtual client %s received no server frame", client->name);
		if (!reset_receive)
		{
			client->tick += 1;
			client->steps += 1;
		}
		reset_flag = QNN_TrainingNetworkRoundReset();
		QNN_ArenaObserverWrite(out, QNN_ArenaVirtualSeatPlan(action_index),
			&client->previous_action,
			client->tick, client->steps, reset_flag, dt);
		if (reset_flag)
			client->steps = 0;
		QNN_ArenaDeactivate();
	}
}

void QNN_ArenaVirtualShutdown(void)
{
	int index;
	for (index = 0; index < qnn_virtual_client_count; ++index)
		QNN_ContextDestroy(&qnn_virtual_clients[index].state);
	QNN_ContextDestroy(&qnn_server_context);
	qnn_virtual_client_count = 0;
}
