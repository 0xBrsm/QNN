#include "qnn.h"
#include "qnn_collect_helpers.h"
#include "qnn_fault.h"
#include "qnn_io.h"
#include "qnn_obs_shim.h"
#include "qnn_predict.h"
#include "qnn_tick.h"
#include "qnn_arena_observer.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QNN_ARENA_CLIENT_OP_STAGE 1
#define QNN_ARENA_CLIENT_OP_RECEIVE 4
#define QNN_ARENA_CLIENT_OP_RESUME_SIGNON 5
#define QNN_ARENA_CLIENT_OP_RECEIVE_RESET 6
#define QNN_ARENA_CLIENT_OP_STAGE_LOCAL 7
#define QNN_ARENA_CLIENT_OP_SHUTDOWN 255

static float qnn_arena_dt = 1.0f / 20.0f;
static int qnn_arena_tick;
static int qnn_arena_steps;

/* This seat's compiled obs plan (OP_ATTACH_DECL between "signed_on"
 * and OP_RESUME_SIGNON).  No attach = default plan — today's 864-byte
 * frame, bit-identical. */
static qnn_obs_plan_t qnn_arena_seat_plan;
static qboolean qnn_arena_seat_has_plan;

/* Consume + answer one OP_ATTACH_DECL (opcode byte already read).
 * Blocking stdio reads — the driver writes the whole request before
 * reading the reply.  Any validation failure replies the error frame,
 * then exits: an unattachable seat must never silently serve the
 * default plan. */
static void QNN_ArenaClientHandleAttach(void)
{
	uint8_t header[5];
	int seat_index;
	uint32_t json_len;
	char *json;
	char error[256];
	static char reply[4096];
	qnn_obs_decl_t decl;

	if (fread(header, 1, sizeof(header), stdin) != sizeof(header))
		Sys_Error("Arena client stdin closed mid-attach-decl header");
	seat_index = header[0];
	json_len = (uint32_t)header[1] | ((uint32_t)header[2] << 8)
		| ((uint32_t)header[3] << 16) | ((uint32_t)header[4] << 24);
	if (json_len == 0 || json_len > QNN_OBS_DECL_JSON_MAX)
		Sys_Error("Arena client attach-decl length %u out of range (1..%d)",
			json_len, QNN_OBS_DECL_JSON_MAX);
	json = malloc(json_len + 1);
	if (json == NULL)
		Sys_Error("Arena client attach-decl: out of memory (%u bytes)",
			json_len);
	if (fread(json, 1, json_len, stdin) != json_len)
		Sys_Error("Arena client stdin closed mid-attach-decl JSON");
	json[json_len] = 0;

	if (seat_index != 0)
	{
		snprintf(error, sizeof(error),
			"arena client is single-seat: seat_index must be 0 (got %d)",
			seat_index);
		goto reject;
	}
	if (qnn_arena_seat_has_plan)
	{
		snprintf(error, sizeof(error),
			"obs declaration was already attached for this seat");
		goto reject;
	}
	if (!QNN_ObsDeclParseJson(json, (int)json_len, &decl,
			error, sizeof(error))
		|| !QNN_ObsPlanCompile(&decl, &qnn_arena_seat_plan,
			error, sizeof(error)))
		goto reject;
	free(json);
	qnn_arena_seat_has_plan = true;

	if (!QNN_ObsLayoutReplyJson(&qnn_arena_seat_plan, reply, sizeof(reply),
			error, sizeof(error)))
		Sys_Error("Arena client attach-decl reply failed: %s", error);
	fprintf(stdout, "%s\n", reply);
	fflush(stdout);
	return;

reject:
	QNN_WriteError(error);
	Sys_Error("Arena client obs declaration rejected: %s", error);
}

static const char *QNN_ArgString(const char *name, const char *fallback)
{
	int index = COM_CheckParm((char *)name);
	if (index > 0 && index + 1 < com_argc)
		return com_argv[index + 1];
	return fallback;
}

static void QNN_WriteObservation(const qnn_action_t *previous_action, qboolean reset_flag)
{
	QNN_ArenaObserverWrite(stdout,
		qnn_arena_seat_has_plan ? &qnn_arena_seat_plan : NULL,
		previous_action, qnn_arena_tick, qnn_arena_steps, reset_flag);
}

static void QNN_ReceiveServerFrame(const qnn_action_t *previous_action)
{
	float previous_server_time = cl.mtime[0];

	host_frametime = qnn_arena_dt;
	realtime += qnn_arena_dt;
	host_time += qnn_arena_dt;
	QNN_TrainingResetTick();
	NET_Poll();
	CL_ReadFromServer();
	if (cl.mtime[0] == previous_server_time)
		Sys_Error("Arena client received no server frame after step acknowledgement");
	qnn_arena_tick += 1;
	qnn_arena_steps += 1;
	QNN_WriteObservation(previous_action, QNN_TrainingNetworkRoundReset());
	if (QNN_TrainingNetworkRoundReset())
		qnn_arena_steps = 0;
}

static void QNN_ReceiveResetState(const qnn_action_t *previous_action)
{
	host_frametime = 0.0f;
	QNN_TrainingResetTick();
	NET_Poll();
	CL_ReadFromServer();
	QNN_WriteObservation(previous_action, QNN_TrainingNetworkRoundReset());
	if (QNN_TrainingNetworkRoundReset())
		qnn_arena_steps = 0;
}

static void QNN_SendObserverNop(void)
{
	sizebuf_t message;
	byte data[1];

	message.data = data;
	message.maxsize = sizeof(data);
	message.cursize = 0;
	MSG_WriteByte(&message, clc_nop);
	/* NQ carries server reliable acknowledgements in every reverse-direction
	   packet.  Keep that sequencing traffic even though the action payload now
	   travels through the grouped server pipe. */
	if (NET_SendUnreliableMessage(cls.netcon, &message) == -1)
		Sys_Error("Arena observer lost server connection");
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	const char *server;
	const char *name;
	char command[256];
	qnn_action_t previous_action;
	int frame;
	double deadline;
	qboolean announced_signon = false;

	QNN_FaultInit("ppo_arena_client");
	QNN_PredictInit();
	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	QNN_ClearAction(&qnn_pending_action);
	QNN_ClearAction(&previous_action);
	QNN_TrainingResetEpisode();
	memset(&parms, 0, sizeof(parms));
	COM_InitArgv(argc, argv);
	parms.argc = com_argc;
	parms.argv = com_argv;
	parms.memsize = 32 * 1024 * 1024;
	parms.membase = malloc(parms.memsize);
	parms.basedir = basedir;
	server = QNN_ArgString("-qnn_server", "127.0.0.1:26000");
	name = QNN_ArgString("-qnn_name", "qnn_0_0");

	qnn_runtime.fixed_tick_hz = 20;
	Host_Init(&parms);
	QNN_TickRegister();
	QNN_RegisterPerceptionCvars();
	{
		const char *reward_json = getenv("QNN_REWARD_JSON");
		if (reward_json == NULL || reward_json[0] == 0)
			Sys_Error("QNN_REWARD_JSON is required for arena policy clients");
		QNN_TrainingParseRewardWeights(reward_json);
	}
	cls.demonum = -1;
	snprintf(command, sizeof(command), "name %s\nconnect %s\n", name, server);
	Cbuf_AddText(command);

	/* Sign on and continue pumping until the shared server has assigned every
	   named seat, added engine bots (bot mode), reset all matches, and raised
	   the arena-ready bit in the custom training message. */
	deadline = Sys_FloatTime() + 120.0;
	for (frame = 0; Sys_FloatTime() < deadline; ++frame)
	{
		QNN_TrainingResetTick();
		Host_Frame(qnn_arena_dt);
		if (QNN_ArenaObserverReady() && !announced_signon)
		{
			/* svc_signonnum 4 queues the final "begin" command.  A normal
			   client sends it on the next Host_Frame; flush it before parking so
			   signed_on also means client->spawned on the server. */
			CL_SendCmd();
			fprintf(stdout,
				"{\"ok\":true,\"state\":\"signed_on\",\"viewentity\":%d}\n",
				cl.viewentity);
			fflush(stdout);
			announced_signon = true;
			/* Park this fully spawned client while the remaining seats complete
			   stock NetQuake's serial sign-on.  Continuing Host_Frame here would
			   flood the server with idle commands and destabilize later reliable
			   handshakes.  The park doubles as the OP_ATTACH_DECL window: the
			   driver attaches this seat's obs declaration (if any) between
			   "signed_on" and OP_RESUME_SIGNON. */
			for (;;)
			{
				int barrier_op = fgetc(stdin);

				if (barrier_op == QNN_ARENA_CLIENT_OP_RESUME_SIGNON)
					break;
				if (barrier_op == QNN_OBS_OP_ATTACH_DECL)
				{
					QNN_ArenaClientHandleAttach();
					continue;
				}
				Sys_Error("Arena client sign-on barrier was not resumed "
					"(opcode %d)", barrier_op);
			}
		}
		if (QNN_ArenaObserverReady() && QNN_TrainingNetworkArenaReady())
			break;
		usleep(1000);
	}
	if (!QNN_ArenaObserverReady() || !QNN_TrainingNetworkArenaReady())
		Sys_Error("Arena client timed out during sign-on");
	QNN_ArenaObserverPrepare();
	fprintf(stdout, "{\"ok\":true,\"state\":\"ready\",\"viewentity\":%d}\n", cl.viewentity);
	fflush(stdout);
	QNN_WriteObservation(&previous_action, true);

	for (;;)
	{
		int opcode = fgetc(stdin);
		if (opcode == EOF || opcode == QNN_ARENA_CLIENT_OP_SHUTDOWN)
			break;
		if (opcode == QNN_ARENA_CLIENT_OP_STAGE)
		{
			if (fread(&previous_action, 1, sizeof(previous_action), stdin)
				!= sizeof(previous_action))
				break;
			qnn_pending_action = previous_action;
			host_frametime = qnn_arena_dt;
			CL_SendCmd();
			QNN_ClearAction(&qnn_pending_action);
			fputc(QNN_ARENA_CLIENT_OP_STAGE, stdout);
			fflush(stdout);
			continue;
		}
		if (opcode == QNN_ARENA_CLIENT_OP_STAGE_LOCAL)
		{
			if (fread(&previous_action, 1, sizeof(previous_action), stdin)
				!= sizeof(previous_action))
				break;
			host_frametime = qnn_arena_dt;
			QNN_ArenaApplyLocalAction(&previous_action);
			QNN_SendObserverNop();
			continue;
		}
		if (opcode == QNN_ARENA_CLIENT_OP_RECEIVE)
		{
			QNN_ReceiveServerFrame(&previous_action);
			continue;
		}
		if (opcode == QNN_ARENA_CLIENT_OP_RECEIVE_RESET)
		{
			QNN_ReceiveResetState(&previous_action);
			continue;
		}
		Sys_Error("Unknown arena client opcode %d", opcode);
	}

	CL_Disconnect();
	Host_Shutdown();
	return 0;
}
