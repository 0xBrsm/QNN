#include "qnn.h"
#include "qnn_collect_helpers.h"
#include "qnn_fault.h"
#include "qnn_io.h"
#include "qnn_predict.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QNN_ARENA_CLIENT_OP_STAGE 1
#define QNN_ARENA_CLIENT_OP_RECEIVE 4
#define QNN_ARENA_CLIENT_OP_RESUME_SIGNON 5
#define QNN_ARENA_CLIENT_OP_RECEIVE_RESET 6
#define QNN_ARENA_CLIENT_OP_SHUTDOWN 255

static float qnn_arena_dt = 1.0f / 20.0f;
static char qnn_arena_map_id[QNN_MAX_MAP_ID];
static int qnn_arena_tick;
static int qnn_arena_steps;

/* Minimal replacement for stock CL_RelinkEntities in the arena observer.
 * Parsing already filled msg_origins/msg_angles; PPO needs current transforms
 * and self velocity, not renderer efrags, trails, particles, dynamic lights,
 * interpolation, or temporary-entity draw lists. */
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

static const char *QNN_ArgString(const char *name, const char *fallback)
{
	int index = COM_CheckParm((char *)name);
	if (index > 0 && index + 1 < com_argc)
		return com_argv[index + 1];
	return fallback;
}

static qboolean QNN_ArenaClientReady(void)
{
	return cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS;
}

static void QNN_DeriveMapId(char *out, size_t out_size)
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

static void QNN_PrepareClientObservation(void)
{
	char error[256];

	QNN_DeriveMapId(qnn_arena_map_id, sizeof(qnn_arena_map_id));
	if (qnn_arena_map_id[0] == 0
		|| !QNN_PrepareMap(qnn_arena_map_id, error, sizeof(error)))
		Sys_Error("Arena client map preparation failed: %s", error);
	QNN_IOInit(&qnn_map_state);
}

static void QNN_WriteObservation(const qnn_action_t *previous_action, qboolean reset_flag)
{
	static qnn_snapshot_t snapshot;
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];
	qnn_tick_result_t result;

	QNN_CaptureBaseSnapshot(&snapshot);
	if (reset_flag)
		QNN_PredictReset();
	QNN_PredictTick(qnn_arena_dt);
	QNN_PredictSelfVelocity(snapshot.player_velocity);
	snapshot.action_label = *previous_action;
	snapshot.done = QNN_TrainingNetworkRoundReset();
	QNN_DrainSounds(&snapshot);
	QNN_IOUpdate(&snapshot, qnn_arena_dt, reset_flag);
	QNN_IOEmit(&snapshot, &result);
	QNN_IOPackObsBuffer(obs, &result);
	fwrite(obs, 1, sizeof(obs), stdout);
	QNN_WriteTrainingExtrasBinary(
		stdout, &snapshot, qnn_arena_tick, qnn_arena_steps, reset_flag);
	fflush(stdout);
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
		if (QNN_ArenaClientReady() && !announced_signon)
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
			   handshakes. */
			if (fgetc(stdin) != QNN_ARENA_CLIENT_OP_RESUME_SIGNON)
				Sys_Error("Arena client sign-on barrier was not resumed");
		}
		if (QNN_ArenaClientReady() && QNN_TrainingNetworkArenaReady())
			break;
		usleep(1000);
	}
	if (!QNN_ArenaClientReady() || !QNN_TrainingNetworkArenaReady())
		Sys_Error("Arena client timed out during sign-on");
	QNN_PrepareClientObservation();
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
