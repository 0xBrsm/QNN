#include "qnn.h"
#include "qnn_fault.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QNN_ARENA_SERVER_OP_STEP 2
#define QNN_ARENA_SERVER_OP_RESET_MASK 3
#define QNN_ARENA_SERVER_OP_SHUTDOWN 255

static cvar_t qnn_train_cvar = {"qnn_train", "1", false, false};
static cvar_t qnn_arena_selfplay_cvar = {"qnn_arena_selfplay", "0", false, false};
static cvar_t qnn_arena_ready_cvar = {"qnn_arena_ready", "0", false, false};
static cvar_t qnn_inv_weapons_cvar = {"qnn_inv_weapons", "96", false, false};
static cvar_t qnn_inv_shells_cvar = {"qnn_inv_shells", "0", false, false};
static cvar_t qnn_inv_nails_cvar = {"qnn_inv_nails", "0", false, false};
static cvar_t qnn_inv_rockets_cvar = {"qnn_inv_rockets", "100", false, false};
static cvar_t qnn_inv_cells_cvar = {"qnn_inv_cells", "100", false, false};
static cvar_t qnn_inv_armor_cvar = {"qnn_inv_armor", "200", false, false};
static cvar_t qnn_inv_armor_type_cvar = {"qnn_inv_armor_type", "0.8", false, false};
static cvar_t qnn_inv_health_cvar = {"qnn_inv_health", "100", false, false};
static cvar_t qnn_inv_powerups_cvar = {"qnn_inv_powerups", "0", false, false};
static cvar_t qnn_inv_selected_cvar = {"qnn_inv_selected", "0", false, false};
static cvar_t qnn_bot_weapon_pin_cvar = {"qnn_bot_weapon_pin", "0", false, false};

static int QNN_ArgInt(const char *name, int fallback)
{
	int index = COM_CheckParm((char *)name);
	if (index > 0 && index + 1 < com_argc)
		return atoi(com_argv[index + 1]);
	return fallback;
}

static const char *QNN_ArgString(const char *name, const char *fallback)
{
	int index = COM_CheckParm((char *)name);
	if (index > 0 && index + 1 < com_argc)
		return com_argv[index + 1];
	return fallback;
}

static void QNN_RegisterArenaCvars(void)
{
	Cvar_RegisterVariable(&qnn_train_cvar);
	Cvar_RegisterVariable(&qnn_arena_selfplay_cvar);
	Cvar_RegisterVariable(&qnn_arena_ready_cvar);
	Cvar_RegisterVariable(&qnn_inv_weapons_cvar);
	Cvar_RegisterVariable(&qnn_inv_shells_cvar);
	Cvar_RegisterVariable(&qnn_inv_nails_cvar);
	Cvar_RegisterVariable(&qnn_inv_rockets_cvar);
	Cvar_RegisterVariable(&qnn_inv_cells_cvar);
	Cvar_RegisterVariable(&qnn_inv_armor_cvar);
	Cvar_RegisterVariable(&qnn_inv_armor_type_cvar);
	Cvar_RegisterVariable(&qnn_inv_health_cvar);
	Cvar_RegisterVariable(&qnn_inv_powerups_cvar);
	Cvar_RegisterVariable(&qnn_inv_selected_cvar);
	Cvar_RegisterVariable(&qnn_bot_weapon_pin_cvar);
}

static void QNN_ServerFrame(float dt)
{
	QNN_TrainingResetTick();
	Host_Frame(dt);
}

static void QNN_WriteState(const char *state, int port)
{
	fprintf(stdout, "{\"ok\":true,\"port\":%d,\"state\":\"%s\"}\n", port, state);
	fflush(stdout);
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	const char *map_id;
	int external_count;
	int bot_count;
	int selfplay;
	int bot_skill;
	int port;
	int frame;
	float dt = 1.0f / 20.0f;
	char command[256];
	double deadline;

	QNN_FaultInit("ppo_arena_server");
	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	memset(&parms, 0, sizeof(parms));
	COM_InitArgv(argc, argv);
	parms.argc = com_argc;
	parms.argv = com_argv;
	parms.memsize = 64 * 1024 * 1024;
	parms.membase = malloc(parms.memsize);
	parms.basedir = basedir;

	external_count = QNN_ArgInt("-qnn_external", 8);
	bot_count = QNN_ArgInt("-qnn_bots", 8);
	selfplay = QNN_ArgInt("-qnn_selfplay", 0) ? 1 : 0;
	bot_skill = QNN_ArgInt("-qnn_bot_skill", 3);
	port = QNN_ArgInt("-port", 26000);
	map_id = QNN_ArgString("-qnn_map", "qnn_arena8");

	Host_Init(&parms);
	QNN_TickRegister();
	QNN_RegisterArenaCvars();
	Cvar_SetValue("qnn_train", 1.0f);
	Cvar_SetValue("qnn_arena_selfplay", (float)selfplay);
	Cvar_SetValue("qnn_arena_ready", 0.0f);
	Cvar_SetValue("qnn_tick_hz", 20.0f);

	/* Drain quake.rc before appending the arena map command.  On a dedicated
	   server its trailing startdemos command appends "map start"; if both are
	   left in the buffer that command runs after qnn_arena8 and silently
	   replaces the arena while policy clients are signing on. */
	QNN_ServerFrame(dt);

	snprintf(command, sizeof(command),
		"deathmatch 1\ncoop 0\nteamplay 0\nfraglimit 0\ntimelimit 0\nmap %s\n",
		map_id);
	Cbuf_AddText(command);
	for (frame = 0; frame < 2048 && !sv.active; ++frame)
		QNN_ServerFrame(dt);
	if (!sv.active)
		Sys_Error("Timed out loading arena map %s", map_id);
	QNN_WriteState("listening", port);

	/* Clients perform normal NQ sign-on autonomously while the server pumps.
	   Their qnn_<match>_<seat> names make the final assignment independent of
	   UDP arrival order. */
	deadline = Sys_FloatTime() + 120.0;
	for (frame = 0; Sys_FloatTime() < deadline; ++frame)
	{
		QNN_ServerFrame(dt);
		if (QNN_ArenaAssignNamedSeats() >= external_count)
			break;
		/* Give the separately scheduled clients CPU time to complete the
		   original protocol's 1.5-second discovery phase. */
		usleep(1000);
	}
	if (QNN_ArenaAssignNamedSeats() < external_count)
		Sys_Error("Timed out waiting for %d external arena clients", external_count);

	for (frame = 0; frame < bot_count; ++frame)
		if (!QNN_ArenaAddBot((float)bot_skill))
			Sys_Error("Failed to add arena bot %d", frame);
	for (frame = 0; frame < 8; ++frame)
		QNN_ArenaResetMatch(frame);
	for (frame = 0; frame < 16; ++frame)
		QNN_ServerFrame(dt);
	Cvar_SetValue("qnn_arena_ready", 1.0f);
	/* A full self-play world resumes sixteen clients at once.  If the server
	   emits only one or two ready snapshots and immediately parks on stdin,
	   those tail datagrams can be dropped behind queued sign-on traffic and a
	   client waits forever for a frame that will never be retransmitted.  Pace
	   a short readiness window so every socket drains before steady stepping. */
	for (frame = 0; frame < 64; ++frame)
	{
		QNN_ServerFrame(dt);
		usleep(1000);
	}
	QNN_WriteState("ready", port);

	for (;;)
	{
		int opcode = fgetc(stdin);
		if (opcode == EOF || opcode == QNN_ARENA_SERVER_OP_SHUTDOWN)
			break;
		if (opcode == QNN_ARENA_SERVER_OP_STEP)
		{
			QNN_ServerFrame(dt);
			fputc(QNN_ARENA_SERVER_OP_STEP, stdout);
			fflush(stdout);
			continue;
		}
		if (opcode == QNN_ARENA_SERVER_OP_RESET_MASK)
		{
			int mask = fgetc(stdin);
			int match_id;
			if (mask == EOF)
				break;
			for (match_id = 0; match_id < 8; ++match_id)
				if (mask & (1 << match_id))
					QNN_ArenaResetMatch(match_id);
			/* A timeout reset is not an environment step.  Broadcast the new
			   entities directly without physics or advancing sv.time, and tag
			   only the selected matches so their clients reset prediction/GRU
			   state.  Other matches receive an identical state at the same time. */
			QNN_TrainingSetNetworkResetMask(mask);
			SV_SendClientMessages();
			QNN_TrainingSetNetworkResetMask(0);
			fputc(QNN_ARENA_SERVER_OP_RESET_MASK, stdout);
			fflush(stdout);
			continue;
		}
		Sys_Error("Unknown arena server opcode %d", opcode);
	}

	Host_Shutdown();
	return 0;
}
