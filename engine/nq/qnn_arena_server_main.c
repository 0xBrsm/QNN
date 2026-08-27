#include "qnn.h"
#include "qnn_fault.h"
#include "qnn_tick.h"
#include "qnn_arena_virtual.h"
#include "qnn_obs_shim.h"

#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QNN_ARENA_SERVER_OP_STEP 2
#define QNN_ARENA_SERVER_OP_RESET_MASK 3
#define QNN_ARENA_SERVER_OP_STEP_BATCH 7
#define QNN_ARENA_SERVER_OP_SHUTDOWN 255

/* Quiet-window drain for OP_ATTACH_DECL before "ready": the driver
 * sends its declarations right after reading "listening" and blocks on
 * each reply, so once sign-on completes any straggler arrives within a
 * python turnaround.  Each serviced request re-arms the window. */
#define QNN_ARENA_ATTACH_DRAIN_MS 500

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
/* Per-frame QuakeC ammo top-up (qnn_infinite_ammo_frame): pin waves set 1 so
   90s pinned episodes never run the measured weapon dry. Default off. */
static cvar_t qnn_inv_infinite_ammo_cvar = {"qnn_inv_infinite_ammo", "0", false, false};

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

/* Weapon name -> IT_ bitmask. Mirrors QNN_WeaponBit in qnn_trainer_main.c
   (the canonical copy); duplicated here because the arena server does not link
   the trainer main and the IT_ flags are a fixed Quake constant. */
static int QNN_ArenaWeaponBit(const char *name)
{
	if (!name)                             return 0;
	if (!strcmp(name, "axe"))              return 4096;
	if (!strcmp(name, "shotgun"))          return 1;
	if (!strcmp(name, "super_shotgun"))    return 2;
	if (!strcmp(name, "nailgun"))          return 4;
	if (!strcmp(name, "super_nailgun"))    return 8;
	if (!strcmp(name, "grenade_launcher")) return 16;
	if (!strcmp(name, "rocket_launcher"))  return 32;
	if (!strcmp(name, "lightning"))        return 64;
	Sys_Error("Unknown arena weapon name '%s'", name);
	return 0;
}

/* Apply the single-weapon arena inventory (model weapon + bot pin + ammo/armor/
   health) from CLI args onto the qnn_inv_ and qnn_bot_weapon_pin cvars BEFORE the
   map loads and matches spawn, so every match respawn reads the pinned loadout.
   Each arg is optional; absent args leave the registered default untouched. The
   Python side (ArenaServerProcess) only emits the args the scenario specifies. */
static void QNN_ArenaApplyInventory(void)
{
	const char *model_weapon = QNN_ArgString("-qnn_inv_selected", NULL);
	const char *bot_pin = QNN_ArgString("-qnn_bot_weapon_pin", NULL);
	const char *armor_type = QNN_ArgString("-qnn_inv_armor_type", NULL);
	struct { const char *arg; const char *cvar; } ints[] = {
		{"-qnn_inv_shells", "qnn_inv_shells"},
		{"-qnn_inv_nails", "qnn_inv_nails"},
		{"-qnn_inv_rockets", "qnn_inv_rockets"},
		{"-qnn_inv_cells", "qnn_inv_cells"},
		{"-qnn_inv_armor", "qnn_inv_armor"},
		{"-qnn_inv_health", "qnn_inv_health"},
		{"-qnn_inv_infinite_ammo", "qnn_inv_infinite_ammo"},
	};
	char buf[32];
	int i;

	if (model_weapon)
	{
		int bit = QNN_ArenaWeaponBit(model_weapon);
		snprintf(buf, sizeof(buf), "%d", bit);
		Cvar_Set("qnn_inv_selected", buf);
		/* Single-weapon arena: the owned mask is exactly the selected weapon. */
		Cvar_Set("qnn_inv_weapons", buf);
	}
	if (bot_pin)
	{
		snprintf(buf, sizeof(buf), "%d", QNN_ArenaWeaponBit(bot_pin));
		Cvar_Set("qnn_bot_weapon_pin", buf);
	}
	for (i = 0; i < (int)(sizeof(ints) / sizeof(ints[0])); ++i)
	{
		const char *value = QNN_ArgString(ints[i].arg, NULL);
		if (value)
			Cvar_Set((char *)ints[i].cvar, (char *)value);
	}
	if (armor_type)
		Cvar_Set("qnn_inv_armor_type", (char *)armor_type);
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
	Cvar_RegisterVariable(&qnn_inv_infinite_ammo_cvar);
}

static void QNN_ServerFrame(float dt)
{
	QNN_TrainingResetTick();
	Host_Frame(dt);
}

/* ── OP_ATTACH_DECL servicing (WS2, agents/plans/obs-api.md) ─────────
 *
 * The declaration handshake runs between "listening" and "ready":
 * request = u8 opcode (QNN_OBS_OP_ATTACH_DECL) | u8 seat_index |
 * u32le JSON length | declaration JSON; reply = one JSON line —
 * {"ok":true,"layout":{...}} or {"error":...,"ok":false} followed by
 * a hard exit (fail loud — an unattachable seat must never fall back
 * silently to the default plan).
 *
 * Reads here use raw fd-0 reads, NOT stdio: nothing has touched stdin
 * via stdio yet (the fgetc opcode loop starts after "ready"), and raw
 * exact-size reads keep poll() truthful about pending requests. */

static void QNN_ArenaReadStdinExact(uint8_t *buf, size_t size,
	const char *what)
{
	size_t have = 0;

	while (have < size)
	{
		ssize_t got = read(0, buf + have, size - have);

		if (got <= 0)
			Sys_Error("Arena server stdin closed mid-%s", what);
		have += (size_t)got;
	}
}

/* Consume + answer one attach request; the opcode byte is already
 * read.  Any validation failure replies the error frame, then exits —
 * the driver read the reason from the reply.
 *
 * `virtual_seats` = the server drains obs itself (virtual/shadow
 * observers): the compiled plan is stored per seat and sizes that
 * seat's frames.  In EXTERNAL observer mode the server emits no obs —
 * the arena clients carry their own plans — but the driver still
 * attaches through the server pipe, so validate + reply the layout
 * (the negotiation half) with nothing to bind. */
static void QNN_ArenaHandleAttachRequest(qboolean virtual_seats,
	int external_count)
{
	uint8_t header[5];
	int seat_index;
	uint32_t json_len;
	char *json;
	char error[256];
	static char reply[4096];
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;

	QNN_ArenaReadStdinExact(header, sizeof(header), "attach-decl header");
	seat_index = header[0];
	json_len = (uint32_t)header[1] | ((uint32_t)header[2] << 8)
		| ((uint32_t)header[3] << 16) | ((uint32_t)header[4] << 24);
	if (json_len == 0 || json_len > QNN_OBS_DECL_JSON_MAX)
		Sys_Error("Arena attach-decl length %u out of range (1..%d)",
			json_len, QNN_OBS_DECL_JSON_MAX);
	json = malloc(json_len + 1);
	if (json == NULL)
		Sys_Error("Arena attach-decl: out of memory (%u bytes)", json_len);
	QNN_ArenaReadStdinExact((uint8_t *)json, json_len, "attach-decl JSON");
	json[json_len] = 0;

	if (!QNN_ObsDeclParseJson(json, (int)json_len, &decl,
			error, sizeof(error))
		|| !QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)))
		goto reject;
	if (virtual_seats)
	{
		if (!QNN_ArenaVirtualAttachSeatPlan(seat_index, &plan,
				error, sizeof(error)))
			goto reject;
	}
	else if (seat_index < 0 || seat_index >= external_count)
	{
		snprintf(error, sizeof(error),
			"obs declaration seat %d out of range (0..%d)",
			seat_index, external_count - 1);
		goto reject;
	}
	free(json);

	if (!QNN_ObsLayoutReplyJson(&plan, reply, sizeof(reply),
			error, sizeof(error)))
		Sys_Error("Arena attach-decl reply failed: %s", error);
	fprintf(stdout, "%s\n", reply);
	fflush(stdout);
	return;

reject:
	QNN_WriteError(error);
	Sys_Error("Arena seat %d obs declaration rejected: %s",
		seat_index, error);
}

/* Service pending attach requests.  timeout_ms 0 = drain whatever is
 * immediately available (called inside the sign-on pump loops);
 * QNN_ARENA_ATTACH_DRAIN_MS = the final pre-"ready" quiet window. */
static void QNN_ArenaServiceAttachRequests(int timeout_ms,
	qboolean virtual_seats, int external_count)
{
	for (;;)
	{
		struct pollfd pfd;
		int ready;
		uint8_t opcode;

		pfd.fd = 0;
		pfd.events = POLLIN;
		ready = poll(&pfd, 1, timeout_ms);
		if (ready < 0)
			Sys_Error("Arena server poll(stdin) failed");
		if (ready == 0)
			return;
		if (pfd.revents & (POLLHUP | POLLERR)
			&& !(pfd.revents & POLLIN))
			Sys_Error("Arena server stdin closed before ready");
		QNN_ArenaReadStdinExact(&opcode, 1, "opcode");
		if (opcode != QNN_OBS_OP_ATTACH_DECL)
			Sys_Error("Arena server opcode %d before ready (only "
				"OP_ATTACH_DECL=%d is valid pre-ready)",
				opcode, QNN_OBS_OP_ATTACH_DECL);
		QNN_ArenaHandleAttachRequest(virtual_seats, external_count);
	}
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
	int virtual_clients;
	int shadow_clients;
	int bot_skill;
	int port;
	int frame;
	float dt = 1.0f / 20.0f;
	char command[256];
	double deadline;

	QNN_FaultInit("ppo_arena_server");
	/* fd 0 carries the binary opcode/attach protocol — the dedicated
	   console reader must never race it for bytes (Sys_ConsoleInput
	   runs inside every Host_Frame on a dedicated host, including the
	   sign-on pump where OP_ATTACH_DECL arrives). */
	qnn_stdin_is_protocol = true;
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
	virtual_clients = QNN_ArgInt("-qnn_virtual_clients", 0) ? 1 : 0;
	shadow_clients = QNN_ArgInt("-qnn_shadow_clients", 0) ? 1 : 0;
	if (virtual_clients && shadow_clients)
		Sys_Error("Arena virtual and shadow client modes are mutually exclusive");
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
	QNN_ArenaApplyInventory();
	QNN_ArenaConfigureActionSeats(selfplay ? true : false);

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
	if (virtual_clients || shadow_clients)
		QNN_ArenaVirtualConfigure(external_count, selfplay ? true : false,
			shadow_clients ? true : false, getenv("QNN_REWARD_JSON"));
	QNN_WriteState("listening", port);

	/* Clients perform normal NQ sign-on autonomously while the server pumps.
	   Their qnn_<match>_<seat> names make the final assignment independent of
	   UDP arrival order. */
	deadline = Sys_FloatTime() + 120.0;
	for (frame = 0; Sys_FloatTime() < deadline; ++frame)
	{
		QNN_ServerFrame(dt);
		if (virtual_clients || shadow_clients)
			QNN_ArenaVirtualPumpSignon(dt);
		/* OP_ATTACH_DECL arrives between "listening" and "ready"; the
		   driver blocks on each reply BEFORE spawning the external
		   clients this loop waits for, so requests must be serviced
		   inside the pump (a post-loop drain alone would deadlock). */
		QNN_ArenaServiceAttachRequests(0,
			(virtual_clients || shadow_clients) ? true : false,
			external_count);
		if (QNN_ArenaAssignNamedSeats() >= external_count
			&& (!(virtual_clients || shadow_clients) || QNN_ArenaVirtualReady()))
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
	{
		QNN_ServerFrame(dt);
		if (virtual_clients || shadow_clients)
			QNN_ArenaVirtualPumpSignon(dt);
	}
	Cvar_SetValue("qnn_arena_ready", 1.0f);
	/* A full self-play world resumes sixteen clients at once.  If the server
	   emits only one or two ready snapshots and immediately parks on stdin,
	   those tail datagrams can be dropped behind queued sign-on traffic and a
	   client waits forever for a frame that will never be retransmitted.  Pace
	   a short readiness window so every socket drains before steady stepping. */
	for (frame = 0; frame < 64; ++frame)
	{
		QNN_ServerFrame(dt);
		if (virtual_clients || shadow_clients)
			QNN_ArenaVirtualPumpSignon(dt);
		QNN_ArenaServiceAttachRequests(0,
			(virtual_clients || shadow_clients) ? true : false,
			external_count);
		usleep(1000);
	}
	/* Final quiet-window drain: after this, per-seat frame sizes are
	   frozen (QNN_ArenaVirtualWriteInitial emits the first frames). */
	QNN_ArenaServiceAttachRequests(QNN_ARENA_ATTACH_DRAIN_MS,
		(virtual_clients || shadow_clients) ? true : false,
		external_count);
	if (virtual_clients || shadow_clients)
		QNN_ArenaVirtualPrepare();
	QNN_WriteState("ready", port);
	if (virtual_clients || shadow_clients)
		QNN_ArenaVirtualWriteInitial(stdout);

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
		if (opcode == QNN_ARENA_SERVER_OP_STEP_BATCH)
		{
			qnn_action_t actions[16];
			int action_count = fgetc(stdin);
			if (action_count == EOF)
				break;
			if (action_count != external_count || action_count > 16)
				Sys_Error("Arena action batch count %d != external count %d",
					action_count, external_count);
			if (fread(actions, sizeof(qnn_action_t), (size_t)action_count, stdin)
				!= (size_t)action_count)
				break;
			if (virtual_clients || shadow_clients)
				QNN_ArenaVirtualStageActions(actions, action_count);
			if (!QNN_ArenaStageActions(actions, action_count))
				Sys_Error("Invalid arena action batch");
			QNN_ServerFrame(dt);
			if (QNN_ArenaPendingActionCount() != 0)
				Sys_Error("Arena action batch did not map to every policy seat");
			fputc(QNN_ARENA_SERVER_OP_STEP_BATCH, stdout);
			fflush(stdout);
			if (virtual_clients || shadow_clients)
				QNN_ArenaVirtualReceive(stdout, dt, false);
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
			if (virtual_clients || shadow_clients)
				QNN_ArenaVirtualReceive(stdout, dt, true);
			continue;
		}
		Sys_Error("Unknown arena server opcode %d", opcode);
	}

	if (virtual_clients || shadow_clients)
		QNN_ArenaVirtualShutdown();
	Host_Shutdown();
	return 0;
}
