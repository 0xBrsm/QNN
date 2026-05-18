/*
 * qnn_client_main.c — Network NQ client driven by an external policy.
 *
 * Connects to a real NetQuake server on startup using the address passed
 * as a positional argument, then services a binary step protocol on
 * stdin/stdout for as long as the connection lasts.
 *
 * Protocol (no handshake — Python only runs when we're connected):
 *   stdin   stream of (0x01 + qnn_action_t) records
 *   stdout  stream of QNN_OBS_BUFFER_SIZE-byte obs frames
 *
 * The worker connects, prepares the navmesh from cl.worldmodel, then
 * blocks on stdin reading step opcodes.  Each step paces Host_Frame to
 * 20Hz wall clock so the server sees us in real time.  On signon
 * timeout, server disconnect, or stdin EOF, the worker logs to stderr
 * and exits — Python sees EOF on stdout and shuts down its loop.
 *
 * Observation path is pure cl.* (cl_entities, cl.stats, cl.scores,
 * cl.worldmodel) — no sv.edicts reads — so what the model sees here
 * is exactly what it sees during training-via-loopback.
 */

#include "qnn.h"
#include "qnn_fault.h"
#include "qnn_io.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define QNN_CLIENT_TICK_HZ 20
#define QNN_CLIENT_SIGNON_TIMEOUT_SECONDS 30.0
#define QNN_CLIENT_MAX_ADDRESS 256

static float qnn_client_fixed_dt = 1.0f / (float)QNN_CLIENT_TICK_HZ;
static double qnn_client_next_tick_time = 0.0;
static char qnn_client_map_id[QNN_MAX_MAP_ID];
static FILE *qnn_client_engine_log = NULL;
static int qnn_client_tick_index = 0;

/* Dump per-tick engine state so we can see what cl.* actually contains
   when the obs is being captured.  Writes JSONL — one record per tick.
   Triggered by QNN_CLIENT_ENGINE_LOG env var. */
static void QNN_LogEngineState(void)
{
	int i, n;
	int reported = 0;

	if (qnn_client_engine_log == NULL)
		return;

	fprintf(qnn_client_engine_log,
		"{\"t\":%d,\"wall\":%.3f,\"mtime\":[%.3f,%.3f],"
		"\"signon\":%d,\"viewent\":%d,\"num_entities\":%d,"
		"\"maxclients\":%d,\"viewangles\":[%.1f,%.1f,%.1f],"
		"\"velocity\":[%.1f,%.1f,%.1f],\"hp\":%d,\"weapon\":%d,"
		"\"items\":%d,\"scoreboard\":[",
		qnn_client_tick_index, Sys_FloatTime(),
		(double)cl.mtime[0], (double)cl.mtime[1],
		cls.signon, cl.viewentity,
		cl.num_entities, cl.maxclients,
		cl.viewangles[0], cl.viewangles[1], cl.viewangles[2],
		cl.velocity[0], cl.velocity[1], cl.velocity[2],
		cl.stats[STAT_HEALTH], cl.stats[STAT_ACTIVEWEAPON],
		cl.items);
	if (cl.scores != NULL)
	{
		n = cl.maxclients < 16 ? cl.maxclients : 16;
		for (i = 0; i < n; ++i)
		{
			if (cl.scores[i].name[0] == 0) continue;
			if (reported > 0)
				fprintf(qnn_client_engine_log, ",");
			fprintf(qnn_client_engine_log,
				"{\"slot\":%d,\"name\":\"%.16s\",\"frags\":%d,\"colors\":%d}",
				i + 1, cl.scores[i].name, cl.scores[i].frags, cl.scores[i].colors);
			reported++;
		}
	}
	fprintf(qnn_client_engine_log, "],\"entities\":[");
	reported = 0;
	n = cl.num_entities < 32 ? cl.num_entities : 32;
	for (i = 1; i < n; ++i)
	{
		entity_t *e = &cl_entities[i];
		const char *mname = (e->model != NULL) ? e->model->name : "(null)";
		if (e->model == NULL) continue;
		if (reported > 0)
			fprintf(qnn_client_engine_log, ",");
		fprintf(qnn_client_engine_log,
			"{\"i\":%d,\"model\":\"%.32s\",\"origin\":[%.0f,%.0f,%.0f],\"skin\":%d}",
			i, mname, e->origin[0], e->origin[1], e->origin[2], e->skinnum);
		reported++;
	}
	fprintf(qnn_client_engine_log, "]}\n");
	fflush(qnn_client_engine_log);
}

static qboolean QNN_NetClientReady(void)
{
	return (cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static void QNN_SleepUntil(double deadline)
{
	for (;;)
	{
		double now = Sys_FloatTime();
		double remaining = deadline - now;
		if (remaining <= 0.001)
			return;
		{
			struct timespec req;
			req.tv_sec = (time_t)remaining;
			req.tv_nsec = (long)((remaining - (double)req.tv_sec) * 1.0e9);
			nanosleep(&req, NULL);
		}
	}
}

/* "maps/e1m1.bsp" → "e1m1" */
static void QNN_DeriveMapIdFromWorldmodel(char *out, size_t out_size)
{
	const char *name;
	const char *base;
	const char *dot;
	size_t i;
	size_t len;

	out[0] = '\0';
	if (cl.worldmodel == NULL || cl.worldmodel->name[0] == 0)
		return;
	name = cl.worldmodel->name;
	base = strrchr(name, '/');
	base = base ? base + 1 : name;
	dot = strrchr(base, '.');
	len = dot ? (size_t)(dot - base) : strlen(base);
	if (len >= out_size)
		len = out_size - 1;
	for (i = 0; i < len; ++i)
		out[i] = (char)tolower((unsigned char)base[i]);
	out[len] = '\0';
}

static qboolean QNN_RebuildNavIfMapChanged(char *error, size_t error_size)
{
	char map_id[QNN_MAX_MAP_ID];

	QNN_DeriveMapIdFromWorldmodel(map_id, sizeof(map_id));
	if (map_id[0] == 0)
	{
		snprintf(error, error_size, "Server has not loaded a worldmodel");
		return false;
	}
	if (!QNN_PrepareMap(map_id, error, error_size))
		return false;
	if (strcmp(qnn_client_map_id, map_id) != 0)
	{
		strncpy(qnn_client_map_id, map_id, sizeof(qnn_client_map_id) - 1);
		qnn_client_map_id[sizeof(qnn_client_map_id) - 1] = '\0';
		QNN_IOInit(&qnn_map_state);
		QNN_FaultSetContext(qnn_client_map_id);
	}
	return true;
}

static qboolean QNN_EmitObsForAction(const qnn_action_t *action, qboolean reset_flag)
{
	qnn_snapshot_t snapshot;
	qnn_tick_result_t result;
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];

	QNN_CaptureBaseSnapshot(&snapshot);
	snapshot.action_label = *action;
	QNN_DrainSounds(&snapshot);

	QNN_IOUpdate(&snapshot, qnn_client_fixed_dt, reset_flag);
	QNN_IOEmit(&snapshot, &result);
	QNN_IOPackObsBuffer(obs, &result);

	if (fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, stdout) != (size_t)QNN_OBS_BUFFER_SIZE)
		return false;
	fflush(stdout);
	return true;
}

static qboolean QNN_ConnectAndPrepare(const char *address)
{
	char host[QNN_CLIENT_MAX_ADDRESS];
	char command[256];
	char error[256];
	const char *colon;
	double deadline;
	int frame;
	int port;

	cls.demonum = -1;
	QNN_ClearAction(&qnn_pending_action);

	/* Stock NQ's UDP_GetAddrFromName ignores any colon in the address and
	 * uses net_hostport (default 26000) for the destination port.  Split
	 * host:port ourselves and set net_hostport so the engine sends CCREQ
	 * to the right place. */
	colon = strchr(address, ':');
	if (colon != NULL)
	{
		size_t host_len = (size_t)(colon - address);
		if (host_len >= sizeof(host))
			host_len = sizeof(host) - 1;
		memcpy(host, address, host_len);
		host[host_len] = '\0';
		port = atoi(colon + 1);
		if (port <= 0 || port > 65535)
		{
			fprintf(stderr, "qnn_client: invalid port in address %s\n", address);
			return false;
		}
	}
	else
	{
		strncpy(host, address, sizeof(host) - 1);
		host[sizeof(host) - 1] = '\0';
		port = 26000;
	}
	net_hostport = port;
	DEFAULTnet_hostport = port;

	snprintf(command, sizeof(command), "connect %s\n", host);
	Cbuf_AddText(command);

	deadline = Sys_FloatTime() + QNN_CLIENT_SIGNON_TIMEOUT_SECONDS;
	frame = 0;
	while (Sys_FloatTime() < deadline)
	{
		Host_Frame(qnn_client_fixed_dt);
		++frame;
		if (QNN_NetClientReady())
			break;
		if (cls.state == ca_disconnected && frame > 4)
		{
			fprintf(stderr, "qnn_client: connection refused by %s\n", address);
			return false;
		}
		/* If Host_Frame returned without progress (typical when the
		   server hasn't responded yet), pause briefly so we don't burn
		   a CPU core retrying CCREQ_CONNECT — the engine's net layer
		   schedules its own retransmits. */
		{
			struct timespec req;
			req.tv_sec = 0;
			req.tv_nsec = 10 * 1000 * 1000; /* 10ms */
			nanosleep(&req, NULL);
		}
	}
	if (!QNN_NetClientReady())
	{
		fprintf(stderr, "qnn_client: timed out waiting for signon from %s after %.0fs\n",
			address, QNN_CLIENT_SIGNON_TIMEOUT_SECONDS);
		return false;
	}

	if (!QNN_RebuildNavIfMapChanged(error, sizeof(error)))
	{
		fprintf(stderr, "qnn_client: %s\n", error);
		return false;
	}
	return true;
}

static qboolean QNN_Step(const qnn_action_t *action)
{
	double now;
	char error[256];

	if (cls.state != ca_connected)
	{
		fprintf(stderr, "qnn_client: server disconnected\n");
		return false;
	}

	qnn_pending_action = *action;

	now = Sys_FloatTime();
	if (qnn_client_next_tick_time > now)
		QNN_SleepUntil(qnn_client_next_tick_time);
	else
		qnn_client_next_tick_time = now;
	qnn_client_next_tick_time += qnn_client_fixed_dt;

	Host_Frame(qnn_client_fixed_dt);

	if (!QNN_RebuildNavIfMapChanged(error, sizeof(error)))
	{
		fprintf(stderr, "qnn_client: %s\n", error);
		return false;
	}

	QNN_ClearAction(&qnn_pending_action);
	QNN_LogEngineState();
	qnn_client_tick_index += 1;
	return QNN_EmitObsForAction(action, false);
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	const char *address;

	if (argc < 2)
	{
		fprintf(stderr, "usage: %s <server_addr>\n", argv[0] ? argv[0] : "nq_client");
		return 2;
	}
	address = argv[1];

	QNN_FaultInit("nq_client");
	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	QNN_ClearAction(&qnn_pending_action);

	{
		const char *engine_log = getenv("QNN_CLIENT_ENGINE_LOG");
		if (engine_log != NULL && engine_log[0] != 0)
		{
			qnn_client_engine_log = fopen(engine_log, "w");
			if (qnn_client_engine_log == NULL)
				fprintf(stderr, "qnn_client: failed to open engine log %s\n", engine_log);
			else
				fprintf(stderr, "qnn_client: engine state log -> %s\n", engine_log);
		}
	}

	memset(&parms, 0, sizeof(parms));
	COM_InitArgv(argc, argv);
	parms.argc = com_argc;
	parms.argv = com_argv;
	parms.memsize = 32 * 1024 * 1024;
	parms.membase = malloc(parms.memsize);
	parms.basedir = basedir;
	Host_Init(&parms);
	QNN_TickRegister();
	cls.demonum = -1;

	if (!QNN_ConnectAndPrepare(address))
	{
		Host_Shutdown();
		return 1;
	}

	/* Emit one initial obs from post-signon state so Python can run its
	   first model forward pass before sending an action.  No Host_Frame
	   here — the snapshot reflects the world the moment we connected. */
	{
		qnn_action_t zero;
		QNN_ClearAction(&zero);
		if (!QNN_EmitObsForAction(&zero, true))
		{
			Host_Shutdown();
			return 1;
		}
	}

	qnn_client_next_tick_time = Sys_FloatTime();

	for (;;)
	{
		int op;
		qnn_action_t action;

		op = fgetc(stdin);
		if (op == EOF)
			break;
		if (op != QNN_BINARY_OP_STEP)
		{
			fprintf(stderr, "qnn_client: expected step opcode 0x%02x, got 0x%02x\n",
				QNN_BINARY_OP_STEP, op);
			break;
		}
		if (fread(&action, 1, QNN_BINARY_ACTION_SIZE, stdin) != (size_t)QNN_BINARY_ACTION_SIZE)
			break;
		if (!QNN_Step(&action))
			break;
	}

	QNN_FreeMapState(&qnn_map_state);
	CL_Disconnect();
	Host_Shutdown();
	return 0;
}
