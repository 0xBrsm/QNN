/*
 * qnn_client_main.c — Network NQ client driven by an in-process libqnn
 * policy.
 *
 * Usage:
 *   qnn [standard quake args]
 *
 *   qnn +model onnx/qnn_v22.onnx +connect 192.168.1.50  # load + auto-connect
 *   qnn                                                   # idle; load via
 *                                                           `model <path>` at
 *                                                           the console
 *
 * The model is loaded via the `model` console command (registered after
 * Host_Init); paths are resolved relative to cwd (= WORKDIR /app in the
 * shipped image), so `model onnx/qnn_v22.onnx` -> /app/onnx/qnn_v22.onnx.
 * The same command swaps models at runtime; hidden state resets on swap.
 *
 * Quake's standard "+command" argv injection is honored via stuffcmds,
 * so any console command (model, connect, name, say, kill, disconnect,
 * exec) works either from argv or from stdin once the process is running.
 *
 * Per tick (when connected and signon complete):
 *   - Push the previously-computed action into qnn_pending_action
 *   - Run one Host_Frame (engine consumes the action via IN_Move)
 *   - Build a fresh qnn_obs_t from cl.* via QNN_Libqnn_FillObs
 *   - Call qnn_step → next qnn_action_t
 *
 * When disconnected the loop still ticks Host_Frame so the console and
 * network code keep flowing; inference is skipped until signon completes.
 *
 * Observation path is pure cl.* (cl_entities, cl.stats, cl.scores,
 * cl.worldmodel) — no sv.edicts reads — so what the model sees here
 * is exactly what it sees during training-via-loopback.
 */

#include "qnn.h"
#include "qnn_collect_helpers.h"   /* qnn_runtime — defined in qnn_client_runtime_stub.c */
#include "qnn_fault.h"
#include "qnn_io.h"
#include "qnn_onnx.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define QNN_CLIENT_TICK_HZ 20

static float qnn_client_fixed_dt = 1.0f / (float)QNN_CLIENT_TICK_HZ;
static double qnn_client_next_tick_time = 0.0;
static char qnn_client_map_id[QNN_MAX_MAP_ID];
static FILE *qnn_client_engine_log = NULL;
static int qnn_client_tick_index = 0;

/* Active ONNX policy.  NULL until `model <path>` (or `+model <path>` on
 * argv via stuffcmds) loads one; the tick loop skips inference and the
 * bot stands still while NULL.  Hot-swapped by QNN_Cmd_Model_f below. */
static qnn_onnx_ctx_t *qnn_client_onnx_ctx = NULL;

/* NQ servers stuffcmd play/playvol/stopsound/soundlist/soundinfo for
 * audible cues (mod weapon switches, CTF flags, map triggers).  Our
 * headless client links no sound subsystem, so without stubs every
 * such stuffcmd logs "Unknown command".  Silent no-ops keep the
 * console clean; the obs path still gets sound info via the existing
 * cl.sound parser. */
static void QNN_Cmd_SoundStub_f(void)
{
}

static void QNN_Cmd_Model_f(void)
{
	const char *path;
	char path_buf[MAX_OSPATH];
	qnn_onnx_ctx_t *new_ctx;
	size_t len;
	qboolean has_ext;

	if (Cmd_Argc() < 2)
	{
		Con_Printf("usage: model <path[.onnx]>\n");
		Con_Printf("current: %s\n", qnn_client_onnx_ctx != NULL ? "(loaded)" : "(none)");
		return;
	}
	path = Cmd_Argv(1);

	/* Append `.onnx` if absent so `model foo` works as shorthand for
	 * `model foo.onnx`.  Case-insensitive so FOO.ONNX is also accepted
	 * as already-extensioned. */
	len = strlen(path);
	has_ext = false;
	if (len >= 5)
	{
		const char *suf = path + len - 5;
		has_ext = (suf[0] == '.' &&
		           tolower((unsigned char)suf[1]) == 'o' &&
		           tolower((unsigned char)suf[2]) == 'n' &&
		           tolower((unsigned char)suf[3]) == 'n' &&
		           tolower((unsigned char)suf[4]) == 'x');
	}
	if (!has_ext)
	{
		if (len + 5 >= sizeof(path_buf))
		{
			Con_Printf("model: path too long\n");
			return;
		}
		snprintf(path_buf, sizeof(path_buf), "%s.onnx", path);
		path = path_buf;
	}

	new_ctx = QNN_OnnxInit(path);
	if (new_ctx == NULL)
	{
		Con_Printf("model: load failed: %s\n", QNN_OnnxLastError());
		return;
	}
	if (qnn_client_onnx_ctx != NULL)
		QNN_OnnxFree(qnn_client_onnx_ctx);
	qnn_client_onnx_ctx = new_ctx;
	Con_Printf("model: loaded %s\n", path);
}

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

/* Build a fresh obs from cl.* state and run one ONNX inference.
 * Sticky-weapon selection and greedy decoding live inside QNN_OnnxStep,
 * which writes directly into the engine's qnn_action_t. */
static qboolean QNN_BuildAndInfer(
	qnn_onnx_ctx_t *ctx,
	const qnn_action_t *prev_action,
	qboolean reset_flag,
	qnn_action_t *next_action)
{
	qnn_snapshot_t snapshot;
	qnn_tick_result_t result;

	QNN_CaptureBaseSnapshot(&snapshot);
	snapshot.action_label = *prev_action;
	QNN_DrainSounds(&snapshot);

	QNN_IOUpdate(&snapshot, qnn_client_fixed_dt, reset_flag);
	QNN_IOEmit(&snapshot, &result);

	if (QNN_OnnxStep(ctx, &result, next_action) != 0) {
		fprintf(stderr, "qnn_client: QNN_OnnxStep failed: %s\n", QNN_OnnxLastError());
		return false;
	}
	return true;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	qnn_action_t action_a, action_b;
	qnn_action_t *cur_action = &action_a;
	qnn_action_t *next_action = &action_b;
	qboolean was_ready = false;

	QNN_FaultInit("nq_client");
	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	QNN_ClearAction(&qnn_pending_action);
	QNN_ClearAction(&action_a);
	QNN_ClearAction(&action_b);

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

	/* Seed qnn_runtime so qnn_self_common.c's attack_finished
	 * normalization sees the canonical 20 Hz tick rate. */
	qnn_runtime.fixed_tick_hz = QNN_CLIENT_TICK_HZ;

	Host_Init(&parms);
	QNN_TickRegister();
	cls.demonum = -1;

	/* Register the runtime model-swap cmd before stuffcmds runs so that
	 * `+model <path>` on argv is honored as the initial load. */
	Cmd_AddCommand("model", QNN_Cmd_Model_f);

	/* Sound-subsystem no-ops (see QNN_Cmd_SoundStub_f comment). */
	Cmd_AddCommand("play",      QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("playvol",   QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("stopsound", QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("soundlist", QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("soundinfo", QNN_Cmd_SoundStub_f);

	Cbuf_AddText("stuffcmds\n");
	Cbuf_Execute();

	if (qnn_client_onnx_ctx == NULL)
		fprintf(stderr, "qnn_client: no model loaded; type `model <path.onnx>` "
		                "at the console (or pass +model <path> on argv).\n");

	qnn_client_next_tick_time = Sys_FloatTime();

	for (;;)
	{
		double now = Sys_FloatTime();
		char error[256];
		qboolean ready;
		qboolean ok;
		qnn_action_t *swap;

		if (qnn_client_next_tick_time > now)
			QNN_SleepUntil(qnn_client_next_tick_time);
		else
			qnn_client_next_tick_time = now;
		qnn_client_next_tick_time += qnn_client_fixed_dt;

		qnn_pending_action = *cur_action;
		Host_Frame(qnn_client_fixed_dt);
		QNN_ClearAction(&qnn_pending_action);

		ready = QNN_NetClientReady();
		if (!ready)
		{
			was_ready = false;
			continue;
		}

		if (!QNN_RebuildNavIfMapChanged(error, sizeof(error)))
		{
			fprintf(stderr, "qnn_client: %s\n", error);
			CL_Disconnect();
			was_ready = false;
			continue;
		}

		if (qnn_client_onnx_ctx == NULL)
		{
			/* No model loaded: keep the connection alive (Host_Frame
			 * already ran above) but feed zero actions and don't bump
			 * tick state.  `model <path>` at the console will start
			 * inference on the next tick. */
			QNN_ClearAction(cur_action);
			QNN_ClearAction(next_action);
			was_ready = false;
			continue;
		}

		if (!was_ready)
		{
			/* First tick after signon (or after a model load): reset
			 * GRU and prime inference from the post-signon snapshot. */
			qnn_client_tick_index = 0;
			qnn_runtime.tick = 0;
			QNN_ClearAction(cur_action);
			QNN_ClearAction(next_action);
			ok = QNN_BuildAndInfer(qnn_client_onnx_ctx, cur_action, true, next_action);
		}
		else
		{
			QNN_LogEngineState();
			qnn_client_tick_index += 1;
			qnn_runtime.tick = qnn_client_tick_index;
			ok = QNN_BuildAndInfer(qnn_client_onnx_ctx, cur_action, false, next_action);
		}

		if (!ok)
		{
			fprintf(stderr, "qnn_client: inference failed, dropping connection\n");
			CL_Disconnect();
			was_ready = false;
			continue;
		}

		swap = cur_action;
		cur_action = next_action;
		next_action = swap;
		was_ready = true;
	}
}
