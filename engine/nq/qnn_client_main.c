/*
 * qnn_client_main.c — Network NQ client driven by an in-process libqnn
 * policy.
 *
 * Usage:
 *   qnn [standard quake args]
 *
 *   qnn +model qnn_v22 +connect 192.168.1.50              # load + auto-connect
 *   qnn                                                   # idle; load via
 *                                                           `model <name>` at
 *                                                           the console
 *
 * The model is loaded via the `model` console command (registered after
 * Host_Init); the name always resolves to models/<name>.onnx (the /app/models
 * bind mount in the shipped image), so `model qnn_v22` -> models/qnn_v22.onnx.
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
#include "qnn_predict.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>   /* acos / atan2 / M_PI for the decoded-action log */
#include <signal.h>

#define QNN_CLIENT_TICK_HZ 20

static float qnn_client_fixed_dt = 1.0f / (float)QNN_CLIENT_TICK_HZ;
static double qnn_client_next_tick_time = 0.0;
static char qnn_client_map_id[QNN_MAX_MAP_ID];
static FILE *qnn_client_engine_log = NULL;
static FILE *qnn_client_action_log = NULL;
/* QNN_CLIENT_OBS_DUMP: framed QOBS stream of the exact per-tick obs the
 * model saw (canonical wire packer), for offline pytorch replay with full
 * logit/decode introspection. Same record layout as QNN_EmitTick (which
 * lives in qnn_collect_helpers.c, not linked here): "QOBS" + 16-byte
 * header + obs buffer + action struct. No jitter filtering — raw ticks. */
static FILE *qnn_client_obs_dump = NULL;

static void QNN_ClientDumpObs(const qnn_snapshot_t *snapshot,
	int tick, int tick_hz, qboolean reset_flag)
{
	uint8_t obs[QNN_OBS_BUFFER_SIZE];
	uint8_t header[16];
	qnn_tick_result_t result;
	int steps = 0;
	uint16_t flags = reset_flag ? 0x01 : 0;
	uint16_t asize = (uint16_t)sizeof(qnn_action_t);

	QNN_IOEmit(snapshot, &result);
	QNN_IOPackObsBuffer(obs, &result);
	memcpy(header + 0,  &tick,    4);
	memcpy(header + 4,  &steps,   4);
	memcpy(header + 8,  &tick_hz, 4);
	memcpy(header + 12, &flags,   2);
	memcpy(header + 14, &asize,   2);
	fwrite("QOBS", 1, 4, qnn_client_obs_dump);
	fwrite(header, 1, sizeof(header), qnn_client_obs_dump);
	fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, qnn_client_obs_dump);
	fwrite(&snapshot->action_label, 1, sizeof(qnn_action_t), qnn_client_obs_dump);
	fflush(qnn_client_obs_dump);
}
static int qnn_client_tick_index = 0;

/* Set from a signal handler on SIGTERM/SIGINT (e.g. `docker stop`) so the
 * main loop can exit and send a clean clc_disconnect instead of the socket
 * dying as a silent, still-open connection the server has to time out. */
static volatile sig_atomic_t qnn_client_shutdown_requested = 0;

static void QNN_Client_HandleShutdownSignal(int sig)
{
	(void)sig;
	qnn_client_shutdown_requested = 1;
}

/* Console terminal + chat-driven remote console (qnn_client_console.c). */
void QNN_ClientConsoleInit(void);
void QNN_ConsoleRegisterCvars(void);
void QNN_ConsoleExecPending(void);

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
		Con_Printf("usage: model <name>  (always resolves to models/<name>.onnx)\n");
		Con_Printf("current: %s\n", qnn_client_onnx_ctx != NULL ? "(loaded)" : "(none)");
		return;
	}
	path = Cmd_Argv(1);

	/* Resolve the model name -> models/<name>.onnx, always.  The model dir is
	 * the /app/models bind mount in compose.live.yaml, so `model v17` loads
	 * models/v17.onnx.  `.onnx` is appended if absent (case-insensitive) so an
	 * explicit `model v17.onnx` does not become v17.onnx.onnx. */
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
	if ((int)snprintf(path_buf, sizeof(path_buf), "models/%s%s",
	                  path, has_ext ? "" : ".onnx") >= (int)sizeof(path_buf))
	{
		Con_Printf("model: path too long\n");
		return;
	}
	path = path_buf;

	new_ctx = QNN_OnnxInit(path);
	if (new_ctx == NULL)
	{
		Con_Printf("model: load failed: %s\n", QNN_OnnxLastError());
		return;
	}
	if (qnn_client_onnx_ctx != NULL)
		QNN_OnnxFree(qnn_client_onnx_ctx);
	qnn_client_onnx_ctx = new_ctx;

	/* Adopt the model's decision cadence: QNN_OnnxInit refused the load unless
	 * the model carried a valid `tick_hz` stamp, so this is always > 0 (no
	 * default). The whole client loop is driven off qnn_client_fixed_dt — the
	 * decision-tick scheduler, Host_Frame step, and prediction dt — and
	 * qnn_runtime.fixed_tick_hz feeds the latency/back-shift + attack_finished
	 * normalization. A 10 Hz model now decides every 100 ms, holding the
	 * usercmd between decisions. */
	{
		int model_hz = QNN_OnnxTickHz(new_ctx);
		qnn_runtime.fixed_tick_hz = model_hz;
		qnn_client_fixed_dt = 1.0f / (float)model_hz;
	}
	/* The clean load line ("model: loaded <path> [wire / semantics]") is
	 * printed by QNN_OnnxInit, which knows the stamp-selected codec. */
}

/* Escape an engine string (player name, model path) into a JSON-safe,
   pure-ASCII buffer.  Player names are peer-controlled and can carry ",
   \, control bytes, or Quake high-bit color chars — any of which would
   break the JSONL line.  Output is NUL-terminated and truncated to fit
   `out_size` (worst case 6 bytes per input char for \uXXXX). */
static void QNN_JsonEscape(char *out, size_t out_size, const char *in, size_t in_max)
{
	size_t r, w = 0;

	if (out_size == 0)
		return;
	for (r = 0; r < in_max && in[r] != '\0'; ++r)
	{
		unsigned char c = (unsigned char)in[r];
		char esc[7];
		const char *seg;
		size_t seg_len;

		if (c == '"')        { seg = "\\\""; seg_len = 2; }
		else if (c == '\\')  { seg = "\\\\"; seg_len = 2; }
		else if (c >= 0x20 && c < 0x7f) { esc[0] = (char)c; esc[1] = '\0'; seg = esc; seg_len = 1; }
		else { snprintf(esc, sizeof(esc), "\\u%04x", c); seg = esc; seg_len = 6; }

		if (w + seg_len >= out_size)
			break;
		memcpy(out + w, seg, seg_len);
		w += seg_len;
	}
	out[w] = '\0';
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
		"\"items\":%d,"
		"\"ammo\":[%d,%d,%d,%d],\"scoreboard\":[",
		qnn_client_tick_index, Sys_FloatTime(),
		(double)cl.mtime[0], (double)cl.mtime[1],
		cls.signon, cl.viewentity,
		cl.num_entities, cl.maxclients,
		cl.viewangles[0], cl.viewangles[1], cl.viewangles[2],
		cl.velocity[0], cl.velocity[1], cl.velocity[2],
		cl.stats[STAT_HEALTH], cl.stats[STAT_ACTIVEWEAPON],
		cl.items,
		cl.stats[STAT_SHELLS], cl.stats[STAT_NAILS],
		cl.stats[STAT_ROCKETS], cl.stats[STAT_CELLS]);
	if (cl.scores != NULL)
	{
		n = cl.maxclients < 16 ? cl.maxclients : 16;
		for (i = 0; i < n; ++i)
		{
			char name_esc[16 * 6 + 1];
			if (cl.scores[i].name[0] == 0) continue;
			if (reported > 0)
				fprintf(qnn_client_engine_log, ",");
			QNN_JsonEscape(name_esc, sizeof(name_esc), cl.scores[i].name, 16);
			fprintf(qnn_client_engine_log,
				"{\"slot\":%d,\"name\":\"%s\",\"frags\":%d,\"colors\":%d}",
				i + 1, name_esc, cl.scores[i].frags, cl.scores[i].colors);
			reported++;
		}
	}
	fprintf(qnn_client_engine_log, "],\"entities\":[");
	reported = 0;
	n = cl.num_entities < 32 ? cl.num_entities : 32;
	for (i = 1; i < n; ++i)
	{
		entity_t *e = &cl_entities[i];
		char model_esc[32 * 6 + 1];
		const char *mname = (e->model != NULL) ? e->model->name : "(null)";
		if (e->model == NULL) continue;
		if (reported > 0)
			fprintf(qnn_client_engine_log, ",");
		QNN_JsonEscape(model_esc, sizeof(model_esc), mname, 32);
		fprintf(qnn_client_engine_log,
			"{\"i\":%d,\"model\":\"%s\",\"origin\":[%.0f,%.0f,%.0f],\"skin\":%d}",
			i, model_esc, e->origin[0], e->origin[1], e->origin[2], e->skinnum);
		reported++;
	}
	fprintf(qnn_client_engine_log, "]}\n");
	fflush(qnn_client_engine_log);
}

/* Dump the model's DECODED action per tick (the decision, not the engine-state
   result): move signs, attack, look turn magnitude + heading, decided weapon vs
   the currently-held weapon. JSONL, one record per tick. Triggered by the
   QNN_CLIENT_ACTION_LOG env var. This is what separates move-jitter (fb/lr
   flipping) from look-spin (heading sweeping) and shows whether the weapon head
   ever decides to switch (decided != held). */
static void QNN_LogAction(const qnn_action_t *a)
{
	float turn_deg, heading_deg;

	if (qnn_client_action_log == NULL || a == NULL)
		return;

	/* look[0]=cos(turn), look[1]=yaw comp (right), look[2]=pitch comp (up). */
	turn_deg = (float)(acos((double)QNN_Clamp(a->look[0], -1.0f, 1.0f)) * 180.0 / M_PI);
	heading_deg = (float)(atan2((double)a->look[2], (double)a->look[1]) * 180.0 / M_PI);

	/* `held` is the ENGINE-EQUIPPED weapon (cl.stats[STAT_ACTIVEWEAPON] is
	   an IT_ item bitflag — IT_AXE=4096, IT_SHOTGUN=1, ... — not a 1..8 id,
	   so QNN_WeaponId() does the canonical mapping).  It is logged for ONE
	   purpose: detecting ENGINE-FORCED switches (ammo-out / pickup
	   re-equips).  It is NOT the model's weapon choice and must never be
	   used as weapon identity in analysis -- the model has no equip input
	   and never sees this value.  Weapon identity exists only at
	   discharges: read `att_imp` (the 9-way attack-with class, 0 = no
	   attack, 1..8 = attack WITH that impulse).

	   `weapon` (a->weapon) is the separate weapon slot of the retired a26
	   two-output convention; the a27+ single-lane graph publishes no such
	   output (qnn_onnx.c: out->weapon is 0 unless
	   out_present[QNN_ONNX_OUT_WEAPON]), so it reads 0 on every tick here.
	   Kept only so an a26-wire client logs the same schema. */
	fprintf(qnn_client_action_log,
		"{\"t\":%d,\"move\":[%d,%d,%d],\"attack\":%d,"
		"\"turn_deg\":%.1f,\"heading_deg\":%.1f,"
		"\"weapon\":%d,\"att_imp\":%d,\"held\":%d}\n",
		qnn_client_tick_index,
		QNN_ActionAxisSign(a->move, 0), QNN_ActionAxisSign(a->move, 1),
		QNN_ActionAxisSign(a->move, 2), (a->move & 1) ? 1 : 0,
		turn_deg, heading_deg,
		(int)a->weapon, (int)a->attack, QNN_WeaponId());
	fflush(qnn_client_action_log);
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
	/* Self-state prediction: replay this client's own issued cmds onto the
	 * (transport-lagged) server velocity so the obs match the model's
	 * server-aligned training semantics. Reset clears the cmd ring with
	 * the episode; the lag estimate persists (transport doesn't change). */
	if (reset_flag)
		QNN_PredictReset();
	QNN_PredictTick(qnn_client_fixed_dt);
	QNN_PredictSelfVelocity(snapshot.player_velocity);
	snapshot.action_label = *prev_action;
	QNN_DrainSounds(&snapshot);

	QNN_IOUpdate(&snapshot, qnn_client_fixed_dt, reset_flag);
	if (qnn_client_obs_dump != NULL)
		QNN_ClientDumpObs(&snapshot, qnn_client_tick_index,
			(int)(1.0 / qnn_client_fixed_dt + 0.5), reset_flag);
	/* Emit through the loaded model's compiled obs plan (WS2): atlas
	 * parameterization and oracle disclosure ride the plan, so the
	 * engine computes EXACTLY this model's observation. */
	QNN_IOEmitPlan(QNN_OnnxObsPlan(ctx), &snapshot, &result);

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

	signal(SIGTERM, QNN_Client_HandleShutdownSignal);
	signal(SIGINT, QNN_Client_HandleShutdownSignal);

	QNN_ClientConsoleInit();
	QNN_FaultInit("nq_client");
	QNN_PredictInit();
	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	QNN_ClearAction(&qnn_pending_action);
	QNN_ClearAction(&action_a);
	QNN_ClearAction(&action_b);

	{
		const char *engine_log = getenv("QNN_CLIENT_ENGINE_LOG");
		const char *action_log = getenv("QNN_CLIENT_ACTION_LOG");
		if (engine_log != NULL && engine_log[0] != 0)
		{
			qnn_client_engine_log = fopen(engine_log, "w");
			if (qnn_client_engine_log == NULL)
				fprintf(stderr, "qnn_client: failed to open engine log %s\n", engine_log);
			else
				fprintf(stderr, "qnn_client: engine state log -> %s\n", engine_log);
		}
		if (action_log != NULL && action_log[0] != 0)
		{
			qnn_client_action_log = fopen(action_log, "w");
			if (qnn_client_action_log == NULL)
				fprintf(stderr, "qnn_client: failed to open action log %s\n", action_log);
			else
				fprintf(stderr, "qnn_client: decoded-action log -> %s\n", action_log);
		}
		{
			const char *obs_dump = getenv("QNN_CLIENT_OBS_DUMP");
			if (obs_dump != NULL && obs_dump[0] != 0)
			{
				qnn_client_obs_dump = fopen(obs_dump, "wb");
				if (qnn_client_obs_dump == NULL)
					fprintf(stderr, "qnn_client: failed to open obs dump %s\n", obs_dump);
				else
					fprintf(stderr, "qnn_client: QOBS obs dump -> %s\n", obs_dump);
			}
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

	/* Register the model-swap cmd before the command buffer is flushed below,
	 * so quake.rc's `stuffcmds` (queued by Host_Init's `exec quake.rc`) honors
	 * `+model <path>` on argv as the initial load. */
	Cmd_AddCommand("model", QNN_Cmd_Model_f);

	/* Same timing requirement for the perception cvars: without this,
	 * `qnn_fov` set from argv or a startup cfg (e.g. the live share's qnn.cfg) is
	 * "Unknown command" — silently dropped — because QNN_IOInit only runs
	 * at the first QNN tick, long after configs exec. */
	QNN_RegisterPerceptionCvars();
	QNN_ConsoleRegisterCvars();

	/* Sound-subsystem no-ops (see QNN_Cmd_SoundStub_f comment). */
	Cmd_AddCommand("play",      QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("playvol",   QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("stopsound", QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("soundlist", QNN_Cmd_SoundStub_f);
	Cmd_AddCommand("soundinfo", QNN_Cmd_SoundStub_f);

	/* Drain the startup command buffer (quake.rc stuffcmds: +model, connect,
	 * configs) by running one frame, so the buffer executes inside _Host_Frame's
	 * host_abortserver setjmp — exactly how stock Quake processes it. A failed
	 * `connect` (or any other Host_Error) then unwinds back here and we fall
	 * through to the idle loop, instead of longjmp'ing into an unset jump buffer
	 * and crashing (which boot-loops under a restart policy). Loading `+model`
	 * here also adopts the model's tick cadence before the loop sets its pacing
	 * below. (We still don't add a second `stuffcmds` — that loaded +model twice.) */
	Host_Frame(qnn_client_fixed_dt);

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

		if (qnn_client_shutdown_requested)
			break;

		if (qnn_client_next_tick_time > now)
			QNN_SleepUntil(qnn_client_next_tick_time);
		else
			qnn_client_next_tick_time = now;
		qnn_client_next_tick_time += qnn_client_fixed_dt;

		/* Interactive console: run any command line typed at the terminal.
		 * Cbuf executes it inside Host_Frame regardless of key_dest. */
		{
			char *line = Sys_ConsoleInput();
			if (line != NULL)
			{
				Cbuf_AddText(line);
				Cbuf_AddText("\n");
			}
		}

		/* Run any pending chat (qnn_rcon) command and reply to the sender. */
		QNN_ConsoleExecPending();

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

		QNN_LogAction(next_action);   /* the decoded decision this tick */

		swap = cur_action;
		cur_action = next_action;
		next_action = swap;
		was_ready = true;
	}

	CL_Disconnect();
	Host_Shutdown();
	return 0;
}
