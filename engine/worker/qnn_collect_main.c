#include "qnn.h"
#include "qnn_io.h"
#include "qnn_store.h"
#include "qnn_metrics.h"

#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_PROTOCOL "v6"
#define QNN_WORKER_SERVER_NAME "quake-demo-worker"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_COMMAND_TEXT 1024

/* ── Match state from svc_print text ──────────────────────────── */

static int qnn_match_state; /* 0=pre, 1=in match, 2=post match */

/*
 * Called from Con_Printf (patched) for every svc_print line.
 * Scans for mod-generated match start/end text.
 */
void QNN_MatchCheckPrint(const char *text)
{
	if (qnn_match_state == 0)
	{
		if (strstr(text, "match has begun") ||
		    strstr(text, "Match has begun") ||
		    strstr(text, "Match Started") ||
		    strstr(text, "match started") ||
		    strstr(text, "match has started") ||
		    strstr(text, "Game Is Starting In 1 Second") ||
		    strstr(text, "Match is 1v1") || strstr(text, "Match is 2v2") ||
		    strstr(text, "Match is 3v3") || strstr(text, "Match is 4v4") ||
		    strstr(text, "Match is 5v5") || strstr(text, "Match is 6v6") ||
		    strstr(text, "Match is 7v7") || strstr(text, "Match is 8v8") ||
		    strstr(text, "Game Is Starting In 1 Sec"))
		{
			qnn_match_state = 1;
			fprintf(stderr, "[demo] match start detected: %.*s\n",
				(int)strcspn(text, "\n"), text);
		}
	}
	else if (qnn_match_state == 1)
	{
		if (strstr(text, "match is over") ||
		    strstr(text, "Match is over") ||
		    strstr(text, "The match is over") ||
		    strstr(text, "Match Over") ||
		    strstr(text, "Game Over") ||
		    strstr(text, "has WON over") ||
		    strstr(text, "has won over"))
		{
			qnn_match_state = 2;
			fprintf(stderr, "[demo] match end detected: %.*s\n",
				(int)strcspn(text, "\n"), text);
		}
	}
}

#define QNN_DEMO_MOUSE_DEGREES_PER_COUNT 0.066f
#define QNN_DEMO_JUMP_VELOCITY_THRESHOLD 160.0f

typedef struct
{
	int	fixed_tick_hz;
	float	fixed_dt;
	qboolean auto_detect_tick_hz;
	int	resample_hz;	/* requested resample target; 0 = use detected rate */
	int	seed;
	int	tick;
	int	steps;
	qboolean has_reset;
	qboolean done;
	vec3_t	prev_origin;
	vec3_t	prev_view_angles;
	vec3_t	prev_velocity;	/* reconstructed from position deltas */
	qboolean prev_grounded;
	int	prev_waterlevel;
	int	prev_weapon_id;
	int	prev_ammo;
	qboolean has_prev;
	FILE	*store_dump;	/* NULL = no dump, set from QNN_STORE_DUMP env var */
	/* Buffered obs for one-tick delay: we emit obs(t) paired with
	   the action computed at t+1 (the transition FROM state t). */
	uint8_t	buffered_obs[QNN_OBS_BUFFER_SIZE];
	qboolean has_buffered_obs;
} qnn_runtime_t;

static qnn_runtime_t qnn_runtime;

static qboolean QNN_ClientReady(void)
{
	return (cls.demoplayback
		&& cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static float QNN_NormalizeYaw(float delta)
{
	while (delta <= -180.0f)
		delta += 360.0f;
	while (delta > 180.0f)
		delta -= 360.0f;
	return delta;
}



static void QNN_InferAction(qnn_action_t *action, const qnn_snapshot_t *snapshot)
{
	float dt;

	QNN_ClearAction(action);

	if (!qnn_runtime.has_prev)
		return;

	/* View-relative look label: record the NEXT frame's forward direction
	   in the PREVIOUS frame's view-relative coordinates.  This captures
	   the turn delta as a direction vector the model can learn to emit.
	   When the player isn't turning, this is (1,0,0) (straight ahead). */
	{
		vec3_t forward, right, up, cur_forward;
		AngleVectors(qnn_runtime.prev_view_angles, forward, right, up);
		QNN_ForwardFromAngles(snapshot->player_view_angles, cur_forward);
		action->look[0] = DotProduct(cur_forward, forward);
		action->look[1] = DotProduct(cur_forward, right);
		action->look[2] = DotProduct(cur_forward, up);
	}

	/* Velocity-based movement labels: reconstruct velocity from position
	   deltas (demos don't reliably include SU_VELOCITY), simulate one
	   frame of engine physics with zero input, compare residual. */
	dt = qnn_resample.target_hz > 0 ? qnn_resample.target_dt : qnn_runtime.fixed_dt;
	{
		vec3_t cur_vel;
		int i;
		if (dt > 0.0f)
			for (i = 0; i < 3; i++)
				cur_vel[i] = (snapshot->player_origin[i] - qnn_runtime.prev_origin[i]) / dt;
		else
			cur_vel[0] = cur_vel[1] = cur_vel[2] = 0.0f;

		QNN_PhysInferMove(action,
			qnn_runtime.prev_velocity,
			cur_vel,
			qnn_runtime.prev_origin,
			qnn_runtime.prev_view_angles,
			qnn_runtime.prev_grounded,
			qnn_runtime.prev_waterlevel,
			dt);
	}

	action->fire = (snapshot->ammo < qnn_runtime.prev_ammo) ? 1 : 0;

	if (qnn_runtime.prev_velocity[2] <= 32.0f && snapshot->player_velocity[2] >= QNN_DEMO_JUMP_VELOCITY_THRESHOLD)
		action->jump = 1;

	if (snapshot->weapon_id != qnn_runtime.prev_weapon_id && snapshot->weapon_id > 0)
		action->switch_slot = QNN_SwitchSlotFromWeaponId(snapshot->weapon_id);
}

/* Emit one framed tick: header + obs + action. */
static void QNN_EmitTick(FILE *out, const uint8_t *obs,
	const qnn_action_t *action, int tick, int steps, int tick_hz,
	uint16_t flags)
{
	uint8_t header[16];

	memcpy(header + 0, &tick, 4);
	memcpy(header + 4, &steps, 4);
	memcpy(header + 8, &tick_hz, 4);
	memcpy(header + 12, &flags, 2);
	{
		uint16_t asize = (uint16_t)sizeof(qnn_action_t);
		memcpy(header + 14, &asize, 2);
	}
	fwrite("QOBS", 1, 4, out);
	fwrite(header, 1, sizeof(header), out);
	fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, out);
	fwrite(action, 1, sizeof(qnn_action_t), out);
	fflush(out);
}

/* Write a single tick in obs_buffer format: obs + action label + framing.
 *
 * One-tick buffer: the obs from tick t is held until tick t+1 so we can
 * pair it with the action computed at t+1 (the transition FROM state t).
 * This ensures the model sees obs(t) with the action taken from that state,
 * not the action that arrived at that state. */
static void QNN_WriteObsTick(FILE *out, const qnn_snapshot_t *snapshot,
	int tick, int steps, int tick_hz, qboolean reset_flag)
{
	uint8_t cur_obs[QNN_OBS_BUFFER_SIZE];
	qnn_tick_result_t result;
	uint16_t flags = 0;

	(void)tick_hz;
	(void)reset_flag;
	QNN_IOEmit(snapshot, &result);
	QNN_IOPackObsBuffer(cur_obs, &result);
	QNN_IOPushAction(snapshot);

	if (!qnn_runtime.has_buffered_obs)
	{
		if (snapshot->done)
		{
			/* Degenerate: done on first tick (0-1 tick episode).
			   Emit directly so the reader sees the DONE flag. */
			QNN_EmitTick(out, cur_obs, &snapshot->action_label,
				tick, steps, tick_hz, 0x02);
			return;
		}
		/* First tick: buffer the obs; no action to pair yet. */
		memcpy(qnn_runtime.buffered_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
		qnn_runtime.has_buffered_obs = true;
		return;
	}

	/* Emit the buffered obs with the current action (transition from
	   the buffered state to this state). */
	if (snapshot->done) flags |= 0x02;
	if (reset_flag) flags |= 0x01;
	QNN_EmitTick(out, qnn_runtime.buffered_obs, &snapshot->action_label,
		tick, steps, tick_hz, flags);

	/* Buffer current obs for next emit (unless this was the done tick,
	   in which case there is no next action to pair with). */
	if (!snapshot->done)
		memcpy(qnn_runtime.buffered_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
	else
		qnn_runtime.has_buffered_obs = false;
}

static void QNN_SavePrev(const qnn_snapshot_t *snapshot, float dt)
{
	/* Reconstruct velocity from position delta before overwriting prev_origin. */
	if (qnn_runtime.has_prev && dt > 0.0f)
	{
		int i;
		for (i = 0; i < 3; i++)
			qnn_runtime.prev_velocity[i] =
				(snapshot->player_origin[i] - qnn_runtime.prev_origin[i]) / dt;
	}
	else
	{
		qnn_runtime.prev_velocity[0] = 0.0f;
		qnn_runtime.prev_velocity[1] = 0.0f;
		qnn_runtime.prev_velocity[2] = 0.0f;
	}
	VectorCopy(snapshot->player_origin, qnn_runtime.prev_origin);
	VectorCopy(snapshot->player_view_angles, qnn_runtime.prev_view_angles);
	qnn_runtime.prev_grounded = snapshot->grounded;
	qnn_runtime.prev_waterlevel = snapshot->waterlevel;
	qnn_runtime.prev_weapon_id = snapshot->weapon_id;
	qnn_runtime.prev_ammo = snapshot->ammo;
	qnn_runtime.has_prev = true;
}

static void QNN_RuntimeReset(void)
{
	memset(&qnn_runtime, 0, sizeof(qnn_runtime));
	qnn_runtime.fixed_tick_hz = 20;
	qnn_runtime.fixed_dt = 1.0f / 20.0f;
	qnn_runtime.auto_detect_tick_hz = false;
	qnn_runtime.store_dump = NULL;
	{
		const char *dump_path = getenv("QNN_STORE_DUMP");
		if (dump_path != NULL && dump_path[0] != '\0')
		{
			qnn_runtime.store_dump = fopen(dump_path, "w");
			if (qnn_runtime.store_dump != NULL)
				fprintf(stderr, "[demo] store dump: %s\n", dump_path);
			else
				fprintf(stderr, "[demo] store dump: failed to open %s\n", dump_path);
		}
	}
}

#define QNN_DEMO_DETECT_PROBE_DT 0.001f
#define QNN_DEMO_DETECT_MAX_FRAMES 8192

static void QNN_DetectNativeTickHz(void)
{
	double first_mtime;
	double second_mtime;
	float native_dt;
	int detected_hz;
	int frame;

	/* Advance with tiny dt until we see the first server time update */
	first_mtime = cl.mtime[0];
	for (frame = 0; frame < QNN_DEMO_DETECT_MAX_FRAMES; ++frame)
	{
		Host_Frame(QNN_DEMO_DETECT_PROBE_DT);
		if (!cls.demoplayback || cls.state == ca_disconnected)
			return;
		if (cl.mtime[0] != first_mtime)
		{
			first_mtime = cl.mtime[0];
			break;
		}
	}

	/* Now advance until the next distinct server time */
	for (frame = 0; frame < QNN_DEMO_DETECT_MAX_FRAMES; ++frame)
	{
		Host_Frame(QNN_DEMO_DETECT_PROBE_DT);
		if (!cls.demoplayback || cls.state == ca_disconnected)
			return;
		if (cl.mtime[0] != first_mtime)
		{
			second_mtime = cl.mtime[0];
			break;
		}
	}
	if (cl.mtime[0] == first_mtime)
		return; /* couldn't detect, keep default */

	native_dt = (float)(second_mtime - first_mtime);
	if (native_dt <= 0.001f || native_dt > 1.0f)
		return; /* implausible, keep default */

	detected_hz = (int)(1.0f / native_dt + 0.5f);
	if (detected_hz < 5 || detected_hz > 200)
		return; /* out of sane range */

	qnn_runtime.fixed_tick_hz = detected_hz;
	qnn_runtime.fixed_dt = 1.0f / (float)detected_hz;
	fprintf(stderr, "[demo] detected native tick rate: %d Hz (dt=%.4fs)\n",
		detected_hz, qnn_runtime.fixed_dt);
}

static void QNN_CaptureSnapshotLocal(qnn_snapshot_t *snapshot, qboolean reset_flag)
{
	QNN_CaptureBaseSnapshot(snapshot);
	snapshot->done = reset_flag ? false : qnn_runtime.done;
	QNN_DrainSounds(snapshot);
}

static qboolean QNN_ResetWorldLocal(const char *demo_path, int seed, char *error, size_t error_size)
{
	char command[QNN_WORKER_MAX_COMMAND_TEXT];
	int frame;

	if (qnn_map_state.map_name[0] == 0)
	{
		snprintf(error, error_size, "Call hello first so the worker knows which map to load");
		return false;
	}
	if (demo_path == NULL || demo_path[0] == 0)
	{
		snprintf(error, error_size, "reset options must include demo_path");
		return false;
	}

	QNN_ClearAction(&qnn_pending_action);
	qnn_runtime.seed = seed >= 0 ? seed : 0;
	qnn_runtime.done = false;
	qnn_runtime.has_reset = false;
	qnn_sound_count = 0;
	cls.demonum = -1;
	cls.timedemo = false;

	/* Disconnect directly from C to avoid command buffer ordering issues */
	CL_Disconnect();
	if (sv.active)
		Host_ShutdownServer(false);
	cls.state = ca_disconnected;

	snprintf(command, sizeof(command), "playdemo %s\n", demo_path);
	Cbuf_AddText(command);

	for (frame = 0; frame < 4096; ++frame)
	{
		Host_Frame(qnn_runtime.fixed_dt);
		if (QNN_ClientReady())
			break;
		if (frame > 0 && !cls.demoplayback && cls.state == ca_disconnected)
			break;
	}
	if (!QNN_ClientReady())
	{
		fprintf(stderr, "[demo] playdemo failed: demoplayback=%d state=%d signon=%d\n",
			cls.demoplayback, cls.state, cls.signon);
		snprintf(error, error_size, "Timed out waiting for demo playback on %s", demo_path);
		return false;
	}

	if (qnn_runtime.auto_detect_tick_hz)
		QNN_DetectNativeTickHz();

	/* The demo may play on a different map than what hello specified.
	   Detect the actual map from the loaded worldmodel and rebuild the
	   map state so static objects (items, doors, etc.) match reality. */
	if (cl.worldmodel != NULL && cl.worldmodel->name[0] != '\0')
	{
		char demo_map[QNN_MAX_MAP_ID];
		const char *bsp_name = cl.worldmodel->name;
		const char *slash;
		const char *dot;
		size_t len;

		/* Extract bare map name from "maps/dm3.bsp" */
		slash = strrchr(bsp_name, '/');
		if (slash != NULL)
			bsp_name = slash + 1;
		dot = strrchr(bsp_name, '.');
		len = dot ? (size_t)(dot - bsp_name) : strlen(bsp_name);
		if (len >= sizeof(demo_map))
			len = sizeof(demo_map) - 1;
		memcpy(demo_map, bsp_name, len);
		demo_map[len] = '\0';

		if (strcasecmp(demo_map, qnn_map_state.map_name) != 0)
		{
			char map_error[256];
			if (!QNN_PrepareMap(demo_map, map_error, sizeof(map_error)))
				fprintf(stderr, "[demo] map rebuild for %s failed: %s\n", demo_map, map_error);
		}
	}

	qnn_runtime.tick = 0;
	qnn_runtime.steps = 0;
	qnn_runtime.has_reset = true;
	qnn_runtime.done = false;
	QNN_IOInit(&qnn_map_state);
	/* Store init is now called inside QNN_ObjectInit */
	return true;
}

static void QNN_WriteHelloResponse(void)
{
	fprintf(stdout, "{\"capabilities\":[\"demo_playback\",\"navmesh_query_v1\",\"obs_buffer_collect_v1\"],\"map_id\":");
	QNN_WriteJsonString(stdout, qnn_map_state.requested_map_id);
	fprintf(stdout, ",\"ok\":true,\"protocol_version\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_PROTOCOL);
	fprintf(stdout, ",\"server\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_SERVER_NAME);
	fprintf(stdout, ",\"tick_hz\":%d,\"worker_build\":{\"basedir\":", qnn_runtime.fixed_tick_hz);
	QNN_WriteJsonString(stdout, basedir);
	fprintf(stdout, ",\"upstream_commit\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_UPSTREAM_COMMIT);
	fprintf(stdout, "}}\n");
	fflush(stdout);
}

static int QNN_HandleHello(const char *line)
{
	char map_id[QNN_MAX_MAP_ID];
	char error[256];

	memset(map_id, 0, sizeof(map_id));
	memset(error, 0, sizeof(error));
	if (!QNN_JsonExtractString(line, "\"map_id\"", map_id, sizeof(map_id)))
		snprintf(map_id, sizeof(map_id), "E1M1");

	qnn_runtime.fixed_tick_hz = QNN_JsonExtractInt(line, "\"tick_hz\"", qnn_runtime.fixed_tick_hz > 0 ? qnn_runtime.fixed_tick_hz : 20);
	if (qnn_runtime.fixed_tick_hz <= 0)
	{
		qnn_runtime.auto_detect_tick_hz = true;
		qnn_runtime.fixed_tick_hz = 20;
	}
	else
	{
		qnn_runtime.auto_detect_tick_hz = false;
	}
	qnn_runtime.fixed_dt = 1.0f / (float)qnn_runtime.fixed_tick_hz;

	/* Resample target Hz: stored for use after auto-detect during reset.
	 * >0 = fixed target, 0 = use auto-detected source rate. */
	qnn_runtime.resample_hz = QNN_JsonExtractInt(line, "\"resample_hz\"", 0);

	if (!QNN_PrepareMap(map_id, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	QNN_WriteHelloResponse();
	return 0;
}

static int QNN_HandleReset(const char *line)
{
	qnn_snapshot_t snapshot;
	char demo_path[MAX_OSPATH];
	char error[256];
	int seed;

	memset(demo_path, 0, sizeof(demo_path));
	memset(error, 0, sizeof(error));
	/* Interactive reset/step removed — use collect for batch processing. */
	(void)line;
	QNN_WriteError("reset not supported — use collect");
	return 0;
}

static int QNN_HandleCollect(const char *line)
{
	qnn_snapshot_t snapshot;
	char demo_path[MAX_OSPATH];
	char error[256];
	int seed;

	memset(demo_path, 0, sizeof(demo_path));
	memset(error, 0, sizeof(error));
	seed = QNN_JsonExtractInt(line, "\"seed\"", -1);
	{
		int trim = QNN_JsonExtractInt(line, "\"trim_match\"", 0);
		qnn_match_state = trim ? 0 : 1; /* state 1 = emitting immediately */
	}
	if (!QNN_JsonExtractString(line, "\"demo_path\"", demo_path, sizeof(demo_path)))
	{
		QNN_WriteError("reset options must include demo_path");
		return 0;
	}
	if (!QNN_ResetWorldLocal(demo_path, seed, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	/* Set up minimal server state for velocity-based move labeling. */
	QNN_PhysInit();

	/* Init resampler. resample_hz=0 means pass-through (emit every frame
	 * at native recording FPS). resample_hz>0 means downsample to target. */
	QNN_ResampleInit(qnn_runtime.resample_hz);
	if (qnn_runtime.resample_hz > 0)
		fprintf(stderr, "[demo] resample target: %d Hz (source: %d Hz)\n",
			qnn_runtime.resample_hz, qnn_runtime.fixed_tick_hz);

	{
		int emit_hz = qnn_resample.target_hz > 0
			? qnn_resample.target_hz : qnn_runtime.fixed_tick_hz;

		QNN_CaptureSnapshotLocal(&snapshot, true);
		QNN_SavePrev(&snapshot, 0.0f);  /* first tick, no dt */
		QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, true);

		/*
		 * Match trimming: if svc_print contains match start/end text,
		 * only emit ticks during the match (state 1). If no match text
		 * is ever seen, emit all ticks (no trimming).
		 *
		 * Pass 1: run the demo, buffer ticks to a temp file if in
		 * pre-match state. This is too complex — instead we do two
		 * passes or just accept that demos without match text emit
		 * everything.
		 *
		 * Simple approach: skip pre-match ticks. If the demo ends
		 * still in state 0 (no match text found), re-run without
		 * trimming. This is wasteful but correct and rare — only
		 * 16 demos have no match text and they don't need trimming.
		 */
		{
			qboolean emitting = false;

		while (!qnn_runtime.done)
		{
			Host_Frame(qnn_runtime.fixed_dt);
			qnn_runtime.tick += 1;
			qnn_runtime.steps += 1;
			if (!cls.demoplayback || cls.state == ca_disconnected)
				qnn_runtime.done = true;

			/* Match ended — stop collecting. */
			if (qnn_match_state == 2)
			{
				snapshot.done = true;
				qnn_runtime.done = true;
			}

			QNN_CaptureSnapshotLocal(&snapshot, false);
			QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, false);

			/* Start emitting when match begins. */
			if (!emitting && qnn_match_state == 1)
			{
				emitting = true;
				fprintf(stderr, "[demo] emitting from tick %d\n", qnn_runtime.tick);
			}

			/* Pre-match or no match text yet: skip ticks. */
			if (!emitting && !qnn_runtime.done)
			{
				QNN_SavePrev(&snapshot, qnn_runtime.fixed_dt);
				continue;
			}

			/* Store update now runs inside QNN_IOUpdate → QNN_ObjectUpdate */
			if (qnn_runtime.store_dump != NULL)
			{
				QNN_StoreDumpSounds(qnn_runtime.store_dump, qnn_runtime.tick, &snapshot);
				QNN_StoreDumpTick(qnn_runtime.store_dump, qnn_runtime.tick, (float)cl.mtime[0]);
			}

			QNN_ResampleAccumulate(&snapshot, qnn_runtime.fixed_dt);
			if (QNN_ResampleShouldEmit() || qnn_runtime.done)
			{
				if (!snapshot.done)
					QNN_InferAction(&snapshot.action_label, &snapshot);
				else
					QNN_ClearAction(&snapshot.action_label);
				QNN_ResampleApplyActionMerge(&snapshot);
				QNN_WriteObsTick(stdout, &snapshot, qnn_runtime.tick,
					qnn_runtime.steps, emit_hz, false);
				QNN_SavePrev(&snapshot, qnn_runtime.fixed_dt);
			}
			else if (qnn_resample.target_hz <= 0 && !snapshot.done)
			{
				QNN_InferAction(&snapshot.action_label, &snapshot);
				QNN_SavePrev(&snapshot, qnn_runtime.fixed_dt);
			}
		}

		/* If no match text was ever seen, emit a done tick so the
		   reader doesn't block. The Python side will see 0 real
		   ticks for trimmed spectator demos, or collect untrimmed
		   via a second pass for demos without match text. */
		if (!emitting)
		{
			snapshot.done = true;
			QNN_WriteObsTick(stdout, &snapshot, qnn_runtime.tick,
				qnn_runtime.steps, emit_hz, true);
			fprintf(stderr, "[demo] no match text found, emitted done marker only\n");
		}
		}
	}
	fflush(stdout);
	if (qnn_runtime.store_dump != NULL)
	{
		fflush(qnn_runtime.store_dump);
		fclose(qnn_runtime.store_dump);
		qnn_runtime.store_dump = NULL;
	}
	return 0;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	char line[QNN_WORKER_MAX_LINE];

	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	QNN_RuntimeReset();
	QNN_ClearAction(&qnn_pending_action);
	memset(&parms, 0, sizeof(parms));
	COM_InitArgv(argc, argv);
	parms.argc = com_argc;
	parms.argv = com_argv;
	parms.memsize = 32 * 1024 * 1024;
	parms.membase = malloc(parms.memsize);
	parms.basedir = basedir;
	Host_Init(&parms);
	/* Flush startdemos from quake.rc and tear down any auto-demo */
	Cbuf_Execute();
	cls.demonum = -1;
	CL_Disconnect();
	cls.state = ca_disconnected;

	while (fgets(line, sizeof(line), stdin) != NULL)
	{
		if (strstr(line, "\"op\"") != NULL && strstr(line, "hello") != NULL)
		{
			QNN_HandleHello(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "nav_query") != NULL)
		{
			QNN_HandleNavQuery(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "collect") != NULL)
		{
			QNN_HandleCollect(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL)
		{
			fprintf(stdout, "{\"ok\":true}\n");
			fflush(stdout);
			QNN_FreeMapState(&qnn_map_state);
			Host_Shutdown();
			return 0;
		}
	}

	QNN_FreeMapState(&qnn_map_state);
	Host_Shutdown();
	return 0;
}
