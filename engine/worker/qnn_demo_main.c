#include "qnn.h"
#include "qnn_obs.h"
#include "qnn_metrics.h"

#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_PROTOCOL "v6"
#define QNN_WORKER_SERVER_NAME "quake-demo-worker"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_COMMAND_TEXT 1024

#define QNN_DEMO_MOUSE_DEGREES_PER_COUNT 0.066f
#define QNN_DEMO_MOVEMENT_THRESHOLD_DEFAULT 4.0f
#define QNN_DEMO_JUMP_VELOCITY_THRESHOLD 160.0f
#define QNN_DEMO_BASE_DT (1.0f / 20.0f)

typedef struct
{
	int	fixed_tick_hz;
	float	fixed_dt;
	qboolean auto_detect_tick_hz;
	int	resample_hz;	/* requested resample target; 0 = use detected rate */
	float	movement_threshold;	/* position delta threshold for move labels */
	int	seed;
	int	tick;
	int	steps;
	qboolean has_reset;
	qboolean done;
	vec3_t	prev_origin;
	vec3_t	prev_view_angles;
	vec3_t	prev_velocity;
	int	prev_weapon_id;
	int	prev_ammo;
	qboolean has_prev;
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

static float QNN_MovementThresholdForDt(float dt)
{
	float step_dt;

	step_dt = dt > 0.0f ? dt : QNN_DEMO_BASE_DT;
	return qnn_runtime.movement_threshold * (step_dt / QNN_DEMO_BASE_DT);
}

static void QNN_InferAction(qnn_action_t *action, const qnn_snapshot_t *snapshot)
{
	float yaw_delta;
	float pitch_delta;
	float forward_x;
	float forward_y;
	float left_x;
	float left_y;
	float dx;
	float dy;
	float forward_proj;
	float left_proj;
	float yaw_rad;
	float movement_threshold;
	int mouse_yaw;
	int mouse_pitch;

	QNN_ClearAction(action);

	if (!qnn_runtime.has_prev)
		return;

	yaw_delta = QNN_NormalizeYaw(snapshot->player_view_angles[1] - qnn_runtime.prev_view_angles[1]);
	pitch_delta = snapshot->player_view_angles[0] - qnn_runtime.prev_view_angles[0];

	/* Accumulate look deltas for resampling (degrees, converted at emission). */
	QNN_ResampleAccumulateLook(yaw_delta, pitch_delta);

	mouse_yaw = (int)roundf(-yaw_delta / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
	mouse_pitch = (int)roundf(pitch_delta / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
	action->look[0] = QNN_LookAxisFromMouseCount(mouse_yaw);
	action->look[1] = QNN_LookAxisFromMouseCount(mouse_pitch);

	yaw_rad = (float)(qnn_runtime.prev_view_angles[1] * M_PI / 180.0);
	forward_x = cosf(yaw_rad);
	forward_y = sinf(yaw_rad);
	left_x = -forward_y;
	left_y = forward_x;

	dx = snapshot->player_origin[0] - qnn_runtime.prev_origin[0];
	dy = snapshot->player_origin[1] - qnn_runtime.prev_origin[1];
	forward_proj = dx * forward_x + dy * forward_y;
	left_proj = dx * left_x + dy * left_y;
	movement_threshold = QNN_MovementThresholdForDt(
		qnn_resample.target_hz > 0 ? qnn_resample.target_dt : qnn_runtime.fixed_dt);

	if (forward_proj > movement_threshold)
		action->move[0] = 1.0f;
	else if (forward_proj < -movement_threshold)
		action->move[0] = -1.0f;

	if (left_proj > movement_threshold)
		action->move[1] = -1.0f;
	else if (left_proj < -movement_threshold)
		action->move[1] = 1.0f;

	action->fire = (snapshot->ammo < qnn_runtime.prev_ammo) ? 1 : 0;

	if (qnn_runtime.prev_velocity[2] <= 32.0f && snapshot->player_velocity[2] >= QNN_DEMO_JUMP_VELOCITY_THRESHOLD)
		action->jump = 1;

	if (snapshot->weapon_id != qnn_runtime.prev_weapon_id && snapshot->weapon_id > 0)
		action->switch_slot = QNN_SwitchSlotFromWeaponId(snapshot->weapon_id);
}

/* Write a single tick in obs_buffer format: obs + action label + framing. */
static void QNN_WriteObsTick(FILE *out, const qnn_snapshot_t *snapshot,
	int tick, int steps, int tick_hz, qboolean reset_flag)
{
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];
	uint8_t header[16];
	uint16_t flags = 0;

	QNN_PackObsBuffer(obs, snapshot, tick_hz, reset_flag);

	if (reset_flag) flags |= 0x01;
	if (snapshot->done) flags |= 0x02;

	/* Header: tick(4) + steps(4) + tick_hz(4) + flags(2) + action_size(2) */
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
	fwrite(&snapshot->action_label, 1, sizeof(qnn_action_t), out);
	fflush(out);
}

static void QNN_SavePrev(const qnn_snapshot_t *snapshot)
{
	VectorCopy(snapshot->player_origin, qnn_runtime.prev_origin);
	VectorCopy(snapshot->player_view_angles, qnn_runtime.prev_view_angles);
	VectorCopy(snapshot->player_velocity, qnn_runtime.prev_velocity);
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
	qnn_runtime.movement_threshold = QNN_DEMO_MOVEMENT_THRESHOLD_DEFAULT;
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
	QNN_CaptureVisibleEntities(snapshot, qnn_runtime.fixed_dt);
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

	qnn_runtime.tick = 0;
	qnn_runtime.steps = 0;
	qnn_runtime.has_reset = true;
	qnn_runtime.done = false;
	QNN_SemanticReset(&qnn_map_state);
	return true;
}

static void QNN_WriteHelloResponse(void)
{
	fprintf(stdout, "{\"capabilities\":[\"demo_playback\",\"navmesh_query_v1\",\"obs_buffer_collect_v1\"],\"map_id\":");
	QNN_WriteJsonString(stdout, qnn_map_state.requested_map_id);
	fprintf(stdout, ",\"map_state\":");
	QNN_WriteMapStateJson(stdout, &qnn_map_state);
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

	{
		float mt = (float)QNN_JsonExtractInt(line, "\"movement_threshold\"", 0);
		if (mt > 0.0f)
			qnn_runtime.movement_threshold = mt;
	}

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
		QNN_SavePrev(&snapshot);
		QNN_SemanticUpdate(&qnn_map_state, &snapshot, qnn_runtime.fixed_dt, true);
		QNN_WriteObsTick(stdout, &snapshot, qnn_runtime.tick,
			qnn_runtime.steps, emit_hz, true);

		while (!qnn_runtime.done)
		{
			Host_Frame(qnn_runtime.fixed_dt);
			qnn_runtime.tick += 1;
			qnn_runtime.steps += 1;
			if (!cls.demoplayback || cls.state == ca_disconnected)
				qnn_runtime.done = true;
			QNN_CaptureSnapshotLocal(&snapshot, false);
			QNN_SemanticUpdate(&qnn_map_state, &snapshot, qnn_runtime.fixed_dt, false);

			QNN_ResampleAccumulate(&snapshot, qnn_runtime.fixed_dt);
			if (QNN_ResampleShouldEmit() || qnn_runtime.done)
			{
				QNN_InferAction(&snapshot.action_label, &snapshot);
				QNN_ResampleApplyActionMerge(&snapshot);
				QNN_WriteObsTick(stdout, &snapshot, qnn_runtime.tick,
					qnn_runtime.steps, emit_hz, false);
				QNN_SavePrev(&snapshot);
			}
			else if (qnn_resample.target_hz <= 0)
			{
				QNN_InferAction(&snapshot.action_label, &snapshot);
				QNN_SavePrev(&snapshot);
			}
		}
	}
	fflush(stdout);
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
