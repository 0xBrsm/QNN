#include "qnn_worker.h"

#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_PROTOCOL "v5"
#define QNN_WORKER_SERVER_NAME "quake-demo-worker"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_COMMAND_TEXT 1024

#define QNN_DEMO_MOUSE_DEGREES_PER_COUNT 0.066f
#define QNN_DEMO_MOVEMENT_THRESHOLD 4.0f
#define QNN_DEMO_JUMP_VELOCITY_THRESHOLD 160.0f
#define QNN_DEMO_BASE_DT (1.0f / 20.0f)

static const int qnn_demo_look_bins[] = {
	-128, -96, -72, -56, -40, -28, -20, -14, -10, -6, -3, -1,
	0,
	1, 3, 6, 10, 14, 20, 28, 40, 56, 72, 96, 128
};
#define QNN_DEMO_LOOK_BIN_COUNT (sizeof(qnn_demo_look_bins) / sizeof(qnn_demo_look_bins[0]))
#define QNN_DEMO_LOOK_NEUTRAL 12

typedef struct
{
	int	fixed_tick_hz;
	float	fixed_dt;
	qboolean auto_detect_tick_hz;
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
} qnn_demo_runtime_t;

static qnn_demo_runtime_t qnn_demo_runtime;

static qboolean qnn_demo_client_ready(void)
{
	return (cls.demoplayback
		&& cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static int qnn_demo_nearest_look_bin(int mouse_count)
{
	int best_index;
	int best_dist;
	int i;

	best_index = QNN_DEMO_LOOK_NEUTRAL;
	best_dist = abs(qnn_demo_look_bins[QNN_DEMO_LOOK_NEUTRAL] - mouse_count);
	for (i = 0; i < (int)QNN_DEMO_LOOK_BIN_COUNT; ++i)
	{
		int dist;

		dist = abs(qnn_demo_look_bins[i] - mouse_count);
		if (dist < best_dist)
		{
			best_dist = dist;
			best_index = i;
		}
	}
	return best_index;
}

static float qnn_demo_normalize_yaw(float delta)
{
	while (delta <= -180.0f)
		delta += 360.0f;
	while (delta > 180.0f)
		delta -= 360.0f;
	return delta;
}

static float qnn_demo_movement_threshold_for_dt(float dt)
{
	float step_dt;

	step_dt = dt > 0.0f ? dt : QNN_DEMO_BASE_DT;
	return QNN_DEMO_MOVEMENT_THRESHOLD * (step_dt / QNN_DEMO_BASE_DT);
}

static void qnn_demo_infer_action(qnn_worker_action_t *action, const qnn_worker_snapshot_t *snapshot)
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

	qnn_worker_clear_action(action);
	action->look_yaw = QNN_DEMO_LOOK_NEUTRAL;
	action->look_pitch = QNN_DEMO_LOOK_NEUTRAL;

	if (!qnn_demo_runtime.has_prev)
		return;

	yaw_delta = qnn_demo_normalize_yaw(snapshot->player_view_angles[1] - qnn_demo_runtime.prev_view_angles[1]);
	pitch_delta = snapshot->player_view_angles[0] - qnn_demo_runtime.prev_view_angles[0];

	mouse_yaw = (int)roundf(-yaw_delta / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
	mouse_pitch = (int)roundf(pitch_delta / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
	action->look_yaw = qnn_demo_nearest_look_bin(mouse_yaw);
	action->look_pitch = qnn_demo_nearest_look_bin(mouse_pitch);

	yaw_rad = (float)(qnn_demo_runtime.prev_view_angles[1] * M_PI / 180.0);
	forward_x = cosf(yaw_rad);
	forward_y = sinf(yaw_rad);
	left_x = -forward_y;
	left_y = forward_x;

	dx = snapshot->player_origin[0] - qnn_demo_runtime.prev_origin[0];
	dy = snapshot->player_origin[1] - qnn_demo_runtime.prev_origin[1];
	forward_proj = dx * forward_x + dy * forward_y;
	left_proj = dx * left_x + dy * left_y;
	movement_threshold = qnn_demo_movement_threshold_for_dt(qnn_demo_runtime.fixed_dt);

	if (forward_proj > movement_threshold)
		action->move = 1;
	else if (forward_proj < -movement_threshold)
		action->move = 2;

	if (left_proj > movement_threshold)
		action->strafe = 1;
	else if (left_proj < -movement_threshold)
		action->strafe = 2;

	action->fire = (snapshot->ammo < qnn_demo_runtime.prev_ammo) ? 1 : 0;

	if (qnn_demo_runtime.prev_velocity[2] <= 32.0f && snapshot->player_velocity[2] >= QNN_DEMO_JUMP_VELOCITY_THRESHOLD)
		action->jump = 1;

	if (snapshot->weapon_id != qnn_demo_runtime.prev_weapon_id && snapshot->weapon_id > 0)
		action->weapon = qnn_weapon_class_from_id(snapshot->weapon_id) + 1;
}

static void qnn_demo_save_prev(const qnn_worker_snapshot_t *snapshot)
{
	VectorCopy(snapshot->player_origin, qnn_demo_runtime.prev_origin);
	VectorCopy(snapshot->player_view_angles, qnn_demo_runtime.prev_view_angles);
	VectorCopy(snapshot->player_velocity, qnn_demo_runtime.prev_velocity);
	qnn_demo_runtime.prev_weapon_id = snapshot->weapon_id;
	qnn_demo_runtime.prev_ammo = snapshot->ammo;
	qnn_demo_runtime.has_prev = true;
}

static void qnn_demo_runtime_reset(void)
{
	memset(&qnn_demo_runtime, 0, sizeof(qnn_demo_runtime));
	qnn_demo_runtime.fixed_tick_hz = 20;
	qnn_demo_runtime.fixed_dt = 1.0f / 20.0f;
	qnn_demo_runtime.auto_detect_tick_hz = false;
}

#define QNN_DEMO_DETECT_PROBE_DT 0.001f
#define QNN_DEMO_DETECT_MAX_FRAMES 8192

static void qnn_demo_detect_native_tick_hz(void)
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

	qnn_demo_runtime.fixed_tick_hz = detected_hz;
	qnn_demo_runtime.fixed_dt = 1.0f / (float)detected_hz;
	fprintf(stderr, "[demo] detected native tick rate: %d Hz (dt=%.4fs)\n",
		detected_hz, qnn_demo_runtime.fixed_dt);
}

static void qnn_worker_capture_snapshot(qnn_worker_snapshot_t *snapshot, qboolean reset_flag)
{
	qnn_worker_capture_base_snapshot(snapshot);
	snapshot->done = reset_flag ? false : qnn_demo_runtime.done;
	qnn_worker_capture_visible_entities(snapshot, qnn_demo_runtime.fixed_dt);
	qnn_worker_drain_sounds(snapshot);
}

static qboolean qnn_demo_reset_world(const char *demo_path, int seed, char *error, size_t error_size)
{
	char command[QNN_WORKER_MAX_COMMAND_TEXT];
	int frame;

	if (qnn_worker_map_state.map_name[0] == 0)
	{
		snprintf(error, error_size, "Call hello first so the worker knows which map to load");
		return false;
	}
	if (demo_path == NULL || demo_path[0] == 0)
	{
		snprintf(error, error_size, "reset options must include demo_path");
		return false;
	}

	qnn_worker_clear_action(&qnn_worker_pending_action);
	qnn_demo_runtime.seed = seed >= 0 ? seed : 0;
	qnn_demo_runtime.done = false;
	qnn_demo_runtime.has_reset = false;
	qnn_worker_sound_count = 0;
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
		Host_Frame(qnn_demo_runtime.fixed_dt);
		if (qnn_demo_client_ready())
			break;
		if (frame > 0 && !cls.demoplayback && cls.state == ca_disconnected)
			break;
	}
	if (!qnn_demo_client_ready())
	{
		fprintf(stderr, "[demo] playdemo failed: demoplayback=%d state=%d signon=%d\n",
			cls.demoplayback, cls.state, cls.signon);
		snprintf(error, error_size, "Timed out waiting for demo playback on %s", demo_path);
		return false;
	}

	if (qnn_demo_runtime.auto_detect_tick_hz)
		qnn_demo_detect_native_tick_hz();

	qnn_demo_runtime.tick = 0;
	qnn_demo_runtime.steps = 0;
	qnn_demo_runtime.has_reset = true;
	qnn_demo_runtime.done = false;
	qnn_worker_semantic_reset(&qnn_worker_map_state);
	return true;
}

static void qnn_worker_write_hello_response(void)
{
	fprintf(stdout, "{\"capabilities\":[\"demo_playback\",\"navmesh_query_v1\",\"reset_options\",\"token_collect_v1\",\"token_step_v2\"],\"map_id\":");
	qnn_worker_write_json_string(stdout, qnn_worker_map_state.requested_map_id);
	fprintf(stdout, ",\"map_state\":");
	qnn_worker_write_map_state_json(stdout, &qnn_worker_map_state);
	fprintf(stdout, ",\"ok\":true,\"protocol_version\":");
	qnn_worker_write_json_string(stdout, QNN_WORKER_PROTOCOL);
	fprintf(stdout, ",\"server\":");
	qnn_worker_write_json_string(stdout, QNN_WORKER_SERVER_NAME);
	fprintf(stdout, ",\"tick_hz\":%d,\"worker_build\":{\"basedir\":", qnn_demo_runtime.fixed_tick_hz);
	qnn_worker_write_json_string(stdout, basedir);
	fprintf(stdout, ",\"upstream_commit\":");
	qnn_worker_write_json_string(stdout, QNN_WORKER_UPSTREAM_COMMIT);
	fprintf(stdout, "}}\n");
	fflush(stdout);
}

static int qnn_worker_handle_hello(const char *line)
{
	char map_id[QNN_WORKER_MAX_MAP_ID];
	char error[256];

	memset(map_id, 0, sizeof(map_id));
	memset(error, 0, sizeof(error));
	if (!qnn_json_extract_string(line, "\"map_id\"", map_id, sizeof(map_id)))
		snprintf(map_id, sizeof(map_id), "E1M1");

	qnn_demo_runtime.fixed_tick_hz = qnn_json_extract_int(line, "\"tick_hz\"", qnn_demo_runtime.fixed_tick_hz > 0 ? qnn_demo_runtime.fixed_tick_hz : 20);
	if (qnn_demo_runtime.fixed_tick_hz <= 0)
	{
		qnn_demo_runtime.auto_detect_tick_hz = true;
		qnn_demo_runtime.fixed_tick_hz = 20;
	}
	else
	{
		qnn_demo_runtime.auto_detect_tick_hz = false;
	}
	qnn_demo_runtime.fixed_dt = 1.0f / (float)qnn_demo_runtime.fixed_tick_hz;

	if (!qnn_worker_prepare_map(map_id, error, sizeof(error)))
	{
		qnn_worker_write_error(error);
		return 0;
	}

	qnn_worker_write_hello_response();
	return 0;
}

static int qnn_worker_handle_reset(const char *line)
{
	qnn_worker_snapshot_t snapshot;
	char demo_path[MAX_OSPATH];
	char error[256];
	int seed;

	memset(demo_path, 0, sizeof(demo_path));
	memset(error, 0, sizeof(error));
	seed = qnn_json_extract_int(line, "\"seed\"", -1);
	if (!qnn_json_extract_string(line, "\"demo_path\"", demo_path, sizeof(demo_path)))
	{
		qnn_worker_write_error("reset options must include demo_path");
		return 0;
	}
	if (!qnn_demo_reset_world(demo_path, seed, error, sizeof(error)))
	{
		qnn_worker_write_error(error);
		return 0;
	}

	qnn_worker_capture_snapshot(&snapshot, true);
	qnn_demo_save_prev(&snapshot);
	qnn_worker_semantic_update(&qnn_worker_map_state, &snapshot, qnn_demo_runtime.fixed_dt, true);
	qnn_worker_write_token_step_binary(stdout, &snapshot, qnn_demo_runtime.tick, qnn_demo_runtime.steps, qnn_demo_runtime.fixed_tick_hz, true);
	return 0;
}

static int qnn_worker_handle_step(const char *line)
{
	qnn_worker_snapshot_t snapshot;

	(void)line;

	if (!qnn_demo_runtime.has_reset)
	{
		qnn_worker_write_error("Call reset before step");
		return 0;
	}

	qnn_worker_clear_action(&qnn_worker_pending_action);
	if (!qnn_demo_runtime.done)
	{
		Host_Frame(qnn_demo_runtime.fixed_dt);
		qnn_demo_runtime.tick += 1;
		qnn_demo_runtime.steps += 1;
		if (!cls.demoplayback || cls.state == ca_disconnected)
			qnn_demo_runtime.done = true;
	}
	qnn_worker_capture_snapshot(&snapshot, false);
	qnn_demo_infer_action(&snapshot.action_label, &snapshot);
	qnn_demo_save_prev(&snapshot);
	qnn_worker_semantic_update(&qnn_worker_map_state, &snapshot, qnn_demo_runtime.fixed_dt, false);
	qnn_worker_write_token_step_binary(stdout, &snapshot, qnn_demo_runtime.tick, qnn_demo_runtime.steps, qnn_demo_runtime.fixed_tick_hz, false);
	return 0;
}

static int qnn_worker_handle_collect(const char *line)
{
	qnn_worker_snapshot_t snapshot;
	char demo_path[MAX_OSPATH];
	char error[256];
	int seed;

	memset(demo_path, 0, sizeof(demo_path));
	memset(error, 0, sizeof(error));
	seed = qnn_json_extract_int(line, "\"seed\"", -1);
	if (!qnn_json_extract_string(line, "\"demo_path\"", demo_path, sizeof(demo_path)))
	{
		qnn_worker_write_error("reset options must include demo_path");
		return 0;
	}
	if (!qnn_demo_reset_world(demo_path, seed, error, sizeof(error)))
	{
		qnn_worker_write_error(error);
		return 0;
	}

	qnn_worker_capture_snapshot(&snapshot, true);
	qnn_demo_save_prev(&snapshot);
	qnn_worker_semantic_update(&qnn_worker_map_state, &snapshot, qnn_demo_runtime.fixed_dt, true);
	qnn_worker_write_token_step_binary(stdout, &snapshot, qnn_demo_runtime.tick, qnn_demo_runtime.steps, qnn_demo_runtime.fixed_tick_hz, true);

	while (!qnn_demo_runtime.done)
	{
		Host_Frame(qnn_demo_runtime.fixed_dt);
		qnn_demo_runtime.tick += 1;
		qnn_demo_runtime.steps += 1;
		if (!cls.demoplayback || cls.state == ca_disconnected)
			qnn_demo_runtime.done = true;
		qnn_worker_capture_snapshot(&snapshot, false);
		qnn_demo_infer_action(&snapshot.action_label, &snapshot);
		qnn_demo_save_prev(&snapshot);
		qnn_worker_semantic_update(&qnn_worker_map_state, &snapshot, qnn_demo_runtime.fixed_dt, false);
		qnn_worker_write_token_step_binary(stdout, &snapshot, qnn_demo_runtime.tick, qnn_demo_runtime.steps, qnn_demo_runtime.fixed_tick_hz, false);
	}
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	char line[QNN_WORKER_MAX_LINE];

	qnn_worker_resolve_basedir(qnn_worker_basedir_storage, sizeof(qnn_worker_basedir_storage));
	qnn_demo_runtime_reset();
	qnn_worker_clear_action(&qnn_worker_pending_action);
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
			qnn_worker_handle_hello(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "reset") != NULL)
		{
			qnn_worker_handle_reset(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "step") != NULL)
		{
			qnn_worker_handle_step(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "nav_query") != NULL)
		{
			qnn_worker_handle_nav_query(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "collect") != NULL)
		{
			qnn_worker_handle_collect(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL)
		{
			fprintf(stdout, "{\"ok\":true}\n");
			fflush(stdout);
			qnn_worker_free_map_state(&qnn_worker_map_state);
			Host_Shutdown();
			return 0;
		}
	}

	qnn_worker_free_map_state(&qnn_worker_map_state);
	Host_Shutdown();
	return 0;
}
