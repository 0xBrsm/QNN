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

/* ── Match state — defined in common/qnn_match.c ───────────────── */
extern int qnn_match_state;
extern const char *qnn_match_log_tag;

/* Per-native-frame buffers for entity interaction (movers, players). */
#define QNN_MAX_NATIVE_FRAMES 16

/* trigger_push: max tracked push triggers. */
#define QNN_MAX_PUSH_TRIGGERS 8

/* Emission-level frame filters. */
#define QNN_GOD_MODE_HEALTH     250
#define QNN_DEAD_MAX_EMIT       20   /* 1 s at 20 Hz */
#define QNN_FROZEN_MAX_EMIT     40   /* 2 s at 20 Hz */

typedef struct
{
	int	fixed_tick_hz;
	float	fixed_dt;
	int	resample_hz;	/* requested resample target; 0 = use detected rate */
	int	tick;
	int	steps;
	qboolean has_reset;
	qboolean done;
	vec3_t	prev_origin;
	vec3_t	prev_velocity;	/* reconstructed from position deltas */
	int	prev_ammo;
	qboolean has_prev;
	vec3_t	emit_view_angles;
	vec3_t	emit_origin;
	vec3_t	emit_velocity;
	qboolean emit_grounded;
	int	emit_waterlevel;
	int	emit_weapon_id;
	qboolean has_emit_anchor;
	int	native_frame_count;
	qboolean phys_grounded;
	/* Mover tracking for per-frame entity interaction. */
	int	mover_entity_nums[QNN_MAX_PHYS_MOVERS];
	int	mover_model_indices[QNN_MAX_PHYS_MOVERS];
	int	mover_count;
	vec3_t	mover_emit_origins[QNN_MAX_PHYS_MOVERS];
	vec3_t	mover_origins[QNN_MAX_NATIVE_FRAMES][QNN_MAX_PHYS_MOVERS];
	/* Other player positions per native frame for body-block collision. */
	vec3_t	player_origins[QNN_MAX_NATIVE_FRAMES][QNN_MAX_PHYS_PLAYERS];
	int	player_entity_nums[QNN_MAX_PHYS_PLAYERS];
	int	player_count;
	/* trigger_push tracking. */
	int	push_model_indices[QNN_MAX_PUSH_TRIGGERS];
	vec3_t	push_velocities[QNN_MAX_PUSH_TRIGGERS]; /* speed * direction */
	int	push_count;
	float	prev_move[3];	/* last inferred move — carried forward for zero-frame emissions */
	int	prev_fwd_sign;	/* last candidate forward sign for continuity bias */
	int	prev_strafe_sign;
	FILE	*store_dump;	/* NULL = no dump, set from QNN_STORE_DUMP env var */
	/* Buffered obs for one-tick delay: we emit obs(t) paired with
	   the action computed at t+1 (the transition FROM state t). */
	qnn_tick_emit_state_t tick_emit;
	/* Frame filter counters (emission-rate) */
	int	dead_emit_count;
	int	frozen_emit_count;
} qnn_runtime_t;

static qnn_runtime_t qnn_runtime;

/* Build mover/push/player refs into the runtime via shared scanners
 * in common/qnn_collect_helpers.c. */
static void QNN_BuildAllRefs(void)
{
	qnn_runtime.mover_count = QNN_BuildMoverRefs(
		qnn_runtime.mover_entity_nums,
		qnn_runtime.mover_model_indices,
		QNN_MAX_PHYS_MOVERS);
	qnn_runtime.push_count = QNN_BuildPushRefs(
		qnn_runtime.push_model_indices,
		qnn_runtime.push_velocities,
		QNN_MAX_PUSH_TRIGGERS);
	qnn_runtime.player_count = QNN_BuildPlayerRefs(
		cl.viewentity,
		qnn_runtime.player_entity_nums,
		QNN_MAX_PHYS_PLAYERS);
	if (qnn_runtime.mover_count > 0)
		fprintf(stderr, "[demo] tracking %d movers for physics baseline\n",
			qnn_runtime.mover_count);
	if (qnn_runtime.push_count > 0)
		fprintf(stderr, "[demo] tracking %d push triggers for velocity subtraction\n",
			qnn_runtime.push_count);
	if (qnn_runtime.player_count > 0)
		fprintf(stderr, "[demo] tracking %d other players for body-block collision\n",
			qnn_runtime.player_count);
}

static qboolean QNN_ClientReady(void)
{
	return (cls.demoplayback
		&& cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static void QNN_InferEmitAction(qnn_action_t *action, const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);

	if (!qnn_runtime.has_emit_anchor)
		return;

	/* View-relative look label: record the NEXT frame's forward direction
	   in the PREVIOUS frame's view-relative coordinates.  This captures
	   the turn delta as a direction vector the model can learn to emit.
	   When the player isn't turning, this is (1,0,0) (straight ahead). */
	{
		vec3_t forward, right, up, cur_forward;
		AngleVectors(qnn_runtime.emit_view_angles, forward, right, up);
		QNN_ForwardFromAngles(snapshot->player_view_angles, cur_forward);
		action->look[0] = DotProduct(cur_forward, forward);
		action->look[1] = DotProduct(cur_forward, right);
		action->look[2] = DotProduct(cur_forward, up);
	}

	if (snapshot->weapon_id != qnn_runtime.emit_weapon_id && snapshot->weapon_id > 0)
		action->switch_slot = QNN_SwitchSlotFromWeaponId(snapshot->weapon_id);
}

/* Per-native-frame: infer fire from sound/ammo cues. */
static void QNN_InferNativeAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);

	if (!qnn_runtime.has_prev)
		return;

	action->fire = (QNN_SnapshotHasSelfWeaponFireSound(snapshot)
		|| snapshot->ammo < qnn_runtime.prev_ammo) ? 1 : 0;
}

/* Per-emission: determine the player's input by simulating all 9 key
 * combinations and picking the one whose BSP-clipped endpoint best
 * matches the observed position.  All candidates start from the same
 * state and go through the same geometry — no path divergence.
 *
 * Output is snapped to the 9 legal keyboard directions via QNN_SnapMove.
 * Z is handled separately: sound-based jump on ground/air, inversion
 * snap in water. */
static void QNN_InferEmitMove(qnn_action_t *action,
	const qnn_snapshot_t *snapshot, float emit_dt)
{
	int fwd_sign, strafe_sign;
	float raw[3];
	int medium, i;

	if (!qnn_runtime.has_emit_anchor || emit_dt <= 0.0f)
		return;

	/* Need BSP hull for candidate simulation. */
	if (sv.worldmodel == NULL || sv.worldmodel->hulls[1].firstclipnode < 0)
	{
		/* Fallback: raw delta → snap. */
		vec3_t delta, rel;
		for (i = 0; i < 3; i++)
			delta[i] = (snapshot->player_origin[i] - qnn_runtime.emit_origin[i]) / emit_dt;
		QNN_RelativeFrame(qnn_runtime.emit_view_angles, delta, rel);
		raw[0] = rel[0] / QNN_SV_MAXSPEED;
		raw[1] = rel[1] / QNN_SV_MAXSPEED;
		raw[2] = rel[2] / QNN_SV_MAXSPEED;
		goto snap;
	}

	/* Set up movers at their emission-start positions + velocities. */
	if (qnn_runtime.mover_count > 0)
	{
		qnn_mover_state_t ms[QNN_MAX_PHYS_MOVERS];
		int m;
		for (m = 0; m < qnn_runtime.mover_count; m++)
		{
			int nf = qnn_runtime.native_frame_count;
			float *mend = (nf > 0) ? qnn_runtime.mover_origins[nf - 1][m]
				: qnn_runtime.mover_emit_origins[m];
			ms[m].model_index = qnn_runtime.mover_model_indices[m];
			VectorCopy(qnn_runtime.mover_emit_origins[m], ms[m].origin);
			for (i = 0; i < 3; i++)
				ms[m].velocity[i] = (mend[i] - qnn_runtime.mover_emit_origins[m][i]) / emit_dt;
		}
		QNN_PhysSetupMovers(ms, qnn_runtime.mover_count);
	}

	/* Set up other players at mid-window positions. */
	if (qnn_runtime.player_count > 0)
	{
		int nf = qnn_runtime.native_frame_count;
		int mid = (nf > 0) ? nf / 2 : 0;
		if (mid < nf)
			QNN_PhysSetupPlayers(qnn_runtime.player_origins[mid],
				qnn_runtime.player_count);
	}

	{
		qboolean unreachable;

		QNN_PhysBestCandidate(
			qnn_runtime.emit_velocity, qnn_runtime.emit_origin,
			qnn_runtime.emit_view_angles, qnn_runtime.emit_grounded,
			qnn_runtime.emit_waterlevel, emit_dt,
			snapshot->player_origin,
			qnn_runtime.prev_fwd_sign, qnn_runtime.prev_strafe_sign,
			&fwd_sign, &strafe_sign, &unreachable);

		/* External force (knockback, trigger_push) moved the player
		 * beyond what any key combo can explain — carry forward. */
		if (unreachable)
		{
			fwd_sign = qnn_runtime.prev_fwd_sign;
			strafe_sign = qnn_runtime.prev_strafe_sign;
		}
	}
	qnn_runtime.prev_fwd_sign = fwd_sign;
	qnn_runtime.prev_strafe_sign = strafe_sign;

	/* Convert candidate signs to raw move vector for QNN_SnapMove. */
	raw[0] = (float)fwd_sign;
	raw[1] = (float)strafe_sign;
	/* Z: use raw position delta for water inversion, or 0 for ground. */
	{
		vec3_t delta, rel;
		for (i = 0; i < 3; i++)
			delta[i] = (snapshot->player_origin[i] - qnn_runtime.emit_origin[i]) / emit_dt;
		QNN_RelativeFrame(qnn_runtime.emit_view_angles, delta, rel);
		raw[2] = rel[2] / QNN_SV_MAXSPEED;
	}

snap:
	if (qnn_runtime.emit_waterlevel >= 2)
		medium = QNN_MEDIUM_WATER;
	else if (qnn_runtime.emit_grounded)
		medium = QNN_MEDIUM_GROUND;
	else
		medium = QNN_MEDIUM_AIR;

	QNN_SnapMove(raw, medium,
		QNN_SnapshotHasSelfJumpSound(snapshot),
		action->move);
}

/* QNN_EmitTick, QNN_JitterFilter, QNN_ActionIsFrozen, QNN_WriteObsTick,
 * QNN_FlushTickEmit — shared in common/qnn_collect_helpers.c. */

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
	qnn_runtime.prev_ammo = snapshot->ammo;
	qnn_runtime.has_prev = true;
}

static void QNN_SaveEmitAnchor(const qnn_snapshot_t *snapshot)
{
	VectorCopy(snapshot->player_view_angles, qnn_runtime.emit_view_angles);
	VectorCopy(snapshot->player_origin, qnn_runtime.emit_origin);
	if (qnn_runtime.has_prev)
	{
		VectorCopy(qnn_runtime.prev_velocity, qnn_runtime.emit_velocity);
	}
	else
	{
		qnn_runtime.emit_velocity[0] = 0.0f;
		qnn_runtime.emit_velocity[1] = 0.0f;
		qnn_runtime.emit_velocity[2] = 0.0f;
	}
	qnn_runtime.emit_grounded = snapshot->grounded;
	qnn_runtime.emit_waterlevel = snapshot->waterlevel;
	qnn_runtime.emit_weapon_id = snapshot->weapon_id;
	qnn_runtime.has_emit_anchor = true;
	qnn_runtime.native_frame_count = 0;

	/* Snapshot mover origins at the emission anchor. */
	{
		int m;
		for (m = 0; m < qnn_runtime.mover_count; m++)
		{
			entity_t *e = &cl_entities[qnn_runtime.mover_entity_nums[m]];
			VectorCopy(e->origin, qnn_runtime.mover_emit_origins[m]);
		}
	}
}


/* Apply emission-level frame filters.  Returns the FILE* to pass to
 * QNN_WriteObsTick: stdout to emit, NULL to buffer-only (skip). */
static FILE *QNN_EmitFilter(qnn_snapshot_t *snapshot)
{
	int health = snapshot->health;

	/* Always emit the done marker. */
	if (snapshot->done)
	{
		qnn_runtime.dead_emit_count = 0;
		qnn_runtime.frozen_emit_count = 0;
		return stdout;
	}

	/* God-mode: skip entirely. */
	if (health > QNN_GOD_MODE_HEALTH)
		return NULL;

	/* Dead: keep first QNN_DEAD_MAX_EMIT frames, inject fire=1,
	 * zero move (corpse physics ≠ player input, sim diverges
	 * because it runs alive friction+accel vs dead velocity decay). */
	if (health <= 0)
	{
		qnn_runtime.frozen_emit_count = 0;
		qnn_runtime.dead_emit_count++;
		if (qnn_runtime.dead_emit_count > QNN_DEAD_MAX_EMIT)
			return NULL;
		snapshot->action_label.fire = 1;
		snapshot->action_label.move[0] = 0.0f;
		snapshot->action_label.move[1] = 0.0f;
		snapshot->action_label.move[2] = 0.0f;
		return stdout;
	}

	/* Alive — reset dead counter. */
	qnn_runtime.dead_emit_count = 0;

	/* Frozen-alive: keep first QNN_FROZEN_MAX_EMIT frames. */
	if (QNN_ActionIsFrozen(&snapshot->action_label))
	{
		qnn_runtime.frozen_emit_count++;
		if (qnn_runtime.frozen_emit_count > QNN_FROZEN_MAX_EMIT)
			return NULL;
		return stdout;
	}

	/* Active gameplay — reset frozen counter. */
	qnn_runtime.frozen_emit_count = 0;
	return stdout;
}

static void QNN_RuntimeReset(void)
{
	memset(&qnn_runtime, 0, sizeof(qnn_runtime));
	qnn_runtime.fixed_tick_hz = 20;
	qnn_runtime.fixed_dt = 1.0f / 20.0f;
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

static qboolean QNN_ResetWorldLocal(const char *demo_path, char *error, size_t error_size)
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

	/* Reset all per-demo runtime state.  Worker reuse across demos would
	 * otherwise leak prev_velocity, emit anchors, mover/player/push tracking,
	 * buffered obs, jitter filter state, and frame counters into the first
	 * few ticks of the next demo — causing small label drift vs a
	 * fresh-worker collect.  Preserve only the config fields set at hello. */
	{
		int saved_tick_hz = qnn_runtime.fixed_tick_hz;
		float saved_dt = qnn_runtime.fixed_dt;
		int saved_resample = qnn_runtime.resample_hz;
		FILE *saved_store_dump = qnn_runtime.store_dump;
		memset(&qnn_runtime, 0, sizeof(qnn_runtime));
		qnn_runtime.fixed_tick_hz = saved_tick_hz;
		qnn_runtime.fixed_dt = saved_dt;
		qnn_runtime.resample_hz = saved_resample;
		qnn_runtime.store_dump = saved_store_dump;
	}

	qnn_sound_count = 0;
	cls.demonum = -1;
	cls.timedemo = false;

	/* Disconnect directly from C to avoid command buffer ordering issues */
	CL_Disconnect();
	if (sv.active)
		Host_ShutdownServer(false);
	cls.state = ca_disconnected;

	/* Nuke any residual stufftext from the previous demo.  svc_stufftext
	 * messages in demos (cvar sets, disconnect, map loads) accumulate in
	 * cmd_text and would otherwise run before — or worse, after — our
	 * playdemo, corrupting the newly-started demo's state. */
	{
		extern sizebuf_t cmd_text;
		SZ_Clear(&cmd_text);
	}

	snprintf(command, sizeof(command), "playdemo \"%s\"\n", demo_path);
	Cbuf_AddText(command);

	for (frame = 0; frame < 2048; ++frame)
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
	QNN_PhysInit();
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

	/* Resample target Hz: >0 = downsample to target, 0 = emit at native rate. */
	qnn_runtime.resample_hz = QNN_JsonExtractInt(line, "\"resample_hz\"", 0);


	if (!QNN_PrepareMap(map_id, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	QNN_WriteHelloResponse();
	return 0;
}

static int QNN_HandleCollect(const char *line)
{
	qnn_action_t native_action;
	qnn_snapshot_t snapshot;
	char demo_path[MAX_OSPATH];
	char error[256];
	int play_start, play_end;

	memset(demo_path, 0, sizeof(demo_path));
	memset(error, 0, sizeof(error));
	play_start = QNN_JsonExtractInt(line, "\"play_start\"", 0);
	play_end = QNN_JsonExtractInt(line, "\"play_end\"", 999999999);
	qnn_match_state = 1; /* always emit — boundaries are frame-gated */
	if (!QNN_JsonExtractString(line, "\"demo_path\"", demo_path, sizeof(demo_path)))
	{
		QNN_WriteError("reset options must include demo_path");
		return 0;
	}
	if (!QNN_ResetWorldLocal(demo_path, error, sizeof(error)))
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
		QNN_SavePrev(&snapshot, 0.0f);  /* first tick, no dt */
		qnn_runtime.phys_grounded = snapshot.grounded; /* seed from demo */
		QNN_SaveEmitAnchor(&snapshot);
		QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, true);
		QNN_BuildAllRefs();

		{
			qboolean emitting = false;

		while (!qnn_runtime.done)
		{
			Host_Frame(qnn_runtime.fixed_dt);
			qnn_runtime.tick += 1;
			qnn_runtime.steps += 1;
			if (!cls.demoplayback || cls.state == ca_disconnected)
				qnn_runtime.done = true;

			/* Frame-gated emission: play_start..play_end from the
			 * offline analyzer.  Everything outside is skipped. */
			if (qnn_runtime.tick > play_end)
			{
				snapshot.done = true;
				qnn_runtime.done = true;
			}

			QNN_CaptureSnapshotLocal(&snapshot, false);

			if (!emitting && qnn_runtime.tick >= play_start)
			{
				emitting = true;
				QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, true);
				QNN_ResampleInit(qnn_runtime.resample_hz);
				QNN_SaveEmitAnchor(&snapshot);
				fprintf(stderr, "[demo] emitting from tick %d (play_start=%d play_end=%d)\n",
					qnn_runtime.tick, play_start, play_end);
				QNN_SavePrev(&snapshot, qnn_runtime.fixed_dt);
				continue;
			}

			if (!emitting && !qnn_runtime.done)
			{
				QNN_SavePrev(&snapshot, qnn_runtime.fixed_dt);
				continue;
			}

			QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, false);

			if (qnn_runtime.store_dump != NULL)
			{
				QNN_StoreDumpSounds(qnn_runtime.store_dump, qnn_runtime.tick, &snapshot);
				QNN_StoreDumpTick(qnn_runtime.store_dump, qnn_runtime.tick, (float)cl.mtime[0]);
			}

			if (qnn_runtime.native_frame_count < QNN_MAX_NATIVE_FRAMES)
			{
				int nf = qnn_runtime.native_frame_count;
				int m;
				for (m = 0; m < qnn_runtime.mover_count; m++)
				{
					entity_t *me = &cl_entities[qnn_runtime.mover_entity_nums[m]];
					VectorCopy(me->origin, qnn_runtime.mover_origins[nf][m]);
				}
				for (m = 0; m < qnn_runtime.player_count; m++)
				{
					entity_t *pe = &cl_entities[qnn_runtime.player_entity_nums[m]];
					VectorCopy(pe->origin, qnn_runtime.player_origins[nf][m]);
				}
				qnn_runtime.native_frame_count++;
			}
			else if (qnn_runtime.native_frame_count == QNN_MAX_NATIVE_FRAMES)
			{
				fprintf(stderr, "[demo] native frame buffer full at tick %d\n",
					qnn_runtime.tick);
				qnn_runtime.native_frame_count++; /* warn once */
			}

			if (!snapshot.done)
				QNN_InferNativeAction(&native_action, &snapshot);
			else
				QNN_ClearAction(&native_action);

			QNN_ResampleAccumulate(&native_action, qnn_runtime.fixed_dt);
			while (QNN_ResampleShouldEmit() || qnn_runtime.done)
			{
				if (!snapshot.done)
				{
					QNN_InferEmitAction(&snapshot.action_label, &snapshot);
					if (qnn_runtime.native_frame_count > 0)
					{
						float phys_dt = qnn_runtime.native_frame_count * qnn_runtime.fixed_dt;
						QNN_InferEmitMove(&snapshot.action_label,
							&snapshot, phys_dt);
						VectorCopy(snapshot.action_label.move, qnn_runtime.prev_move);
					}
					else
					{
						VectorCopy(qnn_runtime.prev_move, snapshot.action_label.move);
					}
				}
				else
					QNN_ClearAction(&snapshot.action_label);
				QNN_ResampleApplyActionMerge(&snapshot.action_label);
				QNN_WriteObsTick(&qnn_runtime.tick_emit,
					QNN_EmitFilter(&snapshot),
					&snapshot, qnn_runtime.tick,
					qnn_runtime.steps, emit_hz, false);
				if (!snapshot.done)
					QNN_SaveEmitAnchor(&snapshot);
				if (qnn_resample.target_hz <= 0 || qnn_runtime.done)
					break;
			}

			QNN_SavePrev(&snapshot, qnn_runtime.fixed_dt);
		}

		/* If no match text was ever seen, emit a done tick so the
		   reader doesn't block. The Python side will see 0 real
		   ticks for trimmed spectator demos, or collect untrimmed
		   via a second pass for demos without match text. */
		if (!emitting)
		{
			snapshot.done = true;
			QNN_WriteObsTick(&qnn_runtime.tick_emit, stdout,
				&snapshot, qnn_runtime.tick,
				qnn_runtime.steps, emit_hz, true);
			/* Demo ended before play_start — no data emitted. */
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
