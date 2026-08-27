#include "qnn.h"
#include "qnn_collect_helpers.h"
#include "qnn_fault.h"
#include "qnn_io.h"
#include "qnn_store.h"
#include "qnn_metrics.h"
#include "qnn_watchdog.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_SERVER_NAME "quake-demo-worker"

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

	if (snapshot->weapon_id > 0)
		action->attack = QNN_ActionAttackPressed(action->move)
			? snapshot->weapon_id : 0;
	if (qnn_runtime.native_attack_this_window)
		action->move |= 0x01; /* bit 0 = attack press */
}

/* Per-native-frame: infer attack from sound/ammo cues.  Shared with the
 * QW MVD path via qnn_collect_helpers — see QNN_DetectAttackEvent.
 * Jump is set per-emit from QNN_SnapshotHasSelfJumpSound in QNN_SnapMove
 * below, not aggregated here. */
static void QNN_InferNativeAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);
	if (QNN_DetectAttackEvent(snapshot))
	{
		action->move |= 0x01; /* bit 0 = attack press */
		qnn_runtime.native_attack_this_window = true;
	}
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

	{
		float snapped[3];
		int fb_neg, fb_pos, lr_neg, lr_pos, up_neg, up_pos;
		uint8_t fb_lr_ud;
		QNN_SnapMove(raw, medium,
			QNN_SnapshotHasSelfJumpSound(snapshot),
			snapped);
		fb_neg = (snapped[0] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		fb_pos = (snapped[0] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		lr_neg = (snapped[1] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		lr_pos = (snapped[1] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		up_neg = (snapped[2] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		up_pos = (snapped[2] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		fb_lr_ud = QNN_PackInputMask(
			/*alive=*/1, fb_neg, fb_pos, lr_neg, lr_pos,
			up_neg, up_pos, 0, 0);
		/* Preserve attack/jump bits set by native-frame inference. */
		action->move = (uint8_t)((action->move & 0x81) | fb_lr_ud);
	}
}

/* QNN_EmitTick, QNN_JitterFilter, QNN_ActionIsFrozen, QNN_WriteObsTick,
 * QNN_FlushTickEmit, QNN_SavePrev, QNN_EmitFilter — shared in
 * common/qnn_collect_helpers.c. */

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
	qnn_runtime.native_attack_this_window = false;

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


static void QNN_RuntimeReset(void)
{
	memset(&qnn_runtime, 0, sizeof(qnn_runtime));
	qnn_runtime.fixed_tick_hz = 20;  /* default emit rate; overridden by hello.tick_hz */
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

	qnn_runtime.native_hz_detected = detected_hz;
	fprintf(stderr, "[demo] detected native tick rate: %d Hz (dt=%.4fs) — engine tick = %d Hz\n",
		detected_hz, native_dt, qnn_runtime.fixed_tick_hz);
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
		FILE *saved_store_dump = qnn_runtime.store_dump;
		memset(&qnn_runtime, 0, sizeof(qnn_runtime));
		qnn_runtime.fixed_tick_hz = saved_tick_hz;
		qnn_runtime.fixed_dt = saved_dt;
		qnn_runtime.store_dump = saved_store_dump;
	}

	qnn_sound_count = 0;
	cls.demonum = -1;
	cls.timedemo = false;

	/* Clear per-demo entity / event / action state up front so that a
	 * mid-signon failure below can't leave stale qnn_store entries
	 * visible to the next demo on this worker.  The success path below
	 * re-runs QNN_IOInit to rebuild baselines from the newly-loaded
	 * demo. */
	QNN_IOInit(&qnn_map_state);

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
	QNN_TickEmitReset(&qnn_runtime.tick_emit);
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
	/* Resolved perception regime (E6 provenance fix) — see the qw worker's
	 * QNN_WriteHelloResponse for the full rationale. */
	fprintf(stdout,
		",\"tick_hz\":%d,\"perception\":{\"los_clearance\":%d,\"fov\":%.6g},"
		"\"worker_build\":{\"basedir\":",
		qnn_runtime.fixed_tick_hz,
		QNN_LosClearanceResolved() ? 1 : 0,
		(double)QNN_FovResolved());
	QNN_WriteJsonString(stdout, basedir);
	fprintf(stdout, ",\"upstream_commit\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_UPSTREAM_COMMIT);
	fprintf(stdout, "}}\n");
	fflush(stdout);
}

/* QNN_OpIs — shared in common/qnn_collect_helpers.c. */

static int QNN_HandleHello(const char *line)
{
	char map_id[QNN_MAX_MAP_ID];
	char error[256];

	memset(map_id, 0, sizeof(map_id));
	memset(error, 0, sizeof(error));
	if (!QNN_JsonExtractString(line, "\"map_id\"", map_id, sizeof(map_id)))
		snprintf(map_id, sizeof(map_id), "E1M1");

	{
		int tick_hz = QNN_JsonExtractInt(line, "\"tick_hz\"", 20);
		if (tick_hz <= 0) tick_hz = 20;
		qnn_runtime.fixed_tick_hz = tick_hz;
		qnn_runtime.fixed_dt = 1.0f / (float)tick_hz;
		Cvar_SetValue("qnn_tick_hz", (float)tick_hz);
	}

	if (!QNN_PrepareMap(map_id, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	/* Resolve the perception cvars before reporting them — see the qw
	 * worker's QNN_HandleHello for the full rationale (QNN_IOInit
	 * normally does this registration, but doesn't run until the first
	 * demo's reset, too late for hello to reflect the real regime). */
	QNN_RegisterPerceptionCvars();

	QNN_WriteHelloResponse();
	return 0;
}

static int QNN_HandleCollect(const char *line)
{
	qnn_action_t native_action;
	qnn_snapshot_t snapshot;
	qnn_snapshot_t label_snapshot;
	char demo_path[MAX_OSPATH];
	char error[256];
	int play_start, play_end;

	memset(demo_path, 0, sizeof(demo_path));
	memset(error, 0, sizeof(error));
	play_start = QNN_JsonExtractInt(line, "\"play_start\"", 0);
	play_end = QNN_JsonExtractInt(line, "\"play_end\"", 999999999);
	if (!QNN_JsonExtractString(line, "\"demo_path\"", demo_path, sizeof(demo_path)))
	{
		QNN_WriteError("reset options must include demo_path");
		return 0;
	}
	QNN_FaultSetContext(demo_path);
	if (!QNN_ResetWorldLocal(demo_path, error, sizeof(error)))
	{
		QNN_WriteError(error);
		QNN_FaultSetContext(NULL);
		return 0;
	}

	/* Engine tick = emit tick.  fixed_dt set from hello.tick_hz. */

	/* play_start / play_end arrive as native-frame indices from the
	 * classifier.  qnn_runtime.tick increments per emit at fixed_tick_hz,
	 * so we convert once now using the engine-detected native rate.
	 * Fallback: NQ default 72 Hz when detection failed. */
	{
		int native_hz = qnn_runtime.native_hz_detected > 0
			? qnn_runtime.native_hz_detected
			: 72;
		if (play_start > 0)
			play_start = (int)(((long long)play_start * qnn_runtime.fixed_tick_hz) / native_hz);
		if (play_end < 999999999)
			play_end = (int)(((long long)play_end * qnn_runtime.fixed_tick_hz) / native_hz);
		fprintf(stderr, "[demo] play gate (native→emit @ %d Hz / %d Hz): play_start=%d play_end=%d\n",
			native_hz, qnn_runtime.fixed_tick_hz, play_start, play_end);
	}

	{
		int emit_hz = qnn_runtime.fixed_tick_hz;

		QNN_CaptureSnapshotLocal(&snapshot, true);
		QNN_SavePrev(&snapshot, 0.0f);  /* first tick, no dt */
		qnn_runtime.phys_grounded = snapshot.grounded; /* seed from demo */
		QNN_SaveEmitAnchor(&snapshot);
		QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, true);
		QNN_BuildAllRefs();

		{
			qboolean emitting = false;

		QNN_WatchdogBegin(10);
		while (!qnn_runtime.done)
		{
			qnn_runtime.emit_start_native = (float)cl.time;
			QNN_CaptureSnapshotLocal(&snapshot, false);
			Host_Frame(qnn_runtime.fixed_dt);
			QNN_WatchdogTick();
			qnn_runtime.tick += 1;
			qnn_runtime.steps += 1;
			if (!cls.demoplayback || cls.state == ca_disconnected)
				qnn_runtime.done = true;
			if (!qnn_runtime.done)
				QNN_CaptureSnapshotLocal(&label_snapshot, false);
			else
			{
				label_snapshot = snapshot;
				snapshot.done = true;
				label_snapshot.done = true;
			}

			/* Frame-gated emission: play_start..play_end from the
			 * offline analyzer.  Everything outside is skipped. */
			if (qnn_runtime.tick > play_end)
			{
				snapshot.done = true;
				label_snapshot.done = true;
				qnn_runtime.done = true;
			}

			if (!emitting && qnn_runtime.tick >= play_start)
			{
				emitting = true;
				QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, true);
				QNN_SaveEmitAnchor(&snapshot);
				fprintf(stderr, "[demo] emitting from tick %d (play_start=%d play_end=%d)\n",
					qnn_runtime.tick, play_start, play_end);
			}

			if (!emitting && !qnn_runtime.done)
			{
				QNN_SavePrev(&label_snapshot, qnn_runtime.fixed_dt);
				continue;
			}
			if (!emitting)
				continue;

			if (emitting && qnn_runtime.tick > play_start)
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
				QNN_InferNativeAction(&native_action, &label_snapshot);
			else
				QNN_ClearAction(&native_action);

			if (!snapshot.done)
			{
				QNN_InferEmitAction(&label_snapshot.action_label, &label_snapshot);
				if (qnn_runtime.native_frame_count > 0)
				{
					float phys_dt = qnn_runtime.native_frame_count * qnn_runtime.fixed_dt;
					QNN_InferEmitMove(&label_snapshot.action_label,
						&label_snapshot, phys_dt);
					qnn_runtime.prev_move = label_snapshot.action_label.move;
				}
				else
				{
					label_snapshot.action_label.move = qnn_runtime.prev_move;
				}
				snapshot.action_label = label_snapshot.action_label;
			}
			else
				QNN_ClearAction(&snapshot.action_label);
			QNN_WriteObsTick(&qnn_runtime.tick_emit,
				QNN_EmitFilter(&snapshot),
				&snapshot, qnn_runtime.tick,
				qnn_runtime.steps, emit_hz, false);
			if (!snapshot.done)
				QNN_SaveEmitAnchor(&label_snapshot);

			QNN_SavePrev(&label_snapshot, qnn_runtime.fixed_dt);
		}
		QNN_WatchdogEnd();

		/* Emit a done-tick if the replay produced no usable frames —
		   reached play_end before play_start (degenerate range),
		   aborted during signon, spectator demo trimmed to empty,
		   or no match text found.  Without this the Python reader
		   blocks forever waiting for FLAG_DONE. */
		if (!qnn_runtime.tick_emit.has_prev_emitted)
		{
			snapshot.done = true;
			QNN_WriteObsTick(&qnn_runtime.tick_emit, stdout,
				&snapshot, qnn_runtime.tick,
				qnn_runtime.steps, emit_hz, true);
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
	QNN_FaultSetContext(NULL);
	return 0;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	char line[QNN_WORKER_MAX_LINE];

	QNN_FaultInit("nq_demo_worker");
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
	QNN_TickRegister();
	/* Flush startdemos from quake.rc and tear down any auto-demo */
	Cbuf_Execute();
	cls.demonum = -1;
	CL_Disconnect();
	cls.state = ca_disconnected;

	while (fgets(line, sizeof(line), stdin) != NULL)
	{
		/* Match the "op" key precisely — substring search on the raw
		 * line collides with demo filenames that happen to contain
		 * the keyword (e.g. _vs_hello_kitty_).  See QNN_OpIs below. */
		if (QNN_OpIs(line, "hello"))
		{
			QNN_HandleHello(line);
			continue;
		}
		if (QNN_OpIs(line, "nav_query"))
		{
			QNN_HandleNavQuery(line);
			continue;
		}
		if (QNN_OpIs(line, "collect"))
		{
			QNN_HandleCollect(line);
			continue;
		}
		if (QNN_OpIs(line, "shutdown"))
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
