/*
 * qnn_collect_helpers.c — Engine-agnostic helpers shared by the NQ
 * and QW collect main loops.
 *
 * Contents:
 *   - Entity scanners for movers, push triggers, and other players
 *   - 3-frame jitter filter math (XY direction reversal detection)
 *   - Frozen-action predicate
 *   - QOBS framed tick emitter + two-level buffer (obs delay + jitter)
 *   - QNN_DetectAttackEvent (shared by both QWD and MVD inference paths)
 *   - QNN_EmitFilter (dead/frozen/god-mode rate caps)
 *   - QNN_PackSnapshotObs / QNN_WriteObsTickPrepacked* (used by both
 *     the live emit path and the MVD back-shift ring drain)
 *   - QNN_SavePrev (per-tick state advance)
 *   - QNN_FillLookAndSwitch (look/weapon label, shared QWD + MVD)
 *   - Shared back-shift ring (deferred label emit; driven by the MVD
 *     sound writers in qnn_mvd_collect.c, reused by the QWD path)
 *
 * Touches only shared state (qnn_store, cl_entities, cl.model_precache,
 * cl.maxclients, cl.viewentity — all present natively on NQ and
 * synthesized on QW by QNN_SyncEngineCompat).
 *
 * MVD reconstruction (back-shift ring, log-normal hold tails) lives in
 * its own module — see qnn_mvd_collect.c.
 */

#include "qnn.h"
#include "qnn_collect_helpers.h"
#include "qnn_io.h"
#include "qnn_store.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The single collect-runtime instance, shared by both engines' workers. */
qnn_runtime_t qnn_runtime;

float QNN_RuntimeNowSeconds(void)
{
	if (qnn_runtime.fixed_tick_hz > 0)
		return (float)qnn_runtime.tick / (float)qnn_runtime.fixed_tick_hz;
	return qnn_runtime.emit_start_native;
}

static void QNN_WriteJsonEscaped(FILE *out, const char *s)
{
	const unsigned char *p = (const unsigned char *)s;

	fputc('"', out);
	while (*p)
	{
		unsigned char c = *p++;
		if (c == '"' || c == '\\')
		{
			fputc('\\', out);
			fputc(c, out);
		}
		else if (c == '\n')
			fputs("\\n", out);
		else if (c == '\r')
			fputs("\\r", out);
		else if (c == '\t')
			fputs("\\t", out);
		else if (c < 0x20)
			fprintf(out, "\\u%04x", (unsigned)c);
		else
			fputc(c, out);
	}
	fputc('"', out);
}

static void QNN_DumpAttackRoutes(int row, int tick, int steps,
	const qnn_action_t *action, const qnn_attack_route_event_t *routes,
	int route_count)
{
	const char *path;
	FILE *out;
	int i;

	if (route_count <= 0)
		return;
	path = getenv("QNN_ATTACK_ROUTE_DUMP");
	if (path == NULL || path[0] == '\0')
		return;
	out = fopen(path, "a");
	if (out == NULL)
		return;
	for (i = 0; i < route_count; ++i)
	{
		const qnn_attack_route_event_t *r = &routes[i];
		fprintf(out, "{\"row\":%d,\"tick\":%d,\"steps\":%d,"
			"\"demo_path\":", row, tick, steps);
		QNN_WriteJsonEscaped(out, qnn_runtime.demo_path);
		fprintf(out,
			",\"source_tick\":%d,\"dest_tick\":%d,"
			"\"sound_index\":%d,\"weapon_id\":%d,"
			"\"native_time\":%.6f,\"emit_start_native\":%.6f,"
			"\"ping_sec\":%.6f,\"phase\":%.6f,"
			"\"press_offset\":%.6f,"
			"\"deterministic_offset\":%d,"
			"\"route_offset\":%d,"
			"\"emitted_attack\":%d,\"final_weapon\":%d}\n",
			r->source_tick, r->dest_tick, r->sound_index,
			r->weapon_id, r->native_time, r->emit_start_native,
			r->ping_sec, r->phase, r->press_offset,
			r->deterministic_offset,
			r->route_offset, QNN_ActionAttack(action->move),
			action->weapon);
	}
	fclose(out);
}

/* Scan the entity store for movers and record their entity_num +
 * modelindex for BSP collision during physics baseline steps. */
int QNN_BuildMoverRefs(int *out_entity_nums, int *out_model_indices,
	int max_count)
{
	int i, j, count;

	count = 0;
	for (i = 0; i < MAX_EDICTS && count < max_count; i++)
	{
		entity_t *ent;
		int model_index;

		if (qnn_store[i].type != QNN_ENT_MOVER)
			continue;
		ent = &cl_entities[i];
		if (ent->model == NULL || ent->model->name[0] != '*')
			continue;

		/* Find modelindex by matching model pointer against precache. */
		model_index = 0;
		for (j = 1; j < MAX_MODELS; j++)
		{
			if (cl.model_precache[j] == ent->model)
			{
				model_index = j;
				break;
			}
		}
		if (model_index <= 0)
			continue;

		out_entity_nums[count] = i;
		out_model_indices[count] = model_index;
		count++;
	}
	return count;
}

/* Build push trigger references from the entity store overflow.
 * Output velocity is speed * direction (pre-multiplied for direct use). */
int QNN_BuildPushRefs(int *out_model_indices, vec3_t *out_velocities,
	int max_count)
{
	int i, count;

	count = 0;
	for (i = MAX_EDICTS; i < QNN_StoreCapacity() && count < max_count; i++)
	{
		qnn_entity_t *e = &qnn_store[i];
		if (e->type != QNN_ENT_PUSH || e->push_model_index <= 0)
			continue;
		out_model_indices[count] = e->push_model_index;
		out_velocities[count][0] = e->push_direction[0] * e->push_speed;
		out_velocities[count][1] = e->push_direction[1] * e->push_speed;
		out_velocities[count][2] = e->push_direction[2] * e->push_speed;
		count++;
	}
	return count;
}

/* Scan for other players (actors) in the entity store.
 * self_entity_num: caller's own entity (cl.viewentity in NQ,
 * cl.playernum + 1 in QW) — skip this one. */
int QNN_BuildPlayerRefs(int self_entity_num, int *out_entity_nums,
	int max_count)
{
	int i, count;

	count = 0;
	for (i = 1; i <= cl.maxclients && count < max_count; i++)
	{
		if (i == self_entity_num)
			continue;
		if (qnn_store[i].type != QNN_ENT_ACTOR)
			continue;
		out_entity_nums[count++] = i;
	}
	return count;
}

/* 3-frame jitter filter on the per-axis press bits.  If the prev and
 * next ticks share an axis press (both neg, or both pos) but the mid
 * tick disagrees with both — including the case where mid is released
 * — restore the neighbors' bits onto the mid tick.  One-tick anomaly
 * removal for discrete press signals.  Operates on fb and lr axes only;
 * ud bits (and attack/jump) are passed through unchanged.
 *
 * Pure math — no globals.  Used by the WriteObsTick integration in
 * each engine's collect main. */
void QNN_JitterFilter(qnn_action_t *mid, uint8_t prev_move,
	uint8_t next_move)
{
	int axis;

	for (axis = 0; axis < 2; ++axis)
	{
		int neg_bit = 1 + 2 * axis;
		int pos_bit = 2 + 2 * axis;
		int mask = (1 << neg_bit) | (1 << pos_bit);
		int prev_axis = prev_move & mask;
		int next_axis = next_move & mask;
		int mid_axis  = mid->move & mask;

		if (prev_axis == 0 || next_axis == 0)
			continue;
		if (prev_axis != next_axis)
			continue;
		if (mid_axis == prev_axis)
			continue;
		mid->move = (uint8_t)((mid->move & ~mask) | prev_axis);
	}
}

/* True when the action label represents a fully idle tick — zero move,
 * zero look-delta, no attack.  Used by the emit-rate filter to cap runs
 * of frozen frames.  Collect workers store dense full weapon intent in
 * weapon, so it's not part of the activity check; standing on the
 * same weapon doesn't constitute movement. */
qboolean QNN_ActionIsFrozen(const qnn_action_t *a)
{
	return a->move == 0
		&& fabsf(a->look[0] - 1.0f) < QNN_FROZEN_LOOK_TOL
		&& fabsf(a->look[1]) < QNN_FROZEN_LOOK_TOL
		&& fabsf(a->look[2]) < QNN_FROZEN_LOOK_TOL;
}

static int QNN_PoseDiagEnabled(void);
static void QNN_DumpPose(int row, const uint8_t *obs);

/* Emit one framed tick: "QOBS" magic + 16-byte header + obs + action.
 * Header is (tick, steps, tick_hz, flags, action_size) little-endian. */
void QNN_EmitTick(FILE *out, const uint8_t *obs, const qnn_action_t *action,
	int tick, int steps, int tick_hz, uint16_t flags)
{
	uint8_t header[16];
	uint16_t asize = (uint16_t)sizeof(qnn_action_t);

	if (QNN_PoseDiagEnabled())
	{
		/* Per-demo row counter, reset when the demo changes; rows
		 * number the QOBS records of one demo in stream order. */
		static int pose_row;
		static char pose_demo[MAX_OSPATH];

		if (strncmp(pose_demo, qnn_runtime.demo_path,
			sizeof(pose_demo)) != 0)
		{
			strncpy(pose_demo, qnn_runtime.demo_path,
				sizeof(pose_demo) - 1);
			pose_demo[sizeof(pose_demo) - 1] = '\0';
			pose_row = 0;
		}
		QNN_DumpPose(pose_row++, obs);
	}

	memcpy(header + 0,  &tick,    4);
	memcpy(header + 4,  &steps,   4);
	memcpy(header + 8,  &tick_hz, 4);
	memcpy(header + 12, &flags,   2);
	memcpy(header + 14, &asize,   2);

	fwrite("QOBS", 1, 4, out);
	fwrite(header, 1, sizeof(header), out);
	fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, out);
	fwrite(action, 1, sizeof(qnn_action_t), out);
	fflush(out);
}

/* Emit one slim MLOB record: "MLOB" magic + the fixed-size packed
 * qnn_mlob_record_t (no obs buffer).  Used by the matched-emit path to
 * write the per-native-frame labeler stream interleaved with QOBS on the
 * same stdout pipe; the Python orchestration demuxes by magic. */
void QNN_EmitMlob(FILE *out, const qnn_mlob_record_t *rec)
{
	uint8_t buf[24];

	/* Pack explicitly (no struct-padding assumptions on the wire):
	 *   0  native_index      u32
	 *   4  flags             u16
	 *   6  vel[3]            i16 ×3
	 *  12  self_movement_id   u8
	 *  13  self_weapon_id     u8
	 *  14  move               u8  (qnn_action_t offset 0)
	 *  15  weapon             u8
	 *  16  input_mask         u8
	 *  17  op_input           u8
	 *  18  look[3]           f16 ×3  — packed below
	 * Total 24 bytes. */
	uint16_t lh[3];
	int i;

	memcpy(buf + 0,  &rec->native_index, 4);
	memcpy(buf + 4,  &rec->flags,        2);
	memcpy(buf + 6,  rec->vel,           6);
	buf[12] = rec->self_movement_id;
	buf[13] = rec->self_weapon_id;
	buf[14] = rec->action.move;
	buf[15] = rec->action.weapon;
	buf[16] = rec->action.input_mask;
	buf[17] = rec->action.op_input;
	for (i = 0; i < 3; ++i)
		lh[i] = QNN_FloatToHalf(rec->action.look[i]);
	memcpy(buf + 18, lh, 6);

	fwrite(QNN_MLOB_MAGIC, 1, 4, out);
	fwrite(buf, 1, sizeof(buf), out);
	fflush(out);
}

/* ── Shared tick-emission pipeline ───────────────────────────────
 *
 * One-level buffer:
 *   The paired obs+action is held one tick so the 3-frame jitter filter
 *   can check it against prev and next. Demo collect callers pass
 *   pre-command obs with the same tick's action label already aligned.
 *
 * Action history is pushed at EMIT time with the filtered action, not at
 * capture time.  This ensures the next frame's obs has the correct T-1
 * action in its history (1 frame of lag, not 2).
 * ──────────────────────────────────────────────────────────────── */

void QNN_TickEmitReset(qnn_tick_emit_state_t *st)
{
	memset(st, 0, sizeof(*st));
}

static void QNN_TickEmitFlush(qnn_tick_emit_state_t *st)
{
	if (!st->has_jitter_buf)
		return;
	if (st->jitter_out != NULL)
	{
		QNN_EmitTick(st->jitter_out, st->jitter_obs,
			&st->jitter_action,
			st->jitter_tick, st->jitter_steps,
			st->jitter_tick_hz, st->jitter_flags);
		QNN_DumpAttackRoutes(st->emitted_rows, st->jitter_tick,
			st->jitter_steps, &st->jitter_action,
			st->jitter_attack_routes,
			st->jitter_attack_route_count);
		st->emitted_rows++;
		st->has_prev_emitted = true;
	}
	st->has_prev_prev_move = true;
	st->prev_prev_move = st->jitter_action.move;
	st->has_jitter_buf = false;
}

void QNN_FlushTickEmit(qnn_tick_emit_state_t *st)
{
	QNN_TickEmitFlush(st);
}

/* ── Pose diagnostic (QNN_POSE_DIAG) ─────────────────────────────
 * Probe-grid studies need (origin, view_yaw) per emitted QOBS row,
 * which the egocentric obs deliberately omits.  When QNN_POSE_DIAG
 * names a JSONL path, the packer stashes the capture-time pose in the
 * obs buffer's guaranteed-zero tail (entity stream tops out ~1.5 KB of
 * the 4 KB buffer) and QNN_EmitTick — the single choke point every
 * emit path funnels through, in final stream order — reads it back and
 * appends one line per record.  Off by default; the tail stays zero.
 * The tail offset + stash live in qnn_io (QNN_POSE_TAIL_OFF /
 * QNN_IOStashPoseTail) — shared with the closed-loop workers'
 * QNN_POSE_TAIL channel. */

static int QNN_PoseDiagEnabled(void)
{
	static int checked, enabled;

	if (!checked)
	{
		const char *path = getenv("QNN_POSE_DIAG");
		enabled = (path != NULL && path[0] != '\0');
		checked = 1;
	}
	return enabled;
}

static void QNN_DumpPose(int row, const uint8_t *obs)
{
	static FILE *out;
	float pose[4];

	if (!QNN_PoseDiagEnabled())
		return;
	if (out == NULL)
	{
		/* Per-process file: parallel collect workers share the env
		 * var, and concurrent appends to one file would interleave
		 * partial lines. */
		char path[MAX_OSPATH + 32];
		snprintf(path, sizeof(path), "%s.%d.jsonl",
			getenv("QNN_POSE_DIAG"), (int)getpid());
		out = fopen(path, "a");
		if (out == NULL)
			return;
	}
	memcpy(pose, obs + QNN_POSE_TAIL_OFF, sizeof(pose));
	fprintf(out, "{\"demo\":");
	QNN_WriteJsonEscaped(out, qnn_runtime.demo_path);
	fprintf(out, ",\"row\":%d,\"origin\":[%.3f,%.3f,%.3f],"
		"\"view_yaw\":%.3f}\n",
		row, (double)pose[0], (double)pose[1], (double)pose[2],
		(double)pose[3]);
	fflush(out);
}

void QNN_PackSnapshotObs(const qnn_snapshot_t *snapshot, uint8_t *obs_out)
{
	qnn_tick_result_t result;
	QNN_IOEmit(snapshot, &result);
	QNN_IOPackObsBuffer(obs_out, &result);
	if (QNN_PoseDiagEnabled())
		QNN_IOStashPoseTail(obs_out, snapshot);
}

/* Internal: drive the obs/jitter pipeline from pre-packed obs bytes.
 * Shared body of QNN_WriteObsTick (which packs first) and
 * QNN_WriteObsTickPrepacked (caller already packed at capture time
 * to survive deferred emit through the MVD back-shift ring). */
static void QNN_WriteObsTickInner(qnn_tick_emit_state_t *st, FILE *out,
	const uint8_t *cur_obs, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz, qboolean reset_flag,
	const qnn_attack_route_event_t *routes, int route_count)
{
	uint16_t flags = 0;
	if (route_count < 0)
		route_count = 0;
	if (route_count > QNN_MAX_ATTACK_ROUTE_EVENTS)
		route_count = QNN_MAX_ATTACK_ROUTE_EVENTS;
	if (done) flags |= 0x02;
	if (reset_flag) flags |= 0x01;

	if (!st->has_jitter_buf)
	{
		if (done)
		{
			if (out != NULL)
			{
				QNN_EmitTick(out, cur_obs, action,
					tick, steps, tick_hz, flags);
				QNN_DumpAttackRoutes(st->emitted_rows, tick,
					steps, action, routes, route_count);
				st->emitted_rows++;
			}
			return;
		}
		memcpy(st->jitter_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
		st->jitter_action = *action;
		st->jitter_tick = tick;
		st->jitter_steps = steps;
		st->jitter_tick_hz = tick_hz;
		st->jitter_flags = flags;
		st->jitter_out = out;
		st->jitter_attack_route_count = route_count;
		if (route_count > 0)
			memcpy(st->jitter_attack_routes, routes,
				sizeof(qnn_attack_route_event_t) * route_count);
		st->has_jitter_buf = true;
		return;
	}

	/* We have prev_prev (if any), jitter_buf (middle), and current
	   action (next).  Apply the 3-frame filter to the middle. */
	if (st->has_prev_prev_move)
		QNN_JitterFilter(&st->jitter_action,
			st->prev_prev_move,
			action->move);

	/* Emit the (possibly corrected) jitter-buffered tick. */
	if (st->jitter_out != NULL)
	{
		QNN_EmitTick(st->jitter_out, st->jitter_obs,
			&st->jitter_action,
			st->jitter_tick,
			st->jitter_steps,
			st->jitter_tick_hz,
			st->jitter_flags);
		QNN_DumpAttackRoutes(st->emitted_rows,
			st->jitter_tick, st->jitter_steps,
			&st->jitter_action,
			st->jitter_attack_routes,
			st->jitter_attack_route_count);
		st->emitted_rows++;
		st->has_prev_emitted = true;
	}

	st->prev_prev_move = st->jitter_action.move;
	st->has_prev_prev_move = true;

	memcpy(st->jitter_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
	st->jitter_action = *action;
	st->jitter_tick = tick;
	st->jitter_steps = steps;
	st->jitter_tick_hz = tick_hz;
	st->jitter_flags = flags;
	st->jitter_out = out;
	st->jitter_attack_route_count = route_count;
	if (route_count > 0)
		memcpy(st->jitter_attack_routes, routes,
			sizeof(qnn_attack_route_event_t) * route_count);

	if (done)
		QNN_TickEmitFlush(st);
}

void QNN_WriteObsTick(qnn_tick_emit_state_t *st, FILE *out,
	const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz,
	qboolean reset_flag)
{
	uint8_t cur_obs[QNN_OBS_BUFFER_SIZE];
	QNN_PackSnapshotObs(snapshot, cur_obs);
	QNN_WriteObsTickInner(st, out, cur_obs, &snapshot->action_label,
		snapshot->done, tick, steps, tick_hz, reset_flag,
		NULL, 0);
}

void QNN_WriteObsTickPrepacked(qnn_tick_emit_state_t *st, FILE *out,
	const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag)
{
	QNN_WriteObsTickInner(st, out, obs_bytes, action,
		done, tick, steps, tick_hz, reset_flag,
		NULL, 0);
}

void QNN_WriteObsTickPrepackedWithAttackRoutes(qnn_tick_emit_state_t *st,
	FILE *out, const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, const qnn_attack_route_event_t *routes,
	int route_count)
{
	QNN_WriteObsTickInner(st, out, obs_bytes, action,
		done, tick, steps, tick_hz, reset_flag,
		routes, route_count);
}

/* ── Collect runtime helpers ────────────────────────────────────── */

void QNN_SavePrev(const qnn_snapshot_t *snapshot, float dt)
{
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
	qnn_runtime.prev_weapon_id = snapshot->weapon_id;
	qnn_runtime.prev_grounded = snapshot->grounded;
	VectorCopy(snapshot->player_velocity, qnn_runtime.prev_snap_velocity);
	qnn_runtime.has_prev = true;
}

/* Single-frame attack-event detection used by both NQ and QW MVD paths.
 * Returns true if THIS native frame contains a shot event:
 *   - a weapon-fire sound matching the currently-held weapon, OR
 *   - an ammo decrement on the same weapon as the previous frame.
 * The weapon-id guard on the ammo branch prevents weapon-switch
 * cascades — snapshot->ammo jumps between weapons' ammo pools and
 * would otherwise read as a "decrement". */
qboolean QNN_DetectAttackEvent(const qnn_snapshot_t *snapshot)
{
	static FILE *attack_source_dump = NULL;
	static qboolean attack_source_dump_checked = false;
	qboolean sound_attack;
	qboolean ammo_drop;

	if (!qnn_runtime.has_prev)
		return false;
	sound_attack = QNN_SnapshotHasSelfWeaponAttackSound(snapshot);
	ammo_drop = (snapshot->weapon_id == qnn_runtime.prev_weapon_id
		&& snapshot->ammo < qnn_runtime.prev_ammo) ? true : false;
	if (!attack_source_dump_checked)
	{
		const char *path = getenv("QNN_ATTACK_SOURCE_DUMP");
		attack_source_dump_checked = true;
		if (path != NULL && path[0] != '\0')
			attack_source_dump = fopen(path, "a");
	}
	if (attack_source_dump != NULL && (sound_attack || ammo_drop))
	{
		fprintf(attack_source_dump,
			"{\"tick\":%d,\"sound\":%d,\"ammo\":%d,"
			"\"weapon\":%d,\"prev_weapon\":%d,\"ammo_count\":%d,"
			"\"prev_ammo\":%d}\n",
			qnn_runtime.tick, sound_attack ? 1 : 0, ammo_drop ? 1 : 0,
			snapshot->weapon_id, qnn_runtime.prev_weapon_id,
			snapshot->ammo, qnn_runtime.prev_ammo);
		fflush(attack_source_dump);
	}
	return (sound_attack || ammo_drop) ? true : false;
}

/* Shared attack-feasibility probe (see header). Was duplicated verbatim in
 * QNN_QwdPackInputMask and QNN_MvdPackInputMask. */
int QNN_EvalAttackFeasible(const qnn_snapshot_t *snapshot,
	float start_of_tick_attack_finished)
{
	float saved_af;
	int op_attack;

	if (snapshot == NULL || !QNN_WeaponIsValid(snapshot->weapon_id))
		return 0;
	saved_af = QNN_ProgsGetAttackFinished();
	QNN_ProgsSetAttackFinished(start_of_tick_attack_finished);
	op_attack = QNN_ProgsEvalAttack(
		QNN_RuntimeNowSeconds(),
		snapshot->health, snapshot->items_owned,
		snapshot->ammo_shells, snapshot->ammo_nails,
		snapshot->ammo_rockets, snapshot->ammo_cells,
		snapshot->weapon_id, /*button0=*/1);
	QNN_ProgsSetAttackFinished(saved_af);
	return op_attack;
}

/* Fill action->look (view-relative turn delta) and action->weapon
 * (per-frame held-weapon state) from the current snapshot.  Shared by
 * both the QWD usercmd-truth path and the MVD inference path.  weapon is
 * always set to the current weapon when not already set by the caller's
 * usercmd path (which provides press-time prediction); MVD has no
 * usercmd so this is the only source. */
void QNN_FillLookAndSwitch(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	vec3_t forward, right, up, cur_forward;

	if (!qnn_runtime.has_emit_anchor)
		return;

	AngleVectors(qnn_runtime.emit_view_angles, forward, right, up);
	QNN_ForwardFromAngles(snapshot->player_view_angles, cur_forward);
	action->look[0] = DotProduct(cur_forward, forward);
	action->look[1] = DotProduct(cur_forward, right);
	action->look[2] = DotProduct(cur_forward, up);

	/* Held-weapon fallback: only adopt a vanilla weapon id (1..8).  Mod
	 * servers report held ids >8 in the playerstate stat; treat those as
	 * no-weapon (0), matching the obs token's qnn_weapon_subject_from_id
	 * handling.  A bare `> 0` here leaked mod ids into the act_weapon
	 * label and tripped the downstream 0..8 wire guard. */
	if (action->weapon == 0 && QNN_WeaponIsValid(snapshot->weapon_id))
		action->weapon = snapshot->weapon_id;
}

uint8_t QNN_PackOpInput(
	int alive,
	int fb_press, int lr_press,
	int jump_press, int swim_press, int attack_press,
	int in_water,
	int op_jump, int op_attack, int op_impulse,
	int has_impulse)
{
	uint8_t op;

	if (!alive)
		return 0;
	op = 0;
	if (fb_press)
		op |= 0x01;
	if (lr_press)
		op |= 0x02;
	if (jump_press && op_jump)
		op |= 0x04;
	else if (swim_press && in_water)
		op |= 0x04;
	if (attack_press && op_attack)
		op |= 0x08;
	if (has_impulse && op_impulse)
		op |= 0x10;
	return op;
}

/* Pack per-axis bits into the byte layout shared by qnn_action_t.input_mask
 * and qnn_action_t.move (the action press byte).  Generic across the
 * feasibility and press uses — same bit positions, callers supply either
 * feasibility or press intent for each axis. */
uint8_t QNN_PackInputMask(
	int alive,
	int fb_act_neg,  int fb_act_pos,
	int lr_act_neg,  int lr_act_pos,
	int up_act_neg,  int up_act_pos,
	int jump_act,
	int attack_act)
{
	uint8_t m;

	if (!alive)
		return 0;
	m = 0;
	if (attack_act)
		m |= 0x01;	/* bit 0 */
	/* Per-axis 2-bit fields.  Under PURE-FEASIBILITY semantics both
	 * direction bits may be set (e.g. fb_act_neg=fb_act_pos=1 means
	 * "engine accepts EITHER direction this tick" — always true for
	 * fb/lr in pmove). Under demo-AND-engine semantics caller can
	 * still pass mutually-exclusive bits and only one will end up set.
	 * Layout: bit n = neg, bit n+1 = pos. */
	if (fb_act_neg) m |= 0x02;	/* bit 1 */
	if (fb_act_pos) m |= 0x04;	/* bit 2 */
	if (lr_act_neg) m |= 0x08;	/* bit 3 */
	if (lr_act_pos) m |= 0x10;	/* bit 4 */
	if (up_act_neg) m |= 0x20;	/* bit 5 */
	if (up_act_pos) m |= 0x40;	/* bit 6 */
	if (jump_act)
		m |= 0x80;	/* bit 7 */
	return m;
}

FILE *QNN_EmitFilter(qnn_snapshot_t *snapshot)
{
	int health = snapshot->health;

	if (snapshot->done)
	{
		qnn_runtime.dead_emit_count = 0;
		qnn_runtime.frozen_emit_count = 0;
		return stdout;
	}

	/* God-mode (training): skip entirely. */
	if (health > QNN_GOD_MODE_HEALTH)
		return NULL;

	/* Dead: keep first QNN_DEAD_MAX_EMIT frames, inject attack=1, zero
	 * move (corpse physics ≠ player input — alive accel/friction vs
	 * dead velocity decay diverges). */
	if (health <= 0)
	{
		qnn_runtime.frozen_emit_count = 0;
		qnn_runtime.dead_emit_count++;
		if (qnn_runtime.dead_emit_count > QNN_DEAD_MAX_EMIT)
			return NULL;
		snapshot->action_label.move = 0x01; /* bit 0 = attack press */
		return stdout;
	}

	qnn_runtime.dead_emit_count = 0;

	if (QNN_ActionIsFrozen(&snapshot->action_label))
	{
		qnn_runtime.frozen_emit_count++;
		if (qnn_runtime.frozen_emit_count > QNN_FROZEN_MAX_EMIT)
			return NULL;
		return stdout;
	}

	qnn_runtime.frozen_emit_count = 0;
	return stdout;
}

/* Match the literal `"op":"<value>"` JSON key (with or without a space
 * after the colon) anywhere on a worker command line.  Necessary because
 * the dispatch loop runs over fields like the demo filename, and a naive
 * strstr(line, "<value>") false-matches on filenames containing the
 * keyword (e.g. _vs_hello_kitty_ → "hello" → wrong handler). */
int QNN_OpIs(const char *line, const char *value)
{
	char needle[64];
	int n;
	n = snprintf(needle, sizeof(needle), "\"op\":\"%s\"", value);
	if (n > 0 && n < (int)sizeof(needle) && strstr(line, needle) != NULL)
		return 1;
	n = snprintf(needle, sizeof(needle), "\"op\": \"%s\"", value);
	if (n > 0 && n < (int)sizeof(needle) && strstr(line, needle) != NULL)
		return 1;
	return 0;
}

/* ══════════════════════════════════════════════════════════════════════
 *                  SHARED BACK-SHIFT RING (deferred label emit)
 * ══════════════════════════════════════════════════════════════════════
 *
 * The generic ring lets server-observed sound/state events write labels
 * back at the press frame instead of the broadcast frame.  The MVD path
 * (qnn_mvd_collect.c) drives it from weapon-fire / jump sounds; the QWD
 * path reuses the same primitives.  The instance is file-static here so
 * both the NQ and QW collect workers (which link this file) get one
 * shared ring; the MVD module reaches it via QNN_BackShiftRing(). */

static qnn_backshift_ring_t g_backshift_ring;

qnn_backshift_ring_t *QNN_BackShiftRing(void)
{
	return &g_backshift_ring;
}

void QNN_BackShiftReset(void)
{
	memset(&g_backshift_ring, 0, sizeof(g_backshift_ring));
}

static int QNN_BackShiftLatestIndex(const qnn_backshift_ring_t *ring)
{
	return (ring->head + QNN_BACKSHIFT_K - 1) % QNN_BACKSHIFT_K;
}

qboolean QNN_BackShiftSlotAt(qnn_backshift_ring_t *ring,
	int shift_frames, qnn_backshift_slot_t **slot_out)
{
	int idx;

	if (ring->count == 0 || shift_frames < 0)
		return false;
	if (shift_frames >= ring->count)
		shift_frames = ring->count - 1;
	idx = (QNN_BackShiftLatestIndex(ring) + QNN_BACKSHIFT_K
		- shift_frames) % QNN_BACKSHIFT_K;
	*slot_out = &ring->slots[idx];
	return true;
}

static void QNN_BackShiftEmitSlot(qnn_backshift_slot_t *slot,
	qnn_tick_emit_state_t *emit)
{
	if (!slot->valid)
		return;
	QNN_WriteObsTickPrepackedWithAttackRoutes(emit, slot->out,
		slot->obs, &slot->action,
		slot->done, slot->tick, slot->steps, slot->tick_hz,
		slot->reset_flag, slot->attack_routes,
		slot->attack_route_count);
	slot->valid = false;
	slot->attack_route_count = 0;
}

qboolean QNN_BackShiftPrevWeapon(int *prev_weapon_out)
{
	if (!g_backshift_ring.has_prev_weapon_id)
		return false;
	if (prev_weapon_out != NULL)
		*prev_weapon_out = g_backshift_ring.prev_weapon_id;
	return true;
}

int QNN_BackShiftCount(void)
{
	return g_backshift_ring.count;
}

void QNN_BackShiftPush(qnn_tick_emit_state_t *emit, FILE *out,
	const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, qboolean grounded,
	int weapon_id)
{
	qnn_backshift_ring_t *ring = &g_backshift_ring;
	qnn_backshift_slot_t *slot;

	/* When full, the slot at `head` is the oldest — drain it first. */
	if (ring->count == QNN_BACKSHIFT_K)
		QNN_BackShiftEmitSlot(&ring->slots[ring->head], emit);
	else
		ring->count++;

	slot = &ring->slots[ring->head];
	memcpy(slot->obs, obs_bytes, QNN_OBS_BUFFER_SIZE);
	slot->action = *action;
	slot->done = done;
	slot->tick = tick;
	slot->steps = steps;
	slot->tick_hz = tick_hz;
	slot->reset_flag = reset_flag;
	slot->grounded = grounded;
	slot->out = out;
	slot->valid = true;
	slot->attack_route_count = 0;

	ring->head = (ring->head + 1) % QNN_BACKSHIFT_K;
	ring->prev_weapon_id = weapon_id;
	ring->has_prev_weapon_id = true;
}

/* On observed weapon_id transitions, rewrite the trailing `shift_frames`
 * slots back to the new weapon to anchor intent at the press frame
 * instead of the broadcast frame.  The pickup gate (handled at the call
 * site) suppresses the rewrite for server-forced touches. */
void QNN_BackShiftRewriteWeapon(int new_weapon_id,
	int prev_weapon_id, int shift_frames)
{
	qnn_backshift_ring_t *ring = &g_backshift_ring;
	int n;
	int i;

	if (ring->count == 0 || shift_frames <= 0)
		return;

	/* Call site is BEFORE the current push of the post-transition
	 * frame.  `latest` is therefore the previous push's slot.  Rewrite
	 * the trailing `shift_frames` slots so they carry the new weapon.
	 * Only rewrite slots whose label still equals `prev_weapon_id`
	 * so a later transition can't clobber an earlier one when both
	 * land inside the same window. */
	n = shift_frames;
	if (n > ring->count)
		n = ring->count;
	for (i = 0; i < n; i++)
	{
		qnn_backshift_slot_t *slot;
		if (!QNN_BackShiftSlotAt(ring, i, &slot))
			break;
		if (slot->action.weapon == prev_weapon_id)
			slot->action.weapon = new_weapon_id;
	}
}

void QNN_BackShiftFlushAll(qnn_tick_emit_state_t *emit)
{
	qnn_backshift_ring_t *ring = &g_backshift_ring;
	int i;
	int idx;

	/* Drain oldest-to-newest: oldest is at `head` when full, or at
	 * (head - count + K) % K otherwise. */
	idx = (ring->head + QNN_BACKSHIFT_K - ring->count) % QNN_BACKSHIFT_K;
	for (i = 0; i < ring->count; i++)
	{
		QNN_BackShiftEmitSlot(&ring->slots[idx], emit);
		idx = (idx + 1) % QNN_BACKSHIFT_K;
	}
	ring->count = 0;
	ring->head = 0;
	ring->has_prev_weapon_id = false;
	ring->prev_weapon_id = 0;
}
