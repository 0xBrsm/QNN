/*
 * qnn_collect_helpers.c — Engine-agnostic helpers shared by the NQ
 * and QW collect main loops.
 *
 * Contents:
 *   - Entity scanners for movers, push triggers, and other players
 *   - 3-frame jitter filter math (XY direction reversal detection)
 *   - Frozen-action predicate
 *   - QOBS framed tick emitter + two-level buffer (obs delay + jitter)
 *   - QNN_DetectFireEvent (shared by both QWD and MVD inference paths)
 *   - QNN_EmitFilter (dead/frozen/god-mode rate caps)
 *   - QNN_PackSnapshotObs / QNN_WriteObsTickPrepacked* (used by both
 *     the live emit path and the MVD back-shift ring drain)
 *   - QNN_EmitLabelerTick (LOBS writer used by labeler-mode emit)
 *   - QNN_SavePrev (per-tick state advance)
 *   - QNN_FillLookAndSwitch (look/weapon label, shared QWD + MVD)
 *
 * Touches only shared state (qnn_store, cl_entities, cl.model_precache,
 * cl.maxclients, cl.viewentity — all present natively on NQ and
 * synthesized on QW by QNN_SyncEngineCompat).
 *
 * MVD reconstruction (back-shift ring, log-normal hold tails) and the
 * labeler LOBS emit branch live in their own modules — see
 * qnn_mvd_collect.c / qnn_labeler_collect.c.
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

static void QNN_DumpFireRoutes(int row, int tick, int steps,
	const qnn_action_t *action, const qnn_fire_route_event_t *routes,
	int route_count)
{
	const char *path;
	FILE *out;
	int i;

	if (route_count <= 0)
		return;
	path = getenv("QNN_FIRE_ROUTE_DUMP");
	if (path == NULL || path[0] == '\0')
		return;
	out = fopen(path, "a");
	if (out == NULL)
		return;
	for (i = 0; i < route_count; ++i)
	{
		const qnn_fire_route_event_t *r = &routes[i];
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
			"\"emitted_fire\":%d}\n",
			r->source_tick, r->dest_tick, r->sound_index,
			r->weapon_id, r->native_time, r->emit_start_native,
			r->ping_sec, r->phase, r->press_offset,
			r->deterministic_offset,
			r->route_offset, action->fire ? 1 : 0);
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

/* 3-frame jitter filter: if the buffered tick's XY move reverses
 * the previous tick (dot < -0.866, >150°) and the current tick
 * agrees with the previous (dot > 0.6, <~45°), replace the
 * buffered tick's move with the previous tick's move.
 * Only operates on the XY plane; Z is independent (jump/swim).
 *
 * Pure math — no globals.  Used by the WriteObsTick integration in
 * each engine's collect main. */
void QNN_JitterFilter(qnn_action_t *mid, const float *prev_move,
	const float *next_move)
{
	float mp, mm, mn;
	float dp[2], dm[2], dn[2];
	float dot_neighbors, dot_mid_prev, dot_mid_next;

	mp = sqrtf(prev_move[0] * prev_move[0] + prev_move[1] * prev_move[1]);
	mm = sqrtf(mid->move[0] * mid->move[0] + mid->move[1] * mid->move[1]);
	mn = sqrtf(next_move[0] * next_move[0] + next_move[1] * next_move[1]);

	if (mp < 0.01f || mm < 0.01f || mn < 0.01f)
		return;

	dp[0] = prev_move[0] / mp;  dp[1] = prev_move[1] / mp;
	dm[0] = mid->move[0] / mm;  dm[1] = mid->move[1] / mm;
	dn[0] = next_move[0] / mn;  dn[1] = next_move[1] / mn;

	dot_neighbors = dp[0] * dn[0] + dp[1] * dn[1];
	dot_mid_prev  = dm[0] * dp[0] + dm[1] * dp[1];
	dot_mid_next  = dm[0] * dn[0] + dm[1] * dn[1];

	if (dot_neighbors > 0.6f && dot_mid_prev < 0.0f && dot_mid_next < 0.0f)
	{
		mid->move[0] = prev_move[0];
		mid->move[1] = prev_move[1];
	}
}

/* True when the action label represents a fully idle tick — zero move,
 * zero look-delta, no fire.  Used by the emit-rate filter to cap runs
 * of frozen frames.  Collect workers store dense full weapon intent in
 * weapon, so it's not part of the activity check; standing on the
 * same weapon doesn't constitute movement. */
qboolean QNN_ActionIsFrozen(const qnn_action_t *a)
{
	return a->move[0] == 0.0f && a->move[1] == 0.0f && a->move[2] == 0.0f
		&& a->fire == 0
		&& fabsf(a->look[0] - 1.0f) < QNN_FROZEN_LOOK_TOL
		&& fabsf(a->look[1]) < QNN_FROZEN_LOOK_TOL
		&& fabsf(a->look[2]) < QNN_FROZEN_LOOK_TOL;
}

/* Emit one framed tick: "QOBS" magic + 16-byte header + obs + action.
 * Header is (tick, steps, tick_hz, flags, action_size) little-endian. */
void QNN_EmitTick(FILE *out, const uint8_t *obs, const qnn_action_t *action,
	int tick, int steps, int tick_hz, uint16_t flags)
{
	uint8_t header[16];
	uint16_t asize = (uint16_t)sizeof(qnn_action_t);

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

/* ── Shared tick-emission pipeline ───────────────────────────────
 *
 * Two-level buffer:
 *   Level 1 (obs buffer): obs from tick t is held until tick t+1 so we
 *   can pair it with the action computed at t+1.
 *   Level 2 (jitter buffer): the paired obs+action is held one more tick
 *   so the 3-frame jitter filter can check it against prev and next.
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
		QNN_DumpFireRoutes(st->emitted_rows, st->jitter_tick,
			st->jitter_steps, &st->jitter_action,
			st->jitter_fire_routes,
			st->jitter_fire_route_count);
		st->emitted_rows++;
		st->has_prev_emitted = true;
	}
	st->has_prev_prev_move = true;
	st->prev_prev_move[0] = st->jitter_action.move[0];
	st->prev_prev_move[1] = st->jitter_action.move[1];
	st->prev_prev_move[2] = st->jitter_action.move[2];
	st->has_jitter_buf = false;
}

void QNN_FlushTickEmit(qnn_tick_emit_state_t *st)
{
	QNN_TickEmitFlush(st);
}

void QNN_PackSnapshotObs(const qnn_snapshot_t *snapshot, uint8_t *obs_out)
{
	qnn_tick_result_t result;
	QNN_IOEmit(snapshot, &result);
	QNN_IOPackObsBuffer(obs_out, &result);
}

/* Internal: drive the obs/jitter pipeline from pre-packed obs bytes.
 * Shared body of QNN_WriteObsTick (which packs first) and
 * QNN_WriteObsTickPrepacked (caller already packed at capture time
 * to survive deferred emit through the MVD back-shift ring). */
static void QNN_WriteObsTickInner(qnn_tick_emit_state_t *st, FILE *out,
	const uint8_t *cur_obs, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz, qboolean reset_flag,
	const qnn_fire_route_event_t *routes, int route_count)
{
	uint16_t flags = 0;
	if (route_count < 0)
		route_count = 0;
	if (route_count > QNN_MAX_FIRE_ROUTE_EVENTS)
		route_count = QNN_MAX_FIRE_ROUTE_EVENTS;

	if (!st->has_buffered_obs)
	{
		if (done)
		{
			if (out != NULL)
			{
				QNN_EmitTick(out, cur_obs, action,
					tick, steps, tick_hz, 0x02);
				QNN_DumpFireRoutes(st->emitted_rows, tick,
					steps, action, routes, route_count);
				st->emitted_rows++;
			}
			return;
		}
		memcpy(st->buffered_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
		st->has_buffered_obs = true;
		return;
	}

	if (done) flags |= 0x02;
	if (reset_flag) flags |= 0x01;

	if (!st->has_jitter_buf)
	{
		/* First real pair: push into jitter buffer, can't filter yet. */
		memcpy(st->jitter_obs, st->buffered_obs, QNN_OBS_BUFFER_SIZE);
		st->jitter_action = *action;
		st->jitter_tick = tick;
		st->jitter_steps = steps;
		st->jitter_tick_hz = tick_hz;
		st->jitter_flags = flags;
		st->jitter_out = out;
		st->jitter_fire_route_count = route_count;
		if (route_count > 0)
			memcpy(st->jitter_fire_routes, routes,
				sizeof(qnn_fire_route_event_t) * route_count);
		st->has_jitter_buf = true;
	}
	else
	{
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
			QNN_DumpFireRoutes(st->emitted_rows,
				st->jitter_tick, st->jitter_steps,
				&st->jitter_action,
				st->jitter_fire_routes,
				st->jitter_fire_route_count);
			st->emitted_rows++;
			st->has_prev_emitted = true;
		}

		/* Advance: jitter_buf move becomes prev_prev, current becomes jitter_buf. */
		st->prev_prev_move[0] = st->jitter_action.move[0];
		st->prev_prev_move[1] = st->jitter_action.move[1];
		st->prev_prev_move[2] = st->jitter_action.move[2];
		st->has_prev_prev_move = true;

		memcpy(st->jitter_obs, st->buffered_obs, QNN_OBS_BUFFER_SIZE);
		st->jitter_action = *action;
		st->jitter_tick = tick;
		st->jitter_steps = steps;
		st->jitter_tick_hz = tick_hz;
		st->jitter_flags = flags;
		st->jitter_out = out;
		st->jitter_fire_route_count = route_count;
		if (route_count > 0)
			memcpy(st->jitter_fire_routes, routes,
				sizeof(qnn_fire_route_event_t) * route_count);
	}

	if (done)
	{
		QNN_TickEmitFlush(st);
		st->has_buffered_obs = false;
		return;
	}

	memcpy(st->buffered_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
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

void QNN_WriteObsTickPrepackedWithFireRoutes(qnn_tick_emit_state_t *st,
	FILE *out, const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, const qnn_fire_route_event_t *routes,
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

/* Single-frame fire-event detection used by both NQ and QW MVD paths.
 * Returns true if THIS native frame contains a shot event:
 *   - a weapon-fire sound matching the currently-held weapon, OR
 *   - an ammo decrement on the same weapon as the previous frame.
 * The weapon-id guard on the ammo branch prevents weapon-switch
 * cascades — snapshot->ammo jumps between weapons' ammo pools and
 * would otherwise read as a "decrement". */
qboolean QNN_DetectFireEvent(const qnn_snapshot_t *snapshot)
{
	static FILE *fire_source_dump = NULL;
	static qboolean fire_source_dump_checked = false;
	qboolean sound_fire;
	qboolean ammo_drop;

	if (!qnn_runtime.has_prev)
		return false;
	sound_fire = QNN_SnapshotHasSelfWeaponFireSound(snapshot);
	ammo_drop = (snapshot->weapon_id == qnn_runtime.prev_weapon_id
		&& snapshot->ammo < qnn_runtime.prev_ammo) ? true : false;
	if (!fire_source_dump_checked)
	{
		const char *path = getenv("QNN_FIRE_SOURCE_DUMP");
		fire_source_dump_checked = true;
		if (path != NULL && path[0] != '\0')
			fire_source_dump = fopen(path, "a");
	}
	if (fire_source_dump != NULL && (sound_fire || ammo_drop))
	{
		fprintf(fire_source_dump,
			"{\"tick\":%d,\"sound\":%d,\"ammo\":%d,"
			"\"weapon\":%d,\"prev_weapon\":%d,\"ammo_count\":%d,"
			"\"prev_ammo\":%d}\n",
			qnn_runtime.tick, sound_fire ? 1 : 0, ammo_drop ? 1 : 0,
			snapshot->weapon_id, qnn_runtime.prev_weapon_id,
			snapshot->ammo, qnn_runtime.prev_ammo);
		fflush(fire_source_dump);
	}
	return (sound_fire || ammo_drop) ? true : false;
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

	if (action->weapon == 0 && snapshot->weapon_id > 0)
		action->weapon = snapshot->weapon_id;
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

	/* Dead: keep first QNN_DEAD_MAX_EMIT frames, inject fire=1, zero
	 * move (corpse physics ≠ player input — alive accel/friction vs
	 * dead velocity decay diverges). */
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
