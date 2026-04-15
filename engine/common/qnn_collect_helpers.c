/*
 * qnn_collect_helpers.c — Engine-agnostic helpers shared by the NQ
 * and QW collect main loops.
 *
 * Contents:
 *   - Entity scanners for movers, push triggers, and other players
 *   - Jitter filter math (3-frame XY direction reversal detection)
 *   - Frozen-action predicate
 *   - QOBS framed tick emitter
 *
 * Touches only shared state (qnn_store, cl_entities, cl.model_precache,
 * cl.maxclients, cl.viewentity — all present natively on NQ and
 * synthesized on QW by QNN_SyncEngineCompat).
 */

#include "qnn.h"
#include "qnn_io.h"
#include "qnn_store.h"

#include <math.h>
#include <string.h>

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
 * zero look-delta, no fire/switch.  Used by the emit-rate filter to cap
 * runs of frozen frames. */
qboolean QNN_ActionIsFrozen(const qnn_action_t *a)
{
	return a->move[0] == 0.0f && a->move[1] == 0.0f && a->move[2] == 0.0f
		&& a->fire == 0 && a->switch_slot == 0
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

/* Patch the emitted action into the obs buffer's last action history
 * slot so the training data has the correct T-1 action.  The obs was
 * packed before the jitter buffer delayed it, so its action history is
 * stale.  This overwrites token 7 (most recent) in-place. */
static void QNN_PatchActionHistory(uint8_t *obs, const qnn_action_t *action)
{
	int off = QNN_OBS_OFF_ACTION_HISTORY + (QNN_OBS_ACTION_HISTORY_LEN - 1)
		* QNN_OBS_ACTION_HISTORY_DIM * (int)sizeof(float);
	float features[QNN_OBS_ACTION_HISTORY_DIM];
	features[0] = action->move[0];
	features[1] = action->move[1];
	features[2] = action->move[2];
	features[3] = action->look[0];
	features[4] = action->look[1];
	features[5] = action->look[2];
	features[6] = (float)action->fire;
	features[7] = (float)(action->switch_slot < 0 ? 0
		: action->switch_slot > QNN_ACTION_SWITCH_SLOTS
		? QNN_ACTION_SWITCH_SLOTS : action->switch_slot)
		/ (float)QNN_ACTION_SWITCH_SLOTS;
	memcpy(obs + off, features, sizeof(features));
}

static void QNN_TickEmitFlush(qnn_tick_emit_state_t *st)
{
	if (!st->has_jitter_buf)
		return;
	if (st->jitter_out != NULL)
	{
		if (st->has_prev_emitted)
			QNN_PatchActionHistory(st->jitter_obs, &st->prev_emitted_action);
		QNN_EmitTick(st->jitter_out, st->jitter_obs,
			&st->jitter_action,
			st->jitter_tick, st->jitter_steps,
			st->jitter_tick_hz, st->jitter_flags);
		st->prev_emitted_action = st->jitter_action;
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

void QNN_WriteObsTick(qnn_tick_emit_state_t *st, FILE *out,
	const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz,
	qboolean reset_flag)
{
	uint8_t cur_obs[QNN_OBS_BUFFER_SIZE];
	qnn_tick_result_t result;
	uint16_t flags = 0;

	(void)tick_hz;
	(void)reset_flag;
	QNN_IOEmit(snapshot, &result);
	QNN_IOPackObsBuffer(cur_obs, &result);
	/* Action history is NOT pushed into the global ring buffer here.
	   The jitter buffer delays emission by 2 frames, so a capture-time
	   push would leave stale data in the ring buffer by emit time.
	   Instead, QNN_PatchActionHistory overwrites the last history slot
	   (token 7) at emit time with the correct T-1 action.
	   The ring buffer push (QNN_IOPushAction) is only used by the
	   trainer/inference path which has no jitter buffer. */

	if (!st->has_buffered_obs)
	{
		if (snapshot->done)
		{
			if (out != NULL)
			{
				QNN_EmitTick(out, cur_obs, &snapshot->action_label,
					tick, steps, tick_hz, 0x02);
			}
			return;
		}
		memcpy(st->buffered_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
		st->has_buffered_obs = true;
		return;
	}

	if (snapshot->done) flags |= 0x02;
	if (reset_flag) flags |= 0x01;

	if (!st->has_jitter_buf)
	{
		/* First real pair: push into jitter buffer, can't filter yet. */
		memcpy(st->jitter_obs, st->buffered_obs, QNN_OBS_BUFFER_SIZE);
		st->jitter_action = snapshot->action_label;
		st->jitter_tick = tick;
		st->jitter_steps = steps;
		st->jitter_tick_hz = tick_hz;
		st->jitter_flags = flags;
		st->jitter_out = out;
		st->has_jitter_buf = true;
	}
	else
	{
		/* We have prev_prev (if any), jitter_buf (middle), and current
		   action (next).  Apply the 3-frame filter to the middle. */
		if (st->has_prev_prev_move)
			QNN_JitterFilter(&st->jitter_action,
				st->prev_prev_move,
				snapshot->action_label.move);

		/* Emit the (possibly corrected) jitter-buffered tick. */
		if (st->jitter_out != NULL)
		{
			if (st->has_prev_emitted)
				QNN_PatchActionHistory(st->jitter_obs, &st->prev_emitted_action);
			QNN_EmitTick(st->jitter_out, st->jitter_obs,
				&st->jitter_action,
				st->jitter_tick,
				st->jitter_steps,
				st->jitter_tick_hz,
				st->jitter_flags);
			st->prev_emitted_action = st->jitter_action;
			st->has_prev_emitted = true;
		}

		/* Advance: jitter_buf move becomes prev_prev, current becomes jitter_buf. */
		st->prev_prev_move[0] = st->jitter_action.move[0];
		st->prev_prev_move[1] = st->jitter_action.move[1];
		st->prev_prev_move[2] = st->jitter_action.move[2];
		st->has_prev_prev_move = true;

		memcpy(st->jitter_obs, st->buffered_obs, QNN_OBS_BUFFER_SIZE);
		st->jitter_action = snapshot->action_label;
		st->jitter_tick = tick;
		st->jitter_steps = steps;
		st->jitter_tick_hz = tick_hz;
		st->jitter_flags = flags;
		st->jitter_out = out;
	}

	if (snapshot->done)
	{
		QNN_TickEmitFlush(st);
		st->has_buffered_obs = false;
		return;
	}

	memcpy(st->buffered_obs, cur_obs, QNN_OBS_BUFFER_SIZE);
}
