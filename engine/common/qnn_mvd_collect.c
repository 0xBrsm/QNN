/*
 * qnn_mvd_collect.c — MVD reconstruction path for the QW demo worker.
 *
 * Used when no usercmd_t is available: real MVD demos
 * (cls.mvdplayback) or `force_mvd_emit` on QWD.  Three discrete-cmd
 * inference paths and one move path live here, all wired through a
 * single back-shift ring that lets server-observed sound/state events
 * write labels back at the press frame instead of the broadcast frame.
 *
 *   FIRE   sound (weapon-fire PHS multicast) → walkback by full ping
 *          → set slot.action.fire = 1 → co-temporal dedup → chain-fill
 *          across short cooldown gaps → forward log-normal hold tail.
 *
 *   JUMP   sound (player/plyrjmp8.wav) → walkback by full ping → set
 *          slot.action.move[2] = jump_speed → grounded-count chain
 *          gate → forward log-normal hold tail.
 *
 *   SWITCH per-emit `action.weapon = snapshot.weapon_id`.  On
 *          weapon_id transitions, rewrite the trailing K slots back to
 *          the press frame.  Pickup gate (IT_ bit 0→1) suppresses the
 *          rewrite when the transition was a server-forced touch
 *          (impulse handlers can't fire on a bit that wasn't set).
 *
 *   MOVE   per-emit fb/lr from view-relative position-delta sign
 *          (`QNN_MvdInferEmitMove`).  fb/lr back-shifted into the ring
 *          by `QNN_MvdBackShiftWriteMoveXY`; ud is set via the JUMP
 *          path above (water gets ud from snap-time position delta).
 *
 * Hold-tail samplers live in qnn_hold_samplers.c — the per-weapon and
 * jump CDFs are fit on QWD truth, and only run when MVD inference
 * emits BC training labels (NOT when the labeler is collecting its
 * sparse one-tick-per-event training signal — that uses the QWD path
 * in qnn_qwd_collect.c).
 *
 * Module-private state lives in the file-scope `mvd_state` struct.
 * `QNN_MvdCollectReset` is the per-demo reset entry point invoked by
 * the QW main loop.
 *
 * Cross-module touch points:
 *   - `qnn_runtime.tick` / `fixed_tick_hz` (timing scale) — read-only.
 *   - `qnn_runtime.native_fire_this_window` — mirrored for NQ's MVD path
 *     which shares the field (NQ has its own native-fire emit logic in
 *     nq/qnn_collect_main.c).
 *   - `qnn_runtime.emit_*` / `prev_*` / mover refs — read-only state from
 *     the main loop's emit anchor + per-tick capture.
 */

#include "qnn_mvd_collect.h"
#include "qnn_hold_samplers.h"
#include "qnn_io.h"
#include "qnn_store.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/* ══════════════════════════════════════════════════════════════════════
 *                       MODULE-PRIVATE STATE
 * ══════════════════════════════════════════════════════════════════════ */

typedef struct
{
	qnn_backshift_ring_t backshift;
	/* MVD-path fire hold extension.  When a tap-weapon shot is back-
	 * shifted into a ring slot, a hold duration is sampled from a
	 * truncated log-normal fit and fire=1 is written forward into
	 * subsequent emit slots.  Slots that haven't been pushed yet are
	 * covered by fire_hold_remaining: each push decrements the counter
	 * and sets fire=1 on the new slot until it reaches zero. */
	int      fire_hold_remaining;
	uint32_t fire_hold_rng;       /* xorshift32; seeded per demo */
	/* MVD-path jump hold extension — direct analog of fire. */
	int      jump_hold_remaining;
	uint32_t jump_hold_rng;
	/* Per-weapon fire chain-fill state.  Index = QNN subject weapon id
	 * (3..10); slot 0 unused.  See qnn_collect_helpers.h's comment for
	 * the dedup semantics. */
	int      last_fire_shot_tick[11];
	int      last_fire_source_tick[11];
	/* Jump chain-fill / dedup state — single global counter. */
	int      last_jump_shot_tick;
	int      last_jump_source_tick;
} qnn_mvd_state_t;

static qnn_mvd_state_t mvd_state;


/* ══════════════════════════════════════════════════════════════════════
 *                  SHARED BACK-SHIFT RING (used by all paths)
 * ══════════════════════════════════════════════════════════════════════ */

static int QNN_BackShiftLatestIndex(const qnn_backshift_ring_t *ring)
{
	return (ring->head + QNN_BACKSHIFT_K - 1) % QNN_BACKSHIFT_K;
}

static qboolean QNN_BackShiftSlotAt(qnn_backshift_ring_t *ring,
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

/* Convert a press-time offset (seconds, negative = past) to a ring-
 * slot index back from the latest push.  Returns -1 if the press
 * lands beyond the ring's history.  Resolution is 50ms (one slot per
 * 20Hz emit). */
static int QNN_BackShiftOffsetFromPress(float press_offset_sec, int count)
{
	int offset;

	if (press_offset_sec >= 0.0f)
		return 0;
	offset = (int)ceilf(-press_offset_sec / 0.05f);
	if (offset >= count)
		return -1;
	return offset;
}

static qnn_backshift_slot_t *QNN_BackShiftSlotForPress(
	qnn_backshift_ring_t *ring, float press_offset_sec, int *offset_out)
{
	int offset;
	qnn_backshift_slot_t *slot;

	offset = QNN_BackShiftOffsetFromPress(press_offset_sec, ring->count);
	if (offset < 0 || !QNN_BackShiftSlotAt(ring, offset, &slot))
		return NULL;
	if (offset_out != NULL)
		*offset_out = offset;
	return slot;
}

/* Walk the sound event back to its press tick.  In demo time,
 * sound-receive lags cmd-send by one full RTT (cmd→server ping/2 +
 * sound→client ping/2 = full ping).  Caller supplies the per-press
 * ping from the latency tracker. */
static qnn_backshift_slot_t *QNN_BackShiftSlotForSound(
	qnn_backshift_ring_t *ring,
	const qnn_sound_event_t *snd,
	float ping_sec,
	float emit_start_native_time,
	int *offset_out)
{
	float phase = snd->native_time - emit_start_native_time;
	float press_offset = phase - ping_sec;
	return QNN_BackShiftSlotForPress(ring, press_offset, offset_out);
}

static int QNN_PingSecToEmitFrames(float ping_sec)
{
	if (qnn_runtime.fixed_tick_hz <= 0)
		return 0;
	return (int)(ping_sec * (float)qnn_runtime.fixed_tick_hz + 0.5f);
}


/* ── Generic press-event pipeline (dedup + chain-fill + hold tail) ── */
/*
 * Shared between the FIRE and JUMP paths.  Each path supplies callbacks
 * for the action mutation, chain-gate decision, chain-fill (across slots
 * between the previous and current event), and hold-tail sampler.
 *
 * For a sparse-only training signal (one tick per event, no hold,
 * no chain-fill), callers can supply no-op chain_fill and a hold
 * sampler that returns 0.  Today both fire and jump pass real
 * generators because BC labels want the human-feel hold semantics.
 */

typedef void (*qnn_press_set_action_fn)(qnn_action_t *action);

typedef qboolean (*qnn_press_chain_gate_fn)(
	qnn_backshift_ring_t *ring,
	int prev_tick, int cur_tick,
	float ping_sec, void *ctx);

typedef void (*qnn_press_chain_fill_fn)(
	qnn_backshift_ring_t *ring,
	int prev_tick, int cur_tick);

typedef int (*qnn_press_hold_sample_fn)(void *ctx);

typedef struct
{
	qnn_press_set_action_fn   set_action;
	int                      *last_dest_tick;
	int                      *last_source_tick;
	qnn_press_chain_gate_fn   chain_gate;
	qnn_press_chain_fill_fn   chain_fill;
	void                     *chain_gate_ctx;
	qnn_press_hold_sample_fn  hold_sample;
	void                     *hold_sample_ctx;
	int                      *hold_remaining;
} qnn_press_event_cfg_t;

static qboolean QNN_BackShiftApplyPressEvent(
	qnn_backshift_ring_t *ring,
	qnn_backshift_slot_t *slot,
	int offset,
	float ping_sec,
	const qnn_press_event_cfg_t *cfg)
{
	int cur_tick;
	int prev_tick;
	int ext;
	int i;

	/* Dedup co-temporal duplicate sounds for the same event.  Two
	 * distinct presses can't land in the same source emit tick
	 * (qnn_runtime.tick), but a single press may emit multiple sounds
	 * that back-shift to the same destination slot (e.g. LG lstart +
	 * lhit) — those must collapse to one event. */
	if (*cfg->last_dest_tick == slot->tick
		&& *cfg->last_source_tick == qnn_runtime.tick
		&& slot->tick != 0)
		return false;

	cfg->set_action(&slot->action);

	cur_tick = slot->tick;
	prev_tick = *cfg->last_dest_tick;
	if (prev_tick > 0 && cur_tick > prev_tick
		&& cfg->chain_gate(ring, prev_tick, cur_tick,
			ping_sec, cfg->chain_gate_ctx))
	{
		cfg->chain_fill(ring, prev_tick, cur_tick);
	}

	*cfg->last_dest_tick = cur_tick;
	*cfg->last_source_tick = qnn_runtime.tick;

	/* Forward tail.  Sampled after dedup so the RNG stream advances
	 * once per emitted event, not per incoming sound. */
	ext = cfg->hold_sample(cfg->hold_sample_ctx);
	for (i = 1; i < ext && i <= offset; i++)
	{
		qnn_backshift_slot_t *s;
		if (QNN_BackShiftSlotAt(ring, offset - i, &s))
			cfg->set_action(&s->action);
	}
	if (ext > 1 && cfg->hold_remaining != NULL)
		*cfg->hold_remaining =
			(ext - 1 > offset) ? (ext - 1 - offset) : 0;

	return true;
}


/* ── Sound-event iterator (shared by FIRE and JUMP path apply fns) ── */

typedef qboolean (*qnn_backshift_sound_predicate_t)(
	const qnn_sound_event_t *snd);
typedef void (*qnn_backshift_sound_handler_t)(
	qnn_backshift_ring_t *ring,
	const qnn_snapshot_t *snapshot,
	const qnn_sound_event_t *snd,
	int sound_index,
	float ping_sec,
	float emit_start_native_time);

static void QNN_BackShiftForSoundEvents(qnn_backshift_ring_t *ring,
	const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time,
	qnn_backshift_sound_predicate_t match,
	qnn_backshift_sound_handler_t apply)
{
	int i;

	if (ring->count == 0)
		return;

	for (i = 0; i < snapshot->sound_count; ++i)
	{
		const qnn_sound_event_t *snd = &snapshot->sounds[i];

		if (!match(snd))
			continue;
		apply(ring, snapshot, snd, i, ping_sec, emit_start_native_time);
	}
}


/* ══════════════════════════════════════════════════════════════════════
 *                          FIRE PATH
 * ══════════════════════════════════════════════════════════════════════ */
/*
 * Sound (weapon-fire PHS multicast) → back-shift by full ping →
 * fire=1 in target ring slot → dedup against co-temporal duplicates
 * (LG lstart+lhit) → chain-fill across slots from prev fire event if
 * within cooldown + slack → log-normal forward hold tail.
 */

static void QNN_BackShiftFillBetween(qnn_backshift_ring_t *ring,
	int from_tick, int to_tick)
{
	int i;

	if (from_tick >= to_tick - 1)
		return;
	for (i = 0; i < ring->count; ++i)
	{
		qnn_backshift_slot_t *s;
		if (!QNN_BackShiftSlotAt(ring, i, &s))
			break;
		if (s->tick > from_tick && s->tick < to_tick)
			s->action.fire = 1;
	}
}

static void qnn_press_set_fire(qnn_action_t *a)
{
	a->fire = 1;
}

static qboolean qnn_press_chain_gate_fire(qnn_backshift_ring_t *ring,
	int prev_tick, int cur_tick,
	float ping_sec, void *ctx)
{
	int weapon_id = (int)(intptr_t)ctx;
	int cd_emit;
	int slack;

	(void)ring;
	if (weapon_id < 3 || weapon_id > 10)
		return false;
	cd_emit = QNN_FireCooldownEmit(weapon_id);
	if (cd_emit <= 0)
		return false;
	slack = 2 + QNN_PingSecToEmitFrames(ping_sec);
	return (cur_tick - prev_tick) <= cd_emit + slack;
}

static int qnn_press_hold_sample_fire(void *ctx)
{
	int weapon_id = (int)(intptr_t)ctx;
	return QNN_FireHoldFrames(weapon_id, &mvd_state.fire_hold_rng);
}

static void QNN_BackShiftWriteFireAtTime(qnn_backshift_ring_t *ring,
	const qnn_snapshot_t *snapshot, int sound_index,
	const qnn_sound_event_t *snd, float ping_sec,
	float emit_start_native_time)
{
	int offset;
	qnn_backshift_slot_t *slot;
	int weapon_id;
	qnn_press_event_cfg_t cfg;

	slot = QNN_BackShiftSlotForSound(ring, snd,
		ping_sec, emit_start_native_time, &offset);
	if (slot == NULL)
		return;

	weapon_id = snapshot->weapon_id;
	if (weapon_id < 0 || weapon_id > 10)
		return;

	cfg.set_action      = qnn_press_set_fire;
	cfg.last_dest_tick  = &mvd_state.last_fire_shot_tick[weapon_id];
	cfg.last_source_tick = &mvd_state.last_fire_source_tick[weapon_id];
	cfg.chain_gate      = qnn_press_chain_gate_fire;
	cfg.chain_fill      = QNN_BackShiftFillBetween;
	cfg.chain_gate_ctx  = (void *)(intptr_t)weapon_id;
	cfg.hold_sample     = qnn_press_hold_sample_fire;
	cfg.hold_sample_ctx = (void *)(intptr_t)weapon_id;
	cfg.hold_remaining  = &mvd_state.fire_hold_remaining;

	if (!QNN_BackShiftApplyPressEvent(ring, slot, offset, ping_sec, &cfg))
		return;

	if (slot->fire_route_count < QNN_MAX_FIRE_ROUTE_EVENTS)
	{
		qnn_fire_route_event_t *ev =
			&slot->fire_routes[slot->fire_route_count++];
		float phase = snd->native_time - emit_start_native_time;
		ev->source_tick = qnn_runtime.tick;
		ev->dest_tick = slot->tick;
		ev->sound_index = sound_index;
		ev->weapon_id = snapshot->weapon_id;
		ev->native_time = snd->native_time;
		ev->emit_start_native = emit_start_native_time;
		ev->ping_sec = ping_sec;
		ev->phase = phase;
		ev->press_offset = phase - ping_sec;
		ev->deterministic_offset = offset;
		ev->route_offset = offset;
	}
}

static qboolean QNN_BackShiftMatchFireSound(const qnn_sound_event_t *snd)
{
	return QNN_IsSelfWeaponFireSound(snd);
}

static void QNN_BackShiftApplyFireSound(qnn_backshift_ring_t *ring,
	const qnn_snapshot_t *snapshot,
	const qnn_sound_event_t *snd, int sound_index,
	float ping_sec, float emit_start_native_time)
{
	QNN_BackShiftWriteFireAtTime(ring, snapshot, sound_index,
		snd, ping_sec, emit_start_native_time);
}


/* ══════════════════════════════════════════════════════════════════════
 *                          JUMP PATH
 * ══════════════════════════════════════════════════════════════════════ */
/*
 * Sound (player/plyrjmp8.wav) → back-shift by full ping →
 * move[2]=jump_speed in target ring slot → grounded-count chain gate
 * (allow chain-fill only when intervening slots are mostly grounded —
 * a bhop chain) → log-normal forward hold tail.
 */

static void QNN_BackShiftFillJumpBetween(qnn_backshift_ring_t *ring,
	int from_tick, int to_tick)
{
	int i;

	if (from_tick >= to_tick - 1)
		return;
	for (i = 0; i < ring->count; ++i)
	{
		qnn_backshift_slot_t *s;
		if (!QNN_BackShiftSlotAt(ring, i, &s))
			break;
		if (s->tick > from_tick && s->tick < to_tick)
			s->action.move[2] = QNN_SV_JUMP_SPEED / QNN_SV_MAXSPEED;
	}
}

static int QNN_BackShiftJumpGroundedCount(qnn_backshift_ring_t *ring,
	int from_tick, int to_tick)
{
	int i;
	int count = 0;

	if (from_tick >= to_tick)
		return 0;
	for (i = 0; i < ring->count; ++i)
	{
		qnn_backshift_slot_t *s;
		if (!QNN_BackShiftSlotAt(ring, i, &s))
			break;
		if (s->tick > from_tick && s->tick < to_tick && s->grounded)
			count++;
	}
	return count;
}

static void qnn_press_set_jump(qnn_action_t *a)
{
	a->move[2] = QNN_SV_JUMP_SPEED / QNN_SV_MAXSPEED;
}

static qboolean qnn_press_chain_gate_jump(qnn_backshift_ring_t *ring,
	int prev_tick, int cur_tick,
	float ping_sec, void *ctx)
{
	int slack;
	int grounded_count;

	(void)ctx;
	slack = 2 + QNN_PingSecToEmitFrames(ping_sec);
	grounded_count = QNN_BackShiftJumpGroundedCount(ring,
		prev_tick, cur_tick);
	return grounded_count <= slack;
}

static int qnn_press_hold_sample_jump(void *ctx)
{
	(void)ctx;
	return QNN_JumpHoldFrames(&mvd_state.jump_hold_rng);
}

static qboolean QNN_BackShiftMatchJumpSound(const qnn_sound_event_t *snd)
{
	return QNN_IsSelfJumpSound(snd);
}

static void QNN_BackShiftApplyJumpSound(qnn_backshift_ring_t *ring,
	const qnn_snapshot_t *snapshot,
	const qnn_sound_event_t *snd, int sound_index,
	float ping_sec, float emit_start_native_time)
{
	int offset;
	qnn_backshift_slot_t *slot;
	qnn_press_event_cfg_t cfg;
	(void)snapshot;
	(void)sound_index;

	slot = QNN_BackShiftSlotForSound(ring, snd,
		ping_sec, emit_start_native_time, &offset);
	if (slot == NULL)
		return;

	cfg.set_action      = qnn_press_set_jump;
	cfg.last_dest_tick  = &mvd_state.last_jump_shot_tick;
	cfg.last_source_tick = &mvd_state.last_jump_source_tick;
	cfg.chain_gate      = qnn_press_chain_gate_jump;
	cfg.chain_fill      = QNN_BackShiftFillJumpBetween;
	cfg.chain_gate_ctx  = NULL;
	cfg.hold_sample     = qnn_press_hold_sample_jump;
	cfg.hold_sample_ctx = NULL;
	cfg.hold_remaining  = &mvd_state.jump_hold_remaining;

	(void)QNN_BackShiftApplyPressEvent(ring, slot, offset, ping_sec, &cfg);
}


/* ══════════════════════════════════════════════════════════════════════
 *                          SWITCH PATH
 * ══════════════════════════════════════════════════════════════════════ */
/*
 * Per-emit action.weapon = snapshot.weapon_id (filled by
 * QNN_FillLookAndSwitch — shared with QWD path).  On observed
 * weapon_id transitions, rewrite the trailing `shift_frames` slots
 * back to the new weapon to anchor intent at the press frame instead
 * of the broadcast frame.
 *
 * Pickup gate (handled at the call site in qw/qnn_collect_main.c)
 * suppresses rewrite when the transition was a server-forced touch
 * (IT_ bit 0→1 on the same emit): impulse handlers can't fire on a
 * bit that wasn't already on, so a player-intent switch can only land
 * ≥1 frame later — leave that label at the server-observed frame.
 */

void QNN_MvdBackShiftOnWeaponChange(int new_weapon_id,
	int prev_weapon_id, int shift_frames)
{
	qnn_backshift_ring_t *ring = &mvd_state.backshift;
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


/* ══════════════════════════════════════════════════════════════════════
 *                          MOVE PATH
 * ══════════════════════════════════════════════════════════════════════ */
/*
 * fb/lr from view-relative position-delta sign at emit time
 * (QNN_MvdInferEmitMove), then back-shifted into the ring by
 * QNN_MvdBackShiftWriteMoveXY so the label aligns with the press frame.
 *
 * ud is filled by the JUMP path above; water medium gets ud from snap-
 * time position delta in QNN_MvdInferEmitMove.
 */

void QNN_MvdBackShiftWriteMoveXY(const float move[3], int shift_frames)
{
	qnn_backshift_ring_t *ring = &mvd_state.backshift;
	qnn_backshift_slot_t *slot;
	qnn_backshift_slot_t *latest;

	if (ring->count == 0 || shift_frames <= 0)
		return;
	if (!QNN_BackShiftSlotAt(ring, 0, &latest)
		|| !QNN_BackShiftSlotAt(ring, shift_frames, &slot)
		|| slot == latest)
		return;
	/* Only back-shift fb/lr — these come from view-relative velocity
	 * sign, which lags the keyboard press by ping+tick like fire and
	 * switch.  move[2] is back-shifted per-event by the jump writer. */
	slot->action.move[0] = move[0];
	slot->action.move[1] = move[1];
}


/* ══════════════════════════════════════════════════════════════════════
 *                  RING DRAIN (back to the emit pipeline)
 * ══════════════════════════════════════════════════════════════════════ */

static void QNN_BackShiftEmitSlot(qnn_backshift_slot_t *slot,
	qnn_tick_emit_state_t *emit)
{
	if (!slot->valid)
		return;
	QNN_WriteObsTickPrepackedWithFireRoutes(emit, slot->out,
		slot->obs, &slot->action,
		slot->done, slot->tick, slot->steps, slot->tick_hz,
		slot->reset_flag, slot->fire_routes,
		slot->fire_route_count);
	slot->valid = false;
	slot->fire_route_count = 0;
}


/* ══════════════════════════════════════════════════════════════════════
 *                              PUBLIC API
 * ══════════════════════════════════════════════════════════════════════ */

void QNN_MvdCollectReset(uintptr_t demo_path_seed)
{
	memset(&mvd_state, 0, sizeof(mvd_state));
	mvd_state.fire_hold_rng = 0x12345678u ^ (uint32_t)demo_path_seed;
	mvd_state.jump_hold_rng = 0x9e3779b9u ^ (uint32_t)demo_path_seed;
}

qboolean QNN_MvdBackShiftPrevWeapon(int *prev_weapon_out)
{
	if (!mvd_state.backshift.has_prev_weapon_id)
		return false;
	if (prev_weapon_out != NULL)
		*prev_weapon_out = mvd_state.backshift.prev_weapon_id;
	return true;
}

qboolean QNN_MvdBackShiftPrevStatItems(int *prev_items_out)
{
	if (!mvd_state.backshift.has_prev_stat_items)
		return false;
	if (prev_items_out != NULL)
		*prev_items_out = mvd_state.backshift.prev_stat_items;
	return true;
}

int QNN_MvdBackShiftCount(void)
{
	return mvd_state.backshift.count;
}

void QNN_MvdBackShiftPush(qnn_tick_emit_state_t *emit, FILE *out,
	const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, qboolean grounded,
	int weapon_id, int stat_items)
{
	qnn_backshift_ring_t *ring = &mvd_state.backshift;
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
	slot->fire_route_count = 0;

	/* Apply any hold extension that spilled past the previous head. */
	if (mvd_state.fire_hold_remaining > 0)
	{
		slot->action.fire = 1;
		mvd_state.fire_hold_remaining--;
	}
	if (mvd_state.jump_hold_remaining > 0)
	{
		slot->action.move[2] = QNN_SV_JUMP_SPEED / QNN_SV_MAXSPEED;
		mvd_state.jump_hold_remaining--;
	}

	ring->head = (ring->head + 1) % QNN_BACKSHIFT_K;
	ring->prev_weapon_id = weapon_id;
	ring->has_prev_weapon_id = true;
	ring->prev_stat_items = stat_items;
	ring->has_prev_stat_items = true;
}

void QNN_MvdBackShiftWriteFireEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time)
{
	QNN_BackShiftForSoundEvents(&mvd_state.backshift, snapshot,
		ping_sec, emit_start_native_time,
		QNN_BackShiftMatchFireSound,
		QNN_BackShiftApplyFireSound);
}

void QNN_MvdBackShiftWriteJumpEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time)
{
	QNN_BackShiftForSoundEvents(&mvd_state.backshift, snapshot,
		ping_sec, emit_start_native_time,
		QNN_BackShiftMatchJumpSound,
		QNN_BackShiftApplyJumpSound);
}

void QNN_MvdBackShiftFlushAll(qnn_tick_emit_state_t *emit)
{
	qnn_backshift_ring_t *ring = &mvd_state.backshift;
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

void QNN_MvdResetFireChain(int weapon_id)
{
	if (weapon_id >= 0 && weapon_id <= 10)
	{
		mvd_state.last_fire_shot_tick[weapon_id] = 0;
		mvd_state.last_fire_source_tick[weapon_id] = 0;
	}
	mvd_state.fire_hold_remaining = 0;
}


/* ══════════════════════════════════════════════════════════════════════
 *               ACTION INFERENCE (native + emit entry points)
 * ══════════════════════════════════════════════════════════════════════ */

/* Fill action->look (view-relative turn delta) and action->weapon (per-
 * frame held-weapon state) from the current snapshot.  Shared with the
 * QWD path (see qnn_qwd_collect.c) — declared in qnn_collect_helpers.h. */
extern void QNN_FillLookAndSwitch(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

void QNN_MvdInferNativeAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);
	if (QNN_DetectFireEvent(snapshot))
	{
		action->fire = 1;
		qnn_runtime.native_fire_this_window = true;
	}
	/* Jump is handled by QNN_MvdBackShiftWriteJumpEvents at emit time. */
}

void QNN_MvdInferEmitAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);
	QNN_FillLookAndSwitch(action, snapshot);
	action->fire = 0;
}

void QNN_MvdInferEmitMove(qnn_action_t *action,
	const qnn_snapshot_t *snapshot, float emit_dt)
{
	vec3_t rel_vel;
	vec3_t pos_delta_vel;
	int medium;
	int i;

	if (!qnn_runtime.has_emit_anchor)
		return;

	/* Medium at start of window (consistent with emit anchor). */
	if (qnn_runtime.emit_waterlevel >= 2)
		medium = QNN_MEDIUM_WATER;
	else if (qnn_runtime.emit_grounded)
		medium = QNN_MEDIUM_GROUND;
	else
		medium = QNN_MEDIUM_AIR;

	/* fb / lr: view-relative position-delta sign over the emit window.
	 * Position-delta (origin_end - origin_start) / dt is what real MVD
	 * playback can deliver; cl.simvel is not populated on the MVD path. */
	for (i = 0; i < 3; i++)
	{
		pos_delta_vel[i] = (emit_dt > 0.0f)
			? (snapshot->player_origin[i] - qnn_runtime.emit_origin[i]) / emit_dt
			: 0.0f;
	}
	QNN_RelativeFrame(qnn_runtime.emit_view_angles,
		pos_delta_vel, rel_vel);

	if (medium != QNN_MEDIUM_WATER)
	{
		/* Threshold: 20 u/s = eps 0.01 at the 2000-u/s velocity scale.
		 * Air-strafe acceleration is ~3 u/s per native tick; after one
		 * emit window (~5 native ticks) the signal accumulates to ~15 u/s,
		 * just below threshold.  The Python relabeler applies a +2-frame
		 * lookahead to improve strafe accuracy beyond this C-side baseline. */
		static const float eps = 20.0f;
		float sfb = 0.0f, slr = 0.0f;
		if (rel_vel[0] > eps)       sfb =  1.0f;
		else if (rel_vel[0] < -eps) sfb = -1.0f;
		if (rel_vel[1] > eps)       slr =  1.0f;
		else if (rel_vel[1] < -eps) slr = -1.0f;
		if (sfb != 0.0f && slr != 0.0f)
		{
			action->move[0] = sfb * 0.70710678f;
			action->move[1] = slr * 0.70710678f;
		}
		else
		{
			action->move[0] = sfb;
			action->move[1] = slr;
		}
	}
	else
	{
		/* Water: position-delta for fb/lr (same as original). */
		vec3_t delta, rel_delta;
		int i;
		float raw[3];
		for (i = 0; i < 3; i++)
			delta[i] = (emit_dt > 0.0f)
				? (snapshot->player_origin[i] - qnn_runtime.emit_origin[i]) / emit_dt
				: 0.0f;
		QNN_RelativeFrame(qnn_runtime.emit_view_angles, delta, rel_delta);
		raw[0] = rel_delta[0] / QNN_SV_MAXSPEED;
		raw[1] = rel_delta[1] / QNN_SV_MAXSPEED;
		raw[2] = rel_delta[2] / QNN_SV_MAXSPEED;
		{
			float snapped[3];
			QNN_SnapMove(raw, QNN_MEDIUM_WATER,
				QNN_SnapshotHasSelfJumpSound(snapshot), snapped);
			action->move[0] = snapped[0];
			action->move[1] = snapped[1];
			action->move[2] = snapped[2];
		}
		return;  /* ud already filled above */
	}

	/* ud: set by QNN_MvdBackShiftWriteJumpEvents into the back-shifted
	 * ring slot; 0 here at emit time. */
	action->move[2] = 0.0f;
	/* Water ud already handled above via position delta (early return). */
}
