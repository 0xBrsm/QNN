/*
 * qnn_mvd_collect.c — MVD reconstruction path for the QW demo worker.
 *
 * Used when no usercmd_t is available: real MVD demos (cls.mvdplayback)
 * or `force_mvd_emit` on QWD.  Contents:
 *
 *   - Truncated log-normal hold-tail PRNG (fit on QWD-truth corpus)
 *   - Back-shift ring (deferred K-tick emit so server-observed state
 *     changes can rewrite intent labels at the press frame)
 *   - Chain-fill (held-trigger and held-bunnyhop runs collapse to a
 *     continuous label instead of isolated events)
 *   - Per-event sound back-shift (ping + tick-quantization corrected)
 *   - MVD action inference (fire from sound/ammo cues, look/switch from
 *     view-relative state, move from view-relative position-delta sign)
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
#include "qnn_io.h"
#include "qnn_store.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Module-private state ──────────────────────────────────────────── */

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

/* ── Truncated log-normal hold sampler ─────────────────────────────── */

static uint32_t qnn_xorshift32(uint32_t *state)
{
	*state ^= *state << 13;
	*state ^= *state >> 17;
	*state ^= *state << 5;
	return *state;
}

typedef struct { int cd; int n; const uint16_t *cdf; } qnn_hold_cdf_t;

static const uint16_t qnn_hold_cdf_axe[9] = {
	/* Axe (wid=3, mu=0.5992, sigma=0.4241, cd=10) */
	21177, 50622, 61493, 64460, 65238, 65450, 65511, 65529, 65535
};
static const uint16_t qnn_hold_cdf_sg[9] = {
	/* SG (wid=4, mu=0.9943, sigma=0.6484, cd=10) */
	11988, 30261, 43975, 52713, 58074, 61363, 63407, 64701, 65535
};
static const uint16_t qnn_hold_cdf_ssg[13] = {
	/* SSG (wid=5, mu=1.1702, sigma=0.6640, cd=14) */
	8156, 23263, 36503, 46045, 52528, 56869, 59784, 61760, 63115, 64057,
	64721, 65194, 65535
};
static const uint16_t qnn_hold_cdf_gl[11] = {
	/* GL (wid=8, mu=1.0066, sigma=0.6201, cd=12) */
	10834, 29162, 43257, 52190, 57577, 60808, 62767, 63975, 64734, 65219,
	65535
};
static const uint16_t qnn_hold_cdf_rl[15] = {
	/* RL (wid=9, mu=1.0557, sigma=0.5880, cd=16) */
	8743, 26625, 41419, 51014, 56804, 60241, 62290, 63529, 64290, 64766,
	65068, 65263, 65392, 65477, 65535
};

/* Jump hold CDF.  Source: ud=2 runs from QWD truth, split at every
 * additional ground→air transition within the run (each transition is
 * a new engine jump).  Log-normal fit (μ=1.0008, σ=0.6395) truncated to
 * [1, 19] frames at 20 Hz. */
static const uint16_t qnn_hold_cdf_jump[19] = {
	11323, 29206, 42761, 51399, 56679, 59899, 61889, 63139, 63939, 64461,
	64807, 65040, 65200, 65311, 65389, 65444, 65484, 65514, 65535
};

static const qnn_hold_cdf_t *QNN_HoldCDF(int weapon_id)
{
	static const qnn_hold_cdf_t descs[8] = {
		{10,  9, qnn_hold_cdf_axe},  /* wid 3 */
		{10,  9, qnn_hold_cdf_sg},   /* wid 4 */
		{14, 13, qnn_hold_cdf_ssg},  /* wid 5 */
		{ 0,  0, NULL},              /* wid 6 NG  — continuous */
		{ 0,  0, NULL},              /* wid 7 SNG — continuous */
		{12, 11, qnn_hold_cdf_gl},   /* wid 8 */
		{16, 15, qnn_hold_cdf_rl},   /* wid 9 */
		{ 0,  0, NULL},              /* wid 10 LG — continuous */
	};
	if (weapon_id < 3 || weapon_id > 10)
		return NULL;
	return (descs[weapon_id - 3].cdf != NULL) ? &descs[weapon_id - 3] : NULL;
}

static int QNN_SampleFireHold(int weapon_id, uint32_t *rng)
{
	const qnn_hold_cdf_t *d = QNN_HoldCDF(weapon_id);
	uint32_t r;
	int i;

	if (!d)
		return 1;
	r = qnn_xorshift32(rng) & 0xFFFF;
	for (i = 0; i < d->n; i++)
		if (r <= (uint32_t)d->cdf[i])
			return i + 1;
	return d->cd - 1;
}

/* Engine attack cooldown in emit frames at 20 Hz, per QC weapons.qc
 * attack_finished delays.  Single source of truth for both the tap-
 * weapon hold-CDF range and the chain-fill linking window. */
static int QNN_FireCooldownEmit(int weapon_id)
{
	switch (weapon_id)
	{
	case 3:  return 10;  /* Axe — 0.5 s */
	case 4:  return 10;  /* SG  — 0.5 s */
	case 5:  return 14;  /* SSG — 0.7 s */
	case 6:  return  4;  /* NG  — 0.2 s */
	case 7:  return  4;  /* SNG — 0.2 s */
	case 8:  return 12;  /* GL  — 0.6 s */
	case 9:  return 16;  /* RL  — 0.8 s */
	case 10: return  2;  /* LG  — 0.1 s */
	default: return  0;
	}
}

/* Auto-refire weapons whose QC W_Attack re-enters every cooldown while
 * the trigger is held.  Continuous → fixed near-cd tail.  Tap → sampled
 * log-normal. */
static qboolean QNN_FireIsContinuous(int weapon_id)
{
	return (weapon_id == 6 || weapon_id == 7 || weapon_id == 10);
}

static int QNN_FireHoldFrames(int weapon_id, uint32_t *rng)
{
	int cd = QNN_FireCooldownEmit(weapon_id);
	int ext;

	if (QNN_FireIsContinuous(weapon_id))
		ext = cd;
	else if (QNN_HoldCDF(weapon_id))
		ext = QNN_SampleFireHold(weapon_id, rng);
	else
		return 0;
	if (cd > 1 && ext > cd - 1)
		ext = cd - 1;
	return ext;
}

static int QNN_SampleJumpHold(uint32_t *rng)
{
	uint32_t r;
	int i;
	int n = (int)(sizeof(qnn_hold_cdf_jump) / sizeof(qnn_hold_cdf_jump[0]));

	r = qnn_xorshift32(rng) & 0xFFFF;
	for (i = 0; i < n; i++)
		if (r <= (uint32_t)qnn_hold_cdf_jump[i])
			return i + 1;
	return n;
}

static int QNN_JumpHoldFrames(uint32_t *rng)
{
	return QNN_SampleJumpHold(rng);
}

/* ── Back-shift ring internals ─────────────────────────────────────── */

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

static int QNN_PingSecToEmitFrames(float ping_sec)
{
	if (qnn_runtime.fixed_tick_hz <= 0)
		return 0;
	return (int)(ping_sec * (float)qnn_runtime.fixed_tick_hz + 0.5f);
}

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

/* ── Unified press-event pipeline ──────────────────────────────────── */

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

/* Fire-specific callbacks. */

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

/* Jump-specific callbacks. */

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

/* ── Ring slot drain (forwarder to helpers.c emit pipeline) ─────────── */

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

/* ── Public API ───────────────────────────────────────────────────── */

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

int QNN_MvdBackShiftImpulseWalkback(int weapon_id, int cap)
{
	qnn_backshift_ring_t *ring = &mvd_state.backshift;
	int latest;
	int i;

	if (ring->count == 0 || cap <= 0)
		return 0;
	if (cap > ring->count)
		cap = ring->count;
	latest = (ring->head + QNN_BACKSHIFT_K - 1) % QNN_BACKSHIFT_K;
	for (i = 0; i < cap; i++)
	{
		int idx = (latest + QNN_BACKSHIFT_K - i) % QNN_BACKSHIFT_K;
		if (ring->slots[idx].impulse_target_weapon == weapon_id)
			return i + 1;
	}
	return 0;
}

void QNN_MvdBackShiftPush(qnn_tick_emit_state_t *emit, FILE *out,
	const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, qboolean grounded,
	int weapon_id, int impulse_target_weapon, int stat_items)
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
	slot->impulse_target_weapon = impulse_target_weapon;

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

/* ── MVD action inference ─────────────────────────────────────────── */

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
