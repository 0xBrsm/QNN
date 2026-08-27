/*
 * qnn_mvd_collect.c — MVD reconstruction path for the QW demo worker.
 *
 * Used when no usercmd_t is available: real MVD demos
 * (cls.mvdplayback) or `force_mvd_emit` on QWD.  Three discrete-cmd
 * inference paths and one move path live here, all wired through a
 * single back-shift ring that lets server-observed sound/state events
 * write labels back at the press frame instead of the broadcast frame.
 *
 *   ATTACK sound (weapon-fire PHS multicast) → walkback by full ping →
 *          OR the attack bit into slot.action.move AND stamp
 *          slot.action.attack from the SOUND's own class (byte truth) →
 *          co-temporal dedup. The categorical class and event timing are
 *          stamped onto one slot. One operative event; no hold tail (held
 *          frames are masked in training).
 *
 *   JUMP   sound (player/plyrjmp8.wav) → walkback by full ping → OR
 *          the jump bit (bit 7) into slot.action.move → grounded-
 *          count chain gate.  One operative press per event; no hold
 *          tail.  Bit 6 (ud-pos) is reserved for actual swim-up /
 *          jumppad upmove and is NOT set from jump-sound inference.
 *
 *   MOVE   per-emit fb/lr from view-relative position-delta sign
 *          (`QNN_MvdInferEmitMove`).  fb/lr back-shifted into the ring
 *          by `QNN_MvdBackShiftWriteMoveXY`; ud is set via the JUMP
 *          path above (water gets ud from snap-time position delta).
 *
 * Attack/jump labels are sparse: one operative press per sound event, no
 * forward hold tail.  Under input_mask=true training the held-button
 * frames are non-operative and masked from every target, so they are
 * not synthesized (the former log-normal hold samplers were removed).
 *
 * Module-private state lives in the file-scope `mvd_state` struct.
 * `QNN_MvdCollectReset` is the per-demo reset entry point invoked by
 * the QW main loop.
 *
 * Cross-module touch points:
 *   - `qnn_runtime.tick` / `fixed_tick_hz` (timing scale) — read-only.
 *   - `qnn_runtime.native_attack_this_window` — mirrored for NQ's MVD path
 *     which shares the field (NQ has its own native-attack emit logic in
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


/* ══════════════════════════════════════════════════════════════════════
 *                       MODULE-PRIVATE STATE
 * ══════════════════════════════════════════════════════════════════════ */

/* Continuous-weapon trigger-pull collapse window.  A held nailgun/lightning
 * think-chain emits one projectile sound every ~0.1s, but the operative label
 * is the W_Attack TRIGGER PULL (~0.2s cadence), not each nail.  Same-weapon
 * attack sounds closer than this collapse into one operative press — matching
 * BOTH the QWD op-attack bit (QNN_ProgsEvalAttack's attack_finished cooldown
 * gate) and the validator reference (qnn.bc.demo_truth.trigger_events), so the
 * MVD and QWD paths share one trigger-pull cadence after the press is
 * identified.  MUST stay equal to qnn.bc.demo_truth.MERGE_GAP. */
#define QNN_ATTACK_TRIGGER_MERGE_SEC 0.15f

/* Lightning-gun (Thunderbolt) op-attack reconstruction.  UNLIKE every other
 * weapon, LG emits NO per-shot sound: W_FireLightning plays `lhit` throttled to
 * a fixed 0.6s HEARTBEAT (self.t_width = time + 0.6) plus `lstart` once per
 * discharge onset — while LG actually re-fires every 0.2s while held (the
 * player_light think-chain; QNN_WEAPON_LIGHTNING cd = 0.2f, == nailguns).  So
 * the 0.2s op-attacks BETWEEN heartbeats are real operative bolts with no sound,
 * and the sound-driven emitter lands only ~0.5x the true LG op-attack count
 * (the QWD path gets it right via the progs op-attack bit).  Reconstruct them:
 * when two LG attack sounds fall within one burst gap (=> firing was continuous,
 * since the heartbeat only plays WHILE firing), fill op-attacks at the 0.2s
 * op-cadence across the held-LG slots between them.  These are OPERATIVE (not
 * masked no-op holds), so the fill sets input_mask bit 0 too.  Mirrors
 * qnn.bc.demo_truth.lg_op_attack_count (LG_OP_CADENCE 0.2, burst_gap 0.7) —
 * KEEP THESE EQUAL.  Validate MVD/QWD LG -> ~1.0 (NOT the throttled sound-ref). */
#define QNN_LG_OP_CADENCE_SEC 0.2f
#define QNN_LG_BURST_GAP_SEC  0.7f

typedef struct
{
	/* Per-weapon attack dedup state.  Index = action weapon id 1..8;
	 * slot 0 unused. */
	int      last_attack_shot_tick[11];
	int      last_attack_source_tick[11];
	/* Per-weapon native_time (demo seconds) of the last KEPT attack trigger,
	 * for the QNN_ATTACK_TRIGGER_MERGE_SEC think-chain collapse.
	 * -1 = none kept yet this demo. */
	float    last_attack_kept_native[11];
	/* Jump chain-fill / dedup state — single global counter. */
	int      last_jump_shot_tick;
	int      last_jump_source_tick;

} qnn_mvd_state_t;

static qnn_mvd_state_t mvd_state;

/* ══════════════════════════════════════════════════════════════════════
 *          MVD SOUND/MOVE BACK-SHIFT WRITERS (drive the shared ring)
 * ══════════════════════════════════════════════════════════════════════
 *
 * The generic ring (instance, push/flush/slot-at/rewrite/accessors) lives
 * in qnn_collect_helpers.c.  This module reaches it via QNN_BackShiftRing()
 * and the public QNN_BackShiftSlotAt, and adds the MVD-specific sound
 * walk-back: a server-observed weapon-fire / jump sound is mapped to its
 * press frame by full-ping offset and stamps the resolved ring slot. */

static float QNN_BackShiftSlotSeconds(void)
{
	if (qnn_runtime.fixed_tick_hz > 0)
		return 1.0f / (float)qnn_runtime.fixed_tick_hz;
	if (qnn_runtime.fixed_dt > 0.0001f)
		return qnn_runtime.fixed_dt;
	return 1.0f / 77.0f;
}

/* Convert a press-time offset (seconds, negative = past) to a ring-
 * slot index back from the latest push.  Returns -1 if the press
 * lands beyond the ring's history.  Slot width follows the active emit
 * cadence: fixed-rate collects use 1/tick_hz; native-rate collects use
 * the current demo frame dt. */
static int QNN_BackShiftOffsetFromPress(float press_offset_sec, int count,
	float slot_seconds)
{
	int offset;

	if (press_offset_sec >= 0.0f)
		return 0;
	if (slot_seconds <= 0.0001f)
		slot_seconds = 1.0f / 77.0f;
	offset = (int)ceilf(-press_offset_sec / slot_seconds);
	if (offset >= count)
		return -1;
	return offset;
}

static qnn_backshift_slot_t *QNN_BackShiftSlotForPress(
	qnn_backshift_ring_t *ring, float press_offset_sec, int *offset_out)
{
	int offset;
	qnn_backshift_slot_t *slot;

	offset = QNN_BackShiftOffsetFromPress(press_offset_sec, ring->count,
		QNN_BackShiftSlotSeconds());
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


/* ── Generic press-event pipeline (dedup only) ── */
/*
 * Shared between the ATTACK and JUMP paths.  Each path supplies the action
 * mutation and its own dedup-tick state.
 *
 * The label is sparse: exactly one operative press per sound event.  No
 * forward hold tail and no gap-bridging chain-fill are synthesized —
 * both only ever write the attack/jump bit onto within-cooldown frames,
 * which are non-operative and masked from every training target under
 * input_mask=true.  (The self player is always in its own PHS, so its
 * own attack/jump sounds are never dropped; there is nothing to recover.)
 */

typedef void (*qnn_press_set_action_fn)(qnn_action_t *action);

typedef struct
{
	qnn_press_set_action_fn   set_action;
	int                      *last_dest_tick;
	int                      *last_source_tick;
} qnn_press_event_cfg_t;

static qboolean QNN_BackShiftApplyPressEvent(
	qnn_backshift_slot_t *slot,
	const qnn_press_event_cfg_t *cfg)
{
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

	*cfg->last_dest_tick = slot->tick;
	*cfg->last_source_tick = qnn_runtime.tick;

	return true;
}


/* ── Sound-event iterator (shared by ATTACK and JUMP path apply fns) ── */

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
 *                          ATTACK PATH
 * ══════════════════════════════════════════════════════════════════════ */
/*
 * Sound (weapon-fire PHS multicast) → back-shift by full ping →
 * attack=1 in target ring slot → dedup against co-temporal duplicates
 * (LG lstart+lhit).  One operative press per event; no chain-fill,
 * no hold tail (both only ever wrote masked within-cooldown frames).
 */

static void qnn_press_set_attack(qnn_action_t *a)
{
	a->move |= 0x01;       /* bit 0 = attack press */
	/* Assert attack FEASIBILITY on the same slot, from sound truth.  The
	 * press is back-shifted by ping onto a slot whose input_mask was packed
	 * `ping` frames out of phase with the cooldown clock, so its
	 * attack-feasibility bit (input_mask bit 0) is often 0 exactly where
	 * genuine presses land — zeroing the operative label (move&1 &
	 * input_mask&1) for ~1/4-1/3 of shots (the -30% operative-attack gap;
	 * realigning the cooldown stamp via e597bca2 couldn't fix it — the slot's
	 * mask was already frozen at push).  A self weapon-attack sound proves the
	 * weapon was off-cooldown at the true press frame, so feasibility is 1
	 * there by construction.  Phase-locked with the attack bit, mirroring the
	 * sound-truth attack class (mvd-attack-audit.md §Round 5). */
	a->input_mask |= 0x01; /* bit 0 = attack feasible */
}


static void QNN_BackShiftWriteAttackAtTime(qnn_backshift_ring_t *ring,
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

	/* Universal-shift attribution: the firing weapon comes from the
	 * SOUND's own class (the demo's byte truth, 100% correct), NOT from
	 * the held-weapon snapshot at this emit frame. The categorical attack
	 * class is stamped onto the SAME slot the sound back-shifts to.
	 *
	 * Fall back to the held-weapon snapshot only if the sound somehow
	 * doesn't classify (should not happen: the predicate already gated on
	 * QNN_IsSelfWeaponAttackSound). */
	weapon_id = QNN_WeaponIdFromAttackSound(snd);
	if (!QNN_WeaponIsValid(weapon_id))
		weapon_id = snapshot->weapon_id;
	if (!QNN_WeaponIsValid(weapon_id))
		return;

	/* Continuous-weapon trigger-pull collapse (DRY with the validator
	 * reference trigger_events and the QWD cooldown gate): a same-weapon attack
	 * sound within QNN_ATTACK_TRIGGER_MERGE_SEC of the last KEPT trigger is a
	 * think-chain continuation (NG/SNG fire every ~0.1s while held), not a
	 * distinct operative press — drop it so the nailguns match the QWD ~0.2s
	 * trigger cadence instead of one press per nail.  Single-shot weapons
	 * fire >=0.5s apart, so this never collapses them.  Keyed on the sound's
	 * native_time — the same demo-second timeline trigger_events collapses —
	 * independent of where the press back-shifts. */
	if (mvd_state.last_attack_kept_native[weapon_id] >= 0.0f
		&& snd->native_time - mvd_state.last_attack_kept_native[weapon_id]
			< QNN_ATTACK_TRIGGER_MERGE_SEC)
		return;

	cfg.set_action      = qnn_press_set_attack;
	cfg.last_dest_tick  = &mvd_state.last_attack_shot_tick[weapon_id];
	cfg.last_source_tick = &mvd_state.last_attack_source_tick[weapon_id];

	if (!QNN_BackShiftApplyPressEvent(slot, &cfg))
		return;

	/* Lightning-gun op-attack reconstruction (see QNN_LG_OP_CADENCE_SEC).  The
	 * lhit heartbeat + lstart are ~0.5x LG's true 0.2s op-attack rate; recover the
	 * silent between-heartbeat bolts.  A gap within QNN_LG_BURST_GAP_SEC of the
	 * previous LG attack means firing was continuous (the heartbeat only plays
	 * while firing), so fill op-attacks at the 0.2s op-cadence across the
	 * held-LG slots between the two sounds.  Walk from the current press slot
	 * (`offset` frames back) toward older slots; stop at the ring edge or the
	 * first non-LG slot (beam ended / weapon switched) so a non-firing LG hold
	 * or a prior beam is never back-filled. */
	if (weapon_id == QNN_WEAPON_LIGHTNING
		&& mvd_state.last_attack_kept_native[weapon_id] >= 0.0f)
	{
		float gap = snd->native_time
			- mvd_state.last_attack_kept_native[weapon_id];
		if (gap > QNN_ATTACK_TRIGGER_MERGE_SEC && gap <= QNN_LG_BURST_GAP_SEC)
		{
			int hz = (slot->tick_hz > 0) ? slot->tick_hz : 20;
			int step = (int)(QNN_LG_OP_CADENCE_SEC * hz + 0.5f);
			int nfill = (int)(gap / QNN_LG_OP_CADENCE_SEC + 0.5f) - 1;
			int k;
			if (step < 1)
				step = 1;
			for (k = 1; k <= nfill; ++k)
			{
				qnn_backshift_slot_t *fill_slot;
				int back = offset + step * k;
				if (back >= ring->count)
					break;
				if (!QNN_BackShiftSlotAt(ring, back, &fill_slot))
					break;
				qnn_press_set_attack(&fill_slot->action);
				fill_slot->action.attack = (uint8_t)QNN_WEAPON_LIGHTNING;
			}
		}
	}

	/* Only KEPT triggers advance the collapse window — matches
	 * trigger_events' "gap from last counted event". */
	mvd_state.last_attack_kept_native[weapon_id] = snd->native_time;

	/* Attack label: stamp sound-truth weapon on the exact back-shifted
	 * effective-attack frame; all non-attack frames remain impulse 0. */
	slot->action.attack = (uint8_t)weapon_id;

	if (slot->attack_route_count < QNN_MAX_ATTACK_ROUTE_EVENTS)
	{
		qnn_attack_route_event_t *ev =
			&slot->attack_routes[slot->attack_route_count++];
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

static qboolean QNN_BackShiftMatchAttackSound(const qnn_sound_event_t *snd)
{
	return QNN_IsSelfWeaponAttackSound(snd);
}

static void QNN_BackShiftApplyAttackSound(qnn_backshift_ring_t *ring,
	const qnn_snapshot_t *snapshot,
	const qnn_sound_event_t *snd, int sound_index,
	float ping_sec, float emit_start_native_time)
{
	QNN_BackShiftWriteAttackAtTime(ring, snapshot, sound_index,
		snd, ping_sec, emit_start_native_time);
}


/* ══════════════════════════════════════════════════════════════════════
 *                          JUMP PATH
 * ══════════════════════════════════════════════════════════════════════ */
/*
 * Sound (player/plyrjmp8.wav) → back-shift by full ping →
 * move[2]=jump_speed in target ring slot → dedup.  One operative press
 * per event; no chain-fill (bhop is discrete short presses separated by
 * released gaps — bridging merged them into single long runs).
 */

static void qnn_press_set_jump(qnn_action_t *a)
{
	a->move |= 0x80;       /* jump bit only — sound implies button press,
	                        * not raw upmove */
	/* Assert jump FEASIBILITY (input_mask bit 7) on the same slot, from
	 * sound truth — same back-shift phase-skew as attack above, which dropped
	 * the operative jump label even harder (-49%).  A jump sound proves the
	 * jump was feasible (grounded/alive) at the true press frame. */
	a->input_mask |= 0x80; /* bit 7 = jump feasible */
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

	(void)QNN_BackShiftApplyPressEvent(slot, &cfg);
}


/* ══════════════════════════════════════════════════════════════════════
 *                          SWITCH PATH
 * ══════════════════════════════════════════════════════════════════════ */
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

void QNN_MvdBackShiftWriteMoveXY(uint8_t move, int shift_frames)
{
	qnn_backshift_ring_t *ring = QNN_BackShiftRing();
	qnn_backshift_slot_t *slot;
	qnn_backshift_slot_t *latest;
	const uint8_t fb_lr_mask = 0x1E; /* bits 1-4: fb_neg, fb_pos, lr_neg, lr_pos */

	if (ring->count == 0 || shift_frames <= 0)
		return;
	if (!QNN_BackShiftSlotAt(ring, 0, &latest)
		|| !QNN_BackShiftSlotAt(ring, shift_frames, &slot)
		|| slot == latest)
		return;
	/* Only back-shift fb/lr bits — these come from view-relative velocity
	 * sign, which lags the keyboard press by ping+tick like attack and
	 * switch.  ud bits are back-shifted per-event by the jump writer. */
	slot->action.move = (uint8_t)((slot->action.move & ~fb_lr_mask)
		| (move & fb_lr_mask));
}


/* ══════════════════════════════════════════════════════════════════════
 *                              PUBLIC API
 * ══════════════════════════════════════════════════════════════════════ */

void QNN_MvdCollectReset(uintptr_t demo_path_seed)
{
	int w;
	(void)demo_path_seed;  /* hold-sim RNG removed; reset is now pure zero */
	QNN_BackShiftReset();  /* the ring instance lives in qnn_collect_helpers.c */
	memset(&mvd_state, 0, sizeof(mvd_state));
	/* native_time 0 is a valid demo time; sentinel -1 = no kept trigger yet. */
	for (w = 0; w < 11; ++w)
		mvd_state.last_attack_kept_native[w] = -1.0f;
}

void QNN_MvdBackShiftWriteAttackEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time)
{
	QNN_BackShiftForSoundEvents(QNN_BackShiftRing(), snapshot,
		ping_sec, emit_start_native_time,
		QNN_BackShiftMatchAttackSound,
		QNN_BackShiftApplyAttackSound);
}

void QNN_MvdBackShiftWriteJumpEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time)
{
	QNN_BackShiftForSoundEvents(QNN_BackShiftRing(), snapshot,
		ping_sec, emit_start_native_time,
		QNN_BackShiftMatchJumpSound,
		QNN_BackShiftApplyJumpSound);
}

void QNN_MvdResetAttackChain(int weapon_id)
{
	if (QNN_WeaponIsValid(weapon_id))
	{
		mvd_state.last_attack_shot_tick[weapon_id] = 0;
		mvd_state.last_attack_source_tick[weapon_id] = 0;
	}
}


/* ══════════════════════════════════════════════════════════════════════
 *               ACTION INFERENCE (native + emit entry points)
 * ══════════════════════════════════════════════════════════════════════ */

/* Fill action->look from the current snapshot. */
extern void QNN_FillLook(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

void QNN_MvdInferNativeAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);
	if (QNN_SnapshotHasSelfWeaponAttackSound(snapshot))
	{
		action->move |= 0x01; /* bit 0 = attack press */
		qnn_runtime.native_attack_this_window = true;
	}
	/* Jump is handled by QNN_MvdBackShiftWriteJumpEvents at emit time. */
}

void QNN_MvdInferEmitAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	QNN_ClearAction(action);
	QNN_FillLook(action, snapshot);
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
		int fb_neg = (rel_vel[0] < -eps) ? 1 : 0;
		int fb_pos = (rel_vel[0] >  eps) ? 1 : 0;
		int lr_neg = (rel_vel[1] < -eps) ? 1 : 0;
		int lr_pos = (rel_vel[1] >  eps) ? 1 : 0;
		uint8_t fb_lr = QNN_PackInputMask(
			/*alive=*/1, fb_neg, fb_pos, lr_neg, lr_pos,
			0, 0, 0, 0);
		/* Preserve any attack/ud/jump bits the back-shift writers
		 * may have set on this slot. */
		action->move = (uint8_t)((action->move & ~0x1E) | fb_lr);
		return;
	}

	/* Water: position-delta for fb/lr/ud. */
	{
		vec3_t delta, rel_delta;
		int i;
		float raw[3];
		float snapped[3];
		int fb_neg, fb_pos, lr_neg, lr_pos, up_neg, up_pos;
		uint8_t packed;

		for (i = 0; i < 3; i++)
			delta[i] = (emit_dt > 0.0f)
				? (snapshot->player_origin[i] - qnn_runtime.emit_origin[i]) / emit_dt
				: 0.0f;
		QNN_RelativeFrame(qnn_runtime.emit_view_angles, delta, rel_delta);
		raw[0] = rel_delta[0] / QNN_SV_MAXSPEED;
		raw[1] = rel_delta[1] / QNN_SV_MAXSPEED;
		raw[2] = rel_delta[2] / QNN_SV_MAXSPEED;
		QNN_SnapMove(raw, QNN_MEDIUM_WATER,
			QNN_SnapshotHasSelfJumpSound(snapshot), snapped);
		fb_neg = (snapped[0] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		fb_pos = (snapped[0] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		lr_neg = (snapped[1] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		lr_pos = (snapped[1] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		up_neg = (snapped[2] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		up_pos = (snapped[2] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		packed = QNN_PackInputMask(
			/*alive=*/1, fb_neg, fb_pos, lr_neg, lr_pos,
			up_neg, up_pos, 0, 0);
		/* Preserve attack and jump bits set by back-shift writers. */
		action->move = (uint8_t)((action->move & 0x81) | packed);
	}
}
