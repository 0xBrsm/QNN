/*
 * qnn_mvd_collect.h — MVD reconstruction path for the QW demo worker.
 *
 * Used when no usercmd_t is available (real MVD demos OR `force_mvd_emit`
 * on QWD).  Owns the back-shift ring (deferred label emit), the chain-fill
 * + log-normal tail samplers, the per-event sound back-shift, and the MVD
 * action-inference functions (fire from sound/ammo cues, move from
 * view-relative position-delta).
 *
 * All MVD-private state (back-shift ring, fire/jump hold counters,
 * per-weapon dedup tables) lives as module-private static inside
 * qnn_mvd_collect.c.  Callers reset it at demo start via
 * QNN_MvdCollectReset().
 */

#ifndef QNN_MVD_COLLECT_H
#define QNN_MVD_COLLECT_H

#include "qnn.h"
#include "qnn_collect_helpers.h"

/* ── Module reset (per demo) ──────────────────────────────────────── */

/* Reset back-shift ring, per-weapon fire/jump dedup tables, and the
 * log-normal hold counters/RNG seeds.  Caller supplies a seed derived
 * from the demo path so different demos use distinct PRNG streams. */
void QNN_MvdCollectReset(uintptr_t demo_path_seed);

/* ── Action inference ─────────────────────────────────────────────── */

/* Per-native-frame fire detection — emits fire=1 only on the native frame
 * where a shot event is detected (weapon-fire sound OR ammo decrement).
 * Updates the module's native_fire_this_window latch (mirrored in
 * qnn_runtime for NQ compatibility). */
void QNN_MvdInferNativeAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

/* Per-emission look/switch action.  Movement is filled separately by
 * QNN_MvdInferEmitMove. */
void QNN_MvdInferEmitAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

/* Per-emission move label.  fb/lr from view-relative position-delta
 * sign; ud filled by the back-shift jump-event writer (water uses
 * position-delta swim up/down). */
void QNN_MvdInferEmitMove(qnn_action_t *action,
	const qnn_snapshot_t *snapshot, float emit_dt);

/* ── Back-shift ring API ──────────────────────────────────────────── */

/* Accessor: returns true if the ring saw a previous weapon id (i.e.,
 * at least one push has happened).  Sets *prev_weapon out param. */
qboolean QNN_MvdBackShiftPrevWeapon(int *prev_weapon_out);

/* Accessor: returns true if the ring tracked stat_items on the previous
 * push.  Sets *prev_items_out.  Used by the pickup gate to detect
 * IT_ bit 0→1 flips on the same frame as a weapon transition. */
qboolean QNN_MvdBackShiftPrevStatItems(int *prev_items_out);

/* Number of slots currently held in the ring (0..QNN_BACKSHIFT_K). */
int QNN_MvdBackShiftCount(void);

/* Walk the ring back from the latest push (offset 0) up to `cap` slots
 * and return the offset (1-based shift) of the first slot whose recorded
 * impulse_target_weapon matches `weapon_id`.  Returns 0 if no match. */
int QNN_MvdBackShiftImpulseWalkback(int weapon_id, int cap);

/* Push the current emit tick's (pre-packed obs + action + metadata)
 * into the ring.  See qnn_collect_helpers.h's QNN_BackShiftPush for
 * the full contract — this is a thin wrapper that exposes the module-
 * private ring pointer. */
void QNN_MvdBackShiftPush(qnn_tick_emit_state_t *emit, FILE *out,
	const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, qboolean grounded,
	int weapon_id, int impulse_target_weapon, int stat_items);

/* Rewrite the trailing `shift_frames` slots so they carry the new
 * weapon — anchoring intent at the press frame. */
void QNN_MvdBackShiftOnWeaponChange(int new_weapon_id,
	int prev_weapon_id, int shift_frames);

/* Per-event fire/jump back-shift driven by sound native_time. */
void QNN_MvdBackShiftWriteFireEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time);
void QNN_MvdBackShiftWriteJumpEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time);

/* Copy the current emit's move XY to the back-shifted slot. */
void QNN_MvdBackShiftWriteMoveXY(const float move[3], int shift_frames);

/* Drain every remaining slot through `emit`.  Called at demo end. */
void QNN_MvdBackShiftFlushAll(qnn_tick_emit_state_t *emit);

/* Reset the per-weapon fire chain-fill state for `weapon_id` (1..10)
 * and clear any pending hold spillover.  Called when the held weapon
 * changes so a later same-weapon shot isn't false-linked through a
 * different-weapon interval. */
void QNN_MvdResetFireChain(int weapon_id);

#endif /* QNN_MVD_COLLECT_H */
