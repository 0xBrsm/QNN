/*
 * qnn_mvd_collect.h — MVD reconstruction path for the QW demo worker.
 *
 * Used when no usercmd_t is available (real MVD demos OR `force_mvd_emit`
 * on QWD).  Three discrete-cmd inference paths plus one move path:
 *
 *   ATTACK sound (weapon-fire PHS multicast) → walkback by full ping
 *          → attack=weapon impulse 1..8 → co-temporal dedup. One event;
 *          no hold tail.
 *   JUMP   sound (player/plyrjmp8.wav) → walkback by full ping →
 *          move[2]=jump_speed → grounded-count chain gate.
 *          One operative press per event; no hold tail.
 *   MOVE   per-emit fb/lr from view-relative position-delta sign;
 *          back-shifted into the ring by QNN_MvdBackShiftWriteMoveXY
 *
 * All MVD-private state (back-shift ring, per-weapon dedup tables)
 * lives as module-private static inside qnn_mvd_collect.c.  Callers
 * reset it at demo start via QNN_MvdCollectReset().
 */

#ifndef QNN_MVD_COLLECT_H
#define QNN_MVD_COLLECT_H

#include "qnn.h"
#include "qnn_collect_helpers.h"

/* ── Module reset (per demo) ──────────────────────────────────────── */

/* Reset the back-shift ring and per-weapon attack/jump dedup tables.
 * The demo_path_seed argument is retained for call-site compatibility
 * but is no longer used (the hold-sim RNG it seeded was removed). */
void QNN_MvdCollectReset(uintptr_t demo_path_seed);

/* ── Action inference ─────────────────────────────────────────────── */

/* Per-native-frame attack detection — emits attack=1 only on the native frame
 * where a self weapon-fire sound is detected.  Updates the module's
 * native_attack_this_window latch (mirrored in qnn_runtime for NQ compatibility). */
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

/* ── MVD sound/move back-shift writers ────────────────────────────────
 *
 * The generic back-shift ring (push/flush/slot-at/rewrite/accessors) now
 * lives in qnn_collect_helpers.h.  These remain MVD-specific: they walk
 * weapon-fire / jump sound events back to the press frame and stamp the
 * resolved slot via the shared ring. */

/* Per-event attack/jump back-shift driven by sound native_time. */
void QNN_MvdBackShiftWriteAttackEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time);
void QNN_MvdBackShiftWriteJumpEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time);

/* Copy the current emit's move XY to the back-shifted slot. */
void QNN_MvdBackShiftWriteMoveXY(uint8_t move, int shift_frames);

/* Reset the per-weapon attack dedup state for `weapon_id` (1..8).  Called
 * when the held weapon changes so a later same-weapon shot isn't
 * false-linked through a different-weapon interval. */
void QNN_MvdResetAttackChain(int weapon_id);

#endif /* QNN_MVD_COLLECT_H */
