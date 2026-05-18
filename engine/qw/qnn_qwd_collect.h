/*
 * qnn_qwd_collect.h — QWD usercmd-truth path for the QW demo worker.
 *
 * Used when a usercmd_t is available (QWD client-side demos) and
 * `force_mvd_emit` is false.  Reads dem_cmd messages deposited into
 * cl.frames[] each Host_Frame and aggregates them into a single
 * emit-rate action label.
 *
 * Pure cmd-window action extraction — no inference, no back-shift.
 * The back-shift ring's impulse walk-back consumes the returned
 * impulse-target weapon to anchor server-observed weapon transitions
 * to the cmd press that caused them.
 */

#ifndef QNN_QWD_COLLECT_H
#define QNN_QWD_COLLECT_H

#include "qnn.h"

/* Walk the current dem_cmd window into a single emit-rate action label.
 *
 *   fire / jump (move[2] set to QNN_SV_JUMP_SPEED/QNN_SV_MAXSPEED):
 *     OR across all cmds in the window (any press counts).
 *   forward / side move:
 *     Averaged across the window — each cmd is integrated over
 *     native_dt by the recorder's kbutton_t, so averaging N approximates
 *     a single 50ms-integrated cmd.
 *   upmove (swim down):
 *     Most-negative value across the window.
 *   weapon:
 *     Left at 0 — the caller's QNN_FillLookAndSwitch fills it with
 *     snapshot->weapon_id.  The returned impulse-target weapon is
 *     sidecar data for the back-shift ring.
 *
 * Returns the cmd-window impulse target weapon (1..8) or 0 if no
 * weapon-select impulse appeared in the window.  Resolves impulses
 * 1-8 (direct select, gated on inventory ownership) and 10/12
 * (next/prev weapon cycle via QNN_NextWeaponId).
 */
int QNN_QwdExtractAction(qnn_action_t *action);

/* Emit-time QWD action: usercmd extraction + look/switch fill.
 * Returns the same impulse-target weapon sidecar as QNN_QwdExtractAction. */
int QNN_QwdInferEmitAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

#endif /* QNN_QWD_COLLECT_H */
