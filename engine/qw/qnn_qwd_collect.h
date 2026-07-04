/*
 * qnn_qwd_collect.h — QWD usercmd-truth path for the QW demo worker.
 *
 * Used when a usercmd_t is available (QWD client-side demos) and
 * `force_mvd_emit` is false.  Reads dem_cmd messages deposited into
 * cl.frames[] each Host_Frame and aggregates them into a single
 * emit-rate action label.
 *
 * Pure cmd-window decode — no inference, no back-shift, no ring
 * rewriting.  action.weapon is written directly at the press tick from
 * the decoded impulse target, with carry-forward on non-press ticks and
 * a stat-transition check to pick up engine-forced switches (pickup
 * auto-switch via weapon_touch, respawn defaults, etc.).  All inference
 * machinery (back-shift ring, chain-fill, log-normal hold) lives behind
 * mvd_path; force_mvd_emit is the only legal way to run a QWD file
 * through inference.
 */

#ifndef QNN_QWD_COLLECT_H
#define QNN_QWD_COLLECT_H

#include "qnn.h"

/* Reset module-private state (held-weapon tracker + prev-stat).
 * Called at demo start alongside the labeler/MVD module resets. */
void QNN_QwdCollectReset(void);

/* Walk the current dem_cmd window into a single emit-rate action label.
 * Pure cmd-byte decode — does not touch action->weapon or any module
 * state.  Returns the resolved impulse_target weapon (1..8) or 0 if no
 * weapon-select impulse appeared in the window.
 *
 *   attack / jump (move[2] set to QNN_SV_JUMP_SPEED/QNN_SV_MAXSPEED):
 *     OR across all cmds in the window (any press counts).
 *   forward / side move:
 *     Averaged across the window — each cmd is integrated over
 *     native_dt by the recorder's kbutton_t, so averaging N approximates
 *     a single 50ms-integrated cmd.
 *   upmove (swim down):
 *     Most-negative value across the window.
 *
 * action->weapon is left untouched by this function, but the per-cmd loop
 * advances the QC weapon-select predicate (once per emit tick); the label
 * is then read from QNN_ProgsGetSelfWeapon in QNN_QwdBuildActionLabel.
 */
void QNN_QwdExtractAction(qnn_action_t *action, const qnn_snapshot_t *snapshot);

/* Emit-time QWD action: usercmd extraction + canonical action.weapon label
 * (held weapon + ping-gated pending-impulse lead) + look/switch fill. */
void QNN_QwdBuildActionLabel(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

/* Ping-gated weapon-lead clear.  Call once per emit tick on the genuine QWD
 * usercmd path, AFTER QwdBuildActionLabel and BEFORE the back-shift ring push:
 * clears a lead the engine never confirms within its realization window
 * (ping×2 + the remaining attack cooldown the switch must wait out) — a
 * stale-impulse phantom — by walking the shared ring back over the lead window
 * and resetting those slots + this frame to held.  ping_frames =
 * QNN_PressBackShiftFrames(player, emit_hz); emit_hz converts cooldown to
 * frames. */
void QNN_QwdWeaponLeadStep(qnn_action_t *action,
	const qnn_snapshot_t *snapshot, int tick, int ping_frames, int emit_hz);

/* Per-cmd attack-predicate eval + cmd-block aggregation across the
 * current QWD cmd window.  Calls QNN_ProgsEvalAttack once per cmd
 * (advancing the QC attack_finished state at cmd granularity) and
 * aggregates raw usercmd bytes: fmove/smove as mean, umove as
 * jump-canonical-or-most-negative, buttons OR'd, impulse as the
 * last non-zero byte in the window (matches sv_user.c:3575 overwrite
 * semantics).  Jump operativeness now lives in QNN_QwdEvalPmoveJump
 * (pmove-driven, no QC predicate involvement). */
void QNN_QwdEvalOperativePerCmd(
	const qnn_snapshot_t *snapshot,
	int *out_op_attack,
	int *out_fmove,
	int *out_smove,
	int *out_umove,
	int *out_buttons,
	int *out_impulse);

/* Pmove-driven jump operativeness.  Per-cmd PlayerMove() invocation
 * with pmove globals seeded from the snapshot; returns 1 iff any cmd
 * in this tick's window triggered the patched JumpButton() success
 * branch.  See qnn_pmove_hooks.h for the flag and save/restore.
 *
 * synth_button2:
 *   0 = honour each cmd's actual button2 / upmove>0 — the labeler /
 *       op-jump label path uses this.
 *   1 = pure feasibility mode: force button2=1 on every cmd, and save
 *       / restore all persistent pmove carry state (oldbuttons,
 *       prev_simorg/vel) so the synthetic press doesn't contaminate
 *       the next real-jump tick.  Used by QwdPackInputMask to compute
 *       input_mask bit 7 ("would the engine jump if I pressed?"). */
int QNN_QwdEvalPmoveJump(const qnn_snapshot_t *snapshot, int synth_button2);

/* Fill action->input_mask from press bits (read off action->move,
 * already filled by QwdExtractAction) + per-axis op predicate results.
 * Must run exactly once per tick — invoked from QwdBuildActionLabel
 * right after FillLookAndSwitch.  Reads qwd_state.last_op_attack /
 * last_jump_press_any / last_upmove_pos_any stashed by the same-tick
 * QwdExtractAction call.  See QNN_PackInputMask in qnn_collect_helpers.h
 * for the bit layout. */
void QNN_QwdPackInputMask(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

#endif /* QNN_QWD_COLLECT_H */
