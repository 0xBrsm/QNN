/*
 * qnn_labeler_collect.h — Labeler-mode LOBS emit path.
 *
 * Used when `qnn_runtime.labeler_mode` is set.  Bypasses the QOBS
 * pipeline (no jitter filter, no back-shift ring, no fire-hold) and
 * writes a slim LOBS frame per native tick instead.  Target labels come
 * from the action_label filled in main (usercmd truth on QWD, MVD-rule
 * inference on a real MVD).
 *
 * LOBS payload pairs two columns per tick:
 *   cmd_*    — aggregated raw usercmd bytes the player sent (angles,
 *              move, buttons, impulse) in QW wire-format precision
 *   op_input — strict per-axis op mask of those bytes, sourced from
 *              QC predicates (qwprogs.dat via qnn_progs.c) for the
 *              non-trivial axes
 *
 * Trainer derives the CE-keep mask per axis as
 * `(no_press_axis) | (op_input_bit_axis)` — no engine-rule logic on
 * the C side, no rule transcription on the Python side.
 *
 *   bit0 = fb       : cmd_move.fb != 0 (pmove integrates fb always)
 *   bit1 = lr       : cmd_move.lr != 0
 *   bit2 = ud       : (button2 OR umove>0) AND QC PlayerJump fired,
 *                     OR (umove<0) AND waterlevel>=2
 *   bit3 = fire     : button0 AND QC W_WeaponFrame fired
 *   bit4 = impulse  : cmd_impulse != 0 AND QC ImpulseCommands flipped
 *                     self.weapon
 *
 * Bits 5..7 reserved.
 */

#ifndef QNN_LABELER_COLLECT_H
#define QNN_LABELER_COLLECT_H

#include "qnn.h"

/* Reset module-private cooldown state.  Called at demo start. */
void QNN_LabelerCollectReset(void);

/* Handle one native tick in labeler mode.  Aggregates the cmd window
 * (raw usercmd bytes), runs the QC predicates to fill op_input, and
 * emits one LOBS frame to `out`.  Also calls QNN_SavePrev so per-tick
 * state (origin/velocity/grounded) advances under labeler mode the
 * same way it does under the QOBS path. */
void QNN_LabelerHandleTick(const qnn_snapshot_t *snapshot, FILE *out);

/* QNN_EmitLabelerTick — declared in qnn.h alongside the LOBS wire-format
 * documentation; defined in this module. */

#endif /* QNN_LABELER_COLLECT_H */
