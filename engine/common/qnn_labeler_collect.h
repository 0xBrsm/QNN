/*
 * qnn_labeler_collect.h — Labeler-mode LOBS emit path.
 *
 * Used when `qnn_runtime.labeler_mode` is set.  Bypasses the QOBS
 * pipeline (no jitter filter, no back-shift ring, no fire-hold) and
 * writes a slim LOBS frame per native tick instead.  Target labels come
 * from the action_label filled in main (usercmd truth on QWD, MVD-rule
 * inference on a real MVD).
 *
 * Per-axis engine-effectiveness bits in target_valid_mask mirror
 * src/demo/sanitize.py rules:
 *   bit0 = fb     (alive)
 *   bit1 = lr     (alive)
 *   bit2 = ud     (alive; pressed up needs grounded or water,
 *                   pressed down needs water)
 *   bit3 = fire   (alive; if pressed, the global attack_finished
 *                   cooldown must have elapsed)
 *   bit4 = weapon (alive)
 * Bits 5..7 reserved.
 *
 * Fire cooldown is single-counter (NOT per-weapon): vanilla Quake stores
 * attack_finished on the player edict, not per-weapon, so a press
 * carries cooldown across switches.
 */

#ifndef QNN_LABELER_COLLECT_H
#define QNN_LABELER_COLLECT_H

#include "qnn.h"

/* Reset module-private cooldown state.  Called at demo start. */
void QNN_LabelerCollectReset(void);

/* Handle one native tick in labeler mode.  Computes c_rule fire/jump,
 * packs the move target, fills target_valid_mask, and emits one LOBS
 * frame to `out`.  Also calls QNN_SavePrev so the per-tick state
 * (origin/velocity/grounded) advances under labeler mode the same way
 * it does under the QOBS path. */
void QNN_LabelerHandleTick(const qnn_snapshot_t *snapshot, FILE *out);

/* QNN_EmitLabelerTick — declared in qnn.h alongside the LOBS wire-format
 * documentation; defined in this module. */

#endif /* QNN_LABELER_COLLECT_H */
