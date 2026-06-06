/*
 * qnn_hold_samplers.h — Truncated log-normal hold-tail samplers for
 * MVD reconstruction.
 *
 * The MVD inference path emits one fire / jump label tick per
 * server-broadcast sound event.  Real human play streams hold the
 * button down across many cmd-ticks per press (e.g. a SG shot has
 * fire=1 held for ~6-12 frames at 20 Hz emit).  Re-creating that hold
 * pattern in the synthetic label is what gives the trained policy a
 * human-feel firing/jump cadence at deploy time.
 *
 * Per-weapon fire CDFs and a single jump CDF were fit on QWD-truth
 * hold durations and truncated to the engine's attack_finished
 * cooldown — see qnn_hold_samplers.c for the fitted parameters.
 *
 * Not used by the labeler training pipeline.  The labeler trains on
 * the QWD-truth sparse one-tick-per-event signal; these hold samplers
 * only run when MVD inference emits BC training labels.
 */

#ifndef QNN_HOLD_SAMPLERS_H
#define QNN_HOLD_SAMPLERS_H

#include "qnn.h"

#include <stdint.h>

/* Engine attack cooldown in emit frames at 20 Hz, per QC weapons.qc
 * attack_finished delays.  Single source of truth used by the fire
 * chain-fill gate and the hold-CDF range clamp. */
int QNN_FireCooldownEmit(int weapon_id);

/* Forward fire-hold extension in emit frames after a back-shifted
 * sound event.  Continuous-refire weapons (NG/SNG/LG) return their
 * cooldown verbatim; tap weapons sample from a truncated log-normal
 * fit (Axe/SG/SSG/GL/RL).  Returns 0 if no fit is registered. */
int QNN_FireHoldFrames(int weapon_id, uint32_t *rng);

/* Forward jump-hold extension in emit frames after a back-shifted
 * jump sound.  Sampled from the truncated log-normal jump CDF. */
int QNN_JumpHoldFrames(uint32_t *rng);

#endif /* QNN_HOLD_SAMPLERS_H */
