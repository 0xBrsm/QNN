/*
 * qnn_pmove_hooks.h — pmove instrumentation for the QWD labeler.
 *
 * Provides the global flag that patched pmove.c JumpButton() sets on
 * its success branch, plus state save/restore helpers so the labeler's
 * per-cmd pmove driver doesn't contaminate qnn_phys's 9-candidate
 * MVD-inference path that mutates the same pmove globals.
 */

#ifndef QNN_PMOVE_HOOKS_H
#define QNN_PMOVE_HOOKS_H

#include "qnn.h"

/* Set to 1 by the patched pmove.c JumpButton() success branch when the
 * player physically left the ground at the cmd just integrated.  The
 * QWD labeler's per-cmd driver clears this before each PlayerMove()
 * call and OR's the post-call value across the cmd window. */
extern int qnn_pmove_jump_attacked;

/* Snapshot of pmove globals our per-cmd driver mutates.  Saved before
 * we seed pmove from the labeler snapshot and restored after the cmd
 * window finishes, so any MVD inference running in the same tick (via
 * force_mvd_emit) sees unchanged state. */
typedef struct
{
	vec3_t      origin;
	vec3_t      velocity;
	vec3_t      angles;
	usercmd_t   cmd;
	int         dead;
	int         spectator;
	float       waterjumptime;
	int         oldbuttons;
	int         numphysent;
	int         onground;
	int         waterlevel;
	int         watertype;
} qnn_pmove_save_t;

void QNN_PmoveSave(qnn_pmove_save_t *save);
void QNN_PmoveRestore(const qnn_pmove_save_t *save);

#endif /* QNN_PMOVE_HOOKS_H */
