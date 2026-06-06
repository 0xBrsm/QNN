/*
 * qnn_pmove_hooks.c — pmove instrumentation for the QWD labeler.
 *
 * Implements the global flag patched pmove.c JumpButton() sets when
 * the ground-jump success branch runs, plus save/restore of the pmove
 * globals our QWD-labeler per-cmd driver mutates.
 *
 * The flag is a single int rather than an event queue because the
 * labeler only needs "did any cmd in this tick's window fire a jump"
 * — caller clears it before each PlayerMove() call and ORs the
 * post-call value across the window.
 */

#include "qnn_pmove_hooks.h"

/* QW pmove interface (same extern pattern as qnn_phys.c). */
extern playermove_t pmove;
extern int onground;
extern int waterlevel;
extern int watertype;

int qnn_pmove_jump_fired = 0;

void QNN_PmoveSave(qnn_pmove_save_t *save)
{
	VectorCopy(pmove.origin,   save->origin);
	VectorCopy(pmove.velocity, save->velocity);
	VectorCopy(pmove.angles,   save->angles);
	save->cmd            = pmove.cmd;
	save->dead           = pmove.dead;
	save->spectator      = pmove.spectator;
	save->waterjumptime  = pmove.waterjumptime;
	save->oldbuttons     = pmove.oldbuttons;
	save->numphysent     = pmove.numphysent;
	save->onground       = onground;
	save->waterlevel     = waterlevel;
	save->watertype      = watertype;
}

void QNN_PmoveRestore(const qnn_pmove_save_t *save)
{
	VectorCopy(save->origin,   pmove.origin);
	VectorCopy(save->velocity, pmove.velocity);
	VectorCopy(save->angles,   pmove.angles);
	pmove.cmd            = save->cmd;
	pmove.dead           = save->dead;
	pmove.spectator      = save->spectator;
	pmove.waterjumptime  = save->waterjumptime;
	pmove.oldbuttons     = save->oldbuttons;
	pmove.numphysent     = save->numphysent;
	onground             = save->onground;
	waterlevel           = save->waterlevel;
	watertype            = save->watertype;
}
