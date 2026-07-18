#ifndef QNN_ARENA_OBSERVER_H
#define QNN_ARENA_OBSERVER_H

#include "qnn.h"

qboolean QNN_ArenaObserverReady(void);
void QNN_ArenaObserverPrepare(void);
void QNN_ArenaObserverWrite(FILE *out, const qnn_action_t *previous_action,
	int tick, int steps, qboolean reset_flag);

#endif
