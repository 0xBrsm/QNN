#ifndef QNN_ARENA_OBSERVER_H
#define QNN_ARENA_OBSERVER_H

#include "qnn.h"

struct qnn_obs_plan_s;

qboolean QNN_ArenaObserverReady(void);
void QNN_ArenaObserverPrepare(void);
/* Emit + serialize one observation frame for this seat.  `plan` is the
 * seat's compiled obs plan (OP_ATTACH_DECL) — the frame is exactly
 * plan->frame_bytes; NULL = the default plan (today's 864-byte frame,
 * bit-identical — the no-declaration legacy contract).  `dt` is the
 * caller's per-tick step (1/fixed_tick_hz) — fed straight into
 * QNN_PredictTick/QNN_IOUpdate so velocity and IO-delta math track the
 * server's actual tick rate instead of an assumed one. */
void QNN_ArenaObserverWrite(FILE *out, const struct qnn_obs_plan_s *plan,
	const qnn_action_t *previous_action,
	int tick, int steps, qboolean reset_flag, float dt);

#endif
