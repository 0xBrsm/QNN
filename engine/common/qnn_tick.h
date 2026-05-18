/*
 * qnn_tick.h — Centralized engine tick gate.
 *
 * Replaces the per-engine framerate clamps (QW: cl_maxfps + 30/72
 * floor/ceiling in Host_Frame; NQ: hardcoded 1/72 cap in Host_FilterTime)
 * with a single cvar we own.  When qnn_tick_hz > 0, both engines gate
 * at that rate; when 0 (the default), each engine falls back to its
 * native cap.
 *
 * The gate decides only whether to advance the host frame.  It does
 * not touch host_frametime — engines compute that from
 * realtime - oldrealtime as before, so live-play time tracking is
 * unchanged.
 */
#ifndef QNN_TICK_H
#define QNN_TICK_H

#include "qnn.h"

extern cvar_t qnn_tick_hz;

/* Register qnn_tick_hz with the cvar system.  Call once during Host_Init
 * (NQ) / CL_Init (QW). */
void QNN_TickRegister(void);

/* Returns true when enough wall time has elapsed to run a host frame.
 *
 * Mutates *p_realtime by adding incoming_time, and applies a
 * clock-reset guard against negative deltas.  Does not touch
 * *p_oldrealtime — caller updates it after computing host_frametime.
 *
 *   is_timedemo     pass cls.timedemo
 *   incoming_time   the float arg from Host_Frame()
 *   native_cap_hz   upstream cap when qnn_tick_hz is 0 (QW: cl_maxfps
 *                   if set else rate-based; NQ: 72)
 */
qboolean QNN_TickGate(qboolean is_timedemo, float incoming_time,
                      float native_cap_hz,
                      double *p_realtime, double *p_oldrealtime);

#endif /* QNN_TICK_H */
