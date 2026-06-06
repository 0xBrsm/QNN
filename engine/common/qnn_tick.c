#include "qnn_tick.h"

cvar_t qnn_tick_hz = {"qnn_tick_hz", "0"};

void QNN_TickRegister(void)
{
	Cvar_RegisterVariable(&qnn_tick_hz);
}

qboolean QNN_TickGate(qboolean is_timedemo, float incoming_time,
                      float native_cap_hz,
                      double *p_realtime, double *p_oldrealtime)
{
	(void)native_cap_hz;  /* legacy arg — caller still passes it for ABI
	                       * compat, but the cap is no longer honored.
	                       * qnn_tick_hz is the only rate authority. */

	*p_realtime += incoming_time;
	if (*p_oldrealtime > *p_realtime)
		*p_oldrealtime = 0;

	if (is_timedemo)
		return true;

	/* qnn_tick_hz == 0 means "no rate limit" — gate passes every call.
	 * Callers needing native-rate emission pair this with their own
	 * per-server-frame cadence (pump until a sequence advance).
	 * qnn_tick_hz > 0 gates at exactly that rate, with no upstream
	 * cl_maxfps clamp. */
	if (qnn_tick_hz.value <= 0)
		return true;

	if ((*p_realtime - *p_oldrealtime) < 1.0 / qnn_tick_hz.value)
		return false;

	return true;
}
