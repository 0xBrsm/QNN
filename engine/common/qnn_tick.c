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
	float hz;

	*p_realtime += incoming_time;
	if (*p_oldrealtime > *p_realtime)
		*p_oldrealtime = 0;

	if (is_timedemo)
		return true;

	hz = (qnn_tick_hz.value > 0) ? qnn_tick_hz.value : native_cap_hz;
	if (hz <= 0)
		return true;

	if ((*p_realtime - *p_oldrealtime) < 1.0 / hz)
		return false;

	return true;
}
