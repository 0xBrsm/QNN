#ifndef QNN_WATCHDOG_H
#define QNN_WATCHDOG_H

/* Per-demo stall watchdog.
 *
 * QNN_WatchdogBegin arms a SIGALRM-driven heartbeat that fires every
 * ``timeout_seconds``.  The main demo loop must call QNN_WatchdogTick()
 * after each Host_Frame return.  When the handler fires and the tick
 * counter has not advanced since the prior firing, the worker writes a
 * diagnostic line to stderr and calls ``_exit(QNN_WATCHDOG_EXIT_CODE)``.
 *
 * The parent (Python ``bc/collect.py``) recognizes the stderr marker
 * and exit code as a permanent, non-retryable failure.
 *
 * Call QNN_WatchdogEnd after a demo loop finishes cleanly (or before
 * entering any long non-demo work) to cancel the pending alarm.  All
 * three entry points are safe to call from the main thread; the handler
 * itself only touches async-signal-safe primitives.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define QNN_WATCHDOG_EXIT_CODE 77

/* Arm the watchdog with a per-firing interval.  Values < 1 are clamped
 * to 1 (seconds).  If ``QNN_WATCHDOG_SECONDS`` is set in the environment
 * and parses as a positive integer, it overrides ``timeout_seconds``.
 * Installs the SIGALRM handler on first call; subsequent calls only
 * reset the counters and re-arm ``alarm()``.
 */
void QNN_WatchdogBegin(int timeout_seconds);

/* Record progress.  Cheap: one volatile increment, no syscall. */
void QNN_WatchdogTick(void);

/* Cancel any pending alarm.  Idempotent. */
void QNN_WatchdogEnd(void);

#ifdef __cplusplus
}
#endif

#endif  /* QNN_WATCHDOG_H */
