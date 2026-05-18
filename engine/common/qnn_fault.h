#ifndef QNN_FAULT_H
#define QNN_FAULT_H

/* Fatal-signal diagnostics.
 *
 * QNN_FaultInit installs handlers for SIGSEGV / SIGBUS / SIGFPE / SIGILL /
 * SIGABRT that dump a symbolicated stack trace to stderr before the
 * process dies.  The handlers re-raise after logging so the default
 * core-dump path still runs.
 *
 * No runtime behavior change in the happy path; handlers fire only when
 * the engine was about to crash anyway.  Stack traces need the binary
 * linked with ``-rdynamic`` (see build scripts).
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Install handlers.  ``worker_name`` is copied into a small static
 * buffer and emitted in every trace header so logs from parallel
 * workers can be grouped.  Call once, early, from each worker's
 * ``main()``. */
void QNN_FaultInit(const char *worker_name);

/* Update the "current demo / episode" tag that the fault handler
 * prints alongside the trace.  Pass NULL or "" to clear.  Safe to call
 * from the main thread at any time; the handler reads it via
 * async-signal-safe primitives. */
void QNN_FaultSetContext(const char *context);

#ifdef __cplusplus
}
#endif

#endif  /* QNN_FAULT_H */
