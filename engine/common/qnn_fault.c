/* Fatal-signal diagnostics.  See qnn_fault.h for contract. */

#include "qnn_fault.h"

#include <execinfo.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define QNN_FAULT_MAX_FRAMES 64
#define QNN_FAULT_NAME_LEN   64
#define QNN_FAULT_CTX_LEN    256

/* Statics populated by QNN_FaultInit / QNN_FaultSetContext; read by the
 * signal handler via async-signal-safe primitives only. */
static char qnn_fault_worker[QNN_FAULT_NAME_LEN];
static char qnn_fault_ctx[QNN_FAULT_CTX_LEN];
static volatile sig_atomic_t qnn_fault_ctx_len = 0;

/* write() the whole buffer or skip on error; async-signal-safe. */
static void qnn_fault_write(int fd, const void *buf, size_t n)
{
	const char *p = (const char *)buf;
	while (n > 0)
	{
		ssize_t w = write(fd, p, n);
		if (w <= 0)
			return;
		p += (size_t)w;
		n -= (size_t)w;
	}
}

/* strlen is specified signal-safe on POSIX as of issue 7, but we use
 * our own tiny variant to stay strictly within the narrow async-signal
 * set and avoid depending on libc version. */
static size_t qnn_fault_strlen(const char *s)
{
	const char *p = s;
	while (*p != '\0')
		++p;
	return (size_t)(p - s);
}

static const char *qnn_fault_signame(int sig)
{
	switch (sig)
	{
		case SIGSEGV: return "SIGSEGV";
		case SIGBUS:  return "SIGBUS";
		case SIGFPE:  return "SIGFPE";
		case SIGILL:  return "SIGILL";
		case SIGABRT: return "SIGABRT";
		default:      return "?";
	}
}

static void qnn_fault_handler(int sig, siginfo_t *info, void *ucontext)
{
	(void)info;
	(void)ucontext;

	int fd = STDERR_FILENO;

	static const char hdr_open[] = "\n=== QNN FAULT: worker=";
	qnn_fault_write(fd, hdr_open, sizeof(hdr_open) - 1);
	qnn_fault_write(fd, qnn_fault_worker, qnn_fault_strlen(qnn_fault_worker));

	static const char sig_lbl[] = " signal=";
	qnn_fault_write(fd, sig_lbl, sizeof(sig_lbl) - 1);
	const char *signame = qnn_fault_signame(sig);
	qnn_fault_write(fd, signame, qnn_fault_strlen(signame));

	/* Snapshot ctx len once — a second write from the main thread would
	 * race, but we don't care about an intermediate partial update as
	 * long as we don't walk off the buffer. */
	sig_atomic_t ctx_len = qnn_fault_ctx_len;
	if (ctx_len > 0 && ctx_len < QNN_FAULT_CTX_LEN)
	{
		static const char ctx_lbl[] = " ctx=";
		qnn_fault_write(fd, ctx_lbl, sizeof(ctx_lbl) - 1);
		qnn_fault_write(fd, qnn_fault_ctx, (size_t)ctx_len);
	}

	static const char hdr_end[] = " ===\n";
	qnn_fault_write(fd, hdr_end, sizeof(hdr_end) - 1);

	/* backtrace / backtrace_symbols_fd are documented async-signal-safe
	 * on glibc.  Requires the binary to be linked with -rdynamic for
	 * symbol names rather than raw addresses. */
	void *frames[QNN_FAULT_MAX_FRAMES];
	int n = backtrace(frames, QNN_FAULT_MAX_FRAMES);
	if (n > 0)
		backtrace_symbols_fd(frames, n, fd);

	static const char tail[] = "=== end trace ===\n";
	qnn_fault_write(fd, tail, sizeof(tail) - 1);

	/* SA_RESETHAND already restored the default handler; re-raise so
	 * the default kernel path produces a core dump and the parent sees
	 * the correct exit status (WTERMSIG). */
	raise(sig);
}

void QNN_FaultInit(const char *worker_name)
{
	memset(qnn_fault_worker, 0, sizeof(qnn_fault_worker));
	if (worker_name != NULL && worker_name[0] != '\0')
	{
		size_t n = qnn_fault_strlen(worker_name);
		if (n >= sizeof(qnn_fault_worker))
			n = sizeof(qnn_fault_worker) - 1;
		memcpy(qnn_fault_worker, worker_name, n);
	}
	else
	{
		qnn_fault_worker[0] = '?';
	}

	struct sigaction sa;
	static const int sigs[] = { SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT };
	size_t i;

	memset(&sa, 0, sizeof(sa));
	sa.sa_sigaction = qnn_fault_handler;
	sigemptyset(&sa.sa_mask);
	/* SA_SIGINFO for three-arg handler; SA_RESETHAND so a fault inside
	 * the handler itself falls through to the default action instead of
	 * recursing. */
	sa.sa_flags = SA_SIGINFO | SA_RESETHAND;

	for (i = 0; i < sizeof(sigs) / sizeof(sigs[0]); ++i)
		sigaction(sigs[i], &sa, NULL);
}

void QNN_FaultSetContext(const char *context)
{
	if (context == NULL || context[0] == '\0')
	{
		qnn_fault_ctx_len = 0;
		return;
	}
	size_t n = qnn_fault_strlen(context);
	if (n >= QNN_FAULT_CTX_LEN)
		n = QNN_FAULT_CTX_LEN - 1;
	/* Order matters: write the bytes, then publish the length.  A
	 * signal fired mid-update will read a consistent prefix (the
	 * handler bounds its read by ctx_len). */
	memcpy(qnn_fault_ctx, context, n);
	qnn_fault_ctx[n] = '\0';
	qnn_fault_ctx_len = (sig_atomic_t)n;
}
