/* Per-demo stall watchdog.  See qnn_watchdog.h. */

#include "qnn_watchdog.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QNN_WATCHDOG_DEFAULT_SECONDS 10

/* Main-thread increments, handler reads.  sig_atomic_t so reads inside
 * the signal handler are POSIX-correct regardless of host word size. */
static volatile sig_atomic_t qnn_watchdog_counter;
static volatile sig_atomic_t qnn_watchdog_last_seen;
static volatile sig_atomic_t qnn_watchdog_timeout;
static volatile sig_atomic_t qnn_watchdog_installed;

/* itoa for small non-negative ints; writes into buf right-aligned,
 * returns a pointer into buf at the first digit.  Async-signal-safe. */
static const char *qnn_watchdog_utoa(unsigned int v, char *buf, size_t buf_len)
{
	size_t i = buf_len;
	buf[--i] = '\0';
	if (v == 0)
	{
		buf[--i] = '0';
		return &buf[i];
	}
	while (v > 0 && i > 0)
	{
		buf[--i] = (char)('0' + (v % 10));
		v /= 10;
	}
	return &buf[i];
}

static size_t qnn_watchdog_strlen(const char *s)
{
	const char *p = s;
	while (*p != '\0')
		++p;
	return (size_t)(p - s);
}

static void qnn_watchdog_write(const char *buf, size_t n)
{
	while (n > 0)
	{
		ssize_t w = write(STDERR_FILENO, buf, n);
		if (w <= 0)
			return;
		buf += (size_t)w;
		n -= (size_t)w;
	}
}

static void qnn_watchdog_write_cstr(const char *s)
{
	qnn_watchdog_write(s, qnn_watchdog_strlen(s));
}

static void qnn_watchdog_handler(int sig)
{
	(void)sig;

	if (qnn_watchdog_counter != qnn_watchdog_last_seen)
	{
		/* Healthy: main loop advanced since last fire. */
		qnn_watchdog_last_seen = qnn_watchdog_counter;
		alarm((unsigned int)qnn_watchdog_timeout);
		return;
	}

	/* Stalled.  Emit one diagnostic line and exit with a well-known
	 * code so the parent can classify this as permanent. */
	char numbuf[16];
	qnn_watchdog_write_cstr("\n[worker] watchdog: stalled ");
	qnn_watchdog_write_cstr(qnn_watchdog_utoa((unsigned int)qnn_watchdog_timeout, numbuf, sizeof(numbuf)));
	qnn_watchdog_write_cstr("s without main-loop progress\n");

	_exit(QNN_WATCHDOG_EXIT_CODE);
}

void QNN_WatchdogBegin(int timeout_seconds)
{
	if (timeout_seconds < 1)
		timeout_seconds = QNN_WATCHDOG_DEFAULT_SECONDS;

	const char *env = getenv("QNN_WATCHDOG_SECONDS");
	if (env != NULL && env[0] != '\0')
	{
		int parsed = atoi(env);
		if (parsed > 0)
			timeout_seconds = parsed;
	}
	qnn_watchdog_timeout = (sig_atomic_t)timeout_seconds;

	if (!qnn_watchdog_installed)
	{
		struct sigaction sa;
		memset(&sa, 0, sizeof(sa));
		sa.sa_handler = qnn_watchdog_handler;
		sigemptyset(&sa.sa_mask);
		/* No SA_RESETHAND: the handler re-arms itself across firings. */
		sa.sa_flags = 0;
		sigaction(SIGALRM, &sa, NULL);
		qnn_watchdog_installed = 1;
	}

	qnn_watchdog_counter = 0;
	qnn_watchdog_last_seen = 0;
	alarm((unsigned int)timeout_seconds);
}

void QNN_WatchdogTick(void)
{
	qnn_watchdog_counter += 1;
}

void QNN_WatchdogEnd(void)
{
	alarm(0);
}
