/*
 * qnn_sys.c (qw) — System layer and shared globals for the QW demo worker.
 *
 * QW equivalent of nq/qnn_sys.c.  The QW client has a different Sys_*
 * API (Sys_DoubleTime instead of Sys_FloatTime, no Sys_DebugNumber,
 * Sys_SetFPCW).  Most of the JSON/look/switch utilities are shared
 * with NQ via qnn.h and qnn_sys.c.  We compile qnn_sys.c as-is
 * EXCEPT for the globals and Sys_* stubs — those are QW-specific
 * and live here.
 *
 * Strategy: This file provides the QW Sys_* functions and the
 * duplicate globals that the QW engine expects.  The JSON/math
 * utilities from qnn_sys.c are NOT duplicated — we include that
 * file via the build system.  To avoid duplicate symbols, we
 * define QNN_QW_SYS_ONLY=1 to skip the Sys_* stubs in qnn_sys.c.
 *
 * Actually — we take a simpler approach.  This file replaces
 * qnn_sys.c entirely for the QW build.  It contains:
 *   1. QW Sys_* stubs
 *   2. Shared globals
 *   3. JSON utilities (copy from qnn_sys.c)
 *   4. Basedir resolution
 *   5. Tick resampling
 *   6. Binary write helpers
 *   7. Nav query handler
 *   8. QNN_ProgString stub
 *   9. Look/switch helpers
 *
 * This duplication is unfortunate but avoids #ifdef pollution in
 * the shared qnn_sys.c.  All QW-specific differences are isolated
 * here.
 */

#include "qnn.h"
#include "qnn_object.h"
#include "qnn_route.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* ── shared globals ──────────────────────────────────────────────── */

qnn_map_state_t qnn_map_state;
char qnn_basedir_storage[MAX_OSPATH] = ".";

/* QW client declares isDedicated as extern in some paths */
qboolean isDedicated;

/* QW uses basedir/cachedir differently — they're set in host_parms */
char *basedir = qnn_basedir_storage;
char *cachedir = "/tmp";

/* QW doesn't have sys_linerefresh, but some code paths may reference it */

/* ── Sys_* stubs (QW engine system layer) ────────────────────────── */

void Sys_Printf(char *fmt, ...)
{
	(void)fmt;
}

void Sys_Quit(void)
{
	Host_Shutdown();
	exit(0);
}

void Sys_Error(char *error, ...)
{
	va_list argptr;

	va_start(argptr, error);
	vfprintf(stderr, error, argptr);
	va_end(argptr);
	fputc('\n', stderr);
	Host_Shutdown();
	exit(1);
}

int Sys_FileTime(char *path)
{
	struct stat st;

	return stat(path, &st) == -1 ? -1 : (int)st.st_mtime;
}

void Sys_mkdir(char *path)
{
	mkdir(path, 0777);
}

int Sys_FileOpenRead(char *path, int *handle)
{
	int h;
	struct stat st;

	h = open(path, O_RDONLY, 0666);
	*handle = h;
	if (h == -1)
		return -1;
	if (fstat(h, &st) == -1)
		Sys_Error("Error fstating %s", path);
	return (int)st.st_size;
}

int Sys_FileOpenWrite(char *path)
{
	int handle;

	umask(0);
	handle = open(path, O_RDWR | O_CREAT | O_TRUNC, 0666);
	if (handle == -1)
		Sys_Error("Error opening %s: %s", path, strerror(errno));
	return handle;
}

int Sys_FileWrite(int handle, void *src, int count)
{
	return (int)write(handle, src, count);
}

void Sys_FileClose(int handle)
{
	close(handle);
}

void Sys_FileSeek(int handle, int position)
{
	lseek(handle, position, SEEK_SET);
}

int Sys_FileRead(int handle, void *dest, int count)
{
	return (int)read(handle, dest, count);
}

void Sys_DebugLog(char *file, char *fmt, ...)
{
	(void)file;
	(void)fmt;
}

/* QW uses Sys_DoubleTime instead of Sys_FloatTime */
double Sys_DoubleTime(void)
{
	struct timeval tv;
	static int initialized;
	static double base;
	double now;

	gettimeofday(&tv, NULL);
	now = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	if (!initialized)
	{
		base = now;
		initialized = 1;
	}
	return now - base;
}

void Sys_SendKeyEvents(void)
{
}

void Sys_Sleep(void)
{
}

char *Sys_ConsoleInput(void)
{
	return NULL;
}

void Sys_HighFPPrecision(void)
{
}

void Sys_LowFPPrecision(void)
{
}

void Sys_SetFPCW(void)
{
}

void Sys_MakeCodeWriteable(unsigned long startaddr, unsigned long length)
{
	long page_size;
	unsigned long addr;

	page_size = sysconf(_SC_PAGESIZE);
	addr = startaddr & ~(unsigned long)(page_size - 1);
	mprotect((void *)addr, length + startaddr - addr, PROT_READ | PROT_WRITE | PROT_EXEC);
}

/* ── basedir resolution ──────────────────────────────────────────── */

void QNN_ResolveBasedir(char *out, size_t out_size)
{
	const char *env = getenv("QUAKE_BASEDIR");
	if (env && env[0])
		snprintf(out, out_size, "%s", env);
	else
		snprintf(out, out_size, ".");
}

/* QW has no pr_strings — stub QNN_ProgString to return empty */
const char *QNN_ProgString(string_t value)
{
	(void)value;
	return "";
}

/* PM_RecursiveHullCheck — defined in upstream pmovetst.c but not declared
 * in pmove.h.  Without this prototype the call below would default to
 * int-based arg passing and the trace pointer would land in the wrong
 * register. */
qboolean PM_RecursiveHullCheck(hull_t *hull, int num, float p1f, float p2f,
	vec3_t p1, vec3_t p2, pmtrace_t *trace);

/* QNN_TraceLine (QW) — adapt upstream's PM_RecursiveHullCheck to our
 * trace_t.  pmtrace_t and trace_t share layout for the fields we read
 * (fraction, endpos), so we copy those out after the trace.  Shared
 * declaration in qnn_object.h. */
void QNN_TraceLine(const vec3_t start, const vec3_t end, trace_t *trace)
{
	pmtrace_t pm;
	int mover_count;
	int i;

	if (trace == NULL)
		return;
	memset(trace, 0, sizeof(*trace));
	if (cl.worldmodel == NULL)
	{
		trace->fraction = 1.0f;
		VectorCopy(end, trace->endpos);
		return;
	}
	memset(&pm, 0, sizeof(pm));
	pm.fraction = 1.0f;
	PM_RecursiveHullCheck(cl.worldmodel->hulls, 0, 0, 1,
		(float *)start, (float *)end, &pm);
	trace->allsolid = pm.allsolid;
	trace->startsolid = pm.startsolid;
	trace->fraction = pm.fraction;
	VectorCopy(pm.endpos, trace->endpos);

	/* Clip against solid movers at their live origins so movers occlude
	 * like static world geometry (spatial rays shorten at a door face; an
	 * enemy behind a closed door drops from SIGHT).  The mover set is
	 * cached once per observation frame (QNN_TraceMoverCacheRefresh), so
	 * this loop adds no re-scan per ray.  When no mover intersects nearer
	 * than the world trace, `trace` is left exactly as PM_RecursiveHullCheck
	 * produced it — bit-identical to the static-only path.  Mirrors the NQ
	 * QNN_TraceLine (SV_RecursiveHullCheck) for QW/NQ obs parity. */
	mover_count = QNN_TraceMoverCacheRefresh();
	for (i = 0; i < mover_count; i++)
	{
		model_t *m = QNN_TraceMoverModel(i);
		float *origin = QNN_TraceMoverOrigin(i);
		hull_t *hull;
		pmtrace_t mpm;
		vec3_t start_l, end_l;

		if (m == NULL)
			continue;
		hull = &m->hulls[0];
		VectorSubtract(start, origin, start_l);
		VectorSubtract(end, origin, end_l);
		memset(&mpm, 0, sizeof(mpm));
		mpm.fraction = 1.0f;
		PM_RecursiveHullCheck(hull, hull->firstclipnode, 0, 1,
			start_l, end_l, &mpm);
		if (mpm.fraction < trace->fraction)
		{
			vec3_t delta;
			trace->fraction = mpm.fraction;
			trace->allsolid = mpm.allsolid;
			trace->startsolid = mpm.startsolid;
			VectorSubtract(end, start, delta);
			VectorMA(start, mpm.fraction, delta, trace->endpos);
		}
	}
}
