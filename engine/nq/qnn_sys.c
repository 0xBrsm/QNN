#include "qnn.h"
#include "qnn_object.h"
#include "qnn_route.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* ── shared globals ──────────────────────────────────────────────── */

qnn_map_state_t qnn_map_state;
char qnn_basedir_storage[MAX_OSPATH] = ".";

qboolean isDedicated;
char *basedir = qnn_basedir_storage;
char *cachedir = "/tmp";
cvar_t sys_linerefresh = {"sys_linerefresh", "0"};

/* ── Sys_* stubs (Quake engine system layer) ─────────────────────── */

void Sys_DebugNumber(int y, int val)
{
	(void)y;
	(void)val;
}

__attribute__((weak)) void Sys_Printf(char *fmt, ...)
{
	va_list ap;
	static int route_resolved = 0;
	static int route_to_stderr = 0;
	FILE *out;

	if (!route_resolved) {
		const char *env = getenv("QNN_STDOUT_PROTOCOL");
		route_to_stderr = (env != NULL && env[0] != '\0' && env[0] != '0');
		route_resolved = 1;
	}
	out = route_to_stderr ? stderr : stdout;
	va_start(ap, fmt);
	vfprintf(out, fmt, ap);
	va_end(ap);
	fflush(out);
}

void Sys_Quit(void)
{
	Host_Shutdown();
	exit(0);
}

void Sys_Init(void)
{
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

void Sys_Warn(char *warning, ...)
{
	va_list argptr;

	va_start(argptr, warning);
	vfprintf(stderr, warning, argptr);
	va_end(argptr);
	fputc('\n', stderr);
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

void Sys_EditFile(char *filename)
{
	(void)filename;
}

double Sys_FloatTime(void)
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

void Sys_LineRefresh(void)
{
}

void Sys_SendKeyEvents(void)
{
}

void Sys_Sleep(void)
{
}

/* stdin-is-protocol latch: binary-protocol processes (the arena
 * server) own fd 0 for opcode frames — OP_ATTACH_DECL arrives WHILE
 * the dedicated host pumps sign-on frames, so the console reader here
 * must never race it for bytes.  Set once in main() before Host_Init;
 * defaults off so stock dedicated consoles keep working. */
qboolean qnn_stdin_is_protocol;

char *Sys_ConsoleInput(void)
{
	static char text[256];
	fd_set fdset;
	struct timeval timeout;
	ssize_t len;

	if (qnn_stdin_is_protocol)
		return NULL;

	FD_ZERO(&fdset);
	FD_SET(0, &fdset);
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;
	if (select(1, &fdset, NULL, NULL, &timeout) <= 0 || !FD_ISSET(0, &fdset))
		return NULL;
	len = read(0, text, sizeof(text) - 1);
	if (len < 1)
		return NULL;
	/* Strip trailing newline so Cbuf_AddText doesn't double-terminate. */
	if (text[len - 1] == '\n')
		text[len - 1] = '\0';
	else
		text[len] = '\0';
	return text;
}

void Sys_HighFPPrecision(void)
{
}

void Sys_LowFPPrecision(void)
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

const char *QNN_ProgString(string_t value)
{
	if (!value)
		return "";
	return pr_strings + value;
}

/* QNN_TraceLine (NQ) — forwards to upstream's SV_RecursiveHullCheck from
 * world.c, then clips the same segment against every blocking brush
 * submodel at its live origin so entities occlude like static geometry
 * (spatial rays shorten at a door face; an enemy behind a closed door
 * drops from SIGHT).  Shared declaration in qnn_object.h. */
void QNN_TraceLine(const vec3_t start, const vec3_t end, trace_t *trace)
{
	int mover_count;
	int i;

	if (trace == NULL)
		return;
	if (cl.worldmodel == NULL)
	{
		memset(trace, 0, sizeof(*trace));
		trace->fraction = 1.0f;
		VectorCopy(end, trace->endpos);
		return;
	}
	memset(trace, 0, sizeof(*trace));
	/* Upstream NQ convention: caller sets fraction=1 before recursion.
	 * SV_RecursiveHullCheck only WRITES fraction when an obstruction is
	 * found — on a clear line it leaves whatever value was there.  Without
	 * this seed, a memset-zeroed trace returns fraction=0 ("blocked") on
	 * every unobstructed call, which silently broke VIS-gated actor obs
	 * after 536c99b5 flipped QNN_PRIMARY_OBS_ACTOR from PVS to VIS. */
	trace->fraction = 1.0f;
	SV_RecursiveHullCheck(cl.worldmodel->hulls, 0, 0, 1,
		(float *)start, (float *)end, trace);

	/* Clip against blocking brushes at their live origins.  The set is
	 * cached once per observation frame (QNN_TraceMoverCacheRefresh), so
	 * this loop adds no re-scan per ray.  When no mover intersects nearer
	 * than the world trace, `trace` is left exactly as world.c produced
	 * it — bit-identical to the static-only path. */
	mover_count = QNN_TraceMoverCacheRefresh();
	for (i = 0; i < mover_count; i++)
	{
		model_t *m = QNN_TraceMoverModel(i);
		float *origin = QNN_TraceMoverOrigin(i);
		hull_t *hull;
		trace_t mt;
		vec3_t start_l, end_l;

		if (m == NULL)
			continue;
		hull = &m->hulls[0];
		VectorSubtract(start, origin, start_l);
		VectorSubtract(end, origin, end_l);
		memset(&mt, 0, sizeof(mt));
		mt.fraction = 1.0f;
		SV_RecursiveHullCheck(hull, hull->firstclipnode, 0, 1,
			start_l, end_l, &mt);
		if (mt.fraction < trace->fraction)
		{
			vec3_t delta;
			trace->fraction = mt.fraction;
			trace->allsolid = mt.allsolid;
			trace->startsolid = mt.startsolid;
			VectorSubtract(end, start, delta);
			VectorMA(start, mt.fraction, delta, trace->endpos);
		}
	}
	QNN_TraceStaticBlockers(start, end, false, trace);
}

/* QNN_TraceClearance (NQ) — hull-1 line trace carrying the hit plane.
 * Contract in qnn_object.h.  Structure mirrors QNN_TraceLine; deltas:
 * clip hull 1 instead of hull 0, and the mover branch copies mt.plane
 * through, translating the mover-local plane to world
 * (dist' = dist + n·origin).  The world path needs no copy —
 * SV_RecursiveHullCheck writes trace->plane directly. */
void QNN_TraceClearance(const vec3_t start, const vec3_t end, trace_t *trace)
{
	hull_t *hull;
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
	/* Seed fraction=1 — SV_RecursiveHullCheck only writes it on a hit
	 * (same convention note as QNN_TraceLine above). */
	trace->fraction = 1.0f;
	hull = &cl.worldmodel->hulls[1];
	if (hull->clipnodes == NULL || hull->planes == NULL)
		hull = &cl.worldmodel->hulls[0]; /* degenerate map — point fallback */
	SV_RecursiveHullCheck(hull, hull->firstclipnode, 0, 1,
		(float *)start, (float *)end, trace);

	mover_count = QNN_TraceMoverCacheRefresh();
	for (i = 0; i < mover_count; i++)
	{
		model_t *m = QNN_TraceMoverModel(i);
		float *origin = QNN_TraceMoverOrigin(i);
		hull_t *mh;
		trace_t mt;
		vec3_t start_l, end_l;

		if (m == NULL)
			continue;
		mh = &m->hulls[1];
		if (mh->clipnodes == NULL || mh->planes == NULL)
			mh = &m->hulls[0];
		VectorSubtract(start, origin, start_l);
		VectorSubtract(end, origin, end_l);
		memset(&mt, 0, sizeof(mt));
		mt.fraction = 1.0f;
		SV_RecursiveHullCheck(mh, mh->firstclipnode, 0, 1,
			start_l, end_l, &mt);
		if (mt.fraction < trace->fraction)
		{
			vec3_t delta;
			trace->fraction = mt.fraction;
			trace->allsolid = mt.allsolid;
			trace->startsolid = mt.startsolid;
			VectorSubtract(end, start, delta);
			VectorMA(start, mt.fraction, delta, trace->endpos);
			VectorCopy(mt.plane.normal, trace->plane.normal);
			trace->plane.dist = mt.plane.dist
				+ DotProduct(mt.plane.normal, origin);
		}
	}
	QNN_TraceStaticBlockers(start, end, true, trace);
}
