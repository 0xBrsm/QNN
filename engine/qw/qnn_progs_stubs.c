/*
 * qnn_progs_stubs.c — stubs for QW server symbols referenced by the
 * server-side QC VM (pr_exec.c, pr_edict.c, pr_cmds.c) but unused in
 * the labeler-collect predicate-evaluation context.
 *
 * The labeler-mode collect path loads qwprogs.dat through the real QW
 * VM to evaluate per-tick operative-input predicates (see qnn_progs.c).
 * The QC builtin table in pr_cmds.c references many server-side
 * functions for things like reliable client writes, multicast, world
 * collision tracing, frag logs.  None of the QC functions we invoke
 * (W_ChangeWeapon, W_Attack's pre-fire path, W_WeaponFrame's cooldown
 * check) actually call those builtins, but the linker still wants
 * resolutions.  This file provides safe no-op stubs.
 *
 * SV_Error is the exception — it's the engine's fatal-error handler
 * and we want it to die loudly rather than silently no-op.
 */

#include "qwsvdef.h"

/* ── globals the VM references ──────────────────────────────────── */

char       localinfo[MAX_LOCALINFO_STRING + 1];
FILE      *sv_fraglogfile = NULL;
server_static_t svs;
cvar_t     teamplay = {"teamplay", "0"};

/* ── fatal error handler ────────────────────────────────────────── */

void SV_Error(char *error, ...)
{
	char buf[1024];
	va_list ap;
	va_start(ap, error);
	vsnprintf(buf, sizeof(buf), error, ap);
	va_end(ap);
	fprintf(stderr, "qnn_progs SV_Error: %s\n", buf);
	abort();
}

/* ── client reliable writes ─ no client → no-op ─────────────────── */

void ClientReliableCheckBlock(client_t *cl, int maxsize) { (void)cl; (void)maxsize; }
void ClientReliable_FinishWrite(client_t *cl)            { (void)cl; }
void ClientReliableWrite_Begin (client_t *cl, int c, int maxsize) { (void)cl; (void)c; (void)maxsize; }
void ClientReliableWrite_Angle (client_t *cl, float f)   { (void)cl; (void)f; }
void ClientReliableWrite_Angle16(client_t *cl, float f)  { (void)cl; (void)f; }
void ClientReliableWrite_Byte  (client_t *cl, int c)     { (void)cl; (void)c; }
void ClientReliableWrite_Char  (client_t *cl, int c)     { (void)cl; (void)c; }
void ClientReliableWrite_Float (client_t *cl, float f)   { (void)cl; (void)f; }
void ClientReliableWrite_Coord (client_t *cl, float f)   { (void)cl; (void)f; }
void ClientReliableWrite_Long  (client_t *cl, int c)     { (void)cl; (void)c; }
void ClientReliableWrite_Short (client_t *cl, int c)     { (void)cl; (void)c; }
void ClientReliableWrite_String(client_t *cl, char *s)   { (void)cl; (void)s; }
void ClientReliableWrite_SZ    (client_t *cl, void *data, int len) { (void)cl; (void)data; (void)len; }

/* ── server-side messaging / multicast / sound ─ no-op ──────────── */

void SV_FlushSignon(void)                                 {}
void SV_Multicast(vec3_t origin, int to)                  { (void)origin; (void)to; }
void SV_BroadcastPrintf(int level, char *fmt, ...)        { (void)level; (void)fmt; }
void SV_ClientPrintf(client_t *cl, int level, char *fmt, ...) { (void)cl; (void)level; (void)fmt; }
void SV_StartSound(edict_t *e, int ch, char *s, int v, float a) { (void)e; (void)ch; (void)s; (void)v; (void)a; }
int  SV_CalcPing(client_t *cl)                             { (void)cl; return 0; }
int  SV_ModelIndex(char *name)                             { (void)name; return 0; }

/* ── monster AI (called only if QC invokes movetogoal etc.) ─────── */

void     SV_MoveToGoal(void)                                          {}
qboolean SV_CheckBottom(edict_t *ent)                                 { (void)ent; return true; }
qboolean SV_movestep(edict_t *ent, vec3_t move, qboolean relink)      { (void)ent; (void)move; (void)relink; return true; }

/* ── world physics — return inert results ───────────────────────── */

void SV_LinkEdict  (edict_t *ent, qboolean touch_triggers) { (void)ent; (void)touch_triggers; }
void SV_UnlinkEdict(edict_t *ent)                          { (void)ent; }
int  SV_PointContents(vec3_t p)                            { (void)p; return CONTENTS_EMPTY; }

trace_t SV_Move(vec3_t start, vec3_t mins, vec3_t maxs, vec3_t end, int type, edict_t *passedict)
{
	trace_t t;
	(void)mins; (void)maxs; (void)type; (void)passedict;
	memset(&t, 0, sizeof(t));
	t.fraction = 1.0f;
	VectorCopy(end, t.endpos);
	return t;
}
