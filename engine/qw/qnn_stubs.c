/*
 * qnn_stubs.c (qw) — Linker stubs for symbols referenced by QW client
 * code but not needed for headless demo playback.
 */

#include "quakedef.h"
#include <strings.h>

/* VID_LockBuffer / VID_UnlockBuffer — rendering lock, no-op for headless */
void VID_LockBuffer(void) {}
void VID_UnlockBuffer(void) {}

/* _windowed_mouse — cvar referenced by menu.c */
cvar_t _windowed_mouse = {"_windowed_mouse", "0"};

/* stricmp — MSVC name for strcasecmp */
int stricmp(const char *s1, const char *s2)
{
	return strcasecmp(s1, s2);
}

/* CDAudio_Pause — referenced by cl_parse.c on pause */
void CDAudio_Pause(void) {}

/* NQ server compat globals — only referenced by trace_t */
edict_t *sv_player;
server_t sv;
globalvars_t _pr_global_struct;
globalvars_t *pr_global_struct = &_pr_global_struct;
char *pr_strings = "";
