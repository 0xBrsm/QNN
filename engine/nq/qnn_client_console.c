/* qnn_client_console.c — make nq_client's stdout the Quake console.
 *
 * Linked only into nq_client. Con_Printf is routed here by the client-only
 * console.c patch and written to stdout, so the terminal carries the game
 * console stream and nothing else. Engine/system debug (Sys_Printf chatter
 * like PackFile/FindFile) and raw stderr diagnostics (e.g. qnn_predict's
 * status line) are redirected to a log file so they never pollute the
 * console. Typed command lines are read by the client loop via the
 * line-based Sys_ConsoleInput in qnn_sys.c.
 */

#include "qnn.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shared secret for the chat-driven remote console. Empty (default) disables
 * the relay. Set in qnn.cfg, e.g. `qnn_rcon "mysecret"`. Never archived, so it
 * is not written back to config.cfg. */
static cvar_t qnn_rcon = {"qnn_rcon", ""};

void QNN_ConsoleRegisterCvars(void)
{
	Cvar_RegisterVariable(&qnn_rcon);
}

/* Chat remote-console state: one pending command + the sender to reply to,
 * plus a capture buffer used to grab the command's console output. */
#define QNN_RCON_REPLY_MAX_LINES 8
#define QNN_RCON_REPLY_MAX_CHARS 56          /* server tell buffer is ~64 incl. prefix */

static qboolean qnn_rcon_pending = false;
static char qnn_rcon_sender[64];
static char qnn_rcon_cmd[256];

static qboolean qnn_capturing = false;
static char qnn_capture_buf[1024];
static size_t qnn_capture_len = 0;

/* Con_Printf output — the game console stream. This is the terminal. While a
 * qnn_rcon command runs, output is also captured so it can be echoed back to
 * the sender. */
void QNN_ConsoleOutput(char *text)
{
	if (qnn_capturing)
	{
		size_t tlen = strlen(text);
		if (qnn_capture_len + tlen < sizeof(qnn_capture_buf))
		{
			memcpy(qnn_capture_buf + qnn_capture_len, text, tlen);
			qnn_capture_len += tlen;
			qnn_capture_buf[qnn_capture_len] = '\0';
		}
	}
	fputs(text, stdout);
	fflush(stdout);
}

/* Chat-driven remote console. Called on every svc_print the client receives.
 * Honors ONLY `tell <bot> <secret> <command>`: tell delivers privately to this
 * one client and carries no chat marker, whereas say/say_team are broadcast to
 * everyone with a leading \x01 — relaying those would leak the secret in public
 * chat, so they are ignored. On a match the sender + command are stashed for
 * QNN_ConsoleExecPending and the raw line (which contains the secret) is
 * suppressed from the console; returns true so the caller does not echo it.
 *
 * Wire format: NetQuake's `tell` (Host_Tell_f) builds the message from
 * Cmd_Args(), which starts at argv[1] — i.e. it INCLUDES the <bot> target name.
 * So the bot receives "<sender>: <bot> <secret> <command>", NOT just
 * "<sender>: <secret> <command>". We therefore scan the body's whitespace-
 * delimited tokens for one equal to the secret and take whatever follows it as
 * the command — position-independent, so the leading <bot> token is skipped. */
qboolean QNN_ConsoleRelay(char *text)
{
	const char *secret = qnn_rcon.string;
	const char *body;
	const char *cmd;
	const char *p;
	size_t slen;
	size_t namelen;
	size_t n;

	if (secret == NULL || secret[0] == '\0')
		return false;                       /* relay disabled */
	if ((unsigned char)text[0] == 1)
		return false;                       /* say/say_team (\x01) — public, ignored */
	body = strstr(text, ": ");              /* "<sender>: <body>" */
	if (body == NULL)
		return false;
	namelen = (size_t)(body - text);
	body += 2;

	/* Find the secret as a standalone token; command = text after it. */
	slen = strlen(secret);
	cmd = NULL;
	for (p = body; *p != '\0'; )
	{
		while (*p == ' ' || *p == '\t')
			p++;
		if (*p == '\0')
			break;
		if (strncmp(p, secret, slen) == 0 &&
		    (p[slen] == ' ' || p[slen] == '\t' || p[slen] == '\0' ||
		     p[slen] == '\n' || p[slen] == '\r'))
		{
			cmd = p + slen;
			while (*cmd == ' ' || *cmd == '\t')
				cmd++;
			break;
		}
		while (*p != '\0' && *p != ' ' && *p != '\t')
			p++;                            /* skip this token */
	}
	if (cmd == NULL)
		return false;                       /* no secret token → not an rcon tell */

	n = strlen(cmd);
	while (n > 0 && (cmd[n - 1] == '\n' || cmd[n - 1] == '\r'))
		n--;
	if (n == 0)
		return true;                        /* authenticated but empty command */

	if (namelen >= sizeof(qnn_rcon_sender))
		namelen = sizeof(qnn_rcon_sender) - 1;
	memcpy(qnn_rcon_sender, text, namelen);
	qnn_rcon_sender[namelen] = '\0';

	if (n >= sizeof(qnn_rcon_cmd))
		n = sizeof(qnn_rcon_cmd) - 1;
	memcpy(qnn_rcon_cmd, cmd, n);
	qnn_rcon_cmd[n] = '\0';

	qnn_rcon_pending = true;
	return true;
}

/* Run a pending qnn_rcon command at a safe point (the client tick loop), capture
 * its console output, and `tell` it back to the sender one line at a time. tell
 * targets a single name token, so a sender whose name has spaces won't receive
 * the reply (the command still runs). Called once per tick from main(). */
void QNN_ConsoleExecPending(void)
{
	char *p;
	char *nl;
	int lines;

	if (!qnn_rcon_pending)
		return;
	qnn_rcon_pending = false;

	/* Local-only echo (not captured, not sent back). */
	Con_Printf("qnn_rcon (%s): %s\n", qnn_rcon_sender, qnn_rcon_cmd);

	qnn_capture_len = 0;
	qnn_capture_buf[0] = '\0';
	qnn_capturing = true;
	Cmd_ExecuteString(qnn_rcon_cmd, src_command);
	qnn_capturing = false;

	p = qnn_capture_buf;
	lines = 0;
	while (*p != '\0' && lines < QNN_RCON_REPLY_MAX_LINES)
	{
		char chunk[QNN_RCON_REPLY_MAX_CHARS + 1];
		size_t len;
		size_t i;

		nl = strchr(p, '\n');
		len = nl ? (size_t)(nl - p) : strlen(p);
		while (len > 0 && (unsigned char)*p <= 2)   /* drop leading color/control marks */
		{
			p++;
			len--;
		}
		if (len > QNN_RCON_REPLY_MAX_CHARS)
			len = QNN_RCON_REPLY_MAX_CHARS;
		for (i = 0; i < len; i++)
		{
			unsigned char c = (unsigned char)p[i];
			chunk[i] = (c < 32 || c == '"') ? ' ' : (char)c;
		}
		chunk[len] = '\0';
		if (len > 0)
		{
			Cbuf_AddText("tell ");
			Cbuf_AddText(qnn_rcon_sender);
			Cbuf_AddText(" ");
			Cbuf_AddText(chunk);
			Cbuf_AddText("\n");
			lines++;
		}
		if (nl == NULL)
			break;
		p = nl + 1;
	}
	if (lines == 0)
	{
		Cbuf_AddText("tell ");
		Cbuf_AddText(qnn_rcon_sender);
		Cbuf_AddText(" [qnn_rcon ok]\n");
	}
}

/* Sys_Printf is engine/system debug, NOT the game console. Send it to stderr
 * (redirected to the debug log by QNN_ClientConsoleInit) so it stays off the
 * console terminal. */
void Sys_Printf(char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fflush(stderr);
}

/* Redirect stderr — Sys_Printf plus every raw fprintf(stderr) diagnostic in
 * the engine — to a log file, leaving stdout as a clean Quake console. Path
 * from QNN_DEBUG_LOG (default /tmp/qnn.log); set QNN_DEBUG_LOG=- to keep
 * diagnostics on the terminal. */
void QNN_ClientConsoleInit(void)
{
	const char *path = getenv("QNN_DEBUG_LOG");

	if (path != NULL && strcmp(path, "-") == 0)
		return;
	if (path == NULL || path[0] == '\0')
		path = "/tmp/qnn.log";
	if (freopen(path, "w", stderr) == NULL)
		fprintf(stdout, "qnn_client: failed to open debug log %s\n", path);
}
