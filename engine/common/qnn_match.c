/*
 * qnn_match.c — Match start/end detection from svc_print text.
 *
 * Called from a patched Con_Printf in the engine for every svc_print
 * line. Scans for mod-generated tournament strings ("match has begun",
 * "Match Over", etc.) and updates qnn_match_state used by collect
 * main loops to gate emission to in-match frames only.
 *
 * Shared by nq/qnn_collect_main.c and qw/qnn_collect_main.c.
 *
 * Engine-specific log tag: callers set qnn_match_log_tag to "[demo]"
 * (NQ) or "[qw-demo]" (QW) so stderr lines are distinguishable.
 */

#include "qnn.h"

#include <stdio.h>
#include <string.h>

int qnn_match_state = 0;	/* 0=pre, 1=in match, 2=post match */
const char *qnn_match_log_tag = "[demo]";

/* Strip Quake's high-bit colored characters to plain ASCII so pattern
 * matching works regardless of whether the server sent bronze/gold text.
 * QW mods commonly send match text with bit 7 set. */
#define QNN_MATCH_BUF 512

static void QNN_StripHighBit(char *out, const char *in, int max)
{
	int i;
	for (i = 0; i < max - 1 && in[i]; i++)
		out[i] = in[i] & 0x7f;
	out[i] = '\0';
}

void QNN_MatchCheckPrint(const char *text)
{
	char clean[QNN_MATCH_BUF];
	QNN_StripHighBit(clean, text, QNN_MATCH_BUF);

	if (qnn_match_state == 0)
	{
		if (strstr(clean, "match has begun") ||
		    strstr(clean, "Match has begun") ||
		    strstr(clean, "Match Started") ||
		    strstr(clean, "match started") ||
		    strstr(clean, "match has started") ||
		    strstr(clean, "Game Is Starting In 1 Second") ||
		    strstr(clean, "Match is 1v1") || strstr(clean, "Match is 2v2") ||
		    strstr(clean, "Match is 3v3") || strstr(clean, "Match is 4v4") ||
		    strstr(clean, "Match is 5v5") || strstr(clean, "Match is 6v6") ||
		    strstr(clean, "Match is 7v7") || strstr(clean, "Match is 8v8") ||
		    strstr(clean, "Game Is Starting In 1 Sec"))
		{
			qnn_match_state = 1;
			fprintf(stderr, "%s match start detected: %.*s\n",
				qnn_match_log_tag,
				(int)strcspn(clean, "\n"), clean);
		}
	}
	else if (qnn_match_state == 1)
	{
		if (strstr(clean, "match is over") ||
		    strstr(clean, "Match is over") ||
		    strstr(clean, "The match is over") ||
		    strstr(clean, "Match Over") ||
		    strstr(clean, "Game Over") ||
		    strstr(clean, "has WON over") ||
		    strstr(clean, "has won over"))
		{
			qnn_match_state = 2;
			fprintf(stderr, "%s match end detected: %.*s\n",
				qnn_match_log_tag,
				(int)strcspn(clean, "\n"), clean);
		}
	}
}
