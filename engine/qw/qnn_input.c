/*
 * qnn_input.c (qw) — Null input driver for the QW demo worker.
 *
 * During demo playback, user input comes from the demo file's dem_cmd
 * messages, not from real input devices.  This file provides the
 * IN_* stubs that the QW engine expects.
 *
 * Unlike the NQ input (qnn_input.c) which drives the engine in
 * interactive mode, this is purely passive — it just satisfies the
 * linker for functions called by cl_input.c and cl_main.c.
 */

#include "qnn.h"

qnn_action_t qnn_pending_action;

void IN_Init(void)
{
}

void IN_Shutdown(void)
{
}

void IN_Commands(void)
{
}

void IN_Move(usercmd_t *cmd)
{
	/* During demo playback, usercmd comes from the demo file.
	 * No need to inject anything here. */
	(void)cmd;
}

void IN_ModeChanged(void)
{
}
