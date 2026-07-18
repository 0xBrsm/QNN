/*
 * qnn_progs_server.c — server-authoritative attack_finished for NQ builds
 * that host the game IN-PROCESS (the PPO/eval worker).
 *
 * Replaces qnn_progs_stub.c's zero for the cooldown READ path only. The
 * running progs.dat declares ``self.attack_finished`` (an ABSOLUTE QC time);
 * remaining cooldown = max(0, attack_finished - sv.time). The field offset
 * is resolved lazily via ED_FindField (it is QC-declared, not a fixed
 * entvars_t member — same dynamic-offset pattern as qw/qnn_progs.c).
 *
 * History: the stub returned 0 ("always ready") on every non-collect build,
 * so every live eval ran with a zeroed cooldown input while the training
 * corpus carried real values (found 2026-07-05 during the A1 live-collapse
 * hunt; prime suspect for the chronic live op-fire inflation vs the corpus
 * gate). This file closes that train/deploy input mismatch for the worker.
 * The pi LIVE CLIENT (remote server, no sv access) still needs a
 * client-side tracker (fire events + per-weapon cooldown table) — a
 * follow-up; it keeps the stub until then.
 *
 * The QC-VM predicate entry points (EvalAttack / Set) keep stub semantics —
 * they exist for the QW demo-collect path only.
 */

#include "quakedef.h"

static int qnn_progs_server_af_ofs = -2;   /* -2 unresolved, -1 absent */

extern edict_t *sv_player;
/* pr_edict.c exports this without a header prototype — an implicit C
 * declaration would truncate the returned pointer (SIGSEGV); declare it
 * explicitly, as qw/qnn_progs.c does. */
extern ddef_t *ED_FindField(char *name);

static float QNN_ServerAttackFinishedAbs(void)
{
	if (!sv.active || sv_player == NULL || progs == NULL)
		return 0.0f;
	if (qnn_progs_server_af_ofs == -2) {
		ddef_t *def = ED_FindField((char *)"attack_finished");
		qnn_progs_server_af_ofs = def ? (int)def->ofs : -1;
	}
	if (qnn_progs_server_af_ofs < 0)
		return 0.0f;
	return ((eval_t *)((char *)&sv_player->v
		+ qnn_progs_server_af_ofs * 4))->_float;
}

float QNN_ProgsGetAttackCdRemainingSec(float now_seconds)
{
	float af = QNN_ServerAttackFinishedAbs();
	float remaining;
	(void)now_seconds;                  /* server clock is authoritative */
	if (af <= 0.0f)
		return 0.0f;
	remaining = af - (float)sv.time;
	return remaining > 0.0f ? remaining : 0.0f;
}

float QNN_ProgsGetAttackFinished(void)
{
	return QNN_ServerAttackFinishedAbs();
}

void QNN_ProgsSetAttackFinished(float value)
{
	(void)value;                        /* collect-only path; not used here */
}

int QNN_ProgsEvalAttack(
	float now_seconds,
	int health, int items_owned,
	int ammo_shells, int ammo_nails, int ammo_rockets, int ammo_cells,
	int weapon_id, int button0_pressed)
{
	(void)now_seconds; (void)health; (void)items_owned;
	(void)ammo_shells; (void)ammo_nails; (void)ammo_rockets; (void)ammo_cells;
	(void)weapon_id; (void)button0_pressed;
	return 0;                           /* QW demo-collect only */
}
