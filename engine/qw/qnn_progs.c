/*
 * qnn_progs.c — QC VM driver for sanitize-mode predicate evaluation.
 *
 * Loads the actual qwprogs.dat bytecode, allocates a single-player
 * edict pool, and provides per-tick entry points that populate the
 * player edict from a labeler snapshot and invoke specific QC
 * functions (W_ChangeWeapon, W_Attack, W_WeaponFrame, PlayerJump).
 * The QC bytecode itself answers "is this usercmd operative" — no
 * Python or C transcription of the QC logic.
 *
 * Analog of qnn_phys.c, which wraps pmove for physics-based move
 * label inference.  Same shape: link engine code into the worker,
 * inject state from our snapshot, call the engine routine, read
 * back the result.
 *
 * Side-effecting builtins (sound, setmodel, precache_*, sprint,
 * stuffcmd, spawn) are overridden to no-ops after PR_LoadProgs so
 * predicate evaluation doesn't depend on a real game world.  Field
 * writes the QC performs on `self` still happen — those are what we
 * read back.
 */

#include "qwsvdef.h"
#include <stddef.h>

extern builtin_t *pr_builtins;
extern int        pr_numbuiltins;

extern void Hunk_FreeToLowMark(int mark);
extern int  Hunk_LowMark(void);

/* pr_edict.c defines these but doesn't export them through a header. */
extern dfunction_t *ED_FindFunction(char *name);
extern ddef_t      *ED_FindField(char *name);
extern edict_t     *ED_Alloc(void);
extern void         ED_ClearEdict(edict_t *e);

static qboolean qnn_progs_inited = false;
static int      qnn_progs_hunk_mark = 0;
/* Persistent self.attack_finished across QNN_ProgsEvalAttack calls.
 * Reset on each QNN_ProgsInit (= once per demo).  Caller (labeler)
 * sets pr_global_struct->time per tick from native tick / tick_hz so
 * the QC's cooldown gate (`time < self.attack_finished`) decides
 * correctly without us having to advance a fake clock. */
static float    qnn_progs_attack_finished = 0.0f;
/* Field offset (in float-sized units, multiply by 4 for byte offset)
 * for self.attack_finished — declared by progs.qc but absent from
 * progdefs.h's entvars_t.  Resolved via ED_FindField at init time. */
static int      qnn_progs_attack_finished_ofs = -1;

/* Persistent FL_JUMPRELEASED state for QNN_ProgsEvalJump.  Initialized
 * true (jump released) at demo start.  PlayerJump in client.qc:732
 * checks this flag to debounce held-jump (anti-pogo): clears it on a
 * successful jump, the per-tick PlayerPostThink dispatch sets it again
 * whenever button2==0 (jump button released). */
static qboolean qnn_progs_jump_released = true;

/* Persistent self.weapon byte (1..8 axe..LG; 0 = uninitialized) across
 * QNN_ProgsEvalWeaponImpulseOperative calls.  Maintains the player's
 * actual current weapon as our QC-side prediction so the engine's
 * "already on target → no flip" gate in W_ChangeWeapon fires correctly
 * during the snapshot.weapon_id reliable-channel lag window.  Reset
 * to 0 in QNN_ProgsInit.  See QNN_ProgsEvalWeaponImpulseOperative for
 * the sync rule on server-forced switches. */
static int qnn_progs_self_weapon = 0;
/* Previous tick's snapshot.weapon_id; used to detect server-forced
 * switches (pickup auto-select, ammo-out W_BestWeapon, respawn) so
 * we know when to resync qnn_progs_self_weapon to the snapshot. */
static int qnn_progs_prev_snapshot_weapon = 0;

/* QC flag bits (from defs.qc) — duplicated here because progdefs.h
 * doesn't surface them. */
#define QNN_FL_ONGROUND      512
#define QNN_FL_WATERJUMP    2048
#define QNN_FL_JUMPRELEASED 4096

/* Read self.attack_finished from a freshly-injected edict. */
static float qnn_progs_get_attack_finished(edict_t *ed)
{
	if (qnn_progs_attack_finished_ofs < 0) return 0.0f;
	return ((eval_t *)((char *)&ed->v + qnn_progs_attack_finished_ofs * 4))->_float;
}

static void qnn_progs_set_attack_finished(edict_t *ed, float v)
{
	if (qnn_progs_attack_finished_ofs < 0) return;
	((eval_t *)((char *)&ed->v + qnn_progs_attack_finished_ofs * 4))->_float = v;
}

/* ── no-op builtins ─────────────────────────────────────────────── */

static void PF_QnnNoOp(void)
{
	/* QC builtin signature is void(void); return value, if any, lives
	 * in pr_global_struct->return_x.  Most stubs we override return
	 * void so leaving the return slot unchanged is fine. */
}

static void PF_QnnReturnZero(void)
{
	G_FLOAT(OFS_RETURN) = 0.0f;
	G_FLOAT(OFS_RETURN + 1) = 0.0f;
	G_FLOAT(OFS_RETURN + 2) = 0.0f;
}

static void PF_QnnReturnEmptyString(void)
{
	/* String 0 in pr_strings is a single NUL byte, the canonical
	 * empty string.  PR_LoadProgs guarantees pr_strings is set up. */
	G_INT(OFS_RETURN) = 0;
}

/* Spawn override: return a single throwaway edict instead of allocating
 * a fresh one each call.  W_FireRocket / W_FireGrenade / W_FireSpikes
 * each spawn a missile entity; without a stub we'd exhaust the edict
 * pool over a long demo. */
static edict_t *qnn_throwaway_edict = NULL;

static void PF_QnnSpawnThrowaway(void)
{
	/* Caller writes fields and then maybe passes the edict to setmodel /
	 * setorigin (which we also stub).  Zeroing is critical because the
	 * QC reads back values it just wrote (e.g. missile.velocity).  We
	 * don't bother freeing — the same slot is reused indefinitely. */
	if (qnn_throwaway_edict)
		memset(&qnn_throwaway_edict->v, 0, sizeof(qnn_throwaway_edict->v));
	if (qnn_throwaway_edict)
		qnn_throwaway_edict->free = false;
	G_INT(OFS_RETURN) = qnn_throwaway_edict
		? EDICT_TO_PROG(qnn_throwaway_edict) : 0;
}

/* Override the side-effecting builtins.  Indices verified from
 * vendor/quake/QW/server/pr_cmds.c pr_builtin[] table (1-indexed by
 * QuakeC convention; pr_builtins[N] == builtin #N).  Earlier versions
 * of this table had off-by-one errors that were harmless for
 * W_ChangeWeapon but would crash W_Attack's W_Fire* path. */
static void qnn_progs_override_builtins(void)
{
	if (pr_numbuiltins >  2) pr_builtins[ 2] = PF_QnnNoOp;             /* setorigin */
	if (pr_numbuiltins >  3) pr_builtins[ 3] = PF_QnnNoOp;             /* setmodel */
	if (pr_numbuiltins >  4) pr_builtins[ 4] = PF_QnnNoOp;             /* setsize */
	if (pr_numbuiltins >  8) pr_builtins[ 8] = PF_QnnNoOp;             /* sound */
	if (pr_numbuiltins > 14) pr_builtins[14] = PF_QnnSpawnThrowaway;   /* spawn */
	if (pr_numbuiltins > 15) pr_builtins[15] = PF_QnnNoOp;             /* remove */
	if (pr_numbuiltins > 16) pr_builtins[16] = PF_QnnReturnZero;       /* traceline */
	if (pr_numbuiltins > 19) pr_builtins[19] = PF_QnnNoOp;             /* precache_sound */
	if (pr_numbuiltins > 20) pr_builtins[20] = PF_QnnNoOp;             /* precache_model */
	if (pr_numbuiltins > 21) pr_builtins[21] = PF_QnnNoOp;             /* stuffcmd */
	if (pr_numbuiltins > 22) pr_builtins[22] = PF_QnnReturnZero;       /* findradius */
	if (pr_numbuiltins > 23) pr_builtins[23] = PF_QnnNoOp;             /* bprint */
	if (pr_numbuiltins > 24) pr_builtins[24] = PF_QnnNoOp;             /* sprint */
	if (pr_numbuiltins > 25) pr_builtins[25] = PF_QnnNoOp;             /* dprint */
	if (pr_numbuiltins > 26) pr_builtins[26] = PF_QnnReturnEmptyString;/* ftos */
	if (pr_numbuiltins > 27) pr_builtins[27] = PF_QnnReturnEmptyString;/* vtos */
	if (pr_numbuiltins > 35) pr_builtins[35] = PF_QnnNoOp;             /* lightstyle */
	if (pr_numbuiltins > 44) pr_builtins[44] = PF_QnnReturnZero;       /* aim */
	if (pr_numbuiltins > 45) pr_builtins[45] = PF_QnnReturnZero;       /* cvar */
	if (pr_numbuiltins > 52) pr_builtins[52] = PF_QnnNoOp;             /* WriteByte */
	if (pr_numbuiltins > 53) pr_builtins[53] = PF_QnnNoOp;             /* WriteChar */
	if (pr_numbuiltins > 54) pr_builtins[54] = PF_QnnNoOp;             /* WriteShort */
	if (pr_numbuiltins > 55) pr_builtins[55] = PF_QnnNoOp;             /* WriteLong */
	if (pr_numbuiltins > 56) pr_builtins[56] = PF_QnnNoOp;             /* WriteCoord */
	if (pr_numbuiltins > 57) pr_builtins[57] = PF_QnnNoOp;             /* WriteAngle */
	if (pr_numbuiltins > 58) pr_builtins[58] = PF_QnnNoOp;             /* WriteString */
	if (pr_numbuiltins > 59) pr_builtins[59] = PF_QnnNoOp;             /* WriteEntity */
	if (pr_numbuiltins > 72) pr_builtins[72] = PF_QnnNoOp;             /* cvar_set */
	if (pr_numbuiltins > 74) pr_builtins[74] = PF_QnnNoOp;             /* ambientsound */
	if (pr_numbuiltins > 82) pr_builtins[82] = PF_QnnNoOp;             /* multicast */
}

/* ── init ───────────────────────────────────────────────────────── */

qboolean QNN_ProgsInit(void)
{
	dfunction_t *fn;

	/* Always reload.  The QW client's hunk allocator gets reset
	 * between demos (QNN_ResetWorldLocal in qnn_collect_main.c calls
	 * down into Host_ClearMemory), which would dangle every pointer
	 * PR_LoadProgs set up.  Caller invokes this once per QNN_HandleCollect
	 * after the world reset, so we just re-allocate every time. */
	qnn_progs_inited = false;

	/* Save hunk mark so we can reset across episodes if needed. */
	qnn_progs_hunk_mark = Hunk_LowMark();

	memset(&sv, 0, sizeof(sv));
	sv.time  = 1.0;
	sv.state = ss_active;

	/* PR_LoadProgs reads qwprogs.dat via COM_LoadHunkFile, sets up
	 * pr_edict_size, pr_global_struct, pr_functions, etc. */
	PR_LoadProgs();

	/* Now that pr_edict_size is known, allocate the edict pool.
	 * Single player needs MAX_CLIENTS+1 slots reserved (we use
	 * edict 1 as our synthetic self).  Edict 2 is the throwaway
	 * returned by our spawn() override for projectiles etc. */
	sv.edicts     = (edict_t *)Hunk_AllocName(MAX_EDICTS * pr_edict_size, "qnn_edicts");
	sv.num_edicts = MAX_CLIENTS + 2;
	qnn_throwaway_edict = EDICT_NUM(MAX_CLIENTS + 1);
	memset(&qnn_throwaway_edict->v, 0, sizeof(qnn_throwaway_edict->v));
	qnn_throwaway_edict->free = false;

	/* Reset per-demo predicate state (attack_finished is engine state
	 * that persists across W_Attack invocations; reset at demo
	 * boundary so the new player starts with no buffered cooldown). */
	qnn_progs_attack_finished = 0.0f;
	/* Jump released: true at spawn so the very first +jump press can
	 * fire (FL_JUMPRELEASED is set by default after PutClientInServer). */
	qnn_progs_jump_released = true;
	/* self.weapon: 0 sentinel = uninitialized; first call seeds from
	 * snapshot.weapon_id once the demo brings the player up. */
	qnn_progs_self_weapon = 0;
	qnn_progs_prev_snapshot_weapon = 0;

	/* Resolve attack_finished field offset (custom QC field, not in
	 * progdefs.h entvars_t).  See vendor/quake/QW/progs/defs.qc:502. */
	{
		ddef_t *af_def = ED_FindField("attack_finished");
		qnn_progs_attack_finished_ofs = af_def ? (int)af_def->ofs : -1;
		if (qnn_progs_attack_finished_ofs < 0)
			fprintf(stderr, "qnn_progs: warning — attack_finished field not "
				"found in qwprogs.dat (predicate eval will misfire)\n");
	}

	/* Neutralize side-effecting builtins so QC functions can run
	 * without a real world / clients / network. */
	qnn_progs_override_builtins();

	fn = ED_FindFunction("ImpulseCommands");
	if (!fn)
	{
		fprintf(stderr,
			"qnn_progs: PR_LoadProgs loaded qwprogs.dat but ImpulseCommands not found "
			"(progs version mismatch?)\n");
		return false;
	}

	qnn_progs_inited = true;
	fprintf(stderr,
		"qnn_progs: pr_edict_size=%d  pr_numbuiltins=%d\n",
		pr_edict_size, pr_numbuiltins);
	fprintf(stderr,
		"qnn_progs: progs=%p  pr_functions=%p  pr_strings=%p  pr_global_struct=%p  fn=%p\n",
		(void *)progs, (void *)pr_functions, (void *)pr_strings,
		(void *)pr_global_struct, (void *)fn);
	fflush(stderr);
	if (progs)
		fprintf(stderr, "qnn_progs: progs->numfunctions=%d  progs->version=%d\n",
			progs->numfunctions, progs->version);
	if (fn && pr_functions)
		fprintf(stderr, "qnn_progs: fn idx = %ld\n", (long)(fn - pr_functions));
	fflush(stderr);
	return true;
}

/* ── smoke test ─────────────────────────────────────────────────── */

#define IT_AXE              4096
#define IT_SHOTGUN          1
#define IT_SUPER_SHOTGUN    2
#define IT_NAILGUN          4
#define IT_SUPER_NAILGUN    8
#define IT_GRENADE_LAUNCHER 16
#define IT_ROCKET_LAUNCHER  32
#define IT_LIGHTNING        64

static const char *qnn_w_to_name(int w_flag)
{
	switch (w_flag)
	{
	case IT_AXE: return "axe";
	case IT_SHOTGUN: return "sg";
	case IT_SUPER_SHOTGUN: return "ssg";
	case IT_NAILGUN: return "ng";
	case IT_SUPER_NAILGUN: return "sng";
	case IT_GRENADE_LAUNCHER: return "gl";
	case IT_ROCKET_LAUNCHER: return "rl";
	case IT_LIGHTNING: return "lg";
	default: return "?";
	}
}

/* ── per-tick predicate ─────────────────────────────────────────── */

/* impulse byte (1..8) -> IT_* item flag self.weapon uses inside QC. */
int QNN_ImpulseToItemFlag(int impulse)
{
	switch (impulse)
	{
	case 1: return IT_AXE;
	case 2: return IT_SHOTGUN;
	case 3: return IT_SUPER_SHOTGUN;
	case 4: return IT_NAILGUN;
	case 5: return IT_SUPER_NAILGUN;
	case 6: return IT_GRENADE_LAUNCHER;
	case 7: return IT_ROCKET_LAUNCHER;
	case 8: return IT_LIGHTNING;
	default: return 0;
	}
}

int QNN_ProgsEvalWeaponImpulse(
	int health, int items_owned,
	int ammo_shells, int ammo_nails, int ammo_rockets, int ammo_cells,
	int weapon_id, int impulse)
{
	edict_t *self_ed;
	dfunction_t *fn;

	if (!qnn_progs_inited)
		return 0;
	/* 1..8 direct select (W_ChangeWeapon), 10 cycle next, 12 cycle prev.
	 * 9 (cheat) and 11 (serverflags) are valid impulses but irrelevant
	 * to weapon transitions; let the VM handle them anyway — the worst
	 * case is self.weapon stays unchanged and we return prev. */
	if (impulse < 1 || impulse > 12)
		return 0;
	if (health <= 0)
		return 0;

	fn = ED_FindFunction("ImpulseCommands");
	if (!fn)
		return 0;

	self_ed = EDICT_NUM(1);
	memset(&self_ed->v, 0, sizeof(self_ed->v));
	self_ed->free = false;

	self_ed->v.health       = (float)health;
	self_ed->v.items        = (float)items_owned;
	self_ed->v.ammo_shells  = (float)ammo_shells;
	self_ed->v.ammo_nails   = (float)ammo_nails;
	self_ed->v.ammo_rockets = (float)ammo_rockets;
	self_ed->v.ammo_cells   = (float)ammo_cells;
	self_ed->v.weapon       = (float)QNN_ImpulseToItemFlag(weapon_id);
	self_ed->v.impulse      = (float)impulse;

	pr_global_struct->time = sv.time;
	pr_global_struct->self = EDICT_TO_PROG(self_ed);
	PR_ExecuteProgram(fn - pr_functions);

	/* Bump sv.time slightly so subsequent QC calls (e.g. another
	 * predicate eval at a later tick) don't accidentally land on
	 * exactly the same value, which can confuse rate-limited QC. */
	sv.time += 0.013;

	return (int)self_ed->v.weapon;
}


/* impulse byte 1..8 -> currentammo source pool. */
static int qnn_impulse_to_currentammo(int impulse,
	int shells, int nails, int rockets, int cells)
{
	switch (impulse)
	{
	case 1: return 0;          /* axe: no ammo */
	case 2: case 3: return shells;
	case 4: case 5: return nails;
	case 6: case 7: return rockets;
	case 8: return cells;
	default: return 0;
	}
}

int QNN_ProgsEvalAttack(
	int tick, int tick_hz,
	int health, int items_owned,
	int ammo_shells, int ammo_nails, int ammo_rockets, int ammo_cells,
	int weapon_id, int button0_pressed)
{
	edict_t *self_ed;
	dfunction_t *fn;
	float pre_attack_finished;
	float now_seconds;

	if (!qnn_progs_inited) return 0;
	if (tick_hz <= 0) return 0;
	if (health <= 0) return 0;
	if (weapon_id < 1 || weapon_id > 8) return 0;

	fn = ED_FindFunction("W_WeaponFrame");
	if (!fn) return 0;

	now_seconds = (float)tick / (float)tick_hz;

	self_ed = EDICT_NUM(1);
	memset(&self_ed->v, 0, sizeof(self_ed->v));
	self_ed->free = false;

	self_ed->v.health          = (float)health;
	self_ed->v.items           = (float)items_owned;
	self_ed->v.ammo_shells     = (float)ammo_shells;
	self_ed->v.ammo_nails      = (float)ammo_nails;
	self_ed->v.ammo_rockets    = (float)ammo_rockets;
	self_ed->v.ammo_cells      = (float)ammo_cells;
	self_ed->v.weapon          = (float)QNN_ImpulseToItemFlag(weapon_id);
	self_ed->v.currentammo     = (float)qnn_impulse_to_currentammo(
		weapon_id, ammo_shells, ammo_nails, ammo_rockets, ammo_cells);
	self_ed->v.button0         = button0_pressed ? 1.0f : 0.0f;
	qnn_progs_set_attack_finished(self_ed, qnn_progs_attack_finished);
	/* Origin and v_angle: leave zero — traceline, makevectors, etc.
	 * are either pure functions or overridden to return zero, so the
	 * concrete spatial state doesn't affect the operative bit. */

	pre_attack_finished = qnn_progs_attack_finished;
	pr_global_struct->time = now_seconds;
	pr_global_struct->self = EDICT_TO_PROG(self_ed);
	PR_ExecuteProgram(fn - pr_functions);

	/* W_Attack sets self.attack_finished = time + cd[weapon] iff it
	 * actually fired (passed the cooldown gate, had ammo, etc.).
	 * Compare against the value we injected to detect a flip. */
	{
		float post = qnn_progs_get_attack_finished(self_ed);
		if (post > pre_attack_finished)
		{
			qnn_progs_attack_finished = post;
			return 1;
		}
	}
	return 0;
}


/* Operative weapon-impulse predicate with persistent self.weapon state.
 *
 * Returns 1 iff this tick's impulse caused a real weapon flip (and 0
 * for already-on-target / unowned / no-ammo / invalid-impulse).  Unlike
 * QNN_ProgsEvalWeaponImpulse (stateless — re-injects self.weapon from
 * the caller's argument every call), this variant carries our internal
 * `qnn_progs_self_weapon` across calls, so once a press flips
 * self.weapon = SG, subsequent sticky-impulse re-evaluations correctly
 * see "already on SG → no flip" via the engine's own W_ChangeWeapon
 * gate.  This is what we need to avoid the snapshot.weapon_id lag-
 * window producing spurious repeat ops on a single press.
 *
 * Server-forced switches (pickup auto-select, ammo-out W_BestWeapon,
 * respawn) bypass cmd-impulse entirely — they manifest as
 * snapshot.weapon_id changing without an impulse press.  We detect
 * those by tracking prev-tick snapshot.weapon_id: when it changes to a
 * value that doesn't match our prediction, we trust the server and
 * resync.
 *
 * Sync runs unconditionally on every call so callers don't have to
 * gate on cmd_impulse != 0; the predicate-execution branch (which
 * actually runs QC ImpulseCommands) only fires for valid impulses
 * 1..12 on an alive player. */
int QNN_ProgsEvalWeaponImpulseOperative(
	int health, int items_owned,
	int ammo_shells, int ammo_nails, int ammo_rockets, int ammo_cells,
	int snapshot_weapon_id, int impulse)
{
	int prev_flag;
	int post_flag;
	int w;

	if (!qnn_progs_inited) return 0;

	/* Server-forced switch detection / first-time seed.  Runs every
	 * call regardless of impulse so the prev-tick snapshot baseline
	 * stays current without depending on press timing. */
	if (snapshot_weapon_id != qnn_progs_prev_snapshot_weapon
		&& snapshot_weapon_id != qnn_progs_self_weapon)
	{
		qnn_progs_self_weapon = snapshot_weapon_id;
	}
	qnn_progs_prev_snapshot_weapon = snapshot_weapon_id;
	if (qnn_progs_self_weapon == 0)
		qnn_progs_self_weapon = snapshot_weapon_id;

	/* Predicate gates: only run QC for valid impulse presses on an
	 * alive player.  Anything else is a no-op. */
	if (impulse < 1 || impulse > 12) return 0;
	if (health <= 0) return 0;

	prev_flag = QNN_ImpulseToItemFlag(qnn_progs_self_weapon);
	post_flag = QNN_ProgsEvalWeaponImpulse(
		health, items_owned,
		ammo_shells, ammo_nails, ammo_rockets, ammo_cells,
		qnn_progs_self_weapon, impulse);

	/* Update internal self.weapon from the QC result. */
	if (post_flag != 0)
	{
		for (w = 1; w <= 8; ++w)
		{
			if (QNN_ImpulseToItemFlag(w) == post_flag)
			{
				qnn_progs_self_weapon = w;
				break;
			}
		}
	}

	return (post_flag != 0 && post_flag != prev_flag) ? 1 : 0;
}


/* Native-frame remainder of the QC-tracked attack_finished cooldown.
 * Returns 0 when the engine would process the next fire immediately;
 * >0 = engine will reject (or queue) presses for that many frames.
 * Source of truth: qnn_progs_attack_finished, written by W_Attack on
 * every fire QC processes.  Replaces the labeler's earlier hand-coded
 * k_fire_cd_native[9] table. */
int QNN_ProgsGetAttackCdRemaining(int tick, int tick_hz)
{
	float now_seconds;
	float remaining_seconds;
	int   remaining_frames;

	if (!qnn_progs_inited) return 0;
	if (tick_hz <= 0)      return 0;

	now_seconds = (float)tick / (float)tick_hz;
	remaining_seconds = qnn_progs_attack_finished - now_seconds;
	if (remaining_seconds <= 0.0f) return 0;

	remaining_frames = (int)(remaining_seconds * (float)tick_hz + 0.5f);
	if (remaining_frames < 0)   return 0;
	if (remaining_frames > 255) return 255;
	return remaining_frames;
}


float QNN_ProgsGetAttackFinished(void)
{
	return qnn_progs_attack_finished;
}

void QNN_ProgsSetAttackFinished(float value)
{
	qnn_progs_attack_finished = value;
}


int QNN_ProgsEvalJump(
	int tick, int tick_hz,
	int health, int grounded, int waterlevel, int button2_pressed)
{
	edict_t *self_ed;
	dfunction_t *fn;
	int flags_pre;
	float now_seconds;

	if (!qnn_progs_inited) return 0;
	if (tick_hz <= 0) return 0;
	if (health <= 0) return 0;

	/* Mirror the QC dispatch (client.qc:924-929):
	 *   if (self.button2) PlayerJump();
	 *   else              self.flags |= FL_JUMPRELEASED;
	 * PlayerJump itself doesn't check self.button2 — it just clears
	 * FL_JUMPRELEASED if the other gates pass.  Calling it
	 * unconditionally lets the JR bit cycle every tick when button2=0
	 * (the very issue that produced 50%+ false-positive op_jump rates
	 * in an earlier iteration of this code). */
	if (!button2_pressed)
	{
		qnn_progs_jump_released = true;
		return 0;
	}

	fn = ED_FindFunction("PlayerJump");
	if (!fn) return 0;

	now_seconds = (float)tick / (float)tick_hz;

	self_ed = EDICT_NUM(1);
	memset(&self_ed->v, 0, sizeof(self_ed->v));
	self_ed->free = false;

	/* Build self.flags from snapshot signals + persistent JUMPRELEASED.
	 * PlayerJump returns early on: WATERJUMP set, waterlevel >= 2,
	 * !ONGROUND, !JUMPRELEASED.  We don't model WATERJUMP transitions
	 * (the bit's set briefly during a water-exit jump and cleared by
	 * the QC itself — for our once-per-tick predicate we treat it as
	 * never set; that biases slightly toward false-positive operative
	 * bits during the very rare WATERJUMP window). */
	flags_pre = 0;
	if (grounded) flags_pre |= QNN_FL_ONGROUND;
	if (qnn_progs_jump_released) flags_pre |= QNN_FL_JUMPRELEASED;

	self_ed->v.health     = (float)health;
	self_ed->v.flags      = (float)flags_pre;
	self_ed->v.waterlevel = (float)waterlevel;
	self_ed->v.button2    = 1.0f;

	pr_global_struct->time = now_seconds;
	pr_global_struct->self = EDICT_TO_PROG(self_ed);
	PR_ExecuteProgram(fn - pr_functions);

	/* PlayerJump's success branch (player.qc:760-764) clears
	 * FL_JUMPRELEASED only when it actually jumps.  Detect by checking
	 * if the bit went from set to cleared. */
	{
		int flags_post = (int)self_ed->v.flags;
		int jumped = (flags_pre & QNN_FL_JUMPRELEASED)
			&& !(flags_post & QNN_FL_JUMPRELEASED);

		if (jumped)
			qnn_progs_jump_released = false;
		/* else: button2 held without jump firing (gated by ground or
		 * water or anti-pogo) — preserve current jump_released. */

		return jumped ? 1 : 0;
	}
}


void QNN_ProgsSmokeTest(void)
{
	edict_t *self_ed;
	dfunction_t *fn;
	int before;

	if (!qnn_progs_inited)
	{
		fprintf(stderr, "qnn_progs SMOKE: skipped (VM not inited)\n");
		return;
	}

	fn = ED_FindFunction("W_ChangeWeapon");
	if (!fn) { fprintf(stderr, "qnn_progs SMOKE: W_ChangeWeapon missing\n"); return; }

	self_ed = EDICT_NUM(1);

	/* Case 1: own SG, holding axe, press impulse 2 (SG).  Should switch. */
	memset(&self_ed->v, 0, sizeof(self_ed->v));
	self_ed->free = false;
	self_ed->v.health = 100;
	self_ed->v.items  = IT_AXE | IT_SHOTGUN;
	self_ed->v.ammo_shells = 25;
	self_ed->v.weapon = IT_AXE;
	self_ed->v.impulse = 2;
	pr_global_struct->time = 1.0;
	pr_global_struct->self = EDICT_TO_PROG(self_ed);
	before = (int)self_ed->v.weapon;
	PR_ExecuteProgram(fn - pr_functions);
	fprintf(stderr, "qnn_progs SMOKE 1 (own SG, impulse=2): weapon %s -> %s (expected sg)\n",
		qnn_w_to_name(before), qnn_w_to_name((int)self_ed->v.weapon));

	/* Case 2: do NOT own LG, press impulse 8 (LG).  Should reject. */
	memset(&self_ed->v, 0, sizeof(self_ed->v));
	self_ed->free = false;
	self_ed->v.health = 100;
	self_ed->v.items  = IT_AXE | IT_SHOTGUN;
	self_ed->v.ammo_shells = 25;
	self_ed->v.weapon = IT_SHOTGUN;
	self_ed->v.impulse = 8;
	pr_global_struct->self = EDICT_TO_PROG(self_ed);
	before = (int)self_ed->v.weapon;
	PR_ExecuteProgram(fn - pr_functions);
	fprintf(stderr, "qnn_progs SMOKE 2 (no LG, impulse=8): weapon %s -> %s (expected unchanged sg)\n",
		qnn_w_to_name(before), qnn_w_to_name((int)self_ed->v.weapon));

	/* Case 3: own SSG, ammo_shells = 1 (need 2).  Should reject. */
	memset(&self_ed->v, 0, sizeof(self_ed->v));
	self_ed->free = false;
	self_ed->v.health = 100;
	self_ed->v.items  = IT_AXE | IT_SHOTGUN | IT_SUPER_SHOTGUN;
	self_ed->v.ammo_shells = 1;
	self_ed->v.weapon = IT_SHOTGUN;
	self_ed->v.impulse = 3;
	pr_global_struct->self = EDICT_TO_PROG(self_ed);
	before = (int)self_ed->v.weapon;
	PR_ExecuteProgram(fn - pr_functions);
	fprintf(stderr, "qnn_progs SMOKE 3 (own SSG, shells=1, impulse=3): "
		"weapon %s -> %s (expected unchanged sg)\n",
		qnn_w_to_name(before), qnn_w_to_name((int)self_ed->v.weapon));

	/* Cases 4..6 exercise the production QNN_ProgsEvalWeaponImpulse path
	 * (ImpulseCommands dispatch, int post-weapon return).  The cases
	 * above hit W_ChangeWeapon directly; these hit the dispatcher. */

	/* Case 4: direct-select via dispatcher.  Own SG, holding axe,
	 * impulse 2.  Eval should return IT_SHOTGUN. */
	{
		int post = QNN_ProgsEvalWeaponImpulse(
			/*health*/      100,
			/*items*/       IT_AXE | IT_SHOTGUN,
			/*shells*/      25, /*nails*/0, /*rockets*/0, /*cells*/0,
			/*weapon_id*/   1,  /* impulse-space: 1 = axe currently held */
			/*impulse*/     2);
		fprintf(stderr, "qnn_progs SMOKE 4 (Eval direct: axe + impulse=2): "
			"post=%s (expected sg)\n", qnn_w_to_name(post));
	}

	/* Case 5: cycle next via impulse 10.  Own AXE+SG+RL, holding SG,
	 * with shells + rockets.  Cycle ring SG→SSG→NG→SNG→GL→RL skips
	 * unowned weapons and lands on RL. */
	{
		int post = QNN_ProgsEvalWeaponImpulse(
			/*health*/      100,
			/*items*/       IT_AXE | IT_SHOTGUN | IT_ROCKET_LAUNCHER,
			/*shells*/      25, /*nails*/0, /*rockets*/5, /*cells*/0,
			/*weapon_id*/   2,  /* impulse-space: 2 = sg currently held */
			/*impulse*/     10);
		fprintf(stderr, "qnn_progs SMOKE 5 (Eval cycle next: sg + impulse=10, own RL): "
			"post=%s (expected rl)\n", qnn_w_to_name(post));
	}

	/* Case 6: cycle prev via impulse 12.  Cycling backward from SG goes
	 * directly to AXE (CycleWeaponReverseCommand at SG branch sets
	 * self.weapon = IT_AXE with no ammo check), so post = axe. */
	{
		int post = QNN_ProgsEvalWeaponImpulse(
			/*health*/      100,
			/*items*/       IT_AXE | IT_SHOTGUN | IT_ROCKET_LAUNCHER,
			/*shells*/      25, /*nails*/0, /*rockets*/5, /*cells*/0,
			/*weapon_id*/   2,
			/*impulse*/     12);
		fprintf(stderr, "qnn_progs SMOKE 6 (Eval cycle prev: sg + impulse=12): "
			"post=%s (expected axe)\n", qnn_w_to_name(post));
	}

	/* Cases 7..9 exercise QNN_ProgsEvalAttack.  Reset persistent
	 * attack_finished between cases by re-init isn't worth the wiring
	 * here; instead we set tick monotonically so the QC's cooldown
	 * gate sees us "past" any prior attack_finished. */

	/* Case 7: SG held, has shells, button0=1, no cooldown → should fire. */
	{
		int op = QNN_ProgsEvalAttack(
			/*tick*/        100,  /* time = 100/77 ≈ 1.3s, > any prior af */
			/*tick_hz*/     77,
			/*health*/      100,
			/*items*/       IT_AXE | IT_SHOTGUN,
			/*shells*/      25, /*nails*/0, /*rockets*/0, /*cells*/0,
			/*weapon_id*/   2,    /* SG */
			/*button0*/     1);
		fprintf(stderr, "qnn_progs SMOKE 7 (Attack SG, shells=25, button0=1): "
			"operative=%d (expected 1)\n", op);
	}

	/* Case 8: Same state, button0=0 → should NOT fire. */
	{
		int op = QNN_ProgsEvalAttack(
			/*tick*/        200,  /* well past case 7's attack_finished */
			/*tick_hz*/     77,
			/*health*/      100,
			/*items*/       IT_AXE | IT_SHOTGUN,
			/*shells*/      25, /*nails*/0, /*rockets*/0, /*cells*/0,
			/*weapon_id*/   2,
			/*button0*/     0);
		fprintf(stderr, "qnn_progs SMOKE 8 (Attack SG, button0=0): "
			"operative=%d (expected 0)\n", op);
	}

	/* Case 9: SG, no shells, button0=1 → should NOT fire. */
	{
		int op = QNN_ProgsEvalAttack(
			/*tick*/        300,
			/*tick_hz*/     77,
			/*health*/      100,
			/*items*/       IT_AXE | IT_SHOTGUN,
			/*shells*/      0, /*nails*/0, /*rockets*/0, /*cells*/0,
			/*weapon_id*/   2,
			/*button0*/     1);
		fprintf(stderr, "qnn_progs SMOKE 9 (Attack SG, shells=0, button0=1): "
			"operative=%d (expected 0)\n", op);
	}

	/* Cases 10..13 exercise QNN_ProgsEvalJump.  jump_released persists
	 * across calls, so we test the full state machine: press once
	 * (should jump), press again (should NOT, anti-pogo), release
	 * then press again (should jump). */

	/* Reset jump_released for a clean state machine test. */
	qnn_progs_jump_released = true;

	/* Case 10: grounded, dry land, button2=1 (just pressed jump) → jump. */
	{
		int op = QNN_ProgsEvalJump(
			/*tick*/  400, /*tick_hz*/ 77,
			/*health*/ 100,
			/*grounded*/ 1, /*waterlevel*/ 0,
			/*button2*/ 1);
		fprintf(stderr, "qnn_progs SMOKE 10 (Jump grounded, dry, button2=1): "
			"operative=%d (expected 1)\n", op);
	}

	/* Case 11: same state, button2=1 still held (no release between) → anti-pogo, no jump. */
	{
		int op = QNN_ProgsEvalJump(
			/*tick*/  401, /*tick_hz*/ 77,
			/*health*/ 100,
			/*grounded*/ 1, /*waterlevel*/ 0,
			/*button2*/ 1);
		fprintf(stderr, "qnn_progs SMOKE 11 (Jump button2 held, no release): "
			"operative=%d (expected 0)\n", op);
	}

	/* Case 12: button2 released (=0).  Should re-arm JUMPRELEASED but not jump. */
	{
		int op = QNN_ProgsEvalJump(
			/*tick*/  402, /*tick_hz*/ 77,
			/*health*/ 100,
			/*grounded*/ 1, /*waterlevel*/ 0,
			/*button2*/ 0);
		fprintf(stderr, "qnn_progs SMOKE 12 (Jump button2=0, releasing): "
			"operative=%d (expected 0)\n", op);
	}

	/* Case 13: pressed again after release → should jump. */
	{
		int op = QNN_ProgsEvalJump(
			/*tick*/  403, /*tick_hz*/ 77,
			/*health*/ 100,
			/*grounded*/ 1, /*waterlevel*/ 0,
			/*button2*/ 1);
		fprintf(stderr, "qnn_progs SMOKE 13 (Jump pressed after release): "
			"operative=%d (expected 1)\n", op);
	}

	/* Case 14: in air (grounded=0), button2=1 → no jump (FL_ONGROUND fail). */
	{
		qnn_progs_jump_released = true;
		int op = QNN_ProgsEvalJump(
			/*tick*/  500, /*tick_hz*/ 77,
			/*health*/ 100,
			/*grounded*/ 0, /*waterlevel*/ 0,
			/*button2*/ 1);
		fprintf(stderr, "qnn_progs SMOKE 14 (Jump in air, button2=1): "
			"operative=%d (expected 0)\n", op);
	}
}
