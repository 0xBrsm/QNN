/*
 * qnn_qwd_collect.c — QWD usercmd-truth path for the QW demo worker.
 *
 * QWD demos carry dem_cmd messages containing usercmd_t with the
 * recorder's exact inputs.  This module walks the cmd window deposited
 * into cl.frames[] during the current Host_Frame and aggregates it into
 * a single emit-rate action label.  At native (e.g. 77 Hz) > emit
 * (20 Hz), the demo player drains 3-4 dem_cmds per Host_Frame; reading
 * only the latest one would lose sub-50ms button events.  This is the
 * dem_cmd analogue of kbutton_t's impulse-flag aggregation in the
 * live-play CL_BaseMove path:
 *   - attack/jump: OR across cmds (any press in the window counts)
 *   - forward/side move: average (each cmd is integrated over native_dt;
 *     averaging N approximates a single 50ms-integrated cmd)
 *   - upmove (swim down): min across cmds (most-negative captures any
 *     swim-down press regardless of when in the window it occurred)
 *   - attack: action.attack is 0 except on an effective QC W_Attack frame,
 *     where it names the weapon actually fired (categorical supervision).
 *     Each cmd's select impulse + attack button is replayed through
 *     QNN_ProgsStepWeaponFrame to recover that engine outcome.
 *
 * On slow demos (native < emit) the window may be empty for some
 * Host_Frames; we fall back to the latest cmd in cl.frames[] (which
 * is the held state from the previous emit).
 *
 * QWD emission deliberately does NOT run QNN_SnapMove:
 *   - The air branch of the snap zeroes forward, on the rule that
 *     forward-press in air is unrecoverable from observed motion.
 *     That rule is for the MVD physics-reconstruction path; it has
 *     no bearing on QWD where cmd.forwardmove is the player's actual
 *     keystate.  Snapping over QWD silently erased every airborne
 *     +forward press (i.e. all of bunnyhopping/strafe-jumping).
 *   - The ground/air branch also zeroes downward upmove and other
 *     "no engine effect" intents.  Same story: those are real key
 *     presses, just engine-inert; the model should learn to predict
 *     what the player did.
 * The Python collect-side compaction thresholds at 0.1 to produce the
 * 6-button bitfield, so the continuous cmd values land in the same
 * binary representation as a snapped label would have for any nontrivial
 * press — minus the air-forward erasure.
 *
 * look still comes from state-derived deltas (QNN_FillLook in
 * qnn_collect_helpers.c). MVD path keeps the snap (see
 * qnn_mvd_collect.c QNN_MvdInferEmitMove).
 */

#include "qnn_qwd_collect.h"
#include "qnn_collect_helpers.h"
#include "qnn_pmove_hooks.h"

#include <string.h>

/* QW pmove interface — same extern pattern as qnn_phys.c. */
extern playermove_t pmove;
extern void PlayerMove(void);
extern int onground;
extern int waterlevel;

/* Module-private QC weapon predicate and pmove replay state. */
static struct
{
	int		weapon_advanced_tick;	/* qnn_runtime.tick the QC weapon
						 * predicate last advanced for; guards
						 * the >1 QwdBuildActionLabel calls per
						 * tick on the force_mvd_emit path so the
						 * stateful predicate replays the cmd
						 * window exactly once per emit tick. */
	int		pmove_oldbuttons;        /* pmove.oldbuttons carried across pmove-jump driver calls */
	qboolean	pmove_oldbuttons_inited;
	/* Pre-tick origin/velocity snapshot for the pmove jump driver.
	 * cl.simorg/cl.simvel are updated by CL_PredictMove during
	 * Host_Frame and reflect the engine's predicted state AFTER the
	 * current tick's cmds — i.e., post-tick state.  For a faithful
	 * cmd-replay we need PRE-tick state, so we save cl.simorg/simvel
	 * at the end of each call and feed the saved value as the seed
	 * on the next call.  Same idea as prev_origin/prev_snap_velocity
	 * but reads from cl.simorg/simvel directly (no broadcast lag
	 * from ps->origin, no MVD-faithful zeroing of player_velocity). */
	vec3_t		pmove_prev_simorg;
	vec3_t		pmove_prev_simvel;
	qboolean	pmove_prev_sim_inited;
	/* op_attack captured from the per-cmd predicate call inside
	 * QNN_QwdExtractAction's cmd-window loop. Read by
	 * QwdPackInputMask to set input_mask bit 0 (attack) without
	 * re-advancing the QC VM. Only valid when set by the same-tick
	 * QwdExtractAction call; QwdBuildActionLabel inherits the value
	 * implicitly via the QwdExtractAction call that happens inside
	 * it. */
	int		last_op_attack;
	int		last_attack_weapon;	/* weapon actually advanced by QC W_Attack */
	/* op_impulse captured from the same per-cmd QC weapon advance:
	 * 1 iff an edge-triggered weapon-select flipped self.weapon this
	 * tick.  Read by QwdPackOpInput for the op_input impulse bit (no
	 * second stateful predicate). */
	int		last_op_impulse;
	/* This tick's weapon-select impulse (1-8 direct, 10/12 cycle, 0 =
	 * none), stashed for the QwdBuildActionLabel weapon-label call. */
	int		last_weapon_impulse;
	/* Per-tick aggregated press signals captured from the cmd window
	 * so QwdPackInputMask can read them without re-iterating. Jump
	 * and upmove are kept distinct here (vs collapsed into the
	 * legacy op_input ud bit). */
	int		last_jump_press_any;	/* any cmd had buttons & BUTTON_JUMP */
	int		last_upmove_pos_any;	/* any cmd had upmove > 0 (swim up) */
	/* QC predicate state at the START of QwdExtractAction's per-cmd
	 * loop — i.e. BEFORE this tick's attack (if any) advances cooldown.
	 * QwdPackInputMask reads this so its synthetic feasibility check
	 * answers "could a press attack AT THE BEGINNING of this tick" not
	 * "could a press attack NEXT tick after this tick's attack already
	 * advanced the cooldown" (the previous off-by-one — input_mask
	 * was capturing post-loop state, anti-correlated with demo press
	 * on engine-attacked ticks). */
	float		pre_loop_attack_finished;
	/* Previous cmd's raw impulse byte, for edge-triggering the weapon
	 * select.  The replayed usercmd impulse is NOT cleared between cmds
	 * (real QW clears self.impulse after the server consumes it once),
	 * so a held value re-triggers W_ChangeWeapon every cmd — a stale
	 * impulse 1 (select-axe, left over from axe-running) drags
	 * self.weapon back to axe every frame and overrides the
	 * server-correct snapshot->weapon_id.  We feed the impulse to the QC
	 * advance only on its rising edge (when it changes). -1 = unseen. */
	int		prev_cmd_impulse;
	/* Edge-detect for the env-gated attack-edge dump (diagnostic only). */
	int		prev_emit_op_attack;
} qwd_state;

/* Diagnostic: env-gated (QNN_QWD_ATTACK_EDGE_DUMP=<path>) JSONL dump of every
 * operative attack rising edge, capturing native_time (same clock as demo
 * svc_sound), the attack label, the lagged obs weapon, and the QC-advanced
 * weapon.  Read-only — no effect on emitted labels.  Used to measure the gap
 * between the op-attack edge and the attack sound that names the true weapon. */
static void QNN_QwdDumpAttackEdge(const qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	const char *path;
	FILE *out;
	int pressed, feasible;

	if (action == NULL || snapshot == NULL)
		return;
	if (snapshot->health <= 0)        /* alive frames only — no attack while dead */
		return;
	path = getenv("QNN_QWD_ATTACK_EDGE_DUMP");
	if (path == NULL || path[0] == '\0')
		return;
	/* Per-alive-frame attack state, for aligning attack sounds against the
	 * emitted op-attack: `pressed` = demo button0 this tick (act_move bit 0),
	 * `feasible` = input_mask bit 0 (QNN_EvalAttackFeasible). op_attack edge =
	 * rising edge of (pressed & feasible). An attack sound near a (pressed=1,
	 * feasible=0) frame is a feasibility false-negative; near no pressed frame
	 * is a press-detection miss; near a pressed&feasible frame already 1 is an
	 * edge-merge. */
	pressed  = (action->move & 0x01) ? 1 : 0;
	feasible = (action->input_mask & 0x01) ? 1 : 0;
	out = fopen(path, "a");
	if (out == NULL)
		return;
	fprintf(out, "{\"demo_path\":");
	QNN_WriteJsonString(out, qnn_runtime.demo_path);
	fprintf(out,
		",\"t\":%.6f,\"pressed\":%d,\"feasible\":%d,\"qc_weapon\":%d}\n",
		QNN_RuntimeNowSeconds(), pressed, feasible, QNN_ProgsGetSelfWeapon());
	fclose(out);
}

void QNN_QwdCollectReset(void)
{
	memset(&qwd_state, 0, sizeof(qwd_state));
	/* -1 so the guard fires on the first emit tick even if its
	 * qnn_runtime.tick is 0 (memset would otherwise alias tick 0). */
	qwd_state.weapon_advanced_tick = -1;
	qwd_state.prev_cmd_impulse = -1;   /* force the first impulse to edge-trigger */
}

/* Iterate the QWD cmd window and evaluate the per-cmd operative
 * predicates (QC W_WeaponFrame for attack, PlayerJump for jump) while
 * also aggregating the raw usercmd bytes into the per-tick cmd block
 * (mean for fmove/smove/umove, OR for buttons, last-non-zero for
 * impulse).  Single pass.
 *
 * Each cmd's button state is fed individually to the QC predicates so
 * the VM's persistent state (attack_finished, jump_released) advances
 * at cmd granularity — crucial for autohop and similar press/release
 * patterns that the per-native-tick OR'd signal collapses.
 *
 * cmd-block outputs (one per tick):
 *   fmove, smove   = mean across cmds, in raw QW units (int)
 *   umove          = jump_any ? QNN_SV_JUMP_SPEED : last_negative_or_zero
 *                    (matches the engine's actual integration:
 *                    button2 OR upmove>0 both trigger jump, so we
 *                    represent the canonical jump value; pure swim
 *                    presses preserve most-negative)
 *   buttons        = OR'd across cmds
 *   impulse        = last non-zero impulse byte (matches sv_user.c:3575
 *                    which only overwrites self.impulse on non-zero
 *                    cmds, so earlier same-window impulses get
 *                    overwritten — engine processes the last) */
void QNN_QwdEvalOperativePerCmd(
	const qnn_snapshot_t *snapshot,
	int *out_op_attack,
	int *out_fmove,
	int *out_smove,
	int *out_umove,
	int *out_buttons,
	int *out_impulse)
{
	int window_start;
	int window_end;
	int n;
	int i;
	int op_attack = 0;
	int current_weapon = snapshot->weapon_id;
	long forward_sum = 0;
	long side_sum = 0;
	int last_umove_neg = 0;
	int buttons_or = 0;
	int last_nonzero_impulse = 0;
	int jump_any = 0;

	*out_op_attack = 0;
	*out_fmove = 0;
	*out_smove = 0;
	*out_umove = 0;
	*out_buttons = 0;
	*out_impulse = 0;

	window_end = cls.netchan.outgoing_sequence;
	if (window_end <= 0)
		return;

	window_start = qnn_runtime.cmd_seq_window_start;
	n = window_end - window_start;
	if (n <= 0)
	{
		window_start = window_end - 1;
		n = 1;
	}
	if (n > UPDATE_BACKUP)
		n = UPDATE_BACKUP;

	for (i = 0; i < n; ++i)
	{
		int seq = (window_start + i) & UPDATE_MASK;
		usercmd_t *cmd = &cl.frames[seq].cmd;
		int button0 = (cmd->buttons & 1) ? 1 : 0;
		int imp;

		/* Aggregate raw cmd bytes. */
		forward_sum += (long)cmd->forwardmove;
		side_sum    += (long)cmd->sidemove;
		if (cmd->upmove < last_umove_neg)
			last_umove_neg = cmd->upmove;
		if ((cmd->buttons & 2) || cmd->upmove > 0)
			jump_any = 1;
		buttons_or |= (int)cmd->buttons;
		imp = (int)cmd->impulse;
		if (imp != 0)
			last_nonzero_impulse = imp;

		/* Attack predicate (only meaningful if a weapon is held). */
		if (QNN_WeaponIsValid(current_weapon))
		{
			int attack_op = QNN_ProgsEvalAttack(
				QNN_RuntimeNowSeconds(),
				snapshot->health, snapshot->items_owned,
				snapshot->ammo_shells, snapshot->ammo_nails,
				snapshot->ammo_rockets, snapshot->ammo_cells,
				current_weapon, button0);
			if (attack_op)
				op_attack = 1;
		}
	}

	*out_op_attack = op_attack;
	*out_fmove   = (int)(forward_sum / n);
	*out_smove   = (int)(side_sum / n);
	*out_umove   = jump_any ? QNN_SV_JUMP_SPEED : last_umove_neg;
	*out_buttons = buttons_or & 0xFF;
	*out_impulse = last_nonzero_impulse & 0xFF;
}


/* Per-cmd pmove driver for jump operativeness.  Replaces the QC
 * PlayerJump predicate's K-gated workaround with direct observation
 * of pmove's JumpButton firings — frame-exact press attribution at
 * cmd granularity, no snapshot.grounded lag artifact.
 *
 * Per-tick state across calls:
 *   pmove.oldbuttons must persist so anti-pogo (don't double-fire
 *   jump while +jump is held) works correctly.  We carry it across
 *   calls via a static; QNN_PmoveSave/Restore preserves the *outer*
 *   pmove state but we re-inject our own oldbuttons each tick.
 *
 * World-hull only physents.  Same setup as the QW MVD inference
 * path (which doesn't call QNN_PhysSetupMovers/Players either) —
 * sufficient for ground-jump detection on static map geometry.
 * Lifts/movers won't carry a jump in this approximation, matching
 * the existing MVD inference precision/recall envelope.
 *
 * Returns 1 iff any cmd in the current native tick's cmd window
 * triggered pmove's ground-jump success branch (270 z-impulse +
 * onground 0→-1). */
int QNN_QwdEvalPmoveJump(const qnn_snapshot_t *snapshot, int synth_button2)
{
	int window_start;
	int window_end;
	int n;
	int i;
	int op_jump = 0;
	qnn_pmove_save_t save;
	/* Synthetic-press carry-state save/restore.  Pure-feasibility mode
	 * forces button2=1 every cmd and so contaminates pmove_oldbuttons
	 * (anti-pogo state) and pmove_prev_simorg/simvel (next-tick seed).
	 * Snapshot the four fields BEFORE the synthetic loop, restore
	 * after — leaves real-jump label path's state untouched. */
	int saved_pmove_oldbuttons = qwd_state.pmove_oldbuttons;
	qboolean saved_pmove_oldbuttons_inited = qwd_state.pmove_oldbuttons_inited;
	vec3_t saved_pmove_prev_simorg;
	vec3_t saved_pmove_prev_simvel;
	qboolean saved_pmove_prev_sim_inited = qwd_state.pmove_prev_sim_inited;
	VectorCopy(qwd_state.pmove_prev_simorg, saved_pmove_prev_simorg);
	VectorCopy(qwd_state.pmove_prev_simvel, saved_pmove_prev_simvel);

	if (!snapshot)
		return 0;
	/* Defensive: skip if no world geometry to trace against.  PhysInit
	 * sets physents[0].model = cl.worldmodel at demo start, but if
	 * cl.worldmodel happens to be NULL or got freed (e.g., between
	 * demos), PM_PointContents would null-deref inside PlayerMove. */
	if (!cl.worldmodel)
		return 0;
	/* Mid-demo map-change transition guard.  After a fresh svc_serverdata
	 * repoints cl.worldmodel, the mover/player refs (built once at demo
	 * start) still hold the previous map's model_precache indices, which
	 * on the new map resolve to non-bmodel (.mdl) slots — tracing against
	 * a hull-less model faults inside PM_RecursiveHullCheck.  Skip until
	 * the main loop rebuilds the refs (QNN_BuildAllRefs) for the new
	 * worldmodel and updates refs_worldmodel.  op_jump feasibility just
	 * defaults to 0 for the few transition frames, which are noise. */
	if ((void *)cl.worldmodel != qnn_runtime.refs_worldmodel)
		return 0;
	/* Need previous-tick state to seed pmove correctly — snapshot is
	 * post-tick (server already ran pmove over this window's cmds), so
	 * re-iterating from snapshot.origin starts 1-2 ticks past the jump
	 * moment and PM_CatagorizePosition spuriously rejects valid
	 * presses as airborne.  prev_* fields hold the PRE-tick state
	 * (captured by QNN_SavePrev at the end of the previous tick). */
	if (!qnn_runtime.has_prev)
		return 0;

	window_end = cls.netchan.outgoing_sequence;
	if (window_end <= 0)
		return 0;
	window_start = qnn_runtime.cmd_seq_window_start;
	n = window_end - window_start;
	if (n <= 0)
	{
		window_start = window_end - 1;
		n = 1;
	}
	if (n > UPDATE_BACKUP)
		n = UPDATE_BACKUP;

	/* Save existing pmove state — qnn_phys's MVD-inference path
	 * mutates the same globals if running on this tick. */
	QNN_PmoveSave(&save);

	/* Seed pmove from the engine's PRE-tick predicted state captured
	 * at the end of the previous tick (cl.simorg / cl.simvel).
	 * Compared to snapshot-routed prev_origin/prev_velocity:
	 *
	 *   - prev_origin = snapshot->player_origin = ps->origin in labeler
	 *     mode (server-broadcast playerstate); lags by RTT/2 +
	 *     jitter, so over long bhop chains our pmove trajectory
	 *     diverges from the server's and misses sub-tick ground
	 *     touches that PM_CatagorizePosition detected server-side.
	 *
	 *   - cl.simorg / cl.simvel are updated by CL_PredictMove every
	 *     Host_Frame using the demo's own cmds — for QWD playback
	 *     the engine replays the exact recorded cmd stream, so
	 *     cl.simorg == server's pmove output, no broadcast lag.
	 *
	 * Fall back to snapshot if has_prev_sim is false (first tick of
	 * a demo). */
	if (qwd_state.pmove_prev_sim_inited)
	{
		VectorCopy(qwd_state.pmove_prev_simorg, pmove.origin);
		VectorCopy(qwd_state.pmove_prev_simvel, pmove.velocity);
	}
	else
	{
		VectorCopy(qnn_runtime.prev_origin,   pmove.origin);
		VectorCopy(qnn_runtime.prev_velocity, pmove.velocity);
	}
	VectorCopy(snapshot->player_view_angles, pmove.angles);
	onground         = qnn_runtime.prev_grounded ? 0 : -1;
	waterlevel       = snapshot->waterlevel;
	pmove.dead       = (snapshot->health <= 0) ? 1 : 0;
	pmove.spectator  = 0;
	pmove.waterjumptime = 0;
	pmove.oldbuttons = qwd_state.pmove_oldbuttons_inited
		? qwd_state.pmove_oldbuttons : 0;
	qwd_state.pmove_oldbuttons_inited = true;
	/* Populate physents for PM_CatagorizePosition's ground trace.
	 * World hull at physents[0]; movers + other players follow.
	 * Without movers, bhop chains that briefly land on lifts /
	 * platforms get missed (PM_CatagorizePosition traces 1 unit down
	 * and finds nothing → onground=-1 → JumpButton rejects valid
	 * presses).  Other-player physents matter when players briefly
	 * stand on each other's hulls (rare but real on tight maps). */
	memset(&pmove.physents[0], 0, sizeof(pmove.physents[0]));
	pmove.physents[0].model = cl.worldmodel;
	pmove.numphysent = 1;
	{
		qnn_mover_state_t movers[QNN_MAX_PHYS_MOVERS];
		vec3_t player_origins[QNN_MAX_PHYS_PLAYERS];
		int m, n_m, p, n_p;

		n_m = qnn_runtime.mover_count;
		if (n_m > QNN_MAX_PHYS_MOVERS) n_m = QNN_MAX_PHYS_MOVERS;
		for (m = 0; m < n_m; ++m)
		{
			entity_t *me = &cl_entities[qnn_runtime.mover_entity_nums[m]];
			VectorCopy(me->origin, movers[m].origin);
			movers[m].model_index = qnn_runtime.mover_model_indices[m];
			movers[m].velocity[0] = movers[m].velocity[1] = movers[m].velocity[2] = 0.0f;
		}
		if (n_m > 0)
			QNN_PhysSetupMovers(movers, n_m);

		n_p = qnn_runtime.player_count;
		if (n_p > QNN_MAX_PHYS_PLAYERS) n_p = QNN_MAX_PHYS_PLAYERS;
		for (p = 0; p < n_p; ++p)
		{
			int entnum = qnn_runtime.player_entity_nums[p];
			VectorCopy(cl_entities[entnum].origin, player_origins[p]);
		}
		if (n_p > 0)
			QNN_PhysSetupPlayers(player_origins, n_p);
	}

	for (i = 0; i < n; ++i)
	{
		int seq = (window_start + i) & UPDATE_MASK;
		pmove.cmd = cl.frames[seq].cmd;
		if (synth_button2)
		{
			/* Force button2 high; PlayerMove's JumpButton reads
			 * pmove.cmd.buttons & 2. Anti-pogo (oldbuttons & 2)
			 * still applies — that's exactly what feasibility
			 * means: if the demo was already holding button2 last
			 * tick, the engine will NOT fire a new jump even if
			 * we press now, so feasibility=0 in that frame. */
			pmove.cmd.buttons |= 2;
		}
		qnn_pmove_jump_attacked = 0;
		PlayerMove();
		if (qnn_pmove_jump_attacked)
			op_jump = 1;
	}

	if (synth_button2)
	{
		/* Restore the carry state captured before the synthetic loop.
		 * The real-jump label path's per-tick state must not see the
		 * "we just held button2 the whole window" residue. */
		qwd_state.pmove_oldbuttons = saved_pmove_oldbuttons;
		qwd_state.pmove_oldbuttons_inited = saved_pmove_oldbuttons_inited;
		VectorCopy(saved_pmove_prev_simorg, qwd_state.pmove_prev_simorg);
		VectorCopy(saved_pmove_prev_simvel, qwd_state.pmove_prev_simvel);
		qwd_state.pmove_prev_sim_inited = saved_pmove_prev_sim_inited;
	}
	else
	{
		/* Carry pmove.oldbuttons forward — that's the anti-pogo state
		 * the next tick's first cmd needs to see. */
		qwd_state.pmove_oldbuttons = pmove.oldbuttons;

		/* Capture engine's predicted state (post this tick's cmds in
		 * cl.simorg/simvel) as the seed for the next call. */
		VectorCopy(cl.simorg, qwd_state.pmove_prev_simorg);
		VectorCopy(cl.simvel, qwd_state.pmove_prev_simvel);
		qwd_state.pmove_prev_sim_inited = true;
	}

	QNN_PmoveRestore(&save);
	return op_jump;
}


void QNN_QwdExtractAction(qnn_action_t *action, const qnn_snapshot_t *snapshot)
{
	int window_start;
	int window_end;
	int n;
	int i;
	int attack_any;
	int jump_press_any;	/* buttons & BUTTON_JUMP set in any cmd */
	int upmove_pos_any;	/* upmove > 0 in any cmd (swim up) */
	int jump_any;		/* jump_press_any OR upmove_pos_any — drives
				 * the unified ud-pos press bit in action->move. */
	long forward_sum;
	long side_sum;
	int last_upmove;
	int last_nonzero_impulse;
	int op_attack_any;
	int attack_weapon;
	int op_impulse_any;	/* any cmd's weapon-select impulse flipped self.weapon */
	int advance_weapon;

	QNN_ClearAction(action);
	/* Stash slots read by QNN_QwdPackInputMask run from QwdBuildActionLabel. */
	qwd_state.last_op_attack = 0;
	/* Snapshot cooldown state BEFORE the per-cmd loop runs (and possibly
	 * advances it by attacking this tick). QwdPackInputMask uses this so
	 * its bit-0 feasibility check answers "could press attack AT START of
	 * this tick" — which AND'd with the demo press gives the correct
	 * engine-attacked-this-tick label. Without this snapshot, PackInputMask
	 * saw post-loop state and bit 0 ended up anti-correlated with the
	 * demo press on the very ticks the engine actually attacked. */
	qwd_state.pre_loop_attack_finished = QNN_ProgsGetAttackFinished();

	window_end = cls.netchan.outgoing_sequence;
	if (window_end <= 0)
		return;

	window_start = qnn_runtime.cmd_seq_window_start;
	n = window_end - window_start;
	if (n <= 0)
	{
		/* Empty window (slow demo): hold latest cmd as the carried state. */
		window_start = window_end - 1;
		n = 1;
	}
	if (n > UPDATE_BACKUP)
		n = UPDATE_BACKUP;

	attack_any = 0;
	jump_press_any = 0;
	upmove_pos_any = 0;
	jump_any = 0;
	forward_sum = 0;
	side_sum = 0;
	last_upmove = 0;
	last_nonzero_impulse = 0;
	op_attack_any = 0;
	attack_weapon = 0;
	op_impulse_any = 0;
	/* Advance the QC weapon-select predicate exactly once per emit tick.
	 * QwdBuildActionLabel may call QwdExtractAction more than once for the
	 * same tick (force_mvd_emit path), and the predicate is stateful, so a
	 * second window replay would corrupt its self.weapon / sticky-impulse
	 * tracking — gate on qnn_runtime.tick. */
	advance_weapon = (qnn_runtime.tick != qwd_state.weapon_advanced_tick);
	if (advance_weapon)
		qwd_state.weapon_advanced_tick = qnn_runtime.tick;
	for (i = 0; i < n; ++i)
	{
		int seq = (window_start + i) & UPDATE_MASK;
		usercmd_t *cmd = &cl.frames[seq].cmd;
		int imp;
		int button0 = (cmd->buttons & 1) ? 1 : 0;
		int button2 = ((cmd->buttons & 2) || cmd->upmove > 0) ? 1 : 0;

		if (cmd->buttons & 1)
			attack_any = 1;
		/* Split jump button from upmove > 0 so input_mask can record
		 * them on distinct bits. jump_any (used to derive the unified
		 * ud-pos press bit) stays as OR of the two. */
		if (cmd->buttons & 2)
			jump_press_any = 1;
		if (cmd->upmove > 0)
			upmove_pos_any = 1;
		if ((cmd->buttons & 2) || cmd->upmove > 0)
			jump_any = 1;
		forward_sum += (long)cmd->forwardmove;
		side_sum += (long)cmd->sidemove;
		/* Track most-negative upmove across the window so a brief
		 * swim-down press isn't lost when the recorder releases before
		 * the emit boundary. */
		if (cmd->upmove < last_upmove)
			last_upmove = cmd->upmove;

		/* Track the latest weapon-select impulse seen in the window.
		 * 1-8 = direct select (axe..lightning), 10 = nextweapon,
		 * 12 = prevweapon. */
		imp = (int)cmd->impulse;
		if ((imp >= 1 && imp <= 8) || imp == 10 || imp == 12)
			last_nonzero_impulse = imp;

		/* Per-cmd QC think advance: ONE W_WeaponFrame advances self.weapon +
		 * attack_finished on shared persistent state so W_Attack advances the
		 * currently-selected weapon through the engine cooldown gate (a select
		 * pressed during attack cooldown is realized only when the engine would
		 * — no early-process stuck divergence).  This feeds ONLY the attack
		 * op-bit and categorical attack label.
		 *
		 * Tick-guarded (the force_mvd_emit path may call QwdExtractAction
		 * twice per tick) — a double-advance would corrupt
		 * attack_finished. */
		if (advance_weapon && snapshot)
		{
			int attack_op = 0;
			int weapon_after = 0;
			/* Edge-trigger the weapon select.  The replayed cmd impulse
			 * byte is held across cmds (not cleared like the live server
			 * clears self.impulse), so feeding it raw re-runs
			 * W_ChangeWeapon every frame — a stale impulse 1 keeps
			 * re-selecting axe over the (correct) snapshot weapon.  Pass
			 * the impulse to the QC advance only when it CHANGES; 0 while
			 * held.  Matches the live "consume self.impulse once" rule.
			 * Attack (button0) is unaffected — only the select is gated. */
			int imp_edge = (imp != qwd_state.prev_cmd_impulse) ? imp : 0;
			int weapon_op = 0;
			qwd_state.prev_cmd_impulse = imp;
			QNN_ProgsStepWeaponFrame(
				QNN_RuntimeNowSeconds(),
				snapshot->health, snapshot->items_owned,
				snapshot->ammo_shells, snapshot->ammo_nails,
				snapshot->ammo_rockets, snapshot->ammo_cells,
				snapshot->weapon_id, imp_edge, button0,
				&weapon_after, &weapon_op, &attack_op);
			if (attack_op)
			{
				op_attack_any = 1;
				if (QNN_WeaponIsValid(weapon_after))
					attack_weapon = weapon_after;
			}
			/* op_impulse: this cmd's edge-triggered select flipped
			 * self.weapon — the operative weapon-switch this tick.
			 * Captured from the same shared QC advance the attack op
			 * uses, so no second stateful predicate runs. */
			if (weapon_op)
				op_impulse_any = 1;
		}
	}
	/* Stash for QNN_QwdPackInputMask (called once per tick from
	 * QwdBuildActionLabel). The stash is overwritten by each
	 * QwdExtractAction call; the value seen by QwdPackInputMask is
	 * the one set by the QwdBuildActionLabel-internal call. */
	qwd_state.last_op_attack = op_attack_any;
	qwd_state.last_attack_weapon = attack_weapon;
	qwd_state.last_op_impulse = op_impulse_any;
	qwd_state.last_weapon_impulse = last_nonzero_impulse;
	qwd_state.last_jump_press_any = jump_press_any;
	qwd_state.last_upmove_pos_any = upmove_pos_any;

	/* Average move across the window, threshold at the Python compaction
	 * boundary (MOVE_AXIS_THRESHOLD = 0.1) so the press byte matches
	 * the on-disk representation a downstream loader expects.  Each cmd
	 * is integrated over native_dt by the recorder's kbutton_t; averaging
	 * N cmds gives the integrated displacement a single 50ms cmd would
	 * represent — which is what a 20fps policy needs to learn to emit. */
	{
		float fb_avg = QNN_Clamp((float)forward_sum / (n * QNN_SV_MAXSPEED), -1.0f, 1.0f);
		float lr_avg = QNN_Clamp((float)side_sum    / (n * QNN_SV_MAXSPEED), -1.0f, 1.0f);
		int fb_neg = (fb_avg < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		int fb_pos = (fb_avg >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		int lr_neg = (lr_avg < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		int lr_pos = (lr_avg >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		/* up-pos bit captures swim-up / jumppad upmove ONLY
		 * (cmd->upmove > 0 in any cmd of the window).  Jump-button
		 * presses are tracked separately via bit 7 (jump_act) so the
		 * model can learn jump-via-button vs ambient-up as distinct
		 * intents.  up-neg bit captures swim-down (upmove<0). */
		int up_pos = upmove_pos_any ? 1 : 0;
		int up_neg = (last_upmove < 0) ? 1 : 0;
		action->move = QNN_PackInputMask(
			/*alive=*/1,
			fb_neg, fb_pos,
			lr_neg, lr_pos,
			up_neg, up_pos,
			/*jump_act=*/jump_press_any,
			/*attack_act=*/attack_any);
	}

}

void QNN_QwdBuildActionLabel(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	/* Side effects we depend on: QwdExtractAction packs action->move,
	 * stashes the input-mask inputs, and (per-cmd, once per emit tick)
	 * advances the QC weapon-select predicate. */
	QNN_QwdExtractAction(action, snapshot);

	QNN_FillLook(action, snapshot);

	/* Attack label: 0 when no effective attack occurred; otherwise the
	 * impulse class actually advanced by QC W_Attack. */
	action->attack = qwd_state.last_op_attack
		? qwd_state.last_attack_weapon : 0;

	/* action->weapon stays 0: A27 retires the carried select-intent label
	 * (agents/plans/a27-pure-combat-substrate.md invariant 7 — no equipped
	 * weapon or carried weapon intent).  The sole weapon decision is the
	 * 9-class action->attack above.  The struct field survives only because
	 * the 20-byte qnn_action_t layout is pinned. */
	QNN_QwdPackInputMask(action, snapshot);

	QNN_QwdDumpAttackEdge(action, snapshot);
}

/* Pack action->input_mask — pure-feasibility per-axis mask the trainer
 * consumes when input_mask=true. "Would the engine accept this axis if
 * the player pressed AT THE START of this tick?" — no AND with the demo's
 * actual press. State is captured PRE-loop (before this tick's attack
 * advances cooldown) via qwd_state.pre_loop_attack_finished, so bit 0
 * AND demo_press cleanly recovers the engine-attacked-this-tick label.
 *
 * Layout (see QNN_PackInputMask):
 *
 *   bit 0    = attack feasibility : would QC W_Attack trigger if button0=1
 *                                   at START of this tick? (cooldown
 *                                   expired AND held weapon has ammo)
 *   bits 1-2 = forward feasibility: both bits set whenever alive —
 *                                   pmove always processes fmove on
 *                                   ground / in air / in water.
 *   bits 3-4 = side    feasibility: same — always 11 when alive.
 *   bits 5-6 = up      feasibility: 11 when in water (swim up & down
 *                                   both feasible), 00 otherwise.
 *   bit 7    = jump    feasibility: would pmove ground-jump trigger if
 *                                   button2=1 each cmd? (depends on
 *                                   onground + anti-pogo + dead state)
 *
 * Trainer side then computes the engine-outcome label as
 * (feasibility_bit & demo_press_bit) per axis; the schema decouples
 * "engine state" from "what the player did" so the trainer can reason
 * about both independently.
 *
 * Honest tradeoff: fb/lr feasibility carries no info (always 1) under
 * pure-feasibility semantics — the bits stay in the layout for schema
 * consistency across axes. attack and jump are where the bits matter.
 *
 * Implementation:
 *   - Attack feasibility: save the persistent qnn_progs_attack_finished,
 *     call QNN_ProgsEvalAttack with button0=1 (existing predicate runs
 *     ammo/weapon checks), restore the saved value so the real per-cmd
 *     loop's state isn't contaminated. The predicate's own re-zeroing
 *     of self.* per call means per-snapshot ammo state is fresh.
 *   - Jump feasibility: QwdEvalPmoveJump with synth_button2=1, which
 *     saves/restores its own pmove carry state internally.
 *
 * Must run exactly once per tick (from QwdBuildActionLabel) — same
 * cadence requirement as the previous engine-act version. */
void QNN_QwdPackInputMask(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	int op_jump_feasible;
	int op_attack_feasible;
	int in_water;

	if (action == NULL || snapshot == NULL)
		return;
	if (snapshot->health <= 0)
	{
		action->input_mask = 0;
		action->op_input = 0;
		return;
	}

	in_water = (snapshot->waterlevel >= 2);

	/* Attack feasibility via the existing QC predicate.
	 *
	 * Two state-management requirements:
	 *  (a) The synthetic call here must use PRE-loop cooldown as its
	 *      baseline so bit 0 means "could press attack at START of this
	 *      tick" — which AND'd with the demo press recovers the
	 *      engine-attacked-this-tick label. Without this, the per-cmd
	 *      loop in QwdExtractAction has already advanced cooldown by
	 *      this tick's attack, and bit 0 collapses to "could press attack
	 *      NEXT tick" — anti-correlated with demo press on exactly
	 *      the ticks the engine actually attacked.
	 *  (b) After the synthetic call we must restore the POST-loop value
	 *      so downstream code (next tick's per-cmd loop, BC self_token
	 *      attack_finished readout) sees the real engine state.
	 *
	 * We snapshot the post-loop value, switch to the pre-loop value
	 * for the eval, restore the post-loop value when done. */
	/* Synthetic "would a press attack at the START of this tick?" feasibility
	 * (shared helper, baseline = pre-loop cooldown). */
	op_attack_feasible = QNN_EvalAttackFeasible(snapshot,
		qwd_state.pre_loop_attack_finished);

	/* Jump feasibility via pmove with synthetic button2=1; the helper
	 * internally save/restores its persistent carry state.  Computed
	 * here (before op_input) because op_input's ud op-bit reuses it —
	 * see below.  This call leaves pmove carry state untouched, so it
	 * does NOT perturb the genuine QWD path's per-tick jump state
	 * (byte-identical input_mask). */
	op_jump_feasible = QNN_QwdEvalPmoveJump(snapshot, /*synth_button2=*/1);

	/* op_input (strict per-axis operativeness — DISTINCT from the
	 * pure-feasibility input_mask): bit i set iff the player pressed
	 * axis i AND the engine acted on it this tick.  Computed entirely
	 * from signals already produced on this BC QWD path, with NO new
	 * stateful predicate (so input_mask stays byte-identical):
	 *   fb/lr   : press read off action->move (set by QwdExtractAction)
	 *   ud      : (jump press AND the engine would jump = op_jump_feasible),
	 *             OR swim-up in water — operativeness = "pressed AND it
	 *             took effect".  The feasibility eval honours onground +
	 *             anti-pogo, so feasible-while-pressing == jumped.
	 *   attack  : last_op_attack (per-cmd QC W_Attack advance)
	 *   impulse : last_op_impulse (per-cmd QC weapon-select advance) */
	{
		int fb_press = QNN_ActionAxisSign(action->move, 0) != 0;
		int lr_press = QNN_ActionAxisSign(action->move, 1) != 0;
		int attack_press = QNN_ActionAttackPressed(action->move);
		action->op_input = QNN_PackOpInput(
			/*alive=*/1,
			fb_press, lr_press,
			/*jump_press=*/qwd_state.last_jump_press_any,
			/*swim_press=*/qwd_state.last_upmove_pos_any,
			attack_press,
			in_water,
			/*op_jump=*/op_jump_feasible,
			qwd_state.last_op_attack,
			qwd_state.last_op_impulse,
			/*has_impulse=*/qwd_state.last_weapon_impulse != 0);
	}

	/* fb/lr: always feasible — pmove processes fmove/smove in every
	 * physics branch (ground, air, water). Set BOTH direction bits to
	 * signal "either direction is engine-accepted right now". */
	action->input_mask = QNN_PackInputMask(
		1,
		/*fb_act_neg=*/1, /*fb_act_pos=*/1,
		/*lr_act_neg=*/1, /*lr_act_pos=*/1,
		/*up_act_neg=*/in_water ? 1 : 0,
		/*up_act_pos=*/in_water ? 1 : 0,
		op_jump_feasible,
		op_attack_feasible);
}
