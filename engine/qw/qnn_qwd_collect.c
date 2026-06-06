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
 *   - fire/jump: OR across cmds (any press in the window counts)
 *   - forward/side move: average (each cmd is integrated over native_dt;
 *     averaging N approximates a single 50ms-integrated cmd)
 *   - upmove (swim down): min across cmds (most-negative captures any
 *     swim-down press regardless of when in the window it occurred)
 *   - weapon switch: per-frame "held weapon" state, not event.
 *     QWD is ground truth — the player's intent is observable in
 *     the usercmd impulse byte every frame.  Direct labelling:
 *       impulse 1-8 → that weapon (gated on inventory ownership)
 *       impulse 10  → QNN_NextWeaponId(forward) — deterministic
 *                     simulation of server CycleWeaponCommand
 *       impulse 12  → QNN_NextWeaponId(reverse)
 *       no impulse  → carry prior intent, or accept a non-impulse
 *                     server-state change (pickup/respawn/auto-switch)
 *     No ping math — this is INTENT TRACKING from observable usercmds,
 *     not inference.  MVD-side inference (back-shift from server
 *     state changes) lives in qnn_mvd_collect.c.
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
 * look/switch still come from state-derived deltas (FillLookAndSwitch in
 * qnn_collect_helpers.c).  MVD path keeps the snap (see
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

/* Module-private state: tracks the canonical held-weapon label across
 * ticks (carry-forward on non-press ticks, engine-forced override on
 * stat transitions) plus pmove.oldbuttons for the pmove jump driver's
 * cross-tick anti-pogo state. */
static struct
{
	int		held_weapon;       /* canonical action.weapon value */
	int		prev_stat_weapon;  /* prior tick's snapshot->weapon_id */
	qboolean	initialized;
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
	/* op_fire captured from the per-cmd predicate call inside
	 * QNN_QwdExtractAction's cmd-window loop. Read by
	 * QwdPackInputMask to set input_mask bit 0 (attack) without
	 * re-advancing the QC VM. Only valid when set by the same-tick
	 * QwdExtractAction call; QwdBuildActionLabel inherits the value
	 * implicitly via the QwdExtractAction call that happens inside
	 * it. */
	int		last_op_fire;
	/* Last nonzero impulse byte seen anywhere in the cmd window. The
	 * BC QWD path doesn't pack impulse-feasibility into input_mask
	 * (impulse is the weapon-switch byte, not an axis); kept here
	 * because the labeler path's separate LOBS bit uses it. */
	int		last_impulse_any;
	/* Per-tick aggregated press signals captured from the cmd window
	 * so QwdPackInputMask can read them without re-iterating. Jump
	 * and upmove are kept distinct here (vs collapsed into the
	 * legacy op_input ud bit). */
	int		last_jump_press_any;	/* any cmd had buttons & BUTTON_JUMP */
	int		last_upmove_pos_any;	/* any cmd had upmove > 0 (swim up) */
	/* QC predicate state at the START of QwdExtractAction's per-cmd
	 * loop — i.e. BEFORE this tick's fire (if any) advances cooldown.
	 * QwdPackInputMask reads this so its synthetic feasibility check
	 * answers "could a press fire AT THE BEGINNING of this tick" not
	 * "could a press fire NEXT tick after this tick's fire already
	 * advanced the cooldown" (the previous off-by-one — input_mask
	 * was capturing post-loop state, anti-correlated with demo press
	 * on engine-fired ticks). */
	float		pre_loop_attack_finished;
} qwd_state;

void QNN_QwdCollectReset(void)
{
	memset(&qwd_state, 0, sizeof(qwd_state));
}

/* Iterate the QWD cmd window and evaluate the per-cmd operative
 * predicates (QC W_WeaponFrame for fire, PlayerJump for jump) while
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
	int *out_op_fire,
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
	int op_fire = 0;
	int held_weapon = snapshot->weapon_id;
	long forward_sum = 0;
	long side_sum = 0;
	int last_umove_neg = 0;
	int buttons_or = 0;
	int last_nonzero_impulse = 0;
	int jump_any = 0;

	*out_op_fire = 0;
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

		/* Fire predicate (only meaningful if a weapon is held). */
		if (held_weapon >= 1 && held_weapon <= 8)
		{
			int fire_op = QNN_ProgsEvalAttack(
				qnn_runtime.tick,
				qnn_runtime.fixed_tick_hz,
				snapshot->health, snapshot->items_owned,
				snapshot->ammo_shells, snapshot->ammo_nails,
				snapshot->ammo_rockets, snapshot->ammo_cells,
				held_weapon, button0);
			if (fire_op)
				op_fire = 1;
		}
	}

	*out_op_fire = op_fire;
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
		qnn_pmove_jump_fired = 0;
		PlayerMove();
		if (qnn_pmove_jump_fired)
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


int QNN_QwdExtractAction(qnn_action_t *action, const qnn_snapshot_t *snapshot)
{
	int window_start;
	int window_end;
	int n;
	int i;
	int fire_any;
	int jump_press_any;	/* buttons & BUTTON_JUMP set in any cmd */
	int upmove_pos_any;	/* upmove > 0 in any cmd (swim up) */
	int jump_any;		/* jump_press_any OR upmove_pos_any — drives
				 * the unified ud-pos press bit in action->move. */
	long forward_sum;
	long side_sum;
	int last_upmove;
	int last_nonzero_impulse;
	int impulse_any;
	int op_fire_any;
	int impulse_target;
	int held_weapon = snapshot ? snapshot->weapon_id : 0;
	int grounded_effective = snapshot ? (snapshot->grounded ? 1 : 0) : 0;

	QNN_ClearAction(action);
	/* Stash slots read by QNN_QwdPackInputMask run from QwdBuildActionLabel. */
	qwd_state.last_op_fire = 0;
	qwd_state.last_impulse_any = 0;
	/* Snapshot cooldown state BEFORE the per-cmd loop runs (and possibly
	 * advances it by firing this tick). QwdPackInputMask uses this so
	 * its bit-0 feasibility check answers "could press fire AT START of
	 * this tick" — which AND'd with the demo press gives the correct
	 * engine-fired-this-tick label. Without this snapshot, PackInputMask
	 * saw post-loop state and bit 0 ended up anti-correlated with the
	 * demo press on the very ticks the engine actually fired. */
	qwd_state.pre_loop_attack_finished = QNN_ProgsGetAttackFinished();

	window_end = cls.netchan.outgoing_sequence;
	if (window_end <= 0)
		return 0;

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

	fire_any = 0;
	jump_press_any = 0;
	upmove_pos_any = 0;
	jump_any = 0;
	forward_sum = 0;
	side_sum = 0;
	last_upmove = 0;
	last_nonzero_impulse = 0;
	impulse_any = 0;
	op_fire_any = 0;
	for (i = 0; i < n; ++i)
	{
		int seq = (window_start + i) & UPDATE_MASK;
		usercmd_t *cmd = &cl.frames[seq].cmd;
		int imp;
		int button0 = (cmd->buttons & 1) ? 1 : 0;
		int button2 = ((cmd->buttons & 2) || cmd->upmove > 0) ? 1 : 0;

		if (cmd->buttons & 1)
			fire_any = 1;
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
		/* Track any nonzero impulse byte for op_input bit 4 (matches the
		 * labeler-mode aggregation in QwdEvalOperativePerCmd). */
		if (imp != 0)
			impulse_any = imp;

		/* Advance the QC predicate VM per-cmd in NON-labeler mode only.
		 * In non-labeler mode this side-effect drives the BC self_token
		 * attack_finished scalar via QNN_ProgsGetAttackCdRemaining in
		 * QNN_SelfEmitToken, AND captures op_fire_any so QwdPackInputMask
		 * can populate action->input_mask bit 0 without re-advancing.
		 * In LABELER mode the same per-cmd loop runs a SECOND time
		 * inside QNN_QwdEvalOperativePerCmd — running it here too would
		 * double-advance qnn_progs_attack_finished and pre-reject every
		 * fire (regression caught during the impulse/fire F1 audit). */
		if (!qnn_runtime.labeler_mode
			&& snapshot && held_weapon >= 1 && held_weapon <= 8)
		{
			int fire_op = QNN_ProgsEvalAttack(
				qnn_runtime.tick,
				qnn_runtime.fixed_tick_hz,
				snapshot->health, snapshot->items_owned,
				snapshot->ammo_shells, snapshot->ammo_nails,
				snapshot->ammo_rockets, snapshot->ammo_cells,
				held_weapon, button0);
			if (fire_op)
				op_fire_any = 1;
		}
	}
	/* Stash for QNN_QwdPackInputMask (called once per tick from
	 * QwdBuildActionLabel). The stash is overwritten by each
	 * QwdExtractAction call; the value seen by QwdPackInputMask is
	 * the one set by the QwdBuildActionLabel-internal call. */
	qwd_state.last_op_fire = op_fire_any;
	qwd_state.last_impulse_any = impulse_any;
	qwd_state.last_jump_press_any = jump_press_any;
	qwd_state.last_upmove_pos_any = upmove_pos_any;

	/* Resolve the impulse byte to a concrete 1..8 weapon id, gated
	 * on ownership for direct selects.  Server-side cmd dispatch
	 * applies the same ownership gate, so resolved values are
	 * exactly the transitions the server actually produces. */
	impulse_target = 0;
	if (last_nonzero_impulse == 10 || last_nonzero_impulse == 12)
	{
		impulse_target = QNN_NextWeaponId(last_nonzero_impulse == 12);
	}
	else if (last_nonzero_impulse >= 1 && last_nonzero_impulse <= 8)
	{
		int item = QNN_ItemFlagFromImpulse(last_nonzero_impulse);
		if (snapshot && (snapshot->items_owned & item))
			impulse_target = last_nonzero_impulse;
	}

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
			/*attack_act=*/fire_any);
	}

	/* action->weapon is filled by QNN_QwdBuildActionLabel using the
	 * snapshot's weapon_id (server-state at the same point in the tick
	 * the obs sees) plus the impulse_target returned here.  Keeping
	 * extraction pure (no qwd_state, no snapshot dependency) ensures
	 * cur_stat and snapshot->weapon_id stay coherent. */
	return impulse_target;
}

void QNN_QwdBuildActionLabel(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	int impulse_target = QNN_QwdExtractAction(action, snapshot);
	int cur_stat = snapshot->weapon_id;

	/* Canonical action.weapon label.  Three paths:
	 *   (1) impulse press: player chose explicitly.
	 *   (2) engine-forced: server changed the held weapon without an
	 *       impulse (pickup auto-switch via weapon_touch, respawn
	 *       default, ammo-out down-switch in some mods).  Detected as
	 *       a snapshot->weapon_id transition to a value the player
	 *       didn't ask for.
	 *   (3) carry: keep last value (covers cmd-pipeline lag and steady
	 *       state).
	 * No back-shift, no ring rewriting — the label at this tick is
	 * the truth label.  snapshot->weapon_id is used (not QNN_WeaponId)
	 * so cur_stat and the obs byte stay coherent. */
	if (!qwd_state.initialized && cur_stat > 0)
	{
		qwd_state.held_weapon      = cur_stat;
		qwd_state.prev_stat_weapon = cur_stat;
		qwd_state.initialized      = true;
	}
	if (impulse_target != 0)
	{
		qwd_state.held_weapon = impulse_target;
		if (!qwd_state.initialized)
		{
			/* Impulse-first init: seed prev_stat so the next tick's
			 * engine-forced check sees a consistent baseline. */
			qwd_state.prev_stat_weapon = cur_stat;
			qwd_state.initialized      = true;
		}
	}
	else if (qwd_state.initialized
		&& cur_stat != qwd_state.prev_stat_weapon
		&& cur_stat != qwd_state.held_weapon
		&& cur_stat > 0)
	{
		qwd_state.held_weapon = cur_stat;
	}
	qwd_state.prev_stat_weapon = cur_stat;
	if (qwd_state.initialized)
		action->weapon = qwd_state.held_weapon;
	/* else: leave action->weapon at 0; QNN_FillLookAndSwitch fills it
	 * from snapshot->weapon_id for pre-signon frames. */

	QNN_FillLookAndSwitch(action, snapshot);
	QNN_QwdPackInputMask(action, snapshot);
}

/* Pack action->input_mask — pure-feasibility per-axis mask the trainer
 * consumes when input_mask=true. "Would the engine accept this axis if
 * the player pressed AT THE START of this tick?" — no AND with the demo's
 * actual press. State is captured PRE-loop (before this tick's fire
 * advances cooldown) via qwd_state.pre_loop_attack_finished, so bit 0
 * AND demo_press cleanly recovers the engine-fired-this-tick label.
 *
 * Layout (see QNN_PackInputMask):
 *
 *   bit 0    = attack feasibility : would QC W_Attack fire if button0=1
 *                                   at START of this tick? (cooldown
 *                                   expired AND held weapon has ammo)
 *   bits 1-2 = forward feasibility: both bits set whenever alive —
 *                                   pmove always processes fmove on
 *                                   ground / in air / in water.
 *   bits 3-4 = side    feasibility: same — always 11 when alive.
 *   bits 5-6 = up      feasibility: 11 when in water (swim up & down
 *                                   both feasible), 00 otherwise.
 *   bit 7    = jump    feasibility: would pmove ground-jump fire if
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
	int held_weapon;
	float saved_attack_finished;

	if (action == NULL || snapshot == NULL)
		return;
	if (snapshot->health <= 0)
	{
		action->input_mask = 0;
		return;
	}

	in_water = (snapshot->waterlevel >= 2);
	held_weapon = snapshot->weapon_id;

	/* Attack feasibility via the existing QC predicate.
	 *
	 * Two state-management requirements:
	 *  (a) The synthetic call here must use PRE-loop cooldown as its
	 *      baseline so bit 0 means "could press fire at START of this
	 *      tick" — which AND'd with the demo press recovers the
	 *      engine-fired-this-tick label. Without this, the per-cmd
	 *      loop in QwdExtractAction has already advanced cooldown by
	 *      this tick's fire, and bit 0 collapses to "could press fire
	 *      NEXT tick" — anti-correlated with demo press on exactly
	 *      the ticks the engine actually fired.
	 *  (b) After the synthetic call we must restore the POST-loop value
	 *      so downstream code (next tick's per-cmd loop, BC self_token
	 *      attack_finished readout) sees the real engine state.
	 *
	 * We snapshot the post-loop value, switch to the pre-loop value
	 * for the eval, restore the post-loop value when done. */
	saved_attack_finished = QNN_ProgsGetAttackFinished();
	QNN_ProgsSetAttackFinished(qwd_state.pre_loop_attack_finished);
	op_attack_feasible = 0;
	if (held_weapon >= 1 && held_weapon <= 8)
	{
		op_attack_feasible = QNN_ProgsEvalAttack(
			qnn_runtime.tick,
			qnn_runtime.fixed_tick_hz,
			snapshot->health, snapshot->items_owned,
			snapshot->ammo_shells, snapshot->ammo_nails,
			snapshot->ammo_rockets, snapshot->ammo_cells,
			held_weapon, /*button0=*/1);
	}
	QNN_ProgsSetAttackFinished(saved_attack_finished);

	/* Jump feasibility via pmove with synthetic button2=1; the helper
	 * internally save/restores its persistent carry state. */
	op_jump_feasible = QNN_QwdEvalPmoveJump(snapshot, /*synth_button2=*/1);

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
