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
	 * QNN_QwdExtractAction's cmd-window loop. QwdInferEmitAction reads
	 * this to pack action->op_input bit 3 without re-advancing the QC
	 * VM. Only valid when set by the same-tick QwdExtractAction call;
	 * QwdInferEmitAction inherits the value implicitly via the
	 * QwdExtractAction call that happens inside it. */
	int		last_op_fire;
	/* Last nonzero impulse byte seen anywhere in the cmd window — used
	 * by QwdPackOpInput as the input to ProgsEvalWeaponImpulseOperative
	 * for op_input bit 4. Matches the labeler-mode aggregation. */
	int		last_impulse_any;
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
int QNN_QwdEvalPmoveJump(const qnn_snapshot_t *snapshot)
{
	int window_start;
	int window_end;
	int n;
	int i;
	int op_jump = 0;
	qnn_pmove_save_t save;

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
		qnn_pmove_jump_fired = 0;
		PlayerMove();
		if (qnn_pmove_jump_fired)
			op_jump = 1;
	}

	/* Carry pmove.oldbuttons forward — that's the anti-pogo state
	 * the next tick's first cmd needs to see. */
	qwd_state.pmove_oldbuttons = pmove.oldbuttons;

	/* Capture engine's predicted state (post this tick's cmds in
	 * cl.simorg/simvel) as the seed for the next call. */
	VectorCopy(cl.simorg, qwd_state.pmove_prev_simorg);
	VectorCopy(cl.simvel, qwd_state.pmove_prev_simvel);
	qwd_state.pmove_prev_sim_inited = true;

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
	int jump_any;
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
	/* Stash slots read by QNN_QwdPackOpInput run from QwdInferEmitAction. */
	qwd_state.last_op_fire = 0;
	qwd_state.last_impulse_any = 0;

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
		 * QNN_SelfEmitToken, AND captures op_fire_any so QwdPackOpInput
		 * can populate action->op_input bit 3 without re-advancing.
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
	/* Stash for QNN_QwdPackOpInput (called once per tick from
	 * QwdInferEmitAction). The stash is overwritten by each
	 * QwdExtractAction call; the value seen by QwdPackOpInput is the
	 * one set by the QwdInferEmitAction-internal call. */
	qwd_state.last_op_fire = op_fire_any;
	qwd_state.last_impulse_any = impulse_any;

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
		if (cl.stats[STAT_ITEMS] & item)
			impulse_target = last_nonzero_impulse;
	}

	/* Average move across the window: each cmd is integrated over
	 * native_dt by the recorder's kbutton_t; averaging N cmds gives
	 * the integrated displacement a single 50ms cmd would represent —
	 * which is what a 20fps policy needs to learn to emit.  Server-
	 * side, our policy's one 50ms cmd runs as ONE physics sub-step,
	 * so the label should match that one-cmd intent. */
	action->move[0] = QNN_Clamp((float)forward_sum / (n * QNN_SV_MAXSPEED), -1.0f, 1.0f);
	action->move[1] = QNN_Clamp((float)side_sum / (n * QNN_SV_MAXSPEED), -1.0f, 1.0f);
	action->fire = fire_any;

	if (jump_any)
		action->move[2] = QNN_SV_JUMP_SPEED / QNN_SV_MAXSPEED;
	else if (last_upmove < 0)
		action->move[2] = QNN_Clamp((float)last_upmove / QNN_SV_MAXSPEED, -1.0f, 0.0f);

	/* action->weapon is filled by QNN_QwdInferEmitAction using the
	 * snapshot's weapon_id (server-state at the same point in the tick
	 * the obs sees) plus the impulse_target returned here.  Keeping
	 * extraction pure (no qwd_state, no snapshot dependency) ensures
	 * cur_stat and snapshot->weapon_id stay coherent. */
	return impulse_target;
}

void QNN_QwdInferEmitAction(qnn_action_t *action,
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
	QNN_QwdPackOpInput(action, snapshot);
}

/* Pack the per-axis op_input mask onto action->op_input.
 *
 *   bit0 = fb       : forward_sum != 0 (aggregated cmd_move.fb)
 *   bit1 = lr       : side_sum != 0    (aggregated cmd_move.lr)
 *   bit2 = ud       : jump_any AND pmove ground-jump op,
 *                     OR swim-down AND in-water
 *   bit3 = fire     : fire_any AND QC W_WeaponFrame op
 *   bit4 = impulse  : impulse_any != 0 AND QC ImpulseCommands op
 *
 * Reads op_fire and the aggregated impulse byte from qwd_state (stashed
 * by the same-tick QwdExtractAction call). Calls QwdEvalPmoveJump and
 * ProgsEvalWeaponImpulseOperative directly — those have per-tick state
 * that must advance exactly once per tick, so this function must run
 * exactly once per tick (from QwdInferEmitAction).
 *
 * Press bits are derived from the action's already-filled move/fire
 * fields, which encode the same fb/lr/ud/fire press semantics as the
 * labeler's raw cmd_move/cmd_buttons inspection (zero ⇔ no press,
 * action->move[2] > 0 ⇔ jump intent, < 0 ⇔ swim-down). */
void QNN_QwdPackOpInput(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	int op_jump;
	int op_impulse;

	if (action == NULL || snapshot == NULL)
		return;
	if (snapshot->health <= 0)
	{
		action->op_input = 0;
		return;
	}

	op_jump = QNN_QwdEvalPmoveJump(snapshot);
	op_impulse = QNN_ProgsEvalWeaponImpulseOperative(
		snapshot->health, snapshot->items_owned,
		snapshot->ammo_shells, snapshot->ammo_nails,
		snapshot->ammo_rockets, snapshot->ammo_cells,
		snapshot->weapon_id, qwd_state.last_impulse_any);

	action->op_input = QNN_PackOpInput(
		1,
		(action->move[0] != 0.0f),
		(action->move[1] != 0.0f),
		(action->move[2] > 0.0f),
		(action->move[2] < 0.0f),
		action->fire ? 1 : 0,
		(snapshot->waterlevel >= 2),
		op_jump, qwd_state.last_op_fire, op_impulse,
		qwd_state.last_impulse_any != 0);
}
