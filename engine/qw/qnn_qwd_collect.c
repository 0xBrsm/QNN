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

int QNN_QwdExtractAction(qnn_action_t *action)
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
	int impulse_target;

	QNN_ClearAction(action);

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
	for (i = 0; i < n; ++i)
	{
		int seq = (window_start + i) & UPDATE_MASK;
		usercmd_t *cmd = &cl.frames[seq].cmd;
		int imp;

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
	}

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

	/* action->weapon left at 0 here; QNN_FillLookAndSwitch fills it
	 * with snapshot->weapon_id (current server weapon) — the back-shift
	 * ring rewrites that label by walking back to `impulse_target`
	 * when a transition is observed. */
	return impulse_target;
}

int QNN_QwdInferEmitAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot)
{
	int impulse_target = QNN_QwdExtractAction(action);
	QNN_FillLookAndSwitch(action, snapshot);
	return impulse_target;
}
