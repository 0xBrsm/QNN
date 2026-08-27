#include "qnn.h"
#include "qnn_io.h"
#include "qnn_predict.h"

/* Move output is in [-1, 1].  Scale to sv_maxspeed: any larger value
 * is clipped to maxspeed by SV_AirMove anyway. */
#define QNN_INPUT_DEGREES_PER_COUNT 0.066f

qnn_action_t qnn_pending_action;

extern kbutton_t in_attack;
extern kbutton_t in_jump;
extern int in_impulse;

static void QNN_PressButton(kbutton_t *button)
{
	button->state |= 1 + 2;
}

void QNN_ApplyActionLook(const qnn_action_t *action, vec3_t viewangles)
{
	float fwd, yaw_comp, pitch_comp, yaw_deg, pitch_deg;

	fwd = QNN_Clamp(action->look[0], -1.0f, 1.0f);
	yaw_comp = QNN_Clamp(action->look[1], -1.0f, 1.0f);
	pitch_comp = QNN_Clamp(-action->look[2], -1.0f, 1.0f);

	yaw_deg = atan2f(yaw_comp, fwd) * (180.0f / (float)M_PI);
	pitch_deg = atan2f(pitch_comp, fwd) * (180.0f / (float)M_PI);

	if (yaw_deg != 0.0f)
		viewangles[YAW] = anglemod(viewangles[YAW] - yaw_deg);

	if (pitch_deg != 0.0f)
	{
		viewangles[PITCH] += pitch_deg;
		if (viewangles[PITCH] > 80.0f)
			viewangles[PITCH] = 80.0f;
		if (viewangles[PITCH] < -70.0f)
			viewangles[PITCH] = -70.0f;
	}
}

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
	/* 3D view-relative move: (forward, right, up).
	 * Press byte decodes to ±1/0 per axis (QNN_ActionAxisSign), scaled
	 * to sv_maxspeed for the engine's pmove. */
	cmd->forwardmove = (float)QNN_ActionAxisSign(qnn_pending_action.move, 0)
		* QNN_SV_MAXSPEED;
	cmd->sidemove = (float)QNN_ActionAxisSign(qnn_pending_action.move, 1)
		* QNN_SV_MAXSPEED;

	/* View-relative look: look[3] = (forward, right, up) in view frame.
	   look[0] = dot(new_fwd, old_fwd), look[1] = dot(new_fwd, old_right),
	   look[2] = dot(new_fwd, old_up).  atan2(right, forward) recovers the
	   exact yaw delta; atan2(up, forward) recovers pitch.

	   The nonlinear mouse-count curve (QNN_MouseCountFromLookAxis) is
	   bypassed so the model's unit-vector output maps linearly to the
	   turn angle.  BC labels are raw dot products; applying the curve
	   here would under-turn by ~3.3x.  Revisit when PPO needs human-
	   like mouse dynamics — the curve may be reintroduced as a learned
	   or tuned post-processing step at that point. */
	QNN_ApplyActionLook(&qnn_pending_action, cl.viewangles);

	/* Self-state prediction: record the cmd exactly as it leaves — move
	 * values plus the post-turn yaw the message will carry. */
	QNN_PredictRecordCmd(cmd->forwardmove, cmd->sidemove, cl.viewangles[YAW]);

	/* Auto-respawn: QuakeC PlayerDeathThink requires all buttons to be
	   released before it will accept a press to respawn.  Alternate
	   between releasing (even ticks) and pressing (odd ticks) so the
	   server sees a clean edge. */
	if (cl.stats[STAT_HEALTH] <= 0)
	{
		static int respawn_tick = 0;
		respawn_tick++;
		if (respawn_tick & 1)
		{
			QNN_PressButton(&in_attack);
			QNN_PressButton(&in_jump);
		}
		else
		{
			in_attack.state = 0;
			in_jump.state = 0;
		}
	}
	else if (QNN_ActionAttack(qnn_pending_action.move))
		QNN_PressButton(&in_attack);
	else
		in_attack.state = 0;

	/* ud-pos bit = jump (grounded) or swim up (water). */
	if (cl.stats[STAT_HEALTH] > 0 && QNN_ActionAxisSign(qnn_pending_action.move, 2) > 0)
		QNN_PressButton(&in_jump);
	else if (cl.stats[STAT_HEALTH] > 0)
		in_jump.state = 0;

	/* Impulse byte 1..8 is a Quake weapon impulse directly (axe..lightning);
	 * the server's QC rejects impulses for unowned weapons.  Its SOURCE
	 * differs by contract: FULL wires (wire.11/wire.12) carry the held weapon
	 * in `weapon`; the A27 combat wire (wire.13) folds select-and-fire into the
	 * 9-way `attack` (1..8 = that impulse), leaving `weapon` at 0.  Only send
	 * it when it differs from the held weapon: stock W_ChangeWeapon has no
	 * same-weapon guard, so a per-tick impulse re-runs W_SetCurrentAmmo ->
	 * player_run() every server frame, playing the run animation at 4x and
	 * stomping pain frames.  The action log records the raw decided value
	 * (qnn_client_main.c). */
	{
		int impulse = (QNN_IOGetEntityMode() == QNN_ENTITY_MODE_COMBAT)
			? (int)qnn_pending_action.attack
			: (int)qnn_pending_action.weapon;
		if (impulse > 0 && impulse != QNN_WeaponId())
			in_impulse = impulse;
	}
}

void QNN_ArenaApplyLocalAction(const qnn_action_t *action)
{
	usercmd_t cmd;

	qnn_pending_action = *action;
	CL_BaseMove(&cmd);
	IN_Move(&cmd);
	cl.cmd = cmd;

	/* Match CL_SendMove's local edge/impulse consumption without emitting a
	   client-to-server datagram.  The arena server receives this action in the
	   grouped stdin batch instead. */
	in_attack.state &= ~2;
	in_jump.state &= ~2;
	in_impulse = 0;
	QNN_ClearAction(&qnn_pending_action);
}
