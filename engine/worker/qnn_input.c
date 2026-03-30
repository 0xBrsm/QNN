#include "qnn.h"

#define QNN_INPUT_MOVE_DEADZONE 0.25f
#define QNN_INPUT_DEGREES_PER_COUNT 0.066f

qnn_action_t qnn_pending_action;

extern kbutton_t in_attack;
extern kbutton_t in_jump;
extern int in_impulse;

static void QNN_PressButton(kbutton_t *button)
{
	button->state |= 1 + 2;
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
	if (qnn_pending_action.move[0] > QNN_INPUT_MOVE_DEADZONE)
		cmd->forwardmove += cl_forwardspeed.value;
	else if (qnn_pending_action.move[0] < -QNN_INPUT_MOVE_DEADZONE)
		cmd->forwardmove -= cl_backspeed.value;

	if (qnn_pending_action.move[1] < -QNN_INPUT_MOVE_DEADZONE)
		cmd->sidemove -= cl_sidespeed.value;
	else if (qnn_pending_action.move[1] > QNN_INPUT_MOVE_DEADZONE)
		cmd->sidemove += cl_sidespeed.value;

	{
		int look_yaw_count = QNN_MouseCountFromLookAxis(qnn_pending_action.look[0]);
		int look_pitch_count = QNN_MouseCountFromLookAxis(qnn_pending_action.look[1]);

		if (look_yaw_count != 0)
			cl.viewangles[YAW] = anglemod(cl.viewangles[YAW] - (QNN_INPUT_DEGREES_PER_COUNT * look_yaw_count));

		if (look_pitch_count != 0)
		{
			cl.viewangles[PITCH] += QNN_INPUT_DEGREES_PER_COUNT * look_pitch_count;
			if (cl.viewangles[PITCH] > 80.0f)
				cl.viewangles[PITCH] = 80.0f;
			if (cl.viewangles[PITCH] < -70.0f)
				cl.viewangles[PITCH] = -70.0f;
		}
	}

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
	else if (qnn_pending_action.fire)
		QNN_PressButton(&in_attack);
	else
		in_attack.state = 0;

	if (cl.stats[STAT_HEALTH] > 0 && qnn_pending_action.jump)
		QNN_PressButton(&in_jump);
	else if (cl.stats[STAT_HEALTH] > 0)
		in_jump.state = 0;

	if (qnn_pending_action.switch_slot > 0)
		in_impulse = QNN_SwitchImpulseFromSlot(qnn_pending_action.switch_slot, cl.items);
}
