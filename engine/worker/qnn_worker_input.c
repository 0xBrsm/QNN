#include "qnn_worker.h"

qnn_worker_action_t qnn_worker_pending_action;

extern kbutton_t in_attack;
extern kbutton_t in_jump;
extern int in_impulse;

static void qnn_worker_press_button(kbutton_t *button)
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
	if (qnn_worker_pending_action.move == 1)
		cmd->forwardmove += cl_forwardspeed.value;
	else if (qnn_worker_pending_action.move == 2)
		cmd->forwardmove -= cl_backspeed.value;

	if (qnn_worker_pending_action.strafe == 1)
		cmd->sidemove -= cl_sidespeed.value;
	else if (qnn_worker_pending_action.strafe == 2)
		cmd->sidemove += cl_sidespeed.value;

	if (qnn_worker_pending_action.look_yaw_count != 0)
		cl.viewangles[YAW] = anglemod(cl.viewangles[YAW] - (0.066f * qnn_worker_pending_action.look_yaw_count));

	if (qnn_worker_pending_action.look_pitch_count != 0)
	{
		cl.viewangles[PITCH] += 0.066f * qnn_worker_pending_action.look_pitch_count;
		if (cl.viewangles[PITCH] > 80.0f)
			cl.viewangles[PITCH] = 80.0f;
		if (cl.viewangles[PITCH] < -70.0f)
			cl.viewangles[PITCH] = -70.0f;
	}

	if (qnn_worker_pending_action.fire)
		qnn_worker_press_button(&in_attack);
	else
		in_attack.state = 0;

	if (qnn_worker_pending_action.jump)
		qnn_worker_press_button(&in_jump);
	else
		in_jump.state = 0;

	if (qnn_worker_pending_action.weapon > 0)
		in_impulse = qnn_worker_pending_action.weapon;
}
