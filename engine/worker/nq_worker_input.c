#include "nq_worker.h"

nq_worker_action_t nq_worker_pending_action;

extern kbutton_t in_attack;
extern kbutton_t in_jump;
extern int in_impulse;

static void nq_worker_press_button(kbutton_t *button)
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
	if (nq_worker_pending_action.move == 1)
		cmd->forwardmove += cl_forwardspeed.value;
	else if (nq_worker_pending_action.move == 2)
		cmd->forwardmove -= cl_backspeed.value;

	if (nq_worker_pending_action.strafe == 1)
		cmd->sidemove -= cl_sidespeed.value;
	else if (nq_worker_pending_action.strafe == 2)
		cmd->sidemove += cl_sidespeed.value;

	if (nq_worker_pending_action.look_yaw_count != 0)
		cl.viewangles[YAW] = anglemod(cl.viewangles[YAW] - (0.066f * nq_worker_pending_action.look_yaw_count));

	if (nq_worker_pending_action.look_pitch_count != 0)
	{
		cl.viewangles[PITCH] += 0.066f * nq_worker_pending_action.look_pitch_count;
		if (cl.viewangles[PITCH] > 80.0f)
			cl.viewangles[PITCH] = 80.0f;
		if (cl.viewangles[PITCH] < -70.0f)
			cl.viewangles[PITCH] = -70.0f;
	}

	if (nq_worker_pending_action.fire)
		nq_worker_press_button(&in_attack);
	else
		in_attack.state = 0;

	if (nq_worker_pending_action.jump)
		nq_worker_press_button(&in_jump);
	else
		in_jump.state = 0;

	if (nq_worker_pending_action.weapon > 0)
		in_impulse = nq_worker_pending_action.weapon;
}
