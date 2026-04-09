/*
 * qnn_phys.c — Velocity-based movement label inference.
 *
 * Sets up minimal server-side state (dummy sv_player, world edict) so we
 * can call the real engine physics functions (SV_UserFriction, SV_WaterMove)
 * with zero input to predict what velocity
 * would be with no keys pressed.  The residual between predicted and actual
 * velocity tells us which keys were held.
 *
 * This replaces the old position-delta approach which mislabeled knockback,
 * air coasting, and deceleration as intentional movement.
 */

#include "qnn.h"
#include "qnn_io.h"

/* Engine globals from sv_user.c that the physics functions read. */
extern edict_t	*sv_player;
extern float	*origin;
extern float	*velocity;
extern qboolean	onground;
extern usercmd_t cmd;
extern double	host_frametime;

extern cvar_t	sv_friction;
extern cvar_t	sv_edgefriction;
extern cvar_t	sv_stopspeed;
extern cvar_t	sv_maxspeed;
extern cvar_t	sv_accelerate;

extern void SV_UserFriction(void);
extern void SV_WaterMove(void);
extern void SV_ClearWorld(void);

/* Dummy edicts: [0] = world, [1] = player.
   Variable-sized in the engine, but we only need the base struct. */
static edict_t qnn_phys_edicts[2];
static qboolean qnn_phys_initialized = false;

/* Maximum residual from player input per frame, computed from physics.
   Worst case: player at max speed near a ledge (edge friction doubles
   sv_friction).  Residual = friction_loss + acceleration_gain.
   = sv_maxspeed * sv_friction * sv_edgefriction * dt
     + sv_accelerate * sv_maxspeed * dt
   = sv_maxspeed * dt * (sv_friction * sv_edgefriction + sv_accelerate) */

/*
 * QNN_PhysInit — set up minimal server state for physics simulation.
 * Call once after the map is loaded (cl.worldmodel is valid).
 */
void QNN_PhysInit(void)
{
	memset(qnn_phys_edicts, 0, sizeof(qnn_phys_edicts));

	/* World edict [0]: SOLID_BSP, MOVETYPE_PUSH, points at worldmodel. */
	qnn_phys_edicts[0].v.solid = SOLID_BSP;
	qnn_phys_edicts[0].v.movetype = MOVETYPE_PUSH;
	qnn_phys_edicts[0].v.modelindex = 1;

	/* Wire sv.edicts and sv.worldmodel so SV_Move/SV_ClearWorld work. */
	sv.edicts = qnn_phys_edicts;
	sv.num_edicts = 2;
	sv.max_edicts = 2;
	sv.worldmodel = cl.worldmodel;
	sv.models[1] = cl.worldmodel;

	/* Build the area node tree for SV_Move traces. */
	SV_ClearWorld();

	/* Point sv_player at edict [1]. */
	sv_player = &qnn_phys_edicts[1];

	/* Ensure cvars have correct values (they may not be registered
	   during demo playback since the server never started). */
	sv_friction.value = 4.0f;
	sv_stopspeed.value = 100.0f;
	sv_maxspeed.value = 320.0f;
	sv_accelerate.value = 10.0f;
	sv_edgefriction.value = 2.0f;

	qnn_phys_initialized = true;
}

/*
 * QNN_PhysInferMove — Infer movement key labels from velocity change.
 *
 * Simulates one frame of engine physics with zero input on prev_velocity,
 * compares to actual cur_velocity.  The residual is the acceleration
 * from player input, projected onto the facing axes.
 *
 * Writes move[0] (forward/back) and move[1] (strafe) into action.
 */
void QNN_PhysInferMove(
	qnn_action_t *action,
	const vec3_t prev_vel,
	const vec3_t cur_vel,
	const vec3_t prev_origin,
	const vec3_t prev_view_angles,
	qboolean is_grounded,
	int waterlevel,
	float dt)
{
	vec3_t predicted;
	vec3_t residual;
	vec3_t forward, right, up;
	float forward_proj, right_proj;
	float residual_mag;

	if (!qnn_phys_initialized)
		return;

	action->move[0] = 0.0f;
	action->move[1] = 0.0f;

	/* Populate the dummy sv_player from snapshot state. */
	VectorCopy(prev_vel, sv_player->v.velocity);
	VectorCopy(prev_origin, sv_player->v.origin);
	VectorCopy(prev_view_angles, sv_player->v.v_angle);
	sv_player->v.waterlevel = (float)waterlevel;

	/* Player hull for edge friction trace. */
	sv_player->v.mins[0] = QNN_PLAYER_MINS_X;
	sv_player->v.mins[1] = QNN_PLAYER_MINS_Y;
	sv_player->v.mins[2] = QNN_PLAYER_MINS_Z;
	sv_player->v.maxs[0] = QNN_PLAYER_MAXS_X;
	sv_player->v.maxs[1] = QNN_PLAYER_MAXS_Y;
	sv_player->v.maxs[2] = QNN_PLAYER_MAXS_Z;

	if (is_grounded)
		sv_player->v.flags = (float)((int)sv_player->v.flags | FL_ONGROUND);
	else
		sv_player->v.flags = (float)((int)sv_player->v.flags & ~FL_ONGROUND);

	/* Set up the sv_user.c globals that the physics functions read. */
	origin = sv_player->v.origin;
	velocity = sv_player->v.velocity;
	onground = is_grounded;
	host_frametime = (double)dt;

	/* Zero the input command — we want to simulate "no keys pressed". */
	memset(&cmd, 0, sizeof(cmd));

	/* Run the appropriate engine physics path with zero input.
	   Water: SV_WaterMove applies friction + downward drift (cmd=0).
	   Ground: SV_UserFriction applies friction with real BSP edge traces.
	   Air: no friction, velocity carries unchanged. */
	if (waterlevel >= 2)
		SV_WaterMove();
	else if (is_grounded)
		SV_UserFriction();
	/* else: airborne, no friction applied. */

	/* Read back predicted velocity (friction only, no input). */
	VectorCopy(sv_player->v.velocity, predicted);

	/* Residual = actual - predicted = acceleration from input. */
	VectorSubtract(cur_vel, predicted, residual);

	/* Knockback / external force check: if residual exceeds what
	   player input could produce, suppress the label. */
	residual_mag = Length(residual);
	{
		float max_residual = sv_maxspeed.value * dt
			* (sv_friction.value * sv_edgefriction.value + sv_accelerate.value);
		if (residual_mag > max_residual)
			return;
	}

	/* Project residual onto player's horizontal facing axes. */
	AngleVectors(prev_view_angles, forward, right, up);
	forward_proj = residual[0] * forward[0] + residual[1] * forward[1];
	right_proj = residual[0] * right[0] + residual[1] * right[1];

	if (forward_proj > 0.0f)
		action->move[0] = 1.0f;
	else if (forward_proj < 0.0f)
		action->move[0] = -1.0f;

	if (right_proj > 0.0f)
		action->move[1] = 1.0f;
	else if (right_proj < 0.0f)
		action->move[1] = -1.0f;
}
