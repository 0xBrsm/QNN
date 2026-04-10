/*
 * qnn_phys.c — Emission-window movement label inference.
 *
 * At each resample emission point, sub-steps the engine physics at the
 * native tick rate for each of the 9 legal movement key combinations.
 * The candidate whose BSP-clipped end state best matches the observed
 * transition wins.
 *
 * Candidate acceptance is entirely score-based: "none" is simulated like
 * every other candidate, and it only loses when some input candidate
 * produces a strictly better BSP-clipped end state match.
 */

#include "qnn.h"
#include "qnn_io.h"

#include <float.h>
#include <math.h>

extern edict_t	*sv_player;
extern client_t	*host_client;
extern double	host_frametime;

extern cvar_t	sv_friction;
extern cvar_t	sv_edgefriction;
extern cvar_t	sv_stopspeed;
extern cvar_t	sv_maxspeed;
extern cvar_t	sv_accelerate;
extern cvar_t	sv_gravity;

extern qboolean SV_CheckWater(edict_t *ent);
extern void SV_ClientThink(void);
extern void SV_WalkMove(edict_t *ent);

extern trace_t SV_Move(vec3_t start, vec3_t mins, vec3_t maxs, vec3_t end, int type, edict_t *passedict);
extern void SV_ClearWorld(void);
extern int pr_edict_size;
extern globalvars_t *pr_global_struct;

typedef struct
{
	float	move[2];
	int	forward_sign;
	int	strafe_sign;
} qnn_phys_candidate_t;

typedef struct
{
	vec3_t	origin;
	vec3_t	velocity;
	qboolean grounded;
	int	waterlevel;
} qnn_phys_result_t;

static const qnn_phys_candidate_t qnn_phys_candidates[] =
{
	{{ 0.0f,  0.0f},  0,  0},	/* none */
	{{ 1.0f,  0.0f},  1,  0},	/* forward */
	{{-1.0f,  0.0f}, -1,  0},	/* back */
	{{ 0.0f, -1.0f},  0, -1},	/* left */
	{{ 0.0f,  1.0f},  0,  1},	/* right */
	{{ 1.0f, -1.0f},  1, -1},	/* forward+left */
	{{ 1.0f,  1.0f},  1,  1},	/* forward+right */
	{{-1.0f, -1.0f}, -1, -1},	/* back+left */
	{{-1.0f,  1.0f}, -1,  1},	/* back+right */
};

#define QNN_PHYS_NUM_CANDIDATES \
	(int)(sizeof(qnn_phys_candidates) / sizeof(qnn_phys_candidates[0]))

static edict_t qnn_phys_edicts[2];
static client_t qnn_phys_client;
static globalvars_t qnn_phys_globals;
static qboolean qnn_phys_initialized = false;

static float QNN_PhysHorizontalDistance(const vec3_t a, const vec3_t b)
{
	float dx, dy;

	dx = a[0] - b[0];
	dy = a[1] - b[1];
	return (float)sqrt(dx * dx + dy * dy);
}

static void QNN_PhysSetPlayerState(
	const vec3_t prev_vel,
	const vec3_t prev_origin,
	const vec3_t cmd_view_angles,
	qboolean prev_grounded,
	int prev_waterlevel)
{
	link_t saved_area;
	int saved_leafs;

	/* Preserve area node linkage — zeroing the link pointers causes
	 * SV_LinkEdict to skip the unlink and leak duplicate entries. */
	saved_area = sv_player->area;
	saved_leafs = sv_player->num_leafs;

	memset(&sv_player->v, 0, sizeof(sv_player->v));

	sv_player->area = saved_area;
	sv_player->num_leafs = saved_leafs;

	VectorCopy(prev_vel, sv_player->v.velocity);
	VectorCopy(prev_origin, sv_player->v.origin);
	VectorCopy(prev_origin, sv_player->v.oldorigin);
	VectorCopy(cmd_view_angles, sv_player->v.v_angle);
	VectorCopy(cmd_view_angles, sv_player->v.angles);
	sv_player->v.health = 1.0f;
	sv_player->v.movetype = MOVETYPE_WALK;
	sv_player->v.solid = SOLID_SLIDEBOX;
	sv_player->v.waterlevel = (float)prev_waterlevel;
	sv_player->v.view_ofs[2] = 22.0f;
	sv_player->v.mins[0] = QNN_PLAYER_MINS_X;
	sv_player->v.mins[1] = QNN_PLAYER_MINS_Y;
	sv_player->v.mins[2] = QNN_PLAYER_MINS_Z;
	sv_player->v.maxs[0] = QNN_PLAYER_MAXS_X;
	sv_player->v.maxs[1] = QNN_PLAYER_MAXS_Y;
	sv_player->v.maxs[2] = QNN_PLAYER_MAXS_Z;
	sv_player->v.flags = prev_grounded ? FL_ONGROUND : 0.0f;
}

static void QNN_PhysSetCommand(const qnn_phys_candidate_t *candidate)
{
	/* Use values above sv_maxspeed so SV_AirMove clamps wishspeed to 320.
	 * Real players always hold +speed, which doubles cl_forwardspeed from
	 * 200 to 400.  The old code used cl_forwardspeed (200), making wishspeed
	 * only 200 — too low to distinguish "holding forward" from "no input"
	 * at cruising speed, since SV_Accelerate returns when addspeed <= 0. */
	float run_speed = sv_maxspeed.value + 80.0f;	/* 400, same as +speed */

	memset(&qnn_phys_client, 0, sizeof(qnn_phys_client));
	host_client = &qnn_phys_client;
	host_client->edict = sv_player;
	host_client->cmd.forwardmove =
		(short)(candidate->forward_sign > 0 ? run_speed
			: candidate->forward_sign < 0 ? -run_speed
			: 0.0f);
	host_client->cmd.sidemove =
		(short)(candidate->strafe_sign > 0 ? run_speed
			: candidate->strafe_sign < 0 ? -run_speed
			: 0.0f);
	host_client->cmd.upmove = 0;
}

static void QNN_PhysApplyJump(qboolean jump)
{
	if (!jump || ((int)sv_player->v.flags & FL_WATERJUMP))
		return;

	if (sv_player->v.waterlevel >= 2)
	{
		if ((int)sv_player->v.watertype == CONTENTS_WATER)
			sv_player->v.velocity[2] = 100.0f;
		else if ((int)sv_player->v.watertype == CONTENTS_SLIME)
			sv_player->v.velocity[2] = 80.0f;
		else
			sv_player->v.velocity[2] = 50.0f;
		return;
	}

	if (!((int)sv_player->v.flags & FL_ONGROUND))
		return;

	sv_player->v.flags = (float)((int)sv_player->v.flags & ~FL_ONGROUND);
	sv_player->v.velocity[2] += 270.0f;
}

static void QNN_PhysSimulateCandidate(
	const qnn_phys_candidate_t *candidate,
	qboolean jump,
	const vec3_t prev_vel,
	const vec3_t prev_origin,
	const vec3_t cmd_view_angles,
	qboolean prev_grounded,
	int prev_waterlevel,
	float dt,
	float native_dt,
	qnn_phys_result_t *out)
{
	float remaining;
	float step_dt;

	QNN_PhysSetPlayerState(prev_vel, prev_origin, cmd_view_angles,
		prev_grounded, prev_waterlevel);

	if (pr_global_struct == NULL)
		pr_global_struct = &qnn_phys_globals;

	/* Sub-step at native tick rate to match the actual game physics.
	 * The game runs friction+accelerate per native frame (~59Hz), so
	 * a single big step at emission rate (~20Hz) diverges significantly
	 * at cruising speed where friction/accel interaction is non-linear. */
	remaining = dt;
	while (remaining > 0.0001f)
	{
		step_dt = (remaining > native_dt * 1.5f) ? native_dt : remaining;
		remaining -= step_dt;
		host_frametime = (double)step_dt;

		SV_CheckWater(sv_player);
		QNN_PhysSetCommand(candidate);
		if (jump)
		{
			QNN_PhysApplyJump(true);
			jump = false;	/* only apply jump on first sub-step */
		}
		SV_ClientThink();
		if (!SV_CheckWater(sv_player) && !((int)sv_player->v.flags & FL_WATERJUMP))
			sv_player->v.velocity[2] -= sv_gravity.value * (float)step_dt;
		SV_WalkMove(sv_player);
	}

	VectorCopy(sv_player->v.origin, out->origin);
	VectorCopy(sv_player->v.velocity, out->velocity);
	out->grounded = ((int)sv_player->v.flags & FL_ONGROUND) ? true : false;
	out->waterlevel = (int)sv_player->v.waterlevel;
}

/* Compare BSP-clipped candidate end states directly against the observed
 * BSP-clipped transition.  "none" participates like any other candidate. */
static float QNN_PhysCandidateScore(
	const qnn_phys_result_t *result,
	const vec3_t cur_origin,
	const vec3_t cur_vel,
	qboolean cur_grounded,
	int cur_waterlevel)
{
	float score;

	/* Score on horizontal position only — Z is unreliable because
	 * we zero Z velocity to avoid double-applied gravity, so the
	 * simulation can't match observed Z transitions (jumps/falls). */
	score = QNN_PhysHorizontalDistance(result->origin, cur_origin);
	score += 0.02f * QNN_PhysHorizontalDistance(result->velocity, cur_vel);
	score += 120.0f * (float)fabs((float)result->waterlevel - (float)cur_waterlevel);
	if (result->grounded != cur_grounded)
		score += 40.0f;
	return score;
}

void QNN_PhysInit(void)
{
	memset(qnn_phys_edicts, 0, sizeof(qnn_phys_edicts));
	memset(&qnn_phys_client, 0, sizeof(qnn_phys_client));

	/* World edict [0]: SOLID_BSP, MOVETYPE_PUSH, points at worldmodel. */
	qnn_phys_edicts[0].v.solid = SOLID_BSP;
	qnn_phys_edicts[0].v.movetype = MOVETYPE_PUSH;
	qnn_phys_edicts[0].v.modelindex = 1;

	/* Wire sv.edicts and sv.worldmodel so SV_Move/SV_ClearWorld work.
	 * pr_edict_size must match our static edict array stride.
	 * pr_global_struct must be valid for SV_Impact (writes time/self). */
	pr_edict_size = sizeof(edict_t);
	if (pr_global_struct == NULL)
		pr_global_struct = &qnn_phys_globals;
	sv.edicts = qnn_phys_edicts;
	sv.num_edicts = 2;
	sv.max_edicts = 2;
	sv.worldmodel = cl.worldmodel;
	sv.models[1] = cl.worldmodel;

	SV_ClearWorld();
	sv_player = &qnn_phys_edicts[1];
	host_client = &qnn_phys_client;

	/* Ensure cvars have correct values even without a running server. */
	sv_friction.value = 4.0f;
	sv_stopspeed.value = 100.0f;
	sv_maxspeed.value = 320.0f;
	sv_accelerate.value = 10.0f;
	sv_edgefriction.value = 2.0f;
	sv_gravity.value = 800.0f;

	qnn_phys_initialized = true;
}

/*
 * QNN_PhysInferMove — Infer movement labels for one emission window.
 *
 * Called once per resampled emission (not per native frame).  Simulates
 * each candidate key combination for the full window dt, using BSP-clipped
 * physics, and picks the best match against the observed transition.
 */
void QNN_PhysInferMove(
	qnn_action_t *action,
	const vec3_t prev_vel,
	const vec3_t cur_vel,
	const vec3_t prev_origin,
	const vec3_t cmd_view_angles,
	qboolean prev_grounded,
	int prev_waterlevel,
	const vec3_t cur_origin,
	qboolean cur_grounded,
	int cur_waterlevel,
	float dt,
	float native_dt,
	float movement_threshold)
{
	float best_score;
	const qnn_phys_candidate_t *best_candidate;
	int i;

	(void)movement_threshold;

	if (!qnn_phys_initialized || dt <= 0.0f)
		return;

	action->move[0] = 0.0f;
	action->move[1] = 0.0f;

	best_candidate = &qnn_phys_candidates[0];
	best_score = FLT_MAX;

	for (i = 0; i < QNN_PHYS_NUM_CANDIDATES; i++)
	{
		qnn_phys_result_t candidate_result;
		float score;

		QNN_PhysSimulateCandidate(&qnn_phys_candidates[i], false,
			prev_vel, prev_origin, cmd_view_angles,
			prev_grounded, prev_waterlevel, dt, native_dt,
			&candidate_result);
		score = QNN_PhysCandidateScore(&candidate_result,
			cur_origin, cur_vel, cur_grounded, cur_waterlevel);
		if (score < best_score)
		{
			best_score = score;
			best_candidate = &qnn_phys_candidates[i];
		}
	}

	action->move[0] = best_candidate->move[0];
	action->move[1] = best_candidate->move[1];
}
