/*
 * qnn_phys.c — BSP-clipped physics simulation for move label inference.
 *
 * Simulates each of the 9 legal key combinations through the real engine
 * physics (friction, accelerate, gravity, BSP collision, mover push) and
 * picks the one whose endpoint best matches the observed position.
 * All candidates start from the same state and go through the same
 * geometry, avoiding the path-divergence problem of baseline subtraction.
 */

#include "qnn.h"
#include "qnn_io.h"

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
extern void SV_ClearWorld(void);
extern void SV_LinkEdict(edict_t *ent, qboolean touch_triggers);
extern void SV_UnlinkEdict(edict_t *ent);
extern void SV_PushMove(edict_t *pusher, float movetime);
extern int pr_edict_size;
extern globalvars_t *pr_global_struct;

#define QNN_PHYS_EDICT_COUNT (2 + QNN_MAX_PHYS_MOVERS + QNN_MAX_PHYS_PLAYERS)
#define QNN_PHYS_PLAYER_BASE (2 + QNN_MAX_PHYS_MOVERS)
static edict_t qnn_phys_edicts[QNN_PHYS_EDICT_COUNT];
static int qnn_phys_mover_count;
static int qnn_phys_player_count;
static client_t qnn_phys_client;
static globalvars_t qnn_phys_globals;

void QNN_PhysInit(void)
{
	memset(qnn_phys_edicts, 0, sizeof(qnn_phys_edicts));

	qnn_phys_edicts[0].v.solid = SOLID_BSP;
	qnn_phys_edicts[0].v.movetype = MOVETYPE_PUSH;
	qnn_phys_edicts[0].v.modelindex = 1;

	pr_edict_size = sizeof(edict_t);
	if (pr_global_struct == NULL)
		pr_global_struct = &qnn_phys_globals;
	sv.edicts = qnn_phys_edicts;
	sv.num_edicts = 2;
	sv.max_edicts = QNN_PHYS_EDICT_COUNT;
	sv.worldmodel = cl.worldmodel;

	/* Populate sv.models from client precache so BSP collision works
	 * with brush submodels (*1, *2, …) used by movers. */
	{
		int i;
		for (i = 1; i < MAX_MODELS; i++)
			sv.models[i] = cl.model_precache[i];
	}

	SV_ClearWorld();
	qnn_phys_mover_count = 0;
	qnn_phys_player_count = 0;
	sv_player = &qnn_phys_edicts[1];

	memset(&qnn_phys_client, 0, sizeof(qnn_phys_client));
	host_client = &qnn_phys_client;
	host_client->edict = sv_player;

	sv_friction.value = 4.0f;
	sv_stopspeed.value = 100.0f;
	sv_maxspeed.value = QNN_SV_MAXSPEED;
	sv_accelerate.value = QNN_SV_ACCELERATE;
	sv_edgefriction.value = 2.0f;
	sv_gravity.value = QNN_SV_GRAVITY;
}

static void QNN_PhysSetPlayerState(
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int waterlevel)
{
	/* Preserve groundentity across steps so SV_PushMove can detect
	 * "player standing on mover" from the previous step's trace. */
	float saved_groundentity = sv_player->v.groundentity;

	memset(&sv_player->v, 0, sizeof(sv_player->v));

	VectorCopy(vel, sv_player->v.velocity);
	VectorCopy(origin, sv_player->v.origin);
	VectorCopy(origin, sv_player->v.oldorigin);
	VectorCopy(view_angles, sv_player->v.v_angle);
	VectorCopy(view_angles, sv_player->v.angles);
	sv_player->v.health = 1.0f;
	sv_player->v.movetype = MOVETYPE_WALK;
	sv_player->v.solid = SOLID_SLIDEBOX;
	sv_player->v.waterlevel = (float)waterlevel;
	sv_player->v.view_ofs[2] = 22.0f;
	sv_player->v.mins[0] = QNN_PLAYER_MINS_X;
	sv_player->v.mins[1] = QNN_PLAYER_MINS_Y;
	sv_player->v.mins[2] = QNN_PLAYER_MINS_Z;
	sv_player->v.maxs[0] = QNN_PLAYER_MAXS_X;
	sv_player->v.maxs[1] = QNN_PLAYER_MAXS_Y;
	sv_player->v.maxs[2] = QNN_PLAYER_MAXS_Z;
	sv_player->v.flags = grounded ? FL_ONGROUND : 0;
	sv_player->v.groundentity = saved_groundentity;
}

void QNN_PhysSetupMovers(const qnn_mover_state_t *movers, int count)
{
	int i;

	/* Unlink previous movers. */
	for (i = 0; i < qnn_phys_mover_count; i++)
		SV_UnlinkEdict(&qnn_phys_edicts[2 + i]);

	qnn_phys_mover_count = (count > QNN_MAX_PHYS_MOVERS)
		? QNN_MAX_PHYS_MOVERS : count;

	for (i = 0; i < qnn_phys_mover_count; i++)
	{
		edict_t *e = &qnn_phys_edicts[2 + i];
		model_t *m;

		memset(&e->v, 0, sizeof(e->v));
		e->free = false;
		e->v.solid = SOLID_BSP;
		e->v.movetype = MOVETYPE_PUSH;
		e->v.modelindex = (float)movers[i].model_index;
		VectorCopy(movers[i].origin, e->v.origin);
		VectorCopy(movers[i].velocity, e->v.velocity);

		m = sv.models[movers[i].model_index];
		if (m != NULL)
		{
			VectorCopy(m->mins, e->v.mins);
			VectorCopy(m->maxs, e->v.maxs);
		}

		SV_LinkEdict(e, false);
	}

	sv.num_edicts = 2 + qnn_phys_mover_count;
}

void QNN_PhysPushMovers(float dt)
{
	int i;

	for (i = 0; i < qnn_phys_mover_count; i++)
	{
		edict_t *e = &qnn_phys_edicts[2 + i];
		if (e->v.velocity[0] || e->v.velocity[1] || e->v.velocity[2])
			SV_PushMove(e, dt);
	}
}

void QNN_PhysSetupPlayers(const vec3_t *origins, int count)
{
	int i;

	for (i = 0; i < qnn_phys_player_count; i++)
		SV_UnlinkEdict(&qnn_phys_edicts[QNN_PHYS_PLAYER_BASE + i]);

	qnn_phys_player_count = (count > QNN_MAX_PHYS_PLAYERS)
		? QNN_MAX_PHYS_PLAYERS : count;

	for (i = 0; i < qnn_phys_player_count; i++)
	{
		edict_t *e = &qnn_phys_edicts[QNN_PHYS_PLAYER_BASE + i];

		memset(&e->v, 0, sizeof(e->v));
		e->free = false;
		e->v.solid = SOLID_SLIDEBOX;
		e->v.movetype = MOVETYPE_NONE;
		VectorCopy(origins[i], e->v.origin);
		e->v.mins[0] = QNN_PLAYER_MINS_X;
		e->v.mins[1] = QNN_PLAYER_MINS_Y;
		e->v.mins[2] = QNN_PLAYER_MINS_Z;
		e->v.maxs[0] = QNN_PLAYER_MAXS_X;
		e->v.maxs[1] = QNN_PLAYER_MAXS_Y;
		e->v.maxs[2] = QNN_PLAYER_MAXS_Z;

		SV_LinkEdict(e, false);
	}

	sv.num_edicts = 2 + qnn_phys_mover_count + qnn_phys_player_count;
}

/* Simulate one candidate input direction for one emission window.
 * Sets forwardmove/sidemove from the candidate signs, runs the full
 * engine physics, returns the resulting position. */
static void QNN_PhysCandidateStep(
	int forward_sign, int strafe_sign,
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int waterlevel,
	float dt, vec3_t out_origin)
{
	/* Wishspeed is clipped to sv_maxspeed inside SV_AirMove, so passing
	 * sv_maxspeed exactly is the same as passing anything larger. */
	float run_speed = sv_maxspeed.value;

	QNN_PhysSetPlayerState(vel, origin, view_angles, grounded, waterlevel);
	SV_LinkEdict(sv_player, false);
	host_frametime = (double)dt;

	/* Movers are already pushed once by QNN_PhysBestCandidate before
	 * the candidate loop — don't push again per candidate. */

	/* Set candidate input. */
	memset(&qnn_phys_client, 0, sizeof(qnn_phys_client));
	host_client = &qnn_phys_client;
	host_client->edict = sv_player;
	host_client->cmd.forwardmove =
		(short)(forward_sign > 0 ? run_speed
			: forward_sign < 0 ? -run_speed
			: 0.0f);
	host_client->cmd.sidemove =
		(short)(strafe_sign > 0 ? run_speed
			: strafe_sign < 0 ? -run_speed
			: 0.0f);

	SV_CheckWater(sv_player);
	SV_ClientThink();

	if (sv_player->v.waterlevel < 2 && !((int)sv_player->v.flags & FL_WATERJUMP))
		sv_player->v.velocity[2] -= sv_gravity.value * dt;
	SV_WalkMove(sv_player);

	VectorCopy(sv_player->v.origin, out_origin);
}

/* Test all 9 XY key combinations and return the one whose simulated
 * endpoint is closest to the observed position. */
void QNN_PhysBestCandidate(
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int waterlevel,
	float dt, const vec3_t observed,
	int prev_forward, int prev_strafe,
	int *out_forward, int *out_strafe,
	qboolean *out_unreachable)
{
	static const int candidates[][2] = {
		{ 0,  0},	/* none */
		{ 1,  0},	/* forward */
		{-1,  0},	/* back */
		{ 0, -1},	/* left */
		{ 0,  1},	/* right */
		{ 1, -1},	/* forward+left */
		{ 1,  1},	/* forward+right */
		{-1, -1},	/* back+left */
		{-1,  1},	/* back+right */
	};
	vec3_t ends[9];
	float scores[9];
	float best_dist = 1e30f;
	int best = 0;
	int prev_idx = -1;
	int i;

	/* Find the candidate index matching the previous frame's direction
	 * so we can apply continuity bias on ambiguous frames. */
	for (i = 0; i < 9; i++)
	{
		if (candidates[i][0] == prev_forward && candidates[i][1] == prev_strafe)
		{
			prev_idx = i;
			break;
		}
	}

	/* Push movers once before candidates. SV_PushMove modifies mover
	 * origins in-place, so this must happen exactly once. The player
	 * gets carried if standing on a mover (groundentity check). */
	if (qnn_phys_mover_count > 0)
	{
		/* Need a temporary player edict linked so SV_PushMove can
		 * detect standing-on contact for the first candidate. */
		QNN_PhysSetPlayerState(vel, origin, view_angles, grounded, waterlevel);
		SV_LinkEdict(sv_player, false);
		host_frametime = (double)dt;
		QNN_PhysPushMovers(dt);
	}

	for (i = 0; i < 9; i++)
	{
		float dist;

		QNN_PhysCandidateStep(
			candidates[i][0], candidates[i][1],
			vel, origin, view_angles, grounded, waterlevel,
			dt, ends[i]);

		dist = QNN_DistSq(ends[i], observed);
		scores[i] = dist;
		if (dist < best_dist)
		{
			best_dist = dist;
			best = i;
		}
	}

	/* Score in normalized response space.  The raw endpoint distance
	 * is dominated by momentum at cruising speed — the forward axis
	 * contribution is tiny (~0.25 units) vs the strafe (~0.6 units).
	 * Normalize each axis by its local effect size so both get equal
	 * weight in the comparison.
	 *
	 * Basis: x0 = none endpoint, af = forward - x0, as = right - x0.
	 * Project everything into (forward_coeff, strafe_coeff) space. */
	{
		/* candidates[0]=none, [1]=forward, [4]=right */
		vec3_t x0, af, as;
		float af_sq, as_sq;

		VectorCopy(ends[0], x0);
		VectorSubtract(ends[1], x0, af);  /* forward effect */
		VectorSubtract(ends[4], x0, as);  /* right effect */

		af_sq = DotProduct(af, af);
		as_sq = DotProduct(as, as);

		/* Only use normalized scoring when both axes have measurable
		 * effect.  If a wall kills an axis, fall back to raw distance. */
		if (af_sq > 0.001f && as_sq > 0.001f)
		{
			vec3_t r;
			float u_obs_f, u_obs_s;

			VectorSubtract(observed, x0, r);
			u_obs_f = DotProduct(r, af) / af_sq;
			u_obs_s = DotProduct(r, as) / as_sq;

			best_dist = 1e30f;
			best = 0;
			for (i = 0; i < 9; i++)
			{
				vec3_t d;
				float uf, us, score;

				VectorSubtract(ends[i], x0, d);
				uf = DotProduct(d, af) / af_sq;
				us = DotProduct(d, as) / as_sq;
				score = (uf - u_obs_f) * (uf - u_obs_f)
					+ (us - u_obs_s) * (us - u_obs_s);
				scores[i] = score;
				if (score < best_dist)
				{
					best_dist = score;
					best = i;
				}
			}
		}
	}

	/* Continuity bias: if the previous frame's candidate scores within
	 * a small margin of the winner, prefer it.  This prevents single-
	 * frame jitter at decision boundaries (deceleration transitions,
	 * diagonal ambiguity) without overriding clear direction changes. */
	if (prev_idx >= 0 && prev_idx != best)
	{
		float margin = 0.10f;
		if (scores[prev_idx] - best_dist < margin)
			best = prev_idx;
	}

	/* Unreachable detection: if the observed position is farther from
	 * the best candidate than the candidates are from each other,
	 * an external force (knockback, trigger_push) moved the player
	 * beyond what any input can explain.  Caller should carry forward. */
	{
		float spread = 0.0f;
		for (i = 1; i < 9; i++)
		{
			float d = QNN_DistSq(ends[0], ends[i]);
			if (d > spread)
				spread = d;
		}
		*out_unreachable = (best_dist > spread) ? true : false;
	}

	*out_forward = candidates[best][0];
	*out_strafe = candidates[best][1];
}
