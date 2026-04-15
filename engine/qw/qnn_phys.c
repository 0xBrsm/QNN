/*
 * qnn_phys.c (qw) — BSP-clipped physics simulation for move label inference.
 *
 * Feature parity with nq/qnn_phys.c.  Uses QW's native PlayerMove()
 * (from pmove.c) instead of the NQ server physics (SV_ClientThink,
 * SV_WalkMove).  This gives correct QW-specific physics:
 *
 *   - Air acceleration 0.7 (vs NQ 10.0)
 *   - Same BSP collision via pmove's hull tracing
 *   - Same friction / ground acceleration / gravity
 *
 * Simulates each of the 9 legal key combinations through the real QW
 * physics and picks the one whose endpoint best matches the observed
 * position — same approach as the NQ worker.
 *
 * Only called in the MVD path — QWD demos extract actions directly from
 * the usercmd_t in dem_cmd messages.
 */

#include "qnn.h"
#include "qnn_io.h"

#include <math.h>
#include <string.h>

/* QW pmove interface — pmove is a global struct, not a pointer.
 * PlayerMove() is the entry point (not PM_PlayerMove, which is a trace).
 * onground/waterlevel are module-level globals in pmove.c. */
extern playermove_t pmove;
extern void PlayerMove(void);
extern movevars_t movevars;
extern int onground;
extern int waterlevel;
extern int watertype;

/* Candidate wishspeed: slightly above maxspeed so the pmove clamp kicks
 * in reliably and the candidate represents "key held to max". */
#define QNN_PHYS_WISHSPEED_OVERSHOOT 80.0f

/* Target sub-step size (ms) for splitting an emit window into multiple
 * pmove calls.  13 ms corresponds to the 77 Hz server tick that QW's
 * friction/accel integrator was tuned against. */
#define QNN_PHYS_SUBSTEP_MSEC 13

static int qnn_phys_mover_count;
static int qnn_phys_player_count;

void QNN_PhysInit(void)
{
	movevars.gravity = QNN_SV_GRAVITY;
	movevars.stopspeed = 100.0f;
	movevars.maxspeed = QNN_SV_MAXSPEED;
	movevars.accelerate = QNN_SV_ACCELERATE;
	movevars.airaccelerate = 0.7f;	/* QW-specific! NQ uses 10.0 */
	movevars.wateraccelerate = 10.0f;
	movevars.friction = 4.0f;
	movevars.waterfriction = 4.0f;
	movevars.entgravity = 1.0f;

	/* Ensure world hull is present. */
	memset(&pmove.physents[0], 0, sizeof(pmove.physents[0]));
	pmove.physents[0].model = cl.worldmodel;
	pmove.numphysent = 1;
	qnn_phys_mover_count = 0;
	qnn_phys_player_count = 0;
}

void QNN_PhysSetupMovers(const qnn_mover_state_t *movers, int count)
{
	/* Add movers as physents so PlayerMove traces against them.
	 * QW pmove doesn't have SV_PushMove, so movers are static
	 * collision hulls — no player-carry behavior. */
	int i, n;

	n = (count > QNN_MAX_PHYS_MOVERS) ? QNN_MAX_PHYS_MOVERS : count;

	for (i = 0; i < n; i++)
	{
		physent_t *pe = &pmove.physents[1 + i];
		memset(pe, 0, sizeof(*pe));
		pe->model = cl.model_precache[movers[i].model_index];
		VectorCopy(movers[i].origin, pe->origin);
	}
	qnn_phys_mover_count = n;
	pmove.numphysent = 1 + n;
}

void QNN_PhysSetupPlayers(const vec3_t *origins, int count)
{
	int base, i, n;

	base = 1 + qnn_phys_mover_count;
	n = (count > QNN_MAX_PHYS_PLAYERS) ? QNN_MAX_PHYS_PLAYERS : count;

	for (i = 0; i < n; i++)
	{
		physent_t *pe = &pmove.physents[base + i];
		memset(pe, 0, sizeof(*pe));
		pe->model = NULL;
		VectorCopy(origins[i], pe->origin);
		pe->mins[0] = QNN_PLAYER_MINS_X;
		pe->mins[1] = QNN_PLAYER_MINS_Y;
		pe->mins[2] = QNN_PLAYER_MINS_Z;
		pe->maxs[0] = QNN_PLAYER_MAXS_X;
		pe->maxs[1] = QNN_PLAYER_MAXS_Y;
		pe->maxs[2] = QNN_PLAYER_MAXS_Z;
	}
	qnn_phys_player_count = n;
	pmove.numphysent = base + n;
}

/* Simulate one candidate input direction for one emission window.
 *
 * QW's PlayerMove integrates friction/accel at usercmd.msec granularity
 * (typical client sends 13 ms sub-steps at 77 Hz).  Running a single
 * 40–60 ms step introduces timing error — the accelerate/friction
 * linearization diverges from the real client's trajectory and makes
 * the 9-candidate endpoints sensitive to whether the emit window
 * happens to span 2 or 3 native frames.  Instead we split `dt` into
 * `substeps` pmove calls of ~native-tick size so each candidate's
 * trajectory matches what the real server saw over the same interval. */
static void QNN_PhysCandidateStep(
	int forward_sign, int strafe_sign,
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int wl,
	float dt, vec3_t out_origin)
{
	float run_speed = QNN_SV_MAXSPEED + QNN_PHYS_WISHSPEED_OVERSHOOT;
	int total_msec;
	int substeps;
	int sub_msec;
	int k;
	int remainder;

	total_msec = (int)(dt * 1000.0f + 0.5f);
	if (total_msec < 1)
		total_msec = 1;
	substeps = (total_msec + QNN_PHYS_SUBSTEP_MSEC / 2) / QNN_PHYS_SUBSTEP_MSEC;
	if (substeps < 1)
		substeps = 1;
	sub_msec = total_msec / substeps;
	if (sub_msec < 1)
		sub_msec = 1;
	remainder = total_msec - sub_msec * substeps;

	VectorCopy(origin, pmove.origin);
	VectorCopy(vel, pmove.velocity);
	VectorCopy(view_angles, pmove.angles);
	pmove.dead = false;
	pmove.spectator = 0;
	onground = grounded ? 1 : 0;
	waterlevel = wl;

	memset(&pmove.cmd, 0, sizeof(pmove.cmd));
	pmove.cmd.forwardmove =
		(short)(forward_sign > 0 ? run_speed
			: forward_sign < 0 ? -run_speed : 0.0f);
	pmove.cmd.sidemove =
		(short)(strafe_sign > 0 ? run_speed
			: strafe_sign < 0 ? -run_speed : 0.0f);

	/* Run substeps sized ~13 ms each; fold the remainder (0..substeps-1 ms)
	 * into the final step so the total simulated time equals dt exactly. */
	for (k = 0; k < substeps; ++k)
	{
		int msec = sub_msec + (k == substeps - 1 ? remainder : 0);
		if (msec < 1)
			msec = 1;
		if (msec > 255)
			msec = 255;
		pmove.cmd.msec = (byte)msec;
		PlayerMove();
	}

	VectorCopy(pmove.origin, out_origin);
}

void QNN_PhysBestCandidate(
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int wl,
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

	for (i = 0; i < 9; i++)
	{
		if (candidates[i][0] == prev_forward && candidates[i][1] == prev_strafe)
		{
			prev_idx = i;
			break;
		}
	}

	/* QW pmove doesn't have SV_PushMove, so movers are already
	 * positioned at correct origins by QNN_PhysSetupMovers. */

	for (i = 0; i < 9; i++)
	{
		float dist;

		QNN_PhysCandidateStep(
			candidates[i][0], candidates[i][1],
			vel, origin, view_angles, grounded, wl,
			dt, ends[i]);

		dist = QNN_DistSq(ends[i], observed);
		scores[i] = dist;
		if (dist < best_dist)
		{
			best_dist = dist;
			best = i;
		}
	}

	/* Normalized response scoring — identical to NQ.
	 * Basis: x0 = none endpoint, af = forward - x0, as = right - x0.
	 * Project into (forward_coeff, strafe_coeff) space. */
	{
		vec3_t x0, af, as;
		float af_sq, as_sq;

		VectorCopy(ends[0], x0);
		VectorSubtract(ends[1], x0, af);
		VectorSubtract(ends[4], x0, as);

		af_sq = DotProduct(af, af);
		as_sq = DotProduct(as, as);

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

	/* Continuity bias — identical to NQ. */
	if (prev_idx >= 0 && prev_idx != best)
	{
		float margin = 0.10f;
		if (scores[prev_idx] - best_dist < margin)
			best = prev_idx;
	}

	/* Unreachable detection: if observed position is farther from the
	 * best candidate than candidates are from each other, an external
	 * force moved the player beyond what any input can explain. */
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
