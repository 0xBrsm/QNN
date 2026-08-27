/*
 * qnn_action_look_test.c — prove QNN_ApplyActionLook is the exact inverse of
 * the collect label construction, from ANY view attitude.
 *
 * WHY THIS EXISTS
 *
 * The look action is a UNIT VECTOR: the new forward direction expressed in the
 * current view basis. qnn_collect_main.c builds the training label as
 *
 *     look = (dot(new_fwd, forward), dot(new_fwd, right), dot(new_fwd, up))
 *
 * so the applier's job is exactly to undo that projection. The test is a round
 * trip: aim from A to B, build the label the way collect does, apply it, and
 * require the view to land on B.
 *
 * TWO BUGS LIVED IN THAT FUNCTION and both are pinned here:
 *
 *   E9  — pitch was derived as atan2(pitch_comp, fwd), reusing the yaw
 *         denominator. Exact when look[1] == 0 and wrong in proportion to the
 *         yaw share, blowing up toward ±90° as look[0] -> 0. A PURE-AXIS TEST
 *         PASSES THAT BUG, which is how it went undetected — hence the mixed
 *         yaw+pitch rows below, where the old code put "yaw 80 / up 10" at
 *         -45.4° instead of -10°.
 *
 *   E10 — the recovered angles were applied as increments in absolute
 *         view-angle space, which is only valid from a level view because the
 *         label's basis is tilted by the current pitch. A PURE ZERO-PITCH-START
 *         TEST PASSES THAT BUG — hence the sweep over non-level start
 *         attitudes, where the old code missed by ~10° at pitch -45.
 *
 * The sweep at the end is the actual proof: every combination of start and
 * target attitude over the engine's full legal pitch range must round-trip
 * within tolerance. Named cases above it exist for readable failure output.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../common/qnn.h"

#define PITCH 0
#define YAW   1
#define ROLL  2

/* Engine pitch clamp in QNN_ApplyActionLook; targets outside it cannot round
 * trip by definition, so the sweep stays inside. */
#define PITCH_MIN (-70.0f)
#define PITCH_MAX ( 80.0f)

/* anglemod quantizes yaw to 360/65536 deg (~0.0055); float basis math adds a
 * little more. 0.05 deg is comfortably above both and far below anything
 * behaviourally meaningful. */
#define TOL 0.05f

static int checks, failures;

/* Upstream Quake mathlib, verbatim — the test TU deliberately does not link
 * the client, and qnn_input.c needs these two symbols. */
float anglemod(float a)
{
	a = (float)(360.0 / 65536) * ((int)(a * (65536 / 360.0)) & 65535);
	return a;
}

void AngleVectors(vec3_t angles, vec3_t forward, vec3_t right, vec3_t up)
{
	float angle, sr, sp, sy, cr, cp, cy;

	angle = (float)(angles[YAW] * (M_PI * 2.0 / 360.0));
	sy = sinf(angle); cy = cosf(angle);
	angle = (float)(angles[PITCH] * (M_PI * 2.0 / 360.0));
	sp = sinf(angle); cp = cosf(angle);
	angle = (float)(angles[ROLL] * (M_PI * 2.0 / 360.0));
	sr = sinf(angle); cr = cosf(angle);

	forward[0] = cp * cy;
	forward[1] = cp * sy;
	forward[2] = -sp;
	right[0] = (-1 * sr * sp * cy + -1 * cr * -sy);
	right[1] = (-1 * sr * sp * sy + -1 * cr * cy);
	right[2] = -1 * sr * cp;
	up[0] = (cr * sp * cy + -sr * -sy);
	up[1] = (cr * sp * sy + -sr * cy);
	up[2] = cr * cp;
}

/* Shortest signed separation between two yaw angles, degrees. */
static float yaw_delta(float a, float b)
{
	float d = fmodf(a - b + 540.0f, 360.0f) - 180.0f;
	return d;
}

/* Aim from (p0,y0) to (p1,y1): build the collect label, apply it, return the
 * worst residual in degrees. */
static float round_trip(float p0, float y0, float p1, float y1,
                        float *out_pitch, float *out_yaw)
{
	vec3_t start = {p0, y0, 0.0f}, target = {p1, y1, 0.0f};
	vec3_t fwd, right, up, tf, tr, tu, view;
	qnn_action_t action;
	float ep, ey;

	AngleVectors(start, fwd, right, up);
	AngleVectors(target, tf, tr, tu);

	memset(&action, 0, sizeof(action));
	action.look[0] = DotProduct(tf, fwd);
	action.look[1] = DotProduct(tf, right);
	action.look[2] = DotProduct(tf, up);

	view[PITCH] = p0; view[YAW] = y0; view[ROLL] = 0.0f;
	QNN_ApplyActionLook(&action, view);

	ep = view[PITCH] - p1;
	ey = yaw_delta(view[YAW], y1);
	if (out_pitch) *out_pitch = view[PITCH];
	if (out_yaw) *out_yaw = view[YAW];
	return fabsf(ep) > fabsf(ey) ? fabsf(ep) : fabsf(ey);
}

static void named(const char *label, float p0, float y0, float p1, float y1)
{
	float gp = 0.0f, gy = 0.0f;
	float err = round_trip(p0, y0, p1, y1, &gp, &gy);
	int bad = !(err <= TOL);

	checks++;
	if (bad)
		failures++;
	printf("%-38s (%6.1f,%6.1f)->(%6.1f,%6.1f)  landed (%7.2f,%7.2f)  "
	       "err %6.3f  %s\n",
	       label, p0, y0, p1, y1, gp, gy, err, bad ? "FAIL" : "ok");
}

int main(void)
{
	float p0, y0, p1, y1, worst = 0.0f;
	float worst_case[4] = {0, 0, 0, 0};
	long swept = 0;

	printf("QNN_ApplyActionLook round trip: label -> apply -> view angles\n");
	printf("(start pitch, start yaw) -> (target pitch, target yaw)\n\n");

	/* Pure axes — these pass even with BOTH bugs present. */
	named("pure yaw", 0.0f, 0.0f, 0.0f, 30.0f);
	named("pure pitch up", 0.0f, 0.0f, -30.0f, 0.0f);
	named("pure pitch down", 0.0f, 0.0f, 30.0f, 0.0f);

	/* Mixed from level — these catch E9. */
	named("mixed yaw30/up20", 0.0f, 0.0f, -20.0f, 30.0f);
	named("mixed yaw60/up20", 0.0f, 0.0f, -20.0f, 60.0f);
	named("mixed yaw80/up10 (E9: was -45.4)", 0.0f, 0.0f, -10.0f, 80.0f);
	named("mixed yaw45/down25", 0.0f, 0.0f, 25.0f, 45.0f);

	/* Non-level start — these catch E10. */
	named("pitched start (E10: was -49.6/45.1)", -20.0f, 10.0f, -45.0f, 55.0f);
	named("steep start, big turn", -45.0f, 200.0f, -60.0f, 260.0f);
	named("looking down, turn up", 40.0f, 350.0f, -35.0f, 20.0f);

	/* The live geometry from tmp/vertical.dem. */
	named("vertical.dem geometry", 0.0f, 0.0f, -34.0f, 40.0f);

	/* ── The proof: exhaustive over the legal attitude domain ─────────── */
	for (p0 = PITCH_MIN; p0 <= PITCH_MAX; p0 += 5.0f)
		for (y0 = 0.0f; y0 < 360.0f; y0 += 45.0f)
			for (p1 = PITCH_MIN; p1 <= PITCH_MAX; p1 += 5.0f)
				for (y1 = 0.0f; y1 < 360.0f; y1 += 45.0f)
				{
					float err = round_trip(p0, y0, p1, y1, NULL, NULL);
					swept++;
					if (err > worst)
					{
						worst = err;
						worst_case[0] = p0; worst_case[1] = y0;
						worst_case[2] = p1; worst_case[3] = y1;
					}
				}

	checks++;
	printf("\nsweep: %ld attitude pairs over pitch [%.0f,%.0f] x yaw [0,360)\n",
	       swept, PITCH_MIN, PITCH_MAX);
	printf("  worst residual %.4f deg at (%.0f,%.0f)->(%.0f,%.0f)  %s\n",
	       worst, worst_case[0], worst_case[1], worst_case[2], worst_case[3],
	       worst <= TOL ? "ok" : "FAIL");
	if (!(worst <= TOL))
		failures++;

	printf("\n%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
