/*
 * qnn_predict.c — client-side self-state prediction (Phase 1: horizontal
 * velocity).
 *
 * The model's training contract is server-aligned: obs(t) reflects the
 * command sent at tick t (the human act-then-observe pipeline). The live
 * client pays one structural extra tick — it decides AFTER the frame, so
 * its command rides the NEXT frame's message and the reply snapshot lands
 * one tick later still (measured: cmd sent at tick s is first visible in
 * cl.velocity at tick s+1; see src/docs/move-head.md, lag triage). Raw
 * cl.velocity therefore violates the training semantics by the transport
 * lag, which manufactured the a24d strafe-jitter snap-back and taxes
 * closed-loop aim.
 *
 * Fix: replay the client's own issued commands through a mirror of the
 * server's horizontal player physics (SV_UserFriction / SV_Accelerate /
 * SV_AirAccelerate semantics, including NQ's wishspeed quirk in the air
 * branch) on top of the latest server velocity, and feed THAT to the obs
 * builder. The lag is never assumed: per tick, candidate lags L are scored
 * by how well step(prev_server_vel, cmd_sent(t-L)) explains the observed
 * server velocity (EMA per candidate, argmin wins, tie-break to the
 * current estimate) — the online version of the offline cmd-offset sweep
 * that pinned the lag at 1 tick with a 5x error margin (11.9 vs ~60 ups).
 *
 * Horizontal only by design: sv_gravity is a server cvar NQ does not
 * network, so vertical prediction would guess wrong on low-grav maps;
 * vertical stays raw. Rebasing on the server snapshot every tick bounds
 * error accumulation to the replay window.
 *
 * Env: QNN_CLIENT_PREDICT=0 disables (default on);
 *      QNN_CLIENT_PREDICT_LOG=<path> dumps per-tick JSONL
 *      (raw vs predicted velocity, lag estimate, candidate EMAs).
 */

#include "qnn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define QNN_PRED_CMD_RING   8
#define QNN_PRED_MAX_LAG    4      /* candidates 1..MAX_LAG */
#define QNN_PRED_EMA_ALPHA  0.02f  /* ~50-tick horizon for the lag vote */

/* Server movement constants mirrored from sv_user.c / sv_phys.c defaults.
 * Not networked in NQ; a server running non-defaults shows up as a high
 * EMA floor across ALL lag candidates in the predict log. */
#define QNN_PRED_FRICTION   4.0f
#define QNN_PRED_STOPSPEED  100.0f
#define QNN_PRED_ACCELERATE 10.0f
#define QNN_PRED_AIRCAP     30.0f

typedef struct {
	float fwd, side;   /* cmd forwardmove / sidemove (ups) */
	float yaw;         /* cl.viewangles[YAW] sent with the cmd (deg) */
} qnn_pred_cmd_t;

static struct {
	int            enabled;
	FILE          *log;
	qnn_pred_cmd_t cmds[QNN_PRED_CMD_RING];
	int            cmd_tick;          /* tick index of the newest cmd */
	int            n_cmds;
	float          prev_vel[2];       /* server velocity at the previous tick */
	int            prev_grounded;
	int            has_prev;
	float          ema[QNN_PRED_MAX_LAG + 1];   /* [1..MAX_LAG] */
	int            lag;               /* current estimate L-hat */
	int            tick;
	float          pred[2];           /* this tick's predicted velocity */
	int            has_pred;
} qnn_pred;

/* One horizontal physics step: friction (grounded) + accelerate toward the
 * cmd wishdir. Mirrors the python prototype validated against live paired
 * logs (median 11.9 ups next-tick error vs 45.3 copy-prev). */
static void QNN_PredStep(float vel[2], const qnn_pred_cmd_t *cmd,
                         int grounded, float dt)
{
	float vx = vel[0], vy = vel[1];
	float yaw, fx, fy, rx, ry, wx, wy, wishspeed;

	if (grounded) {
		float speed = sqrtf(vx * vx + vy * vy);
		if (speed > 0.0f) {
			float control = speed > QNN_PRED_STOPSPEED ? speed : QNN_PRED_STOPSPEED;
			float newspeed = speed - control * QNN_PRED_FRICTION * dt;
			if (newspeed < 0.0f) newspeed = 0.0f;
			vx *= newspeed / speed;
			vy *= newspeed / speed;
		}
	}

	yaw = cmd->yaw * (float)M_PI / 180.0f;
	fx = cosf(yaw);  fy = sinf(yaw);
	rx = sinf(yaw);  ry = -cosf(yaw);
	wx = fx * cmd->fwd + rx * cmd->side;
	wy = fy * cmd->fwd + ry * cmd->side;
	wishspeed = sqrtf(wx * wx + wy * wy);
	if (wishspeed > 1e-6f) {
		float wnx = wx / wishspeed, wny = wy / wishspeed;
		float target, cur, add;
		if (wishspeed > QNN_SV_MAXSPEED) wishspeed = QNN_SV_MAXSPEED;
		/* NQ quirk: the air branch caps the TARGET at 30 ups but scales
		 * the accel step by the UNCAPPED wishspeed (SV_AirAccelerate). */
		target = grounded ? wishspeed
		                  : (wishspeed > QNN_PRED_AIRCAP ? QNN_PRED_AIRCAP : wishspeed);
		cur = vx * wnx + vy * wny;
		add = target - cur;
		if (add > 0.0f) {
			float accel = QNN_PRED_ACCELERATE * wishspeed * dt;
			if (accel > add) accel = add;
			vx += accel * wnx;
			vy += accel * wny;
		}
	}

	vel[0] = vx;
	vel[1] = vy;
}

static const qnn_pred_cmd_t *QNN_PredCmdAt(int tick)
{
	int back = qnn_pred.cmd_tick - tick;
	if (back < 0 || back >= qnn_pred.n_cmds || back >= QNN_PRED_CMD_RING)
		return NULL;
	return &qnn_pred.cmds[(qnn_pred.cmd_tick - back) & (QNN_PRED_CMD_RING - 1)];
}

void QNN_PredictInit(void)
{
	const char *gate = getenv("QNN_CLIENT_PREDICT");
	const char *log_path = getenv("QNN_CLIENT_PREDICT_LOG");

	qnn_pred.enabled = !(gate != NULL && gate[0] == '0');
	qnn_pred.lag = 1;   /* the measured operating point; the EMA vote owns it from here */
	qnn_pred.cmd_tick = -1;
	if (log_path != NULL && log_path[0] != 0) {
		qnn_pred.log = fopen(log_path, "w");
		if (qnn_pred.log == NULL)
			fprintf(stderr, "qnn_predict: failed to open log %s\n", log_path);
	}
	fprintf(stderr, "qnn_predict: self-state prediction %s\n",
		qnn_pred.enabled ? "ON (horizontal velocity)" : "OFF");
}

void QNN_PredictReset(void)
{
	qnn_pred.n_cmds = 0;
	qnn_pred.cmd_tick = -1;
	qnn_pred.has_prev = 0;
	qnn_pred.has_pred = 0;
	qnn_pred.tick = 0;
	/* keep ema + lag: transport doesn't change across respawns */
}

/* Called from IN_Move with the cmd actually sent this Host_Frame. */
void QNN_PredictRecordCmd(float fwd, float side, float yaw_deg)
{
	qnn_pred_cmd_t *slot;

	if (!qnn_pred.enabled)
		return;
	qnn_pred.cmd_tick += 1;
	slot = &qnn_pred.cmds[qnn_pred.cmd_tick & (QNN_PRED_CMD_RING - 1)];
	slot->fwd = fwd;
	slot->side = side;
	slot->yaw = yaw_deg;
	if (qnn_pred.n_cmds < QNN_PRED_CMD_RING)
		qnn_pred.n_cmds += 1;
}

/* Per tick, after Host_Frame: update the lag vote from the newly observed
 * server velocity, then replay the unreflected cmds onto it. */
void QNN_PredictTick(float dt)
{
	float sv[2];
	int grounded = cl.onground ? 1 : 0;
	int L, best;

	if (!qnn_pred.enabled)
		return;

	sv[0] = cl.velocity[0];
	sv[1] = cl.velocity[1];
	qnn_pred.tick += 1;

	/* lag vote: which cmd offset best explains prev_vel -> sv?
	 * (cmd applied between tick-1 and tick is the one sent at tick-L) */
	if (qnn_pred.has_prev) {
		for (L = 1; L <= QNN_PRED_MAX_LAG; ++L) {
			const qnn_pred_cmd_t *cmd = QNN_PredCmdAt(qnn_pred.cmd_tick - L);
			float trial[2], err;
			if (cmd == NULL)
				continue;
			trial[0] = qnn_pred.prev_vel[0];
			trial[1] = qnn_pred.prev_vel[1];
			QNN_PredStep(trial, cmd, qnn_pred.prev_grounded, dt);
			err = sqrtf((trial[0] - sv[0]) * (trial[0] - sv[0])
			          + (trial[1] - sv[1]) * (trial[1] - sv[1]));
			qnn_pred.ema[L] += QNN_PRED_EMA_ALPHA * (err - qnn_pred.ema[L]);
		}
		best = qnn_pred.lag;
		for (L = 1; L <= QNN_PRED_MAX_LAG; ++L) {
			if (qnn_pred.ema[L] < qnn_pred.ema[best] - 0.5f)
				best = L;   /* hysteresis: switch only on a clear margin */
		}
		qnn_pred.lag = best;
	}
	qnn_pred.prev_vel[0] = sv[0];
	qnn_pred.prev_vel[1] = sv[1];
	qnn_pred.prev_grounded = grounded;
	qnn_pred.has_prev = 1;

	/* replay: sv reflects cmds sent <= cmd_tick - lag; training semantics
	 * want cmds <= cmd_tick reflected -> replay the newest `lag` cmds. */
	qnn_pred.pred[0] = sv[0];
	qnn_pred.pred[1] = sv[1];
	for (L = qnn_pred.lag - 1; L >= 0; --L) {
		const qnn_pred_cmd_t *cmd = QNN_PredCmdAt(qnn_pred.cmd_tick - L);
		if (cmd == NULL)
			continue;
		QNN_PredStep(qnn_pred.pred, cmd, grounded, dt);
	}
	qnn_pred.has_pred = 1;

	if (qnn_pred.log != NULL) {
		fprintf(qnn_pred.log,
			"{\"t\":%d,\"raw\":[%.1f,%.1f],\"pred\":[%.1f,%.1f],"
			"\"lag\":%d,\"ema\":[%.1f,%.1f,%.1f,%.1f]}\n",
			qnn_pred.tick, sv[0], sv[1],
			qnn_pred.pred[0], qnn_pred.pred[1], qnn_pred.lag,
			qnn_pred.ema[1], qnn_pred.ema[2], qnn_pred.ema[3], qnn_pred.ema[4]);
		fflush(qnn_pred.log);
	}
}

/* Overwrite the snapshot's horizontal velocity with the prediction
 * (vertical stays raw — sv_gravity is not networked). */
void QNN_PredictSelfVelocity(vec3_t vel)
{
	if (!qnn_pred.enabled || !qnn_pred.has_pred)
		return;
	vel[0] = qnn_pred.pred[0];
	vel[1] = qnn_pred.pred[1];
}
