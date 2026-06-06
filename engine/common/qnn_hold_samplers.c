/*
 * qnn_hold_samplers.c — Per-weapon fire and jump button-hold CDF
 * samplers fit on QWD-truth.
 *
 * Each CDF maps a uniform [0, 65535] xorshift32 sample to a hold-tail
 * length in emit frames.  Fits are truncated log-normals; the
 * parameters and the underlying QWD corpus are documented in
 * src/docs/input-inference.md.
 *
 * Cross-references:
 *   - qnn_mvd_collect.c — the only consumer.  Sample is drawn after a
 *     sound-event back-shift writes fire=1 (or move[2]=jump) into the
 *     ring; the returned tail extends the label forward across
 *     subsequent emit slots.
 */

#include "qnn_hold_samplers.h"

#include <stddef.h>


/* ── PRNG ─────────────────────────────────────────────────────────── */

static uint32_t qnn_xorshift32(uint32_t *state)
{
	*state ^= *state << 13;
	*state ^= *state >> 17;
	*state ^= *state << 5;
	return *state;
}


/* ── Fire-hold CDFs (per weapon) ──────────────────────────────────── */

typedef struct { int cd; int n; const uint16_t *cdf; } qnn_hold_cdf_t;

static const uint16_t qnn_hold_cdf_axe[9] = {
	/* Axe (wid=3, mu=0.5992, sigma=0.4241, cd=10) */
	21177, 50622, 61493, 64460, 65238, 65450, 65511, 65529, 65535
};
static const uint16_t qnn_hold_cdf_sg[9] = {
	/* SG (wid=4, mu=0.9943, sigma=0.6484, cd=10) */
	11988, 30261, 43975, 52713, 58074, 61363, 63407, 64701, 65535
};
static const uint16_t qnn_hold_cdf_ssg[13] = {
	/* SSG (wid=5, mu=1.1702, sigma=0.6640, cd=14) */
	8156, 23263, 36503, 46045, 52528, 56869, 59784, 61760, 63115, 64057,
	64721, 65194, 65535
};
static const uint16_t qnn_hold_cdf_gl[11] = {
	/* GL (wid=8, mu=1.0066, sigma=0.6201, cd=12) */
	10834, 29162, 43257, 52190, 57577, 60808, 62767, 63975, 64734, 65219,
	65535
};
static const uint16_t qnn_hold_cdf_rl[15] = {
	/* RL (wid=9, mu=1.0557, sigma=0.5880, cd=16) */
	8743, 26625, 41419, 51014, 56804, 60241, 62290, 63529, 64290, 64766,
	65068, 65263, 65392, 65477, 65535
};

/* Jump hold CDF.  Source: ud=2 runs from QWD truth, split at every
 * additional ground→air transition within the run (each transition is
 * a new engine jump).  Log-normal fit (μ=1.0008, σ=0.6395) truncated to
 * [1, 19] frames at 20 Hz. */
static const uint16_t qnn_hold_cdf_jump[19] = {
	11323, 29206, 42761, 51399, 56679, 59899, 61889, 63139, 63939, 64461,
	64807, 65040, 65200, 65311, 65389, 65444, 65484, 65514, 65535
};

static const qnn_hold_cdf_t *QNN_HoldCDF(int weapon_id)
{
	static const qnn_hold_cdf_t descs[8] = {
		{10,  9, qnn_hold_cdf_axe},  /* wid 3 */
		{10,  9, qnn_hold_cdf_sg},   /* wid 4 */
		{14, 13, qnn_hold_cdf_ssg},  /* wid 5 */
		{ 0,  0, NULL},              /* wid 6 NG  — continuous */
		{ 0,  0, NULL},              /* wid 7 SNG — continuous */
		{12, 11, qnn_hold_cdf_gl},   /* wid 8 */
		{16, 15, qnn_hold_cdf_rl},   /* wid 9 */
		{ 0,  0, NULL},              /* wid 10 LG — continuous */
	};
	if (weapon_id < 3 || weapon_id > 10)
		return NULL;
	return (descs[weapon_id - 3].cdf != NULL) ? &descs[weapon_id - 3] : NULL;
}

static int QNN_SampleFireHold(int weapon_id, uint32_t *rng)
{
	const qnn_hold_cdf_t *d = QNN_HoldCDF(weapon_id);
	uint32_t r;
	int i;

	if (!d)
		return 1;
	r = qnn_xorshift32(rng) & 0xFFFF;
	for (i = 0; i < d->n; i++)
		if (r <= (uint32_t)d->cdf[i])
			return i + 1;
	return d->cd - 1;
}


/* ── Fire cooldown (single source of truth, used by chain gate too) ── */

int QNN_FireCooldownEmit(int weapon_id)
{
	switch (weapon_id)
	{
	case 3:  return 10;  /* Axe — 0.5 s */
	case 4:  return 10;  /* SG  — 0.5 s */
	case 5:  return 14;  /* SSG — 0.7 s */
	case 6:  return  4;  /* NG  — 0.2 s */
	case 7:  return  4;  /* SNG — 0.2 s */
	case 8:  return 12;  /* GL  — 0.6 s */
	case 9:  return 16;  /* RL  — 0.8 s */
	case 10: return  2;  /* LG  — 0.1 s */
	default: return  0;
	}
}


/* ── Public hold-sample API ───────────────────────────────────────── */

/* Auto-refire weapons whose QC W_Attack re-enters every cooldown while
 * the trigger is held.  Continuous → fixed near-cd tail.  Tap → sampled
 * log-normal. */
static int QNN_FireIsContinuous(int weapon_id)
{
	return (weapon_id == 6 || weapon_id == 7 || weapon_id == 10);
}

int QNN_FireHoldFrames(int weapon_id, uint32_t *rng)
{
	int cd = QNN_FireCooldownEmit(weapon_id);
	int ext;

	if (QNN_FireIsContinuous(weapon_id))
		ext = cd;
	else if (QNN_HoldCDF(weapon_id))
		ext = QNN_SampleFireHold(weapon_id, rng);
	else
		return 0;
	if (cd > 1 && ext > cd - 1)
		ext = cd - 1;
	return ext;
}

int QNN_JumpHoldFrames(uint32_t *rng)
{
	uint32_t r;
	int i;
	int n = (int)(sizeof(qnn_hold_cdf_jump) / sizeof(qnn_hold_cdf_jump[0]));

	r = qnn_xorshift32(rng) & 0xFFFF;
	for (i = 0; i < n; i++)
		if (r <= (uint32_t)qnn_hold_cdf_jump[i])
			return i + 1;
	return n;
}
