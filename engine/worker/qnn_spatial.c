/*
 * qnn_spatial.c — Spatial token construction via BSP raycasting.
 *
 * 9 sectors: FOV center/left/right, flank left/right, rear left/right,
 * ground, ceiling.  Each sector samples 5 rays and accumulates distance,
 * surface type, clearance, traversability, and drop-off metrics.
 */

#include "qnn_io.h"

#include <math.h>
#include <string.h>

/* ── BSP trace helpers ─────────────────────────────────────────── */

static int QNN_TraceContents(const vec3_t point)
{
	mleaf_t *leaf;

	if (cl.worldmodel == NULL)
		return CONTENTS_EMPTY;
	leaf = Mod_PointInLeaf((float *)point, cl.worldmodel);
	if (leaf == NULL)
		return CONTENTS_EMPTY;
	return leaf->contents;
}

static float QNN_TraceLineDistance(const vec3_t start, const vec3_t end, vec3_t impact)
{
	trace_t trace;
	vec3_t delta;

	if (!cl.worldmodel)
	{
		VectorCopy(end, impact);
		VectorSubtract(end, start, delta);
		return QNN_VecLength(delta);
	}
	memset(&trace, 0, sizeof(trace));
	SV_RecursiveHullCheck(cl.worldmodel->hulls, 0, 0, 1, (float *)start, (float *)end, &trace);
	VectorCopy(trace.endpos, impact);
	VectorSubtract(trace.endpos, start, delta);
	return QNN_VecLength(delta);
}

/* ── Spatial token internals ───────────────────────────────────── */

static void QNN_SpatialReset(qnn_spatial_token_t *token, int sector_id)
{
	memset(token, 0, sizeof(*token));
	token->sector_id = sector_id;
}

static void QNN_SpatialFinalize(qnn_spatial_token_t *token, int samples, float max_dist)
{
	if (samples <= 0)
		return;
	token->mean_dist /= (float)samples;
	token->openness = QNN_Clamp(token->mean_dist / max_dist, 0.0f, 1.0f);
	token->solid_frac /= (float)samples;
	token->water_frac /= (float)samples;
	token->slime_frac /= (float)samples;
	token->lava_frac /= (float)samples;
	token->traversable /= (float)samples;
	token->dropoff /= (float)samples;
	token->clearance /= (float)samples;
}

static void QNN_SpatialSampleRay(qnn_spatial_token_t *token, const vec3_t start, const vec3_t dir, float max_dist)
{
	vec3_t end;
	vec3_t impact;
	vec3_t impact_probe;
	float dist;
	int contents;

	VectorMA(start, max_dist, dir, end);
	dist = QNN_TraceLineDistance(start, end, impact);
	token->mean_dist += dist;
	if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
		token->nearest_dist = dist;

	VectorCopy(impact, impact_probe);
	VectorMA(impact_probe, -1.0f, dir, impact_probe);
	contents = QNN_TraceContents(impact_probe);
	if (dist < max_dist - 1.0f)
		token->solid_frac += 1.0f;
	if (contents == CONTENTS_WATER)
		token->water_frac += 1.0f;
	else if (contents == CONTENTS_SLIME)
		token->slime_frac += 1.0f;
	else if (contents == CONTENTS_LAVA)
		token->lava_frac += 1.0f;
}

/* ── Sector builders ───────────────────────────────────────────── */

static void QNN_BuildHorizontalSpatial(const qnn_snapshot_t *snapshot, qnn_spatial_token_t *token, float center_deg, float span_deg)
{
	int i;
	int samples;
	vec3_t dir;
	vec3_t end;
	vec3_t impact;
	vec3_t down_start;
	vec3_t down_end;
	vec3_t down_impact;
	float yaw_deg;
	float yaw_rad;
	float max_dist;
	float clear_dist;
	float ground_dist;

	samples = 5;
	max_dist = 1024.0f;
	for (i = 0; i < samples; ++i)
	{
		yaw_deg = snapshot->player_view_angles[1] + center_deg + ((float)i - 2.0f) * (span_deg / 4.0f);
		yaw_rad = yaw_deg * (float)M_PI / 180.0f;
		dir[0] = cos((double)yaw_rad);
		dir[1] = sin((double)yaw_rad);
		dir[2] = 0.0f;
		QNN_SpatialSampleRay(token, snapshot->player_origin, dir, max_dist);

		if (i == 2)
		{
			VectorMA(snapshot->player_origin, 64.0f, dir, end);
			clear_dist = QNN_TraceLineDistance(snapshot->player_origin, end, impact);
			token->clearance += QNN_Clamp(clear_dist / 64.0f, 0.0f, 1.0f);

			VectorCopy(impact, down_start);
			down_start[2] += 24.0f;
			VectorCopy(impact, down_end);
			down_end[2] -= 64.0f;
			ground_dist = QNN_TraceLineDistance(down_start, down_end, down_impact);
			token->traversable += (clear_dist > 56.0f && ground_dist <= 40.0f) ? 1.0f : 0.0f;
			token->dropoff += QNN_Clamp((ground_dist - 18.0f) / 46.0f, 0.0f, 1.0f);
		}
		else
		{
			token->clearance += 0.0f;
			token->traversable += 0.0f;
			token->dropoff += 0.0f;
		}
	}
	QNN_SpatialFinalize(token, samples, max_dist);
}

static void QNN_BuildGroundSpatial(const qnn_snapshot_t *snapshot, qnn_spatial_token_t *token)
{
	int i;
	int samples;
	vec3_t offsets[5];
	vec3_t start;
	vec3_t end;
	vec3_t impact;
	float max_dist;
	float dist;
	int contents;

	samples = 5;
	max_dist = 128.0f;
	offsets[0][0] = 0.0f; offsets[0][1] = 0.0f; offsets[0][2] = 0.0f;
	offsets[1][0] = 16.0f; offsets[1][1] = 0.0f; offsets[1][2] = 0.0f;
	offsets[2][0] = -16.0f; offsets[2][1] = 0.0f; offsets[2][2] = 0.0f;
	offsets[3][0] = 0.0f; offsets[3][1] = 16.0f; offsets[3][2] = 0.0f;
	offsets[4][0] = 0.0f; offsets[4][1] = -16.0f; offsets[4][2] = 0.0f;
	for (i = 0; i < samples; ++i)
	{
		VectorAdd(snapshot->player_origin, offsets[i], start);
		VectorCopy(start, end);
		end[2] -= max_dist;
		dist = QNN_TraceLineDistance(start, end, impact);
		token->mean_dist += dist;
		if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
			token->nearest_dist = dist;
		contents = QNN_TraceContents(impact);
		if (dist < max_dist - 1.0f)
			token->solid_frac += 1.0f;
		if (contents == CONTENTS_WATER)
			token->water_frac += 1.0f;
		else if (contents == CONTENTS_SLIME)
			token->slime_frac += 1.0f;
		else if (contents == CONTENTS_LAVA)
			token->lava_frac += 1.0f;
		token->traversable += dist <= 24.0f ? 1.0f : 0.0f;
		token->dropoff += QNN_Clamp((dist - 18.0f) / 48.0f, 0.0f, 1.0f);
		token->clearance += QNN_Clamp(1.0f - (dist / max_dist), 0.0f, 1.0f);
	}
	QNN_SpatialFinalize(token, samples, max_dist);
}

static void QNN_BuildCeilingSpatial(const qnn_snapshot_t *snapshot, qnn_spatial_token_t *token)
{
	int i;
	int samples;
	vec3_t offsets[5];
	vec3_t start;
	vec3_t end;
	vec3_t impact;
	float max_dist;
	float dist;
	int contents;

	samples = 5;
	max_dist = 128.0f;
	offsets[0][0] = 0.0f; offsets[0][1] = 0.0f; offsets[0][2] = 24.0f;
	offsets[1][0] = 16.0f; offsets[1][1] = 0.0f; offsets[1][2] = 24.0f;
	offsets[2][0] = -16.0f; offsets[2][1] = 0.0f; offsets[2][2] = 24.0f;
	offsets[3][0] = 0.0f; offsets[3][1] = 16.0f; offsets[3][2] = 24.0f;
	offsets[4][0] = 0.0f; offsets[4][1] = -16.0f; offsets[4][2] = 24.0f;
	for (i = 0; i < samples; ++i)
	{
		VectorAdd(snapshot->player_origin, offsets[i], start);
		VectorCopy(start, end);
		end[2] += max_dist;
		dist = QNN_TraceLineDistance(start, end, impact);
		token->mean_dist += dist;
		if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
			token->nearest_dist = dist;
		contents = QNN_TraceContents(impact);
		if (dist < max_dist - 1.0f)
			token->solid_frac += 1.0f;
		if (contents == CONTENTS_WATER)
			token->water_frac += 1.0f;
		else if (contents == CONTENTS_SLIME)
			token->slime_frac += 1.0f;
		else if (contents == CONTENTS_LAVA)
			token->lava_frac += 1.0f;
		token->traversable += dist >= 56.0f ? 1.0f : 0.0f;
		token->clearance += QNN_Clamp(dist / max_dist, 0.0f, 1.0f);
	}
	QNN_SpatialFinalize(token, samples, max_dist);
}

/* ── Public API ────────────────────────────────────────────────── */

void QNN_SpatialEmitTokens(const qnn_snapshot_t *snapshot, qnn_spatial_token_t tokens[QNN_SPATIAL_TOKEN_COUNT])
{
	int i;

	for (i = 0; i < QNN_SPATIAL_TOKEN_COUNT; ++i)
		QNN_SpatialReset(&tokens[i], i);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FOV_CENTER], 0.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FOV_LEFT], 40.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FOV_RIGHT], -40.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FLANK_LEFT], 90.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FLANK_RIGHT], -90.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_REAR_LEFT], 150.0f, 30.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_REAR_RIGHT], -150.0f, 30.0f);
	QNN_BuildGroundSpatial(snapshot, &tokens[QNN_SPATIAL_GROUND]);
	QNN_BuildCeilingSpatial(snapshot, &tokens[QNN_SPATIAL_CEILING]);
}
