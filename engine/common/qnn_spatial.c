/*
 * qnn_spatial.c — Center-ray depth atlas (spatial-tokens-v2 rev 8).
 *
 * Coverage is NOT the full sphere: band centers span -75..+75
 * degrees (cells reach +-82.5), leaving two unsampled 7.5-degree
 * half-angle polar cones — see wire.12.md.
 *
 * The map's hull-1 boundary is carved ONCE at map load into its
 * complete face set (qnn_hull_carve); per tick, every atlas cell is
 * the exact first intersection of the cell's CENTER direction with
 * that face set (QNN_CarveQueryRay) — a projection of known geometry,
 * not a discovery trace: no BSP traversal, no clipping degeneracy, and
 * a face plane through the origin (the standing player's floor) hits
 * at distance zero when entered.  Solid movers (doors, plats)
 * participate exactly: each brush submodel's hull 1 is carved once in
 * local space and its faces are translated to the live origin per
 * tick (QW movers translate; they do not rotate).
 *
 * Grid: QNN_OBS_ATLAS_ELEVS elevation bands (centers −75°..+75°, 15°
 * steps) × QNN_OBS_ATLAS_YAWS yaw cells (15° steps counter-clockwise
 * from view yaw; cell 0 = forward), elevation-major.  Depth is 4-bit
 * quantized on a log ladder (QNN_AtlasQuantizeDepth); code 15 = no hit
 * within the band's range limit min(1024, 128/|sin elev|) — the same
 * 1024-unit horizontal / 128-unit vertical contract as v1.
 *
 * Frame convention: yaw-only rotation about Z.  Pitch is deliberately
 * excluded — it is self-state (self.view_pitch); see
 * agents/plans/spatial-tokens-v2.md.
 *
 * Layout rejected on the way here (agents/plans/spatial-tokens-v2.md
 * revs 5–7): per-sector supporting-plane profiles — a finite clipped
 * polygon decoded as an infinite plane over a broad angular volume
 * reconstructs open space too conservatively.  The atlas passed the
 * same reconstruction gate the profiles failed.
 */

#include "qnn_io.h"
#include "qnn_object.h"
#include "qnn_hull_carve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ════════════════════════════════════════════════════════════════════
 *  v1 raycast-scalar spatial tokens (wire.11).
 *
 *  9 sectors: FOV center/left/right, flank left/right, rear left/right,
 *  ground, ceiling.  Each sector samples 5 rays and accumulates
 *  distance, surface type, clearance, traversability, and drop-off.
 *
 *  Retained (restored from the a25/main line) so the single bin can
 *  serve BOTH wire.11 (a24/a25) and wire.12 (a26, the depth atlas
 *  below) models.  QNN_IOEmit runs exactly ONE emitter per tick, chosen
 *  by the loaded codec's spatial_mode (QNN_IOSetSpatialMode at load); a
 *  wire.12 model never runs this v1 raycast, a wire.11 model never runs
 *  the atlas carve.
 * ════════════════════════════════════════════════════════════════════ */

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
	QNN_TraceLine(start, end, &trace);
	VectorCopy(trace.endpos, impact);
	VectorSubtract(trace.endpos, start, delta);
	return QNN_VecLength(delta);
}

/* ── Spatial token internals ───────────────────────────────────── */

static void QNN_SpatialReset(qnn_spatial_token_t *token)
{
	memset(token, 0, sizeof(*token));
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
	float center_rad;
	vec3_t forward, right, up;

	/* Store view-relative unit direction for the sector center */
	center_rad = center_deg * (float)M_PI / 180.0f;
	AngleVectors(snapshot->player_view_angles, forward, right, up);
	{
		float world_x = cosf(snapshot->player_view_angles[1] * (float)M_PI / 180.0f + center_rad);
		float world_y = sinf(snapshot->player_view_angles[1] * (float)M_PI / 180.0f + center_rad);
		float world_dir[3] = { world_x, world_y, 0.0f };
		token->dir[0] = DotProduct(world_dir, forward);
		token->dir[1] = DotProduct(world_dir, right);
		token->dir[2] = DotProduct(world_dir, up);
	}

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
	vec3_t forward, right, up;
	float world_down[3] = { 0.0f, 0.0f, -1.0f };

	/* View-relative direction: straight down */
	AngleVectors(snapshot->player_view_angles, forward, right, up);
	token->dir[0] = DotProduct(world_down, forward);
	token->dir[1] = DotProduct(world_down, right);
	token->dir[2] = DotProduct(world_down, up);

	samples = 5;
	max_dist = 128.0f;
	offsets[0][0] = 0.0f;               offsets[0][1] = 0.0f;               offsets[0][2] = 0.0f;
	offsets[1][0] = QNN_PLAYER_MAXS_X;  offsets[1][1] = 0.0f;               offsets[1][2] = 0.0f;
	offsets[2][0] = QNN_PLAYER_MINS_X;  offsets[2][1] = 0.0f;               offsets[2][2] = 0.0f;
	offsets[3][0] = 0.0f;               offsets[3][1] = QNN_PLAYER_MAXS_Y;  offsets[3][2] = 0.0f;
	offsets[4][0] = 0.0f;               offsets[4][1] = QNN_PLAYER_MINS_Y;  offsets[4][2] = 0.0f;
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
	vec3_t forward, right, up;
	float world_up[3] = { 0.0f, 0.0f, 1.0f };

	/* View-relative direction: straight up */
	AngleVectors(snapshot->player_view_angles, forward, right, up);
	token->dir[0] = DotProduct(world_up, forward);
	token->dir[1] = DotProduct(world_up, right);
	token->dir[2] = DotProduct(world_up, up);

	samples = 5;
	max_dist = 128.0f;
	offsets[0][0] = 0.0f;               offsets[0][1] = 0.0f;               offsets[0][2] = QNN_PLAYER_MAXS_Z - 8.0f;
	offsets[1][0] = QNN_PLAYER_MAXS_X;  offsets[1][1] = 0.0f;               offsets[1][2] = QNN_PLAYER_MAXS_Z - 8.0f;
	offsets[2][0] = QNN_PLAYER_MINS_X;  offsets[2][1] = 0.0f;               offsets[2][2] = QNN_PLAYER_MAXS_Z - 8.0f;
	offsets[3][0] = 0.0f;               offsets[3][1] = QNN_PLAYER_MAXS_Y;  offsets[3][2] = QNN_PLAYER_MAXS_Z - 8.0f;
	offsets[4][0] = 0.0f;               offsets[4][1] = QNN_PLAYER_MINS_Y;  offsets[4][2] = QNN_PLAYER_MAXS_Z - 8.0f;
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

/* v1 public API — build all 9 raycast-scalar sectors. */
void QNN_SpatialEmitTokens(const qnn_snapshot_t *snapshot, qnn_spatial_token_t tokens[QNN_SPATIAL_TOKEN_COUNT])
{
	int i;

	for (i = 0; i < QNN_SPATIAL_TOKEN_COUNT; ++i)
		QNN_SpatialReset(&tokens[i]);
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

/* ════════════════════════════════════════════════════════════════════
 *  Final 24x11 center-ray depth atlas (wire.12) — rev 15.
 * ════════════════════════════════════════════════════════════════════ */

/* ── Carve caches ──────────────────────────────────────────────── */

#define QNN_SPATIAL_MAX_MOVER_CARVES 32
#define QNN_SPATIAL_HORIZ_RANGE      1024.0f
#define QNN_SPATIAL_VERT_RANGE       128.0f
#define QNN_SPATIAL_YAW_STEP_DEG \
	(360.0f / (float)QNN_OBS_ATLAS_YAWS)

/* Elevation band centers, index order = wire row order.  Mirrors
 * engine_norm.ATLAS_ELEV_DEG. */
static const int qnn_atlas_elev_deg[QNN_OBS_ATLAS_ELEVS] = {
	-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75
};

/* World carve, keyed on cl.worldmodel — rebuilt on map change. */
static model_t        *qnn_carve_world_key = NULL;
static qnn_carve_set_t qnn_carve_world;
static int             qnn_carve_world_ok = 0;

/* Per-mover carve cache, keyed on model_t*.  Flushed with the world
 * carve (model pointers recycle across map loads). */
static struct
{
	model_t         *model;
	qnn_carve_set_t  set;
	int              ok;
} qnn_carve_movers[QNN_SPATIAL_MAX_MOVER_CARVES];
static int qnn_carve_mover_count = 0;

static void QNN_SpatialCarveFlush(void)
{
	int i;

	QNN_CarveSetFree(&qnn_carve_world);
	qnn_carve_world_ok = 0;
	for (i = 0; i < qnn_carve_mover_count; ++i)
		QNN_CarveSetFree(&qnn_carve_movers[i].set);
	qnn_carve_mover_count = 0;
}

static void QNN_SpatialCarveEnsureWorld(void)
{
	if (cl.worldmodel == qnn_carve_world_key && qnn_carve_world_ok)
		return;
	QNN_SpatialCarveFlush();
	qnn_carve_world_key = cl.worldmodel;
	if (cl.worldmodel == NULL)
		return;
	qnn_carve_world_ok =
		QNN_CarveModelHull1(cl.worldmodel, &qnn_carve_world) > 0;
	if (qnn_carve_world_ok)
		Con_DPrintf("qnn_spatial: carved %d hull-1 faces for %s\n",
			qnn_carve_world.face_count, cl.worldmodel->name);
}

static qnn_carve_set_t *QNN_SpatialCarveMover(model_t *mod)
{
	int i;

	for (i = 0; i < qnn_carve_mover_count; ++i)
		if (qnn_carve_movers[i].model == mod)
			return qnn_carve_movers[i].ok ? &qnn_carve_movers[i].set : NULL;
	if (qnn_carve_mover_count >= QNN_SPATIAL_MAX_MOVER_CARVES)
		return NULL;
	i = qnn_carve_mover_count++;
	qnn_carve_movers[i].model = mod;
	qnn_carve_movers[i].ok =
		QNN_CarveModelHull1(mod, &qnn_carve_movers[i].set) > 0;
	return qnn_carve_movers[i].ok ? &qnn_carve_movers[i].set : NULL;
}

/* Per-band radial range: the vertical contract caps pitched cells. */
static float QNN_AtlasBandRange(int ei)
{
	float sine = fabsf(sinf(qnn_atlas_elev_deg[ei] * (float)M_PI / 180.0f));
	float max_dist = QNN_SPATIAL_HORIZ_RANGE;

	if (sine > 1e-5f)
	{
		float vert_limited = QNN_SPATIAL_VERT_RANGE / sine;
		if (vert_limited < max_dist)
			max_dist = vert_limited;
	}
	return max_dist;
}

static void QNN_SpatialCarveAtlasAt(
	const qnn_carve_instance_t *insts, int n_insts,
	const float origin[3], float yaw_deg,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS],
	float dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS]);

/* ── Opt-in reconstruction diagnostic ───────────────────────────
 *
 * QNN_SPATIAL_DIAG=/path/to/records.jsonl writes the production
 * quantized atlas codes (and the pre-quantization float distances)
 * beside an independently traced dense 3D depth field on the same
 * grid.  The Python qnn.diag.spatial_reconstruction scorer
 * reconstructs that field using only the codes.  Inert unless the
 * environment variable is set.
 *
 * QNN_SPATIAL_LINEAR=1 additionally disables the carve sets' XY-grid
 * broad-phase (linear face scans), for grid/linear purity proofs.
 */

static FILE *qnn_spatial_diag_file = NULL;
static int qnn_spatial_diag_init = 0;
static int qnn_spatial_diag_calls = 0;
static int qnn_spatial_diag_written = 0;
static int qnn_spatial_diag_stride = 20;
static int qnn_spatial_diag_max = 256;

static void QNN_SpatialDiagInit(void)
{
	const char *path;
	const char *value;

	if (qnn_spatial_diag_init)
		return;
	qnn_spatial_diag_init = 1;
	path = getenv("QNN_SPATIAL_DIAG");
	if (path == NULL || path[0] == '\0')
		return;
	value = getenv("QNN_SPATIAL_DIAG_STRIDE");
	if (value != NULL && atoi(value) > 0)
		qnn_spatial_diag_stride = atoi(value);
	value = getenv("QNN_SPATIAL_DIAG_MAX");
	if (value != NULL && atoi(value) > 0)
		qnn_spatial_diag_max = atoi(value);
	qnn_spatial_diag_file = fopen(path, "w");
	if (qnn_spatial_diag_file == NULL)
		Con_Printf("qnn_spatial: could not open diagnostic %s\n", path);
	else
		Con_Printf("qnn_spatial: reconstruction diagnostic -> %s\n", path);
}

static int QNN_SpatialDiagSampleNow(void)
{
	QNN_SpatialDiagInit();
	if (qnn_spatial_diag_file == NULL ||
	    qnn_spatial_diag_written >= qnn_spatial_diag_max)
		return 0;
	return (qnn_spatial_diag_calls++ % qnn_spatial_diag_stride) == 0;
}

static void QNN_SpatialDiagEmit(const qnn_snapshot_t *snapshot,
	const uint8_t codes[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS],
	const float dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS],
	const uint8_t static_codes[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS],
	const float static_dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS])
{
	FILE *f = qnn_spatial_diag_file;
	int ei, yi, yaw_rel;

	fprintf(f, "{\"schema\":8,\"map\":\"%s\",\"time\":%.6f,",
		cl.worldmodel != NULL ? cl.worldmodel->name : "", (double)cl.mtime[0]);
	/* Probe-grid studies need the query pose: reprojection between
	 * sample points is impossible without it. */
	fprintf(f, "\"origin\":[%.3f,%.3f,%.3f],\"view_yaw\":%.3f,",
		(double)snapshot->player_origin[0],
		(double)snapshot->player_origin[1],
		(double)snapshot->player_origin[2],
		(double)snapshot->player_view_angles[1]);
	fprintf(f, "\"yaw_step\":%d,\"elevations\":[",
		(int)QNN_SPATIAL_YAW_STEP_DEG);
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
		fprintf(f, "%s%d", ei ? "," : "", qnn_atlas_elev_deg[ei]);
	fprintf(f, "],\"max_horiz\":%d,\"max_vert\":%d,",
		(int)QNN_SPATIAL_HORIZ_RANGE, (int)QNN_SPATIAL_VERT_RANGE);

	/* Production payload: elevation-major code rows, exactly the wire
	 * bytes. */
	fprintf(f, "\"atlas_code\":[");
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		if (ei) fputc(',', f);
		fputc('[', f);
		for (yi = 0; yi < QNN_OBS_ATLAS_YAWS; ++yi)
			fprintf(f, "%s%d", yi ? "," : "", (int)codes[ei][yi]);
		fputc(']', f);
	}

	/* Pre-quantization center-ray distances (−1 = miss) for the
	 * float-reference scorer layout. */
	fprintf(f, "],\"atlas_distance\":[");
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		if (ei) fputc(',', f);
		fputc('[', f);
		for (yi = 0; yi < QNN_OBS_ATLAS_YAWS; ++yi)
			fprintf(f, "%s%d", yi ? "," : "",
				dist[ei][yi] < 0.0f ? -1 : (int)lroundf(dist[ei][yi]));
		fputc(']', f);
	}

	/* Static-world teacher target for map-memory experiments.  Unlike the
	 * production fields above, these exclude translated mover instances so
	 * an immutable map table is never penalized for dynamic geometry. */
	fprintf(f, "],\"static_atlas_code\":[");
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		if (ei) fputc(',', f);
		fputc('[', f);
		for (yi = 0; yi < QNN_OBS_ATLAS_YAWS; ++yi)
			fprintf(f, "%s%d", yi ? "," : "", (int)static_codes[ei][yi]);
		fputc(']', f);
	}
	fprintf(f, "],\"static_atlas_distance\":[");
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		if (ei) fputc(',', f);
		fputc('[', f);
		for (yi = 0; yi < QNN_OBS_ATLAS_YAWS; ++yi)
			fprintf(f, "%s%d", yi ? "," : "",
				static_dist[ei][yi] < 0.0f ? -1 :
				(int)lroundf(static_dist[ei][yi]));
		fputc(']', f);
	}
	fprintf(f, "],\"truth\":[");

	/* Independent dense truth: hull-1 traces on the same grid,
	 * elevation-major, yaw 0..355 relative to view yaw. */
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		float elev_rad = qnn_atlas_elev_deg[ei] * (float)M_PI / 180.0f;
		float sz = sinf(elev_rad);
		float cz = cosf(elev_rad);
		float max_dist = QNN_AtlasBandRange(ei);

		for (yaw_rel = 0; yaw_rel < 360; yaw_rel += (int)QNN_SPATIAL_YAW_STEP_DEG)
		{
			float yaw_rad = (snapshot->player_view_angles[1] + yaw_rel)
				* (float)M_PI / 180.0f;
			vec3_t end;
			trace_t trace;
			float distance;

			end[0] = snapshot->player_origin[0] + cosf(yaw_rad) * cz * max_dist;
			end[1] = snapshot->player_origin[1] + sinf(yaw_rad) * cz * max_dist;
			end[2] = snapshot->player_origin[2] + sz * max_dist;
			QNN_TraceClearance(snapshot->player_origin, end, &trace);
			distance = max_dist * trace.fraction;
			if (ei != 0 || yaw_rel != 0) fputc(',', f);
			fprintf(f, "%d", (int)lroundf(distance));
		}
	}
	fprintf(f, "]}\n");
	fflush(f);
	qnn_spatial_diag_written++;
}

/* ── Public API ────────────────────────────────────────────────── */

/* Carve the panoramic depth atlas from `origin`, with yaw cell 0 placed
 * at `yaw_deg`, against a caller-supplied carve instance set.  The live
 * per-frame emit (world + movers, view yaw) and the load-time probe emit
 * (world only, world-anchored yaw 0) both route through here so their
 * 4-bit codes are byte-identical.  `dist` receives per-cell hit distance
 * (-1 on miss) for the diag path; pass a scratch buffer when unused. */
static void QNN_SpatialCarveAtlasAt(
	const qnn_carve_instance_t *insts, int n_insts,
	const float origin[3], float yaw_deg,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS],
	float dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS])
{
	int ei, yi;

	memset(atlas, QNN_OBS_ATLAS_MISS_CODE,
		QNN_OBS_ATLAS_ELEVS * QNN_OBS_ATLAS_YAWS);

	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		float elev_rad = qnn_atlas_elev_deg[ei] * (float)M_PI / 180.0f;
		float cos_e = cosf(elev_rad);
		float sin_e = sinf(elev_rad);
		float max_dist = QNN_AtlasBandRange(ei);

		for (yi = 0; yi < QNN_OBS_ATLAS_YAWS; ++yi)
		{
			float yaw_rad = (yaw_deg + QNN_SPATIAL_YAW_STEP_DEG * yi)
				* (float)M_PI / 180.0f;
			float dir[3], n_world[3], p_world[3], rel[3];
			int hit;

			dir[0] = cos_e * cosf(yaw_rad);
			dir[1] = cos_e * sinf(yaw_rad);
			dir[2] = sin_e;
			hit = QNN_CarveQueryRay(insts, n_insts,
				origin, dir, max_dist, n_world, p_world);
			if (hit)
			{
				VectorSubtract(p_world, origin, rel);
				dist[ei][yi] = sqrtf(DotProduct(rel, rel));
				atlas[ei][yi] = QNN_AtlasQuantizeDepth(dist[ei][yi]);
			}
			else
				dist[ei][yi] = -1.0f;
		}
	}
}

/* Compatibility carve for the pre-finalization a26 rc1 atlas.  That model
 * family consumed 72 five-degree yaw cells, stored unpacked.  Keep this path
 * isolated from the finalized 24-cell diagnostic/probe contract: it runs only
 * when a loaded ONNX graph explicitly declares spatial_atlas[...,72]. */
static void QNN_SpatialCarveLegacyAtlasAt(
	const qnn_carve_instance_t *insts, int n_insts,
	const float origin[3], float yaw_deg,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX])
{
	int ei, yi;

	memset(atlas, QNN_OBS_ATLAS_MISS_CODE,
		QNN_OBS_ATLAS_ELEVS * QNN_OBS_ATLAS_YAWS_MAX);
	for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
	{
		float elev_rad = qnn_atlas_elev_deg[ei] * (float)M_PI / 180.0f;
		float cos_e = cosf(elev_rad);
		float sin_e = sinf(elev_rad);
		float max_dist = QNN_AtlasBandRange(ei);

		for (yi = 0; yi < QNN_OBS_ATLAS_YAWS_LEGACY; ++yi)
		{
			float yaw_rad = (yaw_deg + 360.0f * yi /
				(float)QNN_OBS_ATLAS_YAWS_LEGACY) * (float)M_PI / 180.0f;
			float dir[3], n_world[3], p_world[3], rel[3];

			dir[0] = cos_e * cosf(yaw_rad);
			dir[1] = cos_e * sinf(yaw_rad);
			dir[2] = sin_e;
			if (QNN_CarveQueryRay(insts, n_insts,
				origin, dir, max_dist, n_world, p_world))
			{
				VectorSubtract(p_world, origin, rel);
				atlas[ei][yi] = QNN_AtlasQuantizeDepth(
					sqrtf(DotProduct(rel, rel)));
			}
		}
	}
}

void QNN_SpatialEmitAtlas(const qnn_snapshot_t *snapshot,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX],
	int yaw_count)
{
	qnn_carve_instance_t insts[1 + QNN_SPATIAL_MAX_MOVER_CARVES];
	uint8_t current_atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS];
	float dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS];
	int n_insts = 0;
	int mover_count;
	int i, yi;

	QNN_SpatialCarveEnsureWorld();
	if (qnn_carve_world_ok)
	{
		insts[n_insts].set = &qnn_carve_world;
		insts[n_insts].origin[0] = 0.0f;
		insts[n_insts].origin[1] = 0.0f;
		insts[n_insts].origin[2] = 0.0f;
		n_insts++;
	}

	/* Solid movers at live origins — same per-frame cache the trace
	 * path uses (qnn_store), so occlusion parity is preserved. */
	mover_count = QNN_TraceMoverCacheRefresh();
	for (i = 0; i < mover_count && n_insts < 1 + QNN_SPATIAL_MAX_MOVER_CARVES; ++i)
	{
		model_t *m = QNN_TraceMoverModel(i);
		qnn_carve_set_t *set;
		float *origin = QNN_TraceMoverOrigin(i);

		if (m == NULL)
			continue;
		set = QNN_SpatialCarveMover(m);
		if (set == NULL)
			continue;
		insts[n_insts].set = set;
		insts[n_insts].origin[0] = origin[0];
		insts[n_insts].origin[1] = origin[1];
		insts[n_insts].origin[2] = origin[2];
		n_insts++;
	}

	if (yaw_count == QNN_OBS_ATLAS_YAWS_LEGACY)
	{
		QNN_SpatialCarveLegacyAtlasAt(insts, n_insts,
			snapshot->player_origin, snapshot->player_view_angles[1], atlas);
		return;
	}

	/* The default/current path remains the validated 24x11 carve. Copy its
	 * compact rows into the max-width tick scratch used by both codecs. */
	QNN_SpatialCarveAtlasAt(insts, n_insts,
		snapshot->player_origin, snapshot->player_view_angles[1],
		current_atlas, dist);
	memset(atlas, QNN_OBS_ATLAS_MISS_CODE, sizeof(uint8_t)
		* QNN_OBS_ATLAS_ELEVS * QNN_OBS_ATLAS_YAWS_MAX);
	for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
		memcpy(atlas[i], current_atlas[i], QNN_OBS_ATLAS_YAWS);

	if (QNN_SpatialDiagSampleNow())
	{
		uint8_t static_atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS];
		float static_dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS];

		if (qnn_carve_world_ok)
			QNN_SpatialCarveAtlasAt(insts, 1,
				snapshot->player_origin, snapshot->player_view_angles[1],
				static_atlas, static_dist);
		else
		{
			memset(static_atlas, QNN_OBS_ATLAS_MISS_CODE,
				sizeof(static_atlas));
			for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
				for (yi = 0; yi < QNN_OBS_ATLAS_YAWS; ++yi)
					static_dist[i][yi] = -1.0f;
		}
		QNN_SpatialDiagEmit(snapshot,
			(const uint8_t (*)[QNN_OBS_ATLAS_YAWS])current_atlas, dist,
			(const uint8_t (*)[QNN_OBS_ATLAS_YAWS])static_atlas,
			static_dist);
	}
}

/* Load-time probe carve: the panoramic atlas from `origin`, world-
 * anchored (yaw cell 0 = world yaw 0), against the STATIC world only.
 * Movers are excluded by design — a load-time probe table describes the
 * fixed map, and movers ride TOKEN_MOVER.  Returns 1 on a built world
 * carve, else 0 with the atlas left all-miss.  Backs the offline
 * nav_query probe_atlas dump consumed by qnn.bc.probe_grid. */
int QNN_SpatialCarveProbeAtlas(const float origin[3], float yaw_deg,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS])
{
	qnn_carve_instance_t world_inst;
	float dist[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS];

	QNN_SpatialCarveEnsureWorld();
	if (!qnn_carve_world_ok)
	{
		memset(atlas, QNN_OBS_ATLAS_MISS_CODE,
			QNN_OBS_ATLAS_ELEVS * QNN_OBS_ATLAS_YAWS);
		return 0;
	}
	world_inst.set = &qnn_carve_world;
	world_inst.origin[0] = 0.0f;
	world_inst.origin[1] = 0.0f;
	world_inst.origin[2] = 0.0f;
	QNN_SpatialCarveAtlasAt(&world_inst, 1, origin, yaw_deg, atlas, dist);
	return 1;
}

int QNN_SpatialBenchmarkWorldAtlas(const float origin[3], int yaw_count,
	int iterations, double *microseconds_per_atlas,
	double *nanoseconds_per_ray, unsigned int *checksum)
{
	qnn_carve_instance_t world_inst;
	clock_t started, finished;
	unsigned int sum = 0;
	int iteration, ei, yi;

	if (origin == NULL || microseconds_per_atlas == NULL
		|| nanoseconds_per_ray == NULL || checksum == NULL
		|| (yaw_count != 72 && yaw_count != 36 && yaw_count != 24)
		|| iterations <= 0)
		return 0;
	QNN_SpatialCarveEnsureWorld();
	if (!qnn_carve_world_ok)
		return 0;
	world_inst.set = &qnn_carve_world;
	world_inst.origin[0] = 0.0f;
	world_inst.origin[1] = 0.0f;
	world_inst.origin[2] = 0.0f;

	started = clock();
	for (iteration = 0; iteration < iterations; ++iteration)
	{
		for (ei = 0; ei < QNN_OBS_ATLAS_ELEVS; ++ei)
		{
			float elev_rad = qnn_atlas_elev_deg[ei] * (float)M_PI / 180.0f;
			float cos_e = cosf(elev_rad);
			float sin_e = sinf(elev_rad);
			float max_dist = QNN_AtlasBandRange(ei);

			for (yi = 0; yi < yaw_count; ++yi)
			{
				float yaw_rad = (360.0f * yi / yaw_count) * (float)M_PI / 180.0f;
				float dir[3], normal[3], point[3], rel[3];
				dir[0] = cos_e * cosf(yaw_rad);
				dir[1] = cos_e * sinf(yaw_rad);
				dir[2] = sin_e;
				if (QNN_CarveQueryRay(&world_inst, 1, origin, dir, max_dist,
					normal, point))
				{
					VectorSubtract(point, origin, rel);
					sum += QNN_AtlasQuantizeDepth(sqrtf(DotProduct(rel, rel)));
				}
				else
					sum += QNN_OBS_ATLAS_MISS_CODE;
			}
		}
	}
	finished = clock();
	if (finished == (clock_t)-1 || started == (clock_t)-1)
		return 0;
	*microseconds_per_atlas =
		1e6 * (double)(finished - started) / (double)CLOCKS_PER_SEC / iterations;
	*nanoseconds_per_ray =
		1000.0 * *microseconds_per_atlas
		/ (QNN_OBS_ATLAS_ELEVS * yaw_count);
	*checksum = sum;
	return 1;
}

int QNN_SpatialWorldFaceCount(void)
{
	QNN_SpatialCarveEnsureWorld();
	return qnn_carve_world_ok ? qnn_carve_world.face_count : 0;
}

void QNN_SpatialWriteWorldFacesJson(FILE *out)
{
	int i, j;

	QNN_SpatialCarveEnsureWorld();
	fprintf(out, "{\"count\":%d,\"vertex_count\":%d,\"grid_cell\":%.3f,"
		"\"faces\":[",
		qnn_carve_world_ok ? qnn_carve_world.face_count : 0,
		qnn_carve_world_ok ? qnn_carve_world.vert_count : 0,
		qnn_carve_world_ok ? (double)qnn_carve_world.grid_cell : 0.0);
	if (qnn_carve_world_ok)
	{
		for (i = 0; i < qnn_carve_world.face_count; ++i)
		{
			const qnn_carve_face_t *face = &qnn_carve_world.faces[i];
			fprintf(out, "%s{\"normal\":[%.6f,%.6f,%.6f],\"dist\":%.6f,"
				"\"mins\":[%.3f,%.3f,%.3f],\"maxs\":[%.3f,%.3f,%.3f],"
				"\"verts\":[",
				i ? "," : "",
				(double)face->normal[0], (double)face->normal[1],
				(double)face->normal[2], (double)face->dist,
				(double)face->mins[0], (double)face->mins[1],
				(double)face->mins[2], (double)face->maxs[0],
				(double)face->maxs[1], (double)face->maxs[2]);
			for (j = 0; j < face->vert_count; ++j)
			{
				const float *v = &qnn_carve_world.verts[
					3 * (face->first_vert + j)];
				fprintf(out, "%s[%.3f,%.3f,%.3f]", j ? "," : "",
					(double)v[0], (double)v[1], (double)v[2]);
			}
			fprintf(out, "]}");
		}
	}
	fprintf(out, "]}");
}

int QNN_SpatialWriteWorldCellsJson(FILE *out)
{
	if (out == NULL || cl.worldmodel == NULL)
		return 0;
	return QNN_CarveWriteHull1CellsJson(cl.worldmodel, out);
}
