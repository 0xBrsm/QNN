extern "C" {
#include "quakedef.h"
}

#include "qnn_navmesh.h"

#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "DetourNavMesh.h"
#include "DetourNavMeshBuilder.h"
#include "DetourNavMeshQuery.h"
#include "Recast.h"

/* Shared error-formatting helper (declared in qnn_navmesh.h, used by both
   qnn_navmesh.cpp and qnn_route.cpp). */
extern "C" void qnn_nav_set_error(char *error, size_t error_size, const char *format, ...)
{
	va_list args;

	if (error == nullptr || error_size == 0)
		return;
	va_start(args, format);
	vsnprintf(error, error_size, format, args);
	va_end(args);
}

namespace {

constexpr unsigned char kAreaWalkable = 1;
constexpr unsigned char kAreaNearWall = 2;
constexpr unsigned short kPolyFlagWalk = 1;
constexpr float kNearWallCost = 3.0f;

static void qnn_navmesh_setup_filter(dtQueryFilter *filter)
{
	filter->setAreaCost(kAreaWalkable, 1.0f);
	filter->setAreaCost(kAreaNearWall, kNearWallCost);
}

struct QnnRcContext : public rcContext
{
	explicit QnnRcContext()
		: rcContext(true)
	{
	}

	std::string last_error;

protected:
	void doLog(const rcLogCategory category, const char *msg, const int len) override
	{
		if (category != RC_LOG_ERROR)
			return;
		last_error.assign(msg, static_cast<size_t>(len));
	}
};

static void qnn_quake_to_recast(const float *quake, float *recast)
{
	recast[0] = quake[0];
	recast[1] = quake[2];
	recast[2] = quake[1];
}

static void qnn_recast_to_quake(const float *recast, float *quake)
{
	quake[0] = recast[0];
	quake[1] = recast[2];
	quake[2] = recast[1];
}

struct RecastBuildGuard
{
	rcHeightfield *solid;
	rcCompactHeightfield *compact;
	rcContourSet *contours;
	rcPolyMesh *poly_mesh;
	rcPolyMeshDetail *detail_mesh;
	unsigned char *nav_data;
	qnn_navmesh_runtime_t *runtime;

	RecastBuildGuard()
		: solid(nullptr), compact(nullptr), contours(nullptr),
		  poly_mesh(nullptr), detail_mesh(nullptr), nav_data(nullptr),
		  runtime(nullptr)
	{
	}

	~RecastBuildGuard()
	{
		rcFreePolyMeshDetail(detail_mesh);
		rcFreePolyMesh(poly_mesh);
		rcFreeContourSet(contours);
		rcFreeCompactHeightfield(compact);
		rcFreeHeightField(solid);
		if (nav_data != nullptr)
			dtFree(nav_data);
		if (runtime != nullptr)
			qnn_navmesh_destroy(runtime);
	}

	RecastBuildGuard(const RecastBuildGuard &) = delete;
	RecastBuildGuard &operator=(const RecastBuildGuard &) = delete;
};

}  // namespace

struct qnn_navmesh_runtime_s
{
	dtNavMesh *navmesh;
	dtNavMeshQuery *query;
	float query_half_extents[3];       /* wide: for items/goals */
	float query_half_extents_tight[3]; /* tight: for agent position */

	qnn_navmesh_runtime_s()
		: navmesh(nullptr), query(nullptr)
	{
		query_half_extents[0] = 48.0f;
		query_half_extents[1] = 128.0f;
		query_half_extents[2] = 48.0f;
		query_half_extents_tight[0] = 24.0f;
		query_half_extents_tight[1] = 56.0f;
		query_half_extents_tight[2] = 24.0f;
	}
};

static void qnn_navmesh_poly_center(const dtMeshTile *tile, const dtPoly *poly, float *center)
{
	int i;

	center[0] = 0.0f;
	center[1] = 0.0f;
	center[2] = 0.0f;
	if (tile == nullptr || poly == nullptr || poly->vertCount <= 0)
		return;

	for (i = 0; i < poly->vertCount; ++i)
	{
		const float *vert;

		vert = &tile->verts[poly->verts[i] * 3];
		center[0] += vert[0];
		center[1] += vert[1];
		center[2] += vert[2];
	}

	center[0] /= (float)poly->vertCount;
	center[1] /= (float)poly->vertCount;
	center[2] /= (float)poly->vertCount;
}

static void qnn_navmesh_poly_bounds(const dtMeshTile *tile, const dtPoly *poly, float *mins, float *maxs)
{
	int i;

	mins[0] = mins[1] = mins[2] = 999999.0f;
	maxs[0] = maxs[1] = maxs[2] = -999999.0f;
	if (tile == nullptr || poly == nullptr || poly->vertCount <= 0)
	{
		mins[0] = mins[1] = mins[2] = 0.0f;
		maxs[0] = maxs[1] = maxs[2] = 0.0f;
		return;
	}

	for (i = 0; i < poly->vertCount; ++i)
	{
		const float *vert;

		vert = &tile->verts[poly->verts[i] * 3];
		mins[0] = fminf(mins[0], vert[0]);
		mins[1] = fminf(mins[1], vert[1]);
		mins[2] = fminf(mins[2], vert[2]);
		maxs[0] = fmaxf(maxs[0], vert[0]);
		maxs[1] = fmaxf(maxs[1], vert[1]);
		maxs[2] = fmaxf(maxs[2], vert[2]);
	}
}

static int qnn_navmesh_push_unique_ref(unsigned long long *refs, int count, int max_count, dtPolyRef ref)
{
	int i;
	unsigned long long value;

	value = static_cast<unsigned long long>(ref);
	for (i = 0; i < count; ++i)
	{
		if (refs[i] == value)
			return count;
	}
	if (count >= max_count)
		return count;
	refs[count] = value;
	return count + 1;
}

static int qnn_navmesh_collect_neighbors(const qnn_navmesh_runtime_t *navmesh, dtPolyRef ref, unsigned long long *refs, int max_refs)
{
	const dtMeshTile *tile;
	const dtPoly *poly;
	int link_index;
	int count;

	if (navmesh == nullptr || navmesh->navmesh == nullptr || refs == nullptr || max_refs <= 0)
		return 0;
	if (dtStatusFailed(navmesh->navmesh->getTileAndPolyByRef(ref, &tile, &poly)))
		return 0;

	count = 0;
	for (link_index = poly->firstLink; link_index != DT_NULL_LINK; link_index = tile->links[link_index].next)
	{
		dtPolyRef neighbor_ref;

		neighbor_ref = tile->links[link_index].ref;
		if (neighbor_ref == 0 || neighbor_ref == ref)
			continue;
		count = qnn_navmesh_push_unique_ref(refs, count, max_refs, neighbor_ref);
	}
	return count;
}

static float qnn_navmesh_horizontal_dist_sq(const float *a, const float *b)
{
	float dx = a[0] - b[0];
	float dz = a[2] - b[2];
	return dx * dx + dz * dz;
}

static float qnn_navmesh_snap_limit(const qnn_navmesh_runtime_t *navmesh, const float *extents)
{
	float limit;

	limit = fmaxf(extents[0], extents[2]) * 0.75f;
	if (navmesh != nullptr && extents == navmesh->query_half_extents_tight)
		limit = fminf(limit, 24.0f);
	else
		limit = fminf(limit, 48.0f);
	return fmaxf(limit, 16.0f);
}

/* extents_override: if non-NULL, use these instead of the runtime wide defaults. */
static int qnn_navmesh_find_nearest_internal(
	const qnn_navmesh_runtime_t *navmesh,
	const float *point,
	dtPolyRef *nearest_ref,
	float *nearest_pt,
	bool *is_over_poly,
	const float *extents_override,
	char *error,
	size_t error_size)
{
	dtQueryFilter filter; qnn_navmesh_setup_filter(&filter);
	float recast_point[3];
	const float *extents;
	dtStatus status;

	if (navmesh == nullptr || navmesh->query == nullptr || navmesh->navmesh == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Navmesh query requested before navmesh was initialized");
		return 0;
	}

	extents = extents_override ? extents_override : navmesh->query_half_extents;

	qnn_quake_to_recast(point, recast_point);
	status = navmesh->query->findNearestPoly(
		recast_point,
		extents,
		&filter,
		nearest_ref,
		nearest_pt,
		is_over_poly);
	if (dtStatusFailed(status))
	{
		qnn_nav_set_error(error, error_size, "Detour findNearestPoly failed");
		return 0;
	}
	/* Reject results that snapped too far horizontally — prevents
	   snapping through thin walls into adjacent rooms. */
	if (*nearest_ref != 0)
	{
		float dist_sq = qnn_navmesh_horizontal_dist_sq(recast_point, nearest_pt);
		float limit = qnn_navmesh_snap_limit(navmesh, extents);
		if (dist_sq > limit * limit)
		{
			*nearest_ref = 0;
			memset(nearest_pt, 0, sizeof(float) * 3);
			if (is_over_poly != nullptr)
				*is_over_poly = false;
			return 0;
		}
	}
	return *nearest_ref != 0 ? 1 : 0;
}

static float qnn_navmesh_path_distance(const float *points, int point_count)
{
	float total;
	int i;

	total = 0.0f;
	for (i = 1; i < point_count; ++i)
	{
		const float dx = points[i * 3 + 0] - points[(i - 1) * 3 + 0];
		const float dy = points[i * 3 + 1] - points[(i - 1) * 3 + 1];
		const float dz = points[i * 3 + 2] - points[(i - 1) * 3 + 2];
		total += sqrtf(dx * dx + dy * dy + dz * dz);
	}
	return total;
}

extern "C" qnn_navmesh_runtime_t *qnn_navmesh_build(
	const float *verts,
	int vertex_count,
	const int *tris,
	int triangle_count,
	const qnn_navmesh_build_config_t *config,
	qnn_navmesh_summary_t *summary,
	char *error,
	size_t error_size)
{
	QnnRcContext ctx;
	rcConfig rc_config;
	RecastBuildGuard guard;
	std::vector<float> recast_verts;
	std::vector<unsigned char> areas;
	int nav_data_size;
	dtNavMeshCreateParams params;
	dtStatus status;
	int i;

	if (summary != nullptr)
		memset(summary, 0, sizeof(*summary));
	if (verts == nullptr || tris == nullptr || config == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Navmesh build requires non-null vertices, triangles, and config");
		return nullptr;
	}
	if (vertex_count < 3 || triangle_count < 1)
	{
		qnn_nav_set_error(error, error_size, "Navmesh build requires at least 3 vertices and 1 triangle");
		return nullptr;
	}

	memset(&rc_config, 0, sizeof(rc_config));
	recast_verts.resize(static_cast<size_t>(vertex_count) * 3u);
	for (i = 0; i < vertex_count; ++i)
		qnn_quake_to_recast(&verts[i * 3], &recast_verts[static_cast<size_t>(i) * 3u]);

	rcCalcBounds(recast_verts.data(), vertex_count, rc_config.bmin, rc_config.bmax);
	rc_config.cs = config->cell_size;
	rc_config.ch = config->cell_height;
	rcCalcGridSize(rc_config.bmin, rc_config.bmax, rc_config.cs, &rc_config.width, &rc_config.height);
	rc_config.walkableSlopeAngle = config->walkable_slope_angle;
	rc_config.walkableHeight = (int)ceilf(config->walkable_height / rc_config.ch);
	rc_config.walkableClimb = (int)ceilf(config->walkable_climb / rc_config.ch);
	rc_config.walkableRadius = (int)ceilf(config->walkable_radius / rc_config.cs);
	rc_config.maxEdgeLen = (int)(config->max_edge_len / rc_config.cs);
	rc_config.maxSimplificationError = config->max_simplification_error;
	rc_config.minRegionArea = config->min_region_size * config->min_region_size;
	rc_config.mergeRegionArea = config->merge_region_size * config->merge_region_size;
	rc_config.maxVertsPerPoly = config->max_verts_per_poly;
	rc_config.detailSampleDist = config->detail_sample_distance < 0.9f ? 0.0f : rc_config.cs * config->detail_sample_distance;
	rc_config.detailSampleMaxError = rc_config.ch * config->detail_sample_max_error;

	nav_data_size = 0;

	guard.solid = rcAllocHeightfield();
	if (guard.solid == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate Recast heightfield");
		return nullptr;
	}
	if (!rcCreateHeightfield(&ctx, *guard.solid, rc_config.width, rc_config.height, rc_config.bmin, rc_config.bmax, rc_config.cs, rc_config.ch))
	{
		qnn_nav_set_error(error, error_size, "Failed to create Recast heightfield");
		return nullptr;
	}

	areas.assign(static_cast<size_t>(triangle_count), 0);
	rcMarkWalkableTriangles(&ctx, rc_config.walkableSlopeAngle, recast_verts.data(), vertex_count, tris, triangle_count, areas.data());
	if (!rcRasterizeTriangles(&ctx, recast_verts.data(), vertex_count, tris, areas.data(), triangle_count, *guard.solid, rc_config.walkableClimb))
	{
		qnn_nav_set_error(error, error_size, "Failed to rasterize triangles into heightfield");
		return nullptr;
	}

	rcFilterLowHangingWalkableObstacles(&ctx, rc_config.walkableClimb, *guard.solid);
	rcFilterLedgeSpans(&ctx, rc_config.walkableHeight, rc_config.walkableClimb, *guard.solid);
	rcFilterWalkableLowHeightSpans(&ctx, rc_config.walkableHeight, *guard.solid);

	guard.compact = rcAllocCompactHeightfield();
	if (guard.compact == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate compact heightfield");
		return nullptr;
	}
	if (!rcBuildCompactHeightfield(&ctx, rc_config.walkableHeight, rc_config.walkableClimb, *guard.solid, *guard.compact))
	{
		qnn_nav_set_error(error, error_size, "Failed to build compact heightfield");
		return nullptr;
	}
	if (!rcErodeWalkableArea(&ctx, rc_config.walkableRadius, *guard.compact))
	{
		qnn_nav_set_error(error, error_size, "Failed to erode walkable area");
		return nullptr;
	}
	if (!rcBuildDistanceField(&ctx, *guard.compact))
	{
		qnn_nav_set_error(error, error_size, "Failed to build distance field");
		return nullptr;
	}
	if (!rcBuildRegions(&ctx, *guard.compact, 0, rc_config.minRegionArea, rc_config.mergeRegionArea))
	{
		qnn_nav_set_error(error, error_size, "Failed to build navigation regions");
		return nullptr;
	}

	guard.contours = rcAllocContourSet();
	if (guard.contours == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate contour set");
		return nullptr;
	}
	if (!rcBuildContours(&ctx, *guard.compact, rc_config.maxSimplificationError, rc_config.maxEdgeLen, *guard.contours, RC_CONTOUR_TESS_WALL_EDGES))
	{
		qnn_nav_set_error(error, error_size, "Failed to build contours");
		return nullptr;
	}

	guard.poly_mesh = rcAllocPolyMesh();
	if (guard.poly_mesh == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate polygon mesh");
		return nullptr;
	}
	if (!rcBuildPolyMesh(&ctx, *guard.contours, rc_config.maxVertsPerPoly, *guard.poly_mesh))
	{
		qnn_nav_set_error(error, error_size, "Failed to build polygon mesh");
		return nullptr;
	}

	guard.detail_mesh = rcAllocPolyMeshDetail();
	if (guard.detail_mesh == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate detail mesh");
		return nullptr;
	}
	if (!rcBuildPolyMeshDetail(&ctx, *guard.poly_mesh, *guard.compact, rc_config.detailSampleDist, rc_config.detailSampleMaxError, *guard.detail_mesh))
	{
		qnn_nav_set_error(error, error_size, "Failed to build detail mesh");
		return nullptr;
	}
	if (guard.poly_mesh->npolys <= 0 || guard.poly_mesh->nverts <= 0)
	{
		qnn_nav_set_error(error, error_size, "Recast produced an empty navmesh");
		return nullptr;
	}

	/* Mark poly areas using distance-field wall proximity.
	   Two-pass approach:
	   1. Sample each poly's wall distance from the compact heightfield.
	   2. Mark a poly as kAreaNearWall only if it is near a wall AND
	      has a neighbor that is NOT near a wall (i.e., a center-of-corridor
	      alternative exists).  Narrow corridors where ALL polys are near
	      walls stay as normal WALK cost — no penalty when there's no choice. */
	{
		const unsigned short *pverts = guard.poly_mesh->verts;
		const unsigned short *ppolys = guard.poly_mesh->polys;
		const int nvp = guard.poly_mesh->nvp;
		const int npoly = guard.poly_mesh->npolys;
		const unsigned short dist_threshold =
			(unsigned short)(config->walkable_radius / config->cell_size * 2.0f);

		/* Pass 1: compute per-poly "near wall" flag via distance field */
		std::vector<bool> poly_near_wall(static_cast<size_t>(npoly), false);

		for (i = 0; i < npoly; ++i)
		{
			if (guard.poly_mesh->areas[i] != RC_WALKABLE_AREA)
				continue;

			float cx = 0, cz = 0;
			int vc = 0;
			const unsigned short *p = &ppolys[i * nvp * 2];
			for (int vi = 0; vi < nvp && p[vi] != RC_MESH_NULL_IDX; ++vi)
			{
				cx += pverts[p[vi] * 3 + 0];
				cz += pverts[p[vi] * 3 + 2];
				vc++;
			}
			if (vc > 0) { cx /= vc; cz /= vc; }

			int gx = (int)cx;
			int gz = (int)cz;
			if (gx < 0) gx = 0;
			if (gz < 0) gz = 0;
			if (gx >= guard.compact->width) gx = guard.compact->width - 1;
			if (gz >= guard.compact->height) gz = guard.compact->height - 1;

			const rcCompactCell *cell = &guard.compact->cells[gx + gz * guard.compact->width];
			for (int si = (int)cell->index, sn = (int)(cell->index + cell->count); si < sn; ++si)
			{
				if (guard.compact->dist[si] < dist_threshold)
				{
					poly_near_wall[static_cast<size_t>(i)] = true;
					break;
				}
			}
		}

		/* Pass 2: only penalize near-wall polys that have a non-near-wall
		   neighbor (meaning a center path exists).  Narrow corridors where
		   ALL polys are near walls stay at normal WALK cost. */
		for (i = 0; i < npoly; ++i)
		{
			guard.poly_mesh->flags[i] = kPolyFlagWalk;

			if (guard.poly_mesh->areas[i] != RC_WALKABLE_AREA)
				continue;

			if (!poly_near_wall[static_cast<size_t>(i)])
			{
				guard.poly_mesh->areas[i] = kAreaWalkable;
				continue;
			}

			/* Check neighbors: does any adjacent poly have open space? */
			int has_open_neighbor = 0;
			const unsigned short *adj = &ppolys[i * nvp * 2 + nvp];
			for (int ei = 0; ei < nvp; ++ei)
			{
				unsigned short ni = adj[ei];
				if (ni == RC_MESH_NULL_IDX)
					continue;
				if (!poly_near_wall[static_cast<size_t>(ni)])
				{
					has_open_neighbor = 1;
					break;
				}
			}

			if (has_open_neighbor)
				guard.poly_mesh->areas[i] = kAreaNearWall;
			else
				guard.poly_mesh->areas[i] = kAreaWalkable;
		}
	}

	memset(&params, 0, sizeof(params));
	params.verts = guard.poly_mesh->verts;
	params.vertCount = guard.poly_mesh->nverts;
	params.polys = guard.poly_mesh->polys;
	params.polyAreas = guard.poly_mesh->areas;
	params.polyFlags = guard.poly_mesh->flags;
	params.polyCount = guard.poly_mesh->npolys;
	params.nvp = guard.poly_mesh->nvp;
	params.detailMeshes = guard.detail_mesh->meshes;
	params.detailVerts = guard.detail_mesh->verts;
	params.detailVertsCount = guard.detail_mesh->nverts;
	params.detailTris = guard.detail_mesh->tris;
	params.detailTriCount = guard.detail_mesh->ntris;
	params.walkableHeight = config->walkable_height;
	params.walkableRadius = config->walkable_radius;
	/* Use walkable_height (not walkable_climb) for the Detour tile param.
	   This controls the Y search extent when linking off-mesh connections
	   to ground polys.  The navmesh surface sits ~44u below floor level
	   (player hull offset + voxel quantization), so 18u walkable_climb
	   is too small to find ground polys under off-mesh endpoints. */
	params.walkableClimb = config->walkable_height;
	rcVcopy(params.bmin, guard.poly_mesh->bmin);
	rcVcopy(params.bmax, guard.poly_mesh->bmax);
	params.cs = rc_config.cs;
	params.ch = rc_config.ch;
	params.buildBvTree = true;

	if (!dtCreateNavMeshData(&params, &guard.nav_data, &nav_data_size))
	{
		qnn_nav_set_error(error, error_size, "Failed to create Detour navmesh tile");
		return nullptr;
	}

	guard.runtime = new (std::nothrow) qnn_navmesh_runtime_t();
	if (guard.runtime == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate navmesh runtime");
		return nullptr;
	}

	guard.runtime->navmesh = dtAllocNavMesh();
	guard.runtime->query = dtAllocNavMeshQuery();
	if (guard.runtime->navmesh == nullptr || guard.runtime->query == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Failed to allocate Detour navmesh/query");
		return nullptr;
	}

	status = guard.runtime->navmesh->init(guard.nav_data, nav_data_size, DT_TILE_FREE_DATA);
	if (dtStatusFailed(status))
	{
		qnn_nav_set_error(error, error_size, "Failed to initialize Detour navmesh");
		return nullptr;
	}
	/* nav_data ownership transferred to Detour navmesh via DT_TILE_FREE_DATA */
	guard.nav_data = nullptr;

	status = guard.runtime->query->init(guard.runtime->navmesh, 2048);
	if (dtStatusFailed(status))
	{
		qnn_nav_set_error(error, error_size, "Failed to initialize Detour navmesh query");
		return nullptr;
	}

	/* Wide extents for goal/item snapping.
	   Keep XZ tighter than the old 64u box to avoid snapping
	   through thin walls into adjacent rooms. */
	guard.runtime->query_half_extents[0] = fmaxf(config->walkable_radius * 3.0f, 48.0f);
	guard.runtime->query_half_extents[1] = fmaxf(config->walkable_height * 2.0f, 128.0f);
	guard.runtime->query_half_extents[2] = fmaxf(config->walkable_radius * 3.0f, 48.0f);

	/* Tight extents for agent position.
	   Y must cover the offset between player origin (center of 56u hull)
	   and navmesh surface (~44u below), while staying under the minimum
	   floor gap (~64u on dm4). */
	guard.runtime->query_half_extents_tight[0] = fmaxf(config->walkable_radius * 1.5f, 24.0f);
	guard.runtime->query_half_extents_tight[1] = config->walkable_height;
	guard.runtime->query_half_extents_tight[2] = fmaxf(config->walkable_radius * 1.5f, 24.0f);

	if (summary != nullptr)
	{
		summary->input_vertex_count = vertex_count;
		summary->input_triangle_count = triangle_count;
		summary->polygon_count = guard.poly_mesh->npolys;
		summary->navmesh_vertex_count = guard.poly_mesh->nverts;
		summary->detail_mesh_count = guard.detail_mesh->nmeshes;
		summary->detail_vertex_count = guard.detail_mesh->nverts;
		summary->detail_triangle_count = guard.detail_mesh->ntris;
	}

	/* Success: release runtime from the guard so it is not destroyed */
	qnn_navmesh_runtime_t *result = guard.runtime;
	guard.runtime = nullptr;
	return result;
}

extern "C" int qnn_navmesh_find_nearest(
	const qnn_navmesh_runtime_t *navmesh,
	const float *point,
	qnn_navmesh_nearest_result_t *result,
	char *error,
	size_t error_size)
{
	dtPolyRef nearest_ref;
	float nearest_pt[3];
	float poly_center[3];
	float hit_pos[3];
	float hit_normal[3];
	float wall_distance;
	bool is_over_poly;
	const dtMeshTile *tile;
	const dtPoly *poly;
	dtQueryFilter filter; qnn_navmesh_setup_filter(&filter);
	dtStatus status;

	if (result == nullptr || point == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Nearest navmesh query requires a result buffer and point");
		return 0;
	}
	memset(result, 0, sizeof(*result));
	memcpy(result->query_point, point, sizeof(result->query_point));

	nearest_ref = 0;
	memset(nearest_pt, 0, sizeof(nearest_pt));
	is_over_poly = false;
	if (!qnn_navmesh_find_nearest_internal(navmesh, point, &nearest_ref, nearest_pt, &is_over_poly, NULL, error, error_size))
		return 0;

	if (dtStatusFailed(navmesh->navmesh->getTileAndPolyByRef(nearest_ref, &tile, &poly)))
	{
		qnn_nav_set_error(error, error_size, "Detour could not resolve the nearest polygon reference");
		return 0;
	}

	qnn_navmesh_poly_center(tile, poly, poly_center);
	wall_distance = 0.0f;
	memset(hit_pos, 0, sizeof(hit_pos));
	memset(hit_normal, 0, sizeof(hit_normal));
	status = navmesh->query->findDistanceToWall(nearest_ref, nearest_pt, 4096.0f, &filter, &wall_distance, hit_pos, hit_normal);
	if (dtStatusFailed(status))
		wall_distance = 0.0f;

	result->found = 1;
	result->is_over_poly = is_over_poly ? 1 : 0;
	result->poly_ref = static_cast<unsigned long long>(nearest_ref);
	qnn_recast_to_quake(nearest_pt, result->nearest_point);
	qnn_recast_to_quake(poly_center, result->poly_center);
	result->wall_distance = wall_distance;
	result->neighbor_count = qnn_navmesh_collect_neighbors(navmesh, nearest_ref, result->neighbor_refs, QNN_NAVMESH_MAX_NEIGHBORS);
	return 1;
}

extern "C" int qnn_navmesh_collect_polys(
	const qnn_navmesh_runtime_t *navmesh,
	qnn_navmesh_poly_record_t **records,
	int *record_count,
	char *error,
	size_t error_size)
{
	int total;
	int tile_index;
	int write_index;
	qnn_navmesh_poly_record_t *out;

	if (records == nullptr || record_count == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Poly enumeration requires output pointers");
		return 0;
	}
	*records = nullptr;
	*record_count = 0;
	if (navmesh == nullptr || navmesh->navmesh == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Navmesh polygons requested before navmesh was initialized");
		return 0;
	}

	total = 0;
	const dtNavMesh *detour_navmesh = navmesh->navmesh;
	for (tile_index = 0; tile_index < navmesh->navmesh->getMaxTiles(); ++tile_index)
	{
		const dtMeshTile *tile;
		int poly_index;

		tile = detour_navmesh->getTile(tile_index);
		if (tile == nullptr || tile->header == nullptr || tile->polys == nullptr)
			continue;
		for (poly_index = 0; poly_index < tile->header->polyCount; ++poly_index)
		{
			if (tile->polys[poly_index].getType() == DT_POLYTYPE_OFFMESH_CONNECTION)
				continue;
			total += 1;
		}
	}
	if (total <= 0)
	{
		qnn_nav_set_error(error, error_size, "Detour navmesh does not contain any polygons");
		return 0;
	}

	out = static_cast<qnn_navmesh_poly_record_t *>(calloc(static_cast<size_t>(total), sizeof(*out)));
	if (out == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Out of memory while enumerating navmesh polygons");
		return 0;
	}

	write_index = 0;
	for (tile_index = 0; tile_index < navmesh->navmesh->getMaxTiles(); ++tile_index)
	{
		const dtMeshTile *tile;
		dtPolyRef base_ref;
		int poly_index;

		tile = detour_navmesh->getTile(tile_index);
		if (tile == nullptr || tile->header == nullptr || tile->polys == nullptr)
			continue;
		base_ref = detour_navmesh->getPolyRefBase(tile);
		for (poly_index = 0; poly_index < tile->header->polyCount; ++poly_index)
		{
			const dtPoly *poly;
			float center[3];
			float mins[3];
			float maxs[3];
			dtPolyRef ref;

			poly = &tile->polys[poly_index];
			if (poly->getType() == DT_POLYTYPE_OFFMESH_CONNECTION)
				continue;

			ref = base_ref | static_cast<dtPolyRef>(poly_index);
			qnn_navmesh_poly_center(tile, poly, center);
			qnn_navmesh_poly_bounds(tile, poly, mins, maxs);
			out[write_index].poly_ref = static_cast<unsigned long long>(ref);
			qnn_recast_to_quake(center, out[write_index].center);
			qnn_recast_to_quake(mins, out[write_index].bounds_min);
			qnn_recast_to_quake(maxs, out[write_index].bounds_max);
			out[write_index].neighbor_count = qnn_navmesh_collect_neighbors(navmesh, ref, out[write_index].neighbor_refs, QNN_NAVMESH_MAX_NEIGHBORS);
			write_index += 1;
		}
	}

	*records = out;
	*record_count = write_index;
	return 1;
}

extern "C" int qnn_navmesh_find_path(
	const qnn_navmesh_runtime_t *navmesh,
	const float *start,
	const float *end,
	qnn_navmesh_path_result_t *result,
	char *error,
	size_t error_size)
{
	dtPolyRef start_ref;
	dtPolyRef end_ref;
	float start_nearest[3];
	float end_nearest[3];
	bool start_over_poly;
	bool end_over_poly;
	dtQueryFilter filter; qnn_navmesh_setup_filter(&filter);
	dtPolyRef path_refs[QNN_NAVMESH_MAX_PATH_REFS];
	int path_count;
	float straight_path[QNN_NAVMESH_MAX_STRAIGHT_POINTS * 3];
	unsigned char straight_flags[QNN_NAVMESH_MAX_STRAIGHT_POINTS];
	dtPolyRef straight_refs[QNN_NAVMESH_MAX_STRAIGHT_POINTS];
	int straight_count;
	dtStatus status;
	int i;

	if (result == nullptr || start == nullptr || end == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Path navmesh query requires result, start, and end points");
		return 0;
	}
	memset(result, 0, sizeof(*result));

	start_ref = 0;
	end_ref = 0;
	memset(start_nearest, 0, sizeof(start_nearest));
	memset(end_nearest, 0, sizeof(end_nearest));
	start_over_poly = false;
	end_over_poly = false;
	if (!qnn_navmesh_find_nearest_internal(navmesh, start, &start_ref, start_nearest, &start_over_poly, navmesh->query_half_extents_tight, error, error_size)
		|| !qnn_navmesh_find_nearest_internal(navmesh, end, &end_ref, end_nearest, &end_over_poly, NULL, error, error_size))
		return 0;

	path_count = 0;
	status = navmesh->query->findPath(
		start_ref,
		end_ref,
		start_nearest,
		end_nearest,
		&filter,
		path_refs,
		&path_count,
		QNN_NAVMESH_MAX_PATH_REFS);
	if (dtStatusFailed(status) || path_count <= 0)
	{
		qnn_nav_set_error(error, error_size, "Detour findPath failed");
		return 0;
	}

	straight_count = 0;
	status = navmesh->query->findStraightPath(
		start_nearest,
		end_nearest,
		path_refs,
		path_count,
		straight_path,
		straight_flags,
		straight_refs,
		&straight_count,
		QNN_NAVMESH_MAX_STRAIGHT_POINTS,
		0);
	if (dtStatusFailed(status) || straight_count <= 0)
	{
		qnn_nav_set_error(error, error_size, "Detour findStraightPath failed");
		return 0;
	}

	result->found = 1;
	result->start_ref = static_cast<unsigned long long>(start_ref);
	result->end_ref = static_cast<unsigned long long>(end_ref);
	qnn_recast_to_quake(start_nearest, result->start_point);
	qnn_recast_to_quake(end_nearest, result->end_point);
	result->path_ref_count = path_count;
	for (i = 0; i < path_count; ++i)
		result->path_refs[i] = static_cast<unsigned long long>(path_refs[i]);
	result->straight_point_count = straight_count;
	for (i = 0; i < straight_count; ++i)
		qnn_recast_to_quake(&straight_path[i * 3], &result->straight_points[i * 3]);
	result->travel_distance = qnn_navmesh_path_distance(result->straight_points, straight_count);
	return 1;
}

extern "C" void qnn_navmesh_free_poly_records(qnn_navmesh_poly_record_t *records)
{
	free(records);
}

extern "C" void qnn_navmesh_destroy(qnn_navmesh_runtime_t *navmesh)
{
	if (navmesh == nullptr)
		return;
	if (navmesh->query != nullptr)
		dtFreeNavMeshQuery(navmesh->query);
	if (navmesh->navmesh != nullptr)
		dtFreeNavMesh(navmesh->navmesh);
	delete navmesh;
}

extern "C" void qnn_navmesh_write_summary_json(FILE *out, const qnn_navmesh_summary_t *summary)
{
	qnn_navmesh_summary_t empty;

	if (summary == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		summary = &empty;
	}
	fprintf(
		out,
		"{\"detail_mesh_count\":%d,\"detail_triangle_count\":%d,\"detail_vertex_count\":%d,"
		"\"input_triangle_count\":%d,\"input_vertex_count\":%d,\"navmesh_vertex_count\":%d,\"polygon_count\":%d}",
		summary->detail_mesh_count,
		summary->detail_triangle_count,
		summary->detail_vertex_count,
		summary->input_triangle_count,
		summary->input_vertex_count,
		summary->navmesh_vertex_count,
		summary->polygon_count);
}

extern "C" void qnn_navmesh_write_nearest_json(FILE *out, const qnn_navmesh_nearest_result_t *result)
{
	qnn_navmesh_nearest_result_t empty;
	int i;

	if (result == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		result = &empty;
	}
	fprintf(out, "{\"found\":%s,\"is_over_poly\":%s,\"neighbor_count\":%d,\"neighbor_refs\":[",
		result->found ? "true" : "false",
		result->is_over_poly ? "true" : "false",
		result->neighbor_count);
	for (i = 0; i < result->neighbor_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "\"%llu\"", result->neighbor_refs[i]);
	}
	fprintf(out, "],\"nearest_point\":[%.3f,%.3f,%.3f],\"poly_center\":[%.3f,%.3f,%.3f],\"poly_ref\":\"%llu\",\"query_point\":[%.3f,%.3f,%.3f],\"wall_distance\":%.3f}",
		result->nearest_point[0], result->nearest_point[1], result->nearest_point[2],
		result->poly_center[0], result->poly_center[1], result->poly_center[2],
		result->poly_ref,
		result->query_point[0], result->query_point[1], result->query_point[2],
		result->wall_distance);
}

extern "C" void qnn_navmesh_write_path_json(FILE *out, const qnn_navmesh_path_result_t *result)
{
	qnn_navmesh_path_result_t empty;
	int i;

	if (result == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		result = &empty;
	}
	fprintf(out, "{\"end_point\":[%.3f,%.3f,%.3f],\"end_ref\":\"%llu\",\"found\":%s,\"path_ref_count\":%d,\"path_refs\":[",
		result->end_point[0], result->end_point[1], result->end_point[2],
		result->end_ref,
		result->found ? "true" : "false",
		result->path_ref_count);
	for (i = 0; i < result->path_ref_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "\"%llu\"", result->path_refs[i]);
	}
	fprintf(out, "],\"start_point\":[%.3f,%.3f,%.3f],\"start_ref\":\"%llu\",\"straight_path\":[",
		result->start_point[0], result->start_point[1], result->start_point[2],
		result->start_ref);
	for (i = 0; i < result->straight_point_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "[%.3f,%.3f,%.3f]",
			result->straight_points[i * 3 + 0],
			result->straight_points[i * 3 + 1],
			result->straight_points[i * 3 + 2]);
	}
	fprintf(out, "],\"straight_point_count\":%d,\"travel_distance\":%.3f}",
		result->straight_point_count,
		result->travel_distance);
}

/* Geometry extraction, default config, and build_from_worldmodel
   moved to qnn_map.c — map.c is the engine interface. */

