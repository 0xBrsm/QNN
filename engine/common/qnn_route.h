#ifndef QNN_ROUTE_H
#define QNN_ROUTE_H

#include <stddef.h>
#include <stdio.h>

#include "qnn_navmesh.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef QNN_ROUTE_RUNTIME_FWD
#define QNN_ROUTE_RUNTIME_FWD
typedef struct qnn_route_runtime_s qnn_route_runtime_t;
#endif

#define QNN_ROUTE_MAX_AREAS 128

typedef enum
{
	QNN_TRAVEL_INVALID = 0,
	QNN_TRAVEL_WALK = 1,
	QNN_TRAVEL_DROP = 2,
	QNN_TRAVEL_TELEPORT = 3,
	QNN_TRAVEL_ELEVATOR = 4,
	QNN_TRAVEL_PUSH = 5
} qnn_route_travel_type_t;

typedef struct
{
	const char *key;
	const char *value;
} qnn_route_property_view_t;

typedef struct
{
	const char *object_id;
	const char *category;
	const char *classname;
	const float *origin;
	const float *angles;
	const qnn_route_property_view_t *properties;
	int property_count;
} qnn_route_static_object_view_t;

typedef struct
{
	int	area_count;
	int	cluster_count;
	int	min_cluster_area_count;
	int	max_cluster_area_count;
	float	avg_cluster_area_count;
	int	walk_link_count;
	int	teleport_link_count;
	int	lift_link_count;
	int	push_link_count;
	int	drop_link_count;
	int	total_link_count;
} qnn_route_summary_t;

typedef struct
{
	int	found;
	int	area_id;
	unsigned long long poly_ref;
	float	query_point[3];
	float	nearest_point[3];
	float	center[3];
	float	bounds_min[3];
	float	bounds_max[3];
	int	cluster_id;
	int	link_count;
	int	special_link_count;
} qnn_route_area_result_t;

typedef struct
{
	int	found;
	int	area_id;
	int	cluster_id;
	float	query_point[3];
	float	nearest_point[3];
	float	center[3];
	float	bounds_min[3];
	float	bounds_max[3];
	int	area_count;
	int	exit_count;
	int	special_exit_count;
} qnn_route_cluster_result_t;

typedef struct
{
	int	found;
	int	start_area_id;
	int	end_area_id;
	int	next_area_id;
	int	first_link_id;
	qnn_route_travel_type_t first_travel_type;
	float	travel_time;
	int	area_count;
	int	area_ids[QNN_ROUTE_MAX_AREAS];
	int	link_count;
	int	link_ids[QNN_ROUTE_MAX_AREAS - 1];
	int	travel_types[QNN_ROUTE_MAX_AREAS - 1];
} qnn_route_route_result_t;

qnn_route_runtime_t *QNN_RouteBuild(
	const qnn_navmesh_runtime_t *navmesh,
	const qnn_route_static_object_view_t *static_objects,
	int static_object_count,
	qnn_route_summary_t *summary,
	char *error,
	size_t error_size);
int QNN_RouteFindArea(
	const qnn_route_runtime_t *oracle,
	const float *point,
	qnn_route_area_result_t *result,
	char *error,
	size_t error_size);
int QNN_RouteFindCluster(
	const qnn_route_runtime_t *oracle,
	const float *point,
	qnn_route_cluster_result_t *result,
	char *error,
	size_t error_size);
int QNN_RouteFind(
	const qnn_route_runtime_t *oracle,
	const float *start,
	const float *end,
	qnn_route_route_result_t *result,
	char *error,
	size_t error_size);
void QNN_RouteDestroy(qnn_route_runtime_t *oracle);
void QNN_RouteWriteAreaJson(FILE *out, const qnn_route_area_result_t *result);
void QNN_RouteWriteClusterJson(FILE *out, const qnn_route_cluster_result_t *result);
void QNN_RouteWriteRouteJson(FILE *out, const qnn_route_route_result_t *result);
const char *QNN_RouteTravelTypeName(qnn_route_travel_type_t travel_type);
int QNN_RoutePathPosition(
	const qnn_route_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	const float *player_pos,
	const float *object_pos,
	float *out_rel,
	float *out_route_cost,
	char *error,
	size_t error_size);
int QNN_RouteGetClusters(
	const qnn_route_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	int *out_cluster_ids,
	int max_clusters,
	int *out_cluster_count);

/* Like QNN_RoutePathPosition but returns the Nth cheapest alternative
   (0 = best, 1 = second best, etc.).  Returns 0 if no Nth path exists. */
int QNN_RoutePathPositionNth(
	const qnn_route_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	const float *player_pos,
	const float *object_pos,
	int nth,
	float *out_rel,
	float *out_route_cost,
	char *error,
	size_t error_size);

#ifdef __cplusplus
} /* close extern "C" */

/* ══════════════════════════════════════════════════════════════════
 * C++ internals — shared across qnn_link.cpp, qnn_cluster.cpp,
 * and qnn_route.cpp.  Not visible to C translation units.
 * ══════════════════════════════════════════════════════════════════ */

#include <cmath>
#include <cstring>
#include <vector>

/* ── Constants ──────────────────────────────────────────────────── */

constexpr float kWalkUnitsPerSecond = 300.0f;
constexpr float kTeleportTravelTime = 0.10f;
constexpr float kLiftBasePenalty = 0.50f;
constexpr float kPushGravity = 800.0f;
constexpr float kPushSampleStep = 0.05f;
constexpr float kPushMaxTime = 2.50f;
constexpr float kPushMaxSnapDistance = 192.0f;
constexpr float kDropMinHeight = 56.0f;
constexpr float kDropMaxHeight = 320.0f;
constexpr float kDropMaxHorizontalGap = 24.0f;
constexpr int kDropMaxLinksPerArea = 2;
constexpr int kClusterMinAreaCount = 8;
constexpr int kClusterTargetAreaCount = 20;
constexpr int kClusterMaxAreaCount = 32;
constexpr float kClusterCostEpsilon = 0.0001f;

/* ── Internal structures ────────────────────────────────────────── */

struct QnnNavArea
{
	int area_id;
	unsigned long long poly_ref;
	float center[3];
	float bounds_min[3];
	float bounds_max[3];
	int cluster_id;
	int link_count;
	int special_link_count;
};

struct QnnNavLink
{
	int link_id;
	int src_area_id;
	int dst_area_id;
	qnn_route_travel_type_t travel_type;
	float start_pos[3];
	float end_pos[3];
	float travel_time;
};

struct QnnNavCluster
{
	int cluster_id;
	int first_area_id;
	int area_count;
	float center[3];
	float bounds_min[3];
	float bounds_max[3];
	int exit_count;
	int special_exit_count;
};

struct QnnRouteEntry
{
	int link_id;
	float cost;
	int exit_cluster;	/* cluster the route exits into (-1 = same cluster) */
	float exit_pos[3];	/* position of the cluster boundary crossing */
};

struct qnn_route_runtime_s
{
	const qnn_navmesh_runtime_t *navmesh;
	std::vector<QnnNavArea> areas;
	std::vector<QnnNavCluster> clusters;
	std::vector<QnnNavLink> links;
	std::vector<std::vector<int>> outgoing_links;
	qnn_route_summary_t summary;
	/* Precomputed routing cache — flattened area_count * area_count tables.
	   Index: src * area_count + dst.
	   route_entries[i] holds one entry per distinct cluster exit from src
	   that can reach dst, including the total route cost and the boundary
	   crossing position.  The model sees N alternative routes, one per exit. */
	std::vector<std::vector<QnnRouteEntry>> route_entries;
};

/* ── Shared inline helpers ──────────────────────────────────────── */

inline float RouteDistance(const float *lhs, const float *rhs)
{
	const float dx = lhs[0] - rhs[0];
	const float dy = lhs[1] - rhs[1];
	const float dz = lhs[2] - rhs[2];
	return sqrtf(dx * dx + dy * dy + dz * dz);
}

inline int IsSpecialTravel(qnn_route_travel_type_t travel_type)
{
	return travel_type != QNN_TRAVEL_INVALID
		&& travel_type != QNN_TRAVEL_WALK;
}

/* ── Cross-TU internal functions ────────────────────────────────── */

/* qnn_link.cpp */
void QNN_LinkBuildWalk(qnn_route_runtime_t *oracle, const qnn_navmesh_poly_record_t *records, int record_count);
void QNN_LinkBuildTeleport(qnn_route_runtime_t *oracle, const qnn_route_static_object_view_t *static_objects, int static_object_count);
void QNN_LinkBuildLift(qnn_route_runtime_t *oracle, const qnn_route_static_object_view_t *static_objects, int static_object_count);
void QNN_LinkBuildPush(qnn_route_runtime_t *oracle, const qnn_route_static_object_view_t *static_objects, int static_object_count);
void QNN_LinkBuildDrop(qnn_route_runtime_t *oracle);

/* qnn_cluster.cpp */
void QNN_ClusterBuild(qnn_route_runtime_t *oracle);
void QNN_ClusterBuildRoutingCache(qnn_route_runtime_t *oracle);
void QNN_ClusterFillSummaryCounts(qnn_route_runtime_t *oracle);
const QnnRouteEntry *QNN_ClusterFindBestRouteEntry(const std::vector<QnnRouteEntry> &entries);
int QNN_ClusterWalkCachedRoute(
	const qnn_route_runtime_t *oracle,
	int start_area_id, int end_area_id,
	int *out_area_ids, int *out_link_ids,
	int max_areas);

/* Shared helpers used by link + route (defined in qnn_link.cpp) */
int QNN_LinkAreaForRef(const qnn_route_runtime_t *oracle, unsigned long long poly_ref);
int QNN_LinkFindNearestAreaId(
	const qnn_route_runtime_t *oracle,
	const float *point,
	qnn_navmesh_nearest_result_t *nearest,
	char *error, size_t error_size);
void QNN_LinkAdd(
	qnn_route_runtime_t *oracle,
	int src_area_id, int dst_area_id,
	qnn_route_travel_type_t travel_type,
	const float *start_pos, const float *end_pos,
	float travel_time);

#endif /* __cplusplus */

#endif /* QNN_ROUTE_H */
