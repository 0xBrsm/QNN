extern "C" {
#include "qnn.h"
#include "qnn_map.h"
}

#include "qnn_route.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

extern "C" const char *QNN_RouteTravelTypeName(qnn_route_travel_type_t travel_type)
{
	switch (travel_type)
	{
	case QNN_TRAVEL_WALK:
		return "WALK";
	case QNN_TRAVEL_DROP:
		return "DROP";
	case QNN_TRAVEL_TELEPORT:
		return "TELEPORT";
	case QNN_TRAVEL_ELEVATOR:
		return "ELEVATOR";
	case QNN_TRAVEL_PUSH:
		return "PUSH";
	default:
		return "INVALID";
	}
}

extern "C" qnn_route_runtime_t *QNN_RouteBuild(
	const qnn_navmesh_runtime_t *navmesh,
	const qnn_route_static_object_view_t *static_objects,
	int static_object_count,
	qnn_route_summary_t *summary,
	char *error,
	size_t error_size)
{
	qnn_navmesh_poly_record_t *records;
	int record_count;
	qnn_route_runtime_t *oracle;
	int area_index;

	if (summary != nullptr)
		memset(summary, 0, sizeof(*summary));
	if (navmesh == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Navigation oracle requires a valid navmesh");
		return nullptr;
	}

	records = nullptr;
	record_count = 0;
	if (!qnn_navmesh_collect_polys(navmesh, &records, &record_count, error, error_size))
		return nullptr;

	std::sort(records, records + record_count, [](const qnn_navmesh_poly_record_t &lhs, const qnn_navmesh_poly_record_t &rhs) {
		return lhs.poly_ref < rhs.poly_ref;
	});

	oracle = new (std::nothrow) qnn_route_runtime_t();
	if (oracle == nullptr)
	{
		qnn_navmesh_free_poly_records(records);
		qnn_nav_set_error(error, error_size, "Out of memory while allocating nav oracle runtime");
		return nullptr;
	}
	oracle->navmesh = navmesh;
	oracle->areas.reserve((size_t)record_count);
	oracle->outgoing_links.resize((size_t)record_count);

	for (area_index = 0; area_index < record_count; ++area_index)
	{
		QnnNavArea area;

		memset(&area, 0, sizeof(area));
		area.area_id = area_index;
		area.poly_ref = records[area_index].poly_ref;
		memcpy(area.center, records[area_index].center, sizeof(area.center));
		memcpy(area.bounds_min, records[area_index].bounds_min, sizeof(area.bounds_min));
		memcpy(area.bounds_max, records[area_index].bounds_max, sizeof(area.bounds_max));
		oracle->areas.push_back(area);
	}

	QNN_LinkBuildWalk(oracle, records, record_count);
	QNN_LinkBuildTeleport(oracle, static_objects, static_object_count);
	QNN_LinkBuildLift(oracle, static_objects, static_object_count);
	QNN_LinkBuildPush(oracle, static_objects, static_object_count);
	QNN_LinkBuildDrop(oracle);
	QNN_ClusterBuild(oracle);

	QNN_ClusterFillSummaryCounts(oracle);
	QNN_ClusterBuildRoutingCache(oracle);
	if (summary != nullptr)
		*summary = oracle->summary;

	qnn_navmesh_free_poly_records(records);
	return oracle;
}

extern "C" int QNN_RouteFindArea(
	const qnn_route_runtime_t *oracle,
	const float *point,
	qnn_route_area_result_t *result,
	char *error,
	size_t error_size)
{
	qnn_navmesh_nearest_result_t nearest;
	int area_id;
	const QnnNavArea *area;

	if (result == nullptr || point == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Area query requires a result buffer and point");
		return 0;
	}
	memset(result, 0, sizeof(*result));
	result->area_id = -1;
	result->cluster_id = -1;
	memcpy(result->query_point, point, sizeof(result->query_point));
	if (oracle == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Area query requested before nav oracle was initialized");
		return 0;
	}

	memset(&nearest, 0, sizeof(nearest));
	area_id = QNN_LinkFindNearestAreaId(oracle, point, &nearest, error, error_size);
	if (area_id < 0)
		return 0;

	area = &oracle->areas[(size_t)area_id];
	result->found = 1;
	result->area_id = area_id;
	result->poly_ref = nearest.poly_ref;
	memcpy(result->nearest_point, nearest.nearest_point, sizeof(result->nearest_point));
	memcpy(result->center, area->center, sizeof(result->center));
	memcpy(result->bounds_min, area->bounds_min, sizeof(result->bounds_min));
	memcpy(result->bounds_max, area->bounds_max, sizeof(result->bounds_max));
	result->cluster_id = area->cluster_id;
	result->link_count = area->link_count;
	result->special_link_count = area->special_link_count;
	return 1;
}

extern "C" int QNN_RouteFindCluster(
	const qnn_route_runtime_t *oracle,
	const float *point,
	qnn_route_cluster_result_t *result,
	char *error,
	size_t error_size)
{
	qnn_navmesh_nearest_result_t nearest;
	int area_id;
	const QnnNavArea *area;
	const QnnNavCluster *cluster;

	if (result == nullptr || point == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Cluster query requires a result buffer and point");
		return 0;
	}
	memset(result, 0, sizeof(*result));
	result->area_id = -1;
	result->cluster_id = -1;
	memcpy(result->query_point, point, sizeof(result->query_point));
	if (oracle == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Cluster query requested before nav oracle was initialized");
		return 0;
	}

	memset(&nearest, 0, sizeof(nearest));
	area_id = QNN_LinkFindNearestAreaId(oracle, point, &nearest, error, error_size);
	if (area_id < 0)
		return 0;

	area = &oracle->areas[(size_t)area_id];
	if (area->cluster_id < 0 || area->cluster_id >= (int)oracle->clusters.size())
	{
		qnn_nav_set_error(error, error_size, "Nearest nav area was missing a valid cluster assignment");
		return 0;
	}

	cluster = &oracle->clusters[(size_t)area->cluster_id];
	result->found = 1;
	result->area_id = area_id;
	result->cluster_id = cluster->cluster_id;
	memcpy(result->nearest_point, nearest.nearest_point, sizeof(result->nearest_point));
	memcpy(result->center, cluster->center, sizeof(result->center));
	memcpy(result->bounds_min, cluster->bounds_min, sizeof(result->bounds_min));
	memcpy(result->bounds_max, cluster->bounds_max, sizeof(result->bounds_max));
	result->area_count = cluster->area_count;
	result->exit_count = cluster->exit_count;
	result->special_exit_count = cluster->special_exit_count;
	return 1;
}

extern "C" int QNN_RouteFind(
	const qnn_route_runtime_t *oracle,
	const float *start,
	const float *end,
	qnn_route_route_result_t *result,
	char *error,
	size_t error_size)
{
	qnn_navmesh_nearest_result_t start_nearest;
	qnn_navmesh_nearest_result_t end_nearest;
	int start_area_id;
	int end_area_id;
	int area_count;

	if (result == nullptr || start == nullptr || end == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Route query requires result, start, and end points");
		return 0;
	}
	memset(result, 0, sizeof(*result));
	result->start_area_id = -1;
	result->end_area_id = -1;
	result->next_area_id = -1;
	result->first_link_id = -1;
	result->first_travel_type = QNN_TRAVEL_INVALID;

	if (oracle == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Route query requested before nav oracle was initialized");
		return 0;
	}

	memset(&start_nearest, 0, sizeof(start_nearest));
	memset(&end_nearest, 0, sizeof(end_nearest));
	start_area_id = QNN_LinkFindNearestAreaId(oracle, start, &start_nearest, error, error_size);
	if (start_area_id < 0)
		return 0;
	end_area_id = QNN_LinkFindNearestAreaId(oracle, end, &end_nearest, error, error_size);
	if (end_area_id < 0)
		return 0;

	result->start_area_id = start_area_id;
	result->end_area_id = end_area_id;

	if (start_area_id == end_area_id)
	{
		result->found = 1;
		result->next_area_id = end_area_id;
		result->travel_time = 0.0f;
		result->area_count = 1;
		result->area_ids[0] = start_area_id;
		return 1;
	}

	area_count = (int)oracle->areas.size();

	{
		const size_t start_idx = (size_t)start_area_id * (size_t)area_count + (size_t)end_area_id;
		const QnnRouteEntry *best_entry = QNN_ClusterFindBestRouteEntry(oracle->route_entries[start_idx]);
		int walked_link_ids[QNN_ROUTE_MAX_AREAS - 1];
		int walked_area_count;

		memset(walked_link_ids, -1, sizeof(walked_link_ids));

		if (best_entry == nullptr)
			return 0;

		result->found = 1;
		result->travel_time = best_entry->cost;

		walked_area_count = QNN_ClusterWalkCachedRoute(
			oracle, start_area_id, end_area_id,
			result->area_ids, walked_link_ids,
			QNN_ROUTE_MAX_AREAS);
		result->area_count = walked_area_count;

		result->link_count = 0;
		for (int i = 0; i < walked_area_count - 1 && i < QNN_ROUTE_MAX_AREAS - 1; ++i)
		{
			if (walked_link_ids[i] >= 0)
			{
				result->link_ids[result->link_count] = walked_link_ids[i];
				result->travel_types[result->link_count] = (int)oracle->links[(size_t)walked_link_ids[i]].travel_type;
				result->link_count += 1;
			}
		}

		if (result->link_count > 0)
		{
			const QnnNavLink &first_link = oracle->links[(size_t)result->link_ids[0]];

			result->first_link_id = first_link.link_id;
			result->first_travel_type = first_link.travel_type;
			result->next_area_id = first_link.dst_area_id;
		}
	}
	return 1;
}

extern "C" void QNN_RouteDestroy(qnn_route_runtime_t *oracle)
{
	delete oracle;
}

extern "C" void QNN_RouteWriteAreaJson(FILE *out, const qnn_route_area_result_t *result)
{
	qnn_route_area_result_t empty;

	if (result == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		empty.area_id = -1;
		empty.cluster_id = -1;
		result = &empty;
	}
	fprintf(
		out,
		"{\"area_id\":%d,\"bounds_max\":[%.3f,%.3f,%.3f],\"bounds_min\":[%.3f,%.3f,%.3f],"
		"\"center\":[%.3f,%.3f,%.3f],\"cluster_id\":%d,\"found\":%s,\"link_count\":%d,"
		"\"nearest_point\":[%.3f,%.3f,%.3f],"
		"\"poly_ref\":\"%llu\",\"query_point\":[%.3f,%.3f,%.3f],\"special_link_count\":%d}",
		result->area_id,
		result->bounds_max[0], result->bounds_max[1], result->bounds_max[2],
		result->bounds_min[0], result->bounds_min[1], result->bounds_min[2],
		result->center[0], result->center[1], result->center[2],
		result->cluster_id,
		result->found ? "true" : "false",
		result->link_count,
		result->nearest_point[0], result->nearest_point[1], result->nearest_point[2],
		result->poly_ref,
		result->query_point[0], result->query_point[1], result->query_point[2],
		result->special_link_count);
}

extern "C" void QNN_RouteWriteClusterJson(FILE *out, const qnn_route_cluster_result_t *result)
{
	qnn_route_cluster_result_t empty;

	if (result == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		empty.area_id = -1;
		empty.cluster_id = -1;
		result = &empty;
	}
	fprintf(
		out,
		"{\"area_count\":%d,\"area_id\":%d,\"bounds_max\":[%.3f,%.3f,%.3f],"
		"\"bounds_min\":[%.3f,%.3f,%.3f],\"center\":[%.3f,%.3f,%.3f],\"cluster_id\":%d,"
		"\"exit_count\":%d,\"found\":%s,\"nearest_point\":[%.3f,%.3f,%.3f],"
		"\"query_point\":[%.3f,%.3f,%.3f],\"special_exit_count\":%d}",
		result->area_count,
		result->area_id,
		result->bounds_max[0], result->bounds_max[1], result->bounds_max[2],
		result->bounds_min[0], result->bounds_min[1], result->bounds_min[2],
		result->center[0], result->center[1], result->center[2],
		result->cluster_id,
		result->exit_count,
		result->found ? "true" : "false",
		result->nearest_point[0], result->nearest_point[1], result->nearest_point[2],
		result->query_point[0], result->query_point[1], result->query_point[2],
		result->special_exit_count);
}

extern "C" void QNN_RouteWriteRouteJson(FILE *out, const qnn_route_route_result_t *result)
{
	qnn_route_route_result_t empty;
	int index;

	if (result == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		empty.start_area_id = -1;
		empty.end_area_id = -1;
		empty.next_area_id = -1;
		empty.first_link_id = -1;
		empty.first_travel_type = QNN_TRAVEL_INVALID;
		result = &empty;
	}
	fprintf(out, "{\"area_count\":%d,\"area_ids\":[", result->area_count);
	for (index = 0; index < result->area_count; ++index)
	{
		if (index > 0)
			fputc(',', out);
		fprintf(out, "%d", result->area_ids[index]);
	}
	fprintf(out, "],\"end_area_id\":%d,\"first_link_id\":%d,\"first_travel_type\":\"%s\"",
		result->end_area_id,
		result->first_link_id,
		QNN_RouteTravelTypeName(result->first_travel_type));
	fprintf(out, ",\"found\":%s,\"link_count\":%d,\"link_ids\":[",
		result->found ? "true" : "false",
		result->link_count);
	for (index = 0; index < result->link_count; ++index)
	{
		if (index > 0)
			fputc(',', out);
		fprintf(out, "%d", result->link_ids[index]);
	}
	fprintf(out, "],\"next_area_id\":%d,\"start_area_id\":%d,\"travel_time\":%.3f,\"travel_types\":[",
		result->next_area_id,
		result->start_area_id,
		result->travel_time);
	for (index = 0; index < result->link_count; ++index)
	{
		if (index > 0)
			fputc(',', out);
		fprintf(out, "\"%s\"", QNN_RouteTravelTypeName((qnn_route_travel_type_t)result->travel_types[index]));
	}
	fprintf(out, "]}");
}

extern "C" int QNN_RoutePathPosition(
	const qnn_route_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	const float *player_pos,
	const float *object_pos,
	float *out_rel,
	float *out_route_cost,
	char *error,
	size_t error_size)
{
	int area_count;
	int axis;

	if (player_pos == nullptr || object_pos == nullptr || out_rel == nullptr || out_route_cost == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Path position query requires player/object positions and output buffers");
		return 0;
	}

	out_rel[0] = 0.0f;
	out_rel[1] = 0.0f;
	out_rel[2] = 0.0f;
	*out_route_cost = 0.0f;

	if (oracle == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Path position query requested before nav oracle was initialized");
		return 0;
	}

	area_count = (int)oracle->areas.size();
	if (player_area_id < 0 || player_area_id >= area_count || object_area_id < 0 || object_area_id >= area_count)
	{
		qnn_nav_set_error(error, error_size, "Path position query requires valid player/object area ids");
		return 0;
	}

	if (player_area_id == object_area_id)
	{
		for (axis = 0; axis < 3; ++axis)
			out_rel[axis] = object_pos[axis] - player_pos[axis];
		*out_route_cost = 0.0f;
		return 1;
	}

	{
		const size_t route_index = (size_t)player_area_id * (size_t)area_count + (size_t)object_area_id;
		const QnnRouteEntry *best_entry = QNN_ClusterFindBestRouteEntry(oracle->route_entries[route_index]);

		if (best_entry == nullptr)
		{
			for (axis = 0; axis < 3; ++axis)
				out_rel[axis] = object_pos[axis] - player_pos[axis];
			*out_route_cost = -1.0f;
			return 1;
		}

		if (best_entry->link_id < 0 || best_entry->link_id >= (int)oracle->links.size())
		{
			qnn_nav_set_error(error, error_size, "Path position query encountered an invalid cached route link");
			return 0;
		}

		{
			const float *target = best_entry->exit_cluster >= 0
				? best_entry->exit_pos
				: object_pos;

			for (axis = 0; axis < 3; ++axis)
				out_rel[axis] = target[axis] - player_pos[axis];
			*out_route_cost = best_entry->cost;
		}
	}

	return 1;
}

extern "C" int QNN_RouteGetClusters(
	const qnn_route_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	int *out_cluster_ids,
	int max_clusters,
	int *out_cluster_count)
{
	int area_count;
	int count;
	int last_cluster;

	if (out_cluster_count != nullptr)
		*out_cluster_count = 0;
	if (oracle == nullptr || out_cluster_ids == nullptr || out_cluster_count == nullptr || max_clusters <= 0)
		return 0;

	area_count = (int)oracle->areas.size();
	if (player_area_id < 0 || player_area_id >= area_count
		|| object_area_id < 0 || object_area_id >= area_count)
		return 0;

	if (player_area_id == object_area_id)
		return 1;

	{
		int walked_areas[QNN_ROUTE_MAX_AREAS];
		int walked_area_count;
		int player_cluster = oracle->areas[(size_t)player_area_id].cluster_id;
		int object_cluster = oracle->areas[(size_t)object_area_id].cluster_id;

		walked_area_count = QNN_ClusterWalkCachedRoute(
			oracle, player_area_id, object_area_id,
			walked_areas, nullptr,
			QNN_ROUTE_MAX_AREAS);

		count = 0;
		last_cluster = player_cluster;
		for (int i = 1; i < walked_area_count - 1 && count < max_clusters; ++i)
		{
			int c = oracle->areas[(size_t)walked_areas[i]].cluster_id;
			if (c != last_cluster && c != player_cluster && c != object_cluster)
			{
				out_cluster_ids[count++] = c;
			}
			last_cluster = c;
		}

		*out_cluster_count = count;
	}

	return 1;
}

extern "C" int QNN_RoutePathPositionNth(
	const qnn_route_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	const float *player_pos,
	const float *object_pos,
	int nth,
	float *out_rel,
	float *out_route_cost,
	char *error,
	size_t error_size)
{
	int area_count;
	int axis;

	if (out_rel == nullptr || out_route_cost == nullptr)
		return 0;
	out_rel[0] = out_rel[1] = out_rel[2] = 0.0f;
	*out_route_cost = 0.0f;

	if (oracle == nullptr || player_pos == nullptr || object_pos == nullptr)
		return 0;

	area_count = (int)oracle->areas.size();
	if (player_area_id < 0 || player_area_id >= area_count
		|| object_area_id < 0 || object_area_id >= area_count)
		return 0;

	if (player_area_id == object_area_id)
	{
		if (nth > 0) return 0; /* only one trivial path */
		for (axis = 0; axis < 3; ++axis)
			out_rel[axis] = object_pos[axis] - player_pos[axis];
		return 1;
	}

	const size_t idx = (size_t)player_area_id * (size_t)area_count + (size_t)object_area_id;
	const std::vector<QnnRouteEntry> &entries = oracle->route_entries[idx];
	if (entries.empty() || nth < 0)
		return 0;

	/* Find the Nth cheapest entry by partial sort. */
	std::vector<QnnRouteEntry> sorted(entries);
	std::sort(sorted.begin(), sorted.end(), [](const QnnRouteEntry &a, const QnnRouteEntry &b) {
		if (a.cost != b.cost) return a.cost < b.cost;
		return a.link_id < b.link_id;
	});

	if (nth >= (int)sorted.size())
		return 0;

	const QnnRouteEntry &entry = sorted[(size_t)nth];
	if (entry.link_id < 0 || entry.link_id >= (int)oracle->links.size())
		return 0;

	{
		const float *target = entry.exit_cluster >= 0
			? entry.exit_pos
			: object_pos;

		for (axis = 0; axis < 3; ++axis)
			out_rel[axis] = target[axis] - player_pos[axis];
		*out_route_cost = entry.cost;
	}
	return 1;
}

/* ── Route build orchestration ──────────────────────────────────── */

extern "C" qboolean QNN_RouteBuildFromWorldmodel(qnn_map_state_t *out, char *error, size_t error_size)
{
	char route_error[256];

	if (cl.worldmodel == NULL)
		return true;

	memset(route_error, 0, sizeof(route_error));

	/* 1. Build navmesh from worldmodel geometry (map.c extracts, navmesh.cpp builds) */
	out->navmesh = QNN_MapBuildNavmesh(
		&out->navmesh_config,
		&out->navmesh_summary,
		route_error,
		sizeof(route_error));
	if (out->navmesh != NULL)
		strncpy(out->navmesh_status, "ready", sizeof(out->navmesh_status) - 1);
	else
		strncpy(out->navmesh_status, "error", sizeof(out->navmesh_status) - 1);
	if (route_error[0] != 0)
		strncpy(out->navmesh_error, route_error, sizeof(out->navmesh_error) - 1);

	/* 2. Build route graph from navmesh + static entity list */
	memset(route_error, 0, sizeof(route_error));
	if (out->navmesh != NULL)
	{
		qnn_route_static_object_view_t *oracle_objects;
		qnn_route_property_view_t *oracle_properties;
		qnn_route_summary_t oracle_summary;
		int total_properties;
		int property_offset;
		int object_index;

		memset(&oracle_summary, 0, sizeof(oracle_summary));
		oracle_objects = NULL;
		oracle_properties = NULL;
		total_properties = 0;
		property_offset = 0;
		for (object_index = 0; object_index < out->static_object_count; ++object_index)
			total_properties += out->static_objects[object_index].property_count;

		if (out->static_object_count > 0)
		{
			oracle_objects = (qnn_route_static_object_view_t *)calloc((size_t)out->static_object_count, sizeof(*oracle_objects));
			if (oracle_objects == NULL)
			{
				snprintf(error, error_size, "Out of memory while preparing route oracle objects");
				return false;
			}
		}
		if (total_properties > 0)
		{
			oracle_properties = (qnn_route_property_view_t *)calloc((size_t)total_properties, sizeof(*oracle_properties));
			if (oracle_properties == NULL)
			{
				free(oracle_objects);
				snprintf(error, error_size, "Out of memory while preparing route oracle properties");
				return false;
			}
		}

		for (object_index = 0; object_index < out->static_object_count; ++object_index)
		{
			int property_index;

			oracle_objects[object_index].object_id = out->static_objects[object_index].object_id;
			oracle_objects[object_index].category = out->static_objects[object_index].category;
			oracle_objects[object_index].classname = out->static_objects[object_index].classname;
			oracle_objects[object_index].origin = out->static_objects[object_index].origin;
			oracle_objects[object_index].angles = out->static_objects[object_index].angles;
			oracle_objects[object_index].property_count = out->static_objects[object_index].property_count;
			oracle_objects[object_index].properties = oracle_properties + property_offset;
			for (property_index = 0; property_index < out->static_objects[object_index].property_count; ++property_index)
			{
				oracle_properties[property_offset + property_index].key = out->static_objects[object_index].properties[property_index].key;
				oracle_properties[property_offset + property_index].value = out->static_objects[object_index].properties[property_index].value;
			}
			property_offset += out->static_objects[object_index].property_count;
		}

		out->route = QNN_RouteBuild(
			out->navmesh,
			oracle_objects,
			out->static_object_count,
			&oracle_summary,
			route_error,
			sizeof(route_error));
		free(oracle_objects);
		free(oracle_properties);
		if (out->route != NULL)
		{
			strncpy(out->route_status, "ready", sizeof(out->route_status) - 1);
			out->route_area_count = oracle_summary.area_count;
			out->route_cluster_count = oracle_summary.cluster_count;
			out->route_min_cluster_area_count = oracle_summary.min_cluster_area_count;
			out->route_max_cluster_area_count = oracle_summary.max_cluster_area_count;
			out->route_avg_cluster_area_count = oracle_summary.avg_cluster_area_count;
			out->route_walk_link_count = oracle_summary.walk_link_count;
			out->route_teleport_link_count = oracle_summary.teleport_link_count;
			out->route_lift_link_count = oracle_summary.lift_link_count;
			out->route_push_link_count = oracle_summary.push_link_count;
			out->route_drop_link_count = oracle_summary.drop_link_count;
		}
		else
		{
			strncpy(out->route_status, "error", sizeof(out->route_status) - 1);
		}
	}
	if (route_error[0] != 0)
		strncpy(out->route_error_msg, route_error, sizeof(out->route_error_msg) - 1);
	return true;
}
