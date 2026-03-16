#include "qnn_nav_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <queue>
#include <utility>
#include <vector>

#include <strings.h>

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
	qnn_nav_travel_type_t travel_type;
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

struct QnnRouteNode
{
	float cost;
	int area_id;

	bool operator>(const QnnRouteNode &other) const
	{
		return cost > other.cost;
	}
};

struct QnnRouteEntry
{
	int link_id;
	float cost;
};

struct QnnClusterQueueNode
{
	float cost;
	int seed_index;
	int area_id;

	bool operator>(const QnnClusterQueueNode &other) const
	{
		if (cost != other.cost)
			return cost > other.cost;
		if (seed_index != other.seed_index)
			return seed_index > other.seed_index;
		return area_id > other.area_id;
	}
};

struct qnn_nav_oracle_runtime_s
{
	const qnn_navmesh_runtime_t *navmesh;
	std::vector<QnnNavArea> areas;
	std::vector<QnnNavCluster> clusters;
	std::vector<QnnNavLink> links;
	std::vector<std::vector<int>> outgoing_links;
	std::vector<unsigned long long> sorted_poly_refs;
	std::vector<int> sorted_area_ids;
	qnn_nav_oracle_summary_t summary;
	/* Precomputed routing cache — flattened area_count * area_count tables.
	   Index: src * area_count + dst.
	   route_entries[i] holds one entry per distinct outgoing first-hop link
	   from src that can reach dst, including the total route cost.
	   TODO: phase 5 needs cluster-to-cluster cost lookups alongside this
	   area-level cache. */
	std::vector<std::vector<QnnRouteEntry>> route_entries;
};

namespace {

static float qnn_nav_distance(const float *lhs, const float *rhs)
{
	const float dx = lhs[0] - rhs[0];
	const float dy = lhs[1] - rhs[1];
	const float dz = lhs[2] - rhs[2];

	return sqrtf(dx * dx + dy * dy + dz * dz);
}

static float qnn_nav_dot3(const float *lhs, const float *rhs)
{
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

static float qnn_nav_bounds_axis_gap(float src_min, float src_max, float dst_min, float dst_max)
{
	if (src_max < dst_min)
		return dst_min - src_max;
	if (dst_max < src_min)
		return src_min - dst_max;
	return 0.0f;
}

static float qnn_nav_bounds_gap_xy(const QnnNavArea &src, const QnnNavArea &dst)
{
	const float dx = qnn_nav_bounds_axis_gap(src.bounds_min[0], src.bounds_max[0], dst.bounds_min[0], dst.bounds_max[0]);
	const float dy = qnn_nav_bounds_axis_gap(src.bounds_min[1], src.bounds_max[1], dst.bounds_min[1], dst.bounds_max[1]);

	return sqrtf(dx * dx + dy * dy);
}

static const char *qnn_nav_property_value(const qnn_nav_oracle_static_object_view_t *object, const char *key)
{
	int property_index;

	if (object == nullptr || key == nullptr)
		return nullptr;
	for (property_index = 0; property_index < object->property_count; ++property_index)
	{
		if (!strcmp(object->properties[property_index].key, key))
			return object->properties[property_index].value;
	}
	return nullptr;
}

static int qnn_nav_parse_vec3_value(const char *value, float *out)
{
	float x;
	float y;
	float z;

	if (value == nullptr || out == nullptr)
		return 0;
	if (sscanf(value, "%f %f %f", &x, &y, &z) != 3)
		return 0;
	out[0] = x;
	out[1] = y;
	out[2] = z;
	return 1;
}

static float qnn_nav_parse_float_value(const char *value, float fallback)
{
	char *end_ptr;
	float parsed;

	if (value == nullptr || value[0] == 0)
		return fallback;
	parsed = strtof(value, &end_ptr);
	if (end_ptr == value)
		return fallback;
	return parsed;
}

static int qnn_nav_has_link(const qnn_nav_oracle_runtime_t *oracle, int src_area_id, int dst_area_id, qnn_nav_travel_type_t travel_type)
{
	size_t link_index;

	if (oracle == nullptr || src_area_id < 0 || src_area_id >= (int)oracle->outgoing_links.size())
		return 0;
	for (link_index = 0; link_index < oracle->outgoing_links[(size_t)src_area_id].size(); ++link_index)
	{
		const QnnNavLink &link = oracle->links[(size_t)oracle->outgoing_links[(size_t)src_area_id][link_index]];
		if (link.dst_area_id == dst_area_id && link.travel_type == travel_type)
			return 1;
	}
	return 0;
}

static int qnn_nav_area_for_ref(const qnn_nav_oracle_runtime_t *oracle, unsigned long long poly_ref)
{
	int low;
	int high;

	low = 0;
	high = (int)oracle->sorted_poly_refs.size() - 1;
	while (low <= high)
	{
		const int mid = low + (high - low) / 2;
		if (oracle->sorted_poly_refs[(size_t)mid] == poly_ref)
			return oracle->sorted_area_ids[(size_t)mid];
		if (oracle->sorted_poly_refs[(size_t)mid] < poly_ref)
			low = mid + 1;
		else
			high = mid - 1;
	}
	return -1;
}

static int qnn_nav_find_nearest_area_id(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *point,
	qnn_navmesh_nearest_result_t *nearest,
	char *error,
	size_t error_size)
{
	int area_id;

	if (!qnn_navmesh_find_nearest(oracle->navmesh, point, nearest, error, error_size))
		return -1;
	area_id = qnn_nav_area_for_ref(oracle, nearest->poly_ref);
	if (area_id < 0)
		qnn_nav_set_error(error, error_size, "Nearest navmesh polygon was not found in oracle area table");
	return area_id;
}

static void qnn_nav_add_link(
	qnn_nav_oracle_runtime_t *oracle,
	int src_area_id,
	int dst_area_id,
	qnn_nav_travel_type_t travel_type,
	const float *start_pos,
	const float *end_pos,
	float travel_time)
{
	QnnNavLink link;

	if (oracle == nullptr
		|| src_area_id < 0
		|| dst_area_id < 0
		|| src_area_id >= (int)oracle->areas.size()
		|| dst_area_id >= (int)oracle->areas.size())
		return;
	if (qnn_nav_has_link(oracle, src_area_id, dst_area_id, travel_type))
		return;

	link.link_id = (int)oracle->links.size();
	link.src_area_id = src_area_id;
	link.dst_area_id = dst_area_id;
	link.travel_type = travel_type;
	memcpy(link.start_pos, start_pos, sizeof(link.start_pos));
	memcpy(link.end_pos, end_pos, sizeof(link.end_pos));
	link.travel_time = travel_time > 0.0f ? travel_time : 0.01f;

	oracle->links.push_back(link);
	oracle->outgoing_links[(size_t)src_area_id].push_back(link.link_id);
	oracle->areas[(size_t)src_area_id].link_count += 1;
	if (travel_type != QNN_NAV_TRAVEL_WALK)
		oracle->areas[(size_t)src_area_id].special_link_count += 1;
}

static void qnn_nav_build_walk_links(qnn_nav_oracle_runtime_t *oracle, const qnn_navmesh_poly_record_t *records, int record_count)
{
	int area_index;

	for (area_index = 0; area_index < record_count; ++area_index)
	{
		int neighbor_index;
		const QnnNavArea &src = oracle->areas[(size_t)area_index];

		for (neighbor_index = 0; neighbor_index < records[area_index].neighbor_count; ++neighbor_index)
		{
			const int dst_area_id = qnn_nav_area_for_ref(oracle, records[area_index].neighbor_refs[neighbor_index]);

			if (dst_area_id < 0)
				continue;
			const float travel_time = qnn_nav_distance(src.center, oracle->areas[(size_t)dst_area_id].center) / kWalkUnitsPerSecond;
			qnn_nav_add_link(
				oracle,
				src.area_id,
				dst_area_id,
				QNN_NAV_TRAVEL_WALK,
				src.center,
				oracle->areas[(size_t)dst_area_id].center,
				travel_time);
		}
	}
}

static void qnn_nav_parse_movedir(
	const qnn_nav_oracle_static_object_view_t *object,
	float *dir)
{
	const char *angle_value;
	const char *angles_value;
	float yaw;
	float pitch;
	float roll;

	dir[0] = 1.0f;
	dir[1] = 0.0f;
	dir[2] = 0.0f;
	if (object == nullptr)
		return;

	angle_value = qnn_nav_property_value(object, "angle");
	if (angle_value != nullptr && angle_value[0] != 0)
	{
		yaw = qnn_nav_parse_float_value(angle_value, 0.0f);
		if (fabsf(yaw - (-1.0f)) < 0.01f)
		{
			dir[0] = 0.0f;
			dir[1] = 0.0f;
			dir[2] = 1.0f;
			return;
		}
		if (fabsf(yaw - (-2.0f)) < 0.01f)
		{
			dir[0] = 0.0f;
			dir[1] = 0.0f;
			dir[2] = -1.0f;
			return;
		}
		yaw = yaw * (float)(M_PI / 180.0);
		dir[0] = cosf(yaw);
		dir[1] = sinf(yaw);
		dir[2] = 0.0f;
		return;
	}

	angles_value = qnn_nav_property_value(object, "angles");
	if (angles_value != nullptr && sscanf(angles_value, "%f %f %f", &pitch, &yaw, &roll) == 3)
	{
		const float pitch_rad = pitch * (float)(M_PI / 180.0);
		const float yaw_rad = yaw * (float)(M_PI / 180.0);
		(void)roll;
		dir[0] = cosf(pitch_rad) * cosf(yaw_rad);
		dir[1] = cosf(pitch_rad) * sinf(yaw_rad);
		dir[2] = -sinf(pitch_rad);
		return;
	}

	if (object->angles != nullptr)
	{
		if (fabsf(object->angles[0]) < 0.01f && fabsf(object->angles[2]) < 0.01f)
		{
			if (fabsf(object->angles[1] - (-1.0f)) < 0.01f)
			{
				dir[0] = 0.0f;
				dir[1] = 0.0f;
				dir[2] = 1.0f;
				return;
			}
			if (fabsf(object->angles[1] - (-2.0f)) < 0.01f)
			{
				dir[0] = 0.0f;
				dir[1] = 0.0f;
				dir[2] = -1.0f;
				return;
			}
		}
		yaw = object->angles[1] * (float)(M_PI / 180.0);
		dir[0] = cosf(yaw);
		dir[1] = sinf(yaw);
		dir[2] = 0.0f;
	}
}

static void qnn_nav_build_teleport_links(qnn_nav_oracle_runtime_t *oracle, const qnn_nav_oracle_static_object_view_t *static_objects, int static_object_count)
{
	int object_index;

	for (object_index = 0; object_index < static_object_count; ++object_index)
	{
		const qnn_nav_oracle_static_object_view_t *source;
		const char *target_name;
		int dest_index;
		qnn_navmesh_nearest_result_t src_nearest;
		qnn_navmesh_nearest_result_t dst_nearest;
		char error[256];
		int src_area_id;
		int dst_area_id;

		source = &static_objects[object_index];
		if (strcasecmp(source->classname, "trigger_teleport") != 0)
			continue;
		target_name = qnn_nav_property_value(source, "target");
		if (target_name == nullptr || target_name[0] == 0)
			continue;

		dest_index = -1;
		for (dest_index = 0; dest_index < static_object_count; ++dest_index)
		{
			const qnn_nav_oracle_static_object_view_t *candidate;
			const char *candidate_name;

			candidate = &static_objects[dest_index];
			if (strcasecmp(candidate->classname, "info_teleport_destination") != 0)
				continue;
			candidate_name = qnn_nav_property_value(candidate, "targetname");
			if (candidate_name != nullptr && !strcmp(candidate_name, target_name))
				break;
		}
		if (dest_index >= static_object_count)
			continue;

		memset(&src_nearest, 0, sizeof(src_nearest));
		memset(&dst_nearest, 0, sizeof(dst_nearest));
		memset(error, 0, sizeof(error));
		src_area_id = qnn_nav_find_nearest_area_id(oracle, source->origin, &src_nearest, error, sizeof(error));
		if (src_area_id < 0)
			continue;
		dst_area_id = qnn_nav_find_nearest_area_id(oracle, static_objects[dest_index].origin, &dst_nearest, error, sizeof(error));
		if (dst_area_id < 0)
			continue;
		qnn_nav_add_link(
			oracle,
			src_area_id,
			dst_area_id,
			QNN_NAV_TRAVEL_TELEPORT,
			source->origin,
			static_objects[dest_index].origin,
			kTeleportTravelTime);
	}
}

static int qnn_nav_parse_object_bounds(const qnn_nav_oracle_static_object_view_t *object, float *mins, float *maxs)
{
	const char *mins_value;
	const char *maxs_value;

	mins_value = qnn_nav_property_value(object, "qnn_model_bounds_min");
	maxs_value = qnn_nav_property_value(object, "qnn_model_bounds_max");
	if (!qnn_nav_parse_vec3_value(mins_value, mins) || !qnn_nav_parse_vec3_value(maxs_value, maxs))
		return 0;
	return 1;
}

static void qnn_nav_build_lift_links(qnn_nav_oracle_runtime_t *oracle, const qnn_nav_oracle_static_object_view_t *static_objects, int static_object_count)
{
	int object_index;

	for (object_index = 0; object_index < static_object_count; ++object_index)
	{
		const qnn_nav_oracle_static_object_view_t *object;
		float bounds_min[3];
		float bounds_max[3];
		float top_pos[3];
		float bottom_pos[3];
		float lift_height;
		float lift_speed;
		float travel_time;
		qnn_navmesh_nearest_result_t top_nearest;
		qnn_navmesh_nearest_result_t bottom_nearest;
		char error[256];
		int top_area_id;
		int bottom_area_id;

		object = &static_objects[object_index];
		if (strcasecmp(object->classname, "func_plat") != 0)
			continue;
		if (!qnn_nav_parse_object_bounds(object, bounds_min, bounds_max))
			continue;

		top_pos[0] = object->origin[0];
		top_pos[1] = object->origin[1];
		top_pos[2] = object->origin[2];
		bottom_pos[0] = object->origin[0];
		bottom_pos[1] = object->origin[1];
		bottom_pos[2] = object->origin[2];

		lift_height = qnn_nav_parse_float_value(qnn_nav_property_value(object, "height"), (bounds_max[2] - bounds_min[2]) - 8.0f);
		if (lift_height <= 0.0f)
			continue;
		bottom_pos[2] -= lift_height;

		lift_speed = qnn_nav_parse_float_value(qnn_nav_property_value(object, "speed"), 150.0f);
		if (lift_speed <= 0.0f)
			lift_speed = 150.0f;
		travel_time = (lift_height / lift_speed) + kLiftBasePenalty;

		memset(&top_nearest, 0, sizeof(top_nearest));
		memset(&bottom_nearest, 0, sizeof(bottom_nearest));
		memset(error, 0, sizeof(error));
		top_area_id = qnn_nav_find_nearest_area_id(oracle, top_pos, &top_nearest, error, sizeof(error));
		if (top_area_id < 0)
			continue;
		bottom_area_id = qnn_nav_find_nearest_area_id(oracle, bottom_pos, &bottom_nearest, error, sizeof(error));
		if (bottom_area_id < 0 || bottom_area_id == top_area_id)
			continue;

		qnn_nav_add_link(
			oracle,
			top_area_id,
			bottom_area_id,
			QNN_NAV_TRAVEL_ELEVATOR,
			top_pos,
			bottom_pos,
			travel_time);
		qnn_nav_add_link(
			oracle,
			bottom_area_id,
			top_area_id,
			QNN_NAV_TRAVEL_ELEVATOR,
			bottom_pos,
			top_pos,
			travel_time);
	}
}

static void qnn_nav_build_push_links(qnn_nav_oracle_runtime_t *oracle, const qnn_nav_oracle_static_object_view_t *static_objects, int static_object_count)
{
	int object_index;

	for (object_index = 0; object_index < static_object_count; ++object_index)
	{
		const qnn_nav_oracle_static_object_view_t *object;
		float movedir[3];
		float launch_speed;
		qnn_navmesh_nearest_result_t src_nearest;
		char error[256];
		int src_area_id;
		float best_progress;
		float best_time;
		int best_area_id;
		float best_start[3];
		float best_end[3];
		float t;

		object = &static_objects[object_index];
		if (strcasecmp(object->classname, "trigger_push") != 0)
			continue;

		memset(&src_nearest, 0, sizeof(src_nearest));
		memset(error, 0, sizeof(error));
		src_area_id = qnn_nav_find_nearest_area_id(oracle, object->origin, &src_nearest, error, sizeof(error));
		if (src_area_id < 0)
			continue;

		qnn_nav_parse_movedir(object, movedir);
		launch_speed = qnn_nav_parse_float_value(qnn_nav_property_value(object, "speed"), 100.0f) * 10.0f;
		best_progress = -1.0f;
		best_time = 0.0f;
		best_area_id = -1;
		memset(best_start, 0, sizeof(best_start));
		memset(best_end, 0, sizeof(best_end));

		for (t = kPushSampleStep; t <= kPushMaxTime; t += kPushSampleStep)
		{
			float sample_pos[3];
			qnn_navmesh_nearest_result_t nearest;
			int sample_area_id;
			float snap_distance;
			float displacement[3];
			float progress;

			sample_pos[0] = object->origin[0] + movedir[0] * launch_speed * t;
			sample_pos[1] = object->origin[1] + movedir[1] * launch_speed * t;
			sample_pos[2] = object->origin[2] + movedir[2] * launch_speed * t - 0.5f * kPushGravity * t * t;

			memset(&nearest, 0, sizeof(nearest));
			sample_area_id = qnn_nav_find_nearest_area_id(oracle, sample_pos, &nearest, error, sizeof(error));
			if (sample_area_id < 0 || sample_area_id == src_area_id)
				continue;

			snap_distance = qnn_nav_distance(sample_pos, nearest.nearest_point);
			if (snap_distance > kPushMaxSnapDistance)
				continue;

			displacement[0] = nearest.nearest_point[0] - object->origin[0];
			displacement[1] = nearest.nearest_point[1] - object->origin[1];
			displacement[2] = nearest.nearest_point[2] - object->origin[2];
			progress = qnn_nav_dot3(displacement, movedir);
			if (progress <= best_progress)
				continue;

			best_progress = progress;
			best_time = t;
			best_area_id = sample_area_id;
			memcpy(best_start, object->origin, sizeof(best_start));
			memcpy(best_end, nearest.nearest_point, sizeof(best_end));
		}

		if (best_area_id >= 0)
		{
			qnn_nav_add_link(
				oracle,
				src_area_id,
				best_area_id,
				QNN_NAV_TRAVEL_PUSH,
				best_start,
				best_end,
				best_time);
		}
	}
}

static void qnn_nav_build_drop_links(qnn_nav_oracle_runtime_t *oracle)
{
	const int area_count = (int)oracle->areas.size();

	/* Build an index of area ids sorted by bounds_max[2] ascending so we can
	   binary-search for the Z window instead of scanning all areas. */
	std::vector<std::pair<float, int>> z_sorted;
	z_sorted.reserve((size_t)area_count);
	for (int i = 0; i < area_count; ++i)
		z_sorted.push_back({oracle->areas[(size_t)i].bounds_max[2], i});
	std::sort(z_sorted.begin(), z_sorted.end());

	for (int src_area_id = 0; src_area_id < area_count; ++src_area_id)
	{
		const QnnNavArea &src = oracle->areas[(size_t)src_area_id];
		int best_area_ids[kDropMaxLinksPerArea];
		float best_scores[kDropMaxLinksPerArea];
		int slot_index;

		for (slot_index = 0; slot_index < kDropMaxLinksPerArea; ++slot_index)
		{
			best_area_ids[slot_index] = -1;
			best_scores[slot_index] = std::numeric_limits<float>::infinity();
		}

		/* dst.bounds_max[2] must satisfy:
		     src.bounds_min[2] - kDropMaxHeight  <= dst.bounds_max[2]
		     dst.bounds_max[2] <= src.bounds_min[2] - kDropMinHeight */
		const float z_lo = src.bounds_min[2] - kDropMaxHeight;
		const float z_hi = src.bounds_min[2] - kDropMinHeight;
		if (z_lo > z_hi)
			continue;

		/* lower_bound for z_lo */
		auto it_lo = std::lower_bound(z_sorted.begin(), z_sorted.end(),
			std::make_pair(z_lo, std::numeric_limits<int>::min()));
		/* upper_bound for z_hi */
		auto it_hi = std::upper_bound(z_sorted.begin(), z_sorted.end(),
			std::make_pair(z_hi, std::numeric_limits<int>::max()));

		for (auto it = it_lo; it != it_hi; ++it)
		{
			const int dst_area_id = it->second;
			const QnnNavArea &dst = oracle->areas[(size_t)dst_area_id];
			float horizontal_gap;
			float score;

			if (dst_area_id == src_area_id)
				continue;
			if (qnn_nav_has_link(oracle, src_area_id, dst_area_id, QNN_NAV_TRAVEL_WALK))
				continue;
			if (qnn_nav_has_link(oracle, src_area_id, dst_area_id, QNN_NAV_TRAVEL_DROP))
				continue;
			if (qnn_nav_has_link(oracle, dst_area_id, src_area_id, QNN_NAV_TRAVEL_WALK))
				continue;

			horizontal_gap = qnn_nav_bounds_gap_xy(src, dst);
			if (horizontal_gap > kDropMaxHorizontalGap)
				continue;
			score = horizontal_gap * 16.0f + (src.bounds_min[2] - dst.bounds_max[2]);
			for (slot_index = 0; slot_index < kDropMaxLinksPerArea; ++slot_index)
			{
				if (score < best_scores[slot_index])
				{
					int shift_index;

					for (shift_index = kDropMaxLinksPerArea - 1; shift_index > slot_index; --shift_index)
					{
						best_scores[shift_index] = best_scores[shift_index - 1];
						best_area_ids[shift_index] = best_area_ids[shift_index - 1];
					}
					best_scores[slot_index] = score;
					best_area_ids[slot_index] = dst_area_id;
					break;
				}
			}
		}

		for (slot_index = 0; slot_index < kDropMaxLinksPerArea; ++slot_index)
		{
			const int best_dst_area_id = best_area_ids[slot_index];
			float travel_time;

			if (best_dst_area_id < 0)
				continue;
			travel_time = qnn_nav_distance(src.center, oracle->areas[(size_t)best_dst_area_id].center) / kWalkUnitsPerSecond;
			qnn_nav_add_link(
				oracle,
				src_area_id,
				best_dst_area_id,
				QNN_NAV_TRAVEL_DROP,
				src.center,
				oracle->areas[(size_t)best_dst_area_id].center,
				travel_time);
		}
	}
}

static int qnn_nav_is_special_travel(qnn_nav_travel_type_t travel_type)
{
	return travel_type != QNN_NAV_TRAVEL_INVALID
		&& travel_type != QNN_NAV_TRAVEL_WALK;
}

static void qnn_nav_sort_unique_ints(std::vector<int> *values)
{
	if (values == nullptr)
		return;
	std::sort(values->begin(), values->end());
	values->erase(std::unique(values->begin(), values->end()), values->end());
}

static int qnn_nav_cluster_first_area_id(const std::vector<int> &area_ids)
{
	if (area_ids.empty())
		return -1;
	return area_ids.front();
}

static int qnn_nav_cluster_max_size(const std::vector<std::vector<int>> &clusters)
{
	int max_size;

	max_size = 0;
	for (const std::vector<int> &cluster : clusters)
	{
		if ((int)cluster.size() > max_size)
			max_size = (int)cluster.size();
	}
	return max_size;
}

static int qnn_nav_choose_seed_area(
	const qnn_nav_oracle_runtime_t *oracle,
	const std::vector<int> &candidate_area_ids,
	const std::vector<int> &selected_area_ids,
	const std::vector<char> &selected_mask,
	const std::vector<int> &special_incidence)
{
	int best_area_id;
	float best_distance;
	int best_special_incidence;

	best_area_id = -1;
	best_distance = -1.0f;
	best_special_incidence = -1;

	/* O(candidate_area_ids * selected_area_ids), which is acceptable for the
	   small area counts in Quake maps. */
	for (int candidate_area_id : candidate_area_ids)
	{
		float min_distance;

		if (candidate_area_id < 0
			|| candidate_area_id >= (int)selected_mask.size()
			|| selected_mask[(size_t)candidate_area_id])
			continue;

		min_distance = 0.0f;
		if (!selected_area_ids.empty())
		{
			min_distance = std::numeric_limits<float>::infinity();
			for (int selected_area_id : selected_area_ids)
			{
				const float distance = qnn_nav_distance(
					oracle->areas[(size_t)candidate_area_id].center,
					oracle->areas[(size_t)selected_area_id].center);

				if (distance < min_distance)
					min_distance = distance;
			}
		}

		if (best_area_id < 0
			|| min_distance > best_distance + kClusterCostEpsilon
			|| (fabsf(min_distance - best_distance) <= kClusterCostEpsilon
				&& special_incidence[(size_t)candidate_area_id] > best_special_incidence)
			|| (fabsf(min_distance - best_distance) <= kClusterCostEpsilon
				&& special_incidence[(size_t)candidate_area_id] == best_special_incidence
				&& candidate_area_id < best_area_id))
		{
			best_area_id = candidate_area_id;
			best_distance = min_distance;
			best_special_incidence = special_incidence[(size_t)candidate_area_id];
		}
	}

	return best_area_id;
}

static void qnn_nav_pick_cluster_seeds(
	const qnn_nav_oracle_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<int> &special_incidence,
	int seed_count,
	std::vector<int> *seed_area_ids)
{
	std::vector<int> special_area_ids;
	std::vector<char> selected_mask;

	if (seed_area_ids == nullptr)
		return;

	seed_area_ids->clear();
	if (oracle == nullptr || component_area_ids.empty() || seed_count <= 0)
		return;

	seed_count = std::min(seed_count, (int)component_area_ids.size());
	selected_mask.assign(oracle->areas.size(), 0);

	for (int area_id : component_area_ids)
	{
		if (special_incidence[(size_t)area_id] > 0)
			special_area_ids.push_back(area_id);
	}

	while ((int)seed_area_ids->size() < seed_count && !special_area_ids.empty())
	{
		const int seed_area_id = qnn_nav_choose_seed_area(
			oracle,
			special_area_ids,
			*seed_area_ids,
			selected_mask,
			special_incidence);

		if (seed_area_id < 0)
			break;
		seed_area_ids->push_back(seed_area_id);
		selected_mask[(size_t)seed_area_id] = 1;
	}

	if (seed_area_ids->empty())
	{
		const int seed_area_id = qnn_nav_choose_seed_area(
			oracle,
			component_area_ids,
			*seed_area_ids,
			selected_mask,
			special_incidence);

		if (seed_area_id >= 0)
		{
			seed_area_ids->push_back(seed_area_id);
			selected_mask[(size_t)seed_area_id] = 1;
		}
	}

	while ((int)seed_area_ids->size() < seed_count)
	{
		const int seed_area_id = qnn_nav_choose_seed_area(
			oracle,
			component_area_ids,
			*seed_area_ids,
			selected_mask,
			special_incidence);

		if (seed_area_id < 0)
			break;
		seed_area_ids->push_back(seed_area_id);
		selected_mask[(size_t)seed_area_id] = 1;
	}
}

static void qnn_nav_assign_component_clusters(
	const qnn_nav_oracle_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	const std::vector<int> &seed_area_ids,
	std::vector<int> *cluster_assignment)
{
	std::vector<char> in_component;
	std::vector<float> best_cost;
	std::priority_queue<QnnClusterQueueNode, std::vector<QnnClusterQueueNode>, std::greater<QnnClusterQueueNode>> queue;

	if (oracle == nullptr || cluster_assignment == nullptr)
		return;

	cluster_assignment->assign(oracle->areas.size(), -1);
	in_component.assign(oracle->areas.size(), 0);
	best_cost.assign(oracle->areas.size(), std::numeric_limits<float>::infinity());

	for (int area_id : component_area_ids)
		in_component[(size_t)area_id] = 1;

	for (size_t seed_index = 0; seed_index < seed_area_ids.size(); ++seed_index)
	{
		const int seed_area_id = seed_area_ids[seed_index];

		best_cost[(size_t)seed_area_id] = 0.0f;
		(*cluster_assignment)[(size_t)seed_area_id] = (int)seed_index;
		queue.push({0.0f, (int)seed_index, seed_area_id});
	}

	while (!queue.empty())
	{
		const QnnClusterQueueNode current = queue.top();
		queue.pop();

		if (current.cost > best_cost[(size_t)current.area_id] + kClusterCostEpsilon)
			continue;
		if ((*cluster_assignment)[(size_t)current.area_id] != current.seed_index
			&& fabsf(current.cost - best_cost[(size_t)current.area_id]) <= kClusterCostEpsilon)
			continue;

		for (int neighbor_area_id : walk_neighbors[(size_t)current.area_id])
		{
			float next_cost;

			if (!in_component[(size_t)neighbor_area_id])
				continue;

			next_cost = current.cost + qnn_nav_distance(
				oracle->areas[(size_t)current.area_id].center,
				oracle->areas[(size_t)neighbor_area_id].center);
			if (next_cost + kClusterCostEpsilon < best_cost[(size_t)neighbor_area_id]
				|| (fabsf(next_cost - best_cost[(size_t)neighbor_area_id]) <= kClusterCostEpsilon
					&& current.seed_index < (*cluster_assignment)[(size_t)neighbor_area_id]))
			{
				best_cost[(size_t)neighbor_area_id] = next_cost;
				(*cluster_assignment)[(size_t)neighbor_area_id] = current.seed_index;
				queue.push({next_cost, current.seed_index, neighbor_area_id});
			}
		}
	}
}

static void qnn_nav_collect_component_clusters(
	const std::vector<int> &component_area_ids,
	const std::vector<int> &cluster_assignment,
	int cluster_count,
	std::vector<std::vector<int>> *clusters)
{
	if (clusters == nullptr)
		return;

	clusters->assign((size_t)cluster_count, std::vector<int>());
	for (int area_id : component_area_ids)
	{
		const int cluster_index = cluster_assignment[(size_t)area_id];

		if (cluster_index >= 0 && cluster_index < cluster_count)
			(*clusters)[(size_t)cluster_index].push_back(area_id);
	}
	for (std::vector<int> &cluster : *clusters)
		qnn_nav_sort_unique_ints(&cluster);
}

static int qnn_nav_choose_merge_target(
	const std::vector<std::vector<int>> &clusters,
	const std::vector<int> &cluster_assignment,
	const std::vector<std::vector<int>> &walk_neighbors,
	int source_cluster_index)
{
	std::vector<int> shared_edges;
	int best_cluster_index;
	int best_shared_edges;
	int best_overflow;
	int best_target_delta;
	int best_first_area_id;

	shared_edges.assign(clusters.size(), 0);
	for (int area_id : clusters[(size_t)source_cluster_index])
	{
		for (int neighbor_area_id : walk_neighbors[(size_t)area_id])
		{
			const int neighbor_cluster_index = cluster_assignment[(size_t)neighbor_area_id];

			if (neighbor_cluster_index >= 0 && neighbor_cluster_index != source_cluster_index)
				shared_edges[(size_t)neighbor_cluster_index] += 1;
		}
	}

	best_cluster_index = -1;
	best_shared_edges = -1;
	best_overflow = std::numeric_limits<int>::max();
	best_target_delta = std::numeric_limits<int>::max();
	best_first_area_id = std::numeric_limits<int>::max();

	for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index)
	{
		const int merged_size = (int)clusters[(size_t)source_cluster_index].size() + (int)clusters[cluster_index].size();
		const int overflow = merged_size > kClusterMaxAreaCount ? merged_size - kClusterMaxAreaCount : 0;
		const int target_delta = abs(merged_size - kClusterTargetAreaCount);
		const int first_area_id = qnn_nav_cluster_first_area_id(clusters[cluster_index]);

		if ((int)cluster_index == source_cluster_index
			|| clusters[cluster_index].empty()
			|| shared_edges[cluster_index] <= 0)
			continue;

		if (best_cluster_index < 0
			|| shared_edges[cluster_index] > best_shared_edges
			|| (shared_edges[cluster_index] == best_shared_edges && overflow < best_overflow)
			|| (shared_edges[cluster_index] == best_shared_edges && overflow == best_overflow && target_delta < best_target_delta)
			|| (shared_edges[cluster_index] == best_shared_edges && overflow == best_overflow && target_delta == best_target_delta
				&& first_area_id < best_first_area_id))
		{
			best_cluster_index = (int)cluster_index;
			best_shared_edges = shared_edges[cluster_index];
			best_overflow = overflow;
			best_target_delta = target_delta;
			best_first_area_id = first_area_id;
		}
	}

	return best_cluster_index;
}

static void qnn_nav_merge_small_clusters(
	const std::vector<std::vector<int>> &walk_neighbors,
	std::vector<std::vector<int>> *clusters)
{
	std::vector<int> cluster_assignment;

	if (clusters == nullptr)
		return;

	cluster_assignment.assign(walk_neighbors.size(), -1);
	for (size_t cluster_index = 0; cluster_index < clusters->size(); ++cluster_index)
	{
		for (int area_id : (*clusters)[cluster_index])
			cluster_assignment[(size_t)area_id] = (int)cluster_index;
	}
	for (;;)
	{
		int source_cluster_index;
		int source_size;
		int source_first_area_id;
		int non_empty_cluster_count;

		source_cluster_index = -1;
		source_size = std::numeric_limits<int>::max();
		source_first_area_id = std::numeric_limits<int>::max();
		non_empty_cluster_count = 0;

		for (size_t cluster_index = 0; cluster_index < clusters->size(); ++cluster_index)
		{
			const int cluster_size = (int)(*clusters)[cluster_index].size();
			const int first_area_id = qnn_nav_cluster_first_area_id((*clusters)[cluster_index]);

			if (cluster_size <= 0)
				continue;
			non_empty_cluster_count += 1;
			if (cluster_size >= kClusterMinAreaCount)
				continue;
			if (source_cluster_index < 0
				|| cluster_size < source_size
				|| (cluster_size == source_size && first_area_id < source_first_area_id))
			{
				source_cluster_index = (int)cluster_index;
				source_size = cluster_size;
				source_first_area_id = first_area_id;
			}
		}

		if (source_cluster_index < 0 || non_empty_cluster_count <= 1)
			break;

		{
			const int target_cluster_index = qnn_nav_choose_merge_target(
				*clusters,
				cluster_assignment,
				walk_neighbors,
				source_cluster_index);

			if (target_cluster_index < 0)
				break;

			(*clusters)[(size_t)target_cluster_index].insert(
				(*clusters)[(size_t)target_cluster_index].end(),
				(*clusters)[(size_t)source_cluster_index].begin(),
				(*clusters)[(size_t)source_cluster_index].end());
			qnn_nav_sort_unique_ints(&(*clusters)[(size_t)target_cluster_index]);
			(*clusters)[(size_t)source_cluster_index].clear();
			for (int area_id : (*clusters)[(size_t)target_cluster_index])
				cluster_assignment[(size_t)area_id] = target_cluster_index;
		}
	}

	{
		std::vector<std::vector<int>> compacted_clusters;

		compacted_clusters.reserve(clusters->size());
		for (const std::vector<int> &cluster : *clusters)
		{
			if (!cluster.empty())
				compacted_clusters.push_back(cluster);
		}
		*clusters = std::move(compacted_clusters);
	}
}

static void qnn_nav_partition_walk_component(
	const qnn_nav_oracle_runtime_t *oracle,
	const std::vector<int> &component_area_ids,
	const std::vector<std::vector<int>> &walk_neighbors,
	const std::vector<int> &special_incidence,
	std::vector<std::vector<int>> *cluster_area_ids)
{
	std::vector<int> seed_area_ids;
	std::vector<int> cluster_assignment;
	std::vector<std::vector<int>> component_clusters;
	int desired_cluster_count;
	const int max_iterations = 100;

	if (cluster_area_ids == nullptr || component_area_ids.empty())
		return;

	if ((int)component_area_ids.size() <= kClusterMaxAreaCount)
	{
		cluster_area_ids->push_back(component_area_ids);
		return;
	}

	desired_cluster_count = std::max(2, ((int)component_area_ids.size() + kClusterTargetAreaCount - 1) / kClusterTargetAreaCount);
	desired_cluster_count = std::min(desired_cluster_count, (int)component_area_ids.size());

	for (int iteration_count = 0; iteration_count < max_iterations; ++iteration_count)
	{
		qnn_nav_pick_cluster_seeds(
			oracle,
			component_area_ids,
			special_incidence,
			desired_cluster_count,
			&seed_area_ids);
		qnn_nav_assign_component_clusters(
			oracle,
			component_area_ids,
			walk_neighbors,
			seed_area_ids,
			&cluster_assignment);
		qnn_nav_collect_component_clusters(
			component_area_ids,
			cluster_assignment,
			(int)seed_area_ids.size(),
			&component_clusters);

		if (qnn_nav_cluster_max_size(component_clusters) > kClusterMaxAreaCount
			&& desired_cluster_count < (int)component_area_ids.size())
		{
			desired_cluster_count += 1;
			continue;
		}

		qnn_nav_merge_small_clusters(walk_neighbors, &component_clusters);
		if (qnn_nav_cluster_max_size(component_clusters) > kClusterMaxAreaCount
			&& desired_cluster_count < (int)component_area_ids.size())
		{
			desired_cluster_count += 1;
			continue;
		}

		for (const std::vector<int> &cluster : component_clusters)
		{
			if (!cluster.empty())
				cluster_area_ids->push_back(cluster);
		}
		return;
	}

	for (const std::vector<int> &cluster : component_clusters)
	{
		if (!cluster.empty())
			cluster_area_ids->push_back(cluster);
	}
}

static void qnn_nav_build_clusters(qnn_nav_oracle_runtime_t *oracle)
{
	std::vector<std::vector<int>> walk_neighbors;
	std::vector<int> special_incidence;
	std::vector<char> visited;
	std::vector<std::vector<int>> cluster_area_ids;

	if (oracle == nullptr)
		return;

	oracle->clusters.clear();
	if (oracle->areas.empty())
		return;

	walk_neighbors.assign(oracle->areas.size(), std::vector<int>());
	special_incidence.assign(oracle->areas.size(), 0);
	visited.assign(oracle->areas.size(), 0);

	for (QnnNavArea &area : oracle->areas)
		area.cluster_id = -1;

	for (const QnnNavLink &link : oracle->links)
	{
		if (link.src_area_id < 0
			|| link.dst_area_id < 0
			|| link.src_area_id >= (int)oracle->areas.size()
			|| link.dst_area_id >= (int)oracle->areas.size())
			continue;

		if (link.travel_type == QNN_NAV_TRAVEL_WALK)
		{
			/* Detour walk adjacency is treated as bidirectional here, so
			   mirror each walk edge for connected-component clustering. */
			walk_neighbors[(size_t)link.src_area_id].push_back(link.dst_area_id);
			walk_neighbors[(size_t)link.dst_area_id].push_back(link.src_area_id);
		}
		else if (qnn_nav_is_special_travel(link.travel_type))
		{
			special_incidence[(size_t)link.src_area_id] += 1;
			special_incidence[(size_t)link.dst_area_id] += 1;
		}
	}

	for (std::vector<int> &neighbors : walk_neighbors)
		qnn_nav_sort_unique_ints(&neighbors);

	for (size_t area_index = 0; area_index < oracle->areas.size(); ++area_index)
	{
		std::queue<int> queue;
		std::vector<int> component_area_ids;

		if (visited[area_index])
			continue;

		visited[area_index] = 1;
		queue.push((int)area_index);
		while (!queue.empty())
		{
			const int area_id = queue.front();
			queue.pop();
			component_area_ids.push_back(area_id);

			for (int neighbor_area_id : walk_neighbors[(size_t)area_id])
			{
				if (!visited[(size_t)neighbor_area_id])
				{
					visited[(size_t)neighbor_area_id] = 1;
					queue.push(neighbor_area_id);
				}
			}
		}

		qnn_nav_sort_unique_ints(&component_area_ids);
		qnn_nav_partition_walk_component(
			oracle,
			component_area_ids,
			walk_neighbors,
			special_incidence,
			&cluster_area_ids);
	}

	std::sort(cluster_area_ids.begin(), cluster_area_ids.end(), [](const std::vector<int> &lhs, const std::vector<int> &rhs) {
		return qnn_nav_cluster_first_area_id(lhs) < qnn_nav_cluster_first_area_id(rhs);
	});

	oracle->clusters.reserve(cluster_area_ids.size());
	for (size_t cluster_index = 0; cluster_index < cluster_area_ids.size(); ++cluster_index)
	{
		QnnNavCluster cluster;
		float center_sum[3];
		const QnnNavArea &first_area = oracle->areas[(size_t)cluster_area_ids[cluster_index][0]];

		memset(&cluster, 0, sizeof(cluster));
		memset(center_sum, 0, sizeof(center_sum));
		cluster.cluster_id = (int)cluster_index;
		cluster.first_area_id = cluster_area_ids[cluster_index][0];
		cluster.area_count = (int)cluster_area_ids[cluster_index].size();
		memcpy(cluster.bounds_min, first_area.bounds_min, sizeof(cluster.bounds_min));
		memcpy(cluster.bounds_max, first_area.bounds_max, sizeof(cluster.bounds_max));

		for (int area_id : cluster_area_ids[cluster_index])
		{
			QnnNavArea &area = oracle->areas[(size_t)area_id];

			area.cluster_id = cluster.cluster_id;
			center_sum[0] += area.center[0];
			center_sum[1] += area.center[1];
			center_sum[2] += area.center[2];
			for (int axis = 0; axis < 3; ++axis)
			{
				cluster.bounds_min[axis] = std::min(cluster.bounds_min[axis], area.bounds_min[axis]);
				cluster.bounds_max[axis] = std::max(cluster.bounds_max[axis], area.bounds_max[axis]);
			}
		}

		cluster.center[0] = center_sum[0] / (float)cluster.area_count;
		cluster.center[1] = center_sum[1] / (float)cluster.area_count;
		cluster.center[2] = center_sum[2] / (float)cluster.area_count;
		oracle->clusters.push_back(cluster);
	}

	{
		std::vector<std::vector<int>> exit_clusters;
		std::vector<std::vector<int>> special_exit_clusters;

		exit_clusters.assign(oracle->clusters.size(), std::vector<int>());
		special_exit_clusters.assign(oracle->clusters.size(), std::vector<int>());
		for (const QnnNavLink &link : oracle->links)
		{
			const int src_cluster_id = oracle->areas[(size_t)link.src_area_id].cluster_id;
			const int dst_cluster_id = oracle->areas[(size_t)link.dst_area_id].cluster_id;

			if (src_cluster_id < 0 || dst_cluster_id < 0 || src_cluster_id == dst_cluster_id)
				continue;

			exit_clusters[(size_t)src_cluster_id].push_back(dst_cluster_id);
			if (qnn_nav_is_special_travel(link.travel_type))
				special_exit_clusters[(size_t)src_cluster_id].push_back(dst_cluster_id);
		}

		for (size_t cluster_index = 0; cluster_index < oracle->clusters.size(); ++cluster_index)
		{
			qnn_nav_sort_unique_ints(&exit_clusters[cluster_index]);
			qnn_nav_sort_unique_ints(&special_exit_clusters[cluster_index]);
			oracle->clusters[cluster_index].exit_count = (int)exit_clusters[cluster_index].size();
			oracle->clusters[cluster_index].special_exit_count = (int)special_exit_clusters[cluster_index].size();
		}
	}
}

/* Build precomputed all-pairs routing cache using reverse Dijkstra from each
   destination.  For each destination D we run Dijkstra backwards over
   incoming edges to recover shortest remaining cost from every area to D.
   We then expand every source area's outgoing links to keep all distinct
   first-hop choices that can reach D, each annotated with its total route
   cost. */
static void qnn_nav_build_routing_cache(qnn_nav_oracle_runtime_t *oracle)
{
	const int area_count = (int)oracle->areas.size();
	const size_t table_size = (size_t)area_count * (size_t)area_count;

	oracle->route_entries.assign(table_size, std::vector<QnnRouteEntry>());

	/* Build incoming (reverse) adjacency: for each link src→dst, record
	   the link as an incoming edge of dst. */
	std::vector<std::vector<int>> incoming_links((size_t)area_count);
	for (size_t li = 0; li < oracle->links.size(); ++li)
		incoming_links[(size_t)oracle->links[li].dst_area_id].push_back((int)li);

	/* For each destination, run reverse Dijkstra. */
	std::vector<float> best_cost((size_t)area_count);

	for (int dst = 0; dst < area_count; ++dst)
	{
		std::fill(best_cost.begin(), best_cost.end(), std::numeric_limits<float>::infinity());
		best_cost[(size_t)dst] = 0.0f;

		std::priority_queue<QnnRouteNode, std::vector<QnnRouteNode>, std::greater<QnnRouteNode>> pq;
		pq.push({0.0f, dst});

		while (!pq.empty())
		{
			const QnnRouteNode cur = pq.top();
			pq.pop();
			if (cur.cost > best_cost[(size_t)cur.area_id] + 0.0001f)
				continue;

			/* Relax all incoming edges: for each link that arrives at
			   cur.area_id, see if going through cur improves the source. */
			for (int li : incoming_links[(size_t)cur.area_id])
			{
				const QnnNavLink &link = oracle->links[(size_t)li];
				const float new_cost = cur.cost + link.travel_time;

				if (new_cost + 0.0001f < best_cost[(size_t)link.src_area_id])
				{
					best_cost[(size_t)link.src_area_id] = new_cost;
					pq.push({new_cost, link.src_area_id});
				}
			}
		}

		/* Expand distinct outgoing first hops for each source area. */
		for (int src = 0; src < area_count; ++src)
		{
			const size_t idx = (size_t)src * (size_t)area_count + (size_t)dst;
			std::vector<QnnRouteEntry> &entries = oracle->route_entries[idx];

			for (int link_id : oracle->outgoing_links[(size_t)src])
			{
				const QnnNavLink &link = oracle->links[(size_t)link_id];
				const float remaining_cost = best_cost[(size_t)link.dst_area_id];

				if (!std::isfinite(remaining_cost))
					continue;
				entries.push_back({link.link_id, link.travel_time + remaining_cost});
			}
		}
	}
}

static const QnnRouteEntry *qnn_nav_find_best_route_entry(const std::vector<QnnRouteEntry> &entries)
{
	const QnnRouteEntry *best_entry;

	best_entry = nullptr;
	for (const QnnRouteEntry &entry : entries)
	{
		if (best_entry == nullptr
			|| entry.cost + 0.0001f < best_entry->cost
			|| (fabsf(entry.cost - best_entry->cost) <= 0.0001f && entry.link_id < best_entry->link_id))
		{
			best_entry = &entry;
		}
	}
	return best_entry;
}

static void qnn_nav_fill_summary_counts(qnn_nav_oracle_runtime_t *oracle)
{
	size_t link_index;

	if (oracle == nullptr)
		return;

	memset(&oracle->summary, 0, sizeof(oracle->summary));
	oracle->summary.area_count = (int)oracle->areas.size();
	oracle->summary.cluster_count = (int)oracle->clusters.size();
	oracle->summary.total_link_count = (int)oracle->links.size();
	if (!oracle->clusters.empty())
	{
		int total_cluster_areas;

		total_cluster_areas = 0;
		oracle->summary.min_cluster_area_count = std::numeric_limits<int>::max();
		for (const QnnNavCluster &cluster : oracle->clusters)
		{
			total_cluster_areas += cluster.area_count;
			oracle->summary.min_cluster_area_count = std::min(oracle->summary.min_cluster_area_count, cluster.area_count);
			oracle->summary.max_cluster_area_count = std::max(oracle->summary.max_cluster_area_count, cluster.area_count);
		}
		oracle->summary.avg_cluster_area_count = (float)total_cluster_areas / (float)oracle->clusters.size();
	}

	for (link_index = 0; link_index < oracle->links.size(); ++link_index)
	{
		switch (oracle->links[link_index].travel_type)
		{
		case QNN_NAV_TRAVEL_WALK:
			oracle->summary.walk_link_count += 1;
			break;
		case QNN_NAV_TRAVEL_TELEPORT:
			oracle->summary.teleport_link_count += 1;
			break;
		case QNN_NAV_TRAVEL_ELEVATOR:
			oracle->summary.lift_link_count += 1;
			break;
		case QNN_NAV_TRAVEL_PUSH:
			oracle->summary.push_link_count += 1;
			break;
		case QNN_NAV_TRAVEL_DROP:
			oracle->summary.drop_link_count += 1;
			break;
		default:
			break;
		}
	}
}

}  // namespace

extern "C" const char *qnn_nav_travel_type_name(qnn_nav_travel_type_t travel_type)
{
	switch (travel_type)
	{
	case QNN_NAV_TRAVEL_WALK:
		return "WALK";
	case QNN_NAV_TRAVEL_DROP:
		return "DROP";
	case QNN_NAV_TRAVEL_TELEPORT:
		return "TELEPORT";
	case QNN_NAV_TRAVEL_ELEVATOR:
		return "ELEVATOR";
	case QNN_NAV_TRAVEL_PUSH:
		return "PUSH";
	case QNN_NAV_TRAVEL_GRAPPLE:
		return "GRAPPLE";
	default:
		return "INVALID";
	}
}

extern "C" qnn_nav_oracle_runtime_t *qnn_nav_oracle_build(
	const qnn_navmesh_runtime_t *navmesh,
	const qnn_nav_oracle_static_object_view_t *static_objects,
	int static_object_count,
	qnn_nav_oracle_summary_t *summary,
	char *error,
	size_t error_size)
{
	qnn_navmesh_poly_record_t *records;
	int record_count;
	qnn_nav_oracle_runtime_t *oracle;
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

	oracle = new (std::nothrow) qnn_nav_oracle_runtime_t();
	if (oracle == nullptr)
	{
		qnn_navmesh_free_poly_records(records);
		qnn_nav_set_error(error, error_size, "Out of memory while allocating nav oracle runtime");
		return nullptr;
	}
	oracle->navmesh = navmesh;
	oracle->areas.reserve((size_t)record_count);
	oracle->outgoing_links.resize((size_t)record_count);
	oracle->sorted_poly_refs.reserve((size_t)record_count);
	oracle->sorted_area_ids.reserve((size_t)record_count);

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
		oracle->sorted_poly_refs.push_back(records[area_index].poly_ref);
		oracle->sorted_area_ids.push_back(area.area_id);
	}

	qnn_nav_build_walk_links(oracle, records, record_count);
	qnn_nav_build_teleport_links(oracle, static_objects, static_object_count);
	qnn_nav_build_lift_links(oracle, static_objects, static_object_count);
	qnn_nav_build_push_links(oracle, static_objects, static_object_count);
	qnn_nav_build_drop_links(oracle);
	qnn_nav_build_clusters(oracle);

	qnn_nav_fill_summary_counts(oracle);
	qnn_nav_build_routing_cache(oracle);
	if (summary != nullptr)
		*summary = oracle->summary;

	qnn_navmesh_free_poly_records(records);
	return oracle;
}

extern "C" int qnn_nav_oracle_find_area(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *point,
	qnn_nav_area_result_t *result,
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
	area_id = qnn_nav_find_nearest_area_id(oracle, point, &nearest, error, error_size);
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

extern "C" int qnn_nav_oracle_find_cluster(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *point,
	qnn_nav_cluster_result_t *result,
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
	area_id = qnn_nav_find_nearest_area_id(oracle, point, &nearest, error, error_size);
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

extern "C" int qnn_nav_oracle_find_route(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *start,
	const float *end,
	qnn_nav_route_result_t *result,
	char *error,
	size_t error_size)
{
	qnn_navmesh_nearest_result_t start_nearest;
	qnn_navmesh_nearest_result_t end_nearest;
	int start_area_id;
	int end_area_id;
	int area_count;
	int cursor;

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
	result->first_travel_type = QNN_NAV_TRAVEL_INVALID;

	if (oracle == nullptr)
	{
		qnn_nav_set_error(error, error_size, "Route query requested before nav oracle was initialized");
		return 0;
	}

	memset(&start_nearest, 0, sizeof(start_nearest));
	memset(&end_nearest, 0, sizeof(end_nearest));
	start_area_id = qnn_nav_find_nearest_area_id(oracle, start, &start_nearest, error, error_size);
	if (start_area_id < 0)
		return 0;
	end_area_id = qnn_nav_find_nearest_area_id(oracle, end, &end_nearest, error, error_size);
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

	/* Use precomputed routing cache for O(1) first-hop lookup and path
	   reconstruction by following the cheapest cached first-hop chain. */
	{
		const size_t start_idx = (size_t)start_area_id * (size_t)area_count + (size_t)end_area_id;
		const QnnRouteEntry *best_entry = qnn_nav_find_best_route_entry(oracle->route_entries[start_idx]);

		if (best_entry == nullptr)
			return 0;

		result->found = 1;
		result->travel_time = best_entry->cost;

		/* Reconstruct the area/link path by following the cheapest first hop
		   for each cursor area toward the fixed destination. */
		cursor = start_area_id;
		result->area_count = 0;
		result->link_count = 0;
		while (cursor != end_area_id && result->area_count < QNN_NAV_ORACLE_MAX_ROUTE_AREAS)
		{
			const size_t idx = (size_t)cursor * (size_t)area_count + (size_t)end_area_id;
			const QnnRouteEntry *cursor_entry = qnn_nav_find_best_route_entry(oracle->route_entries[idx]);
			const int link_id = cursor_entry != nullptr ? cursor_entry->link_id : -1;

			result->area_ids[result->area_count++] = cursor;
			if (link_id < 0)
				break;
			if (result->link_count < QNN_NAV_ORACLE_MAX_ROUTE_AREAS - 1)
			{
				result->link_ids[result->link_count] = link_id;
				result->travel_types[result->link_count] = (int)oracle->links[(size_t)link_id].travel_type;
				result->link_count += 1;
			}
			cursor = oracle->links[(size_t)link_id].dst_area_id;
		}
		/* Append the destination area. */
		if (result->area_count < QNN_NAV_ORACLE_MAX_ROUTE_AREAS)
			result->area_ids[result->area_count++] = end_area_id;

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

extern "C" void qnn_nav_oracle_destroy(qnn_nav_oracle_runtime_t *oracle)
{
	delete oracle;
}

extern "C" void qnn_nav_oracle_write_summary_json(FILE *out, const qnn_nav_oracle_summary_t *summary)
{
	qnn_nav_oracle_summary_t empty;

	if (summary == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		summary = &empty;
	}
	fprintf(
		out,
		"{\"area_count\":%d,\"avg_cluster_area_count\":%.3f,\"cluster_count\":%d,\"drop_link_count\":%d,"
		"\"lift_link_count\":%d,\"max_cluster_area_count\":%d,\"min_cluster_area_count\":%d,"
		"\"push_link_count\":%d,\"teleport_link_count\":%d,\"total_link_count\":%d,\"walk_link_count\":%d}",
		summary->area_count,
		summary->avg_cluster_area_count,
		summary->cluster_count,
		summary->drop_link_count,
		summary->lift_link_count,
		summary->max_cluster_area_count,
		summary->min_cluster_area_count,
		summary->push_link_count,
		summary->teleport_link_count,
		summary->total_link_count,
		summary->walk_link_count);
}

extern "C" void qnn_nav_oracle_write_area_json(FILE *out, const qnn_nav_area_result_t *result)
{
	qnn_nav_area_result_t empty;

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

extern "C" void qnn_nav_oracle_write_cluster_json(FILE *out, const qnn_nav_cluster_result_t *result)
{
	qnn_nav_cluster_result_t empty;

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

extern "C" void qnn_nav_oracle_write_route_json(FILE *out, const qnn_nav_route_result_t *result)
{
	qnn_nav_route_result_t empty;
	int index;

	if (result == nullptr)
	{
		memset(&empty, 0, sizeof(empty));
		empty.start_area_id = -1;
		empty.end_area_id = -1;
		empty.next_area_id = -1;
		empty.first_link_id = -1;
		empty.first_travel_type = QNN_NAV_TRAVEL_INVALID;
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
		qnn_nav_travel_type_name(result->first_travel_type));
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
		fprintf(out, "\"%s\"", qnn_nav_travel_type_name((qnn_nav_travel_type_t)result->travel_types[index]));
	}
	fprintf(out, "]}");
}

extern "C" int qnn_nav_oracle_path_position(
	const qnn_nav_oracle_runtime_t *oracle,
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
		const QnnRouteEntry *best_entry = qnn_nav_find_best_route_entry(oracle->route_entries[route_index]);

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
			const QnnNavLink &link = oracle->links[(size_t)best_entry->link_id];

			for (axis = 0; axis < 3; ++axis)
				out_rel[axis] = link.start_pos[axis] - player_pos[axis];
			*out_route_cost = best_entry->cost;
		}
	}

	return 1;
}

extern "C" int qnn_nav_oracle_route_clusters(
	const qnn_nav_oracle_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	int *out_cluster_ids,
	int max_clusters,
	int *out_cluster_count)
{
	int area_count;
	int cursor;
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

	/* Walk the cached route from player to object, collecting unique
	   intermediate cluster IDs (excluding the player's cluster and the
	   destination cluster). */
	{
		int player_cluster = oracle->areas[(size_t)player_area_id].cluster_id;
		int object_cluster = oracle->areas[(size_t)object_area_id].cluster_id;

		count = 0;
		last_cluster = player_cluster;
		cursor = player_area_id;

		while (cursor != object_area_id && count < max_clusters)
		{
			const size_t idx = (size_t)cursor * (size_t)area_count + (size_t)object_area_id;
			const QnnRouteEntry *entry = qnn_nav_find_best_route_entry(oracle->route_entries[idx]);

			if (entry == nullptr || entry->link_id < 0 || entry->link_id >= (int)oracle->links.size())
				break;

			cursor = oracle->links[(size_t)entry->link_id].dst_area_id;
			if (cursor < 0 || cursor >= area_count)
				break;

			{
				int c = oracle->areas[(size_t)cursor].cluster_id;
				if (c != last_cluster && c != player_cluster && c != object_cluster)
				{
					out_cluster_ids[count++] = c;
					last_cluster = c;
				}
				else
				{
					last_cluster = c;
				}
			}
		}

		*out_cluster_count = count;
	}

	return 1;
}

