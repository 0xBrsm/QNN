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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <strings.h>

namespace {

float Dot3(const float *lhs, const float *rhs)
{
	return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

float BoundsAxisGap(float src_min, float src_max, float dst_min, float dst_max)
{
	if (src_max < dst_min)
		return dst_min - src_max;
	if (dst_max < src_min)
		return src_min - dst_max;
	return 0.0f;
}

float BoundsGapXY(const QnnNavArea &src, const QnnNavArea &dst)
{
	const float dx = BoundsAxisGap(src.bounds_min[0], src.bounds_max[0], dst.bounds_min[0], dst.bounds_max[0]);
	const float dy = BoundsAxisGap(src.bounds_min[1], src.bounds_max[1], dst.bounds_min[1], dst.bounds_max[1]);

	return sqrtf(dx * dx + dy * dy);
}

const char *PropertyValue(const qnn_route_static_object_view_t *object, const char *key)
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

int ParseVec3Value(const char *value, float *out)
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

float ParseFloatValue(const char *value, float fallback)
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

int HasLink(const qnn_route_runtime_t *oracle, int src_area_id, int dst_area_id, qnn_route_travel_type_t travel_type)
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

int ParseObjectBounds(const qnn_route_static_object_view_t *object, float *mins, float *maxs)
{
	const char *mins_value;
	const char *maxs_value;

	mins_value = PropertyValue(object, "qnn_model_bounds_min");
	maxs_value = PropertyValue(object, "qnn_model_bounds_max");
	if (!ParseVec3Value(mins_value, mins) || !ParseVec3Value(maxs_value, maxs))
		return 0;
	return 1;
}

/* Quake uses angle -1 for straight up, -2 for straight down. */
int TrySpecialAngle(float yaw, float *dir)
{
	if (fabsf(yaw - (-1.0f)) < 0.01f)
	{
		dir[0] = 0.0f;
		dir[1] = 0.0f;
		dir[2] = 1.0f;
		return 1;
	}
	if (fabsf(yaw - (-2.0f)) < 0.01f)
	{
		dir[0] = 0.0f;
		dir[1] = 0.0f;
		dir[2] = -1.0f;
		return 1;
	}
	return 0;
}

void ParseMovedir(
	const qnn_route_static_object_view_t *object,
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

	angle_value = PropertyValue(object, "angle");
	if (angle_value != nullptr && angle_value[0] != 0)
	{
		yaw = ParseFloatValue(angle_value, 0.0f);
		if (TrySpecialAngle(yaw, dir))
			return;
		yaw = yaw * (float)(M_PI / 180.0);
		dir[0] = cosf(yaw);
		dir[1] = sinf(yaw);
		dir[2] = 0.0f;
		return;
	}

	angles_value = PropertyValue(object, "angles");
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
			if (TrySpecialAngle(object->angles[1], dir))
				return;
		}
		yaw = object->angles[1] * (float)(M_PI / 180.0);
		dir[0] = cosf(yaw);
		dir[1] = sinf(yaw);
		dir[2] = 0.0f;
	}
}

}  // namespace

/* ── Cross-TU helpers ───────────────────────────────────────────── */

int QNN_LinkAreaForRef(const qnn_route_runtime_t *oracle, unsigned long long poly_ref)
{
	int low;
	int high;

	low = 0;
	high = (int)oracle->areas.size() - 1;
	while (low <= high)
	{
		const int mid = low + (high - low) / 2;
		if (oracle->areas[(size_t)mid].poly_ref == poly_ref)
			return oracle->areas[(size_t)mid].area_id;
		if (oracle->areas[(size_t)mid].poly_ref < poly_ref)
			low = mid + 1;
		else
			high = mid - 1;
	}
	return -1;
}

int QNN_LinkFindNearestAreaId(
	const qnn_route_runtime_t *oracle,
	const float *point,
	qnn_navmesh_nearest_result_t *nearest,
	char *error,
	size_t error_size)
{
	int area_id;

	if (!qnn_navmesh_find_nearest(oracle->navmesh, point, nearest, error, error_size))
		return -1;
	area_id = QNN_LinkAreaForRef(oracle, nearest->poly_ref);
	if (area_id < 0)
		qnn_nav_set_error(error, error_size, "Nearest navmesh polygon was not found in oracle area table");
	return area_id;
}

void QNN_LinkAdd(
	qnn_route_runtime_t *oracle,
	int src_area_id,
	int dst_area_id,
	qnn_route_travel_type_t travel_type,
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
	if (HasLink(oracle, src_area_id, dst_area_id, travel_type))
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
	if (travel_type != QNN_TRAVEL_WALK)
		oracle->areas[(size_t)src_area_id].special_link_count += 1;
}

/* ── Link builders ──────────────────────────────────────────────── */

void QNN_LinkBuildWalk(qnn_route_runtime_t *oracle, const qnn_navmesh_poly_record_t *records, int record_count)
{
	int area_index;

	for (area_index = 0; area_index < record_count; ++area_index)
	{
		int neighbor_index;
		const QnnNavArea &src = oracle->areas[(size_t)area_index];

		for (neighbor_index = 0; neighbor_index < records[area_index].neighbor_count; ++neighbor_index)
		{
			const int dst_area_id = QNN_LinkAreaForRef(oracle, records[area_index].neighbor_refs[neighbor_index]);

			if (dst_area_id < 0)
				continue;
			const float travel_time = RouteDistance(src.center, oracle->areas[(size_t)dst_area_id].center) / kWalkUnitsPerSecond;
			QNN_LinkAdd(
				oracle,
				src.area_id,
				dst_area_id,
				QNN_TRAVEL_WALK,
				src.center,
				oracle->areas[(size_t)dst_area_id].center,
				travel_time);
		}
	}
}

void QNN_LinkBuildTeleport(qnn_route_runtime_t *oracle, const qnn_route_static_object_view_t *static_objects, int static_object_count)
{
	int object_index;

	/* Build name-to-index map for teleport destinations so we avoid
	   O(N) scans per teleport source. */
	std::unordered_map<std::string, int> dest_by_name;
	for (int i = 0; i < static_object_count; ++i)
	{
		if (strcasecmp(static_objects[i].classname, "info_teleport_destination") != 0)
			continue;
		const char *name = PropertyValue(&static_objects[i], "targetname");
		if (name != nullptr && name[0] != 0)
			dest_by_name.emplace(std::string(name), i);
	}

	for (object_index = 0; object_index < static_object_count; ++object_index)
	{
		const qnn_route_static_object_view_t *source;
		const char *target_name;
		qnn_navmesh_nearest_result_t src_nearest;
		qnn_navmesh_nearest_result_t dst_nearest;
		char error[256];
		int src_area_id;
		int dst_area_id;

		source = &static_objects[object_index];
		if (strcasecmp(source->classname, "trigger_teleport") != 0)
			continue;
		target_name = PropertyValue(source, "target");
		if (target_name == nullptr || target_name[0] == 0)
			continue;

		auto it = dest_by_name.find(std::string(target_name));
		if (it == dest_by_name.end())
			continue;
		const int dest_index = it->second;

		memset(&src_nearest, 0, sizeof(src_nearest));
		memset(&dst_nearest, 0, sizeof(dst_nearest));
		memset(error, 0, sizeof(error));
		src_area_id = QNN_LinkFindNearestAreaId(oracle, source->origin, &src_nearest, error, sizeof(error));
		if (src_area_id < 0)
			continue;
		dst_area_id = QNN_LinkFindNearestAreaId(oracle, static_objects[dest_index].origin, &dst_nearest, error, sizeof(error));
		if (dst_area_id < 0)
			continue;
		QNN_LinkAdd(
			oracle,
			src_area_id,
			dst_area_id,
			QNN_TRAVEL_TELEPORT,
			source->origin,
			static_objects[dest_index].origin,
			kTeleportTravelTime);
	}
}

void QNN_LinkBuildLift(qnn_route_runtime_t *oracle, const qnn_route_static_object_view_t *static_objects, int static_object_count)
{
	int object_index;

	for (object_index = 0; object_index < static_object_count; ++object_index)
	{
		const qnn_route_static_object_view_t *object;
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
		if (!ParseObjectBounds(object, bounds_min, bounds_max))
			continue;

		top_pos[0] = object->origin[0];
		top_pos[1] = object->origin[1];
		top_pos[2] = object->origin[2];
		bottom_pos[0] = object->origin[0];
		bottom_pos[1] = object->origin[1];
		bottom_pos[2] = object->origin[2];

		lift_height = ParseFloatValue(PropertyValue(object, "height"), (bounds_max[2] - bounds_min[2]) - 8.0f);
		if (lift_height <= 0.0f)
			continue;
		bottom_pos[2] -= lift_height;

		lift_speed = ParseFloatValue(PropertyValue(object, "speed"), 150.0f);
		if (lift_speed <= 0.0f)
			lift_speed = 150.0f;
		travel_time = (lift_height / lift_speed) + kLiftBasePenalty;

		memset(&top_nearest, 0, sizeof(top_nearest));
		memset(&bottom_nearest, 0, sizeof(bottom_nearest));
		memset(error, 0, sizeof(error));
		top_area_id = QNN_LinkFindNearestAreaId(oracle, top_pos, &top_nearest, error, sizeof(error));
		if (top_area_id < 0)
			continue;
		bottom_area_id = QNN_LinkFindNearestAreaId(oracle, bottom_pos, &bottom_nearest, error, sizeof(error));
		if (bottom_area_id < 0 || bottom_area_id == top_area_id)
			continue;

		QNN_LinkAdd(
			oracle,
			top_area_id,
			bottom_area_id,
			QNN_TRAVEL_ELEVATOR,
			top_pos,
			bottom_pos,
			travel_time);
		QNN_LinkAdd(
			oracle,
			bottom_area_id,
			top_area_id,
			QNN_TRAVEL_ELEVATOR,
			bottom_pos,
			top_pos,
			travel_time);
	}
}

void QNN_LinkBuildPush(qnn_route_runtime_t *oracle, const qnn_route_static_object_view_t *static_objects, int static_object_count)
{
	int object_index;

	for (object_index = 0; object_index < static_object_count; ++object_index)
	{
		const qnn_route_static_object_view_t *object;
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
		src_area_id = QNN_LinkFindNearestAreaId(oracle, object->origin, &src_nearest, error, sizeof(error));
		if (src_area_id < 0)
			continue;

		ParseMovedir(object, movedir);
		launch_speed = ParseFloatValue(PropertyValue(object, "speed"), 100.0f) * 10.0f;
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
			sample_area_id = QNN_LinkFindNearestAreaId(oracle, sample_pos, &nearest, error, sizeof(error));
			if (sample_area_id < 0 || sample_area_id == src_area_id)
				continue;

			snap_distance = RouteDistance(sample_pos, nearest.nearest_point);
			if (snap_distance > kPushMaxSnapDistance)
				continue;

			displacement[0] = nearest.nearest_point[0] - object->origin[0];
			displacement[1] = nearest.nearest_point[1] - object->origin[1];
			displacement[2] = nearest.nearest_point[2] - object->origin[2];
			progress = Dot3(displacement, movedir);
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
			QNN_LinkAdd(
				oracle,
				src_area_id,
				best_area_id,
				QNN_TRAVEL_PUSH,
				best_start,
				best_end,
				best_time);
		}
	}
}

void QNN_LinkBuildDrop(qnn_route_runtime_t *oracle)
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

		auto it_lo = std::lower_bound(z_sorted.begin(), z_sorted.end(),
			std::make_pair(z_lo, std::numeric_limits<int>::min()));
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
			if (HasLink(oracle, src_area_id, dst_area_id, QNN_TRAVEL_WALK))
				continue;
			if (HasLink(oracle, src_area_id, dst_area_id, QNN_TRAVEL_DROP))
				continue;
			if (HasLink(oracle, dst_area_id, src_area_id, QNN_TRAVEL_WALK))
				continue;

			horizontal_gap = BoundsGapXY(src, dst);
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
			travel_time = RouteDistance(src.center, oracle->areas[(size_t)best_dst_area_id].center) / kWalkUnitsPerSecond;
			QNN_LinkAdd(
				oracle,
				src_area_id,
				best_dst_area_id,
				QNN_TRAVEL_DROP,
				src.center,
				oracle->areas[(size_t)best_dst_area_id].center,
				travel_time);
		}
	}
}
