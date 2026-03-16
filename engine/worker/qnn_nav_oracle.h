#ifndef QNN_NAV_ORACLE_H
#define QNN_NAV_ORACLE_H

#include <stddef.h>
#include <stdio.h>

#include "qnn_navmesh.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef QNN_NAV_ORACLE_RUNTIME_FWD
#define QNN_NAV_ORACLE_RUNTIME_FWD
typedef struct qnn_nav_oracle_runtime_s qnn_nav_oracle_runtime_t;
#endif

#define QNN_NAV_ORACLE_MAX_ROUTE_AREAS 128

typedef enum
{
	QNN_NAV_TRAVEL_INVALID = 0,
	QNN_NAV_TRAVEL_WALK = 1,
	QNN_NAV_TRAVEL_DROP = 2,
	QNN_NAV_TRAVEL_TELEPORT = 3,
	QNN_NAV_TRAVEL_ELEVATOR = 4,
	QNN_NAV_TRAVEL_PUSH = 5,
	QNN_NAV_TRAVEL_GRAPPLE = 6
} qnn_nav_travel_type_t;

typedef struct
{
	const char *key;
	const char *value;
} qnn_nav_oracle_property_view_t;

typedef struct
{
	const char *object_id;
	const char *category;
	const char *classname;
	const float *origin;
	const float *angles;
	const qnn_nav_oracle_property_view_t *properties;
	int property_count;
} qnn_nav_oracle_static_object_view_t;

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
} qnn_nav_oracle_summary_t;

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
} qnn_nav_area_result_t;

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
} qnn_nav_cluster_result_t;

typedef struct
{
	int	found;
	int	start_area_id;
	int	end_area_id;
	int	next_area_id;
	int	first_link_id;
	qnn_nav_travel_type_t first_travel_type;
	float	travel_time;
	int	area_count;
	int	area_ids[QNN_NAV_ORACLE_MAX_ROUTE_AREAS];
	int	link_count;
	int	link_ids[QNN_NAV_ORACLE_MAX_ROUTE_AREAS - 1];
	int	travel_types[QNN_NAV_ORACLE_MAX_ROUTE_AREAS - 1];
} qnn_nav_route_result_t;

qnn_nav_oracle_runtime_t *qnn_nav_oracle_build(
	const qnn_navmesh_runtime_t *navmesh,
	const qnn_nav_oracle_static_object_view_t *static_objects,
	int static_object_count,
	qnn_nav_oracle_summary_t *summary,
	char *error,
	size_t error_size);
int qnn_nav_oracle_find_area(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *point,
	qnn_nav_area_result_t *result,
	char *error,
	size_t error_size);
int qnn_nav_oracle_find_cluster(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *point,
	qnn_nav_cluster_result_t *result,
	char *error,
	size_t error_size);
int qnn_nav_oracle_find_route(
	const qnn_nav_oracle_runtime_t *oracle,
	const float *start,
	const float *end,
	qnn_nav_route_result_t *result,
	char *error,
	size_t error_size);
void qnn_nav_oracle_destroy(qnn_nav_oracle_runtime_t *oracle);
void qnn_nav_oracle_write_summary_json(FILE *out, const qnn_nav_oracle_summary_t *summary);
void qnn_nav_oracle_write_area_json(FILE *out, const qnn_nav_area_result_t *result);
void qnn_nav_oracle_write_cluster_json(FILE *out, const qnn_nav_cluster_result_t *result);
void qnn_nav_oracle_write_route_json(FILE *out, const qnn_nav_route_result_t *result);
const char *qnn_nav_travel_type_name(qnn_nav_travel_type_t travel_type);
int qnn_nav_oracle_path_position(
	const qnn_nav_oracle_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	const float *player_pos,
	const float *object_pos,
	float *out_rel,
	float *out_route_cost,
	char *error,
	size_t error_size);
int qnn_nav_oracle_route_clusters(
	const qnn_nav_oracle_runtime_t *oracle,
	int player_area_id,
	int object_area_id,
	int *out_cluster_ids,
	int max_clusters,
	int *out_cluster_count);
#ifdef __cplusplus
}
#endif

#endif
