#ifndef QNN_NAVMESH_H
#define QNN_NAVMESH_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qnn_navmesh_runtime_s qnn_navmesh_runtime_t;

/* Shared error-formatting helper used by navmesh and nav oracle.
   Defined in qnn_navmesh.cpp, declared here so both TUs can share it. */
void qnn_nav_set_error(char *error, size_t error_size, const char *format, ...)
#ifdef __GNUC__
	__attribute__((format(printf, 3, 4)))
#endif
;

#define QNN_NAVMESH_MAX_NEIGHBORS 16
#define QNN_NAVMESH_MAX_PATH_REFS 128
#define QNN_NAVMESH_MAX_STRAIGHT_POINTS 64

typedef struct
{
	float	cell_size;
	float	cell_height;
	float	walkable_slope_angle;
	float	walkable_height;
	float	walkable_climb;
	float	walkable_radius;
	float	max_edge_len;
	float	max_simplification_error;
	int	min_region_size;
	int	merge_region_size;
	int	max_verts_per_poly;
	float	detail_sample_distance;
	float	detail_sample_max_error;
} qnn_navmesh_build_config_t;

typedef struct
{
	int	input_vertex_count;
	int	input_triangle_count;
	int	polygon_count;
	int	navmesh_vertex_count;
	int	detail_mesh_count;
	int	detail_vertex_count;
	int	detail_triangle_count;
} qnn_navmesh_summary_t;

typedef struct
{
	int	found;
	int	is_over_poly;
	unsigned long long poly_ref;
	float query_point[3];
	float nearest_point[3];
	float poly_center[3];
	float wall_distance;
	int	neighbor_count;
	unsigned long long neighbor_refs[QNN_NAVMESH_MAX_NEIGHBORS];
} qnn_navmesh_nearest_result_t;

typedef struct
{
	unsigned long long poly_ref;
	float	center[3];
	float	bounds_min[3];
	float	bounds_max[3];
	int	neighbor_count;
	unsigned long long neighbor_refs[QNN_NAVMESH_MAX_NEIGHBORS];
} qnn_navmesh_poly_record_t;

typedef struct
{
	int	found;
	unsigned long long start_ref;
	unsigned long long end_ref;
	float start_point[3];
	float end_point[3];
	float travel_distance;
	int	path_ref_count;
	unsigned long long path_refs[QNN_NAVMESH_MAX_PATH_REFS];
	int	straight_point_count;
	float straight_points[QNN_NAVMESH_MAX_STRAIGHT_POINTS * 3];
} qnn_navmesh_path_result_t;

qnn_navmesh_runtime_t *qnn_navmesh_build(
	const float *verts,
	int vertex_count,
	const int *tris,
	int triangle_count,
	const qnn_navmesh_build_config_t *config,
	qnn_navmesh_summary_t *summary,
	char *error,
	size_t error_size);
int qnn_navmesh_find_nearest(
	const qnn_navmesh_runtime_t *navmesh,
	const float *point,
	qnn_navmesh_nearest_result_t *result,
	char *error,
	size_t error_size);
int qnn_navmesh_collect_polys(
	const qnn_navmesh_runtime_t *navmesh,
	qnn_navmesh_poly_record_t **records,
	int *record_count,
	char *error,
	size_t error_size);
int qnn_navmesh_find_path(
	const qnn_navmesh_runtime_t *navmesh,
	const float *start,
	const float *end,
	qnn_navmesh_path_result_t *result,
	char *error,
	size_t error_size);
void qnn_navmesh_free_poly_records(qnn_navmesh_poly_record_t *records);
void qnn_navmesh_destroy(qnn_navmesh_runtime_t *navmesh);
void qnn_navmesh_write_summary_json(FILE *out, const qnn_navmesh_summary_t *summary);
void qnn_navmesh_write_nearest_json(FILE *out, const qnn_navmesh_nearest_result_t *result);
void qnn_navmesh_write_path_json(FILE *out, const qnn_navmesh_path_result_t *result);

#ifdef __cplusplus
}
#endif

#endif
