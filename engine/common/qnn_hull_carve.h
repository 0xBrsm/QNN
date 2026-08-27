#ifndef QNN_HULL_CARVE_H
#define QNN_HULL_CARVE_H

/*
 * qnn_hull_carve.h — exact hull-1 boundary geometry for the spatial
 * tokens (spatial-tokens-v2 P1b).
 *
 * Carves a model's clip hull 1 (qbsp pre-expanded by the player bbox)
 * into its COMPLETE set of boundary faces once, then answers per-tick
 * sector queries as exact geometric minima over that face set — no
 * rays, no sampling, nothing can hide between directions.  Adapted
 * from FrikBotNex nav_hull.cpp (same clipnode-tree winding carve; this
 * variant emits face records instead of navmesh triangles).
 *
 * Face convention: `normal` points OUT of the solid, toward the open
 * region (i.e. it faces the player) — the same orientation
 * PM_/SV_RecursiveHullCheck report on trace hits, so the spatial token
 * payload is convention-identical to the P1a trace path.  A point p in
 * open space satisfies normal·p − dist ≥ 0.
 *
 * Coordinates are MODEL-LOCAL (world = identity for cl.worldmodel;
 * brush submodels / movers carve once in local space and queries take
 * a per-instance translation for the live origin).
 */

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct model_s; /* engine model — full definition in per-engine headers */

typedef struct
{
	float normal[3];    /* unit, faces the open region */
	float dist;         /* plane: normal·x = dist */
	float mins[3];      /* polygon AABB (model-local) */
	float maxs[3];
	int   first_vert;   /* index into qnn_carve_set_t.verts (xyz triples) */
	int   vert_count;
} qnn_carve_face_t;

typedef struct
{
	qnn_carve_face_t *faces;
	int               face_count;
	float            *verts;       /* xyz triples, windings back-to-back */
	int               vert_count;

	/* XY uniform-grid broad-phase (built by QNN_CarveModelHull1; pure
	 * acceleration — queries return identical minima with or without
	 * it).  CSR layout: faces overlapping cell (cx, cy) are
	 * cell_faces[cell_start[cy*nx+cx] .. cell_start[cy*nx+cx+1]).
	 * visit_stamp/stamp_counter dedupe faces spanning multiple cells
	 * during a query (grid cells are visited in expanding rings around
	 * the query origin, stopping once the ring's nearest possible
	 * distance exceeds the current best). */
	float grid_org[2];
	float grid_cell;
	int   grid_nx, grid_ny;
	int  *cell_start;              /* nx*ny + 1 */
	int  *cell_faces;
	unsigned *visit_stamp;         /* face_count, query scratch */
	unsigned  stamp_counter;
} qnn_carve_set_t;

/* One participating face set for a query: the set plus the translation
 * of its model-local coordinates into world (0,0,0 for the world set;
 * the live entity origin for a mover).  Non-const: queries advance the
 * set's visit stamp. */
typedef struct
{
	qnn_carve_set_t *set;
	float            origin[3];
} qnn_carve_instance_t;

/* Carve `mod`'s clip hull 1 into `out` (malloc-backed; caller frees
 * with QNN_CarveSetFree).  Falls back to hull 0 when hull 1 has no
 * clipnodes (degenerate map).  Returns face count, 0 on failure. */
int QNN_CarveModelHull1(struct model_s *mod, qnn_carve_set_t *out);

/* Diagnostic-only export of the convex non-solid hull-1 cells that the
 * boundary carve already constructs internally.  Faces are split by the
 * neighboring clipnode leaves so portal adjacency is explicit.  This does
 * not change or retain extra state in the production carve. */
int QNN_CarveWriteHull1CellsJson(struct model_s *mod, FILE *out);

void QNN_CarveSetFree(qnn_carve_set_t *set);

/* Exact first intersection of a ray with the carved face set.  dir must
 * be unit length.  Only front faces participate (normal·dir < 0 — the
 * normals face the open region, so a ray hits a face travelling into its
 * solid); a face plane passing through the origin hits at distance 0
 * when entered, with no clipping degeneracy.  Returns 1 with out_normal
 * (world frame, unit) + out_point (world frame) on a hit within
 * max_dist, else 0.
 *
 * (The rev 4–7 volume-minimum queries — wedge, pyramid, vertical
 * column — were removed with the five-profile layout; recover from git
 * history if a future path needs volume minima.) */
int QNN_CarveQueryRay(const qnn_carve_instance_t *insts, int n_insts,
	const float origin[3], const float dir[3], float max_dist,
	float out_normal[3], float out_point[3]);

#ifdef __cplusplus
}
#endif

#endif /* QNN_HULL_CARVE_H */
