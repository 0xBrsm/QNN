/*
 * qnn_hull_carve.cpp — exact hull-1 boundary geometry + center-ray
 * queries (spatial-tokens-v2 rev 8).  Contract and conventions in
 * qnn_hull_carve.h.  The rev 4–7 volume queries (wedge / pyramid /
 * vertical minima) lived here until the depth atlas superseded them —
 * recover them from git history if a future path needs volume minima.
 *
 * Carve method adapted from FrikBotNex nav_hull.cpp (qbsp-style winding
 * clipping through the clipnode tree): carve the model's padded bounding
 * box through hull 1; each non-solid leaf is a convex polytope; the
 * parts of its faces that border solid are the hull boundary.  A face
 * can border solid on one part and open space on another, so each
 * candidate face is re-filtered through the tree and only solid-backed
 * pieces are emitted.  This variant emits FACE RECORDS (plane + winding
 * + AABB) instead of triangulated navmesh input: no floor drop, no
 * hazard sampling, no triangulation — and normals are flipped at emit
 * so they face the OPEN region (trace-hit convention; see header).
 */

extern "C" {
#include "quakedef.h"
}

#include "qnn_hull_carve.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace {

const double CARVE_EPS       = 0.02;
const double CARVE_EXTENT    = 262144.0; /* base winding half-size */
const double CARVE_BOX_PAD   = 64.0;     /* carve box beyond model bounds */
const double CARVE_FACE_LIFT = 0.5;      /* lift off plane for neighbor test */
const double CARVE_MIN_AREA  = 0.5;

const float  QUERY_BEHIND_EPS = 0.25f;      /* origin-behind-plane tolerance */

struct V3
{
	double x, y, z;
};

static inline V3 v3(double x, double y, double z) { V3 r = {x, y, z}; return r; }
static inline V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V3 vscale(V3 a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline double vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 vcross(V3 a, V3 b)
{
	return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

typedef std::vector<V3> Winding;

/* Outward plane of a convex polytope face: interior satisfies n·x <= d.
   Winding is ordered so its geometric (right-hand) normal equals n. */
struct Face
{
	V3 n;
	double d;
	Winding w;
	bool from_clip; /* came from a clipnode plane (emittable boundary) */
};

typedef std::vector<Face> Polytope;

static V3 winding_normal(const Winding &w)
{
	/* Newell's method */
	V3 n = v3(0, 0, 0);
	for (size_t i = 0; i < w.size(); i++)
	{
		const V3 &a = w[i];
		const V3 &b = w[(i + 1) % w.size()];
		n.x += (a.y - b.y) * (a.z + b.z);
		n.y += (a.z - b.z) * (a.x + b.x);
		n.z += (a.x - b.x) * (a.y + b.y);
	}
	return n;
}

static double winding_area(const Winding &w)
{
	V3 n = winding_normal(w);
	return 0.5 * sqrt(vdot(n, n));
}

/* Large quad lying on plane (n,d), wound so geometric normal == n. */
static Winding base_winding(V3 n, double d)
{
	double ax = fabs(n.x), ay = fabs(n.y), az = fabs(n.z);
	V3 up = (az >= ax && az >= ay) ? v3(1, 0, 0) : v3(0, 0, 1);

	up = vsub(up, vscale(n, vdot(up, n)));
	double ulen = sqrt(vdot(up, up));
	Winding w;
	if (ulen < 1e-9)
		return w;
	up = vscale(up, 1.0 / ulen);

	V3 right = vcross(up, n);
	V3 org = vscale(n, d / vdot(n, n));

	up = vscale(up, CARVE_EXTENT);
	right = vscale(right, CARVE_EXTENT);

	w.push_back(vadd(vsub(org, right), up));
	w.push_back(vadd(vadd(org, right), up));
	w.push_back(vsub(vadd(org, right), up));
	w.push_back(vsub(vsub(org, right), up));

	if (vdot(winding_normal(w), n) < 0)
	{
		Winding rev(w.rbegin(), w.rend());
		w.swap(rev);
	}
	return w;
}

/* Sutherland–Hodgman: keep the part with n·x <= d. Preserves order. */
static Winding clip_winding(const Winding &w, V3 n, double d)
{
	Winding out;
	if (w.size() < 3)
		return out;

	std::vector<double> dist(w.size());
	bool any_front = false, any_back = false;
	for (size_t i = 0; i < w.size(); i++)
	{
		dist[i] = vdot(w[i], n) - d;
		if (dist[i] > CARVE_EPS) any_front = true;
		if (dist[i] < -CARVE_EPS) any_back = true;
	}
	if (!any_front)
		return w; /* fully kept */
	if (!any_back)
		return out; /* fully clipped */

	for (size_t i = 0; i < w.size(); i++)
	{
		size_t j = (i + 1) % w.size();
		double di = dist[i], dj = dist[j];

		if (di <= CARVE_EPS)
			out.push_back(w[i]);
		if ((di < -CARVE_EPS && dj > CARVE_EPS) ||
		    (di > CARVE_EPS && dj < -CARVE_EPS))
		{
			double t = di / (di - dj);
			out.push_back(vadd(w[i], vscale(vsub(w[j], w[i]), t)));
		}
	}
	if (out.size() < 3)
		out.clear();
	return out;
}

struct Builder
{
	hull_t *hull;
	std::vector<qnn_carve_face_t> faces;
	std::vector<float> verts;
};

/* Emit one solid-backed face piece as a face record.  The carve's face
 * normal points INTO the solid (out of the empty polytope); flip so the
 * emitted normal faces the OPEN region, matching trace-hit convention:
 * open-side points satisfy normal·p − dist ≥ 0. */
static void emit_face(Builder *b, const Face &f, const Winding &w, V3 unlift)
{
	qnn_carve_face_t rec;
	double n_len = sqrt(vdot(f.n, f.n));
	if (n_len < 1e-9 || w.size() < 3)
		return;

	rec.normal[0] = (float)(-f.n.x / n_len);
	rec.normal[1] = (float)(-f.n.y / n_len);
	rec.normal[2] = (float)(-f.n.z / n_len);
	rec.dist      = (float)(-f.d / n_len);

	rec.first_vert = (int)(b->verts.size() / 3);
	rec.vert_count = (int)w.size();
	rec.mins[0] = rec.mins[1] = rec.mins[2] = 1e30f;
	rec.maxs[0] = rec.maxs[1] = rec.maxs[2] = -1e30f;
	/* f.n points into solid, while the emitted normal is -f.n (toward
	 * open space).  Reverse the winding with the normal so geometric
	 * winding orientation and rec.normal stay consistent.  The query's
	 * projection-inside test relies on that invariant. */
	for (size_t i = 0; i < w.size(); i++)
	{
		V3 p = vadd(w[w.size() - 1 - i], unlift);
		float x = (float)p.x, y = (float)p.y, z = (float)p.z;
		b->verts.push_back(x);
		b->verts.push_back(y);
		b->verts.push_back(z);
		if (x < rec.mins[0]) rec.mins[0] = x;
		if (y < rec.mins[1]) rec.mins[1] = y;
		if (z < rec.mins[2]) rec.mins[2] = z;
		if (x > rec.maxs[0]) rec.maxs[0] = x;
		if (y > rec.maxs[1]) rec.maxs[1] = y;
		if (z > rec.maxs[2]) rec.maxs[2] = z;
	}
	b->faces.push_back(rec);
}

/* Push a (lifted) face winding through the clipnode tree and emit only
 * the pieces that land in solid leaves — the true hull boundary. */
static void emit_solid_parts(Builder *b, const Face &f, int node_num,
	const Winding &w, V3 unlift)
{
	Winding cur = w;
	while (cur.size() >= 3)
	{
		if (node_num < 0)
		{
			if (node_num == CONTENTS_SOLID || node_num == CONTENTS_SKY)
				emit_face(b, f, cur, unlift);
			return;
		}

		dclipnode_t *node = b->hull->clipnodes + node_num;
		mplane_t *plane = b->hull->planes + node->planenum;
		V3 n = v3(plane->normal[0], plane->normal[1], plane->normal[2]);
		double d = plane->dist;

		/* front: n·x >= d  (children[0]) */
		Winding front = clip_winding(cur, vscale(n, -1.0), -d);
		Winding back = clip_winding(cur, n, d);

		if (front.size() >= 3)
			emit_solid_parts(b, f, node->children[0], front, unlift);
		node_num = node->children[1];
		cur.swap(back);
	}
}

static void carve_leaf(Builder *b, int contents, const Polytope &poly)
{
	if (contents == CONTENTS_SOLID || contents == CONTENTS_SKY)
		return;

	for (size_t i = 0; i < poly.size(); i++)
	{
		const Face &f = poly[i];
		if (!f.from_clip || f.w.size() < 3)
			continue;
		if (winding_area(f.w) < CARVE_MIN_AREA)
			continue;

		/* Lift slightly toward the neighbor so tree classification is
		 * unambiguous; emit_face shifts back via unlift. */
		double f_n_len = sqrt(vdot(f.n, f.n));
		V3 lift = vscale(f.n, CARVE_FACE_LIFT / f_n_len);
		Winding lifted;
		lifted.reserve(f.w.size());
		for (size_t k = 0; k < f.w.size(); k++)
			lifted.push_back(vadd(f.w[k], lift));

		emit_solid_parts(b, f, b->hull->firstclipnode, lifted,
			vscale(lift, -1.0));
	}
}

static void carve_node(Builder *b, int node_num, const Polytope &poly)
{
	if (poly.size() < 4)
		return; /* degenerate region, no volume */

	if (node_num < 0)
	{
		carve_leaf(b, node_num, poly);
		return;
	}

	dclipnode_t *node = b->hull->clipnodes + node_num;
	mplane_t *plane = b->hull->planes + node->planenum;
	V3 n = v3(plane->normal[0], plane->normal[1], plane->normal[2]);
	double d = plane->dist;

	Polytope front, back;
	front.reserve(poly.size() + 1);
	back.reserve(poly.size() + 1);

	for (size_t i = 0; i < poly.size(); i++)
	{
		const Face &f = poly[i];
		Winding fw = clip_winding(f.w, vscale(n, -1.0), -d); /* keep n·x >= d */
		Winding bw = clip_winding(f.w, n, d);                /* keep n·x <= d */
		if (fw.size() >= 3)
		{
			Face nf = {f.n, f.d, Winding(), f.from_clip};
			nf.w.swap(fw);
			front.push_back(nf);
		}
		if (bw.size() >= 3)
		{
			Face nb = {f.n, f.d, Winding(), f.from_clip};
			nb.w.swap(bw);
			back.push_back(nb);
		}
	}

	/* Cap each side with the split plane, clipped to the polytope. */
	Winding cap_front = base_winding(vscale(n, -1.0), -d); /* outward -n */
	Winding cap_back = base_winding(n, d);                 /* outward +n */
	for (size_t i = 0; i < poly.size() && cap_front.size() >= 3; i++)
		cap_front = clip_winding(cap_front, poly[i].n, poly[i].d);
	for (size_t i = 0; i < poly.size() && cap_back.size() >= 3; i++)
		cap_back = clip_winding(cap_back, poly[i].n, poly[i].d);

	if (cap_front.size() >= 3)
	{
		Face cf = {vscale(n, -1.0), -d, Winding(), true};
		cf.w.swap(cap_front);
		front.push_back(cf);
	}
	if (cap_back.size() >= 3)
	{
		Face cb = {n, d, Winding(), true};
		cb.w.swap(cap_back);
		back.push_back(cb);
	}

	carve_node(b, node->children[0], front);
	carve_node(b, node->children[1], back);
}

static Polytope box_polytope(const float *mins, const float *maxs)
{
	Polytope poly;
	double lo[3], hi[3];
	for (int i = 0; i < 3; i++)
	{
		lo[i] = mins[i] - CARVE_BOX_PAD;
		hi[i] = maxs[i] + CARVE_BOX_PAD;
	}

	for (int axis = 0; axis < 3; axis++)
	{
		for (int side = 0; side < 2; side++)
		{
			V3 n = v3(0, 0, 0);
			double d;
			double nsign = (side == 0) ? 1.0 : -1.0;
			if (axis == 0) n.x = nsign;
			else if (axis == 1) n.y = nsign;
			else n.z = nsign;
			d = (side == 0) ? hi[axis] : -lo[axis];

			Face f = {n, d, base_winding(n, d), false};
			for (int a2 = 0; a2 < 3 && f.w.size() >= 3; a2++)
			{
				if (a2 == axis) continue;
				V3 cn = v3(0, 0, 0);
				if (a2 == 0) cn.x = 1.0;
				else if (a2 == 1) cn.y = 1.0;
				else cn.z = 1.0;
				f.w = clip_winding(f.w, cn, hi[a2]);
				if (f.w.size() < 3) break;
				if (a2 == 0) cn.x = -1.0;
				else if (a2 == 1) cn.y = -1.0;
				else cn.z = -1.0;
				f.w = clip_winding(f.w, cn, -lo[a2]);
			}
			if (f.w.size() >= 3)
				poly.push_back(f);
		}
	}
	return poly;
}

/* ── Diagnostic convex-cell export ──────────────────────────────
 *
 * carve_node constructs the exact convex polytope for every non-solid
 * clipnode leaf, then discards the volume after extracting its solid-backed
 * boundary.  The cell-memory diagnostic exposes that intermediate on demand;
 * the production carve retains no extra state. */

enum CellFaceKind
{
	CELL_FACE_SOLID = 0,
	CELL_FACE_PORTAL = 1,
	CELL_FACE_BOUNDS = 2,
	CELL_FACE_UNRESOLVED = 3
};

struct CellFacePiece
{
	V3 n;
	double d;
	Winding w;
	CellFaceKind kind;
	int neighbor;
};

struct CellRecord
{
	int contents;
	int leaf_parent;
	int leaf_side;
	Polytope poly;
	std::vector<CellFacePiece> pieces;
};

struct CellBuilder
{
	hull_t *hull;
	std::vector<CellRecord> cells;
	std::map<std::pair<int, int>, int> leaf_to_cell;
};

static void collect_cell_node(CellBuilder *b, int node_num,
	const Polytope &poly, int leaf_parent, int leaf_side)
{
	if (poly.size() < 4)
		return;
	if (node_num < 0)
	{
		if (node_num != CONTENTS_SOLID && node_num != CONTENTS_SKY)
		{
			CellRecord cell;
			cell.contents = node_num;
			cell.leaf_parent = leaf_parent;
			cell.leaf_side = leaf_side;
			cell.poly = poly;
			int index = (int)b->cells.size();
			b->cells.push_back(cell);
			b->leaf_to_cell[std::make_pair(leaf_parent, leaf_side)] = index;
		}
		return;
	}

	dclipnode_t *node = b->hull->clipnodes + node_num;
	mplane_t *plane = b->hull->planes + node->planenum;
	V3 n = v3(plane->normal[0], plane->normal[1], plane->normal[2]);
	double d = plane->dist;
	Polytope front, back;
	front.reserve(poly.size() + 1);
	back.reserve(poly.size() + 1);

	for (size_t i = 0; i < poly.size(); i++)
	{
		const Face &f = poly[i];
		Winding fw = clip_winding(f.w, vscale(n, -1.0), -d);
		Winding bw = clip_winding(f.w, n, d);
		if (fw.size() >= 3)
		{
			Face nf = {f.n, f.d, Winding(), f.from_clip};
			nf.w.swap(fw);
			front.push_back(nf);
		}
		if (bw.size() >= 3)
		{
			Face nb = {f.n, f.d, Winding(), f.from_clip};
			nb.w.swap(bw);
			back.push_back(nb);
		}
	}

	Winding cap_front = base_winding(vscale(n, -1.0), -d);
	Winding cap_back = base_winding(n, d);
	for (size_t i = 0; i < poly.size() && cap_front.size() >= 3; i++)
		cap_front = clip_winding(cap_front, poly[i].n, poly[i].d);
	for (size_t i = 0; i < poly.size() && cap_back.size() >= 3; i++)
		cap_back = clip_winding(cap_back, poly[i].n, poly[i].d);
	if (cap_front.size() >= 3)
	{
		Face cf = {vscale(n, -1.0), -d, Winding(), true};
		cf.w.swap(cap_front);
		front.push_back(cf);
	}
	if (cap_back.size() >= 3)
	{
		Face cb = {n, d, Winding(), true};
		cb.w.swap(cap_back);
		back.push_back(cb);
	}

	collect_cell_node(b, node->children[0], front, node_num, 0);
	collect_cell_node(b, node->children[1], back, node_num, 1);
}

/* Split a lifted cell face through the complete hull.  Each resulting piece
 * is backed by exactly one neighboring terminal edge, so portal adjacency is
 * explicit even when a single cell face borders several neighboring leaves. */
static void classify_neighbor_parts(CellBuilder *b, int source_cell,
	const Face &face, int node_num, const Winding &w, V3 unlift,
	int leaf_parent, int leaf_side)
{
	Winding cur = w;
	while (cur.size() >= 3)
	{
		if (node_num < 0)
		{
			CellFacePiece piece;
			piece.n = face.n;
			piece.d = face.d;
			piece.neighbor = -1;
			if (node_num == CONTENTS_SOLID || node_num == CONTENTS_SKY)
				piece.kind = CELL_FACE_SOLID;
			else
			{
				std::map<std::pair<int, int>, int>::const_iterator found =
					b->leaf_to_cell.find(std::make_pair(leaf_parent, leaf_side));
				if (found == b->leaf_to_cell.end())
					piece.kind = CELL_FACE_UNRESOLVED;
				else if (found->second == source_cell)
					piece.kind = CELL_FACE_BOUNDS;
				else
				{
					piece.kind = CELL_FACE_PORTAL;
					piece.neighbor = found->second;
				}
			}
			piece.w.reserve(cur.size());
			for (size_t i = 0; i < cur.size(); i++)
				piece.w.push_back(vadd(cur[i], unlift));
			if (winding_area(piece.w) >= CARVE_MIN_AREA)
				b->cells[(size_t)source_cell].pieces.push_back(piece);
			return;
		}

		int current_node = node_num;
		dclipnode_t *node = b->hull->clipnodes + current_node;
		mplane_t *plane = b->hull->planes + node->planenum;
		V3 n = v3(plane->normal[0], plane->normal[1], plane->normal[2]);
		double d = plane->dist;
		Winding front = clip_winding(cur, vscale(n, -1.0), -d);
		Winding back = clip_winding(cur, n, d);

		if (front.size() >= 3)
			classify_neighbor_parts(b, source_cell, face, node->children[0],
				front, unlift, current_node, 0);
		node_num = node->children[1];
		leaf_parent = current_node;
		leaf_side = 1;
		cur.swap(back);
	}
}

static void build_cell_faces(CellBuilder *b)
{
	for (size_t ci = 0; ci < b->cells.size(); ci++)
	{
		CellRecord &cell = b->cells[ci];
		for (size_t fi = 0; fi < cell.poly.size(); fi++)
		{
			const Face &face = cell.poly[fi];
			double n_len = sqrt(vdot(face.n, face.n));
			if (n_len < 1e-9 || face.w.size() < 3)
				continue;
			V3 lift = vscale(face.n, CARVE_FACE_LIFT / n_len);
			Winding lifted;
			lifted.reserve(face.w.size());
			for (size_t vi = 0; vi < face.w.size(); vi++)
				lifted.push_back(vadd(face.w[vi], lift));
			classify_neighbor_parts(b, (int)ci, face,
				b->hull->firstclipnode, lifted, vscale(lift, -1.0), -1, -1);
		}
	}
}

static const char *cell_face_kind_name(CellFaceKind kind)
{
	switch (kind)
	{
	case CELL_FACE_SOLID: return "solid";
	case CELL_FACE_PORTAL: return "portal";
	case CELL_FACE_BOUNDS: return "bounds";
	default: return "unresolved";
	}
}

static void write_v3(FILE *out, V3 value)
{
	fprintf(out, "[%.6f,%.6f,%.6f]", value.x, value.y, value.z);
}

static void write_cells_json(CellBuilder *b, FILE *out)
{
	int solid = 0, portal = 0, bounds = 0, unresolved = 0;
	for (size_t ci = 0; ci < b->cells.size(); ci++)
		for (size_t fi = 0; fi < b->cells[ci].pieces.size(); fi++)
			switch (b->cells[ci].pieces[fi].kind)
			{
			case CELL_FACE_SOLID: solid++; break;
			case CELL_FACE_PORTAL: portal++; break;
			case CELL_FACE_BOUNDS: bounds++; break;
			default: unresolved++; break;
			}

	fprintf(out, "{\"count\":%d,\"solid_faces\":%d,\"portal_faces\":%d,"
		"\"bounds_faces\":%d,\"unresolved_faces\":%d,\"cells\":[",
		(int)b->cells.size(), solid, portal, bounds, unresolved);
	for (size_t ci = 0; ci < b->cells.size(); ci++)
	{
		const CellRecord &cell = b->cells[ci];
		V3 mins = v3(1e30, 1e30, 1e30);
		V3 maxs = v3(-1e30, -1e30, -1e30);
		V3 center = v3(0, 0, 0);
		size_t center_count = 0;
		for (size_t fi = 0; fi < cell.poly.size(); fi++)
			for (size_t vi = 0; vi < cell.poly[fi].w.size(); vi++)
			{
				V3 p = cell.poly[fi].w[vi];
				if (p.x < mins.x) mins.x = p.x;
				if (p.y < mins.y) mins.y = p.y;
				if (p.z < mins.z) mins.z = p.z;
				if (p.x > maxs.x) maxs.x = p.x;
				if (p.y > maxs.y) maxs.y = p.y;
				if (p.z > maxs.z) maxs.z = p.z;
				center = vadd(center, p);
				center_count++;
			}
		if (center_count > 0)
			center = vscale(center, 1.0 / center_count);

		fprintf(out, "%s{\"id\":%d,\"contents\":%d,\"leaf_parent\":%d,"
			"\"leaf_side\":%d,\"center\":", ci ? "," : "", (int)ci,
			cell.contents, cell.leaf_parent, cell.leaf_side);
		write_v3(out, center);
		fprintf(out, ",\"mins\":");
		write_v3(out, mins);
		fprintf(out, ",\"maxs\":");
		write_v3(out, maxs);
		fprintf(out, ",\"planes\":[");
		for (size_t fi = 0; fi < cell.poly.size(); fi++)
		{
			const Face &face = cell.poly[fi];
			double n_len = sqrt(vdot(face.n, face.n));
			V3 normal = n_len > 0 ? vscale(face.n, 1.0 / n_len) : face.n;
			fprintf(out, "%s{\"normal\":", fi ? "," : "");
			write_v3(out, normal);
			fprintf(out, ",\"dist\":%.6f}", n_len > 0 ? face.d / n_len : face.d);
		}
		fprintf(out, "],\"faces\":[");
		for (size_t fi = 0; fi < cell.pieces.size(); fi++)
		{
			const CellFacePiece &piece = cell.pieces[fi];
			double n_len = sqrt(vdot(piece.n, piece.n));
			V3 normal = n_len > 0 ? vscale(piece.n, 1.0 / n_len) : piece.n;
			fprintf(out, "%s{\"kind\":\"%s\",\"neighbor\":%d,"
				"\"normal\":", fi ? "," : "",
				cell_face_kind_name(piece.kind), piece.neighbor);
			write_v3(out, normal);
			fprintf(out, ",\"dist\":%.6f,\"verts\":[",
				n_len > 0 ? piece.d / n_len : piece.d);
			for (size_t vi = 0; vi < piece.w.size(); vi++)
			{
				if (vi) fputc(',', out);
				write_v3(out, piece.w[vi]);
			}
			fprintf(out, "]}");
		}
		fprintf(out, "]}");
	}
	fprintf(out, "]}");
}

/* ── Query helpers ─────────────────────────────────────────────── */

/* 3D distance + closest point from p to an already-world-space winding.
 * Projection-inside → perpendicular projection; else nearest edge point.
 * The winding is convex and oriented to match n. */

/* 2D point-in-polygon (XY projection, translated), crossing parity —
 * robust to either winding orientation. */
static bool point_in_poly_xy(float px, float py,
	const float *verts, int first, int count, const float org[3])
{
	bool in = false;
	for (int i = 0, j = count - 1; i < count; j = i++)
	{
		float xi = verts[3 * (first + i) + 0] + org[0];
		float yi = verts[3 * (first + i) + 1] + org[1];
		float xj = verts[3 * (first + j) + 0] + org[0];
		float yj = verts[3 * (first + j) + 1] + org[1];
		if (((yi > py) != (yj > py)) &&
		    (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
			in = !in;
	}
	return in;
}

} /* namespace */

/* ── Public API ────────────────────────────────────────────────── */

/* Build the XY uniform-grid broad-phase over the face AABBs.  Pure
 * acceleration: queries visit cells in expanding rings and stop when
 * the ring's nearest possible XY distance exceeds the running best, so
 * minima are identical to the linear scan.  On alloc failure the grid
 * stays empty and queries fall back to the linear path. */
static void build_grid(qnn_carve_set_t *out)
{
	const float CELL_MIN = 256.0f;
	float lo[2] = { 1e30f, 1e30f }, hi[2] = { -1e30f, -1e30f };
	for (int i = 0; i < out->face_count; i++)
	{
		const qnn_carve_face_t *f = &out->faces[i];
		for (int a = 0; a < 2; a++)
		{
			if (f->mins[a] < lo[a]) lo[a] = f->mins[a];
			if (f->maxs[a] > hi[a]) hi[a] = f->maxs[a];
		}
	}
	if (lo[0] > hi[0])
		return;

	float cell = CELL_MIN;
	int nx = (int)((hi[0] - lo[0]) / cell) + 1;
	int ny = (int)((hi[1] - lo[1]) / cell) + 1;
	while ((long)nx * ny > 128 * 128)
	{
		cell *= 2.0f;
		nx = (int)((hi[0] - lo[0]) / cell) + 1;
		ny = (int)((hi[1] - lo[1]) / cell) + 1;
	}

	std::vector<int> counts((size_t)nx * ny + 1, 0);
	for (int i = 0; i < out->face_count; i++)
	{
		const qnn_carve_face_t *f = &out->faces[i];
		int x0 = (int)((f->mins[0] - lo[0]) / cell);
		int x1 = (int)((f->maxs[0] - lo[0]) / cell);
		int y0 = (int)((f->mins[1] - lo[1]) / cell);
		int y1 = (int)((f->maxs[1] - lo[1]) / cell);
		for (int y = y0; y <= y1; y++)
			for (int x = x0; x <= x1; x++)
				counts[(size_t)y * nx + x + 1]++;
	}
	for (size_t i = 1; i < counts.size(); i++)
		counts[i] += counts[i - 1];

	int total = counts.back();
	int *cell_start = (int *)malloc(counts.size() * sizeof(int));
	int *cell_faces = (int *)malloc((size_t)total * sizeof(int));
	unsigned *stamps = (unsigned *)calloc((size_t)out->face_count, sizeof(unsigned));
	if (cell_start == NULL || cell_faces == NULL || stamps == NULL)
	{
		free(cell_start);
		free(cell_faces);
		free(stamps);
		return;
	}
	memcpy(cell_start, counts.data(), counts.size() * sizeof(int));

	std::vector<int> cursor(counts.begin(), counts.end() - 1);
	for (int i = 0; i < out->face_count; i++)
	{
		const qnn_carve_face_t *f = &out->faces[i];
		int x0 = (int)((f->mins[0] - lo[0]) / cell);
		int x1 = (int)((f->maxs[0] - lo[0]) / cell);
		int y0 = (int)((f->mins[1] - lo[1]) / cell);
		int y1 = (int)((f->maxs[1] - lo[1]) / cell);
		for (int y = y0; y <= y1; y++)
			for (int x = x0; x <= x1; x++)
				cell_faces[cursor[(size_t)y * nx + x]++] = i;
	}

	out->grid_org[0] = lo[0];
	out->grid_org[1] = lo[1];
	out->grid_cell = cell;
	out->grid_nx = nx;
	out->grid_ny = ny;
	out->cell_start = cell_start;
	out->cell_faces = cell_faces;
	out->visit_stamp = stamps;
	out->stamp_counter = 0;
}

extern "C" int QNN_CarveModelHull1(struct model_s *mod, qnn_carve_set_t *out)
{
	if (out == NULL)
		return 0;
	memset(out, 0, sizeof(*out));
	if (mod == NULL)
		return 0;

	hull_t *hull = &mod->hulls[1];
	if (hull->clipnodes == NULL || hull->planes == NULL)
		hull = &mod->hulls[0]; /* degenerate map — point fallback */
	if (hull->clipnodes == NULL || hull->planes == NULL)
		return 0;

	Builder b;
	b.hull = hull;
	carve_node(&b, hull->firstclipnode, box_polytope(mod->mins, mod->maxs));

	if (b.faces.empty())
		return 0;

	out->faces = (qnn_carve_face_t *)malloc(b.faces.size() * sizeof(qnn_carve_face_t));
	out->verts = (float *)malloc(b.verts.size() * sizeof(float));
	if (out->faces == NULL || out->verts == NULL)
	{
		free(out->faces);
		free(out->verts);
		memset(out, 0, sizeof(*out));
		return 0;
	}
	memcpy(out->faces, b.faces.data(), b.faces.size() * sizeof(qnn_carve_face_t));
	memcpy(out->verts, b.verts.data(), b.verts.size() * sizeof(float));
	out->face_count = (int)b.faces.size();
	out->vert_count = (int)(b.verts.size() / 3);
	/* QNN_SPATIAL_LINEAR=1 skips the broad-phase so every query runs
	 * the linear face scan — the grid/linear purity-proof knob. */
	{
		const char *linear = getenv("QNN_SPATIAL_LINEAR");
		if (linear == NULL || linear[0] == '\0' || linear[0] == '0')
			build_grid(out);
	}
	return out->face_count;
}

extern "C" int QNN_CarveWriteHull1CellsJson(struct model_s *mod, FILE *out)
{
	if (mod == NULL || out == NULL)
		return 0;
	hull_t *hull = &mod->hulls[1];
	if (hull->clipnodes == NULL || hull->planes == NULL)
		hull = &mod->hulls[0];
	if (hull->clipnodes == NULL || hull->planes == NULL)
		return 0;

	CellBuilder builder;
	builder.hull = hull;
	collect_cell_node(&builder, hull->firstclipnode,
		box_polytope(mod->mins, mod->maxs), -1, -1);
	if (builder.cells.empty())
		return 0;
	build_cell_faces(&builder);
	write_cells_json(&builder, out);
	return (int)builder.cells.size();
}

extern "C" void QNN_CarveSetFree(qnn_carve_set_t *set)
{
	if (set == NULL)
		return;
	free(set->faces);
	free(set->verts);
	free(set->cell_start);
	free(set->cell_faces);
	free(set->visit_stamp);
	memset(set, 0, sizeof(*set));
}

namespace {


/* Advance a set's visit stamp; wrap-safe (reset on overflow). */
static unsigned next_stamp(qnn_carve_set_t *set)
{
	if (++set->stamp_counter == 0)
	{
		memset(set->visit_stamp, 0,
			(size_t)set->face_count * sizeof(unsigned));
		set->stamp_counter = 1;
	}
	return set->stamp_counter;
}

} /* namespace */


namespace {

struct RayQuery
{
	float lo[3];           /* model-local ray origin */
	const float *dir;      /* unit direction (shared across instances) */
	float best;            /* current best hit distance (== ray t) */
	float best_n[3];
	float best_p[3];       /* model-local; translated to world on accept */
	int   hit;
	int   best_inst;
	int   best_fi;
};

/* Exact ray-vs-face intersection in model-local space.  Front faces only:
 * carved normals point into the open region, so a ray enters a face when
 * normal·dir < 0.  A plane through the origin (perp ≈ 0) hits at t = 0 —
 * the apex-contact case needs no special path here. */
static void ray_test_face(RayQuery *q, const qnn_carve_set_t *set,
	int inst, int fi)
{
	const qnn_carve_face_t *f = &set->faces[fi];
	float denom = f->normal[0] * q->dir[0] + f->normal[1] * q->dir[1]
		+ f->normal[2] * q->dir[2];
	if (denom >= -1e-9f)
		return; /* back-facing or parallel */

	float perp = f->normal[0] * q->lo[0] + f->normal[1] * q->lo[1]
		+ f->normal[2] * q->lo[2] - f->dist;
	if (perp < -QUERY_BEHIND_EPS)
		return; /* origin behind the face — never constrains */

	float t = perp / -denom;
	if (t < 0.0f)
		t = 0.0f; /* contact-band origin: hit at the apex */
	if (t > q->best ||
	    (t == q->best &&
	     !(q->hit && inst == q->best_inst && fi < q->best_fi)))
		return;

	float p[3] = { q->lo[0] + t * q->dir[0], q->lo[1] + t * q->dir[1],
		q->lo[2] + t * q->dir[2] };
	const float pad = 1e-2f;
	if (p[0] < f->mins[0] - pad || p[0] > f->maxs[0] + pad ||
	    p[1] < f->mins[1] - pad || p[1] > f->maxs[1] + pad ||
	    p[2] < f->mins[2] - pad || p[2] > f->maxs[2] + pad)
		return;

	/* inside test: hit point on the same side of every edge (winding is
	 * convex, wound consistently with the emitted normal) */
	for (int vi = 0; vi < f->vert_count; vi++)
	{
		const float *a = set->verts + 3 * (f->first_vert + vi);
		const float *b = set->verts + 3 * (f->first_vert
			+ (vi + 1) % f->vert_count);
		float ex = b[0] - a[0], ey = b[1] - a[1], ez = b[2] - a[2];
		float qx = p[0] - a[0], qy = p[1] - a[1], qz = p[2] - a[2];
		float cx = ey * qz - ez * qy;
		float cy = ez * qx - ex * qz;
		float cz = ex * qy - ey * qx;
		if (cx * f->normal[0] + cy * f->normal[1] + cz * f->normal[2]
		    < -1e-3f)
			return;
	}

	q->best = t;
	q->best_n[0] = f->normal[0];
	q->best_n[1] = f->normal[1];
	q->best_n[2] = f->normal[2];
	q->best_p[0] = p[0];
	q->best_p[1] = p[1];
	q->best_p[2] = p[2];
	q->hit = 1;
	q->best_inst = inst;
	q->best_fi = fi;
}

} /* namespace */

extern "C" int QNN_CarveQueryRay(const qnn_carve_instance_t *insts, int n_insts,
	const float origin[3], const float dir[3], float max_dist,
	float out_normal[3], float out_point[3])
{
	RayQuery q;
	q.dir = dir;
	q.best = max_dist;
	q.best_n[0] = q.best_n[1] = q.best_n[2] = 0.0f;
	q.best_p[0] = q.best_p[1] = q.best_p[2] = 0.0f;
	q.hit = 0;
	q.best_inst = -1;
	q.best_fi = -1;

	float best_org[3] = { 0, 0, 0 };

	for (int s = 0; s < n_insts; s++)
	{
		qnn_carve_set_t *set = insts[s].set;
		const float *org = insts[s].origin;
		if (set == NULL)
			continue;

		q.lo[0] = origin[0] - org[0];
		q.lo[1] = origin[1] - org[1];
		q.lo[2] = origin[2] - org[2];

		if (set->cell_start == NULL)
		{
			for (int fi = 0; fi < set->face_count; fi++)
				ray_test_face(&q, set, s, fi);
		}
		else
		{
			/* Clip the ray's XY projection to the grid box; DDA-walk
			 * the cells in increasing t.  Faces occupy every cell
			 * their AABB touches, so pruning when the next cell's
			 * entry distance exceeds the running best is exact. */
			float gx0 = set->grid_org[0], gy0 = set->grid_org[1];
			float gx1 = gx0 + set->grid_nx * set->grid_cell;
			float gy1 = gy0 + set->grid_ny * set->grid_cell;
			float t0 = 0.0f, t1 = q.best;
			int degenerate = 0;

			for (int ax = 0; ax < 2 && !degenerate; ax++)
			{
				float d = dir[ax], o = q.lo[ax];
				float g0 = ax == 0 ? gx0 : gy0;
				float g1 = ax == 0 ? gx1 : gy1;
				if (fabsf(d) < 1e-9f)
				{
					if (o < g0 || o >= g1)
						degenerate = 2; /* never inside */
				}
				else
				{
					float ta = (g0 - o) / d, tb = (g1 - o) / d;
					if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
					if (ta > t0) t0 = ta;
					if (tb < t1) t1 = tb;
				}
			}
			if (degenerate == 2 || t0 > t1)
				continue;

			float ex = q.lo[0] + t0 * dir[0], ey = q.lo[1] + t0 * dir[1];
			int cx = (int)floorf((ex - gx0) / set->grid_cell);
			int cy = (int)floorf((ey - gy0) / set->grid_cell);
			if (cx < 0) cx = 0;
			if (cx >= set->grid_nx) cx = set->grid_nx - 1;
			if (cy < 0) cy = 0;
			if (cy >= set->grid_ny) cy = set->grid_ny - 1;

			int step_x = dir[0] > 1e-9f ? 1 : (dir[0] < -1e-9f ? -1 : 0);
			int step_y = dir[1] > 1e-9f ? 1 : (dir[1] < -1e-9f ? -1 : 0);
			float huge = 1e30f;
			float t_dx = step_x != 0 ? set->grid_cell / fabsf(dir[0]) : huge;
			float t_dy = step_y != 0 ? set->grid_cell / fabsf(dir[1]) : huge;
			float t_mx = huge, t_my = huge;
			if (step_x > 0)
				t_mx = t0 + (gx0 + (cx + 1) * set->grid_cell - ex) / dir[0];
			else if (step_x < 0)
				t_mx = t0 + (gx0 + cx * set->grid_cell - ex) / dir[0];
			if (step_y > 0)
				t_my = t0 + (gy0 + (cy + 1) * set->grid_cell - ey) / dir[1];
			else if (step_y < 0)
				t_my = t0 + (gy0 + cy * set->grid_cell - ey) / dir[1];

			unsigned stamp = next_stamp(set);
			for (;;)
			{
				int c = cy * set->grid_nx + cx;
				for (int k = set->cell_start[c]; k < set->cell_start[c + 1]; k++)
				{
					int fi = set->cell_faces[k];
					if (set->visit_stamp[fi] == stamp)
						continue;
					set->visit_stamp[fi] = stamp;
					ray_test_face(&q, set, s, fi);
				}
				float t_next = t_mx < t_my ? t_mx : t_my;
				if (t_next > q.best || t_next > t1)
					break;
				if (t_mx < t_my)
				{
					t_mx += t_dx;
					cx += step_x;
					if (cx < 0 || cx >= set->grid_nx)
						break;
				}
				else
				{
					t_my += t_dy;
					cy += step_y;
					if (cy < 0 || cy >= set->grid_ny)
						break;
				}
			}
		}

		if (q.hit && q.best_inst == s)
		{
			best_org[0] = org[0];
			best_org[1] = org[1];
			best_org[2] = org[2];
		}
	}

	if (q.hit)
	{
		out_normal[0] = q.best_n[0];
		out_normal[1] = q.best_n[1];
		out_normal[2] = q.best_n[2];
		out_point[0] = q.best_p[0] + best_org[0];
		out_point[1] = q.best_p[1] + best_org[1];
		out_point[2] = q.best_p[2] + best_org[2];
	}
	return q.hit;
}
