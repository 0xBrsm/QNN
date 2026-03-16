#include "mapgen/brush.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace mapgen {

static constexpr int MAX_LUXELS = 16;
static constexpr int LUXEL_SIZE = 16;

static float safe_scale(int extent) {
    if (extent <= MAX_LUXELS * LUXEL_SIZE) return 1.0f;
    return std::ceil(static_cast<float>(extent) / (MAX_LUXELS * LUXEL_SIZE));
}

void Plane::write(std::ostream& out) const {
    out << "( " << p1[0] << " " << p1[1] << " " << p1[2] << " ) "
        << "( " << p2[0] << " " << p2[1] << " " << p2[2] << " ) "
        << "( " << p3[0] << " " << p3[1] << " " << p3[2] << " ) "
        << texture << " " << x_off << " " << y_off << " "
        << rotation << " " << x_scale << " " << y_scale << "\n";
}

void Brush::write(std::ostream& out) const {
    out << "{\n";
    for (auto& p : planes) p.write(out);
    out << "}\n";
}

void Entity::write(std::ostream& out) const {
    out << "{\n";
    for (auto& [k, v] : properties)
        out << "\"" << k << "\" \"" << v << "\"\n";
    for (auto& b : brushes) b.write(out);
    out << "}\n";
}

MapFile::MapFile(const TextureConfig& tex) {
    worldspawn.properties.push_back({"classname", "worldspawn"});
    worldspawn.properties.push_back({"wad", tex.wad_name});
    worldspawn.properties.push_back({"worldtype", "2"});
}

void MapFile::add_brush(Brush brush) {
    worldspawn.brushes.push_back(std::move(brush));
}

void MapFile::add_entity(const std::string& classname, int x, int y, int z,
                         const std::vector<std::pair<std::string, std::string>>& extra) {
    Entity e;
    e.properties.push_back({"classname", classname});
    std::ostringstream oss;
    oss << x << " " << y << " " << z;
    e.properties.push_back({"origin", oss.str()});
    for (auto& kv : extra)
        e.properties.push_back(kv);
    entities.push_back(std::move(e));
}

void MapFile::add_light(int x, int y, int z, int brightness) {
    add_entity("light", x, y, z, {{"light", std::to_string(brightness)}});
}

void MapFile::write(std::ostream& out) const {
    worldspawn.write(out);
    for (auto& e : entities) e.write(out);
}

Brush axis_aligned_box(int min_x, int min_y, int min_z,
                       int max_x, int max_y, int max_z,
                       const std::string& texture) {
    int dx = max_x - min_x;
    int dy = max_y - min_y;
    int dz = max_z - min_z;

    float sx_yz0 = safe_scale(dy), sx_yz1 = safe_scale(dz);
    float sy_xz0 = safe_scale(dx), sy_xz1 = safe_scale(dz);
    float sz_xy0 = safe_scale(dx), sz_xy1 = safe_scale(dy);

    Brush b;
    b.planes = {
        // -X face
        Plane{{min_x, 0, 0}, {min_x, 1, 0}, {min_x, 0, 1}, texture, 0, 0, 0.0f, sx_yz0, sx_yz1},
        // +X face
        Plane{{max_x, 0, 0}, {max_x, 0, 1}, {max_x, 1, 0}, texture, 0, 0, 0.0f, sx_yz0, sx_yz1},
        // -Y face
        Plane{{0, min_y, 0}, {0, min_y, 1}, {1, min_y, 0}, texture, 0, 0, 0.0f, sy_xz0, sy_xz1},
        // +Y face
        Plane{{0, max_y, 0}, {1, max_y, 0}, {0, max_y, 1}, texture, 0, 0, 0.0f, sy_xz0, sy_xz1},
        // -Z face (floor)
        Plane{{0, 0, min_z}, {1, 0, min_z}, {0, 1, min_z}, texture, 0, 0, 0.0f, sz_xy0, sz_xy1},
        // +Z face (ceiling)
        Plane{{0, 0, max_z}, {0, 1, max_z}, {1, 0, max_z}, texture, 0, 0, 0.0f, sz_xy0, sz_xy1},
    };
    return b;
}

// --- Geometry builders (ported from layout.py) ---

static void build_stairs(MapFile& m, const Corridor& corridor, const MapgenConfig& cfg) {
    int dz = corridor.z0_b - corridor.z0_a;
    if (dz == 0) return;

    int bridged = std::min(std::abs(dz), cfg.stairs.max_rise);
    int total_run = (corridor.axis == 'x') ? (corridor.x1 - corridor.x0) : (corridor.y1 - corridor.y0);
    int max_steps = total_run / cfg.stairs.min_step_run;
    // Ensure enough steps so each is within walkable_climb.
    int climb = static_cast<int>(cfg.navmesh.walkable_climb);
    int min_steps = (bridged + climb - 1) / climb;  // ceil(bridged / climb)
    int n_steps = std::max(min_steps, std::min(bridged / cfg.stairs.height, max_steps));
    if (n_steps == 0 || n_steps > max_steps) return;

    std::vector<int> rises(n_steps, bridged / n_steps);
    int remainder = bridged - rises[0] * n_steps;
    for (int i = 0; i < remainder; ++i) rises[i]++;

    int base_z = corridor.z0_a;
    int slab_bot = corridor.z0 - cfg.arena.floor_thickness;
    int step_run = total_run / n_steps;
    int cumulative_rise = 0;

    for (int i = 0; i < n_steps; ++i) {
        int run_lo = i * step_run;
        int run_hi = (i < n_steps - 1) ? (i + 1) * step_run : total_run;
        int slab_top;

        if (dz > 0) {
            cumulative_rise += rises[i];
            slab_top = base_z + cumulative_rise;
        } else {
            slab_top = base_z - cumulative_rise;
            cumulative_rise += rises[i];
        }

        if (corridor.axis == 'x') {
            m.add_brush(axis_aligned_box(
                corridor.x0 + run_lo, corridor.y0, slab_bot,
                corridor.x0 + run_hi, corridor.y1, slab_top,
                cfg.textures.floor));
        } else {
            m.add_brush(axis_aligned_box(
                corridor.x0, corridor.y0 + run_lo, slab_bot,
                corridor.x1, corridor.y0 + run_hi, slab_top,
                cfg.textures.floor));
        }
    }
}

static void build_threshold(MapFile& m, const Corridor& corridor,
                            const std::vector<Room>& rooms, const MapgenConfig& cfg) {
    const auto& c = corridor;
    // Skip thresholds when stairs handle the elevation change — emitting both
    // creates an overlapping solid that blocks the stair run.
    if (c.z0_a != c.z0_b) return;

    struct Side { int room_idx; int room_z0; };
    Side sides[] = {{c.room_a, c.z0_a}, {c.room_b, c.z0_b}};

    for (auto& s : sides) {
        if (s.room_z0 <= c.z0) continue;
        if (c.axis == 'x') {
            int mid_x = (c.x0 + c.x1) / 2;
            int tx0 = (s.room_idx == c.room_a) ? c.x0 : mid_x;
            int tx1 = (s.room_idx == c.room_a) ? mid_x : c.x1;
            m.add_brush(axis_aligned_box(tx0, c.y0, c.z0, tx1, c.y1, s.room_z0, cfg.textures.floor));
        } else {
            int mid_y = (c.y0 + c.y1) / 2;
            int ty0 = (s.room_idx == c.room_a) ? c.y0 : mid_y;
            int ty1 = (s.room_idx == c.room_a) ? mid_y : c.y1;
            m.add_brush(axis_aligned_box(c.x0, ty0, c.z0, c.x1, ty1, s.room_z0, cfg.textures.floor));
        }
    }
}

static void fill_gap_x(MapFile& m, const Split& split,
                       const std::vector<const Corridor*>& corridors,
                       int z_lo, int z_hi, const std::string& tex, int cw) {
    int gx0 = split.pos;
    int gx1 = split.pos + cw;

    struct Opening { int y0, y1, z0, z1; };
    std::vector<Opening> openings;
    for (auto* c : corridors)
        openings.push_back({c->y0, c->y1, c->z0, c->z1});
    std::sort(openings.begin(), openings.end(), [](auto& a, auto& b) { return a.y0 < b.y0; });

    int cursor = split.perp_lo;
    for (auto& o : openings) {
        if (o.y0 > cursor)
            m.add_brush(axis_aligned_box(gx0, cursor, z_lo, gx1, o.y0, z_hi, tex));
        if (o.z0 > z_lo)
            m.add_brush(axis_aligned_box(gx0, o.y0, z_lo, gx1, o.y1, o.z0, tex));
        if (o.z1 < z_hi)
            m.add_brush(axis_aligned_box(gx0, o.y0, o.z1, gx1, o.y1, z_hi, tex));
        cursor = o.y1;
    }
    if (cursor < split.perp_hi)
        m.add_brush(axis_aligned_box(gx0, cursor, z_lo, gx1, split.perp_hi, z_hi, tex));
}

static void fill_gap_y(MapFile& m, const Split& split,
                       const std::vector<const Corridor*>& corridors,
                       int z_lo, int z_hi, const std::string& tex, int cw) {
    int gy0 = split.pos;
    int gy1 = split.pos + cw;

    struct Opening { int x0, x1, z0, z1; };
    std::vector<Opening> openings;
    for (auto* c : corridors)
        openings.push_back({c->x0, c->x1, c->z0, c->z1});
    std::sort(openings.begin(), openings.end(), [](auto& a, auto& b) { return a.x0 < b.x0; });

    int cursor = split.perp_lo;
    for (auto& o : openings) {
        if (o.x0 > cursor)
            m.add_brush(axis_aligned_box(cursor, gy0, z_lo, o.x0, gy1, z_hi, tex));
        if (o.z0 > z_lo)
            m.add_brush(axis_aligned_box(o.x0, gy0, z_lo, o.x1, gy1, o.z0, tex));
        if (o.z1 < z_hi)
            m.add_brush(axis_aligned_box(o.x0, gy0, o.z1, o.x1, gy1, z_hi, tex));
        cursor = o.x1;
    }
    if (cursor < split.perp_hi)
        m.add_brush(axis_aligned_box(cursor, gy0, z_lo, split.perp_hi, gy1, z_hi, tex));
}

static void build_gap_fills(MapFile& m, const Layout& layout,
                            int z_lo, int z_hi, const MapgenConfig& cfg) {
    int cw = cfg.rooms.corridor_width;
    for (auto& split : layout.splits) {
        std::vector<const Corridor*> gap_corrs;
        for (auto& c : layout.corridors) {
            if (split.axis == 'x' && c.axis == 'x' && c.x0 == split.pos)
                gap_corrs.push_back(&c);
            else if (split.axis == 'y' && c.axis == 'y' && c.y0 == split.pos)
                gap_corrs.push_back(&c);
        }
        if (split.axis == 'x')
            fill_gap_x(m, split, gap_corrs, z_lo, z_hi, cfg.textures.fill, cw);
        else
            fill_gap_y(m, split, gap_corrs, z_lo, z_hi, cfg.textures.fill, cw);
    }
}

static void build_outer_shell(MapFile& m, const Layout& layout,
                              int z_lo, int z_hi, const MapgenConfig& cfg) {
    int min_x = layout.rooms[0].x0, max_x = layout.rooms[0].x1;
    int min_y = layout.rooms[0].y0, max_y = layout.rooms[0].y1;
    for (auto& r : layout.rooms) {
        min_x = std::min(min_x, r.x0); max_x = std::max(max_x, r.x1);
        min_y = std::min(min_y, r.y0); max_y = std::max(max_y, r.y1);
    }

    int s = cfg.arena.shell_thickness;
    const auto& tex = cfg.textures.shell;

    m.add_brush(axis_aligned_box(min_x - s, min_y - s, z_lo - s, max_x + s, max_y + s, z_lo, tex));
    m.add_brush(axis_aligned_box(min_x - s, min_y - s, z_hi, max_x + s, max_y + s, z_hi + s, tex));
    m.add_brush(axis_aligned_box(min_x - s, min_y - s, z_lo, min_x, max_y + s, z_hi, tex));
    m.add_brush(axis_aligned_box(max_x, min_y - s, z_lo, max_x + s, max_y + s, z_hi, tex));
    m.add_brush(axis_aligned_box(min_x, min_y - s, z_lo, max_x, min_y, z_hi, tex));
    m.add_brush(axis_aligned_box(min_x, max_y, z_lo, max_x, max_y + s, z_hi, tex));
}

static void build_room_brushes(MapFile& m, const Room& room,
                               const Pool* pool, const MapgenConfig& cfg) {
    int x0 = room.x0, y0 = room.y0, x1 = room.x1, y1 = room.y1;
    int z0 = room.z0, z1 = room.z1;
    int ft = cfg.arena.floor_thickness;

    m.add_brush(axis_aligned_box(x0, y0, z1, x1, y1, z1 + ft, cfg.textures.ceiling));

    if (!pool) {
        m.add_brush(axis_aligned_box(x0, y0, z0 - ft, x1, y1, z0, cfg.textures.floor));
        return;
    }

    const auto& p = *pool;
    if (p.y0 > y0)
        m.add_brush(axis_aligned_box(x0, y0, z0 - ft, x1, p.y0, z0, cfg.textures.floor));
    if (p.y1 < y1)
        m.add_brush(axis_aligned_box(x0, p.y1, z0 - ft, x1, y1, z0, cfg.textures.floor));
    if (p.x0 > x0)
        m.add_brush(axis_aligned_box(x0, p.y0, z0 - ft, p.x0, p.y1, z0, cfg.textures.floor));
    if (p.x1 < x1)
        m.add_brush(axis_aligned_box(p.x1, p.y0, z0 - ft, x1, p.y1, z0, cfg.textures.floor));

    int pit_bot = z0 - cfg.pools.depth;
    int pw = cfg.pools.wall_thickness;
    m.add_brush(axis_aligned_box(p.x0, p.y0, pit_bot - ft, p.x1, p.y1, pit_bot, cfg.textures.floor));
    m.add_brush(axis_aligned_box(p.x0 - pw, p.y0 - pw, pit_bot, p.x0, p.y1 + pw, z0, cfg.textures.floor));
    m.add_brush(axis_aligned_box(p.x1, p.y0 - pw, pit_bot, p.x1 + pw, p.y1 + pw, z0, cfg.textures.floor));
    m.add_brush(axis_aligned_box(p.x0, p.y0 - pw, pit_bot, p.x1, p.y0, z0, cfg.textures.floor));
    m.add_brush(axis_aligned_box(p.x0, p.y1, pit_bot, p.x1, p.y1 + pw, z0, cfg.textures.floor));
    m.add_brush(axis_aligned_box(p.x0, p.y0, pit_bot, p.x1, p.y1, z0, p.texture));
}

static void build_corridor_brushes(MapFile& m, const Corridor& c,
                                   const std::vector<Room>& rooms,
                                   const MapgenConfig& cfg) {
    int ft = cfg.arena.floor_thickness;
    m.add_brush(axis_aligned_box(c.x0, c.y0, c.z0 - ft, c.x1, c.y1, c.z0, cfg.textures.floor));
    m.add_brush(axis_aligned_box(c.x0, c.y0, c.z1, c.x1, c.y1, c.z1 + ft, cfg.textures.ceiling));
    build_stairs(m, c, cfg);
    build_threshold(m, c, rooms, cfg);
}

void build_layout(MapFile& m, const Layout& layout, const MapgenConfig& cfg) {
    int ft = cfg.arena.floor_thickness;
    int z_lo = layout.rooms[0].z0 - ft;
    int z_hi = layout.rooms[0].z1 + ft;
    for (auto& r : layout.rooms) {
        z_lo = std::min(z_lo, r.z0 - ft);
        z_hi = std::max(z_hi, r.z1 + ft);
    }
    for (auto& c : layout.corridors) {
        z_lo = std::min(z_lo, c.z0 - ft);
        z_hi = std::max(z_hi, c.z1 + ft);
    }
    for (auto& [idx, pool] : layout.pools) {
        int pit_bot = layout.rooms[idx].z0 - cfg.pools.depth - ft;
        z_lo = std::min(z_lo, pit_bot);
    }

    build_outer_shell(m, layout, z_lo, z_hi, cfg);
    build_gap_fills(m, layout, z_lo, z_hi, cfg);

    for (int i = 0; i < static_cast<int>(layout.rooms.size()); ++i) {
        auto it = layout.pools.find(i);
        const Pool* pool = (it != layout.pools.end()) ? &it->second : nullptr;
        build_room_brushes(m, layout.rooms[i], pool, cfg);
    }

    for (auto& c : layout.corridors)
        build_corridor_brushes(m, c, layout.rooms, cfg);
}

}  // namespace mapgen
