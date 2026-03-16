#include "mapgen/layout.h"

#include <algorithm>
#include <cmath>

namespace mapgen {

// Helpers for random int in [lo, hi] inclusive.
static int rand_int(std::mt19937& rng, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}

static float rand_float(std::mt19937& rng) {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    return d(rng);
}

// Recursive BSP subdivision.
static void subdivide(std::mt19937& rng,
                      int x0, int y0, int x1, int y1,
                      int depth, const MapgenConfig& cfg,
                      std::vector<Room>& rooms,
                      std::vector<Split>& splits) {
    int w = x1 - x0;
    int h = y1 - y0;
    int cw = cfg.rooms.corridor_width;
    int min_sz = cfg.rooms.min_size;
    int max_sz = cfg.rooms.max_size;

    bool can_split_x = w >= min_sz * 2 + cw;
    bool can_split_y = h >= min_sz * 2 + cw;
    bool oversized = w > max_sz || h > max_sz;
    bool at_limit = depth >= cfg.arena.max_depth;

    if ((at_limit && !oversized) || (!can_split_x && !can_split_y)) {
        int n_steps = (cfg.rooms.floor_elevation_max - cfg.rooms.floor_elevation_min)
                      / cfg.rooms.floor_elevation_step;
        int floor_z = cfg.rooms.floor_elevation_min
                      + rand_int(rng, 0, n_steps) * cfg.rooms.floor_elevation_step;
        int ceil_h = cfg.rooms.ceiling_heights[
            rand_int(rng, 0, static_cast<int>(cfg.rooms.ceiling_heights.size()) - 1)];
        rooms.push_back(Room{x0, y0, x1, y1, floor_z, floor_z + ceil_h});
        return;
    }

    bool split_x;
    if (can_split_x && can_split_y)
        split_x = rand_float(rng) < (w >= h ? 0.7f : 0.3f);
    else
        split_x = can_split_x;

    if (split_x) {
        int lo = x0 + min_sz;
        int hi = x1 - min_sz - cw;
        int mid = (lo + hi) / 2;
        int quarter = std::max(1, (hi - lo) / 4);
        int split = rand_int(rng, std::max(lo, mid - quarter), std::min(hi, mid + quarter));
        subdivide(rng, x0, y0, split, y1, depth + 1, cfg, rooms, splits);
        subdivide(rng, split + cw, y0, x1, y1, depth + 1, cfg, rooms, splits);
        splits.push_back(Split{'x', split, y0, y1});
    } else {
        int lo = y0 + min_sz;
        int hi = y1 - min_sz - cw;
        int mid = (lo + hi) / 2;
        int quarter = std::max(1, (hi - lo) / 4);
        int split = rand_int(rng, std::max(lo, mid - quarter), std::min(hi, mid + quarter));
        subdivide(rng, x0, y0, x1, split, depth + 1, cfg, rooms, splits);
        subdivide(rng, x0, split + cw, x1, y1, depth + 1, cfg, rooms, splits);
        splits.push_back(Split{'y', split, x0, x1});
    }
}

// Check if two rooms share a wall separated by corridor_width gap.
static char rooms_share_wall(const Room& a, const Room& b, int cw) {
    if (b.x0 - a.x1 == cw) {
        int ov_y0 = std::max(a.y0, b.y0);
        int ov_y1 = std::min(a.y1, b.y1);
        if (ov_y1 - ov_y0 >= cw) return 'x';
    }
    if (b.y0 - a.y1 == cw) {
        int ov_x0 = std::max(a.x0, b.x0);
        int ov_x1 = std::min(a.x1, b.x1);
        if (ov_x1 - ov_x0 >= cw) return 'y';
    }
    return 0;
}

static std::vector<Corridor> connect_rooms(std::mt19937& rng,
                                           std::vector<Room>& rooms,
                                           const MapgenConfig& cfg) {
    std::vector<Corridor> corridors;
    int n = static_cast<int>(rooms.size());
    int cw = cfg.rooms.corridor_width;
    int headroom = cfg.rooms.corridor_headroom;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            char axis = rooms_share_wall(rooms[i], rooms[j], cw);
            if (!axis) axis = rooms_share_wall(rooms[j], rooms[i], cw);
            if (!axis) continue;

            int i_idx = i, j_idx = j;
            const Room* a = &rooms[i];
            const Room* b = &rooms[j];

            if (axis == 'x' && b->x1 < a->x0) {
                std::swap(a, b);
                std::swap(i_idx, j_idx);
            } else if (axis == 'y' && b->y1 < a->y0) {
                std::swap(a, b);
                std::swap(i_idx, j_idx);
            }

            int low_z = std::min(a->z0, b->z0);
            int high_z = std::max(a->z0, b->z0);
            // constrain_elevations will raise ceilings if needed.
            int ceil_z = std::min({high_z + headroom, a->z1, b->z1});

            Corridor c;
            c.room_a = i_idx;
            c.room_b = j_idx;
            c.z0 = low_z;
            c.z1 = ceil_z;
            c.z0_a = rooms[i_idx].z0;
            c.z0_b = rooms[j_idx].z0;
            c.axis = axis;

            if (axis == 'x') {
                int ov_y0 = std::max(a->y0, b->y0);
                int ov_y1 = std::min(a->y1, b->y1);
                int cy = rand_int(rng, ov_y0 + cw / 2, ov_y1 - cw / 2);
                c.x0 = a->x1;
                c.y0 = cy - cw / 2;
                c.x1 = b->x0;
                c.y1 = cy + cw / 2;
            } else {
                int ov_x0 = std::max(a->x0, b->x0);
                int ov_x1 = std::min(a->x1, b->x1);
                int cx = rand_int(rng, ov_x0 + cw / 2, ov_x1 - cw / 2);
                c.x0 = cx - cw / 2;
                c.y0 = a->y1;
                c.x1 = cx + cw / 2;
                c.y1 = b->y0;
            }

            corridors.push_back(c);
        }
    }
    return corridors;
}

static std::unordered_map<int, Pool> generate_pools(
    std::mt19937& rng, const std::vector<Room>& rooms, const MapgenConfig& cfg) {
    std::unordered_map<int, Pool> pools;
    const auto& pc = cfg.pools;
    const auto& tex = cfg.textures;

    for (int i = 0; i < static_cast<int>(rooms.size()); ++i) {
        const auto& room = rooms[i];
        if (room.width() < pc.min_room_size || room.height() < pc.min_room_size)
            continue;

        float roll = rand_float(rng);
        bool force = static_cast<int>(pools.size()) < pc.min_guaranteed;

        std::string pool_tex;
        if (roll < pc.lava_chance)
            pool_tex = tex.lava;
        else if (roll < pc.water_chance)
            pool_tex = tex.water;
        else if (roll < pc.slime_chance)
            pool_tex = tex.slime;
        else if (force)
            pool_tex = tex.water;  // guarantee met: default to water
        else
            continue;

        int max_pool_x = std::min(pc.max_pool_size, room.width() - pc.border * 2);
        int max_pool_y = std::min(pc.max_pool_size, room.height() - pc.border * 2);
        if (max_pool_x < pc.min_pool_size || max_pool_y < pc.min_pool_size)
            continue;

        int pw = rand_int(rng, pc.min_pool_size, max_pool_x);
        int ph = rand_int(rng, pc.min_pool_size, max_pool_y);
        int px = rand_int(rng, room.x0 + pc.border, room.x1 - pc.border - pw);
        int py = rand_int(rng, room.y0 + pc.border, room.y1 - pc.border - ph);

        pools[i] = Pool{px, py, px + pw, py + ph, pool_tex};
    }
    return pools;
}

static void constrain_elevations(std::vector<Room>& rooms,
                                 std::vector<Corridor>& corridors,
                                 const MapgenConfig& cfg) {
    int max_iters = static_cast<int>(rooms.size()) * 4;
    // Cap max rise to what's actually climbable: corridor_width / min_step_run
    // gives max steps, times walkable_climb gives max bridgeable height.
    int max_steps_per_corridor = cfg.rooms.corridor_width / cfg.stairs.min_step_run;
    int max_climbable = max_steps_per_corridor * static_cast<int>(cfg.navmesh.walkable_climb);
    int max_rise = std::min(cfg.stairs.max_rise, max_climbable);
    int headroom_cfg = cfg.rooms.corridor_headroom;

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (auto& c : corridors) {
            auto& a = rooms[c.room_a];
            auto& b = rooms[c.room_b];
            int dz = a.z0 - b.z0;
            if (std::abs(dz) <= max_rise) continue;

            int excess = std::abs(dz) - max_rise;
            int half = excess / 2;
            int rest = excess - half;
            if (dz > 0) {
                int ha = a.z1 - a.z0;
                a.z0 -= half;
                a.z1 = a.z0 + ha;
                int hb = b.z1 - b.z0;
                b.z0 += rest;
                b.z1 = b.z0 + hb;
            } else {
                int ha = a.z1 - a.z0;
                a.z0 += half;
                a.z1 = a.z0 + ha;
                int hb = b.z1 - b.z0;
                b.z0 -= rest;
                b.z1 = b.z0 + hb;
            }
            changed = true;
        }
        if (!changed) break;
    }

    // Final enforcement: if any corridor still exceeds max_rise after
    // iterative relaxation, clamp the higher room down.
    for (auto& c : corridors) {
        auto& a = rooms[c.room_a];
        auto& b = rooms[c.room_b];
        int dz = std::abs(a.z0 - b.z0);
        if (dz > max_rise) {
            Room& high = (a.z0 > b.z0) ? a : b;
            Room& low = (a.z0 > b.z0) ? b : a;
            int headroom = high.z1 - high.z0;
            high.z0 = low.z0 + max_rise;
            high.z1 = high.z0 + headroom;
        }
    }

    // Raise room ceilings where needed so corridors have CORRIDOR_HEADROOM
    // at the higher end.
    for (auto& c : corridors) {
        auto& a = rooms[c.room_a];
        auto& b = rooms[c.room_b];
        int high_z = std::max(a.z0, b.z0);
        int min_ceiling = high_z + headroom_cfg;
        if (a.z1 < min_ceiling) a.z1 = min_ceiling;
        if (b.z1 < min_ceiling) b.z1 = min_ceiling;
    }

    for (auto& c : corridors) {
        auto& a = rooms[c.room_a];
        auto& b = rooms[c.room_b];
        c.z0_a = a.z0;
        c.z0_b = b.z0;
        c.z0 = std::min(a.z0, b.z0);
        int high_z = std::max(a.z0, b.z0);
        c.z1 = std::min({high_z + headroom_cfg, a.z1, b.z1});
    }
}

Layout generate_layout(std::mt19937& rng, const MapgenConfig& cfg) {
    Layout layout;
    int half = cfg.arena.size / 2;

    subdivide(rng, -half, -half, half, half, 0, cfg, layout.rooms, layout.splits);
    layout.corridors = connect_rooms(rng, layout.rooms, cfg);
    constrain_elevations(layout.rooms, layout.corridors, cfg);
    layout.pools = generate_pools(rng, layout.rooms, cfg);

    return layout;
}

}  // namespace mapgen
