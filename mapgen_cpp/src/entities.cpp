#include "mapgen/entities.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace mapgen {

static int rand_int(std::mt19937& rng, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}

static float rand_float(std::mt19937& rng) {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    return d(rng);
}

static int item_z(const Room& room, const EntityConfig& ec) {
    return room.z0 + ec.item_z_above_floor;
}

static bool in_pool(int x, int y, const Pool* pool) {
    if (!pool) return false;
    return x >= pool->x0 && x <= pool->x1 && y >= pool->y0 && y <= pool->y1;
}

// Check if position is inside a gap fill solid (between BSP splits).
// Gap fills span split.pos to split.pos + corridor_width on the split axis,
// except where corridors punch through.
static bool in_gap_fill(int x, int y, const Layout& layout, int cw) {
    for (auto& s : layout.splits) {
        if (s.axis == 'x') {
            if (x < s.pos || x > s.pos + cw) continue;
            bool in_opening = false;
            for (auto& c : layout.corridors) {
                if (c.axis == 'x' && c.x0 == s.pos && y >= c.y0 && y <= c.y1) {
                    in_opening = true;
                    break;
                }
            }
            if (!in_opening) return true;
        } else {
            if (y < s.pos || y > s.pos + cw) continue;
            bool in_opening = false;
            for (auto& c : layout.corridors) {
                if (c.axis == 'y' && c.y0 == s.pos && x >= c.x0 && x <= c.x1) {
                    in_opening = true;
                    break;
                }
            }
            if (!in_opening) return true;
        }
    }
    return false;
}

// Minimum spacing between placed entities (about 2 player widths).
static constexpr int MIN_ENTITY_SPACING = 64;

static bool too_close(int x, int y,
                      const std::vector<std::pair<int,int>>& placed) {
    for (auto& [px, py] : placed)
        if (std::abs(x - px) < MIN_ENTITY_SPACING &&
            std::abs(y - py) < MIN_ENTITY_SPACING)
            return true;
    return false;
}

static bool valid_pos(int x, int y, const Pool* pool,
                      const Layout& layout, int cw,
                      const std::vector<std::pair<int,int>>& placed) {
    return !in_pool(x, y, pool) &&
           !in_gap_fill(x, y, layout, cw) &&
           !too_close(x, y, placed);
}

static std::pair<int, int> safe_xy(std::mt19937& rng, const Room& room,
                                   const Pool* pool, int margin,
                                   const Layout& layout, int cw,
                                   const std::vector<std::pair<int,int>>& placed = {}) {
    for (int attempt = 0; attempt < 20; ++attempt) {
        int x = rand_int(rng, room.x0 + margin, room.x1 - margin);
        int y = rand_int(rng, room.y0 + margin, room.y1 - margin);
        if (valid_pos(x, y, pool, layout, cw, placed))
            return {x, y};
    }

    if (valid_pos(room.cx(), room.cy(), pool, layout, cw, placed))
        return {room.cx(), room.cy()};

    int corners[][2] = {
        {room.x0 + margin, room.y0 + margin},
        {room.x0 + margin, room.y1 - margin},
        {room.x1 - margin, room.y0 + margin},
        {room.x1 - margin, room.y1 - margin},
    };
    for (auto& c : corners)
        if (valid_pos(c[0], c[1], pool, layout, cw, placed))
            return {c[0], c[1]};

    // Last resort: grid scan for any valid position.
    int step = 32;
    for (int sy = room.y0 + margin; sy <= room.y1 - margin; sy += step)
        for (int sx = room.x0 + margin; sx <= room.x1 - margin; sx += step)
            if (valid_pos(sx, sy, pool, layout, cw, placed))
                return {sx, sy};

    // Absolutely no deconflicted spot — drop spacing requirement.
    for (int sy = room.y0 + margin; sy <= room.y1 - margin; sy += step)
        for (int sx = room.x0 + margin; sx <= room.x1 - margin; sx += step)
            if (!in_pool(sx, sy, pool) && !in_gap_fill(sx, sy, layout, cw))
                return {sx, sy};

    return {room.cx(), room.cy()};
}

static const Pool* get_pool(const Layout& layout, int idx) {
    auto it = layout.pools.find(idx);
    return (it != layout.pools.end()) ? &it->second : nullptr;
}

static const char* WEAPONS[] = {
    "weapon_supershotgun", "weapon_nailgun", "weapon_supernailgun",
    "weapon_grenadelauncher", "weapon_rocketlauncher", "weapon_lightning",
};

static const char* ammo_for_weapon(const char* weapon) {
    if (std::string(weapon) == "weapon_supershotgun") return "item_shells";
    if (std::string(weapon) == "weapon_nailgun") return "item_spikes";
    if (std::string(weapon) == "weapon_supernailgun") return "item_spikes";
    if (std::string(weapon) == "weapon_grenadelauncher") return "item_rockets";
    if (std::string(weapon) == "weapon_rocketlauncher") return "item_rockets";
    if (std::string(weapon) == "weapon_lightning") return "item_cells";
    return "item_shells";
}

// Shorthand: place an entity and record its position for deconfliction.
static void place(MapFile& m, const char* classname, int x, int y, int z,
                  std::vector<std::pair<int,int>>& placed,
                  const std::vector<std::pair<std::string,std::string>>& extra = {}) {
    m.add_entity(classname, x, y, z, extra);
    placed.push_back({x, y});
}

static void place_spawns(MapFile& m, const Layout& layout,
                         std::mt19937& rng, const MapgenConfig& cfg,
                         std::vector<std::pair<int,int>>& placed) {
    const auto& ec = cfg.entities;
    int cw = cfg.rooms.corridor_width;
    static const char* angles[] = {"0", "90", "180", "270"};
    for (int i = 0; i < static_cast<int>(layout.rooms.size()); ++i) {
        auto [x, y] = safe_xy(rng, layout.rooms[i], get_pool(layout, i),
                               ec.margin, layout, cw, placed);
        place(m, "info_player_deathmatch", x, y, item_z(layout.rooms[i], ec),
              placed, {{"angle", angles[rand_int(rng, 0, 3)]}});
    }
    auto [x, y] = safe_xy(rng, layout.rooms[0], get_pool(layout, 0),
                           ec.margin, layout, cw, placed);
    place(m, "info_player_start", x, y, item_z(layout.rooms[0], ec),
          placed, {{"angle", "0"}});
}

static void place_weapons(MapFile& m, const Layout& layout,
                          std::mt19937& rng, const MapgenConfig& cfg,
                          std::vector<std::pair<int,int>>& placed) {
    const auto& ec = cfg.entities;
    int cw = cfg.rooms.corridor_width;
    int n_rooms = static_cast<int>(layout.rooms.size());
    std::vector<const char*> weapons(WEAPONS, WEAPONS + 6);
    if (n_rooms >= ec.second_rl_min_rooms)
        weapons.push_back("weapon_rocketlauncher");

    std::shuffle(weapons.begin(), weapons.end(), rng);

    std::vector<int> room_order(n_rooms);
    for (int i = 0; i < n_rooms; ++i) room_order[i] = i;
    std::shuffle(room_order.begin(), room_order.end(), rng);

    for (int i = 0; i < static_cast<int>(weapons.size()); ++i) {
        int ri = room_order[i % n_rooms];
        const auto& room = layout.rooms[ri];
        const Pool* pool = get_pool(layout, ri);
        auto [x, y] = safe_xy(rng, room, pool, ec.margin, layout, cw, placed);
        place(m, weapons[i], x, y, item_z(room, ec), placed);

        const char* ammo = ammo_for_weapon(weapons[i]);
        int ammo_count = rand_int(rng, 1, 2);
        for (int a = 0; a < ammo_count; ++a) {
            int ax = std::max(room.x0 + 32, std::min(room.x1 - 32, x + rand_int(rng, -80, 80)));
            int ay = std::max(room.y0 + 32, std::min(room.y1 - 32, y + rand_int(rng, -80, 80)));
            if (!in_pool(ax, ay, pool) && !in_gap_fill(ax, ay, layout, cw))
                place(m, ammo, ax, ay, item_z(room, ec), placed);
        }
    }
}

static void place_items(MapFile& m, const Layout& layout,
                        std::mt19937& rng, const MapgenConfig& cfg,
                        std::vector<std::pair<int,int>>& placed) {
    const auto& ec = cfg.entities;
    int cw = cfg.rooms.corridor_width;
    int n_rooms = static_cast<int>(layout.rooms.size());
    int target_health = std::max(ec.health_total_min,
                        std::min(ec.health_total_max, n_rooms * 2 + rand_int(rng, 2, 4)));
    int health_placed = 0;

    for (int i = 0; i < n_rooms; ++i) {
        int count = (health_placed >= target_health) ? 1 : rand_int(rng, ec.health_per_room_min, ec.health_per_room_max);
        const Pool* pool = get_pool(layout, i);
        for (int h = 0; h < count && health_placed < target_health; ++h) {
            auto [x, y] = safe_xy(rng, layout.rooms[i], pool, 32, layout, cw, placed);
            place(m, "item_health", x, y, item_z(layout.rooms[i], ec), placed);
            ++health_placed;
        }
    }

    std::vector<const char*> armors;
    armors.push_back("item_armorInv");
    armors.push_back("item_armor2");
    armors.push_back("item_armor1");
    if (n_rooms >= ec.armor_extra_rooms_5)
        armors.push_back(rand_float(rng) < 0.5f ? "item_armor1" : "item_armor2");
    if (n_rooms >= ec.armor_extra_rooms_7)
        armors.push_back(rand_float(rng) < 0.5f ? "item_armor1" : "item_armor2");

    std::vector<int> room_order(n_rooms);
    for (int i = 0; i < n_rooms; ++i) room_order[i] = i;
    std::shuffle(room_order.begin(), room_order.end(), rng);

    for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
        int ri = room_order[i % n_rooms];
        auto [x, y] = safe_xy(rng, layout.rooms[ri], get_pool(layout, ri),
                               ec.margin, layout, cw, placed);
        place(m, armors[i], x, y, item_z(layout.rooms[ri], ec), placed);
    }
}

static void place_powerups(MapFile& m, const Layout& layout,
                           std::mt19937& rng, const MapgenConfig& cfg,
                           std::vector<std::pair<int,int>>& placed) {
    const auto& ec = cfg.entities;
    int cw = cfg.rooms.corridor_width;
    if (rand_float(rng) < ec.quad_chance) {
        int idx = rand_int(rng, 0, static_cast<int>(layout.rooms.size()) - 1);
        auto [x, y] = safe_xy(rng, layout.rooms[idx], get_pool(layout, idx),
                               ec.margin, layout, cw, placed);
        place(m, "item_artifact_super_damage", x, y, item_z(layout.rooms[idx], ec), placed);
    }
    if (rand_float(rng) < ec.pent_ring_chance) {
        const char* powerup = rand_float(rng) < 0.5f
            ? "item_artifact_invulnerability" : "item_artifact_invisibility";
        int idx = rand_int(rng, 0, static_cast<int>(layout.rooms.size()) - 1);
        auto [x, y] = safe_xy(rng, layout.rooms[idx], get_pool(layout, idx),
                               ec.margin, layout, cw, placed);
        place(m, powerup, x, y, item_z(layout.rooms[idx], ec), placed);
    }
}

static void place_lights(MapFile& m, const Layout& layout) {
    for (auto& room : layout.rooms) {
        m.add_light(room.cx(), room.cy(), room.z1 - 16, 300);
        if (room.width() > 400 && room.height() > 400) {
            int qx = room.width() / 4;
            int qy = room.height() / 4;
            int offsets[][2] = {{-qx, -qy}, {qx, -qy}, {-qx, qy}, {qx, qy}};
            for (auto& o : offsets)
                m.add_light(room.cx() + o[0], room.cy() + o[1], room.z1 - 16, 200);
        }
    }
    for (auto& c : layout.corridors) {
        int cx = (c.x0 + c.x1) / 2;
        int cy = (c.y0 + c.y1) / 2;
        m.add_light(cx, cy, c.z1 - 8, 200);
    }
}

void populate(MapFile& m, const Layout& layout, std::mt19937& rng,
              const MapgenConfig& cfg) {
    std::vector<std::pair<int,int>> placed;
    place_spawns(m, layout, rng, cfg, placed);
    place_weapons(m, layout, rng, cfg, placed);
    place_items(m, layout, rng, cfg, placed);
    place_powerups(m, layout, rng, cfg, placed);
    place_lights(m, layout);
}

}  // namespace mapgen
