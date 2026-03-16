#include "mapgen/config.h"

#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace mapgen {

template <typename T>
static T get_or(const json& j, const char* key, const T& def) {
    if (j.contains(key)) return j[key].get<T>();
    return def;
}

MapgenConfig default_config() { return MapgenConfig{}; }

MapgenConfig load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config file: " + path);
    json j = json::parse(f);
    MapgenConfig c;

    if (j.contains("arena")) {
        auto& a = j["arena"];
        c.arena.size = get_or(a, "size", c.arena.size);
        c.arena.max_depth = get_or(a, "max_depth", c.arena.max_depth);
        c.arena.floor_thickness = get_or(a, "floor_thickness", c.arena.floor_thickness);
        c.arena.shell_thickness = get_or(a, "shell_thickness", c.arena.shell_thickness);
    }

    if (j.contains("rooms")) {
        auto& r = j["rooms"];
        c.rooms.min_size = get_or(r, "min_size", c.rooms.min_size);
        c.rooms.max_size = get_or(r, "max_size", c.rooms.max_size);
        c.rooms.corridor_width = get_or(r, "corridor_width", c.rooms.corridor_width);
        c.rooms.floor_elevation_min = get_or(r, "floor_elevation_min", c.rooms.floor_elevation_min);
        c.rooms.floor_elevation_max = get_or(r, "floor_elevation_max", c.rooms.floor_elevation_max);
        c.rooms.floor_elevation_step = get_or(r, "floor_elevation_step", c.rooms.floor_elevation_step);
        if (r.contains("ceiling_heights"))
            c.rooms.ceiling_heights = r["ceiling_heights"].get<std::vector<int>>();
        c.rooms.corridor_headroom = get_or(r, "corridor_headroom", c.rooms.corridor_headroom);
    }

    if (j.contains("stairs")) {
        auto& s = j["stairs"];
        c.stairs.height = get_or(s, "height", c.stairs.height);
        c.stairs.max_rise = get_or(s, "max_rise", c.stairs.max_rise);
        c.stairs.min_step_run = get_or(s, "min_step_run", c.stairs.min_step_run);
    }

    if (j.contains("pools")) {
        auto& p = j["pools"];
        c.pools.depth = get_or(p, "depth", c.pools.depth);
        c.pools.border = get_or(p, "border", c.pools.border);
        c.pools.wall_thickness = get_or(p, "wall_thickness", c.pools.wall_thickness);
        c.pools.min_room_size = get_or(p, "min_room_size", c.pools.min_room_size);
        c.pools.min_pool_size = get_or(p, "min_pool_size", c.pools.min_pool_size);
        c.pools.max_pool_size = get_or(p, "max_pool_size", c.pools.max_pool_size);
        c.pools.lava_chance = get_or(p, "lava_chance", c.pools.lava_chance);
        c.pools.water_chance = get_or(p, "water_chance", c.pools.water_chance);
        c.pools.slime_chance = get_or(p, "slime_chance", c.pools.slime_chance);
        c.pools.min_guaranteed = get_or(p, "min_guaranteed", c.pools.min_guaranteed);
    }

    if (j.contains("entities")) {
        auto& e = j["entities"];
        c.entities.item_z_above_floor = get_or(e, "item_z_above_floor", c.entities.item_z_above_floor);
        c.entities.margin = get_or(e, "margin", c.entities.margin);
        c.entities.health_per_room_min = get_or(e, "health_per_room_min", c.entities.health_per_room_min);
        c.entities.health_per_room_max = get_or(e, "health_per_room_max", c.entities.health_per_room_max);
        c.entities.health_total_min = get_or(e, "health_total_min", c.entities.health_total_min);
        c.entities.health_total_max = get_or(e, "health_total_max", c.entities.health_total_max);
        c.entities.quad_chance = get_or(e, "quad_chance", c.entities.quad_chance);
        c.entities.pent_ring_chance = get_or(e, "pent_ring_chance", c.entities.pent_ring_chance);
        c.entities.second_rl_min_rooms = get_or(e, "second_rl_min_rooms", c.entities.second_rl_min_rooms);
        c.entities.armor_extra_rooms_5 = get_or(e, "armor_extra_rooms_5", c.entities.armor_extra_rooms_5);
        c.entities.armor_extra_rooms_7 = get_or(e, "armor_extra_rooms_7", c.entities.armor_extra_rooms_7);
    }

    if (j.contains("navmesh")) {
        auto& n = j["navmesh"];
        c.navmesh.cell_size = get_or(n, "cell_size", c.navmesh.cell_size);
        c.navmesh.cell_height = get_or(n, "cell_height", c.navmesh.cell_height);
        c.navmesh.walkable_slope_angle = get_or(n, "walkable_slope_angle", c.navmesh.walkable_slope_angle);
        c.navmesh.walkable_height = get_or(n, "walkable_height", c.navmesh.walkable_height);
        c.navmesh.walkable_climb = get_or(n, "walkable_climb", c.navmesh.walkable_climb);
        c.navmesh.walkable_radius = get_or(n, "walkable_radius", c.navmesh.walkable_radius);
        c.navmesh.max_edge_len = get_or(n, "max_edge_len", c.navmesh.max_edge_len);
        c.navmesh.max_simplification_error = get_or(n, "max_simplification_error", c.navmesh.max_simplification_error);
        c.navmesh.min_region_size = get_or(n, "min_region_size", c.navmesh.min_region_size);
        c.navmesh.merge_region_size = get_or(n, "merge_region_size", c.navmesh.merge_region_size);
        c.navmesh.max_verts_per_poly = get_or(n, "max_verts_per_poly", c.navmesh.max_verts_per_poly);
        c.navmesh.detail_sample_distance = get_or(n, "detail_sample_distance", c.navmesh.detail_sample_distance);
        c.navmesh.detail_sample_max_error = get_or(n, "detail_sample_max_error", c.navmesh.detail_sample_max_error);
    }

    if (j.contains("textures")) {
        auto& t = j["textures"];
        c.textures.floor = get_or<std::string>(t, "floor", c.textures.floor);
        c.textures.ceiling = get_or<std::string>(t, "ceiling", c.textures.ceiling);
        c.textures.shell = get_or<std::string>(t, "shell", c.textures.shell);
        c.textures.fill = get_or<std::string>(t, "fill", c.textures.fill);
        c.textures.lava = get_or<std::string>(t, "lava", c.textures.lava);
        c.textures.water = get_or<std::string>(t, "water", c.textures.water);
        c.textures.slime = get_or<std::string>(t, "slime", c.textures.slime);
        c.textures.wad_name = get_or<std::string>(t, "wad_name", c.textures.wad_name);
    }

    if (j.contains("compile")) {
        auto& co = j["compile"];
        c.compile.enabled = get_or(co, "enabled", c.compile.enabled);
        c.compile.tools_dir = get_or<std::string>(co, "tools_dir", c.compile.tools_dir);
        if (co.contains("qbsp_flags"))
            c.compile.qbsp_flags = co["qbsp_flags"].get<std::vector<std::string>>();
        c.compile.run_vis = get_or(co, "run_vis", c.compile.run_vis);
        c.compile.run_light = get_or(co, "run_light", c.compile.run_light);
    }

    if (j.contains("generation")) {
        auto& g = j["generation"];
        c.generation.max_attempts = get_or(g, "max_attempts", c.generation.max_attempts);
    }

    return c;
}

}  // namespace mapgen
