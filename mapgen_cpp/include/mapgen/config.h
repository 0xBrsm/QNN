#pragma once

#include <string>
#include <vector>

namespace mapgen {

struct ArenaConfig {
    int size = 3072;
    int max_depth = 3;
    int floor_thickness = 16;
    int shell_thickness = 32;
};

struct RoomConfig {
    int min_size = 256;
    int max_size = 1024;
    int corridor_width = 96;
    int floor_elevation_min = -128;
    int floor_elevation_max = 384;
    int floor_elevation_step = 16;
    std::vector<int> ceiling_heights = {128, 160, 192, 224, 256, 320};
    int corridor_headroom = 112;
};

struct StairConfig {
    int height = 8;
    int max_rise = 96;
    int min_step_run = 16;
};

struct PoolConfig {
    int depth = 48;
    int border = 64;
    int wall_thickness = 16;
    int min_room_size = 320;
    int min_pool_size = 96;
    int max_pool_size = 256;
    float lava_chance = 0.15f;
    float water_chance = 0.50f;
    float slime_chance = 0.55f;
    int min_guaranteed = 2;
};

struct EntityConfig {
    int item_z_above_floor = 32;
    int margin = 48;
    int health_per_room_min = 1;
    int health_per_room_max = 3;
    int health_total_min = 10;
    int health_total_max = 20;
    float quad_chance = 0.60f;
    float pent_ring_chance = 0.25f;
    int second_rl_min_rooms = 7;
    int armor_extra_rooms_5 = 5;
    int armor_extra_rooms_7 = 7;
};

struct NavmeshConfig {
    float cell_size = 16.0f;
    float cell_height = 8.0f;
    float walkable_slope_angle = 45.0f;
    float walkable_height = 56.0f;
    float walkable_climb = 18.0f;
    float walkable_radius = 16.0f;
    float max_edge_len = 192.0f;
    float max_simplification_error = 1.3f;
    int min_region_size = 8;
    int merge_region_size = 20;
    int max_verts_per_poly = 6;
    float detail_sample_distance = 6.0f;
    float detail_sample_max_error = 1.0f;
};

struct TextureConfig {
    std::string floor = "tech01_1";
    std::string ceiling = "tech02_1";
    std::string shell = "tech04_1";
    std::string fill = "metal1_1";
    std::string lava = "*lava1";
    std::string water = "*04water1";
    std::string slime = "*slime";
    std::string wad_name = "mapgen_textures.wad";
};

struct CompileConfig {
    bool enabled = true;
    std::string tools_dir;
    std::vector<std::string> qbsp_flags = {"-splitturb", "-splitspecial"};
    bool run_vis = true;
    bool run_light = true;
};

struct GenerationConfig {
    int max_attempts = 10;
};

struct MapgenConfig {
    ArenaConfig arena;
    RoomConfig rooms;
    StairConfig stairs;
    PoolConfig pools;
    EntityConfig entities;
    NavmeshConfig navmesh;
    TextureConfig textures;
    CompileConfig compile;
    GenerationConfig generation;
};

MapgenConfig load_config(const std::string& path);
MapgenConfig default_config();

}  // namespace mapgen
