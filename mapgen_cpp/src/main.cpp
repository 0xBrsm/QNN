#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "mapgen/brush.h"
#include "mapgen/compile.h"
#include "mapgen/config.h"
#include "mapgen/entities.h"
#include "mapgen/layout.h"
#include "mapgen/navcheck.h"
#include "mapgen/textures.h"

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --config <path>       JSON config file (default: built-in defaults)\n"
              << "  --seed <int>          RNG seed (default: random)\n"
              << "  --output <path>       Output .map file path\n"
              << "  --output-dir <path>   Output directory for .map and .bsp\n"
              << "  --no-compile          Skip .bsp compilation\n"
              << "  --no-navcheck         Skip connectivity validation\n"
              << "  --arena-size <int>    Override arena size\n"
              << "  --max-depth <int>     Override BSP subdivision depth\n"
              << "  --max-attempts <int>  Override max generation attempts\n"
              << "  --quiet               Suppress non-error output\n"
              << "  --help                Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string config_path;
    std::string output_path;
    std::string output_dir;
    int seed = -1;
    bool no_compile = false;
    bool do_navcheck = true;
    bool quiet = false;
    int arena_size_override = 0;
    int max_depth_override = 0;
    int max_attempts_override = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--no-compile") no_compile = true;
        else if (arg == "--no-navcheck") do_navcheck = false;
        else if (arg == "--quiet") quiet = true;
        else if (arg == "--arena-size" && i + 1 < argc) arena_size_override = std::atoi(argv[++i]);
        else if (arg == "--max-depth" && i + 1 < argc) max_depth_override = std::atoi(argv[++i]);
        else if (arg == "--max-attempts" && i + 1 < argc) max_attempts_override = std::atoi(argv[++i]);
        else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); return 1; }
    }

    // Load config.
    mapgen::MapgenConfig cfg;
    if (!config_path.empty()) {
        try { cfg = mapgen::load_config(config_path); }
        catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << "\n";
            return 1;
        }
    }

    // Apply CLI overrides.
    if (arena_size_override > 0) cfg.arena.size = arena_size_override;
    if (max_depth_override > 0) cfg.arena.max_depth = max_depth_override;
    if (max_attempts_override > 0) cfg.generation.max_attempts = max_attempts_override;
    if (no_compile) cfg.compile.enabled = false;

    // Determine seed.
    if (seed < 0) {
        seed = static_cast<int>(
            std::chrono::steady_clock::now().time_since_epoch().count() & 0x7FFFFFFF);
    }

    // Determine output path.
    if (output_path.empty()) {
        if (output_dir.empty()) output_dir = ".";
        output_path = output_dir + "/gen_" + std::to_string(seed) + ".map";
    }
    if (output_dir.empty())
        output_dir = fs::path(output_path).parent_path().string();
    fs::create_directories(output_dir);

    // Generation loop with navmesh validation.
    mapgen::Layout best_layout;
    int best_unreachable = 999999;
    int accepted_seed = seed;

    for (int attempt = 0; attempt < cfg.generation.max_attempts; ++attempt) {
        int current_seed = seed + attempt;
        std::mt19937 rng(static_cast<unsigned>(current_seed));
        auto layout = mapgen::generate_layout(rng, cfg);

        if (!do_navcheck) {
            best_layout = std::move(layout);
            accepted_seed = current_seed;
            break;
        }

        // Graph-based connectivity check on room-corridor topology.
        auto nav_result = mapgen::validate_layout_graph(layout, cfg);

        if (!quiet) {
            std::cerr << "Attempt " << attempt + 1 << " (seed " << current_seed << "): "
                      << nav_result.component_count << " components";
            if (!nav_result.unreachable_rooms.empty())
                std::cerr << ", " << nav_result.unreachable_rooms.size() << " unreachable rooms";
            std::cerr << "\n";
        }

        if (nav_result.connected) {
            best_layout = std::move(layout);
            accepted_seed = current_seed;
            best_unreachable = 0;
            break;
        }

        int n_unreachable = static_cast<int>(nav_result.unreachable_rooms.size());
        if (n_unreachable < best_unreachable) {
            best_unreachable = n_unreachable;
            best_layout = layout;
            accepted_seed = current_seed;
        }
    }

    if (best_unreachable > 0 && do_navcheck && !quiet) {
        std::cerr << "WARNING: Best layout has " << best_unreachable
                  << " unreachable rooms (seed " << accepted_seed << ")\n";
    }

    // Build the map file.
    std::mt19937 entity_rng(static_cast<unsigned>(accepted_seed));
    // Re-run layout gen with accepted seed to get deterministic entity placement.
    auto final_layout = mapgen::generate_layout(entity_rng, cfg);

    mapgen::MapFile map_file(cfg.textures);
    mapgen::build_layout(map_file, final_layout, cfg);
    mapgen::populate(map_file, final_layout, entity_rng, cfg);

    // Write texture WAD.
    mapgen::materialize_texture_wad(output_dir, cfg.textures);

    // Write .map file.
    {
        std::ofstream f(output_path);
        if (!f) {
            std::cerr << "Cannot write to " << output_path << "\n";
            return 1;
        }
        map_file.write(f);
    }

    if (!quiet) {
        std::cerr << "Generated " << output_path << " (seed " << accepted_seed
                  << ", " << final_layout.rooms.size() << " rooms, "
                  << final_layout.corridors.size() << " corridors)\n";
    }

    // Compile to .bsp.
    std::string final_path = output_path;
    if (cfg.compile.enabled) {
        try {
            final_path = mapgen::compile_map(output_path, output_dir, cfg.compile);
            if (!quiet)
                std::cerr << "Compiled " << final_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Compilation failed: " << e.what() << "\n";
            return 1;
        }
    }

    // Print the final output path to stdout for scripting.
    std::cout << final_path << "\n";

    return 0;
}
