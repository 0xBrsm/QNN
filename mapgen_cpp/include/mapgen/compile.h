#pragma once

#include <string>

#include "mapgen/config.h"

namespace mapgen {

// Compile a .map file to .bsp using ericw-tools (qbsp, vis, light).
// Returns path to the compiled .bsp file.
std::string compile_map(const std::string& map_path,
                        const std::string& output_dir,
                        const CompileConfig& cfg);

}  // namespace mapgen
