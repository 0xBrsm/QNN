#pragma once

#include <string>

#include "mapgen/config.h"

namespace mapgen {

// Write the mapgen texture WAD into output_dir.
void materialize_texture_wad(const std::string& output_dir,
                             const TextureConfig& tex);

}  // namespace mapgen
