#pragma once

#include <random>

#include "mapgen/brush.h"
#include "mapgen/config.h"
#include "mapgen/layout.h"

namespace mapgen {

void populate(MapFile& m, const Layout& layout, std::mt19937& rng,
              const MapgenConfig& cfg);

}  // namespace mapgen
