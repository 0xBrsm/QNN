#pragma once

#include <string>
#include <vector>

#include "mapgen/config.h"
#include "mapgen/layout.h"

namespace mapgen {

struct NavcheckResult {
    bool connected = false;
    int component_count = 0;
    std::vector<int> unreachable_rooms;
    std::string error;
};

// Graph-based connectivity check: BFS on the room-corridor topology.
NavcheckResult validate_layout_graph(const Layout& layout,
                                     const MapgenConfig& cfg);

}  // namespace mapgen
