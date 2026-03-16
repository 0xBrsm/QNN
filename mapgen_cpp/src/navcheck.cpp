#include "mapgen/navcheck.h"

#include <cmath>
#include <queue>
#include <vector>

namespace mapgen {

NavcheckResult validate_layout_graph(const Layout& layout,
                                     const MapgenConfig& cfg) {
    NavcheckResult result;
    int n = static_cast<int>(layout.rooms.size());
    if (n == 0) {
        result.error = "No rooms";
        return result;
    }

    // Build adjacency list from corridors, filtering out physically
    // unnavigable connections.
    std::vector<std::vector<int>> adj(n);

    for (const auto& c : layout.corridors) {
        if (c.room_a < 0 || c.room_a >= n || c.room_b < 0 || c.room_b >= n)
            continue;

        // Check corridor is wide enough for player.
        int width = (c.axis == 'x') ? (c.y1 - c.y0) : (c.x1 - c.x0);
        if (width < static_cast<int>(cfg.navmesh.walkable_radius * 2))
            continue;

        // Check height difference is climbable (must match build_stairs logic).
        int dz = std::abs(c.z0_b - c.z0_a);
        if (dz > 0) {
            int bridged = std::min(dz, cfg.stairs.max_rise);
            int remainder = dz - bridged;
            int climb = static_cast<int>(cfg.navmesh.walkable_climb);
            // Remainder after stairs must be within a single step.
            if (remainder > climb)
                continue;
            int total_run = (c.axis == 'x') ? (c.x1 - c.x0) : (c.y1 - c.y0);
            int max_steps = total_run / cfg.stairs.min_step_run;
            int min_steps = (bridged + climb - 1) / climb;
            if (min_steps > max_steps)
                continue;
        }

        adj[c.room_a].push_back(c.room_b);
        adj[c.room_b].push_back(c.room_a);
    }

    // BFS from room 0.
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;

    while (!q.empty()) {
        int room = q.front(); q.pop();
        for (int neighbor : adj[room]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }

    for (int i = 0; i < n; ++i)
        if (!visited[i])
            result.unreachable_rooms.push_back(i);

    // Count connected components.
    result.component_count = 1;
    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        ++result.component_count;
        std::queue<int> cq;
        cq.push(i);
        visited[i] = true;
        while (!cq.empty()) {
            int room = cq.front(); cq.pop();
            for (int neighbor : adj[room]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    cq.push(neighbor);
                }
            }
        }
    }

    result.connected = result.unreachable_rooms.empty();
    return result;
}

}  // namespace mapgen
