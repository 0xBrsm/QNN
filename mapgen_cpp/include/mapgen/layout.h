#pragma once

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "mapgen/config.h"

namespace mapgen {

struct Room {
    int x0, y0, x1, y1;
    int z0 = 0;   // floor elevation
    int z1 = 192;  // ceiling elevation

    int cx() const { return (x0 + x1) / 2; }
    int cy() const { return (y0 + y1) / 2; }
    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
    int headroom() const { return z1 - z0; }
};

struct Pool {
    int x0, y0, x1, y1;
    std::string texture;
};

struct Corridor {
    int room_a;  // index into rooms
    int room_b;
    int x0, y0, x1, y1;
    int z0 = 0;       // floor at lower end
    int z1 = 128;     // ceiling
    int z0_a = 0;     // floor at room_a end
    int z0_b = 0;     // floor at room_b end
    char axis = 'x';  // 'x' = east-west, 'y' = north-south
};

struct Split {
    char axis;    // 'x' or 'y'
    int pos;      // coordinate where gap starts
    int perp_lo;  // perpendicular extent, low
    int perp_hi;  // perpendicular extent, high
};

struct Layout {
    std::vector<Room> rooms;
    std::vector<Corridor> corridors;
    std::vector<Split> splits;
    std::unordered_map<int, Pool> pools;  // room_idx -> Pool
};

Layout generate_layout(std::mt19937& rng, const MapgenConfig& cfg);

}  // namespace mapgen
