#pragma once

#include <array>
#include <ostream>
#include <string>
#include <vector>

#include "mapgen/config.h"
#include "mapgen/layout.h"

namespace mapgen {

struct Plane {
    std::array<int, 3> p1, p2, p3;
    std::string texture;
    int x_off = 0;
    int y_off = 0;
    float rotation = 0.0f;
    float x_scale = 1.0f;
    float y_scale = 1.0f;

    void write(std::ostream& out) const;
};

struct Brush {
    std::vector<Plane> planes;
    void write(std::ostream& out) const;
};

struct Entity {
    std::vector<std::pair<std::string, std::string>> properties;
    std::vector<Brush> brushes;
    void write(std::ostream& out) const;
};

struct MapFile {
    Entity worldspawn;
    std::vector<Entity> entities;

    MapFile(const TextureConfig& tex);
    void add_brush(Brush brush);
    void add_entity(const std::string& classname, int x, int y, int z,
                    const std::vector<std::pair<std::string, std::string>>& extra = {});
    void add_light(int x, int y, int z, int brightness = 300);
    void write(std::ostream& out) const;
};

Brush axis_aligned_box(int min_x, int min_y, int min_z,
                       int max_x, int max_y, int max_z,
                       const std::string& texture);

// Emit all geometry into a MapFile.
void build_layout(MapFile& m, const Layout& layout, const MapgenConfig& cfg);

}  // namespace mapgen
