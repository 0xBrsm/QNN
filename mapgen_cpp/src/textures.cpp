#include "mapgen/textures.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mapgen {

static constexpr uint8_t WAD_MIPTEX_TYPE = 68;

struct MipTex {
    std::string name;
    std::vector<uint8_t> data;  // full miptex blob (header + 4 mip levels)
};

static std::vector<uint8_t> downsample(const uint8_t* pixels, int w, int h) {
    int nw = std::max(1, w / 2);
    int nh = std::max(1, h / 2);
    std::vector<uint8_t> out(nw * nh);
    for (int y = 0; y < nh; ++y) {
        for (int x = 0; x < nw; ++x) {
            int tl = pixels[(2 * y) * w + (2 * x)];
            int tr = pixels[(2 * y) * w + std::min(w - 1, 2 * x + 1)];
            int bl = pixels[std::min(h - 1, 2 * y + 1) * w + (2 * x)];
            int br = pixels[std::min(h - 1, 2 * y + 1) * w + std::min(w - 1, 2 * x + 1)];
            out[y * nw + x] = static_cast<uint8_t>((tl + tr + bl + br) / 4);
        }
    }
    return out;
}

static MipTex encode_miptex(const std::string& name, int w, int h,
                            const std::vector<uint8_t>& pixels) {
    auto mip0 = pixels;
    auto mip1 = downsample(mip0.data(), w, h);
    auto mip2 = downsample(mip1.data(), w / 2, h / 2);
    auto mip3 = downsample(mip2.data(), w / 4, h / 4);

    uint32_t header_size = 40;
    uint32_t off0 = header_size;
    uint32_t off1 = off0 + static_cast<uint32_t>(mip0.size());
    uint32_t off2 = off1 + static_cast<uint32_t>(mip1.size());
    uint32_t off3 = off2 + static_cast<uint32_t>(mip2.size());

    MipTex mt;
    mt.name = name;
    mt.data.resize(off3 + mip3.size());

    // Header: 16-byte name, uint32 width, height, offsets[4]
    char name_buf[16] = {};
    std::strncpy(name_buf, name.c_str(), 15);
    std::memcpy(mt.data.data(), name_buf, 16);

    auto write_u32 = [&](size_t pos, uint32_t v) {
        std::memcpy(mt.data.data() + pos, &v, 4);
    };
    write_u32(16, static_cast<uint32_t>(w));
    write_u32(20, static_cast<uint32_t>(h));
    write_u32(24, off0);
    write_u32(28, off1);
    write_u32(32, off2);
    write_u32(36, off3);

    std::memcpy(mt.data.data() + off0, mip0.data(), mip0.size());
    std::memcpy(mt.data.data() + off1, mip1.data(), mip1.size());
    std::memcpy(mt.data.data() + off2, mip2.data(), mip2.size());
    std::memcpy(mt.data.data() + off3, mip3.data(), mip3.size());

    return mt;
}

// Pattern generators (same as Python).
static std::vector<uint8_t> checker(int w, int h, uint8_t a, uint8_t b, int sz) {
    std::vector<uint8_t> p(w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            p[y * w + x] = (((x / sz) + (y / sz)) % 2 == 0) ? a : b;
    return p;
}

static std::vector<uint8_t> trim_band(int w, int h, uint8_t a, uint8_t b, uint8_t accent) {
    std::vector<uint8_t> p(w * h);
    for (int y = 0; y < h; ++y) {
        uint8_t row = (y <= 1 || y >= h - 2) ? accent : (y < h / 2 ? a : b);
        for (int x = 0; x < w; ++x)
            p[y * w + x] = ((x / 8) % 2 == 0) ? row : static_cast<uint8_t>(row + 2);
    }
    return p;
}

static std::vector<uint8_t> rivets(int w, int h, uint8_t base, uint8_t alt, uint8_t rivet) {
    std::vector<uint8_t> p(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t c = (((x / 8) + (y / 8)) % 2 == 0) ? base : alt;
            if ((x % 16 == 2 || x % 16 == 13) && (y % 16 == 2 || y % 16 == 13))
                c = rivet;
            p[y * w + x] = c;
        }
    }
    return p;
}

static std::vector<uint8_t> waves(int w, int h, uint8_t c0, uint8_t c1, uint8_t c2, int period) {
    uint8_t colors[] = {c0, c1, c2};
    std::vector<uint8_t> p(w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            p[y * w + x] = colors[(x / period + y / period + (x + y) / (period * 2)) % 3];
    return p;
}

static void write_wad2(const std::string& path, const std::vector<MipTex>& textures) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write WAD: " + path);

    uint32_t header_size = 12;
    uint32_t filepos = header_size;

    // Compute directory
    struct DirEntry {
        uint32_t offset, disk_size, full_size;
        std::string name;
    };
    std::vector<DirEntry> dir;
    for (auto& t : textures) {
        uint32_t sz = static_cast<uint32_t>(t.data.size());
        dir.push_back({filepos, sz, sz, t.name});
        filepos += sz;
    }

    uint32_t info_offset = filepos;
    uint32_t count = static_cast<uint32_t>(textures.size());

    // Write header: "WAD2" + count + info_offset
    f.write("WAD2", 4);
    f.write(reinterpret_cast<const char*>(&count), 4);
    f.write(reinterpret_cast<const char*>(&info_offset), 4);

    // Write texture data
    for (auto& t : textures)
        f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size());

    // Write directory entries (32 bytes each)
    for (auto& d : dir) {
        f.write(reinterpret_cast<const char*>(&d.offset), 4);
        f.write(reinterpret_cast<const char*>(&d.disk_size), 4);
        f.write(reinterpret_cast<const char*>(&d.full_size), 4);
        uint8_t typ = WAD_MIPTEX_TYPE;
        f.write(reinterpret_cast<const char*>(&typ), 1);
        uint8_t compression = 0;
        f.write(reinterpret_cast<const char*>(&compression), 1);
        uint16_t pad = 0;
        f.write(reinterpret_cast<const char*>(&pad), 2);
        char name_buf[16] = {};
        std::strncpy(name_buf, d.name.c_str(), 15);
        f.write(name_buf, 16);
    }
}

void materialize_texture_wad(const std::string& output_dir,
                             const TextureConfig& tex) {
    std::vector<MipTex> textures;
    textures.push_back(encode_miptex(tex.floor, 64, 64, checker(64, 64, 86, 96, 8)));
    textures.push_back(encode_miptex(tex.ceiling, 64, 64, checker(64, 64, 64, 72, 16)));
    textures.push_back(encode_miptex(tex.shell, 128, 16, trim_band(128, 16, 96, 104, 112)));
    textures.push_back(encode_miptex(tex.fill, 64, 64, rivets(64, 64, 72, 88, 104)));
    textures.push_back(encode_miptex(tex.lava, 64, 64, waves(64, 64, 224, 232, 240, 5)));
    textures.push_back(encode_miptex(tex.water, 64, 64, waves(64, 64, 144, 152, 160, 7)));
    textures.push_back(encode_miptex(tex.slime, 64, 64, waves(64, 64, 184, 192, 200, 9)));

    std::string wad_path = output_dir + "/" + tex.wad_name;
    write_wad2(wad_path, textures);
}

}  // namespace mapgen
