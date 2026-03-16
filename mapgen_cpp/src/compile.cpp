#include "mapgen/compile.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace mapgen {

static std::string find_tool(const std::string& name, const std::string& tools_dir) {
    if (!tools_dir.empty()) {
        auto p = fs::path(tools_dir) / name;
        if (fs::exists(p)) return p.string();
    }
    // Check for vendored ericw-tools relative to the executable.
    auto exe_path = fs::read_symlink("/proc/self/exe");
    auto vendor_dir = exe_path.parent_path().parent_path().parent_path().parent_path()
                      / "vendor" / "ericw-tools" / "bin";
    if (fs::exists(vendor_dir / name))
        return (vendor_dir / name).string();
    // Fallback: rely on PATH.
    return name;
}

static void run_tool(const std::string& tool, const std::vector<std::string>& args,
                     const std::string& cwd, const std::string& ld_path) {
    std::string cmd;
    if (!ld_path.empty())
        cmd = "LD_LIBRARY_PATH=\"" + ld_path + ":$LD_LIBRARY_PATH\" ";
    cmd += tool;
    for (auto& a : args)
        cmd += " \"" + a + "\"";
    cmd += " > /dev/null 2>&1";

    auto prev_cwd = fs::current_path();
    if (!cwd.empty()) fs::current_path(cwd);
    int rc = std::system(cmd.c_str());
    fs::current_path(prev_cwd);

    if (rc != 0)
        throw std::runtime_error(tool + " failed with exit code " + std::to_string(rc));
}

std::string compile_map(const std::string& map_path,
                        const std::string& output_dir,
                        const CompileConfig& cfg) {
    fs::path mp(map_path);
    fs::path out_dir = output_dir.empty() ? mp.parent_path() : fs::path(output_dir);
    fs::create_directories(out_dir);

    std::string qbsp = find_tool("qbsp", cfg.tools_dir);
    std::string vis = find_tool("vis", cfg.tools_dir);
    std::string light = find_tool("light", cfg.tools_dir);

    // Use tool's directory for LD_LIBRARY_PATH (vendored tools need their libs).
    std::string ld_path = fs::path(qbsp).parent_path().string();

    fs::path bsp_path = out_dir / mp.filename().replace_extension(".bsp");
    fs::path prt_path = out_dir / mp.filename().replace_extension(".prt");

    // qbsp
    std::vector<std::string> qbsp_args = cfg.qbsp_flags;
    qbsp_args.push_back(mp.string());
    run_tool(qbsp, qbsp_args, out_dir.string(), ld_path);

    if (!fs::exists(bsp_path))
        throw std::runtime_error("qbsp did not produce " + bsp_path.string());

    // vis (optional, skip if no .prt)
    if (cfg.run_vis && fs::exists(prt_path))
        run_tool(vis, {bsp_path.string()}, out_dir.string(), ld_path);

    // light
    if (cfg.run_light)
        run_tool(light, {bsp_path.string()}, out_dir.string(), ld_path);

    return bsp_path.string();
}

}  // namespace mapgen
