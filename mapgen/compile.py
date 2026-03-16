"""Compile .map files to .bsp using ericw-tools (qbsp, vis, light)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .textures import DEFAULT_TEXTURE_WAD, materialize_texture_wad, normalize_worldspawn_wad

# Default to vendored ericw-tools if available.
_VENDOR_BIN = Path(__file__).resolve().parents[2] / "vendor" / "ericw-tools" / "bin"


def find_tool(name: str) -> str | None:
    """Find a compilation tool: check vendored dir first, then PATH."""
    vendored = _VENDOR_BIN / name
    if vendored.is_file() and os.access(vendored, os.X_OK):
        return str(vendored)
    return shutil.which(name)


def compile_map(
    map_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """Compile a .map to .bsp.

    Returns path to the compiled .bsp file.
    Raises RuntimeError if qbsp or light fails.  vis is skipped if
    no .prt file exists (common for procgen maps that aren't fully sealed).
    """
    map_path = Path(map_path).resolve()
    if output_dir is None:
        output_dir = map_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compile_input = map_path
    source_text = map_path.read_text()
    normalized_text = normalize_worldspawn_wad(source_text, DEFAULT_TEXTURE_WAD)
    if output_dir != map_path.parent or normalized_text != source_text:
        compile_input = output_dir / map_path.name
        compile_input.write_text(normalized_text)

    bsp_path = output_dir / compile_input.with_suffix(".bsp").name
    prt_path = output_dir / compile_input.with_suffix(".prt").name

    # Resolve tool paths.
    tools = {}
    for name in ("qbsp", "vis", "light"):
        path = find_tool(name)
        if path is None:
            raise RuntimeError(
                f"{name} not found. Install ericw-tools or check vendor/ericw-tools/bin"
            )
        tools[name] = path

    # ericw-tools needs its shared libs; set LD_LIBRARY_PATH to vendor bin dir.
    env = os.environ.copy()
    vendor_lib = str(_VENDOR_BIN)
    env["LD_LIBRARY_PATH"] = vendor_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    materialize_texture_wad(output_dir, DEFAULT_TEXTURE_WAD)

    def _run(tool_name: str, cmd: list[str]) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            cwd=output_dir.as_posix(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{tool_name} failed (exit {result.returncode}):\n"
                f"{result.stderr}\n{result.stdout}"
            )
        return result

    # 1. qbsp: .map -> .bsp (and .prt if sealed)
    # -splitturb: prevent qbsp from merging liquid faces into oversized surfaces.
    qbsp_result = _run("qbsp", [tools["qbsp"], "-splitturb", "-splitspecial", compile_input.as_posix()])

    if not bsp_path.exists():
        raise RuntimeError(f"qbsp did not produce {bsp_path}")

    # Check for hull filling warnings — broken collision if present.
    if "No entities in empty space" in qbsp_result.stdout:
        import sys
        print(f"WARNING: qbsp hull filling failed for {map_path.name}", file=sys.stderr)
        for line in qbsp_result.stdout.splitlines():
            if "WARNING 19" in line:
                print(f"  {line.strip()}", file=sys.stderr)

    # 2. vis: optimize PVS (skip if no portal file — map has a leak)
    if prt_path.exists():
        _run("vis", [tools["vis"], bsp_path.as_posix()])
    # else: no .prt = unsealed map, vis not possible — fine for training

    # 3. light: compute lightmaps
    _run("light", [tools["light"], bsp_path.as_posix()])

    return bsp_path
