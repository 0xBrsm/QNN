"""Mapgen texture contract and WAD materialization."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_TEXTURE_WAD = "mapgen_textures.wad"
LEGACY_TEXTURE_WAD = "gfx.wad"
_WAD_MIPTEX_TYPE = 68


@dataclass(frozen=True)
class TextureSet:
    floor: str = "tech01_1"
    ceiling: str = "tech02_1"
    shell: str = "tech04_1"
    fill: str = "metal1_1"
    lava: str = "*lava1"
    water: str = "*04water1"
    slime: str = "*slime"

    def names(self) -> tuple[str, ...]:
        return (
            self.floor,
            self.ceiling,
            self.shell,
            self.fill,
            self.lava,
            self.water,
            self.slime,
        )


MAP_TEXTURES = TextureSet()
TEXTURE_ALIASES = {
    "*water0": MAP_TEXTURES.water,
    "*slime0": MAP_TEXTURES.slime,
}


def materialize_texture_wad(output_dir: Path, wad_name: str = DEFAULT_TEXTURE_WAD) -> Path:
    """Write the mapgen texture WAD into `output_dir` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wad_path = output_dir / wad_name
    _write_wad2(wad_path, _load_required_textures())
    return wad_path


def normalize_worldspawn_wad(map_text: str, wad_name: str = DEFAULT_TEXTURE_WAD) -> str:
    """Replace the legacy mapgen WAD reference with the supported WAD name."""
    legacy = f'"wad" "{LEGACY_TEXTURE_WAD}"'
    current = f'"wad" "{wad_name}"'
    return map_text.replace(legacy, current)


@lru_cache(maxsize=1)
def _load_required_textures() -> tuple[tuple[str, bytes], ...]:
    textures = dict(_build_synthetic_textures())
    textures.update(_extract_id1_textures())
    for alias, canonical in TEXTURE_ALIASES.items():
        textures[alias] = _rename_miptex(textures[canonical], alias)
    wad_names = (*MAP_TEXTURES.names(), *TEXTURE_ALIASES.keys())
    return tuple((name, textures[name]) for name in wad_names)


def _extract_id1_textures() -> dict[str, bytes]:
    wanted = set(MAP_TEXTURES.names())
    found: dict[str, bytes] = {}

    for pak_path in _iter_id1_paks():
        for name, data in _read_pak_entries(pak_path):
            if not name.lower().endswith(".bsp"):
                continue
            found.update(_extract_miptex_blobs(data, wanted - found.keys()))
            if found.keys() >= wanted:
                return found
    return found


def _iter_id1_paks() -> list[Path]:
    paks: list[Path] = []
    for asset_root in _candidate_asset_roots():
        id1_dir = asset_root / "id1"
        for name in ("PAK0.PAK", "PAK1.PAK", "pak0.pak", "pak1.pak"):
            pak_path = id1_dir / name
            if pak_path.is_file():
                paks.append(pak_path)
    return paks


def _candidate_asset_roots() -> list[Path]:
    candidates: list[Path] = []
    env_basedir = os.environ.get("QUAKE_BASEDIR", "").strip()
    if env_basedir:
        candidates.append(Path(env_basedir))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "assets",
            Path("/assets"),
            Path("assets"),
            Path("../assets"),
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "id1").is_dir():
            unique.append(resolved)
    return unique


def _read_pak_entries(pak_path: Path) -> list[tuple[str, bytes]]:
    with pak_path.open("rb") as f:
        if f.read(4) != b"PACK":
            raise ValueError(f"Not a Quake PAK file: {pak_path}")
        dir_offset, dir_size = struct.unpack("<ii", f.read(8))
        f.seek(dir_offset)
        entries = []
        for _ in range(dir_size // 64):
            name = f.read(56).split(b"\0", 1)[0].decode("ascii", errors="ignore")
            offset, size = struct.unpack("<ii", f.read(8))
            entries.append((name, offset, size))

        result = []
        for name, offset, size in entries:
            f.seek(offset)
            result.append((name, f.read(size)))
    return result


def _extract_miptex_blobs(bsp: bytes, wanted: set[str]) -> dict[str, bytes]:
    if not wanted or len(bsp) < 4 or struct.unpack_from("<i", bsp, 0)[0] != 29:
        return {}

    tex_offset, tex_size = struct.unpack_from("<ii", bsp, 4 + 2 * 8)
    texture_lump = bsp[tex_offset:tex_offset + tex_size]
    if len(texture_lump) < 4:
        return {}

    texture_count = struct.unpack_from("<i", texture_lump, 0)[0]
    texture_offsets = struct.unpack_from(f"<{texture_count}i", texture_lump, 4)
    found: dict[str, bytes] = {}
    lowered = {name.lower() for name in wanted}

    for rel_offset in texture_offsets:
        if rel_offset < 0 or rel_offset + 40 > len(texture_lump):
            continue
        name = struct.unpack_from("<16s", texture_lump, rel_offset)[0].split(b"\0", 1)[0].decode("ascii", errors="ignore")
        if name.lower() not in lowered:
            continue
        width, height = struct.unpack_from("<II", texture_lump, rel_offset + 16)
        mip0_offset = struct.unpack_from("<I", texture_lump, rel_offset + 24)[0]
        total = _miptex_pixel_count(width, height)
        end_offset = rel_offset + mip0_offset + total
        if end_offset > len(texture_lump):
            continue
        found[name] = texture_lump[rel_offset:end_offset]
        if len(found) == len(wanted):
            break

    return found


def _build_synthetic_textures() -> tuple[tuple[str, bytes], ...]:
    return (
        (MAP_TEXTURES.floor, _encode_miptex(MAP_TEXTURES.floor, 64, 64, _checker(64, 64, 86, 96, 8))),
        (MAP_TEXTURES.ceiling, _encode_miptex(MAP_TEXTURES.ceiling, 64, 64, _checker(64, 64, 64, 72, 16))),
        (MAP_TEXTURES.shell, _encode_miptex(MAP_TEXTURES.shell, 128, 16, _trim_band(128, 16, 96, 104, 112))),
        (MAP_TEXTURES.fill, _encode_miptex(MAP_TEXTURES.fill, 64, 64, _rivets(64, 64, 72, 88, 104))),
        (MAP_TEXTURES.lava, _encode_miptex(MAP_TEXTURES.lava, 64, 64, _waves(64, 64, (224, 232, 240), 5))),
        (MAP_TEXTURES.water, _encode_miptex(MAP_TEXTURES.water, 64, 64, _waves(64, 64, (144, 152, 160), 7))),
        (MAP_TEXTURES.slime, _encode_miptex(MAP_TEXTURES.slime, 64, 64, _waves(64, 64, (184, 192, 200), 9))),
    )


def _checker(width: int, height: int, a: int, b: int, size: int) -> bytes:
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            use_a = ((x // size) + (y // size)) % 2 == 0
            pixels[y * width + x] = a if use_a else b
    return bytes(pixels)


def _trim_band(width: int, height: int, a: int, b: int, accent: int) -> bytes:
    pixels = bytearray(width * height)
    for y in range(height):
        row_color = accent if y in (0, 1, height - 2, height - 1) else (a if y < height // 2 else b)
        for x in range(width):
            pixels[y * width + x] = row_color if (x // 8) % 2 == 0 else row_color + 2
    return bytes(pixels)


def _rivets(width: int, height: int, base: int, alt: int, rivet: int) -> bytes:
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            color = base if ((x // 8) + (y // 8)) % 2 == 0 else alt
            if x % 16 in (2, 13) and y % 16 in (2, 13):
                color = rivet
            pixels[y * width + x] = color
    return bytes(pixels)


def _waves(width: int, height: int, colors: tuple[int, int, int], period: int) -> bytes:
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            index = (x // period + y // period + (x + y) // (period * 2)) % len(colors)
            pixels[y * width + x] = colors[index]
    return bytes(pixels)


def _encode_miptex(name: str, width: int, height: int, pixels: bytes) -> bytes:
    if width % 16 or height % 16:
        raise ValueError(f"Texture {name} must use 16-unit dimensions, got {width}x{height}")

    mip0 = pixels
    mip1 = _downsample(mip0, width, height)
    mip2 = _downsample(mip1, width // 2, height // 2)
    mip3 = _downsample(mip2, width // 4, height // 4)
    header_size = 40
    mip0_offset = header_size
    mip1_offset = mip0_offset + len(mip0)
    mip2_offset = mip1_offset + len(mip1)
    mip3_offset = mip2_offset + len(mip2)
    name_bytes = name.encode("ascii")
    if len(name_bytes) > 15:
        raise ValueError(f"Texture name is too long for Quake miptex: {name}")

    header = struct.pack(
        "<16sIIIIII",
        name_bytes.ljust(16, b"\0"),
        width,
        height,
        mip0_offset,
        mip1_offset,
        mip2_offset,
        mip3_offset,
    )
    return header + mip0 + mip1 + mip2 + mip3


def _rename_miptex(blob: bytes, name: str) -> bytes:
    name_bytes = name.encode("ascii")
    if len(name_bytes) > 15:
        raise ValueError(f"Texture name is too long for Quake miptex: {name}")
    return name_bytes.ljust(16, b"\0") + blob[16:]


def _downsample(pixels: bytes, width: int, height: int) -> bytes:
    next_width = max(1, width // 2)
    next_height = max(1, height // 2)
    out = bytearray(next_width * next_height)
    for y in range(next_height):
        for x in range(next_width):
            top_left = pixels[(2 * y) * width + (2 * x)]
            top_right = pixels[(2 * y) * width + min(width - 1, 2 * x + 1)]
            bottom_left = pixels[min(height - 1, 2 * y + 1) * width + (2 * x)]
            bottom_right = pixels[min(height - 1, 2 * y + 1) * width + min(width - 1, 2 * x + 1)]
            out[y * next_width + x] = (top_left + top_right + bottom_left + bottom_right) // 4
    return bytes(out)


def _miptex_pixel_count(width: int, height: int) -> int:
    total = 0
    level_width = width
    level_height = height
    for _ in range(4):
        total += level_width * level_height
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)
    return total


def _write_wad2(path: Path, textures: tuple[tuple[str, bytes], ...]) -> None:
    header_size = 12
    filepos = header_size
    directory: list[bytes] = []
    payload = bytearray()

    for name, blob in textures:
        name_bytes = name.encode("ascii")
        if len(name_bytes) > 15:
            raise ValueError(f"Texture name is too long for WAD2: {name}")
        payload.extend(blob)
        directory.append(
            struct.pack(
                "<iiiBBH16s",
                filepos,
                len(blob),
                len(blob),
                _WAD_MIPTEX_TYPE,
                0,
                0,
                name_bytes.ljust(16, b"\0"),
            )
        )
        filepos += len(blob)

    info_offset = header_size + len(payload)
    with path.open("wb") as f:
        f.write(struct.pack("<4sii", b"WAD2", len(directory), info_offset))
        f.write(payload)
        for entry in directory:
            f.write(entry)
