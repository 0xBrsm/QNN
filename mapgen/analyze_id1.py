"""Extract and summarize entity + layout distributions from id1 PAK/BSP files.

Analyzes both entity placement and BSP geometry to characterize map layouts:
rooms (open areas), corridors, height variance, hazard volumes, water areas.

Usage: python -m mapgen.analyze_id1 [--basedir assets/id1]
"""

from __future__ import annotations

import argparse
import struct
import re
import math
from collections import Counter
from pathlib import Path


# --- PAK / BSP reading ---

def read_pak(pak_path: Path) -> dict[str, bytes]:
    """Read a Quake PAK file, return {filename: data}."""
    with open(pak_path, "rb") as f:
        magic = f.read(4)
        if magic != b"PACK":
            raise ValueError(f"Not a PAK file: {pak_path}")
        dir_offset, dir_size = struct.unpack("<ii", f.read(8))
        num_entries = dir_size // 64

        entries = {}
        f.seek(dir_offset)
        for _ in range(num_entries):
            name_raw = f.read(56)
            name = name_raw.split(b"\x00", 1)[0].decode("ascii")
            offset, size = struct.unpack("<ii", f.read(8))
            entries[name] = (offset, size)

        result = {}
        for name, (offset, size) in entries.items():
            f.seek(offset)
            result[name] = f.read(size)

    return result


# BSP29 lump indices.
LUMP_ENTITIES = 0
LUMP_PLANES = 1
LUMP_TEXTURES = 2
LUMP_VERTEXES = 3
LUMP_NODES = 5
LUMP_TEXINFO = 6
LUMP_FACES = 7
LUMP_LEAVES = 10
LUMP_MODELS = 14


def _read_lump(bsp: bytes, lump_idx: int) -> bytes:
    """Read a lump from BSP29 data."""
    base = 4 + lump_idx * 8  # skip version int, each lump = (offset, size)
    offset, size = struct.unpack_from("<ii", bsp, base)
    return bsp[offset:offset + size]


def extract_entity_lump(bsp: bytes) -> str:
    return _read_lump(bsp, LUMP_ENTITIES).decode("ascii", errors="replace")


def extract_vertices(bsp: bytes) -> list[tuple[float, float, float]]:
    """Extract all vertices as (x, y, z) tuples."""
    data = _read_lump(bsp, LUMP_VERTEXES)
    n = len(data) // 12  # 3 floats per vertex
    verts = []
    for i in range(n):
        x, y, z = struct.unpack_from("<fff", data, i * 12)
        verts.append((x, y, z))
    return verts


def extract_leaves(bsp: bytes) -> list[dict]:
    """Extract BSP leaves. Each leaf = 28 bytes in BSP29."""
    data = _read_lump(bsp, LUMP_LEAVES)
    # BSP29 leaf: int type, int visofs, short[3] mins, short[3] maxs,
    #             ushort firstmarksurface, ushort nummarksurfaces,
    #             byte[4] ambient_level
    leaf_size = 28
    n = len(data) // leaf_size
    leaves = []
    for i in range(n):
        off = i * leaf_size
        contents = struct.unpack_from("<i", data, off)[0]
        mins = struct.unpack_from("<hhh", data, off + 8)
        maxs = struct.unpack_from("<hhh", data, off + 14)
        leaves.append({
            "contents": contents,
            "mins": mins,
            "maxs": maxs,
        })
    return leaves


def extract_models(bsp: bytes) -> list[dict]:
    """Extract BSP models. Model 0 = worldspawn geometry."""
    data = _read_lump(bsp, LUMP_MODELS)
    # BSP29 model: float[3] mins, float[3] maxs, float[3] origin,
    #              int[4] headnode, int visleafs, int firstface, int numfaces
    model_size = 64
    n = len(data) // model_size
    models = []
    for i in range(n):
        off = i * model_size
        mins = struct.unpack_from("<fff", data, off)
        maxs = struct.unpack_from("<fff", data, off + 12)
        models.append({"mins": mins, "maxs": maxs})
    return models


def extract_texture_names(bsp: bytes) -> list[str]:
    """Extract texture names from the texture lump."""
    data = _read_lump(bsp, LUMP_TEXTURES)
    if len(data) < 4:
        return []
    num_tex = struct.unpack_from("<i", data, 0)[0]
    names = []
    for i in range(num_tex):
        tex_offset = struct.unpack_from("<i", data, 4 + i * 4)[0]
        if tex_offset < 0 or tex_offset + 16 > len(data):
            continue
        name_raw = data[tex_offset:tex_offset + 16]
        name = name_raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        names.append(name)
    return names


# --- Entity parsing ---

def parse_entities(ent_text: str) -> list[dict[str, str]]:
    entities = []
    current = None
    for line in ent_text.splitlines():
        line = line.strip()
        if line == "{":
            current = {}
        elif line == "}" and current is not None:
            entities.append(current)
            current = None
        elif current is not None:
            m = re.match(r'"([^"]+)"\s+"([^"]*)"', line)
            if m:
                current[m.group(1)] = m.group(2)
    return entities


# --- Layout analysis ---

# Quake BSP leaf content types.
CONTENTS_EMPTY = -1
CONTENTS_SOLID = -2
CONTENTS_WATER = -3
CONTENTS_SLIME = -4
CONTENTS_LAVA = -5
CONTENTS_SKY = -6


def _leaf_volume(leaf: dict) -> float:
    """Approximate volume of a BSP leaf in cubic Quake units."""
    mins, maxs = leaf["mins"], leaf["maxs"]
    dx = maxs[0] - mins[0]
    dy = maxs[1] - mins[1]
    dz = maxs[2] - mins[2]
    return max(0, dx) * max(0, dy) * max(0, dz)


def _leaf_floor_area(leaf: dict) -> float:
    """Approximate XY floor area of a BSP leaf."""
    mins, maxs = leaf["mins"], leaf["maxs"]
    return max(0, maxs[0] - mins[0]) * max(0, maxs[1] - mins[1])


def analyze_layout(bsp: bytes) -> dict:
    """Analyze BSP geometry for layout characteristics."""
    leaves = extract_leaves(bsp)
    verts = extract_vertices(bsp)
    models = extract_models(bsp)
    textures = extract_texture_names(bsp)

    # Categorize leaves by content type.
    empty_leaves = [l for l in leaves if l["contents"] == CONTENTS_EMPTY]
    water_leaves = [l for l in leaves if l["contents"] == CONTENTS_WATER]
    slime_leaves = [l for l in leaves if l["contents"] == CONTENTS_SLIME]
    lava_leaves = [l for l in leaves if l["contents"] == CONTENTS_LAVA]

    # Height analysis from empty (walkable) leaves.
    if empty_leaves:
        floor_zs = [l["mins"][2] for l in empty_leaves]
        ceil_zs = [l["maxs"][2] for l in empty_leaves]
        z_min = min(floor_zs)
        z_max = max(ceil_zs)
        z_range = z_max - z_min

        # Distinct floor elevations (quantize to 16-unit steps).
        floor_levels = sorted(set(round(z / 16) * 16 for z in floor_zs))
        n_levels = len(floor_levels)

        # Average headroom.
        headrooms = [l["maxs"][2] - l["mins"][2] for l in empty_leaves]
        avg_headroom = sum(headrooms) / len(headrooms)
    else:
        z_min = z_max = z_range = avg_headroom = 0
        n_levels = 0
        floor_levels = []

    # Approximate "rooms" vs "corridors" by leaf size.
    # Large open leaves (floor area > 128*128) ~ rooms.
    # Small leaves (floor area < 64*64) ~ corridors/details.
    ROOM_THRESHOLD = 128 * 128
    CORRIDOR_THRESHOLD = 64 * 64
    room_leaves = [l for l in empty_leaves if _leaf_floor_area(l) >= ROOM_THRESHOLD]
    corridor_leaves = [l for l in empty_leaves
                       if CORRIDOR_THRESHOLD <= _leaf_floor_area(l) < ROOM_THRESHOLD]

    # Total areas.
    total_floor_area = sum(_leaf_floor_area(l) for l in empty_leaves)
    room_area = sum(_leaf_floor_area(l) for l in room_leaves)
    corridor_area = sum(_leaf_floor_area(l) for l in corridor_leaves)

    # Liquid volumes.
    water_vol = sum(_leaf_volume(l) for l in water_leaves)
    slime_vol = sum(_leaf_volume(l) for l in slime_leaves)
    lava_vol = sum(_leaf_volume(l) for l in lava_leaves)

    # Liquid textures in texture list.
    liquid_texes = [t for t in textures if t.startswith("*")]

    # World bounds from model 0.
    if models:
        world = models[0]
        world_size_x = world["maxs"][0] - world["mins"][0]
        world_size_y = world["maxs"][1] - world["mins"][1]
        world_size_z = world["maxs"][2] - world["mins"][2]
    else:
        world_size_x = world_size_y = world_size_z = 0

    return {
        "total_leaves": len(leaves),
        "empty_leaves": len(empty_leaves),
        "room_leaves": len(room_leaves),
        "corridor_leaves": len(corridor_leaves),
        "water_leaves": len(water_leaves),
        "slime_leaves": len(slime_leaves),
        "lava_leaves": len(lava_leaves),
        "z_min": z_min,
        "z_max": z_max,
        "z_range": z_range,
        "n_floor_levels": n_levels,
        "avg_headroom": avg_headroom,
        "total_floor_area": total_floor_area,
        "room_area": room_area,
        "corridor_area": corridor_area,
        "water_volume": water_vol,
        "slime_volume": slime_vol,
        "lava_volume": lava_vol,
        "liquid_textures": liquid_texes,
        "world_size": (world_size_x, world_size_y, world_size_z),
        "n_vertices": len(verts),
        "n_models": len(models),
    }


# --- Entity summary ---

def summarize_entities(entities: list[dict[str, str]]) -> dict:
    classes = Counter(e.get("classname", "unknown") for e in entities)
    return {
        "spawns": classes.get("info_player_deathmatch", 0),
        "weapons": sum(classes.get(w, 0) for w in [
            "weapon_supershotgun", "weapon_nailgun", "weapon_supernailgun",
            "weapon_grenadelauncher", "weapon_rocketlauncher", "weapon_lightning",
        ]),
        "ammo": sum(classes.get(a, 0) for a in [
            "item_shells", "item_spikes", "item_rockets", "item_cells",
        ]),
        "health": classes.get("item_health", 0),
        "armor": sum(classes.get(a, 0) for a in [
            "item_armor1", "item_armor2", "item_armorInv",
        ]),
        "powerups": sum(classes.get(p, 0) for p in [
            "item_artifact_super_damage", "item_artifact_invulnerability",
            "item_artifact_invisibility",
        ]),
    }


# --- Main ---

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Analyze id1 map layouts and entities")
    p.add_argument("--basedir", type=str, default="assets/id1")
    p.add_argument("--maps", type=str, nargs="*", default=None)
    p.add_argument("--dm-only", action="store_true", help="Only show DM maps")
    args = p.parse_args(argv)

    basedir = Path(args.basedir)

    all_bsp: dict[str, bytes] = {}
    for pak_name in sorted(list(basedir.glob("[Pp][Aa][Kk]*.PAK")) + list(basedir.glob("[Pp][Aa][Kk]*.pak"))):
        try:
            pak = read_pak(pak_name)
            for name, data in pak.items():
                if name.endswith(".bsp") and "maps/" in name:
                    map_name = name.split("/")[-1].replace(".bsp", "")
                    all_bsp[map_name] = data
        except Exception as e:
            print(f"Warning: {pak_name}: {e}")

    if args.maps:
        targets = args.maps
    elif args.dm_only:
        targets = sorted(k for k in all_bsp if k.startswith("dm"))
    else:
        dm = sorted(k for k in all_bsp if k.startswith("dm"))
        ep = sorted(k for k in all_bsp if re.match(r"e\d+m\d+", k))
        targets = dm + ep

    # Header.
    print(f"{'Map':<8} {'World XY':>10} {'ZRange':>7} {'Levels':>6} "
          f"{'Headrm':>6} {'RoomLf':>6} {'CorrLf':>6} "
          f"{'Water':>6} {'Slime':>6} {'Lava':>6} "
          f"{'Spawn':>5} {'Weap':>4} {'Ammo':>4} {'HP':>3} {'Arm':>3}")
    print("-" * 110)

    for map_name in targets:
        if map_name not in all_bsp:
            print(f"{map_name:<8} (not found)")
            continue
        try:
            bsp = all_bsp[map_name]
            lay = analyze_layout(bsp)
            ent = summarize_entities(parse_entities(extract_entity_lump(bsp)))

            wx, wy, wz = lay["world_size"]
            world_xy = f"{int(wx)}x{int(wy)}"

            # Liquid volumes in K units cubed for readability.
            wvol = int(lay["water_volume"] / 1000) if lay["water_volume"] else 0
            svol = int(lay["slime_volume"] / 1000) if lay["slime_volume"] else 0
            lvol = int(lay["lava_volume"] / 1000) if lay["lava_volume"] else 0

            print(f"{map_name:<8} {world_xy:>10} {int(lay['z_range']):>7} "
                  f"{lay['n_floor_levels']:>6} {int(lay['avg_headroom']):>6} "
                  f"{lay['room_leaves']:>6} {lay['corridor_leaves']:>6} "
                  f"{wvol:>5}K {svol:>5}K {lvol:>5}K "
                  f"{ent['spawns']:>5} {ent['weapons']:>4} {ent['ammo']:>4} "
                  f"{ent['health']:>3} {ent['armor']:>3}")
        except Exception as e:
            print(f"{map_name:<8} ERROR: {e}")


if __name__ == "__main__":
    main()
