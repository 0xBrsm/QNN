"""Minimal Quake BSP parsing focused on entity metadata."""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from quake_ai.utils.io import read_json

PAK_MAGIC = b"PACK"
PAK_DIR_ENTRY_SIZE = 64
PAK_DIR_ENTRY = struct.Struct("<56sii")
QUAKE_BSP_VERSION = 29
NUM_LUMPS = 15
ENTITY_LUMP_INDEX = 0

_ENTITY_BLOCK_RE = re.compile(r"\{([^}]*)\}", re.DOTALL)
_KV_RE = re.compile(r'"([^"]+)"\s+"([^"]*)"')


def parse_entities_text(text: str) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    for block in _ENTITY_BLOCK_RE.findall(text):
        kvs = dict(_KV_RE.findall(block))
        if kvs:
            entities.append(kvs)
    return entities


def _parse_bsp_entities_bytes(raw: bytes, source_name: str) -> List[Dict[str, str]]:
    if len(raw) < 4 + NUM_LUMPS * 8:
        raise ValueError(f"Invalid BSP file {source_name}: too small")

    version = struct.unpack_from("<i", raw, 0)[0]
    if version != QUAKE_BSP_VERSION:
        raise ValueError(f"Unsupported BSP version {version}; expected {QUAKE_BSP_VERSION}")

    lump_offset = 4 + ENTITY_LUMP_INDEX * 8
    entity_ofs, entity_len = struct.unpack_from("<ii", raw, lump_offset)
    if entity_ofs < 0 or entity_len < 0 or entity_ofs + entity_len > len(raw):
        raise ValueError("Entity lump offset/length is invalid")

    text = raw[entity_ofs : entity_ofs + entity_len].decode("latin-1", errors="ignore")
    return parse_entities_text(text)


def _parse_bsp_entities(path: Path) -> List[Dict[str, str]]:
    return _parse_bsp_entities_bytes(path.read_bytes(), str(path))


def _resolve_id1_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name.lower() == "id1" and candidate.is_dir():
        return candidate

    id1_dir = candidate / "id1"
    if id1_dir.is_dir():
        return id1_dir

    raise FileNotFoundError(f"Could not resolve id1 directory under {candidate}")


def is_quake_asset_root(path: str | Path) -> bool:
    try:
        id1_dir = _resolve_id1_dir(path)
    except FileNotFoundError:
        return False
    return any(candidate.exists() for candidate in _iter_pak_paths(id1_dir))


def _pak_index(path: Path) -> int:
    stem = path.stem.lower()
    if stem.startswith("pak") and stem[3:].isdigit():
        return int(stem[3:])
    return -1


def _iter_pak_paths(id1_dir: Path) -> List[Path]:
    candidates = sorted(
        [path for path in id1_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pak" and path.stem.lower().startswith("pak")],
        key=lambda path: path.name.lower(),
    )
    deduped: Dict[str, Path] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.name.lower(), candidate)
    return sorted(deduped.values(), key=lambda path: (_pak_index(path), path.name.lower()), reverse=True)


def _iter_pak_directory_entries(pak_path: Path) -> List[Tuple[str, int, int]]:
    raw = pak_path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"Invalid PAK file {pak_path}: too small")
    if raw[:4] != PAK_MAGIC:
        raise ValueError(f"Invalid PAK file {pak_path}: missing PACK header")

    dir_offset, dir_length = struct.unpack_from("<ii", raw, 4)
    if dir_offset < 0 or dir_length < 0 or dir_offset + dir_length > len(raw):
        raise ValueError(f"Invalid PAK file {pak_path}: bad directory bounds")
    if dir_length % PAK_DIR_ENTRY_SIZE != 0:
        raise ValueError(f"Invalid PAK file {pak_path}: directory length is not aligned")

    entries: List[Tuple[str, int, int]] = []
    for offset in range(dir_offset, dir_offset + dir_length, PAK_DIR_ENTRY_SIZE):
        raw_name, file_offset, file_length = PAK_DIR_ENTRY.unpack_from(raw, offset)
        if file_offset < 0 or file_length < 0 or file_offset + file_length > len(raw):
            raise ValueError(f"Invalid PAK file {pak_path}: bad entry bounds")
        name = raw_name.split(b"\0", 1)[0].decode("latin-1", errors="ignore").replace("\\", "/")
        entries.append((name, file_offset, file_length))
    return entries


def load_quake_asset_bytes(path: str | Path, asset_name: str) -> bytes:
    id1_dir = _resolve_id1_dir(path)
    normalized = asset_name.replace("\\", "/").lstrip("/").lower()

    unpacked = id1_dir / normalized
    if unpacked.exists():
        return unpacked.read_bytes()

    for pak_path in _iter_pak_paths(id1_dir):
        pak_bytes = pak_path.read_bytes()
        for entry_name, file_offset, file_length in _iter_pak_directory_entries(pak_path):
            if entry_name.lower() != normalized:
                continue
            return pak_bytes[file_offset : file_offset + file_length]

    raise FileNotFoundError(f"Could not find asset {asset_name} under {id1_dir}")


def parse_map_entities_from_quake_assets(path: str | Path, map_name: str) -> List[Dict[str, str]]:
    normalized = map_name.lower()
    if normalized.endswith(".bsp"):
        normalized = normalized[:-4]
    raw = load_quake_asset_bytes(path, f"maps/{normalized}.bsp")
    return _parse_bsp_entities_bytes(raw, f"{path}/id1/maps/{normalized}.bsp")


def parse_map_entities(path: str | Path) -> List[Dict[str, str]]:
    map_path = Path(path)
    if map_path.suffix.lower() == ".json":
        payload = read_json(map_path)
        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("JSON map metadata must include list field 'entities'")
        return [dict(e) for e in entities]
    return _parse_bsp_entities(map_path)


def parse_origin(origin: str) -> Tuple[float, float, float]:
    parts = [float(p) for p in origin.split()]
    if len(parts) != 3:
        raise ValueError(f"Invalid origin format: {origin}")
    return parts[0], parts[1], parts[2]


def region_for_point(point: Tuple[float, float, float], grid_size: float = 256.0) -> int:
    x, y, _ = point
    gx = int(math.floor((x / grid_size) + 0.5))
    gy = int(math.floor((y / grid_size) + 0.5))
    # Deterministic packed region ID.
    return (gx + 1024) * 2048 + (gy + 1024)


def extract_map_nodes(entities: Iterable[Mapping[str, str]]) -> Dict[str, object]:
    spawn_points: List[List[float]] = []
    item_nodes: List[Dict[str, object]] = []
    goal_regions: List[int] = []
    all_regions: Dict[int, Tuple[float, float, float]] = {}

    for entity in entities:
        classname = entity.get("classname", "")
        if "origin" not in entity:
            continue
        point = parse_origin(entity["origin"])
        region_id = region_for_point(point)
        all_regions.setdefault(region_id, point)

        if classname in {"info_player_start", "info_player_deathmatch"}:
            spawn_points.append([point[0], point[1], point[2]])
        if classname.startswith("item_"):
            item_nodes.append(
                {
                    "classname": classname,
                    "origin": [point[0], point[1], point[2]],
                    "region_id": region_id,
                }
            )
        if classname == "trigger_changelevel":
            goal_regions.append(region_id)

    if not spawn_points:
        spawn_points.append([0.0, 0.0, 0.0])

    if not goal_regions:
        # Fallback: farthest discovered region from first spawn.
        spawn_region = region_for_point(tuple(spawn_points[0]))
        candidates = sorted(all_regions.keys())
        goal_regions = [candidates[-1] if candidates else spawn_region]

    region_points = {rid: [p[0], p[1], p[2]] for rid, p in all_regions.items()}
    return {
        "spawn_points": spawn_points,
        "item_nodes": item_nodes,
        "goal_regions": sorted(set(goal_regions)),
        "region_points": region_points,
    }
