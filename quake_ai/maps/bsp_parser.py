"""Minimal Quake BSP parsing focused on entity metadata."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from quake_ai.utils.io import read_json

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


def _parse_bsp_entities(path: Path) -> List[Dict[str, str]]:
    raw = path.read_bytes()
    if len(raw) < 4 + NUM_LUMPS * 8:
        raise ValueError(f"Invalid BSP file {path}: too small")

    version = struct.unpack_from("<i", raw, 0)[0]
    if version != QUAKE_BSP_VERSION:
        raise ValueError(f"Unsupported BSP version {version}; expected {QUAKE_BSP_VERSION}")

    lump_offset = 4 + ENTITY_LUMP_INDEX * 8
    entity_ofs, entity_len = struct.unpack_from("<ii", raw, lump_offset)
    if entity_ofs < 0 or entity_len < 0 or entity_ofs + entity_len > len(raw):
        raise ValueError("Entity lump offset/length is invalid")

    text = raw[entity_ofs : entity_ofs + entity_len].decode("latin-1", errors="ignore")
    return parse_entities_text(text)


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
    gx = int(round(x / grid_size))
    gy = int(round(y / grid_size))
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
