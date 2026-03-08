"""Deterministic engine-era map world-model construction."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from quake_ai.maps.bsp_parser import (
    is_quake_asset_root,
    parse_map_entities,
    parse_map_entities_from_quake_assets,
    parse_origin,
    region_for_point,
)
from quake_ai.schemas import MapStateV2, RegionNodeV2, StaticObjectV2
from quake_ai.utils.io import write_json

GRID_SIZE = 256.0
GRID_HALF_EXTENT = GRID_SIZE / 2.0
Z_HALF_EXTENT = 256.0

_SPAWN_CLASSNAMES = {
    "info_player_start",
    "info_player_coop",
    "info_player_deathmatch",
}

_GOAL_CLASSNAMES = {"trigger_changelevel"}

_CATEGORY_ORDER = {
    "spawn": 0,
    "goal": 1,
    "item": 2,
    "trigger": 3,
    "door": 4,
    "lift": 5,
    "mover": 6,
    "monster": 7,
    "misc": 8,
}


def region_center(region_id: int) -> List[float]:
    gx = region_id // 2048 - 1024
    gy = region_id % 2048 - 1024
    return [gx * GRID_SIZE, gy * GRID_SIZE, 0.0]


def _grid_xy(region_id: int) -> Tuple[int, int]:
    center = region_center(region_id)
    return int(round(center[0] / GRID_SIZE)), int(round(center[1] / GRID_SIZE))


def _parse_angles(entity: Mapping[str, str]) -> List[float]:
    if entity.get("angles"):
        parts = [float(part) for part in str(entity["angles"]).split()]
        if len(parts) == 3:
            return parts
    if entity.get("angle"):
        return [0.0, float(entity["angle"]), 0.0]
    return [0.0, 0.0, 0.0]


def _classify_entity(classname: str) -> str:
    if classname in _SPAWN_CLASSNAMES:
        return "spawn"
    if classname in _GOAL_CLASSNAMES:
        return "goal"
    if classname.startswith("item_"):
        return "item"
    if classname.startswith("trigger_"):
        return "trigger"
    if classname.startswith("func_door"):
        return "door"
    if classname.startswith(("func_plat", "func_train", "func_button")):
        return "lift"
    if classname.startswith("func_"):
        return "mover"
    if classname.startswith("monster_"):
        return "monster"
    return "misc"


def _stable_entity_key(indexed_entity: Tuple[int, Mapping[str, str]]) -> Tuple[int, str, Tuple[float, float, float], int]:
    index, entity = indexed_entity
    classname = str(entity.get("classname", ""))
    origin = parse_origin(str(entity.get("origin", "0 0 0")))
    return (_CATEGORY_ORDER[_classify_entity(classname)], classname, origin, index)


def _connect_regions(region_ids: Sequence[int], k: int = 3) -> List[List[int]]:
    ids = sorted(set(int(region_id) for region_id in region_ids))
    if len(ids) <= 1:
        return []

    edges: set[tuple[int, int]] = set()
    for src in ids:
        sx, sy = _grid_xy(src)
        ranked: List[Tuple[int, int]] = []
        for dst in ids:
            if src == dst:
                continue
            dx, dy = _grid_xy(dst)
            ranked.append((abs(sx - dx) + abs(sy - dy), dst))
        ranked.sort(key=lambda row: (row[0], row[1]))
        for _, dst in ranked[:k]:
            edges.add((src, dst))
            edges.add((dst, src))
    return [[src, dst] for src, dst in sorted(edges)]


def _distance_to_goal(region_ids: Sequence[int], edges: Sequence[Sequence[int]], goal_region_ids: Sequence[int]) -> Dict[int, float]:
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges:
        adjacency[int(src)].append(int(dst))

    distances = {int(region_id): float("inf") for region_id in region_ids}
    queue: deque[int] = deque()
    for goal_region_id in sorted(set(int(region_id) for region_id in goal_region_ids)):
        if goal_region_id in distances:
            distances[goal_region_id] = 0.0
            queue.append(goal_region_id)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if distances[neighbor] > distances[current] + 1.0:
                distances[neighbor] = distances[current] + 1.0
                queue.append(neighbor)

    max_finite = max((distance for distance in distances.values() if distance < float("inf")), default=0.0)
    for region_id, value in list(distances.items()):
        if value == float("inf"):
            distances[region_id] = max_finite + 1.0
    return distances


def build_world_model_from_entities(entities: Iterable[Mapping[str, str]], map_id: str, source: str = "entities") -> MapStateV2:
    indexed_entities = [(index, dict(entity)) for index, entity in enumerate(entities) if entity.get("origin")]
    indexed_entities.sort(key=_stable_entity_key)

    region_points: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
    region_object_ids: Dict[int, List[str]] = defaultdict(list)
    region_insertion_order: List[int] = []
    static_objects: List[StaticObjectV2] = []
    spawn_region_ids: List[int] = []
    goal_region_ids: List[int] = []

    for stable_index, (_, entity) in enumerate(indexed_entities):
        origin = parse_origin(str(entity["origin"]))
        region_id = region_for_point(origin)
        classname = str(entity.get("classname", ""))
        category = _classify_entity(classname)
        object_id = f"{category}_{stable_index:04d}"
        angles = _parse_angles(entity)

        properties = {
            str(key): value
            for key, value in entity.items()
            if key not in {"classname", "origin", "angle", "angles"}
        }

        static_objects.append(
            StaticObjectV2(
                object_id=object_id,
                category=category,
                classname=classname,
                region_id=region_id,
                origin=[float(origin[0]), float(origin[1]), float(origin[2])],
                angles=angles,
                properties=properties,
            )
        )
        if region_id not in region_points:
            region_insertion_order.append(region_id)
        region_points[region_id].append(origin)
        region_object_ids[region_id].append(object_id)
        if category == "spawn":
            spawn_region_ids.append(region_id)
        if category == "goal":
            goal_region_ids.append(region_id)

    region_ids = sorted(region_points.keys())
    if not region_ids:
        region_ids = [0]
        region_insertion_order = [0]
        region_points[0] = [(0.0, 0.0, 0.0)]

    if not spawn_region_ids:
        # Match the native worker's fallback semantics before region sorting.
        spawn_region_ids = [region_insertion_order[0]]
    if not goal_region_ids:
        # Match the native worker's fallback semantics before region sorting.
        goal_region_ids = [region_insertion_order[-1]]

    edges = _connect_regions(region_ids)
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges:
        adjacency[int(src)].append(int(dst))
    distances = _distance_to_goal(region_ids, edges, goal_region_ids)

    regions: List[RegionNodeV2] = []
    for region_id in region_ids:
        center = region_center(region_id)
        points = region_points.get(region_id, [(center[0], center[1], center[2])])
        bounds_min = [center[0] - GRID_HALF_EXTENT, center[1] - GRID_HALF_EXTENT, -Z_HALF_EXTENT]
        bounds_max = [center[0] + GRID_HALF_EXTENT, center[1] + GRID_HALF_EXTENT, Z_HALF_EXTENT]
        bounds_min[0] = min(bounds_min[0], min(point[0] for point in points) - 16.0)
        bounds_min[1] = min(bounds_min[1], min(point[1] for point in points) - 16.0)
        bounds_min[2] = min(bounds_min[2], min(point[2] for point in points) - 16.0)
        bounds_max[0] = max(bounds_max[0], max(point[0] for point in points) + 16.0)
        bounds_max[1] = max(bounds_max[1], max(point[1] for point in points) + 16.0)
        bounds_max[2] = max(bounds_max[2], max(point[2] for point in points) + 16.0)

        neighbors = sorted(set(adjacency.get(region_id, [])))
        regions.append(
            RegionNodeV2(
                region_id=region_id,
                center=center,
                neighbors=neighbors,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                object_ids=list(region_object_ids.get(region_id, [])),
                visibility_hints=neighbors[:],
            )
        )

    map_state = MapStateV2(
        map_id=map_id,
        regions=regions,
        static_objects=static_objects,
        spawn_region_ids=sorted(set(spawn_region_ids)),
        goal_region_ids=sorted(set(goal_region_ids)),
        metadata={
            "source": source,
            "grid_size": GRID_SIZE,
            "distance_to_goal": {str(region_id): distances[region_id] for region_id in region_ids},
            "max_distance_to_goal": max(distances.values()) if distances else 0.0,
            "region_count": len(region_ids),
            "static_object_count": len(static_objects),
        },
    )
    map_state.validate()
    return map_state


def build_world_model(map_path: str | Path, map_id: str) -> MapStateV2:
    source = Path(map_path)
    if source.is_dir() and is_quake_asset_root(source):
        return build_world_model_from_quake_assets(source, map_id=map_id)
    entities = parse_map_entities(source)
    world_source = "json_map_metadata" if source.suffix.lower() == ".json" else "bsp_entities"
    return build_world_model_from_entities(entities=entities, map_id=map_id, source=world_source)


def build_world_model_from_quake_assets(path: str | Path, map_id: str) -> MapStateV2:
    entities = parse_map_entities_from_quake_assets(path, map_name=map_id)
    return build_world_model_from_entities(entities=entities, map_id=map_id, source="quake_assets")


def nearest_region_id(map_state: MapStateV2, point: Sequence[float]) -> int:
    if not map_state.regions:
        return 0
    candidate = region_for_point((float(point[0]), float(point[1]), float(point[2])))
    known_region_ids = {region.region_id for region in map_state.regions}
    if candidate in known_region_ids:
        return candidate

    px, py = float(point[0]), float(point[1])
    ranked = []
    for region in map_state.regions:
        dx = px - float(region.center[0])
        dy = py - float(region.center[1])
        ranked.append((dx * dx + dy * dy, region.region_id))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return ranked[0][1]


def write_world_model(path: str | Path, map_state: MapStateV2) -> None:
    write_json(path, map_state.to_dict())
