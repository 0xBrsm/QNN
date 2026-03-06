"""Map feature generation for symbolic navigation context."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from quake_ai.maps.bsp_parser import extract_map_nodes, parse_map_entities
from quake_ai.schemas import MapFeaturesV1
from quake_ai.utils.io import write_json


def _grid_xy(region_id: int) -> Tuple[int, int]:
    packed = region_id
    gx = packed // 2048 - 1024
    gy = packed % 2048 - 1024
    return gx, gy


def _connect_regions(region_ids: Iterable[int], k: int = 2) -> List[List[int]]:
    ids = sorted(set(region_ids))
    edges: List[List[int]] = []
    for src in ids:
        sx, sy = _grid_xy(src)
        ranked = []
        for dst in ids:
            if src == dst:
                continue
            dx, dy = _grid_xy(dst)
            dist = abs(sx - dx) + abs(sy - dy)
            ranked.append((dist, dst))
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        for _, dst in ranked[:k]:
            edges.append([src, dst])
            edges.append([dst, src])
    # Preserve deterministic order.
    unique = sorted({(a, b) for a, b in edges})
    return [[a, b] for a, b in unique]


def _distance_to_goal(region_ids: Iterable[int], edges: Iterable[List[int]], goal_nodes: List[int]) -> Dict[int, float]:
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges:
        adjacency[src].append(dst)

    dist: Dict[int, float] = {rid: float("inf") for rid in region_ids}
    queue: deque[int] = deque()
    for goal in goal_nodes:
        if goal in dist:
            dist[goal] = 0.0
            queue.append(goal)

    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if dist[nxt] > dist[cur] + 1.0:
                dist[nxt] = dist[cur] + 1.0
                queue.append(nxt)

    max_finite = max((d for d in dist.values() if d < float("inf")), default=0.0)
    for key, value in dist.items():
        if value == float("inf"):
            dist[key] = max_finite + 1.0
    return dist


def build_map_features(map_path: str | Path, map_id: str) -> List[MapFeaturesV1]:
    entities = parse_map_entities(map_path)
    nodes = extract_map_nodes(entities)

    region_points = nodes["region_points"]
    region_ids = sorted(int(rid) for rid in region_points.keys())
    if not region_ids:
        region_ids = [0]

    edges = _connect_regions(region_ids)
    goal_nodes = [int(r) for r in nodes["goal_regions"]]
    distances = _distance_to_goal(region_ids, edges, goal_nodes)

    records: List[MapFeaturesV1] = []
    for region_id in region_ids:
        records.append(
            MapFeaturesV1(
                map_id=map_id,
                region_id=region_id,
                spawn_points=nodes["spawn_points"],
                item_nodes=nodes["item_nodes"],
                connectivity_edges=edges,
                goal_nodes=goal_nodes,
                distance_to_goal=distances[region_id],
            )
        )
    return records


def write_map_features(path: str | Path, records: List[MapFeaturesV1]) -> None:
    write_json(path, {"records": [r.to_dict() for r in records]})
