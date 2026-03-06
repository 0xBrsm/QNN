"""Shared navigation graph helpers and observation featurization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from quake_ai.schemas import MapFeaturesV1
from quake_ai.utils.io import read_json

HEADING_VECTORS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.70710678, 0.70710678),
    (0.0, 1.0),
    (-0.70710678, 0.70710678),
    (-1.0, 0.0),
    (-0.70710678, -0.70710678),
    (0.0, -1.0),
    (0.70710678, -0.70710678),
)


def heading_from_yaw(yaw: float) -> int:
    return int(round(yaw / 45.0)) % len(HEADING_VECTORS)


def yaw_from_heading(heading: int) -> float:
    return float((heading % len(HEADING_VECTORS)) * 45.0)


def region_to_point(region_id: int) -> Tuple[float, float, float]:
    gx = region_id // 2048 - 1024
    gy = region_id % 2048 - 1024
    return gx * 256.0, gy * 256.0, 0.0


@dataclass(slots=True)
class NavigationMap:
    records_by_region: Dict[int, MapFeaturesV1]
    adjacency: Dict[int, List[int]]
    goal_regions: set[int]
    item_regions: set[int]
    spawn_regions: List[int]
    max_distance: float

    def distance(self, region_id: int) -> float:
        record = self.records_by_region.get(region_id)
        if record is None:
            return self.max_distance
        return float(record.distance_to_goal)

    def goal_progress(self, region_id: int) -> float:
        if self.max_distance <= 0.0:
            return 0.0
        return float(np.clip(1.0 - self.distance(region_id) / self.max_distance, 0.0, 1.0))


def _build_adjacency(records: Sequence[MapFeaturesV1]) -> Dict[int, List[int]]:
    adjacency: Dict[int, set[int]] = {}
    for record in records:
        adjacency.setdefault(record.region_id, set())
        for edge in record.connectivity_edges:
            src, dst = int(edge[0]), int(edge[1])
            adjacency.setdefault(src, set()).add(dst)
    return {region_id: sorted(neighbors) for region_id, neighbors in adjacency.items()}


def build_navigation_map(records: Sequence[MapFeaturesV1]) -> NavigationMap:
    if not records:
        raise RuntimeError("Navigation map requires at least one region record")

    by_region = {record.region_id: record for record in records}
    sample = records[0]
    adjacency = _build_adjacency(records)
    item_regions = {int(node.get("region_id", 0)) for node in sample.item_nodes}
    spawn_regions = [
        region_id
        for region_id in {int(round(point[0] / 256.0) + 1024) * 2048 + int(round(point[1] / 256.0) + 1024) for point in sample.spawn_points}
        if region_id in by_region
    ]
    if not spawn_regions:
        spawn_regions = [records[0].region_id]

    max_distance = max(float(record.distance_to_goal) for record in records)
    if max_distance <= 0.0:
        max_distance = 1.0

    return NavigationMap(
        records_by_region=by_region,
        adjacency=adjacency,
        goal_regions={int(region) for region in sample.goal_nodes},
        item_regions=item_regions,
        spawn_regions=sorted(set(spawn_regions)),
        max_distance=max_distance,
    )


def load_navigation_map(path: str | Path) -> NavigationMap:
    payload = read_json(path)
    records = [MapFeaturesV1.from_dict(row) for row in payload.get("records", [])]
    return build_navigation_map(records)


def _neighbor_direction(region_id: int, neighbor_id: int) -> np.ndarray:
    x0, y0, _ = region_to_point(region_id)
    x1, y1, _ = region_to_point(neighbor_id)
    direction = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    return direction / norm


def desired_motion_vector(heading: int, move: int, strafe: int) -> np.ndarray:
    vec = np.zeros(2, dtype=np.float32)
    if move == 1:
        vec += np.array(HEADING_VECTORS[heading % len(HEADING_VECTORS)], dtype=np.float32)
    elif move == 2:
        vec -= np.array(HEADING_VECTORS[heading % len(HEADING_VECTORS)], dtype=np.float32)

    if strafe == 1:
        vec += np.array(HEADING_VECTORS[(heading + 2) % len(HEADING_VECTORS)], dtype=np.float32)
    elif strafe == 2:
        vec += np.array(HEADING_VECTORS[(heading - 2) % len(HEADING_VECTORS)], dtype=np.float32)

    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    return vec / norm


def select_neighbor(nav_map: NavigationMap, region_id: int, desired_vec: np.ndarray) -> int:
    neighbors = nav_map.adjacency.get(region_id, [])
    if not neighbors:
        return region_id

    if float(np.linalg.norm(desired_vec)) <= 1e-6:
        return region_id

    current_distance = nav_map.distance(region_id)
    best_neighbor = region_id
    best_score = -float("inf")

    for neighbor in neighbors:
        direction = _neighbor_direction(region_id, neighbor)
        alignment = float(np.dot(desired_vec, direction))
        if alignment <= 0.05:
            continue
        progress = current_distance - nav_map.distance(neighbor)
        score = (alignment * 2.0) + (progress / max(nav_map.max_distance, 1.0))
        if score > best_score:
            best_score = score
            best_neighbor = neighbor

    return best_neighbor


def directional_progress(nav_map: NavigationMap, region_id: int, heading: int) -> np.ndarray:
    current_distance = nav_map.distance(region_id)
    progress_values: List[float] = []

    for candidate_heading in (heading, heading + 2, heading - 2, heading + 4):
        desired = np.array(HEADING_VECTORS[candidate_heading % len(HEADING_VECTORS)], dtype=np.float32)
        neighbor = select_neighbor(nav_map, region_id, desired)
        progress = current_distance - nav_map.distance(neighbor)
        progress_values.append(float(progress / max(nav_map.max_distance, 1.0)))

    return np.array(progress_values, dtype=np.float32)


def build_observation(
    nav_map: NavigationMap,
    region_id: int,
    heading: int,
    player_pos: Sequence[float],
    player_vel: Sequence[float],
    nearby_item_flags: Sequence[int],
    goal_progress: float,
) -> np.ndarray:
    heading_vec = HEADING_VECTORS[heading % len(HEADING_VECTORS)]
    nearby = [float(v) for v in nearby_item_flags[:4]]
    nearby.extend([0.0] * max(0, 4 - len(nearby)))
    local_progress = directional_progress(nav_map, region_id, heading)

    return np.array(
        [
            float(player_pos[0]) / 2048.0,
            float(player_pos[1]) / 2048.0,
            float(player_pos[2]) / 512.0,
            float(player_vel[0]) / 80.0,
            float(player_vel[1]) / 80.0,
            float(player_vel[2]) / 80.0,
            float(heading_vec[0]),
            float(heading_vec[1]),
            nearby[0],
            nearby[1],
            nearby[2],
            nearby[3],
            float(goal_progress),
            nav_map.distance(region_id) / max(nav_map.max_distance, 1.0),
            float(local_progress[0]),
            float(local_progress[1]),
            float(local_progress[2]),
            float(local_progress[3]),
            1.0 if region_id in nav_map.goal_regions else 0.0,
            1.0 if region_id in nav_map.item_regions else 0.0,
        ],
        dtype=np.float32,
    )
