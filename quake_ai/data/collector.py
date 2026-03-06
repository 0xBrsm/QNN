"""Collection pipeline: demo replay -> telemetry/packet artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from engine.adapter import DemoPlaybackHarness
from quake_ai.maps.features import write_map_features
from quake_ai.schemas import EpisodeSummaryV1, MapFeaturesV1, PacketEventV1, TelemetryTickV1
from quake_ai.utils.io import write_json, write_ndjson

from .demo import find_demo_files


def _summaries_from_telemetry(rows: List[TelemetryTickV1], tick_hz: int = 20) -> List[EpisodeSummaryV1]:
    by_episode: Dict[str, List[TelemetryTickV1]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_id].append(row)

    summaries: List[EpisodeSummaryV1] = []
    for episode_id, ticks in sorted(by_episode.items()):
        ticks.sort(key=lambda x: x.tick)
        goal_reached = bool(ticks and ticks[-1].goal_progress >= 1.0)
        items_collected = 0
        previous_flags: List[int] = [0, 0, 0, 0]
        for tick in ticks:
            current_flags = tick.nearby_item_flags[:4] + [0] * max(0, 4 - len(tick.nearby_item_flags[:4]))
            items_collected += sum(1 for prev, cur in zip(previous_flags, current_flags) if prev == 0 and cur == 1)
            previous_flags = current_flags
        time_to_goal = (ticks[-1].tick / tick_hz) if ticks else 0.0
        total_return = sum(float(t.goal_progress) for t in ticks)
        summaries.append(
            EpisodeSummaryV1(
                episode_id=episode_id,
                steps=len(ticks),
                goal_reached=goal_reached,
                items_collected=items_collected,
                time_to_goal=time_to_goal,
                return_value=total_return,
            )
        )
    return summaries


def _region_center(region_id: int) -> List[float]:
    gx = region_id // 2048 - 1024
    gy = region_id % 2048 - 1024
    return [gx * 256.0, gy * 256.0, 0.0]


def _distance_to_goal(region_ids: Iterable[int], edges: Iterable[Tuple[int, int]], goal_regions: Iterable[int]) -> Dict[int, float]:
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges:
        adjacency[src].append(dst)

    queue: List[int] = []
    distances = {int(region_id): float("inf") for region_id in region_ids}
    for goal_region in sorted(set(int(region_id) for region_id in goal_regions)):
        if goal_region in distances:
            distances[goal_region] = 0.0
            queue.append(goal_region)

    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for neighbor in adjacency.get(current, []):
            if distances[neighbor] > distances[current] + 1.0:
                distances[neighbor] = distances[current] + 1.0
                queue.append(neighbor)

    max_finite = max((value for value in distances.values() if value < float("inf")), default=0.0)
    for region_id, value in list(distances.items()):
        if value == float("inf"):
            distances[region_id] = max_finite + 1.0
    return distances


def _connect_observed_regions(rows: List[TelemetryTickV1]) -> Tuple[List[List[int]], Dict[int, Dict[str, int]], List[List[float]], List[int]]:
    by_episode: Dict[str, List[TelemetryTickV1]] = defaultdict(list)
    for row in rows:
        by_episode[row.episode_id].append(row)

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    item_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    spawn_points: List[List[float]] = []
    goal_regions: List[int] = []

    for ticks in by_episode.values():
        ticks.sort(key=lambda row: row.tick)
        if not ticks:
            continue

        spawn_points.append(list(ticks[0].player_pos))
        goal_regions.append(int(ticks[-1].region_id))

        previous_region = int(ticks[0].region_id)
        for tick in ticks:
            region_id = int(tick.region_id)
            if tick.nearby_item_flags[:4]:
                labels = ("item_health", "item_armor", "item_ammo", "item_weapon")
                for idx, flag in enumerate(tick.nearby_item_flags[:4]):
                    if int(flag) == 1:
                        item_counts[region_id][labels[idx]] += 1
            if region_id != previous_region:
                edge_counts[(previous_region, region_id)] += 1
                edge_counts[(region_id, previous_region)] += 1
                previous_region = region_id

    return (
        [[src, dst] for src, dst in sorted(edge_counts.keys())],
        item_counts,
        spawn_points,
        sorted(set(goal_regions)),
    )


def _fallback_edges(region_ids: List[int], existing_edges: List[List[int]], k: int = 2) -> List[List[int]]:
    adjacency: Dict[int, set[int]] = defaultdict(set)
    for src, dst in existing_edges:
        adjacency[int(src)].add(int(dst))

    def _grid_xy(region_id: int) -> Tuple[int, int]:
        return region_id // 2048 - 1024, region_id % 2048 - 1024

    for src in region_ids:
        if adjacency.get(src):
            continue
        sx, sy = _grid_xy(src)
        ranked: List[Tuple[int, int]] = []
        for dst in region_ids:
            if dst == src:
                continue
            dx, dy = _grid_xy(dst)
            ranked.append((abs(sx - dx) + abs(sy - dy), dst))
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        for _, dst in ranked[:k]:
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    return [[src, dst] for src, neighbors in sorted(adjacency.items()) for dst in sorted(neighbors)]


def _map_features_from_telemetry(map_id: str, rows: List[TelemetryTickV1]) -> Tuple[List[MapFeaturesV1], Dict[str, object]]:
    region_ids = sorted({int(row.region_id) for row in rows})
    if not region_ids:
        region_ids = [0]

    edges, item_counts, spawn_points, goal_regions = _connect_observed_regions(rows)
    edges = _fallback_edges(region_ids, edges)

    item_nodes = []
    for region_id, counts in sorted(item_counts.items()):
        classname = max(sorted(counts.keys()), key=lambda key: counts[key])
        item_nodes.append(
            {
                "classname": classname,
                "origin": _region_center(region_id),
                "region_id": region_id,
                "observations": int(sum(counts.values())),
            }
        )

    if not spawn_points:
        spawn_points = [_region_center(region_ids[0])]
    if not goal_regions:
        goal_regions = [region_ids[-1]]

    distances = _distance_to_goal(region_ids, [(int(src), int(dst)) for src, dst in edges], goal_regions)
    records = [
        MapFeaturesV1(
            map_id=map_id,
            region_id=region_id,
            spawn_points=spawn_points,
            item_nodes=item_nodes,
            connectivity_edges=edges,
            goal_nodes=goal_regions,
            distance_to_goal=distances[region_id],
        )
        for region_id in region_ids
    ]
    summary = {
        "map_id": map_id,
        "region_count": len(region_ids),
        "edge_count": len(edges),
        "spawn_count": len(spawn_points),
        "goal_region_count": len(goal_regions),
        "item_region_count": len(item_nodes),
        "goal_regions": goal_regions,
    }
    return records, summary


def _replay_demo_files(demos: List[Path], map_id: str) -> Tuple[List[TelemetryTickV1], List[PacketEventV1], List[Dict[str, str]]]:
    harness = DemoPlaybackHarness(map_id=map_id)
    telemetry_rows: List[TelemetryTickV1] = []
    packet_rows: List[PacketEventV1] = []
    failures: List[Dict[str, str]] = []

    for demo_path in demos:
        try:
            for telemetry, packet in harness.replay(demo_path):
                telemetry_rows.append(telemetry)
                packet_rows.append(packet)
        except Exception as exc:
            failures.append({"demo_path": str(demo_path), "error": str(exc)})

    return telemetry_rows, packet_rows, failures


def collect_from_demos(map_id: str, demo_dir: str | Path, out_dir: str | Path, map_path: str | Path | None = None) -> Dict[str, str]:
    demos = find_demo_files(demo_dir)
    telemetry_rows, packet_rows, failures = _replay_demo_files(demos, map_id=map_id)
    if not telemetry_rows:
        raise RuntimeError("No telemetry rows were collected from the provided demos")
    summaries = _summaries_from_telemetry(telemetry_rows)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    telemetry_path = output / "telemetry.ndjson"
    packets_path = output / "packets.ndjson"
    summaries_path = output / "episode_summaries.ndjson"
    failures_path = output / "replay_failures.ndjson"

    write_ndjson(telemetry_path, (row.to_dict() for row in telemetry_rows))
    write_ndjson(packets_path, (row.to_dict() for row in packet_rows))
    write_ndjson(summaries_path, (row.to_dict() for row in summaries))
    write_ndjson(failures_path, failures)

    if map_path is None:
        map_features, map_summary = _map_features_from_telemetry(map_id=map_id, rows=telemetry_rows)
        write_json(output / "observed_map.json", map_summary)
    else:
        from quake_ai.maps.features import build_map_features

        map_features = build_map_features(map_path, map_id=map_id)
    map_features_path = output / "map_features.json"
    write_map_features(map_features_path, map_features)

    return {
        "telemetry": str(telemetry_path),
        "packets": str(packets_path),
        "summaries": str(summaries_path),
        "map_features": str(map_features_path),
        "failures": str(failures_path),
    }


def load_packets(path: str | Path) -> List[PacketEventV1]:
    from quake_ai.utils.io import read_ndjson

    return [PacketEventV1.from_dict(row) for row in read_ndjson(path)]
