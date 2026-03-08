"""Collection pipeline: demo replay -> telemetry/packet artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, TextIO, Tuple

from engine.adapter import DemoPlaybackHarness
from quake_ai.data.demo_classifier import classify_competitive
from quake_ai.data.demo_metadata import DemoMetadata, build_demo_metadata
from quake_ai.maps.features import write_map_features
from quake_ai.maps.world_model import build_world_model, write_world_model
from quake_ai.schemas import EpisodeSummaryV1, MapFeaturesV1, MapStateV2, PacketEventV1, TelemetryTickV1
from quake_ai.utils.io import write_json, write_ndjson
from quake_ai.data.world_stream import iter_world_ticks_from_demo_episode

from .demo import find_demo_files


@dataclass(slots=True)
class _EpisodeSummaryAccumulator:
    episode_id: str
    steps: int = 0
    goal_reached: bool = False
    items_collected: int = 0
    last_tick: int = 0
    return_value: float = 0.0
    previous_flags: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def add(self, telemetry: TelemetryTickV1) -> None:
        flags = telemetry.nearby_item_flags[:4] + [0] * max(0, 4 - len(telemetry.nearby_item_flags[:4]))
        self.items_collected += sum(1 for prev, cur in zip(self.previous_flags, flags) if prev == 0 and cur == 1)
        self.previous_flags = flags
        self.steps += 1
        self.goal_reached = bool(telemetry.goal_progress >= 1.0)
        self.last_tick = telemetry.tick
        self.return_value += float(telemetry.goal_progress)

    def finalize(self, tick_hz: int = 20) -> EpisodeSummaryV1:
        return EpisodeSummaryV1(
            episode_id=self.episode_id,
            steps=self.steps,
            goal_reached=self.goal_reached,
            items_collected=self.items_collected,
            time_to_goal=(self.last_tick / tick_hz) if self.steps else 0.0,
            return_value=self.return_value,
        )


@dataclass(slots=True)
class _ObservedMapAccumulator:
    region_ids: set[int] = field(default_factory=set)
    edge_counts: Dict[Tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    item_counts: Dict[int, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    spawn_points: List[List[float]] = field(default_factory=list)
    goal_regions: List[int] = field(default_factory=list)

    def begin_episode(self, telemetry: TelemetryTickV1) -> int:
        self.spawn_points.append(list(telemetry.player_pos))
        return int(telemetry.region_id)

    def add_tick(self, telemetry: TelemetryTickV1, previous_region: int | None) -> int:
        region_id = int(telemetry.region_id)
        self.region_ids.add(region_id)
        if telemetry.nearby_item_flags[:4]:
            labels = ("item_health", "item_armor", "item_ammo", "item_weapon")
            for idx, flag in enumerate(telemetry.nearby_item_flags[:4]):
                if int(flag) == 1:
                    self.item_counts[region_id][labels[idx]] += 1
        if previous_region is not None and region_id != previous_region:
            self.edge_counts[(previous_region, region_id)] += 1
            self.edge_counts[(region_id, previous_region)] += 1
        return region_id

    def end_episode(self, last_region: int | None) -> None:
        if last_region is not None:
            self.goal_regions.append(int(last_region))


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


def _map_features_from_observed(map_id: str, observed: _ObservedMapAccumulator) -> Tuple[List[MapFeaturesV1], Dict[str, object]]:
    region_ids = sorted(observed.region_ids)
    if not region_ids:
        region_ids = [0]

    edges = [[src, dst] for src, dst in sorted(observed.edge_counts.keys())]
    edges = _fallback_edges(region_ids, edges)

    item_nodes = []
    for region_id, counts in sorted(observed.item_counts.items()):
        classname = max(sorted(counts.keys()), key=lambda key: counts[key])
        item_nodes.append(
            {
                "classname": classname,
                "origin": _region_center(region_id),
                "region_id": region_id,
                "observations": int(sum(counts.values())),
            }
        )

    spawn_points = list(observed.spawn_points)
    goal_regions = sorted(set(observed.goal_regions))
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


def _write_ndjson_row(handle: TextIO, row: Mapping[str, object]) -> None:
    handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _stream_demo_episode(
    *,
    episode: object,
    map_state: MapStateV2 | None,
    telemetry_handle: TextIO,
    packet_handle: TextIO,
    world_handle: TextIO | None,
    observed: _ObservedMapAccumulator | None,
) -> EpisodeSummaryV1:
    summary = _EpisodeSummaryAccumulator(episode_id=str(getattr(episode, "episode_id")))
    previous_region: int | None = None
    last_region: int | None = None

    harness = DemoPlaybackHarness(map_id=str(getattr(episode, "map_id", "E1M1")))
    for telemetry, packet in harness.replay_episode(episode):
        if previous_region is None and observed is not None:
            previous_region = observed.begin_episode(telemetry)
        if observed is not None:
            last_region = observed.add_tick(telemetry, previous_region)
            previous_region = last_region
        summary.add(telemetry)
        _write_ndjson_row(telemetry_handle, telemetry.to_dict())
        _write_ndjson_row(packet_handle, packet.to_dict())

    if observed is not None:
        observed.end_episode(last_region)

    if map_state is not None and world_handle is not None:
        for world_tick in iter_world_ticks_from_demo_episode(episode, map_state):
            _write_ndjson_row(world_handle, world_tick.to_dict())

    return summary.finalize()


def collect_from_demos(map_id: str, demo_dir: str | Path, out_dir: str | Path, map_path: str | Path | None = None) -> Dict[str, str]:
    demos = find_demo_files(demo_dir)
    map_state = build_world_model(map_path, map_id=map_id) if map_path is not None else None

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    telemetry_path = output / "telemetry.ndjson"
    packets_path = output / "packets.ndjson"
    summaries_path = output / "episode_summaries.ndjson"
    failures_path = output / "replay_failures.ndjson"
    world_ticks_path = output / "world_ticks.ndjson"
    metadata_path = output / "demo_metadata.ndjson"

    harness = DemoPlaybackHarness(map_id=map_id)
    observed = _ObservedMapAccumulator() if map_path is None else None
    summaries: List[EpisodeSummaryV1] = []
    failures: List[Dict[str, str]] = []
    metadata_rows: List[Dict[str, object]] = []
    saw_telemetry = False

    with (
        telemetry_path.open("w", encoding="utf-8") as telemetry_handle,
        packets_path.open("w", encoding="utf-8") as packet_handle,
        (world_ticks_path.open("w", encoding="utf-8") if map_state is not None else nullcontext(None)) as world_handle,
    ):
        for demo_path in demos:
            try:
                episode = harness.load_episode(demo_path)
                meta = classify_competitive(build_demo_metadata(episode, source_path=demo_path))
                summaries.append(
                    _stream_demo_episode(
                        episode=episode,
                        map_state=map_state,
                        telemetry_handle=telemetry_handle,
                        packet_handle=packet_handle,
                        world_handle=world_handle,
                        observed=observed,
                    )
                )
                metadata_rows.append(meta.to_dict())
                saw_telemetry = True
            except Exception as exc:
                failures.append({"demo_path": str(demo_path), "error": str(exc)})
                stub_meta = DemoMetadata(
                    episode_id=Path(demo_path).stem,
                    map_id=map_id,
                    source_path=str(demo_path),
                )
                fallback = classify_competitive(stub_meta)
                metadata_rows.append(fallback.to_dict())

    if not saw_telemetry and not metadata_rows:
        raise RuntimeError("No telemetry rows were collected from the provided demos")

    write_ndjson(summaries_path, (row.to_dict() for row in summaries))
    write_ndjson(failures_path, failures)
    write_ndjson(metadata_path, metadata_rows)

    if map_path is None:
        assert observed is not None
        map_features, map_summary = _map_features_from_observed(map_id=map_id, observed=observed)
        write_json(output / "observed_map.json", map_summary)
    else:
        from quake_ai.maps.features import build_map_features

        map_features = build_map_features(map_path, map_id=map_id)
    map_features_path = output / "map_features.json"
    write_map_features(map_features_path, map_features)

    artifacts = {
        "telemetry": str(telemetry_path),
        "packets": str(packets_path),
        "summaries": str(summaries_path),
        "map_features": str(map_features_path),
        "failures": str(failures_path),
        "metadata": str(metadata_path),
    }
    if map_state is not None:
        map_state_path = output / "world_map.json"
        write_world_model(map_state_path, map_state)
        artifacts["world_map"] = str(map_state_path)
        artifacts["world_ticks"] = str(world_ticks_path)
    return artifacts


def load_packets(path: str | Path) -> List[PacketEventV1]:
    from quake_ai.utils.io import read_ndjson

    return [PacketEventV1.from_dict(row) for row in read_ndjson(path)]
