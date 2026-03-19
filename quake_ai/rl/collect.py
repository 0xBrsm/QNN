"""Collect QTOK token ticks from demo files using the C demo worker."""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from demo import DemoProbe, probe_demo
from engine.bridge import NativeTokenProcess
from engine.token_protocol import write_token_ticks_file
from quake_ai.utils.io import write_json, write_ndjson


_IDLE_ACTION = {
    "move": 0,
    "strafe": 0,
    "look_yaw": 12,
    "look_pitch": 12,
    "fire": 0,
    "jump": 0,
    "weapon": 0,
}


@dataclass(slots=True)
class CollectConfig:
    demo_worker_binary: str
    demo_dir: str
    output_dir: str
    map_id: str
    fixed_tick_hz: int = 0
    asset_root: str = ""
    max_ticks_per_demo: int | None = None
    parallel_workers: int = 0


@dataclass(slots=True)
class CollectResult:
    token_ticks_path: str
    map_state_path: str
    map_states_path: str
    metadata_path: str
    missing_demos_path: str
    source_summary_path: str
    demos_processed: int
    demos_failed: int
    total_ticks: int


@dataclass(slots=True)
class _DemoEntry:
    index: int
    demo_file: Path
    probe: DemoProbe


@dataclass(slots=True)
class _CollectShard:
    entries: list[_DemoEntry]


@dataclass(slots=True)
class _CollectedDemo:
    index: int
    episode_id: str
    map_id: str
    source_path: str
    token_ticks_path: str
    tick_count: int


@dataclass(slots=True)
class _CollectShardResult:
    collected_demos: list[_CollectedDemo]
    map_states: Dict[str, Dict[str, Any]]
    missing_rows: list[tuple[int, Dict[str, Any]]]
    demos_processed: int
    demos_failed: int
    total_ticks: int


def _stage_demos_into_basedir(demo_files: List[Path], asset_root: Path) -> List[Path]:
    """Symlink demo files into the basedir's id1/ so the engine can find them.

    Returns link paths for cleanup.
    """
    gamedir = asset_root / "id1"
    staged: List[Path] = []
    for demo in demo_files:
        link = gamedir / demo.name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(demo.resolve())
        except OSError:
            shutil.copy2(demo, link)
        staged.append(link)
    return staged


def _unstage_demos(staged: List[Path]) -> None:
    for link in staged:
        try:
            link.unlink(missing_ok=True)
        except OSError:
            pass


def _corpus_map_state(corpus_map_id: str, map_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if len(map_states) == 1:
        return next(iter(map_states.values()))
    return {
        "map_id": corpus_map_id,
        "regions": [],
        "static_objects": [],
        "metadata": {
            "map_ids": sorted(map_states.keys()),
            "map_state_count": len(map_states),
            "source": "multi_map_demo_corpus",
        },
    }


def _resolve_parallel_workers(config: CollectConfig, demo_count: int) -> int:
    if demo_count <= 1:
        return 1
    env_workers = int(os.getenv("QUAKE_AI_DEMO_COLLECT_WORKERS", "0") or "0")
    requested = int(config.parallel_workers or 0)
    if requested <= 0:
        requested = env_workers
    if requested <= 0:
        cpu_count = os.cpu_count() or 1
        requested = max(1, cpu_count * 3 // 4)
    return max(1, min(int(requested), demo_count))


def _build_collect_shards(entries: Sequence[_DemoEntry], worker_count: int) -> list[_CollectShard]:
    if worker_count <= 1 or len(entries) <= 1:
        return [_CollectShard(entries=list(entries))]

    # When we have enough workers, give each demo its own shard for
    # maximum parallelism.  The BSP reload cost per shard is negligible
    # compared to the time saved from parallel processing.
    if worker_count >= len(entries):
        return [_CollectShard(entries=[entry]) for entry in entries]

    # Otherwise, round-robin entries across shards, grouping same-map
    # demos together where possible to reuse the worker process.
    groups_by_map: Dict[str, list[_DemoEntry]] = {}
    for entry in entries:
        groups_by_map.setdefault(entry.probe.map_id, []).append(entry)

    shards: list[list[_DemoEntry]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for group in sorted(groups_by_map.values(), key=len, reverse=True):
        for entry in group:
            shard_idx = min(range(worker_count), key=lambda idx: (loads[idx], idx))
            shards[shard_idx].append(entry)
            loads[shard_idx] += 1
    return [_CollectShard(entries=sorted(shard, key=lambda e: e.index)) for shard in shards if shard]


def _collect_episode_ticks(
    proc: NativeTokenProcess,
    *,
    demo_name: str,
    max_ticks: int | None,
) -> list[Any]:
    if hasattr(proc, "collect_episode"):
        ticks = list(proc.collect_episode(seed=0, options={"demo_path": demo_name}, idle_action=_IDLE_ACTION))
        if not ticks:
            raise RuntimeError(f"Demo produced no ticks: {demo_name}")
        if max_ticks is not None and len(ticks) > max_ticks:
            raise RuntimeError(f"Demo exceeded max_ticks_per_demo={max_ticks} before completion: {demo_name}")
        return ticks

    reset_tick = proc.reset(seed=0, options={"demo_path": demo_name})
    ticks = [reset_tick]
    while not ticks[-1].done:
        if max_ticks is not None and len(ticks) >= max_ticks:
            raise RuntimeError(f"Demo exceeded max_ticks_per_demo={max_ticks} before completion: {demo_name}")
        ticks.append(proc.step(_IDLE_ACTION))
    return ticks


def _collect_shard(
    shard: _CollectShard,
    *,
    config: CollectConfig,
    env: Dict[str, str],
    temp_root: Path,
    max_ticks: int | None,
) -> _CollectShardResult:
    collected_demos: list[_CollectedDemo] = []
    map_states: Dict[str, Dict[str, Any]] = {}
    missing_rows: list[tuple[int, Dict[str, Any]]] = []
    demos_processed = 0
    demos_failed = 0
    total_ticks = 0
    current_proc: NativeTokenProcess | None = None
    current_map_id = ""

    def _start_worker(map_id: str) -> None:
        nonlocal current_proc, current_map_id
        if current_proc is not None:
            try:
                current_proc.shutdown()
            except Exception:
                pass
        current_proc = NativeTokenProcess(
            executable=config.demo_worker_binary,
            map_id=map_id,
            fixed_tick_hz=config.fixed_tick_hz,
            env=env,
        )
        current_proc.start()
        current_map_id = map_id
        if current_proc.map_state is not None:
            map_states[map_id] = current_proc.map_state.to_dict()

    try:
        for entry in shard.entries:
            demo_name = entry.demo_file.name
            try:
                if current_proc is None or entry.probe.map_id != current_map_id:
                    _start_worker(entry.probe.map_id)
                if current_proc is None:
                    raise RuntimeError(f"Failed to start demo worker for {entry.probe.map_id}")

                episode_ticks = _collect_episode_ticks(
                    current_proc,
                    demo_name=demo_name,
                    max_ticks=max_ticks,
                )
                tick_count = len(episode_ticks)
                token_ticks_path = temp_root / f"{entry.index:05d}.bin"
                write_token_ticks_file(str(token_ticks_path), episode_ticks)
                token_ticks_path.touch(exist_ok=True)
                collected_demos.append(
                    _CollectedDemo(
                        index=entry.index,
                        episode_id=entry.probe.episode_id,
                        map_id=entry.probe.map_id,
                        source_path=entry.probe.source_path,
                        token_ticks_path=str(token_ticks_path),
                        tick_count=tick_count,
                    )
                )
                demos_processed += 1
                total_ticks += tick_count
                print(f"  [collect] {demo_name}: {tick_count} ticks")
            except Exception as exc:
                demos_failed += 1
                missing_rows.append(
                    (
                        entry.index,
                        {
                            "episode_id": entry.probe.episode_id,
                            "map_id": entry.probe.map_id,
                            "source_path": entry.probe.source_path,
                            "reason": str(exc),
                        },
                    )
                )
                print(f"  [collect] skipping {demo_name}: {exc}")
                print("  [collect] restarting worker after crash...")
                try:
                    if current_proc is not None:
                        current_proc.shutdown()
                except Exception:
                    pass
                current_proc = None
                current_map_id = ""
    finally:
        try:
            if current_proc is not None:
                current_proc.shutdown()
        except Exception:
            pass

    return _CollectShardResult(
        collected_demos=collected_demos,
        map_states=map_states,
        missing_rows=missing_rows,
        demos_processed=demos_processed,
        demos_failed=demos_failed,
        total_ticks=total_ticks,
    )


def _append_token_files(output_path: Path, collected_demos: Sequence[_CollectedDemo]) -> None:
    with output_path.open("wb") as output_file:
        for row in collected_demos:
            with Path(row.token_ticks_path).open("rb") as input_file:
                shutil.copyfileobj(input_file, output_file)


def collect_demo_tokens(config: CollectConfig) -> CollectResult:
    """Play back .dem files through the demo worker and save QTOK binary output."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    demo_dir = Path(config.demo_dir)
    demo_files = sorted(p for p in demo_dir.iterdir() if p.suffix.lower() == ".dem" and p.is_file())
    if not demo_files:
        raise RuntimeError(f"No .dem files found in {demo_dir}")

    asset_root = Path(config.asset_root) if config.asset_root else None
    env: Dict[str, str] = {}
    if asset_root:
        env["QUAKE_BASEDIR"] = str(asset_root)

    staged: List[Path] = []
    if asset_root:
        staged = _stage_demos_into_basedir(demo_files, asset_root)

    probes = [probe_demo(path, fallback_map_id=config.map_id) for path in demo_files]
    entries = [_DemoEntry(index=index, demo_file=demo_file, probe=probe) for index, (demo_file, probe) in enumerate(zip(demo_files, probes))]
    parallel_workers = _resolve_parallel_workers(config, len(entries))
    shards = _build_collect_shards(entries, parallel_workers)
    max_ticks = int(config.max_ticks_per_demo) if config.max_ticks_per_demo and int(config.max_ticks_per_demo) > 0 else None

    demos_processed = 0
    demos_failed = 0
    total_ticks = 0
    map_states: Dict[str, Dict[str, Any]] = {}
    collected_demos: list[_CollectedDemo] = []
    missing_rows_indexed: list[tuple[int, Dict[str, Any]]] = []

    try:
        print(f"  [collect] using {len(shards)} worker shard(s) for {len(entries)} demos")
        with tempfile.TemporaryDirectory(dir=output, prefix="collect_tmp_") as temp_dir:
            temp_root = Path(temp_dir)
            if len(shards) == 1:
                shard_results = [
                    _collect_shard(
                        shards[0],
                        config=config,
                        env=env,
                        temp_root=temp_root,
                        max_ticks=max_ticks,
                    )
                ]
            else:
                with ThreadPoolExecutor(max_workers=len(shards)) as executor:
                    futures = [
                        executor.submit(
                            _collect_shard,
                            shard,
                            config=config,
                            env=env,
                            temp_root=temp_root,
                            max_ticks=max_ticks,
                        )
                        for shard in shards
                    ]
                    shard_results = [future.result() for future in futures]

            for shard_result in shard_results:
                demos_processed += shard_result.demos_processed
                demos_failed += shard_result.demos_failed
                total_ticks += shard_result.total_ticks
                map_states.update(shard_result.map_states)
                collected_demos.extend(shard_result.collected_demos)
                missing_rows_indexed.extend(shard_result.missing_rows)

            collected_demos.sort(key=lambda row: row.index)
            missing_rows_indexed.sort(key=lambda item: item[0])

            if not collected_demos:
                raise RuntimeError("No ticks collected from any demo file")

            token_ticks_path = output / "token_ticks.bin"
            _append_token_files(token_ticks_path, collected_demos)
    finally:
        _unstage_demos(staged)

    corpus_map_id = output.parent.name or demo_dir.name or str(config.map_id)
    token_ticks_path = str(output / "token_ticks.bin")
    map_state_path = str(output / "world_map.json")
    map_states_path = str(output / "map_states.json")
    metadata_path = str(output / "demo_metadata.ndjson")
    missing_demos_path = str(output / "missing_demos.ndjson")
    source_summary_path = str(output / "source_summary.json")

    metadata_rows = [
        {
            "episode_id": row.episode_id,
            "map_id": row.map_id,
            "source_path": row.source_path,
            "tick_count": row.tick_count,
        }
        for row in collected_demos
    ]
    missing_rows = [row for _, row in missing_rows_indexed]

    write_json(map_state_path, _corpus_map_state(corpus_map_id, map_states))
    write_json(map_states_path, {"schema_version": "v1", "map_states": map_states})
    write_ndjson(metadata_path, metadata_rows)
    write_ndjson(missing_demos_path, missing_rows)
    write_json(
        source_summary_path,
        {
            "demo_dir": str(demo_dir),
            "demo_files": len(demo_files),
            "demos_processed": demos_processed,
            "demos_failed": demos_failed,
            "map_ids": sorted(map_states.keys()),
            "missing": missing_rows,
            "missing_or_failed_demos": len(missing_rows),
            "per_demo_ticks": [
                {
                    "demo": Path(row["source_path"]).name,
                    "episode_id": row["episode_id"],
                    "map_id": row["map_id"],
                    "ticks": row["tick_count"],
                }
                for row in metadata_rows
            ],
        },
    )

    return CollectResult(
        token_ticks_path=token_ticks_path,
        map_state_path=map_state_path,
        map_states_path=map_states_path,
        metadata_path=metadata_path,
        missing_demos_path=missing_demos_path,
        source_summary_path=source_summary_path,
        demos_processed=demos_processed,
        demos_failed=demos_failed,
        total_ticks=total_ticks,
    )
