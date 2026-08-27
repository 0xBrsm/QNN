"""Benchmark static-world atlas query cost inside the demo worker.

The timed region is C-side CPU time around direction construction, carved-face
ray queries, and depth quantization. Demo reset, JSON formatting, worker IPC,
and dynamic mover collection are outside the timed region.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from qnn.bc.probe_atlas import DemoQueryWorker
from qnn.utils.io import write_json
from qnn.diag.static_map_memory import (
    canonical_map_name,
    first_demo_per_map,
    load_static_records,
)


def benchmark(
    *, worker_path: str, demo_dir: Path, manifest: Path, asset_root: Path,
    sidecars: list[Path], maps: list[str], yaw_counts: list[int],
    iterations: int, repeats: int, tick_hz: int, output: Path,
) -> dict[str, Any]:
    demos = first_demo_per_map(manifest)
    records = load_static_records(sidecars)
    game_dir = os.path.relpath(demo_dir.absolute(), asset_root.absolute())
    results: dict[str, Any] = {}
    for map_name in maps:
        if map_name not in demos or map_name not in records:
            raise ValueError(f"{map_name}: missing representative demo or sidecar")
        origin = records[map_name][0]["origin"]
        worker = DemoQueryWorker(worker_path, game_dir, asset_root, tick_hz)
        try:
            map_results: dict[str, Any] = {}
            for yaw_count in yaw_counts:
                samples = []
                checksums = []
                for _ in range(repeats):
                    result = worker.nav_query(
                        demos[map_name],
                        "atlas_bench",
                        x=origin[0],
                        y=origin[1],
                        z=origin[2],
                        yaw_count=yaw_count,
                        iterations=iterations,
                    )
                    samples.append(float(result["microseconds_per_atlas"]))
                    checksums.append(int(result["checksum"]))
                values = np.asarray(samples, dtype=np.float64)
                median = float(np.median(values))
                map_results[str(yaw_count)] = {
                    "rays": 11 * yaw_count,
                    "repeats": repeats,
                    "iterations_per_repeat": iterations,
                    "microseconds_per_atlas": {
                        "samples": samples,
                        "min": float(np.min(values)),
                        "p50": median,
                        "p95": float(np.percentile(values, 95)),
                        "max": float(np.max(values)),
                    },
                    "nanoseconds_per_ray_p50": 1000.0 * median / (11 * yaw_count),
                    "single_core_fraction_at_20hz": median * 20.0 / 1e6,
                    "checksums": checksums,
                }
                print(
                    f"{map_name} {yaw_count}x11: {median:.2f} us/atlas, "
                    f"{map_results[str(yaw_count)]['nanoseconds_per_ray_p50']:.1f} ns/ray"
                )
            results[map_name] = map_results
        finally:
            worker.close()

    report = {
        "schema": 1,
        "scope": "static_world_cpu_time_excludes_ipc_reset_json_and_movers",
        "maps": results,
    }
    write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path("assets"))
    parser.add_argument("--sidecars", type=Path, nargs="+", required=True)
    parser.add_argument("--maps", nargs="+", default=["dm2", "dm4", "dm6"])
    parser.add_argument("--yaw-counts", type=int, nargs="+", default=[72, 36, 24])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tick-hz", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    benchmark(
        worker_path=args.worker,
        demo_dir=args.demo_dir,
        manifest=args.manifest,
        asset_root=args.asset_root,
        sidecars=args.sidecars,
        maps=[canonical_map_name(value) for value in args.maps],
        yaw_counts=args.yaw_counts,
        iterations=args.iterations,
        repeats=args.repeats,
        tick_hz=args.tick_hz,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
