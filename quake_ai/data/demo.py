"""Demo discovery and deterministic replay helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from engine.adapter import DemoPlaybackHarness
from quake_ai.schemas import PacketEventV1, TelemetryTickV1


def find_demo_files(demo_dir: str | Path) -> List[Path]:
    root = Path(demo_dir)
    if not root.exists():
        raise FileNotFoundError(f"Demo directory does not exist: {demo_dir}")
    demos = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".dem")
    if not demos:
        raise FileNotFoundError(f"No .dem files found under {demo_dir}")
    return demos


def replay_demos(demo_paths: Iterable[str | Path], map_id: str) -> List[Tuple[TelemetryTickV1, PacketEventV1]]:
    harness = DemoPlaybackHarness(map_id=map_id)
    out: List[Tuple[TelemetryTickV1, PacketEventV1]] = []
    for path in sorted(str(p) for p in demo_paths):
        out.extend(list(harness.replay(path)))
    return out
