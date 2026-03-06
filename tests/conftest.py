from __future__ import annotations

from pathlib import Path

import pytest

from quake_ai.data.collector import collect_from_demos


@pytest.fixture(scope="session")
def demo_dir() -> Path:
    return Path(__file__).parent / "demo_data"


@pytest.fixture(scope="session")
def map_fixture() -> Path:
    return Path(__file__).parent / "fixtures" / "e1m1_map.json"


@pytest.fixture(scope="session")
def collected_artifacts(tmp_path_factory: pytest.TempPathFactory, demo_dir: Path, map_fixture: Path):
    out_dir = tmp_path_factory.mktemp("collect")
    artifacts = collect_from_demos(map_id="E1M1", demo_dir=demo_dir, out_dir=out_dir, map_path=map_fixture)
    return artifacts
