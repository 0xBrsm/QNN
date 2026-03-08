from __future__ import annotations

import os
import subprocess
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


@pytest.fixture(scope="session")
def native_worker_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    binary = tmp_path_factory.mktemp("native_worker") / "native_stub_worker"
    build_script = Path(__file__).resolve().parents[1] / "engine" / "build" / "build_stub_worker.sh"
    subprocess.run(["bash", str(build_script), str(binary)], check=True)
    return binary


def _looks_like_quake_basedir(path: Path) -> bool:
    id1_dir = path / "id1"
    if not id1_dir.is_dir():
        return False
    pak_names = ("PAK0.PAK", "PAK1.PAK", "pak0.pak", "pak1.pak")
    return any((id1_dir / name).exists() for name in pak_names)


@pytest.fixture(scope="session")
def quake_assets_dir() -> Path:
    candidates: list[Path] = []
    env_basedir = os.environ.get("QUAKE_BASEDIR", "").strip()
    if env_basedir:
        candidates.append(Path(env_basedir))
    candidates.extend(
        [
            Path("/assets"),
            Path(__file__).resolve().parents[2] / "assets",
        ]
    )

    for candidate in candidates:
        if _looks_like_quake_basedir(candidate):
            return candidate

    pytest.skip("Quake assets not available; set QUAKE_BASEDIR or provide /assets/id1")


@pytest.fixture(scope="session")
def quake_worker_binary(quake_assets_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    del quake_assets_dir

    binary = tmp_path_factory.mktemp("quake_worker") / "quake_worker"
    build_script = Path(__file__).resolve().parents[1] / "engine" / "build" / "build_quake_worker.sh"
    subprocess.run(["bash", str(build_script), str(binary)], check=True)
    return binary
