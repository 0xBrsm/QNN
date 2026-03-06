from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from engine.native_bridge import NativeEngineProcess, NativeQuakeAdapter


def _build_native_stub(tmp_path: Path) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("cc is required for native bridge tests")

    source = Path(__file__).resolve().parents[1] / "engine" / "native_stub.c"
    binary = tmp_path / "native_stub"
    subprocess.run([compiler, "-O2", "-std=c99", "-o", str(binary), str(source)], check=True)
    return binary


def test_native_engine_process_round_trip(tmp_path: Path) -> None:
    binary = _build_native_stub(tmp_path)
    with NativeEngineProcess(executable=binary, map_id="E1M1", fixed_tick_hz=30) as proc:
        hello = proc.start()
        assert hello["server"] == "native-stub"
        assert int(hello["tick_hz"]) == 30

        reset = proc.reset(seed=7)
        assert len(reset["obs"]) == 20
        assert int(reset["info"]["seed"]) == 7

        response = proc.step({"move": 1, "strafe": 0, "turn": 0, "use": 0})
        assert not bool(response["done"])
        assert float(response["reward"]) > 0.0


def test_native_quake_adapter_uses_process_boundary(tmp_path: Path) -> None:
    binary = _build_native_stub(tmp_path)
    adapter = NativeQuakeAdapter(executable=binary, map_id="E1M1", fixed_tick_hz=25)
    try:
        obs = adapter.reset(seed=11)
        assert obs.shape == (20,)

        done = False
        steps = 0
        while not done:
            obs, reward, done, info = adapter.step({"move": 1, "strafe": 0, "turn": 0, "use": int(steps >= 2)})
            steps += 1
        assert reward > 0.0
        assert bool(info["goal_reached"])
    finally:
        adapter.close()
