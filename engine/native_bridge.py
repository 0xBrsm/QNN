"""Native engine process bridge for engine-backed adapters."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Mapping

import numpy as np


class NativeEngineError(RuntimeError):
    pass


class NativeEngineProcess:
    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        fixed_tick_hz: int = 20,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = str(executable)
        self.map_id = map_id
        self.fixed_tick_hz = fixed_tick_hz
        self.workdir = None if workdir is None else str(workdir)
        self.env = dict(env or {})
        self.proc: subprocess.Popen[str] | None = None

    def _stderr_tail(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read()
        except Exception:
            return ""

    def _ensure_running(self) -> subprocess.Popen[str]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise NativeEngineError("Native engine process is not running")
        return self.proc

    def _request(self, payload: Mapping[str, object]) -> Dict[str, object]:
        proc = self._ensure_running()
        proc.stdin.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            stderr = self._stderr_tail().strip()
            raise NativeEngineError(f"Native engine terminated unexpectedly: {stderr or 'no response'}")

        response = json.loads(line)
        if not bool(response.get("ok", False)):
            raise NativeEngineError(str(response.get("error", "unknown native engine error")))
        return dict(response)

    def start(self) -> Dict[str, object]:
        if self.proc is not None:
            return self._request({"op": "hello", "map_id": self.map_id, "tick_hz": self.fixed_tick_hz})

        env = os.environ.copy()
        env.update(self.env)
        self.proc = subprocess.Popen(
            [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.workdir,
            env=env,
        )
        return self._request({"op": "hello", "map_id": self.map_id, "tick_hz": self.fixed_tick_hz})

    def reset(self, seed: int | None = None) -> Dict[str, object]:
        if self.proc is None:
            self.start()
        return self._request({"op": "reset", "seed": seed if seed is not None else -1})

    def step(self, action: Mapping[str, int]) -> Dict[str, object]:
        if self.proc is None:
            self.start()
        return self._request({"op": "step", "action": {str(k): int(v) for k, v in action.items()}})

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            self._request({"op": "shutdown"})
        except Exception:
            pass

        if self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()
        self.proc.wait(timeout=5)
        self.proc = None

    def __enter__(self) -> "NativeEngineProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.shutdown()


class NativeQuakeAdapter:
    """Thin Python control plane around a native engine worker process."""

    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        fixed_tick_hz: int = 20,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.process = NativeEngineProcess(
            executable=executable,
            map_id=map_id,
            fixed_tick_hz=fixed_tick_hz,
            workdir=workdir,
            env=env,
        )
        self.process.start()

    def ticks_per_second(self) -> int:
        return self.process.fixed_tick_hz

    def reset(self, seed: int | None = None) -> np.ndarray:
        response = self.process.reset(seed=seed)
        return np.asarray(response["obs"], dtype=np.float32)

    def step(self, action: Mapping[str, int]) -> tuple[np.ndarray, float, bool, Dict[str, object]]:
        response = self.process.step(action)
        obs = np.asarray(response["obs"], dtype=np.float32)
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        info = dict(response.get("info", {}))
        return obs, reward, done, info

    def close(self) -> None:
        self.process.shutdown()
