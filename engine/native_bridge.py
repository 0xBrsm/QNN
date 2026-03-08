"""Native engine process bridge for engine-backed adapters."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from quake_ai.actions import ActionLabels, mouse_count_from_look_label
from quake_ai.schemas import MapStateV2, WorldTickV2


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
        extra_args: Sequence[str] | None = None,
    ) -> None:
        self.executable = str(executable)
        self.map_id = map_id
        self.fixed_tick_hz = fixed_tick_hz
        self.workdir = None if workdir is None else str(workdir)
        self.env = dict(env or {})
        self.extra_args = tuple(str(arg) for arg in (extra_args or ()))
        self.proc: subprocess.Popen[bytes] | None = None
        self.hello: Dict[str, object] | None = None
        self.capabilities: tuple[str, ...] = ()
        self.map_state: MapStateV2 | None = None
        self._hello_request = self._serialize_request({"op": "hello", "map_id": self.map_id, "tick_hz": self.fixed_tick_hz})
        self._hello_v2_request = self._serialize_request(
            {"op": "hello", "map_id": self.map_id, "tick_hz": self.fixed_tick_hz, "protocol_version": "v2"}
        )
        self._shutdown_request = self._serialize_request({"op": "shutdown"})

    @staticmethod
    def _serialize_request(payload: Mapping[str, Any]) -> bytes:
        return (json.dumps(dict(payload), separators=(",", ":")) + "\n").encode("utf-8")

    def _stderr_tail(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _ensure_running(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise NativeEngineError("Native engine process is not running")
        return self.proc

    def _step_request(self, action: Mapping[str, int], op: str = "step") -> bytes:
        labels = ActionLabels.from_dict(action)
        return self._serialize_request(
            {
                "op": op,
                "action": {
                    "move": labels.move,
                    "strafe": labels.strafe,
                    "look_yaw": labels.look_yaw,
                    "look_pitch": labels.look_pitch,
                    "look_yaw_count": mouse_count_from_look_label(labels.look_yaw),
                    "look_pitch_count": mouse_count_from_look_label(labels.look_pitch),
                    "fire": labels.fire,
                    "jump": labels.jump,
                    "weapon": labels.weapon,
                },
            }
        )

    def _request(self, payload: bytes) -> Dict[str, object]:
        proc = self._ensure_running()
        proc.stdin.write(payload)
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            stderr = self._stderr_tail().strip()
            raise NativeEngineError(f"Native engine terminated unexpectedly: {stderr or 'no response'}")

        response = json.loads(line)
        if not bool(response.get("ok", False)):
            raise NativeEngineError(str(response.get("error", "unknown native engine error")))
        response_dict = dict(response)
        if isinstance(response_dict.get("capabilities"), list):
            self.capabilities = tuple(str(value) for value in response_dict.get("capabilities", []))
        if isinstance(response_dict.get("map_state"), Mapping):
            self.map_state = MapStateV2.from_dict(response_dict["map_state"])
        return response_dict

    def start(self) -> Dict[str, object]:
        if self.proc is not None:
            return self._request(self._hello_request)

        env = os.environ.copy()
        env.update(self.env)
        self.proc = subprocess.Popen(
            [self.executable, *self.extra_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workdir,
            env=env,
        )
        self.hello = self._request(self._hello_v2_request)
        return dict(self.hello)

    def reset(self, seed: int | None = None, options: Mapping[str, object] | None = None) -> Dict[str, object]:
        if self.proc is None:
            self.start()
        payload: Dict[str, Any] = {"op": "reset", "seed": seed if seed is not None else -1}
        if options:
            payload["options"] = dict(options)
        return self._request(self._serialize_request(payload))

    def step(self, action: Mapping[str, int]) -> Dict[str, object]:
        if self.proc is None:
            self.start()
        return self._request(self._step_request(action))

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            self._request(self._shutdown_request)
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
        self.hello = None
        self.capabilities = ()
        self.map_state = None

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
        extra_args: Sequence[str] | None = None,
        reset_options: Mapping[str, object] | None = None,
    ) -> None:
        self.reset_options = dict(reset_options or {})
        self.process = NativeEngineProcess(
            executable=executable,
            map_id=map_id,
            fixed_tick_hz=fixed_tick_hz,
            workdir=workdir,
            env=env,
            extra_args=extra_args,
        )
        self.process.start()

    def ticks_per_second(self) -> int:
        return self.process.fixed_tick_hz

    def map_state_v2(self) -> MapStateV2 | None:
        return self.process.map_state

    def reset_world(self, seed: int | None = None) -> WorldTickV2:
        response = self.process.reset(seed=seed, options=self.reset_options)
        if "world_tick" not in response:
            raise NativeEngineError("Native engine worker did not return world_tick data")
        return WorldTickV2.from_dict(response["world_tick"])

    def step_world(self, action: Mapping[str, int]) -> tuple[WorldTickV2, float, bool, Dict[str, object]]:
        response = self.process.step(action)
        if "world_tick" not in response:
            raise NativeEngineError("Native engine worker did not return world_tick data")
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        info = dict(response.get("info", {}))
        return WorldTickV2.from_dict(response["world_tick"]), reward, done, info

    def reset(self, seed: int | None = None) -> np.ndarray:
        response = self.process.reset(seed=seed, options=self.reset_options)
        if "obs" not in response:
            raise NativeEngineError("Native engine worker did not return legacy obs data")
        return np.asarray(response["obs"], dtype=np.float32)

    def step(self, action: Mapping[str, int]) -> tuple[np.ndarray, float, bool, Dict[str, object]]:
        response = self.process.step(action)
        if "obs" not in response:
            raise NativeEngineError("Native engine worker did not return legacy obs data")
        obs = np.asarray(response["obs"], dtype=np.float32)
        reward = float(response.get("reward", 0.0))
        done = bool(response.get("done", False))
        info = dict(response.get("info", {}))
        return obs, reward, done, info

    def close(self) -> None:
        self.process.shutdown()
