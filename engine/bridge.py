"""Unified native token bridge for Quake worker subprocesses.

The supported worker surface is the token protocol:

- ``NativeTokenProcess`` reads QTOK token packets
- optional QTRN sidecars carry training extras

``NativeTokenAdapter`` adds reset-option defaults and a narrower public API.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, NoReturn, Sequence

from quake_ai.actions import (
    ActionLabels,
)
from quake_ai.rl.schemas import MapState

from engine.training_protocol import (
    TRAINING_BINARY_HEADER_SIZE,
    TRAINING_BINARY_MAGIC,
    TrustedTrainingExtrasV1,
    decode_binary_training_extras,
)
from engine.token_protocol import (
    TOKEN_BINARY_HEADER_SIZE,
    TOKEN_BINARY_MAGIC,
    TrustedSelfToken,
    TrustedTokenTick,
    decode_binary_token_tick,
)


class NativeEngineError(RuntimeError):
    pass


# Keep as alias for the token path; same error hierarchy.
NativeTokenError = NativeEngineError

_ACTION_KEYS = (
    "move",
    "look",
    "fire",
    "jump",
    "switch",
    "recall_0",
    "recall_1",
    "recall_2",
    "recall_3",
)


# ---------------------------------------------------------------------------
# Base process class — shared subprocess lifecycle
# ---------------------------------------------------------------------------


class NativeProcessBase:
    """Common subprocess lifecycle for native Quake worker processes."""

    # Subclasses set this to negotiate the binary step format with the worker.
    _step_format: str = ""
    # Capability string the worker must advertise after hello.
    _required_capability: str = ""

    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        fixed_tick_hz: int = 20,
        resample_hz: int = 0,
        movement_threshold: float = 0.0,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] | None = None,
        training_format: str = "",
    ) -> None:
        self.executable = str(executable)
        self.map_id = str(map_id)
        self.fixed_tick_hz = int(fixed_tick_hz)
        self.resample_hz = int(resample_hz)
        self.movement_threshold = float(movement_threshold)
        self.workdir = None if workdir is None else str(workdir)
        self.env = dict(env or {})
        self.extra_args = tuple(str(arg) for arg in (extra_args or ()))
        self.training_format = str(training_format)
        self.proc: subprocess.Popen[bytes] | None = None
        self.hello: Dict[str, object] | None = None
        self.capabilities: tuple[str, ...] = ()
        self.map_state: MapState | None = None
        self._training_requested = self.training_format == "binary_v1"
        self._training_enabled = False
        self._last_training_extras: TrustedTrainingExtrasV1 | None = None
        self._last_token_tick: TrustedTokenTick | None = None
        self._current_episode_id = ""
        hello_payload: Dict[str, Any] = {
            "op": "hello",
            "map_id": self.map_id,
            "tick_hz": self.fixed_tick_hz,
            "protocol_version": "v6",
            "step_format": self._step_format,
            "training_format": self.training_format,
        }
        if self.resample_hz > 0:
            hello_payload["resample_hz"] = self.resample_hz
        if self.movement_threshold > 0:
            hello_payload["movement_threshold"] = int(self.movement_threshold)
        self._hello_request = self._serialize_request(hello_payload)
        self._shutdown_request = self._serialize_request({"op": "shutdown"})

    # -- Serialization / IO --------------------------------------------------

    @staticmethod
    def _serialize_request(payload: Mapping[str, Any]) -> bytes:
        return (json.dumps(dict(payload), separators=(",", ":")) + "\n").encode("utf-8")

    def _ensure_running(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise NativeEngineError("Native engine process is not running")
        return self.proc

    def _stderr_tail(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _read_exact(self, size: int) -> bytes:
        proc = self._ensure_running()
        chunks = bytearray()
        while len(chunks) < size:
            chunk = proc.stdout.read(size - len(chunks))
            if not chunk:
                stderr = self._stderr_tail().strip()
                raise NativeEngineError(
                    f"Native engine terminated unexpectedly: {stderr or 'incomplete binary response'}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    # -- JSON protocol -------------------------------------------------------

    def _decode_json_response(self, line: bytes) -> Dict[str, object]:
        response = json.loads(line)
        if not bool(response.get("ok", False)):
            raise NativeEngineError(str(response.get("error", "unknown native engine error")))
        response_dict = dict(response)
        self._update_response_state(response_dict)
        return response_dict

    def _update_response_state(self, response_dict: Mapping[str, object]) -> None:
        if isinstance(response_dict.get("capabilities"), list):
            self.capabilities = tuple(str(v) for v in response_dict.get("capabilities", []))
        if isinstance(response_dict.get("map_state"), Mapping):
            self.map_state = MapState.from_dict(response_dict["map_state"])
        tick_hz = response_dict.get("tick_hz")
        if isinstance(tick_hz, (int, float)) and int(tick_hz) > 0:
            self.fixed_tick_hz = int(tick_hz)

    def _request_json(self, payload: bytes) -> Dict[str, object]:
        proc = self._ensure_running()
        proc.stdin.write(payload)
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            stderr = self._stderr_tail().strip()
            raise NativeEngineError(
                f"Native engine terminated unexpectedly: {stderr or 'no response'}"
            )
        return self._decode_json_response(line)

    # -- Action helpers ------------------------------------------------------

    # Binary step protocol: 1-byte opcode + packed action struct (44 bytes).
    # Matches qnn_worker_action_t layout: move[2] look[2] fire jump switch recall[4].
    _BINARY_OP_STEP = b"\x01"
    _ACTION_PACK_FORMAT = "<2f2f7i"  # 4 floats + 7 ints = 44 bytes

    def _binary_step_request(self, action: Mapping[str, object]) -> bytes:
        labels = ActionLabels.from_dict(action)
        return self._BINARY_OP_STEP + struct.pack(
            self._ACTION_PACK_FORMAT,
            float(labels.move[0]), float(labels.move[1]),
            float(labels.look[0]), float(labels.look[1]),
            int(labels.fire), int(labels.jump), int(labels.switch),
            int(labels.recall_0), int(labels.recall_1),
            int(labels.recall_2), int(labels.recall_3),
        )

    def _action_request(self, action: Mapping[str, object], op: str = "step") -> bytes:
        labels = ActionLabels.from_dict(action)
        return self._serialize_request(
            {
                "op": op,
                "action": labels.to_dict(),
            }
        )

    # -- Training sidecar ----------------------------------------------------

    def _maybe_read_training_binary(self) -> TrustedTrainingExtrasV1 | None:
        if not self._training_enabled:
            self._last_training_extras = None
            return None
        magic = self._read_exact(4)
        if magic != TRAINING_BINARY_MAGIC:
            raise NativeEngineError(f"Unexpected training frame prefix {magic!r}")
        header = magic + self._read_exact(TRAINING_BINARY_HEADER_SIZE - 4)
        extras = decode_binary_training_extras(
            header, self._read_exact, episode_id=self._current_episode_id
        )
        self._last_training_extras = extras
        return extras

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> Dict[str, object]:
        if self.proc is not None:
            return dict(self.hello or {})

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
        self.hello = self._request_json(self._hello_request)
        self._training_enabled = (
            self._training_requested and "training_extras_v1" in self.capabilities
        )
        self._on_hello()
        return dict(self.hello)

    def _on_hello(self) -> None:
        """Hook for subclasses to validate capabilities after hello."""
        if self._required_capability and self._required_capability not in self.capabilities:
            raise NativeEngineError(
                f"Worker does not advertise {self._required_capability} capability"
            )

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            self._request_json(self._shutdown_request)
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
        self._training_enabled = False
        self._last_training_extras = None
        self._last_token_tick = None
        self._current_episode_id = ""

    def __enter__(self) -> "NativeProcessBase":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# NativeTokenProcess — QTOK (token-step) binary packets
# ---------------------------------------------------------------------------


class NativeTokenProcess(NativeProcessBase):
    """Subprocess bridge that reads QTOK token-step binary packets."""

    _step_format = "token_binary_v2"
    _required_capability = "token_step_v2"

    def _read_token_packet(self) -> TrustedTokenTick:
        proc = self._ensure_running()
        prefix = self._read_exact(1)
        if prefix == b"{":
            self._decode_json_response(prefix + proc.stdout.readline())
            raise NativeEngineError("Worker returned JSON when token packet was expected")

        magic = prefix + self._read_exact(3)
        if magic != TOKEN_BINARY_MAGIC:
            raise NativeEngineError(f"Unexpected token packet prefix {magic!r}")
        header = magic + self._read_exact(TOKEN_BINARY_HEADER_SIZE - 4)
        tick = decode_binary_token_tick(header, self._read_exact)
        if int(tick.tick_hz) > 0:
            self.fixed_tick_hz = int(tick.tick_hz)
        self._last_token_tick = tick
        return tick

    def _request_token_packet(self, payload: bytes) -> TrustedTokenTick:
        proc = self._ensure_running()
        proc.stdin.write(payload)
        proc.stdin.flush()
        return self._read_token_packet()

    # -- Public API ----------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> TrustedTokenTick:
        tick, _ = self.reset_with_training(seed=seed, options=options)
        return tick

    def reset_with_training(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[TrustedTokenTick, TrustedTrainingExtrasV1 | None]:
        if self.proc is None:
            self.start()
        payload: Dict[str, Any] = {"op": "reset", "seed": seed if seed is not None else -1}
        if options:
            payload["options"] = dict(options)
        tick = self._request_token_packet(self._serialize_request(payload))
        self._current_episode_id = f"{self.map_id}:{int(tick.tick)}"
        return tick, self._maybe_read_training_binary()

    def step(self, action: Mapping[str, int]) -> TrustedTokenTick:
        tick, _ = self.step_with_training(action)
        return tick

    def step_with_training(
        self, action: Mapping[str, int]
    ) -> tuple[TrustedTokenTick, TrustedTrainingExtrasV1 | None]:
        if self.proc is None:
            self.start()
        tick = self._request_token_packet(self._action_request(action))
        return tick, self._maybe_read_training_binary()

    def collect_episode(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
        idle_action: Mapping[str, int] | None = None,
    ) -> list[TrustedTokenTick]:
        if self.proc is None:
            self.start()

        if "token_collect_v1" not in self.capabilities:
            if idle_action is None:
                raise NativeEngineError("token_collect_v1 is unavailable and no idle_action fallback was provided")
            ticks = [self.reset(seed=seed, options=options)]
            while not ticks[-1].done:
                ticks.append(self.step(idle_action))
            return ticks

        payload: Dict[str, Any] = {"op": "collect", "seed": seed if seed is not None else -1}
        if options:
            payload["options"] = dict(options)

        proc = self._ensure_running()
        proc.stdin.write(self._serialize_request(payload))
        proc.stdin.flush()

        ticks: list[TrustedTokenTick] = []
        while True:
            tick = self._read_token_packet()
            ticks.append(tick)
            if tick.done:
                break
        if ticks:
            self._current_episode_id = f"{self.map_id}:{int(ticks[0].tick)}"
        return ticks

    @staticmethod
    def _vec3_payload(point: Sequence[float]) -> list[float]:
        values = [float(value) for value in point]
        if len(values) != 3:
            raise NativeEngineError("navmesh queries require exactly 3 coordinates")
        return values

    def nav_query(self, *, kind: str, **payload: object) -> Dict[str, object]:
        if self.proc is None:
            self.start()
        if "navmesh_query_v1" not in self.capabilities:
            raise NativeEngineError("Worker does not advertise navmesh_query_v1 capability")
        response = self._request_json(self._serialize_request({"op": "nav_query", "kind": str(kind), **payload}))
        result = response.get("result")
        if not isinstance(result, dict):
            raise NativeEngineError("Worker returned malformed nav_query response")
        return dict(result)

    def navmesh_nearest(self, point: Sequence[float]) -> Dict[str, object]:
        return self.nav_query(kind="nearest", point=self._vec3_payload(point))

    def navmesh_path(self, start: Sequence[float], end: Sequence[float]) -> Dict[str, object]:
        return self.nav_query(
            kind="path",
            start=self._vec3_payload(start),
            end=self._vec3_payload(end),
        )

    def navmesh_area(self, point: Sequence[float]) -> Dict[str, object]:
        return self.nav_query(kind="area", point=self._vec3_payload(point))

    def navmesh_cluster(self, point: Sequence[float]) -> Dict[str, object]:
        return self.nav_query(kind="cluster", point=self._vec3_payload(point))

    def navmesh_route(self, start: Sequence[float], end: Sequence[float]) -> Dict[str, object]:
        return self.nav_query(
            kind="route",
            start=self._vec3_payload(start),
            end=self._vec3_payload(end),
        )


# ---------------------------------------------------------------------------
# NativeObsBufferProcess — direct-pack obs_buffer_v1 binary
# ---------------------------------------------------------------------------

import numpy as np

# Obs buffer layout constants (must match qnn_obs_buffer.h)
OBS_BUFFER_SIZE = 15892

_OBS_FIELDS = {
    "self_scalars":             (    0,   92, np.float32, (23,)),
    "self_weapon_id":           (   92,    4, np.int32,   (1,)),
    "self_movement_id":         (   96,    4, np.int32,   (1,)),
    "self_cluster_id":          (  100,    4, np.int32,   (1,)),
    "object_ids":               (  104, 1280, np.int32,   (64, 5)),
    "object_scalars":           ( 1384, 3328, np.float32, (64, 13)),
    "object_mask":              ( 4712,   64, np.uint8,   (64,)),
    "object_route_cluster_ids": ( 4776, 2048, np.int32,   (64, 8)),
    "event_ids":                ( 6824, 4096, np.int32,   (256, 4)),
    "event_scalars":            (10920, 3072, np.float32, (256, 3)),
    "event_owner":              (13992, 1024, np.int32,   (256,)),
    "event_mask":               (15016,  256, np.uint8,   (256,)),
    "spatial_ids":              (15272,   36, np.int32,   (9,)),
    "spatial_scalars":          (15308,  360, np.float32, (9, 10)),
    "action_history":           (15668,  224, np.float32, (8, 7)),
}


def _unpack_obs_buffer(raw: bytes) -> Dict[str, np.ndarray]:
    """Zero-copy unpack of the 15892-byte obs buffer into numpy arrays."""
    obs: Dict[str, np.ndarray] = {}
    for name, (offset, size, dtype, shape) in _OBS_FIELDS.items():
        arr = np.frombuffer(raw, dtype=dtype, offset=offset, count=int(np.prod(shape)))
        # Convert uint8 mask fields to bool to match TokenObservationEncoder output
        if name in ("object_mask", "event_mask"):
            obs[name] = arr.astype(np.bool_).reshape(shape)
        else:
            obs[name] = arr.reshape(shape).copy()  # copy — raw bytes invalidated next call
    return obs


class NativeObsBufferProcess(NativeProcessBase):
    """Subprocess bridge that reads direct-pack obs_buffer_v1 binary."""

    _step_format = "obs_buffer_v1"
    _required_capability = "obs_buffer_v1"
    _last_obs: Dict[str, np.ndarray] | None = None

    def _read_obs_packet(self) -> Dict[str, np.ndarray]:
        raw = self._read_exact(OBS_BUFFER_SIZE)
        obs = _unpack_obs_buffer(raw)
        self._last_obs = obs
        # Synthesize a minimal _last_token_tick so _action_request can read weapon_bits
        # for weapons_owned. Only self_token.weapon_bits is used.
        scalars = obs["self_scalars"]
        weapon_bits = [float(scalars[i]) for i in range(3, 10)]
        self._last_token_tick = TrustedTokenTick(
            tick=0, steps=0, current_region_id=0,
            self_token=TrustedSelfToken(
                health=0, armor=0, armor_type=0,
                weapon_bits=weapon_bits,
                weapon_super=0, ammo=[0]*4, velocity=[0]*3,
                yaw_sin=0, yaw_cos=0, pitch_sin=0, pitch_cos=0,
                dt=0, weapon_id=0, movement_id=0, cluster_id=0,
            ),
            object_tokens=[], spatial_tokens=[],
        )
        return obs

    def _request_obs_packet(self, payload: bytes) -> Dict[str, np.ndarray]:
        proc = self._ensure_running()
        proc.stdin.write(payload)
        proc.stdin.flush()
        return self._read_obs_packet()

    # -- Public API ----------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> Dict[str, np.ndarray]:
        obs, _ = self.reset_with_training(seed=seed, options=options)
        return obs

    def reset_with_training(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        if self.proc is None:
            self.start()
        payload: Dict[str, Any] = {"op": "reset", "seed": seed if seed is not None else -1}
        if options:
            payload["options"] = dict(options)
        obs = self._request_obs_packet(self._serialize_request(payload))
        self._current_episode_id = f"{self.map_id}:{0}"
        return obs, self._maybe_read_training_binary()

    def step(self, action: Mapping[str, int]) -> Dict[str, np.ndarray]:
        obs, _ = self.step_with_training(action)
        return obs

    def step_with_training(
        self, action: Mapping[str, int]
    ) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        if self.proc is None:
            self.start()
        obs = self._request_obs_packet(self._binary_step_request(action))
        return obs, self._maybe_read_training_binary()


# ---------------------------------------------------------------------------
# Adapter wrappers — thin public API with reset-option defaults
# ---------------------------------------------------------------------------


class NativeTokenAdapter:
    """Control-plane wrapper around a NativeTokenProcess."""

    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        fixed_tick_hz: int = 20,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] | None = None,
        reset_options: Mapping[str, object] | None = None,
        training_format: str = "",
    ) -> None:
        self.reset_options = dict(reset_options or {})
        self.process = NativeTokenProcess(
            executable=executable,
            map_id=map_id,
            fixed_tick_hz=fixed_tick_hz,
            workdir=workdir,
            env=env,
            extra_args=extra_args,
            training_format=training_format,
        )
        self.process.start()

    def ticks_per_second(self) -> int:
        return self.process.fixed_tick_hz

    def map_state_snapshot(self) -> MapState | None:
        return self.process.map_state

    def reset_tokens(self, seed: int | None = None) -> TrustedTokenTick:
        return self.process.reset(seed=seed, options=self.reset_options)

    def reset_tokens_with_training(
        self, seed: int | None = None
    ) -> tuple[TrustedTokenTick, TrustedTrainingExtrasV1 | None]:
        return self.process.reset_with_training(seed=seed, options=self.reset_options)

    def step_tokens(self, action: Mapping[str, int]) -> TrustedTokenTick:
        return self.process.step(action)

    def step_tokens_with_training(
        self, action: Mapping[str, int]
    ) -> tuple[TrustedTokenTick, TrustedTrainingExtrasV1 | None]:
        return self.process.step_with_training(action)

    def navmesh_nearest(self, point: Sequence[float]) -> Dict[str, object]:
        return self.process.navmesh_nearest(point)

    def navmesh_path(self, start: Sequence[float], end: Sequence[float]) -> Dict[str, object]:
        return self.process.navmesh_path(start, end)

    def navmesh_area(self, point: Sequence[float]) -> Dict[str, object]:
        return self.process.navmesh_area(point)

    def navmesh_cluster(self, point: Sequence[float]) -> Dict[str, object]:
        return self.process.navmesh_cluster(point)

    def navmesh_route(self, start: Sequence[float], end: Sequence[float]) -> Dict[str, object]:
        return self.process.navmesh_route(start, end)

    def change_map(self, new_map_id: str) -> MapState | None:
        """Restart the worker subprocess with a different map.

        Returns the new ``MapState`` from the fresh hello handshake.
        """
        self.process.shutdown()
        self.process = NativeTokenProcess(
            executable=self.process.executable,
            map_id=new_map_id,
            fixed_tick_hz=self.process.fixed_tick_hz,
            workdir=self.process.workdir,
            env=self.process.env,
            extra_args=self.process.extra_args,
            training_format=self.process.training_format,
        )
        self.process.start()
        return self.process.map_state

    def reset(self, seed: int | None = None) -> NoReturn:
        del seed
        raise NativeEngineError(
            "Legacy flat observation API has been removed; use reset_tokens() or NativeWorldEnv"
        )

    def step(self, action: Mapping[str, int]) -> NoReturn:
        del action
        raise NativeEngineError(
            "Legacy flat observation API has been removed; use step_tokens() or NativeWorldEnv"
        )

    def close(self) -> None:
        self.process.shutdown()


class NativeObsBufferAdapter:
    """Control-plane wrapper around a NativeObsBufferProcess.

    Returns pre-packed numpy observations directly from C (no Python tensor packing).
    Training extras continue via QTRN.
    """

    def __init__(
        self,
        executable: str | Path,
        map_id: str,
        fixed_tick_hz: int = 20,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] | None = None,
        reset_options: Mapping[str, object] | None = None,
        training_format: str = "",
    ) -> None:
        self.reset_options = dict(reset_options or {})
        self.process = NativeObsBufferProcess(
            executable=executable,
            map_id=map_id,
            fixed_tick_hz=fixed_tick_hz,
            workdir=workdir,
            env=env,
            extra_args=extra_args,
            training_format=training_format,
        )
        self.process.start()

    def ticks_per_second(self) -> int:
        return self.process.fixed_tick_hz

    def map_state_snapshot(self) -> MapState | None:
        return self.process.map_state

    def reset_obs_with_training(
        self, seed: int | None = None
    ) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        return self.process.reset_with_training(seed=seed, options=self.reset_options)

    def step_obs_with_training(
        self, action: Mapping[str, int]
    ) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        return self.process.step_with_training(action)

    def change_map(self, new_map_id: str) -> MapState | None:
        self.process.shutdown()
        self.process = NativeObsBufferProcess(
            executable=self.process.executable,
            map_id=new_map_id,
            fixed_tick_hz=self.process.fixed_tick_hz,
            workdir=self.process.workdir,
            env=self.process.env,
            extra_args=self.process.extra_args,
            training_format=self.process.training_format,
        )
        self.process.start()
        return self.process.map_state

    def close(self) -> None:
        self.process.shutdown()
