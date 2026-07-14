"""Unified native bridge for Quake worker subprocesses.

The supported worker surface is the obs-buffer protocol:

- ``NativeObsBufferProcess`` reads direct-pack obs_buffer_v1 binary
- optional QTRN sidecars carry training extras

``NativeObsBufferAdapter`` adds reset-option defaults and a narrower public API.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from qnn.actions import (
    ActionLabels,
)

from engine.training_protocol import (
    TRAINING_BINARY_HEADER_SIZE,
    TRAINING_BINARY_MAGIC,
    TrustedTrainingExtrasV1,
    decode_binary_training_extras,
)


class NativeEngineError(RuntimeError):
    pass




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
        movement_threshold: float = 0.0,
        workdir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        extra_args: Sequence[str] | None = None,
        training_format: str = "",
    ) -> None:
        self.executable = str(executable)
        self.map_id = str(map_id)
        self.fixed_tick_hz = int(fixed_tick_hz)
        self.movement_threshold = float(movement_threshold)
        self.workdir = None if workdir is None else str(workdir)
        self.env = dict(env or {})
        self.extra_args = tuple(str(arg) for arg in (extra_args or ()))
        self.training_format = str(training_format)
        self.proc: subprocess.Popen[bytes] | None = None
        self.hello: Dict[str, object] | None = None
        self.capabilities: tuple[str, ...] = ()
        self._training_requested = self.training_format == "binary_v1"
        self._training_enabled = False
        self._last_training_extras: TrustedTrainingExtrasV1 | None = None
        self._current_episode_id = ""
        hello_payload: Dict[str, Any] = {
            "op": "hello",
            "map_id": self.map_id,
            "tick_hz": self.fixed_tick_hz,
            "protocol_version": "v6",
            "step_format": self._step_format,
            "training_format": self.training_format,
        }
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
        if isinstance(response_dict.get("map_id"), str):
            self.map_id = response_dict["map_id"]
        tick_hz = response_dict.get("tick_hz")
        if isinstance(tick_hz, (int, float)) and int(tick_hz) > 0:
            self.fixed_tick_hz = int(tick_hz)

    def _request_json(self, payload: bytes) -> Dict[str, object]:
        proc = self._ensure_running()
        proc.stdin.write(payload)
        proc.stdin.flush()
        # Pre-protocol stdout chatter (engine init lines like "Added
        # packfile…", "Quake Initialized") is suppressed by the
        # QNN_STDOUT_PROTOCOL=1 env var ``start()`` sets; this loop is
        # the belt-and-suspenders fallback for workers built before
        # that change. Skip everything until a '{'-prefixed line, the
        # canonical opener for every protocol response.
        while True:
            line = proc.stdout.readline()
            if not line:
                stderr = self._stderr_tail().strip()
                raise NativeEngineError(
                    f"Native engine terminated unexpectedly: {stderr or 'no response'}"
                )
            if line[:1] == b"{":
                return self._decode_json_response(line)

    # -- Action helpers ------------------------------------------------------

    # Binary step protocol: 1-byte opcode + packed action struct (16 bytes).
    # Matches qnn_action_t layout: move (press byte) + weapon + input_mask +
    # pad + look[3]. The press byte bit layout mirrors input_mask:
    #   bit 0   = attack press
    #   bits 1-2 = forward neg / pos
    #   bits 3-4 = side neg / pos
    #   bits 5-6 = up neg / pos
    #   bit 7   = jump press
    _BINARY_OP_STEP = b"\x01"
    _ACTION_PACK_FORMAT = "<4B3f"  # 4 uint8 + 3 float32 = 16 bytes
    _ACTION_PACKET_SIZE = 17

    @staticmethod
    def _pack_press_byte(labels: ActionLabels) -> int:
        t = 0.1
        m0, m1, m2 = float(labels.move[0]), float(labels.move[1]), float(labels.move[2])
        byte = 0
        if int(labels.attack):
            byte |= 0x01
        if m0 < -t:
            byte |= 0x02
        if m0 > t:
            byte |= 0x04
        if m1 < -t:
            byte |= 0x08
        if m1 > t:
            byte |= 0x10
        if m2 < -t:
            byte |= 0x20
        if m2 > t:
            byte |= 0x40
        if m2 > t:
            byte |= 0x80
        return byte

    @classmethod
    def pack_step_request(cls, action: Mapping[str, object]) -> bytes:
        """Validate and pack one complete binary step request.

        Vector drivers can do this once while assembling the action batch,
        then use ``step_send_packed`` without reparsing ActionLabels in the
        latency-sensitive pipe fan-out.
        """
        labels = ActionLabels.from_dict(action)
        return cls._BINARY_OP_STEP + struct.pack(
            cls._ACTION_PACK_FORMAT,
            cls._pack_press_byte(labels),
            int(labels.weapon),
            0,  # input_mask not transmitted from runtime
            0,  # _pad
            float(labels.look[0]), float(labels.look[1]), float(labels.look[2]),
        )

    @classmethod
    def pack_step_batch(
        cls,
        action_batch: Mapping[str, object],
        *,
        num_rows: int,
        normalize_look: bool = False,
    ) -> np.ndarray:
        """Validate and pack a complete ``(B, 17)`` binary step matrix.

        This is field-equivalent to calling :meth:`pack_step_request` for every
        row, but avoids per-row dictionaries, ``ActionLabels`` instances, and
        ``struct.pack`` calls. Batched look normalization can differ by one
        float32 ULP because NumPy reduces the rows together; the engine's
        direction mapping is scale-invariant. The returned uint8 matrix owns
        contiguous packet rows that can be passed to ``os.write`` without a
        copy.
        """
        rows = int(num_rows)
        if rows < 1:
            raise ValueError("num_rows must be >= 1")

        def _vector(name: str) -> np.ndarray:
            value = action_batch.get(name)
            if value is None:
                return np.zeros((rows, 3), dtype=np.float32)
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (rows, 3):
                raise ValueError(
                    f"action {name!r} must have shape ({rows}, 3), got {array.shape}"
                )
            return array

        def _scalar_int(name: str) -> np.ndarray:
            value = action_batch.get(name)
            if value is None:
                return np.zeros(rows, dtype=np.int64)
            array = np.asarray(value)
            if array.shape != (rows,):
                raise ValueError(
                    f"action {name!r} must have shape ({rows},), got {array.shape}"
                )
            return array.astype(np.int64, copy=False)

        move = np.clip(_vector("move"), -1.0, 1.0)
        look = _vector("look")
        if normalize_look:
            norms = np.linalg.norm(look, axis=1, keepdims=True)
            look = np.divide(
                look,
                norms,
                out=look.copy(),
                where=norms > np.float32(1e-6),
            )
        look = np.clip(look, -1.0, 1.0)
        attack = _scalar_int("attack")
        if np.any((attack < 0) | (attack >= 2)):
            raise ValueError("attack out of range [0, 2)")
        # Match ActionLabels.from_dict: engine weapon impulses clamp to 0..8.
        weapon = np.clip(_scalar_int("weapon"), 0, 8)

        press = attack.astype(np.uint8)
        threshold = np.float32(0.1)
        press |= ((move[:, 0] < -threshold).astype(np.uint8) << 1)
        press |= ((move[:, 0] > threshold).astype(np.uint8) << 2)
        press |= ((move[:, 1] < -threshold).astype(np.uint8) << 3)
        press |= ((move[:, 1] > threshold).astype(np.uint8) << 4)
        press |= ((move[:, 2] < -threshold).astype(np.uint8) << 5)
        up = (move[:, 2] > threshold).astype(np.uint8)
        press |= up << 6
        press |= up << 7

        packets = np.zeros((rows, cls._ACTION_PACKET_SIZE), dtype=np.uint8)
        packets[:, 0] = cls._BINARY_OP_STEP[0]
        packets[:, 1] = press
        packets[:, 2] = weapon.astype(np.uint8)
        packets[:, 5:].view("<f4").reshape(rows, 3)[:] = look
        return packets

    def _binary_step_request(self, action: Mapping[str, object]) -> bytes:
        return self.pack_step_request(action)

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
        # Tell the worker that stdout is the protocol channel: Sys_Printf
        # status messages (packfile init, "Quake Initialized", …) must
        # route to stderr instead of polluting the binary obs / JSON
        # response stream. See src/engine/nq/qnn_sys.c Sys_Printf.
        env.setdefault("QNN_STDOUT_PROTOCOL", "1")
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
        self._current_episode_id = ""

    def __enter__(self) -> "NativeProcessBase":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# NativeObsBufferProcess — direct-pack obs_buffer_v1 binary
# ---------------------------------------------------------------------------

# engine_norm phase 2: C side emits the native-width obs buffer per
# qnn.engine_norm (see src/engine/common/qnn_io.{h,c}). The legacy f32
# parser is dead; we use the native dict format end-to-end on this
# bridge. Downstream consumers (qnn.model.policy via ObsEmbedding's
# dequantizers, qnn.eval.live's logging) read the native key set.
from qnn.wire import OBS_BUFFER_SIZE, unpack_obs_buffer_native as _unpack_obs_buffer


class NativeObsBufferProcess(NativeProcessBase):
    """Subprocess bridge that reads direct-pack obs_buffer_v1 binary."""

    _step_format = "obs_buffer_v1"
    _required_capability = "obs_buffer_v1"
    _last_obs: Dict[str, np.ndarray] | None = None

    def _read_raw_obs_packet(self) -> bytes:
        """Read one obs buffer, surfacing control-plane errors immediately."""
        proc = self._ensure_running()
        first = self._read_exact(1)
        if first == b"{":
            # Reset/round-reset failures are JSON on the otherwise-binary
            # response channel. Reading a blind 4096 bytes here used to hang
            # forever on a short error line.
            self._decode_json_response(first + proc.stdout.readline())
            raise NativeEngineError("Worker returned JSON where an observation was expected")
        return first + self._read_exact(OBS_BUFFER_SIZE - 1)

    def _read_obs_packet(self) -> Dict[str, np.ndarray]:
        raw = self._read_raw_obs_packet()
        obs = _unpack_obs_buffer(raw)
        self._last_obs = obs
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
        self.step_send(action)
        return self.step_recv()

    # Split-phase stepping for vectorized drivers: fan out step_send() to
    # every worker first (each engine sims its tick concurrently), then
    # drain with step_recv(). One in-flight step per process, caller-managed.
    def step_send(self, action: Mapping[str, int]) -> None:
        self.step_send_packed(self.pack_step_request(action))

    def step_send_packed(self, payload: bytes | bytearray | memoryview | np.ndarray) -> None:
        """Send a prevalidated 17-byte step packet with one direct syscall."""
        if self.proc is None:
            self.start()
        proc = self._ensure_running()
        fd = proc.stdin.fileno()
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise NativeEngineError("Worker step pipe accepted no bytes")
            view = view[written:]

    def step_recv(self) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        obs = self._read_obs_packet()
        return obs, self._maybe_read_training_binary()

    def step_recv_raw(self) -> tuple[bytes, TrustedTrainingExtrasV1 | None]:
        """Drain one step WITHOUT unpacking the obs buffer — the batched
        drain (vec_env) unpacks all lanes at once via
        ``qnn.wire.unpack_obs_buffer_native_batch``."""
        raw = self._read_raw_obs_packet()
        return raw, self._maybe_read_training_binary()

    def round_reset_raw(self) -> tuple[bytes, TrustedTrainingExtrasV1 | None]:
        """Reset only the local arena match and return its live spawn obs.

        Unlike :meth:`reset_with_training`, this does not disconnect, reload
        the map, repeat sign-on, or respawn unrelated matches.
        """
        if self.proc is None:
            self.start()
        proc = self._ensure_running()
        proc.stdin.write(self._serialize_request({"op": "round_reset"}))
        proc.stdin.flush()
        raw = self._read_raw_obs_packet()
        return raw, self._maybe_read_training_binary()

    # -- Navmesh queries (JSON protocol) ------------------------------------

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
# Adapter wrappers — thin public API with reset-option defaults
# ---------------------------------------------------------------------------


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

    def map_id_snapshot(self) -> str | None:
        return self.process.map_id

    def reset_obs_with_training(
        self, seed: int | None = None
    ) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        return self.process.reset_with_training(seed=seed, options=self.reset_options)

    def step_obs_with_training(
        self, action: Mapping[str, int]
    ) -> tuple[Dict[str, np.ndarray], TrustedTrainingExtrasV1 | None]:
        return self.process.step_with_training(action)

    def change_map(self, new_map_id: str) -> str | None:
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
        return self.process.map_id

    def close(self) -> None:
        self.process.shutdown()


# ---------------------------------------------------------------------------
# NativeClientProcess — live network client (qnn_client_main.c)
# ---------------------------------------------------------------------------


class NativeClientProcess:
    """Subprocess bridge for the live NQ client binary.

    Unlike NativeObsBufferProcess (trainer/collect), the client takes the
    server address as a positional CLI arg, connects on startup, and speaks
    only the binary step protocol — no hello/reset/shutdown JSON.  After
    a successful signon the worker emits one initial obs immediately so
    the caller can run a forward pass before sending its first action;
    subsequent obs are emitted in response to each step.
    """

    _BINARY_OP_STEP = b"\x01"
    _ACTION_PACK_FORMAT = "<4B3f"  # 4 uint8 + 3 float32 = 16 bytes

    @staticmethod
    def _pack_press_byte(labels: ActionLabels) -> int:
        t = 0.1
        m0, m1, m2 = float(labels.move[0]), float(labels.move[1]), float(labels.move[2])
        byte = 0
        if int(labels.attack):
            byte |= 0x01
        if m0 < -t:
            byte |= 0x02
        if m0 > t:
            byte |= 0x04
        if m1 < -t:
            byte |= 0x08
        if m1 > t:
            byte |= 0x10
        if m2 < -t:
            byte |= 0x20
        if m2 > t:
            byte |= 0x40
        if m2 > t:
            byte |= 0x80
        return byte

    def __init__(
        self,
        executable: str | Path,
        server_addr: str,
        *,
        env: Mapping[str, str] | None = None,
        workdir: str | Path | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> None:
        self.executable = str(executable)
        self.server_addr = str(server_addr)
        self.env = dict(env or {})
        self.workdir = None if workdir is None else str(workdir)
        self.extra_args = tuple(str(a) for a in (extra_args or ()))
        self.proc: subprocess.Popen[bytes] | None = None

    def _ensure_running(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise NativeEngineError("Native client process is not running")
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
                    f"Native client terminated: {stderr or 'incomplete obs response'}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def start(self) -> Dict[str, np.ndarray]:
        """Spawn the client, wait for connect+signon, return the initial obs."""
        if self.proc is not None:
            raise NativeEngineError("Native client process is already running")

        env = os.environ.copy()
        env.update(self.env)
        self.proc = subprocess.Popen(
            [self.executable, self.server_addr, *self.extra_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workdir,
            env=env,
        )
        raw = self._read_exact(OBS_BUFFER_SIZE)
        return _unpack_obs_buffer(raw)

    def step(self, action: Mapping[str, object]) -> Dict[str, np.ndarray]:
        """Send one action, return the resulting obs."""
        proc = self._ensure_running()
        labels = ActionLabels.from_dict(action)
        payload = self._BINARY_OP_STEP + struct.pack(
            self._ACTION_PACK_FORMAT,
            self._pack_press_byte(labels),
            int(labels.weapon),
            0,  # input_mask not transmitted from runtime
            0,  # _pad
            float(labels.look[0]), float(labels.look[1]), float(labels.look[2]),
        )
        proc.stdin.write(payload)
        proc.stdin.flush()
        raw = self._read_exact(OBS_BUFFER_SIZE)
        return _unpack_obs_buffer(raw)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()
        self.proc = None

    def __enter__(self) -> "NativeClientProcess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
