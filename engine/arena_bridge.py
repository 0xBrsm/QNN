"""Subprocess bridge for one grouped Quake arena server.

The grouped protocol deliberately separates action staging from world advance:

1. every externally controlled seat sends its action to the NQ server;
2. the arena server advances exactly one frame;
3. every seat drains the resulting observation and QTRN sidecar.

That preserves the synchronous, zero-policy-lag PPO contract while sharing one
server/world simulation across up to eight isolated 1v1 matches.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from engine.bridge import NativeEngineError
from engine.training_protocol import (
    TRAINING_BINARY_HEADER_SIZE,
    TRAINING_BINARY_MAGIC,
    TrustedTrainingExtrasV1,
    decode_binary_training_extras,
)
from qnn.wire import OBS_BUFFER_SIZE


class _ArenaPipeProcess:
    """Shared exact-read and lifecycle helpers for arena subprocesses."""

    def __init__(
        self,
        executable: str | Path,
        *,
        args: Sequence[str],
        env: Mapping[str, str] | None,
        workdir: str | Path | None,
    ) -> None:
        self.executable = str(executable)
        self.args = tuple(str(value) for value in args)
        self.env = dict(env or {})
        self.workdir = None if workdir is None else str(workdir)
        self.proc: subprocess.Popen[bytes] | None = None
        self._stderr_file = None
        self.last_stderr = ""

    def spawn(self) -> None:
        if self.proc is not None:
            raise NativeEngineError("Arena process is already running")
        env = os.environ.copy()
        env.update(self.env)
        env.setdefault("QNN_STDOUT_PROTOCOL", "1")
        # Quake is chatty during combat. A PIPE that is only read after exit
        # eventually fills and freezes long PPO jobs; a temporary file keeps
        # diagnostics without adding one drain thread per engine process.
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        self.proc = subprocess.Popen(
            [self.executable, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            cwd=self.workdir,
            env=env,
        )

    def _ensure_running(self) -> subprocess.Popen[bytes]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise NativeEngineError("Arena process is not running")
        return self.proc

    def _stderr_after_exit(self) -> str:
        if self.proc is None or self.proc.poll() is None:
            return ""
        return self._read_stderr_tail()

    def _read_stderr_tail(self, size: int = 64 * 1024) -> str:
        stream = self._stderr_file
        if stream is None:
            return ""
        stream.flush()
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        stream.seek(max(0, length - size))
        return stream.read().decode("utf-8", errors="replace").strip()

    def read_exact(self, size: int) -> bytes:
        proc = self._ensure_running()
        chunks = bytearray()
        while len(chunks) < size:
            chunk = proc.stdout.read(size - len(chunks))
            if not chunk:
                detail = self._stderr_after_exit()
                raise NativeEngineError(
                    f"Arena process terminated: {detail or 'incomplete binary response'}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def read_json(self) -> dict[str, object]:
        proc = self._ensure_running()
        while True:
            line = proc.stdout.readline()
            if not line:
                detail = self._stderr_after_exit()
                raise NativeEngineError(
                    f"Arena process terminated: {detail or 'missing JSON response'}"
                )
            if line[:1] != b"{":
                continue
            payload = json.loads(line)
            if not bool(payload.get("ok", False)):
                raise NativeEngineError(str(payload.get("error", "arena process error")))
            return dict(payload)

    def write(self, payload: bytes | bytearray | memoryview) -> None:
        proc = self._ensure_running()
        view = memoryview(payload)
        while view:
            written = os.write(proc.stdin.fileno(), view)
            if written <= 0:
                raise NativeEngineError("Arena process accepted no input bytes")
            view = view[written:]

    def stop(self, opcode: int) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self.write(bytes((opcode,)))
            proc.wait(timeout=5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        finally:
            self.last_stderr = self._read_stderr_tail()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None
            self.proc = None


class ArenaServerProcess(_ArenaPipeProcess):
    """One dedicated Quake server hosting up to eight isolated matches."""

    OP_STEP = 2
    OP_RESET_MASK = 3
    OP_SHUTDOWN = 255

    def __init__(
        self,
        executable: str | Path,
        *,
        port: int,
        map_id: str,
        external_count: int,
        bot_count: int,
        bot_skill: int,
        self_play: bool,
        env: Mapping[str, str] | None = None,
        workdir: str | Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.port = int(port)
        super().__init__(
            executable,
            args=(
                "-dedicated", "16",
                "-game", "frikbotnex_train",
                "-port", str(self.port),
                "-qnn_map", str(map_id),
                "-qnn_external", str(int(external_count)),
                "-qnn_bots", str(int(bot_count)),
                "-qnn_bot_skill", str(int(bot_skill)),
                "-qnn_selfplay", "1" if self_play else "0",
                *extra_args,
            ),
            env=env,
            workdir=workdir,
        )

    def start_listening(self) -> None:
        self.spawn()
        state = self.read_json()
        if state.get("state") != "listening":
            raise NativeEngineError(f"Unexpected arena server state {state!r}")

    def wait_ready(self) -> None:
        state = self.read_json()
        if state.get("state") != "ready":
            raise NativeEngineError(f"Unexpected arena server state {state!r}")

    def step(self) -> None:
        self.write(bytes((self.OP_STEP,)))
        if self.read_exact(1) != bytes((self.OP_STEP,)):
            raise NativeEngineError("Invalid arena server step acknowledgement")

    def reset_matches(self, match_mask: int) -> None:
        mask = int(match_mask)
        if not 0 < mask <= 0xFF:
            raise ValueError("match_mask must select at least one of eight matches")
        self.write(bytes((self.OP_RESET_MASK, mask)))
        if self.read_exact(1) != bytes((self.OP_RESET_MASK,)):
            raise NativeEngineError("Invalid arena server reset acknowledgement")

    def close(self) -> None:
        self.stop(self.OP_SHUTDOWN)


class ArenaClientProcess(_ArenaPipeProcess):
    """Headless remote NQ client for one externally controlled arena seat."""

    OP_STAGE = 1
    OP_RECEIVE = 4
    OP_RESUME_SIGNON = 5
    OP_RECEIVE_RESET = 6
    OP_SHUTDOWN = 255

    def __init__(
        self,
        executable: str | Path,
        *,
        server_addr: str,
        match_id: int,
        seat_id: int,
        reward_json: str,
        env: Mapping[str, str] | None = None,
        workdir: str | Path | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        client_env = dict(env or {})
        client_env["QNN_REWARD_JSON"] = str(reward_json)
        self.match_id = int(match_id)
        self.seat_id = int(seat_id)
        self._episode_generation = 0
        super().__init__(
            executable,
            args=(
                "-game", "frikbotnex_train",
                "-qnn_server", str(server_addr),
                "-qnn_name", f"qnn_{self.match_id}_{self.seat_id}",
                *extra_args,
            ),
            env=client_env,
            workdir=workdir,
        )

    def start_connecting(self) -> None:
        self.spawn()

    def wait_signed_on(self) -> None:
        state = self.read_json()
        if state.get("state") != "signed_on":
            raise NativeEngineError(f"Unexpected arena client state {state!r}")

    def resume_signon(self) -> None:
        self.write(bytes((self.OP_RESUME_SIGNON,)))

    def wait_ready(self) -> tuple[bytes, TrustedTrainingExtrasV1]:
        state = self.read_json()
        if state.get("state") != "ready":
            raise NativeEngineError(f"Unexpected arena client state {state!r}")
        return self._read_transition()

    def _read_training(self) -> TrustedTrainingExtrasV1:
        magic = self.read_exact(4)
        if magic != TRAINING_BINARY_MAGIC:
            raise NativeEngineError(f"Unexpected arena training prefix {magic!r}")
        header = magic + self.read_exact(TRAINING_BINARY_HEADER_SIZE - 4)
        return decode_binary_training_extras(
            header,
            self.read_exact,
            episode_id=f"arena:{self.match_id}:{self.seat_id}:{self._episode_generation}",
        )

    def _read_transition(self) -> tuple[bytes, TrustedTrainingExtrasV1]:
        obs = self.read_exact(OBS_BUFFER_SIZE)
        return obs, self._read_training()

    def stage(self, action_packet: bytes | bytearray | memoryview) -> None:
        packet = memoryview(action_packet)
        if len(packet) != 17 or packet[0] != self.OP_STAGE:
            raise ValueError("arena action packet must be the packed 17-byte step format")
        self.write(packet)

    def wait_staged(self) -> None:
        if self.read_exact(1) != bytes((self.OP_STAGE,)):
            raise NativeEngineError("Invalid arena client stage acknowledgement")

    def receive_send(self) -> None:
        self.write(bytes((self.OP_RECEIVE,)))

    def receive_reset_send(self) -> None:
        self.write(bytes((self.OP_RECEIVE_RESET,)))

    def receive_recv(self) -> tuple[bytes, TrustedTrainingExtrasV1]:
        transition = self._read_transition()
        if transition[1].done:
            self._episode_generation += 1
        return transition

    def close(self) -> None:
        self.stop(self.OP_SHUTDOWN)


class ArenaGroupProcess:
    """A shared arena server and its role-neutral external policy seats."""

    def __init__(
        self,
        *,
        server_executable: str | Path,
        client_executable: str | Path,
        port: int,
        map_id: str,
        external_seats: Sequence[tuple[int, int]],
        bot_count: int,
        bot_skill: int,
        self_play: bool,
        reward_json: str,
        env: Mapping[str, str] | None = None,
        workdir: str | Path | None = None,
    ) -> None:
        seats = tuple((int(match_id), int(seat_id)) for match_id, seat_id in external_seats)
        self.server = ArenaServerProcess(
            server_executable,
            port=port,
            map_id=map_id,
            external_count=len(seats),
            bot_count=bot_count,
            bot_skill=bot_skill,
            self_play=self_play,
            env=env,
            workdir=workdir,
        )
        self.clients = tuple(
            ArenaClientProcess(
                client_executable,
                # Original NetQuake's console connect path drops an explicit
                # :port suffix before Datagram_Connect.  Set net_hostport via
                # the engine CLI and use the bare address instead.
                server_addr="127.0.0.1",
                match_id=match_id,
                seat_id=seat_id,
                reward_json=reward_json,
                env=env,
                workdir=workdir,
                extra_args=("-port", str(int(port)), "-qnn_direct_connect"),
            )
            for match_id, seat_id in seats
        )

    def start(self) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        self.server.start_listening()
        try:
            for client in self.clients:
                client.start_connecting()
                # Stock NetQuake's reliable sign-on stream is effectively
                # serial.  Finish each one-time handshake before admitting
                # the next seat; steady-state stepping remains parallel.
                client.wait_signed_on()
            for client in self.clients:
                client.resume_signon()
            self.server.wait_ready()
            return [client.wait_ready() for client in self.clients]
        except Exception as exc:
            self.close()
            client_details = [
                f"seat {client.match_id}:{client.seat_id}: {client.last_stderr[-4000:]}"
                for client in self.clients
                if client.last_stderr
            ]
            if client_details:
                raise NativeEngineError(
                    f"{exc}\nArena client diagnostics:\n" + "\n".join(client_details)
                ) from exc
            raise

    def step(
        self,
        action_packets: Sequence[bytes | bytearray | memoryview],
    ) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        if len(action_packets) != len(self.clients):
            raise ValueError("one action packet is required for each external arena seat")
        for client, packet in zip(self.clients, action_packets, strict=True):
            client.stage(packet)
        for client in self.clients:
            client.wait_staged()
        self.server.step()
        for client in self.clients:
            client.receive_send()
        return [client.receive_recv() for client in self.clients]

    def reset_matches(self, match_mask: int) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        self.server.reset_matches(match_mask)
        for client in self.clients:
            client.receive_reset_send()
        return [client.receive_recv() for client in self.clients]

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.server.close()
