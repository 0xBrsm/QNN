"""Subprocess bridge for one grouped Quake arena server.

The grouped protocol deliberately separates action staging from world advance:

1. observer clients apply their action locally for view/prediction parity;
2. one packed action batch is sent directly to the NQ server;
3. the arena server advances exactly one frame;
4. every seat drains the resulting observation and QTRN sidecar.

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

from engine.bridge import NativeEngineError, NativeObsBufferProcess
from engine.training_protocol import (
    TRAINING_BINARY_HEADER_SIZE,
    TRAINING_BINARY_MAGIC,
    TrustedTrainingExtrasV1,
    decode_binary_training_extras,
)
from qnn.obs_api import (
    Declaration,
    coerce_declaration,
    compile_layout,
    encode_attach_decl,
    parse_layout_reply,
)
from qnn.wire import OBS_BUFFER_SIZE


def _compile_seat_declaration(
    declaration: object,
) -> tuple[Declaration | None, int]:
    """Normalize one seat's optional declaration → (declaration, read size).

    No declaration (the default everywhere today) means the seat runs
    the legacy default plan: nothing is sent on the wire and its obs
    reads stay OBS_BUFFER_SIZE.
    """
    parsed = coerce_declaration(declaration)
    if parsed is None:
        return None, OBS_BUFFER_SIZE
    return parsed, compile_layout(parsed).frame_bytes


def _attach_seat_declaration(
    process: "_ArenaPipeProcess",
    declaration: Declaration,
    seat_index: int,
) -> None:
    """Send one OP_ATTACH_DECL and verify the engine's layout reply.

    The reply must equal the local ``compile_layout`` result exactly —
    any divergence means the two registry mirrors disagree, which is a
    hard error, never something to paper over.
    """
    process.write(encode_attach_decl(declaration, seat_index=seat_index))
    response = process.read_json()
    layout = parse_layout_reply(response.get("layout"))
    expected = compile_layout(declaration)
    if layout != expected:
        raise NativeEngineError(
            f"Arena attach-decl layout for seat {seat_index} disagrees with the "
            f"local compile: engine={layout.to_dict()} local={expected.to_dict()}"
        )


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

    def read_training_frame(
        self,
        *,
        episode_id: str,
    ) -> tuple[TrustedTrainingExtrasV1, bytes]:
        chunks: list[bytes] = []
        magic = self.read_exact(4)
        if magic != TRAINING_BINARY_MAGIC:
            raise NativeEngineError(f"Unexpected arena training prefix {magic!r}")
        header = magic + self.read_exact(TRAINING_BINARY_HEADER_SIZE - 4)
        chunks.append(header)

        def read_more(size: int) -> bytes:
            payload = self.read_exact(size)
            chunks.append(payload)
            return payload

        extras = decode_binary_training_extras(
            header,
            read_more,
            episode_id=episode_id,
        )
        return extras, b"".join(chunks)

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
    OP_STEP_BATCH = 7
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
        observer_seats: Sequence[tuple[int, int]] = (),
        observer_mode: str = "external",
        reward_json: str = "",
        weapon_config: Mapping[str, object] | None = None,
        observer_declarations: Sequence[object] | None = None,
    ) -> None:
        self.port = int(port)
        if observer_mode not in {"external", "virtual", "shadow"}:
            raise ValueError(f"unknown arena observer mode {observer_mode!r}")
        self.observer_mode = observer_mode
        self.observer_seats = tuple(
            (int(match_id), int(seat_id)) for match_id, seat_id in observer_seats
        )
        # Per-observer-seat obs declarations (obs_api v1), aligned with
        # observer_seats. None entries (and a None list — the default
        # everywhere today) keep the legacy default plan for that seat.
        if observer_declarations is None:
            observer_declarations = (None,) * len(self.observer_seats)
        if len(observer_declarations) != len(self.observer_seats):
            raise ValueError(
                f"observer_declarations has {len(observer_declarations)} entries "
                f"for {len(self.observer_seats)} observer seats"
            )
        compiled = tuple(
            _compile_seat_declaration(declaration)
            for declaration in observer_declarations
        )
        self.observer_declarations = tuple(decl for decl, _bytes in compiled)
        self._observer_frame_bytes = tuple(bytes_ for _decl, bytes_ in compiled)
        self._episode_generations = [0] * len(self.observer_seats)
        self.last_training_frames: tuple[bytes, ...] = ()
        server_env = dict(env or {})
        if observer_mode != "external":
            server_env["QNN_REWARD_JSON"] = str(reward_json)
        observer_args: tuple[str, ...] = ()
        if observer_mode == "virtual":
            observer_args = ("-qnn_virtual_clients", "1")
        elif observer_mode == "shadow":
            observer_args = ("-qnn_shadow_clients", "1")
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
                *observer_args,
                *self._weapon_args(weapon_config),
                *extra_args,
            ),
            env=server_env,
            workdir=workdir,
        )

    # Scenario inventory key -> arena-server CLI flag. Weapon names are resolved
    # to IT_ bits inside the engine (QNN_ArenaWeaponBit), so the encoding stays
    # authoritative in one place; ammo/armor/health flow through as-is.
    _WEAPON_ARG_FLAGS: tuple[tuple[str, str], ...] = (
        ("model_weapon", "-qnn_inv_selected"),
        ("bot_weapon_pin", "-qnn_bot_weapon_pin"),
        ("shells", "-qnn_inv_shells"),
        ("nails", "-qnn_inv_nails"),
        ("rockets", "-qnn_inv_rockets"),
        ("cells", "-qnn_inv_cells"),
        ("infinite_ammo", "-qnn_inv_infinite_ammo"),
        ("health", "-qnn_inv_health"),
        ("armor_value", "-qnn_inv_armor"),
        ("armor_type", "-qnn_inv_armor_type"),
    )

    @classmethod
    def _weapon_args(
        cls, weapon_config: Mapping[str, object] | None
    ) -> tuple[str, ...]:
        """Turn a single-weapon arena loadout dict into deterministic server CLI
        args. Only keys present in ``weapon_config`` are emitted; every other
        inventory cvar keeps its registered engine default. Unknown keys fail
        loud (no silent drop)."""
        if not weapon_config:
            return ()
        known = {key for key, _ in cls._WEAPON_ARG_FLAGS}
        unknown = set(weapon_config) - known
        if unknown:
            raise ValueError(f"unknown arena weapon_config keys: {sorted(unknown)}")
        args: list[str] = []
        for key, flag in cls._WEAPON_ARG_FLAGS:
            if key in weapon_config and weapon_config[key] is not None:
                args.extend((flag, str(weapon_config[key])))
        return tuple(args)

    def start_listening(self) -> None:
        self.spawn()
        state = self.read_json()
        if state.get("state") != "listening":
            raise NativeEngineError(f"Unexpected arena server state {state!r}")
        self.attach_declarations()

    def attach_declarations(self) -> None:
        """Attach every declared observer seat's obs plan (OP_ATTACH_DECL).

        Sent between the "listening" and "ready" states so the engine
        compiles per-seat emit plans before the first observer drain.
        Seats without a declaration send nothing and keep the default
        plan — the legacy protocol byte-for-byte.
        """
        for index, declaration in enumerate(self.observer_declarations):
            if declaration is not None:
                _attach_seat_declaration(self, declaration, seat_index=index)

    def wait_ready(self) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        state = self.read_json()
        if state.get("state") != "ready":
            raise NativeEngineError(f"Unexpected arena server state {state!r}")
        return self._read_observer_transitions()

    def _read_observer_transitions(
        self,
    ) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        if getattr(self, "observer_mode", "external") == "external":
            self.last_training_frames = ()
            return []
        transitions: list[tuple[bytes, TrustedTrainingExtrasV1]] = []
        frames: list[bytes] = []
        for index, (match_id, seat_id) in enumerate(self.observer_seats):
            obs = self.read_exact(self._observer_frame_bytes[index])
            extras, raw_frame = self.read_training_frame(
                episode_id=(
                    f"arena:{match_id}:{seat_id}:"
                    f"{self._episode_generations[index]}"
                )
            )
            transitions.append((obs, extras))
            frames.append(raw_frame)
            if extras.done:
                self._episode_generations[index] += 1
        self.last_training_frames = tuple(frames)
        return transitions

    def step(self) -> None:
        self.write(bytes((self.OP_STEP,)))
        if self.read_exact(1) != bytes((self.OP_STEP,)):
            raise NativeEngineError("Invalid arena server step acknowledgement")

    def step_batch(
        self,
        action_packets: Sequence[bytes | bytearray | memoryview],
    ) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        count = len(action_packets)
        if not 1 <= count <= 16:
            raise ValueError("arena action batch must contain 1..16 seats")
        payload = bytearray((self.OP_STEP_BATCH, count))
        for action_packet in action_packets:
            packet = memoryview(action_packet)
            if (len(packet) != NativeObsBufferProcess._ACTION_PACKET_SIZE
                    or packet[0] != ArenaClientProcess.OP_STAGE):
                raise ValueError(
                    "arena action packet must be the packed qnn_action_t step "
                    f"format ({NativeObsBufferProcess._ACTION_PACKET_SIZE} bytes)")
            payload.extend(packet[1:])
        self.write(payload)
        if self.read_exact(1) != bytes((self.OP_STEP_BATCH,)):
            raise NativeEngineError("Invalid arena server batch-step acknowledgement")
        return self._read_observer_transitions()

    def reset_matches(
        self, match_mask: int
    ) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        mask = int(match_mask)
        if not 0 < mask <= 0xFF:
            raise ValueError("match_mask must select at least one of eight matches")
        self.write(bytes((self.OP_RESET_MASK, mask)))
        if self.read_exact(1) != bytes((self.OP_RESET_MASK,)):
            raise NativeEngineError("Invalid arena server reset acknowledgement")
        return self._read_observer_transitions()

    def close(self) -> None:
        self.stop(self.OP_SHUTDOWN)


class ArenaClientProcess(_ArenaPipeProcess):
    """Headless remote NQ client for one externally controlled arena seat."""

    OP_STAGE = 1
    OP_RECEIVE = 4
    OP_RESUME_SIGNON = 5
    OP_RECEIVE_RESET = 6
    OP_STAGE_LOCAL = 7
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
        declaration: object = None,
    ) -> None:
        client_env = dict(env or {})
        client_env["QNN_REWARD_JSON"] = str(reward_json)
        self.match_id = int(match_id)
        self.seat_id = int(seat_id)
        self.declaration, self._frame_bytes = _compile_seat_declaration(declaration)
        self._episode_generation = 0
        self.last_training_frame = b""
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
        self.attach_declaration()

    def attach_declaration(self) -> None:
        """Attach this seat's obs plan (OP_ATTACH_DECL), if declared.

        Sent between "signed_on" and OP_RESUME_SIGNON so the engine
        compiles the emit plan before the first transition drain. No
        declaration = nothing sent, default plan, legacy protocol.
        """
        if self.declaration is not None:
            _attach_seat_declaration(self, self.declaration, seat_index=0)

    def resume_signon(self) -> None:
        self.write(bytes((self.OP_RESUME_SIGNON,)))

    def wait_ready(self) -> tuple[bytes, TrustedTrainingExtrasV1]:
        state = self.read_json()
        if state.get("state") != "ready":
            raise NativeEngineError(f"Unexpected arena client state {state!r}")
        return self._read_transition()

    def _read_training(self) -> TrustedTrainingExtrasV1:
        extras, self.last_training_frame = self.read_training_frame(
            episode_id=f"arena:{self.match_id}:{self.seat_id}:{self._episode_generation}",
        )
        return extras

    def _read_transition(self) -> tuple[bytes, TrustedTrainingExtrasV1]:
        obs = self.read_exact(self._frame_bytes)
        return obs, self._read_training()

    def stage(self, action_packet: bytes | bytearray | memoryview) -> None:
        packet = memoryview(action_packet)
        if (len(packet) != NativeObsBufferProcess._ACTION_PACKET_SIZE
                or packet[0] != self.OP_STAGE):
            raise ValueError(
                "arena action packet must be the packed qnn_action_t step "
                f"format ({NativeObsBufferProcess._ACTION_PACKET_SIZE} bytes)")
        self.write(packet)

    def wait_staged(self) -> None:
        if self.read_exact(1) != bytes((self.OP_STAGE,)):
            raise NativeEngineError("Invalid arena client stage acknowledgement")

    def stage_local(self, action_packet: bytes | bytearray | memoryview) -> None:
        packet = memoryview(action_packet)
        if (len(packet) != NativeObsBufferProcess._ACTION_PACKET_SIZE
                or packet[0] != self.OP_STAGE):
            raise ValueError(
                "arena action packet must be the packed qnn_action_t step "
                f"format ({NativeObsBufferProcess._ACTION_PACKET_SIZE} bytes)")
        payload = bytearray(packet)
        payload[0] = self.OP_STAGE_LOCAL
        self.write(payload)

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
        direct_actions: bool,
        observer_mode: str = "external",
        env: Mapping[str, str] | None = None,
        workdir: str | Path | None = None,
        weapon_config: Mapping[str, object] | None = None,
        declarations: Sequence[object] | None = None,
    ) -> None:
        seats = tuple((int(match_id), int(seat_id)) for match_id, seat_id in external_seats)
        if observer_mode not in {"external", "virtual", "shadow"}:
            raise ValueError(f"unknown arena observer mode {observer_mode!r}")
        if observer_mode != "external" and not direct_actions:
            raise ValueError("virtual and shadow observers require direct actions")
        # Per-seat obs declarations (obs_api v1), aligned with
        # external_seats. Seats may differ — that heterogeneity is the
        # point of the negotiated layout (cross-arch H2H). None (the
        # default everywhere today) keeps every seat on the legacy
        # default plan with nothing sent on the wire.
        if declarations is None:
            declarations = (None,) * len(seats)
        if len(declarations) != len(seats):
            raise ValueError(
                f"declarations has {len(declarations)} entries for {len(seats)} seats"
            )
        self.direct_actions = bool(direct_actions)
        self.observer_mode = observer_mode
        self.shadow_checks = 0
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
            observer_seats=seats,
            observer_mode=observer_mode,
            reward_json=reward_json,
            weapon_config=weapon_config,
            observer_declarations=declarations,
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
                declaration=declaration,
            )
            for (match_id, seat_id), declaration in zip(seats, declarations, strict=True)
            if observer_mode != "virtual"
        )

    @staticmethod
    def _first_difference(left: bytes, right: bytes) -> int:
        return next(
            (index for index, pair in enumerate(zip(left, right, strict=True)) if pair[0] != pair[1]),
            -1,
        )

    def _check_shadow(
        self,
        external: Sequence[tuple[bytes, TrustedTrainingExtrasV1]],
        shadow: Sequence[tuple[bytes, TrustedTrainingExtrasV1]],
    ) -> None:
        if len(external) != len(shadow):
            raise NativeEngineError("arena shadow returned the wrong seat count")
        for index, ((external_obs, external_extras), (shadow_obs, shadow_extras)) in enumerate(
            zip(external, shadow, strict=True)
        ):
            if external_obs != shadow_obs:
                offset = self._first_difference(external_obs, shadow_obs)
                raise NativeEngineError(
                    f"arena shadow obs mismatch at seat {index}, byte {offset}: "
                    f"external={external_obs[offset]} shadow={shadow_obs[offset]}"
                )
            external_frame = self.clients[index].last_training_frame
            shadow_frame = self.server.last_training_frames[index]
            if external_frame != shadow_frame or external_extras != shadow_extras:
                offset = (
                    self._first_difference(external_frame, shadow_frame)
                    if len(external_frame) == len(shadow_frame)
                    else min(len(external_frame), len(shadow_frame))
                )
                raise NativeEngineError(
                    f"arena shadow QTRN mismatch at seat {index}, byte {offset}: "
                    f"external={external_frame[offset] if offset < len(external_frame) else 'EOF'} "
                    f"shadow={shadow_frame[offset] if offset < len(shadow_frame) else 'EOF'}"
                )
        self.shadow_checks += 1

    def start(self) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        self.server.start_listening()
        try:
            if self.observer_mode == "virtual":
                return self.server.wait_ready()
            for client in self.clients:
                client.start_connecting()
                # Stock NetQuake's reliable sign-on stream is effectively
                # serial.  Finish each one-time handshake before admitting
                # the next seat; steady-state stepping remains parallel.
                client.wait_signed_on()
            for client in self.clients:
                client.resume_signon()
            shadow = self.server.wait_ready()
            external = [client.wait_ready() for client in self.clients]
            if self.observer_mode == "shadow":
                # Initial readiness is asynchronous: the external process parks
                # on whichever ready datagram the OS schedules first, while the
                # mirror parser runs in the server frame loop.  Re-broadcast a
                # reset snapshot as an explicit barrier before byte comparison.
                match_mask = 0
                for match_id, _seat_id in self.server.observer_seats:
                    match_mask |= 1 << match_id
                shadow = self.server.reset_matches(match_mask)
                for client in self.clients:
                    client.receive_reset_send()
                external = [client.receive_recv() for client in self.clients]
                self._check_shadow(external, shadow)
            return external
        except Exception as exc:
            self.close()
            server_details = self.server.last_stderr[-4000:] if self.server.last_stderr else ""
            client_details = [
                f"seat {client.match_id}:{client.seat_id}: {client.last_stderr[-4000:]}"
                for client in self.clients
                if client.last_stderr
            ]
            if server_details or client_details:
                diagnostics = []
                if server_details:
                    diagnostics.append(f"Arena server diagnostics:\n{server_details}")
                if client_details:
                    diagnostics.append("Arena client diagnostics:\n" + "\n".join(client_details))
                raise NativeEngineError(
                    f"{exc}\n" + "\n".join(diagnostics)
                ) from exc
            raise

    def step_many(
        self,
        action_packets: Sequence[bytes | bytearray | memoryview],
    ) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        """Advance every policy seat through exactly one shared world tick."""
        observer_mode = getattr(self, "observer_mode", "external")
        if len(action_packets) != len(self.clients):
            if observer_mode != "virtual" or len(action_packets) != len(self.server.observer_seats):
                raise ValueError("one action packet is required for each external arena seat")
        if observer_mode == "virtual":
            return self.server.step_batch(action_packets)
        if self.direct_actions:
            for client, packet in zip(self.clients, action_packets, strict=True):
                client.stage_local(packet)
            shadow = self.server.step_batch(action_packets)
        else:
            for client, packet in zip(self.clients, action_packets, strict=True):
                client.stage(packet)
            for client in self.clients:
                client.wait_staged()
            self.server.step()
        for client in self.clients:
            client.receive_send()
        external = [client.receive_recv() for client in self.clients]
        if observer_mode == "shadow":
            self._check_shadow(external, shadow)
        return external

    def step(
        self,
        action_packets: Sequence[bytes | bytearray | memoryview],
    ) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        """Compatibility alias for callers predating native ``step_many``."""
        return self.step_many(action_packets)

    def reset_matches(self, match_mask: int) -> list[tuple[bytes, TrustedTrainingExtrasV1]]:
        shadow = self.server.reset_matches(match_mask)
        if self.observer_mode == "virtual":
            return shadow
        for client in self.clients:
            client.receive_reset_send()
        external = [client.receive_recv() for client in self.clients]
        if self.observer_mode == "shadow":
            self._check_shadow(external, shadow)
        return external

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.server.close()
