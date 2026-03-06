"""Minimal NetQuake demo parsing for offline imitation data extraction."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from quake_ai.actions import ActionLabels

U_MOREBITS = 1 << 0
U_ORIGIN1 = 1 << 1
U_ORIGIN2 = 1 << 2
U_ORIGIN3 = 1 << 3
U_ANGLE2 = 1 << 4
U_FRAME = 1 << 6
U_ANGLE1 = 1 << 8
U_ANGLE3 = 1 << 9
U_MODEL = 1 << 10
U_COLORMAP = 1 << 11
U_SKIN = 1 << 12
U_EFFECTS = 1 << 13
U_LONGENTITY = 1 << 14

SU_VIEWHEIGHT = 1 << 0
SU_IDEALPITCH = 1 << 1
SU_PUNCH1 = 1 << 2
SU_VELOCITY1 = 1 << 5
SU_ITEMS = 1 << 9
SU_WEAPONFRAME = 1 << 12
SU_ARMOR = 1 << 13
SU_WEAPON = 1 << 14

SVC_NOP = 1
SVC_DISCONNECT = 2
SVC_UPDATESTAT = 3
SVC_VERSION = 4
SVC_SETVIEW = 5
SVC_SOUND = 6
SVC_TIME = 7
SVC_PRINT = 8
SVC_STUFFTEXT = 9
SVC_SETANGLE = 10
SVC_SERVERINFO = 11
SVC_LIGHTSTYLE = 12
SVC_UPDATENAME = 13
SVC_UPDATEFRAGS = 14
SVC_CLIENTDATA = 15
SVC_STOPSOUND = 16
SVC_UPDATECOLORS = 17
SVC_PARTICLE = 18
SVC_DAMAGE = 19
SVC_SPAWNSTATIC = 20
SVC_SPAWNBASELINE = 22
SVC_TEMP_ENTITY = 23
SVC_SETPAUSE = 24
SVC_SIGNONNUM = 25
SVC_CENTERPRINT = 26
SVC_KILLEDMONSTER = 27
SVC_FOUNDSECRET = 28
SVC_SPAWNSTATICSOUND = 29
SVC_INTERMISSION = 30
SVC_FINALE = 31
SVC_CDTRACK = 32
SVC_SELLSCREEN = 33
SVC_CUTSCENE = 34

STAT_HEALTH = 0
STAT_WEAPON = 2
STAT_AMMO = 3
STAT_ARMOR = 4
STAT_WEAPONFRAME = 5
STAT_SHELLS = 6
STAT_ACTIVEWEAPON = 10

TE_SIMPLE_COORD = {0, 1, 2, 3, 4, 7, 8, 10, 11}
TE_BEAM = {5, 6, 9, 13}


def _normalize_yaw(delta: float) -> float:
    while delta <= -180.0:
        delta += 360.0
    while delta > 180.0:
        delta -= 360.0
    return delta


def _yaw_forward(yaw: float) -> tuple[float, float]:
    radians = math.radians(yaw)
    return math.cos(radians), math.sin(radians)


def _active_weapon_id(active_weapon: int, weapon_model: int) -> int:
    if active_weapon > 0 and active_weapon & (active_weapon - 1) == 0:
        return int(math.log2(active_weapon)) + 1
    return max(int(weapon_model), 0)


def infer_action_label(
    current_pos: Sequence[float],
    next_pos: Sequence[float],
    current_yaw: float,
    next_yaw: float,
    terminal_use: bool,
    movement_threshold: float = 4.0,
    turn_threshold: float = 2.0,
) -> Dict[str, int]:
    dx = float(next_pos[0]) - float(current_pos[0])
    dy = float(next_pos[1]) - float(current_pos[1])
    forward_x, forward_y = _yaw_forward(current_yaw)
    left_x, left_y = -forward_y, forward_x

    forward_proj = dx * forward_x + dy * forward_y
    left_proj = dx * left_x + dy * left_y
    yaw_delta = _normalize_yaw(float(next_yaw) - float(current_yaw))

    move = 0
    if forward_proj > movement_threshold:
        move = 1
    elif forward_proj < -movement_threshold:
        move = 2

    strafe = 0
    if left_proj > movement_threshold:
        strafe = 1
    elif left_proj < -movement_threshold:
        strafe = 2

    turn = 0
    if yaw_delta > turn_threshold:
        turn = 1
    elif yaw_delta < -turn_threshold:
        turn = 2

    return ActionLabels(move=move, strafe=strafe, turn=turn, use=1 if terminal_use else 0).to_dict()


@dataclass(slots=True)
class ParsedDemoTick:
    time_s: float
    player_pos: List[float]
    player_vel: List[float]
    yaw: float
    health: int
    armor: int
    ammo: int
    weapon_id: int
    nearby_item_flags: List[int]
    goal_progress: float
    action_label: Dict[str, int]
    done: bool
    done_reason: str
    packet: Dict[str, object]


@dataclass(slots=True)
class ParsedDemoEpisode:
    episode_id: str
    map_id: str
    ticks: List[ParsedDemoTick]


@dataclass(slots=True)
class _Baseline:
    modelindex: int = 0
    frame: int = 0
    colormap: int = 0
    skin: int = 0
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass(slots=True)
class _EntityState:
    baseline: _Baseline = field(default_factory=_Baseline)
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    msg_origins: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    msg_angles: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@dataclass(slots=True)
class _FrameRecord:
    time_s: float
    player_pos: List[float]
    player_vel: List[float]
    yaw: float
    health: int
    armor: int
    ammo: int
    weapon_id: int
    packet: Dict[str, object]
    intermission: bool


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def _read(self, size: int) -> bytes:
        if self.offset + size > len(self.payload):
            raise ValueError("Unexpected end of demo message")
        out = self.payload[self.offset : self.offset + size]
        self.offset += size
        return out

    def byte(self) -> int:
        return struct.unpack("<B", self._read(1))[0]

    def char(self) -> int:
        return struct.unpack("<b", self._read(1))[0]

    def short(self) -> int:
        return struct.unpack("<h", self._read(2))[0]

    def ushort(self) -> int:
        return struct.unpack("<H", self._read(2))[0]

    def long(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def float(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def coord(self) -> float:
        return self.short() * (1.0 / 8.0)

    def angle(self) -> float:
        return self.char() * (360.0 / 256.0)

    def string(self) -> str:
        chars: List[int] = []
        while True:
            value = self.char()
            if value in (-1, 0):
                break
            chars.append(value & 0xFF)
        return bytes(chars).decode("latin-1", errors="replace")


def _parse_baseline(reader: _Reader, entity: _EntityState) -> None:
    entity.baseline.modelindex = reader.byte()
    entity.baseline.frame = reader.byte()
    entity.baseline.colormap = reader.byte()
    entity.baseline.skin = reader.byte()
    for axis in range(3):
        entity.baseline.origin[axis] = reader.coord()
        entity.baseline.angles[axis] = reader.angle()


def _skip_sound(reader: _Reader) -> None:
    field_mask = reader.byte()
    if field_mask & 1:
        reader.byte()
    if field_mask & 2:
        reader.byte()
    reader.short()
    reader.byte()
    for _ in range(3):
        reader.coord()


def _skip_particle(reader: _Reader) -> None:
    for _ in range(3):
        reader.coord()
    for _ in range(3):
        reader.char()
    reader.byte()
    reader.byte()


def _skip_damage(reader: _Reader) -> None:
    reader.byte()
    reader.byte()
    for _ in range(3):
        reader.coord()


def _skip_temp_entity(reader: _Reader) -> None:
    temp_type = reader.byte()
    if temp_type in TE_SIMPLE_COORD:
        for _ in range(3):
            reader.coord()
        return
    if temp_type in TE_BEAM:
        reader.short()
        for _ in range(6):
            reader.coord()
        return
    if temp_type == 12:
        for _ in range(3):
            reader.coord()
        reader.byte()
        reader.byte()
        return
    raise ValueError(f"Unsupported temp entity type {temp_type}")


def _parse_serverinfo(reader: _Reader) -> None:
    reader.long()
    reader.string()
    while reader.string():
        pass
    while reader.string():
        pass


def _parse_clientdata(reader: _Reader, stats: List[int]) -> List[float]:
    bits = reader.ushort()
    if bits & SU_VIEWHEIGHT:
        reader.char()
    if bits & SU_IDEALPITCH:
        reader.char()

    velocity = [0.0, 0.0, 0.0]
    for axis in range(3):
        if bits & (SU_PUNCH1 << axis):
            reader.char()
        if bits & (SU_VELOCITY1 << axis):
            velocity[axis] = reader.char() * 16.0

    if bits & SU_ITEMS:
        reader.long()
    else:
        reader.long()

    stats[STAT_WEAPONFRAME] = reader.byte() if (bits & SU_WEAPONFRAME) else 0
    stats[STAT_ARMOR] = reader.byte() if (bits & SU_ARMOR) else 0
    stats[STAT_WEAPON] = reader.byte() if (bits & SU_WEAPON) else 0
    stats[STAT_HEALTH] = reader.short()
    stats[STAT_AMMO] = reader.byte()
    for slot in range(4):
        stats[STAT_SHELLS + slot] = reader.byte()
    stats[STAT_ACTIVEWEAPON] = reader.byte()
    return velocity


def _parse_update(reader: _Reader, entities: Dict[int, _EntityState], bits: int) -> None:
    if bits & U_MOREBITS:
        bits |= reader.byte() << 8

    entity_num = reader.short() if (bits & U_LONGENTITY) else reader.byte()
    entity = entities.setdefault(entity_num, _EntityState())

    if bits & U_MODEL:
        reader.byte()
    if bits & U_FRAME:
        reader.byte()
    if bits & U_COLORMAP:
        reader.byte()
    if bits & U_SKIN:
        reader.byte()
    if bits & U_EFFECTS:
        reader.byte()

    entity.msg_origins[1] = entity.msg_origins[0][:]
    entity.msg_angles[1] = entity.msg_angles[0][:]

    if bits & U_ORIGIN1:
        entity.msg_origins[0][0] = reader.coord()
    else:
        entity.msg_origins[0][0] = entity.baseline.origin[0]
    if bits & U_ANGLE1:
        entity.msg_angles[0][0] = reader.angle()
    else:
        entity.msg_angles[0][0] = entity.baseline.angles[0]

    if bits & U_ORIGIN2:
        entity.msg_origins[0][1] = reader.coord()
    else:
        entity.msg_origins[0][1] = entity.baseline.origin[1]
    if bits & U_ANGLE2:
        entity.msg_angles[0][1] = reader.angle()
    else:
        entity.msg_angles[0][1] = entity.baseline.angles[1]

    if bits & U_ORIGIN3:
        entity.msg_origins[0][2] = reader.coord()
    else:
        entity.msg_origins[0][2] = entity.baseline.origin[2]
    if bits & U_ANGLE3:
        entity.msg_angles[0][2] = reader.angle()
    else:
        entity.msg_angles[0][2] = entity.baseline.angles[2]

    entity.origin = entity.msg_origins[0][:]
    entity.angles = entity.msg_angles[0][:]


def _build_ticks(frames: List[_FrameRecord], terminated_early: bool = False) -> List[ParsedDemoTick]:
    if not frames:
        return []

    trim_start = 0
    while trim_start < len(frames) - 1:
        current = frames[trim_start]
        next_frame = frames[trim_start + 1]
        current_is_origin = all(abs(value) <= 1e-6 for value in current.player_pos)
        next_jump = math.dist(current.player_pos, next_frame.player_pos)
        if current.health <= 0 and current.ammo <= 0 and current.weapon_id <= 0 and current_is_origin:
            trim_start += 1
            continue
        if current_is_origin and next_jump >= 64.0:
            trim_start += 1
            continue
        break

    frames = frames[trim_start:]
    if not frames:
        return []

    goal_pos = frames[-1].player_pos
    max_dist = max(
        math.dist(frame.player_pos, goal_pos)
        for frame in frames
    )
    if max_dist <= 1e-6:
        max_dist = 1.0

    ticks: List[ParsedDemoTick] = []
    for index, frame in enumerate(frames):
        previous = frames[max(0, index - 1)]
        next_frame = frames[min(index + 1, len(frames) - 1)]
        health_gain = int(frame.health > previous.health)
        armor_gain = int(frame.armor > previous.armor)
        ammo_gain = int(frame.ammo > previous.ammo)
        weapon_gain = int(frame.weapon_id != previous.weapon_id and frame.weapon_id > 0)
        done = index == len(frames) - 1
        goal_progress = 1.0 - (math.dist(frame.player_pos, goal_pos) / max_dist)
        action = infer_action_label(
            current_pos=frame.player_pos,
            next_pos=next_frame.player_pos,
            current_yaw=frame.yaw,
            next_yaw=next_frame.yaw,
            terminal_use=done,
        )
        ticks.append(
            ParsedDemoTick(
                time_s=frame.time_s,
                player_pos=frame.player_pos[:],
                player_vel=frame.player_vel[:],
                yaw=frame.yaw,
                health=frame.health,
                armor=frame.armor,
                ammo=frame.ammo,
                weapon_id=frame.weapon_id,
                nearby_item_flags=[health_gain, armor_gain, ammo_gain, weapon_gain],
                goal_progress=float(max(0.0, min(goal_progress, 1.0))),
                action_label=action,
                done=done,
                done_reason=(
                    "goal_reached"
                    if done and frame.intermission
                    else ("truncated_demo" if done and terminated_early else ("demo_end" if done else ""))
                ),
                packet=dict(frame.packet),
            )
        )
    return ticks


def parse_netquake_demo(path: str | Path, map_id: str, episode_id: str | None = None) -> ParsedDemoEpisode:
    source = Path(path)
    payload = source.read_bytes()
    if b"\n" not in payload:
        raise ValueError(f"Invalid demo header in {source}")

    header_end = payload.index(b"\n") + 1
    body = payload[header_end:]
    offset = 0

    entities: Dict[int, _EntityState] = {}
    stats = [0] * 32
    viewentity = 1
    message_index = 0
    current_time = 0.0
    latest_velocity = [0.0, 0.0, 0.0]
    intermission = False
    frames: List[_FrameRecord] = []
    terminated_early = False

    while offset + 16 <= len(body):
        message_size = struct.unpack("<i", body[offset : offset + 4])[0]
        offset += 4
        header_angles = list(struct.unpack("<fff", body[offset : offset + 12]))
        offset += 12
        if message_size < 0 or offset + message_size > len(body):
            if frames:
                terminated_early = True
                break
            raise ValueError(f"Invalid demo message length in {source}")

        message = body[offset : offset + message_size]
        offset += message_size
        reader = _Reader(message)
        saw_disconnect = False
        parse_error: ValueError | None = None

        while reader.offset < len(message):
            try:
                cmd = reader.byte()
                if cmd == 255:
                    break
                if cmd & 128:
                    _parse_update(reader, entities, cmd & 127)
                    continue

                if cmd == SVC_NOP:
                    continue
                if cmd == SVC_DISCONNECT:
                    saw_disconnect = True
                    break
                if cmd == SVC_UPDATESTAT:
                    stats[reader.byte()] = reader.long()
                    continue
                if cmd == SVC_VERSION:
                    reader.long()
                    continue
                if cmd == SVC_SETVIEW:
                    viewentity = reader.short()
                    continue
                if cmd == SVC_SOUND:
                    _skip_sound(reader)
                    continue
                if cmd == SVC_TIME:
                    current_time = reader.float()
                    continue
                if cmd in {SVC_PRINT, SVC_STUFFTEXT, SVC_CENTERPRINT, SVC_FINALE, SVC_CUTSCENE}:
                    reader.string()
                    if cmd in {SVC_FINALE, SVC_CUTSCENE}:
                        intermission = True
                    continue
                if cmd == SVC_SETANGLE:
                    reader.angle()
                    reader.angle()
                    reader.angle()
                    continue
                if cmd == SVC_SERVERINFO:
                    _parse_serverinfo(reader)
                    continue
                if cmd == SVC_LIGHTSTYLE:
                    reader.byte()
                    reader.string()
                    continue
                if cmd == SVC_UPDATENAME:
                    reader.byte()
                    reader.string()
                    continue
                if cmd == SVC_UPDATEFRAGS:
                    reader.byte()
                    reader.short()
                    continue
                if cmd == SVC_CLIENTDATA:
                    latest_velocity = _parse_clientdata(reader, stats)
                    continue
                if cmd == SVC_STOPSOUND:
                    reader.short()
                    continue
                if cmd == SVC_UPDATECOLORS:
                    reader.byte()
                    reader.byte()
                    continue
                if cmd == SVC_PARTICLE:
                    _skip_particle(reader)
                    continue
                if cmd == SVC_DAMAGE:
                    _skip_damage(reader)
                    continue
                if cmd == SVC_SPAWNSTATIC:
                    _parse_baseline(reader, _EntityState())
                    continue
                if cmd == SVC_SPAWNBASELINE:
                    _parse_baseline(reader, entities.setdefault(reader.short(), _EntityState()))
                    continue
                if cmd == SVC_TEMP_ENTITY:
                    _skip_temp_entity(reader)
                    continue
                if cmd == SVC_SETPAUSE:
                    reader.byte()
                    continue
                if cmd == SVC_SIGNONNUM:
                    reader.byte()
                    continue
                if cmd in {SVC_KILLEDMONSTER, SVC_FOUNDSECRET, SVC_INTERMISSION, SVC_SELLSCREEN}:
                    if cmd == SVC_INTERMISSION:
                        intermission = True
                    continue
                if cmd == SVC_SPAWNSTATICSOUND:
                    for _ in range(3):
                        reader.coord()
                    reader.byte()
                    reader.byte()
                    reader.byte()
                    continue
                if cmd == SVC_CDTRACK:
                    reader.byte()
                    reader.byte()
                    continue
                raise ValueError(f"Unsupported server message {cmd} in {source}")
            except ValueError as exc:
                parse_error = exc
                break

        entity = entities.get(viewentity) or entities.get(1)
        if entity is not None:
            candidate = _FrameRecord(
                time_s=current_time,
                player_pos=entity.origin[:],
                player_vel=list(latest_velocity),
                yaw=float(header_angles[1]),
                health=int(stats[STAT_HEALTH]),
                armor=int(stats[STAT_ARMOR]),
                ammo=int(stats[STAT_AMMO]),
                weapon_id=_active_weapon_id(stats[STAT_ACTIVEWEAPON], stats[STAT_WEAPON]),
                packet={
                    "tick_estimate": len(frames),
                    "direction": "server_to_client",
                    "seq": message_index,
                    "ack": max(0, message_index - 1),
                    "payload_hex": message[:16].hex(),
                },
                intermission=intermission,
            )
            previous = frames[-1] if frames else None
            if previous is None or (
                candidate.time_s != previous.time_s
                or candidate.player_pos != previous.player_pos
                or abs(candidate.yaw - previous.yaw) > 1e-6
                or candidate.health != previous.health
                or candidate.armor != previous.armor
                or candidate.ammo != previous.ammo
                or candidate.weapon_id != previous.weapon_id
            ):
                frames.append(candidate)

        message_index += 1
        if parse_error is not None:
            if frames:
                terminated_early = True
                break
            raise parse_error
        if saw_disconnect:
            break

    return ParsedDemoEpisode(
        episode_id=episode_id or source.stem,
        map_id=map_id,
        ticks=_build_ticks(frames, terminated_early=terminated_early),
    )
