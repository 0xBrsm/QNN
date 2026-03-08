"""Minimal NetQuake demo parsing for offline imitation data extraction.

This parser now preserves match metadata needed for competitive demo
classification: serverinfo payload, player-name updates, player-color updates,
frag updates, and basic duration / tick counts. Downstream consumers can use
these signals to separate competitive FFA/teamplay material from misc demos.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from quake_ai.actions import (
    ActionLabels,
    clamp_weapon_switch,
    look_label_from_pitch_delta,
    look_label_from_yaw_delta,
)

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

_TEXT_CVAR_PATTERNS = {
    "teamplay": re.compile(r"\bteamplay\b\s*:?\s*(-?\d+)\b", re.IGNORECASE),
    "deathmatch": re.compile(r"\bdeathmatch\b\s*:?\s*(-?\d+)\b", re.IGNORECASE),
    "fraglimit": re.compile(r"\bfraglimit\b\s*:?\s*(-?\d+)\b", re.IGNORECASE),
    "timelimit": re.compile(r"\btimelimit\b\s*:?\s*(-?\d+)\b", re.IGNORECASE),
}
_QUOTED_CVAR_CHANGE_PATTERN = re.compile(
    r'"?(teamplay|deathmatch|fraglimit|timelimit)"?\s+changed\s+to\s+"?(-?\d+)"?',
    re.IGNORECASE,
)
_TEAM_NAMES = ("red", "yellow", "blue", "green")
_TEAM_MENU_PATTERNS = (
    re.compile(r"\bfor red team\b.*\bfor blue team\b", re.IGNORECASE),
    re.compile(r"\bfor red team\b.*\bfor yellow team\b", re.IGNORECASE),
)
_TEAM_ASSIGN_PATTERN = re.compile(
    r"\b(?:has joined the|set to)\s+(red|yellow|blue|green)\s+team\b",
    re.IGNORECASE,
)
_TEAM_SCORE_PATTERN = re.compile(
    r"\bthe\s+(red|yellow|blue|green)\s+team\s+has\s+-?\d+\s+frags?\b",
    re.IGNORECASE,
)
_TEAM_WIN_PATTERN = re.compile(
    r"\bthe\s+(red|yellow|blue|green)\s+team\s+has\s+won\b",
    re.IGNORECASE,
)
_DUEL_PATTERN = re.compile(r"\bduel\b", re.IGNORECASE)


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
    terminal_use: bool = False,
    current_pitch: float = 0.0,
    next_pitch: float = 0.0,
    current_weapon_id: int = 0,
    next_weapon_id: int = 0,
    current_ammo: int = 0,
    next_ammo: int = 0,
    current_velocity: Sequence[float] | None = None,
    next_velocity: Sequence[float] | None = None,
    movement_threshold: float = 4.0,
    jump_velocity_threshold: float = 160.0,
) -> Dict[str, int]:
    del terminal_use
    dx = float(next_pos[0]) - float(current_pos[0])
    dy = float(next_pos[1]) - float(current_pos[1])
    forward_x, forward_y = _yaw_forward(current_yaw)
    left_x, left_y = -forward_y, forward_x

    forward_proj = dx * forward_x + dy * forward_y
    left_proj = dx * left_x + dy * left_y
    yaw_delta = _normalize_yaw(float(next_yaw) - float(current_yaw))
    pitch_delta = float(next_pitch) - float(current_pitch)

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

    fire = int(int(next_ammo) < int(current_ammo))
    jump = 0
    if current_velocity is not None and next_velocity is not None:
        current_vz = float(current_velocity[2]) if len(current_velocity) > 2 else 0.0
        next_vz = float(next_velocity[2]) if len(next_velocity) > 2 else 0.0
        if current_vz <= 32.0 and next_vz >= jump_velocity_threshold:
            jump = 1

    return ActionLabels(
        move=move,
        strafe=strafe,
        look_yaw=look_label_from_yaw_delta(yaw_delta),
        look_pitch=look_label_from_pitch_delta(pitch_delta),
        fire=fire,
        jump=jump,
        weapon=clamp_weapon_switch(next_weapon_id if next_weapon_id != current_weapon_id else 0),
    ).to_dict()


@dataclass(slots=True)
class ParsedDemoTick:
    time_s: float
    player_pos: List[float]
    player_vel: List[float]
    view_angles: List[float]
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
    visible_entities: List[Dict[str, object]] = field(default_factory=list)
    events: List[Dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDemoEpisode:
    episode_id: str
    map_id: str
    ticks: List[ParsedDemoTick]
    serverinfo: Dict[str, object] = field(default_factory=dict)
    cvars: Dict[str, object] = field(default_factory=dict)
    player_names: Dict[int, str] = field(default_factory=dict)
    player_colors: Dict[int, int] = field(default_factory=dict)
    frag_updates: List[Dict[str, object]] = field(default_factory=list)
    text_flags: Dict[str, bool] = field(default_factory=dict)
    maxclients: int = 0
    duration_s: float = 0.0
    tick_count: int = 0


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
    modelindex: int = 0
    frame: int = 0
    colormap: int = 0
    skin: int = 0
    effects: int = 0
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    msg_origins: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    msg_angles: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@dataclass(slots=True)
class _FrameRecord:
    time_s: float
    player_pos: List[float]
    player_vel: List[float]
    view_angles: List[float]
    yaw: float
    health: int
    armor: int
    ammo: int
    weapon_id: int
    packet: Dict[str, object]
    intermission: bool
    visible_entities: List[Dict[str, object]]
    events: List[Dict[str, object]]


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
    entity.modelindex = entity.baseline.modelindex
    entity.frame = entity.baseline.frame
    entity.colormap = entity.baseline.colormap
    entity.skin = entity.baseline.skin
    entity.origin = entity.baseline.origin[:]
    entity.angles = entity.baseline.angles[:]
    entity.msg_origins[0] = entity.origin[:]
    entity.msg_origins[1] = entity.origin[:]
    entity.msg_angles[0] = entity.angles[:]
    entity.msg_angles[1] = entity.angles[:]


def _read_sound(reader: _Reader) -> Dict[str, object]:
    field_mask = reader.byte()
    volume = None
    attenuation = None
    if field_mask & 1:
        volume = reader.byte()
    if field_mask & 2:
        attenuation = reader.byte()
    channel = reader.short()
    sound_num = reader.byte()
    origin = [reader.coord() for _ in range(3)]
    return {
        "type": "sound",
        "channel": channel,
        "sound_num": sound_num,
        "origin": origin,
        "volume": volume,
        "attenuation": attenuation,
    }


def _read_particle(reader: _Reader) -> Dict[str, object]:
    origin = [reader.coord() for _ in range(3)]
    direction = [reader.char() for _ in range(3)]
    count = reader.byte()
    color = reader.byte()
    return {
        "type": "particle",
        "origin": origin,
        "direction": direction,
        "count": count,
        "color": color,
    }


def _read_damage(reader: _Reader) -> Dict[str, object]:
    armor = reader.byte()
    blood = reader.byte()
    origin = [reader.coord() for _ in range(3)]
    return {
        "type": "damage",
        "armor": armor,
        "blood": blood,
        "origin": origin,
    }


def _read_temp_entity(reader: _Reader) -> Dict[str, object]:
    temp_type = reader.byte()
    if temp_type in TE_SIMPLE_COORD:
        return {
            "type": "temp_entity",
            "temp_type": temp_type,
            "origin": [reader.coord() for _ in range(3)],
        }
    if temp_type in TE_BEAM:
        entity_num = reader.short()
        coords = [reader.coord() for _ in range(6)]
        return {
            "type": "temp_entity",
            "temp_type": temp_type,
            "entity_num": entity_num,
            "start": coords[:3],
            "end": coords[3:],
        }
    if temp_type == 12:
        return {
            "type": "temp_entity",
            "temp_type": temp_type,
            "origin": [reader.coord() for _ in range(3)],
            "color_start": reader.byte(),
            "color_length": reader.byte(),
        }
    raise ValueError(f"Unsupported temp entity type {temp_type}")


def _parse_serverinfo(reader: _Reader) -> Dict[str, object]:
    info: Dict[str, object] = {}

    try:
        info["protocol"] = reader.long()
    except ValueError:
        return info

    try:
        maxclients = reader.byte()
        info["maxclients"] = maxclients
    except ValueError:
        return info

    try:
        info["gametype"] = reader.byte()
        level_name = reader.string()
    except ValueError:
        return info

    if level_name:
        # Keep the historical key for downstream metadata until the schema is renamed.
        info["level_name"] = level_name
        info["game_dir"] = level_name

    models: List[str] = []
    while True:
        try:
            model = reader.string()
        except ValueError:
            break
        if not model:
            break
        models.append(model)
    if models:
        info["map_or_server"] = models[0]

    sounds: List[str] = []
    while True:
        try:
            sound = reader.string()
        except ValueError:
            break
        if not sound:
            break
        sounds.append(sound)

    info["model_count"] = len(models)
    info["sound_count"] = len(sounds)
    if models:
        info["first_models"] = models[:8]
    if sounds:
        info["first_sounds"] = sounds[:8]
    return info


def _normalize_quake_text(text: str) -> str:
    chars: List[str] = []
    for char in text:
        code = ord(char) & 0x7F
        if code == 0:
            continue
        if code < 32:
            if code in {10, 13}:
                chars.append(" ")
            continue
        chars.append(chr(code))
    return " ".join("".join(chars).split()).lower()


def _consume_pending_cvar(cvars: Dict[str, object], pending_cvar: str | None, text: str) -> str | None:
    if pending_cvar is None:
        return None
    if re.fullmatch(r"-?\d+", text):
        cvars[pending_cvar] = int(text)
        return None
    if not text or any(label in text for label in _TEXT_CVAR_PATTERNS):
        return pending_cvar
    return None


def _update_cvars_from_text(cvars: Dict[str, object], text: str, pending_cvar: str | None) -> str | None:
    pending_cvar = _consume_pending_cvar(cvars, pending_cvar, text)

    for key, pattern in _TEXT_CVAR_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            continue
        try:
            cvars[key] = int(match.group(1))
            pending_cvar = None
        except ValueError:
            continue

    for match in _QUOTED_CVAR_CHANGE_PATTERN.finditer(text):
        key, value = match.groups()
        try:
            cvars[key.lower()] = int(value)
            pending_cvar = None
        except ValueError:
            continue
    if text in _TEXT_CVAR_PATTERNS:
        return text
    if text.endswith(":") and text[:-1] in _TEXT_CVAR_PATTERNS:
        return text[:-1]
    return pending_cvar


def _update_text_flags(text_flags: Dict[str, bool], text: str) -> None:
    if not text:
        return
    if "teamplay" in text:
        text_flags["saw_teamplay_text"] = True
    if any(pattern.search(text) for pattern in _TEAM_MENU_PATTERNS):
        text_flags["saw_team_menu_text"] = True
    if _TEAM_ASSIGN_PATTERN.search(text):
        text_flags["saw_team_assignment_text"] = True
    if _TEAM_SCORE_PATTERN.search(text):
        text_flags["saw_team_score_text"] = True
    if _TEAM_WIN_PATTERN.search(text):
        text_flags["saw_team_win_text"] = True
    if "final scores:" in text and any(f"{team} team" in text for team in _TEAM_NAMES):
        text_flags["saw_team_final_scores_text"] = True
    if "observer" in text:
        text_flags["saw_observer_text"] = True
    if _DUEL_PATTERN.search(text):
        text_flags["saw_duel_text"] = True


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


def _parse_update(reader: _Reader, entities: Dict[int, _EntityState], bits: int, updated_entities: set[int]) -> None:
    if bits & U_MOREBITS:
        bits |= reader.byte() << 8

    entity_num = reader.short() if (bits & U_LONGENTITY) else reader.byte()
    entity = entities.setdefault(entity_num, _EntityState())
    updated_entities.add(entity_num)

    if bits & U_MODEL:
        entity.modelindex = reader.byte()
    elif entity.modelindex <= 0:
        entity.modelindex = entity.baseline.modelindex
    if bits & U_FRAME:
        entity.frame = reader.byte()
    elif entity.frame <= 0:
        entity.frame = entity.baseline.frame
    if bits & U_COLORMAP:
        entity.colormap = reader.byte()
    elif entity.colormap <= 0:
        entity.colormap = entity.baseline.colormap
    if bits & U_SKIN:
        entity.skin = reader.byte()
    elif entity.skin <= 0:
        entity.skin = entity.baseline.skin
    if bits & U_EFFECTS:
        entity.effects = reader.byte()

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


def _snapshot_visible_entities(entities: Dict[int, _EntityState], updated_entities: set[int], viewentity: int) -> List[Dict[str, object]]:
    visible_entities: List[Dict[str, object]] = []
    for entity_num in sorted(updated_entities):
        if entity_num == viewentity:
            continue
        entity = entities.get(entity_num)
        if entity is None:
            continue
        visible_entities.append(
            {
                "entity_num": entity_num,
                "origin": entity.origin[:],
                "angles": entity.angles[:],
                "model_id": int(entity.modelindex),
                "frame": int(entity.frame),
                "effects": int(entity.effects),
                "colormap": int(entity.colormap),
                "skin": int(entity.skin),
            }
        )
    return visible_entities


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
        events = [dict(event) for event in frame.events]
        if health_gain:
            events.append({"type": "pickup_health", "delta": int(frame.health - previous.health)})
        if armor_gain:
            events.append({"type": "pickup_armor", "delta": int(frame.armor - previous.armor)})
        if ammo_gain:
            events.append({"type": "pickup_ammo", "delta": int(frame.ammo - previous.ammo)})
        if weapon_gain:
            events.append({"type": "pickup_weapon", "weapon_id": int(frame.weapon_id)})
        action = infer_action_label(
            current_pos=frame.player_pos,
            next_pos=next_frame.player_pos,
            current_yaw=frame.yaw,
            next_yaw=next_frame.yaw,
            current_pitch=frame.view_angles[0],
            next_pitch=next_frame.view_angles[0],
            current_weapon_id=frame.weapon_id,
            next_weapon_id=next_frame.weapon_id,
            current_ammo=frame.ammo,
            next_ammo=next_frame.ammo,
            current_velocity=frame.player_vel,
            next_velocity=next_frame.player_vel,
        )
        ticks.append(
            ParsedDemoTick(
                time_s=frame.time_s,
                player_pos=frame.player_pos[:],
                player_vel=frame.player_vel[:],
                view_angles=frame.view_angles[:],
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
                visible_entities=[dict(entity) for entity in frame.visible_entities],
                events=events,
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
    serverinfo: Dict[str, object] = {}
    cvars: Dict[str, object] = {}
    player_names: Dict[int, str] = {}
    player_colors: Dict[int, int] = {}
    frag_updates: List[Dict[str, object]] = []
    text_flags: Dict[str, bool] = {}
    maxclients = 0
    pending_cvar: str | None = None

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
        updated_entities: set[int] = set()
        message_events: List[Dict[str, object]] = []

        while reader.offset < len(message):
            try:
                cmd = reader.byte()
                if cmd == 255:
                    break
                if cmd & 128:
                    _parse_update(reader, entities, cmd & 127, updated_entities)
                    continue

                if cmd == SVC_NOP:
                    continue
                if cmd == SVC_DISCONNECT:
                    message_events.append({"type": "disconnect"})
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
                    message_events.append(_read_sound(reader))
                    continue
                if cmd == SVC_TIME:
                    current_time = reader.float()
                    continue
                if cmd in {SVC_PRINT, SVC_STUFFTEXT, SVC_CENTERPRINT, SVC_FINALE, SVC_CUTSCENE}:
                    text = reader.string()
                    normalized_text = _normalize_quake_text(text)
                    pending_cvar = _update_cvars_from_text(cvars, normalized_text, pending_cvar)
                    _update_text_flags(text_flags, normalized_text)
                    if cmd in {SVC_FINALE, SVC_CUTSCENE}:
                        intermission = True
                        message_events.append({"type": "intermission_text", "text": text})
                    continue
                if cmd == SVC_SETANGLE:
                    reader.angle()
                    reader.angle()
                    reader.angle()
                    continue
                if cmd == SVC_SERVERINFO:
                    serverinfo = _parse_serverinfo(reader)
                    if "maxclients" in serverinfo:
                        try:
                            maxclients = max(maxclients, int(serverinfo["maxclients"]))
                        except (TypeError, ValueError):
                            pass
                    continue
                if cmd == SVC_LIGHTSTYLE:
                    reader.byte()
                    reader.string()
                    continue
                if cmd == SVC_UPDATENAME:
                    slot = reader.byte()
                    name = reader.string()
                    if name:
                        player_names[int(slot)] = name
                        maxclients = max(maxclients, int(slot) + 1)
                    continue
                if cmd == SVC_UPDATEFRAGS:
                    slot = reader.byte()
                    frags = reader.short()
                    frag_updates.append(
                        {
                            "player_slot": int(slot),
                            "frags": int(frags),
                            "message_index": int(message_index),
                            "time_s": float(current_time),
                        }
                    )
                    maxclients = max(maxclients, int(slot) + 1)
                    continue
                if cmd == SVC_CLIENTDATA:
                    latest_velocity = _parse_clientdata(reader, stats)
                    continue
                if cmd == SVC_STOPSOUND:
                    reader.short()
                    continue
                if cmd == SVC_UPDATECOLORS:
                    slot = reader.byte()
                    colors = reader.byte()
                    player_colors[int(slot)] = int(colors)
                    maxclients = max(maxclients, int(slot) + 1)
                    continue
                if cmd == SVC_PARTICLE:
                    message_events.append(_read_particle(reader))
                    continue
                if cmd == SVC_DAMAGE:
                    message_events.append(_read_damage(reader))
                    continue
                if cmd == SVC_SPAWNSTATIC:
                    _parse_baseline(reader, _EntityState())
                    continue
                if cmd == SVC_SPAWNBASELINE:
                    _parse_baseline(reader, entities.setdefault(reader.short(), _EntityState()))
                    continue
                if cmd == SVC_TEMP_ENTITY:
                    message_events.append(_read_temp_entity(reader))
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
                        message_events.append({"type": "intermission"})
                    elif cmd == SVC_KILLEDMONSTER:
                        message_events.append({"type": "killed_monster"})
                    elif cmd == SVC_FOUNDSECRET:
                        message_events.append({"type": "found_secret"})
                    else:
                        message_events.append({"type": "sell_screen"})
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
                view_angles=[float(header_angles[0]), float(header_angles[1]), float(header_angles[2])],
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
                visible_entities=_snapshot_visible_entities(entities, updated_entities, viewentity),
                events=message_events,
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
            # tolerate early unknown messages so metadata can be recovered
            terminated_early = True
            break
        if saw_disconnect:
            break

    parsed_ticks = _build_ticks(frames, terminated_early=terminated_early)
    maxclients = max(maxclients, len(player_names))
    return ParsedDemoEpisode(
        episode_id=episode_id or source.stem,
        map_id=map_id,
        ticks=parsed_ticks,
        serverinfo=serverinfo,
        cvars=cvars,
        player_names=player_names,
        player_colors=player_colors,
        frag_updates=frag_updates,
        text_flags=text_flags,
        maxclients=maxclients,
        duration_s=float(current_time),
        tick_count=len(parsed_ticks),
    )


def parse_netquake_demo_metadata(path: str | Path, map_id: str, episode_id: str | None = None) -> ParsedDemoEpisode:
    """Recover demo metadata without building replay ticks or action labels."""
    source = Path(path)
    payload = source.read_bytes()
    if b"\n" not in payload:
        raise ValueError(f"Invalid demo header in {source}")

    header_end = payload.index(b"\n") + 1
    body = payload[header_end:]
    offset = 0

    entities: Dict[int, _EntityState] = {}
    stats = [0] * 32
    message_index = 0
    current_time = 0.0
    serverinfo: Dict[str, object] = {}
    cvars: Dict[str, object] = {}
    player_names: Dict[int, str] = {}
    player_colors: Dict[int, int] = {}
    frag_updates: List[Dict[str, object]] = []
    text_flags: Dict[str, bool] = {}
    maxclients = 0
    pending_cvar: str | None = None

    while offset + 16 <= len(body):
        message_size = struct.unpack("<i", body[offset : offset + 4])[0]
        offset += 4
        offset += 12  # header view angles
        if message_size < 0 or offset + message_size > len(body):
            if message_index > 0 or serverinfo or player_names or frag_updates:
                break
            raise ValueError(f"Invalid demo message length in {source}")

        message = body[offset : offset + message_size]
        offset += message_size
        reader = _Reader(message)
        parse_error: ValueError | None = None
        saw_disconnect = False

        while reader.offset < len(message):
            try:
                cmd = reader.byte()
                if cmd == 255:
                    break
                if cmd & 128:
                    _parse_update(reader, entities, cmd & 127, set())
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
                    reader.short()
                    continue
                if cmd == SVC_SOUND:
                    _read_sound(reader)
                    continue
                if cmd == SVC_TIME:
                    current_time = reader.float()
                    continue
                if cmd in {SVC_PRINT, SVC_STUFFTEXT, SVC_CENTERPRINT, SVC_FINALE, SVC_CUTSCENE}:
                    text = reader.string()
                    normalized_text = _normalize_quake_text(text)
                    pending_cvar = _update_cvars_from_text(cvars, normalized_text, pending_cvar)
                    _update_text_flags(text_flags, normalized_text)
                    continue
                if cmd == SVC_SETANGLE:
                    reader.angle()
                    reader.angle()
                    reader.angle()
                    continue
                if cmd == SVC_SERVERINFO:
                    serverinfo = _parse_serverinfo(reader)
                    if "maxclients" in serverinfo:
                        try:
                            maxclients = max(maxclients, int(serverinfo["maxclients"]))
                        except (TypeError, ValueError):
                            pass
                    continue
                if cmd == SVC_LIGHTSTYLE:
                    reader.byte()
                    reader.string()
                    continue
                if cmd == SVC_UPDATENAME:
                    slot = reader.byte()
                    name = reader.string()
                    if name:
                        player_names[int(slot)] = name
                        maxclients = max(maxclients, int(slot) + 1)
                    continue
                if cmd == SVC_UPDATEFRAGS:
                    slot = reader.byte()
                    frags = reader.short()
                    frag_updates.append(
                        {
                            "player_slot": int(slot),
                            "frags": int(frags),
                            "message_index": int(message_index),
                            "time_s": float(current_time),
                        }
                    )
                    maxclients = max(maxclients, int(slot) + 1)
                    continue
                if cmd == SVC_CLIENTDATA:
                    _parse_clientdata(reader, stats)
                    continue
                if cmd == SVC_STOPSOUND:
                    reader.short()
                    continue
                if cmd == SVC_UPDATECOLORS:
                    slot = reader.byte()
                    colors = reader.byte()
                    player_colors[int(slot)] = int(colors)
                    maxclients = max(maxclients, int(slot) + 1)
                    continue
                if cmd == SVC_PARTICLE:
                    _read_particle(reader)
                    continue
                if cmd == SVC_DAMAGE:
                    _read_damage(reader)
                    continue
                if cmd == SVC_SPAWNSTATIC:
                    _parse_baseline(reader, _EntityState())
                    continue
                if cmd == SVC_SPAWNBASELINE:
                    _parse_baseline(reader, entities.setdefault(reader.short(), _EntityState()))
                    continue
                if cmd == SVC_TEMP_ENTITY:
                    _read_temp_entity(reader)
                    continue
                if cmd == SVC_SETPAUSE:
                    reader.byte()
                    continue
                if cmd == SVC_SIGNONNUM:
                    reader.byte()
                    continue
                if cmd in {SVC_KILLEDMONSTER, SVC_FOUNDSECRET, SVC_INTERMISSION, SVC_SELLSCREEN}:
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

        message_index += 1
        if parse_error is not None:
            if message_index > 1 or serverinfo or player_names or frag_updates:
                break
            raise parse_error
        if saw_disconnect:
            break

    maxclients = max(maxclients, len(player_names))
    return ParsedDemoEpisode(
        episode_id=episode_id or source.stem,
        map_id=map_id,
        ticks=[],
        serverinfo=serverinfo,
        cvars=cvars,
        player_names=player_names,
        player_colors=player_colors,
        frag_updates=frag_updates,
        text_flags=text_flags,
        maxclients=maxclients,
        duration_s=float(current_time),
        tick_count=int(message_index),
    )
