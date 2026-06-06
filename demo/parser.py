"""Lightweight Quake demo parsing for classification and validation.

This module handles:
- Binary .dem file parsing (metadata extraction only, no tick building)
- Competitive demo classification heuristics
- Demo metadata normalization

Training observation data comes from the C demo/native worker, not this parser.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .protocol import (
    Reader,
    SU_ARMOR,
    SU_IDEALPITCH,
    SU_ITEMS,
    SU_PUNCH1,
    SU_VELOCITY1,
    SU_VIEWHEIGHT,
    SU_WEAPON,
    SU_WEAPONFRAME,
    SVC_CDTRACK,
    SVC_CENTERPRINT,
    SVC_CLIENTDATA,
    SVC_CUTSCENE,
    SVC_DAMAGE,
    SVC_DISCONNECT,
    SVC_FINALE,
    SVC_FOUNDSECRET,
    SVC_INTERMISSION,
    SVC_KILLEDMONSTER,
    SVC_LIGHTSTYLE,
    SVC_NOP,
    SVC_PARTICLE,
    SVC_PRINT,
    SVC_SELLSCREEN,
    SVC_SERVERINFO,
    SVC_SETANGLE,
    SVC_SETPAUSE,
    SVC_SETVIEW,
    SVC_SIGNONNUM,
    SVC_SOUND,
    SVC_SPAWNBASELINE,
    SVC_SPAWNSTATIC,
    SVC_SPAWNSTATICSOUND,
    SVC_STOPSOUND,
    SVC_STUFFTEXT,
    SVC_TEMP_ENTITY,
    SVC_TIME,
    SVC_UPDATECOLORS,
    SVC_UPDATEFRAGS,
    SVC_UPDATENAME,
    SVC_UPDATESTAT,
    SVC_VERSION,
    U_ANGLE1,
    U_ANGLE2,
    U_ANGLE3,
    U_COLORMAP,
    U_EFFECTS,
    U_FRAME,
    U_LONGENTITY,
    U_MODEL,
    U_MOREBITS,
    U_ORIGIN1,
    U_ORIGIN2,
    U_ORIGIN3,
    U_SKIN,
    skip_clientdata,
    skip_damage,
    skip_particle,
    skip_sound,
    skip_temp_entity,
)

# ---------------------------------------------------------------------------
# Text / cvar extraction patterns
# ---------------------------------------------------------------------------

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

_CANONICAL_MAP_RE = re.compile(r"(?:^|[^a-z0-9])(e[1-4]m[1-8]|dm[1-6]|end)(?:[^a-z0-9]|$)")


def canonical_map_id(value: str) -> str | None:
    if not value:
        return None
    leaf = value.replace("\\", "/").split("/")[-1]
    stem = leaf.rsplit(".", 1)[0].lower() if "." in leaf else leaf.lower()
    prefix = stem.split("_", 1)[0]
    match = re.match(r"^(e[1-4]m[1-8]|dm[1-6]|end)", prefix)
    if match:
        return match.group(1)
    match = _CANONICAL_MAP_RE.search(stem)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Entity state (needed to advance reader past entity updates)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Protocol parsing helpers
# ---------------------------------------------------------------------------


def _parse_baseline(reader: Reader, entity: _EntityState) -> None:
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


def _parse_serverinfo(reader: Reader) -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        info["protocol"] = reader.long()
    except ValueError:
        return info
    try:
        info["maxclients"] = reader.byte()
    except ValueError:
        return info
    try:
        info["gametype"] = reader.byte()
        level_name = reader.string()
    except ValueError:
        return info
    if level_name:
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


def _parse_update(reader: Reader, entities: Dict[int, _EntityState], bits: int) -> None:
    if bits & U_MOREBITS:
        bits |= reader.byte() << 8
    entity_num = reader.short() if (bits & U_LONGENTITY) else reader.byte()
    entity = entities.setdefault(entity_num, _EntityState())
    if bits & U_MODEL:
        entity.modelindex = reader.byte()
    if bits & U_FRAME:
        entity.frame = reader.byte()
    if bits & U_COLORMAP:
        entity.colormap = reader.byte()
    if bits & U_SKIN:
        entity.skin = reader.byte()
    if bits & U_EFFECTS:
        entity.effects = reader.byte()
    if bits & U_ORIGIN1:
        entity.origin[0] = reader.coord()
    if bits & U_ANGLE1:
        entity.angles[0] = reader.angle()
    if bits & U_ORIGIN2:
        entity.origin[1] = reader.coord()
    if bits & U_ANGLE2:
        entity.angles[1] = reader.angle()
    if bits & U_ORIGIN3:
        entity.origin[2] = reader.coord()
    if bits & U_ANGLE3:
        entity.angles[2] = reader.angle()


# ---------------------------------------------------------------------------
# Text / cvar helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DemoMetadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DemoMetadata:
    episode_id: str
    map_id: str
    source_path: str
    source_url: str | None = None
    protocol: int | None = None
    game_dir: str | None = None
    maxclients: int | None = None
    player_names: Dict[int, str] = field(default_factory=dict)
    player_colors: Dict[int, int] = field(default_factory=dict)
    frag_updates: list[Dict[str, object]] = field(default_factory=list)
    final_frags: Dict[int, int] = field(default_factory=dict)
    deathmatch: int | None = None
    teamplay: int | None = None
    fraglimit: int | None = None
    timelimit: int | None = None
    text_flags: Dict[str, bool] = field(default_factory=dict)
    tick_count: int = 0
    duration_s: float = 0.0
    classification: str = "unknown"
    classification_confidence: float = 0.0
    mode: str = "unknown"  # ffa | teamplay | non_competitive | unknown

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "map_id": self.map_id,
            "source_path": self.source_path,
            "source_url": self.source_url or "",
            "protocol": self.protocol,
            "game_dir": self.game_dir,
            "maxclients": self.maxclients,
            "player_names": dict(self.player_names),
            "player_colors": dict(self.player_colors),
            "frag_updates": list(self.frag_updates),
            "final_frags": dict(self.final_frags),
            "deathmatch": self.deathmatch,
            "teamplay": self.teamplay,
            "fraglimit": self.fraglimit,
            "timelimit": self.timelimit,
            "text_flags": dict(self.text_flags),
            "tick_count": self.tick_count,
            "duration_s": self.duration_s,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "mode": self.mode,
        }


@dataclass(slots=True)
class DemoProbe:
    episode_id: str
    map_id: str
    source_path: str


# ---------------------------------------------------------------------------
# Binary demo parsing (metadata only)
# ---------------------------------------------------------------------------


def _final_frags(frag_updates: list[Dict[str, object]]) -> Dict[int, int]:
    latest: Dict[int, int] = {}
    for row in frag_updates:
        idx = row.get("player_idx")
        frags = row.get("frags")
        if not isinstance(idx, int) or not isinstance(frags, int):
            continue
        latest[int(idx)] = int(frags)
    return latest


def parse_demo_metadata(
    path: str | Path,
    map_id: str,
    *,
    source_path: str = "",
    source_url: str | None = None,
    episode_id: str | None = None,
) -> DemoMetadata:
    """Parse a .dem file and return classification-ready metadata.

    This is a lightweight parser that extracts serverinfo, cvars, player
    names, frag updates, and text flags without building replay ticks.
    """
    source = Path(path)
    payload = source.read_bytes()
    if b"\n" not in payload:
        raise ValueError(f"Invalid demo header in {source}")

    header_end = payload.index(b"\n") + 1
    body = payload[header_end:]
    offset = 0

    entities: Dict[int, _EntityState] = {}
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
        reader = Reader(message)
        parse_error: ValueError | None = None
        saw_disconnect = False

        while reader.pos < len(message):
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
                    reader.byte()
                    reader.long()
                    continue
                if cmd == SVC_VERSION:
                    reader.long()
                    continue
                if cmd == SVC_SETVIEW:
                    reader.short()
                    continue
                if cmd == SVC_SOUND:
                    skip_sound(reader)
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
                    idx = reader.byte()
                    name = reader.string()
                    if name:
                        player_names[int(idx)] = name
                        maxclients = max(maxclients, int(idx) + 1)
                    continue
                if cmd == SVC_UPDATEFRAGS:
                    idx = reader.byte()
                    frags = reader.short()
                    frag_updates.append(
                        {
                            "player_idx": int(idx),
                            "frags": int(frags),
                            "message_index": int(message_index),
                            "time_s": float(current_time),
                        }
                    )
                    maxclients = max(maxclients, int(idx) + 1)
                    continue
                if cmd == SVC_CLIENTDATA:
                    skip_clientdata(reader)
                    continue
                if cmd == SVC_STOPSOUND:
                    reader.short()
                    continue
                if cmd == SVC_UPDATECOLORS:
                    idx = reader.byte()
                    colors = reader.byte()
                    player_colors[int(idx)] = int(colors)
                    maxclients = max(maxclients, int(idx) + 1)
                    continue
                if cmd == SVC_PARTICLE:
                    skip_particle(reader)
                    continue
                if cmd == SVC_DAMAGE:
                    skip_damage(reader)
                    continue
                if cmd == SVC_SPAWNSTATIC:
                    _parse_baseline(reader, _EntityState())
                    continue
                if cmd == SVC_SPAWNBASELINE:
                    _parse_baseline(reader, entities.setdefault(reader.short(), _EntityState()))
                    continue
                if cmd == SVC_TEMP_ENTITY:
                    skip_temp_entity(reader)
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

    # Normalize metadata into DemoMetadata
    map_source = serverinfo.get("map_or_server")
    if map_source is None:
        first_models = serverinfo.get("first_models")
        if isinstance(first_models, list) and first_models:
            map_source = str(first_models[0])
    map_hint = canonical_map_id(str(map_source or map_id)) or map_id

    protocol_value = serverinfo.get("protocol")
    level_name = serverinfo.get("level_name") or serverinfo.get("game_dir")

    return DemoMetadata(
        episode_id=episode_id or source.stem,
        map_id=map_hint,
        source_path=source_path or str(source),
        source_url=source_url,
        protocol=protocol_value if isinstance(protocol_value, int) else None,
        game_dir=str(level_name) if isinstance(level_name, str) else None,
        maxclients=maxclients if maxclients > 0 else None,
        player_names={int(k): str(v) for k, v in player_names.items()},
        player_colors={int(k): int(v) for k, v in player_colors.items()},
        frag_updates=[dict(row) for row in frag_updates],
        final_frags=_final_frags(frag_updates),
        deathmatch=int(cvars["deathmatch"]) if isinstance(cvars.get("deathmatch"), (int, float)) else None,
        teamplay=int(cvars["teamplay"]) if isinstance(cvars.get("teamplay"), (int, float)) else None,
        fraglimit=int(cvars["fraglimit"]) if isinstance(cvars.get("fraglimit"), (int, float)) else None,
        timelimit=int(cvars["timelimit"]) if isinstance(cvars.get("timelimit"), (int, float)) else None,
        text_flags={str(k): bool(v) for k, v in text_flags.items()},
        tick_count=int(message_index),
        duration_s=float(current_time),
    )


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------


def _player_count(meta: DemoMetadata) -> int:
    named_indices = len([name for name in meta.player_names.values() if name])
    max_idx = max(meta.player_names.keys(), default=-1) + 1
    inferred = max(max_idx, named_indices)
    return int(max(0, inferred))


def _frag_rate(meta: DemoMetadata) -> float:
    if meta.duration_s <= 1e-3:
        return float(len(meta.frag_updates))
    return float(len(meta.frag_updates)) / max(meta.duration_s, 1e-6)


def _active_frag_indices(meta: DemoMetadata) -> set[int]:
    active: set[int] = set()
    for row in meta.frag_updates:
        idx = row.get("player_idx")
        frags = row.get("frags")
        if not isinstance(idx, int) or not isinstance(frags, int):
            continue
        if frags == 0 or frags <= -90:
            continue
        active.add(int(idx))
    return active


def classify_competitive(meta: DemoMetadata, *, source_prior: str | None = None) -> DemoMetadata:
    """Assign competitive class + mode label from parsed demo metadata only."""
    del source_prior
    player_count = _player_count(meta)
    frag_rate = _frag_rate(meta)
    frag_count = len(meta.frag_updates)
    duration_ok = meta.duration_s >= 20.0 or meta.tick_count >= 200
    maxclients = meta.maxclients or 0
    dm_cvar = meta.deathmatch
    team_cvar = meta.teamplay
    text_flags = meta.text_flags
    frag_activity = frag_rate >= 0.01 or frag_count >= 8
    active_fraggers = _active_frag_indices(meta)
    explicit_team_text = any(
        bool(text_flags.get(flag))
        for flag in (
            "saw_team_menu_text",
            "saw_team_assignment_text",
            "saw_team_score_text",
            "saw_team_win_text",
            "saw_team_final_scores_text",
        )
    )
    duel_text = bool(text_flags.get("saw_duel_text"))
    two_player_ffa = len(active_fraggers) == 2 and frag_activity

    classification = "unknown"
    confidence = 0.25
    mode = "unknown"

    if team_cvar is not None:
        if int(team_cvar) >= 1:
            classification = "competitive_teamplay"
            mode = "teamplay"
            confidence = 0.85
        else:
            classification = "competitive_ffa"
            mode = "ffa"
            confidence = 0.8
    elif explicit_team_text:
        classification = "competitive_teamplay"
        mode = "teamplay"
        confidence = 0.8
    elif duel_text:
        classification = "competitive_ffa"
        mode = "ffa"
        confidence = 0.8
    elif dm_cvar is not None and int(dm_cvar) >= 1 and two_player_ffa:
        classification = "competitive_ffa"
        mode = "ffa"
        confidence = 0.75
    elif two_player_ffa and duration_ok and player_count <= max(4, len(active_fraggers) + 2):
        classification = "competitive_ffa"
        mode = "ffa"
        confidence = 0.7
    elif frag_count > 0 or player_count > 0 or maxclients > 0:
        classification = "unknown"
        confidence = 0.35 if duration_ok else 0.25
    else:
        classification = "non_competitive"
        mode = "non_competitive"
        confidence = 0.2

    meta.classification = classification
    meta.classification_confidence = float(max(0.0, min(confidence, 0.99)))
    meta.mode = mode
    return meta


def classify_demo(
    path: str | Path,
    map_id: str,
    *,
    source_path: str = "",
    source_url: str | None = None,
) -> DemoMetadata:
    """Parse and classify a demo file in one step."""
    meta = parse_demo_metadata(path, map_id, source_path=source_path, source_url=source_url)
    return classify_competitive(meta)


def probe_demo(path: str | Path, fallback_map_id: str = "dm4") -> DemoProbe:
    source = Path(path)
    default_map_id = canonical_map_id(source.name) or str(fallback_map_id)
    try:
        meta = parse_demo_metadata(source, default_map_id, source_path=str(source))
        map_id = canonical_map_id(meta.map_id) or default_map_id
        episode_id = str(meta.episode_id or source.stem)
    except Exception:
        map_id = default_map_id
        episode_id = source.stem
    return DemoProbe(episode_id=episode_id, map_id=map_id, source_path=str(source))
