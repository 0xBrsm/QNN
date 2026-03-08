"""Demo metadata extraction helpers for competitive classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

from engine.adapter import DemoEpisode
from quake_ai.data.corpus import canonical_map_id


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


def _metadata_value(meta: Mapping[str, object], key: str) -> object | None:
    value = meta.get(key)
    return value if value != "" else None


def _final_frags(frag_updates: list[Dict[str, object]]) -> Dict[int, int]:
    latest: Dict[int, int] = {}
    for row in frag_updates:
        slot = row.get("player_slot")
        frags = row.get("frags")
        if not isinstance(slot, int) or not isinstance(frags, int):
            continue
        latest[int(slot)] = int(frags)
    return latest


def build_demo_metadata(
    episode: DemoEpisode,
    *,
    source_path: str,
    source_url: str | None = None,
) -> DemoMetadata:
    meta = dict(getattr(episode, "metadata", {}) or {})
    serverinfo = dict(meta.get("serverinfo", {}))
    cvars = dict(meta.get("cvars", {}))
    map_source = _metadata_value(serverinfo, "map_or_server")
    if map_source is None:
        first_models = serverinfo.get("first_models")
        if isinstance(first_models, list) and first_models:
            map_source = str(first_models[0])
    map_hint = canonical_map_id(str(map_source or episode.map_id)) or episode.map_id
    maxclients = meta.get("maxclients")
    try:
        maxclients = int(maxclients) if maxclients is not None else None
    except (TypeError, ValueError):
        maxclients = None

    duration_s = meta.get("duration_s")
    try:
        duration_s = float(duration_s) if duration_s is not None else 0.0
    except (TypeError, ValueError):
        duration_s = 0.0

    tick_count = meta.get("tick_count")
    try:
        tick_count = int(tick_count) if tick_count is not None else len(episode.ticks)
    except (TypeError, ValueError):
        tick_count = len(episode.ticks)

    protocol_value = meta.get("protocol")
    if not isinstance(protocol_value, int) and isinstance(serverinfo.get("protocol"), int):
        protocol_value = int(serverinfo["protocol"])

    level_name = _metadata_value(serverinfo, "level_name") or _metadata_value(serverinfo, "game_dir")
    frag_updates = [dict(row) for row in meta.get("frag_updates", []) if isinstance(row, Mapping)]
    player_colors = {
        int(k): int(v)
        for k, v in dict(meta.get("player_colors", {})).items()
        if isinstance(v, int)
    }
    text_flags = {
        str(k): bool(v)
        for k, v in dict(meta.get("text_flags", {})).items()
        if isinstance(k, str)
    }

    return DemoMetadata(
        episode_id=episode.episode_id,
        map_id=map_hint,
        source_path=str(Path(source_path)),
        source_url=source_url,
        protocol=protocol_value if isinstance(protocol_value, int) else None,
        game_dir=str(level_name) if isinstance(level_name, str) else None,
        maxclients=maxclients,
        player_names={int(k): str(v) for k, v in dict(meta.get("player_names", {})).items()},
        player_colors=player_colors,
        frag_updates=frag_updates,
        final_frags=_final_frags(frag_updates),
        deathmatch=int(cvars["deathmatch"]) if isinstance(cvars.get("deathmatch"), (int, float)) else None,
        teamplay=int(cvars["teamplay"]) if isinstance(cvars.get("teamplay"), (int, float)) else None,
        fraglimit=int(cvars["fraglimit"]) if isinstance(cvars.get("fraglimit"), (int, float)) else None,
        timelimit=int(cvars["timelimit"]) if isinstance(cvars.get("timelimit"), (int, float)) else None,
        text_flags=text_flags,
        tick_count=tick_count,
        duration_s=duration_s,
    )
