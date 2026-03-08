"""Heuristics for classifying demos into competitive buckets."""

from __future__ import annotations

import re
from typing import Iterable

from quake_ai.data.demo_metadata import DemoMetadata

_SINGLEPLAYER_HINT = re.compile(r"(^|[^a-z0-9])(e[1-4]m[1-8]|end)([^a-z0-9]|$)", re.IGNORECASE)


def _player_count(meta: DemoMetadata) -> int:
    named_slots = len([name for name in meta.player_names.values() if name])
    max_slot = max(meta.player_names.keys(), default=-1) + 1
    inferred = max(max_slot, named_slots)
    return int(max(0, inferred))


def _frag_rate(meta: DemoMetadata) -> float:
    if meta.duration_s <= 1e-3:
        return float(len(meta.frag_updates))
    return float(len(meta.frag_updates)) / max(meta.duration_s, 1e-6)


def _active_frag_slots(meta: DemoMetadata) -> set[int]:
    active: set[int] = set()
    for row in meta.frag_updates:
        slot = row.get("player_slot")
        frags = row.get("frags")
        if not isinstance(slot, int) or not isinstance(frags, int):
            continue
        if frags == 0 or frags <= -90:
            continue
        active.add(int(slot))
    return active


def classify_competitive(meta: DemoMetadata, *, source_prior: str | None = None) -> DemoMetadata:
    """Assign competitive class + mode label from parsed demo metadata only."""
    del source_prior
    if _SINGLEPLAYER_HINT.search(meta.map_id.lower()):
        meta.classification = "non_competitive"
        meta.classification_confidence = 0.9
        meta.mode = "non_competitive"
        return meta
    player_count = _player_count(meta)
    frag_rate = _frag_rate(meta)
    frag_count = len(meta.frag_updates)
    duration_ok = meta.duration_s >= 20.0 or meta.tick_count >= 200
    maxclients = meta.maxclients or 0
    dm_cvar = meta.deathmatch
    team_cvar = meta.teamplay
    text_flags = meta.text_flags
    frag_activity = frag_rate >= 0.01 or frag_count >= 8
    active_fraggers = _active_frag_slots(meta)
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


def apply_classification(metas: Iterable[DemoMetadata], source_prior: str | None = None) -> list[DemoMetadata]:
    return [classify_competitive(meta, source_prior=source_prior) for meta in metas]
