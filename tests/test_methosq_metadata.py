from __future__ import annotations

from pathlib import Path

import pytest

from engine.adapter import DemoEpisode, load_demo
from quake_ai.data.demo_classifier import classify_competitive
from quake_ai.data.demo_metadata import build_demo_metadata
from quake_ai.data.netquake_demo import parse_netquake_demo_metadata


def _methosq_demo_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "corpus" / "netquake" / "materialized_methosq" / name


def _metadata_only_meta(demo_path: Path):
    parsed = parse_netquake_demo_metadata(demo_path, map_id="unknown")
    episode = DemoEpisode(
        episode_id=parsed.episode_id,
        map_id=parsed.map_id,
        ticks=[],
        metadata={
            "serverinfo": dict(parsed.serverinfo),
            "cvars": dict(parsed.cvars),
            "player_names": dict(parsed.player_names),
            "player_colors": dict(parsed.player_colors),
            "frag_updates": list(parsed.frag_updates),
            "text_flags": dict(parsed.text_flags),
            "maxclients": int(parsed.maxclients),
            "duration_s": float(parsed.duration_s),
            "tick_count": int(parsed.tick_count),
        },
    )
    return classify_competitive(build_demo_metadata(episode, source_path=str(demo_path)))


def test_thresh3_demo_exposes_explicit_teamplay_metadata() -> None:
    demo_path = _methosq_demo_path("thresh3__THRESH3.DEM")
    if not demo_path.exists():
        pytest.skip("THRESH3.DEM not available under artifacts/corpus/netquake/materialized_methosq")

    episode = load_demo(demo_path, map_id="unknown")
    meta = classify_competitive(build_demo_metadata(episode, source_path=str(demo_path)))

    assert meta.protocol == 15
    assert meta.game_dir == "The Abandoned Base"
    assert meta.map_id == "dm3"
    assert meta.maxclients == 16
    assert meta.teamplay == 2
    assert meta.player_names[9] == "D13-Unholy"
    assert meta.player_names[10] == "D11-Thresh"
    assert len(meta.frag_updates) >= 250
    assert meta.tick_count > 30000
    assert meta.duration_s > 1900.0
    assert meta.classification == "competitive_teamplay"
    assert meta.mode == "teamplay"
    assert meta.classification_confidence >= 0.8


def test_thresh2_demo_exposes_team_menu_and_colors() -> None:
    demo_path = _methosq_demo_path("thresh2__thrgib2.dem")
    if not demo_path.exists():
        pytest.skip("thresh2__thrgib2.dem not available under artifacts/corpus/netquake/materialized_methosq")

    episode = load_demo(demo_path, map_id="unknown")
    meta = classify_competitive(build_demo_metadata(episode, source_path=str(demo_path)))

    assert meta.protocol == 15
    assert meta.map_id == "dm4"
    assert meta.game_dir == "The Bad Place"
    assert meta.maxclients == 16
    assert meta.text_flags["saw_team_menu_text"] is True
    assert meta.text_flags["saw_observer_text"] is True
    assert meta.player_colors[1] == 0xDD
    assert meta.player_colors[5] == 0x44
    assert meta.player_names[3] == "agn"
    assert meta.final_frags[3] == -99
    assert len(meta.frag_updates) >= 300
    assert meta.classification == "competitive_teamplay"
    assert meta.mode == "teamplay"
    assert meta.classification_confidence >= 0.8


@pytest.mark.parametrize("name", ["thresh3__THRESH3.DEM", "thresh2__thrgib2.dem"])
def test_metadata_only_parser_matches_classification_signals(name: str) -> None:
    demo_path = _methosq_demo_path(name)
    if not demo_path.exists():
        pytest.skip(f"{name} not available under artifacts/corpus/netquake/materialized_methosq")

    full_meta = classify_competitive(build_demo_metadata(load_demo(demo_path, map_id="unknown"), source_path=str(demo_path)))
    fast_meta = _metadata_only_meta(demo_path)

    assert fast_meta.protocol == full_meta.protocol
    assert fast_meta.map_id == full_meta.map_id
    assert fast_meta.game_dir == full_meta.game_dir
    assert fast_meta.maxclients == full_meta.maxclients
    assert fast_meta.player_names == full_meta.player_names
    assert fast_meta.player_colors == full_meta.player_colors
    assert fast_meta.final_frags == full_meta.final_frags
    assert fast_meta.text_flags == full_meta.text_flags
    assert fast_meta.teamplay == full_meta.teamplay
    assert fast_meta.deathmatch == full_meta.deathmatch
    assert fast_meta.duration_s == full_meta.duration_s
    assert fast_meta.classification == full_meta.classification
    assert fast_meta.mode == full_meta.mode
