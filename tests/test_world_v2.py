from __future__ import annotations

import json

from engine.adapter import DemoPlaybackHarness
from quake_ai.data.dataset import build_world_samples
from quake_ai.data.world_stream import world_ticks_from_demo_episode
from quake_ai.maps.world_model import build_world_model
from quake_ai.models.world_encoder import WorldObservationEncoder
from quake_ai.actions import LOOK_NEUTRAL_LABEL
from quake_ai.utils.io import write_json
from quake_ai.schemas import (
    EntityStateV2,
    MapStateV2,
    PlayerStateV2,
    RegionNodeV2,
    StaticObjectV2,
    WorldEventV2,
    WorldTickV2,
)

import pytest


def test_v2_schema_round_trip_and_validation() -> None:
    map_state = MapStateV2(
        map_id="E1M1",
        regions=[
            RegionNodeV2(
                region_id=1,
                center=[0.0, 0.0, 0.0],
                neighbors=[2],
                bounds_min=[-128.0, -128.0, -64.0],
                bounds_max=[128.0, 128.0, 64.0],
                object_ids=["spawn_0000"],
                visibility_hints=[2],
            ),
            RegionNodeV2(
                region_id=2,
                center=[256.0, 0.0, 0.0],
                neighbors=[1],
                bounds_min=[128.0, -128.0, -64.0],
                bounds_max=[384.0, 128.0, 64.0],
                object_ids=["goal_0001"],
                visibility_hints=[1],
            ),
        ],
        static_objects=[
            StaticObjectV2(
                object_id="spawn_0000",
                category="spawn",
                classname="info_player_start",
                region_id=1,
                origin=[0.0, 0.0, 0.0],
                angles=[0.0, 0.0, 0.0],
                properties={},
            ),
            StaticObjectV2(
                object_id="goal_0001",
                category="goal",
                classname="trigger_changelevel",
                region_id=2,
                origin=[256.0, 0.0, 0.0],
                angles=[0.0, 0.0, 0.0],
                properties={},
            ),
        ],
        spawn_region_ids=[1],
        goal_region_ids=[2],
        metadata={"distance_to_goal": {"1": 1.0, "2": 0.0}, "max_distance_to_goal": 1.0},
    )
    payload = map_state.to_dict()
    restored = MapStateV2.from_dict(payload)
    assert restored.to_dict()["map_id"] == "E1M1"

    world_tick = WorldTickV2(
        episode_id="ep",
        map_id="E1M1",
        tick=0,
        player=PlayerStateV2(
            origin=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            view_angles=[0.0, 0.0, 0.0],
            health=100,
            armor=0,
            ammo=25,
            weapon_id=1,
            grounded=True,
        ),
        current_region_id=1,
        visible_entities=[
            EntityStateV2(
                entity_id="entity_0007",
                entity_num=7,
                classname="item_health",
                region_id=2,
                origin=[256.0, 0.0, 0.0],
                velocity=[0.0, 0.0, 0.0],
                angles=[0.0, 0.0, 0.0],
                model_id=1,
                frame=0,
                visible=True,
                properties={},
            )
        ],
        events=[WorldEventV2(event_type="pickup_health", region_id=2, payload={"delta": 25})],
        action_label={"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0},
        action_history=[{"move": 0, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0}],
        done=False,
        done_reason="",
        reset=True,
        debug={"packet": {"seq": 0}},
    )
    assert WorldTickV2.from_dict(world_tick.to_dict()).to_dict()["episode_id"] == "ep"

    with pytest.raises(ValueError):
        WorldTickV2.from_dict(
            {
                "episode_id": "ep",
                "map_id": "E1M1",
                "tick": -1,
                "player": world_tick.player.to_dict(),
                "current_region_id": 1,
                "visible_entities": [],
                "events": [],
            }
        )


def test_world_tick_allows_negative_player_health() -> None:
    world_tick = WorldTickV2(
        episode_id="ep",
        map_id="E1M1",
        tick=12,
        player=PlayerStateV2(
            origin=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            view_angles=[0.0, 0.0, 0.0],
            health=-12,
            armor=0,
            ammo=0,
            weapon_id=1,
            grounded=False,
        ),
        current_region_id=1,
        visible_entities=[],
        events=[],
        action_label={"move": 0, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0},
        action_history=[],
        done=True,
        done_reason="death",
        reset=False,
        debug={},
    )

    restored = WorldTickV2.from_dict(world_tick.to_dict())

    assert restored.player.health == -12


def test_world_model_is_deterministic(map_fixture) -> None:
    model_a = build_world_model(map_fixture, map_id="E1M1")
    model_b = build_world_model(map_fixture, map_id="E1M1")

    assert model_a.to_dict() == model_b.to_dict()
    assert len({obj.object_id for obj in model_a.static_objects}) == len(model_a.static_objects)
    assert {"spawn", "item", "goal"}.issubset({obj.category for obj in model_a.static_objects})


def test_demo_world_stream_emits_v2_ticks(demo_dir, map_fixture) -> None:
    demo_path = sorted(demo_dir.glob("*.dem"))[0]
    map_state = build_world_model(map_fixture, map_id="E1M1")
    harness = DemoPlaybackHarness(map_id="E1M1")
    episode = harness.load_episode(demo_path)
    world_ticks = world_ticks_from_demo_episode(episode, map_state)

    assert len(world_ticks) == len(episode.ticks)
    assert world_ticks[0].reset
    assert all(tick.map_id == "E1M1" for tick in world_ticks)
    assert any(tick.events for tick in world_ticks)
    assert any(tick.visible_entities for tick in world_ticks)


def test_build_world_samples(collected_artifacts) -> None:
    samples = build_world_samples(collected_artifacts["world_ticks"], collected_artifacts["world_map"])
    encoder = WorldObservationEncoder()

    assert samples
    assert samples[0].obs.shape == (encoder.obs_dim,)
    assert any(sample.action["move"] in {0, 1, 2} for sample in samples)


def test_build_world_samples_accepts_classification_mode_filter(tmp_path) -> None:
    map_state = MapStateV2(
        map_id="DM4",
        regions=[
            RegionNodeV2(
                region_id=0,
                center=[0.0, 0.0, 0.0],
                neighbors=[],
                bounds_min=[-512.0, -512.0, -128.0],
                bounds_max=[512.0, 512.0, 128.0],
                object_ids=[],
                visibility_hints=[],
            )
        ],
        static_objects=[],
        spawn_region_ids=[0],
        goal_region_ids=[0],
        metadata={"distance_to_goal": {"0": 0.0}, "max_distance_to_goal": 1.0},
    )
    world_tick = WorldTickV2(
        episode_id="duel_ep",
        map_id="DM4",
        tick=0,
        player=PlayerStateV2(
            origin=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            view_angles=[0.0, 0.0, 0.0],
            health=100,
            armor=0,
            ammo=25,
            weapon_id=1,
            grounded=True,
        ),
        current_region_id=0,
        visible_entities=[],
        events=[],
        action_label={
            "move": 1,
            "strafe": 0,
            "look_yaw": LOOK_NEUTRAL_LABEL,
            "look_pitch": LOOK_NEUTRAL_LABEL,
            "fire": 0,
            "jump": 0,
            "weapon": 0,
        },
        action_history=[],
        done=False,
        done_reason="",
        reset=True,
        debug={},
    )

    world_map_path = tmp_path / "world_map.json"
    world_ticks_path = tmp_path / "world_ticks.ndjson"
    write_json(world_map_path, map_state.to_dict())
    world_ticks_path.write_text(json.dumps(world_tick.to_dict()) + "\n", encoding="utf-8")

    samples = build_world_samples(
        world_ticks_path,
        world_map_path,
        metadata_index={
            "duel_ep": {
                "episode_id": "duel_ep",
                "mode": "ffa",
                "classification": "competitive_ffa",
                "source_path": "demo.dem",
            }
        },
        mode_filter=["competitive_ffa"],
    )

    assert len(samples) == 1
    assert samples[0].mode == "ffa"
    assert samples[0].source_path == "demo.dem"
