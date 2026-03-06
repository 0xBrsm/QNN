from __future__ import annotations

import gzip
import math
import zlib

import numpy as np
import pytest

from quake_ai.actions import ActionLabels, action_from_list, flatten_action
from quake_ai.data.corpus import canonical_map_id, decompress_demo_payload, materialize_corpus_subset
from quake_ai.data.dataset import build_samples
from quake_ai.data.demo import find_demo_files
from quake_ai.data.netquake_demo import infer_action_label
from quake_ai.maps.features import build_map_features, write_map_features
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import E1M1NavigationEnv
from quake_ai.rl.reward import RewardWeights, shaped_reward
from quake_ai.schemas import EpisodeSummaryV1, MapFeaturesV1, PacketEventV1, TelemetryTickV1
from quake_ai.utils.device import describe_torch_runtime
from quake_ai.utils.io import write_ndjson


def test_schema_round_trip() -> None:
    telemetry = TelemetryTickV1.from_dict(
        {
            "episode_id": "ep",
            "tick": 0,
            "player_pos": [0, 0, 0],
            "player_vel": [0, 0, 0],
            "yaw": 0,
            "health": 100,
            "armor": 0,
            "ammo": 25,
            "weapon_id": 1,
            "nearby_item_flags": [0, 1],
            "goal_progress": 0.1,
            "action_label": {"move": 1, "strafe": 0, "turn": 0, "fire": 0, "use": 0, "jump": 0},
            "done": False,
            "done_reason": "",
        }
    )
    assert telemetry.to_dict()["episode_id"] == "ep"

    packet = PacketEventV1.from_dict(
        {
            "episode_id": "ep",
            "tick_estimate": 0,
            "direction": "client_to_server",
            "seq": 1,
            "ack": 0,
            "payload_type": "move_cmd",
            "decoded_fields": {"size": 3},
        }
    )
    assert packet.seq == 1

    map_record = MapFeaturesV1.from_dict(
        {
            "map_id": "E1M1",
            "region_id": 1,
            "spawn_points": [[0, 0, 0]],
            "item_nodes": [],
            "connectivity_edges": [[1, 2]],
            "goal_nodes": [2],
            "distance_to_goal": 1,
        }
    )
    assert map_record.distance_to_goal == 1

    summary = EpisodeSummaryV1.from_dict(
        {
            "episode_id": "ep",
            "steps": 10,
            "goal_reached": True,
            "items_collected": 3,
            "time_to_goal": 0.5,
            "return": 5.0,
        }
    )
    assert summary.to_dict()["return"] == 5.0


def test_action_round_trip() -> None:
    action = ActionLabels(move=1, strafe=2, turn=0, use=1).to_dict()
    flat = flatten_action(action)
    restored = action_from_list(flat)
    assert restored == action


@pytest.mark.parametrize("invalid_value", [-1, 3])
def test_action_validation_fails(invalid_value: int) -> None:
    with pytest.raises(ValueError):
        ActionLabels(move=invalid_value).validate()


def test_map_parser_feature_generation(map_fixture) -> None:
    records = build_map_features(map_fixture, map_id="E1M1")
    assert records
    assert any(rec.goal_nodes for rec in records)
    assert any(rec.spawn_points for rec in records)


def test_reward_invariants() -> None:
    weights = RewardWeights()
    improved = shaped_reward(5.0, 4.0, False, False, False, False, False, weights)
    regressed = shaped_reward(4.0, 5.0, False, False, False, False, False, weights)
    assert improved > regressed

    complete = shaped_reward(1.0, 0.0, True, True, False, False, False, weights)
    timeout = shaped_reward(1.0, 1.0, False, False, True, True, False, weights)
    assert complete > timeout
    assert math.isfinite(complete)


def test_supervised_step_updates_trunk_weights() -> None:
    obs = np.array(
        [
            [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.1, -0.1, -0.4, 0.0, 1.0],
            [0.1, 0.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.4, 0.0, -0.2, -0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = {
        "move": np.array([1, 1], dtype=np.int64),
        "strafe": np.array([0, 0], dtype=np.int64),
        "turn": np.array([0, 0], dtype=np.int64),
        "use": np.array([0, 0], dtype=np.int64),
    }
    class_weights = {head: np.ones(size, dtype=np.float32) for head, size in {"move": 3, "strafe": 3, "turn": 3, "use": 2}.items()}
    model = MLPGRUPolicy(obs_dim=obs.shape[1], trunk_hidden=32, seed=5, device="cpu")
    before = model.w1.copy()

    model.supervised_step(obs, actions, class_weights, lr=0.05)

    assert not np.allclose(before, model.w1)


def test_describe_torch_runtime_cpu() -> None:
    runtime = describe_torch_runtime("cpu")
    assert runtime["requested_device"] == "cpu"
    assert runtime["resolved_device"] == "cpu"


def test_goal_requires_use(map_fixture, tmp_path) -> None:
    records = build_map_features(map_fixture, map_id="E1M1")
    features_path = tmp_path / "map_features.json"
    write_map_features(features_path, records)

    env = E1M1NavigationEnv(features_path, max_steps=8, seed=3)
    obs = env.reset(seed=3, start_variant=0)
    assert obs.shape[0] == 20

    at_goal_without_use = False
    for _ in range(6):
        _, _, done, info = env.step({"move": 1, "strafe": 0, "turn": 0, "use": 0})
        if bool(info["at_goal"]):
            at_goal_without_use = True
            assert not done
            break

    assert at_goal_without_use
    _, _, done, info = env.step({"move": 0, "strafe": 0, "turn": 0, "use": 1})
    assert done
    assert bool(info["goal_reached"])


def test_infer_action_label_tracks_motion_and_terminal_use() -> None:
    action = infer_action_label(
        current_pos=[0.0, 0.0, 0.0],
        next_pos=[24.0, -16.0, 0.0],
        current_yaw=0.0,
        next_yaw=-12.0,
        terminal_use=True,
    )
    assert action == {"move": 1, "strafe": 2, "turn": 2, "use": 1}


def test_find_demo_files_recurse_case_insensitive(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "episode.DEM").write_text("{}", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    demos = find_demo_files(tmp_path)
    assert demos == [nested / "episode.DEM"]


def test_materialize_corpus_subset_reads_local_dz_manifest(tmp_path) -> None:
    storage = tmp_path / "storage"
    raw_dir = storage / "raw" / "example.com"
    raw_dir.mkdir(parents=True)
    demo_bytes = b"binary-demo-payload"
    (raw_dir / "e1m1_001.dz").write_bytes(gzip.compress(demo_bytes))

    manifest = tmp_path / "manifest.ndjson"
    manifest.write_text(
        "\n".join(
            [
                '{"url":"https://example.com/e1m1_001.dz","sha256":"abc123","storage_backend":"local","storage_root":"%s","local_path":"raw/example.com/e1m1_001.dz","extracted_dem_path":""}'
                % str(storage),
                '{"url":"https://example.com/dm4_001.dz","sha256":"def456","storage_backend":"local","storage_root":"%s","local_path":"raw/example.com/dm4_001.dz","extracted_dem_path":""}'
                % str(storage),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "materialized"
    summary = materialize_corpus_subset(manifest_path=manifest, output_dir=out, map_id="E1M1")

    assert summary["materialized_demos"] == 1
    materialized = sorted(out.glob("*.dem"))
    assert len(materialized) == 1
    assert materialized[0].read_bytes() == demo_bytes


def test_canonical_map_id_matches_prefix_variants() -> None:
    assert canonical_map_id("e1m1a_001.dem") == "e1m1"
    assert canonical_map_id("E1M1-special.zip") == "e1m1"


def test_decompress_demo_payload_supports_dzip_wrapper() -> None:
    demo_bytes = b"-1\nfake-demo"
    dzip_bytes = b"DZ\x03\x00" + (0).to_bytes(8, "little") + zlib.compress(demo_bytes)
    assert decompress_demo_payload(dzip_bytes, source_name="sample.dz") == demo_bytes


def test_build_samples_forces_use_on_goal_region(map_fixture, tmp_path) -> None:
    records = build_map_features(map_fixture, map_id="E1M1")
    goal_region = records[0].goal_nodes[0]
    features_path = tmp_path / "map_features.json"
    write_map_features(features_path, records)

    telemetry_path = tmp_path / "telemetry.ndjson"
    write_ndjson(
        telemetry_path,
        [
            TelemetryTickV1(
                episode_id="ep",
                tick=0,
                player_pos=[1024.0, 1024.0, 0.0],
                player_vel=[0.0, 0.0, 0.0],
                yaw=0.0,
                health=100,
                armor=0,
                ammo=25,
                weapon_id=1,
                nearby_item_flags=[0, 0, 0, 0],
                goal_progress=1.0,
                action_label={"move": 0, "strafe": 0, "turn": 0, "use": 0},
                done=True,
                done_reason="goal_reached",
                region_id=goal_region,
            ).to_dict()
        ],
    )

    samples = build_samples(telemetry_path, features_path)
    assert len(samples) == 1
    assert samples[0].action["use"] == 1
