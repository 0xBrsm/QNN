from __future__ import annotations

import gzip
import math
import struct
import zlib

import numpy as np
import pytest
import torch

from quake_ai.data import corpus
from quake_ai.actions import ACTION_HEADS, ActionLabels, LOOK_NEUTRAL_LABEL, action_from_list, flatten_action, look_label_from_yaw_delta
from quake_ai.data.corpus import canonical_map_id, decompress_demo_payload, extract_demo_bytes, materialize_corpus_subset
from quake_ai.data.dataset import Sample, build_samples, class_weights
from quake_ai.data.demo import find_demo_files
from quake_ai.data.netquake_demo import infer_action_label
from quake_ai.maps.bsp_parser import load_quake_asset_bytes, parse_map_entities_from_quake_assets
from quake_ai.maps.features import build_map_features, write_map_features
from quake_ai.maps.world_model import build_world_model
from quake_ai.models.policy import MLPGRUPolicy
from quake_ai.rl.environment import E1M1NavigationEnv
from quake_ai.rl.reward import RewardWeights, shaped_reward
from quake_ai.schemas import EpisodeSummaryV1, MapFeaturesV1, PacketEventV1, TelemetryTickV1
from quake_ai.utils.device import TorchDeviceSpec, configure_torch_runtime, describe_torch_runtime
from quake_ai.utils.io import write_ndjson


def _build_test_bsp(entity_text: bytes) -> bytes:
    header_size = 4 + 15 * 8
    payload = bytearray(header_size + len(entity_text))
    struct.pack_into("<i", payload, 0, 29)
    struct.pack_into("<ii", payload, 4, header_size, len(entity_text))
    payload[header_size:] = entity_text
    return bytes(payload)


def _build_test_pak(entry_name: str, entry_bytes: bytes) -> bytes:
    file_offset = 12
    dir_offset = file_offset + len(entry_bytes)
    dir_length = 64
    header = b"PACK" + struct.pack("<ii", dir_offset, dir_length)
    directory = struct.pack("<56sii", entry_name.encode("ascii"), file_offset, len(entry_bytes))
    return header + entry_bytes + directory


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
            "action_label": {"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0},
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
    action = ActionLabels(move=1, strafe=2, fire=1, jump=1, weapon=4).to_dict()
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


def test_parse_map_entities_reads_bsp_from_quake_pak(tmp_path) -> None:
    id1_dir = tmp_path / "id1"
    id1_dir.mkdir()
    entity_text = (
        b'{\n"classname" "info_player_start"\n"origin" "0 0 0"\n}\n'
        b'{\n"classname" "trigger_changelevel"\n"origin" "256 0 0"\n}\n'
    )
    bsp_bytes = _build_test_bsp(entity_text)
    pak_bytes = _build_test_pak("maps/e1m1.bsp", bsp_bytes)
    (id1_dir / "PAK0.PAK").write_bytes(pak_bytes)

    loaded_bsp = load_quake_asset_bytes(tmp_path, "maps/e1m1.bsp")
    entities = parse_map_entities_from_quake_assets(tmp_path, "E1M1")

    assert loaded_bsp == bsp_bytes
    assert [entity["classname"] for entity in entities] == ["info_player_start", "trigger_changelevel"]


def test_world_model_and_features_accept_quake_asset_root(tmp_path) -> None:
    id1_dir = tmp_path / "id1"
    id1_dir.mkdir()
    entity_text = (
        b'{\n"classname" "info_player_start"\n"origin" "0 0 0"\n}\n'
        b'{\n"classname" "item_health"\n"origin" "256 0 0"\n}\n'
        b'{\n"classname" "trigger_changelevel"\n"origin" "512 0 0"\n}\n'
    )
    bsp_bytes = _build_test_bsp(entity_text)
    pak_bytes = _build_test_pak("maps/e1m1.bsp", bsp_bytes)
    (id1_dir / "PAK0.PAK").write_bytes(pak_bytes)

    map_state = build_world_model(tmp_path, map_id="E1M1")
    records = build_map_features(tmp_path, map_id="E1M1")

    assert map_state.metadata["source"] == "quake_assets"
    assert map_state.goal_region_ids
    assert any(record.goal_nodes for record in records)


def test_reward_invariants() -> None:
    weights = RewardWeights()
    improved = shaped_reward(5.0, 4.0, False, False, False, False, weights)
    regressed = shaped_reward(4.0, 5.0, False, False, False, False, weights)
    assert improved > regressed

    complete = shaped_reward(1.0, 0.0, True, True, False, False, weights)
    timeout = shaped_reward(1.0, 1.0, False, False, True, True, weights)
    assert complete > timeout
    assert math.isfinite(complete)


def test_combat_reward_invariants() -> None:
    weights = RewardWeights(mode="combat_survival")
    improved_progress = shaped_reward(
        5.0,
        4.5,
        False,
        False,
        False,
        False,
        weights,
        combat_signals={
            "visible_threats": 1.0,
            "health_fraction": 0.8,
            "armor_fraction": 0.3,
        },
    )
    regressed_progress = shaped_reward(
        4.5,
        5.0,
        False,
        False,
        False,
        False,
        weights,
        combat_signals={
            "visible_threats": 1.0,
            "health_fraction": 0.8,
            "armor_fraction": 0.3,
        },
    )
    rewarded = shaped_reward(
        0.0,
        0.0,
        False,
        False,
        False,
        False,
        weights,
        combat_signals={
            "visible_threats": 1.0,
            "effective_fire": 1.0,
            "frag_gain": 1.0,
            "health_fraction": 0.8,
            "armor_fraction": 0.3,
        },
    )
    punished = shaped_reward(
        0.0,
        0.0,
        False,
        False,
        False,
        True,
        weights,
        combat_signals={
            "blind_fire": 1.0,
            "damage_taken": 20.0,
            "player_died": 1.0,
            "health_fraction": 0.1,
            "armor_fraction": 0.0,
        },
    )

    assert improved_progress > regressed_progress
    assert rewarded > punished
    assert math.isfinite(rewarded)


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
        "look_yaw": np.array([LOOK_NEUTRAL_LABEL, LOOK_NEUTRAL_LABEL], dtype=np.int64),
        "look_pitch": np.array([LOOK_NEUTRAL_LABEL, LOOK_NEUTRAL_LABEL], dtype=np.int64),
        "fire": np.array([0, 0], dtype=np.int64),
        "jump": np.array([0, 0], dtype=np.int64),
        "weapon": np.array([0, 0], dtype=np.int64),
    }
    class_weights = {head: np.ones(size, dtype=np.float32) for head, size in ACTION_HEADS.items()}
    model = MLPGRUPolicy(obs_dim=obs.shape[1], trunk_hidden=32, seed=5, device="cpu")
    before = model.w1.copy()

    model.supervised_step(obs, actions, class_weights, lr=0.05)

    assert not np.allclose(before, model.w1)


def test_class_weights_are_tempered_and_clipped() -> None:
    samples = []
    for tick in range(12):
        samples.append(
            Sample(
                episode_id="ep",
                tick=tick,
                obs=np.zeros(4, dtype=np.float32),
                action={
                    "move": 0 if tick < 10 else 2,
                    "strafe": 0,
                    "look_yaw": LOOK_NEUTRAL_LABEL,
                    "look_pitch": LOOK_NEUTRAL_LABEL,
                    "fire": 0 if tick < 11 else 1,
                    "jump": 0,
                    "weapon": 0 if tick < 11 else 4,
                },
                goal_progress=0.0,
                done=False,
            )
        )

    weights = class_weights(samples, power=0.5, min_weight=0.5, max_weight=2.0)

    for head, values in weights.items():
        assert np.all(values >= 0.5)
        assert np.all(values <= 2.0)
    assert weights["fire"][1] > weights["fire"][0]
    assert weights["weapon"][4] > weights["weapon"][0]


def test_describe_torch_runtime_cpu() -> None:
    runtime = describe_torch_runtime("cpu")
    assert runtime["requested_device"] == "cpu"
    assert runtime["resolved_device"] == "cpu"


def test_configure_torch_runtime_skips_high_matmul_precision_on_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: calls.append(value))

    configure_torch_runtime(TorchDeviceSpec(requested="gpu", resolved="cuda", backend="rocm"))

    assert calls == []


def test_configure_torch_runtime_uses_high_matmul_precision_off_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: calls.append(value))

    configure_torch_runtime(TorchDeviceSpec(requested="cpu", resolved="cpu", backend="cpu"))

    assert calls == ["high"]


def test_goal_reaches_automatically(map_fixture, tmp_path) -> None:
    records = build_map_features(map_fixture, map_id="E1M1")
    features_path = tmp_path / "map_features.json"
    write_map_features(features_path, records)

    env = E1M1NavigationEnv(features_path, max_steps=8, seed=3)
    obs = env.reset(seed=3, start_variant=0)
    assert obs.shape[0] == 20

    reached_goal = False
    for _ in range(6):
        _, _, done, info = env.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
        if bool(info["at_goal"]):
            reached_goal = True
            assert done
            assert bool(info["goal_reached"])
            break

    assert reached_goal


def test_infer_action_label_tracks_motion_and_view_delta() -> None:
    action = infer_action_label(
        current_pos=[0.0, 0.0, 0.0],
        next_pos=[24.0, -16.0, 0.0],
        current_yaw=0.0,
        next_yaw=-12.0,
        terminal_use=True,
    )
    assert action == {
        "move": 1,
        "strafe": 2,
        "look_yaw": look_label_from_yaw_delta(-12.0),
        "look_pitch": LOOK_NEUTRAL_LABEL,
        "fire": 0,
        "jump": 0,
        "weapon": 0,
    }


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


def test_extract_demo_bytes_uses_remote_prefix_with_cli_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fake_remote_read(storage_root: str, relative_path: str, username: str, password: str) -> bytes:
        calls.append((storage_root, relative_path, username, password))
        return b"-1\nfake-demo"

    monkeypatch.setattr(corpus, "smbclient", None)
    monkeypatch.setattr(corpus, "_read_remote_bytes_with_cli", fake_remote_read)

    payload = extract_demo_bytes(
        {
            "storage_backend": "remote",
            "storage_root": r"\\pi.local\nqcorpus",
            "local_path": "raw/example.dem",
            "extracted_dem_path": "",
        },
        remote_username="guest",
        remote_password="guest",
        remote_prefix="netquake",
    )

    assert payload == b"-1\nfake-demo"
    assert calls == [(r"\\pi.local\nqcorpus", "netquake/raw/example.dem", "guest", "guest")]


def test_build_samples_preserve_player_action_space(map_fixture, tmp_path) -> None:
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
    assert samples[0].action["look_yaw"] == LOOK_NEUTRAL_LABEL
    assert samples[0].action["weapon"] == 0
