from __future__ import annotations

import json
from pathlib import Path

from engine.adapter import DemoPlaybackHarness
from quake_ai.data.collector import collect_from_demos
from quake_ai.data.dataset import build_samples, split_samples
from quake_ai.data.packet_validation import validate_packet_alignment
from quake_ai.utils.io import read_ndjson, write_ndjson


def test_demo_replay_is_deterministic(demo_dir: Path) -> None:
    demo_path = sorted(demo_dir.glob("*.dem"))[0]
    harness = DemoPlaybackHarness(map_id="E1M1")

    run1 = [(t.to_dict(), p.to_dict()) for t, p in harness.replay(demo_path)]
    run2 = [(t.to_dict(), p.to_dict()) for t, p in harness.replay(demo_path)]
    assert run1 == run2


def test_dataset_split_has_no_episode_leak(collected_artifacts) -> None:
    samples = build_samples(collected_artifacts["telemetry"], collected_artifacts["map_features"])
    split = split_samples(samples, train_ratio=0.7, val_ratio=0.15, seed=7)

    train_eps = {s.episode_id for s in split.train}
    val_eps = {s.episode_id for s in split.val}
    test_eps = {s.episode_id for s in split.test}

    assert train_eps.isdisjoint(val_eps)
    assert train_eps.isdisjoint(test_eps)
    assert val_eps.isdisjoint(test_eps)
    assert len(split.train) + len(split.val) + len(split.test) == len(samples)


def test_packet_validator_detects_mismatch(tmp_path: Path, collected_artifacts) -> None:
    telemetry_path = collected_artifacts["telemetry"]
    packets = list(read_ndjson(collected_artifacts["packets"]))
    packets[0]["tick_estimate"] = 10_000

    broken_packets = tmp_path / "packets_broken.ndjson"
    write_ndjson(broken_packets, packets)

    report = validate_packet_alignment(telemetry_path=telemetry_path, packets_path=str(broken_packets), tick_window=2)
    assert report.unmatched_packets >= 1
    assert report.out_of_window >= 1


def test_collect_writes_expected_artifacts(tmp_path: Path, demo_dir: Path, map_fixture: Path) -> None:
    out = tmp_path / "collect"
    artifacts = collect_from_demos(map_id="E1M1", demo_dir=demo_dir, out_dir=out, map_path=map_fixture)

    for key in ["telemetry", "packets", "summaries", "map_features", "failures"]:
        assert Path(artifacts[key]).exists()

    map_payload = json.loads(Path(artifacts["map_features"]).read_text(encoding="utf-8"))
    assert map_payload["records"]


def test_collect_without_map_path_uses_observed_regions(tmp_path: Path, demo_dir: Path) -> None:
    out = tmp_path / "collect_observed"
    artifacts = collect_from_demos(map_id="E1M1", demo_dir=demo_dir, out_dir=out, map_path=None)

    telemetry_regions = {int(row["region_id"]) for row in read_ndjson(artifacts["telemetry"])}
    map_payload = json.loads(Path(artifacts["map_features"]).read_text(encoding="utf-8"))
    feature_regions = {int(record["region_id"]) for record in map_payload["records"]}

    assert telemetry_regions.issubset(feature_regions)
    assert (out / "observed_map.json").exists()


def test_collect_skips_bad_demo_and_records_failure(tmp_path: Path, demo_dir: Path, map_fixture: Path) -> None:
    working_dir = tmp_path / "mixed_demos"
    working_dir.mkdir()
    first_demo = sorted(demo_dir.glob("*.dem"))[0]
    (working_dir / first_demo.name).write_bytes(first_demo.read_bytes())
    (working_dir / "broken.dem").write_bytes(b"not-a-demo")

    artifacts = collect_from_demos(map_id="E1M1", demo_dir=working_dir, out_dir=tmp_path / "collect_mixed", map_path=map_fixture)
    failures = list(read_ndjson(artifacts["failures"]))

    assert failures
    assert any("broken.dem" in str(row.get("demo_path", "")) for row in failures)
