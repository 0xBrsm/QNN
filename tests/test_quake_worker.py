from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.adapter import DemoPlaybackHarness
from engine.native_bridge import NativeEngineProcess, NativeQuakeAdapter
from quake_ai.actions import LOOK_NEUTRAL_LABEL, look_label_from_mouse_count
from quake_ai.data.world_stream import world_ticks_from_demo_episode
from quake_ai.maps.world_model import build_world_model_from_quake_assets, nearest_region_id
from quake_ai.models.world_encoder import WorldObservationEncoder
from quake_ai.rl.environment import NativeVectorEnv, NativeWorldEnv


_SEMANTIC_EVENT_ALIASES = {
    "damage": "damage_taken",
    "intermission": "goal_reached",
}

_SEMANTIC_EVENT_TYPES = {
    "damage_taken",
    "goal_reached",
    "pickup_ammo",
    "pickup_armor",
    "pickup_health",
    "pickup_item",
    "pickup_weapon",
    "player_died",
}


def _normalized_tick(world_tick: dict[str, object]) -> dict[str, object]:
    normalized = dict(world_tick)
    normalized["episode_id"] = "normalized"
    return normalized


def _quake_worker_env(quake_assets_dir: Path) -> dict[str, str]:
    return {"QUAKE_BASEDIR": str(quake_assets_dir)}


def _require_frikbot_gamedir(quake_assets_dir: Path) -> Path:
    mod_root = quake_assets_dir / "frikbotnex"
    if not (mod_root / "progs.dat").exists():
        pytest.skip("FrikBotNex gamedir not installed under assets/frikbotnex")
    return mod_root


def _normalized_map_state(map_state: dict[str, object]) -> dict[str, object]:
    normalized = dict(map_state)
    metadata = dict(normalized.get("metadata", {}))
    metadata["source"] = "normalized"
    normalized["metadata"] = metadata
    return normalized


def _forward_action_batch(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "move": np.ones(num_envs, dtype=np.int64),
        "strafe": np.zeros(num_envs, dtype=np.int64),
        "look_yaw": np.full(num_envs, LOOK_NEUTRAL_LABEL, dtype=np.int64),
        "look_pitch": np.full(num_envs, LOOK_NEUTRAL_LABEL, dtype=np.int64),
        "fire": np.zeros(num_envs, dtype=np.int64),
        "jump": np.zeros(num_envs, dtype=np.int64),
        "weapon": np.zeros(num_envs, dtype=np.int64),
    }


def _action(
    *,
    move: int = 0,
    strafe: int = 0,
    look_yaw: int = LOOK_NEUTRAL_LABEL,
    look_pitch: int = LOOK_NEUTRAL_LABEL,
    fire: int = 0,
    jump: int = 0,
    weapon: int = 0,
) -> dict[str, int]:
    return {
        "move": move,
        "strafe": strafe,
        "look_yaw": look_yaw,
        "look_pitch": look_pitch,
        "fire": fire,
        "jump": jump,
        "weapon": weapon,
    }


def _look_left() -> int:
    return look_label_from_mouse_count(-20)


def _look_right() -> int:
    return look_label_from_mouse_count(20)


def _retained_e1m1_demo_path() -> Path:
    demo_dir = Path(__file__).resolve().parents[2] / "artifacts" / "runs" / "e1m1_corpus_world_verify" / "demos"
    demos = sorted(demo_dir.glob("*.dem"))
    if not demos:
        pytest.skip("Retained E1M1 demos not available under artifacts/runs/e1m1_corpus_world_verify/demos")
    return demos[0]


def _canonical_event_type(event_type: str) -> str | None:
    canonical = _SEMANTIC_EVENT_ALIASES.get(event_type, event_type)
    if canonical not in _SEMANTIC_EVENT_TYPES:
        return None
    return canonical


def _semantic_event_trace(world_ticks) -> list[str]:
    flattened: list[str] = []
    for world_tick in world_ticks:
        tick_events = [_canonical_event_type(event.event_type) for event in world_tick.events]
        for event_type in tick_events:
            if event_type is None:
                continue
            flattened.append(event_type)
    return flattened


def _region_transition_trace(world_ticks) -> list[int]:
    transitions: list[int] = []
    for world_tick in world_ticks:
        region_id = world_tick.current_region_id
        if region_id is None:
            continue
        if not transitions or transitions[-1] != region_id:
            transitions.append(region_id)
    return transitions


def _assert_world_tick_contract(map_state, world_ticks) -> None:
    assert world_ticks
    region_ids = {region.region_id for region in map_state.regions}
    reset_count = 0

    for expected_tick, world_tick in enumerate(world_ticks):
        assert world_tick.tick == expected_tick
        reset_count += int(bool(world_tick.reset))
        if world_tick.current_region_id is not None:
            assert world_tick.current_region_id in region_ids
            assert world_tick.current_region_id == nearest_region_id(map_state, world_tick.player.origin)
        for event in world_tick.events:
            canonical = _canonical_event_type(event.event_type)
            if canonical is None or event.region_id is None:
                continue
            assert event.region_id in region_ids
            assert event.region_id == world_tick.current_region_id

    assert world_ticks[0].reset
    assert reset_count == 1


def _replay_demo_actions(quake_worker_binary: Path, quake_assets_dir: Path, actions: list[dict[str, int]]) -> list[object]:
    adapter = NativeQuakeAdapter(
        executable=quake_worker_binary,
        map_id="E1M1",
        fixed_tick_hz=20,
        env=_quake_worker_env(quake_assets_dir),
    )
    try:
        live_ticks = [adapter.reset_world(seed=7)]
        for action in actions:
            world_tick, _, done, _ = adapter.step_world(action)
            live_ticks.append(world_tick)
            if done:
                break
    finally:
        adapter.close()
    return live_ticks


def test_quake_worker_round_trip(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    env = _quake_worker_env(quake_assets_dir)

    with NativeEngineProcess(executable=quake_worker_binary, map_id="E1M1", fixed_tick_hz=20, env=env) as proc:
        hello = proc.start()
        assert hello["server"] == "quake-worker"
        assert "listen_local" in hello["capabilities"]
        assert "udp_networking" in hello["capabilities"]
        assert proc.map_state is not None
        assert proc.map_state.map_id == "E1M1"
        assert len(proc.map_state.regions) > 0

        reset = proc.reset(seed=7)
        assert reset["world_tick"]["reset"] is True
        reset_origin = tuple(float(value) for value in reset["world_tick"]["player"]["origin"])

        response = proc.step(_action(move=1))
        assert not bool(response["done"])
        assert int(response["world_tick"]["tick"]) == 1
        step_origin = tuple(float(value) for value in response["world_tick"]["player"]["origin"])
        assert step_origin[1] > reset_origin[1]


def test_quake_worker_frikbot_post_map_command_spawns_bot(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    _require_frikbot_gamedir(quake_assets_dir)
    env = _quake_worker_env(quake_assets_dir)

    with NativeEngineProcess(
        executable=quake_worker_binary,
        map_id="dm6",
        fixed_tick_hz=20,
        env=env,
        extra_args=["-game", "frikbotnex"],
    ) as proc:
        proc.start()
        reset = proc.reset(
            seed=11,
            options={
                "maxplayers": 2,
                "skill": 0,
                "deathmatch": 1,
                "coop": 0,
                "teamplay": 0,
                "samelevel": 1,
                "post_map_commands": "impulse 100",
            },
        )
        reset_players = list(reset["world_tick"]["debug"]["server_players"])
        assert len(reset_players) >= 2

        bot_player = next(
            row
            for row in reset_players
            if int(row["entity_num"]) != 1 and str(row["classname"]) == "player" and str(row["netname"]) != "player"
        )
        reset_origin = tuple(float(value) for value in bot_player["origin"])

        step = proc.step(_action())
        step_players = list(step["world_tick"]["debug"]["server_players"])
        stepped_bot = next(row for row in step_players if str(row["netname"]) == str(bot_player["netname"]))
        step_origin = tuple(float(value) for value in stepped_bot["origin"])

        assert step_origin != reset_origin


def test_quake_worker_native_env_uses_origin_progress_within_region(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    env = NativeWorldEnv(
        executable=quake_worker_binary,
        map_id="E1M1",
        max_steps=32,
        seed=7,
        env=_quake_worker_env(quake_assets_dir),
    )

    try:
        env.reset(seed=7)
        initial_region_id = env.state.current_region_id
        initial_distance = env.state.prev_distance

        _, reward, done, info = env.step(_action(move=1))

        assert env.state.current_region_id == initial_region_id
        assert float(info["movement_delta"]) > 0.0
        assert float(info["distance_to_goal"]) < initial_distance
        assert not bool(info["stuck"])
        assert np.isfinite(reward)
        assert not done
    finally:
        env.close()


def test_quake_worker_map_state_matches_python_asset_world_model(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    env = _quake_worker_env(quake_assets_dir)
    python_map_state = build_world_model_from_quake_assets(quake_assets_dir, map_id="E1M1")

    with NativeEngineProcess(executable=quake_worker_binary, map_id="E1M1", fixed_tick_hz=20, env=env) as proc:
        proc.start()
        assert proc.map_state is not None
        assert _normalized_map_state(proc.map_state.to_dict()) == _normalized_map_state(python_map_state.to_dict())


def test_quake_worker_reset_and_step_are_deterministic(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    env = _quake_worker_env(quake_assets_dir)
    adapter = NativeQuakeAdapter(executable=quake_worker_binary, map_id="E1M1", fixed_tick_hz=20, env=env)
    actions = [
        _action(move=1),
        _action(move=1),
        _action(move=1, look_yaw=_look_left()),
    ]

    try:
        first_reset = adapter.reset_world(seed=17).to_dict()
        first_rollout = [adapter.step_world(action)[0].to_dict() for action in actions]

        second_reset = adapter.reset_world(seed=17).to_dict()
        second_rollout = [adapter.step_world(action)[0].to_dict() for action in actions]
    finally:
        adapter.close()

    assert _normalized_tick(first_reset) == _normalized_tick(second_reset)
    assert [_normalized_tick(tick) for tick in first_rollout] == [_normalized_tick(tick) for tick in second_rollout]


def test_e1m1_demo_world_ticks_match_asset_contract(quake_assets_dir: Path) -> None:
    map_state = build_world_model_from_quake_assets(quake_assets_dir, map_id="E1M1")
    demo_path = _retained_e1m1_demo_path()
    episode = DemoPlaybackHarness(map_id="E1M1").load_episode(demo_path)
    world_ticks = world_ticks_from_demo_episode(episode, map_state)

    _assert_world_tick_contract(map_state, world_ticks)
    semantic_events = _semantic_event_trace(world_ticks)

    assert world_ticks[-1].done
    assert world_ticks[-1].done_reason == "goal_reached"
    assert semantic_events.count("pickup_health") >= 2
    assert "damage_taken" in semantic_events
    assert semantic_events[-1] == "goal_reached"


def test_quake_worker_demo_replay_matches_shared_e1m1_contract(
    quake_worker_binary: Path,
    quake_assets_dir: Path,
) -> None:
    map_state = build_world_model_from_quake_assets(quake_assets_dir, map_id="E1M1")
    demo_path = _retained_e1m1_demo_path()
    episode = DemoPlaybackHarness(map_id="E1M1").load_episode(demo_path)
    demo_ticks = world_ticks_from_demo_episode(episode, map_state)
    live_ticks = _replay_demo_actions(
        quake_worker_binary,
        quake_assets_dir,
        [dict(tick.action_label) for tick in episode.ticks[:256]],
    )

    _assert_world_tick_contract(map_state, live_ticks)

    demo_regions = _region_transition_trace(demo_ticks[: len(live_ticks)])
    live_regions = _region_transition_trace(live_ticks)
    assert live_regions
    assert demo_regions
    assert live_ticks[0].current_region_id == demo_ticks[0].current_region_id
    assert live_regions[: min(3, len(live_regions), len(demo_regions))] == demo_regions[: min(3, len(live_regions), len(demo_regions))]
    assert set(_semantic_event_trace(live_ticks)).issubset(set(_semantic_event_trace(demo_ticks)))


def test_quake_worker_interleaved_resets_stay_healthy(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    adapter = NativeQuakeAdapter(
        executable=quake_worker_binary,
        map_id="E1M1",
        fixed_tick_hz=20,
        env=_quake_worker_env(quake_assets_dir),
    )
    repeated_seed_ticks: list[dict[str, object]] = []
    actions = [
        _action(move=1),
        _action(move=1, look_yaw=_look_left()),
        _action(move=1),
    ]

    try:
        map_state = adapter.map_state_v2()
        assert map_state is not None
        region_ids = {region.region_id for region in map_state.regions}

        for seed in [17, 23, 17, 29, 17]:
            reset_tick = adapter.reset_world(seed=seed).to_dict()
            assert bool(reset_tick["reset"])
            assert int(reset_tick["current_region_id"]) in region_ids
            if seed == 17:
                repeated_seed_ticks.append(_normalized_tick(reset_tick))

            for action in actions:
                world_tick, reward, _, _ = adapter.step_world(action)
                assert world_tick.current_region_id in region_ids
                assert np.isfinite(reward)
    finally:
        adapter.close()

    assert repeated_seed_ticks[0] == repeated_seed_ticks[1] == repeated_seed_ticks[2]


def test_quake_worker_extended_reset_churn_stays_healthy(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    adapter = NativeQuakeAdapter(
        executable=quake_worker_binary,
        map_id="E1M1",
        fixed_tick_hz=20,
        env=_quake_worker_env(quake_assets_dir),
    )
    actions = [
        _action(move=1),
        _action(move=1, look_yaw=_look_left()),
        _action(move=1),
        _action(strafe=1),
        _action(move=1, look_yaw=_look_right()),
        _action(fire=1),
    ]
    repeated_seed_signatures: list[dict[str, object]] = []

    try:
        map_state = adapter.map_state_v2()
        assert map_state is not None
        region_ids = {region.region_id for region in map_state.regions}

        for cycle, seed in enumerate([7, 11, 13, 7, 17, 19, 7, 23, 29, 7, 31, 37]):
            reset_tick = adapter.reset_world(seed=seed).to_dict()
            assert bool(reset_tick["reset"])
            assert int(reset_tick["current_region_id"]) in region_ids
            if seed == 7:
                repeated_seed_signatures.append(_normalized_tick(reset_tick))

            for step_index in range(24):
                action = actions[(cycle + step_index) % len(actions)]
                world_tick, reward, _, info = adapter.step_world(action)
                assert world_tick.current_region_id in region_ids
                assert np.isfinite(reward)
                assert isinstance(info.get("goal_reached", False), bool)
                if world_tick.done:
                    break
    finally:
        adapter.close()

    assert len(repeated_seed_signatures) == 4
    assert repeated_seed_signatures[0] == repeated_seed_signatures[1] == repeated_seed_signatures[2] == repeated_seed_signatures[3]


def test_quake_worker_world_env_encodes_live_ticks(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    encoder = WorldObservationEncoder()
    env = NativeWorldEnv(
        executable=quake_worker_binary,
        map_id="E1M1",
        max_steps=8,
        seed=11,
        env=_quake_worker_env(quake_assets_dir),
    )

    try:
        obs = env.reset(seed=11)
        assert obs.shape == (encoder.obs_dim,)

        next_obs, reward, done, info = env.step(_action(move=1))
        assert next_obs.shape == (encoder.obs_dim,)
        assert np.isfinite(reward)
        assert not done
        assert 0.0 <= float(info["goal_progress"]) <= 1.0
    finally:
        env.close()


def test_quake_worker_vector_env_handles_parallel_startup(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    encoder = WorldObservationEncoder()
    env = NativeVectorEnv(
        num_envs=3,
        executable=quake_worker_binary,
        map_id="E1M1",
        max_steps=4,
        seed=19,
        env=_quake_worker_env(quake_assets_dir),
    )

    try:
        obs = env.reset()
        assert obs.shape == (env.num_envs, encoder.obs_dim)

        saw_done = False
        for _ in range(6):
            next_obs, rewards, dones, infos = env.step(_forward_action_batch(env.num_envs))
            assert next_obs.shape == (env.num_envs, encoder.obs_dim)
            assert np.isfinite(rewards).all()
            assert len(infos) == env.num_envs
            assert all("worker_done" in info for info in infos)
            assert all(0.0 <= float(info["goal_progress"]) <= 1.0 for info in infos)
            saw_done = saw_done or bool(dones.any())

        assert saw_done
    finally:
        env.close()


def test_quake_worker_vector_env_recycles_under_short_churn(quake_worker_binary: Path, quake_assets_dir: Path) -> None:
    encoder = WorldObservationEncoder()

    for cycle, num_envs in enumerate([2, 3, 2]):
        env = NativeVectorEnv(
            num_envs=num_envs,
            executable=quake_worker_binary,
            map_id="E1M1",
            max_steps=8,
            seed=101 + cycle,
            env=_quake_worker_env(quake_assets_dir),
        )

        try:
            obs = env.reset()
            assert obs.shape == (num_envs, encoder.obs_dim)

            done_count = 0
            for _ in range(16):
                next_obs, rewards, dones, infos = env.step(_forward_action_batch(num_envs))
                assert next_obs.shape == (num_envs, encoder.obs_dim)
                assert np.isfinite(rewards).all()
                assert len(infos) == num_envs
                assert all("worker_done" in info for info in infos)
                done_count += int(dones.sum())

            assert done_count > 0
        finally:
            env.close()
