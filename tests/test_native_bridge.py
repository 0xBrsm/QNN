from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.native_bridge import NativeEngineProcess, NativeQuakeAdapter
from quake_ai.actions import LOOK_NEUTRAL_LABEL
from quake_ai.models.competitive_encoder import CompetitiveObservationEncoder
from quake_ai.models.world_encoder import WorldObservationEncoder
from quake_ai.rl.environment import NativeWorldEnv


def test_native_engine_process_round_trip(native_worker_binary: Path) -> None:
    with NativeEngineProcess(executable=native_worker_binary, map_id="E1M1", fixed_tick_hz=30) as proc:
        hello = proc.start()
        assert hello["server"] == "native-stub"
        assert int(hello["tick_hz"]) == 30
        assert "world_v2" in hello["capabilities"]
        assert "reset_options" in hello["capabilities"]
        assert proc.map_state is not None
        assert proc.map_state.map_id == "E1M1"

        reset = proc.reset(seed=7)
        assert len(reset["obs"]) == 20
        assert int(reset["info"]["seed"]) == 7
        assert reset["world_tick"]["reset"] is True

        response = proc.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
        assert not bool(response["done"])
        assert float(response["reward"]) > 0.0
        assert int(response["world_tick"]["tick"]) == 1


def test_native_engine_process_reset_options_round_trip(native_worker_binary: Path) -> None:
    with NativeEngineProcess(executable=native_worker_binary, map_id="E1M1", fixed_tick_hz=20, extra_args=["-game", "frikbotnex"]) as proc:
        proc.start()
        reset = proc.reset(seed=13, options={"maxplayers": 2, "deathmatch": 1, "teamplay": 0, "post_map_commands": "addbot test"})
        assert int(reset["info"]["maxplayers"]) == 2
        assert int(reset["info"]["deathmatch"]) == 1
        assert int(reset["info"]["teamplay"]) == 0

def test_native_quake_adapter_uses_process_boundary(native_worker_binary: Path) -> None:
    adapter = NativeQuakeAdapter(executable=native_worker_binary, map_id="E1M1", fixed_tick_hz=25)
    try:
        map_state = adapter.map_state_v2()
        assert map_state is not None
        assert map_state.goal_region_ids == [4]

        obs = adapter.reset(seed=11)
        assert obs.shape == (20,)

        world_tick = adapter.reset_world(seed=11)
        assert world_tick.reset
        assert world_tick.current_region_id == 1

        done = False
        while not done:
            world_tick, reward, done, info = adapter.step_world({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
        assert world_tick.done
        assert world_tick.current_region_id in {3, 4}
        assert reward > 0.0
        assert bool(info["goal_reached"])
    finally:
        adapter.close()


def test_native_worker_reset_and_step_are_deterministic(native_worker_binary: Path) -> None:
    adapter = NativeQuakeAdapter(executable=native_worker_binary, map_id="E1M1", fixed_tick_hz=20)
    actions = [
        {"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0},
        {"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0},
        {"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 1, "jump": 0, "weapon": 0},
    ]
    try:
        first_reset = adapter.reset_world(seed=17).to_dict()
        first_rollout = [adapter.step_world(action)[0].to_dict() for action in actions]

        second_reset = adapter.reset_world(seed=17).to_dict()
        second_rollout = [adapter.step_world(action)[0].to_dict() for action in actions]
    finally:
        adapter.close()

    assert first_reset == second_reset
    assert first_rollout == second_rollout


def test_native_world_env_encodes_live_world_ticks(native_worker_binary: Path) -> None:
    env = NativeWorldEnv(executable=native_worker_binary, map_id="E1M1", max_steps=8, seed=11)
    encoder = WorldObservationEncoder()
    try:
        obs = env.reset(seed=11)
        assert obs.shape == (encoder.obs_dim,)

        next_obs, reward, done, info = env.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 0, "jump": 0, "weapon": 0})
        assert next_obs.shape == (encoder.obs_dim,)
        assert np.isfinite(reward)
        assert not done
        assert 0.0 <= float(info["goal_progress"]) <= 1.0
    finally:
        env.close()


def test_native_world_env_supports_competitive_observations_and_reward(native_worker_binary: Path) -> None:
    env = NativeWorldEnv(
        executable=native_worker_binary,
        map_id="E1M1",
        max_steps=8,
        reward_mode="combat_survival",
        observation_format="world_v2_competitive",
        native_options={"maxplayers": 2, "deathmatch": 1, "teamplay": 0},
        seed=11,
    )
    encoder = CompetitiveObservationEncoder()
    try:
        obs = env.reset(seed=11)
        assert obs.shape == (encoder.obs_dim,)

        next_obs, reward, done, info = env.step({"move": 1, "strafe": 0, "look_yaw": LOOK_NEUTRAL_LABEL, "look_pitch": LOOK_NEUTRAL_LABEL, "fire": 1, "jump": 0, "weapon": 0})
        assert next_obs.shape == (encoder.obs_dim,)
        assert np.isfinite(reward)
        assert not done
        assert int(info["visible_threats"]) >= 1
        assert float(info["frag_delta"]) >= 1.0
        assert float(info["damage_dealt"]) >= 1.0
        assert float(info["hit_count"]) >= 1.0
        assert float(info["shots_fired"]) >= 1.0
        assert int(info["frags"]) >= 1
    finally:
        env.close()
