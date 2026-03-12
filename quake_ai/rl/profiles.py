"""Training profile definitions and scenario loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

from quake_ai.utils.io import read_json

if TYPE_CHECKING:
    from quake_ai.rl.planning import RuntimePlan

_SCENARIOS_PATH_DEFAULT = __file__.replace("profiles.py", "configs/combat_bot_scenarios.json")


def _deep_merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_config(existing, value)
            continue
        if isinstance(value, list):
            merged[key] = list(value)
            continue
        merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class LiveProfile:
    name: str
    default_demo_dir: str
    auto_demo_dirs: tuple[str, ...]
    collect_out: str
    plan_path: str
    runtime_scale: str = "live"
    profile_group: str = ""
    profile_note: str = ""
    scenario_id: str = ""
    retained_role: str = ""
    comparison_profiles: tuple[str, ...] = ()
    bc_overrides: Dict[str, Any] = field(default_factory=dict)
    ppo_overrides: Dict[str, Any] = field(default_factory=dict)
    eval_overrides: Dict[str, Any] = field(default_factory=dict)


_DEFAULT_BOT_NATIVE_ARGS = ["-game", "frikbotnex_train"]
_DEFAULT_BOT_OPTIONS: Dict[str, Any] = {
    "maxplayers": 2,
    "skill": 0,
    "deathmatch": 1,
    "coop": 0,
    "teamplay": 0,
    "fraglimit": 0,
    "timelimit": 0,
    "samelevel": 1,
    "pre_map_commands": "",
    "post_map_commands": "impulse 100",
}

_BC_DEFAULTS: Dict[str, Any] = {
    "train_ratio": 0.72, "val_ratio": 0.14, "lr": 0.008,
    "use_gru": True, "gru_hidden": 64, "trunk_hidden": 192,
    "class_weight_power": 0.6, "class_weight_min": 0.5, "class_weight_max": 2.5,
    "device": "auto",
}
_PPO_DEFAULTS: Dict[str, Any] = {
    "mode": "pvp", "native_executable": "../artifacts/bin/quake_worker",
    "fixed_tick_hz": 20, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_ratio": 0.2, "policy_lr": 0.00025, "ppo_epochs": 2,
    "value_coef": 0.5, "entropy_coef": 0.002, "max_grad_norm": 0.5,
    "bc_kl_coef": 0.05, "device": "auto",
}
_EVAL_DEFAULTS: Dict[str, Any] = {
    "mode": "pvp", "native_executable": "../artifacts/bin/quake_worker",
    "fixed_tick_hz": 20, "policy_modes": ["greedy", "sampled"],
    "start_mode": "randomized", "holdout_seed_offset": 10000,
    "sample_seed_offset": 20000, "device": "auto",
}
_SCALE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "verify": {
        "bc":   {"batch_size": 512, "epochs": 6, "patience": 2, "seed": 7},
        "ppo":  {"num_envs": 4, "max_steps_per_episode": 512, "rollout_steps": 64,
                 "total_steps": 4096, "minibatch_size": 128, "seed": 11},
        "eval": {"num_episodes": 16, "max_steps_per_episode": 512, "seed": 19},
    },
    "live": {
        "bc":   {"batch_size": 1024, "epochs": 8, "patience": 3, "seed": 11},
        "ppo":  {"num_envs": 8, "max_steps_per_episode": 1024, "rollout_steps": 128,
                 "total_steps": 32768, "minibatch_size": 512, "seed": 17},
        "eval": {"num_episodes": 32, "max_steps_per_episode": 1024, "seed": 23},
    },
}


def _build_scenario_profile(
    *,
    name: str,
    runtime_scale: str,
    default_demo_dir: str,
    auto_demo_dirs: tuple[str, ...],
    scenario: Mapping[str, Any],
    comparison_profiles: tuple[str, ...],
) -> LiveProfile:
    scenario_id = str(scenario["scenario_id"])
    output_root = str(scenario[f"{runtime_scale}_output_root"])
    map_id = str(scenario["map_id"])
    options = _deep_merge_config(_DEFAULT_BOT_OPTIONS, scenario.get("options", {}))
    use_gru = bool(scenario.get("use_gru", True))
    gru_hidden = int(scenario.get("gru_hidden", 64))
    bc_use_gru = bool(scenario.get("bc_use_gru", use_gru))
    bc_gru_hidden = int(scenario.get("bc_gru_hidden", gru_hidden if bc_use_gru else 0))

    return LiveProfile(
        name=name,
        default_demo_dir=default_demo_dir,
        auto_demo_dirs=auto_demo_dirs,
        collect_out=f"{output_root}/collect",
        plan_path=f"{output_root}/live_training_plan.json",
        runtime_scale="verify" if runtime_scale == "verify" else "live",
        profile_group="combat_bot_ladder",
        profile_note=str(scenario.get("skill_note", "")).strip(),
        scenario_id=scenario_id,
        retained_role=str(scenario.get("retained_role", "")).strip(),
        comparison_profiles=comparison_profiles,
        bc_overrides={
            "map_id": map_id,
            "output_dir": f"{output_root}/bc",
            "use_gru": bc_use_gru,
            "gru_hidden": bc_gru_hidden,
        },
        ppo_overrides={
            "map_id": map_id,
            "output_dir": f"{output_root}/ppo",
            "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
            "options": options,
            "use_gru": use_gru,
            "gru_hidden": gru_hidden,
        },
        eval_overrides={
            "map_id": map_id,
            "checkpoint_path": f"{output_root}/ppo/ppo_model.npz",
            "output_dir": f"{output_root}/eval",
            "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
            "options": options,
        },
    )


def _load_bot_ladder_profiles(scenarios_path: str = _SCENARIOS_PATH_DEFAULT) -> dict[str, LiveProfile]:
    scenarios_payload = read_json(scenarios_path)
    scenarios = scenarios_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise RuntimeError(f"{scenarios_path} must define a scenarios list")

    profiles: dict[str, LiveProfile] = {}
    for raw_scenario in scenarios:
        if not isinstance(raw_scenario, Mapping):
            continue
        scenario = dict(raw_scenario)
        base_name = f"combat-bot-{str(scenario['scenario_id']).strip()}"
        verify_name = f"{base_name}-verify"
        live_name = base_name
        comparison = tuple(str(value) for value in scenario.get("comparison_profiles", ()) if str(value).strip())
        profiles[verify_name] = _build_scenario_profile(
            name=verify_name,
            runtime_scale="verify",
            default_demo_dir="tests/demo_data",
            auto_demo_dirs=("tests/demo_data",),
            scenario=scenario,
            comparison_profiles=comparison,
        )
        profiles[live_name] = _build_scenario_profile(
            name=live_name,
            runtime_scale="live",
            default_demo_dir="../artifacts/corpus/netquake/materialized_competitive",
            auto_demo_dirs=("../artifacts/corpus/netquake/materialized_competitive", "tests/demo_data"),
            scenario=scenario,
            comparison_profiles=comparison,
        )
    return profiles


def _build_multi_scenario_profile(name: str, runtime_scale: str) -> LiveProfile:
    is_verify = runtime_scale == "verify"
    output_root = "../artifacts/runs/competitive_bot_multi_verify" if is_verify else "../artifacts/runs/competitive_bot_multi_live"
    return LiveProfile(
        name=name,
        default_demo_dir="tests/demo_data" if is_verify else "../artifacts/corpus/netquake/materialized_competitive",
        auto_demo_dirs=("tests/demo_data",) if is_verify else ("../artifacts/corpus/netquake/materialized_competitive", "tests/demo_data"),
        collect_out=f"{output_root}/collect",
        plan_path=f"{output_root}/live_training_plan.json",
        runtime_scale="verify" if is_verify else "live",
        profile_group="combat_bot_multi",
        profile_note="Multi-scenario PvP training across the full bot ladder.",
        scenario_id="multi",
        bc_overrides={"output_dir": f"{output_root}/bc"},
        ppo_overrides={
            "output_dir": f"{output_root}/ppo",
            "scenario_config_path": _SCENARIOS_PATH_DEFAULT,
        },
        eval_overrides={
            "checkpoint_path": f"{output_root}/ppo/ppo_model.npz",
            "output_dir": f"{output_root}/eval",
            "scenario_config_path": _SCENARIOS_PATH_DEFAULT,
        },
    )


PROFILES: dict[str, LiveProfile] = {
    "combat-bot-multi-verify": _build_multi_scenario_profile("combat-bot-multi-verify", "verify"),
    "combat-bot-multi": _build_multi_scenario_profile("combat-bot-multi", "live"),
}
PROFILES.update(_load_bot_ladder_profiles())


def load_config_with_runtime(
    profile: LiveProfile,
    plan: RuntimePlan,
    requested_device: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scale = _SCALE[profile.runtime_scale]
    bc_cfg: dict[str, Any] = {**_BC_DEFAULTS, **scale["bc"], **profile.bc_overrides}
    ppo_cfg: dict[str, Any] = {
        **_PPO_DEFAULTS, **scale["ppo"],
        "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
        "options": dict(_DEFAULT_BOT_OPTIONS),
        **profile.ppo_overrides,
    }
    eval_cfg: dict[str, Any] = {
        **_EVAL_DEFAULTS, **scale["eval"],
        "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
        "options": dict(_DEFAULT_BOT_OPTIONS),
        **profile.eval_overrides,
    }

    bc_cfg["device"] = requested_device
    ppo_cfg["device"] = requested_device
    eval_cfg["device"] = requested_device

    bc_cfg["batch_size"] = max(int(bc_cfg.get("batch_size", 1)), plan.bc_batch_size)
    ppo_cfg["num_envs"] = max(int(ppo_cfg.get("num_envs", 1)), plan.num_envs)
    ppo_cfg["rollout_steps"] = max(int(ppo_cfg.get("rollout_steps", 1)), plan.rollout_steps)
    ppo_cfg["total_steps"] = max(int(ppo_cfg.get("total_steps", 1)), plan.total_steps)
    ppo_cfg["minibatch_size"] = max(int(ppo_cfg.get("minibatch_size", 1)), plan.minibatch_size)
    eval_cfg["num_episodes"] = max(int(eval_cfg.get("num_episodes", 1)), plan.eval_episodes)
    eval_cfg["num_envs"] = max(int(eval_cfg.get("num_envs", 1)), min(plan.num_envs, int(eval_cfg["num_episodes"])))

    return bc_cfg, ppo_cfg, eval_cfg
