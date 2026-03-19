"""Training profile definitions and scenario loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from quake_ai.utils.io import read_json

if TYPE_CHECKING:
    from quake_ai.rl.planning import RuntimePlan

_CONFIGS_DIR = Path(__file__).with_name("configs")
_SCENARIOS_PATH_DEFAULT = str(_CONFIGS_DIR / "combat_bot_scenarios.json")
_HYPERPARAMS_PATH_DEFAULT = _CONFIGS_DIR / "hyperparams.json"

# ---------------------------------------------------------------------------
# Shared BC (behaviour cloning) location
# ---------------------------------------------------------------------------
# BC is trained once per architecture and shared across all PPO profiles.
BC_OUTPUT_DIR = "assets/runs/bc"
BC_COLLECT_DIR = f"{BC_OUTPUT_DIR}/collect"
BC_CHECKPOINT = f"{BC_OUTPUT_DIR}/bc_best_model.npz"


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
    plan_path: str
    runtime_scale: str = "live"
    profile_group: str = ""
    profile_note: str = ""
    scenario_id: str = ""
    retained_role: str = ""
    comparison_profiles: tuple[str, ...] = ()
    bc_checkpoint: str = BC_CHECKPOINT
    ppo_overrides: Dict[str, Any] = field(default_factory=dict)
    eval_overrides: Dict[str, Any] = field(default_factory=dict)


_DEFAULT_BOT_NATIVE_ARGS = ["-game", "frikbotnex_train"]
_DEFAULT_BOT_OPTIONS: Dict[str, Any] = {
    "maxplayers": 7,
    "skill": 0,
    "deathmatch": 1,
    "coop": 0,
    "teamplay": 0,
    "fraglimit": 0,
    "timelimit": 0,
    "samelevel": 1,
    "pre_map_commands": "",
    "post_map_commands": "impulse 100\nimpulse 100\nimpulse 100\nimpulse 100\nimpulse 100",
}

# ---------------------------------------------------------------------------
# Tunable hyperparameters — loaded from JSON so the orchestrator can evolve
# them without patching Python source.
# ---------------------------------------------------------------------------

def _load_hyperparams(path: Path | None = None) -> dict[str, Any]:
    return dict(read_json(path or _HYPERPARAMS_PATH_DEFAULT))


_HYPERPARAMS = _load_hyperparams()

# Structural constants that never change between experiments.
_BC_STRUCTURAL: Dict[str, Any] = {
    "output_dir": BC_OUTPUT_DIR,
    "train_ratio": 0.72, "val_ratio": 0.14, "lr": 0.008,
    "use_gru": True, "gru_hidden": 64, "trunk_hidden": 192,
    "class_weight_power": 0.6, "class_weight_min": 0.5, "class_weight_max": 2.5,
    "device": "auto",
}
_PPO_STRUCTURAL: Dict[str, Any] = {
    "mode": "pvp", "native_executable": "assets/bin/quake_worker",
    "device": "auto",
    # Architecture must match BC checkpoint — inherit BC defaults.
    "trunk_hidden": 192, "gru_hidden": 64, "use_gru": True,
}
_EVAL_STRUCTURAL: Dict[str, Any] = {
    "mode": "pvp", "native_executable": "assets/bin/quake_worker",
    "policy_modes": ["greedy", "sampled"],
    "start_mode": "randomized", "holdout_seed_offset": 10000,
    "sample_seed_offset": 20000, "device": "auto",
}

# Merged views used by build_bc_config / load_config_with_runtime.
_BC_DEFAULTS: Dict[str, Any] = {**_BC_STRUCTURAL}
_PPO_DEFAULTS: Dict[str, Any] = {**_PPO_STRUCTURAL, **_HYPERPARAMS.get("ppo", {})}
_EVAL_DEFAULTS: Dict[str, Any] = {**_EVAL_STRUCTURAL, **_HYPERPARAMS.get("eval", {})}
_SCALE: Dict[str, Dict[str, Dict[str, Any]]] = _HYPERPARAMS.get("scale", {})


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
    output_root_key = f"{runtime_scale}_output_root"
    output_root = str(
        scenario.get(output_root_key)
        or f"assets/runs/competitive_bot_{scenario_id.replace('-', '_')}_{runtime_scale}"
    )
    map_id = str(scenario["map_id"])
    options = _deep_merge_config(_DEFAULT_BOT_OPTIONS, scenario.get("options", {}))
    use_gru = bool(scenario.get("use_gru", True))
    gru_hidden = int(scenario.get("gru_hidden", 64))

    return LiveProfile(
        name=name,
        default_demo_dir=default_demo_dir,
        auto_demo_dirs=auto_demo_dirs,
        plan_path=f"{output_root}/live_training_plan.json",
        runtime_scale="verify" if runtime_scale == "verify" else "live",
        profile_group="combat_bot_ladder",
        profile_note=str(scenario.get("skill_note", "")).strip(),
        scenario_id=scenario_id,
        retained_role=str(scenario.get("retained_role", "")).strip(),
        comparison_profiles=comparison_profiles,
        ppo_overrides={
            "map_id": map_id,
            "output_dir": output_root,
            "native_args": list(_DEFAULT_BOT_NATIVE_ARGS),
            "options": options,
            "use_gru": use_gru,
            "gru_hidden": gru_hidden,
        },
        eval_overrides={
            "map_id": map_id,
            "checkpoint_path": f"{output_root}/best/best_model.pth",
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
            default_demo_dir="assets/demos",
            auto_demo_dirs=("assets/demos", "tests/demo_data"),
            scenario=scenario,
            comparison_profiles=comparison,
        )
        profiles[live_name] = _build_scenario_profile(
            name=live_name,
            runtime_scale="live",
            default_demo_dir="assets/demos",
            auto_demo_dirs=("assets/demos", "tests/demo_data"),
            scenario=scenario,
            comparison_profiles=comparison,
        )
    return profiles


def _build_multi_scenario_profile(name: str, runtime_scale: str) -> LiveProfile:
    is_verify = runtime_scale == "verify"
    output_root = "assets/runs/competitive_bot_multi_verify" if is_verify else "assets/runs/live"
    return LiveProfile(
        name=name,
        default_demo_dir="assets/demos",
        auto_demo_dirs=("assets/demos", "tests/demo_data"),
        plan_path=f"{output_root}/live_training_plan.json",
        runtime_scale="verify" if is_verify else "live",
        profile_group="combat_bot_multi",
        profile_note="Multi-scenario PvP training across the full bot ladder.",
        scenario_id="multi",
        ppo_overrides={
            "output_dir": output_root,
            "scenario_config_path": _SCENARIOS_PATH_DEFAULT,
        },
        eval_overrides={
            "checkpoint_path": f"{output_root}/best/best_model.pth",
            "output_dir": f"{output_root}/eval",
            "scenario_config_path": _SCENARIOS_PATH_DEFAULT,
        },
    )


PROFILES: dict[str, LiveProfile] = {
    "combat-bot-multi-verify": _build_multi_scenario_profile("combat-bot-multi-verify", "verify"),
    "combat-bot-multi": _build_multi_scenario_profile("combat-bot-multi", "live"),
}
PROFILES.update(_load_bot_ladder_profiles())


def build_bc_config(runtime_scale: str, requested_device: str) -> dict[str, Any]:
    """Build BC config from defaults + scale. Not profile-dependent."""
    scale = _SCALE[runtime_scale]
    cfg: dict[str, Any] = {**_BC_DEFAULTS, **scale["bc"]}
    cfg["device"] = requested_device
    return cfg


def load_config_with_runtime(
    profile: LiveProfile,
    plan: RuntimePlan,
    requested_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build PPO and eval configs from profile + runtime plan."""
    scale = _SCALE[profile.runtime_scale]
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

    ppo_cfg["device"] = requested_device
    eval_cfg["device"] = requested_device

    # Single source of truth: plan.num_envs is the hardware-derived worker
    # count.  Training uses it directly; eval splits it across policy modes
    # (greedy + sampled run in parallel) and caps each mode at num_episodes.
    num_envs = int(ppo_cfg.get("num_envs", 0)) or plan.num_envs
    ppo_cfg["num_envs"] = num_envs
    ppo_cfg["rollout_steps"] = int(ppo_cfg.get("rollout_steps", 0)) or plan.rollout_steps
    ppo_cfg["total_steps"] = max(int(ppo_cfg.get("total_steps", 1)), plan.total_steps)
    ppo_cfg["minibatch_size"] = int(ppo_cfg.get("minibatch_size", 0)) or plan.minibatch_size
    num_eval_modes = max(1, len(eval_cfg.get("policy_modes", ["greedy"])))
    eval_envs_per_mode = num_envs // num_eval_modes
    eval_cfg["num_envs"] = eval_envs_per_mode
    eval_cfg["num_episodes"] = eval_envs_per_mode

    return ppo_cfg, eval_cfg
