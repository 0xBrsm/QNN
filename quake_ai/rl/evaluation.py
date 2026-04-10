"""Policy evaluation and reporting."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping as MappingABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.rl.combat_metrics import iter_weapon_metric_keys
from quake_ai.rl.metrics import (
    EpisodeStatAccumulator,
    append_metric_values,
    build_eval_summary_aliases,
    mean_metric_values,
)
from quake_ai.model.policy import QNNPolicy
from quake_ai.model.observation import SELF_SCALAR_DIM
from quake_ai.rl.environment import NativeWorldEnv
from quake_ai.utils.io import trusted_torch_load, write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest
from mapgen.pool import PROCGEN_SENTINEL


@dataclass(slots=True)
class EvalConfig:
    checkpoint_path: str
    output_dir: str
    map_id: str
    native_executable: str
    native_workdir: str
    fixed_tick_hz: int
    native_env: Dict[str, str]
    native_args: List[str]
    options: Dict[str, object]
    mode: str
    seed: int
    num_episodes: int
    num_envs: int
    max_steps_per_episode: int
    policy_modes: List[str]
    start_mode: str
    holdout_seed_offset: int
    sample_seed_offset: int
    map_features_path: str
    procgen: Dict[str, object] | None
    scenario_config_path: str
    reward_json_path: str
    record_demos: bool
    parallel_policy_modes: bool
    device: str


@dataclass(frozen=True, slots=True)
class _ScenarioSpec:
    scenario_id: str
    map_id: str
    native_args: tuple[str, ...]
    options: Dict[str, object]
    procgen_cfg: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _EpisodeJob:
    episode_index: int
    episode_seed: int
    start_variant: int | None
    scenario: _ScenarioSpec


def _episode_specs(config: EvalConfig) -> List[Tuple[int, int | None]]:
    if config.start_mode == "sequential":
        return [(config.seed + episode, None) for episode in range(config.num_episodes)]
    if config.start_mode == "randomized":
        rng = np.random.default_rng(config.seed + config.holdout_seed_offset)
        return [(int(rng.integers(0, 2**31 - 1)), None) for _ in range(config.num_episodes)]
    raise ValueError(f"Unsupported start_mode {config.start_mode}")


@dataclass(slots=True)
class _EpisodeState:
    episode_index: int
    obs: np.ndarray | Dict[str, np.ndarray]
    scenario_id: str
    step_count: int = 0
    return_value: float = 0.0
    last_info: Mapping[str, object] = field(default_factory=dict)
    rng: torch.Generator | None = None
    hidden: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    metrics: EpisodeStatAccumulator = field(default_factory=EpisodeStatAccumulator)


def _stack_obs(obs_list: Sequence[np.ndarray | Dict[str, np.ndarray]]) -> np.ndarray | Dict[str, np.ndarray]:
    first = obs_list[0]
    if isinstance(first, MappingABC):
        return {
            key: np.stack([obs[key] for obs in obs_list], axis=0)
            for key in first
        }
    return np.stack(obs_list, axis=0)


def _scenario_entries(config: EvalConfig) -> list[Dict[str, Any]]:
    if not config.scenario_config_path:
        return []
    payload = json.loads(Path(config.scenario_config_path).read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, list):
        raise RuntimeError(f"scenario_config_path must define a scenarios list: {config.scenario_config_path}")
    return [dict(scenario) for scenario in scenarios if isinstance(scenario, MappingABC)]


def _scenario_spec(
    config: EvalConfig,
    scenario: Mapping[str, Any] | None,
) -> _ScenarioSpec:
    map_id = str(scenario["map_id"]) if scenario is not None else str(config.map_id)
    native_args = list(scenario["native_args"]) if scenario is not None and "native_args" in scenario else list(config.native_args)
    merged_options = dict(config.options)
    if scenario is not None and "options" in scenario:
        merged_options.update(dict(scenario["options"]))
    scenario_id = str(scenario["scenario_id"]) if scenario is not None else map_id
    procgen_cfg: dict[str, object] | None = None
    if map_id == PROCGEN_SENTINEL:
        basedir = str(config.native_env.get("QUAKE_BASEDIR", "")).strip()
        if not basedir:
            raise RuntimeError("Evaluation procgen requires native_env['QUAKE_BASEDIR']")
        procgen_opts_source = scenario["procgen"] if scenario is not None and "procgen" in scenario else config.procgen
        if not isinstance(procgen_opts_source, Mapping):
            raise RuntimeError("Procgen evaluation requires an explicit procgen config with arena_size, rooms, and cleanup_generated_maps")
        procgen_opts = dict(procgen_opts_source)
        procgen_cfg = {
            "maps_dir": str(Path(basedir) / "id1" / "maps"),
            "arena_size": int(procgen_opts["arena_size"]),
            "rooms": int(procgen_opts["rooms"]),
            "cleanup_generated_maps": bool(procgen_opts["cleanup_generated_maps"]),
        }
    return _ScenarioSpec(
        scenario_id=scenario_id,
        map_id=map_id,
        native_args=tuple(str(value) for value in native_args),
        options=merged_options,
        procgen_cfg=procgen_cfg,
    )


def _scenario_specs(config: EvalConfig) -> list[_ScenarioSpec]:
    scenarios = _scenario_entries(config)
    if not scenarios:
        return [_scenario_spec(config, None)]
    return [_scenario_spec(config, scenario) for scenario in scenarios]


def _episode_jobs(config: EvalConfig) -> tuple[list[_ScenarioSpec], dict[str, deque[_EpisodeJob]]]:
    scenarios = _scenario_specs(config)
    jobs: dict[str, deque[_EpisodeJob]] = {scenario.scenario_id: deque() for scenario in scenarios}
    for episode_index, (episode_seed, start_variant) in enumerate(_episode_specs(config)):
        scenario = scenarios[episode_index % len(scenarios)]
        jobs[scenario.scenario_id].append(
            _EpisodeJob(
                episode_index=episode_index,
                episode_seed=episode_seed,
                start_variant=start_variant,
                scenario=scenario,
            )
        )
    return scenarios, jobs


def _build_eval_env(config: EvalConfig, scenario: _ScenarioSpec) -> NativeWorldEnv:
    if not config.native_executable:
        raise RuntimeError("Evaluation requires native_executable")
    from quake_ai.rl.reward import RewardWeights

    return NativeWorldEnv(
        executable=config.native_executable,
        map_id=scenario.map_id,
        max_steps=config.max_steps_per_episode,
        fixed_tick_hz=config.fixed_tick_hz,
        reward_weights=RewardWeights.from_json(config.reward_json_path),
        mode=config.mode,
        seed=config.seed,
        workdir=config.native_workdir or None,
        env=config.native_env,
        native_args=list(scenario.native_args),
        options=scenario.options,
        procgen=scenario.procgen_cfg,
    )


def _set_demo_recording(env: NativeWorldEnv, episode_index: int, mode: str) -> None:
    """Inject a ``record`` command into the env's pre_map_commands for this episode.

    Quake's ``record`` console command only works when the client is **not**
    yet connected to a server.  The worker's reset sequence runs
    ``pre_map_commands`` before ``map <name>``, so the client is still
    disconnected and ``record`` succeeds.  ``post_map_commands`` runs after
    connection, which causes ``record`` to be silently rejected.
    """
    demo_name = f"eval_{mode}_ep_{episode_index:04d}"
    base_cmds = env.adapter.reset_options.get("_base_pre_map_commands")
    if base_cmds is None:
        base_cmds = str(env.adapter.reset_options.get("pre_map_commands", ""))
        env.adapter.reset_options["_base_pre_map_commands"] = base_cmds
    record_cmd = f"record {demo_name}"
    env.adapter.reset_options["pre_map_commands"] = (
        f"{base_cmds}\n{record_cmd}" if base_cmds else record_cmd
    )


def _episode_rng(config: EvalConfig, mode: str, episode_index: int, device: torch.device) -> torch.Generator:
    offset = 0 if mode == "greedy" else config.sample_seed_offset
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.seed + offset + episode_index)
    return generator


def _select_actions_batch(
    model: QNNPolicy,
    obs_batch: np.ndarray | Dict[str, np.ndarray],
    mode: str,
    states: Sequence[_EpisodeState],
) -> tuple[List[Mapping[str, object]], np.ndarray]:
    hidden_batch = np.stack([state.hidden for state in states], axis=0) if states else model.zero_hidden(0)
    if mode == "greedy":
        action_batch = model.act(obs_batch, mode=mode, hidden=hidden_batch)
    elif mode == "sampled":
        row_generators = [state.rng for state in states]
        if any(generator is None for generator in row_generators):
            raise RuntimeError("Sampled evaluation requires a persistent per-episode RNG")
        action_batch = model.act(obs_batch, mode=mode, hidden=hidden_batch, row_generators=row_generators)
    else:
        raise ValueError(f"Unsupported policy mode {mode}")

    batch_size = next(iter(obs_batch.values())).shape[0] if isinstance(obs_batch, dict) else obs_batch.shape[0]
    actions = [
        (
            {
                "move": action_batch.actions["move"][idx].astype(np.float32, copy=False).tolist(),
                "look": action_batch.actions["look"][idx].astype(np.float32, copy=False).tolist(),
                **{
                    head: int(action_batch.actions[head][idx])
                    for head in ACTION_HEADS
                    if head not in {"move", "look"}
                },
            }
        )
        for idx in range(batch_size)
    ]
    next_hidden = action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)
    return actions, next_hidden


def _step_env(
    env: NativeWorldEnv,
    action: Mapping[str, int],
) -> Tuple[np.ndarray | Dict[str, np.ndarray], float, bool, Dict[str, object]]:
    obs, reward, done, info = env.step(action)
    return obs, reward, done, dict(info)


_AUX_INFO_KEYS = (
    "frag_delta",
    "frag_loss",
    "monster_kill_delta",
    "damage_taken",
    "damage_dealt",
    "hit_count",
    "shots_fired",
    "health_gain",
    "armor_gain",
    "ammo_gain",
    "weapon_pickups",
    "weapon_switches",
    "visible_threats",
    "tracking_cos",
    "fire_pressed",
    "effective_fire",
    "blind_fire",
    "health_fraction",
    "armor_fraction",
    "reward_tracking",
)
_EPISODE_AUX_KEYS = (
    "episode_damage_dealt",
    "episode_hit_count",
    "episode_shots_fired",
)
_WEAPON_AUX_KEYS = tuple(iter_weapon_metric_keys())
_EPISODE_WEAPON_AUX_KEYS = tuple(
    iter_weapon_metric_keys(prefixes=(f"episode_{prefix}" for prefix in ("weapon_damage_dealt", "weapon_hits_landed", "weapon_shots_fired")))
)


def _iter_aux_metric_items(info: Mapping[str, object], keys: tuple[str, ...]) -> List[tuple[str, float]]:
    pairs: List[tuple[str, float]] = []
    for key in keys:
        value = info.get(key)
        if isinstance(value, (int, float)):
            pairs.append((key, float(value)))
    return pairs


def _evaluate_mode(
    config: EvalConfig,
    model: QNNPolicy,
    mode: str,
    episode_specs: Sequence[Tuple[int, int | None]],
) -> Dict[str, float]:
    del episode_specs
    num_envs = max(1, min(config.num_envs, config.num_episodes))
    scenarios, job_queues = _episode_jobs(config)
    executor = ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="nq-eval") if num_envs > 1 else None

    envs: Dict[int, NativeWorldEnv] = {}
    slot_scenarios: Dict[int, _ScenarioSpec] = {}
    returns: List[float] = []
    end_health: List[int] = []
    end_armor: List[int] = []
    done_reasons: Dict[str, int] = {}
    aux_metric_sums: Dict[str, float] = {}
    episode_metric_values: Dict[str, List[float]] = {}
    stuck_steps = 0
    total_steps = 0
    checked_obs_dim = False
    scenario_done_reasons: Dict[str, Dict[str, int]] = {}
    scenario_episode_counts: Dict[str, int] = {}
    scenario_returns: Dict[str, List[float]] = {}
    scenario_stuck_steps: Dict[str, int] = {}
    scenario_total_steps: Dict[str, int] = {}
    scenario_aux_metric_sums: Dict[str, Dict[str, float]] = {}
    scenario_episode_metric_values: Dict[str, Dict[str, List[float]]] = {}

    def _next_job(preferred_scenario_id: str | None = None) -> _EpisodeJob | None:
        if preferred_scenario_id:
            preferred = job_queues.get(preferred_scenario_id)
            if preferred:
                return preferred.popleft()
        for scenario in scenarios:
            queue = job_queues.get(scenario.scenario_id)
            if queue:
                return queue.popleft()
        return None

    def _ensure_env(slot: int, scenario: _ScenarioSpec) -> NativeWorldEnv:
        current = slot_scenarios.get(slot)
        if current is not None and current.scenario_id == scenario.scenario_id:
            return envs[slot]
        existing = envs.pop(slot, None)
        if existing is not None:
            existing.close()
        env = _build_eval_env(config, scenario)
        envs[slot] = env
        slot_scenarios[slot] = scenario
        return env

    try:
        active: Dict[int, _EpisodeState] = {}
        for slot in range(num_envs):
            job = _next_job()
            if job is None:
                break
            env = _ensure_env(slot, job.scenario)
            if config.record_demos:
                _set_demo_recording(env, job.episode_index, mode)
            obs = env.reset(seed=job.episode_seed, start_variant=job.start_variant)
            if not checked_obs_dim:
                # For transformer models, obs_dim is the self_scalars dimension
                # (e.g. 23), not the full flattened observation space.  Skip
                # the dimension check for dict (token) observations since the
                # transformer handles variable-length token sequences natively.
                if not isinstance(obs, dict):
                    env_obs_dim = int(obs.shape[0])
                    if env_obs_dim != model.obs_dim:
                        raise RuntimeError(
                            f"Evaluation checkpoint obs_dim={model.obs_dim} does not match environment obs_dim={env_obs_dim}"
                        )
                checked_obs_dim = True
            active[len(active)] = _EpisodeState(
                episode_index=job.episode_index,
                obs=obs,
                scenario_id=job.scenario.scenario_id,
                rng=None if mode == "greedy" else _episode_rng(config, mode, job.episode_index, model.device),
                hidden=model.zero_hidden(1)[0].copy(),
            )

        while active:
            slot_ids = sorted(active.keys())
            states = [active[slot] for slot in slot_ids]
            obs_batch = _stack_obs([state.obs for state in states])
            actions, next_hidden = _select_actions_batch(
                model=model,
                obs_batch=obs_batch,
                mode=mode,
                states=states,
            )

            if executor is None:
                results = [_step_env(envs[slot], action) for slot, action in zip(slot_ids, actions)]
            else:
                futures = [
                    executor.submit(_step_env, envs[slot], action)
                    for slot, action in zip(slot_ids, actions)
                ]
                results = [future.result() for future in futures]

            for batch_idx, (slot, result) in enumerate(zip(slot_ids, results)):
                obs, reward, done, info = result
                state = active[slot]
                state.hidden = next_hidden[batch_idx].copy()
                state.step_count += 1
                terminal = bool(done or state.step_count >= config.max_steps_per_episode)
                # Reward remains useful even when the episode ends without a special terminal condition.
                state.return_value += float(reward)
                state.last_info = info
                state.metrics.add_step(reward=float(reward), info=info, terminal=terminal)
                total_steps += 1
                scenario_id = str(info.get("scenario_id", state.scenario_id))
                state.scenario_id = scenario_id
                scenario_total_steps[scenario_id] = scenario_total_steps.get(scenario_id, 0) + 1
                if bool(info.get("stuck", False)):
                    stuck_steps += 1
                    scenario_stuck_steps[scenario_id] = scenario_stuck_steps.get(scenario_id, 0) + 1
                for key, value in _iter_aux_metric_items(info, _AUX_INFO_KEYS + _WEAPON_AUX_KEYS):
                    aux_metric_sums[key] = aux_metric_sums.get(key, 0.0) + value
                    scenario_metric_sums = scenario_aux_metric_sums.setdefault(scenario_id, {})
                    scenario_metric_sums[key] = scenario_metric_sums.get(key, 0.0) + value

                if terminal:
                    returns.append(float(state.return_value))
                    end_health.append(int(state.last_info.get("health", 0)))
                    end_armor.append(int(state.last_info.get("armor", 0)))
                    done_reason = str(state.last_info.get("done_reason", "")).strip() or "unknown"
                    done_reasons[done_reason] = done_reasons.get(done_reason, 0) + 1
                    scenario_done_reason_counts = scenario_done_reasons.setdefault(state.scenario_id, {})
                    scenario_done_reason_counts[done_reason] = scenario_done_reason_counts.get(done_reason, 0) + 1
                    scenario_episode_counts[state.scenario_id] = scenario_episode_counts.get(state.scenario_id, 0) + 1
                    scenario_returns.setdefault(state.scenario_id, []).append(float(state.return_value))
                    episode_stats = state.metrics.as_dict()
                    append_metric_values(episode_metric_values, episode_stats)
                    append_metric_values(
                        scenario_episode_metric_values.setdefault(state.scenario_id, {}),
                        episode_stats,
                    )
                    for key, value in _iter_aux_metric_items(
                        state.last_info,
                        _EPISODE_AUX_KEYS + _EPISODE_WEAPON_AUX_KEYS,
                    ):
                        episode_metric_values.setdefault(key, []).append(value)
                        scenario_episode_metric_values.setdefault(state.scenario_id, {}).setdefault(key, []).append(value)

                    next_job = _next_job(slot_scenarios[slot].scenario_id)
                    if next_job is not None:
                        env = _ensure_env(slot, next_job.scenario)
                        if config.record_demos:
                            _set_demo_recording(env, next_job.episode_index, mode)
                        next_obs = env.reset(seed=next_job.episode_seed, start_variant=next_job.start_variant)
                        active[slot] = _EpisodeState(
                            episode_index=next_job.episode_index,
                            obs=next_obs,
                            scenario_id=next_job.scenario.scenario_id,
                            rng=None if mode == "greedy" else _episode_rng(config, mode, next_job.episode_index, model.device),
                            hidden=model.zero_hidden(1)[0].copy(),
                        )
                    else:
                        del active[slot]
                else:
                    state.obs = obs
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in envs.values():
            env.close()

    stuck_rate = float(stuck_steps / max(total_steps, 1))
    episode_metric_means = mean_metric_values(episode_metric_values)

    summary = {
        "death_rate": float(done_reasons.get("player_died", 0) / max(config.num_episodes, 1)),
        "mean_episode_return": float(np.mean(returns)) if returns else 0.0,
        "mean_end_health": float(np.mean(end_health)) if end_health else 0.0,
        "mean_end_armor": float(np.mean(end_armor)) if end_armor else 0.0,
        "stuck_rate": stuck_rate,
        "num_episodes": config.num_episodes,
    }
    for metric_key in _AUX_INFO_KEYS + _WEAPON_AUX_KEYS:
        summary[f"{metric_key}_mean"] = float(aux_metric_sums.get(metric_key, 0.0) / max(total_steps, 1))
    summary["episode_metric_means"] = episode_metric_means
    summary.update(build_eval_summary_aliases(episode_metric_means))
    summary["scenario_metric_means"] = {
        scenario_id: {
            "num_episodes": scenario_episode_counts.get(scenario_id, 0),
            "mean_episode_return": float(np.mean(values)) if values else 0.0,
            "death_rate": float(scenario_done_reasons.get(scenario_id, {}).get("player_died", 0) / max(scenario_episode_counts.get(scenario_id, 0), 1)),
            "stuck_rate": float(scenario_stuck_steps.get(scenario_id, 0) / max(scenario_total_steps.get(scenario_id, 0), 1)),
            **{
                f"{metric_key}_mean": float(
                    scenario_aux_metric_sums.get(scenario_id, {}).get(metric_key, 0.0) / max(scenario_total_steps.get(scenario_id, 0), 1)
                )
                for metric_key in ("frag_delta", "damage_dealt", "hit_count", "shots_fired")
            },
            **build_eval_summary_aliases(mean_metric_values(scenario_episode_metric_values.get(scenario_id, {}))),
        }
        for scenario_id, values in sorted(scenario_returns.items())
    }
    return summary


def _is_sf_checkpoint(path: str | Path) -> bool:
    """Return True if *path* is a Sample Factory format checkpoint (.pth with 'model' key)."""
    p = Path(path)
    if p.suffix != ".pth" or not p.exists():
        return False
    try:
        payload = trusted_torch_load(str(p), map_location="cpu")
        return isinstance(payload, dict) and "model" in payload and ("train_step" in payload or "env_steps" in payload)
    except Exception:
        return False


def _load_sf_checkpoint_as_qnn(
    path: str | Path,
    device: str = "cpu",
    model_config: Dict[str, Any] | None = None,
) -> QNNPolicy:
    """Convert an SF checkpoint to a QNNPolicy in-memory (no temp files).

    Architecture metadata is read from a sidecar JSON if available, otherwise
    from the caller-supplied ``model_config`` (typically model.json from the
    run dir).
    """
    from quake_ai.ppo.checkpoint_converter import sf_to_qnn
    from quake_ai.model.observation import OBS_DIM

    p = Path(path)
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise RuntimeError(f"SF checkpoint sidecar must be a JSON object: {sidecar}")
    elif model_config is not None:
        meta = dict(model_config)
        meta.setdefault("obs_dim", OBS_DIM)
    else:
        raise RuntimeError(
            f"SF checkpoint requires either a sidecar JSON ({sidecar}) or model_config"
        )

    required_keys = (
        "obs_dim",
        "trunk_hidden",
        "gru_hidden",
        "use_gru",
        "d_model",
        "n_heads",
        "n_layers",
        "ffn_dim",
        "action_history_tokens",
        "attn_dropout",
        "readout",
    )
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise RuntimeError(
            f"SF checkpoint sidecar is missing architecture fields ({', '.join(missing)}): {sidecar}"
        )

    return sf_to_qnn(
        sf_checkpoint_path=p,
        obs_dim=int(meta["obs_dim"]),
        trunk_hidden=int(meta["trunk_hidden"]),
        gru_hidden=int(meta["gru_hidden"]),
        use_gru=bool(meta["use_gru"]),
        device=device,
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        ffn_dim=int(meta["ffn_dim"]),
        action_history_tokens=int(meta["action_history_tokens"]),
        attn_dropout=float(meta["attn_dropout"]),
        readout=str(meta["readout"]),
    )


def _load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    model_config: Dict[str, Any] | None = None,
) -> QNNPolicy:
    """Load a checkpoint in either QNN (.pth) or SF (.pth) format."""
    if _is_sf_checkpoint(path):
        return _load_sf_checkpoint_as_qnn(path, device=device, model_config=model_config)
    return QNNPolicy.load(str(path), device=device)


def _evaluate_policy_mode(
    config: EvalConfig,
    mode: str,
    model_config: Dict[str, Any] | None = None,
) -> tuple[str, Dict[str, float], QNNPolicy]:
    model = _load_checkpoint(config.checkpoint_path, device=config.device, model_config=model_config)
    summary = _evaluate_mode(config, model, mode, _episode_specs(config))
    return mode, summary, model


def run_evaluation(
    config: EvalConfig,
    model_config: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    set_global_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mode_summaries: Dict[str, Dict[str, float]] = {}
    model_card_model: QNNPolicy | None = None
    if len(config.policy_modes) > 1 and config.parallel_policy_modes:
        with ThreadPoolExecutor(max_workers=len(config.policy_modes), thread_name_prefix="nq-eval-mode") as executor:
            futures = {
                mode: executor.submit(_evaluate_policy_mode, config, mode, model_config)
                for mode in config.policy_modes
            }
            for mode in config.policy_modes:
                _, summary, mode_model = futures[mode].result()
                mode_summaries[mode] = summary
                if model_card_model is None:
                    model_card_model = mode_model
    else:
        for mode in config.policy_modes:
            _, summary, mode_model = _evaluate_policy_mode(config, mode, model_config)
            mode_summaries[mode] = summary
            if model_card_model is None:
                model_card_model = mode_model
    if model_card_model is None:
        raise RuntimeError("Evaluation did not produce a model instance")

    if len(config.policy_modes) == 1:
        summary: Dict[str, object] = dict(mode_summaries[config.policy_modes[0]])
    else:
        summary = {
            "start_mode": config.start_mode,
            "policy_modes": list(config.policy_modes),
            "modes": mode_summaries,
        }

    write_json(output / "eval_summary.json", summary)
    write_experiment_manifest(output / "eval_manifest.json", asdict(config), summary)

    eval_notes = []
    eval_notes.append("Evaluation uses the native worker token path with object, event, and spatial observations.")
    eval_notes.append("Reward shaping and episode metrics remain in Python while the worker supplies deterministic token ticks.")
    if config.start_mode == "randomized":
        eval_notes.append("Evaluation uses held-out randomized reset seeds.")
    else:
        eval_notes.append("Evaluation uses fixed sequential reset seeds for regression tracking.")
    if set(config.policy_modes) == {"greedy"}:
        eval_notes.append("Evaluation uses greedy actions only.")
    elif set(config.policy_modes) == {"sampled"}:
        eval_notes.append("Evaluation uses stochastic action sampling only.")
    else:
        eval_notes.append("Evaluation reports both greedy and stochastic action-selection modes.")
    if config.num_envs > 1:
        eval_notes.append(f"Evaluation parallelizes episodes across {config.num_envs} environments.")
    if len(config.policy_modes) > 1 and config.parallel_policy_modes:
        eval_notes.append("Evaluation runs policy modes in parallel with isolated model instances.")

    model_card = {
        "model": {
            "checkpoint": str(config.checkpoint_path),
            "architecture": (
                f"transformer encoder + GRU({model_card_model.gru_hidden}) actor-critic"
                if model_card_model.use_gru
                else "transformer encoder actor-critic"
            ),
            "observation_modality": "token dict observation with self/object/event/spatial tensors",
            "action_space": list(ACTION_HEADS.keys()),
        },
        "evaluation": summary,
        "notes": eval_notes,
    }
    write_json(output / "model_card.json", model_card)

    if len(config.policy_modes) == 1:
        return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}

    flattened: Dict[str, float] = {}
    for mode, metrics in mode_summaries.items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flattened[f"{mode}_{key}"] = float(value)
    return flattened


# ---------------------------------------------------------------------------
# CLI entry point (python -m quake_ai.rl.evaluation)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from quake_ai.rl.run_config import (
        load_run_config,
        build_run_eval_config,
        run_output_dirs,
        _require_mapping,
        _require_string,
    )
    from quake_ai.rl.planning import _resolve_asset_root, _validate_native_mod_assets

    parser = argparse.ArgumentParser(description="Multi-episode evaluation of a checkpoint")
    parser.add_argument("run_dir", type=Path, help="Run directory containing run.json")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path relative to run's checkpoints dir")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override eval_num_episodes from config")
    parser.add_argument("--num-envs", type=int, default=None, help="Override eval_num_envs from config")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args()

    run_cfg = load_run_config(args.run_dir.resolve())
    machine = _require_mapping(run_cfg, "machine", "run config")
    model_config = _require_mapping(run_cfg, "model", "run config")

    outputs = run_output_dirs(run_cfg)
    checkpoint_path = outputs["checkpoints"] / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    asset_root = _resolve_asset_root(_require_string(machine, "asset_root", "machine.json"))
    worker_path = Path(run_cfg["run_dir"]).parent.parent / _require_string(machine, "worker_binary", "machine.json")
    if not worker_path.exists():
        worker_path = Path.cwd() / _require_string(machine, "worker_binary", "machine.json")
    if not worker_path.exists():
        raise FileNotFoundError(f"Worker binary not found: {worker_path}")

    native_args = _require_mapping(run_cfg, "scenario", "run config").get("native_args", [])
    _validate_native_mod_assets(asset_root, native_args)

    run_cfg["checkpoint_path"] = str(checkpoint_path)
    eval_cfg = build_run_eval_config(run_cfg, args.device)
    eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(asset_root)}
    eval_cfg["native_executable"] = str(worker_path)

    if args.num_episodes is not None:
        eval_cfg["num_episodes"] = args.num_episodes
    if args.num_envs is not None:
        eval_cfg["num_envs"] = args.num_envs

    config = EvalConfig(**eval_cfg)
    print(f"Run:        {run_cfg['run_dir']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes:   {config.num_episodes}")
    print(f"Envs:       {config.num_envs}")
    print(f"Modes:      {config.policy_modes}")
    print(f"Device:     {config.device}")
    print()

    results = run_evaluation(config, model_config=model_config)

    print()
    print("=== Evaluation Summary ===")
    for key in ("mean_episode_return", "death_rate", "stuck_rate",
                "episode_frag_delta_mean", "episode_damage_dealt_mean",
                "episode_hit_count_mean", "episode_shots_fired_mean"):
        if key in results:
            print(f"  {key}: {results[key]:.3f}")

    output_dir = Path(eval_cfg["output_dir"])
    print(f"\n  Results written to: {output_dir}")


if __name__ == "__main__":
    main()
