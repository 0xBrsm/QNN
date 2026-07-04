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

from qnn.actions import ACTION_HEADS
from qnn.utils.combat_metrics import iter_weapon_metric_keys
from qnn.run.metrics import (
    EpisodeStatAccumulator,
    append_metric_values,
    build_eval_summary_aliases,
    mean_metric_values,
)
from qnn.model.policy import QNNPolicy
from qnn.schema import SELF_SCALAR_DIM
from qnn.vocab import TOKEN_ACTOR
from qnn.env.world import NativeWorldEnv
from qnn.utils.io import trusted_torch_load, write_json
from qnn.utils.repro import set_global_seed, write_experiment_manifest
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
    parallel_policy_modes: bool
    device: str
    # Look aim-prior gain override for the sampled decode. None → the baked
    # decode-contract default (qnn.model.bench.a24.lead_aim.AIM_PRIOR_GAIN); 0.0 → prior
    # disabled (the matched-code control arm). Set via the run config's
    # optional eval_look_aim_prior_gain so the arm is recorded in the frozen
    # run dir rather than in launch-time state.
    look_aim_prior_gain: float | None = None
    # Per-model weapon BAN (csv impulses 1..8) — the SAME spec the ONNX export
    # path applies (tools/export_onnx --weapon-ban / decode.weapon_ban). Set so
    # offline eval predicts deployed behavior: banned classes can never be
    # decided and a held banned weapon force-switches. Empty () → no ban
    # (historical behavior). Set via the run config's optional eval_weapon_ban
    # so the arm is recorded in the frozen run dir.
    weapon_ban: tuple[int, ...] = ()
    # Engine-parity sticky move decode (fb/lr argmax + hold-unless-conf≥τ —
    # what the live bin applies from stamped metadata). None on either axis →
    # legacy per-frame sampling (every eval before 2026-06-11; NOT what live
    # play does). Set both via eval_move_sticky_tau_fb / _lr for live parity.
    move_sticky_tau_fb: float | None = None
    move_sticky_tau_lr: float | None = None
    # Semi-Markov hazard decode tables (eval_move_hazard): mapping with
    # "edges" (ascending dwell-age bucket edges) and per-axis "fb"/"lr"/"ud"
    # release-probability lists (len = len(edges)+1). fb/lr supplement the
    # sticky gate; ud replaces per-frame sampling. None → no hazard decode.
    move_hazard: Dict[str, Any] | None = None
    # Dump per-episode decoded move-class streams (fb,lr,ud per tick) to
    # output_dir/move_streams_<mode>.npz — the live-side input of the
    # move dwell/switch diagnostics. eval_log_action_streams.
    log_action_streams: bool = False
    # Latency-agnostic switch-back suppression (eval_move_switchback_eps):
    # after the decode switches an axis away from class c, gate-driven
    # switch-back to c is suppressed while softmax(c) stays within ±eps of
    # its value at the switch tick (stale obs ⇒ frozen conf; evidence in
    # either direction clears the watermark). Hazard releases stay exempt.
    # None → no suppression (every eval before 2026-06-11).
    move_switchback_eps: float | None = None
    # Stop-onset hazard symmetry (eval_move_stop_onset): from a true stop
    # (both held fb/lr = none) gate-driven presses are suppressed; movement
    # onsets come from the none-row hazard (human WHEN-law, model WHAT).
    # Requires move_hazard fb+lr tables. False → gate presses as before.
    move_stop_onset: bool = False
    # Engagement-gated sticky tau (eval_move_tau_engagement_gated): tau=1
    # (table-only switching) on disengaged frames, sticky_tau on engaged frames.
    # Pairs with a non-combat baseline move_hazard table (rc1o move scheme).
    move_tau_engagement_gated: bool = False
    # Jump (ud) sampled attack-style impulse decode (eval_move_jump_sample): replace
    # the legacy ud argmax with a per-tick Bernoulli on sigmoid((pos−none+bias)/temp).
    move_jump_sample: bool = False
    move_jump_bias: float = 0.0
    move_jump_temp: float = 1.0
    # Live-client latency emulation (eval_obs_lag_ticks): the policy acts on
    # obs from N ticks ago while its actions apply in real time — the bridge
    # equivalent of the measured live client cmd→snapshot round trip
    # (+1 tick on LAN). 0 → zero-lag bridge semantics.
    obs_lag_ticks: int = 0
    # Optional release-candidate decode regime for Python eval parity with
    # in-graph exports. Example: "a24rc1".
    decode_regime: str | None = None
    # OPT-IN batched-forward decode (eval_batched_forward). Default False keeps
    # the per-env B=1 forward path, whose action streams are bit-identical to
    # the sequential (num_envs=1) protocol — every regression/replication eval
    # depends on that. When True, all ACTIVE envs' obs are stacked into ONE
    # batch and decoded with a single model.act(B=N) per macro-step, so the
    # CPU saturates instead of idling in the python<->engine ping-pong. The
    # env returns variable-length entity_* fields (sized to the live
    # entity_count), so the batched path pads them to MAX_TOKEN_OBJECTS before
    # stacking (entity_types pad sentinel -1) — the same thing PPO's QuakeEnv
    # does; the transformer then key-pads from entity_types so the padded rows
    # are masked out and the variable LOGICAL actor count is handled inside the
    # model with no per-env contamination. Per-env
    # decode state (hidden, move sticky/hazard/switchback arrays, the hazard
    # xorshift rng, and the sampled per-row torch.Generator) stays isolated:
    # each is stacked into a contiguous (B,...) array, passed positionally so
    # row i ↔ env i, and scattered back per env after the call. This path is
    # NOT bit-identical to B=1 (a batched matmul reorders float reductions),
    # which forks sampled trajectories at the float level — fine for the aim
    # grid, whose coh_5deg over thousands of LOS ticks is robust to that. Use
    # only where per-frame bit-reproducibility is not required.
    batched_forward: bool = False


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
    # Sticky/hazard decode state: previous emitted class per axis (1 = none
    # at episode start, matching the engine's QNN_OnnxReset) and the current
    # run length per axis (dwell age, for the hazard decode).
    move_prev: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1], dtype=np.int64))
    move_age: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1], dtype=np.int64))
    # Switch-back watermark per fb/lr axis: [axis][0] = banned class
    # (-1 = inactive), [axis][1] = softmax(banned) at the switch tick.
    # Mutated in place by policy.act when move_switchback_eps is set.
    move_swb: np.ndarray = field(default_factory=lambda: np.full((2, 2), -1.0, dtype=np.float32))
    # Per-episode xorshift32 rng for the hazard-release draw, threaded across
    # ticks (mutated in place by policy.act) so the offline hazard is a PERSISTED
    # stochastic stream matching the deployed ONNX (move_state_rng init=entropy,
    # reset=persist) — not a fixed-seed reseed every tick (degenerate). Seeded
    # from the episode seed (non-zero u32) for reproducibility; see _seed_move_rng.
    move_state_rng: np.ndarray = field(default_factory=lambda: np.array([0x9E3779B9], dtype=np.int64))
    # Per-episode ATTACK decode state, threaded like move_state_rng (mutated in
    # place by policy.act): the continuous-weapon hold-tail (attack_state, (1,1)
    # f32, reset per episode) + attack's own xorshift rng for the sampled Bernoulli
    # draw (attack_rng, persisted stochastic stream matching the deployed wire.11).
    attack_state: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    attack_rng: np.ndarray = field(default_factory=lambda: np.array([0x6C078965], dtype=np.int64))
    # Per-tick decoded move classes, filled only when log_action_streams.
    move_trace: List[tuple] = field(default_factory=list)
    # obs_lag_ticks delay line: obs the policy hasn't been shown yet.
    obs_delay: List = field(default_factory=list)


def _pad_entities_to_max(obs: np.ndarray | Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Pad an env obs's variable-length entity_* fields to MAX_TOKEN_OBJECTS.

    NativeWorldEnv returns entity_* arrays sized to the live ``entity_count``;
    the batched forward stacks obs across envs, which needs a fixed leading
    entity dim. Mirrors qnn.ppo.env._pad_entities (entity_types pad sentinel
    = -1 so the model's actor mask treats pad rows as empty; all other
    entity_* fields zero-fill). Returns a shallow copy with the entity arrays
    replaced — the caller's live obs is left untouched. Non-dict obs (legacy
    flat) and obs without entity_count pass straight through.
    """
    from qnn.vocab import MAX_TOKEN_OBJECTS
    if not isinstance(obs, MappingABC) or "entity_count" not in obs:
        return dict(obs) if isinstance(obs, MappingABC) else obs  # type: ignore[return-value]
    n = int(np.asarray(obs["entity_count"]).item())
    out: Dict[str, np.ndarray] = dict(obs)
    for key, value in obs.items():
        if not key.startswith("entity_") or key == "entity_count":
            continue
        arr = np.asarray(value)
        if arr.shape[0] == MAX_TOKEN_OBJECTS:
            continue
        fill = -1 if key == "entity_types" else 0
        padded = np.full((MAX_TOKEN_OBJECTS, *arr.shape[1:]), fill, dtype=arr.dtype)
        if n > 0:
            padded[:n] = arr[:n]
        out[key] = padded
    return out


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
    from qnn.env.reward import RewardWeights

    # Carry the scenario spec id into options so the env reports it in
    # info["scenario_id"] — lets multiple scenarios on one map (the aim grid
    # cells) bucket separately. Procgen/single-scenario leave it as the map id.
    env_options = dict(scenario.options)
    env_options.setdefault("scenario_id", scenario.scenario_id)

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
        options=env_options,
        procgen=scenario.procgen_cfg,
    )


def _seed_move_rng(episode_seed: int) -> np.ndarray:
    """Per-episode xorshift32 seed for the move-decode hazard rng, derived from
    the episode seed (reproducible) and forced non-zero (xorshift32 reseeds on 0)."""
    s = (int(episode_seed) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
    return np.array([s or 0x9E3779B9], dtype=np.int64)


def _seed_attack_rng(episode_seed: int) -> np.ndarray:
    """Per-episode xorshift32 seed for the SAMPLED attack-decode rng — a DIFFERENT
    salt from _seed_move_rng so move + attack draw INDEPENDENT streams (the two are
    decoupled), reproducible from the episode seed, forced non-zero."""
    s = (int(episode_seed) * 40503 + 0x6C078965) & 0xFFFFFFFF
    return np.array([s or 0x6C078965], dtype=np.int64)


def _episode_rng(config: EvalConfig, mode: str, episode_index: int, device: torch.device) -> torch.Generator:
    offset = 0 if mode == "greedy" else config.sample_seed_offset
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(config.seed + offset + episode_index)
    return generator


# Optional per-tick model-internals dump (debug). Set from --model-diag-log;
# closed-loop diagnostics (weapon desired-vs-held echo-lock, look/move streams)
# go straight to JSONL from Python. Single-mode runs only (no thread race).
_MODEL_DIAG_LOG: str | None = None


def _select_actions_batch(
    model: QNNPolicy,
    mode: str,
    states: Sequence[_EpisodeState],
    move_sticky_tau: tuple[float, float] | None = None,
    move_hazard: Dict[str, Any] | None = None,
    move_switchback_eps: float | None = None,
    move_stop_onset: bool = False,
    move_tau_engagement_gated: bool = False,
    move_jump_sample: bool = False,
    move_jump_bias: float = 0.0,
    move_jump_temp: float = 1.0,
) -> tuple[List[Mapping[str, object]], np.ndarray]:
    """Per-env B=1 forwards (NOT one batched forward).

    Token obs are ragged across envs — the entity token count varies per
    frame per env — so cross-env stacking is impossible in general. B=1
    forwards also keep each episode's action stream bit-identical to the
    sequential (num_envs=1) protocol: a batched matmul changes float
    reduction order, which would fork the sampled trajectories and break
    replication against historical runs. Parallelism lives in the env
    *stepping* (ThreadPoolExecutor in _evaluate_mode), which is where the
    wall-clock goes; the d_model=64 CPU forward is negligible per tick.
    """
    actions: List[Mapping[str, object]] = []
    hiddens: List[np.ndarray] = []
    for state in states:
        obs_b = _stack_obs([state.obs])
        hidden_b = np.stack([state.hidden], axis=0)
        sticky_kw = {}
        if move_sticky_tau is not None:
            sticky_kw = {"move_sticky_tau": move_sticky_tau,
                         "move_sticky_prev": state.move_prev[None, :]}
            if move_hazard is not None:
                sticky_kw["move_hazard"] = move_hazard
                sticky_kw["move_dwell_age"] = state.move_age[None, :]
                # thread the per-episode hazard rng (mutated in place) so the
                # release draw is a persisted stochastic stream, not a fixed reseed.
                sticky_kw["move_state_rng"] = state.move_state_rng
            if move_switchback_eps is not None:
                sticky_kw["move_switchback_eps"] = move_switchback_eps
                # view, not copy: policy.act mutates the watermark in place
                sticky_kw["move_switchback_state"] = state.move_swb[None, :, :]
            if move_stop_onset:
                sticky_kw["move_stop_onset"] = True
            if move_tau_engagement_gated:
                sticky_kw["move_tau_engagement_gated"] = True
            if move_jump_sample:
                sticky_kw["move_jump_sample"] = True
                sticky_kw["move_jump_bias"] = move_jump_bias
                sticky_kw["move_jump_temp"] = move_jump_temp
        # ATTACK is always decoded (sampled + hold-tail) — thread its per-episode
        # state + rng (mutated in place) regardless of the move sticky config.
        sticky_kw["attack_state"] = state.attack_state
        sticky_kw["attack_rng"] = state.attack_rng
        if mode == "greedy":
            action_batch = model.act(obs_b, mode=mode, hidden=hidden_b,
                                     diag_log_path=_MODEL_DIAG_LOG, **sticky_kw)
        elif mode == "sampled":
            if state.rng is None:
                raise RuntimeError("Sampled evaluation requires a persistent per-episode RNG")
            action_batch = model.act(obs_b, mode=mode, hidden=hidden_b,
                                     row_generators=[state.rng], diag_log_path=_MODEL_DIAG_LOG,
                                     **sticky_kw)
        else:
            raise ValueError(f"Unsupported policy mode {mode}")
        if move_sticky_tau is not None:
            # emitted move values are class−1 per axis; thread classes + dwell
            # ages back (age = current run length, reset on class change)
            emitted = action_batch.actions["move"][0].astype(np.int64) + 1
            state.move_age = np.where(emitted == state.move_prev,
                                      state.move_age + 1, 1)
            state.move_prev = emitted
        actions.append({
            "move": action_batch.actions["move"][0].astype(np.float32, copy=False).tolist(),
            "look": action_batch.actions["look"][0].astype(np.float32, copy=False).tolist(),
            **{
                head: int(action_batch.actions[head][0])
                for head in ACTION_HEADS
                if head not in {"move", "look"}
            },
        })
        hiddens.append(
            action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)[0]
        )
    next_hidden = np.stack(hiddens, axis=0) if hiddens else model.zero_hidden(0)
    return actions, next_hidden


def _select_actions_batched(
    model: QNNPolicy,
    mode: str,
    states: Sequence[_EpisodeState],
    move_sticky_tau: tuple[float, float] | None = None,
    move_hazard: Dict[str, Any] | None = None,
    move_switchback_eps: float | None = None,
    move_stop_onset: bool = False,
    move_tau_engagement_gated: bool = False,
    move_jump_sample: bool = False,
    move_jump_bias: float = 0.0,
    move_jump_temp: float = 1.0,
) -> tuple[List[Mapping[str, object]], np.ndarray]:
    """ONE batched model.act(B=N) over all active envs (eval_batched_forward).

    Drop-in replacement for ``_select_actions_batch`` that stacks every active
    env's obs into a single batch and runs one forward, so the d_model=64 CPU
    decode amortizes across N envs and the cores saturate (the python per-tick
    cost no longer ping-pongs one-env-at-a-time with the engines). Obs token
    tensors are fixed width, so stacking is the same ``_stack_obs`` the B=1 path
    already uses; the transformer key-pads from entity_types internally.

    Per-env decode state stays ISOLATED by construction: hidden, the sticky
    ``move_prev``/``move_age`` arrays, the ``move_swb`` watermark, and the
    hazard ``move_state_rng`` are each stacked into a contiguous (B,...) array
    whose row i is env i; ``policy.act`` reads/writes row i for env i, and the
    in-place-mutated arrays (swb, rng) are scattered back to each state after
    the call. Sampled mode passes one torch.Generator PER ROW so each episode's
    RNG advances independently (categorical_sample/bernoulli_sample draw row by
    row). NOT bit-identical to the B=1 path — see EvalConfig.batched_forward.
    """
    n = len(states)
    # Stack obs across envs. NativeWorldEnv returns the raw variable-length wire
    # obs (entity_* fields sized to the live entity_count), so the entity rows
    # must be padded to MAX_TOKEN_OBJECTS before cross-env stacking — the same
    # thing PPO's QuakeEnv._pad_entities does so its batched forward works. The
    # padding is replicated locally (NOT imported from qnn.ppo.env, which pulls
    # in gymnasium — absent in the eval/dev container) with identical semantics:
    # entity_types pads with -1 (the empty-slot sentinel the model's actor mask
    # reads), every other entity_* field zero-fills. Pad a shallow copy per env
    # so the live state.obs keeps its originals (downstream per-env metrics read
    # state.obs).
    obs_b = _stack_obs([_pad_entities_to_max(s.obs) for s in states])
    hidden_b = np.stack([s.hidden for s in states], axis=0)

    sticky_kw: Dict[str, Any] = {}
    swb_batch: np.ndarray | None = None
    rng_batch: np.ndarray | None = None
    if move_sticky_tau is not None:
        sticky_kw["move_sticky_tau"] = move_sticky_tau
        sticky_kw["move_sticky_prev"] = np.stack([s.move_prev for s in states], axis=0)
        if move_hazard is not None:
            sticky_kw["move_hazard"] = move_hazard
            sticky_kw["move_dwell_age"] = np.stack([s.move_age for s in states], axis=0)
            # Contiguous (B,) rng — mutated in place by act(); scattered back below
            # so each env's hazard stream persists independently.
            rng_batch = np.concatenate(
                [np.asarray(s.move_state_rng, dtype=np.int64).reshape(-1) for s in states]
            ).astype(np.int64)
            sticky_kw["move_state_rng"] = rng_batch
        if move_switchback_eps is not None:
            sticky_kw["move_switchback_eps"] = move_switchback_eps
            # Contiguous (B,2,2) watermark — mutated in place, scattered back.
            swb_batch = np.stack([s.move_swb for s in states], axis=0)
            sticky_kw["move_switchback_state"] = swb_batch
        if move_stop_onset:
            sticky_kw["move_stop_onset"] = True
        if move_tau_engagement_gated:
            sticky_kw["move_tau_engagement_gated"] = True
        if move_jump_sample:
            sticky_kw["move_jump_sample"] = True
            sticky_kw["move_jump_bias"] = move_jump_bias
            sticky_kw["move_jump_temp"] = move_jump_temp

    # ATTACK is always decoded (sampled + hold-tail) — stack its per-env state +
    # rng into contiguous (B,...) arrays, mutated in place by act(), scattered back
    # below so each env's hold-tail + draw stream persist independently.
    attack_state_batch = np.concatenate(
        [np.asarray(s.attack_state, dtype=np.float32).reshape(1, -1) for s in states], axis=0)
    attack_rng_batch = np.concatenate(
        [np.asarray(s.attack_rng, dtype=np.int64).reshape(-1) for s in states]).astype(np.int64)
    sticky_kw["attack_state"] = attack_state_batch
    sticky_kw["attack_rng"] = attack_rng_batch

    if mode == "greedy":
        action_batch = model.act(obs_b, mode=mode, hidden=hidden_b,
                                 diag_log_path=_MODEL_DIAG_LOG, **sticky_kw)
    elif mode == "sampled":
        row_generators = [s.rng for s in states]
        if any(g is None for g in row_generators):
            raise RuntimeError("Sampled evaluation requires a persistent per-episode RNG")
        action_batch = model.act(obs_b, mode=mode, hidden=hidden_b,
                                 row_generators=row_generators, diag_log_path=_MODEL_DIAG_LOG,
                                 **sticky_kw)
    else:
        raise ValueError(f"Unsupported policy mode {mode}")

    move_np = action_batch.actions["move"].astype(np.int64)  # (B,3) class-1 per axis
    if move_sticky_tau is not None:
        emitted_all = move_np + 1
        for i, s in enumerate(states):
            emitted = emitted_all[i]
            s.move_age = np.where(emitted == s.move_prev, s.move_age + 1, 1)
            s.move_prev = emitted
        # Scatter the in-place-mutated state back to each env.
        if rng_batch is not None:
            for i, s in enumerate(states):
                s.move_state_rng[...] = rng_batch[i:i + 1].astype(
                    np.asarray(s.move_state_rng).dtype).reshape(
                    np.asarray(s.move_state_rng).shape)
        if swb_batch is not None:
            for i, s in enumerate(states):
                s.move_swb[...] = swb_batch[i]

    # Scatter the in-place-mutated ATTACK state + rng back (unconditional — attack
    # is always decoded, independent of the move sticky config).
    for i, s in enumerate(states):
        s.attack_state[...] = attack_state_batch[i:i + 1].astype(
            np.asarray(s.attack_state).dtype).reshape(np.asarray(s.attack_state).shape)
        s.attack_rng[...] = attack_rng_batch[i:i + 1].astype(
            np.asarray(s.attack_rng).dtype).reshape(np.asarray(s.attack_rng).shape)

    actions: List[Mapping[str, object]] = []
    for i in range(n):
        actions.append({
            "move": action_batch.actions["move"][i].astype(np.float32, copy=False).tolist(),
            "look": action_batch.actions["look"][i].astype(np.float32, copy=False).tolist(),
            **{
                head: int(action_batch.actions[head][i])
                for head in ACTION_HEADS
                if head not in {"move", "look"}
            },
        })
    next_hidden = action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)
    return actions, next_hidden


def _step_env(
    env: NativeWorldEnv,
    action: Mapping[str, int],
) -> Tuple[np.ndarray | Dict[str, np.ndarray], float, bool, Dict[str, object]]:
    obs, reward, done, info = env.step(action)
    return obs, reward, done, dict(info)


def _prime_obs_delay(state: _EpisodeState, lag: int) -> None:
    """Fill the delay line at episode start: the policy sees the reset obs
    until real frames are old enough to surface (matches the live client,
    where the first post-signon snapshots arrive before any command has
    taken effect)."""
    if lag > 0:
        state.obs_delay = [state.obs] * (lag + 1)


def _advance_obs_delay(state: _EpisodeState, new_obs, lag: int) -> None:
    """Push the freshest env obs and surface the lag-ticks-old one."""
    if lag <= 0:
        state.obs = new_obs
        return
    state.obs_delay.append(new_obs)
    if len(state.obs_delay) > lag + 1:
        state.obs_delay.pop(0)
    state.obs = state.obs_delay[0]


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


# --- obs-side fire discrimination ("fires into the void") metric --------------
# At each fire tick, the best aim alignment (crosshair->actor cosine) is read
# from the model's OWN observation tokens. entity_rel is view-frame (forward=
# +x), so cos = rel_x / |rel|. A fire with no actor token at all, or with the
# best actor outside the aim cone, is a "blind fire" — the live analog of the
# offline fire-by-crosshair-angle table. Mirrors scripts/analysis/
# fire_target_conditional.py so live numbers compare to the human reference.
# This is the obs-side view ("did the model fire when its own perception showed
# nothing aligned?"); a future engine-side version uses ground-truth world
# positions + LOS, and the divergence between the two is the discrimination gap.
_FIRE_CONE_DEG = 10.0
_FIRE_ANGLE_EDGES_DEG = (2.0, 5.0, 10.0, 20.0, 45.0)
_FIRE_ANGLE_LABELS = ("[0,2)", "[2,5)", "[5,10)", "[10,20)", "[20,45)", "[45,180]")

# Per-tick decoded TURN magnitude (deg) binned by the SAME crosshair->LOS angle
# zones above — the bot-side mirror of the human turn-by-error curve. Matching this
# 2-D (LOS-angle x turn-magnitude) histogram to the human reference matches the
# lock-on STYLE (the turn_mag_scale dampener target; aligned fires follow downstream).
_TURN_MAG_EDGES_DEG = (1.0, 2.0, 5.0, 10.0, 20.0, 45.0)
_TURN_MAG_LABELS = ("[0,1)", "[1,2)", "[2,5)", "[5,10)", "[10,20)", "[20,45)", "[45,180]")


def _fire_aim_best_cos(obs: object) -> float | None:
    """Best crosshair->actor cosine from one env's obs tokens; None if no actor."""
    if not isinstance(obs, Mapping) or "entity_rel" not in obs or "entity_types" not in obs:
        return None
    rel = np.asarray(obs["entity_rel"], dtype=np.float32).reshape(-1, 3)
    et = np.asarray(obs["entity_types"]).reshape(-1).astype(np.int64)
    actor = et == TOKEN_ACTOR
    if not actor.any():
        return None
    r = rel[actor]
    n = np.linalg.norm(r, axis=-1)
    valid = n > 1e-6
    if not valid.any():
        return None
    return float((r[valid, 0] / n[valid]).max())


# --- engine-side LEAD-POINT-referenced aim coherence --------------------------
# tracking_cos bins the bearing to the enemy's RAW ORIGIN (no lead). The swept
# aim-prior gain aims at the velocity-led intercept (ground/splash anchor for
# RL), so for projectiles the raw-origin angle mis-scores a correctly-leading
# bot. The v3 QTRN extension exposes the nearest in-LOS actor's view-frame rel
# pos + ABSOLUTE world velocity + the currently-held weapon; from those we
# recompute the crosshair→LEAD-POINT angle via the LIVE aim-prior geometry
# (qnn.model.bench.a24.lead_aim.compute_lead_aim/held_weapon_trajectory) — the
# very physics the human coh_5deg curve and the deployed prior use — so
# model↔human coherence is computed identically. RL gets
# the same AIM_Z_DROP ground/feet anchor on BOTH sides (the comparability
# requirement); body-center for non-floored weapons.

def _import_aim_skill():
    """The aim_skill module (holds THE shared lead kernel). Follows the
    qnn.diag convention of putting scripts/analysis on sys.path rather than
    relying on the repo root being importable as a `scripts` package."""
    import os
    import sys
    scripts_dir = os.path.join(os.path.dirname(__file__), "../../../scripts/analysis")
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import aim_skill  # type: ignore
    return aim_skill


def _build_lead_physics_tables():
    """(weapon_static (9,7), aim_z_drop (9,2), wid→impulse (256,)) — cached."""
    global _LEAD_PHYS_TABLES
    try:
        return _LEAD_PHYS_TABLES
    except NameError:
        pass
    _LEAD_PHYS_TABLES = _import_aim_skill()._build_physics_tables()
    return _LEAD_PHYS_TABLES


_LEAD_DIST_SCALE = 1000.0
_LEAD_VEL_SCALE = 2000.0
_LEAD_WT_V_HORIZ = 2


def _lead_aim_cos_from_info(
    info: Mapping[str, object],
) -> "tuple[float, float] | None":
    """``(cos, feet_pitch_deg)`` for the nearest in-LOS actor, or None.

    Reads the v3 lead geometry (lead_rel/lead_vel raw u, lead_weapon_id,
    lead_valid) and runs the shared lead kernel. None when no actor participated
    this tick (lead_valid=0) — caller skips, exactly like an out-of-LOS frame.

    ``cos`` is the crosshair→feet-anchored-lead-point cosine (the full 3-D angle,
    comparable to the human coh_5deg curve). ``feet_pitch_deg`` is the SIGNED
    VERTICAL angle from the crosshair (view-forward) to that feet-anchored point —
    the metric the RL-splash feet-aim is supposed to drive to ~0. The crosshair is
    view-forward, so a bot aiming over the head leaves the feet anchor BELOW it
    (negative pitch); a bot on the feet reads ~0. Lateral coh / tracking_cos are
    blind to this (they fold pitch into a single angle), which is how the
    rockets-over-the-head regression slipped past the first A/B (look-aim-decode.md
    §12).
    """
    if not info.get("lead_valid", False):
        return None
    ws, zdrop, wid_to_imp = _build_lead_physics_tables()
    rel = np.asarray(info.get("lead_rel", (0.0, 0.0, 0.0)), dtype=np.float32) / _LEAD_DIST_SCALE
    vel = np.asarray(info.get("lead_vel", (0.0, 0.0, 0.0)), dtype=np.float32) / _LEAD_VEL_SCALE
    # lead_weapon_id is the RAW engine weapon id (1..8, QNN_WeaponId). wid_to_imp
    # is indexed by the ENTITY_IDS-encoded (subject) id, like the human curve's
    # obs self_weapon_id — so convert raw→subject (+2) before the lookup. (The
    # recurring raw-vs-subject ±2 trap; do NOT drop this conversion.)
    raw_wid = int(info.get("lead_weapon_id", 0))
    subj_wid = (raw_wid + 2) & 0xFF if raw_wid > 0 else 0
    imp = int(wid_to_imp[subj_wid])

    # Crosshair→lead-point angle via the LIVE aim-prior geometry: the hitscan
    # ×100 boost (held_weapon_trajectory) collapses lead to the bearing for
    # instant-fire weapons, and the intercept quadratic + per-weapon z-anchor
    # (compute_lead_aim) are the deployed prior's exact physics — so the model
    # coherence and the human coh_5deg curve cannot drift apart.
    import torch
    from qnn.model.bench.a24.lead_aim import compute_lead_aim, held_weapon_trajectory

    ws_t = torch.from_numpy(np.ascontiguousarray(ws, dtype=np.float32))
    imp_t = torch.tensor([imp], dtype=torch.long)
    v_horiz, drop_const, drop_rate = held_weapon_trajectory(ws_t, imp_t)   # (1,) each
    rel_t = torch.from_numpy(rel).reshape(1, 1, 3)
    vel_t = torch.from_numpy(vel).reshape(1, 1, 3)
    aim = compute_lead_aim(rel_t, vel_t, v_horiz, drop_const, drop_rate)[0, 0]  # (3,)
    norm = float(aim.norm().clamp_min(1e-9))
    cos = float((aim[0] / norm).clamp(-1.0, 1.0))
    # signed vertical angle of the feet anchor relative to the crosshair (view-
    # forward). index 0 = forward, index 2 = vertical (the AIM_Z_DROP / gravity
    # axis); index 1 = lateral. atan2(up, horizontal) → −ve when the anchor is
    # below the crosshair (the over-the-head failure), ~0 when on the feet.
    horiz = float((aim[0] ** 2 + aim[1] ** 2) ** 0.5)
    feet_pitch_deg = float(np.degrees(np.arctan2(float(aim[2]), max(horiz, 1e-9))))
    # DRIFT DECOMPOSITION: origin (center-mass) anchor pitch (drop zeroed) + range.
    # feet_pitch = origin_pitch − feet_below_origin; comparing origin_pitch (aim vs
    # center-mass) and range (→ feet_below_origin) to the human isolates whether the
    # closed-loop drift is "engages closer" (range) vs "aims higher" (origin_pitch).
    aim_o = compute_lead_aim(rel_t, vel_t, v_horiz,
                             torch.zeros_like(drop_const), torch.zeros_like(drop_rate))[0, 0]
    horiz_o = float((aim_o[0] ** 2 + aim_o[1] ** 2) ** 0.5)
    origin_pitch_deg = float(np.degrees(np.arctan2(float(aim_o[2]), max(horiz_o, 1e-9))))
    range_u = float(np.linalg.norm(rel)) * _LEAD_DIST_SCALE
    return cos, feet_pitch_deg, origin_pitch_deg, range_u


def _evaluate_mode(
    config: EvalConfig,
    model: QNNPolicy,
    mode: str,
    episode_specs: Sequence[Tuple[int, int | None]],
) -> Dict[str, float]:
    del episode_specs
    num_envs = max(1, min(config.num_envs, config.num_episodes))
    move_sticky_tau = (
        (float(config.move_sticky_tau_fb), float(config.move_sticky_tau_lr))
        if config.move_sticky_tau_fb is not None and config.move_sticky_tau_lr is not None
        else None
    )
    move_streams: Dict[str, np.ndarray] = {}
    scenarios, job_queues = _episode_jobs(config)
    executor = ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="nq-eval") if num_envs > 1 else None

    envs: Dict[int, NativeWorldEnv] = {}
    idx_scenarios: Dict[int, _ScenarioSpec] = {}
    returns: List[float] = []
    end_health: List[int] = []
    end_armor: List[int] = []
    done_reasons: Dict[str, int] = {}
    aux_metric_sums: Dict[str, float] = {}
    episode_metric_values: Dict[str, List[float]] = {}
    stuck_steps = 0
    total_steps = 0
    # obs-side fire-discrimination counters
    fire_ticks = 0
    blind_no_actor = 0
    blind_offcone = 0
    fire_cos_sum = 0.0
    fire_cos_n = 0
    fire_angle_hist = [0] * len(_FIRE_ANGLE_LABELS)
    # engine-side fire x LOS alignment (ground-truth tracking_cos) counters
    los_n = 0
    los_fire = 0
    los_cos_sum = 0.0
    los_cos2_sum = 0.0
    los_cosfire_sum = 0.0
    los_tick_bucket = [0] * len(_FIRE_ANGLE_LABELS)
    los_fire_bucket = [0] * len(_FIRE_ANGLE_LABELS)
    # Per-scenario LOS-angle bucket counts (origin + lead). Heterogeneous-
    # scenario batched evals (the aim grid: one scenario per model_weapon x
    # frikbot_weapon cell) need coh_5deg PER cell, which the global buckets
    # can't give. Keyed by scenario_id → list[int] over _FIRE_ANGLE_LABELS.
    scenario_los_tick_bucket: Dict[str, List[int]] = {}
    scenario_lead_tick_bucket: Dict[str, List[int]] = {}
    # LEAD-POINT-referenced mirror of the two buckets above: same edges, but the
    # angle is crosshair→velocity-led intercept (ground anchor for RL) via the
    # shared lead kernel — the metric comparable to the human coh_5deg curve.
    lead_tick_bucket = [0] * len(_FIRE_ANGLE_LABELS)
    lead_fire_bucket = [0] * len(_FIRE_ANGLE_LABELS)
    lead_n = 0
    lead_cos_sum = 0.0
    lead_cosfire_sum = 0.0
    # SIGNED VERTICAL feet-aim error (deg) — the RL-splash feet-aim's own metric,
    # which the lateral coh / tracking_cos fold away. Accumulated over all lead
    # frames and (separately) over operative-fire frames; per-scenario too.
    lead_pitch_sum = 0.0
    lead_pitch_fire_sum = 0.0
    lead_fire_n = 0
    # RL-ONLY fire-frame feet pitch (raw weapon id 7). The all-weapon fire mean is
    # diluted by hitscan (anchor 0); RL is the only weapon whose feet-aim matters
    # (straight-flight rocket aimed high sails over → lands behind). This isolates it.
    lead_pitch_fire_rl_sum = 0.0
    lead_fire_rl_n = 0
    lead_pitch_fire_rl_vals: list = []   # per-frame RL fire-pitch → distribution (human-likeness)
    lead_origin_pitch_rl_vals: list = []  # origin(center-mass) pitch → drift decomposition
    lead_range_rl_vals: list = []         # target range (u) → feet-below-origin driver
    scenario_lead_pitch_sum: Dict[str, float] = {}
    scenario_lead_pitch_n: Dict[str, int] = {}
    # 2-D: [LOS-angle bin][turn-magnitude bin] count over all in-LOS ticks.
    los_turn_hist = [[0] * len(_TURN_MAG_LABELS) for _ in range(len(_FIRE_ANGLE_LABELS))]
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

    def _ensure_env(idx: int, scenario: _ScenarioSpec) -> NativeWorldEnv:
        current = idx_scenarios.get(idx)
        if current is not None and current.scenario_id == scenario.scenario_id:
            return envs[idx]
        existing = envs.pop(idx, None)
        if existing is not None:
            existing.close()
        env = _build_eval_env(config, scenario)
        envs[idx] = env
        idx_scenarios[idx] = scenario
        return env

    try:
        active: Dict[int, _EpisodeState] = {}
        for idx in range(num_envs):
            job = _next_job()
            if job is None:
                break
            env = _ensure_env(idx, job.scenario)
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
                move_state_rng=_seed_move_rng(job.episode_seed),
                attack_rng=_seed_attack_rng(job.episode_seed),
                hidden=model.zero_hidden(1)[0].copy(),
            )
            _prime_obs_delay(active[len(active) - 1], config.obs_lag_ticks)

        select_actions = (
            _select_actions_batched if config.batched_forward else _select_actions_batch
        )
        while active:
            idx_ids = sorted(active.keys())
            states = [active[idx] for idx in idx_ids]
            actions, next_hidden = select_actions(
                model=model,
                mode=mode,
                states=states,
                move_sticky_tau=move_sticky_tau,
                move_hazard=config.move_hazard,
                move_switchback_eps=config.move_switchback_eps,
                move_stop_onset=config.move_stop_onset,
                move_tau_engagement_gated=config.move_tau_engagement_gated,
                move_jump_sample=config.move_jump_sample,
                move_jump_bias=config.move_jump_bias,
                move_jump_temp=config.move_jump_temp,
            )
            if config.log_action_streams:
                for _i, _st in enumerate(states):
                    mv = actions[_i]["move"]
                    _st.move_trace.append(
                        (int(mv[0]) + 1, int(mv[1]) + 1, int(mv[2]) + 1))

            # obs-side fire discrimination: at each fire tick, score the model's
            # crosshair alignment to the nearest-aligned actor in its OWN obs.
            for _i, _st in enumerate(states):
                if not int(actions[_i].get("attack", 0)):
                    continue
                fire_ticks += 1
                _cos = _fire_aim_best_cos(_st.obs)
                if _cos is None:
                    blind_no_actor += 1
                    fire_angle_hist[-1] += 1  # no actor → treat as widest bin
                    continue
                _ang = float(np.degrees(np.arccos(np.clip(_cos, -1.0, 1.0))))
                fire_cos_sum += _cos
                fire_cos_n += 1
                if _ang > _FIRE_CONE_DEG:
                    blind_offcone += 1
                _b = int(np.digitize(_ang, _FIRE_ANGLE_EDGES_DEG, right=False))
                fire_angle_hist[min(_b, len(_FIRE_ANGLE_LABELS) - 1)] += 1

            if executor is None:
                results = [_step_env(envs[idx], action) for idx, action in zip(idx_ids, actions)]
            else:
                futures = [
                    executor.submit(_step_env, envs[idx], action)
                    for idx, action in zip(idx_ids, actions)
                ]
                results = [future.result() for future in futures]

            for batch_idx, (idx, result) in enumerate(zip(idx_ids, results)):
                obs, reward, done, info = result
                state = active[idx]
                state.hidden = next_hidden[batch_idx].copy()
                state.step_count += 1
                terminal = bool(done or state.step_count >= config.max_steps_per_episode)
                # Reward remains useful even when the episode ends without a special terminal condition.
                state.return_value += float(reward)
                state.last_info = info
                state.metrics.add_step(reward=float(reward), info=info, terminal=terminal)
                total_steps += 1
                # scenario_id resolved up front so the per-scenario LOS/lead
                # buckets (aim-grid coh_5deg per cell) accumulate alongside the
                # global ones below.
                scenario_id = str(info.get("scenario_id", state.scenario_id))
                state.scenario_id = scenario_id
                _sc_los = scenario_los_tick_bucket.setdefault(
                    scenario_id, [0] * len(_FIRE_ANGLE_LABELS))
                _sc_lead = scenario_lead_tick_bucket.setdefault(
                    scenario_id, [0] * len(_FIRE_ANGLE_LABELS))
                # engine-side fire x LOS alignment: tracking_cos is PVS +
                # traceline gated, so it's aim alignment to the nearest in-LOS
                # enemy this tick. Pair it with the OPERATIVE fire (shots_fired
                # > 0: engine honored the attack, not in cooldown) to measure
                # whether the bot fires when actually aimed at a visible target.
                # Using the raw model attack output would inflate p_fire by
                # 2-3x (model fires on cooldown frames the engine ignores).
                _los_cos = float(info.get("tracking_cos", 0.0))
                _los_fired = int(info.get("shots_fired", 0)) > 0
                _los_ang = float(np.degrees(np.arccos(np.clip(_los_cos, -1.0, 1.0))))
                los_n += 1
                los_cos_sum += _los_cos
                los_cos2_sum += _los_cos * _los_cos
                _lb = int(np.digitize(_los_ang, _FIRE_ANGLE_EDGES_DEG, right=False))
                _lb = min(_lb, len(_FIRE_ANGLE_LABELS) - 1)
                los_tick_bucket[_lb] += 1
                _sc_los[_lb] += 1
                # decoded turn magnitude this tick (view-relative look vector,
                # forward=+x ⇒ turn = arccos(look_x/|look|)) → per-LOS-zone hist.
                _lk = actions[batch_idx].get("look") if isinstance(actions[batch_idx], Mapping) else None
                if _lk is not None:
                    _lk = np.asarray(_lk, dtype=np.float64).reshape(-1)
                    _ln = float(np.linalg.norm(_lk))
                    _u0 = (_lk[0] / _ln) if _ln > 1e-8 else 1.0
                    _turn = float(np.degrees(np.arccos(np.clip(_u0, -1.0, 1.0))))
                    _tb = min(int(np.digitize(_turn, _TURN_MAG_EDGES_DEG, right=False)),
                              len(_TURN_MAG_LABELS) - 1)
                    los_turn_hist[_lb][_tb] += 1
                if _los_fired:
                    los_fire += 1
                    los_cosfire_sum += _los_cos
                    los_fire_bucket[_lb] += 1
                # LEAD-POINT-referenced coherence: same operative-fire selection,
                # but the angle is to the velocity-led intercept (ground anchor
                # for RL) via the shared lead kernel — comparable to the human
                # coh_5deg curve. Only on ticks where an actor participated.
                _lead = _lead_aim_cos_from_info(info)
                if _lead is not None:
                    _lead_cos, _feet_pitch, _origin_pitch, _range_u = _lead
                    _lead_ang = float(np.degrees(np.arccos(np.clip(_lead_cos, -1.0, 1.0))))
                    _ldb = min(int(np.digitize(_lead_ang, _FIRE_ANGLE_EDGES_DEG, right=False)),
                               len(_FIRE_ANGLE_LABELS) - 1)
                    lead_n += 1
                    lead_cos_sum += _lead_cos
                    lead_pitch_sum += _feet_pitch
                    scenario_lead_pitch_sum[scenario_id] = scenario_lead_pitch_sum.get(scenario_id, 0.0) + _feet_pitch
                    scenario_lead_pitch_n[scenario_id] = scenario_lead_pitch_n.get(scenario_id, 0) + 1
                    lead_tick_bucket[_ldb] += 1
                    _sc_lead[_ldb] += 1
                    if _los_fired:
                        lead_cosfire_sum += _lead_cos
                        lead_pitch_fire_sum += _feet_pitch
                        lead_fire_n += 1
                        lead_fire_bucket[_ldb] += 1
                        if int(info.get("lead_weapon_id", 0)) == 7:   # RL (raw id)
                            lead_pitch_fire_rl_sum += _feet_pitch
                            lead_fire_rl_n += 1
                            lead_pitch_fire_rl_vals.append(float(_feet_pitch))
                            lead_origin_pitch_rl_vals.append(float(_origin_pitch))
                            lead_range_rl_vals.append(float(_range_u))
                scenario_total_steps[scenario_id] = scenario_total_steps.get(scenario_id, 0) + 1
                if bool(info.get("stuck", False)):
                    stuck_steps += 1
                    scenario_stuck_steps[scenario_id] = scenario_stuck_steps.get(scenario_id, 0) + 1
                for key, value in _iter_aux_metric_items(info, _AUX_INFO_KEYS + _WEAPON_AUX_KEYS):
                    aux_metric_sums[key] = aux_metric_sums.get(key, 0.0) + value
                    scenario_metric_sums = scenario_aux_metric_sums.setdefault(scenario_id, {})
                    scenario_metric_sums[key] = scenario_metric_sums.get(key, 0.0) + value

                if terminal:
                    if config.log_action_streams and state.move_trace:
                        move_streams[f"ep_{state.episode_index:04d}"] = np.asarray(
                            state.move_trace, dtype=np.int8)
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

                    next_job = _next_job(idx_scenarios[idx].scenario_id)
                    if next_job is not None:
                        env = _ensure_env(idx, next_job.scenario)
                        next_obs = env.reset(seed=next_job.episode_seed, start_variant=next_job.start_variant)
                        active[idx] = _EpisodeState(
                            episode_index=next_job.episode_index,
                            obs=next_obs,
                            scenario_id=next_job.scenario.scenario_id,
                            rng=None if mode == "greedy" else _episode_rng(config, mode, next_job.episode_index, model.device),
                            move_state_rng=_seed_move_rng(next_job.episode_seed),
                            attack_rng=_seed_attack_rng(next_job.episode_seed),
                            hidden=model.zero_hidden(1)[0].copy(),
                        )
                        _prime_obs_delay(active[idx], config.obs_lag_ticks)
                    else:
                        del active[idx]
                else:
                    _advance_obs_delay(state, obs, config.obs_lag_ticks)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in envs.values():
            env.close()

    if config.log_action_streams and move_streams:
        streams_path = Path(config.output_dir) / f"move_streams_{mode}.npz"
        streams_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(streams_path, **move_streams)

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
    # obs-side fire discrimination (compare to human reference fire-by-angle curve)
    _blind = blind_no_actor + blind_offcone
    _hist_tot = max(sum(fire_angle_hist), 1)
    summary["obs_fire_ticks"] = int(fire_ticks)
    summary["obs_fire_rate"] = float(fire_ticks / max(total_steps, 1))
    summary["obs_blind_fire_rate"] = float(_blind / max(fire_ticks, 1))
    summary["obs_blind_fire_no_actor_rate"] = float(blind_no_actor / max(fire_ticks, 1))
    summary["obs_blind_fire_offcone_rate"] = float(blind_offcone / max(fire_ticks, 1))
    summary["obs_fire_aim_cos_mean"] = float(fire_cos_sum / max(fire_cos_n, 1))
    summary["obs_fire_cone_deg"] = _FIRE_CONE_DEG
    summary["obs_fire_angle_hist"] = {
        lab: float(c / _hist_tot) for lab, c in zip(_FIRE_ANGLE_LABELS, fire_angle_hist)
    }
    # engine-side fire x LOS-alignment correlation (ground-truth tracking_cos)
    summary["engine_los_ticks"] = int(los_n)
    summary["engine_fire_tracking_cos_mean"] = float(los_cosfire_sum / max(los_fire, 1))
    summary["engine_nofire_tracking_cos_mean"] = float(
        (los_cos_sum - los_cosfire_sum) / max(los_n - los_fire, 1)
    )
    # point-biserial corr between fire (0/1) and tracking_cos
    if los_n > 1:
        _num = los_n * los_cosfire_sum - los_fire * los_cos_sum
        _den_f = los_n * los_fire - los_fire * los_fire           # = n*Σf - (Σf)^2
        _den_c = los_n * los_cos2_sum - los_cos_sum * los_cos_sum
        _den = (_den_f * _den_c) ** 0.5
        summary["engine_fire_tracking_cos_corr"] = float(_num / _den) if _den > 1e-12 else 0.0
    else:
        summary["engine_fire_tracking_cos_corr"] = 0.0
    # P(fire | LOS-aim-angle bucket): live analog of the offline aim table
    summary["engine_los_attack_by_origin_angle"] = {
        lab: {
            "n_ticks": int(t),
            "p_fire": float(f / t) if t else 0.0,
        }
        for lab, t, f in zip(_FIRE_ANGLE_LABELS, los_tick_bucket, los_fire_bucket)
    }
    # LEAD-POINT-referenced mirror (same edges): crosshair→velocity-led intercept
    # (ground/splash anchor for RL via the shared lead kernel). This is the field
    # comparable to the human coh_5deg curve and the one the aim-prior gain sweep
    # should be scored on for projectiles — the raw engine_los_attack_by_origin_angle bins
    # the no-lead bearing and mis-scores a correctly-leading bot.
    summary["engine_lead_cos_mean"] = float(lead_cos_sum / max(lead_n, 1))
    summary["engine_lead_ticks"] = int(lead_n)
    # SIGNED VERTICAL feet-aim error (deg): ~0 = crosshair on the feet anchor,
    # strongly −ve = aiming over the target (the rockets-over-the-head failure).
    # The fire-frame version is the one that gates RL splash payoff.
    summary["engine_lead_feet_pitch_deg_mean"] = float(lead_pitch_sum / max(lead_n, 1))
    summary["engine_lead_feet_pitch_deg_fire_mean"] = float(lead_pitch_fire_sum / max(lead_fire_n, 1))
    # RL-only fire-frame pitch — the decisive "do rockets launch on the feet or high"
    # number (all-weapon fire mean is hitscan-diluted). n=0 → reported as None.
    summary["engine_lead_feet_pitch_deg_fire_rl_mean"] = (
        float(lead_pitch_fire_rl_sum / lead_fire_rl_n) if lead_fire_rl_n else None)
    summary["engine_lead_feet_pitch_deg_fire_rl_n"] = int(lead_fire_rl_n)
    # DISTRIBUTION of RL fire-pitch (human-likeness: match the human's mean AND spread,
    # not a point). p10/p90 spread distinguishes the human-wide head from the aimbot lock.
    if lead_pitch_fire_rl_vals:
        _v = np.asarray(lead_pitch_fire_rl_vals, dtype=np.float64)
        summary["engine_lead_feet_pitch_deg_fire_rl_dist"] = {
            "n": int(_v.size), "mean": float(_v.mean()),
            "p10": float(np.percentile(_v, 10)), "median": float(np.median(_v)),
            "p90": float(np.percentile(_v, 90)), "std": float(_v.std())}
        if lead_origin_pitch_rl_vals:
            _o = np.asarray(lead_origin_pitch_rl_vals, dtype=np.float64)
            _r = np.asarray(lead_range_rl_vals, dtype=np.float64)
            summary["engine_lead_origin_pitch_deg_fire_rl_dist"] = {
                "mean": float(_o.mean()), "median": float(np.median(_o)), "std": float(_o.std())}
            summary["engine_lead_range_u_fire_rl_dist"] = {
                "mean": float(_r.mean()), "median": float(np.median(_r)),
                "p10": float(np.percentile(_r, 10)), "p90": float(np.percentile(_r, 90))}
    summary["engine_los_attack_by_lead_angle"] = {
        lab: {
            "n_ticks": int(t),
            "p_fire": float(f / t) if t else 0.0,
        }
        for lab, t, f in zip(_FIRE_ANGLE_LABELS, lead_tick_bucket, lead_fire_bucket)
    }
    # Decoded turn-magnitude distribution per LOS-angle zone (the lock-on curve):
    # {los_angle_bin: {turn_mag_bin: p}} normalized within each LOS zone. Matched
    # against the human reference to calibrate the turn_mag_scale dampener.
    summary["engine_turn_by_los_angle"] = {
        lab: {
            "n_ticks": int(sum(row)),
            "turn_dist": {tl: (float(c / sum(row)) if sum(row) else 0.0)
                          for tl, c in zip(_TURN_MAG_LABELS, row)},
        }
        for lab, row in zip(_FIRE_ANGLE_LABELS, los_turn_hist)
    }
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
            # Per-scenario LOS-angle bucket counts (origin + lead) so a
            # heterogeneous-scenario batched eval yields coh_5deg per cell —
            # the aim grid reads these instead of the global buckets.
            "engine_los_attack_by_origin_angle": {
                lab: {"n_ticks": int(t)}
                for lab, t in zip(
                    _FIRE_ANGLE_LABELS,
                    scenario_los_tick_bucket.get(scenario_id, [0] * len(_FIRE_ANGLE_LABELS)),
                )
            },
            "engine_los_attack_by_lead_angle": {
                lab: {"n_ticks": int(t)}
                for lab, t in zip(
                    _FIRE_ANGLE_LABELS,
                    scenario_lead_tick_bucket.get(scenario_id, [0] * len(_FIRE_ANGLE_LABELS)),
                )
            },
            "engine_lead_feet_pitch_deg_mean": float(
                scenario_lead_pitch_sum.get(scenario_id, 0.0)
                / max(scenario_lead_pitch_n.get(scenario_id, 0), 1)
            ),
        }
        for scenario_id, values in sorted(scenario_returns.items())
    }
    return summary


def _load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    model_config: Dict[str, Any] | None = None,
) -> QNNPolicy:
    """Load a checkpoint in either QNN (.pth) or SF (.pth) format."""
    from qnn.utils.checkpoint_converter import load_checkpoint
    return load_checkpoint(path, device=device, model_config=model_config)


def _install_decode_regime(model: QNNPolicy, regime: str | None):
    """Install the decode/guard layer from a decode config. ``regime`` is either a
    known regime name (a24rc1/a24rc2/a25rc1a → bundled config) or a path to a
    decode-config JSON. The resolver builds the param-bound guard adapter that
    QNNPolicy.act consumes (policy_decode_action_postprocess + projectile_release_mask
    when dodge is enabled) — the same object tools/export_onnx.ExportWrapper uses.

    Returns the ``ResolvedDecode`` (or ``None`` for the no-regime case) so the
    caller can source move/look decode params FROM the config — even when
    ``guard_module == "none"`` (then ``_regime_mod`` is None but the resolved
    params still drive the move/look operating point, matching export)."""
    if regime is None or str(regime).strip() in ("", "none"):
        model.decode_action_postprocess = None
        model._regime_mod = None
        model._decode_mod = None  # act() falls back to the default decode facade
        return None
    from qnn.model import decode_config as _dc
    resolved = _dc.resolve_decode_config(_dc.config_path_for(str(regime).strip()))
    adapter = resolved.guard_module
    if adapter is None:
        # guard_module="none" (e.g. scrubbed a24rc1 study base): no guard layer, but
        # the regime still provides the geometry decode module + params. Install the
        # decode geometry, leave the guard/fire-postprocess off (act() already guards
        # every _regime_mod use with `is not None`). The resolved params still drive
        # the move/look operating point via _apply_decode_config_params on `resolved`.
        model.decode_action_postprocess = None
        model._regime_mod = None
        model._decode_mod = resolved.decode_module
        return resolved
    # Stash the raw params on the adapter so callers can read decode-config
    # values (e.g. look.turn_mag_scale) without re-parsing the JSON.
    adapter._params = resolved.params
    model.decode_action_postprocess = adapter.policy_decode_action_postprocess
    # Store the resolved guard adapter so QNNPolicy.act can call its
    # projectile_release_mask(obs) (Gate B forced fb/lr hold release) inside the
    # shared move decode — exactly as tools/export_onnx.ExportWrapper does.
    model._regime_mod = adapter
    # Use the decode config's geometry module (not a hard-coded a24) in act().
    model._decode_mod = resolved.decode_module
    return resolved


def _apply_decode_config_params(config: "EvalConfig", model: QNNPolicy, resolved) -> None:
    """Source the MOVE + LOOK decode operating point from the resolved
    decode-config onto the (mutable) EvalConfig + model, mirroring
    tools/export_onnx's precedence exactly so eval and export agree. Called only
    when a regime is resolved; when no regime is set the train.json ``eval_*``
    keys stand as the legacy fallback. Each key is overridden only when the
    config DEFINES it, so a config that omits a knob leaves the eval_* fallback
    in place."""
    p = resolved.params
    # ── look aim-prior gain + ffwd (export: kw["look_aim_prior_gain"] etc.). The
    # model reads model.look_aim_prior_gain / model.look_aim_ffwd directly in
    # policy.act, so set them on the model (config.look_aim_prior_gain is also
    # updated for provenance/consistency; there is no config.look_aim_ffwd field).
    if "look.aim_prior_gain" in p:
        config.look_aim_prior_gain = float(p["look.aim_prior_gain"])
        model.look_aim_prior_gain = float(p["look.aim_prior_gain"])
    if "look.aim_ffwd_gain" in p:
        model.look_aim_ffwd = float(p["look.aim_ffwd_gain"])
    # ── global attack-propensity bias (sampled-mode s-lever): sigmoid(logit+bias)
    if "attack.bias" in p:
        model.attack_bias = float(p["attack.bias"])
    # attack.threshold — in-graph fire decision. None/absent = sampled (Bernoulli);
    # a float τ → deterministic fire iff sigmoid(fire_logit+bias) ≥ τ (commits on
    # confident ticks so attack co-occurs with jump+aim). τ ~ human fire frequency.
    if "attack.threshold" in p:
        model.attack_threshold = float(p["attack.threshold"])
    # ── combined rotation+magnitude: blend kept |z| from θ (0) toward |z+z_prior| (1)
    if "look.aim_mag_gain" in p:
        model.look_aim_mag_gain = float(p["look.aim_mag_gain"])
    # ── head turn-magnitude dampener: scale the head's native |z|=θ before the
    # aim-prior blend (default 1.0 = OFF = bit-identical). Corrects the
    # decode-reachable head over-turn term only (covariate-shift is training-side).
    if "look.turn_mag_scale" in p:
        model.look_turn_mag_scale = float(p["look.turn_mag_scale"])
    # ── per-weapon VERTICAL aim authority (RL-splash feet-aiming; default OFF).
    # A (9,) per-impulse self-limiting pitch gain that restores the feet-anchor
    # authority the rotation blend starves (look-aim-decode.md §12).
    if "look.weapon_pitch_gain" in p:
        model.look_weapon_pitch_gain = [float(x) for x in p["look.weapon_pitch_gain"]]
    # Per-weapon downward pitch bias (degrees; cancels the static RL high-bias that
    # β and feed-forward can't touch → rockets stop sailing over and landing behind).
    if "look.weapon_pitch_bias" in p:
        model.look_weapon_pitch_bias = [float(x) for x in p["look.weapon_pitch_bias"]]
    # Feet-elevation LOCK toggle (default on = hard overwrite). Off → β-blend only
    # (soft bias, head keeps its vertical). See QNNPolicy.look_weapon_pitch_lock.
    if "look.weapon_pitch_lock" in p:
        model.look_weapon_pitch_lock = bool(p["look.weapon_pitch_lock"])
    # Post-expmap pitch mode: "lock" | "shift" | "off" (+ shift strength). "shift"
    # translates the head's fired elevation toward the feet while preserving spread.
    if "look.weapon_pitch_mode" in p:
        model.look_weapon_pitch_mode = str(p["look.weapon_pitch_mode"])
    if "look.weapon_pitch_shift_strength" in p:
        model.look_weapon_pitch_shift_strength = float(p["look.weapon_pitch_shift_strength"])
    # ── hazard-discounted lead: cap the horizontal lead horizon at the expected
    # strafe-hold (20Hz frames; default OFF = linear lead). Stops RL over-leading
    # past the human dwell so rockets don't overshoot behind a strafing target.
    if "look.lead_hold_cap_frames" in p:
        model.look_lead_hold_cap_frames = float(p["look.lead_hold_cap_frames"])
    if "look.lead_hold_cap_radial_frames" in p:
        model.look_lead_hold_cap_radial_frames = float(p["look.lead_hold_cap_radial_frames"])
    # ── aim DEGRADATION (DOWN-half skill knob; default 0 = OFF = bit-identical).
    # Stateful post-head transforms on the look turn-delta; the model reads these
    # attrs directly in policy.act. See look-aim-decode.md §11.
    if "look.aim_degrade_sluggish_tau" in p:
        model.look_aim_degrade_sluggish_tau = float(p["look.aim_degrade_sluggish_tau"])
    if "look.aim_degrade_lag_frames" in p:
        model.look_aim_degrade_lag_frames = float(p["look.aim_degrade_lag_frames"])
    if "look.aim_degrade_tremor_mag" in p:
        model.look_aim_degrade_tremor_mag = float(p["look.aim_degrade_tremor_mag"])
    if "look.aim_degrade_tremor_tau" in p:
        model.look_aim_degrade_tremor_tau = float(p["look.aim_degrade_tremor_tau"])
    if "look.aim_degrade_jitter_mag" in p:
        model.look_aim_degrade_jitter_mag = float(p["look.aim_degrade_jitter_mag"])
    # ── move sticky-tau operating point (export: kw["move_sticky_tau_fb"] etc.)
    if "move.sticky_tau_fb" in p:
        config.move_sticky_tau_fb = float(p["move.sticky_tau_fb"])
    if "move.sticky_tau_lr" in p:
        config.move_sticky_tau_lr = float(p["move.sticky_tau_lr"])
    if "move.switchback_eps" in p:
        config.move_switchback_eps = float(p["move.switchback_eps"])
    if "move.stop_onset" in p:
        config.move_stop_onset = bool(p["move.stop_onset"])
    if "move.tau_engagement_gated" in p:
        config.move_tau_engagement_gated = bool(p["move.tau_engagement_gated"])
    if "move.jump_sample" in p:
        config.move_jump_sample = bool(p["move.jump_sample"])
    if "move.jump_bias" in p:
        config.move_jump_bias = float(p["move.jump_bias"])
    if "move.jump_temp" in p:
        config.move_jump_temp = float(p["move.jump_temp"])
    # ── move hazard (export: materializes resolved.move_hazard as a temp JSON for
    # the loader; the eval path consumes the inline dict directly — the a24rc1
    # lineage carries the lognorm block inline). Only override when the config
    # supplies one AND train.json didn't already pin a table.
    if resolved.move_hazard and config.move_hazard is None:
        config.move_hazard = dict(resolved.move_hazard)


def _evaluate_policy_mode(
    config: EvalConfig,
    mode: str,
    model_config: Dict[str, Any] | None = None,
) -> tuple[str, Dict[str, float], QNNPolicy]:
    model = _load_checkpoint(config.checkpoint_path, device=config.device, model_config=model_config)
    model.look_aim_prior_gain = config.look_aim_prior_gain
    model.weapon_ban = tuple(config.weapon_ban)
    resolved = _install_decode_regime(model, config.decode_regime)
    # When a decode regime/config is resolved, source the MOVE + LOOK operating
    # point FROM the config (decode lives in the decode-config, not train.json),
    # mirroring tools/export_onnx's precedence. With no regime, the train.json
    # eval_* keys stand as the legacy fallback (resolved is None → unchanged).
    # This applies even when guard_module=="none" (_regime_mod is None) because
    # _install_decode_regime still returns the resolved params.
    if resolved is not None:
        _apply_decode_config_params(config, model, resolved)
    summary = _evaluate_mode(config, model, mode, _episode_specs(config))
    return mode, summary, model


def _checkpoint_run_id(checkpoint_path: str) -> str:
    """run_id from the checkpoint's meta sidecar, '' when absent (legacy
    checkpoints, SF payloads, missing sidecar)."""
    sidecar = Path(checkpoint_path).with_suffix(".json")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        return str(meta.get("run_id", ""))
    except (OSError, ValueError):
        return ""


def run_evaluation(
    config: EvalConfig,
    model_config: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    if config.device == "cpu":
        # Same tiny-matmul pathology worker_inference.py pins: the d_model=64
        # B=1 forwards lose to OpenMP spin-wait when torch fans out a thread
        # pool per op, and concurrent eval arms multiply the contention. One
        # thread makes each forward faster AND frees the cores.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # interop pool already initialized — intra-op pin still applies
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

    # Provenance: which run produced the evaluated checkpoint. Read from the
    # checkpoint's meta sidecar (QNNPolicy.save stamps run_id there).
    summary["checkpoint_path"] = str(config.checkpoint_path)
    summary["checkpoint_run_id"] = _checkpoint_run_id(config.checkpoint_path)

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
                f"transformer encoder + GRU({model_card_model.d_gru}) actor-critic"
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
# CLI entry point (python -m qnn.eval.run)
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from qnn.run.config import (
        load_run_config,
        build_run_eval_config,
        run_output_dirs,
        _require_mapping,
        _require_string,
    )
    from qnn.env.planning import resolve_asset_root, validate_native_mod_assets

    parser = argparse.ArgumentParser(description="Multi-episode evaluation of a checkpoint")
    parser.add_argument("run_dir", type=Path, help="Run directory containing run.json")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path relative to run's checkpoints dir")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override eval_num_episodes from config")
    parser.add_argument("--num-envs", type=int, default=None, help="Override eval_num_envs from config")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--model-diag-log", default=None,
                        help="dump per-tick model internals (weapon desired/held/conf, "
                             "look pred, move logits/probs) to this JSONL — debug only")
    args = parser.parse_args()

    if args.model_diag_log:
        global _MODEL_DIAG_LOG
        _MODEL_DIAG_LOG = args.model_diag_log

    run_cfg = load_run_config(args.run_dir.resolve())
    machine = _require_mapping(run_cfg, "machine", "run config")
    model_config = _require_mapping(run_cfg, "model", "run config")

    # Install the polar look grid from the run's pinned config/look_grid.json
    # (no import-time default since d24ee386). The runner entry point does this;
    # the standalone CLI must too or the look head raises at first forward.
    _look_grid_path = args.run_dir / "config" / "look_grid.json"
    if _look_grid_path.exists():
        import json as _json
        from qnn.model.look_bins import install_polar_grid as _install_polar_grid
        _lg = _json.loads(_look_grid_path.read_text())
        _install_polar_grid(
            torch.tensor(_lg["mag_centers_rad"], dtype=torch.float32),
            torch.tensor(_lg["dir_centers_rad"], dtype=torch.float32)
            if "dir_centers_rad" in _lg else None,
        )

    outputs = run_output_dirs(run_cfg)
    checkpoint_path = outputs["checkpoints"] / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    asset_root = resolve_asset_root(_require_string(machine, "asset_root", "machine.json"))
    worker_path = Path(run_cfg["run_dir"]).parent.parent / _require_string(machine, "worker_binary", "machine.json")
    if not worker_path.exists():
        worker_path = Path.cwd() / _require_string(machine, "worker_binary", "machine.json")
    if not worker_path.exists():
        raise FileNotFoundError(f"Worker binary not found: {worker_path}")

    native_args = _require_mapping(run_cfg, "scenario", "run config").get("native_args", [])
    validate_native_mod_assets(asset_root, native_args)

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


# ── Runner entry point (called by run.router) ──────────────────────

def run(ctx: "RunnerContext") -> dict[str, object]:
    """Run standalone evaluation from a frozen run directory."""
    import time as _time

    from qnn.env.planning import validate_native_mod_assets
    from qnn.run.config import build_run_eval_config
    from qnn.run.common import (
        RunnerContext, base_results, ensure_worker, finalize_results,
        prepare_eval_checkpoint, prepare_eval_outputs, require_cfg_string,
    )

    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    prepare_eval_outputs(ctx.run_cfg, resume=ctx.resume)

    # Install the polar look grid from the run's pinned config/look_grid.json.
    # Required since d24ee386 removed the import-time default; the bench runner
    # installs it too but the eval runner was not updated at the same time.
    _look_grid_path = ctx.run_dir / "config" / "look_grid.json"
    if _look_grid_path.exists():
        import json as _json
        from qnn.model.look_bins import install_polar_grid as _install_polar_grid
        import torch as _torch
        _lg = _json.loads(_look_grid_path.read_text())
        _install_polar_grid(
            _torch.tensor(_lg["mag_centers_rad"], dtype=_torch.float32),
            _torch.tensor(_lg["dir_centers_rad"], dtype=_torch.float32)
            if "dir_centers_rad" in _lg else None,
        )

    eval_cfg = build_run_eval_config(ctx.run_cfg, ctx.device)
    validate_native_mod_assets(
        ctx.asset_root,
        eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None,
    )
    worker_path = ensure_worker(ctx.worker_binary, rebuild=False)
    eval_cfg["native_env"] = {"QUAKE_BASEDIR": str(ctx.asset_root)}
    eval_cfg["native_executable"] = str(worker_path)
    eval_cfg["checkpoint_path"] = prepare_eval_checkpoint(
        require_cfg_string(ctx.run_cfg, "checkpoint_path", "run config"),
        str(eval_cfg["output_dir"]),
    )

    started = _time.monotonic()
    results["eval"] = run_evaluation(EvalConfig(**eval_cfg))
    stage_timings["eval"] = _time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
