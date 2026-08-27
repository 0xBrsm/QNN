"""Policy evaluation and reporting."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from collections.abc import Mapping as MappingABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from qnn.actions import ACTION_HEADS
# Per-discharge intercept event schema (decode-fit v2 P1: "every operative
# discharge is a sample"). Filename, row fields and version are PINNED in
# qnn.decode_fit.events so this writer and the fit-side loader can never drift.
from qnn.decode_fit.events import (
    EVENT_SCHEMA_VERSION as _EVENT_SCHEMA_VERSION,
    INTERCEPT_EVENTS_NPZ as _INTERCEPT_EVENTS_NPZ,
    RAW_FIELDS as _EVENT_RAW_FIELDS,
)
from qnn.run.metrics import (
    EpisodeStatAccumulator,
    append_metric_values,
    build_eval_summary_aliases,
    mean_metric_values,
)
from qnn.model.decode_actions import commit_reset_lanes as _commit_reset_lanes
from qnn.model.look_seg_decode import (
    LOOK_COMMIT_STATE_DIM,
    look_commit_reset_lanes as _look_commit_reset_lanes)
from qnn.model.decode import BatchedRNG
from qnn.model.policy import QNNPolicy
from qnn.schema import GATE_STREAM_SCHEMA_VERSION, SELF_SCALAR_DIM
from qnn.vocab import TOKEN_ACTOR, self_weapon_id_to_impulse
from qnn.env.world import NativeWorldEnv
from qnn.wire import OBS_BUFFER_SIZE, unpack_frame_batch, unpack_obs_buffer_native_batch
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
    # decode-contract default (the decode facade's AIM_PRIOR_GAIN); 0.0 → prior
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
    # Dump per-episode decoded move-class streams (fb,lr,ud per tick) to
    # output_dir/move_streams_<mode>.npz — the live-side input of the
    # move dwell/switch diagnostics. eval_log_action_streams.
    log_action_streams: bool = False
    # Dump per-episode raw entity obs streams (entity_count/types/rel/vel/recency
    # + current attack-with weapon choice) to output_dir/acq_streams_<mode>.npz — the
    # closed-loop input of the ACQUISITION (Fitts-throughput) axis. These are the
    # SAME per-frame obs the human-corpus flick kernel consumes
    # (scripts/analysis/acq_submovement.py), scenario-tagged so a heterogeneous
    # batched eval yields a per-cell acquisition band alongside the per-cell
    # intercept ruler. eval_log_acq_streams.
    log_acq_streams: bool = False
    # Live-client latency emulation (eval_obs_lag_ticks): the policy acts on
    # obs from N ticks ago while its actions apply in real time — the bridge
    # equivalent of the measured live client cmd→snapshot round trip
    # (+1 tick on LAN). 0 → zero-lag bridge semantics.
    obs_lag_ticks: int = 0
    # Optional release-candidate decode regime for Python eval parity with
    # in-graph exports. Example: "a25rc1" or a decode-config path.
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
    # decode state (hidden, the commitment lanes, the attack wire slots, and
    # the sampled per-row torch.Generator) stays isolated:
    # each is stacked into a contiguous (B,...) array, passed positionally so
    # row i ↔ env i, and scattered back per env after the call. This path is
    # NOT bit-identical to B=1 (a batched matmul reorders float reductions),
    # which forks sampled trajectories at the float level — fine for the aim
    # grid, whose coh_5deg over thousands of LOS ticks is robust to that. Use
    # only where per-frame bit-reproducibility is not required.
    batched_forward: bool = False
    # PER-LANE decode overrides (the aim-grid closed-loop widener). One dict per
    # ENV SLOT (index 0..num_envs-1), each mapping supported decode-config keys
    # (qnn.model.policy._PER_ROW_DECODE_KEYS: look.aim_prior_gain / aim_mag_gain /
    # turn_mag_scale / aim_degrade_tremor_mag) to that lane's scalars — a single
    # swept key (v1 aim-grid waves) or a multi-key full operating point per lane
    # (decode-fit v2 design rounds); the key set must be UNIFORM across lanes
    # (_per_row_decode_from_states enforces it). Threaded
    # onto each _EpisodeState at creation and applied ROW-BY-ROW inside the one
    # batched model.act(B=N) — the shared GRU forward is unchanged; only the
    # per-sample DECODE differs by lane. So a SINGLE 64-lane eval can run many
    # (gain/α/tremor, scenario) cells instead of one cold subprocess per swept
    # value. None ⇒ every lane uses the model/decode-config scalar (back-compat).
    # Requires batched_forward=True. Length must be ≥ num_envs.
    per_env_decode_overrides: tuple[Mapping[str, float], ...] | None = None
    # Explicit opt-in for homogeneous qnn_arena8 evaluation.  The default
    # process backend retains seeded multi-map/procgen behavior.
    env_backend: str = "process"
    arena_server_binary: str = ""
    arena_client_binary: str = ""
    arena_map_id: str = "qnn_arena8"
    arena_base_port: int = 28900
    arena_bot_skill: int = 3
    arena_matches_per_server: int = 8


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
    # Per-episode ATTACK state slots. The a25 attack_with decode is STATELESS —
    # these pass through policy.act untouched — but the wire keeps the slots for
    # parity with the deployed graph, so the eval threads them the same way.
    attack_state: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    attack_rng: np.ndarray = field(default_factory=lambda: np.array([0x6C078965], dtype=np.int64))
    # a25 commitment decode state — the FULL COMMIT_STATE_DIM lanes from the
    # decode module's reset init (fb/lr commit + rel + water-ud + spare lanes;
    # exactly the ONNX state_loopback memset). movearch models hard-require
    # lanes [5]=ud_cls [6]=ud_rem; the previous hand-rolled 5-lane array only
    # covered fb/lr and broke the first movearch closed-loop eval.
    move_commit: np.ndarray = field(default_factory=lambda: np.asarray(
        _commit_reset_lanes(), dtype=np.float32))
    # a25 LOOK commitment decode state — LOOK_COMMIT_STATE_DIM lanes
    # [cls, rem, elapsed, dur_bucket, dir_bin] from the reset init (rem=0 forces an
    # onset on the first tick). Mutated in place by policy.act's look_commit decode,
    # scattered back per env — the move_commit precedent.
    look_commit: np.ndarray = field(default_factory=lambda: np.asarray(
        _look_commit_reset_lanes(), dtype=np.float32))
    # Per-tick decoded move classes, filled only when log_action_streams.
    move_trace: List[tuple] = field(default_factory=list)
    # Per-tick threat flags (bit0=damage, bit1=INCOMING-projectile appeared
    # [own-fire gated], bit2=incoming projectile present), filled only
    # when log_action_streams — the closed-loop threat-PSTH input.
    threat_trace: List[int] = field(default_factory=list)
    # Per-tick (attack, weapon, turn_deg, keep, discharge, weapon_imp, engaged)
    # for the decode-fit stage-6 rc_humanlikeness gate (band-v5 flat schema),
    # plus the 5 LOOK COMMIT lanes (cls, rem, elapsed, dur_bucket, dir_bin) this
    # tick's look decode left behind — the cross-head coordination trace; -1
    # when the model has no look_seg head — plus the 2 look-TANGENT components
    # (schema 4). Filled only when log_action_streams.
    gate_trace: List[tuple] = field(default_factory=list)
    # Per-tick raw entity obs (n, entity_types, entity_rel, entity_vel,
    # entity LOS state and attack-with weapon choice) for ACQUISITION, filled only when
    # log_acq_streams — the closed-loop analog of the human collect cache the
    # acq_submovement flick kernel reads.
    acq_trace: List[tuple] = field(default_factory=list)
    _last_health: float = 1e9
    _last_nproj: int = 0
    # obs_lag_ticks delay line: obs the policy hasn't been shown yet.
    obs_delay: List = field(default_factory=list)
    # PER-LANE decode override for this env slot (EvalConfig.per_env_decode_overrides
    # [slot]); a dict of supported decode keys → scalar, or None (the model/decode
    # scalar). Collected across the batch and applied row-by-row in the one act().
    decode_override: Mapping[str, float] | None = None


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


@lru_cache(maxsize=8)
def _declaration_for_checkpoint(checkpoint_path: str) -> object:
    """Obs declaration for the checkpoint's run dir (obs_api v1).

    The eval must attach the declaration the model TRAINED on — the
    engine's no-attach default plan tracks the branch head, not any
    given checkpoint. Checkpoints live at <run_dir>/checkpoints/*.pth,
    so the run dir (which declaration_for_run resolves stamps and
    bare-stamp corpus fallbacks against) is two levels up.
    """
    from qnn.obs_api import declaration_for_run
    return declaration_for_run(Path(checkpoint_path).resolve().parent.parent)


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
        declaration=_declaration_for_checkpoint(config.checkpoint_path),
    )


def _seed_attack_rng(episode_seed: int) -> np.ndarray:
    """Per-episode xorshift32 seed for the attack_rng wire slot (pass-through on
    the stateless a25 attack decode; kept for wire parity), reproducible from the
    episode seed, forced non-zero."""
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
    batched_rng_gen: "torch.Generator | None" = None,
) -> tuple[List[Mapping[str, object]], np.ndarray]:
    """Per-env B=1 forwards (NOT one batched forward).

    ``batched_rng_gen`` is accepted (shared-signature with
    :func:`_select_actions_batched`) and IGNORED: the B=1 path draws from each
    episode's own persistent per-lane generator to stay bit-identical to the
    sequential protocol.

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
        # Thread the wire state slots. attack_state/attack_rng pass through the
        # stateless a25 attack decode (kept for wire parity); move_commit_state
        # is mutated in place by the commitment decode.
        sticky_kw = {"attack_state": state.attack_state,
                     "attack_rng": state.attack_rng}
        if getattr(model, "move_commitment", False):
            # view, not copy: policy.act mutates the commitment state in place
            sticky_kw["move_commit_state"] = state.move_commit[None, :]
        if getattr(model, "look_commitment", False):
            sticky_kw["look_commit_state"] = state.look_commit[None, :]
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
        actions.append({
            "move": action_batch.actions["move"][0].astype(np.float32, copy=False).tolist(),
            "look": action_batch.actions["look"][0].astype(np.float32, copy=False).tolist(),
            **{
                head: int(action_batch.actions[head][0])
                for head in action_batch.actions
                if head not in {"move", "look"}
            },
        })
        hiddens.append(
            action_batch.next_hidden.detach().cpu().numpy().astype(np.float32, copy=False)[0]
        )
    next_hidden = np.stack(hiddens, axis=0) if hiddens else model.zero_hidden(0)
    return actions, next_hidden


def _lane_decode_override(
    config: EvalConfig, lane: int
) -> Mapping[str, float] | None:
    """The per-lane decode override for env slot ``lane`` (or None).

    ``EvalConfig.per_env_decode_overrides`` is indexed by env slot; slots recur
    as episodes finish and a fresh job takes the slot, so the override stays
    pinned to the LANE (not the episode). None (default) ⇒ the model/decode
    scalar for every lane (back-compat)."""
    ov = config.per_env_decode_overrides
    if not ov:
        return None
    if lane >= len(ov):
        raise ValueError(
            f"per_env_decode_overrides has {len(ov)} entries but env slot "
            f"{lane} was requested (need one per lane, ≥ num_envs)")
    return ov[lane]


def _per_row_decode_from_states(
    states: Sequence[_EpisodeState],
) -> Mapping[str, np.ndarray] | None:
    """Collect the per-lane ``decode_override`` dicts (one per state, row-aligned)
    into the ``per_row_decode`` array-of-lanes payload ``policy.act`` consumes.

    Returns None when no lane carries an override (the scalar path). When ANY
    lane overrides, EVERY lane must carry the SAME key set — the swept params are
    homogeneous across the grid, and a silent per-lane default would apply the
    wrong operating point. The uniform set may hold ONE key (the v1 aim-grid
    waves, which sweep a single lever per wave) or SEVERAL of the supported
    per-row keys (decode-fit v2 design rounds, where each lane pins its FULL
    operating point, e.g. {look.aim_prior_gain, look.aim_mag_gain,
    look.aim_degrade_tremor_mag, look.turn_mag_scale}): the guard below is on
    the key SET, one row-aligned array falls out per key, and downstream
    nothing assumes a single key — ``QNNPolicy._resolve_per_row_decode``
    iterates whatever keys arrive and ``act()`` applies each lever in its own
    independent per-row branch. Single-key payloads are byte-identical to what
    this always produced. Each returned array is row-aligned with ``states``."""
    overrides = [s.decode_override for s in states]
    present = [o for o in overrides if o]
    if not present:
        return None
    # ``o or {}``: a None/empty lane must land in the SAME-key-set guard below
    # (as the empty set) — bare frozenset(None) would TypeError past the
    # intended ValueError.
    keysets = {frozenset(o or {}) for o in overrides}
    if len(keysets) != 1 or any(o is None for o in overrides):
        raise ValueError(
            "per_env_decode_overrides: every active lane must override the SAME "
            f"decode keys; got key sets {[sorted(o or {}) for o in overrides]}")
    keys = next(iter(keysets))
    return {k: np.array([float(o[k]) for o in overrides], dtype=np.float32)
            for k in keys}


def _select_actions_batched(
    model: QNNPolicy,
    mode: str,
    states: Sequence[_EpisodeState],
    batched_rng_gen: "torch.Generator | None" = None,
) -> tuple[List[Mapping[str, object]], np.ndarray]:
    """ONE batched model.act(B=N) over all active envs (eval_batched_forward).

    Drop-in replacement for ``_select_actions_batch`` that stacks every active
    env's obs into a single batch and runs one forward, so the d_model=64 CPU
    decode amortizes across N envs and the cores saturate (the python per-tick
    cost no longer ping-pongs one-env-at-a-time with the engines). Obs token
    tensors are fixed width, so stacking is the same ``_stack_obs`` the B=1 path
    already uses; the transformer key-pads from entity_types internally.

    Per-env decode state stays ISOLATED by construction: hidden, the commitment
    ``move_commit`` lanes, and the attack wire slots are each stacked into a
    contiguous (B,...) array whose row i is env i; ``policy.act`` reads/writes
    row i for env i, and the in-place-mutated arrays are scattered back to each
    state after the call. Sampled mode passes ONE :class:`BatchedRNG` wrapping a
    single run-scoped generator (``batched_rng_gen``), sized to the active-lane
    count each tick, so every per-row draw is one vectorized dispatcher call
    (``row_uniforms`` fast branch) instead of the O(lanes) per-generator loop the
    per-row list forced through categorical/commit/jump sampling. Draws stay
    independent per row (same law), but the exact stream differs from the per-row
    generators — the batched path is ALREADY declared non-bit-identical to B=1
    (see EvalConfig.batched_forward); this shifts the same non-determinism, it
    does not add reproducibility risk (the generator is deterministically seeded).
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
    # ATTACK wire slots — stateless pass-through on a25 (kept for wire parity);
    # stacked per env and scattered back below.
    attack_state_batch = np.concatenate(
        [np.asarray(s.attack_state, dtype=np.float32).reshape(1, -1) for s in states], axis=0)
    attack_rng_batch = np.concatenate(
        [np.asarray(s.attack_rng, dtype=np.int64).reshape(-1) for s in states]).astype(np.int64)
    sticky_kw["attack_state"] = attack_state_batch
    sticky_kw["attack_rng"] = attack_rng_batch

    # a25 commitment decode state — threaded when the model opts in; mutated
    # in place by act(), scattered back per env below.
    commit_batch: np.ndarray | None = None
    if getattr(model, "move_commitment", False):
        commit_batch = np.stack([s.move_commit for s in states], axis=0)
        sticky_kw["move_commit_state"] = commit_batch
    look_commit_batch: np.ndarray | None = None
    if getattr(model, "look_commitment", False):
        look_commit_batch = np.stack([s.look_commit for s in states], axis=0)
        sticky_kw["look_commit_state"] = look_commit_batch

    # Per-lane decode overrides (aim-grid widener): one operating point per row,
    # applied inside the single act() (the forward stays shared). None ⇒ scalar.
    per_row_decode = _per_row_decode_from_states(states)
    if per_row_decode is not None:
        sticky_kw["per_row_decode"] = per_row_decode

    if mode == "greedy":
        action_batch = model.act(obs_b, mode=mode, hidden=hidden_b,
                                 diag_log_path=_MODEL_DIAG_LOG, **sticky_kw)
    elif mode == "sampled":
        if batched_rng_gen is None:
            raise RuntimeError(
                "Batched sampled evaluation requires a shared BatchedRNG generator")
        row_generators = BatchedRNG(generator=batched_rng_gen, batch_size=n)
        action_batch = model.act(obs_b, mode=mode, hidden=hidden_b,
                                 row_generators=row_generators, diag_log_path=_MODEL_DIAG_LOG,
                                 **sticky_kw)
    else:
        raise ValueError(f"Unsupported policy mode {mode}")

    if commit_batch is not None:
        for i, s2 in enumerate(states):
            s2.move_commit = commit_batch[i].copy()
    if look_commit_batch is not None:
        for i, s2 in enumerate(states):
            s2.look_commit = look_commit_batch[i].copy()
    # Scatter the ATTACK wire slots back (pass-through on a25; kept for parity).
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
                for head in action_batch.actions
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


def _submit_batched_raw(
    envs: Mapping[int, NativeWorldEnv],
    idx_ids: Sequence[int],
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Phase 1 (submit) of the split-phase raw-bytes batched step (borrowed from
    :class:`qnn.ppo.vec_env.VecQuakeEnv`'s submit/receive split).

    Send EVERY active lane's action packet up front, so all engines sim the tick
    CONCURRENTLY (the pipe write returns immediately; each worker sims as soon as
    its packet lands, independent of when Python drains). The send bytes are
    identical to the per-lane ``env.step`` path — both go through
    ``pack_step_request`` (``step_send`` IS ``step_send_packed(pack_step_request(
    ...))``), so nothing about the engine input changes.

    Splitting submit from :func:`_receive_batched_raw` is what lets the eval loop
    run the (engine-independent) per-tick metric accumulation on the PREVIOUS
    tick's results BETWEEN the two phases — overlapping the residual serial
    Python with engine sim so the waves become engine-bound.
    """
    for idx, action in zip(idx_ids, actions):
        envs[idx].step_send_packed(envs[idx].pack_step_action(action))


def _receive_batched_raw(
    envs: Mapping[int, NativeWorldEnv],
    idx_ids: Sequence[int],
) -> List[Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]]:
    """Phase 2 (receive) of the split-phase raw-bytes batched step.

    Drain each reply as RAW wire bytes (no per-lane python unpack under the GIL)
    and unpack ALL lanes in ONE vectorized call
    (``unpack_obs_buffer_native_batch``) — the eval's per-lane
    ``unpack + _pad_entities_to_max + _stack_obs`` collapses to a single batched
    parse. The returned obs are per-lane row VIEWS into the batched arrays
    (entity fields padded to MAX_TOKEN_OBJECTS, the empty-slot sentinels the
    downstream masks already ignore), row-aligned with ``idx_ids`` exactly like
    the old per-lane ``results`` list — so episode budgeting, resets, per-scenario
    bucketing and result ordering are byte-for-byte unchanged.
    """
    n = len(idx_ids)
    # Frame size + field layout follow the lanes' negotiated declaration
    # (obs_api v1). All lanes in an eval serve one model, so the first
    # lane's layout governs; a mismatched lane fails loudly on the row
    # assignment below (frame length != layout.frame_bytes).
    layout = getattr(envs[idx_ids[0]], "layout", None) if idx_ids else None
    raws = np.empty(
        (n, OBS_BUFFER_SIZE if layout is None else layout.frame_bytes),
        dtype=np.uint8,
    )
    meta: List[Tuple[float, bool, Dict[str, object]]] = []
    for row, idx in enumerate(idx_ids):
        raw, reward, done, info = envs[idx].step_recv_raw()
        raws[row] = np.frombuffer(raw, dtype=np.uint8)
        meta.append((float(reward), bool(done), dict(info)))
    if layout is None:
        obs_b = unpack_obs_buffer_native_batch(raws)
    else:
        obs_b = unpack_frame_batch(raws, layout)
    return [({k: obs_b[k][row] for k in obs_b}, m[0], m[1], m[2])
            for row, m in enumerate(meta)]


def _step_batched_raw(
    envs: Mapping[int, NativeWorldEnv],
    idx_ids: Sequence[int],
    actions: Sequence[Mapping[str, object]],
) -> List[Tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]]:
    """Combined submit+receive one-shot batched step (no overlap window).

    Kept for callers/tests that want a single batched step; the eval loop calls
    the two phases directly so it can overlap the metric pass with engine sim.
    Preserves the submit/receive split property (all sends precede all receives),
    so all engines still sim concurrently.
    """
    _submit_batched_raw(envs, idx_ids, actions)
    return _receive_batched_raw(envs, idx_ids)


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


# Only the keys the eval env's info dict (qnn.env.world.NativeWorldEnv.step)
# actually emits. The dead keys stripped here (monster_kill_delta, ammo_gain,
# weapon_switches, visible_threats, fire_pressed, effective_fire, health_fraction,
# armor_fraction, blind_fire [was hardcoded 0], reward_tracking) were never
# populated on the eval path and read a constant 0.0. Per-weapon telemetry
# (weapon_shots_fired_*/hits/damage) never crosses the QTRN struct either — the
# whole _WEAPON_AUX_KEYS set was dead — so it's gone too.
_AUX_INFO_KEYS = (
    "frag_delta",
    "frag_loss",
    "damage_taken",
    "damage_taken_self",
    "damage_taken_other",
    "damage_dealt",
    "hit_count",
    "shots_fired",
    "health_gain",
    "armor_gain",
    "weapon_pickups",
    "tracking_cos",
)
_EPISODE_AUX_KEYS = (
    "episode_damage_dealt",
    "episode_hit_count",
    "episode_shots_fired",
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

# Discharge-anchored INTERCEPT (alignment-at-attack), hitbox-half-width (hbw) form
# — the a25 aim-skill lever's MODEL-SIDE ruler (research/skill-curves.md §14/§14.1;
# the human side is scripts/analysis/aim_intercept_skill.py). On each operative fire
# with an in-LOS actor we score the lead-corrected angle to the lead point divided by
# the angular hitbox radius, so hbw < 1 = crosshair on the enemy's body at any range.
# The model histogram is scored against the human per-weapon POOLED event ladder in
# runs/head_probe/_aim_intercept_skill.json (interception_dist);
# its bin edges MUST mirror aim_intercept_skill.NORM_EDGES so both sides share one bucketing.
# The actor hitbox half-width is the fixed Quake player bbox (±16u; confirmed
# CONSTANT for actor tokens across the QWD corpus), so the range the shared lead
# kernel already returns is the only per-discharge geometry needed — no engine change.
# Shared constant (aim_kernel) so the crest-gate decode's hbw uses the SAME ruler.
from qnn.eval.aim_kernel import ACTOR_HALFW_U as _ACTOR_HALFW_U  # noqa: E402
_INTERCEPT_HBW_EDGES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0,
                        6.0, 10.0, 15.0, 20.0, 30.0)
_INTERCEPT_HBW_LABELS = tuple(
    f"[{lo},{hi})" for lo, hi in zip((0.0,) + _INTERCEPT_HBW_EDGES,
                                     _INTERCEPT_HBW_EDGES + (float("inf"),)))
_INTERCEPT_PCTS = (5, 10, 25, 50, 75, 90, 95, 99)


def _intercept_hbw(lead_ang_deg: float, range_u: float) -> float:
    """Lead-corrected discharge angle (deg) → hitbox-half-widths (range-invariant:
    angle / atan(halfw/dist)). Mirrors aim_intercept_skill._episode with the fixed
    actor half-width; lower = better (crosshair nearer the enemy body)."""
    ang_radius = float(np.degrees(np.arctan2(_ACTOR_HALFW_U, max(float(range_u), 1e-3))))
    return float(lead_ang_deg) / max(ang_radius, 1e-6)


def _hbw_percentiles(counts: "list[int]") -> "dict | None":
    """Median + percentile ladder from an _INTERCEPT_HBW_EDGES histogram (linear
    interp inside the containing bucket; the open tail clamps to its lower edge).
    Mirrors aim_intercept_skill._pctiles_from_hist / _median_from_hist. None when
    the histogram is empty (no operative-fire discharges scored)."""
    edges = (0.0,) + _INTERCEPT_HBW_EDGES + (float("inf"),)
    tot = float(sum(counts))
    if tot <= 0:
        return None
    cum = np.cumsum(counts)
    pctiles: dict[str, float] = {}
    for p in _INTERCEPT_PCTS:
        tgt = p / 100.0 * tot
        k = min(int(np.searchsorted(cum, tgt)), len(counts) - 1)
        lo, hi = edges[k], edges[k + 1]
        prev = cum[k - 1] if k > 0 else 0.0
        if not np.isfinite(hi) or counts[k] <= 0:
            pctiles[f"p{p}"] = round(float(lo), 3)
        else:
            pctiles[f"p{p}"] = round(float(lo + (tgt - prev) / counts[k] * (hi - lo)), 3)
    return {"n_attacks": int(tot), "median_hbw": pctiles["p50"], "percentiles": pctiles}


# Hard row cap for the per-discharge event accumulator. At ~60 B/row a full log
# is ~30 MB of Python lists — nothing against the 16 GB container budget — but
# a pathological run (runaway fire rate × long schedule) must not balloon
# unbounded. Past the cap the HISTOGRAMS keep accumulating (O(1) memory);
# only the raw rows stop, and the npz is stamped truncated=True so the fit
# knows its sample is a prefix, not the full run.
_INTERCEPT_EVENT_CAP = 500_000

# On-disk dtypes for the event columns, pinned HERE (the writer) so the npz is
# stable regardless of what the accumulator appended (plain python floats/ints).
# Names/semantics mirror qnn.decode_fit.events.RAW_FIELDS one-for-one; unicode
# columns take numpy's native '<U#' width from the data.
_EVENT_DTYPES: Dict[str, Any] = {
    "scenario_id": None,          # unicode — lane's scenario id (cell key)
    "weapon": None,               # unicode — discharging weapon abbr
    "hbw": np.float32,            # discharge alignment, hitbox-half-widths
    "range_u": np.float32,        # target range at the shot (engine units)
    "episode": np.int32,          # episode ordinal WITHIN the lane (0-based)
    "env_idx": np.int32,          # lane index
    "tick": np.int64,             # eval macro-step when recorded
}


def _write_intercept_events(
    output_dir: str | Path,
    events: Mapping[str, Sequence[Any]],
    truncated: bool = False,
) -> Path | None:
    """Write the per-discharge intercept event table (decode-fit v2's sample
    unit) to ``output_dir/intercept_events.npz``.

    ``events`` maps every ``qnn.decode_fit.events.RAW_FIELDS`` name to an
    equal-length column (one entry per operative discharge; see the accumulator
    in ``_evaluate_policy_mode``). ``output_dir`` is the eval's summary dir —
    ``<run>/metrics/eval`` (run_output_dirs) — i.e. exactly where
    ``qnn.decode_fit.events.load_run_events`` looks. Zero rows ⇒ no file, None
    returned (the loader treats absence as "predates event logging / no
    discharges"; an empty npz would be indistinguishable noise). The write goes
    through a temp sibling + os.replace so a reader (or the other policy-mode
    thread under parallel_policy_modes — the pinned filename carries no mode
    suffix, so multi-mode runs are last-writer-wins) never sees a torn file.
    """
    missing = set(_EVENT_RAW_FIELDS) - set(events)
    if missing:
        raise ValueError(f"intercept events missing column(s) {sorted(missing)}")
    lens = {k: len(events[k]) for k in _EVENT_RAW_FIELDS}
    if len(set(lens.values())) != 1:
        raise ValueError(f"intercept event columns are ragged: {lens}")
    if lens["hbw"] == 0:
        return None
    path = Path(output_dir) / _INTERCEPT_EVENTS_NPZ
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(events[k], dtype=_EVENT_DTYPES[k])
              for k in _EVENT_RAW_FIELDS}
    # np.savez_* appends ".npz" when absent, so the temp name must end in it
    # for the final os.replace to target the file numpy actually wrote.
    tmp = path.with_name(
        f".{path.stem}.{os.getpid()}-{threading.get_ident()}.tmp.npz")
    np.savez_compressed(
        tmp,
        schema_version=np.int64(_EVENT_SCHEMA_VERSION),
        truncated=np.bool_(truncated),
        **arrays,
    )
    os.replace(tmp, path)
    return path


def _parse_intercept_window(spec: str | None) -> tuple[int, int]:
    """``QNN_EVAL_INTERCEPT_WINDOW`` → (k_pre, k_post) tick counts.

    ``"16,4"`` = 16 pre-roll ticks + 4 forward ticks around the discharge (the
    frozen geometry decode-fit requests — qnn.decode_fit.events); a bare
    ``"4"`` is the symmetric shorthand for ``"4,4"``; empty/unset disables the
    instrument. Malformed values FAIL LOUD rather than silently disabling it.
    """
    spec = (spec or "").strip()
    if not spec:
        return (0, 0)
    parts = spec.split(",")
    if len(parts) > 2:
        raise ValueError(
            f"QNN_EVAL_INTERCEPT_WINDOW must be '<pre>,<post>' or '<k>', "
            f"got {spec!r}")
    try:
        vals = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"QNN_EVAL_INTERCEPT_WINDOW must be integer ticks, got {spec!r}"
        ) from exc
    k_pre, k_post = (vals * 2)[:2]
    if k_pre < 0 or k_post < 0:
        raise ValueError(
            f"QNN_EVAL_INTERCEPT_WINDOW tick counts must be >= 0, got {spec!r}")
    return (k_pre, k_post)


def _write_intercept_windows(output_dir: str | Path, k_pre: int, k_post: int,
                             rows: Mapping[str, Sequence[Any]]) -> Path | None:
    """Write the tracking-window instrument, including an explicit empty file.

    Each row is one discharge's hbw stream over ``[t0 - k_pre, t0 + k_post]``
    (``k_pre + k_post + 1`` slots, the fired tick at index ``k_pre``). The
    window is ASYMMETRIC: the alignment-lead profile needs a long pre-roll
    (does the crosshair converge before the trigger?) while the forward half
    only has to cover the crest-replay hold horizon. ``k_pre``/``k_post`` are
    both stamped — there is no single half-width to infer them from.

    A zero-discharge wave is valid evidence about fire mass but contributes no
    tracking samples. It still needs a schema-bearing artifact so decode-fit can
    distinguish that result from a wave produced before the instrument existed.
    """
    if not (k_pre or k_post):
        return None
    hbw_win = np.asarray(rows["hbw_win"], dtype=np.float32)
    if not len(hbw_win):
        hbw_win = np.empty((0, int(k_pre) + int(k_post) + 1), dtype=np.float32)
    out = Path(output_dir) / "intercept_windows.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        k_pre=np.int64(k_pre),
        k_post=np.int64(k_post),
        scenario_id=np.asarray(rows["scenario_id"], dtype="U64"),
        weapon=np.asarray(rows["weapon"], dtype="U8"),
        episode=np.asarray(rows["episode"], dtype=np.int32),
        env_idx=np.asarray(rows["env_idx"], dtype=np.int32),
        tick=np.asarray(rows["tick"], dtype=np.int64),
        hbw_win=hbw_win)
    return out


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
# (qnn.model.lead_aim.compute_lead_aim/weapon_trajectory) — the
# very physics the human coh_5deg curve and the deployed prior use — so
# model↔human coherence is computed identically. RL gets
# the same AIM_Z_DROP ground/feet anchor on BOTH sides (the comparability
# requirement); body-center for non-floored weapons.

def _build_lead_physics_tables():
    """Impulse-keyed weapon physics and aim-z tables — cached."""
    global _LEAD_PHYS_TABLES
    try:
        return _LEAD_PHYS_TABLES
    except NameError:
        pass
    from qnn.eval import aim_kernel
    _LEAD_PHYS_TABLES = aim_kernel._build_physics_tables()
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
    ws, zdrop = _build_lead_physics_tables()
    rel = np.asarray(info.get("lead_rel", (0.0, 0.0, 0.0)), dtype=np.float32) / _LEAD_DIST_SCALE
    vel = np.asarray(info.get("lead_vel", (0.0, 0.0, 0.0)), dtype=np.float32) / _LEAD_VEL_SCALE
    raw_wid = int(info.get("lead_weapon_id", 0))
    imp = raw_wid if 1 <= raw_wid <= 8 else 0

    # Crosshair→lead-point angle via the LIVE aim-prior geometry: the hitscan
    # ×100 boost (weapon_trajectory) collapses lead to the bearing for
    # instant-fire weapons, and the intercept quadratic + per-weapon z-anchor
    # (compute_lead_aim) are the deployed prior's exact physics — so the model
    # coherence and the human coh_5deg curve cannot drift apart.
    import torch
    from qnn.model.lead_aim import compute_lead_aim, weapon_trajectory

    ws_t = torch.from_numpy(np.ascontiguousarray(ws, dtype=np.float32))
    imp_t = torch.tensor([imp], dtype=torch.long)
    v_horiz, drop_const, drop_rate = weapon_trajectory(ws_t, imp_t)   # (1,) each
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


def _lead_aim_cos_batched(
    infos: Sequence[Mapping[str, object]],
) -> "list[tuple[float, float, float, float] | None]":
    """Batched, row-aligned equivalent of per-lane :func:`_lead_aim_cos_from_info`.

    Runs the shared lead kernel ONCE over every lane that has a participating
    in-LOS actor (lead_valid) instead of one torch round-trip per lane per tick —
    the dominant per-macro-step serial cost of the batched closed-loop eval.

    BIT-IDENTICAL to the per-lane path (up to float32 vs float64 rounding in the
    trailing scalar trig, < 1e-5): ``compute_lead_aim`` / ``weapon_trajectory``
    are elementwise/broadcast over the batch dim — the only reductions are over the
    xyz axis, PER ROW — so batching cannot change any row's float result. Returns a
    list aligned with ``infos``; None for lanes with lead_valid=0 (no actor), exactly
    like the per-lane function's None return.
    """
    out: list = [None] * len(infos)
    valid_idx = [i for i, info in enumerate(infos) if info.get("lead_valid", False)]
    if not valid_idx:
        return out
    from qnn.model.lead_aim import compute_lead_aim, weapon_trajectory
    ws, _zdrop = _build_lead_physics_tables()
    v = len(valid_idx)
    rel = np.empty((v, 3), dtype=np.float32)
    vel = np.empty((v, 3), dtype=np.float32)
    imps = np.empty((v,), dtype=np.int64)
    for k, i in enumerate(valid_idx):
        info = infos[i]
        rel[k] = np.asarray(info.get("lead_rel", (0.0, 0.0, 0.0)), dtype=np.float32) / _LEAD_DIST_SCALE
        vel[k] = np.asarray(info.get("lead_vel", (0.0, 0.0, 0.0)), dtype=np.float32) / _LEAD_VEL_SCALE
        raw_wid = int(info.get("lead_weapon_id", 0))
        imps[k] = raw_wid if 1 <= raw_wid <= 8 else 0
    ws_t = torch.from_numpy(np.ascontiguousarray(ws, dtype=np.float32))
    imp_t = torch.from_numpy(imps)
    v_horiz, drop_const, drop_rate = weapon_trajectory(ws_t, imp_t)   # (v,) each
    rel_t = torch.from_numpy(rel).reshape(v, 1, 3)
    vel_t = torch.from_numpy(vel).reshape(v, 1, 3)
    # feet-anchored (per-weapon z-drop) and origin/center-mass (drop zeroed) — the
    # SAME two compute_lead_aim calls the per-lane function makes, batched over v.
    aim = compute_lead_aim(rel_t, vel_t, v_horiz, drop_const, drop_rate)[:, 0].numpy().astype(np.float64)
    aim_o = compute_lead_aim(
        rel_t, vel_t, v_horiz,
        torch.zeros_like(drop_const), torch.zeros_like(drop_rate))[:, 0].numpy().astype(np.float64)
    norm = np.maximum(np.linalg.norm(aim, axis=-1), 1e-9)
    cos = np.clip(aim[:, 0] / norm, -1.0, 1.0)
    horiz = np.sqrt(aim[:, 0] ** 2 + aim[:, 1] ** 2)
    feet_pitch = np.degrees(np.arctan2(aim[:, 2], np.maximum(horiz, 1e-9)))
    horiz_o = np.sqrt(aim_o[:, 0] ** 2 + aim_o[:, 1] ** 2)
    origin_pitch = np.degrees(np.arctan2(aim_o[:, 2], np.maximum(horiz_o, 1e-9)))
    range_u = np.linalg.norm(rel.astype(np.float64), axis=-1) * _LEAD_DIST_SCALE
    for k, i in enumerate(valid_idx):
        out[i] = (float(cos[k]), float(feet_pitch[k]),
                  float(origin_pitch[k]), float(range_u[k]))
    return out


def _batched_los_turn_bins(
    infos: Sequence[Mapping[str, object]],
    actions: Sequence[Mapping[str, object]],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(los_cos, los_bin, turn_bin)`` arrays over lanes — the per-tick trig +
    ``np.digitize`` computed ONCE as arrays instead of one round-trip per lane.

    Values are identical to the per-lane path: ``np.digitize`` on an array equals
    the per-scalar call element-for-element, and the trig is the same formula.
      * ``los_cos``  : tracking_cos per lane (float64) — the summed quantity.
      * ``los_bin``  : its _FIRE_ANGLE_LABELS bucket index (crosshair→origin angle).
      * ``turn_bin`` : the decoded look's turn-magnitude _TURN_MAG_LABELS bucket.
    """
    los_cos = np.array([float(i.get("tracking_cos", 0.0)) for i in infos], dtype=np.float64)
    los_ang = np.degrees(np.arccos(np.clip(los_cos, -1.0, 1.0)))
    los_bin = np.minimum(np.digitize(los_ang, _FIRE_ANGLE_EDGES_DEG, right=False),
                         len(_FIRE_ANGLE_LABELS) - 1)
    look = np.array([a["look"] for a in actions], dtype=np.float64).reshape(-1, 3)
    ln = np.linalg.norm(look, axis=-1)
    u0 = np.ones_like(ln)
    _m = ln > 1e-8
    u0[_m] = look[_m, 0] / ln[_m]
    turn = np.degrees(np.arccos(np.clip(u0, -1.0, 1.0)))
    turn_bin = np.minimum(np.digitize(turn, _TURN_MAG_EDGES_DEG, right=False),
                          len(_TURN_MAG_LABELS) - 1)
    return los_cos, los_bin, turn_bin


_ARENA_INV_INT_KEYS = ("shells", "nails", "rockets", "cells", "infinite_ammo",
                       "health", "armor_value")


def _arena_weapon_config(options: Mapping[str, object]) -> dict[str, object]:
    """The single-weapon arena loadout (model weapon, bot pin, ammo/armor/health)
    extracted from a scenario's options, in the shape ``ArenaServerProcess``
    turns into engine CLI. Fails loud if the inventory owns more than one weapon
    (the arena is homogeneous single-weapon by construction). Returns ``{}`` when
    the scenario specifies no inventory (free-play arena keeps engine defaults)."""
    inv = options.get("inventory") or {}
    if not isinstance(inv, MappingABC):
        raise ValueError("arena_grid scenario inventory must be a mapping")
    cfg: dict[str, object] = {}
    sel = inv.get("selected_weapon")
    weps = list(inv.get("weapons") or [])
    if sel is not None or weps:
        if weps and (len(weps) != 1 or (sel is not None and str(weps[0]) != str(sel))):
            raise ValueError(
                "arena_grid is single-weapon; a cell may own exactly its selected "
                f"weapon, got weapons={weps} selected_weapon={sel}")
        cfg["model_weapon"] = str(sel if sel is not None else weps[0])
    pin = options.get("bot_weapon_pin")
    if pin:
        cfg["bot_weapon_pin"] = str(pin)
    for key in _ARENA_INV_INT_KEYS:
        value = inv.get(key)
        if value is not None:
            cfg[key] = int(value)
    if inv.get("armor_type") is not None:
        cfg["armor_type"] = float(inv["armor_type"])
    return cfg


def _evaluate_mode(
    config: EvalConfig,
    model: QNNPolicy,
    mode: str,
    episode_specs: Sequence[Tuple[int, int | None]],
) -> Dict[str, float]:
    del episode_specs
    num_envs = max(1, min(config.num_envs, config.num_episodes))
    move_streams: Dict[str, np.ndarray] = {}
    # acquisition obs streams (log_acq_streams): per-episode concatenated entity
    # arrays keyed by episode index, plus a per-episode scenario tag so the
    # per-cell acquisition band falls out of a heterogeneous batched eval.
    acq_streams: Dict[str, np.ndarray] = {}
    acq_scenarios: Dict[str, str] = {}
    scenarios, job_queues = _episode_jobs(config)
    arena_pool = None
    if config.env_backend == "arena_grid":
        if not scenarios:
            raise ValueError("arena_grid evaluation requires at least one scenario")
        for scenario in scenarios:
            if scenario.procgen_cfg is not None or scenario.map_id != config.arena_map_id:
                raise ValueError(
                    "arena_grid evaluation requires non-procgen scenarios whose "
                    f"map_id is {config.arena_map_id!r}"
                )
        # PER-CELL packing (aim grid): each per_env_decode_overrides lane is a
        # distinct cell (one swept decode value), so it needs one scenario per
        # lane and its own scenario_id for the per-scenario intercept ruler. The
        # arena is single-weapon, so every cell must share ONE loadout — only the
        # per-lane decode override varies. Plain single-scenario arenas (free-play
        # stage-6 eval) keep one scenario broadcast across all lanes.
        per_cell = config.per_env_decode_overrides is not None
        if per_cell:
            if len(scenarios) != num_envs:
                raise ValueError(
                    "arena_grid with per_env_decode_overrides requires one "
                    f"scenario cell per lane (scenarios={len(scenarios)}, "
                    f"num_envs={num_envs})")
            scenario_ids = [s.scenario_id for s in scenarios]
        else:
            if len(scenarios) != 1:
                raise ValueError(
                    "arena_grid without per_env_decode_overrides requires exactly "
                    "one scenario (all lanes share it)")
            scenario_ids = None
        weapon_configs = [_arena_weapon_config(s.options) for s in scenarios]
        weapon_config = weapon_configs[0]
        if any(wc != weapon_config for wc in weapon_configs[1:]):
            raise ValueError(
                "arena_grid scenarios must share ONE weapon config (single-weapon "
                "arena); got heterogeneous loadouts across cells")
        if not config.arena_server_binary or not config.arena_client_binary:
            raise ValueError("arena_grid evaluation requires arena server/client binaries")
        from qnn.env.reward import RewardWeights
        from qnn.eval.arena_backend import ArenaEvalPool

        basedir = config.native_env.get("QUAKE_BASEDIR", "")
        if not basedir:
            raise ValueError("arena_grid evaluation requires native_env.QUAKE_BASEDIR")
        if not 1 <= config.arena_matches_per_server <= 8:
            raise ValueError("arena_matches_per_server must be in [1, 8]")
        matches_per_server = min(config.arena_matches_per_server, num_envs)
        while num_envs % matches_per_server:
            matches_per_server -= 1
        arena_pool = ArenaEvalPool(
            num_lanes=num_envs,
            server_executable=config.arena_server_binary,
            client_executable=config.arena_client_binary,
            basedir=basedir,
            workdir=config.native_workdir or None,
            map_id=config.arena_map_id,
            matches_per_server=matches_per_server,
            base_port=config.arena_base_port,
            bot_skill=config.arena_bot_skill,
            max_steps_per_episode=config.max_steps_per_episode,
            fixed_tick_hz=config.fixed_tick_hz,
            reward_weights=RewardWeights.from_json(config.reward_json_path),
            scenario_id=scenarios[0].scenario_id,
            scenario_ids=scenario_ids,
            weapon_config=weapon_config or None,
            declarations=(
                [_declaration_for_checkpoint(config.checkpoint_path)] * num_envs
            ),
        )
    elif config.env_backend != "process":
        raise ValueError(f"unknown eval env_backend {config.env_backend!r}")
    executor = (
        ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="nq-eval")
        if arena_pool is None and num_envs > 1 else None
    )

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
    # Per-weapon attack decomposition (impulse 1..8 = the decoded attack-with
    # weapon, emitted every frame). Splits the aggregate obs_attack_rate into
    # weapon occupancy vs per-weapon attack propensity — catches a low aggregate
    # that is really a weapon-mix / per-weapon-calibration artifact, not a
    # uniform under-attack. Keyed by impulse; named in the summary.
    weapon_attack_ticks: dict[int, int] = {}
    weapon_held_ticks: dict[int, int] = {}
    # obs-feeder sanity: capture the raw attack_finished the model receives
    # closed-loop, to compare against the offline corpus distribution (catches
    # a live obs-assembly bug that offline (cached corpus) can't show).
    _dbg_af: list[float] = []
    _dbg_obs_keys: list[str] | None = None
    # LEAD-CORRECTED blind-attack counters (info-side, LOS-gated). The obs-side
    # offcone above uses the raw crosshair→actor-ORIGIN angle, which counts a
    # correct aim-ahead/tracking shot as "blind"; these use the crosshair→
    # LEAD-POINT angle (the same shared kernel as engine_lead_cos) so a leading
    # shot is scored aligned. no_in_los = attacked with no in-LOS actor
    # (lead_valid=0); offcone_lead = in-LOS actor but the lead-point angle > cone.
    attack_disc_ticks = 0
    blind_attack_no_los = 0
    blind_attack_offcone_lead = 0
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
    # Per-weapon lead-angle buckets (raw engine weapon id → name): closes the
    # loop with the fit's per-weapon coh targets (RL included since 7/07 —
    # its feet-anchored pitch is part of the coh fit, not a separate check).
    _LEAD_WEAPON_NAMES = {1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
                          6: "GL", 7: "RL", 8: "LG"}
    lead_weapon_buckets: Dict[str, list] = {}
    # per-weapon turn-by-LOS (held-keyed — MUST match the human reference's
    # keying; the decode may key on intent, the RULER stays held): the
    # per-weapon EMD input for the (gain, alpha) split optimizer.
    los_turn_hist_w: Dict[str, list] = {}
    lead_fire_bucket = [0] * len(_FIRE_ANGLE_LABELS)
    # Discharge-anchored INTERCEPT histograms (hbw): operative-fire ticks binned into
    # _INTERCEPT_HBW_EDGES — aggregate + per (lead-)weapon + per scenario cell. The
    # a25 aim-skill lever's MODEL-SIDE ruler (§14), scored on the human per-weapon
    # log-normal ladder. Purely additive to the coh (lead-angle) buckets above.
    intercept_hbw_hist = [0] * len(_INTERCEPT_HBW_LABELS)
    intercept_hbw_w: Dict[str, list] = {}
    scenario_intercept_hbw: Dict[str, List[int]] = {}
    # Per-discharge INTERCEPT EVENT rows (decode-fit v2 P1: every operative
    # discharge is one likelihood sample). The SAME (scenario, weapon, hbw,
    # range) the histograms above bin, kept UNBINNED — the histograms censor
    # the tail and collapse to cell medians, which is exactly what starved the
    # v1 grids — plus (env_idx, episode, tick) provenance so response.py can
    # pool raw events and bootstrap by episode cluster. Column-parallel append
    # lists (schema pinned in qnn.decode_fit.events), written to
    # output_dir/intercept_events.npz at summary time via
    # _write_intercept_events. Always on: rows are 4 scalars + 2 short strings
    # per discharge, capped at _INTERCEPT_EVENT_CAP (histograms keep running
    # past the cap; the npz is then stamped truncated=True).
    intercept_events: Dict[str, list] = {k: [] for k in _EVENT_RAW_FIELDS}
    intercept_events_truncated = False
    # DIAGNOSTIC (env-gated, additive): per-discharge hbw window over
    # [t0-pre, t0+post] — the crest-vs-fired timing analysis (does the model
    # fire at its local alignment crest the way the elite-anchor humans do?)
    # and the alignment-LEAD profile (how far ahead of the trigger does the
    # crosshair converge?), which is why the pre-roll is the long side.
    # Env-gated so wave config hashes are untouched; emitted as
    # intercept_windows.npz. QNN_EVAL_INTERCEPT_WINDOW is "<pre>,<post>";
    # a single int is the symmetric shorthand "<k>,<k>".
    _win_pre, _win_post = _parse_intercept_window(
        os.environ.get("QNN_EVAL_INTERCEPT_WINDOW", ""))
    _win_rows: Dict[str, list] = {"scenario_id": [], "weapon": [],
                                  "episode": [], "env_idx": [], "tick": [],
                                  "hbw_win": []}

    def _win_emit(_buf: list, pad_to: int | None = None) -> None:
        _sc, _wn, _ep, _ei, _tk = _buf[0]
        row = _buf[1:]
        if pad_to is not None:
            row = row + [float("nan")] * (pad_to - len(row))
        _win_rows["scenario_id"].append(_sc)
        _win_rows["weapon"].append(_wn)
        _win_rows["episode"].append(_ep)
        _win_rows["env_idx"].append(_ei)
        _win_rows["tick"].append(_tk)
        _win_rows["hbw_win"].append(row)
    _win_trail: Dict[tuple, deque] = {}          # lane key → trailing hbw deque
    _win_pending: Dict[tuple, list] = {}         # lane key → [bufs to fill fwd]
    _WIN_CAP = 100_000
    # Per-LANE episode ordinal (0-based) — the event log's bootstrap-cluster
    # key alongside env_idx. No such counter exists elsewhere in the loop
    # (state.episode_index is the GLOBAL job ordinal; scenario_episode_counts
    # keys on scenario, and a scenario can in principle span lanes), so it is
    # derived here: Pass A bumps a lane's slot on each terminal it sees.
    env_episode_ord: Dict[int, int] = {}
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

    # PER-LANE decode pinning (per_env_decode_overrides): when set, each env slot
    # carries a fixed decode operating point (one grid cell), so a scenario's
    # episodes must ALL run on that scenario's own lane — a lane may never steal
    # another lane's scenario, or its cell would be measured at the wrong swept
    # value. strict_lane forces lane i ↔ scenario i and disables the general
    # work-steal fallback. None (default) keeps the original free-steal fill.
    strict_lane = config.per_env_decode_overrides is not None
    if strict_lane and num_envs != len(scenarios):
        raise ValueError(
            "per_env_decode_overrides requires one env slot per scenario "
            f"(num_envs={num_envs}, scenarios={len(scenarios)}); the aim-grid "
            "packer sets eval_num_envs == number of packed cells")

    def _next_job(preferred_scenario_id: str | None = None,
                  strict: bool = False) -> _EpisodeJob | None:
        if preferred_scenario_id:
            preferred = job_queues.get(preferred_scenario_id)
            if preferred:
                return preferred.popleft()
        if strict:
            return None            # never steal another lane's scenario
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

    def _reset_job(idx: int, job: _EpisodeJob):
        if arena_pool is not None:
            idx_scenarios[idx] = job.scenario
            return arena_pool.reset_lane(idx)
        return _ensure_env(idx, job.scenario).reset(
            seed=job.episode_seed, start_variant=job.start_variant
        )

    try:
        active: Dict[int, _EpisodeState] = {}
        for idx in range(num_envs):
            # strict_lane: pin scenario idx → lane idx so the lane's decode
            # override (per_env_decode_overrides[idx]) matches the cell it runs.
            job = (_next_job(scenarios[idx].scenario_id, strict=True)
                   if strict_lane else _next_job())
            if job is None:
                break
            obs = _reset_job(idx, job)
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
            _lane = len(active)
            active[_lane] = _EpisodeState(
                episode_index=job.episode_index,
                obs=obs,
                scenario_id=job.scenario.scenario_id,
                rng=None if mode == "greedy" else _episode_rng(config, mode, job.episode_index, model.device),
                attack_rng=_seed_attack_rng(job.episode_seed),
                hidden=model.zero_hidden(1)[0].copy(),
                decode_override=_lane_decode_override(config, _lane),
            )
            _prime_obs_delay(active[_lane], config.obs_lag_ticks)

        select_actions = (
            _select_actions_batched if config.batched_forward else _select_actions_batch
        )
        # Batched sampled forward draws every per-row uniform from ONE run-scoped
        # generator (wrapped per tick in a BatchedRNG sized to the active count),
        # so the sampling path is a single vectorized dispatch instead of the
        # O(lanes) per-generator loop. Deterministically seeded; unused by the B=1
        # path (which keeps its per-episode generators). Greedy needs no RNG.
        batched_rng_gen: torch.Generator | None = None
        if config.batched_forward and mode == "sampled":
            _gdev = model.device if model.device.type in {"cpu", "cuda"} else torch.device("cpu")
            batched_rng_gen = torch.Generator(device=_gdev)
            batched_rng_gen.manual_seed(config.seed + config.sample_seed_offset)

        # ── DEFERRED-METRIC OVERLAP (batched path) ────────────────────────────
        # The per-tick metric accumulation is engine-INDEPENDENT: the vectorized
        # lead-aim ruler + LOS/turn binning + the per-lane metric fold read the
        # step results + actions, not the envs. In the batched path we SUBMIT every
        # lane's action (all engines sim the tick concurrently), run the PREVIOUS
        # tick's metric pass WHILE the engines sim, then RECEIVE — so the residual
        # serial Python overlaps engine compute and the wave becomes engine-bound.
        # The pre-step stream logging + obs-side fire discrimination read the
        # PRE-step obs, so they run in the same overlap window (this tick). Only
        # "Pass A" (state advance / episode budgeting / lane resets / per-scenario
        # terminal emission) must stay inline after receive: the next
        # select_actions needs the advanced obs+hidden and the updated `active`
        # set, so Pass A cannot be deferred. The final tick's deferred metrics are
        # flushed after the loop (no dropped last tick). Non-batched paths have no
        # submit/receive split, so they fold metrics inline (no overlap, identical
        # results). These are nested closures so the (many) accumulators stay in
        # one scope; the moved code is byte-for-byte the former inline blocks.
        # band-v5 gate stream fields (discharge / weapon_imp / engaged) need the
        # engine-fed obs scalars; decided once on the first logged tick. When the
        # obs lacks them the flat keys are OMITTED so the band scorer fails loud
        # (pre-v5 npz) instead of scoring an all-zeros discharge stream.
        _v5_gate_fields = {"ok": None}
        # LOOK COMMIT trace (gate schema 3): the LOOK_COMMIT_STATE_DIM lanes
        # policy.act's look-commitment decode just left in state.look_commit —
        # logged tick-aligned with the gate row so stroke class/direction sit on
        # the same index as fb/lr/turn_deg (the cross-head coordination channel;
        # lc_elapsed == 0 marks a stroke onset). A model without a look_seg head
        # never runs that decode, so its lanes are the -1 sentinel, never the
        # reset zeros (which are a REAL state: hold, onset due next tick).
        _has_look_commit = bool(getattr(model, "look_commitment", False))
        _lc_absent = (-1,) * LOOK_COMMIT_STATE_DIM

        def _log_streams(states, actions):
            if config.log_action_streams:
                for _i, _st in enumerate(states):
                    mv = actions[_i]["move"]
                    _st.move_trace.append(
                        (int(mv[0]) + 1, int(mv[1]) + 1, int(mv[2]) + 1))
                    # threat flags for the closed-loop PSTH: damage = health
                    # dropped since last tick; projectile = projectile token
                    # count increased. Same trigger defs as _move_threat_psth.
                    _flags = 0
                    _hsrc = _st.obs.get("health") if isinstance(_st.obs, dict) else None
                    _h = (float(np.asarray(_hsrc).reshape(-1)[0]) if _hsrc is not None
                          else float(_st.last_info.get("health", 1e9) or 1e9))
                    if _h < _st._last_health - 1e-6:
                        _flags |= 1
                    _st._last_health = _h
                    if isinstance(_st.obs, dict) and "entity_types" in _st.obs:
                        from qnn.vocab import (TOKEN_PROJECTILE as _TP,
                                               OWN_FIRE_DIST_U as _OFD)
                        _et = np.asarray(_st.obs["entity_types"]).reshape(-1)
                        _pm = _et == _TP
                        _np_ = int(_pm.sum())
                        # bit1 = INCOMING projectile appeared: count increased
                        # AND nearest projectile beyond the own-fire gate (own
                        # rockets spawn <100u; same definition as the human
                        # reference — qnn.vocab.OWN_FIRE_DIST_U).
                        _incoming = False
                        if _np_ and "entity_rel" in _st.obs:
                            # env obs is FIELD-GRANULAR: entity_rel (N,3) raw
                            # game units (the packed entity_scalars_raw only
                            # exists in the wire/model-input form).
                            _rel = np.asarray(
                                _st.obs["entity_rel"]).reshape(len(_et), -1)[:, 0:3]
                            _pd = np.linalg.norm(
                                _rel[_pm].astype(np.float64), axis=1)
                            _incoming = bool(_pd.size and _pd.min() > _OFD)
                        if _np_ > _st._last_nproj and _incoming:
                            _flags |= 2
                        # bit2 = incoming projectile PRESENT this tick (the
                        # threat-ACTIVE flag — per-tick hazard-ratio target).
                        if _incoming:
                            _flags |= 4
                        _st._last_nproj = _np_
                    _st.threat_trace.append(_flags)
                    # stage-6 gate channels (rc_humanlikeness flat schema):
                    # attack, weapon, per-tick turn magnitude, keep=actor visible.
                    _act = actions[_i]
                    _lk = _act.get("look")
                    if _lk is not None:
                        _lx, _ly, _lz = float(_lk[0]), float(_lk[1]), float(_lk[2])
                        # θ = atan2(|yz|, x) and the TANGENT z = θ·ŷz — the
                        # corpus act_look_tan law, verbatim (see
                        # qnn.bc.cache_look_tan.look_to_tangent /
                        # qnn.model.look_bins.tangent_logmap; NEVER arccos(x),
                        # which manufactures holds below ~1.27°). turn_deg is
                        # |z| in degrees; the tangent carries the direction the
                        # segmenter needs. Pinned against the corpus law by
                        # tests/test_h2h.py.
                        _hyz = float(np.hypot(_ly, _lz))
                        _th = float(np.arctan2(_hyz, _lx))
                        _turn = float(np.degrees(_th))
                        _tsc = (_th / _hyz) if _hyz > 0.0 else 0.0
                        _tan_y, _tan_z = _ly * _tsc, _lz * _tsc
                    else:
                        _turn = 0.0
                        _tan_y, _tan_z = 0.0, 0.0
                    _keep = 0
                    _engaged = 0
                    if isinstance(_st.obs, dict) and "entity_types" in _st.obs:
                        _et_g = np.asarray(_st.obs["entity_types"]).reshape(-1)
                        _keep = int((_et_g == TOKEN_ACTOR).any())
                        # engaged (band v5 context mask) = SIGHT actor among
                        # the live entity slots. A27 derives this directly
                        # from modality; legacy observations fall back to
                        # recency==0.
                        if "entity_count" in _st.obs:
                            _n_g = int(np.asarray(_st.obs["entity_count"]).reshape(-1)[0])
                            if "entity_modality_id" in _st.obs:
                                _mod_g = np.asarray(
                                    _st.obs["entity_modality_id"]
                                ).reshape(-1)[:_n_g]
                                _engaged = int(((_et_g[:_n_g] == TOKEN_ACTOR)
                                                & (_mod_g == 0)).any())
                            elif "entity_recency" in _st.obs:
                                _rec_g = np.asarray(
                                    _st.obs["entity_recency"],
                                    dtype=np.float64,
                                ).reshape(-1)[:_n_g]
                                _engaged = int(((_et_g[:_n_g] == TOKEN_ACTOR)
                                                & (_rec_g <= 0.0)).any())
                    if "weapon" in _act:
                        # a26 action convention: attack is the fire BIT and
                        # the per-tick weapon INTENT rides its own channel;
                        # the discharging weapon is the HELD one
                        # (self_weapon_id), mirroring the a26-line gate row.
                        _attack_g = int(int(np.asarray(
                            _act["attack"]).reshape(-1)[0]) > 0)
                        _wsel_g = int(np.asarray(
                            _act["weapon"]).reshape(-1)[0])
                        _wimp_g = 0
                        if isinstance(_st.obs, dict) and "self_weapon_id" in _st.obs:
                            _wimp_g = int(self_weapon_id_to_impulse(int(np.asarray(
                                _st.obs["self_weapon_id"]).reshape(-1)[0])))
                    else:
                        # a27 convention: attack IS the attack_with class
                        # (0 hold, 1..8 weapon impulse being fired).
                        _wimp_g = int(_act.get("attack", 0) or 0)
                        _attack_g = int(_wimp_g > 0)
                        _wsel_g = _wimp_g
                    # engine-visible DISCHARGE (band v5 attack channel): the
                    # engine honors an attack only when QC attack_finished has
                    # expired — the mirror of the human op-attack decision
                    # frame (move bit0 & input_mask bit0), chain-bolt-free and
                    # cooldown-gated on both sides.
                    _af_g = (_st.obs.get("attack_finished")
                             if isinstance(_st.obs, dict) else None)
                    _disch_g = int(bool(_attack_g) and _af_g is not None
                                   and float(np.asarray(_af_g).reshape(-1)[0]) <= 1e-6)
                    if _v5_gate_fields["ok"] is None:
                        _v5_gate_fields["ok"] = bool(
                            isinstance(_st.obs, dict)
                            and all(k in _st.obs for k in (
                                "attack_finished",
                                "entity_types", "entity_count"))
                            and ("entity_modality_id" in _st.obs
                                 or "entity_recency" in _st.obs))
                    if _has_look_commit:
                        _lc_g = tuple(int(v) for v in np.asarray(
                            _st.look_commit).reshape(-1)[:LOOK_COMMIT_STATE_DIM])
                    else:
                        _lc_g = _lc_absent
                    _st.gate_trace.append(
                        (_attack_g, _wsel_g, _turn,
                         _keep, _disch_g, _wimp_g, _engaged) + _lc_g
                        + (_tan_y, _tan_z))

            # ACQUISITION obs capture: per-tick RAW entity obs (the SAME fields
            # the human collect cache feeds the acq_submovement flick kernel —
            # entity_rel/vel int16 game units, entity_types int8, recency float16,
            # entity arrays are sliced to the live count so padding never
            # leaks in). The kernel derives view-motion + nearest-actor origin
            # bearing from these, so no geometry is duplicated here.
            if config.log_acq_streams:
                for _i, _st in enumerate(states):
                    _o = _st.obs
                    if not (isinstance(_o, Mapping) and "entity_count" in _o):
                        _st.acq_trace.append((0, np.zeros(0, np.int8),
                                              np.zeros((0, 3), np.int16),
                                              np.zeros((0, 3), np.int16),
                                              np.zeros(0, np.float16), 0))
                        continue
                    _n = int(np.asarray(_o["entity_count"]).reshape(-1)[0])
                    if "entity_recency" in _o:
                        _acq_rec = np.asarray(
                            _o["entity_recency"]
                        ).reshape(-1)[:_n].astype(np.float16)
                    else:
                        # The acquisition stream's legacy `rec` lane is an
                        # analysis-only LOS mask. A27 supplies it from modality:
                        # zero for SIGHT, a large finite sentinel for PROXIMITY.
                        _mod = np.asarray(
                            _o["entity_modality_id"]
                        ).reshape(-1)[:_n]
                        _acq_rec = np.where(
                            _mod == 0, 0.0, np.finfo(np.float16).max
                        ).astype(np.float16)
                    _st.acq_trace.append((
                        _n,
                        np.asarray(_o["entity_types"]).reshape(-1)[:_n].astype(np.int8),
                        np.asarray(_o["entity_rel"]).reshape(-1, 3)[:_n].astype(np.int16),
                        np.asarray(_o["entity_vel"]).reshape(-1, 3)[:_n].astype(np.int16),
                        _acq_rec,
                        int(actions[_i].get("attack", 0) or 0),
                    ))

        def _obs_fire_disc(states, actions):
            nonlocal fire_ticks, blind_no_actor, blind_offcone
            nonlocal fire_cos_sum, fire_cos_n, _dbg_obs_keys
            # obs-side fire discrimination: at each fire tick, score the model's
            # crosshair alignment to the nearest-aligned actor in its OWN obs.
            for _i, _st in enumerate(states):
                _w = int(actions[_i].get("attack", 0) or 0)
                if 1 <= _w <= 8:
                    weapon_held_ticks[_w] = weapon_held_ticks.get(_w, 0) + 1
                if _dbg_obs_keys is None and isinstance(_st.obs, dict):
                    _dbg_obs_keys = sorted(_st.obs.keys())
                if isinstance(_st.obs, dict) and "attack_finished" in _st.obs:
                    try:
                        _dbg_af.append(float(np.asarray(_st.obs["attack_finished"]).reshape(-1)[0]))
                    except Exception:
                        pass
                if _w <= 0:
                    continue
                fire_ticks += 1
                if 1 <= _w <= 8:
                    weapon_attack_ticks[_w] = weapon_attack_ticks.get(_w, 0) + 1
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

        def _accumulate_metrics(infos, actions, scenario_ids, env_idxs,
                                episode_ords, tick):
            """Deferred per-tick metric pass (row-aligned infos/actions/
            scenario_ids/env_idxs/episode_ords; ``tick`` = the macro-step these
            rows were captured on).

            The engine-independent half of the former per-lane loop: the batched
            lead-aim ruler + LOS/turn binning computed ONCE, then the per-lane
            metric fold. Runs during the NEXT tick's engine sim (batched) or
            inline (non-batched). Order-independent sums/counts/histograms +
            append-only event rows, so deferring one tick leaves every
            aggregate + npz stream identical.
            """
            nonlocal total_steps, los_n, los_cos_sum, los_cos2_sum, los_fire
            nonlocal los_cosfire_sum, stuck_steps, lead_n, lead_cos_sum, lead_pitch_sum
            nonlocal lead_cosfire_sum, lead_pitch_fire_sum, lead_fire_n
            nonlocal lead_pitch_fire_rl_sum, lead_fire_rl_n
            nonlocal attack_disc_ticks, blind_attack_no_los, blind_attack_offcone_lead
            nonlocal intercept_events_truncated
            # BATCHED per-tick rulers (computed once over all lanes instead of one
            # torch/np round-trip per lane — the serial Python that capped core
            # utilization). Row i ↔ infos[i] ↔ actions[i] ↔ scenario_ids[i]
            # ↔ env_idxs[i] ↔ episode_ords[i].
            _lead_batch = _lead_aim_cos_batched(infos)
            _los_cos_a, _los_bin_a, _turn_bin_a = _batched_los_turn_bins(infos, actions)
            for batch_idx, info in enumerate(infos):
                scenario_id = scenario_ids[batch_idx]
                total_steps += 1
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
                _los_cos = float(_los_cos_a[batch_idx])
                _los_fired = int(info.get("shots_fired", 0)) > 0
                los_n += 1
                los_cos_sum += _los_cos
                los_cos2_sum += _los_cos * _los_cos
                _lb = int(_los_bin_a[batch_idx])
                los_tick_bucket[_lb] += 1
                _sc_los[_lb] += 1
                # decoded turn magnitude this tick (view-relative look vector,
                # forward=+x ⇒ turn = arccos(look_x/|look|)) → per-LOS-zone hist.
                # _turn_bin_a was computed once for all lanes above.
                _lk = actions[batch_idx].get("look") if isinstance(actions[batch_idx], Mapping) else None
                if _lk is not None:
                    _tb = int(_turn_bin_a[batch_idx])
                    los_turn_hist[_lb][_tb] += 1
                    _twn = _LEAD_WEAPON_NAMES.get(int(info.get("lead_weapon_id", 0)))
                    if _twn is not None:
                        _twh = los_turn_hist_w.setdefault(
                            _twn, [[0] * len(_TURN_MAG_LABELS)
                                   for _ in range(len(_FIRE_ANGLE_LABELS))])
                        _twh[_lb][_tb] += 1
                if _los_fired:
                    los_fire += 1
                    los_cosfire_sum += _los_cos
                    los_fire_bucket[_lb] += 1
                # LEAD-POINT-referenced coherence: same operative-fire selection,
                # but the angle is to the velocity-led intercept (ground anchor
                # for RL) via the shared lead kernel — comparable to the human
                # coh_5deg curve. Only on ticks where an actor participated.
                _lead = _lead_batch[batch_idx]
                # LEAD-CORRECTED blind-attack discrimination: on attack ticks,
                # classify against the crosshair→lead-point angle (not the raw
                # crosshair→origin angle the obs-side offcone uses), so a correct
                # aim-ahead shot is not miscounted as blind. LOS-gated: lead_valid=0
                # (no in-LOS actor participating) is the "no actor" analog.
                _attacked = (int(actions[batch_idx].get("attack", 0) or 0)
                             if isinstance(actions[batch_idx], Mapping) else 0)
                if _attacked:
                    attack_disc_ticks += 1
                    if _lead is None:
                        blind_attack_no_los += 1
                    elif float(np.degrees(np.arccos(np.clip(_lead[0], -1.0, 1.0)))) > _FIRE_CONE_DEG:
                        blind_attack_offcone_lead += 1
                if _win_pre or _win_post:
                    # per-tick hbw stream for the crest window (NaN when no
                    # in-LOS actor); trailing buffer + forward fills per lane
                    _wkey = (int(env_idxs[batch_idx]),
                             int(episode_ords[batch_idx]))
                    if _lead is not None:
                        _wl_cos, _, _, _wl_rng = _lead
                        _wl_ang = float(np.degrees(np.arccos(
                            np.clip(_wl_cos, -1.0, 1.0))))
                        _hbw_t = _intercept_hbw(_wl_ang, _wl_rng)
                    else:
                        _hbw_t = float("nan")
                    _tr = _win_trail.get(_wkey)
                    if _tr is None:
                        _tr = _win_trail[_wkey] = deque(maxlen=_win_pre + 1)
                    _tr.append(_hbw_t)
                    _pend = _win_pending.get(_wkey)
                    if _pend:
                        _done_bufs = []
                        for _buf in _pend:
                            _buf.append(_hbw_t)
                            # 1 meta tag + (pre+1) trailing + post forward
                            if len(_buf) == _win_pre + _win_post + 2:
                                _done_bufs.append(_buf)
                        for _buf in _done_bufs:
                            _pend.remove(_buf)
                            _win_emit(_buf)
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
                    _wname = _LEAD_WEAPON_NAMES.get(int(info.get("lead_weapon_id", 0)))
                    if _wname is not None:
                        _wb = lead_weapon_buckets.setdefault(
                            _wname, [0] * len(_FIRE_ANGLE_LABELS))
                        _wb[_ldb] += 1
                    if _los_fired:
                        lead_cosfire_sum += _lead_cos
                        lead_pitch_fire_sum += _feet_pitch
                        lead_fire_n += 1
                        lead_fire_bucket[_ldb] += 1
                        # discharge-anchored INTERCEPT: this operative fire's
                        # alignment in hitbox-half-widths (§14 model-side ruler).
                        _hbw = _intercept_hbw(_lead_ang, _range_u)
                        _hb = min(int(np.digitize(_hbw, _INTERCEPT_HBW_EDGES, right=False)),
                                  len(_INTERCEPT_HBW_LABELS) - 1)
                        intercept_hbw_hist[_hb] += 1
                        scenario_intercept_hbw.setdefault(
                            scenario_id, [0] * len(_INTERCEPT_HBW_LABELS))[_hb] += 1
                        if _wname is not None:
                            intercept_hbw_w.setdefault(
                                _wname, [0] * len(_INTERCEPT_HBW_LABELS))[_hb] += 1
                            # per-discharge EVENT row (decode-fit v2 P1): the
                            # same discharge the two histograms above just
                            # binned, kept unbinned with lane/episode/tick
                            # provenance. Gated on _wname exactly like the
                            # per-weapon histogram — an unknown lead_weapon_id
                            # has no ladder to be scored on. Past the cap the
                            # rows stop (histograms keep running) and the npz
                            # is stamped truncated=True at write time.
                            if len(intercept_events["hbw"]) < _INTERCEPT_EVENT_CAP:
                                intercept_events["scenario_id"].append(scenario_id)
                                intercept_events["weapon"].append(_wname)
                                intercept_events["hbw"].append(float(_hbw))
                                intercept_events["range_u"].append(float(_range_u))
                                intercept_events["episode"].append(int(episode_ords[batch_idx]))
                                intercept_events["env_idx"].append(int(env_idxs[batch_idx]))
                                intercept_events["tick"].append(int(tick))
                            else:
                                intercept_events_truncated = True
                            if ((_win_pre or _win_post)
                                    and len(_win_rows["hbw_win"]) < _WIN_CAP):
                                # crest-window row: identity meta + trailing
                                # (pre+1, NaN-padded left) — the forward `post`
                                # ticks fill on this lane's next ticks. The meta
                                # (scenario/episode/lane/tick) makes the npz
                                # self-contained for the decode-fit tracking
                                # loader (cluster keys + op annotation),
                                # mirroring intercept_events row identity.
                                _twin = list(_win_trail.get(
                                    (int(env_idxs[batch_idx]),
                                     int(episode_ords[batch_idx])), ()))
                                _twin = ([float("nan")] *
                                         (_win_pre + 1 - len(_twin))) + _twin
                                _win_pending.setdefault(
                                    (int(env_idxs[batch_idx]),
                                     int(episode_ords[batch_idx])), []
                                ).append([(scenario_id, _wname,
                                           int(episode_ords[batch_idx]),
                                           int(env_idxs[batch_idx]),
                                           int(tick))] + _twin)
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
                for key, value in _iter_aux_metric_items(info, _AUX_INFO_KEYS):
                    aux_metric_sums[key] = aux_metric_sums.get(key, 0.0) + value
                    scenario_metric_sums = scenario_aux_metric_sums.setdefault(scenario_id, {})
                    scenario_metric_sums[key] = scenario_metric_sums.get(key, 0.0) + value

        _deferred: "tuple[list, list, list, list, list, int] | None" = None
        # Eval macro-step counter (one per while-iteration, i.e. per batched
        # act+step tick) — the event log's `tick` column. Distinct from
        # total_steps, which sums PER-LANE steps across the batch.
        macro_step = -1
        while active:
            macro_step += 1
            idx_ids = sorted(active.keys())
            states = [active[idx] for idx in idx_ids]
            actions, next_hidden = select_actions(
                model=model,
                mode=mode,
                states=states,
                batched_rng_gen=batched_rng_gen,
            )
            if arena_pool is not None:
                # ARENA backend: the grouped engine does its own server-side batched
                # step (arena_pool.step_many), so there is no split-phase submit /
                # receive overlap window. Do the pre-step logging + the previous
                # tick's deferred metric fold inline, then step synchronously. The
                # deferred pattern is kept (order-independent) for parity with the
                # process backend; it just loses the sim-overlap it can't have here.
                _log_streams(states, actions)
                _obs_fire_disc(states, actions)
                if _deferred is not None:
                    _accumulate_metrics(*_deferred)
                    _deferred = None
                results = arena_pool.step_many(idx_ids, actions)
            elif config.batched_forward:
                # SUBMIT — every engine starts simming this tick concurrently.
                _submit_batched_raw(envs, idx_ids, actions)
                # OVERLAP WINDOW (engines simming): this tick's pre-step stream
                # logging + obs-side fire discrimination (both read the PRE-step
                # obs), then the PREVIOUS tick's deferred metric accumulation.
                _log_streams(states, actions)
                _obs_fire_disc(states, actions)
                if _deferred is not None:
                    _accumulate_metrics(*_deferred)
                    _deferred = None
                # RECEIVE — drain + batch-unpack once the engines are done.
                results = _receive_batched_raw(envs, idx_ids)
            else:
                _log_streams(states, actions)
                _obs_fire_disc(states, actions)
                if executor is None:
                    results = [_step_env(envs[idx], action) for idx, action in zip(idx_ids, actions)]
                else:
                    futures = [
                        executor.submit(_step_env, envs[idx], action)
                        for idx, action in zip(idx_ids, actions)
                    ]
                    results = [future.result() for future in futures]

            # ── Pass A (inline; must precede the next select_actions) ──────────
            # State advance + episode budgeting / lane resets / per-scenario
            # terminal emission. Captures (info, action, scenario_id) per lane for
            # the deferred metric pass (_accumulate_metrics).
            _cap_infos: list = []
            _cap_actions: list = []
            _cap_scn: list = []
            # Event-log provenance (lane index + the lane's CURRENT episode
            # ordinal) captured here, AT EVENT TIME: the deferred fold runs a
            # tick later on the batched path, after this very block may have
            # terminated + reset the lane — reading env_episode_ord at fold
            # time would stamp first-tick-of-next-episode rows one episode
            # late. idx_ids also reshuffles as lanes retire, so batch_idx
            # alone can't recover the lane.
            _cap_env: list = []
            _cap_ep: list = []
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
                # scenario_id resolved up front (fallback to the lane's prior id)
                # so the deferred per-scenario buckets and the terminal emission
                # below key on the SAME value.
                scenario_id = str(info.get("scenario_id", state.scenario_id))
                state.scenario_id = scenario_id
                _cap_infos.append(info)
                _cap_actions.append(actions[batch_idx])
                _cap_scn.append(scenario_id)
                _cap_env.append(idx)
                _cap_ep.append(env_episode_ord.get(idx, 0))

                if terminal:
                    if config.log_action_streams and state.move_trace:
                        move_streams[f"ep_{state.episode_index:04d}"] = np.asarray(
                            state.move_trace, dtype=np.int8)
                        if state.threat_trace:
                            move_streams[f"threat_ep_{state.episode_index:04d}"] = np.asarray(
                                state.threat_trace, dtype=np.uint8)
                        if state.gate_trace:
                            move_streams[f"gate_ep_{state.episode_index:04d}"] = np.asarray(
                                state.gate_trace, dtype=np.float32)
                    if config.log_acq_streams and state.acq_trace:
                        # collect-cache layout: (T,) per-frame count + entity
                        # arrays concatenated over the episode (the same shape
                        # acq_submovement._episode_metrics reconstructs from
                        # entity_count.cumsum()).
                        _tr = state.acq_trace
                        _key = f"{state.episode_index:04d}"
                        acq_streams[f"acq_cnt_{_key}"] = np.asarray(
                            [r[0] for r in _tr], dtype=np.uint8)
                        acq_streams[f"acq_typ_{_key}"] = np.concatenate(
                            [r[1] for r in _tr]).astype(np.int8)
                        acq_streams[f"acq_rel_{_key}"] = np.concatenate(
                            [r[2] for r in _tr]).astype(np.int16)
                        acq_streams[f"acq_vel_{_key}"] = np.concatenate(
                            [r[3] for r in _tr]).astype(np.int16)
                        acq_streams[f"acq_rec_{_key}"] = np.concatenate(
                            [r[4] for r in _tr]).astype(np.float16)
                        acq_streams[f"acq_weapon_{_key}"] = np.asarray(
                            [r[5] for r in _tr], dtype=np.uint8)
                        acq_scenarios[_key] = str(state.scenario_id)
                    returns.append(float(state.return_value))
                    end_health.append(int(state.last_info.get("health", 0)))
                    end_armor.append(int(state.last_info.get("armor", 0)))
                    done_reason = str(state.last_info.get("done_reason", "")).strip() or "unknown"
                    done_reasons[done_reason] = done_reasons.get(done_reason, 0) + 1
                    scenario_done_reason_counts = scenario_done_reasons.setdefault(state.scenario_id, {})
                    scenario_done_reason_counts[done_reason] = scenario_done_reason_counts.get(done_reason, 0) + 1
                    scenario_episode_counts[state.scenario_id] = scenario_episode_counts.get(state.scenario_id, 0) + 1
                    # per-LANE episode ordinal: whatever runs on this lane
                    # next (post-reset) is its next episode. Bumped on EVERY
                    # terminal (even when the lane retires) so the count stays
                    # "terminations seen", not "episodes restarted".
                    env_episode_ord[idx] = env_episode_ord.get(idx, 0) + 1
                    scenario_returns.setdefault(state.scenario_id, []).append(float(state.return_value))
                    episode_stats = state.metrics.as_dict()
                    append_metric_values(episode_metric_values, episode_stats)
                    append_metric_values(
                        scenario_episode_metric_values.setdefault(state.scenario_id, {}),
                        episode_stats,
                    )
                    for key, value in _iter_aux_metric_items(
                        state.last_info,
                        _EPISODE_AUX_KEYS,
                    ):
                        episode_metric_values.setdefault(key, []).append(value)
                        scenario_episode_metric_values.setdefault(state.scenario_id, {}).setdefault(key, []).append(value)

                    next_job = _next_job(idx_scenarios[idx].scenario_id,
                                         strict=strict_lane)
                    if next_job is not None:
                        next_obs = _reset_job(idx, next_job)
                        active[idx] = _EpisodeState(
                            episode_index=next_job.episode_index,
                            obs=next_obs,
                            scenario_id=next_job.scenario.scenario_id,
                            rng=None if mode == "greedy" else _episode_rng(config, mode, next_job.episode_index, model.device),
                            attack_rng=_seed_attack_rng(next_job.episode_seed),
                            hidden=model.zero_hidden(1)[0].copy(),
                            decode_override=_lane_decode_override(config, idx),
                        )
                        _prime_obs_delay(active[idx], config.obs_lag_ticks)
                    else:
                        del active[idx]
                else:
                    _advance_obs_delay(state, obs, config.obs_lag_ticks)

            # Dispatch this tick's metric pass. Batched: DEFER so it overlaps the
            # NEXT submit's engine sim (flushed after the loop for the final tick).
            # Non-batched: fold inline (no submit/receive split to overlap).
            if config.batched_forward:
                _deferred = (_cap_infos, _cap_actions, _cap_scn,
                             _cap_env, _cap_ep, macro_step)
            else:
                _accumulate_metrics(_cap_infos, _cap_actions, _cap_scn,
                                    _cap_env, _cap_ep, macro_step)

        # Flush the final tick's deferred metrics (no dropped last tick).
        if _deferred is not None:
            _accumulate_metrics(*_deferred)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in envs.values():
            env.close()
        if arena_pool is not None:
            arena_pool.close()

    if config.log_action_streams and move_streams:
        streams_path = Path(config.output_dir) / f"move_streams_{mode}.npz"
        streams_path.parent.mkdir(parents=True, exist_ok=True)
        # Flat rc_humanlikeness schema (decode-fit stage-6 gate) alongside the
        # legacy per-episode keys: episode-concatenated fb/lr/ud/weapon/attack/
        # turn_deg/keep/lc_* (+ band-v5 discharge/weapon_imp/engaged) +
        # episode_offsets + tick_hz.
        flat: dict = {}
        ep_keys = sorted(k for k in move_streams if k.startswith("ep_"))
        gate_keys = {k.replace("gate_", "").replace("ep_", ""): k
                     for k in move_streams if k.startswith("gate_ep_")}
        if ep_keys and all(k.replace("ep_", "") in gate_keys for k in ep_keys):
            mv = [move_streams[k] for k in ep_keys]
            gt = [move_streams[gate_keys[k.replace("ep_", "")]] for k in ep_keys]
            offs = np.zeros(len(mv) + 1, dtype=np.int64)
            offs[1:] = np.cumsum([len(a) for a in mv])
            mv_cat = np.concatenate(mv, axis=0)
            gt_cat = np.concatenate(gt, axis=0)
            flat = {
                "episode_offsets": offs,
                "fb": mv_cat[:, 0], "lr": mv_cat[:, 1], "ud": mv_cat[:, 2],
                "attack": gt_cat[:, 0].astype(np.int8),
                "weapon": gt_cat[:, 1].astype(np.int8),
                "turn_deg": gt_cat[:, 2].astype(np.float32),
                "keep": gt_cat[:, 3].astype(bool),
                # look-commitment lanes (schema 3) — always present, -1 when
                # the model has no look_seg head (see qnn.schema).
                "lc_cls": gt_cat[:, 7].astype(np.int8),
                "lc_rem": gt_cat[:, 8].astype(np.int16),
                "lc_elapsed": gt_cat[:, 9].astype(np.int16),
                "lc_dur": gt_cat[:, 10].astype(np.int8),
                "lc_dir": gt_cat[:, 11].astype(np.int8),
                # look tangent (schema 4), (n, 2) float16 radians — the corpus
                # act_look_tan quantity, so model streams segment by the human
                # rule (see qnn.schema).
                "look_tan": gt_cat[:, 12:14].astype(np.float16),
            }
            # band-v5 fields — only when the obs carried their engine sources
            # (see _v5_gate_fields above); the scorer fails loud when absent.
            if _v5_gate_fields["ok"]:
                flat["discharge"] = gt_cat[:, 4].astype(np.int8)
                flat["weapon_imp"] = gt_cat[:, 5].astype(np.int8)
                flat["engaged"] = gt_cat[:, 6].astype(bool)
                flat["gate_stream_schema"] = np.asarray(
                    [GATE_STREAM_SCHEMA_VERSION])
        np.savez_compressed(streams_path,
                            tick_hz=np.asarray([float(config.fixed_tick_hz)]),
                            **flat, **move_streams)

    if config.log_acq_streams and acq_streams:
        acq_path = Path(config.output_dir) / f"acq_streams_{mode}.npz"
        acq_path.parent.mkdir(parents=True, exist_ok=True)
        _keys = sorted(acq_scenarios)
        np.savez_compressed(
            acq_path,
            tick_hz=np.asarray([float(config.fixed_tick_hz)]),
            acq_episode_keys=np.asarray(_keys),
            acq_episode_scenarios=np.asarray([acq_scenarios[k] for k in _keys]),
            **acq_streams)

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
    for metric_key in _AUX_INFO_KEYS:
        summary[f"{metric_key}_mean"] = float(aux_metric_sums.get(metric_key, 0.0) / max(total_steps, 1))
    # obs-side fire discrimination (compare to human reference fire-by-angle curve)
    _blind = blind_no_actor + blind_offcone
    _hist_tot = max(sum(fire_angle_hist), 1)
    summary["obs_attack_ticks"] = int(fire_ticks)
    summary["obs_attack_rate"] = float(fire_ticks / max(total_steps, 1))
    # obs-feeder sanity (attack_finished the model actually received).
    summary["dbg_obs_keys"] = _dbg_obs_keys or []
    if _dbg_af:
        _afa = np.asarray(_dbg_af, dtype=np.float64)
        summary["dbg_attack_finished"] = {
            "n": int(_afa.size), "min": float(_afa.min()), "max": float(_afa.max()),
            "mean": float(_afa.mean()), "frac_zero": float((_afa <= 1e-6).mean()),
            "frac_nan": float(np.isnan(_afa).mean()),
            "p50": float(np.percentile(_afa, 50)), "p95": float(np.percentile(_afa, 95))}
    else:
        summary["dbg_attack_finished"] = {"n": 0, "note": "attack_finished absent from obs"}
    # Per-weapon attack decomposition (attack terminology; impulse→name).
    from qnn.utils.combat_metrics import WEAPON_ID_TO_NAME as _WID2N
    _atk_tot = max(sum(weapon_attack_ticks.values()), 1)
    summary["obs_weapon_occupancy"] = {
        _WID2N.get(w, f"w{w}"): float(c / max(total_steps, 1))
        for w, c in sorted(weapon_held_ticks.items())}
    summary["obs_attack_share_per_weapon"] = {
        _WID2N.get(w, f"w{w}"): float(c / _atk_tot)
        for w, c in sorted(weapon_attack_ticks.items())}
    # per-weapon attack rate = attacks-with-w / frames-holding-w (the propensity
    # the aggregate rate averages over the occupancy mix).
    summary["obs_attack_rate_per_weapon"] = {
        _WID2N.get(w, f"w{w}"): float(weapon_attack_ticks.get(w, 0) / c)
        for w, c in sorted(weapon_held_ticks.items()) if c}
    summary["obs_blind_attack_raw_rate"] = float(_blind / max(fire_ticks, 1))
    summary["obs_blind_attack_raw_no_actor_rate"] = float(blind_no_actor / max(fire_ticks, 1))
    summary["obs_blind_attack_raw_offcone_rate"] = float(blind_offcone / max(fire_ticks, 1))
    summary["obs_attack_aim_cos_mean"] = float(fire_cos_sum / max(fire_cos_n, 1))
    summary["obs_attack_cone_deg"] = _FIRE_CONE_DEG
    # LEAD-CORRECTED blind-attack rate (the metric to judge attack discipline by):
    # crosshair→lead-point angle, LOS-gated. The raw obs_blind_attack_raw_* above counts
    # correct aim-ahead shots as blind (over-counts offcone); prefer these.
    _blind_attack = blind_attack_no_los + blind_attack_offcone_lead
    summary["obs_blind_attack_rate"] = float(_blind_attack / max(attack_disc_ticks, 1))
    summary["obs_blind_attack_no_los_rate"] = float(blind_attack_no_los / max(attack_disc_ticks, 1))
    summary["obs_blind_attack_offcone_rate"] = float(blind_attack_offcone_lead / max(attack_disc_ticks, 1))
    summary["obs_attack_disc_ticks"] = int(attack_disc_ticks)
    summary["obs_attack_angle_hist"] = {
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
    summary["engine_turn_by_los_angle_per_weapon"] = {
        w: {lab: {tl: int(c) for tl, c in zip(_TURN_MAG_LABELS, row)}
            for lab, row in zip(_FIRE_ANGLE_LABELS, h)}
        for w, h in sorted(los_turn_hist_w.items())
    }
    summary["engine_los_attack_by_lead_angle_per_weapon"] = {
        w: {lab: {"n_ticks": int(t)} for lab, t in zip(_FIRE_ANGLE_LABELS, b)}
        for w, b in sorted(lead_weapon_buckets.items())
    }
    # Discharge-anchored INTERCEPT (hbw) — the a25 aim-skill lever's MODEL-SIDE ruler
    # (§14). Median + percentile ladder of operative-fire alignment in hitbox-half-
    # widths, aggregate + per (lead-)weapon; scored against the human per-weapon POOLED
    # event ladder in runs/head_probe/_aim_intercept_skill.json
    # (interception_dist).
    summary["engine_intercept_hbw"] = _hbw_percentiles(intercept_hbw_hist)
    summary["engine_intercept_hbw_per_weapon"] = {
        w: _hbw_percentiles(b) for w, b in sorted(intercept_hbw_w.items())
    }
    # Per-discharge intercept EVENT npz (decode-fit v2 P1) — the unbinned rows
    # behind the two ladder fields above, one row per operative discharge.
    # output_dir IS <run>/metrics/eval (run_output_dirs), i.e. exactly the
    # path qnn.decode_fit.events.load_run_events reads. Zero events ⇒ no file
    # (the loader treats absence as "no discharges / predates event logging").
    _write_intercept_events(config.output_dir, intercept_events,
                            truncated=intercept_events_truncated)
    if _win_pre or _win_post:
        # flush pendings cut short by episode end (NaN-pad the forward side)
        for _bufs in _win_pending.values():
            for _buf in _bufs:
                _win_emit(_buf, pad_to=_win_pre + _win_post + 1)
        _write_intercept_windows(config.output_dir, _win_pre, _win_post,
                                 _win_rows)
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
            # per-cell discharge-anchored intercept (hbw) — the grid reads this per
            # (model_weapon × frikbot_weapon) cell to build the gain→quality transfer.
            "engine_intercept_hbw": _hbw_percentiles(
                scenario_intercept_hbw.get(scenario_id, [0] * len(_INTERCEPT_HBW_LABELS))),
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
    known regime name (a25rc1/a25base → bundled config; see
    qnn.model.decode_config.REGIME_CONFIGS) or a path to a decode-config JSON. The resolver builds the param-bound guard adapter that
    QNNPolicy.act consumes (policy_decode_action_postprocess + projectile_release_mask
    when dodge is enabled) — the same object tools/export_onnx.ExportWrapper uses.

    Returns the ``ResolvedDecode`` (or ``None`` for the no-regime case) so the
    caller can source move/look decode params FROM the config — even when
    ``guard_module == "none"`` (then ``_regime_mod`` is None but the resolved
    params still drive the move/look operating point, matching export)."""
    if regime is None or str(regime).strip() in ("", "none"):
        model.decode_action_postprocess = None
        model._regime_mod = None
        # NO default decode facade exists: leaving _decode_mod unset means act()
        # RAISES if it needs the facade. Evals that decode must name a regime /
        # config (or resolve the run's arch via qnn.diag.loader.resolve_decode_module).
        model._decode_mod = None
        return None
    from qnn.model import decode_config as _dc
    resolved = _dc.resolve_decode_config(_dc.config_path_for(str(regime).strip()))
    adapter = resolved.guard_module
    if adapter is None:
        # guard_module="none" (scrubbed study bases): no guard layer, but
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
    # Use the decode config's geometry module in act() (explicit injection).
    model._decode_mod = resolved.decode_module
    return resolved


def _apply_decode_config_params(config: "EvalConfig", model: QNNPolicy, resolved) -> None:
    """Source the whole decode operating point from the resolved decode-config
    onto the (mutable) EvalConfig + model.

    ONE loop over ``qnn.model.decode_config.DECODE_PARAMS`` — the SAME registry
    ``tools/export_onnx`` reads — so eval and export cannot describe different
    decodes. The registry owns the kwarg names, the coercions and the defaults;
    there is no per-key ``if "..." in p`` chain here any more (that chain is
    exactly how attack.crest_* came to be baked into the ONNX while running
    inert offline). Adding a knob = adding a row to the registry.

    ``ResolvedDecode.policy_attrs()`` fails loud on a config that omits a key
    with no code-side default (REQUIRED_PARAM_KEYS / MODULE_REQUIRED_PARAM_KEYS)
    and substitutes the registry default — never a second, drifting copy of it —
    for every optional knob.
    """
    for attr, value in resolved.policy_attrs().items():
        if not hasattr(model, attr):
            # A registry row naming an attribute QNNPolicy does not define would
            # otherwise setattr() a DEAD knob: silently inert in eval, live in
            # the export. Fail loud instead (no silent defaults).
            raise AttributeError(
                f"decode-param registry maps a config key onto QNNPolicy.{attr}, "
                f"which does not exist on {type(model).__name__}. Fix the "
                f"`name` in qnn.model.decode_config.DECODE_PARAMS (it is BOTH "
                f"the policy attribute and the ExportWrapper kwarg).")
        setattr(model, attr, value)
    # The one EvalConfig-side mirror: config.look_aim_prior_gain is provenance
    # only and holds a scalar or None (a (9,) per-impulse vector has no scalar
    # form). There is no config.look_aim_ffwd field.
    _g = model.look_aim_prior_gain
    config.look_aim_prior_gain = _g if isinstance(_g, float) else None
    # a25 LOOK commitment decode: auto-enabled from model SHAPE, not from a
    # config key — a look_seg head with NO classic look head is the sole look
    # mechanism, so the commit decode is mandatory (there is no per-frame look
    # readout to fall back to). It is therefore NOT a DECODE_PARAMS row: deriving
    # it from shape is what makes eval and the exported graph agree by
    # construction (tools/export_onnx derives the same flag the same way).
    # NOTE: head-shape flags live on the NETWORK (model.model), not the QNNPolicy.
    _net = getattr(model, "model", model)
    model.look_commitment = (getattr(_net, "_has_look_seg_head", False)
                             and not getattr(_net, "_has_look_head", False))
    # NOTE: the a24 move keys (move.sticky_tau_*, move.switchback_eps,
    # move.stop_onset, move.tau_engagement_gated, move.jump_*), the inline
    # move_hazard table, and attack.threshold are RETIRED with the a24 arch —
    # a config still carrying them has those keys ignored here (the decode laws
    # no longer exist).


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
    if (
        config.env_backend == "arena_grid"
        and len(config.policy_modes) > 1
        and config.parallel_policy_modes
    ):
        raise ValueError(
            "arena_grid evaluation must run policy modes sequentially because "
            "they share the configured local server ports"
        )
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
    if config.env_backend == "arena_grid":
        eval_notes.append(
            "Evaluation uses native grouped step_many over in-process qnn_arena8 observers."
        )
    else:
        eval_notes.append("Evaluation uses the native worker token path with object, event, and spatial observations.")
    eval_notes.append("Reward shaping and episode metrics remain in Python while the worker supplies deterministic token ticks.")
    if config.env_backend == "arena_grid":
        eval_notes.append(
            "Grouped arena resets use the shared match lifecycle; per-lane reset seeds are not applied."
        )
    elif config.start_mode == "randomized":
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
            deadzone_rad=_lg.get("deadzone_rad"),
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
        RunnerContext, base_results, ensure_arena_workers, ensure_worker, finalize_results,
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
            deadzone_rad=_lg.get("deadzone_rad"),
        )

    eval_cfg = build_run_eval_config(ctx.run_cfg, ctx.device)
    validate_native_mod_assets(
        ctx.asset_root,
        eval_cfg.get("native_args") if isinstance(eval_cfg.get("native_args"), list) else None,
    )
    worker_path = ensure_worker(ctx.worker_binary, rebuild=False)
    if str(eval_cfg.get("env_backend", "process")) == "arena_grid":
        arena_server, arena_client = ensure_arena_workers(
            Path(str(eval_cfg["arena_server_binary"])),
            Path(str(eval_cfg["arena_client_binary"])),
            rebuild=False,
        )
        eval_cfg["arena_server_binary"] = str(arena_server)
        eval_cfg["arena_client_binary"] = str(arena_client)
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
