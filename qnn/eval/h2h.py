"""Head-to-head closed-loop eval: two decode-fitted checkpoints fight each other.

    python -m qnn.eval.h2h \
        --side-a a26rc1b=runs/head_probe/head_probe_atlas_awposw_seed43:runs/decode_fit/head_probe_atlas_awposw_seed43/decode.a26rc1b.json \
        --side-b a26rc2b=runs/head_probe/head_probe_atlas24x11_awposw_seed43:runs/decode_fit/head_probe_atlas24x11_awposw_seed43/decode.a26rc2b.json \
        --rounds 160 --matches 8 --out runs/eval/h2h_rc1b_vs_rc2b/arm0 [--swap]

The engine has supported two externally-driven seats per match since the
self-play arena landed (``qnn_arena_selfplay``: action index ``i`` maps onto
``match i//2`` seat ``i%2``; virtual observers mirror both seats). What never
existed is the Python half: every driver funnels ALL lanes through one
``policy.act``. This module is that missing router — side A owns the even env
ids (seat 0), side B the odd (seat 1), each side runs ITS OWN batched forward
per tick (`qnn.eval.run._select_actions_batched`, states/RNG fully isolated
per side), and the interleaved actions go down the existing
:class:`~qnn.ppo.arena_backend.ArenaGridBackend` in one grouped step.

ROUND PROTOCOL. A round is one bout in one sealed match: it ends when either
seat dies (arena QC ends the bout for both seats) or on the engine timeout
(``max_steps_per_episode`` → truncated, backend auto-resets the match). The
winner is the surviving seat; a suicide loses without crediting the opponent
(``win_reason="suicide"``); both seats dying on one tick is a draw ("double");
timeout is a draw ("timeout"). Every match plays the SAME number of rounds
(no length bias from early stopping). Seat spawns within a cell are fixed, so
a fair pairing runs TWO arms — ``--swap`` gives side B seat 0 — and the caller
aggregates.

MEASUREMENT. Three layers, all per side:
  * rounds.jsonl — one record per round: winner/reason/duration + both seats'
    pre-death context (held weapon impulse, movement_id [1 = airborne], health,
    engaged, shots fired in the round, damage dealt/taken).
  * side_<name>_streams.npz — the band-v5 flat gate-stream schema, one
    "episode" per (lane, round) segment, byte-compatible with what
    ``qnn.eval.run`` writes under ``eval_log_action_streams`` (same field
    construction, lifted verbatim), so the human-band scorer
    (qnn.eval.humanlikeness.human_band) and rate-fidelity ruler run unchanged.
    Carries the cross-head coordination channels too: the look-commit lanes
    (lc_*) and the look tangent (look_tan) — see qnn.schema.
  * summary.json — win ratio + Wilson CI, per-side kills/deaths/suicides,
    op-shot fires/s per weapon, damage per round, stand-still fraction,
    jump-press rate, airborne-at-death rate.

SAME-CHECKPOINT A/B. Two deliberate escape hatches, both stamped into
summary.json: ``--shared-look-grid`` pairs checkpoints whose corpus-fit look
grids disagree (default refuses), and ``--force-look-commit`` drives a DUAL-head
checkpoint (look: polar AND look_seg) through the commitment decode its shape
would not select — the polar-vs-seg arms of one trained model.

The per-tick stream fields must stay IDENTICAL to qnn.eval.run's
``_log_streams`` (band scores are only comparable to prior fits if the
construction matches); the logic is duplicated here deliberately and pinned by
tests/test_h2h.py rather than shared, because run.py builds it inside a
closure over eval-loop state.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from qnn.env.reward import RewardWeights
from qnn.eval.run import (_apply_decode_config_params, _commit_reset_lanes,
                          _EpisodeState, _install_decode_regime,
                          _load_checkpoint, _seed_attack_rng,
                          _select_actions_batched)
from qnn.model.look_seg_decode import LOOK_COMMIT_STATE_DIM
from qnn.model.policy import QNNPolicy
from qnn.ppo.arena_backend import ArenaGridBackend
from qnn.schema import GATE_STREAM_SCHEMA_VERSION
from qnn.vocab import MODALITY_IDS, TOKEN_ACTOR, self_weapon_id_to_impulse

_log = lambda m: print(f"[h2h] {m}", flush=True)  # noqa: E731

# The band-v5 `engaged` mask is "a SIGHT actor is in the live token slots".
MODALITY_SIGHT = MODALITY_IDS["SIGHT"]

TICK_HZ = 20
MAX_STEPS_PER_ROUND = 1800  # 90 s — the arena timeout the backend enforces

# Gate-row layout (qnn.schema GATE_STREAM_SCHEMA_VERSION): the 7 band-v5
# scalars, then the LOOK_COMMIT_STATE_DIM commit lanes, then the 2 look-tangent
# components. Applied through _gate_matrix ONLY — a second hand-written width
# is how the summary block drifted off the row (see _gate_matrix).
GATE_ROW_WIDTH = 7 + LOOK_COMMIT_STATE_DIM + 2


# ── model loading ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SideSpec:
    """One combatant: a trained run + the decode config that made it an rc."""

    name: str
    run_dir: Path
    decode_config: Path

    @classmethod
    def parse(cls, text: str) -> "SideSpec":
        """``name=run_dir:decode_config.json``"""
        name, _, rest = text.partition("=")
        run_dir, _, decode = rest.partition(":")
        if not (name and run_dir and decode):
            raise ValueError(
                f"side spec must be name=run_dir:decode_config.json, got {text!r}")
        return cls(name=name, run_dir=Path(run_dir), decode_config=Path(decode))

    def checkpoint(self) -> Path:
        cks = (sorted((self.run_dir / "checkpoints").glob("best_*.pth"))
               or sorted((self.run_dir / "checkpoints").glob("bc_best_model.pth")))
        if not cks:
            raise FileNotFoundError(f"{self.run_dir}: no best_*.pth checkpoint")
        return cks[0]

    def look_grid(self) -> dict:
        p = self.run_dir / "config" / "look_grid.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: h2h needs the run's pinned look grid")
        return json.loads(p.read_text())


SHARED_LOOK_GRID_CHOICES = ("refuse", "side-a", "side-b")


def _grid_numeric(grid: Mapping[str, Any]) -> dict[str, Any]:
    """The comparable part of a look_grid.json (corpus-fit numbers only —
    metadata like git_commit/created legitimately differs between runs)."""
    return {k: v for k, v in grid.items() if isinstance(v, (int, float, list))}


def _grid_max_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Max |a − b| over every numeric field of two look grids.

    Fails loud when the grids are not element-wise comparable (different field
    sets or different ladder lengths): that is a SHAPE disagreement, not the
    corpus-fit drift the escape hatch exists for, and no scalar delta describes
    it honestly."""
    ka, kb = set(a), set(b)
    if ka != kb:
        raise ValueError(
            f"look grids carry different numeric fields ({sorted(ka ^ kb)}) — "
            "not a fit-drift mismatch; these grids are structurally different")
    worst = 0.0
    for k in sorted(ka):
        va, vb = np.atleast_1d(np.asarray(a[k], dtype=np.float64)), \
            np.atleast_1d(np.asarray(b[k], dtype=np.float64))
        if va.shape != vb.shape:
            raise ValueError(
                f"look grid field {k!r} has different lengths "
                f"({va.shape} vs {vb.shape}) — structurally different grids")
        worst = max(worst, float(np.abs(va - vb).max()) if va.size else 0.0)
    return worst


def _install_shared_look_grid(specs: Sequence[SideSpec],
                              choice: str = "refuse") -> dict[str, Any]:
    """Install the ONE process-global polar grid both models decode on.

    ``choice="refuse"`` (default) REQUIRES the two runs' grids to agree on every
    numeric field: a silent mismatch would decode one side's look head on the
    other side's magnitude ladder. ``side-a``/``side-b`` is the operator's
    explicit escape hatch for pairing checkpoints fitted on different corpora
    (e.g. qwd_v4 vs qwd_v4d_v3vis, which differ in the 4th decimal) — it installs
    THAT side's grid for the whole process and warns loudly with the measured
    delta. There is deliberately NO tolerance-based auto-accept: "close enough"
    is a judgement about the experiment, not about the numbers.

    Returns the provenance record stamped into each arm's summary.json."""
    from qnn.model.look_bins import install_polar_grid

    if choice not in SHARED_LOOK_GRID_CHOICES:
        raise ValueError(f"shared_look_grid must be one of "
                         f"{list(SHARED_LOOK_GRID_CHOICES)}, got {choice!r}")
    grids = [s.look_grid() for s in specs]
    numeric = [_grid_numeric(g) for g in grids]
    identical = numeric[0] == numeric[1]
    if choice == "refuse":
        if not identical:
            diff = [k for k in numeric[0] if numeric[0].get(k) != numeric[1].get(k)]
            raise ValueError(
                f"look grids differ between sides on numeric fields {diff} — "
                "two models with different grids cannot share a process")
        delta = 0.0
        g = grids[0]
    else:
        delta = _grid_max_delta(numeric[0], numeric[1])
        idx = 0 if choice == "side-a" else 1
        g = grids[idx]
        if identical:
            _log(f"--shared-look-grid={choice}: grids are IDENTICAL "
                 "(the hatch changed nothing)")
        else:
            _log("*" * 72)
            _log(f"WARNING: --shared-look-grid={choice} — the two sides' look "
                 "grids DIFFER and this run overrides that refusal.")
            _log(f"WARNING: installing {specs[idx].name}'s grid for BOTH sides; "
                 f"max absolute numeric delta = {delta:.6g}.")
            _log("WARNING: the other side decodes its look head on a ladder it "
                 "was not fitted on — magnitudes are off by that much.")
            _log("*" * 72)
    install_polar_grid(
        torch.tensor(g["mag_centers_rad"], dtype=torch.float32),
        torch.tensor(g["dir_centers_rad"], dtype=torch.float32)
        if "dir_centers_rad" in g else None,
        deadzone_rad=g.get("deadzone_rad"),
    )
    return {"choice": choice, "max_numeric_delta": delta}


FORCE_LOOK_COMMIT_CHOICES = ("none", "a", "b", "both")


def _force_look_commit(model: QNNPolicy, name: str) -> None:
    """Drive this side's look through the look_seg COMMITMENT decode.

    ``_apply_decode_config_params`` derives ``look_commitment`` from model SHAPE
    — a look_seg head with NO classic look head — so a DUAL-head checkpoint
    (both ``look: polar`` and ``look_seg``, e.g.
    ``purecombat_lookseg_v3vis_seed43``) plays polar by default and its seg head
    never decodes. Flipping the flag here, AFTER the shape derivation, is what
    makes a same-checkpoint polar-vs-seg A/B possible: from this point
    ``_select_actions_batched`` stacks + threads ``look_commit_state`` into every
    ``act()`` (it reads the flag per call), and ``policy.act`` takes its
    ``_use_look_commit`` branch because the state is present and the graph has
    ``look_seg`` logits.

    Fails loud on a model with no look_seg head: there is nothing to commit
    with, and act() would silently keep decoding polar."""
    net = getattr(model, "model", model)
    if not getattr(net, "_has_look_seg_head", False):
        raise ValueError(
            f"--force-look-commit names {name}, whose checkpoint has NO look_seg "
            "head — there is no commitment decode to force (a polar-only model "
            "would silently keep its per-frame look readout)")
    model.look_commitment = True


def load_side(spec: SideSpec, *, force_look_commit: bool = False) -> QNNPolicy:
    """Checkpoint + decode config → ready policy, via the SAME loader chain the
    eval runner uses (load → install regime → apply the DECODE_PARAMS registry),
    so an h2h side runs exactly the decode its rc letter shipped.

    ``force_look_commit`` overrides the shape-derived look routing for a
    dual-head checkpoint (see :func:`_force_look_commit`); it must be applied
    after ``_apply_decode_config_params``, which sets the flag from shape."""
    model = _load_checkpoint(str(spec.checkpoint()), device="cpu")
    resolved = _install_decode_regime(model, str(spec.decode_config))
    # _apply_decode_config_params mirrors one provenance scalar onto the eval
    # config; a namespace stands in for the EvalConfig we don't have.
    shim = SimpleNamespace(look_aim_prior_gain=None)
    _apply_decode_config_params(shim, model, resolved)
    if force_look_commit:
        _force_look_commit(model, spec.name)
    return model


# ── per-tick stream logging (band-v5; lifted from qnn.eval.run._log_streams) ─

def _gate_matrix(rows: Sequence[tuple]) -> np.ndarray:
    """Accumulated gate rows → an ``(n, GATE_ROW_WIDTH)`` float matrix.

    The ONE place the row width is applied: both the npz writer and the
    summary's per-weapon fold read the row through here, so widening the gate
    row can never leave a stale literal behind. It did once: the summary block
    kept its own ``reshape(-1, 7)`` after the look_commit lanes landed (12-wide
    rows → ValueError at the end of an otherwise completed run), and at the
    CURRENT 14-wide row that same literal would have been worse than a crash —
    14 is divisible by 7, so it would have silently folded every row into two
    mis-columned ones and reported plausible nonsense."""
    return np.asarray(rows, dtype=np.float64).reshape(-1, GATE_ROW_WIDTH)


def attack_lane_convention(act: Mapping[str, Any]) -> str:
    """``"a26"`` or ``"a27"`` — which action convention this side emits.

    a26-line checkpoints carry the 9-way selector on the WEAPON slot, so
    ``policy.act`` returns ``attack`` = the fire BIT and ``weapon`` = the
    switch impulse: the gate row's attack/discharge lanes are exact.

    a27 pure-combat checkpoints carry the selector on the ATTACK slot: one
    folded lane, ``attack WITH weapon X this tick``, 0 = no attack
    (``decode_actions.attack_lane_gate``). ``attack != 0`` IS the fire
    decision there, so the lanes are exact on both conventions; the a27 side
    also carries the raw bit as ``act["fire"]``. The convention is still
    stamped into summary.json because a reader comparing an a26 and an a27
    side needs to know that a26's ``weapon`` key is a switch impulse while
    a27 folds the switch into the attack lane."""
    return "a26" if "weapon" in act else "a27"


def _log_lane_tick(st: _EpisodeState, act: Mapping[str, Any], *,
                   has_look_commit: bool) -> None:
    """Append this tick's move/gate stream rows for one lane.

    Field-for-field the ``_log_streams`` construction in qnn.eval.run (move
    classes as mv+1; turn from the look vector; keep = any-recency actor;
    engaged = live LOS actor; discharge = attack & attack_finished expired;
    weapon_imp from self_weapon_id; then the LOOK_COMMIT_STATE_DIM commitment
    lanes this tick's look decode left in ``st.look_commit``, or the -1
    sentinel when the side's model has no look_seg head; then the look TANGENT
    z = θ·ŷz, the corpus act_look_tan quantity). Pinned by
    tests/test_h2h.py."""
    mv = act["move"]
    st.move_trace.append((int(mv[0]) + 1, int(mv[1]) + 1, int(mv[2]) + 1))
    lk = act.get("look")
    if lk is not None:
        lx, ly, lz = float(lk[0]), float(lk[1]), float(lk[2])
        # θ = atan2(|yz|, x); tangent = θ·ŷz (never arccos(x)) — the corpus
        # act_look_tan law, mirroring qnn.eval.run._log_streams verbatim.
        hyz = float(np.hypot(ly, lz))
        th = float(np.arctan2(hyz, lx))
        turn = float(np.degrees(th))
        tsc = (th / hyz) if hyz > 0.0 else 0.0
        tan_y, tan_z = ly * tsc, lz * tsc
    else:
        turn = 0.0
        tan_y, tan_z = 0.0, 0.0
    keep = 0
    engaged = 0
    obs = st.obs
    if isinstance(obs, dict) and "entity_types" in obs:
        et = np.asarray(obs["entity_types"]).reshape(-1)
        keep = int((et == TOKEN_ACTOR).any())
        if "entity_count" in obs:
            n = int(np.asarray(obs["entity_count"]).reshape(-1)[0])
            # engaged (band v5 context mask) = SIGHT actor among the live
            # entity slots. A27 (obs-api percept policy v3) derives it
            # directly from modality and carries NO recency column at all;
            # legacy (v1) observations fall back to recency == 0. Mirrors
            # qnn.eval.run._log_streams verbatim — reading recency first and
            # silently leaving engaged=0 on a v3 stream is exactly how the
            # first cross-wire pairing reported engaged_frac = 0.0.
            if "entity_modality_id" in obs:
                mod = np.asarray(obs["entity_modality_id"]).reshape(-1)[:n]
                engaged = int(((et[:n] == TOKEN_ACTOR)
                               & (mod == MODALITY_SIGHT)).any())
            elif "entity_recency" in obs:
                rec = np.asarray(obs["entity_recency"],
                                 dtype=np.float64).reshape(-1)[:n]
                engaged = int(((et[:n] == TOKEN_ACTOR) & (rec <= 0.0)).any())
            else:
                raise RuntimeError(
                    "h2h gate stream: this side's obs carries neither "
                    "'entity_modality_id' (a27 percept v3) nor "
                    "'entity_recency' (legacy) — the band-v5 `engaged` mask "
                    "cannot be derived and would silently log 0 for every "
                    f"tick (obs fields: {sorted(obs)})")
    # ATTACK LANE. Exact on both conventions: a26's attack slot IS the fire
    # bit (``weapon`` rides its own key), and a27's folded lane is gated on
    # the fire bit before it leaves policy.act (``attack != 0`` iff firing —
    # decode_actions.attack_lane_gate), so the truth-value read here is the
    # model's fire decision either way. See :func:`attack_lane_convention`.
    attack = int(act.get("attack", 0) or 0)
    af = obs.get("attack_finished") if isinstance(obs, dict) else None
    disch = int(bool(attack) and af is not None
                and float(np.asarray(af).reshape(-1)[0]) <= 1e-6)
    wimp = 0
    if isinstance(obs, dict) and "self_weapon_id" in obs:
        wimp = int(self_weapon_id_to_impulse(
            int(np.asarray(obs["self_weapon_id"]).reshape(-1)[0])))
    if has_look_commit:
        lc = tuple(int(v) for v in np.asarray(
            st.look_commit).reshape(-1)[:LOOK_COMMIT_STATE_DIM])
    else:
        lc = (-1,) * LOOK_COMMIT_STATE_DIM
    st.gate_trace.append(
        (attack, int(act.get("weapon", 0) or 0), turn, keep, disch, wimp,
         engaged) + lc + (tan_y, tan_z))


def _lane_ctx(st: _EpisodeState) -> dict[str, Any]:
    """The pre-step context snapshot used as a seat's death/kill context."""
    obs = st.obs
    ctx: dict[str, Any] = {}
    if isinstance(obs, dict):
        if "self_movement_id" in obs:
            ctx["movement_id"] = int(np.asarray(obs["self_movement_id"]).reshape(-1)[0])
        if "health" in obs:
            ctx["health"] = float(np.asarray(obs["health"]).reshape(-1)[0])
        if "self_weapon_id" in obs:
            ctx["weapon_imp"] = int(self_weapon_id_to_impulse(
                int(np.asarray(obs["self_weapon_id"]).reshape(-1)[0])))
    if st.gate_trace:
        ctx["engaged"] = int(st.gate_trace[-1][6])
    return ctx


# ── the match driver ─────────────────────────────────────────────────────────

@dataclass
class _LaneLedger:
    """Per-lane per-round counters folded from the step infos."""

    shots: int = 0
    hits: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0

    def fold(self, info: Mapping[str, Any]) -> None:
        self.shots += int(info.get("shots_fired", 0) or 0)
        self.hits += int(info.get("hit_count", 0) or 0)
        self.damage_dealt += float(info.get("damage_direct", 0.0) or 0.0)
        self.damage_dealt += float(info.get("damage_splash", 0.0) or 0.0)
        self.damage_taken += float(info.get("damage_taken_other", 0.0) or 0.0)

    def snapshot(self) -> dict[str, float]:
        return {"shots": self.shots, "hits": self.hits,
                "damage_dealt": round(self.damage_dealt, 1),
                "damage_taken": round(self.damage_taken, 1)}


@dataclass
class _SideStreams:
    """Round-segmented band-v5 stream accumulation for one side."""

    move_rows: list[tuple] = field(default_factory=list)
    gate_rows: list[tuple] = field(default_factory=list)
    offsets: list[int] = field(default_factory=lambda: [0])

    def close_segment(self, st: _EpisodeState) -> None:
        if st.move_trace:
            self.move_rows.extend(st.move_trace)
            self.gate_rows.extend(st.gate_trace)
            self.offsets.append(len(self.move_rows))
        st.move_trace = []
        st.gate_trace = []

    def write(self, path: Path) -> None:
        mv = np.asarray(self.move_rows, dtype=np.int64).reshape(-1, 3)
        gt = _gate_matrix(self.gate_rows)
        np.savez_compressed(
            path,
            tick_hz=np.asarray([float(TICK_HZ)]),
            episode_offsets=np.asarray(self.offsets, dtype=np.int64),
            fb=mv[:, 0], lr=mv[:, 1], ud=mv[:, 2],
            attack=gt[:, 0].astype(np.int8),
            weapon=gt[:, 1].astype(np.int8),
            turn_deg=gt[:, 2].astype(np.float32),
            keep=gt[:, 3].astype(bool),
            discharge=gt[:, 4].astype(np.int8),
            weapon_imp=gt[:, 5].astype(np.int8),
            engaged=gt[:, 6].astype(bool),
            # look-commitment lanes (gate schema 3) — always present, -1 when
            # the side's model has no look_seg head (see qnn.schema).
            lc_cls=gt[:, 7].astype(np.int8),
            lc_rem=gt[:, 8].astype(np.int16),
            lc_elapsed=gt[:, 9].astype(np.int16),
            lc_dur=gt[:, 10].astype(np.int8),
            lc_dir=gt[:, 11].astype(np.int8),
            # look tangent (gate schema 4), (n, 2) float16 radians — the corpus
            # act_look_tan quantity (see qnn.schema).
            look_tan=gt[:, 12:14].astype(np.float16),
            gate_stream_schema=np.asarray([GATE_STREAM_SCHEMA_VERSION]),
        )


def _fresh_state(model: QNNPolicy, obs_row: dict[str, np.ndarray],
                 seed: int) -> _EpisodeState:
    return _EpisodeState(
        episode_index=0,
        obs=obs_row,
        scenario_id="h2h",
        attack_rng=_seed_attack_rng(seed),
        hidden=model.zero_hidden(1)[0].copy(),
        move_commit=np.asarray(_commit_reset_lanes(), dtype=np.float32),
    )


def _obs_row(obs, lane: int) -> dict[str, np.ndarray]:
    """One lane's obs dict from either materialized form: the dense field-major
    dict (homogeneous seats) or the per-lane dict list (heterogeneous seats —
    the cross-arch pairing path)."""
    if isinstance(obs, list):
        return {k: np.asarray(v).copy() for k, v in obs[lane].items()}
    return {k: np.asarray(v[lane]).copy() for k, v in obs.items()}


def _wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# The decode-fit botpin regime: single weapon owned+selected by BOTH seats,
# infinite ammo (a 90 s pin round would otherwise run high-consumption
# weapons dry and corrupt every rate), full kit. Applied symmetrically.
PIN_WEAPON_CONFIGS = {
    "shotgun":         {"model_weapon": "shotgun", "shells": 100},
    "super_shotgun":   {"model_weapon": "super_shotgun", "shells": 100},
    "nailgun":         {"model_weapon": "nailgun", "nails": 200},
    "super_nailgun":   {"model_weapon": "super_nailgun", "nails": 200},
    "grenade_launcher": {"model_weapon": "grenade_launcher", "rockets": 100},
    "rocket_launcher": {"model_weapon": "rocket_launcher", "rockets": 100},
    "lightning":       {"model_weapon": "lightning", "cells": 200},
}
_PIN_BASE = {"infinite_ammo": 1, "health": 100,
             "armor_value": 200, "armor_type": 0.8}


def run_h2h(side_a: SideSpec, side_b: SideSpec, *, out_dir: Path,
            rounds: int, matches: int = 8, map_id: str = "qnn_arena8",
            swap: bool = False, seed: int = 43, base_port: int = 29000,
            reward_json: Path | None = None, pin_weapon: str | None = None,
            shared_look_grid: str = "refuse", force_look_commit: str = "none",
            server_executable: str = "assets/bin/ppo_arena_server",
            client_executable: str = "assets/bin/ppo_arena_client",
            basedir: str = "assets") -> dict[str, Any]:
    """Play ``rounds`` bouts (split evenly across ``matches`` sealed 1v1s) and
    write rounds.jsonl + per-side stream npz + summary.json into ``out_dir``.

    ``shared_look_grid`` / ``force_look_commit`` are the two same-checkpoint-A/B
    escape hatches; both are stamped into summary.json (see
    :func:`_install_shared_look_grid` and :func:`_force_look_commit`). They name
    sides A/B, NOT seats, so ``--swap`` does not move them."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if force_look_commit not in FORCE_LOOK_COMMIT_CHOICES:
        raise ValueError(f"force_look_commit must be one of "
                         f"{list(FORCE_LOOK_COMMIT_CHOICES)}, got "
                         f"{force_look_commit!r}")
    if side_a.name == side_b.name:
        # wins / summary[side] / the per-side stream npz are all keyed by NAME.
        # Same-checkpoint A/B (one run dir, two look mechanisms) is exactly the
        # case that tempts identical names, and it would silently fold both
        # sides' results into one entry.
        raise ValueError(
            f"both sides are named {side_a.name!r} — side names key the wins "
            "table, summary.json and the stream npz; give the arms distinct "
            "names (e.g. lookseg_polar vs lookseg_seg)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The grid hatch names side A/B; _install_shared_look_grid takes them in
    # that order so --swap never re-points which grid gets installed.
    grid_prov = _install_shared_look_grid((side_a, side_b), shared_look_grid)
    seat0, seat1 = (side_b, side_a) if swap else (side_a, side_b)
    forced = {side_a.name: force_look_commit in ("a", "both"),
              side_b.name: force_look_commit in ("b", "both")}
    models = {0: load_side(seat0, force_look_commit=forced[seat0.name]),
              1: load_side(seat1, force_look_commit=forced[seat1.name])}
    for spec in (side_a, side_b):
        if forced[spec.name]:
            _log(f"--force-look-commit: {spec.name} plays the look_seg "
                 "COMMITMENT decode (shape default overridden)")
    # Per-seat look-commitment flag: decides whether the stream's lc_* lanes
    # carry this side's look decode state or the -1 "no look_seg head"
    # sentinel (the two sides can differ — cross-arch pairings are the point).
    has_look_commit = {s: bool(getattr(models[s], "look_commitment", False))
                       for s in (0, 1)}
    # Per-seat obs declarations (obs-api): each side plays on the wire its
    # checkpoint trained on — cross-arch pairings get heterogeneous frames in
    # one world via the per-seat emit plans.
    from qnn.obs_api import declaration_for_run
    decls = {0: declaration_for_run(seat0.run_dir),
             1: declaration_for_run(seat1.run_dir)}
    names = {0: seat0.name, 1: seat1.name}
    _log(f"seat0={seat0.name} seat1={seat1.name} map={map_id} "
         f"matches={matches} rounds={rounds} swap={swap}")

    rw = RewardWeights.from_json(
        reward_json if reward_json else
        Path("runs/eval/seed43_stage6_template/config/reward.json"))
    num_lanes = 2 * matches
    backend = ArenaGridBackend(
        num_lanes=num_lanes,
        server_executable=server_executable,
        client_executable=client_executable,
        basedir=basedir, workdir=None, map_id=map_id,
        matches_per_server=matches, seat_mode="self_play",
        base_port=base_port, bot_skill=0,
        max_steps_per_episode=MAX_STEPS_PER_ROUND,
        fixed_tick_hz=TICK_HZ, reward_weights=rw,
        direct_actions=True, observer_mode="virtual",
        scenario_id=(f"h2h-{map_id}" if pin_weapon is None
                     else f"h2h-{map_id}-pin-{pin_weapon}"),
        weapon_config=(None if pin_weapon is None
                       else {**PIN_WEAPON_CONFIGS[pin_weapon], **_PIN_BASE}),
        declarations=[decls[lane % 2] for lane in range(num_lanes)],
    )
    lanes_of_seat = {s: [e for e in range(num_lanes) if e % 2 == s] for s in (0, 1)}
    match_of_lane = [lane // 2 for lane in range(num_lanes)]

    rounds_per_match = max(1, rounds // matches)
    rounds_done = [0] * matches
    round_start_tick = [0] * matches
    match_live = [True] * matches
    ledgers = {lane: _LaneLedger() for lane in range(num_lanes)}
    streams = {0: _SideStreams(), 1: _SideStreams()}
    # Per-seat action convention, resolved from the first tick's action dict
    # and stamped into summary.json (see :func:`attack_lane_convention`).
    attack_lane: dict[int, str | None] = {0: None, 1: None}
    round_records: list[dict[str, Any]] = []
    wins = {seat0.name: 0, seat1.name: 0, "draw": 0}

    rng = {s: torch.Generator(device="cpu") for s in (0, 1)}
    for s in (0, 1):
        rng[s].manual_seed(seed + 7919 * s + (100003 if swap else 0))

    try:
        obs = backend.reset()
        states: list[_EpisodeState] = [
            _fresh_state(models[lane % 2], _obs_row(obs, lane),
                         seed * 1_000_003 + lane)
            for lane in range(num_lanes)
        ]

        tick = 0
        t0 = time.perf_counter()
        while any(match_live):
            # One batched forward per SIDE; states/RNG stay side-isolated.
            actions_by_lane: dict[int, Mapping[str, Any]] = {}
            for s in (0, 1):
                lanes = lanes_of_seat[s]
                sts = [states[lane] for lane in lanes]
                acts, next_hidden = _select_actions_batched(
                    models[s], "sampled", sts, rng[s])
                for i, lane in enumerate(lanes):
                    actions_by_lane[lane] = acts[i]
                    states[lane].hidden = next_hidden[i]

            # Pre-step context snapshots + stream rows (pre-step obs semantics,
            # same as run.py's overlap window).
            ctx = [_lane_ctx(states[lane]) for lane in range(num_lanes)]
            for lane in range(num_lanes):
                _log_lane_tick(states[lane], actions_by_lane[lane],
                               has_look_commit=has_look_commit[lane % 2])
            for s in (0, 1):
                if attack_lane[s] is None:
                    attack_lane[s] = attack_lane_convention(
                        actions_by_lane[lanes_of_seat[s][0]])
                    _log(f"{names[s]}: {attack_lane[s]} attack-lane "
                         "convention" + (" (fire + weapon folded into the "
                                         "attack slot)"
                                         if attack_lane[s] == "a27" else
                                         " (attack = fire bit, weapon = "
                                         "switch impulse)"))

            move = np.zeros((num_lanes, 3), dtype=np.float32)
            look = np.zeros((num_lanes, 3), dtype=np.float32)
            look[:, 0] = 1.0
            attack = np.zeros(num_lanes, dtype=np.int64)
            weapon = np.zeros(num_lanes, dtype=np.int64)
            for lane, act in actions_by_lane.items():
                move[lane] = np.asarray(act["move"], dtype=np.float32)
                look[lane] = np.asarray(act["look"], dtype=np.float32)
                attack[lane] = int(act.get("attack", 0))
                weapon[lane] = int(act.get("weapon", 0))
            batch = backend.step_many(
                {"move": move, "look": look, "attack": attack, "weapon": weapon})
            tick += 1

            for lane in range(num_lanes):
                states[lane].obs = _obs_row(batch.obs, lane)
                states[lane].step_count += 1
                ledgers[lane].fold(batch.infos[lane])

            # Round boundaries, one per match.
            for m in range(matches):
                if not match_live[m]:
                    continue
                l0, l1 = 2 * m, 2 * m + 1
                died = {lane: bool(batch.infos[lane].get("player_died"))
                        for lane in (l0, l1)}
                timed_out = bool(batch.truncated[l0] or batch.truncated[l1])
                if not (died[l0] or died[l1] or timed_out):
                    continue

                if died[l0] and died[l1]:
                    winner, reason = None, "double"
                elif died[l0] or died[l1]:
                    victim = l0 if died[l0] else l1
                    killer = l1 if victim == l0 else l0
                    suicide = bool(batch.infos[victim].get("player_suicide"))
                    winner = killer
                    reason = "suicide" if suicide else "frag"
                else:
                    winner, reason = None, "timeout"

                rec = {
                    "match": m, "round": rounds_done[m],
                    "ticks": tick - round_start_tick[m],
                    "winner": None if winner is None else names[winner % 2],
                    "winner_seat": None if winner is None else winner % 2,
                    "reason": reason,
                    "seats": {},
                }
                for lane in (l0, l1):
                    rec["seats"][names[lane % 2]] = {
                        "seat": lane % 2, "died": died[lane],
                        **ledgers[lane].snapshot(), **ctx[lane]}
                round_records.append(rec)
                # Incremental flush: a stopped/killed run keeps every
                # completed round (summaries/streams still land at the end).
                with (out_dir / "rounds.jsonl").open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                wins["draw" if winner is None else names[winner % 2]] += 1
                rounds_done[m] += 1
                round_start_tick[m] = tick

                for lane in (l0, l1):
                    streams[lane % 2].close_segment(states[lane])
                    ledgers[lane] = _LaneLedger()

                if rounds_done[m] >= rounds_per_match:
                    match_live[m] = False
                    continue
                # Timeout already reset engine-side (backend receive() returned
                # the live spawn rows, which states[].obs now hold); a death
                # needs the explicit match reset the eval adapter also does.
                if not timed_out:
                    obs = backend.reset_lanes((l0,))
                for lane in (l0, l1):
                    states[lane] = _fresh_state(
                        models[lane % 2],
                        _obs_row(obs, lane) if not timed_out else states[lane].obs,
                        seed * 1_000_003 + 977 * lane + 7 * rounds_done[m])

            if tick % 600 == 0:
                done = sum(rounds_done)
                rate = tick / max(time.perf_counter() - t0, 1e-9)
                _log(f"tick {tick}  rounds {done}/{rounds_per_match * matches}  "
                     f"{rate:.0f} ticks/s  wins={wins}")
    finally:
        backend.close()

    total = sum(rounds_done)
    decided = wins[seat0.name] + wins[seat1.name]
    summary: dict[str, Any] = {
        "sides": {"seat0": seat0.name, "seat1": seat1.name},
        "specs": {spec.name: {"run_dir": str(spec.run_dir),
                              "decode_config": str(spec.decode_config),
                              # which look mechanism this side actually played
                              "force_look_commit": forced[spec.name],
                              "look_commitment": has_look_commit[s]}
                  for s, spec in ((0, seat0), (1, seat1))},
        "map": map_id, "matches": matches, "swap": swap, "seed": seed,
        "pin_weapon": pin_weapon,
        "shared_look_grid": grid_prov,
        "rounds": total, "wins": wins,
        "decided_rounds": decided,
    }
    for s, spec in ((0, seat0), (1, seat1)):
        w = wins[spec.name]
        lo, hi = _wilson(w, decided) if decided else (0.0, 1.0)
        side_rounds = [r for r in round_records]
        deaths = sum(1 for r in side_rounds if r["seats"][spec.name]["died"])
        suicides = sum(1 for r in side_rounds
                       if r["winner"] and r["winner"] != spec.name
                       and r["reason"] == "suicide")
        air_deaths = sum(1 for r in side_rounds
                         if r["seats"][spec.name]["died"]
                         and r["seats"][spec.name].get("movement_id") == 1)
        gt = _gate_matrix(streams[s].gate_rows)
        mv = np.asarray(streams[s].move_rows, dtype=np.int64).reshape(-1, 3)
        ticks_total = max(len(gt), 1)
        per_weapon = {}
        if len(gt):
            eng = gt[:, 6] > 0
            for imp in range(1, 9):
                sel = (gt[:, 5] == imp)
                n_disch = int((gt[sel, 4] > 0).sum())
                n_eng = int((sel & eng).sum())
                if sel.sum():
                    per_weapon[str(imp)] = {
                        "ticks_held": int(sel.sum()),
                        "discharges": n_disch,
                        "fires_per_s_engaged": round(
                            float((gt[sel & eng, 4] > 0).sum())
                            / max(n_eng / TICK_HZ, 1e-9), 3),
                    }
        summary[spec.name] = {
            "win_rate_decided": round(w / decided, 4) if decided else None,
            "win_rate_ci95": [round(lo, 4), round(hi, 4)],
            "deaths": deaths, "deaths_by_suicide": suicides,
            "airborne_death_frac": round(air_deaths / max(deaths, 1), 4),
            "standstill_frac": round(
                float(((mv[:, 0] == 1) & (mv[:, 1] == 1)).mean()), 4) if len(mv) else None,
            "jump_press_rate_per_s": round(
                float((mv[:, 2] == 2).mean()) * TICK_HZ, 3) if len(mv) else None,
            "engaged_frac": round(float((gt[:, 6] > 0).mean()), 4) if len(gt) else None,
            # "a26" ⇒ attack = fire bit, weapon = switch impulse on its own
            # key. "a27" ⇒ one folded lane, nonzero = firing WITH that weapon.
            # The rates below are the fire decision either way; the stamp says
            # how the two sides' action dicts differ (attack_lane_convention).
            "attack_lane_convention": attack_lane[s],
            "discharges_per_s": round(
                float((gt[:, 4] > 0).sum()) / (ticks_total / TICK_HZ), 3),
            "damage_dealt_per_round": round(sum(
                r["seats"][spec.name]["damage_dealt"] for r in side_rounds)
                / max(total, 1), 1),
            "per_weapon": per_weapon,
        }

    # rounds.jsonl was flushed incrementally; rewrite once for a clean file
    # (dedupes any partial line from a mid-write kill).
    (out_dir / "rounds.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in round_records))
    for s, spec in ((0, seat0), (1, seat1)):
        streams[s].write(out_dir / f"side_{spec.name}_streams.npz")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    _log(f"done: {total} rounds  wins={wins}  -> {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.eval.h2h", description=__doc__)
    ap.add_argument("--side-a", required=True, type=SideSpec.parse,
                    help="name=run_dir:decode_config.json")
    ap.add_argument("--side-b", required=True, type=SideSpec.parse)
    ap.add_argument("--rounds", type=int, default=160,
                    help="total rounds (split evenly across matches)")
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--map", default="qnn_arena8")
    ap.add_argument("--swap", action="store_true",
                    help="side B takes seat 0 (run both arms for a fair pairing)")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--base-port", type=int, default=29000)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reward-json", type=Path, default=None)
    ap.add_argument("--pin-weapon", choices=sorted(PIN_WEAPON_CONFIGS),
                    default=None,
                    help="single-weapon pin for BOTH seats (botpin regime: "
                         "owned+selected, infinite ammo)")
    ap.add_argument("--shared-look-grid", choices=list(SHARED_LOOK_GRID_CHOICES),
                    default="refuse",
                    help="grids must match exactly (refuse, default), or "
                         "install that side's grid for BOTH models and warn "
                         "with the measured max numeric delta — for pairing "
                         "checkpoints fitted on different corpora")
    ap.add_argument("--force-look-commit", choices=list(FORCE_LOOK_COMMIT_CHOICES),
                    default="none",
                    help="drive the named side(s) through the look_seg "
                         "COMMITMENT decode instead of the shape default — the "
                         "polar-vs-seg A/B on one dual-head checkpoint")
    a = ap.parse_args(argv)
    run_h2h(a.side_a, a.side_b, out_dir=a.out, rounds=a.rounds,
            matches=a.matches, map_id=a.map, swap=a.swap, seed=a.seed,
            base_port=a.base_port, reward_json=a.reward_json,
            pin_weapon=a.pin_weapon, shared_look_grid=a.shared_look_grid,
            force_look_commit=a.force_look_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
