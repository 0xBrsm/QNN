"""Head-to-head closed-loop eval: two decode-fitted checkpoints fight each other.

    python -m qnn.eval.h2h \
        --side-a a26rc1b=runs/head_probe/head_probe_atlas_awposw_seed43:runs/decode_fit/head_probe_atlas_awposw_seed43/decode.a26rc1b.json \
        --side-b a26rc2b=runs/head_probe/head_probe_atlas24x11_awposw_seed43:runs/decode_fit/head_probe_atlas24x11_awposw_seed43/decode.a26rc2b.json \
        --rounds 160 --matches 8 --out runs/eval/h2h_rc1b_vs_rc2b/arm0

The engine has supported two externally-driven seats per match since the
self-play arena landed (``qnn_arena_selfplay``: action index ``i`` maps onto
``match i//2`` seat ``i%2``; virtual observers mirror both seats). What never
existed is the Python half: every driver funnels ALL lanes through one
``policy.act``. This module is that missing router — each side runs ITS OWN
batched forward per tick over its own lanes
(`qnn.eval.run._select_actions_batched`, states/RNG fully isolated per side),
and the interleaved actions go down the existing
:class:`~qnn.ppo.arena_backend.ArenaGridBackend` in one grouped step. Which
SIDE occupies which SEAT alternates by match (``side_of_lane`` below); the
seat assignment itself, ``lane % 2``, never moves.

ROUND PROTOCOL. A round is one bout in one sealed match: it ends when either
seat dies (arena QC ends the bout for both seats) or on the engine timeout
(``max_steps_per_episode`` → truncated, backend auto-resets the match). The
winner is the surviving seat; a suicide loses without crediting the opponent
(``win_reason="suicide"``); both seats dying on one tick is a draw ("double");
timeout is a draw ("timeout"). Every match plays the SAME number of rounds
(no length bias from early stopping). SEATING IS BALANCED-RANDOM: a seeded
shuffle picks exactly half the matches to seat side A at 0. Balanced because
seat-INDEXED advantages exist regardless of which model occupies the seat
(tiered maps spawn seat 1 on the deck by construction; the engine's fixed
client think order favored seat 0 on hitscan until the arena server grew the
per-frame client-order shuffle, ``qnn_client_shuffle``). Random rather than
a deterministic alternation because the CELLS are not interchangeable either
— the fixed-seating archive shows a seat-by-cell-column-parity interaction
(seat 0 wins 50.9% in even columns vs 46.0% in odd, 45k rounds, z=10.5) —
and any fixed seating pattern risks re-aliasing venue structure onto the
side split. ``seat_flips`` is stamped into summary.json.

MEASUREMENT. Three layers, all per side:
  * rounds.jsonl — one record per round: winner/reason/duration + both seats'
    pre-death context (equipped weapon impulse, movement_id [1 = airborne], health,
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
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from qnn.engine_norm import (IT_AXE, IT_GRENADE_LAUNCHER, IT_LIGHTNING,
                             IT_NAILGUN, IT_ROCKET_LAUNCHER, IT_SHOTGUN,
                             IT_SUPER_NAILGUN, IT_SUPER_SHOTGUN,
                             weapon_feasibility_bits)
from qnn.env.reward import RewardWeights
from qnn.eval.run import (_apply_decode_config_params, _arena_weapon_config,
                          _commit_reset_lanes, _EpisodeState,
                          _install_decode_regime, _load_checkpoint,
                          _seed_attack_rng, _select_actions_batched)
from qnn.eval.look_actuation import (LOOK_ACTUATION_CHOICES, StrokeExecutor,
                                     actuate_window)
from qnn.model.look_seg_decode import LOOK_COMMIT_STATE_DIM
from qnn.model.policy import QNNPolicy
from qnn.ppo.arena_backend import ArenaGridBackend
from qnn.schema import GATE_STREAM_SCHEMA_VERSION
from qnn.vocab import MODALITY_IDS, TOKEN_ACTOR, self_weapon_id_to_impulse

_log = lambda m: print(f"[h2h] {m}", flush=True)  # noqa: E731

# The band-v5 `engaged` mask is "a SIGHT actor is in the live token slots".
MODALITY_SIGHT = MODALITY_IDS["SIGHT"]

# Tick rate is NOT a module constant: each run carries its own fixed_tick_hz
# (config/train.json). h2h no longer requires the two sides to AGREE (Phase 2
# addition) — see _resolve_world_rate — it resolves a WORLD rate (the faster
# side's own rate) that every side's rate must divide evenly, and runs the
# ArenaGridBackend + round timeout at that world rate. A side whose own
# cadence is slower than the world decides (calls its model) only every
# `stride` world ticks; qnn.eval.look_actuation fills the stride-1 gaps
# between a slow side's decisions (move/attack/jump/weapon HOLD, look
# ACTUATED — see run_h2h's `stride`/`actuation` locals). A single 20 Hz
# constant would silently mis-stamp streams and mis-scale the per-second
# summary rates for a slower pairing — see run_h2h's `world_hz` local, used
# for both.
# Round timeout is WALL-CLOCK, converted to ticks at the WORLD rate:
# cross-rate margin comparisons (the Phase-2 10-vs-20 Hz difference-in-
# differences, now the look-actuation-decoupling cells) need rounds of equal
# duration, and 90 s at 20 Hz is the protocol every existing h2h aggregate was
# measured under. (h2h never reads eval_obs_lag_ticks — see qnn.eval.run's
# EvalConfig.obs_lag_ticks — so that tick-count knob isn't a live issue here.)
MAX_ROUND_SECONDS = 90.0


def _max_steps_per_round(world_hz: int) -> int:
    """1800 ticks at 20 Hz — the historical protocol value — 900 at 10 Hz."""
    return int(round(MAX_ROUND_SECONDS * world_hz))


def _resolve_world_rate(tick_hz_by_name: Mapping[str, int],
                        world_hz_override: int | None = None,
                        ) -> "tuple[int, dict[str, int]]":
    """WORLD rate + each side's decision STRIDE.

    Replaces the old equal-tick_hz requirement (Phase 2): the world runs at
    ``max(tick_hz_by_name.values())`` and every side's own rate must divide it
    EVENLY — ``stride = world_hz // side_hz`` world ticks between that side's
    decisions (stride 1 ⇒ decides every world tick, the byte-identical
    same-rate case, including a same-rate pairing slower than 20 Hz — e.g.
    (10, 10) resolves to a 10 Hz world with both strides 1, preserving Phase
    2's own results unchanged). Fails loud naming every side's rate when one
    does not divide (e.g. 15 vs 20 Hz: neither is a divisor of the other, and
    there is no honest world rate to run the pairing at).

    ``world_hz_override`` runs the world FASTER than the fastest decider —
    the look-actuation-decoupling cells' whole point: two 10 Hz-trained
    sides in a 20 Hz world (strides 2/2) so their actuators have engine
    ticks to play through. It may never be slower than the fastest side and
    every side's rate must divide it (same fail-loud law)."""
    world_hz = max(tick_hz_by_name.values())
    if world_hz_override is not None:
        if int(world_hz_override) < world_hz:
            raise ValueError(
                f"--world-hz {world_hz_override} is slower than the fastest "
                f"side ({dict(tick_hz_by_name)} Hz) — the world may run "
                "faster than the deciders, never slower")
        world_hz = int(world_hz_override)
    strides: dict[str, int] = {}
    for name, hz in tick_hz_by_name.items():
        if hz <= 0 or world_hz % hz != 0:
            raise ValueError(
                f"side rates {dict(tick_hz_by_name)} Hz cannot share one "
                f"world: {name!r}'s fixed_tick_hz={hz} does not evenly "
                f"divide the world rate {world_hz} Hz (the fastest side) — "
                "every side's decision cadence must be an integer divisor "
                "of the fastest side's rate")
        strides[name] = world_hz // hz
    return world_hz, strides

# Gate-row layout (qnn.schema GATE_STREAM_SCHEMA_VERSION): the 7 band-v5
# scalars, then the LOOK_COMMIT_STATE_DIM commit lanes, then the 2 look-tangent
# components. Applied through _gate_matrix ONLY — a second hand-written width
# is how the summary block drifted off the row (see _gate_matrix).
# 7 base + look-commit lanes + 2 tangent + 4 schema-6 stationary-menu
# columns (self_weapon_id, weapon_pref, health, weapon_feas)
GATE_ROW_WIDTH = 7 + LOOK_COMMIT_STATE_DIM + 2 + 4


# ── model loading ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SideSpec:
    """One combatant: a trained run + the decode config that made it an rc."""

    name: str
    run_dir: Path
    decode_config: Path

    @classmethod
    def parse(cls, text: str) -> "SideSpec | OnnxSideSpec":
        """``name=run_dir:decode_config.json`` — or ``name=path.onnx`` for a
        DEPLOYED artifact side (decode + look grid baked in-graph, driven by
        qnn.eval.h2h_ort.OrtSide: the cross-generation pairing path, since
        pre-a28 checkpoints load only from their own branches)."""
        name, _, rest = text.partition("=")
        if name and rest.endswith(".onnx") and ":" not in rest:
            return OnnxSideSpec(name=name, onnx_path=Path(rest))
        run_dir, _, decode = rest.partition(":")
        if not (name and run_dir and decode):
            raise ValueError(
                "side spec must be name=run_dir:decode_config.json or "
                f"name=path.onnx, got {text!r}")
        return cls(name=name, run_dir=Path(run_dir), decode_config=Path(decode))

    def checkpoint(self) -> Path:
        # bench layout (checkpoints/best_*.pth | bc_best_model.pth) and PPO
        # run-dir layout (checkpoints/best/best_model.pth) — the same two
        # shapes qnn.diag.load_policy resolves (83f07e51).
        cks = (sorted((self.run_dir / "checkpoints").glob("best_*.pth"))
               or sorted((self.run_dir / "checkpoints").glob("bc_best_model.pth"))
               or sorted((self.run_dir / "checkpoints" / "best").glob("best_model.pth")))
        if not cks:
            raise FileNotFoundError(
                f"{self.run_dir}: no best_*.pth / bc_best_model.pth / "
                "best/best_model.pth checkpoint")
        return cks[0]

    def look_grid(self) -> dict:
        p = self.run_dir / "config" / "look_grid.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: h2h needs the run's pinned look grid")
        return json.loads(p.read_text())

    def fixed_tick_hz(self) -> int:
        """This side's decision cadence, from its train.json (required key —
        matches qnn.run.config's _require_key convention; no 20 Hz fallback:
        an old run missing the stamp must be re-stamped, not silently guessed)."""
        p = self.run_dir / "config" / "train.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: h2h needs the run's train.json for fixed_tick_hz")
        train = json.loads(p.read_text())
        if "fixed_tick_hz" not in train:
            raise ValueError(f"{p}: train.json must define fixed_tick_hz")
        return int(train["fixed_tick_hz"])


@dataclass
class OnnxSideSpec:
    """One combatant given as a DEPLOYED artifact (``name=path.onnx``).

    Decode config, look grid, and RNG are baked in-graph; the side is driven
    by :class:`qnn.eval.h2h_ort.OrtSide` (the Python twin of nq_client's
    ONNX path) — the cross-generation pairing route. ``run_dir`` /
    ``decode_config`` mirror SideSpec for summary stamping only."""

    name: str
    onnx_path: Path
    _side: "object | None" = field(default=None, repr=False)

    @property
    def run_dir(self) -> Path:
        return self.onnx_path

    @property
    def decode_config(self) -> str:
        return "(baked-in-onnx)"

    def load(self, *, seed: int = 0):
        if self._side is None:
            from qnn.eval.h2h_ort import OrtSide  # deferred: needs onnxruntime
            self._side = OrtSide(self.name, self.onnx_path, seed=seed)
        return self._side

    def fixed_tick_hz(self) -> int:
        return int(self.load().tick_hz)

    def look_grid(self) -> dict:  # pragma: no cover — guarded by callers
        raise RuntimeError(
            f"{self.name}: an ONNX side's look grid is baked in-graph — "
            "shared-look-grid installs do not apply")


SHARED_LOOK_GRID_CHOICES = ("refuse", "side-a", "side-b", "per-side")


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

    ``per-side`` is the CROSS-RATE mode (look-actuation-decoupling cell 3):
    each side decodes on its OWN fitted ladder — a 10 Hz-corpus grid and a
    20 Hz-corpus grid are BOTH correct for their own checkpoints, so neither
    refusal nor a shared override is honest. run_h2h re-installs the acting
    side's grid before each of its batched forwards (install_polar_grid is a
    cheap module rebind); side A's is installed here as the initial state.

    Returns the provenance record stamped into each arm's summary.json."""
    from qnn.model.look_bins import install_polar_grid

    if choice not in SHARED_LOOK_GRID_CHOICES:
        raise ValueError(f"shared_look_grid must be one of "
                         f"{list(SHARED_LOOK_GRID_CHOICES)}, got {choice!r}")
    # ONNX sides decode on their own grid IN-GRAPH — the process-global
    # install neither affects nor is needed by them. All-onnx pairings skip
    # entirely; a mixed pairing installs the torch side's own grid (its
    # fitted ladder, no sharing question arises).
    torch_specs = [s for s in specs if not isinstance(s, OnnxSideSpec)]
    if len(torch_specs) < 2:
        if choice != "refuse":
            raise ValueError(
                f"--shared-look-grid={choice} needs two torch sides — an "
                "ONNX side's grid is baked in-graph")
        if torch_specs:
            install_polar_grid(*_grid_install_args(torch_specs[0].look_grid()))
        return {"choice": "onnx-baked",
                "onnx_sides": [s.name for s in specs
                               if isinstance(s, OnnxSideSpec)]}
    grids = [s.look_grid() for s in specs]
    numeric = [_grid_numeric(g) for g in grids]
    identical = numeric[0] == numeric[1]
    if choice == "per-side":
        delta = _grid_max_delta(numeric[0], numeric[1])
        _log(f"--shared-look-grid=per-side: each side decodes on its own "
             f"grid (max numeric delta {delta:.6g}); the acting side's grid "
             "is re-installed before every one of its forwards")
        install_polar_grid(*_grid_install_args(grids[0]))
        return {"choice": choice, "max_numeric_delta": delta}
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
    install_polar_grid(*_grid_install_args(g))
    return {"choice": choice, "max_numeric_delta": delta}


def _grid_install_args(g: Mapping[str, Any]) -> tuple:
    """install_polar_grid's positional args from a parsed look_grid.json."""
    return (
        torch.tensor(g["mag_centers_rad"], dtype=torch.float32),
        torch.tensor(g["dir_centers_rad"], dtype=torch.float32)
        if "dir_centers_rad" in g else None,
        g.get("deadzone_rad"),
    )


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
    if getattr(getattr(model, "model", model), "_has_move_tick_head", False):
        # BENCH ARM C3: the per-tick move decode adopts the BC run's own
        # pinned hazard table — no code-side default (fail-loud contract in
        # policy.act). The h2h side spec carries the run dir.
        from qnn.model.move_tick_decode import params_from_run_dir
        model.move_tick_params = params_from_run_dir(spec.run_dir)
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
    rel_z = float("nan")
    aim_t = aim_h = aim_v = float("nan")
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
                sight = (et[:n] == TOKEN_ACTOR) & (mod == MODALITY_SIGHT)
            elif "entity_recency" in obs:
                rec = np.asarray(obs["entity_recency"],
                                 dtype=np.float64).reshape(-1)[:n]
                sight = (et[:n] == TOKEN_ACTOR) & (rec <= 0.0)
            else:
                raise RuntimeError(
                    "h2h gate stream: this side's obs carries neither "
                    "'entity_modality_id' (a27 percept v3) nor "
                    "'entity_recency' (legacy) — the band-v5 `engaged` mask "
                    "cannot be derived and would silently log 0 for every "
                    f"tick (obs fields: {sorted(obs)})")
            engaged = int(sight.any())
            # rel_z (the E7 in-vivo instrument): WORLD-frame vertical offset
            # of the engaging SIGHT actor, game units, +up = opponent above.
            # entity_rel is the VIEW-basis position (i16 world units) and the
            # basis is tilted by the current pitch, so rotate back with the
            # wire pitch (i8 = deg/90 quantized by 127, Quake +down;
            # world dz = -sin(p)*rel_x + cos(p)*rel_z, roll-free). NaN when
            # not engaged or when the side's obs lacks either field.
            if engaged and "entity_rel" in obs and "view_pitch" in obs:
                rel = np.asarray(obs["entity_rel"],
                                 dtype=np.float64).reshape(-1, 3)[:n]
                i = int(np.argmax(sight))
                p = (float(np.asarray(obs["view_pitch"]).reshape(-1)[0])
                     / 127.0) * (np.pi / 2.0)
                rel_z = float(-np.sin(p) * rel[i, 0] + np.cos(p) * rel[i, 2])
            # AIM ERROR (E7 instrument, ATTACK-GATE RESPEC 2026-08-15):
            # angle (deg) to the MOST-ALIGNED in-SIGHT actor (smallest total
            # error, not just the first sight token like rel_z above),
            # view-basis (crosshair = +x), current-position anchor. No pitch
            # rotation needed (view-basis IS the crosshair frame). entity_rel
            # is wire-quantized int16 game units — cast to float64 before any
            # arithmetic (guards int16 hypot/square overflow). NaN when not
            # engaged or entity_rel absent. Mirrors qnn.eval.run._log_streams
            # (feature-parity mirror; ADDITIVE, never touches the gate row).
            if engaged and "entity_rel" in obs:
                rel_v = np.asarray(obs["entity_rel"],
                                   dtype=np.float64).reshape(-1, 3)[:n]
                if sight.any():
                    rvx, rvy, rvz = rel_v[:, 0], rel_v[:, 1], rel_v[:, 2]
                    h_arr = np.degrees(np.arctan2(rvy, rvx))
                    v_arr = np.degrees(np.arctan2(rvz, np.hypot(rvx, rvy)))
                    t_arr = np.degrees(np.arctan2(np.hypot(rvy, rvz), rvx))
                    best = int(np.argmin(np.where(sight, t_arr, np.inf)))
                    aim_t, aim_h, aim_v = (float(t_arr[best]),
                                          float(h_arr[best]), float(v_arr[best]))
    # ATTACK LANE. Exact on both conventions: a26's attack slot IS the fire
    # bit (``weapon`` rides its own key), and a27's folded lane is gated on
    # the fire bit before it leaves policy.act (``attack != 0`` iff firing —
    # decode_actions.attack_lane_gate), so the truth-value read here is the
    # model's fire decision either way. See :func:`attack_lane_convention`.
    attack = int(act.get("attack", 0) or 0)
    af = obs.get("attack_finished") if isinstance(obs, dict) else None
    af_expired = (af is not None
                 and float(np.asarray(af).reshape(-1)[0]) <= 1e-6)
    disch = int(bool(attack) and af_expired)
    # op_ready: the SAME predicate as ``disch``'s cooldown check, independent
    # of whether the model actually attacked this tick — the
    # attack_conditional gate arm's denominator (qnn.decode_fit.gates),
    # matched to the human corpus op-filter (qnn.human.attack_conditional).
    ready_g = 1.0 if af_expired else 0.0
    wimp = 0
    if isinstance(obs, dict) and "self_weapon_id" in obs:
        wimp = int(self_weapon_id_to_impulse(
            int(np.asarray(obs["self_weapon_id"]).reshape(-1)[0])))
    if has_look_commit:
        lc = tuple(int(v) for v in np.asarray(
            st.look_commit).reshape(-1)[:LOOK_COMMIT_STATE_DIM])
    else:
        lc = (-1,) * LOOK_COMMIT_STATE_DIM
    # schema-6 stationary-menu columns (qnn.schema gate schema 6):
    #   self_weapon_id — equip impulse (0 on wire.13 sides, whose declaration
    #     carries no self_weapon_id: none/unknown per schema);
    #   weapon_pref — the discharging weapon at discharge ticks (equip where
    #     known, else the attack-with class — the a27 folded convention);
    #   health + weapon_feas — the preference ruler's invalidators.
    wpref = disch * (wimp if wimp else attack)
    health = 0
    feas = 0
    if isinstance(obs, dict) and all(
            k in obs for k in ("health", "self_items", "ammo_shells",
                               "ammo_nails", "ammo_rockets", "ammo_cells")):
        health = int(np.asarray(obs["health"]).reshape(-1)[0])
        feas = int(weapon_feasibility_bits(
            np.asarray(obs["self_items"]).reshape(-1)[:1],
            np.asarray(obs["ammo_shells"]).reshape(-1)[:1],
            np.asarray(obs["ammo_nails"]).reshape(-1)[:1],
            np.asarray(obs["ammo_rockets"]).reshape(-1)[:1],
            np.asarray(obs["ammo_cells"]).reshape(-1)[:1])[0])
    st.gate_trace.append(
        (attack, int(act.get("weapon", 0) or 0), turn, keep, disch, wimp,
         engaged) + lc + (tan_y, tan_z, wimp, wpref, health, feas))
    st.relz_trace.append(rel_z)
    st.aim_err_trace.append((aim_t, aim_h, aim_v, ready_g))


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
    relz_rows: list[float] = field(default_factory=list)
    aimerr_rows: list[tuple] = field(default_factory=list)
    offsets: list[int] = field(default_factory=lambda: [0])

    def close_segment(self, st: _EpisodeState) -> None:
        if st.move_trace:
            self.move_rows.extend(st.move_trace)
            self.gate_rows.extend(st.gate_trace)
            self.relz_rows.extend(st.relz_trace)
            self.aimerr_rows.extend(st.aim_err_trace)
            self.offsets.append(len(self.move_rows))
        st.move_trace = []
        st.gate_trace = []
        st.relz_trace = []
        st.aim_err_trace = []

    def write(self, path: Path, *, tick_hz: int) -> None:
        mv = np.asarray(self.move_rows, dtype=np.int64).reshape(-1, 3)
        gt = _gate_matrix(self.gate_rows)
        ae = np.asarray(self.aimerr_rows, dtype=np.float64).reshape(-1, 4)
        np.savez_compressed(
            path,
            tick_hz=np.asarray([float(tick_hz)]),
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
            # schema 6: stationary-menu streams (see _log_lane_tick).
            self_weapon_id=gt[:, 14].astype(np.int8),
            weapon_pref=gt[:, 15].astype(np.int8),
            health=gt[:, 16].astype(np.int16),
            weapon_feas=gt[:, 17].astype(np.uint8),
            # rel_z: ADDITIVE named array, NOT part of the schema-governed gate
            # row (the row layout and GATE_STREAM_SCHEMA_VERSION are shared with
            # qnn.eval.run and decode-fit cache keys — this column is h2h-only).
            # World-frame vertical offset of the engaging SIGHT actor, game
            # units, +up = opponent above; NaN when not engaged or the side's
            # obs lacks entity_rel/view_pitch. See _log_lane_tick.
            rel_z=np.asarray(self.relz_rows, dtype=np.float32),
            # aim_err_{,h_,v_}deg: ADDITIVE named arrays (ATTACK-GATE RESPEC,
            # 2026-08-15), same "never touches the gate row / schema version"
            # convention as rel_z above. Angle (deg) to the most-aligned
            # in-SIGHT actor, view-basis (crosshair = +x), current-position
            # anchor; NaN when not engaged or entity_rel absent. See
            # _log_lane_tick; mirrors qnn.eval.run._log_streams for
            # writer feature-parity.
            aim_err_deg=ae[:, 0].astype(np.float32),
            aim_err_h_deg=ae[:, 1].astype(np.float32),
            aim_err_v_deg=ae[:, 2].astype(np.float32),
            # op_ready: attack_finished expired, independent of whether the
            # side fired this tick — see _log_lane_tick.
            op_ready=ae[:, 3].astype(bool),
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

# Weapon name -> IT_ bit, mirroring QNN_ArenaWeaponBit (qnn_arena_server_main.c)
# and qnn.engine_norm's constants — the encoding a --inventory multi-weapon
# loadout ORs into ``weapons_mask`` (see _inventory_weapon_config below).
_WEAPON_NAME_BITS: dict[str, int] = {
    "axe": IT_AXE, "shotgun": IT_SHOTGUN, "super_shotgun": IT_SUPER_SHOTGUN,
    "nailgun": IT_NAILGUN, "super_nailgun": IT_SUPER_NAILGUN,
    "grenade_launcher": IT_GRENADE_LAUNCHER,
    "rocket_launcher": IT_ROCKET_LAUNCHER, "lightning": IT_LIGHTNING,
}


def _inventory_weapon_config(inventory: Mapping[str, object]) -> dict[str, object]:
    """``--inventory`` JSON (the same ``{"weapons": [...], "selected_weapon":
    ..., "shells": ..., ...}`` shape scenario configs already use — see
    qnn.decode_fit.instruments) -> the ``ArenaGridBackend`` weapon_config
    dict BOTH h2h seats equip (the arena's standing convention: it equips
    every seat, there is no bystander).

    Reuses ``qnn.eval.run._arena_weapon_config`` VERBATIM for the ammo/armor/
    health/model_weapon translation — single- or zero-weapon inventories
    produce byte-identical cfg dicts to what an aim-grid scenario gets.
    ``_arena_weapon_config`` itself stays single-weapon-only by design (the
    arena_grid cell invariant pinned by
    tests/test_per_lane_decode.py::test_arena_weapon_config_rejects_multi_weapon),
    so multi-weapon support — a full-arsenal loadout like the real-map RA
    venue — is layered here instead of loosening that contract: len(weapons)
    > 1 ORs the named weapons into ``weapons_mask``, the raw owned-items
    bitmask wired straight onto the engine's ``qnn_inv_weapons`` cvar via
    ``-qnn_inv_weapons_mask`` (see engine.arena_bridge._WEAPON_ARG_FLAGS and
    QNN_ArenaApplyInventory in qnn_arena_server_main.c). ``selected_weapon``,
    if also given, still names the STARTING weapon on top of the mask and
    must be one of ``weapons``."""
    inv = dict(inventory or {})
    weapons = [str(w) for w in (inv.get("weapons") or [])]
    if len(weapons) <= 1:
        return _arena_weapon_config({"inventory": inv})
    unknown = [w for w in weapons if w not in _WEAPON_NAME_BITS]
    if unknown:
        raise ValueError(f"--inventory: unknown weapon name(s) {unknown}")
    sel = inv.get("selected_weapon")
    if sel is not None and str(sel) not in weapons:
        raise ValueError(
            f"--inventory: selected_weapon={sel!r} is not a member of "
            f"weapons={weapons}")
    single_weapon_inv = dict(inv)
    single_weapon_inv.pop("weapons", None)
    if sel is None:
        single_weapon_inv.pop("selected_weapon", None)
    cfg = _arena_weapon_config({"inventory": single_weapon_inv})
    mask = 0
    for w in weapons:
        mask |= _WEAPON_NAME_BITS[w]
    cfg["weapons_mask"] = mask
    return cfg


def run_h2h(side_a: SideSpec, side_b: SideSpec, *, out_dir: Path,
            rounds: int, matches: int = 8, map_id: str = "qnn_arena8",
            seed: int = 43, base_port: int = 29000,
            reward_json: Path | None = None, pin_weapon: str | None = None,
            inventory: Mapping[str, object] | None = None,
            shared_look_grid: str = "refuse", force_look_commit: str = "none",
            actuation_a: str | None = None, actuation_b: str | None = None,
            world_hz_override: int | None = None,
            server_executable: str = "assets/bin/ppo_arena_server",
            client_executable: str = "assets/bin/ppo_arena_client",
            basedir: str = "assets") -> dict[str, Any]:
    """Play ``rounds`` bouts (split evenly across ``matches`` sealed 1v1s) and
    write rounds.jsonl + per-side stream npz + summary.json into ``out_dir``.

    ``matches`` must be even, with one exception (1): seating is a seeded
    balanced assignment (side_of_lane below), and an odd count would leave
    one side sitting one extra match in whichever seat carries a venue
    advantage — exactly the bias alternation exists to cancel. ``matches=1``
    is the untagged-real-map case (one match per server instance — see
    frikbotnex/client.qc SelectSpawnPoint's untagged fallback, which refuses
    matches>1 on a map with no per-match spawn tiling): with a single match
    there is no cross-match parity to balance, so a single coin flip (not the
    halving shuffle) picks its one seat assignment.

    ``shared_look_grid`` / ``force_look_commit`` are the two same-checkpoint-A/B
    escape hatches; both are stamped into summary.json (see
    :func:`_install_shared_look_grid` and :func:`_force_look_commit`). They name
    sides A/B, NOT seats — which seat a side sits in alternates by match, these
    knobs never do.

    ``actuation_a`` / ``actuation_b`` (``qnn.eval.look_actuation.
    LOOK_ACTUATION_CHOICES``) name the LOOK actuator each side plays through
    when its own decision cadence is slower than the world (``stride > 1`` —
    see :func:`_resolve_world_rate`); REQUIRED then, ignored at stride 1.
    They also name sides A/B, not seats.

    ``inventory`` (mutually exclusive with ``pin_weapon`` — both derive the
    arena's ONE shared weapon_config, applied to both seats, the standing
    equip-everybody convention) is a scenario ``inventory`` JSON object
    translated by :func:`_inventory_weapon_config`; unlike ``pin_weapon`` it
    supports a full-arsenal loadout (e.g. real-map RA venue matches)."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # matches=1 is the other exception: real (untagged) maps only host ONE
    # match per server instance (SelectSpawnPoint's untagged fallback refuses
    # a second concurrent match — no per-match physical isolation without the
    # mapgen grid's tiling), so there is no cross-match seat-parity to
    # balance in the first place; seat fairness for that single match comes
    # from the engine's own per-round spawn rotation instead (see
    # frikbotnex/client.qc SelectSpawnPoint).
    if matches != 1 and matches % 2 != 0:
        raise ValueError(
            f"matches={matches} is odd — seating is a balanced random "
            "assignment (exactly half the matches seat side A at 0), so an "
            "odd count cannot balance: one side would get one extra match "
            "in whichever seat carries a venue advantage (matches=1 is the "
            "one exception — a single real-map match has no cross-match "
            "parity to balance)")
    if pin_weapon is not None and inventory is not None:
        raise ValueError(
            "--pin-weapon and --inventory both set the arena's weapon_config "
            "— they are mutually exclusive")
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
    # The arena server runs ONE world, at the FASTER side's rate; the slower
    # side's own rate must divide it evenly (_resolve_world_rate) — its model
    # decides only every `stride` world ticks (see below).
    tick_hz_a, tick_hz_b = side_a.fixed_tick_hz(), side_b.fixed_tick_hz()
    world_hz, strides_by_name = _resolve_world_rate(
        {side_a.name: tick_hz_a, side_b.name: tick_hz_b},
        world_hz_override=world_hz_override)
    if actuation_a is not None and actuation_a not in LOOK_ACTUATION_CHOICES:
        raise ValueError(f"actuation_a must be one of {LOOK_ACTUATION_CHOICES}, "
                         f"got {actuation_a!r}")
    if actuation_b is not None and actuation_b not in LOOK_ACTUATION_CHOICES:
        raise ValueError(f"actuation_b must be one of {LOOK_ACTUATION_CHOICES}, "
                         f"got {actuation_b!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The grid hatch names side A/B; _install_shared_look_grid takes them in
    # that order regardless of which seat either side plays a given match.
    grid_prov = _install_shared_look_grid((side_a, side_b), shared_look_grid)
    num_lanes = 2 * matches
    # side_of_lane[lane]: which SIDE (0 = side_a, 1 = side_b) sits this lane's
    # seat (lane % 2) for this lane's match (lane // 2). Exactly half the
    # matches (seeded shuffle) seat side A at 0 — balanced so any seat-fixed
    # advantage (tiered maps spawn seat 1 elevated by design) lands on both
    # sides equally, and RANDOMIZED rather than parity-alternated because the
    # arena cells themselves are not interchangeable: the fixed-seating h2h
    # archive shows seat 0 winning 50.9% in even-column cells vs 46.0% in odd
    # (45k rounds, z=10.5, cause unisolated), and any deterministic pattern
    # risks re-aliasing that venue structure onto the side split.
    seat_flip_rng = torch.Generator().manual_seed(seed * 2_654_435_761 % (2**31))
    if matches == 1:
        # No cross-match halving to shuffle — one coin flip picks which side
        # sits seat 0 for the run's only match.
        seat_flips = [int(torch.randint(0, 2, (1,), generator=seat_flip_rng).item())]
    else:
        seat_flips = [0] * (matches // 2) + [1] * (matches // 2)
        seat_flips = [seat_flips[i] for i in
                      torch.randperm(matches, generator=seat_flip_rng).tolist()]
    side_of_lane = [(lane % 2) ^ seat_flips[lane // 2] for lane in range(num_lanes)]
    lanes_of_side = {s: [lane for lane in range(num_lanes) if side_of_lane[lane] == s]
                     for s in (0, 1)}
    # per-side grids (cross-rate cells): the acting side's own ladder is
    # re-installed before each of its forwards in the decide loop below.
    # None ⇒ a single shared grid is already installed process-wide.
    grid_args_by_side = (
        {0: _grid_install_args(side_a.look_grid()),
         1: _grid_install_args(side_b.look_grid())}
        if shared_look_grid == "per-side" else None)
    forced = {side_a.name: force_look_commit in ("a", "both"),
              side_b.name: force_look_commit in ("b", "both")}
    for spec in (side_a, side_b):
        if isinstance(spec, OnnxSideSpec) and forced[spec.name]:
            raise ValueError(f"{spec.name}: --force-look-commit cannot apply "
                             "to an ONNX side (decode is baked in-graph)")
    models = {s: (spec.load(seed=seed + 31 * s)
                  if isinstance(spec, OnnxSideSpec)
                  else load_side(spec, force_look_commit=forced[spec.name]))
              for s, spec in ((0, side_a), (1, side_b))}
    for spec in (side_a, side_b):
        if forced[spec.name]:
            _log(f"--force-look-commit: {spec.name} plays the look_seg "
                 "COMMITMENT decode (shape default overridden)")
    # Per-side look-commitment flag: decides whether the stream's lc_* lanes
    # carry this side's look decode state or the -1 "no look_seg head"
    # sentinel (the two sides can differ — cross-arch pairings are the point).
    has_look_commit = {s: bool(getattr(models[s], "look_commitment", False))
                       for s in (0, 1)}
    # Per-SIDE stride + actuator choice, keyed by NAME like the other
    # same-checkpoint-A/B knobs and then re-keyed to side index 0/1.
    stride = {0: strides_by_name[side_a.name], 1: strides_by_name[side_b.name]}
    actuation_by_name = {side_a.name: actuation_a, side_b.name: actuation_b}
    actuation = {0: actuation_by_name[side_a.name], 1: actuation_by_name[side_b.name]}
    for s, spec in ((0, side_a), (1, side_b)):
        if stride[s] > 1 and isinstance(spec, OnnxSideSpec):
            raise ValueError(
                f"{spec.name}: ONNX sides decide every world tick only "
                f"(stride {stride[s]} in a {world_hz} Hz world) — no "
                "actuation window is built for the deployed-artifact path")
        if stride[s] > 1 and actuation[s] is None:
            raise ValueError(
                f"{spec.name} decides at {spec.fixed_tick_hz()} Hz in a "
                f"{world_hz} Hz world (stride {stride[s]}) — a stride>1 side "
                "needs an explicit --actuation-a/--actuation-b choice "
                f"({list(LOOK_ACTUATION_CHOICES)}); there is no default "
                "actuator to fall back to silently")
        if stride[s] == 1 and actuation[s] is not None:
            _log(f"{spec.name}: stride 1 (decides every world tick) — "
                 f"--actuation={actuation[s]!r} is ignored (no gaps to fill)")
        if actuation[s] == "stroke" and not has_look_commit[s]:
            raise ValueError(
                f"{spec.name}: --actuation=stroke needs the look_seg "
                "COMMITMENT decode (model.look_commitment) — this checkpoint "
                "is not playing it (shape-derived polar side, or a dual-head "
                "checkpoint not covered by --force-look-commit); there is no "
                "commitment state to retime")
    # Per-lane stroke executor state (lazily unused for step/smear sides).
    # ``hz`` is the SIDE's own native rate — tick_hz_a/tick_hz_b are side A/B's
    # rates already, so no re-keying by seat is needed here.
    side_hz = {0: side_a.fixed_tick_hz(), 1: side_b.fixed_tick_hz()}
    stroke_exec: dict[int, StrokeExecutor] = {}
    if actuation[0] == "stroke":
        stroke_exec.update({lane: StrokeExecutor(stride=stride[0], hz=side_hz[0])
                            for lane in lanes_of_side[0]})
    if actuation[1] == "stroke":
        stroke_exec.update({lane: StrokeExecutor(stride=stride[1], hz=side_hz[1])
                            for lane in lanes_of_side[1]})
    # Per-side obs declarations (obs-api): each side plays on the wire its
    # checkpoint trained on — cross-arch pairings get heterogeneous frames in
    # one world via the per-side emit plans. An ONNX side's declaration is
    # its own stamped obs_declaration (exactly what the live engine compiles
    # its emit plan from).
    from qnn.obs_api import declaration_for_run

    def _seat_decl(spec, s):
        if isinstance(spec, OnnxSideSpec):
            meta = models[s].meta
            if "obs_declaration" in meta:
                return json.loads(meta["obs_declaration"])
            # pre-obs-api export: resolve the canonical declaration from the
            # wire stamp — the same shim table declaration_for_run uses for
            # a26-line run dirs (full-stream v1 entities, self_weapon_id
            # exposed). Bare stamps refuse inside WIRE_SHIM lookup.
            wire = meta.get("wire_contract")
            from qnn.obs_api import WIRE_SHIM
            if wire not in WIRE_SHIM:
                raise ValueError(
                    f"{spec.name}: {spec.onnx_path} has neither an "
                    f"obs_declaration stamp nor a shim-resolvable wire "
                    f"stamp ({wire!r}) — re-stamp or re-export it")
            _log(f"{spec.name}: pre-obs-api export — declaration resolved "
                 f"from wire shim {wire}")
            return WIRE_SHIM[wire].declaration
        return declaration_for_run(spec.run_dir)

    decls = {0: _seat_decl(side_a, 0), 1: _seat_decl(side_b, 1)}
    names = {0: side_a.name, 1: side_b.name}
    _log(f"side_a={side_a.name} side_b={side_b.name} map={map_id} "
         f"matches={matches} rounds={rounds} "
         f"seat_flips={''.join(map(str, seat_flips))} "
         f"world_hz={world_hz} "
         f"stride={{0: {stride[0]}, 1: {stride[1]}}} "
         f"actuation={{0: {actuation[0]!r}, 1: {actuation[1]!r}}}")

    rw = RewardWeights.from_json(
        reward_json if reward_json else
        Path("runs/eval/seed43_stage6_template/config/reward.json"))
    if inventory is not None:
        arena_scenario_id = f"h2h-{map_id}-inventory"
        arena_weapon_config = _inventory_weapon_config(inventory)
    elif pin_weapon is not None:
        arena_scenario_id = f"h2h-{map_id}-pin-{pin_weapon}"
        arena_weapon_config = {**PIN_WEAPON_CONFIGS[pin_weapon], **_PIN_BASE}
    else:
        arena_scenario_id = f"h2h-{map_id}"
        arena_weapon_config = None
    backend = ArenaGridBackend(
        num_lanes=num_lanes,
        server_executable=server_executable,
        client_executable=client_executable,
        basedir=basedir, workdir=None, map_id=map_id,
        matches_per_server=matches, seat_mode="self_play",
        base_port=base_port, bot_skill=0,
        max_steps_per_episode=_max_steps_per_round(world_hz),
        fixed_tick_hz=world_hz, reward_weights=rw,
        direct_actions=True, observer_mode="virtual",
        scenario_id=arena_scenario_id,
        weapon_config=arena_weapon_config,
        declarations=[decls[side_of_lane[lane]] for lane in range(num_lanes)],
    )

    rounds_per_match = max(1, rounds // matches)
    rounds_done = [0] * matches
    round_start_tick = [0] * matches
    match_live = [True] * matches
    ledgers = {lane: _LaneLedger() for lane in range(num_lanes)}
    streams = {0: _SideStreams(), 1: _SideStreams()}
    # Per-side action convention, resolved from the first tick's action dict
    # and stamped into summary.json (see :func:`attack_lane_convention`).
    attack_lane: dict[int, str | None] = {0: None, 1: None}
    round_records: list[dict[str, Any]] = []
    wins = {side_a.name: 0, side_b.name: 0, "draw": 0}

    rng = {s: torch.Generator(device="cpu") for s in (0, 1)}
    for s in (0, 1):
        rng[s].manual_seed(seed + 7919 * s)

    try:
        obs = backend.reset()
        states: list[_EpisodeState] = [
            _fresh_state(models[side_of_lane[lane]], _obs_row(obs, lane),
                         seed * 1_000_003 + lane)
            for lane in range(num_lanes)
        ]
        for lane in range(num_lanes):
            if getattr(models[side_of_lane[lane]], "is_ort", False):
                models[side_of_lane[lane]].reset_lane(lane)

        tick = 0
        t0 = time.perf_counter()
        # Per-lane HELD action (move/attack/jump/weapon — everything but
        # look) from that lane's most recent decision, and its precomputed
        # per-world-tick LOOK window (consumed one entry per world tick until
        # the side's next decision refills it). Both are only ever consulted
        # for a stride>1 side's lanes; a stride-1 side's lanes take the fresh
        # decoded action verbatim every tick (see the assembly loop below),
        # unchanged from pre-stride h2h.
        held_action: dict[int, Mapping[str, Any]] = {}
        look_window: dict[int, list] = {}
        while any(match_live):
            # DECIDE: one batched forward per SIDE, but only on that side's
            # OWN decision ticks (tick % stride == 0 — stride 1 decides every
            # tick, the byte-identical fast path). States/RNG stay
            # side-isolated exactly as before; a held-over tick touches
            # neither the model nor the hidden state.
            for s in (0, 1):
                if tick % stride[s] != 0:
                    continue
                lanes = lanes_of_side[s]
                if getattr(models[s], "is_ort", False):
                    # deployed-artifact side: per-lane ORT forward; decode
                    # state (incl. hidden) lives inside the OrtSide.
                    acts = models[s].act_lanes(
                        lanes, [states[lane].obs for lane in lanes])
                    for i, lane in enumerate(lanes):
                        held_action[lane] = acts[i]
                    continue
                if grid_args_by_side is not None:
                    # cross-rate per-side grids: decode THIS side's look head
                    # on its own fitted ladder (cheap module rebind).
                    from qnn.model.look_bins import install_polar_grid
                    install_polar_grid(*grid_args_by_side[s])
                sts = [states[lane] for lane in lanes]
                acts, next_hidden = _select_actions_batched(
                    models[s], "sampled", sts, rng[s])
                for i, lane in enumerate(lanes):
                    states[lane].hidden = next_hidden[i]
                    held_action[lane] = acts[i]
                    if stride[s] > 1:
                        look_window[lane] = actuate_window(
                            actuation[s], acts[i]["look"], stride[s],
                            stroke_exec=stroke_exec.get(lane),
                            look_commit_state=states[lane].look_commit)

            # ASSEMBLE this tick's realized per-lane action: held fields
            # verbatim, look from the actuator window on a stride>1 side
            # (one entry consumed per world tick), verbatim on a stride-1
            # side (no window — this IS the pre-stride action dict, so a
            # stride-1/stride-1 pairing is byte-identical to current h2h).
            actions_by_lane: dict[int, Mapping[str, Any]] = {}
            for lane in range(num_lanes):
                s = side_of_lane[lane]
                if stride[s] == 1:
                    actions_by_lane[lane] = held_action[lane]
                else:
                    act = dict(held_action[lane])
                    act["look"] = look_window[lane].pop(0)
                    actions_by_lane[lane] = act

            # Pre-step context snapshots + stream rows (pre-step obs semantics,
            # same as run.py's overlap window).
            ctx = [_lane_ctx(states[lane]) for lane in range(num_lanes)]
            for lane in range(num_lanes):
                _log_lane_tick(states[lane], actions_by_lane[lane],
                               has_look_commit=has_look_commit[side_of_lane[lane]])
            for s in (0, 1):
                if attack_lane[s] is None:
                    attack_lane[s] = attack_lane_convention(
                        actions_by_lane[lanes_of_side[s][0]])
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
                    "winner": None if winner is None else names[side_of_lane[winner]],
                    # winner_seat is the actual engine seat (lane % 2), which
                    # side sits it varies match to match — see side_of_lane.
                    "winner_seat": None if winner is None else winner % 2,
                    "reason": reason,
                    "seats": {},
                }
                for lane in (l0, l1):
                    rec["seats"][names[side_of_lane[lane]]] = {
                        "seat": lane % 2, "died": died[lane],
                        **ledgers[lane].snapshot(), **ctx[lane]}
                round_records.append(rec)
                # Incremental flush: a stopped/killed run keeps every
                # completed round (summaries/streams still land at the end).
                with (out_dir / "rounds.jsonl").open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                wins["draw" if winner is None else names[side_of_lane[winner]]] += 1
                rounds_done[m] += 1
                round_start_tick[m] = tick

                for lane in (l0, l1):
                    streams[side_of_lane[lane]].close_segment(states[lane])
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
                        models[side_of_lane[lane]],
                        _obs_row(obs, lane) if not timed_out else states[lane].obs,
                        seed * 1_000_003 + 977 * lane + 7 * rounds_done[m])
                    if getattr(models[side_of_lane[lane]], "is_ort", False):
                        # round boundary = h2h's episode: episode-scoped
                        # loopback lanes re-init, rng lanes persist (the
                        # torch side's per-round _fresh_state twin).
                        models[side_of_lane[lane]].reset_lane(lane)

            if tick % 600 == 0:
                done = sum(rounds_done)
                rate = tick / max(time.perf_counter() - t0, 1e-9)
                _log(f"tick {tick}  rounds {done}/{rounds_per_match * matches}  "
                     f"{rate:.0f} ticks/s  wins={wins}")
    finally:
        backend.close()

    total = sum(rounds_done)
    decided = wins[side_a.name] + wins[side_b.name]
    summary: dict[str, Any] = {
        "sides": {"a": side_a.name, "b": side_b.name},
        "specs": {spec.name: {"run_dir": str(spec.run_dir),
                              "decode_config": str(spec.decode_config),
                              # which look mechanism this side actually played
                              "force_look_commit": forced[spec.name],
                              "look_commitment": has_look_commit[s],
                              "fixed_tick_hz": side_hz[s],
                              "stride": stride[s],
                              "actuation": actuation[s]}
                  for s, spec in ((0, side_a), (1, side_b))},
        "world_hz": world_hz,
        "map": map_id, "matches": matches, "seating": "balanced-random",
        "seat_flips": seat_flips,
        "seed": seed,
        "pin_weapon": pin_weapon,
        "inventory": inventory,
        "shared_look_grid": grid_prov,
        "rounds": total, "wins": wins,
        "decided_rounds": decided,
    }
    for s, spec in ((0, side_a), (1, side_b)):
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
                            / max(n_eng / world_hz, 1e-9), 3),
                    }
        summary[spec.name] = {
            "win_rate_decided": round(w / decided, 4) if decided else None,
            "win_rate_ci95": [round(lo, 4), round(hi, 4)],
            "deaths": deaths, "deaths_by_suicide": suicides,
            "airborne_death_frac": round(air_deaths / max(deaths, 1), 4),
            "standstill_frac": round(
                float(((mv[:, 0] == 1) & (mv[:, 1] == 1)).mean()), 4) if len(mv) else None,
            "jump_press_rate_per_s": round(
                float((mv[:, 2] == 2).mean()) * world_hz, 3) if len(mv) else None,
            "engaged_frac": round(float((gt[:, 6] > 0).mean()), 4) if len(gt) else None,
            # "a26" ⇒ attack = fire bit, weapon = switch impulse on its own
            # key. "a27" ⇒ one folded lane, nonzero = firing WITH that weapon.
            # The rates below are the fire decision either way; the stamp says
            # how the two sides' action dicts differ (attack_lane_convention).
            "attack_lane_convention": attack_lane[s],
            "discharges_per_s": round(
                float((gt[:, 4] > 0).sum()) / (ticks_total / world_hz), 3),
            "damage_dealt_per_round": round(sum(
                r["seats"][spec.name]["damage_dealt"] for r in side_rounds)
                / max(total, 1), 1),
            "per_weapon": per_weapon,
        }

    # rounds.jsonl was flushed incrementally; rewrite once for a clean file
    # (dedupes any partial line from a mid-write kill).
    (out_dir / "rounds.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in round_records))
    for s, spec in ((0, side_a), (1, side_b)):
        streams[s].write(out_dir / f"side_{spec.name}_streams.npz", tick_hz=world_hz)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    _log(f"done: {total} rounds  wins={wins}  -> {out_dir}")
    return summary


def _parse_inventory_arg(value: str) -> dict[str, Any]:
    """``--inventory`` value: a literal JSON object, or a path to a .json
    file holding one (checked first so an existing path never gets treated
    as malformed inline JSON)."""
    path = Path(value)
    text = path.read_text() if path.is_file() else value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--inventory: invalid JSON ({exc})") from exc
    if not isinstance(parsed, MappingABC):
        raise argparse.ArgumentTypeError(
            f"--inventory must be a JSON object, got {type(parsed).__name__}")
    return dict(parsed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.eval.h2h", description=__doc__)
    ap.add_argument("--side-a", required=True, type=SideSpec.parse,
                    help="name=run_dir:decode_config.json")
    ap.add_argument("--side-b", required=True, type=SideSpec.parse)
    ap.add_argument("--rounds", type=int, default=160,
                    help="total rounds (split evenly across matches)")
    ap.add_argument("--matches", type=int, default=8,
                    help="sealed 1v1 matches (MUST be even: seating is a "
                         "seeded balanced random assignment — half the "
                         "matches seat side A at 0; 1 is the one exception, "
                         "for untagged real maps which host a single match "
                         "per server instance)")
    ap.add_argument("--map", default="qnn_arena8")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--base-port", type=int, default=29000)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reward-json", type=Path, default=None)
    ap.add_argument("--pin-weapon", choices=sorted(PIN_WEAPON_CONFIGS),
                    default=None,
                    help="single-weapon pin for BOTH seats (botpin regime: "
                         "owned+selected, infinite ammo)")
    ap.add_argument("--inventory", type=_parse_inventory_arg, default=None,
                    help="scenario inventory JSON (literal or a .json file "
                         "path) applied to BOTH seats — "
                         "{\"weapons\": [...], \"selected_weapon\": ..., "
                         "\"shells\"/\"nails\"/\"rockets\"/\"cells\": ..., "
                         "\"armor_value\"/\"armor_type\"/\"health\": ...}; "
                         "see _inventory_weapon_config. Mutually exclusive "
                         "with --pin-weapon. No flag = today's engine "
                         "defaults, unchanged.")
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
    ap.add_argument("--actuation-a", choices=list(LOOK_ACTUATION_CHOICES),
                    default=None,
                    help="side A's LOOK actuator when its own fixed_tick_hz "
                         "is slower than the pairing's world rate (stride>1 "
                         "— required then; ignored at stride 1)")
    ap.add_argument("--actuation-b", choices=list(LOOK_ACTUATION_CHOICES),
                    default=None, help="side B's LOOK actuator (see --actuation-a)")
    ap.add_argument("--world-hz", type=int, default=None,
                    help="run the WORLD faster than the fastest decider "
                         "(e.g. 20 with two 10 Hz sides -> strides 2/2 so "
                         "both actuators have engine ticks to play through; "
                         "never slower than the fastest side; every side's "
                         "rate must divide it)")
    a = ap.parse_args(argv)
    run_h2h(a.side_a, a.side_b, out_dir=a.out, rounds=a.rounds,
            matches=a.matches, map_id=a.map, seed=a.seed,
            base_port=a.base_port, reward_json=a.reward_json,
            pin_weapon=a.pin_weapon, inventory=a.inventory,
            shared_look_grid=a.shared_look_grid,
            force_look_commit=a.force_look_commit,
            actuation_a=a.actuation_a, actuation_b=a.actuation_b,
            world_hz_override=a.world_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
