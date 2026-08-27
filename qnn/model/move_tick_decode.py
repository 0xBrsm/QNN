"""BENCH ARM — the a24 sticky-tau + semi-Markov hazard MOVE decode, revived.

Companion to :mod:`qnn.model.move_tick_head` for cell C3 of
``agents/plans/seg-vs-frame-decision.md``. **Not a canonical decode law** —
the canonical move decode is the a25 commitment decode
(``qnn.model.decode_actions.move_commit_step``).

Why it must exist for the arm to be fair
----------------------------------------
Per-frame INDEPENDENT sampling of a per-tick move head destroys the human
88-99% hold autocorrelation (2.6-2.7x over-switch, the a24 finding that
motivated retiring the head). The per-tick head only ever worked WITH this
decode; a C3 arm that sampled the head per frame would be a strawman. The
whole point of the cell is: per-tick head + its medicine vs seg head +
model-native commitments.

Provenance
----------
Recovered from ``2a4db619^:src/qnn/model/bench/a24/decode.py``
(``move_decode_step``, the eager reference that was bit-for-bit with
``qnn_onnx_decode_core``, qnn_onnx.c:887-1059). Re-expressed BATCHED over rows
in the conventions of the a25 decode (``move_commit_step``): state mutated in
place in the caller's ``move_state`` lanes, ``row_generators`` for per-lane
reproducible streams, stochastic law inert under ``greedy``.

Pinned decode parameters (neutral bench operating point)
--------------------------------------------------------
* ``sticky_tau = 0.6`` on BOTH axes — the C engine's LOAD-TIME DEFAULT
  (``move_decode_params(tau_fb=0.6, tau_lr=0.6)``, mirroring
  ``qnn_onnx_select_codec``, qnn_onnx.c:1681-1742; src/docs/move-decode-c-oracle.md
  §1). The a24 rc lineage shipped FITTED taus instead (a24rc1 0.74/0.92,
  a24rc2 0.95/0.95, a24rc4 0.8/0.99) — those are decode-FIT outputs for
  specific a24 checkpoints and are NOT transferable across architectures
  (feedback: decode/guards don't transfer across archs). The un-fitted
  default is the honest neutral point for a fresh arm; the arm's own
  decode-fit, if it ever runs, replaces it.
* Dwell hazard: the run's OWN pinned ``config/move_hazard.json``
  (``move_hazard_v1``, ``source: corpus_fit``, ``method: empirical``) — the
  bucketed per-(axis, held class, dwell bucket) release table
  ``qnn.human.move_hazard`` tabulates from the training corpus. Adopt, never
  recompute (the look-grid precedent). This is a property of the HUMAN CORPUS,
  not of any model, so it is arm-neutral by construction, and the C1 controls
  already carry the identical file (same corpus, 20 Hz, Fibonacci edges).
  The a24-era log-normal (mu, sigma) equation form is NOT revived: the run
  dirs pin the empirical table, and the equation existed for the float32 ONNX
  graph, which a bench arm never enters.
* NOT revived, deliberately: the switch-back watermark
  (``move.switchback_eps``), stop-onset suppression (``move.stop_onset``),
  ``tau_engagement_gated``, and the ``jump_sample`` bias/temperature. Every
  one of those is a FITTED a24 rc knob, and the plan calls for a neutral
  bench decode. ``move.threat_break_hazard`` and the rest of the a25
  commitment knobs belong to the other arm and are never read here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import torch

from qnn.actions import MOVE_AXIS_CLASSES, MOVE_CLASS_NONE
from qnn.model.decode import inverse_cdf_sample, row_uniforms

# The C engine's load-time sticky default (see module docstring).
DEFAULT_STICKY_TAU: float = 0.6
# fb, lr — the ud axis is a per-tick impulse, never sticky (a24: jump is a
# 1-tick operative impulse, op-masked dwell median 1 frame).
N_STICKY_AXES = 2
HAZARD_SCHEMA = "move_hazard_v1"
# Number of uniforms this decode consumes per row per tick, in a FIXED
# schedule: [fb hazard, fb alt-class, lr hazard, lr alt-class, ud sample].
_N_UNIFORMS = 5


class MoveTickDecodeParams(NamedTuple):
    """Load-time decode params. ``hazard`` is (2, 3, n_buckets) release
    probabilities indexed [axis][held class][dwell bucket]; ``edges`` is the
    (n_buckets - 1,) dwell-age bucket edge vector the table was tabulated
    with (bucket = #{e : dwell_age > e}), so the table carries its own tick
    rate and the decode never assumes one."""

    sticky_tau: torch.Tensor    # (2,) float32
    edges: torch.Tensor         # (n_buckets-1,) int64
    hazard: torch.Tensor        # (2, 3, n_buckets) float32


def params_from_hazard_table(
    table: dict,
    *,
    tau: "float | tuple[float, float]" = DEFAULT_STICKY_TAU,
    device: "torch.device | str" = "cpu",
) -> MoveTickDecodeParams:
    """Build params from a ``move_hazard_v1`` document (the run dir's pinned
    ``config/move_hazard.json``). Fails loud on a foreign schema or method —
    the log-normal variant is not supported here (see module docstring)."""
    schema = str(table.get("schema", ""))
    if schema != HAZARD_SCHEMA:
        raise ValueError(
            f"move_tick decode: expected schema {HAZARD_SCHEMA!r}, got {schema!r}")
    method = str(table.get("method", ""))
    if method != "empirical":
        raise ValueError(
            f"move_tick decode: only the empirical bucketed table is revived, "
            f"got method {method!r} (the a24 log-normal form is not supported)")
    edges = torch.as_tensor(table["edges"], dtype=torch.int64, device=device)
    fb = torch.as_tensor(table["fb"], dtype=torch.float32, device=device)
    lr = torch.as_tensor(table["lr"], dtype=torch.float32, device=device)
    hazard = torch.stack([fb, lr], dim=0)               # (2, 3, n_buckets)
    if hazard.shape[1:] != (MOVE_AXIS_CLASSES, int(edges.numel()) + 1):
        raise ValueError(
            f"move_tick decode: hazard table shape {tuple(hazard.shape)} does "
            f"not match {MOVE_AXIS_CLASSES} held classes x "
            f"{int(edges.numel()) + 1} buckets")
    taus = (float(tau), float(tau)) if isinstance(tau, (int, float)) else (
        float(tau[0]), float(tau[1]))
    return MoveTickDecodeParams(
        sticky_tau=torch.tensor(taus, dtype=torch.float32, device=device),
        edges=edges, hazard=hazard)


def params_from_run_dir(run_dir: "str | Path", **kwargs) -> MoveTickDecodeParams:
    """Adopt the run's own pinned hazard table (``config/move_hazard.json``).

    No implicit default: a run without the pin cannot decode this arm."""
    path = Path(run_dir) / "config" / "move_hazard.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is absent — the per-tick move decode adopts the run's "
            "pinned corpus hazard table (run.init writes it; never recompute).")
    return params_from_hazard_table(json.loads(path.read_text()), **kwargs)


def hazard_release_prob(
    params: MoveTickDecodeParams,
    axis: int,
    held: torch.Tensor,      # (B,) long, held class 0..2
    dwell: torch.Tensor,     # (B,) long, dwell age >= 1
) -> torch.Tensor:
    """P(release next tick | axis, held class, dwell age) — the semi-Markov
    dwell hazard. ``bucket = #{e : dwell_age > e}`` (qnn.human.move_hazard)."""
    bucket = (dwell.unsqueeze(-1) > params.edges.unsqueeze(0)).sum(dim=-1)
    bucket = bucket.clamp(0, params.hazard.shape[-1] - 1)
    return params.hazard[axis][held.clamp(0, MOVE_AXIS_CLASSES - 1), bucket]


def move_tick_step(
    move_logits: torch.Tensor,       # (B, 3, 3) per-tick axis logits
    state: torch.Tensor,             # (B, >=4) float32 — MUTATED IN PLACE
    params: MoveTickDecodeParams,
    *,
    greedy: bool = False,
    row_generators: Any | None = None,
) -> torch.Tensor:
    """One tick of the sticky-tau + hazard MOVE decode → (B, 3) int64 classes.

    State lanes (the ``move_state`` carrier the a25 commitment decode also
    rides, so eval/PPO plumbing is unchanged — only the column semantics
    differ):

        [0] = fb class (< 0 ⇒ unset / episode start), [1] = fb dwell age
        [2] = lr class, [3] = lr dwell age

    The commitment decode's episode-reset lanes (``cls = -1, rem = 0``) are
    read correctly here: an unset lane starts held = NONE at dwell 1, exactly
    the a24 ``move_decode_reset`` init (prev_move = {1,1,1}, dwell = {1,1}).

    The law, per fb/lr axis (recovered from ``move_decode_step``):

      0. ``best = argmax(row)``, ``conf = softmax(row)[best]``.
      1. STICKY GATE — ``conf < tau`` ⇒ hold the previously emitted class.
         This, not the head, is what produces the human hold autocorrelation.
      2. HAZARD RELEASE — only when the gate is holding: with probability
         ``h(axis, held, dwell)`` re-draw from the OTHER two classes in
         proportion to their softmax mass. Stochastic ⇒ inert under
         ``greedy`` (the parity / gate path), matching the a25 convention.
      3. dwell += 1 on a hold, resets to 1 on a switch.

    The ud axis is NOT sticky: it is a per-tick impulse, sampled from its own
    row (or argmax under greedy). Whether the caller USES it is the act path's
    business — on land the jump head owns vertical.
    """
    if move_logits.dim() != 3 or move_logits.shape[-2:] != (3, MOVE_AXIS_CLASSES):
        raise ValueError(
            f"move_tick decode expects (B, 3, {MOVE_AXIS_CLASSES}) logits, "
            f"got {tuple(move_logits.shape)}")
    if state.shape[-1] < 2 * N_STICKY_AXES:
        raise ValueError(
            f"move_tick decode needs at least {2 * N_STICKY_AXES} state lanes "
            f"[fb_cls, fb_dwell, lr_cls, lr_dwell]; got {state.shape[-1]}")
    B = int(move_logits.shape[0])
    dev = move_logits.device
    logits = move_logits.float()
    probs = torch.softmax(logits, dim=-1)                       # (B, 3, 3)

    if greedy:
        u = torch.zeros(B, _N_UNIFORMS, device=dev)
    elif row_generators is None:
        u = torch.rand(B, _N_UNIFORMS, device=dev)
    else:
        u = row_uniforms(row_generators, _N_UNIFORMS, dev)

    out = torch.empty(B, 3, dtype=torch.long, device=dev)
    for ai in range(N_STICKY_AXES):
        row_p = probs[:, ai, :]                                 # (B, 3)
        best = row_p.argmax(dim=-1)
        conf = row_p.gather(1, best.unsqueeze(1)).squeeze(1)
        cls = state[:, ai * 2].long()
        held = torch.where(cls < 0, torch.full_like(cls, MOVE_CLASS_NONE), cls)
        dwell = state[:, ai * 2 + 1].long().clamp(min=1)

        # step 1 — sticky gate (hold-unless-confident).
        emit = torch.where(conf < float(params.sticky_tau[ai]), held, best)

        # step 2 — hazard release, only while the gate is holding.
        if not greedy:
            holding = emit == held
            h = hazard_release_prob(params, ai, held, dwell)
            fire = holding & (u[:, ai * 2] < h)
            alt_p = row_p.scatter(1, held.unsqueeze(1), torch.zeros_like(conf).unsqueeze(1))
            tot = alt_p.sum(dim=-1, keepdim=True)
            # A degenerate row (all mass on the held class) simply cannot
            # release — the a24 `tot > 1e-9` guard, vectorized.
            alt = inverse_cdf_sample(alt_p / tot.clamp_min(1e-9), u[:, ai * 2 + 1])
            emit = torch.where(fire & (tot.squeeze(1) > 1e-9), alt, emit)

        # step 3 — dwell + prev update.
        switched = emit != held
        state[:, ai * 2] = emit.to(state.dtype)
        state[:, ai * 2 + 1] = torch.where(
            switched, torch.ones_like(dwell), dwell + 1).to(state.dtype)
        out[:, ai] = emit

    # ud — per-tick impulse, no stickiness, no hazard.
    ud_p = probs[:, 2, :]
    out[:, 2] = ud_p.argmax(dim=-1) if greedy else inverse_cdf_sample(ud_p, u[:, 4])
    return out
