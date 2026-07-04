"""a24-lineage decode; polar-look / sticky-weapon / aim-prior are this gen's
choices, not invariants — a future gen replaces this module.

This is the SINGLE SOURCE OF TRUTH for the gen-specific orchestration that the
offline Python policy (:meth:`qnn.model.policy.QNNPolicy.act`) and the ONNX
export wrapper (:class:`tools.export_onnx.ExportWrapper`) must agree on
bit-for-bit, since the offline eval has to predict what the deployed ONNX graph
does. Keeping these shared (one definition, both import) is the dedup goal.

What lives here — the a24 generation's decode choices:
  * :func:`decode_look_from_polar` — the hybrid look decode (sampled magnitude
    × continuous circular-mean direction) + aim-prior blend + tangent expmap.
    This is the 2026-06-09 spin fix that replaced the v24 binned look decode.
  * :func:`assemble_aim_prior` — assemble the PRE-SCALED aim-prior blend term
    ``z_prior = gain·z_err + ffwd·z_rate`` from the lead/aim primitive. The
    primitive (``aim_prior_tangent_ffwd``) stays in qnn.model.bench.a24.lead_aim; only the
    gain×term assembly is here.
  * :func:`decide_weapon_sticky` — the sticky-weapon gate (top-2 softmax → hold
    vs switch), with the optional per-model weapon ban.
  * :func:`move_decode_step` / :func:`move_decode_step_graph` — the MOVE-ONLY
    stateful decode (sticky gate / watermark / hazard / stop-onset), threading
    ``move_state`` + ``move_state_rng``. MOVE-ONLY: attack is NOT here.
  * :func:`attack_decode_step` / :func:`attack_decode_step_graph` — the SEPARATE
    ATTACK decode (sigmoid(attack_logit+bias)>0.5 + continuous hold-tail), threading
    its OWN ``attack_state``. move + attack are decoupled (wire.11): all four
    actions decode in-graph here, emitting DECIDED outputs; the engine just reads
    them. They couple only at the collect-time action packing, never in the decode.

What does NOT live here, because the two paths legitimately differ:
  * MAGNITUDE SAMPLING. The policy draws a seeded categorical (row generators,
    temperature); the export draws an in-graph Gumbel-argmax (trace-safe). Each
    caller runs its own sampler (see qnn.model.decode) and passes the resulting
    ``mag_bin`` index in.
  * AIM-PRIOR GAINS. The blend math is shared; the gain SOURCES differ (policy:
    instance overrides w/ AIM_PRIOR_GAIN/AIM_FFWD_GAIN defaults; export:
    config/buffers). Each caller passes its own gains to :func:`assemble_aim_prior`.

TRACE-SAFETY: these functions are called from ``ExportWrapper.forward`` which is
``torch.onnx`` traced. They contain NO ``.item()``, NO Python ``if`` on tensor
values, NO data-dependent control flow, and NO advanced/boolean tensor indexing
(``index_select`` is used deliberately). Anything that traces today still traces.
Keep this module dependency-light: torch + ``look_bins`` + ``lead_aim`` only — do
NOT import policy.py or export_onnx.py (import-cycle hazard).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from qnn.model.look_bins import tangent_expmap

# Re-exported so this module is the generation's complete decode FACADE: the core
# (QNNPolicy.act) resolves one decode_module from the run's decode config and
# reads everything it decodes with through it — move geometry, weapon sticky, AND
# the aim-prior (gains + ffwd primitive). The aim primitives' home stays
# qnn.model.bench.a24.lead_aim; this is the facade seam, not a move.
from qnn.model.bench.a24.lead_aim import (  # noqa: F401
    AIM_FFWD_GAIN, AIM_PRIOR_GAIN, _TICK_DT_MODULE, aim_prior_tangent_ffwd,
)

# ── a24 stateful MOVE decode — C-engine parity (STEP 1) ──────────────────────
# Magic constants, mirrored from src/engine/common/qnn_onnx.c (cited):
QNN_STOP_ONSET_MIN_AGE = 6      # qnn_onnx.c:918  — min mature-stop dwell age
QNN_STOP_ONSET_MAX_AGE = 89     # mature-stop upper dwell age — formerly the last
                                # hazard bucket edge; now a standalone constant
                                # since the equation-direct hazard carries no edges.
QNN_ATTACK_HOLD_TICKS = 5         # qnn_onnx.c:881  — continuous-fire tail length
QNN_MOVE_HAZARD_BUCKETS = 11    # qnn_onnx.c:331  — engine bucket count (empirical-fit path only)


# ── Log-normal dwell-hazard — EQUATION-DIRECT release probability ─────────────
# h(t) = 1 - S(t)/S(t-1), the discrete-time hazard of a log-normal dwell lifetime,
# evaluated per integer dwell-age t (clamped to [1, QNN_HAZARD_MAXAGE]). NO bucketed
# table: the (mu, sigma) equation IS the wire/decode contract — both the eager
# reference and the traced/export graph evaluate it directly, so nothing can drift
# from a stored table. S(t)=0.5·erfc((ln t - mu)/(sigma·√2)).
#
# FLOAT32 / EXPORT NOTE: ONNX has only `Erf` (no `Erfc`), so the export graph must
# compute the tail as 0.5·(1-erf(z)) in float32, which UNDERFLOWS to 0 for z≳3.9
# (age ≳ 280 on the tightest cell) → a 0/0 → h=1 pathology. QNN_HAZARD_MAXAGE clamps
# the dwell-age into the float32-reliable regime; above it the hazard floors at
# h(MAXAGE) (a small POSITIVE rate → can't-freeze). The eager reference computes via
# the SAME float32 torch path so eager == graph == engine byte-for-byte. The float64
# "true" equation lives in qnn.model.move_hazard (fitting/analysis only).
_SQRT2 = math.sqrt(2.0)
QNN_HAZARD_MAXAGE = 128         # float32-reliable dwell-age clamp (~6.4 s @ 20 Hz);
                                # tail floors here (h≈0.016 > 0, can't-freeze).


def _lognorm_hazard_torch(dwell: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                          maxage: int = QNN_HAZARD_MAXAGE) -> torch.Tensor:
    """Vectorised per-age log-normal hazard for the traced/export graph. ``dwell``,
    ``mu``, ``sigma`` are broadcastable (B,) tensors. Uses ``torch.erf`` (→ ONNX
    ``Erf``); erfc(z)=1-erf(z). t clamped to [1, maxage]. THE canonical evaluation —
    the eager reference delegates here so float32 arithmetic matches exactly."""
    t = dwell.clamp(1, int(maxage)).to(torch.float32)
    sig = sigma.clamp_min(1e-6)

    def _surv(a: torch.Tensor) -> torch.Tensor:
        z = (torch.log(a.clamp_min(1e-12)) - mu) / (sig * _SQRT2)
        s = 0.5 * (1.0 - torch.erf(z))                       # 0.5·erfc(z)
        return torch.where(a > 0, s, torch.ones_like(s))

    h = 1.0 - _surv(t) / _surv(t - 1.0).clamp_min(1e-12)
    return h.clamp(1e-9, 1.0 - 1e-9)


def _lognorm_hazard_scalar(age: int, mu: float, sigma: float,
                           maxage: int = QNN_HAZARD_MAXAGE) -> float:
    """Scalar per-age log-normal hazard for the eager reference decode. Delegates to
    ``_lognorm_hazard_torch`` so the eager path uses the IDENTICAL float32 arithmetic
    as the graph/engine — no float64-vs-float32 tail divergence to break parity."""
    return float(_lognorm_hazard_torch(
        torch.tensor([age], dtype=torch.int64),
        torch.tensor([float(mu)], dtype=torch.float32),
        torch.tensor([float(sigma)], dtype=torch.float32),
        maxage).item())
# Continuous-weapon ids (ENTITY_IDS-encoded) that arm the fire hold-tail; the
# committed oracle slice reads ctx->w7_self_weapon_id (qnn_onnx.c at slice time)
# — see src/docs/move-decode-c-oracle.md §4. These mirror QNN_SUBJECT_* in
# qnn_vocab.h: NAILGUN=6, SUPER_NAILGUN=7, THUNDERBOLT=10.
QNN_CONTINUOUS_WEAPON_IDS = (6, 7, 10)
_RNG_DEFAULT_SEED = 0x9E3779B9   # qnn_onnx_rng_uniform reseed when state==0
_RNG_U32 = 0xFFFFFFFF


class MoveDecodeState(NamedTuple):
    """The cross-frame MOVE decode state, threaded in and out of every tick.

    Layout mirrors the fields qnn_onnx_decode_core carries on qnn_onnx_ctx
    (qnn_onnx.c:180-362) that the sticky fb/lr decode reads/writes:

      prev_move        int64 (3,)  per-axis previously-emitted class {0:neg,1:none,2:pos}.
                                   axis 2 (jump) is carried but never used by the sticky path.
      dwell_age        int64 (2,)  ticks the current fb/lr class has been held (semi-Markov age).
      swb_banned       int64 (2,)  per-axis class banned by the switch-back watermark; -1=inactive.
      swb_w            float32 (2,) the banned class's softmax prob at the switch tick (watermark level).
      rng_state        int64 ()    xorshift32 state for hazard draws (uint32 held in int64; persists across reset).

    MOVE-ONLY: the continuous-weapon attack hold-tail lives in the SEPARATE attack
    decode state, never here — move + attack are decoupled in the decode (they
    share a representation only in the collect-time action packing).

    All tensors are 0-dim / 1-dim CPU tensors; integers are int64 (uint32 rng
    masked into the low 32 bits), the watermark is float32 to match the C float.
    """

    prev_move: torch.Tensor      # int64 (3,)
    dwell_age: torch.Tensor      # int64 (2,)
    swb_banned: torch.Tensor     # int64 (2,)
    swb_w: torch.Tensor          # float32 (2,)
    rng_state: torch.Tensor      # int64 ()  (uint32 value)


class MoveDecodeParams(NamedTuple):
    """Load-time (model-stamped decode.*) params, read in qnn_onnx_select_codec
    (qnn_onnx.c:1681-1742); defaults from src/docs/move-decode-c-oracle.md §1.

      sticky_tau       float32 (2,)  fb/lr sticky gate thresholds (default 0.6).
      swb_eps          float         switch-back watermark eps; >0 enables (default 0.0).
      stop_onset       bool          stop-onset suppression; only active when hazard present.
      hazard_present   bool          hazard tables all parsed.
      hazard_lognorm   float32 (2,3,2)  [axis][held_class][mu,sigma] dwell-hazard equation params.
      hazard_maxage    int           dwell-age clamp for the equation tail (can't-freeze floor).
      hazard           float32 (2,3,11)  [axis][held_class][bucket] release probability.
    """

    sticky_tau: torch.Tensor
    swb_eps: float
    stop_onset: bool
    hazard_present: bool
    hazard_lognorm: torch.Tensor   # (2,3,2) [axis][held_class][mu,sigma]
    hazard_maxage: int
    # When True, the sticky-confidence gate's tau is forced to 1.0 (full hold —
    # switching comes only from the hazard equation) on DISENGAGED frames (no target),
    # and uses sticky_tau on ENGAGED frames. Lets a non-combat baseline hazard
    # table carry roaming locomotion while the model's confident argmax flips drive
    # the combat residual. Opt-in; default False reproduces the ungated gate exactly.
    tau_engagement_gated: bool = False
    # JUMP (ud axis) decode mode. Default False = legacy stateless argmax of the
    # (gumbel-perturbed) ud row. True = attack-style per-tick Bernoulli sample on
    # sigmoid((pos−none + jump_bias)/jump_temp) off the move decode's threaded rng:
    # jump is a 1-tick operative IMPULSE (op-masked dwell median 1f, 80% single-tick),
    # not a dwell, so it gets no sticky/hazard/hold — just sampled like attack, minus
    # the continuous hold-tail. The engine gates feasibility (drops +jump airborne).
    jump_sample: bool = False
    jump_bias: float = 0.0
    jump_temp: float = 1.0


def move_decode_params(
    tau_fb: float = 0.6,
    tau_lr: float = 0.6,
    swb_eps: float = 0.0,
    stop_onset: bool = False,
    hazard_lognorm: "list | torch.Tensor | None" = None,
    hazard_maxage: int = QNN_HAZARD_MAXAGE,
    tau_engagement_gated: bool = False,
    jump_sample: bool = False,
    jump_bias: float = 0.0,
    jump_temp: float = 1.0,
) -> MoveDecodeParams:
    """Build :class:`MoveDecodeParams`, mirroring the load-time defaults and the
    ``stop_onset && hazard_present`` gate (qnn_onnx.c:1738 / harness line 274).

    ``hazard_lognorm`` is the (2,3,2) [axis][held_class][mu,sigma] dwell-hazard
    equation — the source of truth; the release probability is evaluated per
    integer dwell-age via :func:`_lognorm_hazard_scalar` / ``_lognorm_hazard_torch``
    (no bucketed table). When absent the hazard path is disabled and (per the C)
    stop-onset with it, so the decode is the plain sticky-gate + watermark machine.
    """
    hazard_present = hazard_lognorm is not None
    if hazard_present:
        ln_t = torch.as_tensor(hazard_lognorm, dtype=torch.float32)
    else:
        ln_t = torch.zeros(2, 3, 2, dtype=torch.float32)
    return MoveDecodeParams(
        sticky_tau=torch.tensor([tau_fb, tau_lr], dtype=torch.float32),
        swb_eps=float(swb_eps),
        # stop-onset requires the hazard equation (qnn_onnx.c:1738; harness:274).
        stop_onset=bool(stop_onset) and hazard_present,
        hazard_present=hazard_present,
        hazard_lognorm=ln_t,
        hazard_maxage=int(hazard_maxage),
        tau_engagement_gated=bool(tau_engagement_gated),
        jump_sample=bool(jump_sample),
        jump_bias=float(jump_bias),
        jump_temp=float(jump_temp),
    )


def move_decode_reset(params: MoveDecodeParams | None = None,
                      rng_state: "int | torch.Tensor" = _RNG_DEFAULT_SEED) -> MoveDecodeState:
    """Mirror oracle_reset / QNN_OnnxReset's move-state init (qnn_onnx.c:433-445).

    prev_move={1,1,1} (none), dwell_age={1,1}, swb_banned={-1,-1}, swb_w={0,0}.
    The RNG state DELIBERATELY PERSISTS across reset (qnn_onnx.c:441 comment) — the
    caller seeds it once at load and passes it through here; pass the live seed in
    to preserve a stream across episode resets, or the default golden ratio
    constant for a fresh stream.
    """
    return MoveDecodeState(
        prev_move=torch.tensor([1, 1, 1], dtype=torch.int64),
        dwell_age=torch.tensor([1, 1], dtype=torch.int64),
        swb_banned=torch.tensor([-1, -1], dtype=torch.int64),
        swb_w=torch.zeros(2, dtype=torch.float32),
        rng_state=torch.as_tensor(int(rng_state) & _RNG_U32, dtype=torch.int64),
    )


def _rng_uniform(state: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """xorshift32 → uniform [0,1), bit-for-bit with qnn_onnx_rng_uniform
    (qnn_onnx.c:865-874). uint32 arithmetic emulated in int64 masked to 32 bits.

    Returns ``(u, new_state)`` where ``u`` is a float32 0-dim tensor and
    ``new_state`` the advanced int64 (uint32) state.

    EXPORT-RISK (step 2): the shift/xor chain needs ONNX BitShift+Xor on int
    tensors and the (x>>8)*2^-24 float conversion; data-flow is fine but the
    int32-wraparound semantics must be reproduced exactly in-graph.
    """
    x = state.to(torch.int64) & _RNG_U32
    # if x == 0 -> reseed (matches the C guard); torch.where keeps it branchless.
    x = torch.where(x == 0, torch.tensor(_RNG_DEFAULT_SEED, dtype=torch.int64), x)
    x = (x ^ ((x << 13) & _RNG_U32)) & _RNG_U32
    x = (x ^ (x >> 17)) & _RNG_U32
    x = (x ^ ((x << 5) & _RNG_U32)) & _RNG_U32
    u = (x >> 8).to(torch.float32) * (1.0 / 16777216.0)
    return u, x


def move_decode_step(
    move_logits: torch.Tensor,
    present_move: bool,
    look: torch.Tensor | None,
    move_state: MoveDecodeState,
    params: MoveDecodeParams,
    present_look: bool = False,
    hazard_release_prob: "torch.Tensor | None" = None,
    projectile_release: bool = False,
    engaged: bool = True,
) -> "tuple[int, list[int], torch.Tensor | None, MoveDecodeState]":
    """One tick of the a24 stateful MOVE decode — bit-for-bit with the move half
    of qnn_onnx_decode_core (qnn_onnx.c:887-1059). PURE: threads decode state in
    and out, mutates nothing.

    MOVE-ONLY: attack is decoded SEPARATELY (:func:`attack_decode_step`), never
    coupled here — the only place move + attack share a representation is the
    collect-time action packing for the on-disk cache, NOT the decode. See
    src/qnn/model/bench/a24/decode.py module docstring + the move/attack-decouple
    invariant.

    Args:
      move_logits: float (9,) — 3 axes × 3 classes (neg/none/pos), row-major
                   (fb=0..2, lr=3..5, up=6..8). Raw/un-sampled for fb/lr; the
                   jump row (up) is the graph's gumbel-perturbed sample.
      present_move: head-presence (graph-build-time).
      look:        float (3,) look vector, clamped to [-1,1] when present_look.
      move_state:  the carried :class:`MoveDecodeState`.
      params:      the load-time :class:`MoveDecodeParams`.

    Returns ``(move_byte, [fb_class, lr_class, jump_class], look_out,
    new_move_state)``. ``move_byte`` is the MOVE press bits (QNN_PackInputMask
    layout, qnn.h:479) with bit0 (attack) ZERO — the engine ORs the decided attack
    bit into the press byte at the very end (Quake usercmd button mask); classes
    are 0/1/2.
    """
    logits = move_logits.to(torch.float32).reshape(9)

    prev_move = move_state.prev_move.clone()
    dwell_age = move_state.dwell_age.clone()
    swb_banned = move_state.swb_banned.clone()
    swb_w = move_state.swb_w.clone()
    rng_state = move_state.rng_state.clone()

    swb_eps = params.swb_eps
    tau = params.sticky_tau

    axis_signs = [0, 0, 0]

    if present_move:
        # Pre-loop stop-onset snapshot (qnn_onnx.c:925-931): read BOTH held
        # classes before the loop mutates prev_move (fb updates before lr).
        stop_age = int(min(int(dwell_age[0]), int(dwell_age[1])))
        stopped = (
            params.stop_onset
            and int(prev_move[0]) == 1
            and int(prev_move[1]) == 1
            and stop_age >= QNN_STOP_ONSET_MIN_AGE
            and stop_age <= QNN_STOP_ONSET_MAX_AGE
        )

        for axis in range(3):
            row = logits[axis * 3: axis * 3 + 3]
            # step 0: argmax + softmax conf of the argmax class.
            best = int(torch.argmax(row).item())  # ties → first max (matches C scan)
            # C scans c=1,2 with strict '>' starting best=0 — argmax matches.
            best_v = row[best]
            if axis < 2:
                sm = torch.exp(row - best_v).sum()
                conf = (1.0 / sm).to(torch.float32)
                held = int(prev_move[axis])
                if projectile_release:
                    # Dodge frame: model has full control — raw argmax, no sticky /
                    # watermark / stop-onset / hazard / watermark-arm. No RNG draw.
                    pass
                else:
                    # step 1: sticky gate (hold-unless-confident). Engagement-gated
                    # regimes force tau=1 (full hold) on disengaged frames, so a
                    # non-combat baseline hazard table alone drives roaming switches.
                    tau_axis = float(tau[axis])
                    if params.tau_engagement_gated and not engaged:
                        tau_axis = 1.0
                    if float(conf) < tau_axis:
                        best = held
                    # step 2: switch-back watermark (clear OR suppress).
                    if swb_eps > 0.0 and int(swb_banned[axis]) >= 0:
                        banned = int(swb_banned[axis])
                        pb = (torch.exp(row[banned] - best_v) / sm).to(torch.float32)
                        if float(pb) < float(swb_w[axis]) - swb_eps:
                            swb_banned[axis] = -1
                        elif best == banned and held != banned:
                            best = held
                    # step 3: stop-onset suppression.
                    if stopped and best != held:
                        best = held
                    # step 4: hazard release (only when the gate is holding). The
                    # release probability is either the LEARNED WHEN-head's per-axis
                    # P(release | held, dwell, context) — head-driven a25, fed the
                    # incoming (held=prev_move, dwell_age) + motor feature so it sees
                    # context a table can't — or the log-normal dwell-hazard EQUATION
                    # evaluated per dwell-age. The RNG draw schedule is identical either way.
                    head_driven = hazard_release_prob is not None
                    if (params.hazard_present or head_driven) and best == held:
                        if head_driven:
                            release_p = float(hazard_release_prob[axis])
                        elif params.hazard_present:
                            mu, sigma = params.hazard_lognorm[axis][held].tolist()
                            release_p = _lognorm_hazard_scalar(
                                int(dwell_age[axis]), float(mu), float(sigma),
                                params.hazard_maxage)
                        else:
                            release_p = 0.0
                        u, rng_state = _rng_uniform(rng_state)
                        if float(u) < release_p:
                            p = [0.0, 0.0, 0.0]
                            tot = 0.0
                            for c in range(3):
                                pc = 0.0 if c == held else float(torch.exp(row[c] - best_v))
                                p[c] = pc
                                tot += pc
                            if tot > 1e-9:
                                u2, rng_state = _rng_uniform(rng_state)
                                draw = float(u2) * tot
                                best = 1 if held == 0 else 0
                                for c in range(3):
                                    if c == held:
                                        continue
                                    if draw < p[c]:
                                        best = c
                                        break
                                    draw -= p[c]
                    # step 5: arm watermark on switch (gate- or hazard-driven).
                    if swb_eps > 0.0 and best != held:
                        swb_banned[axis] = held
                        swb_w[axis] = (torch.exp(row[held] - best_v) / sm).to(torch.float32)
                # step 6: update dwell + prev.
                dwell_age[axis] = dwell_age[axis] + 1 if best == held else 1
                prev_move[axis] = best
            axis_signs[axis] = best - 1   # class 0,1,2 → -1,0,+1

    fb_sign, lr_sign, up_sign = axis_signs
    fb_neg, fb_pos = int(fb_sign < 0), int(fb_sign > 0)
    lr_neg, lr_pos = int(lr_sign < 0), int(lr_sign > 0)
    up_neg, up_pos = int(up_sign < 0), int(up_sign > 0)

    # pack the MOVE press bits — QNN_PackInputMask (qnn_collect_helpers.c:556,
    # qnn.h:479): bit0 attack (engine ORs the decided attack bit in at the end),
    # b1/2 fb neg/pos, b3/4 lr neg/pos, b5/6 up neg/pos, b7 jump. bit0 stays 0 here.
    move_byte = 0
    if fb_neg:
        move_byte |= 0x02
    if fb_pos:
        move_byte |= 0x04
    if lr_neg:
        move_byte |= 0x08
    if lr_pos:
        move_byte |= 0x10
    if up_neg:
        move_byte |= 0x20
    if up_pos:
        move_byte |= 0x40
    if up_pos:                 # jump_act = up_pos (qnn_onnx.c:1049)
        move_byte |= 0x80

    fb_class, lr_class, jump_class = fb_sign + 1, lr_sign + 1, up_sign + 1

    look_out = None
    if present_look and look is not None:
        look_out = torch.clamp(look.to(torch.float32).reshape(3), -1.0, 1.0)

    new_state = MoveDecodeState(
        prev_move=prev_move,
        dwell_age=dwell_age,
        swb_banned=swb_banned,
        swb_w=swb_w,
        rng_state=rng_state.to(torch.int64),
    )
    return move_byte, [fb_class, lr_class, jump_class], look_out, new_state


# ── a24 stateful MOVE decode — TRACE-SAFE in-graph form (STEP 2) ─────────────
# The eager move_decode_step above is the reference (Python control flow, used
# offline + as the parity oracle). The functions below re-express the SAME
# machine with pure tensor ops (torch.where over recurrent state tensors, a
# fixed-schedule in-graph xorshift32) so the whole decode bakes into the ONNX
# graph and runs identically under onnxruntime. Both are validated bit-for-bit
# against the golden fixture (tests/test_move_decode_parity.py).
#
# The state is carried as ONE flat float32 tensor `move_state` (shape (B, 9)), so
# it threads through the ONNX graph exactly like the GRU hidden/next_hidden pair
# (a single named tensor in, a single named tensor out). MOVE-ONLY layout:
#   [0:3]  prev_move   (fb, lr, jump)        int values held as float
#   [3:5]  dwell_age   (fb, lr)              int values held as float
#   [5:7]  swb_banned  (fb, lr)              int (-1..2) held as float
#   [7:9]  swb_w       (fb, lr)              float32
# The rng_state is a uint32 and does NOT survive a float32 round-trip, so it is
# threaded as its OWN int64 tensor `move_state_rng` (B,) alongside the float
# state. Two state tensors in, two out — both mirror hidden/next_hidden. (wire.11
# dropped the two dead trailing slots: the legacy rng-float [9] — rng is the
# separate int64 tensor — and attack_hold [10] — attack now carries its own
# attack_state. So the move state is exactly the move machine's working set.)
MOVE_STATE_DIM = 9
_MOVE_STATE_PREV = slice(0, 3)
_MOVE_STATE_DWELL = slice(3, 5)
_MOVE_STATE_SWBB = slice(5, 7)
_MOVE_STATE_SWBW = slice(7, 9)


def move_decode_reset_flat(
    batch: int = 1,
    rng_state: "int | torch.Tensor" = _RNG_DEFAULT_SEED,
    device: torch.device | str | None = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """The flat (graph-I/O) form of :func:`move_decode_reset`.

    Returns ``(move_state, move_state_rng)`` with shapes ``(batch, 9)`` float32
    and ``(batch,)`` int64. This is the EXACT initial state the C engine must
    memset/seed at episode start (step 3): prev_move={1,1,1}, dwell_age={1,1},
    swb_banned={-1,-1}, swb_w={0,0}, and the rng seeded once (it PERSISTS across
    episode reset in the C — qnn_onnx.c:441 — so the engine seeds move_state_rng
    once at load and carries it, never re-memsetting it).
    """
    s = torch.zeros(batch, MOVE_STATE_DIM, dtype=torch.float32, device=device)
    s[:, _MOVE_STATE_PREV] = 1.0       # prev_move = none
    s[:, _MOVE_STATE_DWELL] = 1.0      # dwell_age = 1
    s[:, _MOVE_STATE_SWBB] = -1.0      # swb_banned = inactive
    # swb_w already 0
    rng = torch.full((batch,), int(rng_state) & _RNG_U32, dtype=torch.int64, device=device)
    return s, rng


def _xor32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise XOR of two non-negative int64 tensors via the identity
    ``a XOR b = a + b - 2*(a AND b)`` (exact for non-negative ints, since
    ``a + b == (a XOR b) + 2*(a AND b)``).

    EXPORT NOTE: the legacy TorchScript ONNX exporter REJECTS ``BitwiseXor`` /
    ``BitwiseOr`` on non-boolean inputs (only ``BitwiseAnd`` has a wired
    symbolic). So XOR is composed from BitwiseAnd + Add/Sub/Mul, all of which
    export and run identically under onnxruntime. Verified bit-for-bit against
    the eager ``^`` on the full fixture rng stream.
    """
    return a + b - 2 * (a & b)


def _rng_uniform_graph(state: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Trace-safe xorshift32 → uniform [0,1), bit-for-bit with ``_rng_uniform``.

    ``state`` is an int64 tensor (any shape) holding a uint32 value. Uses torch
    shifts (<<, >>) → ONNX BitShift, BitwiseAnd → ONNX BitwiseAnd, and a
    composed XOR (see :func:`_xor32`) because the legacy exporter rejects
    integer BitwiseXor. All run identically under onnxruntime. The
    reseed-on-zero guard is a branchless ``torch.where``.

    Returns ``(u, new_state)`` (u float32, new_state int64), same shapes as
    ``state``.
    """
    x = state.to(torch.int64) & _RNG_U32
    seed = torch.full_like(x, _RNG_DEFAULT_SEED)
    x = torch.where(x == 0, seed, x)
    x = _xor32(x, (x << 13) & _RNG_U32) & _RNG_U32
    x = _xor32(x, x >> 17) & _RNG_U32
    x = _xor32(x, (x << 5) & _RNG_U32) & _RNG_U32
    u = (x >> 8).to(torch.float32) * (1.0 / 16777216.0)
    return u, x


def _decode_axis_graph(
    row: torch.Tensor,            # (B, 3) logits for this axis
    held: torch.Tensor,           # (B,) int64 prev class for this axis
    dwell: torch.Tensor,          # (B,) int64 dwell age for this axis
    swb_banned: torch.Tensor,     # (B,) int64
    swb_w: torch.Tensor,          # (B,) float32
    rng_state: torch.Tensor,      # (B,) int64
    stopped: torch.Tensor,        # (B,) bool — pre-loop stop snapshot
    tau: torch.Tensor,            # () float32 — this axis's sticky tau
    swb_eps: float,
    hazard_present: bool,
    hazard_lognorm_axis: torch.Tensor,  # (3, 2) float32 — this axis's [held][mu,sigma]
    hazard_maxage: int,                 # dwell-age clamp for the equation tail
    hazard_release_prob: torch.Tensor | None = None,  # (B,) learned WHEN-head P(release)
    force_release: torch.Tensor | None = None,  # (B,) bool — Gate B forced hold release
    engaged: torch.Tensor | None = None,  # (B,) bool — has a target this tick
    tau_engagement_gated: bool = False,
) -> "tuple[torch.Tensor, ...]":
    """One fb/lr axis, fully vectorised over the batch. Returns
    ``(best, new_dwell, new_swb_banned, new_swb_w, new_rng_state)`` — the
    threaded rng so fb's output feeds lr's input (fb-before-lr preserved)."""
    B = row.shape[0]
    best_v = row.max(dim=-1, keepdim=True).values            # (B,1)
    exp = torch.exp(row - best_v)                            # (B,3)
    sm = exp.sum(dim=-1)                                     # (B,)
    conf = 1.0 / sm                                          # softmax prob of argmax
    argmax = row.argmax(dim=-1).to(torch.int64)             # (B,) — ties→first (C scan)

    # Dodge mask: when set the model has full control this tick — raw argmax,
    # no sticky / watermark / stop-onset / hazard / watermark-arm / RNG draw.
    dodge = (force_release if force_release is not None
             else torch.zeros(B, dtype=torch.bool, device=row.device))

    # step 1: sticky gate — bypassed for dodge rows. Engagement-gated regimes
    # force tau=1 (full hold) on disengaged frames, leaving roaming switches to
    # the (non-combat baseline) hazard table; engaged frames use sticky_tau.
    if tau_engagement_gated and engaged is not None:
        tau_eff = torch.where(engaged, tau, torch.ones_like(conf))
    else:
        tau_eff = tau
    best = torch.where(conf < tau_eff, held, argmax)
    best = torch.where(dodge, argmax, best)

    # step 2: switch-back watermark (clear OR suppress) — bypassed for dodge rows.
    if swb_eps > 0.0:
        active = swb_banned >= 0
        banned = swb_banned.clamp(min=0)                    # safe gather index
        pb = exp.gather(-1, banned.unsqueeze(-1)).squeeze(-1) / sm
        do_clear = active & (pb < (swb_w - swb_eps)) & ~dodge
        do_suppress = active & (~do_clear) & (best == banned) & (held != banned) & ~dodge
        best = torch.where(do_suppress, held, best)
        swb_banned = torch.where(do_clear, torch.full_like(swb_banned, -1), swb_banned)

    # step 3: stop-onset suppression — bypassed for dodge rows.
    if hazard_present:
        best = torch.where(stopped & (best != held) & ~dodge, held, best)

    # step 4: hazard release (only when gate is holding: best == held, not dodging).
    new_rng = rng_state
    if hazard_present or (hazard_release_prob is not None):
        # Dodge rows are excluded from the hazard path — no RNG draw on those ticks.
        holding = (best == held) & ~dodge
        if hazard_release_prob is not None:
            rel_prob = hazard_release_prob                  # (B,)
        elif hazard_present:
            # equation-direct: per-age log-normal hazard, no buckets.
            musig = hazard_lognorm_axis.index_select(0, held.clamp(0, 2))  # (B, 2)
            rel_prob = _lognorm_hazard_torch(
                dwell, musig[:, 0], musig[:, 1], hazard_maxage)            # (B,)
        else:
            rel_prob = torch.zeros(B, dtype=row.dtype, device=row.device)
        u1, rng_after1 = _rng_uniform_graph(rng_state)
        released = holding & (u1 < rel_prob)
        # renormalized softmax over classes != held.
        held_oh = torch.nn.functional.one_hot(held.clamp(0, 2), 3).to(row.dtype)  # (B,3)
        p = exp * (1.0 - held_oh)                           # zero the held class
        tot = p.sum(dim=-1)                                 # (B,)
        has_mass = tot > 1e-9
        do_release = released & has_mass
        u2, rng_after2 = _rng_uniform_graph(rng_after1)
        draw = u2 * tot                                     # (B,)
        cdf = torch.cumsum(p, dim=-1)                       # (B,3)
        valid = (draw.unsqueeze(-1) < cdf) & (p > 0)        # (B,3)
        first_non_held = torch.where(held == 0,
                                     torch.ones_like(held),
                                     torch.zeros_like(held))
        idx = torch.arange(3, device=row.device).reshape(1, 3)
        big = torch.full_like(idx, 99)
        masked_idx = torch.where(valid, idx, big)
        sampled = masked_idx.min(dim=-1).values             # (B,)
        any_valid = valid.any(dim=-1)
        sampled = torch.where(any_valid, sampled, first_non_held).to(torch.int64)
        best = torch.where(do_release, sampled, best)
        # rng schedule: holding & release → after2; holding & ~release → after1;
        # not holding (incl. dodge) → unchanged (no draw).
        new_rng = torch.where(holding,
                              torch.where(do_release, rng_after2, rng_after1),
                              rng_state)

    # step 5: arm watermark on switch — not on dodge rows.
    if swb_eps > 0.0:
        switched = (best != held) & ~dodge
        new_w = exp.gather(-1, held.clamp(0, 2).unsqueeze(-1)).squeeze(-1) / sm
        swb_banned = torch.where(switched, held, swb_banned)
        swb_w = torch.where(switched, new_w, swb_w)

    # step 6: update dwell + prev.
    new_dwell = torch.where(best == held, dwell + 1, torch.ones_like(dwell))
    return best, new_dwell, swb_banned, swb_w, new_rng


def move_decode_step_graph(
    move_logits: torch.Tensor,        # (B, 9) or (B, 3, 3) float
    move_state: torch.Tensor,         # (B, 9) float32 — flat carried state
    move_state_rng: torch.Tensor,     # (B,) int64 — xorshift32 state
    params: MoveDecodeParams,
    present_move: bool = True,
    hazard_release_prob: torch.Tensor | None = None,  # (B,3) learned WHEN-head P(release)
    projectile_release: torch.Tensor | None = None,  # (B,) bool — Gate B forced fb/lr release
    engaged: torch.Tensor | None = None,  # (B,) bool — has a target this tick (tau gating)
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """TRACE-SAFE batched a24 MOVE decode — the in-graph twin of
    :func:`move_decode_step`, bit-for-bit with the golden fixture.

    MOVE-ONLY: attack is decoded SEPARATELY (its own in-graph decode + state);
    move never carries it. Returns ``(move_classes, move_state_out, move_state_rng_out)``:
      move_classes   (B, 3) int64 — decided fb/lr/jump class {0:neg,1:none,2:pos}.
                     The engine assembles the Quake press byte from these + the
                     decided attack bit (this function does NOT pack).
      move_state_out (B, 9) float32 — the updated flat state to thread back in.
      move_state_rng_out (B,) int64 — the advanced xorshift32 state.

    ``present_move`` is a PYTHON bool (graph-build-time head presence, baked as a
    taken/not-taken branch — like the C out_present check), NOT a tensor value.
    """
    logits = move_logits.to(torch.float32).reshape(-1, 3, 3)
    B = logits.shape[0]

    prev_move = move_state[:, _MOVE_STATE_PREV].round().to(torch.int64)   # (B,3)
    dwell_age = move_state[:, _MOVE_STATE_DWELL].round().to(torch.int64)  # (B,2)
    swb_banned = move_state[:, _MOVE_STATE_SWBB].round().to(torch.int64)  # (B,2)
    swb_w = move_state[:, _MOVE_STATE_SWBW]                               # (B,2)
    rng = move_state_rng.to(torch.int64)

    # Per-axis output columns assembled by torch.stack (NO in-place slice
    # assignment — index_put_ traces poorly under the legacy ONNX exporter).
    out_prev_cols = [prev_move[:, 0], prev_move[:, 1], prev_move[:, 2]]
    out_dwell_cols = [dwell_age[:, 0], dwell_age[:, 1]]
    out_swbb_cols = [swb_banned[:, 0], swb_banned[:, 1]]
    out_swbw_cols = [swb_w[:, 0], swb_w[:, 1]]

    classes = prev_move    # default carry (also the no-present case anchor)

    if present_move:
        # Pre-loop stop snapshot (read both held classes before the loop mutates).
        stop_age = torch.minimum(dwell_age[:, 0], dwell_age[:, 1])
        if params.stop_onset:
            stopped = (
                (prev_move[:, 0] == 1) & (prev_move[:, 1] == 1)
                & (stop_age >= QNN_STOP_ONSET_MIN_AGE)
                & (stop_age <= QNN_STOP_ONSET_MAX_AGE)
            )
        else:
            stopped = torch.zeros(B, dtype=torch.bool)

        axis_best = []
        for axis in range(2):       # fb then lr — rng threads fb→lr.
            best, nd, nb, nw, rng = _decode_axis_graph(
                logits[:, axis, :], prev_move[:, axis], dwell_age[:, axis],
                swb_banned[:, axis], swb_w[:, axis], rng, stopped,
                params.sticky_tau[axis], params.swb_eps,
                params.hazard_present, params.hazard_lognorm[axis], params.hazard_maxage,
                hazard_release_prob=(None if hazard_release_prob is None
                                     else hazard_release_prob[:, axis]),
                force_release=projectile_release,   # same (B,) for fb + lr (Gate B)
                engaged=engaged,                    # same (B,) for fb + lr (tau gating)
                tau_engagement_gated=params.tau_engagement_gated,
            )
            axis_best.append(best)
            out_dwell_cols[axis] = nd
            out_swbb_cols[axis] = nb
            out_swbw_cols[axis] = nw
            out_prev_cols[axis] = best
        # jump axis (2): an IMPULSE, not a dwell (op-masked +jump is 1 tick, 80%),
        # so NO sticky/hazard/hold — either legacy argmax or an attack-style per-tick
        # Bernoulli sample off the SAME threaded rng (continues fb→lr→jump). neg is
        # vestigial (crouch ~0), so sampling reduces to pos-vs-none:
        #   p_jump = sigmoid((pos − none + jump_bias) / jump_temp); jump = pos if u<p.
        if params.jump_sample:
            jrow = logits[:, 2, :]
            jl = (jrow[:, 2] - jrow[:, 1] + float(params.jump_bias)) / float(params.jump_temp)
            p_jump = torch.sigmoid(jl)
            uj, rng = _rng_uniform_graph(rng)
            _dev = logits.device
            jump = torch.where(uj < p_jump,
                               torch.full((B,), 2, dtype=torch.int64, device=_dev),
                               torch.ones(B, dtype=torch.int64, device=_dev))   # pos(2) else none(1)
        else:
            jump = logits[:, 2, :].argmax(dim=-1).to(torch.int64)
        out_prev_cols[2] = jump
        classes = torch.stack([axis_best[0], axis_best[1], jump], dim=-1)  # (B,3)

    # Reassemble the flat state by stacking the 9 columns (order = the layout
    # constants); float for everything, rng threaded separately as int64.
    cols = [
        out_prev_cols[0].to(torch.float32), out_prev_cols[1].to(torch.float32),
        out_prev_cols[2].to(torch.float32),
        out_dwell_cols[0].to(torch.float32), out_dwell_cols[1].to(torch.float32),
        out_swbb_cols[0].to(torch.float32), out_swbb_cols[1].to(torch.float32),
        out_swbw_cols[0], out_swbw_cols[1],
    ]
    move_state_out = torch.stack(cols, dim=-1)         # (B, 9)

    return classes, move_state_out, rng.to(torch.int64)


# ── a24 ATTACK decode — its OWN decode + state + rng, fully decoupled from move ─
# move + attack share a representation ONLY at the collect-time action packing;
# the decode keeps them separate. Attack is decoded SAMPLED (temperature-Bernoulli
# on sigmoid((attack_logit + attack_bias) / temp)) — sigmoid is the BCE-learned
# P(attack|state), human attack is stochastic, so a greedy threshold is robotic
# (skill-curves §5/§6). The draw uses attack's OWN xorshift rng (`attack_rng`,
# threaded like move_state_rng), and the continuous-weapon hold-tail rides the
# SEPARATE attack_state. The engine ORs the decided attack bit into the Quake
# press byte and runs no attack decode of its own (wire.11). attack_bias is the
# all-humans propensity calibration (~−1.1; offsets the head's ~1.4× over-fire);
# greedy↔sampled is now a pure decode-regime choice with NO wire change.
ATTACK_STATE_DIM = 1   # [0] attack_hold_ticks (continuous-fire hold-tail countdown)


class AttackDecodeState(NamedTuple):
    """Cross-frame ATTACK decode state (separate from MoveDecodeState). The rng is
    threaded separately as an int64 `attack_rng`, mirroring move_state_rng."""
    attack_hold_ticks: torch.Tensor  # int64 ()


def attack_decode_reset(
    rng_state: "int | torch.Tensor" = _RNG_DEFAULT_SEED,
) -> "tuple[AttackDecodeState, torch.Tensor]":
    """Eager attack-state init: hold-tail countdown = 0 + the seeded xorshift rng
    (which PERSISTS across episode reset like move_state_rng)."""
    return (AttackDecodeState(attack_hold_ticks=torch.zeros((), dtype=torch.int64)),
            torch.as_tensor(int(rng_state) & _RNG_U32, dtype=torch.int64))


def attack_decode_reset_flat(
    batch: int = 1,
    rng_state: "int | torch.Tensor" = _RNG_DEFAULT_SEED,
    device: "torch.device | str | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Flat (graph-I/O) attack-state: ``(B, ATTACK_STATE_DIM)`` float32 (all 0) +
    the ``(B,)`` int64 attack_rng (seeded, persists across reset)."""
    s = torch.zeros(batch, ATTACK_STATE_DIM, dtype=torch.float32, device=device)
    rng = torch.full((batch,), int(rng_state) & _RNG_U32, dtype=torch.int64, device=device)
    return s, rng


def attack_decode_step(
    attack_logit: "torch.Tensor | float",
    self_weapon_id: int,
    attack_state: AttackDecodeState,
    attack_rng: torch.Tensor,
    attack_bias: float = 0.0,
    temperature: float = 1.0,
    attack_threshold: "float | None" = None,
) -> "tuple[int, AttackDecodeState, torch.Tensor]":
    """One tick of the ATTACK decode (eager reference). PURE.

    ``p = sigmoid((attack_logit + attack_bias) / temperature)``. SAMPLED (default,
    ``attack_threshold=None``): a Bernoulli draw off attack's own xorshift rng
    (``u < p``). DETERMINISTIC (``attack_threshold=τ``): fire iff ``p ≥ τ`` — greedy
    fire that commits on confident ticks so it co-occurs with jump+aim (coordinated
    actions like the rocket-jump); τ is calibrated to the human fire frequency. The
    rng is advanced either way so the state loopback is identical. Plus the
    continuous-weapon hold-tail. Bit-for-bit with the graph twin given the same rng.
    """
    attack_hold_ticks = attack_state.attack_hold_ticks.clone()
    is_continuous = int(self_weapon_id) in QNN_CONTINUOUS_WEAPON_IDS
    fl = (float(attack_logit) + float(attack_bias)) / float(temperature)
    p_attack = 1.0 / (1.0 + float(torch.exp(torch.tensor(-fl, dtype=torch.float32))))
    u, new_rng = _rng_uniform(attack_rng)           # advance rng regardless (state contract)
    if attack_threshold is not None:
        attack_bit = 1 if p_attack >= float(attack_threshold) else 0   # deterministic
    else:
        attack_bit = 1 if float(u) < p_attack else 0                  # sampled
    if is_continuous and attack_bit:
        attack_hold_ticks = torch.tensor(QNN_ATTACK_HOLD_TICKS, dtype=torch.int64)
    elif is_continuous and int(attack_hold_ticks) > 0:
        attack_bit = 1
        attack_hold_ticks = attack_hold_ticks - 1
    else:
        attack_hold_ticks = torch.zeros((), dtype=torch.int64)
    return (attack_bit, AttackDecodeState(attack_hold_ticks=attack_hold_ticks.to(torch.int64)),
            new_rng.to(torch.int64))


def attack_decode_step_graph(
    attack_logit: torch.Tensor,         # (B,) or (B,1) float
    self_weapon_id: torch.Tensor,     # (B,) int — ENTITY_IDS-encoded held weapon
    attack_state: torch.Tensor,       # (B, ATTACK_STATE_DIM) float32 — flat state
    attack_rng: torch.Tensor,         # (B,) int64 — xorshift32 state
    attack_bias: float = 0.0,
    temperature: float = 1.0,
    attack_threshold: "float | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """TRACE-SAFE batched ATTACK decode — the in-graph twin of
    :func:`attack_decode_step`. Returns ``(attack_bit, attack_state_out, attack_rng_out)``:
      attack_bit       (B, 1) int64 — the DECIDED (sampled) fire/no-fire bit (engine
                       ORs it into the press byte; no engine-side decode).
      attack_state_out (B, ATTACK_STATE_DIM) float32 — updated hold-tail state.
      attack_rng_out   (B,) int64 — the advanced xorshift32 state.
    """
    fl = (attack_logit.to(torch.float32).reshape(-1) + float(attack_bias)) / float(temperature)
    B = fl.shape[0]
    attack_hold = attack_state[:, 0].round().to(torch.int64)               # (B,)
    p_attack = torch.sigmoid(fl)
    u, rng_out = _rng_uniform_graph(attack_rng.to(torch.int64))   # advance rng regardless
    if attack_threshold is not None:
        atk = (p_attack >= float(attack_threshold)).to(torch.int64)   # deterministic
    else:
        atk = (u < p_attack).to(torch.int64)                          # sampled
    wid = self_weapon_id.reshape(-1).to(torch.int64)
    is_cont = torch.zeros(B, dtype=torch.bool)
    for cw in QNN_CONTINUOUS_WEAPON_IDS:
        is_cont = is_cont | (wid == cw)
    attack_now = is_cont & (atk == 1)
    tail = is_cont & (atk == 0) & (attack_hold > 0)
    attack = torch.where(attack_now | tail, torch.ones_like(atk), atk)
    # arm to ATTACK_HOLD_TICKS on a continuous attack, else decrement while tailing,
    # else reset to 0 (matches the C if/elif/else).
    attack_hold = torch.where(
        attack_now,
        torch.full_like(attack_hold, QNN_ATTACK_HOLD_TICKS),
        torch.where(tail, attack_hold - 1, torch.zeros_like(attack_hold)),
    )
    return (attack.reshape(-1, 1), attack_hold.to(torch.float32).reshape(-1, 1),
            rng_out.to(torch.int64))


def assemble_aim_prior(
    z_err: torch.Tensor,
    z_rate: torch.Tensor,
    gain: float,
    ffwd: float,
) -> torch.Tensor:
    """Assemble the PRE-SCALED aim-prior blend term ``z_prior``.

    ``z_prior = gain·z_err + ffwd·z_rate``. ``gain`` is the single human-percentile
    placement knob (rotation steering); ``ffwd`` is the rate feed-forward. See
    src/docs/look-head.md §5 and the fire-discrimination doc.
    """
    return gain * z_err + ffwd * z_rate


def assemble_pitch_correction(
    z_err: torch.Tensor,
    weapon_pitch_gain: torch.Tensor,
    self_weapon_id: torch.Tensor,
    weapon_pitch_bias: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per-row VERTICAL feet-aim BLEND terms — the RL-splash feet-aiming.

    ``z_err`` is the UNSCALED aim-prior error tangent (the log-mapped error to the
    soft-pooled, feet-anchored lead point; vertical component ``z_err[..., 1]`` is
    the ABSOLUTE turn-to-anchor pitch, + = up / − = down). ``weapon_pitch_gain`` is
    a (9,) per-IMPULSE blend weight β ∈ [0, 1] (impulse 0..8; 0 = OFF, head keeps
    full vertical authority; 1 = vertical fully overridden by the geometric anchor).
    ``self_weapon_id`` is the ENTITY_IDS-encoded held weapon, mapped to impulse via
    the canonical helper (never a bare ±2).

    Returns ``(beta, target_vert)``, each ``(R,)``, consumed by
    :func:`decode_look_from_polar` as the LERP
    ``z_vert ← (1−β)·z_head_vert + β·target_vert`` (i.e. ``z_vert += β·(target_vert
    − z_head_vert)``).

    Why a blend, not the former additive ``k·z_err_vert``: that term was
    SELF-LIMITING against a head that targets center mass, so it lost the per-tick
    tug-of-war and settled at only ``~k/(1+k)`` of the anchor depth (≈1° below
    center at RL range — the rockets-over-the-head failure; look-aim-decode.md §12).
    The blend instead drives the FINAL vertical toward the absolute anchor
    regardless of the head's vertical pull: with the head re-centering at gain h the
    closed-loop equilibrium is ``β·anchor / (1 − (1−β)(1−h))`` (= ``β·anchor`` at
    h=1, = the full anchor at h=0), so β maps near-linearly to achieved feet-depth.
    The feet TARGET lives in ``z_err`` (AIM_Z_DROP, applied inside compute_lead_aim);
    this restores the AUTHORITY the rotation-magnitude blend starves.

    β is gated to enemy-present frames (``z_err`` is exactly zero rows otherwise),
    so the head's vertical look is untouched — NOT damped — when no target is
    perceived (which a bare β·lerp toward 0 would wrongly do).

    VERTICAL BIAS (``weapon_pitch_bias``, a (9,) per-IMPULSE downward offset in
    DEGREES; default None = OFF = bit-identical). The live RL aim sits a persistent
    ~1.5° ABOVE the feet anchor at fire (→ straight rocket sails over → lands
    behind) — an offset that is immune to both β (the proportional weight) and a
    z_rate feed-forward (measured flat), i.e. a STATIC achieved-vs-commanded gap in
    the az/el-coupled tangent decode, not a gain or velocity-lag problem. The bias
    deepens the per-weapon aim target by a fixed angle to cancel that measured
    offset (subtracted = aim DOWN). Folded into ``target_vert`` so it rides β's
    enemy-present gate; the deployed magnitude is set by the fire-pitch→0 sweep
    (β dilutes it, so the swept value already accounts for β).
    """
    from qnn.vocab import self_weapon_id_to_impulse
    imp = self_weapon_id_to_impulse(self_weapon_id.reshape(-1).long()).clamp(0, 8)
    beta = weapon_pitch_gain.reshape(-1).index_select(0, imp)            # (R,)
    z = z_err.reshape(-1, z_err.shape[-1])                               # (R, 2)
    feet_vert = z[:, 1]                                                  # (R,) UN-biased feet-anchor pitch (floor)
    target_vert = feet_vert                                             # drive target
    if weapon_pitch_bias is not None:
        bias_deg = weapon_pitch_bias.reshape(-1).index_select(0, imp)    # (R,) deg
        # deg→rad as a plain multiply (aten::deg2rad does NOT export to opset 18).
        # Drive DOWN past the feet to overcome the head/blend residual; the decode's
        # floor-clamp (at feet_vert) stops the fired aim going BELOW the feet at range.
        target_vert = target_vert - bias_deg * 0.017453292519943295     # + = aim DOWN
    # presence gate: aim_prior_tangent_ffwd zeroes BOTH components on no-enemy
    # frames, so a nonzero L1 means an enemy is perceived. Zeroing β there keeps
    # the lerp a no-op (z_vert unchanged) instead of shrinking the head's vertical.
    present = (z.abs().sum(dim=-1) > 1e-9).to(beta.dtype)               # (R,)
    return beta * present, target_vert, feet_vert


def decode_look_from_polar(
    mag_bin: torch.Tensor,
    dir_logits: torch.Tensor,
    mag_centers: torch.Tensor,
    dir_centers: torch.Tensor,
    z_prior: torch.Tensor | None = None,
    mag_gain: float = 0.0,
    turn_mag_scale: float = 1.0,
    pitch_correction: "tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None" = None,
    feet_elev: "torch.Tensor | None" = None,
    origin_elev: "torch.Tensor | None" = None,
    pitch_mode: str = "lock",
    shift_strength: float = 1.0,
) -> torch.Tensor:
    """(sampled ``mag_bin``, ``dir_logits``) → look unit vector (hybrid decode).

    ``turn_mag_scale`` (default 1.0 = OFF = bit-identical) is a multiplicative
    dampener on the head's NATIVE turn magnitude ``|z| = θ`` — applied BEFORE the
    aim-prior blend. It exists to correct the head's CONDITIONAL turn-magnitude
    over-turn (the open-loop ~1.28× greedy / ~1.34× sampled head term; see
    runs/head_probe/_look_openloop_vs_closedloop.json and the standing
    over-turn check, scripts/analysis/look_openloop_los_overturn.py). The
    aim-prior blend keeps the (scaled) ``|z|`` direction-preserving, so scaling θ
    here dampens the head's own turn WITHOUT touching the aim-prior placement
    steering; gain (placement) and lag (DOWN-half degrade) still compose
    correctly. NOTE this knob reaches ONLY the decode-adjustable HEAD term; the
    dominant closed-loop COVARIATE-SHIFT over-turn (~1.47×) is training-side and
    is NOT touched by it.

    The caller supplies ``mag_bin`` from ITS OWN magnitude sampler (seeded
    categorical offline / in-graph Gumbel-argmax for export) — magnitude
    sampling legitimately differs between the two paths and is NOT unified here.
    This function owns the shared remainder: the continuous DIRECTION readout,
    the z-assembly, the aim-prior blend, and the expmap.

    DIRECTION is the continuous circular mean of the direction softmax, NOT a
    sampled/argmax bin: per-frame direction sampling + the 16-bin (22.5°)
    quantization break the heading-hold the head learned and produce the live
    "spin" (consec-dir reversal 2.1%→6.2% human→sampled). The circular mean is
    defined on every frame and unquantized, restoring near-human heading
    persistence (reversal 3.2%) while leaving the magnitude distribution
    untouched. See src/docs/look-head.md §3 and qnn.model.look_bins.

    ``mag_centers`` / ``dir_centers`` are passed in (MAG_CENTERS / DIR_CENTERS
    from look_bins, possibly device-moved or held as export buffers) so this
    function stays free of device/buffer policy.

    Returns the look unit vector with shape ``mag_bin.shape + (3,)``.
    """
    # polar_to_tangent via index_select (export-friendly vs advanced index —
    # ONNX tracing rejects advanced/boolean indexing).
    theta = mag_centers.index_select(0, mag_bin.reshape(-1)).reshape(mag_bin.shape)
    # Head turn-magnitude dampener (turn_mag_scale, default 1.0 = no-op): scale the
    # head's NATIVE |z| = θ before the aim-prior blend. 1.0 leaves theta untouched
    # (bit-identical); a Python-float != 1.0 multiplies, so the no-op path emits the
    # exact same tensor object/values as before this knob existed.
    if turn_mag_scale != 1.0:
        theta = theta * float(turn_mag_scale)
    # Continuous direction: circular mean of the direction distribution.
    p_dir = torch.softmax(dir_logits, dim=-1)                       # (..., N_DIR)
    cos_phi = (p_dir * torch.cos(dir_centers)).sum(dim=-1)
    sin_phi = (p_dir * torch.sin(dir_centers)).sum(dim=-1)
    phi = torch.atan2(sin_phi, cos_phi)                             # (...,)
    z = torch.stack([theta * torch.cos(phi), theta * torch.sin(phi)], dim=-1)
    if z_prior is not None:
        # AIM-PRIOR blend (pointer-bearing models only): rotation + magnitude, the
        # SOLE blend. Direction is always the rotated heading normalize(z + z_prior)
        # (aim steers HEADING); the kept MAGNITUDE blends from |z|=θ (mag_gain=0,
        # PURE ROTATION — turn-size invariant, the deployed default) toward
        # |z+z_prior| (mag_gain=1, exactly the legacy vector-ADD), and mag_gain>1
        # overshoots into the super-human slew band. So mag_gain (α) carries the
        # whole pure-rotation → ADD → super axis in ONE knob — there is no separate
        # rotate toggle (legacy ADD == mag_gain=1). Vector-ADD's coupling (more
        # gain ⇒ bigger turns ⇒ turn-EMD drifts off-human) is why pure rotation is
        # the default. z_prior is zero on no-enemy frames (reduces to the plain
        # hybrid decode). The caller assembles z_prior PRE-SCALED (assemble_aim_prior).
        # See qnn.model.bench.a24.lead_aim + src/docs/look-head.md §5/§7 / look-aim-decode.md.
        z_t = z + z_prior
        mag = torch.linalg.vector_norm(z, dim=-1, keepdim=True)        # head |z| = θ
        n = torch.linalg.vector_norm(z_t, dim=-1, keepdim=True)
        if mag_gain != 0.0:
            mag = mag + float(mag_gain) * (n - mag)                    # θ → |z+z_prior|
        # where z_t collapses to ~0 (tiny head turn + no aim), keep z unrotated.
        z = torch.where(n > 1e-9, mag * z_t / n.clamp_min(1e-9), z)
    if pitch_correction is not None:
        # VERTICAL feet-aim BLEND (RL-splash): lerp the decoded turn's vertical
        # component toward the absolute geometric anchor, OUTSIDE the rotation-
        # magnitude clamp that starves the anchor's vertical pull. β·(target − z)
        # drives the crosshair to the feet regardless of the head's center-mass
        # pull — unlike the former additive self-limiting term, which lost the
        # per-tick tug-of-war and settled ~1° below center (rockets-over-the-head;
        # §12). β is enemy-gated, so no-target frames are an exact no-op. Default
        # None = bit-identical no-op. Assembled by assemble_pitch_correction.
        #
        # FLOOR CLAMP (feet_vert): the bias drives target_vert BELOW the feet to
        # overcome the residual, which pulls the blended vertical onto the feet up
        # close — but a fixed angular drive aims below the feet at long range, and a
        # straight rocket aimed below-feet hits the FLOOR short ("can only shoot the
        # ground past a max range"). Clamp the fired vertical so it never points
        # below the feet anchor (feet_vert, − = down): torch.maximum keeps the LESS
        # -down of the two. Gated to β>0 (RL + enemy present) via arithmetic mask so
        # no-enemy / non-feet-aim frames are untouched (feet_vert is 0 there).
        # SHIFT mode SKIPS the tangent β-blend: the blend collapses the vertical toward
        # a point, which would destroy the head's spread that the shift exists to keep.
        # The shift (post-expmap, below) is the sole vertical op in that mode.
        if pitch_mode != "shift":
            beta, target_vert, feet_vert = pitch_correction
            beta = beta.reshape(z[..., 1].shape)
            target_vert = target_vert.reshape(z[..., 1].shape)
            feet_vert = feet_vert.reshape(z[..., 1].shape)
            z_vert = z[..., 1] + beta * (target_vert - z[..., 1])   # blend toward biased drive
            gate = (beta > 0).to(z_vert.dtype)                      # RL + enemy present
            z_vert = torch.maximum(z_vert, feet_vert) * gate + z_vert * (1.0 - gate)
            z = torch.stack([z[..., 0], z_vert], dim=-1)
    look_vec = tangent_expmap(z)                                    # (..., 3) fwd,right,up
    if pitch_correction is not None and feet_elev is not None:
        # POST-EXPMAP FEET-ELEVATION LOCK (RL splash robustness). Set the fired
        # look's ELEVATION to the feet anchor (feet_elev), keeping AZIMUTH, AFTER the
        # turn. The tangent-space pitch blend above is coupled to the horizontal
        # tracking turn (expmap shaves the vertical when |yaw turn| is large), which
        # left RL ~1.3° above the feet at range → the rocket rides at body height and
        # a lateral miss FLIES BY instead of hitting the floor (no splash). Locking
        # the elevation after the turn puts the shot ON the ground at the target's
        # feet regardless of the horizontal turn, so a lateral miss still detonates
        # on the floor within the 120u splash. Gated to β>0 (RL + enemy present);
        # feet_elev is 0 on no-enemy frames so non-gated rows are untouched anyway.
        beta_g = pitch_correction[0].reshape(look_vec[..., 0].shape)
        gate_e = (beta_g > 0).to(look_vec.dtype)
        fe = feet_elev.reshape(look_vec[..., 0].shape)
        h = torch.linalg.vector_norm(look_vec[..., :2], dim=-1).clamp_min(1e-6)
        # Target elevation to set (keeping azimuth). Two modes:
        #  • "lock"  → feet_elev exactly (aimbot: collapses the head's vertical variance).
        #  • "shift" → translate the head's OWN fired elevation DOWN by
        #    shift_strength·(origin_elev − feet_elev) = the center-mass→feet angle.
        #    The head tracks center-mass; this moves the aim to the feet while PRESERVING
        #    the head's spread (human RL aim is a wide cloud, not a point). strength can
        #    exceed 1 to fight the closed-loop re-centering ceiling.
        if pitch_mode == "shift" and origin_elev is not None:
            cur = torch.atan2(look_vec[..., 2], h)
            oe = origin_elev.reshape(look_vec[..., 0].shape)
            tgt = cur - float(shift_strength) * (oe - fe)
        else:
            tgt = fe
        ce = torch.cos(tgt)
        setv = torch.stack(
            [ce * look_vec[..., 0] / h, ce * look_vec[..., 1] / h, torch.sin(tgt)], dim=-1)
        look_vec = setv * gate_e.unsqueeze(-1) + look_vec * (1.0 - gate_e).unsqueeze(-1)
    return look_vec


def build_weapon_ban_tensors(
    weapon_ban: "tuple[int, ...]",
    device: torch.device | str | None = None,
) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
    """``weapon_ban`` (csv impulses 1..8) → ``(ban_mask, ban_bool)`` for the gate.

    SINGLE SOURCE OF TRUTH for the weapon-ban tensor construction shared by the
    ONNX :class:`tools.export_onnx.ExportWrapper` and the offline
    :meth:`qnn.model.policy.QNNPolicy.act` so the two decodes are bit-identical
    given the same ban. ``ban_mask`` is an additive pre-softmax mask (``-1e9`` on
    each banned class, 0 elsewhere — banned classes can never win the argmax);
    ``ban_bool`` is a per-class banned flag (a banned HELD weapon force-switches
    to the best legal class). Impulses are 1..8 and map to classes 0..7
    (``class = impulse - 1``).

    Returns ``(None, None)`` for an empty ban so callers pass nothing to
    :func:`decide_weapon_sticky` and run the gate UNBANNED — preserving the
    no-ban behavior bit-for-bit. See src/docs/weapon-head.md.
    """
    ban = tuple(int(b) for b in weapon_ban)
    if not ban:
        return None, None
    ban_mask = torch.zeros(8, device=device)
    ban_bool = torch.zeros(8, dtype=torch.bool, device=device)
    for imp in ban:
        ban_mask[imp - 1] = -1e9          # impulse 1..8 → class 0..7
        ban_bool[imp - 1] = True
    return ban_mask, ban_bool


def decide_weapon_sticky(
    weapon_logits: torch.Tensor,
    current_class: torch.Tensor,
    confidence_thresh: float,
    margin_thresh: float,
    ban_mask: torch.Tensor | None = None,
    ban_bool: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sticky-weapon controller → (N,) int64 engine impulse 1..8.

    Switch to the top-1 class only when it differs from the held weapon AND is
    both confident (top-1 prob ≥ ``confidence_thresh``) and separated
    (top1−top2 prob ≥ ``margin_thresh``); otherwise HOLD the current weapon (a
    no-op impulse). ``current_class`` is the held weapon class (0..7), mapped by
    the caller from ``self_weapon_id`` via the training-time weapon_index_from_id
    so the deployed decision matches the offline path. The returned impulse byte
    (class + 1) goes straight onto the engine wire.

    WEAPON BAN (decode contract, EXPORT path only today — see module note in the
    callers): when ``ban_mask`` / ``ban_bool`` are supplied, banned classes can
    never win the argmax (their logits are masked to -inf so softmax renormalizes
    their mass onto the legal classes before the gate sees it), AND a banned HELD
    weapon force-switches to the best legal class regardless of the confidence
    gate (else the gate would HOLD a banned weapon). When both are None the gate
    runs UNBANNED — this preserves the offline policy's current behavior, which
    applies no ban. See src/docs/weapon-head.md.
    """
    flat = weapon_logits.reshape(-1, weapon_logits.shape[-1])
    if ban_mask is not None:
        # banned classes can never win the argmax (softmax renormalizes their
        # mass onto the legal classes before the gate sees it)
        flat = flat + ban_mask
    probs = torch.softmax(flat, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1)
    desired_class = top2.indices[:, 0]
    confidence = top2.values[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]
    current_class = current_class.reshape(-1)
    should_switch = (
        (desired_class != current_class)
        & (confidence >= confidence_thresh)
        & (margin >= margin_thresh)
    )
    if ban_bool is not None:
        # holding a banned weapon (spawn state, engine-forced ammo-out)
        # force-switches to the best legal class — the gate must never HOLD a
        # banned weapon for lack of confidence.
        held_banned = ban_bool.index_select(0, current_class.clamp(0, 7))
        should_switch = should_switch | held_banned
    chosen_class = torch.where(should_switch, desired_class, current_class)
    return (chosen_class + 1).to(torch.int64)            # class 0..7 → impulse 1..8
