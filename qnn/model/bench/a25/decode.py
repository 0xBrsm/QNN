"""a25 attack-with decode — the joint 9-way head's deploy protocol.

One trace-safe function shared by ``QNNPolicy.act`` and the ONNX export
wrapper (decode-regime-in-model: the engine consumes the FINAL fire bit +
weapon impulse; no engine-side decode). Greedy by construction — the
deterministic commit preserves coordinated actions (rocket jump), and the
offline shape analysis showed raw argmax self-calibrates to ~the human fire
rate (runs/bc/bench/_attack_with_shape.json).

Decode law (per frame):

    # selection (which weapon to attack-with) — sticky hysteresis toward held
    sel    = (l1..l8) + stick_bias * onehot(held)
    choice = argmax(sel) + 1                       # engine impulse 1..8
    # attack decision (whether) — per-weapon operating point, POST-selection
    l0'    = l0 - attack_bias - align_bias         # class-0 = "don't attack" anchor
    fire   = (l_choice + bias_vec[choice]) > l0'   AND NOT veto
    weapon = fire ? choice : held_impulse          # held = server no-op

Knob mapping from the a24 decode contract:
  * ``attack_bias`` (the s-slider / propensity knob) and the alignment-
    conditioned bias both act on the MARGINAL log-odds by subtracting from
    the class-0 logit — sign-compatible with the a24 semantics (bias > 0 →
    more attack).
  * ``bias_vec`` (8,) is the PER-WEAPON operating point (research/attack-head.md
    §11): applied to the SELECTED weapon's logit AFTER the argmax, so it tunes
    each weapon's attack rate to its own human cadence (LG/NG/SNG ~3.5x RL)
    with ZERO effect on weapon selection. A uniform ``bias_vec`` reproduces the
    scalar ``attack_bias`` exactly; ``None`` is bit-identical to no per-weapon
    knob. Attack rate is fit to the per-weapon op-attack targets offline.
  * ``stick_bias`` is selection-side hysteresis toward the held weapon — added
    to the held weapon's logit BEFORE the argmax to stop selection chatter. It
    moves *which* weapon, never *whether* to attack (the attack test re-reads
    the un-stickied logit), so it composes cleanly with ``bias_vec``.
  * hard guards (attack-splash / RL self-splash / LG range) arrive as a
    boolean veto mask from the a24 guard primitives instead of a buried
    logit.
  * ``crest_theta_vec`` / ``crest_hold_ticks`` — the DISCHARGE-QUALITY gate
    ("crest-firing"): a deterministic countdown latch that shifts WHEN a
    commanded fire lands within a bounded window (≤ H ticks) without changing
    the count — hold until crosshair→lead alignment crosses θ_w, blind-fire at
    expiry. Composes AFTER the step (guards outrank); OFF = bit-identical.
    See attack_crest_gate_step + agents/plans/discharge-quality-gate.md.
  * the a24 sticky gate / hazard / switch-back weapon machinery has no analog —
    it is retired for this head; selection happens only at attack frames and
    emitting the held impulse otherwise is a server no-op. ``stick_bias`` is a
    new, single-scalar hysteresis, NOT a revival of that stack.

TRACE-SAFETY: torch-only, no ``.item()``, no data-dependent control flow
(``stick_bias``/``bias_vec`` presence is decided on static python config, not
tensor values).
"""
from __future__ import annotations

import torch

from qnn.model.decode import BatchedRNG, inverse_cdf_sample, row_uniforms
from qnn.model.look_bins import tangent_expmap
from qnn.schema import WEAPON_HEAD_SIZE

# Re-exported so this module is the a25 generation's complete decode FACADE: the core
# (QNNPolicy.act) resolves one decode_module from the run's decode config and reads
# everything it decodes with through it — move commitment, attack-with, AND the
# aim-prior (gains + ffwd primitive). The aim primitives' home is the a25-owned
# qnn.model.bench.a25.lead_aim (cloned from a24 so a25 executes no a24 code); this is
# the facade seam, not a move.
from qnn.model.bench.a25.lead_aim import (  # noqa: F401
    AIM_FFWD_GAIN, AIM_PRIOR_GAIN, _TICK_DT_MODULE, aim_prior_tangent_ffwd,
)

ATTACK_WITH_SIZE = 1 + WEAPON_HEAD_SIZE  # 9


def _attack_with_select(
    weap: torch.Tensor,
    held: torch.Tensor,
    stick_bias: float,
) -> torch.Tensor:
    """Selection argmax over (optionally) held-stickied weapon logits.

    The SINGLE selection law shared by :func:`attack_with_decode_step` and the
    crest gate's per-tick θ re-read (selection is re-evaluated each tick, so a
    weapon switch mid-hold releases against the release-tick selection's θ).
    Returns the (B,) class index 0..7 (impulse − 1)."""
    if stick_bias != 0.0:
        held_idx = (held - 1).clamp(0, WEAPON_HEAD_SIZE - 1)
        stick = torch.zeros_like(weap).scatter(
            -1, held_idx.unsqueeze(-1), float(stick_bias))
        return (weap + stick).argmax(dim=-1)
    return weap.argmax(dim=-1)


def attack_with_decode_step(
    logits9: torch.Tensor,
    held_impulse: torch.Tensor,
    *,
    attack_bias: float = 0.0,
    bias_vec: torch.Tensor | None = None,
    stick_bias: float = 0.0,
    align_bias: torch.Tensor | None = None,
    veto_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy attack-with decode.

    Parameters
    ----------
    logits9 : (B, 9) raw head logits (class 0 = no-attack, 1..8 = attack weapon k)
    held_impulse : (B,) int64 engine impulse 1..8 of the currently held weapon
    attack_bias : global propensity knob (positive = more attack)
    bias_vec : optional (8,) per-weapon attack operating point, applied to the
        SELECTED weapon's logit AFTER the argmax (positive = more attack for
        that weapon). None = no per-weapon knob (bit-identical to today).
    stick_bias : selection hysteresis toward held weapon (>= 0), added to the
        held weapon's logit BEFORE the argmax. 0.0 = plain argmax.
    align_bias : optional (B,) additive alignment bias (positive = more attack)
    veto_mask : optional (B,) bool — hard guard veto rows (True = never attack)

    Returns
    -------
    (attack, weapon_impulse) : (B,) int64 attack bit, (B,) int64 impulse 1..8
    """
    weap = logits9[..., 1:]                                  # (B, 8) raw weapon logits
    held = held_impulse.reshape(weap.shape[:-1]).to(torch.long)  # impulse 1..8
    # ── selection: argmax over (optionally) held-stickied logits ────────────
    rest_arg = _attack_with_select(weap, held, stick_bias)
    # ── attack decision: per-weapon operating point on the SELECTED weapon ──
    # re-read the UN-stickied logit of the chosen weapon so stick_bias moves
    # only *which* weapon, never *whether* to attack.
    sel_logit = weap.gather(-1, rest_arg.unsqueeze(-1)).squeeze(-1)
    if bias_vec is not None:
        sel_logit = sel_logit + bias_vec.reshape(-1).to(sel_logit.dtype)[rest_arg]
    l0 = logits9[..., 0] - float(attack_bias)
    if align_bias is not None:
        l0 = l0 - align_bias.reshape(l0.shape).to(l0.dtype)
    fire = sel_logit > l0
    if veto_mask is not None:
        fire = fire & ~veto_mask.reshape(fire.shape)
    choice = rest_arg.to(held.dtype) + 1                     # impulse 1..8
    weapon_impulse = torch.where(fire, choice, held)
    return fire.to(torch.int64), weapon_impulse.to(torch.int64)


def attack_with_marginal_logit(logits9: torch.Tensor) -> torch.Tensor:
    """(B,) marginal attack log-odds: LSE(l1..l8) - l0 (diagnostics / decode-fit)."""
    return torch.logsumexp(logits9[..., 1:], dim=-1) - logits9[..., 0]


# ── DISCHARGE-QUALITY GATE ("crest-firing"; agents/plans/discharge-quality-gate.md)
#
# When the head commands fire, the decode may HOLD the discharge up to H ticks
# until crosshair→lead-point alignment crosses a per-weapon threshold θ_w — fire
# at first crossing, else BLIND at H expiry (honor the head; never cancel — a
# canceled fire is silent rate starvation, the rc2a failure class). Same family
# as move.threat_break_hazard: a decode bridge for a conditioning the head
# doesn't carry natively (the attack head is alignment-blind); the fitted θ is
# the training register for an eventual alignment feature on the attack head.
#
# The latch stores ONE countdown in the existing attack_state wire slot (lane 0;
# 0 = idle, so the existing zeros-init/episode-reset loopback already means
# idle — no wire bump). Deterministic (no RNG): greedy and sampled run it
# identically, and the export decode-parity gate exercises it for free.

# Angular hitbox radius geometry: the eval's _intercept_hbw law shares this
# half-width; range here is in lead_aim MODULE units (/DIST_SCALE), so the
# atan(halfw/range) ratio is unit-consistent (angle ratios are scale-free).
from qnn.engine_norm import DIST_SCALE as _DIST_SCALE  # noqa: E402
from qnn.eval.aim_kernel import ACTOR_HALFW_U as _ACTOR_HALFW_U  # noqa: E402
_CREST_HALFW_MOD = _ACTOR_HALFW_U / _DIST_SCALE


def crest_alignment_hbw(
    z_err: torch.Tensor,       # (B, 2) UNSCALED aim-prior error tangent (radians)
    aim_range: torch.Tensor,   # (B,) pooled lead-point range, module units
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(hbw, live)`` — crosshair→lead-point alignment in hitbox-half-widths.

    ``hbw = |z_err| / atan(halfw / range)`` — the eval's ``_intercept_hbw`` law
    on the obs-side aim geometry (the small obs-vs-engine ruler gap is absorbed
    by the closed-loop confirmation fit). ``live`` is the enemy-presence gate:
    ``aim_prior_tangent_ffwd`` emits exact-zero z_err/range rows when no enemy
    is perceived (the existing β-gate convention), and a dead row must never
    read as aligned. Trace-safe (torch-only, no data-dependent control flow)."""
    z = z_err.reshape(-1, 2)
    rng = aim_range.reshape(-1).to(z.dtype)
    live = (z.abs().sum(dim=-1) > 1e-9) & (rng > 1e-9)
    ang = torch.linalg.vector_norm(z, dim=-1)                       # (B,) radians
    radius = torch.atan2(torch.full_like(rng, _CREST_HALFW_MOD),
                         rng.clamp_min(1e-6))
    return ang / radius.clamp_min(1e-9), live


def attack_crest_gate_step(
    fire_raw: torch.Tensor,       # (B,) int64 — attack_with_decode_step's decision
    choice: torch.Tensor,         # (B,) int64 — this tick's selected impulse 1..8
    held_impulse: torch.Tensor,   # (B,) int64 — held impulse 1..8 (server no-op)
    attack_state: torch.Tensor,   # (B, ATTACK_STATE_DIM) float32 — lane 0 = countdown
    *,
    crest_theta_vec: torch.Tensor,  # (8,) θ_w per impulse-1, hbw units; <= 0 = OFF for w
    crest_hold_ticks: int,          # H, max hold in ticks (static; caller gates on > 0)
    hbw: torch.Tensor,              # (B,) crosshair→lead alignment (crest_alignment_hbw)
    live: torch.Tensor,             # (B,) bool — enemy perceived (z_err presence gate)
    ready: torch.Tensor,            # (B,) bool — obs attack_finished expired (engine honors)
    veto_mask: torch.Tensor | None = None,  # (B,) bool — hard guards outrank the gate
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """One tick of the crest latch. Returns ``(attack, weapon_impulse,
    attack_state_out)``.

    Decode law (all rows in parallel; countdown ``pending`` = state lane 0):

    * arm     — head fires, gate applies (θ>0, enemy live, weapon ready), not at
                crest, latch idle → HOLD: emit attack=0 + held impulse (a server
                no-op), start the countdown at H.
    * release — at crest (``hbw ≤ θ`` of the CURRENT selection), or countdown
                expiring (pending==1 → blind-fire; the head's discharge is never
                canceled), or a gate-exempt fire (θ≤0 / no enemy / on cooldown /
                already aligned) passing straight through on its own tick.
    * fire    — release minus this tick's hard-guard veto (guards outrank; a
                vetoed release still clears the latch — the discharge is lost,
                same as a vetoed raw fire).

    Cooldown-honesty: ``ready`` gating means a fire the engine would discard
    (attack_finished pending) is passed through raw, never converted into a
    delayed REAL discharge. No restack: arm requires pending==0, so a head
    re-fire during a hold is absorbed into the pending discharge. Exactly one
    attack=1 tick per armed discharge (single-tick preserved)."""
    st = attack_state.reshape(-1, ATTACK_STATE_DIM)
    pending = st[:, 0].round().to(torch.int64)
    fire_b = fire_raw.reshape(-1).to(torch.bool)
    ch = choice.reshape(-1).to(torch.int64)
    theta = crest_theta_vec.reshape(-1).to(hbw.dtype).index_select(
        0, (ch - 1).clamp(0, WEAPON_HEAD_SIZE - 1))
    gate_on = (theta > 0) & live & ready
    # aligned needs live: a dead z_err row (hbw 0/undefined) must hold to expiry
    # (LOS-lost law), not release as a fake crest. θ≤0 rows can never align
    # (hbw > 0 whenever live), so a mid-hold switch to an OFF weapon runs to
    # expiry — the release-tick selection's θ is what's evaluated.
    aligned = live & (hbw <= theta)
    idle = pending == 0
    arm = fire_b & gate_on & ~aligned & idle
    tick = pending > 0
    release = ((fire_b & (aligned | ~gate_on) & idle)
               | (tick & (aligned | (pending == 1))))
    fire = release if veto_mask is None else release & ~veto_mask.reshape(release.shape)
    pending_next = torch.where(
        arm, torch.full_like(pending, int(crest_hold_ticks)),
        torch.where(tick & ~release, pending - 1, torch.zeros_like(pending)))
    held = held_impulse.reshape(-1).to(torch.int64)
    weapon_impulse = torch.where(fire, ch, held)
    # rebuild, don't mutate (trace-safe; lane 0 is the whole a25 attack state).
    attack_state_out = pending_next.to(attack_state.dtype).unsqueeze(-1)
    return fire.to(torch.int64), weapon_impulse, attack_state_out


def attack_with_decode(
    logits9: torch.Tensor,
    self_weapon_id: torch.Tensor,
    obs_tensors,
    move_logits: torch.Tensor,
    *,
    guard=None,
    attack_bias: float = 0.0,
    bias_vec: torch.Tensor | None = None,
    stick_bias: float = 0.0,
    crest_theta_vec: torch.Tensor | None = None,
    crest_hold_ticks: int = 0,
    aim_z_err: torch.Tensor | None = None,
    aim_range: torch.Tensor | None = None,
    attack_state: torch.Tensor | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]":
    """Full a25 attack-with decode — the SINGLE implementation shared by
    ``QNNPolicy.act`` and the ONNX ExportWrapper (decode-regime-in-model: the
    offline decode and the baked-into-ONNX deploy decode are the same code, so
    they cannot skew). Resolves the held impulse and the guard's alignment bias +
    hard-veto mask, runs :func:`attack_with_decode_step`, then composes the
    crest gate (:func:`attack_crest_gate_step`) when armed.

    Parameters
    ----------
    logits9 : (B, 9) raw attack-with logits.
    self_weapon_id : (B,) / (B,1) held-weapon id (obs ``self_weapon_id``).
    obs_tensors : the guard's obs input (each caller passes the form its guard
        expects — dequantised in act, native in export; the guard reads it).
        The crest gate also reads ``attack_finished`` from it (cooldown gate).
    move_logits : (B, MOVE_AXES, classes) or flat — the guard argmaxes it for the
        movement-direction splash checks.
    guard : the resolved guard module (``guard_attack_logit_for_export``); None →
        no alignment bias / veto. The guard applies the SAME hard vetoes
        (attack-splash / RL self-splash / LG range) that the deployed graph bakes.
    attack_bias / bias_vec / stick_bias : the operating-point knobs
        (research/attack-head.md §11), forwarded to :func:`attack_with_decode_step`.
    crest_theta_vec / crest_hold_ticks : the discharge-quality gate
        (attack.crest_theta_vec (8,) θ per impulse-1, ≤0 = OFF per weapon;
        attack.crest_hold_ticks shared H, 0 = OFF globally). Both OFF (the
        default) is BIT-IDENTICAL: the step result passes straight through and
        ``attack_state`` is returned untouched.
    aim_z_err / aim_range : the crest alignment signal — the UNSCALED aim-prior
        error tangent + pooled lead-point range from ``aim_prior_tangent_ffwd``
        (both callers compute it upstream for the look decode). Required when
        the gate is armed; a pointer-less caller must fail loud, not skip.
    attack_state : (B, ATTACK_STATE_DIM) — the countdown latch's wire slot
        (lane 0; zeros = idle). Required when the gate is armed.

    Returns ``(attack, weapon_impulse, attack_state_out)``; with the gate OFF
    ``attack_state_out`` is the input ``attack_state`` unchanged (passthrough).

    TRACE-SAFETY: torch-only, no ``.item()`` / data-dependent control flow (the
    guard presence + knobs are static python config, decided at trace build).
    """
    from qnn.model.weapon_head import weapon_index_from_id

    l9 = logits9.reshape(-1, ATTACK_WITH_SIZE)
    held_impulse = weapon_index_from_id(self_weapon_id.reshape(-1)) + 1
    align_bias = None
    veto_mask = None
    if guard is not None and hasattr(guard, "guard_attack_logit_for_export"):
        # Probe the guard on a zeros logit: the adapter is additive (alignment
        # bias) except hard vetoes buried at -1e9, so the probe separates them —
        # veto = probe < -1e8, align_bias = probe elsewhere.
        # The resolved guard's export hook is a make_guard closure that owns its
        # alignment strength — call it with (obs, move, logit) exactly as the
        # original export did; do NOT pass a 4th strength arg.
        probe = guard.guard_attack_logit_for_export(
            obs_tensors, move_logits, torch.zeros_like(l9[..., :1])).reshape(-1)
        veto_mask = probe < -1.0e8
        align_bias = torch.where(veto_mask, torch.zeros_like(probe), probe)
    fire, weapon_impulse = attack_with_decode_step(
        l9, held_impulse,
        attack_bias=attack_bias, bias_vec=bias_vec, stick_bias=stick_bias,
        align_bias=align_bias, veto_mask=veto_mask)
    # crest activation is STATIC python config (trace-safe branch): OFF returns
    # the step result + the untouched state slot, bit-identical to pre-crest.
    if int(crest_hold_ticks) <= 0 or crest_theta_vec is None:
        return fire, weapon_impulse, attack_state
    if aim_z_err is None or aim_range is None:
        raise ValueError(
            "attack crest gate is armed (crest_hold_ticks>0, crest_theta_vec set) "
            "but no aim geometry was supplied — the gate needs the aim-prior "
            "z_err tangent + pooled range (target pointer required; a "
            "pointer-less model cannot run θ>0).")
    if attack_state is None:
        raise ValueError(
            "attack crest gate is armed but attack_state was not threaded — the "
            "countdown latch rides the (B, ATTACK_STATE_DIM) wire slot.")
    if obs_tensors is None or "attack_finished" not in obs_tensors:
        raise ValueError(
            "attack crest gate is armed but obs 'attack_finished' is missing — "
            "the cooldown gate (ready) keeps the latch from converting "
            "engine-discarded fires into delayed real discharges.")
    # ready: the engine would honor this fire (QC cooldown expired — the same
    # `<= eps` the eval's discharge definition uses; 0 is 0 in any scaling).
    ready = obs_tensors["attack_finished"].reshape(-1).float() <= 1e-6
    hbw, live = crest_alignment_hbw(aim_z_err, aim_range)
    # per-tick selection re-read (same law as the step, incl. stick hysteresis)
    # so a mid-hold weapon switch releases against the NEW weapon's θ.
    choice = _attack_with_select(l9[..., 1:], held_impulse, stick_bias) + 1
    return attack_crest_gate_step(
        fire, choice.to(torch.int64), held_impulse, attack_state,
        crest_theta_vec=crest_theta_vec, crest_hold_ticks=int(crest_hold_ticks),
        hbw=hbw, live=live, ready=ready, veto_mask=veto_mask)


# ── a25 MOVE COMMITMENT decode (segment head → semi-Markov generative) ───────
#
# Replaces the fb/lr sticky-gate + hazard-table + switch-back stack for
# seg-equipped checkpoints: at segment EXPIRY (or a Gate B projectile-release
# interrupt, or episode start), sample the next (class, duration-bucket)
# commitment from the segment head's joint 30-way posterior — at expiry the
# class just held is MASKED OUT (a segment is a maximal run: expiry means
# change, the population the head was trained on); at a Gate B interrupt the
# held class stays available (re-decision, may re-commit) — then run the
# commitment open-loop.
# Dwell statistics and threat-reactivity now come from the MODEL, conditioned
# on state, instead of a context-free corpus table (research/move-head.md §8;
# acceptance gate = the closed-loop threat PSTH).

from qnn.model.bench.a25.move_seg_head import (  # noqa: E402
    FIB_EDGES, N_BUCKETS, N_CLASSES, N_AXES, JOINT)
from qnn.model.decode import gumbel_argmax  # noqa: E402

# concrete duration range per bucket (frames); tail extends one fib step
_BUCKET_LO = list(FIB_EDGES)
_BUCKET_HI = [e - 1 for e in FIB_EDGES[1:]] + [144]

# Flat carried-state width for the commitment decode in the graph. We reuse the
# a24 move_state slot (MOVE_STATE_DIM=9) so the ONNX I/O tensor names/shapes are
# UNCHANGED — only the column semantics + episode-reset init differ:
#   [0]=fb_cls [1]=fb_rem [2]=lr_cls [3]=lr_rem [4]=prev_release
#   [5]=ud_cls [6]=ud_rem  (water-ud swim commit; movearch only)  ([7:9] unused).
# cls<0 = unset (episode start) -> decode samples a fresh commitment.
COMMIT_STATE_DIM = 9
_COMMIT_RESET_LANES = (-1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)


def commit_reset_lanes() -> "list[float]":
    """The per-episode reset init for the commit state (the ``move_state`` slot),
    stamped into the ONNX state_loopback so the engine memsets it at episode start
    exactly like the a24 move_state lanes."""
    return list(_COMMIT_RESET_LANES)


def incoming_projectile_present(entity_types: torch.Tensor,
                                entity_scalars_raw: torch.Tensor) -> torch.Tensor:
    """(B,) bool — an INCOMING projectile is present: any projectile token
    farther than the own-fire gate. The shared threat definition
    (qnn.vocab.OWN_FIRE_DIST_U) — same convention as the human reactivity
    reference, the eval threat_trace bit2, and the offline attribution.

    The second argument is the (B, N, 3) projectile-relative REL in raw game
    units — callers extract it from their obs form (packed wire
    entity_scalars_raw[..., 0:3] on the export path — PROJECTILE_FIELDS start
    with rel and carry NO dist column, engine_norm computes dist at dequant;
    the field-granular eval/env obs passes entity_rel directly). distance =
    |rel|. Torch-only, trace-safe (QNNPolicy.act and the ONNX ExportWrapper)."""
    from qnn.vocab import OWN_FIRE_DIST_U, TOKEN_PROJECTILE
    pm = entity_types == TOKEN_PROJECTILE
    d = torch.linalg.norm(entity_scalars_raw[..., 0:3].float(), dim=-1)
    return (pm & (d > OWN_FIRE_DIST_U)).any(dim=-1)


def move_commit_step(
    seg_logits: torch.Tensor,        # (B, 2, 30) fb/lr joint logits
    commit_state: torch.Tensor,      # (B, 5) [fb_cls, fb_rem, lr_cls, lr_rem, prev_release]; cls<0 = unset
    *,
    release: torch.Tensor | None = None,   # (B,) bool — Gate B interrupt: resample NOW
    greedy: bool = False,
    row_generators=None,
    dur_tilt: tuple[float, float] = (0.0, 0.0),  # per-axis (fb, lr) bucket-index tilt
    water: torch.Tensor | None = None,   # (B,) bool — enables the ud (swim) axis
    threat: torch.Tensor | None = None,  # (B,) bool — incoming projectile present
    threat_break_hazard: float = 0.0,    # per-tick re-decision prob on threat rows
) -> torch.Tensor:
    """One tick of the commitment decode. Mutates ``commit_state`` in place;
    returns (B, 2) int64 fb/lr classes to emit this tick — or (B, 3) with the
    ud column when ``water`` is given and the seg head carries the water-ud
    axis (movearch shape). The ud commit runs ONLY on water rows: leaving
    water resets its lanes and emits none (the jump head owns vertical on
    land); swim-DOWN cells are decode-masked (untrained class — the label
    derivation ignores the 756-segment down population).

    greedy: argmax joint + median-of-bucket duration (deterministic).
    sampled: multinomial joint + uniform-in-bucket duration (per-row
    generators when provided, mirroring categorical_sample's contract).

    dur_tilt: censoring-bias correction — the training labels drop
    right-censored (disproportionately LONG) segments, so the duration
    posterior samples short and the closed-loop change-point rate inflates
    (~0.36 vs human 0.24, _move_cp_cooccurrence.json). Adds
    ``tilt * bucket_index`` to every class's bucket logits before sampling;
    fit per axis by moment-matching the decode-realistic expected duration
    to the human event population (_move_seg_dur_calibration.py). 0.0 = off,
    bit-identical to the uncorrected decode.
    """
    B = seg_logits.shape[0]
    dev = seg_logits.device
    lo = torch.as_tensor(_BUCKET_LO, dtype=torch.long, device=dev)
    hi = torch.as_tensor(_BUCKET_HI, dtype=torch.long, device=dev)
    rel_now = (release.reshape(B).bool() if release is not None
               else torch.zeros(B, dtype=torch.bool, device=dev))
    # RISING-EDGE interrupt: Gate B (mode "any", 1s horizon) can stay active
    # for whole engagements — re-triggering per tick would churn the stream
    # (caught by the closed-loop PSTH acceptance gate: baseline 0.42 vs human
    # 0.24). Resample only when a threat APPEARS; the new commitment then
    # holds through it (the head's threat-conditioned duration owns the WHEN).
    prev_rel = commit_state[:, 4].bool() if commit_state.shape[1] > 4 else torch.zeros_like(rel_now)
    rel = rel_now & ~prev_rel
    if commit_state.shape[1] > 4:
        commit_state[:, 4] = rel_now.to(commit_state.dtype)
    # THREAT-BREAK hazard (move.threat_break_hazard, default 0.0 = off,
    # bit-identical): while an incoming projectile is present, each tick
    # offers a RE-DECISION with probability λ — sustained and memoryless, so
    # the realized changepoint PSTH is a sustained lift (the measured human
    # signature, hazard ratio 1.143) rather than an edge spike (the retired
    # Gate B fingerprint). The head still owns the outcome: held class stays
    # available, re-committing renews duration with NO changepoint, and its
    # posterior picks the direction. λ is a decode FIT knob (stage-6 trim to
    # the human hazard reference); it bridges until a head carries the
    # conditioning natively (attribution ~1.01 vs human 1.14 — see
    # research/move-head.md "Threat reactivity"). Stochastic ⇒ inert under
    # greedy (the parity/gate path).
    brk = torch.zeros(B, dtype=torch.bool, device=dev)
    if (threat is not None and threat_break_hazard > 0.0 and not greedy):
        thr = threat.reshape(B).bool().to(dev)
        if row_generators is None:
            u_b = torch.rand(B, device=dev)
        elif isinstance(row_generators, BatchedRNG):
            # One vectorized draw instead of the per-generator loop (batched eval /
            # PPO). batch_size == B by construction (the caller sizes it to the row
            # count). Independent per-row uniforms — same law, different stream.
            u_b = row_uniforms(row_generators, 1, dev)[:, 0]
        else:
            u_b = torch.empty(B, device=dev)
            for i, g in enumerate(row_generators):
                u_b[i] = torch.rand(1, generator=g, device=dev)
        brk = thr & (u_b < float(threat_break_hazard))
    out = torch.empty(B, 2, dtype=torch.long, device=dev)
    for ai in range(2):
        cls = commit_state[:, ai * 2].long()
        rem = commit_state[:, ai * 2 + 1].long()
        need = (cls < 0) | (rem <= 0) | rel | brk
        # mask the held class's 10 buckets ONLY at EXPIRY (a maximal run ending
        # means the class changes — the population the head trained on). A
        # Gate B interrupt is a RE-DECISION, not an expiry: the held class
        # stays available, so the head's threat-conditioned posterior owns
        # whether to break the commitment; re-committing renews the duration
        # with NO changepoint. Masking on interrupt forced a double (fb+lr)
        # changepoint per threat edge — convicted as the rhythm-gate rate
        # inflation by the co-occurrence audit (_move_cp_cooccurrence.json:
        # same-tick lift 1.66 vs human 0.96, dwell tail p99 20 vs 32).
        logits = seg_logits[:, ai].clone()                       # (B, 30)
        tilt = float(dur_tilt[ai])
        if tilt != 0.0:
            logits = logits + tilt * torch.arange(
                N_BUCKETS, dtype=logits.dtype, device=dev).repeat(3)
        held = cls.clamp(min=0)
        bucket_cols = torch.arange(N_BUCKETS, device=dev)
        mask_cols = held.unsqueeze(1) * N_BUCKETS + bucket_cols  # (B, 10)
        row_has_held = ((cls >= 0) & (rem <= 0)).unsqueeze(1)
        logits.scatter_(1, mask_cols, torch.where(
            row_has_held.expand(-1, N_BUCKETS),
            torch.full_like(mask_cols, -1, dtype=logits.dtype).fill_(-1e9),
            logits.gather(1, mask_cols)))
        if greedy:
            idx = logits.argmax(dim=-1)
            new_cls = idx // N_BUCKETS
            bk = idx % N_BUCKETS
            dur = (lo[bk] + hi[bk]) // 2
        else:
            probs = torch.softmax(logits, dim=-1)
            if row_generators is None:
                idx = torch.multinomial(probs, 1).squeeze(-1)
                u = torch.rand(B, device=dev)
            elif isinstance(row_generators, BatchedRNG):
                # One vectorized (class, duration) draw instead of the per-row
                # multinomial+rand loop: inverse-CDF categorical over batched
                # uniforms (col 0) + a second uniform (col 1) for the in-bucket
                # duration. Same law as the per-row multinomial, one dispatch.
                _uu = row_uniforms(row_generators, 2, dev)          # (B, 2)
                idx = inverse_cdf_sample(probs, _uu[:, 0])
                u = _uu[:, 1]
            else:
                idx = torch.empty(B, dtype=torch.long, device=dev)
                u = torch.empty(B, device=dev)
                for i, g in enumerate(row_generators):
                    idx[i] = torch.multinomial(probs[i:i + 1], 1, generator=g).squeeze()
                    u[i] = torch.rand(1, generator=g, device=dev)
            new_cls = idx // N_BUCKETS
            bk = idx % N_BUCKETS
            span = (hi[bk] - lo[bk] + 1).to(torch.float32)
            dur = lo[bk] + (u * span).long().clamp(max=(hi[bk] - lo[bk]))
        emit = torch.where(need, new_cls, cls)
        new_rem = torch.where(need, dur, rem) - 1
        commit_state[:, ai * 2] = emit.to(commit_state.dtype)
        commit_state[:, ai * 2 + 1] = new_rem.to(commit_state.dtype)
        out[:, ai] = emit
    if water is None or seg_logits.shape[1] < 3:
        return out
    if commit_state.shape[1] < 7:
        raise ValueError(
            "water-ud commit needs commit_state lanes [5]=ud_cls [6]=ud_rem "
            f"(dim >= 7); got {commit_state.shape[1]} — allocate "
            "COMMIT_STATE_DIM lanes for movearch models")
    w = water.reshape(B).bool().to(dev)
    cls = commit_state[:, 5].long()
    rem = commit_state[:, 6].long()
    need = ((cls < 0) | (rem <= 0)) & w
    logits = seg_logits[:, 2].clone()                        # (B, 30)
    # Down (class 0) is never emitted: its labels are ignored in training,
    # so its cells are untrained mass. In QW you sink by not swimming up.
    logits[:, :N_BUCKETS] = -1e9
    held = cls.clamp(min=0)
    bucket_cols = torch.arange(N_BUCKETS, device=dev)
    mask_cols = held.unsqueeze(1) * N_BUCKETS + bucket_cols
    row_has_held = ((cls >= 0) & (rem <= 0)).unsqueeze(1)
    logits.scatter_(1, mask_cols, torch.where(
        row_has_held.expand(-1, N_BUCKETS),
        torch.full_like(mask_cols, -1, dtype=logits.dtype).fill_(-1e9),
        logits.gather(1, mask_cols)))
    if greedy:
        idx = logits.argmax(dim=-1)
        new_cls = idx // N_BUCKETS
        bk = idx % N_BUCKETS
        dur = (lo[bk] + hi[bk]) // 2
    else:
        probs = torch.softmax(logits, dim=-1)
        if row_generators is None:
            idx = torch.multinomial(probs, 1).squeeze(-1)
            u = torch.rand(B, device=dev)
        elif isinstance(row_generators, BatchedRNG):
            # vectorized (class, duration) draw — same law as the per-row
            # multinomial+rand loop, one dispatch (mirrors the fb/lr branch above).
            _uu = row_uniforms(row_generators, 2, dev)          # (B, 2)
            idx = inverse_cdf_sample(probs, _uu[:, 0])
            u = _uu[:, 1]
        else:
            idx = torch.empty(B, dtype=torch.long, device=dev)
            u = torch.empty(B, device=dev)
            for i, g in enumerate(row_generators):
                idx[i] = torch.multinomial(probs[i:i + 1], 1, generator=g).squeeze()
                u[i] = torch.rand(1, generator=g, device=dev)
        new_cls = idx // N_BUCKETS
        bk = idx % N_BUCKETS
        span = (hi[bk] - lo[bk] + 1).to(torch.float32)
        dur = lo[bk] + (u * span).long().clamp(max=(hi[bk] - lo[bk]))
    emit = torch.where(need, new_cls, cls)
    new_rem = torch.where(need, dur, rem) - 1
    # Non-water rows: reset lanes, emit none. The commit never bridges a
    # water exit — matching the label derivation's censoring at the boundary.
    none_t = torch.ones_like(emit)
    emit = torch.where(w, emit.clamp(min=1), none_t)
    new_rem = torch.where(w, new_rem, torch.zeros_like(new_rem))
    lane_cls = torch.where(w, emit, torch.full_like(emit, -1))
    commit_state[:, 5] = lane_cls.to(commit_state.dtype)
    commit_state[:, 6] = new_rem.to(commit_state.dtype)
    out3 = torch.empty(B, 3, dtype=torch.long, device=dev)
    out3[:, :2] = out
    out3[:, 2] = emit
    return out3


def move_commit_step_graph(
    seg_logits: torch.Tensor,        # (B, 2, JOINT) fb/lr joint logits
    commit_state: torch.Tensor,      # (B, COMMIT_STATE_DIM) float32 — flat carried state
    move_state_rng: torch.Tensor,    # (B,) int64 — passed through (categorical uses ORT RNG)
    *,
    greedy: bool,
    dur_tilt: "tuple[float, float]" = (0.0, 0.0),
    release: torch.Tensor | None = None,   # (B,) bool — Gate B interrupt, or None (off)
    water: torch.Tensor | None = None,      # (B,) bool — movearch: deep-water rows
    jump_logit: torch.Tensor | None = None, # (B,) — movearch: jump head logit
    threat: torch.Tensor | None = None,     # (B,) bool — incoming projectile present
    threat_break_hazard: float = 0.0,       # per-tick re-decision prob on threat rows
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """TRACE-SAFE in-graph twin of :func:`move_commit_step` (fb/lr only) — the
    DECODE LAW baked into the ONNX so deploy == the QNNPolicy.act commitment decode.

    movearch (build-time static: ``water`` + ``jump_logit`` given): returns
    (B, 3) with the ud column composed exactly like QNNPolicy.act — the jump
    head's engine-OUTCOME posterior sampled AS-IS on every non-water row (no
    feasibility gate: air/hold/debounce are priced into the outcome labels
    and the engine enforces the debounce mechanically), water-ud commit
    (lanes [5]/[6], down-masked, reset on water exit) on water rows. Legacy
    (None) path is bit-identical to before — lanes 5..8 stay zeros.

    Returns ``(fblr_classes, commit_state_out, move_state_rng_out)``:
      fblr_classes    (B, 2) int64 — the emitted fb/lr classes this tick.
      commit_state_out (B, COMMIT_STATE_DIM) float32 — updated flat state (rebuilt
                       by torch.stack; NO in-place slice assignment, which the
                       legacy ONNX exporter traces poorly — mirrors move_decode_step_graph).
      move_state_rng_out (B,) int64 — threaded through unchanged. The categorical +
                       in-bucket draws use ORT ``RandomUniformLike`` (gumbel-max, like
                       the a24 jump sample), NOT the xorshift, so the rng is inert
                       here; kept in the signature for wire/state parity with a24.

    ``greedy`` is a PYTHON bool (graph-build-time), baked as a taken branch:
      greedy   → argmax joint + median-of-bucket duration (deterministic; the gate path).
      sampled  → gumbel-max joint + uniform-in-bucket (the deployed stochastic stream).

    Bit-for-bit with the eager greedy law: same tilt, same expiry-only held-class
    mask, same //2 median, same emit/rem update.
    """
    dev = seg_logits.device
    logits_all = seg_logits.to(torch.float32)
    logits_all = logits_all.reshape(logits_all.shape[0], -1, JOINT)  # (B, 2|3, JOINT)
    B = logits_all.shape[0]
    lo = torch.as_tensor(_BUCKET_LO, dtype=torch.long, device=dev)
    hi = torch.as_tensor(_BUCKET_HI, dtype=torch.long, device=dev)
    # per-joint-column class id (constant (JOINT,)) — used to build the expiry mask
    # additively (no scatter_, which traces poorly): col c belongs to class c//N_BUCKETS.
    col_class = (torch.arange(JOINT, device=dev) // N_BUCKETS).reshape(1, JOINT)  # (1,JOINT)
    tilt_ramp = torch.arange(N_BUCKETS, dtype=torch.float32, device=dev).repeat(N_CLASSES)  # (JOINT,)

    rel_now = (release.reshape(B).bool() if release is not None
               else torch.zeros(B, dtype=torch.bool, device=dev))
    prev_rel = commit_state[:, 4].round().to(torch.bool)
    rel = rel_now & (~prev_rel)                      # RISING-EDGE interrupt (matches eager)
    # THREAT-BREAK hazard — the sustained per-tick re-decision law (matches
    # eager; see move_commit_step). Stochastic via torch.rand → ORT
    # RandomUniformLike (the twin's sampling convention); inert under greedy
    # (build-time static branch, so the parity/gate graph is unchanged).
    brk = torch.zeros(B, dtype=torch.bool, device=dev)
    if threat is not None and threat_break_hazard > 0.0 and not greedy:
        u_b = torch.rand(B, device=dev)
        brk = threat.reshape(B).to(torch.bool) & (u_b < float(threat_break_hazard))

    emit_cols = [None, None]
    rem_cols = [None, None]
    for ai in range(N_AXES):
        cls = commit_state[:, ai * 2].round().to(torch.int64)       # (B,)
        rem = commit_state[:, ai * 2 + 1].round().to(torch.int64)   # (B,)
        need = (cls < 0) | (rem <= 0) | rel | brk                   # (B,)
        logits = logits_all[:, ai, :]                               # (B,JOINT)
        tilt = float(dur_tilt[ai])
        if tilt != 0.0:
            logits = logits + tilt * tilt_ramp
        # expiry-only held-class mask: -1e9 on the held class's buckets when a
        # completed segment is ending (cls>=0 & rem<=0). Interrupt is a
        # re-decision (held stays available) — matches the eager scatter law.
        held = cls.clamp(min=0).reshape(B, 1)                       # (B,1)
        row_has_held = ((cls >= 0) & (rem <= 0)).reshape(B, 1)      # (B,1)
        is_held_col = (col_class == held)                          # (B,JOINT) bool
        penalty = torch.where(row_has_held & is_held_col,
                              torch.full((), -1.0e9, device=dev, dtype=logits.dtype),
                              torch.zeros((), device=dev, dtype=logits.dtype))
        logits = logits + penalty
        if greedy:
            idx = logits.argmax(dim=-1)                             # (B,)
        else:
            idx = gumbel_argmax(logits)                            # (B,) ORT RandomUniformLike
        new_cls = torch.div(idx, N_BUCKETS, rounding_mode="floor")  # (B,)
        bk = idx - new_cls * N_BUCKETS                              # (B,) bucket index
        lo_bk = lo.index_select(0, bk)
        hi_bk = hi.index_select(0, bk)
        if greedy:
            dur = torch.div(lo_bk + hi_bk, 2, rounding_mode="floor")
        else:
            u = torch.rand(B, device=dev).clamp_(0.0, 1.0 - 1e-9)  # ORT RandomUniformLike
            span = (hi_bk - lo_bk + 1).to(torch.float32)
            dur = lo_bk + torch.minimum((u * span).to(torch.long), hi_bk - lo_bk)
        emit = torch.where(need, new_cls, cls)                      # (B,)
        new_rem = torch.where(need, dur, rem) - 1                   # (B,)
        emit_cols[ai] = emit
        rem_cols[ai] = new_rem

    fblr = torch.stack([emit_cols[0], emit_cols[1]], dim=-1).to(torch.int64)  # (B,2)
    zeros = torch.zeros(B, dtype=torch.float32, device=dev)
    if water is None or jump_logit is None:
        # Legacy (fb/lr-only) graph: lanes 5..8 carried as 0 — bit-identical.
        cols = [
            emit_cols[0].to(torch.float32), rem_cols[0].to(torch.float32),
            emit_cols[1].to(torch.float32), rem_cols[1].to(torch.float32),
            rel_now.to(torch.float32),
            zeros, zeros, zeros, zeros,
        ]
        commit_state_out = torch.stack(cols, dim=-1)               # (B, COMMIT_STATE_DIM)
        return fblr, commit_state_out, move_state_rng.to(torch.int64)

    # ── movearch ud composition (twin of the QNNPolicy.act block) ──────────
    w = water.reshape(B).to(torch.bool)
    ud_cls = commit_state[:, 5].round().to(torch.int64)
    ud_rem = commit_state[:, 6].round().to(torch.int64)
    ud_logits = logits_all[:, 2, :]
    # down (class 0) decode-masked: untrained cells (labels ignored) — additive,
    # no scatter_ (trace-safe).
    down_pen = torch.where(col_class < 1,
                           torch.full((), -1.0e9, device=dev, dtype=ud_logits.dtype),
                           torch.zeros((), device=dev, dtype=ud_logits.dtype))
    ud_logits = ud_logits + down_pen
    held_u = ud_cls.clamp(min=0).reshape(B, 1)
    row_held_u = ((ud_cls >= 0) & (ud_rem <= 0)).reshape(B, 1)
    pen_u = torch.where(row_held_u & (col_class == held_u),
                        torch.full((), -1.0e9, device=dev, dtype=ud_logits.dtype),
                        torch.zeros((), device=dev, dtype=ud_logits.dtype))
    ud_logits = ud_logits + pen_u
    if greedy:
        idx_u = ud_logits.argmax(dim=-1)
    else:
        idx_u = gumbel_argmax(ud_logits)
    new_cls_u = torch.div(idx_u, N_BUCKETS, rounding_mode="floor")
    bk_u = idx_u - new_cls_u * N_BUCKETS
    lo_u = lo.index_select(0, bk_u)
    hi_u = hi.index_select(0, bk_u)
    if greedy:
        dur_u = torch.div(lo_u + hi_u, 2, rounding_mode="floor")
    else:
        u_u = torch.rand(B, device=dev).clamp_(0.0, 1.0 - 1e-9)
        span_u = (hi_u - lo_u + 1).to(torch.float32)
        dur_u = lo_u + torch.minimum((u_u * span_u).to(torch.long), hi_u - lo_u)
    need_u = ((ud_cls < 0) | (ud_rem <= 0)) & w
    emit_u = torch.where(need_u, new_cls_u, ud_cls)
    rem_u = torch.where(need_u, dur_u, ud_rem) - 1
    one_u = torch.ones_like(emit_u)
    emit_u = torch.where(w, emit_u.clamp(min=1), one_u)
    rem_u = torch.where(w, rem_u, torch.zeros_like(rem_u))
    lane_cls_u = torch.where(w, emit_u, torch.full_like(emit_u, -1))

    # jump: the engine-outcome posterior sampled AS-IS on every land row.
    p_jump = torch.sigmoid(jump_logit.reshape(B).to(torch.float32))
    if greedy:
        fire_j = p_jump > 0.5
    else:
        fire_j = torch.rand(B, device=dev) < p_jump
    ud_out = torch.ones(B, dtype=torch.int64, device=dev)
    ud_out = torch.where(fire_j, torch.full_like(ud_out, 2), ud_out)
    ud_out = torch.where(w, emit_u, ud_out)

    cols = [
        emit_cols[0].to(torch.float32), rem_cols[0].to(torch.float32),
        emit_cols[1].to(torch.float32), rem_cols[1].to(torch.float32),
        rel_now.to(torch.float32),
        lane_cls_u.to(torch.float32), rem_u.to(torch.float32),
        zeros, zeros,
    ]
    commit_state_out = torch.stack(cols, dim=-1)                   # (B, COMMIT_STATE_DIM)
    out3 = torch.stack([emit_cols[0], emit_cols[1], ud_out], dim=-1).to(torch.int64)
    return out3, commit_state_out, move_state_rng.to(torch.int64)


# ── a25 LOOK / AIM-PRIOR / WEAPON-BAN decode (a25-owned clones) ───────────────
#
# BIT-IDENTICAL clones of the reachable a24 decode geometry so the a25 arch never
# imports/executes a24 code (cross-arch decode-coupling ban). These are the a24
# functions the a25 seg+attack_with policy/export actually run:
#   * assemble_aim_prior / assemble_pitch_correction / decode_look_from_polar —
#     the shared look decode (sampled magnitude × continuous direction + aim-prior
#     blend + feet-aim pitch + expmap).
#   * build_weapon_ban_tensors — the weapon-ban tensor constructor (executed at
#     export construction; empty ban → (None, None)).
# The a24 move sticky/hazard/switch-back stack, the split-attack decode step, and
# decide_weapon_sticky are NOT cloned — the a25 move commitment (above) + the
# attack_with joint decode replace them and never call them.
#
# TRACE-SAFETY: called from ExportWrapper.forward (torch.onnx traced). No .item(),
# no python `if` on tensor values, no advanced/boolean indexing (index_select used).


def assemble_aim_prior(
    z_err: torch.Tensor,
    z_rate: torch.Tensor,
    gain: float,
    ffwd: float,
) -> torch.Tensor:
    """Assemble the PRE-SCALED aim-prior blend term ``z_prior``.

    ``z_prior = gain·z_err + ffwd·z_rate``. ``gain`` is the human-percentile placement
    knob (rotation steering); ``ffwd`` is the rate feed-forward.
    """
    return gain * z_err + ffwd * z_rate


def assemble_pitch_correction(
    z_err: torch.Tensor,
    weapon_pitch_gain: torch.Tensor,
    self_weapon_id: torch.Tensor,
    weapon_pitch_bias: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per-row VERTICAL feet-aim BLEND terms — the RL-splash feet-aiming.

    ``z_err`` is the UNSCALED aim-prior error tangent (vertical component
    ``z_err[..., 1]`` is the absolute turn-to-anchor pitch, + = up / − = down).
    ``weapon_pitch_gain`` is a (9,) per-IMPULSE blend weight β ∈ [0, 1] (0 = OFF).
    ``self_weapon_id`` is ENTITY_IDS-encoded, mapped to impulse via the canonical helper.

    Returns ``(beta, target_vert, feet_vert)``, consumed by :func:`decode_look_from_polar`
    as the LERP ``z_vert ← (1−β)·z_head_vert + β·target_vert``. β is enemy-gated
    (``z_err`` is exactly zero rows otherwise). ``weapon_pitch_bias`` (per-IMPULSE deg,
    default None) deepens the target to cancel the static RL fire-high offset.
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
        target_vert = target_vert - bias_deg * 0.017453292519943295     # + = aim DOWN
    # presence gate: aim_prior_tangent_ffwd zeroes BOTH components on no-enemy frames,
    # so a nonzero L1 means an enemy is perceived. Zeroing β there keeps the lerp a
    # no-op (z_vert unchanged) instead of shrinking the head's vertical.
    present = (z.abs().sum(dim=-1) > 1e-9).to(beta.dtype)               # (R,)
    return beta * present, target_vert, feet_vert


def decode_look_from_polar(
    mag_bin: torch.Tensor,
    dir_logits: torch.Tensor,
    mag_centers: torch.Tensor,
    dir_centers: torch.Tensor,
    z_prior: torch.Tensor | None = None,
    mag_gain: float = 0.0,
    turn_mag_scale: "float | torch.Tensor" = 1.0,
    hold_drift_eps: float = 0.0,
    hold_passthrough: bool = False,
    pitch_correction: "tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None" = None,
    feet_elev: "torch.Tensor | None" = None,
    origin_elev: "torch.Tensor | None" = None,
    pitch_mode: str = "lock",
    shift_strength: float = 1.0,
) -> torch.Tensor:
    """(sampled ``mag_bin``, ``dir_logits``) → look unit vector (hybrid decode).

    ``turn_mag_scale`` (default 1.0 = OFF = bit-identical) multiplicatively dampens the
    head's NATIVE turn magnitude ``|z| = θ`` BEFORE the aim-prior blend. The caller
    supplies ``mag_bin`` from ITS OWN magnitude sampler (seeded categorical offline /
    in-graph Gumbel-argmax for export). This function owns the shared remainder: the
    continuous DIRECTION readout (circular mean of the direction softmax), the
    z-assembly, the aim-prior blend, the feet-aim pitch, and the expmap.

    Returns the look unit vector with shape ``mag_bin.shape + (3,)``.
    """
    # polar_to_tangent via index_select (export-friendly vs advanced index).
    theta = mag_centers.index_select(0, mag_bin.reshape(-1)).reshape(mag_bin.shape)
    # Head turn-magnitude dampener (turn_mag_scale, default 1.0 = no-op).
    # A per-ROW tensor (the aim-grid per-lane override) dampens each lane's θ
    # independently; the scalar/export path is unchanged (bit-identical).
    if torch.is_tensor(turn_mag_scale):
        theta = theta * turn_mag_scale.reshape(theta.shape).to(theta.dtype)
    elif turn_mag_scale != 1.0:
        theta = theta * float(turn_mag_scale)
    # Continuous direction: circular mean of the direction distribution.
    p_dir = torch.softmax(dir_logits, dim=-1)                       # (..., N_DIR)
    cos_phi = (p_dir * torch.cos(dir_centers)).sum(dim=-1)
    sin_phi = (p_dir * torch.sin(dir_centers)).sum(dim=-1)
    phi = torch.atan2(sin_phi, cos_phi)                             # (...,)
    z = torch.stack([theta * torch.cos(phi), theta * torch.sin(phi)], dim=-1)
    if hold_drift_eps > 0.0:
        # HOLD-DRIFT (default 0.0 = OFF = bit-identical): replace exact holds with an
        # eps-magnitude turn along the head's continuous direction φ; the aim-prior
        # blend below rotates the eps onto the true error direction when present.
        is_hold = (theta == 0).unsqueeze(-1)
        drift = float(hold_drift_eps) * torch.stack(
            [torch.cos(phi), torch.sin(phi)], dim=-1)
        z = torch.where(is_hold, drift, z)
    if z_prior is not None:
        # AIM-PRIOR blend: direction is the rotated heading normalize(z + z_prior); the
        # kept MAGNITUDE blends from |z|=θ (mag_gain=0, PURE ROTATION) toward |z+z_prior|
        # (mag_gain=1, legacy vector-ADD), mag_gain>1 overshoots into super-human slew.
        # z_prior is zero on no-enemy frames (reduces to the plain hybrid). PRE-SCALED
        # by assemble_aim_prior.
        z_t = z + z_prior
        mag = torch.linalg.vector_norm(z, dim=-1, keepdim=True)        # head |z| = θ
        n = torch.linalg.vector_norm(z_t, dim=-1, keepdim=True)
        if torch.is_tensor(mag_gain):
            # per-row α (the in-graph skill input): rows at 0 are exact no-ops. mag/n
            # are (R,1) → α must be (R,1) too else the broadcast explodes.
            mag = mag + mag_gain.reshape(-1, 1) * (n - mag)
        elif mag_gain != 0.0:
            mag = mag + float(mag_gain) * (n - mag)                    # θ → |z+z_prior|
        # where z_t collapses to ~0 (tiny head turn + no aim), keep z unrotated.
        z = torch.where(n > 1e-9, mag * z_t / n.clamp_min(1e-9), z)
    if pitch_correction is not None:
        # VERTICAL feet-aim BLEND (RL-splash): lerp the decoded turn's vertical toward
        # the absolute geometric anchor, OUTSIDE the rotation-magnitude clamp. β is
        # enemy-gated, so no-target frames are an exact no-op. Default None = no-op.
        # FLOOR CLAMP (feet_vert): never aim below the feet at range. SHIFT mode SKIPS
        # the tangent β-blend (it would collapse the head's vertical spread).
        if pitch_mode != "shift":
            beta, target_vert, feet_vert = pitch_correction
            beta = beta.reshape(z[..., 1].shape)
            target_vert = target_vert.reshape(z[..., 1].shape)
            feet_vert = feet_vert.reshape(z[..., 1].shape)
            z_vert = z[..., 1] + beta * (target_vert - z[..., 1])   # blend toward biased drive
            gate = (beta > 0).to(z_vert.dtype)                      # RL + enemy present
            z_vert = torch.maximum(z_vert, feet_vert) * gate + z_vert * (1.0 - gate)
            z = torch.stack([z[..., 0], z_vert], dim=-1)
    if hold_passthrough:
        # HOLD PASS-THROUGH (default False = bit-identical): frames where the head
        # commanded an exact hold (θ == 0) emit an exact hold — the aim-prior
        # magnitude blend above otherwise converts every engaged hold into an
        # α·|aim-error| micro-correction, driving measured exact-hold occupancy to
        # zero (humans: ~14% of engaged frames). Turning frames are untouched.
        # Mutually exclusive with hold_drift_eps in spirit (drift REPLACES holds;
        # pass-through PRESERVES them) — pass-through wins on the θ==0 rows.
        _hold_rows = (theta == 0).unsqueeze(-1)
        z = torch.where(_hold_rows, torch.zeros_like(z), z)
    look_vec = tangent_expmap(z)                                    # (..., 3) fwd,right,up
    if pitch_correction is not None and feet_elev is not None:
        # POST-EXPMAP FEET-ELEVATION LOCK (RL splash robustness). Set the fired look's
        # ELEVATION to the feet anchor (feet_elev) keeping AZIMUTH, AFTER the turn.
        # Gated to β>0 (RL + enemy present); feet_elev is 0 on no-enemy frames.
        beta_g = pitch_correction[0].reshape(look_vec[..., 0].shape)
        gate_e = (beta_g > 0).to(look_vec.dtype)
        if hold_passthrough:
            # elevation lock would re-pitch pass-through holds; keep them exact
            gate_e = gate_e * (theta != 0).to(look_vec.dtype).reshape(gate_e.shape)
        fe = feet_elev.reshape(look_vec[..., 0].shape)
        h = torch.linalg.vector_norm(look_vec[..., :2], dim=-1).clamp_min(1e-6)
        # Target elevation to set (keeping azimuth). "lock" → feet_elev exactly;
        # "shift" → translate the head's OWN fired elevation DOWN by
        # shift_strength·(origin_elev − feet_elev), preserving the head's spread.
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

    ``ban_mask`` is an additive pre-softmax mask (``-1e9`` on each banned class);
    ``ban_bool`` is a per-class banned flag. Impulses 1..8 map to classes 0..7
    (``class = impulse - 1``). Empty ban → ``(None, None)`` (run UNBANNED).
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


# ── a25 recurrent-state reset scaffolding (graph-I/O; a25-owned) ──────────────
# The a25 wire reuses the a24 move_state slot (the commit layout occupies the same 9
# lanes) and the attack_state/attack_rng slots. These reset helpers give the ONNX
# export/parity scaffolding a25-native initial state so the LIVE a25 export executes
# no a24 code. The rng seed literals ARE the wire/state contract (mirrored in
# src/engine/common/qnn_onnx.c).
# attack_state lane 0 = the crest-gate countdown (attack_crest_gate_step); 0 = idle,
# so the existing zeros-init/episode-reset loopback semantics are unchanged. With
# the gate OFF the slot passes through untouched (wire parity, bit-identical).
ATTACK_STATE_DIM = 1
_RNG_DEFAULT_SEED = 0x9E3779B9          # xorshift reseed-on-zero constant
_RNG_U32 = 0xFFFFFFFF


def move_decode_reset_flat(
    batch: int = 1,
    rng_state: "int | torch.Tensor" = _RNG_DEFAULT_SEED,
    device: "torch.device | str | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Flat (graph-I/O) a25 move-commit reset: ``(move_state (B,9), move_state_rng (B,))``.

    The move_state slot carries the commit layout ([fb_cls,fb_rem,lr_cls,lr_rem,
    prev_release, 0,0,0,0]) initialized to cls<0 = unset (episode start → resample),
    matching :func:`commit_reset_lanes` (the loopback init the engine memsets). The rng
    is threaded through the commitment decode inert (categorical uses ORT RNG) but kept
    for wire/state parity with a24.
    """
    s = torch.tensor(_COMMIT_RESET_LANES, dtype=torch.float32, device=device
                     ).unsqueeze(0).repeat(batch, 1)
    rng = torch.full((batch,), int(rng_state) & _RNG_U32, dtype=torch.int64, device=device)
    return s, rng


def attack_decode_reset_flat(
    batch: int = 1,
    rng_state: "int | torch.Tensor" = _RNG_DEFAULT_SEED,
    device: "torch.device | str | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Flat (graph-I/O) a25 attack-state reset: ``(B, ATTACK_STATE_DIM)`` zeros + the
    ``(B,)`` int64 attack_rng. Zeros = the crest latch idle (lane 0 countdown = 0);
    with the gate OFF the slots pass through unchanged (wire parity)."""
    s = torch.zeros(batch, ATTACK_STATE_DIM, dtype=torch.float32, device=device)
    rng = torch.full((batch,), int(rng_state) & _RNG_U32, dtype=torch.int64, device=device)
    return s, rng
