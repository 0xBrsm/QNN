"""a25 attack-with decode — the joint 9-way head's deploy protocol.

One trace-safe function shared by ``QNNPolicy.act`` and the ONNX export
wrapper (decode-regime-in-model: the engine consumes the FINAL fire bit +
weapon impulse; no engine-side decode). Greedy by construction — the
deterministic commit preserves coordinated actions (rocket jump), and the
offline shape analysis showed raw argmax self-calibrates to ~the human fire
rate (runs/bc/bench/_attack_with_shape.json).

Decode law (per frame) — two explicitly parameterised decisions:

    select_score = (l1..l8) + bias_vec + preference_bias_vec
    ideal  = argmax(select_score)                  # best weapon this frame
    choice = hysteresis(select_score, held, switch_margin)
    fire_score = (l1..l8)[choice] + bias_vec[choice] + fire_bias_vec[choice]
    # attack decision (whether) — conditional on the weapon that will fire
    l0'    = l0 - attack_bias - align_bias         # class-0 = "don't attack" anchor
    fire   = fire_score > l0'   AND NOT veto
    weapon = fire ? choice+1 : held_impulse        # held = server no-op

Knob mapping:
  * ``attack_bias`` (the s-slider / propensity knob) and the alignment-
    conditioned bias act on the MARGINAL log-odds by subtracting from the
    class-0 logit (bias > 0 → more attack).
  * ``bias_vec`` (8,) is the LEGACY JOINT operating point. It is retained so
    existing a26 configs keep their exact meaning: it affects both selection
    and firing. New fits emit it as all-zero and use the two explicit vectors
    below; silently reinterpreting an old config would make branch artifacts
    non-reproducible.
  * ``fire_bias_vec`` (8,) affects only the fire comparison for the selected
    weapon. It cannot change the ideal or clear a switch margin.
  * ``preference_bias_vec`` (8,) affects only weapon selection. It is the
    closed-loop occupancy-restoration lever used after adding hysteresis; it
    does not directly lift a weapon over the no-attack class.
  * ``switch_margin`` is weapon-switch hysteresis (anti-jitter / anti-camp):
    leave the held weapon only when the ideal beats it by the margin. Under the
    split-v1 law the fire test is conditional on that selected weapon, so every
    per-weapon fire correction is identifiable from the emitted action stream.
    The final closed-loop gate fits switching, occupancy, and firing jointly.
    Feasibility rides the score (dead held scores low → forced switch).
  * hard guards (attack-splash / RL self-splash / LG range) arrive as a
    boolean veto mask from the a24 guard primitives instead of a buried
    logit.
  * ``crest_theta_vec`` / ``crest_hold_ticks`` — the DISCHARGE-QUALITY gate
    ("crest-firing"): a deterministic countdown latch that shifts WHEN a
    commanded fire lands within a bounded window (≤ H ticks) without changing
    the count — hold until crosshair→lead alignment crosses θ_w, blind-fire at
    expiry. A hold only continues while alignment is CONVERGING (the
    feed-forward rate predicts the next tick's hbw improving); the moment it's
    predicted to worsen instead, the gate releases immediately rather than
    waiting out a hold that isn't going to pay off — same check at arm time
    (never start a hold that's already diverging) and every hold tick
    (bail the instant it turns). Composes AFTER the step (guards outrank);
    OFF = bit-identical. See attack_crest_gate_step +
    agents/plans/discharge-quality-gate.md.
  * the a24 sticky gate / hazard / switch-back weapon machinery has no analog —
    it is retired for this head; selection happens only at attack frames and
    emitting the held impulse otherwise is a server no-op. ``switch_margin`` is a
    single-scalar switch hysteresis, NOT a revival of that stack.

TRACE-SAFETY: torch-only, no ``.item()``, no data-dependent control flow
(``switch_margin``/vector presence is decided on static python config, not tensor
values).
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
# qnn.model.lead_aim (cloned from a24 so a25 executes no a24 code); this is
# the facade seam, not a move.
from qnn.model.lead_aim import (  # noqa: F401
    AIM_FFWD_GAIN, AIM_PRIOR_GAIN, _TICK_DT_MODULE, aim_prior_tangent_ffwd,
)

ATTACK_WITH_SIZE = 1 + WEAPON_HEAD_SIZE  # 9


def _attack_with_select(
    score: torch.Tensor,
    held: torch.Tensor,
    switch_margin: float,
) -> torch.Tensor:
    """Weapon SELECTION with switch hysteresis (anti-jitter / anti-camp).

    Switch off the currently-held weapon only when the ideal weapon (argmax of
    ``score`` = weapon logits + per-weapon bias_vec) beats the held weapon by
    more than ``switch_margin``; otherwise stay on held. This replaces the a25
    ``stick_bias`` (which biased selection toward held AND perturbed the fire
    decision, causing weapon camping): the margin only gates *which* weapon and
    cannot change fire rate. Feasibility rides the score (an infeasible held
    scores low → the margin is cleared → forced switch off a dead weapon).

    The SINGLE selection law shared by :func:`attack_with_decode_step` and the
    crest gate's per-tick θ re-read. ``switch_margin`` = 0 → always take the
    ideal (no hysteresis). Returns the (B,) class index 0..7 (impulse − 1)."""
    n = score.shape[-1]
    held_idx = (held.reshape(score.shape[:-1]).to(torch.long) - 1).clamp(0, n - 1)
    w_star = score.argmax(dim=-1)
    s_star = score.gather(-1, w_star.unsqueeze(-1)).squeeze(-1)
    s_held = score.gather(-1, held_idx.unsqueeze(-1)).squeeze(-1)
    switch = (s_star - s_held) > float(switch_margin)
    return torch.where(switch, w_star, held_idx)


def attack_with_decode_step(
    logits9: torch.Tensor,
    held_impulse: torch.Tensor,
    *,
    attack_bias: float = 0.0,
    bias_vec: torch.Tensor | None = None,
    fire_bias_vec: torch.Tensor | None = None,
    preference_bias_vec: torch.Tensor | None = None,
    switch_margin: float = 0.0,
    align_bias: torch.Tensor | None = None,
    veto_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy attack-with decode.

    Two DECOUPLED decisions (the a25 ``stick_bias`` coupling — which perturbed
    the fire decision and caused weapon camping — is retired):

      1. WEAPON — switch off the held weapon only if the ideal beats it by
         ``switch_margin`` (:func:`_attack_with_select`); else select held.
      2. ATTACK — under the explicit split-v1 vectors, fire iff the SELECTED
         weapon's raw+legacy+fire score beats no-attack. This makes each
         weapon's rate correction observable on the weapon that actually
         fires. Legacy configs (both explicit vectors absent) retain the old
         ideal-weapon fire comparison exactly.

    Parameters
    ----------
    logits9 : (B, 9) raw head logits (class 0 = no-attack, 1..8 = attack weapon k)
    held_impulse : (B,) int64 engine impulse 1..8 of the currently held weapon.
    attack_bias : global propensity knob (positive = more attack)
    bias_vec : optional legacy (8,) JOINT operating point, added to both the
        selection and fire scores. Existing configs retain their exact law.
    fire_bias_vec : optional (8,) fire-only operating point. It is gathered at
        the ideal selected weapon and cannot affect selection.
    preference_bias_vec : optional (8,) selection-only operating point. It
        cannot directly affect the no-attack comparison.
    switch_margin : weapon-switch hysteresis (>= 0) — leave held only if the
        ideal weapon's score exceeds held's by this margin. 0.0 = always ideal.
    align_bias : optional (B,) additive alignment bias (positive = more attack)
    veto_mask : optional (B,) bool — hard guard veto rows (True = never attack)

    Returns
    -------
    (attack, weapon_impulse) : (B,) int64 attack bit, (B,) int64 impulse 1..8
    """
    weap = logits9[..., 1:]                                  # (B, 8) raw weapon logits
    held = held_impulse.reshape(weap.shape[:-1]).to(torch.long)  # impulse 1..8
    legacy = (torch.zeros_like(weap) if bias_vec is None
              else bias_vec.reshape(-1).to(weap.dtype))
    select_score = weap + legacy
    if preference_bias_vec is not None:
        select_score = select_score + preference_bias_vec.reshape(-1).to(weap.dtype)
    # ── weapon selection ───────────────────────────────────────────────────
    ideal_idx = select_score.argmax(dim=-1)
    sel_idx = _attack_with_select(select_score, held, switch_margin)
    # Explicit vectors opt into split-v1: condition fire on the weapon that
    # will actually be emitted. With both absent, preserve the historical a26
    # law byte-for-byte (fire on ideal even when hysteresis holds another).
    split_v1 = fire_bias_vec is not None or preference_bias_vec is not None
    fire_idx = sel_idx if split_v1 else ideal_idx
    # ── attack decision ────────────────────────────────────────────────────
    fire_score = weap + legacy
    if fire_bias_vec is not None:
        fire_score = fire_score + fire_bias_vec.reshape(-1).to(weap.dtype)
    score_ideal = fire_score.gather(-1, fire_idx.unsqueeze(-1)).squeeze(-1)
    l0 = logits9[..., 0] - float(attack_bias)
    if align_bias is not None:
        l0 = l0 - align_bias.reshape(l0.shape).to(l0.dtype)
    fire = score_ideal > l0
    if veto_mask is not None:
        fire = fire & ~veto_mask.reshape(fire.shape)
    choice = sel_idx.to(held.dtype) + 1                      # impulse 1..8
    weapon_impulse = torch.where(fire, choice, held)
    return fire.to(torch.int64), weapon_impulse.to(torch.int64)


def attack_lane_gate(fire: torch.Tensor, weapon_impulse: torch.Tensor) -> torch.Tensor:
    """The A27 single-lane ACTION convention: ``attack WITH weapon X this tick``.

    ``(B,)`` int64, 0 = no attack, 1..8 = press attack while selecting that
    impulse. The A27 pure-combat wire has NO separate weapon slot, so the engine
    derives BOTH the attack button and the weapon impulse from this one value
    (``qnn_onnx.c`` ``qnn_onnx_decode_core``: ``attack_bit = attack_decided !=
    0``; ``qnn_input.c``: ``impulse = action.attack`` in combat mode). The decode
    itself emits the HELD impulse on no-fire ticks — the A26 SWITCH convention,
    where ``attack`` is a separate fire bit and a held impulse is a server no-op
    — so the lane MUST be gated on the fire bit before it leaves the model, or
    the bot holds the trigger down every tick.

    Same-tick select-and-fire is intact: the server's QC runs ``ImpulseCommands``
    (the weapon change) BEFORE the ``button0`` attack check inside one
    ``W_WeaponFrame``, so an impulse emitted on the fire tick fires the weapon it
    selects. Weapon switching therefore never needs a no-fire-tick impulse.

    Shared by ``QNNPolicy.act`` and the ONNX ExportWrapper — one law, so offline
    and deploy cannot fork. TRACE-SAFE (``torch.where`` only).
    """
    f = fire.reshape(-1).to(torch.bool)
    w = weapon_impulse.reshape(-1).to(torch.int64)
    return torch.where(f, w, torch.zeros_like(w))


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
    diverging: torch.Tensor,        # (B,) bool — hbw predicted to WORSEN next tick (feed-forward)
    veto_mask: torch.Tensor | None = None,  # (B,) bool — hard guards outrank the gate
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """One tick of the crest latch. Returns ``(attack, weapon_impulse,
    attack_state_out)``.

    Decode law (all rows in parallel; countdown ``pending`` = state lane 0):

    * arm     — head fires, gate applies (θ>0, enemy live, weapon ready), not at
                crest, not diverging, latch idle → HOLD: emit attack=0 + held
                impulse (a server no-op), start the countdown at H.
    * release — at crest (``hbw ≤ θ`` of the CURRENT selection), or predicted to
                diverge (waiting won't help — fire now instead of riding a hold
                that's only getting worse), or countdown expiring (pending==1 →
                blind-fire; the head's discharge is never canceled), or a
                gate-exempt fire (θ≤0 / no enemy / on cooldown / already aligned)
                passing straight through on its own tick.
    * fire    — release minus this tick's hard-guard veto (guards outrank; a
                vetoed release still clears the latch — the discharge is lost,
                same as a vetoed raw fire).

    Cooldown-honesty: ``ready`` gating means a fire the engine would discard
    (attack_finished pending) is passed through raw, never converted into a
    delayed REAL discharge. No restack: arm requires pending==0, so a head
    re-fire during a hold is absorbed into the pending discharge. Exactly one
    attack=1 tick per armed discharge (single-tick preserved).

    Convergence gating: ``diverging`` is re-read every tick (arm AND each hold
    tick share the identical ``stop_waiting`` predicate) — a hold is only ever
    worth continuing while the feed-forward rate predicts hbw improving; the
    tick it turns, the gate stops waiting immediately rather than riding out
    the remaining hold to a worse blind-fire at expiry."""
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
    # diverging needs live too, same reasoning as aligned: a dead row (LOS
    # lost) must hold to expiry, not release as a fake divergence.
    stop_waiting = aligned | (live & diverging)
    idle = pending == 0
    arm = fire_b & gate_on & ~stop_waiting & idle
    tick = pending > 0
    release = ((fire_b & (stop_waiting | ~gate_on) & idle)
               | (tick & (stop_waiting | (pending == 1))))
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
    obs_tensors,
    move_logits: torch.Tensor,
    *,
    self_weapon_id: torch.Tensor | None = None,
    guard=None,
    attack_bias: float = 0.0,
    bias_vec: torch.Tensor | None = None,
    fire_bias_vec: torch.Tensor | None = None,
    preference_bias_vec: torch.Tensor | None = None,
    switch_margin: float = 0.0,
    crest_theta_vec: torch.Tensor | None = None,
    crest_hold_ticks: int = 0,
    aim_z_err: torch.Tensor | None = None,
    aim_range: torch.Tensor | None = None,
    aim_z_rate: torch.Tensor | None = None,
    attack_state: torch.Tensor | None = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]":
    """Full a25 attack-with decode — the SINGLE implementation shared by
    ``QNNPolicy.act`` and the ONNX ExportWrapper (decode-regime-in-model: the
    offline decode and the baked-into-ONNX deploy decode are the same code, so
    they cannot skew). Resolves the held impulse and the guard's alignment
    bias + hard-veto mask, runs :func:`attack_with_decode_step`, then composes
    the crest gate (:func:`attack_crest_gate_step`) when armed.

    Parameters
    ----------
    logits9 : (B, 9) raw attack-with logits.
    obs_tensors : the guard's obs input (each caller passes the form its guard
        expects — dequantised in act, native in export; the guard reads it).
        The crest gate also reads ``attack_finished`` from it (cooldown gate).
    move_logits : (B, MOVE_AXES, classes) or flat — the guard argmaxes it for the
        movement-direction splash checks.
    self_weapon_id : (B,) / (B,1) held-weapon id (obs ``self_weapon_id``), or
        None for a held-weapon-blind graph (the A27 pure-combat contract, which
        has no held-weapon input — see agents/plans/decode-fit-reconciliation.md
        "A27 Port Boundary"). None falls back to always selecting the ideal
        weapon (``switch_margin`` has no effect: there is no held weapon to
        hysterese against) until A27 restores a held-weapon signal.
    guard : the resolved guard module (``guard_attack_logit_for_export``); None →
        no alignment bias / veto. The guard applies the SAME hard vetoes
        (attack-splash / RL self-splash / LG range) that the deployed graph bakes.
    attack_bias / bias_vec / fire_bias_vec / preference_bias_vec /
        switch_margin : the operating-point knobs
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
    aim_z_rate : the same function's feed-forward tangent DELTA (one tick
        ahead, under the target's current relative velocity) — reconstructs
        ``z_next = aim_z_err + aim_z_rate`` to predict whether hbw is
        converging or diverging, so the gate never holds a discharge that
        isn't going to improve. Required when the gate is armed, same as
        aim_z_err/aim_range.
    attack_state : (B, ATTACK_STATE_DIM) — the countdown latch's wire slot
        (lane 0; zeros = idle). Required when the gate is armed.

    Returns ``(attack, weapon_impulse, attack_state_out)``; with the gate OFF
    ``attack_state_out`` is the input ``attack_state`` unchanged (passthrough).

    TRACE-SAFETY: torch-only, no ``.item()`` / data-dependent control flow (the
    guard presence + knobs are static python config, decided at trace build).
    """
    from qnn.vocab import self_weapon_id_to_impulse

    l9 = logits9.reshape(-1, ATTACK_WITH_SIZE)
    # Naive ideal weapon (no bias_vec, no hysteresis) — a guard-probe heuristic
    # only; the real selection (with switch_margin) happens below.
    choice = l9[..., 1:].argmax(dim=-1) + 1
    held_impulse = (
        self_weapon_id_to_impulse(self_weapon_id.reshape(-1))
        if self_weapon_id is not None else choice)
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
            obs_tensors, move_logits, choice,
            torch.zeros_like(l9[..., :1])).reshape(-1)
        veto_mask = probe < -1.0e8
        align_bias = torch.where(veto_mask, torch.zeros_like(probe), probe)
    fire, weapon_impulse = attack_with_decode_step(
        l9, held_impulse,
        attack_bias=attack_bias, bias_vec=bias_vec,
        fire_bias_vec=fire_bias_vec,
        preference_bias_vec=preference_bias_vec,
        switch_margin=switch_margin,
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
    if aim_z_rate is None:
        raise ValueError(
            "attack crest gate is armed but aim_z_rate was not supplied — the "
            "convergence check needs the feed-forward tangent delta from the "
            "same aim_prior_tangent_ffwd call that produced aim_z_err/aim_range.")
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
    # Feed-forward convergence check: predict next tick's z_err under the
    # target's current relative velocity (z_next = z_err + z_rate, exactly the
    # quantity aim_prior_tangent_ffwd's z_rate is defined from) and run it
    # through the SAME hbw law. diverging = the gate would be worse off a tick
    # from now, so holding can't pay off — release now instead of waiting.
    hbw_next, _ = crest_alignment_hbw(
        aim_z_err.reshape(-1, 2) + aim_z_rate.reshape(-1, 2), aim_range)
    diverging = hbw_next > hbw
    # per-tick selection re-read (same law as the step, incl. switch hysteresis)
    # so a mid-hold weapon switch releases against the NEW weapon's θ.
    _score = (l9[..., 1:] if bias_vec is None
              else l9[..., 1:] + bias_vec.reshape(-1).to(l9.dtype))
    if preference_bias_vec is not None:
        _score = _score + preference_bias_vec.reshape(-1).to(l9.dtype)
    choice = _attack_with_select(_score, held_impulse, switch_margin) + 1
    return attack_crest_gate_step(
        fire, choice.to(torch.int64), held_impulse, attack_state,
        crest_theta_vec=crest_theta_vec, crest_hold_ticks=int(crest_hold_ticks),
        hbw=hbw, live=live, ready=ready, diverging=diverging, veto_mask=veto_mask)


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

from qnn.model.move_seg_head import (  # noqa: E402
    FIB_EDGES, N_BUCKETS, N_CLASSES, N_AXES, JOINT)
from qnn.model.decode import gumbel_argmax  # noqa: E402

# concrete duration range per bucket (frames); tail extends one fib step
_BUCKET_LO = list(FIB_EDGES)
_BUCKET_HI = [e - 1 for e in FIB_EDGES[1:]] + [144]

# Flat carried-state width for the commitment decode in the graph. We reuse the
# a24 move_state slot so the ONNX I/O tensor NAME is unchanged across dim bumps
# (move_state_out rode (B,11)->(B,9) at wire.9->wire.11 the same way — a
# decode-regime change, not a wire bump; see wire.11.md) — only the column
# semantics + episode-reset init differ:
#   [0]=fb_cls [1]=fb_rem [2]=lr_cls [3]=lr_rem [4]=prev_release
#   [5]=ud_cls [6]=ud_rem  (water-ud swim commit; movearch only)
#   [7]=ammo-lockout baseline (last-seen watched-ammo value)
#   [8]=engagement cooldown counter
#   [9]=ammo-lockout staleness ticks (ticks since last observed decrement)
# cls<0 = unset (episode start) -> decode samples a fresh commitment.
COMMIT_STATE_DIM = 10
_COMMIT_RESET_LANES = (-1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)


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


def world_enemy_present(entity_types: torch.Tensor) -> torch.Tensor:
    """(B,) bool — an ACTOR (other player) entity is observed. The low-end gate
    for the movement engagement scalar (no actor => engagement 0). Torch-only,
    trace-safe."""
    from qnn.vocab import TOKEN_ACTOR
    return (entity_types == TOKEN_ACTOR).any(dim=-1)


def world_engaged_active(entity_types: torch.Tensor,
                         entity_event_actions: torch.Tensor,
                         entity_event_count: torch.Tensor,
                         entity_scalars_raw: torch.Tensor) -> torch.Tensor:
    """(B,) bool — an EXTERNAL combat event this tick: a target (ACTOR) is
    firing (an observed FIRE action event) OR a non-self projectile is present.
    Drives the movement engagement scalar to 1 (with a cooldown hold applied by
    the caller). Purely external — no bot intent. Torch-only, trace-safe.

    entity_event_actions : (B, N, MAX_ENTITY_EVENTS) action ids (see vocab.ACTION_IDS)
    entity_event_count   : (B, N) valid-event count per entity
    entity_scalars_raw   : (B, N, >=3) — rel[0:3] lead (projectile distance gate)
    """
    from qnn.vocab import TOKEN_ACTOR, ACTION_IDS
    is_actor = (entity_types == TOKEN_ACTOR)                          # (B,N)
    ev = entity_event_actions.shape[-1]
    evt_idx = torch.arange(ev, device=entity_types.device)
    evt_valid = evt_idx.reshape(1, 1, -1) < entity_event_count.unsqueeze(-1)  # (B,N,ev)
    is_fire = (entity_event_actions == int(ACTION_IDS["FIRE"]))       # (B,N,ev)
    actor_fire = (is_actor.unsqueeze(-1) & is_fire & evt_valid).any(dim=-1).any(dim=-1)  # (B,)
    nonself_proj = incoming_projectile_present(entity_types, entity_scalars_raw)
    return actor_fire | nonself_proj


# ── obs → move-decode signals: ONE derivation, both decode call sites ───────
#
# The commitment decode's EXTERNAL inputs (threat, engagement, ammo lockout)
# are not head outputs — they are read off the observation. That read is
# decode logic, not caller glue, so it lives here beside the predicates it
# feeds and is called by BOTH QNNPolicy.act and the ONNX ExportWrapper (the
# same doctrine attack_with_decode states: one implementation, no twin).
#
# It used to be assembled independently at each call site, and the copies were
# NOT equivalent — the eval copy read ``entity_rel`` (raw game units) while the
# export copy read ``entity_scalars_raw[..., 0:3]`` (the DEQUANTIZED rel, i.e.
# raw / DIST_SCALE). Against the OWN_FIRE_DIST_U = 120 u gate the export copy
# could never be true (|rel| / 1000 ≤ 32.8), so the whole projectile half of
# the threat / engagement signal was silently dead in the shipped graph while
# live in eval. The eval copy was additionally wrapped in a key-presence guard
# that left every signal None — and the knob a silent no-op — on any obs whose
# entity fields use the dequantized key names.
#
# Both hazards are structural: derive the signals in ONE place, in ONE unit
# convention, and RAISE on an obs that cannot supply them.


def _entity_types(obs: "Mapping[str, torch.Tensor]", rows: int, why: str) -> torch.Tensor:
    if "entity_types" not in obs:
        raise KeyError(f"{why} but the obs lacks 'entity_types'")
    return torch.as_tensor(obs["entity_types"]).reshape(rows, -1).long()


def projectile_rel_raw(obs: "Mapping[str, torch.Tensor]", rows: int,
                       entity_types: torch.Tensor) -> torch.Tensor:
    """(rows, N, 3) PROJECTILE-token rel in RAW GAME UNITS (zero elsewhere) —
    the convention every distance gate in this module is written in
    (``OWN_FIRE_DIST_U`` et al).

    Two obs forms carry the vector, and they are the SAME quantity in
    different encodings, so either is accepted and both yield identical
    values — that equivalence is what makes eval and deploy comparable:

    * ``entity_rel`` — the native/wire field, int16 raw game units (an ONNX
      graph input; the field-granular env obs and the BC native cache carry
      it verbatim).
    * ``entity_scalars_raw[..., 0:3]`` — the dequantized packing, rel /
      ``DIST_SCALE``, multiplied back out here.

    Non-projectile rows are zeroed rather than returned as-is: the two
    encodings only agree on projectile tokens (the packed layout leads with
    rel for PROJECTILE but with half-extents for ACTOR/ITEM/MOVER, which put
    rel at [3:6]), and every caller masks to projectiles anyway. Zeroing makes
    the two sources interchangeable everywhere instead of only where the
    caller happens to look.

    Raises when neither key is present: a decode that silently drops its own
    world input is worse than one that stops.
    """
    rel = obs.get("entity_rel")
    if rel is not None:
        rel = torch.as_tensor(rel).reshape(rows, -1, 3).float()
    else:
        esr = obs.get("entity_scalars_raw")
        if esr is None:
            raise KeyError(
                "move decode needs the entity rel vector: obs carries neither "
                "'entity_rel' (native) nor 'entity_scalars_raw' (dequantized)")
        esr = torch.as_tensor(esr)
        esr = esr.reshape(rows, -1, esr.shape[-1])
        rel = esr[..., 0:3].float() * float(_DIST_SCALE)
    from qnn.vocab import TOKEN_PROJECTILE
    keep = (entity_types == TOKEN_PROJECTILE).unsqueeze(-1)
    return torch.where(keep, rel, torch.zeros_like(rel))


def move_threat_signal(obs: "Mapping[str, torch.Tensor]", rows: int) -> torch.Tensor:
    """(rows,) bool — the threat-break input: an incoming projectile is
    present. Obs-side twin of :func:`incoming_projectile_present`, shared by
    both decode paths so the distance gate sees the same units everywhere."""
    et = _entity_types(obs, rows, "move.threat_break_hazard is enabled")
    return incoming_projectile_present(et, projectile_rel_raw(obs, rows, et))


def move_engagement_signals(
    obs: "Mapping[str, torch.Tensor]", rows: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """``(enemy_present, engaged_active, ammo_pools, held_impulse)`` — the
    external engagement inputs of the idle stillness bias, derived from one
    obs by one implementation for both decode paths.

    ``ammo_pools`` is the dequantized ``self_ammo_pools`` (per-pool normalized
    fraction), NOT the raw byte counts: :func:`ammo_staleness_step` only ever
    compares a pool against its own previous value, so the normalization is
    irrelevant to the law — but the two paths must pick the SAME one, or a
    tick that switches the held weapon across pools compares two different
    scales and disagrees about whether ammo dropped.

    Accepts either entity-event-count spelling (``entity_event_counts`` is the
    dequantizer/schema output, ``entity_event_count`` the native wire field).
    Raises on an obs missing any required field — the caller asked for the
    engagement gate, so it must be computable.
    """
    from qnn.vocab import self_weapon_id_to_impulse
    missing = [k for k in ("entity_types", "entity_event_actions",
                           "self_ammo_pools", "self_weapon_id") if k not in obs]
    counts = obs.get("entity_event_counts")
    if counts is None:
        counts = obs.get("entity_event_count")
        if counts is None:
            missing.append("entity_event_counts")
    if missing:
        raise KeyError(
            "move.idle_none_bias is enabled but the obs lacks the engagement "
            f"fields {sorted(missing)} — the engagement gate cannot be "
            "derived from this observation source")
    et = _entity_types(obs, rows, "move.idle_none_bias is enabled")
    rel = projectile_rel_raw(obs, rows, et)
    actions = torch.as_tensor(obs["entity_event_actions"]).reshape(
        rows, et.shape[1], -1).long()
    cnt = torch.as_tensor(counts).reshape(rows, -1).long()
    ammo_pools = torch.as_tensor(obs["self_ammo_pools"]).reshape(rows, 4).float()
    held_impulse = self_weapon_id_to_impulse(
        torch.as_tensor(obs["self_weapon_id"]).reshape(rows).long())
    return (world_enemy_present(et),
            world_engaged_active(et, actions, cnt, rel),
            ammo_pools, held_impulse)


# ── Ammo-lockout: a world-results override for the engagement gate ──────────
#
# Some external attack-disable states have no engine-visible flag at all (a
# competitive mod's pre-match freeze; any other server-side lockout) — but
# their world-result is unmistakable: the bot keeps commanding fire and its
# ammo never moves. Reuses the SAME shared threshold as the engagement
# cooldown hold (idle_cooldown_ticks) rather than a second knob: if the ammo
# pool relevant to the held weapon hasn't decremented in over that many
# ticks, `engagement_none_bias` forces E to 0 outright — full idle stillness
# — regardless of what enemy_present/engaged_active say. Needs 2 lanes of
# persistent state (a baseline ammo value + a staleness counter), threaded on
# commit_state lanes [7]/[9] by the caller (move_commit_step /
# move_commit_step_graph); see agents/plans/discharge-quality-gate.md's
# convergence-gating addition for the same "reuse an existing signal, add a
# small counter" shape.

_AMMO_POOL_BY_IMPULSE = torch.tensor(
    [-1, -1, 0, 0, 1, 1, 2, 2, 3], dtype=torch.long)
# impulse:   0     1    2    3   4    5   6   7   8
#          (pad) (axe) SG  SSG  NG  SNG  GL  RL  LG
# pool:     -1    -1    0    0   1    1   2   2   3   (shells,nails,rockets,cells)
# Axe (impulse 1, melee) has no ammo pool: -1 EXEMPTS it below — there is
# nothing to observe a decrement of, so it must never accrue staleness.


def held_ammo_for_impulse(
    ammo_pools: torch.Tensor,   # (B, 4) [shells, nails, rockets, cells]
    impulse: torch.Tensor,      # (B,) held weapon impulse 1..8
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(ammo, has_pool)`` — the ammo count relevant to the held weapon,
    gathered from ``ammo_pools``. ``has_pool`` is False for the axe (no ammo
    pool at all); callers must exempt those rows from staleness tracking.
    Trace-safe (index_select/gather only)."""
    imp = impulse.reshape(-1).to(torch.long).clamp(0, 8)
    pool_idx = _AMMO_POOL_BY_IMPULSE.to(imp.device).index_select(0, imp)  # (B,) -1..3
    has_pool = pool_idx >= 0
    safe_idx = pool_idx.clamp(min=0)
    ammo = ammo_pools.reshape(-1, 4).gather(-1, safe_idx.unsqueeze(-1)).squeeze(-1)
    return torch.where(has_pool, ammo, torch.zeros_like(ammo)), has_pool


def ammo_staleness_step(
    ammo_prev: torch.Tensor,    # (B,) last tick's watched-ammo value (state)
    stale_prev: torch.Tensor,   # (B,) ticks since last observed decrement (state)
    ammo_now: torch.Tensor,     # (B,) this tick's watched-ammo value
    has_pool: torch.Tensor,     # (B,) bool — False (axe) exempts the row
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(ammo_next, stale_next)`` — tick-to-tick ammo-decrement tracking. A
    row with no ammo pool (``has_pool`` False) never accrues staleness: both
    outputs hold at 0, so it can never trigger the lockout override.
    Trace-safe (no data-dependent control flow)."""
    decreased = ammo_now < ammo_prev
    stale_next = torch.where(
        has_pool,
        torch.where(decreased, torch.zeros_like(stale_prev), stale_prev + 1.0),
        torch.zeros_like(stale_prev))
    ammo_next = torch.where(has_pool, ammo_now, torch.zeros_like(ammo_now))
    return ammo_next, stale_next


def engagement_none_bias(cooldown_prev: torch.Tensor,
                         enemy_present: torch.Tensor,
                         engaged_active: torch.Tensor,
                         idle_none_bias: "tuple[float, float]",
                         base: float,
                         cooldown_ticks: int,
                         ammo_lockout: "torch.Tensor | None" = None
                         ) -> "tuple[torch.Tensor, torch.Tensor]":
    """Engagement-gated stillness bias for the move commitment decode.

    External engagement E in [0,1]: 1 while actively engaged (target firing or
    non-self projectile) OR within the cooldown hold; ``base`` when an enemy is
    merely present; 0 when there is no target. The cooldown is a retriggerable
    one-shot: any active tick re-arms it to ``cooldown_ticks`` and it counts
    down otherwise, so E holds at 1 for ``cooldown_ticks`` after combat stops.

    ``ammo_lockout`` (B,) bool, optional: forces E to 0 outright — as if no
    enemy were present — regardless of enemy_present/engaged_active, when
    :func:`ammo_staleness_step` has observed no ammo decrement for longer than
    ``cooldown_ticks`` (the SAME shared threshold, not a second knob). See the
    module-level ammo-lockout comment above.

    Returns ``(none_bias (B,2), cooldown_new (B,))``: ``none_bias[:,ai] =
    idle_none_bias[ai] * (1 - E)`` — a per-axis additive bias on the seg head's
    ``none`` (stand-still) class that vanishes in combat (E=1) and is full when
    idle (E=0). Torch-only, trace-safe. Carries the counter on commit_state
    lane 8 (no wire change)."""
    cd = torch.where(engaged_active.bool(),
                     torch.full_like(cooldown_prev, float(cooldown_ticks)),
                     (cooldown_prev - 1.0).clamp(min=0.0))
    E = torch.where(cd > 0.0, torch.ones_like(cd),
                    torch.where(enemy_present.bool(),
                                torch.full_like(cd, float(base)),
                                torch.zeros_like(cd)))
    if ammo_lockout is not None:
        E = torch.where(ammo_lockout.reshape(E.shape).bool(),
                        torch.zeros_like(E), E)
    one_minus_E = 1.0 - E
    nb = torch.stack([float(idle_none_bias[0]) * one_minus_E,
                      float(idle_none_bias[1]) * one_minus_E], dim=-1)  # (B,2)
    return nb, cd


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
    recommit: bool = False,              # allow re-commit to the held class at expiry
    enemy_present: torch.Tensor | None = None,   # (B,) bool — an actor is observed
    engaged_active: torch.Tensor | None = None,  # (B,) bool — target firing / non-self projectile
    idle_none_bias: "tuple[float, float]" = (0.0, 0.0),  # per-axis (fb,lr) idle stand-still bias
    idle_engagement_base: float = 0.5,   # E when an enemy is present but not active
    idle_cooldown_ticks: int = 20,       # 1s @ 20Hz engagement hold after combat
    ammo_pools: torch.Tensor | None = None,      # (B,4) [shells,nails,rockets,cells]
    held_impulse: torch.Tensor | None = None,    # (B,) held weapon impulse 1..8
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
    # Engagement-gated stillness bias (idle_none_bias, default (0,0)=off): when
    # disengaged, raise the ``none`` (stand-still) class so the bot stops
    # pointless idle strafing; vanishes in combat (E=1). Cooldown counter on
    # lane 8 (no wire change). Ammo-lockout override (lanes 7/9, see
    # ammo_staleness_step above engagement_none_bias) needs one more lane than
    # the base engagement feature, so it's gated separately — a caller still
    # on the pre-ammo-lockout 9-lane state gets engagement without the
    # override rather than losing the whole feature.
    none_bias = None
    if (idle_none_bias != (0.0, 0.0) and enemy_present is not None
            and engaged_active is not None and commit_state.shape[1] > 8):
        ammo_lockout = None
        if (ammo_pools is not None and held_impulse is not None
                and commit_state.shape[1] > 9):
            _ammo_now, _has_pool = held_ammo_for_impulse(ammo_pools, held_impulse)
            _ammo_next, _stale_next = ammo_staleness_step(
                commit_state[:, 7], commit_state[:, 9], _ammo_now, _has_pool)
            commit_state[:, 7] = _ammo_next.to(commit_state.dtype)
            commit_state[:, 9] = _stale_next.to(commit_state.dtype)
            ammo_lockout = _stale_next > float(idle_cooldown_ticks)
        none_bias, _cd_new = engagement_none_bias(
            commit_state[:, 8], enemy_present.reshape(B), engaged_active.reshape(B),
            idle_none_bias, idle_engagement_base, idle_cooldown_ticks,
            ammo_lockout=ammo_lockout)
        commit_state[:, 8] = _cd_new.to(commit_state.dtype)
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
        if none_bias is not None:                                # engagement stillness bias
            logits[:, N_BUCKETS:2 * N_BUCKETS] = (              # class 1 = none buckets
                logits[:, N_BUCKETS:2 * N_BUCKETS] + none_bias[:, ai].unsqueeze(1))
        held = cls.clamp(min=0)
        bucket_cols = torch.arange(N_BUCKETS, device=dev)
        mask_cols = held.unsqueeze(1) * N_BUCKETS + bucket_cols  # (B, 10)
        # recommit: allow the head to re-commit to the held class at expiry (no
        # forced switch) — the held class stays available so a maximal run can
        # be extended. Off (default) = the maximal-run law (expiry forces a
        # class change), bit-identical to pre-knob.
        row_has_held = ((cls >= 0) & (rem <= 0)).unsqueeze(1)
        if recommit:
            row_has_held = torch.zeros_like(row_has_held)
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
    jump_threshold: float = 0.0,            # >0 = deterministic jump gate p_jump > τ
    threat: torch.Tensor | None = None,     # (B,) bool — incoming projectile present
    threat_break_hazard: float = 0.0,       # per-tick re-decision prob on threat rows
    recommit: bool = False,                 # allow re-commit to the held class at expiry
    enemy_present: torch.Tensor | None = None,   # (B,) bool — an actor is observed
    engaged_active: torch.Tensor | None = None,  # (B,) bool — target firing / non-self projectile
    idle_none_bias: "tuple[float, float]" = (0.0, 0.0),  # per-axis (fb,lr) idle stand-still bias
    idle_engagement_base: float = 0.5,      # E when an enemy is present but not active
    idle_cooldown_ticks: int = 20,          # 1s @ 20Hz engagement hold after combat
    ammo_pools: torch.Tensor | None = None,      # (B,4) [shells,nails,rockets,cells]
    held_impulse: torch.Tensor | None = None,    # (B,) held weapon impulse 1..8
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

    # Engagement-gated stillness bias (idle_none_bias, default off) — trace-safe
    # twin of the eager path. Cooldown counter on commit_state lane 8. Ammo-
    # lockout override (lanes 7/9) needs one more lane than base engagement,
    # so it's gated separately — same reasoning as the eager twin.
    none_bias = None
    cooldown_new = commit_state[:, 8] if commit_state.shape[1] > 8 else None
    ammo_baseline_new = commit_state[:, 7] if commit_state.shape[1] > 7 else None
    ammo_stale_new = commit_state[:, 9] if commit_state.shape[1] > 9 else None
    if (idle_none_bias != (0.0, 0.0) and enemy_present is not None
            and engaged_active is not None and commit_state.shape[1] > 8):
        ammo_lockout = None
        if (ammo_pools is not None and held_impulse is not None
                and commit_state.shape[1] > 9):
            _ammo_now, _has_pool = held_ammo_for_impulse(ammo_pools, held_impulse)
            ammo_baseline_new, ammo_stale_new = ammo_staleness_step(
                commit_state[:, 7], commit_state[:, 9], _ammo_now, _has_pool)
            ammo_lockout = ammo_stale_new > float(idle_cooldown_ticks)
        none_bias, cooldown_new = engagement_none_bias(
            commit_state[:, 8], enemy_present.reshape(B), engaged_active.reshape(B),
            idle_none_bias, idle_engagement_base, idle_cooldown_ticks,
            ammo_lockout=ammo_lockout)

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
        if none_bias is not None:                                   # engagement stillness bias (class 1 = none)
            logits = logits + torch.where(
                col_class == 1, none_bias[:, ai].reshape(B, 1),
                torch.zeros((), device=dev, dtype=logits.dtype))
        # expiry-only held-class mask: -1e9 on the held class's buckets when a
        # completed segment is ending (cls>=0 & rem<=0). Interrupt is a
        # re-decision (held stays available) — matches the eager scatter law.
        held = cls.clamp(min=0).reshape(B, 1)                       # (B,1)
        row_has_held = ((cls >= 0) & (rem <= 0)).reshape(B, 1)      # (B,1)
        if recommit:                                                # allow re-commit: never mask held
            row_has_held = torch.zeros_like(row_has_held)
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
    cd_lane = cooldown_new.to(torch.float32) if cooldown_new is not None else zeros    # lane 8
    ammo_lane = ammo_baseline_new.to(torch.float32) if ammo_baseline_new is not None else zeros  # lane 7
    stale_lane = ammo_stale_new.to(torch.float32) if ammo_stale_new is not None else zeros       # lane 9
    if water is None or jump_logit is None:
        # Legacy (fb/lr-only) graph: lane 5,6 carried as 0 — bit-identical;
        # lane 7 = ammo-lockout baseline, lane 8 = engagement cooldown, lane
        # 9 = ammo-lockout staleness ticks (all 0 when the features are off).
        cols = [
            emit_cols[0].to(torch.float32), rem_cols[0].to(torch.float32),
            emit_cols[1].to(torch.float32), rem_cols[1].to(torch.float32),
            rel_now.to(torch.float32),
            zeros, zeros, ammo_lane, cd_lane, stale_lane,
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

    # jump: the engine-outcome posterior on every land row. jump_threshold τ>0
    # gates DETERMINISTICALLY (jump iff p_jump > τ) — only the most-confident,
    # context-motivated jumps fire, matching the eager act path and making
    # deploy==offline (the ORT RandomUniformLike stream drops out). τ=0 keeps
    # the AS-IS decode (greedy argmax>0.5 / sampled Bernoulli).
    p_jump = torch.sigmoid(jump_logit.reshape(B).to(torch.float32))
    if jump_threshold > 0.0:
        fire_j = p_jump > float(jump_threshold)
    elif greedy:
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
        ammo_lane, cd_lane, stale_lane,      # lane 7 ammo baseline, 8 cooldown, 9 ammo stale
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
    weapon_impulse: torch.Tensor,
    weapon_pitch_bias: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per-row VERTICAL feet-aim BLEND terms — the RL-splash feet-aiming.

    ``z_err`` is the UNSCALED aim-prior error tangent (vertical component
    ``z_err[..., 1]`` is the absolute turn-to-anchor pitch, + = up / − = down).
    ``weapon_pitch_gain`` is a (9,) per-IMPULSE blend weight β ∈ [0, 1] (0 = OFF).
    ``weapon_impulse`` is the attack-with intent in engine order (0..8).

    Returns ``(beta, target_vert, feet_vert)``, consumed by :func:`decode_look_from_polar`
    as the LERP ``z_vert ← (1−β)·z_head_vert + β·target_vert``. β is enemy-gated
    (``z_err`` is exactly zero rows otherwise). ``weapon_pitch_bias`` (per-IMPULSE deg,
    default None) deepens the target to cancel the static RL fire-high offset.
    """
    imp = weapon_impulse.reshape(-1).long().clamp(0, 8)
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
    mag_bin: "torch.Tensor | None" = None,
    dir_logits: "torch.Tensor | None" = None,
    mag_centers: "torch.Tensor | None" = None,
    dir_centers: "torch.Tensor | None" = None,
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
    *,
    theta_override: "torch.Tensor | None" = None,
    phi_override: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """(sampled ``mag_bin``, ``dir_logits``) → look unit vector (hybrid decode).

    ``turn_mag_scale`` (default 1.0 = OFF = bit-identical) multiplicatively dampens the
    head's NATIVE turn magnitude ``|z| = θ`` BEFORE the aim-prior blend. The caller
    supplies ``mag_bin`` from ITS OWN magnitude sampler (seeded categorical offline /
    in-graph Gumbel-argmax for export). This function owns the shared remainder: the
    continuous DIRECTION readout (circular mean of the direction softmax), the
    z-assembly, the aim-prior blend, the feet-aim pitch, and the expmap.

    ``theta_override`` / ``phi_override`` (the look_seg COMMIT decode path): when
    given, skip the polar mag_bin/dir readout and take the per-tick signed
    magnitude θ and direction φ DIRECTLY from the commit playout
    (``look_commit_step`` / ``look_commit_step_graph``). ``turn_mag_scale`` still
    applies to the supplied θ; the shared z-assembly + aim-prior blend + feet-aim
    pitch + expmap remainder is IDENTICAL to the polar path — the single source of
    truth co-decoded with tools/export_onnx.ExportWrapper.

    Returns the look unit vector with shape ``mag_bin.shape + (3,)``.
    """
    if theta_override is not None:
        # COMMIT path: θ, φ supplied by the segment-commitment playout.
        theta = theta_override
        phi = phi_override
    else:
        # polar_to_tangent via index_select (export-friendly vs advanced index).
        theta = mag_centers.index_select(0, mag_bin.reshape(-1)).reshape(mag_bin.shape)
        # Continuous direction: circular mean of the direction distribution.
        p_dir = torch.softmax(dir_logits, dim=-1)                   # (..., N_DIR)
        cos_phi = (p_dir * torch.cos(dir_centers)).sum(dim=-1)
        sin_phi = (p_dir * torch.sin(dir_centers)).sum(dim=-1)
        phi = torch.atan2(sin_phi, cos_phi)                         # (...,)
    # Head turn-magnitude dampener (turn_mag_scale, default 1.0 = no-op).
    # A per-ROW tensor (the aim-grid per-lane override) dampens each lane's θ
    # independently; the scalar/export path is unchanged (bit-identical).
    if torch.is_tensor(turn_mag_scale):
        theta = theta * turn_mag_scale.reshape(theta.shape).to(theta.dtype)
    elif turn_mag_scale != 1.0:
        theta = theta * float(turn_mag_scale)
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
