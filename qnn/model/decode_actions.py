"""a25 attack-with decode — the joint 9-way head's deploy protocol.

One trace-safe function shared by ``QNNPolicy.act`` and the ONNX export
wrapper (decode-regime-in-model: the engine consumes the FINAL fire bit +
weapon impulse; no engine-side decode). Greedy by construction — the
deterministic commit preserves coordinated actions (rocket jump), and the
offline shape analysis showed raw argmax self-calibrates to ~the human fire
rate (runs/bc/bench/_attack_with_shape.json).

Decode law (per frame) — two explicitly parameterised decisions:

    select_score = (l1..l8) + preference_bias_vec
    choice = argmax(select_score)                  # best weapon this frame
    fire_score  = (l1..l8)[intent] + fire_bias_vec[intent]
    # attack decision (whether) — conditional on the weapon that will fire
    l0'         = l0 - align_bias                 # class-0 = "don't attack" anchor
    fire   = fire_score > l0'   AND NOT veto
    weapon = choice+1                             # the selected weapon

Knob mapping:
  * the alignment-conditioned guard bias acts on the MARGINAL log-odds by
    subtracting from the class-0 logit (bias > 0 → more attack).
  * ``fire_bias_vec`` (8,) affects only the fire comparison for the selected
    weapon. It cannot change which weapon is selected.
  * ``preference_bias_vec`` (8,) affects only weapon selection; it does not
    directly lift a weapon over the no-attack class. Weapon SELECTION
    otherwise belongs to the network. The post-a26 selection laws (human
    transition/continuation, SPRT accumulator, switch lockout, static
    feasibility mask, fire-gather selector, LG range/alignment guards) were
    removed 2026-08-26, and the held-weapon anchor + ``weapon.switch_margin``
    hysteresis on 2026-08-26; their keys FAIL LOUD. Do not reintroduce a
    decode-side law that decides WHICH weapon among the FEASIBLE set, and do
    not reintroduce a held-weapon concept keyed off engine equip state: there
    is no such signal in the a28 obs contract, so an "anchor" sourced that way
    degenerates to this tick's own argmax and any margin on it becomes a
    silent no-op.
  * ``infeasible_vec`` (8,) — static, config-declared exclusion: masks the
    named weapon(s) to -1e9 before ANY selection/fire logic runs. This is a
    FEASIBILITY declaration (this weapon is not on the table this run), not a
    selection law — it does not pick among what remains feasible.
  * ``af_lockout`` (float, a MULTIPLIER, 0 = none) — restores a weapon-switch
    lockout, but keyed entirely off ``self_arsenal_scalars`` (the engine's own
    ``attack_finished`` countdown), not off a re-derived held-weapon identity.
    On the tick a shot lands, ``attack_finished`` goes 0→x (the engine's own
    true per-weapon refire delay, no cooldown table needed); the EXTENSION
    target ``af_lockout * x`` (ceilinged by ``af_lockout_cap``, seconds, 0 =
    uncapped) is SET ONCE at that rising edge and HELD steady — it does not
    track af's own decrements — then only ticks down, at the same real
    per-tick rate af was observed decaying at, once ``attack_finished``
    actually reaches zero. Net effect: every weapon but the one that just
    fired stays masked for x (real engine cooldown) + min(af_lockout * x,
    af_lockout_cap or ∞) more. af_lockout=1, uncapped reproduces the original
    single-multiplier design (a full extra x); af_lockout=0 is the real
    engine-enforced window with no decode-side extension at all. Real
    precedent: a28rc1h shipped the a26-lineage equivalent
    (switch_lockout_mult=2.0, switch_lockout_cap_ticks=6 = 0.3s at 20Hz,
    formula "lockout = cd + min(cd, T)" — agents/plans/rl-skill-finetune.md).
    State is 4 persisted floats in ``attack_state`` (y, locked_weapon,
    af_prev, dt); see ATTACK_STATE_DIM.
  * hard guards (attack-splash / RL self-splash) arrive as a boolean veto
    mask from the a24 guard primitives instead of a buried logit.
  * the a24 sticky gate / hazard / switch-back weapon machinery has no analog —
    it is retired for this head, along with the a26 held-anchor hysteresis.

TRACE-SAFETY: torch-only, no ``.item()``, no data-dependent control flow
(vector/bool presence is decided on static python config, not tensor values;
af_lockout's own branching inside attack_with_decode is likewise static).
"""
from __future__ import annotations

import torch

from qnn.model.decode import BatchedRNG, gumbel_argmax, inverse_cdf_sample, row_uniforms
from qnn.model.look_bins import tangent_expmap
from qnn.schema import WEAPON_HEAD_SIZE
from qnn.engine_norm import DIST_SCALE as _DIST_SCALE
from qnn.engine_norm import TIME_SCALE as _TIME_SCALE

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

# weapon.af_lockout's dt bootstrap: real seconds/tick, used ONLY until a real
# active-to-active af sample is observed. 20Hz-equivalent (the fastest
# plausible operating rate) so the untested guess under-decays rather than
# over-decays — see the dt derivation in attack_with_decode for why this
# cannot be the discharge's own `af` value.
_DT_BOOTSTRAP_SEC = 0.05

def attack_with_decode_step(
    logits9: torch.Tensor,
    *,
    fire_bias_vec: torch.Tensor | None = None,
    preference_bias_vec: torch.Tensor | None = None,
    switch_margin: float = 0.0,
    align_bias: torch.Tensor | None = None,
    veto_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy attack-with decode.

    Two DECOUPLED decisions (the a25 ``stick_bias`` coupling — which perturbed
    the fire decision and caused weapon camping — is retired):

      1. WEAPON — argmax of the preference-adjusted weapon logits, gated by
         ``switch_margin`` (restored 2026-08-26): the preference-adjusted
         ideal only wins over this tick's own RAW argmax (the a28 obs
         contract carries no equip state, so there is no other anchor to
         compare against — no memory across ticks, purely a same-tick
         confidence gate on how much the preference vector is allowed to
         override the network's own unbiased pick) when it beats that raw
         pick's score by more than the margin. 0.0 (default) is a PROVABLE
         no-op — always take the ideal — so every config that omits it is
         bit-identical to before this knob existed.
      2. ATTACK — under the explicit split-v1 vectors, raw attack intent is
         the best RAW attack class and is independent of the selected
         weapon. That intent class's fire-only calibration is then added to
         its score. Thus a preference/continuation correction cannot
         silently cancel an attack merely because it selected a weapon whose
         raw joint class was low. Legacy configs (all explicit vectors absent)
         retain the old ideal-weapon fire comparison exactly. The two
         decisions stay decoupled: no fire-only term reaches selection, and no
         selection-only term is added into the fire score.

    Parameters
    ----------
    logits9 : (B, 9) raw head logits (class 0 = no-attack, 1..8 = attack weapon k)
    fire_bias_vec : optional (8,) fire-only operating point. It is gathered at
        the ideal selected weapon and cannot affect selection.
    preference_bias_vec : optional (8,) selection-only operating point. It
        cannot directly affect the no-attack comparison.
    switch_margin : (>= 0) leave this tick's raw argmax only if the
        preference-adjusted ideal beats it by this much. 0.0 = always ideal
        (no gate) — the law before this knob existed.
    align_bias : optional (B,) additive alignment bias (positive = more attack)
    veto_mask : optional (B,) bool — hard guard veto rows (True = never attack)
    Returns
    -------
    (attack, weapon_impulse) : (B,) int64 attack bit, (B,) int64 impulse 1..8
    """
    weap = logits9[..., 1:]                                  # (B, 8) raw weapon logits
    # ── weapon selection ───────────────────────────────────────────────────
    select_score = weap
    if preference_bias_vec is not None:
        select_score = select_score + preference_bias_vec.reshape(-1).to(weap.dtype)
    if switch_margin != 0.0:
        held_idx = weap.argmax(dim=-1)               # same-tick raw pick, no memory
        ideal_idx = select_score.argmax(dim=-1)
        s_ideal = select_score.gather(-1, ideal_idx.unsqueeze(-1)).squeeze(-1)
        s_held = select_score.gather(-1, held_idx.unsqueeze(-1)).squeeze(-1)
        sel_idx = torch.where((s_ideal - s_held) > float(switch_margin), ideal_idx, held_idx)
    else:
        sel_idx = select_score.argmax(dim=-1)
    # ── attack decision ────────────────────────────────────────────────────
    # Intent is the model's best RAW attack-with class; the fire test is
    # evaluated there. Selection-only terms must not be able to veto that
    # intent by steering to a low raw class, so preference_bias_vec is absent
    # from this score by construction.
    gather_idx = weap.argmax(dim=-1)
    score_ideal = weap.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)
    if fire_bias_vec is not None:
        fire_cal = fire_bias_vec.reshape(-1).to(weap.dtype).gather(
            0, gather_idx.reshape(-1)).reshape(score_ideal.shape)
        score_ideal = score_ideal + fire_cal
    l0 = logits9[..., 0]
    if align_bias is not None:
        l0 = l0 - align_bias.reshape(l0.shape).to(l0.dtype)
    fire = score_ideal > l0
    if veto_mask is not None:
        fire = fire & ~veto_mask.reshape(fire.shape)
    weapon_impulse = sel_idx.to(torch.int64) + 1             # impulse 1..8
    return fire.to(torch.int64), weapon_impulse


def attack_lane_gate(fire: torch.Tensor, weapon_impulse: torch.Tensor) -> torch.Tensor:
    """The A27 single-lane ACTION convention: ``attack WITH weapon X this tick``.

    ``(B,)`` int64, 0 = no attack, 1..8 = press attack while selecting that
    impulse. The A27 pure-combat wire has NO separate weapon slot, so the engine
    derives BOTH the attack button and the weapon impulse from this one value
    (``qnn_onnx.c`` ``qnn_onnx_decode_core``: ``attack_bit = attack_decided !=
    0``; ``qnn_input.c``: ``impulse = action.attack`` in combat mode). The decode
    itself emits the SELECTED impulse on every tick, fire or not, so the lane
    MUST be gated on the fire bit before it leaves the model, or the bot holds
    the trigger down every tick.

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


def attack_with_decode(
    logits9: torch.Tensor,
    obs_tensors,
    move_logits: torch.Tensor,
    *,
    guard=None,
    fire_bias_vec: torch.Tensor | None = None,
    preference_bias_vec: torch.Tensor | None = None,
    switch_margin: float = 0.0,
    infeasible_vec: torch.Tensor | None = None,
    attack_state: torch.Tensor | None = None,
    af_lockout: float = 0.0,
    af_lockout_cap: float = 0.0,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]":
    """Full attack decode: weapon SELECTION + FIRE + guard. Stateless unless
    ``af_lockout`` is armed.

    Weapon selection belongs to the NETWORK. The decode's selection knobs are
    ``preference_bias_vec`` (additive per-weapon bias) and ``switch_margin``
    (restored 2026-08-26 — a same-tick confidence gate on how much that bias
    is allowed to override the network's own raw pick; 0.0 = no gate, a
    provable no-op). The choice is argmax over whatever the FEASIBILITY layer
    (``infeasible_vec`` + guard + ``af_lockout``, all upstream of selection)
    left standing. There is no held-weapon anchor keyed off engine equip
    state — switch_margin's "anchor" is this tick's own raw argmax, with no
    memory across ticks — see the module docstring.

    The post-a26 selection surface — the human transition/continuation laws,
    the SPRT evidence accumulator, the fire-gather selector and the LG
    range/alignment guards — was REMOVED WHOLESALE on 2026-08-26 (Brian: clean
    slate). Measured on the live RA venue, those laws overrode the network's
    own weapon choice on 52-58% of firing ticks and cost the deployed config
    two thirds of its LG. Do not reintroduce a decode-side law that decides
    WHICH weapon among the feasible set; if selection is wrong, the model is
    wrong. ``infeasible_vec`` and ``af_lockout`` are feasibility, not
    selection — they narrow the candidate set, argmax still owns the pick.
    See qnn.model.decode_config's "REMOVED DECODE LAWS" (the removed keys FAIL
    LOUD) and agents/plans/rl-skill-finetune.md.

    Returns ``(attack, weapon_impulse, attack_state_out)``. ``attack_state_out``
    is the input unchanged unless ``af_lockout`` is armed.

    TRACE-SAFETY: torch-only, no ``.item()`` / data-dependent control flow
    (guard presence and af_lockout are static python config, decided at trace
    build; only ordinary tensor ops read attack_state/obs values).
    """
    l9 = logits9.reshape(-1, ATTACK_WITH_SIZE)
    rows = l9.shape[0]
    # Network._weapon_feasibility_mask has already masked unowned/dry classes
    # to -1e9 upstream; the layers below are the DECODE-side feasibility
    # additions (config-declared / guard / af-lockout), composed by OR into
    # one mask applied once, upstream of every selection/fire computation.
    # ── FEASIBILITY ─────────────────────────────────────────────────────────
    infeasible_cols = None
    if infeasible_vec is not None:
        # weapon.infeasible_vec — static per-run exclusion (e.g. "no SG this
        # config"), restored 2026-08-26 (Brian). Unconditional: masked
        # regardless of what the raw logit or preference_bias_vec say.
        infeasible_cols = (infeasible_vec.reshape(-1).to(l9.dtype) > 0.5).expand(
            rows, WEAPON_HEAD_SIZE)
    # guard.lg_range: LG cannot reach past its beam range, so mark it
    # INFEASIBLE for the tick at the same layer ammo/ownership already ride
    # (Network._weapon_feasibility_mask, applied upstream). The network then
    # picks a weapon that can reach — the decode does not choose for it, and
    # does not veto the shot. Compat-gated on the adapter publishing the hook.
    if guard is not None and hasattr(guard, "lg_select_mask"):
        from qnn.model.guard import LG_IMPULSE
        lg_col = torch.zeros(rows, WEAPON_HEAD_SIZE,
                             dtype=torch.bool, device=l9.device)
        lg_col[:, LG_IMPULSE - 1] = guard.lg_select_mask(
            obs_tensors, rows, device=l9.device).reshape(rows)
        infeasible_cols = lg_col if infeasible_cols is None else infeasible_cols | lg_col
    # weapon.af_lockout — restored 2026-08-26 (Brian), keyed off the engine's
    # OWN attack_finished countdown rather than a re-derived held-weapon
    # identity (see module docstring). ``af_lockout`` is a MULTIPLIER on the
    # real per-discharge cooldown (0 = none, 1 = hold for one more cooldown's
    # worth after the real one clears, 3 = three more, etc — matching the
    # a26-lineage weapon.switch_lockout_mult's role, but against the real
    # observed af value instead of a static per-weapon table).
    # ``af_lockout_cap`` ceilings the EXTENSION at that many seconds (0 = no
    # ceiling) — the a26-lineage weapon.switch_lockout_cap_ticks role, in
    # seconds instead of ticks (real precedent: a28rc1h shipped
    # switch_lockout_cap_ticks=6 = 0.3s at 20Hz, formula "lockout = cd +
    # min(cd, T)"; see agents/plans/rl-skill-finetune.md). State (y,
    # locked_weapon, af_prev, dt) lives in attack_state; read at tick-start
    # (below) so it gates THIS tick's mask, written at tick-end (below) from
    # THIS tick's own fire outcome — no separate anchor-tracking law needed.
    #
    # Both fixed 2026-08-26 (Brian + independent Opus review) after
    # a28rc1h2 shipped with two live bugs:
    #  1. self_arsenal_scalars is dequantized /TIME_SCALE (=60) — af here was
    #     seconds/60, not seconds, so af_lockout_cap (documented AND written
    #     in configs as real seconds) never bound at any real value: 0.3
    #     compared against seconds/60 only binds above 18 real seconds.
    #     Fixed by rescaling to real seconds at the read, below — every
    #     other quantity in this block (y, dt, af_prev) inherits real-second
    #     units for free since they are all derived from `af`.
    #  2. y was armed ONCE at the af 0->positive rising edge and then just
    #     HELD (not decayed) for as long as af stayed active — correct for a
    #     single discharge, but wrong for a held continuous-weapon burst:
    #     the server's think-chain re-bumps attack_finished every 0.1s while
    #     button0 stays down (research/mvd-attack-audit.md), so af never
    #     truly clears mid-burst, and the ENGINE'S OWN continuous-fire
    #     hold-tail (qnn_onnx_apply_continuous_hold_tail, C-side, invisible
    #     to this graph) can keep button0 down for up to 0.25s after the
    #     model's own per-tick decode stops wanting to fire. (That tail is
    #     now per-model: `attack.hold_tail_sec`, 0 on every fresh export, so
    #     new models have no such overhang — but an archived .onnx with no
    #     stamp still inherits the 0.25s one.) Held-then-decay
    #     therefore waited for af to fully clear before EVER starting the
    #     decay countdown, tacking af_lockout's full extension onto the END
    #     of however long the engine kept the button down — not what
    #     "one more cooldown's worth of grace after the model stops" means.
    #     Fixed below: arm/decay are now keyed on `fired` (THIS function's
    #     OWN per-tick fire decision, computed inside the graph, genuinely
    #     memoryless and blind to whatever the C client does afterward) —
    #     re-arm on every genuine fire, decay starts the instant this
    #     function's own decode stops firing, same tick, whether or not the
    #     engine goes on to override the actual button state.
    y_prev = locked_prev = af_prev_in = dt_prev = dt = af = None
    if af_lockout != 0.0:
        if attack_state is None:
            raise ValueError(
                "weapon.af_lockout requires attack_state to be threaded "
                "(ATTACK_STATE_DIM lanes: y, locked_weapon, af_prev, dt)")
        # -1 (not the traced `rows` value) for the dynamic dim, matching l9's
        # own reshape above: a literal -1 keeps ONNX shape inference static
        # on ATTACK_STATE_DIM, where reshape(rows, ...) loses it (rows is
        # itself a traced tensor op, not a python int, once batch is a
        # dynamic_axes dim) — caught via onnx_smoke showing attack_state as
        # (dyn,dyn) instead of (dyn,4), 2026-08-26.
        state = attack_state.reshape(-1, ATTACK_STATE_DIM).to(l9.dtype)
        y_prev, locked_prev, af_prev_in, dt_prev = (
            state[:, 0], state[:, 1], state[:, 2], state[:, 3])
        # self_arsenal_scalars is /TIME_SCALE on the wire (dequant.py); real
        # seconds from here on, matching af_lockout_cap's own units.
        af = (obs_tensors["self_arsenal_scalars"][..., 0].reshape(-1).to(l9.dtype)
              * _TIME_SCALE)
        active = af > 0.0
        real_step = active & (af_prev_in > 0.0)        # a genuine active-to-active sample
        step = torch.clamp(af_prev_in - af, min=0.0)   # real seconds elapsed between samples
        # dt: the free-decay rate, independent of the arm/decay fix above.
        # Sampled from a real active-to-active pair when one exists (and kept
        # from the LAST such sample anywhere in the episode otherwise — never
        # reset per-discharge). The only remaining fallback is the true
        # episode-start bootstrap (dt_prev still exactly its zero-init),
        # which must NOT be this discharge's own `af` (a full cooldown magnitude
        # used as a PER-TICK rate collapses y to 0 in one step the first time a
        # burst is shorter than the real cooldown itself — measured, not
        # theoretical: single-tick RL discharge at af_lockout=1 released two
        # ticks after arming instead of ~4, eating the whole extension before
        # it was ever observable). A conservative fixed guess (20Hz-equivalent,
        # the fastest plausible operating rate) UNDER-decays instead —
        # slightly more lockout than intended, never a premature release —
        # until a real sample (from this or any later cooldown) replaces it.
        dt = torch.where(real_step, step, torch.where(dt_prev > 0.0, dt_prev, _DT_BOOTSTRAP_SEC))
        # Masking for THIS tick reads last tick's ending state (y_prev,
        # locked_prev) — matches locked_weapon's own convention: the tick a
        # fire happens is not masked against itself, only ticks after it.
        lockout_active = (y_prev > 0.0) & (locked_prev > 0.5)
        locked_idx = torch.clamp(
            locked_prev.round().to(torch.long) - 1, 0, WEAPON_HEAD_SIZE - 1)
        locked_onehot = torch.nn.functional.one_hot(
            locked_idx, WEAPON_HEAD_SIZE).to(torch.bool)
        lock_cols = (~locked_onehot) & lockout_active.unsqueeze(-1)
        infeasible_cols = lock_cols if infeasible_cols is None else infeasible_cols | lock_cols
    if infeasible_cols is not None:
        l9 = torch.cat(
            [l9[..., :1], torch.where(
                infeasible_cols, torch.full_like(l9[..., 1:], -1.0e9), l9[..., 1:])],
            dim=-1)
    def _resolved_selection() -> torch.Tensor:
        """The impulse 1..8 this tick will actually discharge — the SAME law
        ``attack_with_decode_step`` is about to apply, evaluated early.

        Every weapon-keyed consumer must read this, never a raw-logit argmax
        that is blind to ``preference_bias_vec`` / ``switch_margin``. Fixed
        2026-08-21 (Brian: "silly oversight, not intentional"): the guard
        zeros-probe was keyed on the naive argmax, so the RL self-splash veto
        was evaluated against the wrong weapon in both directions. Must stay
        in lockstep with attack_with_decode_step's own selection below.
        """
        _weap = l9[..., 1:]
        _score = _weap
        if preference_bias_vec is not None:
            _score = _score + preference_bias_vec.reshape(-1).to(l9.dtype)
        if switch_margin != 0.0:
            _held = _weap.argmax(dim=-1)
            _ideal = _score.argmax(dim=-1)
            _s_ideal = _score.gather(-1, _ideal.unsqueeze(-1)).squeeze(-1)
            _s_held = _score.gather(-1, _held.unsqueeze(-1)).squeeze(-1)
            _idx = torch.where((_s_ideal - _s_held) > float(switch_margin), _ideal, _held)
        else:
            _idx = _score.argmax(dim=-1)
        return (_idx + 1).to(torch.int64)

    selected_impulse = _resolved_selection()
    align_bias = None
    veto_mask = None
    if guard is not None and hasattr(guard, "guard_attack_logit_for_export"):
        # Probe the guard on a zeros logit: the adapter is additive (alignment
        # bias) except hard vetoes buried at -1e9, so the probe separates them —
        # veto = probe < -1e8, align_bias = probe elsewhere.
        probe = guard.guard_attack_logit_for_export(
            obs_tensors, move_logits, selected_impulse,
            torch.zeros_like(l9[..., :1])).reshape(-1)
        veto_mask = probe < -1.0e8
        align_bias = torch.where(veto_mask, torch.zeros_like(probe), probe)
    fire, weapon_impulse = attack_with_decode_step(
        l9,
        fire_bias_vec=fire_bias_vec,
        preference_bias_vec=preference_bias_vec,
        switch_margin=switch_margin,
        align_bias=align_bias, veto_mask=veto_mask)

    attack_state_out = attack_state
    if af_lockout != 0.0:
        fired = fire.reshape(-1).to(torch.bool)
        # y: re-armed on EVERY genuine fire this function decodes (not just
        # the burst's first — mirrors locked_weapon's own re-arm-on-fired
        # rule immediately below), from THIS tick's own observed af — target
        # af_lockout * af, ceilinged by af_lockout_cap (0 = uncapped).
        # Decays the instant this function's OWN decode stops firing,
        # regardless of whatever the engine does with the button afterward.
        y_target = af * float(af_lockout)
        if af_lockout_cap != 0.0:
            y_target = torch.clamp(y_target, max=float(af_lockout_cap))
        y = torch.where(fired, y_target, torch.clamp(y_prev - dt_prev, min=0.0))
        locked_new = torch.where(
            fired, weapon_impulse.reshape(-1).to(l9.dtype), locked_prev)
        attack_state_out = torch.stack([y, locked_new, af, dt], dim=-1)
    return fire, weapon_impulse, attack_state_out



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
    tick that switches the anchor weapon across pools compares two different
    scales and disagrees about whether ammo dropped.

    Accepts either entity-event-count spelling (``entity_event_counts`` is the
    dequantizer/schema output, ``entity_event_count`` the native wire field).
    Raises on an obs missing any required field — the caller asked for the
    engagement gate, so it must be computable.

    ``self_weapon_id`` is the one EXCEPTION, not a required field: the A27+
    pure-combat wire dropped it (schema v6, qnn/schema.py "the held weapon
    CONCEPT is retired"; QNNPolicy.act's attack-with call already falls back
    to anchor-weapon-blind selection on its absence, same reason). Missing here
    just means ``held_impulse`` comes back all-zero — impulse 0 is already
    "none/unknown" by convention (``self_weapon_id_to_impulse``), and
    :func:`held_ammo_for_impulse` treats impulse 0 as ``has_pool=False``, so
    the ammo-lockout override this feeds simply stays inert (matches what
    happens when a caller omits ammo_pools/held_impulse entirely) rather than
    raising for a signal this obs contract cannot supply.
    """
    from qnn.vocab import self_weapon_id_to_impulse
    missing = [k for k in ("entity_types", "entity_event_actions",
                           "self_ammo_pools") if k not in obs]
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
    if "self_weapon_id" in obs:
        held_impulse = self_weapon_id_to_impulse(
            torch.as_tensor(obs["self_weapon_id"]).reshape(rows).long())
    else:
        held_impulse = torch.zeros(rows, dtype=torch.int64)
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
# pool relevant to the anchor weapon hasn't decremented in over that many
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
    impulse: torch.Tensor,      # (B,) anchor weapon impulse 1..8
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(ammo, has_pool)`` — the ammo count relevant to the anchor weapon,
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
    enemy_present: torch.Tensor | None = None,   # (B,) bool — an actor is observed
    engaged_active: torch.Tensor | None = None,  # (B,) bool — target firing / non-self projectile
    idle_none_bias: "tuple[float, float]" = (0.0, 0.0),  # per-axis (fb,lr) idle stand-still bias
    idle_engagement_base: float = 0.5,   # E when an enemy is present but not active
    idle_cooldown_ticks: int = 20,       # 1s @ 20Hz engagement hold after combat
    ammo_pools: torch.Tensor | None = None,      # (B,4) [shells,nails,rockets,cells]
    held_impulse: torch.Tensor | None = None,    # (B,) anchor weapon impulse 1..8
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
        # MAXIMAL-RUN law: at expiry the held class is masked out, so a segment
        # is a maximal run (expiry means change) — the population the seg head
        # was trained on. The move.commit_recommit opt-out was deleted
        # 2026-08-26 (never set by any config).
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
    jump_threshold: float = 0.0,            # >0 = deterministic jump gate p_jump > τ
    threat: torch.Tensor | None = None,     # (B,) bool — incoming projectile present
    threat_break_hazard: float = 0.0,       # per-tick re-decision prob on threat rows
    enemy_present: torch.Tensor | None = None,   # (B,) bool — an actor is observed
    engaged_active: torch.Tensor | None = None,  # (B,) bool — target firing / non-self projectile
    idle_none_bias: "tuple[float, float]" = (0.0, 0.0),  # per-axis (fb,lr) idle stand-still bias
    idle_engagement_base: float = 0.5,      # E when an enemy is present but not active
    idle_cooldown_ticks: int = 20,          # 1s @ 20Hz engagement hold after combat
    ammo_pools: torch.Tensor | None = None,      # (B,4) [shells,nails,rockets,cells]
    held_impulse: torch.Tensor | None = None,    # (B,) anchor weapon impulse 1..8
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
#   * assemble_aim_prior / decode_look_from_polar —
#     the shared look decode (sampled magnitude × continuous direction + aim-prior
#     blend + expmap).
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




def decode_look_from_polar(
    mag_bin: "torch.Tensor | None" = None,
    dir_logits: "torch.Tensor | None" = None,
    mag_centers: "torch.Tensor | None" = None,
    dir_centers: "torch.Tensor | None" = None,
    z_prior: torch.Tensor | None = None,
    mag_gain: float = 0.0,
    turn_mag_scale: "float | torch.Tensor" = 1.0,
    hold_passthrough: bool = False,
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
    z-assembly, the aim-prior blend, and the expmap.

    ``theta_override`` / ``phi_override`` (the look_seg COMMIT decode path): when
    given, skip the polar mag_bin/dir readout and take the per-tick signed
    magnitude θ and direction φ DIRECTLY from the commit playout
    (``look_commit_step`` / ``look_commit_step_graph``). ``turn_mag_scale`` still
    applies to the supplied θ; the shared z-assembly + aim-prior blend + feet-aim
    expmap remainder is IDENTICAL to the polar path — the single source of
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
    if hold_passthrough:
        # HOLD PASS-THROUGH (default False = bit-identical): frames where the head
        # commanded an exact hold (θ == 0) emit an exact hold — the aim-prior
        # magnitude blend above otherwise converts every engaged hold into an
        # α·|aim-error| micro-correction, driving measured exact-hold occupancy to
        # zero (humans: ~14% of engaged frames). Turning frames are untouched.
        _hold_rows = (theta == 0).unsqueeze(-1)
        z = torch.where(_hold_rows, torch.zeros_like(z), z)
    look_vec = tangent_expmap(z)                                    # (..., 3) fwd,right,up
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
# attack_state was fully INERT from 2026-08-26 (every lane's consumer had been
# removed: lane 0 was the crest-gate countdown, lanes 1/2 the removed
# selection laws' anchor + accumulator, lane 3 the switch lockout's tick
# counter) until weapon.af_lockout restored a consumer the same day, reusing
# all 4 lanes for a NEW, simpler state:
#   lane 0 (y)            — seconds remaining in the lockout window
#   lane 1 (locked_weapon) — impulse (0..8) currently exempted from the mask
#   lane 2 (af_prev)      — last tick's self_arsenal_scalars attack_finished
#   lane 3 (dt)           — the real per-tick seconds af was observed to
#                            decay by, captured live and frozen once af hits 0
# See attack_with_decode's af_lockout branch. attack_with_decode is stateless
# (reads/writes nothing) UNLESS af_lockout is armed, in which case it both
# reads and writes this tensor every call. Every config that does not set
# weapon.af_lockout gets the old zero-effect passthrough, unchanged.
#
# The tensor is kept as wire scaffolding independent of whether any consumer
# is currently armed: removing it (and attack_rng, whose seed literals are
# MIRRORED in src/engine/common/qnn_onnx.c) is a deliberate wire/state-contract
# change, not a decode cleanup. The engine binds loopbacks generically from
# `state.loopback` metadata, so dropping the pair is mechanically safe when
# someone chooses to, but the lanes are live again now — do not repurpose
# them for anything else without checking af_lockout carriers first.
ATTACK_STATE_DIM = 4
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
    """Flat (graph-I/O) attack-state reset: ``(B, ATTACK_STATE_DIM)`` zeros + the
    ``(B,)`` int64 attack_rng. All-zeros is the correct off state for
    weapon.af_lockout too (y=0, locked_weapon=0 ⇒ no active lockout). attack_rng
    is INERT (kept for wire/state parity only) regardless."""
    s = torch.zeros(batch, ATTACK_STATE_DIM, dtype=torch.float32, device=device)
    rng = torch.full((batch,), int(rng_state) & _RNG_U32, dtype=torch.int64, device=device)
    return s, rng
