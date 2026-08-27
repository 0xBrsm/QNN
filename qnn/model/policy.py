"""Training-time wrapper around the model.

The compute graph (``Network``) lives in :mod:`qnn.model.network`. This
module owns the surrounding machinery: optimizers, loss shaping,
sampling, hidden-state lifecycle, and checkpoint I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import (
    MOVE_AXES,
    MOVE_AXIS_NAMES,
    MOVE_AXIS_CLASSES,
    MOVE_CLASS_NEG,
    MOVE_CLASS_NONE,
    MOVE_CLASS_POS,
)
from qnn.model.network import (
    ATTACK_HEAD,
    ATTACK_HEAD_SIZE,
    LOOK_HEAD,
    LOOK_HEAD_SIZE,
    JUMP_HEAD,
    MOVE_HEAD,
    MOVE_HEAD_SIZE,
    MOVE_HAZARD_HEAD,
    ModelConfig,
    Network,
    WEAPON_HEAD,
)
from qnn.model.weapon_head import weapon_index_from_id
from qnn.schema import WEAPON_HEAD_SIZE
from qnn.utils.device import configure_torch_runtime, resolve_torch_device
from qnn.utils.io import trusted_torch_load
from qnn.vocab import TOKEN_ACTOR


HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "target": 1.0,
    "move": 1.0,
    "look": 1.0,
    "attack": 1.0,
}

WEAPON_HEAD_CLASS_NAMES: Tuple[Tuple[int, str], ...] = (
    (0, "axe"),
    (1, "shotgun"),
    (2, "super_shotgun"),
    (3, "nailgun"),
    (4, "super_nailgun"),
    (5, "grenade_launcher"),
    (6, "rocket_launcher"),
    (7, "thunderbolt"),
)


@dataclass(slots=True)
class PolicyActionBatch:
    actions: Dict[str, np.ndarray]
    log_probs: torch.Tensor
    values: torch.Tensor
    entropies: Dict[str, torch.Tensor]
    next_hidden: torch.Tensor


def _null_side_channel_scope(*_args, **_kwargs):
    """Default side-channel provider — a no-op scope for the canonical model.

    Bench probes inject ``qnn.model.bench.side_channels.bench_side_channel_scope``
    via ``QNNPolicy(side_channel_provider=...)`` to enter the label-derived
    engagement_ema / target-supervision contexts. The canonical
    model needs none of those, so it uses this.
    """
    import contextlib
    return contextlib.nullcontext()


# There is NO default generation decode facade. The decode module is EXPLICIT
# state: eval/export orchestration injects it from the run's resolved decode
# config (``resolved.decode_module`` → ``policy._decode_mod``), and bare-policy
# entry points (analysis scripts) resolve it per run via
# ``qnn.diag.loader.resolve_decode_module``. ``QNNPolicy._decode()`` raises when
# nothing was injected — explicit-or-fail, never an arch guess.


# PER-ROW (per-lane) DECODE OVERRIDES — the aim-grid closed-loop enabler.
# ``act(per_row_decode=...)`` lets a SINGLE batched forward decode each row with
# a DIFFERENT operating point on these look levers, so many (gain/α/tremor,
# scenario) cells run in one B=N forward instead of one cold eval per swept
# value. The keys below are the four the aim grid sweeps (decode-config dotted
# names, matching a decode.json ``params`` block); each maps to the instance
# attribute the canonical (scalar/vector) decode path reads. A per-row array of
# length B overrides ONLY that lever, ONLY for this call — the forward is
# unchanged and every unlisted lever keeps its instance value. Absent the kwarg
# the instance scalars broadcast to all rows exactly as before (full back-compat).
_PER_ROW_DECODE_KEYS: dict[str, str] = {
    "look.aim_prior_gain": "look_aim_prior_gain",         # rotation-steering gain
    "look.aim_mag_gain": "look_aim_mag_gain",             # α (magnitude blend)
    "look.turn_mag_scale": "look_turn_mag_scale",         # native-θ dampener
    "look.aim_degrade_tremor_mag": "look_aim_degrade_tremor_mag",  # OU tremor mag
}


class QNNPolicy:
    """Feed-forward combat-objective model for BC."""

    def __init__(
        self,
        *,
        obs_dim: int,
        model: ModelConfig | None,
        jump_pos_weight: float,
        attack_focal_gamma: float,
        attack_focal_alpha: float,
        attack_distance_sigma: float,
        jump_distance_sigma: float,
        look_label_smoothing_sigma: float = 0.0,
        seed: int,
        device: str,
        model_factory: Callable[[int, ModelConfig], nn.Module] | None = None,
        side_channel_provider: Callable[..., Any] | None = None,
        graph: Any | None = None,
    ) -> None:
        """Construct a BC policy.

        ``graph`` (a :class:`qnn.model.graph.GraphSpec`) is the declarative
        assembly path: the inner module is built by
        ``qnn.model.graph.build_network`` and the spec is persisted in
        checkpoint meta (``model_graph``) so the checkpoint is fully
        self-describing — no factory rehydration on load. Mutually
        exclusive with ``model_factory``. ``model`` may be ``None`` when
        ``graph`` is given (derived via ``model_config_from_graph``).

        ``side_channel_provider(actions, masks) -> context manager`` is an
        optional hook the trainer enters around each forward pass. The
        canonical model passes ``None`` (no extra context). Bench probes pass
        ``qnn.model.bench.side_channels.bench_side_channel_scope`` to enter the
        label-derived engagement_ema / target-supervision contexts.
        Keeping it a plain callable keeps this class free of any bench import.

        ``model_factory`` is an optional override for the inner ``nn.Module``:
        when ``None`` (the default, used by all production BC training) the
        canonical ``Network`` is built from ``model``. Ablation
        runners (e.g. ``qnn.model.bench``) pass a factory that builds an
        alternate module — typically one that drops the encoder or GRU — but
        the factory must respect ``Network``'s forward contract
        so the canonical BC supervised loop can drive it unchanged.

        The injected module's flags should still be consistent with
        ``model`` (use_gru / use_weapon_head / etc.) since QNNPolicy's
        policy-layer logic — hidden-state shaping, weapon-switch heuristics,
        head-loss gating — reads from ``model``, not from the module.
        """
        if graph is not None:
            if model_factory is not None:
                raise ValueError("pass either graph or model_factory, not both")
            from qnn.model.graph import build_network, model_config_from_graph

            if model is None:
                model = model_config_from_graph(graph)
            model_factory = lambda _obs_dim, _cfg: build_network(_obs_dim, graph)  # noqa: E731
        self.graph = graph
        self.obs_dim = int(obs_dim)
        self._side_channel_provider = side_channel_provider or _null_side_channel_scope
        self.config = model
        self.d_model = int(model.d_model)
        self.use_gru = bool(model.use_gru and model.d_gru > 0)
        self.d_gru = int(model.d_gru) if self.use_gru else 0
        self.use_weapon_head = bool(model.use_weapon_head)
        self.look_bypass_gru = bool(model.look_bypass_gru and self.use_gru)
        self.weapon_context_from_obs = bool(model.weapon_context_from_obs)
        self.d_target = int(model.d_target)
        self.d_move = int(model.d_move)
        self.d_look = int(model.d_look)
        self.d_attack = int(model.d_attack)
        self.d_weapon = int(model.d_weapon)
        self.head_activation = model.head_activation
        # jump_pos_weight > 1.0 upweights the POS class on the move ud-axis CE
        # — direct imbalance fix for the rare jump-positive case (~4% pos rate).
        # Inverse-frequency reference: ~24× for 4% positive rate.
        self.jump_pos_weight = float(jump_pos_weight)
        # attack_focal_gamma > 0 swaps the attack-head BCE for focal BCE
        # (Lin et al. 2017): each frame's BCE is multiplied by
        # (1 - p_t)^gamma so easy examples contribute less gradient and
        # capacity flows to the borderline ready-frame "fire or wait?"
        # decisions. 0 = standard BCE.
        self.attack_focal_gamma = float(attack_focal_gamma)
        # attack_focal_alpha is Lin's per-class prefactor on the focal weight:
        # alpha_t = alpha on positives, (1 - alpha) on negatives. Active
        # only when attack_focal_gamma > 0. Default 0.5 is neutral (both
        # classes weighted equally up to a global scale). To run the Lin
        # recipe end-to-end set attack_pos_weight_override=1.0 alongside —
        # otherwise pos_weight stacks multiplicatively on the positive
        # branch and alpha loses its canonical class-fraction meaning.
        self.attack_focal_alpha = float(attack_focal_alpha)
        # input_mask is a training-time attribute (NOT a ModelConfig
        # field — checkpoint meta stays clean and the same ckpt can be
        # retrained either way). Trainer sets this to True after
        # construction when train.json.input_mask is true. Read by
        # ``_compute_head_losses_and_metrics`` to swap each head's
        # supervisory label from the raw demo button (usercmd) to the
        # engine outcome (act = max(usercmd − infeasibility_mask, 0));
        # for fire this collapses to "label = op_input bit 3".
        self.input_mask: bool = False
        # attack_label_shift is another training-time attribute (NOT a
        # ModelConfig field — keeps checkpoint meta clean and a single
        # ckpt can be retrained either way). Trainer sets this to True
        # after construction when train.json.attack_label_shift is true.
        # When on, ``_compute_head_losses_and_metrics`` swaps the attack
        # BCE *target* from ``actions["attack"]`` to
        # ``actions["attack_shifted"]`` (per-episode +1 op-frame OR). The
        # val precision/recall/F1 metrics still use the original
        # ``attack`` label so they remain comparable to the baseline.
        self.attack_label_shift: bool = False
        # attack_op_only — training-time toggle for the attack head's
        # noop-frame handling. When True (and input_mask is on), op=0
        # frames are MASKED OUT of the BCE loss (no gradient), pos_weight
        # is computed from op=1 counts only, and val precision/recall/F1
        # are scored on op=1 frames only. When False (default — preserves
        # historical behavior), op=0 frames stay in: their label is
        # forced to 0 (feasibility AND demo_press), they contribute
        # gradient pulling the model toward predicting 0 there, and they
        # show up in the metric as TN / occasional FP. Trainer sets this
        # after construction from train.json.attack_op_only.
        self.attack_op_only: bool = False
        # attack_distance_sigma > 0 enables Gaussian-shouldered BCE on the
        # attack head: per-frame BCE is multiplied by 1 at positives and by
        # 1 - exp(-d^2/(2*sigma^2)) at negatives, where d is distance (in
        # frames) to the nearest positive. Adjacent-to-press FPs cost
        # near-zero loss; far-from-press FPs cost full loss. Inference is
        # unchanged. See src/qnn/bc/heads/loss_shaping.py. 0 = standard BCE.
        self.attack_distance_sigma = float(attack_distance_sigma)
        # Same shoulder applied to the move ud-axis (jump) CE. Tuned
        # independently of fire because jump-press timing noise has a
        # different scale.
        self.jump_distance_sigma = float(jump_distance_sigma)
        # >0 swaps the binned look head's hard-argmin cross-entropy for a
        # distance-aware Gaussian soft-target CE (σ in radians). Smooths the
        # foveated sub-degree center bins whose hard distinctions aren't
        # learnable; 0 = standard one-hot CE. See qnn.model.look_bins.
        self.look_label_smoothing_sigma = float(look_label_smoothing_sigma)
        self.n_heads = int(model.n_heads)
        self.n_layers = int(model.n_layers)
        self.d_ffn = int(model.d_ffn)
        self.attn_dropout = float(model.attn_dropout)
        self.head_hidden = (self.d_gru + self.d_model) if self.use_gru else (2 * self.d_model)
        self.seed = int(seed)
        self.device_spec = resolve_torch_device(device)
        configure_torch_runtime(self.device_spec)
        self.device = self.device_spec.device
        self._rocm_inference_pad_batch = 0
        if self.device_spec.backend == "rocm":
            raw_pad_batch = os.environ.get("QNN_ROCM_INFERENCE_PAD_BATCH", "32").strip()
            try:
                self._rocm_inference_pad_batch = max(int(raw_pad_batch), 0)
            except ValueError:
                self._rocm_inference_pad_batch = 32

        torch.manual_seed(self.seed)
        if model_factory is None:
            self.model = Network(obs_dim=self.obs_dim, model=model).to(self.device)
        else:
            built = model_factory(self.obs_dim, model)
            if not isinstance(built, nn.Module):
                raise TypeError(
                    f"model_factory must return nn.Module, got {type(built).__name__}"
                )
            self.model = built.to(self.device)
        self.model.train()
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}
        # Lazily-built (9,7) weapon-trajectory table for the look aim-prior
        # decode (emit_actions sampled branch, pointer-bearing models).
        self._aim_prior_weapon_static: torch.Tensor | None = None
        # Aim-prior gain override. None → the decode facade's AIM_PRIOR_GAIN
        # (the baked decode contract); 0.0 → prior off (control arm in A/B
        # evals — set from the run config's eval_look_aim_prior_gain).
        self.look_aim_prior_gain: float | None = None
        # Global attack-propensity bias — the SAMPLED-mode skill knob. Attack is
        # decoded SAMPLED (temp-bernoulli on sigmoid(logit)), like look/move, so the
        # greedy threshold is inert; this additive logit bias shifts P(attack)=
        # sigmoid(logit+bias) uniformly across states (the s-slider). The angle-
        # conditioned attack_align gate is a SEPARATE bias (the all-humans cone fix).
        # Set from the decode config (attack.bias). See research/skill-curves.md §5.
        self.attack_bias: float = 0.0
        # a25 9-way attack-with operating point. attack_bias_vec is the legacy
        # JOINT vector retained for config compatibility. New fits use explicit
        # fire-only and selection-only vectors so rate trim cannot silently
        # steer weapon choice (the a26/a27 branch-drift failure).
        self.attack_bias_vec: "list | None" = None
        self.attack_fire_bias_vec: "list | None" = None
        self.weapon_preference_bias_vec: "list | None" = None
        self.weapon_switch_margin: float = 0.0
        # DISCHARGE-QUALITY gate ("crest-firing"): hold a commanded discharge up
        # to H ticks until crosshair→lead alignment crosses the per-weapon θ_w,
        # blind-fire at expiry (never cancel — the head's fire-rate invariant).
        # attack_crest_theta_vec: (8,) θ per impulse-1, hbw units (≤0 = OFF for
        # that weapon); attack_crest_hold_ticks: shared max hold H (0 = OFF
        # globally). Both default OFF = bit-identical. The countdown latch rides
        # the caller-threaded attack_state wire slot (lane 0; zeros = idle).
        self.attack_crest_theta_vec: "list | None" = None
        self.attack_crest_hold_ticks: int = 0
        # a25 MOVE COMMITMENT decode (segment head → semi-Markov generative):
        # opt-in via decode config param move.commitment. Requires a move_seg
        # head in the graph + caller-threaded move_commit_state.
        self.move_commitment: bool = False
        # Duration censoring-bias correction (move.commit_dur_tilt): per-axis
        # (fb, lr) bucket-index tilt applied inside move_commit_step; fit by
        # _move_seg_dur_calibration.py. (0, 0) = off, bit-identical.
        self.move_commit_dur_tilt: tuple[float, float] = (0.0, 0.0)
        # Gate B projectile interrupt opt-out (move.commit_interrupt): the
        # interrupt predates the seg head — with threat-conditioned durations
        # (mean dwell ~7 ticks => ~175 ms to the next natural decision point)
        # the forced resample may be redundant, and its both-axes k=0
        # changepoint spike (+0.41 vs human +0.03) breaks the rhythm gate.
        self.move_commit_interrupt: bool = True
        # Re-commit (move.commit_recommit): allow the segment head to re-commit
        # to the held class at expiry (no forced maximal-run switch), so a run
        # can be extended (walk/strafe/rest sustained across commitments).
        # False (default) = the maximal-run law, bit-identical to pre-knob.
        self.move_commit_recommit: bool = False
        # Engagement-gated idle stillness bias (move.idle_none_bias): per-axis
        # (fb,lr) additive bias on the seg head's `none` class, scaled by
        # (1 - engagement) so it damps pointless idle strafing when disengaged
        # and vanishes in combat. base = E when an enemy is merely present;
        # cooldown holds E=1 for N ticks after combat. (0,0) = off, bit-identical.
        self.move_idle_none_bias: "tuple[float, float]" = (0.0, 0.0)
        self.move_idle_engagement_base: float = 0.5
        self.move_idle_cooldown_ticks: int = 20
        # Threat-break hazard (move.threat_break_hazard): while an incoming
        # projectile is present, per-tick RE-DECISION probability λ inside
        # move_commit_step — the sustained human-shaped reactivity assist
        # (human hazard ratio 1.143; the head's own conditioning is ~1.01 —
        # research/move-head.md "Threat reactivity"). 0.0 = off, bit-identical.
        self.move_threat_break_hazard: float = 0.0
        # Jump confidence gate (jump.threshold): τ>0 replaces the AS-IS jump
        # posterior decode (Bernoulli sampled / argmax>0.5) with a DETERMINISTIC
        # gate — jump iff p_jump > τ. The land-jump head's posterior is diffuse
        # (p99 ~0.17), so sampling it scatters unmotivated hops; the gate fires
        # only the most-confident, context-motivated frames (and makes
        # deploy==offline, no jump RNG). 0.0 = off (bit-identical to pre-knob).
        self.jump_threshold: float = 0.0
        # Aim feed-forward override (rate term; kills the strafe trail).
        # None → the decode facade's AIM_FFWD_GAIN; 0.0 → off.
        self.look_aim_ffwd: float | None = None
        # Aim-prior magnitude-blend gain (α): the kept turn magnitude blends from
        # |z|=θ (0 = pure rotation, the deployed default — direction-preserving, aim
        # steers heading only) toward |z+z_prior| (1 = exactly the legacy vector-ADD),
        # and >1 overshoots into the super-human slew. ONE knob spans the whole axis;
        # there is no separate rotate toggle. See look-aim-decode.md.
        self.look_aim_mag_gain: float = 0.0
        # HEAD TURN-MAGNITUDE DAMPENER — an all-humans multiplicative scale on the
        # look head's OWN native turn magnitude |z|=θ (the decode-reachable head
        # term of the look over-turn). Default 1.0 = OFF = BIT-IDENTICAL (the decode
        # short-circuits the multiply). Applied inside decode_look_from_polar BEFORE
        # the aim-prior blend, so it dampens the head's conditional over-turn
        # (open-loop ~1.28× greedy / ~1.34× sampled vs aggregate human) WITHOUT
        # killing aim-prior placement steering (gain) or composing wrong with the
        # DOWN-half degrade (lag). It reaches ONLY the head term — the dominant
        # closed-loop covariate-shift over-turn (~1.47×) is training-side. Set from
        # the decode config (look.turn_mag_scale) by eval/export orchestration. Tune
        # via scripts/analysis/look_openloop_los_overturn.py (~0.78 → ~1.0× human,
        # open-loop). See look-aim-decode.md / look-head.md.
        self.look_turn_mag_scale: float = 1.0
        # HOLD-DRIFT (radians; 0.0 = OFF = bit-identical): on hold frames with a
        # live aim prior, emit eps toward the prior instead of a dead-zero turn.
        # The hold bin is a collect-time quantization artifact (humans sub-
        # hold_max drift ~30% of engaged ticks); a faithful hold rate + the
        # pure-rotation blend's |z|=0 preservation freezes pursuit (2026-07-05
        # lead-pointing investigation). Set from decode config
        # (look.hold_drift_eps). eps < hold_max keeps measured hold rates human.
        self.look_hold_drift_eps: float = 0.0
        # Hold pass-through (look.hold_passthrough): head-commanded exact holds
        # (θ==0) bypass the aim-prior magnitude blend, which otherwise converts
        # every engaged hold into an α·|aim-error| micro-correction (zero-hold
        # occupancy 0.000 vs human ~0.138). Default False = bit-identical.
        self.look_hold_passthrough: bool = False
        # Per-weapon VERTICAL aim authority (RL-splash feet-aiming). A (9,) per-
        # IMPULSE blend weight β∈[0,1] (0..8; 0 = OFF) for the feet-aim BLEND that
        # lerps the decoded vertical toward the feet-anchored lead point — the
        # AUTHORITY the rotation blend starves (look-aim-decode.md §12). None = OFF =
        # bit-identical. Fit per model by the decode-fit pipeline (look.weapon_pitch_gain).
        self.look_weapon_pitch_gain: "list | None" = None
        # Per-weapon (9,) downward pitch BIAS (degrees) added to the feet-aim target —
        # cancels the persistent ~1.5° static high-bias in RL fire aim (rockets over
        # the head → land behind) that β and feed-forward can't touch. None/[] = OFF.
        self.look_weapon_pitch_bias: "list | None" = None
        # Post-expmap pitch mode for RL feet-aim (gated β>0):
        #   "lock"  → hard-set fired elevation to the feet anchor (aimbot; collapses the
        #             head's vertical spread). "off" → no post-expmap op (β-blend only).
        #   "shift" → translate the head's OWN fired elevation DOWN by
        #             shift_strength·(origin→feet angle): moves center-mass tracking to
        #             the feet while PRESERVING the head's human-wide spread.
        # Default "lock" preserves the deployed rc1t. look_weapon_pitch_lock kept as a
        # back-compat alias (False → "off").
        self.look_weapon_pitch_lock: bool = True
        self.look_weapon_pitch_mode: str = "lock"
        self.look_weapon_pitch_shift_strength: float = 1.0
        # Hazard-discounted lead: cap the horizontal lead horizon at the expected
        # strafe-hold (20Hz frames; None/0 = OFF = linear lead). Linear over-leads
        # past the human dwell → rockets overshoot ("land behind"); the cap pulls
        # the lead back. Parametric first-order form of the v·ΣS(a+k)/S(a) discount.
        self.look_lead_hold_cap_frames: "float | None" = None
        # Radial (approach/retreat) hold cap (20Hz frames; None/0 = OFF = full lead).
        # Combat fb dwell — capping all directions with the combat hazard.
        self.look_lead_hold_cap_radial_frames: "float | None" = None
        # ──────────────────────────────────────────────────────────────────────
        # AIM DEGRADATION (the DOWN half of the mechanical-aim skill knob).
        # Decode-side, post-head, STATEFUL transforms on the look turn-delta that
        # LOOSEN per-weapon discharge alignment (raise intercept hbw) into the
        # sub-median human range — so the skill knob spans weak→elite. The UP half is the gain/α
        # lead-pursuit steering above; the bare-head floor already out-tracks a
        # median human, so playing DOWN needs ADDED mechanical error (human-like,
        # not robotic), NOT lower gain. All knobs default 0 = OFF = BIT-IDENTICAL
        # (no buffer is touched, no RNG drawn). The transform operates on the
        # tangent-space turn vector z = logmap(look) (radians), keyed per-row,
        # state reset on episode boundary (reset_mask). See look-aim-decode.md §11.
        #
        #   sluggish  — first-order low-pass (EMA) of the look-delta stream: a slow
        #     hand that UNDER-corrects. tau in frames; alpha=1/(1+tau). 0 = OFF.
        self.look_aim_degrade_sluggish_tau: float = 0.0
        #   lag       — fractional-frame DELAY of the turn-delta: the crosshair
        #     TRAILS (slow reaction). tau in frames (integer + fractional). 0 = OFF.
        self.look_aim_degrade_lag_frames: float = 0.0
        #   tremor    — correlated AR(1)/Ornstein-Uhlenbeck angular offset added to
        #     the look direction (a drifting unsteady hand). _mag = RMS amplitude in
        #     radians; _tau = correlation time in frames (rho=exp(-1/tau)). NOT white
        #     noise. mag 0 = OFF.
        self.look_aim_degrade_tremor_mag: float = 0.0
        self.look_aim_degrade_tremor_tau: float = 5.0
        #   jitter    — WHITE per-frame Gaussian angular noise on z. INCLUDED ONLY
        #     as the baseline to REJECT (it does loosen alignment/raise hbw but breaks heading-hold — the
        #     look "spin" lesson). _mag = per-frame SD in radians. 0 = OFF.
        self.look_aim_degrade_jitter_mag: float = 0.0
        # Per-row stateful buffers (lazily allocated when a mechanism is active;
        # reset on episode boundary). None = unallocated / OFF.
        self._aim_degrade_lp_state: torch.Tensor | None = None   # (R,2) EMA mem
        self._aim_degrade_lag_buf: torch.Tensor | None = None    # (R,L+1,2) ring
        self._aim_degrade_tremor_state: torch.Tensor | None = None  # (R,2) OU mem
        self._aim_degrade_rng: torch.Generator | None = None
        # ATTACK-WITH INTENT (7/08 operator ruling): the decode keys aim
        # geometry + per-weapon knobs on the model's MOST RECENT intent weapon
        # (the attack_with head's last committed nonzero choice), falling back
        # to held — "pre-aim for the weapon you're bringing up". −1 = no
        # intent yet (fresh row / episode boundary) → held.
        self._intent_impulse_buf: torch.Tensor | None = None      # (R,) long
        # Per-model weapon BAN (decode-config provenance; csv impulses 1..8).
        # Set by eval orchestration from the config's weapon_ban param. The a25
        # attack_with decode carries no ban gate — the a25 pins always emit
        # weapon_ban=[] — so this attribute is config provenance only; there is
        # no reachable decode consumer on the a25 shape.
        self.weapon_ban: tuple[int, ...] = ()
        # Optional release-candidate decode hook. Canonical policy decode stays
        # generic; bench/live candidates install their own postprocessor from
        # eval/export orchestration.
        self.decode_action_postprocess: Callable[
            ["QNNPolicy", Any, torch.Tensor, torch.Tensor], torch.Tensor
        ] | None = None
        # Resolved guard adapter from the decode config (the guard_module's
        # make_guard via qnn.model.decode_config), installed alongside
        # decode_action_postprocess by eval/export orchestration. ``act``'s shared
        # MOVE decode reads its ``projectile_release_mask(obs)`` (Gate B forced
        # fb/lr hold release) when the adapter defines it (dodge enabled) —
        # mirroring tools/export_onnx.ExportWrapper. None ⇒ no dodge term.
        self._regime_mod: Any | None = None
        # Generation decode FACADE (move/look-aim/weapon geometry). Injected from
        # the run's decode config (resolved.decode_module) by eval/export; None ⇒
        # ``_decode()`` lazily loads the default. This is the single seam by which
        # the core selects a generation's decode geometry — no scattered
        # ``from qnn.model.bench.aXX.decode import`` in the act path.
        self._decode_mod: Any | None = None

    def _decode(self) -> Any:
        """The resolved generation decode facade (see ``_decode_mod``).

        Returns the injected ``_decode_mod``. There is NO default: a policy with
        no injected decode module cannot decode — raise with the fix instead of
        guessing an arch (the a25/a24 silent-default coupling this replaced)."""
        # getattr (not self._decode_mod) so policies built via __new__ — e.g.
        # parity tests that skip __init__ — fail the same instructive way.
        mod = getattr(self, "_decode_mod", None)
        if mod is None:
            raise RuntimeError(
                "QNNPolicy has no decode module injected (policy._decode_mod is None) "
                "— there is no default decode facade. Resolve the run's decode config "
                "(qnn.model.decode_config.resolve_decode_config → resolved.decode_module) "
                "or, for a bare policy loaded from a run dir, resolve the arch "
                "explicitly via qnn.diag.loader.resolve_decode_module(run_dir, policy) "
                "before calling act()."
            )
        return mod

    def zero_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.d_gru), dtype=np.float32)

    def prepare_act_state(self, n_rows: int) -> dict[str, np.ndarray]:
        """The act() state-threading contract, derived from THIS model's shape
        — the single source consumers must use instead of hand-rolling lane
        widths or calling conventions (the 5-lane EnvState bug; the bare-act
        movearch failures).

        movearch models (no per-tick move head; the jump head owns
        land-vertical) REQUIRE the commitment decode — it is mandatory for the
        shape, not a knob — so it is enabled here. Every commitment model
        needs a caller-carried ``move_commit_state`` at COMMIT_STATE_DIM,
        initialized to the decode module's reset lanes (the ONNX
        state_loopback memset). act() mutates the passed array IN PLACE: keep
        the returned array alive across steps; fancy-indexed row batches are
        copies, so scatter them back after each call. Rows are per-episode —
        re-init a row's lanes on episode reset."""
        if (getattr(self.model, "move_head", None) is None
                and getattr(self.model, "jump_head", None) is not None):
            self.move_commitment = True
        kw: dict[str, np.ndarray] = {}
        if getattr(self, "move_commitment", False):
            from qnn.model.bench.a25.decode import commit_reset_lanes
            kw["move_commit_state"] = np.tile(
                np.asarray(commit_reset_lanes(), dtype=np.float32), (n_rows, 1))
        return kw

    def _tensor(self, value: np.ndarray | torch.Tensor | Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device == self.device and value.dtype == dtype:
                return value
            non_blocking = value.device.type == "cpu" and (
                not isinstance(self.device, torch.device) or self.device.type != "cpu"
            )
            return value.to(device=self.device, dtype=dtype, non_blocking=non_blocking)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _autocast(self):
        dtype_name = str(getattr(
            self, "autocast_dtype", os.environ.get("QNN_AUTOCAST_DTYPE", "fp32"),
        )).lower()
        if dtype_name == "fp32" or self.device.type not in {"cpu", "cuda"}:
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_name)
        if dtype is None or (self.device.type == "cpu" and dtype == torch.float16):
            return torch.amp.autocast(device_type=self.device.type, enabled=False)
        return torch.amp.autocast(device_type=self.device.type, dtype=dtype, enabled=True)

    def _class_weights_for_head(
        self,
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        head: str,
        size: int,
    ) -> torch.Tensor:
        source = class_weights.get(head)
        if source is None:
            return torch.ones((size,), dtype=torch.float32, device=self.device)
        cache = getattr(self, "_class_weights_cache", None)
        if cache is None:
            cache = {}
            self._class_weights_cache = cache
        key = (head, id(source))
        cached = cache.get(key)
        if cached is not None:
            return cached
        tensor = self._tensor(source, dtype=torch.float32)
        cache[key] = tensor
        return tensor

    @staticmethod
    def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            return logits.reshape(-1, logits.shape[-1])
        return logits

    @staticmethod
    def _flatten_targets(target: torch.Tensor) -> torch.Tensor:
        if target.ndim > 1:
            return target.reshape(-1)
        return target

    def _maybe_pad_obs_batch(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], int]:
        # `vel` matches the old `self_scalars` ndim semantics: (B, 3)
        # flat, (B, T, 3) sequence. Native obs replaced self_scalars
        # as a single-field key with per-field arrays.
        sample = obs_dict.get("vel")
        if sample is None:
            sample = obs_dict["self_scalars"]  # legacy fallback
        if sample.ndim != 2 or self._rocm_inference_pad_batch <= 0:
            return obs_dict, 0

        batch_size = int(sample.shape[0])
        target_batch = self._rocm_inference_pad_batch
        if batch_size == 0 or batch_size >= target_batch:
            return obs_dict, 0

        pad_rows = target_batch - batch_size
        padded_obs: Dict[str, torch.Tensor] = {}
        for key, value in obs_dict.items():
            pad_shape = (pad_rows, *value.shape[1:])
            pad_value = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            padded_obs[key] = torch.cat([value, pad_value], dim=0)
        return padded_obs, pad_rows

    @staticmethod
    def _pad_companion(
        tensor: torch.Tensor | None, pad_rows: int,
    ) -> torch.Tensor | None:
        """Zero-pad a companion tensor along dim 0 to match obs padding.

        Used after ``_maybe_pad_obs_batch`` to extend hidden state so
        callers that pass it on ROCm with small batches don't hit a B
        mismatch. Pass-through when the tensor is None or no padding
        was applied.
        """
        if tensor is None or pad_rows <= 0:
            return tensor
        pad_shape = (pad_rows, *tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

    def _inference_dequant(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the native→model-facing dequant chain (Self/Spatial/Entity) for
        inference. Idempotent (each stage short-circuits if its output key is
        present), so it's a no-op on already-dequantized obs. Stateless modules,
        instantiated once and cached."""
        chain = getattr(self, "_dequant_chain", None)
        if chain is None:
            from qnn.model.dequant import (
                SelfDequantizer, SpatialDequantizer, EntityDequantizer,
            )
            chain = (SelfDequantizer().eval(), SpatialDequantizer().eval(),
                     EntityDequantizer().eval())
            self._dequant_chain = chain
        self_dq, spatial_dq, entity_dq = chain
        # The lookup tables must live where _tensor put the obs indices.
        if next(entity_dq.buffers()).device != self.device:
            for stage in chain:
                stage.to(self.device)
        return entity_dq(spatial_dq(self_dq(obs)))

    def _obs_tensors_dequant(
        self, obs: Dict[str, np.ndarray | torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Tensorize an obs dict and run the inference dequant chain.

        Live obs (bridge / engine) arrives at native widths; BC training
        pre-dequantizes at preload, and the canonical Network carries its
        own dequant adapters, but bench-built Networks (e.g. full_4head)
        do not — so dequant here so act()/encode() work uniformly. The
        three dequantizers are idempotent (each short-circuits when its
        output key is already present), so this is a no-op on already-
        dequantized obs. Inference-only: training drives self.model
        directly, not _forward_tensors. Also used by the emit_actions
        aim-prior decode, which needs the dequanted entity_scalars_raw
        regardless of which representation the caller supplied.
        """
        obs_tensors: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            dtype = torch.float32
            if key.endswith("_id") or key.endswith("_ids"):
                dtype = torch.long
            elif key.endswith("_mask"):
                dtype = torch.bool
            obs_tensors[key] = self._tensor(value, dtype=dtype)
        return self._inference_dequant(obs_tensors)

    def _forward_tensors(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        eager_model: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(obs, dict):
            raise ValueError("Token policy expects dict observations")

        obs_tensors = self._obs_tensors_dequant(obs)

        hidden_tensor: torch.Tensor | None = None
        if self.use_gru and hidden is not None:
            hidden_tensor = self._tensor(hidden, dtype=torch.float32)

        # Use `vel` to detect flat-batch (B, 3) vs sequence (B, T, 3).
        # The legacy obs carried `self_scalars` (B, 17) here; the native
        # obs has per-field arrays, with vel matching the same ndim
        # semantics (2D flat, 3D sequence).
        sample = obs_tensors.get("vel")
        if sample is None:
            sample = obs_tensors["self_scalars"]  # legacy fallback
        if sample.ndim == 2:
            batch_size = int(sample.shape[0])
            padded_obs, pad_rows = self._maybe_pad_obs_batch(obs_tensors)
            padded_hidden = (
                self._pad_companion(hidden_tensor, pad_rows)
                if self.use_gru else hidden_tensor
            )
            model_call = self.model.forward if eager_model else self.model
            features, logits, values, next_hidden, target_logits = model_call(
                padded_obs, padded_hidden,
            )
            if pad_rows == 0:
                return features, logits, values, next_hidden, target_logits
            return (
                features[:batch_size],
                {head: tensor[:batch_size] for head, tensor in logits.items()},
                values[:batch_size],
                next_hidden[:batch_size],
                target_logits[:batch_size],
            )

        if sample.ndim != 3:
            raise ValueError("obs must be rank-2 or rank-3")
        reset_mask_tensor = None
        reset_ts: tuple[int, ...] | None = None
        if isinstance(masks, Mapping) and "reset_mask" in masks:
            reset_mask_tensor = self._tensor(masks["reset_mask"], dtype=torch.bool)
            raw_reset_ts = masks.get("reset_ts")
            if raw_reset_ts is not None:
                reset_ts = tuple(int(t) for t in raw_reset_ts)
        model_call = self.model.forward if eager_model else self.model
        return model_call(
            obs_tensors,
            hidden_tensor,
            reset_mask=reset_mask_tensor,
            reset_ts=reset_ts,
        )

    def _optimizer(self, name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        optimizer = self._optimizers.get(name)
        if optimizer is None:
            # bc_weight_decay > 0 (set by the BC trainer from config.weight_decay)
            # switches Adam → AdamW with decoupled weight decay. Default 0.0 =
            # plain Adam, byte-identical to prior behavior.
            wd = float(getattr(self, "bc_weight_decay", 0.0) or 0.0)
            if wd > 0.0:
                optimizer = torch.optim.AdamW(list(params), lr=lr, weight_decay=wd, fused=True)
            else:
                optimizer = torch.optim.Adam(list(params), lr=lr, fused=True)
            self._optimizers[name] = optimizer
        for group in optimizer.param_groups:
            group["lr"] = lr
        return optimizer

    def bc_zero_grad(self) -> None:
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.zero_grad()

    def bc_step(self) -> None:
        opt = self._optimizers.get("bc")
        if opt is not None:
            opt.step()

    # ──────────────────────────────────────────────────────────────────────────
    # AIM DEGRADATION (DOWN-half mechanical-skill knob) — stateful, post-head,
    # decode-side transforms on the look turn-delta. See __init__ for the knobs
    # and look-aim-decode.md §11. Operates in tangent space z = logmap(look)
    # (radians); converts back via expmap. State is per-row, reset on episode
    # boundary (reset_mask). Default all-OFF ⇒ this code path is never entered.
    def _lag_active(self) -> bool:
        """lag_frames may be a scalar OR a (9,) per-IMPULSE vector (per-weapon
        skill system, 7/08). Active when any component is > 0."""
        v = self.look_aim_degrade_lag_frames
        if isinstance(v, (list, tuple)):
            return any(float(x) > 0.0 for x in v)
        return float(v or 0.0) > 0.0

    def _aim_degrade_active(self) -> bool:
        _tr = self.look_aim_degrade_tremor_mag
        _tr_on = (any(float(x) > 0.0 for x in _tr)
                  if isinstance(_tr, (list, tuple)) else float(_tr or 0.0) > 0.0)
        return (
            float(self.look_aim_degrade_sluggish_tau or 0.0) > 0.0
            or self._lag_active()
            or _tr_on
            or float(self.look_aim_degrade_jitter_mag or 0.0) > 0.0
        )

    def _degrade_eff_impulse(self, obs, n_rows: int) -> "torch.Tensor":
        """The intent-keyed effective attack-with impulse per row — the SAME index
        gain/α use (last committed attack_with choice in ``_intent_impulse_buf``,
        held fallback via ``self_weapon_id_to_impulse``). Reset rows have already
        been zeroed in ``_aim_degrade_reset_rows`` (buffer set to -1 → held). This
        reads the intent buffer only (updated once, post-decode), so it returns the
        identical index whether the aim-prior block ran this tick or not."""
        from qnn.vocab import self_weapon_id_to_impulse as _w2i
        if isinstance(obs, Mapping) and "self_weapon_id" in obs:
            wid = self._obs_tensors_dequant(obs)["self_weapon_id"].long().reshape(-1)
        else:
            return torch.zeros(n_rows, dtype=torch.long)
        held = _w2i(wid)
        ib = self._intent_impulse_buf
        if ib is not None and ib.shape[0] == wid.shape[0]:
            eff = torch.where(ib >= 0, ib.to(held.device), held)
        else:
            eff = held
        return eff.clamp(0, 8)

    def _aim_degrade_reset_rows(self, reset_mask: "torch.Tensor | None") -> None:
        """Zero per-row degradation state on episode boundaries (reset_mask True)."""
        if reset_mask is None:
            return
        m = reset_mask.reshape(-1).bool()
        if self._aim_degrade_lp_state is not None:
            self._aim_degrade_lp_state[m] = 0.0
        if self._intent_impulse_buf is not None and self._intent_impulse_buf.shape[0] == m.shape[0]:
            self._intent_impulse_buf[m] = -1
        if self._aim_degrade_tremor_state is not None:
            self._aim_degrade_tremor_state[m] = 0.0
        if self._aim_degrade_lag_buf is not None:
            self._aim_degrade_lag_buf[m] = 0.0

    def _apply_aim_degrade(
        self, look: torch.Tensor, reset_mask: "torch.Tensor | None", obs=None,
        tremor_mag_row: "np.ndarray | None" = None,
    ) -> torch.Tensor:
        """Apply the active degradation mechanism(s) to the decoded look vector.

        look: (R, 3) view-frame unit-ish look. Returns the degraded (R, 3) unit
        look. Mechanisms compose on the tangent turn vector z (radians):
          sluggish (EMA low-pass) → lag (fractional-frame delay) → tremor (OU) /
          jitter (white). All are no-ops at their OFF default. The order matters
          little at the magnitudes we sweep; documented for reproducibility.

        ``tremor_mag_row`` (per-LANE override, length R) supersedes the instance
        scalar ``look_aim_degrade_tremor_mag`` for the tremor stage only, so a
        batched eval can drive a different tremor magnitude per lane. None ⇒ the
        instance scalar (bit-identical to callers that never pass it).
        """
        from qnn.model.look_bins import tangent_logmap, tangent_expmap

        dev = look.device
        R = int(look.shape[0])
        # If the batch row-count changed (eval pads/grows envs) drop stale state —
        # row identity is only stable within a fixed-width batch. The eval keeps a
        # fixed N via eval_batched_forward, so this fires only on (re)alloc.
        def _fit(buf: "torch.Tensor | None", shape) -> "torch.Tensor | None":
            if buf is None or tuple(buf.shape) != tuple(shape):
                return None
            return buf

        z = tangent_logmap(look)                                  # (R,2) radians
        # Head-commanded exact holds (θ==0 → z==0 after decode/pass-through):
        # captured BEFORE any degrade stage so noise-type degraders can respect
        # rests — a low-skill hand is noisy while AIMING but still rests
        # (research/human-band.md: hold occupancy dominates the look channel).
        _hold_rows = (z.abs().sum(dim=-1, keepdim=True) < 1e-9)

        # 1) SLUGGISH — first-order low-pass (EMA) of the turn-delta stream.
        tau_lp = float(self.look_aim_degrade_sluggish_tau or 0.0)
        if tau_lp > 0.0:
            alpha = 1.0 / (1.0 + tau_lp)
            st = _fit(self._aim_degrade_lp_state, (R, 2))
            if st is None:
                st = torch.zeros((R, 2), device=dev, dtype=z.dtype)
            self._aim_degrade_lp_state = st
            self._aim_degrade_reset_rows(reset_mask)
            st = self._aim_degrade_lp_state
            z = alpha * z + (1.0 - alpha) * st
            st.copy_(z)

        # 2) LAG — fractional-frame delay (the crosshair TRAILS). Linear interp
        # between floor(L) and ceil(L) frames back in a per-row ring buffer. L may
        # be a scalar OR a (9,) per-IMPULSE vector (per-weapon skill system,
        # 7/08): the ring is sized to the MAX lag; each row's fractional depth is
        # gathered from its intent-keyed effective impulse (the same index gain/α
        # use). The scalar branch is bit-identical to the pre-vector code.
        _lag_spec = self.look_aim_degrade_lag_frames
        _lag_is_vec = isinstance(_lag_spec, (list, tuple))
        if _lag_is_vec:
            L_vec = torch.as_tensor([float(x) for x in _lag_spec],
                                    dtype=z.dtype, device=dev)          # (9,)
            L_max = float(L_vec.max().item())
        else:
            L_vec = None
            L_max = float(_lag_spec or 0.0)
        if L_max > 0.0:
            depth = int(np.ceil(L_max)) + 1
            buf = _fit(self._aim_degrade_lag_buf, (R, depth, 2))
            if buf is None:
                # init the history to the CURRENT turn (so the first frames don't
                # snap from zero — a transient that would dirty the metric).
                buf = z.unsqueeze(1).repeat(1, depth, 1).contiguous()
            self._aim_degrade_lag_buf = buf
            self._aim_degrade_reset_rows(reset_mask)
            buf = self._aim_degrade_lag_buf
            # shift history: newest at index 0
            buf[:, 1:, :] = buf[:, :-1, :].clone()
            buf[:, 0, :] = z
            if _lag_is_vec:
                # per-row fractional gather from the intent-keyed effective impulse
                eff_imp = self._degrade_eff_impulse(obs, R).to(dev)     # (R,)
                L_row = L_vec[eff_imp]                                  # (R,)
                i0 = torch.floor(L_row).long().clamp_(0, depth - 1)     # (R,)
                i1 = torch.clamp(i0 + 1, max=depth - 1)                 # (R,)
                frac = (L_row - i0.to(z.dtype)).unsqueeze(-1)           # (R,1)
                z0 = torch.gather(buf, 1, i0.view(R, 1, 1).expand(R, 1, 2)).squeeze(1)
                z1 = torch.gather(buf, 1, i1.view(R, 1, 1).expand(R, 1, 2)).squeeze(1)
                z = (1.0 - frac) * z0 + frac * z1
            else:
                L = L_max
                i0 = int(np.floor(L)); i1 = min(i0 + 1, depth - 1)
                frac = L - i0
                z = (1.0 - frac) * buf[:, i0, :] + frac * buf[:, i1, :]

        # 3) TREMOR — correlated AR(1)/OU angular offset (drifting unsteady hand).
        # Per-LANE override (tremor_mag_row) → a (R,1) magnitude tensor that scales
        # the innovation per row; else the instance scalar. The OU recursion is
        # otherwise identical (τ/ρ shared; magnitude is the only per-lane lever).
        if tremor_mag_row is not None:
            mag_tr_t = torch.as_tensor(
                tremor_mag_row, dtype=z.dtype, device=dev).reshape(-1, 1)  # (R,1)
            mag_tr = float(mag_tr_t.max())
        else:
            _tr_spec = self.look_aim_degrade_tremor_mag
            if isinstance(_tr_spec, (list, tuple)):
                # per-IMPULSE tremor vector (the a25rc3c per-weapon DOWN-band
                # fix — the emit's mean-scalar smeared every weapon's tremor
                # onto every other): magnitude gathered from the intent-keyed
                # effective impulse, the same index gain/α/lag use.
                T_vec = torch.as_tensor([float(x) for x in _tr_spec],
                                        dtype=z.dtype, device=dev)       # (9,)
                mag_tr = float(T_vec.max().item())
                mag_tr_t = None
                if mag_tr > 0.0:
                    eff_imp = self._degrade_eff_impulse(obs, R).to(dev)  # (R,)
                    mag_tr_t = T_vec[eff_imp].reshape(-1, 1)             # (R,1)
            else:
                mag_tr_t = None
                mag_tr = float(_tr_spec or 0.0)
        if mag_tr > 0.0:
            tau_tr = max(float(self.look_aim_degrade_tremor_tau or 1.0), 1e-3)
            rho = float(np.exp(-1.0 / tau_tr))
            _sqrt = float(np.sqrt(max(1.0 - rho * rho, 1e-9)))
            innov = (mag_tr_t * _sqrt) if mag_tr_t is not None else (mag_tr * _sqrt)
            if self._aim_degrade_rng is None:
                self._aim_degrade_rng = torch.Generator(device="cpu")
                self._aim_degrade_rng.manual_seed(0xA1A1_BEEF)
            st = _fit(self._aim_degrade_tremor_state, (R, 2))
            if st is None:
                st = torch.zeros((R, 2), device=dev, dtype=z.dtype)
            self._aim_degrade_tremor_state = st
            self._aim_degrade_reset_rows(reset_mask)
            st = self._aim_degrade_tremor_state
            eps = torch.randn((R, 2), generator=self._aim_degrade_rng).to(dev, z.dtype)
            new = rho * st + innov * eps
            st.copy_(new)
            # HOLD-GATED: the OU state keeps evolving (the hand keeps drifting)
            # but hold frames express no offset — rests stay exact.
            z = torch.where(_hold_rows, z, z + new)

        # 4) JITTER — white per-frame Gaussian (REJECT baseline; breaks heading-hold).
        mag_jit = float(self.look_aim_degrade_jitter_mag or 0.0)
        if mag_jit > 0.0:
            if self._aim_degrade_rng is None:
                self._aim_degrade_rng = torch.Generator(device="cpu")
                self._aim_degrade_rng.manual_seed(0xA1A1_BEEF)
            eps = torch.randn((R, 2), generator=self._aim_degrade_rng).to(dev, z.dtype)
            z = z + mag_jit * eps

        return tangent_expmap(z)

    def encode(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, _, _, next_hidden, _ = self._forward_tensors(obs, hidden=hidden)
        return (
            features.detach().cpu().numpy().astype(np.float32),
            next_hidden.detach().cpu().numpy().astype(np.float32),
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        hidden: np.ndarray | None = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            features, logits_t, values_t, next_hidden, _ = self._forward_tensors(obs, hidden=hidden)
        logits = {head: tensor.detach().cpu().numpy().astype(np.float32) for head, tensor in logits_t.items()}
        values = values_t.detach().cpu().numpy().astype(np.float32)
        features_np = features.detach().cpu().numpy().astype(np.float32)
        return logits, values, next_hidden.detach().cpu().numpy().astype(np.float32), features_np

    def _resolve_per_row_decode(
        self,
        spec: "Mapping[str, Any] | None",
        n_rows: int,
    ) -> "dict[str, np.ndarray]":
        """Validate + normalize a ``per_row_decode`` spec into per-row arrays.

        ``spec`` maps a supported decode-config key (see ``_PER_ROW_DECODE_KEYS``)
        to a length-``n_rows`` array-like (or a scalar, which broadcasts). Returns
        a dict keyed by the SAME dotted names → float32 arrays of length
        ``n_rows``. Empty/None ⇒ ``{}`` (the instance-scalar path, bit-identical
        to callers that never pass the kwarg). Device-free: callers move each
        array onto the decode tensor's device at its use site.
        """
        if not spec:
            return {}
        unknown = set(spec) - set(_PER_ROW_DECODE_KEYS)
        if unknown:
            raise ValueError(
                f"per_row_decode: unsupported key(s) {sorted(unknown)}; "
                f"supported: {sorted(_PER_ROW_DECODE_KEYS)}")
        out: dict[str, np.ndarray] = {}
        for key, value in spec.items():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size == 1:
                arr = np.full((n_rows,), float(arr[0]), dtype=np.float32)
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"per_row_decode[{key!r}] has length {arr.shape[0]}, "
                    f"expected {n_rows} (one value per lane)")
            out[key] = arr
        return out

    def _aim_prior_geometry(
        self,
        obs_model: Dict[str, torch.Tensor],
        target_logits: torch.Tensor,
        rows: int,
        device: torch.device,
        reset_mask: torch.Tensor | None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Shared aim-prior geometry off the target pointer — the SINGLE eager
        computation of ``aim_prior_tangent_ffwd`` consumed by the sampled look
        decode (aim-prior blend / feet-aim pitch) AND the attack crest gate
        (per-row hbw). Returns ``(z_aim, z_rate, feet_elev, origin_elev,
        aim_range, eff_imp)``.

        intent-weapon keying: effective impulse = last committed attack_with
        choice (buffer, updated after the weapon decode — i.e. most recent PRIOR
        decision), else held. The intent buffer is reset on episode boundaries
        via ``reset_mask`` (the pre-captured ``_aim_reset_mask``; ``masks`` is
        dropped at act() entry)."""
        from qnn.bc.weapon_physics import build_model_weapon_scalars
        _dmod = self._decode()
        if self._aim_prior_weapon_static is None:
            self._aim_prior_weapon_static = torch.from_numpy(
                build_model_weapon_scalars()).float().to(device)
        # Re-run the (idempotent, cached) dequant already applied by the caller:
        # live obs carries native keys, not entity_scalars_raw.
        dq = obs_model
        esc = dq["entity_scalars_raw"].float()
        esc = esc.reshape(rows, *esc.shape[-2:])                      # (R, N, S)
        etypes = dq["entity_types"].long().reshape(rows, -1)          # (R, N)
        wid = dq["self_weapon_id"].long().reshape(-1)
        from qnn.vocab import self_weapon_id_to_impulse as _w2i
        _held_imp = _w2i(wid)
        _ib = self._intent_impulse_buf
        if _ib is None or _ib.shape[0] != wid.shape[0]:
            _ib = torch.full_like(wid, -1)
            self._intent_impulse_buf = _ib
        if reset_mask is not None:
            _rm = reset_mask.reshape(-1)
            if _rm.shape[0] == _ib.shape[0]:
                _ib[_rm] = -1
        eff_imp = torch.where(_ib >= 0, _ib, _held_imp)
        wid_eff = torch.where(eff_imp > 0, eff_imp + 2, wid)
        _hold_cap = (None if not self.look_lead_hold_cap_frames
                     else float(self.look_lead_hold_cap_frames) * _dmod._TICK_DT_MODULE)
        _hold_cap_rad = (None if not self.look_lead_hold_cap_radial_frames
                         else float(self.look_lead_hold_cap_radial_frames) * _dmod._TICK_DT_MODULE)
        z_aim, z_rate, feet_elev, origin_elev, aim_range = _dmod.aim_prior_tangent_ffwd(
            esc, etypes, wid_eff,
            target_logits.reshape(rows, -1),
            self._aim_prior_weapon_static,
            lead_hold_cap=_hold_cap,
            lead_hold_cap_radial=_hold_cap_rad,
        )
        return z_aim, z_rate, feet_elev, origin_elev, aim_range, eff_imp

    def act(
        self,
        obs: np.ndarray | torch.Tensor | Dict[str, np.ndarray | torch.Tensor],
        *,
        mode: str,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: np.ndarray | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        row_generators: Any | None = None,
        sample_temperatures: Mapping[str, float] | None = None,
        diag_log_path: str | Path | None = None,
        move_commit_state: np.ndarray | None = None,
        attack_state: np.ndarray | None = None,
        attack_rng: np.ndarray | None = None,
        per_row_decode: "Mapping[str, Any] | None" = None,
        rl_extras: bool = False,
    ) -> "PolicyActionBatch | tuple[PolicyActionBatch, Dict[str, Any]]":
        """Emit engine actions from a forward pass.

        Output dict shape matches the engine's action contract (see
        qnn.actions.ActionLabels):
          move   : (B, 3) float in [-1, 1] — view-relative wishvel/maxspeed.
                   Categorical mode argmaxes/samples each axis to {-1, 0, +1};
                   continuous mode passes the regression output through clamp.
                   Up axis is 0 (no jump head).
          look   : (B, 3) float — look_predict unit vector from the head.
          fire   : (B,)   int   — 0/1 from sigmoid(logit) threshold or bernoulli.
          weapon : (B,)   int   — engine weapon byte 1..8 (or 0 = no switch).

        log_probs / values / entropies are placeholders for shape compatibility
        with the action batch consumer; greedy/sampled eval doesn't read them.

        When *diag_log_path* is set, append one JSONL record per call with
        target/look/move/fire internals — for distribution-shift debugging in
        live eval.

        ``rl_extras=True`` returns ``(batch, extras)`` with the raw logits and
        value-head features PPO needs without a second forward. Head-specific
        sampling and engine encoding live in ``qnn.ppo.distributions``.
        """
        # Capture reset_mask BEFORE dropping masks — the stateful aim-degradation
        # decode (DOWN-half skill knob) resets its per-row buffers on episode
        # boundaries. (_forward_tensors also reads it for the GRU; this is a
        # cheap second read, only used when a degradation knob is active.)
        _aim_reset_mask = None
        if isinstance(masks, Mapping) and "reset_mask" in masks:
            _aim_reset_mask = self._tensor(masks["reset_mask"], dtype=torch.bool)
        del masks, generator
        with torch.inference_mode():
            obs_model = self._obs_tensors_dequant(obs)
            features, logits, _, next_hidden, target_logits = self._forward_tensors(
                obs_model, hidden=hidden,
            )

        sample_mode = str(mode).lower()
        if sample_mode not in ("greedy", "sampled"):
            raise ValueError(f"Unsupported policy mode: {mode}")
        temps = dict(sample_temperatures or {})

        # ---- move ----
        # 3 categorical axes (fb, lr, ud), each a 3-class softmax over
        # {neg, none, pos}.  Greedy = argmax per axis; sampled = categorical
        # per axis.  Decoded engine value per axis = class - 1, i.e. {-1, 0, +1}.
        # movearch (full_movearch) models have NO per-tick move head: fb/lr
        # come from the seg commit, ud from the jump head (land) and the seg
        # head's water-ud commit (deep water). The commit path is mandatory.
        _is_movearch = MOVE_HEAD not in logits and JUMP_HEAD in logits
        if _is_movearch:
            move_logits = None
            n_rows = int(logits[JUMP_HEAD].reshape(-1).shape[0])
            if not (self.move_commitment and "move_seg" in logits
                    and move_commit_state is not None):
                raise RuntimeError(
                    "movearch model requires the commitment decode: enable "
                    "move.commitment and thread move_commit_state "
                    "(COMMIT_STATE_DIM lanes)")
        else:
            move_logits = logits[MOVE_HEAD].reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
            n_rows = int(move_logits.shape[0])
        if (self.move_commitment and "move_seg" in logits
                and move_commit_state is not None):
            # a25 COMMITMENT decode: fb/lr from the segment head's sampled
            # (class, duration) commitments (expiry/Gate-B-interrupt/reset →
            # resample; held class masked at expiry only — an interrupt is a
            # re-decision); ud from the cross-gen base decode.
            # Caller threads move_commit_state (R,4) [fb_cls,fb_rem,lr_cls,
            # lr_rem], cls<0 = unset — MUTATED IN PLACE like the swb state.
            from qnn.model.bench.a25.decode import move_commit_step
            from qnn.model.decode import decode_move_axes
            # Slice axes, don't blind-reshape: a water_ud seg head emits
            # (N, 3, JOINT) and reshape(n_rows, 2, -1) would silently garble
            # fb/lr into (N, 2, 45). The commit decode is fb/lr-only; the
            # water-ud axis gets its own decode gate later.
            seg_all = logits["move_seg"]
            if seg_all.dim() != 3:
                seg_all = seg_all.reshape(n_rows, 2, -1)
            seg_logits = seg_all[:, :2, :]
            projectile_release = None
            if (self.move_commit_interrupt
                    and self._regime_mod is not None
                    and hasattr(self._regime_mod, "projectile_release_mask")
                    and isinstance(obs, Mapping)):
                projectile_release = self._regime_mod.projectile_release_mask(
                    self._obs_tensors_dequant(obs))
            commit_t = self._tensor(move_commit_state, dtype=torch.float32).reshape(n_rows, -1).clone()
            water = None
            if _is_movearch:
                # Deep water (waist/submerged) = the swim-feasible context the
                # water-ud axis trained on; ground + feet-wet = jump context;
                # airborne = neither (training rewrote/masked those frames).
                # NOTE: approximates QNN_PackInputMask bits 5-7 from
                # self_movement_id — verify exact parity against the engine
                # pack before the closed-loop A/B.
                _mv_id = self._tensor(obs["self_movement_id"], dtype=torch.long).reshape(-1)
                from qnn.engine_norm import (
                    MOVEMENT_GROUND, MOVEMENT_WATER_LOW, MOVEMENT_WATER_MID,
                    MOVEMENT_WATER_HIGH,
                )
                water = (_mv_id == MOVEMENT_WATER_MID) | (_mv_id == MOVEMENT_WATER_HIGH)
                # ANTI-POGO: training's bit7 is a pmove eval of a FRESH press
                # (qnn_qwd_collect.c QNN_QwdEvalPmoveJump — a held button2
                # cannot re-trigger). The bot's own previous ud-press is
                # carried in commit lane [7], so the live gate replicates the
                # trained population: ground/feet-wet AND not-already-pressing.
                # (Corpus check: this closes the 13% ground-but-infeasible
                # disagreement class; the ~10% airborne-but-feasible class —
                # landing timing inside the pmove tick — stays unrecoverable
                # from movement_id and under-fires slightly.)
                _prev_press = (commit_t[:, 7] > 0.5) if commit_t.shape[1] > 7                     else torch.zeros_like(water)
                jump_ctx = ((_mv_id == MOVEMENT_GROUND)
                            | (_mv_id == MOVEMENT_WATER_LOW)) & ~_prev_press
            # External (world) signals for the commitment decode. Derived by
            # the SHARED decode helpers over the dequantized obs — the same
            # call the ONNX ExportWrapper makes, so eval and deploy read the
            # identical quantity in the identical units (decode's
            # projectile_rel_raw has the history of the two divergent copies).
            _threat = None
            if self.move_threat_break_hazard > 0.0:
                from qnn.model.bench.a25.decode import move_threat_signal
                _threat = move_threat_signal(obs_model, n_rows)
            # Engagement signals (external) for the idle stillness bias.
            _enemy_present = _engaged_active = None
            _ammo_pools = _held_impulse = None
            if self.move_idle_none_bias != (0.0, 0.0):
                from qnn.model.bench.a25.decode import move_engagement_signals
                (_enemy_present, _engaged_active,
                 _ammo_pools, _held_impulse) = move_engagement_signals(
                    obs_model, n_rows)
            commit_out = move_commit_step(
                seg_all if _is_movearch else seg_logits, commit_t,
                release=projectile_release,
                greedy=(sample_mode == "greedy"),
                row_generators=row_generators,
                dur_tilt=self.move_commit_dur_tilt,
                water=water,
                threat=_threat,
                threat_break_hazard=float(self.move_threat_break_hazard),
                recommit=bool(self.move_commit_recommit),
                enemy_present=_enemy_present,
                engaged_active=_engaged_active,
                idle_none_bias=self.move_idle_none_bias,
                idle_engagement_base=float(self.move_idle_engagement_base),
                idle_cooldown_ticks=int(self.move_idle_cooldown_ticks),
                ammo_pools=_ammo_pools, held_impulse=_held_impulse)
            fblr = commit_out[:, :2]
            _cs = np.asarray(move_commit_state)
            _cs[...] = commit_t.detach().cpu().numpy().reshape(_cs.shape).astype(_cs.dtype)
            if _is_movearch:
                # ud: jump head's posterior on jump context; water-ud commit in
                # deep water; none elsewhere. jump.threshold τ>0 replaces the
                # AS-IS sampled/argmax decode with a DETERMINISTIC confidence
                # gate (jump iff p_jump > τ) — only the most-confident,
                # context-motivated frames fire, and deploy==offline (no RNG).
                jl = logits[JUMP_HEAD].reshape(-1)
                p_jump = torch.sigmoid(jl)
                if self.jump_threshold > 0.0:
                    jump_fire = p_jump > float(self.jump_threshold)
                elif sample_mode == "greedy":
                    jump_fire = p_jump > 0.5
                elif row_generators is None:
                    jump_fire = torch.rand_like(p_jump) < p_jump
                else:
                    # One vectorized per-row uniform draw (row_uniforms fast branch
                    # for BatchedRNG; bit-identical per-row torch.rand for a
                    # generator list) — replaces the O(rows) per-generator loop.
                    from qnn.model.decode import row_uniforms
                    u = row_uniforms(row_generators, 1, p_jump.device)[:, 0].reshape(p_jump.shape)
                    jump_fire = u < p_jump
                move_classes = torch.ones(n_rows, MOVE_AXES, dtype=torch.long,
                                          device=fblr.device)
                move_classes[:, :2] = fblr
                ud = torch.ones(n_rows, dtype=torch.long, device=fblr.device)
                ud = torch.where(jump_fire,
                                 torch.full_like(ud, MOVE_CLASS_POS), ud)
                if commit_out.shape[1] > 2:
                    ud = torch.where(water, commit_out[:, 2], ud)
                move_classes[:, 2] = ud
                # Guard shim: downstream guards argmax per-tick move logits
                # for splash-direction checks. movearch has no move head —
                # one-hot the EMITTED commit classes (strictly better: the
                # guard sees what the bot actually does this tick).
                _move_logits_guard = torch.nn.functional.one_hot(
                    move_classes, MOVE_AXIS_CLASSES).to(torch.float32) * 10.0
            else:
                move_classes = decode_move_axes(
                    move_logits, sampled=(sample_mode != "greedy"),
                    temperature=float(temps.get("move", 1.0)),
                    row_generators=row_generators,
                ).clone()
                move_classes[:, :2] = fblr
        else:
            # Basic per-axis readout (cross-gen base, qnn.model.decode):
            # argmax (greedy) / categorical-sample (sampled). This is the ONLY
            # non-commitment move decode — the a24 sticky/hazard/switch-back/
            # stop-onset stack is retired (an a25 model runs the commitment
            # decode above or this plain per-axis readout; no sticky state).
            from qnn.model.decode import decode_move_axes
            move_classes = decode_move_axes(
                move_logits, sampled=(sample_mode != "greedy"),
                temperature=float(temps.get("move", 1.0)),
                row_generators=row_generators,
            )
        move = (move_classes.float() - float(MOVE_CLASS_NONE))             # {-1, 0, +1} per axis

        # PER-LANE decode overrides (aim-grid enabler): resolve once now that
        # n_rows is known. ``_pr`` maps a dotted decode key → a float32 (n_rows,)
        # array; empty ⇒ the instance-scalar path (bit-identical to no-kwarg).
        _pr = self._resolve_per_row_decode(per_row_decode, n_rows)

        # crest-gate alignment signal (attack decode below): filled by the
        # sampled-look aim block when it runs this tick; the attack block
        # computes it itself otherwise (greedy look / all aim knobs at 0).
        _crest_z_err = None
        _crest_z_rate = None
        _crest_aim_range = None

        # ---- look ----
        # Greedy: look_predict is the head's deterministic readout (already
        # unit-normalized inside the model; clamp guards fp noise).  For the
        # polar look head this is the argmax reconstruction — which collapses
        # the hold-dominated turn distribution to a near-frozen view.  Sampled
        # (HYBRID decode — must mirror tools/export_onnx.ExportWrapper so offline
        # eval matches the deployed ONNX graph): sample the MAGNITUDE bin (human
        # turn-size distribution) but take DIRECTION as the continuous circular
        # mean of the direction softmax, NOT a sampled dir bin.  Per-frame
        # direction sampling + 16-bin quantization break the learned heading-hold
        # and cause the live "spin"; the circular mean restores near-human
        # heading persistence with the magnitude distribution untouched.  See
        # src/docs/look-head.md §3 and qnn.model.look_bins.
        if sample_mode == "sampled" and "_look_mag_logits" in logits:
            from qnn.model.look_bins import DIR_CENTERS, MAG_CENTERS, N_DIR, N_MAG
            _dmod = self._decode()
            assemble_aim_prior, decode_look_from_polar = (
                _dmod.assemble_aim_prior, _dmod.decode_look_from_polar)
            t_look = max(float(temps.get("look", 1.0)), 1e-6)
            mag_logits = logits["_look_mag_logits"].reshape(-1, N_MAG + 1)
            dir_logits = logits["_look_dir_logits"].reshape(-1, N_DIR)
            # MAGNITUDE: seeded categorical sample with temperature — the offline
            # policy's own sampler. This legitimately differs from the export's
            # in-graph Gumbel-argmax and is NOT shared. The continuous direction +
            # z-assembly + aim-prior blend + expmap ARE shared, via
            # the decode facade's decode_look_from_polar (single source of truth,
            # co-decoded with tools/export_onnx.ExportWrapper). See
            # src/docs/look-head.md §3.
            mag_bin = self._categorical_sample(
                F.softmax(mag_logits / t_look, dim=-1), row_generators)
            # AIM PRIOR (pointer-bearing models only): assemble the PRE-SCALED
            # blend z_prior = gain·z_aim + ffwd·z_rate, zero on no-enemy frames.
            # The gain SOURCES (instance overrides / AIM_*_GAIN defaults) are the
            # offline path's own; the blend math itself lives in the shared
            # decode. See the facade's lead_aim module and src/docs/look-head.md §5.
            AIM_FFWD_GAIN, AIM_PRIOR_GAIN = _dmod.AIM_FFWD_GAIN, _dmod.AIM_PRIOR_GAIN
            # gain / alpha accept a (9,) per-IMPULSE vector (per-weapon skill
            # system, 7/08): resolved per row below via the canonical
            # self_weapon_id_to_impulse lookup (mirrors weapon_pitch_gain).
            # PER-LANE gain override wins: a (R,1) tensor drives assemble_aim_prior
            # per row (the SAME shape the per-impulse vector path builds), so each
            # lane blends at its own gain in one forward.
            _gain_rows = None
            if "look.aim_prior_gain" in _pr:
                _gain_rows = torch.as_tensor(
                    _pr["look.aim_prior_gain"], dtype=torch.float32,
                    device=mag_logits.device).reshape(-1, 1)
                _gain_is_vec = False
                _aim_gain = float(_gain_rows.max())   # activation test
            else:
                _gain_spec = (AIM_PRIOR_GAIN if self.look_aim_prior_gain is None
                              else self.look_aim_prior_gain)
                _gain_is_vec = isinstance(_gain_spec, (list, tuple))
                _aim_gain = (max(float(g) for g in _gain_spec) if _gain_is_vec
                             else float(_gain_spec))   # activation test only when vec
            _aim_ffwd = (AIM_FFWD_GAIN if self.look_aim_ffwd is None
                         else float(self.look_aim_ffwd))
            # Per-weapon VERTICAL pitch authority (RL feet-aiming): active when any
            # per-impulse gain is nonzero. It needs z_aim (the UNSCALED error), so
            # the aim geometry is computed whenever the prior OR the pitch term is on.
            _pitch_gain = self.look_weapon_pitch_gain
            _pitch_active = (_pitch_gain is not None
                             and any(float(g) != 0.0 for g in _pitch_gain))
            z_prior = None
            pitch_correction = None
            _feet_elev = None
            _origin_elev = None
            if (((_aim_gain > 0.0 or _aim_ffwd > 0.0) or _pitch_active)
                    and target_logits is not None
                    and getattr(self.model, "_has_target_pointer", False)):
                # SHARED geometry (see _aim_prior_geometry: intent-keyed lead
                # tangent + anchors + pooled range); the crest gate below reuses
                # z_aim/range from this tick instead of recomputing.
                (z_aim, z_rate, _feet_elev, _origin_elev, _crest_aim_range,
                 eff_imp) = self._aim_prior_geometry(
                    obs_model, target_logits, mag_bin.shape[0],
                    mag_logits.device, _aim_reset_mask)
                _crest_z_err = z_aim
                _crest_z_rate = z_rate
                if _aim_gain > 0.0 or _aim_ffwd > 0.0:
                    if _gain_rows is not None:
                        # per-LANE override (already (R,1)) — no intent keying.
                        z_prior = assemble_aim_prior(z_aim, z_rate, _gain_rows, _aim_ffwd)
                    elif _gain_is_vec:
                        _gt = torch.as_tensor([float(g) for g in _gain_spec],
                                              dtype=torch.float32, device=mag_logits.device)
                        _gain_rows_vec = _gt[eff_imp].unsqueeze(-1)    # (R,1) intent-keyed
                        z_prior = assemble_aim_prior(z_aim, z_rate, _gain_rows_vec, _aim_ffwd)
                    else:
                        z_prior = assemble_aim_prior(z_aim, z_rate, _aim_gain, _aim_ffwd)
                if _pitch_active:
                    pg = torch.as_tensor([float(g) for g in _pitch_gain],
                                         dtype=torch.float32, device=mag_logits.device)
                    _pb = self.look_weapon_pitch_bias
                    pb = (torch.as_tensor([float(b) for b in _pb], dtype=torch.float32,
                                          device=mag_logits.device) if _pb else None)
                    pitch_correction = _dmod.assemble_pitch_correction(
                        z_aim, pg, wid_eff, weapon_pitch_bias=pb)
            if "look.aim_mag_gain" in _pr:
                # PER-LANE α override: (R,) tensor → decode's mag_gain per-row
                # branch (rows at 0 are exact no-ops).
                _alpha_val = torch.as_tensor(
                    _pr["look.aim_mag_gain"], dtype=torch.float32,
                    device=mag_logits.device).reshape(-1)
                _alpha_spec = None
            else:
                _alpha_spec = self.look_aim_mag_gain or 0.0
            if _alpha_spec is None:
                pass
            elif isinstance(_alpha_spec, (list, tuple)):
                # per-impulse alpha -> per-row tensor (decode's tensor path is
                # exact-no-op at 0 rows, so vec alpha is bit-compatible).
                # Intent-keyed when the aim block computed eff_imp this tick.
                _at = torch.as_tensor([float(a) for a in _alpha_spec], dtype=torch.float32,
                                      device=mag_logits.device)
                try:
                    _imp_a = eff_imp
                except NameError:
                    from qnn.vocab import self_weapon_id_to_impulse as _w2i_a
                    _imp_a = _w2i_a(self._obs_tensors_dequant(obs)["self_weapon_id"].long().reshape(-1))
                _alpha_val = _at[_imp_a]
            else:
                _alpha_val = float(_alpha_spec)
            look = decode_look_from_polar(
                mag_bin, dir_logits,
                MAG_CENTERS.to(mag_logits.device), DIR_CENTERS.to(dir_logits.device),
                z_prior,
                mag_gain=_alpha_val,
                turn_mag_scale=(
                    torch.as_tensor(_pr["look.turn_mag_scale"], dtype=torch.float32,
                                    device=mag_logits.device)
                    if "look.turn_mag_scale" in _pr
                    else float(self.look_turn_mag_scale
                               if self.look_turn_mag_scale is not None else 1.0)),
                hold_drift_eps=float(self.look_hold_drift_eps or 0.0),
                hold_passthrough=bool(self.look_hold_passthrough),
                pitch_correction=pitch_correction,
                feet_elev=(_feet_elev if (_pitch_active and self.look_weapon_pitch_mode != "off") else None),
                origin_elev=(_origin_elev if _pitch_active else None),
                pitch_mode=self.look_weapon_pitch_mode,
                shift_strength=float(self.look_weapon_pitch_shift_strength),
            ).reshape(-1, LOOK_HEAD_SIZE)
        else:
            look = torch.clamp(logits[LOOK_HEAD].reshape(-1, LOOK_HEAD_SIZE), -1.0, 1.0)

        # ---- aim DEGRADATION (DOWN-half skill knob; default OFF = bit-identical)
        # PER-LANE tremor override forces the degrade path (some lane may want
        # tremor even when the instance default is OFF).
        _tremor_row = _pr.get("look.aim_degrade_tremor_mag")
        if self._aim_degrade_active() or (
                _tremor_row is not None and float(np.max(_tremor_row)) > 0.0):
            look = self._apply_aim_degrade(
                look, _aim_reset_mask, obs=obs, tremor_mag_row=_tremor_row)

        # ---- fire (+ weapon, when joint) ----
        # a25 attack-with: ONE 9-way head owns both decisions. Detected by the
        # weapon slot's logit width (8+1); there is no attack head in that graph.
        # Greedy joint decode (qnn.model.bench.a25.decode) — deterministic argmax
        # preserves the rocket-jump coupling; align bias + hard guards compose on
        # the class-0 logit / a veto mask. Stateless: no hold-tail, no sticky gate.
        _wl_raw = logits.get(WEAPON_HEAD) if self.use_weapon_head else None
        is_attack_with = (
            _wl_raw is not None and int(_wl_raw.shape[-1]) == WEAPON_HEAD_SIZE + 1
            and isinstance(obs, Mapping) and "self_weapon_id" in obs
        )
        if is_attack_with:
            from qnn.model.bench.a25.decode import (
                attack_with_decode, attack_with_marginal_logit,
            )
            logits9 = _wl_raw.reshape(-1, WEAPON_HEAD_SIZE + 1)
            _bias_vec = (None if self.attack_bias_vec is None
                         else self._tensor(self.attack_bias_vec, dtype=torch.float32).reshape(-1))
            _fire_bias_vec = (None if self.attack_fire_bias_vec is None
                              else self._tensor(self.attack_fire_bias_vec,
                                                dtype=torch.float32).reshape(-1))
            _preference_bias_vec = (
                None if self.weapon_preference_bias_vec is None
                else self._tensor(self.weapon_preference_bias_vec,
                                  dtype=torch.float32).reshape(-1))
            # ONE shared decode (qnn.model.bench.a25.decode.attack_with_decode) —
            # the SAME call the ONNX ExportWrapper bakes, so the offline and the
            # deployed attack decode (held-impulse + guard align/veto + per-weapon
            # operating point + crest gate) cannot skew. No decode logic inline
            # here. attack_state (the crest countdown latch's wire slot) is
            # caller-threaded and MUTATED IN PLACE like move_commit_state.
            _crest_theta = (None if self.attack_crest_theta_vec is None
                            else self._tensor(self.attack_crest_theta_vec,
                                              dtype=torch.float32).reshape(-1))
            _att_state_t = (None if attack_state is None
                            else self._tensor(attack_state, dtype=torch.float32
                                              ).reshape(int(logits9.shape[0]), -1))
            fire, weapon_impulse, _att_state_out = attack_with_decode(
                logits9,
                self._tensor(obs["self_weapon_id"], dtype=torch.long),
                self._obs_tensors_dequant(obs),
                _move_logits_guard if _is_movearch else logits["move"],
                guard=self._regime_mod,
                attack_bias=float(self.attack_bias), bias_vec=_bias_vec,
                fire_bias_vec=_fire_bias_vec,
                preference_bias_vec=_preference_bias_vec,
                switch_margin=float(self.weapon_switch_margin),
                crest_theta_vec=_crest_theta,
                crest_hold_ticks=int(self.attack_crest_hold_ticks or 0),
                aim_z_err=_crest_z_err, aim_range=_crest_aim_range,
                aim_z_rate=_crest_z_rate,
                attack_state=_att_state_t)
            if attack_state is not None and _att_state_out is not None:
                _as = np.asarray(attack_state)
                _as[...] = (_att_state_out.detach().cpu().numpy()
                            .reshape(_as.shape).astype(_as.dtype))
            attack_logit = attack_with_marginal_logit(logits9)   # diag readout
            fire_prob = torch.sigmoid(attack_logit)              # diag readout
        else:
            # SPLIT attack head (no attack_with): basic cross-gen readout
            # (qnn.model.decode.decode_attack_bit) — sigmoid(logit+bias) > 0.5
            # greedy / temperature-Bernoulli sampled. The a24 stateful attack
            # decode (hold-tail + threaded xorshift rng + scalar threshold) is
            # retired with the a24 arch; the attack_state / attack_rng kwargs
            # are accepted for caller-state-threading compatibility and pass
            # through untouched (the a25 wire keeps the slots for parity).
            from qnn.model.decode import decode_attack_bit
            attack_logit = logits[ATTACK_HEAD].reshape(-1)
            fire_prob = torch.sigmoid(attack_logit + float(self.attack_bias))   # diag-log readout
            fire = decode_attack_bit(
                attack_logit + float(self.attack_bias),
                sampled=(sample_mode == "sampled"),
                temperature=float(temps.get("attack", 1.0)),
                row_generators=row_generators,
            )
            if self.decode_action_postprocess is not None:
                fire = self.decode_action_postprocess(self, obs, move_classes, fire)

        # ---- weapon ----
        # attack_with decoded its impulse jointly above. SPLIT weapon heads
        # (8-way + sticky confidence/margin gate + weapon_ban) are an a24-only
        # protocol, retired with the arch: a split-head model emits the plain
        # per-row argmax impulse (no gate, no hysteresis, no ban).
        if not is_attack_with:
            weapon_impulse = torch.ones(int(move.shape[0]), dtype=torch.long, device=move.device)
        if (not is_attack_with and self.use_weapon_head and WEAPON_HEAD in logits
                and isinstance(obs, Mapping) and "self_weapon_id" in obs):
            weapon_logits = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            weapon_impulse = weapon_logits.argmax(dim=-1).to(torch.long) + 1

        # record the attack_with intent for NEXT tick's aim keying (nonzero
        # decided impulse = committed intent; 0 = keep previous intent).
        if self._intent_impulse_buf is not None:
            _wi = weapon_impulse.reshape(-1).long().to(self._intent_impulse_buf.device)
            if _wi.shape[0] == self._intent_impulse_buf.shape[0]:
                _upd = _wi > 0
                self._intent_impulse_buf[_upd] = _wi[_upd]

        actions = {
            "move":   move.detach().cpu().numpy().astype(np.float32),
            "look":   look.detach().cpu().numpy().astype(np.float32),
            "attack":   fire.detach().cpu().numpy().astype(np.int64),
            "weapon": weapon_impulse.detach().cpu().numpy().astype(np.int64),
        }

        if diag_log_path is not None:
            self._append_act_diagnostics(
                diag_log_path, obs, logits, target_logits, attack_logit, fire_prob,
                actions,
            )

        zero = torch.zeros(int(move.shape[0]), dtype=torch.float32, device=move.device)
        entropies = {
            "move":   zero.clone(),
            "look":   zero.clone(),
            "attack":   zero.clone(),
            "weapon": zero.clone(),
        }
        batch = PolicyActionBatch(
            actions=actions,
            log_probs=zero.clone(),
            values=zero.clone(),
            entropies=entropies,
            next_hidden=next_hidden.detach(),
        )
        if not rl_extras:
            return batch
        return batch, {
            "logits": logits,
            "features": features,
            "target_logits": target_logits,
        }

    @staticmethod
    def _append_act_diagnostics(
        path: str | Path,
        obs: Any,
        logits: Dict[str, torch.Tensor],
        target_logits: torch.Tensor,
        attack_logit: torch.Tensor,
        fire_prob: torch.Tensor,
        actions: Dict[str, np.ndarray],
    ) -> None:
        """Append per-row JSONL records with target/look/move/fire internals."""
        import json as _json
        from qnn.vocab import TOKEN_ACTOR  # local import to keep top-level clean

        # Episode boundaries can produce empty act batches; there is nothing to log.
        if target_logits.shape[0] == 0:
            return
        # Pointer-Off models, such as full_4head, produce (B, 0) target logits.
        # Keep the rest of the diagnostic row and mark target internals as absent.
        has_target = target_logits.numel() > 0

        def _np(x: torch.Tensor) -> np.ndarray:
            return x.detach().cpu().numpy()

        # Mask invalid target indices (TargetPointer pre-masks with -1e9 — pick
        # them up so we can tell "no actor present" from "low confidence".
        tl = _np(target_logits.reshape(target_logits.shape[0], -1)) if has_target else None  # (B, N)
        # entity_types lets us see how many actor indices actually contain a bot
        et = obs.get("entity_types") if isinstance(obs, dict) else None
        actor_counts = None
        # attack_finished input sanity (raw wire + dequantized) — added during
        # the A1 live-collapse hunt to verify the live cooldown value per tick.
        af_raw = af_dq = None
        if isinstance(obs, dict):
            _afr = obs.get("attack_finished")
            if _afr is not None:
                af_raw = np.asarray(_afr).reshape(-1).astype(float).tolist()
            try:
                from qnn.model.dequant import SelfDequantizer
                _dq = SelfDequantizer()({k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()})
                _afd = _dq.get("self_arsenal_scalars")
                if _afd is not None:
                    af_dq = _afd.reshape(-1, _afd.shape[-1])[:, 0].detach().cpu().numpy().astype(float).tolist()
            except Exception:
                af_dq = None
        if et is not None:
            et_np = np.asarray(et)
            actor_counts = (et_np == TOKEN_ACTOR).sum(axis=-1).reshape(-1).tolist()

        # Soft attention probs (with masked indices ~0 thanks to -1e9 logit)
        if has_target:
            tl_t = target_logits.reshape(target_logits.shape[0], -1)
            probs = torch.softmax(tl_t, dim=-1).detach().cpu().numpy()
            argmax_idx = probs.argmax(axis=-1).tolist()
            max_prob = probs.max(axis=-1).tolist()
        else:
            argmax_idx = max_prob = None
        # Entropy in nats
        if has_target:
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)
        else:
            ent = None

        look_prior = _np(logits["_look_prior"]).reshape(-1, 3) if "_look_prior" in logits else None
        look_delta = _np(logits["_look_delta"]).reshape(-1, 3) if "_look_delta" in logits else None
        look_predict = _np(logits[LOOK_HEAD]).reshape(-1, 3)

        # Alignment scalar that feeds the attack head
        if look_prior is not None:
            align = (look_predict * look_prior).sum(axis=-1)
        else:
            align = np.full(look_predict.shape[0], np.nan, dtype=np.float32)

        move_logits_np = (
            _np(logits[MOVE_HEAD]).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES).tolist()
            if MOVE_HEAD in logits else None
        )
        move_prob_np = (
            _np(F.softmax(logits[MOVE_HEAD], dim=-1)).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES).tolist()
            if MOVE_HEAD in logits else None
        )
        attack_logit_np = _np(attack_logit).reshape(-1).tolist()
        fire_prob_np = _np(fire_prob).reshape(-1).tolist()

        # Weapon internals — to diagnose closed-loop switching (echo-lock): the
        # head's DESIRED weapon (argmax) + confidence vs the HELD weapon, and the
        # decided impulse. If desired == held nearly always in closed loop, the
        # head is echoing its held-weapon input (never wants to switch).
        weapon_logits_np = weapon_desired = weapon_conf = held_class = None
        if WEAPON_HEAD in logits and isinstance(obs, dict) and "self_weapon_id" in obs:
            # width-agnostic: 8 (split weapon head) or 9 (a25 attack-with joint)
            wl = logits[WEAPON_HEAD]
            wl = wl.reshape(-1, wl.shape[-1])
            wp = torch.softmax(wl, dim=-1)
            weapon_logits_np = _np(wl).tolist()
            weapon_desired = wp.argmax(dim=-1).detach().cpu().numpy().reshape(-1).tolist()
            weapon_conf = wp.max(dim=-1).values.detach().cpu().numpy().reshape(-1).tolist()
            held = weapon_index_from_id(
                torch.as_tensor(np.asarray(obs["self_weapon_id"])).long().reshape(-1))
            held_class = held.detach().cpu().numpy().reshape(-1).tolist()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for i in range(look_predict.shape[0]):
                rec = {
                    "row": i,
                    "actor_count": (actor_counts[i] if actor_counts is not None else None),
                    "attack_finished_raw": (af_raw[i] if af_raw is not None and i < len(af_raw) else None),
                    "attack_finished_dq": (af_dq[i] if af_dq is not None and i < len(af_dq) else None),
                    "target": ({
                        "argmax_idx": int(argmax_idx[i]),
                        "max_prob": float(max_prob[i]),
                        "entropy_nats": float(ent[i]),
                        "logits": tl[i].tolist(),
                    } if has_target else None),
                    "look": {
                        "pred": look_predict[i].tolist(),
                        "base": (look_prior[i].tolist() if look_prior is not None else None),
                        "base_mag": (
                            float(np.linalg.norm(look_prior[i])) if look_prior is not None else None
                        ),
                        "delta": (look_delta[i].tolist() if look_delta is not None else None),
                        "delta_mag": (
                            float(np.linalg.norm(look_delta[i])) if look_delta is not None else None
                        ),
                        "alignment": float(align[i]),
                    },
                    "move": {
                        "axes":   list(MOVE_AXIS_NAMES),
                        "logits": (move_logits_np[i] if move_logits_np is not None else None),
                        "prob":   (move_prob_np[i]   if move_prob_np   is not None else None),
                        "action": actions["move"][i].tolist(),
                    },
                    "attack": {
                        "logit": float(attack_logit_np[i]),
                        "prob":  float(fire_prob_np[i]),
                        "action": int(actions["attack"][i]),
                    },
                    "weapon": ({
                        "desired": int(weapon_desired[i]),     # argmax class 0..7
                        "held":    int(held_class[i]),         # held weapon class 0..7
                        "conf":    float(weapon_conf[i]),      # softmax prob of desired
                        "decided": int(actions["weapon"][i]),  # impulse 1..8 (or 0)
                        "logits":  weapon_logits_np[i],
                    } if weapon_desired is not None else None),
                }
                f.write(_json.dumps(rec) + "\n")

    @staticmethod
    def _categorical_sample(
        probs: torch.Tensor,
        row_generators: Any | None,
    ) -> torch.Tensor:
        """Sample one class index per row from probs (B, K).

        Thin instance-side alias for the cross-gen base sampler
        (qnn.model.decode.categorical_sample) — retained as the look decode's
        magnitude sampler seam (analysis harnesses monkeypatch it).
        """
        from qnn.model.decode import categorical_sample
        return categorical_sample(probs, row_generators)

    @staticmethod
    def _emit_look_tangent_sums(
        metrics: dict,
        pred_u: torch.Tensor,
        tgt_u: torch.Tensor,
    ) -> None:
        """Emit additive tangent-space sufficient stats for look_r2 / look_ewa_deg.

        See qnn.bc.look_metrics. Log-map both unit vectors to the 2D rotation
        vector at (1,0,0) (||z|| = turn angle), then emit raw sums combined at
        epoch end. Emitted under the ``looksum_`` raw-sum prefix + n_look_valid.
        """
        def _logmap(u: torch.Tensor) -> torch.Tensor:
            # atan2(|yz|, x), not arccos(x) — the fp16 near-1 cosine trap
            # (research/look-head.md 7/06 root cause); mirrors tangent_logmap.
            yz = u[:, 1:3]                                                # (n,2)
            yz_norm = torch.linalg.vector_norm(yz, dim=-1)               # (n,)
            theta = torch.atan2(yz_norm, u[:, 0])                         # (n,)
            scale = torch.where(yz_norm > 1e-8, theta / yz_norm.clamp(min=1e-8),
                                torch.zeros_like(theta))
            return yz * scale[:, None]

        z = _logmap(tgt_u)
        zh = _logmap(pred_u)
        diff = z - zh
        theta = torch.linalg.vector_norm(z, dim=-1)                      # target turn
        cos = (pred_u * tgt_u).sum(dim=-1).clamp(-1.0, 1.0)
        dtheta = torch.arccos(cos)                                       # angular err
        metrics["n_look_valid"] = torch.tensor(float(z.shape[0]), device=z.device)
        metrics["looksum_ssres"] = (diff * diff).sum().detach()
        metrics["looksum_z0"] = z[:, 0].sum().detach()
        metrics["looksum_z1"] = z[:, 1].sum().detach()
        metrics["looksum_zz"] = (z * z).sum().detach()
        metrics["looksum_ewnum"] = (theta * dtheta).sum().detach()
        metrics["looksum_ewden"] = theta.sum().detach()

    @staticmethod
    def _bernoulli_sample(
        prob: torch.Tensor,
        row_generators: Any | None,
    ) -> torch.Tensor:
        """Thin instance-side alias for qnn.model.decode.bernoulli_sample."""
        from qnn.model.decode import bernoulli_sample
        return bernoulli_sample(prob, row_generators)

    def _mean_real_losses(
        self,
        losses: list[torch.Tensor],
        loss_is_real: list[bool | torch.Tensor],
    ) -> torch.Tensor:
        """Average active head losses without synchronizing device flags.

        Most heads are statically active and retain the cheap Python-bool path.
        Sparse target/move/look heads provide scalar device booleans so an
        all-masked microbatch does not force the GPU queue to synchronize merely
        to decide which losses enter the mean.
        """
        if not losses:
            return torch.zeros((), device=self.device)
        if all(isinstance(active, bool) for active in loss_is_real):
            real = [loss for loss, active in zip(losses, loss_is_real) if active]
            return (
                torch.stack(real).mean()
                if real
                else torch.zeros((), device=self.device)
            )

        stacked = torch.stack(losses)
        # torch.full fills on device (scalar rides the kernel args);
        # new_tensor(float) is a synchronizing pageable host→device copy.
        weights = torch.stack([
            active.to(device=stacked.device, dtype=stacked.dtype)
            if isinstance(active, torch.Tensor)
            else torch.full((), float(active), device=stacked.device, dtype=stacked.dtype)
            for active in loss_is_real
        ])
        return (stacked * weights).sum() / weights.sum().clamp(min=1)

    def _compute_head_losses_and_metrics(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        compute_metrics: bool = True,
        target_logits: torch.Tensor | None = None,
        obs: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[bool | torch.Tensor], Dict[str, torch.Tensor | int | float]]:
        weights_map = head_loss_weights or HEAD_LOSS_WEIGHTS
        losses: list[torch.Tensor] = []
        loss_is_real: list[bool | torch.Tensor] = []
        metrics: Dict[str, torch.Tensor | int | float] = {}
        accuracy_components: list[torch.Tensor] = []
        valid_flat = valid_mask.reshape(-1).bool() if valid_mask is not None else None

        # input_mask: pure per-axis feasibility — "would the engine
        # accept this axis press right now?".  When true, each head's
        # label is the engine OUTCOME = feasibility AND demo press;
        # the trainer recombines them per axis (see the per-head label
        # rewrite blocks below).  Bit layout of actions["input_mask"]
        # (packed by QNN_PackInputMask in the collector):
        #
        #   bit 0 = attack feasibility   (W_Attack would fire if button0=1)
        #   bit 1 = forward neg feasibility
        #   bit 2 = forward pos feasibility   (pmove always processes
        #                                      fmove → bits 1-2 both 1
        #                                      whenever alive)
        #   bit 3 = side neg feasibility
        #   bit 4 = side pos feasibility
        #   bit 5 = up neg feasibility   (swim down, water only)
        #   bit 6 = up pos feasibility   (swim up,   water only)
        #   bit 7 = jump feasibility     (ground-jump would fire if
        #                                 button2=1; depends on onground
        #                                 + anti-pogo + alive)
        #
        # Requires the recollected corpus that carries
        # act_input_mask.npy — hard-fails if missing.
        input_mask_on = bool(self.input_mask)
        if input_mask_on and "input_mask" not in actions:
            raise RuntimeError(
                "input_mask=True but actions['input_mask'] is absent. "
                "Recollect the corpus on a post-input_mask branch — the "
                "engine emits act_input_mask.npy as part of every shard."
            )
        input_mask_flat: torch.Tensor | None = None
        if input_mask_on:
            input_mask_flat = self._tensor(
                actions["input_mask"], dtype=torch.long).reshape(-1)

        target_loss_weight = float(weights_map.get("target", 1.0))
        if (
            target_logits is not None
            and "target_probs" in actions
            and target_loss_weight != 0.0
        ):
            target_flat = self._flatten_logits(target_logits)
            dist_t = self._tensor(actions["target_probs"], dtype=torch.float32)
            if dist_t.ndim == 3:
                dist_t = dist_t.reshape(-1, dist_t.shape[-1])
            # dist_t[:, 0] = NO_TARGET; dist_t[:, 1:] = idx probabilities.
            present = (1.0 - dist_t[:, 0]).clamp(min=0.0)
            idx_dist = dist_t[:, 1:]
            # No in-policy gate. Engagement filtering is the caller's job
            # via segment_mask (e.g. `{"act.target": {"$ne": 0}}`). Frames
            # with present=0 are already dropped at the dataset level; what
            # remains contributes to target loss and metrics in proportion
            # to its present value, with the clamp(min=1e-6) in the
            # renormalize keeping the divide numerically safe.
            valid = (
                valid_flat
                if valid_flat is not None
                else torch.ones_like(present, dtype=torch.bool)
            )
            aux_is_real = valid.any()
            # Present-weighted soft CE: -sum_s p_idx * log_softmax(logits).
            # Computed dense over ALL rows with the valid mask folded into the
            # per-frame weight — boolean indexing (`x[valid]`) calls nonzero(),
            # which blocks on a device→host sync; profiling showed those syncs
            # dominating the training step. Weighted-sum over masked rows is
            # value-identical: invalid rows carry weight exactly 0.
            valid_f = valid.to(target_flat.dtype)
            log_probs_all = F.log_softmax(target_flat, dim=-1)
            idx_target_all = idx_dist / present.clamp(min=1e-6).unsqueeze(-1)
            per_frame_ce_all = -(idx_target_all * log_probs_all).sum(dim=-1)
            present_w = present * valid_f
            aux_ce = (present_w * per_frame_ce_all).sum() / present_w.sum().clamp(min=1e-6)
            losses.append(aux_ce * target_loss_weight)
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                aux_has_rows = bool(aux_is_real.item())
                metrics["loss_target"] = aux_ce.detach()
                metrics["target_present_mean"] = present.mean().detach()
                if aux_has_rows:
                    # Metrics run on the sampled reporting step only, so the
                    # subset indexing (and its sync) is off the hot path.
                    present_v = present[valid]
                    idx_target = idx_target_all[valid]
                    # Aggregate KL — primary selection metric for target heads.
                    # KL(label || model) = present-weighted-NLL - entropy(label).
                    ent_per_frame = -(idx_target.clamp(min=1e-8) * idx_target.clamp(min=1e-8).log()).sum(dim=-1)
                    target_entropy = (present_v * ent_per_frame).sum() / present_v.sum().clamp(min=1e-6)
                    metrics["target_kl"] = (aux_ce - target_entropy).detach()
                    metrics["n_target_valid"] = torch.as_tensor(
                        float(present_v.numel()), dtype=target_flat.dtype, device=target_flat.device,
                    )

                    # target_skill sufficient stats (head-first; the proper
                    # scoring rule used for selection — see
                    # research/head-metrics.md). Present-weighted CE sum +
                    # present-weighted marginal label mass per entity slot;
                    # supervised_loop turns these into target_dll / target_skill
                    # (gain over predicting the marginal target distribution,
                    # normalised by its entropy). Distinct reference from
                    # target_kl (distance to the per-frame soft label).
                    present_sum = present_v.sum()
                    metrics["targetdist_n"] = present_sum.detach()
                    metrics["targetdist_ce_sum"] = (aux_ce.detach() * present_sum)
                    target_marg = (present_v.unsqueeze(-1) * idx_target).sum(dim=0)
                    for _k in range(target_marg.shape[0]):
                        metrics[f"targetdist_marg_{_k}"] = target_marg[_k].detach()

                    # Multi-candidate KL — KL restricted to frames where the
                    # head genuinely had a choice (#live enemies in obs > 1).
                    # Single-candidate frames are trivial wins for any head
                    # that points at the only available actor; this metric
                    # isolates real discrimination ability.  Requires
                    # entity_types + entity_ids[..., 2] (player_id) in obs.
                    if (
                        obs is not None
                        and isinstance(obs, dict)
                        and "entity_types" in obs
                        and "entity_ids" in obs
                    ):
                        types_t = self._tensor(obs["entity_types"], dtype=torch.long)
                        eids_t = self._tensor(obs["entity_ids"], dtype=torch.long)
                        if types_t.dim() >= 2 and eids_t.dim() >= 3:
                            types_flat = types_t.reshape(-1, types_t.shape[-1])
                            pids_flat = eids_t.reshape(-1, eids_t.shape[-2], eids_t.shape[-1])[..., 2]
                            live_actor = (types_flat == TOKEN_ACTOR) & (pids_flat > 0)
                            n_live = live_actor.sum(dim=-1)
                            multi_mask = (n_live > 1)
                            multi_valid = valid & multi_mask
                            if bool(multi_valid.any().item()):
                                log_probs_m = F.log_softmax(target_flat[multi_valid], dim=-1)
                                present_m = present[multi_valid]
                                idx_target_m = (
                                    idx_dist[multi_valid]
                                    / present_m.clamp(min=1e-6).unsqueeze(-1)
                                )
                                ce_m = -(idx_target_m * log_probs_m).sum(dim=-1)
                                ent_m = -(idx_target_m.clamp(min=1e-8) * idx_target_m.clamp(min=1e-8).log()).sum(dim=-1)
                                kl_m = (
                                    (present_m * (ce_m - ent_m)).sum()
                                    / present_m.sum().clamp(min=1e-6)
                                )
                                metrics["target_kl_multi"] = kl_m.detach()
                                metrics["n_target_valid_multi"] = torch.as_tensor(
                                    float(present_m.numel()),
                                    dtype=target_flat.dtype,
                                    device=target_flat.device,
                                )

        jump_loss_fn = getattr(getattr(self.model, "jump_head", None), "jump_loss", None)
        if jump_loss_fn is not None and JUMP_HEAD in logits:
            # a25 land-jump head owns its loss (label = press-byte bit7, scored
            # on ground-jump-feasible frames only; see jump_head.py).
            j_loss, _jm = jump_loss_fn(logits, actions, valid_flat, compute_metrics)
            losses.append(j_loss * float(weights_map.get(JUMP_HEAD, 1.0)))
            loss_is_real.append(True)
            if compute_metrics and _jm:
                metrics.update(_jm)

        move_seg_loss_fn = getattr(getattr(self.model, "move_seg_head", None), "move_seg_loss", None)
        if move_seg_loss_fn is not None and "move_seg" in logits and MOVE_HEAD in actions:
            # a25 segment head owns its loss (labels derived on the fly from the
            # sequence move classes; frame-shuffled batches contribute nothing).
            seg_loss, _sm = move_seg_loss_fn(
                logits, actions, valid_mask, compute_metrics,
                ud_weight=float(weights_map.get("move_seg_ud", 1.0)))
            losses.append(seg_loss * float(weights_map.get("move_seg", 1.0)))
            loss_is_real.append(True)
            if compute_metrics and _sm:
                metrics.update(_sm)

        weapon_loss_fn = getattr(getattr(self.model, "weapon_head", None), "weapon_loss", None)
        if weapon_loss_fn is not None and WEAPON_HEAD in logits and WEAPON_HEAD in actions:
            # A head may own its loss (mirrors look_head.look_loss) by
            # exposing a ``weapon_loss`` method. Production heads lack the
            # method, so they fall through to the canonical CE below.
            weapon_loss, _wm = weapon_loss_fn(logits, actions, valid_flat, compute_metrics, obs=obs)
            losses.append(weapon_loss * weights_map.get(WEAPON_HEAD, 1.0))
            loss_is_real.append(True)
            if compute_metrics and _wm:
                metrics.update(_wm)
        elif WEAPON_HEAD in logits and WEAPON_HEAD in actions:
            weapon_logits = logits[WEAPON_HEAD].reshape(-1, WEAPON_HEAD_SIZE)
            weapon_target = self._weapon_target_from_actions(actions)
            # No-weapon frames carry target=-100; F.cross_entropy with
            # ignore_index=-100 skips them on-GPU. Avoid the
            # ``valid.any().item()`` host sync that used to gate the call —
            # syncing per microbatch stalled the ROCm dispatch queue and
            # cost ~10ms per step on the head-probe loop.
            if valid_flat is not None:
                weapon_target = torch.where(
                    valid_flat, weapon_target, torch.full_like(weapon_target, -100)
                )
            valid_weapon = weapon_target >= 0
            weapon_loss = F.cross_entropy(
                weapon_logits, weapon_target, ignore_index=-100, reduction="mean",
            )
            losses.append(weapon_loss * weights_map.get(WEAPON_HEAD, 1.0))
            # Engaged training always has at least one valid weapon frame
            # per microbatch — skip the per-step host sync that previously
            # checked `valid.any().item()`. If you ever train on a corpus
            # where a microbatch could be all-no-weapon, restore the sync
            # or switch to a reduction='sum' / clamped-divisor scheme to
            # avoid the 0/0 → NaN in F.cross_entropy(reduction='mean').
            loss_is_real.append(True)
            if compute_metrics:
                metrics["loss_weapon"] = weapon_loss.detach()
                with torch.no_grad():
                    # Vectorized 8-class confusion matrix: 1 scatter_add
                    # instead of an 8-iteration Python loop with ~10 tensor
                    # ops per iteration. Cuts per-batch weapon-metric kernel
                    # count from ~80 to ~5 — measured ~5-8s/epoch saved at
                    # bs=4096 on this head-probe loop.
                    weapon_probs = F.softmax(weapon_logits, dim=-1)
                    weapon_pred = torch.argmax(weapon_probs, dim=-1)
                    # Map invalid frames to a sentinel out-of-range index
                    # so they don't land in any of the WEAPON_HEAD_SIZE rows.
                    safe_target = torch.where(
                        valid_weapon, weapon_target,
                        torch.full_like(weapon_target, WEAPON_HEAD_SIZE),
                    )
                    safe_pred = torch.where(
                        valid_weapon, weapon_pred,
                        torch.full_like(weapon_pred, WEAPON_HEAD_SIZE),
                    )
                    # Confusion matrix: rows=pred, cols=target, size (K+1)^2.
                    # Last row/col is the "invalid" bucket and is discarded.
                    K = WEAPON_HEAD_SIZE
                    flat_idx = (safe_pred * (K + 1) + safe_target).long()
                    conf = torch.zeros(
                        (K + 1) * (K + 1), dtype=torch.float32, device=weapon_logits.device,
                    )
                    conf.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                    conf = conf.view(K + 1, K + 1)[:K, :K]  # (K, K), drop invalid bucket
                    # Per-class tp/fp/fn: tp = diag; row sum - tp = fp; col sum - tp = fn.
                    tp_all = conf.diagonal()
                    fp_all = conf.sum(dim=1) - tp_all
                    fn_all = conf.sum(dim=0) - tp_all
                    valid_count = conf.sum()
                    metrics["n_weapon_valid"] = valid_count.detach().to(weapon_logits.dtype)
                    metrics["acc_weapon"] = (tp_all.sum() / valid_count.clamp(min=1.0)).detach()
                    metrics["confidence_weapon"] = weapon_probs.max(dim=-1).values.mean().detach()
                    # Per-class precision / recall / F1 + base rate so the
                    # rare classes (axe / GL / NG / SNG together <10% of
                    # frames) don't disappear into the headline number.
                    class_f1s = []
                    for cls_idx, cls_name in WEAPON_HEAD_CLASS_NAMES:
                        tp = tp_all[cls_idx]
                        fp = fp_all[cls_idx]
                        fn = fn_all[cls_idx]
                        metrics[f"tp_weapon_{cls_name}"] = tp.detach()
                        metrics[f"fp_weapon_{cls_name}"] = fp.detach()
                        metrics[f"fn_weapon_{cls_name}"] = fn.detach()
                        prec = tp / (tp + fp).clamp(min=1.0)
                        rec = tp / (tp + fn).clamp(min=1.0)
                        f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                        metrics[f"precision_weapon_{cls_name}"] = prec.detach()
                        metrics[f"recall_weapon_{cls_name}"] = rec.detach()
                        metrics[f"f1_weapon_{cls_name}"] = f1.detach()
                        metrics[f"pos_rate_weapon_{cls_name}"] = (
                            (tp + fn) / valid_count.clamp(min=1.0)
                        ).detach()
                        class_f1s.append(f1)
                    metrics["f1_weapon"] = torch.stack(class_f1s).mean().detach()

                    # weapon_skill sufficient stats (head-first; the proper
                    # scoring rule used for selection — see
                    # research/head-metrics.md). Clean CE sum + true-class
                    # histogram → marginal entropy; supervised_loop derives
                    # weapon_dll / weapon_skill (fraction of the 8-class
                    # marginal entropy the head eliminates). loss_weapon is the
                    # clean per-frame mean CE, so ce_sum = loss_weapon × n.
                    metrics["weapondist_n"] = valid_count.detach().to(weapon_logits.dtype)
                    metrics["weapondist_ce_sum"] = (
                        weapon_loss.detach() * valid_count
                    ).to(weapon_logits.dtype)
                    for cls_idx, _cls_name in WEAPON_HEAD_CLASS_NAMES:
                        metrics[f"weapondist_h_{cls_idx}"] = (
                            tp_all[cls_idx] + fn_all[cls_idx]
                        ).detach().to(weapon_logits.dtype)

        if MOVE_HEAD in logits and MOVE_HEAD in actions:
            # Move = 3 categorical axes (fb, lr, ud) × 3 classes {neg, none,
            # pos}. Labels are uint8[T, 3] axis class indices from the
            # corpus loader.
            move_logits = logits[MOVE_HEAD]
            move_pred = move_logits.reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
            move_target_t = self._tensor(actions[MOVE_HEAD], dtype=torch.long)
            # Distance-weighted shoulder on the ud (jump) axis. Only sensible
            # when targets arrive in (T, B, 3) form so the conv sees a real
            # time axis; single-step inference falls back to plain CE.
            jump_dist_weight_flat: torch.Tensor | None = None
            if self.jump_distance_sigma > 0.0:
                ud_idx = MOVE_AXIS_NAMES.index("ud")
                if move_target_t.ndim == 3:
                    from qnn.bc.loss_shaping import distance_weighted_neg_weights
                    jump_pos_2d = (move_target_t[..., ud_idx] == MOVE_CLASS_POS).to(torch.float32)
                    valid_2d = valid_mask.bool() if valid_mask is not None else None
                    w_2d = distance_weighted_neg_weights(
                        jump_pos_2d, valid_2d, self.jump_distance_sigma,
                    )
                    jump_dist_weight_flat = w_2d.reshape(-1)
                elif move_target_t.ndim == 2 and "jump_distance_to_pos" in actions:
                    # Flat batch (frame-shuffled SGD). The jump-positive mask
                    # is derived from move[..., ud_idx] == MOVE_CLASS_POS;
                    # the per-frame distance was precomputed at preload time.
                    from qnn.bc.loss_shaping import flat_distance_weight
                    jump_pos_1d = (move_target_t[..., ud_idx] == MOVE_CLASS_POS).to(torch.float32)
                    jump_d = self._tensor(actions["jump_distance_to_pos"], dtype=torch.float32).reshape(-1)
                    jump_dist_weight_flat = flat_distance_weight(
                        jump_d, jump_pos_1d, self.jump_distance_sigma,
                    )
            move_target = move_target_t.reshape(-1, MOVE_AXES)
            base_move_valid = valid_flat if valid_flat is not None else torch.ones(
                (move_target.shape[0],), dtype=torch.bool, device=move_target.device,
            )
            # Per-axis label rewrite. When input_mask is off, every axis
            # uses the raw demo button (usercmd) as the label. When on,
            # axis i's label is the engine OUTCOME = (demo intent) AND
            # (per-direction feasibility from input_mask bits). Under
            # pure-feasibility semantics from the C side:
            #   fb (axis 0): feasibility bits 1-2 are always 1 when alive
            #                (pmove always processes fmove). Label =
            #                demo intent unchanged.
            #   lr (axis 1): same — feasibility bits 3-4 always 1.
            #   ud (axis 2): direction-specific. POS feasibility is bit 7
            #                (jump on ground) OR bit 6 (swim up in water).
            #                NEG feasibility is bit 5 (swim down in water).
            #                Demo intent in MOVE_CLASS_POS that's not
            #                feasible (e.g. air-jump press with no
            #                ground) is rewritten to NONE — the engine
            #                couldn't have honoured that press.
            # No frames dropped; the model trains on every frame against
            # the engine-outcome label.
            move_valid_per_axis: list[torch.Tensor] = [base_move_valid] * MOVE_AXES
            if input_mask_on and input_mask_flat is not None:
                none_t = torch.full_like(move_target[:, 0], MOVE_CLASS_NONE)
                rewritten = move_target.clone()
                # fb / lr: feasibility is always 1 (alive frames) — no
                # rewrite needed; demo intent IS the engine outcome.
                # ud: gate the demo intent through per-direction
                # feasibility.
                up_neg_feas = ((input_mask_flat >> 5) & 1) != 0  # swim down
                up_pos_feas = ((input_mask_flat >> 6) & 1) != 0  # swim up
                jump_feas   = ((input_mask_flat >> 7) & 1) != 0  # ground jump
                ud_pos_feas = jump_feas | up_pos_feas
                ud_intent = move_target[:, 2]
                # POS intent: keep only if pos feasible, else NONE.
                # NEG intent: keep only if neg feasible (swim down), else
                # NONE. NONE intent stays NONE.
                pos_mask = (ud_intent == MOVE_CLASS_POS) & ud_pos_feas
                neg_mask = (ud_intent == MOVE_CLASS_NEG) & up_neg_feas
                rewritten[:, 2] = torch.where(
                    pos_mask,
                    torch.full_like(ud_intent, MOVE_CLASS_POS),
                    torch.where(
                        neg_mask,
                        torch.full_like(ud_intent, MOVE_CLASS_NEG),
                        none_t,
                    ),
                )
                move_target = rewritten
            move_is_real = base_move_valid.any()
            # ud (jump) axis is heavily imbalanced (~4% pos rate); upweight
            # the POS class via jump_pos_weight when set above 1.0.  fb/lr
            # are balanced enough that plain CE works.
            ud_class_weight = None
            if self.jump_pos_weight != 1.0:
                ud_class_weight = torch.tensor(
                    [1.0, 1.0, float(self.jump_pos_weight)],
                    dtype=move_pred.dtype, device=move_pred.device,
                )
            ce_per_axis = []
            for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                axis_valid = move_valid_per_axis[axis_i]
                axis_pred = move_pred[axis_valid, axis_i, :]
                axis_target = move_target[axis_valid, axis_i]
                axis_is_real = axis_pred.shape[0] > 0
                if not axis_is_real:
                    ce_axis = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                elif axis_name == "ud":
                    if jump_dist_weight_flat is not None:
                        # Per-frame CE then multiplicative distance weight,
                        # matching the attack-head .mean() reduction so both
                        # heads' loss magnitudes scale the same way.
                        ce_pf = F.cross_entropy(
                            axis_pred, axis_target,
                            weight=ud_class_weight, reduction="none",
                        )
                        ce_axis = (ce_pf * jump_dist_weight_flat[axis_valid]).mean()
                    else:
                        ce_axis = F.cross_entropy(
                            axis_pred, axis_target, weight=ud_class_weight, reduction="mean",
                        )
                else:
                    ce_axis = F.cross_entropy(axis_pred, axis_target, reduction="mean")
                ce_per_axis.append(ce_axis)
            # Optional per-axis weights via head_loss_weights keys
            # "move_fb"/"move_lr"/"move_ud" (default 1.0 = the historical
            # equal-weight mean). The /n_axes normalization is kept so a
            # zeroed axis drops its gradient without rescaling the
            # surviving axes' contribution.
            axis_weights = [
                float(weights_map.get(f"{MOVE_HEAD}_{axis_name}", 1.0))
                for axis_name in MOVE_AXIS_NAMES
            ]
            if all(w == 1.0 for w in axis_weights):
                move_loss = torch.stack(ce_per_axis).mean()  # equal-weight axes
            else:
                move_loss = (
                    torch.stack(ce_per_axis) * move_pred.new_tensor(axis_weights)
                ).sum() / len(ce_per_axis)
            losses.append(move_loss * weights_map.get(MOVE_HEAD, 1.0))
            loss_is_real.append(move_is_real)
            if compute_metrics:
                move_has_rows = bool(move_is_real.item())
                metrics["loss_move"] = move_loss.detach()
                if move_has_rows:
                    with torch.no_grad():
                        # Per-axis argmax computed once; per-axis indexing
                        # below selects each axis's valid frames separately
                        # so op_input-masked axes drop their stale frames.
                        move_argmax_all = torch.argmax(move_pred, dim=-1)  # (B, 3)
                        per_axis_acc: list[torch.Tensor] = []
                        per_axis_macro_f1 = []
                        for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                            axis_valid = move_valid_per_axis[axis_i]
                            metrics[f"loss_move_{axis_name}"] = ce_per_axis[axis_i].detach()
                            pred_axis = move_argmax_all[axis_valid, axis_i]
                            true_axis = move_target[axis_valid, axis_i]
                            if pred_axis.numel() > 0:
                                axis_acc = (pred_axis == true_axis).float().mean()
                            else:
                                axis_acc = torch.zeros((), dtype=move_pred.dtype, device=move_pred.device)
                            metrics[f"acc_move_{axis_name}"] = axis_acc.detach()
                            per_axis_acc.append(axis_acc)
                            # Per-class precision / recall / F1 across all three
                            # classes (neg/none/pos).  Macro-F1 per axis is the
                            # honest single-axis summary that doesn't hide the
                            # rare-class failure modes (jump under ud, backpedal
                            # under fb) behind the dominant "none" class.
                            class_f1s = []
                            for cls_idx, cls_name in ((MOVE_CLASS_NEG, "neg"),
                                                      (MOVE_CLASS_NONE, "none"),
                                                      (MOVE_CLASS_POS, "pos")):
                                pred_cls = pred_axis == cls_idx
                                true_cls = true_axis == cls_idx
                                tp = (pred_cls & true_cls).sum().float()
                                fp = (pred_cls & ~true_cls).sum().float()
                                fn = (~pred_cls & true_cls).sum().float()
                                prec = tp / (tp + fp).clamp(min=1.0)
                                rec = tp / (tp + fn).clamp(min=1.0)
                                f1 = 2.0 * prec * rec / (prec + rec).clamp(min=1e-6)
                                metrics[f"precision_move_{axis_name}_{cls_name}"] = prec.detach()
                                metrics[f"recall_move_{axis_name}_{cls_name}"] = rec.detach()
                                metrics[f"f1_move_{axis_name}_{cls_name}"] = f1.detach()
                                if true_cls.numel() > 0:
                                    metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = true_cls.float().mean().detach()
                                else:
                                    metrics[f"pos_rate_move_{axis_name}_{cls_name}"] = torch.zeros(
                                        (), dtype=move_pred.dtype, device=move_pred.device,
                                    )
                                class_f1s.append(f1)
                            macro = torch.stack(class_f1s).mean()
                            metrics[f"f1_move_{axis_name}"] = macro.detach()
                            per_axis_macro_f1.append(macro)
                        # Equal-axes overall acc/F1 — the mean-of-per-axis
                        # form is identical to the original ``argmax ==
                        # target`` mean when all axes share one valid mask
                        # (i.e. input_mask off), and remains
                        # well-defined when per-axis valids differ.
                        metrics["acc_move"] = torch.stack(per_axis_acc).mean().detach()
                        metrics["f1_move"] = torch.stack(per_axis_macro_f1).mean().detach()
                        # Distributional human-likeness sufficient statistics
                        # (additive; combined at epoch end into move_dll /
                        # move_kl_joint / move_kl_marg / jump calibration — see
                        # qnn.bc.supervised_loop._move_distribution_metrics_from_sums).
                        # f1_move / acc_move reward argmax point-accuracy and are
                        # blind to whether the model reproduces (a) the human
                        # per-axis class marginals, (b) the joint fb/lr/ud combo
                        # histogram, and (c) the ~4% jump rate. These score the
                        # predicted DISTRIBUTION instead. Emitted as scalars (the
                        # metric flush stacks raw-sum metrics into one tensor, so
                        # per-class / per-combo vectors are unrolled); prefix
                        # movedist_ is registered raw-sum in supervised_loop.
                        # The joint uses the all-axes-shared valid mask
                        # (base_move_valid) so the 27-combo histogram is well
                        # defined; move_target is the engine-outcome label (ud
                        # rewritten under input_mask), matching the loss.
                        jv = base_move_valid
                        njv = int(jv.sum().item())
                        if njv > 0:
                            pf = torch.softmax(move_pred[jv], dim=-1)      # (V,3,3)
                            logpf = torch.log_softmax(move_pred[jv], dim=-1)
                            tj = move_target[jv]                            # (V,3)
                            ar = torch.arange(njv, device=move_pred.device)
                            metrics["movedist_n"] = move_pred.new_tensor(float(njv))
                            for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                                # per-axis model NLL sum (for move_dll) + human
                                # class counts + summed model probs (model marginal).
                                metrics[f"movedist_ce_{axis_name}"] = (
                                    -logpf[ar, axis_i, tj[:, axis_i]].sum()
                                ).detach()
                                hist_a = torch.bincount(
                                    tj[:, axis_i], minlength=MOVE_AXIS_CLASSES,
                                ).to(move_pred.dtype)
                                pred_a = pf[:, axis_i, :].sum(0)            # (3,)
                                for c in range(MOVE_AXIS_CLASSES):
                                    metrics[f"movedist_h_{axis_name}_{c}"] = hist_a[c].detach()
                                    metrics[f"movedist_p_{axis_name}_{c}"] = pred_a[c].detach()
                            # Joint combo histogram, combo = fb + 3*lr + 9*ud
                            # (27 bins). Human = counts; model = expected combo
                            # mass = sum_frames outer(P_fb, P_lr, P_ud) — the
                            # model is per-frame axis-independent, so this is its
                            # implied joint aggregated over the feature stream.
                            combo_idx = tj[:, 0] + 3 * tj[:, 1] + 9 * tj[:, 2]
                            jh = torch.bincount(combo_idx, minlength=27).to(move_pred.dtype)
                            # einsum gives E[i,j,k]=[fb,lr,ud]; a plain reshape
                            # would be C-order (fb*9+lr*3+ud), but jh is binned
                            # fb+3*lr+9*ud. permute(2,1,0)→[ud,lr,fb] so the
                            # C-order reshape lands on the SAME combo index as
                            # jh (else move_kl_joint compares scrambled bins).
                            jp = torch.einsum(
                                "vi,vj,vk->ijk", pf[:, 0, :], pf[:, 1, :], pf[:, 2, :],
                            ).permute(2, 1, 0).reshape(27)
                            for m in range(27):
                                metrics[f"movedist_jh_{m}"] = jh[m].detach()
                                metrics[f"movedist_jp_{m}"] = jp[m].detach()
                            # ud argmax-pos count: the "jump collapse" number. The
                            # expected pos prob (movedist_p_ud_2 / n) is the
                            # calibration counterpart — a calibrated jump_pos_weight
                            # =1.0 head has expected≈human while argmax≈0.
                            ud_i = MOVE_AXIS_NAMES.index("ud")
                            am_pos = (
                                move_pred[jv][:, ud_i, :].argmax(-1) == MOVE_CLASS_POS
                            ).sum().to(move_pred.dtype)
                            metrics["movedist_ampos_ud"] = am_pos.detach()

        # a25 move-hazard (WHEN-law) — calibrated BCE on the per-axis release
        # event. Labels are the precomputed, episode-aware columns derived from
        # act_move (qnn.model.bench.a25.hazard_labels); present only for
        # full_6head runs. valid = the column's episode-boundary mask AND the
        # in-distribution segment mask. The head owns its loss (mirrors
        # look_head.look_loss / the weapon when-loss).
        # The head owns its loss (mirrors look_head.look_loss / weapon_loss) —
        # reach it via the built head, not a bench.a25 import. getattr nesting is
        # robust to self.model being None (no head ⇒ skip, like look/weapon).
        hazard_loss_fn = getattr(
            getattr(self.model, "move_hazard_head", None), "hazard_loss", None)
        if (hazard_loss_fn is not None
                and MOVE_HAZARD_HEAD in logits and "move_hazard_release" in actions):
            hz_logits = logits[MOVE_HAZARD_HEAD].reshape(-1, MOVE_AXES)
            hz_release = self._tensor(
                actions["move_hazard_release"], dtype=torch.float32).reshape(-1, MOVE_AXES)
            hz_valid = self._tensor(
                actions["move_hazard_valid"], dtype=torch.bool).reshape(-1, MOVE_AXES)
            if valid_flat is not None:
                hz_valid = hz_valid & valid_flat.unsqueeze(-1)
            hz_loss, hz_metrics = hazard_loss_fn(
                hz_logits, hz_release, hz_valid, compute_metrics)
            losses.append(hz_loss * weights_map.get(MOVE_HAZARD_HEAD, 1.0))
            # Like the weapon head: avoid a per-step host sync on valid.any();
            # an all-masked batch yields loss 0 (denom clamp) and contributes 0.
            loss_is_real.append(True)
            if compute_metrics:
                metrics.update(hz_metrics)

        if ATTACK_HEAD in logits and ATTACK_HEAD in actions:
            attack_logits = logits[ATTACK_HEAD]
            attack_target_t = self._tensor(actions[ATTACK_HEAD], dtype=torch.float32)
            # Two distance-shoulder paths exist:
            #
            # 1. Sequence path (ndim==2, lane-packed pipeline): compute
            #    weights via Conv1d on the (T, B) target stream so each
            #    frame sees its time-axis neighbors.
            # 2. Flat path (ndim==1, GPU-resident frame-shuffled SGD):
            #    no time axis exists in the batch, so we use a
            #    per-frame "distance to nearest positive in same episode"
            #    that was precomputed at preload time and shipped via
            #    actions["attack_distance_to_pos"].
            #
            # Both produce the same loss semantics; only the
            # convolution/precompute boundary moves.
            distance_weight_flat: torch.Tensor | None = None
            if self.attack_distance_sigma > 0.0:
                if attack_target_t.ndim == 2:
                    from qnn.bc.loss_shaping import distance_weighted_neg_weights
                    valid_2d = valid_mask.bool() if valid_mask is not None else None
                    w_2d = distance_weighted_neg_weights(
                        attack_target_t, valid_2d, self.attack_distance_sigma,
                    )
                    distance_weight_flat = w_2d.reshape(-1)
                elif attack_target_t.ndim == 1 and "attack_distance_to_pos" in actions:
                    from qnn.bc.loss_shaping import flat_distance_weight
                    attack_d = self._tensor(actions["attack_distance_to_pos"], dtype=torch.float32)
                    distance_weight_flat = flat_distance_weight(
                        attack_d.reshape(-1), attack_target_t.reshape(-1),
                        self.attack_distance_sigma,
                    )

            attack_pred_full = attack_logits.reshape(-1)
            attack_target_full = attack_target_t.reshape(-1)
            attack_dw_full = distance_weight_flat
            # Optional +1 op-frame shifted loss target. Built per-episode at
            # preload by qnn.bc.supervised_loop._compute_attack_shifted and
            # surfaced as actions["attack_shifted"]. Only the BCE *target*
            # shifts; metrics below stay on the original attack label.
            attack_loss_target_full: torch.Tensor | None = None
            if (
                self.attack_label_shift
                and "attack_shifted" in actions
            ):
                attack_loss_target_full = self._tensor(
                    actions["attack_shifted"], dtype=torch.float32,
                ).reshape(-1)
            if valid_flat is not None:
                attack_pred_full = attack_pred_full[valid_flat]
                attack_target_full = attack_target_full[valid_flat]
                if attack_dw_full is not None:
                    attack_dw_full = attack_dw_full[valid_flat]
                if attack_loss_target_full is not None:
                    attack_loss_target_full = attack_loss_target_full[valid_flat]
            # Label rewrite under input_mask. Off: label is the raw demo
            # button (usercmd, move byte bit 6). On: label becomes the
            # engine OUTCOME = pure feasibility (input_mask bit 0) AND
            # the demo's actual press (current attack_target_full, the
            # usercmd attack bit). Feasibility is "would W_Attack fire
            # if button0=1 right now"; AND with demo press recovers
            # "did W_Attack actually fire this tick".
            if input_mask_on and input_mask_flat is not None:
                input_mask_full = input_mask_flat
                if valid_flat is not None:
                    input_mask_full = input_mask_full[valid_flat]
                feasibility = (input_mask_full & 1).to(attack_target_full.dtype)
                demo_press  = attack_target_full
                attack_target_full = feasibility * demo_press
                if attack_loss_target_full is not None:
                    # Apply the same feasibility AND to the shifted target
                    # so the loss label remains an engine-outcome bit when
                    # input_mask is on.
                    attack_loss_target_full = feasibility * attack_loss_target_full
            attack_pred = attack_pred_full
            attack_target = attack_target_full
            # attack_loss_target = label fed to BCE; defaults to the metric
            # target (current attack label) so behavior is bit-identical
            # when attack_label_shift is off.
            attack_loss_target = (
                attack_loss_target_full
                if attack_loss_target_full is not None
                else attack_target_full
            )
            attack_dw = attack_dw_full
            attack_is_real = attack_target.numel() > 0
            # pos_weight conventionally lives in class_weights[ATTACK_HEAD] (set
            # at training startup from corpus statistics: neg_count/pos_count).
            pos_weight: torch.Tensor | None = None
            if class_weights is not None and ATTACK_HEAD in class_weights:
                cw = class_weights[ATTACK_HEAD]
                pos_weight = cw if isinstance(cw, torch.Tensor) else torch.as_tensor(cw, device=attack_pred.device)
            if attack_is_real:
                # Unified path: per-frame BCE, weighted by (op? * focal? *
                # distance?), then weighted-mean reduction. When
                # attack_op_only is on, the op mask (from input_mask bit 0)
                # zeroes gradient on no-op frames — those rows can't
                # actuate at inference (engine ignores fire during
                # cooldown). When attack_op_only is off (default), op=0
                # frames stay in: their label is feasibility AND demo_press
                # (forced to 0), the BCE pulls predictions toward 0 there,
                # and pos_weight is computed across the full corpus.
                bce = F.binary_cross_entropy_with_logits(
                    attack_pred, attack_loss_target,
                    pos_weight=pos_weight, reduction="none",
                )
                weight = torch.ones_like(bce)
                op_full: torch.Tensor | None = None
                if input_mask_on and input_mask_flat is not None:
                    op_full = (input_mask_flat & 1).to(bce.dtype)
                    if valid_flat is not None:
                        op_full = op_full[valid_flat]
                if self.attack_op_only and op_full is not None:
                    weight = weight * op_full
                if self.attack_focal_gamma > 0.0:
                    p = torch.sigmoid(attack_pred)
                    pt = torch.where(attack_loss_target > 0.5, p, 1.0 - p)
                    alpha_t = torch.where(
                        attack_loss_target > 0.5,
                        torch.full_like(p, self.attack_focal_alpha),
                        torch.full_like(p, 1.0 - self.attack_focal_alpha),
                    )
                    weight = weight * alpha_t * (1.0 - pt).clamp(min=1e-6) ** self.attack_focal_gamma
                if attack_dw is not None:
                    weight = weight * attack_dw
                attack_loss = (weight * bce).sum() / weight.sum().clamp(min=1.0)
            else:
                attack_loss = torch.zeros((), dtype=attack_logits.dtype, device=attack_logits.device)
            losses.append(attack_loss * weights_map.get(ATTACK_HEAD, 1.0))
            loss_is_real.append(attack_is_real)
            if compute_metrics:
                metrics["loss_attack"] = attack_loss.detach()
                # Single fire f1, computed against whatever label the
                # input_mask flag selected. Off → usercmd label; on →
                # engine-outcome label. No separate ``*_masked`` metric
                # — there's only one label per run now, so the metric
                # is unambiguous.
                if attack_target.numel() > 0:
                    with torch.no_grad():
                        pred_pos = (torch.sigmoid(attack_pred) > 0.5)
                        target_pos = attack_target > 0.5
                        # When training is op-only, the metric is also
                        # op-only: op=0 frames had no gradient and their
                        # model predictions are uncalibrated noise. Score
                        # only frames where the head's decision matters.
                        if (self.attack_op_only and input_mask_on
                                and input_mask_flat is not None):
                            op_full = (input_mask_flat & 1).to(torch.bool)
                            if valid_flat is not None:
                                op_full = op_full[valid_flat]
                            pred_pos = pred_pos & op_full
                            target_pos = target_pos & op_full
                            tp = (pred_pos & target_pos).sum()
                            fp = (pred_pos & ~target_pos & op_full).sum()
                            fn = (~pred_pos & target_pos & op_full).sum()
                            tn = (~pred_pos & ~target_pos & op_full).sum()
                        else:
                            tp = (pred_pos & target_pos).sum()
                            fp = (pred_pos & ~target_pos).sum()
                            fn = (~pred_pos & target_pos).sum()
                            tn = (~pred_pos & ~target_pos).sum()
                        metrics["tp_attack"] = tp.detach()
                        metrics["fp_attack"] = fp.detach()
                        metrics["fn_attack"] = fn.detach()
                        metrics["tn_attack"] = tn.detach()
                        n_total = tp + fp + fn + tn
                        metrics["acc_attack"] = ((tp + tn).float() / n_total.clamp(min=1)).detach()
                        prec_denom = (tp + fp).clamp(min=1)
                        rec_denom = (tp + fn).clamp(min=1)
                        prec = tp.float() / prec_denom
                        rec = tp.float() / rec_denom
                        f1_denom = (prec + rec).clamp(min=1e-6)
                        metrics["precision_attack"] = prec.detach()
                        metrics["recall_attack"] = rec.detach()
                        metrics["f1_attack"] = (2.0 * prec * rec / f1_denom).detach()

                        # attack_skill sufficient stats (head-first; the proper
                        # scoring rule used for selection — see
                        # research/head-metrics.md). A CLEAN, unweighted BCE
                        # (no pos_weight / focal / distance weighting — those
                        # shape the gradient, not the likelihood) on the scored
                        # frames, plus the positive count for the base-rate
                        # binary entropy. supervised_loop derives attack_dll /
                        # attack_skill. Scored population mirrors the f1 above:
                        # op-only when attack_op_only + input_mask, else all
                        # valid frames.
                        clean_bce = F.binary_cross_entropy_with_logits(
                            attack_pred, attack_loss_target, reduction="none",
                        )
                        scored = torch.ones_like(attack_loss_target, dtype=torch.bool)
                        if (self.attack_op_only and input_mask_on
                                and input_mask_flat is not None):
                            _op_scored = (input_mask_flat & 1).to(torch.bool)
                            if valid_flat is not None:
                                _op_scored = _op_scored[valid_flat]
                            scored = _op_scored
                        metrics["attackdist_ce_sum"] = (clean_bce * scored).sum().detach()
                        metrics["attackdist_n"] = scored.sum().to(clean_bce.dtype).detach()
                        metrics["attackdist_pos"] = (
                            (attack_loss_target > 0.5) & scored
                        ).sum().to(clean_bce.dtype).detach()
                # Diagnostics for the prior-residual decomposition.
                # mean/std of the prior and delta logits across the same
                # frames the loss sees — answers "is the residual
                # actually doing anything?" Skipped when the prior is
                # off ("_attack_prior" absent / zeros).
                if "_attack_prior" in logits and "_attack_delta" in logits and attack_is_real:
                    with torch.no_grad():
                        prior_full = logits["_attack_prior"].reshape(-1)
                        delta_full = logits["_attack_delta"].reshape(-1)
                        if valid_flat is not None:
                            prior_full = prior_full[valid_flat]
                            delta_full = delta_full[valid_flat]
                        metrics["attack_prior_mean"] = prior_full.mean().detach()
                        metrics["attack_prior_std"] = prior_full.std().detach()
                        metrics["attack_delta_mean"] = delta_full.mean().detach()
                        metrics["attack_delta_std"] = delta_full.std().detach()

        if LOOK_HEAD in logits and LOOK_HEAD in actions:
            # Magnitude-sensitive supervision: regress the raw look_delta
            # output against the geometric residual (demo_unit - look_prior).
            # The residual has bounded magnitude (≤ 2 for unit vectors);
            # forces the head to "pay" the right magnitude for whatever
            # direction it expresses, instead of growing delta arbitrarily
            # large to override the prior.  Quality is tracked via look_r2 /
            # look_ewa_deg (see _emit_look_tangent_sums); the loss is unchanged.
            look_pred = logits[LOOK_HEAD].reshape(-1, LOOK_HEAD_SIZE)
            look_prior = logits["_look_prior"].reshape(-1, LOOK_HEAD_SIZE)
            look_delta = logits["_look_delta"].reshape(-1, LOOK_HEAD_SIZE)

            look_label_raw_t = self._tensor(actions[LOOK_HEAD], dtype=torch.float32)
            look_label_raw = look_label_raw_t.reshape(-1, look_label_raw_t.shape[-1])
            target_norm = torch.linalg.vector_norm(look_label_raw, dim=-1, keepdim=True)
            valid = target_norm.squeeze(-1) > 1e-6
            if valid_flat is not None:
                valid = valid & valid_flat
            aux_is_real = valid.any()
            aux_has_rows = bool(aux_is_real.item()) if compute_metrics else False
            # Bench look heads may carry their own loss (binned/polar/vMF), so a new
            # loss form needs NO canonical change: the head reads its outputs from
            # `logits` (forwarded generically by Network) and returns (loss, metrics).
            look_loss_fn = getattr(getattr(self.model, "look_head", None), "look_loss", None)
            if look_loss_fn is not None:
                # Hook contract (dense): the label covers EVERY row — invalid
                # rows are filled with the no-turn unit vector (+x), which maps
                # to the protected hold bin — and the head folds `valid` into
                # its loss weights. Subset indexing (`x[valid]`) calls
                # nonzero(), whose device→host sync was the profiled training
                # bottleneck; the fill keeps polar_targets NaN-free on the
                # rows the mask zeroes out anyway.
                look_unit = look_label_raw / target_norm.clamp(min=1e-6)
                no_turn = torch.zeros_like(look_unit)
                no_turn[..., 0] = 1.0
                look_label = torch.where(valid.unsqueeze(-1), look_unit, no_turn)
                look_loss, _look_metrics = look_loss_fn(
                    logits, look_label, valid, compute_metrics and aux_has_rows,
                )
                if compute_metrics and aux_has_rows:
                    metrics.update(_look_metrics)
            elif "_look_bins" in logits:
                look_label = look_label_raw[valid] / target_norm[valid].clamp(min=1e-6)
                # Binned (classification) look head: per-axis cross-entropy
                # over foveated tangent bins instead of smooth_l1 on the
                # delta. look_predict (decoded direction) still drives the
                # look_r2 metric. See qnn.model.look_bins.
                from qnn.model.look_bins import (
                    N_BINS as _LOOK_N_BINS,
                    bin_targets as _look_bin_targets,
                    soft_bin_targets as _look_soft_targets,
                    tangent_logmap as _look_logmap,
                )
                bins = logits["_look_bins"].reshape(-1, 2, _LOOK_N_BINS)[valid]
                z_look = _look_logmap(look_label)                      # (V, 2) tangent
                tgt = _look_bin_targets(z_look)                        # (V, 2) long
                n_valid = max(int(bins.shape[0]), 1)
                if self.look_label_smoothing_sigma > 0.0:
                    # Distance-aware Gaussian soft-target CE: per axis,
                    # -(soft * log_softmax(logits)).sum over bins, mean over
                    # rows, summed over the 2 axes — same reduction as the
                    # hard two-CE path below.
                    soft = _look_soft_targets(z_look, self.look_label_smoothing_sigma)
                    logp = F.log_softmax(bins, dim=-1)                 # (V, 2, N_BINS)
                    look_loss = -(soft * logp).sum() / n_valid
                else:
                    look_loss = (
                        F.cross_entropy(
                            bins[:, 0, :], tgt[:, 0], reduction="sum",
                        )
                        + F.cross_entropy(
                            bins[:, 1, :], tgt[:, 1], reduction="sum",
                        )
                    ) / n_valid
                if compute_metrics and aux_has_rows:
                    # Human-likeness sufficient statistics (additive; combined at
                    # epoch end into look_dll / look_emd_deg — see
                    # qnn.bc.look_metrics.humanlike_from_sums). The binned head's
                    # PRODUCT is its per-axis bin DISTRIBUTION, so we score the
                    # distribution, not the decoded mean. look_r2 / look_ewa_deg
                    # (mean-fidelity proxies) are intentionally NOT emitted for the
                    # binned head — for human-likeness they reward mean-regression.
                    # Emit as SCALARS only — the metric flush stacks all raw-sum
                    # metrics into one tensor (qnn.bc.supervised_loop), so per-bin
                    # vectors must be unrolled. Combined into look_dll / look_emd_deg
                    # at epoch end (and these keys popped). Prefix: lookdist_.
                    with torch.no_grad():
                        logp = torch.log_softmax(bins, dim=-1)          # (V,2,N_BINS)
                        p = logp.exp()
                        V = bins.shape[0]
                        ar = torch.arange(V, device=bins.device)
                        metrics["lookdist_n"] = bins.new_tensor(float(V))
                        for a in (0, 1):
                            metrics[f"lookdist_ce_{a}"] = (-logp[ar, a, tgt[:, a]].sum()).detach()
                            hist_a = torch.bincount(tgt[:, a], minlength=_LOOK_N_BINS).to(bins.dtype)
                            pred_a = p[:, a, :].sum(0)                  # (N_BINS,)
                            for b in range(_LOOK_N_BINS):
                                metrics[f"lookdist_h_{a}_{b}"] = hist_a[b].detach()
                                metrics[f"lookdist_p_{a}_{b}"] = pred_a[b].detach()
            else:
                # Target residual: what delta should be to make
                # normalize(base + delta) = look_label. Sum + explicit divisor
                # keeps the all-masked case finite without a host-side branch.
                look_label = look_label_raw[valid] / target_norm[valid].clamp(min=1e-6)
                look_residual = look_label - look_prior[valid]
                look_delta_valid = look_delta[valid]
                look_loss = F.smooth_l1_loss(
                    look_delta_valid, look_residual, beta=0.05, reduction="sum",
                ) / max(int(look_delta_valid.numel()), 1)
            losses.append(look_loss * weights_map.get(LOOK_HEAD, 1.0))
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_look"] = look_loss.detach()
                # Regression-only diagnostics (delta magnitude + tangent-space
                # look_r2 sums). Skipped when the head carries its own loss
                # (hook path) or is binned — those emit their own metrics.
                if aux_has_rows and look_loss_fn is None and "_look_bins" not in logits:
                    with torch.no_grad():
                        # Track delta magnitude so we can confirm the head
                        # is no longer growing it unbounded.
                        metrics["mag_delta_look"] = (
                            torch.linalg.vector_norm(look_delta[valid], dim=-1).mean().detach()
                        )
                        # Smooth, non-saturated look metrics (look_r2,
                        # look_ewa_deg) via tangent-space sufficient statistics,
                        # combined at epoch end in supervised_loop. Plain
                        # cos_sim is NOT tracked: it saturates near 1.0 (no-turn
                        # = (1,0,0)) and drifts with label recollections, so it
                        # can't separate or compare models. See qnn.bc.look_metrics.
                        # Mean-fidelity sums (look_r2 / look_ewa_deg) only for the
                        # regression head; the binned head emits lookdist_* instead.
                        if "_look_bins" not in logits:
                            self._emit_look_tangent_sums(
                                metrics, look_pred[valid], look_label,
                            )

        if compute_metrics:
            metrics["accuracy"] = (
                torch.stack(accuracy_components).mean()
                if accuracy_components
                else torch.zeros((), device=self.device)
            )
        else:
            metrics["accuracy"] = torch.zeros((), device=self.device)

        return losses, loss_is_real, metrics

    def _weapon_target_from_actions(
        self,
        actions: Mapping[str, np.ndarray | torch.Tensor],
    ) -> torch.Tensor:
        """Return dense desired-weapon targets from collected BC labels.

        The collector stores `weapon` as the raw engine weapon byte:
          0 = no weapon held (pre-spawn / dead / transitional),
          1..8 = Quake weapon id in impulse order (axe..thunderbolt).
        The 8-class weapon head trains on weapons only; no-weapon frames
        map to -100 so F.cross_entropy(..., ignore_index=-100) skips
        them while their move/fire/look labels still train.
        """
        weapon = self._tensor(actions[WEAPON_HEAD], dtype=torch.long).reshape(-1)
        bad = (weapon < 0) | (weapon > WEAPON_HEAD_SIZE)
        if weapon.device.type == "cpu":
            if bool(bad.any()):
                sample = weapon[bad][:8].tolist()
                raise ValueError(
                    f"weapon bytes must be in 0..{WEAPON_HEAD_SIZE}, got {sample}"
                )
        else:
            # Corpus validation reports detailed samples once at preload. Keep
            # the per-batch device-side guard without synchronizing the GPU just
            # to inspect a condition that is expected to be false.
            torch._assert_async(
                (~bad).all(),
                f"weapon bytes must be in 0..{WEAPON_HEAD_SIZE}",
            )
        # 1..8 → class 0..7; 0 (no weapon) → -100 ignore.
        target = weapon - 1
        target = target.masked_fill(weapon == 0, -100)
        return target

    def supervised_step(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        class_weights: Mapping[str, np.ndarray | torch.Tensor],
        lr: float,
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        accumulate_only: bool = False,
        head_loss_weights: Mapping[str, float] | None = None,
        loss_scale: float = 1.0,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        optimizer = self._optimizer("bc", self.model.parameters(), lr)
        if not accumulate_only:
            optimizer.zero_grad()

        # Bench side-channel contexts (no-op for the canonical model, which
        # passes no provider). The bench provider enters label-derived
        # engagement_ema / target-supervision scopes.
        with (
            self._autocast(),
            self._side_channel_provider(actions, masks),
        ):
            _, logits, _, next_hidden, target_logits = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
            )
            valid_mask = (
                self._tensor(masks["valid_mask"], dtype=torch.bool)
                if isinstance(masks, Mapping) and "valid_mask" in masks
                else None
            )
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits,
                actions,
                class_weights=class_weights,
                head_loss_weights=head_loss_weights,
                compute_metrics=compute_metrics,
                target_logits=target_logits,
                obs=obs,
                valid_mask=valid_mask,
            )
            loss = self._mean_real_losses(losses, loss_is_real)
        (loss * float(loss_scale)).backward()
        if not accumulate_only:
            optimizer.step()

        metrics["loss"] = loss.detach()
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def evaluate_supervised(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: Mapping[str, np.ndarray | torch.Tensor],
        *,
        hidden: np.ndarray | torch.Tensor | None = None,
        masks: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | None = None,
        head_loss_weights: Mapping[str, float] | None = None,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        with (
            torch.inference_mode(),
            self._autocast(),
            self._side_channel_provider(actions, masks),
        ):
            _, logits, _, next_hidden, target_logits = self._forward_tensors(
                obs,
                hidden=hidden,
                masks=masks,
            )
            valid_mask = (
                self._tensor(masks["valid_mask"], dtype=torch.bool)
                if isinstance(masks, Mapping) and "valid_mask" in masks
                else None
            )
            losses, loss_is_real, metrics = self._compute_head_losses_and_metrics(
                logits,
                actions,
                head_loss_weights=head_loss_weights,
                compute_metrics=compute_metrics,
                target_logits=target_logits,
                obs=obs,
                valid_mask=valid_mask,
            )
        metrics["loss"] = self._mean_real_losses(losses, loss_is_real)
        metrics["_next_hidden"] = next_hidden.detach()
        return metrics

    def ppo_step(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        del args, kwargs
        raise RuntimeError("Combat-objective phase 1 does not support PPO")

    def value_step(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        del args, kwargs
        raise RuntimeError("Combat-objective phase 1 has no value head")

    def save(self, path: str | Path, *, extra_meta: Mapping[str, Any] | None = None) -> None:
        """``extra_meta`` adds provenance keys (e.g. ``run_id``) to the
        checkpoint meta + sidecar; it cannot shadow the schema keys below."""
        from qnn.contracts import current_contract

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        model_cfg = self.config.to_dict()
        meta = {
            "obs_dim": self.obs_dim,
            "model": model_cfg,
            # Declarative assembly spec (None for legacy flat/factory
            # checkpoints). When present, loaders rebuild the module via
            # qnn.model.graph.build_network — the checkpoint is fully
            # self-describing and no probe.json / factory rehydration runs.
            "model_graph": self.graph.to_dict() if self.graph is not None else None,
            # Self-versioning contract block (model↔engine). wire/semantics are
            # the LIVE engine_norm ids (what this code produces); arch is derived
            # from the ModelConfig. The checkpoint is the source of truth — the
            # exporter stamps these exact ids into the ONNX, nothing is inferred
            # from the graph or the filename. See qnn.contracts / src/docs/contracts.
            "contract": current_contract(model_cfg),
            "jump_pos_weight": self.jump_pos_weight,
            "attack_focal_gamma": self.attack_focal_gamma,
            "attack_focal_alpha": self.attack_focal_alpha,
            "attack_distance_sigma": self.attack_distance_sigma,
            "jump_distance_sigma": self.jump_distance_sigma,
            "look_label_smoothing_sigma": self.look_label_smoothing_sigma,
            # input_mask is a training-time label-rewrite toggle (not a
            # ModelConfig field) — persist it so downstream eval / final
            # val pass / PPO seed selection don't silently revert to
            # raw-demo-press labels when reloading a checkpoint that was
            # trained against the masked-outcome labels.
            "input_mask": bool(self.input_mask),
            "attack_op_only": bool(self.attack_op_only),
            "backend": "pytorch",
            "requested_device": self.device_spec.requested,
            "resolved_device": self.device_spec.resolved,
            "accelerator_backend": self.device_spec.backend,
        }
        if extra_meta:
            meta = {**dict(extra_meta), **meta}
        raw_sd = self.model.state_dict()
        clean_sd = {
            key.replace("_orig_mod.", ""): value.detach().cpu()
            for key, value in raw_sd.items()
        }
        payload = {
            "meta": meta,
            "state_dict": clean_sd,
        }
        torch.save(payload, target)
        target.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str,
        model_factory: Callable[[int, ModelConfig], nn.Module] | None = None,
    ) -> "QNNPolicy":
        """Load a saved checkpoint.

        ``model_factory`` mirrors the constructor hook: when None, the
        canonical ``Network`` is built from the saved
        ModelConfig and strict-loaded; when set, the factory builds the
        alternate module (e.g. a head-probe model) and the state_dict
        is loaded into it. The caller is responsible for passing the
        same factory used to train the checkpoint — checkpoints don't
        embed the factory identity.
        """
        source = Path(path)
        payload = trusted_torch_load(source, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload or "meta" not in payload:
            raise ValueError(f"Unrecognised checkpoint format: {source}")
        meta = dict(payload["meta"])
        if "model" not in meta:
            from qnn.utils.checkpoint_converter import migrate_legacy_flat_meta
            migrated = migrate_legacy_flat_meta(meta)
            if migrated is None:
                raise ValueError(
                    f"Checkpoint {source} is missing the 'model' arch block "
                    "and migrate_legacy_flat_meta did not recognize the schema."
                )
            meta = migrated
        # Graph-described checkpoints rebuild through the declarative
        # assembly path; flat/legacy checkpoints stay on the ModelConfig
        # path with the state-dict migrations below. An embedded graph
        # ALWAYS wins — even over a passed model_factory (a legacy
        # probe.json sitting next to a graph checkpoint must not demote
        # it back to factory-dependent), and the ModelConfig bridge is
        # re-derived from the graph so meta.model can never silently
        # diverge from the module actually built.
        graph = None
        if meta.get("model_graph") is not None:
            from qnn.model.graph import GraphSpec

            graph = GraphSpec.from_dict(meta["model_graph"])
            model_factory = None
        model_cfg = None if graph is not None else ModelConfig.from_dict(meta["model"])
        policy = cls(
            obs_dim=int(meta["obs_dim"]),
            model=model_cfg,
            graph=graph,
            # Training-loss hyperparameters — unused at inference, and absent from
            # pre-a22 checkpoints (e.g. a17). Tolerate missing with the canonical
            # train-template defaults so old deployed models keep loading (matches
            # the look_label_smoothing_sigma pattern below).
            jump_pos_weight=float(meta.get("jump_pos_weight", 5.0)),
            attack_focal_gamma=float(meta.get("attack_focal_gamma", 0.0)),
            attack_focal_alpha=float(meta.get("attack_focal_alpha", 0.5)),
            attack_distance_sigma=float(meta.get("attack_distance_sigma", 0.0)),
            jump_distance_sigma=float(meta.get("jump_distance_sigma", 0.0)),
            look_label_smoothing_sigma=float(meta.get("look_label_smoothing_sigma", 0.0)),
            seed=0,
            device=device,
            model_factory=model_factory,
        )
        # Restore the input_mask label-rewrite toggle (default False for
        # pre-fix checkpoints that didn't carry the field).
        policy.input_mask = bool(meta.get("input_mask", False))
        policy.attack_op_only = bool(meta.get("attack_op_only", False))
        # Atlas-width rebuild runs for EVERY load path, including
        # factory/graph-built (head-probe/bench) modules — unlike the
        # legacy-Network-only migrations below, this one exists specifically
        # to keep pre-migration bench checkpoints (the whole rc1 line) loading
        # under today's narrower SPATIAL_SCALAR_DIM. See its docstring.
        from qnn.utils.checkpoint_converter import migrate_legacy_spatial_atlas_dim
        migrate_legacy_spatial_atlas_dim(policy.model, payload["state_dict"])
        if model_factory is None and graph is None:
            from qnn.utils.checkpoint_converter import (
                migrate_drop_action_history,
                migrate_drop_fire_align_scalar,
                migrate_drop_weapon_embed_self,
                migrate_entity_embed,
                migrate_hoist_encoder_obs_embedding,
                migrate_obs_embedding_self_token_builder,
                migrate_rename_fire_head_to_attack_head,
                migrate_rename_tokenizer_to_obs_embedding,
                migrate_rename_trunk_to_encoder,
                migrate_self_attack_finished_scalar,
                migrate_self_scalars,
                migrate_v17_move_heads,
                migrate_wrap_gru_in_temporal,
                migrate_wrap_heads_in_components,
            )

            migrate_rename_trunk_to_encoder(payload["state_dict"])
            migrate_rename_tokenizer_to_obs_embedding(payload["state_dict"])
            migrate_hoist_encoder_obs_embedding(payload["state_dict"])
            migrate_entity_embed(payload["state_dict"])
            migrate_self_scalars(payload["state_dict"])
            migrate_self_attack_finished_scalar(payload["state_dict"])
            migrate_obs_embedding_self_token_builder(payload["state_dict"])
            migrate_v17_move_heads(payload["state_dict"])
            migrate_drop_action_history(payload["state_dict"])
            # fire_head→attack_head runs BEFORE migrate_drop_fire_align_scalar
            # so the latter sees the new ``attack_head.*`` layout.
            migrate_rename_fire_head_to_attack_head(payload["state_dict"])
            migrate_drop_fire_align_scalar(payload["state_dict"])
            migrate_drop_weapon_embed_self(payload["state_dict"])
            # gru → temporal.gru and heads → component-wrapped layout.
            # Run LAST so all prior renames have settled.
            migrate_wrap_gru_in_temporal(payload["state_dict"])
            migrate_wrap_heads_in_components(payload["state_dict"])
        # When model_factory is set the saved state_dict is for a
        # probe-built module (e.g. a Network with slot overrides from
        # qnn.model.bench), not the canonical Network — the legacy
        # migrations don't apply and the strict-load below uses empty
        # allow-prefixes.
        try:
            # strict=False so v17 checkpoints still load:
            #  - migrate_v17_move_heads packs split fb/lr into the unified
            #    move_head and bias-locks the ud axis (no random init)
            #  - migrate_drop_action_history strips the pre-rip-out
            #    action_proj / action_pos_embed weights and truncates
            #    kind_embed from 4 -> 3 rows
            #  - migrate_drop_fire_align_scalar trims the trailing
            #    alignment-scalar column from v17/v20-era attack heads
            #    (settled-null in ablation; the column is dead weight)
            #  - weapon_head / weapon_embed start fresh on v17/v20-pre-v21
            #  - encoder.gru_input_proj weight (pre-v20 mean-actors pool) is
            #    silently dropped
            missing, unexpected = policy.model.load_state_dict(payload["state_dict"], strict=False)
            if model_factory is None and graph is None:
                allowed_missing_prefixes: tuple[str, ...] = (
                    # Pre-v21 checkpoints predate the weapon head; the
                    # whole WeaponHead component (mlp + embed) starts fresh.
                    "weapon_head.",
                    # Pre-MLP-pointer checkpoints carried target_pointer.query_proj
                    # instead of the current MLP scorer; the score module starts
                    # fresh (random-init). See migrate_legacy_flat_meta's d_target note.
                    "target_pointer.score.",
                )
                allowed_unexpected_prefixes: tuple[str, ...] = (
                    "encoder.gru_input_proj.",  # pre-v20: mean-actors pool projection
                    # Pre-refactor TransformerEncoder carried an internal
                    # TargetPointer that was dead weight whenever use_gru=True
                    # (Network's own target_pointer ran instead). Now that
                    # the encoder's internal pointer is removed entirely,
                    # those keys are unexpected on load — silently drop them.
                    "encoder.target_pointer.",
                    # Pre-MLP-pointer single-Linear query projection — superseded
                    # by target_pointer.score; dropped on load.
                    "target_pointer.query_proj.",
                )
            else:
                # No legacy allowances for factory- or graph-built modules —
                # they save and load their own exact state_dict shape.
                allowed_missing_prefixes = ()
                allowed_unexpected_prefixes = ()
            missing_keep = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
            unexpected_keep = [k for k in unexpected if not k.startswith(allowed_unexpected_prefixes)]
            if missing_keep or unexpected_keep:
                raise RuntimeError(
                    f"state_dict mismatch: missing={missing_keep}, unexpected={unexpected_keep}"
                )
        except RuntimeError as exc:
            raise ValueError(
                f"Incompatible checkpoint architecture for {source}. "
                "This code expects the combat-objective BC policy layout."
            ) from exc
        policy.model.to(policy.device)
        return policy

    @classmethod
    def load_for_finetune(
        cls,
        path: str | Path,
        *,
        use_gru: bool,
        d_gru: int,
        device: str,
    ) -> "QNNPolicy":
        del use_gru, d_gru
        return cls.load(path, device=device)
