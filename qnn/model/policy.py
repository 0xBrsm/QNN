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
    ATTACK_FIRE_BIAS,
    ATTACK_FUTURE_HEAD,
    ATTACK_HEAD,
    ATTACK_HEAD_SIZE,
    LOOK_HEAD,
    LOOK_HEAD_SIZE,
    JUMP_HEAD,
    MOVE_HEAD,
    MOVE_HEAD_SIZE,
    MOVE_TICK_HEAD,
    ModelConfig,
    Network,
)

from qnn.schema import WEAPON_HEAD_SIZE
from qnn.utils.device import configure_torch_runtime, resolve_torch_device
from qnn.utils.io import trusted_torch_load
from qnn.vocab import ENTITY_STREAM_COMBAT, TOKEN_ACTOR


HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "target": 1.0,
    "move": 1.0,
    "look": 1.0,
    "attack": 1.0,
}

ATTACK_WEAPON_CLASS_NAMES: Tuple[Tuple[int, str], ...] = (
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


@dataclass(frozen=True, slots=True)
class _LookAimTerms:
    """Resolved per-tick look decode terms shared by the sampled look branches.

    Everything ``decode_look_from_polar`` needs downstream of the head's own
    (θ, φ).
    tick. ``z_err``/``z_rate``/``aim_range`` are None when no aim knob was
    """
    z_prior: "torch.Tensor | None"
    mag_gain: "float | torch.Tensor"
    turn_mag_scale: "float | torch.Tensor"
    z_err: "torch.Tensor | None"
    z_rate: "torch.Tensor | None"
    aim_range: "torch.Tensor | None"


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
        ``model`` (use_gru etc.) since QNNPolicy's policy-layer logic —
        hidden-state shaping and head-loss gating — reads from ``model``,
        not from the module.
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
        self.d_target = int(model.d_target)
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
        self._aim_prior_weapon_physics: torch.Tensor | None = None
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
        # a25 9-way attack-with operating point. attack_bias_vec is the legacy
        # JOINT vector retained for config compatibility. New fits use explicit
        # fire-only and selection-only vectors so rate trim cannot silently
        # steer weapon choice (the a26/a27 branch-drift failure).
        self.attack_fire_bias_vec: "list | None" = None
        self.weapon_preference_bias_vec: "list | None" = None
        # Restored 2026-08-26 (Brian) — a same-tick confidence gate on
        # preference_bias_vec, not an independent selection law; 0.0 = no gate.
        self.weapon_switch_margin: float = 0.0
        # Restored 2026-08-26 (Brian) — feasibility, not selection; see
        # decode_config's "REMOVED DECODE LAWS" comment.
        self.weapon_infeasible_vec: "list | None" = None
        self.weapon_af_lockout: float = 0.0
        self.weapon_af_lockout_cap: float = 0.0
        # a25 MOVE COMMITMENT decode (segment head → semi-Markov generative).
        # DERIVED FROM THE GRAPH, not declared: see the move_commitment
        # property below. The old move.commitment config key was deleted
        # 2026-08-26 — the model already knows whether it has a move_seg head,
        # and a config that disagreed was silently overridden anyway.
        # BENCH ARM (cell C3, agents/plans/seg-vs-frame-decision.md): the
        # revived per-tick move head's sticky-tau + dwell-hazard decode params
        # (qnn.model.move_tick_decode.MoveTickDecodeParams). NOT a
        # DECODE_PARAMS registry row and NOT a decode-config key: the arm never
        # exports, and its hazard table is ADOPTED from the run's own pinned
        # config/move_hazard.json (params_from_run_dir). None = unset; act()
        # fails loud rather than inventing a default.
        self.move_tick_params: Any = None
        # Shape-derived twin of move_commitment for the per-tick arm: eval
        # allocates the move_state lanes when it is set (qnn.eval.run).
        self.move_sticky: bool = False
        # Duration censoring-bias correction (move.commit_dur_tilt): per-axis
        # (fb, lr) bucket-index tilt applied inside move_commit_step; fit by
        # _move_seg_dur_calibration.py. (0, 0) = off, bit-identical.
        self.move_commit_dur_tilt: tuple[float, float] = (0.0, 0.0)
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
        # a25 LOOK COMMITMENT decode (look_seg segment head → semi-Markov look
        # playout): the look_seg head emits (onset-class × duration) + a direction
        # categorical at segment onsets; look_commit_step renders per-tick (θ, φ)
        # which feed the SAME aim-prior blend + expmap as the
        # polar decode (decode_look_from_polar theta/phi override — single source
        # of truth, co-decoded with the ONNX twin look_commit_step_graph).
        # Auto-enabled by prepare_act_state / eval orchestration when the model has
        # a look_seg head and NO classic look head (it is then the sole look
        # mechanism). Requires a caller-threaded look_commit_state at
        # LOOK_COMMIT_STATE_DIM. Holds are first-class (θ==0) so hold_passthrough
        # is retired by construction; per-tick tracking rides the aim-prior gain.
        self.look_commitment: bool = False
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
        # Hold pass-through (look.hold_passthrough): head-commanded exact holds
        # (θ==0) bypass the aim-prior magnitude blend, which otherwise converts
        # every engaged hold into an α·|aim-error| micro-correction (zero-hold
        # occupancy 0.000 vs human ~0.138). Default False = bit-identical.
        self.look_hold_passthrough: bool = False
        # Per-weapon VERTICAL aim authority (RL-splash feet-aiming). A (9,) per-
        # IMPULSE blend weight β∈[0,1] (0..8; 0 = OFF) for the feet-aim BLEND that
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
        self.look_aim_degrade_tremor_mag: float = 0.0
        self.look_aim_degrade_tremor_tau: float = 5.0
        #   jitter    — WHITE per-frame Gaussian angular noise on z. INCLUDED ONLY
        #     as the baseline to REJECT (it does loosen alignment/raise hbw but breaks heading-hold — the
        #     look "spin" lesson). _mag = per-frame SD in radians. 0 = OFF.
        # Per-row stateful buffers (lazily allocated when a mechanism is active;
        # reset on episode boundary). None = unallocated / OFF.
        self._aim_degrade_lp_state: torch.Tensor | None = None   # (R,2) EMA mem
        self._aim_degrade_lag_buf: torch.Tensor | None = None    # (R,L+1,2) ring
        self._aim_degrade_tremor_state: torch.Tensor | None = None  # (R,2) OU mem
        self._aim_degrade_rng: torch.Generator | None = None
        # act()-side previous-attack self-feed (Network.wants_prev_attack).
        # Train/val teacher-force obs["attack_intent_prev"] from labels
        # (_inject_prev_attack); act() has no labels, so it closes the loop on
        # its OWN previous-tick attack choice instead — see
        # _self_feed_prev_attack. Same lazily-allocated / row-count-invalidated
        # / reset_mask-zeroed convention as the aim-degrade buffers above.
        self._prev_attack_state: torch.Tensor | None = None  # (R,) long, 0..8
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

        a28 models (no per-tick move head; the jump head owns
        land-vertical) REQUIRE the commitment decode — it is mandatory for the
        shape, not a knob — so it is enabled here. Every commitment model
        needs a caller-carried ``move_commit_state`` at COMMIT_STATE_DIM,
        initialized to the decode module's reset lanes (the ONNX
        state_loopback memset). act() mutates the passed array IN PLACE: keep
        the returned array alive across steps; fancy-indexed row batches are
        copies, so scatter them back after each call. Rows are per-episode —
        re-init a row's lanes on episode reset."""
        # look_seg is the sole look mechanism (no classic look head) → the look
        # commitment decode is mandatory for the shape, exactly like move above.
        if (getattr(self.model, "_has_look_seg_head", False)
                and not getattr(self.model, "_has_look_head", False)):
            self.look_commitment = True
        kw: dict[str, np.ndarray] = {}
        if getattr(self, "move_commitment", False):
            from qnn.model.decode_actions import commit_reset_lanes
            kw["move_commit_state"] = np.tile(
                np.asarray(commit_reset_lanes(), dtype=np.float32), (n_rows, 1))
        if getattr(self, "look_commitment", False):
            from qnn.model.look_seg_decode import look_commit_reset_lanes
            kw["look_commit_state"] = np.tile(
                np.asarray(look_commit_reset_lanes(), dtype=np.float32), (n_rows, 1))
        if getattr(self.model, "aim_edge", False):
            # Alignment edge (rung-3 A′): the previous tick's per-weapon
            # alignment vector. act() updates it in place from the model's
            # ``_alignment`` aux output; zeros = "no previous tick" (episode
            # start), which the block reads as Δ-invalid.
            kw["alignment_prev"] = np.zeros((n_rows, 8), dtype=np.float32)
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
            chain = tuple(
                stage.eval().to(self.device)
                for stage in (SelfDequantizer(), SpatialDequantizer(), EntityDequantizer())
            )
            self._dequant_chain = chain
        self_dq, spatial_dq, entity_dq = chain
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

    def _aim_degrade_active(self) -> bool:
        _tr = self.look_aim_degrade_tremor_mag
        _tr_on = (any(float(x) > 0.0 for x in _tr)
                  if isinstance(_tr, (list, tuple)) else float(_tr or 0.0) > 0.0)
        return (
            _tr_on
        )

    @property
    def move_commitment(self) -> bool:
        """True iff the graph carries a move_seg head — the commitment decode is
        a property of the MODEL, not a config choice.

        Mirrors ``look_commitment`` (derived the same way) and the export's own
        ``move_commitment and self._has_move_seg``. act() already refuses a
        graph carrying BOTH move_seg and move_tick, so this is unambiguous.
        """
        return bool(getattr(self.model, "_has_move_seg_head", False))

    def _aim_degrade_reset_rows(self, reset_mask: "torch.Tensor | None") -> None:
        """Zero per-row degradation state on episode boundaries (reset_mask True)."""
        if reset_mask is None:
            return
        m = reset_mask.reshape(-1).bool()
        if self._aim_degrade_lp_state is not None:
            self._aim_degrade_lp_state[m] = 0.0
        if self._aim_degrade_tremor_state is not None:
            self._aim_degrade_tremor_state[m] = 0.0
        if self._aim_degrade_lag_buf is not None:
            self._aim_degrade_lag_buf[m] = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # ACT-SIDE PREV-ATTACK SELF-FEED (Network.wants_prev_attack; see
    # _inject_prev_attack for the train/val teacher-forced counterpart).
    #
    # Semantics matched EXACTLY: the attack.v1 label contract
    # (qnn/model/action_labels.py ATTACK_V1 — "act_attack is the class": 0 on
    # no-attack ticks, the 1..8 weapon impulse ONLY on the tick a discharge
    # intent was emitted). _inject_prev_attack builds obs["attack_intent_prev"]
    # by shifting the label column `actions["attack"]` one tick along the
    # sequence axis and zero-filling position 0 (`prev[1:] = at[:-1]`). act()
    # has no label column — its counterpart is its OWN fully-decoded attack
    # choice (`attack_choice` below, the exact value written into
    # actions["attack"]: attack_lane_gate(fire, weapon_impulse) for the a27
    # attack-with lane, matching the label convention 1:1) from the PREVIOUS
    # act() call, self-fed forward one tick — the honest non-teacher-forced
    # deployment condition. Per-row state follows the _apply_aim_degrade
    # convention: a row-count change (batch (re)alloc) drops stale state and
    # reset_mask (episode boundary) zeroes rows — both read as "no known
    # previous attack", matching _inject_prev_attack's zero-fill at sequence
    # position 0.
    def _prev_attack_reset_rows(self, reset_mask: "torch.Tensor | None") -> None:
        """Zero the self-fed previous-attack-class state on episode
        boundaries (reset_mask True) — mirrors _aim_degrade_reset_rows."""
        if reset_mask is None or self._prev_attack_state is None:
            return
        self._prev_attack_state[reset_mask.reshape(-1).bool()] = 0

    def _self_feed_prev_attack(
        self, n_rows: int, device: "torch.device",
        reset_mask: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Return this tick's obs["attack_intent_prev"]: the model's own
        attack_choice from the PREVIOUS act() call, per row. Lazily
        (re)allocates to zeros on first use or on a row-count/device change
        (the _apply_aim_degrade `_fit` pattern), then applies reset_mask.
        Caller (act()) overwrites the returned buffer in place after decoding
        this tick's attack_choice — see the wants_prev_attack block in act()."""
        st = self._prev_attack_state
        if st is None or tuple(st.shape) != (n_rows,) or st.device != device:
            st = torch.zeros(n_rows, dtype=torch.long, device=device)
        self._prev_attack_state = st
        self._prev_attack_reset_rows(reset_mask)
        return self._prev_attack_state

    def _apply_aim_degrade(
        self, look: torch.Tensor, reset_mask: "torch.Tensor | None",
        weapon_impulse: torch.Tensor,
        tremor_mag_row: "np.ndarray | None" = None,
    ) -> torch.Tensor:
        """Apply the active degradation mechanism(s) to the decoded look vector.

        look: (R, 3) view-frame unit-ish look. Returns the degraded (R, 3) unit
        look. TREMOR (AR(1)/OU on the tangent turn vector z, radians) is the only
        mechanism left — the sluggish / lag / jitter research knobs were deleted
        2026-08-26 (absent from every a27/a28 config). No-op at mag 0.

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

        # TREMOR — correlated AR(1)/OU angular offset (drifting unsteady hand).
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
                    mag_tr_t = T_vec[weapon_impulse.to(dev)].reshape(-1, 1)
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
        weapon_impulse: torch.Tensor,
        rows: int,
        device: torch.device,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Shared aim-prior geometry off the target pointer — the SINGLE eager
        computation of ``aim_prior_tangent_ffwd`` consumed by the sampled look
        decode (aim-prior blend)
        (per-row hbw). Returns ``(z_aim, z_rate, feet_elev, origin_elev,
        aim_range, eff_imp)``.

        Weapon keying comes directly from this frame's attack-with logits, so
        geometry and all per-weapon decode knobs use the same predicted intent."""
        from qnn.bc.weapon_physics import build_model_weapon_scalars
        _dmod = self._decode()
        if self._aim_prior_weapon_physics is None:
            self._aim_prior_weapon_physics = torch.from_numpy(
                build_model_weapon_scalars()).float().to(device)
        # Re-run the (idempotent, cached) dequant already applied by the caller:
        # live obs carries native keys, not entity_scalars_raw.
        dq = obs_model
        esc = dq["entity_scalars_raw"].float()
        esc = esc.reshape(rows, *esc.shape[-2:])                      # (R, N, S)
        etypes = dq["entity_types"].long().reshape(rows, -1)          # (R, N)
        eff_imp = weapon_impulse.reshape(-1).long().clamp(0, 8)
        _hold_cap = (None if not self.look_lead_hold_cap_frames
                     else float(self.look_lead_hold_cap_frames) * _dmod._TICK_DT_MODULE)
        _hold_cap_rad = (None if not self.look_lead_hold_cap_radial_frames
                         else float(self.look_lead_hold_cap_radial_frames) * _dmod._TICK_DT_MODULE)
        z_aim, z_rate, feet_elev, origin_elev, aim_range = _dmod.aim_prior_tangent_ffwd(
            esc, etypes, eff_imp,
            target_logits.reshape(rows, -1),
            self._aim_prior_weapon_physics,
            lead_hold_cap=_hold_cap,
            lead_hold_cap_radial=_hold_cap_rad,
        )
        return z_aim, z_rate, feet_elev, origin_elev, aim_range, eff_imp

    def _resolve_look_aim_terms(
        self,
        obs_model: Dict[str, torch.Tensor],
        target_logits: "torch.Tensor | None",
        attack_intent_impulse: torch.Tensor,
        rows: int,
        device: torch.device,
        per_row: "Mapping[str, np.ndarray]",
    ) -> _LookAimTerms:
        """Resolve the per-tick aim-prior / α / turn-scale terms.

        Extracted verbatim from the sampled POLAR look branch so every
        per-frame look head decodes on the SAME operating point: a head arm
        that resolved gains its own way would silently ablate the aim prior and
        make the closed-loop comparison meaningless. The gain SOURCES
        (per-lane override > per-impulse vector > scalar/AIM_*_GAIN defaults)
        are the offline path's own; the blend math lives in the shared decode
        facade. See the facade's lead_aim module and src/docs/look-head.md §5.

        NOT used by the look_seg COMMIT branch, which keeps its own copy
        deliberately (it leaves the crest gate's z_rate unset — folding it in
        here would change that path's attack decode).
        """
        _dmod = self._decode()
        assemble_aim_prior = _dmod.assemble_aim_prior
        AIM_FFWD_GAIN, AIM_PRIOR_GAIN = _dmod.AIM_FFWD_GAIN, _dmod.AIM_PRIOR_GAIN
        # gain / alpha accept a (9,) per-IMPULSE vector (per-weapon skill
        # system, 7/08): resolved per row from attack-with intent.
        # PER-LANE gain override wins: a (R,1) tensor drives assemble_aim_prior
        # per row (the SAME shape the per-impulse vector path builds), so each
        # lane blends at its own gain in one forward.
        _gain_rows = None
        if "look.aim_prior_gain" in per_row:
            _gain_rows = torch.as_tensor(
                per_row["look.aim_prior_gain"], dtype=torch.float32,
                device=device).reshape(-1, 1)
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
        z_prior = None
        _feet_elev = None
        _origin_elev = None
        _z_err = None
        _z_rate = None
        _aim_range = None
        if ((_aim_gain > 0.0 or _aim_ffwd > 0.0)
                and target_logits is not None
                and getattr(self.model, "_has_target_pointer", False)):
            # SHARED geometry (see _aim_prior_geometry: intent-keyed lead
            # tangent + anchors + pooled range); the crest gate reuses
            # z_aim/z_rate/range from this tick instead of recomputing.
            (z_aim, z_rate, _feet_elev, _origin_elev, _aim_range,
             eff_imp) = self._aim_prior_geometry(
                obs_model, target_logits, attack_intent_impulse, rows, device)
            _z_err = z_aim
            _z_rate = z_rate
            if _aim_gain > 0.0 or _aim_ffwd > 0.0:
                if _gain_rows is not None:
                    # per-LANE override (already (R,1)) — no intent keying.
                    z_prior = assemble_aim_prior(z_aim, z_rate, _gain_rows, _aim_ffwd)
                elif _gain_is_vec:
                    _gt = torch.as_tensor([float(g) for g in _gain_spec],
                                          dtype=torch.float32, device=device)
                    _gain_rows_vec = _gt[eff_imp].unsqueeze(-1)    # (R,1) intent-keyed
                    z_prior = assemble_aim_prior(z_aim, z_rate, _gain_rows_vec, _aim_ffwd)
                else:
                    z_prior = assemble_aim_prior(z_aim, z_rate, _aim_gain, _aim_ffwd)
        if "look.aim_mag_gain" in per_row:
            # PER-LANE α override: (R,) tensor → decode's mag_gain per-row
            # branch (rows at 0 are exact no-ops).
            _alpha_val = torch.as_tensor(
                per_row["look.aim_mag_gain"], dtype=torch.float32,
                device=device).reshape(-1)
        else:
            _alpha_spec = self.look_aim_mag_gain or 0.0
            if isinstance(_alpha_spec, (list, tuple)):
                # per-impulse alpha -> per-row tensor (decode's tensor path is
                # exact-no-op at 0 rows, so vec alpha is bit-compatible).
                # Intent-keyed when the aim block computed eff_imp this tick.
                _at = torch.as_tensor([float(a) for a in _alpha_spec], dtype=torch.float32,
                                      device=device)
                _alpha_val = _at[attack_intent_impulse]
            else:
                _alpha_val = float(_alpha_spec)
        _turn_scale = (
            torch.as_tensor(per_row["look.turn_mag_scale"], dtype=torch.float32,
                            device=device)
            if "look.turn_mag_scale" in per_row
            else float(self.look_turn_mag_scale
                       if self.look_turn_mag_scale is not None else 1.0))
        return _LookAimTerms(
            z_prior=z_prior,
            mag_gain=_alpha_val, turn_mag_scale=_turn_scale,
            z_err=_z_err, z_rate=_z_rate, aim_range=_aim_range,
        )

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
        look_commit_state: np.ndarray | None = None,
        alignment_prev: np.ndarray | None = None,
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
          attack : (B,)   int   — 0 = no attack; 1..8 = select-and-fire impulse.

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
            # Self-feed obs["attack_intent_prev"] (Network.wants_prev_attack)
            # from the policy's OWN previous-tick attack choice when the
            # caller hasn't already teacher-forced the column (train/val's
            # _inject_prev_attack path; a caller replaying labelled batches
            # through act() for parity tests). See _self_feed_prev_attack for
            # the exact semantics matched (attack.v1 label contract).
            _prev_attack_self_fed = (
                getattr(self.model, "wants_prev_attack", False)
                and not (isinstance(obs, Mapping) and "attack_intent_prev" in obs)
            )
            if _prev_attack_self_fed:
                _pa_sample = obs_model.get("vel")
                if _pa_sample is None:
                    _pa_sample = obs_model["self_scalars"]
                obs_model["attack_intent_prev"] = self._self_feed_prev_attack(
                    int(_pa_sample.shape[0]), _pa_sample.device, _aim_reset_mask)
            # Alignment edge (rung-3 A′): caller-carried previous-tick
            # alignment lanes (prepare_act_state contract — mutated in place
            # below). Absent → zeros inside the forward (Δ-invalid first tick).
            if alignment_prev is not None:
                obs_model["alignment_prev"] = self._tensor(
                    alignment_prev, dtype=torch.float32)
            features, logits, _, next_hidden, target_logits = self._forward_tensors(
                obs_model, hidden=hidden,
            )
            if alignment_prev is not None:
                _align_now = logits.get("_alignment")
                if _align_now is None:
                    raise RuntimeError(
                        "alignment_prev was threaded but the model emitted no "
                        "_alignment aux output — graph has no alignment edge"
                    )
                alignment_prev[:] = _align_now.float().cpu().numpy()

        sample_mode = str(mode).lower()
        if sample_mode not in ("greedy", "sampled"):
            raise ValueError(f"Unsupported policy mode: {mode}")
        temps = dict(sample_temperatures or {})

        # ---- move ----
        # a28 models have NO per-tick move head: fb/lr come from the seg
        # commit, ud from the jump head (land) and the seg head's water-ud
        # commit (deep water). The commit path is mandatory for the shape.
        move_logits = None
        n_rows = int(logits[JUMP_HEAD].reshape(-1).shape[0])
        # BENCH ARM (cell C3, agents/plans/seg-vs-frame-decision.md): a graph
        # carrying the revived per-tick move head decodes fb/lr through the
        # sticky-tau + dwell-hazard law instead of the commitment law.
        _tick_move = MOVE_TICK_HEAD in logits
        if _tick_move and "move_seg" in logits:
            raise RuntimeError(
                "graph carries BOTH move_seg and move_tick heads — the move "
                "decode would be ambiguous. Train a passenger pair if you "
                "want both, but act() must be given exactly one.")
        if not _tick_move and not (self.move_commitment and "move_seg" in logits
                                   and move_commit_state is not None):
            raise RuntimeError(
                "a28 model requires the commitment decode: enable "
                "move.commitment and thread move_commit_state "
                "(COMMIT_STATE_DIM lanes)")
        if _tick_move and move_commit_state is None:
            raise RuntimeError(
                "the per-tick move arm requires the sticky/hazard decode: "
                "thread move_commit_state (COMMIT_STATE_DIM lanes; the decode "
                "reads [0]=fb_cls [1]=fb_dwell [2]=lr_cls [3]=lr_dwell)")
        # The 9-way attack_with selector lives on the ATTACK slot.
        _wl_raw = logits.get(ATTACK_HEAD)
        is_attack_with = (
            _wl_raw is not None and int(_wl_raw.shape[-1]) == WEAPON_HEAD_SIZE + 1
        )
        _attack_intent_impulse = (
            _wl_raw.reshape(-1, WEAPON_HEAD_SIZE + 1)[..., 1:].argmax(dim=-1) + 1
            if is_attack_with
            else torch.zeros(n_rows, dtype=torch.long, device=features.device)
        )
        if _tick_move:
            # BENCH ARM — a24 sticky-tau + semi-Markov dwell-hazard decode
            # (qnn.model.move_tick_decode; cell C3 of
            # agents/plans/seg-vs-frame-decision.md). fb/lr come from the
            # sticky machine; ud is held IDENTICAL to the control arm so the
            # cell varies exactly one thing — jump head on land, the per-tick
            # head's own ud axis on deep water (where C1's move_seg water-ud
            # commit lives).
            from qnn.model.move_tick_decode import move_tick_step
            params = self.move_tick_params
            if params is None:
                raise RuntimeError(
                    "per-tick move arm: no decode params. Set "
                    "policy.move_tick_params from the run's pinned hazard "
                    "table (qnn.model.move_tick_decode.params_from_run_dir) "
                    "— there is no code-side default.")
            tick_logits = logits[MOVE_TICK_HEAD].reshape(n_rows, MOVE_AXES,
                                                         MOVE_AXIS_CLASSES)
            state_t = self._tensor(move_commit_state, dtype=torch.float32
                                   ).reshape(n_rows, -1).clone()
            tick_out = move_tick_step(
                tick_logits, state_t, params,
                greedy=(sample_mode == "greedy"),
                row_generators=row_generators)
            _cs = np.asarray(move_commit_state)
            _cs[...] = state_t.detach().cpu().numpy().reshape(_cs.shape).astype(_cs.dtype)
            # Water / jump context. DUPLICATED from the commitment branch
            # below on purpose: the arm must not perturb the control decode,
            # so the two copies are kept in lock-step by review, not by a
            # shared helper. Same approximation of QNN_PackInputMask bits 5-7
            # from self_movement_id, same anti-pogo gate on commit lane [7].
            _mv_id = self._tensor(obs["self_movement_id"], dtype=torch.long).reshape(-1)
            from qnn.engine_norm import (
                MOVEMENT_GROUND, MOVEMENT_WATER_LOW, MOVEMENT_WATER_MID,
                MOVEMENT_WATER_HIGH,
            )
            water = (_mv_id == MOVEMENT_WATER_MID) | (_mv_id == MOVEMENT_WATER_HIGH)
            _prev_press = (state_t[:, 7] > 0.5) if state_t.shape[1] > 7 \
                else torch.zeros_like(water)
            jump_ctx = ((_mv_id == MOVEMENT_GROUND)
                        | (_mv_id == MOVEMENT_WATER_LOW)) & ~_prev_press
            jl = logits[JUMP_HEAD].reshape(-1)
            p_jump = torch.sigmoid(jl)
            if self.jump_threshold > 0.0:
                jump_fire = p_jump > float(self.jump_threshold)
            elif sample_mode == "greedy":
                jump_fire = p_jump > 0.5
            elif row_generators is None:
                jump_fire = torch.rand_like(p_jump) < p_jump
            else:
                from qnn.model.decode import row_uniforms
                u = row_uniforms(row_generators, 1, p_jump.device)[:, 0].reshape(
                    p_jump.shape)
                jump_fire = u < p_jump
            jump_fire = jump_fire & jump_ctx
            move_classes = torch.ones(n_rows, MOVE_AXES, dtype=torch.long,
                                      device=tick_out.device)
            move_classes[:, :2] = tick_out[:, :2]
            ud = torch.where(jump_fire,
                             torch.full((n_rows,), MOVE_CLASS_POS,
                                        dtype=torch.long, device=tick_out.device),
                             torch.ones(n_rows, dtype=torch.long,
                                        device=tick_out.device))
            ud = torch.where(water, tick_out[:, 2], ud)
            move_classes[:, 2] = ud
            # Downstream guards argmax per-tick move logits for splash checks.
            # This arm HAS real per-tick logits, but the emitted classes are
            # what the bot actually does after the sticky gate — pass those,
            # matching the control arm's guard shim.
            _move_logits_guard = torch.nn.functional.one_hot(
                move_classes, MOVE_AXIS_CLASSES).to(torch.float32) * 10.0
        if (self.move_commitment and "move_seg" in logits
                and move_commit_state is not None):
            # a25 COMMITMENT decode: fb/lr from the segment head's sampled
            # (class, duration) commitments (expiry/Gate-B-interrupt/reset →
            # resample; held class masked at expiry only — an interrupt is a
            # re-decision); ud from the cross-gen base decode.
            # Caller threads move_commit_state (R,4) [fb_cls,fb_rem,lr_cls,
            # lr_rem], cls<0 = unset — MUTATED IN PLACE like the swb state.
            from qnn.model.decode_actions import move_commit_step
            from qnn.model.decode import decode_move_axes
            # Slice axes, don't blind-reshape: a water_ud seg head emits
            # (N, 3, JOINT) and reshape(n_rows, 2, -1) would silently garble
            # fb/lr into (N, 2, 45). The commit decode is fb/lr-only; the
            # water-ud axis gets its own decode gate later.
            seg_all = logits["move_seg"]
            if seg_all.dim() != 3:
                seg_all = seg_all.reshape(n_rows, 2, -1)
            seg_logits = seg_all[:, :2, :]
            # Gate B commit-interrupt (move.commit_interrupt) was deleted
            # 2026-08-26: OFF in every config, so the commit decode never took
            # a projectile release. The guard's own move-hold release
            # (guard.projectile_release) is a DIFFERENT consumer and is
            # untouched, as is the threat-break lever on `threat=`.
            projectile_release = None
            commit_t = self._tensor(move_commit_state, dtype=torch.float32).reshape(n_rows, -1).clone()
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
                from qnn.model.decode_actions import move_threat_signal
                _threat = move_threat_signal(obs_model, n_rows)
            # Engagement signals (external) for the idle stillness bias.
            _enemy_present = _engaged_active = None
            _ammo_pools = _held_impulse = None
            if self.move_idle_none_bias != (0.0, 0.0):
                from qnn.model.decode_actions import move_engagement_signals
                (_enemy_present, _engaged_active,
                 _ammo_pools, _held_impulse) = move_engagement_signals(
                    obs_model, n_rows)
            commit_out = move_commit_step(
                seg_all, commit_t,
                release=projectile_release,
                greedy=(sample_mode == "greedy"),
                row_generators=row_generators,
                dur_tilt=self.move_commit_dur_tilt,
                water=water,
                threat=_threat,
                threat_break_hazard=float(self.move_threat_break_hazard),
                enemy_present=_enemy_present,
                engaged_active=_engaged_active,
                idle_none_bias=self.move_idle_none_bias,
                idle_engagement_base=float(self.move_idle_engagement_base),
                idle_cooldown_ticks=int(self.move_idle_cooldown_ticks),
                ammo_pools=_ammo_pools, held_impulse=_held_impulse)
            fblr = commit_out[:, :2]
            _cs = np.asarray(move_commit_state)
            _cs[...] = commit_t.detach().cpu().numpy().reshape(_cs.shape).astype(_cs.dtype)
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
        move = (move_classes.float() - float(MOVE_CLASS_NONE))             # {-1, 0, +1} per axis

        # PER-LANE decode overrides (aim-grid enabler): resolve once now that
        # n_rows is known. ``_pr`` maps a dotted decode key → a float32 (n_rows,)
        # array; empty ⇒ the instance-scalar path (bit-identical to no-kwarg).
        _pr = self._resolve_per_row_decode(per_row_decode, n_rows)

        # aim-geometry locals from the look block. No longer consumed by the
        # attack decode (the crest gate that read them is gone); retained only
        # because the sampled-look path unpacks them positionally.
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
        _use_look_commit = (getattr(self, "look_commitment", False)
                            and look_commit_state is not None
                            and "look_seg" in logits)
        if _use_look_commit:
            # a25 LOOK COMMITMENT decode: look_seg (onset-class × duration) +
            # direction → per-tick (θ, φ) via look_commit_step, then the SHARED
            # aim-prior blend + expmap (decode_look_from_polar
            # θ/φ override). Runs in BOTH greedy and sampled modes — holds/strokes
            # ARE the look mechanism (no per-frame readout fallback). The
            # aim-prior/alpha resolution mirrors the polar block below and is
            # kept self-contained so the classic-look path stays byte-for-byte
            # untouched. Holds are first-class (θ==0); no-enemy holds stay exact
            # (z_prior is zero off-target), engaged holds micro-correct toward the
            # target (per-tick tracking rides the aim-prior gain — "tracking never
            # waits out a stroke", agents/plans/look-seg-head.md).
            from qnn.model.look_seg_bins import (
                JOINT as _LJOINT, N_LOOK_DIR as _LNDIR)
            from qnn.model.look_seg_decode import look_commit_step
            _dmod = self._decode()
            assemble_aim_prior = _dmod.assemble_aim_prior
            decode_look_from_polar = _dmod.decode_look_from_polar
            AIM_FFWD_GAIN, AIM_PRIOR_GAIN = _dmod.AIM_FFWD_GAIN, _dmod.AIM_PRIOR_GAIN
            seg_look = logits["look_seg"].reshape(-1, _LJOINT + _LNDIR)
            _look_dev = seg_look.device
            _look_rows = seg_look.shape[0]
            # commit playout (per-tick θ, φ); mutate look_commit_state in place.
            _lc = self._tensor(look_commit_state, dtype=torch.float32
                               ).reshape(_look_rows, -1).clone()
            theta, phi = look_commit_step(
                seg_look, _lc, hz=self.model.look_seg_head.hz,
                greedy=(sample_mode != "sampled"),
                row_generators=(row_generators if sample_mode == "sampled" else None))
            _lc_np = np.asarray(look_commit_state)
            _lc_np[...] = _lc.detach().cpu().numpy().reshape(
                _lc_np.shape).astype(_lc_np.dtype)
            # aim-prior + alpha resolution (mirror of the polar block).
            _gain_rows = None
            if "look.aim_prior_gain" in _pr:
                _gain_rows = torch.as_tensor(
                    _pr["look.aim_prior_gain"], dtype=torch.float32,
                    device=_look_dev).reshape(-1, 1)
                _gain_is_vec = False
                _aim_gain = float(_gain_rows.max())
            else:
                _gain_spec = (AIM_PRIOR_GAIN if self.look_aim_prior_gain is None
                              else self.look_aim_prior_gain)
                _gain_is_vec = isinstance(_gain_spec, (list, tuple))
                _aim_gain = (max(float(g) for g in _gain_spec) if _gain_is_vec
                             else float(_gain_spec))
            _aim_ffwd = (AIM_FFWD_GAIN if self.look_aim_ffwd is None
                         else float(self.look_aim_ffwd))
            z_prior = None
            _feet_elev = None
            _origin_elev = None
            if ((_aim_gain > 0.0 or _aim_ffwd > 0.0)
                    and target_logits is not None
                    and getattr(self.model, "_has_target_pointer", False)):
                (z_aim, z_rate, _feet_elev, _origin_elev, _crest_aim_range,
                 eff_imp) = self._aim_prior_geometry(
                    obs_model, target_logits, _attack_intent_impulse,
                    _look_rows, _look_dev)
                _crest_z_err = z_aim
                if _aim_gain > 0.0 or _aim_ffwd > 0.0:
                    if _gain_rows is not None:
                        z_prior = assemble_aim_prior(z_aim, z_rate, _gain_rows, _aim_ffwd)
                    elif _gain_is_vec:
                        _gt = torch.as_tensor([float(g) for g in _gain_spec],
                                              dtype=torch.float32, device=_look_dev)
                        _gain_rows_vec = _gt[eff_imp].unsqueeze(-1)
                        z_prior = assemble_aim_prior(z_aim, z_rate, _gain_rows_vec, _aim_ffwd)
                    else:
                        z_prior = assemble_aim_prior(z_aim, z_rate, _aim_gain, _aim_ffwd)
            if "look.aim_mag_gain" in _pr:
                _alpha_val = torch.as_tensor(
                    _pr["look.aim_mag_gain"], dtype=torch.float32,
                    device=_look_dev).reshape(-1)
            else:
                _alpha_spec = self.look_aim_mag_gain or 0.0
                if isinstance(_alpha_spec, (list, tuple)):
                    _at = torch.as_tensor([float(a) for a in _alpha_spec],
                                          dtype=torch.float32, device=_look_dev)
                    _alpha_val = _at[_attack_intent_impulse]
                else:
                    _alpha_val = float(_alpha_spec)
            _turn_scale = (
                torch.as_tensor(_pr["look.turn_mag_scale"], dtype=torch.float32,
                                device=_look_dev)
                if "look.turn_mag_scale" in _pr
                else float(self.look_turn_mag_scale
                           if self.look_turn_mag_scale is not None else 1.0))
            look = decode_look_from_polar(
                z_prior=z_prior,
                mag_gain=_alpha_val,
                turn_mag_scale=_turn_scale,
                theta_override=theta, phi_override=phi,
            ).reshape(-1, LOOK_HEAD_SIZE)
        elif sample_mode == "sampled" and "_look_mag_logits" in logits:
            from qnn.model.look_bins import DIR_CENTERS, MAG_CENTERS, N_DIR, N_MAG
            _dmod = self._decode()
            decode_look_from_polar = _dmod.decode_look_from_polar
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
            # AIM PRIOR (pointer-bearing models only): the PRE-SCALED blend
            # z_prior = gain·z_aim + ffwd·z_rate (zero on no-enemy frames), the
            # feet-aim pitch terms, α and the turn dampener — all resolved by
            # _resolve_look_aim_terms, which every per-frame look head shares so
            # no head arm can silently decode at a different operating point.
            _aim = self._resolve_look_aim_terms(
                obs_model, target_logits, _attack_intent_impulse,
                int(mag_bin.shape[0]), mag_logits.device, _pr)
            _crest_z_err, _crest_z_rate = _aim.z_err, _aim.z_rate
            _crest_aim_range = _aim.aim_range
            look = decode_look_from_polar(
                mag_bin, dir_logits,
                MAG_CENTERS.to(mag_logits.device), DIR_CENTERS.to(dir_logits.device),
                _aim.z_prior,
                mag_gain=_aim.mag_gain,
                turn_mag_scale=_aim.turn_mag_scale,
                hold_passthrough=bool(self.look_hold_passthrough),
            ).reshape(-1, LOOK_HEAD_SIZE)
        elif sample_mode == "sampled" and "_look_hold_logit" in logits:
            # XM CONTINUOUS look head (bench arm, qnn.model.look_head_xm):
            # sample the noise-free HOLD bit, then draw ONE noise latent and
            # regress the turn tangent. θ/φ ride decode_look_from_polar's
            # override path, so the aim-prior blend, feet-aim pitch,
            # turn_mag_scale, hold_drift/passthrough and the HOLD semantics are
            # the polar branch's — a hold is θ==0 exactly, as polar's mag_bin==0
            # rows are, and the aim prior treats it the same way (pure rotation
            # keeps holds exact; mag_gain>0 micro-corrects toward the target).
            from qnn.model.decode import bernoulli_sample, row_uniforms
            _dmod = self._decode()
            decode_look_from_polar = _dmod.decode_look_from_polar
            _xm = self.model.look_head
            t_look = max(float(temps.get("look", 1.0)), 1e-6)
            _feat = logits["_look_features"].reshape(-1, _xm.in_dim)
            _look_dev = _feat.device
            _look_rows = int(_feat.shape[0])
            hold_logit = logits["_look_hold_logit"].reshape(-1) / t_look
            hold = bernoulli_sample(torch.sigmoid(hold_logit).float(),
                                    row_generators).bool()
            with torch.inference_mode():
                # The turn MLP re-runs here on the head's stashed features (the
                # forward emits only its diagnostic single draw). inference_mode
                # because those features are inference tensors — an
                # autograd-recording op on them raises.
                if row_generators is None:
                    noise = torch.randn((_look_rows, _xm.d_noise),
                                        device=_look_dev, dtype=_feat.dtype)
                else:
                    # Per-ROW normal draws off the SAME seeded uniform streams
                    # the categorical/Bernoulli samplers use (probit inverse-CDF)
                    # — one generator per lane, so eval stays reproducible.
                    _u = row_uniforms(row_generators, _xm.d_noise, _look_dev
                                      ).clamp(1e-6, 1.0 - 1e-6)
                    noise = ((2.0 ** 0.5) * torch.erfinv(2.0 * _u - 1.0)).to(_feat.dtype)
                z_model = _xm.turn_from_noise(_feat, noise).float()   # (R, 2)
            theta = torch.linalg.vector_norm(z_model, dim=-1)
            phi = torch.atan2(z_model[..., 1], z_model[..., 0])
            theta = torch.where(hold, torch.zeros_like(theta), theta)
            _aim = self._resolve_look_aim_terms(
                obs_model, target_logits, _attack_intent_impulse,
                _look_rows, _look_dev, _pr)
            _crest_z_err, _crest_z_rate = _aim.z_err, _aim.z_rate
            _crest_aim_range = _aim.aim_range
            look = decode_look_from_polar(
                z_prior=_aim.z_prior,
                mag_gain=_aim.mag_gain,
                turn_mag_scale=_aim.turn_mag_scale,
                hold_passthrough=bool(self.look_hold_passthrough),
                theta_override=theta, phi_override=phi,
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
                look, _aim_reset_mask, _attack_intent_impulse,
                tremor_mag_row=_tremor_row)

        # ---- categorical attack ----
        # a25 attack-with: ONE 9-way head owns both decisions. Detected by the
        # attack slot's logit width (8+1).
        # Greedy joint decode (qnn.model.decode_actions) — deterministic argmax
        # preserves the rocket-jump coupling; align bias + hard guards compose on
        # the class-0 logit / a veto mask. Stateless: no hold-tail, no sticky gate.
        if is_attack_with:
            from qnn.model.decode_actions import (
                attack_lane_gate, attack_with_decode, attack_with_marginal_logit,
            )
            logits9 = _wl_raw.reshape(-1, WEAPON_HEAD_SIZE + 1)
            _fire_bias_vec = (None if self.attack_fire_bias_vec is None
                              else self._tensor(self.attack_fire_bias_vec,
                                                dtype=torch.float32).reshape(-1))
            # The checkpoint owns fire calibration.  External decode bias is
            # retained only as a backwards-compatible additive override; new
            # idea-model runs pin it to zero.  Neither vector touches weapon
            # selection in attack_with_decode_step.
            _model_fire_bias = logits[ATTACK_FIRE_BIAS].reshape(
                -1, WEAPON_HEAD_SIZE,
            )[0].to(torch.float32)
            _fire_bias_vec = (
                _model_fire_bias if _fire_bias_vec is None
                else _fire_bias_vec + _model_fire_bias
            )
            _preference_bias_vec = (
                None if self.weapon_preference_bias_vec is None
                else self._tensor(self.weapon_preference_bias_vec,
                                  dtype=torch.float32).reshape(-1))
            _infeasible_vec = (
                None if self.weapon_infeasible_vec is None
                else self._tensor(self.weapon_infeasible_vec,
                                  dtype=torch.float32).reshape(-1))
            # ONE shared decode (qnn.model.decode_actions.attack_with_decode) —
            # the SAME call the ONNX ExportWrapper bakes, so the offline and the
            # deployed attack decode (anchor-impulse + guard align/veto + per-weapon
            # operating point) cannot skew. No decode logic inline here.
            _att_state_t = (None if attack_state is None
                            else self._tensor(attack_state, dtype=torch.float32
                                              ).reshape(int(logits9.shape[0]), -1))
            # Weapon selection is the preference-adjusted argmax — there is no
            # equip state in the obs contract and no held-weapon anchor.
            _dq_attack = self._obs_tensors_dequant(obs)
            fire, weapon_impulse, _att_state_out = attack_with_decode(
                logits9,
                _dq_attack,
                _move_logits_guard,
                guard=self._regime_mod,
                fire_bias_vec=_fire_bias_vec,
                preference_bias_vec=_preference_bias_vec,
                switch_margin=float(self.weapon_switch_margin),
                infeasible_vec=_infeasible_vec,
                attack_state=_att_state_t,
                af_lockout=float(self.weapon_af_lockout),
                af_lockout_cap=float(self.weapon_af_lockout_cap))
            if attack_state is not None and _att_state_out is not None:
                # weapon.af_lockout mutates its state in place — same
                # caller-threaded convention as move_commit_state/
                # look_commit_state (a no-op write when af_lockout is off).
                _as = np.asarray(attack_state)
                _as[...] = _att_state_out.detach().cpu().numpy().reshape(
                    _as.shape).astype(_as.dtype)
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

        if is_attack_with:
            # a27 single-lane action convention ("attack WITH weapon X this
            # tick", 0 = no attack): gate the decode's impulse on the fire bit.
            # attack_with_decode emits the SELECTED impulse on every tick —
            # ungated on the a27 wire it presses the trigger every tick, since
            # the engine derives button0 from `attack != 0`.
            # attack_lane_gate is the SHARED kernel the ONNX export bakes, so
            # offline and deploy cannot fork. (Imported in the is_attack_with
            # decode block above — the same condition guards this use.)
            attack_choice = attack_lane_gate(fire, weapon_impulse)
        else:
            attack_choice = fire.to(torch.long)

        if _prev_attack_self_fed:
            # Carry THIS tick's fully-decoded attack choice forward as next
            # tick's self-fed obs["attack_intent_prev"] — attack_choice is
            # exactly actions["attack"] below, the attack.v1 label column.
            self._prev_attack_state = attack_choice.detach().to(torch.long)

        if True:
            actions = {
                "move":   move.detach().cpu().numpy().astype(np.float32),
                "look":   look.detach().cpu().numpy().astype(np.float32),
                "attack": attack_choice.detach().cpu().numpy().astype(np.int64),
            }
            if is_attack_with:
                # a27: the attack lane FOLDS fire and weapon selection, so the
                # raw fire bit rides beside it as its own channel — a consumer
                # that wants the model's attack decision (eval/h2h attack
                # logging) reads it directly instead of re-deriving the lane
                # convention. The engine bridge packs named keys only
                # (src/engine/bridge.py pack_step_batch / pack_step_request), so
                # the extra key is inert on the wire.
                actions["fire"] = fire.detach().cpu().numpy().astype(np.int64)

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

        look_prior = None
        look_delta = None
        look_predict = _np(logits[LOOK_HEAD]).reshape(-1, 3)
        align = np.full(look_predict.shape[0], np.nan, dtype=np.float32)

        move_logits_np = None
        move_prob_np = None
        attack_logit_np = _np(attack_logit).reshape(-1).tolist()
        fire_prob_np = _np(fire_prob).reshape(-1).tolist()

        # Attack internals: categorical logits, intent, and confidence.
        weapon_logits_np = weapon_desired = weapon_conf = None
        if ATTACK_HEAD in logits and logits[ATTACK_HEAD].shape[-1] == WEAPON_HEAD_SIZE + 1:
            wl = logits[ATTACK_HEAD]
            wl = wl.reshape(-1, wl.shape[-1])
            wp = torch.softmax(wl, dim=-1)
            weapon_logits_np = _np(wl).tolist()
            weapon_desired = wp.argmax(dim=-1).detach().cpu().numpy().reshape(-1).tolist()
            weapon_conf = wp.max(dim=-1).values.detach().cpu().numpy().reshape(-1).tolist()

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
                        "desired": (int(weapon_desired[i]) if weapon_desired is not None else None),
                        "conf": (float(weapon_conf[i]) if weapon_conf is not None else None),
                        "logits": (weapon_logits_np[i] if weapon_logits_np is not None else None),
                    },
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

        move_tick_loss_fn = getattr(
            getattr(self.model, "move_tick_head", None), "move_tick_loss", None)
        if (move_tick_loss_fn is not None and MOVE_TICK_HEAD in logits
                and MOVE_HEAD in actions):
            # BENCH ARM (cell C3, agents/plans/seg-vs-frame-decision.md): the
            # revived per-tick move head owns its loss. Labels are the SAME
            # per-tick act_move classes the seg head derives its segments from
            # — no loader change, no new column.
            mt_loss, _mt = move_tick_loss_fn(
                logits, actions, valid_flat, compute_metrics,
                ud_weight=float(weights_map.get("move_tick_ud", 1.0)),
                input_mask_on=input_mask_on)
            losses.append(mt_loss * float(weights_map.get("move_tick", 1.0)))
            loss_is_real.append(True)
            if compute_metrics and _mt:
                metrics.update(_mt)

        attack_loss_owned = False
        attack_loss_fn = getattr(getattr(self.model, "attack_head", None), "attack_loss", None)
        if attack_loss_fn is not None and ATTACK_HEAD in logits and ATTACK_HEAD in actions:
            attack_loss, attack_metrics = attack_loss_fn(
                logits, actions, valid_flat, compute_metrics, obs=obs,
            )
            losses.append(attack_loss * weights_map.get(ATTACK_HEAD, 1.0))
            loss_is_real.append(True)
            attack_loss_owned = True
            if compute_metrics and attack_metrics:
                metrics.update(attack_metrics)
        look_seg_loss_fn = getattr(getattr(self.model, "look_seg_head", None), "look_seg_loss", None)
        if look_seg_loss_fn is not None and "look_seg" in logits and LOOK_HEAD in actions:
            # a25 look segment head owns its loss (labels derived on the fly from
            # the sequence look turn-delta vectors; frame-shuffled batches
            # contribute nothing). Passenger: the per-frame look head is untouched.
            lseg_loss, _ls = look_seg_loss_fn(
                logits, actions, valid_mask, compute_metrics,
                dir_weight=float(weights_map.get("look_seg_dir", 1.0)))
            losses.append(lseg_loss * float(weights_map.get("look_seg", 1.0)))
            loss_is_real.append(True)
            if compute_metrics and _ls:
                metrics.update(_ls)

        attack_future_loss_fn = getattr(
            getattr(self.model, "attack_future_head", None), "attack_future_loss", None)
        if attack_future_loss_fn is not None and ATTACK_FUTURE_HEAD in logits:
            # a27 MTP aux head owns its loss (label = the per-episode censored
            # time-to-next-op-discharge bucket the BC source derives; see
            # qnn.model.attack_future_head). Training-only passenger — the head
            # is skipped entirely at export, so this block never runs there.
            af_loss, _afm = attack_future_loss_fn(
                logits, actions, valid_flat, compute_metrics)
            losses.append(af_loss * float(weights_map.get(ATTACK_FUTURE_HEAD, 1.0)))
            loss_is_real.append(True)
            if compute_metrics and _afm:
                metrics.update(_afm)

        if LOOK_HEAD in logits and LOOK_HEAD in actions:
            # The polar look head owns its loss (hierarchical mag × dir CE)
            # via the look_loss hook; the policy layer only shapes the label.
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
            look_loss_fn = self.model.look_head.look_loss
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
            losses.append(look_loss * weights_map.get(LOOK_HEAD, 1.0))
            loss_is_real.append(aux_is_real)
            if compute_metrics:
                metrics["loss_look"] = look_loss.detach()

        if compute_metrics:
            metrics["accuracy"] = (
                torch.stack(accuracy_components).mean()
                if accuracy_components
                else torch.zeros((), device=self.device)
            )
        else:
            metrics["accuracy"] = torch.zeros((), device=self.device)

        return losses, loss_is_real, metrics


    def _inject_prev_attack(self, obs, actions):
        """Teacher-force the previous-tick attack column:
        obs["attack_intent_prev"][t] = act_attack[t-1], episode-shifted along
        the sequence axis. Chunk-first rows get class 0 (one teacher-noise row
        per tbptt chunk). Train/val only — act() feeds the bot's own previous
        sampled attack instead.

        Sole consumer (Network.wants_prev_attack): the ``prev_attack`` intent
        node. (The alignment ``aim`` edge computed per-weapon since the
        rung-3 A′ redesign and no longer keys on the previous tick.)"""
        if not getattr(self.model, "wants_prev_attack", False):
            return obs
        if not isinstance(obs, dict) or "attack_intent_prev" in obs:
            return obs
        attack = actions.get("attack") if isinstance(actions, Mapping) else None
        if attack is None:
            raise ValueError(
                "attack_intent_prev needs actions['attack'] to teacher-force")
        at = self._tensor(attack, dtype=torch.long)
        if at.ndim == 1:
            raise ValueError(
                "attack_intent_prev requires sequence batches (T, B); "
                "flat frame-shuffled batches have no previous tick")
        prev = torch.zeros_like(at)
        prev[1:] = at[:-1]
        out = dict(obs)
        out["attack_intent_prev"] = prev
        return out

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
        obs = self._inject_prev_attack(obs, actions)
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
        obs = self._inject_prev_attack(obs, actions)
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
            # Self-versioning contract block (model↔engine). wire/semantics
            # follow the graph's entity_stream (combat -> the LIVE engine_norm
            # ids this code produces; full -> the a26-line pair); arch is
            # derived from the ModelConfig. The checkpoint is the source of
            # truth — the exporter stamps these exact ids into the ONNX,
            # nothing is inferred from the graph shape or the filename. See
            # qnn.contracts / src/docs/contracts.
            "contract": current_contract(
                model_cfg,
                entity_stream=(self.graph.entity_stream if self.graph is not None
                               else ENTITY_STREAM_COMBAT),
            ),
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
        # a28: every loadable checkpoint embeds its graph. Pre-a28 (flat
        # ModelConfig / legacy-migration) checkpoints load from their own
        # branches — this line refuses them loud.
        graph = None
        if meta.get("model_graph") is None and model_factory is None:
            raise ValueError(
                f"Checkpoint {source} carries no model_graph — pre-a28 "
                "checkpoints load from their own branch, not this line."
            )
        if meta.get("model_graph") is not None:
            import dataclasses

            from qnn.model.graph import GraphSpec
            from qnn.utils.checkpoint_converter import sniff_entity_stream

            graph = GraphSpec.from_dict(meta["model_graph"])
            model_factory = None
            # Entity-stream selection: the state dict is the source of truth
            # (a26-line checkpoints predate the spec key). A sniffed FULL
            # stream upgrades the spec default; a spec that explicitly names
            # a stream its own weights contradict fails loud.
            sniffed = sniff_entity_stream(payload["state_dict"])
            if sniffed is not None and sniffed != graph.entity_stream:
                if "entity_stream" in meta["model_graph"]:
                    raise ValueError(
                        f"Checkpoint {source} declares "
                        f"entity_stream={graph.entity_stream!r} but its "
                        f"weights are the {sniffed!r} stream"
                    )
                graph = dataclasses.replace(graph, entity_stream=sniffed)
        model_cfg = None if graph is not None else ModelConfig.from_dict(meta["model"])  # factory path
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
        try:
            # fire_bias was added as an always-present fire-only calibration
            # vector without changing the graph or wire contract.  Older
            # checkpoints are exactly the zero-intercept case; tolerate only
            # that one missing tensor and keep every other mismatch fatal.
            missing, unexpected = policy.model.load_state_dict(payload["state_dict"], strict=False)
            allowed_missing = {"attack_head.fire_bias"}
            bad_missing = [k for k in missing if k not in allowed_missing]
            if bad_missing or unexpected:
                raise RuntimeError(
                    f"state_dict mismatch: missing={bad_missing}, "
                    f"unexpected={unexpected}"
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
