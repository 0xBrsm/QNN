"""Resolve a decode-config JSON → modules + params for export/eval.

The decode config is the self-describing, run-pinned record of the decode/guard
layer: ``decode_module`` (gen-coupled geometry) + ``guard_module`` (the guard set)
+ ``params`` + ``look_grid``/``move_hazard`` references. It replaces the static
``--decode-regime`` registry — A/B by pointing at a different JSON, no code change.

Decode is decoupled from training (you don't backprop through it), so the config
is an EXPORT-time artifact, not part of the trained graph. The exporter stamps the
resolved config's sha256 + the repo git sha into the ONNX ``metadata_props`` so the
exact decode/guard source is recoverable from a shipped model.

Schema (``decode_version`` 1):
  decode_module   dotted import path — gen decode geometry (must read the head params).
  guard_module    dotted import path | "none" — exposes guard_attack_logit_for_export +
                  policy_decode_action_postprocess.
  version         named build id (provenance only), e.g. "a24rc1".
  look_grid       path (run-relative) to the polar look grid; null = code default.
  move_hazard     path to the hazard/dwell table JSON; null = none / head-driven.
  params          flat str->scalar|list map (look.*, weapon.*, guard.*, weapon_ban).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Entry points a guard adapter MUST expose — the bit-for-bit eval/export contract.
_GUARD_REQUIRED = ("guard_attack_logit_for_export", "policy_decode_action_postprocess")

# Historical decode_module/guard_module dotted paths → their current core home.
# 2026-07-27 bench/a25 promotion (generation-scoped bench code must never be
# cross-gen-imported; a25's decode/guard facade outlived the generation, so it
# was promoted out of qnn.model.bench.a25 into qnn.model directly). Decode
# configs on disk are immutable (runs/decode_fit/*.json, deployed provenance) —
# they still name the old dotted paths and are never rewritten. This is an
# EXPLICIT rename table, not a generic alias mechanism: an unrecognized
# qnn.model.bench.a25.* path fails loud (no silent pass-through) so a typo or a
# module this table doesn't know about surfaces immediately instead of a bare
# ModuleNotFoundError three frames down.
_LEGACY_MODULE_RENAME: dict[str, str] = {
    "qnn.model.bench.a25.decode": "qnn.model.decode_actions",
    "qnn.model.bench.a25.guard": "qnn.model.guard",
}


def resolve_module_name(name: str) -> str:
    """Historical dotted path -> current core path (see _LEGACY_MODULE_RENAME),
    else passed through unchanged. Any other qnn.model.bench.a25.* path is
    unrecognized and fails loud rather than silently resolving or falling
    through to importlib's ModuleNotFoundError."""
    if name in _LEGACY_MODULE_RENAME:
        return _LEGACY_MODULE_RENAME[name]
    if name.startswith("qnn.model.bench.a25."):
        raise ValueError(
            f"{name!r}: unrecognized legacy qnn.model.bench.a25.* module path — "
            "add it to _LEGACY_MODULE_RENAME (qnn.model.decode_config) if it "
            "should resolve to a promoted core module, or fix the config if "
            "this is a typo.")
    return name

# Eval regime names → their bundled decode-config JSON. a24 is a RETIRED arch:
# its entries were pruned with its decode modules (a24 template JSONs remain on
# disk as history only — they name modules that no longer exist and will fail
# resolve_decode_config). Regime selection is picking a decode config; unknown
# names error (config_path_for). Shared by tools/export_onnx + qnn.eval.run.
_TEMPLATES = _REPO_ROOT / "src/qnn/model/bench/templates"
REGIME_CONFIGS: dict[str, Path] = {
    "a25rc1": _TEMPLATES / "decode.a25rc1.json",   # a25-owned template (seg-commitment move + 9-way attack-with): names a25 decode/guard modules so a25 decode-fit + export emit a25 refs
    "a25base": _TEMPLATES / "decode.base.json",  # a25-native BASE template (decode-fit emit base): promoted-core module pointers + only LIVE param keys, neutral placeholder values
}


def config_path_for(name_or_path: str | Path) -> Path:
    """Map a regime name (a24rc1/…) to its bundled config, or pass through a path."""
    if str(name_or_path) in REGIME_CONFIGS:
        return REGIME_CONFIGS[str(name_or_path)]
    p = Path(name_or_path)
    if p.exists():
        return p
    raise ValueError(f"unknown decode regime / config path: {name_or_path!r}")

# ── FAIL-LOUD required-key manifest ──────────────────────────────────────────
# Decode params whose ABSENCE silently changes closed-loop behavior via a code-side
# default (the run-config philosophy: no defaults in code — a config missing a
# required value must FAIL LOUD, not fall back). This is the SINGLE source of truth
# validated inside resolve_decode_config, so BOTH consumers (qnn.eval.run and
# tools/export_onnx) get the check and cannot drift apart.
#
# The manifest is PER-DECODE-MODULE, not one global tuple: some silent-default
# keys are concepts a specific decode arch owns (e.g. the a25 segment-commitment
# move decode), and requiring them of an arch whose decode law never consumes them
# would break that arch's configs for no reason. So the required set = a shared
# BASE (the look.* aim-geometry family every look-decode arch carries) UNION the
# extras registered for the config's own ``decode_module``.
#
# BASE (2026-07-13): the look.* aim-geometry family — six knobs every a25 decode-fit
# emits. Each silently flips behavior when omitted: the hazard-discounted lead caps
# default to None=OFF (a25 silently lost the a24 4/5 caps → RL over-leads and rockets
# land behind), and look.turn_mag_scale defaults to 1.0=OFF (a25 silently ran the
# a24 0.7 dampener fit on another model).
REQUIRED_PARAM_KEYS: tuple[str, ...] = (
    "look.aim_prior_gain",
    "look.aim_ffwd_gain",
    "look.aim_mag_gain",
    "look.turn_mag_scale",
    "look.lead_hold_cap_frames",
    "look.lead_hold_cap_radial_frames",
)

# PER-MODULE extras: keys required ONLY when the config names this decode_module,
# so promoting an arch-owned silent default does not force pre-a25 arches
# (qnn.model.decode) or the retired a24 line to carry keys their decode law never
# reads. a24's modules are deleted, so its configs fail at import before this check.
#
#   look.hold_passthrough absent → engaged head-commanded holds (θ==0) are silently
#                          α-blended into micro-corrections instead of passed
#                          through (a25base set true, a25rc1 had OMITTED it → a live
#                          inconsistency this promotion closes).
MODULE_REQUIRED_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "qnn.model.decode_actions": (
        "look.hold_passthrough",
        # The out-of-combat calming gate. All three halves must be stated: the
        # bias says HOW MUCH the bot chills, base/cooldown say WHEN. Until
        # 2026-08-26 only the bias was ever set and the two triggers ran on
        # code defaults no config named — the behaviour was half-configured and
        # unreadable from the config alone.
        "move.idle_none_bias",
        "move.idle_engagement_base",
        "move.idle_cooldown_ticks",
    ),
}
# attack.* / guard.* siblings are NOT yet promoted — see
# src/docs/decode-config-defaults.md for the full catalog and the deferred set.


def _required_param_keys(decode_module: str) -> tuple[str, ...]:
    """Shared BASE keys plus any registered for this config's decode_module."""
    return REQUIRED_PARAM_KEYS + MODULE_REQUIRED_PARAM_KEYS.get(decode_module, ())


def _validate_required_params(
    path: Path, params: dict[str, Any], decode_module: str) -> None:
    """FAIL LOUD when a resolved decode config omits a required param key (the
    shared BASE plus the extras registered for its decode_module). No code-side
    default is substituted — the caller must fix the config."""
    missing = [k for k in _required_param_keys(decode_module) if k not in params]
    if missing:
        raise ValueError(
            f"{path}: decode config (decode_module {decode_module!r}) missing "
            f"required param key(s) {missing}. These have no code-side default "
            f"(fail-loud policy — see REQUIRED_PARAM_KEYS / "
            f"MODULE_REQUIRED_PARAM_KEYS / src/docs/decode-config-defaults.md); add "
            f"them to the config with an explicit value (OFF is 0.0, not omission).")


def _validate_attack_vectors(path: Path, params: dict[str, Any]) -> None:
    """Shape-check the two explicit attack vectors.

    There is only ONE attack law now (the legacy JOINT attack.bias_vec was
    removed 2026-08-26), so there is nothing to disambiguate and no
    attack.vector_semantics to declare: fire_bias_vec calibrates firing,
    preference_bias_vec steers selection, and neither can do the other's job.
    """
    for key in ("attack.fire_bias_vec", "weapon.preference_bias_vec"):
        if key not in params:
            continue
        value = params[key]
        if not isinstance(value, (list, tuple)) or len(value) != 8:
            raise ValueError(f"{path}: {key} must be an 8-element vector")
        try:
            [float(x) for x in value]
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: {key} must contain numeric values") from e


# ── The one ENGINE-APPLIED decode knob (Pattern B) ───────────────────────────
# Every other knob in this file is decoded in-graph or in python. The
# continuous-weapon hold-tail is neither: it lives in the live client
# (qnn_onnx_apply_continuous_hold_tail, src/engine/common/qnn_onnx.c), which
# forces button0 down for this many seconds after a NG/SNG/LG fire so the
# server's 0.1s player_nail/player_light think-chain streams the shots between
# the model's ~0.2s op-fire decisions. There is no in-graph twin and no policy
# attribute, so it is NOT a DECODE_PARAMS row — it travels as a stamp the engine
# reads at load (`decode.attack.hold_tail_sec`).
#
# It is a decode param rather than a wire/semantics bump or a build flag because
# neither alternative can express "this model, not that one": the wire contract
# gates exactly one behavior by design (decided `move` vs raw move_logits), and
# ONE bin serves every registered codec, so a binary-wide switch changes every
# model on the share at once. The 2026-08-26 test build (env
# QNN_CONTINUOUS_HOLD_TAIL, now deleted) had exactly that defect.
#
# DEFAULT 0.0 = the tail is SCRAPPED for everything exported from here on;
# tools/export_onnx always stamps the resolved value, so a config that omits the
# key still produces a graph whose metadata states the law explicitly. Setting
# 0.25 restores the historical behavior for one model. The ENGINE's own
# absent-stamp fallback is 0.25 and applies only to already-exported artifacts
# that predate the key — see QNN_FIRE_HOLD_SEC.
#
# NOTE the python decode has never implemented a hold-tail, so offline eval and
# every decode-fit already model the tail-free world; 0.0 closes that live/offline
# asymmetry rather than widening it.
ATTACK_HOLD_TAIL_KEY: str = "attack.hold_tail_sec"
ATTACK_HOLD_TAIL_DEFAULT: float = 0.0


# Keys consumed OUTSIDE the DECODE_PARAMS registry. This list is exhaustive by
# contract: a decode config key either resolves to a working implementation
# (registry or here) or the config FAILS TO LOAD. There is no third category —
# no silently-ignored retired keys, no benign no-ops, and no parallel list of
# "rejected" keys either (that only duplicated this check with a nicer error).
# Brian 2026-08-26: "They either work or they fail. No exceptions." A key whose
# law was removed must be removed from the config too, or its law restored; a
# config that still names a dead law is not a config that can be honoured, and
# running it anyway ships a decode nobody validated.
_NON_REGISTRY_KEYS: frozenset[str] = frozenset({
    # schema fields / file refs, handled explicitly by callers
    "weapon_ban", "look_grid", "move_hazard",
    # guard.* — bound by qnn.model.guard.make_guard(params), not this registry
    "guard.projectile_release",
    "guard.lg_range",
    # attack.hold_tail_sec — the ENGINE is its consumer (see below). Not
    # silently ignored: it has a working implementation, just not a python one.
    ATTACK_HOLD_TAIL_KEY,
})


# REMOVED DECODE LAWS — why a key you remember might now refuse to load.
# There is no registry of these: a key that names a removed law is simply not
# in DECODE_PARAMS or _NON_REGISTRY_KEYS, so _validate_known_params refuses it
# like any other key with no implementation. Keeping a second parallel list of
# "rejected" keys only duplicated that check with a different error message.
# Recorded here because the RATIONALE is not recoverable from a missing entry:
#
#  * Engagement-conditioned switch margin (weapon.switch_margin_engaged /
#    weapon.switch_margin_gap) and its attack_state lane-1 anchor, 2026-08-19:
#    a rejected design, never shipped in any blessed config.
#  * The human weapon-TRANSITION law, 2026-08-26: on the live RA venue the
#    transition sampler discarded the network's own weapon choice on 52-58%
#    of firing ticks (chosen==argmax_raw 0.482 armed vs 0.742 under the plain
#    margin law), and with continue_prob_vec=[0]*8 it could not reach LG at
#    all (equip 12.5% vs 55.8%). A heuristic overriding the model, which this
#    project does not do. The SPRT evidence accumulator, fire-gather selector
#    and LG range/alignment guards went in the same clean slate.
#  * weapon.switch_margin, 2026-08-26 removal (RESTORED same day — see below):
#    the a28 obs contract carries no equip state, so the held-weapon anchor
#    it originally gated could only ever be this tick's own argmax — the
#    literal "anchor" VOCABULARY was retired for exactly that reason. But the
#    underlying math never needed equip state at all (it never did on the
#    a28 path — the anchor already degenerated to same-tick raw argmax
#    before this removal), so re-litigated and reinstated as a pure same-tick
#    confidence gate. Configs fitted under the pre-removal build still carry
#    their original values and resolve unchanged (it applied on 8.6% of
#    ticks at margin 1.496 vs 30% at margin 0 for the a28rc1e fit).
#
# Weapon SELECTION belongs to the network. Its decode-side selection knobs are
# weapon.preference_bias_vec (additive bias) and weapon.switch_margin (a gate
# on how much that bias may override the network's own same-tick raw pick —
# not an independent judgment about which weapon is good, so it does not
# violate "the network decides"). Do not reintroduce a decode-side law that
# decides WHICH weapon among the feasible set on grounds OTHER than the
# network's own logits, and do not reintroduce a held-weapon concept keyed
# off engine equip state (there is none in the obs contract).
#
# FEASIBILITY is a different question from selection and both of the
# following narrow the candidate set WITHOUT picking among what's left —
# restored 2026-08-26 (Brian) after the wholesale cut above went further than
# warranted:
#  * weapon.infeasible_vec — static per-run exclusion (see DECODE_PARAMS).
#    Unlike the removed static feasibility mask, this is ONE declared vector,
#    not a law with its own accumulator/lockout/matrix machinery riding on it.
#  * weapon.af_lockout (+ weapon.af_lockout_cap) — a switch-lockout, but
#    keyed off the engine's own self_arsenal_scalars attack_finished
#    countdown (real per-weapon refire truth, already in the obs) rather than
#    a re-derived held-weapon anchor, and no mutual-exclusion with a
#    transition/continuation/evidence law. af_lockout is a MULTIPLIER on the
#    real per-discharge cooldown (0 = none), af_lockout_cap a ceiling on the
#    extension in seconds (0 = uncapped) — the a26-lineage
#    switch_lockout_mult / switch_lockout_cap_ticks roles, generalized 2026-
#    08-26 (Brian) to a real observed value instead of a static per-weapon
#    table. Real precedent: a28rc1h shipped switch_lockout_cap_ticks=6 (0.3s
#    at 20Hz) live, formula "lockout = cd + min(cd, T)"
#    (agents/plans/rl-skill-finetune.md). See attack_with_decode's af_lockout
#    branch and ATTACK_STATE_DIM's lane doc in qnn.model.decode_actions.


def _validate_known_params(path: Path, params: dict[str, Any]) -> None:
    """Every key must resolve to a working implementation, or FAIL LOUD."""
    known = {e.key for e in DECODE_PARAMS} | _NON_REGISTRY_KEYS
    unknown = sorted(k for k in params if k not in known)
    if unknown:
        raise ValueError(
            f"{path}: decode config carries key(s) with NO implementation "
            f"{unknown}. Every key either resolves to a working decode knob "
            "(the DECODE_PARAMS registry or _NON_REGISTRY_KEYS) or the config "
            "is refused — there are no silently-ignored keys. Either remove "
            "the key(s) from the config or restore the law they name. If the "
            "key names a decode law that was REMOVED, see the "
            "\"REMOVED DECODE LAWS\" comment above this function for why, and "
            "re-emit the config rather than loading it: its other knobs were "
            "fitted against the law that is gone.")


# ── THE DECODE-PARAM REGISTRY (single source of truth) ───────────────────────
# ONE table maps every decode-config param key to the QNNPolicy attribute that
# carries it, the coercion that normalizes its JSON value, and the default used
# when the config omits it. BOTH consumers read this table and nothing else:
#
#   apply_policy_decode_params                -> ResolvedDecode.policy_attrs()
#   tools/export_onnx.main                    -> ResolvedDecode.export_kwargs()
#
# It exists because the mapping used to be hand-written at all three sites and
# DRIFTED: attack.crest_* reached the exported ONNX but was never applied in
# offline eval, so a crest-gated config shipped a decode qnn.eval never ran.
# Adding a knob to one site only is now impossible-by-construction (and
# tests/test_decode_param_registry.py fails if anyone tries).
#
# CONTRACT: ``name`` is BOTH the QNNPolicy attribute and the ExportWrapper
# kwarg. The two historical exceptions (the tremor pair) carry an explicit
# ``export_name``; nothing else may.
#
# NOT in this table (by design):
#   * guard.* — bound directly by the guard module's own make_guard(params).
#     TWO keys, each ONE quantity, each REQUIRED with no code-side default:
#       guard.projectile_release — Gate B dodge mode: "off" | "rocket" | "any"
#       guard.lg_range           — LG range in units; 0 = off
#     Both absorbed a redundant partner in 2026-08-26 (…_mode and …_u), because
#     a separate on/off flag can disagree with the value beside it.
#   * weapon_ban / look_grid / move_hazard — file refs
#     and schema fields, handled explicitly by the callers.
#   NOTE: there is no "retired key silently ignored" category any more. Keys
#   whose decode laws no longer exist (weapon.sticky_*, move.sticky_tau_*,
#   move.switchback_eps, move.stop_onset, attack.threshold, the a24 guard set)
#   now FAIL the config load — see _validate_known_params.

class _NoDefault:
    """Sentinel: this key has NO code-side default — omission must FAIL LOUD."""
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<required>"


NO_DEFAULT = _NoDefault()


def scalar_or_impulse_vec(v: Any) -> float | list[float]:
    """float, or a (9,) per-IMPULSE vector (the per-weapon skill system, 7/08):
    policy.act resolves per-row via self_weapon_id_to_impulse, and the export
    bakes the vector as a graph constant gathered by that impulse."""
    return [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)


def _axis_pair(v: Any) -> tuple[float, float]:
    """A per-axis (fb, lr) pair. A scalar broadcasts to both axes; a two-element
    list/tuple indexes; a "fb,lr" string parses (some staged configs stringify)."""
    if isinstance(v, str):
        v = [float(x) for x in v.split(",")]
    if isinstance(v, (list, tuple)):
        return (float(v[0]), float(v[1]))
    return (float(v), float(v))


def _float_vec_or_none(v: Any) -> list[float] | None:
    return None if v is None else [float(x) for x in v]


@dataclass(frozen=True)
class DecodeParam:
    """One decode knob: config key -> policy attribute / export kwarg."""
    key: str                      # decode-config ``params`` key
    name: str                     # QNNPolicy attribute AND ExportWrapper kwarg
    coerce: Any                   # callable: raw JSON value -> normalized value
    default: Any = NO_DEFAULT     # NO_DEFAULT => omission raises (fail loud)
    export_name: str | None = None  # ExportWrapper kwarg when it differs
    graph: bool = True            # False => eval-only law, no in-graph twin
    doc: str = ""

    @property
    def kwarg(self) -> str:
        """The ExportWrapper keyword this knob is passed as."""
        return self.export_name or self.name


DECODE_PARAMS: tuple[DecodeParam, ...] = (
    # ── look: aim-prior geometry (the REQUIRED_PARAM_KEYS family) ────────────
    DecodeParam("look.aim_prior_gain", "look_aim_prior_gain", scalar_or_impulse_vec,
                doc="(9,) per-impulse aim-prior blend gain; 0.0 = off"),
    DecodeParam("look.aim_ffwd_gain", "look_aim_ffwd", float,
                doc="feed-forward on the aim point's angular rate"),
    DecodeParam("look.aim_mag_gain", "look_aim_mag_gain", scalar_or_impulse_vec,
                doc="α: blend kept |z| from θ (0) toward |z+z_prior| (1)"),
    DecodeParam("look.turn_mag_scale", "look_turn_mag_scale", float,
                doc="all-humans dampener on the head's native |z|; 1.0 = off"),
    DecodeParam("look.lead_hold_cap_frames", "look_lead_hold_cap_frames", float,
                doc="hazard-discounted horizontal lead cap (20Hz frames); 0 = off"),
    DecodeParam("look.lead_hold_cap_radial_frames", "look_lead_hold_cap_radial_frames",
                float, doc="radial (approach/retreat) lead cap; 0 = off"),
    # ── look: hold handling (hold_passthrough is a25-module REQUIRED) ────────
    DecodeParam("look.hold_passthrough", "look_hold_passthrough", bool,
                doc="head-commanded exact holds (θ==0) bypass the aim-prior blend"),
    # ── look: AIM DEGRADATION (the DOWN half of the skill knob) ──────────────
    # TREMOR is the only degrader left and the only one with an in-graph twin
    # (ExportWrapper look_tremor_*). The eval-side research knobs (sluggish /
    # lag / jitter) and the feet-aim weapon-pitch family were deleted
    # 2026-08-26: absent from every a27/a28 config, never armed post-a26.
    DecodeParam("look.aim_degrade_tremor_mag", "look_aim_degrade_tremor_mag",
                scalar_or_impulse_vec, 0.0, export_name="look_tremor_mag",
                doc="AR(1)/OU angular tremor RMS (rad); 0 = off"),
    DecodeParam("look.aim_degrade_tremor_tau", "look_aim_degrade_tremor_tau",
                float, 5.0, export_name="look_tremor_tau",
                doc="tremor correlation time (frames)"),
    # ── move: the a25 segment-COMMITMENT decode ──────────────────────────────
    DecodeParam("move.commit_dur_tilt", "move_commit_dur_tilt", _axis_pair, (0.0, 0.0),
                doc="per-axis duration censoring-bias tilt"),
    # Re-commit: allow the segment head to re-commit to the held class at expiry
    # (no forced maximal-run switch), so a run can be sustained across
    # commitments. False (default) = maximal-run law, bit-identical.
    # Engagement-gated idle stillness bias: per-axis (fb,lr) additive bias on the
    # seg head's `none` class, scaled by (1 - engagement) — damps pointless idle
    # strafing when disengaged, vanishes in combat. base = E when an enemy is
    # merely present; cooldown = ticks E holds at 1 after combat. (0,0) = off.
    DecodeParam("move.idle_none_bias", "move_idle_none_bias", _axis_pair,
                doc="engagement-gated per-axis idle stillness bias ((0,0) = off)"),
    DecodeParam("move.idle_engagement_base", "move_idle_engagement_base", float,
                doc="E when an enemy is merely present"),
    DecodeParam("move.idle_cooldown_ticks", "move_idle_cooldown_ticks", int,
                doc="ticks E holds at 1 after combat"),
    # Sustained per-tick re-decision prob while an incoming projectile is
    # present (the human-shaped reactivity assist; trim target hazard 1.143)
    DecodeParam("move.threat_break_hazard", "move_threat_break_hazard", float, 0.0,
                doc="per-tick re-decision hazard under incoming threat"),
    # ── jump ─────────────────────────────────────────────────────────────────
    # Jump confidence gate (jump.threshold): the movearch jump head's posterior
    # is diffuse/under-confident (p99 ~0.17 on land frames), so sampling it
    # AS-IS scatters unmotivated hops across low-confidence frames. A threshold
    # τ>0 replaces the Bernoulli/argmax jump decode with a DETERMINISTIC gate
    # (jump iff p_jump > τ) in both the eager act path and the in-graph twin —
    # only the most-confident (context-motivated) jumps fire, and deploy==offline
    # (no jump RNG). 0.0 = off (Bernoulli sampled / argmax>0.5, bit-identical to
    # pre-knob). See runs/head_probe/_jump_calib_a26rc1a.json.
    DecodeParam("jump.threshold", "jump_threshold", float, 0.0,
                doc="deterministic jump gate p_jump > τ; 0.0 = AS-IS sampling"),
    # ── attack / weapon operating point ──────────────────────────────────────
    # Legacy joint vector: retained verbatim so existing a26 artifacts keep
    # their historical selection+fire meaning. New fits pin this to zero.
    # Reconciled explicit vectors: rate fitting and weapon-selection fitting
    # must not share a control (the a26/a27 branch-drift failure).
    DecodeParam("attack.fire_bias_vec", "attack_fire_bias_vec", _float_vec_or_none,
                None, doc="fire-only vector (split_v1)"),
    DecodeParam("weapon.preference_bias_vec", "weapon_preference_bias_vec",
                _float_vec_or_none, None, doc="selection-only vector (split_v1)"),
    # Restored 2026-08-26 (Brian) — a same-tick confidence gate on
    # preference_bias_vec, not an independent selection law (see the
    # "REMOVED DECODE LAWS" comment above). 0.0 default is a PROVABLE no-op
    # (always take the preference-adjusted ideal) — every config that omits
    # it is bit-identical to the law before this knob existed.
    DecodeParam("weapon.switch_margin", "weapon_switch_margin", float, 0.0,
                doc="same-tick raw-argmax vs preference-adjusted-ideal gate; "
                "0.0 = always ideal (no gate)"),
    # Restored 2026-08-26 (Brian) — FEASIBILITY, not selection; see the
    # "REMOVED DECODE LAWS" comment above for why these are exempt from the
    # "do not reintroduce a selection law" rule.
    DecodeParam("weapon.infeasible_vec", "weapon_infeasible_vec", _float_vec_or_none,
                None, doc="static per-weapon exclusion (>0.5 = masked to -1e9, "
                "before any selection/fire logic)"),
    DecodeParam("weapon.af_lockout", "weapon_af_lockout", float, 0.0,
                doc="switch-lockout keyed off self_arsenal_scalars "
                "attack_finished: multiplier on the real observed cooldown "
                "(0 = none/off, 1 = one more cooldown's worth, 3 = three "
                "more, etc)"),
    DecodeParam("weapon.af_lockout_cap", "weapon_af_lockout_cap", float, 0.0,
                doc="ceiling (seconds) on the af_lockout extension; 0 = "
                "uncapped. E.g. 0.3 caps the extension at 0.3s regardless "
                "of af_lockout * cooldown (a28rc1h precedent)"),
)

# Legacy view kept for provenance/tooling: param key -> policy attribute.
PARAM_TO_KWARG: dict[str, str] = {e.key: e.name for e in DECODE_PARAMS}

_DUPES = [k for k in PARAM_TO_KWARG if
          sum(1 for e in DECODE_PARAMS if e.key == k) > 1]
if _DUPES:  # pragma: no cover - import-time structural guard
    raise RuntimeError(f"DECODE_PARAMS has duplicate keys: {_DUPES}")


@dataclass
class ResolvedDecode:
    path: Path
    raw: dict
    sha256: str
    version: str
    decode_module: Any
    guard_module: Any | None
    params: dict[str, Any]
    look_grid: str | None
    move_hazard: str | None

    def _value(self, entry: DecodeParam) -> Any:
        """The normalized value of one registry knob: the config's own value
        coerced, else the registry default. A knob with NO default (the
        fail-loud manifest family) raises when the config omits it — no code
        side substitutes a value."""
        raw = self.params.get(entry.key, None)
        if raw is None:
            if entry.default is NO_DEFAULT:
                raise KeyError(
                    f"{self.path}: decode config omits required param "
                    f"{entry.key!r} (no code-side default — see "
                    f"REQUIRED_PARAM_KEYS / MODULE_REQUIRED_PARAM_KEYS)")
            return entry.default
        try:
            return entry.coerce(raw)
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(
                f"{self.path}: decode param {entry.key!r} value {raw!r} is not "
                f"coercible for policy attribute {entry.name!r}") from e

    def policy_attrs(self) -> dict[str, Any]:
        """{QNNPolicy attribute -> value} for EVERY registry knob (the eval
        and PPO runtime's whole decode operating point). Consumed by
        :func:`apply_policy_decode_params`."""
        return {e.name: self._value(e) for e in DECODE_PARAMS}

    def export_kwargs(self) -> dict[str, Any]:
        """{ExportWrapper kwarg -> value} for every registry knob the GRAPH
        implements. Consumed by tools/export_onnx. graph=False knobs are
        eval-only laws with no in-graph twin and are deliberately absent."""
        return {e.kwarg: self._value(e) for e in DECODE_PARAMS if e.graph}

    def stamp(self) -> dict[str, str]:
        """Flatten params into stampable entries (provenance). Keys are BARE — the
        exporter's build_contract_manifest adds the ``decode.`` namespace prefix."""
        out: dict[str, str] = {}
        for k, v in self.params.items():
            if isinstance(v, list):
                out[k] = ",".join(str(x) for x in v)
            elif isinstance(v, bool):
                out[k] = "1" if v else "0"
            elif isinstance(v, str):
                out[k] = v
            else:
                out[k] = repr(float(v))
        return out


def load_decode_config(path: str | Path) -> dict:
    p = Path(path)
    cfg = json.loads(p.read_text())
    if int(cfg.get("decode_version", 0)) != 1:
        raise ValueError(
            f"{p}: unsupported decode_version {cfg.get('decode_version')!r} (expected 1)")
    for k in ("decode_module", "guard_module", "version"):
        if k not in cfg:
            raise ValueError(f"{p}: missing required key {k!r}")
    return cfg


def resolve_decode_config(
    path: str | Path,
    *,
    look_grid_nbins: int | None = None,
    head_look_nbins: int | None = None,
) -> ResolvedDecode:
    """Load + validate a decode config. Imports the decode/guard modules and
    checks the guard contract. When both bin counts are given, enforces the one
    training-fixed shape constraint: the look grid's bin count must equal the
    head's trained output width (center VALUES are free to tweak; the COUNT is not).
    """
    p = Path(path)
    cfg = load_decode_config(p)
    decode_name = resolve_module_name(cfg["decode_module"])
    decode_mod = importlib.import_module(decode_name)
    params = dict(cfg.get("params", {}))
    _validate_required_params(p, params, decode_name)
    _validate_attack_vectors(p, params)
    _validate_known_params(p, params)
    guard_name = cfg["guard_module"]
    guard_mod = None
    if guard_name and guard_name != "none":
        guard_name = resolve_module_name(guard_name)
        mod = importlib.import_module(guard_name)
        if not hasattr(mod, "make_guard"):
            raise ValueError(
                f"{p}: guard_module {guard_name!r} missing make_guard(params) factory")
        # Bind the config params to the guard adapter (the policy/export consume
        # this object directly as _regime_mod). It exposes the fire-guard entry
        # points always and projectile_release_mask only when dodge is enabled.
        guard_mod = mod.make_guard(params)
        missing = [a for a in _GUARD_REQUIRED if not hasattr(guard_mod, a)]
        if missing:
            raise ValueError(
                f"{p}: guard adapter from {guard_name!r} missing entry points {missing}")
    if (look_grid_nbins is not None and head_look_nbins is not None
            and look_grid_nbins != head_look_nbins):
        raise ValueError(
            f"{p}: look grid bins ({look_grid_nbins}) != head output width "
            f"({head_look_nbins}) — incompatible decode/checkpoint pairing")
    return ResolvedDecode(
        path=p, raw=cfg, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        version=str(cfg["version"]),
        decode_module=decode_mod, guard_module=guard_mod,
        params=dict(cfg.get("params", {})),
        look_grid=cfg.get("look_grid"), move_hazard=cfg.get("move_hazard"),
    )


def install_policy_decode_modules(
    model: Any,
    regime: str | Path | None,
) -> ResolvedDecode | None:
    """Install one decode config's facade and guard on a runtime policy.

    Eval and PPO both call this helper so frozen action fields cannot silently
    run different decode modules during training and retained evaluation.
    Policy attributes are installed separately by
    :func:`apply_policy_decode_params`.
    """
    if regime is None or str(regime).strip() in ("", "none"):
        model.decode_action_postprocess = None
        model._regime_mod = None
        model._decode_mod = None
        return None
    resolved = resolve_decode_config(config_path_for(str(regime).strip()))
    adapter = resolved.guard_module
    if adapter is None:
        model.decode_action_postprocess = None
        model._regime_mod = None
    else:
        adapter._params = resolved.params
        model.decode_action_postprocess = adapter.policy_decode_action_postprocess
        model._regime_mod = adapter
    model._decode_mod = resolved.decode_module
    return resolved


def apply_policy_decode_params(model: Any, resolved: ResolvedDecode) -> None:
    """Apply every registered decode knob and shape-derived runtime flag."""
    for attr, value in resolved.policy_attrs().items():
        if not hasattr(model, attr):
            raise AttributeError(
                f"decode-param registry maps a config key onto QNNPolicy.{attr}, "
                f"which does not exist on {type(model).__name__}. Fix the "
                f"`name` in qnn.model.decode_config.DECODE_PARAMS (it is BOTH "
                f"the policy attribute and the ExportWrapper kwarg)."
            )
        setattr(model, attr, value)
    net = getattr(model, "model", model)
    model.look_commitment = (
        getattr(net, "_has_look_seg_head", False)
        and not getattr(net, "_has_look_head", False)
    )
    model.move_sticky = getattr(net, "_has_move_tick_head", False)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except Exception:
        return ""


def git_is_dirty() -> bool:
    """True if the export SOURCE has uncommitted tracked changes.

    Scoped to ``src/`` + ``tools/`` (the code + decode-config templates that
    produce the export) and ignores untracked files, so unrelated run-dir noise
    (a resident trainer rewriting ``runs/.../bc_history.json``) does not block an
    export. The exact decode JSON used is pinned separately by its sha256."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no",
             "--", "src", "tools"], cwd=_REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return False
