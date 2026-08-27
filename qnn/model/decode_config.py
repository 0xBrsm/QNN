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
#   move.commitment       absent → the a25 segment-commitment move decode silently
#                          reverts to per-axis sampling (HIGH risk; both live a25
#                          templates intend true).
#   look.hold_passthrough absent → engaged head-commanded holds (θ==0) are silently
#                          α-blended into micro-corrections instead of passed
#                          through (a25base set true, a25rc1 had OMITTED it → a live
#                          inconsistency this promotion closes).
MODULE_REQUIRED_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "qnn.model.decode_actions": (
        "move.commitment",
        "look.hold_passthrough",
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
    """Validate the reconciled attack-vector contract without reinterpreting
    legacy configs. A config with no explicit vectors remains an old a26-style
    artifact; any config that opts into split semantics must name split_v1,
    carry all three vectors, and neutralize the branch-dependent joint vector."""
    keys = ("attack.bias_vec", "attack.fire_bias_vec",
            "weapon.preference_bias_vec")
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if not isinstance(value, (list, tuple)) or len(value) != 8:
            raise ValueError(f"{path}: {key} must be an 8-element vector")
        try:
            [float(x) for x in value]
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: {key} must contain numeric values") from e
    semantics = params.get("attack.vector_semantics")
    split_claimed = semantics is not None or any(k in params for k in keys[1:])
    if not split_claimed:
        return
    if semantics != "split_v1":
        raise ValueError(
            f"{path}: explicit fire/preference vectors require "
            "attack.vector_semantics='split_v1'")
    missing = [k for k in keys if k not in params]
    if missing:
        raise ValueError(f"{path}: split_v1 missing required vector(s) {missing}")
    legacy = params["attack.bias_vec"]
    if any(abs(float(x)) > 1e-9 for x in legacy):
        raise ValueError(
            f"{path}: split_v1 requires zero legacy attack.bias_vec; its "
            "nonzero meaning diverged between a26 and a27")


# ── THE DECODE-PARAM REGISTRY (single source of truth) ───────────────────────
# ONE table maps every decode-config param key to the QNNPolicy attribute that
# carries it, the coercion that normalizes its JSON value, and the default used
# when the config omits it. BOTH consumers read this table and nothing else:
#
#   qnn.eval.run._apply_decode_config_params  -> ResolvedDecode.policy_attrs()
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
#   * guard.* — consumed by the guard module, stamped for provenance only.
#   * weapon_ban / look_grid / move_hazard / attack.vector_semantics — file refs
#     and schema fields, handled explicitly by the callers.
#   * retired a24 keys (weapon.sticky_*, move.sticky_tau_*, move.switchback_eps,
#     move.stop_onset, attack.threshold) — their decode laws no longer exist, so
#     a config still carrying them simply has those keys ignored.

class _NoDefault:
    """Sentinel: this key has NO code-side default — omission must FAIL LOUD."""
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<required>"


NO_DEFAULT = _NoDefault()


def scalar_or_impulse_vec(v: Any) -> float | list[float]:
    """float, or a (9,) per-IMPULSE vector (the per-weapon skill system, 7/08):
    policy.act resolves per-row via self_weapon_id_to_impulse, and the export
    bakes the vector as a graph constant gathered by the held impulse."""
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
    DecodeParam("look.hold_drift_eps", "look_hold_drift_eps", float, 0.0,
                doc="on hold frames with a live prior, drift eps rad toward it"),
    # ── look: per-weapon vertical authority (RL feet-aiming) ─────────────────
    DecodeParam("look.weapon_pitch_gain", "look_weapon_pitch_gain",
                _float_vec_or_none, None,
                doc="(9,) per-impulse feet-aim blend weight β; None = off"),
    DecodeParam("look.weapon_pitch_bias", "look_weapon_pitch_bias",
                _float_vec_or_none, None,
                doc="(9,) downward pitch bias (degrees) on the feet-aim target"),
    DecodeParam("look.weapon_pitch_mode", "look_weapon_pitch_mode", str, "lock",
                doc="post-expmap pitch mode: lock | shift | off"),
    DecodeParam("look.weapon_pitch_shift_strength", "look_weapon_pitch_shift_strength",
                float, 1.0, doc="strength of the 'shift' pitch mode"),
    # look.weapon_pitch_lock is the BACK-COMPAT alias of weapon_pitch_mode
    # (False -> "off"); the export threads the mode, never the alias.
    DecodeParam("look.weapon_pitch_lock", "look_weapon_pitch_lock", bool, True,
                graph=False, doc="legacy alias of weapon_pitch_mode (False -> off)"),
    # ── look: AIM DEGRADATION (the DOWN half of the skill knob) ──────────────
    # Only TREMOR has an in-graph twin (ExportWrapper look_tremor_*). sluggish /
    # lag / jitter are eval-side research knobs: lag was RETIRED 7/10 (never
    # emitted by decode-fit — see qnn.decode_fit.emit) and the others are
    # study-only, so they are marked graph=False rather than silently missing.
    DecodeParam("look.aim_degrade_tremor_mag", "look_aim_degrade_tremor_mag",
                scalar_or_impulse_vec, 0.0, export_name="look_tremor_mag",
                doc="AR(1)/OU angular tremor RMS (rad); 0 = off"),
    DecodeParam("look.aim_degrade_tremor_tau", "look_aim_degrade_tremor_tau",
                float, 5.0, export_name="look_tremor_tau",
                doc="tremor correlation time (frames)"),
    DecodeParam("look.aim_degrade_sluggish_tau", "look_aim_degrade_sluggish_tau",
                float, 0.0, graph=False, doc="EMA low-pass on the look delta"),
    DecodeParam("look.aim_degrade_lag_frames", "look_aim_degrade_lag_frames",
                scalar_or_impulse_vec, 0.0, graph=False,
                doc="fractional-frame delay of the turn-delta (RETIRED 7/10)"),
    DecodeParam("look.aim_degrade_jitter_mag", "look_aim_degrade_jitter_mag",
                float, 0.0, graph=False, doc="white per-frame angular noise"),
    # ── move: the a25 segment-COMMITMENT decode ──────────────────────────────
    DecodeParam("move.commitment", "move_commitment", bool,
                doc="a25 segment-head (class,duration) commitment decode"),
    DecodeParam("move.commit_dur_tilt", "move_commit_dur_tilt", _axis_pair, (0.0, 0.0),
                doc="per-axis duration censoring-bias tilt"),
    DecodeParam("move.commit_interrupt", "move_commit_interrupt", bool, True,
                doc="Gate B projectile-interrupt opt-out"),
    # Re-commit: allow the segment head to re-commit to the held class at expiry
    # (no forced maximal-run switch), so a run can be sustained across
    # commitments. False (default) = maximal-run law, bit-identical.
    DecodeParam("move.commit_recommit", "move_commit_recommit", bool, False,
                doc="re-commit to the held class at expiry"),
    # Engagement-gated idle stillness bias: per-axis (fb,lr) additive bias on the
    # seg head's `none` class, scaled by (1 - engagement) — damps pointless idle
    # strafing when disengaged, vanishes in combat. base = E when an enemy is
    # merely present; cooldown = ticks E holds at 1 after combat. (0,0) = off.
    DecodeParam("move.idle_none_bias", "move_idle_none_bias", _axis_pair, (0.0, 0.0),
                doc="engagement-gated per-axis idle stillness bias"),
    DecodeParam("move.idle_engagement_base", "move_idle_engagement_base", float, 0.5,
                doc="E when an enemy is merely present"),
    DecodeParam("move.idle_cooldown_ticks", "move_idle_cooldown_ticks", int, 20,
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
    DecodeParam("attack.bias", "attack_bias", float, 0.0,
                doc="class-0 offset inside the a25 attack_with decode"),
    # Legacy joint vector: retained verbatim so existing a26 artifacts keep
    # their historical selection+fire meaning. New fits pin this to zero.
    DecodeParam("attack.bias_vec", "attack_bias_vec", _float_vec_or_none, None,
                doc="legacy JOINT selection+fire vector (new fits pin to zero)"),
    # Reconciled explicit vectors: rate fitting and weapon-selection fitting
    # must not share a control (the a26/a27 branch-drift failure).
    DecodeParam("attack.fire_bias_vec", "attack_fire_bias_vec", _float_vec_or_none,
                None, doc="fire-only vector (split_v1)"),
    DecodeParam("weapon.preference_bias_vec", "weapon_preference_bias_vec",
                _float_vec_or_none, None, doc="selection-only vector (split_v1)"),
    # weapon.switch_margin: weapon-switch hysteresis (anti-jitter / anti-camp) —
    # leave the held weapon only when the ideal weapon's score beats it by this
    # margin. Replaces the retired attack.stick_bias, which biased selection
    # toward the held weapon AND perturbed the fire decision (weapon camping).
    DecodeParam("weapon.switch_margin", "weapon_switch_margin", float, 0.0,
                doc="weapon-switch hysteresis margin"),
    # a25 discharge-quality gate ("crest-firing"): attack.crest_theta_vec =
    # (8,) per-weapon alignment threshold θ_w in hbw units (≤0 = OFF for that
    # weapon); attack.crest_hold_ticks = shared max hold H in ticks (0 = OFF
    # globally). Both OFF = bit-identical; the countdown latch rides the
    # existing attack_state wire slot (no wire bump). See
    # qnn.model.decode_actions.attack_crest_gate_step.
    DecodeParam("attack.crest_theta_vec", "attack_crest_theta_vec",
                _float_vec_or_none, None, doc="(8,) per-weapon crest θ_w (hbw)"),
    DecodeParam("attack.crest_hold_ticks", "attack_crest_hold_ticks", int, 0,
                doc="shared max crest hold H (ticks); 0 = gate OFF"),
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
        path's whole decode operating point). Consumed by
        qnn.eval.run._apply_decode_config_params."""
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
