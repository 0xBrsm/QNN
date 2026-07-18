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

# Eval regime names → their bundled decode-config JSON. a24 is a RETIRED arch:
# its entries were pruned with its decode modules (a24 template JSONs remain on
# disk as history only — they name modules that no longer exist and will fail
# resolve_decode_config). Regime selection is picking a decode config; unknown
# names error (config_path_for). Shared by tools/export_onnx + qnn.eval.run.
_TEMPLATES = _REPO_ROOT / "src/qnn/model/bench/templates"
REGIME_CONFIGS: dict[str, Path] = {
    "a25rc1": _TEMPLATES / "decode.a25rc1.json",   # a25-owned template (seg-commitment move + 9-way attack-with): names a25 decode/guard modules so a25 decode-fit + export emit a25 refs
    "a25base": _TEMPLATES / "decode.a25base.json",  # a25-native BASE template (decode-fit emit base): a25 module pointers + only LIVE param keys, neutral placeholder values
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
    "qnn.model.bench.a25.decode": (
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


# decode-config param key -> ExportWrapper / policy kwarg. guard.* keys are not
# here: they are consumed by the guard module and stamped for provenance.
# move.stop_onset (bool) and the move_hazard table (a file ref) are handled
# explicitly by the caller, not via this float-kwarg map.
PARAM_TO_KWARG: dict[str, str] = {
    "look.aim_prior_gain": "look_aim_prior_gain",
    "look.aim_ffwd_gain": "look_aim_ffwd",
    # a24-only keys (weapon.sticky_*, move.sticky_tau_*, move.switchback_eps,
    # attack.threshold) were RETIRED with the a24 arch — their decode laws no
    # longer exist; a config carrying them simply has those keys ignored here
    # (and flagged by the export's provenance path if load-bearing).
    "move.commitment": "move_commitment",  # a25 segment-head commitment decode
    "move.commit_dur_tilt": "move_commit_dur_tilt",  # duration censoring-bias tilt
    "move.commit_interrupt": "move_commit_interrupt",  # Gate B interrupt opt-out
    # Sustained per-tick re-decision prob while an incoming projectile is
    # present (the human-shaped reactivity assist; trim target hazard 1.143)
    "move.threat_break_hazard": "move_threat_break_hazard",
    "attack.bias": "attack_bias",  # applied inside the a25 attack_with decode (class-0 offset)
    # a25 9-way attack-with per-weapon operating point (research/attack-head.md
    # §11): attack.bias_vec = (8,) per-weapon attack bias applied POST-argmax;
    # attack.stick_bias = scalar selection hysteresis toward the held weapon.
    "attack.bias_vec": "attack_bias_vec",
    "attack.stick_bias": "attack_stick_bias",
    # a25 discharge-quality gate ("crest-firing"): attack.crest_theta_vec =
    # (8,) per-weapon alignment threshold θ_w in hbw units (≤0 = OFF for that
    # weapon); attack.crest_hold_ticks = shared max hold H in ticks (0 = OFF
    # globally). Both OFF = bit-identical; the countdown latch rides the
    # existing attack_state wire slot (no wire bump). See
    # qnn.model.bench.a25.decode.attack_crest_gate_step.
    "attack.crest_theta_vec": "attack_crest_theta_vec",
    "attack.crest_hold_ticks": "attack_crest_hold_ticks",
}


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

    def kwargs(self) -> dict[str, Any]:
        """The subset of params that maps to ExportWrapper/policy kwargs."""
        out: dict[str, Any] = {}
        for k, kw in PARAM_TO_KWARG.items():
            if k in self.params:
                out[kw] = self.params[k]
        if "weapon_ban" in self.params:
            out["weapon_ban"] = tuple(int(x) for x in self.params["weapon_ban"])
        return out

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
    decode_mod = importlib.import_module(cfg["decode_module"])
    params = dict(cfg.get("params", {}))
    _validate_required_params(p, params, cfg["decode_module"])
    guard_name = cfg["guard_module"]
    guard_mod = None
    if guard_name and guard_name != "none":
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
