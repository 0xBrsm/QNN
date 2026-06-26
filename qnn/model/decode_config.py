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
  guard_module    dotted import path | "none" — exposes guard_fire_logit_for_export +
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
_GUARD_REQUIRED = ("guard_fire_logit_for_export", "policy_decode_action_postprocess")

# Legacy --decode-regime / eval regime names → their bundled decode-config JSON.
# The rc-module chain folded into one param-gated guard; regime selection is now
# picking a decode config. Shared by tools/export_onnx + qnn.eval.run.
_TEMPLATES = _REPO_ROOT / "src/qnn/model/bench/templates"
REGIME_CONFIGS: dict[str, Path] = {
    "a24rc1": _TEMPLATES / "decode.json",          # a24rc1m operating point (20Hz)
    "a24rc1n": _TEMPLATES / "decode.a24rc1n.json", # rc1m heads + rc3 hazard table & "any" dodge (20Hz)
    "a24rc1o": _TEMPLATES / "decode.a24rc1o.json", # rc1n + non-combat baseline hazard + engagement-gated tau + gain 0.02 (20Hz)
    "a24rc1p": _TEMPLATES / "decode.a24rc1p.json", # rc1o + log-normal non-combat hazard (tau 0.75/0.91) + skilled lock-on look schedule (gain 0.015, gentle_all) (20Hz)
    "a24rc2": _TEMPLATES / "decode.a24rc2.json",   # a24rc2a operating point (10Hz)
    "a24rc3": _TEMPLATES / "decode.a24rc3.json",   # newest 20Hz (rc2-lineage @ 20Hz)
}


def config_path_for(name_or_path: str | Path) -> Path:
    """Map a regime name (a24rc1/…) to its bundled config, or pass through a path."""
    if str(name_or_path) in REGIME_CONFIGS:
        return REGIME_CONFIGS[str(name_or_path)]
    p = Path(name_or_path)
    if p.exists():
        return p
    raise ValueError(f"unknown decode regime / config path: {name_or_path!r}")

# decode-config param key -> ExportWrapper / policy kwarg. guard.* keys are not
# here: they are consumed by the guard module and stamped for provenance.
# move.stop_onset (bool) and the move_hazard table (a file ref) are handled
# explicitly by the caller, not via this float-kwarg map.
PARAM_TO_KWARG: dict[str, str] = {
    "look.aim_prior_gain": "look_aim_prior_gain",
    "look.aim_ffwd_gain": "look_aim_ffwd",
    "weapon.sticky_confidence": "weapon_switch_confidence",
    "weapon.sticky_margin": "weapon_switch_margin",
    "move.sticky_tau_fb": "move_sticky_tau_fb",
    "move.sticky_tau_lr": "move_sticky_tau_lr",
    "move.switchback_eps": "move_switchback_eps",
    "attack.threshold": "attack_threshold",
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
