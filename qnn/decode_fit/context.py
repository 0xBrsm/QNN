"""Fit context and artifact root.

Everything a fit stage needs to know about WHERE it runs (checkpoint, corpus,
pinned look-grid) and where its artifacts go. One rule with no exceptions:
every artifact this package writes lives under ``runs/decode_fit/<model_id>/``
— no global ``runs/head_probe`` namespace. Closed-loop waves resume off their
own content-hashed done dirs + the substrate/env staleness check
(``instruments.build_wave_dir``); there is no delete-the-dir invalidation.
(The content-keyed ``manifest.json`` cache died with its only consumer, the
offline pins stage — a26 live-pins redesign.)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]

# ── weapon vocabulary (single copy for the package) ──────────────────────────
# impulse byte (actions["weapon"]) index per weapon; the (9,) decode vectors
# are impulse-keyed via qnn.vocab.self_weapon_id_to_impulse (0=none, 1=axe, …).
WEAPON_IMPULSE = {"Axe": 1, "SG": 2, "SSG": 3, "NG": 4, "SNG": 5,
                  "GL": 6, "RL": 7, "LG": 8}
IMPULSE_NAME = {v: k for k, v in WEAPON_IMPULSE.items()}
MODELNAME_TO_ABBR = {"shotgun": "SG", "super_shotgun": "SSG", "nailgun": "NG",
                     "super_nailgun": "SNG", "lightning": "LG",
                     "rocket_launcher": "RL", "grenade_launcher": "GL"}
ABBR_TO_MODELNAME = {v: k for k, v in MODELNAME_TO_ABBR.items()}
# The intercept-valid weapons the skill vector spans (RL feet-anchored via the
# shared lead kernel; GL = lob and Axe = melee stay out).
INTERCEPT_WEAPONS = ("SG", "SSG", "NG", "SNG", "LG", "RL")
# SSG/NG have no grid cells of their own. Aim and fire cadence are calibrated
# as same-physics families; the individual impulses survive only where weapon
# selection itself is the measured quantity. SG and SNG are the forced-weapon
# representatives because those are the class weapons used by the instrument.
TRANSFER_ALIAS = {"SSG": "SG", "NG": "SNG"}
CALIBRATION_FAMILIES = {
    "SG": ("SG", "SSG"),
    "SNG": ("NG", "SNG"),
    "RL": ("RL",),
    "LG": ("LG",),
}
CALIBRATION_FAMILY_KEY = {
    "SG": "SG+SSG", "SSG": "SG+SSG",
    "NG": "NG+SNG", "SNG": "NG+SNG",
    "RL": "RL", "LG": "LG",
}
CALIBRATION_GROUPS = {
    "SG+SSG": ("SG", "SSG"),
    "NG+SNG": ("NG", "SNG"),
    "RL": ("RL",),
    "LG": ("LG",),
}
CALIBRATION_SOURCE = {
    member: source
    for source, members in CALIBRATION_FAMILIES.items()
    for member in members
}


def calibration_members(weapon: str) -> tuple[str, ...]:
    """Aim/fire-cadence family for ``weapon``; singleton when ungrouped."""
    if weapon in CALIBRATION_GROUPS:
        return CALIBRATION_GROUPS[weapon]
    source = CALIBRATION_SOURCE.get(weapon, weapon)
    return CALIBRATION_FAMILIES.get(source, (weapon,))
# The 4 model weapons the botpin instrument actually sweeps (arena loadouts).
INSTRUMENT_WEAPONS = ("shotgun", "super_nailgun", "rocket_launcher", "lightning")
# frikbot pin → engagement-range archetype tag (weights come from the human
# per-weapon engagement-range mass, `_aim_range_byweapon.json`).
FRIKBOT_TO_PIN = {"shotgun": "fsg", "super_nailgun": "fng",
                  "rocket_launcher": "frl", "lightning": "flg"}


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel_to_repo(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(_REPO)) if p.is_relative_to(_REPO) else str(p)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return ""


@dataclass
class FitContext:
    run_dir: Path
    model_id: str
    checkpoint: Path
    checkpoint_sha: str
    corpus_dir: Path
    corpus_fingerprint: str
    look_grid_path: Path
    look_grid_sha: str
    git_commit: str
    # Corpus-derived human baselines, cached once per collect (qnn.human).
    intercept_path: Path        # at-discharge (report card / crest capture)
    tracking_path: Path         # window-sampled tracking (THE aim ladder)
    acq_path: Path
    op_attack_path: Path
    range_path: Path

    # ── artifact locations ────────────────────────────────────────────────
    @property
    def out_dir(self) -> Path:
        return _REPO / "runs" / "decode_fit" / self.model_id

    @property
    def waves_dir(self) -> Path:
        return self.out_dir / "waves"

    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint": rel_to_repo(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha,
            "model_id": self.model_id,
            "corpus_dir": str(self.corpus_dir),
            "corpus_fingerprint": self.corpus_fingerprint,
            "look_grid": rel_to_repo(self.look_grid_path),
            "look_grid_sha256": self.look_grid_sha,
            "git_commit": self.git_commit,
        }


def resolve_fit_context(run_dir: Path) -> FitContext:
    """Resolve checkpoint, corpus, fingerprint, and the PINNED look-grid from a
    bench run-dir. The look-grid is the run's own ``config/look_grid.json`` —
    never the code default (the corpus-fit look-grid trap)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")
    machine = read_json(run_dir / "config" / "machine.json")
    corpus_dir = (_REPO / machine["bc_data_dir"]) if not Path(
        machine["bc_data_dir"]).is_absolute() else Path(machine["bc_data_dir"])

    from qnn.utils.artifacts import find_best_model
    ckpt = find_best_model(run_dir / "checkpoints")
    if ckpt is None:
        raise FileNotFoundError(
            f"no best checkpoint in {run_dir / 'checkpoints'} "
            f"(best_<run_id>.pth or legacy bc_best_model.pth)")

    summary = run_dir / "checkpoints" / "bc_summary.json"
    fingerprint = ""
    if summary.exists():
        fingerprint = str(read_json(summary).get("collection_fingerprint", ""))

    look_grid = run_dir / "config" / "look_grid.json"
    if not look_grid.exists():
        raise FileNotFoundError(
            f"pinned look-grid missing: {look_grid} (corpus-fit look-grid trap — "
            "the run MUST pin its own grid)")

    git_commit = _git_sha()
    if (run_dir / "run.json").exists():
        git_commit = str(read_json(run_dir / "run.json").get("git_commit", git_commit))

    from qnn.human import baseline_paths
    bp = baseline_paths(corpus_dir)
    return FitContext(
        run_dir=run_dir, model_id=run_dir.name, checkpoint=ckpt,
        checkpoint_sha=sha256_file(ckpt),
        corpus_dir=corpus_dir, corpus_fingerprint=fingerprint,
        look_grid_path=look_grid, look_grid_sha=sha256_file(look_grid),
        git_commit=git_commit,
        intercept_path=bp["intercept"], tracking_path=bp["tracking"],
        acq_path=bp["acquisition"],
        op_attack_path=bp["op_attack"], range_path=bp["range"],
    )
