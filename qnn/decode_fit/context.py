"""Fit context, artifact root, and the content-keyed manifest.

Everything a fit stage needs to know about WHERE it runs (checkpoint, corpus,
pinned look-grid) and where its artifacts go. One rule with no exceptions:
every artifact this package writes lives under ``runs/decode_fit/<model_id>/``
and is indexed in ``manifest.json`` under a content key — (checkpoint sha,
corpus fingerprint, look-grid sha, instrument contract version). A stage asks
the manifest for its artifact and re-runs on a key miss; there is no
delete-the-dir invalidation and no global ``runs/head_probe`` namespace.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]

# Bump when the INSTRUMENT contract changes shape (event schema, scenario
# construction, ruler semantics) — invalidates every cached wave/fit at once.
INSTRUMENT_CONTRACT = 1

# ── weapon vocabulary (single copy for the package) ──────────────────────────
# impulse byte (actions["weapon"]) index per weapon; the (9,) decode vectors
# are impulse-keyed via qnn.vocab.self_weapon_id_to_impulse (0=none, 1=axe, …).
WEAPON_IMPULSE = {"Axe": 1, "SG": 2, "SSG": 3, "NG": 4, "SNG": 5,
                  "GL": 6, "RL": 7, "LG": 8}
MODELNAME_TO_ABBR = {"shotgun": "SG", "super_shotgun": "SSG", "nailgun": "NG",
                     "super_nailgun": "SNG", "lightning": "LG",
                     "rocket_launcher": "RL", "grenade_launcher": "GL"}
ABBR_TO_MODELNAME = {v: k for k, v in MODELNAME_TO_ABBR.items()}
# The intercept-valid weapons the skill vector spans (RL feet-anchored via the
# shared lead kernel; GL = lob and Axe = melee stay out).
INTERCEPT_WEAPONS = ("SG", "SSG", "NG", "SNG", "LG", "RL")
# SSG/SNG have no grid cells of their own — kinematics-identical to SG/NG, they
# TRANSFER off the shotgun/nailgun response while keeping their OWN human ladder.
TRANSFER_ALIAS = {"SSG": "SG", "SNG": "NG"}
# The 4 model weapons the botpin instrument actually sweeps (arena loadouts).
INSTRUMENT_WEAPONS = ("shotgun", "nailgun", "rocket_launcher", "lightning")
# frikbot pin → engagement-range archetype tag (weights come from the human
# per-weapon engagement-range mass, `_aim_range_byweapon.json`).
FRIKBOT_TO_PIN = {"shotgun": "fsg", "nailgun": "fng",
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
    intercept_path: Path
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

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "manifest.json"

    def content_key(self, **extra: Any) -> dict[str, Any]:
        """The cache key every manifest entry carries: same key → same fit
        inputs → the cached artifact is valid. ``extra`` pins stage-specific
        inputs (e.g. the design-round point set)."""
        return {
            "checkpoint_sha": self.checkpoint_sha,
            "corpus_fingerprint": self.corpus_fingerprint,
            "look_grid_sha": self.look_grid_sha,
            "instrument_contract": INSTRUMENT_CONTRACT,
            **extra,
        }

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

    # ── manifest (content-keyed artifact index) ───────────────────────────
    def _manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return read_json(self.manifest_path)
        return {"model_id": self.model_id, "entries": {}}

    def manifest_get(self, stage: str, key: dict[str, Any]) -> Path | None:
        """The cached artifact for ``stage`` iff its stored content key matches
        ``key`` and the file still exists — else None (stage must re-run)."""
        e = self._manifest()["entries"].get(stage)
        if not e or e.get("key") != key:
            return None
        p = _REPO / e["artifact"]
        return p if p.exists() else None

    def manifest_put(self, stage: str, artifact: Path, key: dict[str, Any]) -> None:
        m = self._manifest()
        m["entries"][stage] = {
            "artifact": rel_to_repo(artifact),
            "sha256": sha256_file(artifact),
            "key": key,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(m, indent=2) + "\n")


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
        intercept_path=bp["intercept"], acq_path=bp["acquisition"],
        op_attack_path=bp["op_attack"], range_path=bp["range"],
    )
