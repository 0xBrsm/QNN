"""Turnkey per-model MECHANICAL-AIM decode-fit pipeline.

ONE repeatable step, run after training against ANY checkpoint, that sets the
decode layer so the bot (a) matches human behavior at a baseline skill and
(b) exposes a single signed skill scalar ``s`` (a human percentile) it can slide
along the AIM axis — emitted as a run-pinned decode config + a per-model skill
curve + provenance, ready to bake + re-export.

    python -m qnn.eval.decode_fit_pipeline --run-dir <bench run> \
        [--skill p76] [--write] [--validate] [--refit-aim] [--refit-pins]

Scope (settled — see research/skill-curves.md + research/look-aim-decode.md, and
the project memory `project_decode_fit_pipeline`):

  * The skill scalar is the MECHANICAL aim knob ONLY. Everything tactical /
    strategic (weapon-selection, positioning, survival, the `standing` obs
    scalar) is DEFERRED (needs training) and is NOT steered here — the pipeline
    only emits *measurement readouts* for those axes.
  * The aim knob is one signed scalar in three segments, on the DAMPENED
    substrate (``look.turn_mag_scale`` pinned, dampener-BEFORE-aim fit order):
      - s < floor          → ``look.aim_degrade_lag_frames`` (universal, ~12→1)
      - floor ≤ s ≤ p90    → ``look.aim_prior_gain`` (rotation; per-weapon table,
                             baked as a shared scalar in v1)
      - s > p90            → gain pinned 0.20, ``look.aim_mag_gain`` (α) rides the
                             super-human frontier
  * The skill index is the leave-one-out WITHIN-weapon coh percentile (the
    construct-validity reframe relabels these BEHAVIORAL difficulty percentiles,
    not "skill"; the knob is a valid difficulty gradient regardless).
  * Non-lever axes are pinned to all-humans (calibrated to human, NOT slid on s):
    MOVE (sticky tau + corpus log-normal hazard), LOOK-style (per-weapon
    turn-mag — head-native, verify), ATTACK (sampled decode + ``attack.bias``
    calibration; cone low/off), WEAPON-select (gate diag only, deferred),
    and the ``look.turn_mag_scale`` over-turn dampener.
  * "The process is the deliverable": every checkpoint re-anchors its own
    floor/ceiling, so ``s → (lag, gain_w, α)`` is RE-FIT per checkpoint and the
    emitted artifacts carry mandatory provenance (checkpoint, corpus
    fingerprint, look-grid sha). Guard the corpus-fit look-grid trap
    (`project_corpusfit_lookgrid_export_trap`): the run's pinned look_grid is
    authoritative, never the code default.

This module ORCHESTRATES; it does not duplicate kernels. The closed-loop aim
grid + offline non-lever fits are produced by the existing scripts
(scripts/analysis/*, qnn.bc.decode_fit); the pipeline consumes their cached
artifacts on a fingerprint hit and re-runs them on ``--refit-*``. ``decode.a24rc1q``
(the hand-built "FINAL skill-system decode") is the regression oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── locations ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[3]
HEAD_PROBE = _REPO / "runs" / "head_probe"
TEMPLATES = _REPO / "src" / "qnn" / "model" / "bench" / "templates"
ORACLE_TEMPLATE = TEMPLATES / "decode.a24rc1q.json"
SIDECAR_DIR = _REPO / "src" / "docs" / "skill-curves"

# The 5 direct-fire weapons the lead kernel can score (RL/GL/Axe excluded —
# splash/arc/melee; see _aim_coh_byweapon.json).
DIRECT_FIRE = ("SG", "SSG", "NG", "SNG", "LG")
DEFAULT_SKILL = "p76"            # deployed stance; competitive deploys override up
GAIN_CEILING = 0.20              # gain pinned at p90; above p90 → α frontier

# VERTICAL aim authority (RL-splash feet-aiming, look-aim-decode.md §12). The
# per-weapon pitch correction is error-driven + SELF-LIMITING toward the
# AIM_Z_DROP feet anchor, so its gain is a CONTROL gain (convergence rate), not a
# magnitude — the self-limiting term auto-absorbs per-model head differences (a
# head that already lowers leaves a smaller z_err for the gain to act on), so the
# gain is ~universal and the per-weapon aspect is just WHICH weapons carry a feet
# anchor. The default enables every anchored weapon at this rate; the genuine
# trade (convergence speed vs transient turn-EMD) is settled by the closed-loop
# gate (Stage 6). RL-validated; NG/SNG enabled by anchor-consistency, gate-pending.
DEFAULT_PITCH_GAIN = 0.5

# Decode knobs the eval path (qnn.eval.run) honors but the deploy (ONNX) export
# path does NOT yet thread into ExportWrapper. A non-noop value for any of these
# means the emitted config is eval-faithful but NOT deploy-faithful — the emitter
# flags it so we never silently ship an un-deployable config. ``turn_mag_scale``,
# ``attack.bias`` and ``aim_mag_gain`` (α) are now threaded (Plan A); only the
# STATEFUL degrade lag remains un-exported (needs in-graph ring-buffer state) and
# is only reached below the native floor — that stays a v2 item.
# guard.* params are consumed by the guard adapter (resolve_decode_config →
# make_guard), NOT ExportWrapper kwargs, so they are NOT export gaps.
NON_EXPORTED_KNOBS: dict[str, float] = {
    "look.aim_degrade_lag_frames": 0.0,
}

# Cached human-baseline + aim-curve artifacts (under runs/head_probe/).
ART_COH_BYWEAPON = HEAD_PROBE / "_aim_coh_byweapon.json"
ART_PSEGMENT_DAMP = HEAD_PROBE / "_aim_psegment_dampened.json"
ART_FRONTIER_DAMP = HEAD_PROBE / "_aim_frontier_dampened.json"
ART_DEGRADE_SWEEP = HEAD_PROBE / "_aim_degrade_sweep.json"
ART_SKILL_CURVE_FIT = HEAD_PROBE / "_aim_skill_curve_fit.json"
ART_LOOK_STYLE = HEAD_PROBE / "_look_style_by_weapon.json"
ART_PLAYER_PROFILE = HEAD_PROBE / "_player_profile.json"
ART_RL_VERT = HEAD_PROBE / "_rl_aim_vertical_residual.json"  # §12 vertical gap


def _log(msg: str) -> None:
    print(f"[decode-fit] {msg}", flush=True)


def _read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _parse_skill(s: str) -> float:
    """``p76`` / ``76`` / ``0.76`` → percentile float in [0, 100]."""
    t = str(s).strip().lower().lstrip("p")
    v = float(t)
    if 0.0 <= v <= 1.0:
        v *= 100.0
    if not (0.0 <= v <= 100.0):
        raise ValueError(f"skill percentile out of range: {s!r}")
    return v


# ══ Stage 0 — run context + human baseline ═══════════════════════════════════

@dataclass
class RunContext:
    run_dir: Path
    model_id: str
    checkpoint: Path
    corpus_dir: Path
    corpus_fingerprint: str
    look_grid_path: Path
    look_grid_sha: str
    git_commit: str

    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint.relative_to(_REPO))
            if self.checkpoint.is_relative_to(_REPO) else str(self.checkpoint),
            "model_id": self.model_id,
            "corpus_dir": str(self.corpus_dir),
            "corpus_fingerprint": self.corpus_fingerprint,
            "look_grid": str(self.look_grid_path.relative_to(_REPO))
            if self.look_grid_path.is_relative_to(_REPO) else str(self.look_grid_path),
            "look_grid_sha256": self.look_grid_sha,
            "git_commit": self.git_commit,
        }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return ""


def resolve_run_context(run_dir: Path) -> RunContext:
    """Resolve the checkpoint, corpus, fingerprint, and PINNED look-grid from a
    bench run-dir. The look-grid is the run's own ``config/look_grid.json`` (never
    the code default — the corpus-fit look-grid trap)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")
    machine = _read_json(run_dir / "config" / "machine.json")
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
        fingerprint = str(_read_json(summary).get("collection_fingerprint", ""))

    look_grid = run_dir / "config" / "look_grid.json"
    if not look_grid.exists():
        raise FileNotFoundError(
            f"pinned look-grid missing: {look_grid} (corpus-fit look-grid trap — "
            "the run MUST pin its own grid)")

    git_commit = _git_sha()
    if (run_dir / "run.json").exists():
        git_commit = str(_read_json(run_dir / "run.json").get("git_commit", git_commit))

    return RunContext(
        run_dir=run_dir, model_id=run_dir.name, checkpoint=ckpt,
        corpus_dir=corpus_dir, corpus_fingerprint=fingerprint,
        look_grid_path=look_grid, look_grid_sha=_sha256(look_grid),
        git_commit=git_commit,
    )


def ensure_human_baseline(ctx: RunContext, *, refit: bool) -> dict[str, Any]:
    """Stage 0. The HUMAN reference is corpus-derived and model-AGNOSTIC: per-weapon
    coh percentile distributions (`_aim_coh_byweapon.json`) + the per-player
    profile, ranked by the leave-one-out within-weapon coh percentile (the settled
    behavioral difficulty index). Rebuild on a corpus miss (offline, CPU); reuse
    otherwise."""
    if refit or not ART_COH_BYWEAPON.exists():
        _run_script("scripts/analysis/aim_coh_byweapon.py",
                    ["--collect-dir", str(ctx.corpus_dir), "--out", str(ART_COH_BYWEAPON)],
                    why="per-weapon human coh distributions")
    if refit or not ART_PLAYER_PROFILE.exists():
        _run_script("scripts/analysis/player_profile.py",
                    ["--collect-dir", str(ctx.corpus_dir), "--out", str(ART_PLAYER_PROFILE)],
                    why="per-player profile (within-weapon percentile rank)")

    byweapon = _read_json(ART_COH_BYWEAPON)
    weapons = {w: byweapon["weapons"][w] for w in DIRECT_FIRE if w in byweapon["weapons"]}
    missing = [w for w in DIRECT_FIRE if w not in weapons]
    if missing:
        raise ValueError(f"human per-weapon coh missing direct-fire weapons: {missing}")
    _log(f"Stage 0 human baseline: {len(weapons)} direct-fire weapons, "
         f"{byweapon.get('n_demos_total')} demos, op_filter={byweapon.get('op_filter')}")
    return {
        "per_weapon_coh_pct": {w: weapons[w]["percentiles"] for w in weapons},
        "n_demos": byweapon.get("n_demos_total"),
        "skill_index": "leave-one-out within-weapon coh percentile (behavioral)",
        "artifact": str(ART_COH_BYWEAPON.relative_to(_REPO)),
    }


# ══ Stage 1 — AIM skill curve (closed-loop, dampened substrate) ═══════════════

@dataclass
class AimCurve:
    """The signed ``s → (lag, gain, α)`` aim curve, built on the dampened substrate
    (``look.turn_mag_scale`` pinned). One scalar in, three segments out."""
    turn_mag_scale: float
    floor_pct: float                       # native floor (gain 0, lag 0)
    gain_anchors: list[tuple[float, float]]   # (pct, gain), floor → p90
    alpha_anchors: list[tuple[float, float]]  # (pct, α), p90 → super-human
    lag_anchors: list[tuple[float, float]]    # (pct, lag_frames), below floor
    per_weapon_gain: dict[str, list[dict]]    # diagnostic per-weapon gain table
    provenance_commits: dict[str, str] = field(default_factory=dict)

    def resolve(self, s: float) -> dict[str, Any]:
        """Resolve the operating point at skill percentile ``s``."""
        if s < self.floor_pct:
            lag = float(_interp_anchors(s, self.lag_anchors))
            seg = "down/lag"
            params = {"look.aim_degrade_lag_frames": round(lag, 3),
                      "look.aim_prior_gain": 0.0, "look.aim_mag_gain": 0.0}
        elif s <= 90.0:
            gain = float(_interp_anchors(s, self.gain_anchors))
            seg = "up/gain"
            params = {"look.aim_prior_gain": round(gain, 4),
                      "look.aim_mag_gain": 0.0, "look.aim_degrade_lag_frames": 0.0}
        else:
            alpha = float(_interp_anchors(s, self.alpha_anchors))
            seg = "super/alpha"
            params = {"look.aim_prior_gain": GAIN_CEILING,
                      "look.aim_mag_gain": round(alpha, 4),
                      "look.aim_degrade_lag_frames": 0.0}
        return {"segment": seg, "params": params}


def _interp_anchors(x: float, anchors: list[tuple[float, float]]) -> float:
    """Monotone linear interp over (x, y) anchors, clamped at the ends."""
    if not anchors:
        return 0.0
    xs = np.array([a for a, _ in anchors], float)
    ys = np.array([b for _, b in anchors], float)
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], ys[order]))


def load_aim_curve(ctx: RunContext, *, refit: bool) -> AimCurve:
    """Stage 1. Assemble the signed aim curve from the closed-loop DAMPENED
    artifacts. On ``--refit-aim`` re-run the closed-loop grid scripts (GPU); else
    consume the cached artifacts (and warn on a checkpoint/commit mismatch — the
    process is per-checkpoint)."""
    if refit:
        _refit_aim_closed_loop(ctx)

    for art in (ART_PSEGMENT_DAMP, ART_FRONTIER_DAMP, ART_DEGRADE_SWEEP):
        if not art.exists():
            raise FileNotFoundError(
                f"aim-curve artifact missing: {art.relative_to(_REPO)} — re-run with "
                "--refit-aim (closed-loop, GPU) or produce it via the analysis scripts")

    pseg = _read_json(ART_PSEGMENT_DAMP)
    frontier = _read_json(ART_FRONTIER_DAMP)
    degrade = _read_json(ART_DEGRADE_SWEEP)

    commits = {"psegment": pseg.get("git_commit", ""),
               "frontier": frontier.get("git_commit", "")}
    for name, ck in (("psegment", pseg.get("checkpoint")),
                     ("frontier", frontier.get("checkpoint"))):
        if ck and Path(ck).name != ctx.model_id:
            _log(f"WARNING: {name} artifact was fit on checkpoint {ck!r}, not "
                 f"{ctx.model_id!r} — re-fit per checkpoint (process is the deliverable)")

    tms = float(pseg.get("turn_mag_scale", 1.0))

    # UP/placement: gain → best-match percentile anchors. Build the monotone
    # percentile → gain map (smallest gain that first reaches each percentile).
    pct_to_gain: dict[float, float] = {}
    for a in pseg["anchors"]:
        if not a.get("ok"):
            continue
        pct = _parse_skill(a["best_match_percentile"])
        g = float(a["gain"])
        if pct not in pct_to_gain or g < pct_to_gain[pct]:
            pct_to_gain[pct] = g
    gain_anchors = sorted(pct_to_gain.items())
    floor_pct = gain_anchors[0][0] if gain_anchors else 50.0

    # SUPER-human: α frontier above p90 (p90 → α 0; the human-style edge ≈ α 0.04
    # near ~p95; > that = explicit super-human slew). Anchored to the dampened
    # frontier cells (gain pinned 0.20).
    alpha_anchors = _alpha_anchors_from_frontier(frontier)

    # DOWN/lag: invert the degrade-sweep lag → mean within-weapon percentile.
    lag_anchors = _lag_anchors_from_degrade(degrade, floor_pct)

    # Diagnostic per-weapon gain table (sidecar; v1 bakes a shared scalar).
    per_weapon_gain = {}
    if ART_SKILL_CURVE_FIT.exists():
        fit = _read_json(ART_SKILL_CURVE_FIT)
        per_weapon_gain = fit.get("aggregate_gain_fit", {})

    _log(f"Stage 1 aim curve: turn_mag_scale={tms} floor=p{floor_pct:.0f} "
         f"gain_anchors={gain_anchors} alpha_anchors={alpha_anchors}")
    return AimCurve(
        turn_mag_scale=tms, floor_pct=floor_pct, gain_anchors=gain_anchors,
        alpha_anchors=alpha_anchors, lag_anchors=lag_anchors,
        per_weapon_gain=per_weapon_gain, provenance_commits=commits,
    )


def _alpha_anchors_from_frontier(frontier: dict) -> list[tuple[float, float]]:
    """Map the dampened α-frontier to percentile anchors above p90. p90 → α 0
    (the gain-ceiling base); the efficient-frontier α values extend the curve into
    the super-human band. The human-style edge sits near α≈0.04 (~p95); cells above
    are explicit super-human slew (look-aim-decode §10/§11)."""
    cells = sorted(
        ({"alpha": float(c["alpha"]), "tot": float(c.get("tot", 0.0))}
         for c in frontier.get("frontier", []) if c.get("ok") and float(c["gain"]) >= GAIN_CEILING - 1e-9),
        key=lambda c: c["alpha"])
    anchors = [(90.0, 0.0)]
    # Spread the available α cells across the p90→p99+ band by ascending α.
    edge_pcts = [95.0, 99.0, 99.5, 99.9]
    for i, c in enumerate(cells):
        if c["alpha"] <= 0.0:
            continue
        pct = edge_pcts[i] if i < len(edge_pcts) else 99.9
        anchors.append((pct, c["alpha"]))
    return sorted(set(anchors))


def _lag_anchors_from_degrade(degrade: dict, floor_pct: float) -> list[tuple[float, float]]:
    """Invert the universal lag sweep to (percentile → lag_frames) below the floor.
    Each lag value's per-direct-fire-weapon coh is mapped to its within-weapon
    percentile (the settled index) and averaged → an aggregate percentile per lag;
    the curve is then inverted. Floor maps to lag 0."""
    hpct = degrade.get("human_perweapon_coh_percentiles", {})
    lag_levers = degrade.get("levers", {}).get("lag", [])
    anchors = [(floor_pct, 0.0)]
    for lev in lag_levers:
        lag = float(lev["value"])
        pcts = []
        for w in DIRECT_FIRE:
            pw = lev.get("per_weapon", {}).get(w)
            ref = hpct.get(w)
            if not pw or not ref:
                continue
            pcts.append(_coh_to_percentile(float(pw["coh_5deg"]), ref))
        if pcts:
            anchors.append((float(np.mean(pcts)), lag))
    # Keep a monotone (decreasing percentile → increasing lag) envelope.
    anchors = sorted(set(anchors))
    return anchors


def _coh_to_percentile(coh: float, pct_table: dict[str, float]) -> float:
    """Map a coh value to a percentile by interpolating a {pXX: coh} table."""
    items = sorted((_parse_skill(k), float(v)) for k, v in pct_table.items())
    ps = np.array([p for p, _ in items], float)
    cs = np.array([c for _, c in items], float)
    return float(np.interp(coh, cs, ps))


# ══ Stage 2 — non-lever all-humans pins (offline) ════════════════════════════

def _weapon_pitch_gain_default(k: float = DEFAULT_PITCH_GAIN) -> list[float]:
    """The (9,) per-IMPULSE vertical-authority default, derived from AIM_Z_DROP:
    enable the self-limiting pitch correction (at control gain ``k``) for exactly
    the weapons that carry a non-zero feet anchor (RL/NG/SNG); 0 elsewhere. Tying
    the enable to the anchor table keeps a single source of truth for "which
    weapons aim low" — add an anchor and the correction follows automatically."""
    from qnn.model.bench.a24.lead_aim import AIM_Z_DROP
    return [k if (a != 0.0 or b != 0.0) else 0.0 for (a, b) in AIM_Z_DROP]


def load_nonlever_pins(ctx: RunContext, *, refit: bool) -> dict[str, Any]:
    """Stage 2. The all-humans pins: move sticky-tau + corpus log-normal hazard,
    weapon sticky gate, attack.bias calibration, the look.turn_mag_scale over-turn
    dampener, and weapon_ban. By default these are the VALIDATED pins from the
    oracle template (decode.a24rc1q — itself the hand-fit instance for this
    checkpoint); ``--refit-pins`` re-derives move/weapon/attack via
    qnn.bc.decode_fit.fit + re-pins the corpus hazard."""
    oracle = _read_json(ORACLE_TEMPLATE)
    p = dict(oracle["params"])
    pins = {
        "move.sticky_tau_fb": p.get("move.sticky_tau_fb"),
        "move.sticky_tau_lr": p.get("move.sticky_tau_lr"),
        "move.tau_engagement_gated": p.get("move.tau_engagement_gated"),
        "move.stop_onset": p.get("move.stop_onset"),
        "move.switchback_eps": p.get("move.switchback_eps"),
        "weapon.sticky_confidence": p.get("weapon.sticky_confidence"),
        "weapon.sticky_margin": p.get("weapon.sticky_margin"),
        "weapon_ban": p.get("weapon_ban", []),
        "attack.bias": p.get("attack.bias"),
        "look.turn_mag_scale": p.get("look.turn_mag_scale"),
        "look.aim_ffwd_gain": p.get("look.aim_ffwd_gain", 0.0),
        # VERTICAL aim authority (RL-splash feet-aiming, §12): per-weapon control
        # gain for the self-limiting pitch correction. Oracle template value if it
        # pins one, else the anchor-derived default (enable anchored weapons).
        "look.weapon_pitch_gain": p.get("look.weapon_pitch_gain",
                                        _weapon_pitch_gain_default()),
    }
    move_hazard = oracle.get("move_hazard")
    source = f"oracle template {ORACLE_TEMPLATE.name}"

    if refit:
        fitres = _refit_pins_offline(ctx)
        if fitres is not None:
            pins.update(fitres["fit"])
            move_hazard = {"method": "lognorm", "lognorm": fitres["hazard"]["lognorm"],
                           "tick_hz": fitres["hazard"].get("tick_hz")}
            source = "qnn.bc.decode_fit.fit (re-fit) + corpus hazard"

    _log(f"Stage 2 non-lever pins ({source}): move_tau="
         f"({pins['move.sticky_tau_fb']},{pins['move.sticky_tau_lr']}) "
         f"weapon=({pins['weapon.sticky_confidence']},{pins['weapon.sticky_margin']}) "
         f"attack.bias={pins['attack.bias']} turn_mag_scale={pins['look.turn_mag_scale']} "
         f"weapon_ban={pins['weapon_ban']}")
    return {"pins": pins, "move_hazard": move_hazard, "guards": {
        k: v for k, v in p.items() if k.startswith("guard.")}, "source": source}


# ══ Stage 3 — model-side aim verification ════════════════════════════════════

def verify_model_aim(ctx: RunContext, human: dict) -> dict[str, Any]:
    """Stage 3. Confirm the model reproduces the per-weapon human coh distributions
    and per-weapon look-style turn-mag — gates trusting the curve. Read-only over
    the cached model forwards (`_aim_degrade_sweep.json` base per-weapon coh +
    `_look_style_by_weapon.json` verdict)."""
    out: dict[str, Any] = {"per_weapon_coh": {}, "look_style": None}
    if ART_DEGRADE_SWEEP.exists():
        base = _read_json(ART_DEGRADE_SWEEP).get("base", {}).get("per_weapon", {})
        for w in DIRECT_FIRE:
            model_coh = base.get(w, {}).get("coh_5deg")
            hp = human["per_weapon_coh_pct"].get(w, {})
            if model_coh is None or not hp:
                continue
            pct = _coh_to_percentile(float(model_coh), hp)
            out["per_weapon_coh"][w] = {
                "model_coh_5deg": float(model_coh),
                "human_floor_percentile": round(pct, 1),
                "human_p50": hp.get("p50"), "human_p90": hp.get("p90"),
            }
    if ART_LOOK_STYLE.exists():
        ls = _read_json(ART_LOOK_STYLE)
        out["look_style"] = {
            "verdict": ls.get("verdict"),
            "max_cross_weapon_emd": ls.get("max_cross_weapon_emd")
            or ls.get("cross_weapon_spread"),
            "note": "look-style is weapon-VARIANT but head-native (head reads "
                    "self_weapon_id) — flag only if the model fails to reproduce it",
        }
    floors = [v["human_floor_percentile"] for v in out["per_weapon_coh"].values()]
    out["native_floor_percentile_mean"] = round(float(np.mean(floors)), 1) if floors else None
    # VERTICAL feet gap (§12): how much of the AIM_Z_DROP anchor the DEPLOYED decode
    # delivers vs the splash target — the gap the weapon_pitch_gain correction closes.
    # Read-only over the cached residual artifact (rl_aim_vertical_residual.py),
    # checkpoint-matched so a stale measurement doesn't masquerade as this model's.
    out["vertical_feet_gap"] = _read_vertical_feet_gap(ctx)
    _log(f"Stage 3 model verification: per-weapon native floors "
         f"{ {w: v['human_floor_percentile'] for w, v in out['per_weapon_coh'].items()} }"
         + (f"; RL vertical gap {out['vertical_feet_gap']['RL']['feet_below_origin']}°"
            f" delivered {out['vertical_feet_gap']['RL']['delivered_deployed']}°"
            if out["vertical_feet_gap"].get("RL") else "; vertical gap: not measured"))
    return out


def _read_vertical_feet_gap(ctx: RunContext) -> dict[str, Any]:
    """The per-weapon vertical splash gap from the cached residual artifact: the
    feet anchor (feet_below_origin) vs what the DEPLOYED rotation decode delivers
    (~0 — the rocket-center-mass failure). Empty + a note when absent or stale."""
    if not ART_RL_VERT.exists():
        return {"note": "run scripts/analysis/rl_aim_vertical_residual.py to measure"}
    d = _read_json(ART_RL_VERT)
    if ctx.model_id not in str(d.get("run", "")):
        return {"note": f"stale: residual artifact is for {d.get('run')!r}, not {ctx.model_id} "
                        "— re-run rl_aim_vertical_residual.py on this checkpoint"}
    out: dict[str, Any] = {}
    for w, b in d.get("weapons", {}).items():
        if not b.get("n"):
            continue
        out[w] = {
            "feet_below_origin": b["feet_below_origin"]["median"],
            "delivered_deployed": b["delivered_dep_vert"]["median"],
            "n": b["n"],
        }
    return out


# ══ Stage 4 — synthesize s → params + provenance ═════════════════════════════

def synthesize_skill_curve(
    ctx: RunContext, human: dict, aim: AimCurve, pins: dict, verify: dict,
) -> dict[str, Any]:
    """Stage 4. One artifact: the piecewise signed map + the all-humans pin
    constants + per-axis verdicts + mandatory provenance."""
    return {
        "schema": "skill_curve_v1",
        "axis": "mechanical_aim",
        "scope": "AIM-only mechanical knob; tactical/strategic axes DEFERRED",
        "skill_index": human["skill_index"],
        "substrate": {"look.turn_mag_scale": aim.turn_mag_scale,
                      "fit_order": "dampener-before-aim"},
        "piecewise": {
            "down": {"range": f"s < p{aim.floor_pct:.0f}", "knob": "look.aim_degrade_lag_frames",
                     "anchors": [{"pct": p, "lag_frames": round(v, 3)} for p, v in aim.lag_anchors]},
            "up": {"range": f"p{aim.floor_pct:.0f} ≤ s ≤ p90", "knob": "look.aim_prior_gain",
                   "anchors": [{"pct": p, "gain": round(v, 4)} for p, v in aim.gain_anchors]},
            "super": {"range": "s > p90", "knob": "look.aim_mag_gain (α), gain pinned 0.20",
                      "anchors": [{"pct": p, "alpha": round(v, 4)} for p, v in aim.alpha_anchors]},
        },
        "per_weapon_gain_table": aim.per_weapon_gain,
        "all_humans_pins": pins["pins"],
        "move_hazard": "pinned from corpus (lognorm); see emitted decode config",
        "guards": pins["guards"],
        "non_lever_verdicts": {
            "MOVE": "FLAT weapon-invariant — sticky_tau + corpus lognorm hazard",
            "LOOK_style": verify.get("look_style"),
            "WEAPON_select": "human target RETRACTED (coh artifact); gate diag only, DEFERRED",
            "ATTACK": "all-humans pin — attack.bias calibration, cone low/off, sampled",
        },
        "model_verification": verify,
        "provenance": ctx.provenance(),
        "aim_artifact_commits": aim.provenance_commits,
    }


# ══ Stage 5 — emit + re-export ═══════════════════════════════════════════════

def detect_export_gaps(params: dict[str, Any]) -> list[str]:
    """Return the active decode params the ONNX export path does NOT thread. A
    non-empty list means the emitted config is eval-faithful but NOT yet
    deploy-faithful at this operating point."""
    gaps = []
    for k, noop in NON_EXPORTED_KNOBS.items():
        if k in params and float(params[k]) != noop:
            gaps.append(k)
    return gaps


def emit_decode_config(
    ctx: RunContext, curve: dict, aim: AimCurve, pins: dict, skill: float,
    version: str, out_path: Path,
) -> dict[str, Any]:
    """Stage 5. Bake the chosen ``--skill s`` operating point into a run-pinned
    decode config: start from the oracle template, override the fitted params, bump
    ``version``, and validate via ``resolve_decode_config``."""
    op = aim.resolve(skill)
    cfg = _read_json(ORACLE_TEMPLATE)
    cfg["version"] = version
    cfg["description"] = (
        f"decode-fit pipeline @ s=p{skill:.0f} on {ctx.model_id}: "
        f"aim {op['segment']} {op['params']}; all-humans pins + "
        f"turn_mag_scale={aim.turn_mag_scale} dampener; "
        f"corpus_fp={ctx.corpus_fingerprint[:16]} look_grid_sha={ctx.look_grid_sha[:12]}")
    cfg["look_grid"] = "config/look_grid.json"
    if pins.get("move_hazard"):
        cfg["move_hazard"] = pins["move_hazard"]

    p = cfg["params"]
    for k, v in pins["pins"].items():
        p[k] = v
    for k, v in op["params"].items():
        p[k] = v

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2) + "\n")

    gaps = detect_export_gaps(p)
    resolved_ok, resolved_err = _validate_decode_config(out_path, ctx)
    _log(f"Stage 5 emitted {out_path.relative_to(_REPO)} (version={version}); "
         f"resolve_decode_config {'OK' if resolved_ok else 'FAILED: ' + str(resolved_err)}")
    if gaps:
        _log(f"WARNING export gap — these active params are NOT threaded into ONNX "
             f"export (eval-faithful, NOT deploy-faithful): {gaps}")
    return {"config_path": str(out_path), "operating_point": op,
            "resolve_ok": resolved_ok, "resolve_error": resolved_err,
            "export_gaps": gaps, "config_sha256": _sha256(out_path)}


def write_skill_curve_sidecar(ctx: RunContext, curve: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(curve, indent=2) + "\n")
    _log(f"Stage 5 wrote skill-curve sidecar → {out_path.relative_to(_REPO)}")
    return out_path


def _validate_decode_config(path: Path, ctx: RunContext) -> tuple[bool, str]:
    try:
        from qnn.model import decode_config as _dc
        _dc.resolve_decode_config(path)
        return True, ""
    except Exception as e:  # pragma: no cover - surfaced to the report
        return False, repr(e)


# ══ Stage 6 — closed-loop validation gate ════════════════════════════════════

# Provisional closed-loop gate thresholds (documented, overridable). The turn-EMD
# is the closed-loop arena turn distribution vs human (rc_humanlikeness, raw deg);
# σ_h=0.113 rad ≈ 6.47°, so "within σ_h" ⇒ turn-EMD ≤ ~6.5°. The op-fire rate must
# land within ±25% of the human operative-attack switch rate. Confirm on the first
# real run for this checkpoint — the process is per-model.
GATE_TURN_EMD_DEG_MAX = 6.47
GATE_OPFIRE_REL_TOL = 0.25
SIGMA_H_RAD = 0.113


def validate_closed_loop(
    ctx: RunContext, config_path: Path, skill: float, curve: dict, *, run: bool,
    eval_npz: Path | None = None,
) -> dict[str, Any]:
    """Stage 6. Gate the emitted baseline against humanlikeness: turn-EMD (σ_h
    units), move dwell-EMD, op-fire rate. Scores an arena-eval action-stream npz
    (produced by qnn.eval.run with ``eval_log_action_streams=True``); when no npz
    is available the gate is SKIPPED with the predicted targets recorded so a later
    eval can be scored. The closed-loop eval itself needs the trainer GPU and is
    launched separately (`feedback_detached_docker_for_overnight`)."""
    targets = {}
    for w, pct in curve.get("model_verification", {}).get("per_weapon_coh", {}).items():
        targets[w] = {"human_p50": pct.get("human_p50"), "human_p90": pct.get("human_p90")}
    report: dict[str, Any] = {
        "skill_percentile": skill,
        "config": str(Path(config_path).relative_to(_REPO))
        if Path(config_path).is_relative_to(_REPO) else str(config_path),
        "predicted_targets": targets,
        "sigma_h_rad": SIGMA_H_RAD,
    }

    npz = _find_eval_npz(ctx, eval_npz)
    if not run or npz is None:
        report["status"] = "SKIPPED"
        report["note"] = (
            "closed-loop gate not scored (no action-stream npz). Run an arena eval "
            f"with eval_decode_regime={Path(config_path).name} + "
            "eval_log_action_streams=True inside the trainer, then re-run with "
            "--validate --eval-npz <move_streams_*.npz>.")
        _log(f"Stage 6 validation: SKIPPED ({'no npz found' if npz is None else 'no --validate'})")
        return report

    return _score_eval_npz(ctx, npz, skill, report)


def _find_eval_npz(ctx: RunContext, eval_npz: Path | None) -> Path | None:
    if eval_npz and Path(eval_npz).exists():
        return Path(eval_npz)
    return None


def _score_eval_npz(ctx: RunContext, npz: Path, skill: float, report: dict) -> dict:
    """Score an arena-eval action-stream npz against the human corpus via
    rc_humanlikeness. Provisional PASS/FAIL on turn-EMD + op-fire rate."""
    try:
        sys.path.insert(0, str(_REPO / "scripts" / "analysis"))
        import rc_humanlikeness as rc  # type: ignore
    except Exception as e:  # pragma: no cover
        report["status"] = "ERROR"
        report["note"] = f"rc_humanlikeness import failed: {e!r}"
        return report

    bot = rc.collect_bot(npz)
    human = rc.collect_human(ctx.corpus_dir, "precomputed_val")
    cmp = {
        "turn": rc.cmp_dist(human["turn"], human["hz"], bot["turn"], bot["hz"], mmd=False),
        "move_dwell": {ax: rc.cmp_dist(human["move_dwell"][ax], human["hz"],
                                       bot["move_dwell"][ax], bot["hz"], mmd=False)
                       for ax in ("fb", "lr", "ud") if ax in bot["move_dwell"]},
        "attack_run": rc.cmp_dist(human["attack_run"], human["hz"],
                                  bot["attack_run"], bot["hz"], mmd=False),
    }
    turn_emd_deg = cmp["turn"]["emd_sec"] * bot["hz"]  # back to deg (cmp divides by hz)
    h_opfire = float(human.get("attack_switch", 0.0))
    b_opfire = float(bot.get("attack_switch", 0.0))
    opfire_rel = abs(b_opfire - h_opfire) / h_opfire if h_opfire else float("inf")

    turn_ok = turn_emd_deg <= GATE_TURN_EMD_DEG_MAX
    opfire_ok = opfire_rel <= GATE_OPFIRE_REL_TOL
    passed = turn_ok and opfire_ok
    report.update({
        "status": "PASS" if passed else "FAIL",
        "eval_npz": str(npz),
        "comparison": cmp,
        "gates": {
            "turn_emd_deg": round(turn_emd_deg, 3), "turn_emd_deg_max": GATE_TURN_EMD_DEG_MAX,
            "turn_ok": turn_ok,
            "opfire_human": round(h_opfire, 4), "opfire_bot": round(b_opfire, 4),
            "opfire_rel_delta": round(opfire_rel, 3) if np.isfinite(opfire_rel) else None,
            "opfire_rel_tol": GATE_OPFIRE_REL_TOL, "opfire_ok": opfire_ok,
        },
    })
    _log(f"Stage 6 validation: {report['status']} "
         f"(turn-EMD {turn_emd_deg:.2f}° ≤{GATE_TURN_EMD_DEG_MAX}; "
         f"op-fire {b_opfire:.3f} vs human {h_opfire:.3f})")
    return report


# ══ heavy-stage runners (GPU / multiprocessing) ══════════════════════════════

def _run_script(rel_path: str, args: list[str], *, why: str) -> None:
    script = _REPO / rel_path
    _log(f"running {rel_path} ({why}) …")
    env_py = dict(__import__("os").environ)
    env_py["PYTHONPATH"] = f"{_REPO/'src'}:{env_py.get('PYTHONPATH','')}"
    subprocess.run([sys.executable, str(script), *args], cwd=_REPO, env=env_py, check=True)


def _refit_aim_closed_loop(ctx: RunContext) -> None:
    """Re-run the closed-loop dampened aim grid (GPU). Heavy + gated — see
    `feedback_no_collect_alongside_trainer` / `gpu_resident_oom`. The frontier +
    psegment anchors are produced by aim_frontier_dampened.py against this run."""
    _run_script("scripts/analysis/aim_frontier_dampened.py",
                ["--run-dir", str(ctx.run_dir)],
                why="closed-loop dampened aim frontier + s→gain anchors")
    _run_script("scripts/analysis/aim_degrade_sweep.py",
                ["--run-dir", str(ctx.run_dir), "--collect-dir", str(ctx.corpus_dir)],
                why="DOWN-half universal lag sweep")


def _refit_pins_offline(ctx: RunContext) -> dict | None:
    """Re-derive move/weapon/attack pins + corpus hazard via qnn.bc.decode_fit.fit.
    Needs torch + the corpus; returns None (and warns) if unavailable here."""
    try:
        from qnn.bc import decode_fit as _df
    except Exception as e:  # pragma: no cover
        _log(f"--refit-pins unavailable ({e!r}); keeping oracle pins")
        return None
    return _df.fit(ctx.run_dir, ORACLE_TEMPLATE)


# ══ orchestration ════════════════════════════════════════════════════════════

def run_pipeline(
    run_dir: Path, *, skill: str = DEFAULT_SKILL, write: bool = False,
    validate: bool = False, refit_aim: bool = False, refit_pins: bool = False,
    version: str | None = None, out_dir: Path | None = None,
    eval_npz: Path | None = None,
) -> dict[str, Any]:
    s = _parse_skill(skill)
    ctx = resolve_run_context(run_dir)
    # Output goes to a pipeline-owned dir, NOT under the immutable training run-dir
    # (`feedback_runs_dirs_immutable`). The emitted config's look_grid stays
    # run-relative ("config/look_grid.json") and is resolved against --run-dir at
    # export time, so the config file may live anywhere.
    out_dir = Path(out_dir) if out_dir else (_REPO / "runs" / "decode_fit" / ctx.model_id)
    _log(f"run={ctx.model_id} ckpt={ctx.checkpoint.name} corpus_fp={ctx.corpus_fingerprint[:16]} "
         f"look_grid_sha={ctx.look_grid_sha[:12]} skill=p{s:.0f}")

    human = ensure_human_baseline(ctx, refit=refit_aim)
    aim = load_aim_curve(ctx, refit=refit_aim)
    pins = load_nonlever_pins(ctx, refit=refit_pins)
    verify = verify_model_aim(ctx, human)
    curve = synthesize_skill_curve(ctx, human, aim, pins, verify)

    result: dict[str, Any] = {"run_dir": str(ctx.run_dir), "skill_percentile": s,
                              "skill_curve": curve, "provenance": ctx.provenance()}

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        version = version or f"{ctx.model_id}-fit-p{s:.0f}"
        config_out = out_dir / f"decode.{version}.json"
        emit = emit_decode_config(ctx, curve, aim, pins, s, version, config_out)
        sidecar = write_skill_curve_sidecar(
            ctx, curve, SIDECAR_DIR / f"{ctx.model_id}.skill_curve.json")
        result["emit"] = emit
        result["sidecar"] = str(sidecar)
        gate = validate_closed_loop(ctx, config_out, s, curve, run=validate,
                                    eval_npz=eval_npz)
        result["validation"] = gate
        report_path = out_dir / "decode_fit_report.json"
        emit_ok = emit["resolve_ok"] and not emit["export_gaps"]
        gate_status = gate.get("status")
        if not emit_ok:
            result_label = "NEEDS-REVIEW"      # bad config or un-deployable knob
        elif gate_status == "FAIL":
            result_label = "FAIL"              # closed-loop gate failed
        elif gate_status == "PASS":
            result_label = "PASS"              # emitted + gate scored OK
        else:
            result_label = "PASS-PENDING-VALIDATION"  # emitted OK, gate not scored yet
        report = {"run_dir": str(ctx.run_dir), "skill": f"p{s:.0f}",
                  "version": version, "emit": emit, "validation": gate,
                  "provenance": ctx.provenance(), "result": result_label}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        _log(f"wrote {report_path.relative_to(_REPO)} → {report['result']}")
        result["report"] = str(report_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Turnkey per-model mechanical-aim decode-fit pipeline.")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="bench run-dir (checkpoint + pinned look-grid + corpus)")
    ap.add_argument("--skill", default=DEFAULT_SKILL,
                    help="baseline operating point percentile (pNN / NN / 0.NN); default p76")
    ap.add_argument("--version", default=None,
                    help="emitted decode-config version id (default <model>-fit-pNN)")
    ap.add_argument("--write", action="store_true",
                    help="emit decode.<version>.json + skill-curve sidecar + report")
    ap.add_argument("--validate", action="store_true",
                    help="run the closed-loop validation gate (scores --eval-npz)")
    ap.add_argument("--eval-npz", type=Path, default=None,
                    help="arena-eval action-stream npz (move_streams_*.npz) to score")
    ap.add_argument("--refit-aim", action="store_true",
                    help="re-run the closed-loop aim grid (GPU) instead of reusing cache")
    ap.add_argument("--refit-pins", action="store_true",
                    help="re-derive non-lever pins via qnn.bc.decode_fit.fit")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="emit dir for decode.<version>.json + report "
                         "(default runs/decode_fit/<model>/; never the immutable run-dir)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the full pipeline result JSON here")
    args = ap.parse_args()

    result = run_pipeline(
        args.run_dir, skill=args.skill, write=args.write, validate=args.validate,
        refit_aim=args.refit_aim, refit_pins=args.refit_pins, version=args.version,
        out_dir=args.out_dir, eval_npz=args.eval_npz)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        _log(f"wrote pipeline result → {args.out}")
    else:
        print(json.dumps(result["skill_curve"]["piecewise"], indent=2))


if __name__ == "__main__":
    main()
