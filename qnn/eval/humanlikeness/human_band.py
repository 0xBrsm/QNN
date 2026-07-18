#!/usr/bin/env python3
"""Human-band yardstick — is a behavior stream inside the human envelope?

The question this answers: "does this agent's closed-loop behavior fall within
the RANGE of behaviors real humans produce?" — a membership test against a
band, not a distance against a point with a hand-picked bar. Skill is
deliberately outside the test: the band is a *style* envelope (timing, holds,
switch dynamics, fire cadence); skill slides along the orthogonal decode axis
(research/skill-curves.md). A low-skill human and a high-skill human must both
sit in the band; calibration verifies that.

v5 harness axioms (research/human-band.md; the MMD² statistic is FROZEN):
  1. Perfect-imitation invariance — features are built on ENGINE-VISIBLE
     events (discharges = cooldown-gated attack decisions), never
     button/press semantics; a stream that exactly reproduces the BC training
     labels scores human by construction.
  2. Context-conditioned comparison — every channel is computed on
     keep ∧ engaged frames (engaged = LOS actor present), because the
     deployment context mix (wall-to-wall arena combat) differs from corpus
     demos by construction. Bank, null and anchor share the conditioning.
  3. Effect-size decisions — the verdict is an ANCHORED RATIO,
     mmd2_subject / mmd2_shuffled_human_anchor, per channel and family
     (worst channel); the demo-split-null percentile stays as report-only
     context (a significance bar fails every finite difference at large n).

Method
------
1. Human demos (20 Hz collect cache) are cut into fixed wall-clock windows;
   each window yields a per-channel feature vector in physical units (deg/s,
   events/s, seconds) on its keep ∧ engaged frames. 10 Hz subjects are
   compared against a 10 Hz-decimated view of the same corpus.
2. The BAND is the distribution of window features across the corpus. The
   demo-level split null (never frames/windows — autocorrelation makes those
   exchangeability assumptions false) is retained for the report-only
   percentile context.
3. A SUBJECT (model stream, held-out human, perturbed control) is scored by
   MMD^2 against the human bank; verdict per channel = anchored ratio ≤
   ANCHOR_RATIO_MAX. The anchor (frame-shuffled held-out human — broken
   dynamics, intact marginals) is built once per bank and cached in the bank
   artifact (qnn.human.band_bank).

Channels: look (turn dynamics), move (fb/lr hold-switch), attack (engine
discharge cadence: rate per engaged-second + cooldown-excess gap timing),
weapon (dwell/entropy). ud is deferred (feasibility-bit conditioning).

Calibration (all must hold before a verdict is trusted):
  * held-out human demos       -> IN band
  * label-encoded holdout      -> IN band  (Axiom-1 control: the holdout
                                            re-encoded through BC training-label
                                            semantics — op-event attack,
                                            attack-with held weapon, look-grid
                                            quantized turn)
  * top/bottom skill quartile  -> IN band  (band is skill-invariant)
  * frame-shuffled human       -> OUT      (the anchor class itself: ratio ~1)
  * dwell-stretched human      -> OUT      (a1p-style hold pathology)
  * discharge-thinned human    -> OUT      (a1-style under-fire)

Usage:
  PYTHONPATH=src python -m qnn.eval.humanlikeness.human_band --calibrate
  PYTHONPATH=src python -m qnn.eval.humanlikeness.human_band \
      --npz runs/.../metrics/eval/move_streams_sampled.npz [--npz ...]

Model streams: the flat behavior block written by qnn.eval.run with
eval_log_action_streams=true (move_streams_*.npz): fb/lr/ud {0,1,2},
attack {0,1}, weapon byte, turn_deg, keep, plus the v5 fields discharge
(attack ∧ cooldown-ready), weapon_imp (held weapon impulse 0..8) and engaged
(LOS actor present), episode_offsets, tick_hz. Pre-v5 npz FAIL LOUD (the
attack/engagement conditioning cannot be reconstructed from them); an explicit
--legacy-engaged-keep maps engaged:=keep and drops the attack channel, stamped
into the report — never a silent default.

This module owns the model-specific SCORING; the corpus WINDOW-FEATURE BANK it
scores against is the model-agnostic per-collect human baseline in
``qnn.human.band_bank`` (cached under ``<collect>/human_baseline/``). Writes the
verdict report to runs/head_probe/_human_band.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from qnn.eval.humanlikeness.core import mmd2_rbf, _median_heuristic_gamma
# The corpus WINDOW-FEATURE BANK — the model-agnostic per-collect human baseline this
# test scores against — lives in qnn.human. This module owns only the model-specific
# SCORING (standardization, anchored-ratio verdict, demo-split null context). The bank
# symbols are re-exported so ``human_band.<sym>`` still resolves across both halves
# (decode-fit's stage-6 gate + the calibration CLI).
from qnn.human.band_bank import (  # noqa: F401
    BANK_VERSION,
    CHANNELS,
    FEATURE_NAMES,
    FLICK_DEG,
    HUMAN_HZ,
    ZERO_TURN_DEG,
    bank_cache_path,
    bank_filter,
    decimate2,
    featurize,
    load_human_episodes,
    load_or_build_bank,
    shuffle_episode,
    window_features,
)

DEFAULT_OUT = Path("runs/head_probe/_human_band.json")

# Anchored-ratio verdict bar (Axiom 3): in-band iff
# mmd2_subject / mmd2_shuffled_anchor <= ANCHOR_RATIO_MAX, per channel and for
# the family (worst channel). Provisional default from the v5 qwd calibration
# (2026-07-16, 2712-window bank): every IN case (holdout 0.004, label-encoded
# 0.007, skill quartiles ≤ 0.081) vs every OUT-carrying channel (dwell-stretch
# look 0.432, attack-thin 1.022, shuffle ~1.03 ≈ 1 by construction). 0.2 is
# the geometric midpoint of the worst-IN / lowest-OUT gap [0.081, 0.432] —
# ~2.5x margin on BOTH sides, the widest defensible bar. The decode-fit lead
# finalizes the gate bar; this constant is the scorer's default.
ANCHOR_RATIO_MAX = 0.2

_V5_FIELDS = ("discharge", "weapon_imp", "engaged")


# ---------------------------------------------------------------------------
# Subject (model eval) episode loading
# ---------------------------------------------------------------------------
def load_rc_episodes(npz_path: Path, *,
                     legacy_engaged_keep: bool = False) -> tuple[float, list[dict]]:
    """Flat behavior-stream npz -> (tick_hz, [episode dicts]).

    Requires the v5 flat block from qnn.eval.run (eval_log_action_streams=true):
    fb/lr/attack/weapon/turn_deg/keep + discharge/weapon_imp/engaged. Pre-v5
    npz raise (the discharge + engagement conditioning cannot be reconstructed
    from the raw attack stream) unless ``legacy_engaged_keep`` is passed, which
    maps engaged:=keep and zeroes the attack-channel inputs — the caller must
    surface that degradation (the CLI stamps it into the report; never pass it
    from an automated gate).
    """
    z = np.load(npz_path)
    if "tick_hz" not in z:
        raise ValueError(
            f"{npz_path}: no flat behavior block (legacy move-only npz?) — "
            "re-run the eval with eval_log_action_streams=true")
    missing = [k for k in _V5_FIELDS if k not in z]
    if missing and not legacy_engaged_keep:
        raise ValueError(
            f"{npz_path}: pre-v5 stream npz — missing {missing}. The v5 band "
            "conditions on engaged frames and scores engine discharges; these "
            "cannot be reconstructed from the raw attack stream. Re-run the "
            "eval (qnn.eval.run now writes them), or pass --legacy-engaged-keep "
            "to score the non-attack channels with engaged:=keep.")
    hz = float(np.asarray(z["tick_hz"]).reshape(-1)[0])
    n = int(z["fb"].shape[0])
    offs = np.asarray(z["episode_offsets"], dtype=np.int64).reshape(-1)
    if offs.size == 0 or offs[0] != 0:
        offs = np.concatenate([[0], offs])
    if offs[-1] != n:
        offs = np.concatenate([offs, [n]])
    eps = []
    for s, e in zip(offs[:-1], offs[1:]):
        sl = slice(int(s), int(e))
        keep = np.asarray(z["keep"][sl], dtype=bool)
        eps.append({
            "fb": np.asarray(z["fb"][sl], dtype=np.int64),
            "lr": np.asarray(z["lr"][sl], dtype=np.int64),
            "attack": np.asarray(z["attack"][sl], dtype=np.int64),
            "weapon": np.asarray(z["weapon"][sl], dtype=np.int64),
            "wimp": (np.asarray(z["weapon_imp"][sl], dtype=np.int64)
                     if not missing else np.zeros(e - s, dtype=np.int64)),
            "discharge": (np.asarray(z["discharge"][sl], dtype=bool)
                          if not missing else np.zeros(e - s, dtype=bool)),
            "turn": np.asarray(z["turn_deg"][sl], dtype=np.float64),
            "keep": keep,
            "engaged": (np.asarray(z["engaged"][sl], dtype=bool)
                        if not missing else keep.copy()),
        })
    return hz, eps


# ---------------------------------------------------------------------------
# Band: standardization, gamma, demo-split null, shuffled anchor
# ---------------------------------------------------------------------------
def band_context(bank: dict, seed: int) -> dict:
    """Per-channel robust standardization + fixed RBF bandwidth from the bank."""
    rng = np.random.default_rng(seed)
    ctx = {}
    for ch in CHANNELS:
        x = bank[ch]["X"]
        mu = np.median(x, axis=0)
        iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        sig = np.where(iqr > 0, iqr / 1.349, np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0))
        z = (x - mu) / sig
        sub = z[rng.choice(z.shape[0], min(400, z.shape[0]), replace=False)]
        half = sub.shape[0] // 2
        gamma = _median_heuristic_gamma(sub[:half], sub[half:], rng)
        ctx[ch] = {"mu": mu, "sig": sig, "gamma": gamma}
    return ctx


def _z(ctx: dict, ch: str, x: np.ndarray) -> np.ndarray:
    return (x - ctx[ch]["mu"]) / ctx[ch]["sig"]


def build_null(bank: dict, ctx: dict, n: int, b_splits: int, seed: int,
               skill: dict | None = None) -> dict:
    """Demo-level split null: {channel: (B,) MMD^2 samples}. REPORT-ONLY
    context in v5 (percentiles); the verdict is the anchored ratio.

    Two split families, mixed 50/50 when a ``skill`` map (demo -> coh_5deg) is
    available:
      * random half-splits — pure sampling self-variation;
      * skill-cohort splits — a random CONTIGUOUS block (5-50% of demos,
        sorted by skill) vs the rest. A real human is skill-coherent, so the
        band must contain skill-coherent subpopulations down to small groups
        (~10 demos); random halves alone under-state human self-variation on
        skill-correlated channels (attack cadence/timing, weapon usage —
        v1/v1.2/v4 calibration findings).

    The same B splits are used for every channel so the family (max-rank)
    context can be computed without an independence assumption.
    """
    rng = np.random.default_rng(seed)
    demos = np.unique(np.concatenate([bank[ch]["demo"] for ch in CHANNELS]))
    scored = np.array(sorted((d for d in demos if skill and d in skill),
                             key=lambda d: skill[d])) if skill else np.empty(0, np.int64)
    null = {ch: np.full(b_splits, np.nan) for ch in CHANNELS}
    for b in range(b_splits):
        if scored.size >= 40 and b % 2 == 1:
            frac = rng.uniform(0.05, 0.5)
            width = max(int(frac * scored.size), 10)
            start = rng.integers(0, scored.size - width + 1)
            half = set(scored[start:start + width].tolist())
        else:
            perm = rng.permutation(demos)
            half = set(perm[: demos.size // 2].tolist())
        for ch in CHANNELS:
            dd = bank[ch]["demo"]
            in_half = np.isin(dd, list(half))
            i1 = np.nonzero(in_half)[0]
            i2 = np.nonzero(~in_half)[0]
            if i1.size < n or i2.size < n:
                continue
            a = bank[ch]["X"][rng.choice(i1, n, replace=False)]
            c = bank[ch]["X"][rng.choice(i2, n, replace=False)]
            null[ch][b] = mmd2_rbf(_z(ctx, ch, a), _z(ctx, ch, c), gamma=ctx[ch]["gamma"])
    return {ch: v[np.isfinite(v)] for ch, v in null.items()}


def _mmd2_draws(xs: np.ndarray, xr: np.ndarray, ctx: dict, ch: str,
                n: int, repeats: int, rng: np.random.Generator) -> float:
    """Median MMD² over ``repeats`` subsample draws (subject xs vs reference xr)."""
    n_use = min(n, xs.shape[0], xr.shape[0])
    draws = np.empty(repeats)
    for r in range(repeats):
        a = xs[rng.choice(xs.shape[0], n_use, replace=False)]
        c = xr[rng.choice(xr.shape[0], n_use, replace=False)]
        draws[r] = mmd2_rbf(_z(ctx, ch, a), _z(ctx, ch, c), gamma=ctx[ch]["gamma"])
    return float(np.median(draws))


def anchor_mmd2s(ref_bank: dict, ctx: dict, n: int, repeats: int,
                 seed: int) -> dict[str, float]:
    """Per-channel anchor denominators: median MMD² of the bank's cached
    frame-shuffled held-out-human windows vs the bank minus that holdout."""
    if "_anchor" not in ref_bank:
        raise ValueError(
            "bank has no shuffled-human anchor (pre-v5 artifact) — rebuild it "
            "via qnn.human.band_bank.load_or_build_bank")
    rng = np.random.default_rng(seed)
    hold = set(np.asarray(ref_bank["_anchor_demos"]).tolist())
    out: dict[str, float] = {}
    for ch in CHANNELS:
        xa = ref_bank["_anchor"][ch]["X"]
        dd = ref_bank[ch]["demo"]
        xr = ref_bank[ch]["X"][~np.isin(dd, list(hold))]
        if xa.shape[0] < 8 or xr.shape[0] < 8:
            out[ch] = float("nan")
            continue
        out[ch] = _mmd2_draws(xa, xr, ctx, ch, n, repeats, rng)
    return out


def score_subject(subj_bank: dict, ref_bank: dict, ctx: dict, null: dict,
                  n: int, repeats: int, seed: int) -> dict:
    """Score a subject bank against the band. Returns per-channel + family
    verdicts. The decision field is ``anchored_ratio`` (Axiom 3):
    mmd2_subject / mmd2_shuffled_anchor, in-band iff ≤ ANCHOR_RATIO_MAX; the
    null percentile fields are report-only context."""
    rng = np.random.default_rng(seed)
    anchors = anchor_mmd2s(ref_bank, ctx, n, repeats, seed)
    res: dict = {"channels": {}, "n_windows": {ch: int(subj_bank[ch]["X"].shape[0])
                                               for ch in CHANNELS}}
    fam_ratio: float | None = None
    fam_rank = 0.0
    fam_defined = False
    # Per-split channel ranks -> family null of the worst-channel rank (report-only).
    fam_null = None
    for ch in CHANNELS:
        xs = subj_bank[ch]["X"]
        xr = ref_bank[ch]["X"]
        nul = null[ch]
        if xs.shape[0] < 8 or nul.size < 20 or not math.isfinite(anchors[ch]):
            res["channels"][ch] = {"mmd2": None, "anchored_ratio": None,
                                   "in_band": None, "pct": None,
                                   "note": "insufficient windows"}
            continue
        obs = _mmd2_draws(xs, xr, ctx, ch, n, repeats, rng)
        ratio = obs / anchors[ch]
        pct = float(100.0 * (nul < obs).mean())
        res["channels"][ch] = {
            "mmd2": obs,
            "anchored_ratio": round(float(ratio), 4),
            "anchor_mmd2": round(float(anchors[ch]), 6),
            "in_band": bool(ratio <= ANCHOR_RATIO_MAX),
            # report-only null context (Axiom 3: never the decision)
            "pct": round(pct, 2),
            "null_p50": float(np.percentile(nul, 50)),
            "null_p95": float(np.percentile(nul, 95)),
        }
        fam_ratio = ratio if fam_ratio is None else max(fam_ratio, ratio)
        fam_rank = max(fam_rank, (nul < obs).mean())
        fam_defined = True
        # rank each null sample within its own channel null (report-only)
        order = nul.argsort().argsort() / max(nul.size - 1, 1)
        if fam_null is None:
            fam_null = order
        else:
            m = min(fam_null.size, order.size)
            fam_null = np.maximum(fam_null[:m], order[:m])
    if fam_defined and fam_ratio is not None:
        res["family"] = {
            "worst_anchored_ratio": round(float(fam_ratio), 4),
            "ratio_max": ANCHOR_RATIO_MAX,
            "in_band": bool(fam_ratio <= ANCHOR_RATIO_MAX),
            # report-only max-rank context
            "worst_channel_rank": round(float(fam_rank), 4),
            "null_p95_rank": (round(float(np.percentile(fam_null, 95)), 4)
                              if fam_null is not None and fam_null.size else None),
        }
    return res


# ---------------------------------------------------------------------------
# Perturbation / re-encoding controls
# ---------------------------------------------------------------------------
def perturb_shuffle(ep: dict, rng: np.random.Generator) -> dict:
    """Frame shuffle within episode: marginals intact, dynamics destroyed."""
    return shuffle_episode(ep, rng)


def perturb_dwell2(ep: dict, rng: np.random.Generator) -> dict:
    """Time-stretch ×2: every STATE frame twice (dwells double, switch cadence
    halves — a1p-ish); discharge EVENTS keep one tick and double their gaps
    (repeating an event frame would fabricate sub-cooldown discharge pairs,
    which is a different pathology than stretch)."""
    out = {k: np.repeat(v, 2, axis=0) for k, v in ep.items()}
    d = np.zeros(2 * len(ep["discharge"]), dtype=bool)
    d[0::2] = ep["discharge"]
    out["discharge"] = d
    return out


def perturb_disch_thin3(ep: dict, rng: np.random.Generator) -> dict:
    """Keep every 3rd discharge event: chronic under-fire at intact context
    (a1-ish). Port of the v4 fire-onset thinning onto the discharge stream."""
    out = dict(ep)
    d = ep["discharge"].copy()
    pos = np.nonzero(d)[0]
    d[pos[np.arange(pos.size) % 3 != 0]] = False
    out["discharge"] = d
    return out


PERTURBATIONS = {
    "shuffled": perturb_shuffle,
    "dwell_x2": perturb_dwell2,
    "disch_thin_x3": perturb_disch_thin3,
}


def _look_grid_mag_centers(data_dir: Path) -> tuple[np.ndarray, float]:
    """(mag_centers_rad incl. hold 0, hold_max_rad) from the collect's pinned
    corpus-fit look grid — the grid every run pins (qnn.human.look_grid)."""
    from qnn.human.look_grid import pinned_grid_from_collect
    g = pinned_grid_from_collect(data_dir)
    return (np.asarray(g["mag_centers_rad"], dtype=np.float64),
            float(g["hold_max_rad"]))


def label_encode_episode(ep: dict, mag_centers: np.ndarray,
                         hold_max: float) -> dict:
    """Axiom-1 control: the episode re-encoded through its BC TRAINING-LABEL
    semantics (a perfect imitator's output). Recipe validated against v4 by the
    axiom1_control harness:
      attack : the op-event stream (== discharge; single-tick decisions)
      weapon : a25 attack-with — held weapon changes only AT a discharge, to
               the label weapon there (forward-fill)
      turn   : magnitude snapped to the pinned polar look grid (θ < hold_max →
               0, else nearest non-hold mag center)
      fb/lr, discharge/wimp/keep/engaged : label == raw (identity).
    Must score IN."""
    theta = np.deg2rad(ep["turn"])
    nonhold = mag_centers[mag_centers > 0]
    q = nonhold[np.abs(theta[:, None] - nonhold[None, :]).argmin(axis=1)]
    turn_q = np.rad2deg(np.where(theta < hold_max, 0.0, q))

    disch = ep["discharge"].astype(bool)
    idx = np.where(disch, np.arange(disch.size), -1)
    last = np.maximum.accumulate(idx)
    weapon = np.where(last >= 0, ep["weapon"][np.clip(last, 0, None)],
                      ep["weapon"][0] if disch.size else 0)

    out = dict(ep)
    out["attack"] = disch.astype(ep["attack"].dtype)
    out["weapon"] = weapon.astype(ep["weapon"].dtype)
    out["turn"] = turn_q
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_result(name: str, res: dict) -> None:
    fam = res.get("family", {})
    verdict = "--"
    if fam:
        verdict = "IN BAND" if fam["in_band"] else "OUT"
    print(f"\n{name}   [{verdict}]"
          + (f"  worst-ratio={fam['worst_anchored_ratio']:.3f}"
             f" (max={fam['ratio_max']:.2f})" if fam else ""))
    for ch in CHANNELS:
        c = res["channels"].get(ch, {})
        if c.get("anchored_ratio") is None:
            print(f"  {ch:<8} {c.get('note', 'n/a')}")
            continue
        mark = "ok " if c["in_band"] else "OUT"
        print(f"  {ch:<8} {mark}  ratio={c['anchored_ratio']:7.3f}"
              f"  mmd2={c['mmd2']:+.5f}  pct={c['pct']:6.2f}"
              f"  null[p50={c['null_p50']:+.5f} p95={c['null_p95']:+.5f}]"
              f"  n_win={res['n_windows'][ch]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="precomputed_val")
    ap.add_argument("--npz", type=Path, action="append", default=[],
                    help="model move_streams_*.npz subject(s) to score")
    ap.add_argument("--calibrate", action="store_true",
                    help="run the calibration suite (holdout / label-encoded / "
                         "skill / perturbations)")
    ap.add_argument("--legacy-engaged-keep", action="store_true",
                    help="score pre-v5 subject npz with engaged:=keep and NO "
                         "attack channel (loud, opt-in degradation)")
    ap.add_argument("--window-sec", type=float, default=15.0)
    ap.add_argument("--n-win", type=int, default=256)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--holdout-demos", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--profile", type=Path,
                    default=Path("runs/head_probe/_player_profile.json"),
                    help="player profile JSON for the skill-invariance check")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    report: dict = {
        "_meta": {
            "split": args.split, "data_dir": str(args.data_dir),
            "window_sec": args.window_sec, "n_win": args.n_win,
            "n_null": args.n_null, "repeats": args.repeats, "seed": args.seed,
            "channels": list(CHANNELS), "features": FEATURE_NAMES,
            "bank_version": BANK_VERSION,
            "verdict_rule": ("per-channel in-band iff anchored_ratio = "
                             "MMD^2/anchor <= "
                             f"{ANCHOR_RATIO_MAX} (anchor = frame-shuffled "
                             "held-out human, cached in the bank); family = "
                             "worst channel ratio; null percentiles report-only"),
        },
        "calibration": {},
        "subjects": {},
    }

    print(f"human_band: building 20 Hz v{BANK_VERSION} bank (split={args.split}, "
          f"window={args.window_sec}s) ...")
    bank20, human_eps = load_or_build_bank(args.data_dir, args.split,
                                           HUMAN_HZ, args.window_sec)
    n_win20 = {ch: int(bank20[ch]["X"].shape[0]) for ch in CHANNELS}
    print(f"  windows: {n_win20}")

    skill_map: dict[int, float] = {}
    if args.profile.exists():
        for p in json.loads(args.profile.read_text()).get("players", []):
            c = p.get("aim", {}).get("coh_5deg")
            if c is not None and math.isfinite(c):
                skill_map[int(p["demo_idx"])] = float(c)
    report["_meta"]["skill_proxy"] = f"coh_5deg from {args.profile} ({len(skill_map)} demos)"

    # ---------------- calibration ----------------------------------------
    if args.calibrate:
        if human_eps is None:
            human_eps = load_human_episodes(args.data_dir, args.split)
        rng = np.random.default_rng(args.seed)
        all_demos = np.unique([d for d, _ in human_eps])
        hold = set(rng.choice(all_demos, args.holdout_demos, replace=False).tolist())
        band_bank = bank_filter(bank20, hold, exclude=True)
        ctx = band_context(band_bank, args.seed)
        null = build_null(band_bank, ctx, args.n_win, args.n_null, args.seed,
                          skill=skill_map)
        print(f"\ncalibration band: {all_demos.size - len(hold)} demos; "
              f"holdout {len(hold)} demos")
        hold_eps = [(d, ep) for d, ep in human_eps if d in hold]

        # 1) held-out humans must pass
        subj = bank_filter(bank20, hold, exclude=False)
        res = score_subject(subj, band_bank, ctx, null,
                            args.n_win, args.repeats, args.seed)
        res["expect"] = "in_band"
        report["calibration"]["human_holdout"] = res
        print_result("human_holdout (expect IN)", res)

        # 2) Axiom-1 control: the SAME holdout re-encoded as its BC training
        # labels must also pass (perfect imitation scores human).
        mc, hold_max = _look_grid_mag_centers(args.data_dir)
        eps_l = [label_encode_episode(ep, mc, hold_max) for _, ep in hold_eps]
        subj_l = featurize(eps_l, HUMAN_HZ, args.window_sec,
                           [d for d, _ in hold_eps])
        res = score_subject(subj_l, band_bank, ctx, null,
                            args.n_win, args.repeats, args.seed)
        res["expect"] = "in_band"
        report["calibration"]["label_encoded_holdout"] = res
        print_result("label_encoded_holdout (expect IN)", res)

        # 3) perturbed holdout humans must fail
        for pname, pfun in PERTURBATIONS.items():
            prng = np.random.default_rng(args.seed + 1)
            eps_p = [pfun(ep, prng) for _, ep in hold_eps]
            subj_p = featurize(eps_p, HUMAN_HZ, args.window_sec,
                               [d for d, _ in hold_eps])
            res = score_subject(subj_p, band_bank, ctx, null,
                                args.n_win, args.repeats, args.seed)
            res["expect"] = "out_of_band"
            report["calibration"][pname] = res
            print_result(f"{pname} (expect OUT)", res)

        # 4) skill quartiles must pass (band skill-invariance). Hold out HALF
        # of each quartile as the subject; the other half stays in the band —
        # deployment condition is "new skill-coherent human vs a corpus that
        # spans the skill range", not extrapolation to an amputated cohort.
        coh = {d: c for d, c in skill_map.items() if d in all_demos}
        if len(coh) >= 40:
            order = sorted(coh, key=coh.get)
            q = len(order) // 4
            qrng = np.random.default_rng(args.seed + 2)
            for qname, qfull in (("skill_bottom_quartile", order[:q]),
                                 ("skill_top_quartile", order[-q:])):
                qset = set(qrng.choice(sorted(qfull), len(qfull) // 2,
                                       replace=False).tolist())
                band_q = bank_filter(bank20, qset, exclude=True)
                ctx_q = band_context(band_q, args.seed)
                null_q = build_null(band_q, ctx_q, args.n_win, args.n_null,
                                    args.seed, skill=skill_map)
                subj_q = bank_filter(bank20, qset, exclude=False)
                res = score_subject(subj_q, band_q, ctx_q, null_q,
                                    args.n_win, args.repeats, args.seed)
                res["expect"] = "in_band"
                res["coh5_range"] = [round(coh[d], 4) for d in
                                     (min(qset, key=coh.get), max(qset, key=coh.get))]
                report["calibration"][qname] = res
                print_result(f"{qname} (expect IN)", res)
        else:
            print("  skill check skipped: <40 profiled demos")

    # ---------------- model subjects -------------------------------------
    if args.npz:
        ctx20 = band_context(bank20, args.seed)
        bank10 = ctx10 = None
        null_cache: dict = {}
        for npz in args.npz:
            try:
                hz, eps = load_rc_episodes(
                    npz, legacy_engaged_keep=args.legacy_engaged_keep)
            except ValueError as e:
                print(f"SKIP {e}")
                continue
            subj = featurize(eps, hz, args.window_sec)
            if args.legacy_engaged_keep:
                # legacy npz: the zeroed discharge stream would score as a
                # maximal under-fire artifact — drop the channel VISIBLY.
                subj["attack"] = {"X": np.empty((0, len(FEATURE_NAMES["attack"]))),
                                  "demo": np.empty(0, dtype=np.int64)}
            if math.isclose(hz, HUMAN_HZ):
                ref, ctx, hzkey = bank20, ctx20, 20
            else:
                if bank10 is None:
                    print(f"building {hz:g} Hz decimated bank ...")
                    bank10, human_eps = load_or_build_bank(
                        args.data_dir, args.split, hz, args.window_sec, human_eps)
                    ctx10 = band_context(bank10, args.seed)
                ref, ctx, hzkey = bank10, ctx10, 10
            counts = [int(subj[ch]["X"].shape[0]) for ch in CHANNELS
                      if subj[ch]["X"].shape[0] >= 8]
            if not counts:
                print(f"{npz}: no usable windows, skipped")
                continue
            # null must be built at the n actually compared, else small
            # subjects read against a tighter (larger-n) null
            n_use = min(args.n_win, min(counts))
            if (hzkey, n_use) not in null_cache:
                null_cache[(hzkey, n_use)] = build_null(
                    ref, ctx, n_use, args.n_null, args.seed, skill=skill_map)
            res = score_subject(subj, ref, ctx, null_cache[(hzkey, n_use)],
                                n_use, args.repeats, args.seed)
            res["hz"] = hz
            if args.legacy_engaged_keep:
                res["legacy_engaged_keep"] = (
                    "pre-v5 npz scored with engaged:=keep; attack channel "
                    "dropped (no discharge stream)")
            report["subjects"][str(npz)] = res
            print_result(f"{npz} (hz={hz:g})", res)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
