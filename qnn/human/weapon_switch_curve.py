"""Within-encounter weapon-switch DECISION curve (HUMAN) — the fit target for
the weapon-switch evidence accumulator (agents/plans/weapon-switch-evidence.md).

Ruler: within a v6 ENCOUNTER, when a player makes a firing DECISION, does it
pick the same weapon as the previous decision or a different one, and how does
that depend on how long they waited past the first moment a next shot was
mechanically possible? ``P(w_k+1 != w_k)`` as a smooth function of that delay.
Marginal switch *rate* is explicitly not the target — a memoryless threshold on
noisy instantaneous evidence can match the marginal rate while still
chattering.

WHY THIS IS NOT A GAP-BINNED CURVE (respec 2026-08-19, Brian). The first cut
of this ruler binned ``P(switch)`` by raw inter-discharge gap in seconds. That
formulation is unusable and the (λ, θ) fit built on it refused with a flat
objective (0.00047-0.00085 across λ∈{0.9,0.95}, θ∈[3,9] — no discrimination at
all). Three measured defects, all fixed here:

  1. THE GAP AXIS WAS THE WEAPON AXIS. The engine quantizes the inter-discharge
     gap to the firing weapon's ``attack_finished`` cooldown, so every gap bin
     was one weapon: [0,0.25) = LG/NG (89%), [0.5,0.75) = SG (87%),
     [0.75,1) = RL (82%). The "curve" was a per-weapon marginal switch-rate
     table with weapons relabeled by their refire times — very nearly the
     marginal rate the ruler exists to avoid. Pooling also ATTENUATED the real
     effect: logistic slope on log(gap) is +1.32 pooled but +3.44 conditioned
     on weapon family (deviance drop 1895 over family-only). Fix: condition on
     the weapon and measure delay in units of that weapon's own cooldown
     (``COOLDOWN_S``, the canonical ``qnn.bc.weapon_physics`` table) past the
     first DECIDABLE opportunity (``DECISION_FLOOR_S``).
  2. THE TWO SIDES WERE NOT COUNTING THE SAME EVENT. NG/SNG/LG stream a 0.1s
     think-chain while the trigger is held (``qnn.bc.attack_vocab.CONTINUOUS``),
     so raw discharges over-count trigger pulls ~2x on those weapons; the human
     corpus resolves LG at a 0.20s floor while the bot's own stream resolves it
     at 0.10s. 37% of human pairs and 84% of bot pairs were hold-train
     continuations, not decisions — and the fit's objective mass and its whole
     short-gap gate sat on them. Fix: gate events to hold-train ONSETS on both
     sides, reusing the already-settled definition (``OnsetGate`` /
     ``CONTINUOUS_ONSET_IMPULSES`` / ``DEFAULT_ONSET_GAP_TICKS`` from
     ``qnn.ppo.crest_reward``) rather than inventing a second one.
  3. BINS THEMSELVES. Corpus-mass-chosen edges plus a per-bin ``min_mass``
     zeroing rule meant two candidates could be scored on different bin
     subsets. Fix: no bins anywhere. The target is a TWO-PART curve fit by
     pair-level maximum likelihood with the encounter cluster bootstrap
     resampling the whole fit:

       P(switch | x = 0)  -- a free per-family rate (the ATOM)
       P(switch | x > 0)  -- logit = intercept[family] + slope * log1p(x)

     The atom is not a modelling nicety: x = 0 ("took the earliest shot the
     engine allowed") carries 65% of the human's decision mass, and it is a
     distinct behavioural state rather than a point on a continuum. A
     single-form curve was measured against this corpus first and is
     misspecified exactly there — it fits the marginal switch rate of every
     family to 4 decimals while missing sg_fam's x = 0 rate by 81% relative
     (0.071 fitted vs 0.039 observed on 7833 pairs), which is fatal because
     P(switch | x = 0) is the statistic the placement gate adjudicates. A free
     atom is the saturated-cell MLE there, so it cannot be wrong, and the
     smooth part then describes only the delay continuum it is actually about.

Every other primitive is REUSED, not reinvented (feedback_reuse_existing_tooling,
feedback_no_legacy_paths_without_request):

  * encounter unit      -- ``qnn.human.encounters.corpus_pid_encounter_spans``
                           (the v6 same-opponent pid-run slicer).
  * discharge events    -- ``qnn.eval.humanlikeness.rc._preference_events``
                           (keep-gated, ``attack`` in 1..8 impulse space).
  * forced exclusion    -- ``qnn.eval.humanlikeness.rc._invalidated_pairs``
                           (stationary-menu gate over ``(onset_k, onset_k+1]``,
                           which deliberately spans the train: an LG burst that
                           ended because the cells ran dry is a FORCED switch,
                           and the interval catches it).
  * onset definition    -- ``qnn.ppo.crest_reward`` (above).
  * weapon cooldowns    -- ``qnn.bc.weapon_physics.WEAPON_PHYSICS``.

This module's own job: the train grouping, the encounter-scoped decision
PAIRING, the covariate, the binless curve fit + encounter cluster bootstrap,
and the mid-train switch rate. No held-weapon concept anywhere: weapon
identity exists only at discharges.

Output (<collect>/human_baseline/_weapon_switch_transition_curve.json):
  event stats (discharges -> decisions collapse, per weapon — the
  cross-side comparability check); gap + covariate distributions; the fitted
  curve (per-family intercept, shared slope, bootstrap CIs, P(switch | x=0)
  per family, and the per-family integration grid the fit's objective
  integrates against); the mid-train switch rate per continuous weapon (the
  chatter ruler); per-weapon-FAMILY transition matrix (diagnostic).

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.weapon_switch_curve \\
      --collect-dir artifacts/collect/qwd_v4d_v3vis [--split val] [--workers 8] \\
      [--out <collect>/human_baseline/_weapon_switch_transition_curve.json]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

import qnn.engine_norm as en
from qnn.bc.weapon_physics import WEAPON_PHYSICS
from qnn.eval import aim_kernel as A
from qnn.eval.humanlikeness.rc import _invalidated_pairs, _preference_events
from qnn.human.encounters import corpus_pid_encounter_spans, pid_recency_from_tokens
from qnn.ppo.crest_reward import CONTINUOUS_ONSET_IMPULSES, DEFAULT_ONSET_GAP_TICKS

HZ = 20.0

# Artifact schema tag. Consumers (the h2h adapter, the decode-fit phase) MUST
# check it: the pre-respec gap-binned artifact has the same filename and a
# `curve` key of an entirely different shape, and silently fitting against it
# would reproduce the exact failure this respec removes.
SCHEMA = "weapon-switch-decision-curve-v2"

# ── event / covariate definitions ───────────────────────────────────────────
# Hold-train onset gate, reused verbatim from the PPO crest reward's already-
# settled definition (qnn.ppo.crest_reward): on NG/SNG/LG a discharge is a
# DECISION only when it is >gap_ticks since that weapon's previous discharge;
# everything closer is the engine's 0.1s think-chain re-firing under a held
# trigger. Every other weapon's discharge is its own trigger pull.
ONSET_GAP_TICKS: dict[int, int] = dict(DEFAULT_ONSET_GAP_TICKS)
CONTINUOUS: frozenset[int] = frozenset(CONTINUOUS_ONSET_IMPULSES)

# Per-weapon refire cooldown (literal engine attack_finished delay), from the
# canonical table — never re-typed from the QC.
COOLDOWN_S: dict[int, float] = {
    w: float(p["cooldown"]) for w, p in sorted(WEAPON_PHYSICS.items())
}

# DECISION FLOOR: the shortest gap at which BOTH outcomes (same weapon again /
# a different weapon) are definable, so the curve's domain is outcome-symmetric.
#   * ordinary weapon -- its cooldown. The engine's attack_finished is global,
#     and switching costs nothing extra (W_ChangeWeapon sets no cooldown), so
#     both outcomes share the same mechanical floor.
#   * continuous weapon -- (gap_ticks + 1) ticks. Firing the SAME weapon again
#     sooner than that is a hold-train continuation by definition, not a new
#     decision, so below this floor only a switch can be observed and
#     P(switch) = 1 by construction. Those pairs are not silently dropped:
#     they ARE the mid-train switch statistic (``mid_train_switch``).
DECISION_FLOOR_S: dict[int, float] = {
    w: ((ONSET_GAP_TICKS[w] + 1) / HZ if w in CONTINUOUS else cd)
    for w, cd in COOLDOWN_S.items()
}

# Weapon-mode families (project_weapon_mode_taxonomy). FAMILY/FAMILY_ORDER
# drive the diagnostic transition matrix (axe is not a family and is excluded
# there); the curve fit needs every pair to carry an intercept, so it uses
# FIT_FAMILY/FIT_FAMILY_ORDER, which adds axe as its own.
FAMILY = {2: "sg_fam", 3: "sg_fam", 4: "ng_fam", 5: "ng_fam",
          6: "gl", 7: "rl", 8: "lg"}
FAMILY_ORDER = ("sg_fam", "ng_fam", "gl", "rl", "lg")
FIT_FAMILY = {1: "axe", **FAMILY}
FIT_FAMILY_ORDER = ("axe", *FAMILY_ORDER)

# A family under this many pairs is not fitted at all — it is reported
# unscoreable and its mass is excluded from the coverage the fit's objective
# reports (replaces the old per-bin min-mass zeroing, which could silently
# change which bins a candidate was scored on). Within a fitted family the two
# parts are reported independently: a family can be measurable at x = 0 and
# unmeasurable on the continuum (a bot that ALWAYS takes the earliest shot is a
# real instance), and the objective then covers only the mass it can score
# rather than guessing.
MIN_FAMILY_PAIRS = 100
MIN_PART_PAIRS = 10          # per-part floor (atom cell / x>0 cell)
# Weakly-informative smoothing so a family with zero observed switches (a real
# and expected measurement on the bot side) yields a finite intercept instead
# of separating to -inf: two pseudo-pairs per family, one switch one not, at
# that family's mean covariate, each weighted PSEUDO_W. Vanishes against
# >=MIN_FAMILY_PAIRS real pairs; keeps every bootstrap resample finite.
PSEUDO_W = 0.5
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 17
# Covariate values the fitted curve is RENDERED at in the artifact (reporting
# only — the fit itself is binless and never evaluates on a grid).
REPORT_X = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
_SIG = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))  # noqa: E731

_OBS_KEYS = ("entity_recency", "entity_player_id", "health", "self_items",
             "ammo_shells", "ammo_nails", "ammo_rockets", "ammo_cells")


# ---------------------------------------------------------------------------
# Decision events (hold-train onsets) and encounter-scoped pairing
# ---------------------------------------------------------------------------
def decision_events(idx: np.ndarray, wlab: np.ndarray, *,
                    gap_ticks: dict[int, int] | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Discharge events -> DECISION events (hold-train onsets), time-ordered.

    ``idx`` (frame ticks) / ``wlab`` (impulse 1..8) are one episode's
    discharges. Applies ``qnn.ppo.crest_reward.OnsetGate``'s rule per weapon:
    a continuous-weapon discharge is an onset iff more than ``gap_ticks[w]``
    ticks elapsed since that SAME weapon's previous discharge (the gap is
    measured to the previous discharge, not the previous onset — a held
    trigger never re-onsets however long it runs). Returns
    ``(onset_ticks, onset_weapons)``.
    """
    gt = ONSET_GAP_TICKS if gap_ticks is None else gap_ticks
    idx = np.asarray(idx).reshape(-1)
    wlab = np.asarray(wlab).reshape(-1)
    if idx.size == 0:
        return idx.astype(np.int64), wlab.astype(np.int8)
    on_t: list[np.ndarray] = []
    on_w: list[np.ndarray] = []
    for w in np.unique(wlab):
        t = idx[wlab == w]
        if int(w) in CONTINUOUS:
            is_on = np.concatenate([[True], np.diff(t) > gt[int(w)]])
        else:
            is_on = np.ones(t.size, dtype=bool)
        on_t.append(t[is_on])
        on_w.append(np.full(int(is_on.sum()), int(w), dtype=np.int8))
    t_all = np.concatenate(on_t)
    w_all = np.concatenate(on_w)
    order = np.argsort(t_all, kind="stable")
    return t_all[order].astype(np.int64), w_all[order]


def _last_same_weapon_tick(idx: np.ndarray, wlab: np.ndarray,
                           onset_t: np.ndarray, onset_w: np.ndarray,
                           ) -> np.ndarray:
    """Per pair k: the last tick weapon ``w_k`` actually DISCHARGED before the
    next decision — the tick that set the cooldown clock the next decision had
    to wait out. For an ordinary weapon this is ``onset_t[k]`` itself; for a
    continuous weapon it is the last discharge of its hold train before the
    next decision (so a weapon swapped in MID-BURST is measured from the burst
    discharge that preceded it, never from the burst's onset — which would
    otherwise produce a negative gap)."""
    by_w = {int(w): idx[wlab == w] for w in np.unique(wlab)}
    out = np.empty(max(onset_t.size - 1, 0), dtype=np.int64)
    for k in range(onset_t.size - 1):
        t = by_w[int(onset_w[k])]
        j = int(np.searchsorted(t, onset_t[k + 1])) - 1
        out[k] = t[j]        # >= onset_t[k] by construction, so gap >= 1 tick
    return out


def covariate(gap_frames: np.ndarray, wk: np.ndarray, hz: float = HZ,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Gap (frames) + deciding weapon -> ``(x, in_domain)``.

    ``x`` = refire opportunities SKIPPED past the weapon's first decidable
    opportunity: ``(gap_s - DECISION_FLOOR_S[w]) / COOLDOWN_S[w]``. x = 0 means
    "took the earliest shot available"; x = 3 means "sat out three of this
    weapon's refire cycles". Dimensionless and comparable across weapons, which
    raw seconds is not. ``in_domain`` is False below the decision floor (a
    continuous weapon's mid-train swap, or — on a stream with sub-cooldown
    discharge pairs — a data anomaly); those pairs are reported separately,
    never folded into the curve."""
    gap_s = np.asarray(gap_frames, dtype=np.float64) / float(hz)
    wk = np.asarray(wk).reshape(-1)
    cd = np.array([COOLDOWN_S[int(w)] for w in wk], dtype=np.float64)
    fl = np.array([DECISION_FLOOR_S[int(w)] for w in wk], dtype=np.float64)
    in_domain = gap_s >= fl - 1e-9
    return np.maximum((gap_s - fl) / cd, 0.0), in_domain


def pairs_from_decisions(idx: np.ndarray, wlab: np.ndarray,
                         onset_t: np.ndarray, onset_w: np.ndarray,
                         streams: dict[str, np.ndarray] | None,
                         hz: float = HZ) -> dict[str, np.ndarray] | None:
    """Consecutive DECISION pairs inside ONE encounter -> a pair record, or
    None when the encounter holds fewer than two decisions.

    Record: ``gap_frames`` (onset_k+1 - last discharge of w_k), ``switch``,
    ``forced``, ``wk``/``wk1``, ``x``, ``in_domain``, ``feas_wk`` — all (n,).
    ``feas_wk`` is the weapon-feasibility MASK at the deciding tick (the MENU
    the decision chose from). Without it every rate is a raw rate: a bot whose
    arena offers only RL|LG (mask 192, the h2h box arena) can only ever switch
    RL->LG, while a human holding six weapons has six ways to leave RL, so the
    two switch rates are not the same quantity (feedback_operative_filter_all_
    comparisons). ``idx``/
    ``wlab`` are the WHOLE episode's discharges (needed for the train-end
    lookup and indexed absolutely, exactly as ``_invalidated_pairs`` indexes
    ``streams``); ``onset_t``/``onset_w`` are this encounter's decisions."""
    if onset_t.size < 2:
        return None
    end_t = _last_same_weapon_tick(idx, wlab, onset_t, onset_w)
    gap_frames = (onset_t[1:] - end_t).astype(np.float64)
    wk, wk1 = onset_w[:-1], onset_w[1:]
    inv = _invalidated_pairs(onset_t, streams)
    x, in_domain = covariate(gap_frames, wk, hz)
    if streams is None:
        feas_wk = np.zeros(wk.size, dtype=np.int64)
    else:
        feas_wk = np.asarray(streams["feas"]).reshape(-1)[onset_t[:-1]].astype(np.int64)
    return {
        "gap_frames": gap_frames,
        "switch": (wk1 != wk),
        "forced": (np.zeros(wk.size, dtype=bool) if inv is None
                   else np.asarray(inv, dtype=bool)),
        "wk": wk.astype(np.int8),
        "wk1": wk1.astype(np.int8),
        "x": x,
        "in_domain": in_domain,
        "feas_wk": feas_wk,
    }


def episode_encounter_pairs(
    cnt: np.ndarray, tp: np.ndarray, entity_player_id: np.ndarray,
    entity_recency: np.ndarray, attack_class: np.ndarray,
    feas: np.ndarray, health: np.ndarray, hz: float,
) -> list[dict[str, np.ndarray]]:
    """One episode -> a list of per-encounter decision-pair records.

    The onset gate runs over the WHOLE episode's discharges before the spans
    are cut, so an encounter boundary landing mid-hold-train cannot manufacture
    a spurious decision; the pairing itself never crosses a boundary."""
    pid_seq, rec_seq = pid_recency_from_tokens(cnt, tp, entity_player_id, entity_recency)
    spans = corpus_pid_encounter_spans(pid_seq, rec_seq, hz)
    keep = (1.0 - np.asarray(tp)[:, 0]) != 0.0
    idx, wlab = _preference_events(attack_class, keep)
    onset_t, onset_w = decision_events(idx, wlab)
    streams = {"feas": feas, "health": health}
    out: list[dict[str, np.ndarray]] = []
    for s, e in spans:
        m = (onset_t >= s) & (onset_t < e)
        rec = pairs_from_decisions(idx, wlab, onset_t[m], onset_w[m], streams, hz)
        if rec is not None:
            out.append(rec)
    return out


def _worker(args: tuple) -> tuple[list[dict[str, np.ndarray]], dict[int, list[int]]]:
    sh, dd = args
    recs: list[dict[str, np.ndarray]] = []
    ev: dict[int, list[int]] = {}
    for _ei, _dmi, fsl, esl, arr in A.iter_shard_episodes(
            sh, dd, obs=_OBS_KEYS, acts=("attack", "target_probs")):
        cnt = np.asarray(arr["entity_count"][fsl], dtype=np.int64)
        tp = np.asarray(arr["target_probs"][fsl], dtype=np.float32)
        attack_class = np.asarray(arr["attack"][fsl], dtype=np.int64).reshape(-1)
        feas = en.weapon_feasibility_bits(
            np.asarray(arr["self_items"][fsl]), np.asarray(arr["ammo_shells"][fsl]),
            np.asarray(arr["ammo_nails"][fsl]), np.asarray(arr["ammo_rockets"][fsl]),
            np.asarray(arr["ammo_cells"][fsl]))
        health = np.asarray(arr["health"][fsl], dtype=np.int64).reshape(-1)
        recs.extend(episode_encounter_pairs(
            cnt, tp, arr["entity_player_id"][esl], arr["entity_recency"][esl],
            attack_class, feas, health, HZ))
        keep = (1.0 - tp[:, 0]) != 0.0
        di, dw = _preference_events(attack_class, keep)
        oi, ow = decision_events(di, dw)
        for w in range(1, 9):
            e = ev.setdefault(w, [0, 0])
            e[0] += int((dw == w).sum())
            e[1] += int((ow == w).sum())
    return recs, ev


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def _counted(encounters: list[dict[str, np.ndarray]], key: str,
             *, domain_only: bool = True) -> np.ndarray:
    """Pool one field over non-forced pairs (optionally curve-domain only)."""
    parts = []
    for e in encounters:
        m = ~e["forced"]
        if domain_only:
            m = m & e["in_domain"]
        parts.append(np.asarray(e[key])[m])
    return np.concatenate(parts) if parts else np.empty(0)


def gap_distribution(encounters: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Quantiles + histogram of the decision-to-decision gap (seconds) over
    counted (non-forced, in-domain) pairs."""
    gaps = _counted(encounters, "gap_frames") / HZ
    if gaps.size == 0:
        return {"n": 0}
    edges = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])
    hist, _ = np.histogram(gaps, bins=np.append(edges, np.inf))
    return {
        "n": int(gaps.size),
        "quantiles_s": {f"p{q}": round(float(np.percentile(gaps, q)), 4)
                        for q in (5, 10, 25, 50, 75, 90, 95, 99)},
        "mean_s": round(float(gaps.mean()), 4),
        "histogram": {"edges_s": edges.tolist() + ["inf"], "counts": hist.tolist()},
    }


def covariate_distribution(encounters: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Quantiles of the covariate x (refire opportunities skipped) plus the
    mass of its atom at x = 0 ("took the earliest shot available") — the atom
    is most of the distribution, which is exactly why a smoother with a
    bandwidth would have been the wrong instrument here."""
    x = _counted(encounters, "x")
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "quantiles": {f"p{q}": round(float(np.percentile(x, q)), 4)
                      for q in (5, 10, 25, 50, 75, 90, 95, 99)},
        "mean": round(float(x.mean()), 4),
        "atom_at_zero_frac": round(float(np.mean(x <= 1e-9)), 4),
        "definition": ("x = (gap_s - DECISION_FLOOR_S[w_k]) / COOLDOWN_S[w_k]"
                       " — refire opportunities skipped past the first"
                       " decidable one; 0 = earliest shot available"),
    }


# ---------------------------------------------------------------------------
# Binless curve: per-family intercept + shared slope in log1p(x)
# ---------------------------------------------------------------------------
def _fit_irls(X: np.ndarray, y: np.ndarray, w: np.ndarray, *,
              ridge: float = 1e-6, iters: int = 100, tol: float = 1e-9,
              ) -> np.ndarray | None:
    """Weighted logistic IRLS. Returns coefficients, or None if it fails to
    produce a finite fit (callers decide: fail loud, or drop the resample)."""
    b = np.zeros(X.shape[1], dtype=np.float64)
    eye = np.eye(X.shape[1]) * ridge
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        np.clip(p, 1e-12, 1.0 - 1e-12, out=p)
        wp = w * p * (1.0 - p)
        H = X.T @ (X * wp[:, None]) + eye
        g = X.T @ (w * (y - p))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if not np.all(np.isfinite(b)):
            return None
        if float(np.max(np.abs(step))) < tol:
            break
    return b


def _design(x: np.ndarray, fam_idx: np.ndarray, n_fam: int) -> np.ndarray:
    """[family one-hots | log1p(x)] — the shared-slope design matrix."""
    X = np.zeros((x.size, n_fam + 1), dtype=np.float64)
    X[np.arange(x.size), fam_idx] = 1.0
    X[:, -1] = np.log1p(x)
    return X


def curve_predict(fit: dict[str, Any], x: np.ndarray | float,
                  family: str) -> np.ndarray:
    """P(switch | x, family) under a fitted curve — the only evaluator, used by
    both the artifact's rendering and the decode-fit objective. Two-part:
    ``x <= 0`` reads the family's ATOM rate, ``x > 0`` the logistic. Raises
    KeyError for a part this family was not fitted on (never guesses)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    zero = x <= 0.0
    if zero.any():
        atom = (fit.get("atom") or {}).get(family)
        if atom is None or atom.get("p") is None:
            raise KeyError(
                f"family {family!r} has no fitted x=0 atom; fitted: "
                f"{sorted(k for k, v in (fit.get('atom') or {}).items() if v.get('p') is not None)}")
        out[zero] = float(atom["p"])
    if (~zero).any():
        a = (fit.get("intercepts") or {}).get(family)
        if a is None:
            raise KeyError(f"family {family!r} has no fitted x>0 curve; "
                           f"fitted: {sorted(fit.get('intercepts') or {})}")
        out[~zero] = _SIG(float(a) + float(fit["slope"]) * np.log1p(x[~zero]))
    return out


def _smoothed_rate(n_switch: float, n: float) -> float | None:
    """Pseudo-count-smoothed switch rate — the same weakly-informative
    smoothing the logistic part uses, so a cell with zero observed switches
    (a real and expected measurement on the bot side) is a small finite number
    instead of an exact 0.0 with a degenerate CI."""
    if n <= 0:
        return None
    return float((n_switch + PSEUDO_W) / (n + 2.0 * PSEUDO_W))


def fit_curve(encounters: list[dict[str, np.ndarray]], *,
              n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
              min_family_pairs: int = MIN_FAMILY_PAIRS,
              min_part_pairs: int = MIN_PART_PAIRS,
              ) -> dict[str, Any]:
    """Fit the two-part transition curve

        P(switch | x = 0)  = atom[family]                    (saturated cell)
        P(switch | x > 0)  : logit = intercept[family] + slope * log1p(x)

    by pair-level (pseudo-count-smoothed) maximum likelihood, with the
    ENCOUNTER cluster bootstrap resampling the whole fit — pairs inside one
    encounter share a menu and a combat context and are not independent draws.
    See the module docstring for why the atom is free rather than a point on
    the smooth curve.

    No bins, no edges, no per-cell weight fudge: a family is either fitted
    (>= ``min_family_pairs`` counted pairs) or reported unscoreable, each PART
    needs ``min_part_pairs`` of its own, and the objective downstream covers
    only the human mass it can actually score.

    Returns the fitted params + CIs, ``p_at_zero`` per family (the chatter
    statistic the placement gate adjudicates), the rendered curve at
    ``REPORT_X``, per-family observed-vs-fitted diagnostics, and the per-family
    covariate quantile grid the fit's objective integrates against."""
    n_enc = len(encounters)
    x_all, y_all, fam_all, enc_all = [], [], [], []
    for i, e in enumerate(encounters):
        m = (~e["forced"]) & e["in_domain"]
        if not m.any():
            continue
        x_all.append(e["x"][m])
        y_all.append(e["switch"][m].astype(np.float64))
        fam_all.append(np.array([FIT_FAMILY.get(int(w), "axe")
                                 for w in e["wk"][m]], dtype=object))
        enc_all.append(np.full(int(m.sum()), i, dtype=np.int64))
    if not x_all:
        return {"status": "NO-DATA", "n_pairs": 0, "families": {},
                "atom": {}, "intercepts": {}, "slope": None}
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    fam = np.concatenate(fam_all)
    enc = np.concatenate(enc_all)
    at_zero = x <= 0.0

    counts = {f: int((fam == f).sum()) for f in FIT_FAMILY_ORDER}
    fams = [f for f in FIT_FAMILY_ORDER if counts[f] >= min_family_pairs]
    fam_report: dict[str, Any] = {}
    for f in FIT_FAMILY_ORDER:
        if not counts[f]:
            continue
        mf = fam == f
        fam_report[f] = {
            "n_pairs": counts[f],
            "fitted": f in fams,
            "n_at_zero": int((mf & at_zero).sum()),
            "n_positive_x": int((mf & ~at_zero).sum()),
            "observed_switch_rate": round(float(y[mf].mean()), 4),
            "observed_at_zero": (round(float(y[mf & at_zero].mean()), 4)
                                 if (mf & at_zero).any() else None),
        }
    if not fams:
        return {"status": "NO-SCOREABLE-FAMILY", "n_pairs": int(x.size),
                "families": fam_report, "atom": {}, "intercepts": {},
                "slope": None, "min_family_pairs": min_family_pairs}

    keep = np.isin(fam, fams)
    x, y, fam, enc, at_zero = x[keep], y[keep], fam[keep], enc[keep], at_zero[keep]
    fi = {f: i for i, f in enumerate(fams)}
    fam_idx = np.array([fi[f] for f in fam], dtype=np.int64)

    # ── part 1: the atom at x = 0 (per family, saturated) ───────────────────
    atom_n = np.array([int(((fam_idx == i) & at_zero).sum()) for i in range(len(fams))])
    atom_k = np.array([float(y[(fam_idx == i) & at_zero].sum()) for i in range(len(fams))])
    atom_ok = atom_n >= min_part_pairs

    # ── part 2: the logistic on x > 0 ───────────────────────────────────────
    pos = ~at_zero
    pos_n = np.array([int(((fam_idx == i) & pos).sum()) for i in range(len(fams))])
    curve_fams = [f for f in fams if pos_n[fi[f]] >= min_part_pairs]
    point_slope: float | None = None
    point_int: dict[str, float] = {}
    ci_fams = list(curve_fams)
    if curve_fams:
        cfi = {f: i for i, f in enumerate(curve_fams)}
        sel = pos & np.isin(fam, curve_fams)
        xc, yc = x[sel], y[sel]
        cidx = np.array([cfi[f] for f in fam[sel]], dtype=np.int64)
        Xc = _design(xc, cidx, len(curve_fams))
        # Pseudo-pairs (one switch, one not, per family, at that family's mean x)
        px, pfi_, py = [], [], []
        for f, i in cfi.items():
            mx = float(xc[cidx == i].mean())
            px += [mx, mx]; pfi_ += [i, i]; py += [1.0, 0.0]
        Xp = _design(np.array(px), np.array(pfi_, dtype=np.int64), len(curve_fams))
        yp = np.array(py)
        wp = np.full(yp.size, PSEUDO_W)
        b = _fit_irls(np.vstack([Xc, Xp]), np.concatenate([yc, yp]),
                      np.concatenate([np.ones(yc.size), wp]))
        if b is None:
            raise RuntimeError(
                "curve IRLS produced a non-finite fit on the point sample — "
                "refusing to emit a curve (n_pairs=%d, families=%s)"
                % (xc.size, curve_fams))
        point_int = {f: float(b[i]) for i, f in enumerate(curve_fams)}
        point_slope = float(b[-1])
        cenc = enc[sel]
    else:
        cfi, Xc, yc, cidx, Xp, yp, wp, cenc = {}, None, None, None, None, None, None, None

    # ── encounter cluster bootstrap over BOTH parts ─────────────────────────
    enc_ids = np.unique(enc)
    rng = np.random.default_rng(seed)
    atom_rows_by_enc = {int(i): np.flatnonzero((enc == i) & at_zero) for i in enc_ids}
    curve_rows_by_enc = ({int(i): np.flatnonzero(cenc == i) for i in np.unique(cenc)}
                         if cenc is not None else {})
    boot_atom = np.full((n_bootstrap, len(fams)), np.nan)
    boot_curve = (np.full((n_bootstrap, len(curve_fams) + 1), np.nan)
                  if curve_fams else np.empty((0, 0)))
    n_failed = 0
    for bi in range(n_bootstrap):
        pick = enc_ids[rng.integers(0, enc_ids.size, enc_ids.size)]
        rows = np.concatenate([atom_rows_by_enc[int(j)] for j in pick]) \
            if enc_ids.size else np.empty(0, dtype=np.int64)
        if rows.size:
            bfi, byy = fam_idx[rows], y[rows]
            for i in range(len(fams)):
                m = bfi == i
                if m.any():
                    boot_atom[bi, i] = _smoothed_rate(float(byy[m].sum()),
                                                      float(m.sum()))
        if curve_fams:
            crows = np.concatenate([curve_rows_by_enc[int(j)] for j in pick
                                    if int(j) in curve_rows_by_enc]) \
                if curve_rows_by_enc else np.empty(0, dtype=np.int64)
            if crows.size == 0:
                n_failed += 1
                continue
            bb = _fit_irls(np.vstack([Xc[crows], Xp]),
                           np.concatenate([yc[crows], yp]),
                           np.concatenate([np.ones(crows.size), wp]))
            if bb is None:
                n_failed += 1
                continue
            boot_curve[bi] = bb

    def _ci(v: np.ndarray) -> list[float] | None:
        v = v[np.isfinite(v)]
        if v.size < max(10, n_bootstrap // 10):
            return None
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]

    atom: dict[str, Any] = {}
    p_at_zero: dict[str, Any] = {}
    for f, i in fi.items():
        p = _smoothed_rate(atom_k[i], atom_n[i]) if atom_ok[i] else None
        row = {"n_pairs": int(atom_n[i]), "n_switch": int(atom_k[i]),
               "p": (round(p, 4) if p is not None else None),
               "ci95": (_ci(boot_atom[:, i]) if p is not None else None)}
        if p is None:
            row["note"] = (f"< {min_part_pairs} pairs at x=0 — atom not "
                           "measurable for this family")
        atom[f] = row
        if p is not None:
            p_at_zero[f] = {"p": row["p"], "ci95": row["ci95"]}

    intercept_ci = {f: (_ci(boot_curve[:, i]) if curve_fams else None)
                    for i, f in enumerate(ci_fams)}
    rendered = {}
    for f in fams:
        pts = []
        for xv in REPORT_X:
            if xv <= 0.0:
                if f not in p_at_zero:
                    continue
                pts.append({"x": xv, "p": p_at_zero[f]["p"],
                            "ci95": p_at_zero[f]["ci95"]})
                continue
            if f not in point_int:
                continue
            i = ci_fams.index(f)
            pts.append({"x": xv,
                        "p": round(float(curve_predict(
                            {"atom": atom, "intercepts": point_int,
                             "slope": point_slope}, xv, f)), 4),
                        "ci95": (_ci(_SIG(boot_curve[:, i]
                                          + boot_curve[:, -1] * np.log1p(xv)))
                                 if curve_fams else None)})
        rendered[f] = pts

    fit_obj = {"atom": atom, "intercepts": point_int, "slope": point_slope}
    for f, i in fi.items():
        m = (fam_idx == i) & pos
        fam_report[f]["fitted_at_zero"] = (p_at_zero[f]["p"]
                                           if f in p_at_zero else None)
        fam_report[f]["observed_positive_x_rate"] = (
            round(float(y[m].mean()), 4) if m.any() else None)
        fam_report[f]["fitted_positive_x_rate"] = (
            round(float(curve_predict(fit_obj, x[m], f).mean()), 4)
            if (m.any() and f in point_int) else None)

    # Integration grid: the objective downstream is
    #   mean over HUMAN pairs of (p_bot(x_i, f_i) - p_human(x_i, f_i))^2,
    # i.e. the curve gap integrated against the human's OWN covariate
    # distribution. Storing a per-family quantile grid (not 18k raw x values)
    # makes that integral reproducible from the artifact alone at ~1/200th the
    # size, and quantiles carry the atom at x=0 exactly
    # (feedback_cost_every_consumer).
    qs = np.linspace(0.0, 100.0, 101)
    grid = {f: {"n_pairs": int((fam_idx == fi[f]).sum()),
                "x_quantiles": [round(float(v), 5) for v in
                                np.percentile(x[fam_idx == fi[f]], qs)]}
            for f in fams}

    return {
        "status": "OK",
        "form": ("P(switch | x=0) = atom[family] (free); "
                 "P(switch | x>0): logit = intercept[family] + slope*log1p(x)"),
        "n_pairs": int(x.size),
        "n_encounters": n_enc,
        "families": fam_report,
        "atom": atom,
        "p_at_zero": p_at_zero,
        "intercepts": point_int,
        "intercepts_ci95": intercept_ci,
        "slope": point_slope,
        "slope_ci95": (_ci(boot_curve[:, -1]) if curve_fams else None),
        "rendered": {"x": list(REPORT_X), "by_family": rendered},
        "integration_grid": {"quantile_pct": qs.tolist(), "by_family": grid},
        "min_family_pairs": min_family_pairs,
        "min_part_pairs": min_part_pairs,
        "pseudo_pair_weight": PSEUDO_W,
        "bootstrap": {"n_resamples": n_bootstrap, "seed": seed,
                      "unit": "encounter", "n_failed": n_failed},
    }


# ---------------------------------------------------------------------------
# Mid-train switch rate (the chatter ruler) + diagnostics
# ---------------------------------------------------------------------------
def mid_train_switch(encounters: list[dict[str, np.ndarray]], *,
                     n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                     min_pairs: int = MIN_FAMILY_PAIRS) -> dict[str, Any]:
    """Per continuous weapon: the fraction of its decisions whose next
    discharge came BEFORE the decision floor — i.e. a different weapon fired
    inside the hold train. Every such pair is a switch by construction, so
    this is not part of the curve; it is the purest chatter signature there
    is, and a memoryless threshold shows up here first. Encounter cluster
    bootstrap, same unit as the curve."""
    n_enc = len(encounters)
    per_w: dict[int, np.ndarray] = {}
    for w in sorted(CONTINUOUS):
        num = np.zeros(n_enc, dtype=np.int64)
        den = np.zeros(n_enc, dtype=np.int64)
        for i, e in enumerate(encounters):
            m = (~e["forced"]) & (e["wk"] == w)
            if not m.any():
                continue
            den[i] = int(m.sum())
            num[i] = int((m & ~e["in_domain"]).sum())
        per_w[w] = np.stack([num, den])
    rng = np.random.default_rng(seed)
    picks = (rng.integers(0, n_enc, (n_bootstrap, n_enc)) if n_enc else None)
    out: dict[str, Any] = {}
    for w, nd in per_w.items():
        num, den = nd
        n_dec, n_mid = int(den.sum()), int(num.sum())
        row: dict[str, Any] = {"n_decisions": n_dec, "n_mid_train": n_mid,
                               "rate": (round(n_mid / n_dec, 4) if n_dec else None),
                               "low_n": n_dec < min_pairs}
        if picks is not None and n_dec:
            bn, bd = num[picks].sum(axis=1), den[picks].sum(axis=1)
            v = bn[bd > 0] / np.maximum(bd[bd > 0], 1)
            row["ci95"] = ([round(float(np.percentile(v, 2.5)), 4),
                            round(float(np.percentile(v, 97.5)), 4)]
                           if v.size >= max(10, n_bootstrap // 10) else None)
        else:
            row["ci95"] = None
        out[str(w)] = row
    return {
        "by_weapon_impulse": out,
        "criterion": ("fraction of a continuous weapon's decisions whose next "
                      "discharge fell below DECISION_FLOOR_S (inside the hold "
                      "train) — always a switch, hence excluded from the curve "
                      "domain and adjudicated on its own"),
        "bootstrap": {"n_resamples": n_bootstrap, "seed": seed,
                      "unit": "encounter"},
    }


def atom_by_menu(encounters: list[dict[str, np.ndarray]], *,
                 min_pairs: int = MIN_PART_PAIRS) -> dict[str, Any]:
    """P(switch | x = 0) sliced by the MENU the decision chose from, plus where
    those switches went.

    "Switched off RL" is not one quantity: with only RL and LG feasible the
    single alternative is LG, while a full arsenal offers five. Comparing a
    bot's atom against a human atom pooled over every arsenal it ever held is a
    raw-rate comparison (feedback_operative_filter_all_comparisons), so the
    baseline has to be sliceable by menu and the comparison drawn on the menu
    the bot actually faced. Rows are keyed ``"<family>|menu=<mask>"``; a row
    also reports ``n_feasible_weapons`` (how many the mask offers) and the
    destination-family composition of its switches."""
    rows: dict[str, Any] = {}
    for e in encounters:
        m = (~e["forced"]) & e["in_domain"] & (e["x"] <= 1e-9)
        if not m.any():
            continue
        for f_src, menu, sw, w1 in zip(
                [FIT_FAMILY.get(int(w), "axe") for w in e["wk"][m]],
                e["feas_wk"][m].tolist(), e["switch"][m].tolist(),
                e["wk1"][m].tolist()):
            r = rows.setdefault(f"{f_src}|menu={int(menu)}",
                                {"family": f_src, "menu_mask": int(menu),
                                 "n_feasible_weapons": int(bin(int(menu)).count("1")),
                                 "n_pairs": 0, "n_switch": 0, "to": {}})
            r["n_pairs"] += 1
            if sw:
                r["n_switch"] += 1
                d = FIT_FAMILY.get(int(w1), "axe")
                r["to"][d] = r["to"].get(d, 0) + 1
    out = {}
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["n_pairs"]):
        r["p_at_zero"] = round(r["n_switch"] / r["n_pairs"], 4)
        r["low_n"] = r["n_pairs"] < min_pairs
        r["to"] = dict(sorted(r["to"].items(), key=lambda kv: -kv[1]))
        out[k] = r
    return {"by_family_and_menu": out,
            "criterion": ("P(switch | x=0) conditioned on the feasibility MASK "
                          "at the deciding tick — the menu the decision chose "
                          "from. Compare a bot only against its OWN menu's row; "
                          "pooling over menus compares different question sets.")}


def event_stats(discharge_onset_counts: dict[int, list[int]]) -> dict[str, Any]:
    """Per weapon: raw discharges vs DECISIONS after the onset gate. The
    collapse ratio is the cross-side comparability check — human LG collapses
    at a different ratio than a bot's LG whenever the two streams resolve the
    0.1s think-chain differently, and that mismatch (not the binning) is what
    broke the first fit."""
    rows = {}
    for w in range(1, 9):
        nd, no = discharge_onset_counts.get(w, [0, 0])
        rows[str(w)] = {
            "n_discharges": int(nd), "n_decisions": int(no),
            "collapse_ratio": (round(nd / no, 3) if no else None),
            "continuous": w in CONTINUOUS,
            "cooldown_s": COOLDOWN_S[w], "decision_floor_s": DECISION_FLOOR_S[w],
        }
    return {"by_weapon_impulse": rows,
            "onset_gap_ticks": {str(k): v for k, v in sorted(ONSET_GAP_TICKS.items())}}


def family_transition_matrix(encounters: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Weapon-FAMILY transition matrix over DECISIONS, pooled over all gaps
    (diagnostic only). Axe pairs are excluded on either side."""
    n = len(FAMILY_ORDER)
    fam_idx = {name: i for i, name in enumerate(FAMILY_ORDER)}
    counts = np.zeros((n, n), dtype=np.int64)
    for e in encounters:
        kept = ~e["forced"]
        for a, b in zip(e["wk"][kept].tolist(), e["wk1"][kept].tolist()):
            fa, fb = FAMILY.get(a), FAMILY.get(b)
            if fa is None or fb is None:
                continue
            counts[fam_idx[fa], fam_idx[fb]] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sum > 0, counts / np.maximum(row_sum, 1), np.nan)
    return {
        "families": list(FAMILY_ORDER),
        "counts": counts.tolist(),
        "probs": [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                  for row in probs],
        "note": "diagnostic only, pooled over all x; axe excluded",
    }


def pair_accounting(encounters: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Every decision pair lands in exactly one bucket — no silent drops."""
    tot = fo = mid = sub = 0
    for e in encounters:
        n = e["wk"].size
        f = int(e["forced"].sum())
        kept = ~e["forced"]
        od = kept & ~e["in_domain"]
        cont = np.isin(e["wk"], list(CONTINUOUS))
        tot += n; fo += f
        mid += int((od & cont).sum())
        sub += int((od & ~cont).sum())
    return {
        "n_pairs_total": tot,
        "n_dropped_forced": fo,
        "n_mid_train": mid,
        "n_sub_cooldown_anomaly": sub,
        "n_curve_domain": tot - fo - mid - sub,
        "note": ("mid_train = below the decision floor on a continuous weapon "
                 "(scored by mid_train_switch); sub_cooldown_anomaly = the two "
                 "discharges are closer than the FIRST one's weapon cooldown, "
                 "which that weapon alone cannot produce — so the pair's "
                 "weapons and the engine's own cooldown clock disagree, and it "
                 "is quarantined rather than scored. The ordinary cause is "
                 "ENGINE-FORCED selection (feedback_weapon_switch_is_engine_"
                 "forced): W_WeaponFrame drops ImpulseCommands entirely while "
                 "the cooldown runs, and W_ChangeWeapon refuses an unowned or "
                 "dry weapon, so the weapon that fires can be the one already "
                 "held rather than the one the decision named. A large count "
                 "means requested-vs-forced needs decomposing for that sample "
                 "before its curve means anything."),
    }


# ---------------------------------------------------------------------------
# Corpus pass
# ---------------------------------------------------------------------------
def run(collect_dir: Path, splits: list[str], out_path: Path, n_workers: int,
        ) -> dict[str, Any]:
    """Compute the human within-encounter switch-DECISION curve for a collect
    and write it to ``out_path``. Signature matches the other ``qnn.human``
    corpus creators: (collect_dir, splits, out_path, n_workers)."""
    collect_dir = Path(collect_dir)
    encounters: list[dict[str, np.ndarray]] = []
    ev_counts: dict[int, list[int]] = {}
    used_split = None
    for split in splits:
        dd = collect_dir / f"precomputed_{split}"
        man = dd / "manifest.json"
        if not man.exists():
            continue
        used_split = split
        tasks = [(sh, str(dd)) for sh in json.loads(man.read_text())["shards"]]
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, (recs, ev) in enumerate(pool.imap_unordered(_worker, tasks)):
                encounters.extend(recs)
                for w, (nd, no) in ev.items():
                    row = ev_counts.setdefault(w, [0, 0])
                    row[0] += nd; row[1] += no
                print(f"  [{split}] {i+1}/{len(tasks)} shards done", flush=True)

    curve = fit_curve(encounters)
    out = {
        "_meta": {
            "schema": SCHEMA,
            "contract": (
                "within-encounter consecutive-DECISION transition "
                "P(w_k+1 != w_k) as a smooth function of x = refire "
                "opportunities skipped past the deciding weapon's first "
                "decidable opportunity. Decision = hold-train ONSET "
                "(qnn.ppo.crest_reward's OnsetGate rule on NG/SNG/LG; every "
                "other discharge is its own trigger pull). Encounter = "
                "qnn.human.encounters.corpus_pid_encounter_spans (v6 "
                "same-opponent pid-run slicer). Discharge events = "
                "qnn.eval.humanlikeness.rc._preference_events (keep-gated, "
                "op-filter). Forced (engine-caused) pairs excluded via "
                "qnn.eval.humanlikeness.rc._invalidated_pairs over "
                "(onset_k, onset_k+1] and DROPPED, counted separately. NO "
                "BINS: the curve is per-family-intercept + shared-slope "
                "logistic ML with an encounter cluster bootstrap."),
            "collect_dir": str(collect_dir),
            "split": used_split,
            "hz": HZ,
            "cooldown_s": COOLDOWN_S,
            "decision_floor_s": DECISION_FLOOR_S,
            "n_encounters_with_pairs": len(encounters),
        },
        "event_stats": event_stats(ev_counts),
        "pair_accounting": pair_accounting(encounters),
        "gap_distribution": gap_distribution(encounters),
        "covariate_distribution": covariate_distribution(encounters),
        "curve": curve,
        "atom_by_menu": atom_by_menu(encounters),
        "mid_train_switch": mid_train_switch(encounters),
        "family_transition_matrix": family_transition_matrix(encounters),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    acct = out["pair_accounting"]
    print(f"\nn_encounters_with_pairs={len(encounters)}  "
          f"n_pairs_total={acct['n_pairs_total']}  "
          f"curve_domain={acct['n_curve_domain']}  "
          f"forced={acct['n_dropped_forced']}  mid_train={acct['n_mid_train']}")
    if curve.get("status") == "OK":
        print(f"slope={curve['slope']:.4f} ci={curve['slope_ci95']}  "
              f"P(switch|x=0) per family: "
              + "  ".join(f"{f}={v['p']}" for f, v in curve["p_at_zero"].items()))
    print(f"Written -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <collect>/human_baseline/"
                         "_weapon_switch_transition_curve.json)")
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_weapon_switch_transition_curve.json"
    run(args.collect_dir, [args.split], out, args.workers)


if __name__ == "__main__":
    main()
