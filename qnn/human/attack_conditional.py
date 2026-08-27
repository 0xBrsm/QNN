"""Per-collect HUMAN reference for the ATTACK-GATE RESPEC (Brian 2026-08-15,
agents/plans/a26-superiority-decomposition.md E12/E13; research/human-band.md).

THE PROBLEM this replaces. decode-fit's style gate used to score the bot's
MARGINAL fire rate (fires/s while engaged) against the human marginal —
op_attack in ``qnn.decode_fit.gates``. That ruler is a skill x style
PRODUCT: a model with human-like conditional trigger behavior (fires more
often when well-aligned, same as a human) but ELITE aim availability (well
aligned far more often than a human) is CORRECTLY expected to exceed the
human marginal rate arithmetically — E12/E13 caught exactly this case
(a28p90floor refused on op_attack while decisively beating a26rc1b and every
human-band channel it was scored against). The marginal conflates two
different things: how humanly the trigger is pulled GIVEN aim quality
(style), and how good the aim is (skill, which decode-fit is explicitly
never supposed to cap — feedback_aim_above_human_ceiling).

THE FIX. Score the CONDITIONAL shape instead: P(discharge | total aim-error
bin, op-ready) — the same instrument as
``scripts/analysis/human_fire_tolerance_by_axis.py`` (E7), collapsed to the
TOTAL (unsigned) error axis to match the new ``aim_err_deg`` eval stream
(qnn.eval.run._log_streams / qnn.eval.h2h._log_lane_tick) instead of E7's
signed per-axis conditioning. Plus a HOLD-TEXTURE reference (discharge
run-length / inter-burst gap distributions, engaged-conditioned) so bursty
vs. metronomic trigger shape is also checked — a conditional curve alone
cannot tell a human-shaped burst from a machine-gun hold at the same P(fire).

Both references are corpus-derived and model-agnostic (this module depends
only on the raw collect + qnn.eval.aim_kernel / qnn.eval.humanlikeness.core,
same discipline as every other ``qnn.human`` baseline creator), cached
next to the human-band bank under the collect's ``human_baseline/`` dir.
Content key = the specific collect_dir (this artifact's home) + ``split`` +
``REF_VERSION`` (checked inside the cached doc, band_bank's own staleness
convention — no per-collect content hash exists at this layer either,
consistent with every sibling ``qnn.human`` creator, which cache by file
existence inside a specific collect_dir and never carry a corpus
fingerprint in the filename).

BETWEEN-DEMO SPREAD (the tolerance source; v2, 2026-08-15 respec fix).
**v1 used a SPLIT-HALF-OF-THE-POOL deviation (random 50/50 demo partitions,
each half's own pooled curve vs the other half's) — an AXIOM-3 VIOLATION**:
that statistic is the sampling error of an ESTIMATOR of the pooled mean, so
it shrinks as ~1/sqrt(corpus size) the more of the corpus is scanned (a
24-shard smoke build measured 0.09-0.48; the full-corpus lazy build measured
0.028-0.19 for the SAME quantity) — a significance bar in disguise, exactly
the pathology human-band.md's axioms exist to prevent (an estimator that
tightens with N is a p-value, not an effect size; a large enough corpus
would fail every bot on noise alone). The fix: an EFFECT-SIZE scale that
does NOT shrink with corpus size — the intrinsic BETWEEN-DEMO variability of
individual human sessions around the pooled reference. For each demo (with
enough of its OWN mass, ``PER_DEMO_MIN_TICKS``/``_quantiles``' n>=5 floor):
build that demo's own normalized curve / hold quantiles, and its deviation
from the POOLED LEAVE-THAT-DEMO-OUT reference (every other demo, so no demo
is compared against a reference partly built from itself). The ``p95`` of
that per-demo-deviation distribution, taken over ALL demos (not resampled
halves), is a property of how much individual human sessions vary from each
other — it does not tighten as more demos are added, it CONVERGES to the
population's true between-session spread. Demo-level per-episode raw
counts/lists are cached during the one corpus scan so this is a cheap
re-aggregation, never a re-scan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qnn.eval import aim_kernel as A
from qnn.eval.humanlikeness.core import dwell_times
from qnn.weapons import COOLDOWN_SEC

# v3 (2026-08-15, coordinator iteration 3): hold-texture gap fix — see
# collapse_to_effective_events / hold_texture_samples below. Bumped so every
# cached v1/v2 reference invalidates.
# v2 (2026-08-15): between-demo tolerance derivation (v1's split-half-of-pool
# was an Axiom-3 violation — see module docstring) + the MIN_BINS_FOR_VERDICT
# guard (a 1-populated-bin curve comparison must never gate a verdict).
REF_VERSION = 3
TOKEN_ACTOR = 1
MODALITY_SIGHT = 0
# The pinned human corpus's tick rate (band_bank.HUMAN_HZ's convention) —
# used for the human-side gap-excess/cooldown-tick math below. The BOT side
# uses its OWN wave's tick_hz (qnn.decode_fit.gates reads it from the npz).
HZ = 20.0

IMPULSE_NAME = {1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
                6: "GL", 7: "RL", 8: "LG"}
NAME_TO_IMPULSE = {v: k for k, v in IMPULSE_NAME.items()}

# The 5 families the respec's gate arms score (task spec): SG+SSG and
# NG+SNG pool same-physics members (qnn.decode_fit.context convention); GL
# gets its own family here (it is NOT in qnn.decode_fit.context's
# CALIBRATION_FAMILIES — that grouping only spans the intercept-valid
# weapons — but the trigger/hold reference is defined for every
# attack-with weapon so GL's ground-control cadence has a reference too).
FAMILIES = {
    "SG+SSG": ("SG", "SSG"),
    "NG+SNG": ("NG", "SNG"),
    "GL": ("GL",),
    "RL": ("RL",),
    "LG": ("LG",),
}
FAMILY_IMPULSES = {fam: tuple(NAME_TO_IMPULSE[m] for m in members)
                   for fam, members in FAMILIES.items()}
POOLED = "ALL"

# Total (unsigned) aim-error bin edges, degrees — the angle between the
# crosshair (+x, view basis) and the most-aligned in-SIGHT actor
# (arctan2(hypot(ry, rz), rx), always >= 0). Matches the new
# aim_err_deg eval stream so the bot side bins identically.
EDGES = np.arange(0.0, 40.0 + 1e-9, 2.5)
NBINS = EDGES.shape[0] - 1
BIN_CENTERS = ((EDGES[:-1] + EDGES[1:]) / 2.0).tolist()
# "each curve divided by its own error<5deg bin" (task spec): the
# near-perfect-alignment normalizer.
NORM_BIN_MAX_DEG = 5.0
_NORM_BINS = int(np.searchsorted(EDGES, NORM_BIN_MAX_DEG, side="right")) - 1
_NORM_BINS = max(_NORM_BINS, 1)

MIN_ENGAGED_FRAMES_EP = 30     # episode admit floor (matches qnn.human.op_attack)
# thin-data floor (per family; else pooled) — shared by both new gate arms.
THIN_DATA_MIN_TICKS = 200
# per-DEMO floor for the between-demo tolerance derivation: much lower than
# THIN_DATA_MIN_TICKS (that one gates a SCORED VERDICT; this one only decides
# whether one demo's own curve is stable enough to contribute one sample to
# the spread distribution — noise here widens the tolerance rather than
# silently failing a bot, so it can afford to be permissive).
PER_DEMO_MIN_TICKS = 30
# a curve comparison spanning fewer than this many mutually-populated bins
# cannot characterize a SHAPE (2026-08-15 bin-collapse fix, defect #2: the
# alignment-law weapons' bot-side tracking can be steady enough to
# concentrate op-ready mass into very few error bins) — unscored, never
# gated, regardless of how much raw tick MASS backs those few bins.
MIN_BINS_FOR_VERDICT = 4


# ── per-episode computation ────────────────────────────────────────────────

def _episode_arrays(cnt: np.ndarray, typ: np.ndarray, mod: np.ndarray,
                    rel: np.ndarray, af: np.ndarray, attack: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
    """One episode -> per-tick (err_deg, ready_and_engaged_and_finite,
    fire, weapon_context_impulse, engaged).

    ``rel`` is wire-quantized int16 game units — cast to float64 before any
    arithmetic (an int16 hypot/square would silently overflow/truncate).
    Target selection = the MOST-ALIGNED in-SIGHT actor this frame (smallest
    total error), the E7 / human_fire_tolerance_by_axis.py convention.
    """
    T = len(cnt)
    rel = np.asarray(rel, dtype=np.float64)
    typ = np.asarray(typ)
    mod = np.asarray(mod)
    rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
    total = np.degrees(np.arctan2(np.hypot(ry, rz), rx))
    elig = (typ == TOKEN_ACTOR) & (mod == MODALITY_SIGHT)
    row = np.repeat(np.arange(T, dtype=np.int64), np.asarray(cnt, dtype=np.int64))
    key = np.where(elig, total, np.inf)
    best = np.full(T, np.inf)
    np.minimum.at(best, row, key)
    engaged = np.isfinite(best)
    err = np.full(T, np.nan)
    if engaged.any():
        win = np.flatnonzero(elig & (key == best[row]))
        wrow = row[win]
        first = np.ones(wrow.shape[0], bool)
        first[1:] = wrow[1:] != wrow[:-1]
        win = win[first]
        err[row[win]] = total[win]
    af = np.asarray(af, dtype=np.float64).reshape(-1)
    ready = af <= 1e-6
    attack = np.asarray(attack, dtype=np.int64).reshape(-1)
    fire = (attack >= 1) & (attack <= 8)
    imp = A.action_attack_context(attack)
    ok = engaged & ready & np.isfinite(err)
    return err, ok, fire, imp, engaged


def iter_episode_arrays(collect_dir: "str | Path", split: str, shards: int = 0,
                        ) -> Iterator[tuple[int, np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray]]:
    """Walk the corpus yielding ``(demo_id, err, ok, fire, imp, engaged)`` per
    episode — the per-episode primitive both ``_scan_corpus`` (the pooled
    reference build) and calibration harnesses (which need to apply a
    corruption to a HELD-OUT subject's own episodes before re-aggregating,
    scripts/analysis/attack_arms_calibration.py) consume. Factored out so
    the calibration script never re-derives the aim-error/op-ready/weapon-
    context computation — one source of truth for what an "episode" is
    here."""
    dd = Path(collect_dir) / split
    man = json.loads((dd / "manifest.json").read_text())
    shard_list = man["shards"]
    n = len(shard_list) if not shards else min(shards, len(shard_list))
    for sh in shard_list[:n]:
        for _ei, dmi, fsl, esl, arr in A.iter_shard_episodes(
                sh, str(dd),
                obs=("entity_types", "entity_modality_id", "entity_rel",
                     "attack_finished"),
                acts=("attack",)):
            cnt = np.asarray(arr["entity_count"][fsl], dtype=np.int64)
            af = np.asarray(arr["attack_finished"][fsl])
            if len(af) < MIN_ENGAGED_FRAMES_EP:
                continue
            err, ok, fire, imp, engaged = _episode_arrays(
                cnt, arr["entity_types"][esl], arr["entity_modality_id"][esl],
                arr["entity_rel"][esl], af, arr["attack"][fsl])
            yield int(dmi), err, ok, fire, imp, engaged


def collapse_to_effective_events(fire: np.ndarray, imp: np.ndarray,
                                 hz: float) -> np.ndarray:
    """One event per EFFECTIVE-cadence cooldown window (a refractory-period
    collapse), per EXACT weapon impulse — 2026-08-15 hold-texture fix #2
    (BEAM COOLDOWN-COLLAPSE ASYMMETRY, coordinator iteration 3): the eval
    stream's discharge lane (``attack & attack_finished`` expired) fires on
    the engine's LITERAL per-shot re-entry cadence for a held continuous
    weapon (e.g. LG's 0.1s W_Attack chain-bolts), not the EFFECTIVE cadence
    (``qnn.weapons.COOLDOWN_SEC`` — LG 0.2s, that module's own docstring:
    "differs from WEAPON_PHYSICS[..]['cooldown'] for LG, 0.1 literal vs 0.2
    effective think-chain cadence") the human corpus's op-attack
    decision-frame convention already collapses to. A discharge tick is
    kept only when at least that weapon's OWN effective cooldown (in ticks)
    has elapsed since the last KEPT event for THAT SAME weapon impulse — a
    weapon switch resets the refractory clock (a different weapon's
    cooldown never blocks this one). Apply this to the BOT stream ONLY (the
    human side already IS the collapsed convention, per every existing
    qnn.human / human_band discharge reader); both sides then share one
    event definition before run/gap extraction (``hold_texture_samples``).
    """
    out = np.zeros(fire.shape[0], dtype=bool)
    pos = np.flatnonzero(fire)
    if pos.size == 0:
        return out
    last_kept = -np.inf
    last_imp: int | None = None
    n_cd = len(COOLDOWN_SEC)
    for p in pos:
        w = int(imp[p])
        cd_ticks = float(COOLDOWN_SEC[w]) * hz if 0 <= w < n_cd else 0.0
        if last_imp != w or (p - last_kept) >= cd_ticks:
            out[p] = True
            last_kept = p
            last_imp = w
    return out


def hold_texture_samples(fire: np.ndarray, imp: np.ndarray,
                         keep_mask: np.ndarray, hz: float,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Discharge run-lengths (ticks) + UNTRUNCATED inter-discharge gap
    EXCESS over each exact weapon's own effective refire cooldown (ticks) —
    2026-08-15 hold-texture fix #1 (GAP TRUNCATION BY THE MASK, coordinator
    iteration 3): the OLD extraction (``dwell_times(fire, keep_mask,
    only_value=0)``) counted a mask-run's leading/trailing PARTIAL
    zero-stretch as a real gap — a human's engagement flickering mid-refire-
    cycle truncated the sample below the physically-possible floor (RL's
    0.8s/16-tick cooldown read a 7-tick median gap). A gap sample now exists
    ONLY between two discharges BOTH observed within one unbroken
    (``keep_mask`` AND same-exact-weapon-impulse) run — the same
    "unbroken same-equipped-weapon run" convention as
    ``qnn.human.band_bank._gap_excess`` (the op-attack run convention),
    ported to TICKS (this module's unit) instead of seconds. Per-tick
    weapon cooldown comes from ``qnn.weapons.COOLDOWN_SEC`` (the engine's
    OPERATIVE re-fire cadence), indexed by the EXACT impulse equipped at
    each segment — never a family-pooled value (SG 0.5s vs SSG 0.7s, e.g.,
    differ within "SG+SSG"). Callers pass the BOT's stream through
    ``collapse_to_effective_events`` first (fix #2); the human side is
    passed through unchanged (already collapsed)."""
    fire_i64 = fire.astype(np.int64)
    runs = dwell_times(fire_i64, keep_mask, only_value=1)
    idx = np.nonzero(keep_mask)[0]
    if idx.size < 2:
        return runs, np.empty(0, dtype=np.float64)
    run_break = np.nonzero(np.diff(idx) != 1)[0] + 1
    wpn_break = np.nonzero(np.diff(imp[idx].astype(np.int64)) != 0)[0] + 1
    starts = np.unique(np.concatenate([[0], run_break, wpn_break]))
    bounds = np.append(starts, idx.size)
    n_cd = len(COOLDOWN_SEC)
    gap_chunks: list[np.ndarray] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = idx[s:e]
        ft = seg[fire[seg]]
        if ft.size >= 2:
            w = int(imp[seg[0]])
            cd_ticks = float(COOLDOWN_SEC[w]) * hz if 0 <= w < n_cd else 0.0
            gap_chunks.append(np.diff(ft).astype(np.float64) - cd_ticks)
    gap_excess = (np.concatenate(gap_chunks) if gap_chunks
                 else np.empty(0, dtype=np.float64))
    return runs, gap_excess


def _empty_demo_slot() -> dict[str, Any]:
    return {
        fam: {"ready": np.zeros(NBINS, np.int64), "fire": np.zeros(NBINS, np.int64),
              "runs": [], "gaps": []}
        for fam in (*FAMILIES, POOLED)
    }


def _accumulate_episode(err, ok, fire, imp, engaged,
                        slot: dict[str, Any]) -> None:
    """Fold one episode's per-tick arrays into a demo/subject slot (the same
    accumulation the pooled reference build AND a calibration subject use —
    a corrupted episode is just a different ``(err, ok, fire, imp, engaged)``
    tuple fed through the identical fold). HUMAN-side hold-texture: no
    collapse (the corpus op-attack convention already is one-event-per-
    effective-cooldown-window); ``hold_texture_samples`` supplies the
    truncation-safe gap-excess extraction (fix #1)."""
    idx_all = np.digitize(err, EDGES) - 1
    for fam, imps in (*FAMILY_IMPULSES.items(), (POOLED, None)):
        fam_mask = np.isin(imp, imps) if imps is not None else np.ones_like(imp, bool)
        sel = ok & fam_mask
        if sel.any():
            idx = idx_all[sel]
            good = (idx >= 0) & (idx < NBINS)
            idx = idx[good]
            f = fire[sel][good]
            slot[fam]["ready"] += np.bincount(idx, minlength=NBINS)
            if f.any():
                slot[fam]["fire"] += np.bincount(idx[f], minlength=NBINS)
        keep_mask = engaged & fam_mask
        if keep_mask.any():
            runs, gap_excess = hold_texture_samples(fire, imp, keep_mask, HZ)
            if runs.size:
                slot[fam]["runs"].extend(int(x) for x in runs)
            if gap_excess.size:
                slot[fam]["gaps"].extend(float(x) for x in gap_excess)


def build_demo_slots(episodes: "Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]",
                     ) -> dict[int, dict[str, Any]]:
    """``(demo_id, err, ok, fire, imp, engaged)`` episodes -> {demo_id:
    per-family slot}, the shared aggregation both ``_scan_corpus`` (whole
    corpus) and a calibration subject (a held-out/corrupted demo subset)
    build through."""
    acc: dict[int, dict[str, Any]] = {}
    for dmi, err, ok, fire, imp, engaged in episodes:
        slot = acc.setdefault(dmi, _empty_demo_slot())
        _accumulate_episode(err, ok, fire, imp, engaged, slot)
    return acc


def _scan_corpus(collect_dir: Path, split: str, shards: int,
                 ) -> dict[int, dict[str, Any]]:
    """One pass over the corpus -> {demo_id: per-family {ready, fire, runs,
    gaps}}. Everything downstream (pooled reference + between-demo spread) is
    a cheap re-aggregation over this per-demo cache — no second scan."""
    total, logged = 0, 0
    acc: dict[int, dict[str, Any]] = {}
    for dmi, err, ok, fire, imp, engaged in iter_episode_arrays(
            collect_dir, split, shards):
        slot = acc.setdefault(dmi, _empty_demo_slot())
        _accumulate_episode(err, ok, fire, imp, engaged, slot)
        total += 1
        if total % 2000 == 0 and total // 2000 != logged:
            logged = total // 2000
            print(f"[attack-conditional] {total} episodes scanned "
                  f"({len(acc)} demos so far)", flush=True)
    return acc


# ── corruption controls (calibration; ported recipes from
# qnn.eval.humanlikeness.human_band.perturb_dwell2 / perturb_disch_thin3,
# adapted to this module's raw per-tick (err, ready, fire, imp, engaged)
# representation instead of band_bank's post-processed per-frame dict — the
# two feed shapes differ so the functions can't be called directly, but the
# corruption RECIPE is identical) ───────────────────────────────────────────

def corrupt_dwell_x2(err: np.ndarray, ok: np.ndarray, fire: np.ndarray,
                     imp: np.ndarray, engaged: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """Time-stretch x2 (a1p-style hold pathology): every STATE frame twice
    (err/ok/imp/engaged repeat, dwells double, hold texture stretches);
    discharge EVENTS keep one tick and double their gaps (repeating a fire
    frame would fabricate sub-cooldown discharge pairs, a different
    pathology than stretch) — human_band.perturb_dwell2's exact recipe.
    Operates on ``_episode_arrays``'s output tuple directly (``ok`` is
    already engaged & op-ready & finite-err, ``_accumulate_episode``'s
    selection mask — repeating it is exactly "this tick's op-readiness now
    lasts two ticks")."""
    n = err.shape[0]
    fire2 = np.zeros(2 * n, dtype=bool)
    fire2[0::2] = fire
    return (np.repeat(err, 2), np.repeat(ok, 2), fire2,
            np.repeat(imp, 2), np.repeat(engaged, 2))


def corrupt_disch_thin_x3(err: np.ndarray, ok: np.ndarray, fire: np.ndarray,
                          imp: np.ndarray, engaged: np.ndarray,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Keep every 3rd discharge event in temporal order, clear the rest —
    chronic under-fire at intact aim-error/engagement context
    (human_band.perturb_disch_thin3's exact recipe, ported onto this
    module's ``fire`` stream)."""
    fire2 = fire.copy()
    pos = np.nonzero(fire2)[0]
    fire2[pos[np.arange(pos.size) % 3 != 0]] = False
    return err, ok, fire2, imp, engaged


CORRUPTIONS = {
    "dwell_x2": corrupt_dwell_x2,
    "disch_thin_x3": corrupt_disch_thin_x3,
}


# ── pooling ──────────────────────────────────────────────────────────────

def _pool_curve(demos: dict[int, dict[str, Any]], fam: str,
                ids: "list[int] | None" = None) -> dict[str, np.ndarray]:
    ids = ids if ids is not None else list(demos)
    ready = np.zeros(NBINS, np.int64)
    fire = np.zeros(NBINS, np.int64)
    for d in ids:
        ready = ready + demos[d][fam]["ready"]
        fire = fire + demos[d][fam]["fire"]
    return {"ready": ready, "fire": fire}


def _curve_rate_and_norm(ready: np.ndarray, fire: np.ndarray,
                         ) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(ready > 0, fire / np.maximum(ready, 1), np.nan)
    base_r = ready[:_NORM_BINS].sum()
    base_f = fire[:_NORM_BINS].sum()
    base = (base_f / base_r) if base_r > 0 else np.nan
    norm = rate / base if (base and np.isfinite(base) and base > 0) else \
        np.full(NBINS, np.nan)
    return rate, norm


def curve_deviation(ready: np.ndarray, norm: np.ndarray, ref_norm: np.ndarray,
                    ) -> "tuple[float, int] | None":
    """MASS-WEIGHTED mean absolute deviation between one side's NORMALIZED
    curve and a reference curve, weighted by ``ready`` (this side's own
    per-bin sample mass) — 2026-08-15 bin-collapse fix (defect #2):

    * weighting by mass (rather than an unweighted max over shared bins)
      means a single sparsely-populated bin can no longer dominate the
      statistic the way a MAX does;
    * ``None`` when fewer than ``MIN_BINS_FOR_VERDICT`` bins have BOTH
      curves finite AND this side's own ready-mass > 0 — a curve spanning
      too few populated bins cannot characterize a SHAPE at all (the
      alignment-law weapons, whose bot-side tracking can be steady enough
      to concentrate op-ready mass into very few error bins, is exactly the
      failure mode this guards) and must never produce a gated verdict
      regardless of how much raw tick mass backs those few bins.

    Returns ``(deviation, n_bins_compared)``. Used identically by the
    between-demo tolerance derivation below (comparing a demo's own curve to
    the pooled leave-that-demo-out reference) and by
    ``qnn.decode_fit.gates.attack_conditional_arm`` (comparing the bot's
    curve to the human reference) — one shared statistic, never two rulers
    that could drift apart."""
    ready = np.asarray(ready, dtype=np.float64)
    both = np.isfinite(norm) & np.isfinite(ref_norm) & (ready > 0)
    if int(both.sum()) < MIN_BINS_FOR_VERDICT:
        return None
    w = ready[both]
    dev = np.abs(norm[both] - ref_norm[both])
    return float(np.sum(w * dev) / np.sum(w)), int(both.sum())


def _pool_hold(demos: dict[int, dict[str, Any]], fam: str,
              ids: "list[int] | None" = None) -> tuple[np.ndarray, np.ndarray]:
    """(runs [ticks, int], gap_excess [ticks over cooldown, float]) — gaps
    are EXCESS values (can be negative pre-collapse noise, ideally >= 0
    post-fix), never raw tick counts (fix #1/#2, coordinator iteration 3)."""
    ids = ids if ids is not None else list(demos)
    runs: list[int] = []
    gaps: list[float] = []
    for d in ids:
        runs.extend(demos[d][fam]["runs"])
        gaps.extend(demos[d][fam]["gaps"])
    return np.asarray(runs, np.int64), np.asarray(gaps, np.float64)


def _quantiles(a: np.ndarray) -> "tuple[float, float] | None":
    if a.size < 5:
        return None
    return float(np.percentile(a, 50)), float(np.percentile(a, 90))


# ── shared subject-vs-reference scorers ─────────────────────────────────────
# The ONE curve/hold comparison both qnn.decode_fit.gates (bot streams) and
# scripts/analysis/attack_arms_calibration.py (held-out/corrupted human
# subjects) call — a single ruler, never two that could drift apart.

def score_curve(ready: np.ndarray, fire: np.ndarray,
                ref_block: dict[str, Any]) -> dict[str, Any]:
    """One subject's (ready, fire) per-bin counts vs a reference family
    block (``build_reference``'s per-family dict). ``{"ok": None, "note":
    ...}`` when unscorable (bin-collapse guard or no tolerance available) —
    NEVER a fail; otherwise ``{"ok", "max_deviation", "tol_p95",
    "bins_compared"}``."""
    _, norm = _curve_rate_and_norm(ready, fire)
    ref_norm = np.asarray([np.nan if x is None else x
                          for x in ref_block["rate_norm"]], dtype=np.float64)
    tol = ref_block.get("curve_between_demo_p95_deviation")
    scored = curve_deviation(ready, norm, ref_norm)
    if scored is None or tol is None:
        return {"ok": None, "note": (
            f"fewer than {MIN_BINS_FOR_VERDICT} mutually-populated bins — "
            "curve too concentrated to characterize a shape (bin-collapse "
            "guard)" if scored is None else
            "human tolerance unavailable for this family")}
    dev, n_bins = scored
    return {"ok": bool(dev <= tol), "max_deviation": round(dev, 4),
            "tol_p95": round(float(tol), 4), "bins_compared": n_bins}


def score_hold(runs: np.ndarray, gaps: np.ndarray,
              ref_block: dict[str, Any]) -> dict[str, Any]:
    """One subject's discharge run-length / inter-burst-GAP-EXCESS samples
    (fix #1/#2, coordinator iteration 3: gaps are excess-over-cooldown
    ticks, truncation-safe, collapsed-to-effective-events on the bot side)
    vs a reference family block's p50/p90 + between-demo tolerances."""
    qrun, qgap = _quantiles(runs), _quantiles(gaps)
    checks: dict[str, Any] = {}
    for label, val, ref_key, tol_key in (
            ("run_p50", qrun[0] if qrun else None,
             "run_ticks_p50", "run_p50_between_demo_p95_deviation"),
            ("run_p90", qrun[1] if qrun else None,
             "run_ticks_p90", "run_p90_between_demo_p95_deviation"),
            ("gap_excess_p50", qgap[0] if qgap else None,
             "gap_excess_ticks_p50", "gap_excess_p50_between_demo_p95_deviation"),
            ("gap_excess_p90", qgap[1] if qgap else None,
             "gap_excess_ticks_p90", "gap_excess_p90_between_demo_p95_deviation")):
        ref_val, tol = ref_block.get(ref_key), ref_block.get(tol_key)
        if val is None or ref_val is None or tol is None:
            continue
        dev = abs(val - ref_val)
        checks[label] = {"bot": round(val, 3), "human": round(ref_val, 3),
                         "deviation": round(dev, 3),
                         "tol_p95": round(float(tol), 3),
                         "ok": bool(dev <= tol)}
    if not checks:
        return {"ok": None, "note": "no comparable quantiles — unscored"}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


# ── between-demo spread (the tolerance source; v2 respec fix #1) ──────────

def _between_demo_curve_deviations(demos: dict[int, dict[str, Any]],
                                   fam: str) -> "list[float]":
    """Per-demo deviation of THAT demo's own normalized curve from the
    pooled LEAVE-THAT-DEMO-OUT reference (every other demo) — an EFFECT-SIZE
    spread across human sessions that does NOT shrink as the corpus grows
    (unlike v1's split-half-of-the-pool deviation, which measured the
    pooled-mean estimator's own sampling error). Demos under
    ``PER_DEMO_MIN_TICKS`` own ready-mass, or whose leave-out reference is
    itself under the floor, don't contribute a sample (their own curve
    estimate would be dominated by measurement noise, not real between-demo
    signal); the SAME ``curve_deviation`` statistic the gate arm uses scores
    each contributing demo, so tolerance and observed value are always
    apples-to-apples."""
    ids = list(demos)
    totals = _pool_curve(demos, fam)
    total_ready, total_fire = totals["ready"], totals["fire"]
    out: list[float] = []
    for d in ids:
        own = demos[d][fam]
        if int(own["ready"].sum()) < PER_DEMO_MIN_TICKS:
            continue
        loo_ready = total_ready - own["ready"]
        loo_fire = total_fire - own["fire"]
        if int(loo_ready.sum()) < PER_DEMO_MIN_TICKS:
            continue
        _, own_norm = _curve_rate_and_norm(own["ready"], own["fire"])
        _, loo_norm = _curve_rate_and_norm(loo_ready, loo_fire)
        scored = curve_deviation(own["ready"], own_norm, loo_norm)
        if scored is not None:
            out.append(scored[0])
    return out


def _tagged_hold_arrays(demos: dict[int, dict[str, Any]], fam: str,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated (runs, run_demo_tags, gaps, gap_demo_tags) — lets the
    leave-one-demo-out comparison below select "this demo" / "every other
    demo" via a vectorized boolean mask instead of re-concatenating N-1
    demos' lists per demo (O(demos) instead of O(demos^2))."""
    runs_list: list[int] = []
    run_tags: list[int] = []
    gaps_list: list[float] = []
    gap_tags: list[int] = []
    for d, slot in demos.items():
        r, g = slot[fam]["runs"], slot[fam]["gaps"]
        if r:
            runs_list.extend(r)
            run_tags.extend([d] * len(r))
        if g:
            gaps_list.extend(g)
            gap_tags.extend([d] * len(g))
    return (np.asarray(runs_list, np.int64), np.asarray(run_tags, np.int64),
            np.asarray(gaps_list, np.float64), np.asarray(gap_tags, np.int64))


def _between_demo_hold_deviations(demos: dict[int, dict[str, Any]], fam: str,
                                  ) -> dict[str, "list[float]"]:
    """Per-demo hold-texture (run/gap-excess p50/p90) deviation from the
    pooled leave-that-demo-out reference — same between-demo-spread rule as
    ``_between_demo_curve_deviations``, applied to the four hold quantiles
    (gaps are the fix #1/#2 excess-over-cooldown quantity)."""
    runs, run_tags, gaps, gap_tags = _tagged_hold_arrays(demos, fam)
    out: dict[str, list] = {"run_p50": [], "run_p90": [],
                            "gap_excess_p50": [], "gap_excess_p90": []}
    demo_ids = set(run_tags.tolist()) | set(gap_tags.tolist())
    for d in demo_ids:
        own_r, loo_r = runs[run_tags == d], runs[run_tags != d]
        own_g, loo_g = gaps[gap_tags == d], gaps[gap_tags != d]
        qo_r, ql_r = _quantiles(own_r), _quantiles(loo_r)
        qo_g, ql_g = _quantiles(own_g), _quantiles(loo_g)
        if qo_r and ql_r:
            out["run_p50"].append(abs(qo_r[0] - ql_r[0]))
            out["run_p90"].append(abs(qo_r[1] - ql_r[1]))
        if qo_g and ql_g:
            out["gap_excess_p50"].append(abs(qo_g[0] - ql_g[0]))
            out["gap_excess_p90"].append(abs(qo_g[1] - ql_g[1]))
    return out


def _p95(vals: "list[float]") -> "float | None":
    return float(np.percentile(vals, 95)) if len(vals) >= 4 else None


# ── the built reference (JSON-serializable) ─────────────────────────────────

def families_from_demos(demos: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Per-family (+ ``ALL`` pooled) curve/hold-texture blocks — pooled
    counts, the normalized curve, and the BETWEEN-DEMO p95 tolerances — from
    an arbitrary ``{demo_id: per-family slot}`` dict. Shared by
    ``build_reference`` (the whole corpus, minus nothing) and
    ``scripts/analysis/attack_arms_calibration.py`` (a leave-holdout-out
    subset, so the calibration's held-out subjects are never scored against
    a reference partly built from themselves)."""
    families: dict[str, Any] = {}
    for fam in (*FAMILIES, POOLED):
        pooled = _pool_curve(demos, fam)
        rate, norm = _curve_rate_and_norm(pooled["ready"], pooled["fire"])
        runs, gaps = _pool_hold(demos, fam)
        qrun, qgap = _quantiles(runs), _quantiles(gaps)
        curve_dev = _p95(_between_demo_curve_deviations(demos, fam))
        hold_dev = _between_demo_hold_deviations(demos, fam)
        families[fam] = {
            "n_ready_ticks": int(pooled["ready"].sum()),
            "n_fire_ticks": int(pooled["fire"].sum()),
            "bin_centers_deg": BIN_CENTERS,
            "rate": [None if not np.isfinite(x) else round(float(x), 4)
                     for x in rate],
            "rate_norm": [None if not np.isfinite(x) else round(float(x), 4)
                         for x in norm],
            "curve_between_demo_p95_deviation": curve_dev,
            "n_runs": int(runs.size), "n_gaps": int(gaps.size),
            "run_ticks_p50": qrun[0] if qrun else None,
            "run_ticks_p90": qrun[1] if qrun else None,
            # gap_excess = ticks OVER the exact weapon's own effective
            # cooldown (qnn.weapons.COOLDOWN_SEC), truncation-safe extraction
            # (2026-08-15 fix #1/#2, coordinator iteration 3) — should be
            # >= 0 on both sides now (never a "physically impossible" gap
            # below the refire floor).
            "gap_excess_ticks_p50": qgap[0] if qgap else None,
            "gap_excess_ticks_p90": qgap[1] if qgap else None,
            "run_p50_between_demo_p95_deviation": _p95(hold_dev["run_p50"]),
            "run_p90_between_demo_p95_deviation": _p95(hold_dev["run_p90"]),
            "gap_excess_p50_between_demo_p95_deviation":
                _p95(hold_dev["gap_excess_p50"]),
            "gap_excess_p90_between_demo_p95_deviation":
                _p95(hold_dev["gap_excess_p90"]),
        }
    return families


def build_reference(collect_dir: Path, split: str = "precomputed_train",
                    shards: int = 0) -> dict[str, Any]:
    """Scan the corpus once and build the full conditional-trigger +
    hold-texture reference, keyed by family (+ ``ALL`` pooled). Tolerance is
    the BETWEEN-DEMO p95 deviation (v2, deterministic — no resampling RNG
    needed, unlike v1's random split-half)."""
    demos = _scan_corpus(Path(collect_dir), split, shards)
    families = families_from_demos(demos)
    return {
        "_meta": {
            "ref_version": REF_VERSION,
            "collect_dir": str(collect_dir),
            "split": split,
            "n_demos": len(demos),
            "thin_data_min_ticks": THIN_DATA_MIN_TICKS,
            "per_demo_min_ticks": PER_DEMO_MIN_TICKS,
            "min_bins_for_verdict": MIN_BINS_FOR_VERDICT,
            "contract": (
                "P(discharge | total aim-error bin [deg], op-ready) per "
                "weapon family (current-position anchor, view-basis "
                "crosshair = +x, most-aligned in-SIGHT actor selection — "
                "the E7 methodology collapsed to the unsigned total-error "
                "axis) plus engaged-conditioned discharge run-length / "
                "inter-burst-gap quantiles (ticks). Tolerance = p95 of the "
                "BETWEEN-DEMO deviation (each demo's own curve/quantiles vs "
                "the pooled leave-that-demo-out reference) — an effect-size "
                "spread across human sessions that does NOT shrink with "
                "corpus size (v2, 2026-08-15: v1's random-split-half "
                "deviation was an Axiom-3 violation, a significance bar in "
                "disguise). Curve deviation is MASS-WEIGHTED over "
                "mutually-populated bins (>= MIN_BINS_FOR_VERDICT, else "
                "unscored) — never a max over shared bins, so a single "
                "concentrated bin cannot gate a verdict (v2 bin-collapse "
                "fix). Hold-texture GAP is EXCESS over the exact weapon's "
                "own effective refire cooldown (qnn.weapons.COOLDOWN_SEC), "
                "extracted only between two discharges BOTH observed "
                "within one unbroken engaged/same-weapon run (v3, "
                "2026-08-15: the old zero-run extraction truncated a "
                "mask-broken run's partial gap below the physical cooldown "
                "floor); the BOT stream is additionally collapsed to one "
                "event per effective-cooldown window before extraction (the "
                "eval stream's literal per-shot re-entry cadence for a held "
                "continuous weapon otherwise leaks in as extra events the "
                "human corpus's already-collapsed convention never has). "
                "Gate arms: attack_conditional / attack_hold_texture, "
                "qnn.decode_fit.gates — ATTACK-GATE RESPEC, Brian "
                "2026-08-15."),
        },
        "families": families,
    }


# ── cache ────────────────────────────────────────────────────────────────

def reference_cache_path(collect_dir: "str | Path", split: str) -> Path:
    """Per-collect cache path, next to the human-band bank artifact (both
    live under the collect's own ``human_baseline/`` dir — the file's
    residence inside THIS collect_dir is the content key, same convention as
    every other ``qnn.human`` baseline / the band bank; ``ref_version``
    inside the doc is the staleness check, band_bank's own pattern)."""
    from qnn.human import BASELINE_SUBDIR
    return (Path(collect_dir) / BASELINE_SUBDIR
            / f"_attack_conditional_ref_{split}.json")


def load_or_build_reference(collect_dir: "str | Path",
                            split: str = "precomputed_train", *,
                            shards: int = 0, force: bool = False,
                            ) -> dict[str, Any]:
    """Cached reference for ``collect_dir``; rebuilds when the cache is
    missing, unreadable, or stamped with a stale ``ref_version``."""
    path = reference_cache_path(collect_dir, split)
    if path.exists() and not force:
        try:
            doc = json.loads(path.read_text())
            if doc.get("_meta", {}).get("ref_version") == REF_VERSION:
                return doc
        except (OSError, json.JSONDecodeError):
            pass
    doc = build_reference(collect_dir, split, shards)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=(
        "Backfill the attack-conditional / hold-texture human reference "
        "(ATTACK-GATE RESPEC, 2026-08-15)."))
    ap.add_argument("collect_dir", type=Path)
    ap.add_argument("--split", default="precomputed_train")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    doc = load_or_build_reference(args.collect_dir, args.split,
                                  shards=args.shards, force=args.force)
    path = reference_cache_path(args.collect_dir, args.split)
    print(f"[attack-conditional] {len(doc['families'])} families -> {path}")
    for fam, blk in doc["families"].items():
        print(f"  {fam:8s} n_ready={blk['n_ready_ticks']:>8d} "
              f"n_fire={blk['n_fire_ticks']:>7d} "
              f"curve_p95_dev={blk['curve_between_demo_p95_deviation']}")


if __name__ == "__main__":
    main()
