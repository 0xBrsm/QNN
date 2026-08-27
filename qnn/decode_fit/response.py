"""Per-weapon response fits — the v2 estimator core.

One statistical model replaces the v1 median→PAV→interp→threshold chain: per
weapon, every operative discharge is a LogNormal likelihood sample around a
3-parameter monotone saturating response of the decode lever,

    hbw_i ~ LogNormal( μ_w(g_i, tms_i), σ_w² )
    μ_w(g, tms) = log( floor + (native·(1 + c·(tms − tms_ref)) − floor)·e^(−g/k) )

so NATIVE (the gain-0 baseline), the KNEE (where added gain stops buying
tightening — a smooth functional of ``k``, not a step-threshold detector), and
the INVERSION (target hbw → gain) all derive from ONE fitted curve and cannot
disagree (the v1 SG knee-0.04-vs-p50-gain-0.443 class). The acquisition↔
intercept coupling is the fitted coefficient ``c`` (turn_mag_scale shifts the
native baseline), not a fit-order convention.

Uncertainty is first-class: cluster bootstrap (episodes are the resampling
unit — consecutive shots at one opponent trajectory are not independent) plus
a finite-difference observed-information fallback. A fit whose decision
quantities are undetermined at the swept span says so (``knee_undetermined``,
LR monotonicity p-value) instead of shipping a guess.

Weighted rows: legacy v1 cell medians enter as ``is_median`` pseudo-rows with
Var[log median of n] ≈ σ²·(π/2)/n (asymptotic), which is what lets the whole
machinery run retroactively on the seed43 grids (Phase-1 acceptance) before
any new compute is spent. Range pins are importance-weighted toward the human
engagement-range mass (renormalized over the pins actually present).

Frontier semantics (Brian 2026-07-16 decision 1): the achievable frontier per
weapon is ``max(floor upper-CI, human reachable elite)`` — the fit cannot
claim below what its data supports, and we never target below the best
sustained human per-demo median (pooled-event tails are not median-reachable;
skill-curves §15). Targets below the frontier are REFUSED and the plan rides
the frontier.
"""
from __future__ import annotations

import math
import os
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats

from qnn.decode_fit.context import (ABBR_TO_MODELNAME, CALIBRATION_FAMILIES,
                                    INTERCEPT_WEAPONS, TRANSFER_ALIAS,
                                    WEAPON_IMPULSE)
from qnn.decode_fit.events import CrestWindowTable, EventTable
from qnn.decode_fit.human_refs import hbw_to_pct, pct_to_hbw

# hbw floor for the log transform (a perfect 0.0-hbw discharge is finite skill,
# not −inf); well below the tightest human pooled p1.
_HBW_EPS = 0.05
# median pseudo-rows: Var[log median of n] ≈ σ²·(π/2)/n (asymptotic normality
# of the sample median of a log-normal, in log space).
_MEDIAN_VAR_FACTOR = math.pi / 2.0
# knee = the gain realizing this fraction of the achievable tightening
# (native→floor): g_knee = k·ln(1/(1−ρ)).
KNEE_RHO = 0.9
# a knee is UNDETERMINED when its bootstrap CI spans more than this ratio or
# exceeds the swept gain span — the data cannot place it.
KNEE_CI_RATIO_MAX = 20.0
# LR monotonicity: the gain lever "measurably moves the axis" iff the full
# model beats the flat (span=0) model at this significance.
MONOTONE_P_MAX = 0.01


def _pin_row_weights(table: EventTable, pin_weights: dict[str, float] | None
                     ) -> np.ndarray:
    """Per-row likelihood weights: base ``weight`` × the human engagement-range
    mass of the row's pin (renormalized to mean 1 over the rows present, so the
    total information mass is unchanged — only its allocation)."""
    w = table["weight"].astype(float).copy()
    if pin_weights:
        pw = np.array([pin_weights.get(str(p), np.nan) for p in table["pin"]], float)
        if np.isfinite(pw).any():
            pw = np.where(np.isfinite(pw), pw, np.nanmean(pw))
            pw = pw / max(pw.mean(), 1e-12)
            w = w * pw
    return w


def _pin_strata(table: EventTable, pin_weights: dict[str, float] | None
                ) -> tuple[list[str] | None, np.ndarray | None, np.ndarray | None]:
    """Pin stratification design (the a25rc3c SG confound): ``(pins, pin_idx,
    mix_w)`` — stratum labels, per-row stratum index, and the human range-mix
    weight of each stratum (renormalized over pins PRESENT). None when the
    table carries a single pin (nothing to stratify). Opponent pins shift the
    intercept level multiplicatively (SG vs the RL pin reads 2.2–3.1 hbw at
    ANY knob setting); response cells with inhomogeneous pin coverage bias an
    unstratified curve toward whichever mixture each knob level sampled."""
    pins = sorted({str(p) for p in table["pin"]})
    if len(pins) < 2:
        return None, None, None
    pin_idx = np.array([pins.index(str(p)) for p in table["pin"]], np.int64)
    mw = np.array([(pin_weights or {}).get(p, np.nan) for p in pins], float)
    if np.isfinite(mw).any():
        mw = np.where(np.isfinite(mw), mw, np.nanmean(mw))
    else:
        mw = np.ones(len(pins), float)
    mw = np.maximum(mw, 1e-9)
    return pins, pin_idx, mw / mw.sum()


def _pin_deltas(u_tail: np.ndarray, mix_w: np.ndarray) -> np.ndarray:
    """Full per-stratum log offsets from the (P−1) free parameters: stratum 0
    carries the balancing offset so the MIX-WEIGHTED mean offset is zero —
    the shared curve stays the human range-mix aggregate."""
    d_free = np.asarray(u_tail, float)
    d0 = -float(mix_w[1:] @ d_free) / float(mix_w[0])
    return np.concatenate([[d0], d_free])


@dataclass
class GainResponse:
    """One weapon's fitted gain→intercept response (+ tms coupling)."""
    weapon: str
    floor: float
    native: float                    # gain-0 hbw at tms_ref
    k: float                         # saturation scale (gain units)
    sigma: float                     # log-space event scatter
    tms_coeff: float                 # d(native multiplier)/d(tms − tms_ref)
    tms_ref: float
    n_events: int
    n_clusters: int
    loglik: float
    swept_gain_span: tuple[float, float]
    param_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    monotone_p: float = 1.0          # LR p-value vs the flat (span=0) model
    knee_undetermined: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _boot: np.ndarray | None = None  # (B, 4) bootstrap draws of (floor, native, k, sigma)

    # ── curve evaluation ───────────────────────────────────────────────────
    def native_at(self, tms: float | None) -> float:
        dt = 0.0 if tms is None else float(tms) - self.tms_ref
        return self.native * (1.0 + self.tms_coeff * dt)

    def predict_hbw(self, gain: float, tms: float | None = None) -> float:
        span = max(self.native_at(tms) - self.floor, 0.0)
        return self.floor + span * math.exp(-max(float(gain), 0.0) / self.k)

    def gain_grad(self, gain: float, tms: float | None = None) -> float:
        """d hbw / d gain (≤ 0) — the secant-correction slope."""
        span = max(self.native_at(tms) - self.floor, 0.0)
        return -span / self.k * math.exp(-max(float(gain), 0.0) / self.k)

    def predict_ci(self, gain: float, tms: float | None = None,
                   level: float = 0.95) -> tuple[float, float]:
        """Bootstrap CI of the MEAN response at ``gain`` (curve uncertainty,
        not event scatter). RAISES when no bootstrap draws survived — a
        zero-width CI silently recorded as certainty is exactly the
        degenerate-gate class (a26 first fit: RL/SG/NG point CIs); the fit
        must either carry real uncertainty or fail loud."""
        if self._boot is None or not len(self._boot):
            raise RuntimeError(
                f"{self.weapon}: no bootstrap draws — prediction CI "
                f"unavailable (n_boot_ok=0). The placement gate cannot run "
                f"on a point estimate; fix the bootstrap, never fall back.")
        dt = 0.0 if tms is None else float(tms) - self.tms_ref
        fl, na, kk = self._boot[:, 0], self._boot[:, 1], self._boot[:, 2]
        na = na * (1.0 + self.tms_coeff * dt)
        pred = fl + np.maximum(na - fl, 0.0) * np.exp(-max(float(gain), 0.0) / kk)
        lo, hi = np.percentile(pred, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
        return (float(lo), float(hi))

    def knee(self, rho: float = KNEE_RHO) -> tuple[float, tuple[float, float]]:
        g = self.k * math.log(1.0 / (1.0 - rho))
        if self._boot is None or not len(self._boot):
            ci = self.param_ci.get("k", (self.k, self.k))
            return g, (ci[0] * math.log(1.0 / (1.0 - rho)),
                       ci[1] * math.log(1.0 / (1.0 - rho)))
        ks = self._boot[:, 2] * math.log(1.0 / (1.0 - rho))
        return g, (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))

    def floor_ci(self) -> tuple[float, float]:
        if self._boot is None or not len(self._boot):
            return self.param_ci.get("floor", (self.floor, self.floor))
        return (float(np.percentile(self._boot[:, 0], 2.5)),
                float(np.percentile(self._boot[:, 0], 97.5)))

    def invert(self, target_hbw: float, tms: float | None = None,
               gain_cap: float | None = None) -> dict[str, Any]:
        """Target hbw → the gain reproducing it on the fitted curve. Flags:
        ``below_native`` (target worse than native — the DOWN band owns it),
        ``at_floor`` (target at/below the fitted floor — unreachable by gain).
        ``gain_cap`` bounds the returned gain (e.g. knee(0.95) — past it the
        curve is flat to within noise and more gain buys nothing)."""
        n_eff = self.native_at(tms)
        span = max(n_eff - self.floor, 1e-9)
        t = float(target_hbw)
        if t >= n_eff:
            return {"gain": 0.0, "below_native": True, "at_floor": False}
        if t <= self.floor + 1e-9:
            g = gain_cap if gain_cap is not None else self.knee(0.95)[0]
            return {"gain": float(g), "below_native": False, "at_floor": True}
        g = self.k * math.log(span / (t - self.floor))
        if gain_cap is not None:
            g = min(g, float(gain_cap))
        return {"gain": float(g), "below_native": False, "at_floor": False}


def _nll(u: np.ndarray, g: np.ndarray, dt: np.ndarray, y: np.ndarray,
         w: np.ndarray, med_n: np.ndarray, fit_c: bool,
         pin_idx: np.ndarray | None = None,
         mix_w: np.ndarray | None = None) -> float:
    """Weighted negative log-likelihood in unconstrained parameters
    u = (log floor, log span, log k, log σ [, c] [, δ₁..δ_{P−1}]) — the δs
    are per-pin multiplicative offsets (log-additive) under a mix-weighted
    zero-sum constraint (see ``_pin_strata``)."""
    floor = math.exp(u[0])
    span = math.exp(u[1])
    k = math.exp(u[2])
    sig2 = math.exp(2.0 * u[3])
    base = 5 if fit_c else 4
    c = u[4] if fit_c else 0.0
    # native = floor + span; native_eff = native·(1+c·dt)
    # ⇒ native_eff − floor = span + (floor + span)·c·dt
    n_eff_span = span + (floor + span) * (c * dt)
    mean = floor + np.maximum(n_eff_span, 1e-9) * np.exp(-g / k)
    mu = np.log(np.maximum(mean, 1e-9))
    if pin_idx is not None and len(u) > base:
        # pin offsets DECAY with the lever: δ_p·e^{−g/κ_d} (u[base] = log κ_d).
        # Pins converge at high gain (the decode steering dominates both sides
        # of the matchup — the SG edge evidence); constant offsets misread the
        # converged edge cells as noise. κ_d → ∞ recovers the constant model,
        # and the mix-weighted zero-sum survives the scaling, so the shared
        # curve stays the range-mix aggregate at EVERY lever value.
        kd = math.exp(u[base])
        mu = mu + _pin_deltas(u[base + 1:], mix_w)[pin_idx] * np.exp(-g / kd)
    var = sig2 * np.where(med_n > 0, _MEDIAN_VAR_FACTOR / np.maximum(med_n, 1.0), 1.0)
    ll = -0.5 * (np.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)
    return float(-(w * ll).sum())


def _fit_workers() -> int:
    """Bootstrap solver pool width (DECODEFIT_FIT_WORKERS; default 3/4 of the
    cores). The bootstrap dominated the between-round wall time — ~0.7 s per
    L-BFGS-B solve × 200 draws × weapons, single-core, while the wave pool sat
    idle between rounds."""
    return max(1, int(os.environ.get(
        "DECODEFIT_FIT_WORKERS", str(max(1, (os.cpu_count() or 8) * 3 // 4)))))


_BOOT_CTX: dict | None = None
_NUM_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_POOL = None


def _boot_init(payload: dict) -> None:
    global _BOOT_CTX
    _BOOT_CTX = payload


def _pool_warmup(_i: int) -> None:
    import time as _t
    _t.sleep(0.3)          # hold the task so every worker actually spawns


def _boot_pool():
    """Persistent SPAWN-context solver pool, created once per process.

    SPAWN, not fork: L-BFGS-B's BLAS calls run on the numeric libraries'
    per-process thread pools, and a forked child inherits the parent's
    already-initialized multi-threaded BLAS — capping it post-fork is
    impossible without threadpoolctl, and the resulting N workers × M threads
    contention measured 3× SLOWER than the serial loop. Spawn children read
    the caps from the env snapshot taken at spawn, so the caps are set only
    for the spawn window (the warmup map forces every worker to spawn inside
    it) and restored — wave/freeplay subprocesses inherit an untouched env."""
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        saved = {v: os.environ.get(v) for v in _NUM_THREAD_VARS}
        for v in _NUM_THREAD_VARS:
            os.environ[v] = "1"
        try:
            _POOL = ProcessPoolExecutor(max_workers=_fit_workers(),
                                        mp_context=mp.get_context("spawn"))
            list(_POOL.map(_pool_warmup, range(_fit_workers())))
        except Exception:
            # spawn needs an importable __main__ (a REPL/stdin caller has
            # none) — degrade to the serial loop, never fail the fit
            _mark_pool_broken()
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
    return _POOL


def _mark_pool_broken() -> None:
    global _POOL
    try:
        if _POOL is not None and _POOL is not False:
            _POOL.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    _POOL = False          # sentinel: don't retry every fit
    print("[decode-fit] bootstrap solver pool unavailable — "
          "falling back to the serial loop", flush=True)


def _boot_chunk(args: tuple) -> list:
    payload, rows_chunk = args
    global _BOOT_CTX
    _BOOT_CTX = payload
    return [_boot_mle_one(r) for r in rows_chunk]


def _boot_mle_one(rows: np.ndarray) -> "list[float] | None":
    """One bootstrap draw's (floor, native, k, sigma) — reads the shared
    arrays installed by :func:`_boot_init` (worker-global so the pool pickles
    them once per worker, not once per draw)."""
    p = _BOOT_CTX
    try:
        ub, _ = _fit_mle(p["x"][rows], p["dt"][rows], p["y"][rows],
                         p["w"][rows], p["med_n"][rows], p["fit_c"],
                         p["pin_idx"][rows] if p["pin_idx"] is not None
                         else None,
                         p["mix_w"])
    except Exception:
        return None
    fl = math.exp(ub[0])
    return [fl, fl + math.exp(ub[1]), math.exp(ub[2]), math.exp(ub[3])]


def _cluster_boot_draws(x: np.ndarray, dt: np.ndarray, y: np.ndarray,
                        w: np.ndarray, med_n: np.ndarray, fit_c: bool,
                        pin_idx: np.ndarray | None, mix_w: np.ndarray | None,
                        clusters: np.ndarray, uniq: np.ndarray,
                        rng: np.random.Generator, n_boot: int
                        ) -> list[list[float]]:
    """Cluster-bootstrap (floor, native, k, sigma) draws, solved ACROSS a
    process pool. The pick arrays are pre-drawn serially from ``rng`` (the
    solves never consume it) and the pool map preserves draw order, so the
    result is bit-identical to the old serial loop at any worker count."""
    idx_by_c = {cid: np.flatnonzero(clusters == cid) for cid in uniq}
    rows_list = [np.concatenate([idx_by_c[cid] for cid in
                                 rng.choice(uniq, size=len(uniq), replace=True)])
                 for _ in range(n_boot)]
    payload = {"x": x, "dt": dt, "y": y, "w": w, "med_n": med_n,
               "fit_c": fit_c, "pin_idx": pin_idx, "mix_w": mix_w}
    workers = _fit_workers()
    pool = _boot_pool() if (workers > 1 and n_boot >= 16) else None
    if pool:
        # contiguous chunks preserve draw order; the payload pickles once per
        # chunk (~2·workers), not once per draw
        size = max(1, math.ceil(n_boot / (workers * 2)))
        chunks = [rows_list[i:i + size] for i in range(0, len(rows_list), size)]
        try:
            outs = list(pool.map(_boot_chunk, [(payload, c) for c in chunks]))
            return [r for chunk in outs for r in chunk if r is not None]
        except Exception:
            _mark_pool_broken()
    _boot_init(payload)
    return [r for r in (_boot_mle_one(rows) for rows in rows_list)
            if r is not None]


def _fit_mle(g: np.ndarray, dt: np.ndarray, y: np.ndarray, w: np.ndarray,
             med_n: np.ndarray, fit_c: bool,
             pin_idx: np.ndarray | None = None,
             mix_w: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """L-BFGS-B with a few heuristic restarts (k is the fragile axis)."""
    hb = np.exp(y)
    lo_q = float(np.quantile(hb, 0.1))
    g0_rows = g <= np.quantile(g, 0.2)
    native0 = float(np.exp(np.average(y[g0_rows], weights=w[g0_rows]))) \
        if g0_rows.any() else float(hb.mean())
    floor0 = max(min(lo_q, native0 * 0.8), _HBW_EPS * 2)
    span0 = max(native0 - floor0, 0.05)
    gspan = max(float(g.max() - g.min()), 1e-3)
    sig0 = max(float(np.std(y)), 0.1)
    n_delta = 0 if pin_idx is None or mix_w is None else len(mix_w) - 1
    best: tuple[np.ndarray, float] | None = None
    for k_mult in (0.1, 0.3, 1.0):
        u0 = [math.log(floor0), math.log(span0), math.log(gspan * k_mult), math.log(sig0)]
        if fit_c:
            u0.append(0.5)
        if n_delta:
            # κ_d starts large (≈ constant offsets) so the decay is earned
            u0.append(math.log(gspan * 3.0))
            u0 += [0.0] * n_delta
        res = optimize.minimize(
            _nll, np.array(u0, float),
            args=(g, dt, y, w, med_n, fit_c,
                  pin_idx if n_delta else None, mix_w if n_delta else None),
            method="L-BFGS-B",
            bounds=[(math.log(_HBW_EPS), 4.0), (-6.0, 4.0),
                    (math.log(gspan) - 7.0, math.log(gspan) + 3.0), (-4.0, 2.0)]
            + ([(-5.0, 5.0)] if fit_c else [])
            + ([(math.log(gspan) - 4.0, math.log(gspan) + 6.0)] if n_delta else [])
            + [(-2.0, 2.0)] * n_delta)
        if res.success or res.fun < (best[1] if best else math.inf):
            if best is None or res.fun < best[1]:
                best = (res.x, float(res.fun))
    assert best is not None
    return best


def fit_gain_response(table: EventTable, weapon: str, *,
                      pin_weights: dict[str, float] | None = None,
                      tms_ref: float = 1.0, n_boot: int = 200,
                      seed: int = 0) -> GainResponse:
    """Fit one weapon's gain response on its (α=0, tremor=0) rows. Rows at any
    tms are pooled; the coupling coefficient is fit iff tms actually varies.
    Raises when the weapon has no usable rows (fail loud — no cross-weapon
    fallback; SSG/SNG alias at PLAN level, keeping their own ladders)."""
    t = table.where(weapon=weapon)
    m = (t["alpha"] == 0.0) & (t["tremor"] == 0.0) & np.isfinite(t["gain"])
    t = t.filter(m)
    # thresholds are on EFFECTIVE mass (median pseudo-rows summarize many
    # events) and lever coverage, not raw row count.
    if len(t) == 0 or t["weight"].sum() < 50 or len(np.unique(t["gain"])) < 3:
        raise ValueError(
            f"{weapon}: insufficient gain-arm data ({len(t)} rows, "
            f"{0 if len(t) == 0 else t['weight'].sum():.0f} effective events, "
            f"{0 if len(t) == 0 else len(np.unique(t['gain']))} gain levels) — "
            "the instrument produced too little discharge mass (check fire-rate "
            "invariant / forced loadout); no fallback.")
    y = np.log(np.maximum(t["hbw"], _HBW_EPS))
    g = t["gain"].astype(float)
    dt = t["tms"].astype(float) - tms_ref
    if not np.isfinite(dt).all():
        dt = np.zeros_like(g)
    fit_c = bool(np.ptp(dt[np.isfinite(dt)]) > 1e-6)
    w = _pin_row_weights(t, pin_weights)
    med_n = np.where(t["is_median"], t["weight"], 0.0)
    pins, pin_idx, mix_w = _pin_strata(t, pin_weights)

    u, nll = _fit_mle(g, dt, y, w, med_n, fit_c, pin_idx, mix_w)
    floor, span, k = math.exp(u[0]), math.exp(u[1]), math.exp(u[2])
    sigma = math.exp(u[3])
    c = float(u[4]) if fit_c else 0.0
    _base = 5 if fit_c else 4
    deltas = _pin_deltas(u[_base + 1:], mix_w) if pins else None
    delta_decay = math.exp(u[_base]) if pins else None

    # LR monotonicity: full vs flat (span → 0 ⇒ constant mean at floor+span mix;
    # refit with span pinned tiny). The flat model keeps the pin strata so the
    # LR tests the GAIN term, not the pin composition.
    def _nll_flat() -> float:
        if pins:
            yw_arr = np.empty_like(y)
            for si in range(len(pins)):
                m_s = pin_idx == si
                yw_arr[m_s] = float(np.average(y[m_s], weights=w[m_s])) \
                    if m_s.any() else 0.0
        else:
            yw_arr = np.full_like(y, float(np.average(y, weights=w)))
        sigma_flat = max(float(np.sqrt(np.average((y - yw_arr) ** 2, weights=w))), 1e-3)
        v = sigma_flat ** 2 * np.where(
            med_n > 0, _MEDIAN_VAR_FACTOR / np.maximum(med_n, 1.0), 1.0)
        ll = -0.5 * (np.log(2 * math.pi * v) + (y - yw_arr) ** 2 / v)
        return float(-(w * ll).sum())

    lr = max(2.0 * (_nll_flat() - nll), 0.0)
    monotone_p = float(stats.chi2.sf(lr, df=2))     # span + k vs flat

    # cluster bootstrap (percentile CIs for every decision quantity)
    rng = np.random.default_rng(seed)
    clusters = t["cluster"]
    uniq = np.unique(clusters)
    draws: list[list[float]] = []
    if n_boot > 0 and len(uniq) >= 4:
        draws = _cluster_boot_draws(g, dt, y, w, med_n, fit_c,
                                    pin_idx if pins else None, mix_w,
                                    clusters, uniq, rng, n_boot)
    boot = np.array(draws, float) if draws else None

    def _ci(col: int, point: float) -> tuple[float, float]:
        if boot is None or not len(boot):
            return (point, point)
        return (float(np.percentile(boot[:, col], 2.5)),
                float(np.percentile(boot[:, col], 97.5)))

    gspan = (float(g.min()), float(g.max()))
    knee_point = k * math.log(1.0 / (1.0 - KNEE_RHO))
    k_ci = _ci(2, k)
    knee_ci = (k_ci[0] * math.log(1 / (1 - KNEE_RHO)),
               k_ci[1] * math.log(1 / (1 - KNEE_RHO)))
    knee_undet = bool(
        monotone_p > MONOTONE_P_MAX
        or (knee_ci[0] > 0 and knee_ci[1] / max(knee_ci[0], 1e-12) > KNEE_CI_RATIO_MAX)
        or knee_ci[1] > gspan[1] * 2.0
        or (boot is None))

    return GainResponse(
        weapon=weapon, floor=floor, native=floor + span, k=k, sigma=sigma,
        tms_coeff=c, tms_ref=tms_ref, n_events=int(t["weight"].sum()),
        n_clusters=int(len(uniq)), loglik=-nll, swept_gain_span=gspan,
        param_ci={"floor": _ci(0, floor), "native": _ci(1, floor + span),
                  "k": k_ci, "sigma": _ci(3, sigma)},
        monotone_p=monotone_p, knee_undetermined=knee_undet,
        diagnostics={"lr_stat": round(lr, 2), "fit_c": fit_c,
                     "n_boot_ok": 0 if boot is None else int(len(boot)),
                     "knee_point": round(knee_point, 4),
                     "knee_ci": [round(knee_ci[0], 4), round(knee_ci[1], 4)],
                     "pin_offsets": ({p: round(float(math.exp(d)), 3)
                                      for p, d in zip(pins, deltas)}
                                     if pins else None),
                     "pin_offset_decay": (round(delta_decay, 4)
                                          if pins else None)},
        _boot=boot,
    )


@dataclass
class AlphaResponse:
    """The SUPER-band α ray at a pinned gain — same saturating form, α as the
    lever. ``start`` should agree with the gain arm's prediction at
    ``pinned_gain`` (continuity diagnostic, not a constraint)."""
    weapon: str
    pinned_gain: float
    floor: float
    start: float
    k: float
    sigma: float
    n_events: int
    n_clusters: int
    param_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    monotone_p: float = 1.0
    _boot: np.ndarray | None = None

    def predict_hbw(self, alpha: float) -> float:
        span = max(self.start - self.floor, 0.0)
        return self.floor + span * math.exp(-max(float(alpha), 0.0) / self.k)

    def alpha_grad(self, alpha: float) -> float:
        span = max(self.start - self.floor, 0.0)
        return -span / self.k * math.exp(-max(float(alpha), 0.0) / self.k)

    def floor_ci(self) -> tuple[float, float]:
        if self._boot is None or not len(self._boot):
            return self.param_ci.get("floor", (self.floor, self.floor))
        return (float(np.percentile(self._boot[:, 0], 2.5)),
                float(np.percentile(self._boot[:, 0], 97.5)))

    def predict_ci(self, alpha: float, level: float = 0.95
                   ) -> tuple[float, float]:
        """Bootstrap CI of the MEAN response at ``alpha`` — the α-arm twin of
        ``GainResponse.predict_ci``. RAISES without draws (same fail-loud
        contract): the super-band plan CI was the (pred, pred) hole that let
        NG/SG ship point promises to the gate."""
        if self._boot is None or not len(self._boot):
            raise RuntimeError(
                f"{self.weapon}: α response has no bootstrap draws — "
                f"prediction CI unavailable; fix the bootstrap, never fall "
                f"back to a point CI.")
        fl, st, kk = self._boot[:, 0], self._boot[:, 1], self._boot[:, 2]
        pred = fl + np.maximum(st - fl, 0.0) * np.exp(
            -max(float(alpha), 0.0) / kk)
        lo, hi = np.percentile(pred, [(1 - level) / 2 * 100,
                                      (1 + level) / 2 * 100])
        return (float(lo), float(hi))

    def invert(self, target_hbw: float, alpha_cap: float | None = None
               ) -> dict[str, Any]:
        span = max(self.start - self.floor, 1e-9)
        t = float(target_hbw)
        if t >= self.start:
            return {"alpha": 0.0, "at_floor": False}
        if t <= self.floor + 1e-9:
            a = alpha_cap if alpha_cap is not None else self.k * math.log(1 / (1 - 0.95))
            return {"alpha": float(a), "at_floor": True}
        a = self.k * math.log(span / (t - self.floor))
        if alpha_cap is not None:
            a = min(a, float(alpha_cap))
        return {"alpha": float(a), "at_floor": False}


def fit_alpha_response(table: EventTable, weapon: str, pinned_gain: float, *,
                       pin_weights: dict[str, float] | None = None,
                       n_boot: int = 200, seed: int = 0) -> AlphaResponse | None:
    """Fit the α ray on rows swept in α at (approximately) the pinned gain.
    None when the table carries no α arm for the weapon (plan then stops at
    the gain frontier)."""
    t = table.where(weapon=weapon)
    m = (t["alpha"] > 0.0) & (t["tremor"] == 0.0)
    ta = t.filter(m)
    if len(ta) < 6:
        return None
    # include the α=0 anchor rows at the pinned gain for the start estimate
    anchor = t.filter((t["alpha"] == 0.0) & (t["tremor"] == 0.0)
                      & (np.abs(t["gain"] - pinned_gain) < 1e-6))
    tt = EventTable.concat([ta, anchor]) if len(anchor) else ta
    y = np.log(np.maximum(tt["hbw"], _HBW_EPS))
    a = tt["alpha"].astype(float)
    w = _pin_row_weights(tt, pin_weights)
    med_n = np.where(tt["is_median"], tt["weight"], 0.0)
    pins, pin_idx, mix_w = _pin_strata(tt, pin_weights)
    dt = np.zeros_like(a)
    u, nll = _fit_mle(a, dt, y, w, med_n, fit_c=False,
                      pin_idx=pin_idx, mix_w=mix_w)
    floor, span, k, sigma = (math.exp(u[0]), math.exp(u[1]),
                             math.exp(u[2]), math.exp(u[3]))

    rng = np.random.default_rng(seed)
    clusters = tt["cluster"]
    uniq = np.unique(clusters)
    draws: list[list[float]] = []
    if n_boot > 0 and len(uniq) >= 4:
        draws = _cluster_boot_draws(a, dt, y, w, med_n, False,
                                    pin_idx if pins else None, mix_w,
                                    clusters, uniq, rng, n_boot)
    boot = np.array(draws, float) if draws else None

    def _ci(col: int, point: float) -> tuple[float, float]:
        if boot is None or not len(boot):
            return (point, point)
        return (float(np.percentile(boot[:, col], 2.5)),
                float(np.percentile(boot[:, col], 97.5)))

    return AlphaResponse(
        weapon=weapon, pinned_gain=float(pinned_gain), floor=floor,
        start=floor + span, k=k, sigma=sigma,
        n_events=int(tt["weight"].sum()), n_clusters=int(len(uniq)),
        param_ci={"floor": _ci(0, floor), "start": _ci(1, floor + span),
                  "k": _ci(2, k), "sigma": _ci(3, sigma)},
        _boot=boot,
    )


# Effective event mass a measured cell needs before it may outrank the curve.
MEASURED_FRONTIER_MIN_MASS = 60.0


def measured_frontier(table: EventTable, weapon: str, fit: "GainResponse",
                      pin_weights: dict[str, float] | None = None
                      ) -> dict[str, Any] | None:
    """The best MEASURED (gain, α) cell for ``weapon``, stratum-adjusted to
    the range-mix aggregate — the plan-stage form of 'measurements outrank
    curves'. SG's plateau-then-drop response gives the saturating exponential
    a confident wrong basin whose floor claims cells the sweep MEASURED
    beating it (0.85→1.46 while the 'floor' said 2.40); a refusal must not
    stand on a curve the cells refute. Rows in each (gain, α) cell are
    divided by their pin's fitted offset (``pin_offsets``) so partial-pin
    cells estimate the mix aggregate; cells need ≥ MEASURED_FRONTIER_MIN_MASS
    effective events. Returns ``{gain, alpha, hbw, hbw_hi, n_eff}`` for the
    lowest upper-CI cell, or None when the weapon has no qualifying cell."""
    t = table.where(weapon=weapon)
    t = t.filter(t["tremor"] == 0.0)
    if len(t) == 0:
        return None
    offs = (fit.diagnostics or {}).get("pin_offsets") or {}
    ln_off = np.array([math.log(offs.get(str(p), 1.0) or 1.0)
                       for p in t["pin"]], float)
    y = np.log(np.maximum(t["hbw"], _HBW_EPS)) - ln_off
    w = _pin_row_weights(t, pin_weights)
    keys = np.stack([np.round(t["gain"].astype(float), 4),
                     np.round(t["alpha"].astype(float), 4)], axis=1)
    best: dict[str, Any] | None = None
    for gv, av in {(float(a), float(b)) for a, b in keys}:
        m = (keys[:, 0] == gv) & (keys[:, 1] == av)
        n_eff = float(w[m].sum())
        if n_eff < MEASURED_FRONTIER_MIN_MASS:
            continue
        mu = float(np.average(y[m], weights=w[m]))
        var = float(np.average((y[m] - mu) ** 2, weights=w[m]))
        n_cl = max(len(np.unique(t["cluster"][m])), 1)
        se = math.sqrt(max(var, 1e-9) / n_cl)
        cand = {"gain": gv, "alpha": av, "hbw": float(math.exp(mu)),
                "hbw_hi": float(math.exp(mu + 1.96 * se)), "n_eff": n_eff}
        if best is None or cand["hbw_hi"] < best["hbw_hi"]:
            best = cand
    return best


def apply_measured_frontier(plans: dict[str, Any], table: EventTable,
                            gain_fits: dict[str, "GainResponse"],
                            ladder: dict[str, dict[float, float]],
                            pin_weights: dict[str, dict[str, float]] | None
                            ) -> dict[str, Any]:
    """Refused plans re-anchor on the best measured cell when its UPPER CI
    beats the parametric frontier: the plan takes that cell's levers and
    promises its measured aggregate (band ``frontier-measured``; refusal
    stands only if the measurement is still short of the wish). Aliases
    re-derive from their source. Non-refused plans pass through untouched —
    curve inversions that placed in-band answer to confirmation as usual."""
    out = dict(plans)
    for abbr, plan in plans.items():
        if not getattr(plan, "refused", False) or plan.alias_of:
            continue
        fit = gain_fits.get(abbr)
        if fit is None:
            continue
        mf = measured_frontier(table, abbr, fit,
                               (pin_weights or {}).get(abbr))
        # trigger on the POINT estimate with a 5% margin — cluster-level
        # cell CIs are too wide to ever beat a frontier; the promise interval
        # (hbw, hbw_hi) is what the placement gate adjudicates.
        if mf is None or mf["hbw"] >= plan.frontier_hbw * 0.95:
            continue
        lad = ladder.get(abbr) or {}
        pct = round(hbw_to_pct(mf["hbw"], lad), 1) if lad else plan.frontier_pct
        still_refused = mf["hbw"] > plan.target_hbw
        out[abbr] = dataclasses.replace(
            plan, gain=round(mf["gain"], 4), alpha=round(mf["alpha"], 4),
            pred_hbw=round(mf["hbw"], 4),
            pred_hbw_ci=(round(mf["hbw"], 4), round(mf["hbw_hi"], 4)),
            band="frontier-measured", refused=bool(still_refused),
            frontier_hbw=round(mf["hbw"], 4), frontier_pct=pct,
            achieved_pct=pct,
            notes=(plan.notes + f"; measured frontier: cell (gain {mf['gain']}, "
                   f"α {mf['alpha']}) measured {mf['hbw']:.3f} hbw "
                   f"(hi {mf['hbw_hi']:.3f}, n_eff {mf['n_eff']:.0f}) beats the "
                   f"fitted frontier {plan.frontier_hbw:.3f} — the curve is "
                   "refuted, promising the measurement").lstrip("; "))
        # Family aliases ride the corrected source exactly; only identity and
        # impulse differ. A measured frontier must not reopen member-specific
        # aim coordinates.
        for a2, p2 in plans.items():
            if getattr(p2, "alias_of", None) != abbr:
                continue
            source_plan = out[abbr]
            out[a2] = dataclasses.replace(
                source_plan, weapon=a2, impulse=WEAPON_IMPULSE[a2],
                alias_of=abbr,
                notes=(source_plan.notes + f"; measured frontier via family "
                       f"{abbr}").lstrip("; "))
    return out


@dataclass
class TremorResponse:
    """The universal DOWN band: log-linear degradation shared across weapons,
    per-weapon native intercepts. ``log hbw ~ native_w + slope·tremor``."""
    slope: float
    natives: dict[str, float]
    sigma: float
    n_events: int
    slope_ci: tuple[float, float] = (0.0, 0.0)

    def invert(self, weapon: str, target_hbw: float) -> float:
        """Universal tremor magnitude placing ``weapon`` at ``target_hbw``
        (target must be WORSE than native; 0.0 otherwise)."""
        nat = self.natives.get(weapon)
        if nat is None or target_hbw <= nat or self.slope <= 1e-9:
            return 0.0
        return float((math.log(target_hbw) - math.log(nat)) / self.slope)


def fit_tremor_response(table: EventTable, *, n_boot: int = 200,
                        seed: int = 0) -> TremorResponse | None:
    """Pooled tremor fit on (gain=0, α=0) rows across all weapons. None when no
    tremor arm exists in the table."""
    m = (table["tremor"] >= 0.0) & (table["alpha"] == 0.0) & (table["gain"] == 0.0)
    t = table.filter(m)
    if len(t) == 0 or not (t["tremor"] > 0).any():
        return None
    weapons = sorted(set(t["weapon"]))
    y = np.log(np.maximum(t["hbw"], _HBW_EPS))
    x = t["tremor"].astype(float)
    w = t["weight"].astype(float)
    # weighted least squares with per-weapon intercepts (design matrix)
    X = np.zeros((len(t), len(weapons) + 1))
    for j, wep in enumerate(weapons):
        X[t["weapon"] == wep, j] = 1.0
    X[:, -1] = x
    W = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    resid = y - X @ beta
    sigma = float(np.sqrt(np.average(resid ** 2, weights=w)))
    slope = float(beta[-1])
    # cluster bootstrap on the slope
    rng = np.random.default_rng(seed)
    uniq = np.unique(t["cluster"])
    slopes = []
    if n_boot > 0 and len(uniq) >= 4:
        idx_by_c = {cid: np.flatnonzero(t["cluster"] == cid) for cid in uniq}
        for _ in range(n_boot):
            rows = np.concatenate([idx_by_c[c] for c in
                                   rng.choice(uniq, size=len(uniq), replace=True)])
            try:
                bb, *_ = np.linalg.lstsq(
                    X[rows] * W[rows, None], y[rows] * W[rows], rcond=None)
                slopes.append(float(bb[-1]))
            except Exception:
                continue
    ci = ((float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
          if slopes else (slope, slope))
    return TremorResponse(
        slope=slope,
        natives={wep: float(math.exp(beta[j])) for j, wep in enumerate(weapons)},
        sigma=sigma, n_events=int(w.sum()), slope_ci=ci)


# ── acquisition (tms → Fitts throughput) ─────────────────────────────────────

def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx, my = float((w * x).sum()), float((w * y).sum())
    cov = float((w * (x - mx) * (y - my)).sum())
    vx = float((w * (x - mx) ** 2).sum())
    vy = float((w * (y - my) ** 2).sum())
    if vx <= 0.0 or vy <= 1e-18:
        return 0.0
    return cov / math.sqrt(vx * vy)


ACQ_MARGINAL_CORR_FLOOR = 0.25


def fit_acquisition(cells: list[dict], band: dict, *, target_pct: float = 50.0,
                    min_corr: float = 0.5, n_boot: int = 1000,
                    seed: int = 0) -> dict[str, Any]:
    """tms → throughput inversion onto the human band, with bootstrap CI over
    cells. ``cells`` rows carry {tms, throughput_bits_per_s, n_settled}. Fails
    loud (``unfittable``) when the lever does not move the axis (the bot-pin
    in-view failure: no target-free flicks) — port of the v1 responsiveness
    guard, CI added. NO CLAMP: the fitted tms is a measured human-match;
    sweep-bound targets flag for extension, never accept a boundary.

    The responsiveness corr is weighted by each cell's settled count: at heavy
    damping (low tms) only the easiest acquisitions settle, so a handful of
    survivor-biased events can read deceptively fast — a ~30-settle cell must
    not out-vote a ~300-settle one. A dead lever still reads ~0 under any
    weighting; ``acq_marginal`` marks the ambiguous band above
    ``ACQ_MARGINAL_CORR_FLOOR`` where one seed-replicate extension (never a
    guess) is allowed to decide."""
    from qnn.decode_fit.human_refs import pct_to_throughput, throughput_to_pct
    rows = [(float(c["tms"]), float(c["throughput_bits_per_s"]),
             max(float(c.get("n_settled") or 1.0), 1.0))
            for c in cells if c.get("throughput_bits_per_s") is not None]
    if len({t for t, _, _ in rows}) < 2:
        raise ValueError(f"acquisition fit needs ≥2 tms levels, got {len(rows)} cells")
    tms_v = np.array([t for t, _, _ in rows], float)
    tp_v = np.array([p for _, p, _ in rows], float)
    w_v = np.array([w for _, _, w in rows], float)
    corr = _weighted_corr(tms_v, tp_v, w_v)
    corr_raw = float(np.corrcoef(tms_v, tp_v)[0, 1]) if np.std(tp_v) > 1e-9 else 0.0
    ladder = {float(k): float(v) for k, v in band["ladder"].items()}
    target_tp = pct_to_throughput(float(target_pct), ladder)

    def _invert(tv: np.ndarray, pv: np.ndarray) -> float:
        med: dict[float, list[float]] = {}
        for t, p in zip(tv, pv):
            med.setdefault(round(float(t), 6), []).append(float(p))
        pts = sorted((t, float(np.median(v))) for t, v in med.items())
        ts = np.array([t for t, _ in pts])
        ps = np.array([p for _, p in pts])
        order = np.argsort(ps)
        return float(np.interp(target_tp, ps[order], ts[order]))

    tms_star = _invert(tms_v, tp_v)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(max(n_boot, 0)):
        idx = rng.integers(0, len(tms_v), len(tms_v))
        if len(set(tms_v[idx])) < 2:
            continue
        boots.append(_invert(tms_v[idx], tp_v[idx]))
    ci = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
          if boots else (tms_star, tms_star))
    lo_tp, hi_tp = band["band"]
    med_by_tms = {t: float(np.median(tp_v[tms_v == t])) for t in np.unique(tms_v)}
    achieved = float(np.interp(tms_star, sorted(med_by_tms),
                               [med_by_tms[t] for t in sorted(med_by_tms)]))
    return {
        "turn_mag_scale": round(tms_star, 3),
        "tms_ci": [round(ci[0], 3), round(ci[1], 3)],
        "target_pct": float(target_pct),
        "target_throughput_bits_per_s": round(target_tp, 4),
        "achieved_throughput_bits_per_s": round(achieved, 4),
        "achieved_acq_percentile": round(throughput_to_pct(achieved, ladder), 1),
        "in_band": bool(lo_tp <= achieved <= hi_tp),
        "band_bits_per_s": [lo_tp, hi_tp],
        "tms_throughput_corr": round(corr, 3),
        "tms_throughput_corr_raw": round(corr_raw, 3),
        "unfittable": bool(corr < min_corr),
        "acq_marginal": bool(ACQ_MARGINAL_CORR_FLOOR <= corr < min_corr),
        "sweep_floor_bound": bool(target_tp < float(tp_v.min())),
        "sweep_ceil_bound": bool(target_tp > float(tp_v.max())),
        "swept_tms_range": [round(float(tms_v.min()), 3), round(float(tms_v.max()), 3)],
        "curve": [{"turn_mag_scale": float(t), "throughput_bits_per_s": round(med_by_tms[t], 4)}
                  for t in sorted(med_by_tms)],
        "n_cells": len(rows),
    }


# ── the per-weapon operating-point PLAN ──────────────────────────────────────

@dataclass
class WeaponPlan:
    weapon: str
    impulse: int
    target_pct: float
    target_hbw: float
    gain: float
    alpha: float
    pred_hbw: float
    pred_hbw_ci: tuple[float, float]
    achieved_pct: float
    refused: bool
    frontier_pct: float
    frontier_hbw: float
    band: str                       # "down" | "up" | "super" | "frontier"
    tremor: float = 0.0
    alias_of: str | None = None     # SSG/NG ride SG/SNG responses
    notes: str = ""


def build_plan(gain_fits: dict[str, GainResponse],
               alpha_fits: dict[str, AlphaResponse | None],
               tremor_fit: TremorResponse | None,
               ladder: dict[str, dict[float, float]],
               reachable: dict[str, tuple[float, float]],
               targets: dict[str, float], *,
               tms: float,
               alpha_style_cap: float | None = None) -> dict[str, WeaponPlan]:
    """Resolve every targeted weapon's operating point with refusal semantics
    (decision 1): the achievable frontier is ``max(gain/α floor upper-CI,
    human reachable elite)``; a target tighter than the frontier is REFUSED and
    the plan rides the frontier. SG/SSG and NG/SNG share one response, human
    family ladder, target, and resolved operating point. ``alpha_style_cap``
    bounds α (the hold-destruction lever) — None = uncapped, adjudicated by
    the style gate."""
    plans: dict[str, WeaponPlan] = {}
    for abbr, tgt_pct in targets.items():
        lad = ladder.get(abbr)
        if lad is None or abbr not in INTERCEPT_WEAPONS:
            continue
        src = abbr if abbr in gain_fits else TRANSFER_ALIAS.get(abbr, "")
        fit = gain_fits.get(src)
        if fit is None:
            raise ValueError(f"{abbr}: no gain response (nor alias) — the "
                             "screening round must cover it")
        afit = alpha_fits.get(src)
        target_hbw = pct_to_hbw(float(tgt_pct), lad)
        n_eff = fit.native_at(tms)
        elite = reachable.get(abbr, reachable.get(src, (0.0, 0.0)))[0]
        # What each arm can SUPPORT (CI-conservative), never below the
        # sustained human elite (per-demo-median p10; pooled tails are not
        # median-reachable — skill-curves §15):
        gain_reach = max(fit.floor_ci()[1], elite)
        alpha_reach = math.inf
        if afit is not None:
            a_hi = afit.floor_ci()[1]
            if alpha_style_cap is not None:
                a_hi = max(a_hi, afit.predict_hbw(alpha_style_cap))
            alpha_reach = max(a_hi, elite)
        frontier_hbw = min(gain_reach, alpha_reach)
        refused = bool(target_hbw < frontier_hbw - 1e-9)
        eff_target = max(target_hbw, frontier_hbw)

        gain, alpha, tremor_mag = 0.0, 0.0, 0.0
        knee_g, _ = fit.knee(0.95)
        if eff_target >= n_eff:                      # DOWN band (tremor owns it)
            band = "down"
            if tremor_fit is not None:
                tremor_mag = tremor_fit.invert(src, eff_target)
            pred = n_eff if tremor_fit is None else min(
                math.exp(math.log(max(n_eff, 1e-9))
                         + tremor_fit.slope * tremor_mag), eff_target * 1.5)
            # curve CI at gain 0, shifted with the tremor delta so the plan's
            # promise interval is centered on pred (the placement gate rides
            # this interval; an unshifted gain-0 CI would false-fail every
            # DOWN-band plan by construction)
            _ci0 = fit.predict_ci(0.0, tms)
            _r = pred / max(n_eff, 1e-9)
            ci = (_ci0[0] * _r, _ci0[1] * _r)
        elif eff_target >= gain_reach - 1e-9 or afit is None:
            # UP band: the gain arm supports the target on its own.
            inv = fit.invert(eff_target, tms, gain_cap=knee_g)
            gain = inv["gain"]
            pred = fit.predict_hbw(gain, tms)
            ci = fit.predict_ci(gain, tms)
            band = "frontier" if refused else "up"
        else:
            # SUPER band: below the gain arm's reach — α extends at the pinned
            # (knee) gain. Only entered when the α fit's reach covers eff_target
            # (else frontier_hbw == gain_reach and the branch above took it).
            band = "frontier" if refused else "super"
            gain = float(afit.pinned_gain)
            inv = afit.invert(eff_target, alpha_cap=alpha_style_cap)
            alpha = inv["alpha"]
            pred = afit.predict_hbw(alpha)
            # real curve CI at the placed α — never a (pred, pred) point
            # (the a26 NG/SG degenerate-promise hole)
            ci = afit.predict_ci(alpha)

        plans[abbr] = WeaponPlan(
            weapon=abbr, impulse=WEAPON_IMPULSE[abbr],
            target_pct=float(tgt_pct), target_hbw=round(target_hbw, 4),
            gain=round(gain, 4), alpha=round(alpha, 4),
            pred_hbw=round(pred, 4),
            pred_hbw_ci=(round(ci[0], 4), round(ci[1], 4)),
            achieved_pct=round(hbw_to_pct(pred, lad), 1),
            refused=refused,
            frontier_pct=round(hbw_to_pct(frontier_hbw, lad), 1),
            frontier_hbw=round(frontier_hbw, 4),
            band=band, tremor=round(tremor_mag, 4),
            alias_of=(src if src != abbr else None),
            notes=("target below achievable frontier — riding the frontier "
                   "(decision 1)" if refused else ""),
        )
    # Make family identity structural even for direct callers that supplied
    # distinct member ladders. The CLI already expands one requested member to
    # both and rejects conflicting coordinates; this final copy prevents any
    # later refactor from reintroducing per-member aim knobs.
    for source, members in CALIBRATION_FAMILIES.items():
        present = [member for member in members if member in plans]
        if len(present) < 2:
            continue
        pcts = {plans[member].target_pct for member in present}
        if len(pcts) != 1:
            raise ValueError(
                f"conflicting calibration-family targets for {'+'.join(members)}: "
                + ", ".join(f"{m}=p{plans[m].target_pct:g}" for m in present))
        canonical = plans[source]
        for member in present:
            plans[member] = dataclasses.replace(
                canonical, weapon=member, impulse=WEAPON_IMPULSE[member],
                alias_of=(source if member != source else None))
    return plans


def build_vectors(plans: dict[str, WeaponPlan]) -> dict[str, Any]:
    """The (9,) per-IMPULSE gain/α/tremor decode vectors from a resolved plan.
    Untargeted slots stay 0. Tremor is per-impulse like the others — the old
    universal scalar (mean of the positive plan tremors) smeared every
    weapon's DOWN-band degradation onto every other weapon, so the shipped
    config diverged from every confirmed placement (a25rc3c: SG confirmed at
    tremor 0 shipped with 0.0193)."""
    gain = [0.0] * 9
    alpha = [0.0] * 9
    tremor = [0.0] * 9
    for p in plans.values():
        gain[p.impulse] = round(p.gain, 4)
        alpha[p.impulse] = round(p.alpha, 4)
        tremor[p.impulse] = round(p.tremor, 4)
    return {"gain": gain, "alpha": alpha,
            "tremor": tremor,
            "placed": {w: {"pct": p.target_pct, "impulse": p.impulse,
                           "gain": p.gain, "alpha": p.alpha,
                           "achieved_pct": p.achieved_pct, "band": p.band,
                           "refused": p.refused, "frontier_pct": p.frontier_pct}
                       for w, p in plans.items()}}


# ── the CREST arm (discharge-quality gate θ) ─────────────────────────────────
#
# agents/plans/discharge-quality-gate.md, "Fit procedure" steps 3-4. The third
# response arm, and the only one that costs no episodes: for a discharge the
# head fired at t₀ with forward window ``hbw[t₀ .. t₀+H]``, the gated
# counterfactual is DETERMINISTIC, so one instrumented eval yields the whole
# θ → at-discharge-alignment curve by offline replay.
#
# Two things make this arm structurally different from the gain/α arms and are
# the reason it is a replay + interpolation rather than a parametric MLE:
#
#   * θ does not move the tracking stream at all — it moves WHICH TICK of that
#     stream the trigger lands on. So its coordinate is the AT-DISCHARGE
#     intercept statistic, not the window-sampled tracking statistic the
#     gain/α arms are placed on (decode-fit-v2 addendum 2026-07-18). The arm
#     therefore reports a CAPTURE RATIO — at-discharge / window, the same
#     quantity the human baseline publishes as ``crest_capture`` — which is
#     paired within rows, so the per-pin level offsets that
#     ``measured_frontier`` has to divide out CANCEL, and the arm composes onto
#     whatever level the gain/α arms promised:
#         at_discharge_hbw(θ) = plan.pred_hbw × capture(θ)
#   * the law it replays is CONVERGENCE-gated (2026-07-22), so the curve is not
#     the naive "fire at first hbw ≤ θ": a hold that is predicted to get worse
#     releases immediately. That is why capture(θ) stays bounded as θ → 0
#     instead of collapsing onto a blind expiry-fire.

# θ sweep (hbw units). Fine below 3 (where every weapon's crest geometry lives),
# coarse above, out to a θ no measured window ever exceeds (= gate effectively
# OFF, the curve's own native anchor).
CREST_THETA_GRID: tuple[float, ...] = tuple(
    [round(x, 3) for x in np.arange(0.2, 3.0, 0.1)]
    + [round(x, 3) for x in np.arange(3.0, 12.001, 0.25)])
# H is a reaction-latency bound, not weapon physics (plan §Parameters): shared,
# and hard-capped at 2 ticks / 100 ms.
CREST_H_CHOICES: tuple[int, ...] = (1, 2)
# A cell must carry this much evidence before its θ may be trusted.
CREST_MIN_ROWS = 200
CREST_MIN_CLUSTERS = 8
# Style spend is never free: a gate whose clamped capture gain is smaller than
# this (in capture-ratio points) buys nothing worth deferring discharges for,
# so the plan zeroes it rather than shipping a cosmetic θ.
CREST_MIN_EFFECT = 0.02


def _geo(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted geometric mean — the package's hbw centre (``predict_hbw`` is
    ``exp(µ)`` of a LogNormal, ``measured_frontier`` is ``exp(weighted mean
    log)``). NOTE the human ``crest_capture`` reference is a ratio of per-demo
    MEDIANS; both sides of a ratio use the same centre here, so the shape
    difference largely cancels — the residual is a documented approximation,
    not a silent one."""
    if not len(x):
        return float("nan")
    return float(np.exp(np.average(np.log(np.maximum(x, _HBW_EPS)), weights=w)))


def select_operating_cells(table, weapon: str, gain: float, alpha: float, *,
                           gain_tol: float, alpha_tol: float) -> np.ndarray:
    """Row mask for ``weapon``'s (gain, α) cells within tolerance of a plan's
    resolved operating point, tremor 0. Works on both :class:`EventTable` and
    :class:`CrestWindowTable` (same column names), so the replay numerator and
    the window denominator are guaranteed to be the SAME cells.

    A tolerance exists because θ is fit RETROACTIVELY on the sweep's own waves:
    the plan's resolved (g, α) is an inversion, so the swept grid rarely lands
    on it exactly. The selected cells are recorded in the fit so the distance
    from the placed point is visible, never assumed away."""
    return ((table["weapon"] == weapon)
            & (table["tremor"] == 0.0)
            & (np.abs(table["gain"] - float(gain)) <= float(gain_tol))
            & (np.abs(table["alpha"] - float(alpha)) <= float(alpha_tol)))


def replay_crest(fwd: np.ndarray, cluster: np.ndarray, tick: np.ndarray,
                 theta: float, hold_ticks: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gated counterfactual for every discharge row: ``(offset, hbw,
    absorbed)``.

    Mirrors ``qnn.model.bench.a25.decode.attack_crest_gate_step`` exactly at
    ``ready=True`` (the eval only logs a window for a discharge the engine
    honored, which is precisely the gate's ``ready`` condition):

      * ``stop_waiting[j] = aligned[j] | diverging[j]`` with
        ``aligned = live & (hbw ≤ θ)`` and ``diverging = live & (hbw_{j+1} >
        hbw_j)``; the decode derives ``hbw_next`` from the feed-forward
        ``z_rate``, the replay reads the REALIZED next tick — the quantity that
        feed-forward is estimating (the residual is why step 5's closed-loop
        confirmation exists).
      * release at the first ``j ∈ [0, H)`` with ``stop_waiting``, else blind
        at ``j = H`` (the head's discharge is never canceled).
      * a NaN slot is a dead row (LOS lost / episode ended): neither aligned
        nor diverging, so the countdown runs on to expiry — the plan's
        LOS-lost law, not a fake crest and not a fake divergence.

    ``absorbed`` is the no-restack consequence, which the naive per-row replay
    misses: the gate arms at most one countdown at a time, so a head re-fire at
    a tick inside a live hold is folded into the pending discharge instead of
    starting its own. Absorbed rows leave the at-discharge distribution (and
    are the replay's honest read on the gate's rate cost, which stage 6's
    attack trim is what actually compensates)."""
    H = int(hold_ticks)
    if H < 1:
        raise ValueError(f"crest replay needs hold_ticks >= 1, got {H}")
    if fwd.shape[1] < H + 1:
        raise ValueError(
            f"crest replay at H={H} needs {H + 1} forward slots, window has "
            f"{fwd.shape[1]} — capture at a larger QNN_EVAL_INTERCEPT_WINDOW")
    n = len(fwd)
    w = fwd[:, :H + 1]
    live = np.isfinite(w)
    nxt = fwd[:, 1:H + 2] if fwd.shape[1] >= H + 2 else np.concatenate(
        [fwd[:, 1:H + 1], np.full((n, 1), np.nan)], axis=1)
    stop = (live & (w <= float(theta))) | (live & np.isfinite(nxt) & (nxt > w))
    stop[:, H] = True                       # expiry: pending==1 always releases
    off = stop.argmax(axis=1)
    hbw = w[np.arange(n), off]
    # no-restack, walked per cluster in tick order (the latch is per lane)
    absorbed = np.zeros(n, dtype=bool)
    order = np.lexsort((tick, cluster))
    cur = None
    busy = np.int64(-1)
    for i in order:
        if cluster[i] != cur:
            cur = cluster[i]
            busy = np.int64(np.iinfo(np.int64).min // 2)
        if tick[i] <= busy:
            absorbed[i] = True
        else:
            busy = tick[i] + off[i]
    return off, hbw, absorbed


@dataclass
class CrestResponse:
    """One weapon's replayed θ → at-discharge CAPTURE curve at a fixed H.

    ``capture[i]`` is the at-discharge statistic at ``theta_grid[i]`` divided
    by the window statistic of the SAME cells — dimensionless, directly
    comparable to the human ``crest_capture`` ratio, and multiplicative onto
    the gain/α plan's promised level. ``native_capture`` is the gate-OFF
    anchor (every row fires at t₀, nothing absorbed)."""
    weapon: str
    hold_ticks: int
    theta_grid: np.ndarray
    capture: np.ndarray                     # (T,) point estimates
    capture_ci: np.ndarray                  # (T, 2) bootstrap-by-episode CIs
    native_capture: float
    native_capture_ci: tuple[float, float]
    window_hbw: float
    native_hbw: float
    delay_mean: np.ndarray                  # (T,) mean release offset, ticks
    defer_frac: np.ndarray                  # (T,) released later than t₀
    expiry_frac: np.ndarray                 # (T,) blind release at H
    absorbed_frac: np.ndarray               # (T,) head re-fires folded in
    blind_nan_frac: np.ndarray              # (T,) expiry with no live target
    n_events: int
    n_clusters: int
    cells: list[tuple[float, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ── curve evaluation (linear interpolation on the swept grid) ──────────
    def capture_at(self, theta: float) -> float:
        return float(np.interp(float(theta), self.theta_grid, self.capture))

    def capture_ci_at(self, theta: float) -> tuple[float, float]:
        return (float(np.interp(float(theta), self.theta_grid,
                                self.capture_ci[:, 0])),
                float(np.interp(float(theta), self.theta_grid,
                                self.capture_ci[:, 1])))

    def flags_at(self, theta: float) -> dict[str, float]:
        """The style-spend register at ``theta`` (plan §Style: FLAGGED, never
        gating) — deferral fraction, realized hold, blind-release fraction."""
        return {k: round(float(np.interp(float(theta), self.theta_grid, v)), 4)
                for k, v in (("delay_mean_ticks", self.delay_mean),
                             ("defer_frac", self.defer_frac),
                             ("expiry_frac", self.expiry_frac),
                             ("absorbed_frac", self.absorbed_frac),
                             ("blind_nan_frac", self.blind_nan_frac))}

    def theta_for_capture(self, target: float, *, ci_floor: float | None = None
                          ) -> dict[str, Any]:
        """Invert the curve onto ``target``, never below ``ci_floor``.

        ``capture`` and its lower CI are monotone NON-DECREASING in θ (a looser
        gate defers less and lands nearer native), so the ADMISSIBLE set
        ``A = {θ : capture_lo(θ) ≥ ci_floor}`` is an upper interval. Within A:

          * the target is reachable → return the LOOSEST θ that reaches it
            (largest θ = least intervention = cheapest style spend);
          * the target sits below A entirely → RIDE THE FLOOR: return the
            tightest admissible θ (``min A``) and flag ``reached_target``
            False. This is the frontier-riding semantics ``build_plan`` uses
            for an unreachable gain target, and the reason the floor is a
            separate predicate from the target: when the caller has already
            clamped ``target`` up to the floor, "θ ≤ target" and "θ safe" are
            the same constraint from opposite sides and their intersection is
            empty by construction.
          * A empty → the arm cannot place θ at all without a CI that dips
            below the floor; the caller refuses.
        """
        grid, cap = self.theta_grid, self.capture
        clears_floor = (np.ones(len(grid), bool) if ci_floor is None
                        else self.capture_ci[:, 0] >= float(ci_floor) - 1e-12)
        if not clears_floor.any():
            return {"theta": 0.0, "capture": self.native_capture,
                    "reachable": False, "reached_target": False,
                    "floor_admits_none": True}
        adm = np.flatnonzero(clears_floor)
        hit = adm[cap[adm] <= float(target) + 1e-12]
        i = int(hit[-1]) if len(hit) else int(adm[0])
        return {"theta": float(grid[i]), "capture": float(cap[i]),
                "capture_ci": (float(self.capture_ci[i, 0]),
                               float(self.capture_ci[i, 1])),
                "reachable": True, "reached_target": bool(len(hit) > 0),
                "floor_admits_none": False}


def fit_crest_response(windows: CrestWindowTable, tracking: EventTable,
                       weapon: str, *, hold_ticks: int,
                       pin_weights: dict[str, float] | None = None,
                       theta_grid: "tuple[float, ...] | np.ndarray" = CREST_THETA_GRID,
                       n_boot: int = 200, seed: int = 0) -> CrestResponse:
    """Replay the θ grid over ``windows`` (already filtered to one weapon's
    operating cells) against the window statistic of the SAME cells in
    ``tracking``. Bootstrap is BY EPISODE and PAIRED — one cluster resample
    drives numerator and denominator together, so the capture CI is the CI of
    the ratio, not the (much wider) quotient of two independent CIs.

    Raises on thin evidence (fail loud; there is no cross-weapon fallback for
    a lever that changes when a bot pulls the trigger)."""
    if len(windows) < CREST_MIN_ROWS:
        raise ValueError(
            f"{weapon}: crest arm has {len(windows)} discharge windows at the "
            f"selected cells (min {CREST_MIN_ROWS}) — widen the cell tolerance "
            "or capture windows at the placed operating point; no fallback.")
    if not len(tracking):
        raise ValueError(
            f"{weapon}: crest arm has no window-sampled tracking rows at the "
            "selected cells — the capture denominator is undefined.")
    grid = np.asarray(theta_grid, dtype=np.float64)
    w_win = _pin_row_weights(EventTable(windows.cols), pin_weights)
    w_trk = _pin_row_weights(tracking, pin_weights)
    cl_win, cl_trk = windows["cluster"], tracking["cluster"]
    uniq = np.unique(np.concatenate([cl_win, cl_trk]))
    if len(uniq) < CREST_MIN_CLUSTERS:
        raise ValueError(
            f"{weapon}: crest arm spans {len(uniq)} bootstrap clusters "
            f"(min {CREST_MIN_CLUSTERS}) — a CI from this many episodes is not "
            "honest uncertainty; widen the cell tolerance.")
    iw = np.searchsorted(uniq, cl_win)
    it = np.searchsorted(uniq, cl_trk)
    nC = len(uniq)

    # per-cluster sufficient statistics: a weighted geometric mean over any
    # cluster resample is (Σ_c cnt_c·S_c) / (Σ_c cnt_c·W_c) in log space, so
    # the bootstrap never re-touches a row.
    t_num = np.bincount(it, weights=w_trk * np.log(
        np.maximum(tracking["hbw"], _HBW_EPS)), minlength=nC)
    t_den = np.bincount(it, weights=w_trk, minlength=nC)

    S = np.zeros((len(grid), nC))
    W = np.zeros((len(grid), nC))
    delay, defer, expiry, absorb, blind = (np.zeros(len(grid)) for _ in range(5))
    for j, th in enumerate(grid):
        off, hbw, ab = replay_crest(windows.fwd, cl_win, windows["tick"],
                                    float(th), hold_ticks)
        keep = ~ab
        fin = keep & np.isfinite(hbw)
        S[j] = np.bincount(iw[fin], weights=w_win[fin] * np.log(
            np.maximum(hbw[fin], _HBW_EPS)), minlength=nC)
        W[j] = np.bincount(iw[fin], weights=w_win[fin], minlength=nC)
        kw = w_win[keep]
        ksum = max(kw.sum(), 1e-12)
        delay[j] = float((kw * off[keep]).sum() / ksum)
        defer[j] = float((kw * (off[keep] > 0)).sum() / ksum)
        expiry[j] = float((kw * (off[keep] == hold_ticks)).sum() / ksum)
        absorb[j] = float((w_win * ab).sum() / max(w_win.sum(), 1e-12))
        blind[j] = float((kw * ~np.isfinite(hbw[keep])).sum() / ksum)

    n_num = np.bincount(iw, weights=w_win * np.log(
        np.maximum(windows.fwd[:, 0], _HBW_EPS)), minlength=nC)
    n_den = np.bincount(iw, weights=w_win, minlength=nC)

    def _cap(cnt: np.ndarray) -> tuple[np.ndarray, float]:
        lw = float((t_num @ cnt) / max(float(t_den @ cnt), 1e-12))
        cur = (S @ cnt) / np.maximum(W @ cnt, 1e-12)
        nat = float((n_num @ cnt) / max(float(n_den @ cnt), 1e-12))
        return np.exp(cur - lw), float(math.exp(nat - lw))

    ones = np.ones(nC)
    capture, native_capture = _cap(ones)
    rng = np.random.default_rng(seed)
    draws = np.empty((max(n_boot, 0), len(grid)))
    nat_draws = np.empty(max(n_boot, 0))
    for b in range(max(n_boot, 0)):
        cnt = np.bincount(rng.integers(0, nC, nC), minlength=nC).astype(float)
        draws[b], nat_draws[b] = _cap(cnt)
    if n_boot > 0:
        ci = np.stack([np.percentile(draws, 2.5, axis=0),
                       np.percentile(draws, 97.5, axis=0)], axis=1)
        nat_ci = (float(np.percentile(nat_draws, 2.5)),
                  float(np.percentile(nat_draws, 97.5)))
    else:
        raise ValueError(f"{weapon}: crest arm needs n_boot > 0 — a point "
                         "estimate cannot be adjudicated by the clamp.")
    d = np.diff(capture)
    cells = sorted({(round(float(g), 4), round(float(a), 4))
                    for g, a in zip(windows["gain"], windows["alpha"])})
    return CrestResponse(
        weapon=weapon, hold_ticks=int(hold_ticks), theta_grid=grid,
        capture=capture, capture_ci=ci,
        native_capture=native_capture, native_capture_ci=nat_ci,
        window_hbw=_geo(tracking["hbw"], w_trk),
        native_hbw=_geo(windows.fwd[:, 0], w_win),
        delay_mean=delay, defer_frac=defer, expiry_frac=expiry,
        absorbed_frac=absorb, blind_nan_frac=blind,
        n_events=int(len(windows)), n_clusters=int(nC), cells=cells,
        diagnostics={
            "n_tracking_rows": int(len(tracking)),
            "n_boot": int(n_boot),
            # the curve must be monotone in θ or the inversion is meaningless
            "monotone_frac": round(float((d >= -1e-9).mean()), 4),
            "max_backstep": round(float(-min(d.min(), 0.0)), 5),
            "theta_span": [float(grid[0]), float(grid[-1])],
            "capture_span": [round(float(capture[0]), 4),
                             round(float(capture[-1]), 4)],
        })


@dataclass
class CrestPlan:
    """One weapon's resolved crest operating point. ``theta`` 0.0 = gate OFF
    for this impulse (the explicit-OFF value the emitted config carries)."""
    weapon: str
    impulse: int
    hold_ticks: int
    theta: float
    armed: bool
    refused: bool
    clamped: bool
    target_capture: float
    native_capture: float
    placed_capture: float
    base_hbw: float                  # the gain/α plan's promised tracking level
    at_discharge_hbw: float
    at_discharge_hbw_ci: tuple[float, float]
    at_discharge_pct: float
    native_at_discharge_hbw: float
    native_at_discharge_pct: float
    elite_hbw: float
    # the arm's own UNADJUSTED arena-mix at-discharge level, for contrast with
    # the plan-scale number above (see the scale note in build_crest_plan)
    raw_native_at_discharge_hbw: float = float("nan")
    style_flags: dict[str, float] = field(default_factory=dict)
    alias_of: str | None = None
    notes: str = ""


def build_crest_plan(plans: dict[str, WeaponPlan],
                     crest_fits: dict[str, CrestResponse],
                     ladder: dict[str, dict[float, float]],
                     reachable: dict[str, tuple[float, float]],
                     human_capture: dict[str, float], *,
                     min_effect: float = CREST_MIN_EFFECT
                     ) -> dict[str, CrestPlan]:
    """Plan inversion for the crest arm (plan §Fit procedure step 4).

    ``ladder`` / ``reachable`` are the AT-DISCHARGE (intercept) references —
    NOT the window-sampled tracking ones the gain/α arms are placed on. θ moves
    the tick the trigger lands on, so the at-discharge ladder is the only
    coordinate it acts in. ``human_capture`` is the human family
    ``crest_capture`` p50 (``_aim_tracking_window.json``): the timing target,
    i.e. what fraction of its own typical tracking alignment a human's trigger
    actually captures.

    **Overshoot clamp — the safety-critical rule.** Tightening θ improves the
    at-discharge coordinate for free, so an uncalibrated θ walks the model
    straight past the elite anchor. The clamp is CI-conservative: θ may only be
    placed where the replayed level's bootstrap LOWER bound still sits at or
    above p100 (= the elite anchor exactly, human_refs §16). An over-elite ask
    is REFUSED and the plan rides the clamp — the same semantics as a gain
    refusal riding the achievable frontier, and for the same reason: the fit
    may not promise a placement the band says is not a human coordinate.

    Three ways a weapon ends up OFF, each recorded rather than silently
    zeroed: (a) no crest gap — the model's native capture is already at or
    tighter than the human's, so θ could only overshoot; (b) already over-elite
    natively — the clamp admits no θ at all, a pre-existing condition of the
    (g, α) placement that this lever cannot fix and must not deepen;
    (c) below ``min_effect`` — the admissible tightening is too small to be
    worth deferring discharges for.

    **Scale note (load-bearing, and thin-margin for SG).** The level this arm
    clamps is ``plan.pred_hbw × capture``, i.e. the plan's OWN promised level
    (pin-offset-adjusted onto the human range-mix aggregate, exactly what
    ``measured_frontier`` / the placement gate ride) scaled by the arm's paired
    capture ratio. It therefore assumes the per-pin level offset is COMMON to
    the window and at-discharge statistics — the same pins, the same
    difficulty, measured one tick apart. The arm's own unadjusted arena-mix
    at-discharge level is carried alongside as
    ``raw_native_at_discharge_hbw``; where the two straddle the elite anchor
    the θ is decided by that adjustment and the closed-loop confirmation is
    what settles it. Note also that the CLAMP rides the plan's POINT promise:
    the uncertainty on the base level is the placement gate's business, this
    arm is only responsible for the increment it adds."""
    out: dict[str, CrestPlan] = {}
    for abbr, plan in plans.items():
        src = plan.alias_of or abbr
        fit = crest_fits.get(src)
        lad = ladder.get(abbr)
        if fit is None or lad is None:
            continue
        elite = float(reachable.get(abbr, reachable.get(src, (0.0, 0.0)))[0])
        base = float(plan.pred_hbw)
        nat_cap = float(fit.native_capture)
        nat_hbw = base * nat_cap
        target = float(human_capture.get(abbr, human_capture.get(src, nat_cap)))
        # the tightest capture the band permits at this weapon's placed level
        clamp_cap = elite / max(base, 1e-9)
        refused = bool(target < clamp_cap - 1e-9)
        eff_target = max(target, clamp_cap)

        theta, placed_cap, clamped, note = 0.0, nat_cap, False, ""
        if nat_hbw <= elite + 1e-9:
            note = (f"already at/past the at-discharge elite anchor natively "
                    f"({nat_hbw:.3f} vs p100 {elite:.3f}) — the clamp admits no "
                    f"θ; gate stays OFF (the overshoot predates this lever and "
                    f"is the (gain, α) placement's to answer for)")
        elif eff_target >= nat_cap - 1e-9:
            note = (f"no crest gap: native capture {nat_cap:.3f} already at or "
                    f"tighter than the effective target {eff_target:.3f} "
                    f"(human {target:.3f}) — θ can only overshoot; OFF")
        else:
            inv = fit.theta_for_capture(eff_target, ci_floor=clamp_cap)
            if not inv["reachable"]:
                note = ("no admissible θ: no grid point places the replayed "
                        f"level with its lower CI at or above p100 "
                        f"({elite:.3f} hbw) — refusing rather than promising an "
                        "over-elite placement")
            elif nat_cap - inv["capture"] < min_effect:
                note = (f"admissible tightening {nat_cap - inv['capture']:.4f} "
                        f"< min_effect {min_effect} — the clamp leaves too "
                        f"little for the style spend; OFF")
            else:
                theta = float(inv["theta"])
                placed_cap = float(inv["capture"])
                clamped = not bool(inv["reached_target"])
                note = (f"target capture {target:.3f} is below the p100 clamp "
                        f"{clamp_cap:.3f} — riding the clamp (CI-conservative: "
                        f"the placed level's lower CI stays at or above the "
                        f"elite anchor)" if refused else "")
        armed = theta > 0.0
        at_hbw = base * placed_cap
        cap_lo, cap_hi = (fit.capture_ci_at(theta) if armed
                          else fit.native_capture_ci)
        out[abbr] = CrestPlan(
            weapon=abbr, impulse=WEAPON_IMPULSE[abbr],
            hold_ticks=int(fit.hold_ticks) if armed else 0,
            theta=round(theta, 4), armed=armed, refused=refused,
            clamped=clamped,
            target_capture=round(target, 4),
            native_capture=round(nat_cap, 4),
            placed_capture=round(placed_cap, 4),
            base_hbw=round(base, 4),
            at_discharge_hbw=round(at_hbw, 4),
            at_discharge_hbw_ci=(round(base * cap_lo, 4),
                                 round(base * cap_hi, 4)),
            at_discharge_pct=round(hbw_to_pct(at_hbw, lad), 1),
            native_at_discharge_hbw=round(nat_hbw, 4),
            native_at_discharge_pct=round(hbw_to_pct(nat_hbw, lad), 1),
            elite_hbw=round(elite, 4),
            raw_native_at_discharge_hbw=round(float(fit.native_hbw), 4),
            style_flags=fit.flags_at(theta) if armed else {},
            alias_of=(src if src != abbr else None),
            notes=note)
    # calibration families are indivisible aim coordinates (build_plan's final
    # copy, same reason): SG/SSG and NG/SNG must not diverge in θ.
    for source, members in CALIBRATION_FAMILIES.items():
        present = [m for m in members if m in out]
        if len(present) < 2 or source not in out:
            continue
        canonical = out[source]
        for member in present:
            out[member] = dataclasses.replace(
                canonical, weapon=member, impulse=WEAPON_IMPULSE[member],
                alias_of=(source if member != source else None))
    return out


def build_crest_vectors(crest_plans: dict[str, CrestPlan]) -> dict[str, Any]:
    """The emitted decode params: ``attack.crest_theta_vec`` (8,) per impulse-1
    and the SHARED ``attack.crest_hold_ticks``. Explicit-OFF is 0.0 / 0, never
    omission (fail-loud required-params policy, plan §Parameters). H must agree
    across every armed weapon — it is one shared reaction-latency bound."""
    theta = [0.0] * 8
    holds = {p.hold_ticks for p in crest_plans.values() if p.armed}
    if len(holds) > 1:
        raise ValueError(
            f"crest_hold_ticks must be shared across weapons, got {sorted(holds)} "
            "— H is a reaction-latency bound, not weapon physics")
    for p in crest_plans.values():
        if p.armed:
            theta[p.impulse - 1] = round(float(p.theta), 4)
    return {"attack.crest_theta_vec": theta,
            "attack.crest_hold_ticks": int(holds.pop()) if holds else 0}
