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

from qnn.decode_fit.context import (ABBR_TO_MODELNAME, INTERCEPT_WEAPONS,
                                    TRANSFER_ALIAS, WEAPON_IMPULSE)
from qnn.decode_fit.events import EventTable
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
        not event scatter). Falls back to the point prediction when no
        bootstrap draws survived."""
        if self._boot is None or not len(self._boot):
            p = self.predict_hbw(gain, tms)
            return (p, p)
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
        # trigger on the POINT estimate with a 5% margin — cluster-level CIs
        # on 2-pin cells are too wide to ever beat a frontier, and the 4-pin
        # confirmation instrument adjudicates the promise either way (a
        # frontier-measured plan re-promises AT the confirm measurement in
        # both directions; gates.secant_correction).
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
        # aliases ride the corrected source on their own ladder
        for a2, p2 in plans.items():
            if getattr(p2, "alias_of", None) != abbr:
                continue
            lad2 = ladder.get(a2) or {}
            pct2 = round(hbw_to_pct(mf["hbw"], lad2), 1) if lad2 \
                else p2.frontier_pct
            out[a2] = dataclasses.replace(
                p2, gain=round(mf["gain"], 4), alpha=round(mf["alpha"], 4),
                pred_hbw=round(mf["hbw"], 4),
                pred_hbw_ci=(round(mf["hbw"], 4), round(mf["hbw_hi"], 4)),
                band="frontier-measured",
                refused=bool(mf["hbw"] > p2.target_hbw),
                frontier_hbw=round(mf["hbw"], 4), frontier_pct=pct2,
                achieved_pct=pct2,
                notes=(p2.notes + "; measured frontier via alias "
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

def fit_acquisition(cells: list[dict], band: dict, *, target_pct: float = 50.0,
                    min_corr: float = 0.5, n_boot: int = 1000,
                    seed: int = 0) -> dict[str, Any]:
    """tms → throughput inversion onto the human band, with bootstrap CI over
    cells. ``cells`` rows carry {tms, throughput_bits_per_s, n_flicks}. Fails
    loud (``unfittable``) when the lever does not move the axis (the bot-pin
    in-view failure: no target-free flicks) — port of the v1 responsiveness
    guard, CI added. NO CLAMP: the fitted tms is a measured human-match;
    sweep-bound targets flag for extension, never accept a boundary."""
    from qnn.decode_fit.human_refs import pct_to_throughput, throughput_to_pct
    rows = [(float(c["tms"]), float(c["throughput_bits_per_s"]))
            for c in cells if c.get("throughput_bits_per_s") is not None]
    if len({t for t, _ in rows}) < 2:
        raise ValueError(f"acquisition fit needs ≥2 tms levels, got {len(rows)} cells")
    tms_v = np.array([t for t, _ in rows], float)
    tp_v = np.array([p for _, p in rows], float)
    corr = float(np.corrcoef(tms_v, tp_v)[0, 1]) if np.std(tp_v) > 1e-9 else 0.0
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
        "unfittable": bool(corr < min_corr),
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
    alias_of: str | None = None     # SSG/SNG ride SG/NG responses
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
    the plan rides the frontier. SSG/SNG alias SG/NG responses but keep their
    OWN ladders/targets. ``alpha_style_cap`` bounds α (the hold-destruction
    lever) — None = uncapped, adjudicated by the style gate."""
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
            ci = fit.predict_ci(0.0, tms)
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
            ci = afit.floor_ci() if inv["at_floor"] else (pred, pred)

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
