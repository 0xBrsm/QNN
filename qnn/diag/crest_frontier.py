"""Counterfactual crest-capture frontier: what trigger timing is REACHABLE on
the model's own aim trajectory, as a function of enforced firing cadence.

The rung-3 SG objective is ``crest_capture`` (LOWER is better):

    capture = median(hbw at the fired tick)
              / median(hbw over the +/-TRACKING_K core window around fires)

Both terms move under a counterfactual firing pattern — change WHICH ticks are
discharges and the core-window set changes with them — so every policy here
recomputes NUMERATOR AND DENOMINATOR from the same contiguous trace, under the
exact convention ``runs/eval/_decodealigned_8m_verdict.json`` pins:

  * numerator   = ``hbw_win[:, k_pre]``       → hbw AT each discharge tick
  * denominator = ``hbw_win[:, k_pre-K:k_pre+K+1]`` flattened, K =
    ``qnn.decode_fit.events.TRACKING_K`` → for each discharge, the +/-K ticks
    around it, POOLED WITH MULTIPLICITY (overlapping burst windows contribute
    a tick once per discharge that sees it — crest_report.py's flatten, not a
    set union), non-finite dropped, episode-bounded (a discharge near an
    episode edge contributes a short window, exactly like the npz's NaN pad).

Two policy classes are swept against an enforced cadence X:

  (a) MYOPIC — the class a gamma=0 optimizer can represent: with no
      discounting there is no credit for declining a mediocre shot to be ready
      for a better one, so the decision can only read the CURRENT tick. TWO
      members, because the obvious one is not a ceiling:

        a1. bare threshold, fire iff ready and ``hbw <= tau``. MEASURED to
            fire on the LEADING EDGE of every convergence (62% of its shots on
            the rung3k trace still have the local trough ahead of them, mean
            offset +0.88 ticks), scoring crest ABOVE 1.0 — worse than the real
            policy. So it is a member of the class, NOT a bound on it.
        a2. threshold + causal trough gate: also require that alignment has
            stopped improving (``hbw_t >= hbw_{t-1}``). One bit, computed from
            t and t-1, no discounting and no lookahead — squarely inside the
            gamma=0 class, and it removes the leading-edge artifact.

      Neither dominates (a1 wins at very low cadence, where being under tau
      already implies being near the trough), so the reported myopic ceiling
      is the BETTER OF THE TWO at each cadence.

  (b) DP-OPTIMAL — exact dynamic program over the refire horizon. Lagrangian
      form: maximise ``sum over fires of (lam - hbw_t)`` subject to fires being
      >= R ticks apart, with ``lam`` bisected to hit cadence X. This is what a
      gamma>0 optimizer could in principle reach.

      SURROGATE CAVEAT: the DP optimises the SUM of at-fire hbw, which is not
      the median ratio the metric reports (a median is not additive, so no
      exact DP exists for it). The DP is therefore optimal for its own linear
      objective and merely a strong comparator for crest_capture — it is not a
      proven upper bound on capture at a given cadence. Measured: the bounded
      -lookahead rule below at W=9 BEATS the DP on the median ratio, which is
      this caveat showing up in the data rather than a solver bug.

  (c) BOUNDED LOOKAHEAD — fire iff ready, under tau, and no better tick within
      the next W. The DP is CLAIRVOYANT, so ``dp - myopic`` conflates the value
      of WAITING (what gamma>0 buys) with perfect FORESIGHT (what nothing
      buys). Sweeping W separates them: W=0 is exactly the myopic bare
      threshold, and W=R-1 is "wait for the best tick before you'd be ready
      again anyway". This is the honest measure of how much foresight the
      headroom requires.

CLOSED-LOOP CAVEAT (applies to EVERYTHING in this module): the alignment trace
was produced under the ACTUAL policy. A counterfactual firing pattern would in
reality shift the state distribution — kills land at different times, damage
and engagement structure differ, and the aim trajectory itself would change.
These numbers are an upper bound ON A FIXED OBSERVED TRAJECTORY, the same
assumption the frozen-look oracle makes. They are NOT a closed-loop result.

Input is ``<run>/metrics/eval/intercept_trace.npz`` — the contiguous per-tick
align-hbw stream (``qnn.eval.run._write_intercept_trace``). The trace carries
the SAME ``_LeadRuler.hbw`` values the discharge-window instrument samples, so
the reconstruction below reproduces ``intercept_windows.npz`` by construction;
:func:`actual_capture` is the parity check that pins it.

Usage:
  PYTHONPATH=src python3 -m qnn.diag.crest_frontier \\
    --arm bc_seed11=runs/eval/crest_trace_ae2_s11_sg \\
    --arm rung3k=runs/eval/crest_trace_dampedproject_sg_scout \\
    --cadence 0.01,0.02,0.03,0.05,0.077,0.10,0.12,0.15 \\
    --out runs/eval/_crest_frontier.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from qnn.decode_fit.events import TRACKING_K
from qnn.eval.intercept_trace import INTERCEPT_TRACE_NPZ, load_intercept_trace

# Bisection budget for the cadence solve. The realized fire count is a step
# function of tau/lam, so an exact hit is generally impossible; we bisect to
# the tightest achievable bracket and REPORT the realized cadence.
_BISECT_ITERS = 60


@dataclass(frozen=True)
class Trace:
    """Per-tick align-hbw over episode segments, padded to a rectangle.

    ``hbw`` is (n_seg, T_max) float64 with NaN off the segment and NaN on ticks
    with no in-LOS actor (the eval's ``lead_valid=0``); ``fired`` is the actual
    operative-discharge mask; ``valid`` is ``isfinite(hbw)`` — the ENGAGED ticks,
    the only ones a counterfactual policy may fire on and the cadence
    denominator. Segment = one ``(env_idx, episode_ord)`` lane-episode, the same
    unit ``qnn.eval.run``'s window instrument keys its trail/pending on.
    """

    hbw: np.ndarray            # (n_seg, T) float64, NaN off-segment/off-LOS
    fired: np.ndarray          # (n_seg, T) bool — actual operative discharges
    seg_len: np.ndarray        # (n_seg,) int  — real length of each segment
    env_idx: np.ndarray        # (n_seg,) int
    episode: np.ndarray        # (n_seg,) int

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.hbw)

    @property
    def n_engaged(self) -> int:
        return int(self.valid.sum())

    def cadence(self, fire: np.ndarray) -> float:
        """Fires per ENGAGED tick — the human p_fire denominator (a tick with a
        participating in-LOS actor), NOT per wall-clock tick."""
        return float(fire.sum() / max(self.n_engaged, 1))


def load_trace(run_dir: str | Path) -> Trace:
    """Read ``<run_dir>/metrics/eval/intercept_trace.npz`` into a :class:`Trace`.

    Accepts either a run dir or a direct path to the npz. FAILS LOUD on a
    non-contiguous tick column: the whole point of the trace is that the ticks
    the policy DECLINED to fire on are present, so a gap would silently corrupt
    every cooldown simulation downstream.
    """
    z = load_intercept_trace(run_dir)
    return build_trace(
        env_idx=np.asarray(z["env_idx"], dtype=np.int64),
        episode=np.asarray(z["episode"], dtype=np.int64),
        tick=np.asarray(z["tick"], dtype=np.int64),
        hbw=np.asarray(z["hbw"], dtype=np.float64),
        fired=np.asarray(z["fired"], dtype=bool))


def build_trace(env_idx: np.ndarray, episode: np.ndarray, tick: np.ndarray,
                hbw: np.ndarray, fired: np.ndarray) -> Trace:
    """Group flat per-tick rows into padded (n_seg, T_max) segment arrays."""
    order = np.lexsort((tick, episode, env_idx))
    env_idx, episode, tick = env_idx[order], episode[order], tick[order]
    hbw, fired = hbw[order], fired[order]

    key = np.stack([env_idx, episode], axis=1)
    starts = np.flatnonzero(
        np.r_[True, (key[1:] != key[:-1]).any(axis=1)])
    ends = np.r_[starts[1:], len(env_idx)]
    lens = ends - starts
    T = int(lens.max()) if len(lens) else 0

    # `tick` is the GLOBAL eval macro-step, and an ACTIVE lane sees every one
    # of them, so consecutive rows inside a lane-episode must differ by exactly
    # 1. Anything else means ticks were dropped — which would silently corrupt
    # every cooldown simulation downstream, so it fails loud.
    for s, e in zip(starts, ends):
        d = np.diff(tick[s:e])
        if len(d) and not (d == 1).all():
            raise ValueError(
                f"intercept trace is not contiguous in lane {env_idx[s]} "
                f"episode {episode[s]}: tick deltas {sorted(set(d.tolist()))}")

    H = np.full((len(starts), T), np.nan)
    F = np.zeros((len(starts), T), dtype=bool)
    for i, (s, e) in enumerate(zip(starts, ends)):
        H[i, : e - s] = hbw[s:e]
        F[i, : e - s] = fired[s:e]
    return Trace(hbw=H, fired=F, seg_len=lens,
                 env_idx=env_idx[starts], episode=episode[starts])


# ── the metric ────────────────────────────────────────────────────────────────
def _window_pool(hbw: np.ndarray, fire: np.ndarray, k: int) -> np.ndarray:
    """The tracking-core denominator sample: for every fired tick, the +/-k ticks
    around it, POOLED WITH MULTIPLICITY and episode-bounded.

    This is ``hbw_win[:, k_pre-k : k_pre+k+1]`` flattened with non-finite
    dropped — crest_report.py's convention exactly. Overlapping burst windows
    DO contribute a tick more than once (that is the flatten, not a bug); the
    npz's NaN pad at an episode edge is reproduced by the segment boundary.
    """
    T = hbw.shape[1]
    cols = []
    for off in range(-k, k + 1):
        # shifted[i, t] == hbw[i, t + off] — the window slot `off` ticks from
        # the fire tick; out of range stays NaN (the npz's edge pad).
        shifted = np.full_like(hbw, np.nan)
        if off > 0:
            shifted[:, : T - off] = hbw[:, off:]
        elif off < 0:
            shifted[:, -off:] = hbw[:, : T + off]
        else:
            shifted[:] = hbw
        cols.append(shifted[fire])
    pool = np.concatenate(cols) if cols else np.empty(0)
    return pool[np.isfinite(pool)]


def capture(hbw: np.ndarray, fire: np.ndarray, k: int = TRACKING_K) -> dict:
    """``crest_capture`` for an arbitrary fire mask over a padded trace.

    Numerator AND denominator are both recomputed from ``fire`` — the whole
    point (change the discharges and the core window set changes too)."""
    at = hbw[fire]
    at = at[np.isfinite(at)]
    core = _window_pool(hbw, fire, k)
    if not len(at) or not len(core):
        return {"n_discharges": int(len(at)), "n_core_ticks": int(len(core)),
                "crest_capture_median_ratio": None}
    m_at, m_core = float(np.median(at)), float(np.median(core))
    lg_at, lg_core = float(np.exp(np.mean(np.log(at[at > 0])))), \
        float(np.exp(np.mean(np.log(core[core > 0]))))
    return {
        "n_discharges": int(len(at)),
        "n_core_ticks": int(len(core)),
        "at_discharge_hbw_median": m_at,
        "at_discharge_hbw_mean": float(at.mean()),
        "window_core_hbw_median": m_core,
        "crest_capture_median_ratio": (m_at / m_core) if m_core else None,
        "crest_capture_geomean_ratio": (lg_at / lg_core) if lg_core else None,
    }


def capture_from_windows(run_dir: str | Path, k: int = TRACKING_K) -> dict:
    """The CLOSED-LOOP crest, read straight off ``intercept_windows.npz``.

    The same law ``scripts/analysis/crest_report.py`` applies and the number
    the committed verdicts carry: ``hbw_win[:, k_pre]`` over
    ``hbw_win[:, k_pre-k : k_pre+k+1]`` flattened, non-finite dropped.
    Verified to reproduce ``runs/eval/_decodealigned_8m_verdict.json``'s
    0.9138901405611307 (seed11) and 0.8756880946093624 (rung3k scout) exactly.

    This is the arm the offline frontier is VALIDATED AGAINST — it carries no
    fixed-trajectory assumption, because the trajectory really was produced
    under the policy being scored.
    """
    p = Path(run_dir)
    if p.is_dir():
        p = p / "metrics" / "eval" / "intercept_windows.npz"
    with np.load(p, allow_pickle=False) as z:
        kp = int(z["k_pre"])
        w = np.asarray(z["hbw_win"], dtype=np.float64)
    at = w[:, kp]
    at = at[np.isfinite(at)]
    core = w[:, kp - k: kp + k + 1].ravel()
    core = core[np.isfinite(core)]
    if not len(at) or not len(core):
        return {"n_discharges": int(len(at)), "crest_capture_median_ratio": None}
    return {
        "n_discharges": int(len(at)),
        "n_core_ticks": int(len(core)),
        "at_discharge_hbw_median": float(np.median(at)),
        "window_core_hbw_median": float(np.median(core)),
        "crest_capture_median_ratio": float(np.median(at) / np.median(core)),
    }


def actual_capture(tr: Trace, k: int = TRACKING_K) -> dict:
    """Capture under the run's OWN discharges — the parity check against the
    same run's ``intercept_windows.npz`` verdict number."""
    return capture(tr.hbw, tr.fired & np.isfinite(tr.hbw), k)


# ── policy classes ────────────────────────────────────────────────────────────
def myopic_fire(hbw: np.ndarray, refire: int, tau: float,
                forced: np.ndarray | None = None,
                trough_gate: bool = False) -> np.ndarray:
    """(a) Fire iff READY and ``hbw <= tau``. Cooldown simulated forward.

    The gamma=0 representable ceiling: the decision uses only the CURRENT
    tick's alignment. Vectorised across segments (the recursion is sequential
    in t only), so a 240-episode trace costs T numpy ops, not n_seg*T.

    ``trough_gate`` adds ONE causal bit: fire only once alignment has stopped
    improving (``hbw_t >= hbw_{t-1}``). MEASURED NECESSITY, not a refinement —
    the bare threshold fires on the LEADING EDGE of every convergence (62% of
    its shots on the rung3k trace still have the local trough ahead of them,
    mean offset +0.88 ticks), which drives crest_capture ABOVE 1.0: the fired
    tick is worse-aligned than its own +/-4 neighbourhood. That makes the bare
    rule a poor stand-in for the gamma=0 class rather than a ceiling on it —
    the real policy beats it. The gate uses only t and t-1, needs no
    discounting and no lookahead, so it stays inside the gamma=0 class while
    removing the leading-edge artifact. Report BOTH.

    ``forced`` (see :func:`blind_fire_mask`) are ticks where the observed
    policy discharged with NO in-LOS actor. They burn the cooldown but can
    never earn crest, so a counterfactual that ignores them is handed free
    budget; forcing them holds that behaviour fixed at the observed one and
    optimises only the trigger on engaged ticks. A forced shot that lands in
    cooldown simply does not happen, exactly as the engine would refuse it.
    """
    n, T = hbw.shape
    fire = np.zeros((n, T), dtype=bool)
    cool = np.zeros(n, dtype=np.int64)          # ticks until ready
    ok = np.isfinite(hbw) & (hbw <= tau)
    if trough_gate:
        prev = np.full_like(hbw, np.inf)
        prev[:, 1:] = hbw[:, :-1]
        # NaN previous tick (no LOS last tick) => no trend to read; allow.
        turned = ~(hbw < prev)                  # False only while improving
        ok = ok & (turned | ~np.isfinite(prev))
    if forced is not None:
        ok = ok | forced
    for t in range(T):
        go = ok[:, t] & (cool <= 0)
        fire[:, t] = go
        # refire-1, not refire: firing at t leaves the NEXT allowed fire at
        # t+refire, so exactly refire-1 intervening ticks are blocked. The
        # measured SG gap floor (10) is the pinned check.
        cool = np.where(go, refire - 1, np.maximum(cool - 1, 0))
    return fire


def lookahead_fire(hbw: np.ndarray, refire: int, tau: float, window: int,
                   forced: np.ndarray | None = None) -> np.ndarray:
    """(c) Bounded-lookahead option value: fire iff ready, ``hbw <= tau``, AND
    no strictly better tick within the next ``window`` ticks.

    This exists because the DP is CLAIRVOYANT — it sees the whole episode
    exactly — so ``dp - myopic`` conflates two different advantages: the value
    of WAITING (what gamma>0 buys) and perfect FORESIGHT (what nothing buys).
    This rule isolates the first: it needs only a ``window``-tick prediction of
    alignment, and it is precisely the "fire now vs hold for a better tick
    before I'd be ready again anyway" decision the refire cooldown creates.

    ``window = 0`` degenerates to :func:`myopic_fire`'s bare threshold, so
    sweeping the window traces a continuous path from the myopic class to the
    option-value class and says how much foresight the headroom actually
    requires.
    """
    n, T = hbw.shape
    # best alignment available in (t, t+window]; +inf where nothing is coming
    fwd = np.full((n, T), np.inf)
    for off in range(1, max(window, 0) + 1):
        shifted = np.full((n, T), np.inf)
        if off < T:
            nxt = hbw[:, off:]
            shifted[:, : T - off] = np.where(np.isfinite(nxt), nxt, np.inf)
        fwd = np.minimum(fwd, shifted)

    fire = np.zeros((n, T), dtype=bool)
    cool = np.zeros(n, dtype=np.int64)
    ok = np.isfinite(hbw) & (hbw <= tau) & (hbw <= fwd)
    if forced is not None:
        ok = ok | forced
    for t in range(T):
        go = ok[:, t] & (cool <= 0)
        fire[:, t] = go
        cool = np.where(go, refire - 1, np.maximum(cool - 1, 0))
    return fire


def blind_fire_mask(tr: Trace) -> np.ndarray:
    """Observed operative discharges on ticks with NO in-LOS actor.

    These never reach ``intercept_windows.npz`` (it gates on ``lead_valid``),
    but they DO consume the refire cooldown — on the rung3k scout they are
    22.7% of all discharges. Any counterfactual that fires only on engaged
    ticks therefore enjoys a fire budget the real policy never had, which
    biases the offline frontier OPTIMISTIC. Feed this to the solvers'
    ``forced`` argument to remove that bias.
    """
    return tr.fired & ~np.isfinite(tr.hbw)


def dp_fire(hbw: np.ndarray, refire: int, lam: float,
            forced: np.ndarray | None = None) -> np.ndarray:
    """(b) Exact DP over the refire horizon for ``max sum(lam - hbw_t)``.

        V[t] = max( V[t+1],  (lam - hbw_t) + V[t+R] )

    Backward pass then forward backtrack; both vectorised across segments.
    ``lam`` is the shadow price of a shot — the option value the myopic rule
    cannot represent lives entirely in the ``V[t+1]`` branch (decline now, stay
    ready for a better tick within the next R).
    """
    n, T = hbw.shape
    V = np.zeros((n, T + refire + 1))
    take = np.zeros((n, T), dtype=bool)
    fin = np.isfinite(hbw)
    gain = np.where(fin, lam - np.nan_to_num(hbw, nan=0.0), -np.inf)
    if forced is not None:
        # A forced (blind) shot is not a decision and earns no lambda credit:
        # it is not an engaged-tick discharge, so it is outside the cadence
        # budget. It only costs the cooldown.
        gain = np.where(forced, 0.0, gain)
    for t in range(T - 1, -1, -1):
        v_fire = gain[:, t] + V[:, min(t + refire, T)]
        v_skip = V[:, t + 1]
        take[:, t] = v_fire > v_skip
        if forced is not None:
            take[:, t] |= forced[:, t]       # no choice at a forced tick
            V[:, t] = np.where(forced[:, t], v_fire, np.maximum(v_fire, v_skip))
            continue
        V[:, t] = np.maximum(v_fire, v_skip)

    fire = np.zeros((n, T), dtype=bool)
    cursor = np.zeros(n, dtype=np.int64)
    for t in range(T):
        at = cursor == t
        go = at & take[:, t]
        fire[:, t] = go
        cursor = np.where(go, t + refire, np.where(at, t + 1, cursor))
    return fire


def retime_oracle_fire(tr: Trace, k: int = TRACKING_K) -> np.ndarray:
    """The frozen-look oracle's fire set, rebuilt on the trace: every ACTUAL
    discharge re-timed to the argmin hbw within its own +/-k window.

    TWO deviations from the published .473, both deliberate and both reported
    separately by :func:`oracle_published`:

      1. the published convention keeps the ORIGINAL discharges' denominator;
         scoring this mask with :func:`capture` moves the denominator with the
         re-timed fires, which is the convention the rest of this module uses;
      2. the marginal is NOT preserved here. Two discharges in one burst can
         re-time onto the SAME tick, and a boolean mask collapses them — so
         ``fire.sum() <= tr.fired.sum()``. The published statistic takes a
         median over a LIST of per-discharge minima and keeps the duplicates.
         Check ``n_discharges`` in the returned capture before comparing.

    It also ignores the refire cooldown entirely (re-timed fires can land
    closer than R apart), exactly as the published oracle does.
    """
    n, T = tr.hbw.shape
    fire = np.zeros((n, T), dtype=bool)
    src = tr.fired & np.isfinite(tr.hbw)
    for i, t in zip(*np.nonzero(src)):
        lo, hi = max(0, t - k), min(int(tr.seg_len[i]), t + k + 1)
        w = tr.hbw[i, lo:hi]
        if np.isfinite(w).any():
            fire[i, lo + int(np.nanargmin(w))] = True
    return fire


def oracle_published(tr: Trace, k: int = TRACKING_K) -> float | None:
    """The oracle EXACTLY as pre-registered (scripts/analysis/bank_verdict.py):
    median(row-wise nanmin over the +/-k window) / median(the SAME rows' core,
    center included) — denominator NOT re-derived from the re-timed fires."""
    src = tr.fired & np.isfinite(tr.hbw)
    mins, core = [], _window_pool(tr.hbw, src, k)
    for i, t in zip(*np.nonzero(src)):
        lo, hi = max(0, t - k), min(int(tr.seg_len[i]), t + k + 1)
        w = tr.hbw[i, lo:hi]
        if np.isfinite(w).any():
            mins.append(float(np.nanmin(w)))
    if not mins or not len(core):
        return None
    return float(np.median(mins) / np.median(core))


# ── cadence solve ─────────────────────────────────────────────────────────────
def _solve(tr: Trace, target: float, make_fire, lo: float, hi: float):
    """Bisect a monotone knob (tau or lam) until the realized cadence brackets
    ``target``. Both knobs are monotone increasing in fire count.

    Cadence counts ENGAGED-tick fires only, so forced blind shots (which carry
    NaN hbw) are excluded from the budget automatically."""
    best = None
    eng = np.isfinite(tr.hbw)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        fire = make_fire(mid)
        c = tr.cadence(fire & eng)
        if best is None or abs(c - target) < abs(best[1] - target):
            best = (mid, c, fire)
        if c < target:
            lo = mid
        else:
            hi = mid
    return best


def frontier(tr: Trace, cadences: Sequence[float], refire: int,
             k: int = TRACKING_K, forced: np.ndarray | None = None
             ) -> list[dict]:
    """The (a)/(b) frontier: for each target cadence, solve both policies to
    that cadence and score BOTH with the same recomputed crest metric."""
    hi_tau = float(np.nanmax(tr.hbw)) if np.isfinite(tr.hbw).any() else 1.0
    eng = np.isfinite(tr.hbw)
    rows = []
    for x in cadences:
        b = _solve(tr, x, lambda t: myopic_fire(tr.hbw, refire, t, forced),
                   0.0, hi_tau)
        g = _solve(tr, x, lambda t: myopic_fire(tr.hbw, refire, t, forced,
                                                trough_gate=True),
                   0.0, hi_tau)
        d = _solve(tr, x, lambda l: dp_fire(tr.hbw, refire, l, forced),
                   0.0, hi_tau)
        cb = capture(tr.hbw, b[2] & eng, k)
        cg = capture(tr.hbw, g[2] & eng, k)
        cd = capture(tr.hbw, d[2] & eng, k)
        # the MYOPIC CLASS ceiling is the better of its two members at this
        # cadence — neither variant dominates (the bare rule wins at very low
        # cadence, where being under tau already implies being near the trough)
        best = min((cb, b), (cg, g),
                   key=lambda p: p[0]["crest_capture_median_ratio"])
        rows.append({
            "target_cadence": float(x),
            "myopic_bare": {"tau": b[0], "cadence": b[1], **cb},
            "myopic_trough_gated": {"tau": g[0], "cadence": g[1], **cg},
            "myopic_best": {"variant": ("bare" if best[1] is b
                                        else "trough_gated"),
                            "cadence": best[1][1], **best[0]},
            "dp": {"lam": d[0], "cadence": d[1], **cd},
            "dp_minus_myopic_best": (cd["crest_capture_median_ratio"]
                                     - best[0]["crest_capture_median_ratio"]),
        })
    return rows


# ── refire measurement ────────────────────────────────────────────────────────
def measure_refire(tr: Trace, mass_floor: float = 0.01) -> dict:
    """Infer the refire interval from OBSERVED discharge spacing rather than
    assuming it.

    ``refire`` is the SMALLEST gap that carries real mass (>= ``mass_floor`` of
    all gaps), NOT the raw minimum. The raw minimum is not robust: the BC
    seed-11 arm has exactly ONE gap of 5 ticks in 23,189 (4.3e-05 of them),
    and taking it would hand the counterfactual twice the fire opportunities
    the engine actually allows. ``frac_below_refire`` reports the mass that
    estimator discards, so a genuinely bimodal weapon cannot slip through.
    """
    gaps: list[int] = []
    for i in range(tr.fired.shape[0]):
        t = np.flatnonzero(tr.fired[i, : int(tr.seg_len[i])])
        if len(t) > 1:
            gaps.extend(np.diff(t).tolist())
    if not gaps:
        return {"n_gaps": 0}
    g = np.asarray(gaps)
    vals, cnt = np.unique(g, return_counts=True)
    frac = cnt / cnt.sum()
    solid = vals[frac >= mass_floor]
    refire = int(solid.min()) if len(solid) else int(vals[cnt.argmax()])
    return {
        "n_gaps": int(len(g)),
        "refire": refire,
        "mass_floor": mass_floor,
        "raw_min_gap": int(g.min()),
        "modal_gap": int(vals[cnt.argmax()]),
        "frac_at_refire": float((g == refire).mean()),
        "frac_below_refire": float((g < refire).mean()),
        "median_gap": float(np.median(g)),
        "gap_hist": {int(v): int(c) for v, c in zip(vals[:20], cnt[:20])},
    }


# ── bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap_capture(tr: Trace, fire: np.ndarray, n_boot: int = 2000,
                      seed: int = 17, k: int = TRACKING_K) -> dict:
    """Percentile CI on the capture ratio, resampling EPISODE CLUSTERS.

    DIFFERS FROM THE COMMITTED VERDICTS, deliberately: those resample discharge
    ROWS (crest_report.py's cheap substitute — ``intercept_windows.npz`` has no
    cluster key its script clusters on). Discharges within one lane-episode are
    strongly dependent, so row resampling understates the interval; the trace
    carries ``(env_idx, episode)`` so the cluster bootstrap is available here.
    The POLICY IS HELD FIXED across draws (tau/lam solved once on the full
    data) — the CI is on the metric, not on the policy search.
    """
    rng = np.random.default_rng(seed)
    n = tr.hbw.shape[0]
    draws = []
    for _ in range(n_boot):
        sel = rng.integers(0, n, size=n)
        r = capture(tr.hbw[sel], fire[sel], k)
        v = r["crest_capture_median_ratio"]
        if v is not None:
            draws.append(v)
    if not draws:
        return {"n_boot": n_boot, "n_valid": 0}
    a = np.asarray(draws)
    return {"lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)),
            "n_boot": n_boot, "n_valid": int(len(a)),
            "cluster_unit": "(env_idx, episode) lane-episode"}


def bootstrap_gap(tr: Trace, fire_a: np.ndarray, fire_b: np.ndarray,
                  n_boot: int = 2000, seed: int = 17,
                  k: int = TRACKING_K) -> dict:
    """PAIRED cluster CI on ``capture(b) - capture(a)``: both policies are
    scored on the SAME resampled episodes each draw (they run on one trajectory,
    so the pairing is real — unlike crest_report.py's two-run comparison)."""
    rng = np.random.default_rng(seed)
    n = tr.hbw.shape[0]
    draws = []
    for _ in range(n_boot):
        sel = rng.integers(0, n, size=n)
        h = tr.hbw[sel]
        ra = capture(h, fire_a[sel], k)["crest_capture_median_ratio"]
        rb = capture(h, fire_b[sel], k)["crest_capture_median_ratio"]
        if ra is not None and rb is not None:
            draws.append(rb - ra)
    if not draws:
        return {"n_boot": n_boot, "n_valid": 0}
    a = np.asarray(draws)
    return {"delta_lo": float(np.percentile(a, 2.5)),
            "delta_hi": float(np.percentile(a, 97.5)),
            "delta_mean": float(a.mean()),
            "excludes_zero": bool(np.percentile(a, 2.5) > 0
                                  or np.percentile(a, 97.5) < 0),
            "n_boot": n_boot, "n_valid": int(len(a)),
            "cluster_unit": "(env_idx, episode) lane-episode, PAIRED"}


# ── CLI ───────────────────────────────────────────────────────────────────────
def analyze_arm(run_dir: str | Path, cadences: Sequence[float],
                headline: float, refire: int | None = None,
                n_boot: int = 2000, seed: int = 17,
                charge_blind_fires: bool = True) -> dict:
    tr = load_trace(run_dir)
    rf = measure_refire(tr)
    R = int(refire if refire is not None else rf.get("refire", 10))
    eng = np.isfinite(tr.hbw)
    blind = blind_fire_mask(tr)
    forced = blind if charge_blind_fires else None
    out: dict = {
        "run_dir": str(run_dir),
        "n_segments": int(tr.hbw.shape[0]),
        "n_ticks": int(tr.seg_len.sum()),
        "n_engaged_ticks": tr.n_engaged,
        "engaged_frac": float(tr.n_engaged / max(int(tr.seg_len.sum()), 1)),
        "refire_measured": rf,
        "refire_used": R,
        "blind_fires": {
            "n_total_discharges": int(tr.fired.sum()),
            "n_on_engaged_ticks": int((tr.fired & eng).sum()),
            "n_no_los_actor": int(blind.sum()),
            "frac_of_discharges": float(blind.sum() / max(tr.fired.sum(), 1)),
            "charged_to_counterfactual": bool(charge_blind_fires),
            "note": "discharges with no in-LOS actor never reach the crest "
                    "instrument but do burn the refire cooldown; charging "
                    "them keeps the counterfactual's fire budget honest",
        },
        "actual": {**actual_capture(tr),
                   "cadence": tr.cadence(tr.fired & eng)},
        "oracle_published_convention": oracle_published(tr),
        "oracle_retimed_recomputed_denominator":
            capture(tr.hbw, retime_oracle_fire(tr)),
        "frontier": frontier(tr, cadences, R, forced=forced),
    }
    hi_tau = float(np.nanmax(tr.hbw))
    b = _solve(tr, headline, lambda t: myopic_fire(tr.hbw, R, t, forced),
               0.0, hi_tau)
    g = _solve(tr, headline, lambda t: myopic_fire(tr.hbw, R, t, forced,
                                                  trough_gate=True),
               0.0, hi_tau)
    d = _solve(tr, headline, lambda l: dp_fire(tr.hbw, R, l, forced),
               0.0, hi_tau)
    fb, fg, fd = b[2] & eng, g[2] & eng, d[2] & eng
    # How much FORESIGHT does the headroom actually need? W=0 is the myopic
    # class; W=refire-1 is "wait for the best tick before you'd be ready
    # anyway". This separates option value from the DP's clairvoyance.
    out["lookahead_horizon"] = []
    for W in (0, 1, 2, 3, 5, R - 1, 2 * R):
        s = _solve(tr, headline,
                   lambda t, _w=W: lookahead_fire(tr.hbw, R, t, _w, forced),
                   0.0, hi_tau)
        out["lookahead_horizon"].append({
            "window_ticks": int(W), "cadence": s[1],
            **capture(tr.hbw, s[2] & eng),
        })
    cb, cg = capture(tr.hbw, fb), capture(tr.hbw, fg)
    fm, cm, which = ((fb, cb, "bare")
                     if cb["crest_capture_median_ratio"]
                     <= cg["crest_capture_median_ratio"]
                     else (fg, cg, "trough_gated"))
    out["headline"] = {
        "cadence_target": headline,
        "myopic_bare": {"cadence": b[1], **cb,
                        "ci95": bootstrap_capture(tr, fb, n_boot, seed)},
        "myopic_trough_gated": {"cadence": g[1], **cg,
                                "ci95": bootstrap_capture(tr, fg, n_boot, seed)},
        "myopic_best": {"variant": which, **cm},
        "dp": {"cadence": d[1], **capture(tr.hbw, fd),
               "ci95": bootstrap_capture(tr, fd, n_boot, seed)},
        # THE headline: how much crest is reachable ONLY by a policy that can
        # value waiting, over the best rule that cannot.
        "dp_minus_myopic_best": bootstrap_gap(tr, fm, fd, n_boot, seed),
    }
    return out


# Human SG+SSG AIMED discharge rate per ENGAGED-LOS tick (entity_recency==0 /
# AlignHbw.all_los — the SAME population `cadence_per_engaged_tick` below
# counts, no target_probs engagement label required), and the +/-15% realism
# gate the rung-3 program judges cadence against. Crest bought by firing
# outside this band is not a promotable result.
#
# THIS IS NOT qnn.human.op_attack's 0.077164/tick. That number is measured on
# a NARROWER population (LOS *and* a target_probs-labeled engagement —
# 186,846 human ticks) and belongs to qnn.ppo.pfire_target / the live-pins
# forced-cadence fit, whose bot-side masks share that same narrower
# population — it is the wrong ruler here by construction. Gating this
# module's engaged-tick cadence against it (or gating wall-clock
# `shots_fired_mean` against either number) is the three-denominator
# mis-specification agents/plans/blind-fire-cadence.md documents: the pure-LOS
# population is 1.72x larger than op_attack's, so 0.077164 reads a
# correctly-scoped bot as over-firing by ~48%. Measured here on the pure-LOS
# population (qnn.human.blind_fire, qwd_v4d_v3vis val, SG+SSG pooled,
# aimed = non-blind discharges): 1.0457/s = 0.05229/tick.
HUMAN_RATE_PER_ENGAGED_TICK = 0.05229
CADENCE_GATE = 0.15


def sweep_row(run_dir: str | Path, k: int = TRACKING_K,
              human_rate_per_engaged_tick: float = HUMAN_RATE_PER_ENGAGED_TICK,
              bias_index: int = 1) -> dict:
    """One closed-loop decode-sweep point: the trigger metric plus the WORLD
    RESULTS it was bought with.

    ``human_rate_per_engaged_tick`` defaults to the SG+SSG pooled rate and MUST
    be overridden for any other weapon pin, with that family's aimed rate from
    the pinned collect's ``_blind_fire_byweapon.json`` (same pure-LOS
    population; e.g. RL 0.7602/s / tick_hz). ``bias_index`` is the
    ``attack.fire_bias_vec`` slot of the pinned weapon (subject order; SG=1).

    Crest capture alone cannot answer "can we just turn the bias knob and move
    on?" — a more selective trigger trivially improves crest by firing only on
    easy frames, so the question is what it costs in cadence and what it does
    to accuracy/hits/frags/return. All six come from the run's own artifacts:
    crest from ``intercept_windows.npz``, the rest from ``eval_summary.json``.

    The cadence GATE is discharges per pure-LOS engaged tick
    (``cadence_per_engaged_tick``, from ``engine_los_attack_by_lead_angle``'s
    ``n_ticks``) against ``HUMAN_RATE_PER_ENGAGED_TICK`` — one population on
    both sides of the comparison. ``shots_fired_mean`` (per WALL-CLOCK tick)
    is reported for visibility only and is NEVER gated: the arena box's
    engagement occupancy (~79%) has no reason to match a real map's, so a
    wall-clock rate deviating from an engaged-tick human rate says nothing
    about firing discipline — see the constant's docstring above for the
    mechanically-measured size of the mistake this used to be.
    """
    p = Path(run_dir)
    s = json.loads((p / "metrics" / "eval" / "eval_summary.json").read_text())
    cap = capture_from_windows(p, k)
    los = s.get("engine_los_attack_by_lead_angle") or {}
    engaged = sum(int(v["n_ticks"]) for v in los.values())
    dec = json.loads((p / "config" / "decode.json").read_text())
    cadence_engaged = (cap["n_discharges"] / engaged) if engaged else None
    return {
        "run_dir": str(p),
        "fire_bias": float(dec["params"]["attack.fire_bias_vec"][bias_index]),
        "bias_index": bias_index,
        "crest_capture": cap["crest_capture_median_ratio"],
        "n_discharges": cap["n_discharges"],
        "cadence_per_engaged_tick": cadence_engaged,
        "wall_clock_rate_per_tick_informational": s.get("shots_fired_mean"),
        "human_rate_per_engaged_tick": human_rate_per_engaged_tick,
        "cadence_dev_vs_human": (
            (cadence_engaged - human_rate_per_engaged_tick)
            / human_rate_per_engaged_tick
            if cadence_engaged is not None else None),
        "within_cadence_gate": (
            abs(cadence_engaged - human_rate_per_engaged_tick)
            / human_rate_per_engaged_tick <= CADENCE_GATE
            if cadence_engaged is not None else None),
        "engaged_ticks": engaged,
        "accuracy": s.get("accuracy"),
        "hits_per_episode": s.get("episode_hit_count_mean"),
        "frags_per_episode": s.get("frags_mean"),
        "mean_episode_return": s.get("mean_episode_return"),
        "damage_per_episode": s.get("episode_damage_dealt_mean"),
        "deaths_per_episode": s.get("deaths_mean"),
        "at_discharge_hbw_median": cap.get("at_discharge_hbw_median"),
    }


def sweep_crosscheck(points: Mapping[str, str], reference: str | Path,
                     refire: int | None = None,
                     k: int = TRACKING_K) -> dict:
    """Validate the OFFLINE myopic curve against the CLOSED-LOOP decode sweep.

    Decode thresholding IS a myopic threshold — the decode fires when
    ``fire_score - l0' > 0``, and an ``attack.fire_bias_vec`` offset slides
    that threshold — so each bias point is one realized draw from the same
    policy class the offline myopic curve traces. Comparing them at MATCHED
    CADENCE is the check on the fixed-trajectory assumption.

    WHAT THE MYOPIC COMPARISON DOES AND DOES NOT PROVE. The offline myopic
    rules are two SPECIFIC hand-built functions of (hbw_t, hbw_{t-1}); the
    learned policy is an arbitrary function of the whole observation. So the
    myopic curves bound neither the gamma=0 class from above nor the closed
    loop, and the learned policy beating them is expected rather than alarming
    — crest_capture's DENOMINATOR depends on the local SHAPE of the alignment
    excursion, and the model can read that shape off target velocity and range
    while a threshold on the level cannot. (An earlier revision of this
    docstring claimed the opposite; it was a mis-derivation.)

    THE REAL INTEGRITY CHECK is the option-value frontier. The DP is
    clairvoyant and the W=R-1 lookahead sees the whole refire window, so on a
    FIXED trajectory nothing realizable should beat them at matched cadence.
    A closed-loop point that does indicts the fixed-trajectory assumption —
    that is the flag this function raises.

    Each point is scored at its own realized cadence against the offline
    policies on its OWN trace and on the REFERENCE trace, which separates
    trajectory drift from rule quality.
    """
    ref = load_trace(reference)
    R = int(refire if refire is not None else
            measure_refire(ref).get("refire", 10))
    hi_ref = float(np.nanmax(ref.hbw))
    rows = []
    for name, run_dir in points.items():
        closed = capture_from_windows(run_dir, k)
        own = load_trace(run_dir)
        e_own = np.isfinite(own.hbw)
        f_own, f_ref = blind_fire_mask(own), blind_fire_mask(ref)
        hi_own = float(np.nanmax(own.hbw))
        x = own.cadence(own.fired & e_own)

        def _at(tr_, eng_, forced_, hi_, make):
            s = _solve(tr_, x, make, 0.0, hi_)
            return capture(tr_.hbw, s[2] & eng_, k)["crest_capture_median_ratio"]

        c_closed = closed["crest_capture_median_ratio"]
        own_bare = _at(own, e_own, f_own, hi_own,
                       lambda t: myopic_fire(own.hbw, R, t, f_own))
        own_gate = _at(own, e_own, f_own, hi_own,
                       lambda t: myopic_fire(own.hbw, R, t, f_own, True))
        own_myo = min(own_bare, own_gate)
        own_la1 = _at(own, e_own, f_own, hi_own,
                      lambda t: lookahead_fire(own.hbw, R, t, 1, f_own))
        own_laR = _at(own, e_own, f_own, hi_own,
                      lambda t: lookahead_fire(own.hbw, R, t, R - 1, f_own))
        own_dp = _at(own, e_own, f_own, hi_own,
                     lambda l: dp_fire(own.hbw, R, l, f_own))
        ref_myo = min(
            _at(ref, np.isfinite(ref.hbw), f_ref, hi_ref,
                lambda t: myopic_fire(ref.hbw, R, t, f_ref)),
            _at(ref, np.isfinite(ref.hbw), f_ref, hi_ref,
                lambda t: myopic_fire(ref.hbw, R, t, f_ref, True)))

        best_offline = min(own_laR, own_dp)
        rows.append({
            "point": name,
            "run_dir": str(run_dir),
            "realized_cadence": x,
            "closed_loop_crest": c_closed,
            "offline_myopic_best_own_trace": own_myo,
            "offline_myopic_best_reference_trace": ref_myo,
            "offline_lookahead_w1_own_trace": own_la1,
            "offline_lookahead_wR_own_trace": own_laR,
            "offline_dp_own_trace": own_dp,
            "closed_minus_myopic_best": c_closed - own_myo,
            "closed_minus_best_offline": c_closed - best_offline,
            # only THIS can indict the fixed-trajectory counterfactual
            "beats_option_value_frontier": bool(c_closed < best_offline),
            "n_discharges_closed": closed["n_discharges"],
        })
    bad = [r["point"] for r in rows if r["beats_option_value_frontier"]]
    return {
        "refire_used": R,
        "reference": str(reference),
        "points": rows,
        "violations": bad,
        "verdict": (
            "OK — no closed-loop point beats the option-value frontier "
            "(clairvoyant DP / full-window lookahead) on its own trace, so "
            "the fixed-trajectory counterfactual is not being contradicted. "
            "The learned policy DOES beat the hand-built myopic rules, which "
            "is expected: it reads excursion shape, they read only level."
            if not bad else
            f"SUSPECT — closed-loop beat the CLAIRVOYANT offline frontier at "
            f"{bad}. Nothing realizable can do that on a fixed trajectory, so "
            "either the trajectory really did change under the counterfactual "
            "or the cadence matching is wrong. Do not report the offline "
            "frontier as an upper bound until this is explained."),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=RUN_DIR")
    ap.add_argument("--cadence", default="0.01,0.02,0.03,0.05,0.06,0.077,"
                                         "0.09,0.10,0.12,0.15")
    ap.add_argument("--headline", type=float, default=0.077)
    ap.add_argument("--refire", type=int, default=None,
                    help="override the measured refire interval (ticks)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--crosscheck", action="append", default=None,
                    metavar="NAME=RUN_DIR",
                    help="closed-loop decode-sweep point to validate the "
                         "offline myopic curve against")
    ap.add_argument("--crosscheck-reference", default=None,
                    help="trace the offline curve is drawn on (the bias=0 arm)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cad = [float(x) for x in a.cadence.split(",") if x.strip()]
    res = {"convention": {
        "crest": "median(hbw at fire) / median(pooled +/-TRACKING_K windows "
                 "around fires), both recomputed per policy; matches "
                 "runs/eval/_decodealigned_8m_verdict.json's at_discharge_col "
                 "/ tracking_core_cols",
        "cadence": "fires per ENGAGED tick (finite hbw = lead_valid), the "
                   "human p_fire denominator",
        "bootstrap": "episode-cluster resampling (the verdicts resample "
                     "discharge ROWS); policy held fixed across draws",
        "closed_loop": "UPPER BOUND ON A FIXED OBSERVED TRAJECTORY — the "
                       "trace was generated under the actual policy; a "
                       "counterfactual trigger would change the state "
                       "distribution. Not a closed-loop result.",
        "dp_surrogate": "the DP maximises sum(lam - hbw) at fires, not the "
                        "median ratio; optimal for its own linear objective "
                        "only",
    }, "arms": {}}
    for spec in a.arm:
        name, _, run_dir = spec.partition("=")
        res["arms"][name] = analyze_arm(run_dir, cad, a.headline, a.refire,
                                        a.n_boot, a.seed)
        arm = res["arms"][name]
        h = arm["headline"]
        print(f"{name}: actual crest="
              f"{arm['actual']['crest_capture_median_ratio']:.4f} "
              f"@ cadence {arm['actual']['cadence']:.5f}  R={arm['refire_used']}")
        print(f"  @{a.headline}: myopic_bare="
              f"{h['myopic_bare']['crest_capture_median_ratio']:.4f} "
              f"trough_gated={h['myopic_trough_gated']['crest_capture_median_ratio']:.4f} "
              f"best={h['myopic_best']['crest_capture_median_ratio']:.4f} "
              f"dp={h['dp']['crest_capture_median_ratio']:.4f}")
        print(f"  GAP dp-myopic_best = "
              f"{h['dp_minus_myopic_best'].get('delta_mean'):+.4f} "
              f"[{h['dp_minus_myopic_best'].get('delta_lo'):+.4f}, "
              f"{h['dp_minus_myopic_best'].get('delta_hi'):+.4f}]")
    if a.crosscheck:
        pts = dict(s.partition("=")[::2] for s in a.crosscheck)
        res["sweep"] = sorted((sweep_row(d) for d in pts.values()),
                              key=lambda r: r["fire_bias"])
        print(f"{'bias':>6} {'crest':>7} {'rate/eng':>10} {'vs_human':>9} "
              f"{'gate':>6} {'acc':>6} {'hits/ep':>8} {'frags/ep':>9} "
              f"{'return':>9}")
        for r in res["sweep"]:
            print(f"{r['fire_bias']:+6.2f} {r['crest_capture']:7.4f} "
                  f"{r['cadence_per_engaged_tick']:10.6f} "
                  f"{100 * r['cadence_dev_vs_human']:+8.1f}% "
                  f"{'OK' if r['within_cadence_gate'] else 'BREACH':>6} "
                  f"{r['accuracy']:6.3f} {r['hits_per_episode']:8.2f} "
                  f"{r['frags_per_episode']:9.3f} "
                  f"{r['mean_episode_return']:9.1f}")
        res["sweep_crosscheck"] = sweep_crosscheck(
            pts, a.crosscheck_reference, a.refire)
        print("crosscheck:", res["sweep_crosscheck"]["verdict"])
        print(f"  {'pt':>6} {'X':>8} {'closed':>8} {'myo_best':>9} "
              f"{'LA W=1':>8} {'LA W=R-1':>9} {'DP':>8}")
        for r in res["sweep_crosscheck"]["points"]:
            print(f"  {r['point']:>6} {r['realized_cadence']:8.5f} "
                  f"{r['closed_loop_crest']:8.4f} "
                  f"{r['offline_myopic_best_own_trace']:9.4f} "
                  f"{r['offline_lookahead_w1_own_trace']:8.4f} "
                  f"{r['offline_lookahead_wR_own_trace']:9.4f} "
                  f"{r['offline_dp_own_trace']:8.4f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
