"""Channel-agnostic temporal-statistic + distance functions for human-likeness eval.

The human-likeness objective for QNN's control heads is DISTRIBUTIONAL: the
generated action stream should match a human's *temporal statistics* — dwell
(hold-run) durations, switch-rate, inter-event intervals, turn magnitude — not
per-frame accuracy. Per-frame metrics are momentum-capped and mislead under the
88-99% frame-to-frame autocorrelation these streams carry (see
src/docs/persistence-and-changepoints.md).

This module is the reusable scoring kernel for that suite. It is channel-agnostic:
everything operates on a per-frame discrete label sequence plus an episode/segment
boundary mask. A "run" (hold) is a maximal block of identical labels that lies
entirely inside one contiguous in-distribution segment — runs never span an
episode boundary or a segment-mask gap.

Pure numpy + scipy.stats. No torch (so the human-reference half runs locally).

Distances:
  * wasserstein_distance (EMD, scipy) — the headline 1-D distance, native units.
  * ks_2samp — KS statistic + p-value for a 1-D two-sample test.
  * mmd2_rbf / mmd2_rbf_permutation_test — RBF-kernel MMD^2 (unbiased); accepts
    1-D samples or (n, d) feature-vector samples. The permutation variant adds
    a permutation null -> p-value; mmd2_rbf is the bare statistic for callers
    that bring their own null (e.g. the demo-level split-half null in
    human_band.py, where frame/window permutation would be invalid under
    autocorrelation).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp


# ---------------------------------------------------------------------------
# Segment iteration
# ---------------------------------------------------------------------------
def iter_segments(labels: np.ndarray, keep: np.ndarray) -> Iterable[np.ndarray]:
    """Yield each maximal contiguous in-distribution run of ``labels``.

    ``keep`` is a boolean per-frame in-distribution mask of the same length,
    already restricted to a single episode by the caller (so a False frame —
    an out-of-distribution / masked frame — also breaks a segment, and the
    caller never passes data that spans an episode boundary).
    """
    labels = np.asarray(labels).reshape(-1)
    keep = np.asarray(keep).reshape(-1).astype(bool)
    if labels.shape[0] != keep.shape[0]:
        raise ValueError(f"labels {labels.shape} vs keep {keep.shape} length mismatch")
    n = labels.shape[0]
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < n and keep[j]:
            j += 1
        yield labels[i:j]
        i = j


# ---------------------------------------------------------------------------
# Run-length / dwell extraction
# ---------------------------------------------------------------------------
def run_lengths(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encode one contiguous segment.

    Returns (values, lengths): the label value of each maximal run and its
    length in frames. A run is a block of consecutive identical labels.
    """
    seg = np.asarray(segment).reshape(-1)
    if seg.shape[0] == 0:
        return np.empty(0, seg.dtype), np.empty(0, dtype=np.int64)
    change = np.nonzero(np.diff(seg))[0] + 1
    bounds = np.concatenate(([0], change, [seg.shape[0]]))
    lengths = np.diff(bounds).astype(np.int64)
    values = seg[bounds[:-1]]
    return values, lengths


def dwell_times(
    labels: np.ndarray,
    keep: np.ndarray,
    *,
    only_value=None,
    exclude_value=None,
) -> np.ndarray:
    """All contiguous-run lengths over the in-distribution segments.

    A run is consecutive identical labels WITHIN one in-distribution segment;
    runs never span an episode boundary or a keep-mask gap (the caller passes
    one episode at a time; keep gaps split further).

    ``only_value``   keep only runs whose label == this value (e.g. dwell of a
                     specific held move class, or "locked target" id).
    ``exclude_value`` drop runs whose label == this value (e.g. drop the
                     "no-target" / "centered" class to get engaged-dwell only).
    """
    out: list[np.ndarray] = []
    for seg in iter_segments(labels, keep):
        vals, lens = run_lengths(seg)
        if only_value is not None:
            lens = lens[vals == only_value]
        elif exclude_value is not None:
            lens = lens[vals != exclude_value]
        if lens.size:
            out.append(lens)
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Switch detection
# ---------------------------------------------------------------------------
def switch_mask(segment: np.ndarray) -> np.ndarray:
    """Per-frame "label differs from previous frame in same segment" (frame 0 = False)."""
    seg = np.asarray(segment).reshape(-1)
    if seg.shape[0] == 0:
        return np.empty(0, dtype=bool)
    out = np.zeros(seg.shape[0], dtype=bool)
    out[1:] = seg[1:] != seg[:-1]
    return out


def switch_rate(labels: np.ndarray, keep: np.ndarray) -> tuple[float, int, int]:
    """Fraction of in-segment transitions that are a switch.

    Returns (rate, n_switches, n_transitions). The denominator counts only
    frames that have a within-segment predecessor (so the first frame of each
    segment is excluded), matching the dwell / momentum convention.
    """
    n_switch = 0
    n_trans = 0
    for seg in iter_segments(labels, keep):
        if seg.shape[0] < 2:
            continue
        sw = switch_mask(seg)
        n_switch += int(sw[1:].sum())
        n_trans += seg.shape[0] - 1
    rate = (n_switch / n_trans) if n_trans else 0.0
    return rate, n_switch, n_trans


def preference_pairs(
    weapons: np.ndarray,
    ticks: np.ndarray,
    excluded_pair: np.ndarray | None = None,
) -> tuple[int, int, np.ndarray]:
    """Discharge-anchored weapon-PREFERENCE statistics for one episode.

    "Held weapon" is a fallacy in competitive Quake: weapon scripting churns
    equip state without expressing any choice. Preference is only observable
    at discharges — cache the weapon that attacked and compare the next
    attack against it. A pair whose transition was engine-FORCED (dry ammo
    pool, death/respawn reset) says nothing about preference — a weapon left
    the decision space — so ``excluded_pair`` removes it from numerator AND
    denominator.

    ``weapons``       (N,) weapon label at each discharge, time-ordered.
    ``ticks``         (N,) frame index of each discharge.
    ``excluded_pair`` (N-1,) bool; True drops pair ``(i, i+1)``.

    Returns ``(n_switch, n_pairs, dwell_frames)``. ``dwell_frames`` are the
    frame spans of same-weapon discharge streaks (first→last discharge of the
    streak), for streaks of ≥2 discharges; streaks closed by an exclusion or
    the episode end are included censored, matching the run-length convention
    above.
    """
    w = np.asarray(weapons).reshape(-1)
    t = np.asarray(ticks).reshape(-1)
    if w.shape[0] != t.shape[0]:
        raise ValueError(f"weapons {w.shape} vs ticks {t.shape} length mismatch")
    if w.shape[0] < 2:
        return 0, 0, np.empty(0)
    if excluded_pair is None:
        exc = np.zeros(w.shape[0] - 1, dtype=bool)
    else:
        exc = np.asarray(excluded_pair).reshape(-1).astype(bool)
        if exc.shape[0] != w.shape[0] - 1:
            raise ValueError(f"excluded_pair {exc.shape} != n_pairs {w.shape[0] - 1}")
    counted = ~exc
    n_pairs = int(counted.sum())
    n_switch = int(((w[1:] != w[:-1]) & counted).sum())
    dwell: list[float] = []
    start = 0
    for i in range(w.shape[0] - 1):
        if exc[i] or w[i + 1] != w[i]:
            if i > start:
                dwell.append(float(t[i] - t[start]))
            start = i + 1
    if w.shape[0] - 1 > start:
        dwell.append(float(t[-1] - t[start]))
    return n_switch, n_pairs, np.asarray(dwell, dtype=float)


def inter_event_intervals(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Frames between consecutive switch events within a segment.

    For each segment, the gaps (in frames) between successive switch positions.
    A switch-to-switch interval equals the dwell of the intervening hold. This
    is the inter-event-interval distribution (over/under-switching diagnostic).
    """
    out: list[np.ndarray] = []
    for seg in iter_segments(labels, keep):
        if seg.shape[0] < 2:
            continue
        sw = np.nonzero(switch_mask(seg))[0]
        if sw.shape[0] >= 2:
            out.append(np.diff(sw).astype(np.int64))
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out)


def onset_intervals(labels: np.ndarray, keep: np.ndarray, *, onset_value=1) -> np.ndarray:
    """Frames between consecutive onsets of ``onset_value`` (e.g. fire-onset interval).

    An onset = a frame whose label == onset_value while the previous in-segment
    frame != onset_value. Returns the gaps between successive onset positions.
    """
    out: list[np.ndarray] = []
    for seg in iter_segments(labels, keep):
        if seg.shape[0] < 2:
            continue
        is_on = seg == onset_value
        onset = np.zeros(seg.shape[0], dtype=bool)
        onset[1:] = is_on[1:] & (~is_on[:-1])
        pos = np.nonzero(onset)[0]
        if pos.shape[0] >= 2:
            out.append(np.diff(pos).astype(np.int64))
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Distances / tests
# ---------------------------------------------------------------------------
def _as_2d(x: np.ndarray) -> np.ndarray:
    """Coerce a sample array to (n, d) float64: 1-D input becomes (n, 1)."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _median_heuristic_gamma(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    """RBF gamma = 1 / (2 * median pairwise squared distance) on a pooled subsample."""
    pooled = np.concatenate([_as_2d(a), _as_2d(b)], axis=0)
    n = pooled.shape[0]
    if n > 400:  # subsample pairwise-distance estimate for speed
        pooled = pooled[rng.choice(n, 400, replace=False)]
    d2 = ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(axis=-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    if med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x = _as_2d(x)
    y = _as_2d(y)
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-gamma * d2)


def _mmd2_unbiased(K_xx: np.ndarray, K_yy: np.ndarray, K_xy: np.ndarray) -> float:
    m = K_xx.shape[0]
    n = K_yy.shape[0]
    sum_xx = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1)) if m > 1 else 0.0
    sum_yy = (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1)) if n > 1 else 0.0
    sum_xy = K_xy.sum() / (m * n) if (m and n) else 0.0
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def mmd2_rbf(a: np.ndarray, b: np.ndarray, *, gamma: float) -> float:
    """Unbiased RBF-MMD^2 statistic between samples ``a`` and ``b`` (no test).

    Accepts 1-D samples or (n, d) feature-vector samples. For callers that
    construct their own null distribution (e.g. demo-level split-half
    resampling, where a pooled permutation would break the cluster structure).
    """
    a = _as_2d(a)
    b = _as_2d(b)
    return _mmd2_unbiased(
        _rbf_kernel(a, a, gamma), _rbf_kernel(b, b, gamma), _rbf_kernel(a, b, gamma)
    )


def mmd2_rbf_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 1000,
    gamma: float | None = None,
    max_samples: int = 1000,
    seed: int = 17,
) -> dict:
    """Unbiased RBF-MMD^2 between samples ``a`` and ``b`` with a permutation null.

    The overall "indistinguishable?" verdict on a set of 1-D samples. Subsamples
    each side to ``max_samples`` for tractable O(N^2) kernels, picks the RBF
    bandwidth by the median heuristic if ``gamma`` is None, then estimates the
    null distribution of MMD^2 by ``n_perm`` random relabelings of the pooled set.

    Returns {mmd2, p_value, gamma, n_a, n_b, n_perm}. p_value = P(perm MMD^2 >=
    observed) under the null that a and b are the same distribution; small p =
    distinguishable from human.
    """
    rng = np.random.default_rng(seed)
    a = _as_2d(a)
    b = _as_2d(b)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return {"mmd2": float("nan"), "p_value": float("nan"), "gamma": float("nan"),
                "n_a": int(a.shape[0]), "n_b": int(b.shape[0]), "n_perm": 0}
    if a.shape[0] > max_samples:
        a = a[rng.choice(a.shape[0], max_samples, replace=False)]
    if b.shape[0] > max_samples:
        b = b[rng.choice(b.shape[0], max_samples, replace=False)]
    if gamma is None:
        gamma = _median_heuristic_gamma(a, b, rng)

    pooled = np.concatenate([a, b], axis=0)
    m = a.shape[0]
    K = _rbf_kernel(pooled, pooled, gamma)
    obs = _mmd2_unbiased(K[:m, :m], K[m:, m:], K[:m, m:])

    null = np.empty(n_perm)
    idx = np.arange(pooled.shape[0])
    for i in range(n_perm):
        rng.shuffle(idx)
        Kp = K[np.ix_(idx, idx)]
        null[i] = _mmd2_unbiased(Kp[:m, :m], Kp[m:, m:], Kp[:m, m:])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"mmd2": float(obs), "p_value": p, "gamma": float(gamma),
            "n_a": int(m), "n_b": int(b.shape[0]), "n_perm": int(n_perm)}


@dataclass
class StatCompare:
    emd: float
    ks_stat: float
    ks_p: float
    n_human: int
    n_model: int

    def as_dict(self) -> dict:
        return asdict(self)


def compare_samples(human: np.ndarray, model: np.ndarray) -> StatCompare:
    """1-D comparison of a statistic's human vs model samples: EMD + KS + counts."""
    human = np.asarray(human, dtype=np.float64).reshape(-1)
    model = np.asarray(model, dtype=np.float64).reshape(-1)
    if human.size == 0 or model.size == 0:
        return StatCompare(float("nan"), float("nan"), float("nan"),
                           int(human.size), int(model.size))
    emd = float(wasserstein_distance(human, model))
    ks = ks_2samp(human, model)
    return StatCompare(emd, float(ks.statistic), float(ks.pvalue),
                       int(human.size), int(model.size))


def print_compare_row(label: str, cmp: StatCompare, *, width: int = 22) -> None:
    """Print one aligned comparison row."""
    print(f"  {label:<{width}s}  emd={cmp.emd:10.4f}  ks={cmp.ks_stat:7.4f}  "
          f"ks_p={cmp.ks_p:9.3e}  n_h={cmp.n_human:>8d}  n_m={cmp.n_model:>8d}")


# ---------------------------------------------------------------------------
# Summary helper for a single sample array
# ---------------------------------------------------------------------------
def describe(x: np.ndarray) -> dict:
    """mean / median / p90 / p99 / count of a 1-D sample (NaNs for empty)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": None, "median": None, "p90": None, "p99": None, "n": 0}
    return {
        "mean": round(float(x.mean()), 4),
        "median": round(float(np.median(x)), 4),
        "p90": round(float(np.percentile(x, 90)), 4),
        "p99": round(float(np.percentile(x, 99)), 4),
        "n": int(x.size),
    }
