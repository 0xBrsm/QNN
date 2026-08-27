"""Offline look-segment kernel — promoted from ``scripts/analysis/_look_seg_audit.py``.

The Phase-0 audit's segmentation rule became load-bearing (it defines the
population the ``look_seg`` head trains and is scored against), so it lives in
the package rather than a scripts/ file. The audit imports these functions so
the two can't drift. Torch-free (numpy only) — mirrors ``seg_bins.py``'s
torch-free discipline; the on-the-fly TORCH derivation used inside the loss is
the vectorized twin in ``look_seg_head.derive_look_seg_targets``.

Segmentation (plan agents/plans/look-seg-head.md §0):
  * hold segment  = maximal run of θ==0 (the true-stillness point mass).
  * turning run   = maximal run of θ>0.
  * stroke        = a turning run split on direction reversal (|Δφ| between
                    consecutive tangent vectors above ``REVERSAL_RAD``).

Population = the exact training subset: segment_mask ``act.target != 0`` engaged
runs. Segments touching a run boundary (episode edge OR mask transition) are
RIGHT-CENSORED (dropped from duration stats, counted for the censoring rate).
"""
from __future__ import annotations

import numpy as np

DUR_CAP = 40                                   # per-tick velocity profiles up to this D


def runs_from_bool(mask: np.ndarray):
    """Yield (start, end) half-open index ranges of maximal True runs."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0], idx + 1, [mask.size]))
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if mask[a]:
            out.append((int(a), int(b)))
    return out


def episode_slices(n_lengths):
    off = 0
    for n in n_lengths:
        n = int(n)
        if n > 0:
            yield off, off + n
        off += n


def new_acc():
    return {
        "hold_dur": [], "hold_present_onset": [],
        "stroke_dur": [], "stroke_amp": [],
        "hold_censored": 0, "stroke_censored": 0, "turn_runs": 0,
        "vprof_sum": {d: np.zeros(d) for d in range(2, DUR_CAP + 1)},
        "vprof_cnt": {d: 0 for d in range(2, DUR_CAP + 1)},
        # full-episode holds (no engagement mask) for the conditional-signal
        # check: engagement barely varies WITHIN the engaged population, so the
        # idle-vs-combat contrast is the honest test of context dependence.
        "hold_full_dur": [], "hold_full_present": [],
    }


def segment_engaged(z, theta, tp, ep_lengths, reversal_rad, acc):
    """Walk engaged runs; fill accumulators `acc`. Engaged = P(NO_TARGET) != 1
    (train.py act.target = 1 - P(NO_TARGET), segment_mask $ne 0)."""
    present = 1.0 - tp[:, 0].astype(np.float64)          # engagement scalar
    engaged = present != 0.0
    for a, b in episode_slices(ep_lengths):
        eng = engaged[a:b]
        for ra, rb in runs_from_bool(eng):               # a valid window
            s, e = a + ra, a + rb                         # abs indices
            th = theta[s:e]
            zz = z[s:e]
            pr = present[s:e]
            L = e - s
            hold = th == 0.0
            # alternating hold/turn segments within this valid window
            segs = []  # (kind, lo, hi) lo/hi local to the run
            for (rlo, rhi) in runs_from_bool(hold):
                segs.append(("hold", rlo, rhi))
            for (rlo, rhi) in runs_from_bool(~hold):
                segs.append(("turn", rlo, rhi))
            segs.sort(key=lambda x: x[1])
            for kind, lo, hi in segs:
                boundary = (lo == 0) or (hi == L)
                if kind == "hold":
                    dur = hi - lo
                    if boundary:
                        acc["hold_censored"] += 1
                    else:
                        acc["hold_dur"].append(dur)
                        acc["hold_present_onset"].append(float(pr[lo]))
                    continue
                # turning run -> split into strokes on reversal
                zt = zz[lo:hi]
                if hi - lo >= 2:
                    z0, z1 = zt[:-1], zt[1:]
                    dot = z0[:, 0] * z1[:, 0] + z0[:, 1] * z1[:, 1]
                    crs = z0[:, 0] * z1[:, 1] - z0[:, 1] * z1[:, 0]
                    dphi = np.arctan2(np.abs(crs), dot)
                    rev = np.flatnonzero(dphi > reversal_rad) + 1   # local split pts
                else:
                    rev = np.empty(0, dtype=np.int64)
                cuts = np.concatenate(([0], rev, [hi - lo])).astype(np.int64)
                nstrokes = len(cuts) - 1
                for k in range(nstrokes):
                    slo, shi = int(cuts[k]), int(cuts[k + 1])
                    # a stroke is censored only if it sits at the valid-window edge
                    st_boundary = (lo == 0 and k == 0) or (hi == L and k == nstrokes - 1)
                    seg_th = th[lo + slo:lo + shi]
                    dur = shi - slo
                    if st_boundary:
                        acc["stroke_censored"] += 1
                        continue
                    acc["stroke_dur"].append(dur)
                    acc["stroke_amp"].append(float(seg_th.sum()))
                    if 2 <= dur <= DUR_CAP:
                        tot = seg_th.sum()
                        if tot > 0:
                            v = seg_th / tot
                            acc["vprof_sum"][dur] += v
                            acc["vprof_cnt"][dur] += 1
                acc["turn_runs"] += 1


def segment_full_holds(theta, present, ep_lengths, acc):
    """Full-episode holds (θ==0) for the conditional-signal check — episode
    boundaries only, no engagement mask. Records complete-hold duration +
    target-visibility at onset."""
    for a, b in episode_slices(ep_lengths):
        hold = theta[a:b] == 0.0
        L = b - a
        for rlo, rhi in runs_from_bool(hold):
            if rlo == 0 or rhi == L:
                continue                                 # episode-censored
            acc["hold_full_dur"].append(rhi - rlo)
            acc["hold_full_present"].append(float(present[a + rlo]))
