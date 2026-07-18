"""Transition-penalized decode for labeler per-frame probabilities.

Frame-wise argmax is what inflates the labeler's onset rate: one
low-confidence frame inside a long hold flips class for a single frame,
creating two false onsets (see qnn.labeler.seg_stats).  This module replaces
argmax with an exact per-episode Viterbi decode over the 3 axis classes with
a uniform switch penalty ``lam`` (nats): a class change is taken only when
the accumulated log-prob evidence for the new class exceeds ``lam``.
``lam = 0`` is bit-identical to argmax.

The penalty is a DECODE parameter fit on ground truth, mirroring the
decode-fit pattern used for model heads: ``fit_switch_penalty`` sweeps a
grid, scores each value on the 20 Hz segment-parity gate (onset ratio +
duration-bucket TV + window agreement vs truth), and returns the frontier so
the operating point is chosen on evidence.  This smooths *predictions* under
a fitted budget — it does not synthesize or extend labels.
"""
from __future__ import annotations

import numpy as np

from .seg_stats import (
    downsample_axis,
    segment_parity,
    window_ids,
)

N_CLASSES = 3

# Default λ sweep (nats).  0 = argmax baseline; upper end deliberately past
# any plausible operating point so the frontier shows the over-smoothing turn.
DEFAULT_LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def viterbi_smooth(
    log_probs: np.ndarray,         # (N, 3) per-frame class log-probs, one axis
    episode_starts: np.ndarray,    # (E+1,) int64 row offsets
    lam: float,                    # uniform switch penalty (nats), >= 0
) -> np.ndarray:
    """Exact per-episode Viterbi decode with a uniform switch penalty.

    Maximizes ``sum_t log p(c_t) - lam * #{t : c_t != c_{t-1}}`` independently
    per episode.  Episodes are processed in one batched time loop (padded to
    the longest episode), so the python loop runs max-episode-length times,
    not N.  Returns (N,) int64 classes.  ``lam <= 0`` short-circuits to
    argmax.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    n = lp.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if lam <= 0.0:
        return lp.argmax(axis=1).astype(np.int64)

    episode_starts = np.asarray(episode_starts, dtype=np.int64)
    lengths = np.diff(episode_starts)
    E = lengths.shape[0]
    L = int(lengths.max())

    # Pad episodes into (E, L, 3); padding rows are all-zero (constant per
    # class, so they never change the argmax path of real frames).
    padded = np.zeros((E, L, N_CLASSES), dtype=np.float64)
    row = np.arange(L)
    in_ep = row[None, :] < lengths[:, None]                  # (E, L)
    flat_idx = (episode_starts[:-1, None] + row[None, :])[in_ep]
    padded[in_ep] = lp[flat_idx]

    # dp over time: score (E, 3); back-pointers (E, L, 3) uint8.
    score = padded[:, 0, :].copy()
    bp = np.zeros((E, L, N_CLASSES), dtype=np.uint8)
    for t in range(1, L):
        stay = score                                          # (E, 3)
        best_prev = score.argmax(axis=1)                      # (E,)
        best_val = score[np.arange(E), best_prev]             # (E,)
        switch = best_val[:, None] - lam                      # (E, 3)
        take_stay = stay >= switch
        bp[:, t, :] = np.where(
            take_stay, np.arange(N_CLASSES)[None, :], best_prev[:, None]
        ).astype(np.uint8)
        score = np.maximum(stay, switch) + padded[:, t, :]

    # Backtrace per episode from its own last frame.
    out = np.empty(n, dtype=np.int64)
    states = np.empty((E, L), dtype=np.int64)
    last = lengths - 1
    cur = np.empty(E, dtype=np.int64)
    # final state at each episode's last real frame: recompute score at that
    # frame by walking dp forward is wasteful — instead take argmax of the
    # running score only when the episode ends at L-1; shorter episodes end
    # earlier, so track their final scores during the loop.  Simpler and
    # still O(E*L): re-run a light forward pass storing per-frame best is
    # avoided by noting padding rows are all-zero: for t >= length the stay
    # path never loses (switch costs lam > 0 with no evidence to gain), so
    # the state at L-1 equals the state at the last real frame.
    cur[:] = score.argmax(axis=1)
    for t in range(L - 1, -1, -1):
        states[:, t] = cur
        cur = bp[np.arange(E), t, cur]
    out[flat_idx] = states[in_ep]
    return out


def gate_metrics_20hz(
    pred_native: np.ndarray,       # (N,) native-rate predicted classes
    truth_native: np.ndarray,      # (N,) native-rate truth classes
    episode_starts: np.ndarray,    # (E+1,) native-rate offsets
    stride: int,                   # native->20 Hz window stride
    valid_native: np.ndarray | None = None,
) -> dict:
    """Score one axis stream on the 20 Hz segment-parity gate.

    Downsamples pred and truth (windowed-union) and returns
    ``segment_parity`` plus the 20 Hz window agreement.
    """
    win_id, n_win, win_starts = window_ids(episode_starts, stride)
    p20 = downsample_axis(np.asarray(pred_native, dtype=np.int64), win_id, n_win)
    t20 = downsample_axis(np.asarray(truth_native, dtype=np.int64), win_id, n_win)
    v20 = None
    if valid_native is not None:
        bad = np.bincount(win_id[~np.asarray(valid_native, dtype=bool)],
                          minlength=n_win)
        v20 = bad == 0
    par = segment_parity(p20, t20, win_starts, valid=v20)
    agree = (p20 == t20) if v20 is None else (p20 == t20)[v20]
    par["agree_20hz"] = round(float(agree.mean()), 6) if agree.size else 0.0
    return par


def fit_switch_penalty(
    log_probs: np.ndarray,         # (N, 3) native-rate axis log-probs
    truth_native: np.ndarray,      # (N,) native-rate truth classes
    episode_starts: np.ndarray,
    stride: int,
    *,
    lam_grid: tuple[float, ...] = DEFAULT_LAM_GRID,
    valid_native: np.ndarray | None = None,
) -> dict:
    """Sweep the switch penalty and score each value on the 20 Hz gate.

    Selection: the λ minimizing ``|log(onset_ratio)| + dur_tv`` — onset-rate
    parity and duration-law parity weighted equally, both 0 at perfection.
    Returns ``{"lam": chosen, "frontier": [{lam, onset_ratio, dur_tv,
    agree_20hz, score}, ...]}`` (full frontier retained so the operating
    point is inspectable; ties go to the smaller λ).
    """
    frontier = []
    best = None
    for lam in lam_grid:
        pred = viterbi_smooth(log_probs, episode_starts, lam)
        m = gate_metrics_20hz(pred, truth_native, episode_starts, stride,
                              valid_native=valid_native)
        ratio = m["onset_ratio"]
        score = (abs(float(np.log(ratio))) if ratio else float("inf")) + m["dur_tv"]
        row = {"lam": lam, "onset_ratio": ratio, "dur_tv": m["dur_tv"],
               "agree_20hz": m["agree_20hz"], "score": round(score, 4)}
        frontier.append(row)
        if best is None or score < best[0]:
            best = (score, lam)
    return {"lam": best[1], "frontier": frontier}
