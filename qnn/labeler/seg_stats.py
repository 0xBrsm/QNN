"""Torch-free segment statistics — the labeler's fb/lr segment-parity gate.

The a25 ``move_seg`` head trains on (onset, duration-bucket) targets derived
on the fly from per-frame move classes
(``qnn.model.bench.a25.move_seg_head.derive_segment_targets``), and the
commitment decode's ``dur_tilt`` is calibrated against the duration law those
targets induce.  A relabeled corpus therefore has to be judged on SEGMENT
statistics, not just frame accuracy: one isolated per-frame flip inside a long
hold creates two false onsets and shatters the hold into three short segments
— inflating onset rate and shifting duration mass into the short Fibonacci
buckets, exactly where the bucket law is finest.  A ~90%-frame-accurate
labeler can still be badly wrong here.

This module computes those statistics with the target-derivation semantics:

  * an onset is a class change whose previous frame is valid and in the same
    episode (episode starts are not onsets),
  * a segment's duration runs to the NEXT change,
  * segments cut by an invalid frame or the episode end are right-censored —
    counted as onsets but excluded from the duration histogram.

Run it on the 20 Hz (model-rate) windowed-union downsample — the same streams
the relabel-quality table uses — so durations land in the frame units
``move_seg`` buckets.  The 20 Hz windowing helpers live here too, shared by
the GBT report and the TCN trainer.
"""
from __future__ import annotations

import numpy as np

from qnn.model.bench.a25.seg_bins import N_BUCKETS, bucketize_duration_np

N_CLASSES = 3   # {0: neg, 1: none, 2: pos} — per-axis move classes


# ── 20 Hz windowed-union downsample (episode-boundary-aware) ───────────────────

def window_ids(episode_starts: np.ndarray, stride: "int | np.ndarray"
               ) -> tuple[np.ndarray, int, np.ndarray]:
    """Assign each frame a contiguous global window id, episode-aware.

    ``stride``-frame windows never straddle an episode boundary.  ``stride``
    is a scalar or a per-episode ``(E,)`` array — demos record at their
    native client rate (77 Hz and 60 Hz both common), so a matched corpus
    needs per-episode strides for the windows to be real 20 Hz model frames
    (see ``qnn.labeler.data.matched_episode_strides``).  Returns
    ``(win_id (N,), n_windows, win_episode_starts (E+1,))`` — the last is the
    episode-start offsets of the WINDOW-level stream, for segment stats on the
    downsampled labels.
    """
    episode_starts = np.asarray(episode_starts, dtype=np.int64)
    n = int(episode_starts[-1])
    starts = episode_starts[:-1]
    lengths = np.diff(episode_starts)
    E = starts.shape[0]
    stride_per_ep = (np.full(E, int(stride), dtype=np.int64)
                     if np.isscalar(stride)
                     else np.asarray(stride, dtype=np.int64).reshape(E))
    # within-episode frame offset
    ep_of_frame = np.repeat(np.arange(E), lengths)
    within = np.arange(n) - starts[ep_of_frame]
    win_within = within // stride_per_ep[ep_of_frame]   # window idx inside episode
    n_win_per_ep = (lengths + stride_per_ep - 1) // stride_per_ep
    win_base = np.zeros(E + 1, dtype=np.int64)
    win_base[1:] = np.cumsum(n_win_per_ep)
    win_id = win_base[ep_of_frame] + win_within
    return win_id.astype(np.int64), int(win_base[-1]), win_base


def downsample_axis(
    labels: np.ndarray,            # (N,) int  per-axis class {0=neg,1=none,2=pos}
    win_id: np.ndarray,            # (N,) int  global episode-aware window id
    n_windows: int,
) -> np.ndarray:
    """Downsample a per-frame axis label stream to ~20 Hz, per episode.

    Windowed-union (vectorized): within each window the class is the
    most-common NON-none press if any frame pressed, else none.  Mirrors how
    a 20 Hz controller registers a button held for part of the window.
    Episode boundaries are respected via ``win_id`` (see ``window_ids``).
    """
    # Per-window counts of neg (class 0) and pos (class 2) presses.
    out = np.ones(n_windows, dtype=np.int64)            # default 'none'
    neg = np.bincount(win_id[labels == 0], minlength=n_windows)
    pos = np.bincount(win_id[labels == 2], minlength=n_windows)
    pressed = (neg + pos) > 0
    # most-common pressed class; ties (neg==pos) resolve to neg (argmax-style,
    # matching np.bincount([0,2]).argmax() == 0)
    out[pressed] = np.where(neg[pressed] >= pos[pressed], 0, 2)
    return out


def window_all_true(flags: np.ndarray, win_id: np.ndarray, n_windows: int
                    ) -> np.ndarray:
    """Per-window AND of a per-frame bool stream (window valid iff every
    frame in it is valid)."""
    bad = np.bincount(win_id[~np.asarray(flags, dtype=bool)],
                      minlength=n_windows)
    return bad == 0


# ── segment statistics (derive_segment_targets semantics) ──────────────────────

def segment_stats(
    labels: np.ndarray,            # (N,) int per-axis class stream
    episode_starts: np.ndarray,    # (E+1,) int64 row offsets
    valid: np.ndarray | None = None,   # (N,) bool; None = all valid
) -> dict:
    """Onset + duration statistics of one label stream, matching the torch
    target derivation (see module docstring for the exact semantics).

    Returns counts (not normalized): ``n_frames`` / ``n_valid`` / ``n_onsets``
    / ``onset_rate`` (onsets per valid frame) / ``n_complete`` /
    ``dur_hist (N_BUCKETS,)`` / ``dur_hist_by_class (N_CLASSES, N_BUCKETS)``
    over complete segments / ``dur_median`` (frames, complete segments).
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    episode_starts = np.asarray(episode_starts, dtype=np.int64)
    n = labels.shape[0]
    empty = {
        "n_frames": n, "n_valid": 0, "n_onsets": 0, "onset_rate": 0.0,
        "n_complete": 0, "dur_hist": np.zeros(N_BUCKETS, dtype=np.int64),
        "dur_hist_by_class": np.zeros((N_CLASSES, N_BUCKETS), dtype=np.int64),
        "dur_median": 0.0,
    }
    if n == 0:
        return empty
    valid = (np.ones(n, dtype=bool) if valid is None
             else np.asarray(valid, dtype=bool).reshape(-1))
    lengths = np.diff(episode_starts)
    ep_of = np.repeat(np.arange(lengths.shape[0]), lengths)
    ep_end = episode_starts[ep_of + 1]                  # exclusive end per frame

    chg = np.zeros(n, dtype=bool)
    chg[1:] = ((labels[1:] != labels[:-1]) & valid[1:] & valid[:-1]
               & (ep_of[1:] == ep_of[:-1]))

    pos = np.arange(n, dtype=np.int64)
    # next change at-or-after t (reverse running min), then strictly after t
    nc = np.minimum.accumulate(np.where(chg, pos, n)[::-1])[::-1]
    nca = np.empty(n, dtype=np.int64)
    nca[:-1] = nc[1:]
    nca[-1] = n
    # first invalid frame at-or-after t, then strictly after t
    nb = np.minimum.accumulate(np.where(~valid, pos, n)[::-1])[::-1]
    nba = np.empty(n, dtype=np.int64)
    nba[:-1] = nb[1:]
    nba[-1] = n
    barrier = np.minimum(nba, ep_end)                   # first break after t

    complete = chg & (nca < barrier)                    # segment ends at a change
    dur = (nca - pos)[complete]
    cls = labels[complete]

    n_valid = int(valid.sum())
    n_onsets = int(chg.sum())
    if dur.shape[0]:
        buckets = bucketize_duration_np(dur)
        hist = np.bincount(buckets, minlength=N_BUCKETS).astype(np.int64)
        by_class = np.bincount(cls * N_BUCKETS + buckets,
                               minlength=N_CLASSES * N_BUCKETS
                               ).reshape(N_CLASSES, N_BUCKETS).astype(np.int64)
        dur_median = float(np.median(dur))
    else:
        hist = empty["dur_hist"]
        by_class = empty["dur_hist_by_class"]
        dur_median = 0.0
    return {
        "n_frames": n,
        "n_valid": n_valid,
        "n_onsets": n_onsets,
        "onset_rate": n_onsets / max(1, n_valid),
        "n_complete": int(dur.shape[0]),
        "dur_hist": hist,
        "dur_hist_by_class": by_class,
        "dur_median": dur_median,
    }


def _norm(hist: np.ndarray) -> np.ndarray:
    s = hist.sum()
    return hist / s if s else hist.astype(np.float64)


def segment_parity(
    pred: np.ndarray,
    truth: np.ndarray,
    episode_starts: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict:
    """Predicted-vs-truth segment parity for one axis stream (JSON-friendly).

    Headline numbers: ``onset_ratio`` (pred onset rate / truth onset rate;
    labeler flicker shows up as >1) and ``dur_tv`` (total-variation distance
    between the normalized duration-bucket histograms; the duration-law
    mismatch ``move_seg`` + the ``dur_tilt`` calibration would inherit).
    """
    p = segment_stats(pred, episode_starts, valid)
    t = segment_stats(truth, episode_starts, valid)
    hp, ht = _norm(p["dur_hist"]), _norm(t["dur_hist"])
    return {
        "onset_rate_pred": round(p["onset_rate"], 6),
        "onset_rate_truth": round(t["onset_rate"], 6),
        "onset_ratio": round(p["onset_rate"] / t["onset_rate"], 4)
                       if t["onset_rate"] else None,
        "n_onsets_pred": p["n_onsets"], "n_onsets_truth": t["n_onsets"],
        "n_complete_pred": p["n_complete"], "n_complete_truth": t["n_complete"],
        "dur_median_pred": p["dur_median"], "dur_median_truth": t["dur_median"],
        "dur_tv": round(0.5 * float(np.abs(hp - ht).sum()), 4),
        "dur_hist_pred": [round(x, 4) for x in hp.tolist()],
        "dur_hist_truth": [round(x, 4) for x in ht.tolist()],
        "dur_hist_by_class_pred": [
            [round(x, 4) for x in _norm(row).tolist()]
            for row in p["dur_hist_by_class"]],
        "dur_hist_by_class_truth": [
            [round(x, 4) for x in _norm(row).tolist()]
            for row in t["dur_hist_by_class"]],
    }
