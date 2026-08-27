"""Torch-free grid constants for the a25 ``look_seg`` head.

Single source of truth for the look-segment discretization shared by the torch
head (``qnn.model.look_seg_head``) and torch-free consumers (the
offline audit / baseline scorer, ``scripts/analysis/_look_seg_audit.py``). Keep
this module free of torch imports — mirrors ``seg_bins.py`` for move_seg.

Every value is a Phase-0 MEASUREMENT, not a design choice, fit per corpus tick
rate (the discretization is a quantile/Lloyd-Max fit over frame COUNTS, so it
does not transfer across Hz — a 10Hz stroke of duration 4 covers twice the
wall-clock time of a 20Hz stroke of duration 4). Two fits are committed:

  * 10 Hz — ``runs/head_probe/_look_seg_audit_10hz.json`` (qwd_a27_10hz corpus)
  * 20 Hz — ``runs/head_probe/_look_seg_audit.json`` (qwd_v4 corpus)

from each audit's:
  * ``LookSegBins.dur_edges``      <- duration_buckets.proposed_edges
  * ``LookSegBins.amp_centers_rad``<- stroke_amplitude_grid.sweep[k=8].centers_rad
  * ``LookSegBins.reversal_rad``   <- reversal_split.threshold_rad

Both fits share the same SHAPE (10 duration buckets, 8 amp centers -> JOINT=90
unchanged) — only the values differ. ``bins_for_hz`` is the only way to get at
them; there is no bare module-level "current" table and no silent default, so
a caller that forgets to pick an Hz fails immediately instead of silently
running the wrong grid.

The onset class is ``{hold} ∪ K stroke-amplitude bins`` (K=8), the joint the
head predicts is (onset-class × duration-bucket), and direction is a separate
uniform-bin categorical scored at stroke onsets only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Shape-invariant constants (identical across every Hz fit) ───────────────
N_STROKE_AMP = 8
N_ONSET_CLASSES = 1 + N_STROKE_AMP                        # 9 (0 = hold)
N_DUR_BUCKETS = 10
JOINT = N_ONSET_CLASSES * N_DUR_BUCKETS                   # 90
N_LOOK_DIR = 16                                           # direction categorical bins


@dataclass(frozen=True)
class LookSegBins:
    """One Hz's fitted discretization. All angles in radians."""
    hz: int
    dur_edges: tuple[int, ...]           # bucket i = [edge_i, edge_{i+1}); last = [edge_-1, inf)
    amp_centers_rad: tuple[float, ...]   # Lloyd-Max stroke-amplitude centers, K=8
    reversal_rad: float                 # turning-run split threshold (antimode of |Δφ|)

    def __post_init__(self) -> None:
        if len(self.dur_edges) != N_DUR_BUCKETS:
            raise ValueError(
                f"hz={self.hz}: dur_edges has {len(self.dur_edges)} entries, "
                f"expected N_DUR_BUCKETS={N_DUR_BUCKETS}")
        if len(self.amp_centers_rad) != N_STROKE_AMP:
            raise ValueError(
                f"hz={self.hz}: amp_centers_rad has {len(self.amp_centers_rad)} "
                f"entries, expected N_STROKE_AMP={N_STROKE_AMP}")


# 10 Hz fit (qwd_a27_10hz): holds compress to median 1 tick, strokes median 4;
# strokes are larger than 20 Hz (sub-strokes merge). Values verbatim from the
# pre-parameterization constants (runs/head_probe/_look_seg_audit_10hz.json,
# committed at a096b523) — this table must not drift, every pre-fix a27
# checkpoint (and the self-consistent-but-mislabeled lookseg_w122s1_seed43
# 20Hz-corpus run) trained against exactly these numbers.
_BINS_10HZ = LookSegBins(
    hz=10,
    dur_edges=(1, 2, 3, 4, 5, 6, 7, 8, 9, 23),
    amp_centers_rad=(
        0.09876, 0.4851, 0.96129, 1.52569, 2.18303, 2.97327, 4.06084, 5.86352,
    ),
    reversal_rad=2.10312,                                 # 120.5 deg
)

# 20 Hz fit (qwd_v4, runs/head_probe/_look_seg_audit.json, tick_hz=20,
# segment_mask act.target != 0 engaged runs, artifacts/collect/qwd):
#   dur_edges       <- duration_buckets.proposed_edges
#   amp_centers_rad <- stroke_amplitude_grid.sweep[k=8].centers_rad
#   reversal_rad    <- reversal_split.threshold_rad (126.5 deg)
_BINS_20HZ = LookSegBins(
    hz=20,
    dur_edges=(1, 2, 3, 4, 5, 6, 7, 8, 12, 35),
    amp_centers_rad=(
        0.06648, 0.40075, 0.83292, 1.35992, 1.97849, 2.73984, 3.73937, 5.47457,
    ),
    reversal_rad=2.20784,                                 # 126.5 deg
)

_TABLE: dict[int, LookSegBins] = {10: _BINS_10HZ, 20: _BINS_20HZ}

# Shape parity across fits — LookSegBins.__post_init__ already enforces each
# row against N_DUR_BUCKETS/N_STROKE_AMP; this just double-checks the table
# itself is well-formed (every entry keyed by its own hz).
assert all(hz == bins.hz for hz, bins in _TABLE.items()), "_TABLE key must equal LookSegBins.hz"


def bins_for_hz(hz: int) -> LookSegBins:
    """The fitted grid for corpus tick rate ``hz``. No default: an unknown (or
    unspecified — pass ``resolve_hz`` first) hz raises rather than silently
    running the wrong discretization."""
    try:
        return _TABLE[int(hz)]
    except (TypeError, ValueError, KeyError):
        raise ValueError(
            f"no look_seg bins fit for tick_hz={hz!r}; known: {sorted(_TABLE)}"
        ) from None


# Resolution for HeadNodeSpec.hz == 0 (the graph key absent — every look_seg
# checkpoint saved before hz-parameterization landed, including the
# self-consistent-but-mislabeled lookseg_w122s1_seed43 20Hz-corpus run). This
# is NOT a silent default: it is the one value that reproduces what every
# existing checkpoint actually trained against, so old checkpoints keep
# loading and decoding unchanged. New graphs must stamp an explicit hz.
LEGACY_HZ = 10


def resolve_hz(raw_hz: int) -> int:
    """HeadNodeSpec.hz (0 = absent from the graph) -> a concrete tick rate."""
    return int(raw_hz) if raw_hz else LEGACY_HZ


def bucketize_duration_np(dur: np.ndarray, dur_edges: "tuple[int, ...]") -> np.ndarray:
    """duration frames (>=1) -> bucket 0..N_DUR_BUCKETS-1; numpy mirror of the
    torch ``look_seg_head.bucketize_duration`` (searchsorted-right over
    ``dur_edges``, the caller's ``LookSegBins.dur_edges``)."""
    edges = np.asarray(dur_edges, dtype=np.int64)
    return np.clip(
        np.searchsorted(edges, np.asarray(dur, dtype=np.int64), side="right") - 1,
        0, N_DUR_BUCKETS - 1,
    )
