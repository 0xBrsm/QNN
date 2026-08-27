"""Torch-free bucket constants + deriver for the ``attack_future`` aux head.

Single source of truth for the censored time-to-next-op-discharge
discretization shared by the torch head (:mod:`qnn.model.attack_future_head`)
and the torch-free label producers (the BC loop's resident/streaming preload,
which derives the column PER EPISODE at source-build time). Keep this module
free of torch imports — mirrors ``look_seg_bins.py`` / ``seg_bins.py``.

The target (agents/plans/mtp-attack-future-probe.md):

    class 0 = no op discharge in d ∈ [1..HORIZON]     (~88% of op frames)
    class 1 = d ∈ {1,2}   class 2 = d ∈ {3,4}
    class 3 = d ∈ {5,6}   class 4 = d ∈ {7,8}

``d`` is the number of ticks to the next discharge STRICTLY after t, so k=0 is
excluded — the current tick is the attack head's job. The event predicate is
``act_attack > 0 AND (act_input_mask & 1)``, the same one the cross-head
coordination ruler uses.

Censoring: a frame with no event inside the horizon AND fewer than HORIZON
frames left in its episode is right-censored — the horizon was never observed,
so the label is ``IGNORE`` (-100, the house convention). An event beyond the
horizon is a real observation and lands in class 0. Roughly 13% of frames are
unscored at run ends; that bias is identical in every arm and the aux skill is
never compared to an external base rate.

The horizon is pinned HERE, not in the graph spec (the ``look_seg_bins`` JOINT
precedent): sweeping it is a deliberate code edit, not a config knob.
"""
from __future__ import annotations

import numpy as np

HORIZON = 8                                  # ticks = 400 ms @ 20 Hz
BUCKET_EDGES = (1, 3, 5, 7, 9)               # searchsorted-right → class 1..4
N_CLASSES = 5                                # class 0 = "no discharge in horizon"
IGNORE = -100


def derive_next_discharge_bucket(attack: np.ndarray, op: np.ndarray) -> np.ndarray:
    """(T,) int8 bucket labels for ONE episode.

    ``attack`` is the raw per-frame attack class column (0 = no discharge,
    1..8 = fired weapon k); ``op`` is the per-frame operative bit
    (``act_input_mask & 1``). Both must cover exactly one episode — run breaks
    are structural, so deriving per episode makes cross-run leakage impossible
    and there are no chunk-edge cases.

    Vectorized flipped-cummin (the ``move_seg_head.derive_segment_targets``
    trick), NOT a Python scan: this runs over ~162k episodes at preload.
    """
    a = np.asarray(attack).reshape(-1)
    o = np.asarray(op).reshape(-1)
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    idx = np.arange(n)
    e = (a > 0) & o.astype(bool)
    pos = np.where(e, idx, n)
    nxt = np.minimum.accumulate(pos[::-1])[::-1]      # first event at or after t
    after = np.append(nxt[1:], n)                     # first event STRICTLY after t
    d = after - idx
    bucket = np.searchsorted(
        np.asarray(BUCKET_EDGES), d, side="right",
    ).clip(max=N_CLASSES - 1).astype(np.int8)
    bucket[d > HORIZON] = 0                           # observed, just far away
    # No event to the end of the episode AND the horizon does not fit in what
    # remains → right-censored, never observed.
    bucket[(after == n) & (n - idx - 1 < HORIZON)] = IGNORE
    return bucket
