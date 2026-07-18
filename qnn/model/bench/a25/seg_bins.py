"""Torch-free duration-bucket constants for the a25 ``move_seg`` head.

Single source of truth for the Fibonacci duration-bucket law shared by the
torch head (``qnn.model.bench.a25.move_seg_head``) and torch-free consumers
(the labeler's segment-parity gate, ``qnn.labeler.seg_stats``).  Keep this
module free of torch imports — the GBT labeler path is deliberately
torch-free.
"""
from __future__ import annotations

import numpy as np

FIB_EDGES = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)
N_BUCKETS = len(FIB_EDGES)          # bucket i = [edge_i, edge_{i+1}); last = tail


def bucketize_duration_np(dur: np.ndarray) -> np.ndarray:
    """duration frames (>=1) -> bucket 0..9; numpy mirror of the torch
    ``move_seg_head.bucketize_duration`` (searchsorted-right over FIB_EDGES)."""
    edges = np.asarray(FIB_EDGES, dtype=np.int64)
    return np.clip(
        np.searchsorted(edges, np.asarray(dur, dtype=np.int64), side="right") - 1,
        0, N_BUCKETS - 1,
    )
