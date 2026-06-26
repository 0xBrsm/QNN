"""Shared move-axis kernels for analysis and diagnostic scripts.

Centralises three helpers that were previously copy-pasted across several
``scripts/analysis/`` move/jump scripts:

  unpack_move   — press-byte → (T,3) class array; delegates to
                  ``qnn.actions.decode_move_pressbyte`` (no inline bit math).
  ud_rewrite    — operative-frame ud rewrite under input_mask feasibility bits.
  plan_batches  — GRU episode → padded-lane batch partitioner.

Import from here; do not copy the formulas into new scripts.
"""
from __future__ import annotations

import numpy as np

from qnn.actions import decode_move_pressbyte


# ── Constants shared with padded-forward callers ─────────────────────────────

MAX_TB = 16384       # max time × batch tokens per padded forward
MAX_LANES = 128      # max episodes per padded forward batch


# ── Public API ────────────────────────────────────────────────────────────────

def unpack_move(packed) -> np.ndarray:
    """Decode the BC press-byte (act_move) into a (T, 3) uint8 axis-class array.

    Delegates to ``qnn.actions.decode_move_pressbyte``; the authoritative bit
    layout and formula live there.  Returns classes in {0=neg, 1=none, 2=pos}
    for axes [fb, lr, ud], with the jump bit (bit 7) folded into ud_pos.

    Parameters
    ----------
    packed : array-like, shape (T,), dtype uint8
        Packed press-byte array as stored on disk (``actions/move`` shard key).

    Returns
    -------
    np.ndarray, shape (T, 3), dtype uint8
    """
    return decode_move_pressbyte(np.asarray(packed, dtype=np.uint8).reshape(-1))


def ud_rewrite(move_cls: np.ndarray, input_mask) -> np.ndarray:
    """Rewrite the ud axis of a (T, 3) class array under input_mask feasibility.

    Training and operative-frame evaluation mask infeasible ud presses so the
    model is not penalised for air-jump presses the engine ignores.  This
    function applies the same rewrite offline so human-label statistics match
    what the model is trained against.

    Rewrite rules (mirrors ``qnn.model.policy._compute_head_losses_and_metrics``):
      - ud POS (class 2: jump / swim-up) kept iff  bit7 (ground-jump) OR bit6
        (swim-up) is set in input_mask; else → NONE (class 1).
      - ud NEG (class 0: swim-down) kept iff bit5 is set; else → NONE (class 1).
      - fb / lr axes are untouched (always feasible).

    Parameters
    ----------
    move_cls : np.ndarray, shape (T, 3), dtype compatible with int
        Per-frame axis classes as returned by ``unpack_move``.
    input_mask : array-like, shape (T,), dtype uint8
        Per-frame input_mask byte from the shard (``actions/input_mask``).

    Returns
    -------
    np.ndarray, shape (T, 3), same dtype as *move_cls* (copy, not in-place).
    """
    m = np.asarray(input_mask).reshape(-1).astype(np.int64)
    out = move_cls.copy()
    up_neg = ((m >> 5) & 1) != 0   # swim-down feasible
    up_pos = ((m >> 6) & 1) != 0   # swim-up feasible
    jump   = ((m >> 7) & 1) != 0   # ground-jump feasible
    ud = move_cls[:, 2]
    out[:, 2] = np.where((ud == 2) & (jump | up_pos), 2,
                         np.where((ud == 0) & up_neg, 0, 1))
    return out


def plan_batches(lengths: np.ndarray) -> list[list[int]]:
    """Partition episodes into padded-lane forward batches.

    Episodes are sorted by length (ascending) then packed greedily so that
    ``Tmax * B <= MAX_TB`` and ``B <= MAX_LANES``.  This mirrors the padded-
    forward strategy in ``move_temporal.py`` and ``jump_discrim.py``.

    Parameters
    ----------
    lengths : np.ndarray, shape (E,), dtype int
        Per-episode frame counts (``episode_offsets[1:] - episode_offsets[:-1]``).

    Returns
    -------
    list[list[int]]
        Each inner list is a batch of episode indices.
    """
    order = [int(i) for i in np.argsort(lengths) if lengths[i] > 0]
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for ei in order:
        L = int(lengths[ei])
        if cur and (max(cur_max, L) * (len(cur) + 1) > MAX_TB or len(cur) >= MAX_LANES):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(ei)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches
