"""Derive the a25 move-hazard label columns from the move-action stream.

The hazard head's three inputs/labels are all deterministic functions of the
per-axis move-class sequence (`act_move`, classes {0:neg, 1:none, 2:pos}) — NO
recollect and NO cache growth. They are CROSS-FRAME (need t-1 / t+1 in temporal
order, reset at episode boundaries), so they cannot be derived from a FLATTENED
training batch — but they ARE derived **on the fly in the loader**, per episode,
while the time axis + episode boundary are intact (qnn.bc.streaming_source
``read_rows``, the same hook that derives the attack bit). This head is
temporal-only (full_6head carries the GRU), so it never trains on the
non-temporal frame-shuffle path where the time axis is gone — nothing is written
to disk. See src/docs/move-head.md.

Convention (matches the engine's ``MoveDecodeState`` threading — ``prev_move``
from tick t-1 feeds the tick-t release decision):

  held_class[t] = class[t-1]           # the class held entering frame t
  dwell_age[t]  = run length of class ending at t-1   (reset to 1 at ep start)
  release[t]    = 1[class[t] != class[t-1]]            # did the human release?
  valid[t]      = t is not an episode start            # t-1 must exist

At an episode start there is no prior frame, so held_class/dwell_age take the
``move_decode_reset`` values (none / 1) and ``valid`` is 0 (masked from the
loss). Per axis, independently (the move axes are ~independent — move-head.md
§4). ``valid`` here is the episode-boundary/right-censoring mask only; the
caller ANDs it with the in-distribution segment mask.

Pure numpy, cheap enough to run per-load in the data loader. ``move_classes``
is ``(T, MOVE_AXES)`` int for a single contiguous episode; the loader scans from
the true episode start (``EpisodeRef.row_start``) so a sub-window's dwell_age is
exact. ``derive_hazard_labels`` runs one episode, ``derive_hazard_labels_batched``
maps it over episode-sliced arrays.
"""

from __future__ import annotations

import numpy as np

from qnn.actions import MOVE_AXES, MOVE_CLASS_NONE

# Reset values mirror qnn.model.bench.a24.decode.move_decode_reset.
_RESET_HELD = MOVE_CLASS_NONE  # 1
_RESET_DWELL = 1


def derive_hazard_labels(move_classes: np.ndarray) -> dict[str, np.ndarray]:
    """One contiguous episode → the four hazard columns.

    Args:
      move_classes: ``(T, MOVE_AXES)`` int array of per-axis classes (0/1/2).

    Returns a dict of ``(T, MOVE_AXES)`` arrays: ``held_class`` (int64),
    ``dwell_age`` (int64), ``release`` (float32 {0,1}), ``valid`` (bool).
    """
    cls = np.asarray(move_classes, dtype=np.int64)
    if cls.ndim != 2 or cls.shape[1] != MOVE_AXES:
        raise ValueError(f"move_classes must be (T, {MOVE_AXES}); got {cls.shape}")
    T = cls.shape[0]

    held = np.full((T, MOVE_AXES), _RESET_HELD, dtype=np.int64)
    dwell = np.full((T, MOVE_AXES), _RESET_DWELL, dtype=np.int64)
    release = np.zeros((T, MOVE_AXES), dtype=np.float32)
    valid = np.zeros((T, MOVE_AXES), dtype=bool)
    if T <= 1:
        return {"held_class": held, "dwell_age": dwell, "release": release, "valid": valid}

    prev = cls[:-1]                       # class[t-1] for t in 1..T-1
    cur = cls[1:]                         # class[t]
    held[1:] = prev
    release[1:] = (cur != prev).astype(np.float32)
    valid[1:] = True

    # dwell_age[t] = run length of class[t-1] ending at t-1. Equivalent: the
    # consecutive count of `cls` ending at index t-1. Vectorized run-length:
    same = cls[1:] == cls[:-1]            # (T-1,) per axis: class[i]==class[i-1]
    run_end = np.ones((T, MOVE_AXES), dtype=np.int64)  # run length ending at index i
    for i in range(1, T):
        run_end[i] = np.where(same[i - 1], run_end[i - 1] + 1, 1)
    dwell[1:] = run_end[:-1]              # dwell at t = run length ending at t-1
    return {"held_class": held, "dwell_age": dwell, "release": release, "valid": valid}


def derive_hazard_labels_batched(
    move_classes: np.ndarray,
    episode_lengths: "list[int] | np.ndarray",
) -> dict[str, np.ndarray]:
    """Map :func:`derive_hazard_labels` over concatenated episodes.

    ``move_classes`` is ``(sum(episode_lengths), MOVE_AXES)``; the scan resets at
    every episode boundary so dwell/held/valid never leak across episodes.
    """
    cls = np.asarray(move_classes, dtype=np.int64)
    lengths = [int(n) for n in episode_lengths]
    if sum(lengths) != cls.shape[0]:
        raise ValueError(
            f"episode_lengths sum {sum(lengths)} != frames {cls.shape[0]}"
        )
    out = {k: [] for k in ("held_class", "dwell_age", "release", "valid")}
    off = 0
    for n in lengths:
        ep = derive_hazard_labels(cls[off:off + n])
        for k, v in ep.items():
            out[k].append(v)
        off += n
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}
