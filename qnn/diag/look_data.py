"""Shared loader for look-head analysis scripts.

Provides ``load_pooled_look``, a per-shard function that reads the ragged
entity-token observation arrays from a precomputed cache shard and pools them
to per-frame (F, 3) target vectors weighted by ground-truth ``target_probs``.

Entity-token alignment (canonical reference: docs/target-head.md §2):
  - ``target_probs[:, 0]``  = NO_TARGET mass; not mapped to any entity token.
  - entity token ``j``      ↔ ``target_probs[:, j+1]``.

So for token ``t`` in frame ``f``, with within-frame index ``within = t - starts[f]``:
  col = (within + 1).clip(max=tp.shape[1] - 1)
  weight = tp[f, col]

This is the single authoritative implementation of the pooling kernel.  The nine
look scripts that inline this logic will be migrated to import it here in waves;
Phase 1 of analysis-diag-consolidation.md covers ``look_prior_fit`` and
``look_prior_explore4`` (the two canonical fit scripts).

No torch dependency; CPU-only numpy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

__all__ = ["load_pooled_look"]


def load_pooled_look(cache_dir: Path, shard: dict) -> Dict[str, np.ndarray]:
    """Load and pool one manifest shard's entity tokens to per-frame arrays.

    Reads the ragged entity-token observation arrays (``entity_rel``,
    ``entity_vel``, ``entity_count``) plus ``look`` and ``target_probs`` from
    *shard*, then performs soft-pooling: each entity token is weighted by the
    ground-truth ``target_probs`` mass assigned to that token's within-frame
    slot (``col = within + 1``, clamped to the last valid column).

    Parameters
    ----------
    cache_dir:
        Root directory of the precomputed cache (the directory that contains
        ``manifest.json`` and the shard ``.npy`` files).
    shard:
        One entry from ``manifest["shards"]``.  Must contain at least:
          - ``shard["obs"]["entity_rel"]``   — relative position array filename
          - ``shard["obs"]["entity_vel"]``   — relative velocity array filename
          - ``shard["obs"]["entity_count"]`` — per-frame entity count filename
          - ``shard["actions"]["look"]``     — look label filename
          - ``shard["actions"]["target_probs"]`` — target distribution filename
          - ``shard["episode_lengths"]``     — list/array of per-episode frame counts

    Returns
    -------
    dict with keys:
        ``pooled_rel``     — (F, 3) float64; soft-pooled entity relative position
        ``pooled_vel``     — (F, 3) float64; soft-pooled entity relative velocity
        ``look``           — (F, 3) float64; raw look label (unit direction)
        ``look_lag1``      — (F, 3) float64; look shifted by 1 within each episode
                             (zero-padded at episode starts); same as ``prev_look``
        ``episode_lengths``— 1-D int64 array; number of frames per episode
        ``target_present`` — (F,) float64; mass assigned to any target entity
                             (= 1 − target_probs[:, 0])
        ``target_probs``   — (F, P) float64; full target distribution matrix
        ``frame_id``       — (T,) int64; frame index for each entity token
        ``token_weight``   — (T,) float64; GT weight for each entity token

    Notes
    -----
    ``look_lag1[f]`` is the look *action* at the previous frame within the same
    episode, or a zero vector at episode starts.  It is the "momentum" / copycat
    reference; it should only be used as an upper-bound ceiling, not as a
    PPO-safe grounded feature (it feeds back a prior action).
    """
    obs = shard["obs"]
    act = shard["actions"]

    rel   = np.load(cache_dir / obs["entity_rel"]  ).astype(np.float64)  # (T, 3)
    vel   = np.load(cache_dir / obs["entity_vel"]  ).astype(np.float64)  # (T, 3)
    count = np.load(cache_dir / obs["entity_count"]).astype(np.int64)    # (F,)
    look  = np.load(cache_dir / act["look"]        ).astype(np.float64)  # (F, 3)
    tp    = np.load(cache_dir / act["target_probs"]).astype(np.float64)  # (F, P)

    F = count.shape[0]

    # Build per-frame start offsets into the flattened token array.
    starts = np.zeros(F, dtype=np.int64)
    np.cumsum(count[:-1], out=starts[1:])

    # Expand: for each token t, its frame and within-frame index.
    frame_id = np.repeat(np.arange(F), count)                       # (T,)
    within   = np.arange(rel.shape[0]) - starts[frame_id]           # (T,) 0-based in-frame idx

    # entity token j  ↔  target_probs[:, j+1]  (col 0 = no-target slot).
    col = (within + 1).clip(max=tp.shape[1] - 1)                    # (T,)
    token_weight = tp[frame_id, col]                                 # (T,)

    # Soft-pool: weighted sum of entity tokens → per-frame vectors.
    pooled_rel = np.zeros((F, 3), dtype=np.float64)
    pooled_vel = np.zeros((F, 3), dtype=np.float64)
    np.add.at(pooled_rel, frame_id, token_weight[:, None] * rel)
    np.add.at(pooled_vel, frame_id, token_weight[:, None] * vel)

    # look_lag1: prev look action within each episode (zero at episode starts).
    episode_lengths = np.asarray(shard["episode_lengths"], dtype=np.int64)
    look_lag1 = np.zeros_like(look)
    s = 0
    for n in episode_lengths:
        n = int(n)
        if n > 1:
            look_lag1[s + 1 : s + n] = look[s : s + n - 1]
        s += n

    return {
        "pooled_rel":      pooled_rel,
        "pooled_vel":      pooled_vel,
        "look":            look,
        "look_lag1":       look_lag1,
        "episode_lengths": episode_lengths,
        "target_present":  1.0 - tp[:, 0],
        "target_probs":    tp,
        "frame_id":        frame_id,
        "token_weight":    token_weight,
    }
