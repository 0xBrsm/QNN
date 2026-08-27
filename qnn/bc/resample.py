"""Arbitrary integer-ratio temporal resampling of a native_v1 corpus, applied
at load time (no second collect, no disk cache).

A model trained at ``target_hz`` from a ``corpus_hz`` collect groups every
``R = corpus_hz // target_hz`` consecutive frames *within an episode* into one
output frame:

* obs (row + token-indexed entity fields): take the FIRST frame of each group
  (the state at the group's decision time).
* look / look_tan: compose the R per-tick view turns (sum yaw/pitch) -> the net
  turn over the group. Decimation would undercount the turn by ~R.
* attack (0..8 discharge): the group's discharge (first nonzero across the R).
* input_mask / op_input: OR across the group (operative/feasible in any sub-tick).
* move (packed byte): direction bits from the first frame (the one lossy field —
  physics-rate-specific, no native anchor to re-derive); attack(bit0)+jump(bit7)
  OR'd across the group.
* weapon / target_probs: first frame.
* episode length L -> L // R (trailing remainder dropped).

Rate-dependent DERIVED columns (attack/jump distance-to-pos, look_delta,
engagement EMA, attack-shift, move-hazard) are NOT resampled here — the loader
recomputes them from the resampled base via the same derivation functions, so
they come out at ``target_hz`` for free.
"""
from __future__ import annotations

import numpy as np

# packed-move bit layout (qnn.bc.collect / QNN_PackInputMask)
_MOVE_DIR_BITS = 0b01111110  # fb/lr/ud direction bits
_MOVE_OR_BITS = 0b10000001   # attack(bit0) + jump(bit7)


def resample_ratio(corpus_hz: float, target_hz: float) -> int:
    """Integer group size; raises if the rates aren't an exact integer ratio."""
    if target_hz <= 0 or corpus_hz <= 0:
        raise ValueError(f"hz must be positive, got corpus={corpus_hz} target={target_hz}")
    if target_hz > corpus_hz:
        raise ValueError(f"cannot upsample: target {target_hz} > corpus {corpus_hz}")
    r = corpus_hz / target_hz
    ri = int(round(r))
    if abs(r - ri) > 1e-6:
        raise ValueError(
            f"resample needs an integer ratio; corpus {corpus_hz} / target {target_hz} "
            f"= {r:.4f} is not integral")
    return ri


def group_indices(episode_lengths, ratio: int):
    """Per-episode group-first row indices (into the concatenated array) and the
    new (resampled) episode lengths. ``keep[o]`` for o in 0..ratio-1 gives the
    o-th member of every group, so aggregations are ``ratio`` strided gathers."""
    keep = [np.empty(0, np.int64) for _ in range(ratio)]
    base = 0
    new_lengths = []
    ka = []
    for L in episode_lengths:
        n = int(L) // ratio
        j = np.arange(n, dtype=np.int64)
        ka.append([base + ratio * j + o for o in range(ratio)])
        new_lengths.append(n)
        base += int(L)
    keep = [np.concatenate([a[o] for a in ka]) if ka else np.empty(0, np.int64)
            for o in range(ratio)]
    return keep, new_lengths


def _compose_look(look_members):
    """Compose R view-relative unit-vector turns -> net turn (sum yaw/pitch)."""
    yaw = np.zeros(look_members[0].shape[0], np.float64)
    pitch = np.zeros_like(yaw)
    for v in look_members:
        v = v.astype(np.float64)
        yaw += np.arctan2(v[:, 1], v[:, 0])
        pitch += np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))
    return np.stack([np.cos(pitch) * np.cos(yaw),
                     np.cos(pitch) * np.sin(yaw),
                     np.sin(pitch)], axis=1)


def aggregate_action(key: str, arr: np.ndarray, keep, ratio: int) -> np.ndarray:
    """Resample one action array over the group members ``keep`` (list of ratio
    index arrays). ``arr`` is the full (T, ...) source array."""
    members = [arr[k] for k in keep]
    if key == "look":
        return _compose_look(members).astype(arr.dtype)
    if key == "look_tan":
        return members[0]  # placeholder — loader recomputes from resampled look
    if key == "attack":
        out = members[0].copy()
        for m in members[1:]:
            out = np.where(out > 0, out, m)
        return out.astype(arr.dtype)
    if key in ("input_mask", "op_input"):
        out = members[0].copy()
        for m in members[1:]:
            out = out | m
        return out.astype(arr.dtype)
    if key == "move" and arr.ndim == 1:  # packed byte
        or_bits = members[0].copy()
        for m in members[1:]:
            or_bits = or_bits | m
        return ((members[0] & _MOVE_DIR_BITS) | (or_bits & _MOVE_OR_BITS)).astype(arr.dtype)
    # weapon, target_probs, unpacked move axes, anything else: take the first
    return members[0]


def resample_shard(obs, actions, ep_slices, ratio, *, token_obs_keys, look_tan_from_look=True):
    """Resample one shard's raw arrays (packed ``move``, flat entity obs) per
    episode. ``ep_slices`` is the list of ``(row_start, row_end)`` for the
    episodes contained in this shard, in row order. ``token_obs_keys`` is the
    set of entity token-indexed obs keys (flat ``(total_tokens, ...)``, sliced
    by the row-indexed ``entity_count``). Returns ``(new_obs, new_actions,
    new_ep_lengths)`` — all row counts divided by ``ratio``.
    """
    entity_count = np.asarray(obs["entity_count"])
    tok_offsets = np.concatenate([[0], np.cumsum(entity_count.astype(np.int64))])

    new_ep_lengths = []
    row_keep_parts = [[] for _ in range(ratio)]   # group-member row indices (abs in shard)
    tok_keep_parts = []                           # token indices for kept rows (group-first)
    for (r0, r1) in ep_slices:
        L = int(r1 - r0)
        n = L // ratio
        j = np.arange(n, dtype=np.int64)
        for o in range(ratio):
            row_keep_parts[o].append(r0 + ratio * j + o)
        first_rows = r0 + ratio * j
        # tokens for the group-first rows (entity obs takes the first frame)
        lens = entity_count[first_rows].astype(np.int64)
        if lens.sum():
            ends = np.cumsum(lens)
            within = np.arange(int(lens.sum())) - np.repeat(ends - lens, lens)
            tok_keep_parts.append(np.repeat(tok_offsets[first_rows], lens) + within)
        new_ep_lengths.append(n)
    keep = [np.concatenate(p) if p else np.empty(0, np.int64) for p in row_keep_parts]
    tok_idx = np.concatenate(tok_keep_parts) if tok_keep_parts else np.empty(0, np.int64)

    new_obs = {}
    for k, arr in obs.items():
        arr = np.asarray(arr)
        if k == "entity_count":
            new_obs[k] = arr[keep[0]]
        elif k in token_obs_keys:
            new_obs[k] = arr[tok_idx]
        else:  # row-indexed self/spatial obs -> first frame of each group
            new_obs[k] = arr[keep[0]]
    new_act = {}
    for k, arr in actions.items():
        new_act[k] = aggregate_action(k, np.asarray(arr), keep, ratio)
    if look_tan_from_look and "look_tan" in new_act and "look" in new_act:
        from qnn.bc.cache_look_tan import look_to_tangent
        new_act["look_tan"] = look_to_tangent(new_act["look"].astype(np.float64)).astype(
            np.asarray(actions["look_tan"]).dtype)
    return new_obs, new_act, new_ep_lengths
