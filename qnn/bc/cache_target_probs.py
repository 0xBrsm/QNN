"""Populate the target_probs sidecar cache for an existing BC corpus.

Walks each shard in ``<cache_dir>/precomputed_train`` and
``<cache_dir>/precomputed_val``, runs the target labeler, and writes
``shard_NNNNNN_act_target_probs.npy`` (float16, ``(T_shard, 17)``) next
to the existing shard files. Updates ``manifest.json`` so the shard's
``actions`` dict references the new file — train load picks it up
automatically.

Use this when you have an existing native_v1 cache (no ``target_probs``
field in any shard's actions dict) and want to convert it to a cached
form without recollecting from demos.

Usage:
    python -m qnn.bc.cache_target_probs artifacts/collect/qwd

Workers: defaults to min(30, cpu_count). Per-shard compute is ~8-10s
on the production corpus; full corpus (117+13 shards) lands in
~3-5 minutes with 30 workers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from qnn.bc.target_labeler import (
    label_enemy_target_probs as _LABEL_TARGETS,
    DEFAULT_LABELER_CONFIG as _LABELER_DEFAULT_CONFIG,
)
from qnn.bc.train import (
    _densify_obs_for_labeler,
    _materialize_padded_entity,
    _unpack_attack_bit,
    _unpack_move_axes,
)
from qnn.vocab import MAX_TOKEN_OBJECTS


def _populate_one_shard(args: tuple) -> tuple[int, str, int]:
    """Worker: compute target_probs for one shard, write to disk.

    Returns (shard_idx, target_probs_fname, n_rows) so the caller can
    update the manifest. Skips if the shard already has target_probs.
    """
    split_dir_str, shard_idx, shard = args
    split_dir = Path(split_dir_str)

    if "target_probs" in shard.get("actions", {}):
        # Already cached — nothing to do for this shard.
        return shard_idx, shard["actions"]["target_probs"], 0

    obs_arrays = {
        key: np.load(split_dir / fname, mmap_mode="r")
        for key, fname in shard["obs"].items()
    }
    # u16/u32 → i32 to match the chunked training gather path.
    obs_arrays = {
        key: (np.asarray(arr).astype(np.int32, copy=False)
              if arr.dtype in (np.uint16, np.uint32)
              else arr)
        for key, arr in obs_arrays.items()
    }
    action_arrays = {
        head: np.load(split_dir / fname, mmap_mode="r")
        for head, fname in shard["actions"].items()
    }
    if "move" in action_arrays:
        move_packed = action_arrays["move"]
        action_arrays["move"] = _unpack_move_axes(move_packed)
        action_arrays["attack"] = _unpack_attack_bit(move_packed)

    # Pad + densify the entire shard (T_shard rows). Same path the
    # live train loader takes when target_probs isn't cached.
    shard_obs_padded = _materialize_padded_entity(
        {k: np.asarray(v) for k, v in obs_arrays.items()},
        MAX_TOKEN_OBJECTS,
    )
    shard_dense = _densify_obs_for_labeler(shard_obs_padded)

    # The labeler is episode-local (fire_ticks + temporal extension
    # respect episode boundaries) so we run per-episode and concat.
    episode_lengths = shard.get("episode_lengths", [])
    pieces = []
    start = 0
    for n_samples in episode_lengths:
        end = start + int(n_samples)
        sub_dense = {k: v[start:end] for k, v in shard_dense.items()}
        sub_act = {k: v[start:end] for k, v in action_arrays.items()}
        td = _LABEL_TARGETS(sub_dense, sub_act, config=_LABELER_DEFAULT_CONFIG)
        pieces.append(td.astype(np.float16, copy=False))
        start = end
    td_shard = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 17), dtype=np.float16)

    fname = f"shard{shard_idx:06d}_act_target_probs.npy"
    np.save(split_dir / fname, td_shard)
    return shard_idx, fname, int(td_shard.shape[0])


def _populate_split(split_dir: Path, n_workers: int) -> None:
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = manifest["shards"]
    print(f"  {split_dir.name}: {len(shards)} shards")

    cached_already = sum(1 for s in shards if "target_probs" in s.get("actions", {}))
    pending = [(str(split_dir), i, s) for i, s in enumerate(shards)
               if "target_probs" not in s.get("actions", {})]
    if not pending:
        print(f"    all {len(shards)} shards already cached")
        return

    ctx = mp.get_context("fork")
    completed = 0
    total_rows = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(_populate_one_shard, a): a[1] for a in pending}
        for fut in as_completed(futures):
            shard_idx, fname, n_rows = fut.result()
            shards[shard_idx].setdefault("actions", {})["target_probs"] = fname
            completed += 1
            total_rows += n_rows
            if completed % 10 == 0 or completed == len(pending):
                print(f"    {completed}/{len(pending)} shards  ({total_rows:,} rows cached)")

    # Persist manifest with the new target_probs references.
    manifest_path.write_text(json.dumps(manifest))
    print(f"    manifest updated: {manifest_path}")
    if cached_already:
        print(f"    ({cached_already} shards were already cached)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate target_probs cache for an existing BC corpus.")
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="Path to the BC cache root containing precomputed_{train,val}/.",
    )
    parser.add_argument(
        "--workers", type=int,
        default=min(30, os.cpu_count() or 4),
        help="Parallel workers (default: min(30, cpu_count)).",
    )
    args = parser.parse_args()

    for split in ("precomputed_train", "precomputed_val"):
        split_dir = args.cache_dir / split
        if split_dir.is_dir():
            _populate_split(split_dir, args.workers)
        else:
            print(f"  {split}: missing, skipping")


if __name__ == "__main__":
    main()
