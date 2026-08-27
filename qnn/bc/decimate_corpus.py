"""Materialize a rate-decimated ON-DISK copy of a sharded_v1 BC collect.

Unlike ``qnn.bc.resample`` (load-time resampling, no second collect —
``StreamingSource`` composes ``R`` consecutive frames per group on the fly,
see that module's docstring for the full per-key aggregation contract), this
writes a standalone collect directory at the new rate so every corpus-walking
tool that expects a native_v1 ``sharded_v1`` collect on disk (grid fitters,
``qnn.human``, training) can consume the decimated corpus with zero code
changes — no loader-side resample wiring required.

This module does NOT reimplement any composition rule: every shard's arrays
are decimated by calling ``qnn.bc.resample.resample_shard`` exactly once,
per shard, with the shard's own episode boundaries and the token-indexed
obs keys it actually contains. This module owns only the DISK bookkeeping
around that call:

  * per-episode manifest metadata (``episode_lengths`` and any parallel
    per-episode lists such as ``demo_idxs`` / ``episode_idxs``) — episodes
    whose decimated length is 0 (source length < ratio) are DROPPED from
    these lists (``resample_shard`` itself still emits a 0-length entry
    for them, since it does not know about manifest bookkeeping; this
    module post-filters using that same per-episode ordering, so the row
    data resample_shard already produced needs no further edits — a
    zero-length episode contributed zero rows to begin with);
  * top-level ``collect_metadata.json`` / ``fingerprint.json`` /
    ``filter.json`` provenance, modeled on ``qnn.bc.cache_look_tan`` (the
    existing batch corpus-transform precedent) for CLI shape and
    multiprocessing style;
  * a per-shard progress print and a post-write verification pass.

``collect_metadata.json``'s ``look_grid`` and ``move_hazard`` blocks are
RATE-DEPENDENT fitted tables (Lloyd-Max magnitude bins / dwell-hazard
release curve, both fit at the source tick_hz) and are STRIPPED from the
destination metadata, never copied through. ``qnn.human.ensure_collect_tables``
gates its compute-if-missing on the block's mere PRESENCE in
``collect_metadata.json`` (not file existence, unlike the ``human_baseline/``
artifacts) — leaving the 20 Hz fit in a 10 Hz copy would make the refit
silently skip and every downstream run would adopt a wrong-rate grid.

Import-light by design: numpy + stdlib only (``qnn.bc.resample`` and its own
deferred ``qnn.bc.cache_look_tan`` import are numpy-only) so this runs in the
torch-free devcontainer python; it does NOT import ``qnn.bc.train`` (pulls in
torch) — the token-indexed obs field set below is a deliberate, commented
duplicate of ``qnn.bc.train._NATIVE_TOKEN_INDEXED_OBS_FIELDS``, cross-checked
at runtime against each shard's actual array shapes (fail loud on drift
rather than silently misclassifying a field).

Usage (devcontainer or CPU container):
  PYTHONPATH=src python -m qnn.bc.decimate_corpus \\
      artifacts/collect/qwd_v4d_v3vis artifacts/collect/qwd_v4d_v3vis_10hz --hz 10
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from qnn.bc.resample import resample_ratio, resample_shard

# Mirrors qnn.bc.train._NATIVE_TOKEN_INDEXED_OBS_FIELDS (train.py:845-852).
# Duplicated here (rather than imported) because qnn.bc.train imports torch,
# which this module must not require. If a graph adds a new entity token
# field, mirror it here too — _classify_obs_keys below fails loud on any
# obs key whose on-disk shape matches neither the row count nor the token
# count, so drift is caught rather than silently mis-resampled.
_TOKEN_OBS_KEYS = frozenset({
    "entity_types", "entity_subject_id", "entity_modality_id",
    "entity_player_id", "entity_event_count",
    "entity_event_actions", "entity_event_sources",
    "entity_half_extents", "entity_rel", "entity_vel",
    "entity_path", "entity_path_dist", "entity_eta", "entity_recency",
    "entity_facing", "entity_team", "entity_score",
    "entity_amount", "entity_regen", "entity_state",
})

_SPLITS = ("precomputed_train", "precomputed_val")

# Rate-dependent fitted tables (qnn.human) that must NOT survive into the
# decimated copy's collect_metadata.json — see module docstring.
_STRIP_METADATA_KEYS = ("look_grid", "move_hazard")

_NON_EPISODE_SHARD_KEYS = frozenset({"rows", "episode_lengths", "obs", "actions"})


def _ep_slices_from_lengths(lengths: list[int]) -> list[tuple[int, int]]:
    slices = []
    base = 0
    for L in lengths:
        L = int(L)
        slices.append((base, base + L))
        base += L
    return slices


def _classify_obs_keys(obs: dict[str, np.ndarray], n_rows: int) -> frozenset[str]:
    """Split ``obs`` into token-indexed vs row-indexed by cross-checking
    ``_TOKEN_OBS_KEYS`` membership against each array's actual on-disk
    length. Fails loud on any key whose shape matches neither the row
    count nor the token count (unknown layout — do not guess)."""
    entity_count = obs.get("entity_count")
    if entity_count is None:
        raise RuntimeError("shard obs has no entity_count — not a native_v1 shard")
    entity_count = np.asarray(entity_count)
    if entity_count.shape[0] != n_rows:
        raise RuntimeError(
            f"obs['entity_count'] has {entity_count.shape[0]} rows, expected {n_rows}")
    total_tokens = int(entity_count.astype(np.int64).sum())

    token_keys = set()
    for key, arr in obs.items():
        if key == "entity_count":
            continue
        arr = np.asarray(arr)
        if key in _TOKEN_OBS_KEYS:
            if arr.shape[0] != total_tokens:
                raise RuntimeError(
                    f"obs['{key}'] expected token-indexed length {total_tokens} "
                    f"(sum entity_count), got {arr.shape[0]}")
            token_keys.add(key)
        elif arr.shape[0] != n_rows:
            raise RuntimeError(
                f"obs['{key}'] has {arr.shape[0]} rows, expected {n_rows} row-indexed "
                f"(and it is not in _TOKEN_OBS_KEYS, where it would need "
                f"{total_tokens} token-indexed rows instead) — unknown obs layout, "
                f"add it to _TOKEN_OBS_KEYS if this is a new entity token field")
    return frozenset(token_keys)


def _filtered_parallel_episode_keys(shard: dict[str, Any], n_episodes: int) -> list[str]:
    """Optional per-episode metadata lists (demo_idxs, episode_idxs, ...):
    any shard key, other than the fixed structural ones, whose value is a
    list of exactly one entry per (source) episode."""
    keys = []
    for k, v in shard.items():
        if k in _NON_EPISODE_SHARD_KEYS:
            continue
        if isinstance(v, list) and len(v) == n_episodes:
            keys.append(k)
    return keys


def _validate_manifest_format(manifest: dict[str, Any], path: Path) -> None:
    if not isinstance(manifest, dict) or manifest.get("format") != "sharded_v1":
        raise RuntimeError(f"{path}: expected format='sharded_v1'")
    if manifest.get("format_version") != "native_v1":
        raise RuntimeError(
            f"{path}: expected format_version='native_v1', got "
            f"{manifest.get('format_version')!r}")


def _process_shard(
    args: tuple[str, str, int, dict[str, Any], int],
) -> tuple[int, dict[str, Any], int]:
    """Worker: resample one shard and write its arrays to the destination
    split dir. Returns ``(shard_idx, new_shard_manifest_entry, kept_rows)``.
    Only this shard's arrays are ever resident in memory at once."""
    src_dir_s, dst_dir_s, shard_idx, shard, ratio = args
    src_dir = Path(src_dir_s)
    dst_dir = Path(dst_dir_s)

    ep_lengths = [int(x) for x in shard["episode_lengths"]]
    n_rows = int(shard["rows"])
    ep_slices = _ep_slices_from_lengths(ep_lengths)

    obs = {k: np.load(src_dir / fn, mmap_mode="r") for k, fn in shard["obs"].items()}
    actions = {k: np.load(src_dir / fn, mmap_mode="r") for k, fn in shard["actions"].items()}

    token_keys = _classify_obs_keys(obs, n_rows)
    new_obs, new_act, new_ep_lengths = resample_shard(
        obs, actions, ep_slices, ratio, token_obs_keys=token_keys,
    )

    n_orig_episodes = len(ep_lengths)
    parallel_keys = _filtered_parallel_episode_keys(shard, n_orig_episodes)
    keep_mask = [L > 0 for L in new_ep_lengths]
    kept_lengths = [L for L, keep in zip(new_ep_lengths, keep_mask) if keep]

    new_shard: dict[str, Any] = {}
    for k in shard.keys():
        if k == "rows":
            new_shard[k] = int(sum(kept_lengths))
        elif k == "episode_lengths":
            new_shard[k] = kept_lengths
        elif k in parallel_keys:
            new_shard[k] = [v for v, keep in zip(shard[k], keep_mask) if keep]
        elif k in ("obs", "actions"):
            new_shard[k] = dict(shard[k])  # same filenames as source
        else:
            new_shard[k] = shard[k]

    for k, fn in shard["obs"].items():
        np.save(dst_dir / fn, new_obs[k])
    for k, fn in shard["actions"].items():
        np.save(dst_dir / fn, new_act[k])

    return shard_idx, new_shard, int(sum(kept_lengths))


def _materialize_split(src_split: Path, dst_split: Path, ratio: int, workers: int) -> None:
    manifest = json.loads((src_split / "manifest.json").read_text())
    _validate_manifest_format(manifest, src_split / "manifest.json")
    dst_split.mkdir(parents=True, exist_ok=True)

    shards = manifest["shards"]
    n = len(shards)
    tasks = [(str(src_split), str(dst_split), i, shard, ratio) for i, shard in enumerate(shards)]

    results: dict[int, tuple[dict[str, Any], int]] = {}
    if workers <= 1 or n <= 1:
        for t in tasks:
            idx, new_shard, rows = _process_shard(t)
            results[idx] = (new_shard, rows)
            print(f"  [{src_split.name}] shard {idx + 1}/{n}: {rows} rows written", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_process_shard, t): t[2] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                idx, new_shard, rows = fut.result()
                results[idx] = (new_shard, rows)
                done += 1
                print(f"  [{src_split.name}] shard {done}/{n} (idx {idx}): "
                      f"{rows} rows written", flush=True)

    new_shards = [results[i][0] for i in range(n)]
    new_manifest = dict(manifest)
    new_manifest["shards"] = new_shards
    new_manifest["episodes"] = sum(len(s["episode_lengths"]) for s in new_shards)
    (dst_split / "manifest.json").write_text(json.dumps(new_manifest, indent=1) + "\n")


def _verify_split(src_split: Path, dst_split: Path, ratio: int) -> None:
    """Post-write sanity pass: manifest row-count invariants for every
    shard, plus a byte-identical direct-recompute check on one shard."""
    manifest = json.loads((dst_split / "manifest.json").read_text())
    shards = manifest["shards"]

    assert manifest["episodes"] == sum(len(s["episode_lengths"]) for s in shards), (
        f"{dst_split}: top-level episodes count doesn't match per-shard sums")

    for i, shard in enumerate(shards):
        assert shard["rows"] == sum(shard["episode_lengths"]), (
            f"{dst_split}: shard {i} rows {shard['rows']} != "
            f"sum(episode_lengths) {sum(shard['episode_lengths'])}")
        obs = {k: np.load(dst_split / fn, mmap_mode="r") for k, fn in shard["obs"].items()}
        token_keys = _classify_obs_keys(obs, shard["rows"])
        total_tokens = int(np.asarray(obs["entity_count"]).astype(np.int64).sum())
        for k in token_keys:
            assert obs[k].shape[0] == total_tokens, (
                f"{dst_split}: shard {i} obs['{k}'] length {obs[k].shape[0]} != "
                f"sum(entity_count) {total_tokens}")

    # One-shard byte-identical direct recompute (shard 0).
    src_manifest = json.loads((src_split / "manifest.json").read_text())
    src_shard = src_manifest["shards"][0]
    dst_shard = shards[0]

    src_obs = {k: np.load(src_split / fn) for k, fn in src_shard["obs"].items()}
    src_act = {k: np.load(src_split / fn) for k, fn in src_shard["actions"].items()}
    ep_lengths = [int(x) for x in src_shard["episode_lengths"]]
    ep_slices = _ep_slices_from_lengths(ep_lengths)
    token_keys = _classify_obs_keys(src_obs, int(src_shard["rows"]))

    direct_obs, direct_act, direct_lengths = resample_shard(
        src_obs, src_act, ep_slices, ratio, token_obs_keys=token_keys,
    )
    kept_lengths = [L for L in direct_lengths if L > 0]
    assert kept_lengths == dst_shard["episode_lengths"], (
        f"{dst_split}: shard 0 episode_lengths mismatch — direct recompute "
        f"{kept_lengths} vs written {dst_shard['episode_lengths']}")

    dst_obs = {k: np.load(dst_split / fn) for k, fn in dst_shard["obs"].items()}
    dst_act = {k: np.load(dst_split / fn) for k, fn in dst_shard["actions"].items()}
    for k, arr in direct_obs.items():
        np.testing.assert_array_equal(arr, dst_obs[k], err_msg=f"shard 0 obs['{k}']")
    for k, arr in direct_act.items():
        np.testing.assert_array_equal(arr, dst_act[k], err_msg=f"shard 0 actions['{k}']")


def _write_dst_metadata(
    src_meta: dict[str, Any], dst: Path, src_name: str, src_hz: float, dst_hz: float, ratio: int,
) -> None:
    meta = dict(src_meta)
    for k in _STRIP_METADATA_KEYS:
        meta.pop(k, None)
    meta["tick_hz"] = dst_hz
    meta["decimated"] = {
        "from": src_name,
        "src_tick_hz": src_hz,
        "ratio": ratio,
        "tool": "qnn.bc.decimate_corpus",
    }
    (dst / "collect_metadata.json").write_text(json.dumps(meta, indent=1) + "\n")


def _write_dst_fingerprint(src: Path, dst: Path, ratio: int) -> None:
    src_fp = json.loads((src / "fingerprint.json").read_text())
    obs_from = src_fp.get("fingerprint")
    manifest_bytes = b"".join(
        (dst / split / "manifest.json").read_bytes()
        for split in _SPLITS if (dst / split / "manifest.json").exists()
    )
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    fp = {
        "fingerprint": f"sha256:{digest}",
        "components": {
            "obs_from": obs_from,
            "decimated_by": "qnn.bc.decimate_corpus",
            "ratio": ratio,
        },
        "created": datetime.date.today().isoformat(),
    }
    (dst / "fingerprint.json").write_text(json.dumps(fp, indent=1) + "\n")


def run(src: Path | str, dst: Path | str, dst_hz: float, *, workers: int | None = None) -> None:
    """Materialize a rate-decimated on-disk copy of ``src`` (a sharded_v1
    native_v1 collect) into ``dst`` at ``dst_hz``. Fails loud on: missing/
    wrong-format src, a non-integer rate ratio, or an already-populated
    ``dst`` (never silently overwrites)."""
    src = Path(src)
    dst = Path(dst)

    if not src.is_dir():
        raise FileNotFoundError(f"src collect not found: {src}")
    meta_path = src / "collect_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing — {src} is not a collect dir")
    src_meta = json.loads(meta_path.read_text())
    src_hz = src_meta.get("tick_hz")
    if src_hz is None:
        raise ValueError(f"{meta_path} has no tick_hz")

    ratio = resample_ratio(src_hz, dst_hz)  # raises ValueError: non-integer / upsample / <=0

    if dst.exists():
        if not dst.is_dir():
            raise RuntimeError(f"dst exists and is not a directory: {dst}")
        if any(dst.iterdir()):
            raise RuntimeError(
                f"dst collect already exists and is non-empty: {dst} — refusing to "
                "overwrite; remove it first")
    dst.mkdir(parents=True, exist_ok=True)

    present_splits = [s for s in _SPLITS if (src / s / "manifest.json").exists()]
    if not present_splits:
        raise RuntimeError(
            f"{src}: neither precomputed_train nor precomputed_val has a manifest.json")

    workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    workers = max(1, workers)

    for split in present_splits:
        print(f"[decimate] {split}: {src_hz}Hz -> {dst_hz}Hz (ratio {ratio}), "
              f"{workers} worker(s)", flush=True)
        _materialize_split(src / split, dst / split, ratio, workers)
        print(f"[decimate] {split}: verifying …", flush=True)
        _verify_split(src / split, dst / split, ratio)
        print(f"[decimate] {split}: OK", flush=True)

    _write_dst_metadata(src_meta, dst, src.name, src_hz, dst_hz, ratio)
    _write_dst_fingerprint(src, dst, ratio)

    filter_path = src / "filter.json"
    if filter_path.exists():
        shutil.copyfile(filter_path, dst / "filter.json")

    print(f"[decimate] done: {dst}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="source sharded_v1/native_v1 collect dir")
    ap.add_argument("dst", help="destination dir to materialize into "
                                "(must not already exist non-empty)")
    ap.add_argument("--hz", type=float, required=True, help="destination tick rate")
    ap.add_argument("--workers", type=int, default=None,
                     help="parallel workers across shards (default min(8, cpu count))")
    args = ap.parse_args(argv)
    run(args.src, args.dst, args.hz, workers=args.workers)


if __name__ == "__main__":
    main()
