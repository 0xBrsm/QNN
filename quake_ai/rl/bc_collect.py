#!/usr/bin/env python3
"""Collect BC training data by replaying demos through the demo worker.

Replays .dem files in parallel, reads obs buffer + action labels directly
from the demo worker, splits into train/val, saves as memory-mapped .npy.

One script. Supports resume — already-collected demos are skipped.

Usage:
    python -m quake_ai.rl.bc_collect \
        --demo-dir assets/demos \
        --output assets/bc

    # Resume after interruption (skips completed demos):
    python -m quake_ai.rl.bc_collect \
        --demo-dir assets/demos \
        --output assets/bc

    # Customize parallelism:
    python -m quake_ai.rl.bc_collect \
        --demo-dir assets/demos \
        --output assets/bc \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np

from quake_ai.obs_format import (
    OBS_BUFFER_SIZE, ACTION_SIZE, TICK_HEADER_SIZE,
    SELF_FIELDS, ACTION_FIELDS, FLAG_RESET, FLAG_DONE,
    ENTITY_STREAM_OFFSET,
    unpack_obs_buffer,
)

TICK_TOTAL_SIZE = TICK_HEADER_SIZE + OBS_BUFFER_SIZE + ACTION_SIZE


def _start_worker(demo_worker: str, asset_root: str, resample_hz: int) -> subprocess.Popen:
    env = {**os.environ, "QUAKE_BASEDIR": str(Path(asset_root).resolve())}
    proc = subprocess.Popen(
        [demo_worker, "-game", "demos"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    hello = json.dumps({"op": "hello", "map_id": "start", "resample_hz": resample_hz}) + "\n"
    proc.stdin.write(hello.encode())
    proc.stdin.flush()
    resp = proc.stdout.readline()
    if not resp or b'"ok":true' not in resp:
        err = proc.stderr.read(500).decode(errors="replace")
        raise RuntimeError(f"Demo worker hello failed: {err}")
    return proc


def _collect_one_demo(proc: subprocess.Popen, demo_name: str, trim_match: bool = False) -> list[dict] | None:
    cmd = json.dumps({"op": "collect", "demo_path": demo_name, "seed": 0, "trim_match": 1 if trim_match else 0}) + "\n"
    proc.stdin.write(cmd.encode())
    proc.stdin.flush()

    ticks = []
    while True:
        if proc.poll() is not None:
            return None
        magic = proc.stdout.read(4)
        if not magic or len(magic) < 4:
            return None
        if magic[0:1] == b'{':
            rest = proc.stdout.readline()
            try:
                err = json.loads(magic + rest)
                raise RuntimeError(err.get("error", "unknown error"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(f"Worker error: {(magic + rest)[:200]!r}")
        if magic != b'QOBS':
            raise RuntimeError(f"Bad magic: {magic!r}")
        raw = proc.stdout.read(TICK_TOTAL_SIZE) if proc.poll() is None else b""
        if len(raw) < TICK_TOTAL_SIZE:
            return None

        header = struct.unpack_from("<IIIHH", raw)
        flags = header[3]
        ticks.append({
            "obs": raw[TICK_HEADER_SIZE:TICK_HEADER_SIZE + OBS_BUFFER_SIZE],
            "action": raw[TICK_HEADER_SIZE + OBS_BUFFER_SIZE:],
            "done": bool(flags & FLAG_DONE),
        })
        if flags & FLAG_DONE:
            break
    return ticks


def _save_episode_npy(output_dir: Path, episode_index: int, ticks: list[dict]) -> dict:
    """Unpack v9 obs buffers and action labels, save as per-field .npy files."""
    n = len(ticks)
    prefix = f"ep{episode_index:04d}"

    # Unpack each tick's obs buffer using the v9 parser
    obs_list = [unpack_obs_buffer(t["obs"]) for t in ticks]

    # Remove all god-mode ticks (health > 250, normalized > 2.5).
    # These are pre/post match warmup, match abort, or respawn
    # invulnerability from mods that set health to 666/998.
    keep = [i for i, obs in enumerate(obs_list) if obs["self_scalars"][0] <= 2.5]
    if len(keep) < n:
        ticks = [ticks[i] for i in keep]
        obs_list = [obs_list[i] for i in keep]
        n = len(ticks)
    if n == 0:
        return {"n_samples": 0, "obs": {}, "actions": {}}

    # Stack obs fields into (n, ...) arrays
    obs_arrays: dict[str, np.ndarray] = {}
    for key in obs_list[0]:
        obs_arrays[key] = np.stack([obs[key] for obs in obs_list], axis=0)

    # Unpack action fields (fixed layout)
    action_blob = b"".join(t["action"] for t in ticks)
    act_arrays: dict[str, np.ndarray] = {}
    for name, (offset, dtype, shape) in ACTION_FIELDS.items():
        count = int(np.prod(shape)) if shape else 1
        arr = np.empty((n, *shape) if shape else (n,), dtype=dtype)
        for i in range(n):
            vals = np.frombuffer(action_blob, dtype=dtype, offset=i * ACTION_SIZE + offset, count=count)
            arr[i] = vals.reshape(shape) if shape else vals[0]
        act_arrays[name] = arr

    # Save
    for name, arr in obs_arrays.items():
        np.save(output_dir / f"{prefix}_obs_{name}.npy", arr)
    for name, arr in act_arrays.items():
        np.save(output_dir / f"{prefix}_act_{name}.npy", arr)

    return {
        "n_samples": n,
        "obs": {name: f"{prefix}_obs_{name}.npy" for name in obs_arrays},
        "actions": {name: f"{prefix}_act_{name}.npy" for name in ACTION_FIELDS},
    }


def _collect_and_save(args: tuple) -> dict | None:
    """Collect one demo and save to staging. Runs in a worker process."""
    demo_name, episode_index, stage_dir, demo_worker, asset_root, resample_hz, trim_match = args

    try:
        proc = _start_worker(demo_worker, asset_root, resample_hz)
    except Exception as exc:
        return {"demo": demo_name, "status": "error", "msg": str(exc)[:200]}

    try:
        ticks = _collect_one_demo(proc, demo_name, trim_match=trim_match)
    except Exception as exc:
        try:
            proc.kill()
        except Exception:
            pass
        return {"demo": demo_name, "status": "error", "msg": str(exc)[:200]}

    try:
        proc.stdin.write(json.dumps({"op": "shutdown"}).encode() + b"\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    if ticks is None:
        return {"demo": demo_name, "status": "crash"}
    if len(ticks) < 10:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}

    entry = _save_episode_npy(Path(stage_dir), episode_index, ticks)
    meta = {"demo": demo_name, "entry": entry}
    (Path(stage_dir) / f"ep{episode_index:04d}_meta.json").write_text(json.dumps(meta))
    return {"demo": demo_name, "status": "ok", "ticks": len(ticks)}


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load demo manifest, return list of entries."""
    entries = []
    for line in manifest_path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BC training data from demo files")
    parser.add_argument("--demo-dir", required=True, help="Directory containing .dem files")
    parser.add_argument("--output", required=True, help="Output directory for .npy caches")
    parser.add_argument("--manifest", default="", help="Path to manifest.ndjson (default: <demo-dir>/manifest.ndjson)")
    parser.add_argument("--demo-worker", default="assets/bin/quake_demo_worker")
    parser.add_argument("--asset-root", default="assets")
    parser.add_argument("--resample-hz", type=int, default=20, help="Emission rate (native rate auto-detected)")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = CPU count - 2)")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    output = Path(args.output)
    demo_worker = str(Path(args.demo_worker).resolve())

    # Load manifest if available, filter to included demos
    manifest_path = Path(args.manifest) if args.manifest else demo_dir / "manifest.ndjson"
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        included = [e["file"] for e in manifest if not e.get("bc_exclude", False)]
        excluded = [e["file"] for e in manifest if e.get("bc_exclude", False)]
        demos = sorted([demo_dir / name for name in included if (demo_dir / name).exists()])
        print(f"Manifest: {len(manifest)} entries, {len(excluded)} excluded, {len(included)} included")
    else:
        demos = sorted(list(demo_dir.glob("*.dem")) + list(demo_dir.glob("*.DEM")))
        print(f"No manifest found, using all {len(demos)} .dem files")

    if not demos:
        print(f"No demos to collect")
        sys.exit(1)

    # Resume: check staging for already-collected demos
    stage_dir = output / "staged"
    stage_dir.mkdir(parents=True, exist_ok=True)
    done_file = stage_dir / "done.json"
    done_demos: set[str] = set()
    if done_file.exists():
        done_demos = set(json.loads(done_file.read_text()).get("demos", []))

    # Build trim_match lookup from manifest
    trim_lookup = {}
    if manifest_path.exists():
        for entry in _load_manifest(manifest_path):
            has_start = entry.get("match_start_text", False)
            has_end = entry.get("match_end_text", False)
            trim_lookup[entry["file"]] = bool(has_start and has_end)

    # Build work list — skip already-done demos
    work = []
    next_index = len(done_demos)
    for demo in demos:
        if demo.name in done_demos:
            continue
        trim = trim_lookup.get(demo.name, False)
        work.append((demo.name, next_index, str(stage_dir), demo_worker, args.asset_root, args.resample_hz, trim))
        next_index += 1

    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    print(f"Demos: {len(demos)} total, {len(done_demos)} cached, {len(work)} to collect")
    print(f"Workers: {n_workers}")

    if work:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_collect_and_save, w): w[0] for w in work}
            for future in as_completed(futures):
                demo_name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  {demo_name}... EXCEPTION: {exc}")
                    continue

                status = result["status"]
                if status == "ok":
                    done_demos.add(demo_name)
                    done_file.write_text(json.dumps({"demos": sorted(done_demos)}))
                    print(f"  {demo_name}... {result['ticks']} ticks")
                elif status == "crash":
                    print(f"  {demo_name}... FAILED (worker crash)")
                elif status == "skipped":
                    print(f"  {demo_name}... SKIPPED ({result['ticks']} ticks)")
                else:
                    print(f"  {demo_name}... ERROR: {result.get('msg', '?')}")

    # Build splits from staged episodes
    staged_metas = sorted(stage_dir.glob("ep*_meta.json"))
    if not staged_metas:
        print("No episodes collected!")
        sys.exit(1)

    episodes_info = [json.loads(p.read_text()) for p in staged_metas]

    rng = np.random.default_rng(args.seed)
    indices = list(range(len(episodes_info)))
    rng.shuffle(indices)

    n_train = int(len(indices) * args.train_ratio)
    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    print(f"\nEpisodes: {len(episodes_info)} total — {len(train_indices)} train, {len(val_indices)} val")

    for split_name, split_indices in [("train", train_indices), ("val", val_indices)]:
        split_dir = output / f"precomputed_{split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        for local_idx, global_idx in enumerate(split_indices):
            src_entry = episodes_info[global_idx]["entry"]
            new_prefix = f"ep{local_idx:04d}"
            new_entry = {"n_samples": src_entry["n_samples"], "obs": {}, "actions": {}}
            for key, fname in src_entry["obs"].items():
                new_fname = f"{new_prefix}_obs_{key}.npy"
                shutil.copy2(stage_dir / fname, split_dir / new_fname)
                new_entry["obs"][key] = new_fname
            for key, fname in src_entry["actions"].items():
                new_fname = f"{new_prefix}_act_{key}.npy"
                shutil.copy2(stage_dir / fname, split_dir / new_fname)
                new_entry["actions"][key] = new_fname
            manifest.append(new_entry)

        (split_dir / "manifest.json").write_text(json.dumps(manifest))
        print(f"  {split_name}: {len(split_indices)} episodes → {split_dir}")

    # Metadata
    metadata = {
        "episodes": len(episodes_info),
        "train": len(train_indices),
        "val": len(val_indices),
        "resample_hz": args.resample_hz,
        "seed": args.seed,
    }
    (output / "collect_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
