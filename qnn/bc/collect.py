#!/usr/bin/env python3
"""Collect BC training data by replaying demos through the demo worker.

Replays .dem files in parallel, reads obs buffer + action labels directly
from the demo worker, shards directly into train/val .npy files. No
intermediate per-episode staging — each worker accumulates rows and
flushes complete shards.

Supports resume via append-only done log.

Usage:
    python -m qnn.bc.collect \
        --demo-dir assets/corpus/dem \
        --output assets/collect/prod \
        --workers 30
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from qnn.wire import (
    OBS_BUFFER_SIZE, ACTION_SIZE, TICK_HEADER_SIZE,
    ACTION_FIELDS, FLAG_DONE,
    unpack_obs_buffer,
)

TICK_TOTAL_SIZE = TICK_HEADER_SIZE + OBS_BUFFER_SIZE + ACTION_SIZE
_WORKER_DEMO_PROC: subprocess.Popen | None = None
_WORKER_DEMO_ARGS: tuple[str, str, int, str] | None = None
_ACTION_RECORD_DTYPE = np.dtype(
    {
        "names": list(ACTION_FIELDS.keys()),
        "formats": [dtype if not shape else (dtype, shape) for _, dtype, shape in ACTION_FIELDS.values()],
        "offsets": [offset for offset, _, _ in ACTION_FIELDS.values()],
        "itemsize": ACTION_SIZE,
    }
)


# ── Worker subprocess management ─────────────────────────────────────

def _start_worker(demo_worker: str, asset_root: str, resample_hz: int, game_dir: str) -> subprocess.Popen:
    env = {**os.environ, "QUAKE_BASEDIR": str(Path(asset_root).resolve())}
    proc = subprocess.Popen(
        [demo_worker, "-game", game_dir],
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


def _shutdown_worker() -> None:
    global _WORKER_DEMO_PROC
    proc = _WORKER_DEMO_PROC
    if proc is None:
        return
    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write(json.dumps({"op": "shutdown"}).encode() + b"\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _WORKER_DEMO_PROC = None


def _init_collect_worker(demo_worker: str, asset_root: str, resample_hz: int, game_dir: str) -> None:
    global _WORKER_DEMO_ARGS
    _WORKER_DEMO_ARGS = (demo_worker, asset_root, int(resample_hz), game_dir)
    _shutdown_worker()
    atexit.register(_shutdown_worker)


def _get_collect_worker() -> subprocess.Popen:
    global _WORKER_DEMO_PROC
    if _WORKER_DEMO_ARGS is None:
        raise RuntimeError("collect worker not initialized")
    if _WORKER_DEMO_PROC is not None and _WORKER_DEMO_PROC.poll() is None:
        return _WORKER_DEMO_PROC
    demo_worker, asset_root, resample_hz, game_dir = _WORKER_DEMO_ARGS
    _WORKER_DEMO_PROC = _start_worker(demo_worker, asset_root, resample_hz, game_dir)
    return _WORKER_DEMO_PROC


# ── Demo playback ────────────────────────────────────────────────────

def _collect_one_demo(proc: subprocess.Popen, demo_name: str,
                      play_start: int = 0, play_end: int = 999999999,
                      ) -> list[dict] | None:
    cmd = json.dumps({
        "op": "collect", "demo_path": demo_name, "seed": 0,
        "play_start": play_start, "play_end": play_end,
    }) + "\n"
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


def _unpack_episode(ticks: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Unpack raw ticks into obs and action arrays. Returns None if empty."""
    n = len(ticks)
    if n == 0:
        return None

    obs_list = [unpack_obs_buffer(t["obs"]) for t in ticks]
    obs_arrays: dict[str, np.ndarray] = {}
    for key in obs_list[0]:
        obs_arrays[key] = np.stack([obs[key] for obs in obs_list], axis=0)

    action_blob = b"".join(t["action"] for t in ticks)
    records = np.frombuffer(action_blob, dtype=_ACTION_RECORD_DTYPE, count=n)
    act_arrays: dict[str, np.ndarray] = {}
    for name, (_, _, shape) in ACTION_FIELDS.items():
        field = np.asarray(records[name])
        act_arrays[name] = field.copy().reshape(n, *shape) if shape else field.copy()

    return obs_arrays, act_arrays


# ── Train/val split assignment ───────────────────────────────────────

def _split_for_demo(demo_name: str, train_ratio: float, seed: int) -> str:
    """Deterministic train/val assignment from demo name hash."""
    h = hashlib.sha256(f"{seed}:{demo_name}".encode()).digest()
    frac = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    return "train" if frac < train_ratio else "val"


# ── Shard writer ─────────────────────────────────────────────────────

class _ShardWriter:
    """Accumulates episodes and flushes shards to disk.

    On resume, loads the existing manifest and continues numbering
    shards from where the previous run left off so we never overwrite.
    """

    def __init__(self, split_dir: Path, shard_rows: int):
        self.split_dir = split_dir
        self.shard_rows = max(1, shard_rows)
        self._obs_bufs: dict[str, list[np.ndarray]] = {}
        self._act_bufs: dict[str, list[np.ndarray]] = {}
        self._episode_lengths: list[int] = []
        self._current_rows = 0
        split_dir.mkdir(parents=True, exist_ok=True)

        # Resume: load existing manifest and continue from last shard
        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists():
            try:
                prev = json.loads(manifest_path.read_text())
                self.shards: list[dict] = prev.get("shards", [])
                self.shard_idx = len(self.shards)
            except (json.JSONDecodeError, KeyError):
                self.shards = []
                self.shard_idx = 0
        else:
            self.shards = []
            self.shard_idx = 0

    def add_episode(self, obs: dict[str, np.ndarray], actions: dict[str, np.ndarray], n_samples: int) -> None:
        for key, arr in obs.items():
            self._obs_bufs.setdefault(key, []).append(arr)
        for key, arr in actions.items():
            self._act_bufs.setdefault(key, []).append(arr)
        self._episode_lengths.append(n_samples)
        self._current_rows += n_samples
        if self._current_rows >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self._episode_lengths:
            return
        prefix = f"shard{self.shard_idx:06d}"
        obs_files: dict[str, str] = {}
        for key, arrays in self._obs_bufs.items():
            fname = f"{prefix}_obs_{key}.npy"
            np.save(self.split_dir / fname, np.concatenate(arrays, axis=0))
            obs_files[key] = fname
        act_files: dict[str, str] = {}
        for key, arrays in self._act_bufs.items():
            fname = f"{prefix}_act_{key}.npy"
            np.save(self.split_dir / fname, np.concatenate(arrays, axis=0))
            act_files[key] = fname
        self.shards.append({
            "rows": self._current_rows,
            "episode_lengths": self._episode_lengths,
            "obs": obs_files,
            "actions": act_files,
        })
        self.shard_idx += 1
        self._obs_bufs.clear()
        self._act_bufs.clear()
        self._episode_lengths = []
        self._current_rows = 0
        # Write manifest after every shard so a killed process doesn't
        # lose all progress.  On resume we reload this.
        self._write_manifest()

    def _write_manifest(self) -> None:
        total_episodes = sum(len(s["episode_lengths"]) for s in self.shards)
        manifest = {
            "format": "sharded_v1",
            "episodes": total_episodes,
            "shard_rows": self.shard_rows,
            "shards": self.shards,
        }
        (self.split_dir / "manifest.json").write_text(json.dumps(manifest))

    def write_manifest(self) -> None:
        self.flush()


# ── Per-worker collect function ──────────────────────────────────────

_DEMO_TIMEOUT = 90  # seconds — kill worker if a single demo takes longer.
# The healthy path for even the longest 4on4 QWDs is well under this;
# values above this indicate an engine deadlock that won't recover by
# waiting longer.

# Error messages with these prefixes mean the failure is intrinsic to the
# demo / environment (won't be fixed by retry). They skip the retry loop
# and count as errors immediately.
_PERMANENT_ERROR_PREFIXES = (
    "demo_preamble_incompatible",
    "Missing BSP",
)


def _is_permanent_error(msg: str) -> bool:
    return any(msg.startswith(p) for p in _PERMANENT_ERROR_PREFIXES)


def _collect_demo(args: tuple) -> dict:
    """Collect one demo, return unpacked arrays. Runs in worker process."""
    import signal

    demo_name, play_start, play_end = args

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"demo timed out after {_DEMO_TIMEOUT}s")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(_DEMO_TIMEOUT)
    try:
        proc = _get_collect_worker()
        ticks = _collect_one_demo(proc, demo_name,
                                  play_start=play_start, play_end=play_end)
    except TimeoutError:
        _shutdown_worker()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"demo": demo_name, "status": "error", "msg": "timeout"}
    except Exception as exc:
        _shutdown_worker()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"demo": demo_name, "status": "error", "msg": str(exc)[:200]}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if ticks is None:
        _shutdown_worker()
        return {"demo": demo_name, "status": "crash"}

    if len(ticks) < 10:
        return {"demo": demo_name, "status": "skipped", "ticks": len(ticks)}

    result = _unpack_episode(ticks)
    if result is None:
        return {"demo": demo_name, "status": "skipped", "ticks": 0}

    obs, actions = result
    return {
        "demo": demo_name,
        "status": "ok",
        "ticks": len(ticks),
        "obs": obs,
        "actions": actions,
    }


# ── Done tracking (append-only) ─────────────────────────────────────

def _load_done_set(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text().splitlines() if line.strip()}


def _append_done(done_path: Path, demo_name: str) -> None:
    with open(done_path, "a") as f:
        f.write(demo_name + "\n")


# ── Manifest loading ─────────────────────────────────────────────────

def _load_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    for line in manifest_path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _available_maps(asset_root: Path) -> set[str]:
    """Return the set of map names (lowercase, no extension) whose BSPs
    are reachable by the demo worker — either as loose .bsp files under
    */maps/ or packed inside */pak*.pak."""
    import struct
    maps: set[str] = set()
    if not asset_root.exists():
        return maps
    for bsp in asset_root.rglob("maps/*.bsp"):
        maps.add(bsp.stem.lower())
    for pak in asset_root.rglob("pak*.pak"):
        try:
            with open(pak, "rb") as f:
                head = f.read(12)
                if head[:4] != b"PACK":
                    continue
                dir_ofs, dir_len = struct.unpack("<II", head[4:12])
                f.seek(dir_ofs)
                directory = f.read(dir_len)
            for i in range(dir_len // 64):
                name = directory[i * 64 : i * 64 + 56].rstrip(b"\x00").decode("latin-1", "ignore")
                if name.startswith("maps/") and name.endswith(".bsp"):
                    maps.add(name[5:-4].lower())
        except (OSError, struct.error):
            continue
    return maps


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BC training data from demo files")
    parser.add_argument("--demo-dir", required=True, help="Directory containing .dem files")
    parser.add_argument("--output", required=True, help="Output directory for sharded .npy caches")
    parser.add_argument("--manifest", default="", help="Path to manifest.ndjson (default: auto-detected)")
    parser.add_argument("--demo-worker", default="assets/bin/nq_demo_worker")
    parser.add_argument("--asset-root", default="assets")
    parser.add_argument("--resample-hz", type=int, default=20, help="Emission rate (native rate auto-detected)")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=30, help="Parallel workers")
    parser.add_argument("--shard-rows", type=int, default=262144, help="Rows per shard")
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    output = Path(args.output)
    demo_worker = str(Path(args.demo_worker).resolve())

    try:
        game_dir = str(demo_dir.resolve().relative_to(Path(args.asset_root).resolve()))
    except ValueError:
        game_dir = str(demo_dir)

    # Load manifest if available
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        corpus_manifest = demo_dir.parent / f"{demo_dir.name}_manifest.ndjson"
        legacy_manifest = demo_dir / "manifest.ndjson"
        manifest_path = corpus_manifest if corpus_manifest.exists() else legacy_manifest

    available_maps = _available_maps(Path(args.asset_root))

    bounds_lookup: dict[str, tuple[int, int]] = {}
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        included = []
        excluded = []
        missing_map = 0
        for e in manifest:
            if e.get("bc_exclude", False):
                excluded.append(e["file"])
                continue
            mp = (e.get("map") or "").lower()
            if mp and mp not in available_maps:
                excluded.append(e["file"])
                missing_map += 1
                continue
            included.append(e["file"])
        demos = sorted([demo_dir / name for name in included if (demo_dir / name).exists()])
        for entry in manifest:
            bounds_lookup[entry["file"]] = (
                entry.get("play_start", 0),
                entry.get("play_end", 999999999),
            )
        tail = f" ({missing_map} skipped for missing BSP)" if missing_map else ""
        print(f"Manifest: {len(manifest)} entries, {len(excluded)} excluded{tail}, {len(included)} included")
    else:
        demos = sorted(
            list(demo_dir.glob("*.dem")) + list(demo_dir.glob("*.DEM"))
            + list(demo_dir.glob("*.qwd")) + list(demo_dir.glob("*.mvd"))
        )
        print(f"No manifest found, using all {len(demos)} demo files")

    if not demos:
        print("No demos to collect")
        sys.exit(1)

    # Resume via append-only done log
    done_path = output / "done.log"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_demos = _load_done_set(done_path)

    # Build work list with play boundaries from manifest
    work = []
    for demo in demos:
        if demo.name in done_demos:
            continue
        ps, pe = bounds_lookup.get(demo.name, (0, 999999999))
        work.append((demo.name, ps, pe))

    n_workers = max(1, args.workers)
    print(f"Demos: {len(demos)} total, {len(done_demos)} cached, {len(work)} to collect")
    print(f"Workers: {n_workers}")

    if not work:
        train_manifest = output / "precomputed_train" / "manifest.json"
        val_manifest = output / "precomputed_val" / "manifest.json"
        if train_manifest.exists() and val_manifest.exists():
            print("Done (no new data).")
            return
        print("No new work but precomputed splits missing — rebuild needed.")
        # Fall through to rebuild from done log + raw re-collect
        # (This path means done.log exists but shards were deleted — rare.)
        print("Cannot rebuild without re-collecting. Clear done.log to force full re-collect.")
        sys.exit(1)

    # Shard writers for train/val — write directly, no staging
    train_writer = _ShardWriter(output / "precomputed_train", args.shard_rows)
    val_writer = _ShardWriter(output / "precomputed_val", args.shard_rows)

    import time as _time
    collected = 0
    skipped = 0
    errors = 0
    total_ticks = 0
    t_start = _time.monotonic()
    total_work = len(work)
    _PROGRESS_INTERVAL = max(1, total_work // 20)  # ~5% increments

    # Crash/error is usually transient (worker startup contention,
    # mid-demo OOM, stdin race) and clears on a fresh worker. Retry
    # within the same run; demos that fail every attempt stay out of
    # done.log so the next collect picks them up.
    RETRY_MAX = 2

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_collect_worker,
        initargs=(demo_worker, args.asset_root, args.resample_hz, game_dir),
    ) as pool:
        pending = {pool.submit(_collect_demo, w): (w, 0) for w in work}
        while pending:
            for future in as_completed(list(pending)):
                w, attempt = pending.pop(future)
                demo_name = w[0]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  {demo_name}... EXCEPTION: {exc}")
                    errors += 1
                    continue

                status = result["status"]
                if status == "ok":
                    split = _split_for_demo(demo_name, args.train_ratio, args.seed)
                    writer = train_writer if split == "train" else val_writer
                    ticks = result["ticks"]
                    writer.add_episode(result["obs"], result["actions"], ticks)
                    _append_done(done_path, demo_name)
                    collected += 1
                    total_ticks += ticks
                elif status == "skipped":
                    _append_done(done_path, demo_name)
                    skipped += 1
                elif status in ("crash", "error"):
                    msg = result.get("msg", "worker crash") if status == "error" else "worker crash"
                    # Permanent errors (corrupt QWD preamble, missing BSP)
                    # won't be fixed by retrying — fail them now.
                    permanent = status == "error" and _is_permanent_error(msg)
                    if not permanent and attempt < RETRY_MAX:
                        print(f"  {demo_name}... {status} ({msg}), retry {attempt+1}/{RETRY_MAX}")
                        retry_future = pool.submit(_collect_demo, w)
                        pending[retry_future] = (w, attempt + 1)
                        continue  # don't count as final yet
                    suffix = "permanent" if permanent else f"after {attempt+1} attempts"
                    tag = "FAILED (worker crash)" if status == "crash" else f"ERROR: {msg}"
                    print(f"  {demo_name}... {tag} — {suffix}")
                    errors += 1
                else:
                    print(f"  {demo_name}... unexpected status={status!r}")
                    errors += 1

                # Progress report
                done_count = collected + skipped + errors
                if done_count % _PROGRESS_INTERVAL == 0 or done_count == total_work:
                    elapsed = _time.monotonic() - t_start
                    rate = done_count / max(elapsed, 0.01)
                    ticks_rate = total_ticks / max(elapsed, 0.01)
                    remaining = (total_work - done_count) / max(rate, 0.01)
                    mins, secs = divmod(int(remaining), 60)
                    hrs, mins = divmod(mins, 60)
                    eta = f"{hrs}h{mins:02d}m" if hrs else f"{mins}m{secs:02d}s"
                    print(
                        f"  [{done_count}/{total_work}] "
                        f"{rate:.1f} demos/s, {ticks_rate/1000:.0f}K ticks/s, "
                        f"{total_ticks/1e6:.1f}M ticks total, "
                        f"ETA {eta}"
                    )

    # Flush remaining data and write manifests
    train_writer.write_manifest()
    val_writer.write_manifest()

    train_eps = sum(len(s["episode_lengths"]) for s in train_writer.shards)
    val_eps = sum(len(s["episode_lengths"]) for s in val_writer.shards)

    metadata = {
        "episodes": train_eps + val_eps,
        "train": train_eps,
        "val": val_eps,
        "collected": collected,
        "skipped": skipped,
        "errors": errors,
        "resample_hz": args.resample_hz,
        "seed": args.seed,
    }
    (output / "collect_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nDone: {collected} collected, {skipped} skipped, {errors} errors")
    print(f"  train: {train_eps} episodes in {len(train_writer.shards)} shards")
    print(f"  val: {val_eps} episodes in {len(val_writer.shards)} shards")


if __name__ == "__main__":
    main()
