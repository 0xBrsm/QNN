"""Backfill compact 2-D tangent look labels beside the cached 3-vector form.

Writes ``shard_NNNNNN_act_look_tan.npy`` (float16, ``(T_shard, 2)``) next to
each shard's ``act_look`` array: the tangent-space turn vector
``z = theta * (yz / |yz|)`` with the magnitude recovered via
``theta = atan2(|yz|, x)``.

Why this exists (2026-07-06 root cause, research/look-head.md): the collect
packer stores the look unit vector as float16. Near 1.0 the fp16 grid is
4.88e-4, and the small-turn signal ``1 - cos(theta) ~ theta^2/2`` is
quadratically small — every turn below ~1.27 deg rounds x to exactly 1.0 and
the ``arccos(x)`` logmap emits a manufactured hold; surviving magnitudes comb
onto ``sqrt(n) * 1.7905 deg``. The transverse components carry the same
rotation *linearly* (``sin(theta) ~ theta``) near zero where fp16 is dense, so
the erased band is fully recoverable from yz: ~0.0003 deg error at fovea
scale, an order of magnitude below the demo angle16 source floor (0.0055 deg).
``|z| = theta`` is linear over the full [0, pi] with no hemisphere fold, so
float16 tangent storage is numerically sufficient everywhere (0.05% relative).

MANIFEST IS NOT TOUCHED. ``manifest.json`` is a collection-fingerprint
component; registering the new arrays would invalidate every run pinned to
the current fingerprint (including daemon-queued jobs, which verify at load).
The sidecars sit unregistered until a deliberate ``--register`` pass bumps
the manifest + fingerprint as its own versioned step.

Usage (devcontainer, CPU):
  PYTHONPATH=src python -m qnn.bc.cache_look_tan --collect artifacts/collect/qwd
  PYTHONPATH=src python -m qnn.bc.cache_look_tan --collect artifacts/collect/qwd --register
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def look_to_tangent(look: np.ndarray) -> np.ndarray:
    """(N, 3) unit look vectors (any float dtype) -> (N, 2) float16 tangent.

    Magnitude from ``atan2(|yz|, x)`` — well-conditioned over the whole range,
    immune to the near-1 cosine trap, tolerant of slightly non-unit inputs.
    Exact zeros (yz == 0) map to exact zero tangents.
    """
    x = look[:, 0].astype(np.float64)
    yz = look[:, 1:3].astype(np.float64)
    n = np.linalg.norm(yz, axis=1)
    theta = np.arctan2(n, x)                       # [0, pi]
    scale = np.zeros_like(theta)
    nz = n > 0.0
    scale[nz] = theta[nz] / n[nz]
    return (yz * scale[:, None]).astype(np.float16)


def _convert_one_shard(args: tuple[str, int, str]) -> tuple[int, str, int, float]:
    """Worker: convert one shard's act_look. Returns (idx, fname, rows, max_err_deg).

    max_err_deg is the round-trip angle between expmap(z) and the original
    stored vector — the conversion's own fidelity, not the fp16 cast's.
    """
    split_dir_str, shard_idx, look_fname = args
    split_dir = Path(split_dir_str)
    out_fname = look_fname.replace("_act_look.npy", "_act_look_tan.npy")
    out_path = split_dir / out_fname

    look = np.load(split_dir / look_fname)
    z = look_to_tangent(look)
    np.save(out_path, z)

    # Round-trip check on a stride sample: expmap(z) vs stored vector.
    s = slice(None, None, max(1, len(z) // 20000))
    z64 = z[s].astype(np.float64)
    theta = np.linalg.norm(z64, axis=1)
    d = np.zeros_like(z64)
    nz = theta > 0
    d[nz] = z64[nz] / theta[nz, None]
    rec = np.column_stack([np.cos(theta), np.sin(theta)[:, None] * d])
    orig = look[s].astype(np.float64)
    on = np.linalg.norm(orig, axis=1)
    ok = on > 0  # stored zero vectors (episode-boundary padding) → z=0; skip
    orig[ok] /= on[ok, None]
    cos = np.clip((rec[ok] * orig[ok]).sum(axis=1), -1.0, 1.0)
    max_err = float(np.degrees(np.arccos(cos)).max()) if ok.any() else 0.0
    return shard_idx, out_fname, int(z.shape[0]), max_err


def _convert_split(split_dir: Path, n_workers: int, register: bool) -> None:
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = manifest["shards"]

    pending = []
    for i, s in enumerate(shards):
        look_fname = s.get("actions", {}).get("look")
        if not look_fname:
            continue
        out = split_dir / look_fname.replace("_act_look.npy", "_act_look_tan.npy")
        if not out.exists():
            pending.append((str(split_dir), i, look_fname))
    print(f"  {split_dir.name}: {len(shards)} shards, {len(pending)} to convert")

    max_err = 0.0
    total = 0
    if pending:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_convert_one_shard, a) for a in pending]
            done = 0
            for fut in as_completed(futures):
                _, _, rows, err = fut.result()
                total += rows
                max_err = max(max_err, err)
                done += 1
                if done % 20 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} shards ({total:,} rows, "
                          f"worst round-trip {max_err:.4f} deg)")

    if register:
        changed = False
        for s in shards:
            look_fname = s.get("actions", {}).get("look")
            if look_fname and "look_tan" not in s["actions"]:
                s["actions"]["look_tan"] = look_fname.replace(
                    "_act_look.npy", "_act_look_tan.npy")
                changed = True
        if changed:
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"    manifest UPDATED — the collection fingerprint is now stale; "
                  f"re-fingerprint before pinning new runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collect", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--register", action="store_true",
                    help="also register look_tan in manifest.json (INVALIDATES "
                         "the pinned collection fingerprint — run only when no "
                         "queued/pinned job depends on it)")
    args = ap.parse_args()
    collect = Path(args.collect)
    for split in ("precomputed_train", "precomputed_val"):
        d = collect / split
        if d.exists():
            _convert_split(d, args.workers, args.register)


if __name__ == "__main__":
    main()
