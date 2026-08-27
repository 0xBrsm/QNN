"""Matched-pair collect: one native-rate replay → two corpora.

Drives ``qw_demo_worker`` in matched-emit mode (``matched_emit:1``,
``tick_hz:0``).  Each worker pass emits two interleaved framed streams on
one pipe (see qnn.wire / qnn_collect_main.c):

  - a slim **MLOB** record every native frame — the move-labeler input
    subset (view-frame vel and self_movement_id) plus the
    usercmd-TRUTH action (move press byte, look, op_input) and the native
    frame index;  and
  - the full **QOBS** at each 20 Hz demo-time boundary (the model corpus),
    each frame tagged with the native index it was sampled at.

This module demuxes the two streams and writes TWO sharded corpora:

  <output>/slim/   — the native-rate labeler corpus (MLOB), one row per
                     native frame.
  <output>/qobs/   — the 20 Hz model corpus, identical framing to a normal
                     BC collect PLUS a per-frame ``native_index`` obs array
                     so a labeler prediction (indexed by native frame) can
                     be resampled to 20 Hz by exact lookup and rewrite the
                     QOBS move.

Both corpora share the standard ``_ShardWriter`` layout (precomputed_train
/ precomputed_val), so downstream loaders read them like any BC cache.

Usage:
    PYTHONPATH=src python -m qnn.labeler.matched_collect \\
        --demo-dir   artifacts/corpus/qwd \\
        --manifest   artifacts/corpus/qwd_manifest.ndjson \\
        --output     artifacts/collect/qwd_matched \\
        --workers    1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from qnn.bc import collect as bc
from qnn.wire import parse_mlob_frame  # noqa: F401  (documents the MLOB schema)


# Slim labeler field subset (on-disk names match qnn.labeler.collect).
_SLIM_OBS_FIELDS = ("vel", "self_movement_id")
_SLIM_ACT_FIELDS = ("move", "look", "op_input")


# ── MLOB → slim episode ──────────────────────────────────────────────

def _unpack_mlob_episode(mlob_ticks: list[dict]) -> tuple[dict, dict]:
    """Stack the per-native-frame MLOB records into (obs, act) arrays.

    obs:  vel (i16, T×3), self_movement_id (u8, T),
          native_index (u32, T) — the slim corpus key.
    act:  move (u8, T), look (f16, T×3), op_input (u8, T).
    """
    n = len(mlob_ticks)
    obs = {
        "vel": np.stack([t["vel"] for t in mlob_ticks], axis=0).astype(np.int16),
        "self_movement_id": np.array([t["self_movement_id"] for t in mlob_ticks], dtype=np.uint8),
        "native_index":     np.array([t["native_index"]     for t in mlob_ticks], dtype=np.uint32),
    }
    act = {
        "move":     np.array([t["move"] for t in mlob_ticks], dtype=np.uint8),
        "look":     np.stack([t["look"] for t in mlob_ticks], axis=0).astype(np.float16),
        "op_input": np.array([t["op_input"] for t in mlob_ticks], dtype=np.uint8),
    }
    return obs, act


# ── QOBS matched episode (full obs + per-frame native_index) ─────────

def _unpack_qobs_matched(qobs_ticks: list[dict]) -> list[tuple[dict, dict]]:
    """Unpack the 20 Hz QOBS frames into (obs, act) sub-episodes and attach
    a per-frame ``native_index`` obs array.  Reuses the canonical BC
    unpacker for the obs/action arrays (no segments.drop here — a single
    full run), then slots native_index in aligned to the frame axis. """
    if not qobs_ticks:
        return []
    episodes = bc._unpack_episode(
        qobs_ticks, combat_only=False, labels=None,
        drop_label_names=(), total_frames=0, sight_only=False,
        target_probs_cache=False,
    )
    native_index = np.array([t["native_index"] for t in qobs_ticks], dtype=np.uint32)
    # _unpack_episode with no drop labels returns a single full-length
    # run, so native_index aligns 1:1 with the (single) episode's frames.
    out: list[tuple[dict, dict]] = []
    offset = 0
    for obs, act in episodes:
        # Frame count = leading dim of any per-frame self field.
        n = int(obs["health"].shape[0]) if "health" in obs else \
            int(next(iter(act.values())).shape[0])
        obs = dict(obs)
        obs["native_index"] = native_index[offset:offset + n]
        offset += n
        out.append((obs, act))
    return out


# ── Per-demo pool worker ──────────────────────────────────────────────

def _collect_demo_matched(args: tuple) -> dict:
    """Pool entry: matched collect for one demo.  Returns a status dict
    with BOTH corpora's episodes under ``slim_episodes`` / ``qobs_episodes``."""
    (demo_name, force_mvd_emit, play_start, play_end) = args

    proc = None
    try:
        proc = bc._get_collect_worker()
        pair = bc._collect_one_demo_matched(
            proc, demo_name, play_start=play_start, play_end=play_end,
            force_mvd_emit=force_mvd_emit)
    except Exception as exc:
        err_tail = bc._worker_stderr_tail(proc)
        bc._shutdown_worker()
        msg = f"{exc}\n{err_tail}" if err_tail else str(exc)
        return {"demo": demo_name, "status": "error", "msg": msg[-4000:]}

    if pair is None:
        err_tail = bc._worker_stderr_tail(proc)
        rc = proc.poll() if proc is not None else None
        bc._shutdown_worker()
        if rc == bc._WATCHDOG_EXIT_CODE:
            return {"demo": demo_name, "status": "error",
                    "msg": f"watchdog stall\n{err_tail}"[-4000:]}
        return {"demo": demo_name, "status": "crash",
                "msg": err_tail or "worker crash"}

    bc._shutdown_worker()
    mlob_ticks, qobs_ticks = pair
    if len(mlob_ticks) < 10:
        return {"demo": demo_name, "status": "skipped", "ticks": len(mlob_ticks)}

    slim_obs, slim_act = _unpack_mlob_episode(mlob_ticks)
    slim_rows = int(slim_obs["native_index"].shape[0])
    qobs_eps = _unpack_qobs_matched(qobs_ticks)
    qobs_sized = [(o, a, int(o["native_index"].shape[0])) for o, a in qobs_eps]
    qobs_sized = [(o, a, r) for o, a, r in qobs_sized if r > 0]

    return {
        "demo": demo_name, "status": "ok",
        "ticks": len(mlob_ticks),
        "slim_episodes": [{"obs": slim_obs, "actions": slim_act, "rows": slim_rows}],
        "qobs_episodes": [{"obs": o, "actions": a, "rows": r}
                          for o, a, r in qobs_sized],
    }


# ── Orchestration (two corpora) ───────────────────────────────────────

def run_matched(
    *, output: Path, demo_dir: Path, manifest_path: Path, asset_root: Path,
    demo_worker: str, game_dir: str, workers: int, shard_rows: int,
    train_ratio: float, seed: int, force_mvd_emit: bool,
    play_start: int, play_end: int,
) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    slim_dir = output / "slim"
    qobs_dir = output / "qobs"
    slim_dir.mkdir(parents=True, exist_ok=True)
    qobs_dir.mkdir(parents=True, exist_ok=True)

    manifest = bc._load_manifest(manifest_path)
    available_maps = bc._available_maps(asset_root)
    demo_idx_map: dict[str, int] = {}
    selected: list[str] = []
    for pos, e in enumerate(manifest):
        demo_idx_map[e["file"]] = pos
        mp = (e.get("map") or "").lower()
        if mp and mp not in available_maps:
            continue
        if (demo_dir / e["file"]).exists():
            selected.append(e["file"])
    demos = sorted(selected)
    if not demos:
        sys.exit("No demos to collect")

    done_path = output / "done.log"
    done = bc._load_done_set(done_path)
    work = [(d, force_mvd_emit, play_start, play_end)
            for d in demos if d not in done]
    print(f"Matched collect: {len(demos)} demos, {len(done)} cached, {len(work)} to do")
    print(f"  slim → {slim_dir}\n  qobs → {qobs_dir}")
    if not work:
        print("Done (no new data).")
        return

    splits = [bc._split_for_demo(w[0], train_ratio, seed) for w in work]
    slim_train = bc._ShardWriter(slim_dir / "precomputed_train", shard_rows)
    slim_val   = bc._ShardWriter(slim_dir / "precomputed_val", shard_rows)
    qobs_train = bc._ShardWriter(qobs_dir / "precomputed_train", shard_rows)
    qobs_val   = bc._ShardWriter(qobs_dir / "precomputed_val", shard_rows)

    collected = skipped = errors = 0

    def consume(idx: int, payload: object) -> None:
        nonlocal collected, skipped, errors
        demo_name = work[idx][0]
        if isinstance(payload, BaseException):
            print(f"  {demo_name}... EXCEPTION: {payload}"); errors += 1; return
        res = payload
        st = res["status"]
        if st == "ok":
            di = demo_idx_map[demo_name]
            sw = slim_train if splits[idx] == "train" else slim_val
            qw = qobs_train if splits[idx] == "train" else qobs_val
            for ei, ep in enumerate(res["slim_episodes"]):
                sw.add_episode(ep["obs"], ep["actions"], int(ep["rows"]), di, ei)
            for ei, ep in enumerate(res["qobs_episodes"]):
                qw.add_episode(ep["obs"], ep["actions"], int(ep["rows"]), di, ei)
            bc._append_done(done_path, demo_name); collected += 1
        elif st == "skipped":
            bc._append_done(done_path, demo_name); skipped += 1
        else:
            print(f"  {demo_name}... {st.upper()}: {res.get('msg','')[:300]}"); errors += 1

    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=bc._init_collect_worker,
        initargs=(demo_worker, str(asset_root), 0, game_dir,
                  (), (), (), ()),
    ) as pool:
        futures = {pool.submit(_collect_demo_matched, w): i for i, w in enumerate(work)}
        pending: dict[int, object] = {}
        nxt = 0
        for fut in as_completed(futures):
            i = futures.pop(fut)
            try:
                pending[i] = fut.result()
            except Exception as exc:
                pending[i] = exc
            while nxt in pending:
                consume(nxt, pending.pop(nxt)); nxt += 1

    for w in (slim_train, slim_val, qobs_train, qobs_val):
        w.write_manifest()

    import json
    meta = {
        "format": "matched_v1", "force_mvd_emit": bool(force_mvd_emit),
        "collected": collected, "skipped": skipped, "errors": errors,
        "slim_fields": {"obs": list(_SLIM_OBS_FIELDS) + ["native_index"],
                        "act": list(_SLIM_ACT_FIELDS)},
        "qobs_native_index": True,
    }
    (output / "collect_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone: {collected} collected, {skipped} skipped, {errors} errors")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--asset-root", type=Path, default=Path("assets"))
    ap.add_argument("--demo-worker", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard-rows", type=int, default=200_000)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--force-mvd-emit", action="store_true")
    ap.add_argument("--play-start", type=int, default=0)
    ap.add_argument("--play-end", type=int, default=999999999)
    args = ap.parse_args()

    asset_root = args.asset_root.resolve()
    demo_dir = args.demo_dir.resolve()
    manifest_path = (args.manifest if args.manifest
                     else demo_dir.parent / f"{demo_dir.name}_manifest.ndjson")
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")
    demo_worker = args.demo_worker or Path(bc._default_demo_worker("qwd"))
    if not Path(demo_worker).is_absolute():
        demo_worker = Path.cwd() / demo_worker
    if not Path(demo_worker).exists():
        sys.exit(f"demo worker not found: {demo_worker}")
    game_dir = bc._game_dir_for_demo_dir(demo_dir, asset_root)

    run_matched(
        output=args.output, demo_dir=demo_dir, manifest_path=manifest_path,
        asset_root=asset_root, demo_worker=str(demo_worker), game_dir=game_dir,
        workers=args.workers, shard_rows=args.shard_rows,
        train_ratio=args.train_ratio, seed=args.seed,
        force_mvd_emit=args.force_mvd_emit,
        play_start=args.play_start, play_end=args.play_end,
    )


if __name__ == "__main__":
    main()
