"""Build mesh-based probe-grid spatial fields for a collect cache.

Probe placement is the map's Detour navmesh: one world-anchored panoramic
depth atlas carved at every walkable poly center (``qnn.bc.probe_atlas`` →
``navatlas_<map>.json``), against the static world only.  This is the
load-time static table the design calls for — 0 wire bytes/frame, probes
anchored to reachable space, and the panorama payload carved fresh by the
engine rather than harvested from demo frames.

For each cache row this writes two extra obs fields per shard:

- ``obs_probe_atlas``   (rows, K, 11, 36) u8 — the K nearest probes'
  panoramas, rolled into the row's view frame, nibble-packed.
- ``obs_probe_offsets`` (rows, K, 5) f16 — per-probe relative pose in the
  row's view frame (see qnn.schema.PROBE_OFFSET_DIM).

Row poses come from the QNN_POSE_DIAG sidecar written during the collect
(one line per emitted QOBS record); the collect's segment drops are
reproduced with the collect module's own ``_label_keep_mask`` so kept pose
rows align 1:1 with cache rows per demo (verified per demo, hard error on
mismatch).  Poses are used only to pick each row's K-nearest probes and
encode their relative pose — the probes themselves are map-static, so there
is no temporal-leakage concern and no exclusion window.

Usage:
    PYTHONPATH=src python -m qnn.bc.probe_grid \
        --cache artifacts/collect/tmp/qwd_probe \
        --pose-dir artifacts/collect/tmp/qwd_probe_pose \
        --demo-manifest artifacts/corpus/qwd_probe_manifest.ndjson \
        --navatlas-dir artifacts/collect/tmp/qwd_probe/navatlas \
        --k 4
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from qnn.bc.collect import _label_keep_mask
from qnn.bc.probe_atlas import decode_dump
from qnn.engine_norm import ATLAS_YAWS
from qnn.schema import PROBE_OFFSET_DIM, SPATIAL_TOKEN_COUNT

YAW_STEP_DEG = 360.0 / ATLAS_YAWS
DIST_SCALE = 512.0


def load_pose(pose_dir: Path) -> dict[str, np.ndarray]:
    """Per-demo (n, 4) [x, y, z, view_yaw] in emitted-record order."""
    rows: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    for f in glob.glob(str(pose_dir / "pose.*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[Path(r["demo"]).name].append(
                (int(r["row"]), [*r["origin"], r["view_yaw"]])
            )
    out: dict[str, np.ndarray] = {}
    for demo, items in rows.items():
        items.sort(key=lambda t: t[0])
        if [i for i, _ in items] != list(range(len(items))):
            raise ValueError(f"{demo}: pose rows not contiguous")
        out[demo] = np.asarray([v for _, v in items], dtype=np.float64)
    return out


def kept_pose(
    pose: np.ndarray, entry: dict, drop_labels: tuple[str, ...]
) -> np.ndarray:
    labels = entry["labels"]
    if isinstance(labels, str):
        labels = json.loads(labels.replace("'", '"'))
    keep = _label_keep_mask(
        labels, drop_labels, int(entry["total_frames"]), len(pose)
    )
    return pose[keep]


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


def load_navatlas(navatlas_dir: Path, map_name: str) -> "MapProbes":
    """navatlas_<map>.json → MapProbes (poly-center probes for one map)."""
    path = navatlas_dir / f"navatlas_{map_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no probe-atlas dump for map {map_name!r} at {path}; "
            f"generate it with qnn.bc.probe_atlas"
        )
    with open(path) as fh:
        result = json.load(fh)
    positions, panoramas = decode_dump(result)
    return MapProbes(positions, panoramas)


class MapProbes:
    """One map's mesh-based probe table + per-row K-nearest assignment.

    ``positions`` are probe viewpoints (poly center + carve z-offset) and
    ``panoramas`` are world-anchored (yaw cell 0 = world yaw 0), carved at
    yaw 0 — so every probe's yaw residual is exactly 0.
    """

    def __init__(self, positions: np.ndarray, panoramas: np.ndarray):
        from scipy.spatial import cKDTree

        if len(positions) != len(panoramas):
            raise ValueError("positions/panoramas length mismatch")
        self.positions = np.asarray(positions, dtype=np.float64)
        self.panoramas = np.asarray(panoramas, dtype=np.uint8)  # (P, 11, 72)
        self.tree = cKDTree(self.positions)

    def assign(
        self, pos: np.ndarray, view_yaw: np.ndarray, k: int,
        *, workers: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rows → (rows, K, 11, 24) view-frame panoramas + (rows, K, 5) offsets.

        ``workers`` is the scipy KD-tree query fan-out: -1 (all cores) for the
        offline batch build, 1 for the per-tick single-row closed-loop call,
        where spinning up a thread pool per query costs more than it saves."""
        query_k = min(k, len(self.positions))
        if query_k < k:
            raise ValueError(
                f"probe table has {len(self.positions)} probes < k={k}"
            )
        dists, idxs = self.tree.query(pos, k=query_k, workers=workers)
        if query_k == 1:
            idxs = idxs[:, None]
        chosen = idxs  # k nearest — probes are map-static, no exclusion

        shifts = np.round(view_yaw / YAW_STEP_DEG).astype(np.int64) % ATLAS_YAWS
        pano = self.panoramas[chosen]  # (n, k, 11, 72) world-anchored
        out_pano = np.empty_like(pano)
        for s in np.unique(shifts):
            m = shifts == s
            out_pano[m] = np.roll(pano[m], -int(s), axis=-1)

        delta = self.positions[chosen] - pos[:, None, :]  # (n, k, 3)
        yaw_rad = np.radians(view_yaw)[:, None]
        cos, sin = np.cos(yaw_rad), np.sin(yaw_rad)
        fwd = cos * delta[..., 0] + sin * delta[..., 1]
        left = -sin * delta[..., 0] + cos * delta[..., 1]
        dist = np.linalg.norm(delta, axis=-1)
        # Probe residual is 0 (carved at yaw 0), so eps = -row residual,
        # broadcast across the k probes.
        row_resid = view_yaw - YAW_STEP_DEG * np.round(view_yaw / YAW_STEP_DEG)
        eps = np.zeros_like(fwd) - row_resid[:, None]
        offsets = np.stack(
            [fwd / DIST_SCALE, left / DIST_SCALE, delta[..., 2] / DIST_SCALE,
             eps / YAW_STEP_DEG, dist / DIST_SCALE],
            axis=-1,
        ).astype(np.float16)
        assert offsets.shape[-1] == PROBE_OFFSET_DIM
        return out_pano, offsets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--demo-manifest", type=Path, required=True)
    parser.add_argument("--navatlas-dir", type=Path, required=True,
                        help="dir of navatlas_<map>.json from qnn.bc.probe_atlas")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--export-tables", type=Path, default=None,
        help="also save the per-map probe tables (positions + world-"
             "anchored panoramas) to this .npz — the load-time table a "
             "closed-loop probe path consumes (qnn.eval.probe_table)",
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="with --export-tables: skip writing shard fields/manifests "
             "(read-only on the cache)",
    )
    args = parser.parse_args(argv)
    if args.tables_only and args.export_tables is None:
        parser.error("--tables-only requires --export-tables")

    entries: list[dict] = []
    with open(args.demo_manifest) as fh:
        for line in fh:
            entries.append(json.loads(line))
    by_file = {e["file"]: e for e in entries}
    demo_by_pos = [e["file"] for e in entries]

    filt = json.load(open(args.cache / "filter.json"))
    drop_labels = tuple(filt.get("segments", {}).get("drop", []))

    # Row→shard spans across both splits, and the demo pose alignment.
    splits = ("precomputed_train", "precomputed_val")
    manifests = {s: json.load(open(args.cache / s / "manifest.json")) for s in splits}
    demo_spans: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    for split in splits:
        for si, shard in enumerate(manifests[split]["shards"]):
            offset = 0
            for di, el in zip(shard["demo_idxs"], shard["episode_lengths"]):
                demo_spans[demo_by_pos[di]].append((split, si, offset, el))
                offset += el

    pose_all = load_pose(args.pose_dir)
    demo_pose: dict[str, np.ndarray] = {}
    for demo, spans in demo_spans.items():
        kept = kept_pose(pose_all[demo], by_file[demo], drop_labels)
        total = sum(l for _, _, _, l in spans)
        if len(kept) != total:
            raise ValueError(f"{demo}: kept pose {len(kept)} != cache rows {total}")
        demo_pose[demo] = kept

    # One MapProbes per map present in the cache, from the navatlas dumps.
    cache_maps = sorted({by_file[d]["map"] for d in demo_spans})
    maps: dict[str, MapProbes] = {
        m: load_navatlas(args.navatlas_dir, m) for m in cache_maps
    }
    summary = {m: {"probes": int(len(mp.positions))} for m, mp in maps.items()}
    for m, s in summary.items():
        print(f"{m}: {s['probes']} poly-probes")

    if args.export_tables is not None:
        arrays: dict[str, np.ndarray] = {}
        for m, mp in maps.items():
            arrays[f"{m}__positions"] = mp.positions
            arrays[f"{m}__panoramas"] = mp.panoramas
        args.export_tables.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.export_tables,
            __meta__=np.frombuffer(
                json.dumps({
                    "placement": "navmesh_poly", "k": args.k,
                    "offset_dim": PROBE_OFFSET_DIM, "maps": summary,
                }).encode(), dtype=np.uint8,
            ),
            **arrays,
        )
        print(f"probe tables -> {args.export_tables}")
        if args.tables_only:
            return 0

    # Per shard, assemble (rows, K, ...) fields in row order.
    for split in splits:
        for si, shard in enumerate(manifests[split]["shards"]):
            rows = shard["rows"]
            pano_out = np.zeros(
                (rows, args.k, SPATIAL_TOKEN_COUNT, ATLAS_YAWS), dtype=np.uint8,
            )
            off_out = np.zeros(
                (rows, args.k, PROBE_OFFSET_DIM), dtype=np.float16,
            )
            offset = 0
            for di, el in zip(shard["demo_idxs"], shard["episode_lengths"]):
                demo = demo_by_pos[di]
                entry = by_file[demo]
                # Episode's position inside the demo's kept-row sequence.
                seq0 = 0
                for sp, s2, st, ln in demo_spans[demo]:
                    if (sp, s2, st) == (split, si, offset):
                        break
                    seq0 += ln
                else:
                    raise AssertionError("episode span not found")
                pose = demo_pose[demo][seq0:seq0 + el]
                pano, offs = maps[entry["map"]].assign(
                    pose[:, :3], pose[:, 3], args.k,
                )
                pano_out[offset:offset + el] = pano
                off_out[offset:offset + el] = offs
                offset += el
            prefix = f"shard{si:06d}"
            np.save(args.cache / split / f"{prefix}_obs_probe_atlas.npy",
                    pack_nibbles(pano_out))
            np.save(args.cache / split / f"{prefix}_obs_probe_offsets.npy",
                    off_out)
            shard["obs"]["probe_atlas"] = f"{prefix}_obs_probe_atlas.npy"
            shard["obs"]["probe_offsets"] = f"{prefix}_obs_probe_offsets.npy"
            print(f"{split} shard {si}: wrote probe fields ({rows} rows)")
        with open(args.cache / split / "manifest.json", "w") as fh:
            json.dump(manifests[split], fh)

    with open(args.cache / "probe_grid.json", "w") as fh:
        json.dump(
            {"placement": "navmesh_poly", "k": args.k,
             "offset_dim": PROBE_OFFSET_DIM, "maps": summary},
            fh, indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
