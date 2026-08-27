"""Per-equipped-weapon HUMAN engagement-RANGE distribution (the decode-fit pooling prior).

For each equipped-weapon impulse W (1..8 = Axe,SG,SSG,NG,SNG,GL,RL,LG), accumulates the
distribution of ENGAGEMENT RANGE (units) over engaged-LOS frames — the Euclidean
norm of the ``rel`` of the MOST-ALIGNED LOS actor (the SAME referent the lead-aim
kernel scores: the argmin over ``_lead_aim_angle_deg_live`` masked to LOS actors). Humans
self-select the range they fight each weapon at, so this per-weapon range histogram
is the prior for POOLING the closed-loop botpin grid (which averages 4 frikbot
range-archetype pins). Uniform pin pooling biases native placement toward
off-range cells (e.g. it scored pmh60 LG native at p20 by averaging in 128u/180u
pins where humans mostly fight LG at ~350u); weighting the pins by THIS
distribution range-matches the pool. See skill-curves.md §1.7 correction (7/06).

Uses the canonical aim_skill kernel (``A``) for shard/manifest loading,
engaged-frame definition (target_probs argmax>0 AND >=1 LOS actor recency==0),
episode admit threshold (>=30 engaged frames), and the 1/_DIST_SCALE unscaling.

The 4 frikbot grid pins are range archetypes (aim_grid_closedloop.py): shotgun=128u,
nailgun=180u (hitscan), rocket_launcher=180u (projectile/dodging), lightning=350u.
pin_masses assign each frame's range to the nearest pin range by Voronoi bins on the
range axis: [0,154) -> 128, [154,265) -> 180, [265,inf) -> 350 (154 = midpoint
128/180, 265 = midpoint 180/350). pin_weights carry the 180 mass split 50/50 between
fng and frl (they share range; they differ only by target motion, which the range
axis cannot separate), normalized to sum 1 per weapon.

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.aim_range \\
      --collect-dir artifacts/collect/qwd [--split both] [--workers N] \\
      [--out <collect>/human_baseline/_aim_range_byweapon.json]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# The canonical aim_skill kernel `A` — the shared lead/geometry kernel, physics
# tables, and loading/engaged-frame logic (no copying).
from qnn.eval import aim_kernel as A

IMPULSE_NAME = {                         # impulse 1..8 = Axe,SG,SSG,NG,SNG,GL,RL,LG
    1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
    6: "GL", 7: "RL", 8: "LG",
}
MIN_ENGAGED_FRAMES_EP = 30               # episode admit (matches aim_skill)
_WN, _ZD = A._build_physics_tables()

# Direct-fire weapons emitted with the fitting weapons SSG/SNG (which transfer off
# SG/NG on the grid) — the JSON carries all six so the pipeline can key by abbr.
EMIT_WEAPONS = ("SG", "SSG", "NG", "SNG", "LG", "RL")

# Range histogram: 32u buckets to 1024u (32 finite bins) + one overflow bucket.
HIST_STEP = 32
HIST_MAX = 1024
HIST_EDGES = list(range(0, HIST_MAX + HIST_STEP, HIST_STEP))   # [0,32,...,1024] (33)
N_BINS = len(HIST_EDGES)                                       # 33 (last = overflow >=1024)
OVERFLOW_IDX = N_BINS - 1                                      # index 32

# Frikbot pin ranges (aim_grid_closedloop.py archetypes) + the range->pin Voronoi.
PIN_RANGES = {"fsg": 128, "fng": 180, "frl": 180, "flg": 350}
PIN_LABELS = ("128", "180", "350")           # nearest-pin-range mass buckets
PIN_VORONOI = (154.0, 265.0)                 # [0,154)->128, [154,265)->180, [265,inf)->350
# frikbot model-weapon -> grid pin tag (matches the grid `frikbot_weapon` field).
FRIKBOT_TO_PIN = {"shotgun": "fsg", "nailgun": "fng",
                  "rocket_launcher": "frl", "lightning": "flg"}


def _episode_perweapon_range(
    ep_cnt, ep_rel, ep_vel, ep_typ, ep_rec, ep_weapon, ep_engaged,
) -> dict[int, np.ndarray]:
    """Return {impulse: ranges[n_los_frames]} for one episode.

    Mirrors the aim_skill engaged-LOS geometry/masking EXACTLY, but per
    engaged-LOS frame emits the RANGE (world units) of the most-aligned LOS actor
    — the same argmin referent the lead-aim kernel picks — instead of an alignment flag.
    """
    # Shared densify (raw padded arrays; caller applies the unit scale) — aim_kernel.
    all_rel, all_vel, all_los = A.densify_entities(ep_cnt, ep_rel, ep_vel, ep_typ, ep_rec)
    all_rel *= (1.0 / A._DIST_SCALE)   # world units (same unscaling as the aim_skill kernel)
    all_vel *= (1.0 / A._VEL_SCALE)

    has_los = all_los.any(axis=1)
    eng = ep_engaged & has_los
    eng_idx = np.where(eng)[0]

    out: dict[int, np.ndarray] = {}
    if not len(eng_idx):
        return out

    rel = all_rel[eng_idx]
    vel = all_vel[eng_idx]
    los = all_los[eng_idx]
    imp = A.action_attack_context(ep_weapon)[eng_idx]

    angles = A._lead_aim_angle_deg_live(rel, vel, imp, _WN)   # (M, 16)
    angles = np.where(los, angles, np.inf)
    min_a = angles.min(axis=1)                       # (M,)
    arg = angles.argmin(axis=1)                      # most-aligned LOS actor slot
    finite = np.isfinite(min_a)                      # NaN/inf guard

    M = rel.shape[0]
    # rel here is normalized (/_DIST_SCALE, as the angle kernel needs); recover the
    # world-unit range by multiplying the norm back by _DIST_SCALE.
    rng = np.linalg.norm(rel[np.arange(M), arg], axis=-1) * A._DIST_SCALE   # world units
    imp_f = imp[finite]
    rng_f = rng[finite].astype(np.float64)
    for w in np.unique(imp_f):
        out[int(w)] = rng_f[imp_f == w]
    return out


def _accumulate(ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hist[N_BINS], pin[3]) counts for one weapon's ranges."""
    hist = np.zeros(N_BINS, dtype=np.int64)
    idx = np.clip((ranges // HIST_STEP).astype(np.int64), 0, OVERFLOW_IDX)
    for i in idx:
        hist[i] += 1
    pin = np.zeros(3, dtype=np.int64)
    pin[0] = int((ranges < PIN_VORONOI[0]).sum())
    pin[1] = int(((ranges >= PIN_VORONOI[0]) & (ranges < PIN_VORONOI[1])).sum())
    pin[2] = int((ranges >= PIN_VORONOI[1]).sum())
    return hist, pin


def _worker(args: tuple) -> dict[int, np.ndarray]:
    """Returns {impulse: concatenated ranges} for one shard (mirrors the
    aim_skill shard loading; accumulates ranges, not alignment pairs)."""
    sh, data_dir_str = args
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for _ei, _dmi, fsl, esl, arr in A.iter_shard_episodes(
            sh, data_dir_str,
            obs=("entity_rel", "entity_vel", "entity_types", "entity_recency"),
            acts=("target_probs", "attack")):
        ep_engaged = np.asarray(arr["target_probs"][fsl]).argmax(axis=1) > 0
        if int(ep_engaged.sum()) < MIN_ENGAGED_FRAMES_EP:
            continue
        per = _episode_perweapon_range(
            ep_cnt=np.asarray(arr["entity_count"][fsl], dtype=np.int32),
            ep_rel=np.asarray(arr["entity_rel"][esl]),
            ep_vel=np.asarray(arr["entity_vel"][esl]),
            ep_typ=np.asarray(arr["entity_types"][esl]),
            ep_rec=np.asarray(arr["entity_recency"][esl]),
            ep_weapon=np.asarray(arr["attack"][fsl]),
            ep_engaged=ep_engaged,
        )
        for w, r in per.items():
            if len(r):
                acc[w].append(r)
    return {w: np.concatenate(rs) for w, rs in acc.items()}


def _pin_weights(pin: np.ndarray) -> dict[str, float]:
    """{fsg,fng,frl,flg} weights: 128 mass -> fsg, 180 mass split 50/50 -> fng+frl,
    350 mass -> flg; normalized to sum 1."""
    m128, m180, m350 = float(pin[0]), float(pin[1]), float(pin[2])
    raw = {"fsg": m128, "fng": m180 / 2.0, "frl": m180 / 2.0, "flg": m350}
    tot = sum(raw.values())
    if tot <= 0:
        return {k: 0.0 for k in raw}
    return {k: round(v / tot, 6) for k, v in raw.items()}


def run(collect_dir: Path, splits: list[str], out_path: Path, n_workers: int) -> None:
    # Accumulate per-impulse range counts globally (the range PRIOR is aggregate
    # mass, not a per-player percentile — no demo_idx bucketing needed).
    hist_acc: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(N_BINS, dtype=np.int64))
    pin_acc: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    n_frames: dict[int, int] = defaultdict(int)

    for split in splits:
        dd = collect_dir / f"precomputed_{split}"
        man = dd / "manifest.json"
        if not man.exists():
            print(f"  [{split}] no manifest, skip")
            continue
        shards = json.loads(man.read_text())["shards"]
        tasks = [(sh, str(dd)) for sh in shards]
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
                for w, ranges in r.items():
                    h, p = _accumulate(ranges)
                    hist_acc[w] += h
                    pin_acc[w] += p
                    n_frames[w] += int(len(ranges))
                print(f"  [{split}] {i+1}/{len(tasks)} shards done", flush=True)

    weapons_out: dict[str, Any] = {}
    for imp in range(1, 9):
        name = IMPULSE_NAME[imp]
        if name not in EMIT_WEAPONS:
            continue
        hist = hist_acc.get(imp, np.zeros(N_BINS, dtype=np.int64))
        pin = pin_acc.get(imp, np.zeros(3, dtype=np.int64))
        weapons_out[name] = {
            "impulse": imp,
            "n_frames": int(n_frames.get(imp, 0)),
            "hist_edges": HIST_EDGES,
            "hist": [int(x) for x in hist],
            "pin_masses": {PIN_LABELS[i]: int(pin[i]) for i in range(3)},
            "pin_weights": _pin_weights(pin),
        }

    out = {
        "metric": "engagement_range_units",
        "measurement": "range_of_most_aligned_los_actor_per_action_attack_context",
        "splits": splits,
        "collect_dir": str(collect_dir),
        "engaged_def": "target_probs argmax>0 AND >=1 LOS actor (recency==0)",
        "episode_admit_min_engaged_frames": MIN_ENGAGED_FRAMES_EP,
        "referent": ("Euclidean norm of rel of the argmin over "
                     "_lead_aim_angle_deg_live masked to LOS actors — the SAME "
                     "most-aligned referent the lead-aim kernel scores."),
        "hist": {"step_units": HIST_STEP, "max_units": HIST_MAX,
                 "n_bins": N_BINS, "overflow_bin_index": OVERFLOW_IDX,
                 "note": ("hist[i] counts range in [edges[i], edges[i+1]) for i<"
                          f"{OVERFLOW_IDX}; hist[{OVERFLOW_IDX}] = overflow (>= "
                          f"{HIST_MAX}u).")},
        "pin_ranges": PIN_RANGES,
        "pin_voronoi": {"128": "[0,154)", "180": "[154,265)", "350": "[265,inf)"},
        "note": ("pin_weights: the 180u pin mass is split 50/50 between fng and "
                 "frl — they share engagement RANGE and differ only by target "
                 "motion (hitscan vs projectile/dodging), which the range axis "
                 "cannot separate. Weights normalized to sum 1 per weapon."),
        "usage": ("range-weighted pooling prior for qnn.eval.skill_vector "
                  "perweapon_grid_coh / alpha grid (replaces uniform frikbot "
                  "averaging); consumed by decode_fit_pipeline.build_perweapon_anchors."),
        "weapons": weapons_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    # Console report.
    print(f"\n{'weapon':>6} {'imp':>3} {'n_frames':>9}  "
          f"{'fsg(128)':>9}{'fng(180)':>9}{'frl(180)':>9}{'flg(350)':>9}")
    for name in EMIT_WEAPONS:
        e = weapons_out.get(name)
        if not e:
            continue
        w = e["pin_weights"]
        print(f"{name:>6} {e['impulse']:>3} {e['n_frames']:>9}  "
              f"{w['fsg']:>9.4f}{w['fng']:>9.4f}{w['frl']:>9.4f}{w['flg']:>9.4f}")
    print(f"\nWritten -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="both", choices=["val", "train", "both"])
    # cap at 8: other CPU evals share this box (feedback_no_collect_alongside_trainer).
    ap.add_argument("--workers", type=int, default=min(8, max(1, mp.cpu_count() - 1)),
                    help="capped at 8 (other CPU evals run concurrently)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <collect>/human_baseline/_aim_range_byweapon.json)")
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_aim_range_byweapon.json"
    workers = min(8, args.workers)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    print(f"Collect: {args.collect_dir}  splits={splits}  workers={workers}")
    run(args.collect_dir, splits, out, workers)


if __name__ == "__main__":
    main()
