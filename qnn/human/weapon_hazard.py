"""Data-driven weapon WHEN-hazard table: P(intent-switch | held weapon, dwell-age, n_owned).

The weapon decode's WHAT (which weapon) is solved by the head; the residual is
WHEN (when to leave the held weapon). Like move's dwell-hazard and the look grid,
*when* a human switches weapon is a property of the **corpus**, not a trained
model. It depends on three things the corpus settles (validated against the clean
corpus, F1 0.53 / per-cell train→held-out corr 0.96):

  hazard[held_impulse][dwell_bucket][n_owned] = P(act.weapon != self_weapon_id | ...)
  held_impulse  = 1..8 (Axe SG SSG NG SNG GL RL LG); Axe is ditched fast, RL/LG kept
  dwell_bucket  = #{e : dwell_age > edges[e]}  (negative duration dependence)
  n_owned       = #weapons owned (0..8); more options → higher switch hazard (+0.09 AUC)

Held / dwell / n_owned are the signals that survived a full sweep; EHP and target
distance were tested and add ~nothing on top (EHP +0.005 AUC and is hold-inertia not
self-splash; distance is noise — see weapon-head.md). Kept simple/dense by design.

**Intent-only + combat-fit.** The hazard is the per-frame probability the human
intends a weapon other than the equipped one, fit **only on engaged frames**
(`target_probs` argmax != 0 ≈ act.target != 0); roaming has a ~3.7× lower, different
switch regime that would dilute it. Forced switches (ammo-out / pickup / respawn)
live in `self_weapon_id` itself, so the bot follows them via the engine and is never
charged here. dwell_age is computed over the FULL episode (true hold duration), the
hazard accumulated only on engaged frames. The decode gates the head's WHAT pick on a
Bernoulli(hazard) draw with a persisted RNG (move-style).

Computed as a post-collect step (beside ``qnn.human.move_hazard`` / ``look_grid``), recorded
in ``collect_metadata.json['weapon_hazard']``; a *run* pins it into
``config/weapon_hazard.json`` (adopt-not-recompute). Rate-dependent like move.

NumPy only (``qnn.vocab`` / ``qnn.engine_norm`` are torch-free) — ``qnn.bc.collect``
imports this and must stay torch-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qnn import engine_norm as en
from qnn.vocab import self_weapon_id_to_impulse

# The tick-aware dwell-age bucket edges are the SAME as move (same corpus timing,
# validated 20 Hz Fibonacci set, corr 0.96 train→held-out) — import from their owner
# rather than duplicate the table + the fallback.
from qnn.human.move_hazard import EDGES_BY_HZ, default_edges  # noqa: F401,E402

N_HELD = 8           # impulse 1..8 stored at rows 0..7
N_OWNED_BINS = 9     # n_owned 0..8 → index 0..8 (small ints, used directly; dense)
HELD_NAMES = ["Axe", "SG", "SSG", "NG", "SNG", "GL", "RL", "LG"]
# Weapon-ownership bits in the engine items bitmask (deploy-consistent: the engine
# computes n_owned the same way from its own item state).
_WEAPON_BITS = [b for b in range(32) if (en.ITEMS_WEAPON_MASK >> b) & 1]


def n_owned_from_items(items: np.ndarray) -> np.ndarray:
    """Count owned weapons from the engine items bitmask (popcount over weapon bits)."""
    items = np.asarray(items, dtype=np.int64).reshape(-1)
    n = np.zeros(len(items), dtype=np.int64)
    for b in _WEAPON_BITS:
        n += (items >> b) & 1
    return np.clip(n, 0, N_OWNED_BINS - 1)


def _accum(intent_sum: np.ndarray, tot: np.ndarray, held: np.ndarray, act: np.ndarray,
           n_owned: np.ndarray, engaged: np.ndarray, edges: np.ndarray) -> None:
    """Accumulate per-(held,bucket,n_owned) intent counts over one episode.

    dwell_age is over the FULL episode (true hold duration); the hazard is accumulated
    only on engaged frames carrying a real weapon. ``intent_sum``/``tot`` are
    (N_HELD, nb, N_OWNED_BINS) at row = held impulse - 1."""
    n = len(held)
    if n == 0:
        return
    dwell = np.ones(n, dtype=np.int64)
    for i in range(1, n):
        dwell[i] = dwell[i - 1] + 1 if held[i] == held[i - 1] else 1
    bucket = (dwell[:, None] > edges[None, :]).sum(axis=1)
    keep = (held >= 1) & (engaged != 0)          # combat frames with a real weapon
    rows = held[keep] - 1
    bk = bucket[keep]
    no = n_owned[keep]
    intent = (act[keep] != held[keep]).astype(np.float64)
    np.add.at(tot, (rows, bk, no), 1.0)
    np.add.at(intent_sum, (rows, bk, no), intent)


def tabulate_hazard(collect_dir: str | Path, edges: list[int]) -> dict:
    """Walk a collect's train split and tabulate the 3-axis combat intent-hazard.

    Returns {edges, hazard (N_HELD×nb×N_OWNED, Laplace-smoothed), counts, n_frames,
    n_engaged_frames}."""
    collect_dir = Path(collect_dir)
    base = collect_dir / "precomputed_train"
    man = json.loads((base / "manifest.json").read_text())
    edges_arr = np.asarray(edges, dtype=np.int64)
    nb = len(edges) + 1
    intent_sum = np.zeros((N_HELD, nb, N_OWNED_BINS), dtype=np.float64)
    tot = np.zeros((N_HELD, nb, N_OWNED_BINS), dtype=np.float64)
    n_frames = n_engaged = 0
    for si, sh in enumerate(man["shards"]):
        wp = sh.get("actions", {}).get("weapon") or f"shard{si:06d}_act_weapon.npy"
        sp = sh.get("obs", {}).get("self_weapon_id") or f"shard{si:06d}_obs_self_weapon_id.npy"
        ip = sh.get("obs", {}).get("self_items") or f"shard{si:06d}_obs_self_items.npy"
        tpf = sh.get("actions", {}).get("target_probs") or f"shard{si:06d}_act_target_probs.npy"
        act = np.asarray(np.load(base / wp, mmap_mode="r")).reshape(-1).astype(np.int64)
        held = np.asarray(self_weapon_id_to_impulse(
            np.asarray(np.load(base / sp, mmap_mode="r")).reshape(-1).astype(np.int64)))
        n_owned = n_owned_from_items(np.load(base / ip, mmap_mode="r"))
        tp = np.asarray(np.load(base / tpf, mmap_mode="r"))
        engaged = (tp.reshape(tp.shape[0], -1).argmax(axis=1) != 0).astype(np.int64)
        off = 0
        for nn in sh.get("episode_lengths", []):
            nn = int(nn)
            if nn <= 0:
                continue
            sl = slice(off, off + nn)
            _accum(intent_sum, tot, held[sl], act[sl], n_owned[sl], engaged[sl], edges_arr)
            off += nn
        n_frames += int(act.shape[0])
        n_engaged += int(engaged.sum())
    hazard = (intent_sum + 1.0) / (tot + 2.0)    # Laplace-smoothed
    hazard = np.where(tot > 0, hazard, 0.0)
    return {
        "edges": list(edges),
        "hazard": hazard.round(6).tolist(),       # (N_HELD, nb, N_OWNED_BINS)
        "counts": tot.sum(axis=(1, 2)).astype(np.int64).tolist(),
        "n_frames": n_frames,
        "n_engaged_frames": n_engaged,
    }


def compute_hazard_from_collect(collect_dir: str | Path,
                                tick_hz: int | float | None = None) -> dict:
    """Build the ``weapon_hazard`` metadata block for a collect (tick-aware edges +
    the 3-axis combat intent-hazard table)."""
    collect_dir = Path(collect_dir)
    if tick_hz is None:
        meta_path = collect_dir / "collect_metadata.json"
        if meta_path.exists():
            tick_hz = json.loads(meta_path.read_text()).get("tick_hz")
    edges = default_edges(tick_hz)
    tab = tabulate_hazard(collect_dir, edges)
    return {
        "schema": "weapon_hazard_v1",
        "axes": ["held_impulse", "dwell_bucket", "n_owned"],
        "tick_hz": tick_hz,
        "segment": "engaged (target_probs argmax != 0)",
        "n_frames": tab["n_frames"],
        "n_engaged_frames": tab["n_engaged_frames"],
        "edges": tab["edges"],
        "hazard": tab["hazard"],
        "counts": tab["counts"],
    }


def pinned_hazard_from_collect(collect_dir: str | Path) -> dict:
    """Assemble the ``config/weapon_hazard.json`` a run pins from the collect's
    recorded ``weapon_hazard`` block (adopt-not-recompute, like move/look-grid)."""
    collect_dir = Path(collect_dir)
    meta = json.loads((collect_dir / "collect_metadata.json").read_text())
    wh = meta.get("weapon_hazard")
    if not wh or "hazard" not in wh:
        raise ValueError(
            f"{collect_dir}/collect_metadata.json has no weapon_hazard — recollect "
            "(or backfill via `python -m qnn.human <collect_dir>`).")
    return {
        "schema": "weapon_hazard_v1",
        "axes": wh.get("axes", ["held_impulse", "dwell_bucket", "n_owned"]),
        "source": "corpus_fit",
        "corpus": str(collect_dir),
        "tick_hz": wh.get("tick_hz"),
        "segment": wh.get("segment"),
        "edges": wh["edges"],
        "hazard": wh["hazard"],
    }


def _backfill(collect_dir: str | Path) -> dict:
    """Compute the weapon_hazard block and inject it into an existing collect's
    ``collect_metadata.json`` in place."""
    collect_dir = Path(collect_dir)
    meta_path = collect_dir / "collect_metadata.json"
    meta = json.loads(meta_path.read_text())
    block = compute_hazard_from_collect(collect_dir, tick_hz=meta.get("tick_hz"))
    meta["weapon_hazard"] = block
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return block


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tabulate / backfill the weapon-hazard table for a collect.")
    ap.add_argument("collect_dir", type=Path)
    ap.add_argument("--backfill", action="store_true",
                    help="inject the computed weapon_hazard block into collect_metadata.json")
    ap.add_argument("--tick-hz", type=float, default=None)
    args = ap.parse_args()
    block = (_backfill(args.collect_dir) if args.backfill
             else compute_hazard_from_collect(args.collect_dir, tick_hz=args.tick_hz))
    haz = np.asarray(block["hazard"])  # (N_HELD, nb, N_OWNED)
    frac = block["n_engaged_frames"] / max(block["n_frames"], 1)
    print(f"weapon_hazard: tick_hz={block['tick_hz']} edges={block['edges']} "
          f"n_frames={block['n_frames']:,} engaged_frac={frac:.3f} axes={block['axes']}")
    print("  hazard marginalized over n_owned (mean weighted by counts), per held × dwell:")
    cnt = np.asarray(block["counts"])
    haz_mean = haz.mean(axis=2)  # simple mean over n_owned for display
    for w in range(N_HELD):
        print(f"    {HELD_NAMES[w]:>4} (n={int(cnt[w]):>8}) [" + " ".join(f"{x:.3f}" for x in haz_mean[w]) + "]")
    if args.backfill:
        print(f"\nwrote weapon_hazard -> {args.collect_dir}/collect_metadata.json")


if __name__ == "__main__":
    main()
