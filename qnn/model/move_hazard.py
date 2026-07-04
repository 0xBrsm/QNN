"""Data-driven move-axis dwell-hazard release table: tabulate, sufficient stats.

The move decode is sticky: once an axis (fb / lr) commits to a class it HOLDS,
and only releases (allows a switch) with a per-frame probability that depends on
how long it has been held. That release probability is a property of the *human
corpus* (how long people hold strafe/forward before changing), not of any trained
model — exactly like the look grid. And, exactly like the look grid, it is
**rate-dependent**: a 10 Hz collect holds ~half as many frames as a 20 Hz one for
the same wall-clock dwell, so the dwell-age bucket edges differ by tick rate.

  hazard[axis][held_class][bucket] = P(switch next frame | held class, dwell bucket)
  bucket b = #{e : dwell_age > edges[e]}        (len(edges)+1 buckets; dwell
                                                 resets at episode boundaries)
  axis      = 0:fb 1:lr      held_class = 0:neg 1:none 2:pos

This is computed at COLLECT time (beside ``qnn.model.look_grid``) and recorded in
``collect_metadata.json['move_hazard']``; a *run* pins the table into its own
``config/move_hazard.json`` (adopt-not-recompute, like the look grid). The move
decode reads ``edges`` + ``fb`` + ``lr`` from the run's pinned table, so the table
carries its own bucket edges and the decode never assumes a tick rate.

NumPy only — no torch. ``qnn.bc.collect`` and its import chain must stay
torch-free, so this module (imported by the collect) cannot pull torch.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

# Dwell-age bucket edges per tick rate (11 buckets = 10 edges). These are the
# proven tables' edges: 10 Hz is the data-driven set from the original a24rc2
# tabulation; 20 Hz is Fibonacci (spacing ~2× the 10 Hz set, matching the ~2×
# frames-per-dwell at double the tick rate). The table carries whichever it used.
EDGES_BY_HZ: dict[int, list[int]] = {
    10: [1, 2, 3, 4, 6, 9, 13, 19, 28, 42],
    20: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
}
N_AXES = 2          # fb, lr
N_HELD = 3          # neg, none, pos
HELD_NAMES = ["neg", "none", "pos"]
AXIS_NAMES = ["fb", "lr"]


def default_edges(tick_hz: int | float | None) -> list[int]:
    """Bucket edges for a tick rate. Falls back to the 20 Hz set when unknown
    (the dense default; long-dwell tail still resolves at lower rates)."""
    try:
        return EDGES_BY_HZ[int(round(float(tick_hz)))]
    except (TypeError, ValueError, KeyError):
        return EDGES_BY_HZ[20]


def unpack_move(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Packed move byte (N,) uint8 → (fb_class, lr_class), each (N,) in {0,1,2}.

    Bit layout (engine move usercmd packing): bit1=back, bit2=fwd, bit3=left,
    bit4=right. class = 1 + positive − negative ∈ {0:neg, 1:none, 2:pos}."""
    v = np.asarray(v, dtype=np.int64)
    fb = 1 + ((v >> 2) & 1) - ((v >> 1) & 1)
    lr = 1 + ((v >> 4) & 1) - ((v >> 3) & 1)
    return fb, lr


def _accum(rel: np.ndarray, tot: np.ndarray, axis_idx: int, c: np.ndarray, edges: np.ndarray) -> None:
    """Accumulate dwell-survival release counts for one axis over one episode.

    ``rel``/``tot`` are (N_AXES, N_HELD, n_bucket); ``c`` is the per-frame class
    sequence for one episode (dwell resets at the episode boundary)."""
    if len(c) < 2:
        return
    dwell = np.ones(len(c), dtype=np.int64)
    for i in range(1, len(c)):
        dwell[i] = dwell[i - 1] + 1 if c[i] == c[i - 1] else 1
    held = c[:-1]
    bucket = (dwell[:-1, None] > edges[None, :]).sum(axis=1)
    switched = (c[1:] != c[:-1]).astype(np.float64)
    np.add.at(tot, (axis_idx, held, bucket), 1.0)
    np.add.at(rel, (axis_idx, held, bucket), switched)


def tabulate_hazard(collect_dir: str | Path, edges: list[int]) -> dict:
    """Walk a collect's train-split move labels and tabulate the dwell-hazard
    release table against ``edges``. Returns {edges, fb, lr, counts, n_frames}."""
    collect_dir = Path(collect_dir)
    man_path = collect_dir / "precomputed_train" / "manifest.json"
    man = json.loads(man_path.read_text())
    edges_arr = np.asarray(edges, dtype=np.int64)
    nb = len(edges) + 1
    rel = np.zeros((N_AXES, N_HELD, nb), dtype=np.float64)
    tot = np.zeros((N_AXES, N_HELD, nb), dtype=np.float64)
    n_frames = 0
    for si, sh in enumerate(man["shards"]):
        mv = sh.get("actions", {}).get("move")
        path = collect_dir / "precomputed_train" / (mv or f"shard{si:06d}_act_move.npy")
        v = np.asarray(np.load(path, mmap_mode="r"))
        fb, lr = unpack_move(v)
        off = 0
        for n in sh.get("episode_lengths", []):
            n = int(n)
            if n <= 0:
                continue
            _accum(rel, tot, 0, fb[off:off + n], edges_arr)
            _accum(rel, tot, 1, lr[off:off + n], edges_arr)
            off += n
        n_frames += int(v.shape[0])
    hazard = np.where(tot > 0, rel / np.maximum(tot, 1.0), 0.0)
    return {
        "edges": list(edges),
        "fb": hazard[0].round(6).tolist(),
        "lr": hazard[1].round(6).tolist(),
        "counts": {"fb": tot[0].sum(axis=-1).astype(np.int64).tolist(),
                   "lr": tot[1].sum(axis=-1).astype(np.int64).tolist()},
        "n_frames": n_frames,
    }


# ── log-normal equation form ─────────────────────────────────────────────────
# The bucketed table above is a non-parametric estimate of the discrete-time
# dwell hazard h(t)=P(switch next frame | held, dwell-age t). On the real human
# corpus (qwd) a 2-param LOG-NORMAL lifetime per (axis, held_class) recovers
# 95–99% of all the dwell-age structure the 11-param table captures (held-out
# per-frame Bernoulli LL), MATCHES/beats it in the long-dwell tail, and — unlike
# the table — can never read 0.0 there (the statue-mode freeze) because its tail
# is a smooth positive decay. Weibull/gamma are wrong (increasing hazard → forced
# release); human move dwell is heavy-tailed = the log-normal family. See
# research/move-head.md and project memory.
#
# The decode/C/export wire contract is unchanged: this fits (mu, sigma) per cell
# from fine integer-age counts, then EXPANDS the equation back into the same
# (2,3,11) bucketed `fb`/`lr` table the decode already consumes. The equation is
# the source of truth (recorded for provenance + rate-invariant re-expansion);
# the table is its compiled artifact. ``method="lognorm"`` opts in; the default
# stays the empirical tabulation so existing collects are byte-identical.
LOGNORM_MAXAGE = 400      # integer-age horizon for fitting + tail bucket integration
_SQRT2 = float(np.sqrt(2.0))


def _contiguous_runs(mask: np.ndarray) -> "list[np.ndarray]":
    """Frame indices of each maximal run of True in a 1-D boolean ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.split(idx, splits)


def _accum_fine(rel: np.ndarray, tot: np.ndarray, ax: int, cseq: np.ndarray) -> None:
    """Accumulate per-integer-age release/total for one dwell sequence (one axis,
    one contiguous run). Dwell resets at the start of ``cseq`` (the run boundary)."""
    if len(cseq) < 2:
        return
    change = np.empty(len(cseq), dtype=bool)
    change[0] = True
    change[1:] = cseq[1:] != cseq[:-1]
    idx = np.arange(len(cseq))
    age = idx - np.maximum.accumulate(np.where(change, idx, 0)) + 1
    held = cseq[:-1]
    age_held = np.minimum(age[:-1], LOGNORM_MAXAGE)
    switched = (cseq[1:] != cseq[:-1]).astype(np.float64)
    np.add.at(tot, (ax, held, age_held), 1.0)
    np.add.at(rel, (ax, held, age_held), switched)


def _fine_counts(collect_dir: str | Path,
                 noncombat: bool = False,
                 combat: bool = False) -> "tuple[np.ndarray, np.ndarray, int]":
    """Walk train-split move labels accumulating per-(axis, held, integer-age)
    release/total counts (the un-bucketed sufficient stats the equation is fit to).
    Returns (rel, tot) of shape (N_AXES, N_HELD, LOGNORM_MAXAGE+1) — index t holds
    dwell-age t (1..MAXAGE; ages > MAXAGE lump into the top index) — plus n_frames.

    Conditioning (mutually exclusive; default = ALL frames, episode-run dwell):
    ``noncombat=True`` restricts to NON-COMBAT frames (target_probs argmax == 0) —
    the context-free locomotion baseline of the rc1o move scheme. ``combat=True``
    is the inverse: ENGAGED frames (target_probs argmax != 0) — the in-fight
    dwell law (juke/strafe under fire). Both feed each contiguous conditioned run
    as its own dwell sequence so a held class never bridges a regime gap. These
    three (combat / non-combat / all) are the core human-movement static tables."""
    if noncombat and combat:
        raise ValueError("noncombat and combat are mutually exclusive")
    collect_dir = Path(collect_dir)
    base = collect_dir / "precomputed_train"
    man = json.loads((base / "manifest.json").read_text())
    shape = (N_AXES, N_HELD, LOGNORM_MAXAGE + 1)
    rel = np.zeros(shape, dtype=np.float64)
    tot = np.zeros(shape, dtype=np.float64)
    n_kept = 0
    for si, sh in enumerate(man["shards"]):
        a = sh.get("actions", {})
        v = np.asarray(np.load(base / (a.get("move") or f"shard{si:06d}_act_move.npy"), mmap_mode="r"))
        fb, lr = unpack_move(v)
        keep = None
        if noncombat or combat:
            tp = np.asarray(np.load(base / a["target_probs"], mmap_mode="r"), dtype=np.float32)
            engaged = np.argmax(tp, axis=1) != 0      # target_probs[:,0] = no-target slot
            keep = engaged if combat else ~engaged
        off = 0
        for n in sh.get("episode_lengths", []):
            n = int(n)
            if n <= 0:
                continue
            sl = slice(off, off + n); off += n
            if keep is not None:
                for run in _contiguous_runs(keep[sl]):
                    idx = run + sl.start
                    _accum_fine(rel, tot, 0, fb[idx])
                    _accum_fine(rel, tot, 1, lr[idx])
                    n_kept += len(run)
            else:
                _accum_fine(rel, tot, 0, fb[sl])
                _accum_fine(rel, tot, 1, lr[sl])
                n_kept += n
    return rel, tot, n_kept


def _lognorm_survival(ages: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """S(t)=P(T>t) for a log-normal lifetime in frames; S(0)=1. ``ages`` >= 0."""
    ages = np.asarray(ages, dtype=np.float64)
    s = np.ones_like(ages)
    pos = ages > 0
    z = (np.log(np.maximum(ages[pos], 1e-12)) - mu) / (max(sigma, 1e-6) * _SQRT2)
    from scipy.special import erfc
    s[pos] = 0.5 * erfc(z)
    return s


def _lognorm_hazard(ages: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Discrete-time hazard h(t)=1-S(t)/S(t-1) for integer ``ages`` (>=1)."""
    s = _lognorm_survival(ages, mu, sigma)
    s_prev = _lognorm_survival(ages - 1, mu, sigma)
    return np.clip(1.0 - s / np.maximum(s_prev, 1e-12), 1e-9, 1.0 - 1e-9)


def fit_lognorm(rel: np.ndarray, tot: np.ndarray) -> np.ndarray:
    """MLE (mu, sigma) per (axis, held_class) by maximizing the per-frame binomial
    likelihood of the discrete hazard over fine integer ages. Returns (N_AXES,
    N_HELD, 2) = [mu, sigma]. Cells with too little data fall back to (mu, sigma)
    of a memoryless fit so the expansion never NaNs."""
    from scipy import optimize
    ages = np.arange(1, LOGNORM_MAXAGE + 1)
    out = np.zeros((N_AXES, N_HELD, 2), dtype=np.float64)
    for ax in range(N_AXES):
        for hc in range(N_HELD):
            t = tot[ax, hc, 1:]
            s = rel[ax, hc, 1:]
            if t.sum() < 100:
                out[ax, hc] = [2.0, 1.0]
                continue

            def nll(p, t=t, s=s):
                h = _lognorm_hazard(ages, p[0], np.exp(p[1]))
                return -(s * np.log(h) + (t - s) * np.log1p(-h)).sum()

            res = optimize.minimize(nll, np.array([2.0, 0.0]), method="Nelder-Mead",
                                    options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
            out[ax, hc] = [float(res.x[0]), float(np.exp(res.x[1]))]
    return out


def lognorm_buckets(params: np.ndarray, edges: list[int],
                    maxage: int = LOGNORM_MAXAGE) -> np.ndarray:
    """Expand fitted (mu, sigma) per cell into the (N_AXES, N_HELD, len(edges)+1)
    bucketed release table the decode consumes — OCCUPANCY-CORRECT so the value
    equals what the empirical bucket would converge to under the equation:

        h[b] = (S(a_lo-1) - S(a_hi)) / Σ_{t=a_lo..a_hi} S(t-1)

    i.e. (expected releases in the bucket) / (expected trials in the bucket), with
    trials at age t weighted by survival S(t-1) (frames that reach age t). The
    overflow bucket integrates ages (edges[-1], maxage], so its value is a small
    POSITIVE rate (never the table's sampling-fragile 0.0)."""
    edges = list(edges)
    nb = len(edges) + 1
    out = np.zeros((N_AXES, N_HELD, nb), dtype=np.float64)
    for ax in range(N_AXES):
        for hc in range(N_HELD):
            mu, sigma = params[ax, hc]
            for b in range(nb):
                a_lo = (edges[b - 1] + 1) if b > 0 else 1
                a_hi = edges[b] if b < nb - 1 else maxage
                if a_hi < a_lo:
                    continue
                ages = np.arange(a_lo, a_hi + 1)
                releases = _lognorm_survival(np.array([a_lo - 1]), mu, sigma)[0] \
                    - _lognorm_survival(np.array([a_hi]), mu, sigma)[0]
                trials = _lognorm_survival(ages - 1, mu, sigma).sum()
                out[ax, hc, b] = releases / max(trials, 1e-12)
    return np.clip(out, 0.0, 1.0)


def lognorm_hazard_from_collect(collect_dir: str | Path,
                                tick_hz: int | float | None = None,
                                noncombat: bool = False,
                                combat: bool = False) -> dict:
    """Build the ``move_hazard`` block from the LOG-NORMAL equation: fit (mu,sigma)
    per cell from fine-age counts, expand to the tick-aware bucketed fb/lr table.
    Records the equation params (provenance + rate-invariant re-expansion) and the
    derived table the decode reads — same schema fields as the empirical block plus
    ``method`` and ``lognorm``.

    Conditioning (the three core static tables): default ALL train frames;
    ``noncombat=True`` = the rc1o context-free locomotion baseline (target==0); the
    deployed move scheme uses the NON-COMBAT law + engagement-gated tau. ``combat=True``
    = the in-fight dwell law (target!=0) — used to PREDICT an engaged opponent's
    movement (the aim-lead motion model), not for self-decode."""
    collect_dir = Path(collect_dir)
    if tick_hz is None:
        meta_path = collect_dir / "collect_metadata.json"
        if meta_path.exists():
            tick_hz = json.loads(meta_path.read_text()).get("tick_hz")
    edges = default_edges(tick_hz)
    rel, tot, n_frames = _fine_counts(collect_dir, noncombat=noncombat, combat=combat)
    params = fit_lognorm(rel, tot)
    table = lognorm_buckets(params, edges)
    regime = "combat" if combat else ("noncombat" if noncombat else "all")
    prov = {"combat": "COMBAT (target!=0) train frames; contiguous combat runs",
            "noncombat": "NON-COMBAT (target==0) train frames; contiguous non-combat runs",
            "all": "all train frames; episode-run dwell"}[regime]
    return {
        "schema": "move_hazard_v1",
        "method": "lognorm",
        "regime": regime,
        "provenance": prov,
        "tick_hz": tick_hz,
        "n_frames": n_frames,
        "edges": list(edges),
        "fb": table[0].round(6).tolist(),
        "lr": table[1].round(6).tolist(),
        "lognorm": {
            "regime": regime,
            "noncombat": bool(noncombat),
            "fb": [[round(float(m), 5), round(float(s), 5)] for m, s in params[0]],
            "lr": [[round(float(m), 5), round(float(s), 5)] for m, s in params[1]],
            "param_names": ["mu", "sigma"],
            "maxage": LOGNORM_MAXAGE,
        },
        "none_tail_release": {"fb": float(table[0, 1, -1]), "lr": float(table[1, 1, -1])},
    }


def compute_hazard_from_collect(collect_dir: str | Path, tick_hz: int | float | None = None,
                                method: str = "empirical", noncombat: bool = False,
                                combat: bool = False) -> dict:
    """Build the ``move_hazard`` metadata block for a collect: tick-aware bucket
    edges, the fb/lr release tables, and a statue-mode tail diagnostic (long-dwell
    'none' release must be > 0, else a no-contact bot can freeze).

    ``tick_hz`` defaults to the collect's recorded rate; the edges are chosen from
    it so a 10 Hz and a 20 Hz corpus get their own tables.

    ``method="lognorm"`` fits a log-normal dwell lifetime per cell and expands it
    into the same bucketed table (equation-as-source — see
    :func:`lognorm_hazard_from_collect`); the default ``"empirical"`` tabulates the
    raw bucketed release counts (byte-identical to pre-equation collects).

    ``noncombat=True`` (lognorm only) fits the rc1o non-combat locomotion baseline."""
    if method == "lognorm":
        return lognorm_hazard_from_collect(collect_dir, tick_hz=tick_hz,
                                           noncombat=noncombat, combat=combat)
    collect_dir = Path(collect_dir)
    if tick_hz is None:
        meta_path = collect_dir / "collect_metadata.json"
        if meta_path.exists():
            tick_hz = json.loads(meta_path.read_text()).get("tick_hz")
    edges = default_edges(tick_hz)
    tab = tabulate_hazard(collect_dir, edges)
    fb = np.asarray(tab["fb"]); lr = np.asarray(tab["lr"])
    return {
        "schema": "move_hazard_v1",
        "method": "empirical",
        "tick_hz": tick_hz,
        "n_frames": tab["n_frames"],
        "edges": tab["edges"],
        "fb": tab["fb"],
        "lr": tab["lr"],
        "counts": tab["counts"],
        "none_tail_release": {"fb": float(fb[1, -1]), "lr": float(lr[1, -1])},
    }


def pinned_hazard_from_collect(collect_dir: str | Path) -> dict:
    """Assemble the ``config/move_hazard.json`` a run pins, from the collect's
    recorded ``move_hazard`` block. Mirrors ``look_grid.pinned_grid_from_collect``:
    the collect generates the table, the run adopts it, existing runs never change
    retroactively."""
    collect_dir = Path(collect_dir)
    meta = json.loads((collect_dir / "collect_metadata.json").read_text())
    mh = meta.get("move_hazard")
    if not mh or "fb" not in mh:
        raise ValueError(
            f"{collect_dir}/collect_metadata.json has no move_hazard — recollect "
            "(or backfill via `python -m qnn.model.move_hazard <collect_dir> --backfill`) first.")
    pinned = {
        "schema": "move_hazard_v1",
        "source": "corpus_fit",
        "method": mh.get("method", "empirical"),
        "corpus": str(collect_dir),
        "tick_hz": mh.get("tick_hz"),
        "edges": mh["edges"],
        "fb": mh["fb"],
        "lr": mh["lr"],
    }
    if "lognorm" in mh:                 # carry equation provenance into the run pin
        pinned["lognorm"] = mh["lognorm"]
    return pinned


def _backfill(collect_dir: str | Path, method: str = "empirical") -> dict:
    """Compute the move_hazard block and inject it into an existing collect's
    ``collect_metadata.json`` in place (for corpora collected before this module)."""
    collect_dir = Path(collect_dir)
    meta_path = collect_dir / "collect_metadata.json"
    meta = json.loads(meta_path.read_text())
    block = compute_hazard_from_collect(collect_dir, tick_hz=meta.get("tick_hz"), method=method)
    meta["move_hazard"] = block
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return block


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tabulate / backfill the move-hazard table for a collect.")
    ap.add_argument("collect_dir", type=Path)
    ap.add_argument("--backfill", action="store_true",
                    help="inject the computed move_hazard block into collect_metadata.json")
    ap.add_argument("--method", choices=["empirical", "lognorm"], default="empirical",
                    help="empirical bucketed tabulation (default) or log-normal equation fit")
    ap.add_argument("--noncombat", action="store_true",
                    help="(lognorm) fit the rc1o non-combat locomotion baseline (target==0, contiguous runs)")
    ap.add_argument("--combat", action="store_true",
                    help="(lognorm) fit the in-fight dwell law (target!=0, contiguous combat runs)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the computed move_hazard block to this JSON (the static-table artifact)")
    ap.add_argument("--tick-hz", type=float, default=None)
    args = ap.parse_args()
    block = (_backfill(args.collect_dir, method=args.method) if args.backfill
             else compute_hazard_from_collect(args.collect_dir, tick_hz=args.tick_hz,
                                               method=args.method, noncombat=args.noncombat,
                                               combat=args.combat))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(block, indent=2) + "\n")
        print(f"wrote {args.out}")
    print(f"move_hazard: method={block.get('method','empirical')} tick_hz={block['tick_hz']} "
          f"edges={block['edges']} n_frames={block['n_frames']:,}")
    print("none-row last-bucket release (statue-mode check; 0.0 => freeze risk): "
          f"fb={block['none_tail_release']['fb']:.4f} lr={block['none_tail_release']['lr']:.4f}")
    if "lognorm" in block:
        ln = block["lognorm"]
        print(f"  log-normal (mu, sigma) per cell  [median_dwell = exp(mu) frames]:")
        for a in AXIS_NAMES:
            for c, (mu, sg) in zip(HELD_NAMES, ln[a]):
                print(f"    {a}.{c:4} mu={mu:6.3f} sigma={sg:5.3f}  median={np.exp(mu):6.1f}f")
    for a in range(N_AXES):
        rows = block[AXIS_NAMES[a]]
        print(f"  {AXIS_NAMES[a]}:")
        for c in range(N_HELD):
            print(f"    {HELD_NAMES[c]:4} [" + " ".join(f"{x:.3f}" for x in rows[c]) + "]")
    if args.backfill:
        print(f"\nwrote move_hazard -> {args.collect_dir}/collect_metadata.json")


if __name__ == "__main__":
    main()
