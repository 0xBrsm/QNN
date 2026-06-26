"""Move-head analysis functions — importable core compute for the move slice.

Each public function corresponds to one canonical analysis script:

    momentum_baseline    ← move_momentum_baseline.py
    stream_dynamics      ← move_stream_dynamics.py
    jump_discrim         ← jump_discrim.py
    jump_onset_probe     ← jump_onset_probe.py   (GBT fit, not in analyze())
    rate_fidelity        ← rate_fidelity.py       (two-corpus, not in analyze())

The OPERATIVE filter is applied by construction in every function that
operates on op-masked frames:

    jump_feas = ((im >> 7) | (im >> 6)) & 1   (ground-jump OR swim-up feasible)

This invariant is a correctness requirement documented in docs/move-head.md.
Do NOT remove or relax these filters at call sites.

``analyze(policy, source, *, segment="all")`` runs the (policy, source)-
compatible subset — currently ``jump_discrim`` (model forward) — and returns a
per-head dict suitable for the unified Phase-2 schema.  ``momentum_baseline``
and ``stream_dynamics`` require a raw cache path, not a resident source, and
are not included in ``analyze()``; call them directly with ``data_dir``.

Functions that require additional inputs (rate_fidelity needs two corpora at
different sample rates; jump_onset_probe trains a GBT) remain standalone named
functions with their own signatures.

Usage (thin-wrapper pattern)::

    from qnn.diag.move import (
        momentum_baseline,
        stream_dynamics,
        jump_discrim,
        jump_onset_probe,
        rate_fidelity,
        analyze,
    )
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from qnn.diag.move_metrics import plan_batches, ud_rewrite, unpack_move

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_EPS = 1e-12
AXES = ("fb", "lr", "ud")
JUMP_GAP = 8       # ticks of no-jump history required for an onset
ANTICIPATE = 4     # secondary label: onset within next 1..ANTICIPATE ticks
TOKEN_PROJECTILE, TOKEN_ACTOR = 0, 1  # qnn_object.h
NO_ENT_DIST = 8192.0

DWELL_BUCKETS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)   # ticks @ 20 Hz

LAGS_SEC = [0.1, 0.2, 0.3, 0.5, 1.0]   # move autocorrelation wall-clock lags
VEL_TAIL = [200.0, 400.0]               # deg/s fast-flick thresholds
BANDS = [(0.0, 3.0), (3.0, 5.0), (5.0, 10.0)]  # Hz; 5-10 unrepresentable at 10 Hz
MIN_SEG_FFT = 16                         # min kept-segment length for a periodogram

SCALAR_COLS = [
    "obs_vel", "obs_view_pitch", "obs_health", "obs_effective_armor",
    "obs_ammo_shells", "obs_ammo_nails", "obs_ammo_rockets", "obs_ammo_cells",
    "obs_attack_finished", "obs_self_movement_id", "obs_self_weapon_id",
    "obs_self_items",
]
SPATIAL_COLS = [
    "obs_spatial_clearance", "obs_spatial_dropoff", "obs_spatial_mean_dist",
    "obs_spatial_nearest_dist", "obs_spatial_openness", "obs_spatial_solid_frac",
    "obs_spatial_traversable", "obs_spatial_water_frac", "obs_spatial_lava_frac",
    "obs_spatial_slime_frac", "obs_spatial_dir",
]
ENTITY_COLS = ["obs_entity_count", "obs_entity_types", "obs_entity_rel",
               "obs_entity_vel", "obs_entity_eta", "obs_entity_facing"]
ACTION_COLS = ["act_move", "act_target_probs"]


# ---------------------------------------------------------------------------
# momentum_baseline  (move_momentum_baseline.py core)
# ---------------------------------------------------------------------------

def _mb_unpack(packed: np.ndarray) -> np.ndarray:
    """Packed move byte → (T,3) int64 class indices [fb,lr,ud]."""
    return unpack_move(packed).astype(np.int64)


def _mb_episodes(root: Path, split: str):
    """Yield (move (T,3) rewritten, keep (T,) in-dist mask) per episode."""
    manifest = json.loads((root / split / "manifest.json").read_text())
    for shard in manifest["shards"]:
        a = shard["actions"]
        move = ud_rewrite(
            _mb_unpack(np.load(root / split / a["move"], mmap_mode="r")),
            np.asarray(np.load(root / split / a["input_mask"], mmap_mode="r")).reshape(-1),
        )
        tp = np.asarray(
            np.load(root / split / a["target_probs"], mmap_mode="r"), dtype=np.float32
        )
        keep = (1.0 - tp[:, 0]) != 0.0
        start = 0
        for length in shard["episode_lengths"]:
            stop = start + int(length)
            yield move[start:stop], keep[start:stop]
            start = stop


def momentum_baseline(
    data_dir: Path,
    *,
    alpha: float = 1.0,
) -> dict:
    """Markov-1 vs marginal move-predictability baseline (pure numpy, no model).

    For each move axis (fb / lr / ud), fits a first-order Markov transition
    matrix on TRAIN within-episode transitions and evaluates the cross-entropy
    on in-distribution VAL frames (``1 - target_probs[:,0] != 0``), applying
    the ud input_mask feasibility rewrite.

    Δloglik (dll, nats) is the improvement over the human marginal base rate:
    ``marginal = 0`` by construction; ``markov1 > 0`` = pure autocorrelation
    gain with no game-state access.

    Parameters
    ----------
    data_dir:
        Root of the cache tree containing ``precomputed_train`` and
        ``precomputed_val`` sub-directories, each with a ``manifest.json``.
    alpha:
        Laplace smoothing applied to raw transition counts (default 1.0).

    Returns
    -------
    dict with keys:

    * ``markov1_mean_dll`` — mean Δloglik over axes (the headline scalar).
    * ``per_axis`` — fb/lr/ud → dll (nats).
    * ``copy_rate`` — fb/lr/ud → frame fraction where class_t == class_{t-1}.
    * ``n_scored`` — number of in-dist VAL frames scored (per axis, all equal).
    * ``_detail`` — list of per-axis records with full breakdown for table
      printing: ``{axis, h_marg, ce_markov, dll, skill, copy_rate}``.
    """
    data_dir = Path(data_dir)

    # TRAIN: per-axis transition counts (prev→curr), within-episode.
    trans = [np.full((3, 3), alpha) for _ in AXES]
    for move, _keep in _mb_episodes(data_dir, "precomputed_train"):
        if move.shape[0] < 2:
            continue
        for ax in range(3):
            prev, cur = move[:-1, ax], move[1:, ax]
            np.add.at(trans[ax], (prev, cur), 1)
    P = [t / t.sum(axis=1, keepdims=True) for t in trans]

    # VAL: score marginal vs markov1 on in-distribution frames.
    ce_markov = np.zeros(3)
    n_scored = np.zeros(3)
    copy_hits = np.zeros(3)
    hist = [np.zeros(3) for _ in AXES]
    for move, keep in _mb_episodes(data_dir, "precomputed_val"):
        if move.shape[0] < 2:
            continue
        score = keep[1:]
        prev, cur = move[:-1], move[1:]
        for ax in range(3):
            p, c = prev[score, ax], cur[score, ax]
            ce_markov[ax] += -np.log(np.clip(P[ax][p, c], _EPS, 1.0)).sum()
            n_scored[ax] += p.shape[0]
            copy_hits[ax] += (p == c).sum()
            np.add.at(hist[ax], c, 1)

    dll_axes = []
    detail = []
    for ax in range(3):
        hm = hist[ax] / max(hist[ax].sum(), _EPS)
        nz = hm > 0
        h_marg = float(-(hm[nz] * np.log(hm[nz])).sum())
        ce = float(ce_markov[ax] / max(n_scored[ax], 1.0))
        dll = h_marg - ce
        dll_axes.append(dll)
        cr = float(copy_hits[ax] / max(n_scored[ax], 1))
        skill = dll / h_marg if h_marg > _EPS else 0.0
        detail.append({
            "axis": AXES[ax],
            "h_marg": h_marg,
            "ce_markov": ce,
            "dll": dll,
            "skill": skill,
            "copy_rate": cr,
        })

    mean_dll = float(np.mean(dll_axes))
    return {
        "markov1_mean_dll": mean_dll,
        "per_axis": {AXES[i]: dll_axes[i] for i in range(3)},
        "copy_rate": {
            AXES[i]: float(copy_hits[i] / max(n_scored[i], 1)) for i in range(3)
        },
        "n_scored": int(n_scored[0]),
        "_detail": detail,
    }


# ---------------------------------------------------------------------------
# stream_dynamics  (move_stream_dynamics.py core)
# ---------------------------------------------------------------------------

def _sd_runs(classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encode: (values, lengths)."""
    if len(classes) == 0:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int64)
    change = np.flatnonzero(classes[1:] != classes[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(classes)]])
    return classes[starts], ends - starts


def _sd_hazard(
    sub_dwells: np.ndarray, sub_finals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hazard P(run ends at age | reached age), bucketed by dwell age."""
    los = np.array((0,) + DWELL_BUCKETS, dtype=np.int64)
    his = np.array(DWELL_BUCKETS + (np.iinfo(np.int64).max,), dtype=np.int64)
    exposed = np.where(sub_finals, sub_dwells - 1, sub_dwells)
    n = np.maximum(
        np.minimum(exposed[:, None], his[None, :]) - los[None, :], 0
    ).sum(axis=0)
    end_bucket = np.digitize(sub_dwells[~sub_finals], DWELL_BUCKETS, right=True)
    end = np.bincount(end_bucket, minlength=len(DWELL_BUCKETS) + 1)
    return end, n


def _sd_axis_stats(episodes: list[np.ndarray]) -> dict:
    """Compute dwell / switch / reversal / hazard stats for one move axis.

    Parameters
    ----------
    episodes:
        List of (T,) int class arrays in {0, 1, 2} (neg / none / pos).
    """
    occ = np.zeros(3, dtype=np.int64)
    switches = 0
    reversals = 0
    pairs = 0
    dwell_all: list[np.ndarray] = []
    final_flags: list[np.ndarray] = []
    run_vals: list[np.ndarray] = []
    for ep in episodes:
        if len(ep) < 2:
            continue
        occ += np.bincount(ep, minlength=3)
        d = ep[1:] != ep[:-1]
        switches += int(d.sum())
        pairs += len(d)
        rev = (np.abs(ep[1:].astype(np.int64) - ep[:-1].astype(np.int64)) == 2)
        reversals += int(rev.sum())
        vals, lens = _sd_runs(ep)
        dwell_all.append(lens)
        run_vals.append(vals)
        f = np.zeros(len(lens), dtype=bool)
        f[-1] = True
        final_flags.append(f)
    dwells = np.concatenate(dwell_all) if dwell_all else np.empty(0, dtype=np.int64)
    finals = np.concatenate(final_flags) if final_flags else np.empty(0, dtype=bool)
    rvals = np.concatenate(run_vals) if run_vals else np.empty(0, dtype=np.int8)

    haz_end, haz_n = _sd_hazard(dwells, finals)
    per_class: dict = {}
    for cls in (0, 1, 2):
        m = rvals == cls
        ce, cn = _sd_hazard(dwells[m], finals[m])
        per_class[str(cls - 1)] = {
            (f"<={DWELL_BUCKETS[i]}" if i < len(DWELL_BUCKETS) else f">{DWELL_BUCKETS[-1]}"):
                (float(ce[i] / cn[i]) if cn[i] else None)
            for i in range(len(DWELL_BUCKETS) + 1)
        }
    occ_t = max(int(occ.sum()), 1)
    return {
        "n_frames": int(occ.sum()),
        "occupancy": {
            "-1": float(occ[0] / occ_t),
            "0": float(occ[1] / occ_t),
            "+1": float(occ[2] / occ_t),
        },
        "switch_rate": float(switches / max(pairs, 1)),
        "reversal_rate_frame": float(reversals / max(pairs, 1)),
        "reversal_rate_of_switch": float(reversals / max(switches, 1)),
        "dwell": {
            "mean": float(dwells.mean()) if len(dwells) else None,
            "median": float(np.median(dwells)) if len(dwells) else None,
            "p90": float(np.percentile(dwells, 90)) if len(dwells) else None,
            "cv": float(dwells.std() / max(dwells.mean(), 1e-9)) if len(dwells) else None,
        },
        "hazard_by_dwell_age": {
            (f"<={DWELL_BUCKETS[i]}" if i < len(DWELL_BUCKETS) else f">{DWELL_BUCKETS[-1]}"):
                (float(haz_end[i] / haz_n[i]) if haz_n[i] else None)
            for i in range(len(DWELL_BUCKETS) + 1)
        },
        "hazard_by_dwell_age_per_class": per_class,
    }


def _sd_stats_for(episodes_by_axis: list[list[np.ndarray]]) -> dict:
    return {AXES[a]: _sd_axis_stats(episodes_by_axis[a]) for a in range(3)}


def _sd_load_human(cache_dir: Path) -> list[list[np.ndarray]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    by_axis: list[list[np.ndarray]] = [[], [], []]
    for shard in manifest["shards"]:
        cls = unpack_move(
            np.load(cache_dir / shard["actions"]["move"])
        ).astype(np.int8)
        s = 0
        for n in shard["episode_lengths"]:
            n = int(n)
            for a in range(3):
                by_axis[a].append(cls[s:s + n, a])
            s += n
    return by_axis


def _sd_load_actionlog(path: Path) -> list[list[np.ndarray]]:
    """Live QNN_CLIENT_ACTION_LOG JSONL: split on backward t."""
    by_axis: list[list[np.ndarray]] = [[], [], []]
    seg: list[list[int]] = []
    last_t = -1

    def flush():
        if len(seg) > 2:
            arr = np.asarray(seg, dtype=np.int8)
            for a in range(3):
                by_axis[a].append(arr[:, a])
        seg.clear()

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        t = int(rec["t"])
        if t <= last_t:
            flush()
        last_t = t
        seg.append([int(v) + 1 for v in rec["move"]])
    flush()
    return by_axis


def _sd_load_bot(npz_path: Path) -> list[list[np.ndarray]]:
    z = np.load(npz_path)
    by_axis: list[list[np.ndarray]] = [[], [], []]
    for key in sorted(z.files):
        ep = z[key]                         # (T,3) classes 0/1/2
        for a in range(3):
            by_axis[a].append(ep[:, a].astype(np.int8))
    return by_axis


def stream_dynamics(
    *,
    human_cache: Path | None = None,
    bot_npz: list[tuple[str, Path]] | None = None,
    bot_actionlog: list[tuple[str, Path]] | None = None,
) -> dict:
    """Per-axis move-stream temporal dynamics (occupancy, switch, dwell, hazard).

    Computes the full temporal characterization of the movement command stream,
    separately for each supplied source.  Sources are identified by a
    ``(label, path)`` pair.

    Parameters
    ----------
    human_cache:
        Path to a ``precomputed_val`` directory containing ``manifest.json``
        and the ``act_move`` shards.
    bot_npz:
        List of ``(label, path)`` pairs for ``move_streams_sampled.npz`` files
        written by ``qnn.eval.run`` with ``eval_log_action_streams=True``.
        Each npz has episode arrays ``(T,3)`` with classes 0/1/2.
    bot_actionlog:
        List of ``(label, path)`` pairs for live ``QNN_CLIENT_ACTION_LOG``
        JSONL files.  Segments split on backwards ``t``.

    Returns
    -------
    dict mapping source label → per-axis stats dict.  Each per-axis dict has
    keys: ``n_frames``, ``occupancy``, ``switch_rate``, ``reversal_rate_frame``,
    ``reversal_rate_of_switch``, ``dwell``, ``hazard_by_dwell_age``,
    ``hazard_by_dwell_age_per_class``.
    """
    report: dict = {}
    if human_cache is not None:
        report["human"] = _sd_stats_for(_sd_load_human(Path(human_cache)))
    for label, path in (bot_npz or []):
        report[label] = _sd_stats_for(_sd_load_bot(Path(path)))
    for label, path in (bot_actionlog or []):
        report[label] = _sd_stats_for(_sd_load_actionlog(Path(path)))
    return report


# ---------------------------------------------------------------------------
# jump_discrim  (jump_discrim.py core)
# ---------------------------------------------------------------------------

def _jd_collect_pjump(policy, source, device) -> np.ndarray:
    """Per-frame P(ud=pos) in global row order via padded-lane batches."""
    import torch
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    lengths = offs[1:] - offs[:-1]
    n_total = int(offs[-1])
    out = np.zeros(n_total, dtype=np.float32)
    obs = source.obs
    obs_keys = list(obs.keys())
    batches = plan_batches(lengths)
    print(f"[jump] {len(batches)} batches", flush=True)
    t0 = time.monotonic()
    with torch.inference_mode():
        for bi, batch in enumerate(batches):
            B = len(batch)
            Tmax = int(max(lengths[ei] for ei in batch))
            obs_seq = {
                k: torch.zeros((Tmax, B, *obs[k].shape[1:]),
                               dtype=obs[k].dtype, device=device)
                for k in obs_keys
            }
            for b, ei in enumerate(batch):
                lo, hi = int(offs[ei]), int(offs[ei + 1])
                idx = torch.arange(lo, hi, device=device, dtype=torch.long)
                for k in obs_keys:
                    obs_seq[k][: hi - lo, b] = obs[k].index_select(0, idx)
            _, logits, _, _, _ = policy._forward_tensors(obs_seq, hidden=None)
            mv = logits["move"].reshape(Tmax, B, 3, 3)
            pj = (
                torch.softmax(mv.float(), dim=-1)[..., 2, 2]
                .detach().cpu().numpy()
            )  # ud=pos
            for b, ei in enumerate(batch):
                lo, hi = int(offs[ei]), int(offs[ei + 1])
                out[lo:hi] = pj[: hi - lo, b]
            if bi % 5 == 0 or bi == len(batches) - 1:
                print(f"[jump]   {bi+1}/{len(batches)} "
                      f"({time.monotonic()-t0:.1f}s)", flush=True)
    return out


def _jd_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC, tie-aware. label is bool."""
    npos = int(label.sum())
    nneg = int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks_sorted = np.arange(1, len(score) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = ranks_sorted
    return float((ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def jump_discrim(
    policy,
    source,
    *,
    device=None,
) -> dict:
    """Jump discrimination: is P(jump) predictive of where humans jump?

    Measures whether the model's P(ud=pos) is informative about actual human
    jump events (AUC, mean-prob splits, bunny-hop probe).

    The OPERATIVE filter is applied by construction:

        jump_feas = ((im >> 7) | (im >> 6)) & 1

    Only frames where the engine would accept a jump press (ground-jump OR
    swim-up feasible) are counted as human jumps — raw ud==2 inflates the rate
    with infeasible air-presses the engine ignored.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy (eval mode).
    source:
        Resident source loaded with ``segment_mask=None`` so no-jump frames
        are retained.
    device:
        Inference device.  Defaults to ``policy.device``.

    Returns
    -------
    dict with keys: ``n_total``, ``human_jump_rate``, ``model_pjump_mean``,
    ``auc_all``, ``auc_present``, ``mean_pjump_given_jump``,
    ``mean_pjump_given_nojump``, ``pjump_percentiles``,
    ``auc_after_jump_window``, ``n_after_jump_window``,
    ``human_interjump_interval``.
    """
    import torch

    if device is None:
        device = torch.device(policy.device)
    else:
        device = torch.device(device)

    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    n_total = int(offs[-1])
    print(f"[jump] {n_total:,} rows, {len(offs)-1} episodes", flush=True)

    human = (
        source.actions["move"].detach().cpu().numpy().astype(np.int64)
    )  # (n_total, 3)

    # OPERATIVE-FRAME FILTER: a jump press is only honoured by the engine when
    # feasible (input_mask bit 7 ground-jump OR bit 6 swim-up). Raw ud==2
    # counts infeasible air-jump presses the engine ignored, inflating the
    # human jump rate and every discrimination conditional.
    im = (
        source.actions["input_mask"]
        .detach().cpu().numpy()
        .astype(np.uint8)
        .reshape(-1)
    )
    jump_feas = (((im >> 7) & 1) | ((im >> 6) & 1)).astype(bool)
    human_jump = (human[:, 2] == 2) & jump_feas                      # OPERATIVE filter

    p_target = (
        1.0 - source.actions["target_probs"]
        .detach().cpu().numpy()[:, 0]
        .astype(np.float64)
    )
    present = p_target >= 0.5

    pj = _jd_collect_pjump(policy, source, device)

    res: dict[str, Any] = {
        "n_total": n_total,
        "human_jump_rate": float(human_jump.mean()),
        "model_pjump_mean": float(pj.mean()),
        "auc_all": _jd_auc(pj, human_jump),
        "auc_present": (
            _jd_auc(pj[present], human_jump[present])
            if present.any() else float("nan")
        ),
        "mean_pjump_given_jump": (
            float(pj[human_jump].mean()) if human_jump.any() else float("nan")
        ),
        "mean_pjump_given_nojump": float(pj[~human_jump].mean()),
        "pjump_percentiles": {
            p: float(np.percentile(pj, p)) for p in (50, 90, 95, 99)
        },
    }

    # Bunny-hop probe: among frames 1..8 ticks AFTER a human jump (within
    # episode), does P(jump) anticipate the next hop?
    after_jump = np.zeros(n_total, dtype=bool)
    for e in range(len(offs) - 1):
        lo, hi = int(offs[e]), int(offs[e + 1])
        jl = human_jump[lo:hi]
        if jl.any():
            jidx = np.where(jl)[0]
            for k in range(1, 9):
                nxt = jidx + k
                nxt = nxt[nxt < (hi - lo)]
                after_jump[lo + nxt] = True
    res["auc_after_jump_window"] = (
        _jd_auc(pj[after_jump], human_jump[after_jump])
        if after_jump.any() else float("nan")
    )
    res["n_after_jump_window"] = int(after_jump.sum())

    # Human inter-jump interval distribution (within episode).
    intervals = []
    for e in range(len(offs) - 1):
        lo, hi = int(offs[e]), int(offs[e + 1])
        jidx = np.where(human_jump[lo:hi])[0]
        if len(jidx) >= 2:
            intervals.extend(np.diff(jidx).tolist())
    intervals_arr = (
        np.asarray(intervals, dtype=np.float64)
        if intervals else np.array([np.nan])
    )
    res["human_interjump_interval"] = {
        "n": int(len(intervals_arr)),
        "mean": float(np.nanmean(intervals_arr)),
        "median": float(np.nanmedian(intervals_arr)),
        "frac_le_3": float(np.mean(intervals_arr <= 3)),
        "frac_le_6": float(np.mean(intervals_arr <= 6)),
    }
    return res


# ---------------------------------------------------------------------------
# rate_fidelity  (rate_fidelity.py core — two-corpus, not in analyze())
# ---------------------------------------------------------------------------

def _rf_keep_segments(keep: np.ndarray):
    """Yield (start, stop) for each contiguous True run in a boolean mask."""
    k = np.asarray(keep, bool)
    if not k.any():
        return
    edges = np.flatnonzero(np.diff(np.r_[0, k.view(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        yield int(s), int(e)


class _RfACF:
    """Streaming Pearson correlation between an axis and its lagged self."""

    __slots__ = ("n", "sx", "sy", "sxy", "sxx", "syy")

    def __init__(self):
        self.n = self.sx = self.sy = self.sxy = self.sxx = self.syy = 0.0

    def add(self, x: np.ndarray, y: np.ndarray):
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxy += (x * y).sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()

    def r(self) -> float:
        n = self.n
        if n < 2:
            return float("nan")
        cov = n * self.sxy - self.sx * self.sy
        vx = n * self.sxx - self.sx ** 2
        vy = n * self.syy - self.sy ** 2
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def _rf_analyze(episodes, hz: float) -> dict:
    """Compute move-ACF + look velocity/spectral stats for one stream."""
    lags = [max(1, round(s * hz)) for s in LAGS_SEC]
    acf = {ax: {lg: _RfACF() for lg in lags} for ax in ("fb", "lr")}
    vel: list[np.ndarray] = []
    band_pow = np.zeros(len(BANDS))
    tot_pow = 0.0
    for ep in episodes:
        move, turn, keep = ep["move"], ep["turn"], ep["keep"]
        for s, e in _rf_keep_segments(keep):
            if e - s < 2:
                continue
            for ai, ax in enumerate(("fb", "lr")):
                a = move[s:e, ai].astype(np.float64) - 1.0   # {-1,0,1}
                for lg in lags:
                    if a.size > lg:
                        acf[ax][lg].add(a[:-lg], a[lg:])
            seg = turn[s:e].astype(np.float64)
            vel.append(seg * hz)                              # deg/s
            if seg.size >= MIN_SEG_FFT:
                z = seg - seg.mean()
                ps = np.abs(np.fft.rfft(z)) ** 2
                fr = np.fft.rfftfreq(z.size, d=1.0 / hz)
                tot_pow += ps.sum()
                for bi, (lo, hi) in enumerate(BANDS):
                    band_pow[bi] += ps[(fr >= lo) & (fr < hi)].sum()
    v = np.concatenate(vel) if vel else np.zeros(1)
    return {
        "hz": hz,
        "move_acf": {
            ax: {
                round(LAGS_SEC[i], 2): acf[ax][lags[i]].r()
                for i in range(len(lags))
            }
            for ax in ("fb", "lr")
        },
        "look_vel": {
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            **{f"ge{int(t)}": float((v >= t).mean()) for t in VEL_TAIL},
        },
        "look_band_frac": [float(x) for x in np.round(band_pow / max(tot_pow, 1e-9), 4)],
    }


def _rf_bot_episodes(npz_path: Path):
    z = np.load(npz_path)
    offs = z["episode_offsets"]
    fb = z["fb"].astype(np.int64)
    lr = z["lr"].astype(np.int64)
    turn = z["turn_deg"].astype(np.float64)
    keep = z["keep"].astype(bool)
    for i in range(len(offs) - 1):
        sl = slice(int(offs[i]), int(offs[i + 1]))
        yield {
            "move": np.stack([fb[sl], lr[sl], np.zeros_like(fb[sl])], axis=1),
            "turn": turn[sl],
            "keep": keep[sl],
        }


def rate_fidelity(
    corpus_20: Path,
    corpus_10: Path,
    *,
    eval_dir: Path,
    split: str = "precomputed_val",
    human_episodes_fn=None,
) -> dict:
    """Rate-aware command-stream fidelity (move ACF + look velocity/spectral).

    Computes the principled 10 Hz vs 20 Hz rate-fidelity instrument
    (references.md §1): move command-stream autocorrelation at matched
    wall-clock lags, and look turn-velocity / spectral-power by band.

    The human reference is loaded at BOTH rates from the respective corpora;
    each bot stream is routed to its rate-matched human by ``tick_hz``.

    Parameters
    ----------
    corpus_20:
        Root of the 20 Hz corpus (e.g. ``artifacts/collect/qwd``).
    corpus_10:
        Root of the 10 Hz corpus (e.g. ``artifacts/collect/qwd_10hz``).
    eval_dir:
        Directory containing ``rc_*_streams.npz`` bot streams
        (written by ``rc_3way_frikbot_eval.py``).
    split:
        Sub-directory under each corpus root (default ``"precomputed_val"``).
    human_episodes_fn:
        Callable ``(root, split) → iterable[ep_dict]`` that yields per-episode
        dicts with keys ``move (T,3)``, ``turn (T,)``, ``keep (T,)``.
        Defaults to the ``_episodes`` function from
        ``scripts/analysis/humanlikeness_human_reference``.

    Returns
    -------
    dict with keys ``_meta``, ``human`` (``"20hz"`` / ``"10hz"``), ``bots``
    (name → result dict).
    """
    eval_dir = Path(eval_dir)
    corpus_20 = Path(corpus_20)
    corpus_10 = Path(corpus_10)

    if human_episodes_fn is None:
        import sys
        import importlib
        # resolve humanlikeness_human_reference relative to the scripts dir
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "analysis"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        mod = importlib.import_module("humanlikeness_human_reference")
        human_episodes_fn = mod._episodes

    streams = sorted(eval_dir.glob("rc_*_streams.npz"))
    if not streams:
        raise FileNotFoundError(f"no rc_*_streams.npz under {eval_dir}")
    bots = {
        s.stem.replace("rc_", "").replace("_streams", ""): s for s in streams
    }

    human = {
        20.0: _rf_analyze(human_episodes_fn(corpus_20, split), 20.0),
        10.0: _rf_analyze(human_episodes_fn(corpus_10, split), 10.0),
    }
    rcs = {}
    for name, path in bots.items():
        hz = float(np.load(path)["tick_hz"][0])
        rcs[name] = _rf_analyze(_rf_bot_episodes(path), hz)

    return {
        "_meta": {
            "instrument": (
                "references.md §1 rate-aware: move command-stream "
                "autocorrelation + look velocity/spectral, rate-matched human"
            ),
            "lags_sec": LAGS_SEC,
            "vel_tail_degps": VEL_TAIL,
            "bands_hz": BANDS,
            "corpus_20": str(corpus_20),
            "corpus_10": str(corpus_10),
            "split": split,
        },
        "human": {f"{int(k)}hz": v for k, v in human.items()},
        "bots": rcs,
    }


# ---------------------------------------------------------------------------
# jump_onset_probe  (jump_onset_probe.py core — trains GBT, not in analyze())
# ---------------------------------------------------------------------------

def _jp_ep_shift(x: np.ndarray, k: int, offs: np.ndarray) -> np.ndarray:
    """x shifted by +k ticks into the past, episode-respecting (edge-padded)."""
    out = np.empty_like(x)
    out[k:] = x[:-k]
    out[:k] = x[0]
    for e in range(len(offs) - 1):
        lo, hi = int(offs[e]), min(int(offs[e]) + k, int(offs[e + 1]))
        out[lo:hi] = x[lo]
    return out


def _jp_build_labels(
    move_packed: np.ndarray, offs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """onset / drop / anticipation masks, episode-respecting."""
    arr = np.asarray(move_packed, dtype=np.uint8)
    jump = (((arr >> 6) & 1) | ((arr >> 7) & 1)).astype(bool)
    jump_btn = ((arr >> 7) & 1).astype(bool)
    n = len(jump)
    recent = np.zeros(n, dtype=bool)
    for k in range(1, JUMP_GAP + 1):
        recent |= _jp_ep_shift(jump, k, offs)
    onset = jump & ~recent
    onset_btn = jump_btn & ~recent
    drop = recent & ~onset
    upcoming = np.zeros(n, dtype=bool)
    ep_of = np.searchsorted(offs, np.arange(n), side="right") - 1
    for k in range(1, ANTICIPATE + 1):
        tmp = np.zeros(n, dtype=bool)
        tmp[:-k] = onset[k:]
        tmp[ep_of != np.roll(ep_of, -k)] = False
        upcoming |= tmp
    return jump, onset, onset_btn, drop, upcoming


def _jp_entity_features(cols: dict) -> tuple[np.ndarray, list[str]]:
    """Nearest actor / projectile aggregates from ragged entity arrays."""
    cnt = cols["obs_entity_count"].astype(np.int64).reshape(-1)
    starts = np.concatenate([[0], np.cumsum(cnt)])
    types = cols["obs_entity_types"]
    rel = cols["obs_entity_rel"].astype(np.float32)
    vel = cols["obs_entity_vel"].astype(np.float32)
    eta = cols["obs_entity_eta"].astype(np.float32)
    facing = cols["obs_entity_facing"].astype(np.float32)
    dist = np.linalg.norm(rel, axis=1)

    n = len(cnt)
    frame_idx = np.repeat(np.arange(n, dtype=np.int64), cnt)
    feats = np.full((n, 12), 0.0, dtype=np.float32)
    feats[:, 1] = NO_ENT_DIST
    feats[:, 9] = NO_ENT_DIST
    for tid, c_n, c_d, detail in (
        (TOKEN_ACTOR, 0, 1, True),
        (TOKEN_PROJECTILE, 8, 9, False),
    ):
        m = np.where(types == tid)[0]
        fi = frame_idx[m]
        feats[:, c_n] = np.bincount(fi, minlength=n)
        order = np.lexsort((dist[m], fi))
        frames, first = np.unique(fi[order], return_index=True)
        j = m[order][first]
        feats[frames, c_d] = dist[j]
        if detail:
            feats[frames, 2:5] = rel[j]
            feats[frames, 5:8] = vel[j]
            feats[frames, 10] = eta[j]
            feats[frames, 11] = facing[j]
    names = [
        "n_actors", "actor_dist", "actor_rel_x", "actor_rel_y", "actor_rel_z",
        "actor_vel_x", "actor_vel_y", "actor_vel_z",
        "n_projectiles", "proj_dist", "actor_eta", "actor_facing",
    ]
    return feats, names


def _jp_feature_groups(
    cols: dict, offs: np.ndarray, keep_masks=None
) -> tuple[dict, dict]:
    """Build feature groups; keep_masks from TRAIN applies to VAL for parity."""
    f32 = lambda k: cols[k].astype(np.float32)
    vel = f32("obs_vel")
    vel2, vel5 = _jp_ep_shift(vel, 2, offs), _jp_ep_shift(vel, 5, offs)
    motion = np.column_stack([
        vel,
        np.linalg.norm(vel[:, :2], axis=1),
        vel2, vel5,
        vel - vel2,
        f32("obs_view_pitch").reshape(len(vel), -1),
    ])
    motion_names = (
        ["vel_x", "vel_y", "vel_z", "speed_xy"]
        + [f"vel_{a}_lag{k}" for k in (2, 5) for a in "xyz"]
        + ["acc_x", "acc_y", "acc_z", "view_pitch"]
    )

    state_keys = [
        "obs_health", "obs_effective_armor", "obs_ammo_shells",
        "obs_ammo_nails", "obs_ammo_rockets", "obs_ammo_cells",
        "obs_attack_finished", "obs_self_movement_id",
        "obs_self_weapon_id", "obs_self_items",
    ]
    state = np.column_stack([f32(k).reshape(len(vel), -1) for k in state_keys])
    state_names = [k.removeprefix("obs_") for k in state_keys]

    computed_masks: dict = {}
    sp_parts, sp_names = [], []
    for k in SPATIAL_COLS:
        a = f32(k).reshape(len(vel), -1)
        if keep_masks is None:
            keep = a.std(axis=0) > 1e-6
            computed_masks[k] = keep
        else:
            keep = keep_masks[k]
        sp_parts.append(a[:, keep])
        base = k.removeprefix("obs_spatial_")
        sp_names += [f"sp_{base}_{i}" for i in np.where(keep)[0]]
    spatial = np.column_stack(sp_parts)

    ent, ent_names = _jp_entity_features(cols)
    groups = {
        "motion": (motion, motion_names),
        "state": (state, state_names),
        "spatial": (spatial, sp_names),
        "entity": (ent, ent_names),
    }
    return groups, (keep_masks if keep_masks is not None else computed_masks)


def _jp_load_shards(cache: Path, n_shards: int | None) -> tuple[dict, np.ndarray]:
    """Concatenate needed columns; return (cols, episode_offsets)."""
    man = json.loads((cache / "manifest.json").read_text())
    shards = man["shards"] if n_shards is None else man["shards"][:n_shards]
    cols: dict[str, list[np.ndarray]] = {}
    ep_lens: list[int] = []
    for si, sh in enumerate(shards):
        assert sum(sh["episode_lengths"]) == sh["rows"], (
            f"shard {si} not episode-aligned"
        )
        ep_lens.extend(sh["episode_lengths"])
        for name in SCALAR_COLS + SPATIAL_COLS + ENTITY_COLS + ACTION_COLS:
            arr = np.load(cache / f"shard{si:06d}_{name}.npy")
            cols.setdefault(name, []).append(arr)
    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    offs = np.concatenate([[0], np.cumsum(ep_lens)]).astype(np.int64)
    return out, offs


def _jp_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC, tie-aware."""
    npos = int(label.sum())
    nneg = int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks_sorted = np.arange(1, len(score) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = ranks_sorted
    return float((ranks[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def jump_onset_probe(
    data_dir: Path,
    *,
    train_shards: int = 12,
    val_shards: int | None = None,
) -> dict:
    """GBT-based jump-onset predictability from raw obs features (no model).

    Fits a ``HistGradientBoostingClassifier`` directly on per-frame obs
    features (motion / state / spatial / entity groups) against human
    jump-onset labels.  AUC > 0.6 implies the features carry the signal;
    near-chance (AUC ≈ 0.5) implies a perception/feature gap no retrain can
    fix.

    The feature ladder ``motion → +state → +spatial → +entity → all`` isolates
    which group carries the signal, evaluated on three label types:
    ``onset`` (cluster start), ``onset_btn`` (button-only onset), and
    ``upcoming`` (onset within next 1..4 ticks).

    Parameters
    ----------
    data_dir:
        Root cache directory with ``precomputed_train`` and ``precomputed_val``
        sub-directories.
    train_shards:
        Number of TRAIN shards to fit on (default 12).
    val_shards:
        Limit VAL shards (``None`` = all; pass ``1`` for smoke runs).

    Returns
    -------
    dict with keys: ``train_shards``, ``jump_gap``, ``anticipate``,
    ``n_train``, ``n_val``, ``train_onset_rate``, ``val_onset_rate``,
    ``train_jump_rate``, ``val_jump_rate``, ``groups``
    (label_key/feature_combo → AUC dict).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    data_dir = Path(data_dir)
    t0 = time.monotonic()
    res: dict[str, Any] = {
        "train_shards": train_shards,
        "jump_gap": JUMP_GAP,
        "anticipate": ANTICIPATE,
        "groups": {},
    }
    data: dict[str, dict] = {}
    for split, n in (("train", train_shards), ("val", val_shards)):
        cols, offs = _jp_load_shards(
            data_dir / f"precomputed_{split}", n
        )
        jump, onset, onset_btn, drop, upcoming = _jp_build_labels(
            cols["act_move"], offs
        )
        groups, sp_masks = _jp_feature_groups(
            cols, offs,
            None if split == "train" else data["train"]["sp_masks"],
        )
        p_target = 1.0 - cols["act_target_probs"][:, 0].astype(np.float64)
        dry = (
            cols["obs_spatial_water_frac"].reshape(len(jump), -1) == 0
        ).all(axis=1)
        data[split] = dict(
            groups=groups, sp_masks=sp_masks, onset=onset,
            onset_btn=onset_btn, drop=drop, upcoming=upcoming,
            present=p_target >= 0.5, dry=dry, jump=jump,
        )
        res[f"n_{split}"] = len(jump)
        res[f"{split}_onset_rate"] = float(onset.mean())
        res[f"{split}_jump_rate"] = float(jump.mean())
        print(
            f"[probe] {split}: {len(jump):,} rows, onset {onset.mean():.4%}, "
            f"jump {jump.mean():.4%} ({time.monotonic()-t0:.0f}s)",
            flush=True,
        )

    ladder = [
        ("motion", ["motion"]),
        ("motion+state", ["motion", "state"]),
        ("motion+spatial", ["motion", "spatial"]),
        ("motion+entity", ["motion", "entity"]),
        ("all", ["motion", "state", "spatial", "entity"]),
    ]

    for label_key in ("onset", "onset_btn", "upcoming"):
        for name, parts in ladder:
            Xtr = np.column_stack([data["train"]["groups"][p][0] for p in parts])
            Xva = np.column_stack([data["val"]["groups"][p][0] for p in parts])
            fnames = sum((data["train"]["groups"][p][1] for p in parts), [])
            keep_tr = ~data["train"]["drop"]
            keep_va = ~data["val"]["drop"]
            if label_key == "upcoming":
                keep_tr &= ~data["train"]["onset"]
                keep_va &= ~data["val"]["onset"]
            ytr = data["train"][label_key][keep_tr]
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                min_samples_leaf=200, early_stopping=True, random_state=17,
            )
            clf.fit(Xtr[keep_tr], ytr)
            pv = clf.predict_proba(Xva[keep_va])[:, 1].astype(np.float64)
            yva = data["val"][label_key][keep_va]
            pres = data["val"]["present"][keep_va]
            dry = data["val"]["dry"][keep_va]
            entry: dict[str, Any] = {
                "auc": _jp_auc(pv, yva),
                "auc_target_present": _jp_auc(pv[pres], yva[pres]),
                "auc_target_absent": _jp_auc(pv[~pres], yva[~pres]),
                "auc_dry": _jp_auc(pv[dry], yva[dry]),
                "n_pos_val": int(yva.sum()),
                "n_features": Xtr.shape[1],
            }
            if name == "all":
                entry["feature_names"] = fnames
            res["groups"][f"{label_key}/{name}"] = entry
            print(
                f"[probe] {label_key:9s} {name:15s} AUC {entry['auc']:.3f} "
                f"(present {entry['auc_target_present']:.3f} / absent "
                f"{entry['auc_target_absent']:.3f} / dry {entry['auc_dry']:.3f}) "
                f"[{time.monotonic()-t0:.0f}s]",
                flush=True,
            )

    return res


# ---------------------------------------------------------------------------
# analyze  — (policy, source)-compatible entry point
# ---------------------------------------------------------------------------

def analyze(
    policy,
    source,
    *,
    segment: str = "all",
    device=None,
) -> dict:
    """Run the (policy, source)-compatible move-head analysis functions.

    Calls ``jump_discrim`` (requires a model forward) and returns a per-head
    dict.  ``momentum_baseline`` and ``stream_dynamics`` require the raw cache
    path (not a resident source) and are not called here; use them directly
    with ``data_dir``.  ``rate_fidelity`` requires two corpora; call it
    directly.  ``jump_onset_probe`` trains a GBT; call it directly.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy in eval mode.
    source:
        Resident source (``segment_mask=None`` recommended so all frames,
        including no-jump frames, are present for jump discrimination).
    segment:
        Metadata tag (e.g. ``"all"`` or ``"engaged"``).
    device:
        Inference device.  Defaults to ``policy.device``.

    Returns
    -------
    dict with keys matching the Phase-2 per-head schema:
    ``segment``, ``jump_discrim``.
    """
    jd = jump_discrim(policy, source, device=device)
    return {
        "segment": segment,
        "jump_discrim": jd,
    }
