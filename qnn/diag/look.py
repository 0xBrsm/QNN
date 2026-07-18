"""Look-head analysis functions — importable core compute for the look slice.

Each public function corresponds to one canonical analysis script:

    look_prior_fit          ← look_prior_fit.py
    look_prior_explore4     ← look_prior_explore4.py
    look_history_attention  ← look_history_attention.py
    look_horizon_ceiling    ← look_horizon_ceiling.py
    look_metric_references  ← look_metric_references.py
    look_target_intersection← look_target_intersection.py
    aim_point_z_offset      ← aim_point_z_offset.py
    look_ground_spin        ← look_ground_spin.py  (model-based; see note)
    look_aim_prior_decode   ← look_aim_prior_decode.py  (model-based)

The look head does NOT apply an operative-frame filter (no input_mask gating).
Do NOT add one.

``analyze(policy, source) -> dict`` covers only the resident-source-compatible
metrics (currently none — all look analyses are cache-based or have additional
requirements).  Heterogeneous scripts with their own signatures are exposed as
named module functions.

Usage (thin-wrapper pattern)::

    from qnn.diag.look import (
        look_prior_fit,
        look_prior_explore4,
        look_history_attention,
        look_horizon_ceiling,
        look_metric_references,
        look_target_intersection,
        aim_point_z_offset,
        look_ground_spin,
        look_aim_prior_decode,
        analyze,
    )

Note on ``look_ground_spin``:
    The original script contained an import bug (``from qnn.model.bench import
    HEADS`` instead of ``from qnn.model.bench.heads import HEADS``) and was also
    missing the ``install_polar_grid`` call required before accessing
    ``MAG_CENTERS``/``DIR_CENTERS``.  The folded version here fixes both.  The
    before/after equivalence gate cannot be satisfied because the original script
    fails to import; the folded function is the corrected canonical form.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qnn.bc.look_metrics import tangent_logmap_np
from qnn.diag.look_data import load_pooled_look

__all__ = [
    "look_prior_fit",
    "look_prior_explore4",
    "look_history_attention",
    "look_horizon_ceiling",
    "look_metric_references",
    "look_target_intersection",
    "aim_point_z_offset",
    "look_ground_spin",
    "look_aim_prior_decode",
    "analyze",
]

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(n > _EPS, v / np.maximum(n, _EPS), 0.0)


# ---------------------------------------------------------------------------
# look_prior_fit  (look_prior_fit.py core)
# ---------------------------------------------------------------------------

def look_prior_fit(cache: Path) -> dict:
    """Empirically fit a look PRIOR as a linear combination of geometric features.

    Per frame, in the view-relative tangent space (log-map at forward=(1,0,0),
    ||z|| = turn angle):

      z      = logmap(look_label)                      TARGET (the actual turn)
      e_tgt  = logmap(normalize(Σ target_probs·rel))   direction to the GT target
      lead   = logmap(tangential component of Σ target_probs·vel)
      p      = logmap(look_lag1[t-1])                  momentum (prev ACTION — reference)

    Reports OLS gains and look_r2 (variance explained vs no-turn baseline) for
    single features and combinations.

    Parameters
    ----------
    cache:
        Precomputed val cache directory (contains ``manifest.json``).

    Returns
    -------
    dict with keys: ``n_frames``, ``e_tgt_gain1_r2``, ``fits`` (list).
    """
    manifest = json.loads((cache / "manifest.json").read_text())

    Z, ETGT, LEAD, PREV, VALID = [], [], [], [], []
    for shard in manifest["shards"]:
        d = load_pooled_look(cache, shard)
        pooled_rel, pooled_vel, look, prev = (
            d["pooled_rel"], d["pooled_vel"], d["look"], d["look_lag1"]
        )
        base_look = _unit(pooled_rel)                           # dir to target
        proj = (pooled_vel * base_look).sum(-1, keepdims=True) * base_look
        lead_dir = _unit(pooled_vel - proj)

        Z.append(tangent_logmap_np(_unit(look)))
        ETGT.append(tangent_logmap_np(base_look))
        LEAD.append(tangent_logmap_np(lead_dir))
        PREV.append(tangent_logmap_np(_unit(prev)))
        VALID.append(
            (np.linalg.norm(look, axis=-1) > _EPS)
            & (np.linalg.norm(pooled_rel, axis=-1) > _EPS)
        )

    Z = np.concatenate(Z); ETGT = np.concatenate(ETGT)
    LEAD = np.concatenate(LEAD); PREV = np.concatenate(PREV)
    VALID = np.concatenate(VALID)
    Z, ETGT, LEAD, PREV = Z[VALID], ETGT[VALID], LEAD[VALID], PREV[VALID]

    def stack(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1)

    y = stack(Z)
    feats = {"e_tgt": ETGT, "lead": LEAD, "prev": PREV}
    ss_tot_zero = float((y * y).sum())  # no-turn baseline == look_r2 scale

    def fit(names: list[str]) -> tuple[dict, float]:
        X = np.stack([stack(feats[k]) for k in names], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2_zero = 1.0 - float((resid * resid).sum()) / ss_tot_zero
        return {k: round(float(b), 4) for k, b in zip(names, beta)}, round(r2_zero, 4)

    out: dict[str, Any] = {"n_frames": int(VALID.sum()), "fits": []}
    # gain=1 sanity (full snap to target): should be NEGATIVE (overshoot).
    snap_resid = y - stack(ETGT)
    out["e_tgt_gain1_r2"] = round(
        1.0 - float((snap_resid * snap_resid).sum()) / ss_tot_zero, 4
    )
    for names in (
        ["e_tgt"],
        ["lead"],
        ["prev"],
        ["e_tgt", "lead"],
        ["e_tgt", "lead", "prev"],
    ):
        gains, r2 = fit(names)
        out["fits"].append({"features": names, "gains": gains, "look_r2": r2})

    return out


# ---------------------------------------------------------------------------
# look_prior_explore4  (look_prior_explore4.py core)
# ---------------------------------------------------------------------------

def _shift(a: np.ndarray, epl: list[int], k: int) -> np.ndarray:
    """Shift array by k frames within episodes; zero-fill at boundaries."""
    out = np.zeros_like(a)
    s = 0
    for n in epl:
        n = int(n)
        if k >= 0 and n > k:
            out[s + k:s + n] = a[s:s + n - k]
        elif k < 0 and n > -k:
            out[s:s + n + k] = a[s - k:s + n]
        s += n
    return out


def look_prior_explore4(cache: Path) -> dict:
    """Round-4 look-prior exploration: pursuit sweep at small velocity scales.

    Sweeps pursuit at SMALL vel scales (0.02..1.0) and fits combinations of
    prev1 + lagged-target + pursuit. Target-frames only, correct col=j+1
    alignment.

    Parameters
    ----------
    cache:
        Precomputed val cache directory.

    Returns
    -------
    dict with keys: ``n_frames``, ``pursuit_small_scale`` (list), ``combos`` (list).
    """
    _DEG = 180.0 / np.pi
    manifest = json.loads((cache / "manifest.json").read_text())
    SC = [0.02, 0.05, 0.1, 0.3, 1.0]
    cols: dict[str, list] = {k: [] for k in ("z", "prev1", "etgt", "etgt_l2")}
    for c in SC:
        cols[f"purs{c}"] = []
    V = []
    for sh in manifest["shards"]:
        d = load_pooled_look(cache, sh)
        pr, pv, look = d["pooled_rel"], d["pooled_vel"], d["look"]
        epl = [int(n) for n in d["episode_lengths"]]
        tgt = d["target_present"]
        base = _unit(pr)
        etgt = tangent_logmap_np(base)
        z = tangent_logmap_np(_unit(look))
        cols["z"].append(z)
        cols["prev1"].append(tangent_logmap_np(_unit(_shift(look, epl, 1))))
        cols["etgt"].append(etgt)
        cols["etgt_l2"].append(_shift(etgt, epl, 2))
        for c in SC:
            cols[f"purs{c}"].append(tangent_logmap_np(_unit(pr + c * pv)) - etgt)
        V.append(
            (np.linalg.norm(look, -1) > _EPS)
            & (np.linalg.norm(pr, -1) > _EPS)
            & (tgt > 0.5)
        )

    A = {k: np.concatenate(v) for k, v in cols.items()}
    v = np.concatenate(V)
    for k in A:
        A[k] = A[k][v]
    y = A["z"].reshape(-1)
    ss = float((y * y).sum())

    def fit(names: list[str]) -> tuple[dict, float]:
        X = np.stack([A[n].reshape(-1) for n in names], 1)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        return (
            {n: round(float(x), 4) for n, x in zip(names, b)},
            round(1 - float((r * r).sum()) / ss, 4),
        )

    big = np.linalg.norm(A["z"], -1) * _DEG > 1.0
    zu = _unit(A["z"][big])
    out: dict[str, Any] = {
        "n_frames": int(v.sum()),
        "pursuit_small_scale": [],
        "combos": [],
    }
    for c in SC:
        g, r = fit([f"purs{c}"])
        out["pursuit_small_scale"].append({
            "scale": c,
            "gain": g[f"purs{c}"],
            "r2": r,
            "align": round(
                float((zu * _unit(A[f"purs{c}"][big])).sum(-1).mean()), 4
            ),
        })
    best_purs = max(
        SC, key=lambda c: out["pursuit_small_scale"][SC.index(c)]["r2"]
    )
    for names in (
        ["prev1"],
        ["etgt_l2"],
        ["prev1", "etgt"],
        ["prev1", "etgt_l2"],
        ["prev1", "etgt_l2", f"purs{best_purs}"],
    ):
        g, r = fit(names)
        out["combos"].append({"features": names, "gains": g, "look_r2": r})
    return out


# ---------------------------------------------------------------------------
# look_history_attention  (look_history_attention.py core)
# ---------------------------------------------------------------------------

#: Tap-depth values for the history-attention sweep.
HISTORY_XS = (1, 2, 3, 4, 6, 8, 12, 16)
HISTORY_X_MAX = max(HISTORY_XS)
HISTORY_CERT = [("all", 0.0, 1.01), ("p>=0.8", 0.8, 1.01)]


def _ha_shard(cache: Path, shard: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Load one shard for look_history_attention: pooled_rel, look, max_p, eps."""
    obs, act = shard["obs"], shard["actions"]
    rel = np.load(cache / obs["entity_rel"]).astype(np.float64)
    count = np.load(cache / obs["entity_count"]).astype(np.int64)
    look = np.load(cache / act["look"]).astype(np.float64)
    tp = np.load(cache / act["target_probs"]).astype(np.float64)
    F = count.shape[0]
    starts = np.zeros(F, dtype=np.int64)
    np.cumsum(count[:-1], out=starts[1:])
    frame_id = np.repeat(np.arange(F), count)
    within = np.arange(rel.shape[0]) - starts[frame_id]
    w = tp[frame_id, (within + 1).clip(max=tp.shape[1] - 1)]
    pooled_rel = np.zeros((F, 3))
    np.add.at(pooled_rel, frame_id, w[:, None] * rel)
    max_p = tp[:, 1:].max(axis=1) if tp.shape[1] > 1 else np.zeros(F)
    return pooled_rel, look, max_p, [int(n) for n in shard["episode_lengths"]]


def look_history_attention(cache: Path) -> dict:
    """OLS attention over X previous looks — linear attention ceiling vs depth.

    Fits a proper linear attention (full per-component 2×2 map per tap) for
    each depth X in ``HISTORY_XS``, with and without the grounded e_tgt feature.
    All X share the same valid frame set (requires X_MAX history).

    Parameters
    ----------
    cache:
        Precomputed val cache directory.

    Returns
    -------
    dict with keys: ``cache``, ``x_max``, ``by_certainty`` (keyed by certainty bucket).
    """
    manifest = json.loads((cache / "manifest.json").read_text())
    Z, ETGT, MAXP, HIST, VALID = [], [], [], [], []
    for sh in manifest["shards"]:
        prel, look, max_p, eps = _ha_shard(cache, sh)
        z = tangent_logmap_np(_unit(look))              # (F,2)
        F = z.shape[0]
        hist = np.zeros((F, HISTORY_X_MAX, 2))
        valid = np.zeros(F, dtype=bool)
        s = 0
        for n in eps:
            for i in range(1, HISTORY_X_MAX + 1):
                if n > i:
                    hist[s + i:s + n, i - 1] = z[s:s + n - i]
            if n > HISTORY_X_MAX:
                valid[s + HISTORY_X_MAX:s + n] = True
            s += n
        Z.append(z)
        ETGT.append(tangent_logmap_np(_unit(prel)))
        MAXP.append(max_p)
        HIST.append(hist)
        VALID.append(valid)

    Z = np.concatenate(Z)
    ETGT = np.concatenate(ETGT)
    MAXP = np.concatenate(MAXP)
    HIST = np.concatenate(HIST)
    VALID = np.concatenate(VALID)

    def fit(mask: np.ndarray, hist_x: int, with_etgt: bool) -> float | None:
        m = mask & VALID
        cols = [HIST[m, i, :] for i in range(hist_x)]
        if with_etgt:
            cols.append(ETGT[m])
        if not cols:
            return None
        X = np.concatenate(cols, axis=1)
        Y = Z[m]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        return round(
            1.0 - float((resid * resid).sum()) / max(float((Y * Y).sum()), _EPS),
            4,
        )

    out: dict[str, Any] = {
        "cache": str(cache),
        "x_max": HISTORY_X_MAX,
        "by_certainty": {},
    }
    for bn, lo, hi in HISTORY_CERT:
        cm = (MAXP >= lo) & (MAXP < hi)
        rows = {
            "n": int((cm & VALID).sum()),
            "e_tgt_only": fit(cm, 0, True),
            "hist": {str(x): fit(cm, x, False) for x in HISTORY_XS},
            "hist+e_tgt": {str(x): fit(cm, x, True) for x in HISTORY_XS},
        }
        out["by_certainty"][bn] = rows
    return out


# ---------------------------------------------------------------------------
# look_horizon_ceiling  (look_horizon_ceiling.py core)
# ---------------------------------------------------------------------------

_HORIZON_KS = (1, 2, 4, 8, 16)
_HORIZON_CERT_BUCKETS = [
    ("all", 0.0, 1.01),
    ("p<0.5", 0.0, 0.5),
    ("0.5-0.8", 0.5, 0.8),
    ("p>=0.8", 0.8, 1.01),
]


def _hc_shard(
    cache: Path, shard: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Load pooled_rel, pooled_vel, look, max_p, eps for look_horizon_ceiling."""
    obs, act = shard["obs"], shard["actions"]
    rel = np.load(cache / obs["entity_rel"]).astype(np.float64)
    vel = np.load(cache / obs["entity_vel"]).astype(np.float64)
    count = np.load(cache / obs["entity_count"]).astype(np.int64)
    look = np.load(cache / act["look"]).astype(np.float64)
    tp = np.load(cache / act["target_probs"]).astype(np.float64)
    F = count.shape[0]
    starts = np.zeros(F, dtype=np.int64)
    np.cumsum(count[:-1], out=starts[1:])
    frame_id = np.repeat(np.arange(F), count)
    within = np.arange(rel.shape[0]) - starts[frame_id]
    w = tp[frame_id, (within + 1).clip(max=tp.shape[1] - 1)]
    pooled_rel = np.zeros((F, 3))
    np.add.at(pooled_rel, frame_id, w[:, None] * rel)
    pooled_vel = np.zeros((F, 3))
    np.add.at(pooled_vel, frame_id, w[:, None] * vel)
    max_p = tp[:, 1:].max(axis=1) if tp.shape[1] > 1 else np.zeros(F)
    return pooled_rel, pooled_vel, look, max_p, [int(n) for n in shard["episode_lengths"]]


def _accumulate(z: np.ndarray, ep_lengths: list[int], K: int) -> np.ndarray:
    """Z_K[t] = sum_{j<K} z[t+j] within each episode; NaN where window overflows."""
    out = np.full_like(z, np.nan)
    s = 0
    for n in ep_lengths:
        if n >= K:
            blk = z[s:s + n]
            csum = np.concatenate(
                [np.zeros((1, z.shape[1])), np.cumsum(blk, axis=0)]
            )
            out[s:s + n - K + 1] = csum[K:] - csum[:n - K + 1]
        s += n
    return out


def look_horizon_ceiling(cache: Path) -> dict:
    """K-tick accumulated turn vs grounded features — the horizon R² ceiling.

    For each horizon K and certainty bucket, fits the K-tick accumulated
    tangent (Z_K) against grounded features (e_tgt, lead) and momentum (prev).

    Parameters
    ----------
    cache:
        Precomputed val cache directory.

    Returns
    -------
    dict with keys: ``cache``, ``note``, ``by_certainty``.
    """
    manifest = json.loads((cache / "manifest.json").read_text())
    Z, ETGT, LEAD, PREV, MAXP = [], [], [], [], []
    ZK: dict[int, list] = {K: [] for K in _HORIZON_KS}
    for shard in manifest["shards"]:
        pooled_rel, pooled_vel, look, max_p, eps = _hc_shard(cache, shard)
        base = _unit(pooled_rel)
        proj = (pooled_vel * base).sum(-1, keepdims=True) * base
        lead = _unit(pooled_vel - proj)
        z = tangent_logmap_np(_unit(look))           # (F,2)
        prev = np.zeros_like(z)
        s = 0
        for n in eps:
            if n > 1:
                prev[s + 1:s + n] = z[s:s + n - 1]
            s += n
        Z.append(z)
        ETGT.append(tangent_logmap_np(base))
        LEAD.append(tangent_logmap_np(lead))
        PREV.append(prev)
        MAXP.append(max_p)
        for K in _HORIZON_KS:
            ZK[K].append(_accumulate(z, eps, K))

    ETGT = np.concatenate(ETGT)
    LEAD = np.concatenate(LEAD)
    PREV = np.concatenate(PREV)
    MAXP = np.concatenate(MAXP)
    ZK_cat = {K: np.concatenate(v) for K, v in ZK.items()}
    feats = {"e_tgt": ETGT, "lead": LEAD, "prev": PREV}

    def fit(y2: np.ndarray, names: list[str], mask: np.ndarray) -> tuple[float | None, int]:
        m = mask & np.isfinite(y2).all(axis=1)
        for k in names:
            m = m & np.isfinite(feats[k]).all(axis=1)
        if m.sum() < 100:
            return None, 0
        y = y2[m].reshape(-1)
        X = np.stack([feats[k][m].reshape(-1) for k in names], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float((y * y).sum())
        r2 = 1.0 - float((resid * resid).sum()) / max(ss_tot, _EPS)
        return round(r2, 4), int(m.sum())

    out: dict[str, Any] = {
        "cache": str(cache),
        "note": "Z_K = sum of per-tick tangents (small-angle proxy)",
        "by_certainty": {},
    }
    for bname, lo, hi in _HORIZON_CERT_BUCKETS:
        cmask = (MAXP >= lo) & (MAXP < hi)
        rows: dict[int, dict] = {}
        for K in _HORIZON_KS:
            grounded, n = fit(ZK_cat[K], ["e_tgt", "lead"], cmask)
            etgt_only, _ = fit(ZK_cat[K], ["e_tgt"], cmask)
            momentum, _ = fit(ZK_cat[K], ["prev"], cmask)
            combined, _ = fit(ZK_cat[K], ["e_tgt", "lead", "prev"], cmask)
            rows[K] = {
                "n": n,
                "e_tgt": etgt_only,
                "grounded(e_tgt+lead)": grounded,
                "momentum(prev)": momentum,
                "combined": combined,
            }
        out["by_certainty"][bname] = rows
    return out


# ---------------------------------------------------------------------------
# look_metric_references  (look_metric_references.py core)
# ---------------------------------------------------------------------------

def look_metric_references(cache: Path) -> dict:
    """Reference look metrics for trivial predictors: no-turn and persistence.

    Computes ``look_r2`` and ``look_ewa_deg`` for the no-turn (predict (1,0,0))
    and persistence (predict look[t-1]) baselines over the full validation
    stream.

    Parameters
    ----------
    cache:
        Precomputed val cache directory.

    Returns
    -------
    dict with keys: ``episodes``, ``no_turn``, ``persistence``, each holding
    ``n``, ``look_r2``, ``look_ewa_deg``.
    """
    from qnn.bc.baseline import _load_episodes
    from qnn.bc.look_metrics import LookSums, look_sums_np, r2_and_ewa_from_sums

    _eps_val = 1e-6

    def _unit_rows(look: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        look = np.asarray(look, dtype=np.float64)
        norm = np.linalg.norm(look, axis=-1)
        valid = norm > _eps_val
        unit = np.zeros_like(look)
        unit[valid] = look[valid] / norm[valid, None]
        return unit, valid

    eps = _load_episodes(cache)
    noturn = LookSums(0, 0, 0, 0, 0, 0, 0)
    persist = LookSums(0, 0, 0, 0, 0, 0, 0)
    n_eps = 0
    for ep in eps:
        if "look" not in ep:
            continue
        unit, valid = _unit_rows(ep["look"])
        if unit.shape[0] < 2 or valid.sum() < 1:
            continue
        n_eps += 1
        tgt = unit[valid]
        noturn = noturn + look_sums_np(
            np.tile(np.array([1.0, 0.0, 0.0]), (tgt.shape[0], 1)), tgt,
        )
        both = valid[1:] & valid[:-1]
        if both.any():
            persist = persist + look_sums_np(unit[:-1][both], unit[1:][both])

    out: dict[str, Any] = {"episodes": n_eps}
    for key, s in (("no_turn", noturn), ("persistence", persist)):
        r2, ewa, _ = r2_and_ewa_from_sums(s)
        out[key] = {
            "n": int(s.n),
            "look_r2": round(r2, 4),
            "look_ewa_deg": round(ewa, 2),
        }
    return out


# ---------------------------------------------------------------------------
# look_target_intersection  (look_target_intersection.py core)
# ---------------------------------------------------------------------------

_TI_WINDOWS = (1, 2, 4, 8)
_TI_P_BUCKETS = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
_TI_B0_BANDS_DEG = [(0.0, 5.0), (5.0, 15.0), (15.0, 45.0), (45.0, 180.0)]
_TI_EPS = 1e-9


def _bearing_deg(rel: np.ndarray) -> np.ndarray:
    """View-frame XYZ (forward=+x) → crosshair→entity angle in degrees."""
    n = np.linalg.norm(rel, axis=-1)
    cos = np.where(n > _TI_EPS, rel[:, 0] / np.maximum(n, _TI_EPS), 1.0)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx < _TI_EPS or sy < _TI_EPS:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _ti_collect_pairs(
    cache: Path, shard: dict
) -> dict[int, np.ndarray]:
    """Return {W: array of (b0_deg, bW_deg, p) rows} for one shard."""
    obs, act = shard["obs"], shard["actions"]
    rel = np.load(cache / obs["entity_rel"]).astype(np.float64)
    count = np.load(cache / obs["entity_count"]).astype(np.int64)
    pid = np.load(cache / obs["entity_player_id"]).astype(np.int64)
    tp = np.load(cache / act["target_probs"]).astype(np.float64)
    F = count.shape[0]

    bearing = _bearing_deg(rel)
    starts = np.zeros(F, dtype=np.int64)
    np.cumsum(count[:-1], out=starts[1:])
    frame_id = np.repeat(np.arange(F), count)
    within = np.arange(rel.shape[0]) - starts[frame_id]
    prob = tp[frame_id, (within + 1).clip(max=tp.shape[1] - 1)]

    frame_players: list[dict[int, tuple[float, float]]] = [dict() for _ in range(F)]
    sel = pid != 0
    for t in np.nonzero(sel)[0]:
        frame_players[frame_id[t]][int(pid[t])] = (
            bearing[t],
            prob[t],
        )

    out: dict[int, list] = {w: [] for w in _TI_WINDOWS}
    f0 = 0
    for n in shard["episode_lengths"]:
        n = int(n)
        for W in _TI_WINDOWS:
            for f in range(f0, f0 + n - W):
                here = frame_players[f]
                there = frame_players[f + W]
                for p_id, (b0, p) in here.items():
                    if p_id in there:
                        bW = there[p_id][0]
                        out[W].append((b0, bW, p))
        f0 += n
    return {
        w: (np.asarray(v, dtype=np.float64) if v else np.empty((0, 3)))
        for w, v in out.items()
    }


def _ti_summarize(rows: np.ndarray) -> dict:
    """rows: (N,3) = (b0_deg, bW_deg, p). Bucket by p and by b0 band."""
    if rows.shape[0] == 0:
        return {"n": 0}
    b0, bW, p = rows[:, 0], rows[:, 1], rows[:, 2]
    closure = b0 - bW
    res: dict = {
        "n": int(rows.shape[0]),
        "mean_b0_deg": round(float(b0.mean()), 3),
        "mean_closure_deg": round(float(closure.mean()), 3),
        "overall_gain": round(float(closure.sum() / max(b0.sum(), _TI_EPS)), 4),
        "corr_closure_p": round(_pearson(closure, p), 4),
        "corr_closure_b0": round(_pearson(closure, b0), 4),
        "by_p_bucket": [],
        "by_b0_band_corr_closure_p": [],
    }
    for lo, hi in _TI_P_BUCKETS:
        m = (p >= lo) & (p < hi)
        c = int(m.sum())
        entry: dict = {"p_range": [lo, hi], "n": c}
        if c > 0:
            entry["mean_b0_deg"] = round(float(b0[m].mean()), 3)
            entry["mean_closure_deg"] = round(float(closure[m].mean()), 3)
            entry["gain"] = round(
                float(closure[m].sum() / max(b0[m].sum(), _TI_EPS)), 4
            )
        res["by_p_bucket"].append(entry)
    for lo, hi in _TI_B0_BANDS_DEG:
        m = (b0 >= lo) & (b0 < hi)
        c = int(m.sum())
        entry = {"b0_range_deg": [lo, hi], "n": c}
        if c >= 3:
            entry["corr_closure_p"] = round(_pearson(closure[m], p[m]), 4)
            entry["mean_closure_deg"] = round(float(closure[m].mean()), 3)
        res["by_b0_band_corr_closure_p"].append(entry)
    return res


def look_target_intersection(cache: Path) -> dict:
    """Multi-frame look→target intersection vs the target probability distribution.

    Measures whether the crosshair moves toward high-probability target tokens
    over windows of W ticks.

    Parameters
    ----------
    cache:
        Precomputed val cache directory.

    Returns
    -------
    dict with keys: ``cache``, ``windows`` (keyed by window size W as str).
    """
    manifest = json.loads((cache / "manifest.json").read_text())
    acc: dict[int, list] = {w: [] for w in _TI_WINDOWS}
    for shard in manifest["shards"]:
        for w, rows in _ti_collect_pairs(cache, shard).items():
            if rows.shape[0]:
                acc[w].append(rows)
    out: dict[str, Any] = {"cache": str(cache), "windows": {}}
    for w in _TI_WINDOWS:
        rows = np.concatenate(acc[w]) if acc[w] else np.empty((0, 3))
        out["windows"][str(w)] = _ti_summarize(rows)
    return out


# ---------------------------------------------------------------------------
# aim_point_z_offset  (aim_point_z_offset.py core)
# ---------------------------------------------------------------------------

#: Projectile speed per impulse (u/s); None = hitscan (t=0).
_IMPULSE_SPEED = {2: None, 3: None, 4: 1000.0, 5: 1000.0, 6: 600.0,
                  7: 1000.0, 8: None}
_IMPULSE_NAME = {2: "sg", 3: "ssg", 4: "ng", 5: "sng", 6: "gl",
                 7: "rl", 8: "lg"}
_TEAMMATE = 1
_RANGE_BUCKETS = ((0.0, 300.0), (300.0, 700.0), (700.0, 4096.0))
_MIN_TP_WEIGHT = 0.5
_MAX_RECENCY = 0.05       # seconds; visible-now gate
_GROUNDED_VZ = 30.0       # |target world-ish vz| u/s proxy


def _lead_time(rel: np.ndarray, vel: np.ndarray, speed: float) -> np.ndarray:
    """Smallest non-negative root of |rel + vel t| = speed t, per row."""
    a = (vel * vel).sum(-1) - speed * speed
    b = 2.0 * (rel * vel).sum(-1)
    c = (rel * rel).sum(-1)
    disc = b * b - 4.0 * a * c
    ok = disc >= 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(np.abs(a) < 1e-9, -1e-9, a)
    roots = np.stack([(-b + sq) / (2 * a_safe), (-b - sq) / (2 * a_safe)])
    roots = np.where((roots >= 0) & ok[None], roots, np.inf)
    t = roots.min(axis=0)
    return np.where(np.isfinite(t), t, 0.0)


def _quant(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(x.size),
        "mean": round(float(x.mean()), 2),
        "p10": round(float(q[0]), 2),
        "p25": round(float(q[1]), 2),
        "median": round(float(q[2]), 2),
        "p75": round(float(q[3]), 2),
        "p90": round(float(q[4]), 2),
    }


def _apzo_collect_shard(cache: Path, shard: dict) -> dict[str, np.ndarray]:
    """Collect per-attack-frame aim-z metrics for one shard."""
    from qnn.vocab import TOKEN_ACTOR
    obs, act = shard["obs"], shard["actions"]
    rel = np.load(cache / obs["entity_rel"]).astype(np.float64)
    vel = np.load(cache / obs["entity_vel"]).astype(np.float64)
    half = np.load(cache / obs["entity_half_extents"]).astype(np.float64)
    types = np.load(cache / obs["entity_types"])
    team = np.load(cache / obs["entity_team"])
    rec = np.load(cache / obs["entity_recency"]).astype(np.float64)
    count = np.load(cache / obs["entity_count"]).astype(np.int64)
    self_vel = np.load(cache / obs["vel"]).astype(np.float64)
    pitch = np.load(cache / obs["view_pitch"]).astype(np.float64) * 90.0 / 127.0
    weap = np.load(cache / obs["self_weapon_id"]).astype(np.int64)
    move = np.load(cache / act["move"])
    tp = np.load(cache / act["target_probs"]).astype(np.float64)

    F = count.shape[0]
    starts = np.zeros(F, dtype=np.int64)
    np.cumsum(count[:-1], out=starts[1:])

    attack = (move & 1).astype(bool)
    impulse = np.maximum(weap - 2, 0)
    tgt_col = tp.argmax(axis=1)
    tgt_w = tp[np.arange(F), tgt_col]
    has_tgt = (
        (tgt_col > 0) & (tgt_col - 1 < count) & (tgt_w > _MIN_TP_WEIGHT)
    )
    keep = attack & has_tgt & np.isin(impulse, list(_IMPULSE_SPEED))
    fi = np.nonzero(keep)[0]
    tok = starts[fi] + (tgt_col[fi] - 1)

    actor = (
        (types[tok] == TOKEN_ACTOR)
        & (team[tok] != _TEAMMATE)
        & (rec[tok] <= _MAX_RECENCY)
    )
    fi, tok = fi[actor], tok[actor]

    r, v = rel[tok], vel[tok]
    imp = impulse[fi]
    t = np.zeros(len(fi))
    for ip, speed in _IMPULSE_SPEED.items():
        if speed is not None:
            m = imp == ip
            t[m] = _lead_time(r[m], v[m], speed)
    p = r + v * t[:, None]

    g_w = np.zeros(len(fi))
    g_w[imp == 6] = 800.0
    deployed_dz = (g_w - 800.0) * t * t / 2.0

    return {
        "impulse": imp,
        "range": np.linalg.norm(p, axis=-1),
        "miss_z": p[:, 2],
        "miss_y": p[:, 1],
        "bearing_z": r[:, 2],
        "resid_z": p[:, 2] + deployed_dz,
        "elev_deg": np.degrees(
            np.arctan2(p[:, 2], np.linalg.norm(p[:, :2], axis=-1))
        ),
        "pitch_deg": pitch[fi],
        "half_z": half[tok][:, 2],
        "grounded": np.abs(v[:, 2] + self_vel[fi, 2]) < _GROUNDED_VZ,
        "t_lead": t,
    }


def aim_point_z_offset(cache: Path) -> dict:
    """Per-weapon human aim-point vertical offset from demo attack frames.

    Measures the vertical miss of the lead-corrected intercept point relative
    to the deployed aim-prior's anchor, stratified by range and grounded/airborne.

    Parameters
    ----------
    cache:
        Precomputed val cache directory.

    Returns
    -------
    dict with keys: ``corpus``, ``n_attack_target_frames``, ``weapons``.
    """
    manifest = json.loads((cache / "manifest.json").read_text())
    cols: dict[str, list] = {}
    for shard in manifest["shards"]:
        for k, a in _apzo_collect_shard(cache, shard).items():
            cols.setdefault(k, []).append(a)
    d = {k: np.concatenate(v) for k, v in cols.items()}

    out: dict[str, Any] = {
        "corpus": str(cache),
        "n_attack_target_frames": int(d["impulse"].size),
        "weapons": {},
    }
    for ip, name in _IMPULSE_NAME.items():
        m = d["impulse"] == ip
        w: dict = {
            "n": int(m.sum()),
            "miss_z_units": _quant(d["miss_z"][m]),
            "resid_z_units": _quant(d["resid_z"][m]),
            "miss_y_units": _quant(d["miss_y"][m]),
            "bearing_z_units": _quant(d["bearing_z"][m]),
            "elev_deg": _quant(d["elev_deg"][m]),
            "view_pitch_deg": _quant(d["pitch_deg"][m]),
            "target_half_z": _quant(d["half_z"][m]),
            "by_range": {},
            "by_grounded": {},
        }
        for lo, hi in _RANGE_BUCKETS:
            rm = m & (d["range"] >= lo) & (d["range"] < hi)
            w["by_range"][f"{int(lo)}-{int(hi)}"] = {
                "miss_z": _quant(d["miss_z"][rm]),
                "resid_z": _quant(d["resid_z"][rm]),
            }
        for g, label in ((True, "grounded"), (False, "airborne")):
            gm = m & (d["grounded"] == g)
            w["by_grounded"][label] = _quant(d["miss_z"][gm])
        if _IMPULSE_SPEED[ip] is not None and m.sum() > 1000:
            t_, y_ = d["t_lead"][m], d["miss_z"][m]
            A, B = float(np.median(y_)), 0.0
            for _ in range(20):
                B = float(np.median((y_ - A)[t_ > 0.05] / t_[t_ > 0.05]))
                A = float(np.median(y_ - B * t_))
            w["drop_fit"] = {"A_units": round(A, 2), "B_units_per_s": round(B, 2)}
        out["weapons"][name] = w
    return out


# ---------------------------------------------------------------------------
# look_ground_spin  (look_ground_spin.py core — model-based, corrected)
# ---------------------------------------------------------------------------

PRESENT = 0.5        # target_probs[:,0] <= 1-PRESENT → a target is present
MIN_TURN_DEG = 3.0   # alignment only meaningful on real turns
FLICKS = (15.0, 45.0, 90.0)
TAUS_LOOK = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0)


def _gs_sample_bin(logits: np.ndarray, tau: float, rng: np.random.Generator) -> np.ndarray:
    """(T,K) logits → (T,) sampled bin index at temperature tau."""
    z = logits / max(tau, 1e-6)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    return (rng.random((len(z), 1)) > cdf[:, :-1]).sum(axis=1)


def _gs_consec_cos(z: np.ndarray, thr_rad: float) -> np.ndarray:
    """(T,2) tangent stream → cosine between consecutive turn dirs (both ≥ thr)."""
    if len(z) < 2:
        return np.empty(0)
    a, b = z[1:], z[:-1]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na >= thr_rad) & (nb >= thr_rad)
    if not ok.any():
        return np.empty(0)
    return (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])


def _gs_tangent_dir_deg(
    vec3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(M,3) unit-ish view-frame dirs → (tangent_2d (M,2), turn_mag_deg (M,))."""
    from qnn.model.look_bins import tangent_logmap
    v = torch.tensor(vec3, dtype=torch.float32)
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    z = tangent_logmap(v / n).numpy()
    mag = np.degrees(np.linalg.norm(z, axis=-1))
    return z, mag


def _gs_emd1d(a: np.ndarray, b: np.ndarray) -> float:
    xs = np.sort(np.concatenate([a, b]))
    sa = np.sort(a); sb = np.sort(b)
    Fa = np.searchsorted(sa, xs, side="right") / len(a)
    Fb = np.searchsorted(sb, xs, side="right") / len(b)
    return float(np.trapz(np.abs(Fa - Fb), xs))


def look_ground_spin(
    policy,
    source,
    *,
    run_dir: Path,
    n_episodes: int = 500,
    seed: int = 17,
    device: str | torch.device | None = None,
) -> dict:
    """Spin / grounding analysis of the polar look head.

    Runs the model over a sample of val episodes (B=1 per-episode, full GRU
    context), captures ``look_mag_logits`` / ``look_dir_logits`` /
    ``look_predict`` via forward hooks, and computes:

    * Turn-magnitude distribution (human vs sampled vs mean) with flick rates
      and EMD vs human.
    * Directional persistence: consecutive-frame cosine between turn directions
      (human is the heading-hold ceiling; sampled tests per-frame quantization
      artifacts).
    * Temperature sweep (joint mag+dir, decoupled dir-only, hybrid variants).
    * Target grounding: cosine alignment between look_predict and the direction
      to the most-likely target token.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy in eval mode.  Must have a ``look_head`` with
        ``look_mag_logits``, ``look_dir_logits``, and ``look_predict`` outputs.
    source:
        Resident source loaded with ``segment_mask=None`` (no operative filter).
    run_dir:
        Run directory; used only to install the polar grid from
        ``config/look_grid.json`` (if present) before accessing
        ``MAG_CENTERS``/``DIR_CENTERS``.
    n_episodes:
        Maximum number of episodes to evaluate (linearly spaced).
    seed:
        RNG seed for sampling.
    device:
        Inference device.  Defaults to ``policy.device``.

    Returns
    -------
    dict with keys: ``run``, ``n_episodes``, ``frames_turn``, ``turn_mag_deg``,
    ``turn_mag_by_presence``, ``grounding``, ``directional_persistence``,
    ``temperature_sweep``.
    """
    from qnn.diag.loader import load_policy as _load_policy  # grid install via loader
    from qnn.model.look_bins import install_polar_grid
    from qnn.bc.train import TOKEN_ACTOR

    # Install polar grid from run config.
    run_dir = Path(run_dir)
    look_grid_path = run_dir / "config" / "look_grid.json"
    if look_grid_path.exists():
        import qnn.model.look_bins as _lb
        _lg = json.loads(look_grid_path.read_text())
        install_polar_grid(
            torch.tensor(_lg["mag_centers_rad"], dtype=torch.float32),
            torch.tensor(_lg["dir_centers_rad"], dtype=torch.float32),
            deadzone_rad=_lg.get("deadzone_rad"),
        )

    import qnn.model.look_bins as _lb_live
    mag_centers_deg = np.degrees(_lb_live.MAG_CENTERS.numpy())
    mag_centers_rad = _lb_live.MAG_CENTERS.numpy()
    dir_centers_rad = _lb_live.DIR_CENTERS.numpy()
    dcos = np.cos(dir_centers_rad)
    dsin = np.sin(dir_centers_rad)
    thr_rad = np.radians(MIN_TURN_DEG)

    if device is None:
        device = torch.device(policy.device)
    else:
        device = torch.device(device)

    rng = np.random.default_rng(seed)
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    n_eps = len(offs) - 1
    obs, acts = source.obs, source.actions
    rel_key = "entity_scalars_raw" if "entity_scalars_raw" in obs else "entity_rel"
    obs_keys = list(obs.keys())

    net = policy.model
    caps: dict[str, Any] = {}

    def _cap(m: Any, i: Any, o: Any) -> None:
        caps["mag"] = o.look_mag_logits.detach().float().cpu().numpy()
        caps["dir"] = o.look_dir_logits.detach().float().cpu().numpy()
        caps["pred"] = o.look_predict.detach().float().cpu().numpy()

    net.look_head.register_forward_hook(_cap)

    sel = np.unique(
        np.linspace(0, n_eps - 1, min(n_episodes, n_eps)).astype(int)
    )
    H_mag, S_mag, M_mag = [], [], []
    PRES_MASK = []
    al_h, al_m, al_nact = [], [], []
    PC_h, PC_m, PC_s = [], [], []
    SWEEP = {t: {"pc": [], "mag": []} for t in TAUS_LOOK}
    SWEEP_DIR = {t: {"pc": [], "mag": []} for t in TAUS_LOOK}
    HYB: dict[str, list] = {"pc": [], "mag": []}
    HYB_AMAX: dict[str, list] = {"pc": [], "mag": []}
    HYB_CMEAN: dict[str, list] = {"pc": [], "mag": []}
    t0 = time.monotonic()

    with torch.inference_mode():
        for bi, ei in enumerate(sel):
            lo, hi = int(offs[ei]), int(offs[ei + 1])
            T = hi - lo
            if T < 3:
                continue
            idx = torch.arange(lo, hi, device=device)
            obs_seq = {
                k: obs[k].index_select(0, idx).unsqueeze(1)
                for k in obs_keys
            }
            policy._forward_tensors(obs_seq, hidden=None)
            mag_l = caps["mag"].reshape(T, -1)
            dir_l = caps["dir"].reshape(T, -1)
            pred = caps["pred"].reshape(T, 3)

            human_look = (
                acts["look"].index_select(0, idx).float().cpu().numpy()
            )
            _, hmag = _gs_tangent_dir_deg(human_look)
            zpred, mmag = _gs_tangent_dir_deg(pred)

            mp = np.exp(mag_l - mag_l.max(1, keepdims=True))
            mp /= mp.sum(1, keepdims=True)
            cdf = np.cumsum(mp, axis=1)
            u = rng.random((T, 1))
            mbin = (u > cdf[:, :-1]).sum(axis=1)
            smag = mag_centers_deg[np.clip(mbin, 0, len(mag_centers_deg) - 1)]
            H_mag.append(hmag); S_mag.append(smag); M_mag.append(mmag)

            dp = np.exp(dir_l - dir_l.max(1, keepdims=True))
            dp /= dp.sum(1, keepdims=True)
            dcdf = np.cumsum(dp, axis=1)
            dbin = (rng.random((T, 1)) > dcdf[:, :-1]).sum(axis=1)
            mb = np.clip(mbin, 0, len(mag_centers_rad) - 1)
            db = np.clip(dbin, 0, len(dir_centers_rad) - 1)
            theta = mag_centers_rad[mb]; phi = dir_centers_rad[db]
            zsamp = np.stack(
                [theta * np.cos(phi), theta * np.sin(phi)], axis=1
            )
            zhuman = _gs_tangent_dir_deg(human_look)[0]
            PC_h.append(_gs_consec_cos(zhuman, thr_rad))
            PC_m.append(_gs_consec_cos(zpred, thr_rad))
            PC_s.append(_gs_consec_cos(zsamp, thr_rad))

            for tau in TAUS_LOOK:
                mbt = np.clip(
                    _gs_sample_bin(mag_l, tau, rng), 0, len(mag_centers_rad) - 1
                )
                dbt = np.clip(
                    _gs_sample_bin(dir_l, tau, rng), 0, len(dir_centers_rad) - 1
                )
                th = mag_centers_rad[mbt]; ph = dir_centers_rad[dbt]
                zt = np.stack([th * np.cos(ph), th * np.sin(ph)], axis=1)
                SWEEP[tau]["pc"].append(_gs_consec_cos(zt, thr_rad))
                SWEEP[tau]["mag"].append(np.degrees(th))

            th1 = mag_centers_rad[mb]
            for tau in TAUS_LOOK:
                dbt = np.clip(
                    _gs_sample_bin(dir_l, tau, rng), 0, len(dir_centers_rad) - 1
                )
                ph = dir_centers_rad[dbt]
                zt = np.stack([th1 * np.cos(ph), th1 * np.sin(ph)], axis=1)
                SWEEP_DIR[tau]["pc"].append(_gs_consec_cos(zt, thr_rad))
                SWEEP_DIR[tau]["mag"].append(np.degrees(th1))

            pn = np.linalg.norm(zpred, axis=1, keepdims=True)
            phat = zpred / np.clip(pn, 1e-9, None)
            zhyb = th1[:, None] * phat
            HYB["pc"].append(_gs_consec_cos(zhyb, thr_rad))
            HYB["mag"].append(np.degrees(th1))

            da = dir_l.argmax(1)
            pha = dir_centers_rad[np.clip(da, 0, len(dir_centers_rad) - 1)]
            za = np.stack([th1 * np.cos(pha), th1 * np.sin(pha)], axis=1)
            HYB_AMAX["pc"].append(_gs_consec_cos(za, thr_rad))
            HYB_AMAX["mag"].append(np.degrees(th1))

            dpf = np.exp(dir_l - dir_l.max(1, keepdims=True))
            dpf /= dpf.sum(1, keepdims=True)
            cx = dpf @ dcos; sy2 = dpf @ dsin
            phc = np.arctan2(sy2, cx)
            zc = np.stack([th1 * np.cos(phc), th1 * np.sin(phc)], axis=1)
            HYB_CMEAN["pc"].append(_gs_consec_cos(zc, thr_rad))
            HYB_CMEAN["mag"].append(np.degrees(th1))

            tp = (
                acts["target_probs"]
                .index_select(0, idx)
                .float()
                .cpu()
                .numpy()
            )
            present = tp[:, 0] <= (1.0 - PRESENT)
            PRES_MASK.append(present)
            relall = obs[rel_key].index_select(0, idx)
            if rel_key == "entity_scalars_raw":
                relall = relall[:, :, 3:6]
            relall = relall.float().cpu().numpy()
            N = relall.shape[1]
            best = np.clip(tp[:, 1:].argmax(1), 0, N - 1)
            tgt_rel = relall[np.arange(T), best]
            ztgt, tmag = _gs_tangent_dir_deg(tgt_rel)
            et = obs.get("entity_types")
            n_act = (
                (et.index_select(0, idx).cpu().numpy() == TOKEN_ACTOR).sum(1)
                if et is not None
                else np.zeros(T, dtype=np.int64)
            )

            def cos2(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                na = np.linalg.norm(a, axis=1)
                nb = np.linalg.norm(b, axis=1)
                ok = (na > 1e-9) & (nb > 1e-9)
                c = np.zeros(len(a))
                c[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
                return c, ok

            ch, okh = cos2(_gs_tangent_dir_deg(human_look)[0], ztgt)
            cm, okm = cos2(zpred, ztgt)
            valid = present & (tmag >= MIN_TURN_DEG)
            mh = valid & okh & (hmag >= MIN_TURN_DEG)
            mm = valid & okm & (mmag >= MIN_TURN_DEG)
            al_h.append(ch[mh])
            al_m.append(cm[mm])
            al_nact.append(n_act[mm])
            if bi % 50 == 0:
                print(
                    f"[look]   {bi+1}/{len(sel)} ({time.monotonic()-t0:.1f}s)",
                    flush=True,
                )

    H = np.concatenate(H_mag)
    S = np.concatenate(S_mag)
    M = np.concatenate(M_mag)
    PM = np.concatenate(PRES_MASK)
    AH = np.concatenate(al_h)
    AM = np.concatenate(al_m)
    AN = np.concatenate(al_nact)
    PCH = np.concatenate(PC_h)
    PCM = np.concatenate(PC_m)
    PCS = np.concatenate(PC_s)

    def persist(c: np.ndarray) -> dict:
        return {
            "n": int(len(c)),
            "mean_consec_cos": float(c.mean()) if len(c) else float("nan"),
            "frac_persist_gt0.5": float((c > 0.5).mean()) if len(c) else float("nan"),
            "frac_reversal_lt-0.5": float((c < -0.5).mean()) if len(c) else float("nan"),
        }

    def flicks(x: np.ndarray) -> dict:
        return {f"ge_{int(d)}deg": float((x >= d).mean()) for d in FLICKS}

    def pct(x: np.ndarray) -> list:
        return [round(float(v), 2) for v in np.percentile(x, [50, 90, 99])]

    align_by_nact: dict[str, dict] = {}
    for lo_n, hi_n, lab in [
        (1, 1, "1_actor"), (2, 2, "2_actors"), (3, 99, "3plus_actors")
    ]:
        m = (AN >= lo_n) & (AN <= hi_n)
        align_by_nact[lab] = {
            "n": int(m.sum()),
            "model_mean_cos": float(AM[m].mean()) if m.any() else float("nan"),
        }

    run_name = Path(run_dir).name if run_dir else "unknown"
    report = {
        "run": run_name,
        "n_episodes": int(len(sel)),
        "frames_turn": int(len(H)),
        "turn_mag_deg": {
            "human_pctiles": pct(H),
            "sampled_pctiles": pct(S),
            "mean_pctiles": pct(M),
            "human_flicks": flicks(H),
            "sampled_flicks": flicks(S),
            "mean_flicks": flicks(M),
            "emd_sampled_vs_human_deg": _gs_emd1d(S, H),
            "emd_mean_vs_human_deg": _gs_emd1d(M, H),
        },
        "turn_mag_by_presence": {
            "target_present": {
                "n": int(PM.sum()),
                "human_flicks": flicks(H[PM]) if PM.any() else None,
                "sampled_flicks": flicks(S[PM]) if PM.any() else None,
                "emd_sampled_vs_human_deg": _gs_emd1d(S[PM], H[PM]) if PM.sum() > 1 else None,
            },
            "no_target": {
                "n": int((~PM).sum()),
                "human_flicks": flicks(H[~PM]) if (~PM).any() else None,
                "sampled_flicks": flicks(S[~PM]) if (~PM).any() else None,
                "emd_sampled_vs_human_deg": _gs_emd1d(S[~PM], H[~PM]) if (~PM).sum() > 1 else None,
            },
        },
        "grounding": {
            "frames_aligned": int(len(AM)),
            "human_mean_cos": float(AH.mean()),
            "model_mean_cos": float(AM.mean()),
            "human_frac_cos_gt0.5": float((AH > 0.5).mean()),
            "model_frac_cos_gt0.5": float((AM > 0.5).mean()),
            "model_align_by_n_actors": align_by_nact,
        },
        "directional_persistence": {
            "_note": (
                f"consecutive-frame cosine between turn dirs (both >= "
                f"{MIN_TURN_DEG}deg). human = heading-hold ceiling; mean = "
                "head's deterministic stream; sampled = per-frame (mag,dir) "
                "draws. sampled << human ⇒ sampling flips heading (move-twin)."
            ),
            "human": persist(PCH),
            "mean": persist(PCM),
            "sampled": persist(PCS),
        },
        "temperature_sweep": {
            "_note": (
                f"polar_sample temperature τ. Goal: restore human persistence "
                f"(consec-cos {PCH.mean():.3f}, reversal {(PCH < -0.5).mean():.3f}) "
                "WITHOUT wrecking turn-magnitude (human flicks/EMD below). "
                "Pick the highest τ whose persistence ≈ human and mag EMD stays low."
            ),
            "human_ref": {
                "mean_consec_cos": float(PCH.mean()),
                "frac_reversal_lt-0.5": float((PCH < -0.5).mean()),
                "flicks": flicks(H),
                "pctiles_deg": pct(H),
            },
            "by_tau": {
                f"{t:.2f}": {
                    **persist(np.concatenate(SWEEP[t]["pc"])),
                    "turn_mag": {
                        **flicks(np.concatenate(SWEEP[t]["mag"])),
                        "emd_vs_human_deg": _gs_emd1d(np.concatenate(SWEEP[t]["mag"]), H),
                    },
                }
                for t in TAUS_LOOK
            },
            "by_dir_tau_mag_fixed1": {
                f"{t:.2f}": {
                    **persist(np.concatenate(SWEEP_DIR[t]["pc"])),
                    "turn_mag": {
                        **flicks(np.concatenate(SWEEP_DIR[t]["mag"])),
                        "emd_vs_human_deg": _gs_emd1d(np.concatenate(SWEEP_DIR[t]["mag"]), H),
                    },
                }
                for t in TAUS_LOOK
            },
            "hybrid_sampled_mag_pred_dir": {
                **persist(np.concatenate(HYB["pc"])),
                "turn_mag": {
                    **flicks(np.concatenate(HYB["mag"])),
                    "emd_vs_human_deg": _gs_emd1d(np.concatenate(HYB["mag"]), H),
                },
            },
            "hybrid_sampled_mag_argmax_dir": {
                **persist(np.concatenate(HYB_AMAX["pc"])),
                "turn_mag": {
                    **flicks(np.concatenate(HYB_AMAX["mag"])),
                    "emd_vs_human_deg": _gs_emd1d(np.concatenate(HYB_AMAX["mag"]), H),
                },
            },
            "hybrid_sampled_mag_cmean_dir": {
                **persist(np.concatenate(HYB_CMEAN["pc"])),
                "turn_mag": {
                    **flicks(np.concatenate(HYB_CMEAN["mag"])),
                    "emd_vs_human_deg": _gs_emd1d(np.concatenate(HYB_CMEAN["mag"]), H),
                },
            },
        },
    }
    return report


# ---------------------------------------------------------------------------
# look_aim_prior_decode  (look_aim_prior_decode.py core — model-based)
# ---------------------------------------------------------------------------

HOLD_DEG = 0.5      # "didn't turn" threshold for hold-rate distortion
GAINS = (0.0, 0.0077, 0.015, 0.03, 0.06, 0.12)


def _apd_consec_cos(z: np.ndarray, thr_rad: float) -> np.ndarray:
    if len(z) < 2:
        return np.empty(0)
    a, b = z[1:], z[:-1]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na >= thr_rad) & (nb >= thr_rad)
    if not ok.any():
        return np.empty(0)
    return (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])


def _apd_emd1d(a: np.ndarray, b: np.ndarray) -> float:
    xs = np.sort(np.concatenate([a, b]))
    sa = np.sort(a); sb = np.sort(b)
    Fa = np.searchsorted(sa, xs, side="right") / len(a)
    Fb = np.searchsorted(sb, xs, side="right") / len(b)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(np.abs(Fa - Fb), xs))


def _apd_unit_logmap(vec3: torch.Tensor) -> np.ndarray:
    """(T,3) → (T,2) tangent; zero rows stay zero."""
    from qnn.model.look_bins import tangent_logmap
    n = vec3.norm(dim=-1, keepdim=True)
    z = tangent_logmap(vec3 / n.clamp_min(1e-6))
    return (z * (n > 1e-6)).float().cpu().numpy()


def _apd_cos2(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-9) & (nb > 1e-9)
    c = np.zeros(len(a))
    c[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    return c, ok


def look_aim_prior_decode(
    policy,
    source,
    *,
    run_dir: Path,
    n_episodes: int = 500,
    seed: int = 17,
    device: str | torch.device | None = None,
) -> dict:
    """Aim-prior look decode — offline gain sweep (no-retrain variant).

    Blends a procedural aim anchor into the decoded turn:

        z_out = z_hybrid + gain · z_anchor

    where z_anchor is the log-map of a soft-pooled unit aim direction.
    Sweeps two anchor types (``bearing`` and ``lead``) across ``GAINS``.

    The polar grid is installed from ``run_dir/config/look_grid.json``; the
    live module attributes are used (avoids the stale-snapshot trap).

    Parameters
    ----------
    policy:
        Loaded QNNPolicy in eval mode.  Must have ``look_head`` +
        ``target_pointer`` hooks and ``_has_target_pointer=True``.
    source:
        Resident source loaded with ``segment_mask=None``.
    run_dir:
        Run directory; used to install the polar grid.
    n_episodes:
        Maximum number of episodes to evaluate.
    seed:
        RNG seed.
    device:
        Inference device.  Defaults to ``policy.device``.

    Returns
    -------
    dict with keys: ``run``, ``n_episodes``, ``n_frames``,
    ``frames_anchor_engaged``, ``gains_swept``, ``fitted_gain_ref``,
    ``human``, ``variants``.  Returns ``{"error": "no_target_pointer"}`` if
    the policy has no target pointer.
    """
    from qnn.bc.weapon_physics import (
        ACTOR_REL_OFFSET, ACTOR_VEL_OFFSET, ACTOR_TEAM_OFFSET,
        TEAM_TEAMMATE_VALUE, build_model_weapon_scalars,
    )
    from qnn.model.bench.a25.lead_aim import (
        compute_lead_aim, held_weapon_trajectory, pooled_aim_vec,
    )
    from qnn.model.look_bins import install_polar_grid
    from qnn.vocab import self_weapon_id_to_impulse
    from qnn.bc.train import TOKEN_ACTOR

    run_dir = Path(run_dir)
    look_grid_path = run_dir / "config" / "look_grid.json"
    if look_grid_path.exists():
        _lg = json.loads(look_grid_path.read_text())
        install_polar_grid(
            torch.tensor(_lg["mag_centers_rad"], dtype=torch.float32),
            torch.tensor(_lg["dir_centers_rad"], dtype=torch.float32),
            deadzone_rad=_lg.get("deadzone_rad"),
        )

    import qnn.model.look_bins as _lb_live
    mag_centers_rad = _lb_live.MAG_CENTERS.numpy()
    dir_centers_rad = _lb_live.DIR_CENTERS.numpy()
    dcos = np.cos(dir_centers_rad)
    dsin = np.sin(dir_centers_rad)
    thr_rad = np.radians(MIN_TURN_DEG)

    if device is None:
        device = torch.device(policy.device)
    else:
        device = torch.device(device)

    net = policy.model
    if not getattr(net, "_has_target_pointer", False):
        return {"error": "no_target_pointer"}

    rng = np.random.default_rng(seed)
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    n_eps = len(offs) - 1
    obs, acts = source.obs, source.actions
    obs_keys = list(obs.keys())

    caps: dict[str, Any] = {}
    net.look_head.register_forward_hook(
        lambda m, i, o: caps.update(
            mag=o.look_mag_logits.detach().float().cpu().numpy(),
            dir=o.look_dir_logits.detach().float().cpu().numpy(),
        )
    )
    net.target_pointer.register_forward_hook(
        lambda m, i, o: caps.update(tl=o.target_logits.detach())
    )

    weapon_static = torch.from_numpy(build_model_weapon_scalars()).float().to(device)

    sel = np.unique(
        np.linspace(0, n_eps - 1, min(n_episodes, n_eps)).astype(int)
    )
    variants = [(a, g) for a in ("bearing", "lead") for g in GAINS if g > 0]
    V = {
        ("hybrid", 0.0): {"z": []},
        **{k: {"z": []} for k in variants},
    }
    H_mag_list, ZH = [], []
    ZTGT, ZAIM = [], []
    PRES, ENG = [], []
    t0 = time.monotonic()

    with torch.inference_mode():
        for bi, ei in enumerate(sel):
            lo, hi = int(offs[ei]), int(offs[ei + 1])
            T = hi - lo
            if T < 3:
                continue
            idx = torch.arange(lo, hi, device=device)
            obs_seq = {
                k: obs[k].index_select(0, idx).unsqueeze(1)
                for k in obs_keys
            }
            policy._forward_tensors(obs_seq, hidden=None)
            mag_l = caps["mag"].reshape(T, -1)
            dir_l = caps["dir"].reshape(T, -1)
            target_logits = caps["tl"].reshape(T, -1)

            esc = obs["entity_scalars_raw"].index_select(0, idx).reshape(
                T, -1, obs["entity_scalars_raw"].shape[-1]
            ).float()
            rel = esc[..., ACTOR_REL_OFFSET:ACTOR_REL_OFFSET + 3]
            vel = esc[..., ACTOR_VEL_OFFSET:ACTOR_VEL_OFFSET + 3]
            etypes = obs["entity_types"].index_select(0, idx).reshape(T, -1).long()
            enemy = (etypes == TOKEN_ACTOR) & (
                esc[..., ACTOR_TEAM_OFFSET] != TEAM_TEAMMATE_VALUE
            )
            wid = obs["self_weapon_id"].index_select(0, idx).reshape(T)
            imp = self_weapon_id_to_impulse(wid.long())
            v_h, drop_a, drop_b = held_weapon_trajectory(weapon_static, imp)
            aim_pts = compute_lead_aim(rel, vel, v_h, drop_a, drop_b)
            aim_u = pooled_aim_vec(aim_pts, target_logits, enemy)
            bear_u = pooled_aim_vec(rel, target_logits, enemy)
            z_aim = _apd_unit_logmap(aim_u)
            z_bear = _apd_unit_logmap(bear_u)
            ENG.append(np.linalg.norm(z_aim, axis=1) > 1e-9)

            mp = np.exp(mag_l - mag_l.max(1, keepdims=True))
            mp /= mp.sum(1, keepdims=True)
            mbin = (
                rng.random((T, 1)) > np.cumsum(mp, axis=1)[:, :-1]
            ).sum(axis=1)
            th1 = mag_centers_rad[np.clip(mbin, 0, len(mag_centers_rad) - 1)]
            dp = np.exp(dir_l - dir_l.max(1, keepdims=True))
            dp /= dp.sum(1, keepdims=True)
            phc = np.arctan2(dp @ dsin, dp @ dcos)
            z_c = np.stack(
                [th1 * np.cos(phc), th1 * np.sin(phc)], axis=1
            )

            V[("hybrid", 0.0)]["z"].append(z_c)
            for anchor, g in variants:
                za = z_aim if anchor == "lead" else z_bear
                V[(anchor, g)]["z"].append(z_c + g * za)

            human_look = acts["look"].index_select(0, idx).float()
            zh = _apd_unit_logmap(human_look)
            ZH.append(zh)
            H_mag_list.append(np.degrees(np.linalg.norm(zh, axis=1)))
            tp_arr = (
                acts["target_probs"].index_select(0, idx).float().cpu().numpy()
            )
            PRES.append(tp_arr[:, 0] <= (1.0 - PRESENT))
            best = np.clip(tp_arr[:, 1:].argmax(1), 0, rel.shape[1] - 1)
            tgt_rel = rel[
                torch.arange(T, device=rel.device),
                torch.as_tensor(best, device=rel.device),
            ]
            ZTGT.append(_apd_unit_logmap(tgt_rel))
            ZAIM.append(z_aim)
            if bi % 50 == 0:
                print(
                    f"[aimprior]   {bi+1}/{len(sel)} ({time.monotonic()-t0:.1f}s)",
                    flush=True,
                )

    H = np.concatenate(H_mag_list)
    zh_cat = np.concatenate(ZH)
    ztgt = np.concatenate(ZTGT)
    zaim = np.concatenate(ZAIM)
    pres = np.concatenate(PRES)
    eng = np.concatenate(ENG)
    hmag_deg = H
    tgt_turn = np.degrees(np.linalg.norm(ztgt, axis=1)) >= MIN_TURN_DEG

    def _persist(c: np.ndarray) -> dict:
        return {
            "n": int(len(c)),
            "mean_consec_cos": float(c.mean()) if len(c) else float("nan"),
            "frac_reversal_lt-0.5": float((c < -0.5).mean()) if len(c) else float("nan"),
        }

    def evaluate(z: np.ndarray) -> dict:
        mag_deg = np.degrees(np.linalg.norm(z, axis=1))
        pc = _apd_consec_cos(z, thr_rad)
        cg, okg = _apd_cos2(z, ztgt)
        mg = pres & tgt_turn & okg & (mag_deg >= MIN_TURN_DEG)
        ca, oka = _apd_cos2(z, zaim)
        ma = eng & oka & (mag_deg >= MIN_TURN_DEG)
        return {
            "turn_mag": {
                **{f"ge_{int(d)}deg": float((mag_deg >= d).mean()) for d in FLICKS},
                "hold_rate_lt0.5deg": float((mag_deg < HOLD_DEG).mean()),
                "emd_vs_human_deg": _apd_emd1d(mag_deg, hmag_deg),
            },
            "persistence": _persist(pc),
            "target_grounding": {
                "n": int(mg.sum()),
                "mean_cos": float(cg[mg].mean()) if mg.any() else float("nan"),
            },
            "aim_alignment": {
                "n": int(ma.sum()),
                "mean_cos": float(ca[ma].mean()) if ma.any() else float("nan"),
            },
        }

    ch, okh = _apd_cos2(zh_cat, ztgt)
    mh = pres & tgt_turn & okh & (hmag_deg >= MIN_TURN_DEG)
    cah, okah = _apd_cos2(zh_cat, zaim)
    mah = eng & okah & (hmag_deg >= MIN_TURN_DEG)

    report = {
        "run": Path(run_dir).name,
        "n_episodes": int(len(sel)),
        "n_frames": int(len(H)),
        "frames_anchor_engaged": int(eng.sum()),
        "gains_swept": list(GAINS),
        "fitted_gain_ref": 0.0077,
        "human": {
            "turn_mag": {
                **{f"ge_{int(d)}deg": float((hmag_deg >= d).mean()) for d in FLICKS},
                "hold_rate_lt0.5deg": float((hmag_deg < HOLD_DEG).mean()),
            },
            "persistence": _persist(_apd_consec_cos(zh_cat, thr_rad)),
            "target_grounding": {
                "n": int(mh.sum()),
                "mean_cos": float(ch[mh].mean()),
            },
            "aim_alignment": {
                "n": int(mah.sum()),
                "mean_cos": float(cah[mah].mean()),
            },
        },
        "variants": {
            f"{anchor}@g={g:g}" if g else "hybrid_baseline": evaluate(
                np.concatenate(V[(anchor, g)]["z"])
            )
            for (anchor, g) in [("hybrid", 0.0)] + variants
        },
    }
    return report


# ---------------------------------------------------------------------------
# analyze  — (policy, source)-compatible entry point
# ---------------------------------------------------------------------------

def analyze(
    policy,
    source,
    *,
    segment: str = "all",
    **opts: Any,
) -> dict:
    """Run the (policy, source)-compatible look-head analysis.

    Look analysis does not apply an operative-frame filter by design.
    The cache-based functions (``look_prior_fit``, ``look_horizon_ceiling``,
    etc.) require a ``cache`` path argument and are called directly.
    Model-based functions (``look_ground_spin``, ``look_aim_prior_decode``)
    require a ``run_dir`` argument and are called directly.

    This entry point is a placeholder that satisfies the Phase-2 per-head
    module contract.  It returns an empty ``segment`` tag for now; expand it
    when a (policy, source)-compatible look metric is identified.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy in eval mode.
    source:
        Resident source (any segment).
    segment:
        Metadata tag.

    Returns
    -------
    dict with keys: ``segment``, ``note``.
    """
    return {
        "segment": segment,
        "note": (
            "look analyze() has no resident-source-compatible metrics yet. "
            "Use the named functions directly: look_prior_fit(cache), "
            "look_ground_spin(policy, source, run_dir=...), "
            "look_aim_prior_decode(policy, source, run_dir=...), etc."
        ),
    }
