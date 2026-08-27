"""Attack-head analysis functions — importable core compute for the attack slice.

Each public function corresponds to one canonical analysis script:

    void_fire_rate_by_target   ← fire_target_conditional.py
    fire_offset_distribution   ← attack_offset_distribution.py
    empirical_fire_range       ← attack_empirical_range.py   (source-only, no model)
    input_ablation             ← attack_input_ablation.py

The OPERATIVE filter is applied by construction in every function that operates on
op-masked frames:

    op = input_mask & 1   (bit 0 of act.input_mask)

This invariant is a correctness requirement documented in research/attack-head.md §3.
Do NOT remove or relax these filters at call sites.

``analyze(policy, source)`` runs the (policy, source)-compatible subset and returns a
per-head dict suitable for the unified Phase-2 schema.  Functions that require
additional inputs (e.g. ``empirical_fire_range`` which needs the QNN_DIST_SCALE
constant, or ``input_ablation`` which needs a run_dir + slice layout) remain
standalone named functions with their own signatures.

Usage (thin-wrapper pattern)::

    from qnn.diag.attack import (
        void_fire_rate_by_target,
        fire_offset_distribution,
        empirical_fire_range,
        input_ablation,
        analyze,
    )
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from qnn.bc.supervised_loop import make_resident_source_from_cache
from qnn.model.bench.side_channels import (
    _engagement_ema_scope,
    _target_supervision_scope,
)
from qnn.model.policy import QNNPolicy

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Target-presence probability buckets for void-fire stratification.
VOID_FIRE_EDGES = [0.0, 0.05, 0.2, 0.5, 0.8, 1.0001]
VOID_FIRE_LABELS = ["none[0,.05)", "[.05,.2)", "[.2,.5)", "[.5,.8)", "locked[.8,1]"]

#: Crosshair→target angle buckets (degrees) for the aim-alignment table.
BEARING_EDGES = [0.0, 2.0, 5.0, 10.0, 20.0, 45.0, 180.001]
BEARING_LABELS = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "[20,45)", "[45,180]"]

#: Probability threshold above which the target-token best guess is "present".
PRESENT_THRESH = 0.5

#: Per-forward padded grid budget for void_fire_rate_by_target.
MAX_TB = 16384
MAX_LANES = 128

#: Weapon names indexed by impulse (1..8).
WEAPON_NAMES = ("NONE", "AXE", "SG", "SSG", "NG", "SNG", "GL", "RL", "LG")

#: Action impulse to short name (used by empirical_fire_range).
IMPULSE_WEAPON_NAMES = {1: "axe", 2: "sg", 3: "ssg", 4: "ng",
                        5: "sng", 6: "gl", 7: "rl", 8: "lg"}

#: Slice indices for entity_scalars_raw rel vector.
_ESC_REL_BEGIN, _ESC_REL_END = 3, 6

# 140-dim input layout for EngagedGeomWeaponEmbedAttackHead at d_model=64,
# weapon_embed_dim=8. See engaged_geom_weapon_embed_attack_head.py forward()
# for the cat order; in_dim from the look stack is 2 * d_model = 128.
ATTACK_INPUT_SLICES: dict[str, tuple[int, int]] = {
    "self_readout":   (0, 64),
    "target_feat":    (64, 128),
    "engagement_ema": (128, 129),
    "weapon_embed":   (129, 137),
    "dist_norm":      (137, 138),
    "radial_norm":    (138, 139),
    "tang_norm":      (139, 140),
}

# Cooldown scalar in the arsenal token.
SELF_SCALAR_SLICES: dict[str, tuple[int, int]] = {
    "attack_finished": (0, 1),
}

# 19-dim actor-scalar layout feeding obs_embedding.proj_actor.
ACTOR_SCALAR_SLICES: dict[str, tuple[int, int]] = {
    "half_extents": (0, 3),
    "rel":          (3, 6),
    "dist":         (6, 7),
    "vel":          (7, 10),
    "path":         (10, 13),
    "path_dist":    (13, 14),
    "eta":          (14, 15),
    "facing":       (15, 16),
    "team":         (16, 17),
    "score":        (17, 18),
    "recency":      (18, 19),
}


# ---------------------------------------------------------------------------
# void_fire_rate_by_target  (fire_target_conditional.py core)
# ---------------------------------------------------------------------------

def _plan_batches(lengths: np.ndarray) -> list[list[int]]:
    """Length-bucket episodes into padded batches under the T*B budget."""
    order = [int(i) for i in np.argsort(lengths) if lengths[i] > 0]
    batches, cur, cur_max = [], [], 0
    for ei in order:
        L = int(lengths[ei])
        nmax = max(cur_max, L)
        if cur and (nmax * (len(cur) + 1) > MAX_TB or len(cur) >= MAX_LANES):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(ei)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches


def _collect_fire_probs_padded(policy: QNNPolicy, source, device) -> np.ndarray:
    """Per-frame model fire prob in global row order, via padded lane batches.

    Uses zero-initial hidden per episode (exact within-episode GRU continuity).
    Does NOT apply the operative filter — returns all frames.
    """
    offs = np.asarray(source.episode_offsets, dtype=np.int64)
    lengths = offs[1:] - offs[:-1]
    n_total = int(offs[-1])
    probs = np.zeros(n_total, dtype=np.float32)
    obs = source.obs
    obs_keys = list(obs.keys())
    batches = _plan_batches(lengths)
    print(f"[fire-cond]   {len(batches)} padded batches "
          f"(budget T*B={MAX_TB:,}, lane cap {MAX_LANES})", flush=True)
    t0 = time.monotonic()
    with torch.inference_mode():
        for bi, batch in enumerate(batches):
            B = len(batch)
            Tmax = int(max(lengths[ei] for ei in batch))
            obs_seq = {
                k: torch.zeros((Tmax, B, *obs[k].shape[1:]), dtype=obs[k].dtype, device=device)
                for k in obs_keys
            }
            for b, ei in enumerate(batch):
                lo, hi = int(offs[ei]), int(offs[ei + 1])
                idx = torch.arange(lo, hi, device=device, dtype=torch.long)
                for k in obs_keys:
                    obs_seq[k][: hi - lo, b] = obs[k].index_select(0, idx)
            _, logits, _, _, _ = policy._forward_tensors(obs_seq, hidden=None)
            fire = torch.sigmoid(logits["attack"].reshape(Tmax, B)).detach().float().cpu().numpy()
            for b, ei in enumerate(batch):
                lo, hi = int(offs[ei]), int(offs[ei + 1])
                probs[lo:hi] = fire[: hi - lo, b]
            if bi % 5 == 0 or bi == len(batches) - 1:
                done = sum(len(batches[j]) for j in range(bi + 1))
                print(f"[fire-cond]   batch {bi+1}/{len(batches)} "
                      f"(T={Tmax} B={B}, {done} eps done, "
                      f"{time.monotonic()-t0:.1f}s)", flush=True)
    return probs


def _read_void_labels(source):
    """Extract (p_target, attack_op, human_fire) from a resident source.

    attack_op is the operative mask (input_mask bit 0); p_target is the
    probability that at least one target is present (1 - no-target slot).
    """
    acts = source.actions
    p_target = 1.0 - acts["target_probs"].detach().cpu().numpy()[:, 0].astype(np.float64)
    im = acts["input_mask"].reshape(-1).detach().cpu().numpy().astype(np.uint8)
    attack_op = (im & 1).astype(bool)                               # OPERATIVE filter
    if "attack" in acts:
        human_fire = acts["attack"].reshape(-1).detach().cpu().numpy().astype(bool)
    else:
        mv = acts["move"].reshape(-1).detach().cpu().numpy().astype(np.uint8)
        human_fire = (mv & 1).astype(bool)
    return p_target, attack_op, human_fire


def _compute_bearing(source, device):
    """Crosshair→target angle (deg) per frame, to the most-likely target token.

    entity_rel is view-frame (forward=+x), so bearing = arccos(rel_x / |rel|).
    Target token = argmax over target_probs[:, 1:] (slot 0 = no-target;
    entity token j ↔ slot j+1). Returns (bearing_deg, actor_frac_present).
    """
    from qnn.bc.train import TOKEN_ACTOR
    tp = source.actions["target_probs"]                              # (F, 17)
    obs = source.obs
    if "entity_scalars_raw" in obs:
        rel_all = obs["entity_scalars_raw"][:, :, 3:6].to(torch.float32)  # (F, N, 3)
    else:
        rel_all = obs["entity_rel"].to(torch.float32)                      # (F, N, 3)
    F, N = rel_all.shape[0], rel_all.shape[1]
    best = torch.argmax(tp[:, 1:], dim=1).clamp(max=N - 1)          # (F,)
    fidx = torch.arange(F, device=rel_all.device)
    rel = rel_all[fidx, best]                                        # (F, 3)
    nrm = torch.linalg.norm(rel, dim=-1)
    cosx = torch.where(nrm > 1e-9, rel[:, 0] / nrm.clamp_min(1e-9), torch.ones_like(nrm))
    bearing = torch.rad2deg(torch.arccos(cosx.clamp(-1.0, 1.0))).detach().cpu().numpy()
    actor_frac = float("nan")
    et = obs.get("entity_types")
    if et is not None:
        besttype = et[fidx, best]
        present = (tp[:, 0] <= 1.0 - PRESENT_THRESH)
        if present.any():
            actor_frac = float((besttype[present] == TOKEN_ACTOR).float().mean().item())
    return bearing, actor_frac


def _stratify_by_presence(probs, p_target, op, human_fire, threshold):
    """Fire rate stratified by target-presence probability bucket."""
    idx = np.clip(np.digitize(p_target, VOID_FIRE_EDGES[1:-1], right=False),
                  0, len(VOID_FIRE_LABELS) - 1)
    model_pred = probs > threshold
    out = []
    for b, lab in enumerate(VOID_FIRE_LABELS):
        m = idx == b
        mo = m & op
        out.append({
            "bucket": lab, "n": int(m.sum()), "n_op": int(mo.sum()),
            "model_mean_prob_op": float(probs[mo].mean()) if mo.any() else 0.0,
            "model_fire_rate_op": float(model_pred[mo].mean()) if mo.any() else 0.0,
            "model_fire_rate": float(model_pred[m].mean()) if m.any() else 0.0,
            "human_fire_rate_op": float(human_fire[mo].mean()) if mo.any() else 0.0,
            "human_fire_rate": float(human_fire[m].mean()) if m.any() else 0.0,
        })
    return out


def _stratify_by_bearing(probs, p_target, bearing, op, human_fire, threshold):
    """Among target-present + operative frames, fire rate by crosshair→target angle."""
    present = (p_target >= PRESENT_THRESH) & op
    idx = np.clip(np.digitize(bearing, BEARING_EDGES[1:-1], right=False),
                  0, len(BEARING_LABELS) - 1)
    model_pred = probs > threshold
    out = []
    for b, lab in enumerate(BEARING_LABELS):
        m = present & (idx == b)
        out.append({
            "bearing_deg": lab,
            "n": int(m.sum()),
            "model_fire_rate": float(model_pred[m].mean()) if m.any() else 0.0,
            "model_mean_prob": float(probs[m].mean()) if m.any() else 0.0,
            "human_fire_rate": float(human_fire[m].mean()) if m.any() else 0.0,
        })
    return out


def void_fire_rate_by_target(
    policy: QNNPolicy,
    source,
    *,
    threshold: float = 0.5,
    device: str | torch.device | None = None,
) -> dict:
    """Conditional fire-rate by target presence — the "fires into the void" metric.

    Runs the model over ``source`` (which should have ``segment_mask=None`` so
    no-target frames are KEPT), stratifies the predicted fire probability by the
    target-presence signal ``p = 1 - target_probs[:, 0]``, and returns a dict
    with per-bucket model + human fire rates (op-masked) plus the void ratio.

    The operative filter is applied by construction: ``op = input_mask & 1``.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy (eval mode).
    source:
        Resident source loaded with ``segment_mask=None``.
    threshold:
        Sigmoid threshold for predicted-positive classification.
    device:
        Inference device.  Defaults to ``policy.device``.

    Returns
    -------
    dict with keys: ``threshold``, ``buckets`` (list), ``aim_present_thresh``,
    ``aim_buckets`` (list), ``void_ratio_model_op``, ``void_ratio_human_op``,
    ``bearing_actor_frac``.
    """
    if device is None:
        device = torch.device(policy.device)
    else:
        device = torch.device(device)

    p_target, op, human_fire = _read_void_labels(source)
    bearing, actor_frac = _compute_bearing(source, device)
    print(f"[fire-cond] bearing computed; best-target-token is an actor in "
          f"{actor_frac:.3f} of present frames", flush=True)

    probs = _collect_fire_probs_padded(policy, source, device)
    rows = _stratify_by_presence(probs, p_target, op, human_fire, threshold)
    aim = _stratify_by_bearing(probs, p_target, bearing, op, human_fire, threshold)

    def _r(lab, key):
        return next(r[key] for r in rows if r["bucket"] == lab)

    void_model = _r("none[0,.05)", "model_fire_rate_op") / max(
        _r("locked[.8,1]", "model_fire_rate_op"), 1e-9)
    void_human = _r("none[0,.05)", "human_fire_rate_op") / max(
        _r("locked[.8,1]", "human_fire_rate_op"), 1e-9)

    return {
        "threshold": threshold,
        "buckets": rows,
        "aim_present_thresh": PRESENT_THRESH,
        "aim_buckets": aim,
        "void_ratio_model_op": void_model,
        "void_ratio_human_op": void_human,
        "bearing_actor_frac": actor_frac,
    }


# ---------------------------------------------------------------------------
# fire_offset_distribution  (attack_offset_distribution.py core)
# ---------------------------------------------------------------------------

def _collect_predictions_flat(
    policy: QNNPolicy, source, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat-batch forward pass; returns (probs, attack_target, op, weapon_imp, ep_offsets).

    The operative filter ``op = input_mask & 1`` is extracted here so callers
    can apply it consistently.  weapon_imp is in impulse space (1..8).
    """
    n_total = source.n_total_rows
    probs = np.empty(n_total, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            idx = torch.arange(start, end, device=source.device, dtype=torch.long)
            obs_b, act_b = source.gather(idx)
            with (
                _engagement_ema_scope(act_b),
                _target_supervision_scope(act_b, None),
            ):
                _, logits, *_ = policy._forward_tensors(obs_b)
            attack_logits = logits["attack"].reshape(-1, 9)
            probs[start:end] = (
                1.0 - torch.softmax(attack_logits, dim=-1)[:, 0]
            ).detach().float().cpu().numpy()
    attack_target = (
        source.actions["attack"].reshape(-1).detach().cpu().numpy() > 0
    )
    input_mask = (
        source.actions["input_mask"].reshape(-1).detach().cpu().numpy().astype(np.uint8)
    )
    weapon_imp = source.actions["attack"].reshape(-1).detach().cpu().numpy().astype(np.int8)
    op = (input_mask & 1).astype(bool)                              # OPERATIVE filter
    return probs, attack_target, op, weapon_imp, np.asarray(source.episode_offsets, dtype=np.int64)


def _signed_offsets(
    pred: np.ndarray, true: np.ndarray, ep_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each pred==1 row, signed offset to nearest in-episode true==1 row.

    Returns (offsets_int32, isolated_mask_bool) where isolated means no true
    event exists in that episode.
    """
    offsets: list[int] = []
    isolated: list[bool] = []
    for s, e in zip(ep_offsets[:-1], ep_offsets[1:]):
        s, e = int(s), int(e)
        p_idx = np.flatnonzero(pred[s:e])
        t_idx = np.flatnonzero(true[s:e])
        if p_idx.size == 0:
            continue
        if t_idx.size == 0:
            offsets.extend([0] * p_idx.size)
            isolated.extend([True] * p_idx.size)
            continue
        pos = np.searchsorted(t_idx, p_idx)
        lo = np.clip(pos - 1, 0, t_idx.size - 1)
        hi = np.clip(pos,     0, t_idx.size - 1)
        d_lo = p_idx - t_idx[lo]
        d_hi = p_idx - t_idx[hi]
        pick_lo = np.abs(d_lo) <= np.abs(d_hi)
        signed = np.where(pick_lo, d_lo, d_hi).astype(np.int32)
        offsets.extend(signed.tolist())
        isolated.extend([False] * p_idx.size)
    return np.asarray(offsets, dtype=np.int32), np.asarray(isolated, dtype=bool)


def _offset_summary(offsets: np.ndarray, isolated: np.ndarray, label: str) -> dict:
    """Summary statistics for a signed-offset array."""
    near = offsets[~isolated]
    if near.size == 0:
        return {
            "label": label,
            "n_total": int(offsets.size),
            "n_isolated": int(isolated.sum()),
        }
    return {
        "label":        label,
        "n_total":      int(offsets.size),
        "n_isolated":   int(isolated.sum()),
        "isolated_pct": float(isolated.mean() * 100),
        "near_count":   int(near.size),
        "median":       float(np.median(near)),
        "p25":          float(np.percentile(near, 25)),
        "p75":          float(np.percentile(near, 75)),
        "mean":         float(near.mean()),
        "lead_pct":     float((near < 0).mean() * 100),
        "exact_pct":    float((near == 0).mean() * 100),
        "lag_pct":      float((near > 0).mean() * 100),
        "within_1_pct": float((np.abs(near) <= 1).mean() * 100),
        "within_3_pct": float((np.abs(near) <= 3).mean() * 100),
        "within_5_pct": float((np.abs(near) <= 5).mean() * 100),
    }


def _offset_histogram(
    offsets: np.ndarray, isolated: np.ndarray, edges: np.ndarray
) -> list[int]:
    near = offsets[~isolated]
    hist, _ = np.histogram(near, bins=edges)
    return hist.tolist()


def fire_offset_distribution(
    policy: QNNPolicy,
    source,
    *,
    threshold: float,
    batch_size: int = 4096,
) -> dict:
    """Signed temporal offset analysis for trained attack heads.

    For each predicted-positive frame (model fires) pairs it with the nearest
    in-episode true-positive (human fires) and reports the signed offset:

        offset = pred_frame − nearest_true_frame

    Negative ⇒ model anticipates (fires before human).
    Positive ⇒ model lingers (fires after human).
    Zero     ⇒ exact match.

    The operative filter ``op = input_mask & 1`` plus ``weapon > 0`` are
    applied by construction (frames with no weapon selected are excluded).

    Parameters
    ----------
    policy:
        Loaded QNNPolicy (eval mode).
    source:
        Resident source with ``segment_mask={"act.target": {"$ne": 0}}``.
    threshold:
        Sigmoid threshold for predicted-positive.
    batch_size:
        Flat-batch size for the forward pass.

    Returns
    -------
    dict with keys: ``threshold``, ``n_pred``, ``n_true``, ``pred_to_true``,
    ``true_to_pred``, ``histogram_edges``, ``pred_to_true_histogram``,
    ``true_to_pred_histogram``, ``per_weapon_pred_offsets``.
    """
    probs, target, op, weapon, ep_off = _collect_predictions_flat(
        policy, source, batch_size
    )

    keep = op & (weapon > 0)                                        # OPERATIVE filter
    pred = (probs >= threshold) & keep
    true = target & keep

    print(f"[off] threshold={threshold}  n_pred={int(pred.sum())}  "
          f"n_true={int(true.sum())}")

    pred_offs, pred_iso = _signed_offsets(pred, true, ep_off)
    true_offs, true_iso = _signed_offsets(true, pred, ep_off)

    pred_sum = _offset_summary(pred_offs, pred_iso,
                               "pred → nearest true (lead<0 / lag>0)")
    true_sum = _offset_summary(true_offs, true_iso,
                               "true → nearest pred (model late<0 / model early>0)")

    edges = np.array([-100, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 100], dtype=np.int64)
    pred_hist = _offset_histogram(pred_offs, pred_iso, edges)
    true_hist = _offset_histogram(true_offs, true_iso, edges)

    # Per-weapon breakdown on pred → true offsets.
    per_weapon: dict[str, dict] = {}
    for w in range(1, 9):
        mask_w = pred & (weapon == w)
        if not mask_w.any():
            continue
        off_w, iso_w = _signed_offsets(mask_w, true, ep_off)
        s = _offset_summary(off_w, iso_w, WEAPON_NAMES[w])
        per_weapon[WEAPON_NAMES[w]] = s

    return {
        "threshold": threshold,
        "n_pred": int(pred.sum()),
        "n_true": int(true.sum()),
        "pred_to_true": pred_sum,
        "true_to_pred": true_sum,
        "histogram_edges": edges.tolist(),
        "pred_to_true_histogram": pred_hist,
        "true_to_pred_histogram": true_hist,
        "per_weapon_pred_offsets": per_weapon,
    }


# ---------------------------------------------------------------------------
# empirical_fire_range  (attack_empirical_range.py core — source-only, no model)
# ---------------------------------------------------------------------------

def empirical_fire_range(source) -> dict[str, dict]:
    """Per-weapon empirical fire-range distribution from demo data.

    For every val frame with ``attack=1`` AND ``input_mask_bit0=1`` (an
    engine-accepted demo press), computes the soft-target distance
    ``|Σ p · rel|`` in world units and bins by actual action weapon.

    The operative filter ``engine_ready = input_mask & 1`` is applied by
    construction.

    Parameters
    ----------
    source:
        Resident source with ``segment_mask={"act.target": {"$ne": 0}}``.
        Must have ``obs["entity_scalars_raw"]`` in the current native_v1 format.

    Returns
    -------
    dict mapping weapon name → per-weapon stats dict with keys:
    ``weapon``, ``entity_id``, ``impulse``, ``engine_range_qu``,
    ``n_fires``, ``quantiles_qu``, ``mean_qu``, ``pct_beyond_engine_range``.
    """
    from qnn.bc.weapon_physics import QNN_DIST_SCALE, WEAPON_PHYSICS

    td = source.actions["target_probs"]
    present = (1.0 - td[..., 0]).clamp(min=1e-6)
    target_probs_idx = td[..., 1:] / present.unsqueeze(-1)           # (n, N)

    with torch.inference_mode():
        obs = source.obs
        rel = obs["entity_scalars_raw"][..., _ESC_REL_BEGIN:_ESC_REL_END]
        soft_rel = (target_probs_idx.unsqueeze(-1) * rel).sum(dim=-2)  # (n, 3)
        soft_dist_qu = torch.linalg.vector_norm(soft_rel, dim=-1) * QNN_DIST_SCALE

    soft_dist = soft_dist_qu.float().cpu().numpy()
    attack_np = source.actions["attack"].reshape(-1).detach().cpu().numpy().astype(int)
    input_mask = (
        source.actions["input_mask"].reshape(-1).detach().cpu().numpy().astype(np.uint8)
    )
    engine_ready = (input_mask & 1).astype(bool)                    # OPERATIVE filter
    fired = (attack_np > 0) & engine_ready

    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    out: dict[str, dict] = {}
    for impulse, name in IMPULSE_WEAPON_NAMES.items():
        m = fired & (attack_np == impulse)
        n = int(m.sum())
        if n < 50:
            continue
        dists = soft_dist[m]
        engine_range = float(WEAPON_PHYSICS[impulse].get("range", float("nan")))
        beyond = (
            float(np.mean(dists > engine_range) * 100.0)
            if not np.isnan(engine_range)
            else float("nan")
        )
        out[name] = {
            "weapon": name,
            "impulse": impulse,
            "engine_range_qu": engine_range,
            "n_fires": n,
            "quantiles_qu": {f"{q:.2f}": float(np.quantile(dists, q)) for q in qs},
            "mean_qu": float(dists.mean()),
            "pct_beyond_engine_range": beyond,
        }
    return out


# ---------------------------------------------------------------------------
# input_ablation  (attack_input_ablation.py core)
# ---------------------------------------------------------------------------

def _resolve_linear(policy: QNNPolicy, qualified_name: str) -> nn.Linear:
    """Walk the model graph by dotted name and return the named Linear."""
    target: Any = policy.model
    for part in qualified_name.split("."):
        target = getattr(target, part)
    if not isinstance(target, nn.Linear):
        raise ValueError(
            f"{qualified_name!r} is not a Linear (got {type(target).__name__})"
        )
    return target


def _local_input_saliency(
    policy: QNNPolicy,
    val_episodes: list[dict[str, Any]],
    *,
    target: str,
    head_logit: str = "attack",
) -> dict[str, np.ndarray]:
    """Per-dim ``|∂head_logit / ∂Linear-input|`` over val, contextvars wired.

    Mirrors :func:`qnn.diag.input_ablation.input_saliency` but threads both
    ``_engagement_ema_scope`` and ``_target_supervision_scope`` so modern
    attack heads that call ``current_engagement_ema_context()`` or read the
    GT-target pointer don't raise.
    """
    linear = _resolve_linear(policy, target)
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_m, inp):
        x = inp[0].detach().clone().requires_grad_(True)
        x.retain_grad()
        captured["x"] = x
        return (x,) + tuple(inp[1:])

    handle = linear.register_forward_pre_hook(pre_hook)
    in_features = int(linear.weight.shape[1])
    abs_grad_sum = torch.zeros(in_features, dtype=torch.float32)
    abs_input_sum = torch.zeros_like(abs_grad_sum)
    n_seen = 0
    try:
        for ep in val_episodes:
            obs_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(1).to(policy.device)
                for k, v in ep["obs"].items()
            }
            act_t = {
                k: torch.from_numpy(np.ascontiguousarray(v)).to(policy.device)
                for k, v in ep["actions"].items()
            }
            policy.model.zero_grad(set_to_none=True)
            with (
                _engagement_ema_scope(act_t),
                _target_supervision_scope(act_t, None),
            ):
                _, logits, *_ = policy._forward_tensors(obs_t)
            if head_logit not in logits:
                raise RuntimeError(
                    f"head_logit={head_logit!r} not in forward logits "
                    f"(have {sorted(logits)})"
                )
            logits[head_logit].sum().backward()
            x = captured.get("x")
            if x is None or x.grad is None:
                continue
            g = x.grad.detach().reshape(-1, x.shape[-1]).float().cpu()
            xa = x.detach().reshape(-1, x.shape[-1]).float().cpu()
            abs_grad_sum += g.abs().sum(dim=0)
            abs_input_sum += xa.abs().sum(dim=0)
            n_seen += g.shape[0]
    finally:
        handle.remove()
        policy.model.zero_grad(set_to_none=True)

    denom = max(n_seen, 1)
    mean_abs_grad = (abs_grad_sum / denom).numpy()
    mean_abs_input = (abs_input_sum / denom).numpy()
    return {
        "mean_abs_grad": mean_abs_grad,
        "mean_abs_input": mean_abs_input,
        "saliency": mean_abs_grad * mean_abs_input,
        "n_frames": float(n_seen),
    }


def _episodes_from_source(source) -> list[dict[str, Any]]:
    """Slice a ResidentSource into per-episode dicts for the diag ablation toolkit.

    Every action key is forwarded (including ``engagement_ema``,
    ``attack_shifted``, ``attack_distance_to_pos``) so contextvar scopes
    inside ``evaluate_supervised`` are correctly wired.
    """
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    eps: list[dict[str, Any]] = []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        ep_obs = {k: v[s:e].detach().cpu().numpy() for k, v in source.obs.items()}
        ep_act = {k: v[s:e].detach().cpu().numpy() for k, v in source.actions.items()}
        eps.append({"obs": ep_obs, "actions": ep_act, "n_samples": int(e - s)})
    return eps


def _format_slice_ablation_table(
    rows: dict[str, dict[str, float]],
    slices: dict[str, tuple[int, int]],
) -> str:
    """Render slice-ablation results as a markdown table sorted by Δ desc."""
    items = []
    for name, r in rows.items():
        base = float(r["baseline_loss"])
        abl = float(r["ablated_loss"])
        d = float(r["delta"])
        pct = (d / base * 100.0) if base > 0 else float("nan")
        lo, hi = slices[name]
        items.append((name, lo, hi - lo, base, abl, d, pct))
    items.sort(key=lambda r: r[5], reverse=True)
    lines = [
        "| slice | dims | baseline | ablated | Δ loss | % Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, lo, n, base, abl, d, pct in items:
        lines.append(
            f"| `{name}` ({lo}:{lo + n}) | {n} | {base:.4f} | {abl:.4f} | "
            f"{d:+.4f} | {pct:+.1f}% |"
        )
    return "\n".join(lines)


def _format_saliency_table(
    sal: dict[str, np.ndarray],
    slices: dict[str, tuple[int, int]],
) -> tuple[str, list[tuple[str, float]]]:
    """Aggregate per-dim saliency into named slices, sort and render."""
    sal_per_dim = sal["saliency"]
    mean_abs_grad = sal["mean_abs_grad"]
    mean_abs_input = sal["mean_abs_input"]
    rows = []
    for name, (lo, hi) in slices.items():
        s = float(np.mean(sal_per_dim[lo:hi]))
        g = float(np.mean(mean_abs_grad[lo:hi]))
        x = float(np.mean(mean_abs_input[lo:hi]))
        rows.append((name, lo, hi - lo, g, x, s))
    rows.sort(key=lambda r: r[5], reverse=True)
    lines = [
        "| rank | slice | dims | mean \\|∂loss/∂x\\| | mean \\|x\\| | saliency (grad × \\|x\\|) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    rank_pairs: list[tuple[str, float]] = []
    for rank, (name, lo, n, g, x, s) in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{name}` ({lo}:{lo + n}) | {n} | {g:.2e} | {x:.2e} | {s:.2e} |"
        )
        rank_pairs.append((name, s))
    return "\n".join(lines), rank_pairs


def _ranking_agreement(
    ablation_rows: dict[str, dict[str, float]],
    saliency_rank_pairs: list[tuple[str, float]],
) -> str:
    """Spearman-like comment on whether the two methods agree on top blocks."""
    abl_sorted = sorted(
        ablation_rows.items(), key=lambda kv: float(kv[1]["delta"]), reverse=True
    )
    abl_order = [k for k, _ in abl_sorted]
    sal_order = [k for k, _ in saliency_rank_pairs]
    top_abl = set(abl_order[:3])
    top_sal = set(sal_order[:3])
    overlap = top_abl & top_sal
    return (
        f"Top-3 ablation blocks: {abl_order[:3]}; top-3 saliency blocks: "
        f"{sal_order[:3]}; overlap = {sorted(overlap)} "
        f"({len(overlap)}/3)."
    )


def input_ablation(
    policy: QNNPolicy,
    source,
    *,
    target: str = "attack_head.mlp.0",
    self_target: str = "obs_embedding.self_builders.arsenal.projs.0",
    actor_target: str = "obs_embedding.proj_actor",
    attack_input_slices: dict[str, tuple[int, int]] | None = None,
    self_scalar_slices: dict[str, tuple[int, int]] | None = None,
    actor_scalar_slices: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """Attack-head input ablation + saliency (3 sections).

    Drives ``input_slice_ablation`` and ``_local_input_saliency`` against
    three named Linear modules:

    * Section 1 — ``target`` (default ``attack_head.mlp.0``): the 140-dim
      head MLP input.
    * Section 2 — ``self_target`` (default ``obs_embedding.self_proj``): the
      17-dim self-scalars encoder input.
    * Section 3 — ``actor_target`` (default ``obs_embedding.proj_actor``): the
      19-dim per-actor scalars encoder input.

    The ``source`` must already be sliced to the desired episode subset (e.g.
    via ``source.head(n)``); call ``_episodes_from_source`` on it before
    passing to this function.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy (eval mode).  Must expose the named Linear modules.
    source:
        Episode list from ``_episodes_from_source`` — a list of dicts
        ``{"obs": ..., "actions": ..., "n_samples": int}``.
    target, self_target, actor_target:
        Dotted module paths for each of the three Linears.
    attack_input_slices, self_scalar_slices, actor_scalar_slices:
        Slice dicts (name → (lo, hi)).  Defaults to module-level constants.

    Returns
    -------
    dict with keys ``baseline_loss``, ``head``, ``self_scalars``,
    ``actor_scalars``, each holding ``ablation_rows``, ``saliency``,
    ``ablation_md``, ``saliency_md``, ``interpretation``.
    """
    from qnn.diag.input_ablation import input_slice_ablation
    from qnn.diag.ablation import episode_val_loss

    if attack_input_slices is None:
        attack_input_slices = ATTACK_INPUT_SLICES
    if self_scalar_slices is None:
        self_scalar_slices = SELF_SCALAR_SLICES
    if actor_scalar_slices is None:
        actor_scalar_slices = ACTOR_SCALAR_SLICES

    val_episodes = source if isinstance(source, list) else _episodes_from_source(source)

    baseline = episode_val_loss(policy, val_episodes)

    # --- Section 1: attack head MLP input ---
    head_ablation_rows = input_slice_ablation(
        policy, val_episodes, target=target, slices=attack_input_slices,
    )
    head_sal = _local_input_saliency(
        policy, val_episodes, target=target, head_logit="attack",
    )
    head_ablation_md = _format_slice_ablation_table(head_ablation_rows, attack_input_slices)
    head_saliency_md, head_sal_pairs = _format_saliency_table(head_sal, attack_input_slices)
    head_agreement = _ranking_agreement(head_ablation_rows, head_sal_pairs)
    head_essential = [
        k for k, r in head_ablation_rows.items() if float(r["delta"]) >= 0.05
    ]
    head_dead = [
        k for k, r in head_ablation_rows.items() if abs(float(r["delta"])) < 0.005
    ]
    head_interp = (
        f"Essential blocks (Δloss ≥ 0.05): {head_essential or 'none'}. "
        f"Near-dead blocks (|Δloss| < 0.005): {head_dead or 'none'}. "
        f"{head_agreement} "
        "Strong agreement here means saliency is a safe screen for further "
        "single-dim drilldowns; disagreement means a block is large-magnitude "
        "but the loss is robust to zeroing it (likely redundancy with another "
        "block) and ablation deltas should be trusted over raw gradient size."
    )

    # --- Section 2: encoder self_scalars ---
    self_ablation_rows = input_slice_ablation(
        policy, val_episodes, target=self_target, slices=self_scalar_slices,
    )
    self_sal = _local_input_saliency(
        policy, val_episodes, target=self_target, head_logit="attack",
    )
    self_ablation_md = _format_slice_ablation_table(self_ablation_rows, self_scalar_slices)
    self_saliency_md, self_sal_pairs = _format_saliency_table(self_sal, self_scalar_slices)
    self_agreement = _ranking_agreement(self_ablation_rows, self_sal_pairs)
    self_essential = [
        k for k, r in self_ablation_rows.items() if float(r["delta"]) >= 0.005
    ]
    self_dead = [
        k for k, r in self_ablation_rows.items() if abs(float(r["delta"])) < 0.0005
    ]
    af = float(self_ablation_rows["attack_finished"]["delta"])
    flags = []
    if abs(af) < 0.005:
        flags.append(
            f"`attack_finished` ablation Δ = {af:+.4f} — surprisingly small "
            "for the cooldown gate; flag for follow-up"
        )
    else:
        flags.append(
            f"`attack_finished` ablation Δ = {af:+.4f} (acts as the cooldown "
            "gate as expected)"
        )
    self_interp = (
        f"Essential scalars (Δloss ≥ 0.005): {self_essential or 'none'}. "
        f"Near-dead scalars (|Δloss| < 0.0005): {self_dead or 'none'}. "
        f"{self_agreement} "
        f"Spot check: {flags[0]}."
    )

    # --- Section 3: encoder actor_scalars ---
    actor_ablation_rows = input_slice_ablation(
        policy, val_episodes, target=actor_target, slices=actor_scalar_slices,
    )
    actor_sal = _local_input_saliency(
        policy, val_episodes, target=actor_target, head_logit="attack",
    )
    actor_ablation_md = _format_slice_ablation_table(actor_ablation_rows, actor_scalar_slices)
    actor_saliency_md, actor_sal_pairs = _format_saliency_table(actor_sal, actor_scalar_slices)
    actor_agreement = _ranking_agreement(actor_ablation_rows, actor_sal_pairs)
    actor_essential = [
        k for k, r in actor_ablation_rows.items() if float(r["delta"]) >= 0.005
    ]
    actor_dead = [
        k for k, r in actor_ablation_rows.items() if abs(float(r["delta"])) < 0.0005
    ]

    a_rel       = float(actor_ablation_rows["rel"]["delta"])
    a_dist      = float(actor_ablation_rows["dist"]["delta"])
    a_team      = float(actor_ablation_rows["team"]["delta"])
    a_recency   = float(actor_ablation_rows["recency"]["delta"])
    a_vel       = float(actor_ablation_rows["vel"]["delta"])
    a_path      = float(actor_ablation_rows["path"]["delta"])
    a_path_dist = float(actor_ablation_rows["path_dist"]["delta"])
    a_eta       = float(actor_ablation_rows["eta"]["delta"])
    a_facing    = float(actor_ablation_rows["facing"]["delta"])
    a_score     = float(actor_ablation_rows["score"]["delta"])
    h_radial    = float(head_ablation_rows["radial_norm"]["delta"])
    h_tang      = float(head_ablation_rows["tang_norm"]["delta"])
    h_eema      = float(head_ablation_rows["engagement_ema"]["delta"])

    actor_flags: list[str] = []
    if abs(a_rel) < 0.005:
        actor_flags.append(
            f"`rel` ablation Δ = {a_rel:+.4f} — surprisingly small for the "
            "geometry-of-engagement signal; the head may be reading geometry "
            "entirely from its dedicated `dist_norm` / `radial_norm` / "
            "`tang_norm` columns instead of via the target token"
        )
    else:
        actor_flags.append(
            f"`rel` ablation Δ = {a_rel:+.4f} — the encoder's per-actor "
            "view-frame offset is essential for fire decisions (matches the "
            "intuition that `target_feat` is geometry-bearing)"
        )
    actor_flags.append(
        f"`dist` ablation Δ = {a_dist:+.4f} vs head-side `dist_norm` Δ = "
        f"+{float(head_ablation_rows['dist_norm']['delta']):.4f}: "
        + (
            "dist at the encoder is largely redundant with `rel` (since "
            "dist = |rel|) and the head-side `dist_norm` is the load-bearing "
            "scalar copy"
            if abs(a_dist) < 0.01
            else "encoder-side `dist` carries meaningful signal of its own "
                 "even after `rel` is present"
        )
    )
    if abs(a_team) < 0.001:
        actor_flags.append(
            f"`team` ablation Δ = {a_team:+.4f} — effectively dead. The "
            "trained model is NOT using the team bit to gate fire decisions; "
            "either the demo cache contains no teammate-firing-opportunity "
            "frames so there's nothing to learn from, or the encoder learned "
            "to ignore team entirely. Worth flagging for follow-up if "
            "friendly-fire avoidance is in scope."
        )
    else:
        actor_flags.append(
            f"`team` ablation Δ = {a_team:+.4f} — meaningful; the encoder is "
            "leaning on team to gate fire (friendly-fire avoidance signal)"
        )
    if abs(a_recency) < 0.001:
        actor_flags.append(
            f"`recency` ablation Δ = {a_recency:+.4f} — surprisingly dead. "
            "Either the cache is dominated by recency ≈ 0 (visible-now) "
            "frames so there's no variance, or the head has no stale-target "
            "discount"
        )
    else:
        actor_flags.append(
            f"`recency` ablation Δ = {a_recency:+.4f} — the head discounts "
            "stale-perception targets"
        )
    actor_flags.append(
        f"`vel` ablation Δ = {a_vel:+.4f} vs head-side `radial_norm` Δ = "
        f"{h_radial:+.4f}, `tang_norm` Δ = {h_tang:+.4f}, `engagement_ema` "
        f"Δ = {h_eema:+.4f}: "
        + (
            "encoder vel is dead AND head-side velocity-derived columns "
            "are also near-dead — the model genuinely is not using closing "
            "kinematics for the fire decision (likely the BC label is the "
            "C-side hitscan flag which already integrates kinematics)"
            if abs(a_vel) < 0.005 and abs(h_radial) < 0.005 and abs(h_tang) < 0.005
            else "encoder vel and head radial/tang play asymmetric roles; "
                 "the head copies are the load-bearing path"
        )
    )
    actor_flags.append(
        "Move/target-head-relevant scalars largely sit out the fire decision: "
        f"`path` Δ = {a_path:+.4f}, `path_dist` Δ = {a_path_dist:+.4f}, "
        f"`eta` Δ = {a_eta:+.4f}, `facing` Δ = {a_facing:+.4f}, "
        f"`score` Δ = {a_score:+.4f}"
    )
    actor_interp = (
        f"Essential actor scalars (Δloss ≥ 0.005): {actor_essential or 'none'}. "
        f"Near-dead scalars (|Δloss| < 0.0005): {actor_dead or 'none'}. "
        f"{actor_agreement} "
        "Spot checks: " + "; ".join(actor_flags) + "."
    )

    return {
        "baseline_loss": baseline,
        "head": {
            "ablation_rows": head_ablation_rows,
            "saliency": {k: v.tolist() if hasattr(v, "tolist") else v
                         for k, v in head_sal.items()},
            "ablation_md": head_ablation_md,
            "saliency_md": head_saliency_md,
            "interpretation": head_interp,
        },
        "self_scalars": {
            "ablation_rows": self_ablation_rows,
            "saliency": {k: v.tolist() if hasattr(v, "tolist") else v
                         for k, v in self_sal.items()},
            "ablation_md": self_ablation_md,
            "saliency_md": self_saliency_md,
            "interpretation": self_interp,
        },
        "actor_scalars": {
            "ablation_rows": actor_ablation_rows,
            "saliency": {k: v.tolist() if hasattr(v, "tolist") else v
                         for k, v in actor_sal.items()},
            "ablation_md": actor_ablation_md,
            "saliency_md": actor_saliency_md,
            "interpretation": actor_interp,
        },
    }


# ---------------------------------------------------------------------------
# analyze  — (policy, source)-compatible entry point
# ---------------------------------------------------------------------------

def analyze(
    policy: QNNPolicy,
    source,
    *,
    segment: str = "engaged",
    threshold: float = 0.5,
    batch_size: int = 4096,
) -> dict:
    """Run the (policy, source)-compatible attack-head analysis functions.

    Calls ``fire_offset_distribution`` (requires engaged source) and returns a
    per-head dict.  ``void_fire_rate_by_target`` and ``empirical_fire_range``
    require different source configurations and are not included here; call
    them directly with the appropriate source.

    Parameters
    ----------
    policy:
        Loaded QNNPolicy in eval mode.
    source:
        Resident source with ``segment_mask={"act.target": {"$ne": 0}}``
        (engaged frames).
    segment:
        Metadata tag (e.g. ``"engaged"``).
    threshold:
        Sigmoid threshold for predicted-positive.
    batch_size:
        Flat-batch size for the forward pass.

    Returns
    -------
    dict with keys matching the Phase-2 per-head schema:
    ``segment``, ``offset_distribution``.
    """
    offset = fire_offset_distribution(
        policy, source, threshold=threshold, batch_size=batch_size
    )
    return {
        "segment": segment,
        "offset_distribution": offset,
    }
