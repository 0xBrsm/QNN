"""Tangent-space look metrics — a smooth, non-saturated score for the look head.

The ``look`` label is a *view-relative turn-delta unit vector*: "no turn this
tick" is the fixed vector ``(1,0,0)`` and the per-tick turn magnitude is
``arccos(look[0])``. Because most ticks are small turns, look vectors cluster
near ``(1,0,0)`` and ``cos_sim_look`` saturates near 1.0 — a copy-previous-frame
predictor scores ~0.99. That metric can't separate models.

Tangent / log-map representation fixes this. Map each unit vector to the 2D
rotation vector at ``(1,0,0)``::

    theta = arccos(u[0])                       # turn magnitude (rad)
    z     = theta * (u[1], u[2]) / ||(u[1], u[2])||   in R^2,  ||z|| = theta

``z`` has magnitude equal to the turn angle and direction equal to the turn
direction; "no turn" maps to ``0``. On ``z`` we report:

* **look_r2** — variance explained: ``1 - SS_res / SS_tot``. The no-turn
  predictor (``z_hat = 0``) scores ~0 by construction; a 40° flick contributes
  ``~40^2`` to ``SS_tot`` while a 1° tick contributes ``~1^2``, so large turns
  dominate automatically — the turn-weighting the error buckets were faking,
  with no buckets. 1.0 = perfect, 0 = no better than predicting no-turn,
  negative = worse than that baseline.
* **look_ewa_deg** — turn-magnitude-weighted mean angular error in degrees:
  ``sum(theta * dtheta) / sum(theta)`` where ``dtheta = arccos(pred . tgt)``.
  Interpretable companion to look_r2.

Both are computed from additive sufficient statistics so they aggregate
correctly across batches / shards (R² is *not* a per-batch average). The
trainer emits the sums as raw-sum metrics and combines them at epoch end via
:func:`r2_and_ewa_from_sums`; the same combiner backs the offline reference
script, so the in-loop and offline numbers cannot drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-8


@dataclass(frozen=True, slots=True)
class LookSums:
    """Additive sufficient statistics for the tangent-space look metrics."""
    n: float          # count of valid frames
    ss_res: float     # sum ||z - z_hat||^2
    z0: float         # sum z[0]
    z1: float         # sum z[1]
    zz: float         # sum ||z||^2
    ew_num: float     # sum theta * dtheta   (rad^2)
    ew_den: float     # sum theta            (rad)

    def __add__(self, other: "LookSums") -> "LookSums":
        return LookSums(
            self.n + other.n, self.ss_res + other.ss_res,
            self.z0 + other.z0, self.z1 + other.z1, self.zz + other.zz,
            self.ew_num + other.ew_num, self.ew_den + other.ew_den,
        )


def tangent_logmap_np(u: np.ndarray) -> np.ndarray:
    """Log-map unit vectors ``(...,3)`` to tangent rotation vectors ``(...,2)``.

    ``||result|| == arccos(u[...,0])`` (the turn angle); no-turn maps to 0.
    """
    u = np.asarray(u, dtype=np.float64)
    u0 = np.clip(u[..., 0], -1.0, 1.0)
    theta = np.arccos(u0)                              # (...,)
    yz = u[..., 1:3]                                   # (...,2)
    yz_norm = np.linalg.norm(yz, axis=-1)              # (...,)
    scale = np.where(yz_norm > _EPS, theta / np.maximum(yz_norm, _EPS), 0.0)
    return yz * scale[..., None]


def look_sums_np(pred_u: np.ndarray, tgt_u: np.ndarray) -> LookSums:
    """Sufficient statistics over a set of (already unit) pred/target vectors."""
    z = tangent_logmap_np(tgt_u)
    zh = tangent_logmap_np(pred_u)
    diff = z - zh
    ss_res = float((diff * diff).sum())
    theta = np.linalg.norm(z, axis=-1)                 # target turn magnitude
    cos = np.clip((np.asarray(pred_u, np.float64) * np.asarray(tgt_u, np.float64)).sum(-1), -1.0, 1.0)
    dtheta = np.arccos(cos)                            # angular error
    return LookSums(
        n=float(z.shape[0]),
        ss_res=ss_res,
        z0=float(z[..., 0].sum()), z1=float(z[..., 1].sum()),
        zz=float((z * z).sum()),
        ew_num=float((theta * dtheta).sum()), ew_den=float(theta.sum()),
    )


def r2_and_ewa_from_sums(s: LookSums | dict) -> tuple[float, float, float]:
    """Combine sufficient statistics into ``(look_r2, look_ewa_deg, r2_noturn)``.

    ``r2_noturn`` is the variance explained by the trivial no-turn predictor
    (``z_hat = 0``) on the SAME frames — a built-in calibration reference that
    should sit at ~0 (slightly negative). Backend-agnostic (plain floats) so the
    trainer and the offline reference script share one implementation. NaN when
    undefined (no frames / no turns).
    """
    if isinstance(s, dict):
        s = LookSums(
            n=s["n"], ss_res=s["ss_res"], z0=s["z0"], z1=s["z1"],
            zz=s["zz"], ew_num=s["ew_num"], ew_den=s["ew_den"],
        )
    if s.n <= 0:
        return float("nan"), float("nan"), float("nan")
    ss_tot = s.zz - (s.z0 * s.z0 + s.z1 * s.z1) / s.n      # sum ||z - zbar||^2
    if ss_tot <= _EPS:
        return float("nan"), float("nan"), float("nan")
    r2 = 1.0 - s.ss_res / ss_tot
    r2_noturn = 1.0 - s.zz / ss_tot                        # z_hat = 0 baseline
    ewa_deg = math.degrees(s.ew_num / s.ew_den) if s.ew_den > _EPS else float("nan")
    return r2, ewa_deg, r2_noturn


def humanlike_from_sums(
    n: float, ce: np.ndarray, pred: np.ndarray, hist: np.ndarray,
) -> tuple[float, float]:
    """Combine binned-head distributional sufficient statistics into ``(look_dll, look_emd_deg)``.

    ``look_dll`` is the per-frame Δloglik (nats, higher=better) = CE(human marginal)
    − CE(model), summed over the two tangent axes — the decision metric. Per-bin
    area terms cancel in the difference, so this matches the offline ablation's
    tangent-density Δloglik (``scripts/analysis/_ablate_look_head.py``).

    The binned look head is a per-axis distribution over foveated tangent bins; for
    human-LIKENESS (not aim accuracy) the question is whether that distribution matches
    the human's, which point metrics (look_r2 / look_ewa_deg, computed on the decoded
    mean) cannot see. Sufficient statistics (additive across batches), per axis a∈{0,1}:

      n            count of valid look frames
      ce[a]        sum over frames of  -log p_model[a, human_bin]
      pred[a,b]    sum over frames of  softmax(logits)[a,b]   (model's marginal bin dist)
      hist[a,b]    count of frames whose human bin == b        (human marginal bin dist)

    * **look_dll** — Δloglik (nats, higher=better): ``CE(human marginal) − CE(model)``,
      summed over the two axes. The decision metric and a PROPER scoring rule:
      hedging probability mass to the center is penalized (low prob on the bin the
      human actually turned to). 0 = no better than the static human turn
      distribution; positive = beats it; negative = worse.
    * **look_emd_deg** — mean per-axis Wasserstein-1 (degrees) between the model's
      marginal bin distribution and the human's: does the model flick as hard/often.

    NaN when no valid frames. Backend-agnostic (numpy); shares the offline reference in
    scripts/analysis/_look_humanlike.py so in-loop and offline numbers cannot drift.
    """
    if n <= 0:
        return float("nan"), float("nan")
    from qnn.model.look_bins import CENTERS  # tangent bin centers (rad), foveated
    centers_deg = np.degrees(CENTERS.numpy().astype(np.float64))
    gaps = np.diff(centers_deg)                            # (N_BINS-1,) non-uniform
    ce = np.asarray(ce, np.float64); pred = np.asarray(pred, np.float64); hist = np.asarray(hist, np.float64)
    dll = 0.0; emds = []
    for a in (0, 1):
        marg = hist[a] / max(hist[a].sum(), _EPS)
        nz = marg > 0
        ce_marg = float(-(marg[nz] * np.log(marg[nz])).sum())   # entropy of human marginal
        ce_model = float(ce[a] / n)
        dll += ce_marg - ce_model                                # Δloglik (nats) per axis
        pm = pred[a] / max(pred[a].sum(), _EPS)
        emds.append(float((np.abs(np.cumsum(pm)[:-1] - np.cumsum(marg)[:-1]) * gaps).sum()))
    return float(dll), float(np.mean(emds))
