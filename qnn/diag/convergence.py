"""Convergence diagnostics — separate "still descending" from "true asymptotic difference".

Many of our 8-epoch ablation comparisons are in regimes where neither config has
converged. The "winner" at epoch 7 may simply have converged faster, not better.
This module quantifies how converged each run was and warns when a comparison is
unreliable.

Functions are pure history analysis — no model load required, runs in milliseconds.
"""
from __future__ import annotations

from typing import Any
import math


# Empirical thresholds from the e16 calibration run on this BC setup.
# slope_per_epoch (val_loss units / epoch) at the end of training:
#   < 0.0020  → converged, comparisons reliable
#   < 0.0050  → near convergence, comparisons partly biased
#   ≥ 0.0050  → still descending, comparisons biased toward fast-converging configs
CONVERGED_THRESHOLD = 0.0020
NEAR_CONVERGED_THRESHOLD = 0.0050


def late_epoch_slope(history: list[dict[str, Any]], *, window: int = 3, key: str = "val_loss") -> float | None:
    """Average per-epoch change in ``key`` over the last ``window`` epochs.

    Returns the slope (negative = improving). Falls back to fewer epochs if
    history is shorter than ``window`` allows.
    """
    pts = [(int(h["epoch"]), float(h[key])) for h in history if key in h and "epoch" in h]
    if len(pts) < 2:
        return None
    pts.sort()
    tail = pts[-min(window, len(pts)):]
    if len(tail) < 2:
        return None
    # Linear fit slope
    xs = [p[0] for p in tail]
    ys = [p[1] for p in tail]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def classify_convergence(slope: float | None) -> str:
    """Tag a slope as ``converged`` / ``near`` / ``descending`` / ``unknown``."""
    if slope is None:
        return "unknown"
    mag = abs(slope)
    if mag < CONVERGED_THRESHOLD:
        return "converged"
    if mag < NEAR_CONVERGED_THRESHOLD:
        return "near"
    return "descending"


def extrapolate_asymptote(
    history: list[dict[str, Any]],
    *,
    key: str = "val_loss",
    min_points: int = 4,
) -> dict[str, float] | None:
    """Fit ``loss(epoch) = loss_inf + a * exp(-b * epoch)`` and return parameters.

    Returns dict with ``loss_inf``, ``a``, ``b``, ``residual_rms``, ``predicted_at_inf``.
    Returns None if fit is unstable or history too short.
    """
    pts = [(int(h["epoch"]), float(h[key])) for h in history if key in h and "epoch" in h]
    pts.sort()
    if len(pts) < min_points:
        return None

    # Crude grid search over loss_inf, then linearize the exponential.
    # For each candidate loss_inf, fit log(loss - loss_inf) = log(a) - b * epoch.
    losses = [y for _, y in pts]
    epochs = [x for x, _ in pts]
    best = None
    cur_min = min(losses)
    # Try loss_inf candidates between (min - 0.5*range) and (min - tiny_eps)
    rng = max(losses) - min(losses)
    if rng <= 0:
        return None

    for frac in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        loss_inf = cur_min - frac * rng
        residuals = [y - loss_inf for y in losses]
        if any(r <= 0 for r in residuals):
            continue
        log_r = [math.log(r) for r in residuals]
        # Linear fit: log_r = log_a - b * epoch
        n = len(epochs)
        mx = sum(epochs) / n
        my = sum(log_r) / n
        num = sum((x - mx) * (y - my) for x, y in zip(epochs, log_r))
        den = sum((x - mx) ** 2 for x in epochs)
        if den == 0:
            continue
        b = -num / den
        if b <= 0:
            continue
        log_a = my + b * mx
        a = math.exp(log_a)
        # Compute residual RMS
        preds = [loss_inf + a * math.exp(-b * x) for x in epochs]
        rms = math.sqrt(sum((p - y) ** 2 for p, y in zip(preds, losses)) / n)
        if best is None or rms < best["residual_rms"]:
            best = {
                "loss_inf": loss_inf,
                "a": a,
                "b": b,
                "residual_rms": rms,
                "predicted_at_inf": loss_inf,
            }
    return best


def reliability_report(history: list[dict[str, Any]], *, key: str = "val_loss") -> dict[str, Any]:
    """Aggregate all convergence stats for one run's history."""
    slope = late_epoch_slope(history, key=key)
    asym = extrapolate_asymptote(history, key=key)
    last = history[-1] if history else None
    return {
        "last_epoch": int(last["epoch"]) if last and "epoch" in last else None,
        "last_value": float(last[key]) if last and key in last else None,
        "slope_last3": slope,
        "convergence": classify_convergence(slope),
        "asymptote_fit": asym,
    }


def compare_runs(
    history_a: list[dict[str, Any]],
    history_b: list[dict[str, Any]],
    *,
    key: str = "val_loss",
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Any]:
    """Compare two configs' final-epoch values and flag whether the comparison is reliable.

    A comparison is reliable when both configs are near-converged AND their slopes
    are similar (so the convergence-speed bias roughly cancels).
    """
    ra = reliability_report(history_a, key=key)
    rb = reliability_report(history_b, key=key)
    delta = None
    if ra["last_value"] is not None and rb["last_value"] is not None:
        delta = ra["last_value"] - rb["last_value"]
    # Reliability: both at least "near", and slopes within 0.002 of each other
    slope_a = ra["slope_last3"]
    slope_b = rb["slope_last3"]
    slope_gap = abs((slope_a or 0) - (slope_b or 0))
    both_near = ra["convergence"] in ("converged", "near") and rb["convergence"] in ("converged", "near")
    reliable = both_near and slope_gap < 0.0020
    warning = None
    if not both_near:
        warning = "one or both configs still descending — comparison biased toward whichever converges faster"
    elif slope_gap >= 0.0020:
        warning = f"slope mismatch ({slope_a:+.4f} vs {slope_b:+.4f}) — comparison may be confounded by convergence speed"
    # Asymptote-projected delta if both fit cleanly
    asym_delta = None
    if ra.get("asymptote_fit") and rb.get("asymptote_fit"):
        asym_delta = ra["asymptote_fit"]["loss_inf"] - rb["asymptote_fit"]["loss_inf"]
    return {
        "label_a": label_a,
        "label_b": label_b,
        "delta_at_last_epoch": delta,
        "asymptote_delta": asym_delta,
        "reliable": reliable,
        "warning": warning,
        "report_a": ra,
        "report_b": rb,
    }
