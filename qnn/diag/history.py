"""Per-head loss / accuracy curves from bc_history.json.

Free diagnostics — no model load required, just JSON parsing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    """Load the per-epoch history list from a BC run directory."""
    p = Path(run_dir) / "checkpoints" / "bc_history.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("history", [])


def per_head_curves(history: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Extract per-epoch trajectories for the canonical head metrics.

    Returns dict mapping metric name to list of (epoch, value) pairs.
    """
    keys = [
        "val_loss", "train_loss", "train_eval_loss", "train_proxy_gap",
        "val_f1_move", "val_f1_fire", "val_f1_weapon",
        "val_cos_sim_look", "val_acc_target",
        "val_f1_move_fb", "val_f1_move_lr", "val_f1_move_ud",
    ]
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    for entry in history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        for k in keys:
            v = entry.get(k)
            if v is None and k.startswith("val_"):
                v = entry.get(k[len("val_"):])  # fall back to unprefixed
            if v is not None:
                out[k].append((int(epoch), float(v)))
    return {k: v for k, v in out.items() if v}


def best_epoch(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the entry with minimum val_loss."""
    candidates = [h for h in history if h.get("val_loss") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda h: h["val_loss"])


def train_val_gap_progression(history: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """Per-epoch (epoch, train_eval_loss - val_loss) — overfit/underfit signal.

    Uses train_eval_loss (eval-mode forward on held-out train shards) when present,
    falling back to train_loss. Positive trending up = overfitting.
    """
    out: list[tuple[int, float]] = []
    for entry in history:
        ep = entry.get("epoch")
        train = entry.get("train_eval_loss") or entry.get("train_loss")
        val = entry.get("val_loss")
        if ep is None or train is None or val is None:
            continue
        out.append((int(ep), float(val) - float(train)))
    return out


def still_improving(history: list[dict[str, Any]], *, last_n: int = 2) -> dict[str, bool]:
    """For each head metric, did the last ``last_n`` epochs show improvement?

    Returns dict of metric → True if still descending in val_loss-equivalent direction.
    """
    curves = per_head_curves(history)
    result: dict[str, bool] = {}
    # Loss/proxy: lower is better; F1/acc: higher is better
    descending_better = {"val_loss", "train_loss", "train_eval_loss", "train_proxy_gap"}
    for metric, points in curves.items():
        if len(points) < last_n + 1:
            continue
        tail = [v for _, v in points[-(last_n + 1):]]
        if metric in descending_better:
            result[metric] = tail[-1] < tail[0]
        else:
            result[metric] = tail[-1] > tail[0]
    return result
