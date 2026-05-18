"""Copy-previous-action baseline for BC data.

Measures how well a trivial "repeat last frame's action" policy performs
on each action head.  This sets the floor that any trained model must beat
and characterizes the temporal autocorrelation of the training corpus.

Usage:
    python -m qnn.bc.baseline artifacts/collect/qwd/precomputed_val
    python -m qnn.bc.baseline artifacts/collect/qwd/precomputed_train --json
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict

import numpy as np

from qnn.actions import ACTION_HEADS, CONTINUOUS_ACTION_HEADS


def _load_episodes(cache_dir: Path) -> list[dict[str, np.ndarray]]:
    """Load action arrays per episode from precomputed cache."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    episodes: list[dict[str, np.ndarray]] = []
    if isinstance(manifest, dict) and manifest.get("format") == "sharded_v1":
        for shard in manifest.get("shards", []):
            action_arrays = {
                head: np.load(cache_dir / fname, mmap_mode="r")
                for head, fname in shard["actions"].items()
            }
            start = 0
            for n_samples in shard.get("episode_lengths", []):
                end = start + int(n_samples)
                episodes.append({head: values[start:end] for head, values in action_arrays.items()})
                start = end
        return episodes

    for entry in manifest:
        episodes.append({
            head: np.load(cache_dir / fname, mmap_mode="r")
            for head, fname in entry["actions"].items()
        })
    return episodes


def _angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame angular error in degrees between 3D vectors."""
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    tgt_n = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    cos_sim = np.clip(np.sum(pred_n * tgt_n, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim))


def _target_turn_deg(look: np.ndarray) -> np.ndarray:
    """Per-frame turn magnitude in degrees from look labels."""
    mag = np.sqrt(look[:, 1] ** 2 + look[:, 2] ** 2)
    return np.degrees(np.arcsin(np.clip(mag, 0.0, 1.0)))


def copy_previous_baseline(episodes: Sequence[dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Evaluate copy-previous-action baseline across all action heads."""
    _LOOK_BINS = [("0_1", 0.0, 1.0), ("1_5", 1.0, 5.0), ("5_15", 5.0, 15.0), ("15p", 15.0, 180.0)]

    total_frames = 0
    # Continuous heads: collect angular errors for look, component L1 for move
    look_errors: list[np.ndarray] = []
    look_target_deg: list[np.ndarray] = []
    move_l1: list[np.ndarray] = []
    # Discrete heads: count tp/fp/fn
    discrete_counts: Dict[str, list[int]] = {h: [0, 0, 0] for h in ACTION_HEADS if h not in CONTINUOUS_ACTION_HEADS}

    for ep in episodes:
        n = len(next(iter(ep.values())))
        if n < 2:
            continue
        total_frames += n - 1  # skip frame 0 (no previous)

        # Look: copy previous
        look = ep["look"]
        pred_look = look[:-1]  # frame T-1 as prediction for frame T
        tgt_look = look[1:]
        look_errors.append(_angular_error_deg(pred_look, tgt_look))
        look_target_deg.append(_target_turn_deg(tgt_look))

        # Move: copy previous, L1 error
        move = ep["move"]
        pred_move = move[:-1]
        tgt_move = move[1:]
        move_l1.append(np.abs(pred_move - tgt_move).sum(axis=1))

        # Discrete heads: copy previous, count matches
        for head in discrete_counts:
            if head not in ep:
                continue
            vals = ep[head].flatten()
            pred = vals[:-1]
            tgt = vals[1:]
            pos_pred = pred != 0
            pos_tgt = tgt != 0
            match = pred == tgt
            discrete_counts[head][0] += int((pos_pred & match).sum())  # tp
            discrete_counts[head][1] += int((pos_pred & ~match).sum())  # fp
            discrete_counts[head][2] += int((pos_tgt & ~match).sum())  # fn

    all_look_err = np.concatenate(look_errors)
    all_look_tdeg = np.concatenate(look_target_deg)
    all_move_l1 = np.concatenate(move_l1)

    result: Dict[str, Any] = {
        "total_frames": total_frames,
        "look": {
            "mae_angle_deg": float(all_look_err.mean()),
        },
        "move": {
            "mae_l1": float(all_move_l1.mean()),
        },
    }

    # Look bins
    for tag, lo, hi in _LOOK_BINS:
        mask = (all_look_tdeg >= lo) & (all_look_tdeg < hi)
        cnt = int(mask.sum())
        result["look"][f"n_{tag}"] = cnt
        result["look"][f"pct_{tag}"] = round(100.0 * cnt / max(total_frames, 1), 1)
        if cnt > 0:
            result["look"][f"mae_{tag}_deg"] = float(all_look_err[mask].mean())

    # Move per-component
    # Recompute per-component for reporting
    move_fwd_l1: list[np.ndarray] = []
    move_str_l1: list[np.ndarray] = []
    move_up_l1: list[np.ndarray] = []
    for ep in episodes:
        move = ep["move"]
        if move.shape[0] < 2:
            continue
        diff = np.abs(move[:-1] - move[1:])
        move_fwd_l1.append(diff[:, 0])
        move_str_l1.append(diff[:, 1])
        move_up_l1.append(diff[:, 2])
    result["move"]["mae_forward"] = float(np.concatenate(move_fwd_l1).mean())
    result["move"]["mae_strafe"] = float(np.concatenate(move_str_l1).mean())
    result["move"]["mae_up"] = float(np.concatenate(move_up_l1).mean())

    # Discrete heads
    for head, (tp, fp, fn) in discrete_counts.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
        accuracy = tp / max(tp + fp + fn, 1) if (tp + fp + fn) > 0 else 0.0
        result[head] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=str, help="Path to precomputed_{train,val} directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    episodes = _load_episodes(data_dir)
    result = copy_previous_baseline(episodes)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Copy-previous-action baseline  ({result['total_frames']} frames)")
    print()
    print("=== Look (angular error) ===")
    look = result["look"]
    print(f"  Overall:  {look['mae_angle_deg']:.2f} deg")
    print(f"  {'Bin':10s}  {'Count':>6s}  {'Pct':>5s}  {'MAE':>8s}")
    for tag in ["0_1", "1_5", "5_15", "15p"]:
        n = look.get(f"n_{tag}", 0)
        pct = look.get(f"pct_{tag}", 0.0)
        mae = look.get(f"mae_{tag}_deg", 0.0)
        label = tag.replace("_", "-").replace("p", "+")
        print(f"  {label:10s}  {n:6d}  {pct:4.1f}%  {mae:7.2f}°")
    print()
    print("=== Move (L1 error) ===")
    move = result["move"]
    print(f"  Overall:  {move['mae_l1']:.4f}")
    print(f"  Forward:  {move['mae_forward']:.4f}")
    print(f"  Strafe:   {move['mae_strafe']:.4f}")
    print(f"  Up:       {move['mae_up']:.4f}")
    print()
    print("=== Discrete heads ===")
    for head in sorted(result):
        if head in ("total_frames", "look", "move"):
            continue
        d = result[head]
        print(f"  {head:12s}  P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}  (tp={d['tp']} fp={d['fp']} fn={d['fn']})")


if __name__ == "__main__":
    main()
