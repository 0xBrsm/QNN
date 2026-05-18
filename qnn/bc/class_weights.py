"""Per-head class weights derived from training episode statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict

import numpy as np
import torch

from qnn.bc.supervised_loop import PrecomputedEpisode


def fire_pos_weight(
    episodes: Sequence[PrecomputedEpisode],
    *,
    override: float = 0.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor | None, str]:
    """Return ``(weight_tensor, log_line)`` for the fire BCE pos_weight slot.

    Counts positive vs negative fire frames across ``episodes`` and returns
    ``neg_count / pos_count`` as a 0-d tensor.  ``override > 0`` substitutes
    that value for the auto-computed weight; the auto value is still
    surfaced in the log line for sanity-checking.

    Returns ``(None, message)`` when no positive frames are present so the
    caller can skip the slot.
    """
    fire_pos = 0
    fire_neg = 0
    for ep in episodes:
        f = ep.actions.get("fire")
        if f is None:
            continue
        arr = np.asarray(f).reshape(-1)
        p = int((arr > 0).sum())
        fire_pos += p
        fire_neg += int(arr.shape[0]) - p
    if fire_pos == 0:
        return None, "fire weight requested but no positive fire frames in training data; skipping pos_weight"

    auto = float(fire_neg) / float(fire_pos)
    base_rate = float(fire_pos) / float(fire_pos + fire_neg)
    if override > 0.0:
        chosen = float(override)
        line = (
            f"fire base_rate={base_rate:.4f}  pos_weight={chosen:.2f} "
            f"(override; auto would be {auto:.2f})  (pos={fire_pos} neg={fire_neg})"
        )
    else:
        chosen = auto
        line = f"fire base_rate={base_rate:.4f}  pos_weight={chosen:.2f}  (pos={fire_pos} neg={fire_neg})"

    return torch.tensor(chosen, dtype=torch.float32, device=device), line


def fire_class_weights(
    episodes: Sequence[PrecomputedEpisode],
    *,
    head_loss_weights: dict[str, float],
    override: float = 0.0,
    device: torch.device | str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return ``{"fire": pos_weight_tensor}`` if the head is enabled.

    The fire head is the only head currently using the per-head class
    weights slot.  Skip the slot entirely when its head loss weight is 0.
    """
    weights: Dict[str, torch.Tensor] = {}
    if head_loss_weights.get("fire", 0.0) <= 0.0:
        return weights
    tensor, line = fire_pos_weight(episodes, override=override, device=device)
    print(f"  [bc] {line}")
    if tensor is not None:
        weights["fire"] = tensor
    return weights
