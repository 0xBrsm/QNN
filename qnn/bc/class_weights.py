"""Per-head class weights derived from training episode statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from qnn.bc.supervised_loop import Source


def attack_pos_weight_from_counts(
    attack_pos: int,
    attack_neg: int,
    *,
    override: float = 0.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor | None, str]:
    """Build the attack BCE pos_weight tensor from raw counts."""
    if attack_pos == 0:
        return None, "attack weight requested but no positive attack frames in training data; skipping pos_weight"

    auto = float(attack_neg) / float(attack_pos)
    base_rate = float(attack_pos) / float(attack_pos + attack_neg)
    if override > 0.0:
        chosen = float(override)
        line = (
            f"attack base_rate={base_rate:.4f}  pos_weight={chosen:.2f} "
            f"(override; auto would be {auto:.2f})  (pos={attack_pos} neg={attack_neg})"
        )
    else:
        chosen = auto
        line = f"attack base_rate={base_rate:.4f}  pos_weight={chosen:.2f}  (pos={attack_pos} neg={attack_neg})"

    return torch.tensor(chosen, dtype=torch.float32, device=device), line


def attack_class_weights(
    source: "Source",
    *,
    head_loss_weights: dict[str, float],
    override: float = 0.0,
    device: torch.device | str | None = None,
    op_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return ``{"attack": pos_weight_tensor}`` if the head is enabled.

    Source-driven so both resident and streaming pipelines use the same
    call site. The attack head is currently the only head using this idx;
    if its loss weight is 0 the idx is skipped entirely.

    ``op_only=True`` restricts the pos/neg counts to op=1 frames (matches
    the loss's op-mask when ``QNNPolicy.attack_op_only`` is on). Default
    ``False`` counts the full corpus for back-compat.
    """
    if head_loss_weights.get("attack", 0.0) <= 0.0:
        return {}
    pos, neg = source.attack_pos_neg_counts(op_only=op_only)
    tensor, line = attack_pos_weight_from_counts(pos, neg, override=override, device=device)
    suffix = " (op-only)" if op_only else ""
    print(f"  [bc] {line}{suffix}")
    return {"attack": tensor} if tensor is not None else {}
