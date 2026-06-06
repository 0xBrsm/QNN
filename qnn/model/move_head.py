"""Move head: 3 independent categorical axes (fb, lr, ud).

Each axis is a 3-class softmax over {neg, none, pos}. Implemented as one
Linear(features, 9) and reshaped to (B*, 3 axes, 3 classes). No priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model._mlp import make_head_mlp

OUT_DIM = MOVE_AXES * MOVE_AXIS_CLASSES  # 9 logits


@dataclass(frozen=True, slots=True)
class MoveHeadInput:
    features: torch.Tensor  # (B*, in_dim)


@dataclass(frozen=True, slots=True)
class MoveHeadOutput:
    logits: torch.Tensor  # (B*, MOVE_AXES, MOVE_AXIS_CLASSES)


class MoveHead(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int, activation: str) -> None:
        super().__init__()
        self.mlp = make_head_mlp(in_dim, OUT_DIM, bottleneck_dim, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        logits = self.mlp(inp.features).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)
