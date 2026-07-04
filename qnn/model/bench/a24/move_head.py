"""a24 move head — CLS-readout MLP (graph move type ``"cls"``).

Override of the canonical :class:`qnn.model.move_head.MoveHead`: a plain MLP
over the leading ``in_dim`` features (the CLS self-readout, or the
GRU-integrated CLS when a temporal node is active). Registered in
``qnn.model.graph.build.HEAD_TYPES`` as move type ``"cls"``.

The ``[..., :in_dim]`` slice exists so a head sized for the readout alone drops
a zeroed ``target_feat`` tail (pointer Off) instead of reading dead dims; with a
pointer active the caller sizes ``in_dim`` to include it and the slice is a no-op.
"""

from __future__ import annotations

from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model._mlp import make_head_mlp
from qnn.model.move_head import MoveHeadInput, MoveHeadOutput

_MOVE_OUT_DIM = MOVE_AXES * MOVE_AXIS_CLASSES


class CLSMoveHead(nn.Module):
    """Move head: MLP over the leading ``in_dim`` features.

    No-GRU: in_dim=d_model reads the CLS self_readout (target_feat half is zeros).
    GRU:    in_dim=d_gru reads gru_flat (the GRU-integrated CLS; target_feat half zeros).
    """

    def __init__(self, *, in_dim: int, d_move: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, _MOVE_OUT_DIM, d_move, activation)

    def forward(self, inp: MoveHeadInput) -> MoveHeadOutput:
        feats = inp.features[..., : self.in_dim]
        logits = self.mlp(feats).reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        return MoveHeadOutput(logits=logits)


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("move", "cls")
def _build_move_cls(head, dims, d_model):
    return CLSMoveHead(in_dim=dims["motor_in"], d_move=head.d_hidden, activation=head.activation)
