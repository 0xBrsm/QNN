"""Pre-attention encoder — slice tokens straight through, no attention.

Drop-in for ``TransformerEncoder`` (same ``EncoderInput`` →
``EncoderOutput`` contract) that does no self-attention mixing. Useful
for ablations that probe how much of the model's signal comes from the
input embedding alone vs the attention mixing.

Pair with an ``ObsEmbedding(include_spatial=False)`` obs-embedding slot in
``Network`` for the lean head-probe layout (self + entities, no spatial
sub-block). The encoder itself is stateless — all the input
construction lives in the obs embedding.
"""

from __future__ import annotations

from torch import nn

from qnn.model.transformer import EncoderInput, EncoderOutput


class PreAttnEncoder(nn.Module):
    def forward(self, inp: EncoderInput) -> EncoderOutput:
        tokens = inp.tokens
        return EncoderOutput(
            self_readout=tokens[:, inp.self_slice.start, :],
            entity_outs=tokens[:, inp.entity_slice, :],
            entity_mask=inp.entity_mask,
            self_block=tokens[:, inp.self_slice, :],
        )


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_encoder  # noqa: E402


@register_encoder("passthrough")
def _build_encoder_passthrough(encoder):
    return PreAttnEncoder()
