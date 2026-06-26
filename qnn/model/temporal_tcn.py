"""Separable-TCN temporal component — a drop-in sibling of :class:`Temporal`.

Same I/O contract as the canonical GRU ``Temporal`` (``TemporalInput`` →
``TemporalOutput``) so it plugs straight into the ``temporal`` slot of
:class:`qnn.model.network.Network`. Where the GRU integrates the encoder pool
with a recurrent hidden state, this integrates it with a stack of **causal,
dilated, depthwise-separable** convolutions: a bounded receptive field instead
of unbounded (decaying) memory.

Receptive field
---------------
``RF = 1 + (kernel_size - 1) * sum(dilations)``. With ``kernel_size=3`` and
``dilations=(1,2,4,8,16,16)`` that is ``RF = 95`` (~96 frames), matching a
TBPTT window of 96. Each layer is depthwise (a per-channel ``k``-tap temporal
filter, ``k*C`` params) + pointwise (a 1x1 channel mix, ``C^2`` params), so a
6-layer stack at ``C=64`` is ~26k params — parity with the ``d_gru=64`` GRU.

Reset handling
--------------
Episode boundaries arrive via ``reset_mask`` (same as the GRU). A segment id
(cumulative reset count) gates every convolution tap: a tap from frame ``t-Δ``
into frame ``t`` contributes only when both are in the same segment, so the
receptive field never crosses an episode boundary.

Cross-chunk state (known limitation)
------------------------------------
The canonical BC loop carries the temporal state across TBPTT chunks in a
fixed ``(n_lanes, d_gru)`` buffer (``qnn.bc.supervised_loop``). A TCN's true
cross-chunk state is its last ``RF-1`` *input* frames (``(RF-1)*C`` floats),
which does not fit that buffer. So this module does **not** carry raw context
across chunk boundaries: each chunk is zero-left-padded at its start (a "cold
start"). Within a chunk it is exact; only the first ``RF-1`` frames after a
chunk boundary see less history than the stateful GRU would.

Consequence for parity ablations: run the GRU-vs-TCN comparison with
``tbptt_limit`` at least the typical episode length so chunk boundaries rarely
fall mid-episode, making the cold start negligible. True stateful-TCN parity
would require generalizing the loop's carry width to a per-temporal
``carry_dim`` — a canonical-BC change, deliberately out of scope here.
``next_hidden`` is returned as zeros shaped ``(B, hidden_dim)`` to satisfy the
loop's buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model.temporal import TemporalInput, TemporalOutput

# Re-exported so callers can build the input/output dataclasses without
# importing both modules.
__all__ = ["SeparableTCN", "TemporalInput", "TemporalOutput"]


@dataclass(frozen=True, slots=True)
class TCNConfig:
    """Separable-TCN architecture knobs. ``channels`` defaults to ``hidden_dim``."""
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 16)
    channels: int = 0  # 0 -> use hidden_dim
    separable: bool = True
    activation: str = "gelu"
    dropout: float = 0.0

    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "none":
        return nn.Identity()
    raise ValueError(f"activation must be 'gelu' / 'relu' / 'none', got {name!r}")


class _SepTCNLayer(nn.Module):
    """One residual block: causal dilated depthwise conv -> pointwise -> act.

    Operates on ``(B, L, C)`` with an explicit per-tap masked sum so that
    individual taps crossing an episode boundary can be zeroed (``F.conv1d``
    cannot mask individual taps). ``k`` taps means ``k`` shifts; cheap for the
    ``k=3`` stack used here. ``separable=False`` falls back to a dense per-tap
    channel mix (``k`` ``(C,C)`` matrices) for the same receptive field.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        *,
        separable: bool,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.separable = bool(separable)

        if self.separable:
            # Depthwise temporal filter: one k-tap filter per channel.
            self.depthwise_weight = nn.Parameter(torch.empty(channels, kernel_size))
            nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
            self.depthwise_bias = nn.Parameter(torch.zeros(channels))
            # Pointwise channel mix (1x1). Network._init_weights xavier-inits it.
            self.pointwise = nn.Linear(channels, channels)
        else:
            # Dense: a (C_out, C_in) mix per tap. Network._init_weights inits these.
            self.taps = nn.ModuleList(
                nn.Linear(channels, channels, bias=(j == kernel_size - 1))
                for j in range(kernel_size)
            )
            self.pointwise = nn.Identity()

        self.act = _activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _tap_valid(self, seg_id: torch.Tensor, delta: int) -> torch.Tensor:
        """(B, L) float mask: 1 where frame t and t-delta share a segment."""
        if delta == 0:
            return torch.ones_like(seg_id, dtype=torch.float32)
        seg_shift = F.pad(seg_id, (delta, 0), value=-1)[:, : seg_id.shape[1]]
        return (seg_id == seg_shift).to(torch.float32)

    def _shift(self, x: torch.Tensor, delta: int) -> torch.Tensor:
        """Shift along time so out[:, t] = x[:, t-delta], zero-padded at the front."""
        if delta == 0:
            return x
        return F.pad(x, (0, 0, delta, 0))[:, : x.shape[1], :]

    def forward(self, x: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C); seg_id: (B, L) int64
        k, dil = self.kernel_size, self.dilation
        acc = torch.zeros_like(x)
        for j in range(k):
            delta = (k - 1 - j) * dil  # j=k-1 is the current frame (delta 0)
            shifted = self._shift(x, delta)
            valid = self._tap_valid(seg_id, delta).unsqueeze(-1)
            if self.separable:
                acc = acc + shifted * self.depthwise_weight[:, j] * valid
            else:
                acc = acc + self.taps[j](shifted) * valid
        if self.separable:
            acc = acc + self.depthwise_bias
        y = self.dropout(self.act(self.pointwise(acc)))
        return x + y


class SeparableTCN(nn.Module):
    """Causal dilated separable-TCN with the :class:`Temporal` I/O contract.

    ``d_model`` is the encoder-pool width (input), ``hidden_dim`` is the output
    width (must equal ``d_gru`` so downstream head dims match the GRU path).
    Internal conv width is ``cfg.channels`` (defaults to ``hidden_dim``);
    in/out projections are added only when widths differ, so the parity config
    (``d_model == hidden_dim == channels == 64``) is pure conv stack (~26k).
    """

    def __init__(self, d_model: int, hidden_dim: int, cfg: TCNConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TCNConfig()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        # out_dim — width of flat_out; matches the GRU Temporal contract.
        self.out_dim = self.hidden_dim
        channels = int(cfg.channels) or self.hidden_dim
        self.channels = channels
        self.context_len = cfg.receptive_field() - 1
        self.cfg = cfg

        self.in_proj = nn.Linear(d_model, channels) if d_model != channels else None
        self.layers = nn.ModuleList(
            _SepTCNLayer(
                channels, cfg.kernel_size, dil,
                separable=cfg.separable, activation=cfg.activation, dropout=cfg.dropout,
            )
            for dil in cfg.dilations
        )
        self.out_proj = nn.Linear(channels, hidden_dim) if channels != hidden_dim else None

    def forward(self, inp: TemporalInput) -> TemporalOutput:
        flat_pool = inp.flat_pool
        if inp.seq_shape is None:
            seq_len, batch_size = 1, int(flat_pool.shape[0])
            pool_seq = flat_pool.unsqueeze(0)  # (1, B, d_model)
        else:
            seq_len, batch_size = inp.seq_shape
            pool_seq = flat_pool.reshape(seq_len, batch_size, self.d_model)

        # (B, T, C)
        x = pool_seq.permute(1, 0, 2)
        if self.in_proj is not None:
            x = self.in_proj(x)

        # Segment ids gate cross-boundary taps. reset_mask[t]=True => new
        # episode begins at t, so seg_id increments there (cumsum inclusive).
        if inp.reset_mask is None:
            seg_id = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=x.device)
        else:
            reset_bt = inp.reset_mask.to(torch.bool).reshape(seq_len, batch_size).permute(1, 0)
            seg_id = torch.cumsum(reset_bt.to(torch.int64), dim=1)

        h = x
        for layer in self.layers:
            h = layer(h, seg_id)
        if self.out_proj is not None:
            h = self.out_proj(h)  # (B, T, hidden_dim)

        if inp.seq_shape is None:
            flat_out = h[:, 0, :]  # (B, hidden_dim)
        else:
            flat_out = h.permute(1, 0, 2).reshape(seq_len * batch_size, self.hidden_dim)

        # No cross-chunk raw-state carry (see module docstring): satisfy the
        # loop's fixed (B, d_gru) buffer with zeros.
        next_hidden = torch.zeros(
            (batch_size, self.hidden_dim), dtype=flat_out.dtype, device=flat_out.device,
        )
        return TemporalOutput(flat_out=flat_out, next_hidden=next_hidden)
