"""Temporal component: per-step feature mixer over the time axis.

Wraps the recurrence (or alternative — e.g. TCN, attention window) so
the orchestrator can stay flat. The component owns the seq-vs-flat
branching internally:

  * Flat input  (seq_shape=None): single-step on (1, B, d_model), squeezes back.
  * Sequence    (seq_shape=(T,B)): reshapes flat pool to (T, B, d_model),
                                   runs the recurrence (with optional
                                   per-step reset_mask), reshapes output
                                   back to flat (T*B, hidden_dim).

Default impl is a single-layer GRU. Alternatives (TCN, attention-window,
identity pass-through) would be sibling classes with the same I/O contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TemporalInput:
    flat_pool: torch.Tensor                       # (B*, d_model)
    hidden: torch.Tensor | None                   # (1, B, H) or (B, H), last hidden state
    reset_mask: torch.Tensor | None               # (T*B,) bool — episode boundaries
    seq_shape: tuple[int, int] | None             # (T, B) if sequence, None if flat
    # Sorted host-known timesteps where any lane resets. The mask remains the
    # source of truth for which lanes reset at each boundary.
    reset_ts: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class TemporalOutput:
    flat_out: torch.Tensor      # (B*, hidden_dim)
    next_hidden: torch.Tensor   # (B, hidden_dim) — carry-forward for next call


class Temporal(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        # out_dim — width of flat_out, the temporal feature handed downstream.
        self.out_dim = self.hidden_dim
        self.gru = nn.GRU(self.d_model, self.hidden_dim, batch_first=False)

    def forward(self, inp: TemporalInput) -> TemporalOutput:
        if inp.seq_shape is None:
            # Flat: single-step on (1, B, d_model).
            batch_size = int(inp.flat_pool.shape[0])
            h0 = self._initial_hidden(
                inp.hidden, batch_size,
                dtype=inp.flat_pool.dtype, device=inp.flat_pool.device,
            )
            step, h_final = self.gru(inp.flat_pool.unsqueeze(0), h0)
            return TemporalOutput(
                flat_out=step.squeeze(0),
                next_hidden=h_final.squeeze(0),
            )

        seq_len, batch_size = inp.seq_shape
        pool_seq = inp.flat_pool.reshape(seq_len, batch_size, self.d_model)
        h0 = self._initial_hidden(
            inp.hidden, batch_size,
            dtype=pool_seq.dtype, device=pool_seq.device,
        )
        if inp.reset_mask is None:
            out_seq, h_final = self.gru(pool_seq, h0)
        else:
            reset_seq = inp.reset_mask.to(
                device=pool_seq.device, dtype=torch.bool,
            ).reshape(seq_len, batch_size)
            boundary_ts = (
                [int(t) for t in inp.reset_ts]
                if inp.reset_ts is not None
                else reset_seq.any(dim=1).nonzero().flatten().tolist()
            )
            if not boundary_ts or boundary_ts[0] != 0:
                boundary_ts = [0, *boundary_ts]
            boundary_ts.append(seq_len)
            h = h0
            outs = []
            for start, end in zip(boundary_ts[:-1], boundary_ts[1:]):
                reset_t = reset_seq[start].view(1, batch_size, 1)
                h = h.masked_fill(reset_t, 0.0)
                out_segment, h = self.gru(pool_seq[start:end], h)
                outs.append(out_segment)
            out_seq = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
            h_final = h
        return TemporalOutput(
            flat_out=out_seq.reshape(seq_len * batch_size, self.hidden_dim),
            next_hidden=h_final.squeeze(0),
        )

    def _initial_hidden(
        self,
        hidden: torch.Tensor | None,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if hidden is None:
            return torch.zeros((1, batch_size, self.hidden_dim), dtype=dtype, device=device)
        h = hidden.to(device=device, dtype=dtype)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        return h.contiguous()


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_temporal  # noqa: E402


@register_temporal("gru")
def _build_temporal_gru(temporal, d_model):
    return Temporal(d_model, temporal.d_gru)
