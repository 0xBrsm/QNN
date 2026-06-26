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
            h = h0
            outs = []
            for t in range(seq_len):
                reset_t = reset_seq[t].view(1, batch_size, 1)
                h = h.masked_fill(reset_t, 0.0)
                out_t, h = self.gru(pool_seq[t:t + 1], h)
                outs.append(out_t)
            out_seq = torch.cat(outs, dim=0)
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
