"""Custom PPO recurrent core that fuses current transformer state with memory."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.model.core import ModelCore, ModelCoreIdentity


class QuakeConcatRNNCore(ModelCore):
    """GRU core that returns ``concat(x_t, h_t)`` instead of memory alone."""

    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)

        self.cfg = cfg
        self.input_size = int(input_size)
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(self.input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(self.input_size, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.rnn_num_layers = int(cfg.rnn_num_layers)
        self.core_output_size = self.input_size + int(cfg.rnn_size)

    def forward(self, head_output, rnn_states):
        is_packed = isinstance(head_output, PackedSequence)
        is_batched_sequence = torch.is_tensor(head_output) and head_output.ndim == 3
        if is_packed or is_batched_sequence:
            core_input = head_output
        else:
            core_input = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            core_output, new_rnn_states = self.core(core_input, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            core_output, (h, c) = self.core(core_input, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if is_packed:
            fused = PackedSequence(
                torch.cat([head_output.data, core_output.data], dim=1),
                core_output.batch_sizes,
                core_output.sorted_indices,
                core_output.unsorted_indices,
            )
        elif is_batched_sequence:
            fused = torch.cat([head_output, core_output], dim=-1)
        else:
            fused = torch.cat([head_output, core_output.squeeze(0)], dim=1)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return fused, new_rnn_states


def make_quake_core(cfg, input_size: int):
    if cfg.use_rnn:
        return QuakeConcatRNNCore(cfg, input_size)
    return ModelCoreIdentity(cfg, input_size)
