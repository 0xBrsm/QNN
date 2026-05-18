"""Custom PPO recurrent core that fuses current transformer state with memory."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.model.core import ModelCore, ModelCoreIdentity


class QuakeConcatRNNCore(ModelCore):
    """GRU core that splits cat(self, pool, target_feat) and fuses
    cat(self, h_t, target_feat).

    The encoder emits three d_model summaries:
      * ``self_readout`` — transformer self-token readout ("the now")
      * ``pool``         — self + mean(actors) projection, fed to the GRU
      * ``target_feat``  — attention-pooled entity feature for head conditioning

    The core routes pool through the GRU and passes self_readout and
    target_feat through unchanged.  Action heads downstream read
    cat(self_readout, h_t, target_feat).
    """

    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)

        self.cfg = cfg
        self.input_size = int(input_size)
        # Encoder packs three d_model-sized vectors in order.
        assert self.input_size % 3 == 0, (
            f"QuakeConcatRNNCore expects encoder output divisible by 3 (self, pool, target_feat); got {self.input_size}"
        )
        self.d_model = self.input_size // 3
        self.self_dim = self.d_model
        self.pool_dim = self.d_model
        self.target_dim = self.d_model
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(self.pool_dim, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(self.pool_dim, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.rnn_num_layers = int(cfg.rnn_num_layers)
        # Heads see self + gru_out + target_feat.
        self.core_output_size = self.self_dim + int(cfg.rnn_size) + self.target_dim

    def _split(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.split(t, [self.self_dim, self.pool_dim, self.target_dim], dim=-1)

    def forward(self, head_output, rnn_states):
        is_packed = isinstance(head_output, PackedSequence)
        is_batched_sequence = torch.is_tensor(head_output) and head_output.ndim == 3
        if is_packed:
            self_data, pool_data, target_data = self._split(head_output.data)
            core_input = PackedSequence(
                pool_data,
                head_output.batch_sizes,
                head_output.sorted_indices,
                head_output.unsorted_indices,
            )
        elif is_batched_sequence:
            self_readout, pool, target_feat = self._split(head_output)
            core_input = pool
        else:
            self_readout, pool, target_feat = self._split(head_output)
            core_input = pool.unsqueeze(0)

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
                torch.cat([self_data, core_output.data, target_data], dim=1),
                core_output.batch_sizes,
                core_output.sorted_indices,
                core_output.unsorted_indices,
            )
        elif is_batched_sequence:
            fused = torch.cat([self_readout, core_output, target_feat], dim=-1)
        else:
            fused = torch.cat([self_readout, core_output.squeeze(0), target_feat], dim=1)

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
