"""Custom PPO recurrent core that runs the GRU on the encoder's self readout.

Matches the BC ``Network`` temporal layout: the transformer's ``cls_readout``
slot is the per-step pool fed to the GRU; ``target_feat`` flows past the
GRU unchanged. The fused output passed to the action / value heads is
``cat(cls_readout, h_t, target_feat)`` — a strict superset of BC's motor
features (``cat(h_t, target_feat)``), so warm-starting from a BC encoder
+ GRU stays meaningful and the new ``cls_readout`` slice picks up its
gradient signal from PPO.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.model.core import ModelCore, ModelCoreIdentity


class QuakeConcatRNNCore(ModelCore):
    """RNN core that splits the encoder's ``cat(cls_readout, target_feat)``
    output, runs the GRU on ``cls_readout``, and fuses
    ``cat(cls_readout, h_t, target_feat)`` for downstream heads.
    """

    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)

        self.cfg = cfg
        self.input_size = int(input_size)
        # Encoder packs two d_model-sized vectors in order: cls_readout, target_feat.
        assert self.input_size % 2 == 0, (
            f"QuakeConcatRNNCore expects encoder output divisible by 2 "
            f"(cls_readout, target_feat); got {self.input_size}"
        )
        self.d_model = self.input_size // 2
        self.self_dim = self.d_model
        self.target_dim = self.d_model
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(self.self_dim, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(self.self_dim, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.rnn_num_layers = int(cfg.rnn_num_layers)
        # Heads see cls_readout + gru_out + target_feat.
        self.core_output_size = self.self_dim + int(cfg.rnn_size) + self.target_dim

    def _split(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.split(t, [self.self_dim, self.target_dim], dim=-1)

    def forward(self, head_output, rnn_states):
        is_packed = isinstance(head_output, PackedSequence)
        is_batched_sequence = torch.is_tensor(head_output) and head_output.ndim == 3
        if is_packed:
            self_data, target_data = self._split(head_output.data)
            core_input = PackedSequence(
                self_data,
                head_output.batch_sizes,
                head_output.sorted_indices,
                head_output.unsorted_indices,
            )
        elif is_batched_sequence:
            cls_readout, target_feat = self._split(head_output)
            core_input = cls_readout
        else:
            cls_readout, target_feat = self._split(head_output)
            core_input = cls_readout.unsqueeze(0)

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
            fused = torch.cat([cls_readout, core_output, target_feat], dim=-1)
        else:
            fused = torch.cat([cls_readout, core_output.squeeze(0), target_feat], dim=1)

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
