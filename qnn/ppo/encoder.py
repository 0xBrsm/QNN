"""PPO encoder wrapping the native token transformer trunk."""

from __future__ import annotations

import numpy as np
import torch.serialization

# SF checkpoints contain numpy scalars.  PyTorch 2.6+ defaults
# weights_only=True which rejects them.  Allowlist here because SF
# spawns the learner as a separate process and this module is imported
# by the learner subprocess when it builds the model (before loading
# checkpoints).
torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])

try:
    from sample_factory.model.encoder import Encoder
except ImportError as exc:
    raise ImportError("sample-factory is required: pip install sample-factory>=2.0.0") from exc

from qnn.model.transformer import TransformerTrunk


class QuakeTransformerEncoder(Encoder):
    """Transformer-based encoder with tokenised observation groups."""

    def __init__(self, cfg, obs_space) -> None:
        super().__init__(cfg)

        d_model: int = int(getattr(cfg, "quake_d_model", 64))
        n_heads: int = int(getattr(cfg, "quake_n_heads", 1))
        n_layers: int = int(getattr(cfg, "quake_n_layers", 2))
        ffn_dim: int = int(getattr(cfg, "quake_ffn_dim", 256))
        dropout: float = float(getattr(cfg, "quake_attn_dropout", 0.0))

        self.trunk = TransformerTrunk(
            obs_dim=0,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.use_rnn: bool = bool(getattr(cfg, "use_rnn", False))
        self.d_model: int = d_model
        self.encoder_out_size: int = 2 * d_model

    def forward(self, obs_dict):
        import torch as _torch
        # PPO does not use target_logits (no labels during rollout); the
        # TargetPointer still runs inside the trunk so gradient can flow via
        # target_feat.  Discard target_logits here.
        self_readout, target_feat, _target_logits, _target_query = self.trunk(obs_dict)
        return _torch.cat([self_readout, target_feat], dim=-1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_quake_encoder(cfg, obs_space) -> Encoder:
    """Factory returning the transformer encoder."""
    return QuakeTransformerEncoder(cfg, obs_space)
