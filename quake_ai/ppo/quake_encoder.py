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

from quake_ai.model.transformer import TransformerTrunk


class QuakeTransformerEncoder(Encoder):
    """Transformer-based encoder with tokenised observation groups."""

    def __init__(self, cfg, obs_space) -> None:
        super().__init__(cfg)

        d_model: int = int(getattr(cfg, "quake_d_model", 64))
        n_heads: int = int(getattr(cfg, "quake_n_heads", 1))
        n_layers: int = int(getattr(cfg, "quake_n_layers", 2))
        ffn_dim: int = int(getattr(cfg, "quake_ffn_dim", 256))
        dropout: float = float(getattr(cfg, "quake_attn_dropout", 0.0))
        readout: str = str(getattr(cfg, "quake_readout", "self") or "self")
        action_history_tokens: int = int(getattr(cfg, "quake_action_history_tokens", 0))

        self.trunk = TransformerTrunk(
            obs_dim=0,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            readout=readout,
            action_history_tokens=action_history_tokens,
        )
        self.encoder_out_size: int = d_model

    def forward(self, obs_dict):
        return self.trunk(obs_dict)

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_quake_encoder(cfg, obs_space) -> Encoder:
    """Factory returning the transformer encoder."""
    return QuakeTransformerEncoder(cfg, obs_space)
