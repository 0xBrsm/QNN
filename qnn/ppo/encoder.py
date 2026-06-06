"""PPO encoder wrapping the native token transformer encoder."""

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

from qnn.model.target import TargetPointer, TargetPointerInput
from qnn.model.transformer import ObsEmbedding, TransformerEncoder
from qnn.schema import WEAPON_HEAD_SIZE


class QuakeTransformerEncoder(Encoder):
    """Transformer-based encoder with tokenised observation groups."""

    def __init__(self, cfg, obs_space) -> None:
        super().__init__(cfg)

        d_model: int = int(getattr(cfg, "quake_d_model"))
        n_heads: int = int(getattr(cfg, "quake_n_heads"))
        n_layers: int = int(getattr(cfg, "quake_n_layers"))
        ffn_dim: int = int(getattr(cfg, "quake_ffn_dim"))
        dropout: float = float(getattr(cfg, "quake_attn_dropout"))
        self_weapon_embed_in_self: bool = bool(getattr(cfg, "quake_self_weapon_embed_in_self", False))

        self.obs_embedding = ObsEmbedding(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon_embed_in_self,
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        # TargetPointer is no longer embedded in TransformerEncoder; the
        # PPO wrapper owns its own pointer so it can soft-pool target_feat
        # for the value/policy heads. Plain config — no GT teacher forcing,
        # no weapon injection in the query (PPO has no labels at rollout).
        self.target_pointer = TargetPointer(
            d_model=d_model,
            query_in_dim=d_model,
            inject_weapon=False,
            weapon_vocab=WEAPON_HEAD_SIZE,
            hard_target=False,
            linear_idx_prior=False,
            gt_dist_target_feat=False,
            prev_target_in_query=False,
        )
        self.use_rnn: bool = bool(getattr(cfg, "use_rnn", False))
        self.d_model: int = d_model
        self.encoder_out_size: int = 2 * d_model

    def forward(self, obs_dict):
        import torch as _torch
        enc_out = self.encoder(self.obs_embedding(obs_dict))
        tp_out = self.target_pointer(TargetPointerInput(
            query=enc_out.self_readout,
            entity_outs=enc_out.entity_outs,
            entity_mask=enc_out.entity_mask,
        ))
        return _torch.cat([enc_out.self_readout, tp_out.target_feat], dim=-1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_quake_encoder(cfg, obs_space) -> Encoder:
    """Factory returning the transformer encoder."""
    return QuakeTransformerEncoder(cfg, obs_space)
