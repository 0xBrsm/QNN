"""Flat-feature head probe — generic MLP over named per-frame features.

Shared model class for ``fire`` and ``target`` flat-feature probes.
Wraps the ``qnn.bc.heads.features`` torch registry: each probe declares
a feature list (e.g. ``("self_health_armor", "target_pooled_rel", …)``)
and an MLP shape, and ``FlatFeatureHead`` extracts those features from
the canonical BC obs dict per batch and routes the MLP output into the
appropriate slot of the BC forward contract.

* The fire probe routes its output to ``logits["fire"]`` so the
  canonical fire BCE path computes the loss.
* The target probe routes to ``target_logits`` (and leaves ``logits``
  empty) so the canonical target soft-CE path computes the loss against
  ``actions["target_dist"]``.

Either way the rest of the BC pipeline — shard streaming, eval cadence,
checkpointing, history — runs unchanged. Same canonical entry point as
``fire_token``: no parallel pipeline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from qnn.bc.heads.features import build_feature_vector, feature_vector_dim
from qnn.model.policy import FIRE_HEAD, FIRE_HEAD_SIZE
from qnn.vocab import MAX_TOKEN_OBJECTS


_TARGET_ROUTE = "target"
_FIRE_ROUTE = "fire"


class FlatFeatureHead(nn.Module):
    """Flat-feature MLP that matches the BC supervised-loop forward contract.

    ``feature_names`` selects builders from ``qnn.bc.heads.features``;
    ``output_route`` is either ``"fire"`` (write to ``logits['fire']``,
    size 1) or ``"target"`` (write to ``target_logits``, size
    ``MAX_TOKEN_OBJECTS``).

    All non-routed return tensors are zero placeholders; the canonical
    loss path skips heads not present in ``logits`` (and the runner
    zeros their head-loss weights anyway).
    """

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        output_route: str,
        hidden: int,
        n_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if output_route not in (_FIRE_ROUTE, _TARGET_ROUTE):
            raise ValueError(
                f"output_route must be {_FIRE_ROUTE!r} or {_TARGET_ROUTE!r}, "
                f"got {output_route!r}"
            )
        self.feature_names = tuple(feature_names)
        self.output_route = output_route
        self.in_dim = feature_vector_dim(self.feature_names)
        self.output_dim = (
            FIRE_HEAD_SIZE if output_route == _FIRE_ROUTE else MAX_TOKEN_OBJECTS
        )
        layers: list[nn.Module] = []
        prev = self.in_dim
        for _ in range(int(n_hidden_layers)):
            layers += [nn.Linear(prev, int(hidden)), nn.GELU(), nn.Dropout(float(dropout))]
            prev = int(hidden)
        layers.append(nn.Linear(prev, self.output_dim))
        self.head = nn.Sequential(*layers)
        # d_model is reported so QNNPolicy's runtime-shape computations
        # (self.head_hidden etc.) have something to read. ModelConfig
        # already carries the canonical d_model.
        self.d_model = max(1, int(prev))

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        target_gt: torch.Tensor | None = None,
        target_dist_slot: torch.Tensor | None = None,
        prev_target_dist: torch.Tensor | None = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del hidden, reset_mask, target_gt, prev_target_dist

        sample = obs["self_scalars"]
        is_seq = sample.ndim == 3
        if is_seq:
            T = int(sample.shape[0])
            B = int(sample.shape[1])
            flat_obs: Dict[str, torch.Tensor] = {
                k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()
            }
            tds_flat = (
                target_dist_slot.reshape(T * B, target_dist_slot.shape[-1])
                if target_dist_slot is not None else None
            )
            BB = T * B
        else:
            B = int(sample.shape[0])
            T = 0
            flat_obs = obs
            tds_flat = target_dist_slot
            BB = B

        features_flat = build_feature_vector(self.feature_names, flat_obs, tds_flat)  # (BB, F)
        out_flat = self.head(features_flat)                                            # (BB, D_out)

        device = features_flat.device
        dtype = features_flat.dtype
        # Default placeholders for the two output slots; the routed slot
        # gets overwritten below.
        fire_flat = torch.zeros((BB, FIRE_HEAD_SIZE), dtype=dtype, device=device)
        target_logits_flat = torch.zeros((BB, MAX_TOKEN_OBJECTS), dtype=dtype, device=device)

        if self.output_route == _FIRE_ROUTE:
            fire_flat = out_flat
        else:
            target_logits_flat = out_flat

        target_query_flat = torch.zeros((BB, self.d_model), dtype=dtype, device=device)
        values_flat = torch.zeros((BB,), dtype=dtype, device=device)

        if is_seq:
            features = features_flat.reshape(T, B, -1)
            fire_logits = fire_flat.reshape(T, B, FIRE_HEAD_SIZE)
            target_logits = target_logits_flat.reshape(T, B, MAX_TOKEN_OBJECTS)
            target_query = target_query_flat.reshape(T, B, self.d_model)
            values = values_flat.reshape(T, B)
            next_hidden = torch.zeros((B, 0), dtype=dtype, device=device)
        else:
            features = features_flat
            fire_logits = fire_flat
            target_logits = target_logits_flat
            target_query = target_query_flat
            values = values_flat
            next_hidden = torch.zeros((B, 0), dtype=dtype, device=device)

        # logits dict only carries the fire head when this is a fire
        # probe; target_logits flows through the separate return slot
        # the canonical model uses for the pointer.
        logits: Dict[str, torch.Tensor] = (
            {FIRE_HEAD: fire_logits} if self.output_route == _FIRE_ROUTE else {}
        )
        return features, logits, values, next_hidden, target_logits, target_query
