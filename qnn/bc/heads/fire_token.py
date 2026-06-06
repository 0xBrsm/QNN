"""Fire-head probe with trunk-style token projections, no transformer or GRU.

Same loss / metrics / label as ``qnn.bc.heads.fire`` (binary BCE on
``act.fire``), but the encoder is replaced by the trunk's own
projections + embeddings via ``TokenizedFeatureEncoder``. The encoder
runs ``qnn.model.transformer.Tokenizer`` and soft-pools the resulting
actor entity tokens by the labeler's GT slot distribution, returning
``cat(self_token, target_feat)`` — i.e. exactly what the full BC model
would feed to the fire head if you switched off the transformer trunk
and the GRU.

This is an *oracle-pointer* probe: the GT target distribution is
privileged and not available at inference. The result is therefore an
upper bound on what a head-only fire model can do given perfect target
identification, isolating head architecture from pointer error.

The module's forward signature matches
``qnn.model.policy._CombatObjectiveNet.forward`` so the canonical BC
supervised loop can drive it without modification. ``QNNPolicy``
constructs the head via its ``model_factory`` hook; ``logits`` only
carries the ``fire`` head, and the canonical loss path keys off ``head
in logits`` so every other head loss is skipped automatically.
``target_logits`` / ``target_query`` are returned as zero placeholders
— the runner sets ``head_loss_weights["target"]`` to 0 so the target
loss path runs but contributes zero gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from qnn.bc.heads.fire import fire_bce_loss, fire_metrics
from qnn.bc.heads.spec import (
    HeadBuildResult,
    HeadLossSpec,
    HeadSpec,
    neutral_model_config,
)
from qnn.bc.heads.tokenized import TokenizedFeatureEncoder
from qnn.model.policy import FIRE_HEAD, FIRE_HEAD_SIZE, ModelConfig
from qnn.vocab import MAX_TOKEN_OBJECTS


class FireTokenHead(nn.Module):
    """Tokenized encoder + configurable MLP fire head; matches the BC forward contract.

    Forward returns the same ``(features, logits, values, next_hidden,
    target_logits, target_query)`` tuple ``_CombatObjectiveNet`` does:

      * ``features``     — encoder output (``2 * d_model`` per frame)
      * ``logits``       — ``{"fire": (B, 1)}`` only; canonical loss code
                           skips heads not present in this dict.
      * ``values``       — zeros (no value head in BC anyway).
      * ``next_hidden``  — ``(B, 0)`` placeholder; no GRU.
      * ``target_logits``— zeros ``(B, MAX_TOKEN_OBJECTS)``; the target
                           loss path triggers off this being non-None.
                           Runner gates the target weight to 0.
      * ``target_query`` — zeros ``(B, d_model)``; only used by the
                           ``target_pid_aux`` path which the runner does
                           not enable.
    """

    def __init__(
        self,
        *,
        d_model: int,
        self_weapon_embed_in_self: bool,
        strip_self_embeds: bool,
        weapon_embed_concat: bool,
        hidden: int,
        n_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = TokenizedFeatureEncoder(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon_embed_in_self,
            strip_self_embeds=strip_self_embeds,
            weapon_embed_concat=weapon_embed_concat,
        )
        self.d_model = self.encoder.d_model
        layers: list[nn.Module] = []
        prev = self.encoder.output_dim
        for _ in range(int(n_hidden_layers)):
            layers += [nn.Linear(prev, int(hidden)), nn.GELU(), nn.Dropout(float(dropout))]
            prev = int(hidden)
        layers.append(nn.Linear(prev, FIRE_HEAD_SIZE))
        self.head = nn.Sequential(*layers)

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
        # Memoryless probe — hidden / reset_mask / target_gt /
        # prev_target_dist are accepted to match the BC trainer's call
        # signature; ignored here.
        del hidden, reset_mask, target_gt, prev_target_dist

        if target_dist_slot is None:
            raise RuntimeError(
                "FireTokenHead requires target_dist_slot — the canonical "
                "BC supervised loop must pass it (supervised_step and "
                "evaluate_supervised both derive it from actions.target_dist)."
            )

        sample = obs["self_scalars"]
        is_seq = sample.ndim == 3
        if is_seq:
            T = int(sample.shape[0])
            B = int(sample.shape[1])
            flat_obs: Dict[str, torch.Tensor] = {
                k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()
            }
            tds_flat = target_dist_slot.reshape(T * B, target_dist_slot.shape[-1])
            BB = T * B
        else:
            B = int(sample.shape[0])
            T = 0
            flat_obs = obs
            tds_flat = target_dist_slot
            BB = B

        features_flat = self.encoder(flat_obs, tds_flat)            # (BB, 2D)
        fire_flat = self.head(features_flat)                         # (BB, 1)

        device = features_flat.device
        dtype = features_flat.dtype
        target_logits_flat = torch.zeros((BB, MAX_TOKEN_OBJECTS), dtype=dtype, device=device)
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

        logits: Dict[str, torch.Tensor] = {FIRE_HEAD: fire_logits}
        return features, logits, values, next_hidden, target_logits, target_query


def _build_fire_token(probe: Mapping[str, Any]) -> HeadBuildResult:
    """HeadBuilder for ``fire_token`` — every key required in probe.json.

    Reads probe.json knobs:

      d_model (int): tokenizer width.
      self_weapon_embed_in_self (bool).
      strip_self_embeds (bool): replace tokenizer self_token with raw
        ``Linear(self_scalars)`` — no kind/armor/movement/powerup/weapon
        adds.
      weapon_embed_concat (bool): concat dedicated weapon embed between
        self and target_feat. Mutually exclusive with
        self_weapon_embed_in_self.
      hidden (int): MLP hidden width.
      n_hidden_layers (int): MLP depth.
      dropout (float): MLP dropout (set 0.0 to disable).

    Returns the (model_config, model_factory) tuple the runner feeds to
    run_behavior_cloning. The head-loss-weights map that limits
    canonical losses to fire BCE lives in train.json, like every other
    BC training knob.
    """
    d_model = int(_required(probe, "d_model"))
    self_weapon = bool(_required(probe, "self_weapon_embed_in_self"))
    strip_self_embeds = bool(_required(probe, "strip_self_embeds"))
    weapon_embed_concat = bool(_required(probe, "weapon_embed_concat"))
    hidden = int(_required(probe, "hidden"))
    n_hidden_layers = int(_required(probe, "n_hidden_layers"))
    dropout = float(_required(probe, "dropout"))

    model_config = neutral_model_config(
        d_model=d_model, self_weapon_embed_in_self=self_weapon,
    )

    def factory(obs_dim: int, model_cfg: ModelConfig) -> FireTokenHead:
        del obs_dim, model_cfg
        return FireTokenHead(
            d_model=d_model,
            self_weapon_embed_in_self=self_weapon,
            strip_self_embeds=strip_self_embeds,
            weapon_embed_concat=weapon_embed_concat,
            hidden=hidden,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
        )

    return model_config, factory


def _required(probe: Mapping[str, Any], key: str) -> Any:
    if key not in probe:
        raise RuntimeError(
            f"probe.json must define {key!r} for head=fire_token "
            "(no Python-level defaults — see qnn.bc.heads.templates)."
        )
    return probe[key]


FIRE_TOKEN = HeadSpec(
    name="fire",
    loss=HeadLossSpec(
        loss_fn=fire_bce_loss,
        metrics_fn=fire_metrics,
        label_key="fire",
        output_dim=1,
        selection_metric="f1_fire",
        selection_lower_is_better=False,
    ),
    build=_build_fire_token,
)
