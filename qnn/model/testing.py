"""Synthetic input builders for unit-testing model components.

One builder per component. Each returns the component's typed Input
dataclass populated with random tensors of the right shape, sized from
small defaults so tests run in milliseconds. Pass a ``seed`` for
deterministic output.

Pattern:

    from qnn.model.testing import make_head_features
    from qnn.model.look_head import PurePolarLookHead

    feats = make_head_features(batch=8, in_dim=128)
    out = PurePolarLookHead(128, 64, "gelu")(feats)
    assert out.look_predict.shape == (8, 3)

The builders intentionally produce *valid* inputs (e.g. actor masks have
at least one true entry per row, target_logits are pre-masked to -1e9
where actor_mask is False) so a component test failing means the
component is broken, not the fixture.
"""

from __future__ import annotations

import torch

from qnn.model.attack_with_head import AttackSelectorInput
from qnn.model.target import TargetPointerInput
from qnn.model.temporal import TemporalInput
from qnn.model.transformer import EncoderInput, ObsEmbedding
from qnn.schema import (
    ACTOR_SCALAR_DIM,
    SELF_SCALAR_DIM,
    SELF_STATE_SCALAR_DIM,
    SELF_ARSENAL_SCALAR_DIM,
    SELF_MOTION_SCALAR_DIM,
    SPATIAL_SCALAR_DIM,
    SPATIAL_TOKEN_COUNT,
    WEAPON_HEAD_SIZE,
)
from qnn.vocab import (
    MAX_ENTITY_EVENTS,
    MAX_TOKEN_OBJECTS,
    TOKEN_ACTOR,
    TOKEN_PROJECTILE,
)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _actor_mask(batch: int, n_entities: int, gen: torch.Generator) -> torch.Tensor:
    """Boolean mask with at least one True per row — pointer math assumes
    a non-empty actor set when computing softmax over masked logits."""
    mask = torch.rand(batch, n_entities, generator=gen) > 0.5
    # Guarantee at least one True per row.
    mask[:, 0] = True
    return mask


def _masked_logits(batch: int, n_entities: int, actor_mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Pre-masked logits: -1e9 where actor_mask is False (TargetPointer convention)."""
    logits = torch.randn(batch, n_entities, generator=gen)
    return logits.masked_fill(~actor_mask, -1e9)


def make_obs_dict(batch: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """Synthetic flat-batch obs dict matching the token observation schema.

    Shapes match what TransformerEncoder / Network expect at the
    ``(B*, ...)`` (post-flatten) layer. Use directly with ``EncoderInput``
    or hand to ``Network.forward`` for a smoke test.
    """
    gen = _gen(seed)
    N = MAX_TOKEN_OBJECTS
    obs: dict[str, torch.Tensor] = {
        "vel": torch.randn(batch, 3, generator=gen),
        "self_scalars": torch.randn(batch, SELF_SCALAR_DIM, generator=gen),
        "self_state_scalars":   torch.randn(batch, SELF_STATE_SCALAR_DIM,   generator=gen),
        "self_arsenal_scalars": torch.randn(batch, SELF_ARSENAL_SCALAR_DIM, generator=gen),
        "self_motion_scalars":  torch.randn(batch, SELF_MOTION_SCALAR_DIM,  generator=gen),
        "self_weapon_readiness": torch.rand(batch, WEAPON_HEAD_SIZE, generator=gen),
        "self_ammo_pools": torch.rand(batch, 4, generator=gen),
        "self_armor_type_id": torch.zeros(batch, 1, dtype=torch.long),
        "self_movement_id": torch.zeros(batch, 1, dtype=torch.long),
        "self_powerup_ids":         torch.zeros(batch, 5, dtype=torch.long),
        "self_state_powerup_ids":   torch.zeros(batch, 3, dtype=torch.long),
        "self_arsenal_powerup_ids": torch.zeros(batch, 1, dtype=torch.long),
        "self_motion_powerup_ids":  torch.zeros(batch, 1, dtype=torch.long),
        "entity_ids": torch.zeros(batch, N, 3, dtype=torch.long),
        "entity_types": torch.zeros(batch, N, dtype=torch.long),
        "entity_scalars_raw": torch.randn(batch, N, ACTOR_SCALAR_DIM, generator=gen) * 0.1,
        "entity_event_actions": torch.zeros(batch, N, MAX_ENTITY_EVENTS, dtype=torch.long),
        "entity_event_sources": torch.zeros(batch, N, MAX_ENTITY_EVENTS, dtype=torch.long),
        "entity_event_counts": torch.zeros(batch, N, dtype=torch.long),
        "spatial_scalars": torch.randn(batch, SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM, generator=gen) * 0.1,
        "action_history": torch.zeros(batch, 8, 8),
    }
    # Populate one slot per A27 combat token type so every projection
    # in ObsEmbedding sees a live input (otherwise gradient tests think
    # those Linears are dead). At least one actor with a forward rel
    # offset so downstream pointer/look math has signal.
    obs["entity_types"][:, 0] = TOKEN_ACTOR
    obs["entity_types"][:, 1] = TOKEN_PROJECTILE
    obs["entity_scalars_raw"][:, 0, 3:6] = torch.tensor([0.5, 0.0, 0.0])
    # Give at least one entity event so action_embed and event-projection
    # paths get exercised.
    obs["entity_event_counts"][:, 0] = 1
    obs["entity_event_actions"][:, 0, 0] = 1
    return obs


def make_encoder_input(
    batch: int = 4, *, seed: int = 0, d_model: int = 64, include_spatial: bool = True,
) -> EncoderInput:
    """Build a ready-to-encode ``EncoderInput`` from a synthetic obs dict.

    Materializes a fresh ``ObsEmbedding`` to tokenize — tests asserting
    encoder shape contracts should not depend on a specific embedding
    instance.
    """
    obs_embedding = ObsEmbedding(
        d_model=d_model, include_spatial=include_spatial,
    ).eval()
    return obs_embedding(make_obs_dict(batch, seed=seed))


def make_target_pointer_input(
    batch: int = 4,
    *,
    d_model: int = 64,
    n_entities: int = MAX_TOKEN_OBJECTS,
    seed: int = 0,
) -> TargetPointerInput:
    gen = _gen(seed)
    entity_mask = _actor_mask(batch, n_entities, gen)
    # Enemy mask: AND-of actor-mask with a coin-flip per slot so the
    # canonical pointer's softmax sees a non-empty enemy set on most rows.
    enemy_coin = torch.randint(0, 2, (batch, n_entities), generator=gen).bool()
    enemy_mask = entity_mask & enemy_coin
    return TargetPointerInput(
        entity_outs=torch.randn(batch, n_entities, d_model, generator=gen),
        entity_mask=entity_mask,
        enemy_mask=enemy_mask,
        self_readout=torch.randn(batch, d_model, generator=gen),
    )


def make_temporal_input(
    batch: int = 4,
    *,
    d_model: int = 64,
    seq_shape: tuple[int, int] | None = None,
    seed: int = 0,
) -> TemporalInput:
    """Flat by default. Pass ``seq_shape=(T, B)`` for a sequence input;
    ``batch`` is then ignored in favor of ``T*B`` from seq_shape."""
    gen = _gen(seed)
    if seq_shape is None:
        return TemporalInput(
            flat_pool=torch.randn(batch, d_model, generator=gen),
            hidden=None, reset_mask=None, seq_shape=None,
        )
    T, B = seq_shape
    return TemporalInput(
        flat_pool=torch.randn(T * B, d_model, generator=gen),
        hidden=None, reset_mask=None, seq_shape=(T, B),
    )


def make_head_features(batch: int = 4, *, in_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """The shared feature cat every non-selector head consumes (a28)."""
    return torch.randn(batch, in_dim, generator=_gen(seed))


def make_attack_selector_input(
    batch: int = 4,
    *,
    selector_dim: int = 128,
    seed: int = 0,
) -> AttackSelectorInput:
    gen = _gen(seed)
    return AttackSelectorInput(
        selector=torch.randn(batch, selector_dim, generator=gen),
    )


__all__ = [
    "make_obs_dict",
    "make_encoder_input",
    "make_target_pointer_input",
    "make_temporal_input",
    "make_head_features",
    "make_attack_selector_input",
]
