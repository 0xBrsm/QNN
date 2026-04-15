"""Observation shape constants for the model tensor contract.

v9: Per-type entity tokens with variable-length wire format.
The C worker packs observations with type-tagged entity tokens.
This module provides dimension constants consumed by the model.
"""

from __future__ import annotations

from qnn.vocab import (
    ENTITY_VOCAB_SIZE,
    ACTION_VOCAB_SIZE,
    MODALITY_VOCAB_SIZE,
    MAX_PLAYER_SLOTS,
    MAX_TOKEN_OBJECTS,
    MAX_ENTITY_EVENTS,
    PROJECTILE_SCALAR_DIM,
    ACTOR_SCALAR_DIM,
    ITEM_SCALAR_DIM,
    MOVER_SCALAR_DIM,
    PROJECTILE_ID_DIM,
    ACTOR_ID_DIM,
    ITEM_ID_DIM,
    MOVER_ID_DIM,
)

SPATIAL_TOKEN_COUNT = 9
SPATIAL_SCALAR_DIM = 13

ACTION_HISTORY_LEN = 8
ACTION_HISTORY_DIM = 8  # move[3] + look[3] + fire + switch

SELF_SCALAR_DIM = 14

from qnn.wire import MAX_ENTITY_SCALAR_DIM, MAX_ENTITY_ID_DIM

# v9: obs_dim is unused by the transformer trunk (it uses token dicts).
# Kept as 0 for callers that still reference it (e.g. bc_train model init).
OBS_DIM = 0

# Canonical obs shape schema — single source of truth for env, checkpoint, and adapter.
OBS_SCHEMA: dict[str, tuple[int, ...]] = {
    "self_scalars": (SELF_SCALAR_DIM,),
    "self_weapon_id": (1,),
    "self_armor_type_id": (1,),
    "self_powerup_ids": (5,),
    "self_movement_id": (1,),
    "entity_types": (MAX_TOKEN_OBJECTS,),
    "entity_scalars_raw": (MAX_TOKEN_OBJECTS, MAX_ENTITY_SCALAR_DIM),
    "entity_ids": (MAX_TOKEN_OBJECTS, MAX_ENTITY_ID_DIM),
    "entity_event_actions": (MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS),
    "entity_event_sources": (MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS),
    "entity_event_counts": (MAX_TOKEN_OBJECTS,),
    "spatial_scalars": (SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM),
    "action_history": (ACTION_HISTORY_LEN, ACTION_HISTORY_DIM),
}
