"""Observation shape constants for the model tensor contract.

The C worker packs observations directly into the obs buffer. This module
provides dimension constants consumed by the model, environment, and PPO
wrappers. No Python-side encoding — just shape metadata.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from quake_ai.vocab import MODALITY_IDS, SUBJECT_IDS

MAX_OBJECT_TOKENS = 16
MAX_ENTITY_EVENTS = 4
ENTITY_EVENT_ID_DIM = 3  # subject, action, qualifier
SPATIAL_TOKEN_COUNT = 9
MAX_ROUTE_CLUSTERS = 8

ACTION_HISTORY_LEN = 8
ACTION_HISTORY_DIM = 7

SELF_SCALAR_DIM = 14
SELF_ID_DIM = 10  # weapon(1) + armor_type(1) + powerup_ids(5) + powerup_count(1) + movement(1) + cluster(1)
OBJECT_ID_DIM = 7
OBJECT_SCALAR_DIM = 17
SPATIAL_SCALAR_DIM = 10

# Flat obs_dim for Sample Factory observation space sizing.
OBS_DIM = int(
    SELF_SCALAR_DIM
    + SELF_ID_DIM
    + (MAX_OBJECT_TOKENS * (OBJECT_ID_DIM + OBJECT_SCALAR_DIM + 1 + MAX_ROUTE_CLUSTERS
                            + MAX_ENTITY_EVENTS * ENTITY_EVENT_ID_DIM + MAX_ENTITY_EVENTS + 1))
    + SPATIAL_TOKEN_COUNT
    + (SPATIAL_TOKEN_COUNT * SPATIAL_SCALAR_DIM)
    + (ACTION_HISTORY_LEN * ACTION_HISTORY_DIM)
)

_THREAT_SUBJECT_IDS = frozenset(
    {
        SUBJECT_IDS["PLAYER"],
        SUBJECT_IDS["PROJECTILE_NAIL"],
        SUBJECT_IDS["PROJECTILE_GRENADE"],
        SUBJECT_IDS["PROJECTILE_ROCKET"],
        SUBJECT_IDS["LIGHTNING_BEAM"],
    }
)

_THREAT_SUBJECT_IDS_ARRAY = np.fromiter(_THREAT_SUBJECT_IDS, dtype=np.int32)


def observation_signature_dim(obs: Mapping[str, np.ndarray]) -> int:
    return int(sum(int(np.prod(value.shape, dtype=np.int64)) for value in obs.values()))


def visible_threat_count(obs: Mapping[str, np.ndarray]) -> int:
    mask = np.asarray(obs["object_mask"], dtype=bool)
    ids = np.asarray(obs["object_ids"], dtype=np.int32)
    if mask.size == 0:
        return 0
    active = ids[mask]
    if active.size == 0:
        return 0
    subject_ids = active[:, 0]
    modality_ids = active[:, 2]
    return int(
        np.sum(
            np.isin(subject_ids, _THREAT_SUBJECT_IDS_ARRAY)
            & (modality_ids == MODALITY_IDS["VISUAL"])
        )
    )
