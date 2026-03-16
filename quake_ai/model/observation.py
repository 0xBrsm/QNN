"""Pack native token ticks into the model-facing tensor contract.

The active observation boundary is:
    - one normalized self token
    - padded object rows
    - padded global event atoms with object-owner indices
    - fixed spatial rows
    - rolling normalized action history

No embeddings are constructed here. This module only converts native semantic
tokens into stable numpy tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np

from engine.token_protocol import TrustedTokenTick
from quake_ai.actions import normalized_action_features
from quake_ai.vocab import MODALITY_IDS, SPATIAL_SECTOR_IDS, SUBJECT_IDS

MAX_OBJECT_TOKENS = 64
MAX_EVENT_ATOMS = 256
SPATIAL_TOKEN_COUNT = 9
MAX_ROUTE_CLUSTERS = 8

ACTION_HISTORY_LEN = 8
ACTION_HISTORY_DIM = 7

SELF_SCALAR_DIM = 23
SELF_ID_DIM = 3
OBJECT_ID_DIM = 5
OBJECT_SCALAR_DIM = 8
EVENT_ID_DIM = 4
EVENT_SCALAR_DIM = 3
SPATIAL_SCALAR_DIM = 10


_THREAT_SUBJECT_IDS = frozenset(
    {
        SUBJECT_IDS["PLAYER"],
        SUBJECT_IDS["PROJECTILE_NAIL"],
        SUBJECT_IDS["PROJECTILE_GRENADE"],
        SUBJECT_IDS["PROJECTILE_ROCKET"],
        SUBJECT_IDS["LIGHTNING_BEAM"],
    }
)

# Cached default spatial ids — constant across all ticks.
_DEFAULT_SPATIAL_IDS = np.asarray(
    [SPATIAL_SECTOR_IDS[name] for name in SPATIAL_SECTOR_IDS], dtype=np.int32
)




def observation_signature_dim(obs: Mapping[str, np.ndarray]) -> int:
    return int(sum(int(np.prod(value.shape, dtype=np.int64)) for value in obs.values()))


@dataclass(slots=True)
class TokenObservationEncoder:
    """Encode native token ticks into the shared tensor contract."""
    _action_history: list[list[float]] = field(default_factory=list)

    @property
    def obs_dim(self) -> int:
        return int(
            SELF_SCALAR_DIM
            + SELF_ID_DIM
            + (MAX_OBJECT_TOKENS * (OBJECT_ID_DIM + OBJECT_SCALAR_DIM + 1))
            + (MAX_EVENT_ATOMS * (EVENT_ID_DIM + EVENT_SCALAR_DIM + 2))
            + SPATIAL_TOKEN_COUNT
            + (SPATIAL_TOKEN_COUNT * SPATIAL_SCALAR_DIM)
            + (ACTION_HISTORY_LEN * ACTION_HISTORY_DIM)
        )

    @property
    def object_ids_shape(self) -> tuple[int, int]:
        return (MAX_OBJECT_TOKENS, OBJECT_ID_DIM)

    @property
    def object_scalars_shape(self) -> tuple[int, int]:
        return (MAX_OBJECT_TOKENS, OBJECT_SCALAR_DIM)

    @property
    def event_ids_shape(self) -> tuple[int, int]:
        return (MAX_EVENT_ATOMS, EVENT_ID_DIM)

    @property
    def event_scalars_shape(self) -> tuple[int, int]:
        return (MAX_EVENT_ATOMS, EVENT_SCALAR_DIM)

    @property
    def spatial_scalars_shape(self) -> tuple[int, int]:
        return (SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM)

    @property
    def action_history_shape(self) -> tuple[int, int]:
        return (ACTION_HISTORY_LEN, ACTION_HISTORY_DIM)

    def reset(self) -> None:
        self._action_history.clear()

    def encode(self, tick: TrustedTokenTick) -> Dict[str, np.ndarray]:
        if tick.reset:
            self.reset()

        # --- self token (single row, no loop) ---
        # v5: all scalars arrive pre-normalized from the C worker.
        st = tick.self_token
        self_cluster_id = int(st.cluster_id)
        self_scalars = np.array([
            st.health,
            st.armor,
            st.armor_type,
            *st.weapon_bits,
            st.weapon_super,
            *st.ammo,
            *st.velocity,
            st.yaw_sin, st.yaw_cos, st.pitch_sin, st.pitch_cos,
            st.dt,
        ], dtype=np.float32)

        # --- objects: bulk fill into pre-allocated arrays ---
        objects = tick.object_tokens
        n_obj = min(len(objects), MAX_OBJECT_TOKENS)

        object_ids = np.zeros((MAX_OBJECT_TOKENS, OBJECT_ID_DIM), dtype=np.int32)
        object_scalars = np.zeros((MAX_OBJECT_TOKENS, OBJECT_SCALAR_DIM), dtype=np.float32)
        object_mask = np.zeros(MAX_OBJECT_TOKENS, dtype=np.bool_)
        object_route_cluster_ids = np.zeros((MAX_OBJECT_TOKENS, MAX_ROUTE_CLUSTERS), dtype=np.int32)

        if n_obj > 0:
            ids_buf = np.empty((n_obj, OBJECT_ID_DIM), dtype=np.int32)
            sc_buf = np.empty((n_obj, OBJECT_SCALAR_DIM), dtype=np.float32)
            rc_buf = np.zeros((n_obj, MAX_ROUTE_CLUSTERS), dtype=np.int32)
            for i in range(n_obj):
                obj = objects[i]
                ids_buf[i, 0] = obj.subject_id
                ids_buf[i, 1] = obj.qualifier_id
                ids_buf[i, 2] = obj.modality_id
                ids_buf[i, 3] = obj.player_id
                ids_buf[i, 4] = int(obj.cluster_id)
                # v5: rel_x/y/z and route_cost arrive pre-normalized from C worker
                sc_buf[i, 0] = obj.rel_x
                sc_buf[i, 1] = obj.rel_y
                sc_buf[i, 2] = obj.rel_z
                sc_buf[i, 3] = obj.route_cost
                sc_buf[i, 4] = obj.recency
                sc_buf[i, 5] = obj.confidence
                sc_buf[i, 6] = obj.magnitude
                sc_buf[i, 7] = obj.state
                rc = getattr(obj, 'route_cluster_ids', None) or []
                for j in range(min(len(rc), MAX_ROUTE_CLUSTERS)):
                    rc_buf[i, j] = rc[j]
            object_ids[:n_obj] = ids_buf
            object_scalars[:n_obj] = sc_buf
            object_mask[:n_obj] = True
            object_route_cluster_ids[:n_obj] = rc_buf

        # --- events: flatten nested events, bulk fill ---
        event_ids = np.zeros((MAX_EVENT_ATOMS, EVENT_ID_DIM), dtype=np.int32)
        event_scalars = np.zeros((MAX_EVENT_ATOMS, EVENT_SCALAR_DIM), dtype=np.float32)
        event_owner = np.zeros(MAX_EVENT_ATOMS, dtype=np.int32)
        event_mask = np.zeros(MAX_EVENT_ATOMS, dtype=np.bool_)

        ei = 0
        for owner_idx in range(n_obj):
            for evt in objects[owner_idx].events:
                if ei >= MAX_EVENT_ATOMS:
                    break
                event_ids[ei, 0] = evt.subject_id
                event_ids[ei, 1] = evt.action_id
                event_ids[ei, 2] = evt.qualifier_id
                event_ids[ei, 3] = evt.modality_id
                event_scalars[ei, 0] = evt.recency
                event_scalars[ei, 1] = evt.confidence
                event_scalars[ei, 2] = evt.magnitude
                event_owner[ei] = owner_idx
                ei += 1
            if ei >= MAX_EVENT_ATOMS:
                break
        if ei > 0:
            np.clip(event_scalars[:ei], 0.0, 1.0, out=event_scalars[:ei])
            event_mask[:ei] = True

        # --- spatial: bulk fill ---
        spatials = tick.spatial_tokens
        n_sp = min(len(spatials), SPATIAL_TOKEN_COUNT)
        spatial_ids = _DEFAULT_SPATIAL_IDS.copy()
        spatial_scalars = np.zeros((SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM), dtype=np.float32)

        if n_sp > 0:
            sp_buf = np.empty((n_sp, SPATIAL_SCALAR_DIM), dtype=np.float32)
            for i in range(n_sp):
                sp = spatials[i]
                spatial_ids[i] = sp.sector_id
                sp_buf[i, 0] = sp.nearest_dist
                sp_buf[i, 1] = sp.mean_dist
                sp_buf[i, 2] = sp.openness
                sp_buf[i, 3] = sp.clearance
                sp_buf[i, 4] = sp.traversable
                sp_buf[i, 5] = sp.dropoff
                sp_buf[i, 6] = sp.solid_frac
                sp_buf[i, 7] = sp.water_frac
                sp_buf[i, 8] = sp.slime_frac
                sp_buf[i, 9] = sp.lava_frac
            # v5: spatial scalars arrive pre-normalized from C worker
            spatial_scalars[:n_sp] = sp_buf

        action_history = np.zeros((ACTION_HISTORY_LEN, ACTION_HISTORY_DIM), dtype=np.float32)
        if self._action_history:
            recent_history = self._action_history[-ACTION_HISTORY_LEN:]
            action_history[: len(recent_history)] = np.asarray(recent_history, dtype=np.float32)

        if tick.action_label and not tick.reset:
            self._action_history.append(normalized_action_features(tick.action_label))
            if len(self._action_history) > ACTION_HISTORY_LEN:
                del self._action_history[:-ACTION_HISTORY_LEN]

        return {
            "self_scalars": self_scalars,
            "self_weapon_id": np.array([st.weapon_id], dtype=np.int32),
            "self_movement_id": np.array([st.movement_id], dtype=np.int32),
            "self_cluster_id": np.array([self_cluster_id], dtype=np.int32),
            "object_ids": object_ids,
            "object_scalars": object_scalars,
            "object_mask": object_mask,
            "object_route_cluster_ids": object_route_cluster_ids,
            "event_ids": event_ids,
            "event_scalars": event_scalars,
            "event_owner": event_owner,
            "event_mask": event_mask,
            "spatial_ids": spatial_ids,
            "spatial_scalars": spatial_scalars,
            "action_history": action_history,
        }


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
            np.isin(subject_ids, np.fromiter(_THREAT_SUBJECT_IDS, dtype=np.int32))
            & (modality_ids == MODALITY_IDS["VISUAL"])
        )
    )
