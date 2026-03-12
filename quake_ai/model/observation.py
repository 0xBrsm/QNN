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
import math
from typing import Dict, Mapping

import numpy as np

from engine.token_protocol import TrustedTokenTick
from quake_ai.actions import normalized_action_features
from quake_ai.vocab import MODALITY_IDS, SPATIAL_SECTOR_IDS, SUBJECT_IDS

MAX_OBJECT_TOKENS = 128
MAX_EVENT_ATOMS = 256
SPATIAL_TOKEN_COUNT = 9

ACTION_HISTORY_LEN = 8
ACTION_HISTORY_DIM = 7

SELF_SCALAR_DIM = 27
OBJECT_ID_DIM = 4
OBJECT_SCALAR_DIM = 7
EVENT_ID_DIM = 4
EVENT_SCALAR_DIM = 3
SPATIAL_SCALAR_DIM = 10

_ABS_POS_SCALE = 4096.0
_REL_POS_SCALE = 4096.0
_SPATIAL_DIST_SCALE = 1024.0
_VEL_SCALE = 2000.0
_HEALTH_SCALE = 250.0
_ARMOR_SCALE = 200.0
_ARMOR_TYPE_SCALE = 0.8
_AMMO_CAPS = (100.0, 200.0, 100.0, 100.0)
_DT_SCALE = 0.1

_THREAT_SUBJECT_IDS = frozenset(
    {
        SUBJECT_IDS["PLAYER"],
        SUBJECT_IDS["PROJECTILE_NAIL"],
        SUBJECT_IDS["PROJECTILE_GRENADE"],
        SUBJECT_IDS["PROJECTILE_ROCKET"],
        SUBJECT_IDS["LIGHTNING_BEAM"],
    }
)

_WEAPON_BITS = (1, 2, 4, 8, 16, 32, 64)

# Cached default spatial ids — constant across all ticks.
_DEFAULT_SPATIAL_IDS = np.asarray(
    [SPATIAL_SECTOR_IDS[name] for name in SPATIAL_SECTOR_IDS], dtype=np.int32
)


def _normalized_dt_value(tick_hz: int) -> float:
    hz = max(int(tick_hz), 1)
    return float(min((1.0 / float(hz)) / _DT_SCALE, 1.0))


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
            + 1
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
        st = tick.self_token
        origin = st.origin
        vel = st.velocity
        pitch = math.radians(float(st.view_angles[0]) if len(st.view_angles) > 0 else 0.0)
        yaw = math.radians(float(st.view_angles[1]) if len(st.view_angles) > 1 else 0.0)
        wo = int(st.weapons_owned)
        self_scalars = np.array([
            min(float(st.health) / _HEALTH_SCALE, 1.0),
            min(float(st.armor) / _ARMOR_SCALE, 1.0),
            min(float(st.armor_type) / _ARMOR_TYPE_SCALE, 1.0),
            min(float(st.ammo_shells) / _AMMO_CAPS[0], 1.0),
            min(float(st.ammo_nails) / _AMMO_CAPS[1], 1.0),
            min(float(st.ammo_rockets) / _AMMO_CAPS[2], 1.0),
            min(float(st.ammo_cells) / _AMMO_CAPS[3], 1.0),
            float(origin[0]) / _ABS_POS_SCALE if len(origin) > 0 else 0.0,
            float(origin[1]) / _ABS_POS_SCALE if len(origin) > 1 else 0.0,
            float(origin[2]) / _ABS_POS_SCALE if len(origin) > 2 else 0.0,
            float(vel[0]) / _VEL_SCALE if len(vel) > 0 else 0.0,
            float(vel[1]) / _VEL_SCALE if len(vel) > 1 else 0.0,
            float(vel[2]) / _VEL_SCALE if len(vel) > 2 else 0.0,
            math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch),
            1.0 if st.grounded else 0.0,
            min(float(st.waterlevel) / 3.0, 1.0),
            *[1.0 if (wo & bit) else 0.0 for bit in _WEAPON_BITS],
            _normalized_dt_value(getattr(tick, "tick_hz", 20)),
        ], dtype=np.float32)

        # --- objects: bulk fill into pre-allocated arrays ---
        objects = tick.object_tokens
        n_obj = min(len(objects), MAX_OBJECT_TOKENS)

        object_ids = np.zeros((MAX_OBJECT_TOKENS, OBJECT_ID_DIM), dtype=np.int32)
        object_scalars = np.zeros((MAX_OBJECT_TOKENS, OBJECT_SCALAR_DIM), dtype=np.float32)
        object_mask = np.zeros(MAX_OBJECT_TOKENS, dtype=np.bool_)

        if n_obj > 0:
            ids_buf = np.empty((n_obj, OBJECT_ID_DIM), dtype=np.int32)
            sc_buf = np.empty((n_obj, OBJECT_SCALAR_DIM), dtype=np.float32)
            for i in range(n_obj):
                obj = objects[i]
                ids_buf[i, 0] = obj.subject_id
                ids_buf[i, 1] = obj.qualifier_id
                ids_buf[i, 2] = obj.modality_id
                ids_buf[i, 3] = obj.player_id
                sc_buf[i, 0] = obj.rel_x
                sc_buf[i, 1] = obj.rel_y
                sc_buf[i, 2] = obj.rel_z
                sc_buf[i, 3] = obj.recency
                sc_buf[i, 4] = obj.confidence
                sc_buf[i, 5] = obj.magnitude
                sc_buf[i, 6] = obj.state
            sc_buf[:, :3] /= _REL_POS_SCALE
            np.clip(sc_buf[:, 3:], 0.0, 1.0, out=sc_buf[:, 3:])
            object_ids[:n_obj] = ids_buf
            object_scalars[:n_obj] = sc_buf
            object_mask[:n_obj] = True

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
                sp_buf[i, 3] = sp.solid_frac
                sp_buf[i, 4] = sp.water_frac
                sp_buf[i, 5] = sp.slime_frac
                sp_buf[i, 6] = sp.lava_frac
                sp_buf[i, 7] = sp.traversable
                sp_buf[i, 8] = sp.dropoff
                sp_buf[i, 9] = sp.clearance
            np.clip(sp_buf[:, :2], 0.0, _SPATIAL_DIST_SCALE, out=sp_buf[:, :2])
            sp_buf[:, :2] /= _SPATIAL_DIST_SCALE
            np.clip(sp_buf[:, 2:], 0.0, 1.0, out=sp_buf[:, 2:])
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
            "self_weapon_id": np.asarray(int(st.weapon_id), dtype=np.int32),
            "object_ids": object_ids,
            "object_scalars": object_scalars,
            "object_mask": object_mask,
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
