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
    MAX_PLAYER_INDICES,
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

SPATIAL_TOKEN_COUNT = 11  # depth-atlas elevation bands (rev 8)
SPATIAL_SCALAR_DIM = 48  # per band: [depth_norm x 24 yaw cells, hit x 24]
# Probe-grid relative pose, per probe: offset xyz in the agent's view
# frame (/DIST_SCALE), yaw-cell residual (deg/5), |offset|/DIST_SCALE.
PROBE_OFFSET_DIM = 5

# Spatial token payload source (rev-10 probe-grid direction). Lives here
# rather than in qnn.model.graph.spec because both the spec (which
# validates the name) and qnn.model.transformer (which dispatches on it)
# need it, and transformer cannot import the graph package without a cycle.
SPATIAL_SOURCE_EGO = "ego"
SPATIAL_SOURCE_PROBE_GRID = "probe_grid"
SPATIAL_SOURCE_POOLED9 = "pooled9"
# probe_grid + an egocentric near-field floor ring (rev-11): the k probes
# carry the mid/far field, and 3 extra tokens carry the agent's own steep
# downward bands (the drop signal the probes structurally can't supply from
# a distance — see agents/plans/spatial-tokens-v2.md). Same k knob as
# probe_grid; the ring is sourced in-graph from the agent's spatial_atlas.
SPATIAL_SOURCE_PROBE_GRID_NF = "probe_grid_nf"
SPATIAL_SOURCES = (
    SPATIAL_SOURCE_EGO, SPATIAL_SOURCE_PROBE_GRID, SPATIAL_SOURCE_POOLED9,
    SPATIAL_SOURCE_PROBE_GRID_NF,
)
# Sources that fuse k precomputed probes (carry the probe-count knob).
PROBE_SPATIAL_SOURCES = (SPATIAL_SOURCE_PROBE_GRID, SPATIAL_SOURCE_PROBE_GRID_NF)

SELF_SCALAR_DIM = 18  # flat layout: legacy 17 + view_pitch at idx 17

# Version of the flat gate-stream schema in move_streams_*.npz (the band /
# rc_humanlikeness subject format written by qnn.eval.run and read by
# qnn.eval.humanlikeness). 2 = band-v5 fields (discharge / weapon_imp /
# engaged). Bump on ANY field change: eval caches (e.g. decode-fit freeplay
# waves) key on it so stale-schema npz are re-produced, not silently reused.
GATE_STREAM_SCHEMA_VERSION = 2

# Self-subtoken widths consumed by the three ObsEmbedding projections
# (self_proj_state / self_proj_arsenal / self_proj_motion).
SELF_STATE_SCALAR_DIM   = 2  # health, effective_armor
SELF_ARSENAL_SCALAR_DIM = 1  # attack_finished
SELF_MOTION_SCALAR_DIM  = 4  # vel_xyz, view_pitch

# Quake weapon classes: axe, shotgun, super_shotgun, nailgun,
# super_nailgun, grenade_launcher, rocket_launcher, thunderbolt.
# Indexed 1..8 in the game's impulse byte; class 0 is reserved for
# "no weapon held" in the self-token embedding (pre-spawn / dead /
# transitional). qnn.model.policy.WEAPON_HEAD_SIZE imports this.
WEAPON_HEAD_SIZE = 8

from qnn.wire import MAX_ENTITY_SCALAR_DIM, MAX_ENTITY_ID_DIM

# v9: obs_dim is unused by the transformer encoder (it uses token dicts).
# Kept as 0 for callers that still reference it (e.g. bc_train model init).
OBS_DIM = 0

# Canonical obs shape schema — single source of truth for env, checkpoint, and adapter.
OBS_SCHEMA: dict[str, tuple[int, ...]] = {
    "self_scalars": (SELF_SCALAR_DIM,),
    "self_state_scalars":   (SELF_STATE_SCALAR_DIM,),
    "self_arsenal_scalars": (SELF_ARSENAL_SCALAR_DIM,),
    "self_motion_scalars":  (SELF_MOTION_SCALAR_DIM,),
    # Frame-to-frame change in the anchor-relative look vector
    # (look[t-1] - look[t-2]); ~0 under steady rotation, ≈ angular
    # acceleration. A first-class self-motion field sourced at the obs
    # boundary (look-label Δ at preload; engine-computed at inference) —
    # never reconstructed in-model. NOT angular velocity (a single look
    # vector is that); see qnn.model.tokens.obs_fields.
    "look_delta": (3,),
    "self_weapon_id": (1,),
    "self_weapon_readiness": (WEAPON_HEAD_SIZE,),
    "self_armor_type_id": (1,),
    "self_state_powerup_ids":   (3,),
    "self_arsenal_powerup_ids": (1,),
    "self_motion_powerup_ids":  (1,),
    "self_movement_id": (1,),
    "entity_types": (MAX_TOKEN_OBJECTS,),
    "entity_scalars_raw": (MAX_TOKEN_OBJECTS, MAX_ENTITY_SCALAR_DIM),
    "entity_ids": (MAX_TOKEN_OBJECTS, MAX_ENTITY_ID_DIM),
    "entity_event_actions": (MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS),
    "entity_event_sources": (MAX_TOKEN_OBJECTS, MAX_ENTITY_EVENTS),
    "entity_event_counts": (MAX_TOKEN_OBJECTS,),
    "spatial_scalars": (SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM),
}
