"""Observation shape constants for the model tensor contract.

v9: Per-type entity tokens with variable-length wire format.
The C worker packs observations with type-tagged entity tokens.
This module provides dimension constants consumed by the model.
"""

from __future__ import annotations

from qnn.vocab import (
    MAX_TOKEN_OBJECTS,
    MAX_ENTITY_EVENTS,
    ACTOR_SCALAR_DIM,
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
# engaged). 3 = per-tick LOOK COMMITMENT state (lc_cls / lc_rem / lc_elapsed /
# lc_dur / lc_dir — the LOOK_COMMIT_STATE_DIM lanes of
# qnn.model.look_seg_decode, tick-aligned with every other flat column;
# onset = an lc_elapsed RESTART, i.e. the counter dropping vs the previous
# tick — look_commit_step stores the post-increment counter, so the onset-tick
# floor is 0 or 1 depending on where the state is snapshotted; do not test
# `== 0`). Every lc_* column is -1 on models
# with no look_seg head (the look-commitment decode never ran), so the keys
# and their lengths exist unconditionally and -1 is unambiguous: each real
# lane is a non-negative class/counter.
# 4 = per-tick LOOK TANGENT `look_tan`, (n, 2) float16 — the emitted look
# vector's tangent ``z = θ · ŷz`` with ``θ = atan2(|yz|, x)``, i.e. EXACTLY the
# corpus ``act_look_tan`` quantity (qnn.bc.cache_look_tan.look_to_tangent /
# qnn.model.look_bins.tangent_logmap), in RADIANS. This is what makes model
# streams segmentable by the same rule as human streams
# (qnn.model.look_seg_segment.segment_engaged reads θ=|z| for hold-vs-stroke and
# Δφ between consecutive z for reversals): `turn_deg` carries the magnitude only,
# and stroke DIRECTION is otherwise unrecoverable for a polar-driven model (its
# lc_* lanes are all -1). float16 matches the corpus sidecar's own storage
# (~0.05% relative on |z|, ~3e-4 deg at fovea scale — an order below the demo
# angle16 floor). A tick with no look action stores (0, 0), the same hold that
# `turn_deg == 0` records. Bump on ANY field change: eval caches (e.g.
# decode-fit freeplay waves) key on it so stale-schema npz are re-produced,
# not silently reused.
# 5 = per-tick HELD weapon impulse `weapon_held` (int8, 0 = none/unknown, from
# obs self_weapon_id). The attack-with era made `weapon`/`weapon_imp`
# events-only (attack CLASS, 0 off fire ticks), which silently starved every
# held-weapon denominator: gates.py's op-shot ruler read fires/held ≡ tick_hz
# for every weapon and rc.py's weapon dwell/switch measured the fire toggle —
# the a28rc1 trim stall.
# 6 = the "held weapon" CONCEPT is retired (weapon scripting churns equip
# state without expressing choice — Brian 2026-08-09). `weapon_held` is
# renamed `self_weapon_id` (raw engine equip, impulse-coded: the op-shot
# denominator and equip-churn diagnostic, never a behavioral signal) and
# three additions serve the weapon-PREFERENCE ruler (core.preference_pairs):
# `weapon_pref` (int8, equipped impulse at DISCHARGE ticks, 0 elsewhere —
# the bot-side analog of the corpus attack class), `weapon_feas` (uint8
# bitmask, bit imp−1 = weapon in the DECISION SPACE per
# engine_norm.weapon_feasibility_bits — the ownership/ammo half of the
# model's own attack-with choice-set predicate), and `health` (int16 raw).
# The ruler invalidates preference carry on stationary-menu violations
# (feas change / death in the pair window) by the SAME rule on both sides.
# (Amended same-day from a raw ammo-column draft before any artifact was
# produced at 6 — feasibility is read as the model's own quantity instead
# of reconstructed from raw pools.)
GATE_STREAM_SCHEMA_VERSION = 6

# ONNX graph-input name → native obs-dict key (the export naming convention;
# shared by tools/export_onnx.py and the ORT h2h side driver — one table).
ONNX_OBS_KEY_REMAP = {
    "self_health":          "health",
    "self_effective_armor": "effective_armor",
    "self_ammo_shells":     "ammo_shells",
    "self_ammo_nails":      "ammo_nails",
    "self_ammo_rockets":    "ammo_rockets",
    "self_ammo_cells":      "ammo_cells",
    "self_vel":             "vel",
    "self_attack_finished": "attack_finished",
    "view_pitch":           "view_pitch",
    "self_movement_id":     "self_movement_id",
    "self_items":           "self_items",
    "look_delta":           "look_delta",
}

# Self-subtoken widths consumed by the three ObsEmbedding projections
# (self_proj_state / self_proj_arsenal / self_proj_motion).
SELF_STATE_SCALAR_DIM   = 2  # health, effective_armor
SELF_ARSENAL_SCALAR_DIM = 1  # attack_finished
SELF_MOTION_SCALAR_DIM  = 4  # vel_xyz, view_pitch

# Quake weapon classes: axe, shotgun, super_shotgun, nailgun,
# super_nailgun, grenade_launcher, rocket_launcher, thunderbolt.
# Indexed 1..8 in the game's impulse byte. A27's attack-with head adds its own
# class 0 for "no attack"; there is no equipped-weapon observation class.
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
