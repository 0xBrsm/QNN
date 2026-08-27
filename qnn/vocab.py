"""Shared semantic token ids for the native token-step protocol.

v9: Entity vocab (subjects + sources unified), action vocab renumbered,
qualifier vocab replaced by source IDs in entity vocab.
"""

from __future__ import annotations

from typing import Dict

# Entity vocab: subjects + sources in one table (44 entries).
# Weapons occupy a contiguous block in Quake impulse order
# (axe, shotgun, super_shotgun, nailgun, super_nailgun, grenade,
# rocket, lightning) so the entity_embed rows for "weapon X" and
# "weapon X+1" are adjacent and the model isn't asked to learn that
# SUPER_SHOTGUN at idx 42 is in the shotgun family.
ENTITY_IDS: Dict[str, int] = {
    "NONE": 0,
    "PLAYER": 1,
    "WEAPON": 2,
    "AXE": 3,
    "SHOTGUN": 4,
    "SUPER_SHOTGUN": 5,
    "NAILGUN": 6,
    "SUPER_NAILGUN": 7,
    "GRENADE_LAUNCHER": 8,
    "ROCKET_LAUNCHER": 9,
    "THUNDERBOLT": 10,
    "AMMO": 11,
    "SHELLS": 12,
    "NAILS": 13,
    "ROCKETS": 14,
    "CELLS": 15,
    "BACKPACK": 16,
    "ARMOR": 17,
    "ARMOR_GREEN": 18,
    "ARMOR_YELLOW": 19,
    "ARMOR_RED": 20,
    "HEALTH": 21,
    "MEGAHEALTH": 22,
    "POWERUP": 23,
    "QUAD": 24,
    "PENT": 25,
    "RING": 26,
    "SUIT": 27,
    "PROJECTILE_NAIL": 28,
    "PROJECTILE_GRENADE": 29,
    "PROJECTILE_ROCKET": 30,
    "LIGHTNING_BEAM": 31,
    "GROUND": 32,
    "WATER": 33,
    "SLIME": 34,
    "LAVA": 35,
    "GIB": 36,
    "BUTTON": 37,
    "PLATFORM": 38,
    "TELEPORTER": 39,
    "DOOR": 40,
    "KEYED": 41,
    "SECRET": 42,
    "TRAIN": 43,
}

ENTITY_VOCAB_SIZE = 44


ACTION_IDS: Dict[str, int] = {
    "NONE": 0,
    "FIRE": 1,
    "JUMP": 2,
    "LAND": 3,
    "PICKUP": 4,
    "ENTER": 5,
    "BREATH": 6,
    "EXIT": 7,
    "PAIN": 8,
    "DEATH": 9,
    "CONNECT": 10,
    "DISCONNECT": 11,
    "RESPAWN": 12,
    "ACTIVE": 13,
    "ENDING": 14,
    "BOUNCE": 15,
    "TELEPORT": 16,
    "MOVE": 17,
    "ACTIVATE": 18,
    "REJECT": 19,
}

ACTION_VOCAB_SIZE = 20

MODALITY_IDS: Dict[str, int] = {
    "SIGHT": 0,
    "PROXIMITY": 1,
    "SOUND": 2,
    "MEMORY": 3,
}

MODALITY_VOCAB_SIZE = 4

# A27 combat tokens deliberately expose only these two channels. SOUND and
# MEMORY remain in the engine-wide vocabulary for event/store compatibility,
# but are invalid model inputs for the combat graph.
COMBAT_MODALITY_IDS: Dict[str, int] = {
    "SIGHT": MODALITY_IDS["SIGHT"],
    "PROXIMITY": MODALITY_IDS["PROXIMITY"],
}
COMBAT_MODALITY_VOCAB_SIZE = len(COMBAT_MODALITY_IDS)

MAX_PLAYER_INDICES = 32

# Spatial atlas elevation bands (spatial-tokens-v2 rev 8). Band index =
# wire row = model token row. Names carry the band's center elevation;
# yaw cells inside a band are positional (cell 0 = view forward,
# counter-clockwise), not vocabulary.
SPATIAL_BAND_IDS: Dict[str, int] = {
    "Elev_n75": 0,
    "Elev_n60": 1,
    "Elev_n45": 2,
    "Elev_n30": 3,
    "Elev_n15": 4,
    "Elev_0": 5,
    "Elev_p15": 6,
    "Elev_p30": 7,
    "Elev_p45": 8,
    "Elev_p60": 9,
    "Elev_p75": 10,
}

# A27 combat token type tags (wire format). ITEM/MOVER retain their historical
# constants for offline tooling, but the A27 producer/parser accepts only 0/1.
TOKEN_PROJECTILE = 0
TOKEN_ACTOR = 1
TOKEN_ITEM = 2
TOKEN_MOVER = 3

# INCOMING-projectile gate for reactivity instruments (human PSTH refs, eval
# threat_trace, offline threat scorers — ONE definition everywhere): a newly
# appeared projectile counts as an incoming threat only if the nearest
# projectile is farther than this (game units). Own rockets spawn ~16-60u
# ahead and otherwise contaminate every threat trigger with fire-while-
# strafing correlation (the phantom "+0.032 @ 200 ms human dodge peak",
# retracted 2026-07-11 — the clean human response is a small SUSTAINED lift).
OWN_FIRE_DIST_U = 120.0

# Per-type scalar dimensions (includes dist after rel[3], path_dist after path[3])
PROJECTILE_SCALAR_DIM = 7
ACTOR_SCALAR_DIM = 18
ITEM_SCALAR_DIM = 15
MOVER_SCALAR_DIM = 14

# Entity-stream selector. "combat" is the A27 stream (actor/projectile only,
# no recency); "full" is the a26-line stream (recency on every type, live
# item/mover tokens, 4-way modality vocab). Selected per checkpoint at load
# (see QNNPolicy.load) so a26-line models forward bit-faithfully on this line.
ENTITY_STREAM_COMBAT = "combat"
ENTITY_STREAM_FULL = "full"
ENTITY_STREAMS = (ENTITY_STREAM_COMBAT, ENTITY_STREAM_FULL)

# Full-stream (a26) per-type scalar dims: combat dims + the trailing recency
# scalar on projectile/actor; item/mover carry recency already (their dims
# are shared with the combat constants above, which kept them for offline
# tooling when the A27 stream dropped the token types).
FULL_PROJECTILE_SCALAR_DIM = PROJECTILE_SCALAR_DIM + 1  # 8
FULL_ACTOR_SCALAR_DIM = ACTOR_SCALAR_DIM + 1            # 19

# Per-type ID counts
PROJECTILE_ID_DIM = 2  # subject_id, modality_id
ACTOR_ID_DIM = 3        # subject_id, modality_id, player_id
ITEM_ID_DIM = 2          # subject_id, modality_id
MOVER_ID_DIM = 2         # subject_id, modality_id

MAX_ENTITY_EVENTS = 4
MAX_TOKEN_OBJECTS = 16

# ── self.weapon_id ENTITY_IDS-encoding helper ─────────────────────
#
# obs["self_weapon_id"] is written by the collector using the ENTITY_IDS
# vocab above so the byte can index the shared entity_embed table when
# the model wants to (NONE=0, …, AXE=3, SHOTGUN=4, …, THUNDERBOLT=10).
# This is DIFFERENT from actions["weapon"] which uses the impulse byte
# (0=no weapon, 1=axe, …, 8=LG). Past bugs repeatedly fed self_weapon_id
# directly into an embedding sized WEAPON_HEAD_SIZE+1 (impulse-indexed)
# with a .clamp(0, WEAPON_HEAD_SIZE), silently collapsing RL+LG (and
# wasting indices 1, 2) — the model trained on a 5-class weapon embed
# instead of 8. ALWAYS use this helper before indexing any
# impulse-keyed table (e.g. weapon_embed_self of size 9, the BC weapon
# head's 8-class output). For entity_embed (size 44) use
# self_weapon_id directly — that's what the encoding is for.
def self_weapon_id_to_impulse(weapon_id):
    """Translate ENTITY_IDS-encoded obs.self_weapon_id → impulse byte (0..8).

    Accepts a Python int, np.ndarray, or torch.Tensor; returns the same
    type. Pure arithmetic: ``max(0, weapon_id - 2)``. The 0/1/2 region
    (NONE/PLAYER/WEAPON in ENTITY_IDS) maps to impulse=0 (no weapon).
    """
    # Lazy imports so qnn.vocab stays a leaf module.
    try:
        import torch  # type: ignore
        if isinstance(weapon_id, torch.Tensor):
            return (weapon_id - 2).clamp_min(0)
    except ImportError:
        pass
    try:
        import numpy as _np  # type: ignore
        if isinstance(weapon_id, _np.ndarray):
            return _np.maximum(weapon_id.astype(_np.int64) - 2, 0)
    except ImportError:
        pass
    return max(0, int(weapon_id) - 2)


# Reverse lookups
ENTITY_NAMES = {v: k for k, v in ENTITY_IDS.items()}
ACTION_NAMES = {v: k for k, v in ACTION_IDS.items()}
MODALITY_NAMES = {v: k for k, v in MODALITY_IDS.items()}
SPATIAL_BAND_NAMES = {v: k for k, v in SPATIAL_BAND_IDS.items()}
