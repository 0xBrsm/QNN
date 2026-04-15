"""Shared semantic token ids for the native token-step protocol.

v9: Entity vocab (subjects + sources unified), action vocab renumbered,
qualifier vocab replaced by source IDs in entity vocab.
"""

from __future__ import annotations

from typing import Dict

# Entity vocab: subjects + sources in one table (42 entries)
ENTITY_IDS: Dict[str, int] = {
    "NONE": 0,
    "PLAYER": 1,
    "WEAPON": 2,
    "AXE": 3,
    "SHOTGUN": 4,
    "NAILGUN": 5,
    "GRENADE_LAUNCHER": 6,
    "ROCKET_LAUNCHER": 7,
    "THUNDERBOLT": 8,
    "AMMO": 9,
    "SHELLS": 10,
    "NAILS": 11,
    "ROCKETS": 12,
    "CELLS": 13,
    "BACKPACK": 14,
    "ARMOR": 15,
    "ARMOR_GREEN": 16,
    "ARMOR_YELLOW": 17,
    "ARMOR_RED": 18,
    "HEALTH": 19,
    "MEGAHEALTH": 20,
    "POWERUP": 21,
    "QUAD": 22,
    "PENT": 23,
    "RING": 24,
    "SUIT": 25,
    "PROJECTILE_NAIL": 26,
    "PROJECTILE_GRENADE": 27,
    "PROJECTILE_ROCKET": 28,
    "LIGHTNING_BEAM": 29,
    "GROUND": 30,
    "WATER": 31,
    "SLIME": 32,
    "LAVA": 33,
    "GIB": 34,
    "BUTTON": 35,
    "PLATFORM": 36,
    "TELEPORTER": 37,
    "DOOR": 38,
    "KEYED": 39,
    "SECRET": 40,
    "TRAIN": 41,
}

ENTITY_VOCAB_SIZE = 42

# Aliases for backward compat
SUBJECT_IDS = ENTITY_IDS

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

MAX_PLAYER_SLOTS = 32

SPATIAL_SECTOR_IDS: Dict[str, int] = {
    "FOV_Center": 0,
    "FOV_Left": 1,
    "FOV_Right": 2,
    "Flank_Left": 3,
    "Flank_Right": 4,
    "Rear_Left": 5,
    "Rear_Right": 6,
    "Ground_State": 7,
    "Ceiling_State": 8,
}

# Token type tags (wire format)
TOKEN_PROJECTILE = 0
TOKEN_ACTOR = 1
TOKEN_ITEM = 2
TOKEN_MOVER = 3

# Per-type scalar dimensions (includes dist after rel[3], path_dist after path[3])
PROJECTILE_SCALAR_DIM = 8
ACTOR_SCALAR_DIM = 19
ITEM_SCALAR_DIM = 15
MOVER_SCALAR_DIM = 14

# Per-type ID counts
PROJECTILE_ID_DIM = 2  # subject_id, modality_id
ACTOR_ID_DIM = 3        # subject_id, modality_id, player_id
ITEM_ID_DIM = 2          # subject_id, modality_id
MOVER_ID_DIM = 2         # subject_id, modality_id

MAX_ENTITY_EVENTS = 4
MAX_TOKEN_OBJECTS = 16

# Reverse lookups
ENTITY_NAMES = {v: k for k, v in ENTITY_IDS.items()}
ACTION_NAMES = {v: k for k, v in ACTION_IDS.items()}
MODALITY_NAMES = {v: k for k, v in MODALITY_IDS.items()}
SPATIAL_SECTOR_NAMES = {v: k for k, v in SPATIAL_SECTOR_IDS.items()}

# Deprecated — kept for import compat, will remove
QUALIFIER_IDS = {"NONE": 0}
QUALIFIER_NAMES = {0: "NONE"}
