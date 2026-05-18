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
# SUPER_SHOTGUN at slot 42 is in the shotgun family.
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
