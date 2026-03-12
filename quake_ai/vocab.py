"""Shared semantic token ids for the native token-step protocol."""

from __future__ import annotations

from typing import Dict

SUBJECT_IDS: Dict[str, int] = {
    "NONE": 0,
    "PLAYER": 1,
    "BACKPACK": 2,
    "AXE": 3,
    "SHOTGUN": 4,
    "NAILGUN": 5,
    "GRENADE_LAUNCHER": 6,
    "ROCKET_LAUNCHER": 7,
    "THUNDERBOLT": 8,
    "SHELLS": 9,
    "NAILS": 10,
    "ROCKETS": 11,
    "CELLS": 12,
    "HEALTH": 13,
    "MEGAHEALTH": 14,
    "ARMOR_GREEN": 15,
    "ARMOR_YELLOW": 16,
    "ARMOR_RED": 17,
    "QUAD": 18,
    "PENT": 19,
    "RING": 20,
    "SUIT": 21,
    "PROJECTILE_NAIL": 22,
    "PROJECTILE_GRENADE": 23,
    "PROJECTILE_ROCKET": 24,
    "LIGHTNING_BEAM": 25,
    "TELEPORTER": 26,
    "DOOR": 27,
    "PLATFORM": 28,
    "TRAIN": 29,
    "BUTTON": 30,
}

ACTION_IDS: Dict[str, int] = {
    "NONE": 0,
    "FIRE": 1,
    "IMPACT": 2,
    "BOUNCE": 3,
    "PICKUP": 4,
    "RESPAWN": 5,
    "PAIN": 6,
    "DEATH": 7,
    "WARNING": 8,
    "ACTIVE": 9,
    "JUMP": 10,
    "LAND": 11,
    "ENTER": 12,
    "EXIT": 13,
    "TELEPORT": 14,
    "MOVE": 15,
    "ACTIVATE": 16,
    "REJECT": 17,
    "BREATH": 18,
}

QUALIFIER_IDS: Dict[str, int] = {
    "NONE": 0,
    "DROWN": 1,
    "WATER": 2,
    "LAVA": 3,
    "SLIME": 4,
    "FLESH": 5,
    "WORLD": 6,
    "KEYED": 7,
    "SECRET": 8,
    "INVISIBLE": 9,
}

MODALITY_IDS: Dict[str, int] = {
    "NONE": 0,
    "VISUAL": 1,
    "AUDITORY": 2,
    "SPATIAL": 3,
    "MENTAL": 4,
}

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

SUBJECT_NAMES = {value: key for key, value in SUBJECT_IDS.items()}
ACTION_NAMES = {value: key for key, value in ACTION_IDS.items()}
QUALIFIER_NAMES = {value: key for key, value in QUALIFIER_IDS.items()}
MODALITY_NAMES = {value: key for key, value in MODALITY_IDS.items()}
SPATIAL_SECTOR_NAMES = {value: key for key, value in SPATIAL_SECTOR_IDS.items()}
