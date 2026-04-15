"""Shared combat-metric helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

WEAPON_ID_TO_NAME: Dict[int, str] = {
    1: "axe",
    2: "shotgun",
    3: "super_shotgun",
    4: "nailgun",
    5: "super_nailgun",
    6: "grenade_launcher",
    7: "rocket_launcher",
    8: "thunderbolt",
}

WEAPON_TOTAL_DEBUG_KEYS = {
    "weapon_damage_dealt": "weapon_damage_dealt_total",
    "weapon_hits_landed": "weapon_hits_landed_total",
    "weapon_shots_fired": "weapon_shots_fired_total",
}

WEAPON_STEP_PREFIXES = tuple(WEAPON_TOTAL_DEBUG_KEYS.keys())


def weapon_metric_key(prefix: str, weapon_id: int) -> str:
    name = WEAPON_ID_TO_NAME.get(int(weapon_id), f"w{int(weapon_id)}")
    return f"{prefix}_{name}"


def flatten_weapon_metrics(prefix: str, values_by_weapon: Mapping[int, float]) -> Dict[str, float]:
    return {
        weapon_metric_key(prefix, weapon_id): float(value)
        for weapon_id, value in sorted(values_by_weapon.items())
        if float(value) != 0.0
    }


def weapon_metric_debug_values(raw_values: object) -> Dict[int, float]:
    if not isinstance(raw_values, list):
        return {}
    values: Dict[int, float] = {}
    for weapon_id, raw in enumerate(raw_values):
        if weapon_id <= 0:
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if numeric != 0.0:
            values[weapon_id] = numeric
    return values


def iter_weapon_metric_keys(prefixes: Iterable[str] = WEAPON_STEP_PREFIXES) -> list[str]:
    keys: list[str] = []
    for prefix in prefixes:
        for weapon_id in sorted(WEAPON_ID_TO_NAME):
            keys.append(weapon_metric_key(prefix, weapon_id))
    return keys
