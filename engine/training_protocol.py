"""Binary parser for training_extras_v1 sidecar frames."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable

TRAINING_BINARY_MAGIC = b"QTRN"
TRAINING_BINARY_VERSION = 1

TRAINING_FLAG_RESET = 0x0001
TRAINING_FLAG_DONE = 0x0002

_HEADER_FORMAT = "<4sHHIIhhHHHHhhHHHH17f"
TRAINING_BINARY_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

_DAMAGE_FORMAT = "<hhHH8f"
_DAMAGE_SIZE = struct.calcsize(_DAMAGE_FORMAT)

_ITEM_FORMAT = "<hhHHHH4f"
_ITEM_SIZE = struct.calcsize(_ITEM_FORMAT)

_DEATH_FORMAT = "<hhHH"
_DEATH_SIZE = struct.calcsize(_DEATH_FORMAT)

_SPAWN_FORMAT = "<hH3f"
_SPAWN_SIZE = struct.calcsize(_SPAWN_FORMAT)


@dataclass(frozen=True, slots=True)
class TrainingDamageRecordV1:
    attacker_entity_num: int
    target_entity_num: int
    weapon_id: int
    flags: int
    health_before: float
    health_after: float
    armor_before: float
    armor_after: float
    armor_type_before: float
    armor_type_after: float
    damage_health: float
    damage_armor: float


@dataclass(frozen=True, slots=True)
class TrainingItemRecordV1:
    actor_entity_num: int
    item_entity_num: int
    event_kind: int
    category: int
    weapon_id: int
    flags: int
    amount: float
    origin: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TrainingDeathRecordV1:
    victim_entity_num: int
    attacker_entity_num: int
    weapon_id: int
    flags: int


@dataclass(frozen=True, slots=True)
class TrainingSpawnRecordV1:
    player_entity_num: int
    flags: int
    origin: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TrustedTrainingExtrasV1:
    episode_id: str
    tick: int
    steps: int
    reset: bool
    done: bool
    self_entity_num: int
    weapon_id: int
    frag_gain: int
    frag_loss: int
    player_died: bool
    hit_count: int
    shots_fired: int
    damage_dealt: float
    damage_taken: float
    health_before: float
    health_after: float
    armor_before: float
    armor_after: float
    armor_type_before: float
    armor_type_after: float
    edp_raw: float
    pickup_health: float
    pickup_armor: float
    pickup_ammo: float
    weapon_pickups: float
    item_pickups: float
    episode_damage_dealt: float
    episode_hit_count: float
    episode_shots_fired: float
    damage_records: tuple[TrainingDamageRecordV1, ...]
    item_records: tuple[TrainingItemRecordV1, ...]
    death_records: tuple[TrainingDeathRecordV1, ...]
    spawn_records: tuple[TrainingSpawnRecordV1, ...]


def decode_binary_training_extras(
    header: bytes,
    read_exact: Callable[[int], bytes],
    *,
    episode_id: str,
) -> TrustedTrainingExtrasV1:
    (
        magic,
        version,
        flags,
        tick,
        steps,
        self_entity_num,
        weapon_id,
        damage_count,
        item_count,
        death_count,
        spawn_count,
        frag_gain,
        frag_loss,
        player_died,
        hit_count,
        shots_fired,
        _pad,
        damage_dealt,
        damage_taken,
        health_before,
        health_after,
        armor_before,
        armor_after,
        armor_type_before,
        armor_type_after,
        edp_raw,
        pickup_health,
        pickup_armor,
        pickup_ammo,
        weapon_pickups,
        item_pickups,
        episode_damage_dealt,
        episode_hit_count,
        episode_shots_fired,
    ) = struct.unpack(_HEADER_FORMAT, header)
    if magic != TRAINING_BINARY_MAGIC:
        raise ValueError(f"Unexpected training magic {magic!r}")
    if int(version) != TRAINING_BINARY_VERSION:
        raise ValueError(f"Unexpected training version {version}")

    damage_records = tuple(
        TrainingDamageRecordV1(*struct.unpack(_DAMAGE_FORMAT, read_exact(_DAMAGE_SIZE)))
        for _ in range(int(damage_count))
    )
    item_records = tuple(
        TrainingItemRecordV1(
            actor_entity_num=record[0],
            item_entity_num=record[1],
            event_kind=record[2],
            category=record[3],
            weapon_id=record[4],
            flags=record[5],
            amount=record[6],
            origin=(record[7], record[8], record[9]),
        )
        for record in (struct.unpack(_ITEM_FORMAT, read_exact(_ITEM_SIZE)) for _ in range(int(item_count)))
    )
    death_records = tuple(
        TrainingDeathRecordV1(*struct.unpack(_DEATH_FORMAT, read_exact(_DEATH_SIZE)))
        for _ in range(int(death_count))
    )
    spawn_records = tuple(
        TrainingSpawnRecordV1(
            player_entity_num=record[0],
            flags=record[1],
            origin=(record[2], record[3], record[4]),
        )
        for record in (struct.unpack(_SPAWN_FORMAT, read_exact(_SPAWN_SIZE)) for _ in range(int(spawn_count)))
    )
    return TrustedTrainingExtrasV1(
        episode_id=str(episode_id),
        tick=int(tick),
        steps=int(steps),
        reset=bool(int(flags) & TRAINING_FLAG_RESET),
        done=bool(int(flags) & TRAINING_FLAG_DONE),
        self_entity_num=int(self_entity_num),
        weapon_id=int(weapon_id),
        frag_gain=int(frag_gain),
        frag_loss=int(frag_loss),
        player_died=bool(player_died),
        hit_count=int(hit_count),
        shots_fired=int(shots_fired),
        damage_dealt=float(damage_dealt),
        damage_taken=float(damage_taken),
        health_before=float(health_before),
        health_after=float(health_after),
        armor_before=float(armor_before),
        armor_after=float(armor_after),
        armor_type_before=float(armor_type_before),
        armor_type_after=float(armor_type_after),
        edp_raw=float(edp_raw),
        pickup_health=float(pickup_health),
        pickup_armor=float(pickup_armor),
        pickup_ammo=float(pickup_ammo),
        weapon_pickups=float(weapon_pickups),
        item_pickups=float(item_pickups),
        episode_damage_dealt=float(episode_damage_dealt),
        episode_hit_count=float(episode_hit_count),
        episode_shots_fired=float(episode_shots_fired),
        damage_records=damage_records,
        item_records=item_records,
        death_records=death_records,
        spawn_records=spawn_records,
    )
