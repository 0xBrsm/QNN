"""Action space helpers for discrete multi-head Quake control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

LOOK_MOUSE_BINS: tuple[int, ...] = (
    -128,
    -96,
    -72,
    -56,
    -40,
    -28,
    -20,
    -14,
    -10,
    -6,
    -3,
    -1,
    0,
    1,
    3,
    6,
    10,
    14,
    20,
    28,
    40,
    56,
    72,
    96,
    128,
)
LOOK_NEUTRAL_LABEL = LOOK_MOUSE_BINS.index(0)
LOOK_MAX_ABS_MOUSE = max(abs(value) for value in LOOK_MOUSE_BINS)
WEAPON_SWITCH_SLOTS = 8

ACTION_HEADS = {
    "move": 3,        # 0 none, 1 forward, 2 back
    "strafe": 3,      # 0 none, 1 left, 2 right
    "look_yaw": len(LOOK_MOUSE_BINS),
    "look_pitch": len(LOOK_MOUSE_BINS),
    "fire": 2,        # 0 off, 1 on
    "jump": 2,        # 0 off, 1 on
    "weapon": WEAPON_SWITCH_SLOTS + 1,  # 0 none, 1..8 direct switch
}


def clamp_weapon_switch(slot: int) -> int:
    return max(0, min(int(slot), WEAPON_SWITCH_SLOTS))


def look_label_from_mouse_count(mouse_count: int) -> int:
    target = int(mouse_count)
    best_index = LOOK_NEUTRAL_LABEL
    best_distance = float("inf")
    for index, candidate in enumerate(LOOK_MOUSE_BINS):
        distance = abs(candidate - target)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def mouse_count_from_look_label(label: int) -> int:
    idx = int(label)
    if idx < 0 or idx >= len(LOOK_MOUSE_BINS):
        raise ValueError(f"look label out of range [0, {len(LOOK_MOUSE_BINS)})")
    return int(LOOK_MOUSE_BINS[idx])


def _legacy_turn_to_look_yaw(turn: int) -> int:
    legacy_turn = int(turn)
    if legacy_turn == 1:
        return look_label_from_mouse_count(-10)
    if legacy_turn == 2:
        return look_label_from_mouse_count(10)
    return LOOK_NEUTRAL_LABEL


def normalized_action_features(action: Mapping[str, int]) -> List[float]:
    labels = ActionLabels.from_dict(action)
    return [
        float(labels.move) / 2.0,
        float(labels.strafe) / 2.0,
        float(mouse_count_from_look_label(labels.look_yaw)) / float(LOOK_MAX_ABS_MOUSE),
        float(mouse_count_from_look_label(labels.look_pitch)) / float(LOOK_MAX_ABS_MOUSE),
        float(labels.fire),
        float(labels.jump),
        float(labels.weapon) / float(WEAPON_SWITCH_SLOTS),
    ]


@dataclass(slots=True)
class ActionLabels:
    move: int = 0
    strafe: int = 0
    look_yaw: int = LOOK_NEUTRAL_LABEL
    look_pitch: int = LOOK_NEUTRAL_LABEL
    fire: int = 0
    jump: int = 0
    weapon: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, int]) -> "ActionLabels":
        look_yaw = payload.get("look_yaw")
        if look_yaw is None:
            look_yaw = _legacy_turn_to_look_yaw(int(payload.get("turn", 0)))
        look_pitch = payload.get("look_pitch", LOOK_NEUTRAL_LABEL)
        action = cls(
            move=int(payload.get("move", 0)),
            strafe=int(payload.get("strafe", 0)),
            look_yaw=int(look_yaw),
            look_pitch=int(look_pitch),
            fire=int(payload.get("fire", payload.get("attack", 0))),
            jump=int(payload.get("jump", 0)),
            weapon=clamp_weapon_switch(int(payload.get("weapon", payload.get("weapon_switch", 0)))),
        )
        action.validate()
        return action

    def validate(self) -> None:
        for key, size in ACTION_HEADS.items():
            value = int(getattr(self, key))
            if value < 0 or value >= size:
                raise ValueError(f"{key} out of range [0, {size})")

    def to_dict(self) -> Dict[str, int]:
        return {
            "move": self.move,
            "strafe": self.strafe,
            "look_yaw": self.look_yaw,
            "look_pitch": self.look_pitch,
            "fire": self.fire,
            "jump": self.jump,
            "weapon": self.weapon,
        }


def flatten_action(action: Mapping[str, int]) -> List[int]:
    labels = ActionLabels.from_dict(action)
    return [
        labels.move,
        labels.strafe,
        labels.look_yaw,
        labels.look_pitch,
        labels.fire,
        labels.jump,
        labels.weapon,
    ]


def action_from_list(values: List[int]) -> Dict[str, int]:
    if len(values) != len(ACTION_HEADS):
        raise ValueError(f"Expected {len(ACTION_HEADS)} values, got {len(values)}")
    labels = ActionLabels(
        move=int(values[0]),
        strafe=int(values[1]),
        look_yaw=int(values[2]),
        look_pitch=int(values[3]),
        fire=int(values[4]),
        jump=int(values[5]),
        weapon=clamp_weapon_switch(int(values[6])),
    )
    labels.validate()
    return labels.to_dict()


def idle_action() -> Dict[str, int]:
    return ActionLabels().to_dict()
