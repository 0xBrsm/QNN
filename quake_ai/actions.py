"""Canonical action helpers for mixed continuous/discrete Quake control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

LOOK_BASE_DEGREES_PER_COUNT = 0.066
LOOK_NONLINEAR_BASE_COUNT = 256.0
LOOK_HIGH_GAIN_MULTIPLIER = 2.0  # sensitivity 3 near center, 6 at the edge
LOOK_DEADZONE = 0.03
MOVE_DEADZONE = 0.25
WEAPON_SWITCH_SLOTS = 5
RECALL_REGISTER_SIZE = 64
ACTION_SCHEMA_VERSION = 1

ACTION_HEADS = {
    "move": 2,
    "look": 3,
    "jump": 2,
    "fire": 2,
    "switch": WEAPON_SWITCH_SLOTS + 1,
    "recall_0": RECALL_REGISTER_SIZE + 1,
    "recall_1": RECALL_REGISTER_SIZE + 1,
    "recall_2": RECALL_REGISTER_SIZE + 1,
    "recall_3": RECALL_REGISTER_SIZE + 1,
}
CONTINUOUS_ACTION_HEADS = frozenset({"move", "look"})

# Deterministic head ordering (Python 3.7+ dict preserves insertion order).
# Shared by checkpoint_converter and other modules that need a canonical order.
HEAD_ORDER: list[str] = list(ACTION_HEADS.keys())
DISCRETE_ACTION_HEADS = frozenset({"jump", "fire", "switch", "recall_0", "recall_1", "recall_2", "recall_3"})

_SWITCH_SLOT_TO_IMPULSES = {
    1: (3, 2),  # prefer SSG when available
    2: (5, 4),  # prefer SNG when available
    3: (6,),
    4: (7,),
    5: (8,),
}
_IMPULSE_TO_REQUIRED_BIT = {
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 16,
    7: 32,
    8: 64,
}


def clamp_switch(slot: int) -> int:
    return max(0, min(int(slot), WEAPON_SWITCH_SLOTS))


def clamp_recall_target(target: int) -> int:
    return max(0, min(int(target), RECALL_REGISTER_SIZE))


def clamp_unit(value: float) -> float:
    return max(-1.0, min(float(value), 1.0))


def _vector2(value: object, *, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    try:
        first = float(value[0])  # type: ignore[index]
        second = float(value[1])  # type: ignore[index]
    except Exception as exc:  # pragma: no cover - defensive input validation
        raise ValueError(f"Expected a 2-vector, got {value!r}") from exc
    return (clamp_unit(first), clamp_unit(second))


def _vector3(value: object, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        x = float(value[0])  # type: ignore[index]
        y = float(value[1])  # type: ignore[index]
        z = float(value[2])  # type: ignore[index]
    except Exception as exc:  # pragma: no cover - defensive input validation
        raise ValueError(f"Expected a 3-vector, got {value!r}") from exc
    return (clamp_unit(x), clamp_unit(y), clamp_unit(z))


def move_vector_from_buttons(move_button: int, strafe_button: int) -> tuple[float, float]:
    forward = 0.0
    right = 0.0
    if int(move_button) == 1:
        forward = 1.0
    elif int(move_button) == 2:
        forward = -1.0
    if int(strafe_button) == 1:
        right = -1.0
    elif int(strafe_button) == 2:
        right = 1.0
    return (forward, right)


def switch_impulse_from_slot(switch_slot: int, *, weapons_owned: int | None = None) -> int:
    slot = clamp_switch(switch_slot)
    if slot <= 0:
        return 0

    candidates = _SWITCH_SLOT_TO_IMPULSES.get(slot, ())
    if weapons_owned is None:
        return int(candidates[0]) if candidates else 0

    owned_mask = int(weapons_owned)
    for candidate in candidates:
        required_bit = _IMPULSE_TO_REQUIRED_BIT.get(candidate, 0)
        if required_bit and (owned_mask & required_bit):
            return int(candidate)
    return 0


def switch_slot_from_weapon_id(weapon_id: int) -> int:
    weapon = int(weapon_id)
    if weapon <= 0:
        return 0
    if weapon in (2, 3):
        return 1
    if weapon in (4, 5):
        return 2
    if weapon == 6:
        return 3
    if weapon == 7:
        return 4
    if weapon == 8:
        return 5
    return 0


def _look_count_curve(magnitude: float) -> float:
    magnitude = max(0.0, min(float(magnitude), 1.0))
    return LOOK_NONLINEAR_BASE_COUNT * magnitude * (
        1.0 + ((LOOK_HIGH_GAIN_MULTIPLIER - 1.0) * magnitude * magnitude)
    )


def _look_magnitude_from_count(count_magnitude: float) -> float:
    target = max(0.0, min(float(count_magnitude) / LOOK_NONLINEAR_BASE_COUNT, LOOK_HIGH_GAIN_MULTIPLIER))
    lo = 0.0
    hi = 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        value = mid * (1.0 + ((LOOK_HIGH_GAIN_MULTIPLIER - 1.0) * mid * mid))
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mouse_count_from_look_axis(axis: float) -> int:
    clamped = clamp_unit(axis)
    sign = -1.0 if clamped < 0.0 else 1.0
    magnitude = abs(clamped)
    if magnitude <= LOOK_DEADZONE:
        return 0
    normalized = (magnitude - LOOK_DEADZONE) / max(1.0 - LOOK_DEADZONE, 1e-6)
    return int(round(sign * _look_count_curve(normalized)))


def look_axis_from_mouse_count(mouse_count: int) -> float:
    count = int(mouse_count)
    if count == 0:
        return 0.0
    sign = -1.0 if count < 0 else 1.0
    normalized = _look_magnitude_from_count(abs(count))
    axis = LOOK_DEADZONE + ((1.0 - LOOK_DEADZONE) * normalized)
    return sign * clamp_unit(axis)


def normalized_action_features(action: Mapping[str, object]) -> List[float]:
    labels = ActionLabels.from_dict(action)
    return [
        float(labels.move[0]),
        float(labels.move[1]),
        float(labels.look[0]),
        float(labels.look[1]),
        float(labels.look[2]),
        float(labels.fire),
        float(labels.jump),
        float(labels.switch) / float(WEAPON_SWITCH_SLOTS),
    ]


@dataclass(slots=True)
class ActionLabels:
    move: tuple[float, float] = (0.0, 0.0)
    look: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jump: int = 0
    fire: int = 0
    switch: int = 0
    recall_0: int = 0
    recall_1: int = 0
    recall_2: int = 0
    recall_3: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionLabels":
        move = _vector2(payload.get("move"), default=(0.0, 0.0))
        look = _vector3(payload.get("look"), default=(0.0, 0.0, 0.0))

        action = cls(
            move=move,
            look=look,
            fire=int(payload.get("fire", 0)),
            jump=int(payload.get("jump", 0)),
            switch=clamp_switch(int(payload.get("switch", 0))),
            recall_0=clamp_recall_target(int(payload.get("recall_0", 0))),
            recall_1=clamp_recall_target(int(payload.get("recall_1", 0))),
            recall_2=clamp_recall_target(int(payload.get("recall_2", 0))),
            recall_3=clamp_recall_target(int(payload.get("recall_3", 0))),
        )
        action.validate()
        return action

    def validate(self) -> None:
        for value in (*self.move, *self.look):
            if float(value) < -1.0 or float(value) > 1.0:
                raise ValueError("continuous action values must be in [-1, 1]")
        if self.fire < 0 or self.fire >= ACTION_HEADS["fire"]:
            raise ValueError(f"fire out of range [0, {ACTION_HEADS['fire']})")
        if self.jump < 0 or self.jump >= ACTION_HEADS["jump"]:
            raise ValueError(f"jump out of range [0, {ACTION_HEADS['jump']})")
        if self.switch < 0 or self.switch >= ACTION_HEADS["switch"]:
            raise ValueError(f"switch out of range [0, {ACTION_HEADS['switch']})")
        for head in ("recall_0", "recall_1", "recall_2", "recall_3"):
            value = int(getattr(self, head))
            if value < 0 or value >= ACTION_HEADS[head]:
                raise ValueError(f"{head} out of range [0, {ACTION_HEADS[head]})")

    def to_dict(self) -> Dict[str, object]:
        return {
            "move": [float(self.move[0]), float(self.move[1])],
            "look": [float(self.look[0]), float(self.look[1]), float(self.look[2])],
            "jump": int(self.jump),
            "fire": int(self.fire),
            "switch": int(self.switch),
            "recall_0": int(self.recall_0),
            "recall_1": int(self.recall_1),
            "recall_2": int(self.recall_2),
            "recall_3": int(self.recall_3),
        }


def flatten_action(action: Mapping[str, object]) -> List[float]:
    labels = ActionLabels.from_dict(action)
    return [
        float(labels.move[0]),
        float(labels.move[1]),
        float(labels.look[0]),
        float(labels.look[1]),
        float(labels.look[2]),
        float(labels.jump),
        float(labels.fire),
        float(labels.switch),
        float(labels.recall_0),
        float(labels.recall_1),
        float(labels.recall_2),
        float(labels.recall_3),
    ]


def action_from_list(values: Sequence[float]) -> Dict[str, object]:
    if len(values) != 12:
        raise ValueError(f"Expected 12 values, got {len(values)}")
    payload: Dict[str, object] = {
        "move": [float(values[0]), float(values[1])],
        "look": [float(values[2]), float(values[3]), float(values[4])],
        "jump": int(round(float(values[5]))),
        "fire": int(round(float(values[6]))),
        "switch": int(round(float(values[7]))),
        "recall_0": int(round(float(values[8]))),
        "recall_1": int(round(float(values[9]))),
        "recall_2": int(round(float(values[10]))),
        "recall_3": int(round(float(values[11]))),
    }
    return ActionLabels.from_dict(payload).to_dict()


def idle_action() -> Dict[str, object]:
    return ActionLabels().to_dict()
