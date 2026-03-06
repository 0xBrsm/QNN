"""Action space helpers for discrete multi-head Quake control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

ACTION_HEADS = {
    "move": 3,   # 0 none, 1 forward, 2 back
    "strafe": 3, # 0 none, 1 left, 2 right
    "turn": 3,   # 0 none, 1 left, 2 right
    "use": 2,    # 0 off, 1 on
}


@dataclass(slots=True)
class ActionLabels:
    move: int = 0
    strafe: int = 0
    turn: int = 0
    use: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, int]) -> "ActionLabels":
        action = cls(
            move=int(payload.get("move", 0)),
            strafe=int(payload.get("strafe", 0)),
            turn=int(payload.get("turn", 0)),
            use=int(payload.get("use", 0)),
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
            "turn": self.turn,
            "use": self.use,
        }


def flatten_action(action: Mapping[str, int]) -> List[int]:
    labels = ActionLabels.from_dict(action)
    return [labels.move, labels.strafe, labels.turn, labels.use]


def action_from_list(values: List[int]) -> Dict[str, int]:
    if len(values) != len(ACTION_HEADS):
        raise ValueError(f"Expected {len(ACTION_HEADS)} values, got {len(values)}")
    labels = ActionLabels(
        move=int(values[0]),
        strafe=int(values[1]),
        turn=int(values[2]),
        use=int(values[3]),
    )
    labels.validate()
    return labels.to_dict()
