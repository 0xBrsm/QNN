"""Canonical action helpers for mixed continuous/discrete Quake control.

Action label specification (v4)
===============================

The model speaks two action representations:

  • **Engine-facing** — what `qnn_input.c` reads from `qnn_pending_action`
    over the inference wire and what live-play / PPO consume.  This is the
    legacy continuous form (``ActionLabels`` below) and stays float[3] for
    move + look so the engine bridge does not need to change.

  • **Training-facing** — what the BC corpus stores on disk and what the
    model's move head emits.  Move is **3 categorical axes** (fb, lr, ud),
    each a 3-class softmax over {neg, none, pos}.  This matches Quake's
    actual input space (per-axis ternary keys) and recovers the recall
    advantage that the binary-buttons experiment lost.

All vectors are in the **view-relative frame** (forward, right, up),
the same coordinate space used for entity ``rel[]`` positions and
spatial ``dir[]`` sectors in the observation tokens.

ENGINE-FACING (ActionLabels):

  move  float[3]  View-relative direction / sv_maxspeed.
                  ``move[0]``=forward sign, ``move[1]``=right sign,
                  ``move[2]``=up sign.  Each component lands in
                  {-1, 0, +1} after categorical decode.  qnn_input.c
                  maps move[0]/move[1] → cmd.forwardmove / sidemove
                  and ``move[2] > 0.5`` → +jump (which is also +moveup
                  in water).

  look  float[3]  Next frame's forward direction in the current frame's
                  view-relative coordinates.  Approximately unit length.
                  [1, 0, 0] = no turn.

  fire    int     0 = not firing, 1 = firing.
  switch  int     0 = no switch, 1-6 = weapon idx.

TRAINING-FACING (on-disk corpus):

  move    uint8 packed (one bit per direction — mirrors the
                       input_mask byte produced by QNN_PackInputMask):
            bit 0 = attack press
            bit 1 = fb neg            bit 2 = fb pos
            bit 3 = lr neg            bit 4 = lr pos
            bit 5 = ud neg            bit 6 = ud pos (swim-up / jumppad
                                                       upmove>0, NOT
                                                       jump button)
            bit 7 = jump button press
            Loader (qnn.bc.train._unpack_move_axes) collapses each axis
            pair into ``uint8[T, 3]`` class indices in {0=neg,1=none,
            2=pos} via ``class = 1 + pos_bit - neg_bit``.  Attack and
            jump bits are extracted separately.

  look    float16[3]  Same semantics as engine-facing.  fp16 is finer
                      than the source mouse quantization (~0.066°) so
                      the precision drop is below the signal floor.

  fire    uint8       0/1.
  weapon  uint8       raw engine weapon byte:
                        0 = no weapon held (pre-spawn / dead / transitional),
                        1..8 = Quake weapon id in impulse order (axe..LG).
                      No-weapon frames stay in the corpus so move/fire/look
                      labels still train; the 8-class weapon head masks
                      them out of its CE loss via ignore_index=-100.
                      The engine-facing `switch` idx 0-6 is derived from
                      the weapon head's argmax at inference time; it is
                      not stored on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

LOOK_BASE_DEGREES_PER_COUNT = 0.066
LOOK_NONLINEAR_BASE_COUNT = 256.0
LOOK_HIGH_GAIN_MULTIPLIER = 2.0  # sensitivity 3 near center, 6 at the edge
LOOK_DEADZONE = 0.03
MOVE_DEADZONE = 0.25
# Weapon action: 8 model classes (0..7 = axe..lightning).  The PPO /
# select_actions action emits a class index in this range; the engine
# bridge maps class+1 → Quake impulse byte 1..8 before packing into
# the wire so the engine receives a direct impulse.  No "no switch"
# class — the model picks a weapon every frame; emitting the
# currently-held weapon is a server-side no-op.
WEAPON_ACTION_SIZE = 8
ACTION_SCHEMA_VERSION = 3

# Move = 3 categorical axes, each a 3-class softmax {neg, none, pos}.
# Class index encoding per axis:
#   0 = neg (back / left / down)
#   1 = none (axis released)
#   2 = pos (forward / right / up)
# Decoded engine-facing axis value = class - 1, i.e. {-1, 0, +1}.
MOVE_AXES = 3
MOVE_AXIS_NAMES: tuple[str, ...] = ("fb", "lr", "ud")
MOVE_AXIS_INDEX: Dict[str, int] = {n: i for i, n in enumerate(MOVE_AXIS_NAMES)}
MOVE_AXIS_CLASSES = 3
MOVE_CLASS_NEG = 0
MOVE_CLASS_NONE = 1
MOVE_CLASS_POS = 2
# Deadband for converting a continuous axis value (from the C collector or
# the inference encode) into a categorical class.  The C-side cmd values
# normalized by maxspeed sit near 0, ±0.6 (running), or ±1 (clamped); 0.1
# cleanly separates "pressed" from "released" without dropping any real
# press.
MOVE_AXIS_THRESHOLD = 0.1

ACTION_HEADS = {
    "move": MOVE_AXES * MOVE_AXIS_CLASSES,  # 9 logits → reshape (3 axes, 3 classes)
    "look": 3,
    "attack": 2,
    "weapon": WEAPON_ACTION_SIZE,
}
CONTINUOUS_ACTION_HEADS = frozenset({"look"})

# Deterministic head ordering (Python 3.7+ dict preserves insertion order).
# Shared by checkpoint_converter and other modules that need a canonical order.
HEAD_ORDER: list[str] = list(ACTION_HEADS.keys())
DISCRETE_ACTION_HEADS = frozenset({"move", "attack", "weapon"})


def clamp_weapon(value: int) -> int:
    """Clamp a weapon impulse byte to the engine wire range.

    ActionLabels.weapon carries the engine impulse byte 0..8 directly
    (0 = no impulse this frame, 1..8 = axe..lightning).  PPO and BC
    inference always emit 1..8; 0 only appears when the weapon head
    is disabled via head_loss_weights.
    """
    return max(0, min(int(value), WEAPON_ACTION_SIZE))


def clamp_unit(value: float) -> float:
    return max(-1.0, min(float(value), 1.0))


def move_axes_from_continuous(move: Sequence[float]) -> tuple[int, int, int]:
    """Threshold a continuous view-relative move into per-axis class indices.

    Used by the BC collector.  Each axis becomes one of:
      0 = neg (component < -threshold)
      1 = none (|component| ≤ threshold)
      2 = pos (component > +threshold)
    """
    t = MOVE_AXIS_THRESHOLD
    out: list[int] = []
    for i in range(MOVE_AXES):
        v = float(move[i])
        if v >  t:   out.append(MOVE_CLASS_POS)
        elif v < -t: out.append(MOVE_CLASS_NEG)
        else:        out.append(MOVE_CLASS_NONE)
    return out[0], out[1], out[2]


def pack_move_axes(fb: int, lr: int, ud: int) -> int:
    """Pack three axis class indices (each in 0..2) into a uint8.

    Layout: bits 0-1 = fb, bits 2-3 = lr, bits 4-5 = ud, bits 6-7 reserved.
    """
    return (int(fb) & 0x3) | ((int(lr) & 0x3) << 2) | ((int(ud) & 0x3) << 4)


def unpack_move_axes(packed: int) -> tuple[int, int, int]:
    """Inverse of pack_move_axes."""
    p = int(packed)
    return p & 0x3, (p >> 2) & 0x3, (p >> 4) & 0x3


def continuous_from_move_axes(axes: Sequence[int]) -> tuple[float, float, float]:
    """Decode three axis class indices into engine-facing float[3] move.

    Class 0 = -1, class 1 = 0, class 2 = +1.  Up set to ±1.0 so qnn_input.c's
    >0.5 jump threshold registers (the same channel encodes +moveup in water).
    """
    return tuple(float(int(axes[i]) - MOVE_CLASS_NONE) for i in range(MOVE_AXES))  # type: ignore[return-value]


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
        float(labels.move[2]),
        float(labels.look[0]),
        float(labels.look[1]),
        float(labels.look[2]),
        float(labels.attack),
        float(labels.weapon) / float(WEAPON_ACTION_SIZE - 1),  # class 0..7 → [0, 1]
    ]


@dataclass(slots=True)
class ActionLabels:
    move: tuple[float, float, float] = (0.0, 0.0, 0.0)
    look: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attack: int = 0
    weapon: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionLabels":
        move = _vector3(payload.get("move"), default=(0.0, 0.0, 0.0))
        look = _vector3(payload.get("look"), default=(0.0, 0.0, 0.0))

        action = cls(
            move=move,
            look=look,
            attack=int(payload.get("attack", 0)),
            weapon=clamp_weapon(int(payload.get("weapon", 0))),
        )
        action.validate()
        return action

    def validate(self) -> None:
        for value in (*self.move, *self.look):
            if float(value) < -1.0 or float(value) > 1.0:
                raise ValueError("continuous action values must be in [-1, 1]")
        if self.attack < 0 or self.attack >= ACTION_HEADS["attack"]:
            raise ValueError(f"attack out of range [0, {ACTION_HEADS['attack']})")
        # weapon is the engine impulse byte 0..8 (0 = no impulse,
        # 1..8 = axe..lightning).  ACTION_HEADS["weapon"]=8 is the
        # PPO model class count; the impulse range is 1..8 plus the
        # 0 no-op sentinel, hence the wider check here.
        if self.weapon < 0 or self.weapon > WEAPON_ACTION_SIZE:
            raise ValueError(f"weapon impulse out of range [0, {WEAPON_ACTION_SIZE}]")

    def to_dict(self) -> Dict[str, object]:
        return {
            "move": [float(self.move[0]), float(self.move[1]), float(self.move[2])],
            "look": [float(self.look[0]), float(self.look[1]), float(self.look[2])],
            "attack": int(self.attack),
            "weapon": int(self.weapon),
        }


def flatten_action(action: Mapping[str, object]) -> List[float]:
    labels = ActionLabels.from_dict(action)
    return [
        float(labels.move[0]),
        float(labels.move[1]),
        float(labels.move[2]),
        float(labels.look[0]),
        float(labels.look[1]),
        float(labels.look[2]),
        float(labels.attack),
        float(labels.weapon),
    ]


def action_from_list(values: Sequence[float]) -> Dict[str, object]:
    if len(values) != 8:
        raise ValueError(f"Expected 8 values, got {len(values)}")
    payload: Dict[str, object] = {
        "move": [float(values[0]), float(values[1]), float(values[2])],
        "look": [float(values[3]), float(values[4]), float(values[5])],
        "attack": int(round(float(values[6]))),
        "weapon": int(round(float(values[7]))),
    }
    return ActionLabels.from_dict(payload).to_dict()


def idle_action() -> Dict[str, object]:
    return ActionLabels().to_dict()
