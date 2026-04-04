"""Observation buffer format constants.

Single source of truth for the obs buffer wire format in Python.
Must match qnn_obs.h on the C side.
"""

import numpy as np

# ---- v9 obs buffer layout (events folded into entity tokens) ----
# self_scalars:             14 × f32 =   56 bytes   offset 0
# self_weapon_id:            1 × i32 =    4          offset 56
# self_armor_type_id:        1 × i32 =    4          offset 60
# self_powerup_ids:          5 × i32 =   20          offset 64
# self_powerup_count:        1 × i32 =    4          offset 84
# self_movement_id:          1 × i32 =    4          offset 88
# self_cluster_id:           1 × i32 =    4          offset 92
# object_ids:           16×7 × i32 =  448          offset 96
# object_scalars:       16×17× f32 = 1088          offset 544
# object_mask:              16× u8  =   16          offset 1632
# object_route_cluster_ids:16×8×i32 =  512          offset 1648
# object_event_ids:    16×4×3× i32 =  768          offset 2160
# object_event_scalars:  16×4× f32 =  256          offset 2928
# object_event_counts:     16× u8  =   16          offset 3184
# spatial_ids:            9 × i32 =   36          offset 3200
# spatial_scalars:      9×10× f32 =  360          offset 3236
# action_history:       8×7 × f32 =  224          offset 3596
# total:                                            3820

OBS_BUFFER_SIZE = 3820
ACTION_SIZE = 44  # sizeof(qnn_action_t)

# Per-tick header emitted by demo worker collect mode.
TICK_HEADER_SIZE = 16  # tick(4) + steps(4) + tick_hz(4) + flags(2) + action_size(2)
TICK_HEADER_FORMAT = "<IIIHH"

TICK_MAGIC = b"QOBS"
TICK_MAGIC_SIZE = 4

FLAG_RESET = 0x01
FLAG_DONE = 0x02

# Observation buffer field layout: name → (offset, dtype, shape)
OBS_FIELDS = {
    "self_scalars":             (0,     np.float32, (14,)),
    "self_weapon_id":           (56,    np.int32,   (1,)),
    "self_armor_type_id":       (60,    np.int32,   (1,)),
    "self_powerup_ids":         (64,    np.int32,   (5,)),
    "self_powerup_count":       (84,    np.int32,   (1,)),
    "self_movement_id":         (88,    np.int32,   (1,)),
    "self_cluster_id":          (92,    np.int32,   (1,)),
    "object_ids":               (96,    np.int32,   (16, 7)),
    "object_scalars":           (544,   np.float32, (16, 17)),
    "object_mask":              (1632,  np.uint8,   (16,)),
    "object_route_cluster_ids": (1648,  np.int32,   (16, 8)),
    "object_event_ids":         (2160,  np.int32,   (16, 4, 3)),
    "object_event_scalars":     (2928,  np.float32, (16, 4)),
    "object_event_counts":      (3184,  np.uint8,   (16,)),
    "spatial_ids":              (3200,  np.int32,   (9,)),
    "spatial_scalars":          (3236,  np.float32, (9, 10)),
    "action_history":           (3596,  np.float32, (8, 7)),
}

# Action struct layout: name → (offset, dtype, shape)
ACTION_FIELDS = {
    "move":     (0,  np.float32, (2,)),
    "look":     (8,  np.float32, (2,)),
    "fire":     (16, np.int32,   ()),
    "jump":     (20, np.int32,   ()),
    "switch":   (24, np.int32,   ()),
    "recall_0": (28, np.int32,   ()),
    "recall_1": (32, np.int32,   ()),
    "recall_2": (36, np.int32,   ()),
    "recall_3": (40, np.int32,   ()),
}


def unpack_obs_buffer(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack a raw obs buffer into a dict of numpy arrays."""
    obs = {}
    for name, (offset, dtype, shape) in OBS_FIELDS.items():
        count = int(np.prod(shape))
        arr = np.frombuffer(raw, dtype=dtype, offset=offset, count=count).reshape(shape)
        if dtype == np.uint8 and "mask" in name:
            obs[name] = arr.astype(np.bool_)
        else:
            obs[name] = arr.copy()
    return obs


def unpack_action(raw: bytes) -> dict[str, np.ndarray]:
    """Unpack raw action bytes into a dict."""
    action = {}
    for name, (offset, dtype, shape) in ACTION_FIELDS.items():
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(raw, dtype=dtype, offset=offset, count=count)
        action[name] = arr.reshape(shape).copy() if shape else arr[0].copy()
    return action
