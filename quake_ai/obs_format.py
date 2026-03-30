"""Observation buffer format constants.

Single source of truth for the obs buffer wire format in Python.
Must match qnn_obs.h on the C side.
"""

import numpy as np

OBS_BUFFER_SIZE = 15892
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
    "self_scalars":             (0,     np.float32, (23,)),
    "self_weapon_id":           (92,    np.int32,   (1,)),
    "self_movement_id":         (96,    np.int32,   (1,)),
    "self_cluster_id":          (100,   np.int32,   (1,)),
    "object_ids":               (104,   np.int32,   (64, 5)),
    "object_scalars":           (1384,  np.float32, (64, 13)),
    "object_mask":              (4712,  np.uint8,   (64,)),
    "object_route_cluster_ids": (4776,  np.int32,   (64, 8)),
    "event_ids":                (6824,  np.int32,   (256, 4)),
    "event_scalars":            (10920, np.float32, (256, 3)),
    "event_owner":              (13992, np.int32,   (256,)),
    "event_mask":               (15016, np.uint8,   (256,)),
    "spatial_ids":              (15272, np.int32,   (9,)),
    "spatial_scalars":          (15308, np.float32, (9, 10)),
    "action_history":           (15668, np.float32, (8, 7)),
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
        # Convert uint8 mask fields to bool
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
