"""ctypes wrapper for ``QNN_HitTest`` — single source of truth for
whether a weapon would hit a given target with the player's current aim.

The C implementation (``src/engine/common/qnn_hit_test.{c,h}``) is a
verbatim port of ``qnn.bc.weapon_physics``'s slot_would_be_hit logic.
Both the labeler and the BC dataloader's per-frame entity_hit_test
precompute call into this wrapper; the engine links the C object
directly for live-play. Single function, three call sites.

Library path resolution:
  1. ``QNN_HIT_TEST_LIB`` env var (absolute path).
  2. ``<repo>/artifacts/bin/libqnn_hit_test.so`` relative to this file.
  3. ctypes' default loader search (LD_LIBRARY_PATH etc.).

The library is built by ``scripts/build_hit_test_lib.sh``; CI / dev
containers should run it once at setup.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_LIB_NAME = "libqnn_hit_test.so"


def _load_library() -> ctypes.CDLL:
    env_path = os.environ.get("QNN_HIT_TEST_LIB")
    if env_path:
        return ctypes.CDLL(env_path)
    # Default: <repo>/artifacts/bin/libqnn_hit_test.so. This file lives
    # at <repo>/src/qnn/hit_test.py; walk up twice to repo root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / "artifacts" / "bin" / _LIB_NAME
    if candidate.exists():
        return ctypes.CDLL(str(candidate))
    # Last resort: let ctypes' resolver search standard paths.
    return ctypes.CDLL(_LIB_NAME)


_LIB = _load_library()

_QNN_HitTest = _LIB.QNN_HitTest
_QNN_HitTest.argtypes = [
    ctypes.c_int,                                       # weapon_id_impulse
    ctypes.POINTER(ctypes.c_float),                     # look_unit[3]
    ctypes.POINTER(ctypes.c_float),                     # rel[3]
    ctypes.POINTER(ctypes.c_float),                     # vel[3]
    ctypes.POINTER(ctypes.c_float),                     # half_extents[3]
]
_QNN_HitTest.restype = ctypes.c_bool


def _to_f3(vec: np.ndarray) -> ctypes.Array:
    """Convert a 3-vector to a contiguous ctypes float array."""
    a = np.ascontiguousarray(vec, dtype=np.float32)
    if a.shape != (3,):
        raise ValueError(f"expected 3-vector, got shape {a.shape}")
    return (ctypes.c_float * 3)(*a)


def hit_test(weapon_id_impulse: int, look_unit: np.ndarray, rel: np.ndarray,
             vel: np.ndarray, half_extents: np.ndarray) -> bool:
    """Single-frame wrapper around QNN_HitTest. See header docs.

    All vectors are 3-element numpy arrays in world units. ``look_unit``
    should be unit-norm; the C side defends against degenerate inputs.
    """
    return bool(_QNN_HitTest(
        int(weapon_id_impulse),
        _to_f3(look_unit),
        _to_f3(rel),
        _to_f3(vel),
        _to_f3(half_extents),
    ))


def hit_test_batch(weapon_id_impulse: int, look_unit: np.ndarray,
                   rel: np.ndarray, vel: np.ndarray,
                   half_extents: np.ndarray) -> np.ndarray:
    """Per-slot batch helper: shapes (3,), (N,3), (N,3), (N,3) → (N,) bool.

    Loops in Python around the C call — fine for offline precompute and
    test parity, not hot enough for per-frame in the inner training
    loop. The BC dataloader precomputes once at shard-load and caches,
    so this is fast enough in practice.
    """
    rel = np.asarray(rel, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    half = np.asarray(half_extents, dtype=np.float32)
    n = rel.shape[0]
    out = np.zeros(n, dtype=bool)
    look_arr = _to_f3(look_unit)
    for i in range(n):
        out[i] = bool(_QNN_HitTest(
            int(weapon_id_impulse), look_arr,
            _to_f3(rel[i]), _to_f3(vel[i]), _to_f3(half[i]),
        ))
    return out
