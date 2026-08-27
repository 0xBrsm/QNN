"""Load exported per-map probe tables for closed-loop probe-grid obs.

The tables are written by ``qnn.bc.probe_grid --export-tables`` (per map:
world-anchored probe panoramas + positions, one probe per Detour navmesh
poly center). Closed-loop, ``assign_closed_loop`` produces the exact obs
fields the probe-grid model trained on — the same ``MapProbes.assign`` math
(yaw roll into the view frame + 5-scalar relative-pose encoding).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qnn.bc.probe_grid import MapProbes


def load_tables(path: str | Path) -> tuple[dict[str, MapProbes], dict]:
    """npz → ({map_name: MapProbes}, meta dict)."""
    data = np.load(path)
    meta = json.loads(bytes(data["__meta__"]).decode())
    maps: dict[str, MapProbes] = {}
    for name in meta["maps"]:
        maps[name] = MapProbes(
            data[f"{name}__positions"],
            data[f"{name}__panoramas"],
        )
    return maps, meta


def assign_closed_loop(
    probes: MapProbes, pos: np.ndarray, view_yaw: float, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One live pose → (probe_atlas (K, 11, 24) u8, probe_offsets (K, 5) f16)."""
    pano, offsets = probes.assign(
        np.asarray(pos, dtype=np.float64).reshape(1, 3),
        np.asarray([view_yaw], dtype=np.float64),
        k=k, workers=1,
    )
    return pano[0], offsets[0]
