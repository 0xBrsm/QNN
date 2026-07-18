"""Per-weapon engine constants, parsed from the canonical X-macro header.

Single source of truth is ``src/engine/common/qnn_weapon.h``'s
``QNN_WEAPON_LIST`` (raw id order, Axe=1..LG=8): the cooldown column there is
the OPERATIVE re-fire cadence — the value that feeds QNN_MvdStampAttackFinished
and therefore the collect's ``input_mask`` bit0 (op-attack) stamping. Note this
differs from ``qnn.bc.weapon_physics.WEAPON_PHYSICS[..]["cooldown"]`` for LG
(0.1 literal W_Attack delay vs 0.2 effective think-chain cadence); any gap /
cadence math on op-attack discharge events must use THIS table.

Parsed at import (same pattern as ``qnn.bc.attack_vocab`` ↔ qnn_demo_sounds.h)
so Python can never drift from what the engine ships. Torch-free leaf module.

Exposes:
  COOLDOWN_SEC : (9,) float64, index = raw/impulse weapon id (0 = none → 0.0)
  WEAPON_LABELS: {raw id -> short label} ("Axe", "SG", ...)
"""
from __future__ import annotations

import os
import re

import numpy as np

_HDR = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "engine", "common", "qnn_weapon.h",
)

N_WEAPONS = 8  # raw ids 1..8


def _load() -> tuple[np.ndarray, dict[int, str]]:
    txt = open(_HDR).read()
    m = re.search(r"#define\s+QNN_WEAPON_LIST\(X\)(.*?)\n\n", txt, re.S)
    body = m.group(1) if m else txt
    rows = re.findall(
        r'X\(\s*QNN_WEAPON_\w+\s*,\s*"([^"]+)"\s*,\s*([0-9.]+)f?\s*\)', body)
    if len(rows) != N_WEAPONS:
        raise RuntimeError(
            f"parsed {len(rows)} weapon rows from {_HDR}, expected {N_WEAPONS}")
    cd = np.zeros(N_WEAPONS + 1, dtype=np.float64)
    labels: dict[int, str] = {}
    for i, (label, sec) in enumerate(rows, start=1):
        cd[i] = float(sec)
        labels[i] = label
    return cd, labels


COOLDOWN_SEC, WEAPON_LABELS = _load()
