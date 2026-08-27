#!/usr/bin/env python3
"""Single source of truth for the fire-sound vocabulary, parsed straight
from the engine's canonical X-macro header so the Python audit can never
drift from what qnn_event.c / qw_classifier.c actually classify.

Promoted from the loose ``scripts/fire_vocab.py`` into the package so every
importer pulls the same vocab (no more ``sys.path.insert(0, 'scripts')``).

Exposes:
  SOUND_WEAPON : {sound_path -> weapon class 1..8}
  WNAME        : {class -> short name}
  CONTINUOUS   : frozenset of continuous-fire classes (NG/SNG/LG)
"""
from __future__ import annotations

import os
import re

# qnn_demo_sounds.h lives at src/engine/common/ — three parents up from this
# module (src/qnn/bc/fire_vocab.py -> src/) then engine/common.
_HDR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "engine", "common", "qnn_demo_sounds.h",
)

# subject token (qnn.h: case N -> QNN_SUBJECT_*) -> collect act_attack class
_SUBJECT_CLASS = {
    "QNN_SUBJECT_AXE": 1,
    "QNN_SUBJECT_SHOTGUN": 2,
    "QNN_SUBJECT_SUPER_SHOTGUN": 3,
    "QNN_SUBJECT_NAILGUN": 4,
    "QNN_SUBJECT_SUPER_NAILGUN": 5,
    "QNN_SUBJECT_GRENADE_LAUNCHER": 6,
    "QNN_SUBJECT_ROCKET_LAUNCHER": 7,
    "QNN_SUBJECT_THUNDERBOLT": 8,
}
WNAME = {1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG", 6: "GL", 7: "RL", 8: "LG"}

# Continuous-fire weapons stream a 0.1s think-chain (player_nail1<->player_nail2,
# player_light1<->player_light2) firing one projectile sound every ~0.1s while
# the trigger is held, so their raw sound count is ~2x the W_Attack trigger-pull
# count. See src/docs/mvd-attack-audit.md §Round 4. The validator collapses these
# sound streams into trigger events when comparing against an op-fire reference.
CONTINUOUS = frozenset({4, 5, 8})  # NG, SNG, LG


def _load() -> dict[str, int]:
    txt = open(_HDR).read()
    # isolate the QNN_FIRE_SOUND_LIST(X) macro body
    m = re.search(r"#define\s+QNN_FIRE_SOUND_LIST\(X\)(.*?)\n\n", txt, re.S)
    body = m.group(1) if m else txt
    out: dict[str, int] = {}
    for path, subj in re.findall(r'X\(\s*"([^"]+)"\s*,\s*(QNN_SUBJECT_\w+)\s*\)', body):
        if subj not in _SUBJECT_CLASS:
            raise SystemExit(f"unknown subject {subj} for {path} — update _SUBJECT_CLASS")
        out[path] = _SUBJECT_CLASS[subj]
    if not out:
        raise SystemExit(f"parsed no fire sounds from {_HDR}")
    return out


def _load_jump() -> frozenset[str]:
    """Parse QNN_JUMP_SOUND_LIST(X) — the self jump-initiation sound(s)."""
    txt = open(_HDR).read()
    m = re.search(r"#define\s+QNN_JUMP_SOUND_LIST\(X\)(.*?)\n\n", txt, re.S)
    body = m.group(1) if m else txt
    out = frozenset(re.findall(r'X\(\s*"([^"]+)"\s*\)', body))
    if not out:
        raise SystemExit(f"parsed no jump sounds from {_HDR}")
    return out


SOUND_WEAPON = _load()
# self jump-initiation sounds (player/plyrjmp8.wav). One per jump.
JUMP_SOUNDS = _load_jump()

if __name__ == "__main__":
    for p, c in SOUND_WEAPON.items():
        print(f"{c} {WNAME[c]:>4}  {p}")
    print("jump:", sorted(JUMP_SOUNDS))
