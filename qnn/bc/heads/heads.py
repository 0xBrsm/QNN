"""Head registry — central lookup for per-head HeadSpec.

Per-head specs live in their own modules (target.py, fire.py, …). This
file imports each and exposes a single ``HEADS`` dict by name so the
probe trainer and CLI dispatch by string. Adding a new head =
``HEADS[name] = NEW_SPEC`` here + a new file.
"""

from __future__ import annotations

from qnn.bc.heads.fire import FIRE
from qnn.bc.heads.fire_token import FIRE_TOKEN
from qnn.bc.heads.spec import HeadSpec
from qnn.bc.heads.target import TARGET
from qnn.bc.heads.weapon_token import WEAPON_TOKEN


HEADS: dict[str, HeadSpec] = {
    TARGET.name:       TARGET,
    FIRE.name:         FIRE,
    FIRE_TOKEN.name:   FIRE_TOKEN,
    WEAPON_TOKEN.name: WEAPON_TOKEN,
}
