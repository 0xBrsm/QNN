"""Head registry — central lookup for per-head HeadSpec.

Per-head specs live in their own modules (target.py, fire.py, …). This
file imports each and exposes a single ``HEADS`` dict by name so the
probe trainer and CLI dispatch by string. Adding a new head =
``HEADS[name] = NEW_SPEC`` here + a new file.
"""

from __future__ import annotations

from qnn.model.bench.attack import ATTACK
from qnn.model.bench.attack_look_style import ATTACK_LOOK_STYLE
from qnn.model.bench.attack_preattn import ATTACK_PREATTN
from qnn.model.bench.attack_preattn_oracle import ATTACK_PREATTN_ORACLE
from qnn.model.bench.spec import HeadSpec
from qnn.model.bench.target import TARGET
from qnn.model.bench.target_constant_query import TARGET_CONSTANT_QUERY
from qnn.model.bench.target_mlp_query import TARGET_MLP_QUERY
from qnn.model.bench.target_mlp_query_full_stack import TARGET_MLP_QUERY_FULL_STACK
from qnn.model.bench.target_self_query import (
    TARGET_SELF_QUERY, TARGET_SELF_QUERY_ENEMY,
)
from qnn.model.bench.target_weapon_query import TARGET_WEAPON_QUERY
from qnn.model.bench.weapon_aim import WEAPON_AIM
from qnn.model.bench.weapon_preattn import WEAPON_PREATTN


HEADS: dict[str, HeadSpec] = {
    TARGET.name:                     TARGET,
    TARGET_CONSTANT_QUERY.name:      TARGET_CONSTANT_QUERY,
    TARGET_MLP_QUERY.name:           TARGET_MLP_QUERY,
    TARGET_MLP_QUERY_FULL_STACK.name: TARGET_MLP_QUERY_FULL_STACK,
    TARGET_SELF_QUERY.name:          TARGET_SELF_QUERY,
    TARGET_SELF_QUERY_ENEMY.name:    TARGET_SELF_QUERY_ENEMY,
    TARGET_WEAPON_QUERY.name:        TARGET_WEAPON_QUERY,
    ATTACK.name:                 ATTACK,
    ATTACK_PREATTN.name:         ATTACK_PREATTN,
    ATTACK_PREATTN_ORACLE.name:  ATTACK_PREATTN_ORACLE,
    ATTACK_LOOK_STYLE.name:      ATTACK_LOOK_STYLE,
    WEAPON_AIM.name:             WEAPON_AIM,
    WEAPON_PREATTN.name:         WEAPON_PREATTN,
}
