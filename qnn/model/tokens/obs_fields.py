"""Obs field catalog — declarative token inputs for core and bench models.

Every bench ablation that builds a token from observations names the
*fields* it wants (scalars + vocab embeds) and composes them with the
``FieldSource`` primitives below. This module replaces the magic offsets
(``_VEL_BEGIN``), duplicated weapon-subject tables, and copy-pasted
mask/sum/einsum snippets that each bespoke probe used to re-derive.

Two halves:

* **Field specs** — a registry mapping a field name to *how it is pulled
  from the dequanted obs dict*. ``ObsAccessor`` (see ``obs_accessor.py``)
  resolves these; nothing here holds tensors or parameters.
    - ``ScalarSpec`` — a fixed-width float contribution. Either a plain
      slice of a (dequanted) obs tensor, or a small computed quantity
      (weapon static-table lookup, held-weapon readiness gather, the
      engagement EMA aux scalar).
    - ``VocabSpec`` — a discrete id field looked up in a shared embedding
      table, with a mask / reduce rule.

* **FieldSource primitives** — declarative, parameter-free token parts
  that ``TokenBuilder`` interprets:
    - ``ScalarGroup(names)`` — the named scalars, concatenated and pushed
      through ONE ``Linear(sum_width, d_model)`` (the builder owns it).
    - ``VocabEmbed(name)`` — masked embedding lookup, additive.
    - ``VocabSum(name)``  — masked embedding lookup summed over the id axis.
    - ``WeaponReadiness()`` — the per-weapon ``bw,wd->bd`` einsum.
    - ``KindTag(kind)`` — add the encoder's token-kind embedding (self /
      entity / spatial). Encoder self-blocks use it; head bundles do not.

Constants are re-exported from their owners (``qnn.model.dequant``,
``qnn.bc.weapon_physics``, ``qnn.vocab``) — never re-declared here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Single-source constants from their owners. WEAPON_SUBJECT_IDS / WT_* are
# re-exported (token_builder + obs_accessor import them from here);
# MODEL_TOKEN_SCALAR_DIM is used below for the weapon_static field width.
from qnn.bc.weapon_physics import (  # noqa: F401  (WT_* re-exported)
    MODEL_TOKEN_SCALAR_DIM, WT_DAMAGE, WT_RADIUS,
)
from qnn.model.dequant import WEAPON_SUBJECT_IDS  # noqa: F401  (re-exported)
from qnn.schema import SELF_SCALAR_DIM

# Token-kind rows mirror qnn.model.transformer (self / entity / spatial).
KIND_SELF = 0
KIND_ENTITY = 1
KIND_SPATIAL = 2


# ── Scalar fields ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ScalarSpec:
    """One named scalar contribution of fixed ``width``.

    ``slice_key`` set ⇒ the field is ``dequant_obs[slice_key][..., start:stop]``.
    ``slice_key`` None ⇒ the field is *computed* and resolved by the
    ``ObsAccessor`` method registered under ``compute`` (e.g. a weapon
    static-table lookup or an aux signal). Exactly one of the two applies.
    """
    name: str
    width: int
    slice_key: str | None = None
    start: int = 0
    stop: int = 0
    compute: str | None = None   # accessor method name for computed fields


def _slice(name: str, key: str, start: int, stop: int) -> ScalarSpec:
    return ScalarSpec(name, stop - start, slice_key=key, start=start, stop=stop)


def _computed(name: str, width: int, compute: str) -> ScalarSpec:
    return ScalarSpec(name, width, compute=compute)


SCALAR_FIELDS: dict[str, ScalarSpec] = {
    # Full monolithic self scalar bundle. The canonical ObsEmbedding uses this
    # as one ScalarGroup so legacy self_proj weights map 1:1 into TokenBuilder.
    "self_scalars":     _slice("self_scalars",     "self_scalars",          0, SELF_SCALAR_DIM),
    # Prefer the canonical post-dequant subtoken tensors over slicing the
    # legacy 17-wide self_scalars (less index-fragile).
    "health_armor":    _slice("health_armor",    "self_state_scalars",   0, 2),
    "attack_finished": _slice("attack_finished", "self_arsenal_scalars", 0, 1),
    "velocity":        _slice("velocity",         "self_motion_scalars",  0, 3),
    "view_pitch":      _slice("view_pitch",       "self_motion_scalars",  3, 4),
    "vel_pitch":       _slice("vel_pitch",        "self_motion_scalars",  0, 4),
    # Computed (accessor owns the logic; see ObsAccessor).
    "weapon_static":   _computed("weapon_static",   MODEL_TOKEN_SCALAR_DIM, "weapon_static"),
    "weapon_dmg_rad":  _computed("weapon_dmg_rad",  2, "weapon_dmg_rad"),
    "held_readiness":  _computed("held_readiness",  1, "held_readiness"),
    # look_delta = look[t-1] - look[t-2]: the frame-to-frame change in the
    # anchor-relative look vector (~0 under steady rotation; ≈ angular
    # acceleration, NOT angular velocity — a single look vector is that).
    # A first-class self-motion obs field; formerly reconstructed in-model
    # from a carried look history, now sourced directly at the obs boundary.
    "look_delta":         _slice("look_delta",            "look_delta",  0, 3),
    "engagement":      _computed("engagement",      1, "engagement"),
}


# ── Vocab (discrete id) fields ───────────────────────────────────────

@dataclass(frozen=True, slots=True)
class VocabSpec:
    """One named discrete-id field looked up in a shared embedding table.

    ``table`` selects which embedding the builder uses ("entity" or
    "movement"). ``masked`` zeroes the contribution where ``id <= 0``
    (the canonical NONE sentinel). ``reduce`` is "none" (single id per
    row), "sum" (sum a (B, P) id bundle, e.g. powerups), or "readiness"
    (the per-weapon readiness einsum special case).
    """
    name: str
    obs_key: str
    table: str            # "entity" | "movement"
    masked: bool
    reduce: str = "none"  # "none" | "sum" | "readiness"


VOCAB_FIELDS: dict[str, VocabSpec] = {
    "armor_type":       VocabSpec("armor_type",       "self_armor_type_id",       "entity",   masked=True),
    "weapon_id":        VocabSpec("weapon_id",        "self_weapon_id",           "entity",   masked=True),
    "movement_id":      VocabSpec("movement_id",      "self_movement_id",         "movement", masked=False),
    "powerup_state":    VocabSpec("powerup_state",    "self_state_powerup_ids",   "entity",   masked=True, reduce="sum"),
    "powerup_arsenal":  VocabSpec("powerup_arsenal",  "self_arsenal_powerup_ids", "entity",   masked=True, reduce="sum"),
    "powerup_motion":   VocabSpec("powerup_motion",   "self_motion_powerup_ids",  "entity",   masked=True, reduce="sum"),
    "powerup_all":      VocabSpec("powerup_all",      "self_powerup_ids",         "entity",   masked=True, reduce="sum"),
    "weapon_readiness": VocabSpec("weapon_readiness", "self_weapon_readiness",    "entity",   masked=False, reduce="readiness"),
}


# ── FieldSource primitives (declarative, parameter-free) ─────────────

@dataclass(frozen=True, slots=True)
class ScalarGroup:
    """Named scalar fields concatenated → one Linear(sum_width, d_model).

    Field order within ``names`` fixes the Linear's input layout, so it is
    set by the spec (never by probe.json) to keep parameter layout
    deterministic.
    """
    names: tuple[str, ...]

    def __init__(self, names) -> None:  # accept list/tuple/varargs-ish
        object.__setattr__(self, "names", tuple(names))

    @property
    def width(self) -> int:
        return sum(SCALAR_FIELDS[n].width for n in self.names)


@dataclass(frozen=True, slots=True)
class VocabEmbed:
    """Masked embedding lookup for a single-id vocab field, additive."""
    name: str


@dataclass(frozen=True, slots=True)
class VocabSum:
    """Masked embedding lookup summed over a (B, P) id bundle (powerups)."""
    name: str


@dataclass(frozen=True, slots=True)
class WeaponReadiness:
    """Per-weapon readiness × entity_embed einsum (bw,wd->bd)."""


# Ammo pools, in the fixed order the dequant emits ``self_ammo_pools`` (B, 4).
# The four ammo TYPES, not weapons: shells feed SG/SSG, nails NG/SNG, rockets
# GL/RL, cells LG. Order fixes the AmmoPools embedding-table row layout, so it
# lives here (never in probe.json), same discipline as ScalarGroup field order.
AMMO_POOL_NAMES = ("shells", "nails", "rockets", "cells")
NUM_AMMO_POOLS = len(AMMO_POOL_NAMES)


@dataclass(frozen=True, slots=True)
class AmmoPools:
    """Per-ammo-pool fraction × learned ammo-type embed einsum (bp,pd->bd).

    The post-masking arsenal signal (P2): 4 ammo-type embeddings each scaled by
    that pool's normalized ammo fraction. Weapon identity/ownership are owned by
    the feasibility mask, so this input carries only the graded ammo level. The
    embedding table is a NEW parameter owned by the TokenBuilder (not tied to the
    entity vocab — ammo types are not entities)."""


@dataclass(frozen=True, slots=True)
class KindTag:
    """Add the encoder's token-kind embedding row (self/entity/spatial)."""
    kind: int = KIND_SELF


# A token is an ordered tuple of these.
FieldSource = "ScalarGroup | VocabEmbed | VocabSum | WeaponReadiness | AmmoPools | KindTag"


def canonical_self_fields(include_weapon_id: bool) -> tuple:
    """Fields for the production monolithic self token.

    This mirrors the pre-TokenBuilder ObsEmbedding self block exactly:
    full-width self scalar projection, self kind tag, armor type, movement,
    optional held-weapon identity, and all powerups.

    The TokenSpec twin is ``qnn.model.graph.spec.monolithic_self_token``;
    a unit test asserts ``token_fields(monolithic_self_token(x))`` equals
    this tuple (tokens must not import graph, so they cannot share code).
    """
    fields: list[object] = [
        ScalarGroup(["self_scalars"]),
        KindTag(KIND_SELF),
        VocabEmbed("armor_type"),
        VocabEmbed("movement_id"),
    ]
    if include_weapon_id:
        fields.append(VocabEmbed("weapon_id"))
    fields.append(VocabSum("powerup_all"))
    return tuple(fields)


# Canonical "motion token" field list — the single source of truth so the
# motion subtoken is identical everywhere it is built (move/look move-token
# heads, the split-self embedding, the weapon arsenal probe). Includes look_delta
# (angular velocity); historically split_self/weapon used vel_pitch alone and
# omitted it, which diverged from the move/look heads. Order fixes the Linear
# layout, so it lives here, never in probe.json.
MOTION_FIELDS = (
    ScalarGroup(["velocity", "view_pitch", "look_delta"]),
    VocabEmbed("movement_id"),
    VocabSum("powerup_motion"),
)

# Canonical lean "state token" — the self state subtoken: health/armor scalars,
# armor_type, state powerups (PENT/RING/MEGAHEALTH). Deliberately EXCLUDES
# weapon_id (an incumbent leak for the weapon head) and powerup_arsenal (lives in
# the arsenal token). The move head uses a separate, richer state bundle that
# intentionally includes weapon_id/attack_finished/arsenal — that is correct for
# a motor head but must NOT be reused where an incumbent leak matters.
SELF_STATE_FIELDS = (
    ScalarGroup(["health_armor"]),
    VocabEmbed("armor_type"),
    VocabSum("powerup_state"),
)
