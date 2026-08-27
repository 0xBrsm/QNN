"""Canonical engine-native obs field table.

Parallel to :mod:`qnn.vocab`: a single source of truth for the on-disk
and on-the-wire encoding of every observation field, plus the
dequantization to model-facing floats applied by the model's
``SelfDequantizer`` (and, later, the analogous entity / spatial
dequantizers).

Design principle
----------------
The wire format mirrors the QW network protocol's per-field encoding
(matching ``qnn_io.h`` on the C side), plus QC-internal state that the
protocol does not transmit (``attack_finished``). The collect pipeline
keeps everything that requires world geometry or cross-tick state
(spatial sectors, entity oracle, pathing). All normalization to floats
happens model-side via this table.

Native-width policy
-------------------
Per-field native dtype is chosen to match how the engine actually
carries the value:

- Health, armor, ammo, score — **u8** (engine byte ranges).
- Distances and coordinates (``rel``, ``path``, ``path_dist``) — **i16/u16**
  raw Quake units. See "Deviations from
  Quake" below for the precision note.
- Velocities — **i16** clamped to ±MAX_VELOCITY (sv_maxvelocity).
- Time scalars (``attack_finished``, ``eta``) —
  **f16 seconds**. u8 fails because the meaningful range spans 0.1s up
  to 60s and we need ~10ms precision at the low end.
- Already-normalized [0,1] floats (``facing``, ``team``, ``score``) —
  **u8** with 1/255 precision.
- Spatial depth-atlas codes — **u8**, with codes 0–14 indexing the distance
  ladder and 15 denoting no hit inside the band's range.
- Categorical IDs — **u8** (vocab fits in <256).
- ``cl.items`` — **i32**, matches Quake's ``int items``.
- ``act_target_probs`` — NOT on disk. Recomputed at training start
  from obs+actions by ``qnn.bc.train._compute_target_probs`` (~3 µs/frame
  CPU, ~25s for an 8M-frame corpus). Decouples LabelerConfig from the
  cache fingerprint.

Deviations from Quake (with rationale)
--------------------------------------
Most fields preserve Quake's native widths. The places where we
deviate, and why:

1. **effective_armor (u8)** vs engine's ``(STAT_ARMOR u8, armortype
   float)`` pair. We multiply ``raw_armor × armor_type_factor`` and
   round to a single u8 (range 0..160 fits with one byte to spare).
   Why: the *model's* signal is buffer strength; the armor type still
   reaches the model via the ``cl.items`` bits (IT_ARMOR1/2/3) which
   the dequantizer extracts as a categorical embedding. See the
   armor-encoding project memory.

2. **rel / path coordinates (i16 1-unit)** vs the QW protocol's
   ``MSG_WriteCoord`` (i16 in 1/8-unit fixed point, range ±4096). We
   keep i16 but store at 1-unit precision instead of 1/8-unit. Why:
   1-unit ≈ 2.5cm is already finer than any model-meaningful threshold
   (combat distances are 100+ units), and the wider ±32k range
   tolerates view-frame rotation without saturation on edge cases.

3. **half_extents (u8 saturating)** vs engine's float bbox half-sizes.
   Saturates at 255 raw units. Why: players/items/most movers are
   <128; saves 3 B/idx × 16 indices = 48 B/frame; rare custom-map
   movers >255 get clipped but the model's response to "huge mover"
   is indistinguishable above that anyway.

4. **dist removed entirely** from the wire (was a redundant field
   alongside rel). Recomputed by the dequantizer via
   ``|rel| / DIST_SCALE``. Zero precision loss; saves 2 B/idx.

5. **act_target_probs NOT on disk at all** — the target labeler is
   ~3 µs/frame and runs deterministically on (obs, actions, config).
   We recompute it once at training start instead of caching, which
   decouples LabelerConfig from the cache fingerprint and avoids any
   lossy on-disk compression of the multi-hot tail. ~25s startup
   overhead for an 8M-frame corpus, negligible against multi-hour
   training. See ``qnn.bc.train._compute_target_probs``.

6. **act_move (u8 packed: move + fire)** vs engine's
   ``(forwardmove/sidemove/upmove i16 × 3, buttons.bit0 fire)``. We
   collapse the three i16 axes to discrete classes {neg, none, pos}
   per axis (2 bits each, 6 bits total) and pack BUTTON_ATTACK as
   bit 6 of the same byte (bit 7 reserved for future jump). Why:
   the BC labeler emits coarse intent, not exact button-pressure,
   and the 2-bit-per-axis classifier matches the head's softmax
   output shape. Packing fire into the same byte uses bits we'd
   otherwise waste — net 7 bits in 1 byte, saving 1 byte/frame vs
   keeping fire as its own byte. The loader splits the byte back
   into (T, 3) move + (T,) fire arrays so the heads see the same
   tensor shapes they always did.

7. **act_look (f16 × 3)** vs engine's ``cl.viewangles[3] (float
   degrees)``. Native ``cmd_angles`` are i16 (QW 65536/360 quant);
   our act_look is the per-emit cur_forward · anchor_basis dot
   products in [-1, 1]. Why: matches the look head's anchor-relative
   target representation; aim precision argument keeps it at f16
   instead of i8 (~0.5° step at unit length would be visible in
   smoothness).

8. **Synthesized fields not in any Quake protocol idx**:
   ``attack_finished`` (QC self.attack_finished cooldown, our f16
   seconds), ``self_movement_id`` (composite ground/air/water-level
   we synthesize), entity ``eta``/``facing``/``team``/``score``/
   ``path``/``path_dist``, every spatial-atlas field, and every event scalar
   are all derived in
   the C-side oracle/spatial/event modules — no Quake-native source
   to deviate from.

Usercmd coverage (action side)
------------------------------
QW ``usercmd_t`` has 7 fields. What we carry, drop, or transform:

  msec               u8  — DROPPED. Frame duration; not relevant
                            once tick-rate is fixed at collect time.
  angles[3]          i16 × 3 (65536/360 quant) — TRANSFORMED to
                            act_look (f16 × 3 anchor-basis dots).
                            See deviation #7.
  forwardmove        i16 — TRANSFORMED to act_move fb-axis class
                            {neg, none, pos} (2 bits).
  sidemove           i16 — TRANSFORMED to act_move lr-axis class.
  upmove             i16 — TRANSFORMED to act_move ud-axis class.
  buttons.bit0       (BUTTON_ATTACK) — captured as raw evidence in
                            act_move bit 0. The supervised attack
                            outcome is stored separately as act_attack.
  buttons.bit1       (BUTTON_JUMP)   — captured as act_move bit 7;
                            swim-up remains act_move bit 6.
  impulse 1..8       — not stored as an independent switch command.
                            ``act_attack`` records the resolved impulse only
                            when an effective engine attack occurs.
  impulse 9, 10..    — DROPPED. Quake's `impulse 9` is a cheat;
                            `impulse 10`/`11` cycle weapons (not
                            used in BC corpus); 100+ are mod-specific.
                            Anything outside the 1..8 weapon range is
                            ignored at the action labeler boundary.

The raw attack bit and categorical attack field intentionally answer different
questions: usercmd press evidence versus accepted engine action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ── Contract identity ────────────────────────────────────────────
# The model↔engine contract is versioned on THREE INDEPENDENT axes. Two are
# stamped into every exported model (tools/export_onnx.py) and checked at load;
# the registry is src/docs/contracts/.
#
# WIRE_CONTRACT_ID — the WIRE contract = the SYNTAX of the exchange: the set /
#   order / dtypes / shapes of the ONNX I/O tensors. The LOAD-BREAKER — a wrong
#   wire is a parse error ("Invalid input name"). A model DECLARES a wire
#   contract; a bin-side CODEC implements one (spec vs impl, like PNG vs libpng
#   — so we version the contract, not the codec). BUMP on any I/O tensor added /
#   removed / retyped / reshaped. Monotonic across the full history; the load set:
#     wire.7  packed scalars (self_scalars[17] / entity_scalars_raw /
#             spatial_scalars), 13 ONNX inputs — token-spec ~v11 (v17/v22).
#     wire.8  native split, 43 inputs — engine/Python wire at 0.21.0; NO ONNX
#             graph was ever exported at 43 inputs (the native exporter postdates
#             look_delta), so this id is reconstructed.
#     wire.9  native split, 44 obs (= wire.8 + look_delta) + the IN-GRAPH MOVE
#             decode: the RECURRENT MOVE-DECODE STATE pair (move_state (B,11 f32)
#             + move_state_rng (B, i64)) is threaded as I/O, and the a24 stateful
#             MOVE decode (sticky gate / switch-back watermark / hazard /
#             stop-onset + the continuous-fire hold-tail) runs IN-GRAPH — so
#             `move` is the DECIDED 3-axis class (B,3 i64), NOT raw logits, and
#             the engine threads move_state frame-to-frame like hidden/next_hidden.
#             Current HEAD; what this exporter produces; the deployed full_4head.
#             RECLAIMED: an earlier wire.9 (native-split obs + engine-side per-axis
#             argmax of raw move_logits) and a wire.10 (the in-graph shape) were
#             distinguished during the move-decode migration, but no wire.10 was
#             ever finalized as a release — the in-graph migration stayed UNDER
#             wire.9 through active a24 development rather than bumping the number,
#             and the old engine-argmax wire.9 has no surviving artifact. Net:
#             wire.9 now means the in-graph-decode native contract; wire.10 is gone.
#     wire.11 = wire.9 + the IN-GRAPH ATTACK decode. The recurrent ATTACK-decode
#             state — the continuous-fire hold-tail (attack_state, B,1 f32) AND
#             attack's own xorshift rng (attack_rng, B,i64) — is threaded as I/O,
#             and the SAMPLED attack decode (Bernoulli on sigmoid((fire_logit+bias)
#             /temp) off attack_rng, + hold-tail) runs IN-GRAPH — so `attack` is the
#             DECIDED bit (B,1 i64), the `fire_logit` output is REMOVED, and the
#             engine ORs the bit into the Quake press byte and runs NO attack
#             sigmoid/threshold/hold-tail of its own. (move_state also dropped its
#             two dead trailing slots → (B,9).) With
#             this, all three actions (move/look/attack) are decided in-graph
#             → the engine is decode-agnostic and decode-regime changes no longer
#             touch the wire. wire.11 REPLACES wire.9 for the a24 gen (re-export);
#             wire.10 stays burned (never reuse). (jumps 9→11 deliberately.)
#   (wire.1–6 = older flat + packed lineages, pre-v11, out of the load set. The
#    faithful-load floor is wire.7: below it action_history / recall / cluster
#    are no longer emitted by the engine. See the registry.)
#
# SEMANTICS_CONTRACT_ID — the SEMANTICS contract = the MEANING behind those
#   tensors: normalization scales + vocab id mappings (see _semantics_sig in
#   tools/export_onnx.py). Same syntax, different meaning (vocab renumber, scale
#   change) is a SILENT failure — the bin turns it loud by refusing on a
#   semantics_sig mismatch. BUMP on any scale constant or vocab id mapping change
#   (tensors unchanged). One version across the load set (wire.7→.9 / v17→v24);
#   it DID move earlier (token-spec v11: entity vocab 42→44, weapon renumber).
#
# (The third axis, ARCH — model internals / weight layout — is checkpoint-side
#  only; the live bin ignores it. Tracked by qnn/utils/checkpoint_converter.py.)
# wire.13.x = the A27 pure-combat obs contract: the spatial-tokens-v2 depth
# atlas PLUS the actor/projectile-only combat entity stream (SIGHT/PROXIMITY
# current-frame, no item/mover rows) and the 9-way categorical attack head.
# It is a DISTINCT wire from a26's wire.12 (atlas + FULL entity stream,
# semantics.1) — the number moved off wire.12 because a26 already reclaimed
# that shape.
#
# The atlas GRID moved under a27 exactly as it did under a26, and both
# resolutions have deployed artifacts, so — same rule as wire.12.x — each
# gets its own id:
#
#   wire.13.1  a27 rc1: 72 yaw cells per band, unpacked one code per byte.
#              This is the deployed a27rc1a line.
#   wire.13.2  finalized frontier: 24 yaw cells per band, nibble-packed to
#              12 bytes per row. HEAD — what this exporter produces.
#
# BARE `wire.13` IS RETIRED AND MUST NOT BE RE-USED, for the same reason bare
# `wire.12` is: it was stamped on both families while the frontier was in
# flight, so it cannot select a codec without inspecting tensor shapes. The
# bin refuses it with a re-stamp hint (QNN_RETIRED_WIRES in
# src/engine/common/qnn_onnx.c). Same rule as wire.10.
#
# a24/a25's wire.11 (v1 raycast) and both a26 atlas families (wire.12.1 /
# wire.12.2) remain live codecs in the bin — one bin serves every artifact
# family; wire.13.2 is HEAD.
WIRE_CONTRACT_ID = "wire.13.2"
# The a27 rc1 atlas line (ATLAS_YAWS_LEGACY-wide, unpacked). Still loadable:
# its codec is registered and its artifacts are deployed.
WIRE_CONTRACT_ID_ATLAS_LEGACY = "wire.13.1"
SEMANTICS_CONTRACT_ID = "semantics.2"


# ── Engine resource caps ─────────────────────────────────────────
# Must equal the matching #defines in src/engine/common/qnn_vocab.h
# and src/engine/common/qnn_io.h. A parity test enforces this.

MAX_HEALTH         = 100         # normal cap (mega temporarily exceeds, then decays)
MAX_ARMOR_EFFECT   = 160         # max effective armor HP = red 200 × 0.8
MAX_SHELLS         = 100
MAX_NAILS          = 200
MAX_ROCKETS        = 100
MAX_CELLS          = 100
MAX_VELOCITY       = 2000        # sv_maxvelocity, per-axis clamp from sv_phys.c
TIME_SCALE         = 60.0        # canonical seconds normalization
DIST_SCALE         = 1000.0      # canonical world-unit normalization (entity rel/dist/path/half_extents, spatial near/mean dist)


# ── cl.items bit layout ──────────────────────────────────────────
# Native Quake/QW bit positions from vendor/quake/QW/progs/defs.qc.
# Wire stores cl.items as u32 to match svc_updatestatlong(STAT_ITEMS)
# exactly. The model's SelfDequantizer extracts the bits below into
# the categorical IDs the ObsEmbedding's embedding lookups consume.

IT_SHOTGUN          = 1 <<  0
IT_SUPER_SHOTGUN    = 1 <<  1
IT_NAILGUN          = 1 <<  2
IT_SUPER_NAILGUN    = 1 <<  3
IT_GRENADE_LAUNCHER = 1 <<  4
IT_ROCKET_LAUNCHER  = 1 <<  5
IT_LIGHTNING        = 1 <<  6
# bits 7..11: IT_EXTRA_WEAPON, ammo-type-owned flags — not consumed
IT_AXE              = 1 << 12
IT_ARMOR1           = 1 << 13   # green
IT_ARMOR2           = 1 << 14   # yellow
IT_ARMOR3           = 1 << 15   # red
# bit 16: IT_SUPERHEALTH transient at pickup — model uses health>100 instead
# bits 17..18: IT_KEY1, IT_KEY2 — level keys, not deathmatch
IT_INVISIBILITY     = 1 << 19   # ring
IT_INVULNERABILITY  = 1 << 20   # pent
IT_SUIT             = 1 << 21
IT_QUAD             = 1 << 22

ITEMS_WEAPON_MASK   = (IT_SHOTGUN | IT_SUPER_SHOTGUN | IT_NAILGUN
                       | IT_SUPER_NAILGUN | IT_GRENADE_LAUNCHER
                       | IT_ROCKET_LAUNCHER | IT_LIGHTNING | IT_AXE)
ITEMS_ARMOR_MASK    = IT_ARMOR1 | IT_ARMOR2 | IT_ARMOR3
ITEMS_POWERUP_MASK  = IT_INVISIBILITY | IT_INVULNERABILITY | IT_SUIT | IT_QUAD
ITEMS_MEANINGFUL    = ITEMS_WEAPON_MASK | ITEMS_ARMOR_MASK | ITEMS_POWERUP_MASK


# ── Movement category IDs ────────────────────────────────────────
# Mirrors the switch in qnn_self_common.c:115-121. Sent as u8.
MOVEMENT_GROUND     = 0
MOVEMENT_AIR        = 1
MOVEMENT_WATER_LOW  = 2  # waterlevel==1: feet wet
MOVEMENT_WATER_MID  = 3  # waterlevel==2: waist deep
MOVEMENT_WATER_HIGH = 4  # waterlevel==3: submerged


# ── Field descriptor ─────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    """One scalar or vector idx in the wire / cache layout.

    Attributes:
        name:       Wire field name. Also the npy filename suffix in
                    the sharded cache.
        dtype:      On-disk numpy dtype. Mirrors the C struct member.
        shape:      Per-frame shape (empty tuple = scalar).
        scale:      Divide the native int by this to get the
                    model-facing float in [-1, 1] or [0, 1]. ``None``
                    for non-scalar fields (categoricals, bitfields)
                    which the dequantizer handles via ``transform``.
        transform:  Non-linear handling on the model side:
                      - ``None``           — plain scalar, scaled
                      - ``"embedding"``    — categorical ID → embedding
                      - ``"bitfield"``     — extract bits into IDs
                      - ``"clamped"``      — int16 with hard clamp at ±scale
        source:     Provenance — where the value comes from in the
                    engine. Cross-reference with the QW protocol where
                    applicable.
    """
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    scale: Optional[float] = None
    transform: Optional[str] = None
    source: str = ""

    @property
    def bytes_per_frame(self) -> int:
        elems = 1
        for d in self.shape:
            elems *= d
        return elems * np.dtype(self.dtype).itemsize


# ── Self block ───────────────────────────────────────────────────
# Order matches the new qnn_io.h struct layout (to be defined). Total
# wire bytes = sum(f.bytes_per_frame for f in SELF_FIELDS).

SELF_FIELDS: Tuple[Field, ...] = (
    Field(
        name="health",
        dtype=np.uint8, shape=(),
        scale=float(MAX_HEALTH), transform=None,
        source="cl.stats[STAT_HEALTH] — svc_updatestat byte. "
               "Max ~250 briefly with megahealth.",
    ),
    Field(
        name="effective_armor",
        dtype=np.uint8, shape=(),
        scale=float(MAX_ARMOR_EFFECT), transform=None,
        source="round(STAT_ARMOR × armortype) — C computes the "
               "product, model divides. armortype ∈ {0, 0.3, 0.6, "
               "0.8}. See project_armor_encoding memory for the "
               "rationale on storing the product rather than raw + "
               "type. Type bits also live in self_items.",
    ),
    Field(
        name="ammo_shells",
        dtype=np.uint8, shape=(),
        scale=float(MAX_SHELLS), transform=None,
        source="cl.stats[STAT_SHELLS] — svc_updatestat byte.",
    ),
    Field(
        name="ammo_nails",
        dtype=np.uint8, shape=(),
        scale=float(MAX_NAILS), transform=None,
        source="cl.stats[STAT_NAILS] — svc_updatestat byte.",
    ),
    Field(
        name="ammo_rockets",
        dtype=np.uint8, shape=(),
        scale=float(MAX_ROCKETS), transform=None,
        source="cl.stats[STAT_ROCKETS] — svc_updatestat byte.",
    ),
    Field(
        name="ammo_cells",
        dtype=np.uint8, shape=(),
        scale=float(MAX_CELLS), transform=None,
        source="cl.stats[STAT_CELLS] — svc_updatestat byte.",
    ),
    Field(
        name="vel",
        dtype=np.int16, shape=(3,),
        scale=float(MAX_VELOCITY), transform="clamped",
        source="player_state.velocity rotated into view frame "
               "(C-side QNN_RelativeFrame). Each axis clamped to "
               "±MAX_VELOCITY by sv_phys.c:SV_CheckVelocity every "
               "physics tick. Protocol sends i16 per axis "
               "(MSG_WriteShort).",
    ),
    Field(
        name="attack_finished",
        dtype=np.float16, shape=(),
        scale=TIME_SCALE, transform=None,
        source="QC self.attack_finished cooldown remaining, "
               "converted to seconds (frames / fixed_tick_hz). "
               "NOT in the QW protocol — sourced from the QC VM "
               "evaluator. f16 chosen over u8 because the "
               "meaningful range spans 0.1s (recency-like) to "
               "~2s with sub-frame precision needs.",
    ),
    Field(
        name="self_movement_id",
        dtype=np.uint8, shape=(),
        scale=None, transform="embedding",
        source="Composite from player_state.waterlevel and "
               "pmove.onground: 0=ground, 1=air, 2/3/4=water "
               "low/mid/submerged. Fed to ObsEmbedding.movement_embed.",
    ),
    Field(
        name="self_items",
        dtype=np.int32, shape=(),
        scale=None, transform="bitfield",
        source="cl.items raw bitfield — mirrors Quake's `int items` "
               "(see vendor/quake/WinQuake/client.h:158) and the "
               "svc_updatestatlong(STAT_ITEMS) protocol idx. "
               "SelfDequantizer extracts: 7 weapon-owned flags "
               "(sg..lg) and axe, armor_type_id (from IT_ARMOR1/2/3 "
               "mask), and 4 powerup IDs (INVIS/PENT/SUIT/QUAD). "
               "Megahealth comes from health>100, NOT from "
               "IT_SUPERHEALTH (which is transient). i32 (not u32) "
               "to match Quake's signed declaration and dodge "
               "PyTorch's patchy u32 dispatch — bit-AND semantics "
               "are identical.",
    ),
    Field(
        name="view_pitch",
        dtype=np.int8, shape=(),
        scale=1.0, transform=None,
        source="snapshot.player_view_angles[0] (engine pitch in "
               "degrees) divided by 90 on the engine side, packed "
               "as i8. Engine clamps pitch to ±70° in "
               "CL_AdjustAngles so the i8 range is comfortable. "
               "Feeds the model's self.motion subtoken; replaces "
               "the pitch signal that used to live implicitly in "
               "the 9 spatial sectors' dir vectors.",
    ),
    Field(
        name="look_delta",
        dtype=np.float16, shape=(3,),
        scale=1.0, transform=None,
        source="look[t-1] - look[t-2]: the change between the two most "
               "recent realized look vectors (cur_forward · anchor_basis "
               "dots), computed in QNN_ComputeLookDelta. ~0 under steady "
               "rotation; ≈ angular acceleration, NOT velocity. f16 for the "
               "same near-zero precision rationale as act_look. WIRE-ONLY: "
               "dropped before the NPY cache (the BC preload re-derives it "
               "from the look column); feeds inference (bridge + ONNX).",
    ),
)


SELF_BLOCK_BYTES: int = sum(f.bytes_per_frame for f in SELF_FIELDS)
# Computed at import time; expect 27 bytes:
#   health 1 + effective_armor 1 + ammo×4 4 + vel 6 + attack_finished 2
#   + movement_id 1 + items 4 + view_pitch 1 = 20
#   + look_delta 6 = 27


# ── Spatial block ────────────────────────────────────────────────
# Final wire.12 center-ray depth atlas (spatial-tokens-v2 rev 15).
# ATLAS_ELEVS elevation bands × ATLAS_YAWS yaw cells; each cell is the
# exact first intersection of the cell's CENTER direction with the
# carved hull-1 face set (world + live-translated brush movers), 4-bit
# quantized: codes 0..14 index ATLAS_DEPTH_LEVELS, code 15 is the miss
# sentinel. Two adjacent yaw codes share one wire byte: low nibble = even
# cell, high nibble = odd cell. The model dequantizer expands the packed row.
#
# Grid convention: elevation-major. Band b centers at
# ATLAS_ELEV_DEG[b] (−75°..+75°, 15° steps); yaw cell y centers at
# 15°·y counter-clockwise from the player's view yaw (cell 0 = straight
# ahead), wrapping the full circle. Pitch is excluded from the frame —
# it is self-state (view_pitch). Per-band trace range is
# ATLAS_BAND_LIMIT[b] = min(1024, 128/|sin elev|): the horizontal
# contract is 1024 units, the vertical contract 128, matching v1.
# Validated by the reconstruction gate (wire.12.md) before adoption.

ATLAS_YAWS = 24
ATLAS_PACKED_BYTES = ATLAS_YAWS // 2
assert ATLAS_YAWS % 2 == 0
# Pre-wire.12 rc1-line atlas width (72 yaw cells, unpacked one code/byte, no
# nibble packing). Kept ONLY so the corpus caches and checkpoints of the
# still-deployed a26 rc1 line stay loadable/decodable under the current
# (narrower) atlas — see qnn.model.dequant.SpatialDequantizer._decode and
# qnn.utils.checkpoint_converter.migrate_legacy_spatial_atlas_dim. Never use
# this for new work.
ATLAS_YAWS_LEGACY = 72
ATLAS_ELEV_DEG: Tuple[int, ...] = (
    -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75,
)
ATLAS_DEPTH_LEVELS: Tuple[int, ...] = (
    0, 8, 16, 24, 36, 52, 72, 100, 136, 184, 248, 336, 456, 620, 1016,
)
ATLAS_MISS_CODE = 15
ATLAS_HORIZ_RANGE = 1024.0
ATLAS_VERT_RANGE = 128.0
ATLAS_BAND_LIMIT: Tuple[float, ...] = tuple(
    min(ATLAS_HORIZ_RANGE,
        ATLAS_VERT_RANGE / abs(math.sin(math.radians(e))))
    if e != 0 else ATLAS_HORIZ_RANGE
    for e in ATLAS_ELEV_DEG
)

SPATIAL_FIELDS: Tuple[Field, ...] = (
    Field(
        name="atlas",
        dtype=np.uint8, shape=(ATLAS_PACKED_BYTES,),
        scale=None, transform=None,
        source="One packed elevation row: 24 fifteen-degree yaw cells, two "
               "4-bit codes per byte (low nibble even, high nibble odd). "
               "Each code is the exact center-ray depth against the carved "
               "hull-1 face set quantized to ATLAS_DEPTH_LEVELS (0..14); "
               "15 is miss. The dequantizer expands the row, clamps decoded "
               "depth to the band limit, and derives hit from the miss code.",
    ),
)

SPATIAL_TOKEN_COUNT = len(ATLAS_ELEV_DEG)
SPATIAL_BLOCK_BYTES: int = SPATIAL_TOKEN_COUNT * sum(f.bytes_per_frame for f in SPATIAL_FIELDS)
# Computed at import time; expect 11 × 12 = 132 bytes.


# ── Entity block (variable-length per type) ──────────────────────
# The entity stream is variable-length and type-tagged on the wire
# (see qnn_io.c:127-241). Each per-type table below lists the scalars
# emitted for that token type, in the wire order. Categorical IDs
# (subject_id, modality_id, player_id) and the event arrays are
# common to all types and listed separately in ENTITY_COMMON_FIELDS.
#
# Cache layout: VARIABLE-LENGTH ON DISK, dynamic-pad at batch
# assembly. Each shard stores:
#   entity_indptr  (rows + 1,) u32 — byte offset of each row in the
#                                    packed data blob.
#   entity_data    (total_bytes,) u8 — packed variable rows. Per row:
#                                       count u8, then per-token
#                                       [type_tag u8, common_fields,
#                                       per-type scalars] packed
#                                       densely (no padding).
#
# The dataloader reads each batch row's variable-length data and
# pads to max-token-count-in-batch at collation time. This is the
# only padding that exists — there is no on-disk padding to a global
# 16-idx maximum. PyTorch's nn.MultiheadAttention requires
# rectangular batch tensors; the batch-time pad is forced by that.
#
# Savings vs the legacy fixed (rows, 16, 19) f16 layout (~608 B per
# frame for scalars alone): typical DM frames carry 4-12 tokens
# averaging 22 B native; padding to batch-max ~12 in a typical batch
# gives ~265 B/frame at the model boundary, vs 608 B today. On disk,
# the variable-length blob avoids the padding entirely.

ENTITY_COMMON_FIELDS: Tuple[Field, ...] = (
    Field(
        name="type",
        dtype=np.int8, shape=(),
        scale=None, transform=None,
        source="A27 combat token type tag (TOKEN_PROJECTILE=0, "
               "TOKEN_ACTOR=1). -1 for empty idx in dense cache.",
    ),
    Field(
        name="subject_id",
        dtype=np.uint8, shape=(),
        scale=None, transform="embedding",
        source="ENTITY_IDS vocab entry. Categorical, fed to "
               "entity_embed.",
    ),
    Field(
        name="modality_id",
        dtype=np.uint8, shape=(),
        scale=None, transform="embedding",
        source="A27 combat modality — SIGHT or PROXIMITY only. "
               "Categorical, fed to modality_embed.",
    ),
    Field(
        name="player_id",
        dtype=np.uint8, shape=(),
        scale=None, transform="embedding",
        source="ACTOR only — entity idx index for player identity. "
               "0 for non-actor tokens. Fed to player_embed.",
    ),
    Field(
        name="event_count",
        dtype=np.uint8, shape=(),
        scale=None, transform=None,
        source="Number of valid (action_id, source_id) pairs in the "
               "events arrays for this token. 0..MAX_ENTITY_EVENTS (4).",
    ),
    Field(
        name="event_actions",
        dtype=np.uint8, shape=(4,),
        scale=None, transform="embedding",
        source="ACTION_IDS — FIRE/JUMP/PAIN/etc. that this entity "
               "emitted recently. Fed to action_embed. Indices past "
               "event_count are zero.",
    ),
    Field(
        name="event_sources",
        dtype=np.uint8, shape=(4,),
        scale=None, transform="embedding",
        source="ENTITY_IDS — subject of each event (e.g., who fired). "
               "Fed to entity_embed. Aligned with event_actions.",
    ),
)


# Per-type scalar tables. Order matches the C wire packing in
# qnn_io.c:142-241 and the ONNX emit funcs in qnn_onnx.c:194-326.
# scale=DIST_SCALE applies to distance fields; the C-side normalization
# moves to the model in the new wire format (raw i16 native units on
# disk, model divides at dequant time).

# Reusable Field templates so the same scalar definition isn't
# duplicated per type below.
_F_HALF_EXTENTS = Field(
    name="half_extents",
    dtype=np.uint8, shape=(3,),
    scale=DIST_SCALE, transform=None,
    source="Entity bbox half-sizes in Quake units. Native raw, model "
           "divides by DIST_SCALE. Actor bounds are well below the u8 "
           "ceiling; values saturate at 255 Quake units.",
)
_F_REL = Field(
    name="rel",
    dtype=np.int16, shape=(3,),
    scale=DIST_SCALE, transform=None,
    source="Position relative to player, view-frame rotated. Quake "
           "units. Protocol MSG_WriteCoord is i16 (1/8 unit); we use "
           "1-unit precision over ±32k range.",
)
# Note: no _F_DIST. The C side computed `dist = |rel|` and stored
# both for convenience. On disk we drop the redundant magnitude and
# have the dequantizer compute it via `torch.linalg.norm(rel, dim=-1)`.
# Same information, 2 B/idx × 16 indices = 32 B/frame saved, zero
# precision loss.
_F_VEL = Field(
    name="vel",
    dtype=np.int16, shape=(3,),
    scale=float(MAX_VELOCITY), transform="clamped",
    source="Entity velocity rotated into player view frame, clamped "
           "to ±sv_maxvelocity. i16 native.",
)
_F_PATH = Field(
    name="path",
    dtype=np.int16, shape=(3,),
    scale=DIST_SCALE, transform=None,
    source="Navmesh waypoint direction in view frame. Same units as "
           "rel; falls back to rel when no path is computed.",
)
_F_PATH_DIST = Field(
    name="path_dist",
    dtype=np.uint16, shape=(),
    scale=DIST_SCALE, transform=None,
    source="Total navmesh path length to entity. u16 native.",
)
_F_ETA = Field(
    name="eta",
    dtype=np.float16, shape=(),
    scale=TIME_SCALE, transform=None,
    source="Estimated time-to-arrival in seconds. Same f16 rationale "
           "as attack_finished (range / precision tradeoff).",
)
_F_RECENCY = Field(
    name="recency",
    dtype=np.float16, shape=(),
    scale=TIME_SCALE, transform=None,
    source="Time since last observed, seconds. f16. Full-stream (a26) "
           "only — the A27 combat stream dropped it.",
)
PROJECTILE_FIELDS: Tuple[Field, ...] = (
    _F_REL, _F_VEL,
)

ACTOR_FIELDS: Tuple[Field, ...] = (
    _F_HALF_EXTENTS,
    _F_REL, _F_VEL, _F_PATH, _F_PATH_DIST, _F_ETA,
    Field(
        name="facing",
        dtype=np.uint8, shape=(),
        scale=255.0, transform=None,
        source="Pre-normalized [0, 1] from `(1 - dot(their_forward, "
               "to_us)) * 0.5` in qnn_oracle.c:503. 0 = facing toward "
               "us, 1 = facing away. u8.",
    ),
    Field(
        name="team",
        dtype=np.uint8, shape=(),
        scale=1.0, transform=None,
        source="QNN_IsSameTeam result, {0, 1}. Could be a bit but the "
               "1-byte cost dominates anyway.",
    ),
    Field(
        name="score",
        dtype=np.uint8, shape=(),
        scale=255.0, transform=None,
        source="frags / max_frags in [0, 1]. u8 with 1/255 precision.",
    ),
)

ENTITY_FIELDS: dict[str, Tuple[Field, ...]] = {
    "projectile": PROJECTILE_FIELDS,
    "actor":      ACTOR_FIELDS,
}

# Per-type scalar block byte totals (excluding common IDs / events).
ENTITY_SCALAR_BYTES: dict[str, int] = {
    name: sum(f.bytes_per_frame for f in fields)
    for name, fields in ENTITY_FIELDS.items()
}
# Expected (the per-type difference is exactly what makes the wire
# variable-length on the C side):
#   projectile: 6 + 6 = 12
#   actor:      3 + 6 + 6 + 6 + 2 + 2 + 1 + 1 + 1 = 28


# ── Full-stream (a26) per-type scalar tables ─────────────────────
# The a26-line stream, verbatim from the pre-split codec layer
# (215eb608): recency trails every type, and item/mover are live token
# types (the A27 combat stream deleted them). Kept ALONGSIDE the combat
# tables so a26-line checkpoints load and forward on this line —
# selected per checkpoint via the model's ``entity_stream``. The combat
# tables above stay the canonical ENTITY_FIELDS objects; nothing here
# feeds the A27 wire/cache math.

FULL_PROJECTILE_FIELDS: Tuple[Field, ...] = (
    _F_REL, _F_VEL, _F_RECENCY,
)

FULL_ACTOR_FIELDS: Tuple[Field, ...] = ACTOR_FIELDS + (_F_RECENCY,)

ITEM_FIELDS: Tuple[Field, ...] = (
    _F_HALF_EXTENTS,
    _F_REL, _F_PATH, _F_PATH_DIST, _F_ETA,
    Field(
        name="amount",
        dtype=np.uint8, shape=(),
        scale=None, transform="item_amount",
        source="Raw engine pickup amount from qnn_item_defs (e.g. 100 "
               "for green armor, 25 for health box, 200 for nail box). "
               "u8 saturating; max raw value in the def table is 200 "
               "(nails large pack), fits comfortably. Model-side "
               "normalization uses the per-subject lookup in "
               "ITEM_AMOUNT_MULT / ITEM_AMOUNT_CONST below — keeps the "
               "engine ignorant of MAX_HEALTH / armor_type / etc.",
    ),
    Field(
        name="regen",
        dtype=np.float16, shape=(),
        scale=TIME_SCALE, transform=None,
        source="Item respawn timer remaining, seconds. f16.",
    ),
    _F_RECENCY,
)

MOVER_FIELDS: Tuple[Field, ...] = (
    _F_HALF_EXTENTS,
    _F_REL, _F_PATH, _F_PATH_DIST, _F_ETA,
    Field(
        name="state",
        dtype=np.uint8, shape=(),
        scale=255.0, transform=None,
        source="Pre-normalized [0, 1] mover state (door open fraction, "
               "platform position, etc.). Already produced as float in "
               "qnn_event.c; u8 quantizes.",
    ),
    _F_RECENCY,
)

FULL_ENTITY_FIELDS: dict[str, Tuple[Field, ...]] = {
    "projectile": FULL_PROJECTILE_FIELDS,
    "actor":      FULL_ACTOR_FIELDS,
    "item":       ITEM_FIELDS,
    "mover":      MOVER_FIELDS,
}

# Stream-keyed views. "combat" IS the canonical dict above (same object;
# existing consumers — contracts fingerprinting, wire math — unchanged).
ENTITY_FIELDS_BY_STREAM: dict[str, dict[str, Tuple[Field, ...]]] = {
    "combat": ENTITY_FIELDS,
    "full":   FULL_ENTITY_FIELDS,
}
ENTITY_SCALAR_BYTES_BY_STREAM: dict[str, dict[str, int]] = {
    "combat": ENTITY_SCALAR_BYTES,
    "full": {
        name: sum(f.bytes_per_frame for f in fields)
        for name, fields in FULL_ENTITY_FIELDS.items()
    },
}
# Expected full-stream totals: projectile 14, actor 30, item 24, mover 22.


# ── Item amount normalization table ──────────────────────────────
# Item/mover tokens are NOT part of the wire.13 combat obs stream (the model
# reads only actor/projectile — see model/transformer.py), but they remain
# live engine entities: the C store/reward path populates item `amount`, and
# the wire.11/wire.12 codecs (a24/a25/a26 models, supported in the same bin)
# emit the full entity stream including items. This per-subject affine
# (``amount = raw × mult + const``) is the value-semantics of that item
# `amount` field — it mirrors qnn_onnx.c:w7_item_amount / the C
# QNN_NormalizeItemAmount table. Kept here (and fingerprinted by
# semantics_sig) so a silent drift is caught; subjects not in the table get
# (0, 0) → amount = 0.
from qnn.vocab import ENTITY_IDS, ENTITY_VOCAB_SIZE  # noqa: E402

ITEM_AMOUNT_MULT  = np.zeros(ENTITY_VOCAB_SIZE, dtype=np.float32)
ITEM_AMOUNT_CONST = np.zeros(ENTITY_VOCAB_SIZE, dtype=np.float32)

# Raw-dependent: ammo / health / armor.
ITEM_AMOUNT_MULT[ENTITY_IDS["HEALTH"]]        = 1.0 / MAX_HEALTH
ITEM_AMOUNT_MULT[ENTITY_IDS["MEGAHEALTH"]]    = 1.0 / MAX_HEALTH
ITEM_AMOUNT_MULT[ENTITY_IDS["ARMOR_GREEN"]]   = 0.3 / MAX_ARMOR_EFFECT
ITEM_AMOUNT_MULT[ENTITY_IDS["ARMOR_YELLOW"]]  = 0.6 / MAX_ARMOR_EFFECT
ITEM_AMOUNT_MULT[ENTITY_IDS["ARMOR_RED"]]     = 0.8 / MAX_ARMOR_EFFECT
ITEM_AMOUNT_MULT[ENTITY_IDS["SHELLS"]]        = 1.0 / MAX_SHELLS
ITEM_AMOUNT_MULT[ENTITY_IDS["NAILS"]]         = 1.0 / MAX_NAILS
ITEM_AMOUNT_MULT[ENTITY_IDS["ROCKETS"]]       = 1.0 / MAX_ROCKETS
ITEM_AMOUNT_MULT[ENTITY_IDS["CELLS"]]         = 1.0 / MAX_CELLS

# Constant per subject: powerup pickups (just "you got one") and weapon
# pickups (the ammo bonus the weapon comes with).
for _s in ("QUAD", "PENT", "RING", "SUIT"):
    ITEM_AMOUNT_CONST[ENTITY_IDS[_s]] = 1.0
ITEM_AMOUNT_CONST[ENTITY_IDS["SHOTGUN"]]           =  5.0 / MAX_SHELLS
ITEM_AMOUNT_CONST[ENTITY_IDS["SUPER_SHOTGUN"]]     =  5.0 / MAX_SHELLS
ITEM_AMOUNT_CONST[ENTITY_IDS["NAILGUN"]]           = 30.0 / MAX_NAILS
ITEM_AMOUNT_CONST[ENTITY_IDS["SUPER_NAILGUN"]]     = 30.0 / MAX_NAILS
ITEM_AMOUNT_CONST[ENTITY_IDS["GRENADE_LAUNCHER"]]  =  5.0 / MAX_ROCKETS
ITEM_AMOUNT_CONST[ENTITY_IDS["ROCKET_LAUNCHER"]]   =  5.0 / MAX_ROCKETS
ITEM_AMOUNT_CONST[ENTITY_IDS["THUNDERBOLT"]]       = 15.0 / MAX_CELLS
del _s


# ── Action block ─────────────────────────────────────────────────
# What the BC trainer / PPO rollout consume as ground-truth actions
# per frame. These are not part of the obs wire; they live next to
# obs in the sharded cache (act_*.npy files). Included here so the
# normalization spec is in one place.
#
# act_target_probs is the dominant cost: today 17 × f32 = 68 B/frame
# and 94% one-hot empirically. The native encoding stores top-1 plus
# an optional top-2 (idx, weight) entry, since 99%+ of rows fit in
# the (idx, idx2, w2) triple.

ACTION_FIELDS: Tuple[Field, ...] = (
    Field(
        name="act_move",
        dtype=np.uint8, shape=(),
        scale=None, transform="bitfield",
        source="Packed action byte — bit layout mirrors input_mask: "
               "bit 0 = attack press, bits 1-2 = fb neg/pos, bits 3-4 = "
               "lr neg/pos, bits 5-6 = ud neg/pos (bit 6 unified: swim-up "
               "OR jump), bit 7 = explicit jump press (diagnostic). See "
               "qnn.bc.collect:_compact_action_arrays. Loader in "
               "qnn.bc.train splits back into (T, 3) move axis classes "
               "and (T,) attack / jump streams.",
    ),
    Field(
        name="act_attack",
        dtype=np.uint8, shape=(),
        scale=None, transform="embedding",
        source="Categorical effective attack 0..8 (0 = no attack, 1 = axe, "
               "..., 8 = thunderbolt). Impulse-indexed (NOT ENTITY_IDS); "
               "there is no parallel fire or action-side weapon label.",
    ),
    Field(
        name="act_look",
        dtype=np.float16, shape=(3,),
        scale=1.0, transform=None,
        source="Per-emit view delta — forward dotted with the previous "
               "anchor basis (3 floats in [-1, 1]). Kept f16 for fine "
               "aim precision; i8 = 0.008 step would be ~0.5° at typical "
               "view distances which is noticeable in look smoothness.",
    ),
    Field(
        name="act_op_input",
        dtype=np.uint8, shape=(),
        scale=None, transform="bitfield",
        source="Strict per-axis OPERATIVENESS mask (QNN_PackOpInput): "
               "bit i = 1 iff the player pressed axis i AND the engine "
               "acted on it this tick. bit0=fb, bit1=lr, bit2=ud, "
               "bit3=fire, bit4=impulse. Semantically DISTINCT from "
               "input_mask (pure feasibility — would the engine accept a "
               "press): op_input AND's the press with the per-axis op "
               "predicate. Wire offset 3 (formerly the _pad byte), so the "
               "addition is byte-additive — the action struct stays 16 B. "
               "0 on paths that don't compute it (NQ, MVD inference). "
               "Selected by the move-labeler subset collect; the BC "
               "full collect emits it additively without touching any "
               "pre-existing field.",
    ),
    # act_target_probs was a sparse (T, 3) u8 encoding of the labeler's
    # (T, 17) f32 distribution. Now recomputed at training start from
    # obs+actions by qnn.bc.train._compute_target_probs instead of being
    # baked into the cache. Decouples LabelerConfig from the cache
    # fingerprint (tune labeler without recollect) and eliminates the
    # sparse-truncation calibration drift on multi-hot rows.
)

ACTION_BLOCK_BYTES: int = sum(f.bytes_per_frame for f in ACTION_FIELDS)
# Expected: 1 (move+raw attack press) + 1 (attack category) + 6 (look)
# + 1 (op_input) = 9 bytes
# (was 11 with sparse target_probs; 12 before fire was packed into move;
# ~86 B in the original float32 layout: 1+1+1+6+68).  Note: this is the
# CACHE block; the on-wire action struct (qnn.wire.ACTION_SIZE) stays 16 B
# (op_input reuses the former _pad slot — additive, byte-identical).


# ── Total wire/cache budget ──────────────────────────────────────
# Per-frame totals. Entity is variable-length on disk (see layout
# note above); the padded figure below is only a worst-case sanity
# bound, not the actual disk footprint.

_ENTITY_IDX_BYTES_PADDED = sum(f.bytes_per_frame for f in ENTITY_COMMON_FIELDS) + max(
    ENTITY_SCALAR_BYTES.values()
)
ENTITY_BLOCK_BYTES_PADDED: int = 16 * _ENTITY_IDX_BYTES_PADDED  # worst-case bound

# Per-frame budget vs today's ~1169 B f16 cache:
#   self    20 B  (was 42)
#   spatial 132 B (final wire.12 atlas; two 4-bit codes per byte)
#   entity  variable, typically ~265 B at batch-max-pad (was ~608)
#                                                       worst-case bound = ENTITY_BLOCK_BYTES_PADDED
#   actions 12 B  (was 86)
#   typical total ~430 B — the atlas restores the v1 spatial budget while
#   retaining the geometry gate (see wire.12.md).
