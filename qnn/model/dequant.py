"""Native-width → model-facing tensor adapters.

Three dequantizer modules convert the engine-native dicts produced
by the new wire format (see ``qnn.engine_norm``) into the float /
int tensors the existing Tokenizer + heads consume:

- ``SelfDequantizer``    — health, armor, ammo, velocity, items
                            bitfield, attack cooldown.
- ``SpatialDequantizer`` — 9 sector tokens with dir, distances,
                            clearance / openness / fractions.
- ``EntityDequantizer``  — variable-length entity tokens (per-type
                            scalars: actor / projectile / item / mover).
                            Added in a follow-up commit.

Each owns the per-field normalization (``/QNN_MAX_HEALTH``,
``/QNN_VELOCITY_SCALE``, ``/QNN_TIME_SCALE``, ``/QNN_DIST_SCALE``,
``/127`` for i8 unit vectors, ``/255`` for u8 [0, 1] fractions) and
any bit / index demuxing. Output dict keys feed the Tokenizer's
existing per-type projections and embedding lookups, so trained
checkpoints load and run unchanged once the dataloader produces
native-format obs dicts.

Native obs is the ONLY input contract — there is no legacy
passthrough. Reading older f16 caches happens in a dedicated
loader path (qnn.bc.loader) that produces native dicts from the
legacy bytes. Removing the passthrough keeps the model surface
single-format and prevents drift between two code paths.

Lives in qnn.model (not qnn.engine_norm) so the pure-data table
stays free of a torch dependency, mirroring the qnn.vocab /
qnn.model.transformer split.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from qnn import engine_norm as en
from qnn.vocab import (
    ENTITY_IDS,
    TOKEN_PROJECTILE, TOKEN_ACTOR, TOKEN_ITEM, TOKEN_MOVER,
    ACTOR_SCALAR_DIM,
)


_ARMOR_SUBJECT_GREEN  = ENTITY_IDS["ARMOR_GREEN"]
_ARMOR_SUBJECT_YELLOW = ENTITY_IDS["ARMOR_YELLOW"]
_ARMOR_SUBJECT_RED    = ENTITY_IDS["ARMOR_RED"]
_SUBJECT_QUAD         = ENTITY_IDS["QUAD"]
_SUBJECT_PENT         = ENTITY_IDS["PENT"]
_SUBJECT_RING         = ENTITY_IDS["RING"]
_SUBJECT_SUIT         = ENTITY_IDS["SUIT"]
_SUBJECT_MEGAHEALTH   = ENTITY_IDS["MEGAHEALTH"]

# Self-scalars vector slot layout. Mirrors the legacy float-32 self
# block consumed by Tokenizer.self_proj (nn.Linear(17, d_model)). The
# trained checkpoints have weights indexed by these positions; do not
# reorder without an architectural retrain.
_SELF_SCALAR_DIM = 17
_SLOT_HEALTH          = 0
_SLOT_ARMOR           = 1
_SLOT_WEAPON_SG       = 2
_SLOT_WEAPON_SSG      = 3
_SLOT_WEAPON_NG       = 4
_SLOT_WEAPON_SNG      = 5
_SLOT_WEAPON_GL       = 6
_SLOT_WEAPON_RL       = 7
_SLOT_WEAPON_LG       = 8
_SLOT_AMMO_SHELLS     = 9
_SLOT_AMMO_NAILS      = 10
_SLOT_AMMO_ROCKETS    = 11
_SLOT_AMMO_CELLS      = 12
_SLOT_VEL_X           = 13
_SLOT_VEL_Y           = 14
_SLOT_VEL_Z           = 15
_SLOT_ATTACK_FINISHED = 16


class SelfDequantizer(nn.Module):
    """Engine-native self block → Tokenizer-ready float / int tensors."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, obs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert a native obs dict to the Tokenizer-expected obs dict.

        Idempotent: if obs already carries the dequantized
        ``self_scalars`` key (e.g. the GPU-resident preload step in
        qnn.bc.supervised_loop.preload_episodes_to_gpu ran the dequant
        once at startup), pass through unchanged.
        """
        if "self_scalars" in obs:
            return dict(obs)
        out: dict[str, torch.Tensor] = dict(obs)

        # All native self fields share batch dim 0.
        health      = obs["health"]            # (B,) u8
        eff_armor   = obs["effective_armor"]   # (B,) u8
        ammo_sh     = obs["ammo_shells"]       # (B,) u8
        ammo_na     = obs["ammo_nails"]        # (B,) u8
        ammo_rk     = obs["ammo_rockets"]      # (B,) u8
        ammo_ce     = obs["ammo_cells"]        # (B,) u8
        vel         = obs["vel"]               # (B, 3) i16
        af          = obs["attack_finished"]   # (B,) f16, seconds
        items       = obs["self_items"]        # (B,) u32

        device = health.device
        batch = health.shape[0]

        # Scalar normalizations. Promote to float32 for the matmul-bound
        # downstream; division is a free cast on modern accelerators.
        scalars = torch.zeros(batch, _SELF_SCALAR_DIM, device=device, dtype=torch.float32)
        scalars[:, _SLOT_HEALTH]       = health.to(torch.float32)    / en.MAX_HEALTH
        scalars[:, _SLOT_ARMOR]        = eff_armor.to(torch.float32) / en.MAX_ARMOR_EFFECT
        scalars[:, _SLOT_AMMO_SHELLS]  = ammo_sh.to(torch.float32)   / en.MAX_SHELLS
        scalars[:, _SLOT_AMMO_NAILS]   = ammo_na.to(torch.float32)   / en.MAX_NAILS
        scalars[:, _SLOT_AMMO_ROCKETS] = ammo_rk.to(torch.float32)   / en.MAX_ROCKETS
        scalars[:, _SLOT_AMMO_CELLS]   = ammo_ce.to(torch.float32)   / en.MAX_CELLS
        scalars[:, _SLOT_VEL_X:_SLOT_VEL_Z + 1] = (
            vel.to(torch.float32) / en.MAX_VELOCITY
        )
        scalars[:, _SLOT_ATTACK_FINISHED] = (
            af.to(torch.float32) / en.TIME_SCALE
        )

        # cl.items bit-extracted weapon flags. Boolean → float {0.0,
        # 1.0}, sitting in slots 2..8 so they line up byte-for-byte
        # with the legacy weapon_sg..weapon_lg floats.
        items_i64 = items.to(torch.int64)
        scalars[:, _SLOT_WEAPON_SG]  = ((items_i64 & en.IT_SHOTGUN)          != 0).to(torch.float32)
        scalars[:, _SLOT_WEAPON_SSG] = ((items_i64 & en.IT_SUPER_SHOTGUN)    != 0).to(torch.float32)
        scalars[:, _SLOT_WEAPON_NG]  = ((items_i64 & en.IT_NAILGUN)          != 0).to(torch.float32)
        scalars[:, _SLOT_WEAPON_SNG] = ((items_i64 & en.IT_SUPER_NAILGUN)    != 0).to(torch.float32)
        scalars[:, _SLOT_WEAPON_GL]  = ((items_i64 & en.IT_GRENADE_LAUNCHER) != 0).to(torch.float32)
        scalars[:, _SLOT_WEAPON_RL]  = ((items_i64 & en.IT_ROCKET_LAUNCHER)  != 0).to(torch.float32)
        scalars[:, _SLOT_WEAPON_LG]  = ((items_i64 & en.IT_LIGHTNING)        != 0).to(torch.float32)

        # Armor type ID: 0 if no armor bit set, else GREEN/YELLOW/RED.
        # Higher tier wins if multiple bits are set (defensive — engine
        # only ever sets one).
        armor_type = torch.zeros(batch, dtype=torch.int64, device=device)
        armor_type = torch.where(
            (items_i64 & en.IT_ARMOR1) != 0,
            torch.full_like(armor_type, _ARMOR_SUBJECT_GREEN),
            armor_type,
        )
        armor_type = torch.where(
            (items_i64 & en.IT_ARMOR2) != 0,
            torch.full_like(armor_type, _ARMOR_SUBJECT_YELLOW),
            armor_type,
        )
        armor_type = torch.where(
            (items_i64 & en.IT_ARMOR3) != 0,
            torch.full_like(armor_type, _ARMOR_SUBJECT_RED),
            armor_type,
        )

        # Powerup IDs: pack present powerups (incl. megahealth via
        # health>100) into the leading slots of a (B, 5) tensor. Empty
        # trailing slots stay 0 (NONE), which the Tokenizer masks with
        # `pmask = (pids > 0)`. Order matches the legacy emitter in
        # qnn_self_common.c:124-133 (QUAD, PENT, RING, SUIT, MEGAHEALTH).
        flags = torch.stack([
            (items_i64 & en.IT_QUAD)            != 0,
            (items_i64 & en.IT_INVULNERABILITY) != 0,
            (items_i64 & en.IT_INVISIBILITY)    != 0,
            (items_i64 & en.IT_SUIT)            != 0,
            health.to(torch.int64) > 100,                # megahealth
        ], dim=1)  # (B, 5) bool
        subject_ids = torch.tensor(
            [_SUBJECT_QUAD, _SUBJECT_PENT, _SUBJECT_RING, _SUBJECT_SUIT, _SUBJECT_MEGAHEALTH],
            dtype=torch.int64, device=device,
        )  # (5,)
        values = subject_ids.unsqueeze(0).expand(batch, -1)        # (B, 5)

        # Per-row prefix sum over flags gives each present powerup a
        # unique compact slot 0..4 in legacy order. Absent powerups get
        # routed to a sentinel slot (5) which we then discard — this
        # avoids the non-deterministic-write hazard scatter has when
        # multiple sources alias the same destination.
        slot_idx = flags.to(torch.int64).cumsum(dim=1) - 1         # (B, 5)
        write_idx = torch.where(
            flags, slot_idx, torch.full_like(slot_idx, 5)
        )                                                          # (B, 5)
        scratch = torch.zeros(batch, 6, dtype=torch.int64, device=device)
        scratch.scatter_(1, write_idx, values)
        powerup_ids = scratch[:, :5].contiguous()

        out["self_scalars"] = scalars
        out["self_weapon_id"]     = obs["self_weapon_id"].to(torch.int64).unsqueeze(-1)
        out["self_armor_type_id"] = armor_type.unsqueeze(-1)
        out["self_movement_id"]   = obs["self_movement_id"].to(torch.int64).unsqueeze(-1)
        out["self_powerup_ids"]   = powerup_ids
        return out


# ── SpatialDequantizer ───────────────────────────────────────────

# spatial_scalars slot layout — mirrors qnn_onnx.c:374-386. The
# Tokenizer's spatial_proj is nn.Linear(13, d_model); trained
# checkpoints have weights indexed by these positions, so reordering
# requires retraining.
_SPATIAL_SCALAR_DIM   = 13
_SP_DIR_X             = 0
_SP_DIR_Y             = 1
_SP_DIR_Z             = 2
_SP_NEAREST_DIST      = 3
_SP_MEAN_DIST         = 4
_SP_OPENNESS          = 5
_SP_CLEARANCE         = 6
_SP_TRAVERSABLE       = 7
_SP_DROPOFF           = 8
_SP_SOLID_FRAC        = 9
_SP_WATER_FRAC        = 10
_SP_SLIME_FRAC        = 11
_SP_LAVA_FRAC         = 12


class SpatialDequantizer(nn.Module):
    """Engine-native spatial block → Tokenizer-ready ``spatial_scalars``.

    Input: per-field native-typed tensors (per qnn.engine_norm.SPATIAL_FIELDS).
    Output: ``spatial_scalars`` (B, 9, 13) float32 in the layout the
    Tokenizer's ``spatial_proj`` consumes.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, obs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Idempotent: if obs already has the dequantized
        # ``spatial_scalars`` (preload ran the dequant), pass through.
        if "spatial_scalars" in obs:
            return dict(obs)
        out: dict[str, torch.Tensor] = dict(obs)

        d         = obs["spatial_dir"]            # (B, 9, 3) i8
        nearest   = obs["spatial_nearest_dist"]   # (B, 9) u16
        mean      = obs["spatial_mean_dist"]      # (B, 9) u16
        openness  = obs["spatial_openness"]       # (B, 9) u8
        clearance = obs["spatial_clearance"]      # (B, 9) u8
        trav      = obs["spatial_traversable"]    # (B, 9) u8
        dropoff   = obs["spatial_dropoff"]        # (B, 9) u8
        solid     = obs["spatial_solid_frac"]     # (B, 9) u8
        water     = obs["spatial_water_frac"]     # (B, 9) u8
        slime     = obs["spatial_slime_frac"]     # (B, 9) u8
        lava      = obs["spatial_lava_frac"]      # (B, 9) u8

        batch = d.shape[0]
        scalars = torch.zeros(
            batch, en.SPATIAL_TOKEN_COUNT, _SPATIAL_SCALAR_DIM,
            device=d.device, dtype=torch.float32,
        )
        # i8 unit vector → float in [-1, 1] via /127.
        scalars[:, :, _SP_DIR_X:_SP_DIR_Z + 1] = d.to(torch.float32) / 127.0
        # Raw Quake unit distances → [0, 1]-ish via /DIST_SCALE.
        scalars[:, :, _SP_NEAREST_DIST] = nearest.to(torch.float32) / en.DIST_SCALE
        scalars[:, :, _SP_MEAN_DIST]    = mean.to(torch.float32)    / en.DIST_SCALE
        # u8 [0, 1] floats — re-divide by 255 to recover the
        # already-clamped [0, 1] float values qnn_spatial.c emitted.
        scalars[:, :, _SP_OPENNESS]    = openness.to(torch.float32)  / 255.0
        scalars[:, :, _SP_CLEARANCE]   = clearance.to(torch.float32) / 255.0
        scalars[:, :, _SP_TRAVERSABLE] = trav.to(torch.float32)      / 255.0
        scalars[:, :, _SP_DROPOFF]     = dropoff.to(torch.float32)   / 255.0
        scalars[:, :, _SP_SOLID_FRAC]  = solid.to(torch.float32)     / 255.0
        scalars[:, :, _SP_WATER_FRAC]  = water.to(torch.float32)     / 255.0
        scalars[:, :, _SP_SLIME_FRAC]  = slime.to(torch.float32)     / 255.0
        scalars[:, :, _SP_LAVA_FRAC]   = lava.to(torch.float32)      / 255.0

        out["spatial_scalars"] = scalars
        return out


# ActionDequantizer was deleted along with the sparse act_target_dist
# encoding. The target distribution is now recomputed at training start
# from obs+actions by qnn.bc.train._compute_target_dist and arrives as
# dense (T, TARGET_DIST_CLASSES) float32 — heads consume it directly,
# no model-side expansion needed.


# ── EntityDequantizer ────────────────────────────────────────────

# Per-type scalar slot layouts in the legacy (B, N, ACTOR_SCALAR_DIM=19)
# entity_scalars_raw tensor that the Tokenizer's per-type Linear
# projections consume. These mirror the C side qnn_onnx.c:194-318
# emit_{actor,projectile,item,mover} functions exactly so trained
# checkpoints stay valid.
#
# Slot indices the model expects (post-dist-recompute):
#   ACTOR:      [hx,hy,hz, rx,ry,rz, dist, vx,vy,vz, px,py,pz, pd, eta, fac,team,score, rec]
#   PROJECTILE: [rx,ry,rz, dist, vx,vy,vz, rec, 0..0]                        (8 used / 19)
#   ITEM:       [hx,hy,hz, rx,ry,rz, dist, px,py,pz, pd, eta, amt, regen, rec, 0..0]  (15)
#   MOVER:      [hx,hy,hz, rx,ry,rz, dist, px,py,pz, pd, eta, state, rec, 0..0]       (14)


_ITEM_AMOUNT_MULT  = torch.tensor(en.ITEM_AMOUNT_MULT,  dtype=torch.float32)
_ITEM_AMOUNT_CONST = torch.tensor(en.ITEM_AMOUNT_CONST, dtype=torch.float32)


class EntityDequantizer(nn.Module):
    """Engine-native entity block → ``(B, N, 19)`` ``entity_scalars_raw``.

    Native obs is the only input contract; the dataloader is
    responsible for materializing the per-field tensors below.

    Inputs (from the dataloader after variable-length read + batch
    pad to N_max-in-batch):

      entity_types          (B, N) i64/i8 — type per slot; -1 for empty
      entity_subject_id     (B, N) u8
      entity_modality_id    (B, N) u8
      entity_player_id      (B, N) u8     — actor only, 0 elsewhere
      entity_event_count    (B, N) u8
      entity_event_actions  (B, N, 4) u8
      entity_event_sources  (B, N, 4) u8
      entity_half_extents   (B, N, 3) u8  — zero for projectile
      entity_rel            (B, N, 3) i16
      entity_vel            (B, N, 3) i16 — zero for item / mover
      entity_path           (B, N, 3) i16 — zero for projectile
      entity_path_dist      (B, N)    u16 — zero for projectile
      entity_eta            (B, N)    f16 — zero for projectile
      entity_recency        (B, N)    f16
      entity_facing         (B, N)    u8  — actor only
      entity_team           (B, N)    u8  — actor only
      entity_score          (B, N)    u8  — actor only
      entity_amount         (B, N)    u8  — item only
      entity_regen          (B, N)    f16 — item only
      entity_state          (B, N)    u8  — mover only

    Outputs (Tokenizer-ready):

      entity_scalars_raw   (B, N, 19) f32 — legacy slot layout per type
      entity_types         (B, N)    i64
      entity_ids           (B, N, 3) i64
      entity_event_actions (B, N, 4) i64
      entity_event_sources (B, N, 4) i64
      entity_event_counts  (B, N)    i64
    """

    def __init__(self) -> None:
        super().__init__()
        # Per-subject item amount lookup tables. Register as buffers so
        # they follow the module's device under .to() / .cuda() and stay
        # serializable in state_dicts (without being trained parameters).
        self.register_buffer("_amount_mult",  _ITEM_AMOUNT_MULT,  persistent=False)
        self.register_buffer("_amount_const", _ITEM_AMOUNT_CONST, persistent=False)

    def forward(
        self, obs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Idempotent: if obs already has the dequantized
        # ``entity_scalars_raw`` (preload ran the dequant), pass through.
        if "entity_scalars_raw" in obs:
            return dict(obs)
        out: dict[str, torch.Tensor] = dict(obs)

        et          = obs["entity_types"].to(torch.int64)         # (B, N)
        subject_id  = obs["entity_subject_id"].to(torch.int64)    # (B, N)
        half        = obs["entity_half_extents"].to(torch.float32) / en.DIST_SCALE
        rel         = obs["entity_rel"].to(torch.float32)         / en.DIST_SCALE
        vel         = obs["entity_vel"].to(torch.float32)         / en.MAX_VELOCITY
        path        = obs["entity_path"].to(torch.float32)        / en.DIST_SCALE
        path_dist   = obs["entity_path_dist"].to(torch.float32)   / en.DIST_SCALE
        eta         = obs["entity_eta"].to(torch.float32)         / en.TIME_SCALE
        recency     = obs["entity_recency"].to(torch.float32)     / en.TIME_SCALE
        facing      = obs["entity_facing"].to(torch.float32)      / 255.0
        team        = obs["entity_team"].to(torch.float32)
        score       = obs["entity_score"].to(torch.float32)       / 255.0
        regen       = obs["entity_regen"].to(torch.float32)       / en.TIME_SCALE
        state       = obs["entity_state"].to(torch.float32)       / 255.0

        # Item amount = raw × mult[subject] + const[subject].
        # Raw-dependent subjects (ammo/health/armor) have const=0; the
        # mult folds in MAX_<type> and the armor type factor where
        # relevant. Constant subjects (powerups, weapons) have mult=0
        # and const set to the canonical normalized value — wire raw
        # bytes are unused but harmless.
        raw_amount = obs["entity_amount"].to(torch.float32)       # (B, N)
        subj_clamp = subject_id.clamp_min(0).clamp_max(self._amount_mult.numel() - 1)
        amount = raw_amount * self._amount_mult[subj_clamp] + self._amount_const[subj_clamp]

        # dist = |rel| / DIST_SCALE — but rel is already pre-scaled, so:
        dist = torch.linalg.norm(rel, dim=-1)                     # (B, N)

        batch, n_max = et.shape
        scalars = torch.zeros(
            batch, n_max, ACTOR_SCALAR_DIM,
            device=et.device, dtype=torch.float32,
        )

        # Per-type writes via boolean masks. Each branch is the slot
        # layout the C-side emit_{type} produced; the legacy
        # entity_scalars_raw passes through the Tokenizer's per-type
        # Linear, which projects only the [:type_scalar_dim] prefix.

        mask_actor = (et == TOKEN_ACTOR)
        if mask_actor.any():
            scalars[..., 0:3]   = torch.where(mask_actor.unsqueeze(-1), half,       scalars[..., 0:3])
            scalars[..., 3:6]   = torch.where(mask_actor.unsqueeze(-1), rel,        scalars[..., 3:6])
            scalars[..., 6]     = torch.where(mask_actor,               dist,       scalars[..., 6])
            scalars[..., 7:10]  = torch.where(mask_actor.unsqueeze(-1), vel,        scalars[..., 7:10])
            scalars[..., 10:13] = torch.where(mask_actor.unsqueeze(-1), path,       scalars[..., 10:13])
            scalars[..., 13]    = torch.where(mask_actor,               path_dist,  scalars[..., 13])
            scalars[..., 14]    = torch.where(mask_actor,               eta,        scalars[..., 14])
            scalars[..., 15]    = torch.where(mask_actor,               facing,     scalars[..., 15])
            scalars[..., 16]    = torch.where(mask_actor,               team,       scalars[..., 16])
            scalars[..., 17]    = torch.where(mask_actor,               score,      scalars[..., 17])
            scalars[..., 18]    = torch.where(mask_actor,               recency,    scalars[..., 18])

        mask_proj = (et == TOKEN_PROJECTILE)
        if mask_proj.any():
            scalars[..., 0:3]  = torch.where(mask_proj.unsqueeze(-1), rel,     scalars[..., 0:3])
            scalars[..., 3]    = torch.where(mask_proj,               dist,    scalars[..., 3])
            scalars[..., 4:7]  = torch.where(mask_proj.unsqueeze(-1), vel,     scalars[..., 4:7])
            scalars[..., 7]    = torch.where(mask_proj,               recency, scalars[..., 7])

        mask_item = (et == TOKEN_ITEM)
        if mask_item.any():
            scalars[..., 0:3]   = torch.where(mask_item.unsqueeze(-1), half,       scalars[..., 0:3])
            scalars[..., 3:6]   = torch.where(mask_item.unsqueeze(-1), rel,        scalars[..., 3:6])
            scalars[..., 6]     = torch.where(mask_item,               dist,       scalars[..., 6])
            scalars[..., 7:10]  = torch.where(mask_item.unsqueeze(-1), path,       scalars[..., 7:10])
            scalars[..., 10]    = torch.where(mask_item,               path_dist,  scalars[..., 10])
            scalars[..., 11]    = torch.where(mask_item,               eta,        scalars[..., 11])
            scalars[..., 12]    = torch.where(mask_item,               amount,     scalars[..., 12])
            scalars[..., 13]    = torch.where(mask_item,               regen,      scalars[..., 13])
            scalars[..., 14]    = torch.where(mask_item,               recency,    scalars[..., 14])

        mask_mover = (et == TOKEN_MOVER)
        if mask_mover.any():
            scalars[..., 0:3]   = torch.where(mask_mover.unsqueeze(-1), half,       scalars[..., 0:3])
            scalars[..., 3:6]   = torch.where(mask_mover.unsqueeze(-1), rel,        scalars[..., 3:6])
            scalars[..., 6]     = torch.where(mask_mover,               dist,       scalars[..., 6])
            scalars[..., 7:10]  = torch.where(mask_mover.unsqueeze(-1), path,       scalars[..., 7:10])
            scalars[..., 10]    = torch.where(mask_mover,               path_dist,  scalars[..., 10])
            scalars[..., 11]    = torch.where(mask_mover,               eta,        scalars[..., 11])
            scalars[..., 12]    = torch.where(mask_mover,               state,      scalars[..., 12])
            scalars[..., 13]    = torch.where(mask_mover,               recency,    scalars[..., 13])

        # Pack entity_ids into the (B, N, 3) layout the Tokenizer reads.
        ids = torch.stack([
            obs["entity_subject_id"].to(torch.int64),
            obs["entity_modality_id"].to(torch.int64),
            obs["entity_player_id"].to(torch.int64),
        ], dim=-1)

        out["entity_scalars_raw"]   = scalars
        out["entity_types"]         = et
        out["entity_ids"]           = ids
        out["entity_event_actions"] = obs["entity_event_actions"].to(torch.int64)
        out["entity_event_sources"] = obs["entity_event_sources"].to(torch.int64)
        out["entity_event_counts"]  = obs["entity_event_count"].to(torch.int64)
        return out
