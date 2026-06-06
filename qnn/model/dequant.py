"""Native-width → model-facing tensor adapters.

Three dequantizer modules convert the engine-native dicts produced
by the new wire format (see ``qnn.engine_norm``) into the float /
int tensors the existing ObsEmbedding + heads consume:

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
any bit / index demuxing. Output dict keys feed the ObsEmbedding's
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

# Weapon vocab IDs in axe-first impulse order (engine impulse 1..8).
# These pair with the per-weapon readiness vector (B, 8) the dequant
# emits — the ObsEmbedding looks each up in `entity_embed` and weights
# it by the readiness scalar to form the arsenal subtoken contribution.
_WEAPON_SUBJECT_IDS = (
    ENTITY_IDS["AXE"],
    ENTITY_IDS["SHOTGUN"],
    ENTITY_IDS["SUPER_SHOTGUN"],
    ENTITY_IDS["NAILGUN"],
    ENTITY_IDS["SUPER_NAILGUN"],
    ENTITY_IDS["GRENADE_LAUNCHER"],
    ENTITY_IDS["ROCKET_LAUNCHER"],
    ENTITY_IDS["THUNDERBOLT"],
)
_N_WEAPONS = len(_WEAPON_SUBJECT_IDS)

# Legacy 17-wide self_scalars layout. The ObsEmbedding no longer uses
# this tensor (it reads the three subtoken tensors below instead) but
# downstream ablation heads / feature registry entries / labeler probes
# still index it by these positions, so the dequant keeps emitting it
# as-is. Do not reorder without an architectural retrain.
_SELF_SCALAR_DIM = 17
_IDX_HEALTH          = 0
_IDX_ARMOR           = 1
_IDX_WEAPON_SG       = 2
_IDX_WEAPON_SSG      = 3
_IDX_WEAPON_NG       = 4
_IDX_WEAPON_SNG      = 5
_IDX_WEAPON_GL       = 6
_IDX_WEAPON_RL       = 7
_IDX_WEAPON_LG       = 8
# Public — bench heads (e.g. weapon_aim) read these indices off self_scalars
# directly; keep the source-of-truth here so an obs-layout change can't drift
# between the dequantizer that writes them and the bench consumers that read.
IDX_AMMO_SHELLS      = 9
IDX_AMMO_NAILS       = 10
IDX_AMMO_ROCKETS     = 11
IDX_AMMO_CELLS       = 12
_IDX_VEL_X           = 13
_IDX_VEL_Y           = 14
_IDX_VEL_Z           = 15
IDX_ATTACK_FINISHED  = 16
# Back-compat private aliases for the dequantizer's internal writes.
_IDX_AMMO_SHELLS     = IDX_AMMO_SHELLS
_IDX_AMMO_NAILS      = IDX_AMMO_NAILS
_IDX_AMMO_ROCKETS    = IDX_AMMO_ROCKETS
_IDX_AMMO_CELLS      = IDX_AMMO_CELLS
_IDX_ATTACK_FINISHED = IDX_ATTACK_FINISHED

# Self subtoken scalar widths, consumed by the three projections in
# ObsEmbedding (`self_proj_state`, `self_proj_arsenal`, `self_proj_motion`).
SELF_STATE_SCALAR_DIM   = 2  # health, effective_armor
SELF_ARSENAL_SCALAR_DIM = 1  # attack_finished
SELF_MOTION_SCALAR_DIM  = 4  # vel_xyz, view_pitch


class SelfDequantizer(nn.Module):
    """Engine-native self block → ObsEmbedding-ready float / int tensors.

    Emits three subtoken scalar tensors (state / arsenal / motion), a
    per-weapon readiness vector, and powerup IDs grouped by which
    subtoken they route into. The legacy 17-wide ``self_scalars`` tensor
    is kept alongside for feature-registry / labeler-probe consumers
    that still index it by idx position.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, obs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert a native obs dict to the ObsEmbedding-expected obs dict.

        Idempotent: if obs already carries the dequantized
        ``self_state_scalars`` key (e.g. ``make_resident_source`` ran the
        dequant once at startup), pass through unchanged.
        """
        if "self_state_scalars" in obs:
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
        view_pitch  = obs["view_pitch"]        # (B,) i8 — deg/90 quantized

        device = health.device
        batch = health.shape[0]

        # ── Normalized scalar floats ────────────────────────────────
        health_f    = health.to(torch.float32)    / en.MAX_HEALTH
        eff_armor_f = eff_armor.to(torch.float32) / en.MAX_ARMOR_EFFECT
        ammo_sh_f   = ammo_sh.to(torch.float32)   / en.MAX_SHELLS
        ammo_na_f   = ammo_na.to(torch.float32)   / en.MAX_NAILS
        ammo_rk_f   = ammo_rk.to(torch.float32)   / en.MAX_ROCKETS
        ammo_ce_f   = ammo_ce.to(torch.float32)   / en.MAX_CELLS
        vel_f       = vel.to(torch.float32)       / en.MAX_VELOCITY
        af_f        = af.to(torch.float32)        / en.TIME_SCALE
        # i8 already encodes pitch/90 with /127 quantization on the wire,
        # so divide by 127 to recover the [-1, 1] interval.
        pitch_f     = view_pitch.to(torch.float32) / 127.0

        # ── Subtoken scalar tensors fed to the three ObsEmbedding projs ─
        out["self_state_scalars"] = torch.stack(
            [health_f, eff_armor_f], dim=1,
        )                                                              # (B, 2)
        out["self_arsenal_scalars"] = af_f.unsqueeze(-1)               # (B, 1)
        out["self_motion_scalars"] = torch.cat(
            [vel_f, pitch_f.unsqueeze(-1)], dim=1,
        )                                                              # (B, 4)

        # ── Legacy flat self_scalars (kept for downstream consumers) ─
        scalars = torch.zeros(batch, _SELF_SCALAR_DIM, device=device, dtype=torch.float32)
        scalars[:, _IDX_HEALTH]       = health_f
        scalars[:, _IDX_ARMOR]        = eff_armor_f
        scalars[:, _IDX_AMMO_SHELLS]  = ammo_sh_f
        scalars[:, _IDX_AMMO_NAILS]   = ammo_na_f
        scalars[:, _IDX_AMMO_ROCKETS] = ammo_rk_f
        scalars[:, _IDX_AMMO_CELLS]   = ammo_ce_f
        scalars[:, _IDX_VEL_X:_IDX_VEL_Z + 1] = vel_f
        scalars[:, _IDX_ATTACK_FINISHED] = af_f

        items_i64 = items.to(torch.int64)
        # 7 ammo-using weapon-owned bits in legacy idx order (SG..LG).
        owned_bits = torch.stack([
            (items_i64 & en.IT_SHOTGUN)          != 0,
            (items_i64 & en.IT_SUPER_SHOTGUN)    != 0,
            (items_i64 & en.IT_NAILGUN)          != 0,
            (items_i64 & en.IT_SUPER_NAILGUN)    != 0,
            (items_i64 & en.IT_GRENADE_LAUNCHER) != 0,
            (items_i64 & en.IT_ROCKET_LAUNCHER)  != 0,
            (items_i64 & en.IT_LIGHTNING)        != 0,
        ], dim=1).to(torch.float32)                                    # (B, 7)
        scalars[:, _IDX_WEAPON_SG:_IDX_WEAPON_LG + 1] = owned_bits

        # ── Per-weapon readiness in ENTITY_IDS weapon order ───────
        # Order: [AXE, SG, SSG, NG, SNG, GL, RL, LG] — matches
        # _WEAPON_SUBJECT_IDS so the ObsEmbedding can fold this into a
        # single embedding-lookup-and-sum.
        # readiness = 0.1 + 0.9 × (shots_remaining / MAX_shots), masked by
        # ownership. For every ammo-weapon, pool_cap = cost_per_shot ×
        # MAX_shots exactly (e.g. SSG: 2 × 50 = 100 = MAX_SHELLS), so
        # shots_remaining/MAX_shots collapses to the normalized pool
        # fraction the dequant already computes. Two pool-sharing
        # weapons (GL/RL, SG/SSG, NG/SNG) read the same scalar with
        # different MAX_shots semantically, but the collapse means
        # they share the same readiness value here — which is correct:
        # SSG firing leaves SG ammo at the same fraction.
        axe_owned = ((items_i64 & en.IT_AXE) != 0).to(torch.float32)
        sh_ready  = self._ammo_readiness(ammo_sh_f)
        na_ready  = self._ammo_readiness(ammo_na_f)
        rk_ready  = self._ammo_readiness(ammo_rk_f)
        ce_ready  = self._ammo_readiness(ammo_ce_f)

        readiness = torch.stack([
            axe_owned,                            # AXE: 1 if owned else 0
            sh_ready * owned_bits[:, 0],          # SG
            sh_ready * owned_bits[:, 1],          # SSG
            na_ready * owned_bits[:, 2],          # NG
            na_ready * owned_bits[:, 3],          # SNG
            rk_ready * owned_bits[:, 4],          # GL
            rk_ready * owned_bits[:, 5],          # RL
            ce_ready * owned_bits[:, 6],          # LG
        ], dim=1)                                                      # (B, 8)

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

        # ── Powerup IDs routed by primary effect (per self-spatial design)
        # state ← PENT (invuln), RING (invisibility), MEGAHEALTH
        # arsenal ← QUAD (damage mult)
        # motion ← SUIT (lava/slime traversal)
        # Each slot holds the powerup's ENTITY_IDS subject if present,
        # else 0 (NONE) — ObsEmbedding masks zeros at embed-lookup time.
        zero = torch.zeros(batch, dtype=torch.int64, device=device)
        have_pent = (items_i64 & en.IT_INVULNERABILITY) != 0
        have_ring = (items_i64 & en.IT_INVISIBILITY)    != 0
        have_mega = health.to(torch.int64) > 100
        have_quad = (items_i64 & en.IT_QUAD)            != 0
        have_suit = (items_i64 & en.IT_SUIT)            != 0

        out["self_state_powerup_ids"] = torch.stack([
            torch.where(have_pent, torch.full_like(zero, _SUBJECT_PENT),       zero),
            torch.where(have_ring, torch.full_like(zero, _SUBJECT_RING),       zero),
            torch.where(have_mega, torch.full_like(zero, _SUBJECT_MEGAHEALTH), zero),
        ], dim=1)                                                      # (B, 3)
        out["self_arsenal_powerup_ids"] = torch.where(
            have_quad, torch.full_like(zero, _SUBJECT_QUAD), zero,
        ).unsqueeze(-1)                                                # (B, 1)
        out["self_motion_powerup_ids"] = torch.where(
            have_suit, torch.full_like(zero, _SUBJECT_SUIT), zero,
        ).unsqueeze(-1)                                                # (B, 1)

        out["self_scalars"]         = scalars
        out["self_weapon_readiness"] = readiness
        out["self_weapon_id"]       = obs["self_weapon_id"].to(torch.int64).unsqueeze(-1)
        out["self_armor_type_id"]   = armor_type.unsqueeze(-1)
        out["self_movement_id"]     = obs["self_movement_id"].to(torch.int64).unsqueeze(-1)
        return out

    @staticmethod
    def _ammo_readiness(pool_norm: torch.Tensor) -> torch.Tensor:
        # 0.1 floor keeps owned-empty distinct from not-owned (which the
        # caller's ownership mask zeros out entirely). 0.9 × pool_norm
        # gives full 1.0 at pool cap.
        return 0.1 + 0.9 * pool_norm.clamp(0.0, 1.0)


# ── SpatialDequantizer ───────────────────────────────────────────

# spatial_scalars idx layout — mirrors qnn_onnx.c:374-386. The
# ObsEmbedding's spatial_proj is nn.Linear(13, d_model); trained
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
    """Engine-native spatial block → ObsEmbedding-ready ``spatial_scalars``.

    Input: per-field native-typed tensors (per qnn.engine_norm.SPATIAL_FIELDS).
    Output: ``spatial_scalars`` (B, 9, 13) float32 in the layout the
    ObsEmbedding's ``spatial_proj`` consumes.
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


# ActionDequantizer was deleted along with the sparse act_target_probs
# encoding. The target distribution is now recomputed at training start
# from obs+actions by qnn.bc.train._compute_target_probs and arrives as
# dense (T, TARGET_PROBS_CLASSES) float32 — heads consume it directly,
# no model-side expansion needed.


# ── EntityDequantizer ────────────────────────────────────────────

# Per-type scalar idx layouts in the legacy (B, N, ACTOR_SCALAR_DIM=19)
# entity_scalars_raw tensor that the ObsEmbedding's per-type Linear
# projections consume. These mirror the C side qnn_onnx.c:194-318
# emit_{actor,projectile,item,mover} functions exactly so trained
# checkpoints stay valid.
#
# Idx indices the model expects (post-dist-recompute):
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

      entity_types          (B, N) i64/i8 — type per idx; -1 for empty
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

    Outputs (ObsEmbedding-ready):

      entity_scalars_raw   (B, N, 19) f32 — legacy idx layout per type
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

        # Per-type writes via boolean masks. Each branch is the idx
        # layout the C-side emit_{type} produced; the legacy
        # entity_scalars_raw passes through the ObsEmbedding's per-type
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

        # Pack entity_ids into the (B, N, 3) layout the ObsEmbedding reads.
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
