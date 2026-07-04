"""The a24-lineage guard set — decode-time vetoes shared by Python eval and ONNX
export so the two paths stay in lockstep.

This is the UNIFIED guard module (the former rc1/rc2/a25rc1a chain folded into
one). The attack guards (attack-splash / RL EHR self-splash / LG range) are
regime-invariant. The projectile dodge-release (Gate B move-hold release) is the
only axis that varied across the old regimes, so it is param-SELECTED here:

  off    — no release (former a25rc1a behavior)
  rocket — straight-rocket closed-form closest-approach, 0.2s (former a24rc1)
  any    — rocket+grenade+nail, splash radius, gravity-arc trajectory sampling,
           1.0s (former a24rc2)

:func:`make_guard` binds the params from a decode config to a small adapter
exposing the entry points the policy/export call (``guard_attack_logit_for_export``,
``policy_decode_action_postprocess``, and ``projectile_release_mask`` only when the
mode is not "off" — so an off-mode export bakes no release nodes, matching the old
a25rc1a graph). Selecting a regime is now picking a decode config, not a module.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from typing import Any

import torch

from qnn.actions import MOVE_AXES, MOVE_CLASS_POS
from qnn import engine_norm as en
from qnn.bc.weapon_physics import (
    WEAPON_PHYSICS,
    ACTOR_TEAM_OFFSET as _ACTOR_TEAM_OFFSET,
    TEAM_TEAMMATE_VALUE as _TEAM_TEAMMATE_VALUE,
)
from qnn.vocab import ENTITY_IDS, SPATIAL_SECTOR_IDS, TOKEN_ACTOR, TOKEN_PROJECTILE


# Spatial/entity distances are normalized by DIST_SCALE by the dequantizers.
ATTACK_SPLASH_GUARD_NEAREST: float = 80.0 / en.DIST_SCALE
ATTACK_SPLASH_GUARD_MEAN: float = 150.0 / en.DIST_SCALE
ATTACK_SPLASH_GUARD_ACTOR_MARGIN: float = 120.0 / en.DIST_SCALE
_FOV_CENTER_IDX: int = SPATIAL_SECTOR_IDS["FOV_Center"]
_GROUND_SECTOR_IDX: int = SPATIAL_SECTOR_IDS["Ground_State"]
_SP_NEAREST_DIST: int = 3
_SP_MEAN_DIST: int = 4
_ACTOR_DIST: int = 6
_ACTOR_REL_FWD: int = 3   # ACTOR_REL_OFFSET; view-frame rel x = forward component
_SPLASH_GUARD_WEAPON_IDS: tuple[int, int] = (
    ENTITY_IDS["GRENADE_LAUNCHER"],
    ENTITY_IDS["ROCKET_LAUNCHER"],
)

# ── LG (thunderbolt) range + alignment guard ─────────────────────────────
# The lightning bolt is a traceline of ~600 u; beyond that it hits nothing, and
# off the aim cone the thin beam misses. Suppress LG attack unless some enemy
# actor is BOTH within range AND inside the aim cone — the only case the beam
# can connect. self_weapon_id is ENTITY_IDS-encoded (subject space), so LG is
# THUNDERBOLT (10), NOT a raw impulse.
_LG_ID: int = ENTITY_IDS["THUNDERBOLT"]
LG_RANGE_U: float = 600.0
LG_RANGE: float = LG_RANGE_U / en.DIST_SCALE
LG_ALIGN_HALF_ANGLE_DEG: float = 15.0
_LG_ALIGN_COS: float = math.cos(math.radians(LG_ALIGN_HALF_ANGLE_DEG))

# ── Gate B: rocket incoming-hit move-release trigger ─────────────────────
# Release the move hold when a perceived ROCKET is projected to strike the
# player within PROJECTILE_RELEASE_HORIZON_S *if current movement continues*.
# Straight-line closest-approach of the rocket path relative to the player
# (relative velocity = rocket_vel - player_vel, both view-frame world vel).
# Rockets only: they fly straight and fast so the linear projection is valid;
# grenades arc (projection invalid) and nails would over-trigger.
_ROCKET_ID: int = ENTITY_IDS["PROJECTILE_ROCKET"]
# Projectile token layout in entity_scalars_raw (dequant): rel[0:3], dist[3],
# vel[4:7], recency[7]. Distinct from the actor layout used above.
_PROJ_REL_OFFSET: int = 0
_PROJ_VEL_OFFSET: int = 4
PROJECTILE_HIT_RADIUS_U: float = 45.0
_PROJ_HIT_RADIUS_SQ: float = (PROJECTILE_HIT_RADIUS_U / en.DIST_SCALE) ** 2
PROJECTILE_RELEASE_HORIZON_S: float = 0.200
# rel is /DIST_SCALE, vel is /MAX_VELOCITY; rel_norm(t) = rel + vel·(MAX_VELOCITY/
# DIST_SCALE)·t_sec, so the closest-approach t runs [0, horizon_s · factor].
_PROJ_HORIZON_T: float = PROJECTILE_RELEASE_HORIZON_S * (en.MAX_VELOCITY / en.DIST_SCALE)


def projectile_release_mask_rocket(
    obs_tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """(B,) rows where a rocket is projected to hit the player within the horizon
    if current movement continues — the Gate B move-hold release trigger.

    Straight-rocket closed-form closest-approach (former a24rc1 ``rocket`` mode)."""
    entity_types = obs_tensors.get("entity_types")
    entity_scalars = obs_tensors.get("entity_scalars_raw")
    entity_ids = obs_tensors.get("entity_ids")
    motion = obs_tensors.get("self_motion_scalars")
    present = [t for t in (entity_types, entity_scalars, entity_ids, motion) if t is not None]
    if len(present) < 4 or any(t.numel() == 0 for t in present):
        n = 1 if not present else present[0].reshape(present[0].shape[0], -1).shape[0]
        return torch.zeros(n, dtype=torch.bool,
                           device=(present[0].device if present else "cpu"))
    device = entity_scalars.device
    entity_types = entity_types.to(device=device).reshape(-1, entity_types.shape[-1])
    entity_scalars = entity_scalars.to(device=device).reshape(
        -1, entity_scalars.shape[-2], entity_scalars.shape[-1])
    entity_ids = entity_ids.to(device=device).reshape(-1, entity_ids.shape[-2], entity_ids.shape[-1])
    motion = motion.to(device=device).reshape(-1, motion.shape[-1])

    proj_rel = entity_scalars[:, :, _PROJ_REL_OFFSET:_PROJ_REL_OFFSET + 3]   # (B, N, 3)
    proj_vel = entity_scalars[:, :, _PROJ_VEL_OFFSET:_PROJ_VEL_OFFSET + 3]   # (B, N, 3) world, view-frame
    player_vel = motion[:, 0:3].unsqueeze(1)                                  # (B, 1, 3)
    rel_vel = proj_vel - player_vel                                           # (B, N, 3) relative ("continue")

    rv = (proj_rel * rel_vel).sum(dim=-1)                                     # (B, N)
    vv = (rel_vel * rel_vel).sum(dim=-1)                                      # (B, N)
    t_star = (-rv / vv.clamp_min(1e-12)).clamp(0.0, _PROJ_HORIZON_T)          # (B, N)
    closest = proj_rel + rel_vel * t_star.unsqueeze(-1)                       # (B, N, 3)
    min_d2 = (closest * closest).sum(dim=-1)                                  # (B, N)

    subj = entity_ids[:, :, 0]                                                # (B, N) subject id
    is_rocket = (entity_types == TOKEN_PROJECTILE) & (subj == _ROCKET_ID)
    threat = is_rocket & (min_d2 < _PROJ_HIT_RADIUS_SQ)
    return threat.any(dim=1)                                                  # (B,)

# ── RL EHR-scaled self-splash guard ──────────────────────────────────────
# Rocket radius damage is 120; T_RadiusDamage uses findradius(damage+40)=160u
# and the attacker branch halves self-damage, so self-splash at surface
# distance d is 0.5*(120 - 0.5*d) for d < 160u, else 0 (frikbotnex/combat.qc
# T_RadiusDamage). Max self-splash = 60 at d=0 → a player with EHR >= 60
# cannot rocket-suicide on flat ground. Solving self_splash == (EHR - margin)
# for the lethal standoff: d_safe = 4*(60 - EHR + margin), clamped to
# [0, 160] world units. EHR = health + effective_armor (both in HP).
_RL_LAUNCHER_ID: int = ENTITY_IDS["ROCKET_LAUNCHER"]
_RL_SPLASH_MAX_SELF: float = 60.0     # 0.5 * 120 (point-blank attacker damage)
_RL_SPLASH_FINDRADIUS: float = 160.0  # damage + 40; splash = 0 beyond this
ROCKET_EHR_SAFETY_MARGIN_HP: float = 10.0   # hold fire if it would leave < this HP
ROCKET_FLOOR_MIN_PITCH_DEG: float = 12.0    # below this downward pitch, skip the floor test
_DEG2RAD: float = 3.141592653589793 / 180.0


def attack_splash_guard_mask(
    obs_tensors: Mapping[str, torch.Tensor],
    move_classes: torch.Tensor,
) -> torch.Tensor:
    """Rows where a24rc1 suppresses attack to avoid square wall splash.

    Condition:
      held in {GL, RL}
      AND not decoded jump
      AND FOV-center nearest < 80u
      AND FOV-center mean < 150u
      AND no perceived actor within nearest + 120u
    """
    device = move_classes.device
    moves = move_classes.to(device=device).reshape(-1, MOVE_AXES)
    false = torch.zeros(moves.shape[0], dtype=torch.bool, device=device)
    spatial = obs_tensors.get("spatial_scalars")
    entity_types = obs_tensors.get("entity_types")
    entity_scalars = obs_tensors.get("entity_scalars_raw")
    weapon_id = obs_tensors.get("self_weapon_id")
    if spatial is None or entity_types is None or entity_scalars is None or weapon_id is None:
        return false
    if spatial.numel() == 0 or entity_types.numel() == 0 or entity_scalars.numel() == 0:
        return false

    spatial = spatial.to(device=device).reshape(-1, spatial.shape[-2], spatial.shape[-1])
    entity_types = entity_types.to(device=device).reshape(-1, entity_types.shape[-1])
    entity_scalars = entity_scalars.to(device=device).reshape(
        -1, entity_scalars.shape[-2], entity_scalars.shape[-1]
    )
    weapon_id = weapon_id.to(device=device).reshape(-1)

    nearest = spatial[:, _FOV_CENTER_IDX, _SP_NEAREST_DIST]
    mean = spatial[:, _FOV_CENTER_IDX, _SP_MEAN_DIST]
    held_splash_weapon = (
        (weapon_id == _SPLASH_GUARD_WEAPON_IDS[0])
        | (weapon_id == _SPLASH_GUARD_WEAPON_IDS[1])
    )
    jumping = moves[:, 2] == MOVE_CLASS_POS

    actor_mask = entity_types == TOKEN_ACTOR
    actor_dist = entity_scalars[:, :, _ACTOR_DIST]
    inf_dist = torch.full_like(actor_dist, float("inf"))
    nearest_actor = torch.where(actor_mask, actor_dist, inf_dist).amin(dim=1)
    actor_within_guard_band = nearest_actor <= (
        nearest + ATTACK_SPLASH_GUARD_ACTOR_MARGIN
    )

    return (
        held_splash_weapon
        & ~jumping
        & (nearest < ATTACK_SPLASH_GUARD_NEAREST)
        & (mean < ATTACK_SPLASH_GUARD_MEAN)
        & ~actor_within_guard_band
    )


def rocket_self_splash_guard_mask(
    obs_tensors: Mapping[str, torch.Tensor],
    move_classes: torch.Tensor,
) -> torch.Tensor:
    """Rows where an RL shot's self-splash would be lethal given EHR.

    RL only. Unlike :func:`attack_splash_guard_mask` this deliberately
    IGNORES the jump exemption and the nearby-actor escape: at low EHR even a
    "justified" point-blank rocket at an adjacent enemy self-kills, and the
    rocket-jump-into-floor case is invisible to the yaw-only FOV_Center sector
    the other guard keys off.

    The shot is suppressed when it detonates within the EHR-scaled safe
    standoff ``d_safe = 4*(60 - EHR + margin)`` (clamped [0, 160] u) on any of:
      * a horizontal wall  — FOV_Center nearest, or
      * the floor          — Ground-sector nearest along a downward aim, taken
                             as the slant distance ``ground / sin(pitch)`` so
                             shallow downward shots at distant lower enemies
                             pass while a steep rocket-jump blast is caught, or
      * a point-blank enemy — nearest perceived actor in the forward hemisphere.
                             The rocket detonates on the body, not the wall
                             behind it, so a close enemy is a near surface the
                             FOV_Center BSP geometry misses (the live self-frag
                             that slipped past the wall/floor tests).

    ``move_classes`` is used only for the all-False fallback shape/device.
    """
    device = move_classes.device
    n_rows = move_classes.to(device=device).reshape(-1, MOVE_AXES).shape[0]
    false = torch.zeros(n_rows, dtype=torch.bool, device=device)

    state = obs_tensors.get("self_state_scalars")    # (B, 2) health, eff_armor (normalized)
    motion = obs_tensors.get("self_motion_scalars")  # (B, 4) vel(3), view_pitch (deg/90)
    spatial = obs_tensors.get("spatial_scalars")
    weapon_id = obs_tensors.get("self_weapon_id")
    entity_types = obs_tensors.get("entity_types")
    entity_scalars = obs_tensors.get("entity_scalars_raw")
    if (state is None or motion is None or spatial is None or weapon_id is None
            or entity_types is None or entity_scalars is None):
        return false
    if (state.numel() == 0 or motion.numel() == 0 or spatial.numel() == 0
            or entity_types.numel() == 0 or entity_scalars.numel() == 0):
        return false

    state = state.to(device=device).reshape(-1, state.shape[-1])
    motion = motion.to(device=device).reshape(-1, motion.shape[-1])
    spatial = spatial.to(device=device).reshape(-1, spatial.shape[-2], spatial.shape[-1])
    weapon_id = weapon_id.to(device=device).reshape(-1)
    entity_types = entity_types.to(device=device).reshape(-1, entity_types.shape[-1])
    entity_scalars = entity_scalars.to(device=device).reshape(
        -1, entity_scalars.shape[-2], entity_scalars.shape[-1]
    )

    # EHR in HP from the normalized state scalars.
    ehr = state[:, 0] * en.MAX_HEALTH + state[:, 1] * en.MAX_ARMOR_EFFECT     # (B,)
    # Lethal standoff (world units) → normalized; clamp to [0, findradius].
    d_safe_u = (4.0 * (_RL_SPLASH_MAX_SELF - ehr + ROCKET_EHR_SAFETY_MARGIN_HP)).clamp(
        0.0, _RL_SPLASH_FINDRADIUS
    )
    d_safe = d_safe_u / en.DIST_SCALE                                          # (B,)

    # Horizontal wall in the yaw aim direction (already normalized).
    wall = spatial[:, _FOV_CENTER_IDX, _SP_NEAREST_DIST]                       # (B,)

    # Floor under a downward aim. view_pitch is deg/90, positive = looking down.
    pitch_deg = motion[:, 3] * 90.0                                           # (B,)
    aiming_down = pitch_deg > ROCKET_FLOOR_MIN_PITCH_DEG
    sin_pitch = torch.sin(pitch_deg.clamp(min=0.0) * _DEG2RAD)
    ground = spatial[:, _GROUND_SECTOR_IDX, _SP_NEAREST_DIST]                  # (B,)
    # Slant distance to the floor impact; +inf where not aiming down so it
    # drops out of the min() (and avoids the sin→0 blow-up).
    floor_slant = ground / sin_pitch.clamp(min=1e-3)
    inf = torch.full_like(floor_slant, float("inf"))
    floor_slant = torch.where(aiming_down, floor_slant, inf)

    # Nearest perceived actor in the forward hemisphere (rel-forward > 0). The
    # rocket detonates on the body, so this is the detonation distance the
    # wall/floor surfaces don't see.
    actor_fwd = entity_scalars[:, :, _ACTOR_REL_FWD]                          # (B, N)
    actor_mask = (entity_types == TOKEN_ACTOR) & (actor_fwd > 0.0)
    actor_dist = entity_scalars[:, :, _ACTOR_DIST]                            # (B, N)
    inf_a = torch.full_like(actor_dist, float("inf"))
    nearest_actor = torch.where(actor_mask, actor_dist, inf_a).amin(dim=1)    # (B,)

    surface = torch.minimum(torch.minimum(wall, floor_slant), nearest_actor)  # (B,)
    held_rl = weapon_id == _RL_LAUNCHER_ID
    return held_rl & (surface < d_safe)


def lg_range_guard_mask(
    obs_tensors: Mapping[str, torch.Tensor],
    move_classes: torch.Tensor,
) -> torch.Tensor:
    """Rows where a24rc1 suppresses LG attack — the beam cannot connect.

    Condition:
      held weapon == THUNDERBOLT (LG)
      AND no enemy actor that is BOTH
        within range (dist <= LG_RANGE) AND
        inside the aim cone (forward component / dist >= cos(half-angle)).

    Pure geometry over perceived enemy actors — teammates excluded. No
    dependence on the target-pointer softmax: alignment against the view
    forward already encodes "who the crosshair is on", and weighting by the
    (noisy, slot-confounded) target belief would only false-suppress
    connecting shots when belief lags aim.
    """
    device = move_classes.device
    n_rows = move_classes.to(device=device).reshape(-1, MOVE_AXES).shape[0]
    false = torch.zeros(n_rows, dtype=torch.bool, device=device)
    entity_types = obs_tensors.get("entity_types")
    entity_scalars = obs_tensors.get("entity_scalars_raw")
    weapon_id = obs_tensors.get("self_weapon_id")
    if entity_types is None or entity_scalars is None or weapon_id is None:
        return false
    if entity_types.numel() == 0 or entity_scalars.numel() == 0:
        return false

    entity_types = entity_types.to(device=device).reshape(-1, entity_types.shape[-1])
    entity_scalars = entity_scalars.to(device=device).reshape(
        -1, entity_scalars.shape[-2], entity_scalars.shape[-1]
    )
    weapon_id = weapon_id.to(device=device).reshape(-1)

    held_lg = weapon_id == _LG_ID
    fwd = entity_scalars[:, :, _ACTOR_REL_FWD]          # (B, N) view-forward rel
    dist = entity_scalars[:, :, _ACTOR_DIST]            # (B, N) range
    team = entity_scalars[:, :, _ACTOR_TEAM_OFFSET]     # (B, N)
    enemy = (entity_types == TOKEN_ACTOR) & (team != _TEAM_TEAMMATE_VALUE)
    in_range = dist <= LG_RANGE
    # cos(angle to forward) = fwd / dist >= cos(theta)  ⇒  fwd >= dist * cos(theta).
    # Divide-free; negative fwd (behind) fails since dist*cos(theta) > 0.
    aligned = fwd >= dist * _LG_ALIGN_COS
    hittable = (enemy & in_range & aligned).any(dim=1)  # (B,)
    return held_lg & ~hittable


def apply_attack_splash_guard(
    obs_tensors: Mapping[str, torch.Tensor],
    move_classes: torch.Tensor,
    fire: torch.Tensor,
) -> torch.Tensor:
    """Apply the a24rc1 attack splash + LG-range guards to a decoded fire tensor."""
    mask = attack_splash_guard_mask(obs_tensors, move_classes)
    mask = mask | rocket_self_splash_guard_mask(obs_tensors, move_classes)
    mask = mask | lg_range_guard_mask(obs_tensors, move_classes)
    return torch.where(mask, torch.zeros_like(fire), fire)


def _actor_min_angle_rad(
    obs_tensors: Mapping[str, torch.Tensor],
) -> "torch.Tensor | None":
    """(B,) minimum angle (radians) from crosshair to nearest visible actor.

    Uses entity_scalars_raw[:,:,_ACTOR_REL_FWD] (dequantized X / forward
    component, /DIST_SCALE) and the full 3-component relative vector to compute
    arccos(unit_rel · forward).  Returns ``None`` when obs are absent.  Frames
    with no visible actor get angle = π (widest bucket).
    """
    entity_types = obs_tensors.get("entity_types")
    entity_scalars = obs_tensors.get("entity_scalars_raw")
    if entity_types is None or entity_scalars is None:
        return None
    if entity_types.numel() == 0 or entity_scalars.numel() == 0:
        return None
    types   = entity_types.reshape(-1, entity_types.shape[-1])       # (B, N)
    scalars = entity_scalars.reshape(-1, entity_scalars.shape[-2],
                                    entity_scalars.shape[-1])        # (B, N, S)
    # Full relative position (3 components, /DIST_SCALE) for norm.
    rel     = scalars[:, :, _ACTOR_REL_FWD:_ACTOR_REL_FWD + 3]      # (B, N, 3)
    norms   = torch.linalg.vector_norm(rel, dim=-1).clamp(min=1e-6)  # (B, N)
    cos_a   = (rel[:, :, 0] / norms).clamp(-1.0, 1.0)               # (B, N)
    angles  = torch.acos(cos_a)                                      # (B, N) rad
    is_actor = (types == TOKEN_ACTOR)
    pi_fill  = torch.full_like(angles, math.pi)
    angles   = torch.where(is_actor, angles, pi_fill)
    return angles.amin(dim=1)                                         # (B,)


# ── Alignment-conditioned fire-logit bias ─────────────────────────────────
# Calibrated from the human p_fire-by-LOS-angle distribution (attack_profile
# analysis, QWD val+train corpus).  Reference bucket: [0,2)° = 0.125 p_fire.
# Biases = log(p_human[bucket] / p_human_ref), so applying them at strength=1
# makes the attack rate per bucket match the human distribution shape.
# The threshold buckets match engine_los_attack_by_origin_angle evaluation buckets.
_ATTACK_ALIGN_THRESHOLDS_RAD = torch.tensor(
    [math.radians(d) for d in (2.0, 5.0, 10.0, 20.0, 45.0)],
    dtype=torch.float32)

# log(p_human[bucket] / 0.125) per bucket: [0-2)°, [2-5)°, [5-10)°, [10-20)°, [20-45)°, [45+)°
_ATTACK_ALIGN_BIASES = torch.tensor(
    [0.000, -0.092, -0.362, -0.820, -1.461, -1.789], dtype=torch.float32)


def attack_alignment_logit_bias(
    obs_tensors: Mapping[str, torch.Tensor],
    attack_logit: torch.Tensor,
    strength: float = 1.0,
    thresholds_rad: "torch.Tensor | None" = None,
    bias_vals: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Apply angle-conditioned logit bias to suppress attack at wide misalignment.

    The bias is calibrated to the human p_fire distribution: at ``strength=1``
    the attack rate shape per angle bucket matches the human corpus.
    ``strength`` is the human-percentile knob — higher values match higher-
    skill human fire selectivity.  At ``strength=0`` the logit is unchanged.

    ``thresholds_rad`` / ``bias_vals`` default to the corpus-fit constants above.
    """
    if strength <= 0.0:
        return attack_logit
    min_angle = _actor_min_angle_rad(obs_tensors)
    if min_angle is None:
        return attack_logit
    dev = attack_logit.device
    thr = (thresholds_rad if thresholds_rad is not None
           else _ATTACK_ALIGN_THRESHOLDS_RAD).to(dev)
    bv  = (bias_vals if bias_vals is not None
           else _ATTACK_ALIGN_BIASES).to(dev)
    bucket = torch.bucketize(min_angle, thr)                         # (B,)
    bias   = bv[bucket] * strength                                   # (B,)
    # Broadcast to attack_logit shape (B,) or (B,1).
    return attack_logit + bias.reshape(-1, *([1] * (attack_logit.dim() - 1)))


def policy_decode_action_postprocess(
    policy: Any,
    obs: Any,
    move_classes: torch.Tensor,
    fire: torch.Tensor,
) -> torch.Tensor:
    """QNNPolicy.act hook for a24rc1 Python eval/live paths."""
    if not isinstance(obs, Mapping):
        return fire
    obs_tensors = policy._obs_tensors_dequant(obs)
    return apply_attack_splash_guard(obs_tensors, move_classes, fire)


def guard_attack_logit_for_export(
    obs_tensors: Mapping[str, torch.Tensor],
    move_logits: torch.Tensor,
    attack_logit: torch.Tensor,
    attack_align_strength: float = 0.0,
) -> torch.Tensor:
    """ONNX export helper: apply alignment bias then force guarded logits below threshold."""
    move_flat = move_logits.reshape(-1, MOVE_AXES, move_logits.shape[-1])
    move_classes = move_flat.argmax(dim=-1)
    # Alignment-conditioned discrimination bias (before hard guards).
    if attack_align_strength > 0.0:
        attack_logit = attack_alignment_logit_bias(
            obs_tensors, attack_logit, strength=attack_align_strength)
    mask = attack_splash_guard_mask(obs_tensors, move_classes)
    mask = mask | rocket_self_splash_guard_mask(obs_tensors, move_classes)
    mask = mask | lg_range_guard_mask(obs_tensors, move_classes)
    return torch.where(
        mask.unsqueeze(-1), torch.full_like(attack_logit, -1.0e9), attack_logit)


# ── Generalized dodge ("any" mode) — per-projectile-subject physics, former a24rc2.
# Each observed projectile subject maps to the weapon that launched it; we take
# that weapon's splash + gravity. rocket (RL=7) straight 120u splash; grenade
# (GL=6) 800u/s² arc 120u splash; nail (NG=4) straight, no splash → direct-hit.
_PROJ_SUBJECTS: dict[int, int] = {
    ENTITY_IDS["PROJECTILE_ROCKET"]:  7,
    ENTITY_IDS["PROJECTILE_GRENADE"]: 6,
    ENTITY_IDS["PROJECTILE_NAIL"]:    4,
}
DODGE_HORIZON_S: float = 1.0
_DODGE_NSTEPS: int = 16
_VEL_T_FACTOR: float = en.MAX_VELOCITY / en.DIST_SCALE
_G_FACTOR: float = 0.5 / en.DIST_SCALE


def _subject_radius_u(weapon_id: int) -> float:
    splash = float(WEAPON_PHYSICS[weapon_id].get("splash", 0.0))
    return splash if splash > 0.0 else PROJECTILE_HIT_RADIUS_U


def _subject_gravity(weapon_id: int) -> float:
    return float(WEAPON_PHYSICS[weapon_id].get("gravity", 0.0))


def projectile_release_mask_any(
    obs_tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """(B,) rows where ANY inbound projectile is projected to land inside its
    splash radius within DODGE_HORIZON_S, if current movement continues — the
    generalized Gate B dodge-release trigger (former a24rc2 ``any`` mode).

    Trajectory sampled at ``_DODGE_NSTEPS`` over the horizon so the grenade arc
    uses the same path code as straight rockets/nails."""
    entity_types = obs_tensors.get("entity_types")
    entity_scalars = obs_tensors.get("entity_scalars_raw")
    entity_ids = obs_tensors.get("entity_ids")
    motion = obs_tensors.get("self_motion_scalars")
    present = [t for t in (entity_types, entity_scalars, entity_ids, motion) if t is not None]
    if len(present) < 4 or any(t.numel() == 0 for t in present):
        n = 1 if not present else present[0].reshape(present[0].shape[0], -1).shape[0]
        return torch.zeros(n, dtype=torch.bool,
                           device=(present[0].device if present else "cpu"))
    device = entity_scalars.device
    entity_types = entity_types.to(device=device).reshape(-1, entity_types.shape[-1])
    entity_scalars = entity_scalars.to(device=device).reshape(
        -1, entity_scalars.shape[-2], entity_scalars.shape[-1])
    entity_ids = entity_ids.to(device=device).reshape(-1, entity_ids.shape[-2], entity_ids.shape[-1])
    motion = motion.to(device=device).reshape(-1, motion.shape[-1])

    proj_rel = entity_scalars[:, :, _PROJ_REL_OFFSET:_PROJ_REL_OFFSET + 3]    # (B, N, 3)
    proj_vel = entity_scalars[:, :, _PROJ_VEL_OFFSET:_PROJ_VEL_OFFSET + 3]    # (B, N, 3)
    player_vel = motion[:, 0:3].unsqueeze(1)                                  # (B, 1, 3)
    rel_vel = proj_vel - player_vel                                           # (B, N, 3)

    subj = entity_ids[:, :, 0]                                                # (B, N)
    is_proj = (entity_types == TOKEN_PROJECTILE)                             # (B, N)
    known = torch.zeros_like(subj, dtype=torch.bool)
    rad2 = torch.full_like(subj, (PROJECTILE_HIT_RADIUS_U / en.DIST_SCALE) ** 2, dtype=torch.float32)
    grav = torch.zeros_like(subj, dtype=torch.float32)
    for sid, wid in _PROJ_SUBJECTS.items():
        m = (subj == sid)
        known = known | m
        rad2 = torch.where(m, torch.tensor((_subject_radius_u(wid) / en.DIST_SCALE) ** 2,
                                           dtype=torch.float32, device=device), rad2)
        grav = torch.where(m, torch.tensor(_subject_gravity(wid),
                                           dtype=torch.float32, device=device), grav)
    is_proj = is_proj & known

    ts = torch.linspace(DODGE_HORIZON_S / _DODGE_NSTEPS, DODGE_HORIZON_S, _DODGE_NSTEPS,
                        device=device, dtype=torch.float32)                   # (T,)
    t_disp = ts * _VEL_T_FACTOR                                               # (T,)
    pos = proj_rel.unsqueeze(2) + rel_vel.unsqueeze(2) * t_disp.view(1, 1, -1, 1)   # (B,N,T,3)
    z_drop = (grav.unsqueeze(-1) * (ts * ts).view(1, 1, -1) * _G_FACTOR)            # (B,N,T)
    zeros = torch.zeros_like(z_drop)
    drop3 = torch.stack([zeros, zeros, z_drop], dim=-1)                       # (B,N,T,3)
    pos = pos - drop3
    d2 = (pos * pos).sum(dim=-1)                                              # (B, N, T)
    min_d2 = d2.min(dim=2).values                                            # (B, N)
    threat = is_proj & (min_d2 < rad2)
    return threat.any(dim=1)                                                 # (B,)


def make_guard(params: Mapping[str, Any]) -> types.SimpleNamespace:
    """Bind a decode config's ``params`` to a guard adapter exposing the entry
    points the policy/export call. ``projectile_release_mask`` is set ONLY when
    the dodge-release is enabled (mode != "off"), so an off-mode export bakes no
    release nodes — matching the former a25rc1a graph. The attack guards are
    regime-invariant and always present.

    New decode params wired here:
      guard.attack_align_strength (float, default 0.0): human-distribution attack
        discrimination strength.  0 = flat threshold (current), 1 = full human
        p_attack-by-angle shape.  Applied as a logit bias before the threshold.
    """
    mode    = str(params.get("guard.projectile_release_mode", "rocket"))
    enabled = bool(params.get("guard.projectile_release", True)) and mode != "off"
    attack_align_strength = float(params.get("guard.attack_align_strength", 0.0))

    def _guard_attack_logit_for_export(obs_tensors, move_logits, attack_logit):
        return guard_attack_logit_for_export(
            obs_tensors, move_logits, attack_logit,
            attack_align_strength=attack_align_strength)

    def _preprocess_attack_logit(obs_tensors, attack_logit):
        return attack_alignment_logit_bias(
            obs_tensors, attack_logit, strength=attack_align_strength)

    adapter = types.SimpleNamespace(
        guard_attack_logit_for_export=_guard_attack_logit_for_export,
        policy_decode_action_postprocess=policy_decode_action_postprocess,
        apply_attack_splash_guard=apply_attack_splash_guard,
        attack_splash_guard_mask=attack_splash_guard_mask,
        rocket_self_splash_guard_mask=rocket_self_splash_guard_mask,
        lg_range_guard_mask=lg_range_guard_mask,
    )
    if attack_align_strength > 0.0:
        adapter.preprocess_attack_logit = _preprocess_attack_logit
    if enabled:
        adapter.projectile_release_mask = (
            projectile_release_mask_any if mode == "any"
            else projectile_release_mask_rocket)
    return adapter
