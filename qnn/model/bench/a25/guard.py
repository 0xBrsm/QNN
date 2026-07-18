"""The a25 decode-time guard set — a25-owned, NARROWED clone.

Cloned from the a24 lineage (``qnn.model.bench.a24.guard``) so the a25 arch
executes its OWN guard vetoes and never imports/executes a24 code (cross-arch
decode-coupling ban). The a25 guard set is deliberately NARROWER than a24 — it
keeps ONLY the two vetoes the a25 seg+attack_with operating point uses:

  1. RL EHR self-splash veto (:func:`rocket_self_splash_guard_mask`) — "don't
     rocket yourself to death": suppress an RL attack whose self-splash would be
     lethal given the current effective health.
  2. Gate B projectile move-hold release (:func:`projectile_release_mask_rocket`
     / :func:`projectile_release_mask_any`, mode-selected) — force an fb/lr hold
     release when an inbound projectile is projected to hit if movement continues.

DROPPED from the a24 set (NOT cloned — the a25 operating point does not use them):
  * ``attack_splash_guard_mask`` (square-wall waste veto),
  * ``lg_range_guard_mask`` (LG beam range/cone veto),
  * ``attack_alignment_logit_bias`` / ``attack_align_strength`` (angle-conditioned
    fire discrimination), and ``_actor_min_angle_rad``.

The kept functions are wired into the :func:`make_guard` adapter the a25 policy /
ONNX export consume. The self-splash veto is the ONLY OR-in in
:func:`guard_attack_logit_for_export`; the a25 ``attack_with_decode`` reads that
veto via its zeros-probe (align_bias is then always zero for a25). BIT-IDENTICAL
to the a24 originals for the two kept vetoes.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any

import torch

from qnn.actions import MOVE_AXES
from qnn import engine_norm as en
from qnn.bc.weapon_physics import WEAPON_PHYSICS
from qnn.vocab import ENTITY_IDS, SPATIAL_SECTOR_IDS, TOKEN_ACTOR, TOKEN_PROJECTILE


# Spatial sector / column indices (dequantized spatial_scalars layout).
_FOV_CENTER_IDX: int = SPATIAL_SECTOR_IDS["FOV_Center"]
_GROUND_SECTOR_IDX: int = SPATIAL_SECTOR_IDS["Ground_State"]
_SP_NEAREST_DIST: int = 3
_ACTOR_DIST: int = 6
_ACTOR_REL_FWD: int = 3   # ACTOR_REL_OFFSET; view-frame rel x = forward component

# ── Gate B: inbound-projectile move-hold release trigger ─────────────────────
# Release the move hold when a perceived projectile is projected to strike the player
# within the horizon *if current movement continues*. Straight-line closest-approach
# relative to the player (relative velocity = proj_vel - player_vel, both view-frame).
_ROCKET_ID: int = ENTITY_IDS["PROJECTILE_ROCKET"]
# Projectile token layout in entity_scalars_raw (dequant): rel[0:3], dist[3],
# vel[4:7], recency[7].
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
    """(B,) rows where a rocket is projected to hit the player within the horizon if
    current movement continues — the Gate B move-hold release trigger (straight-rocket
    closed-form closest-approach)."""
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


# ── Generalized dodge ("any" mode) — per-projectile-subject physics.
# Each observed projectile subject maps to the weapon that launched it; we take that
# weapon's splash + gravity. rocket (RL=7) straight 120u splash; grenade (GL=6)
# 800u/s² arc 120u splash; nail (NG=4) straight, no splash → direct-hit.
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
    """(B,) rows where ANY inbound projectile is projected to land inside its splash
    radius within DODGE_HORIZON_S, if current movement continues — the generalized Gate
    B dodge-release trigger. Trajectory sampled at ``_DODGE_NSTEPS`` over the horizon so
    the grenade arc uses the same path code as straight rockets/nails."""
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


# ── RL EHR-scaled self-splash veto ───────────────────────────────────────────
# Rocket radius damage 120; findradius(damage+40)=160u; attacker branch halves
# self-damage → self-splash at surface distance d is 0.5*(120 - 0.5*d) for d < 160u.
# Max self-splash = 60 at d=0. Lethal standoff: d_safe = 4*(60 - EHR + margin), clamped
# to [0, 160]. EHR = health + effective_armor (HP).
_RL_LAUNCHER_ID: int = ENTITY_IDS["ROCKET_LAUNCHER"]
_RL_SPLASH_MAX_SELF: float = 60.0     # 0.5 * 120 (point-blank attacker damage)
_RL_SPLASH_FINDRADIUS: float = 160.0  # damage + 40; splash = 0 beyond this
ROCKET_EHR_SAFETY_MARGIN_HP: float = 10.0   # hold fire if it would leave < this HP
ROCKET_FLOOR_MIN_PITCH_DEG: float = 12.0    # below this downward pitch, skip the floor test
_DEG2RAD: float = 3.141592653589793 / 180.0


def rocket_self_splash_guard_mask(
    obs_tensors: Mapping[str, torch.Tensor],
    move_classes: torch.Tensor,
) -> torch.Tensor:
    """Rows where an RL shot's self-splash would be lethal given EHR.

    RL only. Suppressed when the shot detonates within the EHR-scaled safe standoff
    ``d_safe = 4*(60 - EHR + margin)`` (clamped [0, 160] u) on a horizontal wall
    (FOV_Center nearest), the floor (Ground-sector slant ``ground / sin(pitch)``), or a
    point-blank enemy (nearest actor in the forward hemisphere).

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
    # Slant distance to the floor impact; +inf where not aiming down.
    floor_slant = ground / sin_pitch.clamp(min=1e-3)
    inf = torch.full_like(floor_slant, float("inf"))
    floor_slant = torch.where(aiming_down, floor_slant, inf)

    # Nearest perceived actor in the forward hemisphere (rel-forward > 0).
    actor_fwd = entity_scalars[:, :, _ACTOR_REL_FWD]                          # (B, N)
    actor_mask = (entity_types == TOKEN_ACTOR) & (actor_fwd > 0.0)
    actor_dist = entity_scalars[:, :, _ACTOR_DIST]                            # (B, N)
    inf_a = torch.full_like(actor_dist, float("inf"))
    nearest_actor = torch.where(actor_mask, actor_dist, inf_a).amin(dim=1)    # (B,)

    surface = torch.minimum(torch.minimum(wall, floor_slant), nearest_actor)  # (B,)
    held_rl = weapon_id == _RL_LAUNCHER_ID
    return held_rl & (surface < d_safe)


def apply_self_splash_guard(
    obs_tensors: Mapping[str, torch.Tensor],
    move_classes: torch.Tensor,
    fire: torch.Tensor,
) -> torch.Tensor:
    """Apply ONLY the RL self-splash veto to a decoded fire tensor (a25 guard set)."""
    mask = rocket_self_splash_guard_mask(obs_tensors, move_classes)
    return torch.where(mask, torch.zeros_like(fire), fire)


def policy_decode_action_postprocess(
    policy: Any,
    obs: Any,
    move_classes: torch.Tensor,
    fire: torch.Tensor,
) -> torch.Tensor:
    """QNNPolicy.act hook (guard contract entry point). a25 applies ONLY the RL
    self-splash veto (the attack_with path does not call this, but the guard_module
    contract requires it to be present)."""
    if not isinstance(obs, Mapping):
        return fire
    obs_tensors = policy._obs_tensors_dequant(obs)
    return apply_self_splash_guard(obs_tensors, move_classes, fire)


def guard_attack_logit_for_export(
    obs_tensors: Mapping[str, torch.Tensor],
    move_logits: torch.Tensor,
    attack_logit: torch.Tensor,
) -> torch.Tensor:
    """ONNX export / attack-with helper: force RL self-splash rows' logit to -1e9.

    The a25 attack_with decode probes this on a zeros logit to recover a hard-veto
    mask; there is no alignment bias in the a25 guard set, so non-veto rows return the
    input logit unchanged (align_bias resolves to zero)."""
    move_flat = move_logits.reshape(-1, MOVE_AXES, move_logits.shape[-1])
    move_classes = move_flat.argmax(dim=-1)
    mask = rocket_self_splash_guard_mask(obs_tensors, move_classes)
    return torch.where(
        mask.unsqueeze(-1), torch.full_like(attack_logit, -1.0e9), attack_logit)


def make_guard(params: Mapping[str, Any]) -> types.SimpleNamespace:
    """Bind a decode config's ``params`` to the a25 guard adapter.

    Keeps the Gate B ``projectile_release_mode`` selection ("off"/"rocket"/"any") and
    the always-present RL self-splash veto. Drops the a24 attack-splash / LG-range /
    alignment-bias wiring (not in the a25 guard set). ``projectile_release_mask`` is set
    ONLY when the dodge-release is enabled (mode != "off")."""
    mode    = str(params.get("guard.projectile_release_mode", "rocket"))
    enabled = bool(params.get("guard.projectile_release", True)) and mode != "off"

    adapter = types.SimpleNamespace(
        guard_attack_logit_for_export=guard_attack_logit_for_export,
        policy_decode_action_postprocess=policy_decode_action_postprocess,
        apply_self_splash_guard=apply_self_splash_guard,
        rocket_self_splash_guard_mask=rocket_self_splash_guard_mask,
    )
    if enabled:
        adapter.projectile_release_mask = (
            projectile_release_mask_any if mode == "any"
            else projectile_release_mask_rocket)
    return adapter
