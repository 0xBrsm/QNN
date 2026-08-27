"""Closed-form per-entity lead-corrected aim points — a25-owned clone.

Cloned from the a24 lineage (``qnn.model.bench.a24.lead_aim``) so the a25 arch
executes its OWN aim-prior geometry and never imports/executes a24 code
(cross-arch decode-coupling ban). This is a BIT-IDENTICAL clone of the reachable
a24 aim-prior path: :func:`aim_prior_tangent_ffwd` + its dependencies
(:func:`compute_lead_aim`, :func:`weapon_trajectory`, :func:`pooled_aim_vec`)
and the decode-contract constants (``AIM_PRIOR_GAIN``, ``AIM_FFWD_GAIN``,
``AIM_Z_DROP``, ``_TICK_DT_MODULE``). The a24 ``aim_prior_tangent`` (non-ffwd)
variant is NOT cloned — the a25 policy/export only call the ``_ffwd`` form.

For each entity (target candidate), compute "where would I have to aim to hit
this entity with the attack-with choice's trajectory?" The single formula
handles all three weapon classes by parameterization:

* Hitscan: ``v_horiz ≈ QNN_VEL_SCALE`` (sentinel max). Lead time collapses to ≈ 0
  and the aim point collapses to the entity's current position.
* Projectile: ``v_horiz < QNN_VEL_SCALE``. Lead time is the smallest positive
  root of the standard intercept quadratic.

The z-axis then applies a per-weapon EMPIRICAL anchor ``−(A + B·t)``
(``AIM_Z_DROP``) fitted to where demonstrators actually put the crosshair
relative to the lead point.

Closed form, fully differentiable, no model parameters. Consumed by the a25 look
decode (aim_vec prior — soft-pooled by the target pointer's softmax, never
argmaxed) in ``policy.py`` and the ONNX ``ExportWrapper``.

Implementation notes:

- All comparisons and selections done via arithmetic masking
  (``(x < y).to(dtype)``) rather than ``torch.where``. Nested ``torch.where``
  chains on bf16 + non-contiguous inputs hung ROCm on the real BC training path.
- ``torch.no_grad()`` decorator on ``_impl`` — the inputs are obs scalars and a
  static weapon-trajectory table, neither trainable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qnn.bc.weapon_physics import (
    ACTOR_REL_OFFSET as _ACTOR_REL_OFFSET,
    ACTOR_VEL_OFFSET as _ACTOR_VEL_OFFSET,
    ACTOR_TEAM_OFFSET as _ACTOR_TEAM_OFFSET,
    TEAM_TEAMMATE_VALUE as _TEAM_TEAMMATE_VALUE,
)

# Origin→feet height of the QW player hull (VEC_HULL_MIN.z = −24u), in /DIST_SCALE.
# The RL/NG/SNG z-drop is clamped to never exceed this so the aim never points below
# the target's feet (a straight rocket aimed below-feet hits the floor short — the
# "can only shoot the ground past a max range" bug). See compute_lead_aim.
_AIM_Z_FEET_CAP: float = 24.0 / 1000.0
# Shooter MUZZLE height above its own origin, in /DIST_SCALE. entity_rel is
# ORIGIN-to-origin but every direct-fire weapon spawns/traces from origin + '0 0 16'.
# NOT the eye (view_ofs '0 0 22'): the eye is only the camera; projectiles leave from
# the +16 muzzle. Without this the aim vector is built as if fired from the origin, so
# a shot aimed at the target's feet (−24u) rides 16u ABOVE the feet at range →
# center-mass. Subtracting the muzzle height puts the aim on the true muzzle→feet line.
_AIM_MUZZLE_OFS: float = 16.0 / 1000.0


def compute_lead_aim(
    entity_rel: torch.Tensor,        # (B, N, 3) view-frame XYZ, /DIST_SCALE
    entity_vel: torch.Tensor,        # (B, N, 3) view-frame relative vel, /VEL_SCALE
    v_horiz: torch.Tensor,           # (B,) projectile speed (/VEL_SCALE)
    drop_const: torch.Tensor,        # (B,) aim-z anchor A (/DIST_SCALE)
    drop_rate: torch.Tensor,         # (B,) aim-z anchor B (/DIST_SCALE per s)
    lead_hold_cap: float | None = None,         # TANGENTIAL (strafe) hold cap (module units); None = OFF
    lead_hold_cap_radial: float | None = None,  # RADIAL (approach/retreat) hold cap (module units); None = OFF
) -> torch.Tensor:
    """Per-entity lead-corrected aim points ``(B, N, 3)``.

    The aim point is the target's predicted position at the projectile intercept
    horizon ``t_projectile`` (the quadratic solve below; ~0 for hitscan via the
    sentinel). The ballistic Z-drop is applied over that same flight time.

    HAZARD-DISCOUNTED LEAD (both default None = OFF = bit-identical linear lead).
    Linear ``v·t_lead`` assumes the target holds its motion for the full projectile
    flight, but the human movement dwell is log-normal with a short hold, so linear
    OVER-leads. The cap clamps the displacement horizon at the expected hold, applied
    PER-AXIS in the LOS frame: RADIAL (approach/retreat) and TANGENTIAL (strafe)
    components each capped at their OWN combat dwell; the ballistic Z-drop always uses
    the true flight time ``t_lead``."""
    return _compute_lead_aim_impl(
        entity_rel, entity_vel, v_horiz, drop_const, drop_rate,
        lead_hold_cap, lead_hold_cap_radial)


@torch.no_grad()
def _compute_lead_aim_impl(
    entity_rel: torch.Tensor,
    entity_vel: torch.Tensor,
    v_horiz: torch.Tensor,
    drop_const: torch.Tensor,
    drop_rate: torch.Tensor,
    lead_hold_cap: float | None = None,
    lead_hold_cap_radial: float | None = None,
) -> torch.Tensor:
    # Broadcast weapon scalars to (B, 1).
    v_h = v_horiz.unsqueeze(-1)                                  # (B, 1)

    # Quadratic coefficients: (|v|² - v_h²) t² + 2(r·v) t + r·r = 0
    r_sq = (entity_rel * entity_rel).sum(dim=-1)                 # (B, N)
    v_sq = (entity_vel * entity_vel).sum(dim=-1)                 # (B, N)
    rv   = (entity_rel * entity_vel).sum(dim=-1)                 # (B, N)
    a = v_sq - v_h * v_h                                         # (B, N)
    b = 2.0 * rv                                                 # (B, N)
    c = r_sq                                                     # (B, N)

    disc = b * b - 4.0 * a * c                                   # (B, N)
    reachable_f = (disc >= 0).to(disc.dtype)                     # (B, N) 0/1
    sqrt_disc = torch.sqrt(disc.clamp(min=0.0))

    # a_safe = a if |a| >= eps else sign(a) * eps. Avoid div-by-zero.
    eps = 1e-6
    small_a_f = (a.abs() < eps).to(a.dtype)
    a_sign = a.sign() + (a == 0).to(a.dtype)
    a_safe = a * (1.0 - small_a_f) + small_a_f * a_sign * eps

    t_plus  = (-b + sqrt_disc) / (2.0 * a_safe)
    t_minus = (-b - sqrt_disc) / (2.0 * a_safe)

    # Smallest non-negative root. Arithmetic mask: replace negative
    # candidates with 1e9 so torch.minimum picks the valid one.
    big = 1e9
    pos_p = (t_plus  >= 0).to(t_plus.dtype)
    pos_m = (t_minus >= 0).to(t_minus.dtype)
    cand_p = t_plus  * pos_p + big * (1.0 - pos_p)
    cand_m = t_minus * pos_m + big * (1.0 - pos_m)
    t_lead = torch.minimum(cand_p, cand_m)                       # (B, N)

    valid_f = reachable_f * (t_lead < big).to(t_lead.dtype)
    t_lead = t_lead * valid_f                                    # t_projectile (B, N)

    # Prediction horizon: aim at the target's position at the projectile intercept
    # t_lead; entity_vel is RELATIVE, so own strafe is already covered.
    #
    # Hazard-discounted lead (PER-AXIS): cap each LOS-frame velocity component's
    # lead horizon at its own combat dwell. Default OFF (both None) → full t_lead,
    # bit-identical. The Z-drop always keeps the FULL t_lead (true flight time).
    _tan_on = lead_hold_cap is not None and lead_hold_cap > 0.0
    _rad_on = lead_hold_cap_radial is not None and lead_hold_cap_radial > 0.0
    if _tan_on or _rad_on:
        rel_hat = entity_rel / torch.linalg.vector_norm(
            entity_rel, dim=-1, keepdim=True).clamp_min(1e-6)        # (B, N, 3)
        v_radial = (entity_vel * rel_hat).sum(-1, keepdim=True) * rel_hat   # (B,N,3)
        v_tang = entity_vel - v_radial                              # (B, N, 3)
        t_tan = (torch.clamp(t_lead, max=float(lead_hold_cap)) if _tan_on else t_lead)
        t_rad = (torch.clamp(t_lead, max=float(lead_hold_cap_radial)) if _rad_on else t_lead)
        disp = (v_radial * t_rad.unsqueeze(-1) + v_tang * t_tan.unsqueeze(-1))
    else:
        disp = entity_vel * t_lead.unsqueeze(-1)
    aim_point = entity_rel + disp                                  # (B, N, 3)

    # Z-axis anchor — the per-weapon empirical body/ground offset, linear in flight
    # time: aim ``A + B·t_seconds`` BELOW the lead point. t_lead is in module units
    # (t_seconds · VEL_SCALE/DIST_SCALE = 2·t_seconds), so the per-second rate is
    # halved here.
    net_z = -(drop_const.unsqueeze(-1)
              + 0.5 * drop_rate.unsqueeze(-1) * t_lead)          # (B, N)
    # FEET CAP — never aim BELOW the target's feet. Clamp the drop at the origin→feet
    # height so the aim lands AT the feet (ground under the target = splash).
    net_z = net_z.clamp(min=-_AIM_Z_FEET_CAP)
    # MUZZLE FRAME — shift the aim point down by the shooter's muzzle height so the
    # aim vector is measured from the +16 muzzle. Combined with the feet-cap the total
    # floor is −(24+16)=−40u = the true muzzle→feet vertical.
    # cat instead of in-place index_put — keeps the ONNX trace clean.
    return torch.cat(
        [aim_point[..., :2],
         (aim_point[..., 2] + net_z - _AIM_MUZZLE_OFS).unsqueeze(-1)], dim=-1)


def weapon_trajectory(
    weapon_physics: torch.Tensor,    # (9, 7) from build_model_weapon_scalars
    weapon_impulse: torch.Tensor,    # (B,) impulse-indexed long, 0..8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Look up per-batch trajectory terms for an attack-with impulse."""
    idx = weapon_impulse.long().clamp(0, weapon_physics.shape[0] - 1)
    table = weapon_physics[idx]                                  # (B, 7)
    v_horiz = table[:, 2]                                        # WT_V_HORIZ
    # Hitscan rows carry the sentinel v_horiz == 1.0 (2000 u/s); boost it 100× so the
    # intercept collapses to the bearing (arithmetic mask, ROCm-safe).
    hitscan_f = (v_horiz >= 0.999).to(v_horiz.dtype)
    v_horiz = v_horiz * (1.0 + 99.0 * hitscan_f)
    drop = weapon_physics.new_tensor(AIM_Z_DROP)[idx]            # (B, 2)
    return v_horiz, drop[:, 0], drop[:, 1]


# Decode-contract constant: the aim-prior blend gain for pointer-bearing models,
# z_out = z_hybrid + GAIN · z_aim. (See src/docs/look-head.md; a24-fit default.)
AIM_PRIOR_GAIN: float = 0.015

# Feed-forward gain on the aim point's per-tick angular MOTION. The error-gain blend
# is a P-controller: against a constantly strafing target it settles at a steady-state
# TRAIL ∝ target angular rate / gain. The rate term moves the look WITH the pooled aim
# point each tick (1.0 = exact rate match; entity_vel is already RELATIVE).
AIM_FFWD_GAIN: float = 1.0

# Impulse-indexed aim-z anchor (A, B): aim ``A + B·t_flight`` BELOW the lead point.
# A in /DIST_SCALE, B in /DIST_SCALE per second. Fitted to the corpus (RL ground
# splash anchor, NG/SNG body aim with range tilt, GL/hitscan 0).
#                 none      axe       sg        ssg       ng
AIM_Z_DROP = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.002, 0.035),
              # sng         gl          rl              lg
              (0.002, 0.035), (0.0, 0.0), (0.021, 0.033), (0.0, 0.0))

# One engine tick (20 Hz contract) in this module's time unit: the intercept algebra
# runs on rel/DIST_SCALE and vel/VEL_SCALE, so t_module = t_seconds · (VEL_SCALE /
# DIST_SCALE) = 0.05 · 2 = 0.1. Sourced from the aim kernel that owns those scales.
from qnn.eval.aim_kernel import TICK_DT_MODULE as _TICK_DT_MODULE  # noqa: E402


def pooled_aim_vec(
    per_entity_aim: torch.Tensor,    # (B, N, 3)
    target_logits: torch.Tensor,     # (B, N) — pre-masked, -1e9 on non-candidates
    candidate_mask: torch.Tensor,    # (B, N) bool — gate: zero vec when none
) -> torch.Tensor:
    """Soft-pool per-entity aim by target softmax → unit aim_vec.

    Returns the zero vector on frames with no candidate; callers gate on the norm.
    """
    probs = F.softmax(target_logits, dim=-1)
    pooled = (probs.unsqueeze(-1) * per_entity_aim).sum(dim=-2)  # (B, 3)
    has_candidate = candidate_mask.any(dim=-1, keepdim=True).to(pooled.dtype)
    pooled = pooled * has_candidate
    norm = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True).clamp(min=1e-6)
    return pooled / norm * has_candidate


def aim_prior_tangent_ffwd(
    entity_scalars_raw: torch.Tensor,
    entity_types: torch.Tensor,
    weapon_impulse: torch.Tensor,
    target_logits: torch.Tensor,
    weapon_physics: torch.Tensor,
    tick_dt: float = _TICK_DT_MODULE,
    lead_hold_cap: float | None = None,
    lead_hold_cap_radial: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(z_aim, z_rate, feet_elev, origin_elev, aim_range)`` — the error tangent,
    the aim point's per-tick angular displacement (feed-forward term), the pooled
    feet/origin anchor ELEVATIONS (signed, − = below crosshair), and the pooled
    target RANGE. ``feet_elev`` is the true elevation of the feet aim point (atan2
    of its up vs horizontal), used to LOCK the RL fired pitch onto the ground at
    the target's feet after the turn. z_rate is the tangent delta between the
    pooled aim point now and one tick ahead. ``aim_range`` is the pooled lead
    point's distance in MODULE units (/DIST_SCALE) — the per-discharge geometry
    the crest gate's hbw ruler consumes (hbw = |z| / atan(halfw/range), the
    eval's _intercept_hbw law). All are zero rows when no enemy is perceived.

    ``tick_dt`` is the per-decision-tick step in this module's time unit; it MUST
    scale with the model's decision cadence. Defaults to the 20 Hz value."""
    from qnn.model.look_bins import tangent_logmap
    from qnn.vocab import TOKEN_ACTOR

    rel = entity_scalars_raw[..., _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET + 3]
    vel = entity_scalars_raw[..., _ACTOR_VEL_OFFSET:_ACTOR_VEL_OFFSET + 3]
    team = entity_scalars_raw[..., _ACTOR_TEAM_OFFSET]
    enemy = (entity_types.long() == TOKEN_ACTOR) & (team != _TEAM_TEAMMATE_VALUE)
    imp = weapon_impulse.reshape(-1).long().clamp(0, 8)
    v_horiz, drop_const, drop_rate = weapon_trajectory(weapon_physics, imp)
    # The per-weapon aim-z anchor (AIM_Z_DROP) is applied inside compute_lead_aim,
    # i.e. before pooling — so it rides the error term AND the ffwd rate term.
    aim_pts = compute_lead_aim(
        rel.float(), vel.float(), v_horiz, drop_const, drop_rate,
        lead_hold_cap, lead_hold_cap_radial)
    aim_u = pooled_aim_vec(aim_pts, target_logits.float(), enemy)   # (R, 3) unit (view frame)
    norm = aim_u.norm(dim=-1, keepdim=True)
    gate = (norm > 1e-6)
    z = tangent_logmap(aim_u / norm.clamp_min(1e-6)) * gate
    # Pooled target RANGE (module units): the norm of the pooled lead point
    # BEFORE normalization — pooled_aim_vec discards it, so re-pool here (same
    # softmax/candidate gating, additive, trace-safe). Zero on no-enemy rows.
    _probs = F.softmax(target_logits.float(), dim=-1)
    _pooled_pt = (_probs.unsqueeze(-1) * aim_pts).sum(dim=-2)        # (R, 3)
    aim_range = (torch.linalg.vector_norm(_pooled_pt, dim=-1)
                 * enemy.any(dim=-1).to(aim_pts.dtype))
    # TRUE elevation of the feet aim point (fwd=+x, up=+z; − = below crosshair). Zero
    # rows on no-enemy frames (aim_u zeroed) so the decode's gate leaves them untouched.
    _feh = torch.linalg.vector_norm(aim_u[..., :2], dim=-1)
    feet_elev = torch.atan2(aim_u[..., 2], _feh.clamp_min(1e-9)) * gate.squeeze(-1)
    # ORIGIN (center-mass) anchor elevation — same lead point with the z-drop ZEROED.
    # feet_below_origin = origin_elev − feet_elev is the center-mass→feet angle used by
    # the spread-preserving pitch SHIFT. Zero rows on no-enemy frames.
    aim_pts_o = compute_lead_aim(
        rel.float(), vel.float(), v_horiz,
        torch.zeros_like(drop_const), torch.zeros_like(drop_rate),
        lead_hold_cap, lead_hold_cap_radial)
    aim_o = pooled_aim_vec(aim_pts_o, target_logits.float(), enemy)
    _oeh = torch.linalg.vector_norm(aim_o[..., :2], dim=-1)
    origin_elev = torch.atan2(aim_o[..., 2], _oeh.clamp_min(1e-9)) * (
        aim_o.norm(dim=-1) > 1e-6)

    # one tick ahead: the lead point rides the (relative) target velocity
    next_u = pooled_aim_vec(
        aim_pts + vel.float() * tick_dt, target_logits.float(), enemy)
    nnorm = next_u.norm(dim=-1, keepdim=True)
    z_next = tangent_logmap(next_u / nnorm.clamp_min(1e-6)) * (nnorm > 1e-6)
    z_rate = (z_next - z) * gate
    return z, z_rate, feet_elev, origin_elev, aim_range
