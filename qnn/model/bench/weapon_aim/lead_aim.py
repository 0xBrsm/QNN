"""Closed-form per-entity lead-corrected aim points.

For each entity (target candidate), compute "where would I have to aim
to hit this entity given my held weapon's trajectory?" The single
formula handles all three weapon classes by parameterization:

* Hitscan: ``v_horiz ≈ QNN_VEL_SCALE`` (sentinel max), ``gravity = 0``.
  Lead time collapses to ≈ 0 and the aim point collapses to the
  entity's current position.
* Linear projectile: ``v_horiz < QNN_VEL_SCALE``, ``gravity = 0``. Lead
  time is the smallest positive root of the standard intercept
  quadratic.
* Ballistic (GL): ``gravity > 0``. Lead time solved with linear
  approximation; ``½·g·t²`` added to the view-frame z component.

Closed form, fully differentiable, no model parameters.

Implementation notes:

- All comparisons and selections done via arithmetic masking
  (``(x < y).to(dtype)``) rather than ``torch.where``. Nested
  ``torch.where`` chains on bf16 + non-contiguous inputs hung ROCm
  on the real BC training path.
- ``torch.no_grad()`` decorator on ``_impl`` — the inputs are obs
  scalars and a static weapon-trajectory table, neither trainable, so
  there's nothing useful to differentiate through.
"""

from __future__ import annotations

import torch


def compute_lead_aim(
    entity_rel: torch.Tensor,        # (B, N, 3) view-frame XYZ, /DIST_SCALE
    entity_vel: torch.Tensor,        # (B, N, 3) view-frame relative vel, /VEL_SCALE
    v_horiz: torch.Tensor,           # (B,) projectile speed (/VEL_SCALE)
    gravity: torch.Tensor,           # (B,) downward acceleration (/VEL_SCALE)
) -> torch.Tensor:
    """Per-entity lead-corrected aim points ``(B, N, 3)``."""
    return _compute_lead_aim_impl(entity_rel, entity_vel, v_horiz, gravity)


@torch.no_grad()
def _compute_lead_aim_impl(
    entity_rel: torch.Tensor,
    entity_vel: torch.Tensor,
    v_horiz: torch.Tensor,
    gravity: torch.Tensor,
) -> torch.Tensor:
    # Broadcast weapon scalars to (B, 1).
    v_h = v_horiz.unsqueeze(-1)                                  # (B, 1)
    g   = gravity.unsqueeze(-1)                                  # (B, 1)

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
    t_lead = t_lead * valid_f

    # Lead point: r + v·t (view-frame, all axes).
    aim_point = entity_rel + entity_vel * t_lead.unsqueeze(-1)   # (B, N, 3)

    # Gravity compensation on view-frame z-axis (only nonzero for GL).
    # Inputs are normalized by DIST_SCALE=1000 / VEL_SCALE=2000, so the
    # closed-form solver returns t in units of (VEL_SCALE/DIST_SCALE) = 2
    # raw seconds. The linear ``entity_vel * t_lead`` term is self-correcting
    # (the 2× absorbs into VEL_SCALE-normalized v). The t² gravity term is
    # not: ``0.5·g·t²`` with normalized g and t-prime gives 2× the correct
    # DIST_SCALE-normalized drop. Compensate with 0.25 (= 0.5 ÷
    # (VEL_SCALE/DIST_SCALE)) so the drop lands in DIST_SCALE units.
    z_compensate = 0.25 * g * t_lead * t_lead                    # (B, N)
    # In-place z-axis add — avoids materializing a (B, N, 2) zero pad and the
    # subsequent cat allocation.
    aim_point = aim_point.clone()
    aim_point[..., 2] = aim_point[..., 2] + z_compensate
    return aim_point


def held_weapon_trajectory(
    weapon_static: torch.Tensor,     # (9, 7) from build_model_weapon_scalars
    weapon_id: torch.Tensor,         # (B,) impulse-indexed long, 0..8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up per-batch (v_horiz, gravity) for the held weapon."""
    idx = weapon_id.long().clamp(0, weapon_static.shape[0] - 1)
    table = weapon_static[idx]                                   # (B, 7)
    v_horiz = table[:, 2]                                        # (B,)
    gravity = table[:, 4]                                        # (B,)
    return v_horiz, gravity


def pooled_aim_vec(
    per_entity_aim: torch.Tensor,    # (B, N, 3)
    target_logits: torch.Tensor,     # (B, N) — pre-masked, -1e9 on non-actor
    actor_mask: torch.Tensor,        # (B, N) bool
) -> torch.Tensor:
    """Soft-pool per-entity aim by target softmax → unit aim_vec."""
    import torch.nn.functional as F
    probs = F.softmax(target_logits, dim=-1)
    pooled = (probs.unsqueeze(-1) * per_entity_aim).sum(dim=-2)  # (B, 3)
    has_actor = actor_mask.any(dim=-1, keepdim=True).to(pooled.dtype)
    pooled = pooled * has_actor
    norm = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True).clamp(min=1e-6)
    return pooled / norm
