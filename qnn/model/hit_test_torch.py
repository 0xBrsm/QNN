"""GPU-vectorized hit_test for the BC training loop.

Mirrors ``qnn_hit_test.c`` / ``qnn.bc.weapon_physics`` algorithms in
torch so the model can compute per-slot hit_test on the GPU without
per-batch ctypes round-trips. Tied to the C version by a parity test
in tests/test_hit_test_torch_vs_c.py.

The C function is the long-term source of truth for the labeler and
live play; this module exists because the BC training loop forwards
batches of (T, B, N, ...) tensors that already live on the GPU and
calling out to C per slot per frame per batch would dominate
wall-time. The torch op runs in one fused pass over the batch.

Inputs are in **view frame**, where the player's aim is implicitly
``(1, 0, 0)`` (+X = forward). Caller does not need to pass look —
the function bakes that in. All distance/velocity values are expected
in obs-buffer scale (rel · QNN_DIST_SCALE = world units, vel ·
QNN_VEL_SCALE = world units/sec, half · QNN_DIST_SCALE = world units).
"""
from __future__ import annotations

import math

import torch

# Scale constants (must match weapon_physics.py + qnn_hit_test.c).
_QNN_DIST_SCALE = 1000.0
_QNN_VEL_SCALE = 2000.0

# Per-weapon physics. Index = impulse id (0..8). Matches WEAPON_PHYSICS
# in weapon_physics.py and the static table in qnn_hit_test.c.
# Tuple layout: (hitscan, range, spread_deg, speed, splash, max_t, gravity).
_WEAPON_PHYSICS = [
    (False, 0.0,    0.0, 0.0,    0.0,   0.0, 0.0),  # 0 none
    (True,  64.0,   0.0, 0.0,    0.0,   0.0, 0.0),  # 1 axe
    (True,  1500.0, 2.3, 0.0,    0.0,   0.0, 0.0),  # 2 sg
    (True,  1500.0, 8.0, 0.0,    0.0,   0.0, 0.0),  # 3 ssg
    (False, 0.0,    0.0, 1000.0, 0.0,   3.0, 0.0),  # 4 ng
    (False, 0.0,    0.0, 1000.0, 0.0,   3.0, 0.0),  # 5 sng
    (False, 0.0,    0.0, 600.0,  120.0, 2.5, 800.0),  # 6 gl
    (False, 0.0,    0.0, 1000.0, 120.0, 4.0, 0.0),  # 7 rl
    (True,  600.0,  0.0, 0.0,    0.0,   0.0, 0.0),  # 8 lg
]


def _build_weapon_luts(device: torch.device) -> dict[str, torch.Tensor]:
    """Per-weapon constant tensors indexed by impulse id."""
    n = len(_WEAPON_PHYSICS)
    cols = list(zip(*_WEAPON_PHYSICS))
    hitscan = torch.tensor(cols[0], dtype=torch.bool, device=device)
    rng = torch.tensor(cols[1], dtype=torch.float32, device=device)
    spread = torch.tensor(cols[2], dtype=torch.float32, device=device) * (math.pi / 180.0)
    speed = torch.tensor(cols[3], dtype=torch.float32, device=device)
    splash = torch.tensor(cols[4], dtype=torch.float32, device=device)
    max_t = torch.tensor(cols[5], dtype=torch.float32, device=device)
    gravity = torch.tensor(cols[6], dtype=torch.float32, device=device)
    return dict(
        hitscan=hitscan, range=rng, spread=spread, speed=speed,
        splash=splash, max_t=max_t, gravity=gravity, n=n,
    )


def _entity_ids_to_impulse(weapon_id: torch.Tensor) -> torch.Tensor:
    """Map ENTITY_IDS weapon id (NONE=0, AXE=3, ..., LG=10) to impulse
    (NONE=0, AXE=1, ..., LG=8)."""
    # impulse = weapon_id - 2 for weapons (3..10); 0 stays 0.
    impulse = torch.clamp(weapon_id - 2, min=0).long()
    impulse = torch.where(weapon_id == 0, torch.zeros_like(impulse), impulse)
    return impulse


def hit_test_torch(
    weapon_id_entity: torch.Tensor,   # (B,)    ENTITY_IDS-encoded
    entity_rel: torch.Tensor,         # (B, N, 3) view-frame, obs-scaled
    entity_vel: torch.Tensor,         # (B, N, 3) view-frame, obs-scaled
    entity_half: torch.Tensor,        # (B, N, 3) world / DIST_SCALE
    actor_mask: torch.Tensor,         # (B, N) bool — non-actors → False
) -> torch.Tensor:
    """Per-slot hit_test as (B, N) bool tensor.

    Look direction is implicit (1, 0, 0) — player's forward in view
    frame. Non-actor slots and slots with weapon_id=0 return False.
    """
    device = entity_rel.device
    B, N = entity_rel.shape[:2]
    luts = _build_weapon_luts(device)

    # Un-scale to world units.
    rel = entity_rel.float() * _QNN_DIST_SCALE                    # (B, N, 3)
    vel = entity_vel.float() * _QNN_VEL_SCALE                     # (B, N, 3)
    half = entity_half.float() * _QNN_DIST_SCALE                  # (B, N, 3)
    # Clamp bbox to >=4u per axis (matches projectile_test in C/Python).
    half_clamped = torch.clamp(half, min=4.0)

    impulse = _entity_ids_to_impulse(weapon_id_entity).clamp(0, luts["n"] - 1)  # (B,)
    is_hitscan = luts["hitscan"][impulse]                          # (B,) bool
    w_range = luts["range"][impulse]                               # (B,)
    w_spread = luts["spread"][impulse]                             # (B,)
    w_speed = luts["speed"][impulse]                               # (B,)
    w_splash = luts["splash"][impulse]                             # (B,)
    w_max_t = luts["max_t"][impulse]                               # (B,)
    w_gravity = luts["gravity"][impulse]                           # (B,)

    # ── Hitscan path ────────────────────────────────────────────────
    # Ray-AABB slab. look = (1, 0, 0). origin = 0. Box at rel with half.
    # Per slot dim i:  p_i = -rel_i; d_i = look_i.
    # For look = (1, 0, 0):
    #   axis 0: d=1, t1 = (-half - p_x) / 1 = -half_x + rel_x,
    #                t2 = ( half + rel_x).  Sort then tmin/tmax.
    #   axes 1, 2: d=0, slab degenerate — require |p_i| <= half_i
    #              i.e. |rel_i| <= half_i.
    rel_x = rel[..., 0]                                            # (B, N)
    rel_y = rel[..., 1]
    rel_z = rel[..., 2]
    half_x = half[..., 0]
    half_y = half[..., 1]
    half_z = half[..., 2]
    in_yz = (rel_y.abs() <= half_y) & (rel_z.abs() <= half_z)      # (B, N) bool
    t1 = rel_x - half_x
    t2 = rel_x + half_x
    tmin = torch.minimum(t1, t2)
    tmax = torch.maximum(t1, t2)
    ray_dist = torch.where(tmin > 0.0, tmin, torch.zeros_like(tmin))
    ray_hit_geom = in_yz & (tmin <= tmax) & (tmax >= 0.0)
    dist_finite = torch.where(
        ray_hit_geom, ray_dist, torch.full_like(ray_dist, float("inf")),
    )
    # Direct slab hit within range
    hit_slab = dist_finite <= w_range.unsqueeze(-1)                # (B, N)
    # Spread-cone admission: ang_off = acos(rel.x / |rel|). Compared
    # against the weapon's spread half-angle, AND rel_norm <= range.
    rel_norm = torch.linalg.vector_norm(rel, dim=-1).clamp(min=1e-3)   # (B, N)
    cos_off = (rel_x / rel_norm).clamp(-1.0, 1.0)
    ang_off = torch.acos(cos_off)
    spread = w_spread.unsqueeze(-1)
    hit_cone = (spread > 0.0) & (ang_off <= spread) & (rel_norm <= w_range.unsqueeze(-1))
    hit_hitscan = (hit_slab | hit_cone) & is_hitscan.unsqueeze(-1)

    # ── Projectile path ─────────────────────────────────────────────
    # dt = 0.05s, n_steps = max_t / dt (max 80 for RL). For each step
    # we test if the projectile is within the bbox or within splash of
    # bbox edge. The whole loop runs over a constant range and produces
    # a (B, N, n_steps) hit mask we reduce with any().
    dt = 0.05
    n_max_steps = int(round(max(p[5] for p in _WEAPON_PHYSICS) / dt))  # = 80
    t_steps = torch.arange(n_max_steps, device=device, dtype=torch.float32) * dt  # (S,)
    # Active mask per (B,): for projectile-armed slots only, and per
    # step: t < w_max_t. Hitscan slots can be skipped early.
    is_projectile = (~is_hitscan)                                # (B,)
    speed = w_speed.unsqueeze(-1).unsqueeze(-1)                  # (B, 1, 1)
    gravity = w_gravity.unsqueeze(-1).unsqueeze(-1)              # (B, 1, 1)
    splash = w_splash.unsqueeze(-1)                              # (B, 1)
    max_t = w_max_t.unsqueeze(-1)                                # (B, 1)
    # proj position at each step t: along (+1,0,0) at speed - gravity z drop.
    #   proj_x(t) = speed * t
    #   proj_y(t) = 0
    #   proj_z(t) = -0.5 * gravity * t^2
    proj_x = speed * t_steps.view(1, 1, -1)                      # (B, 1, S)
    proj_z = -0.5 * gravity * (t_steps.view(1, 1, -1) ** 2)      # (B, 1, S)
    # target position at each step t (per slot): rel + vel * t.
    targ_x = rel_x.unsqueeze(-1) + vel[..., 0].unsqueeze(-1) * t_steps.view(1, 1, -1)
    targ_y = rel_y.unsqueeze(-1) + vel[..., 1].unsqueeze(-1) * t_steps.view(1, 1, -1)
    targ_z = rel_z.unsqueeze(-1) + vel[..., 2].unsqueeze(-1) * t_steps.view(1, 1, -1)
    diff_x = proj_x.expand_as(targ_x) - targ_x
    diff_y = -targ_y                                              # proj_y = 0
    diff_z = proj_z.expand_as(targ_z) - targ_z
    # Direct hit at each step: |diff| <= half_clamped axis-wise.
    abs_x = diff_x.abs()
    abs_y = diff_y.abs()
    abs_z = diff_z.abs()
    hcx = half_clamped[..., 0].unsqueeze(-1)
    hcy = half_clamped[..., 1].unsqueeze(-1)
    hcz = half_clamped[..., 2].unsqueeze(-1)
    direct = (abs_x <= hcx) & (abs_y <= hcy) & (abs_z <= hcz)    # (B, N, S)
    # Splash hit: distance from projectile to bbox surface <= splash.
    gap_x = torch.clamp(abs_x - hcx, min=0.0)
    gap_y = torch.clamp(abs_y - hcy, min=0.0)
    gap_z = torch.clamp(abs_z - hcz, min=0.0)
    gap_norm = torch.sqrt(gap_x ** 2 + gap_y ** 2 + gap_z ** 2)
    splash_active = splash.unsqueeze(-1) > 0.0                   # (B, 1, 1)
    splash_hit = splash_active & (gap_norm <= splash.unsqueeze(-1))
    # Step-active mask: only steps with t < max_t count.
    step_active = t_steps.view(1, 1, -1) < max_t.unsqueeze(-1)
    any_hit_step = ((direct | splash_hit) & step_active).any(dim=-1)  # (B, N)
    hit_projectile = any_hit_step & is_projectile.unsqueeze(-1)

    final = (hit_hitscan | hit_projectile) & actor_mask
    return final
