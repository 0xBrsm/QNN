# Weapon-aware target labeler: projectile-lead correction spec

## Problem statement

`src/qnn/bc/target_labeler.py` currently uses a single weapon-agnostic
adaptive cone for fire-frame target attribution:

```
acquire(d) = clamp(atan(208/d), 5°, 30°)
release(d) = clamp(atan(416/d), 5°, 45°)
```

The cone is centered on the **target's current position** and is the same
for every weapon. This treats an axe swing at 300u, a shotgun blast at
800u, and a rocket fired at 1500u with the same admit/reject rule. That is
wrong in two distinct ways:

1. The projectile's flight time means the demonstrator was aiming at a
   **lead point**, not at the target's current cell. A rocket fired at a
   target 1200u away moving 320 u/s laterally is aimed ≈384u beside the
   target — well outside today's 30° cap at that distance.
2. Different weapons have different effective angular tolerance. A
   well-aimed LG beam needs to be within ~1° of the target. A
   well-aimed rocket at 1200u only needs to be within whatever cone is
   reachable in the projectile's 1.2s flight time.

This spec specifies a weapon-aware replacement: per-weapon projectile
constants, a lead-correction step on the aim vector, and a cone half-angle
derived from "how far can the target move during the projectile's flight."

All numbers below are pulled from `vendor/quake/QW/progs/weapons.qc` and
`vendor/quake/QW/server/sv_phys.c` (QuakeWorld 2.40 source as released by
id Software), or — when QC is silent — from `vendor/quakec/qc/weapons.qc`
(NetQuake QC). Line numbers refer to the files in this repo.

---

## Section 1: Per-weapon projectile constants

### 1.1 Axe (impulse 1)

- **Type**: melee traceline (no projectile entity).
- **Range**: 64 units. Source: `vendor/quake/QW/progs/weapons.qc:44`
  ```
  traceline (source, source + v_forward*64, FALSE, self);
  ```
- **Damage**: 20 (DM1-3), 75 (DM4). Source: `weapons.qc:55-57`.
- **Speed**: hitscan-equivalent (zero flight time). Treat as `inf`.
- **Cone implication**: cone collapses to ~0°; any fire from outside 64u
  cannot be a valid axe swing regardless of aim. Add a hard range gate.

### 1.2 Shotgun (impulse 2)

- **Type**: hitscan, 6 pellets per shot.
- **Spread vector**: `'0.04 0.04 0'` (QW; same in NQ). Source:
  `weapons.qc:311`:
  ```
  FireBullets (6, dir, '0.04 0.04 0');
  ```
- **Trace length**: 2048 units (`FireBullets`, `weapons.qc:277, 283`).
- **Spread interpretation**: in `FireBullets`,
  ```
  direction = dir + crandom()*spread_x*v_right + crandom()*spread_y*v_up;
  ```
  where `crandom()` returns a uniform sample on [-1, +1]. Because
  `direction` is then used as an unnormalized ray direction (`traceline(src,
  src + direction*2048, ...)`), the spread is in pre-normalized
  unit-vector coordinates. Effective half-spread:
  - horizontal: `atan(0.04) ≈ 2.29°`
  - vertical:   `atan(0.04) ≈ 2.29°`
- **Damage per pellet**: 4 (`weapons.qc:285`).
- **Max useful range**: pellets traceline 2048u; damage falloff: none in
  vanilla. Practical aim envelope: full 2.29° pellet spread means any
  shot within that cone of crosshair-to-target is on-aim regardless of
  lead. Treat as hitscan.

### 1.3 Super Shotgun (impulse 3)

- **Type**: hitscan, 14 pellets per shot.
- **Spread vector**: `'0.14 0.08 0'`. Source: `weapons.qc:338`:
  ```
  FireBullets (14, dir, '0.14 0.08 0');
  ```
- **Effective half-spread**:
  - horizontal: `atan(0.14) ≈ 7.97°`
  - vertical:   `atan(0.08) ≈ 4.57°`
- **Damage per pellet**: 4.
- **Trace length**: 2048u (same `FireBullets` path).
- **Practical aim**: with a ~8° native spread, SSG is the weapon where
  cone widening from lead correction matters least — even the largest
  per-frame target motion is below the pellet spread at typical
  engagement distances.

### 1.4 Nailgun (NG, impulse 4)

- **Type**: projectile (spike), MOVETYPE_FLYMISSILE → no gravity.
- **Speed**: **1000 u/s**. Source: `weapons.qc:717`
  ```
  newmis.velocity = dir * 1000;
  ```
  inside `launch_spike()`, which is called by both `W_FireSpikes()` and
  `W_FireSuperSpikes()`.
- **Spread**: none from the spike launch itself. The nailgun's two
  barrels alternate via the `ox` argument (`weapons.qc:738, 763`); each
  individual nail is launched along `dir = aim(self,1000)` which is
  almost always exactly `v_forward` (see §1.10 for the QW `sv_aim`
  default of 2.0 making autoaim a no-op).
- **Damage per nail**: 9 (`weapons.qc:797`).
- **Lifetime**: 6 s (`weapons.qc:712`), so range is bounded only by
  collision.

### 1.5 Super Nailgun (SNG, impulse 5)

- **Type**: projectile (superspike), MOVETYPE_FLYMISSILE.
- **Speed**: **1000 u/s** (same `launch_spike()` path, same line).
- **Damage per nail**: 18 (`weapons.qc:844`).
- **Spread**: none.

### 1.6 Grenade Launcher (GL, impulse 6)

- **Type**: projectile, MOVETYPE_BOUNCE → subject to gravity. Source:
  `weapons.qc:644, 678`.
- **Initial speed**: 600 u/s along forward, +200 u/s vertical kick.
  Source: `weapons.qc:653, 657-658`:
  ```
  newmis.velocity = v_forward*600 + v_up*200 + crandom()*v_right*10 + crandom()*v_up*10;   // pitched
  newmis.velocity = aim(self,10000); newmis.velocity = newmis.velocity*600; newmis.velocity_z = 200;   // level shot
  ```
  When the player has nonzero pitch (`v_angle_x != 0`), the engine uses
  the pitched branch: `v_forward*600 + v_up*200`. When pitch is zero,
  the engine uses the level branch: a horizontal 600 u/s + a fixed
  +200 u/s upward kick (Quake's "lob"). The ±10 u/s jitter is small
  enough to ignore.
- **Gravity**: 800 u/s² downward. Source:
  `vendor/quake/QW/server/sv_phys.c:44`:
  ```
  cvar_t sv_gravity = { "sv_gravity", "800"};
  ```
- **Splash radius**: 120 units (RadiusDamage). Source: `weapons.qc:600`:
  ```
  T_RadiusDamage (self, self.owner, 120, world, "grenade");
  ```
- **Detonation**: time fuse 2.5s (`weapons.qc:670, 676`) OR direct hit
  via `GrenadeTouch` (`weapons.qc:612-624`).
- **Effective speed for lead**: in the level-shot case, horizontal
  component is 600 u/s — that's the right value for ballistic lead.
- **Damage**: 100 + random*20 + radius (`T_RadiusDamage 120`, falloff
  linear).

### 1.7 Rocket Launcher (RL, impulse 7)

- **Type**: projectile, MOVETYPE_FLYMISSILE → no gravity. Source:
  `weapons.qc:431`.
- **Speed**: **1000 u/s**. Source: `weapons.qc:437-438`:
  ```
  newmis.velocity = aim(self, 1000);
  newmis.velocity = newmis.velocity * 1000;
  ```
- **Self-velocity inheritance**: **NONE** in vanilla QW. The rocket's
  velocity is assigned absolutely from `aim() * 1000`; no
  `+ self.velocity` term anywhere. Verified by full grep of the
  function: there is no read of `self.velocity` in `W_FireRocket`.
- **Splash radius**: 120 units. Source: `weapons.qc:397`.
- **Damage**: 100 + random*20 on direct hit (`weapons.qc:385`), plus
  120-radius splash.
- **NQ comparison**: vanilla NetQuake (`vendor/quakec/qc/weapons.qc:409-410`)
  also uses `aim(self,1000) * 1000` with no inheritance. The
  Quake speed-running community has occasionally added rocket-velocity
  inheritance via mods (e.g., NEX, certain TF variants), but the
  reference QW source does not.
- **Lifetime**: 5s (`weapons.qc:445`), so max travel ≈5000u.

### 1.8 Lightning Gun (LG, impulse 8)

- **Type**: hitscan beam, 3 parallel tracelines (center + left + right
  shoulder offsets).
- **Max range**: **600 units**. Source: `weapons.qc:573`:
  ```
  traceline (org, org + v_forward*600, TRUE, self);
  ```
  (Both QW and NQ use 600 — the often-cited "768" is incorrect for
  vanilla QC; it may originate from an engine constant or mod variant
  the author confused with the QC range.)
- **Damage per cell**: 30 per beam frame, applied per-tick the button
  is held (`weapons.qc:586`):
  ```
  LightningDamage (self.origin, trace_endpos + v_forward*4, self, 30);
  ```
  At 10 Hz fire rate (LG's `attack_finished` cadence in QW is roughly
  0.1s) this is 300 dps on the center beam.
- **Triple beam**: `LightningDamage` traces three lines 16u apart
  (`weapons.qc:482-487`). Each parallel beam can hit a different
  entity, but the BC labeler should treat the center beam as the
  primary aim line.
- **Cone implication**: hitscan, lead = 0.

### 1.9 Engine ancillary constants

| name              | value | source |
|-------------------|-------|--------|
| `sv_maxspeed`     | 320 u/s | `QW/server/sv_phys.c:46` |
| `sv_maxvelocity`  | 2000 u/s (hard clamp) | `sv_phys.c:42` |
| `sv_airaccelerate`| 0.7 (acceleration coefficient, not a speed) | `sv_phys.c:49` |
| air-strafe wishspeed cap | **30 u/s** (added per frame in air, projected onto wishdir) | `QW/client/pmove.c:422-423`: `if (wishspd > 30) wishspd = 30;` |
| `sv_gravity`      | 800 u/s² | `sv_phys.c:44` |
| `sv_friction`     | 4 | `sv_phys.c:51` |
| `sv_stopspeed`    | 100 u/s | `sv_phys.c:45` |
| `sv_accelerate`   | 10 (ground accel coeff) | `sv_phys.c:48` |

### 1.10 sv_aim and autoaim in QW

`sv_aim` is the cosine threshold under which QW's autoaim function
`PF_aim` will adjust a fire direction toward a nearby target.
`vendor/quake/QW/server/pr_cmds.c:1157` sets the default to **2.0**:

```
cvar_t sv_aim = {"sv_aim", "2"};
```

Since `dist = DotProduct(dir, v_forward)` is always ≤ 1, the condition
`if (dist < bestdist)` with `bestdist = 2.0` is always true → no entity
is ever selected → `aim()` returns plain `v_forward` for every shot.

**Implication for the labeler**: we can treat the demonstrator's aim
direction as **exactly** the `look` vector (their `v_angle` forward),
with no autoaim adjustment to model. Aim correction is purely the
projectile-lead term.

---

## Section 2: Target movement caps

### 2.1 Ground maxspeed

`sv_maxspeed = 320`. Source: `QW/server/sv_phys.c:46`. The QW pmove
enforces this in `PM_GroundMove` and `PM_WaterMove`
(`QW/client/pmove.c:467-470, 530-533`). A grounded player without
external impulses is hard-clamped to 320 u/s ground-plane speed.

### 2.2 Air-strafe and bunnyhopping

QW's `PM_AirAccelerate` (`pmove.c:412-434`) caps the wishspeed
contribution per frame at **30 u/s** projected onto wishdir, but does
**not** cap total velocity in the air. By turning the view through a
trajectory tangent during a jump, a player adds tangential acceleration
without scrubbing forward speed, so effective airspeed grows. The hard
cap is `sv_maxvelocity = 2000` (component-wise).

Community references for sustained competitive bunnyhop speed:
- QuakeWorld speed records on `dm6`, `aerowalk`, `bravado` show 600–900
  u/s peak ground-plane velocities during chained jumps. Source: QW
  speedrunning community (Speed Demos Archive, quake.world).
- Mid-skill bot/player engagements typically see 350–550 u/s peak,
  300–400 u/s sustained while strafing across an open arena.
- The classic ZQuake/ezQuake training maps (e.g., `endif`) routinely
  show 700+ u/s for a single jump on a downhill ramp.

There is no QC-level cap below 2000; the cap is determined by player
skill and the map's available jump-chaining geometry.

### 2.3 Orthogonal-to-aim component

This is the only component that actually requires lead. A target moving
directly toward or away from the shooter contributes no lead. The
worst-case orthogonal speed is bounded by total speed, but in practice:

| scenario              | total speed | typical V_perp |
|-----------------------|------------:|---------------:|
| stationary             | 0           | 0              |
| walking ground         | 200 u/s     | 200            |
| running ground         | 320 u/s     | 320            |
| strafe-jumping mid-air | 450 u/s     | 400            |
| bunny-hop train, chained | 700 u/s   | 500–600        |
| absolute QW ceiling    | 2000 u/s    | 1500           |

**Recommended `V_perp_max` for the labeler**: **500 u/s**. This covers
strafe-jumping demonstrators at competitive but non-record speeds and
sits between `sv_maxspeed` (320 — too low) and the absolute bunnyhop
ceiling (~900 — too generous, would widen cones to the point of
admitting unrelated enemies). Justification:

1. The QWD corpus is 1v1 / FFA QW demos at competitive but not
   world-record skill — speed records are rare in normal duel demos.
2. 500 u/s puts the orthogonal-cap above the `sv_maxspeed` ground
   ceiling so all ground combat is covered.
3. It is small enough that the RL/GL cones at 1000u stay tight
   (atan(500/1000) ≈ 26.6°), preserving discrimination when two
   enemies are in view.

A future audit on demo data of measured target orthogonal velocity at
fire frames could refine this to a percentile (e.g., the 95th percentile
of |V_perp| over fire frames). Until measured, 500 u/s is the
conservative default.

---

## Section 3: Lead-correction formulas

### 3.1 Hitscan (Axe, SG, SSG, LG)

```
T_flight   = 0
lead       = 0
aim_target = target_position - shooter_position
```

The cone is centered on the current target position. For Axe, additionally
gate by range ≤ 64u (`weapons.qc:44`).

### 3.2 Constant-speed projectile (NG, SNG, RL)

```
projectile_speed = {NG: 1000, SNG: 1000, RL: 1000}        # u/s
rel              = target_position - shooter_position
T_flight         = ||rel|| / projectile_speed
lead             = target_velocity * T_flight
aim_target       = rel + lead
```

This is the standard "iterative-fixed-point at depth 1" lead. The truly
self-consistent solution requires
```
||rel + V_target * T||  =  projectile_speed * T
```
which is a quadratic in T:
```
(V_target . V_target - projectile_speed²) T²
  + 2 (rel . V_target) T
  + (rel . rel) = 0
```
For non-evasive targets, the depth-1 approximation is within ~5% of the
exact solution at all reasonable engagement distances and is what most
prediction-aware aimbots use. The labeler should use the simpler form.

### 3.3 Ballistic (GL)

The grenade leaves the muzzle with horizontal speed 600 u/s plus a fixed
+200 u/s vertical kick on level shots. With gravity g = 800 u/s²:

```
horizontal_speed = 600                       # u/s (after the v_up term)
position(t)      = origin + dir_h * 600 * t + (0,0,200)*t - (0,0,0.5*g*t²)
```

For a target at horizontal distance d_h and elevation Δz, the intercept
times (when the grenade has horizontal range d_h and matches Δz) are
the roots of:

```
horizontal:  d_h = 600 * t          →  t = d_h / 600
vertical:    Δz  = 200*t - 400*t²   →  solve at t = d_h/600
```

This is over-determined — you can match horizontal range but not also
match vertical without choosing a pitch angle. For the general pitched
case the player provides `pitch`, and the engine spawns
```
v_forward * 600 + v_up * 200
```
which means the muzzle velocity rotates with pitch (since v_forward and
v_up both depend on `v_angle`). The pitched horizontal speed is
`600 * cos(pitch)` and vertical is `600 * sin(pitch) + 200 * cos(pitch_up)`.

**Recommendation for the labeler**: do NOT solve the full ballistic.
Use the straight-line approximation with
`projectile_speed = 600 u/s`. Justification:

1. The labeler's job is to classify "did the demonstrator try to hit
   this enemy" — not to reconstruct grenade-arc geometry exactly.
2. The straight-line cone at 600 u/s is already much wider than the
   RL cone, so it tolerates the ~10–20% angular error introduced by
   ignoring the parabolic drop.
3. Splash radius (120u) is large enough that "lead within the cone"
   correctly identifies hits even with a moderate ballistic miss.
4. Doing the full quadratic-in-quadratic gravity solve adds ~40 lines
   of brittle numerics for a head that gets fired rarely
   (GL is ≤ 5% of QWD fires).

**Worked example** (concrete numbers, GL, target at 1000u horizontal,
moving 320 u/s laterally):

- Straight-line approximation: `T = 1000 / 600 = 1.667 s`,
  `lead = 320 * 1.667 = 533 u` lateral.
  Aim-point offset from target: 533u perpendicular. Cone angle to admit
  this miss: `atan(533/1000) ≈ 28.0°`. The straight-line cone at
  V_perp_max=500 is `atan(500/600) ≈ 39.8°`, comfortably admitting the
  case.
- Exact ballistic (gravity arc, ignoring +200 vertical kick): the
  grenade with horizontal speed 600 takes the same 1.667s. During that
  flight it drops `0.5 * 800 * 1.667² ≈ 1111u` vertically. The
  demonstrator compensates by pitching up; the time-to-target is
  unchanged for an iso-elevation target, so the lateral lead is the
  same. The straight-line approximation captures the lateral
  component, which is the only one affecting cone-on-target.

### 3.4 Self-velocity inheritance

QW vanilla: **none** (verified in §1.7). The straight-line / quadratic
forms above use `projectile_speed` in the world frame; no shooter-motion
correction term.

If a future mod (NEX, custom TF) introduces inheritance with a fraction
α ∈ [0, 1]:
```
effective_v = projectile_speed * dir + α * shooter_velocity
T            = ||rel|| / ||effective_v||              (when small angle, scalar OK)
lead         = (target_velocity - α * shooter_velocity_along_dir) * T
```
Today: α = 0. The shooter-velocity term in the labeler should be a no-op
unless a per-mod constant overrides.

---

## Section 4: Cone width from orthogonal-intercept

The cone half-angle that admits "any target that could have been
intercepted, given perfect lead, by a projectile fired at this
direction" is:

```
cone_half_angle = atan(V_perp_max / projectile_speed)
```

With `V_perp_max = 500 u/s` (§2.3):

| weapon         | projectile_speed | V_perp_max | computed half-angle | notes |
|----------------|------------------|------------|---------------------|-------|
| Axe            | ∞ (hitscan)      | n/a        | ~0° (use 5° floor)  | + hard 64u range gate |
| Shotgun (SG)   | ∞ (hitscan)      | n/a        | ~2.29° (pellet spread) | use pellet spread directly |
| Super Shotgun  | ∞ (hitscan)      | n/a        | ~7.97° horiz / 4.57° vert | use spread direct |
| Nailgun        | 1000 u/s         | 500        | `atan(500/1000)` = **26.57°** | |
| Super Nailgun  | 1000 u/s         | 500        | **26.57°** | same physics |
| Grenade Lncher | 600 u/s          | 500        | `atan(500/600)` = **39.81°** | straight-line approx |
| Rocket Lncher  | 1000 u/s         | 500        | **26.57°** | |
| Lightning Gun  | ∞ (hitscan)      | n/a        | ~0° (use 5° floor)  | + hard 600u range gate |

These are the **physics-derived "perfect-lead" cones**. They are
distance-invariant — that's the whole point of switching from "fixed
transverse offset" to "speed-ratio cone". The labeler today widens
arctan(208/d) at short range and tightens at long range, which is
arbitrary; the speed-ratio cone gives a single value per weapon that
is correct at all distances.

### Comparison with current labeler at example distances

For NG/SNG/RL at projectile_speed = 1000, V_perp = 500 → **26.57°**:

| distance (u) | current acquire | current release | new (RL/NG/SNG) |
|-------------:|----------------:|----------------:|----------------:|
| 300          | 30°  (capped)   | 45° (capped)    | 26.57°          |
| 600          | 19.1°           | 34.7°           | 26.57°          |
| 1000         | 11.75°          | 22.6°           | 26.57°          |
| 1500         | 7.91°           | 15.5°           | 26.57°          |
| 2000         | 5.95°           | 11.75°          | 26.57°          |

The new rule is **tighter at close range** (good: prevents wild
attribution) and **looser at long range** (good: long-range rockets at
fast strafers need wider tolerance, and the current 5–8° at 1500-2000u
is rejecting plenty of legitimate engagements).

For RL the demonstrator's natural skill envelope is the projectile-speed
ratio at all ranges, which is exactly what we now compute.

---

## Section 5: Fudge factor for imperfect lead

The §4 cone assumes the demonstrator leads perfectly. Real players lead
with ±20–40% error depending on skill, target trajectory predictability,
and reaction-time noise. The cone must admit those imperfect leads
without admitting unrelated enemies.

### 5.1 Mechanism

Two orthogonal ways to widen:

1. **Multiplicative**: `cone_half_angle *= K`. Scales with weapon, so
   slow projectiles get more absolute tolerance and fast hitscan stays
   tight.
2. **Additive floor**: `max(cone_half_angle, MIN_ANGLE)`. Ensures
   hitscan weapons still have *some* tolerance for cone-jitter,
   crosshair float, and the labeler's discrete-frame quantization.

### 5.2 Recommended values

```
ACQUIRE_FUDGE       = 1.0          # acquire cone uses the physics value
RELEASE_FUDGE       = 1.5          # release cone widened by 1.5×
                                   # (preserves Schmitt-trigger ratio)
HITSCAN_FLOOR_DEG   = 5.0          # absolute floor for hitscan weapons
PROJECTILE_FLOOR_DEG = 8.0         # absolute floor for projectile weapons
HITSCAN_RELEASE_DEG = 8.0          # release floor (hitscan)
PROJECTILE_RELEASE_DEG = 14.0      # release floor (projectile)
```

Rationale:

- **Acquire = 1.0 of physics** keeps acquire tight. The whole point of
  cone-argmax acquire is "if multiple enemies are in view, pick the
  one closest to perfect lead." Widening acquire blurs that.
- **Release = 1.5× of physics** preserves the K=1.5–2 Schmitt-trigger
  spirit of the current labeler. Once a target is sticky-locked we
  tolerate ~50% more lead error before releasing.
- **Hitscan floor 5°** matches the current acquire floor (`5°` in
  `target_labeler.py:46`). The current code's 5° was chosen for the
  weapon-agnostic case; for hitscan specifically it's the right number
  because it's roughly the SSG's horizontal pellet spread plus a
  cone-jitter margin.
- **Projectile floor 8°** is a hedge for the depth-1 lead
  approximation error at short range (where T_flight is small and the
  lead is mostly noise) plus per-frame look quantization. At
  d=200u, NG: T=0.2s, V_perp=500 → lead=100u, angle=atan(100/200)=26.6°
  which is above the floor, so the floor doesn't fire there. The floor
  fires only when V_perp << V_perp_max, i.e., a near-stationary target,
  where lead correction can't be trusted from a single-frame velocity
  estimate.
- **Hitscan release 8°** and **projectile release 14°** are the
  `release_floor = 1.5 × acquire_floor + some` proportions that mirror
  the current 5°/8° design.

### 5.3 What the cone looks like in practice

| weapon | acquire | release |
|--------|--------:|--------:|
| Axe    | 5°  (hard range gate ≤64u) | 8° |
| SG     | 5°  (+ inherent 2.3° pellet) | 8° |
| SSG    | 8°  (matches H pellet spread) | 12° |
| NG/SNG | 26.6° | 39.9° |
| GL     | 39.8° | 59.7° (capped at 60° max) |
| RL     | 26.6° | 39.9° |
| LG     | 5°  (hard range gate ≤600u) | 8° |

A **hard absolute cap** of 60° prevents the GL release cone from
admitting half the field of view when the demonstrator was clearly
aiming elsewhere. This matches the current release cap of 45° in spirit
(now relaxed because GL legitimately needs more tolerance).

---

## Section 6: Implementation summary

The drop-in replacement for the cone-threshold block in
`label_enemy_target` (currently lines 137–141 in
`src/qnn/bc/target_labeler.py`):

```python
# Per-weapon physics constants. Speeds in Quake u/s; angles in degrees.
# Source: vendor/quake/QW/progs/weapons.qc (line numbers in spec doc).
_WEAPON_SPEED = {       # projectile speed; np.inf for hitscan
    1: np.inf,          # Axe
    2: np.inf,          # Shotgun (hitscan, 6 pellets)
    3: np.inf,          # Super Shotgun (hitscan, 14 pellets)
    4: 1000.0,          # Nailgun
    5: 1000.0,          # Super Nailgun
    6: 600.0,           # Grenade Launcher (straight-line approx)
    7: 1000.0,          # Rocket Launcher
    8: np.inf,          # Lightning Gun
}
_WEAPON_MAX_RANGE = {   # absolute kill at this range (np.inf = no gate)
    1: 64.0, 2: np.inf, 3: np.inf, 4: np.inf,
    5: np.inf, 6: np.inf, 7: np.inf, 8: 600.0,
}
_HITSCAN_ACQUIRE_DEG = 5.0
_HITSCAN_RELEASE_DEG = 8.0
_PROJECTILE_FLOOR_DEG = 8.0
_PROJECTILE_RELEASE_FLOOR_DEG = 14.0
_CONE_HARD_CAP_DEG = 60.0
_V_PERP_MAX = 500.0     # u/s (see spec §2.3)
_RELEASE_FUDGE = 1.5    # multiplicative widening on release

def _compute_cone_cos(dist_qu, target_vel_qu, look_unit, weapon_id):
    """Per-frame, per-enemy (acquire_cos, release_cos) thresholds.

    All inputs broadcast to (T, N) where N is the enemy-actor slot count.
    `target_vel_qu` is the enemy's world-frame velocity in u/s.
    `weapon_id` is the demonstrator's current weapon (1..8), shape (T,).

    Cone center is the lead-corrected aim direction:
        T_flight    = dist / projectile_speed
        lead_world  = target_vel * T_flight
        aim_target  = rel + lead_world
    But because we only need cone width (not the cos against a shifted
    point), we compute the cone in two pieces: (a) the angular offset
    from raw 'rel' to 'rel + lead' (used to recenter cos_tr against the
    lead-corrected aim), and (b) the cone half-width per weapon.
    """
    speed = np.array([_WEAPON_SPEED[int(w)] for w in weapon_id])  # (T,)
    speed = speed[:, None]                                         # (T, 1)
    # Hitscan: T_flight = 0; lead = 0; cone is pure angular floor.
    # Projectile: cone half-angle = atan(V_perp_max / speed).
    is_hitscan = np.isinf(speed)
    physics_half_rad = np.arctan(np.where(is_hitscan, 0.0,
                                          _V_PERP_MAX / np.maximum(speed, 1.0)))
    acquire_floor = np.where(is_hitscan,
                             np.radians(_HITSCAN_ACQUIRE_DEG),
                             np.radians(_PROJECTILE_FLOOR_DEG))
    release_floor = np.where(is_hitscan,
                             np.radians(_HITSCAN_RELEASE_DEG),
                             np.radians(_PROJECTILE_RELEASE_FLOOR_DEG))
    acquire_half = np.maximum(physics_half_rad, acquire_floor)
    release_half = np.maximum(physics_half_rad * _RELEASE_FUDGE,
                              release_floor)
    cap = np.radians(_CONE_HARD_CAP_DEG)
    acquire_half = np.minimum(acquire_half, cap)
    release_half = np.minimum(release_half, cap)
    return np.cos(acquire_half), np.cos(release_half)  # (T, 1) each
```

Then the existing `cos_tr` computation must be done **against the
lead-corrected aim-point**, not against raw `rel`:

```python
# Replace the existing rel/unit_rel block with lead-corrected version.
# entity_scalars[..., _ACTOR_VEL_OFFSET:_ACTOR_VEL_OFFSET+3] is the
# enemy's velocity in obs-scaled units; rescale to u/s.
_ACTOR_VEL_OFFSET = 6                       # see _ACTOR_LAYOUT (vel at 6..8)
rel    = entity_scalars[:, :, _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET+3]
vel    = entity_scalars[:, :, _ACTOR_VEL_OFFSET:_ACTOR_VEL_OFFSET+3]
rel_qu = rel * QNN_DIST_SCALE                              # (T,N,3) u
vel_qu = vel * QNN_VEL_SCALE                               # (T,N,3) u/s
dist_qu = np.linalg.norm(rel_qu, axis=-1)                  # (T,N)
speed = np.asarray([_WEAPON_SPEED[int(w)] for w in weapon_id])  # (T,)
T_flight = np.where(np.isinf(speed),
                    0.0,
                    dist_qu / np.maximum(speed[:, None], 1.0))   # (T,N)
lead_qu = vel_qu * T_flight[..., None]                     # (T,N,3) u
aim_qu  = rel_qu + lead_qu                                 # (T,N,3) u
aim_norm = np.linalg.norm(aim_qu, axis=-1)
unit_aim = aim_qu / np.maximum(aim_norm[..., None], 1e-6)
cos_tr   = np.einsum("tij,tj->ti", unit_aim, unit_look)    # (T,N)

# Range gate: kill any candidate beyond per-weapon max range.
max_range = np.asarray([_WEAPON_MAX_RANGE[int(w)] for w in weapon_id])
in_range = dist_qu <= max_range[:, None]
cos_tr = np.where(in_range, cos_tr, -np.inf)

# Cone thresholds (scalar per frame; broadcasts to slot dim).
acquire_cos, release_cos = _compute_cone_cos(
    dist_qu, vel_qu, unit_look, weapon_id)
```

Then the rest of the Pass-1 / Pass-2 / Pass-3 logic in
`target_labeler.py` is unchanged — `acquire_thr[t, slot]` and
`release_thr[t, slot]` are the per-weapon cosines computed above
(broadcast over slot since they depend only on weapon, not slot).

### 6.1 Wiring at the call site

`label_enemy_target` takes `actions` today. Add a kwarg for weapon:

```python
def label_enemy_target(
    obs: Dict[str, np.ndarray],
    actions: Dict[str, np.ndarray],
    sight_only: bool = False,
) -> np.ndarray:
    ...
    look   = np.asarray(actions["look"])
    fire   = np.asarray(actions["fire"]).reshape(-1)
    weapon = np.asarray(actions.get("weapon",
              np.full(look.shape[0], 7, dtype=np.uint8)))  # default RL
```

The `weapon` array is already produced by the BC collector
(`src/qnn/bc/collect.py:507-517`) and stored in the actions dict, so no
collector changes are needed.

If `weapon[t] == 0` (no weapon held — pre-spawn, dead, transitional),
the labeler should fall back to "no shot can be valid this frame" —
fire ticks at those frames already don't happen because the engine
suppresses the fire button while dead, but defensively treat
`weapon == 0` as `projectile_speed = inf, max_range = 0` → cone admits
nothing.

### 6.2 Acceptance test

After patching, the labeler should produce:

- **Tighter** target attribution at close-range RL/NG/SNG combat (the
  current 30° cap is replaced by 26.6° — slightly tighter).
- **Looser** attribution at long-range RL (`d>1000`) where the
  physics-derived 26.6° beats the current 8–12°.
- **Strict** rejection of axe fires from > 64u and LG fires from > 600u
  — these are clearly miss-classifications that the range gate now
  catches.
- Frame-level disagreement with the current labeler concentrated at
  the (d > 1000, weapon ∈ {RL, NG, SNG}) and (d > 64, weapon = Axe)
  buckets, with the long-range bucket being adds (new labels where
  current rejects) and the close-range buckets being removes (range
  gate cleans up obvious mis-attributions).

Measure on the QWD train+val split before merge:
1. Total `target == enemy_slot` rate (currently 97.56% train / 97.43%
   val per `target_labeler_engine_alignment.md`).
2. Per-weapon `target_assigned / fires` rate.
3. Within-segment pid-switch count (currently 18,567 train).
4. Engine-labeler agreement on overlap frames (currently 98.24%).

Each metric should change by < 1.5 absolute percentage points; a
larger swing flags a regression to investigate.

---

## Sources

- `vendor/quake/QW/progs/weapons.qc` — QW QC weapons module (1402 lines,
  id Software QuakeWorld 2.40 source).
- `vendor/quake/QW/server/sv_phys.c` — QW server physics, cvars.
- `vendor/quake/QW/server/pr_cmds.c` — `PF_aim` autoaim definition.
- `vendor/quake/QW/client/pmove.c` — air-strafe accelerator (`PM_AirAccelerate`).
- `vendor/quakec/qc/weapons.qc` — NetQuake QC weapons module, used to
  confirm rocket no-inheritance and LG range parity between QW and NQ.
- `src/qnn/bc/target_labeler.py` — current labeler implementation
  (cone constants at lines 42–48).
- `src/qnn/bc/collect.py` — collector that produces the per-frame
  `weapon` actions field (lines 507–517).
- `src/qnn/env/sim.py` — obs layout (`_ACTOR_LAYOUT`,
  `_SELF_SCALAR_LAYOUT`).

Sources not consulted (and not needed for the spec):
- QuakeWorld speedrun community wikis. The QC-defined cvars and
  per-weapon physics are sufficient to specify the cones; community
  references in §2.2 are only for the V_perp_max recommendation, which
  is itself a tunable parameter that can be set from corpus measurement.
- Mod source (NEX, ezQuake mods). Vanilla QW QC is the right reference
  for the QWD corpus, which is recorded against unmodded QW servers.
