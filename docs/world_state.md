# World State

Design document for the worker's local world state representation. The goal is to mirror server-authoritative game state using raw game values, with no normalization or token-formatting concerns at this layer.

## Token Stream

Up to 16 entity tokens per tick in a single variable-length stream. Each token is self-describing by subject_id. Four entity classes, emitted in priority order:

1. **Projectiles** — immediate threats, highest priority
2. **Actors** (players) — opponents and teammates
3. **Items** — pickups and objectives
4. **Movers** (doors, platforms, trains, buttons, teleporters, push triggers) — world context

When more than 16 candidates exist, lower-priority classes are dropped first. Within a class, recency breaks ties.

All entity state updates from server transport (PVS), falling back to sound events when the entity is outside PVS. Transport is authoritative; sound is inference.

## Time

Use server time delta (`cl.mtime[0] - cl.mtime[1]`) as dt, not our fixed tick rate. This keeps timers (item regen, decay) synchronized with the server's actual simulation time. Units are seconds, same as QuakeC `time`.

## Items

Items are static map entities with known spawn locations. They are created at map load from BSP baselines and never move. Their state changes are: available, picked up (respawning), available again.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | What it is (vocab enum: HEALTH, MEGAHEALTH, ARMOR_GREEN, etc.) |
| entity_num | int | Server edict number, stable across the map lifetime |
| origin | vec3 | Absolute map position (xyz), fixed at spawn |
| regen | float | Seconds until respawn. 0 = available. Set to respawn time on pickup, counts down each tick. |
| amount | int | Raw game value the item provides |

### Item Table

#### Health

| Subject | Classname | Spawnflags | Amount | Unit | Respawn (s) |
|---------|-----------|------------|--------|------|-------------|
| HEALTH | item_health | & 1 | 15 | hp | 20 |
| HEALTH | item_health | (none) | 25 | hp | 20 |
| MEGAHEALTH | item_health | & 2 | 100 | hp | 120 |

#### Armor

| Subject | Classname | Amount | Absorb | Unit | Respawn (s) |
|---------|-----------|--------|--------|------|-------------|
| ARMOR_GREEN | item_armor1 | 100 | 0.3 | armor | 20 |
| ARMOR_YELLOW | item_armor2 | 150 | 0.6 | armor | 20 |
| ARMOR_RED | item_armorInv | 200 | 0.8 | armor | 20 |

#### Ammo

| Subject | Classname | Spawnflags | Amount | Unit | Respawn (s) |
|---------|-----------|------------|--------|------|-------------|
| SHELLS | item_shells | (none) | 20 | shells | 30 |
| SHELLS | item_shells | & 1 | 40 | shells | 30 |
| NAILS | item_spikes | (none) | 25 | nails | 30 |
| NAILS | item_spikes | & 1 | 50 | nails | 30 |
| ROCKETS | item_rockets | (none) | 5 | rockets | 30 |
| ROCKETS | item_rockets | & 1 | 10 | rockets | 30 |
| CELLS | item_cells | (none) | 6 | cells | 30 |
| CELLS | item_cells | & 1 | 12 | cells | 30 |

#### Weapons

Weapons give the weapon itself plus a fixed ammo bonus on pickup.

| Subject | Classname | Ammo bonus | Ammo type | Respawn (s) |
|---------|-----------|------------|-----------|-------------|
| SHOTGUN | weapon_supershotgun | 5 | shells | 30 |
| NAILGUN | weapon_nailgun | 30 | nails | 30 |
| NAILGUN | weapon_supernailgun | 30 | nails | 30 |
| GRENADE_LAUNCHER | weapon_grenadelauncher | 5 | rockets | 30 |
| ROCKET_LAUNCHER | weapon_rocketlauncher | 5 | rockets | 30 |
| THUNDERBOLT | weapon_lightning | 15 | cells | 30 |

#### Powerups

| Subject | Classname | Duration (s) | Respawn (s) |
|---------|-----------|--------------|-------------|
| QUAD | item_artifact_super_damage | 30 | 60 |
| PENT | item_artifact_invulnerability | 30 | 300 |
| RING | item_artifact_invisibility | 30 | 300 |
| SUIT | item_artifact_envirosuit | 30 | 60 |

## Movers

Movers are brush entities that change position in response to player interaction. They have deterministic timing from the BSP entity lump.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | What it is (DOOR, PLATFORM, TRAIN, BUTTON) |
| entity_num | int | Server edict number |
| origin | vec3 | Absolute map position (xyz), from transport or baseline |
| speed | float | Movement speed (units/s). Used by route planner. |
| wait | float | Seconds before returning to base state. -1 = permanent. Used by route planner. |
| state | float | 0 = base position, 1 = activated position |

### Defaults

| Subject | Classname | Speed | Wait |
|---------|-----------|-------|------|
| DOOR | func_door | 100 | 3 |
| DOOR | func_door_secret | 100 | 3 |
| PLATFORM | func_plat | 150 | 3 (hardcoded) |
| TRAIN | func_train | 100 | per path_corner |
| BUTTON | func_button | 40 | 1 |

### State Transitions

- **Transport available**: state derived from live origin vs baseline origin.
- **Transport unavailable, sound fires**: state toggles (sound = state transition).
- **Transport unavailable, no sound**: state holds. After `wait` seconds at state=1, assume return to state=0.

## Teleporters

Static map entities that relocate the player. No moving parts, no state changes.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | TELEPORTER |
| entity_num | int | Server edict number |
| origin | vec3 | Trigger brush position |
| destination | vec3 | Exit position (from linked info_teleport_destination) |

## Push Triggers

Static map entities that apply velocity to the player. No state changes.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | (none currently) |
| entity_num | int | Server edict number |
| origin | vec3 | Trigger brush position |
| speed | float | Push speed (default 1000) |
| direction | vec3 | Push direction (movedir from BSP) |

## Players

Dynamic entities. Only exist in `cl_entities` when the server is sending updates (player is in PVS). All fields from transport except velocity which is derived.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | PLAYER |
| entity_num | int | Server edict number (1 to maxclients) |
| origin | vec3 | Last server position (msg_origins[0]) |
| angles | vec3 | Facing direction (pitch, yaw, roll) |
| velocity | vec3 | Derived from msg_origins[0] - msg_origins[1] / server_dt |
| colormap | int | Pants/shirt color. Bottom 4 bits = team (pants). |
| effects | int | EF_DIMLIGHT = Quad or Pent glow active |

### Sources

- **In PVS**: all fields from server transport each frame.
- **Out of PVS**: no updates. Sound events provide position, weapon, powerup, action (fire, pain, death, jump).
- **Team**: colormap latched once from `cl.scores` at connect. Stable across the map.
- **Powerup**: EF_DIMLIGHT from effects (Quad/Pent). Ring/Suit have no visual indicator — sound only.
- **Disconnect**: detected from scoreboard name transition (non-empty → empty). Actor entry cleared. Needs: emit a DISCONNECT event to the token stream so the model knows a player left.
- **Connect**: detected from scoreboard name transition (empty → non-empty). Needs: emit a CONNECT event to the token stream so the model knows a new player joined. New QNN_ACTION_CONNECT vocab entry required.

## Open Issues

### Backpacks

Backpacks (progs/backpack.mdl) are dynamic pickups dropped on player death. They have edict numbers and appear in transport. Currently untracked — dropped when the old actor struct was split into actor/projectile stores. Need to add as a fifth entity type or extend the projectile store's ephemeral lifecycle.

### Projectile impact events

Rocket explosions, nail ricochets, and lightning hits have entity_num -1 (world sounds). The model sees projectiles in flight but never the impact — the projectile vanishes and the explosion sound has no entity attribution. Grenade bounces are the exception (entity still alive). Need spatial matching to connect impact sounds to recently-vanished projectiles.

### Demo clipping

Demo corpus contains pre-match warmup and post-match dead time that is training noise. Need a way to detect match boundaries for clipping. `cl.intermission` covers some cases but not all — some mods end matches without it, and demos may have dead time from the spectator stepping away. Needs a robust heuristic or metadata-based approach at the collection/pipeline level, not in the store.

## Projectiles

Dynamic entities spawned at runtime by weapon fire. No BSP baseline. Only exist in `cl_entities` while in PVS and alive.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| subject_id | int | PROJECTILE_NAIL, PROJECTILE_GRENADE, PROJECTILE_ROCKET, LIGHTNING_BEAM |
| entity_num | int | Server edict number (transient, reused) |
| origin | vec3 | Last server position (msg_origins[0]) |
| velocity | vec3 | Derived from msg_origins delta (direction + speed) |

### Sources

- **In PVS**: position and direction from transport each frame.
- **Out of PVS**: nothing. Fire sound (on player) and impact sound (on projectile) are the only events.
- **Owner**: not sent by server. Inferred from preceding fire sound on the same entity_num or nearby player.

## Item State

### State Transitions

- **Map load**: All items start with regen = 0 (available).
- **Pickup detected** (sound event): regen set to respawn time for that item type (20, 30, 60, 120, or 300s).
- **Each tick**: regen counts down by dt. Clamps at 0.
- **Respawn detected** (sound event or regen reaches 0): regen = 0.
- **Visible in PVS**: regen confirmed 0 (item is present on the server).
- **Not in PVS**: regen continues counting down. No inference from absence.

### Item Definition Table

Static lookup from (classname, spawnflags) to (subject_id, amount, regen_time). Populated at compile time. Used at map init to resolve each BSP item entity into its store fields.

### Sources

- subject_id, origin, entity_num, amount, regen_time: BSP entity lump at map load via item definition table.
- Pickup: `items/r_item1.wav`, `items/health1.wav`, `items/r_item2.wav`, `items/armor1.wav`, `weapons/pkup.wav`, `weapons/lock4.wav`, `items/damage.wav`, `items/protect.wav`, `items/inv1.wav`, `items/suit.wav`.
- Respawn: `items/itembk2.wav`.
- PVS confirmation: entity present in `cl_entities` with matching model.
