# Token Spec v9

## Overview

Per-type entity tokens with type-specific projection layers. Variable-length wire format with type tags. Fixed `[16, d_model]` output after projection + padding.

Priority order for the 16-slot stream: projectiles > actors > items > movers. Nail streams aggregated before trim. Recency breaks ties within a class.

## Vocab Tables

### Entity Vocab (42 entries, shared by subject_id and event source_id)

NONE(0), PLAYER(1), WEAPON(2), AXE(3), SHOTGUN(4), NAILGUN(5), GRENADE_LAUNCHER(6), ROCKET_LAUNCHER(7), THUNDERBOLT(8), AMMO(9), SHELLS(10), NAILS(11), ROCKETS(12), CELLS(13), BACKPACK(14), ARMOR(15), ARMOR_GREEN(16), ARMOR_YELLOW(17), ARMOR_RED(18), HEALTH(19), MEGAHEALTH(20), POWERUP(21), QUAD(22), PENT(23), RING(24), SUIT(25), PROJECTILE_NAIL(26), PROJECTILE_GRENADE(27), PROJECTILE_ROCKET(28), LIGHTNING_BEAM(29), GROUND(30), WATER(31), SLIME(32), LAVA(33), GIB(34), BUTTON(35), PLATFORM(36), TELEPORTER(37), DOOR(38), KEYED(39), SECRET(40), TRAIN(41).

Grouped: player, weapons, ammo, armor, health, powerups, projectiles, environment, movers. Source-only entries (WEAPON, AMMO, ARMOR, GROUND, WATER, SLIME, LAVA, GIB, KEYED, SECRET) interleaved in their category.

### Action Vocab (20 entries, event action_id)

NONE(0), FIRE(1), JUMP(2), LAND(3), PICKUP(4), ENTER(5), BREATH(6), EXIT(7), PAIN(8), DEATH(9), CONNECT(10), DISCONNECT(11), RESPAWN(12), ACTIVE(13), ENDING(14), BOUNCE(15), TELEPORT(16), MOVE(17), ACTIVATE(18), REJECT(19).

Grouped: player actions, connection, item lifecycle, projectile, mover actions.

### Modality Vocab (5 entries)

NONE(0), SIGHT(1), PROXIMITY(2), SOUND(3), MEMORY(4).

### Player ID

Entity_num (1..maxclients). Per-match identifier. Own embedding table.

## Token Types

### Projectile

Emits when in PVS (PROXIMITY modality). Lingers briefly after leaving PVS.

| Field | Type | Normalization |
|-------|------|---------------|
| subject_id | ID | entity vocab |
| modality_id | ID | modality vocab (always PROXIMITY) |
| rel[3] | scalar | /DIST_SCALE |
| vel[3] | scalar | /VELOCITY_SCALE |
| recency | scalar | seconds / TIME_SCALE |

7 scalars, 2 IDs. Projection: `proj_projectile(7 → d_model)`.

### Actor (Player)

Emits when in FOV (SIGHT) or heard (SOUND). Lingers after observation via recency.

| Field | Type | Normalization |
|-------|------|---------------|
| subject_id | ID | entity vocab (always PLAYER) |
| modality_id | ID | modality vocab |
| player_id | ID | player embed |
| half_extents[3] | scalar | /DIST_SCALE |
| rel[3] | scalar | /DIST_SCALE |
| vel[3] | scalar | /VELOCITY_SCALE |
| path[3] | scalar | /DIST_SCALE |
| eta | scalar | /TIME_SCALE |
| facing | scalar | 0=looking at us, 1=away |
| team | scalar | 0 or 1 (IsSameTeam) |
| score | scalar | frags / max_frags, unclamped, 0 if max is 0 |
| recency | scalar | seconds / TIME_SCALE |

17 scalars, 3 IDs. Projection: `proj_actor(17 → d_model)`.

### Item

Emits when in PVS (PROXIMITY) or heard (SOUND). Includes backpacks (amount=0).

| Field | Type | Normalization |
|-------|------|---------------|
| subject_id | ID | entity vocab |
| modality_id | ID | modality vocab |
| half_extents[3] | scalar | /DIST_SCALE |
| rel[3] | scalar | /DIST_SCALE |
| path[3] | scalar | /DIST_SCALE |
| eta | scalar | /TIME_SCALE |
| amount | scalar | per-type (see below) |
| regen | scalar | seconds / TIME_SCALE |
| recency | scalar | seconds / TIME_SCALE |

13 scalars, 2 IDs. Projection: `proj_item(13 → d_model)`.

#### Amount Normalization

| Type | Raw | Cap | Example |
|------|-----|-----|---------|
| Health | hp | 100 | 25hp → 0.25 |
| Megahealth | hp | 100 | 100hp → 1.0 |
| Armor (all tiers) | effective HP (amount × absorb) | 160 | Green 30 → 0.19, Yellow 90 → 0.56, Red 160 → 1.0 |
| Shells | count | 100 | 20 → 0.2 |
| Nails | count | 200 | 25 → 0.125 |
| Rockets | count | 100 | 5 → 0.05 |
| Cells | count | 100 | 6 → 0.06 |
| Weapons | ammo bonus / ammo cap | per type | SSG 5/100=0.05, NG 30/200=0.15 |
| Powerups | 1.0 | - | always 1.0 |
| Backpacks | 0.0 | - | unknown contents |

### Mover

Emits when in PVS (PROXIMITY) or heard (SOUND).

| Field | Type | Normalization |
|-------|------|---------------|
| subject_id | ID | entity vocab |
| modality_id | ID | modality vocab |
| half_extents[3] | scalar | /DIST_SCALE |
| rel[3] | scalar | /DIST_SCALE |
| path[3] | scalar | /DIST_SCALE |
| eta | scalar | /TIME_SCALE |
| state | scalar | 0 = base, 1 = activated |
| recency | scalar | seconds / TIME_SCALE |

12 scalars, 2 IDs. Projection: `proj_mover(12 → d_model)`.

## Events

Up to 4 events per token. Each event is 2 IDs:

| Field | Type | Table |
|-------|------|-------|
| action_id | ID | action vocab |
| source_id | ID | entity vocab (shared) |

Events are embedded: `action_embed(action) + entity_embed(source)`, added to the token representation. Binary presence — if the atom is alive, it's on the token; if not, it's gone.

Connect/disconnect are events:
- CONNECT/NONE — player joined
- DISCONNECT/NONE — player left (only for tracked actors)

Sight-derived events:
- ACTIVE/POWERUP — EF_DIMLIGHT on a player in PVS

All entity types can have events. Examples:
- Actor: FIRE/SHOTGUN, PAIN/NONE, DEATH/GIB, ACTIVE/POWERUP, DISCONNECT/NONE
- Item: RESPAWN/NONE
- Mover: MOVE/NONE, MOVE/SECRET, ACTIVATE/KEYED
- Projectile: BOUNCE/NONE

## Wire Format

Variable-length: each token is a type tag byte followed by the IDs and scalars for that type, then events.

```
[n_tokens: u8]
per token:
  [type_tag: u8]
  [id_0: i32] [id_1: i32] ...
  [scalar_0: f32] [scalar_1: f32] ...
  [event_count: u8]
  per event:
    [evt_action: i32] [evt_source: i32]
```

Self token, spatial tokens, and action history remain fixed-format.

## Model Pipeline

```
for each token in wire stream:
    type = read_type_tag()
    ids, scalars = read_fields(type)
    
    # Type-specific scalar projection
    repr = proj_{type}(scalars)
    
    # Shared ID embeddings
    repr += entity_embed(subject_id)
    repr += modality_embed(modality_id)
    if type == ACTOR:
        repr += player_embed(player_id)
    
    # Events (binary presence, no recency scaling)
    for each event:
        repr += action_embed(action_id) + entity_embed(source_id)
    
    # Normalize (TBD — L2, LayerNorm, or none)
    token_sequence.append(repr)

# Pad to 16 tokens
token_sequence = pad(token_sequence, 16, zero_vector)
```

## C Worker Changes

1. Oracle builds per-type token data instead of unified `qnn_entity_token_t`
2. Pack variable-length wire with type tags
3. Drop from tokens: confidence, magnitude, qualifier_id, weapon_subject_id, powerup_subject_id, distance, rel_yaw, rel_pitch, route_cluster_ids
4. Add: score on actors, backpacks as items with amount=0
5. Fix: armor normalization to /160 (max effective armor), score unclamped with div-zero guard
6. Keep: self token, spatial tokens, action history in fixed format
7. Half extents on actors, items, movers (not projectiles)

## Recency

Recency starts at 0 when observed, ticks up by dt each frame (in seconds, /TIME_SCALE on the token). Entity drops from the token stream when recency >= modality threshold.

### Modality Thresholds

| Modality | Threshold | Use case |
|----------|-----------|----------|
| SIGHT | 2.0s | Player you saw stays in memory |
| PROXIMITY | 0.1s | Item/mover in PVS, brief awareness |
| SOUND | 0.1s | Heard something, brief awareness |
| MEMORY | 1.0s | Recalled entity from model action |

### Observation Rules

1. **New observation**: recency resets to 0, modality = channel of new observation
2. **Same tick, multiple observations**: higher priority modality wins (SIGHT > PROXIMITY > SOUND > MEMORY)
3. **Threshold**: max(current threshold, new modality threshold). Once you've seen something (2s window), hearing it later doesn't shorten the window.
4. **Drop**: entity removed from token stream when recency >= threshold

### Time Scale

All time values use /TIME_SCALE (60): recency, regen, eta. Per tick at 20hz, values change by 0.05/60 ≈ 0.000833. The projection layer compensates for small magnitude — consistent scale across all time fields is more important than absolute value size.

### Token Scalar

`recency` on the token is raw seconds since last observation, divided by TIME_SCALE (60). The model sees how stale the information is. Lower = fresher. 0 = just observed this tick. The threshold determines token stream presence, not the scalar value — the scalar just communicates staleness to the model.

## Token Normalization

Raw sum of projection + embeddings + events. No explicit normalization in the tokenizer. The transformer's first LayerNorm (pre-attention in each block) handles magnitude normalization. Confirmed: current architecture applies `ln1 = LayerNorm(d_model)` before attention in every `TransformerBlock`.

## Open Questions

- **Event recency**: currently binary presence (alive or gone). Revisit if 0.1s window proves too coarse for distinguishing same-tick vs last-tick events.
- **Cluster embeddings**: needs liquid geometry in navmesh first (see cluster_embeddings.md)
- **Projectile impact events**: entity_num -1, need spatial matching (see world_state.md)
- **Demo clipping**: pre/post match dead time detection
